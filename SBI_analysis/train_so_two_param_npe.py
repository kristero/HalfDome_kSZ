#!/usr/bin/env python3
"""Train a P0/beta NPE while marginalizing the other Battaglia parameters."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sbi_for_cluster import (
    ForwardingCapture,
    build_prior_from_bounds,
    configure_torch_threads,
    make_npe,
    maybe_save_pickle,
    parse_validation_losses_from_training_output,
    sample_posterior,
    save_pickle,
    save_training_output_and_validation_losses,
    save_x_transform,
    to_jsonable,
    transform_x_for_sbi,
)


DEFAULT_DATASET = Path(
    "/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/"
    "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/"
    "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/lustre/work/kristero10/adrian_two_param_npe_baseline_deproj0"
)
DEFAULT_TARGETS = ("P0", "beta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train q(P0,beta|x). The other seven parameters remain varied in "
            "the simulations and are therefore marginalized under their prior."
        )
    )
    parser.add_argument(
        "--prepared-dataset",
        type=Path,
        default=Path(os.environ.get("PREPARED_DATASET", DEFAULT_DATASET)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TWO_PARAM_OUTPUT_DIR",
                DEFAULT_OUTPUT_ROOT / "asinh",
            )
        ),
    )
    parser.add_argument("--target-params", default="P0,beta")
    parser.add_argument("--holdout-last-n", type=int, default=1000)
    parser.add_argument(
        "--x-rescale-mode",
        choices=("asinh", "none"),
        default=os.environ.get("SBI_X_RESCALE_MODE", "asinh"),
    )
    parser.add_argument("--x-rescale-eps", type=float, default=1.0e-30)
    parser.add_argument("--x-standardize-eps", type=float, default=1.0e-8)
    parser.add_argument("--density-estimator", default="maf")
    parser.add_argument("--stop-after-epochs", type=int, default=60)
    parser.add_argument("--num-battaglia-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def first_array(data: Any, names: tuple[str, ...], label: str) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise KeyError(f"Prepared dataset is missing {label}; tried {names}")


def scalar_string(value: Any) -> str:
    array = np.asarray(value)
    return str(array.reshape(()).item()) if array.size == 1 else str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def load_inputs(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        theta = np.asarray(
            first_array(data, ("theta",), "theta"),
            dtype=np.float32,
        )
        x = np.asarray(
            first_array(data, ("x",), "x"),
            dtype=np.float32,
        )
        obs = np.asarray(
            first_array(
                data,
                ("obs", "x_obs", "observed"),
                "observation",
            ),
            dtype=np.float32,
        ).reshape(-1)
        prior_low = np.asarray(
            first_array(data, ("prior_low",), "prior_low"),
            dtype=np.float32,
        )
        prior_high = np.asarray(
            first_array(data, ("prior_high",), "prior_high"),
            dtype=np.float32,
        )
        param_names = [
            str(value)
            for value in first_array(
                data,
                ("param_names", "theta_columns"),
                "parameter names",
            )
        ]
        obs_source = (
            scalar_string(data["obs_source"])
            if "obs_source" in data.files
            else "prepared_obs"
        )
        sobol_global_row = (
            np.asarray(data["sobol_global_row"], dtype=np.int64)
            if "sobol_global_row" in data.files
            else None
        )

    if theta.ndim != 2 or x.ndim != 2:
        raise ValueError(
            f"Expected theta/x to be 2D; got theta={theta.shape}, x={x.shape}"
        )
    if theta.shape[0] != x.shape[0]:
        raise ValueError(
            f"theta/x row mismatch: {theta.shape[0]} != {x.shape[0]}"
        )
    if theta.shape[1] != len(param_names):
        raise ValueError(
            f"theta has {theta.shape[1]} columns but "
            f"{len(param_names)} names"
        )
    if obs.size != x.shape[1]:
        raise ValueError(
            f"obs has {obs.size} values but x has {x.shape[1]} columns"
        )
    if (
        prior_low.shape != (theta.shape[1],)
        or prior_high.shape != prior_low.shape
    ):
        raise ValueError(
            f"Prior bounds do not match theta: low={prior_low.shape}, "
            f"high={prior_high.shape}, theta={theta.shape}"
        )
    if (
        not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(obs))
    ):
        raise ValueError("theta, x, and obs must contain only finite values")
    if np.any(prior_high <= prior_low):
        raise ValueError(
            "Every prior upper bound must exceed its lower bound"
        )
    if (
        sobol_global_row is not None
        and sobol_global_row.shape != (theta.shape[0],)
    ):
        raise ValueError(
            "sobol_global_row does not match the dataset row count"
        )

    return {
        "theta": theta,
        "x": x,
        "obs": obs,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "param_names": param_names,
        "obs_source": obs_source,
        "sobol_global_row": sobol_global_row,
    }


def main() -> int:
    args = parse_args()
    configure_torch_threads()
    dataset_path = args.prepared_dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    completed_marker = output_dir / "density_estimator.pkl"
    if (
        completed_marker.exists()
        and not args.overwrite
        and not args.validate_only
    ):
        raise FileExistsError(
            f"A trained estimator already exists at {completed_marker}. "
            "Use --overwrite only when you intentionally want to replace it."
        )

    inputs = load_inputs(dataset_path)
    theta_full = inputs["theta"]
    x_full = inputs["x"]
    param_names = inputs["param_names"]
    target_names = [
        value.strip()
        for value in args.target_params.split(",")
        if value.strip()
    ]
    if target_names != list(DEFAULT_TARGETS):
        raise ValueError(
            f"This analysis requires target order {list(DEFAULT_TARGETS)}; "
            f"got {target_names}."
        )
    if len(set(target_names)) != len(target_names):
        raise ValueError(
            f"Target parameter names must be unique: {target_names}"
        )
    missing = [
        name for name in target_names if name not in param_names
    ]
    if missing:
        raise KeyError(
            f"Target parameters missing from prepared dataset: {missing}; "
            f"available={param_names}"
        )
    target_indices = np.asarray(
        [param_names.index(name) for name in target_names],
        dtype=np.int64,
    )
    nuisance_names = [
        name for name in param_names if name not in target_names
    ]

    n_rows = int(theta_full.shape[0])
    holdout_n = int(args.holdout_last_n)
    if holdout_n <= 0 or holdout_n >= n_rows:
        raise ValueError(
            f"--holdout-last-n must be in 1..{n_rows - 1}; "
            f"got {holdout_n}"
        )
    n_train = n_rows - holdout_n
    train_indices = np.arange(n_train, dtype=np.int64)
    heldout_indices = np.arange(n_train, n_rows, dtype=np.int64)
    if np.intersect1d(train_indices, heldout_indices).size:
        raise RuntimeError("Training and held-out indices overlap")
    if train_indices.size + heldout_indices.size != n_rows:
        raise RuntimeError(
            "Training and held-out indices do not partition the dataset"
        )

    nuisance_indices = [
        param_names.index(name) for name in nuisance_names
    ]
    nuisance_std = np.std(
        theta_full[:n_train, nuisance_indices],
        axis=0,
        ddof=1,
    )
    if (
        np.any(~np.isfinite(nuisance_std))
        or np.any(nuisance_std <= 0.0)
    ):
        raise ValueError(
            "At least one nuisance parameter does not vary in the "
            "training simulations: "
            + repr(dict(zip(nuisance_names, nuisance_std.tolist())))
        )

    selected_low = inputs["prior_low"][target_indices]
    selected_high = inputs["prior_high"][target_indices]
    split_report = {
        "prepared_dataset": str(dataset_path),
        "available_rows": n_rows,
        "n_train": n_train,
        "holdout_last_n": holdout_n,
        "train_index_first": int(train_indices[0]),
        "train_index_last": int(train_indices[-1]),
        "heldout_index_first": int(heldout_indices[0]),
        "heldout_index_last": int(heldout_indices[-1]),
        "split_overlap_count": 0,
        "x_dim": int(x_full.shape[1]),
        "source_theta_dim": int(theta_full.shape[1]),
        "target_theta_dim": len(target_names),
        "source_param_names": param_names,
        "target_param_names": target_names,
        "target_param_indices": target_indices,
        "nuisance_param_names": nuisance_names,
        "nuisance_training_std": nuisance_std,
        "target_prior_low": selected_low,
        "target_prior_high": selected_high,
        "obs_source": inputs["obs_source"],
        "x_rescale_mode": args.x_rescale_mode,
        "sobol_mapping_available": (
            inputs["sobol_global_row"] is not None
        ),
    }
    write_json(
        output_dir / "preflight_validation.json",
        split_report,
    )
    print(
        json.dumps(
            to_jsonable(split_report),
            indent=2,
            sort_keys=True,
        )
    )
    if args.validate_only:
        print("Validation-only check passed; no NPE was trained.")
        return 0

    np.save(output_dir / "train_indices.npy", train_indices)
    np.save(
        output_dir / "heldout_test_indices.npy",
        heldout_indices,
    )
    np.save(
        output_dir / "target_param_indices.npy",
        target_indices,
    )
    if inputs["sobol_global_row"] is not None:
        np.save(
            output_dir / "heldout_sobol_global_row.npy",
            inputs["sobol_global_row"][heldout_indices],
        )

    theta_train = np.ascontiguousarray(
        theta_full[:n_train, target_indices],
        dtype=np.float32,
    )
    x_train = np.ascontiguousarray(
        x_full[:n_train],
        dtype=np.float32,
    )
    obs = np.ascontiguousarray(
        inputs["obs"],
        dtype=np.float32,
    )
    prior = build_prior_from_bounds(
        {
            "prior_low": selected_low,
            "prior_high": selected_high,
        },
        args.device,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    theta_t = torch.as_tensor(
        theta_train,
        dtype=torch.float32,
        device=args.device,
    )
    x_t, obs_t, x_transform = transform_x_for_sbi(
        x_train,
        obs,
        args,
    )
    save_x_transform(
        output_dir / "x_transform.npz",
        x_transform,
        train_indices,
    )
    np.save(
        output_dir / "obs_transformed.npy",
        obs_t.detach().cpu().numpy().astype(np.float32),
    )

    print(f"Training target parameters: {target_names}")
    print(f"Marginalized nuisance parameters: {nuisance_names}")
    print(
        f"Training rows: {n_train}; held-out rows: {holdout_n}"
    )
    print(
        f"theta target shape: {tuple(theta_t.shape)}; "
        f"x shape: {tuple(x_t.shape)}"
    )
    print(
        f"x transform: {args.x_rescale_mode}; "
        f"estimator: {args.density_estimator}"
    )

    inference = make_npe(
        prior,
        args.density_estimator,
        args.device,
    )
    captured = ForwardingCapture(stream=os.sys.stdout)
    with contextlib.redirect_stdout(captured):
        density_estimator = (
            inference.append_simulations(theta_t, x_t).train(
                stop_after_epochs=int(args.stop_after_epochs),
                show_train_summary=True,
            )
        )
    training_output = captured.getvalue()
    validation_losses = (
        parse_validation_losses_from_training_output(
            training_output
        )
    )
    training_summary_path = (
        save_training_output_and_validation_losses(
            output_dir,
            training_output,
            validation_losses,
        )
    )

    posterior = inference.build_posterior(density_estimator)
    save_pickle(
        output_dir / "density_estimator.pkl",
        density_estimator,
        "density estimator",
    )
    save_pickle(
        output_dir / "posterior.pkl",
        posterior,
        "posterior",
    )
    maybe_save_pickle(
        output_dir / "inference.pkl",
        inference,
        "inference",
    )
    maybe_save_pickle(
        output_dir / "prior.pkl",
        prior,
        "two-parameter prior",
    )
    if hasattr(density_estimator, "state_dict"):
        torch.save(
            density_estimator.state_dict(),
            output_dir / "density_estimator_state_dict.pt",
        )

    battaglia_samples = sample_posterior(
        posterior,
        obs_t,
        int(args.num_battaglia_samples),
    )
    battaglia_samples = np.asarray(
        battaglia_samples,
        dtype=np.float32,
    ).reshape(-1, len(target_names))
    np.save(
        output_dir / "battaglia12_posterior_samples.npy",
        battaglia_samples,
    )

    metadata = {
        **split_report,
        "seed": int(args.seed),
        "device": args.device,
        "density_estimator": args.density_estimator,
        "stop_after_epochs": int(args.stop_after_epochs),
        "num_battaglia_samples": int(
            battaglia_samples.shape[0]
        ),
        "best_validation_performance": (
            validation_losses[-1]
            if validation_losses
            else None
        ),
        "validation_performances": validation_losses,
        "training_summary_stdout_path": str(
            training_summary_path
        ),
        "x_transform_path": str(
            output_dir / "x_transform.npz"
        ),
        "marginalization": (
            "Direct two-output NPE trained on P0,beta while all "
            "seven nuisance parameters vary according to the "
            "simulation design."
        ),
    }
    write_json(
        output_dir / "run_metadata.json",
        metadata,
    )
    print(f"Completed two-parameter NPE: {output_dir}")

    del theta_t, x_t, inference, density_estimator, posterior
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
