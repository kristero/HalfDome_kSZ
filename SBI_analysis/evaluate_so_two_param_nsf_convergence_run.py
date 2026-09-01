#!/usr/bin/env python3
"""Checkpointed held-out evaluation for one two-parameter NPE run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_so_two_param_npe import (
    PARAMS,
    load_data,
    metric_rows,
    summaries,
    write_csv,
    write_json,
)
from run_so_sbc import (
    apply_x_transform,
    load_pickle,
    load_posterior as load_or_build_posterior,
    load_x_transform,
    sample_posterior_at_x,
)


DEFAULT_DATASET = Path(
    "/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/"
    "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/"
    "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz"
)
DEFAULT_BATTAGLIA_OBSERVATION = Path(
    "/lustre/work/kristero10/adrian_fisher_baseline_deproj0/"
    "battaglia12_baseline_deproj0_observation/prepared/"
    "battaglia12_masked_baseline_noise_cross_deproj0_binned_dell.npy"
)
DEFAULT_BATTAGLIA_REPORT = Path(
    "/lustre/work/kristero10/adrian_fisher_baseline_deproj0/"
    "battaglia12_baseline_deproj0_observation/prepared/"
    "battaglia12_baseline_deproj0_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("asinh",), required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument(
        "--expected-density-estimator",
        choices=("maf", "nsf"),
        default="nsf",
    )
    parser.add_argument("--expected-hidden-features", type=int, default=64)
    parser.add_argument("--expected-num-transforms", type=int, default=6)
    parser.add_argument("--holdout-last-n", type=int, default=1000)
    parser.add_argument("--num-posterior-samples", type=int, default=2000)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--battaglia-samples", type=int, default=0)
    parser.add_argument("--battaglia-batch-size", type=int, default=2000)
    parser.add_argument(
        "--battaglia-observation",
        type=Path,
        default=DEFAULT_BATTAGLIA_OBSERVATION,
    )
    parser.add_argument(
        "--battaglia-validation-report",
        type=Path,
        default=DEFAULT_BATTAGLIA_REPORT,
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--restart-battaglia",
        action="store_true",
        help="Replace only Battaglia12 samples; reuse held-out profile samples.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def scalar_string(value: Any, default: str = "") -> str:
    array = np.asarray(value)
    return str(array.reshape(()).item()) if array.size == 1 else default


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def load_observation(
    prepared_dataset: Path,
    observation_path: Path,
    report_path: Path,
    expected_x_dim: int,
) -> tuple[np.ndarray, str]:
    observation_path = observation_path.expanduser().resolve()
    report_path = report_path.expanduser().resolve()
    if not observation_path.is_file():
        raise FileNotFoundError(f"Validated Battaglia12 observation is missing: {observation_path}")
    if not report_path.is_file():
        raise FileNotFoundError(f"Battaglia12 validation report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("all_checks_passed", False):
        raise ValueError(f"Battaglia12 validation did not pass: {report_path}")
    if report.get("required_product") != "masked_baseline_noise_cross_deproj0":
        raise ValueError(f"Wrong Battaglia12 product in {report_path}")
    report_dataset = Path(report["prepared_dataset"]).expanduser().resolve()
    if report_dataset != prepared_dataset.expanduser().resolve():
        raise ValueError(
            "Battaglia12 observation was validated against another dataset: "
            f"{report_dataset} != {prepared_dataset}"
        )
    report_observation = Path(
        report["outputs"]["binned_dell"]
    ).expanduser().resolve()
    if report_observation != observation_path:
        raise ValueError(
            f"Observation/report mismatch: {observation_path} != {report_observation}"
        )
    observation = np.asarray(np.load(observation_path), dtype=np.float32).reshape(-1)
    if observation.size != expected_x_dim:
        raise ValueError(
            f"Battaglia12 observation has {observation.size} features; expected {expected_x_dim}."
        )
    if not np.all(np.isfinite(observation)):
        raise ValueError(f"Battaglia12 observation contains non-finite values: {observation_path}")
    return observation, f"validated_baseline_deproj0:{observation_path}"


def prepare_battaglia_contract(
    *,
    run_dir: Path,
    observation: np.ndarray,
    transformed_observation: np.ndarray,
    observation_source: str,
    validation_report: Path,
    restart: bool,
) -> Path:
    final_path = run_dir / "battaglia12_posterior_samples.npy"
    partial_path = run_dir / "evaluation" / "battaglia12_posterior_samples_partial.npy"
    contract_path = run_dir / "battaglia12_conditioning_contract.npz"
    if restart:
        for path in (final_path, partial_path, contract_path):
            path.unlink(missing_ok=True)
    existing_samples = final_path.is_file() or partial_path.is_file()
    if existing_samples:
        if not contract_path.is_file():
            raise FileNotFoundError(
                "Existing Battaglia12 samples have no conditioning contract. "
                "Rerun with --restart-battaglia."
            )
        with np.load(contract_path, allow_pickle=True) as contract:
            saved_observation = np.asarray(contract["observation"], dtype=np.float32)
            saved_transformed = np.asarray(
                contract["transformed_observation"], dtype=np.float32
            )
            saved_source = scalar_string(contract["observation_source"])
        if not np.array_equal(saved_observation, observation):
            raise ValueError("Existing Battaglia12 samples used a different observation.")
        if not np.array_equal(saved_transformed, transformed_observation):
            raise ValueError("Existing Battaglia12 samples used a different transform.")
        if saved_source != observation_source:
            raise ValueError("Existing Battaglia12 observation provenance differs.")
    else:
        np.savez_compressed(
            contract_path,
            observation=np.asarray(observation, dtype=np.float32),
            transformed_observation=np.asarray(
                transformed_observation, dtype=np.float32
            ),
            observation_source=np.asarray(observation_source),
            validation_report=np.asarray(str(validation_report.expanduser().resolve())),
            x_transform=np.asarray(str((run_dir / "x_transform.npz").resolve())),
        )
    return contract_path


def load_saved_posterior(run_dir: Path) -> Any:
    posterior_path = run_dir / "posterior.pkl"
    if posterior_path.is_file():
        return load_pickle(posterior_path)
    return load_or_build_posterior(run_dir)


def validate_run(
    *,
    run_dir: Path,
    data: dict[str, Any],
    mode: str,
    n_train: int,
    holdout_last_n: int,
    seed: int,
    expected_density_estimator: str,
    expected_hidden_features: int,
    expected_num_transforms: int,
) -> dict[str, Any]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    n_rows = int(data["theta"].shape[0])
    training_pool_n = n_rows - int(holdout_last_n)
    if not 0 < n_train <= training_pool_n:
        raise ValueError(f"Invalid N_train={n_train} for training pool {training_pool_n}.")

    train_indices = np.asarray(np.load(run_dir / "train_indices.npy"), dtype=np.int64)
    heldout_indices = np.asarray(
        np.load(run_dir / "heldout_test_indices.npy"), dtype=np.int64
    )
    target_indices = np.asarray(
        np.load(run_dir / "target_param_indices.npy"), dtype=np.int64
    )
    expected_train = np.random.default_rng(seed).permutation(training_pool_n)[:n_train]
    expected_heldout = np.arange(training_pool_n, n_rows, dtype=np.int64)
    transform = load_x_transform(run_dir)
    transform_indices = np.asarray(transform.get("train_indices", []), dtype=np.int64)

    checks = {
        "density_estimator": (run_dir / "density_estimator.pkl").is_file(),
        "posterior": (run_dir / "posterior.pkl").is_file(),
        "n_train": int(metadata.get("n_train", -1)) == int(n_train),
        "mode": metadata.get("x_rescale_mode") == mode,
        "estimator": (
            str(metadata.get("density_estimator", "")).lower()
            == str(expected_density_estimator).lower()
        ),
        "internal_z_score_x": metadata.get("internal_z_score_x") == "none",
        "hidden_features": (
            int(metadata.get("hidden_features", -1))
            == int(expected_hidden_features)
        ),
        "num_transforms": (
            int(metadata.get("num_transforms", -1))
            == int(expected_num_transforms)
        ),
        "seed": int(metadata.get("seed", -1)) == int(seed),
        "target_params": metadata.get("target_param_names") == list(PARAMS),
        "target_indices": np.array_equal(target_indices, data["indices"]),
        "nested_train_indices": np.array_equal(train_indices, expected_train),
        "transform_train_indices": np.array_equal(transform_indices, expected_train),
        "heldout_indices": np.array_equal(heldout_indices, expected_heldout),
        "overlap_count": int(np.intersect1d(train_indices, heldout_indices).size),
    }
    failed = [
        key
        for key, value in checks.items()
        if (key == "overlap_count" and value != 0)
        or (key != "overlap_count" and not value)
    ]
    if failed:
        raise ValueError(f"Run validation failed for {run_dir}: {failed}; {checks}")
    return {
        "checks": checks,
        "metadata": metadata,
        "training_pool_rows": training_pool_n,
        "heldout_indices": heldout_indices,
        "train_indices": train_indices,
        "transform": transform,
    }


def checkpointed_heldout_samples(
    *,
    posterior: Any,
    transform: dict[str, Any],
    x_eval: np.ndarray,
    output_path: Path,
    done_path: Path,
    count: int,
    seed: int,
    device: str,
    checkpoint_every: int,
) -> np.ndarray:
    shape = (x_eval.shape[0], count, len(PARAMS))
    if output_path.is_file():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != shape or output.dtype != np.float32:
            raise ValueError(f"Cannot resume {output_path}: {output.shape}, {output.dtype} != {shape}, float32")
    else:
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=shape
        )
        output[:] = np.nan
        output.flush()

    if done_path.is_file():
        done = np.asarray(np.load(done_path), dtype=bool)
        if done.shape != (x_eval.shape[0],):
            raise ValueError(f"Invalid checkpoint shape in {done_path}: {done.shape}")
    else:
        # Without the separately written completion mask, an interrupted
        # memmap initialization can leave zero-filled rows that look valid.
        output[:] = np.nan
        output.flush()
        done = np.zeros(x_eval.shape[0], dtype=bool)
        atomic_save(done_path, done)

    pending_since_flush = 0
    for local_index in np.flatnonzero(~done):
        np.random.seed(seed + int(local_index))
        torch.manual_seed(seed + int(local_index))
        samples = sample_posterior_at_x(
            posterior,
            apply_x_transform(x_eval[local_index], transform),
            count,
            device,
        )
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (count, len(PARAMS)) or not np.all(np.isfinite(samples)):
            raise ValueError(
                f"Invalid posterior samples at held-out index {local_index}: {samples.shape}"
            )
        output[local_index] = samples
        done[local_index] = True
        pending_since_flush += 1
        if pending_since_flush >= checkpoint_every:
            output.flush()
            atomic_save(done_path, done)
            pending_since_flush = 0
            print(f"Held-out checkpoint: {int(done.sum())}/{done.size}", flush=True)

    output.flush()
    atomic_save(done_path, done)
    if not np.all(done):
        raise RuntimeError("Held-out sampling ended with incomplete rows.")
    return np.load(output_path, mmap_mode="r")


def checkpointed_battaglia_samples(
    *,
    posterior: Any,
    transformed_observation: np.ndarray,
    run_dir: Path,
    count: int,
    batch_size: int,
    seed: int,
    device: str,
) -> np.ndarray | None:
    if count <= 0:
        return None
    final_path = run_dir / "battaglia12_posterior_samples.npy"
    partial_path = run_dir / "evaluation" / "battaglia12_posterior_samples_partial.npy"
    if final_path.is_file():
        samples = np.asarray(np.load(final_path), dtype=np.float32)
        if samples.ndim == 2 and samples.shape[1] == len(PARAMS) and samples.shape[0] >= count:
            return samples[:count]
        raise ValueError(f"Invalid existing Battaglia12 samples: {final_path}, {samples.shape}")
    if partial_path.is_file():
        samples = np.asarray(np.load(partial_path), dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != len(PARAMS):
            raise ValueError(f"Invalid partial Battaglia12 samples: {samples.shape}")
    else:
        samples = np.empty((0, len(PARAMS)), dtype=np.float32)

    while samples.shape[0] < count:
        requested = min(batch_size, count - samples.shape[0])
        np.random.seed(seed + 100_000 + samples.shape[0])
        torch.manual_seed(seed + 100_000 + samples.shape[0])
        new = np.asarray(
            sample_posterior_at_x(
                posterior, transformed_observation, requested, device
            ),
            dtype=np.float32,
        )
        if new.shape != (requested, len(PARAMS)) or not np.all(np.isfinite(new)):
            raise ValueError(f"Invalid Battaglia12 posterior batch: {new.shape}")
        samples = np.concatenate((samples, new), axis=0)
        atomic_save(partial_path, samples)
        print(f"Battaglia12 checkpoint: {samples.shape[0]}/{count}", flush=True)
    atomic_save(final_path, samples[:count])
    partial_path.unlink(missing_ok=True)
    return samples[:count]


def main() -> int:
    args = parse_args()
    if args.num_posterior_samples <= 1 or args.checkpoint_every <= 0:
        raise ValueError("Posterior sample count and checkpoint interval must be positive.")
    if args.battaglia_samples < 0 or args.battaglia_batch_size <= 0:
        raise ValueError("Invalid Battaglia12 sample configuration.")
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))

    dataset_path = args.prepared_dataset.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    if args.restart:
        for path in (
            evaluation_dir / "heldout_posterior_samples.npy",
            evaluation_dir / "heldout_sampling_done.npy",
            evaluation_dir / "heldout_metrics.csv",
            evaluation_dir / "heldout_summary.csv",
            evaluation_dir / "heldout_summary.json",
            evaluation_dir / "evaluation_complete.json",
            evaluation_dir / "battaglia12_posterior_samples_partial.npy",
            run_dir / "battaglia12_posterior_samples.npy",
            run_dir / "battaglia12_conditioning_contract.npz",
        ):
            path.unlink(missing_ok=True)
    data = load_data(dataset_path)
    validation = validate_run(
        run_dir=run_dir,
        data=data,
        mode=args.mode,
        n_train=args.n_train,
        holdout_last_n=args.holdout_last_n,
        seed=args.seed,
        expected_density_estimator=args.expected_density_estimator,
        expected_hidden_features=args.expected_hidden_features,
        expected_num_transforms=args.expected_num_transforms,
    )
    observation, observation_source = load_observation(
        dataset_path,
        args.battaglia_observation,
        args.battaglia_validation_report,
        data["x"].shape[1],
    )
    preflight = {
        "prepared_dataset": dataset_path,
        "run_dir": run_dir,
        "mode": args.mode,
        "n_train": args.n_train,
        "holdout_last_n": args.holdout_last_n,
        "observation_source": observation_source,
        "validation": validation["checks"],
    }
    write_json(evaluation_dir / "evaluation_preflight.json", preflight)
    if args.validate_only:
        print(json.dumps(preflight, default=str, indent=2, sort_keys=True))
        return 0

    heldout = validation["heldout_indices"]
    x_eval = np.ascontiguousarray(data["x"][heldout], dtype=np.float32)
    truth = np.ascontiguousarray(
        data["theta"][np.ix_(heldout, data["indices"])], dtype=np.float64
    )
    posterior = load_saved_posterior(run_dir)
    samples = checkpointed_heldout_samples(
        posterior=posterior,
        transform=validation["transform"],
        x_eval=x_eval,
        output_path=evaluation_dir / "heldout_posterior_samples.npy",
        done_path=evaluation_dir / "heldout_sampling_done.npy",
        count=args.num_posterior_samples,
        seed=args.seed,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
    )

    rows = metric_rows(
        args.mode,
        samples,
        truth,
        heldout,
        data["low"],
        data["high"],
    )
    for row in rows:
        row["n_train"] = int(args.n_train)
    frame = pd.DataFrame(rows)
    frame.to_csv(evaluation_dir / "heldout_metrics.csv", index=False)
    summary_rows = summaries(frame, args.rank_bins)
    for row in summary_rows:
        row["n_train"] = int(args.n_train)
    write_csv(evaluation_dir / "heldout_summary.csv", summary_rows)
    write_json(evaluation_dir / "heldout_summary.json", summary_rows)

    transformed_observation = apply_x_transform(
        observation, validation["transform"]
    )
    battaglia_contract = None
    if args.battaglia_samples > 0:
        battaglia_contract = prepare_battaglia_contract(
            run_dir=run_dir,
            observation=observation,
            transformed_observation=transformed_observation,
            observation_source=observation_source,
            validation_report=args.battaglia_validation_report,
            restart=args.restart_battaglia,
        )
    battaglia = checkpointed_battaglia_samples(
        posterior=posterior,
        transformed_observation=transformed_observation,
        run_dir=run_dir,
        count=args.battaglia_samples,
        batch_size=args.battaglia_batch_size,
        seed=args.seed,
        device=args.device,
    )
    completion = {
        **preflight,
        "num_test_profiles": int(len(heldout)),
        "num_posterior_samples_per_test": int(args.num_posterior_samples),
        "battaglia12_samples": 0 if battaglia is None else int(battaglia.shape[0]),
        "battaglia12_observation_source": observation_source,
        "battaglia12_conditioning_contract": (
            "" if battaglia_contract is None else str(battaglia_contract)
        ),
        "heldout_metrics": str(evaluation_dir / "heldout_metrics.csv"),
        "heldout_summary": str(evaluation_dir / "heldout_summary.csv"),
    }
    write_json(evaluation_dir / "evaluation_complete.json", completion)
    print(f"Completed convergence evaluation: {evaluation_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
