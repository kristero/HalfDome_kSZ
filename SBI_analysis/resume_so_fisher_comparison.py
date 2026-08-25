#!/usr/bin/env python3
"""Resume the NPE sampling and plotting stage of the SO Fisher comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from plot_battaglia12_so_getdist import (
    apply_x_transform,
    available_n_values,
    find_run_dir,
    load_posterior,
    load_x_transform,
    plot_getdist,
    sample_posterior_at_x,
)
from run_so_fisher_analysis import (
    BATTAGLIA12,
    DEFAULT_CASE,
    DEFAULT_FISHER_ROOT,
    DEFAULT_PREPARED_DATASET,
    DEFAULT_SBI_RUN_ROOT,
    write_constraint_table,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse completed Fisher samples and run only the missing saved-NPE "
            "sampling, GetDist plots, and constraint table."
        )
    )
    parser.add_argument("--prepared-dataset", type=Path, default=DEFAULT_PREPARED_DATASET)
    parser.add_argument("--fisher-root", type=Path, default=DEFAULT_FISHER_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sbi-run-root", type=Path, default=DEFAULT_SBI_RUN_ROOT)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument(
        "--npe-n-train",
        type=int,
        default=0,
        help="NPE training size; 0 selects the largest completed run.",
    )
    parser.add_argument("--sbi-samples", type=int, default=20_000)
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=2_000,
        help="Checkpoint the NPE chain after this many newly drawn samples.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=20_000,
        help="Maximum samples per chain passed to GetDist; 0 uses every sample.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("all", "combined", "none"),
        default="all",
        help="Generate all three legacy corners, only the three-chain corner, or no plots.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Ignore any completed or partial saved NPE chain.",
    )
    parser.add_argument(
        "--overwrite-plots",
        action="store_true",
        help="Regenerate corner plots that already exist.",
    )
    return parser.parse_args()


def load_sample_array(path: Path, expected_parameters: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed output is missing: {path}")
    samples = np.asarray(np.load(path), dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < expected_parameters:
        raise ValueError(
            f"Expected {path} to have shape (samples, >= {expected_parameters}); "
            f"found {samples.shape}."
        )
    samples = samples[:, :expected_parameters]
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if samples.shape[0] < 1000:
        raise ValueError(f"Only {samples.shape[0]} finite samples remain in {path}.")
    return samples


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def checkpointed_npe_samples(
    *,
    posterior: Any,
    transformed_observation: np.ndarray,
    final_path: Path,
    partial_path: Path,
    target: int,
    batch_size: int,
    device: str,
    n_parameters: int,
    force: bool,
) -> np.ndarray:
    if final_path.is_file() and not force:
        samples = load_sample_array(final_path, n_parameters)
        print(
            f"Reusing completed NPE chain: {final_path} ({samples.shape[0]} samples)",
            flush=True,
        )
        return samples

    if partial_path.is_file() and not force:
        samples = load_sample_array(partial_path, n_parameters)
        print(
            f"Resuming partial NPE chain: {samples.shape[0]}/{target}",
            flush=True,
        )
    else:
        samples = np.empty((0, n_parameters), dtype=np.float64)

    while samples.shape[0] < target:
        requested = min(batch_size, target - samples.shape[0])
        new_samples = sample_posterior_at_x(
            posterior,
            transformed_observation,
            requested,
            device,
        )
        if new_samples.ndim != 2 or new_samples.shape[1] < n_parameters:
            raise ValueError(
                f"NPE returned shape {new_samples.shape}; expected "
                f"(samples, >= {n_parameters})."
            )
        new_samples = np.asarray(new_samples[:, :n_parameters], dtype=np.float64)
        new_samples = new_samples[np.all(np.isfinite(new_samples), axis=1)]
        if new_samples.size == 0:
            raise ValueError("NPE batch contained no finite posterior samples.")
        samples = np.concatenate((samples, new_samples), axis=0)
        atomic_save(partial_path, samples)
        print(f"NPE checkpoint: {samples.shape[0]}/{target}", flush=True)

    samples = samples[:target]
    atomic_save(final_path, samples)
    partial_path.unlink(missing_ok=True)
    print(f"Saved completed NPE chain: {final_path}", flush=True)
    return samples


def plot_subset(samples: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or samples.shape[0] <= maximum:
        return samples
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(samples.shape[0], size=maximum, replace=False))
    return samples[indices]


def maybe_plot(
    sample_sets: list[dict[str, Any]],
    param_names: list[str],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    if output_path.is_file() and not args.overwrite_plots:
        print(f"Reusing completed plot: {output_path}", flush=True)
        return
    print(f"Generating GetDist plot: {output_path.name}", flush=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    plot_getdist(
        sample_sets,
        param_names,
        BATTAGLIA12,
        temporary,
        filled_last_only=True,
        dpi=args.dpi,
    )
    temporary.replace(output_path)
    print(f"Saved completed plot: {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    if args.sbi_samples < 1000:
        raise ValueError("--sbi-samples must be at least 1000.")
    if args.sample_batch_size <= 0:
        raise ValueError("--sample-batch-size must be positive.")

    prepared_dataset = args.prepared_dataset.expanduser().resolve()
    fisher_root = args.fisher_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fisher_root / "analysis"
    )
    sbi_run_root = args.sbi_run_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(prepared_dataset, allow_pickle=True) as data:
        param_names = [str(value) for value in data["param_names"]]
        prior_low = np.asarray(data["prior_low"], dtype=np.float64)
        prior_high = np.asarray(data["prior_high"], dtype=np.float64)
    n_parameters = len(param_names)
    if BATTAGLIA12.size != n_parameters:
        raise ValueError(
            f"Battaglia12 has {BATTAGLIA12.size} parameters but the dataset has "
            f"{n_parameters}: {param_names}."
        )

    fisher_untruncated = load_sample_array(
        output_dir / "fisher_samples_untruncated.npy",
        n_parameters,
    )
    fisher_truncated = load_sample_array(
        output_dir / "fisher_samples_prior_truncated.npy",
        n_parameters,
    )
    fiducial_dell = np.asarray(
        np.load(output_dir / "fiducial_battaglia12_dell.npy"),
        dtype=np.float32,
    ).reshape(-1)
    print(
        "Loaded completed Fisher chains: "
        f"untruncated={fisher_untruncated.shape}, "
        f"truncated={fisher_truncated.shape}",
        flush=True,
    )

    available = available_n_values(sbi_run_root, args.case)
    if not available:
        raise FileNotFoundError(
            f"No completed NPE runs found for case={args.case} under {sbi_run_root}."
        )
    n_train = args.npe_n_train if args.npe_n_train > 0 else available[-1]
    if n_train not in available:
        raise ValueError(
            f"NPE N={n_train} is unavailable for {args.case}; available={available}."
        )
    run_dir = find_run_dir(sbi_run_root, args.case, n_train)
    transform = load_x_transform(run_dir)
    transformed_observation = apply_x_transform(fiducial_dell, transform)
    if transformed_observation.size != fiducial_dell.size:
        raise ValueError("Saved NPE x transform changed the observation dimension.")
    np.save(
        output_dir / "battaglia12_observation_transformed.npy",
        transformed_observation,
    )

    final_path = output_dir / f"sbi_samples_N{n_train}.npy"
    partial_path = output_dir / f"sbi_samples_N{n_train}_partial.npy"
    if final_path.is_file() and not args.force_resample:
        sbi_samples = load_sample_array(final_path, n_parameters)
        print(
            f"Reusing completed NPE chain: {final_path} ({sbi_samples.shape[0]} samples)",
            flush=True,
        )
    else:
        print(f"Loading saved NPE from {run_dir}", flush=True)
        posterior = load_posterior(run_dir)
        sbi_samples = checkpointed_npe_samples(
            posterior=posterior,
            transformed_observation=transformed_observation,
            final_path=final_path,
            partial_path=partial_path,
            target=args.sbi_samples,
            batch_size=args.sample_batch_size,
            device=args.device,
            n_parameters=n_parameters,
            force=args.force_resample,
        )

    full_sample_sets = [
        {"label": "Fisher Gaussian", "samples": fisher_untruncated},
        {"label": "Fisher + prior", "samples": fisher_truncated},
        {"label": f"NPE N={n_train:,}", "samples": sbi_samples},
    ]
    plot_sample_sets = [
        {
            "label": item["label"],
            "samples": plot_subset(
                np.asarray(item["samples"]),
                args.plot_samples,
                args.seed + index,
            ),
        }
        for index, item in enumerate(full_sample_sets)
    ]

    write_constraint_table(
        output_dir / "fisher_vs_sbi_constraints.csv",
        param_names,
        BATTAGLIA12,
        prior_high - prior_low,
        full_sample_sets,
    )

    if args.plot_mode in {"all", "combined"}:
        maybe_plot(
            plot_sample_sets,
            param_names,
            output_dir / "fisher_untruncated_truncated_vs_sbi_corner.jpg",
            args,
        )
    if args.plot_mode == "all":
        maybe_plot(
            [plot_sample_sets[0], plot_sample_sets[2]],
            param_names,
            output_dir / "fisher_untruncated_vs_sbi_corner.jpg",
            args,
        )
        maybe_plot(
            [plot_sample_sets[1], plot_sample_sets[2]],
            param_names,
            output_dir / "fisher_prior_truncated_vs_sbi_corner.jpg",
            args,
        )

    write_json(
        output_dir / "fisher_comparison_resume_summary.json",
        {
            "prepared_dataset": str(prepared_dataset),
            "output_dir": str(output_dir),
            "sbi_run_root": str(sbi_run_root),
            "case": args.case,
            "npe_run_dir": str(run_dir),
            "npe_n_train": n_train,
            "x_transform_mode": transform.get("mode", "none"),
            "n_fisher_untruncated": fisher_untruncated.shape[0],
            "n_fisher_prior_truncated": fisher_truncated.shape[0],
            "n_sbi_samples": sbi_samples.shape[0],
            "n_plot_samples_max_per_chain": args.plot_samples,
            "plot_mode": args.plot_mode,
        },
    )
    print(f"Fisher/SBI continuation complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
