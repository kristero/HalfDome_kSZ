#!/usr/bin/env python3
"""Resume the NPE sampling and plotting stage of the SO Fisher comparison."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from plot_battaglia12_so_getdist import (
    apply_x_transform,
    available_n_values,
    find_run_dir,
    load_x_transform,
    plot_getdist,
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
    parser.add_argument(
        "--fisher-analysis-dir",
        type=Path,
        default=None,
        help="Directory containing the completed Fisher sample arrays.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--battaglia-observation",
        type=Path,
        default=None,
        help="Validated, binned baseline-deproj0 Battaglia12 D_ell vector.",
    )
    parser.add_argument(
        "--observation-validation-report",
        type=Path,
        default=None,
        help="Validation JSON written with the matched Battaglia12 observation.",
    )
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
        "--sampling-method",
        choices=("mcmc", "direct"),
        default="mcmc",
        help=(
            "NPE posterior sampler. MCMC avoids the DirectPosterior rejection "
            "stall caused by neural-density leakage outside a bounded prior."
        ),
    )
    parser.add_argument(
        "--mcmc-method",
        choices=("slice_np", "slice_np_vectorized"),
        default="slice_np_vectorized",
    )
    parser.add_argument("--mcmc-warmup-steps", type=int, default=200)
    parser.add_argument("--mcmc-thin", type=int, default=2)
    parser.add_argument("--mcmc-num-chains", type=int, default=8)
    parser.add_argument("--mcmc-num-workers", type=int, default=1)
    parser.add_argument(
        "--leakage-diagnostic-samples",
        type=int,
        default=10_000,
        help="Raw density-estimator draws used to measure prior-support leakage.",
    )
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


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def build_sampling_posterior(
    run_dir: Path,
    sampling_method: str,
    mcmc_method: str,
) -> tuple[Any, Any]:
    inference_path = run_dir / "inference.pkl"
    density_path = run_dir / "density_estimator.pkl"
    if not inference_path.is_file() or not density_path.is_file():
        raise FileNotFoundError(
            "MCMC rebuilding and leakage diagnostics require both "
            f"{inference_path} and {density_path}."
        )

    inference = load_pickle(inference_path)
    density_estimator = load_pickle(density_path)
    if sampling_method == "mcmc":
        posterior = inference.build_posterior(
            density_estimator,
            sample_with="mcmc",
            mcmc_method=mcmc_method,
        )
    else:
        posterior = inference.build_posterior(density_estimator)
    return posterior, density_estimator


def raw_density_leakage_diagnostic(
    density_estimator: Any,
    transformed_observation: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    count: int,
    device: str,
) -> dict[str, Any]:
    if count <= 0:
        return {"status": "disabled", "n_raw_samples": 0}

    import torch

    context = torch.as_tensor(
        np.asarray(transformed_observation, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).reshape(1, -1)
    with torch.no_grad():
        try:
            raw = density_estimator.sample(int(count), context=context)
        except TypeError:
            raw = density_estimator.sample((int(count),), context=context)
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().numpy()
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, prior_low.size)
    finite = np.all(np.isfinite(raw), axis=1)
    within = finite & np.all(
        (raw >= prior_low.reshape(1, -1))
        & (raw <= prior_high.reshape(1, -1)),
        axis=1,
    )
    return {
        "status": "ok",
        "n_raw_samples": int(raw.shape[0]),
        "n_finite": int(np.count_nonzero(finite)),
        "n_within_prior": int(np.count_nonzero(within)),
        "fraction_within_prior": float(np.mean(within)),
        "raw_min": np.nanmin(raw, axis=0).tolist(),
        "raw_max": np.nanmax(raw, axis=0).tolist(),
    }


def sample_posterior_configured(
    posterior: Any,
    x_obs: np.ndarray,
    num_samples: int,
    device: str,
    sample_kwargs: dict[str, Any],
) -> np.ndarray:
    import torch

    x_t = torch.as_tensor(
        np.asarray(x_obs, dtype=np.float32), dtype=torch.float32, device=device
    )
    posterior_x = posterior
    if hasattr(posterior, "set_default_x"):
        maybe_posterior = posterior.set_default_x(x_t)
        if maybe_posterior is not None:
            posterior_x = maybe_posterior

    samples = posterior_x.sample(
        (int(num_samples),),
        x=x_t,
        show_progress_bars=True,
        **sample_kwargs,
    )
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    elif samples.ndim > 2:
        samples = samples.reshape(-1, samples.shape[-1])
    return samples


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
    sample_kwargs: dict[str, Any],
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
        new_samples = sample_posterior_configured(
            posterior,
            transformed_observation,
            requested,
            device,
            sample_kwargs,
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
    if (
        output_path.is_file()
        and not args.overwrite_plots
        and not args.force_resample
    ):
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
    if min(
        args.mcmc_warmup_steps,
        args.mcmc_thin,
        args.mcmc_num_chains,
        args.mcmc_num_workers,
    ) <= 0:
        raise ValueError("All MCMC controls must be positive.")

    prepared_dataset = args.prepared_dataset.expanduser().resolve()
    fisher_root = args.fisher_root.expanduser().resolve()
    fisher_analysis_dir = (
        args.fisher_analysis_dir.expanduser().resolve()
        if args.fisher_analysis_dir is not None
        else fisher_root / "analysis"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fisher_analysis_dir / "matched_baseline_deproj0_comparison"
    )
    battaglia_observation_path = (
        args.battaglia_observation.expanduser().resolve()
        if args.battaglia_observation is not None
        else fisher_root
        / "battaglia12_baseline_deproj0_observation"
        / "prepared"
        / "battaglia12_masked_baseline_noise_cross_deproj0_binned_dell.npy"
    )
    validation_report_path = (
        args.observation_validation_report.expanduser().resolve()
        if args.observation_validation_report is not None
        else battaglia_observation_path.parent
        / "battaglia12_baseline_deproj0_validation.json"
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
        fisher_analysis_dir / "fisher_samples_untruncated.npy",
        n_parameters,
    )
    fisher_truncated = load_sample_array(
        fisher_analysis_dir / "fisher_samples_prior_truncated.npy",
        n_parameters,
    )
    if not validation_report_path.is_file():
        raise FileNotFoundError(
            "Matched Battaglia12 validation report is missing. The legacy clean "
            f"fiducial_battaglia12_dell.npy is not a valid observation here: "
            f"{validation_report_path}"
        )
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if not validation_report.get("all_checks_passed", False):
        raise ValueError(f"Battaglia12 observation did not pass validation: {validation_report_path}")
    if validation_report.get("required_product") != "masked_baseline_noise_cross_deproj0":
        raise ValueError(
            f"Validation report has the wrong product: {validation_report.get('required_product')!r}"
        )
    report_dataset = Path(validation_report["prepared_dataset"]).expanduser().resolve()
    if report_dataset != prepared_dataset:
        raise ValueError(
            "Battaglia12 observation was validated against a different prepared dataset: "
            f"{report_dataset} != {prepared_dataset}"
        )
    report_observation = Path(
        validation_report["outputs"]["binned_dell"]
    ).expanduser().resolve()
    if report_observation != battaglia_observation_path:
        raise ValueError(
            "Observation path does not match its validation report: "
            f"{battaglia_observation_path} != {report_observation}"
        )
    battaglia_dell = np.asarray(
        np.load(battaglia_observation_path),
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
    transform_mode = str(transform.get("mode", "none")).lower().replace("-", "_")
    if transform_mode != "asinh":
        raise ValueError(
            f"Expected the main saved NPE transform to be asinh(x/s), found {transform_mode!r} "
            f"in {run_dir / 'x_transform.npz'}."
        )
    transformed_observation = apply_x_transform(battaglia_dell, transform)
    if transformed_observation.size != battaglia_dell.size:
        raise ValueError("Saved NPE x transform changed the observation dimension.")
    np.save(
        output_dir / "battaglia12_baseline_deproj0_observation_transformed.npy",
        transformed_observation,
    )

    final_path = output_dir / f"sbi_samples_N{n_train}.npy"
    partial_path = output_dir / (
        f"sbi_samples_N{n_train}_{args.sampling_method}_partial.npy"
    )
    conditioning_contract_path = output_dir / f"npe_conditioning_contract_N{n_train}.npz"
    if args.force_resample:
        for stale_path in (final_path, partial_path, conditioning_contract_path):
            stale_path.unlink(missing_ok=True)
        print(
            "Removed the incompatible saved NPE chain and conditioning contract; "
            "starting a checkpointed replacement.",
            flush=True,
        )
    existing_chain = final_path.is_file() or partial_path.is_file()
    if existing_chain and not args.force_resample:
        if not conditioning_contract_path.is_file():
            raise FileNotFoundError(
                "Refusing to reuse an NPE chain without an observation contract: "
                f"{conditioning_contract_path}. Use --force-resample to replace it."
            )
        with np.load(conditioning_contract_path, allow_pickle=True) as contract:
            contract_x = np.asarray(contract["transformed_observation"], dtype=np.float32)
            contract_run = str(np.asarray(contract["npe_run_dir"]).reshape(()).item())
            contract_observation = str(
                np.asarray(contract["battaglia_observation"]).reshape(()).item()
            )
        if not np.array_equal(contract_x, transformed_observation):
            raise ValueError(
                "Refusing to reuse NPE samples conditioned on a different observation. "
                "Use --force-resample to replace them."
            )
        if Path(contract_run).expanduser().resolve() != run_dir:
            raise ValueError("Saved NPE chain uses a different density-estimator run.")
        if Path(contract_observation).expanduser().resolve() != battaglia_observation_path:
            raise ValueError("Saved NPE chain uses a different Battaglia12 profile.")
    else:
        np.savez_compressed(
            conditioning_contract_path,
            transformed_observation=transformed_observation,
            binned_dell=battaglia_dell,
            npe_run_dir=np.asarray(str(run_dir)),
            battaglia_observation=np.asarray(str(battaglia_observation_path)),
            validation_report=np.asarray(str(validation_report_path)),
            transform_path=np.asarray(str(run_dir / "x_transform.npz")),
        )
    sampling_preflight_path = output_dir / f"npe_sampling_preflight_N{n_train}.json"
    if final_path.is_file() and not args.force_resample:
        sbi_samples = load_sample_array(final_path, n_parameters)
        print(
            f"Reusing completed NPE chain: {final_path} ({sbi_samples.shape[0]} samples)",
            flush=True,
        )
    else:
        print(f"Loading saved NPE from {run_dir}", flush=True)
        posterior, density_estimator = build_sampling_posterior(
            run_dir,
            args.sampling_method,
            args.mcmc_method,
        )
        try:
            leakage = raw_density_leakage_diagnostic(
                density_estimator,
                transformed_observation,
                prior_low,
                prior_high,
                args.leakage_diagnostic_samples,
                args.device,
            )
        except Exception as exc:
            leakage = {"status": "failed", "error": repr(exc)}
        preflight = {
            "run_dir": str(run_dir),
            "n_train": n_train,
            "sampling_method": args.sampling_method,
            "mcmc_method": args.mcmc_method,
            "mcmc_warmup_steps": args.mcmc_warmup_steps,
            "mcmc_thin": args.mcmc_thin,
            "mcmc_num_chains": args.mcmc_num_chains,
            "mcmc_num_workers": args.mcmc_num_workers,
            "leakage": leakage,
        }
        write_json(sampling_preflight_path, preflight)
        print(f"NPE sampling preflight: {json.dumps(preflight, indent=2)}", flush=True)

        sample_kwargs: dict[str, Any] = {}
        if args.sampling_method == "mcmc":
            sample_kwargs = {
                "method": args.mcmc_method,
                "warmup_steps": args.mcmc_warmup_steps,
                "thin": args.mcmc_thin,
                "num_chains": args.mcmc_num_chains,
                "num_workers": args.mcmc_num_workers,
                "init_strategy": "proposal",
            }
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
            sample_kwargs=sample_kwargs,
        )
        inside = np.all(
            (sbi_samples >= prior_low.reshape(1, -1))
            & (sbi_samples <= prior_high.reshape(1, -1)),
            axis=1,
        )
        if not np.all(inside):
            raise ValueError(
                f"Only {np.mean(inside):.3%} of final NPE samples lie within the prior."
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
            "fisher_analysis_dir": str(fisher_analysis_dir),
            "output_dir": str(output_dir),
            "battaglia_observation": str(battaglia_observation_path),
            "observation_validation_report": str(validation_report_path),
            "npe_conditioning_contract": str(conditioning_contract_path),
            "sbi_run_root": str(sbi_run_root),
            "case": args.case,
            "npe_run_dir": str(run_dir),
            "npe_n_train": n_train,
            "x_transform_mode": transform_mode,
            "n_fisher_untruncated": fisher_untruncated.shape[0],
            "n_fisher_prior_truncated": fisher_truncated.shape[0],
            "n_sbi_samples": sbi_samples.shape[0],
            "sampling_method": args.sampling_method,
            "mcmc_method": args.mcmc_method if args.sampling_method == "mcmc" else None,
            "sampling_preflight": str(sampling_preflight_path),
            "n_plot_samples_max_per_chain": args.plot_samples,
            "plot_mode": args.plot_mode,
        },
    )
    print(f"Fisher/SBI continuation complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
