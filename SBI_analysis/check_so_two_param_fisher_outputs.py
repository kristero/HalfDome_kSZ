#!/usr/bin/env python3
"""Read-only completeness and numerical checks for SO two-parameter and Fisher outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SIZES = (
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    98304, 131072, 196608, 262144, 327680, 393216, 458752, 523288,
)
CORNERS = (256, 32768, 523288)


def scalar_string(value: Any) -> str:
    array = np.asarray(value)
    return str(array.reshape(()).item()) if array.size == 1 else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--two-param-root",
        type=Path,
        default=Path(
            "/lustre/work/kristero10/"
            "adrian_two_param_nsf_convergence_baseline_deproj0"
        ),
    )
    parser.add_argument(
        "--fisher-root",
        type=Path,
        default=Path("/lustre/work/kristero10/adrian_fisher_baseline_deproj0"),
    )
    parser.add_argument(
        "--section",
        choices=("all", "two-param", "fisher"),
        default="all",
        help="Check all outputs, only the two-parameter convergence, or only Fisher.",
    )
    return parser.parse_args()


def load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return {}


def require_files(paths: list[Path], errors: list[str]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {path}")


def check_array(
    path: Path,
    expected_tail: tuple[int, ...],
    errors: list[str],
    minimum_rows: int = 0,
) -> tuple[int, ...]:
    if not path.is_file():
        errors.append(f"missing: {path}")
        return ()
    try:
        values = np.load(path, mmap_mode="r")
        shape = tuple(values.shape)
        if expected_tail and shape[-len(expected_tail):] != expected_tail:
            errors.append(f"wrong shape {path}: {shape}, expected tail {expected_tail}")
        if minimum_rows and (not shape or shape[0] < minimum_rows):
            errors.append(f"too few rows {path}: {shape}, expected >= {minimum_rows}")
        flat = np.asarray(values).reshape(-1)
        if flat.size and not np.all(np.isfinite(flat)):
            errors.append(f"non-finite values: {path}")
        return shape
    except Exception as exc:
        errors.append(f"cannot load {path}: {exc}")
        return ()


def check_two_param(root: Path, errors: list[str], warnings: list[str]) -> None:
    for n_train in SIZES:
        run = root / "asinh" / f"N{n_train}"
        evaluation = run / "evaluation"
        require_files(
            [
                run / "density_estimator.pkl",
                run / "posterior.pkl",
                run / "run_metadata.json",
                run / "x_transform.npz",
                run / "train_indices.npy",
                evaluation / "evaluation_complete.json",
                evaluation / "evaluation_preflight.json",
                evaluation / "heldout_metrics.csv",
                evaluation / "heldout_summary.csv",
                evaluation / "heldout_posterior_samples.npy",
            ],
            errors,
        )
        metadata = load_json(run / "run_metadata.json", errors)
        if metadata:
            if metadata.get("density_estimator") != "nsf":
                errors.append(f"N={n_train}: density estimator is not NSF")
            if int(metadata.get("n_train", -1)) != n_train:
                errors.append(f"N={n_train}: run_metadata n_train mismatch")
            if metadata.get("x_rescale_mode") != "asinh":
                errors.append(f"N={n_train}: x_rescale_mode is not asinh")
        transform_path = run / "x_transform.npz"
        if transform_path.is_file():
            with np.load(transform_path, allow_pickle=True) as transform:
                mode = scalar_string(transform["mode"]) if "mode" in transform.files else ""
                scale = np.asarray(transform["scale"]) if "scale" in transform.files else np.empty(0)
            if mode != "asinh" or scale.shape != (40,) or not np.all(np.isfinite(scale)):
                errors.append(f"N={n_train}: invalid saved asinh transform")
        completion = load_json(evaluation / "evaluation_complete.json", errors)
        if completion:
            checks = completion.get("validation", {})
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.append(f"N={n_train}: evaluation validation failed: {failed}")
            if int(completion.get("num_test_profiles", -1)) != 1000:
                errors.append(f"N={n_train}: expected 1000 held-out profiles")
            if int(completion.get("num_posterior_samples_per_test", -1)) != 2000:
                errors.append(f"N={n_train}: expected 2000 samples per profile")
        check_array(
            evaluation / "heldout_posterior_samples.npy",
            (2000, 2),
            errors,
            minimum_rows=1000,
        )

    for n_train in CORNERS:
        run = root / "asinh" / f"N{n_train}"
        samples = run / "battaglia12_posterior_samples.npy"
        contract = run / "battaglia12_conditioning_contract.npz"
        check_array(samples, (2,), errors, minimum_rows=20000)
        if not contract.is_file():
            errors.append(f"N={n_train}: missing corrected Battaglia contract: {contract}")
            continue
        with np.load(contract, allow_pickle=True) as data:
            source = scalar_string(data["observation_source"])
            observation = np.asarray(data["observation"], dtype=np.float32)
            transformed = np.asarray(data["transformed_observation"], dtype=np.float32)
        if not source.startswith("validated_baseline_deproj0:"):
            errors.append(f"N={n_train}: wrong Battaglia source: {source!r}")
        if observation.shape != (40,) or transformed.shape != (40,):
            errors.append(f"N={n_train}: wrong Battaglia observation dimensions")
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(transformed)):
            errors.append(f"N={n_train}: non-finite Battaglia conditioning vector")

    summary = root / "summary"
    required_summary = [
        "convergence_summary.json",
        "heldout_metrics_all_runs.csv",
        "convergence_metrics.csv",
        "validation_performance.csv",
        "correlation_vs_dataset_size.jpg",
        "prior_normalized_rmse_vs_dataset_size.jpg",
        "rmse_over_posterior_std_vs_dataset_size.jpg",
        "posterior_std_over_prior_vs_dataset_size.jpg",
        "validation_performance_vs_dataset_size.jpg",
        "true_vs_mean_min_mid_max_asinh.jpg",
        "battaglia12_P0_beta_corner_min_mid_max_asinh.jpg",
        "sbc_rank_cdf_max_dataset_size.jpg",
        "battaglia12_constraints_min_mid_max.csv",
    ]
    require_files([summary / name for name in required_summary], errors)
    summary_json = load_json(summary / "convergence_summary.json", errors)
    if summary_json.get("missing_runs"):
        errors.append(f"two-parameter summary reports missing runs: {summary_json['missing_runs']}")


def check_fisher(root: Path, errors: list[str], warnings: list[str]) -> None:
    analysis = root / "analysis"
    require_files(
        [
            analysis / "derivatives_richardson.npy",
            analysis / "covariance_shrunk.npy",
            analysis / "inverse_covariance.npy",
            analysis / "fisher_matrix_theta.npy",
            analysis / "fisher_covariance_theta.npy",
            analysis / "fisher_samples_untruncated.npy",
            analysis / "fisher_samples_prior_truncated.npy",
            analysis / "fisher_derivative_stability.jpg",
            analysis / "covariance_diagnostics.jpg",
        ],
        errors,
    )
    if not (analysis / "fisher_analysis_summary.json").is_file():
        warnings.append(
            "Legacy fisher_analysis_summary.json is absent; this is acceptable when "
            "the original direct-posterior stage was replaced by the matched resume job."
        )
    check_array(analysis / "derivatives_richardson.npy", (9, 40), errors)
    check_array(analysis / "covariance_shrunk.npy", (40, 40), errors)
    check_array(analysis / "inverse_covariance.npy", (40, 40), errors)
    check_array(analysis / "fisher_matrix_theta.npy", (9, 9), errors)
    check_array(analysis / "fisher_covariance_theta.npy", (9, 9), errors)
    check_array(
        analysis / "fisher_samples_untruncated.npy", (9,), errors, minimum_rows=10000
    )
    check_array(
        analysis / "fisher_samples_prior_truncated.npy", (9,), errors, minimum_rows=10000
    )

    validation = root / "battaglia12_baseline_deproj0_observation" / "prepared"
    validation_json = load_json(
        validation / "battaglia12_baseline_deproj0_validation.json", errors
    )
    if validation_json and not validation_json.get("all_checks_passed", False):
        errors.append("matched Battaglia12 validation report did not pass")
    check_array(
        validation / "battaglia12_masked_baseline_noise_cross_deproj0_binned_dell.npy",
        (40,),
        errors,
    )

    convergence = analysis / "covariance_convergence"
    convergence_json = load_json(
        convergence / "covariance_convergence_summary.json", errors
    )
    require_files(
        [
            convergence / "covariance_convergence_metrics.csv",
            convergence / "fisher_parameter_convergence.csv",
            convergence / "preferred_covariance.npy",
            convergence / "preferred_precision.npy",
            convergence / "preferred_fisher_covariance_theta.npy",
            convergence / "preferred_p0_beta_covariance_theta.npy",
            convergence / "covariance_and_fisher_diagnostics_vs_rows.jpg",
            convergence / "p0_beta_marginalized_ellipses_maxN.jpg",
        ],
        errors,
    )
    check_array(convergence / "preferred_covariance.npy", (40, 40), errors)
    check_array(convergence / "preferred_precision.npy", (40, 40), errors)
    check_array(
        convergence / "preferred_fisher_covariance_theta.npy", (9, 9), errors
    )
    if convergence_json:
        acceptance = convergence_json.get("acceptance_checks", {})
        for name, passed in acceptance.items():
            if not passed:
                warnings.append(f"Fisher scientific acceptance check is false: {name}")

    matched = analysis / "matched_baseline_deproj0_comparison"
    matched_summary = load_json(
        matched / "fisher_comparison_resume_summary.json", errors
    )
    require_files(
        [
            matched / "fisher_vs_sbi_constraints.csv",
            matched / "fisher_untruncated_truncated_vs_sbi_corner.jpg",
            matched / "fisher_untruncated_vs_sbi_corner.jpg",
            matched / "fisher_prior_truncated_vs_sbi_corner.jpg",
        ],
        errors,
    )
    if matched_summary:
        if matched_summary.get("x_transform_mode") != "asinh":
            errors.append("matched Fisher/SBI comparison did not use asinh")
        n_train = int(matched_summary.get("npe_n_train", -1))
        sample_path = matched / f"sbi_samples_N{n_train}.npy"
        contract_path = matched / f"npe_conditioning_contract_N{n_train}.npz"
        check_array(sample_path, (9,), errors, minimum_rows=1000)
        require_files([contract_path], errors)


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    two_param_root = args.two_param_root.expanduser().resolve()
    fisher_root = args.fisher_root.expanduser().resolve()
    if args.section in ("all", "two-param"):
        check_two_param(two_param_root, errors, warnings)
        print(f"Two-parameter root: {two_param_root}")
    if args.section in ("all", "fisher"):
        check_fisher(fisher_root, errors, warnings)
        print(f"Fisher root: {fisher_root}")
    if errors:
        print("\nCOMPLETION/NUMERICAL ERRORS:")
        for item in errors:
            print(f"  - {item}")
    else:
        print("\nCOMPUTATION CHECK: PASSED")
    if warnings:
        print("\nSCIENTIFIC WARNINGS:")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("\nSCIENTIFIC ACCEPTANCE FLAGS: PASSED")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
