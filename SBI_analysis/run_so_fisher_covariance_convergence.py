#!/usr/bin/env python3
"""Covariance convergence and Fisher stability for baseline-deproj0 SO data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from run_so_fisher_analysis import (
    BATTAGLIA12,
    DEFAULT_FISHER_ROOT,
    DEFAULT_PREPARED_DATASET,
    DEFAULT_RAW_DIR,
    PARAMETER_NAMES,
    binned_residuals,
    jsonable,
    linear_dependence_diagnostics,
    make_bin_plan,
    regularized_inverse,
    validate_raw_alignment,
)


DEFAULT_SIZES = (500, 1000, 2000, 5000, 10000, 20000)
METHODS = (
    "raw_fixed",
    "raw_oas",
    "detrended_fixed",
    "detrended_oas",
)
METHOD_LABELS = {
    "raw_fixed": "Raw, 5% shrinkage",
    "raw_oas": "Raw, OAS",
    "detrended_fixed": "Detrended, 5% shrinkage",
    "detrended_oas": "Detrended, OAS",
}
METHOD_COLORS = {
    "raw_fixed": "#7f7f7f",
    "raw_oas": "#1f77b4",
    "detrended_fixed": "#d62728",
    "detrended_oas": "#2ca02c",
}
PARAMETER_LABELS = {
    "P0": r"$P_0$",
    "xc": r"$x_{\rm c}$",
    "beta": r"$\beta$",
    "alpha_m_P0": r"$\alpha_{m,P_0}$",
    "alpha_m_xc": r"$\alpha_{m,x_{\rm c}}$",
    "alpha_m_beta": r"$\alpha_{m,\beta}$",
    "alpha_z_P0": r"$\alpha_{z,P_0}$",
    "alpha_z_xc": r"$\alpha_{z,x_{\rm c}}$",
    "alpha_z_beta": r"$\alpha_{z,\beta}$",
}


def parse_int_list(value: str) -> list[int]:
    values = [
        int(part.replace("_", ""))
        for part in str(value).replace(",", " ").split()
        if part
    ]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Expected a non-empty list of positive integers.")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("Sizes must be unique and increasing.")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(part) for part in str(value).replace(",", " ").split() if part]
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("Expected positive floating-point values.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dataset", type=Path, default=DEFAULT_PREPARED_DATASET)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--fisher-analysis-dir",
        type=Path,
        default=DEFAULT_FISHER_ROOT / "analysis",
        help="Directory containing derivatives_richardson.npy.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--sizes",
        type=parse_int_list,
        default=list(DEFAULT_SIZES),
        help="Nested nearest-neighbour counts, comma separated.",
    )
    parser.add_argument("--random-rows", type=int, default=20_000)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--bin-weighting",
        choices=("2ell_plus_1", "uniform", "ell"),
        default="2ell_plus_1",
    )
    parser.add_argument("--fixed-shrinkage", type=float, default=0.05)
    parser.add_argument("--detrend-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--covariance-eigenvalue-floor", type=float, default=1.0e-10)
    parser.add_argument("--fisher-eigenvalue-floor", type=float, default=1.0e-10)
    parser.add_argument("--fisher-rcond", type=float, default=1.0e-8)
    parser.add_argument(
        "--fisher-floor-grid",
        type=parse_float_list,
        default=[1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6],
    )
    parser.add_argument("--no-hartlap", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: jsonable(row.get(field, "")) for field in fields})


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "metadata": cache_dir / "cache_metadata.json",
        "local_residuals": cache_dir / "local_residuals_ranked.npy",
        "local_theta": cache_dir / "local_theta_ranked.npy",
        "local_indices": cache_dir / "local_indices_ranked.npy",
        "local_distance": cache_dir / "local_normalized_distance.npy",
        "local_global_rows": cache_dir / "local_sobol_global_row_ranked.npy",
        "random_residuals": cache_dir / "random_residuals.npy",
        "random_theta": cache_dir / "random_theta.npy",
        "random_indices": cache_dir / "random_indices.npy",
        "prior_low": cache_dir / "prior_low.npy",
        "prior_high": cache_dir / "prior_high.npy",
        "ell_binned": cache_dir / "ell_binned.npy",
    }


def cache_complete(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def build_residual_cache(
    *,
    prepared_dataset: Path,
    raw_dir: Path,
    cache_dir: Path,
    sizes: list[int],
    random_rows: int,
    chunk_rows: int,
    bin_weighting: str,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    clean_cl, noisy_cl, theta, sobol_global_row, metadata = validate_raw_alignment(
        prepared_dataset,
        raw_dir,
    )
    prior_low = np.asarray(metadata["prior_low"], dtype=np.float64)
    prior_high = np.asarray(metadata["prior_high"], dtype=np.float64)
    prior_width = prior_high - prior_low
    if np.any(prior_width <= 0.0):
        raise ValueError("Prior widths must be positive.")

    max_local = max(sizes)
    if max_local > theta.shape[0] or random_rows > theta.shape[0]:
        raise ValueError(
            f"Requested local/random rows {max_local}/{random_rows}; "
            f"only {theta.shape[0]} rows are available."
        )

    normalized = (theta - BATTAGLIA12) / prior_width
    distance_squared = np.einsum("ij,ij->i", normalized, normalized)
    nearest = np.argpartition(distance_squared, max_local - 1)[:max_local]
    nearest = nearest[np.argsort(distance_squared[nearest])].astype(np.int64)
    local_read_order = np.sort(nearest)

    rng = np.random.default_rng(seed)
    random_indices = np.sort(
        rng.choice(theta.shape[0], size=random_rows, replace=False).astype(np.int64)
    )
    plan = make_bin_plan(
        np.asarray(metadata["ell"], dtype=np.float64),
        np.asarray(metadata["bin_ell_min"], dtype=np.float64),
        np.asarray(metadata["bin_ell_max"], dtype=np.float64),
        bin_weighting,
    )

    print(f"Caching {max_local} nearest residual rows...", flush=True)
    local_read_residuals = binned_residuals(
        noisy_cl,
        clean_cl,
        local_read_order,
        np.asarray(metadata["ell"], dtype=np.float64),
        plan,
        chunk_rows,
    )
    ranked_positions = np.searchsorted(local_read_order, nearest)
    local_residuals = np.ascontiguousarray(
        local_read_residuals[ranked_positions], dtype=np.float64
    )

    print(f"Caching {random_rows} random residual rows...", flush=True)
    random_residuals = binned_residuals(
        noisy_cl,
        clean_cl,
        random_indices,
        np.asarray(metadata["ell"], dtype=np.float64),
        plan,
        chunk_rows,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    np.save(paths["local_residuals"], local_residuals)
    np.save(paths["local_theta"], theta[nearest])
    np.save(paths["local_indices"], nearest)
    np.save(paths["local_distance"], np.sqrt(distance_squared[nearest]))
    np.save(paths["local_global_rows"], sobol_global_row[nearest])
    np.save(paths["random_residuals"], random_residuals)
    np.save(paths["random_theta"], theta[random_indices])
    np.save(paths["random_indices"], random_indices)
    np.save(paths["prior_low"], prior_low)
    np.save(paths["prior_high"], prior_high)
    np.save(paths["ell_binned"], np.asarray(metadata["ell_binned"], dtype=np.float64))

    cache_metadata = {
        "prepared_dataset": str(prepared_dataset.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "bin_weighting": bin_weighting,
        "sizes": sizes,
        "max_local_rows": max_local,
        "random_rows": random_rows,
        "chunk_rows": chunk_rows,
        "seed": seed,
        "n_bins": len(plan),
        "parameter_names": PARAMETER_NAMES,
        "alignment": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(paths["metadata"], cache_metadata)
    return cache_metadata


def load_cache(cache_dir: Path, sizes: list[int]) -> dict[str, Any]:
    paths = cache_paths(cache_dir)
    if not cache_complete(paths):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError("Residual cache is incomplete:\n  " + "\n  ".join(missing))
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    local_residuals = np.load(paths["local_residuals"], mmap_mode="r")
    local_theta = np.load(paths["local_theta"], mmap_mode="r")
    local_distance = np.load(paths["local_distance"], mmap_mode="r")
    if max(sizes) > local_residuals.shape[0]:
        raise ValueError(
            f"Cache has {local_residuals.shape[0]} local rows but N={max(sizes)} was requested. "
            "Rerun with --rebuild-cache."
        )
    return {
        "metadata": metadata,
        "local_residuals": local_residuals,
        "local_theta": local_theta,
        "local_indices": np.load(paths["local_indices"], mmap_mode="r"),
        "local_distance": local_distance,
        "random_residuals": np.load(paths["random_residuals"], mmap_mode="r"),
        "random_theta": np.load(paths["random_theta"], mmap_mode="r"),
        "random_indices": np.load(paths["random_indices"], mmap_mode="r"),
        "prior_low": np.load(paths["prior_low"]),
        "prior_high": np.load(paths["prior_high"]),
        "ell_binned": np.load(paths["ell_binned"]),
    }


def polynomial_design(
    theta: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    degree: int,
) -> np.ndarray:
    normalized = (np.asarray(theta, dtype=np.float64) - center) / scale
    columns = [np.ones(normalized.shape[0], dtype=np.float64)]
    columns.extend(normalized[:, index] for index in range(normalized.shape[1]))
    if degree == 2:
        for first in range(normalized.shape[1]):
            for second in range(first, normalized.shape[1]):
                columns.append(normalized[:, first] * normalized[:, second])
    return np.column_stack(columns)


def cross_fitted_detrend(
    theta: np.ndarray,
    residuals: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    degree: int,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    design = polynomial_design(theta, center, scale, degree)
    n_rows = design.shape[0]
    if folds < 2 or folds > n_rows:
        raise ValueError(f"Cross-fit folds must be in 2..{n_rows}; found {folds}.")
    if n_rows - int(np.ceil(n_rows / folds)) <= design.shape[1]:
        raise ValueError(
            f"Too few rows ({n_rows}) for {design.shape[1]} detrending features "
            f"and {folds} folds."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    fold_id = np.empty(n_rows, dtype=np.int64)
    fold_id[permutation] = np.arange(n_rows) % folds
    predicted = np.empty_like(residuals, dtype=np.float64)
    ranks = []
    for fold in range(folds):
        test = fold_id == fold
        train = ~test
        coefficients, _, rank, _ = np.linalg.lstsq(
            design[train],
            np.asarray(residuals[train], dtype=np.float64),
            rcond=1.0e-10,
        )
        predicted[test] = design[test] @ coefficients
        ranks.append(int(rank))
    detrended = np.asarray(residuals, dtype=np.float64) - predicted
    return detrended, {
        "degree": degree,
        "folds": folds,
        "n_features": int(design.shape[1]),
        "minimum_fold_fit_rank": min(ranks),
        "maximum_fold_fit_rank": max(ranks),
    }


def fixed_diagonal_shrinkage(
    residuals: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, float]:
    sample = np.cov(np.asarray(residuals, dtype=np.float64), rowvar=False, ddof=1)
    target = np.diag(np.diag(sample))
    return (1.0 - shrinkage) * sample + shrinkage * target, shrinkage


def oas_correlation_covariance(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(residuals, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    sample_covariance = centered.T @ centered / (values.shape[0] - 1.0)
    sigma = np.sqrt(np.clip(np.diag(sample_covariance), 0.0, None))
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("OAS requires finite, positive residual standard deviations.")

    standardized = centered / sigma
    empirical = standardized.T @ standardized / values.shape[0]
    n_features = empirical.shape[0]
    mu = np.trace(empirical) / n_features
    alpha = np.mean(empirical**2)
    numerator = alpha + mu**2
    denominator = (values.shape[0] + 1.0) * (
        alpha - (mu**2) / n_features
    )
    shrinkage = 1.0 if denominator <= 0.0 else min(numerator / denominator, 1.0)
    shrunk_correlation = (1.0 - shrinkage) * empirical
    shrunk_correlation.flat[:: n_features + 1] += shrinkage * mu
    diagonal = np.sqrt(np.clip(np.diag(shrunk_correlation), 0.0, None))
    shrunk_correlation = np.divide(
        shrunk_correlation,
        np.outer(diagonal, diagonal),
        out=np.eye(n_features, dtype=np.float64),
        where=np.outer(diagonal, diagonal) > 0.0,
    )
    covariance = np.outer(sigma, sigma) * shrunk_correlation
    return 0.5 * (covariance + covariance.T), float(shrinkage)


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(sigma, sigma)
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance, dtype=np.float64),
        where=denominator > 0.0,
    )


def covariance_builder(method: str, fixed_shrinkage: float) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
    if method.endswith("_fixed"):
        return lambda residuals: fixed_diagonal_shrinkage(residuals, fixed_shrinkage)
    if method.endswith("_oas"):
        return oas_correlation_covariance
    raise ValueError(f"Unsupported covariance method: {method}")


def fisher_from_covariance(
    covariance: np.ndarray,
    derivatives_q: np.ndarray,
    n_rows: int,
    covariance_floor: float,
    fisher_floor: float,
    fisher_rcond: float,
    apply_hartlap: bool,
) -> dict[str, Any]:
    precision, covariance_diagnostics, covariance_repaired = regularized_inverse(
        covariance,
        covariance_floor,
    )
    hartlap = 1.0
    n_bins = covariance.shape[0]
    if apply_hartlap:
        if n_rows <= n_bins + 2:
            raise ValueError(
                f"Hartlap correction requires N>{n_bins + 2}; found {n_rows}."
            )
        hartlap = (n_rows - n_bins - 2.0) / (n_rows - 1.0)
        precision *= hartlap

    fisher = derivatives_q @ precision @ derivatives_q.T
    fisher = 0.5 * (fisher + fisher.T)
    fisher_eigenvalues = np.linalg.eigvalsh(fisher)
    largest = float(fisher_eigenvalues[-1])
    tolerance = largest * fisher_rcond
    identified = fisher_eigenvalues > tolerance
    identified_values = fisher_eigenvalues[identified]
    fisher_rank = int(np.count_nonzero(identified))
    identified_condition = (
        float(identified_values[-1] / identified_values[0])
        if identified_values.size
        else float("inf")
    )
    fisher_covariance, fisher_diagnostics, fisher_repaired = regularized_inverse(
        fisher,
        fisher_floor,
    )
    return {
        "covariance": covariance_repaired,
        "precision": precision,
        "covariance_diagnostics": covariance_diagnostics,
        "hartlap_factor": hartlap,
        "fisher": fisher_repaired,
        "fisher_covariance_q": fisher_covariance,
        "fisher_diagnostics": fisher_diagnostics,
        "fisher_rank": fisher_rank,
        "fisher_rcond": fisher_rcond,
        "identified_condition": identified_condition,
        "fisher_eigenvalues": fisher_eigenvalues,
    }


def evaluate_candidates(
    *,
    cache: dict[str, Any],
    sizes: list[int],
    derivatives: np.ndarray,
    output_dir: Path,
    fixed_shrinkage: float,
    detrend_degree: int,
    crossfit_folds: int,
    covariance_floor: float,
    fisher_floor: float,
    fisher_rcond: float,
    apply_hartlap: bool,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prior_low = np.asarray(cache["prior_low"], dtype=np.float64)
    prior_high = np.asarray(cache["prior_high"], dtype=np.float64)
    prior_width = prior_high - prior_low
    derivatives_q = np.asarray(derivatives, dtype=np.float64) * prior_width[:, None]
    random_covariance = np.cov(
        np.asarray(cache["random_residuals"], dtype=np.float64),
        rowvar=False,
        ddof=1,
    )
    random_sigma = np.sqrt(np.diag(random_covariance))
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []

    for n_rows in sizes:
        print(f"Evaluating covariance N={n_rows}...", flush=True)
        theta = np.asarray(cache["local_theta"][:n_rows], dtype=np.float64)
        raw = np.asarray(cache["local_residuals"][:n_rows], dtype=np.float64)
        distance = np.asarray(cache["local_distance"][:n_rows], dtype=np.float64)
        detrended, detrend_metadata = cross_fitted_detrend(
            theta,
            raw,
            BATTAGLIA12,
            prior_width,
            detrend_degree,
            crossfit_folds,
            seed + n_rows,
        )
        raw_r2, raw_parameter_correlation = linear_dependence_diagnostics(
            theta,
            raw,
            BATTAGLIA12,
            prior_width,
        )
        detrended_r2, detrended_parameter_correlation = linear_dependence_diagnostics(
            theta,
            detrended,
            BATTAGLIA12,
            prior_width,
        )
        run_dir = output_dir / f"N{n_rows}"
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "raw_residual_r_squared.npy", raw_r2)
        np.save(run_dir / "detrended_residual_r_squared.npy", detrended_r2)
        np.save(run_dir / "raw_residual_parameter_correlation.npy", raw_parameter_correlation)
        np.save(
            run_dir / "detrended_residual_parameter_correlation.npy",
            detrended_parameter_correlation,
        )
        raw_sample_covariance = np.cov(raw, rowvar=False, ddof=1)
        raw_sigma = np.sqrt(np.diag(raw_sample_covariance))
        sigma_ratio = raw_sigma / random_sigma
        local_random_frobenius = (
            np.linalg.norm(raw_sample_covariance - random_covariance, ord="fro")
            / np.linalg.norm(random_covariance, ord="fro")
        )

        for method in METHODS:
            values = detrended if method.startswith("detrended_") else raw
            builder = covariance_builder(method, fixed_shrinkage)
            covariance, shrinkage = builder(values)
            result = fisher_from_covariance(
                covariance,
                derivatives_q,
                n_rows,
                covariance_floor,
                fisher_floor,
                fisher_rcond,
                apply_hartlap,
            )
            fisher_covariance_theta = (
                prior_width[:, None]
                * result["fisher_covariance_q"]
                * prior_width[None, :]
            )
            np.save(run_dir / f"{method}_covariance.npy", result["covariance"])
            np.save(run_dir / f"{method}_precision.npy", result["precision"])
            np.save(run_dir / f"{method}_fisher_matrix_normalized.npy", result["fisher"])
            np.save(
                run_dir / f"{method}_fisher_covariance_normalized.npy",
                result["fisher_covariance_q"],
            )
            np.save(
                run_dir / f"{method}_fisher_covariance_theta.npy",
                fisher_covariance_theta,
            )

            covariance_diagnostics = result["covariance_diagnostics"]
            fisher_diagnostics = result["fisher_diagnostics"]
            selected_r2 = detrended_r2 if method.startswith("detrended_") else raw_r2
            selected_corr = (
                detrended_parameter_correlation
                if method.startswith("detrended_")
                else raw_parameter_correlation
            )
            metric_rows.append({
                "n_rows": n_rows,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "detrended": method.startswith("detrended_"),
                "shrinkage": shrinkage,
                "hartlap_factor": result["hartlap_factor"],
                "local_distance_median": float(np.median(distance)),
                "local_distance_max": float(np.max(distance)),
                "local_random_covariance_frobenius": float(local_random_frobenius),
                "local_random_sigma_ratio_min": float(np.min(sigma_ratio)),
                "local_random_sigma_ratio_median": float(np.median(sigma_ratio)),
                "local_random_sigma_ratio_max": float(np.max(sigma_ratio)),
                "residual_r2_median": float(np.median(selected_r2)),
                "residual_r2_max": float(np.max(selected_r2)),
                "residual_parameter_correlation_abs_max": float(np.max(np.abs(selected_corr))),
                "detrend_features": detrend_metadata["n_features"],
                "detrend_fit_rank_min": detrend_metadata["minimum_fold_fit_rank"],
                "covariance_condition_before_floor": covariance_diagnostics["condition_before_floor"],
                "covariance_condition_after_floor": covariance_diagnostics["condition_after_floor"],
                "covariance_modes_floored": covariance_diagnostics["n_eigenvalues_floored"],
                "fisher_condition_before_floor": fisher_diagnostics["condition_before_floor"],
                "fisher_condition_after_floor": fisher_diagnostics["condition_after_floor"],
                "fisher_modes_floored": fisher_diagnostics["n_eigenvalues_floored"],
                "fisher_identified_rank": result["fisher_rank"],
                "fisher_identified_condition": result["identified_condition"],
            })

            normalized_std = np.sqrt(np.diag(result["fisher_covariance_q"]))
            theta_std = np.sqrt(np.diag(fisher_covariance_theta))
            for index, parameter in enumerate(PARAMETER_NAMES):
                parameter_rows.append({
                    "n_rows": n_rows,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "parameter": parameter,
                    "parameter_index": index,
                    "truth": BATTAGLIA12[index],
                    "std": float(theta_std[index]),
                    "std_over_prior": float(normalized_std[index]),
                })

    return metric_rows, parameter_rows


def paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def rows_for_method(
    rows: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    return sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: int(row["n_rows"]),
    )


def plot_covariance_summary(
    metrics: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    panels = (
        ("covariance_condition_after_floor", "Covariance condition", True),
        ("fisher_condition_after_floor", "Fisher condition", True),
        ("fisher_identified_rank", "Identified Fisher rank", False),
        ("residual_r2_max", r"Maximum local residual $R^2$", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(18.0 / 2.54, 12.0 / 2.54))
    for axis, (key, ylabel, logarithmic_y) in zip(axes.flat, panels):
        for method in METHODS:
            rows = rows_for_method(metrics, method)
            axis.plot(
                [row["n_rows"] for row in rows],
                [row[key] for row in rows],
                marker="o",
                ms=3.0,
                lw=0.9,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log")
        if logarithmic_y:
            axis.set_yscale("log")
        axis.set_xlabel("Local covariance rows")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25, lw=0.5)
    axes[1, 1].axhline(0.05, color="black", lw=0.7, ls="--")
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_local_diagnostics(
    metrics: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    rows = rows_for_method(metrics, "raw_fixed")
    sizes = np.asarray([row["n_rows"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(18.0 / 2.54, 6.2 / 2.54))

    axes[0].plot(sizes, [row["local_distance_median"] for row in rows], marker="o", lw=0.9, label="median")
    axes[0].plot(sizes, [row["local_distance_max"] for row in rows], marker="s", lw=0.9, label="maximum")
    axes[0].set_ylabel(r"Distance from Battaglia12 in prior units")
    axes[0].legend(frameon=False)

    axes[1].plot(sizes, [row["local_random_sigma_ratio_min"] for row in rows], marker="o", lw=0.9, label="minimum")
    axes[1].plot(sizes, [row["local_random_sigma_ratio_median"] for row in rows], marker="s", lw=0.9, label="median")
    axes[1].plot(sizes, [row["local_random_sigma_ratio_max"] for row in rows], marker="^", lw=0.9, label="maximum")
    axes[1].axhline(1.0, color="black", lw=0.7)
    axes[1].axhspan(0.8, 1.25, color="0.85")
    axes[1].set_ylabel(r"$\sigma_{\rm local}/\sigma_{\rm random}$")
    axes[1].legend(frameon=False)

    axes[2].plot(sizes, [row["local_random_covariance_frobenius"] for row in rows], marker="o", lw=0.9)
    axes[2].set_ylabel("Local/random covariance difference")

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Local covariance rows")
        axis.grid(True, alpha=0.25, lw=0.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_parameter_convergence(
    parameter_rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18.0 / 2.54, 17.0 / 2.54), sharex=True)
    for axis, parameter in zip(axes.flat, PARAMETER_NAMES):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in parameter_rows
                    if row["method"] == method and row["parameter"] == parameter
                ),
                key=lambda row: int(row["n_rows"]),
            )
            axis.plot(
                [row["n_rows"] for row in rows],
                [row["std_over_prior"] for row in rows],
                marker="o",
                ms=2.5,
                lw=0.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log")
        axis.set_title(PARAMETER_LABELS[parameter])
        axis.grid(True, alpha=0.25, lw=0.5)
    for axis in axes[-1]:
        axis.set_xlabel("Local covariance rows")
    for axis in axes[:, 0]:
        axis.set_ylabel("Marginalized std / prior width")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_p0_beta_convergence(
    parameter_rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(18.0 / 2.54, 6.5 / 2.54))
    for axis, parameter in zip(axes, ("P0", "beta")):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in parameter_rows
                    if row["method"] == method and row["parameter"] == parameter
                ),
                key=lambda row: int(row["n_rows"]),
            )
            axis.plot(
                [row["n_rows"] for row in rows],
                [row["std_over_prior"] for row in rows],
                marker="o",
                ms=3.0,
                lw=0.9,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log")
        axis.set_xlabel("Local covariance rows")
        axis.set_ylabel("Marginalized std / prior width")
        axis.set_title(PARAMETER_LABELS[parameter])
        axis.grid(True, alpha=0.25, lw=0.5)
    axes[0].legend(frameon=False, fontsize=6.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_oas_shrinkage(
    metrics: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0 / 2.54, 6.5 / 2.54))
    for method in ("raw_oas", "detrended_oas"):
        rows = rows_for_method(metrics, method)
        axis.plot(
            [row["n_rows"] for row in rows],
            [row["shrinkage"] for row in rows],
            marker="o",
            ms=3.0,
            lw=0.9,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Local covariance rows")
    axis.set_ylabel("OAS correlation shrinkage")
    axis.grid(True, alpha=0.25, lw=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_p0_beta_ellipses(
    output_dir: Path,
    n_rows: int,
    dpi: int,
) -> None:
    p0_index = PARAMETER_NAMES.index("P0")
    beta_index = PARAMETER_NAMES.index("beta")
    selected = np.array([p0_index, beta_index], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(8.5 / 2.54, 7.0 / 2.54))
    extents = []
    for method in METHODS:
        covariance = np.load(
            output_dir / f"N{n_rows}" / f"{method}_fisher_covariance_theta.npy"
        )[np.ix_(selected, selected)]
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        for chi_squared, linestyle, alpha in ((2.30, "-", 0.16), (5.99, "--", 0.0)):
            width, height = 2.0 * np.sqrt(chi_squared * eigenvalues[::-1])
            ellipse = Ellipse(
                (BATTAGLIA12[p0_index], BATTAGLIA12[beta_index]),
                width,
                height,
                angle=angle,
                facecolor=METHOD_COLORS[method] if method == "detrended_oas" and alpha else "none",
                edgecolor=METHOD_COLORS[method],
                alpha=alpha if alpha else 1.0,
                lw=1.0,
                ls=linestyle,
                label=METHOD_LABELS[method] if chi_squared == 2.30 else None,
            )
            axis.add_patch(ellipse)
            extents.append((width, height))
    maximum_width = max(value[0] for value in extents)
    maximum_height = max(value[1] for value in extents)
    axis.set_xlim(BATTAGLIA12[p0_index] - 0.65 * maximum_width, BATTAGLIA12[p0_index] + 0.65 * maximum_width)
    axis.set_ylim(BATTAGLIA12[beta_index] - 0.65 * maximum_height, BATTAGLIA12[beta_index] + 0.65 * maximum_height)
    axis.axvline(BATTAGLIA12[p0_index], color="black", lw=0.7, ls=":")
    axis.axhline(BATTAGLIA12[beta_index], color="black", lw=0.7, ls=":")
    axis.set_xlabel(PARAMETER_LABELS["P0"])
    axis.set_ylabel(PARAMETER_LABELS["beta"])
    axis.legend(frameon=False, fontsize=6.2)
    axis.grid(True, alpha=0.2, lw=0.5)
    figure.tight_layout()
    figure.savefig(output_dir / "p0_beta_marginalized_ellipses_maxN.jpg", dpi=dpi)
    plt.close(figure)


def fisher_floor_sensitivity(
    *,
    output_dir: Path,
    n_rows: int,
    method: str,
    floor_grid: list[float],
    derivatives: np.ndarray,
    prior_width: np.ndarray,
    covariance_floor: float,
    apply_hartlap: bool,
    dpi: int,
) -> list[dict[str, Any]]:
    covariance = np.load(output_dir / f"N{n_rows}" / f"{method}_covariance.npy")
    precision, _, _ = regularized_inverse(covariance, covariance_floor)
    if apply_hartlap:
        n_bins = covariance.shape[0]
        precision *= (n_rows - n_bins - 2.0) / (n_rows - 1.0)
    derivatives_q = derivatives * prior_width[:, None]
    fisher = derivatives_q @ precision @ derivatives_q.T
    rows: list[dict[str, Any]] = []
    for floor in floor_grid:
        covariance_q, diagnostics, _ = regularized_inverse(fisher, floor)
        std = np.sqrt(np.diag(covariance_q))
        for parameter, value in zip(PARAMETER_NAMES, std):
            rows.append({
                "n_rows": n_rows,
                "method": method,
                "fisher_eigenvalue_floor": floor,
                "parameter": parameter,
                "std_over_prior": float(value),
                "fisher_modes_floored": diagnostics["n_eigenvalues_floored"],
                "fisher_condition_after_floor": diagnostics["condition_after_floor"],
            })

    figure, axis = plt.subplots(figsize=(10.0 / 2.54, 7.0 / 2.54))
    for parameter in PARAMETER_NAMES:
        selected = [row for row in rows if row["parameter"] == parameter]
        axis.plot(
            [row["fisher_eigenvalue_floor"] for row in selected],
            [row["std_over_prior"] for row in selected],
            marker="o",
            ms=2.5,
            lw=0.8,
            label=PARAMETER_LABELS[parameter],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Relative Fisher eigenvalue floor")
    axis.set_ylabel("Marginalized std / prior width")
    axis.grid(True, alpha=0.25, lw=0.5)
    axis.legend(frameon=False, ncol=3, fontsize=6.5)
    figure.tight_layout()
    figure.savefig(output_dir / "fisher_floor_sensitivity_maxN.jpg", dpi=dpi)
    plt.close(figure)
    return rows


def convergence_stability(
    parameter_rows: list[dict[str, Any]],
    sizes: list[int],
    method: str,
) -> dict[str, Any]:
    if len(sizes) < 2:
        return {"available": False}
    previous, final = sizes[-2], sizes[-1]
    changes = {}
    for parameter in PARAMETER_NAMES:
        previous_row = next(
            row
            for row in parameter_rows
            if row["method"] == method
            and row["parameter"] == parameter
            and row["n_rows"] == previous
        )
        final_row = next(
            row
            for row in parameter_rows
            if row["method"] == method
            and row["parameter"] == parameter
            and row["n_rows"] == final
        )
        denominator = abs(float(final_row["std_over_prior"]))
        changes[parameter] = (
            abs(float(final_row["std_over_prior"]) - float(previous_row["std_over_prior"]))
            / denominator
            if denominator > 0.0
            else float("inf")
        )
    return {
        "available": True,
        "method": method,
        "previous_n": previous,
        "final_n": final,
        "relative_change_by_parameter": changes,
        "maximum_relative_change": max(changes.values()),
        "p0_relative_change": changes["P0"],
        "beta_relative_change": changes["beta"],
    }


def main() -> int:
    args = parse_args()
    if args.analysis_only and args.cache_only:
        raise ValueError("--analysis-only and --cache-only cannot be combined.")
    if args.random_rows <= 0 or args.chunk_rows <= 0:
        raise ValueError("Random and chunk row counts must be positive.")
    if not 0.0 <= args.fixed_shrinkage <= 1.0:
        raise ValueError("--fixed-shrinkage must be between zero and one.")
    if args.crossfit_folds < 2:
        raise ValueError("--crossfit-folds must be at least two.")
    for value, name in (
        (args.covariance_eigenvalue_floor, "covariance eigenvalue floor"),
        (args.fisher_eigenvalue_floor, "Fisher eigenvalue floor"),
        (args.fisher_rcond, "Fisher rcond"),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive.")

    prepared_dataset = args.prepared_dataset.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    fisher_analysis_dir = args.fisher_analysis_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fisher_analysis_dir / "covariance_convergence"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    paths = cache_paths(cache_dir)
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    should_build = args.rebuild_cache or not cache_complete(paths)
    if should_build:
        if args.analysis_only:
            raise FileNotFoundError(
                "--analysis-only was requested but the residual cache is incomplete."
            )
        cache_started = time.perf_counter()
        build_residual_cache(
            prepared_dataset=prepared_dataset,
            raw_dir=raw_dir,
            cache_dir=cache_dir,
            sizes=args.sizes,
            random_rows=args.random_rows,
            chunk_rows=args.chunk_rows,
            bin_weighting=args.bin_weighting,
            seed=args.seed,
        )
        timings["cache_build_seconds"] = time.perf_counter() - cache_started
    else:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        mismatches = []
        if metadata.get("bin_weighting") != args.bin_weighting:
            mismatches.append("bin weighting")
        if int(metadata.get("random_rows", -1)) != args.random_rows:
            mismatches.append("random row count")
        if int(metadata.get("seed", -1)) != args.seed:
            mismatches.append("seed")
        if max(args.sizes) > int(metadata.get("max_local_rows", -1)):
            mismatches.append("maximum local row count")
        if mismatches:
            raise ValueError(
                "Existing residual cache does not match: "
                + ", ".join(mismatches)
                + ". Rerun with --rebuild-cache."
            )
        print(f"Reusing residual cache: {cache_dir}", flush=True)
        timings["cache_build_seconds"] = 0.0

    if args.cache_only:
        timings["total_seconds"] = time.perf_counter() - total_started
        write_json(output_dir / "covariance_convergence_timings.json", timings)
        print(f"Residual cache complete: {cache_dir}")
        return 0

    cache = load_cache(cache_dir, args.sizes)
    derivative_path = fisher_analysis_dir / "derivatives_richardson.npy"
    if not derivative_path.is_file():
        raise FileNotFoundError(
            f"Richardson derivatives not found: {derivative_path}. "
            "Complete the derivative stage first or pass --fisher-analysis-dir."
        )
    derivatives = np.asarray(np.load(derivative_path), dtype=np.float64)
    expected_shape = (len(PARAMETER_NAMES), cache["local_residuals"].shape[1])
    if derivatives.shape != expected_shape:
        raise ValueError(
            f"Derivative shape {derivatives.shape} does not match {expected_shape}."
        )

    analysis_started = time.perf_counter()
    metrics, parameter_rows = evaluate_candidates(
        cache=cache,
        sizes=args.sizes,
        derivatives=derivatives,
        output_dir=output_dir,
        fixed_shrinkage=args.fixed_shrinkage,
        detrend_degree=args.detrend_degree,
        crossfit_folds=args.crossfit_folds,
        covariance_floor=args.covariance_eigenvalue_floor,
        fisher_floor=args.fisher_eigenvalue_floor,
        fisher_rcond=args.fisher_rcond,
        apply_hartlap=not args.no_hartlap,
        seed=args.seed,
    )
    timings["candidate_analysis_seconds"] = time.perf_counter() - analysis_started
    write_csv(output_dir / "covariance_convergence_metrics.csv", metrics)
    write_csv(output_dir / "fisher_parameter_convergence.csv", parameter_rows)

    preferred_source = output_dir / f"N{max(args.sizes)}"
    preferred_products = {
        "preferred_covariance.npy": "detrended_oas_covariance.npy",
        "preferred_precision.npy": "detrended_oas_precision.npy",
        "preferred_fisher_matrix_normalized.npy": (
            "detrended_oas_fisher_matrix_normalized.npy"
        ),
        "preferred_fisher_covariance_normalized.npy": (
            "detrended_oas_fisher_covariance_normalized.npy"
        ),
        "preferred_fisher_covariance_theta.npy": (
            "detrended_oas_fisher_covariance_theta.npy"
        ),
    }
    for output_name, source_name in preferred_products.items():
        np.save(output_dir / output_name, np.load(preferred_source / source_name))
    p0_beta_indices = np.asarray(
        [PARAMETER_NAMES.index("P0"), PARAMETER_NAMES.index("beta")],
        dtype=np.int64,
    )
    preferred_theta_covariance = np.load(
        output_dir / "preferred_fisher_covariance_theta.npy"
    )
    np.save(
        output_dir / "preferred_p0_beta_covariance_theta.npy",
        preferred_theta_covariance[
            np.ix_(p0_beta_indices, p0_beta_indices)
        ],

    )
    paper_style()
    plot_covariance_summary(
        metrics,
        output_dir / "covariance_and_fisher_diagnostics_vs_rows.jpg",
        args.dpi,
    )
    plot_local_diagnostics(
        metrics,
        output_dir / "local_covariance_diagnostics_vs_rows.jpg",
        args.dpi,
    )
    plot_parameter_convergence(
        parameter_rows,
        output_dir / "all_parameter_constraints_vs_covariance_rows.jpg",
        args.dpi,
    )
    plot_p0_beta_convergence(
        parameter_rows,
        output_dir / "p0_beta_constraints_vs_covariance_rows.jpg",
        args.dpi,
    )
    plot_oas_shrinkage(
        metrics,
        output_dir / "oas_shrinkage_vs_covariance_rows.jpg",
        args.dpi,
    )
    plot_p0_beta_ellipses(output_dir, max(args.sizes), args.dpi)

    prior_width = np.asarray(cache["prior_high"]) - np.asarray(cache["prior_low"])
    floor_rows = fisher_floor_sensitivity(
        output_dir=output_dir,
        n_rows=max(args.sizes),
        method="detrended_oas",
        floor_grid=args.fisher_floor_grid,
        derivatives=derivatives,
        prior_width=prior_width,
        covariance_floor=args.covariance_eigenvalue_floor,
        apply_hartlap=not args.no_hartlap,
        dpi=args.dpi,
    )
    write_csv(output_dir / "fisher_floor_sensitivity.csv", floor_rows)

    preferred_metrics = next(
        row
        for row in metrics
        if row["method"] == "detrended_oas" and row["n_rows"] == max(args.sizes)
    )
    stability = convergence_stability(
        parameter_rows,
        args.sizes,
        "detrended_oas",
    )
    floor_values_by_parameter: dict[str, list[float]] = {}
    for parameter in PARAMETER_NAMES:
        floor_values_by_parameter[parameter] = [
            float(row["std_over_prior"])
            for row in floor_rows
            if row["parameter"] == parameter
        ]
    floor_relative_ranges = {
        parameter: (
            (max(values) - min(values)) / min(values)
            if values and min(values) > 0.0
            else float("inf")
        )
        for parameter, values in floor_values_by_parameter.items()
    }

    summary = {
        "prepared_dataset": prepared_dataset,
        "raw_dir": raw_dir,
        "fisher_analysis_dir": fisher_analysis_dir,
        "output_dir": output_dir,
        "sizes": args.sizes,
        "preferred_method": "detrended_oas",
        "preferred_max_n_metrics": preferred_metrics,
        "constraint_convergence": stability,
        "fisher_floor_relative_range_by_parameter": floor_relative_ranges,
        "configuration": {
            "random_rows": args.random_rows,
            "bin_weighting": args.bin_weighting,
            "fixed_shrinkage": args.fixed_shrinkage,
            "detrend_degree": args.detrend_degree,
            "crossfit_folds": args.crossfit_folds,
            "covariance_eigenvalue_floor": args.covariance_eigenvalue_floor,
            "fisher_eigenvalue_floor": args.fisher_eigenvalue_floor,
            "fisher_rcond": args.fisher_rcond,
            "hartlap_applied": not args.no_hartlap,
            "oas_hartlap_caveat": (
                "Hartlap is exact for an unshrunk sample covariance, not OAS; "
                "at N=20000 and p=40 its numerical effect is about 0.2%."
            ),
        },
        "acceptance_checks": {
            "local_residual_r2_below_0p05": preferred_metrics["residual_r2_max"] < 0.05,
            "p0_beta_last_step_below_10pct": (
                stability.get("p0_relative_change", float("inf")) < 0.10
                and stability.get("beta_relative_change", float("inf")) < 0.10
            ),
            "all_constraints_last_step_below_10pct": stability.get(
                "maximum_relative_change", float("inf")
            ) < 0.10,
            "all_fisher_modes_identified_at_rcond": preferred_metrics[
                "fisher_identified_rank"
            ] == len(PARAMETER_NAMES),
            "p0_beta_floor_sensitivity_below_10pct": (
                floor_relative_ranges["P0"] < 0.10
                and floor_relative_ranges["beta"] < 0.10
            ),
        },
        "limitations": [
            "One noisy realization exists per theta, so detrending assumes a smooth conditional residual mean.",
            "Fixed-theta repeated noise realizations remain the preferred covariance source.",
            "A uniform bounded prior has no local Gaussian Fisher curvature; weak Fisher modes must not be hidden by flooring.",
        ],
    }
    timings["total_seconds"] = time.perf_counter() - total_started
    write_json(output_dir / "covariance_convergence_summary.json", summary)
    write_json(output_dir / "covariance_convergence_timings.json", timings)

    print(f"Covariance convergence analysis complete: {output_dir}")
    print("Preferred diagnostic method: detrended_oas")
    print("Inspect covariance_convergence_summary.json before using Fisher constraints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
