#!/usr/bin/env python3
"""Fisher forecast for masked SO baseline-noise deproj0 and comparison to SBI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_battaglia12_so_getdist import (
    LABEL_BY_NAME,
    apply_x_transform,
    available_n_values,
    find_run_dir,
    load_posterior,
    load_x_transform,
    plot_getdist,
    sample_posterior_at_x,
)


PARAMETER_NAMES = [
    "P0",
    "xc",
    "beta",
    "alpha_m_P0",
    "alpha_m_xc",
    "alpha_m_beta",
    "alpha_z_P0",
    "alpha_z_xc",
    "alpha_z_beta",
]
BATTAGLIA12 = np.array(
    [18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415],
    dtype=np.float64,
)

DEFAULT_PREPARED_DATASET = Path(
    "/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/"
    "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/"
    "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz"
)
DEFAULT_RAW_DIR = Path("/lustre/work/kristero10/adrian_dataset")
DEFAULT_FISHER_ROOT = Path("/lustre/work/kristero10/adrian_fisher_baseline_deproj0")
DEFAULT_SBI_RUN_ROOT = Path(
    "/home/kristero10/HalfDome_kSZ/SBI_analysis/outputs/cluster_outputs/"
    "SBI_Adrian_SO_dataset_size_ell80_7979_dataset_row_metadata_verified_asinh"
)
DEFAULT_CASE = "masked_baseline_noise_cross_deproj0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build central finite-difference Fisher constraints, estimate the "
            "SO covariance from aligned noise-clean residuals, and compare with "
            "the largest saved baseline-deproj0 NPE."
        )
    )
    parser.add_argument("--prepared-dataset", type=Path, default=DEFAULT_PREPARED_DATASET)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
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
    parser.add_argument("--covariance-rows", type=int, default=20_000)
    parser.add_argument("--covariance-random-rows", type=int, default=20_000)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument(
        "--bin-weighting",
        choices=("2ell_plus_1", "uniform", "ell"),
        default="2ell_plus_1",
    )
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument(
        "--no-hartlap",
        action="store_true",
        help="Do not apply the finite-covariance-sample Hartlap factor.",
    )
    parser.add_argument("--eigenvalue-floor", type=float, default=1.0e-10)
    parser.add_argument("--fisher-samples", type=int, default=100_000)
    parser.add_argument("--sbi-samples", type=int, default=100_000)
    parser.add_argument("--gibbs-burn-in", type=int, default=5_000)
    parser.add_argument("--gibbs-thin", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for row in rows:
        row["row_1based"] = int(row["row_1based"])
        row["parameter_index"] = int(row["parameter_index"])
        row["sign"] = int(row["sign"])
        row["step_fraction"] = float(row["step_fraction"])
        row["step_absolute"] = float(row["step_absolute"])
    return rows


def bin_weights(ell: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "uniform":
        return np.ones_like(ell, dtype=np.float64)
    if weighting == "ell":
        return np.asarray(ell, dtype=np.float64)
    if weighting == "2ell_plus_1":
        return 2.0 * np.asarray(ell, dtype=np.float64) + 1.0
    raise ValueError(f"Unsupported bin weighting: {weighting}")


def make_bin_plan(
    ell: np.ndarray,
    bin_ell_min: np.ndarray,
    bin_ell_max: np.ndarray,
    weighting: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    dl_factor = ell * (ell + 1.0) / (2.0 * np.pi)
    plan: list[tuple[np.ndarray, np.ndarray]] = []
    for low, high in zip(bin_ell_min, bin_ell_max):
        indices = np.flatnonzero((ell >= float(low)) & (ell <= float(high)))
        if indices.size == 0:
            raise ValueError(f"No multipoles in requested bin {low}-{high}.")
        weights = bin_weights(ell[indices], weighting)
        weights = dl_factor[indices] * weights / np.sum(weights)
        plan.append((indices, np.asarray(weights, dtype=np.float64)))
    return plan


def align_ell_columns(values: np.ndarray, ell: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] == ell.size:
        return values
    ell_index = np.rint(ell).astype(np.int64)
    if np.allclose(ell, ell_index) and values.shape[-1] > int(ell_index.max()):
        return values[..., ell_index]
    raise ValueError(
        f"Cannot align spectrum length {values.shape[-1]} with ell length {ell.size} "
        f"and range {ell.min()}..{ell.max()}."
    )


def bin_spectra(values: np.ndarray, plan: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"Expected one or two-dimensional spectra, found {values.shape}.")
    output = np.empty((values.shape[0], len(plan)), dtype=np.float64)
    for bin_index, (indices, weights) in enumerate(plan):
        output[:, bin_index] = values[:, indices] @ weights
    return output[0] if one_dimensional else output


def find_clean_spectrum(fisher_root: Path, row: dict[str, Any]) -> Path:
    row_dir = fisher_root / "variations" / (
        f"row{int(row['row_1based']):03d}_{row['label']}"
    )
    matches = sorted(row_dir.glob("*masked_no_noise_cl*.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one masked no-noise C_ell in {row_dir}; found {matches}."
        )
    return matches[0]


def load_derivative_profiles(
    fisher_root: Path,
    manifest: list[dict[str, Any]],
    ell: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    param_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    binned: dict[int, np.ndarray] = {}
    source_rows: list[dict[str, Any]] = []
    for row in manifest:
        path = find_clean_spectrum(fisher_root, row)
        profile = np.asarray(np.load(path), dtype=np.float64).squeeze()
        if profile.ndim != 1:
            raise ValueError(f"Expected one-dimensional C_ell in {path}, got {profile.shape}.")
        profile = align_ell_columns(profile, ell)
        binned[row["row_1based"]] = bin_spectra(profile, plan)
        source_rows.append(
            {
                "row_1based": row["row_1based"],
                "label": row["label"],
                "path": str(path),
            }
        )

    fiducial_rows = [row for row in manifest if row["parameter"] == "fiducial"]
    if len(fiducial_rows) != 1:
        raise ValueError(f"Expected one fiducial row, found {len(fiducial_rows)}.")
    fiducial = binned[fiducial_rows[0]["row_1based"]]

    derivatives_small = np.empty((len(param_names), fiducial.size), dtype=np.float64)
    derivatives_large = np.empty_like(derivatives_small)
    derivatives_richardson = np.empty_like(derivatives_small)
    stability = np.empty(len(param_names), dtype=np.float64)
    details: list[dict[str, Any]] = []

    for parameter_index, parameter in enumerate(param_names):
        rows = [row for row in manifest if row["parameter"] == parameter]
        fractions = sorted({row["step_fraction"] for row in rows})
        if len(fractions) != 2:
            raise ValueError(
                f"{parameter} needs exactly two step fractions; found {fractions}."
            )
        small_fraction, large_fraction = fractions
        derivatives: dict[float, np.ndarray] = {}
        for fraction in fractions:
            minus = [
                row for row in rows
                if row["step_fraction"] == fraction and row["sign"] == -1
            ]
            plus = [
                row for row in rows
                if row["step_fraction"] == fraction and row["sign"] == 1
            ]
            if len(minus) != 1 or len(plus) != 1:
                raise ValueError(
                    f"{parameter}, fraction={fraction} needs one plus and one minus row."
                )
            step = float(plus[0]["step_absolute"])
            derivatives[fraction] = (
                binned[plus[0]["row_1based"]] - binned[minus[0]["row_1based"]]
            ) / (2.0 * step)

        small = derivatives[small_fraction]
        large = derivatives[large_fraction]
        step_ratio = large_fraction / small_fraction
        if step_ratio <= 1.0:
            raise ValueError(f"Invalid step ratio for {parameter}: {step_ratio}")
        richardson = (
            step_ratio**2 * small - large
        ) / (step_ratio**2 - 1.0)
        denominator = max(float(np.linalg.norm(richardson)), np.finfo(float).tiny)

        derivatives_small[parameter_index] = small
        derivatives_large[parameter_index] = large
        derivatives_richardson[parameter_index] = richardson
        stability[parameter_index] = np.linalg.norm(small - large) / denominator
        details.append(
            {
                "parameter": parameter,
                "small_fraction": small_fraction,
                "large_fraction": large_fraction,
                "step_ratio": step_ratio,
                "relative_small_large_difference": stability[parameter_index],
            }
        )

    return (
        fiducial,
        derivatives_small,
        derivatives_large,
        derivatives_richardson,
        details,
    )


def theta_array_diagnostics(
    candidate: np.ndarray,
    reference: np.ndarray,
    param_names: list[str],
) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "candidate_shape": list(candidate.shape),
        "reference_shape": list(reference.shape),
        "float32_representation_equal": False,
    }
    if candidate.shape != reference.shape:
        return diagnostics

    candidate32 = candidate.astype(np.float32)
    reference32 = reference.astype(np.float32)
    finite = np.isfinite(candidate) & np.isfinite(reference)
    diagnostics["all_finite"] = bool(np.all(finite))
    diagnostics["float32_representation_equal"] = bool(
        np.all(finite) and np.array_equal(candidate32, reference32)
    )

    difference = np.abs(candidate - reference)
    difference = np.where(np.isfinite(difference), difference, np.inf)
    flat_index = int(np.argmax(difference))
    row_index, parameter_index = np.unravel_index(
        flat_index,
        difference.shape,
    )
    diagnostics.update(
        {
            "max_absolute_difference": float(difference[row_index, parameter_index]),
            "worst_row_index": int(row_index),
            "worst_parameter_index": int(parameter_index),
            "worst_parameter": param_names[parameter_index],
            "candidate_value": float(candidate[row_index, parameter_index]),
            "reference_value": float(reference[row_index, parameter_index]),
            "candidate_float32": float(candidate32[row_index, parameter_index]),
            "reference_float32": float(reference32[row_index, parameter_index]),
        }
    )
    return diagnostics


def validate_raw_alignment(
    prepared_dataset: Path,
    raw_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    clean_cl_path = raw_dir / "sbi_masked_no_noise_cl.npy"
    noisy_cl_path = raw_dir / "sbi_masked_baseline_noise_cross_deproj0_cl.npy"
    clean_metadata_path = raw_dir / "sbi_masked_no_noise.npz"
    noisy_metadata_path = raw_dir / "sbi_masked_baseline_noise_cross_deproj0.npz"

    for path in (
        clean_cl_path,
        noisy_cl_path,
        clean_metadata_path,
        noisy_metadata_path,
        prepared_dataset,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    clean_cl = np.load(clean_cl_path, mmap_mode="r")
    noisy_cl = np.load(noisy_cl_path, mmap_mode="r")
    if clean_cl.shape != noisy_cl.shape or clean_cl.ndim != 2:
        raise ValueError(
            f"Raw C_ell arrays are not aligned 2D arrays: "
            f"clean={clean_cl.shape}, noisy={noisy_cl.shape}."
        )

    with np.load(noisy_metadata_path, allow_pickle=True) as data:
        theta = np.asarray(data["theta"], dtype=np.float64)
        noisy_global = np.asarray(data["sobol_global_row"], dtype=np.int64)
        raw_param_names = [
            str(value) for value in data[
                "theta_columns" if "theta_columns" in data.files else "param_names"
            ]
        ]
    with np.load(clean_metadata_path, allow_pickle=True) as data:
        clean_global = np.asarray(data["sobol_global_row"], dtype=np.int64)
        clean_theta = np.asarray(data["theta"], dtype=np.float64)

    with np.load(prepared_dataset, allow_pickle=True) as data:
        prepared_global = np.asarray(data["sobol_global_row"], dtype=np.int64)
        prepared_theta = np.asarray(data["theta"], dtype=np.float64)
        param_names = [str(value) for value in data["param_names"]]
        prior_low = np.asarray(data["prior_low"], dtype=np.float64)
        prior_high = np.asarray(data["prior_high"], dtype=np.float64)
        ell = np.asarray(data["ell_unbinned"], dtype=np.float64)
        ell_binned = np.asarray(data["ell"], dtype=np.float64)
        bin_ell_min = np.asarray(data["bin_ell_min"], dtype=np.float64)
        bin_ell_max = np.asarray(data["bin_ell_max"], dtype=np.float64)

    if raw_param_names != param_names or param_names != PARAMETER_NAMES:
        raise ValueError(
            f"Parameter order mismatch: raw={raw_param_names}, prepared={param_names}."
        )
    n_rows = clean_cl.shape[0]
    for name, array in (
        ("theta", theta),
        ("clean theta", clean_theta),
        ("noisy sobol_global_row", noisy_global),
        ("clean sobol_global_row", clean_global),
        ("prepared theta", prepared_theta),
        ("prepared sobol_global_row", prepared_global),
    ):
        if array.shape[0] != n_rows:
            raise ValueError(f"{name} has {array.shape[0]} rows; C_ell has {n_rows}.")

    theta_diagnostics = {
        "clean_noisy": theta_array_diagnostics(
            clean_theta,
            theta,
            param_names,
        ),
        "prepared_raw": theta_array_diagnostics(
            prepared_theta,
            theta,
            param_names,
        ),
    }
    checks = {
        "clean_noisy_sobol_global_row_equal": bool(
            np.array_equal(clean_global, noisy_global)
        ),
        "prepared_raw_sobol_global_row_equal": bool(
            np.array_equal(prepared_global, noisy_global)
        ),
        "clean_noisy_theta_equal": bool(
            theta_diagnostics["clean_noisy"]["float32_representation_equal"]
        ),
        "prepared_raw_theta_equal": bool(
            theta_diagnostics["prepared_raw"]["float32_representation_equal"]
        ),
        "theta_comparison": "exact after conversion to float32",
        "theta_diagnostics": theta_diagnostics,
        "n_rows": int(n_rows),
        "mapping_prefix": noisy_global[:10].tolist(),
        "clean_cl_path": str(clean_cl_path),
        "noisy_cl_path": str(noisy_cl_path),
        "clean_metadata_path": str(clean_metadata_path),
        "noisy_metadata_path": str(noisy_metadata_path),
    }
    failed = [
        name
        for name in (
            "clean_noisy_sobol_global_row_equal",
            "prepared_raw_sobol_global_row_equal",
            "clean_noisy_theta_equal",
            "prepared_raw_theta_equal",
        )
        if not checks[name]
    ]
    if failed:
        details = []
        diagnostic_key_by_check = {
            "clean_noisy_theta_equal": "clean_noisy",
            "prepared_raw_theta_equal": "prepared_raw",
        }
        for name in failed:
            diagnostic_key = diagnostic_key_by_check.get(name)
            if diagnostic_key is None:
                details.append(name)
                continue
            diagnostic = theta_diagnostics[diagnostic_key]
            row_index = diagnostic.get("worst_row_index")
            global_row = (
                int(noisy_global[row_index])
                if row_index is not None and row_index < noisy_global.size
                else None
            )
            details.append(
                f"{name} "
                f"(max_abs={diagnostic.get('max_absolute_difference')}, "
                f"array_row={row_index}, sobol_global_row={global_row}, "
                f"parameter={diagnostic.get('worst_parameter')}, "
                f"candidate={diagnostic.get('candidate_value')}, "
                f"raw={diagnostic.get('reference_value')}, "
                f"candidate_float32={diagnostic.get('candidate_float32')}, "
                f"raw_float32={diagnostic.get('reference_float32')})"
            )
        raise ValueError(
            "Row-alignment validation failed, so noisy-clean subtraction is unsafe: "
            + "; ".join(details)
        )

    metadata = {
        **checks,
        "param_names": param_names,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "ell": ell,
        "ell_binned": ell_binned,
        "bin_ell_min": bin_ell_min,
        "bin_ell_max": bin_ell_max,
    }
    return clean_cl, noisy_cl, theta, noisy_global, metadata


def nearest_parameter_indices(
    theta: np.ndarray,
    center: np.ndarray,
    prior_width: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0 or count > theta.shape[0]:
        raise ValueError(f"Requested covariance rows {count}; available={theta.shape[0]}.")
    normalized = (theta - center) / prior_width
    distance_squared = np.einsum("ij,ij->i", normalized, normalized)
    if count == theta.shape[0]:
        indices = np.arange(theta.shape[0], dtype=np.int64)
    else:
        indices = np.argpartition(distance_squared, count - 1)[:count]
    indices = np.sort(indices.astype(np.int64))
    return indices, distance_squared[indices]


def binned_residuals(
    noisy_cl: np.ndarray,
    clean_cl: np.ndarray,
    row_indices: np.ndarray,
    ell: np.ndarray,
    plan: list[tuple[np.ndarray, np.ndarray]],
    chunk_rows: int,
) -> np.ndarray:
    output = np.empty((row_indices.size, len(plan)), dtype=np.float64)
    for start in range(0, row_indices.size, chunk_rows):
        stop = min(start + chunk_rows, row_indices.size)
        indices = row_indices[start:stop]
        noisy = align_ell_columns(np.asarray(noisy_cl[indices], dtype=np.float64), ell)
        clean = align_ell_columns(np.asarray(clean_cl[indices], dtype=np.float64), ell)
        output[start:stop] = bin_spectra(noisy - clean, plan)
        print(f"  covariance residuals: {stop}/{row_indices.size}", flush=True)
    if not np.all(np.isfinite(output)):
        raise ValueError("Non-finite values found in binned noisy-clean residuals.")
    return output


def linear_dependence_diagnostics(
    theta: np.ndarray,
    residuals: np.ndarray,
    center: np.ndarray,
    prior_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = (np.asarray(theta, dtype=np.float64) - center) / prior_width
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = residuals - residuals.mean(axis=0, keepdims=True)

    design = np.column_stack([np.ones(x.shape[0]), x_centered])
    coefficients, _, _, _ = np.linalg.lstsq(design, residuals, rcond=None)
    predicted = design @ coefficients
    sse = np.sum((residuals - predicted) ** 2, axis=0)
    sst = np.sum(y_centered**2, axis=0)
    r_squared = np.where(sst > 0.0, 1.0 - sse / sst, 0.0)

    numerator = x_centered.T @ y_centered
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=0)[:, None]
        * np.sum(y_centered**2, axis=0)[None, :]
    )
    correlation = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return r_squared, correlation


def regularized_inverse(
    matrix: np.ndarray,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(np.max(eigenvalues))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Matrix is not positive semidefinite with a positive scale.")
    floor = max(largest * eigenvalue_floor, np.finfo(float).tiny)
    clipped = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / clipped)) @ eigenvectors.T
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    diagnostics = {
        "smallest_eigenvalue": float(eigenvalues[0]),
        "largest_eigenvalue": largest,
        "eigenvalue_floor_absolute": float(floor),
        "condition_before_floor": (
            float(largest / eigenvalues[0]) if eigenvalues[0] > 0 else math.inf
        ),
        "condition_after_floor": float(clipped[-1] / clipped[0]),
        "n_eigenvalues_floored": int(np.sum(eigenvalues < floor)),
    }
    return inverse, diagnostics, repaired


def draw_truncated_standard_normal(
    a: float,
    b: float,
    rng: np.random.Generator,
    standard_normal: Any,
) -> float:
    """Draw N(0, 1) restricted to [a, b], including numerically remote tails."""
    cdf_a = float(standard_normal.cdf(a))
    cdf_b = float(standard_normal.cdf(b))
    interval = cdf_b - cdf_a
    epsilon = np.finfo(float).eps
    if interval > 1.0e-14:
        quantile = cdf_a + rng.random() * interval
        quantile = min(max(quantile, epsilon), 1.0 - epsilon)
        return float(standard_normal.inv_cdf(quantile))

    if a > 0.0:
        rate = 0.5 * (a + math.sqrt(a * a + 4.0))
        for _ in range(100_000):
            value = a + rng.exponential(1.0 / rate)
            if value <= b and rng.random() <= math.exp(
                -0.5 * (value - rate) ** 2
            ):
                return float(value)
        raise RuntimeError(f"Could not sample truncated normal tail [{a}, {b}].")

    if b < 0.0:
        return -draw_truncated_standard_normal(
            -b,
            -a,
            rng,
            standard_normal,
        )

    mode = min(max(0.0, a), b)
    maximum_log_density = -0.5 * mode * mode
    for _ in range(100_000):
        value = rng.uniform(a, b)
        if rng.random() <= math.exp(
            -0.5 * value * value - maximum_log_density
        ):
            return float(value)
    raise RuntimeError(f"Could not sample truncated normal interval [{a}, {b}].")


def truncated_gaussian_gibbs(
    mean: np.ndarray,
    covariance: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    n_samples: int,
    burn_in: int,
    thin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    from statistics import NormalDist

    standard_normal = NormalDist()
    dimension = mean.size
    current = np.minimum(
        np.maximum(mean.copy(), low + 1.0e-12),
        high - 1.0e-12,
    )
    conditionals: list[tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(dimension):
        other = np.asarray(
            [j for j in range(dimension) if j != i],
            dtype=np.int64,
        )
        cov_other = covariance[np.ix_(other, other)]
        beta = covariance[i, other] @ np.linalg.pinv(
            cov_other,
            rcond=1.0e-12,
        )
        variance = covariance[i, i] - beta @ covariance[other, i]
        variance = max(
            float(variance),
            np.finfo(float).eps * covariance[i, i],
        )
        conditionals.append((other, beta, math.sqrt(variance)))

    total_iterations = burn_in + n_samples * thin
    samples = np.empty((n_samples, dimension), dtype=np.float64)
    stored = 0
    for iteration in range(total_iterations):
        for i, (other, beta, standard_deviation) in enumerate(conditionals):
            conditional_mean = mean[i] + beta @ (
                current[other] - mean[other]
            )
            a = (low[i] - conditional_mean) / standard_deviation
            b = (high[i] - conditional_mean) / standard_deviation
            standardized_value = draw_truncated_standard_normal(
                float(a),
                float(b),
                rng,
                standard_normal,
            )
            current[i] = (
                conditional_mean + standard_deviation * standardized_value
            )
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            samples[stored] = current
            stored += 1
    return samples


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = sigma[:, None] * sigma[None, :]
    return np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )


def plot_derivatives(
    ell_binned: np.ndarray,
    derivative_small_q: np.ndarray,
    derivative_large_q: np.ndarray,
    derivative_richardson_q: np.ndarray,
    param_names: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18.0 / 2.54, 15.5 / 2.54),
        sharex=True,
    )
    for i, (axis, parameter) in enumerate(zip(axes.flat, param_names)):
        axis.plot(
            ell_binned,
            derivative_large_q[i],
            color="#999999",
            lw=0.8,
            label="2%",
        )
        axis.plot(
            ell_binned,
            derivative_small_q[i],
            color="#1f77b4",
            lw=0.9,
            ls="--",
            label="1%",
        )
        axis.plot(
            ell_binned,
            derivative_richardson_q[i],
            color="#d62728",
            lw=1.0,
            label="Richardson",
        )
        axis.axhline(0.0, color="black", lw=0.45, alpha=0.5)
        axis.set_title(
            rf"$\partial D_\ell/\partial q_{{{i + 1}}}$: "
            rf"${LABEL_BY_NAME.get(parameter, parameter)}$",
            pad=2.0,
        )
        axis.grid(True, alpha=0.2, lw=0.4)
        if i >= 6:
            axis.set_xlabel(r"$\ell$")
        if i % 3 == 0:
            axis.set_ylabel(r"$\partial D_\ell/\partial q$")
    axes.flat[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_covariance_diagnostics(
    covariance: np.ndarray,
    local_random_sigma_ratio: np.ndarray,
    residual_r_squared: np.ndarray,
    ell_binned: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.bbox": "tight",
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18.0 / 2.54, 5.4 / 2.54),
    )
    image = axes[0].imshow(
        covariance_to_correlation(covariance),
        origin="lower",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    axes[0].set_xlabel(r"$D_\ell$ bin")
    axes[0].set_ylabel(r"$D_\ell$ bin")
    axes[0].set_title("Local residual correlation")
    figure.colorbar(image, ax=axes[0], pad=0.02, label="Correlation")

    axes[1].plot(
        ell_binned,
        local_random_sigma_ratio,
        marker="o",
        ms=2.5,
        lw=0.8,
        color="#1f77b4",
    )
    axes[1].axhline(1.0, color="black", lw=0.7, ls="--")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"$\sigma_{\rm local}/\sigma_{\rm random}$")
    axes[1].set_title("Covariance locality")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(
        ell_binned,
        residual_r_squared,
        marker="o",
        ms=2.5,
        lw=0.8,
        color="#d62728",
    )
    axes[2].set_xlabel(r"$\ell$")
    axes[2].set_ylabel(r"$R^2$")
    axes[2].set_title("Residual parameter dependence")
    axes[2].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def write_constraint_table(
    path: Path,
    param_names: list[str],
    truth: np.ndarray,
    prior_width: np.ndarray,
    sample_sets: list[dict[str, Any]],
) -> None:
    fields = [
        "method",
        "parameter",
        "truth",
        "mean",
        "std",
        "std_over_prior",
        "q16",
        "median",
        "q84",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sample_sets:
            samples = np.asarray(item["samples"], dtype=np.float64)
            for i, parameter in enumerate(param_names):
                q16, median, q84 = np.quantile(
                    samples[:, i],
                    [0.16, 0.5, 0.84],
                )
                standard_deviation = np.std(samples[:, i], ddof=1)
                writer.writerow(
                    {
                        "method": item["label"],
                        "parameter": parameter,
                        "truth": truth[i],
                        "mean": np.mean(samples[:, i]),
                        "std": standard_deviation,
                        "std_over_prior": standard_deviation / prior_width[i],
                        "q16": q16,
                        "median": median,
                        "q84": q84,
                    }
                )


def main() -> int:
    args = parse_args()
    prepared_dataset = args.prepared_dataset.expanduser().resolve()
    raw_dir = args.raw_dir.expanduser().resolve()
    fisher_root = args.fisher_root.expanduser().resolve()
    sbi_run_root = args.sbi_run_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else fisher_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (0.0 <= args.shrinkage <= 1.0):
        raise ValueError("--shrinkage must be between 0 and 1.")
    if args.chunk_rows <= 0 or args.gibbs_thin <= 0:
        raise ValueError("--chunk-rows and --gibbs-thin must be positive.")

    print("Validating raw row alignment...", flush=True)
    clean_cl, noisy_cl, theta, sobol_global_row, metadata = validate_raw_alignment(
        prepared_dataset,
        raw_dir,
    )
    param_names = list(metadata["param_names"])
    prior_low = np.asarray(metadata["prior_low"], dtype=np.float64)
    prior_high = np.asarray(metadata["prior_high"], dtype=np.float64)
    prior_width = prior_high - prior_low
    ell = np.asarray(metadata["ell"], dtype=np.float64)
    ell_binned = np.asarray(metadata["ell_binned"], dtype=np.float64)
    plan = make_bin_plan(
        ell,
        np.asarray(metadata["bin_ell_min"]),
        np.asarray(metadata["bin_ell_max"]),
        args.bin_weighting,
    )
    if len(plan) != ell_binned.size:
        raise ValueError(
            f"Bin-plan length {len(plan)} differs from stored ell length "
            f"{ell_binned.size}."
        )

    manifest_path = fisher_root / "fisher_variations_manifest.csv"
    manifest = read_manifest(manifest_path)
    print("Loading finite-difference spectra...", flush=True)
    (
        fiducial_dell,
        derivative_small,
        derivative_large,
        derivative_richardson,
        derivative_details,
    ) = load_derivative_profiles(
        fisher_root,
        manifest,
        ell,
        plan,
        param_names,
    )

    np.save(output_dir / "fiducial_battaglia12_dell.npy", fiducial_dell)
    np.save(output_dir / "derivatives_small_step.npy", derivative_small)
    np.save(output_dir / "derivatives_large_step.npy", derivative_large)
    np.save(output_dir / "derivatives_richardson.npy", derivative_richardson)

    derivative_small_q = derivative_small * prior_width[:, None]
    derivative_large_q = derivative_large * prior_width[:, None]
    derivative_richardson_q = derivative_richardson * prior_width[:, None]
    plot_derivatives(
        ell_binned,
        derivative_small_q,
        derivative_large_q,
        derivative_richardson_q,
        param_names,
        output_dir / "fisher_derivative_stability.jpg",
        args.dpi,
    )

    print("Selecting covariance rows nearest Battaglia12...", flush=True)
    local_indices, local_distance_squared = nearest_parameter_indices(
        theta,
        BATTAGLIA12,
        prior_width,
        args.covariance_rows,
    )
    rng = np.random.default_rng(args.seed)
    random_count = min(args.covariance_random_rows, theta.shape[0])
    random_indices = np.sort(
        rng.choice(
            theta.shape[0],
            size=random_count,
            replace=False,
        ).astype(np.int64)
    )
    np.save(output_dir / "covariance_local_indices.npy", local_indices)
    np.save(output_dir / "covariance_random_indices.npy", random_indices)
    np.save(
        output_dir / "covariance_local_sobol_global_row.npy",
        sobol_global_row[local_indices],
    )

    print("Binning local noisy-clean residuals...", flush=True)
    local_residuals = binned_residuals(
        noisy_cl,
        clean_cl,
        local_indices,
        ell,
        plan,
        args.chunk_rows,
    )
    print("Binning random noisy-clean residuals...", flush=True)
    random_residuals = binned_residuals(
        noisy_cl,
        clean_cl,
        random_indices,
        ell,
        plan,
        args.chunk_rows,
    )

    covariance_local_sample = np.cov(local_residuals, rowvar=False, ddof=1)
    covariance_random_sample = np.cov(random_residuals, rowvar=False, ddof=1)
    diagonal_target = np.diag(np.diag(covariance_local_sample))
    covariance = (
        (1.0 - args.shrinkage) * covariance_local_sample
        + args.shrinkage * diagonal_target
    )
    inverse_covariance, covariance_condition, covariance = regularized_inverse(
        covariance,
        args.eigenvalue_floor,
    )
    hartlap_factor = 1.0
    n_covariance, n_bins = local_residuals.shape
    if not args.no_hartlap:
        if n_covariance <= n_bins + 2:
            raise ValueError(
                f"Hartlap correction requires n_covariance > n_bins + 2; "
                f"found {n_covariance} and {n_bins}."
            )
        hartlap_factor = (
            n_covariance - n_bins - 2.0
        ) / (
            n_covariance - 1.0
        )
        inverse_covariance *= hartlap_factor

    residual_r_squared, residual_parameter_correlation = (
        linear_dependence_diagnostics(
            theta[random_indices],
            random_residuals,
            BATTAGLIA12,
            prior_width,
        )
    )
    local_random_sigma_ratio = np.sqrt(
        np.diag(covariance_local_sample)
        / np.diag(covariance_random_sample)
    )
    covariance_relative_difference = (
        np.linalg.norm(
            covariance_local_sample - covariance_random_sample,
            ord="fro",
        )
        / np.linalg.norm(covariance_random_sample, ord="fro")
    )

    np.save(output_dir / "covariance_local_sample.npy", covariance_local_sample)
    np.save(output_dir / "covariance_random_sample.npy", covariance_random_sample)
    np.save(output_dir / "covariance_shrunk.npy", covariance)
    np.save(output_dir / "inverse_covariance.npy", inverse_covariance)
    np.save(output_dir / "residual_r_squared.npy", residual_r_squared)
    np.save(
        output_dir / "residual_parameter_correlation.npy",
        residual_parameter_correlation,
    )
    plot_covariance_diagnostics(
        covariance,
        local_random_sigma_ratio,
        residual_r_squared,
        ell_binned,
        output_dir / "covariance_diagnostics.jpg",
        args.dpi,
    )

    if np.max(residual_r_squared) > 0.05:
        warnings.warn(
            "Residual-vs-parameter R^2 exceeds 0.05. The 512k-row residual "
            "covariance is not behaving like parameter-independent fixed-theta noise."
        )
    if np.any(
        (local_random_sigma_ratio < 0.8)
        | (local_random_sigma_ratio > 1.25)
    ):
        warnings.warn(
            "At least one local/random residual sigma ratio lies outside 0.8..1.25."
        )

    print("Computing Fisher matrix...", flush=True)
    fisher_q = (
        derivative_richardson_q
        @ inverse_covariance
        @ derivative_richardson_q.T
    )
    inverse_fisher_q, fisher_condition, fisher_q = regularized_inverse(
        fisher_q,
        args.eigenvalue_floor,
    )
    fisher_covariance_theta = (
        prior_width[:, None]
        * inverse_fisher_q
        * prior_width[None, :]
    )
    fisher_matrix_theta = (
        fisher_q
        / prior_width[:, None]
        / prior_width[None, :]
    )
    np.save(output_dir / "fisher_matrix_normalized.npy", fisher_q)
    np.save(output_dir / "fisher_covariance_normalized.npy", inverse_fisher_q)
    np.save(output_dir / "fisher_matrix_theta.npy", fisher_matrix_theta)
    np.save(
        output_dir / "fisher_covariance_theta.npy",
        fisher_covariance_theta,
    )

    fisher_untruncated = rng.multivariate_normal(
        BATTAGLIA12,
        fisher_covariance_theta,
        size=args.fisher_samples,
        check_valid="warn",
    )
    print("Sampling prior-truncated Fisher Gaussian...", flush=True)
    fisher_truncated = truncated_gaussian_gibbs(
        BATTAGLIA12,
        fisher_covariance_theta,
        prior_low,
        prior_high,
        args.fisher_samples,
        args.gibbs_burn_in,
        args.gibbs_thin,
        rng,
    )
    np.save(
        output_dir / "fisher_samples_untruncated.npy",
        fisher_untruncated,
    )
    np.save(
        output_dir / "fisher_samples_prior_truncated.npy",
        fisher_truncated,
    )

    available = available_n_values(sbi_run_root, args.case)
    if not available:
        raise FileNotFoundError(
            f"No completed NPE runs found for case={args.case} "
            f"under {sbi_run_root}."
        )
    n_train = args.npe_n_train if args.npe_n_train > 0 else available[-1]
    if n_train not in available:
        raise ValueError(
            f"NPE N={n_train} is unavailable for {args.case}; "
            f"available={available}."
        )
    run_dir = find_run_dir(sbi_run_root, args.case, n_train)
    print(f"Sampling saved NPE from {run_dir}...", flush=True)
    posterior = load_posterior(run_dir)
    transform = load_x_transform(run_dir)
    transformed_observation = apply_x_transform(fiducial_dell, transform)
    if transformed_observation.size != fiducial_dell.size:
        raise ValueError(
            "Saved NPE x transform changed the observation dimension."
        )
    sbi_samples = sample_posterior_at_x(
        posterior,
        transformed_observation,
        args.sbi_samples,
        args.device,
    )
    if sbi_samples.shape[1] < len(param_names):
        raise ValueError(
            f"NPE returned shape {sbi_samples.shape}; expected at least "
            f"{len(param_names)} parameters."
        )
    sbi_samples = sbi_samples[:, : len(param_names)]
    finite = np.all(np.isfinite(sbi_samples), axis=1)
    sbi_samples = sbi_samples[finite]
    if sbi_samples.shape[0] < 1000:
        raise ValueError(
            f"Only {sbi_samples.shape[0]} finite NPE samples remain."
        )
    np.save(output_dir / f"sbi_samples_N{n_train}.npy", sbi_samples)
    np.save(
        output_dir / "battaglia12_observation_transformed.npy",
        transformed_observation,
    )

    all_sample_sets = [
        {
            "label": "Fisher Gaussian",
            "samples": fisher_untruncated,
        },
        {
            "label": "Fisher + prior",
            "samples": fisher_truncated,
        },
        {
            "label": f"NPE N={n_train:,}",
            "samples": sbi_samples,
        },
    ]
    plot_getdist(
        all_sample_sets,
        param_names,
        BATTAGLIA12,
        output_dir / "fisher_untruncated_truncated_vs_sbi_corner.jpg",
        filled_last_only=True,
        dpi=args.dpi,
    )
    plot_getdist(
        [all_sample_sets[0], all_sample_sets[2]],
        param_names,
        BATTAGLIA12,
        output_dir / "fisher_untruncated_vs_sbi_corner.jpg",
        filled_last_only=True,
        dpi=args.dpi,
    )
    plot_getdist(
        [all_sample_sets[1], all_sample_sets[2]],
        param_names,
        BATTAGLIA12,
        output_dir / "fisher_prior_truncated_vs_sbi_corner.jpg",
        filled_last_only=True,
        dpi=args.dpi,
    )
    write_constraint_table(
        output_dir / "fisher_vs_sbi_constraints.csv",
        param_names,
        BATTAGLIA12,
        prior_width,
        all_sample_sets,
    )

    diagnostics = {
        "prepared_dataset": str(prepared_dataset),
        "raw_dir": str(raw_dir),
        "fisher_root": str(fisher_root),
        "output_dir": str(output_dir),
        "alignment": metadata,
        "bin_weighting": args.bin_weighting,
        "n_bins": len(plan),
        "covariance": {
            "estimator": (
                "masked_baseline_noise_cross_deproj0 minus row-matched "
                "masked_no_noise; nearest parameters to Battaglia12"
            ),
            "fixed_theta_repeats_available": False,
            "assumption": (
                "Residual covariance is locally representative at Battaglia12. "
                "This is an approximation, not a fixed-theta noise ensemble."
            ),
            "n_local_rows": int(local_indices.size),
            "n_random_rows": int(random_indices.size),
            "local_normalized_distance_min": float(
                np.sqrt(local_distance_squared.min())
            ),
            "local_normalized_distance_median": float(
                np.sqrt(np.median(local_distance_squared))
            ),
            "local_normalized_distance_max": float(
                np.sqrt(local_distance_squared.max())
            ),
            "shrinkage_to_diagonal": args.shrinkage,
            "hartlap_factor": hartlap_factor,
            "local_random_covariance_frobenius_relative_difference": float(
                covariance_relative_difference
            ),
            "local_random_sigma_ratio_min": float(
                np.min(local_random_sigma_ratio)
            ),
            "local_random_sigma_ratio_max": float(
                np.max(local_random_sigma_ratio)
            ),
            "residual_r_squared_max": float(
                np.max(residual_r_squared)
            ),
            "residual_parameter_correlation_abs_max": float(
                np.max(np.abs(residual_parameter_correlation))
            ),
            "conditioning": covariance_condition,
        },
        "derivatives": {
            "method": (
                "central 1% and 2% prior-width steps with "
                "Richardson extrapolation"
            ),
            "details": derivative_details,
        },
        "fisher": {
            "conditioning": fisher_condition,
            "fiducial": BATTAGLIA12,
            "prior_low": prior_low,
            "prior_high": prior_high,
            "n_untruncated_samples": int(fisher_untruncated.shape[0]),
            "n_truncated_samples": int(fisher_truncated.shape[0]),
        },
        "sbi": {
            "run_dir": str(run_dir),
            "n_train": n_train,
            "x_transform_mode": transform.get("mode", "none"),
            "x_transform_path": transform.get("path", ""),
            "n_samples": int(sbi_samples.shape[0]),
            "observation": (
                "clean masked Battaglia12 mean D_ell, "
                "binned identically to SBI"
            ),
        },
    }
    write_json(
        output_dir / "fisher_analysis_summary.json",
        diagnostics,
    )

    print(f"Fisher analysis complete: {output_dir}")
    print(
        "Important covariance caveat: no fixed-theta noise repeats were "
        "available; inspect covariance_diagnostics.jpg and "
        "fisher_analysis_summary.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
