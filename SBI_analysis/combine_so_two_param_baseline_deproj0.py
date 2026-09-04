#!/usr/bin/env python3
"""Combine two-parameter SO baseline-deproj0 spectra into an SBI dataset.

The row-to-parameter mapping comes from worker manifests, never filesystem scan
order. Linear C_ell is converted to linear D_ell without clipping or log
transformation. The prepared x array uses the established Delta-ell=200,
(2 ell + 1)-weighted bins.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FULL_PARAMETERS = [
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
TARGET_PARAMETERS = ["P0", "beta"]
PRODUCT = "masked_baseline_noise_cross_deproj0"


def parse_args() -> argparse.Namespace:
    root = Path(
        "/lustre/work/kristero10/adrian_two_param_so_baseline_deproj0/"
        "block_offset0_n16384"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sobol-csv",
        type=Path,
        default=root / "design" / "battaglia_sobol_P0_beta_16384.csv",
    )
    parser.add_argument(
        "--design-metadata",
        type=Path,
        default=None,
        help="Default: the Sobol CSV with suffix .npz.",
    )
    parser.add_argument("--manifest-dir", type=Path, default=root / "run_manifests")
    parser.add_argument("--output-dir", type=Path, default=root / "prepared")
    parser.add_argument(
        "--dataset-name",
        default=(
            "so_two_param_P0_beta_masked_baseline_noise_cross_deproj0_"
            "ell80_7979_sbi_run.npz"
        ),
    )
    parser.add_argument(
        "--unbinned-name",
        default=(
            "so_two_param_P0_beta_masked_baseline_noise_cross_deproj0_"
            "ell80_7979_unbinned_dell.npy"
        ),
    )
    parser.add_argument(
        "--unbinned-no-noise-name",
        default=(
            "so_two_param_P0_beta_masked_no_noise_"
            "ell80_7979_unbinned_dell.npy"
        ),
    )
    parser.add_argument("--ell-min", type=int, default=80)
    parser.add_argument("--ell-max", type=int, default=7979)
    parser.add_argument("--mask-seed", type=int, default=12345)
    parser.add_argument("--noise-seed-base", type=int, default=1_000_000)
    parser.add_argument("--sequence-offset", type=int, default=0)
    parser.add_argument("--test-last-n", type=int, default=1000)
    parser.add_argument(
        "--moped-local-n",
        type=int,
        default=2048,
        help=(
            "Nearest training rows around Battaglia12 used for clean derivatives "
            "and paired-noise covariance."
        ),
    )
    parser.add_argument(
        "--skip-moped",
        action="store_true",
        help="Save only the 40-bin vectors; default also saves two MOPED summaries.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Create a partial dataset for diagnostics instead of requiring all rows.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    return array.reshape(()).item()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_csv_theta(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Sobol CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        missing = [name for name in FULL_PARAMETERS if name not in header]
        if missing:
            raise ValueError(f"Sobol CSV is missing columns {missing}; header={header}")
        rows = [
            [float(row[name]) for name in FULL_PARAMETERS]
            for row in reader
        ]
    theta = np.asarray(rows, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != len(FULL_PARAMETERS):
        raise ValueError(f"Invalid Sobol theta shape: {theta.shape}")
    if not np.all(np.isfinite(theta)):
        raise ValueError("Sobol CSV contains non-finite parameter values")
    return theta


def load_design_metadata(path: Path, expected_rows: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Design metadata NPZ not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "theta",
            "theta_full",
            "param_names",
            "prior_low",
            "prior_high",
            "sobol_row",
            "noise_seed",
            "noise_seed_base",
            "sequence_offset",
            "sobol_sequence_index",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Design metadata is missing arrays: {sorted(missing)}")
        result = {name: np.asarray(data[name]) for name in data.files}

    if result["theta"].shape != (expected_rows, 2):
        raise ValueError(f"Unexpected target theta shape: {result['theta'].shape}")
    if result["theta_full"].shape != (expected_rows, 9):
        raise ValueError(f"Unexpected full theta shape: {result['theta_full'].shape}")
    if [str(value) for value in result["param_names"]] != TARGET_PARAMETERS:
        raise ValueError(
            f"Expected target parameter order {TARGET_PARAMETERS}; got {result['param_names']}"
        )
    return result


def load_success_records(
    manifest_dir: Path,
    expected_rows: int,
    mask_seed: int,
    noise_seed_base: int,
    sequence_offset: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[str]]]:
    manifests = sorted(manifest_dir.glob("chunk_*.csv"))
    if not manifests:
        raise FileNotFoundError(f"No chunk manifests found under {manifest_dir}")

    successes: dict[int, list[dict[str, Any]]] = {}
    failures: dict[int, list[str]] = {}
    for manifest in manifests:
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "sobol_row",
                "sobol_sequence_index",
                "mask_seed",
                "noise_seed",
                "status",
                "no_noise_output_path",
                "output_path",
                "row_log",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{manifest} is missing columns {sorted(missing)}")
            for record in reader:
                row = int(record["sobol_row"])
                if row < 1 or row > expected_rows:
                    raise ValueError(f"{manifest} contains out-of-range row {row}")
                expected_sequence_index = sequence_offset + row
                expected_noise_seed = noise_seed_base + expected_sequence_index
                if int(record["sobol_sequence_index"]) != expected_sequence_index:
                    raise ValueError(
                        f"Row {row} sequence index is "
                        f"{record['sobol_sequence_index']}, expected "
                        f"{expected_sequence_index}"
                    )
                if int(record["mask_seed"]) != mask_seed:
                    raise ValueError(
                        f"Row {row} mask seed is {record['mask_seed']}, expected {mask_seed}"
                    )
                if int(record["noise_seed"]) != expected_noise_seed:
                    raise ValueError(
                        f"Row {row} noise seed is {record['noise_seed']}, "
                        f"expected {expected_noise_seed}"
                    )

                record["_manifest"] = str(manifest)
                if record["status"] == "success":
                    noisy_output = Path(record["output_path"])
                    clean_output = Path(record["no_noise_output_path"])
                    missing_outputs = [
                        path
                        for path in (clean_output, noisy_output)
                        if not path.is_file() or path.stat().st_size == 0
                    ]
                    if missing_outputs:
                        failures.setdefault(row, []).append(
                            "recorded success but outputs are missing: "
                            + ", ".join(map(str, missing_outputs))
                        )
                    else:
                        successes.setdefault(row, []).append(record)
                else:
                    failures.setdefault(row, []).append(
                        f"{manifest}: status={record['status']}, log={record['row_log']}"
                    )

    chosen: dict[int, dict[str, Any]] = {}
    for row, records in successes.items():
        noisy_paths = {
            str(Path(record["output_path"]).resolve()) for record in records
        }
        clean_paths = {
            str(Path(record["no_noise_output_path"]).resolve()) for record in records
        }
        if len(noisy_paths) != 1 or len(clean_paths) != 1:
            raise ValueError(
                f"Row {row} has successful manifests pointing to different outputs: "
                f"clean={sorted(clean_paths)}, noisy={sorted(noisy_paths)}"
            )
        chosen[row] = records[-1]
    return chosen, failures

def source_ell_for_profile(size: int, ell_min: int, ell_max: int) -> np.ndarray:
    expected_selected = ell_max - ell_min + 1
    if size > ell_max:
        return np.arange(size, dtype=np.float64)
    if size == expected_selected:
        return np.arange(ell_min, ell_max + 1, dtype=np.float64)
    raise ValueError(
        f"Cannot infer ell for profile length {size}; expected {expected_selected} "
        f"(ell={ell_min}..{ell_max}) or at least {ell_max + 1} (starting at ell=0)."
    )


def select_cl(path: Path, ell_min: int, ell_max: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(np.load(path), dtype=np.float64).squeeze()
    if raw.ndim != 1:
        raise ValueError(f"{path} must contain one C_ell vector; shape={raw.shape}")
    source_ell = source_ell_for_profile(raw.size, ell_min, ell_max)
    selected = (source_ell >= ell_min) & (source_ell <= ell_max)
    ell = source_ell[selected]
    cl = raw[selected]
    expected = ell_max - ell_min + 1
    if ell.size != expected or cl.size != expected:
        raise ValueError(f"{path} produced {cl.size} selected multipoles, expected {expected}")
    if not np.all(np.isfinite(cl)):
        raise ValueError(f"{path} contains non-finite C_ell values")
    return ell, cl


def make_bins(ell: np.ndarray) -> dict[str, Any]:
    if ell[0] != 80 or ell[-1] != 7979 or ell.size != 7900:
        raise ValueError(
            "The established Delta-ell=200 contract requires ell=80..7979."
        )
    edges = np.r_[np.arange(80, 7881, 200), 7979].astype(np.int64)
    bin_min = edges[:-1].copy()
    bin_max = edges[1:].copy()
    bin_max[:-1] -= 1

    indices = []
    weights = []
    centers = []
    counts = []
    for low, high in zip(bin_min, bin_max):
        index = np.flatnonzero((ell >= low) & (ell <= high))
        weight = 2.0 * ell[index] + 1.0
        indices.append(index)
        weights.append(weight / weight.sum())
        centers.append(float(np.average(ell[index], weights=weight)))
        counts.append(int(index.size))

    return {
        "indices": indices,
        "weights": weights,
        "ell_binned": np.asarray(centers, dtype=np.float32),
        "bin_ell_min": bin_min.astype(np.float32),
        "bin_ell_max": bin_max.astype(np.float32),
        "bin_counts": np.asarray(counts, dtype=np.int64),
    }


def bin_dell(dell: np.ndarray, bins: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            np.dot(dell[index], weight)
            for index, weight in zip(bins["indices"], bins["weights"])
        ],
        dtype=np.float32,
    )


def oas_covariance_standardized(
    residuals: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Shrink correlations while retaining each measured bin variance."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(f"Expected at least three residual vectors; got {values.shape}")
    centered = values - values.mean(axis=0, keepdims=True)
    bin_scale = centered.std(axis=0, ddof=1)
    if np.any(~np.isfinite(bin_scale)) or np.any(bin_scale <= 0.0):
        raise ValueError("Every paired-noise bin must have positive finite scatter")

    standardized = centered / bin_scale[None, :]
    n_samples, n_features = standardized.shape
    empirical = standardized.T @ standardized / float(n_samples)
    mu = float(np.trace(empirical) / n_features)
    alpha = float(np.mean(empirical**2))
    denominator = (n_samples + 1.0) * (alpha - mu**2 / n_features)
    shrinkage = (
        1.0
        if denominator <= 0.0
        else min((alpha + mu**2) / denominator, 1.0)
    )
    shrunk = (1.0 - shrinkage) * empirical
    shrunk.flat[:: n_features + 1] += shrinkage * mu
    covariance = bin_scale[:, None] * shrunk * bin_scale[None, :]
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, float(shrinkage), bin_scale


def stable_inverse(
    matrix: np.ndarray,
    relative_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    symmetric = 0.5 * (
        np.asarray(matrix, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
        raise ValueError("Covariance eigendecomposition is not finite and positive")
    floor = max(float(eigenvalues[-1]) * relative_floor, 1.0e-300)
    floored = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors * (1.0 / floored)) @ eigenvectors.T
    condition = float(floored[-1] / floored[0])
    return 0.5 * (inverse + inverse.T), eigenvalues, condition


def build_moped_compression(
    x_noisy: np.ndarray,
    x_clean: np.ndarray,
    theta: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
    test_indices: np.ndarray,
    local_n_requested: int,
) -> dict[str, Any]:
    """Build local two-parameter MOPED summaries without using held-out rows."""
    x_noisy = np.asarray(x_noisy, dtype=np.float64)
    x_clean = np.asarray(x_clean, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    prior_low = np.asarray(prior_low, dtype=np.float64)
    prior_high = np.asarray(prior_high, dtype=np.float64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    if x_noisy.shape != x_clean.shape or x_noisy.ndim != 2:
        raise ValueError(
            f"MOPED requires matching 2D noisy/clean arrays; "
            f"got {x_noisy.shape} and {x_clean.shape}"
        )
    if theta.shape != (x_noisy.shape[0], 2):
        raise ValueError(f"MOPED expected theta shape ({x_noisy.shape[0]}, 2)")
    if local_n_requested < 8:
        raise ValueError("--moped-local-n must be at least 8")

    all_indices = np.arange(x_noisy.shape[0], dtype=np.int64)
    training_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    local_n = min(int(local_n_requested), training_indices.size)
    if local_n < 8:
        raise ValueError("At least eight non-held-out rows are required for MOPED")

    fiducial_theta = np.asarray([18.1, 4.35], dtype=np.float64)
    prior_width = prior_high - prior_low
    if np.any(prior_width <= 0.0):
        raise ValueError("MOPED prior widths must be positive")
    if np.any(fiducial_theta <= prior_low) or np.any(fiducial_theta >= prior_high):
        raise ValueError("Battaglia12 P0/beta must lie strictly inside the prior")

    normalized_delta = (
        theta[training_indices] - fiducial_theta[None, :]
    ) / prior_width[None, :]
    radius = np.linalg.norm(normalized_delta, axis=1)
    nearest_order = np.argsort(radius, kind="stable")[:local_n]
    local_indices = training_indices[nearest_order]
    local_delta = normalized_delta[nearest_order]
    local_radius = radius[nearest_order]

    # A local quadratic fit captures curvature while its linear coefficients
    # provide derivatives exactly at Battaglia12.
    design = np.column_stack(
        [
            np.ones(local_n),
            local_delta[:, 0],
            local_delta[:, 1],
            local_delta[:, 0] ** 2,
            local_delta[:, 0] * local_delta[:, 1],
            local_delta[:, 1] ** 2,
        ]
    )
    bandwidth = max(float(np.percentile(local_radius, 75.0)), 1.0e-12)
    regression_weight = np.exp(-0.5 * (local_radius / bandwidth) ** 2)
    sqrt_weight = np.sqrt(regression_weight)[:, None]
    coefficients, _, regression_rank, _ = np.linalg.lstsq(
        design * sqrt_weight,
        x_clean[local_indices] * sqrt_weight,
        rcond=None,
    )
    if regression_rank != design.shape[1]:
        raise ValueError(
            f"Local quadratic derivative fit is rank {regression_rank}, "
            f"expected {design.shape[1]}"
        )

    fiducial_mean = coefficients[0]
    derivatives = np.stack(
        [
            coefficients[1] / prior_width[0],
            coefficients[2] / prior_width[1],
        ],
        axis=0,
    )
    if not np.all(np.isfinite(derivatives)):
        raise ValueError("Local clean-spectrum derivatives are not finite")

    paired_noise_residuals = (
        x_noisy[local_indices] - x_clean[local_indices]
    )
    covariance, oas_shrinkage, bin_scale = oas_covariance_standardized(
        paired_noise_residuals
    )
    precision, covariance_eigenvalues, covariance_condition = stable_inverse(
        covariance
    )

    moped_weights = np.empty((x_noisy.shape[1], 2), dtype=np.float64)
    for parameter_index in range(2):
        derivative = derivatives[parameter_index]
        numerator = precision @ derivative
        projected_information = 0.0
        for previous in range(parameter_index):
            projection = float(derivative @ moped_weights[:, previous])
            numerator -= projection * moped_weights[:, previous]
            projected_information += projection**2
        information = float(derivative @ precision @ derivative)
        denominator_squared = information - projected_information
        tolerance = max(abs(information), 1.0) * 1.0e-12
        if denominator_squared <= tolerance:
            raise ValueError(
                f"MOPED parameter {TARGET_PARAMETERS[parameter_index]} is "
                "numerically degenerate after conditioning on earlier parameters"
            )
        moped_weights[:, parameter_index] = numerator / np.sqrt(
            denominator_squared
        )

    compressed_covariance = moped_weights.T @ covariance @ moped_weights
    compressed_derivatives = moped_weights.T @ derivatives.T
    fisher_original = derivatives @ precision @ derivatives.T
    fisher_compressed = compressed_derivatives.T @ compressed_derivatives
    fisher_scale = max(float(np.max(np.abs(fisher_original))), 1.0e-300)
    fisher_relative_error = float(
        np.max(np.abs(fisher_compressed - fisher_original)) / fisher_scale
    )
    covariance_identity_error = float(
        np.max(np.abs(compressed_covariance - np.eye(2)))
    )
    if fisher_relative_error > 1.0e-7 or covariance_identity_error > 1.0e-7:
        raise ValueError(
            "MOPED numerical validation failed: "
            f"Fisher relative error={fisher_relative_error:.3e}, "
            f"compressed covariance error={covariance_identity_error:.3e}"
        )

    x_moped = (x_noisy - fiducial_mean[None, :]) @ moped_weights
    return {
        "x_moped": x_moped.astype(np.float32),
        "weights": moped_weights,
        "fiducial_mean": fiducial_mean,
        "fiducial_theta": fiducial_theta,
        "derivatives": derivatives,
        "covariance": covariance,
        "precision": precision,
        "compressed_covariance": compressed_covariance,
        "compressed_derivatives": compressed_derivatives,
        "fisher_original": fisher_original,
        "fisher_compressed": fisher_compressed,
        "local_indices": local_indices,
        "local_radius": local_radius,
        "paired_noise_bin_scale": bin_scale,
        "oas_shrinkage": oas_shrinkage,
        "covariance_eigenvalues": covariance_eigenvalues,
        "covariance_condition": covariance_condition,
        "fisher_relative_error": fisher_relative_error,
        "covariance_identity_error": covariance_identity_error,
        "regression_rank": int(regression_rank),
        "regression_bandwidth": bandwidth,
    }


def write_metadata_csv(
    path: Path,
    rows: list[int],
    records: dict[int, dict[str, Any]],
    theta_full: np.ndarray,
    sequence_offset: int,
) -> None:
    fields = [
        "dataset_index",
        "sobol_row",
        "sobol_sequence_index",
        "mask_seed",
        "noise_seed",
        *FULL_PARAMETERS,
        "source_no_noise_cl_path",
        "source_cl_path",
        "source_manifest",
        "row_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset_index, row in enumerate(rows):
            record = records[row]
            payload = {
                "dataset_index": dataset_index,
                "sobol_row": row,
                "sobol_sequence_index": sequence_offset + row,
                "mask_seed": record["mask_seed"],
                "noise_seed": record["noise_seed"],
                "source_no_noise_cl_path": record["no_noise_output_path"],
                "source_cl_path": record["output_path"],
                "source_manifest": record["_manifest"],
                "row_log": record["row_log"],
            }
            payload.update(
                {
                    name: theta_full[row - 1, index]
                    for index, name in enumerate(FULL_PARAMETERS)
                }
            )
            writer.writerow(payload)

def main() -> int:
    args = parse_args()
    if args.sequence_offset < 0:
        raise ValueError("--sequence-offset must be non-negative")
    if args.test_last_n < 0:
        raise ValueError("--test-last-n must be non-negative")

    sobol_csv = args.sobol_csv.expanduser().resolve()
    design_metadata_path = (
        args.design_metadata.expanduser().resolve()
        if args.design_metadata is not None
        else sobol_csv.with_suffix(".npz")
    )
    manifest_dir = args.manifest_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / args.dataset_name
    unbinned_path = output_dir / args.unbinned_name
    unbinned_no_noise_path = output_dir / args.unbinned_no_noise_name
    metadata_csv_path = dataset_path.with_name(dataset_path.stem + "_metadata.csv")
    manifest_json_path = dataset_path.with_name(dataset_path.stem + "_manifest.json")
    completion_path = output_dir / "combination_complete.json"

    outputs = [
        dataset_path,
        unbinned_path,
        unbinned_no_noise_path,
        metadata_csv_path,
        manifest_json_path,
        completion_path,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite {existing}; use --force")

    theta_csv = load_csv_theta(sobol_csv)
    n_expected = theta_csv.shape[0]
    design = load_design_metadata(design_metadata_path, n_expected)
    if not np.array_equal(theta_csv, design["theta_full"].astype(np.float64)):
        raise ValueError("Sobol CSV values do not exactly match theta_full in design metadata")
    if int(scalar(design["noise_seed_base"])) != args.noise_seed_base:
        raise ValueError(
            "Combiner --noise-seed-base does not match design metadata: "
            f"{args.noise_seed_base} != {scalar(design['noise_seed_base'])}"
        )
    if int(scalar(design["sequence_offset"])) != args.sequence_offset:
        raise ValueError(
            "Combiner --sequence-offset does not match design metadata: "
            f"{args.sequence_offset} != {scalar(design['sequence_offset'])}"
        )
    expected_sequence_indices = (
        args.sequence_offset
        + np.arange(1, n_expected + 1, dtype=np.int64)
    )
    if not np.array_equal(
        design["sobol_sequence_index"], expected_sequence_indices
    ):
        raise ValueError("Design Sobol sequence indices do not match the requested block")

    records, failures = load_success_records(
        manifest_dir,
        n_expected,
        args.mask_seed,
        args.noise_seed_base,
        args.sequence_offset,
    )
    missing_rows = sorted(set(range(1, n_expected + 1)).difference(records))
    if missing_rows and not args.allow_missing:
        preview = missing_rows[:20]
        raise RuntimeError(
            f"{len(missing_rows)} of {n_expected} rows lack valid paired outputs. "
            f"First missing rows: {preview}"
        )
    selected_rows = sorted(records)
    if not selected_rows:
        raise RuntimeError("No valid generated spectra were found")

    first_ell, _ = select_cl(
        Path(records[selected_rows[0]]["output_path"]),
        args.ell_min,
        args.ell_max,
    )
    clean_first_ell, _ = select_cl(
        Path(records[selected_rows[0]]["no_noise_output_path"]),
        args.ell_min,
        args.ell_max,
    )
    if not np.array_equal(clean_first_ell, first_ell):
        raise ValueError("First clean and noisy spectra use different ell grids")

    bins = make_bins(first_ell)
    n_rows = len(selected_rows)
    n_bins = len(bins["indices"])
    x = np.empty((n_rows, n_bins), dtype=np.float32)
    x_no_noise = np.empty((n_rows, n_bins), dtype=np.float32)
    unbinned = np.lib.format.open_memmap(
        unbinned_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, first_ell.size),
    )
    unbinned_no_noise = np.lib.format.open_memmap(
        unbinned_no_noise_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, first_ell.size),
    )
    dl_factor = first_ell * (first_ell + 1.0) / (2.0 * np.pi)

    for output_index, row in enumerate(selected_rows):
        ell, cl = select_cl(
            Path(records[row]["output_path"]),
            args.ell_min,
            args.ell_max,
        )
        clean_ell, clean_cl = select_cl(
            Path(records[row]["no_noise_output_path"]),
            args.ell_min,
            args.ell_max,
        )
        if not np.array_equal(ell, first_ell) or not np.array_equal(
            clean_ell, first_ell
        ):
            raise ValueError(f"Row {row} uses a different ell grid")

        dell = cl * dl_factor
        clean_dell = clean_cl * dl_factor
        unbinned[output_index] = dell.astype(np.float32)
        unbinned_no_noise[output_index] = clean_dell.astype(np.float32)
        x[output_index] = bin_dell(dell, bins)
        x_no_noise[output_index] = bin_dell(clean_dell, bins)
        if (output_index + 1) % 256 == 0 or output_index + 1 == n_rows:
            unbinned.flush()
            unbinned_no_noise.flush()
            print(f"Processed {output_index + 1}/{n_rows}", flush=True)
    del unbinned
    del unbinned_no_noise

    row_index = np.asarray(selected_rows, dtype=np.int64) - 1
    theta_full = theta_csv[row_index].astype(np.float32)
    target_indices = [FULL_PARAMETERS.index(name) for name in TARGET_PARAMETERS]
    theta = theta_full[:, target_indices]
    prior_low = design["prior_low"].astype(np.float32)
    prior_high = design["prior_high"].astype(np.float32)
    sobol_rows = np.asarray(selected_rows, dtype=np.int64)
    sobol_sequence_indices = args.sequence_offset + sobol_rows
    noise_seeds = args.noise_seed_base + sobol_sequence_indices
    test_count = min(args.test_last_n, n_rows)
    test_indices = np.arange(n_rows - test_count, n_rows, dtype=np.int64)

    moped = None
    if not args.skip_moped:
        moped = build_moped_compression(
            x,
            x_no_noise,
            theta,
            prior_low,
            prior_high,
            test_indices,
            args.moped_local_n,
        )
        print(
            "MOPED validation: "
            f"local_n={moped['local_indices'].size}, "
            f"OAS={moped['oas_shrinkage']:.4f}, "
            f"cov_condition={moped['covariance_condition']:.4e}, "
            f"Fisher_error={moped['fisher_relative_error']:.3e}",
            flush=True,
        )

    metadata = {
        "product": PRODUCT,
        "statistic": (
            "weighted mean of linear D_ell; D_ell=ell(ell+1)C_ell/(2pi); "
            "weights=2ell+1; no clipping, floor, log10, or asinh applied"
        ),
        "n_rows": n_rows,
        "n_expected_rows": n_expected,
        "complete": not missing_rows,
        "missing_rows": missing_rows,
        "failed_manifest_rows": sorted(failures),
        "varying_parameters": TARGET_PARAMETERS,
        "fixed_parameters": {
            name: float(theta_csv[0, FULL_PARAMETERS.index(name)])
            for name in FULL_PARAMETERS
            if name not in TARGET_PARAMETERS
        },
        "mask_seed": args.mask_seed,
        "noise_seed_policy": (
            "noise_seed_base + sequence_offset + one_based_local_sobol_row"
        ),
        "noise_seed_base": args.noise_seed_base,
        "sequence_offset": args.sequence_offset,
        "sobol_sequence_index_semantics": (
            "one-based index in the common two-dimensional Sobol sequence"
        ),
        "same_mask_all_rows": True,
        "independent_noise_all_rows": True,
        "paired_masked_no_noise_saved": True,
        "beam_applied_to_signal": True,
        "beam_fwhm_arcmin": 2.0,
        "ell_min": args.ell_min,
        "ell_max": args.ell_max,
        "bin_weighting": "2ell_plus_1",
        "bin_width": 200,
        "source_sobol_csv": str(sobol_csv),
        "source_design_metadata": str(design_metadata_path),
        "source_manifest_dir": str(manifest_dir),
        "unbinned_dell_path": str(unbinned_path),
        "unbinned_no_noise_dell_path": str(unbinned_no_noise_path),
        "moped": {
            "enabled": moped is not None,
            "method": (
                "Battaglia12-local quadratic clean derivatives; paired "
                "noisy-minus-clean standardized-bin OAS covariance; "
                "standard two-parameter MOPED"
            ),
            "fit_excludes_test_indices": True,
            "local_n_requested": args.moped_local_n,
            "local_n_used": (
                int(moped["local_indices"].size) if moped is not None else 0
            ),
            "oas_shrinkage": (
                float(moped["oas_shrinkage"]) if moped is not None else None
            ),
            "covariance_condition": (
                float(moped["covariance_condition"])
                if moped is not None
                else None
            ),
            "fisher_relative_error": (
                float(moped["fisher_relative_error"])
                if moped is not None
                else None
            ),
            "compressed_covariance_identity_error": (
                float(moped["covariance_identity_error"])
                if moped is not None
                else None
            ),
            "recommended_npe_preprocessing": (
                "standardize signed MOPED components using training rows; "
                "do not reuse the 40-bin asinh scale"
            ),
        },
    }

    payload: dict[str, Any] = {
        "theta": theta,
        "theta_full": theta_full,
        "x": x,
        "x_40": x,
        "x_no_noise": x_no_noise,
        "obs": x[-1],
        "obs_theta": theta[-1],
        "obs_index": np.asarray(n_rows - 1, dtype=np.int64),
        "obs_source": np.asarray("dataset-row"),
        "test_indices": test_indices,
        "ell": bins["ell_binned"],
        "ell_binned": bins["ell_binned"],
        "ell_unbinned": first_ell.astype(np.float32),
        "bin_counts": bins["bin_counts"],
        "bin_ell_min": bins["bin_ell_min"],
        "bin_ell_max": bins["bin_ell_max"],
        "prior_low": prior_low,
        "prior_high": prior_high,
        "param_names": np.asarray(TARGET_PARAMETERS),
        "theta_columns": np.asarray(TARGET_PARAMETERS),
        "full_param_names": np.asarray(FULL_PARAMETERS),
        "sobol_row": sobol_rows,
        "sobol_sequence_index": sobol_sequence_indices,
        "sobol_global_row": sobol_sequence_indices,
        "mask_seed": np.full(n_rows, args.mask_seed, dtype=np.int64),
        "noise_seed": noise_seeds,
        "case_name": np.asarray(PRODUCT),
        "product": np.asarray(PRODUCT),
        "source_sobol_csv_path": np.asarray(str(sobol_csv)),
        "unbinned_dell_path": np.asarray(str(unbinned_path)),
        "unbinned_no_noise_dell_path": np.asarray(
            str(unbinned_no_noise_path)
        ),
        "metadata_json": np.asarray(json.dumps(jsonable(metadata), sort_keys=True)),
    }
    if moped is not None:
        payload.update(
            {
                "x_moped": moped["x_moped"],
                "obs_moped": moped["x_moped"][-1],
                "moped_param_names": np.asarray(TARGET_PARAMETERS),
                "moped_weights": moped["weights"],
                "moped_fiducial_mean": moped["fiducial_mean"],
                "moped_fiducial_theta": moped["fiducial_theta"],
                "moped_derivatives": moped["derivatives"],
                "moped_covariance": moped["covariance"],
                "moped_precision": moped["precision"],
                "moped_compressed_covariance": moped[
                    "compressed_covariance"
                ],
                "moped_compressed_derivatives": moped[
                    "compressed_derivatives"
                ],
                "moped_fisher_original": moped["fisher_original"],
                "moped_fisher_compressed": moped["fisher_compressed"],
                "moped_local_indices": moped["local_indices"],
                "moped_local_radius": moped["local_radius"],
                "moped_paired_noise_bin_scale": moped[
                    "paired_noise_bin_scale"
                ],
                "moped_oas_shrinkage": np.asarray(moped["oas_shrinkage"]),
                "moped_covariance_eigenvalues": moped[
                    "covariance_eigenvalues"
                ],
            }
        )
    np.savez_compressed(dataset_path, **payload)

    write_metadata_csv(
        metadata_csv_path,
        selected_rows,
        records,
        theta_csv,
        args.sequence_offset,
    )
    with manifest_json_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(metadata), handle, indent=2, sort_keys=True)
    with completion_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "complete": not missing_rows,
                "dataset": str(dataset_path),
                "unbinned_dell": str(unbinned_path),
                "unbinned_no_noise_dell": str(unbinned_no_noise_path),
                "n_rows": n_rows,
                "n_expected_rows": n_expected,
                "x_shape": list(x.shape),
                "x_moped_shape": (
                    list(moped["x_moped"].shape) if moped is not None else None
                ),
                "theta_shape": list(theta.shape),
                "sequence_offset": args.sequence_offset,
                "sobol_sequence_index_first": int(sobol_sequence_indices[0]),
                "sobol_sequence_index_last": int(sobol_sequence_indices[-1]),
                "moped_enabled": moped is not None,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {dataset_path}")
    print(f"Wrote {unbinned_path}")
    print(f"Wrote {unbinned_no_noise_path}")
    print(f"Wrote {metadata_csv_path}")
    print(f"Wrote {manifest_json_path}")
    print(f"Wrote {completion_path}")
    print(f"theta shape: {theta.shape}")
    print(f"x shape: {x.shape}")
    if moped is not None:
        print(f"x_moped shape: {moped['x_moped'].shape}")
    print(f"unbinned noisy D_ell shape: ({n_rows}, {first_ell.size})")
    print(f"unbinned clean D_ell shape: ({n_rows}, {first_ell.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
