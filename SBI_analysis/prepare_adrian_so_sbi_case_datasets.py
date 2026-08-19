#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PARAM_NAMES = np.asarray(
    [
        "P0",
        "xc",
        "beta",
        "alpha_m_P0",
        "alpha_m_xc",
        "alpha_m_beta",
        "alpha_z_P0",
        "alpha_z_xc",
        "alpha_z_beta",
    ],
    dtype=str,
)

SOBOL_PRIOR_BOUNDS = {
    "P0": [1.832524, 34.341221],
    "xc": [0.150011, 0.844503],
    "beta": [3.480627, 5.216611],
    "alpha_m_P0": [0.000312, 0.292251],
    "alpha_m_xc": [-0.099718, 0.099795],
    "alpha_m_beta": [-0.019935, 0.099767],
    "alpha_z_P0": [-1.363457, -0.228839],
    "alpha_z_xc": [0.147393, 1.314474],
    "alpha_z_beta": [0.083808, 0.745884],
}

DEFAULT_PRODUCTS = [
    "unmasked_no_noise",
    "masked_no_noise",
    "masked_baseline_noise_cross_deproj0",
    "masked_baseline_noise_cross_deproj2",
    "masked_goal_noise_cross_deproj0",
    "masked_goal_noise_cross_deproj2",
]

PRODUCT_ALIASES = {
    "unmasked": "unmasked_no_noise",
    "masked": "masked_no_noise",
    "no_noise": "masked_no_noise",
    "baseline_deproj0": "masked_baseline_noise_cross_deproj0",
    "baseline_deproj2": "masked_baseline_noise_cross_deproj2",
    "goal_deproj0": "masked_goal_noise_cross_deproj0",
    "goal_deproj2": "masked_goal_noise_cross_deproj2",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


def ell_tag(ell_min: float | None, ell_max: float | None) -> str:
    def fmt(value: float | None, fallback: str) -> str:
        if value is None:
            return fallback
        value = float(value)
        text = str(int(value)) if value.is_integer() else f"{value:g}"
        return text.replace("-", "m").replace(".", "p")

    return f"ell{fmt(ell_min, 'min')}_{fmt(ell_max, 'max')}"


def normalize_products(values: list[str]) -> list[str]:
    products: list[str] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            product = PRODUCT_ALIASES.get(part, part)
            if product and product not in products:
                products.append(product)
    return products


def bin_weights(ell_values: np.ndarray, weighting: str) -> np.ndarray:
    ell_values = np.asarray(ell_values, dtype=np.float64)
    weighting = str(weighting or "2ell_plus_1").lower()
    if weighting in {"uniform", "none", "flat"}:
        return np.ones_like(ell_values, dtype=np.float64)
    if weighting == "ell":
        return ell_values
    if weighting in {"2ell_plus_1", "modes", "mode_count"}:
        return 2.0 * ell_values + 1.0
    raise ValueError(f"Unsupported bin weighting: {weighting!r}")


def make_delta200_bins(source_ell: np.ndarray, weighting: str) -> dict[str, Any]:
    source_ell = np.asarray(source_ell, dtype=np.float64).reshape(-1)
    edges = np.r_[np.arange(80, 7881, 200), 7979].astype(np.int64)
    bin_min = edges[:-1].copy()
    bin_max = edges[1:].copy()
    bin_max[:-1] -= 1

    matrix = np.zeros((source_ell.size, bin_min.size), dtype=np.float64)
    centers: list[float] = []
    counts: list[int] = []
    for i, (lo, hi) in enumerate(zip(bin_min, bin_max)):
        idx = np.flatnonzero((source_ell >= float(lo)) & (source_ell <= float(hi)))
        if idx.size == 0:
            raise ValueError(
                f"No unbinned ell values found for SO bin {lo}-{hi}; "
                f"source range is {source_ell.min()}-{source_ell.max()}."
            )
        weights = bin_weights(source_ell[idx], weighting)
        matrix[idx, i] = weights / np.sum(weights)
        centers.append(float(np.average(source_ell[idx], weights=weights)))
        counts.append(int(idx.size))

    dl_factor = source_ell * (source_ell + 1.0) / (2.0 * np.pi)
    return {
        "ell": np.asarray(centers, dtype=np.float32),
        "bin_ell_min": np.asarray(bin_min, dtype=np.float32),
        "bin_ell_max": np.asarray(bin_max, dtype=np.float32),
        "bin_counts": np.asarray(counts, dtype=np.int64),
        "cl_to_dl_matrix": np.ascontiguousarray(dl_factor[:, None] * matrix, dtype=np.float32),
    }


def ell_mask(
    ell: np.ndarray,
    bin_ell_min: np.ndarray,
    bin_ell_max: np.ndarray,
    ell_min: float | None,
    ell_max: float | None,
    selection: str,
) -> np.ndarray:
    ell = np.asarray(ell, dtype=float)
    bin_ell_min = np.asarray(bin_ell_min, dtype=float)
    bin_ell_max = np.asarray(bin_ell_max, dtype=float)
    low = -np.inf if ell_min is None else float(ell_min)
    high = np.inf if ell_max is None else float(ell_max)
    if low > high:
        raise ValueError(f"ell_min={ell_min} is larger than ell_max={ell_max}")
    if selection == "center":
        mask = (ell >= low) & (ell <= high)
    elif selection == "contained":
        mask = (bin_ell_min >= low) & (bin_ell_max <= high)
    elif selection == "overlap":
        mask = (bin_ell_max >= low) & (bin_ell_min <= high)
    else:
        raise ValueError(f"Unsupported ell selection mode: {selection!r}")
    if not np.any(mask):
        raise ValueError(f"No bins selected for ell_min={ell_min}, ell_max={ell_max}, selection={selection}")
    return np.asarray(mask, dtype=bool)


def resolve_obs_index(obs_index: int, n_rows: int) -> int:
    idx = int(obs_index)
    if idx < 0:
        idx = int(n_rows) + idx
    if not (0 <= idx < int(n_rows)):
        raise IndexError(f"obs_index={obs_index} resolves to {idx}, outside 0..{int(n_rows) - 1}")
    return idx


def product_cl_path(input_dir: Path, product: str) -> Path:
    return input_dir / f"sbi_{product}_cl.npy"


def product_metadata_path(
    input_dir: Path,
    product: str,
    explicit_metadata: Path | None,
    metadata_dir: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_metadata is not None:
        candidates.append(explicit_metadata)
    if metadata_dir is not None:
        candidates.append(metadata_dir / f"sbi_{product}.npz")
    candidates.append(input_dir / f"sbi_{product}.npz")
    if metadata_dir is not None:
        candidates.extend(sorted(metadata_dir.glob("sbi_*.npz")))
    candidates.extend(sorted(input_dir.glob("sbi_*.npz")))
    for path in candidates:
        if path.is_file():
            return path
    return None


def product_metadata_csv_path(input_dir: Path, product: str, metadata_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if metadata_dir is not None:
        candidates.append(metadata_dir / f"sbi_{product}_metadata.csv")
    candidates.append(input_dir / f"sbi_{product}_metadata.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


def sobol_csv_columns(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    col_to_idx = {name: idx for idx, name in enumerate(header)}
    missing = [str(name) for name in PARAM_NAMES if str(name) not in col_to_idx]
    if missing:
        raise KeyError(f"Sobol CSV {path} is missing columns {missing}; header={header}")
    return col_to_idx


def load_sobol_theta_for_global_rows(path: Path, sobol_global_row: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Sobol CSV not found: {path}")
    rows = np.asarray(sobol_global_row, dtype=np.int64).reshape(-1)
    if rows.size == 0:
        raise ValueError("Cannot load Sobol theta for empty sobol_global_row")
    if np.any(rows <= 0):
        raise ValueError("sobol_global_row must be 1-based positive row numbers")

    col_to_idx = sobol_csv_columns(path)
    max_row = int(np.max(rows))
    theta_table = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        max_rows=max_row,
        usecols=[col_to_idx[str(name)] for name in PARAM_NAMES],
        dtype=np.float32,
    )
    theta_table = np.asarray(theta_table, dtype=np.float32)
    if theta_table.ndim == 1:
        theta_table = theta_table.reshape(1, -1)
    if theta_table.shape[0] < max_row:
        raise ValueError(f"Sobol CSV {path} has only {theta_table.shape[0]} rows, need row {max_row}")
    return np.ascontiguousarray(theta_table[rows - 1], dtype=np.float32)


def parse_expected_sobol_prefix(value: str) -> np.ndarray:
    parts = [part for part in str(value).replace(",", " ").split() if part]
    return np.asarray([int(part) for part in parts], dtype=np.int64)


def load_sobol_global_row(path: Path) -> np.ndarray:
    """Load the array-index -> 1-based Sobol-row permutation from a small file."""
    if not path.is_file():
        raise FileNotFoundError(f"sobol_global_row file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "sobol_global_row" in data.files:
                values = data["sobol_global_row"]
            elif len(data.files) == 1:
                values = data[data.files[0]]
            else:
                raise KeyError(
                    f"{path} must contain 'sobol_global_row'; available keys are {data.files}"
                )
    elif suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            first = next(reader, None)
        if first is None:
            raise ValueError(f"sobol_global_row file is empty: {path}")
        try:
            float(first[0])
            has_header = False
        except ValueError:
            has_header = True
        values = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1 if has_header else 0,
            usecols=0,
            dtype=np.float64,
        )
    else:
        values = np.loadtxt(path, dtype=np.float64)

    raw = np.asarray(values).reshape(-1)
    if raw.size == 0:
        raise ValueError(f"sobol_global_row file is empty: {path}")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"sobol_global_row must be numeric, got dtype={raw.dtype} from {path}")
    raw_float = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(raw_float)):
        raise ValueError(f"sobol_global_row contains non-finite values: {path}")
    if not np.all(raw_float == np.floor(raw_float)):
        raise ValueError(f"sobol_global_row contains non-integer values: {path}")
    return np.ascontiguousarray(raw_float, dtype=np.int64)


def validate_sobol_global_row(
    rows: np.ndarray,
    expected_rows: int,
    expected_prefix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate the complete Adrian array-index -> Sobol-row permutation."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    expected_rows = int(expected_rows)
    if rows.size != expected_rows:
        raise ValueError(
            f"sobol_global_row has {rows.size} entries, but C_ell has {expected_rows} rows"
        )
    if np.any(rows < 1) or np.any(rows > expected_rows):
        raise ValueError(
            f"sobol_global_row must stay in 1..{expected_rows}; "
            f"found min={int(rows.min())}, max={int(rows.max())}"
        )

    counts = np.bincount(rows, minlength=expected_rows + 1)[1:]
    missing = np.flatnonzero(counts == 0) + 1
    duplicate = np.flatnonzero(counts > 1) + 1
    if missing.size or duplicate.size:
        raise ValueError(
            "sobol_global_row is not a permutation of the expected Sobol rows: "
            f"missing(first 10)={missing[:10].tolist()}, "
            f"duplicated(first 10)={duplicate[:10].tolist()}"
        )

    prefix = np.asarray(expected_prefix if expected_prefix is not None else [], dtype=np.int64)
    if prefix.size:
        actual = rows[: prefix.size]
        if not np.array_equal(actual, prefix):
            raise ValueError(
                "sobol_global_row does not match the known source-machine prefix: "
                f"expected={prefix.tolist()}, actual={actual.tolist()}"
            )

    canonical_bytes = np.ascontiguousarray(rows, dtype="<i8").tobytes()
    return {
        "n_rows": expected_rows,
        "min": int(rows.min()),
        "max": int(rows.max()),
        "unique": int(np.unique(rows).size),
        "known_prefix_checked": prefix.tolist(),
        "sha256_int64_le": hashlib.sha256(canonical_bytes).hexdigest(),
        "is_identity": bool(np.array_equal(rows, np.arange(1, expected_rows + 1, dtype=np.int64))),
        "is_complete_permutation": True,
    }


def read_sobol_mapping_metadata(
    sobol_csv: Path,
    sobol_global_row_path: Path,
    cl_n_rows: int,
    n_rows: int,
    n_ell: int,
    ell_start: int,
    ell_stop: int,
    expected_prefix: np.ndarray,
) -> dict[str, Any]:
    full_mapping = load_sobol_global_row(sobol_global_row_path)
    validation = validate_sobol_global_row(full_mapping, cl_n_rows, expected_prefix)
    selected_mapping = np.ascontiguousarray(full_mapping[:n_rows], dtype=np.int64)
    theta = load_sobol_theta_for_global_rows(sobol_csv, selected_mapping)

    ell = np.arange(int(ell_start), int(ell_stop) + 1, dtype=np.float32)
    if ell.size != int(n_ell):
        raise ValueError(
            f"Fallback ell range {ell_start}..{ell_stop} has {ell.size} values, "
            f"but C_ell arrays have {n_ell} columns."
        )

    prior_low = np.asarray([SOBOL_PRIOR_BOUNDS[str(name)][0] for name in PARAM_NAMES], dtype=np.float32)
    prior_high = np.asarray([SOBOL_PRIOR_BOUNDS[str(name)][1] for name in PARAM_NAMES], dtype=np.float32)
    return {
        "theta": np.ascontiguousarray(theta, dtype=np.float32),
        "ell_unbinned": np.ascontiguousarray(ell, dtype=np.float32),
        "param_names": np.ascontiguousarray(PARAM_NAMES, dtype=str),
        "theta_columns": np.ascontiguousarray(PARAM_NAMES, dtype=str),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "sobol_global_row": selected_mapping,
        "metadata_path": "",
        "metadata_csv_path": "",
        "sobol_csv_path": str(sobol_csv),
        "sobol_global_row_path": str(sobol_global_row_path),
        "sobol_global_row_validation": validation,
        "theta_source": "sobol_global_row_mapping_plus_sobol_csv",
    }


def read_metadata_csv(
    path: Path,
    n_rows: int,
    n_ell: int,
    ell_start: int,
    ell_stop: int,
    sobol_csv: Path | None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for idx, row in enumerate(reader):
            if idx >= int(n_rows):
                break
            rows.append(row)

    if len(rows) != int(n_rows):
        raise ValueError(f"Metadata CSV {path} has {len(rows)} rows, expected {n_rows}")

    theta_columns_present = all(str(name) in fieldnames for name in PARAM_NAMES)
    row_col = next(
        (
            name
            for name in (
                "sobol_global_row",
                "Sobol_global_row",
                "sobol_row",
                "global_row",
                "row",
            )
            if name in fieldnames
        ),
        None,
    )

    sobol_global_row = (
        np.asarray([int(float(row[row_col])) for row in rows], dtype=np.int64)
        if row_col is not None
        else np.arange(1, int(n_rows) + 1, dtype=np.int64)
    )

    if theta_columns_present:
        theta = np.asarray(
            [[float(row[str(name)]) for name in PARAM_NAMES] for row in rows],
            dtype=np.float32,
        )
        theta_source = "metadata_csv_theta_columns"
    elif sobol_csv is not None and row_col is not None:
        theta = load_sobol_theta_for_global_rows(sobol_csv, sobol_global_row)
        theta_source = "metadata_csv_sobol_global_row_plus_sobol_csv"
    else:
        raise KeyError(
            f"Metadata CSV {path} does not contain all theta columns {PARAM_NAMES.tolist()}. "
            "Pass --sobol-csv and make sure the CSV contains sobol_global_row."
        )

    ell = np.arange(int(ell_start), int(ell_stop) + 1, dtype=np.float32)
    if ell.size != int(n_ell):
        raise ValueError(
            f"Fallback ell range {ell_start}..{ell_stop} has {ell.size} values, "
            f"but C_ell arrays have {n_ell} columns."
        )

    prior_low = np.asarray([SOBOL_PRIOR_BOUNDS[str(name)][0] for name in PARAM_NAMES], dtype=np.float32)
    prior_high = np.asarray([SOBOL_PRIOR_BOUNDS[str(name)][1] for name in PARAM_NAMES], dtype=np.float32)
    return {
        "theta": np.ascontiguousarray(theta, dtype=np.float32),
        "ell_unbinned": np.ascontiguousarray(ell, dtype=np.float32),
        "param_names": np.ascontiguousarray(PARAM_NAMES, dtype=str),
        "theta_columns": np.ascontiguousarray(PARAM_NAMES, dtype=str),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "sobol_global_row": np.ascontiguousarray(sobol_global_row, dtype=np.int64),
        "metadata_path": "",
        "metadata_csv_path": str(path),
        "sobol_csv_path": str(sobol_csv) if sobol_csv is not None else "",
        "theta_source": theta_source,
    }


def read_sobol_csv_metadata(
    path: Path,
    n_rows: int,
    n_ell: int,
    ell_start: int,
    ell_stop: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Sobol CSV not found: {path}")
    if int(n_rows) <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")

    col_to_idx = sobol_csv_columns(path)

    theta = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        max_rows=int(n_rows),
        usecols=[col_to_idx[str(name)] for name in PARAM_NAMES],
        dtype=np.float32,
    )
    theta = np.asarray(theta, dtype=np.float32)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    if theta.shape != (int(n_rows), PARAM_NAMES.size):
        raise ValueError(f"Sobol theta shape {theta.shape} does not match expected {(int(n_rows), PARAM_NAMES.size)}")

    ell = np.arange(int(ell_start), int(ell_stop) + 1, dtype=np.float32)
    if ell.size != int(n_ell):
        raise ValueError(
            f"Fallback ell range {ell_start}..{ell_stop} has {ell.size} values, "
            f"but C_ell arrays have {n_ell} columns."
        )

    prior_low = np.asarray([SOBOL_PRIOR_BOUNDS[str(name)][0] for name in PARAM_NAMES], dtype=np.float32)
    prior_high = np.asarray([SOBOL_PRIOR_BOUNDS[str(name)][1] for name in PARAM_NAMES], dtype=np.float32)
    return {
        "theta": np.ascontiguousarray(theta, dtype=np.float32),
        "ell_unbinned": np.ascontiguousarray(ell, dtype=np.float32),
        "param_names": np.ascontiguousarray(PARAM_NAMES, dtype=str),
        "theta_columns": np.ascontiguousarray(PARAM_NAMES, dtype=str),
        "prior_low": prior_low,
        "prior_high": prior_high,
        "sobol_global_row": np.arange(1, int(n_rows) + 1, dtype=np.int64),
        "metadata_path": "",
        "metadata_csv_path": "",
        "sobol_csv_path": str(path),
        "theta_source": "sobol_csv_identity_row_order",
    }


def read_metadata(path: Path, theta_path: Path | None) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}

    if theta_path is not None:
        if theta_path.suffix.lower() == ".npz":
            with np.load(theta_path, allow_pickle=True) as theta_data:
                theta = theta_data["theta"] if "theta" in theta_data.files else theta_data[theta_data.files[0]]
        else:
            theta = np.load(theta_path, allow_pickle=True)
        payload["theta"] = theta

    if "theta" not in payload:
        raise KeyError(f"Metadata file {path} does not contain theta. Pass --theta-path if theta is stored separately.")
    if "ell" not in payload:
        raise KeyError(f"Metadata file {path} does not contain ell.")

    theta = np.asarray(payload["theta"], dtype=np.float32)
    ell = np.asarray(payload["ell"], dtype=np.float32)
    theta_columns = np.asarray(payload.get("theta_columns", PARAM_NAMES), dtype=str)
    prior_low = np.asarray(
        payload.get("prior_low", [SOBOL_PRIOR_BOUNDS[str(name)][0] for name in theta_columns]),
        dtype=np.float32,
    )
    prior_high = np.asarray(
        payload.get("prior_high", [SOBOL_PRIOR_BOUNDS[str(name)][1] for name in theta_columns]),
        dtype=np.float32,
    )
    sobol_global_row = np.asarray(payload["sobol_global_row"], dtype=np.int64) if "sobol_global_row" in payload else None

    if theta.ndim != 2:
        raise ValueError(f"theta must be 2D, got {theta.shape}")
    if ell.ndim != 1:
        raise ValueError(f"ell must be 1D, got {ell.shape}")
    if theta.shape[1] != theta_columns.size:
        raise ValueError(f"theta columns {theta.shape[1]} do not match theta_columns size {theta_columns.size}")
    return {
        "theta": np.ascontiguousarray(theta, dtype=np.float32),
        "ell_unbinned": np.ascontiguousarray(ell, dtype=np.float32),
        "param_names": np.ascontiguousarray(theta_columns, dtype=str),
        "theta_columns": np.ascontiguousarray(theta_columns, dtype=str),
        "prior_low": np.ascontiguousarray(prior_low, dtype=np.float32),
        "prior_high": np.ascontiguousarray(prior_high, dtype=np.float32),
        "sobol_global_row": sobol_global_row,
        "metadata_path": str(path),
        "metadata_csv_path": "",
        "sobol_csv_path": "",
        "theta_source": "metadata_npz_theta",
    }


def bin_product_cl(
    cl_path: Path,
    matrix: np.ndarray,
    n_rows: int,
    n_ell: int,
    chunk_rows: int,
) -> np.ndarray:
    cl = np.load(cl_path, mmap_mode="r")
    if cl.ndim != 2:
        raise ValueError(f"{cl_path} must be 2D, got {cl.shape}")
    if cl.shape[0] < n_rows:
        raise ValueError(f"{cl_path} rows {cl.shape[0]} are fewer than requested rows {n_rows}")
    if cl.shape[1] != n_ell:
        raise ValueError(f"{cl_path} ell dimension {cl.shape[1]} does not match metadata ell size {n_ell}")

    out = np.empty((n_rows, matrix.shape[1]), dtype=np.float32)
    for start in range(0, n_rows, int(chunk_rows)):
        stop = min(start + int(chunk_rows), n_rows)
        out[start:stop] = np.asarray(cl[start:stop], dtype=np.float32) @ matrix
        print(f"  binned rows {stop}/{n_rows}")
    return np.ascontiguousarray(out, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Prepare compact SBI-ready Delta=200 D_ell datasets from Adrian's consolidated SO "
            "sbi_<product>_cl.npy files. The large C_ell arrays are read with mmap_mode='r'."
        )
    )
    parser.add_argument("--input-dir", default="/lustre/work/kristero10/adrian_dataset")
    parser.add_argument("--metadata-dir", default="", help="Optional directory containing sbi_<product>.npz metadata files.")
    parser.add_argument("--metadata-npz", default="", help="Optional metadata NPZ to use for theta/ell for every product.")
    parser.add_argument("--theta-path", default="", help="Optional theta file if theta is not inside the metadata NPZ.")
    parser.add_argument(
        "--sobol-csv",
        default="",
        help=(
            "Fallback CSV containing the 9 Battaglia theta columns. Used when metadata NPZ files are absent; "
            "only the rows needed to match the C_ell file are read."
        ),
    )
    parser.add_argument(
        "--sobol-global-row-path",
        default="",
        help=(
            "Small .npy/.npz/.csv/.txt file containing the complete 1-based sobol_global_row "
            "permutation in C_ell array order. Together with --sobol-csv this replaces the large metadata files."
        ),
    )
    parser.add_argument(
        "--expected-sobol-prefix",
        default="",
        help=(
            "Optional comma/space-separated source-machine prefix, for example "
            "108566,634,163005,417786. Preparation fails if the mapping does not match it."
        ),
    )
    parser.add_argument(
        "--allow-sobol-identity-fallback",
        action="store_true",
        help=(
            "Allow the unsafe fallback theta[i] = Sobol CSV row i when no metadata NPZ/CSV is present. "
            "Do not use this for Adrian consolidated files unless sobol_global_row[i] == i + 1 has been verified."
        ),
    )
    parser.add_argument("--ell-start", type=int, default=80, help="Fallback first unbinned ell when using --sobol-csv.")
    parser.add_argument("--ell-stop", type=int, default=7979, help="Fallback last unbinned ell when using --sobol-csv.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--ell-min", type=float, default=80.0)
    parser.add_argument("--ell-max", type=float, default=7979.0)
    parser.add_argument("--ell-selection", choices=("center", "contained", "overlap"), default="center")
    parser.add_argument("--bin-weighting", default="2ell_plus_1")
    parser.add_argument("--obs-index", type=int, default=-1)
    parser.add_argument("--test-last-n", type=int, default=100)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional debug limit; 0 uses all rows.")
    parser.add_argument("--no-compress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    input_dir = resolve_path(args.input_dir, root)
    metadata_dir = resolve_path(args.metadata_dir, root) if args.metadata_dir else None
    metadata_npz = resolve_path(args.metadata_npz, root) if args.metadata_npz else None
    theta_path = resolve_path(args.theta_path, root) if args.theta_path else None
    sobol_csv = resolve_path(args.sobol_csv, root) if args.sobol_csv else None
    sobol_global_row_path = (
        resolve_path(args.sobol_global_row_path, root) if args.sobol_global_row_path else None
    )
    expected_sobol_prefix = parse_expected_sobol_prefix(args.expected_sobol_prefix)
    tag = ell_tag(args.ell_min, args.ell_max)
    output_dir = (
        resolve_path(args.output_dir, root)
        if args.output_dir
        else root / "SBI_analysis" / "data_for_cluster" / f"adrian_so_sbi_cases_{tag}_dataset_row"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if metadata_npz is not None and not metadata_npz.is_file():
        raise FileNotFoundError(f"Metadata NPZ not found: {metadata_npz}")
    if metadata_dir is not None and not metadata_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")
    if theta_path is not None and not theta_path.is_file():
        raise FileNotFoundError(f"Theta path not found: {theta_path}")
    if sobol_csv is not None and not sobol_csv.is_file():
        raise FileNotFoundError(f"Sobol CSV not found: {sobol_csv}")
    if sobol_global_row_path is not None and not sobol_global_row_path.is_file():
        raise FileNotFoundError(f"sobol_global_row path not found: {sobol_global_row_path}")
    if sobol_global_row_path is not None and sobol_csv is None:
        raise ValueError("--sobol-global-row-path requires --sobol-csv to reconstruct theta")
    if int(args.chunk_rows) <= 0:
        raise ValueError("--chunk-rows must be positive")

    products = normalize_products(args.products)
    save_func = np.savez if args.no_compress else np.savez_compressed

    index: dict[str, Any] = {
        "source": "adrian_consolidated_so_sbi_dataset",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ell_tag": tag,
        "ell_min": float(args.ell_min),
        "ell_max": float(args.ell_max),
        "ell_selection": args.ell_selection,
        "bin_weighting": args.bin_weighting,
        "obs_source": "dataset-row",
        "cases": {},
        "missing_products": [],
    }

    reference_metadata: dict[str, Any] | None = None
    fallback_metadata_cache: dict[tuple[int, ...], dict[str, Any]] = {}
    for product in products:
        cl_path = product_cl_path(input_dir, product)
        if not cl_path.is_file():
            message = f"Missing product C_ell file: {cl_path}"
            if args.allow_missing:
                print(f"Skipping {product}: {message}")
                index["missing_products"].append(product)
                continue
            raise FileNotFoundError(message)

        cl_probe = np.load(cl_path, mmap_mode="r")
        if cl_probe.ndim != 2:
            raise ValueError(f"{cl_path} must be 2D, got {cl_probe.shape}")
        cl_n_rows = int(cl_probe.shape[0])
        cl_n_ell = int(cl_probe.shape[1])
        del cl_probe

        n_rows = min(int(args.max_rows), cl_n_rows) if int(args.max_rows) > 0 else cl_n_rows
        metadata_path = product_metadata_path(input_dir, product, metadata_npz, metadata_dir)
        metadata_csv_path = product_metadata_csv_path(input_dir, product, metadata_dir)
        if metadata_path is not None:
            metadata = read_metadata(metadata_path, theta_path)
        elif metadata_csv_path is not None:
            print(
                f"No metadata NPZ found for {product}; using metadata CSV {metadata_csv_path} "
                "for theta/sobol_global_row."
            )
            metadata = read_metadata_csv(
                metadata_csv_path,
                n_rows=n_rows,
                n_ell=cl_n_ell,
                ell_start=int(args.ell_start),
                ell_stop=int(args.ell_stop),
                sobol_csv=sobol_csv,
            )
        elif sobol_csv is not None and sobol_global_row_path is not None:
            cache_key = (cl_n_rows, n_rows, cl_n_ell)
            if cache_key not in fallback_metadata_cache:
                print(
                    f"No metadata NPZ/CSV found for {product}; reconstructing theta from "
                    f"{sobol_global_row_path} and {sobol_csv}."
                )
                fallback_metadata_cache[cache_key] = read_sobol_mapping_metadata(
                    sobol_csv=sobol_csv,
                    sobol_global_row_path=sobol_global_row_path,
                    cl_n_rows=cl_n_rows,
                    n_rows=n_rows,
                    n_ell=cl_n_ell,
                    ell_start=int(args.ell_start),
                    ell_stop=int(args.ell_stop),
                    expected_prefix=expected_sobol_prefix,
                )
            metadata = fallback_metadata_cache[cache_key]
        elif sobol_csv is not None and args.allow_sobol_identity_fallback:
            cache_key = (n_rows, cl_n_ell)
            if cache_key not in fallback_metadata_cache:
                print(
                    f"No metadata NPZ found for {product}; using Sobol CSV {sobol_csv} "
                    f"and fallback ell={args.ell_start}..{args.ell_stop}. "
                    "This assumes C_ell row i corresponds to Sobol CSV row i."
                )
                fallback_metadata_cache[cache_key] = read_sobol_csv_metadata(
                    sobol_csv,
                    n_rows=n_rows,
                    n_ell=cl_n_ell,
                    ell_start=int(args.ell_start),
                    ell_stop=int(args.ell_stop),
                )
            metadata = fallback_metadata_cache[cache_key]
        else:
            raise FileNotFoundError(
                f"Could not find metadata NPZ for product={product!r}. Expected "
                f"{input_dir / f'sbi_{product}.npz'} or metadata CSV "
                f"{input_dir / f'sbi_{product}_metadata.csv'}. "
                "Pass --sobol-global-row-path with the complete array-index -> Sobol-row permutation, "
                "or pass --metadata-npz / --metadata-dir. The textual ordering rule and a Sobol CSV do "
                "not contain that permutation. The identity-row fallback remains disabled because "
                "Adrian's consolidated arrays are filesystem-scan ordered."
            )

        if reference_metadata is None:
            reference_metadata = metadata
        else:
            if not np.array_equal(metadata["theta_columns"], reference_metadata["theta_columns"]):
                raise ValueError(f"Theta columns differ for {product}: {metadata_path}")
            if not np.allclose(metadata["ell_unbinned"], reference_metadata["ell_unbinned"]):
                raise ValueError(f"ell differs for {product}: {metadata_path}")
            if metadata["theta"].shape[1] != reference_metadata["theta"].shape[1]:
                raise ValueError(f"theta shape differs for {product}: {metadata_path}")
            if not np.array_equal(metadata["sobol_global_row"], reference_metadata["sobol_global_row"]):
                raise ValueError(f"sobol_global_row ordering differs for {product}")

        theta_full = metadata["theta"]
        n_rows_full = int(theta_full.shape[0])
        if n_rows > n_rows_full:
            raise ValueError(f"Need {n_rows} theta rows for {product}, but metadata contains only {n_rows_full}")
        theta = np.ascontiguousarray(theta_full[:n_rows], dtype=np.float32)

        ell_unbinned = metadata["ell_unbinned"]
        bins = make_delta200_bins(ell_unbinned, args.bin_weighting)
        mask = ell_mask(
            bins["ell"],
            bins["bin_ell_min"],
            bins["bin_ell_max"],
            args.ell_min,
            args.ell_max,
            args.ell_selection,
        )
        selected = np.flatnonzero(mask)
        matrix = np.ascontiguousarray(bins["cl_to_dl_matrix"][:, selected], dtype=np.float32)

        print("")
        print(f"Preparing product: {product}")
        print(f"  C_ell path: {cl_path}")
        print(f"  metadata: {metadata_path if metadata_path is not None else 'none'}")
        print(f"  metadata CSV: {metadata_csv_path if metadata_csv_path is not None else 'none'}")
        if metadata_path is None and sobol_csv is not None:
            print(f"  Sobol CSV: {sobol_csv}")
        print(f"  theta source: {metadata.get('theta_source', 'unknown')}")
        if metadata.get("sobol_global_row_validation"):
            print(f"  ordering validation: {metadata['sobol_global_row_validation']}")
        print(f"  rows: {n_rows}")
        print(f"  unbinned ell: {ell_unbinned.size}")
        print(f"  selected bins: {selected.size}")

        x = bin_product_cl(
            cl_path,
            matrix,
            n_rows=n_rows,
            n_ell=int(ell_unbinned.size),
            chunk_rows=int(args.chunk_rows),
        )
        obs_index = resolve_obs_index(args.obs_index, n_rows)
        test_last_n = int(args.test_last_n)
        if test_last_n < 0:
            raise ValueError("--test-last-n must be non-negative")
        if test_last_n > n_rows:
            raise ValueError(f"--test-last-n={test_last_n} exceeds dataset rows={n_rows}")
        test_indices = np.arange(n_rows - test_last_n, n_rows, dtype=np.int64) if test_last_n else np.empty(0, dtype=np.int64)
        sobol_global_row = metadata["sobol_global_row"]
        sobol_payload = (
            np.ascontiguousarray(sobol_global_row[:n_rows], dtype=np.int64)
            if sobol_global_row is not None
            else np.empty(0, dtype=np.int64)
        )

        metadata_payload = {
            "product": product,
            "source_cl_path": str(cl_path),
            "source_metadata_path": str(metadata_path) if metadata_path is not None else "",
            "source_metadata_csv_path": str(metadata_csv_path) if metadata_csv_path is not None else "",
            "source_sobol_csv_path": str(sobol_csv) if metadata_path is None and sobol_csv is not None else "",
            "source_sobol_global_row_path": metadata.get("sobol_global_row_path", ""),
            "sobol_global_row_validation": metadata.get("sobol_global_row_validation", {}),
            "theta_source": metadata.get("theta_source", "unknown"),
            "n_rows": n_rows,
            "n_ell_unbinned": int(ell_unbinned.size),
            "n_selected_bins": int(selected.size),
            "ell_min": float(args.ell_min),
            "ell_max": float(args.ell_max),
            "ell_selection": args.ell_selection,
            "bin_weighting": args.bin_weighting,
            "statistic": "weighted mean of linear D_ell; x equals binned D_ell",
            "obs_source": "dataset-row",
            "obs_index": int(obs_index),
            "test_last_n": int(test_last_n),
        }

        output_path = output_dir / f"so_{product}_{tag}_sbi_run.npz"
        payload = {
            "theta": theta,
            "x": x,
            "obs": np.ascontiguousarray(x[obs_index], dtype=np.float32),
            "obs_theta": np.ascontiguousarray(theta[obs_index], dtype=np.float32),
            "obs_index": np.asarray(obs_index, dtype=np.int64),
            "obs_source": np.asarray("dataset-row"),
            "obs_profile_path": np.asarray(""),
            "test_indices": test_indices,
            "ell": np.ascontiguousarray(bins["ell"][mask], dtype=np.float32),
            "ell_binned": np.ascontiguousarray(bins["ell"][mask], dtype=np.float32),
            "ell_unbinned": np.ascontiguousarray(ell_unbinned, dtype=np.float32),
            "bin_counts": np.ascontiguousarray(bins["bin_counts"][mask], dtype=np.int64),
            "bin_ell_min": np.ascontiguousarray(bins["bin_ell_min"][mask], dtype=np.float32),
            "bin_ell_max": np.ascontiguousarray(bins["bin_ell_max"][mask], dtype=np.float32),
            "prior_low": metadata["prior_low"],
            "prior_high": metadata["prior_high"],
            "param_names": metadata["param_names"],
            "theta_columns": metadata["theta_columns"],
            "sobol_global_row": sobol_payload,
            "case_name": np.asarray(product),
            "source_case": np.asarray(product),
            "product": np.asarray(product),
            "source_cl_path": np.asarray(str(cl_path)),
            "source_metadata_path": np.asarray(str(metadata_path) if metadata_path is not None else ""),
            "source_metadata_csv_path": np.asarray(str(metadata_csv_path) if metadata_csv_path is not None else ""),
            "source_sobol_csv_path": np.asarray(str(sobol_csv) if metadata_path is None and sobol_csv is not None else ""),
            "source_sobol_global_row_path": np.asarray(str(metadata.get("sobol_global_row_path", ""))),
            "sobol_global_row_validation_json": np.asarray(
                json.dumps(jsonable(metadata.get("sobol_global_row_validation", {})), sort_keys=True)
            ),
            "theta_source": np.asarray(str(metadata.get("theta_source", "unknown"))),
            "ell_min": np.asarray(float(args.ell_min), dtype=np.float32),
            "ell_max": np.asarray(float(args.ell_max), dtype=np.float32),
            "ell_selection": np.asarray(args.ell_selection),
            "metadata_json": np.asarray(json.dumps(jsonable(metadata_payload), sort_keys=True)),
        }
        save_func(output_path, **payload)

        index["cases"][product] = {
            "path": str(output_path),
            "product": product,
            "source_cl_path": str(cl_path),
            "source_metadata_path": str(metadata_path) if metadata_path is not None else "",
            "source_metadata_csv_path": str(metadata_csv_path) if metadata_csv_path is not None else "",
            "source_sobol_csv_path": str(sobol_csv) if metadata_path is None and sobol_csv is not None else "",
            "source_sobol_global_row_path": metadata.get("sobol_global_row_path", ""),
            "sobol_global_row_validation": metadata.get("sobol_global_row_validation", {}),
            "theta_source": metadata.get("theta_source", "unknown"),
            "n_rows": n_rows,
            "x_shape": list(x.shape),
            "obs_index": int(obs_index),
            "obs_source": "dataset-row",
            "test_last_n": int(test_last_n),
        }
        print(f"Wrote {output_path}  x_shape={x.shape}")

        del x

    if not index["cases"]:
        raise RuntimeError("No product datasets were written.")

    index_path = output_dir / "case_dataset_index.json"
    write_json(index_path, index)
    print(f"Wrote index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
