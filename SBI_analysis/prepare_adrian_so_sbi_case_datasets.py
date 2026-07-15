#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    col_to_idx = {name: idx for idx, name in enumerate(header)}
    missing = [str(name) for name in PARAM_NAMES if str(name) not in col_to_idx]
    if missing:
        raise KeyError(f"Sobol CSV {path} is missing columns {missing}; header={header}")

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
        "sobol_csv_path": str(path),
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
    fallback_metadata_cache: dict[tuple[int, int], dict[str, Any]] = {}
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
        if metadata_path is not None:
            metadata = read_metadata(metadata_path, theta_path)
        elif sobol_csv is not None:
            cache_key = (n_rows, cl_n_ell)
            if cache_key not in fallback_metadata_cache:
                print(
                    f"No metadata NPZ found for {product}; using Sobol CSV {sobol_csv} "
                    f"and fallback ell={args.ell_start}..{args.ell_stop}."
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
                f"{input_dir / f'sbi_{product}.npz'} or pass --metadata-npz / --metadata-dir. "
                "Since you only have *_cl.npy files, pass --sobol-csv /home/kristero10/tSZ_data/battaglia_sobol_1048576.csv."
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
        if metadata_path is None and sobol_csv is not None:
            print(f"  Sobol CSV: {sobol_csv}")
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
            "source_sobol_csv_path": str(sobol_csv) if metadata_path is None and sobol_csv is not None else "",
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
            "source_sobol_csv_path": np.asarray(str(sobol_csv) if metadata_path is None and sobol_csv is not None else ""),
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
            "source_sobol_csv_path": str(sobol_csv) if metadata_path is None and sobol_csv is not None else "",
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
