#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
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

@dataclass(frozen=True)
class BinningSpec:
    bin_min: np.ndarray
    bin_max: np.ndarray
    ell: np.ndarray
    weights: list[np.ndarray]
    members: list[np.ndarray]
    matrix: np.ndarray


@dataclass(frozen=True)
class ProfileRecord:
    path: Path
    label: str
    deproj: int
    sobol_basename: str
    sobol_row: int


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


def make_so_delta200_spec(source_ell: np.ndarray, weighting: str = "2ell_plus_1") -> BinningSpec:
    source_ell = np.asarray(source_ell, dtype=np.float64).reshape(-1)
    edges = np.r_[np.arange(80, 7881, 200), 7979].astype(np.int64)
    bin_min = edges[:-1].copy()
    bin_max = edges[1:].copy()
    bin_max[:-1] -= 1

    members: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    centers: list[float] = []
    matrix = np.zeros((source_ell.size, bin_min.size), dtype=np.float64)
    for lo, hi in zip(bin_min, bin_max):
        idx = np.flatnonzero((source_ell >= float(lo)) & (source_ell <= float(hi)))
        if idx.size == 0:
            raise ValueError(
                f"No unbinned ell values found for SO bin {lo}-{hi}; "
                f"source range is {source_ell.min()}-{source_ell.max()}."
            )
        w = bin_weights(source_ell[idx], weighting)
        normalized_w = w / np.sum(w)
        matrix[idx, len(members)] = normalized_w
        members.append(idx)
        weights.append(w)
        centers.append(float(np.average(source_ell[idx], weights=w)))

    return BinningSpec(
        bin_min=bin_min,
        bin_max=bin_max,
        ell=np.asarray(centers, dtype=np.float32),
        weights=weights,
        members=members,
        matrix=matrix,
    )


def bin_last_axis(values: np.ndarray, spec: BinningSpec) -> np.ndarray:
    return apply_bin_matrix(values, spec.matrix)


def apply_bin_matrix(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.ascontiguousarray(values @ matrix, dtype=np.float32)


def parse_profile_filename(path: Path) -> ProfileRecord | None:
    name = path.name
    if "halfdome_fullsky_masked_no_noise_cl" in name:
        label = "no_noise"
    elif "halfdome_fullsky_masked_goal_noise_cross_cl" in name:
        label = "goal"
    elif "halfdome_fullsky_masked_baseline_noise_cross_cl" in name:
        label = "baseline"
    else:
        return None

    deproj_match = re.search(r"_deproj(\d+)_", name)
    sobol_match = re.search(r"sobol_battaglia_(?P<tag>.+?)_row(?P<row>\d+)", name)
    if deproj_match is None or sobol_match is None:
        return None

    sobol_basename = "battaglia_" + sobol_match.group("tag")
    return ProfileRecord(
        path=path,
        label=label,
        deproj=int(deproj_match.group(1)),
        sobol_basename=sobol_basename,
        sobol_row=int(sobol_match.group("row")),
    )


def discover_records(input_dirs: list[Path], extensions: tuple[str, ...]) -> list[ProfileRecord]:
    records: list[ProfileRecord] = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        for path in sorted(input_dir.rglob("*")):
            if path.suffix.lower() not in extensions:
                continue
            record = parse_profile_filename(path)
            if record is not None:
                records.append(record)
    return records


def logical_case(label: str, deproj: int) -> str:
    if label == "no_noise":
        return "no_noise"
    return f"{label}_deproj{int(deproj)}"


def requested_cases(deprojections: list[int]) -> list[str]:
    cases = ["no_noise"]
    for deproj in deprojections:
        for label in ("goal", "baseline"):
            cases.append(logical_case(label, deproj))
    return cases


def build_profile_index(
    records: list[ProfileRecord],
    deprojections: list[int],
    no_noise_deproj: int | None,
) -> dict[tuple[str, int], dict[str, ProfileRecord]]:
    allowed_deproj = set(int(value) for value in deprojections)
    requested = set(requested_cases(deprojections))
    index: dict[tuple[str, int], dict[str, ProfileRecord]] = {}

    for record in records:
        if record.label == "no_noise":
            if no_noise_deproj is not None and record.deproj != no_noise_deproj:
                continue
            if no_noise_deproj is None and record.deproj not in allowed_deproj:
                continue
            case = "no_noise"
        else:
            if record.deproj not in allowed_deproj:
                continue
            case = logical_case(record.label, record.deproj)
            if case not in requested:
                continue

        key = (record.sobol_basename, record.sobol_row)
        row = index.setdefault(key, {})
        previous = row.get(case)
        if previous is not None and previous.path != record.path:
            raise ValueError(
                f"Duplicate profile for {key} case {case}:\n"
                f"  {previous.path}\n"
                f"  {record.path}"
            )
        row[case] = record

    return index


def load_sobol_csv(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Sobol CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Sobol CSV has no header: {path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        missing = [name for name in PARAM_NAMES if name not in fieldnames]
        if missing:
            raise KeyError(f"{path} is missing parameter columns: {missing}")
        rows = [[float(row[name]) for name in PARAM_NAMES] for row in reader]
    if not rows:
        raise ValueError(f"Sobol CSV has no data rows: {path}")
    return list(PARAM_NAMES), np.asarray(rows, dtype=np.float32)


def find_sobol_csv(sobol_dir: Path, basename: str) -> Path | None:
    direct = sobol_dir / f"{basename}.csv"
    if direct.is_file():
        return direct
    matches = sorted(sobol_dir.rglob(f"{basename}.csv"))
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {basename}.csv under {sobol_dir}: {matches}")
    return matches[0] if matches else None


def resolve_sobol_csv(sobol_dir: Path, basename: str) -> Path:
    path = find_sobol_csv(sobol_dir, basename)
    if path is not None:
        return path
    raise FileNotFoundError(
        f"Could not find {basename}.csv under {sobol_dir}. "
        "For split 32768 files this is usually under "
        "Sobol_tSZ/splits_battaglia_sobol_32768. If you only have the full "
        "CSV, use a filename like battaglia_sobol_32768.csv."
    )


def theta_for_key(
    key: tuple[str, int],
    sobol_dir: Path,
    cache: dict[str, tuple[list[str], np.ndarray]],
    rows_per_split: int,
) -> np.ndarray:
    basename, row = key
    source_basename = basename
    table_row = int(row)
    csv_path = find_sobol_csv(sobol_dir, basename)

    if csv_path is None:
        split_match = re.fullmatch(r"(battaglia_sobol_\d+)_(\d+)", basename)
        if split_match is not None:
            full_basename = split_match.group(1)
            csv_path = find_sobol_csv(sobol_dir, full_basename)
            if csv_path is not None:
                source_basename = full_basename
                table_row = global_row_for_key(key, rows_per_split)

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find {basename}.csv under {sobol_dir}, and could not "
            f"fall back to an unsplit CSV for {basename}."
        )

    cache_key = str(csv_path)
    if cache_key not in cache:
        cache[cache_key] = load_sobol_csv(csv_path)
    _, table = cache[cache_key]
    if table_row < 1 or table_row > table.shape[0]:
        raise IndexError(
            f"Row {table_row} is outside {source_basename}.csv with {table.shape[0]} rows "
            f"while resolving profile key {basename} row {row}"
        )
    return table[table_row - 1]


def split_index_from_basename(basename: str) -> int | None:
    match = re.search(r"battaglia_sobol_\d+_(\d+)$", basename)
    if match is None:
        return None
    return int(match.group(1))


def global_row_for_key(key: tuple[str, int], rows_per_split: int) -> int:
    basename, row = key
    split_index = split_index_from_basename(basename)
    if split_index is None:
        return int(row)
    return (int(split_index) - 1) * int(rows_per_split) + int(row)


def complete_keys(
    index: dict[tuple[str, int], dict[str, ProfileRecord]],
    cases: list[str],
) -> tuple[list[tuple[str, int]], dict[str, int]]:
    missing_counts = {case: 0 for case in cases}
    complete: list[tuple[str, int]] = []

    for key, row in index.items():
        missing = [case for case in cases if case not in row]
        if missing:
            for case in missing:
                missing_counts[case] += 1
        else:
            complete.append(key)

    return complete, missing_counts


def read_profile(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1), None
    if suffix in {".txt", ".csv", ".dat"}:
        raw = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return raw[:, 1].astype(np.float32), raw[:, 0].astype(np.float32)
        return raw.reshape(-1).astype(np.float32), None
    if suffix in {".fits", ".fit", ".fts"}:
        return read_fits_profile(path)
    raise ValueError(f"Unsupported profile extension: {path}")


def read_fits_profile(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        from astropy.io import fits
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Reading FITS profiles requires astropy. Use .npy inputs or install astropy for {path}."
        ) from exc

    profile_candidates = []
    ell_candidates = []
    with fits.open(path, memmap=False) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue
            if getattr(data, "dtype", None) is not None and data.dtype.fields:
                for field in data.dtype.names or ():
                    arr = np.asarray(data[field]).squeeze()
                    if arr.size < 2 or not np.issubdtype(arr.dtype, np.number):
                        continue
                    key = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")
                    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                    if key in {"ell", "l", "multipole"}:
                        ell_candidates.append(arr)
                    elif key not in {"index", "row", "pixel"}:
                        priority = 0 if "cl" in key or "power" in key else 1
                        profile_candidates.append((priority, hdu_index, arr))
            else:
                arr = np.asarray(data).squeeze()
                if arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                    profile_candidates.append((2, hdu_index, np.asarray(arr, dtype=np.float32).reshape(-1)))

    if not profile_candidates:
        raise ValueError(f"No profile-like numeric array found in {path}")
    profile_candidates.sort(key=lambda item: (item[0], -item[2].size))
    profile = profile_candidates[0][2]
    ell = ell_candidates[0] if ell_candidates and ell_candidates[0].size == profile.size else None
    return profile, ell


def write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True)


def make_memmap(path: Path, shape: tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Combine SO no-noise/goal/baseline C_ell outputs into one SBI-ready NPZ. "
            "The file stores one no-noise dataset and goal/baseline datasets for the requested deprojections."
        )
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=[str(root / "tSZ_visuals" / "outputs" / "so_noise_sobol32768_rows1_2_local")],
        help="Directories containing SO-noise profile files.",
    )
    parser.add_argument(
        "--sobol-dir",
        default=str(root / "Sobol_tSZ"),
        help="Directory containing battaglia_sobol_*.csv files.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "so_multi_case_delta200_sbi_dataset.npz"),
    )
    parser.add_argument("--deprojections", nargs="+", type=int, default=[0, 2])
    parser.add_argument(
        "--no-noise-deproj",
        type=int,
        default=None,
        help="Which deprojection-tagged no-noise file to use. Default accepts the first requested deprojection found.",
    )
    parser.add_argument(
        "--default-case",
        default="no_noise",
        help="Case copied to the SBI-compatible x key. Example: no_noise, goal_deproj0, baseline_deproj2.",
    )
    parser.add_argument(
        "--obs-index",
        type=int,
        default=-1,
        help="Row index in the sorted combined dataset to copy to obs. Use --no-obs to skip.",
    )
    parser.add_argument("--no-obs", action="store_true")
    parser.add_argument("--rows-per-split", type=int, default=128)
    parser.add_argument("--bin-weighting", default="2ell_plus_1")
    parser.add_argument(
        "--floor-dl",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".npy"],
        help="Profile extensions to read. Use .npy for cluster SO outputs; .txt/.fits are also supported.",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="Skip rows missing one or more requested cases.")
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed. Faster, larger output.",
    )
    parser.add_argument(
        "--skip-unbinned-cl",
        action="store_true",
        help=(
            "Do not store the large unbinned cl_<case> arrays in the final NPZ. "
            "This is much faster and smaller when only SBI-ready binned D_ell vectors are needed."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for --no-compress --skip-unbinned-cl.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary memmap .npy files used while assembling the NPZ.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fast:
        args.no_compress = True
        args.skip_unbinned_cl = True

    root = repo_root()
    input_dirs = [resolve_path(path, root) for path in args.input_dirs]
    sobol_dir = resolve_path(args.sobol_dir, root)
    output_path = resolve_path(args.output, root)
    extensions = tuple((ext if ext.startswith(".") else f".{ext}").lower() for ext in args.extensions)
    cases = requested_cases([int(value) for value in args.deprojections])

    if args.default_case not in cases:
        raise ValueError(f"--default-case must be one of {cases}, got {args.default_case!r}")

    print("Discovering SO profiles...")
    records = discover_records(input_dirs, extensions)
    if not records:
        raise FileNotFoundError(f"No SO profile records found under {input_dirs} with extensions {extensions}")

    deprojections = [int(value) for value in args.deprojections]
    no_noise_deproj = args.no_noise_deproj if args.no_noise_deproj is not None else deprojections[0]
    index = build_profile_index(records, deprojections, no_noise_deproj)
    keys, missing_counts = complete_keys(index, cases)
    if not keys:
        raise ValueError(
            "No complete rows found for requested cases. "
            f"Requested cases={cases}; missing counts among discovered rows={missing_counts}."
        )
    if any(missing_counts.values()) and not args.allow_incomplete:
        raise ValueError(
            "Some discovered rows are missing requested cases. "
            f"Requested cases={cases}; missing counts={missing_counts}. "
            "Use --allow-incomplete to skip incomplete rows."
        )

    keys = sorted(keys, key=lambda key: global_row_for_key(key, args.rows_per_split))
    print(f"Discovered profile files: {len(records)}")
    print(f"Complete aligned rows: {len(keys)}")
    if any(missing_counts.values()):
        print(f"Missing-case counts among incomplete rows: {missing_counts}")

    first_profile, first_ell = read_profile(index[keys[0]][cases[0]].path)
    ell_unbinned = (
        np.asarray(first_ell, dtype=np.float32).reshape(-1)
        if first_ell is not None
        else np.arange(first_profile.size, dtype=np.float32)
    )
    n_rows = len(keys)
    n_ell = ell_unbinned.size
    spec = make_so_delta200_spec(ell_unbinned, args.bin_weighting)
    dl_factor = ell_unbinned.astype(np.float64) * (ell_unbinned.astype(np.float64) + 1.0) / (2.0 * np.pi)
    dl_bin_matrix = dl_factor[:, None] * spec.matrix
    n_bins = spec.ell.size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_", dir=str(output_path.parent)))
    print(f"Temporary array directory: {temp_dir}")

    theta = make_memmap(temp_dir / "theta.npy", (n_rows, PARAM_NAMES.size))
    cl_cases = (
        {case: make_memmap(temp_dir / f"cl_{case}.npy", (n_rows, n_ell)) for case in cases}
        if not args.skip_unbinned_cl
        else {}
    )
    cl_binned_cases = {case: make_memmap(temp_dir / f"cl_binned_{case}.npy", (n_rows, n_bins)) for case in cases}
    dl_binned_cases = {case: make_memmap(temp_dir / f"dl_binned_{case}.npy", (n_rows, n_bins)) for case in cases}
    x_cases = {case: make_memmap(temp_dir / f"x_{case}.npy", (n_rows, n_bins)) for case in cases}

    metadata_rows: list[dict[str, Any]] = []
    sobol_cache: dict[str, tuple[list[str], np.ndarray]] = {}

    for out_idx, key in enumerate(keys):
        theta[out_idx] = theta_for_key(key, sobol_dir, sobol_cache, args.rows_per_split)
        row_meta: dict[str, Any] = {
            "row_index": out_idx,
            "sobol_basename": key[0],
            "sobol_row": key[1],
            "sobol_global_row": global_row_for_key(key, args.rows_per_split),
        }

        for case in cases:
            record = index[key][case]
            cl, ell = read_profile(record.path)
            if cl.size != n_ell:
                raise ValueError(f"{record.path} has length {cl.size}, expected {n_ell}")
            if ell is not None and not np.allclose(np.asarray(ell).reshape(-1), ell_unbinned):
                raise ValueError(f"{record.path} has an ell grid different from the first profile.")

            cl = np.ascontiguousarray(cl, dtype=np.float32)
            cl_binned = bin_last_axis(cl, spec)
            dl_binned = apply_bin_matrix(cl, dl_bin_matrix)

            if cl_cases:
                cl_cases[case][out_idx] = cl
            cl_binned_cases[case][out_idx] = cl_binned
            dl_binned_cases[case][out_idx] = dl_binned
            x_cases[case][out_idx] = dl_binned
            row_meta[f"{case}_path"] = str(record.path)
            row_meta[f"{case}_source_deproj"] = record.deproj

        metadata_rows.append(row_meta)
        if (out_idx + 1) % 100 == 0 or out_idx + 1 == n_rows:
            print(f"Processed {out_idx + 1}/{n_rows} rows", flush=True)

    for mmap in [theta, *cl_cases.values(), *cl_binned_cases.values(), *dl_binned_cases.values(), *x_cases.values()]:
        mmap.flush()

    obs_index = int(args.obs_index)
    if obs_index < 0:
        obs_index = n_rows + obs_index
    if not args.no_obs and not (0 <= obs_index < n_rows):
        raise IndexError(f"--obs-index {args.obs_index} resolves to {obs_index}, outside 0..{n_rows - 1}")

    prior_low = np.asarray([SOBOL_PRIOR_BOUNDS[name][0] for name in PARAM_NAMES], dtype=np.float32)
    prior_high = np.asarray([SOBOL_PRIOR_BOUNDS[name][1] for name in PARAM_NAMES], dtype=np.float32)
    binning_summary = {
        "name": "SO",
        "delta_ell": 200,
        "n_bins": int(n_bins),
        "statistic": "weighted mean of linear D_ell; x_* equals dl_binned_*",
        "weighting": args.bin_weighting,
        "bin_ell_min": spec.bin_min.astype(int).tolist(),
        "bin_ell_max": spec.bin_max.astype(int).tolist(),
    }
    summary = {
        "input_dirs": [str(path) for path in input_dirs],
        "sobol_dir": str(sobol_dir),
        "output": str(output_path),
        "n_rows": int(n_rows),
        "n_ell_unbinned": int(n_ell),
        "n_bins": int(n_bins),
        "cases": cases,
        "no_noise_source_deproj": int(no_noise_deproj),
        "default_case": args.default_case,
        "obs_index": None if args.no_obs else int(obs_index),
        "obs_case": None if args.no_obs else args.default_case,
        "binning": binning_summary,
        "x_representation": "binned_linear_d_ell",
        "stores_unbinned_cl": not args.skip_unbinned_cl,
        "compressed": not args.no_compress,
    }

    payload: dict[str, Any] = {
        "theta": theta,
        "param_names": PARAM_NAMES,
        "theta_columns": PARAM_NAMES,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "case_names": np.asarray(cases, dtype=str),
        "default_case": np.asarray(args.default_case),
        "x": x_cases[args.default_case],
        "ell": np.ascontiguousarray(spec.ell, dtype=np.float32),
        "ell_binned": np.ascontiguousarray(spec.ell, dtype=np.float32),
        "ell_unbinned": np.ascontiguousarray(ell_unbinned, dtype=np.float32),
        "bin_counts": np.ascontiguousarray(spec.bin_max - spec.bin_min + 1, dtype=np.int64),
        "bin_ell_min": np.ascontiguousarray(spec.bin_min, dtype=np.float32),
        "bin_ell_max": np.ascontiguousarray(spec.bin_max, dtype=np.float32),
        "binning_json": np.asarray(json.dumps(binning_summary, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(summary, sort_keys=True)),
    }
    if not args.no_obs:
        payload["obs"] = np.ascontiguousarray(x_cases[args.default_case][obs_index], dtype=np.float32)
        payload["obs_theta"] = np.ascontiguousarray(theta[obs_index], dtype=np.float32)
        payload["obs_index"] = np.asarray(obs_index, dtype=np.int64)

    for case in cases:
        payload[f"x_{case}"] = x_cases[case]
        payload[f"cl_binned_{case}"] = cl_binned_cases[case]
        payload[f"dl_binned_{case}"] = dl_binned_cases[case]
        if cl_cases:
            payload[f"cl_{case}"] = cl_cases[case]

    save_func = np.savez if args.no_compress else np.savez_compressed
    print(f"Writing combined dataset: {output_path}")
    save_func(output_path, **payload)

    metadata_csv = output_path.with_name(output_path.stem + "_metadata.csv")
    summary_json = output_path.with_suffix(".json")
    write_metadata_csv(metadata_csv, metadata_rows)
    write_json(summary_json, summary)

    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_csv}")
    print(f"Wrote {summary_json}")
    print(f"theta shape: {theta.shape}")
    print(f"default x shape: {x_cases[args.default_case].shape}")
    if cl_cases:
        print(f"unbinned C_ell shape per case: {(n_rows, n_ell)}")
    else:
        print("unbinned C_ell arrays were skipped (--skip-unbinned-cl/--fast)")
    print(f"cases: {', '.join(cases)}")

    if args.keep_temp:
        print(f"Kept temporary arrays: {temp_dir}")
    else:
        del theta, cl_cases, cl_binned_cases, dl_binned_cases, x_cases
        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
