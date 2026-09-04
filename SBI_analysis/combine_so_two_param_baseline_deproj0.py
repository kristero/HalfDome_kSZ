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
    root = Path("/lustre/work/kristero10/adrian_two_param_so_baseline_deproj0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sobol-csv",
        type=Path,
        default=root / "design" / "battaglia_sobol_P0_beta_32768.csv",
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
    parser.add_argument("--ell-min", type=int, default=80)
    parser.add_argument("--ell-max", type=int, default=7979)
    parser.add_argument("--mask-seed", type=int, default=12345)
    parser.add_argument("--noise-seed-base", type=int, default=1_000_000)
    parser.add_argument("--test-last-n", type=int, default=1000)
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
                "mask_seed",
                "noise_seed",
                "status",
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
                expected_noise_seed = noise_seed_base + row
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
                    output = Path(record["output_path"])
                    if not output.is_file() or output.stat().st_size == 0:
                        failures.setdefault(row, []).append(
                            f"recorded success but output is missing: {output}"
                        )
                    else:
                        successes.setdefault(row, []).append(record)
                else:
                    failures.setdefault(row, []).append(
                        f"{manifest}: status={record['status']}, log={record['row_log']}"
                    )

    chosen: dict[int, dict[str, Any]] = {}
    for row, records in successes.items():
        output_paths = {str(Path(record["output_path"]).resolve()) for record in records}
        if len(output_paths) != 1:
            raise ValueError(
                f"Row {row} has successful manifests pointing to different outputs: "
                f"{sorted(output_paths)}"
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


def write_metadata_csv(
    path: Path,
    rows: list[int],
    records: dict[int, dict[str, Any]],
    theta_full: np.ndarray,
) -> None:
    fields = [
        "dataset_index",
        "sobol_row",
        "mask_seed",
        "noise_seed",
        *FULL_PARAMETERS,
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
                "mask_seed": record["mask_seed"],
                "noise_seed": record["noise_seed"],
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
    metadata_csv_path = dataset_path.with_name(dataset_path.stem + "_metadata.csv")
    manifest_json_path = dataset_path.with_name(dataset_path.stem + "_manifest.json")
    completion_path = output_dir / "combination_complete.json"

    outputs = [
        dataset_path,
        unbinned_path,
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

    records, failures = load_success_records(
        manifest_dir,
        n_expected,
        args.mask_seed,
        args.noise_seed_base,
    )
    missing_rows = sorted(set(range(1, n_expected + 1)).difference(records))
    if missing_rows and not args.allow_missing:
        preview = missing_rows[:20]
        raise RuntimeError(
            f"{len(missing_rows)} of {n_expected} rows lack a valid successful output. "
            f"First missing rows: {preview}"
        )
    selected_rows = sorted(records)
    if not selected_rows:
        raise RuntimeError("No valid generated spectra were found")

    first_ell, first_cl = select_cl(
        Path(records[selected_rows[0]]["output_path"]),
        args.ell_min,
        args.ell_max,
    )
    bins = make_bins(first_ell)
    n_rows = len(selected_rows)
    x = np.empty((n_rows, len(bins["indices"])), dtype=np.float32)
    unbinned = np.lib.format.open_memmap(
        unbinned_path,
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
        if not np.array_equal(ell, first_ell):
            raise ValueError(f"Row {row} uses a different ell grid")
        dell = cl * dl_factor
        unbinned[output_index] = dell.astype(np.float32)
        x[output_index] = bin_dell(dell, bins)
        if (output_index + 1) % 256 == 0 or output_index + 1 == n_rows:
            unbinned.flush()
            print(f"Processed {output_index + 1}/{n_rows}", flush=True)
    del unbinned

    row_index = np.asarray(selected_rows, dtype=np.int64) - 1
    theta_full = theta_csv[row_index].astype(np.float32)
    target_indices = [FULL_PARAMETERS.index(name) for name in TARGET_PARAMETERS]
    theta = theta_full[:, target_indices]
    prior_low = design["prior_low"].astype(np.float32)
    prior_high = design["prior_high"].astype(np.float32)
    sobol_rows = np.asarray(selected_rows, dtype=np.int64)
    noise_seeds = args.noise_seed_base + sobol_rows
    test_count = min(max(args.test_last_n, 0), n_rows)
    test_indices = np.arange(n_rows - test_count, n_rows, dtype=np.int64)

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
        "noise_seed_policy": "noise_seed_base + one_based_sobol_row",
        "noise_seed_base": args.noise_seed_base,
        "same_mask_all_rows": True,
        "independent_noise_all_rows": True,
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
    }

    np.savez_compressed(
        dataset_path,
        theta=theta,
        theta_full=theta_full,
        x=x,
        obs=x[-1],
        obs_theta=theta[-1],
        obs_index=np.asarray(n_rows - 1, dtype=np.int64),
        obs_source=np.asarray("dataset-row"),
        test_indices=test_indices,
        ell=bins["ell_binned"],
        ell_binned=bins["ell_binned"],
        ell_unbinned=first_ell.astype(np.float32),
        bin_counts=bins["bin_counts"],
        bin_ell_min=bins["bin_ell_min"],
        bin_ell_max=bins["bin_ell_max"],
        prior_low=prior_low,
        prior_high=prior_high,
        param_names=np.asarray(TARGET_PARAMETERS),
        theta_columns=np.asarray(TARGET_PARAMETERS),
        full_param_names=np.asarray(FULL_PARAMETERS),
        sobol_row=sobol_rows,
        sobol_global_row=sobol_rows,
        mask_seed=np.full(n_rows, args.mask_seed, dtype=np.int64),
        noise_seed=noise_seeds,
        case_name=np.asarray(PRODUCT),
        product=np.asarray(PRODUCT),
        source_sobol_csv_path=np.asarray(str(sobol_csv)),
        unbinned_dell_path=np.asarray(str(unbinned_path)),
        metadata_json=np.asarray(json.dumps(jsonable(metadata), sort_keys=True)),
    )
    write_metadata_csv(
        metadata_csv_path,
        selected_rows,
        records,
        theta_csv,
    )
    with manifest_json_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(metadata), handle, indent=2, sort_keys=True)
    with completion_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "complete": not missing_rows,
                "dataset": str(dataset_path),
                "unbinned_dell": str(unbinned_path),
                "n_rows": n_rows,
                "n_expected_rows": n_expected,
                "x_shape": list(x.shape),
                "theta_shape": list(theta.shape),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {dataset_path}")
    print(f"Wrote {unbinned_path}")
    print(f"Wrote {metadata_csv_path}")
    print(f"Wrote {manifest_json_path}")
    print(f"Wrote {completion_path}")
    print(f"theta shape: {theta.shape}")
    print(f"x shape: {x.shape}")
    print(f"unbinned D_ell shape: ({n_rows}, {first_ell.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
