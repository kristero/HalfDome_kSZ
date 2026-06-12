#!/usr/bin/env python3
"""
Combine y100-only Battaglia Sobol products into an SBI-ready dataset.

This is the single-lightcone version of combine_sbi_dataset.py. It discovers
generated C_l/profile files, matches them to Sobol/Battaglia parameters, and
writes aligned arrays plus metadata without requiring y102 files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from build_battaglia_emulator import (
    deduplicate_profile_records,
    load_profile_group,
    load_sobol_tables,
    mean_cl_at_each_ell,
    split_cli_values,
    split_comma_values,
)
from combine_sbi_dataset import add_global_sobol_rows, reorder_by_metadata, split_index_from_csv


DEFAULT_SOBOL_CSVS = [
    *(f"/home/kristero10/tSZ_data/battaglia_sobol_512_{idx}.csv" for idx in range(1, 5)),
    *(f"/home/kristero10/tSZ_data/battaglia_sobol_1024_{idx}.csv" for idx in range(5, 9)),
    *(f"/home/kristero10/tSZ_data/battaglia_sobol_2048_{idx}.csv" for idx in range(9, 17)),
]
DEFAULT_INCLUDE_REGEX = [
    (
        r"sobol_battaglia_sobol_512_[1-4]_row[0-9]+|"
        r"sobol_battaglia_sobol_1024_[5-8]_row[0-9]+|"
        r"sobol_battaglia_sobol_2048_(9|1[0-6])_row[0-9]+"
    )
]
DEFAULT_EXCLUDE_REGEX = [
    (
        r"sobol_battaglia_sobol_512_row[0-9]+|"
        r"sobol_battaglia_sobol_1024_[1-4]_row[0-9]+|"
        r"sobol_battaglia_sobol_2048_[1-8]_row[0-9]+|"
        r"sobol_battaglia_sobol_256"
    )
]


def combine_y100_records(y100_by_key, x_columns: list[str], ell):
    import numpy as np
    import pandas as pd

    rows = []
    theta_values = []
    cl_y100_values = []

    for key in sorted(y100_by_key):
        records = y100_by_key[key]
        params = records[0].params
        cl_y100 = mean_cl_at_each_ell(records, "y100")

        theta_values.append([params[col] for col in x_columns])
        cl_y100_values.append(cl_y100)

        row = {
            "param_key": "|".join(key),
            "y100_n": len(records),
            "y100_files": ";".join(str(record.path) for record in records),
            "sobol_csv": str(records[0].csv_path),
            "sobol_row": records[0].csv_row,
        }
        row.update(params)
        rows.append(row)

    metadata = pd.DataFrame(rows)
    return (
        np.asarray(theta_values, dtype=float),
        np.asarray(cl_y100_values, dtype=float),
        metadata,
        np.asarray(ell, dtype=float),
    )


def write_sbi_outputs(args, x_columns, ell, theta, cl_y100, metadata):
    import numpy as np

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compatibility aliases: for a y100-only dataset the "mean" is just y100.
    cl_mean = cl_y100
    cl_stack = cl_y100[:, None, :]
    cl_concat = cl_y100
    cl_log10_y100 = np.log10(np.maximum(cl_y100, args.target_floor))
    cl_log10_mean = cl_log10_y100
    cl_log10_concat = cl_log10_y100

    dataset_path = output_dir / args.dataset_name
    np.savez_compressed(
        dataset_path,
        theta=theta,
        theta_columns=np.asarray(x_columns, dtype=str),
        ell=ell,
        cl_mean=cl_mean,
        cl_y100=cl_y100,
        cl_stack=cl_stack,
        cl_concat=cl_concat,
        cl_log10_mean=cl_log10_mean,
        cl_log10_y100=cl_log10_y100,
        cl_log10_concat=cl_log10_concat,
        seed_ids=np.asarray([100], dtype=int),
        target_floor=np.asarray(args.target_floor, dtype=float),
        param_key=metadata["param_key"].astype(str).to_numpy(),
        sobol_global_row=metadata["sobol_global_row"].to_numpy(dtype=int),
        sobol_split=metadata["sobol_split"].to_numpy(dtype=int),
        sobol_row=metadata["sobol_row"].to_numpy(dtype=int),
    )

    metadata_path = output_dir / args.metadata_name
    metadata.to_csv(metadata_path, index=False)

    manifest_path = output_dir / args.manifest_name
    manifest = {
        "dataset_npz": str(dataset_path),
        "metadata_csv": str(metadata_path),
        "n_parameter_points": int(theta.shape[0]),
        "n_parameters": int(theta.shape[1]),
        "n_ell": int(ell.shape[0]),
        "theta_columns": list(x_columns),
        "realizations": ["y100"],
        "arrays": {
            "theta": "Battaglia parameter matrix, shape (n_points, n_parameters)",
            "cl_y100": "Seed/lightcone 100 C_l/profile arrays, shape (n_points, n_ell)",
            "cl_mean": "Compatibility alias for cl_y100, shape (n_points, n_ell)",
            "cl_stack": "Seed axis kept separate, shape (n_points, 1, n_ell)",
            "cl_concat": "Compatibility alias for cl_y100, shape (n_points, n_ell)",
            "cl_log10_*": "log10(max(raw, target_floor)) versions of the same targets",
        },
        "target_floor": float(args.target_floor),
        "ell_min": int(args.ell_min),
        "ell_max": None if args.ell_max is None else int(args.ell_max),
        "expected_points": int(args.expected_points),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Wrote y100 SBI dataset: {dataset_path}", flush=True)
    print(f"Wrote metadata: {metadata_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


def expected_rows_from_tables(tables, rows_per_split: int):
    rows = []
    for table in tables:
        split_index = split_index_from_csv(str(table.path))
        split_value = split_index if split_index is not None else -1
        for row_number in range(1, len(table.rows) + 1):
            if split_index is None or split_index < 1:
                global_row = row_number
            else:
                global_row = (split_index - 1) * int(rows_per_split) + row_number
            rows.append(
                {
                    "sobol_global_row": int(global_row),
                    "sobol_split": int(split_value),
                    "sobol_row": int(row_number),
                    "sobol_csv": str(table.path),
                }
            )
    return rows


def missing_expected_rows(tables, metadata, rows_per_split: int):
    import pandas as pd

    present = {
        (str(Path(csv_path).expanduser().resolve()), int(row))
        for csv_path, row in zip(metadata["sobol_csv"], metadata["sobol_row"])
    }

    missing = [
        row
        for row in expected_rows_from_tables(tables, rows_per_split)
        if (str(Path(row["sobol_csv"]).expanduser().resolve()), int(row["sobol_row"])) not in present
    ]
    return pd.DataFrame(missing)


def report_missing_expected_rows(args, tables, metadata):
    output_dir = Path(args.output_dir).expanduser().resolve()
    missing = missing_expected_rows(tables, metadata, args.rows_per_split)
    if missing.empty:
        print("No missing Sobol rows found relative to the provided Sobol CSVs.", flush=True)
        return missing

    output_dir.mkdir(parents=True, exist_ok=True)
    missing_path = output_dir / args.missing_name
    missing.to_csv(missing_path, index=False)

    print(f"Missing expected y100 rows: {len(missing)}", flush=True)
    print(f"Wrote missing-row report: {missing_path}", flush=True)
    print("First missing rows:", flush=True)
    preview_columns = ["sobol_global_row", "sobol_split", "sobol_row", "sobol_csv"]
    print(missing.loc[:, preview_columns].head(20).to_string(index=False), flush=True)
    return missing


def parse_optional_ell_max(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "all", "full", "unbounded"}:
        return None

    try:
        ell_max = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ell-max must be an integer, or one of: none, all, full"
        ) from exc
    if ell_max < 0:
        return None
    return ell_max


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine generated y100-only Battaglia Sobol C_l products for later SBI use."
    )
    parser.add_argument("--y100-dirs", nargs="+", default=["/lustre/work/kristero10/tSZ_data/y100"])
    parser.add_argument("--sobol-csvs", nargs="+", default=DEFAULT_SOBOL_CSVS)
    parser.add_argument("--profile-globs", nargs="+", default=["*tSZ_cl*.fits"])
    parser.add_argument("--include-path-regex", nargs="*", default=DEFAULT_INCLUDE_REGEX)
    parser.add_argument("--exclude-path-regex", nargs="*", default=DEFAULT_EXCLUDE_REGEX)
    parser.add_argument("--output-dir", default="/lustre/work/kristero10/tSZ_data/sbi_battaglia_y100_2048")
    parser.add_argument("--dataset-name", default="sbi_battaglia_y100_2048.npz")
    parser.add_argument("--metadata-name", default="sbi_battaglia_y100_2048_metadata.csv")
    parser.add_argument("--manifest-name", default="sbi_battaglia_y100_2048_manifest.json")
    parser.add_argument("--missing-name", default="sbi_battaglia_y100_missing_rows.csv")
    parser.add_argument("--ell-min", type=int, default=2)
    parser.add_argument(
        "--ell-max",
        type=parse_optional_ell_max,
        default=4096,
        help="Maximum ell to keep. Use none/all/full, or any negative integer, for no upper cut.",
    )
    parser.add_argument("--key-precision", type=int, default=12)
    parser.add_argument("--expected-points", type=int, default=2048)
    parser.add_argument("--rows-per-split", type=int, default=128)
    parser.add_argument("--target-floor", type=float, default=1.0e-40)
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if fewer than expected points are present.")
    args = parser.parse_args(argv)

    args.y100_dirs = split_cli_values(args.y100_dirs)
    args.sobol_csvs = split_cli_values(args.sobol_csvs)
    args.profile_globs = split_cli_values(args.profile_globs)
    args.include_path_regex = split_comma_values(args.include_path_regex)
    args.exclude_path_regex = split_comma_values(args.exclude_path_regex)
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    tables, x_columns = load_sobol_tables(args.sobol_csvs)
    print(f"Using parameter columns: {', '.join(x_columns)}", flush=True)
    print(f"Using Sobol CSV tags: {', '.join(table.tag for table in tables)}", flush=True)

    y100_by_key, ell = load_profile_group(
        "y100",
        args.y100_dirs,
        args.profile_globs,
        tables,
        x_columns,
        args.ell_min,
        args.ell_max,
        args.key_precision,
        args.include_path_regex,
        args.exclude_path_regex,
    )
    y100_by_key = deduplicate_profile_records(y100_by_key, "y100")

    theta, cl_y100, metadata, ell = combine_y100_records(y100_by_key, x_columns, ell)
    metadata = add_global_sobol_rows(metadata, args.rows_per_split)
    metadata, theta, cl_y100 = reorder_by_metadata(metadata, theta, cl_y100)

    print(f"Matched y100 parameter points: {theta.shape[0]}", flush=True)
    print(f"Profile length after ell cut: {cl_y100.shape[1]}", flush=True)
    missing = None
    if args.expected_points and theta.shape[0] != args.expected_points:
        missing = report_missing_expected_rows(args, tables, metadata)

    if args.expected_points and theta.shape[0] != args.expected_points and not args.allow_missing:
        missing_count = "unknown" if missing is None else str(len(missing))
        raise ValueError(
            f"Expected {args.expected_points} y100 points, got {theta.shape[0]}. "
            f"Missing-row report contains {missing_count} rows. "
            "Use --allow-missing if you intentionally want a partial SBI dataset."
        )

    write_sbi_outputs(args, x_columns, ell, theta, cl_y100, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
