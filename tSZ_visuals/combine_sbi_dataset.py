#!/usr/bin/env python3
"""
Combine the y100/y102 Battaglia Sobol products into an SBI-ready dataset.

This script intentionally does not train an emulator. It only discovers the
generated C_l/profile files, matches y100 and y102 by Sobol/Battaglia
parameters, and writes aligned arrays plus metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from build_battaglia_emulator import (
    combine_realizations,
    deduplicate_profile_records,
    load_profile_group,
    load_sobol_tables,
    print_match_diagnostics,
    split_cli_values,
    split_comma_values,
)


DEFAULT_SOBOL_CSVS = [
    f"/home/kristero10/tSZ_data/battaglia_sobol_512_{idx}.csv"
    for idx in range(1, 5)
]
DEFAULT_INCLUDE_REGEX = [r"sobol_battaglia_sobol_512_[1-4]_row[0-9]+"]
DEFAULT_EXCLUDE_REGEX = [r"sobol_battaglia_sobol_512_row[0-9]+|sobol_battaglia_sobol_256"]


def split_index_from_csv(path: str) -> int | None:
    match = re.search(r"_([0-9]+)\.csv$", Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def add_global_sobol_rows(metadata, rows_per_split: int):
    metadata = metadata.copy()
    split_indices = []
    global_rows = []

    for csv_path, row in zip(metadata["sobol_csv"], metadata["sobol_row"], strict=True):
        split_index = split_index_from_csv(str(csv_path))
        split_indices.append(split_index if split_index is not None else -1)
        if split_index is None or split_index < 1:
            global_rows.append(int(row))
        else:
            global_rows.append((split_index - 1) * int(rows_per_split) + int(row))

    metadata.insert(0, "sobol_global_row", global_rows)
    metadata.insert(1, "sobol_split", split_indices)
    return metadata


def reorder_by_metadata(metadata, *arrays):
    import numpy as np

    order = np.lexsort(
        (
            metadata["sobol_row"].to_numpy(dtype=int),
            metadata["sobol_split"].to_numpy(dtype=int),
            metadata["sobol_global_row"].to_numpy(dtype=int),
        )
    )
    metadata_ordered = metadata.iloc[order].reset_index(drop=True)
    return (metadata_ordered, *(np.asarray(array)[order] for array in arrays))


def write_sbi_outputs(args, x_columns, ell, theta, cl_mean, cl_y100, cl_y102, metadata):
    import numpy as np

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cl_stack = np.stack([cl_y100, cl_y102], axis=1)
    cl_concat = np.concatenate([cl_y100, cl_y102], axis=1)
    cl_log10_mean = np.log10(np.maximum(cl_mean, args.target_floor))
    cl_log10_y100 = np.log10(np.maximum(cl_y100, args.target_floor))
    cl_log10_y102 = np.log10(np.maximum(cl_y102, args.target_floor))
    cl_log10_concat = np.concatenate([cl_log10_y100, cl_log10_y102], axis=1)

    dataset_path = output_dir / args.dataset_name
    np.savez_compressed(
        dataset_path,
        theta=theta,
        theta_columns=np.asarray(x_columns, dtype=str),
        ell=ell,
        cl_mean=cl_mean,
        cl_y100=cl_y100,
        cl_y102=cl_y102,
        cl_stack=cl_stack,
        cl_concat=cl_concat,
        cl_log10_mean=cl_log10_mean,
        cl_log10_y100=cl_log10_y100,
        cl_log10_y102=cl_log10_y102,
        cl_log10_concat=cl_log10_concat,
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
        "arrays": {
            "theta": "Battaglia parameter matrix, shape (n_points, n_parameters)",
            "cl_mean": "Mean of y100 and y102 C_l/profile arrays, shape (n_points, n_ell)",
            "cl_y100": "Seed/lightcone 100 C_l/profile arrays, shape (n_points, n_ell)",
            "cl_y102": "Seed/lightcone 102 C_l/profile arrays, shape (n_points, n_ell)",
            "cl_stack": "Seed axis kept separate, shape (n_points, 2, n_ell)",
            "cl_concat": "y100 and y102 concatenated, shape (n_points, 2*n_ell)",
            "cl_log10_*": "log10(max(raw, target_floor)) versions of the same targets",
        },
        "target_floor": float(args.target_floor),
        "ell_min": int(args.ell_min),
        "ell_max": None if args.ell_max is None else int(args.ell_max),
        "expected_points": int(args.expected_points),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"Wrote SBI dataset: {dataset_path}", flush=True)
    print(f"Wrote metadata: {metadata_path}", flush=True)
    print(f"Wrote manifest: {manifest_path}", flush=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine generated y100/y102 Battaglia Sobol C_l products for later SBI use."
    )
    parser.add_argument("--y100-dirs", nargs="+", default=["/lustre/work/kristero10/tSZ_data/y100"])
    parser.add_argument("--y102-dirs", nargs="+", default=["/lustre/work/kristero10/tSZ_data/y102"])
    parser.add_argument("--sobol-csvs", nargs="+", default=DEFAULT_SOBOL_CSVS)
    parser.add_argument("--profile-globs", nargs="+", default=["*tSZ_cl*.fits"])
    parser.add_argument("--include-path-regex", nargs="*", default=DEFAULT_INCLUDE_REGEX)
    parser.add_argument("--exclude-path-regex", nargs="*", default=DEFAULT_EXCLUDE_REGEX)
    parser.add_argument("--output-dir", default="/lustre/work/kristero10/tSZ_data/sbi_battaglia_y100_y102")
    parser.add_argument("--dataset-name", default="sbi_battaglia_y100_y102_512.npz")
    parser.add_argument("--metadata-name", default="sbi_battaglia_y100_y102_512_metadata.csv")
    parser.add_argument("--manifest-name", default="sbi_battaglia_y100_y102_512_manifest.json")
    parser.add_argument("--ell-min", type=int, default=2)
    parser.add_argument("--ell-max", type=int, default=4096)
    parser.add_argument("--key-precision", type=int, default=12)
    parser.add_argument("--expected-points", type=int, default=512)
    parser.add_argument("--rows-per-split", type=int, default=128)
    parser.add_argument("--target-floor", type=float, default=1.0e-40)
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if fewer than expected matched points are present.")
    args = parser.parse_args(argv)

    args.y100_dirs = split_cli_values(args.y100_dirs)
    args.y102_dirs = split_cli_values(args.y102_dirs)
    args.sobol_csvs = split_cli_values(args.sobol_csvs)
    args.profile_globs = split_cli_values(args.profile_globs)
    args.include_path_regex = split_comma_values(args.include_path_regex)
    args.exclude_path_regex = split_comma_values(args.exclude_path_regex)
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import numpy as np

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

    y102_by_key, ell_102 = load_profile_group(
        "y102",
        args.y102_dirs,
        args.profile_globs,
        tables,
        x_columns,
        args.ell_min,
        args.ell_max,
        args.key_precision,
        args.include_path_regex,
        args.exclude_path_regex,
    )
    y102_by_key = deduplicate_profile_records(y102_by_key, "y102")

    if ell.shape != ell_102.shape or not np.allclose(ell, ell_102):
        raise ValueError("y100 and y102 ell grids differ")

    common_keys = sorted(set(y100_by_key).intersection(y102_by_key))
    print_match_diagnostics(y100_by_key, y102_by_key, common_keys)

    theta, cl_mean, cl_y100, cl_y102, metadata, ell = combine_realizations(
        y100_by_key,
        y102_by_key,
        x_columns,
        ell,
    )
    metadata = add_global_sobol_rows(metadata, args.rows_per_split)
    metadata, theta, cl_mean, cl_y100, cl_y102 = reorder_by_metadata(
        metadata,
        theta,
        cl_mean,
        cl_y100,
        cl_y102,
    )

    print(f"Matched parameter points: {theta.shape[0]}", flush=True)
    print(f"Profile length after ell cut: {cl_mean.shape[1]}", flush=True)
    if args.expected_points and theta.shape[0] != args.expected_points and not args.allow_missing:
        raise ValueError(
            f"Expected {args.expected_points} matched y100/y102 points, got {theta.shape[0]}. "
            "Use --allow-missing if you intentionally want a partial SBI dataset."
        )

    write_sbi_outputs(args, x_columns, ell, theta, cl_mean, cl_y100, cl_y102, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
