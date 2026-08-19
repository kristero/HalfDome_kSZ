#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from prepare_adrian_so_sbi_case_datasets import (
    load_sobol_global_row,
    load_sobol_theta_for_global_rows,
    normalize_products,
    parse_expected_sobol_prefix,
    product_cl_path,
    validate_sobol_global_row,
)


DEFAULT_PRODUCTS = [
    "unmasked_no_noise",
    "masked_no_noise",
    "masked_baseline_noise_cross_deproj0",
    "masked_baseline_noise_cross_deproj2",
    "masked_goal_noise_cross_deproj0",
    "masked_goal_noise_cross_deproj2",
]

THETA_COLUMNS = [
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


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Adrian C_ell/Sobol ordering before or after SBI dataset preparation."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--sobol-csv", required=True)
    parser.add_argument(
        "--sobol-global-row-path",
        default="",
        help=(
            "Optional standalone mapping file. If omitted, the mapping is read from the first "
            "sbi_<product>.npz in --input-dir."
        ),
    )
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument(
        "--expected-sobol-prefix",
        default="108566,634,163005,417786",
        help="Known first source-machine row IDs; pass an empty string to disable this check.",
    )
    parser.add_argument(
        "--prepared-dir",
        default="",
        help="Optional prepared dataset directory; enables exact theta/mapping round-trip checks.",
    )
    parser.add_argument("--ell-tag", default="ell80_7979")
    parser.add_argument("--sample-checks", type=int, default=1024)
    parser.add_argument(
        "--skip-product-metadata-check",
        action="store_true",
        help="Skip checking every sbi_<product>.npz and metadata CSV against the common mapping.",
    )
    parser.add_argument("--report", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    sobol_csv = Path(args.sobol_csv).expanduser().resolve()
    prepared_dir = Path(args.prepared_dir).expanduser().resolve() if args.prepared_dir else None
    products = normalize_products(args.products)
    if not products:
        raise ValueError("No products selected")
    mapping_path = (
        Path(args.sobol_global_row_path).expanduser().resolve()
        if args.sobol_global_row_path
        else input_dir / f"sbi_{products[0]}.npz"
    )

    shapes: dict[str, list[int]] = {}
    n_rows: int | None = None
    for product in products:
        path = product_cl_path(input_dir, product)
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw product: {path}")
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2:
            raise ValueError(f"{path} must be 2D, got {array.shape}")
        shapes[product] = [int(value) for value in array.shape]
        if n_rows is None:
            n_rows = int(array.shape[0])
        elif int(array.shape[0]) != n_rows:
            raise ValueError(f"Product row counts differ: {shapes}")
        del array

    if n_rows is None:
        raise ValueError("No products selected")

    mapping = load_sobol_global_row(mapping_path)
    prefix = parse_expected_sobol_prefix(args.expected_sobol_prefix)
    validation = validate_sobol_global_row(mapping, n_rows, prefix)

    sample_count = min(max(int(args.sample_checks), 1), n_rows)
    sample_indices = np.unique(np.linspace(0, n_rows - 1, sample_count, dtype=np.int64))
    expected_theta = load_sobol_theta_for_global_rows(sobol_csv, mapping[sample_indices])

    product_metadata_checks: dict[str, Any] = {}
    if not args.skip_product_metadata_check:
        for product in products:
            metadata_path = input_dir / f"sbi_{product}.npz"
            metadata_csv_path = input_dir / f"sbi_{product}_metadata.csv"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Missing product metadata NPZ: {metadata_path}")
            if not metadata_csv_path.is_file():
                raise FileNotFoundError(f"Missing product metadata CSV: {metadata_csv_path}")

            with np.load(metadata_path, allow_pickle=False) as data:
                required = {"theta", "theta_columns", "ell", "sobol_global_row"}
                missing = sorted(required.difference(data.files))
                if missing:
                    raise KeyError(f"{metadata_path} is missing required keys {missing}; keys={data.files}")
                product_mapping = np.asarray(data["sobol_global_row"], dtype=np.int64).reshape(-1)
                product_theta_shape = tuple(int(value) for value in data["theta"].shape)
                product_theta_sample = np.asarray(data["theta"][sample_indices], dtype=np.float32)
                product_ell = np.asarray(data["ell"])
                product_columns = [str(value) for value in np.asarray(data["theta_columns"]).reshape(-1)]

            if not np.array_equal(product_mapping, mapping):
                if product_mapping.size == mapping.size:
                    mismatch = np.flatnonzero(product_mapping != mapping)
                    first = int(mismatch[0])
                else:
                    first = min(product_mapping.size, mapping.size)
                raise ValueError(
                    f"sobol_global_row differs for {product}; first mismatch/length boundary index={first}"
                )
            if product_theta_shape != (n_rows, len(THETA_COLUMNS)):
                raise ValueError(
                    f"theta shape for {product} is {product_theta_shape}, "
                    f"expected {(n_rows, len(THETA_COLUMNS))}"
                )
            if product_columns != THETA_COLUMNS:
                raise ValueError(
                    f"theta_columns for {product} are {product_columns}, expected {THETA_COLUMNS}"
                )
            if product_ell.ndim != 1 or product_ell.size != shapes[product][1]:
                raise ValueError(
                    f"ell shape for {product} is {product_ell.shape}, "
                    f"but C_ell shape is {shapes[product]}"
                )
            if not np.allclose(product_theta_sample, expected_theta, rtol=1e-6, atol=1e-7):
                delta = float(np.max(np.abs(product_theta_sample - expected_theta)))
                raise ValueError(
                    f"theta does not equal SobolCSV[sobol_global_row-1] for {product}; "
                    f"max sampled absolute difference={delta}"
                )

            with metadata_csv_path.open("r", encoding="utf-8") as handle:
                csv_header = [part.strip() for part in handle.readline().strip().split(",")]
            required_csv_columns = set(THETA_COLUMNS + ["sobol_global_row"])
            missing_csv_columns = sorted(required_csv_columns.difference(csv_header))
            if missing_csv_columns:
                raise KeyError(
                    f"{metadata_csv_path} is missing columns {missing_csv_columns}; header={csv_header}"
                )

            product_metadata_checks[product] = {
                "metadata_npz": str(metadata_path),
                "metadata_csv": str(metadata_csv_path),
                "mapping_exact_match": True,
                "sampled_theta_matches_full_sobol_csv": True,
                "sampled_rows": int(sample_indices.size),
                "theta_shape": list(product_theta_shape),
                "ell_shape": list(product_ell.shape),
                "metadata_csv_required_columns_present": True,
            }

    prepared_checks: dict[str, Any] = {}
    if prepared_dir is not None:
        for product in products:
            path = prepared_dir / f"so_{product}_{args.ell_tag}_sbi_run.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing prepared dataset: {path}")
            with np.load(path, allow_pickle=False) as data:
                prepared_mapping = np.asarray(data["sobol_global_row"], dtype=np.int64)
                prepared_theta = np.asarray(data["theta"][sample_indices], dtype=np.float32)
                theta_source = str(np.asarray(data["theta_source"]).item())

            if not np.array_equal(prepared_mapping, mapping):
                mismatch = np.flatnonzero(prepared_mapping != mapping)
                raise ValueError(
                    f"Prepared mapping differs for {product}; first mismatch index={int(mismatch[0])}"
                )
            if not np.array_equal(prepared_theta, expected_theta):
                delta = float(np.max(np.abs(prepared_theta - expected_theta)))
                raise ValueError(
                    f"Prepared theta does not equal SobolCSV[sobol_global_row-1] for {product}; "
                    f"max sampled absolute difference={delta}"
                )
            prepared_checks[product] = {
                "path": str(path),
                "theta_source": theta_source,
                "mapping_exact_match": True,
                "sampled_theta_exact_match": True,
                "sampled_rows": int(sample_indices.size),
            }

    report = {
        "status": "passed",
        "raw_product_shapes": shapes,
        "sobol_global_row_path": str(mapping_path),
        "sobol_csv": str(sobol_csv),
        "mapping_validation": validation,
        "product_metadata_checks": product_metadata_checks,
        "prepared_checks": prepared_checks,
        "provable_from_supplied_files": [
            "the mapping is a complete permutation of 1..N",
            "the mapping matches the known source-machine prefix",
            "all selected raw products have the same row count",
            "all product NPZ files carry the same mapping and theta ordering",
            "all product metadata CSV files contain theta and sobol_global_row columns",
            "prepared theta equals SobolCSV[sobol_global_row-1] when --prepared-dir is supplied",
        ],
        "not_provable_from_cl_plus_parameter_csv_alone": (
            "That mapping entry i is the simulation identity of C_ell[i]. This requires the mapping "
            "recorded during consolidation, or another retained per-row simulation identifier."
        ),
    }

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else (prepared_dir or input_dir) / "adrian_ordering_validation.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(jsonable(report), indent=2, sort_keys=True), encoding="utf-8")

    print("Ordering audit passed.")
    print(f"Products: {shapes}")
    print(f"Mapping validation: {validation}")
    if product_metadata_checks:
        print(f"Product metadata checks: {list(product_metadata_checks)}")
    if prepared_checks:
        print(f"Prepared theta round-trip checks: {list(prepared_checks)}")
    print(f"Report: {report_path}")
    print("Important: C_ell[i] -> simulation identity cannot be inferred from C_ell values and theta CSV alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
