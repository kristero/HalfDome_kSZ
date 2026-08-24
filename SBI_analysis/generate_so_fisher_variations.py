#!/usr/bin/env python3
"""Create the finite-difference parameter grid for the SO Fisher analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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
DEFAULT_OUTPUT_ROOT = Path(
    "/lustre/work/kristero10/adrian_fisher_baseline_deproj0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write Battaglia12 central finite-difference variations for the "
            "masked baseline-noise deproj0 SO Fisher calculation."
        )
    )
    parser.add_argument(
        "--prepared-dataset",
        type=Path,
        default=DEFAULT_PREPARED_DATASET,
        help="Prepared SBI NPZ containing param_names, prior_low, and prior_high.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for the variation table and simulation outputs.",
    )
    parser.add_argument(
        "--step-fractions",
        type=float,
        nargs="+",
        default=(0.01, 0.02),
        help="Finite-difference steps as fractions of each prior width.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing variation tables.",
    )
    return parser.parse_args()


def scalar_strings(values: np.ndarray) -> list[str]:
    return [format(float(value), ".17g") for value in values]


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    prepared_path = args.prepared_dataset.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset does not exist: {prepared_path}")

    with np.load(prepared_path, allow_pickle=True) as data:
        required = {"param_names", "prior_low", "prior_high"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"{prepared_path} is missing required arrays: {missing}; "
                f"available={data.files}"
            )
        param_names = [str(value) for value in data["param_names"]]
        prior_low = np.asarray(data["prior_low"], dtype=np.float64)
        prior_high = np.asarray(data["prior_high"], dtype=np.float64)

    if param_names != PARAMETER_NAMES:
        raise ValueError(
            "Unexpected parameter order. "
            f"Expected {PARAMETER_NAMES}, found {param_names}."
        )
    if prior_low.shape != BATTAGLIA12.shape or prior_high.shape != BATTAGLIA12.shape:
        raise ValueError(
            f"Prior arrays must have shape {BATTAGLIA12.shape}; "
            f"found {prior_low.shape} and {prior_high.shape}."
        )

    prior_width = prior_high - prior_low
    if np.any(prior_width <= 0):
        raise ValueError("All prior widths must be positive.")
    outside = (BATTAGLIA12 <= prior_low) | (BATTAGLIA12 >= prior_high)
    if np.any(outside):
        names = [param_names[i] for i in np.flatnonzero(outside)]
        raise ValueError(f"Battaglia12 is not strictly inside the prior for: {names}")

    fractions = sorted(set(float(value) for value in args.step_fractions))
    if not fractions or any(value <= 0 for value in fractions):
        raise ValueError("--step-fractions must contain positive values.")

    theta_rows: list[np.ndarray] = [BATTAGLIA12.copy()]
    manifest: list[dict[str, object]] = [
        {
            "row_1based": 1,
            "label": "fiducial",
            "parameter": "fiducial",
            "parameter_index": -1,
            "side": "fiducial",
            "sign": 0,
            "step_fraction": 0.0,
            "step_absolute": 0.0,
        }
    ]

    for parameter_index, parameter in enumerate(param_names):
        for fraction in fractions:
            step = fraction * prior_width[parameter_index]
            for sign, side in ((-1, "minus"), (1, "plus")):
                theta = BATTAGLIA12.copy()
                theta[parameter_index] += sign * step
                if not (
                    prior_low[parameter_index]
                    < theta[parameter_index]
                    < prior_high[parameter_index]
                ):
                    raise ValueError(
                        f"{parameter} {side} step {fraction:g} leaves the prior: "
                        f"{theta[parameter_index]:.8g}"
                    )
                label = f"{parameter}_{side}_f{fraction:.6g}".replace(".", "p")
                theta_rows.append(theta)
                manifest.append(
                    {
                        "row_1based": len(theta_rows),
                        "label": label,
                        "parameter": parameter,
                        "parameter_index": parameter_index,
                        "side": side,
                        "sign": sign,
                        "step_fraction": fraction,
                        "step_absolute": step,
                    }
                )

    theta_array = np.vstack(theta_rows)
    expected_rows = 1 + 2 * len(param_names) * len(fractions)
    if theta_array.shape != (expected_rows, len(param_names)):
        raise AssertionError(f"Unexpected variation-grid shape: {theta_array.shape}")

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_root / "fisher_variations.csv",
        output_root / "fisher_variations_manifest.csv",
        output_root / "fisher_variations_manifest.json",
        output_root / "fisher_variations_theta.npy",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Variation outputs already exist. Use --overwrite to replace them:\n"
            + joined
        )

    write_csv(
        outputs[0],
        param_names,
        [scalar_strings(theta) for theta in theta_array],
    )

    manifest_header = [
        "row_1based",
        "label",
        "parameter",
        "parameter_index",
        "side",
        "sign",
        "step_fraction",
        "step_absolute",
        *param_names,
    ]
    manifest_rows: list[list[object]] = []
    for metadata, theta in zip(manifest, theta_array):
        manifest_rows.append(
            [metadata[key] for key in manifest_header[:8]] + scalar_strings(theta)
        )
    write_csv(outputs[1], manifest_header, manifest_rows)

    json_rows = []
    for metadata, theta in zip(manifest, theta_array):
        row = dict(metadata)
        row["theta"] = {
            name: float(value) for name, value in zip(param_names, theta)
        }
        json_rows.append(row)

    payload = {
        "prepared_dataset": str(prepared_path),
        "output_root": str(output_root),
        "parameter_names": param_names,
        "prior_low": prior_low.tolist(),
        "prior_high": prior_high.tolist(),
        "prior_width": prior_width.tolist(),
        "fiducial": BATTAGLIA12.tolist(),
        "step_fractions": fractions,
        "n_rows": int(theta_array.shape[0]),
        "rows": json_rows,
    }
    outputs[2].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.save(outputs[3], theta_array)

    print(f"Wrote {theta_array.shape[0]} Fisher variations to {output_root}")
    for path in outputs:
        print(f"  {path}")
    print("CSV row i is the same as PBS array index i.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
