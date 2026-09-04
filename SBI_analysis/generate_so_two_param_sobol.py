#!/usr/bin/env python3
"""Generate a two-parameter Battaglia Sobol design for the SO simulator.

Only P0 and beta vary. The other seven parameters are repeated at their
Battaglia12 fiducial values so the CSV remains compatible with the existing
nine-column Julia parameter reader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

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
TARGET_PARAMETERS = ["P0", "beta"]

DEFAULT_PRIOR = {
    "P0": (1.832524, 34.341221),
    "beta": (3.480627, 5.216611),
}
BATTAGLIA12 = {
    "P0": 18.1,
    "xc": 0.497,
    "beta": 4.35,
    "alpha_m_P0": 0.154,
    "alpha_m_xc": -0.00865,
    "alpha_m_beta": 0.0393,
    "alpha_z_P0": -0.758,
    "alpha_z_xc": 0.731,
    "alpha_z_beta": 0.415,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "/lustre/work/kristero10/adrian_two_param_so_baseline_deproj0/"
            "block_offset0_n16384/design/battaglia_sobol_P0_beta_16384.csv"
        ),
    )
    parser.add_argument("--n-samples", type=int, default=16384)
    parser.add_argument("--sobol-seed", type=int, default=12345)
    parser.add_argument(
        "--sequence-offset",
        type=int,
        default=0,
        help=(
            "Skip this many points in the same Sobol sequence. Use 16384 for "
            "the second non-overlapping 16k block."
        ),
    )
    scramble = parser.add_mutually_exclusive_group()
    scramble.add_argument("--scramble", dest="scramble", action="store_true")
    scramble.add_argument("--no-scramble", dest="scramble", action="store_false")
    parser.set_defaults(scramble=True)

    parser.add_argument("--p0-low", type=float, default=DEFAULT_PRIOR["P0"][0])
    parser.add_argument("--p0-high", type=float, default=DEFAULT_PRIOR["P0"][1])
    parser.add_argument("--beta-low", type=float, default=DEFAULT_PRIOR["beta"][0])
    parser.add_argument("--beta-high", type=float, default=DEFAULT_PRIOR["beta"][1])

    for name, default in BATTAGLIA12.items():
        if name in TARGET_PARAMETERS:
            continue
        parser.add_argument(
            f"--fixed-{name.replace('_', '-')}",
            dest=f"fixed_{name}",
            type=float,
            default=default,
        )

    parser.add_argument(
        "--noise-seed-base",
        type=int,
        default=1_000_000,
        help=(
            "Worker uses noise_seed = noise_seed_base + sequence_offset "
            "+ one-based local Sobol row."
        ),
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate existing CSV/NPZ/JSON against the requested settings and exit.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def validate_args(args: argparse.Namespace) -> None:
    if args.n_samples < 2:
        raise ValueError("--n-samples must be at least 2")
    if args.sequence_offset < 0:
        raise ValueError("--sequence-offset must be non-negative")
    if not args.p0_low < args.p0_high:
        raise ValueError("--p0-low must be smaller than --p0-high")
    if not args.beta_low < args.beta_high:
        raise ValueError("--beta-low must be smaller than --beta-high")
    if args.noise_seed_base < 0:
        raise ValueError("--noise-seed-base must be non-negative")
    if not is_power_of_two(args.n_samples):
        print(
            "WARNING: n_samples is not a power of two; the Sobol design loses "
            "its strongest balance property."
        )
    balanced_continuation = (
        is_power_of_two(args.n_samples)
        and is_power_of_two(args.sequence_offset + args.n_samples)
    )
    if args.sequence_offset and not balanced_continuation:
        print(
            "WARNING: this nonzero sequence offset is not a compatible "
            "base-2 continuation, so the requested block can have weaker balance."
        )


def draw_unit_sobol(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    try:
        import scipy
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError(
            "SciPy is required. Activate the same Python environment used for SBI."
        ) from exc

    engine = qmc.Sobol(d=2, scramble=args.scramble, seed=args.sobol_seed)
    balanced_block = (
        is_power_of_two(args.n_samples)
        and is_power_of_two(args.sequence_offset + args.n_samples)
    )
    if balanced_block:
        if args.sequence_offset:
            engine.fast_forward(args.sequence_offset)
        exponent = int(round(math.log2(args.n_samples)))
        unit = engine.random_base2(exponent)
        method = (
            f"fast_forward({args.sequence_offset})_then_random_base2({exponent})"
        )
    else:
        if args.sequence_offset:
            engine.fast_forward(args.sequence_offset)
        unit = engine.random(args.n_samples)
        method = f"fast_forward({args.sequence_offset})_then_random"

    unit = np.asarray(unit, dtype=np.float64)
    if unit.shape != (args.n_samples, 2):
        raise RuntimeError(f"Unexpected Sobol shape: {unit.shape}")
    if not np.all(np.isfinite(unit)) or np.any(unit < 0.0) or np.any(unit >= 1.0):
        raise RuntimeError("Sobol engine returned values outside [0,1)")
    return unit, f"scipy-{scipy.__version__}:{method}"


def build_theta(args: argparse.Namespace, unit: np.ndarray) -> np.ndarray:
    values = {name: np.full(args.n_samples, BATTAGLIA12[name]) for name in PARAMETER_NAMES}
    values["P0"] = args.p0_low + unit[:, 0] * (args.p0_high - args.p0_low)
    values["beta"] = args.beta_low + unit[:, 1] * (args.beta_high - args.beta_low)

    for name in PARAMETER_NAMES:
        if name in TARGET_PARAMETERS:
            continue
        values[name].fill(float(getattr(args, f"fixed_{name}")))

    return np.column_stack([values[name] for name in PARAMETER_NAMES]).astype(
        np.float64,
        copy=False,
    )


def validate_design(
    theta_full: np.ndarray,
    unit: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    p0 = theta_full[:, PARAMETER_NAMES.index("P0")]
    beta = theta_full[:, PARAMETER_NAMES.index("beta")]
    pairs = np.column_stack([p0, beta])
    unique_pairs = np.unique(pairs, axis=0).shape[0]
    if unique_pairs != args.n_samples:
        raise ValueError(
            f"Sobol design contains duplicate P0/beta pairs: {unique_pairs}/{args.n_samples}"
        )

    fixed_checks = {}
    for name in PARAMETER_NAMES:
        if name in TARGET_PARAMETERS:
            continue
        column = theta_full[:, PARAMETER_NAMES.index(name)]
        expected = float(getattr(args, f"fixed_{name}"))
        passed = bool(np.all(column == expected))
        fixed_checks[name] = {
            "value": expected,
            "unique_count": int(np.unique(column).size),
            "passed": passed,
        }
        if not passed:
            raise ValueError(f"Fixed parameter {name} changed across the design")

    corr = float(np.corrcoef(unit[:, 0], unit[:, 1])[0, 1])
    return {
        "n_rows": int(args.n_samples),
        "unique_target_pairs": int(unique_pairs),
        "unit_min": unit.min(axis=0).tolist(),
        "unit_max": unit.max(axis=0).tolist(),
        "unit_pearson_correlation": corr,
        "fixed_parameter_checks": fixed_checks,
        "p0_sample_minmax": [float(p0.min()), float(p0.max())],
        "beta_sample_minmax": [float(beta.min()), float(beta.max())],
    }


def write_csv(path: Path, theta_full: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PARAMETER_NAMES)
        writer.writerows(theta_full)


def validate_existing_design(
    args: argparse.Namespace,
    output_csv: Path,
    metadata_json: Path,
    metadata_npz: Path,
) -> None:
    """Refuse to reuse a design whose scientific configuration has changed."""
    missing = [
        path for path in (output_csv, metadata_json, metadata_npz) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Existing design is incomplete; missing: {missing}")

    expected_prior_low = np.asarray([args.p0_low, args.beta_low], dtype=np.float64)
    expected_prior_high = np.asarray([args.p0_high, args.beta_high], dtype=np.float64)
    expected_rows = np.arange(1, args.n_samples + 1, dtype=np.int64)
    expected_sequence_indices = args.sequence_offset + expected_rows
    fixed_names = [name for name in PARAMETER_NAMES if name not in TARGET_PARAMETERS]
    expected_fixed = np.asarray(
        [getattr(args, f"fixed_{name}") for name in fixed_names], dtype=np.float64
    )

    with np.load(metadata_npz, allow_pickle=False) as saved:
        theta = np.asarray(saved["theta"], dtype=np.float64)
        theta_full = np.asarray(saved["theta_full"], dtype=np.float64)
        checks = {
            "theta shape": theta.shape == (args.n_samples, len(TARGET_PARAMETERS)),
            "theta_full shape": theta_full.shape
            == (args.n_samples, len(PARAMETER_NAMES)),
            "target parameter names": np.array_equal(
                np.asarray(saved["param_names"]).astype(str),
                np.asarray(TARGET_PARAMETERS),
            ),
            "full parameter names": np.array_equal(
                np.asarray(saved["full_param_names"]).astype(str),
                np.asarray(PARAMETER_NAMES),
            ),
            "prior lower bounds": np.array_equal(saved["prior_low"], expected_prior_low),
            "prior upper bounds": np.array_equal(saved["prior_high"], expected_prior_high),
            "fixed parameter names": np.array_equal(
                np.asarray(saved["fixed_param_names"]).astype(str),
                np.asarray(fixed_names),
            ),
            "fixed parameter values": np.array_equal(
                saved["fixed_param_values"], expected_fixed
            ),
            "Sobol rows": np.array_equal(saved["sobol_row"], expected_rows),
            "Sobol sequence indices": np.array_equal(
                saved["sobol_sequence_index"], expected_sequence_indices
            ),
            "Sobol seed": int(np.asarray(saved["sobol_seed"]).item())
            == args.sobol_seed,
            "Sobol sequence offset": int(np.asarray(saved["sequence_offset"]).item())
            == args.sequence_offset,
            "Sobol scrambling": bool(np.asarray(saved["scramble"]).item())
            == args.scramble,
            "noise seed base": int(np.asarray(saved["noise_seed_base"]).item())
            == args.noise_seed_base,
            "noise seeds": np.array_equal(
                saved["noise_seed"], args.noise_seed_base + expected_sequence_indices
            ),
        }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Existing Sobol design does not match the requested configuration: "
            + ", ".join(failed)
            + ". Use a new RUN_ROOT for a changed design."
        )

    csv_theta = np.loadtxt(output_csv, delimiter=",", skiprows=1, dtype=np.float64)
    if csv_theta.shape != theta_full.shape or not np.array_equal(csv_theta, theta_full):
        raise ValueError("Existing CSV values do not exactly match theta_full in its NPZ")

    with metadata_json.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("csv_sha256") != sha256_file(output_csv):
        raise ValueError("Existing CSV checksum does not match its JSON metadata")

    print(f"Validated existing Sobol design: {output_csv}")
    print(f"  rows: {args.n_samples}")
    print(f"  P0 prior: [{args.p0_low}, {args.p0_high}]")
    print(f"  beta prior: [{args.beta_low}, {args.beta_high}]")
    print(
        f"  Sobol seed/scramble/offset: "
        f"{args.sobol_seed}/{args.scramble}/{args.sequence_offset}"
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_csv = args.output_csv.expanduser().resolve()
    metadata_json = output_csv.with_suffix(".json")
    metadata_npz = output_csv.with_suffix(".npz")
    if args.validate_existing:
        validate_existing_design(args, output_csv, metadata_json, metadata_npz)
        return 0

    existing = [path for path in (output_csv, metadata_json, metadata_npz) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing design files: {existing}. Use --force."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    unit, generator = draw_unit_sobol(args)
    theta_full = build_theta(args, unit)
    validation = validate_design(theta_full, unit, args)
    write_csv(output_csv, theta_full)

    target_indices = np.asarray(
        [PARAMETER_NAMES.index(name) for name in TARGET_PARAMETERS],
        dtype=np.int64,
    )
    prior_low = np.asarray([args.p0_low, args.beta_low], dtype=np.float64)
    prior_high = np.asarray([args.p0_high, args.beta_high], dtype=np.float64)
    fixed_names = [name for name in PARAMETER_NAMES if name not in TARGET_PARAMETERS]
    fixed_values = np.asarray(
        [getattr(args, f"fixed_{name}") for name in fixed_names],
        dtype=np.float64,
    )
    sobol_rows = np.arange(1, args.n_samples + 1, dtype=np.int64)
    sobol_sequence_indices = args.sequence_offset + sobol_rows
    noise_seeds = args.noise_seed_base + sobol_sequence_indices

    np.savez_compressed(
        metadata_npz,
        theta=theta_full[:, target_indices],
        theta_full=theta_full,
        param_names=np.asarray(TARGET_PARAMETERS),
        theta_columns=np.asarray(TARGET_PARAMETERS),
        full_param_names=np.asarray(PARAMETER_NAMES),
        fixed_param_names=np.asarray(fixed_names),
        fixed_param_values=fixed_values,
        prior_low=prior_low,
        prior_high=prior_high,
        sobol_unit=unit,
        sobol_row=sobol_rows,
        sobol_sequence_index=sobol_sequence_indices,
        noise_seed=noise_seeds,
        noise_seed_base=np.asarray(args.noise_seed_base, dtype=np.int64),
        sobol_seed=np.asarray(args.sobol_seed, dtype=np.int64),
        sequence_offset=np.asarray(args.sequence_offset, dtype=np.int64),
        scramble=np.asarray(args.scramble),
    )

    metadata = {
        "product": "masked_baseline_noise_cross_deproj0",
        "varying_parameters": TARGET_PARAMETERS,
        "full_parameter_columns": PARAMETER_NAMES,
        "prior": {
            "P0": [args.p0_low, args.p0_high],
            "beta": [args.beta_low, args.beta_high],
        },
        "fixed_parameters": {
            name: float(getattr(args, f"fixed_{name}")) for name in fixed_names
        },
        "n_samples": args.n_samples,
        "sobol_dimension": 2,
        "sobol_seed": args.sobol_seed,
        "sobol_scramble": args.scramble,
        "sequence_offset": args.sequence_offset,
        "generator": generator,
        "noise_seed_policy": (
            "noise_seed_base + sequence_offset + one_based_local_sobol_row"
        ),
        "noise_seed_base": args.noise_seed_base,
        "same_mask_policy": "one fixed mask_seed supplied by the PBS worker",
        "validation": validation,
        "files": {
            "csv": str(output_csv),
            "npz": str(metadata_npz),
        },
    }
    metadata["csv_sha256"] = sha256_file(output_csv)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(metadata), handle, indent=2, sort_keys=True)

    print(f"Wrote {output_csv}")
    print(f"Wrote {metadata_npz}")
    print(f"Wrote {metadata_json}")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
