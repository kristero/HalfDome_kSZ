#!/usr/bin/env python3
"""Package one Battaglia12 realization for reusable SO/SBI validation.

The Julia simulation is run separately for deprojection 0 and 2. This script
collects the five masked spectra, converts C_ell to linear D_ell, applies the
same Delta-ell=200 mode-count binning as the training dataset, and records the
requested FITS maps without reading them into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from combine_so_two_param_baseline_deproj0 import (
    FULL_PARAMETERS,
    TARGET_PARAMETERS,
    bin_dell,
    make_bins,
    select_cl,
)


BATTAGLIA12 = np.asarray(
    [
        18.1,
        0.497,
        4.35,
        0.154,
        -0.00865,
        0.0393,
        -0.758,
        0.731,
        0.415,
    ],
    dtype=np.float64,
)
CASES = [
    "masked_no_noise",
    "masked_baseline_noise_cross_deproj0",
    "masked_baseline_noise_cross_deproj2",
    "masked_goal_noise_cross_deproj0",
    "masked_goal_noise_cross_deproj2",
]


def parse_args() -> argparse.Namespace:
    root = Path(
        "/home/kristero10/final_products/tSZ/battaglia12/"
        "masked_so_baseline_goal"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deproj0-dir", type=Path, default=root / "raw" / "deproj0")
    parser.add_argument("--deproj2-dir", type=Path, default=root / "raw" / "deproj2")
    parser.add_argument("--output-dir", type=Path, default=root / "prepared")
    parser.add_argument(
        "--output-name",
        default="battaglia12_masked_so_all_noise_ell80_7979.npz",
    )
    parser.add_argument("--ell-min", type=int, default=80)
    parser.add_argument("--ell-max", type=int, default=7979)
    parser.add_argument("--mask-seed", type=int, default=12345)
    parser.add_argument("--noise-seed", type=int, default=2_000_001)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} under {directory} matching {pattern!r}; "
            f"found {len(matches)}: {matches}"
        )
    path = matches[0].resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty {label}: {path}")
    return path


def file_record(path: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def main() -> int:
    args = parse_args()
    deproj0_dir = args.deproj0_dir.expanduser().resolve()
    deproj2_dir = args.deproj2_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    completion_path = output_dir / "battaglia12_product_complete.json"

    existing = [path for path in (output_path, manifest_path, completion_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Refusing to overwrite {existing}; use --force")

    spectrum_paths = {
        "masked_no_noise": require_one(
            deproj0_dir,
            "*masked_no_noise_cl*deproj0*lmax7979.npy",
            "masked no-noise C_ell",
        ),
        "masked_baseline_noise_cross_deproj0": require_one(
            deproj0_dir,
            "*masked_baseline_noise_cross_cl*deproj0*lmax7979.npy",
            "baseline deproj0 cross C_ell",
        ),
        "masked_baseline_noise_cross_deproj2": require_one(
            deproj2_dir,
            "*masked_baseline_noise_cross_cl*deproj2*lmax7979.npy",
            "baseline deproj2 cross C_ell",
        ),
        "masked_goal_noise_cross_deproj0": require_one(
            deproj0_dir,
            "*masked_goal_noise_cross_cl*deproj0*lmax7979.npy",
            "goal deproj0 cross C_ell",
        ),
        "masked_goal_noise_cross_deproj2": require_one(
            deproj2_dir,
            "*masked_goal_noise_cross_cl*deproj2*lmax7979.npy",
            "goal deproj2 cross C_ell",
        ),
    }

    map_paths = {
        "apodized_mask": require_one(
            deproj0_dir,
            "*apodized_mask_map*deproj0*lmax7979.fits",
            "deproj0 mask map",
        ),
        "masked_no_noise": require_one(
            deproj0_dir,
            "*masked_clean_signal_map*deproj0*lmax7979.fits",
            "masked no-noise map",
        ),
        "masked_baseline_noisy_split1_deproj0": require_one(
            deproj0_dir,
            "*masked_baseline_noisy_split1_map*deproj0*lmax7979.fits",
            "baseline deproj0 split-1 map",
        ),
        "masked_baseline_noisy_split2_deproj0": require_one(
            deproj0_dir,
            "*masked_baseline_noisy_split2_map*deproj0*lmax7979.fits",
            "baseline deproj0 split-2 map",
        ),
        "masked_goal_noisy_split1_deproj0": require_one(
            deproj0_dir,
            "*masked_goal_noisy_split1_map*deproj0*lmax7979.fits",
            "goal deproj0 split-1 map",
        ),
        "masked_goal_noisy_split2_deproj0": require_one(
            deproj0_dir,
            "*masked_goal_noisy_split2_map*deproj0*lmax7979.fits",
            "goal deproj0 split-2 map",
        ),
    }

    ell = None
    cl_values = []
    for case in CASES:
        case_ell, case_cl = select_cl(
            spectrum_paths[case], args.ell_min, args.ell_max
        )
        if ell is None:
            ell = case_ell
        elif not np.array_equal(case_ell, ell):
            raise ValueError(f"{case} has a different multipole grid")
        cl_values.append(case_cl)
    assert ell is not None

    cl = np.stack(cl_values, axis=0)
    dell_factor = ell * (ell + 1.0) / (2.0 * np.pi)
    dell = cl * dell_factor[None, :]
    bins = make_bins(ell)
    x = np.stack([bin_dell(row, bins) for row in dell], axis=0)
    if not np.all(np.isfinite(x)):
        raise ValueError("Packaged binned D_ell contains non-finite values")

    theta_target = BATTAGLIA12[
        [FULL_PARAMETERS.index(name) for name in TARGET_PARAMETERS]
    ]
    map_keys = np.asarray(list(map_paths))
    map_values = np.asarray([str(map_paths[key]) for key in map_keys])
    payload: dict[str, Any] = {
        "case_names": np.asarray(CASES),
        "theta": theta_target[None, :].astype(np.float32),
        "theta_full": BATTAGLIA12[None, :].astype(np.float32),
        "param_names": np.asarray(TARGET_PARAMETERS),
        "full_param_names": np.asarray(FULL_PARAMETERS),
        "cl": cl.astype(np.float32),
        "dell": dell.astype(np.float32),
        "x": x.astype(np.float32),
        "ell_unbinned": ell.astype(np.float32),
        "ell": bins["ell_binned"],
        "ell_binned": bins["ell_binned"],
        "bin_ell_min": bins["bin_ell_min"],
        "bin_ell_max": bins["bin_ell_max"],
        "bin_counts": bins["bin_counts"],
        "mask_seed": np.asarray(args.mask_seed, dtype=np.int64),
        "noise_seed": np.asarray(args.noise_seed, dtype=np.int64),
        "map_keys": map_keys,
        "map_paths": map_values,
        "default_case": np.asarray("masked_baseline_noise_cross_deproj0"),
    }
    for index, case in enumerate(CASES):
        payload[f"x_{case}"] = x[index][None, :].astype(np.float32)
        payload[f"obs_{case}"] = x[index].astype(np.float32)
        payload[f"dell_{case}"] = dell[index].astype(np.float32)
    payload["obs"] = payload["obs_masked_baseline_noise_cross_deproj0"]
    payload["obs_theta"] = theta_target.astype(np.float32)
    np.savez_compressed(output_path, **payload)

    manifest = {
        "complete": True,
        "purpose": "Reusable Battaglia12 masked SO validation product",
        "physics": {
            "mask_fsky": 0.4,
            "mask_apodization_arcmin": 60.0,
            "mask_seed": args.mask_seed,
            "same_mask_all_cases": True,
            "noise_seed_base": args.noise_seed,
            "independent_noise_by_case_deprojection_and_split": True,
            "beam_applied_to_signal": True,
            "beam_fwhm_arcmin": 2.0,
            "spectra": "masked pseudo-C_ell and split-noise cross-C_ell",
            "binned_statistic": (
                "linear D_ell, weighted by 2ell+1 in Delta-ell=200 bins"
            ),
        },
        "theta_full": dict(zip(FULL_PARAMETERS, BATTAGLIA12.tolist())),
        "case_order": CASES,
        "spectrum_files": {
            key: file_record(path, include_hash=True)
            for key, path in spectrum_paths.items()
        },
        "map_files": {
            key: file_record(path, include_hash=False)
            for key, path in map_paths.items()
        },
        "output_npz": str(output_path),
        "output_npz_sha256": sha256_file(output_path),
        "shapes": {
            "cl": list(cl.shape),
            "dell": list(dell.shape),
            "x": list(x.shape),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with completion_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "complete": True,
                "dataset": str(output_path),
                "manifest": str(manifest_path),
                "cases": CASES,
                "map_keys": list(map_paths),
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(f"Wrote {output_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {completion_path}")
    print(f"cases: {', '.join(CASES)}")
    print(f"D_ell shape: {dell.shape}; binned x shape: {x.shape}")
    print("FITS maps were validated by path and size but were not read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
