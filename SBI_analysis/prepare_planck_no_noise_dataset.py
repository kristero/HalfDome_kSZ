#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_SIZES = "1024,2048,4096,8192,16384,32768,50e3,70e3,85e3,100e3"
DEFAULT_SOURCE_DATASET = Path("emulator_tSZ/outputs/binned_16e3_16/sbi16e3_bin16_dataset_100_000.npz")
DEFAULT_OBSERVED_DL = Path("HPC_output/HalfDome/tSZ_HalfDome_fiducial_nside4096_cluster.npy")
DEFAULT_OUTPUT = Path("SBI_analysis/data_for_cluster/planck_binned16_no_noise_sbi_run.npz")


def parse_dataset_size(value: str) -> int:
    raw = str(value).strip().lower().replace("_", "")
    if not raw:
        raise ValueError("Empty dataset size")
    if raw.endswith("k"):
        size = float(raw[:-1]) * 1_000.0
    else:
        size = float(raw)
    rounded = int(round(size))
    if rounded <= 0 or not np.isclose(size, rounded):
        raise ValueError(f"Dataset size must be a positive integer count, got {value!r}")
    return rounded


def parse_dataset_sizes(value: str) -> list[int]:
    parts = [part for part in str(value or "").replace(";", ",").replace(" ", ",").split(",") if part]
    sizes = [parse_dataset_size(part) for part in parts]
    unique_sizes: list[int] = []
    seen: set[int] = set()
    for size in sizes:
        if size not in seen:
            unique_sizes.append(size)
            seen.add(size)
    return unique_sizes


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return root / path


def npz_scalar_to_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return default


def json_from_npz(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    raw = npz_scalar_to_str(data, key, default="")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def bin_weights(ell: np.ndarray, weighting: str) -> np.ndarray:
    weighting = str(weighting or "uniform").lower()
    if weighting in {"uniform", "none", "flat"}:
        return np.ones_like(ell, dtype=np.float64)
    if weighting == "ell":
        return ell.astype(np.float64)
    if weighting in {"2ell_plus_1", "modes", "mode_count"}:
        return 2.0 * ell.astype(np.float64) + 1.0
    raise ValueError(f"Unsupported bin weighting {weighting!r}")


def binned_log10_dl_from_ranges(
    dl_values: np.ndarray,
    bin_min: np.ndarray,
    bin_max: np.ndarray,
    *,
    statistic: str = "mean",
    weighting: str = "uniform",
    floor_dl: float = 1.0e-40,
) -> np.ndarray:
    dl_values = np.asarray(dl_values, dtype=np.float64).reshape(-1)
    bin_min = np.asarray(bin_min, dtype=np.int64).reshape(-1)
    bin_max = np.asarray(bin_max, dtype=np.int64).reshape(-1)
    if bin_min.shape != bin_max.shape:
        raise ValueError("bin_min and bin_max must have the same shape")

    pieces = []
    statistic = str(statistic or "mean").lower()
    for lo, hi in zip(bin_min, bin_max):
        if lo < 0 or hi >= dl_values.size or hi < lo:
            raise ValueError(
                f"Invalid bin range {lo}:{hi} for observed D_l vector of length {dl_values.size}"
            )
        ell = np.arange(lo, hi + 1, dtype=np.float64)
        log10_dl = np.log10(np.maximum(dl_values[lo : hi + 1], float(floor_dl)))
        if statistic == "mean":
            pieces.append(float(np.average(log10_dl, weights=bin_weights(ell, weighting))))
        elif statistic == "median":
            pieces.append(float(np.median(log10_dl)))
        else:
            raise ValueError(f"Unsupported bin statistic {statistic!r}")
    return np.asarray(pieces, dtype=np.float32)


def build_planck_no_noise_dataset(
    source_dataset_path: str | Path,
    observed_dl_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_dataset_path = Path(source_dataset_path).expanduser()
    observed_dl_path = Path(observed_dl_path).expanduser()
    if not source_dataset_path.is_file():
        raise FileNotFoundError(f"Missing Planck-binned SBI source dataset: {source_dataset_path}")
    if not observed_dl_path.is_file():
        raise FileNotFoundError(f"Missing synthetic observed D_l vector: {observed_dl_path}")

    with np.load(source_dataset_path, allow_pickle=True) as data:
        required = {"theta", "ell", "bin_ell_min", "bin_ell_max", "prior_low", "prior_high"}
        missing = sorted(required - set(data.files))
        if missing:
            raise KeyError(f"{source_dataset_path} is missing required keys: {missing}")

        x_key = "x_log10_dl" if "x_log10_dl" in data.files else "x"
        if x_key not in data.files:
            raise KeyError(f"{source_dataset_path} must contain x_log10_dl or x")

        theta = np.ascontiguousarray(data["theta"], dtype=np.float32)
        x = np.ascontiguousarray(data[x_key], dtype=np.float32)
        ell = np.ascontiguousarray(data["ell"], dtype=np.float32).reshape(-1)
        bin_min = np.asarray(data["bin_ell_min"], dtype=np.int64).reshape(-1)
        bin_max = np.asarray(data["bin_ell_max"], dtype=np.int64).reshape(-1)
        if "bin_counts" in data.files:
            bin_counts = np.ascontiguousarray(data["bin_counts"], dtype=np.int64)
        else:
            bin_counts = np.ascontiguousarray(bin_max - bin_min + 1, dtype=np.int64)
        prior_low = np.ascontiguousarray(data["prior_low"], dtype=np.float32)
        prior_high = np.ascontiguousarray(data["prior_high"], dtype=np.float32)
        if "theta_columns" in data.files:
            param_names = np.asarray(data["theta_columns"]).astype(str)
        elif "x_columns" in data.files:
            param_names = np.asarray(data["x_columns"]).astype(str)
        else:
            param_names = np.asarray([f"theta_{idx}" for idx in range(theta.shape[1])])
        binning = json_from_npz(data, "binning_json")

    if theta.ndim != 2 or x.ndim != 2:
        raise ValueError(f"Expected theta and x to be 2D, got {theta.shape} and {x.shape}")
    if theta.shape[0] != x.shape[0]:
        raise ValueError(f"theta rows {theta.shape[0]} do not match x rows {x.shape[0]}")
    if x.shape[1] != 16 or ell.size != 16:
        raise ValueError(f"Planck setup expects 16 bins, got x={x.shape}, ell={ell.shape}")
    if not (bin_min.size == bin_max.size == ell.size):
        raise ValueError("Planck bin metadata must have one range per ell bin")

    observed_dl = np.asarray(np.load(observed_dl_path, allow_pickle=False), dtype=np.float64).reshape(-1)
    statistic = str(binning.get("statistic", "mean"))
    weighting = str(binning.get("weighting", "uniform"))
    obs = binned_log10_dl_from_ranges(
        observed_dl,
        bin_min,
        bin_max,
        statistic=statistic,
        weighting=weighting,
    )
    if obs.shape != (16,):
        raise ValueError(f"Planck no-noise observation should have shape (16,), got {obs.shape}")

    payload: dict[str, np.ndarray] = {
        "theta": theta,
        "x": x,
        "obs": np.ascontiguousarray(obs, dtype=np.float32),
        "ell": ell,
        "prior_low": prior_low,
        "prior_high": prior_high,
        "param_names": np.asarray(param_names),
        "bin_counts": bin_counts,
        "bin_ell_min": np.ascontiguousarray(bin_min, dtype=np.float32),
        "bin_ell_max": np.ascontiguousarray(bin_max, dtype=np.float32),
        "noise_enabled": np.asarray(False),
        "noise_mode": np.asarray("none"),
        "source_dataset": np.asarray(str(source_dataset_path)),
        "source_observed_dl": np.asarray(str(observed_dl_path)),
        "binning_json": np.asarray(json.dumps(binning, sort_keys=True)),
    }
    summary = {
        "source_dataset": str(source_dataset_path),
        "source_observed_dl": str(observed_dl_path),
        "n_rows": int(theta.shape[0]),
        "x_dim": int(x.shape[1]),
        "theta_dim": int(theta.shape[1]),
        "ell_min": float(ell[0]),
        "ell_max": float(ell[-1]),
        "bin_ell_min": bin_min.astype(int).tolist(),
        "bin_ell_max": bin_max.astype(int).tolist(),
        "noise_enabled": False,
        "noise_mode": "none",
        "binning_statistic": statistic,
        "binning_weighting": weighting,
    }
    return payload, summary


def write_dataset(output_path: str | Path, payload: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")


def validate_dataset_sizes(sizes: list[int], n_rows: int) -> None:
    if not sizes:
        raise ValueError("At least one dataset size is required")
    too_large = [size for size in sizes if size > n_rows]
    if too_large:
        raise ValueError(f"Dataset sizes exceed available rows {n_rows}: {too_large}")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Prepare the Planck 16-bin, no-noise SBI dataset for cluster dataset-size sweeps."
    )
    parser.add_argument("--source-dataset", default=str(root / DEFAULT_SOURCE_DATASET))
    parser.add_argument("--observed-dl", default=str(root / DEFAULT_OBSERVED_DL))
    parser.add_argument("--output", default=str(root / DEFAULT_OUTPUT))
    parser.add_argument("--dataset-sizes", default=DEFAULT_DATASET_SIZES)
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and dataset sizes without writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload, summary = build_planck_no_noise_dataset(args.source_dataset, args.observed_dl)
    sizes = parse_dataset_sizes(args.dataset_sizes)
    validate_dataset_sizes(sizes, int(summary["n_rows"]))
    summary["dataset_sizes_checked"] = sizes
    if args.check_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    write_dataset(args.output, payload, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
