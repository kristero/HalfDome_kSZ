#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CASES = [
    "no_noise",
    "goal_deproj0",
    "baseline_deproj0",
    "goal_deproj2",
    "baseline_deproj2",
]

BATTAGLIA12_THETA_BY_NAME = {
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

BATTAGLIA12_FILENAMES = {
    "no_noise": (
        "halfdome_fullsky_masked_no_noise_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    ),
    "goal_deproj0": (
        "halfdome_fullsky_masked_goal_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    ),
    "baseline_deproj0": (
        "halfdome_fullsky_masked_baseline_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj0_lmax7979.npy"
    ),
    "goal_deproj2": (
        "halfdome_fullsky_masked_goal_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj2_lmax7979.npy"
    ),
    "baseline_deproj2": (
        "halfdome_fullsky_masked_baseline_noise_cross_cl_m200c_nside4096_base_cosmo_fid_"
        "gaussbeam_2p0arcmin_so_fsky0p4_apo60p0arcmin_seed12345_deproj2_lmax7979.npy"
    ),
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


def ell_tag(ell_min: float | None, ell_max: float | None) -> str:
    def fmt(value: float | None, fallback: str) -> str:
        if value is None:
            return fallback
        value = float(value)
        if value.is_integer():
            text = str(int(value))
        else:
            text = f"{value:g}"
        return text.replace("-", "m").replace(".", "p")

    return f"ell{fmt(ell_min, 'min')}_{fmt(ell_max, 'max')}"


def scalar_string(value: Any, default: str = "") -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(()).item())
    return default


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
        raise ValueError(
            f"No binned ell values selected for ell_min={ell_min}, ell_max={ell_max}, "
            f"selection={selection!r}"
        )
    return np.asarray(mask, dtype=bool)


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


def make_bin_matrix(
    ell_unbinned: np.ndarray,
    bin_ell_min: np.ndarray,
    bin_ell_max: np.ndarray,
    weighting: str,
) -> np.ndarray:
    ell_unbinned = np.asarray(ell_unbinned, dtype=np.float64).reshape(-1)
    matrix = np.zeros((ell_unbinned.size, bin_ell_min.size), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(bin_ell_min, bin_ell_max)):
        idx = np.flatnonzero((ell_unbinned >= float(lo)) & (ell_unbinned <= float(hi)))
        if idx.size == 0:
            raise ValueError(f"No unbinned ell values found for bin {lo}-{hi}")
        weights = bin_weights(ell_unbinned[idx], weighting)
        matrix[idx, i] = weights / np.sum(weights)
    return matrix


def read_profile(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2 and 1 in arr.shape:
        return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.ascontiguousarray(arr[:, -1], dtype=np.float32)
    raise ValueError(f"Cannot interpret profile array from {path}: shape={arr.shape}")


def battaglia12_theta(param_names: np.ndarray) -> np.ndarray:
    missing = [name for name in param_names if str(name) not in BATTAGLIA12_THETA_BY_NAME]
    if missing:
        raise KeyError(f"Battaglia12 truth is missing parameters: {missing}")
    return np.asarray([BATTAGLIA12_THETA_BY_NAME[str(name)] for name in param_names], dtype=np.float32)


def battaglia12_obs_for_case(
    case: str,
    battaglia12_dir: Path,
    ell_unbinned: np.ndarray,
    dl_bin_matrix: np.ndarray,
    selected_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    if case not in BATTAGLIA12_FILENAMES:
        raise KeyError(f"No Battaglia12 filename configured for case={case!r}")
    path = battaglia12_dir / BATTAGLIA12_FILENAMES[case]
    if not path.is_file():
        raise FileNotFoundError(f"Battaglia12 observation profile not found for {case}: {path}")
    cl = read_profile(path)
    if cl.size != ell_unbinned.size:
        raise ValueError(f"{path} has length {cl.size}, expected {ell_unbinned.size}")
    dl_binned = np.asarray(cl, dtype=np.float64) @ dl_bin_matrix
    return np.ascontiguousarray(dl_binned[selected_mask], dtype=np.float32), str(path)


def resolve_obs_index(obs_index: int, n_rows: int) -> int:
    idx = int(obs_index)
    if idx < 0:
        idx = int(n_rows) + idx
    if not (0 <= idx < int(n_rows)):
        raise IndexError(f"obs_index={obs_index} resolves to {idx}, outside 0..{int(n_rows) - 1}")
    return idx


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Extract case-specific SBI-ready datasets from the combined SO multi-case dataset. "
            "The selected ell range is applied to the already binned Delta=200 x vectors."
        )
    )
    parser.add_argument(
        "--combined-dataset",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "so_multi_case_delta200_sbi_dataset.npz"),
        help="Combined multi-case NPZ from build_so_multi_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for case-specific NPZs. Default uses SBI_analysis/data_for_cluster/so_noise_sbi_cases_<elltag>.",
    )
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--ell-min", type=float, default=80.0)
    parser.add_argument("--ell-max", type=float, default=7979.0)
    parser.add_argument(
        "--ell-selection",
        choices=("center", "contained", "overlap"),
        default="center",
        help="How binned ell values are selected relative to --ell-min/--ell-max.",
    )
    parser.add_argument(
        "--obs-source",
        choices=("battaglia12", "dataset-row"),
        default="battaglia12",
        help="Observation to store in each case NPZ. Default is the saved Battaglia12 SO-noise profile.",
    )
    parser.add_argument(
        "--battaglia12-dir",
        default=str(root / "tSZ_visuals" / "outputs" / "so_noise_battaglia12_fiducial_local"),
        help="Directory containing the saved Battaglia12 fiducial SO-noise profiles.",
    )
    parser.add_argument("--obs-index", type=int, default=-1)
    parser.add_argument(
        "--test-last-n",
        type=int,
        default=100,
        help="Store these final row indices as diagnostics/test indices in each output NPZ.",
    )
    parser.add_argument("--no-compress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    combined_path = resolve_path(args.combined_dataset, root)
    battaglia12_dir = resolve_path(args.battaglia12_dir, root)
    tag = ell_tag(args.ell_min, args.ell_max)
    output_dir = (
        resolve_path(args.output_dir, root)
        if args.output_dir
        else root / "SBI_analysis" / "data_for_cluster" / f"so_noise_sbi_cases_{tag}_{args.obs_source}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not combined_path.is_file():
        raise FileNotFoundError(f"Combined dataset not found: {combined_path}")

    print(f"Reading combined dataset: {combined_path}")
    data = np.load(combined_path, allow_pickle=True)

    theta = np.asarray(data["theta"], dtype=np.float32)
    prior_low = np.asarray(data["prior_low"], dtype=np.float32)
    prior_high = np.asarray(data["prior_high"], dtype=np.float32)
    param_names = np.asarray(data["param_names"], dtype=str)
    theta_columns = np.asarray(data["theta_columns"], dtype=str) if "theta_columns" in data.files else param_names

    ell = np.asarray(data["ell_binned"] if "ell_binned" in data.files else data["ell"], dtype=np.float32)
    bin_ell_min = np.asarray(data["bin_ell_min"], dtype=np.float32)
    bin_ell_max = np.asarray(data["bin_ell_max"], dtype=np.float32)
    bin_counts = np.asarray(data["bin_counts"], dtype=np.int64) if "bin_counts" in data.files else np.ones_like(ell, dtype=np.int64)
    ell_unbinned = np.asarray(data["ell_unbinned"], dtype=np.float32) if "ell_unbinned" in data.files else None

    binning = {}
    if "binning_json" in data.files:
        binning = json.loads(scalar_string(data["binning_json"], "{}"))
    weighting = str(binning.get("weighting", "2ell_plus_1"))

    mask = ell_mask(ell, bin_ell_min, bin_ell_max, args.ell_min, args.ell_max, args.ell_selection)
    selected = np.flatnonzero(mask)
    n_rows = int(theta.shape[0])
    obs_index = resolve_obs_index(args.obs_index, n_rows)
    obs_theta_battaglia12 = battaglia12_theta(param_names)

    dl_bin_matrix = None
    if args.obs_source == "battaglia12":
        if ell_unbinned is None:
            raise KeyError("Combined dataset must contain ell_unbinned to build Battaglia12 observations.")
        full_bin_matrix = make_bin_matrix(ell_unbinned, bin_ell_min, bin_ell_max, weighting)
        dl_factor = ell_unbinned.astype(np.float64) * (ell_unbinned.astype(np.float64) + 1.0) / (2.0 * np.pi)
        dl_bin_matrix = dl_factor[:, None] * full_bin_matrix

    test_last_n = int(args.test_last_n)
    if test_last_n < 0:
        raise ValueError("--test-last-n must be non-negative")
    if test_last_n > n_rows:
        raise ValueError(f"--test-last-n={test_last_n} exceeds dataset rows={n_rows}")
    test_indices = np.arange(n_rows - test_last_n, n_rows, dtype=np.int64) if test_last_n else np.empty(0, dtype=np.int64)

    source_case_names = [str(x) for x in data["case_names"]] if "case_names" in data.files else DEFAULT_CASES
    missing_cases = [case for case in args.cases if f"x_{case}" not in data.files]
    if missing_cases:
        raise KeyError(f"Combined dataset is missing cases {missing_cases}; available cases={source_case_names}")

    metadata_base = {
        "source_combined_dataset": str(combined_path),
        "n_rows": n_rows,
        "ell_min": float(args.ell_min),
        "ell_max": float(args.ell_max),
        "ell_selection": args.ell_selection,
        "ell_tag": tag,
        "selected_bin_indices": selected.astype(int),
        "n_selected_bins": int(selected.size),
        "obs_source": args.obs_source,
        "obs_index": int(obs_index) if args.obs_source == "dataset-row" else None,
        "test_last_n": int(test_last_n),
        "test_indices_start": int(test_indices[0]) if test_indices.size else None,
        "test_indices_stop": int(test_indices[-1]) if test_indices.size else None,
    }

    save_func = np.savez if args.no_compress else np.savez_compressed
    index: dict[str, Any] = {
        "combined_dataset": str(combined_path),
        "output_dir": str(output_dir),
        "ell_tag": tag,
        "ell_min": float(args.ell_min),
        "ell_max": float(args.ell_max),
        "ell_selection": args.ell_selection,
        "cases": {},
    }

    for case in args.cases:
        x_case = np.asarray(data[f"x_{case}"], dtype=np.float32)
        if x_case.ndim != 2:
            raise ValueError(f"x_{case} must be 2D, got shape {x_case.shape}")
        if x_case.shape[0] != n_rows:
            raise ValueError(f"x_{case} rows {x_case.shape[0]} do not match theta rows {n_rows}")
        if x_case.shape[1] != ell.size:
            raise ValueError(f"x_{case} dim {x_case.shape[1]} does not match ell size {ell.size}")

        x_selected = np.ascontiguousarray(x_case[:, mask], dtype=np.float32)
        if args.obs_source == "battaglia12":
            assert dl_bin_matrix is not None
            obs, obs_profile_path = battaglia12_obs_for_case(case, battaglia12_dir, ell_unbinned, dl_bin_matrix, mask)
            obs_theta = obs_theta_battaglia12
            obs_index_payload = np.asarray(-1, dtype=np.int64)
        else:
            obs = np.ascontiguousarray(x_selected[obs_index], dtype=np.float32)
            obs_theta = np.ascontiguousarray(theta[obs_index], dtype=np.float32)
            obs_profile_path = ""
            obs_index_payload = np.asarray(obs_index, dtype=np.int64)

        metadata = dict(metadata_base)
        metadata.update(
            {
                "case": case,
                "x_shape": list(x_selected.shape),
                "obs_case": case,
                "obs_source": args.obs_source,
                "obs_profile_path": obs_profile_path,
            }
        )

        output_path = output_dir / f"so_{case}_{tag}_sbi_run.npz"
        payload = {
            "theta": theta,
            "x": x_selected,
            "obs": obs,
            "obs_theta": np.ascontiguousarray(obs_theta, dtype=np.float32),
            "obs_index": obs_index_payload,
            "obs_source": np.asarray(args.obs_source),
            "obs_profile_path": np.asarray(obs_profile_path),
            "test_indices": test_indices,
            "ell": np.ascontiguousarray(ell[mask], dtype=np.float32),
            "ell_binned": np.ascontiguousarray(ell[mask], dtype=np.float32),
            "bin_counts": np.ascontiguousarray(bin_counts[mask], dtype=np.int64),
            "bin_ell_min": np.ascontiguousarray(bin_ell_min[mask], dtype=np.float32),
            "bin_ell_max": np.ascontiguousarray(bin_ell_max[mask], dtype=np.float32),
            "prior_low": prior_low,
            "prior_high": prior_high,
            "param_names": param_names,
            "theta_columns": theta_columns,
            "case_name": np.asarray(case),
            "source_case": np.asarray(case),
            "source_combined_dataset": np.asarray(str(combined_path)),
            "ell_min": np.asarray(float(args.ell_min), dtype=np.float32),
            "ell_max": np.asarray(float(args.ell_max), dtype=np.float32),
            "ell_selection": np.asarray(args.ell_selection),
            "metadata_json": np.asarray(json.dumps(jsonable(metadata), sort_keys=True)),
        }
        save_func(output_path, **payload)

        index["cases"][case] = {
            "path": str(output_path),
            "n_rows": n_rows,
            "x_shape": list(x_selected.shape),
            "obs_index": int(obs_index) if args.obs_source == "dataset-row" else None,
            "obs_source": args.obs_source,
            "obs_profile_path": obs_profile_path,
            "test_last_n": int(test_last_n),
        }
        print(f"Wrote {case}: {output_path}  x_shape={x_selected.shape}")

    index_path = output_dir / "case_dataset_index.json"
    write_json(index_path, index)
    print(f"Wrote index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
