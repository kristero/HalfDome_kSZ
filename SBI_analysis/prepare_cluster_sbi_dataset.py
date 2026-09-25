#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
DEFAULT_PARAM_NAMES = np.asarray(list(SOBOL_PRIOR_BOUNDS.keys()))
DEFAULT_PRIOR_LOW = np.asarray([SOBOL_PRIOR_BOUNDS[name][0] for name in DEFAULT_PARAM_NAMES], dtype=np.float32)
DEFAULT_PRIOR_HIGH = np.asarray([SOBOL_PRIOR_BOUNDS[name][1] for name in DEFAULT_PARAM_NAMES], dtype=np.float32)
FLOOR_DL = 1.0e-40


@dataclass(frozen=True)
class BinningSpec:
    name: str
    bin_min: np.ndarray
    bin_max: np.ndarray
    ell: np.ndarray
    statistic: str
    weighting: str

    @property
    def n_bins(self) -> int:
        return int(self.ell.size)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def npz_scalar_to_str(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return default


def npz_json_dict(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    raw = npz_scalar_to_str(data, key, default="")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
    weighting = str(weighting or "uniform").lower()
    if weighting in {"uniform", "none", "flat"}:
        return np.ones_like(ell_values, dtype=np.float64)
    if weighting == "ell":
        return ell_values.astype(np.float64)
    if weighting in {"2ell_plus_1", "modes", "mode_count"}:
        return 2.0 * ell_values.astype(np.float64) + 1.0
    raise ValueError(f"Unsupported bin weighting {weighting!r}")


def weighted_center(lo: int, hi: int, weighting: str) -> float:
    ell_values = np.arange(int(lo), int(hi) + 1, dtype=np.float64)
    return float(np.average(ell_values, weights=bin_weights(ell_values, weighting)))


def make_planck_spec(statistic: str = "mean", weighting: str = "uniform") -> BinningSpec:
    bin_min = np.asarray([21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085], dtype=np.int64)
    bin_max = np.asarray([26, 34, 45, 59, 77, 101, 132, 172, 223, 291, 379, 493, 641, 834, 1084, 1410], dtype=np.int64)
    ell = np.asarray([weighted_center(lo, hi, weighting) for lo, hi in zip(bin_min, bin_max)], dtype=np.float32)
    return BinningSpec("PLANCK", bin_min, bin_max, ell, statistic, weighting)


def make_so_spec(statistic: str = "mean", weighting: str = "2ell_plus_1") -> BinningSpec:
    edges = np.r_[np.arange(80, 7881, 200), 7979].astype(np.int64)
    bin_min = edges[:-1].copy()
    bin_max = edges[1:].copy()
    bin_max[:-1] -= 1
    ell = np.asarray([weighted_center(lo, hi, weighting) for lo, hi in zip(bin_min, bin_max)], dtype=np.float32)
    return BinningSpec("SO", bin_min, bin_max, ell, statistic, weighting)


def make_binning_spec(name: str, statistic: str | None = None, weighting: str | None = None) -> BinningSpec:
    name = str(name).strip().upper()
    if name == "PLANCK":
        return make_planck_spec(statistic or "mean", weighting or "uniform")
    if name == "SO":
        return make_so_spec(statistic or "mean", weighting or "2ell_plus_1")
    raise ValueError("binning must be PLANCK or SO")


def bin_member_indices(source_ell: np.ndarray, spec: BinningSpec) -> list[np.ndarray]:
    source_ell = np.asarray(source_ell, dtype=np.float64).reshape(-1)
    members: list[np.ndarray] = []
    for lo, hi in zip(spec.bin_min, spec.bin_max):
        idx = np.flatnonzero((source_ell >= float(lo)) & (source_ell <= float(hi)))
        if idx.size == 0:
            raise ValueError(
                f"No source ell values found for {spec.name} bin {lo}-{hi}. "
                f"Source ell range is {source_ell.min()}-{source_ell.max()}."
            )
        members.append(idx)
    return members


def bin_last_axis_log10(values: np.ndarray, source_ell: np.ndarray, spec: BinningSpec) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_ell = np.asarray(source_ell, dtype=np.float64).reshape(-1)
    if values.shape[-1] != source_ell.size:
        raise ValueError(f"values last axis {values.shape[-1]} does not match source ell length {source_ell.size}")

    pieces = []
    statistic = str(spec.statistic).lower()
    for idx in bin_member_indices(source_ell, spec):
        part = values[..., idx]
        if statistic == "mean":
            weights = bin_weights(source_ell[idx], spec.weighting)
            pieces.append(np.average(part, axis=-1, weights=weights))
        elif statistic == "median":
            pieces.append(np.median(part, axis=-1))
        else:
            raise ValueError(f"Unsupported bin statistic {spec.statistic!r}")
    return np.ascontiguousarray(np.stack(pieces, axis=-1), dtype=np.float32)


def cl_to_log10_dl(cl: np.ndarray, ell: np.ndarray, floor_dl: float = FLOOR_DL) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    factor = ell * (ell + 1.0) / (2.0 * math.pi)
    return np.log10(np.maximum(np.asarray(cl, dtype=np.float64) * factor, float(floor_dl)))


def dl_to_log10_dl(dl: np.ndarray, floor_dl: float = FLOOR_DL) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(dl, dtype=np.float64), float(floor_dl)))


def values_to_log10_dl(values: np.ndarray, ell: np.ndarray, kind: str) -> np.ndarray:
    kind = str(kind).lower()
    if kind == "log10_dl":
        return np.asarray(values, dtype=np.float64)
    if kind == "dl":
        return dl_to_log10_dl(values)
    if kind == "cl":
        return cl_to_log10_dl(values, ell)
    raise ValueError("kind must be cl, dl, or log10_dl")


def gaussian_beam_window(ell: np.ndarray, fwhm_arcmin: float) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    fwhm_arcmin = float(fwhm_arcmin)
    if fwhm_arcmin < 0.0:
        raise ValueError("gaussian beam FWHM must be non-negative")
    if fwhm_arcmin == 0.0:
        return np.ones_like(ell, dtype=np.float64)
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
    sigma_rad = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma_rad**2)


def apply_gaussian_beam_to_log10_dl(values: np.ndarray, ell: np.ndarray, fwhm_arcmin: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    if values.shape[-1] != ell.size:
        raise ValueError(f"Beam ell length {ell.size} does not match values last dimension {values.shape[-1]}")
    beam = gaussian_beam_window(ell, fwhm_arcmin)
    beam_log_factor = np.log10(np.maximum(beam**2, FLOOR_DL)).astype(np.float32)
    return np.ascontiguousarray(values + beam_log_factor, dtype=np.float32)


def infer_matrix_kind(key: str, target_kind: str, values: np.ndarray) -> str:
    key_l = str(key).lower()
    target_l = str(target_kind).lower()
    if "log10" in key_l or "log10" in target_l:
        return "log10_dl"
    if key_l.startswith("cl") or "c_l" in key_l:
        return "cl"
    if "dl" in key_l or "d_l" in key_l:
        return "dl"
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size:
        med = float(np.nanmedian(finite[: min(finite.size, 100000)]))
        if med < 0.0:
            return "log10_dl"
        if abs(med) > 1.0e-15:
            return "dl"
    return "cl"


def choose_source_matrix(data: np.lib.npyio.NpzFile, source_mode: str, xgpaint_cl_key: str) -> tuple[str, str]:
    if str(source_mode).lower() == "xgpaint":
        if xgpaint_cl_key not in data.files:
            cl_keys = [key for key in data.files if key.startswith("cl")]
            raise KeyError(f"{xgpaint_cl_key!r} not found. Available C_l keys: {cl_keys}")
        return xgpaint_cl_key, "cl"

    for key in ("x_log10_dl", "x_binned", "x"):
        if key in data.files:
            return key, infer_matrix_kind(key, npz_scalar_to_str(data, "target_kind", default=""), data[key])
    for key in ("cl_mean", "cl_y100", "cl_concat"):
        if key in data.files:
            return key, "cl"
    raise KeyError(f"Could not choose an x matrix. Available keys: {data.files}")


def source_param_names(data: np.lib.npyio.NpzFile, theta: np.ndarray) -> np.ndarray:
    for key in ("theta_columns", "param_names", "x_columns"):
        if key in data.files:
            return np.asarray(data[key]).astype(str)
    if theta.shape[1] == DEFAULT_PARAM_NAMES.size:
        return DEFAULT_PARAM_NAMES.copy()
    return np.asarray([f"theta_{idx}" for idx in range(theta.shape[1])])


def source_prior_bounds(data: np.lib.npyio.NpzFile, param_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "prior_low" in data.files and "prior_high" in data.files:
        return np.asarray(data["prior_low"], dtype=np.float32), np.asarray(data["prior_high"], dtype=np.float32)
    lows = []
    highs = []
    for name in param_names:
        if str(name) not in SOBOL_PRIOR_BOUNDS:
            raise KeyError(f"Missing prior bounds for parameter {name!r}")
        lo, hi = SOBOL_PRIOR_BOUNDS[str(name)]
        lows.append(lo)
        highs.append(hi)
    return np.asarray(lows, dtype=np.float32), np.asarray(highs, dtype=np.float32)


def direct_grid_match(source_ell: np.ndarray, spec: BinningSpec, n_columns: int, assume_prebinned: bool) -> bool:
    if int(n_columns) != spec.n_bins:
        return False
    source_ell = np.asarray(source_ell, dtype=np.float64).reshape(-1)
    if source_ell.size != spec.n_bins:
        return bool(assume_prebinned)
    if np.allclose(source_ell, spec.ell, rtol=1.0e-4, atol=1.0e-3):
        return True
    return bool(assume_prebinned)


def process_matrix_to_target_bins(
    matrix: np.ndarray,
    source_ell: np.ndarray,
    matrix_kind: str,
    spec: BinningSpec,
    chunk_rows: int,
    assume_prebinned: bool,
) -> np.ndarray:
    matrix = np.asarray(matrix)
    source_ell = np.asarray(source_ell, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"simulation matrix must be 2D, got {matrix.shape}")
    if direct_grid_match(source_ell, spec, matrix.shape[1], assume_prebinned) and matrix_kind == "log10_dl":
        return np.ascontiguousarray(matrix, dtype=np.float32)
    if matrix.shape[1] != source_ell.size:
        raise ValueError(f"matrix columns {matrix.shape[1]} do not match ell length {source_ell.size}")

    out = np.empty((matrix.shape[0], spec.n_bins), dtype=np.float32)
    for start in range(0, matrix.shape[0], int(chunk_rows)):
        stop = min(start + int(chunk_rows), matrix.shape[0])
        log10_dl = values_to_log10_dl(matrix[start:stop], source_ell, matrix_kind)
        out[start:stop] = bin_last_axis_log10(log10_dl, source_ell, spec)
    return np.ascontiguousarray(out, dtype=np.float32)


def load_and_process_simulations(
    path: Path,
    source_mode: str,
    spec: BinningSpec,
    xgpaint_cl_key: str,
    chunk_rows: int,
    assume_prebinned: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source dataset: {path}")

    with np.load(path, allow_pickle=True) as data:
        if "theta" not in data.files:
            raise KeyError(f"{path} is missing theta")
        if "ell" not in data.files:
            raise KeyError(f"{path} is missing ell")

        theta = np.ascontiguousarray(data["theta"], dtype=np.float32)
        param_names = source_param_names(data, theta)
        prior_low, prior_high = source_prior_bounds(data, param_names)
        source_ell = np.asarray(data["ell"], dtype=np.float64).reshape(-1)
        x_key, matrix_kind = choose_source_matrix(data, source_mode, xgpaint_cl_key)
        x = process_matrix_to_target_bins(data[x_key], source_ell, matrix_kind, spec, chunk_rows, assume_prebinned)
        source_binning_json = npz_json_dict(data, "binning_json")

    if theta.shape[0] != x.shape[0]:
        raise ValueError(f"theta rows {theta.shape[0]} do not match x rows {x.shape[0]}")

    payload = {
        "theta": theta,
        "x": x,
        "ell": np.ascontiguousarray(spec.ell, dtype=np.float32),
        "prior_low": np.ascontiguousarray(prior_low, dtype=np.float32),
        "prior_high": np.ascontiguousarray(prior_high, dtype=np.float32),
        "param_names": np.asarray(param_names).astype(str),
    }
    summary = {
        "source_dataset": str(path),
        "source_mode": source_mode,
        "source_x_key": x_key,
        "source_matrix_kind": matrix_kind,
        "source_ell_min": float(np.nanmin(source_ell)),
        "source_ell_max": float(np.nanmax(source_ell)),
        "source_binning_json": source_binning_json,
        "n_rows": int(theta.shape[0]),
        "theta_dim": int(theta.shape[1]),
        "x_dim": int(x.shape[1]),
    }
    return payload, summary


def read_spectrum_file(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).squeeze(), None
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            ell = np.asarray(data["ell"], dtype=np.float64).squeeze() if "ell" in data.files else None
            for key in (
                "obs",
                "x_obs",
                "x_obs_log10_dl",
                "x_observed_log10",
                "log10_dl",
                "dl",
                "d_l",
                "cl",
                "c_l",
                "profile",
                "data",
                "x",
            ):
                if key in data.files:
                    return np.asarray(data[key], dtype=np.float64).squeeze(), ell
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]], dtype=np.float64).squeeze(), ell
            raise ValueError(f"Could not choose observed spectrum inside {path}; keys={data.files}")
    if suffix in (".csv", ".txt", ".dat"):
        raw = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return raw[:, 1], raw[:, 0]
        return raw.squeeze(), None
    raise ValueError(f"Unsupported observed spectrum extension: {path}")


def infer_observed_kind(path: Path, spectrum: np.ndarray, ell: np.ndarray) -> str:
    name = normalize_name(path.stem)
    tokens = set(name.split("_"))
    if "log10" in tokens and ({"dl", "d", "ell"} & tokens):
        return "log10_dl"
    if "dl" in tokens or "dell" in tokens or "d_ell" in name:
        return "dl"
    if "cl" in tokens or "cell" in tokens or "c_ell" in name:
        return "cl"

    good = np.isfinite(spectrum) & np.isfinite(ell)
    finite = spectrum[good] if np.any(good) else spectrum[np.isfinite(spectrum)]
    if finite.size == 0:
        raise ValueError(f"Observed spectrum {path} has no finite values")
    med_signed = float(np.nanmedian(finite))
    med_abs = float(np.nanmedian(np.abs(finite)))
    if med_signed < 0.0:
        return "log10_dl"
    return "dl" if med_abs > 1.0e-15 else "cl"


def process_observation_to_target_bins(
    path: Path,
    input_kind: str,
    spec: BinningSpec,
    assume_prebinned: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    spectrum, obs_ell = read_spectrum_file(path)
    spectrum = np.asarray(spectrum, dtype=np.float64).squeeze()
    if spectrum.ndim != 1:
        raise ValueError(f"Observed spectrum must be 1D after loading, got {spectrum.shape}")
    if obs_ell is None:
        obs_ell = np.arange(spectrum.size, dtype=np.float64)
    else:
        obs_ell = np.asarray(obs_ell, dtype=np.float64).reshape(-1)
    if obs_ell.size != spectrum.size:
        raise ValueError(f"Observed ell length {obs_ell.size} does not match spectrum length {spectrum.size}")

    kind = infer_observed_kind(path, spectrum, obs_ell) if str(input_kind).lower() == "auto" else str(input_kind).lower()
    if spectrum.size == spec.n_bins and assume_prebinned:
        obs = np.ascontiguousarray(values_to_log10_dl(spectrum, spec.ell, kind), dtype=np.float32)
        direct = True
    else:
        log10_dl = values_to_log10_dl(spectrum, obs_ell, kind)
        obs = bin_last_axis_log10(log10_dl.reshape(1, -1), obs_ell, spec).reshape(-1)
        direct = False

    summary = {
        "obs_path": str(path),
        "obs_input_kind_requested": input_kind,
        "obs_input_kind_used": kind,
        "obs_direct_prebinned": bool(direct),
        "obs_source_length": int(spectrum.size),
        "obs_source_ell_min": float(np.nanmin(obs_ell)),
        "obs_source_ell_max": float(np.nanmax(obs_ell)),
    }
    return obs, summary


def default_emulated_dataset(root: Path, binning: str) -> Path:
    if str(binning).upper() == "SO":
        return root / "emulator_tSZ" / "outputs" / "binned_40" / "sbi_100k_uniform_prior_emulated_log10_dl.npz"
    return root / "emulator_tSZ" / "outputs" / "binned_16e3_16" / "sbi16e3_bin16_dataset_100_000.npz"


def default_xgpaint_dataset(root: Path) -> Path:
    return root / "HPC_output" / "HalfDome" / "ydata_12274" / "sbi_battaglia_y100_12274.npz"


def default_obs_path(root: Path, binning: str) -> Path:
    if str(binning).upper() == "SO":
        return root / "emulator_tSZ" / "outputs" / "binned_40" / "x_obs_log10_dl.npy"
    return root / "HPC_output" / "HalfDome" / "tSZ_HalfDome_fiducial_nside4096_cluster.npy"


def resolve_observation_path(args: argparse.Namespace, root: Path, source_dataset: Path) -> Path:
    if args.obs_path:
        return resolve_path(args.obs_path, root)
    if args.obs_mode == "emulated":
        candidate = source_dataset.parent / "x_obs_log10_dl.npy"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"--obs-mode emulated expected {candidate}. Provide --obs-path to choose the emulated observation."
        )
    return default_obs_path(root, args.binning)


def validate_payload(prepared: dict[str, np.ndarray]) -> None:
    if prepared["theta"].ndim != 2:
        raise ValueError(f"theta must be 2D, got {prepared['theta'].shape}")
    if prepared["x"].ndim != 2:
        raise ValueError(f"x must be 2D, got {prepared['x'].shape}")
    if prepared["theta"].shape[0] != prepared["x"].shape[0]:
        raise ValueError("theta and x row counts do not match")
    if prepared["obs"].shape != (prepared["x"].shape[1],):
        raise ValueError(f"obs shape {prepared['obs'].shape} does not match x dimension {prepared['x'].shape[1]}")
    for key in ("theta", "x", "obs", "ell", "prior_low", "prior_high"):
        if not np.all(np.isfinite(prepared[key])):
            raise ValueError(f"{key} contains non-finite values")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Prepare a binned SBI dataset for run_sbi_for_cluster.pbs.")
    parser.add_argument("--source-mode", choices=("emulated", "xgpaint"), default="emulated")
    parser.add_argument("--binning", choices=("SO", "PLANCK"), default="SO")
    parser.add_argument("--source-dataset", default="")
    parser.add_argument("--xgpaint-cl-key", default="cl_mean")
    parser.add_argument(
        "--obs-mode",
        choices=("default", "emulated"),
        default="default",
        help="emulated means use x_obs_log10_dl.npy next to --source-dataset unless --obs-path is supplied.",
    )
    parser.add_argument("--obs-path", default="")
    parser.add_argument("--obs-input-kind", choices=("auto", "cl", "dl", "log10_dl"), default="auto")
    parser.add_argument("--apply-gaussian-beam", action="store_true")
    parser.add_argument("--gaussian-beam-fwhm-arcmin", type=float, default=0.0)
    parser.add_argument("--bin-statistic", default="")
    parser.add_argument("--bin-weighting", default="")
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--no-assume-prebinned", action="store_true")
    parser.add_argument(
        "--output",
        default=str(root / "SBI_analysis" / "data_for_cluster" / "prepared_cluster_sbi_run.npz"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    if args.source_dataset:
        source_dataset = resolve_path(args.source_dataset, root)
    elif args.source_mode == "emulated":
        source_dataset = default_emulated_dataset(root, args.binning)
    else:
        source_dataset = default_xgpaint_dataset(root)
    obs_path = resolve_observation_path(args, root, source_dataset)
    output_path = resolve_path(args.output, root)
    assume_prebinned = not bool(args.no_assume_prebinned)

    spec = make_binning_spec(args.binning, args.bin_statistic or None, args.bin_weighting or None)
    prepared, simulation_summary = load_and_process_simulations(
        source_dataset,
        args.source_mode,
        spec,
        args.xgpaint_cl_key,
        args.chunk_rows,
        assume_prebinned,
    )
    obs, obs_summary = process_observation_to_target_bins(obs_path, args.obs_input_kind, spec, assume_prebinned)
    prepared["obs"] = np.ascontiguousarray(obs, dtype=np.float32)

    beam_fwhm = float(args.gaussian_beam_fwhm_arcmin if args.apply_gaussian_beam else 0.0)
    if args.apply_gaussian_beam:
        prepared["x"] = apply_gaussian_beam_to_log10_dl(prepared["x"], prepared["ell"], beam_fwhm)
        prepared["obs"] = apply_gaussian_beam_to_log10_dl(prepared["obs"], prepared["ell"], beam_fwhm)

    validate_payload(prepared)

    binning_summary = {
        "name": spec.name,
        "n_bins": spec.n_bins,
        "statistic": spec.statistic,
        "weighting": spec.weighting,
        "bin_ell_min": spec.bin_min.astype(int).tolist(),
        "bin_ell_max": spec.bin_max.astype(int).tolist(),
    }
    beam_summary = {
        "applied_in_preparation": bool(args.apply_gaussian_beam),
        "fwhm_arcmin": beam_fwhm,
        "runner_should_use_mode": "off",
        "runner_should_use_fwhm_arcmin": 0.0,
    }
    summary = {
        **simulation_summary,
        **obs_summary,
        "binning": binning_summary,
        "gaussian_beam": beam_summary,
        "output_path": str(output_path),
    }
    save_payload = {
        **prepared,
        "bin_counts": np.ascontiguousarray(spec.bin_max - spec.bin_min + 1, dtype=np.int64),
        "bin_ell_min": np.ascontiguousarray(spec.bin_min, dtype=np.float32),
        "bin_ell_max": np.ascontiguousarray(spec.bin_max, dtype=np.float32),
        "source_dataset": np.asarray(str(source_dataset)),
        "source_mode": np.asarray(str(args.source_mode)),
        "source_x_key": np.asarray(str(simulation_summary["source_x_key"])),
        "obs_path": np.asarray(str(obs_path)),
        "binning_json": np.asarray(json.dumps(binning_summary, sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(summary, sort_keys=True)),
        "gaussian_beam_applied": np.asarray(bool(args.apply_gaussian_beam)),
        "gaussian_beam_fwhm_arcmin": np.asarray(beam_fwhm),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_payload)
    summary_path = output_path.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True)

    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")
    print(f"theta shape: {prepared['theta'].shape}")
    print(f"x shape: {prepared['x'].shape}")
    print(f"obs shape: {prepared['obs'].shape}")
    print(f"obs kind: {obs_summary['obs_input_kind_used']}")
    print(f"gaussian beam applied in preparation: {args.apply_gaussian_beam}, fwhm={beam_fwhm}")
    print("PBS runner beam settings: SBI_GAUSSIAN_BEAM_MODE=off SBI_GAUSSIAN_BEAM_FWHM_ARCMIN=0.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
