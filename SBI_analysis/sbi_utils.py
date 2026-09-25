#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PARAM_NAMES = [
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


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def selected_ell_mask(ell: np.ndarray, ell_min: int, ell_max: int | None) -> np.ndarray:
    mask = np.asarray(ell, dtype=float) >= float(ell_min)
    if ell_max is not None:
        mask &= np.asarray(ell, dtype=float) <= float(ell_max)
    if not np.any(mask):
        raise ValueError(f"Empty ell selection for ell_min={ell_min}, ell_max={ell_max}")
    return mask


def cl_to_log10_dl(cl: np.ndarray, ell: np.ndarray, floor: float = 1.0e-40) -> np.ndarray:
    cl = np.asarray(cl, dtype=float)
    ell = np.asarray(ell, dtype=float)
    factor = ell * (ell + 1.0) / (2.0 * math.pi)
    dl = cl * factor
    return np.log10(np.maximum(dl, floor))


def dl_to_log10_dl(dl: np.ndarray, floor: float = 1.0e-40) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(dl, dtype=float), floor))


def validate_cl_scale(name: str, values: np.ndarray) -> None:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values")
    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs > 1.0e-6:
        raise ValueError(
            f"{name} has max |value|={max_abs:.6g}, which is far too large for raw tSZ C_l. "
            "This usually means the dataset was built from the FITS Index/ell column. "
            "Rebuild the emulator combined dataset with the fixed FITS reader."
        )


def load_combined_dataset(path: str | Path, ell_min: int, ell_max: int | None) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing combined dataset {path}. Run the emulator builder with SAVE_COMBINED_DATASET=true first."
        )

    data = np.load(path, allow_pickle=True)
    required = ["x", "y_combined", "y100", "y102", "ell", "x_columns"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")

    ell_all = np.asarray(data["ell"], dtype=float).reshape(-1)
    mask = selected_ell_mask(ell_all, ell_min, ell_max)
    theta = np.asarray(data["x"], dtype=float)
    y_combined = np.asarray(data["y_combined"], dtype=float)[:, mask]
    y100 = np.asarray(data["y100"], dtype=float)[:, mask]
    y102 = np.asarray(data["y102"], dtype=float)[:, mask]
    ell = ell_all[mask]
    x_columns = [str(col) for col in np.asarray(data["x_columns"]).tolist()]

    validate_cl_scale("y_combined", y_combined)
    validate_cl_scale("y100", y100)
    validate_cl_scale("y102", y102)

    return {
        "theta": theta,
        "y_combined_cl": y_combined,
        "y100_cl": y100,
        "y102_cl": y102,
        "ell": ell,
        "x_columns": x_columns,
        "source_path": str(path),
    }


def dataset_to_log10_dl(dataset: dict[str, Any], floor: float = 1.0e-40) -> dict[str, np.ndarray]:
    ell = dataset["ell"]
    return {
        "combined": cl_to_log10_dl(dataset["y_combined_cl"], ell, floor),
        "y100": cl_to_log10_dl(dataset["y100_cl"], ell, floor),
        "y102": cl_to_log10_dl(dataset["y102_cl"], ell, floor),
    }


def estimate_lightcone_sigma(
    x_y100: np.ndarray,
    x_y102: np.ndarray,
    target: str,
    sigma_floor: float,
    scale: float,
) -> np.ndarray:
    diff = np.asarray(x_y100, dtype=float) - np.asarray(x_y102, dtype=float)
    if diff.shape[0] < 2:
        sigma_single = np.abs(diff[0]) / math.sqrt(2.0)
    else:
        sigma_single = np.nanstd(diff, axis=0, ddof=1) / math.sqrt(2.0)

    if target == "single_lightcone":
        sigma = sigma_single
    elif target == "two_lightcone_mean":
        sigma = sigma_single / math.sqrt(2.0)
    else:
        raise ValueError("noise.target must be single_lightcone or two_lightcone_mean")

    sigma = np.asarray(sigma, dtype=float) * float(scale)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    return np.maximum(sigma, float(sigma_floor))


def prior_bounds(config: dict[str, Any], x_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    prior = config["prior"]
    missing = [name for name in x_columns if name not in prior]
    if missing:
        raise KeyError(f"Config prior is missing emulator parameters: {missing}")
    low = np.asarray([prior[name][0] for name in x_columns], dtype=np.float32)
    high = np.asarray([prior[name][1] for name in x_columns], dtype=np.float32)
    if not np.all(high > low):
        raise ValueError("Every prior upper bound must be greater than the lower bound")
    return low, high


def sample_uniform_prior(
    rng: np.random.Generator,
    n: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    return rng.uniform(low, high, size=(int(n), low.size)).astype(np.float32)


def patch_sklearn_pickle_compat(artifact: dict[str, Any]) -> None:
    pca = artifact.get("pca")
    if pca is not None:
        defaults = {
            "power_iteration_normalizer": "auto",
            "n_oversamples": 10,
        }
        for name, value in defaults.items():
            if not hasattr(pca, name):
                setattr(pca, name, value)


def load_emulator(path: str | Path) -> dict[str, Any]:
    import joblib

    artifact = joblib.load(Path(path).expanduser())
    patch_sklearn_pickle_compat(artifact)
    return artifact


def inverse_emulator_targets(y_t: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log10":
        return np.power(10.0, y_t)
    if target_transform == "none":
        return y_t
    raise ValueError(f"Unknown emulator target_transform={target_transform!r}")


def predict_emulator_cl(
    artifact: dict[str, Any],
    theta: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    out = []
    for start in range(0, theta.shape[0], int(batch_size)):
        batch = theta[start : start + int(batch_size)]
        x_s = artifact["x_scaler"].transform(batch)
        z = artifact["regressor"].predict(x_s)
        y_t = artifact["pca"].inverse_transform(z)
        out.append(inverse_emulator_targets(y_t, artifact.get("target_transform", "log10")))
    cl = np.vstack(out)
    validate_cl_scale("emulator prediction", cl)
    return cl


def load_observed_vector(
    path: str | Path,
    ell_target: np.ndarray,
    input_kind: str = "auto",
    floor: float = 1.0e-40,
) -> np.ndarray:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Observed spectrum not found: {path}")
    spectrum, ell = read_spectrum_file(path)
    if ell is None:
        ell = np.arange(spectrum.size, dtype=float)
    ell = np.asarray(ell, dtype=float).reshape(-1)
    spectrum = np.asarray(spectrum, dtype=float).reshape(-1)
    if ell.size != spectrum.size:
        raise ValueError(f"Observed ell length {ell.size} does not match spectrum length {spectrum.size}")

    valid = np.isfinite(ell) & np.isfinite(spectrum)
    if not np.any(valid):
        raise ValueError(f"Observed spectrum {path} has no finite values")
    interp = np.interp(ell_target, ell[valid], spectrum[valid])
    kind = infer_spectrum_kind(path, interp, ell_target) if input_kind == "auto" else input_kind
    if kind == "cl":
        return cl_to_log10_dl(interp, ell_target, floor)
    if kind == "dl":
        return dl_to_log10_dl(interp, floor)
    if kind == "log10_dl":
        return interp
    raise ValueError("observed_input_kind must be auto, cl, dl, or log10_dl")


def infer_spectrum_kind(path: Path, spectrum: np.ndarray, ell: np.ndarray) -> str:
    name = path.name.lower()
    if "log10" in name and ("dl" in name or "d_l" in name):
        return "log10_dl"
    if "dl" in name or "d_l" in name:
        return "dl"
    if "cl" in name or "c_l" in name or path.suffix.lower() in (".fits", ".fit", ".fts"):
        return "cl"
    good = np.isfinite(spectrum) & (ell >= 100) & (ell <= max(ell.max(), 100))
    median = float(np.nanmedian(np.abs(spectrum[good]))) if np.any(good) else float(np.nanmedian(np.abs(spectrum)))
    return "dl" if median > 1.0e-15 else "cl"


def read_spectrum_file(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float).squeeze(), None
    if suffix == ".npz":
        data = np.load(path)
        ell = np.asarray(data["ell"], dtype=float).squeeze() if "ell" in data else None
        for key in ("log10_dl", "dl", "d_l", "cl", "c_l", "profile", "data"):
            if key in data:
                return np.asarray(data[key], dtype=float).squeeze(), ell
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=float).squeeze(), ell
        raise ValueError(f"Could not choose observed spectrum inside {path}; keys={data.files}")
    if suffix in (".csv", ".txt", ".dat"):
        delimiter = "," if suffix == ".csv" else None
        raw = np.loadtxt(path, delimiter=delimiter)
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            return raw[:, 1], raw[:, 0]
        return raw.squeeze(), None
    if suffix in (".fits", ".fit", ".fts"):
        return read_fits_spectrum(path)
    raise ValueError(f"Unsupported observed spectrum extension: {path}")


def read_fits_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        from astropy.io import fits
    except ModuleNotFoundError:
        return read_fits_spectrum_fitsio(path)

    profile_candidates = []
    ell_candidates = []
    preferred = {"c_l", "cl", "dl", "d_l", "power", "profile"}
    ell_names = {"ell", "l", "multipole"}
    skip = {"index", "row", "pixel"}
    with fits.open(path, memmap=False) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            data = hdu.data
            if data is None:
                continue
            if getattr(data, "dtype", None) is not None and data.dtype.fields:
                for field in data.dtype.names or ():
                    arr = np.asarray(data[field]).squeeze()
                    if arr.size < 2 or not np.issubdtype(arr.dtype, np.number):
                        continue
                    key = normalize_name(field)
                    arr = np.asarray(arr, dtype=float).reshape(-1)
                    if key in ell_names:
                        ell_candidates.append(arr)
                    elif key not in skip:
                        priority = 0 if key in preferred or "cl" in key or "power" in key else 1
                        profile_candidates.append((priority, hdu_index, arr))
            else:
                arr = np.asarray(data).squeeze()
                if arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                    profile_candidates.append((2, hdu_index, np.asarray(arr, dtype=float).reshape(-1)))
    if not profile_candidates:
        raise ValueError(f"No spectrum-like numeric data found in {path}")
    profile_candidates.sort(key=lambda item: (item[0], -item[2].size))
    profile = profile_candidates[0][2]
    ell = ell_candidates[0] if ell_candidates and ell_candidates[0].size == profile.size else None
    return profile, ell


def read_fits_spectrum_fitsio(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    import fitsio

    profile_candidates = []
    ell_candidates = []
    ell_names = {"ell", "l", "multipole"}
    skip = {"index", "row", "pixel"}
    with fitsio.FITS(str(path)) as hdul:
        for hdu_index, hdu in enumerate(hdul):
            data = hdu.read()
            if data is None:
                continue
            if getattr(data, "dtype", None) is not None and data.dtype.fields:
                for field in data.dtype.names or ():
                    arr = np.asarray(data[field]).squeeze()
                    if arr.size < 2 or not np.issubdtype(arr.dtype, np.number):
                        continue
                    key = normalize_name(field)
                    arr = np.asarray(arr, dtype=float).reshape(-1)
                    if key in ell_names:
                        ell_candidates.append(arr)
                    elif key not in skip:
                        priority = 0 if "cl" in key or "power" in key else 1
                        profile_candidates.append((priority, hdu_index, arr))
            else:
                arr = np.asarray(data).squeeze()
                if arr.size >= 2 and np.issubdtype(arr.dtype, np.number):
                    profile_candidates.append((2, hdu_index, np.asarray(arr, dtype=float).reshape(-1)))
    if not profile_candidates:
        raise ValueError(f"No spectrum-like numeric data found in {path}")
    profile_candidates.sort(key=lambda item: (item[0], -item[2].size))
    profile = profile_candidates[0][2]
    ell = ell_candidates[0] if ell_candidates and ell_candidates[0].size == profile.size else None
    return profile, ell


def normalize_name(name: str) -> str:
    import re

    normalized = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
