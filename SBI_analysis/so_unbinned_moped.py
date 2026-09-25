"""Memory-bounded MOPED fitting on individual signed D_ell coordinates."""
import json
from pathlib import Path

import numpy as np

from prepare_validate_battaglia12_sbi_observation import make_cl_to_dl_matrix
from so_sbi_compression import FIDUCIAL, PARAM_NAMES, asinh_coordinates, fit_moped


def require(condition, message):
    if not condition:
        raise ValueError(message)


def open_raw(path, metadata_path, prepared, ell_min=80, ell_max=7979):
    """Require actual metadata row identities, never assume Sobol CSV ordering."""
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_path, allow_pickle=False) as metadata:
        theta = np.asarray(metadata["theta"], dtype=np.float32)
        ell = np.asarray(metadata["ell"], dtype=np.int64)
        np.testing.assert_array_equal(metadata["theta_columns"], PARAM_NAMES)
        np.testing.assert_array_equal(theta, prepared["theta"])
        np.testing.assert_array_equal(metadata["sobol_global_row"], prepared["sobol_global_row"])
        require(str(metadata["product"]) == str(prepared["product"]), "Raw product differs")
    np.testing.assert_array_equal(ell, np.arange(ell_min, ell_max + 1))
    require(raw.shape == (len(theta), len(ell)) and raw.dtype.kind == "f", "Invalid raw spectral array")
    return raw, ell


def validate_rebin(raw, ell, prepared, block_rows=2048):
    """Check every raw row against the 40-bin values used by existing models."""
    metadata = json.loads(str(prepared["metadata_json"]))
    matrix = make_cl_to_dl_matrix(ell, prepared["bin_ell_min"],
                                  prepared["bin_ell_max"], metadata["bin_weighting"])
    worst = 0.0
    for start in range(0, len(raw), block_rows):
        stop = min(start + block_rows, len(raw))
        values = np.asarray(raw[start:stop], dtype=np.float32)
        require(np.isfinite(values).all(), "Nonfinite raw spectrum")
        actual = values @ matrix
        expected = prepared["x"][start:stop]
        # BLAS summation may differ across machines. Scale the absolute tolerance
        # by each spectrum, rather than letting cancellation near zero reject it.
        tolerance = (1e-6 * np.max(np.abs(expected), axis=1, keepdims=True)
                     + 2e-5 * np.abs(expected) + 1e-27)
        error = float(np.max(np.abs(actual - expected) / tolerance))
        require(error <= 1, f"Raw/40-bin mismatch in rows {start}:{stop}; tolerance ratio={error:g}")
        worst = max(worst, error)
    return dict(rows_checked=len(raw), maximum_tolerance_ratio=worst)


def dell_values(raw, ell, rows, columns=slice(None)):
    selected_ell = np.asarray(ell[columns], dtype=np.float64)
    factor = selected_ell * (selected_ell + 1) / (2 * np.pi)
    values = np.asarray(raw[rows, columns], dtype=np.float64) * factor
    require(np.isfinite(values).all(), "Nonfinite D_ell values")
    return values


def fit_scaling(raw, ell, fit_indices, feature_block=64):
    """Exact optimization-set medians/moments, one feature block at a time.

    Uses the same lower median and signed-asinh convention as the 40-bin run.
    Validation and test rows cannot affect the transform.
    """
    require(feature_block > 0 and len(fit_indices) >= 2, "Invalid scaling block or fit rows")
    result = {key: np.empty(len(ell), dtype=np.float64) for key in ("scale", "mean", "std")}
    k = (len(fit_indices) - 1) // 2
    for start in range(0, len(ell), feature_block):
        stop = min(start + feature_block, len(ell))
        columns = slice(start, stop)
        values = dell_values(raw, ell, fit_indices, columns)
        absolute = np.abs(values)
        absolute.partition(k, axis=0)
        scale = np.maximum(absolute[k].copy(), 1e-30)
        del absolute
        values = np.arcsinh(values / scale)
        result["scale"][columns] = scale
        result["mean"][columns] = values.mean(axis=0)
        result["std"][columns] = np.maximum(values.std(axis=0, ddof=1), 1e-8)
        print(f"Scaling multipoles {stop}/{len(ell)}", flush=True)
    return result


def fit_unbinned(raw_noisy, raw_clean, ell, theta, fit_indices, low, high,
                 local_n=20000, shrinkage=.05, rcond=1e-6, feature_block=64):
    base = fit_scaling(raw_noisy, ell, fit_indices, feature_block)
    radius = np.linalg.norm((theta[fit_indices] - FIDUCIAL) / (high - low), axis=1)
    nearest = fit_indices[np.argsort(radius, kind="stable")[:min(local_n, len(fit_indices))]]
    print(f"Fitting local MOPED from {len(nearest)} optimization rows and {len(ell)} multipoles", flush=True)
    noisy = asinh_coordinates(dell_values(raw_noisy, ell, nearest), base)
    clean = asinh_coordinates(dell_values(raw_clean, ell, nearest), base)
    diagnostics = fit_moped(noisy, clean, theta[nearest], high - low,
                            len(nearest), shrinkage, rcond)
    diagnostics["local_dataset_indices"] = nearest[diagnostics["local_indices"]]
    error = np.linalg.norm(diagnostics["fisher"] - diagnostics["compressed_fisher"])
    error /= max(np.linalg.norm(diagnostics["fisher"]), 1e-30)
    require(error < 1e-8, f"Local Fisher identity failed: {error:g}")
    diagnostics["fisher_relative_error"] = np.asarray(error)
    transform = dict(**base, matrix=diagnostics["matrix"],
                     projection_center=diagnostics["center"], ell=ell)
    return transform, diagnostics


def write_projected(raw, ell, transform, fit_indices, destination, block_rows=2048):
    """Project all rows and fit final standardization on optimization rows only."""
    destination = Path(destination)
    temporary = destination.with_suffix(".unscaled.npy")
    shape = (len(raw), transform["matrix"].shape[1])
    projected = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float64, shape=shape)
    for start in range(0, len(raw), block_rows):
        stop = min(start + block_rows, len(raw))
        values = asinh_coordinates(dell_values(raw, ell, slice(start, stop)), transform)
        projected[start:stop] = (values - transform["projection_center"]) @ transform["matrix"]
        if start == 0 or stop == len(raw) or (start // block_rows) % 32 == 0:
            print(f"Projected rows {stop}/{len(raw)}", flush=True)
    projected.flush()
    optimization = np.asarray(projected[fit_indices])
    transform["output_mean"] = optimization.mean(axis=0)
    transform["output_std"] = np.maximum(optimization.std(axis=0, ddof=1), 1e-8)
    del optimization
    target = destination.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=shape)
    for start in range(0, len(raw), block_rows):
        stop = min(start + block_rows, len(raw))
        output[start:stop] = ((projected[start:stop] - transform["output_mean"])
                             / transform["output_std"])
    output.flush()
    del output, projected
    target.replace(destination)
    temporary.unlink()
    return transform


def project_observation(spectrum, ell, transform):
    require(spectrum.shape == (len(ell),), "Observation multipoles differ")
    values = np.asarray(spectrum, dtype=np.float64) * ell * (ell + 1) / (2 * np.pi)
    normalized = asinh_coordinates(values, transform)
    compressed = (normalized - transform["projection_center"]) @ transform["matrix"]
    return np.asarray((compressed - transform["output_mean"]) / transform["output_std"], dtype=np.float32)
