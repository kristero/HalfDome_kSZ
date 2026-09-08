"""Train-only PCA and local MOPED in the same signed-asinh SO coordinates.

This module intentionally needs only NumPy, not sbi or scikit-learn.
MOPED preserves the fitted, fixed-covariance local mean Fisher matrix,
not necessarily the information of the full non-Gaussian simulator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PARAM_NAMES = (
    "P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
    "alpha_z_P0", "alpha_z_xc", "alpha_z_beta",
)
FIDUCIAL = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
METHODS = ("bins40", "pca", "moped")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    """Atomic checkpoint: an interrupted write never looks like a finished row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def array_digest(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        keys = ("theta", "x", "prior_low", "prior_high", "param_names",
                "sobol_global_row", "ell_binned", "bin_ell_min", "bin_ell_max",
                "product", "metadata_json")
        result = {key: data[key] for key in keys}
    names = tuple(str(item) for item in result["param_names"])
    if names != PARAM_NAMES:
        raise ValueError(f"Unexpected parameter order: {names}; expected {PARAM_NAMES}")
    x, theta = result["x"], result["theta"]
    if x.ndim != 2 or x.shape[1] != 40 or theta.shape != (len(x), 9):
        raise ValueError(f"Expected (N,40) D_ell and (N,9) theta, got {x.shape}, {theta.shape}")
    if not np.isfinite(x).all() or not np.isfinite(theta).all():
        raise ValueError("Nonfinite spectra or parameters")
    low, high = result["prior_low"], result["prior_high"]
    if low.shape != (9,) or high.shape != (9,) or not np.all(high > low):
        raise ValueError("Invalid saved prior bounds")
    if not np.all((FIDUCIAL > low) & (FIDUCIAL < high)):
        raise ValueError("Battaglia12 must be inside the saved prior for this local MOPED fit")
    ids = result["sobol_global_row"]
    if (ids.shape != (len(x),) or ids.dtype.kind not in "iu"
            or np.any(ids < 1) or len(np.unique(ids)) != len(ids)):
        raise ValueError("Missing, duplicate, or invalid Sobol row identities")
    metadata = json.loads(str(result["metadata_json"].item()))
    if metadata.get("statistic") != "weighted mean of linear D_ell; x equals binned D_ell":
        raise ValueError("Input is not the verified signed, binned linear D_ell product")
    return result


def check_pair(noisy, clean):
    if str(noisy["product"].item()) != "masked_baseline_noise_cross_deproj0":
        raise ValueError("This experiment requires masked baseline deproj0 noise")
    if str(clean["product"].item()) != "masked_no_noise":
        raise ValueError("MOPED requires the matched masked_no_noise product, not full-sky")
    for key in ("theta", "param_names", "sobol_global_row", "ell_binned",
                "bin_ell_min", "bin_ell_max", "prior_low", "prior_high"):
        if not np.array_equal(noisy[key], clean[key]):
            raise ValueError(f"Noisy/clean alignment or observable contract differs: {key}")
    if noisy["x"].shape != clean["x"].shape:
        raise ValueError("Noisy and clean spectra have different shapes")
    nmeta, cmeta = (json.loads(str(d["metadata_json"].item())) for d in (noisy, clean))
    if nmeta.get("bin_weighting") != cmeta.get("bin_weighting"):
        raise ValueError("Noisy/clean bin weighting differs")


def split_rows(theta, low, high, holdout, validation_fraction, seed):
    if not 0 < holdout < len(theta) or not 0 < validation_fraction < .5:
        raise ValueError("Require 0 < holdout < N and 0 < validation_fraction < 0.5")
    supported = np.all((theta >= low) & (theta <= high), axis=1)
    original_test = np.arange(len(theta) - holdout, len(theta))
    test = original_test[supported[original_test]]
    pool = np.flatnonzero(supported[:len(theta) - holdout])
    pool = np.random.default_rng(seed).permutation(pool)
    n_fit = int((1 - validation_fraction) * len(pool))
    if n_fit < 100 or len(test) < 3 or len(pool) - n_fit < 2:
        raise ValueError("Too few in-prior optimization, validation, or held-out rows")
    return dict(pool=pool, fit=pool[:n_fit], validation=pool[n_fit:], test=test,
                excluded=np.flatnonzero(~supported), original_test=original_test)


def fit_asinh(x):
    # NumPy's usual median averages two entries; Torch uses the lower entry.
    absolute = np.abs(np.asarray(x, dtype=np.float64))
    k = (len(x) - 1) // 2
    scale = np.maximum(np.partition(absolute, k, axis=0)[k], 1e-30)
    values = np.arcsinh(x / scale)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0, ddof=1), 1e-8)
    return dict(scale=scale, mean=mean, std=std)


def asinh_coordinates(x, transform):
    return (np.arcsinh(np.asarray(x, dtype=np.float64) / transform["scale"])
            - transform["mean"]) / transform["std"]


def fit_pca(values, components):
    if not 1 <= components <= values.shape[1]:
        raise ValueError("PCA components must lie in 1..40")
    center = values.mean(axis=0)
    centered = values - center
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered / (len(values) - 1))
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    return dict(matrix=vectors[:, :components], center=center,
                eigenvalues=eigenvalues,
                explained_variance_fraction=eigenvalues / eigenvalues.sum())


def quadratic_design(delta):
    columns = [np.ones(len(delta))] + [delta[:, j] for j in range(delta.shape[1])]
    columns += [delta[:, i] * delta[:, j]
                for i in range(delta.shape[1]) for j in range(i, delta.shape[1])]
    return np.column_stack(columns)


def shrunk_covariance(residuals, shrinkage):
    if not 0 < shrinkage <= 1:
        raise ValueError("Covariance shrinkage must be in (0,1]")
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        raise ValueError("Paired residuals have zero or invalid bin variance")
    return (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))


def moped_basis(derivatives, covariance, rcond):
    """An SVD rotation of MOPED; drop unresolved directions, not parameter columns."""
    if not 0 < rcond < 1:
        raise ValueError("MOPED rcond must be in (0,1)")
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, derivatives)
    left, singular, _ = np.linalg.svd(whitened, full_matrices=False)
    keep = singular > singular[0] * rcond
    if not np.any(keep):
        raise ValueError("No identified local mean-sensitivity directions")
    weights = np.linalg.solve(chol.T, left[:, keep])
    fisher = whitened.T @ whitened
    compressed = weights.T @ derivatives
    return dict(matrix=weights, singular_values=singular, fisher=fisher,
                compressed_fisher=compressed.T @ compressed,
                compressed_covariance=weights.T @ covariance @ weights)


def fit_moped(noisy, clean, theta, width, local_n, shrinkage, rcond):
    """Fit near Battaglia12 using optimization rows only, in asinh coordinates.

    The paired residual's conditional mean is fitted too: asinh(noisy)-
    asinh(clean) is not generally zero-mean even for unbiased raw noise.
    The residual covariance is local and approximate, not a covariance over
    the prior's changing clean signal. No 64-profile ensemble is used.
    """
    delta = (theta - FIDUCIAL) / width
    radius = np.linalg.norm(delta, axis=1)
    local_n = min(local_n, len(theta))
    if local_n < 600:
        raise ValueError("Use at least 600 optimization rows for the 55-term local fit")
    nearest = np.argsort(radius, kind="stable")[:local_n]
    design = quadratic_design(delta[nearest])
    clean_coeff = np.linalg.lstsq(design, clean[nearest], rcond=None)[0]
    residual = noisy[nearest] - clean[nearest]
    bias_coeff, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Local quadratic design is rank deficient")
    mean_coeff = clean_coeff + bias_coeff
    # Derivatives are with respect to prior-normalized theta for conditioning.
    derivatives = mean_coeff[1:10].T
    residual_centered = residual - design @ bias_coeff
    # Account for the fitted mean's regression degrees of freedom.
    residual_centered *= np.sqrt((local_n - 1) / (local_n - rank))
    covariance = shrunk_covariance(residual_centered, shrinkage)
    result = moped_basis(derivatives, covariance, rcond)
    half = local_n // 2
    half_coeff = np.linalg.lstsq(design[:half], noisy[nearest[:half]], rcond=None)[0]
    derivative_change = np.linalg.norm(half_coeff[1:10].T - derivatives, axis=0)
    derivative_change /= np.maximum(np.linalg.norm(derivatives, axis=0), 1e-30)
    fit_error = np.sqrt(np.mean((clean[nearest] - design @ clean_coeff)**2, axis=0))
    result.update(center=mean_coeff[0], derivatives=derivatives, covariance=covariance,
                  local_indices=nearest, local_radius=radius[nearest],
                  derivative_relative_change_half=derivative_change,
                  clean_fit_rmse_over_noise_std=fit_error / np.sqrt(np.diag(covariance)),
                  regression_condition=np.asarray(np.linalg.cond(design)))
    return result


def project(x, transform):
    base = asinh_coordinates(x, transform)
    values = (base - transform["projection_center"]) @ transform["matrix"]
    return np.asarray((values - transform["output_mean"]) / transform["output_std"],
                      dtype=np.float32)


def finish_transform(base_transform, values, matrix, center):
    output = (values - center) @ matrix
    return dict(**base_transform, matrix=matrix, projection_center=center,
                output_mean=output.mean(axis=0),
                output_std=np.maximum(output.std(axis=0, ddof=1), 1e-8))


def metrics_from_samples(samples, truth, low, high):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != len(truth) or len(samples) < 2:
        raise ValueError("Invalid posterior sample shape")
    if not np.isfinite(samples).all() or np.any((samples < low) | (samples > high)):
        raise ValueError("Posterior samples are nonfinite or outside the prior")
    mean, std = samples.mean(axis=0), samples.std(axis=0, ddof=1)
    if np.any(std <= 0):
        raise ValueError("Degenerate posterior standard deviation")
    error = mean - truth
    q = np.quantile(samples, [.025, .16, .84, .975], axis=0)
    return dict(mean=mean, std=std, truth=np.asarray(truth), error=error,
                normalized_error_prior=error / (high - low), pull=error / std,
                std_over_prior=std / (high - low),
                coverage68=(truth >= q[1]) & (truth <= q[2]),
                coverage95=(truth >= q[0]) & (truth <= q[3]))


def pearson_columns(truth, mean):
    a, b = truth - truth.mean(axis=0), mean - mean.mean(axis=0)
    denominator = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
    return np.divide((a*b).sum(axis=0), denominator,
                     out=np.full(truth.shape[1], np.nan), where=denominator > 0)
