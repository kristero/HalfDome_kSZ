"""Stability of the halo-scatter correlation structure across mass bins.

All metrics use *residualised* ln F on the shells resolved in both bins: per bin,
subtract the mean and a linear trend in log10 M (so the within-bin mass gradient
cannot create correlation), then standardise every shell to unit variance (so
the null hypothesis concerns correlation only, not the amplitude of scatter).

Permutation null (exact under exchangeability): pool the two standardised
residual sets, reassign labels at the original N_a, N_b, recompute the metrics;
p = (1 + #{T_perm >= T_obs}) / (1 + n_perm).  Metrics:
  frobenius_rms : RMS over off-diagonal elements of R_a - R_b
  delta_a, delta_lnL : |difference| of the 2-parameter model
                   rho_ij = a + (1-a) exp(-|ln x_i - ln x_j|/L)
  max_angle_deg  : largest principal angle between the top-2 eigen-subspaces
A min-p combination over the four metrics is calibrated on the same permutations.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import subspace_angles

from . import covariance as cov

METRICS = ("frobenius_rms", "delta_a", "delta_lnL", "max_angle_deg")


def residualise(logv: np.ndarray, logmass=None, standardize: bool = True) -> np.ndarray:
    X = cov.residualise_mass(logv, logmass) if logmass is not None else logv - logv.mean(axis=0)
    if standardize:
        with np.errstate(divide="ignore", invalid="ignore"):   # degenerate resamples -> NaN (skipped)
            X = X / X.std(axis=0, ddof=1)
    return X


def correlation(Z: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.corrcoef(Z, rowvar=False))


def frobenius_rms(Ra: np.ndarray, Rb: np.ndarray) -> float:
    p = Ra.shape[0]
    off = ~np.eye(p, dtype=bool)
    return float(np.sqrt(np.mean((Ra[off] - Rb[off])**2)))


def top_subspace(R: np.ndarray, k: int = 2) -> np.ndarray:
    values, vectors = np.linalg.eigh(R)
    return vectors[:, np.argsort(values)[::-1][:k]]


def principal_angles_deg(Ra: np.ndarray, Rb: np.ndarray, k: int = 2) -> np.ndarray:
    angles = subspace_angles(top_subspace(Ra, k), top_subspace(Rb, k))
    return np.sort(np.degrees(angles))[::-1]


def model_parameters(R: np.ndarray, lnx: np.ndarray) -> tuple[float, float]:
    fit = cov.fit_corr_model(R, lnx)
    return fit["a"], fit["L"]


def pair_statistics(Za: np.ndarray, Zb: np.ndarray, lnx: np.ndarray, k: int = 2) -> dict:
    Ra, Rb = correlation(Za), correlation(Zb)
    aa, La = model_parameters(Ra, lnx)
    ab, Lb = model_parameters(Rb, lnx)
    angles = principal_angles_deg(Ra, Rb, k)
    return {"frobenius_rms": frobenius_rms(Ra, Rb), "delta_a": abs(aa - ab),
            "delta_lnL": abs(np.log(La) - np.log(Lb)), "max_angle_deg": float(angles[0]),
            "angles_deg": angles, "a": (aa, ab), "L": (La, Lb)}


def permutation_test(Za: np.ndarray, Zb: np.ndarray, lnx: np.ndarray, n_perm: int = 500,
                     seed: int = 0, k: int = 2) -> dict:
    """Permutation null for 'both bins share one correlation matrix'."""
    rng = np.random.default_rng(seed)
    observed = pair_statistics(Za, Zb, lnx, k)
    pooled = np.vstack([Za, Zb])
    na = len(Za)
    null = np.empty((n_perm, len(METRICS)))
    for i in range(n_perm):
        order = rng.permutation(len(pooled))
        s = pair_statistics(pooled[order[:na]], pooled[order[na:]], lnx, k)
        null[i] = [s[m] for m in METRICS]
    obs = np.array([observed[m] for m in METRICS])
    p = (1 + np.sum(null >= obs[None, :], axis=0)) / (1 + n_perm)
    # min-p combination calibrated on the same permutations
    ranks = np.empty_like(null)
    for j in range(len(METRICS)):
        column = null[:, j]
        ranks[:, j] = (1 + np.sum(column[None, :] >= column[:, None], axis=1)) / (1 + n_perm)
    min_null = ranks.min(axis=1)
    combined = (1 + np.sum(min_null <= p.min())) / (1 + n_perm)
    return {"observed": observed, "p_values": dict(zip(METRICS, p.tolist())),
            "p_combined_minp": float(combined), "null": null, "n_perm": n_perm,
            "n_a": int(na), "n_b": int(len(Zb)), "p_shells": int(Za.shape[1])}


def decorrelation_radius(R: np.ndarray, lnx: np.ndarray, reference: int = 0) -> float:
    """Descriptive: first radius where corr(shell, reference shell) crosses zero (ln-x interpolation)."""
    r = np.asarray(R)[reference]
    order = np.arange(reference + 1, len(r))
    for i in order:
        if r[i] < 0:
            j = i - 1
            t = r[j] / (r[j] - r[i])
            return float(np.exp(lnx[j] + t * (lnx[i] - lnx[j])))
    return float("nan")


def bootstrap_model(logv: np.ndarray, logmass, lnx: np.ndarray, draws: int = 200, seed: int = 0) -> dict:
    """Bootstrap (a, L) of one bin; residualisation is redone inside every draw."""
    rng = np.random.default_rng(seed)
    n = len(logv)
    a = np.full(draws, np.nan)
    L = np.full(draws, np.nan)
    for i in range(draws):
        rows = rng.integers(0, n, n)
        Z = residualise(logv[rows], None if logmass is None else np.asarray(logmass)[rows])
        if not np.all(np.isfinite(Z)):
            continue
        a[i], L[i] = model_parameters(correlation(Z), lnx)
    return {"a": a, "L": L}
