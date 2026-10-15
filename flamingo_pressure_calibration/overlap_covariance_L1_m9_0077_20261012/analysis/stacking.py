"""Stacks and seeded, paired halo bootstraps.

Pairing convention: one ``rows`` matrix [draw, n] per halo sample is generated
once and reused for every variant (field, membership selection, AGN estimator,
cut-off, objective), so differences between variants are paired draw by draw.
``bootstrap_rows(n, draws, seed)`` equals radii234 fit_study.py:100-101
(``default_rng(20261008 + bin).integers(0, n, size=(draws, n))``); with a larger
``draws`` the first 128 rows are unchanged (sequential generator stream).
"""
from __future__ import annotations

import numpy as np

LEGACY_SEED = 20261008      # radii234 fit_study: seed = LEGACY_SEED + bin
STUDY_SEED = 20261012       # this study


def mean_stack(values: np.ndarray, weights=None) -> np.ndarray:
    if weights is None:
        return values.mean(axis=0)
    w = np.asarray(weights, float)
    return (w / w.sum()) @ values


def std_stack(values: np.ndarray, ddof: int = 1) -> np.ndarray:
    return values.std(axis=0, ddof=ddof)


def median_stack(values: np.ndarray) -> np.ndarray:
    return np.median(values, axis=0)


def weighted_stack(values, weights):
    """sensitivity_projects analysis_common.weighted_stack (61-68), verbatim maths.

    Returns (mean, reliability-weighted SD, effective number of haloes).
    """
    weights = np.asarray(weights, float).copy()
    weights /= weights.sum()
    mean = weights @ values
    variance = weights @ ((values - mean)**2)
    correction = 1 - np.square(weights).sum()
    std = np.sqrt(variance / correction) if correction > 0 else np.full_like(mean, np.nan)
    return mean, std, 1 / np.square(weights).sum()


def log_mean_stack(values: np.ndarray, weights=None) -> np.ndarray:
    """Geometric mean exp(<ln F>) per shell; shells with any F <= 0 give NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.where(values > 0, values, np.nan))
    if weights is None:
        return np.exp(np.mean(logs, axis=0))
    w = np.asarray(weights, float)
    return np.exp((w / w.sum()) @ logs)


def inclusion_weights(strata, population, sample=None) -> np.ndarray:
    """w_h = population[stratum] / sample[stratum] (sample counted from ``strata`` if None)."""
    strata = np.asarray(strata)
    population = np.asarray(population, float)
    if sample is None:
        sample = np.bincount(strata, minlength=len(population)).astype(float)
    return (population / np.asarray(sample, float))[strata]


def bootstrap_rows(n: int, draws: int, seed: int) -> np.ndarray:
    """[draws, n] resampling positions 0..n-1 (radii234-compatible generator call)."""
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def stratified_bootstrap_rows(strata, draws: int, seed: int) -> np.ndarray:
    """Resample within each stratum (sizes fixed); returns positions [draws, n]."""
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, len(strata)), dtype=np.int64)
    for s in np.unique(strata):
        where = np.flatnonzero(strata == s)
        out[:, where] = where[rng.integers(0, len(where), size=(draws, len(where)))]
    return out


def counts_matrix(rows: np.ndarray, n: int) -> np.ndarray:
    """Multiplicity of each halo in each draw: [draws, n] float64."""
    rows = np.atleast_2d(rows)
    draws = rows.shape[0]
    flat = (np.arange(draws)[:, None] * n + rows).ravel()
    return np.bincount(flat, minlength=draws * n).reshape(draws, n).astype(np.float64)


def bootstrap_means(values: np.ndarray, rows: np.ndarray, weights=None, chunk: int = 256) -> np.ndarray:
    """Bootstrap stack means [draws, shell] via multiplicity matrices (memory-light)."""
    n = values.shape[0]
    out = np.empty((rows.shape[0], values.shape[1]))
    w = None if weights is None else np.asarray(weights, float)
    for start in range(0, rows.shape[0], chunk):
        counts = counts_matrix(rows[start:start + chunk], n)
        if w is None:
            out[start:start + chunk] = counts @ values / n
        else:
            cw = counts * w[None, :]
            out[start:start + chunk] = (cw @ values) / cw.sum(axis=1, keepdims=True)
    return out


def resample_moments(values: np.ndarray, rows_one: np.ndarray):
    """Mean and SD (ddof=1) of one resample, computed exactly as radii234 fit_study."""
    sample = values[rows_one]
    return sample.mean(axis=0), sample.std(axis=0, ddof=1)


def percentile_interval(samples: np.ndarray, axis: int = 0, levels=(16, 50, 84)) -> np.ndarray:
    """NaN-aware percentiles (draws that failed are NaN and are ignored)."""
    return np.nanpercentile(samples, levels, axis=axis)
