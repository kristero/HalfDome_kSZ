"""Halo-scatter and mean-profile covariances, shrinkage, factor models.

Two different covariances appear in this study and must not be confused:

* *intrinsic* (Lee22 Fig. 1): covariance/correlation across haloes of the
  per-halo ln F on resolved shells (``halo_log_covariance``);
* *sampling* covariance of ln(mean F) used by the GLS fit
  (``bootstrap_log_mean_covariance``; the delta-method limit is
  ``delta_log_mean_covariance`` = Cov(F)/(N m m^T), whose correlation is that of
  the *linear* per-halo F).

Shrinkage follows Schaefer & Strimmer (2005) for a correlation matrix shrunk
towards a fixed target T (here a pooled correlation):
lambda* = sum_{i!=j} Var(r_ij) / sum_{i!=j} (r_ij - t_ij)^2, clipped to [0, 1].
Var(r_ij) is estimated by default from halo resamples of the sample correlation
(``correlation_bootstrap_variance``): the analytic Schaefer-Strimmer variance of
the standardised products ignores the self-normalisation of r and overestimates
lambda by ~2.5x at N = 300 for the heavy-tailed (lognormal) linear profiles
(analysis/tests: shrinkage test).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


# ----------------------------------------------------------------------------- basics
def log_values(values: np.ndarray, mask: np.ndarray, drop_nonpositive: bool = True):
    """ln F on ``mask`` shells; haloes with any F <= 0 there are dropped (returned ``kept``)."""
    sub = values[:, np.asarray(mask, bool)]
    kept = np.all(sub > 0, axis=1) & np.all(np.isfinite(sub), axis=1)
    if not drop_nonpositive and not kept.all():
        raise ValueError("non-positive values on requested shells")
    return np.log(sub[kept]), kept


def weighted_cov(X: np.ndarray, weights=None) -> np.ndarray:
    """Unbiased covariance of rows of X; reliability weights if given."""
    if weights is None:
        return np.atleast_2d(np.cov(X, rowvar=False))
    w = np.asarray(weights, float)
    w = w / w.sum()
    mean = w @ X
    D = X - mean
    return (D * w[:, None]).T @ D / (1 - np.sum(w**2))


def cov_to_corr(C: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    return R


def corr_to_cov(R: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return R * np.outer(sd, sd)


def halo_log_covariance(values: np.ndarray, mask: np.ndarray, weights=None,
                        logmass=None, residualise: bool = False) -> dict:
    """Intrinsic scatter: covariance/correlation of per-halo ln F on ``mask`` shells.

    ``residualise`` removes a linear trend in ``logmass`` (weighted if ``weights``).
    """
    logs, kept = log_values(values, mask)
    w = None if weights is None else np.asarray(weights, float)[kept]
    if residualise and logmass is not None:
        logs = residualise_mass(logs, np.asarray(logmass)[kept], w)
    C = weighted_cov(logs, w)
    return {"cov": C, "corr": cov_to_corr(C), "sd": np.sqrt(np.diag(C)), "n": int(kept.sum()),
            "dropped": int((~kept).sum()), "kept": kept,
            "n_eff": float(1 / np.sum((w / w.sum())**2)) if w is not None else float(kept.sum())}


def residualise_mass(X: np.ndarray, logmass: np.ndarray, weights=None) -> np.ndarray:
    """Subtract the (weighted) mean and a linear trend in log mass from every column."""
    w = np.ones(len(X)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mx = w @ X
    mm = w @ logmass
    dm = logmass - mm
    var = w @ dm**2
    slope = (w * dm) @ (X - mx) / var if var > 0 else np.zeros(X.shape[1])
    return X - mx - dm[:, None] * slope[None, :]


# ----------------------------------------------------------------------------- mean covariance
def bootstrap_log_mean_covariance(values: np.ndarray, mask: np.ndarray, rows: np.ndarray,
                                  weights=None, chunk: int = 256) -> dict:
    """Cov[ln mean F] over halo bootstrap draws ``rows`` [draw, n] on ``mask`` shells."""
    sub = values[:, np.asarray(mask, bool)]
    n = sub.shape[0]
    w = None if weights is None else np.asarray(weights, float)
    logs = np.empty((rows.shape[0], sub.shape[1]))
    for start in range(0, rows.shape[0], chunk):
        block = rows[start:start + chunk]
        flat = (np.arange(len(block))[:, None] * n + block).ravel()
        counts = np.bincount(flat, minlength=len(block) * n).reshape(len(block), n).astype(float)
        if w is None:
            means = counts @ sub / n
        else:
            cw = counts * w[None, :]
            means = cw @ sub / cw.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs[start:start + chunk] = np.log(means)
    good = np.all(np.isfinite(logs), axis=1)
    C = np.atleast_2d(np.cov(logs[good], rowvar=False))
    return {"cov": C, "corr": cov_to_corr(C), "sd": np.sqrt(np.diag(C)),
            "draws": int(good.sum()), "failed_draws": int((~good).sum())}


def delta_log_mean_covariance(values: np.ndarray, mask: np.ndarray, weights=None,
                              n_mean=None, mean=None) -> np.ndarray:
    """Delta-method Cov[ln mean F] = Cov(F) / (N_eff m m^T).

    ``n_mean``/``mean`` let a covariance estimated on one halo set describe the mean of
    another (cross-fitting); by default both come from ``values``.
    """
    sub = values[:, np.asarray(mask, bool)]
    if weights is None:
        S = np.atleast_2d(np.cov(sub, rowvar=False))
        m = sub.mean(axis=0) if mean is None else mean
        neff = len(sub) if n_mean is None else n_mean
    else:
        w = np.asarray(weights, float)
        S = weighted_cov(sub, w)
        m = (w / w.sum()) @ sub if mean is None else mean
        neff = 1 / np.sum((w / w.sum())**2) if n_mean is None else n_mean
    return S / (neff * np.outer(m, m))


# ----------------------------------------------------------------------------- shrinkage
def schafer_strimmer_lambda(X: np.ndarray, target_corr: np.ndarray) -> float:
    """Optimal shrinkage intensity of corr(X) towards ``target_corr`` (fixed target)."""
    X = np.asarray(X, float)
    n, p = X.shape
    if n < 3 or p < 2:
        return 1.0
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    W = Z[:, :, None] * Z[:, None, :]
    wbar = W.mean(axis=0)
    r = n / (n - 1) * wbar
    var_r = n / (n - 1)**3 * np.sum((W - wbar)**2, axis=0)
    off = ~np.eye(p, dtype=bool)
    denominator = np.sum((r[off] - np.asarray(target_corr)[off])**2)
    if denominator <= 0:
        return 1.0
    return float(np.clip(np.sum(var_r[off]) / denominator, 0.0, 1.0))


def correlation_bootstrap_variance(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Element-wise variance of corr(X) over halo resamples ``rows`` [draw, n]."""
    rr = np.array([np.corrcoef(X[r], rowvar=False) for r in rows])
    return rr.var(axis=0, ddof=1)


def lambda_from_variance(R: np.ndarray, target: np.ndarray, var_r: np.ndarray) -> float:
    """lambda = sum_{i!=j} Var(r_ij) / sum_{i!=j} (r_ij - t_ij)^2, clipped to [0, 1]."""
    p = R.shape[0]
    off = ~np.eye(p, dtype=bool)
    denominator = np.sum((np.asarray(R)[off] - np.asarray(target)[off])**2)
    if denominator <= 0:
        return 1.0
    return float(np.clip(np.sum(np.asarray(var_r)[off]) / denominator, 0.0, 1.0))


def shrink_correlation(R: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    out = lam * np.asarray(target) + (1 - lam) * np.asarray(R)
    np.fill_diagonal(out, 1.0)
    return out


def shrunk_covariance(C: np.ndarray, target_corr: np.ndarray, X: np.ndarray | None = None,
                      var_r: np.ndarray | None = None, force_pooled: bool = False,
                      method: str = "bootstrap", draws: int = 50, rng=None) -> tuple[np.ndarray, float]:
    """diag(sd) [lam T + (1-lam) R] diag(sd) with R = corr(C), sd = sqrt(diag C).

    lam: 1 if ``force_pooled`` (N < 3p: pooled correlation x own variances);
    else from ``var_r`` (element-wise variance of the sample correlation) if given;
    else ``method='bootstrap'`` (``draws`` halo resamples of X; robust for the
    heavy-tailed linear profiles) or ``method='ss'`` (Schaefer-Strimmer analytic
    variance, which assumes light tails and overestimates lam for lognormal F).
    """
    sd = np.sqrt(np.diag(C))
    R = cov_to_corr(C)
    if force_pooled:
        lam = 1.0
    elif var_r is not None:
        lam = lambda_from_variance(R, target_corr, var_r)
    elif method == "ss":
        lam = schafer_strimmer_lambda(X, target_corr)
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        rows = rng.integers(0, len(X), size=(draws, len(X)))
        lam = lambda_from_variance(R, target_corr, correlation_bootstrap_variance(X, rows))
    return corr_to_cov(shrink_correlation(R, target_corr, lam), sd), lam


def nearest_correlation(R: np.ndarray, floor: float = 1e-8, method: str = "clip",
                        tol: float = 1e-12, max_iter: int = 2000) -> np.ndarray:
    """Unit-diagonal PSD correlation matrix close to R.

    ``method='clip'`` (default; used by the step-0.3 products): eigenvalues clipped at
    ``floor`` then the diagonal rescaled to 1.  ``method='higham'``: Higham (2002)
    alternating projections with Dykstra's correction (nearest correlation matrix in
    Frobenius norm, eigenvalues >= floor); ~15-25% smaller Frobenius change than
    'clip' on the pooled targets here.
    """
    R = (np.asarray(R, float) + np.asarray(R, float).T) / 2
    if method == "higham":
        Y = R.copy()
        dS = np.zeros_like(R)
        for _ in range(max_iter):
            Rk = Y - dS
            w, V = np.linalg.eigh(Rk)
            X = (V * np.maximum(w, floor)) @ V.T
            dS = X - Rk
            Yn = X.copy()
            np.fill_diagonal(Yn, 1.0)
            converged = np.linalg.norm(Yn - Y) <= tol * np.linalg.norm(Y)
            Y = Yn
            if converged:
                break
        R = Y
    values, vectors = np.linalg.eigh(R)
    values = np.maximum(values, floor)
    out = (vectors * values) @ vectors.T
    d = np.sqrt(np.diag(out))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return out


def corr_model(lnx: np.ndarray, a: float, L: float) -> np.ndarray:
    """rho_ij = a + (1 - a) exp(-|ln x_i - ln x_j| / L)."""
    d = np.abs(np.subtract.outer(lnx, lnx))
    return a + (1 - a) * np.exp(-d / L)


def fit_corr_model(R: np.ndarray, lnx: np.ndarray, valid=None,
                   a_bounds=(-1.0, 1.0), lnL_bounds=(np.log(1e-3), np.log(1e3))) -> dict:
    """Least squares (a, L) on the off-diagonal elements; a profiled in closed form."""
    R = np.asarray(R, float)
    p = len(lnx)
    iu = np.triu_indices(p, 1)
    d = np.abs(np.subtract.outer(lnx, lnx))[iu]
    r = R[iu]
    use = np.isfinite(r) if valid is None else (np.asarray(valid)[iu] & np.isfinite(r))
    d, r = d[use], r[use]
    if len(r) < 2:
        return {"a": np.nan, "L": np.nan, "rms": np.nan, "pairs": int(len(r))}

    def profile(lnL):
        e = np.exp(-d / np.exp(lnL))
        u = 1 - e
        a = np.sum((r - e) * u) / max(np.sum(u * u), 1e-300)
        a = float(np.clip(a, *a_bounds))
        return a, np.sum((r - (a + (1 - a) * e))**2)

    grid = np.linspace(lnL_bounds[0], lnL_bounds[1], 121)
    E = np.exp(-d[None, :] / np.exp(grid)[:, None])
    U = 1 - E
    a_grid = np.clip(np.sum((r[None, :] - E) * U, axis=1) / np.maximum(np.sum(U * U, axis=1), 1e-300),
                     *a_bounds)
    costs = np.sum((r[None, :] - (a_grid[:, None] + (1 - a_grid[:, None]) * E))**2, axis=1)
    k = int(np.argmin(costs))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    if hi > lo:
        res = minimize_scalar(lambda g: profile(g)[1], bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-6})
        lnL = float(res.x) if res.fun <= costs[k] else float(grid[k])
    else:
        lnL = float(grid[k])
    a, cost = profile(lnL)
    return {"a": a, "L": float(np.exp(lnL)), "rms": float(np.sqrt(cost / len(r))), "pairs": int(len(r)),
            "at_L_bound": bool(lnL <= lnL_bounds[0] + 1e-6 or lnL >= lnL_bounds[1] - 1e-6)}


def pooled_correlation(corrs, masks, ns, n_shells: int, lnx=None, min_weight: float = 1.0,
                       fill: str = "nearest", projection: str = "clip") -> dict:
    """N-weighted Fisher-z average of per-bin correlations on a common shell grid.

    Weights are N - 3 per bin.  Shells observed by no pooled bin are filled
    (``fill='nearest'``) with the correlations of the nearest covered shell, two
    filled shells sharing one source get that source's adjacent-shell
    correlation; ``fill='model'`` uses the (a, L) model fitted to the pooled
    entries instead (requires ``lnx``).  The result is projected to the nearest
    unit-diagonal PSD matrix (``projection='clip'`` default, or ``'higham'``); the
    projection change is reported.  Note: pooling bins whose correlation structure
    differs (electron/gas densities across 13.3-15.1) gives a strongly indefinite raw
    average (min eigenvalue -0.4 to -0.6); pool neighbouring bins only in later tests.
    """
    zsum = np.zeros((n_shells, n_shells))
    wsum = np.zeros((n_shells, n_shells))
    for R, m, n in zip(corrs, masks, ns):
        idx = np.flatnonzero(m)
        w = max(n - 3, 0)
        if w <= 0 or len(idx) == 0:
            continue
        z = np.arctanh(np.clip(R, -0.999999, 0.999999))
        zsum[np.ix_(idx, idx)] += w * z
        wsum[np.ix_(idx, idx)] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.tanh(zsum / wsum)
    observed = wsum >= min_weight
    pooled[~observed] = np.nan
    np.fill_diagonal(pooled, 1.0)
    covered = np.diag(wsum) >= min_weight
    cov_idx = np.flatnonzero(covered)
    filled = pooled.copy()
    model = None
    if lnx is not None and len(cov_idx) >= 3:
        model = fit_corr_model(pooled[np.ix_(cov_idx, cov_idx)], np.asarray(lnx)[cov_idx],
                               valid=observed[np.ix_(cov_idx, cov_idx)])
    if len(cov_idx) and (~observed).any():
        if fill == "model" and model is not None and np.isfinite(model["a"]):
            filled = np.where(observed, pooled, corr_model(np.asarray(lnx), model["a"], model["L"]))
        else:
            source = np.array([cov_idx[np.argmin(np.abs(cov_idx - i))] for i in range(n_shells)])
            adjacent = {}
            for c in np.unique(source):
                neighbours = [j for j in (c - 1, c + 1) if 0 <= j < n_shells and observed[c, j]]
                adjacent[c] = float(np.mean([pooled[c, j] for j in neighbours])) if neighbours else 1.0
            for i in range(n_shells):
                for j in range(n_shells):
                    if observed[i, j]:
                        continue
                    si, sj = source[i], source[j]
                    filled[i, j] = adjacent[si] if si == sj and i != j else pooled[si, sj]
    np.fill_diagonal(filled, 1.0)
    projected = nearest_correlation(filled, method=projection) if np.all(np.isfinite(filled)) else filled
    change = np.abs(projected - filled)
    return {"corr": projected, "raw": pooled, "observed": observed, "weight": wsum,
            "covered": covered, "model": model, "fill": fill,
            "projection_max_change": float(np.nanmax(change)),
            "projection_max_change_observed": float(np.nanmax(np.where(observed, change, 0.0)))}


# ----------------------------------------------------------------------------- factor model
def factor_analysis(S: np.ndarray, k: int, n: int | None = None, max_iter: int = 5000,
                    tol: float = 1e-10, psi_floor: float = 1e-6) -> dict:
    """EM maximum-likelihood factor model S ~ L L^T + diag(psi) (k factors).

    Works with rank-deficient S (N < p).  Returns loadings [p, k], psi, cov.
    """
    S = (np.asarray(S, float) + np.asarray(S, float).T) / 2
    p = S.shape[0]
    values, vectors = np.linalg.eigh(S)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    noise = max(np.mean(values[k:]) if p > k else 0.0, 0.0)
    L = vectors[:, :k] * np.sqrt(np.maximum(values[:k] - noise, 1e-12))
    floor = psi_floor * np.maximum(np.diag(S), 1e-300)
    psi = np.maximum(np.diag(S) - np.sum(L**2, axis=1), floor)
    previous = -np.inf
    loglik = -np.inf
    for iteration in range(max_iter):
        inv_psi = 1 / psi
        M = np.eye(k) + (L.T * inv_psi) @ L
        Minv = np.linalg.inv(M)
        sigma_inv = np.diag(inv_psi) - (inv_psi[:, None] * L) @ Minv @ (L.T * inv_psi)
        beta = L.T @ sigma_inv
        Ezz = np.eye(k) - beta @ L + beta @ S @ beta.T
        L = S @ beta.T @ np.linalg.inv(Ezz)
        psi = np.maximum(np.diag(S - L @ beta @ S), floor)
        sign, logdet = np.linalg.slogdet(L @ L.T + np.diag(psi))
        loglik = -0.5 * (logdet + np.trace(np.linalg.solve(L @ L.T + np.diag(psi), S)))
        if abs(loglik - previous) < tol * max(1.0, abs(loglik)):
            break
        previous = loglik
    cov = L @ L.T + np.diag(psi)
    return {"loadings": L, "psi": psi, "cov": cov, "iterations": iteration + 1,
            "loglik_per_sample": float(loglik), "explained_fraction": float(np.trace(L @ L.T) / np.trace(S))}


def hartlap(n: int, p: int) -> float:
    """Hartlap et al. (2007) inverse-covariance debiasing factor (N - p - 2)/(N - 1)."""
    return float((n - p - 2) / (n - 1)) if n > 1 else float("nan")
