"""Shell-averaged B12/B16 fits on ln(mean profile): diagonal and GLS objectives.

Model: radii234 ``models.shape`` averaged over each shell with ``models.shell_grid``
(32-point Gauss-Legendre, volume weights).  Parameters are fitted in ln(theta)
with radii234 ``models.BOUNDS``, x0 = clip(ln start, lo + 1e-7, hi - 1e-7),
``least_squares(..., max_nfev=500, ftol=xtol=gtol=1e-9)``.

Objectives (all on ln of the stacked mean):
  (a) ``objective_a``: diagonal, sigma_s = max(SD_s/m_s, 0.05)    [radii234 fit_one]
  (b) ``objective_b``: diagonal, sigma_s = error on ln(mean) (no floor)
  (c) ``objective_c``: GLS, C = C_stat (+ sigma_d^2 I), whitened with Cholesky;
      ``sigma_d='solve'`` iterates the explicit model-discrepancy term to chi2/nu = 1.

``jac='2-point'`` reproduces ``models.fit_profile`` bit for bit (diagonal mode);
``jac='analytic'`` uses d ln S / d ln theta = <F d ln F/d ln theta>_shell / S
(the shell-averaged derivative; cf. flamingo_stacks.Objective.evaluate and
extend_analytic.coefficient_derivatives).
Acceptance: success, bound distance > 1e-5, Jacobian condition < 1e6 (fit_study).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, solve_triangular
from scipy.optimize import brentq, least_squares

from . import _legacy

M = _legacy.models()
P = _legacy.pilot_physics()
PARAM_NAMES = M.PARAM_NAMES
BOUNDS = M.BOUNDS
DEFAULT_START = {"b12": np.asarray(P.B12[:3], float), "b16": np.asarray(P.B16[:3], float)}
MIN_SHELLS = 6


# ----------------------------------------------------------------------------- model
def shape(family: str, x, parameters):
    return M.shape(family, x, parameters)


def shell_grid(edges):
    return M.shell_grid(tuple(np.asarray(edges, float).tolist()))


def shell_model(family: str, parameters, edges) -> np.ndarray:
    """Shell-averaged model on every shell of ``edges`` (exactly models.shell)."""
    return M.shell(family, parameters, np.asarray(edges, float))


def dlnshape(family: str, x, parameters) -> np.ndarray:
    """d ln F / d ln theta_k at radii x: [3, *x.shape]."""
    p, a, b = parameters
    x = np.asarray(x, float)
    out = np.empty((3,) + x.shape)
    out[0] = 1.0
    if family == "b12":
        q = x / a
        out[1] = .3 + b * q / (1 + q)
        out[2] = -b * np.log1p(q)
    elif family == "b16":
        q = x / .5
        qa = q**a
        out[1] = (b - .2) / a * np.log1p(qa) - (b - .2) * qa / (1 + qa) * np.log(q)
        out[2] = -b / a * np.log1p(qa)
    else:
        raise ValueError(family)
    return out


def shell_model_jacobian(family: str, parameters, edges):
    """S [shell] and J = d ln S / d ln theta [shell, 3] (exact shell averages)."""
    x, w = shell_grid(edges)
    F = shape(family, x, parameters)
    Fw = F * w
    S = np.sum(Fw, axis=1)
    D = dlnshape(family, x, parameters)
    J = np.einsum("ksn,sn->sk", D, Fw) / S[:, None]
    return S, J


def numerical_jacobian(family: str, parameters, edges, step: float = 1e-6) -> np.ndarray:
    """Central differences of ln S in ln theta (for the analytic-Jacobian check)."""
    theta = np.log(np.asarray(parameters, float))
    out = np.empty((len(edges) - 1, 3))
    for k in range(3):
        up, down = theta.copy(), theta.copy()
        up[k] += step
        down[k] -= step
        out[:, k] = (np.log(shell_model(family, np.exp(up), edges))
                     - np.log(shell_model(family, np.exp(down), edges))) / (2 * step)
    return out


# ----------------------------------------------------------------------------- core solver
def accepted(fit) -> bool:
    return bool(fit is not None and fit["success"] and fit["bound_distance"] > 1e-5
                and fit["jacobian_condition"] < 1e6)


def _core(family, mean, mask, edges, start, jac, sigma=None, chol=None, max_nfev=500):
    """One least-squares solve. Exactly one of ``sigma`` (diagonal) / ``chol`` (lower L)."""
    mask = np.asarray(mask, bool)
    lo, hi = np.log(np.asarray(BOUNDS[family], float))
    initial = np.log(np.asarray(start if start is not None else DEFAULT_START[family], float))
    edges = np.asarray(edges, float)
    target = np.log(mean[mask])
    cache = {}

    def evaluate(theta):
        key = theta.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = shell_model_jacobian(family, np.exp(theta), edges)
        return cache[key]

    if chol is None:
        error = sigma[mask]
        if jac == "2-point":
            # Literal models.fit_profile residual (bit-for-bit reproduction).
            def residual(theta):
                prediction = shell_model(family, np.exp(theta), edges)
                return (np.log(prediction[mask]) - np.log(mean[mask])) / error
            jacobian = "2-point"
        else:
            def residual(theta):
                return (np.log(evaluate(theta)[0][mask]) - target) / error

            def jacobian(theta):
                return evaluate(theta)[1][mask] / error[:, None]
    else:
        def residual(theta):
            return solve_triangular(chol, np.log(evaluate(theta)[0][mask]) - target, lower=True)

        if jac == "2-point":
            jacobian = "2-point"
        else:
            def jacobian(theta):
                return solve_triangular(chol, evaluate(theta)[1][mask], lower=True)

    fitted = least_squares(residual, np.clip(initial, lo + 1e-7, hi - 1e-7), jac=jacobian,
                           bounds=(lo, hi), max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    p = np.exp(fitted.x)
    singular = np.linalg.svd(fitted.jac, compute_uv=False)
    bound = float(np.min(np.minimum(fitted.x - lo, hi - fitted.x)))
    S, J = shell_model_jacobian(family, p, edges)
    rms = float(np.sqrt(np.mean((S[mask] / mean[mask] - 1)**2)) * 100)
    if chol is None:
        Jw = J[mask] / sigma[mask][:, None]
    else:
        Jw = solve_triangular(chol, J[mask], lower=True)
    try:
        cov_log = np.linalg.inv(Jw.T @ Jw)
    except np.linalg.LinAlgError:
        cov_log = np.full((3, 3), np.nan)
    chi2 = float(np.sum(fitted.fun**2))
    nshell = int(mask.sum())
    return {"parameters": p.tolist(), "log_parameters": fitted.x.copy(), "success": bool(fitted.success),
            "bound_distance": bound, "jacobian_condition": float(singular[0] / max(singular[-1], 1e-30)),
            "rms_percent": rms, "nfev": int(fitted.nfev), "chi2": chi2, "n_shells": nshell,
            "ndof": nshell - 3, "chi2_per_dof": chi2 / max(nshell - 3, 1),
            "cov_log_parameters": cov_log, "jacobian_lnS": J[mask], "mask": mask.copy(),
            "model": S, "status": int(fitted.status)}


# ----------------------------------------------------------------------------- objectives
def fit_diagonal(family, mean, mask, sigma, start=None, edges=None, jac="analytic"):
    edges = P.EDGES if edges is None else edges
    fit = _core(family, np.asarray(mean, float), mask, edges, start, jac, sigma=np.asarray(sigma, float))
    fit["objective"] = "diagonal"
    fit["sigma"] = np.asarray(sigma, float)[np.asarray(mask, bool)]
    fit["sigma_d"] = 0.0
    fit["accepted"] = accepted(fit)
    return fit


def fit_gls(family, mean, mask, cov, start=None, edges=None, jac="analytic", sigma_d: float = 0.0):
    """GLS with covariance ``cov`` given on the ``mask`` shells (p x p) or on all shells."""
    edges = P.EDGES if edges is None else edges
    mask = np.asarray(mask, bool)
    C = np.asarray(cov, float)
    if C.shape[0] != mask.sum():
        C = C[np.ix_(mask, mask)]
    C = C + sigma_d**2 * np.eye(len(C))
    chol = cholesky(C, lower=True)
    fit = _core(family, np.asarray(mean, float), mask, edges, start, jac, chol=chol)
    fit["objective"] = "gls"
    fit["sigma_d"] = float(sigma_d)
    fit["cov_total"] = C
    fit["accepted"] = accepted(fit)
    return fit


def objective_a(family, mean, std, mask, start=None, edges=None, jac="2-point", floor: float = .05):
    """radii234 fit_study.fit_one: valid = mask & finite & mean > 0; sigma = max(SD/mean, 5%)."""
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    valid = np.asarray(mask, bool) & np.isfinite(mean) & (mean > 0)
    if valid.sum() < MIN_SHELLS:
        return None
    error = np.maximum(np.divide(std, mean, out=np.ones_like(std), where=mean > 0), floor)
    # models.fit_profile applies np.maximum(error, .05) once more (idempotent).
    error = np.maximum(error, floor)
    fit = fit_diagonal(family, mean, valid, error, start=start, edges=edges, jac=jac)
    fit["objective"] = "a: diagonal SD/mean, 5% floor"
    return fit


def objective_b(family, mean, sigma_log, mask, start=None, edges=None, jac="analytic"):
    """Diagonal error on ln(mean) (bootstrap or delta method), no floor."""
    mean = np.asarray(mean, float)
    valid = np.asarray(mask, bool) & np.isfinite(mean) & (mean > 0) & np.isfinite(sigma_log) & (sigma_log > 0)
    if valid.sum() < MIN_SHELLS:
        return None
    fit = fit_diagonal(family, mean, valid, np.where(valid, sigma_log, 1.0), start=start, edges=edges, jac=jac)
    fit["objective"] = "b: diagonal error on ln mean"
    return fit


def objective_c(family, mean, cov_stat, mask, start=None, edges=None, jac="analytic",
                sigma_d="solve", tol: float = 1e-5, max_iter: int = 60, sigma_d_start: float | None = None):
    """GLS on ln(mean) with ``cov_stat`` (+ explicit discrepancy sigma_d).

    ``cov_stat`` is given on the ``mask`` shells or on all shells.  With
    ``sigma_d='solve'`` the discrepancy is iterated until chi2/nu = 1 (sigma_d = 0
    if chi2/nu <= 1 already): for fixed theta, chi2(s) = sum_k u_k^2/(lambda_k + s)
    with C_stat = U diag(lambda) U^T, u = U^T r, solved with brentq, then theta is
    refitted; repeated until sigma_d changes by < tol (relative).  ``sigma_d_start``
    (e.g. the full-sample value for a bootstrap replicate) only changes the first
    iterate, not the fixed point; chi2/nu at sigma_d = 0 is then not evaluated.
    """
    mean = np.asarray(mean, float)
    mask = np.asarray(mask, bool) & np.isfinite(mean) & (mean > 0)
    if mask.sum() < MIN_SHELLS:
        return None
    C = np.asarray(cov_stat, float)
    if C.shape[0] == len(mean):
        C = C[np.ix_(mask, mask)]
    elif C.shape[0] != mask.sum():
        raise ValueError("cov_stat must be given on the mask shells or on all shells")
    if sigma_d != "solve":
        fit = fit_gls(family, mean, mask, C, start=start, edges=edges, jac=jac, sigma_d=float(sigma_d))
        fit["objective"] = "c: GLS"
        fit["sigma_d_iterations"] = 0
        return fit
    warm = sigma_d_start is not None and np.isfinite(sigma_d_start) and sigma_d_start > 0
    s2 = float(sigma_d_start)**2 if warm else 0.0
    fit = fit_gls(family, mean, mask, C, start=start, edges=edges, jac=jac, sigma_d=float(np.sqrt(s2)))
    nu = fit["ndof"]
    chi2_stat_only = np.nan if warm else fit["chi2_per_dof"]
    iterations = 0
    if nu > 0 and (warm or fit["chi2"] > nu):
        values, vectors = eigh(C)
        values = np.maximum(values, 0.0)
        for iterations in range(1, max_iter + 1):
            r = np.log(fit["model"][mask]) - np.log(mean[mask])
            u2 = (vectors.T @ r)**2

            def g(s):
                return float(np.sum(u2 / (values + s)) - nu)

            if g(0.0) <= 0:
                s_new = 0.0
            else:
                hi = max(4 * s2, float(np.mean(r**2)), 1e-12)
                while g(hi) > 0:
                    hi *= 4
                s_new = brentq(g, 0.0, hi, xtol=1e-16, rtol=1e-12)
            fit = fit_gls(family, mean, mask, C, start=fit["parameters"], edges=edges, jac=jac,
                          sigma_d=float(np.sqrt(s_new)))
            converged = abs(np.sqrt(s_new) - np.sqrt(s2)) <= tol * max(np.sqrt(s_new), 1e-8)
            s2 = s_new
            if converged:
                break
    fit["objective"] = "c: GLS + sigma_d (chi2/nu = 1)"
    fit["sigma_d_iterations"] = iterations
    fit["chi2_per_dof_at_sigma_d_0"] = chi2_stat_only
    return fit


def objective_a_chi2(family, mean, std, mask, parameters, edges=None, floor: float = .05) -> float:
    """chi^2 of objective (a) at given parameters (to compare minimisers)."""
    edges = P.EDGES if edges is None else edges
    mean = np.asarray(mean, float)
    valid = np.asarray(mask, bool) & np.isfinite(mean) & (mean > 0)
    error = np.maximum(np.divide(std, mean, out=np.ones_like(std), where=mean > 0), floor)
    model = shell_model(family, np.asarray(parameters, float), edges)
    return float(np.sum(((np.log(model[valid]) - np.log(mean[valid])) / error[valid])**2))


# ----------------------------------------------------------------------------- diagnostics
def influence(J: np.ndarray, C: np.ndarray) -> dict:
    """Influence rows H = (J^T C^-1 J)^-1 J^T C^-1 [3, shell] and participation-ratio
    effective number of shells per parameter, (sum|H_k|)^2 / sum H_k^2."""
    J = np.asarray(J, float)
    C = np.asarray(C, float)
    if C.ndim == 1:
        C = np.diag(C**2)
    factor = cho_factor(C, lower=True)
    CiJ = cho_solve(factor, J)
    F = J.T @ CiJ
    Finv = np.linalg.inv(F)
    H = Finv @ CiJ.T
    neff = np.sum(np.abs(H), axis=1)**2 / np.sum(H**2, axis=1)
    return {"H": H, "n_eff": neff, "cov_log_parameters": Finv}


def fit_influence(fit: dict) -> dict:
    """Influence rows for a fit returned by objective_a/b/c (its own weighting)."""
    if fit is None:
        return None
    if fit["objective"].startswith("c"):
        return influence(fit["jacobian_lnS"], fit["cov_total"])
    return influence(fit["jacobian_lnS"], fit["sigma"])


def cross_fit(family, values, mask, cov_builder, rng, n_splits: int = 1, start=None,
              edges=None, jac="analytic", sigma_d="solve"):
    """Cross-fitting: covariance from one random half, mean from the other, swap, average.

    ``cov_builder(cov_values, mean_values) -> C`` returns Cov[ln mean] on the mask
    shells for the mean of ``mean_values`` estimated from ``cov_values`` only.
    Returns the mean of the ln-parameters over halves and splits, and the fits.
    """
    n = len(values)
    fits = []
    for _ in range(n_splits):
        order = rng.permutation(n)
        halves = (order[:n // 2], order[n // 2:])
        for cov_rows, mean_rows in (halves, halves[::-1]):
            C = cov_builder(values[cov_rows], values[mean_rows])
            mean = values[mean_rows].mean(axis=0)
            fit = objective_c(family, mean, C, mask, start=start, edges=edges, jac=jac, sigma_d=sigma_d)
            fits.append(fit)
    good = [f for f in fits if f is not None and accepted(f)]
    if not good:
        return {"parameters": None, "fits": fits, "accepted_halves": 0}
    logp = np.mean([f["log_parameters"] for f in good], axis=0)
    return {"parameters": np.exp(logp).tolist(), "log_parameters": logp, "fits": fits,
            "accepted_halves": len(good), "halves": len(fits)}


def plain_fit(fit: dict | None) -> dict | None:
    """JSON-friendly subset of a fit dict."""
    if fit is None:
        return None
    keys = ("parameters", "success", "bound_distance", "jacobian_condition", "rms_percent", "nfev",
            "chi2", "n_shells", "ndof", "chi2_per_dof", "sigma_d", "accepted", "objective")
    out = {k: fit[k] for k in keys if k in fit}
    err = np.sqrt(np.clip(np.diag(fit["cov_log_parameters"]), 0, None))
    out["analytic_sigma_ln_parameters"] = err.tolist()
    return out


# ============================================================================ general objectives (d)-(g)
# Added for the observable-driven calibration test.  The legacy objectives (a)-(c)
# above are unchanged (B12/B16, bit-exact paths); everything below works for any
# ``families.Family`` (b12, b16, gnfw5, b12env, b12spline) and any of:
#   sigma   : diagonal errors                     chi2 = sum (r_s / sigma_s)^2
#   cov     : full covariance (Cholesky)          chi2 = r^T C^-1 r
#   weights : (d) per-shell weights w_s >= 0      chi2 = sum w_s r_s^2
#   metric  : (e) PSD matrix M (e.g. observable)  chi2 = r^T M r   (M = U diag(l) U^T, W = l^1/2 U^T)
# with r = ln model - ln mean on the fit shells, plus family penalty residuals
# (spline prior) and (g) optional extra residual callables extra(p) -> array,
# appended to the residual vector (their Jacobian by finite differences).
from . import families as FAM  # noqa: E402


def whitener_from_metric(M: np.ndarray, rel_tol: float = 1e-12) -> np.ndarray:
    """W with W^T W = M on the positive eigen-space of a symmetric PSD matrix M."""
    M = (np.asarray(M, float) + np.asarray(M, float).T) / 2
    values, vectors = np.linalg.eigh(M)
    keep = values > rel_tol * max(values.max(), 0.0)
    return (vectors[:, keep] * np.sqrt(values[keep])).T


def _restrict(matrix, mask, n_all):
    A = np.asarray(matrix, float)
    if A.shape[0] == n_all and A.shape[0] != int(np.sum(mask)):
        return A[np.ix_(mask, mask)]
    if A.shape[0] != int(np.sum(mask)):
        raise ValueError("matrix must be given on the fit shells or on all shells")
    return A


def fit_general(family, mean, mask, *, sigma=None, cov=None, weights=None, metric=None,
                extra_residuals=None, start=None, edges=None, jac: str = "analytic",
                fd_step: float = 1e-6, max_nfev: int = 1000, objective: str | None = None) -> dict | None:
    """Least squares in theta (ln of positive parameters) for any family and metric.

    Exactly one of ``sigma`` / ``cov`` / ``weights`` / ``metric`` (none: unit weights).
    Arrays may be given on all shells or on the fit shells.  Returns the same keys as
    the legacy fits plus ``theta``, ``n_params``, ``chi2_profile`` (whitened profile part),
    ``family``; ``cov_theta`` is (J_w^T J_w)^-1 of the full residual vector.
    """
    fam = FAM.get(family)
    edges = P.EDGES if edges is None else np.asarray(edges, float)
    mean = np.asarray(mean, float)
    mask = np.asarray(mask, bool) & np.isfinite(mean) & (mean > 0)
    n_all = len(mean)
    if mask.sum() < max(MIN_SHELLS, fam.n + 1):
        return None
    given = [v is not None for v in (sigma, cov, weights, metric)]
    if sum(given) > 1:
        raise ValueError("give at most one of sigma, cov, weights, metric")
    target = np.log(mean[mask])
    if cov is not None:
        L = cholesky(_restrict(cov, mask, n_all), lower=True)
        whiten = lambda v: solve_triangular(L, v, lower=True)                    # noqa: E731
        mode = "cov"
    elif metric is not None:
        W = whitener_from_metric(_restrict(metric, mask, n_all))
        whiten = lambda v: W @ v                                                  # noqa: E731
        mode = "metric"
    else:
        if weights is not None:
            w = np.asarray(weights, float)
            w = w[mask] if len(w) == n_all else w
            scale = np.sqrt(np.maximum(w, 0.0))
            mode = "weights"
        else:
            s = np.ones(mask.sum()) if sigma is None else np.asarray(sigma, float)
            s = s[mask] if len(s) == n_all else s
            scale = 1.0 / s
            mode = "sigma"
        whiten = lambda v: (v.T * scale).T                                            # noqa: E731
    extras = list(extra_residuals or [])
    lo, hi = fam.theta_bounds()
    theta0 = np.clip(fam.to_theta(fam.start if start is None else start), lo + 1e-7, hi - 1e-7)
    cache = {}

    def evaluate(theta):
        key = theta.tobytes()
        if key not in cache:
            cache.clear()
            p = fam.to_params(theta)
            S, J = FAM.shell_model_jacobian(fam, p, edges)
            cache[key] = (p, S, J)
        return cache[key]

    def extra_values(theta):
        p = fam.to_params(theta)
        return [np.atleast_1d(np.asarray(f(p), float)) for f in extras]

    def residual(theta):
        p, S, _ = evaluate(theta)
        parts = [whiten(np.log(S[mask]) - target)]
        if fam.penalty is not None:
            parts.append(fam.penalty(p)[0])
        parts.extend(extra_values(theta))
        return np.concatenate(parts)

    def jacobian(theta):
        p, S, J = evaluate(theta)
        rows = [whiten(J[mask])]
        if fam.penalty is not None:
            rows.append(fam.penalty(p)[1])
        if extras:
            base = np.concatenate(extra_values(theta))
            fd = np.empty((len(base), fam.n))
            for k in range(fam.n):
                step = np.zeros(fam.n)
                step[k] = fd_step * max(1.0, abs(theta[k]))
                up = theta + step
                if up[k] > hi[k]:
                    up = theta - step
                    fd[:, k] = (base - np.concatenate(extra_values(up))) / step[k]
                else:
                    fd[:, k] = (np.concatenate(extra_values(up)) - base) / step[k]
            rows.append(fd)
        return np.vstack(rows)

    fitted = least_squares(residual, theta0, jac=jacobian if jac == "analytic" else "2-point",
                           bounds=(lo, hi), max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    p, S, J = evaluate(fitted.x)
    Jw = jacobian(fitted.x)
    singular = np.linalg.svd(fitted.jac, compute_uv=False)
    try:
        cov_theta = np.linalg.inv(Jw.T @ Jw)
    except np.linalg.LinAlgError:
        cov_theta = np.full((fam.n, fam.n), np.nan)
    nshell = int(mask.sum())
    chi2_profile = float(np.sum(whiten(np.log(S[mask]) - target)**2))
    out = {"parameters": p.tolist(), "theta": fitted.x.copy(), "names": fam.names, "family": fam.name,
           "success": bool(fitted.success),
           "bound_distance": float(np.min(np.minimum(fitted.x - lo, hi - fitted.x))),
           "jacobian_condition": float(singular[0] / max(singular[-1], 1e-30)),
           "rms_percent": float(np.sqrt(np.mean((S[mask] / mean[mask] - 1)**2)) * 100),
           "nfev": int(fitted.nfev), "chi2": float(np.sum(fitted.fun**2)), "chi2_profile": chi2_profile,
           "n_shells": nshell, "n_params": fam.n, "ndof": nshell - fam.n,
           "chi2_per_dof": chi2_profile / max(nshell - fam.n, 1), "cov_theta": cov_theta,
           "jacobian_lnS": J[mask], "mask": mask.copy(), "model": S, "metric_mode": mode,
           "objective": objective or mode, "status": int(fitted.status)}
    out["accepted"] = accepted(out)
    return out


def objective_d(family, mean, weights, mask, **kw):
    """(d) chi2 = sum_s w_s (ln model - ln mean)_s^2 for an arbitrary weight vector."""
    return fit_general(family, mean, mask, weights=weights, objective="d: per-shell weights", **kw)


def objective_e(family, mean, metric, mask, **kw):
    """(e) chi2 = r^T M r for a PSD metric (e.g. ``observable_metric``)."""
    return fit_general(family, mean, mask, metric=metric, objective="e: quadratic metric", **kw)


def normalise_metric(K: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale K so that trace(K) = trace(C^-1): lambda = 1 then gives the importance term the
    same total weight as the statistical term.  Returns (K scaled, scale factor)."""
    Ci = np.linalg.inv(cov)
    scale = float(np.trace(Ci) / np.trace(K)) if np.trace(K) > 0 else 1.0
    return K * scale, scale


def objective_f(family, mean, cov, K, mask, lam: float, normalise: str | None = "trace", **kw):
    """(f) hybrid chi2 = r^T C^-1 r + lam r^T K r (K trace-normalised to C^-1 by default)."""
    mask = np.asarray(mask, bool) & np.isfinite(mean) & (np.asarray(mean) > 0)
    n_all = len(mean)
    C = _restrict(cov, mask, n_all)
    Kf = _restrict(K, mask, n_all)
    scale = 1.0
    if normalise == "trace":
        Kf, scale = normalise_metric(Kf, C)
    metric = np.linalg.inv(C) + lam * Kf
    fit = fit_general(family, mean, mask, metric=metric, objective=f"f: hybrid lambda={lam:g}", **kw)
    if fit is not None:
        fit["lambda"] = float(lam)
        fit["K_scale"] = scale
        r = np.log(fit["model"][fit["mask"]]) - np.log(np.asarray(mean)[fit["mask"]])
        fit["chi2_stat_part"] = float(r @ np.linalg.solve(C, r))
        fit["chi2_importance_part"] = float(lam * r @ Kf @ r)
    return fit


def hybrid_scan(family, mean, cov, K, mask, lams=(0.1, 1.0, 10.0, 100.0), normalise="trace", **kw) -> dict:
    """(f) for several lambda; each fit warm-starts from the previous one."""
    out = {}
    start = kw.pop("start", None)
    for lam in lams:
        fit = objective_f(family, mean, cov, K, mask, lam, normalise=normalise, start=start, **kw)
        out[float(lam)] = fit
        if fit is not None and fit["accepted"]:
            start = fit["parameters"]
    return out


def objective_g(family, mean, mask, extra_residuals, **kw):
    """(g) any base weighting (sigma/cov/weights/metric keyword) + appended extra residuals."""
    return fit_general(family, mean, mask, extra_residuals=extra_residuals, objective="g: extra residuals", **kw)


def observable_metric(jacobians_dO_dF, observables, lambdas, profile) -> np.ndarray:
    """K = sum_O lambda_O g_O g_O^T / O^2 with g_O,s = F_s dO/dF_s = dO/d ln F_s.

    ``jacobians_dO_dF`` [n_obs, shell] are dO/dF_s; ``profile`` F_s converts them to the
    ln F residual space used by every objective here.
    """
    G = np.asarray(jacobians_dO_dF, float) * np.asarray(profile, float)[None, :]
    O = np.asarray(observables, float)
    lam = np.asarray(lambdas, float)
    return np.einsum("o,os,ot->st", lam / O**2, G, G)


def objective_a_general(family, mean, std, mask, floor: float = .05, **kw):
    """Objective (a) (sigma = max(SD/mean, 5%)) for any family."""
    mean = np.asarray(mean, float)
    sigma = np.maximum(np.divide(std, mean, out=np.ones_like(std), where=mean > 0), floor)
    return fit_general(family, mean, mask, sigma=sigma, objective="a: diagonal SD/mean, 5% floor", **kw)


def objective_c_general(family, mean, cov_stat, mask, sigma_d="solve", tol: float = 1e-5, max_iter: int = 60, **kw):
    """Objective (c) (GLS + sigma_d to chi2/nu = 1) for any family (same iteration as objective_c)."""
    mean = np.asarray(mean, float)
    mask = np.asarray(mask, bool) & np.isfinite(mean) & (mean > 0)
    C = _restrict(cov_stat, mask, len(mean))
    if sigma_d != "solve":
        fit = fit_general(family, mean, mask, cov=C + float(sigma_d)**2 * np.eye(len(C)), **kw)
        if fit is not None:
            fit["sigma_d"] = float(sigma_d)
        return fit
    start = kw.pop("start", None)
    fit = fit_general(family, mean, mask, cov=C, start=start, objective="c: GLS + sigma_d", **kw)
    if fit is None:
        return None
    nu = fit["ndof"]
    s2 = 0.0
    if nu > 0 and fit["chi2_profile"] > nu:
        values, vectors = eigh(C)
        values = np.maximum(values, 0.0)
        for _ in range(max_iter):
            r = np.log(fit["model"][mask]) - np.log(mean[mask])
            u2 = (vectors.T @ r)**2

            def g(s):
                return float(np.sum(u2 / (values + s)) - nu)

            if g(0.0) <= 0:
                s_new = 0.0
            else:
                top = max(4 * s2, float(np.mean(r**2)), 1e-12)
                while g(top) > 0:
                    top *= 4
                s_new = brentq(g, 0.0, top, xtol=1e-16, rtol=1e-12)
            fit = fit_general(family, mean, mask, cov=C + s_new * np.eye(len(C)), start=fit["parameters"],
                              objective="c: GLS + sigma_d", **kw)
            converged = abs(np.sqrt(s_new) - np.sqrt(s2)) <= tol * max(np.sqrt(s_new), 1e-8)
            s2 = s_new
            if converged:
                break
    fit["sigma_d"] = float(np.sqrt(s2))
    fit["cov_total"] = C + s2 * np.eye(len(C))
    return fit


# ----------------------------------------------------------------------------- residual diagnostics
def shell_residuals(family, params, mean, edges, cut: float | None = None) -> dict:
    """Per-shell model residuals on every shell: relative S/m - 1 and ln S - ln m.

    ``cut`` truncates the model at that radius (``families.shell_model_truncated``);
    shells beyond the cut then have relative residual -1.
    """
    mean = np.asarray(mean, float)
    S = (FAM.shell_model(family, params, edges) if cut is None
         else FAM.shell_model_truncated(family, params, edges, cut))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mean > 0, S / mean - 1, np.nan)
        ln = np.where((mean > 0) & (S > 0), np.log(S) - np.log(mean), np.nan)
    return {"relative": rel, "ln": ln, "model": S}


def weighted_rms(values, weights=None, mask=None) -> float:
    """sqrt(sum w v^2 / sum w) over finite entries of ``mask`` (unweighted if weights None)."""
    v = np.asarray(values, float)
    use = np.isfinite(v) if mask is None else (np.asarray(mask, bool) & np.isfinite(v))
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    use &= w > 0
    if not use.any():
        return float("nan")
    return float(np.sqrt(np.sum(w[use] * v[use]**2) / np.sum(w[use])))


REGIONS = (("inner_lt_1R", 0.0, 1.0), ("1_to_2R", 1.0, 2.0), ("outer_2_to_4R", 2.0, 4.0))


def region_rms(relative, edges, mask, regions=REGIONS) -> dict:
    """Unweighted RMS (percent) of relative residuals per radial region (shells fully inside)."""
    edges = np.asarray(edges, float)
    out = {}
    for name, a, b in regions:
        inside = np.asarray(mask, bool) & (edges[:-1] >= a - 1e-12) & (edges[1:] <= b + 1e-12)
        out[name] = 100 * weighted_rms(relative, None, inside)
    return out


def fit_cut_grid(family, mean, base, edges, fit_radii, cut_radii, fitter, importance=None,
                 eval_max: float = 4.0, regions=REGIONS) -> dict:
    """Fit to each R_fit, truncate at each R_cut, evaluate on all resolved shells to ``eval_max``.

    ``fitter(mask) -> fit dict`` performs the fit on the given shell mask (any objective);
    ``importance`` [shell] weights the RMS used to choose (R_fit, R_cut).  ``cut_radii`` may
    contain None (untruncated).  Returns RMS tables [R_fit, R_cut] (percent), per-region
    tables, the fits and the minimising (R_fit, R_cut).
    """
    edges = np.asarray(edges, float)
    mean = np.asarray(mean, float)
    evaluate = np.asarray(base, bool) & (edges[1:] <= eval_max + 1e-12) & np.isfinite(mean) & (mean > 0)
    weighted = np.full((len(fit_radii), len(cut_radii)), np.nan)
    per_region = {name: np.full((len(fit_radii), len(cut_radii)), np.nan) for name, _, _ in regions}
    fits = {}
    for i, rf in enumerate(fit_radii):
        mask = np.asarray(base, bool) & (edges[1:] <= rf + 1e-12)
        fit = fitter(mask)
        fits[rf] = fit
        if fit is None:
            continue
        for j, rc in enumerate(cut_radii):
            res = shell_residuals(family, fit["parameters"], mean, edges, cut=rc)
            weighted[i, j] = 100 * weighted_rms(res["relative"], importance, evaluate)
            for name, value in region_rms(res["relative"], edges, evaluate, regions).items():
                per_region[name][i, j] = value
    best = None
    if np.isfinite(weighted).any():
        i, j = np.unravel_index(np.nanargmin(weighted), weighted.shape)
        best = {"R_fit": fit_radii[i], "R_cut": cut_radii[j], "weighted_rms_percent": float(weighted[i, j])}
    return {"weighted_rms_percent": weighted, "region_rms_percent": per_region, "fits": fits, "best": best,
            "fit_radii": list(fit_radii), "cut_radii": list(cut_radii), "evaluation_shells": int(evaluate.sum())}
