"""B12 / B16 fits on the fine grid with Test F's / G's fitter (read-only), shared by H4 and H6.

Forms (x = r/R200c; analysis.fitting / radii234 models, identical to Test G's P0 / E0 cores):
  B12 pressure   P(x) = 0.5176 P0 q^-0.3 (1 + q)^-beta,              q = x / x_c       (parameters P0, x_c, beta)
  B16 electrons  E(x) = rho0 q^-0.2 (1 + q^alpha)^-((beta - 0.2)/alpha), q = x / 0.5   (parameters rho0, alpha, beta)
Objective (a), Test F's current diagonal: on the bin's resolved shells with x_hi <= R_fit (base mask of the bin on
the fine grid, finite positive mean), residual r_s = (ln S_s(theta) - ln m_s) / sigma_s with S_s the exact shell
average of the model, m_s the inclusion-weighted mean stack and sigma_s = max(SD_s / m_s, 5%) (SD = the
reliability-weighted halo-to-halo standard deviation); least squares in theta = ln(parameters) within Test F's
bounds (f_fit.fit_objective -> analysis.fitting.objective_a, analytic Jacobian).
"""
from __future__ import annotations

import h_common as HC
import numpy as np

FC, FF = HC.FC, HC.FF
FAM = FF.FAM
AF = FC.F                              # analysis.fitting
FAMILY = {"P": "b12", "E": "b16"}
PARAMS = {"P": ("P0", "x_c", "beta"), "E": ("rho0", "alpha", "beta")}


def bin_data(pr, field, sel, c, rows=None):
    """Test F's BinData on the fine grid (inclusion-weighted mean, reliability SD, base mask)."""
    return FF.BinData(pr, field, sel, c, rows)


def fit(bd, r_fit, start=None):
    """Objective (a) fit of one bin; returns Test F's fit dict (parameters, cov_log_parameters, jacobian_lnS on the
    mask, sigma, model on the 67 fine shells, chi2, rms_percent, accepted, ...) or None."""
    return FF.fit_objective(bd, "a", r_fit, start=start)


def model_shells(field, params, cut=4.0, edges=None):
    """Shell averages of the fitted form on the 75 basis shells, hard-cut at ``cut`` (None: no cut)."""
    edges = HC.FINE if edges is None else edges
    fam = FAM.get(FAMILY[field])
    if cut is None:
        return FAM.shell_model(fam, np.asarray(params, float), edges)
    return FAM.shell_model_truncated(fam, np.asarray(params, float), edges, float(cut))


def dlnS_dlntheta(field, params, edges=None):
    """d ln S_s / d ln theta_k [75, 3] of the exact shell averages (analysis.fitting.shell_model_jacobian)."""
    edges = HC.FINE if edges is None else edges
    _, J = AF.shell_model_jacobian(FAMILY[field], np.asarray(params, float), edges)
    return J


def influence_full(fitd, n=HC.N_X):
    """Fit influence H = (J^T W J)^-1 J^T W [3, n] (d ln theta / d ln m_s) of an objective-(a) fit, zero outside the
    fit mask; W = diag(sigma^-2) (analysis.fitting.influence)."""
    inf = AF.influence(fitd["jacobian_lnS"], fitd["sigma"])
    H = np.zeros((3, n))
    H[:, np.flatnonzero(fitd["mask"])] = inf["H"]
    return H, inf["n_eff"]


def powerlaw_shell_average(A, n, lo, hi):
    """Exact shell average of A x^n over [lo, hi] (volume-weighted): 3 A (hi^(n+3) - lo^(n+3)) / ((n+3)(hi^3 - lo^3))."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if abs(n + 3) < 1e-9:
        return 3 * A * np.log(hi / lo) / (hi**3 - lo**3)
    return 3 * A * (hi**(n + 3) - lo**(n + 3)) / ((n + 3) * (hi**3 - lo**3))


def loglinear_extension(mean, x_from, edges_fit=HC.X200, edges_out=HC.FINE):
    """Log-linear (power-law) fit ln m_s = a + n ln x_s over the fine shells with x_lo >= x_from (x_s = volume-weighted
    shell radius, unweighted least squares), extrapolated as exact shell averages onto every basis shell beyond 4R.
    mean [nb, 67] -> ext [nb, 75] (zero inside 4R), slopes [nb], amplitudes [nb]."""
    mid = HC.mids(edges_fit)
    use = np.asarray(edges_fit)[:-1] >= x_from - 1e-9
    out = np.zeros((len(mean), len(edges_out) - 1))
    slopes, amps = np.zeros(len(mean)), np.zeros(len(mean))
    lo, hi = np.asarray(edges_out)[:-1], np.asarray(edges_out)[1:]
    beyond = lo >= 4.0 - 1e-9
    for c, m in enumerate(mean):
        ok = use & (m > 0)
        n, a = np.polyfit(np.log(mid[ok]), np.log(m[ok]), 1)
        slopes[c], amps[c] = n, np.exp(a)
        out[c, beyond] = powerlaw_shell_average(np.exp(a), n, lo[beyond], hi[beyond])
    return out, slopes, amps
