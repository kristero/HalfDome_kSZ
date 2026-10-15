"""Physical ingredients of Test G: linear 2-halo term, splashback scale, outer temperatures, p_keep, scatter moments.

2-halo term (parameter-free).  For a halo of mass M200c and radius R200c (z = 0, physical = comoving Mpc):
    n_2h(x) = Ebar [1 + b(M) xi_lin(x R200c)],   Ebar = KAPPA_E Omega_m   (E units)
with xi_lin from CLASS P_lin(k, z = 0) and b(M) Tinker et al. (2010) for Delta = 200/Omega_m (inputs/
xi_bias_z0.npz, made by inputs/make_xi_bias.py).  A bin's model uses the inclusion-weighted mean of
b(M_h) xi_lin(x R_h) over its haloes, evaluated at the shell-quadrature nodes (the stacked mean of the per-halo
term, exact for the stack in x = r/R200c).

p_keep (ownership bookkeeping, per field): smoothed A2-ii / all ratio of the stacked profiles,
    p_keep(x, M) = 1 - f(M) / [1 + (x / x_h(M))^-n],  f = f0 + f1 v,  ln x_h = l0 + l1 v,  v = log10 M - 14.5,
fitted to ln(mean_A2ii / mean_all) on the shells beyond 1.34 R200c (volume-weighted shell radius), weights N_eff.

Scatter moments (per calibrated bin, from per-halo integrals, inclusion-weighted):
  S1 (Test F): M = <X^2>/<X>^2 for X = Y(<4R), N_e(<4R) and <Y N>/(<Y><N>)        (F's lognormal_4R)
  S2: the 4 x 4 matrix M_ij = <X_i X_j>/(<X_i><X_j>) of X = (Y_in, Y_out, N_in, N_out),
      in = inside 1 R200c, out = 1.5 R200c .. R_cut; ln-amplitude covariance C = ln M (lognormal).
"""
from __future__ import annotations

import g_common as GC
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, least_squares

FAM = GC.FF.FAM


# ============================================================================ xi_lin, bias
class LinearHalo:
    def __init__(self, path=GC.XI_BIAS):
        z = np.load(path, allow_pickle=False)
        self.r = np.asarray(z["r_mpc"], float)
        self.xi_lin = np.asarray(z["xi_lin"], float)
        self.xi_nl = np.asarray(z["xi_nl"], float)
        self.mass = np.asarray(z["m200c_msun"], float)
        self.bias = np.asarray(z["bias_tinker10"], float)
        self.nu = np.asarray(z["nu"], float)
        self.omega_m = float(z["omega_m"])
        self.meta = str(z["meta"])
        self._lr, self._lxi = np.log(self.r), np.log(self.xi_lin)
        self._lm = np.log(self.mass)

    def xi(self, r_mpc):
        """xi_lin(r): log-log interpolation; held at xi(0.05 Mpc) below the table (inside the core, where the
        bound component dominates by >100x) and log-log extrapolated beyond 60 Mpc (never reached: 12 R200c of
        the largest halo is 34 Mpc)."""
        lr = np.log(np.maximum(np.asarray(r_mpc, float), self.r[0]))
        return np.exp(np.interp(lr, self._lr, self._lxi))

    def b(self, m_msun):
        return np.interp(np.log(np.asarray(m_msun, float)), self._lm, self.bias)

    def nu_of(self, m_msun):
        return np.interp(np.log(np.asarray(m_msun, float)), self._lm, self.nu)

    def spot_values(self):
        out = {"bias": {}, "xi_lin": {}}
        for m in (1e13, 1e14, 1e15):
            out["bias"][f"{m:.0e}"] = float(self.b(m))
        for r in (1.0, 2.0, 5.0, 10.0):
            out["xi_lin"][f"{r:g}"] = float(self.xi(r))
        return out


def quad_nodes(edges):
    """32-point Gauss-Legendre nodes and volume weights per shell (exactly families.shell_grid)."""
    return FAM.shell_grid(np.asarray(edges, float))


def two_halo_template(lh: LinearHalo, m200, r200, weights, x_nodes):
    """<b(M) xi_lin(x R)>_w at the nodes x [S, n] (inclusion-weighted over the haloes given)."""
    w = np.asarray(weights, float) / np.sum(weights)
    out = np.zeros_like(x_nodes, dtype=float)
    bh = lh.b(m200)
    for h in range(len(w)):
        out += w[h] * bh[h] * lh.xi(x_nodes * r200[h])
    return out


def bin_templates(lh, pr, rows_by_bin=None, edges=None):
    """Per calibrated bin: 2-halo template at the basis-shell nodes [N_CAL, S, n], the inclusion-weighted mean
    bias, mean log10 M200c and mean R200c."""
    edges = GC.BASIS_EDGES if edges is None else edges
    x, _ = quad_nodes(edges)
    T = np.zeros((GC.N_CAL,) + x.shape)
    bmean, logm, rmean = np.zeros(GC.N_CAL), np.zeros(GC.N_CAL), np.zeros(GC.N_CAL)
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b) if rows_by_bin is None else np.asarray(rows_by_bin[c])
        w = pr["weight"][rows]
        m, r = pr["m200c_msun"][rows], pr["r200c_mpc"][rows]
        T[c] = two_halo_template(lh, m, r, w, x)
        ww = w / w.sum()
        bmean[c] = ww @ lh.b(m)
        logm[c] = ww @ np.log10(m)
        rmean[c] = ww @ r
    return {"T2h": T, "b": bmean, "logm": logm, "r200": rmean}


# ============================================================================ splashback scale
def _mu_nfw(y):
    return np.log1p(y) - y / (1 + y)


def r200m_over_r200c(c200c, omega_m=GC.OMEGA_M):
    """NFW: R200m / R200c from the halo's concentration (mean density inside R200m = 200 Omega_m rho_c)."""
    out = np.full(len(c200c), np.nan)
    for i, c in enumerate(np.asarray(c200c, float)):
        if not np.isfinite(c) or c <= 0:
            continue
        f = (lambda x, c=c: _mu_nfw(c * x) / _mu_nfw(c) / x**3 - omega_m)
        out[i] = brentq(f, 1.0, 10.0)
    return out


# ============================================================================ p_keep (ownership bookkeeping)
PKEEP_X0 = 1.34


def shell_radius(edges=GC.F_EDGES):
    """Volume-weighted mean radius of each shell."""
    e = np.asarray(edges, float)
    return 0.75 * (e[1:]**4 - e[:-1]**4) / (e[1:]**3 - e[:-1]**3)


def pkeep_value(params, x, v):
    """p_keep(x, v) for parameters (f0, f1, l0, l1, n); x any shape, v scalar."""
    f0, f1, l0, l1, n = params
    xh = np.exp(l0 + l1 * v)
    f = np.clip(f0 + f1 * v, 0.0, 1.0)
    return 1.0 - f / (1.0 + (np.asarray(x, float) / xh)**(-n))


def fit_pkeep(mean_all, mean_a2ii, logm, neff):
    """Fit the logistic p_keep surface to ln(mean_A2ii / mean_all) beyond PKEEP_X0 (rows: calibrated bins)."""
    xc = shell_radius()
    sel = xc > PKEEP_X0
    v = np.asarray(logm, float) - 14.5
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_all[:, sel] > 0, mean_a2ii[:, sel] / mean_all[:, sel], np.nan)
    y = np.log(np.clip(ratio, 1e-6, None))
    good = np.isfinite(y)
    wts = np.sqrt(np.asarray(neff, float))[:, None] * np.ones_like(y)

    def res(p):
        m = np.log(np.clip(pkeep_value(p, xc[sel][None, :], v[:, None]), 1e-12, None))
        return ((m - y) * wts)[good]

    lo = [0.0, -1.0, np.log(1.2), -2.0, 1.0]            # (f0, f1, l0, l1, n)
    hi = [1.0, 1.0, np.log(30.0), 2.0, 40.0]
    best = None
    for start in ([0.7, 0.0, np.log(3.4), 0.0, 6.0], [0.5, 0.0, np.log(4.0), 0.0, 4.0], [0.9, 0.0, np.log(3.0), 0.0, 10.0]):
        sol = least_squares(res, start, bounds=(lo, hi), ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
        if best is None or sol.cost < best.cost:
            best = sol
    p = best.x
    fit = pkeep_value(p, xc[None, :], v[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        measured = np.where(mean_all > 0, mean_a2ii / np.where(mean_all > 0, mean_all, 1.0), np.nan)
        resid = fit[:, sel] / measured[:, sel] - 1
    rms = np.sqrt(np.nanmean(resid**2, axis=1))
    return {"params": p, "success": bool(best.success), "cost": float(best.cost), "rms_rel_per_bin": rms,
            "measured_ratio": measured, "smoothed_at_shells": fit}


def pkeep_nodes(params, logm, edges=GC.BASIS_EDGES):
    """p_keep at the quadrature nodes of ``edges`` per calibrated bin: [N_CAL, S, n]."""
    x, _ = quad_nodes(edges)
    v = np.asarray(logm, float) - 14.5
    return np.stack([pkeep_value(params, x, vv) for vv in v])


# ============================================================================ scatter moments
def per_halo_integrals(values, lo, hi, edges=GC.F_EDGES):
    """sum_s F_s V_s over shells inside [lo, hi] (edges), dimensionless (x^3 units)."""
    e = np.asarray(edges, float)
    inside = (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)
    vol = 4 * np.pi / 3 * np.diff(e**3)
    v = np.nan_to_num(np.asarray(values, float), nan=0.0)
    return v[:, inside] @ vol[inside]


def moment_matrix(X, w):
    """M_ij = <X_i X_j>_w / (<X_i>_w <X_j>_w) for the columns of X [halo, k]."""
    w = np.asarray(w, float) / np.sum(w)
    m = w @ X
    second = np.einsum("h,hi,hj->ij", w, X, X)
    return second / np.outer(m, m)


def scatter_tables(pr, sel, rows_by_bin=None, cuts=None):
    """S1 [N_CAL, 3] (ln<Y^2>/<Y>^2, ln<N^2>/<N>^2, <YN>/(<Y><N>); 4R aperture) and S2 {R_cut: [N_CAL, 4, 4]}
    for the gas selection ``sel`` ('main' or 'a2ii')."""
    cuts = GC.S2_CUTS if cuts is None else cuts
    P, E = pr[f"P_{sel}"], pr[f"E_{sel}"]
    s1 = np.zeros((GC.N_CAL, 3))
    s2 = {rc: np.zeros((GC.N_CAL, 4, 4)) for rc in cuts}
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b) if rows_by_bin is None else np.asarray(rows_by_bin[c])
        w = pr["weight"][rows]
        Y4 = per_halo_integrals(P[rows], 0.0, 4.0)
        N4 = per_halo_integrals(E[rows], 0.0, 4.0)
        M = moment_matrix(np.c_[Y4, N4], w)
        s1[c] = (np.log(M[0, 0]), np.log(M[1, 1]), M[0, 1])
        Yin = per_halo_integrals(P[rows], 0.0, 1.0)
        Nin = per_halo_integrals(E[rows], 0.0, 1.0)
        for rc in cuts:
            Yout = per_halo_integrals(P[rows], 1.5, rc)
            Nout = per_halo_integrals(E[rows], 1.5, rc)
            s2[rc][c] = moment_matrix(np.c_[Yin, Yout, Nin, Nout], w)
    return s1, s2


def s2_summary(M):
    """sigma_in, sigma_out, rho (ln-amplitude) for P (indices 0, 1) and E (2, 3) from a 4 x 4 moment matrix."""
    C = np.log(M)
    out = {}
    for f, (i, o) in (("P", (0, 1)), ("E", (2, 3))):
        si, so = np.sqrt(max(C[i, i], 0)), np.sqrt(max(C[o, o], 0))
        out[f] = {"sigma_in": si, "sigma_out": so, "rho": C[i, o] / (si * so) if si > 0 and so > 0 else np.nan}
    out["rho_P_E_in"] = C[0, 2] / np.sqrt(C[0, 0] * C[2, 2]) if C[0, 0] > 0 and C[2, 2] > 0 else np.nan
    out["rho_P_E_out"] = C[1, 3] / np.sqrt(C[1, 1] * C[3, 3]) if C[1, 1] > 0 and C[3, 3] > 0 else np.nan
    return out


def gh1(sigma2, n=GC.GH1):
    """1-D lognormal amplitude nodes (unit mean, <A^2> = exp(sigma^2)) and weights (Test F's gauss_hermite)."""
    t, w = np.polynomial.hermite.hermgauss(n)
    s = np.sqrt(max(sigma2, 0.0))
    return np.exp(s * np.sqrt(2) * t - sigma2 / 2), w / np.sqrt(np.pi)


def gh2(C2, n=GC.GH2):
    """2-D correlated lognormal amplitudes (A_in, A_out) with ln-covariance C2 [2, 2], unit means:
    nodes [n*n, 2] and weights [n*n] (tensor Gauss-Hermite, Cholesky-correlated)."""
    t, w = np.polynomial.hermite.hermgauss(n)
    C2 = np.asarray(C2, float)
    C2 = (C2 + C2.T) / 2
    ev, U = np.linalg.eigh(C2)
    L = U @ np.diag(np.sqrt(np.clip(ev, 0, None)))          # C2 = L L^T (PSD-safe)
    z = np.sqrt(2) * np.stack(np.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
    wz = np.outer(w, w).ravel() / np.pi
    lnA = z @ L.T - 0.5 * np.diag(C2)[None, :]
    return np.exp(lnA), wz


# ============================================================================ normalisation and temperatures
BANDS = ((0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0))


def band_mask(lo, hi, edges=GC.F_EDGES):
    e = np.asarray(edges, float)
    return (e[:-1] >= lo - 1e-9) & (e[1:] <= hi + 1e-9)


def normalisation_checks(pr, bprof, bhalo):
    """Stacked (inclusion-weighted, volume-integrated per band) ratios per calibrated bin:
    E/G (electrons per gas density, -> KAPPA_E), and P / [(E/200)(T_M/T200)] (-> KAPPA_P x T_e/T_M)."""
    rows_all = pr["halo_rows"]
    G = bprof["G_geo"][rows_all]
    TM = bprof["T_M"][rows_all]
    T200 = bhalo["T200_K"][rows_all]
    P, E = pr["P_main"], pr["E_main"]
    vol = np.diff(GC.F_EDGES**3)
    out = {"E_over_G": np.zeros((GC.N_CAL, len(BANDS))), "P_over_ET": np.zeros((GC.N_CAL, len(BANDS)))}
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        for k, (lo, hi) in enumerate(BANDS):
            sh = band_mask(lo, hi)
            Eb = np.nansum(E[rows][:, sh] * vol[sh], axis=1)
            Gb = np.nansum(G[rows][:, sh] * vol[sh], axis=1)
            Pb = np.nansum(P[rows][:, sh] * vol[sh], axis=1)
            ETb = np.nansum(E[rows][:, sh] * TM[rows][:, sh] * vol[sh], axis=1) / T200[rows] / 200.0
            out["E_over_G"][c, k] = (w @ Eb) / (w @ Gb)
            out["P_over_ET"][c, k] = (w @ Pb) / (w @ ETb)
    return out


def temperature_profiles(pr, bprof, bhalo, sel="main"):
    """Per calibrated bin and band: mass-weighted T_M/T200 (stack of sum m T / sum m, via the gas density as
    the mass weight), electron-weighted T_e/T200 (E-weighted T_M), and the pressure-implied
    T_P/T200 = (P/P200) / (KAPPA_P E/200)."""
    rows_all = pr["halo_rows"]
    G = bprof["G_geo"][rows_all]
    TM = bprof["T_M"][rows_all] / bhalo["T200_K"][rows_all][:, None]
    P, E = pr[f"P_{sel}"], pr[f"E_{sel}"]
    vol = np.diff(GC.F_EDGES**3)
    out = {k: np.zeros((GC.N_CAL, len(BANDS))) for k in ("T_M", "T_e", "T_P")}
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        for k, (lo, hi) in enumerate(BANDS):
            sh = band_mask(lo, hi)
            g = np.nan_to_num(G[rows][:, sh]) * vol[sh]
            e = np.nan_to_num(E[rows][:, sh]) * vol[sh]
            t = np.nan_to_num(TM[rows][:, sh])
            p = np.nan_to_num(P[rows][:, sh]) * vol[sh]
            out["T_M"][c, k] = (w @ (g * t).sum(1)) / (w @ g.sum(1))
            out["T_e"][c, k] = (w @ (e * t).sum(1)) / (w @ e.sum(1))
            out["T_P"][c, k] = (w @ p.sum(1)) / (GC.KAPPA_P * (w @ e.sum(1)) / 200.0)
    return out


def temperature_shells(pr, bprof, bhalo, sel="main"):
    """Electron-weighted T/T200 per calibrated bin and F shell (stacked), for the tau(x) comparison."""
    rows_all = pr["halo_rows"]
    TM = bprof["T_M"][rows_all] / bhalo["T200_K"][rows_all][:, None]
    E = pr[f"E_{sel}"]
    out = np.full((GC.N_CAL, GC.N_F), np.nan)
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][rows] / pr["weight"][rows].sum()
        e = np.nan_to_num(E[rows])
        t = np.nan_to_num(TM[rows])
        den = w @ e
        out[c] = np.where(den > 0, (w @ (e * t)) / np.where(den > 0, den, 1), np.nan)
    return out


def two_halo_ratios(pr, templates, ebar, sel="main"):
    """Measured stacked E / predicted Ebar [1 + <b xi>] (volume-integrated per band) per calibrated bin."""
    x, wq = quad_nodes(GC.F_EDGES)
    T35 = templates["T2h"][:, :GC.N_F]
    pred_shell = ebar * (1 + np.sum(T35 * wq[None], axis=2))       # [N_CAL, 35] shell averages
    vol = np.diff(GC.F_EDGES**3)
    out = np.zeros((GC.N_CAL, len(BANDS)))
    for c, b in enumerate(GC.CAL_BINS):
        mean = pr[f"meanE_{sel}"][b]
        for k, (lo, hi) in enumerate(BANDS):
            sh = band_mask(lo, hi)
            out[c, k] = (mean[sh] @ vol[sh]) / (pred_shell[c, sh] @ vol[sh])
    return out, pred_shell
