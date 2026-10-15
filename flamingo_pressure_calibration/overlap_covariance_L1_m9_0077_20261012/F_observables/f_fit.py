"""Fitting toolkit of Test F: per-bin data, objectives (a)-(f), flexible families, truncation, direct
observable fits, partial pooling and scatter models.  Uses analysis.fitting / analysis.families unchanged.

Every model is returned as shell averages on the 42 basis shells (35 F shells to 4R + 7 extension
shells to 12R), per calibrated bin: arrays [10, 42] for pressure (Pe/P200) and electrons.
"""
from __future__ import annotations

import f_common as FC
import numpy as np
from scipy.optimize import brentq, least_squares

F, FAM, COV, S = FC.F, FC.FAM, FC.COV, FC.S
FIELD_FAMILY = {"P": "b12", "E": "b16"}
PARAM_NAMES = {"b12": ("P0", "xc", "beta"), "b16": ("rho0", "alpha", "beta")}
BOOT_DRAWS = FC.BOOT_DRAWS
LAMBDA_DRAWS = FC.LAMBDA_DRAWS
LAMBDAS = FC.LAMBDAS
SPLINE_KNOTS = (0.1, 0.4, 1.2, 3.0)
SPLINE_PRIOR = 0.3


# ============================================================================ electron analogues of the
# library's two-component and spline families (same constructions as analysis.families, B16 core)
def b16_env_family(pivot: float = 2.0) -> FAM.Family:
    def shape(x, p):
        x = np.asarray(x, float)
        return FAM.M.shape("b16", x, p[:3]) + p[3] * (x / pivot)**(-p[4])

    def dln(x, p):
        x = np.asarray(x, float)
        core = FAM.M.shape("b16", x, p[:3])
        env = p[3] * (x / pivot)**(-p[4])
        total = core + env
        out = np.empty((5,) + x.shape)
        out[:3] = FAM._b16_dln(x, p) * (core / total)
        out[3] = env / total
        out[4] = -np.log(x / pivot) * env / total
        return out

    lo, hi = FAM.M.BOUNDS["b16"]
    return FAM.Family("b16env", ("rho0", "alpha", "beta", "A_env", "s_env"), list(lo) + [1e-6, -2.0],
                      list(hi) + [1e5, 10.], [True] * 4 + [False], list(FAM.P.B16[:3]) + [1.0, 1.0], shape, dln,
                      info={"pivot_x": pivot})


def b16_spline_family(knots_x=SPLINE_KNOTS, prior_sigma: float = SPLINE_PRIOR, knot_bounds=(-0.6, 3.0)) -> FAM.Family:
    from scipy.interpolate import CubicSpline
    knots_u = np.log(np.asarray(knots_x, float))
    K = len(knots_u)
    bases = [CubicSpline(knots_u, np.eye(K)[k], bc_type="natural") for k in range(K)]

    def basis(x):
        u = np.clip(np.log(np.asarray(x, float)), knots_u[0], knots_u[-1])
        return np.stack([b(u) for b in bases])

    def shape(x, p):
        return FAM.M.shape("b16", x, p[:3]) * np.maximum(1 + np.tensordot(p[3:], basis(x), axes=1), 1e-6)

    def dln(x, p):
        Bx = basis(x)
        corr = np.maximum(1 + np.tensordot(p[3:], Bx, axes=1), 1e-6)
        out = np.empty((3 + K,) + np.shape(x))
        out[:3] = FAM._b16_dln(x, p)
        out[3:] = Bx / corr
        return out

    def penalty(p):
        c = np.asarray(p[3:], float)
        jac = np.zeros((K, 3 + K))
        jac[:, 3:] = np.eye(K) / prior_sigma
        return c / prior_sigma, jac

    lo, hi = FAM.M.BOUNDS["b16"]
    return FAM.Family("b16spline", ("rho0", "alpha", "beta") + tuple(f"c{k}" for k in range(K)),
                      list(lo) + [knot_bounds[0]] * K, list(hi) + [knot_bounds[1]] * K,
                      [True] * 3 + [False] * K, list(FAM.P.B16[:3]) + [0.0] * K, shape, dln, penalty,
                      info={"knots_x": list(knots_x), "prior_sigma": prior_sigma})


B12SPLINE = FAM.b12_spline_family(SPLINE_KNOTS, SPLINE_PRIOR)
B16ENV = b16_env_family()
B16SPLINE = b16_spline_family()
FLEX = {"P": {"gnfw5": FAM.GNFW5, "env": FAM.B12ENV, "spline": B12SPLINE},
        "E": {"gnfw5": FAM.GNFW5, "env": B16ENV, "spline": B16SPLINE}}


def family_obj(name):
    if isinstance(name, FAM.Family):
        return name
    return FAM.get(name)


# ============================================================================ per-bin data
class BinData:
    """One calibrated bin and field: inclusion-weighted mean, reliability SD, base mask, per-halo values."""

    def __init__(self, pr, field: str, sel: str, c: int, rows=None):
        b = FC.CAL_BINS[c]
        all_rows = np.flatnonzero(pr["mass_bin"] == b)
        self.rows = all_rows if rows is None else np.asarray(rows)
        self.field, self.c, self.bin = field, c, b
        self.family = FIELD_FAMILY[field]
        self.values = np.asarray(pr[f"{field}_{sel}"][self.rows], float)
        self.w = np.asarray(pr["weight"][self.rows], float)
        self.n = len(self.rows)
        self.mean, self.sd, self.neff = S.weighted_stack(self.values, self.w)
        self.base = np.asarray(pr["base_mask"][b], bool) & (self.mean > 0)
        self.logmass = float(np.mean(FC.MASS_EDGES[b:b + 2]))
        self.cov_base = None

    def mask(self, r_fit: float) -> np.ndarray:
        return self.base & FC.shells_within(FC.F_EDGES, r_fit)

    # -------- GLS ingredients (bootstrap Cov[ln mean], correlation variance for the shrinkage intensity)
    def prepare_gls(self, seed: int):
        rows = S.bootstrap_rows(self.n, BOOT_DRAWS, seed)
        cb = COV.bootstrap_log_mean_covariance(self.values, self.base, rows, weights=self.w)
        self.cov_base = cb["cov"]
        sub = self.values[:, self.base]
        with np.errstate(divide="ignore", invalid="ignore"):
            self.var_r = COV.correlation_bootstrap_variance(sub, rows[:LAMBDA_DRAWS])
        self.corr_base = COV.cov_to_corr(self.cov_base)
        return self

    def gls_cov(self, mask, target):
        """Shrunk bootstrap Cov[ln mean] on the mask shells (pipeline objective (c) recipe); a relative
        eigenvalue floor of 1e-10 guards the Cholesky factorisation of rank-deficient small-N cases."""
        sel = np.searchsorted(np.flatnonzero(self.base), np.flatnonzero(mask))
        C = self.cov_base[np.ix_(sel, sel)]
        var_r = self.var_r[np.ix_(sel, sel)]
        force = self.n < 3 * mask.sum() or not np.all(np.isfinite(var_r))
        Cs, lam = COV.shrunk_covariance(C, target[np.ix_(mask, mask)], var_r=var_r, force_pooled=force)
        Cs = (Cs + Cs.T) / 2
        ev = np.linalg.eigvalsh(Cs)
        floor = 1e-10 * max(ev.max(), 1e-300)
        if ev.min() < floor:
            Cs = Cs + (floor - min(ev.min(), 0.0)) * np.eye(len(Cs))
        return Cs, lam, bool(force)


def pooled_targets(bds: list[BinData]) -> list[np.ndarray]:
    """Shrinkage target per bin: (N-3)-weighted Fisher-z pooled correlation of the nearest other calibrated
    bins (c-1, c+1 first; analysis README: pool neighbours only), widened until the pool holds >= 3p haloes
    (the N = 21 and N = 6 top bins), projected to a correlation matrix."""
    lnx = np.log(np.sqrt(np.maximum(FC.F_EDGES[:-1], 1e-3) * FC.F_EDGES[1:]))
    out = []
    for c, bd in enumerate(bds):
        # nearest neighbours first; widen the pool until it holds >= 3 p haloes (p = shells of this bin)
        order = sorted((k for k in range(len(bds)) if k != c), key=lambda k: (abs(k - c), k))
        use, total = [], 0
        for k in order:
            if len(use) >= 2 and total >= 3 * bd.base.sum():
                break
            use.append(bds[k])
            total += bds[k].n
        corrs, masks, ns = [], [], []
        for u in use:
            corrs.append(u.corr_base)
            masks.append(u.base)
            ns.append(u.n)
        pooled = COV.pooled_correlation(corrs, masks, ns, FC.N_F, lnx=lnx)
        out.append(pooled["corr"])
    return out


# ============================================================================ fits
def _ok(fit):
    return fit is not None and bool(fit.get("accepted", False))


def fit_objective(bd: BinData, objective: str, r_fit: float, family=None, start=None, weights=None, metric=None,
                  target=None, lam=None, sigma_d=None):
    """One fit of the bin's mean profile on base shells <= r_fit.  objective in a, c, d, e, f."""
    family = family or bd.family
    mask = bd.mask(r_fit)
    if mask.sum() < max(F.MIN_SHELLS, family_obj(family).n + 1):
        return None
    legacy = family in ("b12", "b16")
    edges = FC.F_EDGES
    if objective == "a":
        if legacy:
            return F.objective_a(family, bd.mean, bd.sd, mask, start=start, edges=edges, jac="analytic")
        return F.objective_a_general(family, bd.mean, bd.sd, mask, start=start, edges=edges)
    if objective == "c":
        Cs, lamb, force = bd.gls_cov(mask, target)
        fit = (F.objective_c(family, bd.mean, Cs, mask, start=start, edges=edges) if legacy
               else F.objective_c_general(family, bd.mean, Cs, mask, start=start, edges=edges))
        if fit is not None:
            fit["shrinkage_lambda"], fit["pooled_only"] = lamb, force
            fit["C_stat"] = Cs
        return fit
    if objective == "d":
        return F.objective_d(family, bd.mean, weights, mask, start=start, edges=edges)
    if objective == "e":
        return F.objective_e(family, bd.mean, metric, mask, start=start, edges=edges)
    if objective == "f":
        Cs, _, _ = bd.gls_cov(mask, target)
        C = Cs + (sigma_d or 0.0)**2 * np.eye(len(Cs))
        return F.objective_f(family, bd.mean, C, metric, mask, lam, start=start, edges=edges)
    raise ValueError(objective)


def shells_from_fit(family, params, cut=4.0, window=None):
    """Basis-shell averages [42] of a fitted family truncated at ``cut`` (hard), or multiplied by a
    logistic window W = [1 + (x/x_t)^(1/w_ln)]^-1 (window = (x_t, width_dex))."""
    fam = family_obj(family)
    if window is None:
        return FAM.shell_model_truncated(fam, np.asarray(params, float), FC.BASIS_EDGES, float(cut))
    x, w = FAM.shell_grid(FC.BASIS_EDGES)
    xt, width = window
    W = 1.0 / (1.0 + np.exp(np.log(x / xt) / (width * np.log(10.))))
    return np.sum(fam.shape(x, np.asarray(params, float)) * W * w, axis=1)


def flux_matched_edge(family, params, cut, width=0.05):
    """x_t' with sum_s (F W)_s V_s = Y_hard(<cut) (Test C flux matching, here per bin at z = 0)."""
    vol = FC.SHELL_VOLUME
    target = shells_from_fit(family, params, cut) @ vol

    def g(lx):
        return shells_from_fit(family, params, window=(np.exp(lx), width)) @ vol - target

    return float(np.exp(brentq(g, np.log(cut) - 0.5, np.log(cut) + 0.5, xtol=1e-10)))


def calibrated_edge(family, params, target_fn, lo=1.0, hi=FC.X_END):
    """Hard edge x_t in [lo, hi] with target_fn(shells(x_t)) = 0 (monotone in x_t); clamped if no root."""
    f_lo = target_fn(shells_from_fit(family, params, lo))
    f_hi = target_fn(shells_from_fit(family, params, hi))
    if f_lo * f_hi > 0:
        return (lo if abs(f_lo) < abs(f_hi) else hi), False
    x = brentq(lambda t: target_fn(shells_from_fit(family, params, t)), lo, hi, xtol=1e-6)
    return float(x), True


# ============================================================================ partial pooling (tabulated)
def partial_pooling(bds: list[BinData], degree: int = 2):
    """Per F shell: ln mean across calibrated bins shrunk towards a weighted polynomial trend in log M
    (random-effects, DerSimonian-Laird tau^2); sampling variance of ln mean by the delta method.
    Shells with a non-positive mean in any bin are left unpooled.  Returns [10, 35] and diagnostics."""
    nb = len(bds)
    y = np.array([np.log(np.where(bd.mean > 0, bd.mean, np.nan)) for bd in bds])
    v = np.zeros_like(y)
    for c, bd in enumerate(bds):
        w = bd.w / bd.w.sum()
        var = w @ (bd.values - bd.mean)**2 / max(1 - np.sum(w**2), 1e-12)
        v[c] = var / (bd.neff * bd.mean**2)
    xm = np.array([bd.logmass - 14.5 for bd in bds])
    X = np.vander(xm, degree + 1)
    out = np.exp(y)
    tau2 = np.full(FC.N_F, np.nan)
    for s in range(FC.N_F):
        ys, vs = y[:, s], v[:, s]
        if not np.all(np.isfinite(ys)) or not np.all(vs > 0):
            continue
        wf = 1 / vs
        beta = np.linalg.lstsq(X * np.sqrt(wf)[:, None], ys * np.sqrt(wf), rcond=None)[0]
        q = np.sum(wf * (ys - X @ beta)**2)
        dof = nb - (degree + 1)
        cfac = np.sum(wf) - np.sum(wf**2) / np.sum(wf)
        t2 = max(0.0, (q - dof) / cfac) if cfac > 0 else 0.0
        wr = 1 / (vs + t2)
        beta = np.linalg.lstsq(X * np.sqrt(wr)[:, None], ys * np.sqrt(wr), rcond=None)[0]
        trend = X @ beta
        shrunk = (ys / vs + trend / max(t2, 1e-300)) / (1 / vs + 1 / max(t2, 1e-300)) if t2 > 0 else trend
        out[:, s] = np.exp(shrunk)
        tau2[s] = t2
    return out, {"tau2": tau2, "sampling_var": v}


# ============================================================================ helpers
def pad_basis(F35):
    F35 = np.asarray(F35, float)
    return np.pad(F35, [(0, 0)] * (F35.ndim - 1) + [(0, FC.N_B - F35.shape[-1])])


def residual_metrics(model35, truth35, base, weights_by_family=None):
    """Relative residuals model/truth - 1 on base shells: region RMS per bin and importance-weighted RMS
    per family over all bins (weights [10, 35], summed globally)."""
    rel = np.where(base & (truth35 > 0), model35 / np.where(truth35 > 0, truth35, 1) - 1, np.nan)
    out = {}
    for name, a, b in FC.REGIONS:
        inside = (FC.F_EDGES[:-1] >= a - 1e-12) & (FC.F_EDGES[1:] <= b + 1e-12)
        vals = rel[:, inside]
        out[f"rms_{name}"] = float(np.sqrt(np.nanmean(vals**2)) * 100) if np.isfinite(vals).any() else np.nan
    if weights_by_family:
        for fam, W in weights_by_family.items():
            use = np.isfinite(rel) & (W > 0)
            out[f"irms_{fam}"] = float(np.sqrt(np.sum(W[use] * rel[use]**2) / np.sum(W[use])) * 100) if use.any() else np.nan
    return out, rel


# ============================================================================ observable weights (F1 -> F3)
# (family, Jacobian key, multipole or quantile index, lambda within the family); k = 2 for quadratic forms
FIELD_OBS = {
    "P": [("tSZ", "yytot", e, 0.5 if e == 8000 else 1.0) for e in FC.SEL_ELL]
         + [("tSZxDM", "ydmtot_P", e, 0.5 if e == 8000 else 1.0) for e in FC.SEL_ELL],
    "E": [("kSZ", "ksz1h", e, 0.5 if e == 8000 else 1.0) for e in FC.SEL_ELL]
         + [("DM", "dm_mean", None, 1.0), ("DM", "dm_var", None, 1.0)]
         + [("DM", "dmq", p, 1.0) for p in range(len(FC.QUANTILES))]
         + [("tSZxDM", "ydmtot_E", e, 0.5 if e == 8000 else 1.0) for e in FC.SEL_ELL],
}
FAMILY_WEIGHT = {"P": {"tSZ": 0.75, "tSZxDM": 0.25}, "E": {"kSZ": 0.375, "DM": 0.375, "tSZxDM": 0.25}}
K_ORDER = {"yytot": 2, "ydmtot_P": 1, "ksz1h": 2, "dm_mean": 1, "dm_var": 2, "dmq": 1, "ydmtot_E": 1}
TRUTH_KEY = {"yytot": "yytot", "ydmtot_P": "ydmtot", "ydmtot_E": "ydmtot", "ksz1h": "ksz1h", "dm_mean": "dm_mean",
             "dm_var": "dm_var"}


def obs_name(key, at):
    if key == "dmq":
        return f"dm_{FC.Q_NAMES[at]}"
    if at is None:
        return key
    return f"{key}_{at}"


def jacobian_maps(J: dict, Jq, truth_spectra: dict, truth_quantiles) -> dict:
    """Normalised sensitivities g_O[c, s] = dO/dln F[c, s] / O_truth for every FIELD_OBS entry
    (35 F shells; J from Basis.jacobians, Jq [quantile, c, s] from the Mecke pass)."""
    out = {}
    for field, rows in FIELD_OBS.items():
        for fam, key, at, _ in rows:
            if key == "dmq":
                g = Jq[at][:, :FC.N_F] / truth_quantiles[at]
            elif at is None:
                g = J[key][:, :FC.N_F] / truth_spectra[TRUTH_KEY[key]]
            else:
                i = FC.SEL_INDEX[FC.SEL_ELL.index(at)]
                g = J[key][:, :FC.N_F, i] / truth_spectra[TRUTH_KEY[key]][i]
            out[(field, fam, key, at)] = g
    return out


def field_weights(gmaps: dict, field: str) -> tuple[np.ndarray, dict]:
    """(d) weights w[c, s] = sum_O lambda_O |g_O[c, s]| / k_O (normalised importance), each observable's
    map normalised to unit sum over (c, s) first; family maps W_fam (global, unit sum) for the RMS scores."""
    fams = {}
    for (f, fam, key, at), g in gmaps.items():
        if f != field:
            continue
        a = np.abs(g) / K_ORDER[key]
        if not np.isfinite(a).all() or a.sum() <= 0:          # e.g. a quantile pinned at the grid origin
            continue
        a = a / a.sum()
        lam = [r[3] for r in FIELD_OBS[field] if r[0] == fam and r[1] == key and r[2] == at][0]
        fams.setdefault(fam, []).append((lam, a))
    W_fam = {}
    for fam, items in fams.items():
        tot = sum(l for l, _ in items)
        W_fam[fam] = sum(l * a for l, a in items) / tot
    w = sum(FAMILY_WEIGHT[field][fam] * W_fam[fam] for fam in W_fam)
    return w, W_fam


def field_metric(gmaps: dict, field: str) -> np.ndarray:
    """(e) K[c] = sum_O lambda_O g_O[c] g_O[c]^T (g = dO/dln F / O), family weights FAMILY_WEIGHT,
    lambda normalised within each family; [10, 35, 35]."""
    K = np.zeros((FC.N_CAL, FC.N_F, FC.N_F))
    tot = {}
    for fam, key, at, lam in FIELD_OBS[field]:
        tot[fam] = tot.get(fam, 0.0) + lam
    for fam, key, at, lam in FIELD_OBS[field]:
        g = np.nan_to_num(gmaps[(field, fam, key, at)], nan=0.0, posinf=0.0, neginf=0.0)
        K += FAMILY_WEIGHT[field][fam] * lam / tot[fam] * np.einsum("cs,ct->cst", g, g)
    return K
