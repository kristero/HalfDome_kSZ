"""Fitting of Test G: per-field objectives, F's machinery for the non-physical forms, a joint fitter for models
with global parameters, the model-discrepancy term sigma_d and AIC / BIC.

Objectives (on ln of the bin's inclusion-weighted mean stack, resolved shells x_hi <= R_fit, Test F conventions):
  (a) current diagonal       r_s = (ln S_s - ln m_s) / max(SD_s/m_s, 5%)
  (d) importance-diagonal    r_s = sqrt(w_s) (ln S_s - ln m_s), w from Test F's F1 importance maps, field-specific:
                             pressure w = W_tSZ;  electrons w = (W_kSZ + W_DM) / 2   (each map sums to 1 over bins x shells)
E0/P0 and E1/P1 are fitted bin by bin with Test F's own functions (f_fit.fit_objective, objective_a_general,
objective_d; the E1/P1 start of f_methods._flex_start), so they reproduce Test F's grid_a / flex_env fits.
Physical models: all bins and the global parameters in one least-squares problem (theta = ln p for positive
parameters), started from per-bin fits with the globals fixed (multi-start in x_t) and from two global starts.
sigma_d: for chi2/nu = 1 with the shrunk bootstrap Cov[ln mean] of Test F's objective (c) (BinData.gls_cov), at the
fitted parameters, nu = n_shells - (k_bin + k_global/n_bins).
AIC / BIC: per (field, fit target, objective, R_fit), errors inflated by a common sigma_ref per bin = the smallest
sigma_d among the compared models; chi2 = sum_bins r^T (C + sigma_ref^2 I)^-1 r at each model's fitted parameters,
k = all free parameters, n = all fitted shells.
"""
from __future__ import annotations

import g_common as GC
import g_models as GM
import numpy as np
from scipy.optimize import brentq, least_squares

FF = GC.FF
FAM = FF.FAM
F = FF.F
import f_methods as FM  # noqa: E402  (Test F, read-only)

S_ENV_MAX = 3.0                     # E1/P1 outer slope bound (Test F: 10); F's own 4R fits have s <= 1.8


def _env_family(field):
    """Test F's core + A (x/2)^-s family (f_fit.FLEX[...]['env'], same shape and derivatives) with the outer slope
    bounded by S_ENV_MAX: a steeper cusp has a divergent enclosed mass in the unresolved core."""
    fam = FF.FLEX[field]["env"]
    hi = np.array(fam.upper, float)
    hi[fam.names.index("s_env")] = S_ENV_MAX
    return FAM.Family(fam.name, fam.names, fam.lower, hi, fam.log, fam.start, fam.shape, fam.dln, fam.penalty,
                      dict(fam.info, s_env_max=S_ENV_MAX))


ENV = {f: _env_family(f) for f in ("P", "E")}
MAX_NFEV_BIN = 1000                 # per-bin stage-A fits (analytic Jacobian)
MAX_NFEV_JOINT = 4000               # joint fits; reaching it is recorded ('nfev') and flagged in the fit tables


# ============================================================================ data
class FieldData:
    """One field ('P'/'E') and gas selection ('main'/'a2ii') on a halo sample (all, or CV training rows)."""

    def __init__(self, pr, field, sel, wmap, rows_by_bin=None, gls=True, seed=GC.SEED):
        rows_by_bin = rows_by_bin or [None] * GC.N_CAL
        self.field, self.sel = field, sel
        self.bds = [FF.BinData(pr, field, sel, c, rows_by_bin[c]) for c in range(GC.N_CAL)]
        self.mean = np.array([bd.mean for bd in self.bds])
        self.sd = np.array([bd.sd for bd in self.bds])
        self.base = np.array([bd.base for bd in self.bds])
        self.wmap = np.asarray(wmap, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.sig_a = np.maximum(np.divide(self.sd, self.mean, out=np.ones_like(self.sd), where=self.mean > 0), .05)
        self.targets = None
        if gls:
            for c, bd in enumerate(self.bds):
                bd.prepare_gls(seed + 100 * c + (7 if field == "E" else 0) + (3 if sel != "main" else 0))
            self.targets = FF.pooled_targets(self.bds)

    def masks(self, r_fit):
        return np.array([bd.mask(r_fit) for bd in self.bds])

    def scale(self, objective):
        if objective == "a":
            return 1.0 / self.sig_a
        return np.sqrt(np.maximum(self.wmap, 0.0))


def field_weight_map(imp, field):
    """Objective (d) weights, field-specific: pressure = tSZ map; electrons = (kSZ + DM) / 2."""
    if field == "P":
        return np.asarray(imp["Wfam_P_tSZ"], float)
    return 0.5 * np.asarray(imp["Wfam_E_kSZ"], float) + 0.5 * np.asarray(imp["Wfam_E_DM"], float)


# ============================================================================ F-machinery fits (E0/P0, E1/P1)
def fit_core_F(data: FieldData, objective, r_fit, fits_a=None):
    """Model 0: Test F's fit_objective per bin (objective a; d with G's weights, started from the a-fit).
    Returns per-bin fit dicts (None -> fallback to the (a) 2R fit as in Test F)."""
    out = []
    for c, bd in enumerate(data.bds):
        if objective == "a":
            fit = FF.fit_objective(bd, "a", r_fit)
        else:
            start = fits_a[c]["parameters"] if fits_a and fits_a[c] is not None else None
            fit = FF.fit_objective(bd, "d", r_fit, start=start, weights=data.wmap[c])
        out.append(fit)
    return out


def fit_env_F(data: FieldData, objective, r_fit, fit_a4, fit_a2):
    """Model 1 (core + A (x/2)^-s): Test F's family, start (f_methods._flex_start) and objective functions."""
    fam = ENV[data.field]
    out = []
    for c, bd in enumerate(data.bds):
        start = FM._flex_start("env", data.field, fit_a4[c], fit_a2[c], bd)
        mask = bd.mask(r_fit)
        if objective == "a":
            fit = F.objective_a_general(fam, bd.mean, bd.sd, mask, start=start, edges=GC.F_EDGES)
        else:
            fit = F.objective_d(fam, bd.mean, data.wmap[c], mask, start=start, edges=GC.F_EDGES)
        if fit is None:
            fit = {"parameters": list(start), "accepted": False, "success": False, "note": "too few shells: start"}
        out.append(fit)
    return out


# ============================================================================ joint fitter
def _theta_bounds(pars):
    lo = np.array([p.theta_bounds()[0] for p in pars])
    hi = np.array([p.theta_bounds()[1] for p in pars])
    return lo, hi


def _to_theta(pars, values):
    return np.array([p.to_theta(v) for p, v in zip(pars, values)], float)


def _to_values(pars, theta):
    return np.array([p.to_value(t) for p, t in zip(pars, theta)], float)


class JointProblem:
    def __init__(self, model: GM.Model, data: FieldData, objective, r_fit, ctx, x, wq, T2h):
        self.model, self.data, self.objective, self.r_fit = model, data, objective, r_fit
        self.ctx, self.x, self.wq, self.T2h = ctx, x, wq, T2h
        self.masks = data.masks(r_fit)
        self.scale = data.scale(objective)
        with np.errstate(divide="ignore"):
            self.lnm = np.log(np.where(data.mean > 0, data.mean, 1.0))
        self.nb = GC.N_CAL
        self.lo_b, self.hi_b = _theta_bounds(model.per_bin)
        self.lo_g, self.hi_g = _theta_bounds(model.globals)

    def shells(self, pb, pg):
        b, o = self.model.nodes(pb, pg, self.ctx, self.x, self.T2h)
        return np.sum((b + o) * self.wq[None], axis=-1)

    def residual_bins(self, pb, pg, bins=None):
        S = self.shells(pb, pg)
        out = []
        for c in (range(self.nb) if bins is None else bins):
            m = self.masks[c]
            with np.errstate(divide="ignore", invalid="ignore"):
                r = (np.log(np.maximum(S[c, m], 1e-300)) - self.lnm[c, m]) * self.scale[c, m]
            out.append(r)
        return out

    # ---- analytic derivatives (theta space)
    def shells_jac(self, pb, pg, bins=None, ctx=None, T2h=None):
        F, dB, dG = self.model.nodes_with_derivatives(pb, pg, ctx or self.ctx, self.x, self.T2h if T2h is None else T2h)
        S = np.sum(F * self.wq[None], axis=-1)
        dSb = np.sum(dB * self.wq[None, None], axis=-1) / S[:, None, :]
        dSg = np.sum(dG * self.wq[None, None], axis=-1) / S[:, None, :]
        return S, dSb, dSg

    # ---- per-bin fit with fixed globals (one bin's problem)
    def fit_bin(self, c, pb_start, pg, xt_starts=(None,)):
        best = None
        ctx = dict(self.ctx)
        ctx["b"] = np.asarray(self.ctx["b"])[c:c + 1]
        ctx["logm"] = np.asarray(self.ctx["logm"])[c:c + 1]
        m = self.masks[c]
        cache = {}

        def ev(th):
            key = th.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = self.shells_jac(_to_values(self.model.per_bin, th)[None], pg, ctx=ctx, T2h=self.T2h[c:c + 1])
            return cache[key]

        def fun(th):
            S = ev(th)[0][0]
            return (np.log(np.maximum(S[m], 1e-300)) - self.lnm[c, m]) * self.scale[c, m]

        def jac(th):
            dSb = ev(th)[1][0]
            return (dSb[:, m] * self.scale[c, m]).T

        for xt in xt_starts:
            p0 = np.array(pb_start, float).copy()
            if xt is not None and "x_t" in self.model.names()[0]:
                p0[self.model.names()[0].index("x_t")] = xt
            th0 = np.clip(_to_theta(self.model.per_bin, p0), self.lo_b + 1e-7, self.hi_b - 1e-7)
            sol = least_squares(fun, th0, jac=jac, bounds=(self.lo_b, self.hi_b), method="trf", x_scale="jac",
                                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=MAX_NFEV_BIN)
            if best is None or sol.cost < best.cost:
                best = sol
        return _to_values(self.model.per_bin, best.x), best

    # ---- joint
    def pack(self, pb, pg):
        return np.r_[np.concatenate([_to_theta(self.model.per_bin, pb[c]) for c in range(self.nb)]),
                     _to_theta(self.model.globals, pg)]

    def unpack(self, theta):
        kb = self.model.kb
        pb = np.array([_to_values(self.model.per_bin, theta[c * kb:(c + 1) * kb]) for c in range(self.nb)])
        pg = _to_values(self.model.globals, theta[self.nb * kb:])
        return pb, pg

    def _ev(self, theta):
        key = theta.tobytes()
        if getattr(self, "_ck", None) != key:
            pb, pg = self.unpack(theta)
            self._cv = self.shells_jac(pb, pg)
            self._ck = key
        return self._cv

    def fun(self, theta):
        S = self._ev(theta)[0]
        out = []
        for c in range(self.nb):
            m = self.masks[c]
            out.append((np.log(np.maximum(S[c, m], 1e-300)) - self.lnm[c, m]) * self.scale[c, m])
        return np.concatenate(out)

    def jac(self, theta):
        _, dSb, dSg = self._ev(theta)
        kb, kg = self.model.kb, self.model.kg
        n = int(self.masks.sum())
        J = np.zeros((n, self.nb * kb + kg))
        r0 = 0
        for c in range(self.nb):
            m = self.masks[c]
            k = int(m.sum())
            sc = self.scale[c, m]
            J[r0:r0 + k, c * kb:(c + 1) * kb] = (dSb[c][:, m] * sc).T
            if kg:
                J[r0:r0 + k, self.nb * kb:] = (dSg[c][:, m] * sc).T
            r0 += k
        return J

    def bounds(self):
        lo = np.r_[np.tile(self.lo_b, self.nb), self.lo_g]
        hi = np.r_[np.tile(self.hi_b, self.nb), self.hi_g]
        return lo, hi

    def solve(self, pb0, pg0):
        lo, hi = self.bounds()
        th0 = np.clip(self.pack(pb0, pg0), lo + 1e-7, hi - 1e-7)
        sol = least_squares(self.fun, th0, jac=self.jac, bounds=(lo, hi), method="trf", x_scale="jac", ftol=1e-11,
                            xtol=1e-11, gtol=1e-11, max_nfev=MAX_NFEV_JOINT)
        return sol


def fit_joint(model: GM.Model, data: FieldData, objective, r_fit, ctx, x, wq, T2h, core_start, extra_starts=None,
              global_starts=None):
    """Fit a physical (or diagnostic) model jointly over the calibrated bins.

    core_start [nb, 3]: starting core parameters (natural units); extra_starts: dict name -> [nb] or scalar for other
    per-bin parameters; global_starts: list of dicts name -> value (several global starts, best cost kept)."""
    prob = JointProblem(model, data, objective, r_fit, ctx, x, wq, T2h)
    names_b, names_g = model.names()
    pb0 = np.zeros((GC.N_CAL, model.kb))
    pb0[:, :3] = core_start
    for k, p in enumerate(model.per_bin[3:], start=3):
        v = (extra_starts or {}).get(p.name, p.start)
        pb0[:, k] = v
    gstarts = global_starts or [{}]
    xt_starts = (1.5, 2.5, 4.0) if "x_t" in names_b else (None,)
    best, stage_a = None, None
    for gs in gstarts:
        pg0 = np.array([gs.get(p.name, p.start) for p in model.globals], float)
        # stage A: bins independently with the globals at their start (multi-start in x_t)
        pbA = np.zeros_like(pb0)
        for c in range(GC.N_CAL):
            pbA[c], _ = prob.fit_bin(c, pb0[c], pg0, xt_starts)
        if model.kg == 0:
            theta = prob.pack(pbA, pg0)
            sol = type("Sol", (), {"x": theta, "cost": 0.5 * float(np.sum(prob.fun(theta)**2)), "success": True,
                                   "nfev": 0, "status": 1, "jac": None})
        else:
            sol = prob.solve(pbA, pg0)
        if best is None or sol.cost < best.cost:
            best, stage_a = sol, pbA
    pb, pg = prob.unpack(best.x)
    lo, hi = prob.bounds()
    near = np.minimum(best.x - lo, hi - best.x) < 1e-5
    at_bound_b = near[:prob.nb * model.kb].reshape(prob.nb, model.kb)
    at_bound_g = near[prob.nb * model.kb:]
    res = prob.residual_bins(pb, pg)
    out = {"model": model.name, "field": model.field, "objective": objective, "r_fit": r_fit, "pb": pb, "pg": pg,
           "names_b": names_b, "names_g": names_g, "cost": float(best.cost), "success": bool(best.success),
           "nfev": int(best.nfev), "at_bound_b": at_bound_b, "at_bound_g": at_bound_g,
           "chi2_bins": np.array([float(np.sum(r**2)) for r in res]), "n_shells": prob.masks.sum(axis=1)}
    if model.kg and getattr(best, "jac", None) is not None:
        sv = np.linalg.svd(best.jac, compute_uv=False)
        out["jacobian_condition"] = float(sv[0] / max(sv[-1], 1e-300))
    else:
        out["jacobian_condition"] = np.nan
    out["accepted"] = bool(out["success"] and not at_bound_g.any())
    return out


# ============================================================================ sigma_d, AIC / BIC
def sigma_d_solve(r, C, nu):
    """sigma_d with r^T (C + sigma_d^2 I)^-1 r = nu (0 if already <= nu); NaN if nu <= 0."""
    if nu <= 0 or len(r) == 0:
        return np.nan
    ev, U = np.linalg.eigh((C + C.T) / 2)
    ev = np.maximum(ev, 0.0)
    u2 = (U.T @ r)**2

    def g(s):
        return float(np.sum(u2 / (ev + s)) - nu)

    if g(0.0) <= 0:
        return 0.0
    hi = max(float(np.mean(r**2)), 1e-12)
    while g(hi) > 0:
        hi *= 4
    return float(np.sqrt(brentq(g, 0.0, hi, xtol=1e-18, rtol=1e-12)))


def chi2_inflated(r, C, s2):
    M = (C + C.T) / 2 + s2 * np.eye(len(C))
    return float(r @ np.linalg.solve(M, r))


def bin_residuals(data: FieldData, shells35, r_fit):
    """ln S - ln m on each bin's fit shells (base & x_hi <= R_fit) and the shrunk GLS covariance there."""
    out = []
    for c, bd in enumerate(data.bds):
        m = bd.mask(r_fit)
        if m.sum() == 0:
            out.append((np.zeros(0), np.zeros((0, 0)), m))
            continue
        r = np.log(np.maximum(shells35[c, m], 1e-300)) - np.log(bd.mean[m])
        C, _, _ = bd.gls_cov(m, data.targets[c])
        out.append((r, C, m))
    return out
