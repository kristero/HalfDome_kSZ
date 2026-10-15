"""All calibration methods of Test F on one halo sample (full sample or a CV training set).

fit_all(...) returns {label: {"P": [10, 42], "E": [10, 42], "meta": {...}}} (basis-shell averages per
calibrated bin) and fit records.  Labels:
  grid_{o}_Rf{rf}_Rc{rc}   objective o in a/c/d/e/f (B12 pressure, B16 electrons), hard cut   (F2/F3)
  sig_{o}_Rf{rf}_Rc{rc}    the same with the flux-matched 0.05-dex logistic edge (Test C convention)
  lam_Rf{rf}_l{lam}        hybrid (f) lambda scan at R_cut = 4                                   (F3)
  flex_{fam}_{o}           gNFW (5 free), core + outer power law (B12/B16 + A (x/2)^-s), core x spline,
                           objective a or d, fitted and cut at 4R                                (F4)
  tab_stack, tab_pooled    measured mean stacks to 4R, raw and partially pooled in mass (ceiling) (F4)
  trunc_flux, trunc_obs    (a) 2R fit with a per-bin calibrated hard edge x_t in [1, 12] R200c matched
                           to Y / N_e(<4R) or to the bin's yy-1h(ell=1000) / DM-variance contribution (F4)
  direct_bin, direct_trend observable-space fits (per bin; smooth log-M trend, joint P+E with linearised
                           DM quantiles), weak profile chi^2 regulariser                          (F4)
"""
from __future__ import annotations

import f_common as FC
import f_fit as FF
import numpy as np
from scipy.optimize import least_squares

OBJECTIVES = ("a", "c", "d", "e", "f")
SIGMA_OBS = 0.01           # 1% tolerance per observable in the direct fits
ETA = 0.1                  # weak profile chi^2 regulariser (mean chi^2 per shell x ETA)
SIGMOID_WIDTH = 0.05


class SelBasis:
    """The basis kernels restricted to the selected multipoles (fast direct fits)."""

    def __init__(self, B):
        i = FC.SEL_INDEX
        self.B = B
        self.K = {k: getattr(B, k)[..., i] for k in ("K_yy1h", "K_ksz1h", "K_ydm1h")}
        self.K_dmvar = B.K_dmvar
        self.W_y, self.W_tau = B.W_y[..., i], B.W_tau[..., i]
        self.W_fb_y, self.W_fb_tau = B.W_fb_y[:, i], B.W_fb_tau[:, i]
        self.pv = B.pv[:, i]
        self.dmz = B.dmz
        self.s2_lin = B.s2_lin
        self.fb = {k: (v[i] if np.ndim(v) == 1 and len(v) == len(B.ell) else v) for k, v in B.fb.items()
                   if not k.startswith("J_")}
        self.mean_y, self.dm_mean = B.mean_y, B.dm_mean

    def global_obs(self, P, E, SPP=None, SEE=None, SPE=None):
        SPP = np.einsum("cs,ct->cst", P, P) if SPP is None else SPP
        SEE = np.einsum("cs,ct->cst", E, E) if SEE is None else SEE
        SPE = np.einsum("cs,ct->cst", P, E) if SPE is None else SPE
        WP = self.W_fb_y + np.einsum("cs,zcsl->zl", P, self.W_y, optimize=True)
        WE = self.W_fb_tau + np.einsum("cs,zcsl->zl", E, self.W_tau, optimize=True)
        o = {"yytot": np.einsum("cstl,cst->l", self.K["K_yy1h"], SPP) + self.fb["yy1h"] + np.sum(self.pv * WP**2, axis=0),
             "ksz1h": np.einsum("cstl,cst->l", self.K["K_ksz1h"], SEE) + self.fb["ksz1h"],
             "ydmtot": np.einsum("cstl,cst->l", self.K["K_ydm1h"], SPE) + self.fb["ydm1h"]
             + np.sum(self.pv * WP * self.dmz[:, None] * WE, axis=0),
             "dm_mean": float(np.sum(E * self.dm_mean) + self.fb["dm_mean"]),
             "dm_var": float(np.einsum("cst,cst->", self.K_dmvar, SEE) + self.fb["dm_var"])}
        return o


def _ln(params):
    return np.log(np.asarray(params, float))


# ============================================================================ main driver
def fit_all(pr, sel, imp, B, truth, rows_by_bin=None, seed=FC.SEED, parts=("grid", "flex", "tab", "trunc", "direct"),
            lambdas=FF.LAMBDAS, lam_choice=None, log=None, objectives=OBJECTIVES):
    """Fit every method on the sample.

    imp: dict with w_P, w_E [10, 35] (objective d), K_P, K_E [10, 35, 35] (metric), W_fam_P/E (RMS).
    truth: dict with the sample's truth: 'P', 'E' [10, 35] means, 'SPP', 'SEE', 'SPE' [10, 35, 35],
           'obs' (global scalars incl. dm_q*), 'Qdet' [5] (quantiles of the deterministic mean) and
           'Jq_det' [5, 10, 42] (Mecke Jacobians at the deterministic mean).
    lam_choice: hybrid lambda per field (None: choose on this sample by the family error at R_fit = 4).
    """
    log = log or (lambda *a: None)
    rows_by_bin = rows_by_bin or [None] * FC.N_CAL
    bds = {f: [FF.BinData(pr, f, sel, c, rows_by_bin[c]) for c in range(FC.N_CAL)] for f in ("P", "E")}
    targets = {f: None for f in ("P", "E")}
    if objectives != ("a",):
        for f in ("P", "E"):
            for c, bd in enumerate(bds[f]):
                bd.prepare_gls(seed + 100 * c + (7 if f == "E" else 0))
        targets = {f: FF.pooled_targets(bds[f]) for f in ("P", "E")}
    models, records = {}, []

    def add(label, P, E, meta):
        models[label] = {"P": np.asarray(P, float), "E": np.asarray(E, float), "meta": meta}

    # ---------------- objective fits at every R_fit (per bin, per field)
    fits = {}          # (field, objective, r_fit) -> list over bins of fit dicts
    for f in ("P", "E"):
        w = imp[f"w_{f}"]
        Km = imp[f"K_{f}"]
        for rf in FC.R_FIT:
            for c, bd in enumerate(bds[f]):
                fa = FF.fit_objective(bd, "a", rf)
                fits.setdefault((f, "a", rf), []).append(fa)
                if objectives == ("a",):
                    continue
                start = fa["parameters"] if fa is not None else None
                fc = FF.fit_objective(bd, "c", rf, start=start, target=targets[f][c])
                fd = FF.fit_objective(bd, "d", rf, start=start, weights=w[c])
                fe = FF.fit_objective(bd, "e", rf, start=start, metric=Km[c])
                for o, fit in zip("cde", (fc, fd, fe)):
                    fits.setdefault((f, o, rf), []).append(fit)
                sd = fc["sigma_d"] if fc is not None else 0.0
                for lam in lambdas:
                    ff = FF.fit_objective(bd, "f", rf, start=start, metric=Km[c], target=targets[f][c], lam=lam,
                                          sigma_d=sd)
                    fits.setdefault((f, f"f{lam:g}", rf), []).append(ff)
        log(f"objective fits {f} done")
    # lambda choice: smallest mean |family error| at R_fit = R_cut = 4 (on this sample)
    sb = SelBasis(B)
    if objectives == ("a",):
        lam_choice = lam_choice or {}
    elif lam_choice is None:
        lam_choice = {}
        for f in ("P", "E"):
            best = None
            for lam in lambdas:
                P4 = _shells(fits, "P" if f == "P" else "P", "a", 4.0, 4.0, bds)
                E4 = _shells(fits, "E", "a", 4.0, 4.0, bds)
                trial = _shells(fits, f, f"f{lam:g}", 4.0, 4.0, bds)
                Pm, Em = (trial, E4) if f == "P" else (P4, trial)
                err = _family_error(sb, Pm, Em, truth, f)
                if best is None or err < best[0]:
                    best = (err, lam)
            lam_choice[f] = best[1]
    if "f" in objectives:
        for f in ("P", "E"):
            for rf in FC.R_FIT:
                fits[(f, "f", rf)] = fits[(f, f"f{lam_choice[f]:g}", rf)]
    # ---------------- F2 / F3 grids (hard cut and flux-matched logistic edge) and the lambda scan
    if "grid" in parts:
        for o in objectives:
            for rf in FC.R_FIT:
                for rc in FC.R_CUT:
                    P = _shells(fits, "P", o, rf, rc, bds)
                    E = _shells(fits, "E", o, rf, rc, bds)
                    add(f"grid_{o}_Rf{rf:g}_Rc{rc:g}", P, E, {"objective": o, "R_fit": rf, "R_cut": rc, "edge": "hard",
                                                             "lambda": lam_choice if o == "f" else None})
                    if o == "a":
                        P = _shells(fits, "P", o, rf, rc, bds, sigmoid=True)
                        E = _shells(fits, "E", o, rf, rc, bds, sigmoid=True)
                        add(f"sig_{o}_Rf{rf:g}_Rc{rc:g}", P, E, {"objective": o, "R_fit": rf, "R_cut": rc,
                                                                "edge": "flux-matched logistic 0.05 dex"})
        for rf in ([r for r in (2.0, 3.0, 4.0) if r in FC.R_FIT] if "f" in objectives else []):
            for lam in lambdas:
                P = _shells(fits, "P", f"f{lam:g}", rf, 4.0, bds)
                E = _shells(fits, "E", f"f{lam:g}", rf, 4.0, bds)
                add(f"lam_Rf{rf:g}_l{lam:g}", P, E, {"objective": "f", "lambda": lam, "R_fit": rf, "R_cut": 4.0})
    for (f, o, rf), lst in fits.items():
        for c, fit in enumerate(lst):
            records.append(_record(f, o, rf, c, fit))
    log("grids done")
    # ---------------- F4: flexible families (fitted and cut at 4R; objective a and d)
    if "flex" in parts:
        for name in ("gnfw5", "env", "spline"):
            for o in ("a", "d"):
                out = {}
                for f in ("P", "E"):
                    fam = FF.FLEX[f][name]
                    arr = np.zeros((FC.N_CAL, FC.N_B))
                    for c, bd in enumerate(bds[f]):
                        base_fit = fits[(f, "a", 4.0)][c]
                        start = _flex_start(name, f, base_fit, fits[(f, "a", 2.0)][c], bd)
                        mask = bd.mask(4.0)
                        if o == "a":
                            fit = FF.F.objective_a_general(fam, bd.mean, bd.sd, mask, start=start, edges=FC.F_EDGES)
                        else:
                            fit = FF.F.objective_d(fam, bd.mean, imp[f"w_{f}"][c], mask, start=start, edges=FC.F_EDGES)
                        if fit is None:
                            fit = {"parameters": start, "accepted": False}
                        arr[c] = FF.shells_from_fit(fam, fit["parameters"], 4.0)
                        records.append(_record(f, f"flex_{name}_{o}", 4.0, c, fit, family=fam.name))
                    out[f] = arr
                add(f"flex_{name}_{o}", out["P"], out["E"], {"family": name, "objective": o, "R_fit": 4.0, "R_cut": 4.0})
        log("flexible families done")
    # ---------------- F4: tabulated stacks (raw and partially pooled)
    if "tab" in parts:
        add("tab_stack", FF.pad_basis(np.array([bd.mean for bd in bds["P"]])),
            FF.pad_basis(np.array([bd.mean for bd in bds["E"]])), {"method": "measured mean stack to 4R"})
        pp, ip = FF.partial_pooling(bds["P"])
        pe, ie = FF.partial_pooling(bds["E"])
        add("tab_pooled", FF.pad_basis(pp), FF.pad_basis(pe),
            {"method": "partial pooling in log M (quadratic trend, DerSimonian-Laird tau^2)",
             "tau2_P": ip["tau2"], "tau2_E": ie["tau2"]})
    # ---------------- F4: truncation radius as a calibrated parameter (base: objective a fitted to 2R)
    if "trunc" in parts:
        _calibrated_truncation(fits, bds, B, truth, add, records)
        log("calibrated truncation done")
    # ---------------- F4: direct observable-space fits
    if "direct" in parts:
        _direct_fits(fits, bds, B, sb, truth, add, records, log)
        log("direct fits done")
    return models, records, {"lambda_choice": lam_choice}


# ============================================================================ helpers
def _shells(fits, f, o, rf, rc, bds, sigmoid=False):
    arr = np.zeros((FC.N_CAL, FC.N_B))
    family = FF.FIELD_FAMILY[f]
    for c, fit in enumerate(fits[(f, o, rf)]):
        if fit is None:
            fit = fits[(f, "a", rf)][c] or fits[(f, "a", 2.0)][c]
        p = fit["parameters"]
        if sigmoid:
            xt = FF.flux_matched_edge(family, p, rc, SIGMOID_WIDTH)
            arr[c] = FF.shells_from_fit(family, p, window=(xt, SIGMOID_WIDTH))
        else:
            arr[c] = FF.shells_from_fit(family, p, rc)
    return arr


def _record(f, o, rf, c, fit, family=None):
    rec = {"field": f, "method": o, "R_fit": rf, "bin": FC.CAL_LABELS[c], "family": family or FF.FIELD_FAMILY[f]}
    if fit is None:
        rec["status"] = "no fit (too few shells)"
        return rec
    for k in ("accepted", "chi2_per_dof", "sigma_d", "rms_percent", "shrinkage_lambda", "n_shells", "jacobian_condition",
              "bound_distance"):
        if k in fit and fit[k] is not None:
            rec[k] = fit[k]
    rec["parameters"] = "; ".join(f"{v:.6g}" for v in fit["parameters"])
    return rec


def _family_error(sb, P, E, truth, field):
    """Mean |relative error| over the field's primary family observables (moment forms; quantiles by the
    linear Mecke response around the deterministic truth mean)."""
    o = sb.global_obs(P, E)
    t = truth["obs"]
    errs = []
    if field == "P":
        for e in FC.PRIMARY_ELL:
            i = FC.SEL_ELL.index(e)
            errs.append(abs(o["yytot"][i] / t[f"yytot_{e}"] - 1))
            errs.append(0.5 * abs(o["ydmtot"][i] / t[f"ydmtot_{e}"] - 1))
    else:
        for e in FC.PRIMARY_ELL:
            i = FC.SEL_ELL.index(e)
            errs.append(abs(o["ksz1h"][i] / t[f"ksz1h_{e}"] - 1))
            errs.append(0.5 * abs(o["ydmtot"][i] / t[f"ydmtot_{e}"] - 1))
        errs.append(abs(o["dm_mean"] / t["dm_mean"] - 1))
        errs.append(abs(o["dm_var"] / t["dm_var"] - 1))
        q = _linear_quantiles(E, truth)
        for p, name in enumerate(FC.Q_NAMES):
            errs.append(abs(q[p] / t[f"dm_{name}"] - 1))
    return float(np.mean(errs))


def _linear_quantiles(E, truth):
    m = FF.pad_basis(truth["E"])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where((E > 0) & (m > 0), np.log(np.where(E > 0, E, 1) / np.where(m > 0, m, 1)), 0.0)
    return truth["Qdet"] + np.einsum("pcs,cs->p", truth["Jq_det"], d)


def _flex_start(name, f, fit4, fit2, bd):
    p4 = np.asarray(fit4["parameters"] if fit4 is not None else FF.FAM.get(FF.FIELD_FAMILY[f]).start, float)
    p2 = np.asarray(fit2["parameters"] if fit2 is not None else p4, float)
    if name == "gnfw5":
        if f == "P":
            return [0.5176 * p4[0], p4[1], 1.0, p4[2] + 0.3, 0.3]
        return [p4[0], 0.5, p4[1], p4[2], 0.2]
    if name == "env":
        i = FC.edge_index(FC.F_EDGES, 2.5)
        level = float(bd.mean[i]) if bd.mean[i] > 0 else 1e-3
        return list(p2) + [max(0.3 * level, 1e-7), 1.0]
    return list(p4) + [0.0] * len(FF.SPLINE_KNOTS)


def _calibrated_truncation(fits, bds, B, truth, add, records):
    vol = FC.SHELL_VOLUME
    i1000 = FC.SEL_INDEX[FC.SEL_ELL.index(1000)]
    out = {"flux": {}, "obs": {}}
    edges = {"flux": {}, "obs": {}}
    for f in ("P", "E"):
        family = FF.FIELD_FAMILY[f]
        for kind in ("flux", "obs"):
            arr = np.zeros((FC.N_CAL, FC.N_B))
            xt_list = []
            for c, bd in enumerate(bds[f]):
                fit = fits[(f, "a", 2.0)][c]
                p = fit["parameters"]
                if kind == "flux":
                    target = float(truth[f][c] @ vol[:FC.N_F])
                    fn = (lambda s, t=target: s @ vol - t)
                elif f == "P":
                    Kc = B.K_yy1h[c, :, :, i1000]
                    target = float(np.einsum("st,st->", Kc, B.pad2(truth["SPP"][c])))
                    fn = (lambda s, K=Kc, t=target: s @ K @ s - t)
                else:
                    Kc = B.K_dmvar[c]
                    target = float(np.einsum("st,st->", Kc, B.pad2(truth["SEE"][c])))
                    fn = (lambda s, K=Kc, t=target: s @ K @ s - t)
                xt, ok = FF.calibrated_edge(family, p, fn)
                arr[c] = FF.shells_from_fit(family, p, xt)
                xt_list.append(xt)
                records.append({"field": f, "method": f"trunc_{kind}", "R_fit": 2.0, "bin": FC.CAL_LABELS[c],
                                "family": family, "x_t": xt, "root_found": ok})
            out[kind][f] = arr
            edges[kind][f] = xt_list
    for kind in ("flux", "obs"):
        add(f"trunc_{kind}", out[kind]["P"], out[kind]["E"],
            {"method": f"(a) 2R fit, per-bin hard edge matched to {'Y/N_e(<4R)' if kind == 'flux' else 'yy1h(1000)/DM variance'}",
             "x_t_P": edges[kind]["P"], "x_t_E": edges[kind]["E"]})


def _reg_residuals(bd, family, params):
    mask = bd.mask(4.0)
    S = FF.FAM.shell_model(family, params, FC.F_EDGES)
    sig = np.maximum(np.divide(bd.sd, bd.mean, out=np.ones_like(bd.sd), where=bd.mean > 0), 0.05)
    r = (np.log(S[mask]) - np.log(bd.mean[mask])) / sig[mask]
    return np.sqrt(ETA / mask.sum()) * r


def _direct_fits(fits, bds, B, sb, truth, add, records, log):
    """Per-bin and smooth-trend observable-space fits against the sample's (scatter-inclusive) truth."""
    i_sel = {e: k for k, e in enumerate(FC.SEL_ELL)}
    WtotP_truth = sb.W_fb_y + np.einsum("cs,zcsl->zl", FF.pad_basis(truth["P"]), sb.W_y, optimize=True)
    WtotE_truth = sb.W_fb_tau + np.einsum("cs,zcsl->zl", FF.pad_basis(truth["E"]), sb.W_tau, optimize=True)
    P_bin = np.zeros((FC.N_CAL, FC.N_B))
    E_bin = np.zeros((FC.N_CAL, FC.N_B))
    for c in range(FC.N_CAL):
        bdP, bdE = bds["P"][c], bds["E"][c]
        SPP, SEE, SPE = (B.pad2(truth[k][c]) for k in ("SPP", "SEE", "SPE"))
        mP, mE = FF.pad_basis(truth["P"][c]), FF.pad_basis(truth["E"][c])
        t1h = np.einsum("stl,st->l", sb.K["K_yy1h"][c], SPP)
        t2h = np.einsum("zl,zsl,s->l", sb.pv * WtotP_truth, sb.W_y[:, c], mP)
        tmy = float(mP @ sb.mean_y[c])
        tx = np.einsum("stl,st->l", sb.K["K_ydm1h"][c], SPE)
        wl = np.array([0.5 if e == 8000 else 1.0 for e in FC.SEL_ELL])

        def resP(theta):
            p = np.exp(theta)
            s = FF.FAM.shell_model("b12", p, FC.BASIS_EDGES) * (FC.BASIS_EDGES[1:] <= 4 + 1e-12)
            o1 = np.einsum("stl,s,t->l", sb.K["K_yy1h"][c], s, s)
            o2 = np.einsum("zl,zsl,s->l", sb.pv * WtotP_truth, sb.W_y[:, c], s)
            ox = np.einsum("stl,s,t->l", sb.K["K_ydm1h"][c], s, mE)
            r = [np.sqrt(0.75 * wl / (wl.sum() + 4)) * (o1 / t1h - 1) / FF_SIG,
                 np.sqrt(0.75 / (wl.sum() + 4)) * (o2[:3] / t2h[:3] - 1) / FF_SIG,
                 np.sqrt(0.75 / (wl.sum() + 4)) * np.atleast_1d(float(s @ sb.mean_y[c]) / tmy - 1) / FF_SIG,
                 np.sqrt(0.25 / 4) * (ox[:4] / tx[:4] - 1) / FF_SIG,
                 _reg_residuals(bdP, "b12", p)]
            return np.concatenate(r)

        start = fits[("P", "a", 4.0)][c]["parameters"]
        lo, hi = np.log(np.asarray(FF.FAM.M.BOUNDS["b12"], float))
        sol = least_squares(resP, np.clip(_ln(start), lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=400,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        pP = np.exp(sol.x)
        P_bin[c] = FF.shells_from_fit("b12", pP, 4.0)
        records.append({"field": "P", "method": "direct_bin", "R_fit": 4.0, "bin": FC.CAL_LABELS[c], "family": "b12",
                        "parameters": "; ".join(f"{v:.6g}" for v in pP), "success": bool(sol.success),
                        "cost": float(sol.cost)})
        tk = np.einsum("stl,st->l", sb.K["K_ksz1h"][c], SEE)
        tdm = float(mE @ sb.dm_mean[c])
        tvar = float(np.einsum("st,st->", sb.K_dmvar[c], SEE))
        tx2 = tx

        def resE(theta, sP=P_bin[c]):
            p = np.exp(theta)
            s = FF.FAM.shell_model("b16", p, FC.BASIS_EDGES) * (FC.BASIS_EDGES[1:] <= 4 + 1e-12)
            ok = np.einsum("stl,s,t->l", sb.K["K_ksz1h"][c], s, s)
            ox = np.einsum("stl,s,t->l", sb.K["K_ydm1h"][c], sP, s)
            r = [np.sqrt(0.375 * wl / wl.sum()) * (ok / tk - 1) / FF_SIG,
                 np.sqrt(0.375 / 2) * np.array([float(s @ sb.dm_mean[c]) / tdm - 1,
                                                float(s @ sb.K_dmvar[c] @ s) / tvar - 1]) / FF_SIG,
                 np.sqrt(0.25 / 4) * (ox[:4] / tx2[:4] - 1) / FF_SIG,
                 _reg_residuals(bdE, "b16", p)]
            return np.concatenate(r)

        start = fits[("E", "a", 4.0)][c]["parameters"]
        lo, hi = np.log(np.asarray(FF.FAM.M.BOUNDS["b16"], float))
        sol = least_squares(resE, np.clip(_ln(start), lo + 1e-7, hi - 1e-7), bounds=(lo, hi), max_nfev=400,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        pE = np.exp(sol.x)
        E_bin[c] = FF.shells_from_fit("b16", pE, 4.0)
        records.append({"field": "E", "method": "direct_bin", "R_fit": 4.0, "bin": FC.CAL_LABELS[c], "family": "b16",
                        "parameters": "; ".join(f"{v:.6g}" for v in pE), "success": bool(sol.success),
                        "cost": float(sol.cost)})
    add("direct_bin", P_bin, E_bin, {"method": "per-bin observable-space fit (1h, 2h response, mean, cross; kSZ, "
                                               "DM mean/variance), sigma 1%, eta 0.1", "R_fit": 4.0, "R_cut": 4.0})
    log("direct per-bin done")
    # ---------------- smooth log-M trend, joint P + E, all four families (quantiles linearised)
    xm = np.array([bd.logmass - 14.5 for bd in bds["P"]])
    starts = []
    for f in ("P", "E"):
        lp = np.array([_ln(fits[(f, "a", 4.0)][c]["parameters"]) for c in range(FC.N_CAL)])
        coef = np.polyfit(xm, lp, 1)                  # [2, 3]: slope, intercept
        starts.append(np.r_[coef[1], coef[0]])        # intercepts (3), slopes (3)
    theta0 = np.r_[starts[0], starts[1]]
    t = truth["obs"]
    keys = [("yytot", e) for e in FC.SEL_ELL] + [("ksz1h", e) for e in FC.SEL_ELL] + [("ydmtot", e) for e in FC.SEL_ELL]
    fam_w = {"yytot": 0.25, "ksz1h": 0.25, "ydmtot": 0.25}
    wl = {e: (0.5 if e == 8000 else 1.0) for e in FC.SEL_ELL}
    wsum = sum(wl.values())

    def build(theta):
        P = np.zeros((FC.N_CAL, FC.N_B))
        E = np.zeros((FC.N_CAL, FC.N_B))
        pars = []
        for c in range(FC.N_CAL):
            pp = np.exp(theta[0:3] + theta[3:6] * xm[c])
            pe = np.exp(theta[6:9] + theta[9:12] * xm[c])
            pp = np.clip(pp, *[np.asarray(b) for b in FF.FAM.M.BOUNDS["b12"]])
            pe = np.clip(pe, *[np.asarray(b) for b in FF.FAM.M.BOUNDS["b16"]])
            P[c] = FF.FAM.shell_model("b12", pp, FC.BASIS_EDGES) * (FC.BASIS_EDGES[1:] <= 4 + 1e-12)
            E[c] = FF.FAM.shell_model("b16", pe, FC.BASIS_EDGES) * (FC.BASIS_EDGES[1:] <= 4 + 1e-12)
            pars.append((pp, pe))
        return P, E, pars

    def res(theta):
        P, E, pars = build(theta)
        o = sb.global_obs(P, E)
        r = []
        for key, e in keys:
            r.append(np.sqrt(fam_w[key] * wl[e] / wsum) * (o[key][i_sel[e]] / t[f"{key}_{e}"] - 1) / FF_SIG)
        q = _linear_quantiles(E, truth)
        dm = [o["dm_mean"] / t["dm_mean"] - 1, o["dm_var"] / t["dm_var"] - 1] + \
             [q[p] / t[f"dm_{n}"] - 1 for p, n in enumerate(FC.Q_NAMES)]
        r.append(np.sqrt(0.25 / len(dm)) * np.asarray(dm) / FF_SIG)
        for c in range(FC.N_CAL):
            r.append(_reg_residuals(bds["P"][c], "b12", pars[c][0]) / np.sqrt(FC.N_CAL))
            r.append(_reg_residuals(bds["E"][c], "b16", pars[c][1]) / np.sqrt(FC.N_CAL))
        return np.concatenate([np.atleast_1d(v) for v in r])

    sol = least_squares(res, theta0, max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9, diff_step=1e-5)
    P, E, pars = build(sol.x)
    add("direct_trend", P, E, {"method": "joint P+E smooth log-M trend (ln p = a + b (logM - 14.5)), all four "
                                         "families, quantiles linearised (Mecke), sigma 1%, eta 0.1",
                               "theta": sol.x.tolist(), "success": bool(sol.success), "cost": float(sol.cost),
                               "nfev": int(sol.nfev)})
    for c in range(FC.N_CAL):
        for f, k in (("P", 0), ("E", 1)):
            records.append({"field": f, "method": "direct_trend", "R_fit": 4.0, "bin": FC.CAL_LABELS[c],
                            "family": FF.FIELD_FAMILY[f], "parameters": "; ".join(f"{v:.6g}" for v in pars[c][k]),
                            "success": bool(sol.success)})


FF_SIG = SIGMA_OBS
