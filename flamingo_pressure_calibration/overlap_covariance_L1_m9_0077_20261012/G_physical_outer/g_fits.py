"""G stage 'fits': every field model on the all-gas and the A2-ii stacks, objectives (a) and (d), every R_fit.

Order: E0/P0 and E1/P1 (Test F's functions, bin by bin) -> electron physical models (joint fits, worker pool)
-> pressure physical models (P2i needs E3's infall globals at R_fit = 4, objective a, same fit target).
For every fit: per-bin and global parameters, shell averages of the bound and outer components on the 42 basis
shells for every R_cut (hard cut), the painting variant x p_keep (all-gas fits), chi2, sigma_d (shrunk bootstrap
GLS covariance), region RMS against the fitted stack; then AIC / BIC per (field, target, objective, R_fit).
Diagnostics: E3pb (per-bin infall amplitude and slope; R_fit = 4, objective a, all gas) and the radius of the
steepest logarithmic slope of every physical fit.
Outputs: results/fits.npz, results/fits_summary.json, tables/G_fit_records.csv, tables/G_fit_globals.csv,
tables/G_aic_bic.csv.
"""
from __future__ import annotations

import time

import g_common as GC
import g_fit as GF
import g_models as GM
import g_physics as PH
import numpy as np

FF = GC.FF
_G: dict = {}
GLOBAL_STARTS = {
    "E2g6": [{}],
    "E2gf": [{"gamma_t": 6.0}, {"gamma_t": 3.0}],
    "E3": [{"gamma_t": 6.0, "A_inf": 1.0, "a_inf": 0.0, "s_inf": 1.5},
           {"gamma_t": 4.0, "A_inf": 0.3, "a_inf": 0.5, "s_inf": 2.5},
           {"gamma_t": 6.0, "A_inf": 0.5, "a_inf": -5.0, "s_inf": 2.0}],
    "E3pb": [{"gamma_t": 6.0}],
    "P2g6": [{"tau_0": 0.3, "q_tau": 1.0}, {"tau_0": 0.15, "q_tau": 0.3}],
    "P2gf": [{"gamma_t": 6.0, "tau_0": 0.3, "q_tau": 1.0}, {"gamma_t": 4.0, "tau_0": 0.15, "q_tau": 0.3}],
    "P2i": [{"gamma_t": 6.0, "tau_0": 0.3, "q_tau": 1.0}, {"gamma_t": 4.0, "tau_0": 0.15, "q_tau": 0.3}],
    "P2s": [{"gamma_t": 6.0, "tau_0": 0.3, "q_tau": 1.0, "x_sh": 4.0},
            {"gamma_t": 4.0, "tau_0": 0.15, "q_tau": 0.3, "x_sh": 3.0}],
    "P2m": [{"gamma_t": 6.0, "tau_0": 0.3, "q_tau": 1.0, "p_M": 0.3},
            {"gamma_t": 4.0, "tau_0": 0.15, "q_tau": 0.3, "p_M": 0.0}],
}


def context(prep, infall=None, templates=None):
    """Model context: Ebar, KAPPA_P, per-bin mean b and log M, b_pivot, E3 infall (P2i)."""
    t = templates or {"b": prep["bin_b"], "logm": prep["bin_logm"]}
    return {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"]), "b": np.asarray(t["b"], float),
            "logm": np.asarray(t["logm"], float), "b_pivot": float(prep["b_pivot"]), "infall": infall}


def core_params(fit, fallback):
    f = fit if fit is not None else fallback
    return np.asarray(f["parameters"], float)


# ============================================================================ worker
def _physical_job(task):
    field, name, target, objective, r_fit = task
    data = _G["data"][(field, target)]
    model = GM.get(name)
    ctx = dict(_G["ctx"])
    if model.uses_infall:
        ctx["infall"] = _G["infall"][target]
    rs = 2.0 if r_fit >= 2.0 else r_fit
    core = _G["core"][(field, target, objective, rs)]
    core_a2 = _G["core"][(field, target, "a", 2.0)]
    start = np.array([core_params(core[c], core_a2[c]) for c in range(GC.N_CAL)])
    t0 = time.perf_counter()
    res = GF.fit_joint(model, data, objective, r_fit, ctx, _G["x35"], _G["wq35"], _G["T2h35"], start,
                       global_starts=GLOBAL_STARTS[name])
    res["seconds"] = time.perf_counter() - t0
    res["key"] = GC.fit_key(field, name, target, objective, r_fit)
    return res


# ============================================================================ helpers
def params_from_F(fits, fallback, model):
    """Per-bin natural parameters [nb, kb] from Test F-style fit dicts (fallback: the (a) 2R fit)."""
    pb = np.zeros((GC.N_CAL, model.kb))
    ok = np.zeros(GC.N_CAL, bool)
    for c in range(GC.N_CAL):
        f = fits[c]
        if f is None or "parameters" not in f:
            p = list(fallback[c]["parameters"])[:3]
            if model.kb > 3:                                  # env fallback: core + negligible outer term
                p = p + [1e-6 if model.field == "E" else 1e-8, 1.0]
            pb[c] = p[:model.kb]
            continue
        pb[c] = np.asarray(f["parameters"], float)[:model.kb]
        ok[c] = bool(f.get("accepted", False))
    return pb, ok


def all_shells(model, pb, pg, ctx, pk=None):
    """Bound / outer shell averages on the basis shells for every R_cut (+ x p_keep); and uncut on the F shells."""
    x42, wq42, T42 = _G["x42"], _G["wq42"], _G["T2h42"]
    out = {"B": [], "O": [], "Bp": [], "Op": []}
    for rc in GC.R_CUT:
        B, O = GM.shells(model, pb, pg, ctx, x42, wq42, T42, cut=rc, edges=GC.BASIS_EDGES)
        out["B"].append(B)
        out["O"].append(O)
        if pk is not None:
            Bp, Op = GM.shells(model, pb, pg, ctx, x42, wq42, T42, cut=rc, edges=GC.BASIS_EDGES, factor=pk)
        else:
            Bp, Op = np.zeros_like(B), np.zeros_like(O)
        out["Bp"].append(Bp)
        out["Op"].append(Op)
    B35, O35 = GM.shells(model, pb, pg, ctx, _G["x35"], _G["wq35"], _G["T2h35"])
    return {k: np.array(v) for k, v in out.items()}, B35 + O35


def region_rms(S35, mean, base):
    rel = np.where(base & (mean > 0), S35 / np.where(mean > 0, mean, 1) - 1, np.nan)
    out = {}
    for name, m in GC.region_masks().items():
        v = rel[:, m]
        out[name] = float(np.sqrt(np.nanmean(v**2)) * 100) if np.isfinite(v).any() else np.nan
    per_bin = {}
    for name, m in GC.region_masks().items():
        v = rel[:, m]
        with np.errstate(invalid="ignore"):
            per_bin[name] = np.sqrt(np.nanmean(v**2, axis=1)) * 100
    return out, per_bin, rel


# ============================================================================ main
def run(pr, prep, imp, rows_by_bin=None, templates=None, pkeep_params=None, tasks_filter=None, gls=True,
        workers=None, log=GC.log, targets=GC.FIT_TARGETS):
    """Fit every model; returns a dict of fit results keyed by fit_key (also used by the CV stage with training
    rows / templates / p_keep)."""
    timer = GC.Timer()
    x35, wq35 = PH.quad_nodes(GC.F_EDGES)
    x42, wq42 = PH.quad_nodes(GC.BASIS_EDGES)
    T42 = (templates or prep)["T2h"]
    _G.clear()
    _G.update(x35=x35, wq35=wq35, x42=x42, wq42=wq42, T2h42=T42, T2h35=T42[:, :GC.N_F])
    ctx = context(prep, templates=None if templates is None else {"b": templates["b"], "logm": templates["logm"]})
    _G["ctx"] = ctx
    logm = ctx["logm"]
    pk_par = pkeep_params or {f: prep[f"pkeep_params_{f}"] for f in ("P", "E")}
    pknodes = {f: PH.pkeep_nodes(pk_par[f], logm) for f in ("P", "E")}
    wmap = {f: GF.field_weight_map(imp, f) for f in ("P", "E")}
    data = {}
    for f in ("P", "E"):
        for t in targets:
            data[(f, t)] = GF.FieldData(pr, f, t, wmap[f], rows_by_bin=rows_by_bin, gls=gls)
    _G["data"] = data
    log("field data ready", round(timer.mark("data"), 1))
    want = tasks_filter or (lambda f, m, t, o, r: True)
    results = {}
    # ---------------- model 0 and 1 (Test F's functions), all R_fit (2 and 4 always: starts)
    core, env = {}, {}
    rf_all = sorted(set(GC.R_FIT) | {2.0, 4.0})
    for f in ("P", "E"):
        for t in targets:
            d = data[(f, t)]
            for rf in rf_all:
                core[(f, t, "a", rf)] = GF.fit_core_F(d, "a", rf)
            for rf in rf_all:
                core[(f, t, "d", rf)] = GF.fit_core_F(d, "d", rf, fits_a=core[(f, t, "a", rf)])
            for o in GC.OBJECTIVES:
                for rf in GC.R_FIT:
                    if want(f, f"{f}1", t, o, rf):
                        env[(f, t, o, rf)] = GF.fit_env_F(d, o, rf, core[(f, t, "a", 4.0)], core[(f, t, "a", 2.0)])
    _G["core"] = core
    for (f, t, o, rf), fits in core.items():
        if rf not in GC.R_FIT or not want(f, f"{f}0", t, o, rf):
            continue
        m = GM.get(f"{f}0")
        pb, ok = params_from_F(fits, core[(f, t, "a", 2.0)], m)
        results[GC.fit_key(f, m.name, t, o, rf)] = {"model": m.name, "field": f, "objective": o, "r_fit": rf, "pb": pb,
                                                    "pg": np.zeros(0), "accepted_bins": ok, "F_fits": fits}
    for (f, t, o, rf), fits in env.items():
        m = GM.get(f"{f}1")
        fallback = [{"parameters": core_params(core[(f, t, "a", 2.0)][c], core[(f, t, "a", 4.0)][c])} for c in range(GC.N_CAL)]
        pb, ok = params_from_F(fits, fallback, m)
        results[GC.fit_key(f, m.name, t, o, rf)] = {"model": m.name, "field": f, "objective": o, "r_fit": rf, "pb": pb,
                                                    "pg": np.zeros(0), "accepted_bins": ok, "F_fits": fits}
    log("models 0/1 done", round(timer.mark("F-machinery fits"), 1))
    # ---------------- physical models: electrons, then pressure
    workers = GC.WORKERS if workers is None else workers
    e_tasks = [("E", m, t, o, rf) for m in GM.FIELD_MODELS["E"] if m in GM.PHYSICAL for t in targets
               for o in GC.OBJECTIVES for rf in GC.R_FIT if want("E", m, t, o, rf)]
    if want("E", "E3pb", "main", "a", 4.0) and tasks_filter is None:
        e_tasks.append(("E", "E3pb", "main", "a", 4.0))
    for res in GC.pool_map(_physical_job, e_tasks, workers, ordered=False):
        results[res["key"]] = res
    log(f"electron physical fits: {len(e_tasks)}", round(timer.mark("E fits"), 1))
    infall = {}
    for t in targets:
        k = GC.fit_key("E", "E3", t, "a", 4.0)
        if k in results:
            r = results[k]
            infall[t] = (r["pg"][r["names_g"].index("A_inf")], r["pg"][r["names_g"].index("a_inf")],
                         r["pg"][r["names_g"].index("s_inf")])
    _G["infall"] = infall
    p_tasks = [("P", m, t, o, rf) for m in GM.FIELD_MODELS["P"] if m in GM.PHYSICAL for t in targets
               for o in GC.OBJECTIVES for rf in GC.R_FIT if want("P", m, t, o, rf)
               and (m != "P2i" or t in infall)]
    for res in GC.pool_map(_physical_job, p_tasks, workers, ordered=False):
        results[res["key"]] = res
    log(f"pressure physical fits: {len(p_tasks)}", round(timer.mark("P fits"), 1))
    # ---------------- shells, residuals, sigma_d
    for key, res in results.items():
        f, name, t, o, rf = GC.parse_key(key)
        m = GM.get(name)
        c2 = dict(ctx)
        if m.uses_infall:
            c2["infall"] = infall[t]
        sh, S35 = all_shells(m, res["pb"], res["pg"], c2, pknodes[f] if t == "main" else None)
        res.update(sh)
        res["S35"] = S35
        d = data[(f, t)]
        res["rms"], res["rms_bins"], _ = region_rms(S35, d.mean, d.base)
        if gls:
            br = GF.bin_residuals(d, S35, rf)
            kg_share = m.kg / GC.N_CAL
            res["sigma_d"] = np.array([GF.sigma_d_solve(r, C, len(r) - (m.kb + kg_share)) for r, C, _ in br])
            res["_br"] = br
        res["infall_used"] = c2.get("infall")
    log("shells and sigma_d", round(timer.mark("post"), 1))
    return results, {"infall": infall, "core": core, "data": data, "ctx": ctx, "pknodes": pknodes,
                     "seconds": timer.marks}


def aic_bic(results):
    rows = []
    groups = {}
    for key, res in results.items():
        f, name, t, o, rf = GC.parse_key(key)
        if name not in GM.FIELD_MODELS[f]:
            continue
        groups.setdefault((f, t, o, rf), []).append(key)
    for (f, t, o, rf), keys in groups.items():
        sig = np.array([results[k]["sigma_d"] for k in keys])           # [models, bins]
        with np.errstate(all="ignore"):
            sref = np.nanmin(np.where(np.isfinite(sig), sig, np.nan), axis=0)
        sref = np.nan_to_num(sref, nan=0.0)
        for k in keys:
            res = results[k]
            m = GM.get(res["model"])
            chi2, n = 0.0, 0
            for c, (r, C, _) in enumerate(res["_br"]):
                if len(r) == 0:
                    continue
                chi2 += GF.chi2_inflated(r, C, sref[c]**2)
                n += len(r)
            kfree = m.n_free(GC.N_CAL)
            rows.append({"field": f, "fit_target": t, "objective": o, "R_fit": rf, "model": res["model"], "k": kfree,
                         "n_shells": n, "chi2_inflated": chi2, "AIC": chi2 + 2 * kfree, "BIC": chi2 + kfree * np.log(max(n, 1)),
                         "sigma_ref_median": float(np.median(sref)),
                         "sigma_d_median": float(np.nanmedian(res["sigma_d"])) if np.isfinite(res["sigma_d"]).any() else np.nan,
                         "sigma_d_max": float(np.nanmax(res["sigma_d"])) if np.isfinite(res["sigma_d"]).any() else np.nan})
    # deltas within each group
    by = {}
    for r in rows:
        by.setdefault((r["field"], r["fit_target"], r["objective"], r["R_fit"]), []).append(r)
    for grp in by.values():
        a0 = min(r["AIC"] for r in grp)
        b0 = min(r["BIC"] for r in grp)
        for r in grp:
            r["dAIC"] = r["AIC"] - a0
            r["dBIC"] = r["BIC"] - b0
    return rows


def steepest_slopes(results, prep, pr, lh=None):
    """Radius of the steepest log slope of each physical fit (x in 0.5-10 R200c), per bin."""
    lh = lh or PH.LinearHalo()
    xf = np.geomspace(0.5, 10.0, 400)
    T = np.zeros((GC.N_CAL, len(xf)))
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        T[c] = PH.two_halo_template(lh, pr["m200c_msun"][rows], pr["r200c_mpc"][rows], pr["weight"][rows], xf[None, :])[0]
    out = {}
    for key, res in results.items():
        f, name, t, o, rf = GC.parse_key(key)
        m = GM.get(name)
        if not m.physical:
            continue
        ctx = context(prep)
        if m.uses_infall:
            ctx["infall"] = res["infall_used"]
        b, oo = m.nodes(res["pb"], res["pg"], ctx, xf[None, :], T[:, None, :])
        tot = (b + oo)[:, 0, :]
        slope = np.gradient(np.log(tot), np.log(xf), axis=1)
        k = np.argmin(slope, axis=1)
        out[key] = (xf[k], slope[np.arange(GC.N_CAL), k])
    return out


def main():
    timer = GC.Timer()
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    imp = dict(np.load(GC.FBASE / "importance_F.npz"))
    prep = GC.load_prepare()
    results, info = run(pr, prep, imp)
    slopes = steepest_slopes(results, prep, pr)
    rows_aic = aic_bic(results)
    # ---------------- records
    rec, glob = [], []
    for key, res in sorted(results.items()):
        f, name, t, o, rf = GC.parse_key(key)
        m = GM.get(name)
        nb_, ng_ = m.names()
        for c in range(GC.N_CAL):
            row = {"key": key, "field": f, "model": name, "fit_target": t, "objective": o, "R_fit": rf,
                   "bin": GC.CAL_LABELS[c], **{f"p_{n}": res["pb"][c, k] for k, n in enumerate(nb_)},
                   "sigma_d": res["sigma_d"][c], **{f"rms_{n}": v[c] for n, v in res["rms_bins"].items()}}
            if "chi2_bins" in res:
                row["chi2"] = res["chi2_bins"][c]
                row["n_shells"] = int(res["n_shells"][c])
                row["at_bound"] = ";".join(n for k, n in enumerate(nb_) if res["at_bound_b"][c, k])
            if "accepted_bins" in res:
                row["accepted"] = bool(res["accepted_bins"][c])
                ff = res["F_fits"][c]
                if ff is not None:
                    for k2 in ("chi2_per_dof", "rms_percent", "jacobian_condition", "bound_distance", "n_shells"):
                        if k2 in ff and ff[k2] is not None:
                            row[f"F_{k2}"] = ff[k2]
            if key in slopes:
                row["x_steepest"] = slopes[key][0][c]
                row["slope_steepest"] = slopes[key][1][c]
            rec.append(row)
        g = {"key": key, "field": f, "model": name, "fit_target": t, "objective": o, "R_fit": rf,
             **{f"g_{n}": res["pg"][k] for k, n in enumerate(ng_)}, **{f"rms_{n}": v for n, v in res["rms"].items()},
             "sigma_d_median": float(np.nanmedian(res["sigma_d"])) if np.isfinite(res["sigma_d"]).any() else np.nan}
        for k2 in ("cost", "success", "nfev", "jacobian_condition", "accepted", "seconds"):
            if k2 in res:
                g[k2] = res[k2]
        if "at_bound_g" in res:
            g["globals_at_bound"] = ";".join(n for k, n in enumerate(ng_) if res["at_bound_g"][k])
        if res.get("infall_used") is not None:
            g["infall_used"] = "; ".join(f"{v:.6g}" for v in res["infall_used"])
        glob.append(g)
    GC.write_csv(GC.TABLES / "G_fit_records.csv", rec)
    GC.write_csv(GC.TABLES / "G_fit_globals.csv", glob)
    GC.write_csv(GC.TABLES / "G_aic_bic.csv", rows_aic)
    keys = sorted(results)
    np.savez_compressed(GC.RESULTS / "fits.npz", keys=np.array(keys), r_cut=np.array(GC.R_CUT),
                        B=np.array([results[k]["B"] for k in keys]), O=np.array([results[k]["O"] for k in keys]),
                        Bp=np.array([results[k]["Bp"] for k in keys]), Op=np.array([results[k]["Op"] for k in keys]),
                        S35=np.array([results[k]["S35"] for k in keys]),
                        sigma_d=np.array([results[k]["sigma_d"] for k in keys]),
                        params_json=np.array(GC.json.dumps(GC.plain({k: {"pb": results[k]["pb"], "pg": results[k]["pg"],
                                                                         "names": GM.get(results[k]["model"]).names(),
                                                                         "infall_used": results[k].get("infall_used")}
                                                                     for k in keys}))))
    GC.write_json(GC.RESULTS / "fits_summary.json", {"n_fits": len(keys), "infall_E3_a4": info["infall"],
                                                     "seconds": info["seconds"], "total_seconds": timer.mark("total")})
    GC.log("fits done", round(timer.mark("total"), 1))


if __name__ == "__main__":
    main()
