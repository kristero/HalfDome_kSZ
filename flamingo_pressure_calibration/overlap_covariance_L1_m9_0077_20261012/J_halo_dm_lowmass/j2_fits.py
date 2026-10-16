"""Test J2 fitting stages on Test F's 35-shell grid, in Test G's configuration (one process per stage, run by
run_test_j2.py), with the calibrated support extended to the 14 bins 12.7-15.5.

Configuration (before any F / G import; nothing is written into F_observables/ or G_physical_outer/):
  G_RESULTS = <J2 results>/work/g35, F_RESULTS = <J2 results>/work/g35/f_base, F/G_MODE = J's mode;
  f_common.SUPPORT / CAL_BINS / N_CAL / CAL_LABELS set on the imported module object to the 14 J bins (j2_common.J_BINS)
  BEFORE g_common is imported (g_common copies them), exactly as j_common.set_support does for Test J1.
Data: inputs/j2_fitdata.npz (j2_inputs.py): per-halo 35-shell profiles, the fit masks (strict rule >= 13.5, S400 rule
12.7-13.5, see j2_common), the RC grids of the low-mass bins and the importance-derived fit weights of each definition.

Stage 'fits' (J2a), per definition D in D0-D3 and fit radius R_fit (j2_common.DEFS), per calibrated bin c:
  B16 electrons  E(x) = rho0 q^-0.2 (1 + q^alpha)^-((beta - 0.2)/alpha), q = x/0.5          (rho0, alpha, beta)
  B12 pressure   P(x) = 0.5176 P0 q^-0.3 (1 + q)^-beta,             q = x/x_c                (P0, x_c, beta)
  objective (a)  chi2 = sum_s [(ln S_s - ln m_s)/sigma_s]^2, sigma_s = max(SD_s/m_s, 0.05)
  objective (d)  chi2 = sum_s w_s (ln S_s - ln m_s)^2, w_s = J1's DM-family importance of D summed into the fit shell,
                 normalised per bin (only for R_fit <= the boundary of D: beyond it w = 0); 'dI' (D0 only): Test I's
                 recipe w = 0.5 w_kSZ + 0.5 w_DM
  objective (L)  chi2 = r^T (C_L22/h)^-1 r, r = ln S - ln m, C_L22 = inclusion-weighted sample covariance of the haloes'
                 ln F / N_eff, h = (N - p - 2)/(N - 1) (Hartlap; shells where any halo has F <= 0 dropped; bins with
                 N <= p + 2 have no L fit: painted with the (a) fit, 'L*')
  (S_s = exact volume average of the model over shell s (32-point Gauss-Legendre), m_s = inclusion-weighted stacked
  mean, SD_s = reliability-weighted halo-to-halo standard deviation; least squares in ln parameters, analytic Jacobian)
  plus: the RC variant of the low-mass bins ((a), (d) at R_fit = boundary, exact averages over the RC shells) and
  B12 (a) for pressure at R_fit = 4 (all 14 bins; A2-ii).
  Parity: the A2-ii fits of the bins >= 13.5 against Test I's fits35.json (same data, mask and objective).
Stage 'ladder' (J2b): Test G's Lee22 ladder (g_selection: Cells, covariances, pb_start, _model_job with the three
  likelihoods and the spatial CV, summarise with G's selection rule), electrons only, per definition on its own data,
  mass and concentration cells, at the definition's ladder radii and outer options (O0 inside a boundary; O0/O1/O2
  for D0 and D3).  The free break mass log10 M_cut is bounded to [12.9, 15.3] instead of G's [13.5, 15.3] because the
  calibrated range now starts at 10^12.7 (the lowest bin stays below any allowed break).  The sigma_d floor reference
  is PB with the best of the definition's outer options.  Parity run (full mode): D0 on Test I's 10 bins, R_fit 4,
  with G's original bound, against Test I's ladder (I2_ladder_models.csv, field E, R_fit 4).
Outputs (J2 results): fits.json, fits_parity.json, ladder_summary.json, ladder_<D>.npz (results_json),
  tables/J2a_fits.csv, tables/J2b_ladder_models_<D>.csv, J2b_ladder_ladder_<D>.csv, J2b_ladder_cells_<D>.csv,
  ladder_parity.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import j2_common as C  # noqa: E402

G35 = C.J2_RESULTS / "work" / "g35"
MCUT_LO = 12.9
_STATE = {"mcut_lo": MCUT_LO}


def configure(workers):
    (G35 / "f_base").mkdir(parents=True, exist_ok=True)
    os.environ["F_MODE"] = C.MODE
    os.environ["G_MODE"] = C.MODE
    os.environ["G_RESULTS"] = str(G35)
    os.environ["F_RESULTS"] = str(G35 / "f_base")
    os.environ["G_WORKERS"] = str(workers)
    os.environ["F_WORKERS"] = str(workers)
    os.environ["STUDY_ROOT"] = str(C.ROOT)
    for sub in ("F_observables", "G_physical_outer"):
        p = str(C.ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import f_common as FC
    set_bins(FC, C.J_BINS)
    import g_common as GC
    assert list(GC.CAL_BINS) == list(C.J_BINS), (GC.CAL_BINS, C.J_BINS)
    assert np.max(np.abs(np.asarray(GC.F_EDGES) - C.F35)) < 1e-7
    return FC, GC


def set_bins(FC, bins):
    mask = np.zeros(len(FC.SUPPORT["support"]), bool)
    mask[list(bins)] = True
    FC.SUPPORT["support"] = mask
    FC.CAL_BINS = list(bins)
    FC.N_CAL = len(bins)
    FC.CAL_LABELS = [FC.bin_label(b) for b in bins]
    gc = sys.modules.get("g_common")
    if gc is not None:
        gc.CAL_BINS, gc.N_CAL, gc.CAL_LABELS = FC.CAL_BINS, FC.N_CAL, FC.CAL_LABELS


# ============================================================================ fits (J2a)
def fit_rec(fit, extra=None):
    if fit is None:
        return None
    rec = {k: fit.get(k) for k in ("chi2", "chi2_per_dof", "n_shells", "rms_percent", "accepted", "success")}
    rec["parameters"] = np.asarray(fit["parameters"], float).tolist()
    m = np.asarray(fit.get("mask"), bool)
    rec["mask"] = m.tolist() if m.ndim else None
    rec.update(extra or {})
    return rec


def l22_cov(values, w):
    """Test I's l22_cov: C_L22 / N_eff (NaN-free; shells with any F <= 0 flagged in ok), N, ok."""
    v = np.asarray(values, float)
    w = np.asarray(w, float) / np.sum(w)
    ok = np.all(np.isfinite(v) & (v > 0), axis=0)
    L = np.log(np.where(v > 0, v, 1.0))
    mu = w @ L
    dd = L - mu
    Cs = (dd.T * w) @ dd / max(1 - np.sum(w**2), 1e-12)
    neff = 1 / np.sum(w**2)
    return Cs / neff, len(v), ok


def stage_fits():
    FC, GC = configure(1)
    FF = GC.FF
    AF = FF.F
    d = C.load_npz(C.J2_INPUT)
    F35 = d["f35_edges"]
    X = d["x200_edges"]
    w = d["weight"]
    mb = d["mass_bin"]
    out = {"fits": {}, "rc": {}, "definitions": __doc__, "f35_edges": F35, "j_bins": C.J_BINS}
    rows_csv = []

    def bindata(values, mask_b, rows):
        mean, sd, neff = FC.S.weighted_stack(values[rows], w[rows])
        base = np.asarray(mask_b, bool) & (mean > 0)
        return types.SimpleNamespace(mean=mean, sd=sd, base=base, values=values[rows], w=w[rows], n=len(rows),
                                     mask=lambda r, base=base: base & FC.shells_within(F35, r))

    def one(fam, bd, obj, rf, weights=None, start=None):
        if obj == "a":
            bd.family = fam
            return FF.fit_objective(bd, "a", rf, start=start)
        if obj in ("d", "dI"):
            bd.family = fam
            return FF.fit_objective(bd, "d", rf, start=start, weights=weights)
        C_, N, okL = l22_cov(bd.values, bd.w)
        mask = bd.mask(rf) & okL
        p = int(mask.sum())
        h = (N - p - 2) / (N - 1) if N > p + 2 else np.nan
        if not (np.isfinite(h) and p >= 4):
            return None, {"hartlap": h, "N": N, "p": p}
        try:
            return AF.fit_general(fam, bd.mean, mask, cov=C_ / h, start=start, edges=F35), {"hartlap": h, "N": N,
                                                                                            "p": p}
        except Exception as exc:                          # noqa: BLE001
            C.log("L fit failed", repr(exc))
            return None, {"hartlap": h, "N": N, "p": p, "error": repr(exc)}

    jobs = []
    for D, spec in C.DEFS.items():
        sel = spec["sel"]
        vals = d[f"E_{sel}_35"]
        for rf in spec["rfit"]:
            objs = ["a", "L"] + (["d"] if rf <= spec["R"] + 1e-9 else [])
            if D == "D0" and rf >= 3.0:
                objs.append("dI")
            jobs.append((D, sel, vals, rf, objs))
    for D, sel, vals, rf, objs in jobs:
        for c, b in enumerate(C.J_BINS):
            rows = np.flatnonzero(mb == b)
            bd = bindata(vals, d[f"mask35_{sel}"][b], rows)
            fa = one("b16", bd, "a", rf)
            st = fa["parameters"] if fa is not None else None
            for obj in objs:
                extra = {"bin": C.bin_label(b), "engine_bin": b, "D": D, "R_fit": rf, "objective": obj, "rule": "S400"
                         if b < 5 else "strict"}
                if obj == "a":
                    f = fa
                elif obj in ("d", "dI"):
                    wk = "w35_DM_" + D if obj == "d" else "w35_dI_D0"
                    f = one("b16", bd, obj, rf, weights=d[wk][c], start=st)
                else:
                    f, info = one("b16", bd, "L", rf, start=st)
                    extra.update(info)
                rec = fit_rec(f, extra)
                out["fits"][f"{D}|{obj}|{rf:g}|{b}"] = rec
                if rec is not None:
                    rows_csv.append({k: v for k, v in rec.items() if k != "mask"})
        C.log("fits", D, rf)
    # ---- pressure B12 (a), R_fit 4, A2-ii
    for c, b in enumerate(C.J_BINS):
        rows = np.flatnonzero(mb == b)
        bd = bindata(d["P_a2ii_35"], d["mask35_P"][b], rows)
        for rf in (4.0,):
            f = one("b12", bd, "a", rf)
            rec = fit_rec(f, {"bin": C.bin_label(b), "engine_bin": b, "D": "P", "R_fit": rf, "objective": "a",
                              "rule": "S400" if b < 5 else "strict"})
            out["fits"][f"P|a|{rf:g}|{b}"] = rec
            if rec is not None:
                rows_csv.append({k: v for k, v in rec.items() if k != "mask"})
    # ---- RC variant (low bins): [0, x_h] + T1 groups outside x_h, objectives (a), (d) at R_fit = boundary
    for D, spec in C.DEFS.items():
        sel = spec["sel"]
        rf = spec["R"]
        for b in C.LOW_BINS:
            rows_b = d[f"rows_bin{b}"]
            eidx = d[f"rc_edges_{sel}_{b}"]
            edges = X[eidx]
            vals = d[f"rc_E_{sel}_{b}"]
            mean, sd, _ = FC.S.weighted_stack(vals, w[rows_b])
            mask = (mean > 0) & C.inside(edges, rf)
            rec_info = {"bin": C.bin_label(b), "engine_bin": b, "D": D, "R_fit": rf, "rule": "RC",
                        "rc_edges": edges.tolist(), "n_points": int(mask.sum()), "x_h": float(d["x_h"][b])}
            fa = AF.objective_a("b16", mean, sd, mask, edges=edges, jac="analytic")
            out["rc"][f"{D}|a|{rf:g}|{b}"] = fit_rec(fa, dict(rec_info, objective="a"))
            if fa is None:
                out["rc"][f"{D}|a|{rf:g}|{b}"] = dict(rec_info, objective="a", parameters=None,
                                                      note=f"{int(mask.sum())} RC shells < {AF.MIN_SHELLS}: no fit")
            wv = d[f"wrc_DM_{D}_{b}"]
            fd = None
            if fa is not None:
                fd = AF.objective_d("b16", mean, wv, mask, start=fa["parameters"], edges=edges)
            out["rc"][f"{D}|d|{rf:g}|{b}"] = fit_rec(fd, dict(rec_info, objective="d")) if fd is not None else \
                dict(rec_info, objective="d", parameters=None, note="no (a) fit / too few RC shells")
            for key in (f"{D}|a|{rf:g}|{b}", f"{D}|d|{rf:g}|{b}"):
                r = out["rc"][key]
                rows_csv.append({k: v for k, v in r.items() if k not in ("mask", "rc_edges")})
    C.write_json(C.J2_RESULTS / "fits.json", out)
    C.write_csv(C.J2_RESULTS / "tables" / "J2a_fits.csv", rows_csv)
    # ---- parity with Test I (A2-ii, bins >= 13.5; full mode only)
    par = {}
    ifits = C.I_RESULTS / "fits35.json"
    if not C.SMOKE and ifits.exists():
        ref = C.read_json(ifits)["fits"]
        worst = {}
        for key_j, key_i in (("D0|a|{r}|{b}", "E|a|{r}|{c}"), ("D0|L|{r}|{b}", "E|L|{r}|{c}"),
                             ("D1|a|{r}|{b}", "E|a|{r}|{c}"), ("D2|a|{r}|{b}", "E|a|{r}|{c}"),
                             ("P|a|{r}|{b}", "P|a|{r}|{c}")):
            for c, b in enumerate(range(5, 15)):
                for r in ("1", "1.34", "2", "3", "4"):
                    a = out["fits"].get(key_j.format(r=r, b=b))
                    i_ = ref.get(key_i.format(r=r, c=c))
                    if a is None or i_ is None or a.get("parameters") is None or i_.get("parameters") is None:
                        continue
                    rel = float(np.max(np.abs(np.asarray(a["parameters"]) / np.asarray(i_["parameters"]) - 1)))
                    tag = key_j.split("|")[0] + "|" + key_j.split("|")[1] + f"|{r}"
                    worst[tag] = max(worst.get(tag, 0.0), rel)
        par = {"max_rel_parameter_difference_vs_TestI_fits35": worst,
               "max_overall": max(worst.values()) if worst else None,
               "note": "J2 'D*|a|R' on bins 13.5-15.5 = Test I 'E|a|R' (same 35-shell A2-ii data, strict mask, objective)"}
    C.write_json(C.J2_RESULTS / "fits_parity.json", par)
    C.log("fits done; parity", par.get("max_overall"))


# ============================================================================ ladder (J2b)
def stage_ladder(workers):
    FC, GC = configure(workers)
    import g_physics as PH
    import g_selection as GS
    orig_build = GS.build_model

    def build_model_j2(field, spec, outer, n_cells, frozen=None):
        m = orig_build(field, spec, outer, n_cells, frozen=frozen)
        i = m.ix("log10Mcut")
        if i is not None:
            m.lo[i] = _STATE["mcut_lo"]
        return m

    GS.build_model = build_model_j2
    d = C.load_npz(C.J2_INPUT)
    prep = np.load(C.G_PREPARE)
    ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
    lh = PH.LinearHalo()
    conc = np.asarray(d["concentration_200c"], float)
    summary = {"sigma_floor": {}, "chosen": {}, "mcut_bounds": {}, "seconds": {}, "definitions": __doc__}
    runs = [(D, spec["sel"], spec["lad_rfit"], spec["outers"], list(C.J_BINS), MCUT_LO) for D, spec in C.DEFS.items()]
    if not C.SMOKE:
        runs.append(("D0parityI", "a2ii", (4.0,), ("O0", "O1", "O2"), list(range(5, 15)), 13.5))
    for D, sel, rfits, outers, bins, mlo in runs:
        t0 = time.perf_counter()
        set_bins(FC, bins)
        _STATE["mcut_lo"] = mlo
        pr = {"E_main": d[f"E_{sel}_35"], "weight": d["weight"], "mass_bin": d["mass_bin"],
              "m200c_msun": d["m200c"], "r200c_mpc": d["r200c"], "fold": d["fold"],
              "halo_rows": d["rows"], "base_mask": d[f"mask35_{sel}"], "strict_mask": d[f"mask35_{sel}"]}
        cells, covs, pbstart, cell_rows = {}, {}, {}, []
        for dset in ("mass", "massc"):
            cs = GS.Cells(pr, "E", dset, conc, lh)
            cells[("E", dset)] = cs
            for rf in rfits:
                covs[("E", dset, rf)] = GS.covariances(cs, rf)
                pbstart[("E", dset, rf)] = GS.pb_start(cs, covs[("E", dset, rf)], rf, ctx, "O0", "E")
                for i, cv in enumerate(covs[("E", dset, rf)]):
                    cell_rows.append({"D": D, "dset": dset, "R_fit": rf, "cell": i, "bin": GC.CAL_LABELS[cs.bin[i]],
                                      "N": cv["N"], "N_eff": cs.neff[i], "logM": cs.logm[i],
                                      "c_mean": float(np.exp(cs.lnc[i])), "shells": cv["p"], "hartlap": cv["hartlap"],
                                      "L22_used": cv["C_L22"] is not None, "shrink_lambda": cv["shrink_lambda"],
                                      "pooled_only": cv["pooled_only"]})
        GS._W.clear()
        GS._W.update(cells=cells, covs=covs, ctx=ctx, pbstart=pbstart, pr=pr, conc=conc)
        sig_tasks = [("E", ds, rf, tuple(outers) if rf > 1.34 + 1e-9 else ("O0",)) for ds in ("mass", "massc")
                     for rf in rfits]
        sigma = {}
        for key, res, best in GC.pool_map(_sigma_job_j2, sig_tasks, ordered=False):
            sigma[key] = {"by_outer": {o: {"sigma_d": v["sigma_d"], "iterations": v["iterations"]}
                                       for o, v in res.items()}, "reference_outer": best,
                          "sigma_ref": res[best]["sigma_d"]}
        GS._W["sigma"] = sigma
        tasks = []
        for (f, ds, rf, outs) in sig_tasks:
            for o in outs:
                for spec in GS.model_specs(f, ds):
                    tasks.append((f, ds, rf, o, spec[0]))
        results = [r for r in GC.pool_map(GS._model_job, tasks, ordered=False)]
        GS._W.clear()
        rows, ladder, chosen = GS.summarise(results, sigma)
        for r in rows:
            r["D"] = D
        for r in ladder:
            r["D"] = D
        C.write_csv(C.J2_RESULTS / "tables" / f"J2b_ladder_models_{D}.csv", rows)
        C.write_csv(C.J2_RESULTS / "tables" / f"J2b_ladder_ladder_{D}.csv", ladder)
        C.write_csv(C.J2_RESULTS / "tables" / f"J2b_ladder_cells_{D}.csv", cell_rows)
        np.savez_compressed(C.J2_RESULTS / f"ladder_{D}.npz",
                            results_json=np.array(json.dumps(C.plain(results))),
                            bins=np.array(bins))
        summary["sigma_floor"][D] = {f"{k[1]}|{k[2]:g}": v for k, v in sigma.items()}
        summary["chosen"][D] = chosen
        summary["mcut_bounds"][D] = [mlo, 15.3]
        summary["seconds"][D] = time.perf_counter() - t0
        C.log(f"ladder {D}: {len(results)} models, {summary['seconds'][D]:.0f} s; chosen", chosen)
        C.write_json(C.J2_RESULTS / "ladder_summary.json", summary)
    set_bins(FC, C.J_BINS)
    # ---- parity with Test I's ladder (E, R_fit 4, 10 bins)
    par = {}
    iref = C.I_RESULTS / "tables" / "I2_ladder_models.csv"
    if not C.SMOKE and iref.exists():
        import csv
        ref = {}
        with open(iref) as fh:
            for r in csv.DictReader(fh):
                if r["field"] == "E" and abs(float(r["R_fit"]) - 4.0) < 1e-9:
                    ref[(r["dset"], r["model"])] = r
        mine = {}
        with open(C.J2_RESULTS / "tables" / "J2b_ladder_models_D0parityI.csv") as fh:
            for r in csv.DictReader(fh):
                mine[(r["dset"], r["model"])] = r
        worst = {}
        for key, r in mine.items():
            q = ref.get(key)
            if q is None:
                continue
            for col in ("chi2_floor", "chi2_stat", "chi2_L22", "sigma_floor", "cv_chi2"):
                try:
                    a, b = float(r[col]), float(q[col])
                except (KeyError, ValueError):
                    continue
                if b != 0 and np.isfinite(a) and np.isfinite(b):
                    worst[col] = max(worst.get(col, 0.0), abs(a / b - 1))
        sel_i = C.read_json(C.I_RESULTS / "ladder_selection_summary.json")["chosen"]
        par = {"max_rel_vs_TestI_I2_ladder_models_E_Rfit4": worst, "models_compared": len(mine),
               "selected_J2": summary["chosen"]["D0parityI"], "selected_TestI": {k: v for k, v in sel_i.items()
                                                                                 if k.startswith("E|") and k.endswith("|4")}}
    C.write_json(C.J2_RESULTS / "ladder_parity.json", par)


def _sigma_job_j2(task):
    """g_selection._sigma_job with the definition's outer options (the per-bin reference PB / PBcell with the best of
    them sets sigma_d)."""
    import g_selection as GS
    field, dset, r_fit, outers = task
    W = GS._W
    cells, covs, ctx = W["cells"][(field, dset)], W["covs"][(field, dset, r_fit)], W["ctx"]
    pbpar = W["pbstart"][(field, dset, r_fit)]
    spec = [s for s in GS.model_specs(field, dset) if s[0] == ("PB" if dset == "mass" else "PBcell")][0]
    out = {}
    for o in outers:
        model = GS.build_model(field, spec, o, cells.n)
        starts = GS.starts_for(model, pbpar, cells, field)
        s2, theta = 0.0, None
        for it in range(25):
            prob = GS.Problem(model, cells, covs, "floor", s2, ctx, r_fit)
            sol = prob.fit(starts if theta is None else [theta] + starts[:1])
            theta = sol.x
            nu = prob.n_data - model.k
            new = GS.solve_sigma(cells, covs, GS.raw_residuals(model, theta, cells, covs, r_fit, ctx), nu)
            done = abs(np.sqrt(new) - np.sqrt(s2)) <= 1e-5 * max(np.sqrt(new), 1e-6)
            s2 = new
            if done:
                break
        out[o] = {"sigma_d": float(np.sqrt(s2)), "iterations": it + 1, "theta": theta}
    best = min(out, key=lambda o: out[o]["sigma_d"])
    return (field, dset, r_fit), out, best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fits", "ladder"))
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    (C.J2_RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if a.stage == "fits":
        stage_fits()
    else:
        stage_ladder(a.workers)
    C.log(f"{a.stage} done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
