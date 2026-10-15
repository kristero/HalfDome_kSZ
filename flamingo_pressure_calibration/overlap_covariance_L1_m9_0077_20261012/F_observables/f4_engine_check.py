"""Engine-exact check of the production basis and of the edge convention (kernels/engine.py, read-only).

For the objective-(a) B12/B16 fits through 2R and 4R (cut at 4R) the continuous profiles are propagated
by the engine itself (FitProfile, default fine grid, hard cut at 4R) and compared with the in-basis
prediction of the same fits (their shell averages on the truth-shaped basis): this measures the basis
representation error of smooth model profiles.  The same fits with the flux-matched 0.05-dex logistic
edge (per-bin x_t' at z = 0, Test C convention; fallback haloes keep the hard 4R cut through a step window)
are run on an extended grid and compared with the hard cut, in the engine and in the basis.
Output: products/engine_check.json, tables/F_engine_check.csv.
"""
from __future__ import annotations

import csv

import f_basis as FB
import f_common as FC
import f_fit as FF
import numpy as np

E = FC.E
WIDTH = 0.05


def records(path=FC.TABLES / "F_fit_records.csv"):
    out = {}
    with open(path) as handle:
        for r in csv.DictReader(handle):
            if r["method"] == "a" and r.get("parameters"):
                out[(r["field"], float(r["R_fit"]), r["bin"])] = np.array([float(v) for v in r["parameters"].split(";")])
    return out


def main():
    timer = FC.Timer()
    rec = records()
    B = FB.Basis("T")
    s = FC.SUPPORT
    sky = FC.make_sky()
    variants, inbasis, pairs = [], {}, []
    cal_of_mass = np.full(len(sky.mass), -1)
    lm = np.log10(sky.mass)
    for c, b in enumerate(FC.CAL_BINS):
        inside = (lm >= FC.MASS_EDGES[b]) & (lm < FC.MASS_EDGES[b + 1]) & (lm <= FC.MAX_LOGMASS)
        cal_of_mass[inside] = c
    xt_max = 1.0
    params = {}
    for rf in (2.0, 4.0):
        for f, fam in (("P", "b12"), ("E", "b16")):
            p = np.zeros((15, 3))
            p[:] = np.asarray(FF.FAM.get(fam).start, float)
            for c, b in enumerate(FC.CAL_BINS):
                p[b] = rec[(f, rf, FC.CAL_LABELS[c])]
            params[(f, rf)] = p
            xts = [FF.flux_matched_edge(fam, p[b], 4.0, WIDTH) for b in FC.CAL_BINS]
            params[(f, rf, "xt")] = np.array(xts)
            xt_max = max(xt_max, max(xts))
    end = xt_max * 10**(10 * WIDTH)
    grid_hard = E.make_grid(4.)
    grid_ext = E.make_grid(end, breaks=[4.0], linear=[(xt_max / 10**(4 * WIDTH), xt_max * 10**(4 * WIDTH), 81)])
    for rf in (2.0, 4.0):
        for f, field in (("P", 0), ("E", 1)):
            prof = E.FitProfile(field, params[(f, rf)], f"{f}{rf:g}", s["mass_edges"], s["support"], s["max_logmass"])
            variants.append(E.Variant(f"{f}{rf:g}_hard", prof, grid_hard, 4.0))
            variants.append(E.Variant(f"{f}{rf:g}_hardext", prof, grid_ext, 4.0))
            window = np.where(grid_ext.x[None, :] <= 4.0 * (1 + 1e-12), 1.0, 0.0) * np.ones((len(sky.mass), 1))
            for c in range(FC.N_CAL):
                rows = cal_of_mass == c
                window[rows] = E.sigmoid_window(grid_ext.x, params[(f, rf, "xt")][c], WIDTH)[None, :]
            variants.append(E.Variant(f"{f}{rf:g}_sigmoid", prof, grid_ext, None, window=window))
            fam = FF.FIELD_FAMILY[f]
            hard = np.array([FF.shells_from_fit(fam, params[(f, rf)][b], 4.0) for b in FC.CAL_BINS])
            sig = np.array([FF.shells_from_fit(fam, params[(f, rf)][b], window=(params[(f, rf, "xt")][c], WIDTH))
                            for c, b in enumerate(FC.CAL_BINS)])
            inbasis[(f, rf, "hard")] = hard
            inbasis[(f, rf, "sigmoid")] = sig
        pairs += [(f"P{rf:g}_hard", f"E{rf:g}_hard"), (f"P{rf:g}_sigmoid", f"E{rf:g}_sigmoid")]
    result = E.Engine(sky).run(variants, ell=FC.ELL, workers=FC.WORKERS, pairs=pairs, verbose=False)
    labels = list(result["labels"])
    plabels = list(result["pair_labels"])
    rows, out = [], {"timing": result["timing"], "x_t_flux_matched": {f"{k[0]}{k[1]:g}": v for k, v in params.items()
                                                                      if len(k) == 3}}
    for rf in (2.0, 4.0):
        for edge in ("hard", "sigmoid"):
            Pm, Em = inbasis[("P", rf, edge)], inbasis[("E", rf, edge)]
            o = B.observables(Pm, Em)
            iP, iE = labels.index(f"P{rf:g}_{edge}"), labels.index(f"E{rf:g}_{edge}")
            ix = plabels.index(f"P{rf:g}_{edge}xE{rf:g}_{edge}")
            eng = {"yy1h": result["cl_1h"][iP], "yy2h": result["cl_2h"][iP], "ksz1h": result["ksz_cl_1h"][iE],
                   "ksz2h": result["ksz_cl_2h_approx"][iE], "dmcl1h": result["dm_cl_1h"][iE],
                   "dmcl2h": result["dm_cl_2h"][iE], "ydm1h": result["cross_cl_1h"][ix], "ydm2h": result["cross_cl_2h"][ix],
                   "ymean": result["mean"][iP], "dm_mean": result["dm_mean"][iE], "dm_var": result["dm_var"][iE]}
            for k, v in eng.items():
                for e, i in (zip(FC.SEL_ELL, FC.SEL_INDEX) if np.ndim(v) else [(None, None)]):
                    ev = float(v[i]) if i is not None else float(v)
                    bv = float(o[k][i]) if i is not None else float(o[k])
                    rows.append({"fit": f"a_Rf{rf:g}", "edge": edge, "observable": k if e is None else f"{k}_{e}",
                                 "engine": ev, "basis": bv, "basis_over_engine_minus_1": bv / ev - 1})
            for p, qn in enumerate(FC.Q_NAMES):
                rows.append({"fit": f"a_Rf{rf:g}", "edge": edge, "observable": f"dm_{qn}",
                             "engine": float(result["dm_quantiles"][iE][p]), "basis": np.nan,
                             "basis_over_engine_minus_1": np.nan})
    # sigmoid / hard in the engine and in the basis
    for rf in (2.0, 4.0):
        for k_e, k_b in (("cl_total", "yytot"), ("ksz_cl_1h", "ksz1h"), ("dm_cl_1h", "dmcl1h")):
            field = "P" if k_e == "cl_total" else "E"
            i_s, i_h = labels.index(f"{field}{rf:g}_sigmoid"), labels.index(f"{field}{rf:g}_hardext")
            ratio_e = result[k_e][i_s] / result[k_e][i_h] - 1
            ob_s = B.observables(inbasis[("P", rf, "sigmoid")], inbasis[("E", rf, "sigmoid")])
            ob_h = B.observables(inbasis[("P", rf, "hard")], inbasis[("E", rf, "hard")])
            ratio_b = ob_s[k_b] / ob_h[k_b] - 1
            for e, i in zip(FC.SEL_ELL, FC.SEL_INDEX):
                rows.append({"fit": f"a_Rf{rf:g}", "edge": "sigmoid/hard", "observable": f"{k_b}_{e}",
                             "engine": float(ratio_e[i]), "basis": float(ratio_b[i]),
                             "basis_over_engine_minus_1": float(ratio_b[i] - ratio_e[i])})
        iE_s, iE_h = labels.index(f"E{rf:g}_sigmoid"), labels.index(f"E{rf:g}_hardext")
        for p, qn in enumerate(FC.Q_NAMES):
            rows.append({"fit": f"a_Rf{rf:g}", "edge": "sigmoid/hard", "observable": f"dm_{qn}",
                         "engine": float(result["dm_quantiles"][iE_s][p] / result["dm_quantiles"][iE_h][p] - 1),
                         "basis": np.nan, "basis_over_engine_minus_1": np.nan})
        i_h4, i_he = labels.index(f"P{rf:g}_hard"), labels.index(f"P{rf:g}_hardext")
        out[f"hard_vs_hardext_P{rf:g}_max"] = float(np.max(np.abs(result["cl_total"][i_he] / result["cl_total"][i_h4] - 1)))
    FC.write_csv(FC.TABLES / "F_engine_check.csv", rows)
    hard_rows = [r for r in rows if r["edge"] == "hard" and np.isfinite(r["basis_over_engine_minus_1"])]
    out["basis_vs_engine_hard"] = {
        "max_abs_primary_ell": max(abs(r["basis_over_engine_minus_1"]) for r in hard_rows if not r["observable"].endswith("_8000")),
        "max_abs_ell_8000": max(abs(r["basis_over_engine_minus_1"]) for r in hard_rows if r["observable"].endswith("_8000")),
        "per_row": {f"{r['fit']}|{r['observable']}": r["basis_over_engine_minus_1"] for r in hard_rows}}
    out["seconds"] = timer.mark("total")
    FC.write_json(FC.PRODUCTS / "engine_check.json", out)
    FC.log("engine check done", out["seconds"], out["basis_vs_engine_hard"]["max_abs_primary_ell"])


if __name__ == "__main__":
    main()
