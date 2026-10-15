"""Group 'grids_scaling', part 2: sensitivity of B12 (pressure) / B16 (electron) fits to the radial grid.

Objective (a) of Tests B/F (analysis.fitting.objective_a: diagonal, sigma = max(SD/mean, 5%), ln residuals,
shell-averaged model), strict-resolved shells with outer edge <= R_fit, calibrated bins 13.5-15.5.
Grids: fit (reference), x200, old, lee22.  R_fit = 1.34 (all four grids; the old grid has no 1.34 edge and is
fitted to its 1.3508 edge; every other grid is fitted to exactly 1.34), 2 and 4 R200c (fit, x200, old).
Diagnostic case 'x200@1.3508' (R_fit = 1.34 block only): x200 fitted to its 1.3508 edge, i.e. with the single
1.34-1.3508 sliver shell added, to measure what one boundary sliver does to the fit.  50 paired bootstrap replicates (the core's per-bin seeds, so the replicate
stacks are the same halo resamples on every grid); the sample SD sets the weights in every replicate fit.

Writes work/fits_results.json and work/figdata_fits.pkl.
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np

import gs_common as C

sys.path.insert(0, str(C.STUDY))
from analysis import fitting as AF  # noqa: E402  (read-only library of the study)

sc = C.sc
T0 = time.time()
WORK = C.HERE / "work"
WORK.mkdir(exist_ok=True)
N_REP = 50
CAL = C.calibrated(C.bins())
FIT_E = sc.edges("fit")
FAMILY = {"P": "b12", "E": "b16"}
PNAMES = {"P": ("P0", "xc", "beta"), "E": ("rho0", "alpha", "beta")}
CASES = {1.34: ("fit", "x200", "old", "lee22", "x200@1.3508"), 2.0: ("fit", "x200", "old"),
         4.0: ("fit", "x200", "old")}
X_INT = (1.0, 2.0, 4.0)
FINE = {X: np.r_[0.0, np.geomspace(1e-4, X, 600)] for X in X_INT}


def rfit_eff(grid, rfit):
    """Outer edge of the fit range: exactly R_fit, except on the old grid (no 1.34 edge), which uses 1.3508."""
    e = sc.edges(grid)
    tol = 0.011 if (grid == "old" and abs(rfit - 1.34) < 1e-9) else 1e-9
    return float(e[e <= rfit + tol].max())


def grid_and_rfit(label, rfit):
    """'x200@1.3508' -> ('x200', 1.3508 snapped to the stored edge); plain grid names -> (grid, rfit_eff)."""
    if "@" in label:
        g, r = label.split("@")
        e = sc.edges(g)
        return g, float(e[np.argmin(np.abs(e - float(r)))])
    return label, rfit_eff(label, rfit)


def model_integral(family, p, X):
    e = FINE[X]
    return float(np.sum(AF.shell_model(family, p, e) * 4 * np.pi / 3 * np.diff(e**3)))


def one_fit(family, mean, sd, mask, edges, start=None):
    fit = AF.objective_a(family, mean, sd, mask, start=start, edges=edges, jac="analytic")
    return fit


out = {"description": __doc__, "n_replicates": N_REP, "cases": {}}
figd = {}
for field in ("P", "E"):
    fam = FAMILY[field]
    for rfit, grids in CASES.items():
        case = f"{field}_Rfit{rfit:g}"
        res_case = {"field": field, "family": fam, "R_fit": rfit, "grids": {}, "vs_fit": {}}
        samples = {}            # grid -> per bin dict
        reps = {}               # grid -> [bin][rep] params / integrals / models
        for g in grids:
            gname, rf = grid_and_rfit(g, rfit)
            e = sc.edges(gname)
            per_bin = {}
            rep_store = []
            for b in CAL:
                st = sc.stack_bin(sc.make_key(field, grid=gname), b, "mean", "strict", n_boot=C.N_BOOT, return_boot=True)
                mask = st["resolved"] & (e[1:] <= rf + 1e-12)
                fit = one_fit(fam, st["value"], st["sd"], mask, e)
                p = np.asarray(fit["parameters"]) if fit is not None else np.full(3, np.nan)
                per_bin[b["label"]] = {
                    "parameters": dict(zip(PNAMES[field], p.tolist())), "accepted": bool(fit and fit["accepted"]),
                    "chi2_per_dof": fit["chi2_per_dof"] if fit else None, "rms_percent": fit["rms_percent"] if fit else None,
                    "n_shells": int(mask.sum()), "R_fit_effective": rf,
                    "first_shell_edge": float(e[np.flatnonzero(mask)[0]]) if mask.any() else None,
                    "model_integral": {f"<{X:g}": model_integral(fam, p, X) for X in X_INT},
                    "stack_integral": {f"<{X:g}": float(sc.integral(st["value"], gname, X)) if X <= e[-1] + 1e-9 else None
                                       for X in X_INT}}
                rp = []
                for r in range(N_REP):
                    fr = one_fit(fam, st["boot"][r], st["sd"], mask, e, start=p)
                    pr = np.asarray(fr["parameters"]) if fr is not None else np.full(3, np.nan)
                    rp.append({"p": pr, "I": np.array([model_integral(fam, pr, X) for X in X_INT]),
                               "m": AF.shell_model(fam, pr, FIT_E)})
                rep_store.append(rp)
                per_bin[b["label"]]["model_fit_grid"] = AF.shell_model(fam, p, FIT_E)
            samples[g] = per_bin
            reps[g] = rep_store
            print(f"{case} {g}  {time.time() - T0:.0f} s", flush=True)
        # comparisons with the fit grid (paired)
        for g in grids:
            if g == "fit":
                continue
            comp = {}
            dP = np.full((len(CAL), N_REP + 1, 3), np.nan)
            dI = np.full((len(CAL), N_REP + 1, len(X_INT)), np.nan)
            mv = np.full((len(CAL), N_REP + 1, 35), np.nan)
            mr = np.full((len(CAL), N_REP + 1, 35), np.nan)
            for i, b in enumerate(CAL):
                s_g, s_f = samples[g][b["label"]], samples["fit"][b["label"]]
                pg = np.array(list(s_g["parameters"].values()))
                pf = np.array(list(s_f["parameters"].values()))
                dP[i, 0] = np.log(pg / pf)
                dI[i, 0] = np.log(np.array(list(s_g["model_integral"].values())) /
                                  np.array(list(s_f["model_integral"].values())))
                mv[i, 0], mr[i, 0] = s_g["model_fit_grid"], s_f["model_fit_grid"]
                for r in range(N_REP):
                    a, f_ = reps[g][i][r], reps["fit"][i][r]
                    dP[i, r + 1] = np.log(a["p"] / f_["p"])
                    dI[i, r + 1] = np.log(a["I"] / f_["I"])
                    mv[i, r + 1], mr[i, r + 1] = a["m"], f_["m"]
                comp[b["label"]] = {
                    "dln_params": dict(zip(PNAMES[field], dP[i, 0].tolist())),
                    "dln_params_err": dict(zip(PNAMES[field], np.nanstd(dP[i, 1:], axis=0, ddof=1).tolist())),
                    "dln_model_integral": {f"<{X:g}": float(dI[i, 0, k]) for k, X in enumerate(X_INT)},
                    "dln_model_integral_err": {f"<{X:g}": float(np.nanstd(dI[i, 1:, k], ddof=1))
                                               for k, X in enumerate(X_INT)}}
            obs = C.OBS_P if field == "P" else C.OBS_E
            imp = C.impact_rows(mv, mr, field, obs, n_err=N_REP)
            # summary over bins
            summ = {"max_abs_dln_param": {n: float(np.nanmax(np.abs(dP[:, 0, k]))) for k, n in enumerate(PNAMES[field])},
                    "max_abs_dln_param_over_err": {n: float(np.nanmax(np.abs(dP[:, 0, k]) /
                                                                       np.nanstd(dP[:, 1:, k], axis=1, ddof=1)))
                                                   for k, n in enumerate(PNAMES[field])},
                    "max_abs_dln_model_integral": {f"<{X:g}": float(np.nanmax(np.abs(dI[:, 0, k])))
                                                   for k, X in enumerate(X_INT)}}
            res_case["vs_fit"][g] = {"per_bin": comp, "impact_cut4R": imp, "summary": summ}
            figd[(case, g)] = {"dP": dP, "dI": dI, "logm": np.array([b["mean_logm200c"] for b in CAL])}
        for g in grids:
            for lab in samples[g]:
                samples[g][lab].pop("model_fit_grid", None)
        res_case["grids"] = samples
        out["cases"][case] = res_case

# parity with Test F's objective (a) fits on the fit grid (F_fit_records.csv)
parity = {}
try:
    import csv
    rows = list(csv.DictReader(open(C.STUDY / "F_observables" / "results" / "tables" / "F_fit_records.csv")))
    worst = 0.0
    for r in rows:
        if r["method"] != "a" or r["R_fit"] not in ("2", "4") or r["field"] not in ("P", "E"):
            continue
        case = f"{r['field']}_Rfit{float(r['R_fit']):g}"
        mine = out["cases"][case]["grids"]["fit"].get(r["bin"])
        if mine is None:
            continue
        pf = np.array([float(v) for v in r["parameters"].split(";")])
        pm = np.array(list(mine["parameters"].values()))
        worst = max(worst, float(np.max(np.abs(pm / pf - 1))))
    parity["max_rel_dev_vs_TestF_objective_a_params_Rfit2_4"] = worst
except Exception as exc:       # noqa: BLE001
    parity["error"] = repr(exc)
out["parity_TestF"] = parity
out["runtime_s"] = time.time() - T0
(WORK / "fits_results.json").write_text(json.dumps(C.jsonify(out), indent=1))
with open(WORK / "figdata_fits.pkl", "wb") as fh:
    pickle.dump({"figd": figd, "cal": [b["label"] for b in CAL]}, fh)
print(f"done {time.time() - T0:.0f} s; parity {parity}", flush=True)
