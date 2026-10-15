"""Test I local analysis (after the maas run is fetched): I4 tables, parity with Test G, figures, verification.json.

  python -B make_figures.py [--smoke]
Reads results/ (or results_smoke/): fits35.json, finefits.json, ladder_selection_summary.json, tables/I2_ladder_*.csv,
lvariants.json, evaluate.npz, tables/I_scores.csv; Test G's all-gas ladder and A2-ii scores
(G_physical_outer/results/tables/G_selection_models.csv, G_selection_ladder.csv, G_fit_records.csv, G_scores.csv).
Writes tables/I*_*.csv, figures/*.png (figures_smoke/ in smoke mode), verification.json (verification_smoke.json).
Definitions: see i_common (sensitive-part score S_fam, family weights) and i_fine (family RMS, det / S1).
  S_fam,in  = sqrt((S_fam^2 - c_fam) / (1 - c_fam)), c_fam = importance share beyond R_cut: the score restricted to
              the painted shells (the removed shells contribute exactly c_fam to S_fam^2 since d = -1 there).
  adequacy  a (model, R_fit, R_cut) is 'adequate' for a probe family when its S1 family RMS against the A2-ii truth
            is <= the threshold (1%, 2%, 5%); 'minimal' = fewest parameters k, then smallest R_fit, then R_cut.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
FAM_FIELD = {"tSZ": "P", "tSZxDM_P": "P", "kSZ": "E", "DM": "E", "tSZxDM_E": "E"}
FAM_LABEL = {"tSZ": "tSZ $C_\\ell^{yy}$ (total, $\\ell$ 100-3000)", "kSZ": "kSZ $C_\\ell$ (1-halo, $\\ell$ 100-3000)",
             "DM": "FRB DM PDF (mean, variance, quantiles)", "tSZxDM": "tSZ $\\times$ DM (total)"}


def rcsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return np.nan


def wcsv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def s_in(S, c):
    S, c = fnum(S), fnum(c)
    if not np.isfinite(S) or not np.isfinite(c) or c >= 1:
        return np.nan
    return float(np.sqrt(max(S**2 - c, 0.0) / (1 - c)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    res = HERE / ("results_smoke" if a.smoke else "results")
    tab = res / "tables"
    figd = HERE / ("figures_smoke" if a.smoke else "figures")
    figd.mkdir(exist_ok=True)
    gres = ROOT / "G_physical_outer" / ("results_smoke" if a.smoke else "results")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    ver = {"mode": "smoke" if a.smoke else "full", "checks": {}}
    scores = rcsv(tab / "I_scores.csv")
    for r in scores:
        for fam in ("tSZ", "kSZ", "DM", "tSZxDM"):
            if r.get(f"S_{fam}") not in (None, ""):
                r[f"Sin_{fam}"] = s_in(r[f"S_{fam}"], r[f"cutshare_{fam}"])
    # ======================================================================== I2 ladder vs Test G's all-gas ladder
    lad = rcsv(tab / "I2_ladder_models.csv")
    gl = rcsv(gres / "tables" / "G_selection_models.csv")
    summ = json.loads((res / "ladder_selection_summary.json").read_text())
    gsumm = json.loads((gres / "selection_summary.json").read_text())
    comp = []
    for key, ch in summ["chosen"].items():
        g = gsumm["chosen"].get(key, {})
        f, d, rf = key.split("|")
        row = {"field": f, "dset": d, "R_fit": rf, "selected_A2ii": ch["model"], "k_A2ii": ch["k"],
               "chi2nu_floor_A2ii": ch["chi2nu_floor"], "rule_satisfied_A2ii": ch["rule_satisfied"],
               "best_BIC_A2ii": ch["best_BIC"], "best_CV_A2ii": ch["best_CV"],
               "sigma_floor_A2ii": summ["sigma_floor"][str((f, d, float(rf)))]["sigma_ref"]
               if str((f, d, float(rf))) in summ["sigma_floor"] else None,
               "selected_allgas": g.get("model"), "k_allgas": g.get("k"), "chi2nu_floor_allgas": g.get("chi2nu_floor"),
               "best_BIC_allgas": g.get("best_BIC"), "best_CV_allgas": g.get("best_CV")}
        for nm, src in (("A2ii", lad), ("allgas", gl)):
            for inner in ("PB", "PL1", "PL3", "BPLall", "PL3g", "PL3gM"):
                cand = [x for x in src if x["field"] == f and x["dset"] == d and abs(fnum(x["R_fit"]) - float(rf)) < 1e-9
                        and x["inner"] == inner and np.isfinite(fnum(x.get("BIC_floor")))]
                if cand:
                    b = min(cand, key=lambda x: fnum(x["BIC_floor"]))
                    row[f"chi2nu_floor_{inner}_{nm}"] = fnum(b["chi2nu_floor"])
                    row[f"chi2nu_L22_{inner}_{nm}"] = fnum(b.get("chi2nu_L22"))
                    row[f"dBIC_{inner}_{nm}"] = fnum(b["dBIC_floor"])
        comp.append(row)
    wcsv(tab / "I2_ladder_vs_allgas.csv", comp)
    # selection rule re-applied from the table (fewest k with dBIC <= 6 and CV within the scatter)
    agree = []
    for key, ch in summ["chosen"].items():
        f, d, rf = key.split("|")
        grp = [x for x in lad if x["field"] == f and x["dset"] == d and abs(fnum(x["R_fit"]) - float(rf)) < 1e-9
               and np.isfinite(fnum(x.get("BIC_floor")))]
        cand = [x for x in grp if x.get("pass_bic") == "True" and x.get("pass_cv") == "True"]
        sel = min(cand, key=lambda x: (fnum(x["k"]), fnum(x["BIC_floor"]))) if cand else \
            min(grp, key=lambda x: fnum(x["BIC_floor"]))
        ties = [x["model"] for x in (cand or grp) if fnum(x["k"]) == fnum(sel["k"])
                and abs(fnum(x["BIC_floor"]) - fnum(sel["BIC_floor"])) <= 1e-6 * max(1.0, abs(fnum(sel["BIC_floor"])))]
        agree.append(ch["model"] in ties)          # exact ties (e.g. PB = PL3 with two bins in smoke) count as agreement
    ver["checks"]["selection_rule_reapplied_matches"] = bool(all(agree))
    # ---- figure: ladder chi2/nu and dBIC vs k (mass data set)
    rfs = sorted({fnum(x["R_fit"]) for x in lad})
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    for i, f in enumerate(("P", "E")):
        for j, (col, lab) in enumerate((("chi2nu_floor", r"$\chi^2/\nu$ (floor likelihood)"), ("dBIC_floor", r"$\Delta$BIC (floor)"))):
            ax = axs[i, j]
            for n, rf in enumerate(rfs):
                for src, ls, mk, tag in ((lad, "-", "o", "A2-ii"), (gl, ":", "s", "all gas (Test G)")):
                    rows = [x for x in src if x["field"] == f and x["dset"] == "mass" and abs(fnum(x["R_fit"]) - rf) < 1e-9
                            and np.isfinite(fnum(x.get(col)))]
                    ks = sorted({int(fnum(x["k"])) for x in rows})
                    yv = [min(fnum(x[col]) for x in rows if int(fnum(x["k"])) == k) for k in ks]
                    if col == "dBIC_floor":
                        yv = [max(y, 0.3) for y in yv]
                    ax.plot(ks, yv, ls=ls, marker=mk, ms=4, color=OKABE[n % 8], mfc=OKABE[n % 8] if tag == "A2-ii" else "none",
                            label=f"$R_{{fit}}$={rf:g} {tag}")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks([4, 6, 8, 10, 15, 20, 30, 40])
            ax.set_xlabel("number of free parameters k (all 10 mass bins)")
            ax.set_ylabel(lab)
            if col == "dBIC_floor":
                ax.axhline(6, color="0.3", lw=0.8, ls="--")
            if col == "chi2nu_floor":
                ax.axhline(1, color="0.3", lw=0.8, ls="--")
            ax.set_title(f"{'pressure' if f == 'P' else 'electron density'}: best model per k (mass bins)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=7)
    fig.savefig(figd / "fig_I2_ladder_chi2_dbic_vs_k.png", dpi=150)
    plt.close(fig)
    # ======================================================================== I1 tables and parity with Test G
    i1 = [r for r in scores if r["block"] == "I1"]
    rows = []
    for r in i1:
        row = {k: r[k] for k in ("grid", "field", "objective", "R_fit", "R_cut", "prediction", "dm_quantiles")}
        for fam in ("tSZ", "kSZ", "DM", "tSZxDM"):
            if r.get(f"rms_{fam}") not in (None, ""):
                row[f"rms_{fam}"] = fnum(r[f"rms_{fam}"])
                row[f"S_{fam}"] = fnum(r.get(f"S_{fam}"))
                row[f"Sin_{fam}"] = r.get(f"Sin_{fam}")
        row["err_dm_mean"], row["err_dm_var"] = r.get("err_dm_mean"), r.get("err_dm_var")
        rows.append(row)
    wcsv(tab / "I1_scores.csv", rows)
    f35 = json.loads((res / "fits35.json").read_text())
    fine = json.loads((res / "finefits.json").read_text())["fits"]
    grec = rcsv(gres / "tables" / "G_fit_records.csv")
    par = []
    for x in grec:
        if x["fit_target"] != "a2ii" or x["model"] not in ("P0", "E0") or x["objective"] != "a":
            continue
        f = x["field"]
        names = ("p_P0", "p_xc", "p_beta") if f == "P" else ("p_rho0", "p_alpha", "p_beta")
        gp = np.array([fnum(x[n]) for n in names])
        labels = [lb for lb in ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7",
                                "14.7-14.9", "14.9-15.1", "15.1-15.3", "15.3-15.5")]
        c = labels.index(x["bin"]) if x["bin"] in labels else None
        if a.smoke or c is None:
            continue
        mine = f35["fits"].get(f"{f}|a|{fnum(x['R_fit']):g}|{c}")
        if mine is None:
            continue
        par.append(float(np.max(np.abs(np.asarray(mine["parameters"]) / gp - 1))))
    ver["checks"]["I1_f35_objective_a_params_vs_G_max_rel"] = max(par) if par else None
    ver["checks"]["I1_f35_objective_a_params_vs_G_n"] = len(par)
    # fine vs F grid parameters (grid dependence) and observables
    gd = []
    for key, v in f35["fits"].items():
        f, o, rf, c = key.split("|")
        if o != "a" or v is None:
            continue
        w = fine.get(key)
        if w is None:
            continue
        d = np.log(np.asarray(w["parameters"]) / np.asarray(v["parameters"]))
        gd.append({"field": f, "R_fit": rf, "bin": int(c), **{f"dlnp{i}": float(d[i]) for i in range(3)}})
    wcsv(tab / "I1_grid_dependence_params.csv", gd)
    gobs = []
    for r in i1:
        if r["grid"] != "x200":
            continue
        tw = [s for s in i1 if s["grid"] == "f35" and s["objective"] == "a" and s["field"] == r["field"]
              and s["R_fit"] == r["R_fit"] and s["R_cut"] == r["R_cut"] and s["prediction"] == r["prediction"]]
        if tw:
            t = tw[0]
            gobs.append({"field": r["field"], "R_fit": r["R_fit"], "R_cut": r["R_cut"], "prediction": r["prediction"],
                         **{f"rms_{fam}_x200": r.get(f"rms_{fam}") for fam in ("tSZ", "kSZ", "DM", "tSZxDM")},
                         **{f"rms_{fam}_f35": t.get(f"rms_{fam}") for fam in ("tSZ", "kSZ", "DM", "tSZxDM")}})
    wcsv(tab / "I1_grid_dependence_observables.csv", gobs)
    # G's scores (35-shell basis, F's truth) vs Test I (fine basis, H's truth) where configurations coincide
    gs = rcsv(gres / "tables" / "G_scores.csv")
    pg = []
    for x in gs:
        if x["fit_target"] != "a2ii" or x["eval"] != "a2ii" or x["model"] not in ("P0", "E0") or x["prediction"] not in ("det", "S1"):
            continue
        lab = f"I1|f35|{x['field']}|{x['objective']}|Rf{fnum(x['R_fit']):g}|Rc{fnum(x['R_cut']):g}"
        mine = [r for r in i1 if r["label"] == lab and r["prediction"] == x["prediction"]]
        if not mine:
            continue
        m = mine[0]
        row = {"field": x["field"], "objective": x["objective"], "R_fit": x["R_fit"], "R_cut": x["R_cut"],
               "prediction": x["prediction"]}
        for fam in ("tSZ", "kSZ", "DM"):
            if x.get(f"rms_{fam}") not in (None, "") and m.get(f"rms_{fam}") not in (None, ""):
                row[f"rms_{fam}_G"] = fnum(x[f"rms_{fam}"])
                row[f"rms_{fam}_I"] = fnum(m[f"rms_{fam}"])
        pg.append(row)
    wcsv(tab / "I1_vs_G_scores.csv", pg)
    # ---- figure: I1 heat maps (S1, family RMS) R_fit x R_cut per objective
    R_FIT = sorted({fnum(r["R_fit"]) for r in i1})
    R_CUT = sorted({fnum(r["R_cut"]) for r in i1})
    fams = [("tSZ", "P"), ("tSZxDM", "P"), ("kSZ", "E"), ("DM", "E")]
    objs = [("f35", "a"), ("f35", "d"), ("f35", "L"), ("x200", "a")]
    fig, axs = plt.subplots(len(fams), len(objs), figsize=(12, 10.5), constrained_layout=True)
    for i, (fam, f) in enumerate(fams):
        for j, (g, o) in enumerate(objs):
            ax = axs[i, j]
            Z = np.full((len(R_FIT), len(R_CUT)), np.nan)
            for r in i1:
                if r["grid"] == g and r["objective"] == o and r["field"] == f and \
                        r["prediction"] == ("det" if fam == "DM" else "S1"):
                    v = fnum(r.get(f"rms_{fam}"))
                    Z[R_FIT.index(fnum(r["R_fit"])), R_CUT.index(fnum(r["R_cut"]))] = v
            im = ax.imshow(np.log10(np.maximum(Z, 1e-4)), origin="lower", cmap="viridis", vmin=-3, vmax=0, aspect="auto")
            for (p, q), v in np.ndenumerate(Z):
                if np.isfinite(v):
                    ax.text(q, p, f"{100 * v:.1f}", ha="center", va="center", fontsize=6.5,
                            color="w" if np.log10(max(v, 1e-4)) < -1.5 else "k")
            ax.set_xticks(range(len(R_CUT)), [f"{x:g}" for x in R_CUT])
            ax.set_yticks(range(len(R_FIT)), [f"{x:g}" for x in R_FIT])
            ax.grid(False)
            ax.set_xlabel("$R_{cut}$ [$R_{200c}$]")
            ax.set_ylabel("$R_{fit}$ [$R_{200c}$]")
            ax.set_title(f"{fam} | {'B12' if f == 'P' else 'B16'} obj. ({o}), {g} grid", fontsize=8)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, label="log10 family RMS error vs A2-ii truth (numbers: %)")
    fig.suptitle("I1: B12/B16 refits on the A2-ii stacks - family RMS error (S1; DM: det, PDF passes) [%]")
    fig.savefig(figd / "fig_I1_rfit_rcut_family_rms.png", dpi=150)
    plt.close(fig)
    # ======================================================================== I3 / I4: error and score vs k
    i3 = [r for r in scores if r["block"] == "I3"]
    i1a = [r for r in i1 if r["grid"] == "f35"]
    rows = []
    for r in i3 + i1a:
        f = r["field"]
        for fam in (("tSZ", "tSZxDM") if f == "P" else ("kSZ", "DM", "tSZxDM")):
            if r.get(f"rms_{fam}") in (None, ""):
                continue
            rows.append({"block": r["block"], "model": r.get("model") if r["block"] == "I3" else
                         f"{'B12' if f == 'P' else 'B16'}({r['objective']})", "role": r.get("role"),
                         "dset": r.get("dset"), "k": int(fnum(r["k"])), "field": f, "family": fam, "R_fit": fnum(r["R_fit"]),
                         "R_cut": fnum(r["R_cut"]), "prediction": r["prediction"], "rms": fnum(r[f"rms_{fam}"]),
                         "S": fnum(r.get(f"S_{fam}")), "S_in": r.get(f"Sin_{fam}"), "cutshare": fnum(r.get(f"cutshare_{fam}"))})
    wcsv(tab / "I4_error_score_vs_k.csv", rows)
    # minimal adequate model per probe family
    mins = []
    for fam in ("tSZ", "kSZ", "DM", "tSZxDM"):
        for thr in (0.01, 0.02, 0.05):
            for pred in ("S1", "det"):
                c = [x for x in rows if x["family"] == fam and x["prediction"] == pred and np.isfinite(x["rms"])
                     and x["rms"] <= thr]
                if fam == "tSZxDM":
                    c = [x for x in c if x["field"] == "E"] + [x for x in c if x["field"] == "P"]
                if not c:
                    mins.append({"family": fam, "threshold": thr, "prediction": pred, "model": None})
                    continue
                b = min(c, key=lambda x: (x["k"], x["R_fit"], x["R_cut"]))
                mins.append({"family": fam, "threshold": thr, "prediction": pred, **b})
    wcsv(tab / "I4_minimal_adequate.csv", mins)
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for j, (fam, f) in enumerate((("tSZ", "P"), ("kSZ", "E"), ("DM", "E"), ("tSZxDM", "E"))):
        for i, col in enumerate(("rms", "S")):
            ax = axs[i, j]
            for n, rf in enumerate(sorted({x["R_fit"] for x in rows if x["block"] == "I3"})):
                for cutsel, mk in (("fit", "o"), ("4", "^")):
                    sel = [x for x in rows if x["family"] == fam and x["field"] == f and abs(x["R_fit"] - rf) < 1e-9
                           and x["prediction"] == ("det" if fam == "DM" else "S1")
                           and ((cutsel == "fit" and abs(x["R_cut"] - rf) < 1e-9) or (cutsel == "4" and x["R_cut"] == 4.0))]
                    if not sel:
                        continue
                    ax.scatter([x["k"] for x in sel], [max(x[col], 1e-4) for x in sel], marker=mk, s=18,
                               color=OKABE[n % 8], facecolors=OKABE[n % 8] if cutsel == "4" else "none",
                               label=f"$R_{{fit}}$={rf:g}, $R_{{cut}}$={'$R_{fit}$' if cutsel == 'fit' else '4'}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("free parameters k")
            ax.set_ylabel("family RMS error vs A2-ii truth" if col == "rms" else "sensitive-part score $S_{fam}$")
            ax.set_title(f"{fam} ({'P' if f == 'P' else 'E'} models)", fontsize=8)
            if col == "rms":
                for thr in (0.01, 0.05):
                    ax.axhline(thr, color="0.4", lw=0.7, ls="--")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=7)
    fig.suptitle("I3/I4: observable error and sensitive-part score vs number of parameters (ladder models and B12/B16)")
    fig.savefig(figd / "fig_I4_error_score_vs_k.png", dpi=150)
    plt.close(fig)
    # ---- residual profiles of key models (R_fit = R_cut = 4) with the importance weight
    ev = np.load(res / "evaluate.npz")
    xe = ev["x_edges"]
    xm = np.sqrt(np.maximum(xe[:-1], 0.0179) * xe[1:])
    nb = ev["mstack_P"].shape[0]
    show = sorted({0, nb // 2, nb - 2 if nb > 2 else nb - 1})
    fig, axs = plt.subplots(2, len(show), figsize=(4 * len(show), 6.5), constrained_layout=True, squeeze=False)
    for i, f in enumerate(("P", "E")):
        labs = [k for k in ev.files if k.startswith("M4__") and f"|{f}|" in k and "Rf4|Rc4" in k
                and (k.startswith("M4__I3") or "I1|f35|" in k and "|a|" in k)]
        for j, c in enumerate(show):
            ax = axs[i, j]
            m = ev[f"mstack_{f}"][c]
            seen = set()
            n = -1
            for k in labs:
                name = k.split("|")[3] if k.startswith("M4__I3") else ("B12" if f == "P" else "B16")
                if name in seen or len(seen) >= 7:
                    continue
                seen.add(name)
                n += 1
                M = ev[k][c]
                ok = (m > 0) & (M > 0)
                ax.plot(xm[ok], np.log(M[ok] / m[ok]), color=OKABE[n % 8], lw=1.1, label=name)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xscale("log")
            ax.set_ylim(-0.6, 0.6)
            ax.set_xlabel("$x = r/R_{200c}$")
            ax.set_ylabel("ln(model / A2-ii stack)")
            ax.set_title(f"{'pressure' if f == 'P' else 'electrons'}, bin {c} ($R_{{fit}}$=$R_{{cut}}$=4)", fontsize=8)
            tw = ax.twinx()
            for n2, fam in enumerate(("tSZ",) if f == "P" else ("kSZ", "DM")):
                w = ev[f"w__{fam}"][c]
                dl = np.log(xe[1:] / np.maximum(xe[:-1], 0.0179))
                tw.fill_between(xm, 0, w / dl / np.sum(ev[f"w__{fam}"]), step="mid", alpha=0.18,
                                color=["0.3", OKABE[3]][n2], label=f"{fam} importance / d ln x")
            tw.set_ylabel("importance density (share per unit ln x)")
            tw.grid(False)
            if j == 0:
                ax.legend(fontsize=6, loc="lower left")
                tw.legend(fontsize=6, loc="upper right")
    fig.savefig(figd / "fig_I4_residual_profiles.png", dpi=150)
    plt.close(fig)
    # ======================================================================== I5 stacking choices
    i5 = [r for r in scores if r["block"] == "I5"]
    rows5 = []
    for r in i5:
        f = r["field"]
        for fam in (("tSZ", "tSZxDM") if f == "P" else ("kSZ", "DM", "tSZxDM")):
            if r.get(f"rms_{fam}") in (None, ""):
                continue
            rows5.append({"variant": r["variant"], "model": r["model"], "field": f, "family": fam, "R_fit": fnum(r["R_fit"]),
                          "R_cut": fnum(r["R_cut"]), "prediction": r["prediction"], "rms": fnum(r[f"rms_{fam}"]),
                          "S": fnum(r.get(f"S_{fam}")), "selected": r.get("selected")})
    wcsv(tab / "I5_variants_observables.csv", rows5)
    shifts = []
    for key, v in f35["variants"].items():
        var, f, rf, i = key.split("|")
        ref = f35["variants"].get(f"a2ii|{f}|{rf}|{i}") if var != "m05" else None
        if v is None or ref is None:
            continue
        d = np.log(np.asarray(v["parameters"]) / np.asarray(ref["parameters"]))
        shifts.append({"variant": var, "field": f, "R_fit": rf, "bin": int(i), **{f"dlnp{k}": float(d[k]) for k in range(3)}})
    wcsv(tab / "I5_parameter_shifts.csv", shifts)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    variants = ["a2ii", "a2ii_agn", "allgas", "a2i", "unweighted", "median", "m05", "perhalo", "xray"]
    for j, (fam, f) in enumerate((("tSZ", "P"), ("kSZ", "E"), ("DM", "E"))):
        ax = axs[j]
        models = sorted({x["model"] for x in rows5 if x["field"] == f})
        wbar = 0.8 / max(len(models) * 2, 1)
        n = 0
        for mdl in models:
            for rf in (2.0, 4.0):
                vals = []
                for v in variants:
                    s = [x for x in rows5 if x["variant"] == v and x["model"] == mdl and x["family"] == fam
                         and x["R_fit"] == rf and x["R_cut"] == 4.0 and x["prediction"] == ("det" if fam == "DM" else "S1")]
                    vals.append(s[0]["rms"] if s else np.nan)
                ax.bar(np.arange(len(variants)) + n * wbar, np.array(vals) * 100, width=wbar, color=OKABE[n % 8],
                       label=f"{mdl}, $R_{{fit}}$={rf:g}")
                n += 1
        ax.set_xticks(np.arange(len(variants)) + 0.4 - wbar / 2, variants, rotation=45, ha="right")
        ax.set_ylabel("family RMS error vs A2-ii truth [%] ($R_{cut}$ = 4)")
        if any(np.isfinite(x["rms"]) and x["rms"] > 0 for x in rows5 if x["family"] == fam and x["R_cut"] == 4.0):
            ax.set_yscale("log")
        ax.set_title(fam)
        ax.legend(fontsize=6)
    fig.suptitle("I5: stacking choice -> fit -> observable error against the A2-ii painting target")
    fig.savefig(figd / "fig_I5_stacking_choices.png", dpi=150)
    plt.close(fig)
    # ======================================================================== verification
    i0 = json.loads((HERE / "inputs" / "i0_variants.json").read_text())
    ver["checks"]["I0_variant_a2ii_vs_G_stacks_max_rel"] = {k: v for k, v in i0.items() if k.startswith("parity")}
    ver["checks"]["n_score_rows"] = len(scores)
    ver["checks"]["cutshare_zero_at_Rcut4"] = bool(all(fnum(r.get(f"cutshare_{fam}")) == 0.0 for r in scores
                                                      if fnum(r["R_cut"]) == 4.0 for fam in ("tSZ", "kSZ", "DM")
                                                      if r.get(f"cutshare_{fam}") not in (None, "")))
    ver["checks"]["scores_finite_fraction"] = float(np.mean([np.isfinite(fnum(r.get("S_tSZ") or r.get("S_kSZ")))
                                                             for r in scores]))
    man = json.loads((res / "run_manifest.json").read_text()) if (res / "run_manifest.json").exists() else {}
    if man:
        inv = man["invocations"][-1]
        ver["run"] = {"stages": {k: {kk: v.get(kk) for kk in ("status", "wall_seconds", "returncode")}
                                 for k, v in inv["stages"].items()}, "exit": inv.get("exit"),
                      "h_inputs_sha256": inv.get("h_inputs_sha256"), "scratch_removed": inv.get("scratch_removed")}
        hloc = ROOT / "H_sensitivity" / ("results_smoke" if a.smoke else "results")
        import hashlib
        same = {}
        for n, d in (inv.get("h_inputs_sha256") or {}).items():
            p = hloc / n
            same[n] = (hashlib.sha256(p.read_bytes()).hexdigest() == d) if p.exists() else None
        ver["checks"]["H_inputs_used_equal_local_H_results"] = same
    (HERE / ("verification_smoke.json" if a.smoke else "verification.json")).write_text(json.dumps(ver, indent=1, default=str) + "\n")
    print(json.dumps(ver["checks"], indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
