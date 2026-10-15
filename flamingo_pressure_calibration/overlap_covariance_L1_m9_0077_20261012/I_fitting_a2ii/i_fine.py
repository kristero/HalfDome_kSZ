"""Test I stages on Test H's fine grid (67 X200 shells to 4 R200c + 8 extension shells), in Test H's configuration.

Configuration (before any F / G / H import): h_common is imported read-only with
  H_RESULTS = <I results>/work/h      (Test I's own directory: F / G modules write nothing outside it)
  <I results>/work/h/work/g/f_base/profiles.npz -> Test H's fine per-halo profiles (symlink, read-only)
  <...>/f_base/work/basis_HA.npz      -> Test H's fine A2-ii basis (hard link into Test I's scratch on maas, so it
                                         survives a cleanup of Test H's scratch; symlink in smoke)
Test H's truth (truth.npz, truth_summary.json), importance (sens.npz) and DM-PDF fallback slot
(work/pdf_fallback_h.npz) are read from Test H's results directory (I_H_RESULTS).

Stages
  finefits  B12 / B16 objective (a) on the fine 67-shell A2-ii stacks, R_fit in {1, 1.34, 1.5, 2, 3, 4}
            (h_fits.fit: the grid-dependence check of I1).
  evaluate  every painting of I1 (fits35.json, finefits.json), I3 (the ladder models of ladder_selection.npz)
            and I5 (fits35.json variants, lvariants.json) on the fine A2-ii basis HA:
              shells     exact shell averages of the fitted form on the 75 fine shells, hard cut at R_cut
              pairing    a pressure model with the A2-ii truth's mean electron profile, an electron model with the
                         truth's mean pressure (field-separated scoring as in Test G)
              det        one profile per calibrated bin
              S1         lognormal amplitude scatter of the whole profile, sigma^2 = ln <X^2>/<X>^2 of the per-halo
                         X = Y(<4R), N_e(<4R) of the bin (A2-ii), cross <Y N>/(<Y><N>) (Test F / G / H)
              DM PDF     f_pdf passes (Test H's h_pdf) for the listed electron paintings (others: DM mean and
                         variance only, no DM family score)
              errors     O_model / O_truth - 1 against the A2-ii scatter-inclusive truth (and the truth mean)
              family RMS sqrt(mean of squared errors) over Test F's primary members: tSZ yytot ell 100-3000,
                         kSZ ksz1h ell 100-3000, DM dm_mean, dm_var, dm_q16 ... dm_q999, tSZxDM ydmtot ell 100-3000
              S_fam      the sensitive-part score (i_common) with Test H's A2-ii importance
Outputs: results/finefits.json, results/evaluate.npz, results/evaluate_items.json, tables/I_scores.csv,
tables/I_errors_long.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import i_common as IC  # noqa: E402

HDIR = IC.WORK / "h"
FBASE = HDIR / "work" / "g" / "f_base"
H_FBASE = IC.H_RESULTS / "work" / "g" / "f_base"


def configure(workers, scratch=None):
    FBASE.mkdir(parents=True, exist_ok=True)
    prof = FBASE / "profiles.npz"
    if not prof.exists():
        prof.symlink_to((H_FBASE / "profiles.npz").resolve())
    src_basis = (H_FBASE / "work" / "basis_HA.npz").resolve()
    if scratch:
        sw = Path(scratch).resolve() / "f_base_work"
        sw.mkdir(parents=True, exist_ok=True)
        dst = sw / "basis_HA.npz"
        if not dst.exists():
            try:
                os.link(src_basis, dst)
            except OSError:
                shutil.copyfile(src_basis, dst)
        os.environ["H_SCRATCH"] = str(Path(scratch).resolve())
    else:
        (FBASE / "work").mkdir(parents=True, exist_ok=True)
        dst = FBASE / "work" / "basis_HA.npz"
        if not dst.exists():
            dst.symlink_to(src_basis)
    os.environ["H_MODE"] = IC.MODE
    os.environ["H_RESULTS"] = str(HDIR)
    os.environ["H_WORKERS"] = str(workers)
    os.environ["STUDY_ROOT"] = str(IC.ROOT)
    sys.path.insert(0, str(IC.ROOT / "H_sensitivity"))


# ============================================================================ finefits
def stage_finefits():
    import h_common as HC
    import h_fits as HF
    FC = HC.FC
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    out = {}
    for field in ("P", "E"):
        for rf in IC.R_FIT_I1:
            for c in range(FC.N_CAL):
                f = HF.fit(HF.bin_data(pr, field, "a2ii", c), rf)
                out[f"{field}|a|{rf:g}|{c}"] = None if f is None else {
                    "parameters": np.asarray(f["parameters"]).tolist(), "chi2": f["chi2"],
                    "chi2_per_dof": f["chi2_per_dof"], "n_shells": int(f["n_shells"]), "rms_percent": f["rms_percent"]}
    IC.write_json(IC.RESULTS / "finefits.json", {"fits": out, "x_edges": HC.X200})
    IC.log("finefits done")


# ============================================================================ evaluate
FAMS = {"P": ("tSZ", "tSZxDM"), "E": ("kSZ", "DM", "tSZxDM")}
SCORE_FAMS = {"P": {"tSZ": "tSZ", "tSZxDM": "tSZxDM_P"}, "E": {"kSZ": "kSZ", "DM": "DM", "tSZxDM": "tSZxDM_E"}}


def build_items(HC, pr, lad):
    """Every painting: dict(label, block, field, M75 [N_CAL, 75] untruncated-to-4R shells, R_cut, pdf list, meta)."""
    import h_fits as HF
    import g_physics as PH
    import g_selection as GS
    FC = HC.FC
    items = []
    fits35 = IC.read_json(IC.RESULTS / "fits35.json")
    finefits = IC.read_json(IC.RESULTS / "finefits.json")["fits"]

    def add(label, block, field, M4, r_cut, pdf, **meta):
        items.append(dict(label=label, block=block, field=field, M4=M4, R_cut=float(r_cut), pdf=tuple(pdf), **meta))

    def b_shells(field, recs):
        if any(r is None for r in recs):
            return None
        return np.array([HF.model_shells(field, r["parameters"], cut=4.0) for r in recs])

    # ---- I1
    for grid, src, objs in (("f35", fits35["fits"], IC.OBJECTIVES), ("x200", finefits, ("a",))):
        for field in ("P", "E"):
            for obj in objs:
                for rf in IC.R_FIT_I1:
                    recs = [src.get(f"{field}|{obj}|{rf:g}|{c}") for c in range(FC.N_CAL)]
                    M4 = b_shells(field, recs)
                    if M4 is None:
                        continue
                    for rc in IC.R_CUT_I1:
                        diag = abs(rc - rf) < 1e-9 or rc == 4.0
                        pdf = ("det", "S1") if diag else ("det",)
                        add(f"I1|{grid}|{field}|{obj}|Rf{rf:g}|Rc{rc:g}", "I1", field, M4, rc, pdf, grid=grid,
                            model="B12" if field == "P" else "B16", objective=obj, R_fit=rf, k=3 * FC.N_CAL)
    # ---- I3 (ladder models; evaluated at the calibrated bins' mean log M and mean ln c)
    if lad is not None:
        results, chosen = lad
        res_by = {(r["field"], r["dset"], float(r["R_fit"]), r["model"]): r for r in results}
        xF, wqF = PH.quad_nodes(HC.FINE)
        lh = PH.LinearHalo()
        prep = np.load(IC.ROOT / "G_physical_outer" / ("results_smoke" if IC.SMOKE else "results") / "prepare.npz")
        ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
        rows = [np.flatnonzero(pr["mass_bin"] == b) for b in FC.CAL_BINS]
        T2h = np.array([PH.two_halo_template(lh, pr["m200c_msun"][r], pr["r200c_mpc"][r], pr["weight"][r], xF)
                        for r in rows])
        w = [pr["weight"][r] / pr["weight"][r].sum() for r in rows]
        logm = np.array([ww @ np.log10(pr["m200c_msun"][r]) for ww, r in zip(w, rows)])
        lnc = np.array([ww @ np.log(pr["concentration_200c"][r]) for ww, r in zip(w, rows)])
        view = types.SimpleNamespace(bin=np.arange(FC.N_CAL), logm=logm, lnc=lnc, n=FC.N_CAL)
        LAD = {"x": xF, "wq": wqF, "T2h": T2h, "ctx": ctx, "view": view}

        def lshells(field, dset, name, theta, frozen=None):
            inner, outer = name.split("+")
            spec = [s for s in GS.model_specs(field, dset) if s[0] == inner][0]
            model = GS.build_model(field, spec, outer, FC.N_CAL, frozen=frozen)
            Fin, Fout = model.evaluate(np.asarray(theta, float), view, xF, wqF, HC.N_FINE, T2h, ctx, jac=False,
                                       parts=True)
            M = Fin + Fout
            M[:, ~(HC.FINE[1:] <= 4.0 + 1e-9)] = 0.0
            return M, model.k

        def best_outer(field, dset, rf, inner):
            cands = [r for (f, d, r_, m), r in res_by.items() if f == field and d == dset and abs(r_ - rf) < 1e-9
                     and m.split("+")[0] == inner and "floor" in r["fits"]]
            if not cands:
                return None
            n = lambda r: r["fits"]["floor"]["n_data"]                      # noqa: E731
            return min(cands, key=lambda r: r["fits"]["floor"]["chi2"] + r["k"] * np.log(n(r)))

        for field in ("P", "E"):
            for rf in sorted({float(k.split("|")[2]) for k in chosen}):
                ch = chosen.get(f"{field}|mass|{rf:g}")
                if ch is None:
                    continue
                pick = {"selected": res_by.get((field, "mass", rf, ch["model"]))}
                for inner in ("PB", "PL1", "PL3", "BPLall"):
                    pick[inner] = best_outer(field, "mass", rf, inner)
                gam = [best_outer(field, "mass", rf, i) for i in ("PL3g", "PL3gM")]
                gam = [g for g in gam if g is not None]
                if gam:
                    pick["gamma"] = min(gam, key=lambda r: r["fits"]["floor"]["chi2"] + r["k"] *
                                        np.log(r["fits"]["floor"]["n_data"]))
                cm = [best_outer(field, "massc", rf, i) for i in ("PL1cA", "PL3cA", "PL3c3", "L22best", "L22best_L22")]
                cm = [g for g in cm if g is not None]
                if cm:
                    pick["cbest"] = min(cm, key=lambda r: r["fits"]["floor"]["chi2"] + r["k"] *
                                        np.log(r["fits"]["floor"]["n_data"]))
                cuts = sorted({rf, 4.0})
                for role, r in pick.items():
                    if r is None:
                        continue
                    M4, k = lshells(field, r["dset"], r["model"], r["fits"]["floor"]["theta"], r.get("frozen_shape"))
                    for rc in cuts:
                        add(f"I3|{field}|{role}|{r['model']}|{r['dset']}|Rf{rf:g}|Rc{rc:g}", "I3", field, M4, rc,
                            ("det", "S1"), grid="f35", model=r["model"], role=role, dset=r["dset"], R_fit=rf,
                            k=int(r["k"]), objective="floor")
    # ---- I5 (variants): B12 / B16 and the Lee22 models of lvariants.json
    var = dict(np.load(IC.VARIANTS_NPZ))
    for v in IC.VARIANTS:
        bins = var[f"bins_{v}"]
        idx = []
        for b in FC.CAL_BINS:
            lo = FC.MASS_EDGES[b]
            idx.append(int(np.flatnonzero((bins[:, 0] <= lo + 1e-9) & (bins[:, 1] > lo + 1e-9))[0]))
        for field in ("P", "E"):
            for rf in IC.R_FIT_I5:
                recs = [fits35["variants"].get(f"{v}|{field}|{rf:g}|{i}") for i in idx]
                M4 = b_shells(field, recs)
                if M4 is not None:
                    for rc in sorted({rf, 4.0}):
                        add(f"I5|{v}|{field}|B|Rf{rf:g}|Rc{rc:g}", "I5", field, M4, rc,
                            ("det", "S1") if rc == 4.0 else ("det",), grid="f35", variant=v,
                            model="B12" if field == "P" else "B16", R_fit=rf, k=3 * len(bins), objective="a")
    if lad is not None and (IC.RESULTS / "lvariants.json").exists():
        lv = IC.read_json(IC.RESULTS / "lvariants.json")
        for key, r in lv.items():
            if "theta" not in r:
                continue
            v, field, rf, name = key.split("|")
            rf = float(rf)
            M4, k = lshells(field, "mass", name, r["theta"])
            for rc in sorted({rf, 4.0}):
                add(f"I5|{v}|{field}|{name}|Rf{rf:g}|Rc{rc:g}", "I5", field, M4, rc,
                    ("det", "S1") if rc == 4.0 else ("det",), grid="f35", variant=v, model=name, R_fit=rf, k=k,
                    objective="diag", selected=bool(r.get("selected")))
    return items


def cut(HC, M4, rc):
    M = np.array(M4, float, copy=True)
    M[:, ~(HC.FINE[1:] <= rc + 1e-9)] = 0.0
    return M


def stage_evaluate():
    import h_common as HC
    import h_params as HPm
    import h_pdf as HP
    import g_physics as PH
    FC, FB = HC.FC, HC.FB
    pr = HC.load_npz(FC.PRODUCTS / "profiles.npz")
    lad = None
    if (IC.RESULTS / "ladder_selection.npz").exists():
        results = json.loads(str(np.load(IC.RESULTS / "ladder_selection.npz")["results_json"]))
        chosen = IC.read_json(IC.RESULTS / "ladder_selection_summary.json")["chosen"]
        lad = (results, chosen)
    items = build_items(HC, pr, lad)
    IC.log(f"evaluate: {len(items)} paintings")
    s1, _ = PH.scatter_tables(pr, "a2ii", cuts=(4.0,))
    ts = IC.read_json(IC.H_RESULTS / "truth_summary.json")
    truth = dict(ts["observables"]["truth_a2ii"])
    tmean = dict(ts["observables"]["truthmean_a2ii"])
    for p, qn in enumerate(FC.Q_NAMES):
        truth[f"dm_{qn}"] = float(ts["pdf"]["truth_a2ii"]["quantiles"][p])
        tmean[f"dm_{qn}"] = float(ts["pdf"]["truthmean_a2ii"]["quantiles"][p])
    Pt = HP.pad(pr["meanP_a2ii"][FC.CAL_BINS])
    Et = HP.pad(pr["meanE_a2ii"][FC.CAL_BINS])
    # ---------------- DM PDF passes (forked; no basis loaded yet), checkpointed per group
    ck_path = IC.WORK / "pdf_quantiles.json"
    quant = IC.read_json(ck_path) if ck_path.exists() else {}
    fb = np.load(IC.H_RESULTS / "work" / "pdf_fallback_h.npz")
    fallback = (fb["hist"], float(fb["overflow"]))
    prio = {"I1": 1, "I3": 0, "I5": 2}
    todo = []
    for it in items:
        if it["field"] != "E":
            continue
        M = cut(HC, it["M4"], it["R_cut"])
        for v in it["pdf"]:
            lab = f"{it['label']}|{v}"
            if lab in quant:
                continue
            ent = HP.det_entries(M) if v == "det" else HP.s1_entries(M, s1[:, 1])
            diag = abs(it["R_cut"] - it.get("R_fit", -1)) < 1e-9 or it["R_cut"] == 4.0
            todo.append(((0 if diag else 1, prio[it["block"]], 0 if v == "det" else 1), lab, ent))
    todo.sort(key=lambda t: t[0])
    if os.environ.get("I_PDF_LIMIT"):                    # smoke tests only: the first N histograms
        todo = todo[:int(os.environ["I_PDF_LIMIT"])]
    IC.log(f"PDF histograms to run: {len(todo)} (cached {len(quant)})")
    G = HP.GROUP
    for start in range(0, len(todo), G):
        part = [(lab, ent) for _, lab, ent in todo[start:start + G]]
        HC.mem("before PDF group")
        res, _, _ = HP.run_specs(part, "HA", fallback=fallback, label=f"I pdf {start // G + 1}")
        for lab, s in res.items():
            quant[lab] = {"quantiles": np.asarray(s["quantiles"], float).tolist(), "mean": float(s["mean"]),
                          "variance": float(s["variance"])}
        IC.write_json(ck_path, quant)
        IC.log(f"PDF {min(start + G, len(todo))}/{len(todo)}")
    # ---------------- moment observables on the A2-ii fine basis (serial; no fork with the basis loaded)
    HC.mem("before basis")
    B = FB.Basis("HA")
    sens = dict(np.load(IC.H_RESULTS / "sens.npz"))
    W = IC.family_weights(sens, FC.CAL_BINS, n_x=HC.N_X)
    IMP = IC.importance_maps(sens, FC.CAL_BINS, n_x=HC.N_X)
    ell = np.asarray(sens["ell"], float)
    mstack = {"P": pr["meanP_a2ii"][FC.CAL_BINS], "E": pr["meanE_a2ii"][FC.CAL_BINS]}
    fams_all = {f: [k for k in FC.FAMILIES[f] if FC.primary(k)] for f in FC.FAMILIES}
    rows, long_rows, meta_items = [], [], []
    store = {}
    for n_it, it in enumerate(items):
        f = it["field"]
        M = cut(HC, it["M4"], it["R_cut"])
        base = {k: it.get(k) for k in ("label", "block", "field", "grid", "model", "role", "dset", "objective",
                                         "variant", "R_fit", "R_cut", "k", "selected")}
        # sensitive-part scores (prediction-independent)
        sc_ = {}
        for fam, wkey in SCORE_FAMS[f].items():
            s, lost = IC.sens_score(it["M4"][:, :HC.N_X], mstack[f], W[wkey], HC.X200, it["R_cut"])
            sc_[f"S_{fam}"] = s
            sc_[f"cutshare_{fam}"] = lost
        lin = {}
        Mx = M[:, :HC.N_X]
        for nm, I in IMP.items():
            if (f == "P") != (nm in ("yytot", "ydmtot_P")):
                continue
            k = 2 if nm in ("yytot", "dm_var", "ksz1h") else 1
            L = IC.linear_response(Mx, mstack[f], I, k)
            if np.ndim(L):
                for e in (100, 300, 1000, 3000):
                    lin[f"lin_{nm}_{e}"] = float(L[int(np.argmin(np.abs(ell - e)))])
            else:
                lin[f"lin_{nm}"] = float(L)
        for v in ("det", "S1"):
            if f == "P":
                mom = HPm.s1_moments(M, Et, s1) if v == "S1" else dict(P=M, E=Et)
            else:
                mom = HPm.s1_moments(Pt, M, s1) if v == "S1" else dict(P=Pt, E=M)
            o = B.scalars(B.observables(**mom))
            q = quant.get(f"{it['label']}|{v}")
            if q is not None:
                for p_, qn in enumerate(FC.Q_NAMES):
                    o[f"dm_{qn}"] = float(q["quantiles"][p_])
            err = {k: o[k] / truth[k] - 1 for k in o if k in truth and truth[k]}
            errm = {k: o[k] / tmean[k] - 1 for k in o if k in tmean and tmean[k]}
            row = dict(base, prediction=v, dm_quantiles=q is not None, **sc_, **lin)
            for fam in FAMS[f]:
                keys = [k for k in fams_all[fam] if k in err]
                if fam == "DM" and q is None:
                    row[f"rms_{fam}"] = np.nan
                    row[f"rms_{fam}_vs_mean"] = np.nan
                    row["err_dm_mean"] = err.get("dm_mean")
                    row["err_dm_var"] = err.get("dm_var")
                    continue
                row[f"rms_{fam}"] = float(np.sqrt(np.mean([err[k]**2 for k in keys]))) if keys else np.nan
                row[f"rms_{fam}_vs_mean"] = float(np.sqrt(np.mean([errm[k]**2 for k in keys]))) if keys else np.nan
            rows.append(row)
            if it["R_cut"] == 4.0 or abs(it["R_cut"] - (it.get("R_fit") or -1)) < 1e-9:
                for fam in FAMS[f]:
                    for k in fams_all[fam]:
                        if k in err:
                            long_rows.append(dict(label=it["label"], prediction=v, family=fam, observable=k,
                                                  error=err[k], error_vs_mean=errm.get(k)))
        if it["R_cut"] == 4.0:
            store[f"M4__{it['label']}"] = it["M4"][:, :HC.N_X]
        meta_items.append(base)
        if n_it % 50 == 0:
            IC.log(f"observables {n_it + 1}/{len(items)}")
    store["x_edges"] = HC.X200
    store["mstack_P"], store["mstack_E"] = mstack["P"], mstack["E"]
    for fam, w in W.items():
        store[f"w__{fam}"] = w
    store["s1"] = s1
    np.savez_compressed(IC.RESULTS / "evaluate.npz", **store)
    IC.write_json(IC.RESULTS / "evaluate_items.json", {"items": meta_items, "truth_a2ii": truth,
                                                       "truthmean_a2ii": tmean, "definitions": __doc__})
    IC.write_csv(IC.TABLES / "I_scores.csv", rows)
    IC.write_csv(IC.TABLES / "I_errors_long.csv", long_rows)
    IC.log("evaluate done")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("finefits", "evaluate"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args(argv)
    configure(a.workers, a.scratch)
    {"finefits": stage_finefits, "evaluate": stage_evaluate}[a.stage]()


if __name__ == "__main__":
    main()
