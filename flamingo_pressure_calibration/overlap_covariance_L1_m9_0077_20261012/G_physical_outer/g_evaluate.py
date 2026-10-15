"""G stage 'evaluate': every fit x R_cut x evaluation target through Test F's in-basis engine.

Evaluations (g_common.EVALS): 'all' (all-gas fit vs the all-gas truth), 'paint' (all-gas fit x p_keep vs the A2-ii
truth: the HalfDome painting target), 'a2ii' (fit to the A2-ii stacks vs the A2-ii truth).
Field-separated scoring: a pressure model is paired with the truth's mean electron profile (tSZ family), an electron
model with the truth's mean pressure (kSZ and DM-PDF families); tSZ x DM is scored for combinations of P and E
choices.  Predictions:
  det  one profile per bin;
  S1   Test F's lognormal amplitude (moments of per-halo Y(<4R), N_e(<4R); DM PDF: 9 Gauss-Hermite nodes);
  S2   two amplitudes, A_in on the bound component and A_out on the outer one (moments of per-halo integrals
       inside 1 R200c and over 1.5 R200c - R_cut; DM PDF: correlated GH2 x GH2 nodes), R_cut >= 2.
DM quantiles need a PDF pass per variant; the passes run in priority order and are checkpointed in work/
(rerunning the stage reuses finished passes with the same specification):
  1  det + S1, objective (a), R_fit = R_cut in {2, 3, 4}, evaluations all / paint
  2  det, objective (a), every other R_fit with R_cut in {2, 3, 4}; det, objective (d), R_cut = 4; a2ii det
  3  S2 (models with an outer component, objective a, 4/4, all / paint); S1 objective (d) 4/4 paint; S1 a2ii 4/4
Evaluations without a PDF pass (R_cut < 2, and S1/S2 outside the lists) carry the DM mean and variance only
(dm_quantiles_included = False) and get no DM family score.
Outputs: results/evaluate.npz, tables/G_scores.csv, tables/G_errors_long.csv (R_cut = 4 or R_fit = R_cut),
tables/G_cross_scores.csv, results/choices.json (candidate configurations for the CV stage).
"""
from __future__ import annotations

import hashlib

import f2_run as R
import f_basis as FB
import g_common as GC
import g_models as GM
import g_pdf as GP
import numpy as np

FAMILIES = {"P": ("tSZ",), "E": ("kSZ", "DM")}


# ============================================================================ second moments
def moments(variant, BP, OP, BE, OE, s1, M):
    """(P, E, SPP, SEE, SPE) for components (bound, outer) of both fields [nb, ns]; s1 [nb, 3]; M [nb, 4, 4]."""
    P, E = BP + OP, BE + OE
    outer = lambda a, b: np.einsum("cs,ct->cst", a, b)        # noqa: E731
    if variant == "det":
        return P, E, None, None, None
    if variant == "S1":
        return (P, E, np.exp(s1[:, 0])[:, None, None] * outer(P, P), np.exp(s1[:, 1])[:, None, None] * outer(E, E),
                s1[:, 2][:, None, None] * outer(P, E))
    m = lambda i, j: M[:, i, j][:, None, None]                 # noqa: E731
    SPP = m(0, 0) * outer(BP, BP) + m(0, 1) * (outer(BP, OP) + outer(OP, BP)) + m(1, 1) * outer(OP, OP)
    SEE = m(2, 2) * outer(BE, BE) + m(2, 3) * (outer(BE, OE) + outer(OE, BE)) + m(3, 3) * outer(OE, OE)
    SPE = m(0, 2) * outer(BP, BE) + m(0, 3) * outer(BP, OE) + m(1, 2) * outer(OP, BE) + m(1, 3) * outer(OP, OE)
    return P, E, SPP, SEE, SPE


def family_rms(err, fam):
    vals = [err[k] for k in GC.FC.FAMILIES[fam] if GC.FC.primary(k) and k in err and np.isfinite(err[k])]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else np.nan


def errors(sc, truth):
    err = {k: sc[k] / truth["obs"][k] - 1 for k in sc if k in truth["obs"] and truth["obs"][k]}
    errm = {k: sc[k] / truth["mean_obs"][k] - 1 for k in sc if k in truth["mean_obs"] and truth["mean_obs"][k]}
    return err, errm


def truths(pr, tr, ts):
    out = {}
    for sel in ("main", "a2ii"):
        t = R.truth_dict(pr, tr, ts, sel)
        t["Ppad"] = np.pad(t["P"], ((0, 0), (0, GC.N_B - GC.N_F)))
        t["Epad"] = np.pad(t["E"], ((0, 0), (0, GC.N_B - GC.N_F)))
        out[sel] = t
    return out


# ============================================================================ evaluation list
def evaluation_list(keys, r_cut):
    out = []
    for i, key in enumerate(keys):
        f, m, t, o, rf = GC.parse_key(key)
        for j, rc in enumerate(r_cut):
            for ev, fit_t, paint, tsel in GC.EVALS:
                if fit_t != t:
                    continue
                out.append({"label": f"{key}|Rc{rc:g}|{ev}", "i": i, "j": j, "field": f, "model": m, "fit_target": t,
                            "objective": o, "R_fit": rf, "R_cut": float(rc), "eval": ev, "paint": paint, "truth": tsel})
    return out


def comps(fz, e):
    if e["paint"]:
        return fz["Bp"][e["i"], e["j"]], fz["Op"][e["i"], e["j"]]
    return fz["B"][e["i"], e["j"]], fz["O"][e["i"], e["j"]]


def pdf_passes(evals):
    """Priority-ordered PDF passes: list of (name, [(eval, variant)])."""
    E = [e for e in evals if e["field"] == "E"]
    diag = lambda e: e["R_fit"] == e["R_cut"] and e["R_cut"] >= 2.0             # noqa: E731
    p1 = [(e, v) for e in E for v in ("det", "S1") if e["objective"] == "a" and diag(e) and e["eval"] in ("all", "paint")]
    taken = {(e["label"], v) for e, v in p1}
    p2 = [(e, "det") for e in E if (e["label"], "det") not in taken and e["R_cut"] >= 2.0
          and (e["objective"] == "a" or e["R_cut"] == 4.0)]
    taken |= {(e["label"], v) for e, v in p2}
    has_outer = {"E1", "E2g6", "E2gf", "E3"}
    p3 = [(e, "S2") for e in E if e["objective"] == "a" and e["R_fit"] == 4.0 and e["R_cut"] == 4.0
          and e["model"] in has_outer and e["eval"] in ("all", "paint")]
    p3 += [(e, "S1") for e in E if e["R_fit"] == 4.0 and e["R_cut"] == 4.0 and (e["label"], "S1") not in taken
           and ((e["objective"] == "d" and e["eval"] == "paint") or (e["objective"] == "a" and e["eval"] == "a2ii"))]
    return [("p1", p1), ("p2", p2), ("p3", p3)]


def pdf_specs(items, fz, prep, truth_E):
    specs = []
    for e, v in items:
        Bc, Oc = comps(fz, e)
        Ef = Bc + Oc
        if v == "det":
            per_bin = GP.det_entries(Ef)
        elif v == "S1":
            per_bin = GP.s1_entries(Ef, prep[f"S1_{e['truth']}"])
        else:
            per_bin = GP.s2_entries(Bc, Oc, prep[f"S2_{e['truth']}_Rc{e['R_cut']:g}"])
        specs.append((f"{e['label']}|{v}", per_bin))
    return specs


def spec_hash(specs):
    h = hashlib.sha256()
    for lab, per_bin in specs:
        h.update(lab.encode())
        for rows, w in per_bin:
            h.update(np.ascontiguousarray(np.asarray(rows, float)).tobytes())
            h.update(np.ascontiguousarray(np.asarray(w, float)).tobytes())
    return h.hexdigest()


def run_pdf_passes(evals, fz, prep, log=GC.log):
    fb = np.load(GC.WORK / "pdf_fallback_g.npz")
    fallback = (fb["hist"], float(fb["overflow"]))
    quant, info = {}, {}
    for name, items in pdf_passes(evals):
        if not items:
            continue
        specs = pdf_specs(items, fz, prep, None)
        digest = spec_hash(specs)
        path = GC.WORK / f"pdf_eval_{name}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            if str(z["digest"]) == digest:
                for lab, q in zip(z["labels"], z["quantiles"]):
                    quant[str(lab)] = q
                info[name] = {"reused": True, "histograms": len(specs)}
                log(f"PDF pass {name}: reused {len(specs)} histograms")
                continue
        q, mom, timings = GP.run_hists(specs, fallback, label=f"PDF pass {name}")
        labels = list(q)
        np.savez_compressed(path, digest=np.array(digest), labels=np.array(labels),
                            quantiles=np.array([q[k] for k in labels]),
                            moments=np.array([mom[k] for k in labels]))
        quant.update(q)
        info[name] = {"histograms": len(specs), "entries": int(sum(t["entries"] for t in timings)),
                      "wall_seconds": float(sum(t["wall_seconds"] for t in timings))}
        log(f"PDF pass {name}: {info[name]}")
    return quant, info


# ============================================================================ main
def evaluate_one(B, e, Bf, Of, truth, prep, variant, quant):
    """Observables, errors and score row of one evaluation and variant (field model + the truth's other field)."""
    f, tsel = e["field"], e["truth"]
    zero = np.zeros_like(Bf)
    if f == "P":
        BP, OP, BE, OE = Bf, Of, truth["Epad"], zero
    else:
        BP, OP, BE, OE = truth["Ppad"], zero, Bf, Of
    M = prep.get(f"S2_{tsel}_Rc{e['R_cut']:g}") if variant == "S2" else None
    P, E, SPP, SEE, SPE = moments(variant, BP, OP, BE, OE, prep[f"S1_{tsel}"], M)
    o = B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE)
    sc = B.scalars(o)
    q = quant.get(f"{e['label']}|{variant}")
    if q is not None:
        for p, qn in enumerate(GC.FC.Q_NAMES):
            sc[f"dm_{qn}"] = float(q[p])
    err, errm = errors(sc, truth)
    row = {k: e[k] for k in ("field", "model", "fit_target", "objective", "R_fit", "R_cut", "eval")}
    row.update(prediction=variant, dm_quantiles_included=q is not None)
    for fam in FAMILIES[f]:
        if fam == "DM" and q is None:
            row["rms_DM"] = np.nan
            row["rms_DM_vs_mean"] = np.nan
            row["DM_mean_var_only_rms"] = float(np.sqrt(np.mean([err["dm_mean"]**2, err["dm_var"]**2])))
            continue
        row[f"rms_{fam}"] = family_rms(err, fam)
        row[f"rms_{fam}_vs_mean"] = family_rms(errm, fam)
    if f == "E":
        row["score_E"] = np.nanmean([row.get("rms_kSZ", np.nan), row.get("rms_DM", np.nan)]) if q is not None else np.nan
    Ff = (Bf + Of)[:, :GC.N_F]
    tm = truth["P"] if f == "P" else truth["E"]
    return row, sc, err, errm, Ff, tm


def main():
    timer = GC.Timer()
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    tr = dict(np.load(GC.FBASE / "truth.npz"))
    ts = GC.read_json(GC.FBASE / "truth_summary.json")
    im = dict(np.load(GC.FBASE / "importance_F.npz"))
    prep = GC.load_prepare()
    fz = dict(np.load(GC.RESULTS / "fits.npz"))
    keys = [str(k) for k in fz["keys"]]
    r_cut = [float(r) for r in fz["r_cut"]]
    B = FB.Basis("T")
    T = truths(pr, tr, ts)
    evals = evaluation_list(keys, r_cut)
    GC.log(f"{len(evals)} evaluations")
    quant, pdf_info = run_pdf_passes(evals, fz, prep)
    GC.log("PDF passes done", round(timer.mark("pdf"), 1))
    base = pr["base_mask"][GC.CAL_BINS]
    wfam = {"P": {"tSZ": im["Wfam_P_tSZ"]}, "E": {"kSZ": im["Wfam_E_kSZ"], "DM": im["Wfam_E_DM"]}}
    rows, long_rows = [], []
    obs_names = None
    values, rel, relm, labels = [], [], [], []
    for e in evals:
        Bf, Of = comps(fz, e)
        truth = T[e["truth"]]
        variants = ["det", "S1"] + (["S2"] if e["R_cut"] >= 2.0 else [])
        for v in variants:
            row, sc, err, errm, Ff, tm = evaluate_one(B, e, Bf, Of, truth, prep, v, quant)
            res, _ = GC.FF.residual_metrics(Ff, tm, base, wfam[e["field"]])
            row.update({f"profile_{k}": val for k, val in res.items()})
            bud = B.budgets(Bf + Of, Bf + Of)
            tb = B.budgets(truth["P"], truth["E"])
            key4 = "Y" if e["field"] == "P" else "Ne"
            row["budget4R_rms_bins"] = float(np.sqrt(np.mean((bud[key4] / tb[key4] - 1)**2)))
            rows.append(row)
            if obs_names is None:
                obs_names = sorted(k for k in truth["obs"])
            labels.append(f"{e['label']}|{v}")
            values.append([sc.get(k, np.nan) for k in obs_names])
            rel.append([err.get(k, np.nan) for k in obs_names])
            relm.append([errm.get(k, np.nan) for k in obs_names])
            if e["R_cut"] == 4.0 or e["R_fit"] == e["R_cut"]:
                fams = FAMILIES[e["field"]]
                for k in obs_names:
                    if any(k in GC.FC.FAMILIES[fam] for fam in fams) and k in err:
                        long_rows.append({"label": e["label"], "prediction": v, "observable": k, "value": sc[k],
                                          "truth": truth["obs"][k], "rel_error": err[k],
                                          "rel_error_vs_truth_mean": errm.get(k, np.nan)})
    GC.log("field evaluations done", round(timer.mark("fields"), 1))
    # ---------------- tSZ x DM: combinations of P and E fits (same target, objective, R_fit = R_cut)
    cross = []
    idx = {k: i for i, k in enumerate(keys)}
    for ev, fit_t, paint, tsel in GC.EVALS:
        truth = T[tsel]
        for o in GC.OBJECTIVES:
            for rf in [r for r in GC.R_FIT if r >= 2.0]:
                j = r_cut.index(rf) if rf in r_cut else None
                if j is None:
                    continue
                for mp in GM.FIELD_MODELS["P"]:
                    kp = GC.fit_key("P", mp, fit_t, o, rf)
                    if kp not in idx:
                        continue
                    for me in GM.FIELD_MODELS["E"]:
                        ke = GC.fit_key("E", me, fit_t, o, rf)
                        if ke not in idx:
                            continue
                        ep = {"i": idx[kp], "j": j, "paint": paint}
                        ee = {"i": idx[ke], "j": j, "paint": paint}
                        BP, OP = comps(fz, ep)
                        BE, OE = comps(fz, ee)
                        for v in ("det", "S1", "S2"):
                            M = prep.get(f"S2_{tsel}_Rc{rf:g}") if v == "S2" else None
                            P, E, SPP, SEE, SPE = moments(v, BP, OP, BE, OE, prep[f"S1_{tsel}"], M)
                            sc = B.scalars(B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE))
                            err, errm = errors(sc, truth)
                            cross.append({"eval": ev, "objective": o, "R_fit": rf, "R_cut": rf, "P_model": mp,
                                          "E_model": me, "prediction": v, "rms_tSZxDM": family_rms(err, "tSZxDM"),
                                          "rms_tSZxDM_vs_mean": family_rms(errm, "tSZxDM"),
                                          "rms_tSZ": family_rms(err, "tSZ"), "rms_kSZ": family_rms(err, "kSZ")})
    GC.log("cross combinations done", round(timer.mark("cross"), 1))
    # ---------------- candidate choices for the CV stage (painting target, R_cut = 4)
    choices = candidate_choices(rows)
    GC.write_csv(GC.TABLES / "G_scores.csv", rows)
    GC.write_csv(GC.TABLES / "G_errors_long.csv", long_rows)
    GC.write_csv(GC.TABLES / "G_cross_scores.csv", cross)
    np.savez_compressed(GC.RESULTS / "evaluate.npz", labels=np.array(labels), observables=np.array(obs_names),
                        values=np.array(values), rel_error=np.array(rel), rel_error_vs_truth_mean=np.array(relm))
    GC.write_json(GC.RESULTS / "choices.json", choices)
    GC.write_json(GC.RESULTS / "evaluate_summary.json", {"evaluations": len(evals), "rows": len(rows),
                                                         "pdf": pdf_info, "seconds": timer.marks})
    GC.log("evaluate done", round(timer.mark("total"), 1))


def candidate_choices(rows):
    """Configurations cross-validated in the CV stage: fixed references + the best scoring ones (painting target,
    R_cut = 4, S1).  Pressure score: tSZ RMS; electrons: mean of the kSZ and DM-PDF RMS (DM quantiles required)."""
    fixed = {"P": [("P0", "a", 4.0), ("P1", "d", 4.0), ("P1", "a", 4.0), ("P2gf", "a", 4.0), ("P2m", "a", 4.0)],
             "E": [("E0", "a", 4.0), ("E1", "d", 4.0), ("E1", "a", 4.0), ("E2gf", "a", 4.0), ("E3", "a", 4.0)]}
    out = {"rule": "painting evaluation (all-gas fit x p_keep vs the A2-ii truth), R_cut = 4, prediction S1; "
                   "P: min tSZ RMS; E: min mean(kSZ RMS, DM RMS) among evaluations with exact DM quantiles",
           "fixed": fixed, "best": {}, "best_physical": {}}
    for f in ("P", "E"):
        cand = [r for r in rows if r["field"] == f and r["eval"] == "paint" and r["R_cut"] == 4.0 and r["prediction"] == "S1"]
        score = (lambda r: r.get("rms_tSZ", np.nan)) if f == "P" else (lambda r: r.get("score_E", np.nan))
        cand = [r for r in cand if np.isfinite(score(r))]
        if not cand:
            continue
        best = min(cand, key=score)
        out["best"][f] = (best["model"], best["objective"], best["R_fit"], float(score(best)))
        phys = [r for r in cand if r["model"] in GM.PHYSICAL]
        if phys:
            bp = min(phys, key=score)
            out["best_physical"][f] = (bp["model"], bp["objective"], bp["R_fit"], float(score(bp)))
    cv = {}
    for f in ("P", "E"):
        lst = list(fixed[f])
        for d in (out["best"], out["best_physical"]):
            if f in d and tuple(d[f][:3]) not in lst:
                lst.append(tuple(d[f][:3]))
        cv[f] = lst
    out["cv_candidates"] = cv
    return out


if __name__ == "__main__":
    main()
