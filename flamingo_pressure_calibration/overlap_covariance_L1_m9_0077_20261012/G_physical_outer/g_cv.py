"""G stage 'cv': spatial-block cross-validation of the per-field candidate choices (Test F5's folds).

Folds: Test F's 4^3 blocks of the 1 Gpc box in 8 folds (profiles.npz 'fold').  For every fold k everything that
is calibrated is redone on the training haloes (the other 7/8 of the volume): the stacks, the fits (global
parameters included), the 2-halo templates (training haloes' masses and radii), p_keep and the S1 / S2 scatter
tables.  Kept from the full sample: the importance maps of objective (d) and KAPPA_E (a gas-composition constant).
Out-of-fold prediction (F5's combination): per calibrated bin the fold models are weighted by the held-out
inclusion-weight shares omega[c, k]; linear terms sum_k omega P_k, second moments sum_k omega S_k (so the
fold-to-fold variance enters the 1-halo terms), DM PDF a mixture of the fold models' deposits.  Scored against the
full-sample truths (all gas; A2-ii for the painting evaluation).  Candidates: results/choices.json, R_cut = 4.
Outputs: tables/G_cv_scores.csv, tables/G_cv_fold_parameters.csv, results/cv_summary.json.
"""
from __future__ import annotations

import time

import f_basis as FB
import g_common as GC
import g_evaluate as GE
import g_fits as GFS
import g_models as GM
import g_pdf as GP
import g_physics as PH
import numpy as np

FOLDS = GC.FC.FOLDS
_S: dict = {}


def omega(pr):
    out = np.zeros((GC.N_CAL, FOLDS))
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][rows]
        out[c] = np.bincount(pr["fold"][rows], weights=w, minlength=FOLDS) / w.sum()
    return out


def train_rows(pr, k):
    out = []
    for b in GC.CAL_BINS:
        rows = np.flatnonzero(pr["mass_bin"] == b)
        out.append(rows[pr["fold"][rows] != k])
    return out


def fold_prep(pr, prep, k, lh):
    rows = train_rows(pr, k)
    tpl = PH.bin_templates(lh, pr, rows)
    p = dict(prep)
    p.update(T2h=tpl["T2h"], bin_b=tpl["b"], bin_logm=tpl["logm"])
    neff = np.zeros(GC.N_CAL)
    means = {}
    for f in ("P", "E"):
        for sel in ("main", "a2ii"):
            m = np.zeros((GC.N_CAL, GC.N_F))
            for c, r in enumerate(rows):
                w = pr["weight"][r] / pr["weight"][r].sum()
                m[c] = w @ np.nan_to_num(pr[f"{f}_{sel}"][r])
                neff[c] = 1.0 / np.sum(w**2)
            means[(f, sel)] = m
    pk = {}
    for f in ("P", "E"):
        fit = PH.fit_pkeep(means[(f, "main")], means[(f, "a2ii")], tpl["logm"], neff)
        pk[f] = fit["params"]
        p[f"pkeep_params_{f}"] = fit["params"]
    for sel in ("main", "a2ii"):
        s1, s2 = PH.scatter_tables(pr, sel, rows)
        p[f"S1_{sel}"] = s1
        for rc, M in s2.items():
            p[f"S2_{sel}_Rc{rc:g}"] = M
    return rows, tpl, pk, p


def _fold_job(k):
    pr, prep, imp, cands, lh = _S["pr"], _S["prep"], _S["imp"], _S["cands"], _S["lh"]
    t0 = time.perf_counter()
    rows, tpl, pk, p = fold_prep(pr, prep, k, lh)
    want_set = {(f, m, o, rf) for f, lst in cands.items() for (m, o, rf) in lst}
    if any(m == "P2i" for (_, m, _, _) in want_set):
        want_set.add(("E", "E3", "a", 4.0))

    def want(f, m, t, o, rf):
        return t == "main" and (f, m, o, rf) in want_set

    res, info = GFS.run(pr, p, imp, rows_by_bin=rows, templates=tpl, pkeep_params=pk, tasks_filter=want, gls=False,
                        workers=1, log=lambda *a: None, targets=("main",))
    keep = {}
    for key, r in res.items():
        f, m, t, o, rf = GC.parse_key(key)
        if (f, m, o, rf) in want_set:
            keep[key] = {"B": r["B"], "O": r["O"], "Bp": r["Bp"], "Op": r["Op"], "pb": r["pb"], "pg": r["pg"],
                         "names": GM.get(m).names()}
    scat = {k2: v for k2, v in p.items() if k2.startswith("S1_") or k2.startswith("S2_")}
    return k, keep, scat, pk, time.perf_counter() - t0


def main():
    timer = GC.Timer()
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    tr = dict(np.load(GC.FBASE / "truth.npz"))
    ts = GC.read_json(GC.FBASE / "truth_summary.json")
    imp = dict(np.load(GC.FBASE / "importance_F.npz"))
    prep = GC.load_prepare()
    choices = GC.read_json(GC.RESULTS / "choices.json")
    cands = {f: [tuple([v[0], v[1], float(v[2])]) for v in lst] for f, lst in choices["cv_candidates"].items()}
    lh = PH.LinearHalo()
    _S.update(pr=pr, prep=prep, imp=imp, cands=cands, lh=lh)
    folds, scat, pks, secs = {}, {}, {}, {}
    for k, keep, sc, pk, sec in GC.pool_map(_fold_job, range(FOLDS), ordered=False):
        folds[k], scat[k], pks[k], secs[k] = keep, sc, pk, sec
        GC.log(f"fold {k}: {len(keep)} fits in {sec:.0f} s")
    _S.clear()
    om = omega(pr)
    B = FB.Basis("T")
    T = GE.truths(pr, tr, ts)
    jcut = list(GC.R_CUT).index(4.0)
    rc = 4.0
    # ---------------- DM PDF mixtures for the electron candidates
    specs = []
    scatter_pdf = {("E1", "d", 4.0)}
    for d in (choices.get("best", {}), choices.get("best_physical", {})):
        if "E" in d:
            scatter_pdf.add((d["E"][0], d["E"][1], float(d["E"][2])))
    s2_pdf = {(v[0], v[1], float(v[2])) for v in [choices.get("best_physical", {}).get("E")] if v}
    for (m, o, rf) in cands.get("E", []):
        key = GC.fit_key("E", m, "main", o, rf)
        for ev, paint, tsel in (("all", False, "main"), ("paint", True, "a2ii")):
            for v in ("det", "S1", "S2"):
                if v == "S1" and not (ev == "paint" and (m, o, rf) in scatter_pdf):
                    continue
                if v == "S2" and not (ev == "paint" and (m, o, rf) in s2_pdf and m != "E0"):
                    continue
                per_fold = []
                for k in range(FOLDS):
                    fk = folds[k][key]
                    Bc = fk["Bp" if paint else "B"][jcut]
                    Oc = fk["Op" if paint else "O"][jcut]
                    if v == "det":
                        per_fold.append(GP.det_entries(Bc + Oc))
                    elif v == "S1":
                        per_fold.append(GP.s1_entries(Bc + Oc, scat[k][f"S1_{tsel}"]))
                    else:
                        per_fold.append(GP.s2_entries(Bc, Oc, scat[k][f"S2_{tsel}_Rc{rc:g}"]))
                specs.append((f"cv|{key}|{ev}|{v}", GP.mixture_entries(per_fold, om)))
    fb = np.load(GC.WORK / "pdf_fallback_g.npz")
    quant, _, timings = GP.run_hists(specs, (fb["hist"], float(fb["overflow"])), label="CV PDF") if specs else ({}, {}, [])
    GC.log("CV PDF done", round(timer.mark("pdf"), 1))
    # ---------------- out-of-fold scores
    rows = []
    for f, lst in cands.items():
        for (m, o, rf) in lst:
            key = GC.fit_key(f, m, "main", o, rf)
            for ev, paint, tsel in (("all", False, "main"), ("paint", True, "a2ii")):
                truth = T[tsel]
                for v in ("det", "S1", "S2"):
                    P = np.zeros((GC.N_CAL, GC.N_B)); E = np.zeros_like(P)
                    SPP = np.zeros((GC.N_CAL, GC.N_B, GC.N_B)); SEE = np.zeros_like(SPP); SPE = np.zeros_like(SPP)
                    for k in range(FOLDS):
                        fk = folds[k][key]
                        Bf = fk["Bp" if paint else "B"][jcut]
                        Of = fk["Op" if paint else "O"][jcut]
                        zero = np.zeros_like(Bf)
                        if f == "P":
                            BP, OP, BE, OE = Bf, Of, truth["Epad"], zero
                        else:
                            BP, OP, BE, OE = truth["Ppad"], zero, Bf, Of
                        M = scat[k].get(f"S2_{tsel}_Rc{rc:g}")
                        pk_, ek_, spp, see, spe = GE.moments("det" if v == "det" else v, BP, OP, BE, OE,
                                                             scat[k][f"S1_{tsel}"], M)
                        outer = lambda a, b: np.einsum("cs,ct->cst", a, b)       # noqa: E731
                        spp = outer(pk_, pk_) if spp is None else spp
                        see = outer(ek_, ek_) if see is None else see
                        spe = outer(pk_, ek_) if spe is None else spe
                        w = om[:, k]
                        P += w[:, None] * pk_
                        E += w[:, None] * ek_
                        SPP += w[:, None, None] * spp
                        SEE += w[:, None, None] * see
                        SPE += w[:, None, None] * spe
                    sc = B.scalars(B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE))
                    q = quant.get(f"cv|{key}|{ev}|{v}")
                    if q is not None:
                        for p_, qn in enumerate(GC.FC.Q_NAMES):
                            sc[f"dm_{qn}"] = float(q[p_])
                    err, _ = GE.errors(sc, truth)
                    row = {"field": f, "model": m, "objective": o, "R_fit": rf, "R_cut": rc, "eval": ev, "prediction": v,
                           "dm_quantiles_included": q is not None}
                    for fam in GE.FAMILIES[f]:
                        row[f"rms_{fam}"] = GE.family_rms(err, fam) if (fam != "DM" or q is not None) else np.nan
                    for k2 in ("yytot_100", "yytot_1000", "yytot_3000", "ksz1h_100", "ksz1h_1000", "ksz1h_3000",
                               "dm_mean", "dm_var", "dm_q16", "dm_q50", "dm_q999"):
                        if k2 in err:
                            row[f"err_{k2}"] = err[k2]
                    rows.append(row)
    # ---------------- parameter stability across folds
    stab = []
    for k in range(FOLDS):
        for key, fk in folds[k].items():
            f, m, t, o, rf = GC.parse_key(key)
            nb, ng = fk["names"]
            row = {"fold": k, "key": key, **{f"g_{n}": fk["pg"][i] for i, n in enumerate(ng)}}
            if "x_t" in nb:
                xt = fk["pb"][:, nb.index("x_t")]
                row.update({f"x_t_{GC.CAL_LABELS[c]}": xt[c] for c in range(GC.N_CAL)})
            stab.append(row)
        stab.append({"fold": k, "key": "pkeep_P", **{f"p{i}": v for i, v in enumerate(pks[k]["P"])}})
        stab.append({"fold": k, "key": "pkeep_E", **{f"p{i}": v for i, v in enumerate(pks[k]["E"])}})
    GC.write_csv(GC.TABLES / "G_cv_scores.csv", rows)
    GC.write_csv(GC.TABLES / "G_cv_fold_parameters.csv", stab)
    GC.write_json(GC.RESULTS / "cv_summary.json", {"omega": om, "candidates": cands, "fold_seconds": secs,
                                                   "pdf_timing": timings, "seconds": timer.marks})
    GC.log("cv done", round(timer.mark("total"), 1))


if __name__ == "__main__":
    main()
