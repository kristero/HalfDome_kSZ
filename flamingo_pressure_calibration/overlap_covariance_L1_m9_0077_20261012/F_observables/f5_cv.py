"""F5: spatial-block cross-validation (4^3 blocks of the 1 Gpc box -> 8 folds of 8 blocks).

For every fold k the whole calibration is redone on the training haloes (the other 7/8 of the volume):
training truth (means, second moments, per-halo DM PDF), importance weights and metric K, GLS covariances,
the hybrid lambda, the objective-(a) (R_fit, R_cut) choice per observable family and every F2-F4 method.
Nothing is tuned on the held-out fold.  Scoring is out-of-fold: each held-out halo is predicted by the
model trained without its fold; per bin the predictions are combined with the held-out inclusion-weight
shares omega[c, k] (linear terms: sum_k omega P_k; 1h terms and DM variance: sum_k omega P_k P_k^T; DM PDF:
sum_k omega x intensity(P_k)) and compared with the full truth (which is exactly the omega-weighted sum of
the held-out truths).  Mass-trend stability: fitted parameters per bin across folds.
Passes: (1) per-(fold, bin) per-halo DM intensities -> training-truth quantiles; (2) per-fold fits
(3 workers over folds); (3) one PDF pass: training-set predictions of the 48 objective-(a) grid models
(config selection) and the out-of-fold combinations.
Outputs: products/cv.npz, products/cv_summary.json, tables/F5_cv_*.csv.
"""
from __future__ import annotations

import multiprocessing
import time

import f_basis as FB
import f_common as FC
import f_fit as FF
import f_methods as FM
import f_pdf as FP
import f2_run as R
import numpy as np

FOLDS = FC.FOLDS
CV_METHODS = None          # all labels except the sigmoid duplicates and the lambda scan
SCATTER_CV = ("tab_stack", "grid_a_Rf2_Rc4")
_G: dict = {}


def omega(pr):
    """[10, 8] held-out inclusion-weight share of each fold in each calibrated bin."""
    out = np.zeros((FC.N_CAL, FOLDS))
    for c, b in enumerate(FC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        w = pr["weight"][rows]
        out[c] = np.bincount(pr["fold"][rows], weights=w, minlength=FOLDS) / w.sum()
    return out


def train_rows(pr, k):
    out = []
    for b in FC.CAL_BINS:
        rows = np.flatnonzero(pr["mass_bin"] == b)
        out.append(rows[pr["fold"][rows] != k])
    return out


# ============================================================================ per-fold fits
def _fold_job(k):
    pr, B, tr = _G["pr"], _G["B"], _G["tr"]
    rows = train_rows(pr, k)
    truth = _G["train_truth"][k]
    imp = _G["train_imp"][k]
    t0 = time.perf_counter()
    models, records, info = FM.fit_all(pr, "main", imp, B, truth, rows_by_bin=rows, seed=FC.SEED + 1000 * (k + 1))
    keep = {lab: m for lab, m in models.items() if not lab.startswith("sig_") and not lab.startswith("lam_")}
    params = [r for r in records if r.get("method") in ("a", "direct_trend") and r.get("R_fit") in (2.0, 4.0)]
    return k, keep, params, info, time.perf_counter() - t0


def training_truth(pr, B, k, Qtrain, Qdet_k, Jq_k):
    rows = train_rows(pr, k)
    P = np.zeros((FC.N_CAL, FC.N_F)); E = np.zeros_like(P)
    SPP = np.zeros((FC.N_CAL, FC.N_F, FC.N_F)); SEE = np.zeros_like(SPP); SPE = np.zeros_like(SPP)
    for c, r in enumerate(rows):
        w = pr["weight"][r] / pr["weight"][r].sum()
        vp, ve = pr["P_main"][r], pr["E_main"][r]
        P[c], E[c] = w @ vp, w @ ve
        SPP[c] = np.einsum("h,hs,ht->st", w, vp, vp)
        SEE[c] = np.einsum("h,hs,ht->st", w, ve, ve)
        SPE[c] = np.einsum("h,hs,ht->st", w, vp, ve)
    full = B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE)
    mean = B.observables(P, E)
    obs = B.scalars(full)
    mobs = B.scalars(mean)
    for p, q in enumerate(FC.Q_NAMES):
        obs[f"dm_{q}"] = float(Qtrain[p])
        mobs[f"dm_{q}"] = float(Qdet_k[p])
    truth = {"P": P, "E": E, "SPP": SPP, "SEE": SEE, "SPE": SPE, "obs": obs, "mean_obs": mobs,
             "Qdet": np.asarray(Qdet_k, float), "Jq_det": Jq_k}
    Jt = B.jacobians(P, E, SPP=SPP, SEE=SEE, SPE=SPE)
    Jd = B.jacobians(P, E)
    gs = FF.jacobian_maps(Jt, Jq_k, full, Qtrain)
    gd = FF.jacobian_maps(Jd, Jq_k, full, Qtrain)
    imp = {}
    for f in ("P", "E"):
        imp[f"w_{f}"] = FF.field_weights(gs, f)[0]
        imp[f"K_{f}"] = FF.field_metric(gd, f)
    return truth, imp


def lognormal_train(pr, k, aperture=1.0):
    rows = train_rows(pr, k)
    out = np.zeros((FC.N_CAL, 3))
    for c, r in enumerate(rows):
        w = pr["weight"][r] / pr["weight"][r].sum()
        Y = pr[f"P_main_int{aperture:g}R"][r]
        Ne = pr[f"E_main_int{aperture:g}R"][r]
        out[c] = (np.log(w @ Y**2 / (w @ Y)**2), np.log(w @ Ne**2 / (w @ Ne)**2), (w @ (Y * Ne)) / ((w @ Y) * (w @ Ne)))
    return out


def main():
    timer = FC.Timer()
    pr = dict(np.load(FC.PRODUCTS / "profiles.npz"))
    tr = dict(np.load(FC.PRODUCTS / "truth.npz"))
    ts = FC.read_json(FC.PRODUCTS / "truth_summary.json")
    B = FB.Basis("T")
    fb = np.load(FC.WORK / "pdf_fallback.npz")
    fbh, fbo = fb["hist"], float(fb["overflow"])
    om = omega(pr)
    # ---------------- pass 1: per-(fold, bin) per-halo intensities -> training-truth quantiles per fold
    ent = FP.Entries(B.ns)
    hid = {}
    for k in range(FOLDS):
        for c in range(FC.N_CAL):
            hid[(k, c)] = ent.new_hist(f"fold{k}_bin{c}")
    wsum = np.zeros((FOLDS, FC.N_CAL))
    for c, b in enumerate(FC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        for k in range(FOLDS):
            sel = rows[pr["fold"][rows] == k]
            wsum[k, c] = pr["weight"][sel].sum()
            if len(sel):
                ent.add(c, pr["E_main"][sel], hid[(k, c)], pr["weight"][sel])
    ent.freeze()
    raw = FP.run(ent, "T", include_fallback=False, fallback=(fbh, fbo), return_raw=True)
    H, OV = raw["raw_hist"], raw["raw_overflow"]
    Qtrain = np.zeros((FOLDS, len(FC.QUANTILES)))
    Qcheck = None
    for k in range(FOLDS):
        inten = np.zeros(FP.DM_POINTS)
        over = 0.0
        for c in range(FC.N_CAL):
            wt = wsum[[j for j in range(FOLDS) if j != k], c].sum()
            for j in range(FOLDS):
                if j != k:
                    inten += H[hid[(j, c)]] / wt
                    over += OV[hid[(j, c)]] / wt
        Qtrain[k] = FP.summarise(inten + fbh, over + fbo)["quantiles"]
    # check: all folds pooled reproduces the full truth quantiles
    inten = sum(H[hid[(j, c)]] / wsum[:, c].sum() for j in range(FOLDS) for c in range(FC.N_CAL))
    over = sum(OV[hid[(j, c)]] / wsum[:, c].sum() for j in range(FOLDS) for c in range(FC.N_CAL))
    Qcheck = FP.summarise(inten + fbh, over + fbo)["quantiles"]
    q_full = np.asarray(ts["pdf"]["truth_main"]["quantiles"])
    FC.log("pass 1 done", timer.mark("pass1"), "pooled-fold check", np.max(np.abs(Qcheck / q_full - 1)))
    # ---------------- training truths and importance per fold
    train_truth, train_imp = {}, {}
    for k in range(FOLDS):
        Qdet_k = np.asarray(ts["pdf"][f"trainmean_fold{k}"]["quantiles"], float)
        train_truth[k], train_imp[k] = training_truth(pr, B, k, Qtrain[k], Qdet_k, tr[f"Jq_trainmean_fold{k}"])
    # ---------------- pass 2: per-fold fits (3 workers over folds)
    _G.update(pr=pr, B=B, tr=tr, train_truth=train_truth, train_imp=train_imp)
    fold_models, fold_params, fold_info, fold_seconds = {}, {}, {}, {}
    for k, keep, params, info, sec in FC.pool_map(_fold_job, range(FOLDS), ordered=False):
        fold_models[k], fold_params[k], fold_info[k], fold_seconds[k] = keep, params, info, sec
        FC.log(f"fold {k} fitted in {sec:.0f} s ({len(keep)} models)")
    _G.clear()
    labels = sorted(fold_models[0])
    FC.log("fits done", timer.mark("fits"))
    # ---------------- pass 3a: training-set predictions of the grid_a models (selection on training data only)
    grid_a = [lab for lab in labels if lab.startswith("grid_a_")]
    ent = FP.Entries(B.ns)
    sel_hist = {}
    for k in range(FOLDS):
        for lab in grid_a:
            h = ent.new_hist(f"train{k}|{lab}")
            sel_hist[(k, lab)] = h
            for c in range(FC.N_CAL):
                ent.add(c, fold_models[k][lab]["E"][c], h, 1.0)
    ent.freeze()
    res_a = FP.run(ent, "T", include_fallback=False, fallback=(fbh, fbo))
    q_sel = {key: res_a["summaries"][h]["quantiles"] for key, h in sel_hist.items()}
    FC.log("pass 3a done", timer.mark("pass3a"))
    selection = {}
    for k in range(FOLDS):
        t = train_truth[k]
        best = {}
        for lab in grid_a:
            m = fold_models[k][lab]
            sc = B.scalars(B.observables(m["P"], m["E"]))
            for p, qn in enumerate(FC.Q_NAMES):
                sc[f"dm_{qn}"] = float(q_sel[(k, lab)][p])
            err = {kk: sc[kk] / t["obs"][kk] - 1 for kk in sc if kk in t["obs"]}
            fs = R.family_scores(err)
            for fam in FC.FAMILIES:
                v = fs[f"rms_{fam}"]
                if fam not in best or v < best[fam][0]:
                    best[fam] = (v, lab)
        selection[k] = {fam: v[1] for fam, v in best.items()}
    # ---------------- pass 3b: out-of-fold combinations (curated methods, selected configs, scatter)
    curated = [lab for lab in labels if not lab.startswith("grid_")] + \
        [f"grid_{o}_Rf{rf:g}_Rc4" for o in "acdef" for rf in (2.0, 3.0, 4.0)]
    curated = [lab for lab in curated if lab in labels]
    ent = FP.Entries(B.ns)
    cv_hist = {}

    def add_combo(name, chooser, scatter=False):
        h = ent.new_hist(f"cv|{name}")
        cv_hist[name] = h
        for c in range(FC.N_CAL):
            for k in range(FOLDS):
                if om[c, k] <= 0:
                    continue
                Ek = fold_models[k][chooser(k)]["E"][c]
                if scatter:
                    amp, wq = R.gauss_hermite(lnm[k][c, 1])
                    ent.add(c, amp[:, None] * Ek[None, :], h, om[c, k] * wq)
                else:
                    ent.add(c, Ek, h, om[c, k])

    lnm = {k: lognormal_train(pr, k) for k in range(FOLDS)}
    for lab in curated:
        add_combo(lab, lambda k, lab=lab: lab)
    for fam in FC.FAMILIES:
        add_combo(f"selected_a_{fam}", lambda k, fam=fam: selection[k][fam])
    for lab in SCATTER_CV:
        add_combo(f"{lab}|lognormal_1R", lambda k, lab=lab: lab, scatter=True)
    for fam in FC.FAMILIES:
        add_combo(f"selected_a_{fam}|lognormal_1R", lambda k, fam=fam: selection[k][fam], scatter=True)
    ent.freeze()
    res = FP.run(ent, "T", include_fallback=False, fallback=(fbh, fbo))
    q_of = {lab: res["summaries"][h]["quantiles"] for lab, h in cv_hist.items()}
    FC.log("pass 3b done", timer.mark("pass3b"))
    # ---------------- out-of-fold combinations and scores against the full truth
    truth = R.truth_dict(pr, tr, ts, "main")
    rows_out, score_rows = [], []

    def combine(getter):
        P = np.zeros((FC.N_CAL, FC.N_B)); E = np.zeros_like(P)
        SPP = np.zeros((FC.N_CAL, FC.N_B, FC.N_B)); SEE = np.zeros_like(SPP); SPE = np.zeros_like(SPP)
        for k in range(FOLDS):
            Pk, Ek, sp, se, sx = getter(k)
            wk = om[:, k]
            P += wk[:, None] * Pk
            E += wk[:, None] * Ek
            SPP += wk[:, None, None] * sp
            SEE += wk[:, None, None] * se
            SPE += wk[:, None, None] * sx
        return P, E, SPP, SEE, SPE

    def getter(chooser, scatter=False):
        def g(k):
            m = fold_models[k][chooser(k)]
            a = np.exp(lnm[k][:, 0])[:, None, None] if scatter else 1.0
            b = np.exp(lnm[k][:, 1])[:, None, None] if scatter else 1.0
            x = lnm[k][:, 2][:, None, None] if scatter else 1.0
            return (m["P"], m["E"], a * np.einsum("cs,ct->cst", m["P"], m["P"]),
                    b * np.einsum("cs,ct->cst", m["E"], m["E"]), x * np.einsum("cs,ct->cst", m["P"], m["E"]))
        return g

    def score(name, comb, quantiles, kind):
        P, E, SPP, SEE, SPE = comb
        o = B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE)
        sc = B.scalars(o)
        if quantiles is not None:
            for p, qn in enumerate(FC.Q_NAMES):
                sc[f"dm_{qn}"] = float(quantiles[p])
        err = {kk: sc[kk] / truth["obs"][kk] - 1 for kk in sc if kk in truth["obs"] and truth["obs"][kk]}
        for kk in sorted(err):
            rows_out.append({"model": name, "kind": kind, "observable": kk, "value": sc[kk], "truth": truth["obs"][kk],
                             "rel_error": err[kk]})
        score_rows.append({"model": name, "kind": kind, "dm_quantiles_included": quantiles is not None,
                           **R.family_scores(err)})

    for lab in labels:
        if lab in q_of or lab.startswith("grid_a_"):
            score(lab, combine(getter(lambda k, lab=lab: lab)), q_of.get(lab), "fixed method")
    for fam in FC.FAMILIES:
        name = f"selected_a_{fam}"
        score(name, combine(getter(lambda k, fam=fam: selection[k][fam])), q_of[name], "training-selected (a) config")
        score(name + "|lognormal_1R", combine(getter(lambda k, fam=fam: selection[k][fam], True)),
              q_of[name + "|lognormal_1R"], "training-selected (a) config + lognormal scatter")
    for lab in SCATTER_CV:
        score(f"{lab}|lognormal_1R", combine(getter(lambda k, lab=lab: lab, True)), q_of[f"{lab}|lognormal_1R"],
              "fixed method + lognormal scatter")
    # ---------------- stability of the fitted parameters across folds
    stab = []
    for k in range(FOLDS):
        for r in fold_params[k]:
            stab.append({"fold": k, **r})
    FC.write_csv(FC.TABLES / "F5_cv_errors_long.csv", rows_out)
    FC.write_csv(FC.TABLES / "F5_cv_scores.csv", score_rows)
    FC.write_csv(FC.TABLES / "F5_cv_fold_parameters.csv", stab)
    FC.write_json(FC.PRODUCTS / "cv_summary.json", {
        "omega": om, "training_quantiles": Qtrain, "pooled_fold_quantile_check": float(np.max(np.abs(Qcheck / q_full - 1))),
        "selection": selection, "lambda_choice": {k: fold_info[k]["lambda_choice"] for k in fold_info},
        "fold_fit_seconds": fold_seconds, "pdf_timing": {"3a": res_a["timing"], "3b": res["timing"]},
        "seconds": timer.marks})
    FC.log("cv done", timer.mark("total"))


if __name__ == "__main__":
    main()
