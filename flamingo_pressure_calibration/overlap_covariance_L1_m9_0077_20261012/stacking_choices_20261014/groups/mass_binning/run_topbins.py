"""Mass-binning group, part 5: the sparse top bins (15.1-15.3: 21 haloes, 15.3-15.5: 6) and minimum haloes per bin.

* precision of the individual top-bin stacks (paired global bootstrap) and leave-one-out influence;
* pooled stacks: >= 15.1 (27 haloes) and >= 14.9 (154 haloes);
* trend fit: per shell, weighted least squares of ln F against log M200c over the four bins 14.3-15.1
  (weights 1 / bootstrap variance of ln F), extrapolated to the top bins' mean masses (paired replicates refit);
* observable impact of each treatment of the top bins (first order, paired bootstrap);
* share of each observable carried by each calibrated bin (sensitivity to a uniform 1% change of one bin);
* continuous painting (every calibrated halo painted at its own mass through the node set) for each treatment, vs the
  truth and vs the reference painted the same way;
* statistical error of the observables from the finite reference sample: census-aware bootstrap (default), the
  core's stratified design, and delete-one jackknife of the census bins.
Outputs: work/res_topbins.json, work/web_topbins.json.
"""
from mbcommon import (BAND_LABELS, CAL_LABELS, CENSUS_BINS, EDGES_FIT, FIELDS_WEB, N_BOOT, OBS_FIELDS, OBS_KEEP, TAB, W,
                      WORK, band_ratios, continuous_paint, dump, impacts, interp_lnF, load, record_from_stack, ref_key,
                      sc, sig, sig_list, web_record, wmean_boot, xvol)
import numpy as np

ST = np.load(WORK / "scheme_stacks.npz")
REF = "M200c_0.2"
labels = list(ST[f"P__{REF}__labels"])
TOP = ("15.1-15.3", "15.3-15.5")
NODES = ("14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1")
bins = {b["label"]: b for b in sc.mass_bins(REF)}


def ref_arrays(field):
    return (ST[f"{field}__{REF}__value"], ST[f"{field}__{REF}__boot"], ST[f"{field}__{REF}__strict"],
            ST[f"{field}__{REF}__m200"])


def pooled_bin(labs):
    rows = np.concatenate([bins[lab]["rows"] for lab in labs])
    w = TAB["weight"][rows]
    lo, hi = bins[labs[0]]["lo"], bins[labs[-1]]["hi"]
    return {"scheme": REF, "label": f"{lo:.1f}-{hi:.1f}", "index": bins[labs[0]]["index"], "lo": lo, "hi": hi,
            "rows": rows, "weights": w, "mean_log_mass": float(np.average(TAB["logm200c"][rows], weights=w)),
            "mean_logm200c": float(np.average(TAB["logm200c"][rows], weights=w)), "N": int(len(rows)),
            "N_eff": float(w.sum()**2 / (w**2).sum()), "sum_weight": float(w.sum()),
            "strata": np.concatenate([bins[lab]["strata"] for lab in labs])}


POOLS = {"pool_15.1": ("15.1-15.3", "15.3-15.5"), "pool_14.9": ("14.9-15.1", "15.1-15.3", "15.3-15.5")}
pools = {k: pooled_bin(v) for k, v in POOLS.items()}


def trend_fit(field, m_targets):
    """Per shell WLS of ln F on log M200c over NODES; returns (pred [T, S], pred replicates [B, T, S], n_nodes [S])."""
    V, B, Sm, M = ref_arrays(field)
    ii = [labels.index(n) for n in NODES]
    x = M[ii]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.where(V[ii] > 0, V[ii], np.nan))                          # [K, S]
        yb = np.log(np.where(B[ii] > 0, B[ii], np.nan))                         # [K, Bt, S]
        var = np.nanvar(yb, axis=1, ddof=1)                                     # [K, S]
    res_nodes = Sm[ii]                                                          # [K, S]
    S = V.shape[1]
    pred = np.full((len(m_targets), S), np.nan)
    predb = np.full((B.shape[1], len(m_targets), S), np.nan)
    nn = res_nodes.sum(axis=0)
    for s in range(S):
        use = res_nodes[:, s] & np.isfinite(y[:, s]) & (var[:, s] > 0)
        if use.sum() < 2:
            continue
        wv = 1.0 / var[use, s]
        X = np.vstack([np.ones(use.sum()), x[use] - 14.7]).T
        A = X.T @ (wv[:, None] * X)
        Ainv = np.linalg.inv(A)
        coef = Ainv @ X.T @ (wv * y[use, s])
        cb = (Ainv @ X.T @ (wv[:, None] * yb[use][:, :, s])).T                  # [Bt, 2]
        for t, mt in enumerate(m_targets):
            pred[t, s] = np.exp(coef[0] + coef[1] * (mt - 14.7))
            predb[:, t, s] = np.exp(cb[:, 0] + cb[:, 1] * (mt - 14.7))
    return pred, predb, nn


res = {"individual": {}, "pooled": {}, "trend": {}, "impacts": {}, "bin_shares": {}, "notes": {}}
web = []
for field in FIELDS_WEB:
    V, B, Sm, M = ref_arrays(field)
    F = load(ref_key(field))
    # ---- individual top bins: relative bootstrap error per band and leave-one-out influence
    for lab in TOP + ("14.9-15.1",):
        i = labels.index(lab)
        rel = B[i].std(axis=0, ddof=1) / V[i]
        b = bins[lab]
        loo = []
        for k in range(b["N"]):
            keep = np.delete(b["rows"], k)
            loo.append(np.average(np.asarray(F[keep], float), weights=TAB["weight"][keep], axis=0) / V[i] - 1)
        loo = np.array(loo)
        e = {}
        for blab, idx in zip(BAND_LABELS, [np.flatnonzero((xvol(EDGES_FIT) >= lo) & (xvol(EDGES_FIT) < hi))
                                           for lo, hi in ((0, .1), (.1, .25), (.25, .5), (.5, 1), (1, 1.5), (1.5, 2),
                                                          (2, 3), (3, 4))]):
            sel = idx[Sm[i][idx]]
            if not len(sel):
                e[blab] = None
                continue
            e[blab] = {"rel_err": sig(np.median(rel[sel])),
                       "loo_max_abs": sig(np.median(np.max(np.abs(loo[:, sel]), axis=0))),
                       "jackknife_se": sig(np.median(np.sqrt((b["N"] - 1) / b["N"] * ((loo[:, sel] - loo[:, sel].mean(
                           axis=0))**2).sum(axis=0))))}
        res["individual"].setdefault(lab, {})[field] = e
        res["individual"][lab]["N"] = b["N"]
    # ---- pooled stacks: web records + band ratios of each top bin to the pooled stack (paired)
    for pk, pb in pools.items():
        rr = sc.stack_bin(ref_key(field), pb, "mean", "strict", n_boot=N_BOOT)
        web.append(record_from_stack(f"mass_binning:{pk}", field, REF, pb, rr))
        v, bt, _ = wmean_boot(F, pb["rows"])
        out = {"N": pb["N"], "mean_logm200c": sig(pb["mean_logm200c"]), "vs_bins": {}}
        for lab in POOLS[pk]:
            i = labels.index(lab)
            out["vs_bins"][lab] = band_ratios(v, V[i], bt, B[i], Sm[i] & np.asarray(rr["resolved"], bool))
        res["pooled"].setdefault(pk, {})[field] = out
    # ---- trend fit and extrapolation
    mt = [bins[lab]["mean_logm200c"] for lab in TOP]
    pred, predb, nn = trend_fit(field, mt)
    tr = {"nodes": NODES, "n_nodes_resolved_per_shell": [int(v) for v in nn]}
    for t, lab in enumerate(TOP):
        i = labels.index(lab)
        resolved = Sm[i] & (nn >= 2) & np.isfinite(pred[t])
        tr[lab] = {"measured_over_trend": band_ratios(V[i], pred[t], B[i], predb[:, t], resolved),
                   "trend_over_measured": band_ratios(pred[t], V[i], predb[:, t], B[i], resolved)}
        b = bins[lab]
        err = predb[:, t].std(axis=0, ddof=1)
        # N, N_eff: the haloes of the bin that would be painted with the prediction; n_used: haloes of the node bins
        # that enter the fit in each shell (resolved nodes; 0 where fewer than 2 nodes are resolved)
        ii = [labels.index(n) for n in NODES]
        n_fit = np.sum([Sm[k] * bins[n]["N"] for k, n in zip(ii, NODES)], axis=0) * (nn >= 2)
        web.append(web_record("mass_binning:trend_top", field, REF, lab, b["mean_log_mass"], b["N"],
                              b["N_eff"], EDGES_FIT, pred[t], err,
                              np.full(35, np.nan), np.full(35, np.nan), n_fit.astype(int), (nn >= 2)))
    res["trend"][field] = tr

# ---------------------------------------------------------------------------------------------- observable impacts
cal_idx = [labels.index(c) for c in CAL_LABELS]
for field in OBS_FIELDS:
    V, B, Sm, M = ref_arrays(field)
    F = load(ref_key(field))
    Fr, Br = V[cal_idx], np.moveaxis(B[cal_idx], 0, 1)
    variants = {}
    # (a) pooled >= 15.1, step: both top bins painted with the pooled stack
    pv, pb_, _ = wmean_boot(F, pools["pool_15.1"]["rows"])
    Fv, Bv = Fr.copy(), Br.copy()
    for lab in TOP:
        Fv[CAL_LABELS.index(lab)] = pv
        Bv[:, CAL_LABELS.index(lab)] = pb_
    variants["pool_15.1_step"] = (Fv, Bv)
    # (b) pooled >= 15.1 as a node at its mean mass; smooth painting (ln F linear in log M) through the nodes
    #     ... 14.7-14.9, 14.9-15.1, pooled (extrapolated above the pooled node's mass)
    nodes_m = np.concatenate([M[:13], [pools["pool_15.1"]["mean_logm200c"]]])
    nodes_v = np.vstack([V[:13], pv[None]])
    nodes_b = np.concatenate([B[:13], pb_[None]], axis=0)
    Fv, Bv = Fr.copy(), Br.copy()
    for lab in TOP:
        m = bins[lab]["mean_logm200c"]
        Fv[CAL_LABELS.index(lab)] = interp_lnF(nodes_m, nodes_v, m)[0]
        Bv[:, CAL_LABELS.index(lab)] = interp_lnF(nodes_m, np.moveaxis(nodes_b, 1, 0), m)[0]
    variants["pool_15.1_smooth"] = (Fv, Bv)
    # (c) pooled >= 14.9 (step: the three top bins painted with it)
    pv9, pb9, _ = wmean_boot(F, pools["pool_14.9"]["rows"])
    Fv, Bv = Fr.copy(), Br.copy()
    for lab in POOLS["pool_14.9"]:
        Fv[CAL_LABELS.index(lab)] = pv9
        Bv[:, CAL_LABELS.index(lab)] = pb9
    variants["pool_14.9_step"] = (Fv, Bv)
    # (d) trend extrapolation from 14.3-15.1 (unresolved / unfitted shells keep the measured values)
    pred, predb, nn = trend_fit(field, [bins[lab]["mean_logm200c"] for lab in TOP])
    Fv, Bv = Fr.copy(), Br.copy()
    for t, lab in enumerate(TOP):
        k = CAL_LABELS.index(lab)
        ok = np.isfinite(pred[t])
        Fv[k, ok] = pred[t, ok]
        Bv[:, k, ok] = predb[:, t, ok]
    variants["trend_14.3-15.1"] = (Fv, Bv)
    # (e) top two bins dropped (painted with the 14.9-15.1 stack, i.e. nearest populated bin)
    Fv, Bv = Fr.copy(), Br.copy()
    i9 = CAL_LABELS.index("14.9-15.1")
    for lab in TOP:
        Fv[CAL_LABELS.index(lab)] = Fr[i9]
        Bv[:, CAL_LABELS.index(lab)] = Br[:, i9]
    variants["nearest_14.9-15.1"] = (Fv, Bv)
    # (f) pooled >= 14.9 as a node at its mean mass; smooth painting through ..., 14.7-14.9, pooled
    nodes_m9 = np.concatenate([M[:12], [pools["pool_14.9"]["mean_logm200c"]]])
    nodes_v9 = np.vstack([V[:12], pv9[None]])
    nodes_b9 = np.concatenate([B[:12], pb9[None]], axis=0)
    Fv, Bv = Fr.copy(), Br.copy()
    for lab in POOLS["pool_14.9"]:
        m = bins[lab]["mean_logm200c"]
        Fv[CAL_LABELS.index(lab)] = interp_lnF(nodes_m9, nodes_v9, m)[0]
        Bv[:, CAL_LABELS.index(lab)] = interp_lnF(nodes_m9, np.moveaxis(nodes_b9, 1, 0), m)[0]
    variants["pool_14.9_smooth"] = (Fv, Bv)
    for name, (Fv, Bv) in variants.items():
        res["impacts"].setdefault(name, {})[field] = impacts(Fv, Fr, Bv, Br, field)

    # ---- continuous painting of each treatment (every calibrated halo at its own mass through the node set)
    cal_bins = [bins[c] for c in CAL_LABELS]
    trend_nodes_v = V[:13].copy()
    trend_nodes_b = B[:13].copy()
    tv, tb = V[13:15].copy(), B[13:15].copy()
    ok_ = np.isfinite(pred)
    tv[ok_] = pred[ok_]
    tb = np.where(np.moveaxis(ok_[:, None, :].repeat(B.shape[1], 1), 0, 0), np.moveaxis(predb, 0, 1), tb)
    node_sets = {
        "reference": (M, V, B),
        "pool_15.1_node": (nodes_m, nodes_v, nodes_b),
        "pool_14.9_node": (nodes_m9, nodes_v9, nodes_b9),
        "trend_14.3-15.1_nodes": (M, np.vstack([trend_nodes_v, tv]), np.concatenate([trend_nodes_b, tb], axis=0)),
        "drop_top_extrapolate": (M[:13], V[:13], B[:13]),
        "drop_top_flat": (np.concatenate([M[:13], [15.6]]), np.vstack([V[:13], V[12][None]]),
                          np.concatenate([B[:13], B[12][None]], axis=0)),
    }
    painted = {k: continuous_paint(*v, cal_bins)[:2] for k, v in node_sets.items()}
    P0, P0b = painted["reference"]
    for name, (Pv, Pb) in painted.items():
        res.setdefault("continuous_painting", {}).setdefault(name, {})[field] = {
            "vs_truth": impacts(Pv, Fr, Pb, Br, field),
            "vs_reference_painted_same_way": impacts(Pv, P0, Pb, P0b, field)}
    # share of each observable carried by each calibrated bin: dO/O for +1% in one bin / 1%
    shares = {}
    for k, lab in enumerate(CAL_LABELS):
        d = np.zeros((10, 35))
        d[k] = np.log(1.01)
        imp = sc.observable_impact(d, field)
        shares[lab] = {o: sig(imp[o]["dO_over_O"] / 0.01 / imp[o]["k"]) for o in OBS_KEEP[field]}
    res["bin_shares"][field] = shares
    # statistical error of the observables from the finite reference sample (global bootstrap replicates of the
    # reference stacks vs the reference): all calibrated bins resampled, or only some of them
    groups = {"all_calibrated": CAL_LABELS, "top_two_only": TOP, "14.9-15.5_only": ("14.9-15.1",) + TOP,
              "13.5-14.9_only": CAL_LABELS[:7], "census_14.5-15.5_only": CAL_LABELS[5:],
              "sampled_13.5-14.5_only": CAL_LABELS[:5]}
    Br_core = np.moveaxis(np.array([wmean_boot(F, bins[c]["rows"], design="core")[1] for c in CAL_LABELS]), 0, 1)
    out = {}
    for design, BB in (("census", Br), ("core", Br_core)):
        out[design] = {}
        for gname, glabs in groups.items():
            sel = [CAL_LABELS.index(g) for g in glabs]
            vals = {k: [] for k in OBS_KEEP[field]}
            for bb in range(BB.shape[0]):
                Fb = Fr.copy()
                Fb[sel] = BB[bb][sel]
                imp = sc.impact_from_stacks(Fb, Fr, field)
                for k in OBS_KEEP[field]:
                    vals[k].append(imp[k]["dO_over_O"])
            out[design][gname] = {k: sig(100 * np.std(v, ddof=1)) for k, v in vals.items()}
    # delete-one jackknife within each census bin (bins independent: variances add)
    jk = {}
    for gname, glabs in (("top_two_only", TOP), ("census_14.5-15.5_only", CAL_LABELS[5:])):
        var = {k: 0.0 for k in OBS_KEEP[field]}
        for lab in glabs:
            rows = bins[lab]["rows"]
            k_ = CAL_LABELS.index(lab)
            reps = {k: [] for k in OBS_KEEP[field]}
            Fall = np.asarray(F[rows], float)
            wv = TAB["weight"][rows]
            okm = np.isfinite(Fall)
            num_all = (wv[:, None] * np.where(okm, Fall, 0)).sum(axis=0)
            den_all = (wv[:, None] * okm).sum(axis=0)
            for i in range(len(rows)):
                Fb = Fr.copy()
                Fb[k_] = (num_all - wv[i] * np.where(okm[i], Fall[i], 0)) / (den_all - wv[i] * okm[i])
                imp = sc.impact_from_stacks(Fb, Fr, field)
                for k in OBS_KEEP[field]:
                    reps[k].append(imp[k]["dO_over_O"])
            n = len(rows)
            for k in OBS_KEEP[field]:
                x = np.asarray(reps[k])
                var[k] += (n - 1) / n * np.sum((x - x.mean())**2)
        jk[gname] = {k: sig(100 * np.sqrt(v)) for k, v in var.items()}
    out["jackknife"] = jk
    res.setdefault("reference_sample_error_pct", {})[field] = out

res["notes"]["individual"] = ("rel_err: median over the band's resolved shells of bootstrap SD / stack (paired global "
                              "bootstrap); loo_max_abs: median over shells of the largest |change| when one halo is "
                              "left out; jackknife_se: delete-one jackknife standard error (fractional).")
res["notes"]["trend"] = ("measured top-bin stack / trend prediction (ln F linear in log M200c, WLS over 14.3-15.1 with "
                         "bootstrap-variance weights; >= 2 resolved nodes per shell); err from refitting each paired "
                         "replicate.")
res["notes"]["impacts"] = ("per cent, first order, relative to the reference (each top bin painted with its own stack); "
                           "err = paired global-bootstrap SD (census design; the reference's own top-bin noise is part "
                           "of it).  *_step: the pooled stack painted on every pooled bin; *_smooth: the pooled stack "
                           "as a node at the pooled mean log M200c, ln F linear in log M through the nodes, evaluated "
                           "at each bin's mean mass.")
res["notes"]["continuous_painting"] = (
    "each treatment as a node set; every halo of the 10 calibrated bins painted at its own log M200c (ln F linear "
    "between bracketing nodes, linear extrapolation beyond the end nodes; mbcommon.continuous_paint).  reference: all "
    "15 0.2-dex nodes; pool_15.1_node / pool_14.9_node: the pooled stack replaces the pooled bins' nodes; "
    "trend_14.3-15.1_nodes: the top two nodes replaced by the trend prediction; drop_top_extrapolate: nodes up to "
    "14.9-15.1, extrapolated above; drop_top_flat: the 14.9-15.1 stack used for every halo above its mean mass.  "
    "vs_truth: painted bin means vs the true bin means; vs_reference_painted_same_way: vs the reference node set.")
res["notes"]["reference_sample_error_pct"] = (
    "SD (per cent) of the first-order impact when the reference stacks of the listed calibrated bins are replaced by "
    "bootstrap replicates (others fixed).  census: plain within-bin resampling in the census bins 14.5-15.5, "
    "mass_bin x class_main strata in the sampled bins (default of this group); core: mass_bin x class_main strata "
    "everywhere (the core's stack_bin convention; class strata of 1-3 haloes in the top bins suppress the variance); "
    "jackknife: delete-one within each census bin, variances summed over bins.")
res["notes"]["bin_shares"] = ("(dO/O) / (k x 1%) for a uniform +1% change of one calibrated bin's stack: the share of "
                              "the observable's calibrated-bin response carried by that bin (sums to calibrated_share).")
dump(res, WORK / "res_topbins.json")
dump(web, WORK / "web_topbins.json")
print("done", len(web))
