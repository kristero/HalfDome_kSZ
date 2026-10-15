"""Mass-binning group, part 4: sample size per bin.

For every M200c_0.2 bin and N in NS (N < bin size), R random subsamples without replacement of the bin's haloes are
stacked with their inclusion weights (same estimator, statistic, grid as the reference).  Per radial band the
statistic is the median over the full bin's strict-resolved shells of (subsample stack / full-bin stack).
* bias (mean and median over realisations), scatter (SD), finite-population-corrected scatter
  SD_inf = SD / sqrt((N_bin - N) / (N_bin - 1)) (the error of an N-halo stack drawn from an infinite population),
  per-halo amplitude sigma_1 = SD_inf sqrt(N), and N needed for 1 / 3 / 5% precision: (sigma_1 / p)^2.
* resolved range under the strict rule as a function of N (the rule asks EVERY halo of the stack to be resolved).
* precision of the per-halo lognormal scatter (SD of ln F per band; SD of ln of the normalised integrals to 4 R200c,
  the painting scatter inputs) vs N.
* covariance stability (Lee22 grid, strict-resolved shells of the full bin, unweighted ln F like Lee22): element
  noise of the correlation matrix, leading-eigenvector stability, precision-matrix bias / noise with Hartlap.
* observable scatter: R realisations with N haloes in every calibrated bin -> SD of the first-order impact.
* web variants N25 / N50 / N100: realisation 0 of every bin with more than N haloes.
Outputs: work/res_samplesize.json, work/web_samplesize.json.
"""
from mbcommon import (BAND_IDX, BAND_LABELS, CAL_LABELS, EDGES_FIT, FIELDS_WEB, GLOBAL_SEED, N_BOOT, OBS_KEEP, TAB,
                      W, WORK, dump, load, record_from_stack, ref_key, sc, sig, web_record)
import numpy as np

R = 400
NS = (10, 25, 50, 100, 200, 300, 500)
DETAIL_BINS = ("13.5-13.7", "13.9-14.1", "14.1-14.3", "14.5-14.7", "14.7-14.9")
WEB_NS = (25, 50, 100)
FIELDS = FIELDS_WEB

bins = sc.mass_bins("M200c_0.2")
DATA = {f: load(ref_key(f)) for f in FIELDS}
LEE = {f: load(ref_key(f, "lee22")) for f in ("P", "E")}
INT4 = {"P": sc.integral(DATA["P"], "fit", 4.0), "E": sc.integral(DATA["E"], "fit", 4.0)}


def draws(b, N):
    rng = np.random.default_rng([GLOBAL_SEED, int(b["index"]), int(N)])
    n = b["N"]
    return np.array([rng.choice(n, N, replace=False) for _ in range(R)])


def wmat(idx, w):
    M = np.zeros((idx.shape[0], len(w)))
    np.put_along_axis(M, idx, w[idx], axis=1)
    return M


def band_stat(ratio, resolved):
    """ratio [R, S] -> [R, n_bands] (median over resolved shells in band; NaN if none)."""
    out = np.full((ratio.shape[0], len(BAND_IDX)), np.nan)
    for k, idx in enumerate(BAND_IDX):
        sel = idx[resolved[idx]]
        if len(sel):
            out[:, k] = np.nanmedian(ratio[:, sel], axis=1)
    return out


res = {"settings": {"R": R, "NS": NS, "detail_bins": DETAIL_BINS}, "precision": {}, "resolved_range": {},
       "scatter_precision": {}, "covariance": {}, "observable_scatter": {}, "analytic_sigma1": {}, "notes": {}}
cal_stacks = {N: {f: [] for f in ("P", "E")} for N in NS}     # realisations of the calibrated bins
web = []
web_rows = {}

for b in bins:
    rows, w, nb = b["rows"], b["weights"], b["N"]
    det = sc.resolution_detail(rows, "fit")
    strict_full = det["strict"]
    resolved_h = det["resolved"]                       # [n, S]
    full = {f: sc.stack(DATA[f][rows], w, "mean", n_boot=0) for f in FIELDS}
    # analytic per-halo amplitude: reliability-weighted per-halo SD / mean (per shell), band median
    res["analytic_sigma1"][b["label"]] = {
        f: {lab: sig(np.median((full[f]["sd"] / full[f]["value"])[idx[strict_full[idx]]]))
            if strict_full[idx].any() else None for lab, idx in zip(BAND_LABELS, BAND_IDX)} for f in FIELDS}
    res["analytic_sigma1"][b["label"]]["N"] = nb
    res["analytic_sigma1"][b["label"]]["N_eff"] = sig(b["N_eff"])
    res["analytic_sigma1"][b["label"]]["population"] = sig(b["sum_weight"])
    for N in NS:
        if N >= nb:
            if b["label"] in CAL_LABELS:            # bin smaller than N: every realisation uses the full bin
                for f in ("P", "E"):
                    cal_stacks[N][f].append(np.broadcast_to(full[f]["value"], (R, 35)))
            continue
        idx = draws(b, N)
        Wm = wmat(idx, w)
        fpc = np.sqrt((nb - N) / (nb - 1))
        entry_N = {}
        for f in FIELDS:
            v = np.asarray(DATA[f][rows], float)
            ok = np.isfinite(v)
            with np.errstate(divide="ignore", invalid="ignore"):
                st = (Wm @ np.where(ok, v, 0.0)) / (Wm @ ok)
                ratio = st / full[f]["value"][None, :]
            if b["label"] in CAL_LABELS and f in ("P", "E"):
                cal_stacks[N][f].append(st)
            if b["label"] not in DETAIL_BINS:
                continue
            bs = band_stat(ratio, strict_full)
            e = {}
            for k, lab in enumerate(BAND_LABELS):
                col = bs[:, k]
                if not np.isfinite(col).any():
                    e[lab] = None
                    continue
                sd = np.nanstd(col, ddof=1)
                # per-shell scatter (FPC-corrected) -> per-shell N for 1/3/5%, median over the band's shells
                sel = BAND_IDX[k][strict_full[BAND_IDX[k]]]
                sd_sh = np.nanstd(ratio[:, sel], axis=0, ddof=1) / fpc
                e[lab] = {"mean_bias": sig(np.nanmean(col) - 1), "median_bias": sig(np.nanmedian(col) - 1),
                          "sd": sig(sd), "q16": sig(np.nanpercentile(col, 16) - 1),
                          "q84": sig(np.nanpercentile(col, 84) - 1), "sd_inf": sig(sd / fpc),
                          "sigma1": sig(sd / fpc * np.sqrt(N)),
                          "sigma1_shell_median": sig(np.median(sd_sh) * np.sqrt(N))}
            entry_N[f] = e
        if b["label"] in DETAIL_BINS:
            res["precision"].setdefault(b["label"], {})[str(N)] = entry_N
            # strict-rule resolved range of the subsamples
            first = []
            for r in range(R):
                s = np.all(resolved_h[idx[r]], axis=0)
                first.append(EDGES_FIT[:-1][np.argmax(s)] if s.any() else np.nan)
            first = np.array(first)
            res["resolved_range"].setdefault(b["label"], {})[str(N)] = {
                "first_resolved_edge_median": sig(np.nanmedian(first)),
                "first_resolved_edge_q16": sig(np.nanpercentile(first, 16)),
                "first_resolved_edge_q84": sig(np.nanpercentile(first, 84))}
            # precision of the per-halo scatter: SD of ln F (band median) and SD of ln(normalised integral to 4R)
            sp = {}
            for f in ("P", "E"):
                v = np.asarray(DATA[f][rows], float)
                pos = np.isfinite(v) & (v > 0)
                lv = np.where(pos, np.log(np.where(pos, v, 1.0)), 0.0)
                Wp = Wm[:, :, None] * pos[None, :, :]
                tot = Wp.sum(axis=1)
                mu = np.einsum("rns,ns->rs", Wp, lv) / tot
                var = np.einsum("rns,ns->rs", Wp, lv**2) / tot - mu**2
                sdl = np.sqrt(np.clip(var, 0, None))
                sdl_full = full[f]["sd_ln"]
                rel = sdl / sdl_full[None, :]
                rb = band_stat(rel, strict_full)
                li = np.log(INT4[f][rows])
                mu_i = (Wm @ li) / Wm.sum(axis=1)
                sd_i = np.sqrt((Wm @ li**2) / Wm.sum(axis=1) - mu_i**2)
                sd_i_full = np.sqrt(np.average((li - np.average(li, weights=w))**2, weights=w))
                sp[f] = {"sd_ln_shells": {lab: ({"median_rel_bias": sig(np.nanmedian(rb[:, k]) - 1),
                                                 "rel_sd": sig(np.nanstd(rb[:, k], ddof=1))}
                                                if np.isfinite(rb[:, k]).any() else None)
                                          for k, lab in enumerate(BAND_LABELS)},
                         "sd_ln_int4R": {"full": sig(sd_i_full), "median_rel_bias": sig(np.median(sd_i / sd_i_full) - 1),
                                         "rel_sd": sig(np.std(sd_i / sd_i_full, ddof=1)),
                                         "rel_sd_inf": sig(np.std(sd_i / sd_i_full, ddof=1) / fpc)}}
            res["scatter_precision"].setdefault(b["label"], {})[str(N)] = sp
        # web variants: realisation 0
        if N in WEB_NS:
            sub = rows[idx[0]]
            sb = dict(b)
            sb.update(rows=sub, weights=TAB["weight"][sub], strata=b["strata"][idx[0]], N=N,
                      N_eff=float(TAB["weight"][sub].sum()**2 / (TAB["weight"][sub]**2).sum()),
                      mean_log_mass=float(np.average(TAB["logm200c"][sub], weights=TAB["weight"][sub])))
            web_rows.setdefault(N, {})[b["label"]] = idx[0]
            for f in FIELDS:
                rr = sc.stack_bin(ref_key(f), sb, "mean", "strict", n_boot=N_BOOT)
                web.append(record_from_stack(f"mass_binning:N{N}", f, "M200c_0.2", sb, rr))

# ------------------------------------------------------------------------------------------- covariance stability
NC = (10, 25, 30, 50, 100, 200, 300, 500)
for lab in DETAIL_BINS:
    b = sc.find_bin("M200c_0.2", lab)
    rows, nb = b["rows"], b["N"]
    strict = np.asarray(sc.resolution_mask(rows, "lee22", "strict"), bool)
    out_b = {"p_shells": int(strict.sum()),
             "shell_range": [sig(sc.edges("lee22")[:-1][strict].min()), sig(sc.edges("lee22")[1:][strict].max())]}
    for f in ("P", "E"):
        X = np.asarray(LEE[f][rows][:, strict], float)
        good = np.all(np.isfinite(X) & (X > 0), axis=1)
        L = np.log(X[good])
        n_good = len(L)
        p = L.shape[1]
        Cf = np.cov(L, rowvar=False)
        Rf = np.corrcoef(L, rowvar=False)
        ev, evec = np.linalg.eigh(Rf)
        v1, v2 = evec[:, -1], evec[:, -2]
        off = ~np.eye(p, dtype=bool)
        e_f = {"n_haloes_used": n_good, "dropped_nonpositive": int((~good).sum()),
               "mean_abs_offdiag_rho_full": sig(np.mean(np.abs(Rf[off]))),
               "leading_eigen_fraction_full": sig(ev[-1] / ev.sum()),
               "condition_number_full": sig(np.linalg.cond(Cf)), "byN": {}}
        rng = np.random.default_rng([GLOBAL_SEED, 77, int(b["index"]), 1 if f == "P" else 2])
        for N in NC:
            if N >= n_good:
                continue
            dev2, cos1, cos2, trb, cond = [], [], [], [], []
            Rs = []
            for r in range(R):
                ii = rng.choice(n_good, N, replace=False)
                Ls = L[ii]
                Rn = np.corrcoef(Ls, rowvar=False)
                Rs.append(Rn[off])
                dev2.append(np.mean((Rn[off] - Rf[off])**2))
                e2, vv = np.linalg.eigh(Rn)
                cos1.append(abs(vv[:, -1] @ v1))
                cos2.append(abs(vv[:, -2] @ v2))
                if N > p + 2:
                    Cn = np.cov(Ls, rowvar=False)
                    H = (N - p - 2) / (N - 1) * np.linalg.inv(Cn)
                    trb.append(np.trace(H @ Cf) / p)
                    cond.append(np.linalg.cond(Cn))
            Rs = np.array(Rs)
            fpc = np.sqrt((n_good - N) / (n_good - 1))
            th = np.mean((1 - Rf[off]**2) / np.sqrt(N - 1))
            e_f["byN"][str(N)] = {
                "rms_dev_offdiag": sig(np.sqrt(np.mean(dev2))), "mean_sd_offdiag": sig(np.mean(Rs.std(axis=0, ddof=1))),
                "mean_sd_offdiag_inf": sig(np.mean(Rs.std(axis=0, ddof=1)) / fpc),
                "theory_sd_offdiag": sig(th), "leading_eigvec_cos_median": sig(np.median(cos1)),
                "leading_eigvec_cos_q16": sig(np.percentile(cos1, 16)), "second_eigvec_cos_median": sig(np.median(cos2)),
                "hartlap_factor": sig((N - p - 2) / (N - 1)) if N > p + 2 else None,
                "precision_trace_ratio_mean": sig(np.mean(trb)) if trb else None,
                "precision_trace_ratio_sd": sig(np.std(trb, ddof=1)) if len(trb) > 2 else None,
                "condition_number_median": sig(np.median(cond)) if cond else None,
                "invertible": bool(N > p)}
        out_b[f] = e_f
    res["covariance"][lab] = out_b

# ------------------------------------------------------------------------------------- observable scatter vs N
ref_cal = {f: np.array([sc.stack(DATA[f][sc.find_bin("M200c_0.2", c)["rows"]],
                                 sc.find_bin("M200c_0.2", c)["weights"], "mean", n_boot=0)["value"]
                        for c in CAL_LABELS]) for f in ("P", "E")}
for N in NS:
    out_N = {}
    for f in ("P", "E"):
        arr = np.stack([np.asarray(a) for a in cal_stacks[N][f]], axis=1)      # [R, 10, 35]
        vals = {k: [] for k in OBS_KEEP[f]}
        for r in range(R):
            imp = sc.impact_from_stacks(arr[r], ref_cal[f], f)
            for k in OBS_KEEP[f]:
                vals[k].append(imp[k]["dO_over_O"])
        out_N[f] = {k: {"sd_pct": sig(100 * np.std(v, ddof=1)), "mean_pct": sig(100 * np.mean(v)),
                        "median_pct": sig(100 * np.median(v)), "q16_pct": sig(100 * np.percentile(v, 16)),
                        "q84_pct": sig(100 * np.percentile(v, 84))} for k, v in vals.items()}
        if N in WEB_NS:
            out_N[f]["realisation0_pct"] = {k: sig(100 * v[0]) for k, v in vals.items()}
    out_N["bins_subsampled"] = [c for c in CAL_LABELS if sc.find_bin("M200c_0.2", c)["N"] > N]
    res["observable_scatter"][str(N)] = out_N

res["notes"]["precision"] = (
    "per bin / N / field / band: realisations of (subsample stack / full-bin stack), band = median over the full bin's "
    "strict-resolved shells.  mean_bias / median_bias over realisations; sd; sd_inf = sd / FPC; sigma1 = sd_inf sqrt(N) "
    "(per-halo fractional amplitude of the band statistic); sigma1_shell_median = the same for single shells, median "
    "over the band.  N for precision p = (sigma1 / p)^2.")
res["notes"]["observable_scatter"] = (
    "R = 400 realisations, each with an independent random N-halo subsample in every calibrated bin that has more "
    "than N haloes (others full); first-order impact relative to the full-sample stacks (per cent).  Relative to the "
    "full sample, so the finite population (300-895 per bin) makes the large-N values lower limits of the error "
    "relative to an infinite population.")
res["notes"]["covariance"] = (
    "Lee22 grid (27 shells 0.04-1.34 R200c), shells strict-resolved for the full bin, ln F per halo, unweighted (as "
    "Lee22).  rms_dev_offdiag: RMS over off-diagonal elements of (rho_N - rho_full); mean_sd_offdiag: mean over "
    "elements of the SD over realisations; theory: mean (1 - rho^2)/sqrt(N - 1).  precision_trace_ratio: "
    "tr(H_N C_full)/p with the Hartlap-corrected inverse H_N (1 = unbiased; its SD is the precision-matrix noise).")
dump(res, WORK / "res_samplesize.json")
dump(web, WORK / "web_samplesize.json")
np.savez(WORK / "samplesize_web_rows.npz", **{f"N{N}__{lab}": v for N, d in web_rows.items() for lab, v in d.items()})
print("done", len(web))
