"""Derived numbers quoted in the corrected SECTION.md (text revision of 2026-10-14).

Reads only results.json in this folder and writes work/text_checks.json.  Nothing here is new physics:
it re-expresses saved results (ranges, sums of per-bin contributions, pulls, baseline conversions).
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
A = R["analyses"]
V = R["variants"]
out = {}

# 1. integrated painting estimate converted from 'vs smooth 0.1 dex' to 'vs smooth 0.2 dex' (metric C baseline)
pa = A["painting_assignment_estimate"]
cp = A["scheme_impacts_continuous_painting"]
pairs = {"y_mean": ("P", "y_mean"), "DM_mean": ("E", "DM_mean"),
         "tSZ_1h_lowell": ("P", "tSZ_yy_total_ell100"), "kSZ_1h_lowell": ("E", "kSZ_1h_ell100")}
base = pa["M200c_0.2|smooth"]["vs_smooth_M200c_0.1_pct"]
conv = {}
for S in ("M200c_0.3", "M200c_0.5"):
    a = pa[f"{S}|smooth"]["vs_smooth_M200c_0.1_pct"]
    for k, (f, o) in pairs.items():
        est = 100 * ((1 + a[k] / 100) / (1 + base[k] / 100) - 1)
        c = cp[S][f]["vs_reference_painted_same_way"][o]
        conv[f"{S}|{k}"] = {"estimate_vs_smooth_0.2_pct": round(est, 3), "metricC_obs": o,
                            "metricC_pct": c["pct"], "metricC_err": c["err_pct"],
                            "pull": round((est - c["pct"]) / c["err_pct"], 2)}
out["painting_estimate_vs_metricC"] = conv

# 2. mass-trend extremes (central slopes, % per 0.1 dex), calibrated bins 13.5-15.1 and 15.1-15.3 separately
mt = A["mass_trend"]


def slopes(field, bins, bands):
    rows = []
    for e in mt[field]:
        s = e.get("slope_central_per_0p1dex")
        if not s or e["bin"] not in bins:
            continue
        for b in bands:
            if s.get(b):
                rows.append((round(100 * s[b]["value"], 2), round(100 * s[b]["err"], 2), e["bin"], b))
    return rows


cal = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1"]
inner = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1"]
outer = ["2-3", "3-4"]
tr = {}
for f in ("P", "E"):
    ri = slopes(f, cal + ["15.1-15.3"], inner)
    tr[f"{f}|inner_max_13.5-15.3"] = max(ri)
    tr[f"{f}|inner_max_13.5-15.1"] = max(slopes(f, cal, inner))
    ro = slopes(f, cal, outer)
    tr[f"{f}|outer_range_13.5-15.1"] = [min(ro), max(ro)]
    tr[f"{f}|outer_15.1-15.3"] = slopes(f, ["15.1-15.3"], outer)
tr["E|0.1-0.5_range_13.7-15.1"] = [min(slopes("E", cal[1:], ["0.1-0.25", "0.25-0.5"])),
                                   max(slopes("E", cal[1:], ["0.1-0.25", "0.25-0.5"]))]
tr["P|0.5-1_by_bin"] = slopes("P", cal, ["0.5-1"])
curv = {}
for f in ("P", "E"):
    for e in mt[f]:
        c = e.get("curvature_per_dex2")
        if c and e["bin"] in ("13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3"):
            curv[f"{f}|{e['bin']}"] = {b: c[b] for b in ("0.25-0.5", "0.5-1") if c.get(b) is not None}
tr["curvature_0.25-1_13.5-14.3"] = curv
tr["all_negative_curvature_0.25-1_13.5-14.3"] = all(v < 0 for d in curv.values() for v in d.values())
out["mass_trend"] = tr

# 3. subsample bias at 2-4 R200c: mean vs median, standard error of the mean = sd / sqrt(R)
pr = A["sample_size"]["precision"]
Rn = A["sample_size"]["settings"]["R"]
sb = {}
for band in ("2-3", "3-4"):
    for N in ("10", "25", "50"):
        mean = [pr[b][N]["P"][band]["mean_bias"] for b in pr]
        se = [pr[b][N]["P"][band]["sd"] / math.sqrt(Rn) for b in pr]
        med = [pr[b][N]["P"][band]["median_bias"] for b in pr]
        pulls = [m / s for m, s in zip(mean, se)]
        sb[f"{band}|N{N}"] = {"mean_bias_pct": [round(100 * min(mean), 1), round(100 * max(mean), 1)],
                              "se_pct": [round(100 * min(se), 1), round(100 * max(se), 1)],
                              "median_bias_pct": [round(100 * min(med), 1), round(100 * max(med), 1)],
                              "max_abs_mean_over_se": round(max(abs(p) for p in pulls), 2),
                              "worst_bin": list(pr)[int(np.argmax(np.abs(pulls)))],
                              "sd_range": [round(min(pr[b][N]["P"][band]["sd"] for b in pr), 3),
                                           round(max(pr[b][N]["P"][band]["sd"] for b in pr), 3)]}
out["subsample_bias_P"] = sb

# 4. scatter precision of sd ln(Y_4R) (field P) and covariance stability, per field
sp = A["sample_size"]["scatter_precision"]
sc_ = {}
for f in ("P", "E"):
    for N in ("10", "100"):
        x = [sp[b][N][f]["sd_ln_int4R"] for b in sp if N in sp[b]]
        sc_[f"{f}|N{N}"] = {k: [round(min(v[k] for v in x), 3), round(max(v[k] for v in x), 3)]
                            for k in ("rel_sd", "rel_sd_inf", "median_rel_bias")}
out["scatter_precision"] = sc_
cv = A["sample_size"]["covariance"]
cs = {"p_shells": [min(cv[b]["p_shells"] for b in cv), max(cv[b]["p_shells"] for b in cv)]}
for f in ("P", "E"):
    for N in ("30", "100"):
        xs = [cv[b][f]["byN"][N] for b in cv]
        cs[f"{f}|N{N}"] = {k: [round(min(x[k] for x in xs), 4), round(max(x[k] for x in xs), 4)]
                           for k in ("mean_sd_offdiag_inf", "second_eigvec_cos_median",
                                     "precision_trace_ratio_mean", "precision_trace_ratio_sd")}
out["covariance"] = cs

# 5. PCHIP / linear per-bin contributions for 0.5 dex
it = A["interpolant_test"]
pb = {}
for interp in ("linear", "pchip"):
    for cmp in ("vs_truth", "vs_0.2dex_same_interp"):
        for f, o in (("P", "tSZ_yy_total_ell1000"), ("E", "kSZ_1h_ell3000"), ("E", "DM_mean"), ("E", "DM_var")):
            x = it[interp]["M200c_0.5"][f][cmp][o]
            pb[f"{interp}|{cmp}|{o}"] = {"pct": x["pct"], "err": x["err_pct"],
                                         "per_bin": {k: round(v, 2) for k, v in x["per_bin_pct"].items()}}
out["M200c_0.5_per_bin"] = pb
x = it["pchip"]["M200c_0.5"]["P"]["vs_truth"]["tSZ_yy_total_ell1000"]["per_bin_pct"]
out["M200c_0.5_pchip_tSZ1000_14.9-15.3_sum"] = round(x["14.9-15.1"] + x["15.1-15.3"], 2)
sig = {}
for interp in ("linear", "pchip"):
    for S in ("M200c_0.2", "M200c_0.3", "M200c_0.5"):
        rows = []
        for f in ("P", "E"):
            for o, y in it[interp][S][f]["vs_truth"].items():
                rows.append((o, y["pct"], y["err_pct"], round(y["pct"] / y["err_pct"], 2)))
        sig[f"{interp}|{S}"] = {"max_abs_central": max(abs(r[1]) for r in rows),
                                "max_err": max(r[2] for r in rows),
                                "beyond_2sigma": [r for r in rows if abs(r[3]) > 2]}
out["interpolant_vs_truth_significance"] = sig

# 6. M500c_0.2 same-mass band ratios in the 14.1-15.1 M500c bins, and per-bin metric-C tSZ contributions
svr = A["scheme_vs_reference_same_mass"]["P"]["M500c_0.2"]
m5 = {}
for e in svr:
    if e["bin"] in ("14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1"):
        core = [round(100 * (e["bands"][b]["ratio"] - 1), 1) for b in ("<0.1", "0.1-0.25") if e["bands"][b]]
        mid = round(100 * (e["bands"]["1-1.5"]["ratio"] - 1), 1)
        m5[e["bin"]] = {"N": e["N"], "mean_logm200c": e["mean_logm200c"], "core_lt_0.25": core, "1-1.5": mid}
out["M500c_0.2_bands_14.1-15.1"] = m5
comp = A["m500_vs_m200"]["composition"]["M500c_0.2"]
out["M500c_0.2_dlogc"] = {e["bin"]: (e["N"], e["dlog_c200c_vs_M200c_bins_at_same_mean_m200"]) for e in comp}
pbin = {}
for o in ("tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000"):
    x = V["mass_binning:M500c_0.2"]["impacts"]["P"]["impact"][o]["per_bin_pct"]
    pbin[o] = {"14.3-14.9": round(x["14.3-14.5"] + x["14.5-14.7"] + x["14.7-14.9"], 2),
               "15.1-15.3": round(x["15.1-15.3"], 2)}
out["M500c_0.2_metricC_per_bin"] = pbin

# 7. top-bin treatments: significance of the paired changes
tb = A["top_bins"]
tsig = {}
for k, v in tb["continuous_painting"].items():
    if k == "reference":
        continue
    for o, y in v["P"]["vs_reference_painted_same_way"].items():
        if o.startswith("tSZ"):
            tsig[f"{k}|{o}"] = (round(y["pct"], 2), round(y["err_pct"], 2), round(y["pct"] / y["err_pct"], 2))
for k in ("nearest_14.9-15.1",):
    for o, y in tb["impacts"][k]["P"].items():
        if o.startswith("tSZ"):
            tsig[f"{k}_step|{o}"] = (round(y["pct"], 2), round(y["err_pct"], 2), round(y["pct"] / y["err_pct"], 2))
out["top_bin_significance"] = tsig

# 8. normalisation: second-order formula and pure Jensen against all measured bands
jd = A["normalisation_own_vs_binmean"]["jensen_decomposition"]
js = {}
for S in jd:
    for key in ("pure_jensen_pct", "second_order_pct", "trend_model_pct"):
        p = [(x["measured_pct"] - x[key]) / x["measured_err_pct"]
             for bv in jd[S].values() for fv in bv["fields"].values() for x in fv.values()
             if x and x.get(key) is not None and x.get("measured_err_pct")]
        p = np.array(p)
        js[f"{S}|{key}"] = {"n": int(p.size), "rms_pull": round(float(np.sqrt(np.mean(p**2))), 2),
                            "max_abs_pull": round(float(np.max(np.abs(p))), 2)}
b = jd["M200c_0.5"]["13.5-14.0"]["fields"]
js["measured_over_pure_13.5-14.0"] = {f"{f}|{band}": round(b[f][band]["measured_pct"] / b[f][band]["pure_jensen_pct"], 2)
                                     for f, band in (("P", "0.25-0.5"), ("P", "0.5-1"), ("E", "0.25-0.5"),
                                                     ("E", "0.5-1"), ("T_e", "0.5-1"))}
out["jensen"] = js

# 9. haloes per reference bin (sampled and census)
out["N_per_bin"] = {e["bin"]: e["N"] for e in A["wide_vs_fine"]["P"]["M200c_0.2"]}

(HERE / "work" / "text_checks.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
