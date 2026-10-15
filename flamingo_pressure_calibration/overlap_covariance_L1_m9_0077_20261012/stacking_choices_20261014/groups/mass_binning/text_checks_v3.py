"""Derived numbers for the third text revision of SECTION.md (2026-10-14, after verifier round 2).

Reads only results.json in this folder and writes work/text_checks_v3.json.  Like text_checks.py it adds no
new physics: it re-expresses saved results (ranges with their field and bin, sums of per-bin contributions,
pulls / significances, baseline conversions) so that every number quoted in the corrected text can be traced.
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
A = R["analyses"]
V = R["variants"]
out = {}
pct = lambda x: round(100 * x, 2)  # noqa: E731

# 1. integrated painting estimate vs metric C (same 0.2-dex baseline) and vs metric B (linear proxies, vs truth)
pa = A["painting_assignment_estimate"]
cp = A["scheme_impacts_continuous_painting"]
pairs = {"y_mean": ("P", "y_mean"), "DM_mean": ("E", "DM_mean"),
         "tSZ_1h_lowell": ("P", "tSZ_yy_total_ell100"), "kSZ_1h_lowell": ("E", "kSZ_1h_ell100")}
base = pa["M200c_0.2|smooth"]["vs_smooth_M200c_0.1_pct"]
est = {}
for S in ("M200c_0.3", "M200c_0.5"):
    a = pa[f"{S}|smooth"]["vs_smooth_M200c_0.1_pct"]
    for k, (f, o) in pairs.items():
        e = 100 * ((1 + a[k] / 100) / (1 + base[k] / 100) - 1)
        c = cp[S][f]["vs_reference_painted_same_way"][o]
        row = {"estimate_vs_smooth_0.2": round(e, 2), "metricC": [round(c["pct"], 2), round(c["err_pct"], 2)],
               "pull_vs_C": round((e - c["pct"]) / c["err_pct"], 1)}
        if k in ("y_mean", "DM_mean"):  # linear proxies can be compared with the truth directly
            t = pa[f"{S}|smooth"]["vs_truth_pct"][k]
            b = cp[S][f]["vs_truth"][o]
            row.update({"estimate_vs_truth": round(t, 2), "metricB": [round(b["pct"], 2), round(b["err_pct"], 2)],
                        "pull_vs_B": round((t - b["pct"]) / b["err_pct"], 1)})
        est[f"{S}|{k}"] = row
out["painting_estimate"] = est

# 2. residual mass trend: slopes (% per 0.1 dex) by field / band / bin, with errors
mt = A["mass_trend"]
CAL = ["13.5-13.7", "13.7-13.9", "13.9-14.1", "14.1-14.3", "14.3-14.5", "14.5-14.7", "14.7-14.9", "14.9-15.1"]


def slope_rows(field, bins, bands):
    rows = []
    for e in mt[field]:
        s = e.get("slope_central_per_0p1dex") or {}
        if e["bin"] in bins:
            rows += [(pct(s[b]["value"]), pct(s[b]["err"]), e["bin"], b) for b in bands if s.get(b)]
    return rows


tr = {}
for f in ("P", "E"):
    inner = slope_rows(f, CAL, ["0.1-0.25", "0.25-0.5"])
    tr[f"{f}|0.1-0.5|13.5-14.9"] = [min(r for r in inner if r[2] != "14.9-15.1"),
                                    max(r for r in inner if r[2] != "14.9-15.1")]
    tr[f"{f}|0.1-0.5|14.9-15.1"] = slope_rows(f, ["14.9-15.1"], ["0.1-0.25", "0.25-0.5"])
    tr[f"{f}|0.5-1|by_bin"] = slope_rows(f, CAL, ["0.5-1"])
    outer = slope_rows(f, CAL, ["2-3", "3-4"])
    tr[f"{f}|2-4|13.5-15.1"] = [min(outer), max(outer)]
    tr[f"{f}|2-4|15.1-15.3"] = slope_rows(f, ["15.1-15.3"], ["2-3", "3-4"])
    tr[f"{f}|13.7-13.9_row"] = slope_rows(f, ["13.7-13.9"], ["0.25-0.5", "0.5-1", "1-1.5", "2-3", "3-4"])
tr["max_inner_slope_13.5-15.1"] = max(slope_rows("P", CAL, ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1"]) +
                                      slope_rows("E", CAL, ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1"]))
curv = {"13.5-14.3": [], "14.3-15.1": []}
for f in ("P", "E"):
    for e in mt[f]:
        c = e.get("curvature_per_dex2") or {}
        for b in ("0.25-0.5", "0.5-1"):
            if c.get(b) is None:
                continue
            if e["bin"] in CAL[:4]:
                curv["13.5-14.3"].append((round(c[b], 2), f, e["bin"], b))
            elif e["bin"] in CAL[4:]:
                curv["14.3-15.1"].append((round(c[b], 2), f, e["bin"], b))
tr["curvature_0.25-1"] = {k: {"min": min(v), "max": max(v), "n_pos": sum(x[0] > 0 for x in v), "n": len(v)}
                          for k, v in curv.items()}
out["mass_trend"] = tr

# 3. 0.5-dex scheme: significance and per-bin sums, linear vs PCHIP
it = A["interpolant_test"]
pb = {}
for interp in ("linear", "pchip"):
    for cmp in ("vs_truth", "vs_0.2dex_same_interp"):
        for f, o in (("P", "tSZ_yy_total_ell100"), ("P", "tSZ_yy_total_ell1000"), ("P", "tSZ_yy_total_ell3000"),
                     ("P", "y_mean"), ("E", "kSZ_1h_ell1000"), ("E", "kSZ_1h_ell3000"), ("E", "DM_mean"),
                     ("E", "DM_var")):
            x = it[interp]["M200c_0.5"][f][cmp][o]
            p = x["per_bin_pct"]
            pb[f"{interp}|{cmp}|{o}"] = {
                "pct": round(x["pct"], 2), "err": round(x["err_pct"], 2), "sigma": round(abs(x["pct"]) / x["err_pct"], 1),
                "13.5-13.7/13.7-13.9/13.9-14.1": [round(p[b], 2) for b in ("13.5-13.7", "13.7-13.9", "13.9-14.1")],
                "sum_13.5-14.1": round(sum(p[b] for b in ("13.5-13.7", "13.7-13.9", "13.9-14.1")), 2),
                "15.3-15.5": round(p["15.3-15.5"], 2),
                "14.9-15.3": round(p["14.9-15.1"] + p["15.1-15.3"], 2)}
out["M200c_0.5"] = pb

# 4. metric A / metric C ratio for 0.3 dex (kSZ ell=3000, DM variance)
r03 = {}
for o in ("kSZ_1h_ell3000", "DM_var"):
    a = V["mass_binning:M200c_0.3"]["impacts_at_reference_nodes"]["E"]["impact"][o]["pct"]
    c = V["mass_binning:M200c_0.3"]["impacts"]["E"]["impact"][o]["pct"]
    r03[o] = {"A": round(a, 2), "C": round(c, 2), "A_over_C": round(a / c, 2)}
out["M200c_0.3_A_over_C"] = r03

# 5. normalisation: significance of the 0.2-dex tSZ bias, measured / pure Jensen, Pbar_arith_over_geo, P a+alpha
nb = A["normalisation_own_vs_binmean"]
ni = nb["impacts"]["M200c_0.2"]["P"]
out["binmean_norm_0.2_tSZ_sigma"] = {o: round(ni[o]["pct"] / ni[o]["err_pct"], 1)
                                     for o in ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000")}
out["Pbar_arith_over_geo"] = {S: [min(e["Pbar_arith_over_geo"] for e in nb["bins"][S]),
                                  max(e["Pbar_arith_over_geo"] for e in nb["bins"][S])] for S in nb["bins"]}
jd = nb["jensen_decomposition"]
b = jd["M200c_0.5"]["13.5-14.0"]["fields"]
out["measured_over_pure_13.5-14.0"] = {f"{f}|{band}": round(b[f][band]["measured_pct"] / b[f][band]["pure_jensen_pct"], 2)
                                       for f, band in (("P", "0.25-0.5"), ("P", "0.5-1"), ("E", "0.25-0.5"),
                                                       ("E", "0.5-1"), ("T_e", "0.5-1"))}
aa = {f: [round(jd["M200c_0.5"][bn]["fields"][f][band]["a_plus_alpha"], 2)
          for bn in ("13.5-14.0", "14.0-14.5", "14.5-15.0", "15.0-15.5") for band in ("0.25-0.5", "0.5-1")
          if jd["M200c_0.5"][bn]["fields"][f][band]]
      for f in ("P", "E", "T_e", "K")}
out["a_plus_alpha_0.5dex_0.25-1"] = {f: [min(v), max(v)] for f, v in aa.items()}
out["K_beta_check_13.5-14.0"] = {band: {"beta_K": round(b["K"][band]["beta"], 2),
                                        "beta_T - 2/3 beta_E": round(b["T_e"][band]["beta"] - 2 / 3 * b["E"][band]["beta"], 2)}
                                 for band in ("0.25-0.5", "0.5-1")}

# 6. M500c_0.2 stacks at the same mean M200c (bins with mean log M200c >= 14.3), with errors; concentration offsets
svr = A["scheme_vs_reference_same_mass"]["P"]["M500c_0.2"]
m5 = {}
for e in svr:
    if 14.3 <= e["mean_logm200c"] < 15.3:
        m5[e["bin"]] = {"N": e["N"], "mean_logm200c": e["mean_logm200c"],
                        "core": {b: [pct(e["bands"][b]["ratio"] - 1), pct(e["bands"][b]["err"])]
                                 for b in ("<0.1", "0.1-0.25") if e["bands"][b]},
                        "1-1.5": [pct(e["bands"]["1-1.5"]["ratio"] - 1), pct(e["bands"]["1-1.5"]["err"])]}
out["M500c_0.2_same_mass_bands"] = m5
comp = A["m500_vs_m200"]["composition"]["M500c_0.2"]
cal = [(e["bin"], e["N"], e["mean_logm200c"], round(e["dlog_c200c_vs_M200c_bins_at_same_mean_m200"], 4))
       for e in comp if 13.5 <= e["mean_logm200c"] < 15.5]
out["M500c_0.2_dlogc_calibrated"] = cal

# 7. sample size: scatter precision and covariance stability per field; finite-population note
sp = A["sample_size"]["scatter_precision"]
spr = {}
for f in ("P", "E"):
    for N in ("10", "100"):
        xs = [sp[bn][N][f]["sd_ln_int4R"] for bn in sp]
        spr[f"{f}|N{N}"] = {k: [pct(min(x[k] for x in xs)), pct(max(x[k] for x in xs))]
                            for k in ("rel_sd", "rel_sd_inf", "median_rel_bias")}
out["scatter_precision_pct"] = spr
cv = A["sample_size"]["covariance"]
cvr = {}
for f in ("P", "E"):
    for N in ("30", "100"):
        xs = [cv[bn][f]["byN"][N] for bn in cv]
        cvr[f"{f}|N{N}"] = {k: [round(min(x[k] for x in xs), 3), round(max(x[k] for x in xs), 3)]
                            for k in ("mean_sd_offdiag_inf", "second_eigvec_cos_median",
                                      "precision_trace_ratio_mean", "precision_trace_ratio_sd")}
out["covariance"] = cvr
osc = A["sample_size"]["observable_scatter"]
Nbin = {e["bin"]: e["N"] for e in A["wide_vs_fine"]["P"]["M200c_0.2"]}
out["observable_scatter_population"] = {N: [Nbin[bn] for bn in osc[N]["bins_subsampled"]] for N in osc}
out["sd_times_sqrtN"] = A["observable_scatter_times_sqrtN_pct"]
nn = A["N_needed_subsampling"]
out["P_2-3_N3pct_vs_available"] = {bn: (nn[bn]["P"]["2-3"]["N_3pct"], Nbin[bn]) for bn in nn}
out["sampled_bins_N_13.5-14.5"] = [Nbin[bn] for bn in CAL[:5]]

# 8. top bins: shares, option noise and significance
sh = A["top_bins"]["bin_shares"]
shares = {}
for f, obs in (("P", ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000")),
               ("E", ("kSZ_1h_ell3000", "DM_mean", "DM_var"))):
    for o in obs:
        tot = sum(sh[f][bn][o] for bn in sh[f])
        top = sh[f]["15.1-15.3"][o] + sh[f]["15.3-15.5"][o]
        samp = sum(sh[f][bn][o] for bn in CAL[:5])
        shares[o] = {"top_two_pct": pct(top), "top_two_of_calibrated_pct": pct(top / tot), "sampled_pct": pct(samp)}
out["bin_shares"] = shares
tcp = A["top_bins"]["continuous_painting"]
out["top_bin_option_err_tSZ"] = {k: [round(v["P"]["vs_reference_painted_same_way"][o]["err_pct"], 2)
                                     for o in ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000")]
                                 for k, v in tcp.items() if k != "reference"}

(HERE / "work" / "text_checks_v3.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
