#!/usr/bin/env python
"""Group 'selection': check the numbers changed or added in the third SECTION.md correction (2026-10-14).

Read-only: reads results.json (including text_support and text_support_v2) and prints one line per claim with the
value it is based on and PASS/FAIL against the rounding used in SECTION.md.  Writes nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
V, S, TS, T2 = R["variants"], R["summary"]["variants"], R["text_support"], R["text_support_v2"]
LAB = [b["label"] for b in R["bins"]]
CAL, TOP, FALLBACK = LAB[5:14], "15.3-15.5", LAB[1:5]
n_fail = 0


def check(name, value, lo, hi):
    """lo <= value <= hi, elementwise for tuples/lists."""
    global n_fail
    if isinstance(value, (list, tuple)):
        ok = len(value) == len(lo) == len(hi) and all(a <= v <= b for v, a, b in zip(value, lo, hi))
    else:
        ok = lo <= value <= hi
    n_fail += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {value!r}")


def rng(vals):
    vals = [v for v in vals if v is not None]
    return min(vals), max(vals)


def irdev(v, f, ann, labs):
    return rng([100 * V[v]["integral_ratio"][f][l][ann]["dev"] for l in labs])


def bands_present(v, f, band):
    return [l for l in LAB if V[v]["band_ratio"][f][l][band] is not None]


# --- what was measured: band coverage, strict first edges, importance shares
check("<0.1 band only in 14.5-15.5", bands_present("selection:e_fullion", "E", "<0.1") == LAB[10:], True, True)
check("0.1-0.25 band only in 13.9-15.5", bands_present("selection:e_fullion", "E", "0.1-0.25") == LAB[7:], True, True)
imp = T2["importance_share"]
check("DM_mean calibrated share 58.4%", imp["DM_mean"]["calibrated_13.5-15.5"], 0.5835, 0.5845)
check("DM_mean fallback share 41.6%", imp["DM_mean"]["painted_fallback_12.7-13.5"], 0.4155, 0.4165)
check("kSZ1000 calibrated share 82.6%", imp["kSZ_1h_ell1000"]["calibrated_13.5-15.5"], 0.8255, 0.8265)
check("DM_var calibrated share 75.6%", imp["DM_var"]["calibrated_13.5-15.5"], 0.7555, 0.7565)
check("tSZ calibrated shares 97.5-98.6%", rng([imp[f"tSZ_yy_total_ell{l}"]["calibrated_13.5-15.5"] for l in (100, 1000, 3000)]),
      (0.9745, 0.9855), (0.9750, 0.9860))
first = [b["strict_first_edge"] for b in R["bins"][5:12]]
check("strict mask starts at 0.065-0.32 R200c in 13.5-14.9", rng(first), (0.0645, 0.3205), (0.0655, 0.3215))

# --- AGN
loc = TS["agn_heated_location"]["P"]
check("groups 13.5-14.9 inside 0.1R 51-71%", rng([loc[l]["<0.1"]["fraction"] for l in CAL[:7]]), (0.505, 0.705), (0.515, 0.715))
check("groups 13.5-14.9 at 2-4R 15-30%", rng([loc[l]["2-4"]["fraction"] for l in CAL[:7]]), (0.145, 0.295), (0.155, 0.305))
check("clusters 14.9-15.3 at 2-4R 48-54%", rng([loc[l]["2-4"]["fraction"] for l in CAL[7:]]), (0.475, 0.535), (0.485, 0.545))
check("clusters 14.9-15.3 inside 0.1R 31-33%", rng([loc[l]["<0.1"]["fraction"] for l in CAL[7:]]), (0.305, 0.325), (0.315, 0.335))
tb = T2["top_bin_agn_per_halo"]
check("top bin inner fraction 81%", tb["bin_inner_fraction"], 0.805, 0.815)
check("top bin two haloes 48% / 46% of inner heated Y", sorted(tb["share_of_bin_inner_heated_Y"])[-2:], [0.455, 0.475], [0.465, 0.485])
check("top bin own inner fractions 98% / 91%", [tb["own_inner_fraction"][2], tb["own_inner_fraction"][3]], [0.975, 0.905], [0.985, 0.915])
ae = TS["agn_kept_extremes"]
check("E heated single shell cal <= 0.005%", ae["E"]["cal"]["max_abs_shell_dev"], 0, 5e-5)
check("E top-bin shell +1.7e-3", ae["E"]["top"]["max_abs_shell_dev"], 1.65e-3, 1.75e-3)
check("G top-bin shell +1.9e-3", ae["G"]["top"]["max_abs_shell_dev"], 1.85e-3, 1.95e-3)
check("P/E/G band medians <= 3.2e-4", max(ae[f]["cal"]["max_abs_band_median_dev"] for f in "PEG"), 0, 3.25e-4)
check("T_e / K band medians 4.1e-4 / 4.3e-4", [ae["T_e"]["cal"]["max_abs_band_median_dev"], ae["K"]["cal"]["max_abs_band_median_dev"]],
      [4.05e-4, 4.25e-4], [4.15e-4, 4.35e-4])
est = T2["all_painted_bins_estimate"]
check("AGN kept all-painted max 0.021% (mean y)", max(abs(x["all_painted_bins_ESTIMATE"]) for x in est["selection:agn_kept"].values()),
      0.0205, 0.0215)
sig = TS["impact_significance"]["selection:agn_kept"]
rel = [sig[o]["err_pct"] / sig[o]["value_pct"] for o in ("tSZ_yy_total_ell100", "tSZ_yy_total_ell1000", "tSZ_yy_total_ell3000",
                                                          "kSZ_1h_ell100", "kSZ_1h_ell1000", "kSZ_1h_ell3000", "DM_mean", "DM_var")]
check("AGN kept paired errors 5.9-7.4% of the values", rng(rel), (0.0585, 0.0735), (0.0595, 0.0745))

# --- gas phase
x = est["selection:xray"]
check("X-ray all-painted kSZ1000 -2.6%", x["kSZ_1h_ell1000"]["all_painted_bins_ESTIMATE"], -2.65, -2.55)
check("X-ray all-painted mean DM -2.23%", x["DM_mean"]["all_painted_bins_ESTIMATE"], -2.235, -2.225)
check("X-ray all-painted DM var -4.52%", x["DM_var"]["all_painted_bins_ESTIMATE"], -4.525, -4.515)
check("X-ray removes 2.0-5.1% of N_e(<4R) in 12.7-13.5", irdev("selection:xray", "E", "0-4", FALLBACK), (-5.15, -2.05), (-5.05, -1.95))
lm = T2["low_mass_cold_sf"]
check("cold/SF N_e(<R200c) 3.1-17% in 12.7-13.5", rng([lm[l]["Ne_R200c_pct"] for l in FALLBACK]), (3.05, 16.5), (3.15, 17.5))
check("cold/SF N_e(<4R) 7.3% in 12.5-12.7", lm["12.5-12.7"]["Ne_4R_pct"], 7.25, 7.35)
cs3 = [100 * R["cold_sf_share_of_total"]["E"][l]["band_strict"]["3-4"]["value"] for l in CAL + [TOP]]
check("cold/SF 3-4R electron share 2.9-3.4% in 13.5-15.5", rng(cs3), (2.85, 3.35), (2.95, 3.45))

# --- electrons
check("full ionisation N_e(<4R) +0.54 to +1.57% in 12.7-13.5", irdev("selection:e_fullion", "E", "0-4", FALLBACK), (0.535, 1.565), (0.545, 1.575))
check("gas/mu_XGP N_e(<4R) -7.3 to -8.4% in 12.7-13.5", irdev("selection:e_gas_muxgp", "E", "0-4", FALLBACK), (-8.45, -7.35), (-8.35, -7.25))
check("gas/mu_XGP N_e(<4R) -8.5 to -8.8% in 13.5-15.5", irdev("selection:e_gas_muxgp", "E", "0-4", CAL + [TOP]), (-8.85, -8.55), (-8.75, -8.45))
g = est["selection:e_gas_muxgp"]
check("gas/mu_XGP all-painted mean DM -8.2%, kSZ1000 -16.7%", [g["DM_mean"]["all_painted_bins_ESTIMATE"], g["kSZ_1h_ell1000"]["all_painted_bins_ESTIMATE"]],
      [-8.25, -16.75], [-8.15, -16.65])
fi = R["text_support"]["soap_only_conversion"]["per_particle_fullion_Ne4R_dev_pct"]
below = [100 * (1 - 1 / (1 + v / 100)) for v in fi]
check("SWIFT n_e 0.20-0.42% below full ionisation", rng(below), (0.195, 0.415), (0.205, 0.425))
neg = [100 * V["selection:e_fullion"]["band_ratio"]["E"][l][b]["dev"] for l in LAB[7:14] for b in ("0.25-0.5", "0.5-1")
       if V["selection:e_fullion"]["band_ratio"]["E"][l][b] is not None]
check("SWIFT up to 0.05% above full ionisation (0.25-1R, 13.9-15.3)", min(neg), -0.055, -0.045)

# --- ownership
a2 = T2["A2ii_single_shells"]
check("A2-ii no shell change inside 1R", max(abs(a2[f]["inside_1R"]["max_abs_shell_dev_pct"]) for f in "PEG"), 0, 0)
check("A2-ii single shells inside 1.5R <= 0.12%", max(abs(a2[f]["inside_1.5R"]["max_abs_shell_dev_pct"]) for f in "PEG"), 0.115, 0.12)
og = T2["ownership_groups"]
check("own-satellite variants tSZ ell100 -2.2 to -7.2%", og["own_satellite_removing_tSZ_ell100"]["range"], [-7.25, -2.2], [-7.15, -2.15])
check("neighbour-only max |impact| <= 0.09%", max(abs(v[1]) for v in og["neighbour_only_max_abs_impact"].values()), 0, 0.09)
bo = og["bound_only_integral_dev_0-4"]
check("bound-only Y(<4R) loss 24-52%", rng([bo[v]["P"][i] for v in bo for i in (0, 1)]), (-52.5, -24.5), (-51.5, -23.5))
check("A2-ii all-painted mean DM -4.29%", est["selection:A2ii"]["DM_mean"]["all_painted_bins_ESTIMATE"], -4.295, -4.285)
ex = [S[f"selection:exc_all_k{k}"]["impact_pct"]["tSZ_yy_total_ell3000"]["value"] for k in ("0.5", "1", "2")]
check("excision tSZ3000 -3.1 / -3.3 / +2.3%", ex, [-3.15, -3.35, 2.25], [-3.05, -3.25, 2.35])

# --- halo sample
cg = T2["clean_per_bin_groups"]
c100 = cg["tSZ_yy_total_ell100"]
check("CLEAN ell100 decomposition -1.37 / +2.17 (= +1.15 +0.16 +0.86)",
      [c100["13.5-14.9"], c100["14.9-15.5"], c100["14.9-15.1"], c100["15.1-15.3"], c100["15.3-15.5"]],
      [-1.375, 2.165, 1.145, 0.155, 0.855], [-1.365, 2.175, 1.155, 0.165, 0.865])
check("CLEAN ell3000 from 14.3-15.1 +3.99", cg["tSZ_yy_total_ell3000"]["14.3-15.1"], 3.985, 3.995)
cs = TS["impact_significance"]["selection:clean"]
check("CLEAN mean y -2.05 +- 0.29 (7 sigma)", [cs["y_mean"]["value_pct"], cs["y_mean"]["err_pct"], cs["y_mean"]["sigma"]],
      [-2.055, 0.285, 6.5], [-2.045, 0.295, 7.5])
check("CLEAN kSZ3000 2.5 sigma, DM var 1.2 sigma", [cs["kSZ_1h_ell3000"]["sigma"], cs["DM_var"]["sigma"]], [2.45, 1.15], [2.55, 1.25])
cb = S["selection:clean"]["band_dev_pct"]
check("CLEAN 1.5-3R P -3.0 to -20.1%", rng(cb["P"]["1.5-2"]["cal"] + cb["P"]["2-3"]["cal"]), (-20.15, -3.05), (-20.05, -2.95))
check("CLEAN 1.5-3R G -2.4 to -10%", rng(cb["G"]["1.5-2"]["cal"] + cb["G"]["2-3"]["cal"]), (-10.5, -2.45), (-9.5, -2.35))
ab = S["selection:overlap_annulus"]["band_dev_pct"]["G"]
check("annulus G +16 to +80% at 1.5-4R", rng(ab["1.5-2"]["cal"] + ab["2-3"]["cal"] + ab["3-4"]["cal"]), (15.5, 79.5), (16.5, 80.5))
ib = S["selection:overlap_in1R"]["band_dev_pct"]["P"]
check("in-1R 0.1-0.25 P -16 to -28%", ib["0.1-0.25"]["cal"], [-28.5, -16.5], [-27.5, -15.5])

print(f"\n{n_fail} failures")
