#!/usr/bin/env python
"""Round-4 re-verification of SECTION.md (2026-10-14), part A: numbers read from results.json.

Prints the values behind the 8 previously wrong claims and behind spot-checked statements.
Read-only. Run: python -B verify/v4_check.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRP = HERE.parent
R = json.loads((GRP / "results.json").read_text())
BANDS = R["bands_R200c"]
CAL = R["calibrated_bins"]
MAIN, TOP = CAL[:8], CAL[8:]


def br(v, f, b, band, key="ratio"):
    for e in R["band_ratios"][v][f][b]:
        if e["band"] == band:
            return e[key]
    raise KeyError((v, f, b, band))


def rng(vals):
    vals = [x for x in vals if x is not None]
    return (min(vals), max(vals), len(vals)) if vals else None


def hdr(t):
    print("\n==== " + t)


# ---------------------------------------------------------------- claim 0 and Q_M / Q_X trends
hdr("C0: Q_M / Q_X (P) per band, main bins + top bins (ratio +- err)")
for v in ("estimators:Q_M", "estimators:Q_X"):
    for band in BANDS:
        row = [(b, br(v, "P", b, band), br(v, "P", b, band, "err")) for b in CAL]
        print(f" {v[11:]:4s} {band:>8}: " + "  ".join(f"{b}:{r:.4g}+-{e:.2g}" for b, r, e in row if r is not None))

# ---------------------------------------------------------------- claim 4: nT_X
hdr("C4: n_e kT_X (P) band ratios")
for b in CAL:
    print(f" {b}: " + "  ".join(f"{band}:{br('estimators:nT_X','P',b,band):.4g}" if br('estimators:nT_X','P',b,band) is not None else f"{band}:-" for band in BANDS))
for band in BANDS:
    print(f"  {band}: main range {rng([br('estimators:nT_X','P',b,band) for b in MAIN])}, max err {max([br('estimators:nT_X','P',b,band,'err') or 0 for b in MAIN]):.3g}")

# ---------------------------------------------------------------- claims 1/2: bootstrap
hdr("C1/C2: bootstrap per field, band values")
BP = R["bootstrap"]["per_field"]
for fld in ("P", "E", "G", "T_e"):
    for q in ("unstrat_over_strat", "jack_over_boot", "boot_over_analytic"):
        print(f" -- {fld} {q}")
        for band in BANDS:
            vals = {b: next(e for e in BP[fld][b] if e["band"] == band) for b in CAL}
            main = [vals[b].get(q) for b in MAIN if vals[b]["resolved"]]
            top = [vals[b].get(q) for b in TOP if vals[b]["resolved"]]
            mm = [(b, vals[b].get(q)) for b in MAIN if vals[b]["resolved"]]
            if main:
                arg = max(mm, key=lambda t: t[1])
                print(f"   {band:>8}: main {min(main):.3g}-{max(main):.3g} (max {arg[0]}) median {np.median(main):.3g}"
                      f" n>1: {sum(x > 1 for x in main)}/{len(main)}; top {top}")
            else:
                print(f"   {band:>8}: main unresolved; top {top}")
for fld in ("P", "E", "G", "T_e"):
    for b in TOP:
        u = [(e["band"], e.get("unstrat_over_strat"), e.get("jack_over_boot")) for e in BP[fld][b] if e["resolved"]]
        print(f" {fld} {b}: unstrat max {max(x[1] for x in u):.3g} at {max(u, key=lambda t: t[1])[0]}, "
              f"jack {min(x[2] for x in u):.3g}-{max(x[2] for x in u):.3g} (max at {max(u, key=lambda t: t[2])[0]})")

hdr("Section 4 table rows (P, E): rel err, SD median/mean, N_eff 3%")
for fld in ("P", "E"):
    for q in ("rel_err_mean", "err_median_over_err_mean", "N_eff_for_3pct_mean"):
        print(f" -- {fld} {q}")
        for band in BANDS:
            vals = {b: next(e for e in BP[fld][b] if e["band"] == band) for b in CAL}
            main = [vals[b].get(q) for b in MAIN if vals[b]["resolved"]]
            top = [vals[b].get(q) for b in TOP if vals[b]["resolved"]]
            if main:
                print(f"   {band:>8}: main {min(main):.4g}-{max(main):.4g}; top {top}")

# ---------------------------------------------------------------- claim 3: mass weighting decomposition
hdr("C3: mass_weighting_decomposition, ranges over the 8 main bins")
MW = R["diagnostics"]["mass_weighting_decomposition"]["per_bin_band"]
keys = list(MW[MAIN[0]].keys())
for k in keys:
    out = []
    for band in BANDS:
        vals = [MW[b][k][band] for b in MAIN if MW[b][k][band] is not None]
        if vals:
            out.append(f"{band}:{min(vals):.4g}-{max(vals):.4g}")
    print(f" {k}: " + "  ".join(out))
# smooth-hot log-excess fractions
for band in ("2-3", "3-4"):
    fr = [np.log(MW[b]["P_QM_over_geom_smooth_hot_only"][band]) / np.log(MW[b]["P_QM_over_geom_all"][band]) for b in MAIN]
    frg = [np.log(MW[b]["G_QM_over_geom_smooth_hot_only"][band]) / np.log(MW[b]["G_QM_over_geom_all"][band]) for b in MAIN]
    print(f" log-excess fraction smooth/all {band}: P {min(fr):.3f}-{max(fr):.3f}; G {min(frg):.3f}-{max(frg):.3f}")
for b in MAIN:
    print(f"  {b} 3-4: sum_m_pe_share_cold {MW[b]['sum_m_pe_share_cold']['3-4']:.3f}  "
          f"not-smooth-hot mass share {1 - MW[b]['mass_share_smooth_hot']['3-4']:.3f}")

# ---------------------------------------------------------------- spot checks
hdr("S1: impacts (%)")
IM = R["impact"]
for v in IM:
    for f in IM[v]:
        s = []
        for o, d in IM[v][f].items():
            s.append(f"{o}={100*d['dO_over_O']:+.4g}+-{100*d['err']:.2g} (res {100*d['dO_over_O_resolved_only']:+.4g})")
        print(f" {v} {f}: " + "; ".join(s))
pb = IM["estimators:Q_V"]["P"]["tSZ_yy_total_ell100"]["per_bin"]
print("  Q_V tSZ ell100 share 14.5-15.5:", 100 * sum(pb[b] for b in CAL[5:]), "of", 100 * sum(pb.values()))
print("  nT_X tSZ ell100 per_bin:", IM["estimators:nT_X"]["P"]["tSZ_yy_total_ell100"]["per_bin"])

hdr("S2: Table 1.2 Q_V, Table 1.3, T table")
for v, f in (("estimators:Q_V", "P"), ("estimators:Q_V_y", "P"), ("estimators:Q_V", "E"), ("estimators:Q_V", "G"),
             ("estimators:Q_M", "G"), ("estimators:T_M", "T_e"), ("estimators:T_V", "T_e"), ("estimators:T_X", "T_e"),
             ("estimators:T_PE", "T_e")):
    print(f" -- {v} {f}")
    for band in BANDS:
        main = [(b, br(v, f, b, band), br(v, f, b, band, "err")) for b in MAIN if br(v, f, b, band) is not None]
        top = [round(br(v, f, b, band), 4) for b in TOP if br(v, f, b, band) is not None]
        if main:
            print(f"   {band:>8}: {min(x[1] for x in main):.4g}-{max(x[1] for x in main):.4g} ({len(main)} bins, "
                  f"max err {max(x[2] for x in main):.3g}); per bin {[round(x[1], 4) for x in main]}; top {top}")
qvp = [br("estimators:Q_V", "P", b, bd) for b in MAIN for bd in BANDS if br("estimators:Q_V", "P", b, bd) is not None]
d = [abs(br("estimators:Q_V", "G", b, bd) - br("estimators:Q_V", "E", b, bd)) for b in MAIN for bd in BANDS
     if br("estimators:Q_V", "E", b, bd) is not None]
print("  max |Q_V G - Q_V E| over main bins:", max(d))

hdr("S3: statistics band ratios")
for st in ("median", "geomean", "trimmed_mean", "y_weighted_mean", "unweighted_mean"):
    for f in ("P", "E", "T_e", "K"):
        out = []
        for band in BANDS:
            main = [br(f"estimators:{st}", f, b, band) for b in MAIN if br(f"estimators:{st}", f, b, band) is not None]
            errs = [(b, br(f"estimators:{st}", f, b, band, "err")) for b in MAIN if br(f"estimators:{st}", f, b, band) is not None]
            if main:
                me = max(errs, key=lambda t: t[1])
                out.append(f"{band}:{min(main):.4g}-{max(main):.4g}" + (f"[maxerr {me[1]:.3g} {me[0]}]" if band == "3-4" else ""))
        print(f" {st} {f}: " + "  ".join(out))

hdr("S4: filling factor")
FF = R["diagnostics"]["filling_factor"]["per_bin_band"]
for band in BANDS:
    vals = [(b, e["fill_mean"]) for b in CAL for e in FF[b] if e["band"] == band and e["fill_mean"] is not None]
    print(f"  {band}: {len(vals)} bins {vals[0][0]}..{vals[-1][0]}: {min(v for _, v in vals):.4f}-{max(v for _, v in vals):.4f}; "
          f"main spread {max(v for b, v in vals if b in MAIN) - min(v for b, v in vals if b in MAIN):.4f}")
for b in ("12.7-12.9", "13.5-13.7", "14.3-14.5", "14.5-14.7", "14.9-15.1"):
    e = [e for e in FF[b] if e["band"] == "3-4"][0]
    print(f"  {b} 3-4: mean {e['fill_mean']}, q16 {e['fill_q16']}, q84 {e['fill_q84']}")
mf = R["diagnostics"]["filling_factor"]["median_fill_by_clumping_calibrated_resolved"]
for e in mf:
    print("  ", e)
sq = R["diagnostics"]["filling_factor"]["stacked_QV_ratio_times_mean_fill_minus_1"]
print("  1/<f> vs QV: median", min(v["median_abs_dev"] for v in sq.values()), max(v["median_abs_dev"] for v in sq.values()),
      " max", min(v["max_abs_dev"] for v in sq.values()), max(v["max_abs_dev"] for v in sq.values()))

hdr("S5: integral ratios, bins 13.5-15.5")
IR = R["diagnostics"]["integral_ratios"]["values"]
for v in IR:
    for f in IR[v]:
        s = []
        for u in ("<1.0R", "<2.0R", "<4.0R"):
            vals = [IR[v][f][b][u] for b in CAL]
            s.append(f"{u}:{min(vals):.4g}-{max(vals):.4g}")
        print(f" {v} {f}: " + " ".join(s))

hdr("S6: T_X decomposition, main bins")
TD = R["diagnostics"]["T_X_decomposition"]["per_bin_band"]
for k in TD[MAIN[0]]:
    out = []
    for band in BANDS:
        vals = [TD[b][k][band] for b in MAIN if TD[b][k][band] is not None]
        if vals:
            out.append(f"{band}:{min(vals):.4g}-{max(vals):.4g}")
    print(f" {k}: " + "  ".join(out))
# kTe split over the 10 calibrated bins
lo, hi, mid = [], [], []
for b in CAL:
    for band in BANDS:
        kt, lx = TD[b]["kTe_keV"][band], TD[b]["LX_weighting_TX_over_Te_xray"][band]
        if kt is None or lx is None:
            continue
        (lo if kt < 0.7 else hi if kt >= 1.5 else mid).append((b, band, lx))
print(f"  kTe<0.7: n={len(lo)} {min(x[2] for x in lo)}-{max(x[2] for x in lo)}; >=1.5: n={len(hi)} "
      f"{min(x[2] for x in hi)}-{max(x[2] for x in hi)}; main-only n={sum(x[0] in MAIN for x in hi)} "
      f"{min(x[2] for x in hi if x[0] in MAIN)}-{max(x[2] for x in hi if x[0] in MAIN)}; mid n={len(mid)} "
      f"{min(x[2] for x in mid)}-{max(x[2] for x in mid)}")
for grp, bins in (("13.5-14.1", MAIN[:3]), ("14.3-15.1", MAIN[4:])):
    for band in ("1-1.5", "1.5-2", "2-3", "3-4"):
        v = [TD[b]["TX_all_over_TX_smooth"][band] for b in bins]
        print(f"  TX_all/TX_smooth {grp} {band}: {min(v):.4g}-{max(v):.4g}")
for b in MAIN[:3]:
    print(f"  {b} 3-4: smooth LX {TD[b]['smooth_gas_TX_over_Te_xray']['3-4']}, all LX {TD[b]['LX_weighting_TX_over_Te_xray']['3-4']}")

hdr("S7: clumping 14.5-14.7, distribution, class composition")
CL = R["diagnostics"]["clumping"]["per_bin_band"]["14.5-14.7"]
print("  C_rho med", [e["C_rho_median"] for e in CL]); print("  C_P med", [e["C_P_median"] for e in CL]); print("  r", [e["r_rhoP_median"] for e in CL])
DI = R["diagnostics"]["distribution"]["per_field"]["P"]
for band in BANDS:
    es = [next(e for e in DI[b] if e["band"] == band) for b in MAIN]
    es = [e for e in es if e["resolved"]]
    if not es:
        continue
    g = lambda k: (min(e[k] for e in es), max(e[k] for e in es))
    dev = max(abs(e["mean_over_geomean"] / e["exp_sd2_over_2"] - 1) for e in es)
    print(f"  {band}: sd_ln {g('sd_ln')}, skew {g('skew_ln')}, m/gm {g('mean_over_geomean')}, lognorm {g('exp_sd2_over_2')}, "
          f"cum4 {g('exp_cumulants_to_4th')}, m/med {g('mean_over_median')}, corrYF {g('corr_Y200_F')}, maxdev {dev:.4f}")
CC = R["diagnostics"]["class_composition"]["per_bin"]
for c in ("class1", "class2"):
    u = [CC[b][c]["unweighted_frac"] for b in MAIN[:5]]
    w = [CC[b][c]["weighted_frac"] for b in MAIN[:5]]
    print(f"  {c} 13.5-14.5: unweighted {min(u):.4f}-{max(u):.4f}, weighted {min(w):.4f}-{max(w):.4f}")
UW = R["diagnostics"]["unweighted_vs_weighted_M200c_0.5"]
for b in ("12.5-13.0", "13.0-13.5", "13.5-14.0", "14.0-14.5"):
    print(f"  {b}: dlogM {UW[b]['unweighted_logm200c'] - UW[b]['mean_logm200c']:.4f}")

hdr("S8: N per bin and strata")
BI = R["bin_info"]
print("  ", {b: (BI[b]["N"], BI[b]["N_eff"]) for b in CAL})
