"""Third-round text check of SECTION.md (2026-10-14).  Read-only.

Prints, from results.json and recheck_numbers.json in this folder, the per-bin
numbers behind every range in SECTION.md that the verifier flagged or that has
the same kind of risk (which bins produce a range, inner bands resolved in
fewer bins, top-bin values, per-field bootstrap factors, impacts and errors).
Run: python -B check_section_round3.py [part ...]
parts: resolved trends impact boot diag tx qm dist misc  (default: all)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
RC = json.loads((HERE / "recheck_numbers.json").read_text())
BANDS = R["bands_R200c"]
CAL = R["calibrated_bins"]
MAIN, TOP = CAL[:8], CAL[8:]
parts = sys.argv[1:] or ["all"]


def want(p):
    return "all" in parts or p in parts


def br(variant, field, b, band, key="ratio"):
    for e in R["band_ratios"][variant][field][b]:
        if e["band"] == band:
            return e.get(key)
    return None


def bt(field, b, band, key):
    for e in R["bootstrap"]["per_field"][field][b]:
        if e["band"] == band:
            return e.get(key)
    return None


def row(lab, vals, bins=MAIN + TOP):
    s = " ".join("%8s" % ("-" if v is None else ("%.4g" % v)) for v in vals)
    print("%-34s %s" % (lab, s))


def hdr(bins=MAIN + TOP):
    print("%-34s %s" % ("", " ".join("%8s" % b for b in bins)))


if want("resolved"):
    print("\n### which main bins are resolved per band (Q_V P ratio not None)")
    for band in BANDS:
        rb = [b for b in MAIN if br("estimators:Q_V", "P", b, band) is not None]
        print(band, len(rb), rb)

if want("trends"):
    print("\n### per-bin band ratios, pressure estimators")
    hdr()
    for v in ("Q_M", "Q_X", "nT_X"):
        for band in BANDS:
            row(f"{v} P {band}", [br("estimators:" + v, "P", b, band) for b in MAIN + TOP])
    print("\n### Q_X 0.5-1 errors")
    row("Q_X P 0.5-1 err", [br("estimators:Q_X", "P", b, "0.5-1", "err") for b in MAIN + TOP])
    print("\n### unweighted P, E per bin")
    for f in ("P", "E"):
        for band in BANDS:
            row(f"unw {f} {band}", [br("estimators:unweighted_mean", f, b, band) for b in MAIN + TOP])
    row("unw P 2-3 err", [br("estimators:unweighted_mean", "P", b, "2-3", "err") for b in MAIN + TOP])
    print("\n### y-weighted P per bin, errors")
    for band in BANDS:
        row(f"yw P {band}", [br("estimators:y_weighted_mean", "P", b, band) for b in MAIN + TOP])
    for band in ("2-3", "3-4"):
        row(f"yw P {band} err", [br("estimators:y_weighted_mean", "P", b, band, "err") for b in MAIN + TOP])
    print("\n### statistic errors at 3-4 R200c, P (main bins)")
    for st in ("median", "geomean", "trimmed_mean", "y_weighted_mean", "unweighted_mean"):
        e = [br("estimators:" + st, "P", b, "3-4", "err") for b in MAIN]
        print(st, "min %.4f max %.4f" % (min(e), max(e)), "argmax", MAIN[int(np.argmax(e))])
    print("\n### temperature errors, max over main bins and all bands")
    for v in ("T_M", "T_V", "T_X", "T_PE"):
        e = [(br("estimators:" + v, "T_e", b, band, "err"), b, band) for b in MAIN for band in BANDS]
        e = [x for x in e if x[0] is not None]
        print(v, max(e))
    print("\n### Q_V max paired errors per field (main bins)")
    for f in ("P", "E", "G"):
        e = [(br("estimators:Q_V", f, b, band, "err"), b, band) for b in MAIN for band in BANDS]
        e = [x for x in e if x[0] is not None]
        print(f, max(e))
    e = [(br("estimators:nT_X", "P", b, band, "err"), b, band) for b in MAIN for band in BANDS]
    print("nT_X", max(x for x in e if x[0] is not None))

if want("impact"):
    print("\n### impacts dO/O (%) +- err; resolved-only")
    for v, fl in R["impact"].items():
        for f, obs in fl.items():
            for k, o in obs.items():
                print("%-28s %-2s %-26s %+8.3f +- %.3f  res-only %+8.3f" % (
                    v, f, k, 100 * o["dO_over_O"], 100 * o["err"], 100 * o["dO_over_O_resolved_only"]))
    print("\nper-bin share Q_V tSZ ell100 / ell3000, nT_X tSZ ell100")
    for v, f, k in (("estimators:Q_V", "P", "tSZ_yy_total_ell100"), ("estimators:Q_V", "P", "tSZ_yy_total_ell3000"),
                    ("estimators:Q_V", "P", "y_mean"), ("estimators:nT_X", "P", "tSZ_yy_total_ell100"),
                    ("estimators:nT_X", "P", "y_mean"), ("estimators:unweighted_mean", "P", "tSZ_yy_total_ell100")):
        pb = R["impact"][v][f][k]["per_bin"]
        print(v, k, {b: pb[b] for b in CAL})

if want("boot"):
    print("\n### bootstrap rows (P and E), all bands")
    keys = ("rel_err_mean", "boot_over_analytic", "unstrat_over_strat", "jack_over_boot",
            "err_median_over_err_mean", "N_eff_for_3pct_mean")
    for f in ("P", "E", "G", "T_e"):
        for k in keys:
            print(f"-- {f} {k}")
            hdr()
            for band in BANDS:
                row(band, [bt(f, b, band, k) for b in MAIN + TOP])
    print("\n### per-field extremes in top bins and medians per band in main bins")
    for f in ("P", "E", "G", "T_e"):
        for k in ("unstrat_over_strat", "jack_over_boot"):
            for b in TOP:
                vals = [(bt(f, b, band, k), band) for band in BANDS if bt(f, b, band, k) is not None]
                print(f, k, b, "max", max(vals), "min", min(vals))
            med = {band: float(np.median([bt(f, b, band, k) for b in MAIN if bt(f, b, band, k) is not None]))
                   for band in BANDS}
            ngt = {band: sum(1 for b in MAIN if (bt(f, b, band, k) or 0) > 1) for band in BANDS}
            print(f, k, "main median per band", {b: round(v, 3) for b, v in med.items()}, "n>1", ngt)

if want("diag"):
    d = R["diagnostics"]
    print("\n### identities"); print(json.dumps(d["identities"], indent=0)[:3000])
    print("\n### identity extremes (recheck)"); print(json.dumps(RC["identity_extremes_all_bins_resolved_halo_shells"]))
    print("\n### check_Y200", d["check_Y200_own_vs_shell_integral"], RC["Y200_own_vs_shell_integral_extremes"])
    print("\n### unit conversion", d["mass_weighting_decomposition"]["unit_conversion_check_max_rel_dev"])
    ff = d["filling_factor"]
    print("\n### filling factor per band (fill_mean, fill_median) over main and top bins")
    for band in BANDS:
        fm = []
        for b in CAL + ["12.7-12.9"]:
            for e in ff["per_bin_band"][b]:
                if e["band"] == band:
                    fm.append((b, e.get("fill_mean"), e.get("fill_median"), e.get("fill_q16"), e.get("fill_q84")))
        print(band, fm)
    for k in ("spearman_fill_vs_lnCrho_calibrated_resolved", "median_fill_by_clumping_calibrated_resolved",
              "fill_vs_smoothing_length_calibrated_resolved", "stacked_QV_ratio_times_mean_fill_minus_1"):
        print("--", k); print(json.dumps(ff[k])[:2500])
    print("\n### clumping 14.5-14.7")
    print(d["clumping"]["per_bin_band"]["14.5-14.7"])
    print("\n### integral ratios, calibrated bins: min-max per estimator/field/u")
    for v, fl in d["integral_ratios"]["values"].items():
        for f, bins in fl.items():
            out = {}
            for u in ("<0.5R", "<1.0R", "<2.0R", "<4.0R"):
                vals = [bins[b][u] for b in CAL if b in bins and bins[b].get(u) is not None]
                if vals:
                    out[u] = (min(vals), max(vals))
            print(v, f, out)
    print("\n### class composition, calibrated bins")
    for b in CAL:
        print(b, d["class_composition"]["per_bin"][b])
    print("\n### unweighted vs weighted, 0.5-dex bins")
    for b, x in d["unweighted_vs_weighted_M200c_0.5"].items():
        print(b, "N", x["N"], "mean", x["mean_logm200c"], "unw", x["unweighted_logm200c"],
              "dlogm %+.3f" % (x["unweighted_logm200c"] - x["mean_logm200c"]))
        for f in ("P",):
            for e in x["ratio"][f]:
                if e["ratio"] is not None:
                    print("   ", f, e["band"], e["ratio"], e["err"])

if want("tx"):
    txd = R["diagnostics"]["T_X_decomposition"]["per_bin_band"]
    print("\n### T_X decomposition per bin")
    for k in ("kTe_keV", "TX_over_Te", "phase_cut_Te_xray_over_Te", "LX_weighting_TX_over_Te_xray",
              "smooth_gas_TX_over_Te_xray", "TX_all_over_TX_smooth", "LX_share_substructure_bound"):
        print("--", k)
        hdr()
        for band in BANDS:
            row(band, [txd[b][k][band] for b in MAIN + TOP])
    print(json.dumps(RC["LX_weighting_kTe_split"]))

if want("qm"):
    mwd = R["diagnostics"]["mass_weighting_decomposition"]["per_bin_band"]
    print("\n### mass-weighting decomposition per bin")
    for k in mwd[MAIN[0]]:
        print("--", k)
        hdr()
        for band in BANDS:
            row(band, [mwd[b][k][band] for b in MAIN + TOP])
    for band, v in RC["mass_weighting_shares"].items():
        print(band, v["main_range"])

if want("dist"):
    dist = R["diagnostics"]["distribution"]["per_field"]
    print("\n### distribution per bin")
    for f in ("P", "E"):
        for k in ("sd_ln", "skew_ln", "exp_sd2_over_2", "exp_cumulants_to_4th", "mean_over_geomean",
                  "mean_over_median", "corr_Y200_F", "pred_yweight_ratio", "zero_frac"):
            print("--", f, k)
            hdr()
            for band in BANDS:
                vals = []
                for b in MAIN + TOP:
                    e = [x for x in dist[f][b] if x["band"] == band][0]
                    vals.append(e.get(k))
                row(band, vals)
