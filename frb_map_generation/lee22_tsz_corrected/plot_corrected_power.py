#!/usr/bin/env python3
"""Compton-y power of the corrected Lee22 maps (idark, 2026-09-30) against FLAMINGO L2p8_m9 and CLASS-SZ, and the size of
the low-z painting fix. Inputs: outputs/lee22_corrected_20260930/spectra/corrected_spectra.npz (new maps, production maps
re-transformed with the same map2alm), outputs/lee22_tsz_verification_20260929/synthesis/synthesis_inputs.npz (CLASS-SZ,
FLAMINGO lightcones 0/1 divided by pixwin^2) and verify_A_xgpaint_audit/v4b_cl_*.npz (the verifier's prediction of the
low-z fix: stored map with the interpolation error removed). HalfDome maps are point samples (no pixel window).
Run: python frb_map_generation/lee22_tsz_corrected/plot_corrected_power.py"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/lee22_corrected_20260930"
VER = ROOT / "outputs/lee22_tsz_verification_20260929"
EDGES = np.unique(np.round(np.geomspace(30, 8000, 31)).astype(int))
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])
DATA, B12C, PAINT, CORR, HYB, CAL = "#111111", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"
RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 14.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
      "legend.fontsize": 12, "xtick.direction": "in", "ytick.direction": "in", "savefig.facecolor": "white", "pdf.fonttype": 42}


def band(cl, ell):
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    return np.array([dl[lo:hi].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def main():
    N = np.load(str(OUT / "spectra/corrected_spectra.npz"))
    S = np.load(str(VER / "synthesis/synthesis_inputs.npz"))
    meta = json.loads(str(N["metadata_json"]))
    ell = N["ell"].astype(float)
    es = S["ell"].astype(float)
    D = {k: band(N["cl_y_" + k], ell) for k in ("b12", "lee22p", "lee22p_m1", "lee22p_m1_b12hi", "lee22p_m1_calib")}
    D.update({"prod_" + k: band(N["cl_yprod_" + k], ell) for k in ("b12", "lee22p")})
    for k in ("th_CS_b12_paint_all_hiprec", "th_CSX_lee22", "th_CSX_lee22m1", "flamingo_lc0", "flamingo_lc1"):
        D[k] = band(S["cl_" + k], es)
    fl = D["flamingo_lc0"]
    # analytic predictions of each map (workstream B of the 2026-09-29 verification, curves.npz): CLASS-SZ with Tinker08
    # ('CS'; Lee22 as 'CSX' = CLASS-SZ one-halo of the two mass pieces + validated Python two-halo) and the exact per-halo
    # sum over the HalfDome catalogue with the painted aperture ('CAT' = catalogue one-halo + CLASS-SZ two-halo)
    B = np.load(str(VER / "B_class_sz/analysis/curves.npz")); eb = B["ell"].astype(float)
    hyb_1h = B["th_CS_lee22m1_paint_low_1h"] + B["th_CS_b12_paint_high_1h"]            # one-halo terms add exactly
    hyb_2h = (np.sqrt(B["th_CS_lee22m1_paint_low_2h"]) + np.sqrt(B["th_CS_b12_paint_high_2h"])) ** 2   # 2h: coherent sum (approx.)
    AN = {"b12": ("th_CS_b12_paint_all_hiprec", "th_CAT_b12"), "lee22p": ("th_CSX_lee22", "th_CAT_lee22"),
          "lee22p_m1": ("th_CSX_lee22m1", "th_CAT_lee22m1"), "lee22p_m1_b12hi": (None, None),
          "lee22p_m1_calib": ("th_CS_lee22m1_paint_calib", "th_CAT_lee22m1_calib")}
    CS, CAT = {}, {}
    for k, (cs, cat) in AN.items():
        CS[k] = band(B[cs], eb) if cs else band(hyb_1h + hyb_2h, eb)
        CAT[k] = band(B[cat], eb) if cat else None
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(13.5, 15.8))
        fig.set_size_inches(13.5, 19.5)
        gs = fig.add_gridspec(4, 1, height_ratios=[3.0, 1.3, 1.45, 1.2], hspace=0.30, left=0.11, right=0.975, top=0.895, bottom=0.04)
        ax = fig.add_subplot(gs[0]); axa = fig.add_subplot(gs[1], sharex=ax); ax2 = fig.add_subplot(gs[2], sharex=ax); ax3 = fig.add_subplot(gs[3])
        ax.plot(CEN, 1e12 * fl, color=DATA, lw=3.4, zorder=7)
        ax.plot(CEN, 1e12 * D["flamingo_lc1"], color=DATA, lw=1.6, ls=(0, (1.5, 1.5)), zorder=7)
        styles = (("b12", B12C, "-", "o"), ("lee22p", PAINT, "-", "o"), ("lee22p_m1", CORR, "-", "D"),
                  ("lee22p_m1_b12hi", HYB, "-", "s"), ("lee22p_m1_calib", CAL, "-", "^"))
        for k, c, ls, mk in styles:
            ax.plot(CEN, 1e12 * D[k], color=c, lw=2.6, ls=ls, marker=mk, ms=5.5, zorder=5)
            ax.plot(CEN, 1e12 * CS[k], color=c, lw=1.8, ls="--", zorder=4)
            ax2.plot(CEN, D[k] / fl, color=c, lw=2.6, marker=mk, ms=5.5, zorder=5)
            axa.plot(CEN, D[k] / CS[k], color=c, lw=1.8, ls="--", zorder=4)
            if CAT[k] is not None:
                axa.plot(CEN, D[k] / CAT[k], color=c, lw=2.6, marker=mk, ms=5, zorder=5)
        axa.axhline(1, color=DATA, lw=1.0)
        axa.set_ylim(0.8, 1.12); axa.set_ylabel("map / analytic")
        axa.text(0.99, 0.06, "solid: / exact per-halo sum over the catalogue (painted aperture);  dashed: / CLASS-SZ (Tinker08)",
                 transform=axa.transAxes, ha="right", fontsize=11.5, color="0.3")
        ax2.fill_between(CEN, np.minimum(1, D["flamingo_lc1"] / fl), np.maximum(1, D["flamingo_lc1"] / fl), color="0.8", lw=0)
        ax2.axhline(1, color=DATA, lw=1.2)
        for a in (ax, axa, ax2):
            a.set_xscale("log"); a.set_xlim(28, 8500); a.grid(color="0.9", lw=0.6)
            a.tick_params(which="both", top=True, right=True)
            a.axvspan(28, 100, color="0.94", zorder=0)
        ax.set_yscale("log"); ax.set_ylim(0.03, 12)
        ax.set_ylabel(r"$10^{12}\,D_\ell^{yy} = 10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$")
        ax2.set_yscale("log"); ax2.set_ylim(0.2, 6); ax2.set_yticks([0.25, 0.5, 1, 2, 4]); ax2.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax2.set_ylabel("HalfDome / FLAMINGO lc0"); ax2.set_xlabel(r"$\ell$")
        axa.tick_params(labelbottom=False); ax.tick_params(labelbottom=False)
        ax2.text(0.015, 0.06, "grey band: FLAMINGO lightcone 1 / lightcone 0;  shaded: $\\ell<100$ (cosmic variance)",
                 transform=ax2.transAxes, fontsize=11.5, color="0.3")
        for x, k, c, lab, dy in ((2300, "lee22p_m1", CORR, r"Lee22$\times(1+z)^2$", 1.3), (700, "lee22p_m1_b12hi", HYB, "hybrid", 1.35),
                                 (5200, "lee22p_m1_calib", CAL, "calibrated range", 0.66), (5200, "lee22p", PAINT, "Lee22 as printed", 0.62),
                                 (1500, "b12", B12C, "B12", 0.68)):
            i = np.argmin(np.abs(CEN - x))
            ax.text(CEN[i], 1e12 * D[k][i] * dy, lab, color=c, fontsize=12.5, fontweight="bold", ha="center")
        ax.text(3300, 1e12 * fl[np.argmin(np.abs(CEN - 3300))] * 0.74, "FLAMINGO", color=DATA, fontsize=12.5, fontweight="bold", ha="center")
        # panel 3: low-z fix, new / production map spectra, per ell, with the verifier's prediction
        e_n = np.arange(2, 401)
        pts = np.array([3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300])
        for k, c in (("b12", B12C), ("lee22p", PAINT)):
            r = N["cl_y_" + k][e_n] / N["cl_yprod_" + k][e_n]
            ax3.plot(e_n, r, color=c, lw=2.0)
            v = np.load(str(VER / ("verify_A_xgpaint_audit/v4b_cl_%s.npz" % ("b12" if k == "b12" else "lee22"))))
            pred = (v["cl_corrected"] / v["cl_stored"])[pts]
            ax3.plot(pts, pred, "o", ms=6.5, mfc="white", mec=c, mew=1.6, ls="none")
        ax3.axhline(1, color=DATA, lw=1.0)
        ax3.set_xscale("log"); ax3.set_xlim(2, 400); ax3.set_ylim(0.6, 1.05); ax3.grid(color="0.9", lw=0.6)
        ax3.tick_params(which="both", top=True, right=True)
        ax3.set_xlabel(r"$\ell$"); ax3.set_ylabel("low-z fix:\nnew / production")
        ax3.text(0.99, 0.08, "lines: repainted maps (exact profiles for halos at z < 0.1);  circles: verifier's prediction (NSIDE 1024)",
                 transform=ax3.transAxes, ha="right", fontsize=11.5, color="0.3")
        h = [Line2D([], [], color=DATA, lw=3.4, label="FLAMINGO L2p8_m9 lightcone 0 ($/w_\\ell^2$)"),
             Line2D([], [], color=DATA, lw=1.6, ls=(0, (1.5, 1.5)), label="FLAMINGO lightcone 1"),
             Line2D([], [], color=B12C, lw=2.6, marker="o", label="HalfDome Battaglia12 (low-z fixed)"),
             Line2D([], [], color=PAINT, lw=2.6, marker="o", label="HalfDome Lee22 Table 7 as printed (low-z fixed)"),
             Line2D([], [], color=CORR, lw=2.6, marker="D", label=r"HalfDome Lee22$\times(1+z)^2$ (bookkeeping-corrected)"),
             Line2D([], [], color=HYB, lw=2.6, marker="s", label=r"hybrid: Lee22$\times(1+z)^2$, Battaglia12 above $10^{14.8}h^{-1}M_\odot$"),
             Line2D([], [], color=CAL, lw=2.6, marker="^", label=r"Lee22$\times(1+z)^2$, calibrated range only (1e13-$10^{14.8}h^{-1}$, $z\leq2$)"),
             Line2D([], [], color="0.4", lw=1.8, ls="--", label="dashed: CLASS-SZ halo model of the same profile (Tinker08, $x_{out}$=4)")]
        fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.995), fontsize=11.8, handlelength=2.6)
        fig.text(0.54, 0.902, "HalfDome maps painted on idark 2026-09-30 (NSIDE 4096, 4 $R_{200c}$ projected aperture, point samples); "
                 "paint cosmology $h$=0.68, $\\Omega_b$=0.049", ha="center", fontsize=11.5, color="0.3")
        (OUT / "plots").mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(str(OUT / "plots" / ("yy_power_corrected_lee22." + ext)), dpi=170 if ext == "png" else None)
        plt.close(fig)
    rows = []
    for i, c in enumerate(CEN):
        row = {"ell_band": "%d-%d" % (EDGES[i], EDGES[i + 1]), "ell_center": round(float(c), 1), "1e12_Dl_flamingo_lc0": float(1e12 * fl[i])}
        for k in ("b12", "lee22p", "lee22p_m1", "lee22p_m1_b12hi", "lee22p_m1_calib"):
            row[k + "_over_flamingo"] = float(D[k][i] / fl[i])
            row[k + "_over_class_sz"] = float(D[k][i] / CS[k][i])
            row[k + "_over_catalogue_sum"] = float(D[k][i] / CAT[k][i]) if CAT[k] is not None else float("nan")
        row["b12_over_b12prod"] = float(D["b12"][i] / D["prod_b12"][i]); row["lee22p_over_lee22pprod"] = float(D["lee22p"][i] / D["prod_lee22p"][i])
        rows.append(row)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    with open(str(OUT / "analysis/yy_power_corrected_lee22.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for l in (100, 300, 1000, 3000, 5000):
        i = np.argmin(np.abs(CEN - l))
        print("ell~%5d  " % l + "  ".join("%s %.2f" % (k, D[k][i] / fl[i]) for k in ("b12", "lee22p", "lee22p_m1", "lee22p_m1_b12hi", "lee22p_m1_calib")))
    for k in ("b12", "lee22p"):
        print("low-z fix", k, meta["lowz_fix"][k])
        for l in (10, 20, 30, 50, 100, 300):
            print("   ell %d  new/production %.4f" % (l, N["cl_y_" + k][l] / N["cl_yprod_" + k][l]))


if __name__ == "__main__":
    main()
