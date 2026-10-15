#!/usr/bin/env python3
"""Separate plot of the low-z painting fix (2026-10-07, report helper; reads saved spectra only, no recomputation).

Same quantity as panel 4 of frb_map_generation/outputs/lee22_corrected_20260930/plots/yy_power_corrected_lee22.png
(made by frb_map_generation/lee22_tsz_corrected/plot_corrected_power.py): C_ell of the 2026-09-30 maps with exact per-halo
profiles below z = 0.1 divided by C_ell of the production maps (same map2alm), and the verifier's prediction
(verify_A_xgpaint_audit/v4b_cl_*.npz: stored NSIDE 1024 map with the interpolation error removed).
Run: LD_LIBRARY_PATH=<env>/lib <env>/bin/python plot_lowz_fix_ratio.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FMG = Path("/home/cbllover/HalfDome/frb_map_generation")
N = np.load(str(FMG / "outputs/lee22_corrected_20260930/spectra/corrected_spectra.npz"))
VER = FMG / "outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
OUT = Path(__file__).resolve().parents[2] / "fig/1_lee22"
B12C, PAINT, DATA = "#0072B2", "#E69F00", "#111111"
RC = {"font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
      "legend.fontsize": 11.5, "xtick.direction": "in", "ytick.direction": "in", "savefig.facecolor": "white", "pdf.fonttype": 42}

e_n = np.arange(2, 401)
pts = np.array([3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300])
with plt.rc_context(RC):
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for k, c, lab, v in (("b12", B12C, "Battaglia12 map", "b12"), ("lee22p", PAINT, "Lee22 Table 7 map (as printed)", "lee22")):
        r = N["cl_y_" + k][e_n] / N["cl_yprod_" + k][e_n]
        ax.plot(e_n, r, color=c, lw=2.0, label=lab + ": repainted / production")
        p = np.load(str(VER / ("v4b_cl_%s.npz" % v)))
        ax.plot(pts, (p["cl_corrected"] / p["cl_stored"])[pts], "o", ms=6.5, mfc="white", mec=c, mew=1.6, ls="none",
                label=lab + ": verifier's prediction (NSIDE 1024)")
    ax.axhline(1, color=DATA, lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(2, 400); ax.set_ylim(0.6, 1.05); ax.grid(color="0.9", lw=0.6)
    ax.tick_params(which="both", top=True, right=True)
    ax.set_xlabel(r"multipole $\ell$"); ax.set_ylabel(r"$C_\ell^{yy}$ new / production")
    ax.set_title("Low-z painting fix: exact per-halo profiles for the 79,302 halos at z < 0.1", fontsize=13)
    ax.legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT / "15_lowz_fix_cl_ratio.png"), dpi=170)
    fig.savefig(str(OUT / "15_lowz_fix_cl_ratio.pdf"))
for k in ("b12", "lee22p"):
    print(k, " ".join("ell%d %.4f" % (l, N["cl_y_" + k][l] / N["cl_yprod_" + k][l]) for l in (10, 20, 30, 50, 100, 300)))
