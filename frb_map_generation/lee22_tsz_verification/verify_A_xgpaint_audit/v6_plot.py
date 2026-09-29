"""Verifier A figure: production-interpolator error at low z (vs XGPaint direct LOS) and its effect on the map C_ell."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
O = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/verify_A_xgpaint_audit"
B12, LEE = "#0072B2", "#E69F00"
R = list(csv.DictReader(open(f"{O}/v1c_itp_vs_direct_grid.csv")))
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
styles = {"0.100": ":", "1.000": "--", "3.900": "-"}
for mod, col, lab in (("b12_allz", B12, "Battaglia12 cache"), ("lee22_allz", LEE, "Lee22 Table 7 cache")):
    for x, ls in styles.items():
        rr = [r for r in R if r["map"] == mod and r["log10M"] == "14.0000" and r["x"] == x]
        z = np.array([float(r["z"]) for r in rr]); e = np.abs([float(r["rel"]) for r in rr])
        o = np.argsort(z)
        ax[0].plot(z[o], np.maximum(e[o], 1e-10), ls=ls, color=col, lw=2, label=f"{lab}, x = {float(x):.1f}")
rr = [r for r in R if r["map"] == "lee22_allz" and r["log10M"] == "14.9675" and r["x"] == "1.000"]
z = np.array([float(r["z"]) for r in rr]); e = np.abs([float(r["rel"]) for r in rr]); o = np.argsort(z)
ax[0].plot(z[o], e[o], "o-", color="#CC79A7", ms=5, lw=1.5, label=r"Lee22, $10^{14.9675}\,M_\odot$ (shape-clip kink), x = 1")
ax[0].axvline(0.0206, color="0.5", lw=1, ls="-.")
ax[0].text(0.024, 3e-8, "first cache z cell\n(0.001-0.0206) ends", fontsize=8, color="0.35")
ax[0].axvspan(0.1, 2.0, color="0.92", zorder=0)
ax[0].text(0.35, 2e-2, "workstream A\ntested z = 0.1-2", fontsize=8, color="0.35")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-10, 50)
ax[0].set_xlabel("halo redshift z"); ax[0].set_ylabel(r"|y$_{\rm interpolator}$ / y$_{\rm direct\ LOS}$ $-$ 1|")
ax[0].set_title(r"production y($\theta$) caches, $M_{200c}=10^{14}\,M_\odot$ (x = r$_\perp$/R$_{200c}$)", fontsize=10)
ax[0].legend(fontsize=7, loc="lower left", frameon=False); ax[0].grid(alpha=0.3, which="major")
for mod, col, lab in (("b12", B12, "Battaglia12 all-z map"), ("lee22", LEE, "Lee22 all-z map")):
    d = np.load(f"{O}/v4b_cl_{mod}.npz"); ell = d["ell"]
    r = d["cl_stored"] / d["cl_corrected"]
    k = np.ones(5) / 5; rs = np.convolve(r, k, mode="same")
    ax[1].plot(ell[3:-3], rs[3:-3], color=col, lw=2, label=lab)
ax[1].axhline(1, color="black", lw=0.8)
ax[1].set_xscale("log"); ax[1].set_xlim(4, 2000)
ax[1].set_xlabel(r"multipole $\ell$"); ax[1].set_ylabel(r"$C_\ell$(stored) / $C_\ell$(z<0.1 halos re-evaluated exactly)")
ax[1].set_title("effect on the stored maps (NSIDE 1024, 5-multipole running mean)", fontsize=10)
ax[1].legend(fontsize=8, frameon=False); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{O}/fig_lowz_interpolator_error.png", dpi=150)
