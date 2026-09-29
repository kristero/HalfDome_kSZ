"""Figures for workstream A.  (1) painted y(theta) interpolators vs the independent LOS integral;
(2) overview of the exported 3-D tables at M200c = 1e14 Msun.  Colours per the session rules.
Run after compare_painted_profiles.py and make_profile_tables.py."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit")
BLUE, ORANGE, GREEN, VERM, PINK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#555555", "legend.frameon": False})

# ------------------------------------------------------------------ figure 1
rows = list(csv.DictReader(open(OUT / "painted_profile_vs_independent_los.csv")))
for r in rows:
    for k in r:
        if k != "model":
            r[k] = float(r[k])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for model, col, lab in (("b12", BLUE, "Battaglia12 (painted interpolator)"), ("lee22", ORANGE, "Lee22 Table 7 (painted interpolator)")):
    for lm, lw in ((13.0, 1.2), (14.0, 1.8), (15.0, 2.4)):
        sel = [r for r in rows if r["model"] == model and r["logM"] == lm and r["z"] == 0.5]
        x = np.array([r["x"] for r in sel]); yi = np.array([r["y_itp"] for r in sel])
        yl = np.array([r["y_indep_matched"] for r in sel])
        ax.plot(x, yi, color=col, lw=lw, label=lab if lm == 14.0 else None)
        ax.plot(x[::3], yl[::3], "o", ms=3.5, mfc="none", mec="black", mew=0.8,
                label="independent LOS integral (lee22_models.py)" if (model == "b12" and lm == 14.0) else None)
        if model == "b12":
            ax.text(0.017, yi[0], f"$10^{{{int(lm)}}}$", ha="right", va="center", fontsize=8, color="#333333")
ax.axvline(4.0, color="#888888", ls="--", lw=1)
ax.text(3.6, 1.2e-4, "4 R200c\naperture", ha="right", va="top", fontsize=8, color="#555555")
ax.set(xscale="log", yscale="log", xlim=(0.009, 5), ylim=(1e-9, 3e-4), xlabel=r"$x = r_\perp / R_{200c}$",
       ylabel="Compton-y", title=r"z = 0.5, $M_{200c}$ = $10^{13,14,15}\,M_\odot$ (line width)")
ax.legend(fontsize=8, loc="lower left")
ax = axes[1]
for model, col, lab in (("b12", BLUE, "Battaglia12"), ("lee22", ORANGE, "Lee22 Table 7")):
    first = True
    for lm in (13.0, 13.5, 14.0, 14.5, 15.0):
        for z in (0.1, 0.5, 1.0, 2.0):
            sel = [r for r in rows if r["model"] == model and r["logM"] == lm and r["z"] == z]
            x = np.array([r["x"] for r in sel])
            d = np.array([r["y_itp"] / r["y_indep_matched"] - 1 for r in sel])
            ax.plot(x, 1e4 * d, color=col, lw=0.8, alpha=0.6, label=f"{lab}: 20 (M, z)" if first else None)
            first = False
ax.axhline(0, color="black", lw=0.8)
ax.axvline(4.0, color="#888888", ls="--", lw=1)
ax.set(xscale="log", xlim=(0.015, 5), xlabel=r"$x = r_\perp / R_{200c}$",
       ylabel=r"$10^4 \times$ (interpolator / independent $-$ 1)",
       title="painted interpolator vs independent integral\n(same rho_crit incl. radiation)")
ax.annotate("Lee22, $10^{15}\\,M_\\odot$: shape clip at $10^{14.97}$\n(kink in log M, 1 grid cell away)", xy=(2.0, -4.6), xytext=(0.03, -4.2), fontsize=8, color="#555555", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.annotate("z = 0.1: interpolation in z\nat the low-z grid end", xy=(3.0, -0.85), xytext=(0.3, -1.6), fontsize=8, color="#555555", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig_painted_profile_check.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ figure 2
t = np.load(OUT / "profile_tables.npz")
X, Z, LM = t["x_r200c"], t["z"], t["log10_M200c_msun"]
i = int(np.argmin(np.abs(LM - 14.0)))
zsel = [0.0, 1.0, 2.0]
fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True)
for c, z in enumerate(zsel):
    j = int(np.argmin(np.abs(Z - z)))
    ax = axes[0, c]
    ax.axvspan(0.04, 1.34, color="#f0f0f0", zorder=0)
    ax.plot(X, t["Pe_keV_cm3_b12"][i, j], color=BLUE, lw=2, label="Battaglia12 x 0.5176 (painted)")
    ax.plot(X, t["Pe_keV_cm3_lee22_painted"][i, j], color=ORANGE, lw=2, label="Lee22 Table 7 as painted")
    ax.plot(X, t["Pe_keV_cm3_lee22_x1pz2"][i, j], color=GREEN, lw=2, ls="--", label=r"Lee22 x $(1+z)^2$ (M1 hypothesis)")
    ax.set(xscale="log", yscale="log", title=f"electron pressure, z = {z:g}", ylim=(1e-7, 0.6))
    if c == 0:
        ax.set_ylabel(r"$P_e$ [keV cm$^{-3}$], physical")
        ax.legend(fontsize=8, loc="lower left")
    ax = axes[1, c]
    ax.axvspan(0.04, 1.34, color="#f0f0f0", zorder=0)
    ax.plot(X, t["ne_cm3_b16_xgpaint"][i, j], color=BLUE, lw=2, label="Battaglia16 (XGPaint ne2d)")
    ax.plot(X, t["ne_cm3_lee22_literal"][i, j], color=VERM, lw=2, label="Lee22 Table 8 literal eq. 9")
    ax.plot(X, t["ne_cm3_lee22_xgpaint_ne2d"][i, j], color=ORANGE, lw=2, label="Lee22 x 0.6016 (production)")
    ax.plot(X, t["ne_cm3_lee22_tngcal"][i, j], color=GREEN, lw=2, ls="--", label="Lee22 x C(z) (tng_calibrated)")
    ax.set(xscale="log", yscale="log", title=f"electron density, z = {z:g}", xlabel=r"$x = r/R_{200c}$", ylim=(1e-6, 3e-1))
    if c == 0:
        ax.set_ylabel(r"$n_e$ [cm$^{-3}$], physical")
        ax.legend(fontsize=8, loc="lower left")
fig.suptitle(r"profile_tables.npz at $M_{200c} = 10^{14}\,M_\odot$ (physical; $6.8\times10^{13}\,h^{-1}M_\odot$), HalfDome cosmology;"
             " grey band = Lee22 fitted radii 0.04-1.34 R200c", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "fig_profile_tables_overview.png", dpi=150)
plt.close(fig)
print("figures written")
