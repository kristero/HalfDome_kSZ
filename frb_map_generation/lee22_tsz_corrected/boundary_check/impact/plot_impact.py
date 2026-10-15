#!/usr/bin/env python
"""Figure for the Lee22 boundary-bug impact test: (A) CAMELS-TNG measured stacked n_e and P_e ratio buggy/fixed in the
Lee22 mass range; (B) TNG300 expected bias of the stacked mean (analytic, Lee22 bins with N >= 30) with the CAMELS gas
placed in a 205 Mpc/h box.  Reads impact/tables/*.csv written by analyze_impact.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/impact"
C = np.genfromtxt(os.path.join(OUT, "tables", "camels_stack_ratio_per_shell.csv"), delimiter=",", names=True, dtype=None, encoding=None)
T = np.genfromtxt(os.path.join(OUT, "tables", "tng300_expected_bias_per_shell.csv"), delimiter=",", names=True, dtype=None, encoding=None)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BAND = "#cde2fb"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def xc(tab, m):
    lo = np.maximum(tab["x_lo"][m], 0.03 / 1.035)
    return np.sqrt(lo * tab["x_hi"][m])


fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
for ax, q, title in ((axs[0], "ne", "CAMELS-TNG, electron density"), (axs[1], "pe", "CAMELS-TNG, electron pressure")):
    m = (np.abs(C["z"] - 0.0) < 0.01) & (C["logM_lo"] == 13.0) & (C["logM_hi"] == 14.2) & (C["q"] == q)
    x = xc(C, m)
    nh = int(C["nhalo"][m][0])
    ax.axvspan(0.025, 0.04, color=SHADE, lw=0)
    ax.axvspan(1.34, 2.6, color=SHADE, lw=0)
    ax.fill_between(x, (C["ratio_shift_p16"][m] - 1) * 100, (C["ratio_shift_p84"][m] - 1) * 100, color=BAND, lw=0,
                    label="16 random periodic translations, 16th-84th pct")
    ax.plot(x, (C["ratio_s0"][m] - 1) * 100, color=BLUE, lw=2, label="actual CAMELS box faces")
    ax.plot(x, (C["expL"][m] - 1) * 100, color=ORANGE, lw=2, ls="--", label="exact expectation over translations")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.025, 2.6)
    ax.set_xlabel("r / R200c")
    ax.set_ylabel("stacked buggy / fixed - 1  [%]")
    ax.set_title(f"{title}: c3fe9b8 (buggy) vs fixed\n25 Mpc/h boxes, z = 0, logM 13.0-14.2 ({nh} halos)", fontsize=9, color=INK, loc="left")
    ax.grid(True, color=GRID, lw=0.6)
axs[0].legend(loc="lower left", frameon=False, fontsize=8)

ax = axs[2]
good = T["N_halos"] >= 30
env_lo, env_hi, xt = [], [], None
for k in range(128):
    m = good & (T["shell"] == k)
    env_lo.append(T["expected_bias"][m].min())
    env_hi.append(T["expected_bias"][m].max())
m0 = good & (T["z"] == 0.0) & (T["logM_lo"] == 13.0)
xt = np.sqrt(np.maximum(T["x_lo"][m0], 0.03 / 1.035) * T["x_hi"][m0])
ax.axvspan(0.025, 0.04, color=SHADE, lw=0)
ax.axvspan(1.34, 2.6, color=SHADE, lw=0)
ax.fill_between(xt, np.array(env_lo) * 100, np.array(env_hi) * 100, color=BAND, lw=0,
                label="TNG300 expectation, 51 Lee22 bins (z 0-2, logM 13.0-14.2, N >= 30)")
for q, ls, lab in (("ne", "-", "CAMELS gas in a 205 Mpc/h box, n_e"), ("pe", ":", "same, P_e")):
    m = (np.abs(C["z"] - 0.0) < 0.01) & (C["logM_lo"] == 13.0) & (C["logM_hi"] == 14.2) & (C["q"] == q)
    ax.plot(xc(C, m), (C["expT"][m] - 1) * 100, color=ORANGE, lw=2, ls=ls, label=lab + " (z = 0, logM 13.0-14.2)")
m = (np.abs(C["z"] - 0.0) < 0.01) & (C["logM_lo"] == 13.0) & (C["logM_hi"] == 14.2) & (C["q"] == "ne")
ax.plot(xc(C, m), (C["isoT"][m] - 1) * 100, color=AQUA, lw=2, ls="--", label="isotropic law, same halos")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xscale("log")
ax.set_xlim(0.025, 2.6)
ax.set_xlabel("r / R200c")
ax.set_ylabel("expected stacked bias  [%]")
ax.set_title("TNG300 box (205 Mpc/h): expected bias of the stacked mean\n(random face placement)", fontsize=9, color=INK, loc="left")
ax.grid(True, color=GRID, lw=0.6)
ax.set_ylim(-1.3, 0.05)
ax.legend(loc="lower left", frameon=False, fontsize=7.5)
fig.text(0.5, -0.02, "grey bands: outside the Lee22 fit range 0.04-1.34 R200c", ha="center", color=INK2, fontsize=8)
fig.savefig(os.path.join(OUT, "fig_lee22_boundary_bug_impact.png"), dpi=160, bbox_inches="tight")
print("wrote", os.path.join(OUT, "fig_lee22_boundary_bug_impact.png"))
