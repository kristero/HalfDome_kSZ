#!/usr/bin/env python3
"""One figure for the Konietzka-definition check (reads konietzka_check_results.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_konietzka_check_20260930")
R = json.loads((OUT / "konietzka_check_results.json").read_text())
# reference categorical palette (dataviz skill, light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                           "#008300", "#4a3aa7", "#e34948")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

sh = R["shells"]
z = np.array([s["z"] for s in sh])
g = lambda k: np.array([s[k] for s in sh])  # noqa: E731
fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.4), gridspec_kw=dict(width_ratios=[1.05, 1.0, 1.1]))

# (a) implied f_e
a = ax[0]
a.errorbar(z, g("fe_K"), yerr=g("fe_K_err"), fmt="o", ms=5, color=INK, label="Konietzka TNG300 'mass13to14' (implied)",
           zorder=5)
a.plot(z, g("fe_camels_nominal"), color=BLUE, lw=2, label=r"CAMELS $R_{200c}$, window $10^{13}$-$10^{14}$")
a.plot(z, g("fe_camels_nominal_hydroN"), color=BLUE, lw=2, ls="--", label=r"same, $\times$ TNG300 hydro abundance")
a.plot(z, g("fe_camels_ext"), color=ORANGE, lw=2, label=r"CAMELS $R_{200c}$, window $10^{13}$-$9\times10^{14}$")
a.plot(z, g("fe_camels_ext_hydroN"), color=ORANGE, lw=2, ls="--", label=r"same ($9\times10^{14}$), $\times$ TNG300 hydro abundance")
a.set_xlabel("halo redshift (0.1-wide shells)")
a.set_ylabel(r"implied $f_e(<R_{200c})$ of the $10^{13}$-$10^{14}\,M_\odot$ window")
a.set_ylim(0.3, 1.05)

a.set_title("(a) electron content per shell (Tinker08-normalised)", fontsize=10, loc="left", color=INK)

# (b) hit ratio
b = ax[1]
b.plot(z, g("hit_tng_over_nominal"), "o", ms=5, color=INK, label="Konietzka hit fraction / model", zorder=5)
b.axhline(1.0, color=BLUE, lw=2, label=r"$R_{200c}$, nominal window")
alo = np.interp(z, [0, 1], [R["A_lo"]["0.0"], R["A_lo"]["1.0"]])
b.plot(z, alo, color=BLUE, lw=2, ls="--", label="  same, TNG300 hydro abundance")
b.plot(z, g("hit_ext_over_nom"), color=ORANGE, lw=2, label=r"$R_{200c}$, window to $9\times10^{14}$")
b.plot(z, g("hit_ext_over_nom_hydroN"), color=ORANGE, lw=2, ls="--", label="  same, TNG300 hydro abundance")
b.set_xlabel("halo redshift (0.1-wide shells)")
b.set_ylabel("per-shell hit fraction / nominal model")
b.set_ylim(0.6, 1.4)
hA, lA = a.get_legend_handles_labels()
from matplotlib.lines import Line2D  # noqa: E402
blank = Line2D([], [], alpha=0)
kidx = [i for i, l in enumerate(lA) if l.startswith("Konietzka")][0]
rest = [i for i in range(len(lA)) if i != kidx]
fig.legend([hA[kidx], blank] + [hA[i] for i in rest],
           ["Konietzka TNG300 'mass13to14' (a: implied f_e, b: hit fraction)", " "] + [lA[i] for i in rest],
           loc="lower left", bbox_to_anchor=(0.04, 0.0), ncol=3, fontsize=8.2)
b.set_title("(b) ray hit fraction (R200c spheres = 1)", fontsize=10, loc="left", color=INK)

# (c) electrons vs area at z ~ 0
c = ax[2]
d0 = R["camels_definitions"]
z0 = min(d0, key=lambda k: float(k))
D = d0[z0]
pick = [("R200c, SF n_e = 0 (Konietzka+25)", "SF $n_e$=0", GREEN, "s"),
        ("R200m sphere", "$R_{200m}$", VIOLET, "D"),
        ("1.5 R200c sphere", "1.5 $R_{200c}$", AQUA, "^"),
        ("2 R200c sphere", "2 $R_{200c}$", AQUA, "v"),
        ("FoF members, all", "FoF cells", YELLOW, "P"),
        ("rho_DM > 57 rho_crit,0 (comoving), FoF", r"$\rho_{DM}>57\rho_{c,0}$", MAGENTA, "X"),
        ("rho_DM > 28.5 rho_crit,0 (comoving), FoF", r"$>28.5\rho_{c,0}$", MAGENTA, "x"),
        ("rho_DM > 114 rho_crit,0 (comoving), FoF", r"$>114\rho_{c,0}$", MAGENTA, "+")]
for key, lab, col, mk in pick:
    v = D[key]
    c.errorbar(v["area"], v["Ne"], xerr=v["area_err"], yerr=v["Ne_err"], fmt=mk, ms=7, color=col, mec=col, lw=1)
    c.annotate(lab, (v["area"], v["Ne"]), textcoords="offset points", xytext=(6, 5 if mk == "s" else -3), fontsize=7.8,
               color=INK2)
s0 = sh[0]
c.plot([1.0], [1.0], "o", ms=8, mfc="none", mec=BLUE, mew=2)
c.annotate("$R_{200c}$ baseline", (1.0, 1.0), textcoords="offset points", xytext=(-30, -18), fontsize=7.8, color=INK2)
tx, ty = s0["hit_tng_over_nominal"], s0["fe_K"] / s0["fe_camels_nominal"]
c.plot([tx], [ty], "*", ms=15, color=INK)
c.annotate("Konietzka (target)", (tx, ty), textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK)
ex, ey = s0["hit_ext_over_nom"], s0["fe_camels_ext"] / s0["fe_camels_nominal"]
eh, ehy = s0["hit_ext_over_nom_hydroN"], s0["fe_camels_ext_hydroN"] / s0["fe_camels_nominal"]
c.plot([ex], [ey], "o", ms=8, color=ORANGE)
c.plot([eh], [ehy], "o", ms=8, mfc="white", mec=ORANGE, mew=2)
c.annotate("", xy=(eh, ehy), xytext=(ex, ey), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
c.annotate(r"window to $9\times10^{14}$", (ex, ey), textcoords="offset points", xytext=(6, 4), fontsize=7.8, color=INK2)
c.annotate("+ hydro abundance", (eh, ehy), textcoords="offset points", xytext=(-20, -13), fontsize=7.8, color=INK2)
c.set_xscale("log")
c.set_xticks([0.7, 1, 1.5, 2, 3, 4])
c.set_xticklabels(["0.7", "1", "1.5", "2", "3", "4"])
c.set_xlim(0.65, 5)
c.set_ylim(0.5, 2.4)
c.set_xlabel("hit area / $\\pi R_{200c}^2$ (window-summed)")
c.set_ylabel("electrons / $N_e(<R_{200c})$ (window-summed)")
c.set_title(f"(c) candidate definitions, CAMELS z = {float(z0):.2f}", fontsize=10, loc="left", color=INK)
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_konietzka_check.{ext}", dpi=150)
print("wrote", OUT / "fig_konietzka_check.png")
