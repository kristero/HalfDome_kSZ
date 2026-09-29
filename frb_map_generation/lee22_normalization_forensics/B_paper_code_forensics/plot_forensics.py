#!/usr/bin/env python3
"""Figure: (a) per-redshift-slice TNG/Lee22-literal electron-content ratio (1e13-1e14 Msun, R200c
spheres) with the candidate functional forms; (b) cumulative all-ray mean ratio versus source
redshift, showing why the constant 0.6016 ('XGPaint-native') matches TNG only near z_s = 1.
Reads the JSON products of zresolved_mean_dm_test.py (run both modes first)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "outputs/lee22_normalization_forensics_20260928/B_paper_code_forensics"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OM = 0.3089

co = json.load(open(OUT / "zresolved_mean_dm_coarse_forms.json"))["1e13-1e14 Msun"]
fine = json.load(open(OUT / "zresolved_mean_dm_test.json"))
sl = co["slices"]
zlo = np.array([s["z_lo"] for s in sl]); zhi = np.array([s["z_hi"] for s in sl])
zm = 0.5 * (zlo + zhi); R = np.array([s["R"] for s in sl])
fits = co["fits"]
zz = np.linspace(0, 2.05, 200)
E2 = OM * (1 + zz) ** 3 + 1 - OM

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
a.errorbar(zm, R, xerr=[zm - zlo, zhi - zm], fmt="o", ms=7, color=INK, mec="white", mew=1.5, lw=1.2,
           label="TNG300 / Lee22 literal (slices)", zorder=5)
a.plot(zz, fits["(1+z)^3"]["A"] * (1 + zz) ** 3, color=S1, lw=2,
       label=f"(1+z)$^3$/{fits['(1+z)^3']['inverse_A']:.2f}  (rms {fits['(1+z)^3']['rms_log']:.2f})")
a.plot(zz, fits["E^2(z)"]["A"] * E2, color=S3, lw=2, ls="--",
       label=f"E$^2$(z)/{fits['E^2(z)']['inverse_A']:.2f}  (rms {fits['E^2(z)']['rms_log']:.2f})")
a.plot(zz, fits["(1+z)^3/E^2 [repo comoving_hypothesis]"]["A"] * (1 + zz) ** 3 / E2, color=S4, lw=2, ls="-.",
       label=f"(1+z)$^3$/E$^2$ [repo option]  (rms {fits['(1+z)^3/E^2 [repo comoving_hypothesis]']['rms_log']:.2f})")
a.plot(zz, np.full_like(zz, fits["const"]["A"]), color=S2, lw=2, ls=":",
       label=f"constant {fits['const']['A']:.2f}  (rms {fits['const']['rms_log']:.2f})")
a.axhline(0.6016316602, color=INK2, lw=1, ls=(0, (1, 2)))
a.text(2.02, 0.6016 * 1.06, "XGPaint-native 0.602", color=INK2, ha="right", va="bottom", fontsize=8.5)
a.set_yscale("log"); a.set_xlim(0, 2.05)
a.set_yticks([0.3, 0.5, 1, 2, 4]); a.set_yticklabels(["0.3", "0.5", "1", "2", "4"]); a.minorticks_off()
a.set_xlabel("halo redshift z"); a.set_ylabel("N$_e$(<R200c): TNG300 / Lee22 literal")
a.set_title("(a) per-slice electron-content ratio, 10$^{13}$-10$^{14}$ M$_\\odot$", loc="left", color=INK, fontsize=10)
a.legend(frameon=False, fontsize=8.2, loc="upper left")

zs = np.array(fine["z_s"]); t = np.array(fine["tng_mean_all_rays"]); l = np.array(fine["lee22_literal_mean_all_rays"])
ok = np.isfinite(l) & (zs <= 2.95)
b.grid(True, color=GRID, lw=0.8); b.set_axisbelow(True)
b.plot(zs[ok], t[ok] / l[ok] / 0.6016316602, color=S1, lw=2, marker="o", ms=4, mec="white", mew=0.8)
b.axhline(1.0, color=INK2, lw=1, ls=(0, (1, 2)))
b.axvline(1.0, color=GRID, lw=1.2)
i1 = np.argmin(np.abs(zs - 1.0)); i01 = np.argmin(np.abs(zs - 0.1)); i2 = np.argmin(np.abs(zs - 2.0))
for i in (i01, i1, i2):
    b.annotate(f"{t[i]/l[i]/0.6016316602:.2f}", (zs[i], t[i] / l[i] / 0.6016316602), textcoords="offset points",
               xytext=(6, -12), fontsize=8.5, color=INK)
b.set_xlabel("source redshift z$_s$"); b.set_ylabel("<DM>$_{TNG}$ / <DM>$_{XGPaint-native}$ (all rays)")
b.set_title("(b) cumulative mean ratio: match only near z$_s$ = 1", loc="left", color=INK, fontsize=10)
b.set_xlim(0, 3.0)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"lee22_zresolved_tng_ratio.{ext}", dpi=160)
print("wrote", OUT / "lee22_zresolved_tng_ratio.png")
