#!/usr/bin/env python3
"""Figures for the F_sz_budget analysis (reads tables_*.npz, budget_results.json, budget_curves.npz)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as A  # noqa: E402
import szhm as S  # noqa: E402

OUT = A.OUT
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2.0})
res = json.loads((OUT / "budget_results.json").read_text())
cv = dict(np.load(OUT / "budget_curves.npz"))
kfit = res["konietzka_zs_test"]["fits_zs_0p1_to_2"]["C0(1+z)^k"]["params"]
c3 = res["konietzka_zs_test"]["fits_zs_0p1_to_2"]["C0(1+z)^3"]["params"][0]
C_tng = lambda z: kfit[0] * (1 + z) ** kfit[1]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "{}.{}".format(name, ext), dpi=170 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 1. redshift budgets
z = cv["validation_z"]
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
zf, wf = cv["frb_z"], cv["frb_w_literal"]
for ax in axes:
    ax.fill_between(zf, 0, wf, color="#b8b7b2", alpha=0.55, lw=0, label="FRB halo DM weight (z_s=1, 1e13-1e14, R200c)")
ax = axes[0]
for l, ls in ((1000, "--"), (3000, "-")):
    ax.plot(z, cv["validation_lee22_literal_{}_dz".format(l)], color=BLUE, ls=ls, label="Lee22 (literal = xgpnative shape), l={}".format(l))
    ax.plot(z, cv["validation_b16_{}_dz".format(l)], color=ORANGE, ls=ls, label="Battaglia16, l={}".format(l))
    ax.plot(z, cv["validation_lee22_tngcorr3_{}_dz".format(l)], color=AQUA, ls=ls,
            label="Lee22 x {:.3f}(1+z)^3 (TNG-calibrated), l={}".format(c3, l))
ax.set_title("kSZ power per unit z (normalized), 3R200c spheres, painted halo range", fontsize=11)
ax = axes[1]
for l, ls in ((300, ":"), (3000, "-"), (8000, "--")):
    ax.plot(z, cv["validation_lee22p_{}_dz".format(l)], color=BLUE, ls=ls, label="Lee22 pressure, l={}".format(l))
    ax.plot(z, cv["validation_b12_{}_dz".format(l)], color=ORANGE, ls=ls, label="Battaglia12, l={}".format(l))
ax.set_title("tSZ power per unit z (normalized), 4R200c aperture, painted halo range", fontsize=11)
for ax in axes:
    ax.set_xscale("log"); ax.set_xlim(0.01, 4); ax.set_ylim(0, None)
    ax.set_xlabel("halo redshift z"); ax.set_ylabel("(1/C_l) dC_l/dz  or  (1/<DM>) d<DM>/dz")
    ax.legend(fontsize=8, loc="upper right")
save(fig, "sz_redshift_budget")

# ------------------------------------------------------------------ 2. TNG z_s test
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
zs, tm, te = cv["tng_zs"], cv["tng_mean"], cv["tng_err"]
keep = zs <= 2.5 + 1e-9
ax.errorbar(zs[keep], tm[keep], yerr=te[keep], fmt="o", ms=5, color=INK, label="TNG300 (Konietzka), within R200, 1e13-1e14")
zg = cv["hm_zgrid"]
ax.plot(zg, cv["hm_cum_literal"], color=BLUE, ls=":", label="Lee22 literal (halo model)")
ax.plot(zg, cv["hm_cum_literal"] * A.XGP, color=BLUE, label="Lee22 x 0.6016 (xgpnative)")
key_k = [k for k in cv if k.startswith("hm_cum_TNG_z_s_fit") and k.endswith("_k")][0]
key_3 = [k for k in cv if k.startswith("hm_cum_TNG_z_s_fit") and k.endswith("_3")][0]
ax.plot(zg, cv[key_k], color=AQUA, label="Lee22 x {:.3f}(1+z)^{:.2f} (fit)".format(*kfit))
ax.plot(zg, cv[key_3], color=YELLOW, ls="--", label="Lee22 x {:.3f}(1+z)^3 (fit)".format(c3))
ax.axvspan(2.0, 2.5, color="#efeeea", lw=0)
ax.set_xlabel("source redshift z_s"); ax.set_ylabel("mean halo DM over all rays [pc cm^-3]")
ax.set_xlim(0, 2.5); ax.set_ylim(0, None); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Halo DM vs source redshift: TNG against Lee22 normalizations", fontsize=11)
ax = axes[1]
ax.plot(cv["C_eff_z"], cv["C_eff"], "o", ms=5, color=INK, label="TNG / Lee22-literal, per dz_s = 0.1 step")
zz = np.linspace(0, 2.5, 200)
ax.plot(zz, C_tng(zz), color=AQUA, label="{:.3f}(1+z)^{:.2f}".format(*kfit))
ax.plot(zz, c3 * (1 + zz) ** 3, color=YELLOW, ls="--", label="{:.3f}(1+z)^3".format(c3))
ax.axhline(A.XGP, color=BLUE, label="0.6016 (xgpnative)")
ax.axhline(1.0, color=BLUE, ls=":", label="1 (literal)")
ax.set_yscale("log"); ax.set_xlim(0, 2.5); ax.set_ylim(0.2, 30)
ax.axvspan(2.0, 2.5, color="#efeeea", lw=0)
ax.set_xlabel("halo redshift z"); ax.set_ylabel("effective density correction C(z)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Correction TNG requires: not constant", fontsize=11)
save(fig, "tng_zs_test_Ceff")

# ------------------------------------------------------------------ 3. spectra
d = A.load("validation")
sp = {"lit": A.spectra(d, "tau_lee22_literal", "ksz"), "b16": A.spectra(d, "tau_b16", "ksz"),
      "lp": A.spectra(d, "y_lee22p", "tsz"), "b12": A.spectra(d, "y_b12", "tsz")}
ell = d["ell"]
fac = ell * (ell + 1) / (2 * np.pi)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
ax = axes[0]
ax.plot(ell, fac * A.cl_total(sp["lit"]), color=BLUE, ls=":", label="Lee22 literal (paper eq. 9)")
ax.plot(ell, fac * A.cl_total(sp["lit"]) * A.XGP ** 2, color=BLUE, label="Lee22 xgpnative (x0.6016)")
ax.plot(ell, fac * A.cl_total(sp["lit"], C_tng), color=AQUA, label="Lee22 x {:.3f}(1+z)^{:.2f} (TNG z_s fit)".format(*kfit))
ax.plot(ell, fac * A.cl_total(sp["b16"]), color=ORANGE, label="Battaglia16")
for l, p in A.PAINTED_KSZ.items():
    ax.plot([l], [p["lee22_literal"]], "s", color=BLUE, mfc="white", ms=8)
    ax.plot([l], [p["lee22_xgpnative"]], "s", color=BLUE, ms=8)
    ax.plot([l], [p["b16"]], "s", color=ORANGE, ms=8)
ax.plot([], [], "s", color=INK2, ms=8, label="painted HalfDome maps (squares)")
ax.set_xscale("log"); ax.set_xlim(100, 10000); ax.set_ylim(0, None)
ax.set_xlabel("multipole l"); ax.set_ylabel("kSZ D_l [uK^2]")
ax.set_title("kSZ halo model (1h+2h), painted halo range, 3R200c", fontsize=11)
ax.legend(fontsize=8)
ax = axes[1]
lp = [row for row in res["pressure_corrections"]]
bb = fac * A.cl_total(sp["b12"])
ax.plot(ell, fac * A.cl_total(sp["lp"]) / bb, color=BLUE, label="Lee22 pressure, literal")
ax.plot(ell, fac * A.cl_total(sp["lp"], lambda z: (1 + z) ** 2) / bb, color=AQUA, label="Lee22 x (1+z)^2")
ax.plot(ell, fac * A.cl_total(sp["lp"], lambda z: 0.6774 ** 2 * (1 + z) ** 2) / bb, color=AQUA, ls="--", label="Lee22 x h^2 (1+z)^2")
ax.plot(ell, fac * A.cl_total(sp["lp"], lambda z: (1 + z) ** 3) / bb, color=YELLOW, label="Lee22 x (1+z)^3")
pr = res["validation_vs_painted"]
pl = [30, 100, 300, 1000, 2000, 3000, 5000, 8000]
ax.plot(pl, [pr["tsz_{}".format(l)]["painted_ratio"] for l in pl], "s", color=BLUE, ms=7, label="painted maps, literal")
cw = res["validation_catalogue_weighted"]
ax.plot(pl, [cw["tsz_ratio_lee22p_over_b12_{}".format(l)] for l in pl], "o", color=BLUE, mfc="white", ms=7,
        label="catalogue-weighted halo sum, literal")
ax.axhline(1, color=INK2, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(20, 10000)
ax.set_xlabel("multipole l"); ax.set_ylabel("C_l^yy / C_l^yy(Battaglia12)")
ax.set_title("tSZ ratio to Battaglia12, painted halo range, 4R200c aperture", fontsize=11)
ax.legend(fontsize=8)
save(fig, "sz_spectra_corrections")

# ------------------------------------------------------------------ 4. mass budgets
lnM = cv["validation_lnM"] / np.log(10)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
for l, ls in ((1000, "--"), (3000, "-")):
    axes[0].plot(lnM, cv["validation_lee22_literal_{}_dlnM".format(l)], color=BLUE, ls=ls, label="Lee22, l={}".format(l))
    axes[0].plot(lnM, cv["validation_b16_{}_dlnM".format(l)], color=ORANGE, ls=ls, label="Battaglia16, l={}".format(l))
for l, ls in ((300, ":"), (3000, "-")):
    axes[1].plot(lnM, cv["validation_lee22p_{}_dlnM".format(l)], color=BLUE, ls=ls, label="Lee22 pressure, l={}".format(l))
    axes[1].plot(lnM, cv["validation_b12_{}_dlnM".format(l)], color=ORANGE, ls=ls, label="Battaglia12, l={}".format(l))
for ax, t in zip(axes, ("kSZ 1-halo", "tSZ 1-halo")):
    ax.axvspan(13 - np.log10(0.68), 14.8 - np.log10(0.68), color="#efeeea", lw=0, label="Lee22 fit mass range")
    ax.set_xlabel("log10 M200c [Msun]"); ax.set_ylabel("(1/C_l^1h) dC_l^1h/dlnM"); ax.set_title(t + " power per ln M (normalized)", fontsize=11)
    ax.legend(fontsize=8); ax.set_ylim(0, None)
save(fig, "sz_mass_budget")
print("figures written")
