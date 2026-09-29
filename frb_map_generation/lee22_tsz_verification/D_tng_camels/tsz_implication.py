#!/usr/bin/env python
"""Implication of the CAMELS result (TNG-model P_e/P200 = printed Lee22 x (1+z)^2 in the calibrated mass range) for
the painted Lee22 tSZ spectrum, using the existing halo-model tables of the forensics budget (F_sz_budget,
run 'validation' = painted HalfDome halo range 6.56e12-3.79e15 Msun, z = 0.002-3.86, 4 R200c full-LOS aperture,
Tinker08 HMF, HalfDome painting cosmology).  Since the correction is a pure amplitude factor per halo,
y_l -> (1+z)^2 y_l and both the 1-halo and 2-halo terms scale with (1+z)^4 inside the z integral.

Variants: Lee22 printed (= painted), x (1+z)^2 at all z, x (1+z)^2 only up to z = 2 (held at 9 above; the Lee22
fits were calibrated to z <= 2), the CAMELS 13.5 < log M200c/Msun < 14 amplitude fit c0 (1+z)^k (read from
tables/redshift_trend_powerlaw_fits.csv), and Battaglia12 (P_e = 0.5176 P_th).
Mass decomposition: 1-halo term restricted to M200c <= Mmax (1e14, 3e14, 10^14.8/0.6774 Msun = Lee22 upper fit limit,
all), corrected Lee22 / B12 and printed Lee22 / B12, to show where the corrected-Lee22 excess over B12 comes from.
Output: tsz_implication.csv, tsz_implication_mass_cuts.csv, fig_D8_tsz_implication.png (D_tng_camels output dir).
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python tsz_implication.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/F_sz_budget")
import szhm as S  # noqa: E402
import camels_common as cc  # noqa: E402

TABLES = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget/tables_validation.npz"
trapz = S._trapz
d = dict(np.load(TABLES))
z, M, ell = d["z"], d["M"], d["ell"]
lnM = np.log(M)
dVdz = d["chi"] ** 2 / d["Hc"]
k = (ell[None, :] + 0.5) / d["chi"][:, None]
Plin = d["D"][:, None] ** 2 * np.exp(np.interp(np.log(k), np.log(d["k_lin"]), np.log(d["p_lin0"])))


def cl(key, corr=None):
    X = d[key]
    w = np.ones_like(z) if corr is None else corr(z)
    Xc = X * w[:, None, None]
    dC1 = trapz(dVdz[:, None, None] * d["dn"][:, :, None] * Xc ** 2, lnM, axis=1)
    Ib = trapz((d["dn"] * d["bias"])[:, :, None] * Xc, lnM, axis=1)
    dC2 = dVdz[:, None] * Ib ** 2 * Plin
    return trapz(dC1 + dC2, z, axis=0), trapz(dC1, z, axis=0), dC1


FITS = os.path.join(cc.OUTDIR, "tables", "redshift_trend_powerlaw_fits.csv")
_f = np.genfromtxt(FITS, delimiter=",", names=True, dtype=None, encoding=None)
_row = _f[(_f["quantity"] == "P_e/P200") & (np.isclose(_f["logM_lo"], 13.5)) & (_f["x"] == "amp(0.04-1.34)")][0]
K_CAM, C_CAM = float(_row["k_exponent"]), float(_row["ratio_at_z0_fit"])
CAM_NAME = f"Lee22 x {C_CAM:.2f} (1+z)^{K_CAM:.2f} (CAMELS 13.5-14)"

variants = {
    "Lee22 printed (= painted)": ("y_lee22p", None, cc.COLORS["lee_print"], "-"),
    "Lee22 x (1+z)^2 (M1, all z)": ("y_lee22p", lambda zz: (1 + zz) ** 2, cc.COLORS["lee_corr"], "-"),
    "Lee22 x (1+min(z,2))^2": ("y_lee22p", lambda zz: (1 + np.minimum(zz, 2.0)) ** 2, cc.COLORS["lee_corr"], "--"),
    CAM_NAME: ("y_lee22p", lambda zz: C_CAM * (1 + zz) ** K_CAM, cc.COLORS["other"], "-"),
    "Battaglia12 x 0.5176": ("y_b12", None, cc.COLORS["b12"], "-"),
}
res = {name: cl(key, corr) for name, (key, corr, c, ls) in variants.items()}
base = res["Lee22 printed (= painted)"][0]
b12 = res["Battaglia12 x 0.5176"][0]
L = [300, 1000, 2000, 3000, 5000]
rows = ["variant,ell,D_ell_1e12,ratio_to_printed,ratio_to_B12,median_z_1h"]
lines = []
for name, (tot, one, dC1) in res.items():
    parts = []
    for l in L:
        i = int(np.argmin(np.abs(ell - l)))
        Dl = ell[i] * (ell[i] + 1) / (2 * np.pi) * tot[i] * 1e12
        cum = np.concatenate([[0], np.cumsum(0.5 * (dC1[1:, i] + dC1[:-1, i]) * np.diff(z))])
        zmed = np.interp(0.5, cum / cum[-1], z)
        rows.append(f"{name},{int(ell[i])},{Dl:.4f},{tot[i] / base[i]:.4f},{tot[i] / b12[i]:.4f},{zmed:.3f}")
        parts.append(f"l={int(ell[i])}: D={Dl:.3f}e-12, /printed {tot[i] / base[i]:.2f}, /B12 {tot[i] / b12[i]:.2f}, z_med(1h) {zmed:.2f}")
    lines.append(f"{name}: " + "; ".join(parts))
out = cc.OUTDIR
with open(os.path.join(out, "tsz_implication.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- mass decomposition of the 1-halo term
def c1_cut(key, w, mmax):
    X = d[key] * w[:, None, None] * (M <= mmax)[None, :, None]
    return trapz(dVdz[:, None] * trapz(d["dn"][:, :, None] * X ** 2, lnM, axis=1), z, axis=0)


one = np.ones_like(z)
w2 = (1 + z) ** 2
mrows = ["Mmax_Msun,ell,C1h_lee22_x(1+z)^2_over_B12,C1h_lee22_printed_over_B12,C1h_B12_fraction_of_total_B12_1h"]
b12_all = c1_cut("y_b12", one, np.inf)
mlines = []
for mmax in (1e14, 3e14, 10 ** 14.8 / 0.6774, np.inf):
    a_ = c1_cut("y_lee22p", w2, mmax)
    p_ = c1_cut("y_lee22p", one, mmax)
    b_ = c1_cut("y_b12", one, mmax)
    parts = []
    for l in (300, 1000, 3000):
        i = int(np.argmin(np.abs(ell - l)))
        mrows.append(f"{mmax:.4e},{int(ell[i])},{a_[i] / b_[i]:.4f},{p_[i] / b_[i]:.4f},{b_[i] / b12_all[i]:.4f}")
        parts.append(f"l={int(ell[i])}: corr/B12 {a_[i] / b_[i]:.2f}, printed/B12 {p_[i] / b_[i]:.2f} (B12 1h share {b_[i] / b12_all[i]:.2f})")
    mlines.append(f"M200c <= {mmax:.3g} Msun: " + "; ".join(parts))
with open(os.path.join(out, "tsz_implication_mass_cuts.csv"), "w") as f:
    f.write("\n".join(mrows) + "\n")
print("\n1-halo mass decomposition:\n" + "\n".join(mlines))


# ---------------------------------------------------------------- figure
MCUTS = ((1e14, ":", r"$M_{200c}\leq10^{14}$"), (3e14, "-.", r"$\leq3\times10^{14}$"),
         (10 ** 14.8 / 0.6774, "--", r"$\leq10^{14.8}h^{-1}$ (Lee22 fit limit)"), (np.inf, "-", "all painted halos"))
fig, axs = plt.subplots(1, 3, figsize=(17, 5.0), constrained_layout=True)
for name, (key, corr, c, ls) in variants.items():
    tot = res[name][0]
    axs[0].loglog(ell, ell * (ell + 1) / (2 * np.pi) * tot * 1e12, color=c, ls=ls, label=name)
    axs[1].semilogx(ell, tot / b12, color=c, ls=ls, label=name)
axs[0].set_xlabel(r"$\ell$")
axs[0].set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$ (halo model, 1h+2h)")
axs[0].set_xlim(30, 1e4)
axs[0].legend(fontsize=8, loc="lower right")
axs[0].set_title("halo-model tSZ auto spectrum", fontsize=10)
axs[1].set_xlabel(r"$\ell$")
axs[1].set_ylabel(r"ratio to Battaglia12 (1h+2h)")
axs[1].set_xlim(30, 1e4)
axs[1].set_ylim(0, 4.2)
axs[1].axhline(1, color="0.5", lw=0.8)
axs[1].set_title("ratio to Battaglia12", fontsize=10)
for mmax, ls, lab in MCUTS:
    a_ = c1_cut("y_lee22p", w2, mmax)
    p_ = c1_cut("y_lee22p", one, mmax)
    b_ = c1_cut("y_b12", one, mmax)
    axs[2].semilogx(ell, a_ / b_, color=cc.COLORS["lee_corr"], ls=ls, label=f"Lee22 x (1+z)^2, {lab}")
    axs[2].semilogx(ell, p_ / b_, color=cc.COLORS["lee_print"], ls=ls, label=f"Lee22 printed, {lab}")
axs[2].axhline(1, color="0.5", lw=0.8)
axs[2].set_xlim(30, 1e4)
axs[2].set_ylim(0, 6.3)
axs[2].set_xlabel(r"$\ell$")
axs[2].set_ylabel(r"$C_\ell^{1h}(M_{200c}\leq M_{max})$ / same for Battaglia12")
axs[2].set_title("1-halo term by upper mass cut (Msun)", fontsize=10)
axs[2].legend(fontsize=7, ncol=2, loc="upper center")
fig.suptitle("Halo-model estimate of the painted Lee22 tSZ with the CAMELS-validated $(1+z)^2$ pressure correction "
             "(F_sz_budget tables: HalfDome halo range, Tinker08, 4 R200c full-LOS aperture;\nLee22 amplitude extrapolated "
             "above $10^{14.8}h^{-1}M_\odot$ with the printed broken power law, CAMELS tests only $M_{200c}<10^{14}M_\odot$)",
             fontsize=10.5)
fig.savefig(os.path.join(out, "fig_D8_tsz_implication.png"), dpi=120)
