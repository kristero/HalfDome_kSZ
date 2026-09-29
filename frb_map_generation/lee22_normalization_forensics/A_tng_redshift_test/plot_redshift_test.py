#!/usr/bin/env python3
"""Figure for test A: TNG / model electron-content ratio per halo-redshift slice, fits, abundance check, and the
redshift distribution of the halo DM at z_s = 1."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "frb_map_generation" / "outputs" / "lee22_normalization_forensics_20260928" / "A_tng_redshift_test"
MODE = sys.argv[1] if len(sys.argv) > 1 else "snapshot"

BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "legend.frameon": False})


def load(name):
    with open(OUT / name) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


sl = load("tng_vs_model_slices.csv")
xs = load("extra_checks_slices.csv")
xf = load("extra_checks_fits.csv")
fits = load("powerlaw_fits.csv")
cum = load("tng_vs_model_cumulative.csv")


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def fit_of(window, model, mode=MODE, zr="0.1-2.0"):
    return sel(fits, window=window, model=model, mode=mode, z_range=zr)[0]


fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2), gridspec_kw=dict(height_ratios=[1, 0.92], hspace=0.62))
zz = np.linspace(0.0, 2.05, 200)
for ax, window, title in ((axes[0, 0], "mass13to14", "M200c = 1e13 - 1e14 Msun"),
                          (axes[0, 1], "mass12to14", "M200c = 1e12 - 1e14 Msun")):
    rows = [r for r in sel(sl, window=window, mode=MODE) if r["z_hi"] <= 2.0 + 1e-9]
    for model, color, label, dx in (("lee22_literal", BLUE, "Lee22 no-c literal (eq. 9)", -0.012),
                                    ("b16", ORANGE, "Battaglia16 (XGPaint)", 0.0),
                                    ("fb_full", AQUA, "f_b M200c fully ionized (TNG e- fraction)", 0.012)):
        z = np.array([r["zmean_dm_weighted_" + model] for r in rows])
        C = np.array([r["C_" + model] for r in rows])
        e = np.array([r["C_err_" + model] for r in rows])
        infit = np.array([r["z_lo"] >= 0.1 - 1e-9 for r in rows])
        ax.errorbar(z[infit] + dx, C[infit], e[infit], fmt="o", ms=4.5, color=color, mec="white", mew=0.8,
                    elinewidth=1.2, capsize=0, zorder=3)
        ax.errorbar(z[~infit] + dx, C[~infit], e[~infit], fmt="o", ms=4.5, mfc="white", color=color, elinewidth=1.2,
                    capsize=0, zorder=3)
        f = fit_of(window, model)
        ax.plot(zz, f["C0"] * (1 + zz) ** f["k"], color=color, lw=2,
                label="{}: C0 = {:.3f} +- {:.3f}, k = {:.2f} +- {:.2f}".format(label, f["C0"], f["C0_err"], f["k"], f["k_err"]))
    # Lee22 literal x (1+z)^3: comoving-volume n_e normalized by the physical n200(z) (extra_checks.py, snapshot mode)
    if MODE == "snapshot":
        xr = [r for r in xs if r["check"] == "lee22_literal_x_(1+z)^3" and r["window"] == window and r["z_hi"] <= 2.0 + 1e-9]
        zc3 = np.array([float(r["note"].split("zw ")[1]) for r in xr])
        C3 = np.array([r["value"] for r in xr])
        e3 = np.array([float(r["note"].split("err ")[1].split(";")[0]) for r in xr])
        ax.errorbar(zc3 + 0.024, C3, e3, fmt="D", ms=4, color=VIOLET, mec="white", mew=0.8, elinewidth=1.2, capsize=0,
                    zorder=3)
        f3 = [r for r in xf if r["tag"] == "lee22_literal_x_(1+z)^3" and r["window"] == window][0]
        ax.plot(zz, f3["C0"] * (1 + zz) ** f3["k"], color=VIOLET, lw=2,
                label="Lee22 literal x (1+z)^3: C0 = {:.3f} +- {:.3f}, k = {:.2f} +- {:.2f}".format(
                    f3["C0"], f3["C0_err"], f3["k"], f3["k_err"]))
        for val, lab in ((0.6774 ** 3, "h^3"), (0.6774 ** 2 * 0.76 ** 2, "h^2 X_H^2")):
            ax.axhline(val, color=VIOLET, lw=0.7, ls=":")
            ax.text(2.04, val, lab, color=INK2, fontsize=7, ha="right", va="bottom")
    ax.axhline(0.6016, color=GRAY, lw=1.5, ls="--", label="constant 0.602 (XGPaint-native Lee22)")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.12, 7)
    ax.set_xlim(0, 2.05)
    ax.set_xlabel("halo redshift (DM-weighted mean of the slice)")
    ax.set_ylabel("C(z) = <dDM>_TNG / <dDM>_model per slice")
    ax.set_title("{}  [{} redshift assignment]".format(title, MODE), fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.4, loc="upper left", bbox_to_anchor=(-0.02, -0.14), ncol=1)

# abundance x geometry check: slice Nbar ratio
ax = axes[1, 0]
for window, color, lab in (("mass13to14", BLUE, "1e13-1e14"), ("mass12to14", VIOLET, "1e12-1e14")):
    for mode, ls, mk in (("snapshot", "-", "o"), ("continuous", ":", "s")):
        rows = [r for r in sel(sl, window=window, mode=mode) if r["z_hi"] <= 2.0 + 1e-9]
        z = np.array([0.5 * (r["z_lo"] + r["z_hi"]) for r in rows])
        rat = np.array([r["nbar_ratio_tng_over_model"] for r in rows])
        p = np.array([r["tng_slice_hit"] for r in rows]); ep = np.array([r["tng_slice_hit_err"] for r in rows])
        err = rat * ep / ((1 - p) * -np.log1p(-p))
        ax.errorbar(z, rat, err, fmt=mk + ls, color=color, ms=4, lw=1.4, mfc=color if mode == "snapshot" else "white",
                    label="{} ({})".format(lab, mode))
ax.axhline(1, color=INK2, lw=0.8)
for zb in (0.6, 0.85, 1.25, 1.75):
    ax.axvline(zb, color=GRAY, lw=0.6, ls="--")
ax.set_xlim(0, 2.05)
ax.set_xlabel("slice mid redshift (dashed: snapshot segment boundaries)")
ax.set_ylabel("halo crossings per ray, TNG / Tinker08 model")
ax.set_title("Abundance x area check (slice hit fractions)", fontsize=10, color=INK, loc="left")
ax.legend(fontsize=7.6, loc="lower left", ncol=2)

# redshift distribution of the halo DM at z_s = 1, 1e13-1e14
ax = axes[1, 1]
rows = [r for r in sel(sl, window="mass13to14", mode=MODE) if r["z_hi"] <= 1.0 + 1e-9]
zc = np.array([0.5 * (r["z_lo"] + r["z_hi"]) for r in rows])
dz = np.array([r["z_hi"] - r["z_lo"] for r in rows])
t = np.array([r["tng_dDM"] for r in rows]); te = np.array([r["tng_dDM_err"] for r in rows])
ax.errorbar(zc, t / dz, te / dz, fmt="o", color=INK, ms=5, label="TNG (Konietzka, within R200)", zorder=4)
lit = np.array([r["model_dDM_lee22_literal"] for r in rows])
f = fit_of("mass13to14", "lee22_literal")
zw = np.array([r["zmean_dm_weighted_lee22_literal"] for r in rows])
corr = lit * f["C0"] * (1 + zw) ** f["k"]
for arr, color, lab, ls in ((lit, BLUE, "Lee22 literal", "-"), (0.6016316602 * lit, GRAY, "Lee22 x 0.602 (XGPaint-native)", "--"),
                            (corr, AQUA, "Lee22 literal x fitted C(z)", "-"),
                            (np.array([r["model_dDM_b16"] for r in rows]), ORANGE, "Battaglia16", "-")):
    ax.step(np.append(zc - dz / 2, 1.0), np.append(arr / dz, arr[-1] / dz[-1]), where="post", color=color, lw=2, ls=ls,
            label="{} (sum {:.1f})".format(lab, arr.sum()))
ax.set_xlim(0, 1.0)
ax.set_xlabel("halo redshift")
ax.set_ylabel("d<DM>/dz  [pc cm$^{-3}$ per unit z], all rays")
ax.set_title("Where the z_s = 1 halo DM comes from (1e13-1e14; TNG sum {:.1f})".format(t.sum()), fontsize=10,
             color=INK, loc="left")
ax.legend(fontsize=7.6, loc="upper right")
fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.06, wspace=0.22)
for ext in ("png", "pdf"):
    fig.savefig(OUT / "lee22_redshift_test_{}.{}".format(MODE, ext), dpi=160)
print("wrote", OUT / "lee22_redshift_test_{}.png".format(MODE))
