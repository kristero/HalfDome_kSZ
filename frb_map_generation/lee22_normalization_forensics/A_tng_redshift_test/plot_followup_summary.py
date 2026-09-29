#!/usr/bin/env python3
"""Summary figure for test A (reads the CSVs written by followup_checks.py and per_crossing_dm_by_slice.py)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "frb_map_generation" / "outputs" / "lee22_normalization_forensics_20260928" / "A_tng_redshift_test"
# reference palette slots 1-4 in fixed order; TNG in ink, references in gray
LIT, B16, CORR, FB = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
H = 0.6774
XGP = 0.6016316602183194
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "axes.titlesize": 9.5,
                     "axes.titleweight": "bold", "axes.titlelocation": "left"})


def load(name):
    with open(OUT / name) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


sl = load("followup_pseudo_bin_slices.csv")
ft = load("followup_pseudo_bin_fits.csv")
cu = load("followup_cumulative_ratios_vs_zs.csv")
bu = load("followup_zs1_budget_13to14.csv")
pc = load("per_crossing_dm_by_slice_mass13to14.csv")
mq = load("followup_macquart_check.csv")


def fit(bin_, model, q="C"):
    return [r for r in ft if r["bin"] == bin_ and r["model"] == model and r["quantity"] == q][0]


fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6), gridspec_kw=dict(hspace=0.36, wspace=0.30))
zz = np.linspace(0.0, 2.05, 200)

# ---------------------------------------------------------------- (a), (b): C(z) per halo-redshift slice
for ax, bin_, title in ((axes[0, 0], "13-14", "(a) TNG/model per halo-z slice, 1e13-1e14 Msun"),
                        (axes[0, 1], "12-14", "(b) same, 1e12-1e14 Msun")):
    for model, color, label, dx in (("lee22_literal", LIT, "Lee22 no-c, literal eq. 9", -0.012),
                                    ("b16", B16, "Battaglia16 (XGPaint)", 0.0),
                                    ("fb_full", FB, "f_b M200c fully ionized", 0.012)):
        rows = [r for r in sl if r["bin"] == bin_ and r["model"] == model and r["z_hi"] <= 2.0 + 1e-9]
        z = np.array([r["z_snap_weighted"] for r in rows]) + dx
        C = np.array([r["C"] for r in rows])
        e = np.array([r["C_err"] for r in rows])
        infit = np.array([r["z_lo"] >= 0.1 - 1e-9 for r in rows])
        ax.errorbar(z[infit], C[infit], e[infit], fmt="o", ms=5, color=color, mec="white", mew=0.9, elinewidth=1.2,
                    capsize=0, zorder=3)
        ax.errorbar(z[~infit], C[~infit], e[~infit], fmt="o", ms=5, mfc="white", color=color, elinewidth=1.2,
                    capsize=0, zorder=3)
        f = fit(bin_, model)
        ax.plot(zz, f["C0"] * (1 + zz) ** f["k"], color=color, lw=2,
                label="{}: C0 = {:.3f}, k = {:+.2f}".format(label, f["C0"], f["k"]))
    ax.axhline(XGP, color=GRAY, lw=1.6, ls="--", label="constant 0.602 (XGPaint-native Lee22)")
    ax.plot(zz, H ** 3 * (1 + zz) ** 3, color=INK2, lw=1.3, ls=":", label="h^3 (1+z)^3 (comoving kpc/h volumes)")
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 12)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(-0.03, 2.05)
    ax.set_xlabel("halo redshift (snapshot redshift, DM-weighted per slice)")
    ax.set_ylabel("C = <dDM>_TNG / <dDM>_model")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7.8)
axes[0, 0].text(1.98, 0.2, "open symbols: 0 < z < 0.1, not in the fit\nerrors: bootstrap over 120k rays (stat. only)",
                ha="right", va="bottom", fontsize=7.3, color=INK2)

# ---------------------------------------------------------------- (c): cumulative ratio vs source redshift
ax = axes[0, 2]
for bin_, mk, ls in (("13-14", "o", "-"), ("12-14", "s", "--")):
    rows = [r for r in cu if r["bin"] == bin_ and r["z_s"] <= 3.0 + 1e-9]
    zsv = np.array([r["z_s"] for r in rows])
    for key, color, lab in (("ratio_tng_over_lee22_xgpnative", LIT, "Lee22 x 0.602 (XGPaint-native)"),
                            ("ratio_tng_over_lee22_literal", CORR, "Lee22 literal"),
                            ("ratio_tng_over_b16", B16, "Battaglia16")):
        ax.plot(zsv, [r[key] for r in rows], ls=ls, marker=mk, ms=4.5, color=color, lw=1.8, mec="white", mew=0.7,
                label="{} ({})".format(lab, bin_))
ax.axhline(1.0, color=INK2, lw=0.9)
ax.axvline(1.0, color=GRAY, lw=1, ls=":")
ax.set_xlabel("source redshift z_s")
ax.set_ylabel("<DM>_TNG / <DM>_model  (all rays, cumulative)")
ax.set_title("(c) Cumulative TNG/model vs source z")
ax.set_ylim(0.12, 2.0)
ax.set_yscale("log")
ax.set_yticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0]); ax.set_yticklabels(["0.3", "0.5", "0.7", "1.0", "1.5", "2.0"])
ax.text(1.03, 1.9, "z_s = 1", color=INK2, fontsize=7.5, va="top")
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(loc="lower right", fontsize=7.0, ncol=1)

# ---------------------------------------------------------------- (d): where the z_s = 1 halo DM comes from
ax = axes[1, 0]
edges = np.array([bu[0]["z_lo"]] + [r["z_hi"] for r in bu])
for key, color, ls, lab in (("lee22_literal", LIT, "-", "Lee22 literal"), ("lee22_xgpnative", LIT, "--", "Lee22 x 0.602"),
                            ("lee22_corrected_free", CORR, "-", "Lee22 literal x fitted C(z)"),
                            ("b16", B16, "-", "Battaglia16")):
    v = np.array([r[key + "_dDM"] for r in bu]) / np.diff(edges)
    ax.stairs(v, edges, color=color, lw=2, ls=ls, label="{} (sum {:.1f})".format(lab, np.sum(v * np.diff(edges))))
tv = np.array([r["TNG_dDM"] for r in bu]) / np.diff(edges)
ax.plot(0.5 * (edges[1:] + edges[:-1]), tv, "o", color=INK, ms=6, mec="white", mew=0.9,
        label="TNG, Konietzka within R200 (sum {:.1f})".format(np.sum(tv * np.diff(edges))))
ax.set_xlabel("halo redshift")
ax.set_ylabel("d<DM>/dz  [pc cm^-3], all rays, z_s = 1")
ax.set_title("(d) z_s = 1 halo-DM budget, 1e13-1e14 Msun")
ax.set_xlim(0, 1)
ax.legend(fontsize=7.6)

# ---------------------------------------------------------------- (e): DM per crossing by halo redshift
ax = axes[1, 1]
zc = np.array([0.5 * (r["z_lo"] + r["z_hi"]) for r in pc])
for key, color, ls, mk, lab, dx in (("tng", INK, "-", "o", "TNG ray increments", 0.0),
                                    ("lee22_xgpnative", LIT, "--", "s", "Lee22 x 0.602", -0.012),
                                    ("lee22_corrected", CORR, "-", "D", "Lee22 literal x fitted C(z)", 0.012),
                                    ("b16", B16, "-", "^", "Battaglia16", 0.024)):
    med = np.array([r[key + "_median"] for r in pc])
    lo = np.array([r[key + "_p16"] for r in pc]); hi = np.array([r[key + "_p84"] for r in pc])
    ax.errorbar(zc + dx, med, [med - lo, hi - med], color=color, ls=ls, marker=mk, ms=5, mec="white", mew=0.8,
                lw=1.8, elinewidth=0.9, capsize=2, label=lab)
ax.set_yscale("log")
ax.set_xlabel("halo redshift (slice centre)")
ax.set_ylabel("DM per halo crossing [pc cm^-3]  (median, 16-84 %)")
ax.set_title("(e) DM per crossing, 1e13-1e14 Msun")
ax.legend(fontsize=7.6, loc="lower left")

# ---------------------------------------------------------------- (f): Macquart check of the TNG electron field
ax = axes[1, 2]
rows = [r for r in mq if r["z_s"] <= 5.5]
zsv = np.array([r["z_s"] for r in rows])
ax.plot(zsv, [r["fd_cumulative"] for r in rows], "-o", color=INK, ms=3.5, lw=1.6,
        label="cumulative: <DM_LSS>_TNG / all baryons ionized")
ax.plot(zsv, [r["fd_slice"] for r in rows], "o", color=GRAY, ms=3.5, label="per source-redshift slice")
ax.plot(zsv, [0.955 / (1 + z) ** 3 * 1 for z in zsv], color=LIT, lw=1.4, ls=":",
        label="what a spurious (1+z)^-3 would look like")
ax.set_ylim(0.1, 1.2)
ax.set_xlabel("source redshift z_s")
ax.set_ylabel("f_d = TNG / (n_e0 c/H0 int (1+z)/E dz)")
ax.set_title("(f) TNG total DM vs Macquart (f_d)")
ax.legend(fontsize=7.6, loc="center right")

fig.suptitle("Lee22 density normalization vs IllustrisTNG's own electron column (Konietzka TNG300 rays, "
             "Tinker08/Planck15 halo model, gas inside R200c)", x=0.06, ha="left", fontsize=11, fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(OUT / "lee22_redshift_test_summary.{}".format(ext), dpi=150, bbox_inches="tight")
print("wrote", OUT / "lee22_redshift_test_summary.png")
