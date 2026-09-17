#!/usr/bin/env python3
"""Projected electron-column profiles: Battaglia16 gas-density fit versus the Lee22 no-concentration
fit (XGPaint-native reading), on a mass x redshift grid, with the Lee22 calibration ranges marked.

Input: outputs/projected_profiles_20260917/projected_profiles.csv (compute_projected_profile_grid.jl).
Lee22 calibration (Lee et al. 2022, MNRAS 517, 420): 27 radial bins over 0.04-1.34 R200,
halo masses 1e13-10^14.8 h^-1 Msun (h = 0.6774 in TNG; 0.68 here), 20 TNG300 snapshots to z = 2.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = 0.68
LEE_RADII = (0.04, 1.34)
LEE_MASS = (1e13 / H, 10 ** 14.8 / H)
LEE_ZMAX = 2.0
BLUE, ORANGE, RED = "#0072B2", "#D55E00", "#B2182B"
RC = {"font.size": 14, "axes.labelsize": 17, "axes.titlesize": 15, "xtick.labelsize": 13,
      "ytick.labelsize": 13, "legend.fontsize": 14, "axes.linewidth": 1.0}


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:  # Lee22 Figure-5 scaling: dimensionless density weighted by the shell volume
        r["ne_over_n200_x3"] = str(float(r["ne_3d_cm3"]) / float(r["n200_eq9_cm3"]) * float(r["impact_r200c"]) ** 3)
    data = {}
    for r in rows:
        key = (r["model"], float(r["mass_msun"]), float(r["redshift"]))
        data.setdefault(key, []).append(r)
    masses = sorted({float(r["mass_msun"]) for r in rows})
    redshifts = sorted({float(r["redshift"]) for r in rows})
    return data, masses, redshifts


def arrays(rows, key):
    return np.array([float(r[key]) for r in rows])


def mass_label(m):
    e = int(np.floor(np.log10(m)))
    a = m / 10 ** e
    return (r"$10^{%d}\,M_\odot$" % e) if abs(a - 1) < 1e-6 else (r"$%.1f\times10^{%d}\,M_\odot$" % (a, e))


def mass_note(m):
    if m < LEE_MASS[0]:
        return "below Lee22 fit range: non calibr."
    if m > LEE_MASS[1]:
        return "above Lee22 fit range: non calibr."
    return "inside Lee22 fit range"


def plot_grid(data, masses, redshifts, out, quantity, ylabel, stem, xlabel=r"impact parameter  $b/R_{200c}$",
              tag_left=False):
    plt.rcParams.update(RC)
    nrow, ncol = len(masses), len(redshifts)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol + 1.2, 3.35 * nrow + 1.6), sharex=True, sharey="row")
    for i, m in enumerate(masses):
        for j, z in enumerate(redshifts):
            ax = axes[i, j]
            b16 = data[("battaglia16", m, z)]
            lee = data[("lee22_noconc", m, z)]
            x = arrays(b16, "impact_r200c")
            calibrated_mz = LEE_MASS[0] <= m <= LEE_MASS[1] and z <= LEE_ZMAX
            ax.axvspan(x.min() * .8, LEE_RADII[0], color=".5", alpha=.14, lw=0, zorder=0)
            ax.axvspan(LEE_RADII[1], x.max() * 1.2, color=".5", alpha=.14, lw=0, zorder=0)
            ax.plot(x, arrays(b16, quantity), color=BLUE, lw=2.8, label="Battaglia16 density fit")
            ax.plot(x, arrays(lee, quantity), color=ORANGE, lw=2.8, ls="--",
                    label="Lee22 no-c fit" + ("" if calibrated_mz else " (extrapolated)"), alpha=1 if calibrated_mz else .8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x.min(), x.max())
            ax.grid(alpha=.15, which="both")
            ax.tick_params(direction="in", which="both", top=True, right=True)
            notes = []
            if not (LEE_MASS[0] <= m <= LEE_MASS[1]):
                notes.append("mass non calibr.")
            if z > LEE_ZMAX:
                notes.append("z non calibr.")
            if notes:
                ax.text(.03 if tag_left else .97, .96, "\n".join(notes), transform=ax.transAxes,
                        ha="left" if tag_left else "right", va="top", color=RED, fontsize=13, fontweight="bold")
            r200 = float(b16[0]["r200c_mpc"])
            theta = float(b16[0]["theta200c_arcmin"])
            ax.text(.03, .04, r"$R_{200c}$ = %.2f Mpc, $\theta_{200c}$ = %.1f$'$" % (r200, theta),
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=".3")
            if i == 0:
                ax.set_title("$z$ = %.1f" % z + ("" if z <= LEE_ZMAX else "  (Lee22 fit: $z \\leq 2$)"), pad=8)
            if j == 0:
                ax.set_ylabel(ylabel)
                ax.text(-.36, .5, mass_label(m) + ("\n(HalfDome floor)" if abs(m - 7.327e12) < 1e9 else "") +
                        "\n" + mass_note(m).replace(": ", ":\n"), transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=13, color=RED if "non" in mass_note(m) else ".15")
            if i == nrow - 1:
                ax.set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=".5", alpha=.25, lw=0))
    labels.append(("$b$" if "b/" in xlabel else "$r$") + r" outside the Lee22 fit range 0.04-1.34 $R_{200c}$: non calibr.")
    labels = [l.replace(" (extrapolated)", "") for l in labels]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995),
               handlelength=2.8, columnspacing=1.8)
    fig.subplots_adjust(left=.11, right=.99, top=.905, bottom=.06, hspace=.12, wspace=.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / (stem + "." + ext)), dpi=170 if ext == "png" else None)
    plt.close(fig)


def plot_ratio(data, masses, redshifts, out):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(redshifts), figsize=(3.9 * len(redshifts) + 1.6, 5.6), sharey=True)
    colors = plt.get_cmap("viridis")(np.linspace(.05, .9, len(masses)))
    for j, z in enumerate(redshifts):
        ax = axes[j]
        for i, m in enumerate(masses):
            b16 = data[("battaglia16", m, z)]
            lee = data[("lee22_noconc", m, z)]
            x = arrays(b16, "impact_r200c")
            ratio = arrays(lee, "column_rest_pc_cm3") / arrays(b16, "column_rest_pc_cm3")
            calibrated = LEE_MASS[0] <= m <= LEE_MASS[1]
            ax.plot(x, ratio, color=colors[i], lw=2.6, ls="-" if calibrated else (0, (3, 1.5)),
                    label=mass_label(m) + ("" if calibrated else "  (mass non calibr.)"))
        ax.axvspan(x.min() * .8, LEE_RADII[0], color=".5", alpha=.14, lw=0, zorder=0)
        ax.axvspan(LEE_RADII[1], x.max() * 1.2, color=".5", alpha=.14, lw=0, zorder=0)
        ax.axhline(1, color=".4", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(.05, 50)
        ax.grid(alpha=.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.set_title("$z$ = %.1f" % z + ("" if z <= LEE_ZMAX else "  (z non calibr.)"),
                     color=".1" if z <= LEE_ZMAX else RED, pad=8)
        ax.set_xlabel(r"$b/R_{200c}$")
    axes[0].set_ylabel("Lee22 no-c / Battaglia16\n(projected electron column)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=".5", alpha=.25, lw=0))
    labels.append(r"$b$ outside 0.04-1.34 $R_{200c}$: non calibr.")
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.5)
    fig.subplots_adjust(left=.085, right=.99, top=.78, bottom=.14, wspace=.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / ("projected_column_ratio_lee22_over_b16." + ext)), dpi=170 if ext == "png" else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("frb_map_generation/outputs/projected_profiles_20260917"))
    args = parser.parse_args()
    data, masses, redshifts = load(args.output / "projected_profiles.csv")
    plot_grid(data, masses, redshifts, args.output, "column_rest_pc_cm3",
              r"$\int n_e\,dl$  [pc cm$^{-3}$]", "projected_electron_column_b16_vs_lee22")
    plot_grid(data, masses, redshifts, args.output, "ne_3d_cm3",
              r"$n_e(r)$  [cm$^{-3}$]", "electron_density_3d_b16_vs_lee22", xlabel=r"radius  $r/R_{200c}$")
    plot_grid(data, masses, redshifts, args.output, "ne_over_n200_x3",
              r"$(n_e/n_{200})\,(r/R_{200c})^3$", "electron_density_scaled_n200_x3_b16_vs_lee22",
              xlabel=r"radius  $r/R_{200c}$", tag_left=True)
    plot_ratio(data, masses, redshifts, args.output)
    print("Saved figures in", args.output)


if __name__ == "__main__":
    main()
