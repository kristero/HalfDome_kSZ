#!/usr/bin/env python3
"""Electron-pressure profiles: Battaglia12 fiducial thermal-pressure fit versus the Lee22 no-concentration
pressure fit (arXiv v1 Table 7 = MNRAS Table A1, XGPaint mapping), on a mass x redshift grid, with the Lee22
calibration ranges marked. Companion to plot_projected_profile_comparison.py (electron density).

Input: outputs/pressure_profiles_20260919/pressure_profiles.csv (compute_pressure_profile_grid.jl).
Lee22 pressure fit range: radii 0.04-1.34 R200, halo masses 1e13-10^14.8 h^-1 Msun (h = 0.68 here),
20 TNG300 snapshots to z = 2 (same range as the Lee22 density fit).
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
MOST_MASSIVE_HALO_MSUN = 3.785236255949654e15  # true global max of halo_mass_m200c in lightcone_100.hdf5 (at z=0.344)
MOST_MASSIVE_HALO_Z = 0.344
BLUE, ORANGE, RED = "#0072B2", "#D55E00", "#B2182B"
RC = {"font.size": 14, "axes.labelsize": 17, "axes.titlesize": 15, "xtick.labelsize": 13,
      "ytick.labelsize": 13, "legend.fontsize": 14, "axes.linewidth": 1.0}


def load(path):
    rows = list(csv.DictReader(open(path)))
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
    return (r"$10^{%d}\,M_\odot$" % e) if abs(a - 1) < 1e-6 else (r"$%.2f\times10^{%d}\,M_\odot$" % (a, e))


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
            b12 = data[("battaglia12", m, z)]
            lee = data[("lee22_noconc", m, z)]
            x = arrays(b12, "x_r200c")
            calibrated_mz = LEE_MASS[0] <= m <= LEE_MASS[1] and z <= LEE_ZMAX
            ax.axvspan(x.min() * .8, LEE_RADII[0], color=".5", alpha=.14, lw=0, zorder=0)
            ax.axvspan(LEE_RADII[1], x.max() * 1.2, color=".5", alpha=.14, lw=0, zorder=0)
            ax.plot(x, arrays(b12, quantity), color=BLUE, lw=2.8, label="Battaglia12 pressure fit")
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
            r200 = float(b12[0]["r200c_mpc"])
            theta = float(b12[0]["theta200c_arcmin"])
            info = r"$R_{200c}$ = %.2f Mpc, $\theta_{200c}$ = %.1f$'$" % (r200, theta)
            if abs(m / MOST_MASSIVE_HALO_MSUN - 1) < 1e-6:
                info += "; host $z$=%.2f" % MOST_MASSIVE_HALO_Z
            ax.text(.03, .04, info, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, color=".3")
            if i == 0:
                ax.set_title("$z$ = %.1f" % z + ("" if z <= LEE_ZMAX else "  (Lee22 fit: $z \\leq 2$)"), pad=8)
            if j == 0:
                ax.set_ylabel(ylabel)
                extra = ""
                if abs(m - 7.327e12) < 1e9:
                    extra = "\n(HalfDome floor)"
                elif abs(m / MOST_MASSIVE_HALO_MSUN - 1) < 1e-6:
                    extra = "\n(most massive halo)"
                ax.text(-.36, .5, mass_label(m) + extra +
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
            b12 = data[("battaglia12", m, z)]
            lee = data[("lee22_noconc", m, z)]
            x = arrays(b12, "x_r200c")
            ratio = arrays(lee, "y_projected") / arrays(b12, "y_projected")
            calibrated = LEE_MASS[0] <= m <= LEE_MASS[1]
            ax.plot(x, ratio, color=colors[i], lw=2.6, ls="-" if calibrated else (0, (3, 1.5)),
                    label=mass_label(m) + ("" if calibrated else "  (mass non calibr.)"))
        ax.axvspan(x.min() * .8, LEE_RADII[0], color=".5", alpha=.14, lw=0, zorder=0)
        ax.axvspan(LEE_RADII[1], x.max() * 1.2, color=".5", alpha=.14, lw=0, zorder=0)
        ax.axhline(1, color=".4", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(.02, 20)
        ax.grid(alpha=.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.set_title("$z$ = %.1f" % z + ("" if z <= LEE_ZMAX else "  (z non calibr.)"),
                     color=".1" if z <= LEE_ZMAX else RED, pad=8)
        ax.set_xlabel(r"$b/R_{200c}$")
    axes[0].set_ylabel("Lee22 no-c / Battaglia12\n(projected Compton-$y$)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=".5", alpha=.25, lw=0))
    labels.append(r"$b$ outside 0.04-1.34 $R_{200c}$: non calibr.")
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.5)
    fig.subplots_adjust(left=.085, right=.99, top=.78, bottom=.14, wspace=.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(out / ("pressure_projected_y_ratio_lee22_over_b12." + ext)), dpi=170 if ext == "png" else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("frb_map_generation/outputs/pressure_profiles_20260919"))
    args = parser.parse_args()
    data, masses, redshifts = load(args.output / "pressure_profiles.csv")
    plot_grid(data, masses, redshifts, args.output, "y_projected",
              r"Compton $y(b)$", "pressure_projected_y_b12_vs_lee22")
    plot_grid(data, masses, redshifts, args.output, "pe_3d_kev_cm3",
              r"$P_e(r)$  [keV cm$^{-3}$]", "pressure_3d_kev_cm3_b12_vs_lee22", xlabel=r"radius  $r/R_{200c}$")
    plot_grid(data, masses, redshifts, args.output, "pe_over_p200_x3",
              r"$(P_e/P_{200})\,(r/R_{200c})^3$", "pressure_scaled_p200_x3_b12_vs_lee22",
              xlabel=r"radius  $r/R_{200c}$", tag_left=True)
    plot_ratio(data, masses, redshifts, args.output)
    print("Saved figures in", args.output)


if __name__ == "__main__":
    main()
