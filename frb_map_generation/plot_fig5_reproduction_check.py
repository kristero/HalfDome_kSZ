#!/usr/bin/env python3
"""Check whether our Battaglia16-vs-Lee22 electron-density comparison can reproduce the qualitative
shape of Lee et al. 2022 (arXiv:2205.01710) Figure 5, in the paper's own convention
(n_e/n200)(r/R200c)^3, at the two mass bins closest to Fig. 5's panels.

Context (2026-09-19): a reviewer noted that at M ~ 1e14 Msun our comparison plots show Lee22 with a
higher amplitude than Battaglia16 at essentially every radius, whereas Lee+2022's own Fig. 5 shows the
opposite trend at large r (their "Battaglia (2016)" cyan reference curve sits above the TNG/best-fit
curve at small r and the two converge near r ~ R200c). This script shows that removing the f_b
(baryon-fraction) factor from our physical Battaglia16 density -- i.e. plotting the same quantity that
this repository's own 2026-09-16/17 notes already found matches Lee+2022's plotted "Battaglia (2016)"
curve -- reproduces that qualitative shape; our normal, physically-normalized Battaglia16 (which
correctly includes f_b, per XGPaint's own "mistake in battaglia 2016: need f_b to convert from m to
gas" comment) does not, and was never intended to look like the paper's own (f_b-uncorrected) reference
curve. See LEE22_IMPLEMENTATION_CHECK_20260916.md (Section 2) and
LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md (Section 6) for the full derivation of this f_b factor.

Input: outputs/projected_profiles_20260917/projected_profiles.csv (compute_projected_profile_grid.jl).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F_B = 0.049 / 0.31  # HalfDome Omega_b/Omega_m; TNG's 0.0486/0.3089 agrees to <1%
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", ".35"
RC = {"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 12.5, "legend.fontsize": 11.5}
PANELS = [(3e13, "M ~ 3e13 M$_\\odot$\n(Lee+22 left panel: 10$^{13.2-13.4}\\,h^{-1}M_\\odot$)"),
          (1e14, "M = 1e14 M$_\\odot$\n(Lee+22 right panel: 10$^{13.8-14.0}\\,h^{-1}M_\\odot$)")]
Z = 0.1  # closest grid point to Lee+2022 Fig. 5's z = 0


def load(path):
    rows = list(csv.DictReader(open(path)))
    data = {}
    for r in rows:
        data.setdefault((r["model"], float(r["mass_msun"]), float(r["redshift"])), []).append(r)
    return data


def scaled_ne_over_n200_x3(rows):
    """(n_e/n200)(r/R200c)^3, Lee+2022's own Fig. 5 y-axis convention."""
    rows = sorted(rows, key=lambda r: float(r["impact_r200c"]))
    x = np.array([float(r["impact_r200c"]) for r in rows])
    y = np.array([float(r["ne_3d_cm3"]) / float(r["n200_eq9_cm3"]) * xi ** 3 for r, xi in zip(rows, x)])
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("frb_map_generation/outputs/projected_profiles_20260917"))
    args = parser.parse_args()
    data = load(args.input / "projected_profiles.csv")

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.0), sharey=True)
    for ax, (mass, title) in zip(axes, PANELS):
        x, y_b16 = scaled_ne_over_n200_x3(data[("battaglia16", mass, Z)])
        _, y_lee = scaled_ne_over_n200_x3(data[("lee22_noconc", mass, Z)])
        keep = (x >= 0.03) & (x <= 1.8)
        ax.plot(x[keep], y_b16[keep] / F_B, color=GREY, lw=2.6, label="Battaglia16 / f_b\n(Lee+22's own Fig.5 curve)")
        ax.plot(x[keep], y_lee[keep], color=ORANGE, lw=2.8, label="Lee22 no-c (our Table 8 fit)\n≈ Lee+22's TNG/best-fit curve")
        ax.plot(x[keep], y_b16[keep], color=BLUE, lw=2.8, ls="--", label="Battaglia16, physical\n(our current comparison plots)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.03, 1.8)
        ax.set_ylim(3e-4, 3)
        ax.axvline(1.34, color=".7", lw=1, ls=":")
        ax.set_xlabel(r"$r/R_{200c}$")
        ax.set_title(title, fontsize=12.5, pad=10)
        ax.grid(alpha=.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    axes[0].set_ylabel(r"$(n_e/n_{200})\,(r/R_{200c})^3$   (Lee+2022 Fig.5 convention)")
    axes[1].legend(loc="lower right", fontsize=10, framealpha=.95)
    fig.suptitle("Reproducing the shape of Lee+2022 Fig. 5 needs Battaglia16 WITHOUT the f_b (baryon-fraction) factor",
                 fontsize=13.5, y=1.06)
    fig.text(0.5, 0.965, "z = 0.1 (Lee+22 use z=0); vertical dotted line: Lee22's fitted radial range ends at 1.34 R200c",
             ha="center", fontsize=10.5, color=".3")
    fig.subplots_adjust(top=0.82, wspace=0.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(args.input / ("fig5_reproduction_check." + ext)), dpi=170 if ext == "png" else None,
                    bbox_inches="tight")
    print("Saved", args.input / "fig5_reproduction_check.png")


if __name__ == "__main__":
    main()
