#!/usr/bin/env python3
"""Old vs new Battaglia16 electron density: with f_b (production convention) vs without f_b
(the reading that matches Lee+2022 Fig. 5's own plotted 'Battaglia (2016)' curve, see
LEE22_FIG5_SHAPE_CHECK_20260919.md amendment 3).

Context (2026-09-19): removing f_b is NOT a general physical fix. Stein et al. 2020 (the WebSky
paper, arXiv:2001.08787, eq. 3.15) uses the exact same Battaglia (2016) density fit and writes it
as n_e = [1-Yp/2] rho_b / m_p * F(x|M,z), where rho_b is "the mean comoving BARYON density" --
i.e. f_b * rho_crit, not rho_crit alone. That is exactly XGPaint's "with f_b" convention (the
"mistake in battaglia 2016: need f_b to convert from m to gas" comment), confirmed independently
by a third, peer-reviewed, widely-used paper. So "with f_b" is the physically standard reading,
matching WebSky; "without f_b" only reproduces Lee+2022's own Fig. 5 rendering and should not be
used as a general Battaglia16 profile.

Input: outputs/projected_profiles_fig5_check_20260919/projected_profiles.csv
(compute_projected_profile_grid.jl re-run with a finer mass/z grid; z=0 rows used here).
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

F_B = 0.049 / 0.31


def load(path):
    rows = list(csv.DictReader(open(path)))
    data = {}
    for r in rows:
        data.setdefault((r["model"], float(r["mass_msun"]), float(r["redshift"])), []).append(r)
    return data


def curve(rows_):
    rows_ = sorted(rows_, key=lambda r: float(r["impact_r200c"]))
    x = np.array([float(r["impact_r200c"]) for r in rows_])
    y = np.array([float(r["ne_3d_cm3"]) for r in rows_])
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                         default=Path("frb_map_generation/outputs/projected_profiles_fig5_check_20260919/projected_profiles.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("frb_map_generation/outputs/fig5_match_20260919"))
    args = parser.parse_args()
    data = load(args.input)

    masses = [(3.0e13, "M = 3e13 M$_\\odot$\n(Lee+22 left panel)"), (1.17e14, "M = 1.17e14 M$_\\odot$\n(Lee+22 right panel)")]
    z = 0.0

    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 12.5, "legend.fontsize": 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    for ax, (mass, title) in zip(axes, masses):
        x, y_withfb = curve(data[("battaglia16", mass, z)])
        y_nofb = y_withfb / F_B
        ax.plot(x, y_withfb, color="#0b6b78", lw=2.8,
                 label="Battaglia16, with f_b\n(production convention; matches WebSky/Stein+20 eq. 3.15)")
        ax.plot(x, y_nofb, color="#d55e00", lw=2.4, ls="--",
                 label="Battaglia16, without f_b\n(matches Lee+22 Fig. 5's own curve only)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.03, 3); ax.set_xlabel(r"$r/R_{200c}$")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.axhline(1, color="none")  # keep autoscale consistent
    axes[0].set_ylabel(r"$n_e$ [cm$^{-3}$]")
    axes[0].text(0.05, 0.05, f"ratio = 1/f_b = {1/F_B:.2f}x, constant at all r", transform=axes[0].transAxes,
                 fontsize=10, color=".25")
    axes[1].legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    fig.suptitle("Battaglia16 electron density: production convention (with f_b) vs the Fig.-5-only reading (without f_b)",
                 fontsize=12.5, y=1.02)
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(args.output_dir / ("battaglia16_fb_comparison." + ext)), dpi=170 if ext == "png" else None,
                    bbox_inches="tight")
    print("Saved", args.output_dir / "battaglia16_fb_comparison.png")


if __name__ == "__main__":
    main()
