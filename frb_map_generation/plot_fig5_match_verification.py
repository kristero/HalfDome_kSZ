#!/usr/bin/env python3
"""Overlay our XGPaint-computed 'Battaglia16 (no f_b)' and 'Lee22 (:literal)' curves against
electron-density values digitized directly from Lee et al. 2022 (arXiv:2205.01710) Figure 5's
own PNG, to verify the normalization choice identified in LEE22_FIG5_SHAPE_CHECK_20260919.md
(amendment 2). Digitization: pixel-color matching against the published cyan ('Battaglia (2016)')
and red ('Simulation') curves on the rendered PDF page, calibrated against the axis tick labels;
see fig5_digitized.pkl / the conversation record for the extraction code. A small pixel range in
the right panel (x in [0.16, 0.21]) is masked out where the in-plot legend box overlaps the
curves.

Inputs: outputs/fig5_match_20260919/fig5_match.csv (compute_fig5_match_grid.jl) and a digitized
data pickle (regenerate by re-running the extraction against tmp/pdfs/lee2022_arxiv_2205.01710v1.pdf
page 11 if unavailable; not committed since it is derived from someone else's published figure).
"""
import argparse
import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PANELS = [("left", 3.0e13, "10$^{13.2}$-10$^{13.4}$ $h^{-1}M_\\odot$"),
          ("right", 1.17e14, "10$^{13.8}$-10$^{14.0}$ $h^{-1}M_\\odot$")]
RC = {"font.size": 13, "axes.labelsize": 14, "axes.titlesize": 12.5, "legend.fontsize": 10.5}


def load_grid(path):
    rows = list(csv.DictReader(open(path)))
    data = {}
    for r in rows:
        data.setdefault((r["curve"], float(r["mass_msun"])), []).append(r)
    out = {}
    for key, rows_ in data.items():
        rows_ = sorted(rows_, key=lambda r: float(r["impact_r200c"]))
        out[key] = (np.array([float(r["impact_r200c"]) for r in rows_]),
                    np.array([float(r["ne_over_n200_x3"]) for r in rows_]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=Path("frb_map_generation/outputs/fig5_match_20260919/fig5_match.csv"))
    parser.add_argument("--digitized", type=Path, default=Path("/tmp/claude-0/-home-cbllover-HalfDome/015746ae-94b7-42cb-8dc5-0e8c33dc8490/scratchpad/fig5_digitized.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("frb_map_generation/outputs/fig5_match_20260919"))
    args = parser.parse_args()

    grid = load_grid(args.grid)
    digitized = pickle.load(open(args.digitized, "rb"))

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    for ax, (panel, mass, title) in zip(axes, PANELS):
        xb, yb = grid[("battaglia16_no_fb", mass)]
        xl, yl = grid[("lee22_literal", mass)]
        xs_c, ys_c = digitized[(panel, "cyan")]
        xs_r, ys_r = digitized[(panel, "red")]
        oc = np.argsort(xs_c); xs_c, ys_c = xs_c[oc], ys_c[oc]
        orr = np.argsort(xs_r); xs_r, ys_r = xs_r[orr], ys_r[orr]
        if panel == "right":
            keep_c = ~((xs_c > 0.16) & (xs_c < 0.21)); xs_c, ys_c = xs_c[keep_c], ys_c[keep_c]
            keep_r = ~((xs_r > 0.16) & (xs_r < 0.21)); xs_r, ys_r = xs_r[keep_r], ys_r[keep_r]
        ax.scatter(xs_c[::4], ys_c[::4], s=4, color="#17becf", alpha=0.5, label="Fig. 5, digitized: Battaglia (2016)")
        ax.scatter(xs_r[::4], ys_r[::4], s=4, color="#d62728", alpha=0.5, label="Fig. 5, digitized: Simulation")
        ax.plot(xb, yb, color="#0b6b78", lw=2.6, label="XGPaint Battaglia16, no f_b (this work)")
        ax.plot(xl, yl, color="#8c1616", lw=2.6, ls="--", label="XGPaint Lee22 no-c, :literal (this work)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.03, 1.6); ax.set_ylim(3e-4, 3)
        ax.set_xlabel(r"$r/R_{200c}$")
        ax.set_title(title, fontsize=12.5)
        ax.grid(alpha=0.15, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    axes[0].set_ylabel(r"$(n_e/n_{200})\,(r/R_{200c})^3$")
    axes[1].legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    fig.suptitle("Reproducing Lee+2022 Fig. 5: Battaglia16 without f_b, Lee22 with the literal eq. (9) reading",
                 fontsize=13, y=1.02)
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(str(args.output_dir / ("fig5_match_verification." + ext)), dpi=170 if ext == "png" else None,
                    bbox_inches="tight")
    print("Saved", args.output_dir / "fig5_match_verification.png")


if __name__ == "__main__":
    main()
