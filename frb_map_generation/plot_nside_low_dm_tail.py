#!/usr/bin/env python3
"""Low-DM tail of the Battaglia16 1R200c halo PDF versus HEALPix NSIDE of the ray pixels."""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

from .compare_tng_halfdome_direct import DirectComparison
from .compare_tng_lee22_battaglia16 import UPPER_ENTRIES, TO_1E14_ENTRIES

STYLE = {"font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
         "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14, "lines.linewidth": 2.2,
         "xtick.direction": "in", "ytick.direction": "in", "svg.fonttype": "none", "savefig.facecolor": "white"}
NSIDES = (2048, 4096, 8192)
NCOL = {2048: "#E69F00", 4096: "#0072B2", 8192: "#009E73"}
WINDOWS = (("Total", "all", "m1e10_to_1e16"), (r"$10^{13}-10^{14}$", "m1e13_to_1e14", "m1e13_to_1e14"),
           (r"$10^{10}-10^{14}$", "m1e10_to_1e14", "m1e10_to_1e14"))
FLOOR_1R200C = 22.3  # grazing DM of a floor-mass halo at z=0.5, Battaglia16, projected 1R200c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="/home/cbllover/HalfDome")
    ap.add_argument("--output-dir", default="frb_map_generation/outputs/minimal_dm_20260916")
    a = ap.parse_args()
    direct = DirectComparison(a.project)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    products = {n: {w: direct.load_halfdome(1., n, 1.0, w, validated_200c=True) for _, _, w in WINDOWS} for n in NSIDES}
    rows, stats = [], {}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(20, 10.5), gridspec_kw=dict(height_ratios=(2.6, 1.2), hspace=.08, wspace=.25),
                                 sharex="col")
        for j, (title, tng_label, w) in enumerate(WINDOWS):
            top, bottom = axes[0, j], axes[1, j]
            edges = products[4096][w]["edges"]; centers = products[4096][w]["centers"]
            tng_pdf, tng_count, tng_zero = direct.histogram_from_values(direct.tng_values(tng_label, 1.), edges)
            direct.draw_pdf(top, centers, tng_pdf, color=".15", lw=2.6)
            ref = products[4096][w]
            for n in NSIDES:
                pr = products[n][w]
                np.testing.assert_allclose(pr["edges"], edges, rtol=1e-12, atol=0)
                direct.draw_pdf(top, centers, pr["pdf"], color=NCOL[n], ls="-" if n == 4096 else "--", lw=2.0)
                ratio = np.where((pr["counts"] >= 5) & (ref["counts"] >= 5), pr["pdf"] / np.where(ref["pdf"] > 0, ref["pdf"], np.nan), np.nan)
                bottom.plot(centers, ratio, color=NCOL[n], ls="-" if n == 4096 else "--")
                cum = np.cumsum(pr["counts"])
                low = {thr: int(cum[np.searchsorted(edges, thr) - 1]) for thr in (5., 10., 20., 25., 30., 50., 100.)}
                stats[(n, w)] = dict(nside=n, window=w, intersections=int(pr["attrs"]["unique_halo_frb_intersection_count"]) if "attrs" in pr else None,
                                    zero_fraction=pr["zero_fraction"], in_range=int(pr["counts"].sum()),
                                    min_dm_bin_low=float(edges[np.flatnonzero(pr["counts"])[0]]) if pr["counts"].any() else np.nan,
                                    **{f"n_below_{int(k)}": v for k, v in low.items()})
                rows.append(stats[(n, w)])
            top.axvline(FLOOR_1R200C, color=".5", ls=":", lw=1.4)
            top.text(FLOOR_1R200C * 1.08, 3e-7, "grazing floor,\n$7.3\\times10^{12}\\,M_\\odot$, $z=0.5$", fontsize=11.5, color=".4", va="bottom")
            top.set_title(title, pad=8); top.set_yscale("log"); top.set_xscale("log"); top.set_xlim(1, 3000)
            top.set_ylim(1e-7, 3e-2)
            top.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
            bottom.axhline(1, color=".45", lw=1); bottom.set_ylim(.5, 1.5); bottom.set_xscale("log")
            bottom.set_xlabel(r"DM [pc cm$^{-3}$]")
            for ax in (top, bottom):
                ax.grid(color=".9", lw=.6); ax.tick_params(which="major", length=6)
            if j == 0:
                top.set_ylabel(r"$p(\mathrm{DM})$"); bottom.set_ylabel("ratio to NSIDE 4096")
            zf = ", ".join(f"{n}: {100*(1-products[n][w]['zero_fraction']):.2f}%" for n in NSIDES)
            top.text(.03, .04, "rays with DM>0\n" + zf, transform=top.transAxes, fontsize=11, color=".3", va="bottom")
        handles = [plt.Line2D([], [], color=".15", label="IllustrisTNG (within $R_{200}$)")]
        handles += [plt.Line2D([], [], color=NCOL[n], ls="-" if n == 4096 else "--", label=f"HalfDome B16, $1\\,R_{{200c}}$, NSIDE {n}") for n in NSIDES]
        axes[0, 2].legend(handles=handles, loc="upper right", frameon=False, fontsize=13)
        fig.suptitle(r"Low-DM tail versus ray-pixel resolution  |  $z_s=1$, 120k rays, same seed, catalogue and cache", y=.965)
        for ext in ("png", "svg"):
            fig.savefig(out / f"nside_low_dm_tail_b16_1r200c.{ext}", dpi=190, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
    with open(out / "nside_low_dm_tail_counts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
