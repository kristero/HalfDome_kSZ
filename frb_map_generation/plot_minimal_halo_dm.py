#!/usr/bin/env python3
"""Visualize the smallest DM a ray can receive from one halo (grazing the aperture edge)."""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"font.family": "DejaVu Sans", "font.size": 15, "axes.labelsize": 17, "axes.titlesize": 17,
         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12.5, "lines.linewidth": 2.2,
         "xtick.direction": "in", "ytick.direction": "in", "svg.fonttype": "none", "savefig.facecolor": "white"}
ZCOL = {0.05: "#4B0055", 0.2: "#005F73", 0.5: "#0A9396", 0.8: "#EE9B00", 1.0: "#AE2012"}
NSIDE_PIX_ARCMIN = {2048: 1.718, 4096: 0.859, 8192: 0.4295, 16384: 0.2148}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="frb_map_generation/outputs/minimal_dm_20260916/minimal_halo_dm_grid.csv")
    ap.add_argument("--output-dir", default="frb_map_generation/outputs/minimal_dm_20260916")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.grid)))
    col = lambda k, z: np.array([float(r[k]) for r in rows if float(r["z"]) == z])
    zs = sorted({float(r["z"]) for r in rows})
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
        fig.subplots_adjust(left=.05, right=.99, bottom=.13, top=.86, wspace=.27)
        # (a) grazing DM versus mass, projected apertures
        ax = axes[0]
        for z in zs:
            m = 10**col("log10_m200c_msun", z)
            ax.plot(m, col("b16_edge_dm_1r200c", z), color=ZCOL[z], label=f"$z={z:g}$")
            ax.plot(m, col("b16_edge_dm_3r200c", z), color=ZCOL[z], ls="--", lw=1.6)
            ax.plot(m, col("lee22_best_edge_dm_1r200c", z), color=ZCOL[z], ls=":", lw=1.8)
        z = 0.5
        m = 10**col("log10_m200c_msun", z)
        for mk, dy in ((7.327e12, 1.25), (1e13, 1.25), (1e14, 1.25), (1e15, 1.25)):
            i = np.argmin(np.abs(np.log10(m) - np.log10(mk)))
            for key, ls, off in (("b16_edge_dm_1r200c", "-", 1.35), ("b16_edge_dm_3r200c", "--", 0.72), ("lee22_best_edge_dm_1r200c", ":", 0.72)):
                v = col(key, z)[i]
                if key == "lee22_best_edge_dm_1r200c" and mk not in (7.327e12, 1e14, 1e15):
                    continue
                dx = -16 if key == "lee22_best_edge_dm_1r200c" else 0
                ax.annotate(f"{v:.1f}", (m[i], v), xytext=(dx, 9 if off > 1 else -13), textcoords="offset points",
                            ha="center", fontsize=10.5, color=ZCOL[z], fontweight="bold")
            ax.axvline(mk, color=".85", lw=.8, zorder=0)
        ax.axvline(7.327e12, color=".4", lw=1, ls="-.")
        ax.text(7.5e12, 800, "HalfDome\nfloor", fontsize=11, color=".35", va="top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$M_{200c}$ [$M_\odot$]"); ax.set_ylabel(r"DM of a ray at the aperture edge [pc cm$^{-3}$]")
        ax.set_title("Grazing ray: projected aperture, long LOS")
        h = [plt.Line2D([], [], color=".2", ls="-", label=r"Battaglia16, edge at $1\,R_{200c}$"),
             plt.Line2D([], [], color=".2", ls="--", label=r"Battaglia16, edge at $3\,R_{200c}$"),
             plt.Line2D([], [], color=".2", ls=":", label=r"Lee22 best (corrected), edge at $1\,R_{200c}$")]
        leg1 = ax.legend(handles=h, loc="upper left", frameon=False)
        ax.add_artist(leg1)
        ax.legend(handles=[plt.Line2D([], [], color=ZCOL[z], label=f"$z={z:g}$") for z in zs], loc="lower right", frameon=False, ncol=2)
        ax.text(.98, .42, "annotated values: $z=0.5$", transform=ax.transAxes, ha="right", fontsize=11, color=ZCOL[.5])
        ax.grid(color=".9", lw=.6)

        # (b) spherical truncation: DM versus impact parameter near the edge, floor mass
        ax = axes[1]
        for z in (0.2, 0.5, 1.0):
            m = 10**col("log10_m200c_msun", z)
            i = 0  # floor mass
            b = np.array([0.5, 0.9, 0.99])
            v = np.array([col("b16_sphere_dm_b0p5r200c", z)[i], col("b16_sphere_dm_b0p9r200c", z)[i], col("b16_sphere_dm_b0p99r200c", z)[i]])
            ax.plot(1 - b, v, "o-", color=ZCOL[z], label=f"sphere, $z={z:g}$")
            ax.axhline(col("b16_edge_dm_1r200c", z)[i], color=ZCOL[z], ls="-", lw=1.2, alpha=.7)
        # analytic extension: DM ~ sqrt(1-b) near the edge (chord length), anchored at b=0.99
        z = 0.5; v99 = col("b16_sphere_dm_b0p99r200c", z)[0]
        eps = np.logspace(-5, -2, 40)
        ax.plot(eps, v99 * np.sqrt(eps / .01), ls="--", color=ZCOL[z], lw=1.4)
        ax.annotate(r"$\propto\sqrt{1-b/R_{200c}}$ (chord)", (3e-4, v99 * np.sqrt(3e-2)), fontsize=11, color=ZCOL[.5])
        for z in (0.5,):
            ax.annotate(f"{col('b16_edge_dm_1r200c', z)[0]:.1f}: projected floor, $z$={z:g}", (2e-5, col("b16_edge_dm_1r200c", z)[0] * 1.15), fontsize=10.5, color=ZCOL[z])
            ax.annotate(f"{v99:.2f} at $b=0.99R$", (1.1e-2, v99 * 0.62), fontsize=10.5, color=ZCOL[z])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel(r"$1 - b/R_{200c}$  (distance of the ray inside the edge)")
        ax.set_ylabel(r"DM [pc cm$^{-3}$]")
        ax.set_title(r"Battaglia16, $M_{200c}=7.3\times10^{12}\,M_\odot$: sphere vs projected")
        ax.legend(loc="lower left", frameon=False)
        ax.grid(color=".9", lw=.6)

        # (c) angular size of the aperture versus HEALPix pixel size
        ax = axes[2]
        for z in zs:
            m = 10**col("log10_m200c_msun", z)
            ax.plot(m, col("theta200c_arcmin", z), color=ZCOL[z], label=f"$z={z:g}$")
        for n, p in NSIDE_PIX_ARCMIN.items():
            ax.axhline(p, color=".3", ls=":" if n != 4096 else "-", lw=1.1)
            ax.text(4.2e15, p * 1.05, f"NSIDE {n} pixel", fontsize=10.5, ha="right", color=".3")
        ax.axvline(7.327e12, color=".4", lw=1, ls="-.")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$M_{200c}$ [$M_\odot$]"); ax.set_ylabel(r"$\theta_{200c}$ [arcmin]")
        ax.set_title(r"Aperture radius vs ray-pixel size")
        ax.legend(loc="upper left", frameon=False, ncol=2)
        ax.grid(color=".9", lw=.6)
        fig.suptitle("Smallest DM one halo can add to a ray  |  HalfDome conventions", y=.975, fontsize=20)
        for ext in ("png", "svg"):
            fig.savefig(out / f"minimal_halo_dm.{ext}", dpi=190, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
    print("saved", out / "minimal_halo_dm.png")


if __name__ == "__main__":
    main()
