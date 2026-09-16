#!/usr/bin/env python3
"""Plot the current cache/direct profile audit; optionally select nearby tests.

This does not repaint maps or change the halo population. Nearby test points
come from a complete catalogue scan, followed by a small deterministic test
sample. The sample is only for checking numerical interpolation, not science.
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def select_nearby(output_dir, catalogue):
    edges = [0, .01, .02, .05, .1, 1]
    counts = np.zeros(len(edges)-1, dtype=np.int64)
    nearby = []
    with h5py.File(catalogue) as h5:
        for start in range(0, len(h5["redshift"]), 1_000_000):
            z = np.asarray(h5["redshift"][start:start+1_000_000], dtype=float)
            mass = np.asarray(h5["halo_mass_m200c"][start:start+1_000_000], dtype=float)/.68
            valid = np.isfinite(z) & np.isfinite(mass) & (mass>0) & (z>=0) & (z<=1)
            counts += np.histogram(z[valid], edges)[0]
            keep = valid & (z<.1)
            nearby.extend(zip((start+np.flatnonzero(keep)).tolist(), mass[keep].tolist(), z[keep].tolist()))
    points = np.array(nearby)
    # All of the very nearest halos plus mass-ordered representatives at .02-.1.
    chosen = list(points[points[:, 2]<.02])
    for lo, hi in [(.02, .05), (.05, .1)]:
        chunk = points[(points[:, 2]>=lo)&(points[:, 2]<hi)]
        if len(chunk):
            chunk = chunk[np.argsort(chunk[:, 1])]
            chosen.extend(chunk[np.unique(np.linspace(0, len(chunk)-1, min(30, len(chunk))).astype(int))])
    with (output_dir / "nearby_catalogue_test_points.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["catalogue_row_zero_based", "mass_msun", "redshift"])
        writer.writerows(chosen)
    result = dict(catalogue=str(catalogue), foreground_redshift_edges=edges,
                  foreground_counts=counts.tolist(), numerical_test_points=len(chosen),
                  nearby_mass_min_msun=float(np.min(points[:, 1])),
                  nearby_mass_max_msun=float(np.max(points[:, 1])))
    (output_dir / "nearby_catalogue_counts.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def profile_plots(output_dir):
    grid = np.genfromtxt(output_dir / "production_cache_vs_direct_profiles.csv",
                         delimiter=",", names=True, dtype=None, encoding="utf8")
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, constrained_layout=True)
    for i, z in enumerate([.1, .5, 1.]):
        for j, mass in enumerate([13., 14., 15.]):
            ax = axes[i, j]
            selected = grid[(grid["redshift"]==z)&(grid["log10_mass_msun"]==mass)&(grid["r_perp_r200c"]<=3)]
            for name, color in [("battaglia16", "#28749a"), ("lee2022", "#d24863")]:
                data = selected[selected["profile"]==name]
                ax.loglog(data["r_perp_r200c"], data["direct_dm_pc_cm3"], color=color, label=name+" direct")
                ax.scatter(data["r_perp_r200c"], data["cached_dm_pc_cm3"], color=color, marker="x", s=25,
                           label=name+" production cache")
            ax.set_title(rf"$\log_{{10}}(M_{{200c}}/M_\odot)={mass:g}$, $z_{{halo}}={z:g}$")
            ax.grid(alpha=.2, which="both")
            if i == 2:
                ax.set_xlabel(r"$R_\perp/R_{200c}$")
            if j == 0:
                ax.set_ylabel(r"Observed halo DM [pc cm$^{-3}$]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Current production profiles: source z=1 includes foreground halos at every z below 1")
    fig.savefig(output_dir / "current_production_profile_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    result = {}
    for name in ["battaglia16", "lee2022"]:
        data = grid[(grid["profile"]==name)&(grid["log10_mass_msun"]>=12.86)&(grid["r_perp_r200c"]<=3)]
        result[name] = [{"z": float(z), "minimum_cache_over_direct": float(np.min(data[data["redshift"]==z]["cached_over_direct"])),
                         "maximum_cache_over_direct": float(np.max(data[data["redshift"]==z]["cached_over_direct"]))}
                        for z in np.unique(data["redshift"])]
    (output_dir / "profile_grid_summary.json").write_text(json.dumps(result, indent=2))

    density_path = output_dir / "three_dimensional_density_profiles.csv"
    if density_path.exists():
        density = np.genfromtxt(density_path, delimiter=",", names=True)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, constrained_layout=True)
        for j, z in enumerate([.1, .5, 1.]):
            d = density[(density["redshift"]==z)&(density["log10_mass_msun"]==14)]
            axes[0, j].loglog(d["r_r200c"], d["b16_ne_cm3"], color="#28749a", label="Battaglia16")
            axes[0, j].loglog(d["r_r200c"], d["lee22_ne_cm3"], color="#d24863", label="Lee22, no concentration")
            axes[0, j].set_title(rf"Halo redshift $z={z:g}$")
            for name, color in [("battaglia16", "#28749a"), ("lee2022", "#d24863")]:
                d = grid[(grid["profile"]==name)&(grid["redshift"]==z)&(grid["log10_mass_msun"]==14)&(grid["r_perp_r200c"]<=3)]
                axes[1, j].loglog(d["r_perp_r200c"], d["direct_dm_pc_cm3"], color=color)
            axes[0, j].set_ylabel(r"Physical $n_e(r)$ [cm$^{-3}$]")
            axes[1, j].set_ylabel(r"Observed halo DM [pc cm$^{-3}$]")
            axes[1, j].set_xlabel(r"$r/R_{200c}$ (top); $R_\perp/R_{200c}$ (bottom)")
            for ax in axes[:, j]:
                ax.grid(alpha=.2, which="both")
                ax.axvspan(.001, .04, color="0.8", alpha=.2)
                ax.axvspan(1.34, 3, color="0.8", alpha=.2)
                ax.set_xlim(.001, 3)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(r"Current $M_{200c}=10^{14}M_\odot$ profiles; grey bands are outside Lee22's radial fit")
        fig.savefig(output_dir / "physical_density_and_column_check.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lee22_power_audit_20260909")
    parser.add_argument("--select-nearby", action="store_true")
    parser.add_argument("--select-only", action="store_true", help="Select test points before running Julia; do not require its outputs yet.")
    parser.add_argument("--catalogue", type=Path, default=ROOT.parent / "lightcone_100.hdf5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.select_nearby or args.select_only:
        select_nearby(args.output_dir, args.catalogue)
    if args.select_only:
        return
    profile_plots(args.output_dir)


if __name__ == "__main__":
    main()
