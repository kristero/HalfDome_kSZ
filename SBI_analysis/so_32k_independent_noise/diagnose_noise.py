#!/usr/bin/env python3
"""Run actual Julia/HEALPix noise controls and save reproducible audit plots."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import generate as g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", default="julia")
    parser.add_argument("--output", type=Path, default=g.BUNDLE / "validation_plots")
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--lmax", type=int, default=255)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.nside < 64 or args.nside & (args.nside - 1) or not 80 < args.lmax <= min(7979, 2*args.nside):
        raise ValueError("Use a power-of-two NSIDE>=64 and 80<lmax<=2*NSIDE")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((g.BUNDLE / "config.json").read_text())
    config.update(noise_cases=["baseline", "goal"], deprojections=[0, 2])
    seed_count = g.validate_config(config)
    g.validate_noise_tables(config)
    rows = []
    for scheme in ("repeated", "stride1", "independent"):
        for product in g.products(config) if scheme == "independent" else ["baseline_deproj0"]:
            for mode in g.MODES:
                for row in range(1, 4):
                    root, s1, s2 = g.seeds(config, mode, row, product)
                    if scheme != "independent":
                        root = config["noise_seed_bases"][mode] + (row - 1 if scheme == "stride1" else 0)
                        s1, s2 = root + 10101, root + 10102
                    for split, seed in enumerate((s1, s2), 1):
                        rows.append([scheme, product, mode, row, split, seed, root])
    plan = out / "seed_plan.csv"
    provenance = dict(config=config, nside=args.nside, lmax=args.lmax,
        source_sha256={str(p.relative_to(g.BUNDLE)): g.sha256(p) for p in [
            g.BUNDLE / "generate.py", g.BUNDLE / "diagnose_noise.jl",
            g.BUNDLE / "diagnose_noise.py", g.BUNDLE / "safe_paint.jl", g.BUNDLE / "test_paint.jl",
            g.BUNDLE / "simulator/tSZ_visuals/run_halfdome_fullsky_so_noise.jl"]},
        noise_sha256={p.name: g.sha256(p) for p in (g.BUNDLE / "noise").glob("*.txt")})
    if args.plots_only:
        if json.loads((out / "diagnostic_inputs.json").read_text()) != provenance:
            raise ValueError("Existing diagnostics came from different inputs/code")
    else:
        (out / "validation_report.json").unlink(missing_ok=True)
        with plan.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scheme", "product", "mode", "row", "split", "seed", "root"])
            writer.writerows(rows)
        cmd = [args.julia, "--startup-file=no", f"--project={g.BUNDLE / 'vendor/XGPaint'}", "--threads=2",
               str(g.BUNDLE / "diagnose_noise.jl"), str(out), str(plan), str(args.nside), str(args.lmax),
               str(config["mask_seed"])]
        with (out / "julia_diagnostics.log").open("w") as log:
            subprocess.run(cmd, check=True, timeout=1800, stdout=log, stderr=subprocess.STDOUT,
                           env=g.simulator_environment(2))
        painter_cmd = [args.julia, "--startup-file=no", f"--project={g.BUNDLE / 'vendor/XGPaint'}", "--threads=4",
                       str(g.BUNDLE / "test_paint.jl"), str(out)]
        with (out / "painter_diagnostics.log").open("w") as log:
            subprocess.run(painter_cmd, check=True, timeout=300, stdout=log, stderr=subprocess.STDOUT,
                           env=g.simulator_environment(4))
        g.write_json(out / "diagnostic_inputs.json", provenance)

    corr = np.loadtxt(out / "map_correlations.csv", delimiter=",")
    auto = np.loadtxt(out / "auto_cl.csv", delimiter=",")
    cross = np.loadtxt(out / "masked_cross_cl.csv", delimiter=",")
    select = lambda scheme: [i for i, r in enumerate(rows) if r[0] == scheme]
    checks = {}
    rep = select("repeated")
    stride = select("stride1")
    new = select("independent")
    checks["repeated_row_maps_identical"] = bool(np.allclose(corr[rep[0], rep[2]], 1, atol=1e-12))
    checks["stride1_reuses_previous_split2"] = bool(np.isclose(corr[stride[1], stride[2]], 1, atol=1e-12))
    offdiag = corr[np.ix_(new, new)][~np.eye(len(new), dtype=bool)]
    checks["new_streams_have_no_duplicate_maps"] = bool(np.max(np.abs(offdiag)) < 0.1)
    enlarged = dict(config, n_rows=524288)
    large_seed_count = g.validate_config(enlarged)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                         "savefig.bbox": "tight"})
    def save(fig, name):
        fig.savefig(out / (name + ".png"), dpi=180)
        fig.savefig(out / (name + ".pdf"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, scheme, title in zip(axes, ("repeated", "stride1", "independent"),
            ("Repeated seed: rows duplicate", "Stride 1: adjacent splits duplicate", "New seed allocation")):
        indices = [i for i in select(scheme) if rows[i][1] == "baseline_deproj0" and rows[i][2] == "two_param"]
        im = ax.imshow(corr[np.ix_(indices, indices)], vmin=-1, vmax=1, cmap="RdBu_r")
        labels = [f"r{rows[i][3]} s{rows[i][4]}" for i in indices]
        ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels, title=title)
        ax.tick_params(axis="x", rotation=50)
    fig.colorbar(im, ax=axes, label="Noise-map Pearson correlation", shrink=0.8)
    fig.suptitle(f"Same SO mask; actual HEALPix noise draws (NSIDE={args.nside}, ell<= {args.lmax})")
    save(fig, "01_seed_reuse_vs_independent")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    images = [np.loadtxt(out / f"map_{scheme}_row{row}.csv", delimiter=",")
              for scheme, row in (("repeated", 1), ("repeated", 2), ("independent", 2))]
    mask = np.loadtxt(out / "mask.csv", delimiter=",")
    vmax = np.quantile(np.abs(np.array(images)[:, mask > 0.99]), 0.99)
    for ax, a, title in zip(axes, images, ("Row 1", "Row 2: repeated seed", "Row 2: new independent seed")):
        im = ax.imshow(np.ma.masked_where(mask == 0, a), origin="lower", extent=(-180, 180, -90, 90),
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set(title=title, xlabel="Longitude [deg]", ylabel="Latitude [deg]")
        ax.set_facecolor("0.9")
    fig.colorbar(im, ax=axes, label="Masked noise, Compton-y units", shrink=0.8)
    fig.suptitle("Baseline deproj0: same mask and noise power, different realization")
    save(fig, "02_noise_map_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    power_results = {}
    colors = ["#222222", "#ca3b37", "#2778ae", "#209d68"]
    for color, (product, (case, d)) in zip(colors, g.products(config).items()):
        table = np.loadtxt(g.BUNDLE / f"noise/SO_LAT_Nell_T_atmv1_{case}_fsky0p4_ILC_tSZ.txt")
        ell, nl = table[:, 0], table[:, d + 1]
        axes[0].loglog(ell, ell*(ell+1)*nl/(2*np.pi), label=product.replace("_", " "), color=color)
        indices = [i for i in new if rows[i][1] == product]
        selected = ell <= args.lmax
        ls, target = ell[selected].astype(int), nl[selected]
        groups = ls // 25
        ratios, centres = [], []
        for b in np.unique(groups):
            take = groups == b
            weight = 2*ls[take]+1
            ratios.append(np.average(auto[ls[take]][:, indices] / target[take, None], axis=0, weights=weight).mean())
            centres.append(np.average(ls[take], weights=weight))
        axes[1].plot(centres, ratios, marker="o", ms=3, color=color, label=product.replace("_", " "))
        weights = 2*ls+1
        ratio = np.average((auto[ls][:, indices] / target[:, None]).mean(axis=1), weights=weights)
        sigma = np.sqrt(2 / (len(indices) * weights.sum()))
        power_results[product] = dict(mean_power_ratio=float(ratio), expected_sigma=float(sigma))
        checks[product + "_power_within_5sigma"] = bool(abs(ratio - 1) < 5*sigma)
    axes[0].set(xlabel=r"$\ell$", ylabel=r"$\ell(\ell+1)N_\ell/(2\pi)$", title="Full native SO input curves")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color="0.5", ls="--", lw=1)
    axes[1].set(xlabel=r"$\ell$", ylabel=r"Mean $\widehat N_\ell/N_\ell$", title="Actual harmonic draws / input power")
    axes[1].grid(alpha=0.2)
    save(fig, "03_noise_power_and_products")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ls = np.arange(80, args.lmax+1)
    for scheme, ax in zip(("repeated", "independent"), axes):
        indices = [i for i in select(scheme) if rows[i][1] == "baseline_deproj0" and rows[i][2] == "two_param"
                   and rows[i][4] == 2]
        for i in indices:
            ax.plot(ls, ls*(ls+1)*cross[ls, i//2]/(2*np.pi), lw=0.8, alpha=0.8, label=f"Row {rows[i][3]}")
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set(title="Repeated seeds (curves coincide)" if scheme == "repeated" else "Independent row seeds",
               xlabel=r"$\ell$", ylabel=r"Masked noise-only $D_\ell^{A\times B}$")
        ax.legend()
    fig.suptitle("Two noise splits, fixed mask; individual cross spectra can be negative")
    save(fig, "04_signed_cross_noise_spectra")

    painter = np.loadtxt(out / "painter_overlap_errors.csv", delimiter=",")
    checks["ring_locked_painter_matches_serial"] = bool(np.all(painter[:, 2] < 1e-12))
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(painter[:, 0], 100*painter[:, 1], "o-", color="#c4444b", label="Original threaded painter")
    ax.plot(painter[:, 0], 100*painter[:, 2], "s--", color="#248778", label="Ring-locked painter")
    ax.set(xlabel="Repeated run", ylabel="Relative L1 map error vs serial [%]",
           title="Overlapping-halo stress test (not a catalogue bias estimate)", xticks=painter[:, 0])
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "05_threaded_painter_regression")

    g.write_json(out / "validation_report.json", dict(passed=all(checks.values()), checks=checks,
        max_absolute_independent_map_correlation=float(np.max(np.abs(offdiag))),
        actual_noise_maps=len(rows), configured_rows_per_mode=config["n_rows"],
        unique_seeds_all_products_configured_n=seed_count,
        painter_stress_max_legacy_relative_error=float(painter[:, 1].max()),
        painter_stress_max_fixed_relative_error=float(painter[:, 2].max()),
        unique_seeds_all_products_524288=large_seed_count, harmonic_power=power_results,
        diagnostic_nside=args.nside, diagnostic_lmax=args.lmax,
        limitations="Noise-only reduced-resolution checks; not full-resolution halo generation or a survey covariance validation."))
    print(json.dumps(json.loads((out / "validation_report.json").read_text()), indent=2))
    if not all(checks.values()):
        raise RuntimeError("Noise diagnostics failed: inspect validation_report.json")


if __name__ == "__main__":
    main()
