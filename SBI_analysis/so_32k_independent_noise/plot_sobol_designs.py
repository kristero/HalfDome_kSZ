#!/usr/bin/env python3
"""Inspect actual Sobol parameter tables, not posterior samples or noisy spectra."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from generate import BUNDLE, MODES, NAMES, load_designs, sha256, validate_config, write_json

LABELS = (r"$P_0$", r"$x_{\rm c}$", r"$\beta$", r"$\alpha_{m,P_0}$", r"$\alpha_{m,x_{\rm c}}$",
          r"$\alpha_{m,\beta}$", r"$\alpha_{z,P_0}$", r"$\alpha_{z,x_{\rm c}}$", r"$\alpha_{z,\beta}$")
COLORS = {"nine_param": "#27788e", "two_param": "#cf7735"}


def read_table(path, n):
    with path.open(newline="") as stream:
        if tuple(next(csv.reader(stream))) != NAMES:
            raise ValueError(f"Wrong column order: {path}")
    theta = np.loadtxt(path, delimiter=",", skiprows=1, max_rows=n, ndmin=2)
    if theta.shape != (n, 9):
        raise ValueError(f"Need {n} rows and nine columns in {path}; found {theta.shape}")
    return theta


def design_summary(theta, mode, config, bins=32):
    low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
    if theta.ndim != 2 or theta.shape[1] != 9 or not np.isfinite(theta).all():
        raise ValueError("Expected finite (N,9) parameters")
    if np.any(theta < low - 1e-12) or np.any(theta > high + 1e-12):
        raise ValueError(f"Out-of-prior values in {mode}; no clipping is performed")
    varying = [0, 2] if mode == "two_param" else list(range(9))
    if np.flatnonzero(np.ptp(theta, axis=0) > 0).tolist() != varying:
        raise ValueError(f"Incorrect varying columns for {mode}")
    fixed = [i for i in range(9) if i not in varying]
    if fixed:
        np.testing.assert_allclose(theta[:, fixed], np.broadcast_to(
            np.array(config["fiducial"])[fixed], (len(theta), len(fixed))), atol=1e-12, rtol=0)
    unique = len(np.unique(theta, axis=0))
    if unique != len(theta):
        raise ValueError(f"{mode} has {len(theta)-unique} duplicate parameter rows")
    unit = (theta - low) / (high - low)
    i = np.arange(1, len(theta)+1) / len(theta)
    rows = []
    for j, name in enumerate(NAMES):
        counts, _ = np.histogram(theta[:, j], bins=np.linspace(low[j], high[j], bins+1))
        if j in varying:
            ordered = np.sort(unit[:, j])
            # A descriptive CDF distance, NOT an iid Kolmogorov-Smirnov p-value.
            distance = max(np.max(i - ordered), np.max(ordered - (i - 1/len(theta))))
        else:
            distance = None
        rows.append(dict(param=name, varying=j in varying, prior_low=float(low[j]), prior_high=float(high[j]),
            sample_min=float(theta[:, j].min()), sample_max=float(theta[:, j].max()),
            mean=float(theta[:, j].mean()), std=float(theta[:, j].std()),
            min_bin_count=int(counts.min()), max_bin_count=int(counts.max()),
            max_abs_cdf_minus_uniform=float(distance) if distance is not None else None))
    correlation = np.corrcoef(unit[:, varying], rowvar=False)
    offdiag = correlation[~np.eye(len(varying), dtype=bool)]
    return dict(n_rows=len(theta), n_unique_rows=unique, histogram_bins=bins, parameters=rows,
                varying_params=[NAMES[j] for j in varying], correlation=correlation.tolist(),
                max_abs_offdiagonal_correlation=float(np.max(np.abs(offdiag))))


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180)
    fig.savefig(output / (name + ".pdf"))
    plt.close(fig)


def marginal_plot(tables, n, config, output):
    low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        edges = np.linspace(low[j], high[j], 33)
        centres = (edges[:-1] + edges[1:])/2
        for mode in ("nine_param", "two_param"):
            if mode == "two_param" and j not in (0, 2):
                ax.axvline(config["fiducial"][j], color=COLORS[mode], lw=2, label="2-param: fixed")
                continue
            count, _ = np.histogram(tables[mode][:n, j], bins=edges)
            ax.bar(centres, 100*count/n, width=np.diff(edges), color=COLORS[mode], alpha=0.15)
            ax.step(edges, 100*np.r_[count, count[-1]]/n, where="post", color=COLORS[mode],
                    lw=1.5, ls="-" if mode == "nine_param" else "--",
                    label="9-param: varied" if mode == "nine_param" else "2-param: varied")
        ax.axvline(config["fiducial"][j], color="black", ls=":", lw=1.1, label="Battaglia12")
        ax.set(xlabel=LABELS[j], ylabel="Rows per bin [%]", xlim=(low[j], high[j]), ylim=(0, 4.2))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="y", alpha=0.15)
    axes[0, 0].legend(fontsize=8, loc="lower center")
    axes[0, 1].legend(fontsize=8, loc="lower center")
    fig.suptitle(f"Actual Sobol parameter distributions: {n:,} rows per design\n"
                 "32 equal-width bins; uniform varied parameters give 3.125% per bin", fontsize=13)
    save(fig, output, f"sobol_marginals_N{n}")


def corner_plot(theta, n, config, output, max_points):
    low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
    k = min(n, max_points)
    fig, axes = plt.subplots(9, 9, figsize=(13, 13))
    fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.95, hspace=0.06, wspace=0.06)
    for row in range(9):
        for col in range(9):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if col == row:
                count, edges = np.histogram(theta[:n, col], np.linspace(low[col], high[col], 33))
                ax.step(edges, 100*np.r_[count, count[-1]]/n, where="post", color=COLORS["nine_param"])
                ax.set_ylim(0, 4.2)
                ax.set_yticks([])
            else:
                ax.scatter(theta[:k, col], theta[:k, row], s=1.5, alpha=0.55, rasterized=True,
                           color=COLORS["nine_param"], edgecolors="none")
                ax.set_ylim(low[row], high[row])
                ax.axhline(config["fiducial"][row], color="black", ls=":", lw=0.6)
                ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.axvline(config["fiducial"][col], color="black", ls=":", lw=0.6)
            ax.set_xlim(low[col], high[col])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=7, labelbottom=row == 8, labelleft=col == 0 and row > 0)
            if row == 8:
                ax.set_xlabel(LABELS[col], fontsize=10)
            if col == 0 and row > 0:
                ax.set_ylabel(LABELS[row], fontsize=10)
    fig.suptitle(f"9-parameter Sobol design: N={n:,} (not posterior constraints)\n"
                 f"Diagonal: all rows; joint panels: first {k:,} rows; dotted lines: Battaglia12", fontsize=13)
    save(fig, output, f"sobol_nine_param_corner_N{n}")


def coverage_plot(tables, sizes, config, output):
    counts = sorted(set([min(sizes), max(sizes), *[v for v in (256, 2048) if v <= max(sizes)]]))
    fig, axes = plt.subplots(2, len(counts), figsize=(3.1*len(counts), 6.4), squeeze=False,
                             constrained_layout=True)
    low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
    for r, mode in enumerate(MODES):
        for c, n in enumerate(counts):
            ax = axes[r, c]
            points = tables[mode][:n]
            if n <= 2048:
                ax.scatter(points[:, 0], points[:, 2], s=3, alpha=0.7, color=COLORS[mode], rasterized=True)
            else:
                hist, ex, ey = np.histogram2d(points[:, 0], points[:, 2], bins=(
                    np.linspace(low[0], high[0], 65), np.linspace(low[2], high[2], 65)))
                im = ax.pcolormesh(ex, ey, hist.T / (n/4096), cmap="Blues", vmin=0, vmax=2, rasterized=True)
            ax.axvline(config["fiducial"][0], color="black", ls=":", lw=0.8)
            ax.axhline(config["fiducial"][2], color="black", ls=":", lw=0.8)
            ax.set(xlim=(low[0], high[0]), ylim=(low[2], high[2]), xlabel=LABELS[0],
                   ylabel=("2-param\n" if r == 0 else "9-param\n") + LABELS[2], title=f"N={n:,}")
    if any(n > 2048 for n in counts):
        fig.colorbar(im, ax=axes, label="Count / uniform expectation (density panels only)", shrink=0.7)
    fig.suptitle("P0-beta Sobol coverage: points for small prefixes, all rows for density panels", fontsize=13)
    save(fig, output, "sobol_P0_beta_prefix_coverage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE / "config.json")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--two-param-csv", type=Path)
    parser.add_argument("--nine-param-csv", type=Path)
    parser.add_argument("--output", type=Path, default=BUNDLE / "sobol_plots")
    parser.add_argument("--corner-points", type=int, default=1024)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    sizes = sorted(set(args.sizes or [config["n_rows"]]))
    if any(n < 2 or n & (n-1) for n in sizes) or args.corner_points < 1:
        parser.error("Use power-of-two sample sizes >=2 and positive --corner-points")
    if bool(args.two_param_csv) != bool(args.nine_param_csv):
        parser.error("Supply both full CSV paths or neither")
    shipped = load_designs(config)
    sources = dict(zip(MODES, (args.two_param_csv, args.nine_param_csv))) if args.two_param_csv else {
        mode: value[0] for mode, value in shipped.items()}
    tables, report = {}, dict(config=config, sources={}, designs={}, note=
        "Parameter-design diagnostics only. Sobol rows are not iid; no statistical p-values are assigned.")
    for mode, path in sources.items():
        theta = read_table(path, max(sizes))
        prefix_n = min(len(shipped[mode][1]), len(theta))
        if not np.array_equal(theta[:prefix_n], shipped[mode][1][:prefix_n]):
            raise ValueError(f"{path} does not extend the configured design exactly")
        tables[mode] = theta
        report["sources"][mode] = dict(path=str(path.resolve()), sha256=sha256(path),
            verified_prefix_rows=prefix_n, source_row_offset=config["sequence_offset"])
        for n in sizes:
            report["designs"][f"{mode}_N{n}"] = design_summary(theta[:n], mode, config)
            print(f"Validated {mode}: N={n:,}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                         "savefig.bbox": "tight"})
    for n in sizes:
        marginal_plot(tables, n, config, args.output)
        corner_plot(tables["nine_param"], n, config, args.output, args.corner_points)
    coverage_plot(tables, sizes, config, args.output)
    write_json(args.output / "sobol_distribution_report.json", report)
    rows = [dict(design=key, **p) for key, value in report["designs"].items() for p in value["parameters"]]
    with (args.output / "sobol_parameter_statistics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved plots and statistics:", args.output)


if __name__ == "__main__":
    main()
