"""Plot the frozen production prior, old SBI bounds and new FLAMINGO fits.

All inputs are read-only. The grey samples are the accepted production design;
the coloured points are effective clean-spectrum fits, not posterior samples.
Run with --help to select a different archived design, fit summary or output.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
import numpy as np


PARAMETERS = ["P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
              "alpha_z_P0", "alpha_z_xc", "alpha_z_beta"]
LABELS = [r"$P_{0,0}$", r"$x_{c,0}$", r"$\beta_0$",
          r"$\alpha_{m,P_0}$", r"$\alpha_{m,x_c}$", r"$\alpha_{m,\beta}$",
          r"$\alpha_{z,P_0}$", r"$\alpha_{z,x_c}$", r"$\alpha_{z,\beta}$"]
# Keep the physical parameter order and the colour convention of earlier plots.
VARIANTS = ["L1_m9", "fgas-8sigma", "Mstar-1sigma"]
FIT_LABELS = ["FLAMINGO fiducial", r"FLAMINGO $f_{\rm gas}-8\sigma$",
              r"FLAMINGO $M_\star-1\sigma$"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "^", "D"]
OLD_COLOR = "#882E72"
SAMPLE_COLOR = "#9C9C9C"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arguments():
    analysis = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--production-root", type=Path, default=analysis /
                        "flamingo_extended_prior/cluster_results/independent_noise_20260915")
    parser.add_argument("--old-metadata", type=Path, default=analysis /
                        "flamingo_tsz/metadata_manifest.json")
    parser.add_argument("--old-comparison", type=Path, default=analysis /
                        "flamingo_tsz/cluster_results/comparison/comparison_summary.json")
    parser.add_argument("--fit-summary", type=Path, default=analysis /
                        "flamingo_cosmology_refit/cluster_results/results/comparison_summary.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "plots")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_inputs(args):
    root = args.production_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    config = manifest["prior"]
    metadata = read_json(args.old_metadata)["verified_bundle"]
    old_comparison = read_json(args.old_comparison)
    summary = read_json(args.fit_summary)
    for order in (manifest["parameter_order"], config["parameter_order"],
                  metadata["param_names"], summary["parameter_order"]):
        if order != PARAMETERS:
            raise ValueError("Parameter ordering differs from the nine-parameter contract")
    if summary["fits_are_posterior_estimates"] or summary["noise_used_for_fitting"]:
        raise ValueError("Expected effective parameter fits to clean spectra")

    # Verify the frozen code and design before importing any production helper.
    input_paths = [manifest_path, args.old_metadata, args.old_comparison, args.fit_summary]
    for directory, key in (("code", "code_sha256"), ("design", "design_sha256")):
        for name, expected in manifest[key].items():
            path = root / directory / name
            if sha256(path) != expected:
                raise ValueError("Frozen input checksum mismatch: {}".format(path))
            input_paths.append(path)
    before = {str(path.resolve()): sha256(path) for path in input_paths}
    # Avoid writing __pycache__ to the frozen production directory.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("frozen_production_prior", root / "code/prior.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = module.JointPrior(config)
    theta = np.load(root / "design/theta.npy", allow_pickle=False)
    if theta.shape != (manifest["accepted_count"], len(PARAMETERS)):
        raise ValueError("Unexpected production-design shape")
    if not prior.contains(theta).all():
        raise ValueError("Some production rows fail the frozen joint prior")

    old_low = np.asarray(metadata["prior_low"], dtype=float)
    old_high = np.asarray(metadata["prior_high"], dtype=float)
    np.testing.assert_array_equal(old_low, old_comparison["original_prior_low"])
    np.testing.assert_array_equal(old_high, old_comparison["original_prior_high"])
    if not np.all((prior.low <= old_low) & (old_low < old_high) & (old_high <= prior.high)):
        raise ValueError("Old bounds do not lie inside the extended rectangular envelope")
    points = [("Battaglia12", module.B12, "black", "*")]
    support = {}
    for name, label, color, marker in zip(VARIANTS, FIT_LABELS, COLORS, MARKERS):
        values = np.asarray(summary["corrected_parameters"][name], dtype=float)
        good, metrics = prior.contains(values, return_metrics=True)
        if not good:
            raise ValueError("New FLAMINGO fit lies outside the current joint prior: " + name)
        outside = (values < old_low) | (values > old_high)
        support[name] = {"inside_current_joint_prior": good, "joint_metrics": metrics,
                         "outside_old_prior_parameters": np.asarray(PARAMETERS)[outside].tolist()}
        points.append((label, values, color, marker))
    return dict(theta=theta, prior=prior, points=points, old_low=old_low, old_high=old_high,
                summary=summary, support=support, before=before)


def axis_limits(data, index):
    low, high = data["prior"].low[index], data["prior"].high[index]
    if index in data["prior"].config["log_uniform_indices"]:
        padding = .025 * np.log(high / low)
        return low * np.exp(-padding), high * np.exp(padding)
    padding = .025 * (high - low)
    return low - padding, high + padding


def format_parameter_axis(ax, data, index, direction="x"):
    axis = ax.xaxis if direction == "x" else ax.yaxis
    getattr(ax, "set_" + direction + "lim")(axis_limits(data, index))
    if index in data["prior"].config["log_uniform_indices"]:
        getattr(ax, "set_" + direction + "scale")("log")
        ticks = [1, 3, 10, 30, 60] if index == 0 else [.1, .3, 1, 4]
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FuncFormatter(lambda value, pos: "{:g}".format(value)))
        axis.set_minor_locator(FixedLocator([]))
    else:
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))


def legend_handles():
    return [Patch(facecolor=SAMPLE_COLOR, alpha=.55, label="8k accepted prior design"),
            Line2D([], [], color=OLD_COLOR, ls="--", lw=2, label="Old SBI prior boundaries"),
            Line2D([], [], color="black", marker="*", ls="none", markersize=14, label="Battaglia12")
            ] + [Line2D([], [], color=c, marker=m, ls="none", markersize=10, label=label)
                 for label, c, m in zip(FIT_LABELS, COLORS, MARKERS)]


def marginal_panel(ax, data, index, bins, compact=False):
    prior = data["prior"]
    low, high = prior.low[index], prior.high[index]
    spacing = np.geomspace if index in prior.config["log_uniform_indices"] else np.linspace
    edges = spacing(low, high, bins + 1)
    # Probability MASS per bin: logarithmic bin widths for P0 and xc, linear
    # widths otherwise. This avoids labelling dP/dlog(theta) as dP/dtheta.
    weights = np.full(len(data["theta"]), 1.0 / len(data["theta"]))
    counts, _, _ = ax.hist(data["theta"][:, index], bins=edges, weights=weights,
                           color=SAMPLE_COLOR, alpha=.55, edgecolor=".35", linewidth=.45)
    np.testing.assert_allclose(counts.sum(), 1.0, rtol=0, atol=2e-12)
    ax.set_ylim(0, counts.max() / .65)
    for bound in (data["old_low"][index], data["old_high"][index]):
        ax.axvline(bound, color=OLD_COLOR, ls="--", lw=1.7, zorder=3)
    # Distinct marker heights separate nearly coincident fits, notably xc.
    for height, (_, theta, color, marker) in zip((.73, .80, .87, .94), data["points"]):
        ax.axvline(theta[index], ymax=height, color=color, lw=.8, alpha=.4, zorder=2)
        ax.plot(theta[index], height, transform=ax.get_xaxis_transform(),
                marker=marker, color=color, ls="none", markersize=8 if compact else 12,
                markeredgecolor="white", markeredgewidth=.7, zorder=6)
    format_parameter_axis(ax, data, index)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: "{:g}%".format(100 * value)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(axis="y", alpha=.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if compact:
        ax.set_yticks([])


def pair_panel(ax, data, x, y, compact=False):
    theta = data["theta"]
    ax.scatter(theta[:, x], theta[:, y], s=2 if compact else 5, alpha=.18,
               color=".48", linewidths=0, rasterized=True)
    old_low, old_high = data["old_low"], data["old_high"]
    ax.add_patch(Rectangle((old_low[x], old_low[y]), old_high[x] - old_low[x],
                           old_high[y] - old_low[y], facecolor="none", edgecolor=OLD_COLOR,
                           linestyle="--", linewidth=1.8, zorder=4))
    for _, point, color, marker in data["points"]:
        size = (100 if marker == "*" else 48) if compact else (230 if marker == "*" else 125)
        ax.scatter(point[x], point[y], s=size, color=color, marker=marker,
                   edgecolors="white", linewidths=.7, zorder=6)
    format_parameter_axis(ax, data, x)
    format_parameter_axis(ax, data, y, "y")


def save_figure(fig, output, name, dpi):
    for extension in ("png", "pdf"):
        fig.savefig(output / (name + "." + extension), dpi=dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def make_figures(data, args):
    plt.rcParams.update({"font.size":16, "axes.labelsize":20, "axes.titlesize":20,
                         "xtick.labelsize":15, "ytick.labelsize":15, "legend.fontsize":14,
                         "pdf.fonttype":42, "ps.fonttype":42, "axes.linewidth":1.0})
    # Primary presentation figure: amplitudes, mass exponents, redshift exponents.
    fig, axes = plt.subplots(3, 3, figsize=(15.8, 12.8))
    for index, ax in enumerate(axes.flat):
        marginal_panel(ax, data, index, args.bins)
        ax.set_xlabel(LABELS[index])
        if index % 3 == 0:
            ax.set_ylabel("Prior mass per bin")
    fig.suptitle("Cosmology-corrected FLAMINGO fits", y=.995, fontsize=23)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.51, .962),
               ncol=3, frameon=False, columnspacing=1.8)
    fig.subplots_adjust(top=.86, bottom=.065, left=.075, right=.985, wspace=.25, hspace=.32)
    save_figure(fig, args.output, "prior_all_parameters", args.dpi)

    # Update the original three projections for direct visual comparison.
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, (x, y) in zip(axes, ((1, 0), (1, 2), (6, 7))):
        pair_panel(ax, data, x, y)
        ax.set(xlabel=LABELS[x], ylabel=LABELS[y])
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.80, bottom=.15, left=.06, right=.985, wspace=.25)
    save_figure(fig, args.output, "joint_prior", args.dpi)

    # Supplementary large-format figure: every one of the 36 parameter pairs.
    fig, axes = plt.subplots(9, 9, figsize=(27, 26))
    for y in range(9):
        for x in range(9):
            ax = axes[y, x]
            if x > y:
                ax.set_visible(False)
                continue
            if x == y:
                marginal_panel(ax, data, x, args.bins, compact=True)
            else:
                pair_panel(ax, data, x, y, compact=True)
            ax.tick_params(axis="both", labelsize=12, pad=3)
            if y == 8:
                ax.set_xlabel(LABELS[x], fontsize=21, labelpad=9)
            else:
                ax.tick_params(labelbottom=False)
            if x == 0 and y > 0:
                ax.set_ylabel(LABELS[y], fontsize=21, labelpad=9)
            elif x != y:
                ax.tick_params(labelleft=False)
    fig.text(.60, .91, "Cosmology-corrected FLAMINGO fits", fontsize=29, ha="center")
    fig.legend(handles=legend_handles(), loc="upper left", bbox_to_anchor=(.42, .885),
               ncol=1, frameon=False, fontsize=22, labelspacing=1.0)
    fig.subplots_adjust(left=.055, right=.99, bottom=.055, top=.985, hspace=.12, wspace=.12)
    save_figure(fig, args.output, "joint_prior_all_parameters", min(args.dpi, 170))


def export_tables(data, args):
    fields = ["parameter", "old_sbi_lower", "old_sbi_upper", "extended_base_lower",
              "extended_base_upper", "accepted_design_min", "accepted_design_max", "Battaglia12"] + VARIANTS
    with (args.output / "prior_and_fits.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, name in enumerate(PARAMETERS):
            values = [name, data["old_low"][index], data["old_high"][index],
                      data["prior"].low[index], data["prior"].high[index],
                      data["theta"][:, index].min(), data["theta"][:, index].max()]
            values.extend(point[index] for _, point, _, _ in data["points"])
            writer.writerow(dict(zip(fields, values)))


def main():
    args = arguments()
    if args.bins < 4 or args.dpi < 72:
        raise ValueError("Use at least 4 bins and 72 dpi")
    args.output = args.output.resolve()
    # Never write plots into a production input directory, including its parent.
    for source in (args.production_root, args.old_metadata.parent, args.fit_summary.parent):
        source = source.resolve()
        if source == args.output or source in args.output.parents or args.output in source.parents:
            raise ValueError("Select a separate output directory outside the input roots")
    data = load_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    make_figures(data, args)
    export_tables(data, args)
    after = {path: sha256(path) for path in data["before"]}
    if after != data["before"]:
        raise RuntimeError("An input changed while plots were generated")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "parameter_order": PARAMETERS, "accepted_design_rows": len(data["theta"]),
              "old_prior_source": "verified 40-bin nine-parameter SBI bundle support",
              "flamingo_points": "selected clean-spectrum fits including approximate abundance and geometry response",
              "points_are_posterior_constraints": False,
              "marginal_histograms": "probability mass; log-spaced P0 and xc bins, linear bins otherwise",
              "bins_per_parameter": args.bins, "full_corner_pair_count": 36,
              "production_inputs_unchanged": True, "input_sha256": data["before"],
              "plotter_sha256": sha256(__file__), "fit_support": data["support"],
              "optimizer_status": {item["variant"]: {
                  "success": item["selected_optimizer_success"],
                  "message": item["selected_optimizer_message"]} for item in data["summary"]["variants"]},
              "output_sha256": {path.name: sha256(path) for path in sorted(args.output.iterdir())
                                  if path.suffix in (".png", ".pdf", ".csv")}}
    write_json(args.output / "plot_provenance.json", report)
    print(json.dumps({"output": str(args.output), "rows_checked": len(data["theta"]),
                      "parameters": len(PARAMETERS), "pairs": 36,
                      "verified_unchanged_inputs": len(after), "fit_support": data["support"]}, indent=2))


if __name__ == "__main__":
    main()
