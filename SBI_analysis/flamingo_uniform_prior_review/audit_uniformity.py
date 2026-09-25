"""Read-only diagnosis of prior distortion and loss of old-prior coverage.

This script produces diagnostics only. It does not define a replacement
production prior, run maps, or modify the frozen 8k dataset.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from scipy.stats import qmc


LABELS = [r"$P_{0,0}$", r"$x_{c,0}$", r"$\beta_0$",
          r"$\alpha_{m,P_0}$", r"$\alpha_{m,x_c}$", r"$\alpha_{m,\beta}$",
          r"$\alpha_{z,P_0}$", r"$\alpha_{z,x_c}$", r"$\alpha_{z,\beta}$"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_cuts(module, prior, theta):
    """Attribute rejected rows sequentially, in the frozen gate's order."""
    _, xc, beta = module.evolved(theta, *prior.interpolation_points)
    low, high = prior.config["beta_range_on_interpolation_domain"]
    bmin, bmax = beta.min(1), beta.max(1)
    masks = {"beta_lower": bmin >= low, "beta_upper": bmax <= high}
    # The frozen contains() computes the conservative tail bound for all rows
    # inside the base rectangle, before rejecting any beta failures.
    good, metrics = prior.contains(theta, return_metrics=True)
    masks["central_column_tail"] = metrics["tail"] <= prior.config["maximum_central_column_tail"]
    _, xc, beta = module.evolved(theta, *prior.size_points)
    size = xc / beta / prior.reference_size
    lo, hi = prior.config["size_ratio_to_battaglia12"]
    masks["relative_size"] = (size.min(1) >= lo) & (size.max(1) <= hi)
    lo, hi = prior.config["Y200_ratio_to_battaglia12"]
    masks["finite_Y200"] = (metrics["Y200_min"] >= lo) & (metrics["Y200_max"] <= hi)
    surviving = np.ones(len(theta), dtype=bool)
    stages = []
    for name, mask in masks.items():
        rejected = surviving & ~mask
        surviving &= mask
        stages.append(dict(cut=name, rejected=int(rejected.sum()),
                           rejected_fraction=float(rejected.mean()),
                           cumulative_surviving_fraction=float(surviving.mean())))
    np.testing.assert_array_equal(surviving, good)
    return good, dict(rows=len(theta), accepted=int(good.sum()),
                     acceptance_fraction=float(good.mean()), sequential_cuts=stages,
                     beta_min=float(bmin.min()), beta_max=float(bmax.max()),
                     fraction_with_beta_at_or_below_0p7=float((bmin <= .7).mean()))


def box_extrema(module, prior, low, high):
    """Exact evolved-amplitude extrema over parameter and M,z rectangles."""
    unit = np.array(list(itertools.product((0., 1.), repeat=9)))
    corners = low + unit * (high - low)
    _, xc, beta = module.evolved(corners, *prior.interpolation_points)
    return dict(lower=low.tolist(), upper=high.tolist(),
                beta_min=float(beta.min()), beta_max=float(beta.max()),
                xc_max=float(xc.max()),
                corners_passing_current_cuts=int(prior.contains(corners).sum()),
                corner_count=len(corners), is_validated_production_prior=False)


def plot_marginals(output, prior, current, linear_accepted, old_low, old_high):
    plt.rcParams.update({"font.size":16, "axes.labelsize":21, "xtick.labelsize":14,
                         "ytick.labelsize":14, "legend.fontsize":15,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for k, ax in enumerate(axes.flat):
        edges = np.linspace(prior.low[k], prior.high[k], 33)
        for values, color, style in ((current, ".4", "-"), (linear_accepted, "#0072B2", "-")):
            mass, _ = np.histogram(values[:, k], edges)
            mass = mass / len(values)
            # Repeat edges to show the exact bin masses without smoothing.
            ax.plot(np.repeat(edges, 2)[1:-1], np.repeat(mass, 2), color=color,
                    ls=style, lw=1.8)
        ax.axhline(1 / 32, color="#D55E00", ls="--", lw=2.2)
        for bound in (old_low[k], old_high[k]):
            ax.axvline(bound, color="#882E72", ls=":", alpha=.75)
        ax.set(xlabel=LABELS[k], xlim=(prior.low[k], prior.high[k]), ylim=(0, None))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: "{:g}%".format(100 * value)))
        ax.grid(axis="y", alpha=.15)
        if k % 3 == 0:
            ax.set_ylabel("Probability per linear bin")
    handles = [Line2D([], [], color=".4", lw=2, label="Current 8k design"),
               Line2D([], [], color="#0072B2", lw=2, label="Linear-uniform proposals + current cuts"),
               Line2D([], [], color="#D55E00", ls="--", lw=2, label="Desired uniform marginals"),
               Line2D([], [], color="#882E72", ls=":", lw=2, label="Old prior boundaries")]
    fig.suptitle("Joint rejection reshapes the prior", y=.985, fontsize=23)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, .955),
               ncol=2, frameon=False)
    fig.subplots_adjust(top=.855, bottom=.07, left=.09, right=.985, hspace=.34, wspace=.29)
    for ext in ("png", "pdf"):
        fig.savefig(output / ("uniformity_diagnosis." + ext), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--old-metadata", type=Path, required=True)
    parser.add_argument("--fit-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposal-power", type=int, default=18)
    args = parser.parse_args()
    root = args.production_root.resolve()
    output = args.output.resolve()
    for source in (root, args.old_metadata.resolve().parent, args.fit_summary.resolve().parent):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Use a separate diagnostic output directory")
    paths = [root / "manifest.json", root / "design/theta.npy", root / "code/prior.py",
             args.old_metadata, args.fit_summary]
    before = {str(path): digest(path) for path in paths}
    manifest = json.loads(paths[0].read_text())
    if digest(paths[1]) != manifest["design_sha256"]["theta.npy"]:
        raise ValueError("Production design hash mismatch")
    if digest(paths[2]) != manifest["code_sha256"]["prior.py"]:
        raise ValueError("Production support-code hash mismatch")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("frozen_prior", paths[2])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = module.JointPrior(manifest["prior"])
    current = np.load(paths[1], allow_pickle=False)
    old = json.loads(args.old_metadata.read_text())["verified_bundle"]
    fits = json.loads(args.fit_summary.read_text())
    if not (old["param_names"] == fits["parameter_order"] == prior.names):
        raise ValueError("Parameter ordering mismatch")
    old_low, old_high = np.array(old["prior_low"]), np.array(old["prior_high"])
    unit = qmc.Sobol(9, scramble=True, seed=91015).random_base2(args.proposal_power)
    old_points = old_low + unit * (old_high - old_low)
    _, old_report = summarize_cuts(module, prior, old_points)
    linear_points = prior.low + unit * (prior.high - prior.low)
    linear_good, linear_report = summarize_cuts(module, prior, linear_points)
    old_members = ((current >= old_low) & (current <= old_high)).all(1)
    box_fraction_linear = np.prod((old_high - old_low) / (prior.high - prior.low))
    hull = np.array([old_low, old_high] + list(fits["corrected_parameters"].values()))
    report = dict(parameter_order=prior.names, current_rows=len(current),
                  current_rows_inside_entire_old_box=int(old_members.sum()),
                  current_rows_inside_individual_old_ranges=((current >= old_low) & (current <= old_high)).sum(0).tolist(),
                  old_uniform_probe=old_report, extended_linear_uniform_probe=linear_report,
                  old_box_fraction_of_extended_linear_box=float(box_fraction_linear),
                  expected_old_box_rows_in_8192_unconditioned_linear_draws=float(8192 * box_fraction_linear),
                  full_exploratory_box=box_extrema(module, prior, prior.low, prior.high),
                  minimum_box_containing_old_and_fits=box_extrema(module, prior, hull.min(0), hull.max(0)),
                  maximum_abs_mass_exponent_from_beta_gate=float(np.log(50 / 2.8) / (3.7 * np.log(10))),
                  recommended_uniform_measure="physical parameter values for all nine coordinates",
                  production_modified=False, replacement_prior_defined=False,
                  input_sha256=before, script_sha256=digest(__file__))
    output.mkdir(parents=True, exist_ok=True)
    plot_marginals(output, prior, current, linear_points[linear_good], old_low, old_high)
    if before != {str(path): digest(path) for path in paths}:
        raise RuntimeError("An input changed during the diagnostic")
    report["output_sha256"] = {p.name: digest(p) for p in sorted(output.glob("uniformity_diagnosis.*"))}
    (output / "uniformity_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
