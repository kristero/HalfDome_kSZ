"""Create guarded uniform-prior plots and a fully accepted parameter design.

Reads saved comparison inputs only. Never changes the nine-parameter campaign,
submits cluster jobs, or refits FLAMINGO. Run with --help for paths/counts.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np

from flat_prior import FlatPrior, HERE

LABELS = [r"$P_{0,0}$", r"$\beta_0$"]
OLD_COLOR = "#882E72"
GRAY = "#D5D8DC"
VARIANTS = ["L1_m9", "fgas-8sigma", "Mstar-1sigma"]
FIT_LABELS = ["FLAMINGO fiducial", r"FLAMINGO $f_{\rm gas}-8\sigma$", r"FLAMINGO $M_\star-1\sigma$"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "^", "D"]


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arguments():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference-root", type=Path, default=HERE.parent / "flamingo_linear_prior/cluster_results")
    parser.add_argument("--output", type=Path, default=HERE / "outputs_guarded")
    parser.add_argument("--count", type=int, default=8192, help="Parameter-design rows, not simulated maps")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def load_references(root, prior):
    paths = [root / "inputs/old_metadata.json", root / "inputs/old_comparison.json",
             root / "inputs/flamingo_fits.json", root / "code/prior.py", root / "code/prior.json"]
    review = read_json(root / "review_manifest.json")
    for path in paths:
        if sha256(path) != review[path.relative_to(root).as_posix()]:
            raise ValueError("Archived input hash mismatch: " + str(path))
    metadata = read_json(paths[0])["verified_bundle"]
    comparison, fits = read_json(paths[1]), read_json(paths[2])
    order = prior.config["full_parameter_order"]
    if metadata["param_names"] != order or fits["parameter_order"] != order:
        raise ValueError("Reference parameter order mismatch")
    if fits["fits_are_posterior_estimates"] or fits["noise_used_for_fitting"]:
        raise ValueError("Expected saved clean-spectrum effective fits")
    np.testing.assert_array_equal(metadata["prior_low"], comparison["original_prior_low"])
    np.testing.assert_array_equal(metadata["prior_high"], comparison["original_prior_high"])
    active = [order.index(name) for name in prior.names]
    old_low = np.asarray(metadata["prior_low"])[active]
    old_high = np.asarray(metadata["prior_high"])[active]
    reference = np.array([prior.config["reference_parameters"][name] for name in prior.names])
    points = [("Battaglia12", reference, "black", "*")]
    for variant, label, color, marker in zip(VARIANTS, FIT_LABELS, COLORS, MARKERS):
        point = np.asarray(fits["corrected_parameters"][variant])[active]
        if not prior.contains(point):
            raise ValueError("Reference point fails the guarded two-parameter prior")
        points.append((label, point, color, marker))
    spec = importlib.util.spec_from_file_location("archived_nine_parameter_guards", paths[3])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    np.testing.assert_array_equal(prior.expand(reference), module.B12)
    guards = module.JointPrior(read_json(paths[4]))
    return old_low, old_high, points, guards, {str(p.resolve()): sha256(p) for p in paths}


def legend_handles():
    return [Patch(facecolor=GRAY, edgecolor=".4", label="Uniform guarded prior"),
            Line2D([], [], color=OLD_COLOR, ls="--", lw=2, label="Old SBI prior boundaries"),
            Line2D([], [], color="black", marker="*", ls="none", markersize=13, label="Battaglia12")
            ] + [Line2D([], [], color=c, marker=m, ls="none", markersize=10, label=label)
                 for label, c, m in zip(FIT_LABELS, COLORS, MARKERS)]


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def joint_panel(ax, prior, old_low, old_high, points, fill=True):
    if fill:
        beta = np.linspace(*prior.beta_limits, 1200)
        lower, upper = prior.amplitude_bounds(beta).T
        ax.fill_betweenx(beta, lower, upper, facecolor=GRAY, edgecolor=".4", lw=1.6)
    ax.add_patch(Rectangle(old_low, *(old_high-old_low), fill=False,
                           edgecolor=OLD_COLOR, ls="--", lw=2, zorder=4))
    for _, values, color, marker in points:
        ax.scatter(*values, color=color, marker=marker, s=220 if marker == "*" else 120,
                   edgecolor="white", linewidth=.8, zorder=6)
    ax.set(xlabel=LABELS[0], ylabel=LABELS[1],
           xlim=(prior.low[0]-1.5, prior.high[0]+1.5),
           ylim=(prior.low[1]-.35, prior.high[1]+.35))
    style_axis(ax)


def marginal_panel(ax, prior, old_low, old_high, points, index):
    # Integrate constant joint density over the permitted other coordinate.
    # The small Y200 corner cut means marginals are nearly, not exactly, flat.
    if index == 0:
        low, high = prior.low[0], prior.high[0]
        coords = np.unique(np.r_[np.linspace(low, high, 400), np.linspace(low, min(high,low+.2), 160)])
        density = np.array([np.diff(prior.beta_bounds(p))[0] for p in coords]) / prior.area
    else:
        low, high = prior.beta_limits
        coords = np.linspace(low, high, 800)
        density = np.diff(prior.amplitude_bounds(coords), axis=-1).ravel() / prior.area
    ax.fill_between(coords, density, color=GRAY)
    ax.plot(np.r_[low,coords,high], np.r_[0,density,0], color=".35", lw=2)
    for boundary in (old_low[index], old_high[index]):
        ax.axvline(boundary, color=OLD_COLOR, ls="--", lw=1.7)
    for height, (_, point, color, marker) in zip((.75, .81, .87, .93), points):
        ax.axvline(point[index], ymax=height, color=color, lw=.8, alpha=.45)
        ax.plot(point[index], height, transform=ax.get_xaxis_transform(), marker=marker,
                ls="none", color=color, markersize=11, markeredgecolor="white", markeredgewidth=.6)
    ax.set(xlabel=LABELS[index], ylabel="Probability density",
           xlim=(prior.low[index]-.03*prior.width[index], prior.high[index]+.03*prior.width[index]),
           ylim=(0, density.max()/.65))
    style_axis(ax)


def save(fig, output, name, dpi):
    for extension in ("png", "pdf"):
        fig.savefig(output / (name + "." + extension), dpi=dpi, bbox_inches="tight",
                    pad_inches=.12, facecolor="white")
    plt.close(fig)


def make_figures(args, prior, old_low, old_high, points, grid):
    plt.rcParams.update({"font.size":16, "axes.labelsize":21, "axes.titlesize":20,
                         "xtick.labelsize":15, "ytick.labelsize":15, "legend.fontsize":14,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.1))
    joint_panel(axes[0], prior, old_low, old_high, points)
    for index, ax in enumerate(axes[1:]):
        marginal_panel(ax, prior, old_low, old_high, points, index)
    fig.suptitle(r"Guarded $P_0$–$\beta$ prior", fontsize=24, y=.99)
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(.52,.93), ncol=3, frameon=False)
    fig.subplots_adjust(top=.73, bottom=.16, left=.055, right=.985, wspace=.32)
    save(fig, args.output, "prior_comparison", args.dpi)

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    joint_panel(ax, prior, old_low, old_high, points)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=.78, bottom=.11, left=.12, right=.97)
    save(fig, args.output, "joint_prior", args.dpi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    for index, ax in enumerate(axes):
        marginal_panel(ax, prior, old_low, old_high, points, index)
    fig.legend(handles=legend_handles(), loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.76, bottom=.16, left=.09, right=.985, wspace=.25)
    save(fig, args.output, "marginal_priors", args.dpi)

    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    ax.pcolormesh(grid["p0_edges"], grid["beta_edges"], grid["passes"],
                  cmap=ListedColormap(["#F3D8D5", GRAY]), vmin=0, vmax=1, shading="flat", rasterized=True)
    joint_panel(ax, prior, old_low, old_high, points, fill=False)
    ax.axhline(grid["beta_min_from_slope"], color="#9F5149", ls=":", lw=1.5)
    ax.axhline(grid["beta_max_from_size"], color="#9F5149", ls=":", lw=1.5)
    handles = [Patch(facecolor=GRAY, label="Included in the prior"),
               Patch(facecolor="#F3D8D5", label="Excluded by guardrails")]
    fig.legend(handles=handles + legend_handles()[1:], loc="upper center", ncol=2, frameon=False)
    ax.set_title("Guardrails applied", pad=14)
    fig.subplots_adjust(top=.72, bottom=.11, left=.12, right=.97)
    save(fig, args.output, "guarded_support", args.dpi)

    # Resolve the small low-P0/high-beta exclusion that a full-range plot hides.
    fig, ax = plt.subplots(figsize=(8.5,6.5))
    beta = np.linspace(10.3, prior.beta_limits[1], 500)
    lower = prior.amplitude_bounds(beta)[:,0]
    ax.fill_betweenx(beta, 1, lower, color="#F3D8D5", label="Excluded by $Y_{200}$")
    ax.fill_betweenx(beta, lower, 1.14, color=GRAY, label="Included")
    ax.plot(lower, beta, color=".3", lw=2)
    ax.set(xlabel=LABELS[0], ylabel=LABELS[1], xlim=(.995,1.14), ylim=(10.3,10.9),
           title="Integrated-pressure boundary")
    ax.axhline(prior.beta_limits[1], color=".35", ls="--", lw=1.5)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    save(fig,args.output,"low_amplitude_boundary",args.dpi)


def guard_diagnostics(prior, guards):
    # Equal-area cells illustrate exclusions. Sampling uses the exact guard.
    p0_edges = np.linspace(*[prior.low[0], prior.high[0]], 257)
    beta_edges = np.linspace(*[prior.low[1], prior.high[1]], 257)
    p0, beta = np.meshgrid((p0_edges[1:]+p0_edges[:-1])/2, (beta_edges[1:]+beta_edges[:-1])/2)
    theta = np.column_stack((p0.ravel(), beta.ravel()))
    good = guards.contains(prior.expand(theta))
    np.testing.assert_array_equal(good, prior.contains(theta))
    beta_min, beta_max = prior.beta_limits
    return dict(p0_edges=p0_edges, beta_edges=beta_edges, passes=good.reshape(p0.shape),
                beta_min_from_slope=beta_min, beta_max_from_size=beta_max)


def main():
    args = arguments()
    if args.count < 1:
        raise ValueError("--count must be positive")
    args.reference_root = args.reference_root.resolve()
    args.output = args.output.resolve()
    if args.output == args.reference_root or args.reference_root in args.output.parents:
        raise ValueError("Output must remain outside the frozen reference campaign")
    args.output.mkdir(parents=True, exist_ok=True)
    prior = FlatPrior()
    old_low, old_high, points, guards, inputs_before = load_references(args.reference_root, prior)
    theta, proposal_ids = prior.sobol(args.count, return_proposal_ids=True)
    full = prior.expand(theta)
    np.save(args.output / "theta_P0_beta.npy", theta)
    np.save(args.output / "theta_nine_parameters.npy", full)
    np.save(args.output / "proposal_ids.npy", proposal_ids)
    np.savetxt(args.output / "parameters.csv", full, delimiter=",", header=",".join(prior.config["full_parameter_order"]), comments="")
    with (args.output / "prior_and_reference_values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["parameter", "proposal_lower", "proposal_upper", "guarded_envelope_lower", "guarded_envelope_upper", "old_lower", "old_upper", "Battaglia12"] + VARIANTS)
        for index, name in enumerate(prior.names):
            bounds = [prior.low[0],prior.high[0]] if index == 0 else prior.beta_limits.tolist()
            writer.writerow([name, prior.low[index], prior.high[index]] + bounds + [old_low[index], old_high[index]] + [point[1][index] for point in points])
    grid = guard_diagnostics(prior, guards)
    make_figures(args, prior, old_low, old_high, points, grid)
    np.savez_compressed(args.output / "guarded_support_grid.npz", **grid)
    beta = np.linspace(*prior.beta_limits, 1200)
    np.savetxt(args.output / "support_boundary.csv", np.column_stack((beta,prior.amplitude_bounds(beta))),
               delimiter=",", header="beta,P0_lower,P0_upper", comments="")

    # Check agreement with the frozen production guards, support normalization,
    # stable accepted prefix, fixed values, and rejection of the small Y corner.
    assert prior.contains(theta).all()
    np.testing.assert_array_equal(theta, prior.sobol(args.count + 1)[:args.count])
    np.testing.assert_allclose(prior.log_prob(theta), -np.log(prior.area), rtol=0, atol=1e-14)
    assert np.isneginf(prior.log_prob([0, 4]))
    for name, value in prior.config["fixed_parameters"].items():
        np.testing.assert_array_equal(full[:,prior.config["full_parameter_order"].index(name)], np.full(args.count,value))
    # All four former rectangle corners must now be rejected.
    corners = np.array([[p,b] for p in (1,60) for b in (2.8,16)])
    assert not prior.contains(corners).any()
    assert not prior.contains([1.,10.8])
    assert prior.contains([1.1,10.8])
    design_passes = guards.contains(full)
    assert design_passes.all()
    dense_config = dict(guards.config, Y200_grid_size=33)
    dense_passes = type(guards)(dense_config).contains(full)
    assert dense_passes.all()
    # Independent normalization integral in the other coordinate resolves the
    # narrow P0 corner explicitly instead of relying on a coarse 2D mesh.
    from scipy.integrate import quad
    corner_p0 = float(prior.amplitude_bounds(prior.beta_limits[1])[0])
    independent_area, independent_error = quad(lambda p: float(np.diff(prior.beta_bounds(p))[0]),
        prior.low[0], prior.high[0], points=[corner_p0], epsabs=1e-8, epsrel=1e-10)
    np.testing.assert_allclose(prior.area, independent_area, rtol=1e-9, atol=1e-8)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), design_rows=args.count,
        parameter_order=prior.names, fixed_parameters=prior.config["fixed_parameters"],
        prior_area=prior.area, joint_density=1/prior.area, log_density=-np.log(prior.area),
        normalizing_mass=prior.normalizing_mass, beta_limits=prior.beta_limits.tolist(),
        P0_lower_at_beta_upper=corner_p0, beta_upper_at_P0_one=float(prior.beta_bounds(1)[1]),
        area_integration_error=prior.area_integration_error, independent_area=independent_area,
        independent_area_error=independent_error, all_former_corners_rejected=True,
        marginals="Computed from permitted cross-section widths; approximately flat with a small low-P0/high-beta exclusion",
        rejection_cuts_applied=True, sobol_prefix_check="passed", fixed_coordinates_check="passed",
        old_guard_diagnostic=dict(applied_to_prior=True, design_passing=int(design_passes.sum()),
            design_failing=int((~design_passes).sum()), equal_area_grid_size=[256,256],
            grid_fraction_passing=float(grid["passes"].mean()), beta_min_from_slope=grid["beta_min_from_slope"],
            beta_max_from_size=grid["beta_max_from_size"], denser_33x33_passing=int(dense_passes.sum())),
        flamingo_markers=prior.config["flamingo_marker_interpretation"],
        reference_points=[dict(label=label,parameters=values.tolist()) for label,values,_,_ in points],
        old_prior_comparison="Projected support bounds from the verified nine-parameter 40-bin SBI bundle; no old histogram or density is inferred",
        full_map_validation_performed=False, cluster_jobs_submitted=False,
        source_input_sha256=inputs_before,
        source_code_sha256={p.name:sha256(p) for p in (HERE/"prior.json",HERE/"flat_prior.py",HERE/"make_plots.py",
                                                      HERE/"guardrails.py",HERE/"guardrails.json",HERE/"torch_prior.py")})
    for name, expected in inputs_before.items():
        assert sha256(Path(name)) == expected, "Read-only input changed: " + name
    write_json(args.output / "validation.json", report)
    artifacts = {p.name:sha256(p) for p in args.output.iterdir() if p.is_file() and p.name != "artifact_sha256.json"}
    write_json(args.output / "artifact_sha256.json", artifacts)
    print(json.dumps({key:report[key] for key in ("design_rows","prior_area","joint_density","old_guard_diagnostic","cluster_jobs_submitted")},indent=2))


if __name__ == "__main__":
    main()
