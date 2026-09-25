"""Illustrate the implemented tSZ algorithms without allocating a full sky.

Boundary curves evaluate the same analytic chord-mean functions as the Julia
code at two fixed profile shapes. Pixel drawings use actual HEALPix geometry.
The scheduling timeline and 1D beam example are explicitly illustrative.
Measured errors/times are loaded separately from the completed cluster audit.
"""
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import re

import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "tsz_reuse_cache_20260921"
COMPLETED = ROOT.parent / "tsz_reuse_cache_completed_20260922"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
BLUE, ORANGE, DARK = "#197ca3", "#c66b27", "#343d46"
plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
                    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
                    "savefig.dpi": 210, "svg.fonttype": "none"})
META = {"utc": datetime.now(timezone.utc).isoformat(), "full_sky_computation": False}


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(PLOTS / (name + "." + extension), bbox_inches="tight")
    plt.close(fig)


def pressure(radius, xc, beta_raw, gamma=-.3):
    """Fixed alpha=1: p=(r/xc)^gamma (1+r/xc)^(-beta_raw)."""
    return np.exp(gamma * np.log(radius / xc) - beta_raw * np.log1p(radius / xc))


def pressure_derivative(radius, xc, beta_raw):
    return pressure(radius, xc, beta_raw) * (-.3 / radius - beta_raw / (xc + radius))


def chord_mean(x, xc, beta_raw, nodes=128):
    if nodes == 16:
        rule = np.loadtxt(SOURCE / "edge_extension/legendre16.csv", delimiter=",")
        u, weights = rule[:, 0], rule[:, 1]
    else:
        points, weights = leggauss(nodes)
        u, weights = (points + 1) / 2, weights / 2
    x = np.atleast_1d(x)
    radius = np.sqrt(x[:, None] ** 2 * (1 - u ** 2) + 16 * u ** 2)
    value = pressure(radius, xc, beta_raw) @ weights
    derivative = (pressure_derivative(radius, xc, beta_raw) * x[:, None] * (1 - u**2) / radius) @ weights
    return value, derivative


def sphere():
    fig, ax = plt.subplots(figsize=(8.5, 5.3), layout="constrained")
    ax.add_patch(Circle((0, 0), 4, facecolor="#e4f0f6", edgecolor=BLUE, lw=2))
    b = 2.3
    length = np.sqrt(16 - b*b)
    ax.plot([b, b], [-5, 5], color="#999999", ls="--", lw=2)
    ax.plot([b, b], [-length, length], color=BLUE, lw=5)
    ax.plot([0, b], [0, 0], color=DARK, lw=1.4)
    ax.scatter([0], [0], color=DARK, s=35)
    ax.annotate(r"$x=R_\perp/R_{200c}$", (b/2, 0), xytext=(.4, -.8))
    ax.annotate(r"$2L=2\sqrt{X^2-x^2}$", (b, .8), xytext=(3.1, 1.5),
                arrowprops={"arrowstyle": "->", "color": DARK})
    ax.text(-3.7, 2.5, r"$X=4$", fontsize=20)
    ax.text(-3.6, -2.8, "Gas inside sphere", color=BLUE)
    ax.text(2.55, 4.15, "Outside: excluded", color="#777777", fontsize=14)
    ax.set(xlabel=r"Transverse distance / $R_{200c}$", ylabel=r"Line-of-sight distance / $R_{200c}$",
           xlim=(-4.7, 7), ylim=(-5, 5), aspect="equal")
    ax.grid(alpha=.15)
    save(fig, "spherical_chord")


def boundary():
    x = np.linspace(2.8, 5.2, 601)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8), sharex=True, layout="constrained")
    rows, checks = [], []
    for row, (xc, beta_raw, title) in enumerate([(.497, 4.35, "Battaglia12 shape"), (4., 2.8, "Bright / shallow shape")]):
        mean, derivative = chord_mean(x, xc, beta_raw)
        exterior16, derivative16 = chord_mean(x, xc, beta_raw, nodes=16)
        old = np.where(x < 4, mean, pressure(x, xc, beta_raw))
        old_derivative = np.where(x < 4, derivative, pressure_derivative(x, xc, beta_raw))
        new = np.where(x < 4, mean, exterior16)
        new_derivative = np.where(x < 4, derivative, derivative16)
        edge = float(pressure(4, xc, beta_raw))
        physical = 2*np.sqrt(np.maximum(16 - x*x, 0))*mean
        expected_derivative = 2/3 * pressure_derivative(4, xc, beta_raw)
        edge_value, edge_derivative = chord_mean(np.array([4.]), xc, beta_raw, nodes=16)
        assert np.isclose(edge_value[0], edge, rtol=1e-13)
        assert np.isclose(edge_derivative[0], expected_derivative, rtol=1e-13)
        outside_error = float(np.max(np.abs(exterior16[x >= 4] / mean[x >= 4] - 1)))
        assert outside_error < 1e-10
        assert np.all(physical[x >= 4] == 0)
        for column, a, b in [(0, old/edge, new/edge), (1, 4*old_derivative/edge, 4*new_derivative/edge)]:
            axes[row, column].plot(x/4, a, color=ORANGE, lw=2.6, label="Previous cache")
            axes[row, column].plot(x/4, b, color=BLUE, lw=2, ls="--", label="Smooth continuation")
        axes[row, 2].plot(x/4, physical/(8*edge), color=DARK, lw=2.5, label="Same physical projection")
        for column, ax in enumerate(axes[row]):
            ax.axvspan(1, 1.3, color="#aaaaaa", alpha=.15)
            ax.axvline(1, color="#777777", lw=1, ls=":")
            ax.set_title([title + ": cache", "Derivative", "Painted signal"][column])
            ax.set_xlim(.7, 1.3); ax.grid(alpha=.2)
        axes[row, 0].set_ylabel(r"$h(x)/p(X)$")
        axes[row, 1].set_ylabel(r"$Xh'(x)/p(X)$")
        axes[row, 2].set_ylabel(r"$y(x)/[2AXp(X)]$")
        checks.append(dict(shape=title, xc=xc, beta_raw=beta_raw, gamma=-.3, alpha=1,
                           edge_derivative=float(expected_derivative), exterior_rule_relative_error=outside_error))
        rows.extend(zip(np.full(len(x), row), x, old, new, old_derivative, new_derivative, physical))
    axes[0, 0].legend(fontsize=11); axes[0, 2].legend(fontsize=11)
    for ax in axes[-1]: ax.set_xlabel(r"Projected radius $x/X$")
    save(fig, "smooth_boundary")
    np.savetxt(ROOT / "boundary_curves.csv", np.array(rows), delimiter=",",
               header="shape_index,x,previous_h,smooth_h,previous_dh_dx,smooth_dh_dx,physical_y_over_A", comments="")
    META["boundary_checks"] = checks


def cache_counts():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    old, new = np.array([512, 256, 128]), np.array([256, 128, 64])
    x = np.arange(3)
    for offset, values, color, name in [(-.18, old, ORANGE, "Default"), (.18, new, BLUE, "Half all")]:
        bars = axes[0].bar(x + offset, values, .34, color=color, label=name)
        axes[0].bar_label(bars, fontsize=14, padding=3)
    axes[0].set(xticks=x, xticklabels=[r"$\ln\theta$", r"$\ln z$", r"$\log_{10}M$"], ylabel="Cache nodes", ylim=(0, 600))
    axes[0].legend()
    bars = axes[1].bar([0, 1], [128, 16], color=[ORANGE, BLUE])
    axes[1].bar_label(bars, labels=["128 MiB", "16 MiB"], padding=4, fontsize=16)
    axes[1].set(xticks=[0, 1], xticklabels=["Default", "Half all"], ylabel="Values array per model [MiB]", ylim=(0, 155))
    for ax in axes: ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    save(fig, "cache_nodes")
    META["cache"] = dict(default_nodes=old.tolist(), half_nodes=new.tolist(), default_values=int(old.prod()),
                         half_values=int(new.prod()), spacing_ratios=((old-1)/(new-1)).tolist(),
                         default_values_MiB=128, half_values_MiB=16)


def pixel_children():
    parent_nside = 4096
    parent = int(hp.ang2pix(parent_nside, np.pi/2, 1.1, nest=True))
    theta, phi = hp.pix2ang(parent_nside, parent, nest=True)
    centre = np.array(hp.pix2vec(parent_nside, parent, nest=True))
    east = np.array([-np.sin(phi), np.cos(phi), 0])
    north = np.cross(centre, east)
    factor = 180*60/np.pi
    def project(vectors):
        vectors = np.asarray(vectors).reshape(3, -1)
        return np.vstack((east @ vectors, north @ vectors)) / (centre @ vectors) * factor
    boundary = project(hp.boundaries(parent_nside, parent, step=16, nest=True))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7), sharex=True, sharey=True, layout="constrained")
    records = []
    for ax, nside, count in zip(axes, [4096, 8192, 16384], [1, 4, 16]):
        ids = np.arange(parent*count, (parent+1)*count)
        vectors = np.array(hp.pix2vec(nside, ids, nest=True))
        check = hp.vec2pix(parent_nside, *vectors, nest=True)
        assert np.all(check == parent)
        for child in ids:
            outline = project(hp.boundaries(nside, int(child), step=12, nest=True))
            ax.plot(*np.column_stack((outline, outline[:, :1])), color="#b6c3cc", lw=1)
        ax.plot(*np.column_stack((boundary, boundary[:, :1])), color=DARK, lw=2)
        xy = project(vectors)
        ax.scatter(xy[0], xy[1], color=BLUE, s=50, zorder=5)
        ax.set(title=f"NSIDE {nside}\n{count} centre sample" + ("s" if count > 1 else ""),
               xlabel="East [arcmin]", aspect="equal", xticks=[-.5, 0, .5], yticks=[-.5, 0, .5])
        records.append(dict(nside=nside, parent_nside=4096, parent_nested_pixel=parent,
                            children=count, centres_arcmin=xy.T.tolist()))
    axes[0].set_ylabel("North [arcmin]")
    save(fig, "child_centres")
    META["pixel_geometry"] = records


def scheduling():
    # Toy block costs show the algorithm, not a captured thread trace.
    costs = [1]*8 + [2, 2, 3, 3, 4, 6, 8, 12]
    static, greedy = [[] for _ in range(4)], [[] for _ in range(4)]
    for thread in range(4):
        start = 0
        for block in range(4*thread, 4*thread+4):
            static[thread].append((start, costs[block], block)); start += costs[block]
    free = [(0, thread) for thread in range(4)]
    for block, cost in enumerate(costs):
        start, thread = heapq.heappop(free)
        greedy[thread].append((start, cost, block))
        heapq.heappush(free, (start+cost, thread))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for ax, assignments, name in zip(axes, [static, greedy], ["Static: fixed contiguous blocks", "Greedy: next block when available"]):
        for thread, blocks in enumerate(assignments):
            for start, cost, block in blocks:
                color = plt.get_cmap("viridis")(block / 15)
                ax.broken_barh([(start, cost)], (thread-.32, .64), facecolors=color, edgecolors="white")
        finish = max(start+cost for blocks in assignments for start, cost, _ in blocks)
        ax.axvline(finish, color=DARK, ls="--", lw=1)
        ax.set(title=name, yticks=range(4), yticklabels=["Worker 1", "Worker 2", "Worker 3", "Worker 4"], ylim=(3.6, -.8))
        ax.text(finish+.5, 1.5, f"Finish: {finish}", fontsize=14)
        ax.grid(axis="x", alpha=.2)
    axes[-1].set_xlabel("Illustrative elapsed time [arbitrary units]")
    axes[-1].set_xlim(0, 35)
    save(fig, "greedy_scheduling")
    counts = {}
    for name, folder in [("shared_static", SOURCE / "controls/002"), ("shared_greedy", SOURCE / "work_balance/controls/000")]:
        text = (folder / "run.log").read_text()
        values = re.findall(r"CHUNK (\d+):(\d+) selected=(\d+) elapsed_read=([\d.]+) elapsed_paint=([\d.]+)", text)
        assert len(values) == 43
        cumulative = np.array([float(v[-1]) for v in values])
        counts[name] = dict(total_paint_seconds=float(cumulative[-1]), last_chunk_seconds=float(cumulative[-1]-cumulative[-2]))
    META["measured_painting"] = counts
    META["scheduling_illustration"] = dict(costs=costs, static=static, greedy=greedy, units="arbitrary; not measured cluster trace")


def box(ax, xy, text, width=2.15, height=.7, color="#e7f1f6"):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=.08", facecolor=color, edgecolor=BLUE, lw=1.2))
    ax.text(xy[0]+width/2, xy[1]+height/2, text, ha="center", va="center", fontsize=14)


def pipeline_and_beam():
    fig, ax = plt.subplots(figsize=(13.5, 3.1), layout="constrained")
    for x, text in [(0, "Paint projected halos\nNSIDE 8192"), (2.85, r"$a_{\ell m}$"+"\n"+r"$\ell\leq12287$"),
                    (5.7, r"$B_\ell a_{\ell m}$"+"\n2 arcmin beam"), (8.55, "Synthesize map\nNSIDE 4096")]:
        box(ax, (x, .8), text)
        if x < 8:
            ax.add_patch(FancyArrowPatch((x+2.25, 1.15), (x+2.72, 1.15), arrowstyle="-|>", mutation_scale=18, color=DARK))
    ax.text(10.78, .28, "Mask + independent noise splits\n"+r"$C_\ell^{AB}\ \rightarrow$ unbinned MOPED", ha="right", va="center", fontsize=15)
    ax.add_patch(FancyArrowPatch((9.62, .72), (9.62, .55), arrowstyle="-|>", mutation_scale=15, color=DARK))
    ax.set(xlim=(-.2, 11), ylim=(-.05, 1.75)); ax.axis("off")
    save(fig, "output_pipeline")
    x = np.linspace(-3, 3, 1201)
    sigma_halo, offset = .18, .18
    sigma_beam = 2 / np.sqrt(8*np.log(2))
    raw = lambda t: np.exp(-.5*((t-offset)/sigma_halo)**2)
    sigma_total = np.hypot(sigma_halo, sigma_beam)
    smoothed = lambda t: sigma_halo/sigma_total*np.exp(-.5*((t-offset)/sigma_total)**2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    spacing = {n: float(hp.nside2resol(n, arcmin=True)) for n in (4096, 8192)}
    axes[0].plot(x, raw(x), color=DARK, lw=2)
    for n, color, marker in [(4096, ORANGE, "o"), (8192, BLUE, "x")]:
        points = np.arange(-3*spacing[n], 3.01*spacing[n], spacing[n])
        axes[0].scatter(points, raw(points), color=color, marker=marker, s=45, label=str(n), zorder=5)
    axes[0].set(title="Before beam: narrow halo", ylabel="Illustrative y", xlabel="Angle [arcmin]", xlim=(-1.5, 1.5))
    axes[0].legend()
    axes[1].plot(x, smoothed(x), color=DARK, lw=2)
    coarse_points = np.arange(-3*spacing[4096], 3.01*spacing[4096], spacing[4096])
    axes[1].scatter(coarse_points, smoothed(coarse_points), color=ORANGE, s=45, label="4096 samples")
    axes[1].set(title="After beam: smooth field", xlabel="Angle [arcmin]", xlim=(-3, 3)); axes[1].legend()
    ell = np.arange(1, 16001)
    sigma_rad = np.deg2rad(2/60)/np.sqrt(8*np.log(2))
    power = np.exp(-ell*(ell+1)*sigma_rad*sigma_rad)
    axes[2].semilogy(ell, power, color=BLUE, lw=2)
    axes[2].axvline(7979, color=ORANGE, ls="--", lw=1.4, label="Science ellmax")
    axes[2].axvline(12287, color="#777777", ls=":", lw=1.4, label="Synthesis cutoff")
    axes[2].set(title="2 arcmin beam", xlabel=r"$\ell$", ylabel=r"Power transmission $B_\ell^2$", ylim=(1e-6, 1.5))
    axes[2].legend(fontsize=11)
    for ax in axes: ax.grid(alpha=.2)
    save(fig, "sampling_before_after_beam")
    META["output_map"] = {str(n): dict(npix=int(hp.nside2npix(n)), pixel_scale_arcmin=spacing[n],
                                      one_Float64_map_GiB=hp.nside2npix(n)*8/2**30) for n in (4096, 8192)}
    META["beam_power"] = {str(l): float(np.exp(-l*(l+1)*sigma_rad*sigma_rad)) for l in (2000, 4000, 6000, 7979, 12287)}


def main():
    # Hash the exact code whose behavior is described; no numerical producer is edited.
    files = ["benchmark.jl", "spherical_truncation_profiles.jl", "edge_extension/smooth_exterior.jl",
             "edge_extension/entrypoint.jl", "edge_extension/controls/001/task.toml", "work_balance/balanced_painter.jl"]
    META["source_sha256"] = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files}
    completed = json.loads((COMPLETED / "results.json").read_text())
    META["measured_smooth_half_errors"] = [r for r in completed["errors"] if r["group"] == "smooth_exterior" and r["task"] == 1]
    sphere(); boundary(); cache_counts(); pixel_children(); scheduling(); pipeline_and_beam()
    (ROOT / "illustration_data.json").write_text(json.dumps(META, indent=2) + "\n")
    print(json.dumps({"plots": len(list(PLOTS.glob('*.png'))), "beam_power": META["beam_power"], "output_map": META["output_map"]}, indent=2))


if __name__ == "__main__":
    main()
