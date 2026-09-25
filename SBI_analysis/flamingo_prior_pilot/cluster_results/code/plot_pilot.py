"""Standalone, publication-friendly plots of the completed cluster pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
import toml

from conditional_prior import required_beta_amplitude
from prior_model import (ELECTRON_FRACTION, FIDUCIAL, HIGH, LOW, NAMES,
    bin_cl, central_column, finite_pressure_integral, parameters,
    pressure_shape, save_json, to_unit)

COLORS = {"L1_m9": "#0072B2", "fgas-8sigma": "#D55E00", "Mstar-1sigma": "#009E73"}
LABELS = {"L1_m9": "FLAMINGO fiducial", "fgas-8sigma": "FLAMINGO fgas -8 sigma",
          "Mstar-1sigma": "FLAMINGO Mstar -1 sigma"}
EXTREME_LABELS = {
    "extreme_xc_high": r"$x_{c,0}=4$",
    "extreme_beta_low": r"$\beta_0=2.8$",
    "extreme_beta_high": r"$\beta_0=16$",
    "extreme_wide_steep": r"$x_{c,0}=4,\ \beta_0=16$",
    "extreme_compact_steep_evolving": r"$x_{c,0}=0.1,\ \beta_0=16,\ \alpha_{z,\beta}=1.5$",
    "extreme_mass_redshift_amplitude": r"$\alpha_{m,P}=1.5,\ \alpha_{z,P}=-4.5$",
}
PDF_BUNDLE = None


def finish(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / (name+".png"), dpi=190, bbox_inches="tight")
    fig.savefig(output / (name+".pdf"), bbox_inches="tight")
    if PDF_BUNDLE is not None:
        PDF_BUNDLE.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def read_spectra(directory):
    result = {}
    for key, filename in (("clean", "masked_clean_cl.npy"), ("noisy", "masked_noisy_cross_cl.npy")):
        result["ell"], result[key] = bin_cl(np.load(directory / filename))
    return result


def spectrum_plots(root, campaign, result, output):
    baseline = read_spectra(campaign / "results/HalfDome")
    ell = baseline["ell"]
    targets, fits = {}, {}
    residual_limit = max(10., 125*max(f["best"]["max_fractional"] for f in result["fits"].values()))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col",
        gridspec_kw={"height_ratios": [2.2, 1]}, constrained_layout=True)
    for col, (variant, fit_result) in enumerate(result["fits"].items()):
        target = read_spectra(campaign / "results" / variant)
        fit = read_spectra(root / "results" / fit_result["best"]["name"])
        targets[variant], fits[variant] = target, fit
        color = COLORS[variant]
        axes[0, col].loglog(ell, baseline["clean"]*1e12, color="black", lw=1.5, label="HalfDome / Battaglia12")
        axes[0, col].loglog(ell, target["clean"]*1e12, color=color, lw=2.5, label=LABELS[variant])
        axes[0, col].loglog(ell, fit["clean"]*1e12, color="#CC79A7", ls="--", lw=1.8, label="Repainted parameter fit")
        axes[0, col].set_title(LABELS[variant])
        axes[0, col].legend(fontsize=8, loc="lower left")
        axes[1, col].semilogx(ell, 100*(fit["clean"]/target["clean"]-1), color="#CC79A7", ls="--", lw=2, label="Fit")
        axes[1, col].axhspan(-5, 5, color="#999999", alpha=.15)
        axes[1, col].axhline(0, color="grey", lw=.7)
        axes[1, col].set_ylim(-residual_limit, residual_limit)
        axes[1, col].set_xlabel(r"Multipole $\ell$")
        best = fit_result["best"]
        axes[1, col].text(.98, .06, "Fit RMS {:.2f}% | max {:.2f}%".format(
            100*best["rms_fractional"], 100*best["max_fractional"]),
            transform=axes[1, col].transAxes, ha="right", fontsize=8)
    axes[0, 0].set_ylabel(r"$10^{12}\widetilde D_\ell^{yy}$")
    axes[1, 0].set_ylabel("Fit minus FLAMINGO / FLAMINGO [%]")
    fig.suptitle("Clean spectra: full Nside 4096 maps, original 2 arcmin beam and fsky=0.4 mask\n"
                 "40 bins; effective spectral matches on the HalfDome catalogue; residual shading: +/-5% bin goal", fontsize=11)
    finish(fig, output, "01_clean_flamingo_fits")

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.loglog(ell, baseline["clean"]*1e12, color="black", lw=2.5, label="HalfDome / Battaglia12")
    for variant in targets:
        ax.loglog(ell, targets[variant]["clean"]*1e12, color=COLORS[variant], lw=2, label=LABELS[variant])
        ax.loglog(ell, fits[variant]["clean"]*1e12, color=COLORS[variant], ls="--", lw=1.1,
                  label=LABELS[variant].replace("FLAMINGO", "Fit to"))
    palette = plt.cm.plasma(np.linspace(.1, .85, len(result["extremes"])))
    for color, (name, data) in zip(palette, result["extremes"].items()):
        if data["status"] != "full_map_completed":
            continue
        data = read_spectra(root / "results" / name)
        ax.loglog(ell, data["clean"]*1e12, color=color, alpha=.85, lw=1.2,
                  label=EXTREME_LABELS[name])
    for p0, style in ((1., ":"), (60., "-.")):
        ax.loglog(ell, baseline["clean"]*1e12*(p0/FIDUCIAL[0])**2, color="grey",
                  ls=style, label=r"$P_0="+str(int(p0))+r"$ (exact amplitude rescaling)")
    ax.set(xlabel=r"Multipole $\ell$", ylabel=r"$10^{12}\widetilde D_\ell^{yy}$",
           title="tSZ power across the proposed range\nShape extremes use full maps; unspecified parameters stay at Battaglia12")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, .5))
    finish(fig, output, "02_tsz_with_extremes")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex="col", constrained_layout=True)
    noise_stats = {}
    for col, variant in enumerate(targets):
        target, fit = targets[variant], fits[variant]
        color = COLORS[variant]
        axes[0, col].semilogx(ell, target["clean"]*1e12, color=color, lw=2, label="FLAMINGO clean")
        axes[0, col].semilogx(ell, target["noisy"]*1e12, color=color, marker=".", ms=3, lw=.7, alpha=.7, label="FLAMINGO + SO")
        axes[0, col].semilogx(ell, fit["clean"]*1e12, color="#CC79A7", ls="--", label="Fit clean")
        axes[0, col].semilogx(ell, fit["noisy"]*1e12, color="#CC79A7", marker="x", ms=3, lw=.7, alpha=.7, label="Fit + SO")
        scale = np.max(target["clean"])
        clean_residual = (fit["clean"]-target["clean"])/scale
        noisy_residual = (fit["noisy"]-target["noisy"])/scale
        axes[1, col].semilogx(ell, 100*clean_residual, color="black", label="Clean residual")
        axes[1, col].semilogx(ell, 100*noisy_residual, color=color, label="Noisy residual")
        axes[1, col].axhline(0, color="grey", lw=.6)
        axes[1, col].set_xlabel(r"Multipole $\ell$")
        axes[0, col].set_title(LABELS[variant])
        axes[0, col].legend(fontsize=8)
        axes[1, col].legend(fontsize=8)
        noise_stats[variant] = dict(
            rms_noise_change_over_peak_target=float(np.sqrt(np.mean((noisy_residual-clean_residual)**2))),
            clean_peak_Dl=float(scale))
    axes[0, 0].set_ylabel(r"$10^{12}\widetilde D_\ell^{yy}$ (signed)")
    axes[1, 0].set_ylabel("Fit minus target / peak clean target [%]")
    fig.suptitle("Noise shown separately: identical cached SO splits and mask for every map\n"
                 "Root seed 12345; split seeds 22446 and 22447; noise was excluded from fitting", fontsize=12)
    finish(fig, output, "03_shared_so_noise")
    save_json(root / "noise_comparison.json", noise_stats)


def prior_plots(root, result, output):
    design = np.load(root / "audit/prior_design.npz")
    audit = np.load(root / "audit/sobol_audit.npz")
    example_audit = np.load(root / "audit/examples_audit.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    axes[0].hist(audit["min_beta"], bins=np.geomspace(.1, 30, 70), color="#0072B2", alpha=.85)
    axes[0].set_xscale("log")
    axes[0].axvline(.7, color="#D55E00", ls="--", label="LOS convergence")
    axes[0].axvline(2.7, color="black", ls=":", label="Infinite-volume energy")
    axes[0].set(xlabel=r"Minimum $\beta(M,z)$ on interpolation grid", ylabel="Prior draws")
    axes[0].legend(fontsize=8)
    axes[1].hist(np.log10(audit["min_Y200_ratio"]), bins=75, color="#0072B2", alpha=.6, label="Minimum over M,z")
    axes[1].hist(np.log10(audit["max_Y200_ratio"]), bins=75, color="#D55E00", alpha=.6, label="Maximum over M,z")
    axes[1].set(xlabel=r"$\log_{10}(Y_{200}/Y_{200,\mathrm{B12}})$")
    axes[1].legend(fontsize=8)
    am, az = np.meshgrid(np.linspace(-.2, .4, 180), np.linspace(-.5, 1.5, 180))
    required = required_beta_amplitude(am, az)
    shown = axes[2].pcolormesh(am, az, required, norm=LogNorm(2.8, 50), cmap="viridis", shading="auto")
    axes[2].contour(am, az, required, levels=[16], colors=["red"], linewidths=1.5)
    axes[2].plot(FIDUCIAL[5], FIDUCIAL[8], "*", color="white", ms=10)
    axes[2].set(xlabel=r"$\alpha_{m,\beta}$", ylabel=r"$\alpha_{z,\beta}$",
                title=r"Required $\beta_0$ for $\min\beta\geq2.8$")
    fig.colorbar(shown, ax=axes[2], label=r"Conditional lower bound on $\beta_0$")
    fig.suptitle("65,536 prior draws | slope grid: 1e12–1e15.7 Msun, z=0.001–5; Y200 grid also includes z=0\n"
                 "Red boundary: no allowed beta0 <= 16 can satisfy the conservative outer-energy condition", fontsize=11)
    finish(fig, output, "04_prior_support_and_conditions")

    # All 9 single-parameter edges, then every pair with the other 7 at B12.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    edge = np.arange(1, 19)
    ypos = np.repeat(np.arange(9), 2)+np.tile([-.13, .13], 9)
    for i, index in enumerate(edge):
        color = "#0072B2" if i % 2 == 0 else "#D55E00"
        axes[0].plot([np.log10(example_audit["min_Y200_ratio"][index]),
                      np.log10(example_audit["max_Y200_ratio"][index])], [ypos[i]]*2, color=color, lw=3, marker="o", ms=3)
    axes[0].set_yticks(np.arange(9))
    axes[0].set_yticklabels([str(i+1)+". "+name for i, name in enumerate(NAMES)])
    axes[0].invert_yaxis()
    axes[0].set(xlabel=r"Range of $\log_{10}(Y_{200}/Y_{200,\mathrm{B12}})$",
                title="Single edges: low blue / high orange")
    fail = np.full((9, 9), np.nan)
    span = np.full((9, 9), np.nan)
    index = 19
    for i in range(9):
        for j in range(i+1, 9):
            sl = slice(index, index+4)
            fail[i, j] = fail[j, i] = np.mean(example_audit["min_beta"][sl] <= .7)*100
            span[i, j] = span[j, i] = np.log10(example_audit["max_Y200_ratio"][sl].max()/example_audit["min_Y200_ratio"][sl].min())
            index += 4
    for ax, values, title, cmap in ((axes[1], fail, "Pair corners with divergent LOS [%]", "Reds"),
                                  (axes[2], span, "Pair-corner Y200 span [dex]", "magma")):
        im = ax.imshow(values, cmap=cmap, vmin=0)
        ax.set_xticks(range(9))
        ax.set_xticklabels(range(1, 10))
        ax.set_yticks(range(9))
        ax.set_yticklabels(range(1, 10))
        ax.set(xlabel="Parameter number in left panel", title=title)
        fig.colorbar(im, ax=ax, shrink=.8)
    finish(fig, output, "05_edges_and_combinations")


def lee22_electron_parameters(mass, z, concentration=4.5):
    """Lee22 arXiv v1 Table 1, Eq.12; normalization pivot is M_cut."""
    mcut = 10**13.64/.6774  # h^-1 Msun converted using the TNG cosmology.
    m = np.asarray(mass)/mcut
    c = concentration/10.
    p0 = 6.*c*(1+z)**(-1.38)*m**np.where(m < 1, 1.09, .76)
    xc = 1.02*c**(-1.24)*(1+z)**(-.33)*m**(-.29)
    beta = 6.3*c**(-.82)*(1+z)**(-.07)*m**(.01)
    return p0, xc, beta


def profile_plots(root, result, output):
    radius = np.geomspace(.01, 4, 240)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, constrained_layout=True)
    for row, mass in enumerate((1e13, 1e14, 1e15)):
        for col, z in enumerate((0., 1., 3.)):
            ax = axes[row, col]
            p0, xc, beta = [float(p[0]) for p in parameters(FIDUCIAL, mass, z)]
            ax.loglog(radius, ELECTRON_FRACTION*p0*pressure_shape(radius, xc, beta),
                      color="black", lw=2, label="Battaglia12")
            for variant, fit in result["fits"].items():
                p0, xc, beta = [float(p[0]) for p in parameters(fit["best"]["theta"], mass, z)]
                ax.loglog(radius, ELECTRON_FRACTION*p0*pressure_shape(radius, xc, beta),
                          color=COLORS[variant], label="Fit: "+variant)
            for name in ("extreme_beta_low", "extreme_compact_steep_evolving"):
                p0, xc, beta = [float(p[0]) for p in parameters(result["extremes"][name]["theta"], mass, z)]
                ax.loglog(radius, ELECTRON_FRACTION*p0*pressure_shape(radius, xc, beta),
                          color="#CC79A7" if "compact" in name else "#E69F00", ls=":", lw=1.2,
                          label="Compact/steep extreme" if "compact" in name else "Shallow-slope extreme")
            p0, xc, beta = lee22_electron_parameters(mass, z)
            ax.loglog(radius, p0*pressure_shape(radius, xc, beta), color="#777777", ls="--",
                      label="Lee22 formula, fixed c=4.5")
            ax.axvspan(.04, 1.34, color="grey", alpha=.06)
            ax.set_title(r"$M=10^{"+str(int(np.log10(mass)))+r"}\,M_\odot,\ z="+str(z)+r"$")
            ax.set_ylim(1e-8, 1e3)
            if row == 2:
                ax.set_xlabel(r"$r/R_{200c}$")
            if col == 0:
                ax.set_ylabel(r"$P_e/P_{200}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.035), ncol=4, fontsize=8)
    fig.suptitle("Pressure profiles implied by the spectral fits and extremes\n"
                 "Grey band: Lee22 radial fit interval; mass calibration roughly 1.5e13–1.5e14 Msun, z <= 2\n"
                 "Other masses, redshifts and radii are illustrative extrapolations", fontsize=11)
    finish(fig, output, "06_pressure_profiles_and_lee22")

    cutoffs = np.geomspace(4, 1e8, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for beta, color in zip((.5, .7, .8, 1.05, 1.3, 2.8, 4.35), plt.cm.viridis(np.linspace(0, 1, 7))):
        values = np.array([central_column(1., 4., beta, cutoff) for cutoff in cutoffs])
        at_painter = central_column(1., 4., beta, 1e5)
        axes[0].semilogx(cutoffs, values/at_painter, color=color, label=r"$\beta="+str(beta)+r"$")
        if beta > .7:
            total = central_column(1., 4., beta)
            axes[1].loglog(cutoffs, np.maximum(1-values/total, 1e-16), color=color)
    axes[0].axvline(1e5, color="black", ls=":")
    axes[1].axvline(1e5, color="black", ls=":")
    axes[1].axhline(.01, color="#D55E00", ls="--")
    axes[0].set(xlabel=r"LOS endpoint / $R_{200}$", ylabel="Central column / value at current endpoint",
                title=r"Cutoff sensitivity at $x_c=4$ and fixed local beta")
    axes[1].set(xlabel=r"LOS endpoint / $R_{200}$", ylabel="Fraction of infinite column omitted",
                ylim=(1e-10, 1), title="Convergent cases; red line = 1%")
    axes[0].legend(fontsize=8, ncol=2)
    finish(fig, output, "07_line_of_sight_cutoff")


def numerical_plots(root, result, output):
    summary = json.loads((root / "audit/interpolation_summary.json").read_text())
    records = []
    for path in (root / "results").glob("*/profile_validation.toml"):
        data = toml.load(path)
        data["name"] = path.parent.name
        records.append(data)
    fig, ax = plt.subplots(figsize=(11, max(4, len(records)*.38)), constrained_layout=True)
    for i, record in enumerate(records):
        independent = summary[record["name"]]["all_grid"]
        ax.plot([100*independent["p95"], 100*independent["maximum"]],
                [i, i], color="#0072B2", lw=2)
        ax.plot(100*independent["p95"], i, "o", color="#0072B2")
    for i, record in enumerate(records):
        ax.plot(100*summary[record["name"]]["z_ge_0p1"]["p95"], i, "s", color="#D55E00",
                label=r"95th percentile, $z\geq0.1$ only" if i == 0 else None)
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([r["name"] for r in records], fontsize=8)
    ax.set_xscale("log")
    ax.axvline(1, color="#D55E00", ls="--", label="1%")
    ax.set(xlabel="Profile interpolation relative error [%]", title="Independent LOS quadrature versus production interpolation\n"
           "Blue: all-grid 95th percentile to maximum; orange: z >= 0.1 only; significant y samples")
    ax.legend()
    finish(fig, output, "08_interpolation_accuracy")
    pixels = json.loads((root / "audit/pixel_resolution.json").read_text())
    chosen = ["Battaglia12", "fit_L1_m9", "fit_fgas-8sigma", "fit_Mstar-1sigma", "compact_steep_evolving"]
    fig, axes = plt.subplots(1, len(chosen), figsize=(16, 4.5), constrained_layout=True)
    for ax, name in zip(axes, chosen):
        for nside, style, color in ((4096, "-", "#0072B2"), (8192, "--", "#D55E00")):
            rows = [r for r in pixels["rows"] if r["name"] == name and r["nside"] == nside and r["mass_Msun"] == 1e14]
            z = np.array([r["z"] for r in rows])
            median = np.array([r["median"] for r in rows])
            ax.plot(z, median, style, color=color, marker="o", label="Nside "+str(nside))
            ax.fill_between(z, [r["p16"] for r in rows], [r["p84"] for r in rows], color=color, alpha=.15)
        ax.axhline(1, color="black", lw=.8)
        ax.set(title=name.replace("fit_", "Fit: ").replace("_", " "), xlabel="Redshift")
        if name == "compact_steep_evolving":
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Sampled halo flux / continuous halo flux")
    fig.suptitle("Pixel sampling at M=1e14 Msun: 32 actual HEALPix placements per case\n"
                 "Lines: median; bands: 16–84 percentiles; last panel logarithmic. A beam cannot restore missing flux.", fontsize=11)
    finish(fig, output, "09_pixel_sampling_limits")


def main():
    global PDF_BUNDLE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .18,
                         "savefig.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    result = json.loads((args.root / "pilot_results.json").read_text())
    output = args.root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    with PdfPages(output / "tsz_prior_pilot_plots.pdf") as PDF_BUNDLE:
        spectrum_plots(args.root, args.campaign, result, output)
        prior_plots(args.root, result, output)
        profile_plots(args.root, result, output)
        numerical_plots(args.root, result, output)
    PDF_BUNDLE = None
    print("Saved nine plot sets as PNG and PDF: "+str(output), flush=True)


if __name__ == "__main__":
    main()
