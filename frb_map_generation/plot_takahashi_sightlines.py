#!/usr/bin/env python3
"""Readable finite-source comparison plots; never manufactures missing results."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_halfdome_takahashi import MODELS, SURVEYS, read_rows


def column(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def style(axis):
    axis.set_xscale("log")
    axis.grid(alpha=.18, which="both")
    axis.axhline(0, color=".5", lw=.8)
    axis.tick_params(direction="in", which="both")


def plot_diagnostics(root):
    with h5py.File(str(root/"rays/source_positions.h5"), "r") as h5:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, survey in zip(axes[0], SURVEYS):
            observed = h5[survey+"/observed_redshifts"][:]
            sampled = h5[survey+"/redshift"][:]
            weights = h5[survey+"/analysis_weight"][:]
            edges = np.linspace(0, 2.2, 23)
            ax.hist(observed, edges, weights=np.ones(len(observed))/len(observed),
                    histtype="step", lw=2.8, color=".2", label="Observed catalogue")
            ax.hist(sampled, edges, weights=weights, histtype="step", lw=1.8,
                    linestyle="--", color="#d65a59", label="100,000 rays (weighted)")
            ax.set(title=survey.upper()+" source redshifts", xlabel="Source redshift", ylabel="Fraction per bin")
            ax.legend(fontsize=9)
            ax.grid(alpha=.2)
        longitude = h5["longitude_deg"][:]
        latitude = h5["latitude_deg"][:]
        ax = axes[1, 0]
        ax.hist2d(longitude, np.sin(np.deg2rad(latitude)), bins=(72, 36), cmap="viridis")
        ax.set(xlabel="Simulation longitude [deg]", ylabel="sin(latitude)",
               title="Equal-area random directions; common to all models")
        ax = axes[1, 1]
        for survey, color in (("planck", "#245c78"), ("act", "#d65a59")):
            z = h5[survey+"/redshift"][:]
            order = np.argsort(z)
            ax.plot(z[order], np.cumsum(h5[survey+"/analysis_weight"][:][order]),
                    color=color, label=survey.upper())
        ax.set(xlabel="Source redshift", ylabel="Cumulative source fraction", title="Empirical redshift CDF")
        ax.legend()
        ax.grid(alpha=.2)
        fig.suptitle("100,000 explicit sightlines per survey redshift distribution", fontsize=15)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.text(.5, .015, "Sources are not placed in host halos. Random NSIDE=4096 pixel centres; no observational sky mask.", ha="center", fontsize=10)
        fig.savefig(str(root/"plots/source_redshift_and_sky_diagnostics.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_comparisons(root, inputs):
    results = read_rows(root/"analysis/sightline_comparison.csv")
    observations = read_rows(inputs/"digitized/takahashi_fig13_approximate.csv")
    for survey, (nsource, beam) in SURVEYS.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col",
                                  gridspec_kw={"height_ratios": [3, 1.4]})
        observed = [row for row in observations if
                    ("ACT" in row["series"]) == (survey == "act")]
        for p, (model, label, color, _) in enumerate(MODELS):
            rows = [r for r in results if r["survey"] == survey and r["filter"] == survey and r["model"] == model]
            x = column(rows, "theta_arcmin")
            value, error, previous = [column(rows, key) for key in
                                      ("cross_100k", "sampling_sigma", "previous_fullmap")]
            ax, ratio = axes[:, p]
            ax.errorbar(x, value/1e-5, yerr=error/1e-5, color=color, marker="o", ms=4,
                        lw=1.5, capsize=2, label="100k individual sightlines")
            ax.plot(x, previous/1e-5, color=".2", ls="--", lw=1.8, label="Previous full-map mean")
            if observed:
                ax.errorbar(column(observed, "theta_plotted_arcmin"), column(observed, "w_yDM_pc_cm3")/1e-5,
                    yerr=np.array([column(observed, "error_lower_pc_cm3"), column(observed, "error_upper_pc_cm3")])/1e-5,
                    color=".6", fmt="s", ms=3, alpha=.65, lw=1, label="Takahashi (digitized)")
            valid = np.abs(previous) > .01*np.max(np.abs(previous))
            ratio.errorbar(x[valid], 100*(value[valid]/previous[valid]-1),
                yerr=100*error[valid]/np.abs(previous[valid]), color=color, marker="o", ms=3, capsize=2)
            ax.set_title(label, fontsize=11, pad=12)
            ax.set_ylabel(r"$w_{y,\mathrm{DM}}\ [10^{-5}\,\mathrm{pc\,cm^{-3}}]$")
            ratio.set(xlabel=r"$\theta$ [arcmin]", ylabel="100k vs mean [%]")
            for a in (ax, ratio):
                style(a)
                a.set_xlim(1, 1000)
                a.axvspan(1, 10 if survey == "planck" else 10**.25, color=".9", zorder=-10)
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle("{}: 100,000 rays following {} observed redshifts; {:.1f}' y beam".format(
            survey.upper(), nsource, beam), fontsize=16, y=.98)
        fig.tight_layout(rect=(0, .085, 1, .94))
        fig.text(.5, .045, "Halo-only HalfDome: spherical 3R200c, Battaglia12 pressure, all resolved foreground halos. No host/IGM DM or survey noise/mask.", ha="center", fontsize=10)
        fig.text(.5, .018, "Errors: finite-source jackknife on one fixed sky. Percentages are linear; omitted where |previous| < 1% of its peak. Preferred Lee22 is diagnostic only.", ha="center", fontsize=9)
        fig.savefig(str(root/"plots"/("sightlines100k_vs_previous_"+survey+".png")), dpi=170, bbox_inches="tight")
        plt.close(fig)


def plot_medlock(root):
    results = read_rows(root/"analysis/sightline_comparison.csv")
    references = read_rows(root/"references/medlock_fig5_best_fit_digitized.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, panel, survey in zip(axes, ("act", "planck_milca", "planck_nilc"), ("act", "planck", "planck")):
        for model, label, color, _ in MODELS[:2]:
            rows = [r for r in results if r["survey"] == survey and r["filter"] == "unbeamed"
                    and r["model"] == model and float(r["theta_arcmin"]) >= (2 if survey == "act" else 10)]
            ax.errorbar(column(rows, "theta_arcmin"), column(rows, "cross_100k")/1e-5,
                yerr=column(rows, "sampling_sigma")/1e-5, marker="o", ms=4,
                color=color, lw=1.5, capsize=2, label="HD 100k: "+label)
        ref = [r for r in references if r["survey"] == panel]
        ax.plot(column(ref, "theta_arcmin"), column(ref, "w_yDM_pc_cm3")/1e-5,
                color="black", lw=2, label="BP best fit (digitized; sources z=2)")
        ax.fill_between(column(ref, "theta_arcmin"),
            (column(ref, "w_yDM_pc_cm3")-column(ref, "digitization_absolute_uncertainty"))/1e-5,
            (column(ref, "w_yDM_pc_cm3")+column(ref, "digitization_absolute_uncertainty"))/1e-5,
            color=".4", alpha=.18)
        ax.set_title(panel.replace("_", " ").upper(), fontsize=13)
        ax.set(xlabel=r"$\theta$ [arcmin]", ylabel=r"$w_{y,\mathrm{DM}}\ [10^{-5}\,\mathrm{pc\,cm^{-3}}]$")
        style(ax)
        ax.set_xlim(2 if survey == "act" else 10, 1000)
        ax.legend(fontsize=8)
    fig.suptitle("HalfDome versus Medlock & Nagai Figure 5: unbeamed model references", fontsize=15, y=.98)
    fig.tight_layout(rect=(0, .13, 1, .92))
    fig.text(.5, .065, "NOT a matched source-kernel test: HD follows observed redshifts; published BP places all FRBs at z=2. Both curves here have no beam.", ha="center", fontsize=10)
    fig.text(.5, .022, "MILCA/NILC share the ideal HD prediction; BP fits differ. Grey BP band: image readout uncertainty, not posterior errors. No extrapolation.", ha="center", fontsize=9)
    fig.savefig(str(root/"plots/halfdome_vs_medlock_fig5.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="frb_map_generation/outputs/takahashi_100k_20260914")
    parser.add_argument("--inputs", default="frb_map_generation/outputs/observational_comparison_inputs_20260913")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    (root/"plots").mkdir(parents=True, exist_ok=True)
    plot_diagnostics(root)
    if not args.diagnostics_only:
        plot_comparisons(root, Path(args.inputs))
        plot_medlock(root)
    print("Saved plots in " + str(root/"plots"))


if __name__ == "__main__":
    main()
