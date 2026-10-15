#!/usr/bin/env python3
"""Plot matched 3R200c kSZ spectra with the published Websky halo reference."""
from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922"
KSZ = OUT / "ksz"
TCMB_UK = 2.7255e6
COLORS = dict(lee22_previous="#D55E00", lee22_paper="#0072B2",
              battaglia16="#009E73", class_sz_b16="#222222", websky_halo="#AA4499")
LABELS = dict(lee22_previous="Lee22: previous normalization",
              lee22_paper="Lee22: paper normalization",
              battaglia16="Battaglia16: HalfDome",
              class_sz_b16="Battaglia16: CLASS-SZ",
              websky_halo="Websky halo: Stein+2020 reference")


def weighted_band(ell, values, lo, hi):
    keep = (ell >= lo) & (ell < hi) & np.isfinite(values)
    assert keep.any(), (lo, hi)
    weights = 2*ell[keep]+1
    return float(np.average(values[keep], weights=weights))


def bin_values(ell, values, edges):
    return np.array([weighted_band(ell, values, lo, hi) for lo, hi in zip(edges[:-1], edges[1:])])


def save_figure(fig, name):
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = tomllib.loads((KSZ / "settings.toml").read_text())
    with np.load(KSZ / "map_spectra.npz") as archive:
        maps = dict(archive)
    ell = maps["ell"]
    scale = float(maps["lee22_paper_over_previous"])
    spectra = {}
    for selection in ("restricted", "full"):
        for model in ("lee22_previous", "lee22_paper", "battaglia16"):
            name = selection + "_" + model
            spectra[name] = ell*(ell+1)/(2*np.pi) * maps["cl_" + name] * TCMB_UK**2
        with np.load(KSZ / ("class_sz_"+selection) / "class_sz_spectra.npz") as theory:
            theory_ell = theory["ell"]
            theory_dl = (theory["dl_ksz_1h"]+theory["dl_ksz_2h"])*TCMB_UK**2
        valid = theory_dl > 0
        interpolation = PchipInterpolator(np.log(theory_ell[valid]), np.log(theory_dl[valid]), extrapolate=False)
        prediction = np.full(ell.shape, np.nan, dtype=float)
        support = (ell >= theory_ell[valid].min()) & (ell <= theory_ell[valid].max())
        prediction[support] = np.exp(interpolation(np.log(ell[support])))
        spectra[selection + "_class_sz_b16"] = prediction
        assert np.allclose(spectra[selection+"_lee22_paper"][2:],
                           spectra[selection+"_lee22_previous"][2:]*scale**2, rtol=1e-14)

    reference = np.loadtxt(OUT / "reference/websky_halo_digitized.csv", delimiter=",", skiprows=1)
    websky = np.full(ell.shape, np.nan, dtype=float)
    websky[reference[:, 0].astype(int)] = reference[:, 1]
    spectra["websky_halo"] = websky
    edges = np.unique(np.rint(np.geomspace(70, 5501, 52)).astype(int))
    centers = bin_values(ell, ell, edges)
    binned = {key: bin_values(ell, values, edges) for key, values in spectra.items()}
    columns = dict(ell_min=edges[:-1], ell_max_exclusive=edges[1:], ell_effective=centers, **binned)
    for selection in ("restricted", "full"):
        for model in ("lee22_previous", "lee22_paper", "battaglia16"):
            key = selection + "_" + model
            columns[key + "_percent_vs_class_b16"] = 100*(binned[key]/binned[selection+"_class_sz_b16"]-1)
    with (OUT / "ksz_binned_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    np.savez_compressed(OUT / "ksz_comparison_arrays.npz", ell=ell, **spectra)

    plt.rcParams.update({"font.size":16, "axes.labelsize":20, "axes.titlesize":18,
                         "xtick.labelsize":16, "ytick.labelsize":16, "legend.fontsize":15,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), sharex="col",
                             gridspec_kw={"height_ratios":[3.1, 1], "hspace":.08, "wspace":.2})
    handles = []
    for col, selection in enumerate(("restricted", "full")):
        top, bottom = axes[:, col]
        for model in LABELS:
            key = "websky_halo" if model == "websky_halo" else selection + "_" + model
            line, = top.loglog(centers, binned[key], color=COLORS[model], lw=2.3,
                               ls="--" if model == "class_sz_b16" else (":" if model == "websky_halo" else "-"),
                               label=LABELS[model])
            if col == 0:
                handles.append(line)
        # A model-matched residual is scientifically interpretable: the two
        # Lee22 models are shown above, but CLASS-SZ here implements B16 only.
        residual = columns[selection+"_battaglia16_percent_vs_class_b16"]
        bottom.semilogx(centers, residual, color=COLORS["battaglia16"], lw=2)
        bottom.axhline(0, color="#555555", lw=1)
        bottom.set_ylim(-20, 20)
        bottom.set_yticks([-20, -10, 0, 10, 20])
        bottom.set_xlabel(r"Multipole $\ell$")
        for axis in (top, bottom):
            axis.set_xlim(70, 5500)
            axis.grid(alpha=.18, which="major")
        top.set_ylim(.003, 6)
        top.set_title(r"$10^{13}\leq M_{200c}/M_\odot<10^{14}$; $z\leq1$" if col == 0 else "Full HalfDome catalogue")
    axes[0, 0].set_ylabel(r"$D_\ell^{\rm kSZ}$ [$\mu$K$^2$]")
    axes[1, 0].set_ylabel("B16 / CLASS-SZ\nminus 1 [%]", fontsize=16)
    fig.suptitle(r"Halo kSZ: finite $3R_{200c}$ spheres", y=1.005, fontsize=22)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .965), ncol=2, frameon=False)
    fig.subplots_adjust(top=.805, bottom=.115)
    fig.text(.5, .025, "Websky: broader, compensated reference. Lee22: extrapolated beyond its fitted radial range.",
             ha="center", fontsize=13)
    save_figure(fig, "ksz_3r200c_restricted_and_full")

    # A single-panel version retains the linear ordinate of the Websky figure.
    fig, axis = plt.subplots(figsize=(11.5, 7.6))
    for model in LABELS:
        key = "websky_halo" if model == "websky_halo" else "full_"+model
        axis.semilogx(centers, binned[key], color=COLORS[model], lw=2.5,
                      ls="--" if model == "class_sz_b16" else (":" if model == "websky_halo" else "-"),
                      label=LABELS[model])
    axis.set(xlim=(70, 5500), ylim=(0, 4.2), xlabel=r"Multipole $\ell$",
             ylabel=r"$D_\ell^{\rm kSZ}$ [$\mu$K$^2$]",
             title=r"Full HalfDome catalogue; $3R_{200c}$")
    axis.legend(frameon=False, loc="upper left", fontsize=14)
    axis.grid(alpha=.2)
    fig.text(.5, .025, "Websky halo reference retains the paper's selection and compensation.", ha="center", fontsize=13)
    fig.subplots_adjust(bottom=.15)
    save_figure(fig, "ksz_3r200c_full_websky_reference")

    summary = dict(sphere_r200c=3, cosmology=cfg["cosmology"],
                   amplitude_ratio_paper_over_previous=scale, power_ratio_paper_over_previous=scale**2,
                   binning="(2ell+1)-weighted mean D_ell; 51 logarithmic bins, 70<=ell<=5500",
                   reference="Vector-extracted halo-only Stein+2020, not matched to HalfDome cuts.",
                   class_sz_model="Battaglia16 AGN, 1-halo + 2-halo; no independent CLASS-SZ Lee22 curve.",
                   residual_display_limits_percent=[-20, 20], residual_csv_unclipped=True,
                   bands={}, b16_class_signed_percent={})
    for center in (1000, 3000, 5000):
        summary["bands"][str(center)] = dict(ell_min=center-50, ell_max_exclusive=center+50,
            D_ell_uK2={name: weighted_band(ell, values, center-50, center+50) for name, values in spectra.items()})
    for selection in ("restricted", "full"):
        summary["b16_class_signed_percent"][selection] = {
            str(center): 100*(summary["bands"][str(center)]["D_ell_uK2"][selection+"_battaglia16"]/
                             summary["bands"][str(center)]["D_ell_uK2"][selection+"_class_sz_b16"]-1)
            for center in (1000, 3000, 5000)}
    (OUT / "ksz_comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
