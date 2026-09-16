#!/usr/bin/env python3
"""Publication-sized halo-DM PDFs and finite-FRB cross-correlations.

This file changes presentation, not density profiles or saved PDF binning.
Long scientific qualifications belong in captions/metadata, not on the figures.
Run as: python -m frb_map_generation.make_publication_comparisons
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np

from .compare_tng_lee22_battaglia16 import (
    Lee22TngBattagliaComparison, UPPER_ENTRIES, TO_1E14_ENTRIES)
from .compare_halfdome_takahashi import MODELS, SURVEYS, read_rows, write_rows


STYLE = {
    "font.family": "DejaVu Sans", "font.size": 18,
    "axes.labelsize": 20, "axes.titlesize": 21,
    "xtick.labelsize": 17, "ytick.labelsize": 17,
    "legend.fontsize": 16, "figure.titlesize": 24,
    "lines.linewidth": 2.5, "axes.linewidth": 1.1,
    "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white",
}
MODEL_NAMES = {"battaglia16": "Battaglia16", "lee22_legacy": "Lee22",
               "lee22_preferred": "Lee22 + c (test)"}
COUNT_STYLES = {
    "observed": ("#0072B2", "o", "-"),
    "10k": ("#D55E00", "^", "--"),
    "100k": ("#009E73", "s", ":"),
}


def values(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def finish(fig, stem, root, book, manifest, caption):
    """Keep each vector figure and a high-resolution preview; also add to book."""
    fig.savefig(root/"plots"/(stem+".png"), dpi=200, bbox_inches="tight", pad_inches=.15)
    fig.savefig(root/"plots"/(stem+".svg"), bbox_inches="tight", pad_inches=.15)
    book.savefig(fig, bbox_inches="tight", pad_inches=.15)
    manifest.append({"figure": stem, "caption": caption})
    plt.close(fig)


def common_axis(ax, xlim, percent=False):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.grid(axis="both", which="major", color=".85", linewidth=.65)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    if percent:
        ax.axhline(0, color=".45", linewidth=1)
        ax.yaxis.set_major_locator(MaxNLocator(4))


def halo_pdf_figures(root, project, book, manifest):
    comparison = Lee22TngBattagliaComparison(project)
    comparison.validate_inputs()
    direct = comparison.direct
    audit = []
    for include_lee in (False, True):
        for group_name, entries, xmax in (("upper_mass_limits", UPPER_ENTRIES, 5000.),
                                          ("to_1e14", TO_1E14_ENTRIES, 10000.)):
            fig = plt.figure(figsize=(17.5, 12.5))
            outer = fig.add_gridspec(2, 3, hspace=.38, wspace=.32,
                                    left=.075, right=.985, bottom=.065, top=.89)
            for j, (label, tng_label, hd_label) in enumerate(entries):
                inner = outer[j//3, j%3].subgridspec(2, 1, height_ratios=(2.7, 1.25), hspace=.06)
                top = fig.add_subplot(inner[0])
                bottom = fig.add_subplot(inner[1], sharex=top)
                hd = direct.load_halfdome(1., 4096, 3., hd_label, validated_200c=True)
                tng_pdf, tng_count, tng_zero = direct.histogram_from_values(
                    direct.tng_values(tng_label, 1.), hd["edges"])
                old_counts = direct.histogram_from_values(
                    direct.tng_values(tng_label, 1.), comparison.lee22["windows"][hd_label]["edges"])[1]
                np.testing.assert_array_equal(tng_count, old_counts)
                direct.draw_pdf(top, hd["centers"], tng_pdf, color=".15", lw=2.7)
                models = [("Battaglia16", hd, "#0072B2", "--")]
                if include_lee:
                    models.append(("Lee22", comparison.lee22["windows"][hd_label], "#D55E00", "-"))
                for name, product, color, line in models:
                    direct.draw_pdf(top, product["centers"], product["pdf"],
                                    color=color, ls=line, lw=2.5)
                    delta = direct.percent_difference(product["pdf"], tng_pdf, product["counts"], tng_count)
                    # Preserve NaNs so unsupported bins are not bridged.
                    bottom.plot(product["centers"], delta, color=color, ls=line)
                    for b in range(len(tng_pdf)):
                        if include_lee:
                            audit.append(dict(group=group_name, window=hd_label, model=name,
                                dm_low=product["edges"][b], dm_high=product["edges"][b+1],
                                tng_pdf=tng_pdf[b], halfdome_pdf=product["pdf"][b],
                                tng_count=int(tng_count[b]), halfdome_count=int(product["counts"][b]),
                                percent_difference=delta[b], tng_zero_fraction=tng_zero,
                                halfdome_zero_fraction=product["zero_fraction"]))
                title = "Total" if j == 0 else label.replace(r"\,M_\odot", "")
                top.set_title(title, pad=10)
                top.set_yscale("log")
                common_axis(top, (.1, xmax))
                common_axis(bottom, (.1, xmax), percent=True)
                top.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
                top.tick_params(labelbottom=False)
                bottom.set_xlabel(r"DM [pc cm$^{-3}$]")
                if j % 3 == 0:
                    top.set_ylabel(r"$p(\mathrm{DM})$")
                    bottom.set_ylabel(r"$\Delta p/p_{\rm TNG}$ [%]", fontsize=18)
                if not np.any(hd["counts"]):
                    top.text(.5, .1, "HD: no resolved halos", transform=top.transAxes,
                             ha="center", fontsize=16, color=".35")
                    bottom.set_yticks([])
                    bottom.text(.5, .6, "Undefined", transform=bottom.transAxes,
                                ha="center", fontsize=16, color=".4")
            legend_ax = fig.add_subplot(outer[1, 2])
            legend_ax.axis("off")
            handles = [Line2D([], [], color=".15", label="IllustrisTNG"),
                       Line2D([], [], color="#0072B2", ls="--", label="HalfDome: Battaglia16")]
            if include_lee:
                handles.append(Line2D([], [], color="#D55E00", label="HalfDome: Lee22"))
            legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=20,
                             labelspacing=1.1, handlelength=2.8)
            fig.suptitle(r"Halo DM PDFs  |  $z_s=1$  |  $M_{200c}/M_\odot$", y=.985)
            stem = "halo_pdf_"+("b16_lee22_tng_" if include_lee else "b16_tng_")+group_name
            caption = ("Halo-only positive-DM PDFs, unchanged historical z=1/4096/120k products; "
                "external projected 3R200c aperture, original LOS prescriptions, NOT the later spherical-cut spectra test. "
                "TNG is Ralf Konietzka's catalogue. Total HD is the stored full resolved 1e10-1e16 window. "
                "Mass panels use physical M200c; HD resolution floor is about 7.327e12 Msun. "
                "PDFs are normalized in the original stored [0.1,30000] pc cm^-3 bins; no rebinning or smoothing. "
                "Linear percentages are 100*(HD-TNG)/TNG only where both counts >=10. No synthetic mass windows. "
                "A zero-halo HD window has a delta at DM=0 and no normalized positive-DM PDF.")
            finish(fig, stem, root, book, manifest, caption)
    write_rows(root/"analysis/halo_pdf_comparison_bins.csv", audit)


APERTURE_STYLES = {
    # (aperture multiplier, legend label, colour, line style)
    3.0: (r"HalfDome: Battaglia16, $3\,R_{200c}$", "#0072B2", "--"),
    1.0: (r"HalfDome: Battaglia16, $1\,R_{200c}$", "#D55E00", "-"),
}


def halo_pdf_aperture_figures(root, project, book, manifest):
    """Battaglia16 at 3R200c and 1R200c versus TNG, same layout as halo_pdf_figures.

    Both HalfDome products share the rays, catalogue, DM cache, M200c windows and
    histogram bins; only the generator-owned angular aperture differs.
    """
    direct = Lee22TngBattagliaComparison(project).direct
    direct.validate_fixed_200c_input()
    products = {}
    for aperture in APERTURE_STYLES:
        products[aperture] = {
            label: direct.load_halfdome(1., 4096, aperture, label, validated_200c=True)
            for label in dict.fromkeys(e[2] for e in UPPER_ENTRIES + TO_1E14_ENTRIES)}
    for label, product in products[1.0].items():
        np.testing.assert_allclose(product["edges"], products[3.0][label]["edges"], rtol=1e-12, atol=0)
        assert np.isclose(product["aperture_r200"], 1.0) and np.isclose(products[3.0][label]["aperture_r200"], 3.0)
    audit = []
    for group_name, entries, xmax in (("upper_mass_limits", UPPER_ENTRIES, 5000.),
                                      ("to_1e14", TO_1E14_ENTRIES, 10000.)):
        fig = plt.figure(figsize=(17.5, 12.5))
        outer = fig.add_gridspec(2, 3, hspace=.38, wspace=.32,
                                left=.075, right=.985, bottom=.065, top=.89)
        for j, (label, tng_label, hd_label) in enumerate(entries):
            inner = outer[j//3, j%3].subgridspec(2, 1, height_ratios=(2.7, 1.25), hspace=.06)
            top = fig.add_subplot(inner[0])
            bottom = fig.add_subplot(inner[1], sharex=top)
            edges = products[3.0][hd_label]["edges"]
            tng_pdf, tng_count, tng_zero = direct.histogram_from_values(
                direct.tng_values(tng_label, 1.), edges)
            direct.draw_pdf(top, products[3.0][hd_label]["centers"], tng_pdf, color=".15", lw=2.7)
            any_counts = False
            for aperture, (name, color, line) in APERTURE_STYLES.items():
                product = products[aperture][hd_label]
                any_counts |= bool(np.any(product["counts"]))
                direct.draw_pdf(top, product["centers"], product["pdf"], color=color, ls=line, lw=2.5)
                delta = direct.percent_difference(product["pdf"], tng_pdf, product["counts"], tng_count)
                bottom.plot(product["centers"], delta, color=color, ls=line)
                for b in range(len(tng_pdf)):
                    audit.append(dict(group=group_name, window=hd_label, aperture_r200c=aperture,
                        dm_low=product["edges"][b], dm_high=product["edges"][b+1],
                        tng_pdf=tng_pdf[b], halfdome_pdf=product["pdf"][b],
                        tng_count=int(tng_count[b]), halfdome_count=int(product["counts"][b]),
                        percent_difference=delta[b], tng_zero_fraction=tng_zero,
                        halfdome_zero_fraction=product["zero_fraction"]))
            title = "Total" if j == 0 else label.replace(r"\,M_\odot", "")
            top.set_title(title, pad=10)
            top.set_yscale("log")
            common_axis(top, (.1, xmax))
            common_axis(bottom, (.1, xmax), percent=True)
            top.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            top.tick_params(labelbottom=False)
            bottom.set_xlabel(r"DM [pc cm$^{-3}$]")
            if j % 3 == 0:
                top.set_ylabel(r"$p(\mathrm{DM})$")
                bottom.set_ylabel(r"$\Delta p/p_{\rm TNG}$ [%]", fontsize=18)
            if not any_counts:
                top.text(.5, .1, "HD: no resolved halos", transform=top.transAxes,
                         ha="center", fontsize=16, color=".35")
                bottom.set_yticks([])
                bottom.text(.5, .6, "Undefined", transform=bottom.transAxes,
                            ha="center", fontsize=16, color=".4")
        legend_ax = fig.add_subplot(outer[1, 2])
        legend_ax.axis("off")
        handles = [Line2D([], [], color=".15", label="IllustrisTNG")]
        handles += [Line2D([], [], color=color, ls=line, label=name)
                    for name, color, line in APERTURE_STYLES.values()]
        legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=20,
                         labelspacing=1.1, handlelength=2.8)
        fig.suptitle(r"Halo DM PDFs  |  $z_s=1$  |  $M_{200c}/M_\odot$", y=.985)
        stem = "halo_pdf_b16_3r200c_1r200c_tng_"+group_name
        caption = ("Halo-only positive-DM PDFs at z=1, NSIDE=4096, 120k uniform rays. Both HalfDome curves are "
            "Battaglia16 XGPaint products with identical rays, catalogue, DM cache, M200c windows and histogram bins; "
            "they differ only in the generator-owned external projected aperture, 3R200c (historical product) versus "
            "1R200c (new companion run). Original LOS prescriptions; NOT the later spherical-cut spectra test. "
            "TNG is Ralf Konietzka's catalogue. Total HD is the stored full resolved 1e10-1e16 window. "
            "Mass panels use physical M200c; HD resolution floor is about 7.327e12 Msun. "
            "PDFs are normalized in the original stored [0.1,30000] pc cm^-3 bins; no rebinning or smoothing. "
            "Linear percentages are 100*(HD-TNG)/TNG only where both counts >=10. No synthetic mass windows. "
            "A zero-halo HD window has a delta at DM=0 and no normalized positive-DM PDF. "
            "Each curve is renormalized over its own in-range rays, so the zero-DM ray fractions matter for the "
            "comparison: total window zero fractions are {:.4f} (3R200c) and {:.4f} (1R200c); per-window values are "
            "in analysis/halo_pdf_aperture_comparison_bins.csv.").format(
                products[3.0]["m1e10_to_1e16"]["zero_fraction"], products[1.0]["m1e10_to_1e16"]["zero_fraction"])
        finish(fig, stem, root, book, manifest, caption)
    write_rows(root/"analysis/halo_pdf_aperture_comparison_bins.csv", audit)


SERIES_APERTURES = {
    # multiplier: (short label, colour, line style, line width)
    1.0: (r"$1\,R_{200c}$", "#B2182B", "-", 2.3),
    2.0: (r"$2\,R_{200c}$", "#E08214", "-", 2.3),
    3.0: (r"$3\,R_{200c}$", "#0072B2", "--", 2.5),
    4.0: (r"$4\,R_{200c}$", "#1B9E77", "-", 2.3),
    5.0: (r"$5\,R_{200c}$", "#7B3294", "-", 2.3),
}
REPEAT_3R200C_RUN = "zsrc1p0_nside4096_nrays120000_allhalos_r200cx3p0_m200cprofile_seed42_repeat20260916"
REPEAT_STYLE = ("#E6AB02", ":", 1.4)


def halo_pdf_aperture_series_figures(root, project, book, manifest):
    """Battaglia16 at 1-5 R200c (plus a repeated 3R200c run) versus TNG.

    Same layout as halo_pdf_figures. Every HalfDome product shares rays,
    catalogue, DM cache, M200c windows and bins; only the aperture differs.
    Each top panel lists the percentage of rays that intersect at least one
    resolved halo inside the aperture (1 - zero fraction) for that window.
    """
    direct = Lee22TngBattagliaComparison(project).direct
    direct.validate_fixed_200c_input()
    windows = tuple(dict.fromkeys(e[2] for e in UPPER_ENTRIES + TO_1E14_ENTRIES))
    products = {ap: {w: direct.load_halfdome(1., 4096, ap, w, validated_200c=True) for w in windows}
                for ap in SERIES_APERTURES}
    repeat = {w: direct.load_halfdome(1., 4096, 3.0, w, validated_200c=True, run_name=REPEAT_3R200C_RUN)
              for w in windows}
    reference = products[3.0]
    consistency = {}
    for w in windows:
        for ap in SERIES_APERTURES:
            np.testing.assert_allclose(products[ap][w]["edges"], reference[w]["edges"], rtol=1e-12, atol=0)
            assert np.isclose(products[ap][w]["aperture_r200"], ap)
        np.testing.assert_allclose(repeat[w]["edges"], reference[w]["edges"], rtol=1e-12, atol=0)
        consistency[w] = dict(
            max_abs_count_difference=int(np.max(np.abs(repeat[w]["counts"] - reference[w]["counts"]))),
            total_count_reference=int(reference[w]["counts"].sum()),
            total_count_repeat=int(repeat[w]["counts"].sum()),
            zero_fraction_reference=reference[w]["zero_fraction"],
            zero_fraction_repeat=repeat[w]["zero_fraction"])
    worst = max(v["max_abs_count_difference"] for v in consistency.values())
    audit, hits = [], []
    for group_name, entries, xmax in (("upper_mass_limits", UPPER_ENTRIES, 5000.),
                                      ("to_1e14", TO_1E14_ENTRIES, 10000.)):
        fig = plt.figure(figsize=(17.5, 12.5))
        outer = fig.add_gridspec(2, 3, hspace=.38, wspace=.32,
                                left=.075, right=.985, bottom=.065, top=.89)
        for j, (label, tng_label, hd_label) in enumerate(entries):
            inner = outer[j//3, j%3].subgridspec(2, 1, height_ratios=(2.7, 1.25), hspace=.06)
            top = fig.add_subplot(inner[0])
            bottom = fig.add_subplot(inner[1], sharex=top)
            edges = reference[hd_label]["edges"]
            tng_pdf, tng_count, tng_zero = direct.histogram_from_values(
                direct.tng_values(tng_label, 1.), edges)
            direct.draw_pdf(top, reference[hd_label]["centers"], tng_pdf, color=".15", lw=2.7)
            any_counts = False
            drawn = [(ap, products[ap][hd_label], SERIES_APERTURES[ap][1:]) for ap in SERIES_APERTURES]
            drawn.append(("3.0 repeat", repeat[hd_label], REPEAT_STYLE))
            for ap, product, (color, line, lw) in drawn:
                any_counts |= bool(np.any(product["counts"]))
                direct.draw_pdf(top, product["centers"], product["pdf"], color=color, ls=line, lw=lw)
                delta = direct.percent_difference(product["pdf"], tng_pdf, product["counts"], tng_count)
                bottom.plot(product["centers"], delta, color=color, ls=line, lw=lw)
                for b in range(len(tng_pdf)):
                    audit.append(dict(group=group_name, window=hd_label, aperture_r200c=ap,
                        dm_low=product["edges"][b], dm_high=product["edges"][b+1],
                        tng_pdf=tng_pdf[b], halfdome_pdf=product["pdf"][b],
                        tng_count=int(tng_count[b]), halfdome_count=int(product["counts"][b]),
                        percent_difference=delta[b], tng_zero_fraction=tng_zero,
                        halfdome_zero_fraction=product["zero_fraction"],
                        halfdome_hit_percent=100.*(1.-product["zero_fraction"])))
            # Percentage of rays that pass through at least one halo (DM > 0).
            top.text(.03, .05+.068*len(SERIES_APERTURES), "TNG: {:.1f}%".format(100.*(1.-tng_zero)),
                     transform=top.transAxes, ha="left", va="bottom", fontsize=12.5,
                     color=".15", fontweight="bold")
            for k, ap in enumerate(SERIES_APERTURES):
                hit = 100.*(1.-products[ap][hd_label]["zero_fraction"])
                top.text(.03, .05+.068*(len(SERIES_APERTURES)-1-k), "{:.0f}R: {:.1f}%".format(ap, hit),
                         transform=top.transAxes, ha="left", va="bottom", fontsize=12.5,
                         color=SERIES_APERTURES[ap][1], fontweight="bold")
                if group_name == "upper_mass_limits" or hd_label not in {e[2] for e in UPPER_ENTRIES}:
                    hits.append(dict(window=hd_label, aperture_r200c=ap, hit_percent=hit,
                                     zero_fraction=products[ap][hd_label]["zero_fraction"],
                                     tng_hit_percent=100.*(1.-tng_zero)))
            title = "Total" if j == 0 else label.replace(r"\,M_\odot", "")
            top.set_title(title, pad=10)
            top.set_yscale("log")
            common_axis(top, (.1, xmax))
            common_axis(bottom, (.1, xmax), percent=True)
            top.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            top.tick_params(labelbottom=False)
            bottom.set_xlabel(r"DM [pc cm$^{-3}$]")
            if j % 3 == 0:
                top.set_ylabel(r"$p(\mathrm{DM})$")
                bottom.set_ylabel(r"$\Delta p/p_{\rm TNG}$ [%]", fontsize=18)
            if not any_counts:
                top.text(.58, .3, "HD: no resolved halos", transform=top.transAxes,
                         ha="center", fontsize=16, color=".35")
                bottom.set_yticks([])
                bottom.text(.5, .6, "Undefined", transform=bottom.transAxes,
                            ha="center", fontsize=16, color=".4")
        legend_ax = fig.add_subplot(outer[1, 2])
        legend_ax.axis("off")
        handles = [Line2D([], [], color=".15", label="IllustrisTNG")]
        handles += [Line2D([], [], color=c, ls=ls, lw=lw, label="HalfDome: Battaglia16, "+name)
                    for name, c, ls, lw in SERIES_APERTURES.values()]
        handles.append(Line2D([], [], color=REPEAT_STYLE[0], ls=REPEAT_STYLE[1], lw=REPEAT_STYLE[2],
                              label=r"$3\,R_{200c}$ repeat run (consistency)"))
        legend_ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=17,
                         labelspacing=.9, handlelength=2.8, bbox_to_anchor=(.5, 1.02))
        legend_ax.text(.5, .02, "Panel %: rays with DM > 0 (through $\\geq$1 halo)\n"
                       "TNG: any halo in window; HD: resolved halo in aperture",
                       transform=legend_ax.transAxes, ha="center", va="bottom", fontsize=14, color=".3")
        fig.suptitle(r"Halo DM PDFs  |  $z_s=1$  |  $M_{200c}/M_\odot$", y=.985)
        stem = "halo_pdf_b16_r200c_apertures_tng_"+group_name
        caption = ("Halo-only positive-DM PDFs at z=1, NSIDE=4096, 120k uniform rays. All HalfDome curves are "
            "Battaglia16 XGPaint products with identical rays, catalogue, DM cache, M200c windows and histogram bins; "
            "they differ only in the generator-owned external projected aperture, 1-5 R200c. The 3R200c dashed curve is "
            "the historical 2026-08-26 product; the dotted repeat is a fresh 2026-09-16 run of the same configuration "
            "(largest per-bin count difference over all windows: {}). Original LOS prescriptions; NOT the later "
            "spherical-cut spectra test. TNG is Ralf Konietzka's catalogue. Total HD is the stored full resolved "
            "1e10-1e16 window. Mass panels use physical M200c; HD resolution floor is about 7.327e12 Msun. "
            "PDFs are normalized in the original stored [0.1,30000] pc cm^-3 bins over in-range rays only; no rebinning "
            "or smoothing. Panel percentages give the share of the 120k rays that intersect at least one resolved halo "
            "within the aperture for that window; the remaining rays sit in an undrawn delta at DM=0. "
            "Linear percentages are 100*(HD-TNG)/TNG only where both counts >=10. No synthetic mass windows.").format(worst)
        finish(fig, stem, root, book, manifest, caption)
    write_rows(root/"analysis/halo_pdf_aperture_series_bins.csv", audit)
    write_rows(root/"analysis/halo_hit_percentages.csv", hits)
    (root/"analysis/repeat_3r200c_consistency.json").write_text(json.dumps(consistency, indent=2))
    return consistency


LEE22_1R200C_VARIANTS = {
    # key: (run directory name, expected provenance, label, colour, line style, width)
    "noconc_corrected": dict(
        run="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_noconc_corrected_seed42",
        expect=dict(provenance_lee2022_concentration_mode="none", provenance_lee2022_normalization="baryon_fraction",
                    provenance_lee2022_n0_pivot="mcut"),
        label="HalfDome: Lee22 no-c (Table A2), corrected", color="#D55E00", ls="-", lw=2.5),
    "pref_corrected": dict(
        run="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_pref_corrected_seed42",
        expect=dict(provenance_lee2022_concentration_mode="duffy2008", provenance_lee2022_normalization="baryon_fraction",
                    provenance_lee2022_concentration_source="tng_mean"),
        label="HalfDome: Lee22 best fit (Table 3, c + BPL), corrected", color="#009E73", ls="-", lw=2.5),
    "noconc_literal": dict(
        run="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_noconc_literal_seed42",
        expect=dict(provenance_lee2022_concentration_mode="none", provenance_lee2022_normalization="literal",
                    provenance_lee2022_n0_pivot="legacy_1e14"),
        label="HalfDome: Lee22 no-c, literal eq. 9 (previous)", color="#D55E00", ls=":", lw=2.0),
    "pref_literal": dict(
        run="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_pref_literal_seed42",
        expect=dict(provenance_lee2022_concentration_mode="duffy2008", provenance_lee2022_normalization="literal",
                    provenance_lee2022_concentration_source="duffy2008"),
        label="HalfDome: Lee22 best fit, literal eq. 9 + Duffy08 c", color="#009E73", ls=":", lw=2.0),
    "noconc_corrected_zhyp": dict(
        run="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_noconc_corrected_zhyp_seed42",
        expect=dict(provenance_lee2022_concentration_mode="none", provenance_lee2022_normalization="baryon_fraction",
                    provenance_lee2022_n0_pivot="mcut", provenance_lee2022_redshift_scaling="comoving_hypothesis"),
        label=r"HalfDome: Lee22 no-c, corrected $\times(1+z)^3/E^2$ (hypothesis)", color="#D55E00", ls="-.", lw=2.0),
    "pref_corrected_zhyp": dict(
        run="zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_pref_corrected_zhyp_seed42",
        expect=dict(provenance_lee2022_concentration_mode="duffy2008", provenance_lee2022_normalization="baryon_fraction",
                    provenance_lee2022_concentration_source="tng_mean", provenance_lee2022_redshift_scaling="comoving_hypothesis"),
        label=r"HalfDome: Lee22 best fit, corrected $\times(1+z)^3/E^2$ (hypothesis)", color="#009E73", ls="-.", lw=2.0),
}
# Figure sets: main (corrected only), normalization sensitivity, redshift-scaling hypothesis.
LEE22_FIGURE_SETS = (
    ("", ["noconc_corrected", "pref_corrected"]),
    ("sensitivity_", ["noconc_corrected", "pref_corrected", "noconc_literal", "pref_literal"]),
    ("zscaling_", ["noconc_corrected", "pref_corrected", "noconc_corrected_zhyp", "pref_corrected_zhyp"]),
)


def load_lee22_1r200c(project, key):
    """Load one Lee22 1R200c histogram product and verify its provenance contract."""
    spec = LEE22_1R200C_VARIANTS[key]
    path = Path(project)/"frb_map_generation/outputs"/spec["run"]/"halfdome_uniform_fixedz_foreground_mass_histograms.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    text = lambda v: v.decode() if isinstance(v, bytes) else str(v)
    with h5py.File(path, "r") as h:
        attrs = dict(h.attrs.items())
        labels = [text(v) for v in h["window_label"][:]]
        edges = np.asarray(h["pdf_bin_edges_pc_cm3"][:], float)
        centers = np.asarray(h["pdf_bin_centers_pc_cm3"][:], float)
        density = np.asarray(h["pdf_density_per_pc_cm3"][:], float)
        counts = np.asarray(h["pdf_count"][:], np.int64)
    if density.shape[0] != len(labels):
        density, counts = density.T, counts.T
    contract = dict(provenance_dm_profile="lee2022", provenance_halo_mass_definition="M200c",
                    provenance_halo_radius_definition="R200c", provenance_aperture_geometry_owner="generator",
                    provenance_xgpaint_profile_input_mass_dataset="halo_mass_m200c", **spec["expect"])
    for k, v in contract.items():
        actual = text(attrs.get(k, ""))
        if actual != v:
            raise ValueError("Lee22 provenance mismatch {}: {!r} != {!r} in {}".format(k, actual, v, path))
    numeric = dict(provenance_source_redshift=1.0, provenance_nside=4096, n_rays=120000,
                   provenance_halo_extension_r200_multiplier=1.0, provenance_frb_seed=42)
    for k, v in numeric.items():
        if not np.isclose(float(attrs[k]), v):
            raise ValueError("Lee22 numeric provenance mismatch {}: {} != {} in {}".format(k, attrs[k], v, path))
    if int(attrs["provenance_catalog_streamed_halos"]) != int(attrs["provenance_catalog_total_halos"]):
        raise ValueError("Lee22 product did not stream the complete catalogue: {}".format(path))
    return {label: dict(edges=edges, centers=centers, pdf=density[i], counts=counts[i],
                        zero_fraction=1.-counts[i].sum()/120000., attrs=attrs, path=path)
            for i, label in enumerate(labels)}


def halo_pdf_lee22_figures(root, project, book, manifest):
    """TNG versus Battaglia16 and the two Lee22 density fits, all at 1R200c.

    Figure 1 (per layout group): corrected Lee22 conventions. Figure 2: the same with the
    literal-normalization variants overlaid, showing the size of the implementation check.
    """
    direct = Lee22TngBattagliaComparison(project).direct
    direct.validate_fixed_200c_input()
    windows = tuple(dict.fromkeys(e[2] for e in UPPER_ENTRIES + TO_1E14_ENTRIES))
    b16 = {w: direct.load_halfdome(1., 4096, 1.0, w, validated_200c=True) for w in windows}
    lee = {}
    for key in LEE22_1R200C_VARIANTS:
        try:
            lee[key] = load_lee22_1r200c(project, key)
        except FileNotFoundError as missing:
            print("skipping Lee22 variant {} (missing product: {})".format(key, missing))
    for key, product in lee.items():
        for w in windows:
            np.testing.assert_allclose(product[w]["edges"], b16[w]["edges"], rtol=1e-12, atol=0)
    audit, hits = [], []
    for prefix, keys in LEE22_FIGURE_SETS:
        if any(k not in lee for k in keys):
            print("skipping figure set {!r}: missing {}".format(prefix, [k for k in keys if k not in lee]))
            continue
        sensitivity = prefix == "sensitivity_"
        for group_name, entries, xmax in (("upper_mass_limits", UPPER_ENTRIES, 5000.),
                                          ("to_1e14", TO_1E14_ENTRIES, 10000.)):
            fig = plt.figure(figsize=(17.5, 12.5))
            outer = fig.add_gridspec(2, 3, hspace=.38, wspace=.32,
                                    left=.075, right=.985, bottom=.065, top=.89)
            for j, (label, tng_label, hd_label) in enumerate(entries):
                inner = outer[j//3, j%3].subgridspec(2, 1, height_ratios=(2.7, 1.25), hspace=.06)
                top = fig.add_subplot(inner[0])
                bottom = fig.add_subplot(inner[1], sharex=top)
                edges = b16[hd_label]["edges"]
                tng_pdf, tng_count, tng_zero = direct.histogram_from_values(direct.tng_values(tng_label, 1.), edges)
                direct.draw_pdf(top, b16[hd_label]["centers"], tng_pdf, color=".15", lw=2.7)
                drawn = [("battaglia16", b16[hd_label], "#0072B2", "--", 2.5)]
                drawn += [(k, lee[k][hd_label], LEE22_1R200C_VARIANTS[k]["color"], LEE22_1R200C_VARIANTS[k]["ls"],
                           LEE22_1R200C_VARIANTS[k]["lw"]) for k in keys]
                any_counts = False
                for name, product, color, line, lw in drawn:
                    any_counts |= bool(np.any(product["counts"]))
                    direct.draw_pdf(top, product["centers"], product["pdf"], color=color, ls=line, lw=lw)
                    delta = direct.percent_difference(product["pdf"], tng_pdf, product["counts"], tng_count)
                    bottom.plot(product["centers"], delta, color=color, ls=line, lw=lw)
                    if prefix == "" and (group_name == "upper_mass_limits" or hd_label not in {e[2] for e in UPPER_ENTRIES}):
                        for b in range(len(tng_pdf)):
                            audit.append(dict(window=hd_label, model=name, dm_low=edges[b], dm_high=edges[b+1],
                                tng_pdf=tng_pdf[b], halfdome_pdf=product["pdf"][b], tng_count=int(tng_count[b]),
                                halfdome_count=int(product["counts"][b]), percent_difference=delta[b],
                                halfdome_zero_fraction=product["zero_fraction"]))
                        hits.append(dict(window=hd_label, model=name, hit_percent=100.*(1.-product["zero_fraction"]),
                                         tng_hit_percent=100.*(1.-tng_zero)))
                # rays with DM > 0
                lines = [("TNG", ".15", 100.*(1.-tng_zero))] + [
                    ({"battaglia16": "B16", "noconc_corrected": "Lee22 no-c", "pref_corrected": "Lee22 best"}.get(name, name),
                     color, 100.*(1.-product["zero_fraction"]))
                    for name, product, color, line, lw in drawn if name in ("battaglia16", "noconc_corrected", "pref_corrected")]
                for k, (nm, color, hit) in enumerate(lines):
                    top.text(.03, .05+.068*(len(lines)-1-k), "{}: {:.1f}%".format(nm, hit), transform=top.transAxes,
                             ha="left", va="bottom", fontsize=11.5, color=color, fontweight="bold")
                title = "Total" if j == 0 else label.replace(r"\,M_\odot", "")
                top.set_title(title, pad=10)
                top.set_yscale("log")
                common_axis(top, (.1, xmax))
                common_axis(bottom, (.1, xmax), percent=True)
                # Lee22 differences reach thousands of percent at low DM; keep +-100% readable.
                bottom.set_yscale("symlog", linthresh=100., linscale=1.2)
                bottom.set_yticks([-100, 0, 100, 1000])
                bottom.set_yticklabels(["-100", "0", "100", "1000"])
                top.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
                top.tick_params(labelbottom=False)
                bottom.set_xlabel(r"DM [pc cm$^{-3}$]")
                if j % 3 == 0:
                    top.set_ylabel(r"$p(\mathrm{DM})$")
                    bottom.set_ylabel(r"$\Delta p/p_{\rm TNG}$ [%]", fontsize=18)
                if not any_counts:
                    top.text(.58, .3, "HD: no resolved halos", transform=top.transAxes, ha="center", fontsize=16, color=".35")
                    bottom.set_yticks([])
                    bottom.text(.5, .6, "Undefined", transform=bottom.transAxes, ha="center", fontsize=16, color=".4")
            legend_ax = fig.add_subplot(outer[1, 2])
            legend_ax.axis("off")
            handles = [Line2D([], [], color=".15", label="IllustrisTNG (within $R_{200}$)"),
                       Line2D([], [], color="#0072B2", ls="--", lw=2.5, label=r"HalfDome: Battaglia16, $1\,R_{200c}$")]
            handles += [Line2D([], [], color=LEE22_1R200C_VARIANTS[k]["color"], ls=LEE22_1R200C_VARIANTS[k]["ls"],
                               lw=LEE22_1R200C_VARIANTS[k]["lw"], label=LEE22_1R200C_VARIANTS[k]["label"]) for k in keys]
            legend_ax.legend(handles=handles, loc="upper center", frameon=False, fontsize=14 if prefix else 17,
                             labelspacing=.9, handlelength=2.8, bbox_to_anchor=(.5, 1.02))
            legend_ax.text(.5, .02, "All HalfDome curves: projected $1\\,R_{200c}$ aperture, same rays and halos\n"
                           "Panel %: rays with DM > 0 (through $\\geq$1 halo)",
                           transform=legend_ax.transAxes, ha="center", va="bottom", fontsize=13.5, color=".3")
            fig.suptitle(r"Halo DM PDFs  |  $z_s=1$  |  $M_{200c}/M_\odot$", y=.985)
            stem = "halo_pdf_b16_lee22_1r200c_tng_" + prefix + group_name
            caption = ("Halo-only positive-DM PDFs at z=1, NSIDE=4096, 120k uniform rays; every HalfDome curve uses the same rays, "
                "catalogue, M200c windows, bins and the generator-owned projected 1R200c aperture with the profile's own long LOS. "
                "Battaglia16 is the XGPaint product. Lee22 curves are local implementations of arXiv:2205.01710v1 eq. (9), (10), (12): "
                "'no-c' is the concentration-free fit (arXiv Table 8 = MNRAS Table A2); 'best fit' is Table 3 with concentration and the "
                "mass break, using a mean-concentration proxy (no per-halo scatter). 'Corrected' applies the implementation check: "
                "electron density multiplied by Omega_b/Omega_m (as XGPaint does for Battaglia16), the common M_cut pivot of eq. (12) "
                "for n0, the TNG mean concentrations quoted in Lee22 sec. 2.3, and shape parameters frozen above the fit range "
                "10^14.8 h^-1 Msun so the LOS stays finite. 'Literal' repeats the previous conventions. The '(1+z)^3/E^2 "
                "hypothesis' variants additionally assume the fitted densities were comoving values normalized by the z=0 "
                "critical density; this is a bookkeeping hypothesis suggested by the fitted redshift slopes, not a documented "
                "convention. TNG is Ralf Konietzka's "
                "catalogue; its within-r200 definition is not documented here. PDFs are normalized over in-range rays only; panel "
                "percentages give the rays with DM>0. Percentages 100*(HD-TNG)/TNG only where both counts >=10, drawn on a "
                "symmetric-log axis that is linear within +-100%.")
            finish(fig, stem, root, book, manifest, caption)
    write_rows(root/"analysis/halo_pdf_lee22_1r200c_bins.csv", audit)
    write_rows(root/"analysis/halo_hit_percentages_lee22_1r200c.csv", hits)
    summary = {k: dict(path=str(next(iter(v.values()))["path"]),
                       model_family=(lambda a: a.decode() if isinstance(a, bytes) else str(a))(next(iter(v.values()))["attrs"]["provenance_dm_model_family"]),
                       total_zero_fraction=v["m1e10_to_1e16"]["zero_fraction"]) for k, v in lee.items()}
    (root/"analysis/lee22_1r200c_products.json").write_text(json.dumps(summary, indent=2))


def select_cross(rows, survey, model, n, unbeamed=False):
    return sorted([r for r in rows if r["survey"] == survey and r["model"] == model
                   and int(r["nrays"]) == n and r["filter"] == ("unbeamed" if unbeamed else survey)],
                  key=lambda r: float(r["theta_arcmin"]))


def count_label(n):
    return str(n)+" FRBs" if n < 100 else ("10k FRBs" if n == 10000 else "100k FRBs")


def observed_errorbar(ax, obs):
    ax.errorbar(values(obs, "theta_plotted_arcmin"), 1e5*values(obs, "w_yDM_pc_cm3"),
                yerr=1e5*np.array([values(obs, "error_lower_pc_cm3"), values(obs, "error_upper_pc_cm3")]),
                fmt="D", color=".15", ms=5, capsize=3, lw=1.3, alpha=.75, zorder=4)


def cross_figures(root, inputs, book, manifest):
    rows = read_rows(root/"analysis/source_count_comparison.csv")
    observations = read_rows(inputs/"digitized/takahashi_fig13_approximate.csv")
    for survey, (nobs, beam) in SURVEYS.items():
        # One panel per density model keeps only source-count differences overlaid.
        fig, axes = plt.subplots(2, 3, figsize=(19.5, 8.8), sharex="col",
                                 gridspec_kw={"height_ratios": [2.8, 1.3]})
        xmin = 10 if survey == "planck" else 10**.25
        obs = [r for r in observations if ("ACT" in r["series"]) == (survey == "act")
               and float(r["theta_bin_lower_arcmin"]) >= xmin-1e-10]
        for j, (model, _, _, _) in enumerate(MODELS):
            top, bottom = axes[:, j]
            parent = [r for r in select_cross(rows, survey, model, 100000)
                      if float(r["theta_lower_arcmin"]) >= xmin-1e-10]
            x, full = values(parent, "theta_arcmin"), values(parent, "previous_fullmap")
            valid = abs(full) > .01*np.max(abs(full))
            top.plot(x, full*1e5, color=".5", ls=(0, (6, 3)), lw=1.8, zorder=1)
            observed_errorbar(top, obs)
            for n, style_key in ((nobs, "observed"), (10000, "10k"), (100000, "100k")):
                data = [r for r in select_cross(rows, survey, model, n)
                        if float(r["theta_lower_arcmin"]) >= xmin-1e-10]
                w, sigma = values(data, "cross"), values(data, "sigma")
                color, marker, line = COUNT_STYLES[style_key]
                top.errorbar(x, w*1e5, yerr=sigma*1e5, color=color, marker=marker,
                             ls=line, ms=5, mfc="white", mew=1.5, capsize=3, elinewidth=1.1, zorder=3)
                delta = np.where(valid, 100*(w/full-1), np.nan)
                error = np.where(valid, 100*sigma/abs(full), np.nan)
                bottom.errorbar(x, delta, yerr=error, color=color, marker=marker,
                                ls=line, ms=4, mfc="white", capsize=2, elinewidth=1)
            top.set_title(MODEL_NAMES[model], pad=12)
            for ax in (top, bottom):
                common_axis(ax, (10 if survey == "planck" else 10**.25, 1000), percent=(ax is bottom))
                ax.yaxis.set_major_locator(MaxNLocator(5))
            top.axhline(0, color=".5", lw=1)
            bottom.set_xlabel(r"$\theta$ [arcmin]")
            if j == 0:
                top.set_ylabel(r"$w_{y,\rm DM}$ [$10^{-5}$ pc cm$^{-3}$]")
                bottom.set_ylabel(r"$\Delta w/w_{\rm full}$ [%]", fontsize=18)
        handles = [Line2D([], [], color=c, marker=m, ls=l, mfc="white", label=count_label(n))
                   for n, (c,m,l) in zip((nobs,10000,100000), COUNT_STYLES.values())]
        handles += [Line2D([], [], color=".5", ls="--", label="Full-sky mean"),
                    Line2D([], [], color=".15", marker="D", ls="none", label="Takahashi")]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=18)
        fig.suptitle(survey.upper()+"  |  tSZ x halo DM", y=.985)
        fig.subplots_adjust(left=.08, right=.99, bottom=.19, top=.90, hspace=.09, wspace=.30)
        finish(fig, "takahashi_source_counts_"+survey, root, book, manifest,
            "Nested random catalogues with the observed redshift distribution; all resolved foreground halos, spherical 3R200c DM, "
            "Battaglia12 tSZ pressure. y beam="+str(beam)+" arcmin; no noise or survey mask. "
            "Small-catalogue bars are delete-one-FRB jackknife; independent unused 90k mean-DM(z) calibration is held fixed. "
            "100k values/errors are the previous stratified jackknife results, unchanged. Percentages relative to the full-map mean "
            "are linear, shown only above 1% of its peak absolute amplitude. Preferred Lee22+c is an extrapolation diagnostic. "
            "Takahashi points/errors are approximate digitizations, not exact data vectors or covariance.")


def medlock_figures(root, parent, book, manifest):
    rows = read_rows(root/"analysis/source_count_comparison.csv")
    refs = read_rows(parent/"references/medlock_fig5_best_fit_digitized.csv")
    for model, _, _, _ in MODELS:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.3))
        for ax, panel, survey in zip(axes, ("act", "planck_milca", "planck_nilc"), ("act", "planck", "planck")):
            nobs = SURVEYS[survey][0]
            minimum = 2 if survey == "act" else 10
            for n, key in ((nobs, "observed"), (10000, "10k"), (100000, "100k")):
                data = [r for r in select_cross(rows, survey, model, n, True) if float(r["theta_arcmin"]) >= minimum]
                color, marker, line = COUNT_STYLES[key]
                ax.errorbar(values(data, "theta_arcmin"), values(data, "cross")*1e5,
                            yerr=values(data, "sigma")*1e5, color=color, marker=marker,
                            ls=line, ms=5, mfc="white", capsize=3, elinewidth=1)
            ref = [r for r in refs if r["survey"] == panel]
            x, w, e = [values(ref, k) for k in ("theta_arcmin", "w_yDM_pc_cm3", "digitization_absolute_uncertainty")]
            ax.plot(x, w*1e5, color=".1", lw=2.6)
            ax.fill_between(x, (w-e)*1e5, (w+e)*1e5, color=".3", alpha=.15)
            common_axis(ax, (minimum, 1000))
            ax.axhline(0, color=".5", lw=1)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.set_title(panel.replace("_", " ").upper(), pad=12)
            ax.set_xlabel(r"$\theta$ [arcmin]")
        axes[0].set_ylabel(r"$w_{y,\rm DM}$ [$10^{-5}$ pc cm$^{-3}$]")
        handles = [Line2D([], [], color=c, marker=m, ls=l, mfc="white", label=label)
                   for label, (c,m,l) in zip(("31 / 71 FRBs", "10k FRBs", "100k FRBs"), COUNT_STYLES.values())]
        handles.append(Line2D([], [], color=".1", label=r"BP fit ($z_s=2$)"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=18, frameon=False)
        fig.suptitle(MODEL_NAMES[model]+"  |  BP reference, no beam", y=.99)
        fig.subplots_adjust(left=.08, right=.99, bottom=.24, top=.84, wspace=.30)
        finish(fig, "medlock_source_counts_"+model, root, book, manifest,
            "Black curves digitized from Medlock & Nagai arXiv:2608.06455v1 Figure 5: independent BP best fits to ACT, MILCA and NILC. "
            "These are correlation functions, not 3D density profiles. Published BP uses all FRBs at z=2 and no beam. "
            "Here HD is also unbeamed, but retains the observed redshift distributions, so this is NOT a matched-kernel quantitative test. "
            "BP grey width is image-readout uncertainty, not a posterior interval. Different pressure/cosmology/boundary prescriptions also remain. "
            "HD mock bars are jackknife; preferred Lee22 remains diagnostic only.")


def redshift_figure(root, parent, book, manifest):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    with h5py.File(root/"catalogues/nested_sightlines.h5", "r") as selected, \
            h5py.File(parent/"rays/source_positions.h5", "r") as full:
        for ax, survey in zip(axes, SURVEYS):
            obs = selected[survey+"/observed_redshifts"][:]
            edges = np.linspace(0, 2.2, 23)
            ax.hist(obs, bins=edges, weights=np.ones(len(obs))/len(obs), histtype="step", lw=3.5, color=".2")
            for n, (color, _, line) in ((10000, COUNT_STYLES["10k"]), (100000, COUNT_STYLES["100k"])):
                group = selected[survey+"/n10000"] if n == 10000 else full[survey]
                ax.hist(group["redshift"][:], bins=edges, weights=group["analysis_weight"][:],
                        histtype="step", lw=2.5, color=color, ls=line)
            ax.set(title=survey.upper(), xlabel=r"$z_s$", ylabel="Fraction per bin")
            ax.grid(alpha=.2)
        handles = [Line2D([], [], color=".2", label="Observed / 71 or 31"),
                   Line2D([], [], color=COUNT_STYLES["10k"][0], ls="--", label="10k"),
                   Line2D([], [], color=COUNT_STYLES["100k"][0], ls=":", label="100k")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.86, wspace=.27)
        fig.suptitle("Matched source redshifts", y=.99)
    finish(fig, "source_counts_redshifts", root, book, manifest,
           "Empirical observed redshifts are replicated, not interpolated. The 71/31 subset has one source per observed entry. "
           "10k counts are balanced, with stratum weights restoring the original histogram exactly. Curves deliberately overlap.")


def jackknife_diagnostic(root, book, manifest):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    rows = read_rows(root/"analysis/source_count_comparison.csv")
    with h5py.File(root/"catalogues/nested_sightlines.h5", "r") as data:
        for ax, survey in zip(axes, SURVEYS):
            samples = data[survey+"/observed_count_resampling_cross"][:].reshape(-1, 2, 3, 12)
            # First bin >= 10 arcmin: index 4. Show B16 and Lee22, retaining
            # the preferred diagnostic in the stored arrays, not crowding this plot.
            for p, model in enumerate(("battaglia16", "lee22_legacy")):
                r = select_cross(rows, survey, model, SURVEYS[survey][0])[4]
                ensemble = samples[:, 0, p, 4]*1e5
                color = ("#0072B2", "#D55E00")[p]
                ax.plot(np.sort(ensemble), np.arange(1, len(ensemble)+1)/len(ensemble),
                        color=color, lw=2.4, label=MODEL_NAMES[model])
                center, sigma = float(r["cross"])*1e5, float(r["sigma"])*1e5
                ax.axvline(center, color=color, ls="--", lw=1.5)
                ax.axvspan(center-sigma, center+sigma, color=color, alpha=.1)
            ax.set_title(survey.upper()+"  |  "+str(SURVEYS[survey][0])+" FRBs")
            ax.set_xlabel(r"$w_{y,\rm DM}$ [$10^{-5}$ pc cm$^{-3}$]")
            ax.set_ylabel("Cumulative fraction")
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_ylim(0, 1.03)
            ax.grid(alpha=.18)
            ax.legend(fontsize=16)
    fig.suptitle(r"Jackknife check  |  $10-17.8$ arcmin", y=.99)
    fig.subplots_adjust(left=.09, right=.98, bottom=.21, top=.79, wspace=.27)
    finish(fig, "observed_count_jackknife_check", root, book, manifest,
        "5000 conditional redshift-stratified catalogues drawn from the held-out 10k pool. Dashed vertical lines show the actual selected "
        "71/31 catalogue, shaded widths its delete-one-source 1-sigma jackknife. Empirical CDFs include every draw; horizontal scale is "
        "linear near zero and logarithmic beyond |w/1e-5|=1. These conditional ensembles illustrate skewness and rare-halo sampling; "
        "they are not cosmological or observational mocks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="/home/cbllover/HalfDome")
    parser.add_argument("--output", default="frb_map_generation/outputs/publication_comparison_20260915")
    parser.add_argument("--parent", default="frb_map_generation/outputs/takahashi_100k_20260914")
    parser.add_argument("--inputs", default="frb_map_generation/outputs/observational_comparison_inputs_20260913")
    parser.add_argument("--pdf", default="output/pdf/halfdome_publication_comparisons_20260915.pdf")
    parser.add_argument("--aperture-only", action="store_true",
                        help="Only export the Battaglia16 3R200c versus 1R200c versus TNG halo-PDF figures")
    parser.add_argument("--aperture-series-only", action="store_true",
                        help="Only export the Battaglia16 1-5 R200c (plus 3R200c repeat) versus TNG halo-PDF figures")
    parser.add_argument("--lee22-1r200c-only", action="store_true",
                        help="Only export the TNG versus Battaglia16 versus Lee22 (two fits) 1R200c halo-PDF figures")
    args = parser.parse_args()
    root, parent, pdf = Path(args.output), Path(args.parent), Path(args.pdf)
    (root/"plots").mkdir(parents=True, exist_ok=True)
    (root/"analysis").mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    if args.aperture_only or args.aperture_series_only or args.lee22_1r200c_only:
        maker = (halo_pdf_lee22_figures if args.lee22_1r200c_only else
                 halo_pdf_aperture_series_figures if args.aperture_series_only else halo_pdf_aperture_figures)
        with plt.rc_context(STYLE), PdfPages(pdf) as book:
            maker(root, Path(args.project), book, manifest)
        (root/"figure_captions.json").write_text(json.dumps(manifest, indent=2))
        print("Saved {} figures to {} and {}".format(len(manifest), root/"plots", pdf))
        return
    with plt.rc_context(STYLE), PdfPages(pdf) as book:
        halo_pdf_figures(root, Path(args.project), book, manifest)
        cross_figures(root, Path(args.inputs), book, manifest)
        medlock_figures(root, parent, book, manifest)
        redshift_figure(root, parent, book, manifest)
        jackknife_diagnostic(root, book, manifest)
    (root/"figure_captions.json").write_text(json.dumps(manifest, indent=2))
    print("Saved {} figures to {} and {}".format(len(manifest), root/"plots", pdf))


if __name__ == "__main__":
    main()
