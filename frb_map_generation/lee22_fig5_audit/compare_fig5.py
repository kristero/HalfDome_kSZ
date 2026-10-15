#!/usr/bin/env python3
"""Compare the actual HalfDome density routine with Lee22's vector Figure 5.

Run export_profiles.jl first. This script needs numpy, scipy, matplotlib and
PyMuPDF. --fetch downloads only the public paper/source, never simulation data.
The figure is named figures/fig_4.pdf in the arXiv source despite being Figure 5
in the compiled paper. It plots x**3 * n_e/n_200, not a projected column.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs/lee22_fig5_audit_20260922"
REFERENCE = OUT / "reference"
FIGURE = REFERENCE / "source/figures/fig_4.pdf"
PROFILE = HERE.parent / "lee2022_frb_dm_profile.jl"
EXPECTED_PROFILE_SHA256 = "1fe10765ba8e38ae42c0cf1d20630babff0b8b10aa2a41e1a617bdbc50ae8806"
COLORS = {"simulation": "#d62728", "with_c_bpl": "#2ca02c",
          "without_c_bpl": "#ff7f0e", "without_c_no_bpl": "#8c564b",
          "battaglia16": "#17becf"}


def fetch_reference():
    import requests
    REFERENCE.mkdir(parents=True, exist_ok=True)
    response = requests.get("https://arxiv.org/src/2205.01710v1", timeout=90)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
        for member in ("figures/fig_4.pdf", "sample631.tex"):
            target = REFERENCE / "source" / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())


def extract_curves():
    """Invert vector coordinates using the figure's labelled log-axis ticks.

    These point coordinates are independent of raster resolution. We check the
    minor x=0.2 tick and all four major y ticks before extracting any curves.
    Saved coordinate precision is not a statistical error estimate.
    """
    with pymupdf.open(FIGURE) as document:
        page = document[0]
        drawings = page.get_drawings()
        assert abs(page.rect.width - 1931.984375) < 0.01
        assert "Without c and with BPL in mass" in page.get_text()
    x01, x1 = 404.0775146484375, 905.6053466796875
    panel_offset = 892.8
    y1 = 60.74560546875
    y1e_minus3 = 397.4245300292969
    y_dex = (y1e_minus3 - y1) / 3
    assert abs(10**((555.0524291992188-x1)/(x1-x01)) - 0.2) < 1e-7
    for y, expected in [(397.4245300292969, -3), (285.1982116699219, -2),
                        (172.971923828125, -1), (y1, 0)]:
        assert abs((y1-y)/y_dex-expected) < 5e-7
    curves = {}
    for column in (0, 1):
        curves[column] = {}
        for name, hex_color in COLORS.items():
            rgb = np.array(matplotlib.colors.to_rgb(hex_color))
            candidates = [d for d in drawings if len(d["items"]) == 102
                          and d["color"] is not None
                          and np.max(np.abs(np.array(d["color"])-rgb)) < 1e-5
                          and d["rect"].y1 < 432
                          and int(d["rect"].x0 > 1032) == column]
            assert len(candidates) == 1, (column, name, len(candidates))
            lines = candidates[0]["items"]
            assert all(item[0] == "l" for item in lines)
            points = [lines[0][1]] + [item[2] for item in lines]
            x = 10**((np.array([p.x for p in points])-x1-column*panel_offset)/(x1-x01))
            q = 10**((y1-np.array([p.y for p in points]))/y_dex)
            curves[column][name] = (x, q)
            assert np.all(np.diff(x) > 0)
    # Independent check: the figure's middle panels use (data-model)/data.
    # Recover them on their linear axis and compare with the top log-axis paths.
    residual_checks = []
    for column in (0, 1):
        rgb = np.array(matplotlib.colors.to_rgb(COLORS["without_c_bpl"]))
        candidates = [d for d in drawings if len(d["items"]) == 102
                      and d["color"] is not None
                      and np.max(np.abs(np.array(d["color"])-rgb)) < 1e-5
                      and 432 < d["rect"].y0 < 570
                      and int(d["rect"].x0 > 1032) == column]
        assert len(candidates) == 1
        lines = candidates[0]["items"]
        y = np.array([lines[0][1].y] + [item[2].y for item in lines])
        plotted_residual = (501.8760070800781-y)*.25/(501.8760070800781-457.7085266113281)
        top_residual = 1-curves[column]["without_c_bpl"][1]/curves[column]["simulation"][1]
        residual_checks.append(float(np.max(np.abs(top_residual-plotted_residual))))
    assert max(residual_checks) < 2e-6
    with (OUT / "figure5_vector_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "curve", "x", "x3_ne_over_n200"])
        for column, panel in curves.items():
            for name, (x, q) in panel.items():
                writer.writerows(zip([column]*len(x), [name]*len(x), x, q))
    return curves, max(residual_checks)


def paper_formula(logmass_hinv, x, z=0.0):
    """Independent evaluation of Table A2, with the common Eq. 12 mass pivot."""
    ratio = 10**(logmass_hinv - 13.61)
    n0 = 6.8 * ratio**0.68 * (1+z)**-2.11
    xc = 7.9 * ratio**(0.47 if ratio < 1 else -0.45) * (1+z)**-0.67
    beta = 19.5 * ratio**(0.70 if ratio < 1 else -0.18) * (1+z)**-0.31
    return n0 * (x/xc)**-0.3 * (1+x/xc)**-beta


def stats(ratio):
    residual = 100 * (ratio-1)
    return {"mean_signed_percent": float(residual.mean()),
            "mean_absolute_percent": float(np.abs(residual).mean()),
            "min_percent": float(residual.min()), "max_percent": float(residual.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_reference()
    assert hashlib.sha256(PROFILE.read_bytes()).hexdigest() == EXPECTED_PROFILE_SHA256
    curves, vector_residual_error = extract_curves()
    data = np.genfromtxt(OUT / "halfdome_profiles.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    factor = float(data["native_factor"][0])
    errors = []
    for cosmology in ("HalfDome", "TNG"):
        group = data[data["cosmology"] == cosmology]
        for mass in np.unique(group["log10_mass_hinv"]):
            for z in (0.0, 0.5, 1.0):
                rows = group[(group["log10_mass_hinv"] == mass) & (group["redshift"] == z)]
                expected = paper_formula(mass, rows["x"], z)
                errors.append(np.max(np.abs(rows["ne_paper_m3"]/rows["n200_paper_m3"]/expected-1)))
    assert max(errors) < 5e-12
    metadata = {
        "reference_url": "https://arxiv.org/src/2205.01710v1",
        "reference_figure_sha256": hashlib.sha256(FIGURE.read_bytes()).hexdigest(),
        "reference_version": "arXiv v1; publisher text was access-blocked",
        "profile_sha256": EXPECTED_PROFILE_SHA256,
        "native_over_paper": factor,
        "native_normalization_change_percent": 100*(factor-1),
        "paper_over_native": 1/factor,
        "max_independent_formula_relative_error": float(max(errors)),
        "max_vector_top_vs_residual_panel_absolute_fraction_error": vector_residual_error,
        "comparison": "3-D x^3 n_e/n_200 at z=0; bin centres, no concentration and BPL",
        "statistical_limitation": "No raw Lee22 halo weights, quartile membership or covariance; not a chi-square test.",
        "panels": [],
    }
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 19, "axes.titlesize": 17,
                         "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1.4]}, layout="constrained")
    output_rows = []
    for column, mass in enumerate((13.3, 13.9)):
        panel = curves[column]
        x, figure_no_c = panel["without_c_bpl"]
        selected = (x >= 0.04) & (x <= 1.34)
        x, figure_no_c = x[selected], figure_no_c[selected]
        rows = data[(data["cosmology"] == "HalfDome") & (data["redshift"] == 0)
                    & np.isclose(data["log10_mass_hinv"], mass, rtol=0, atol=1e-10)]
        # Log interpolation uses exported Julia densities, never a fitted amplitude.
        q_paper = np.exp(np.interp(np.log(x), np.log(rows["x"]),
                          np.log(rows["ne_paper_m3"]/rows["n200_paper_m3"]))) * x**3
        q_native = factor*q_paper
        exact = paper_formula(mass, x)*x**3
        interpolation_error = float(np.max(np.abs(q_paper/exact-1)))
        assert interpolation_error < 0.001
        bin_models = np.array([paper_formula(m, x)*x**3 for m in np.linspace(mass-.1, mass+.1, 101)])
        simulation = panel["simulation"][1][selected]
        # Shape diagnostic only: these fitted nuisance parameters never enter a plotted curve.
        fit = least_squares(lambda p: np.log(paper_formula(p[0], x)*x**3*np.exp(p[1])/figure_no_c),
                            [mass, 0], bounds=([mass-.1, -1], [mass+.1, 1]))
        result = {
            "log10_mass_hinv_centre": mass,
            "physical_mass_bin_HalfDome_msun": [10**(mass-.1)/.68, 10**(mass+.1)/.68],
            "n_comparison_points": len(x),
            "paper_convention_over_figure_no_c": stats(q_paper/figure_no_c),
            "native_over_figure_no_c": stats(q_native/figure_no_c),
            "paper_convention_over_low_c_simulation": stats(q_paper/simulation),
            "native_over_low_c_simulation": stats(q_native/simulation),
            "max_export_interpolation_error": interpolation_error,
            "shape_diagnostic_only": {"effective_logmass": float(fit.x[0]), "amplitude": float(np.exp(fit.x[1])),
                                      "max_absolute_percent": float(100*np.max(np.abs(np.expm1(fit.fun))))},
            "native_bin_envelope_over_figure_min_percent": float(100*np.min(factor*bin_models.min(axis=0)/figure_no_c-1)),
            "native_bin_envelope_over_figure_max_percent": float(100*np.max(factor*bin_models.max(axis=0)/figure_no_c-1)),
        }
        metadata["panels"].append(result)
        upper, lower = axes[:, column]
        upper.plot(x, simulation, color="0.55", lw=1.8, label="TNG: lowest concentration quartile")
        upper.plot(x, figure_no_c, color=COLORS["without_c_bpl"], lw=3, label="Lee22 figure: no concentration + BPL")
        inside = x <= 1
        outside = x >= x[np.flatnonzero(inside)[-1]]
        upper.plot(x[inside], q_paper[inside], color="#0072B2", lw=2.5, ls="--", label="HalfDome code: paper normalization")
        upper.plot(x[inside], q_native[inside], color="#222222", lw=2.5, label="HalfDome code: current PDF normalization")
        upper.plot(x[outside], q_paper[outside], color="#0072B2", lw=2.5, ls=":")
        upper.plot(x[outside], q_native[outside], color="#222222", lw=2.5, ls=":")
        upper.fill_between(x, factor*bin_models.min(axis=0), factor*bin_models.max(axis=0),
                           color="0.3", alpha=.12, label="Shading: mass-bin span (current normalization)")
        lower.plot(x, 100*(q_paper/figure_no_c-1), color="#0072B2", lw=2.5, ls="--")
        lower.plot(x, 100*(q_native/figure_no_c-1), color="#222222", lw=2.5)
        lower.fill_between(x, 100*(factor*bin_models.min(axis=0)/figure_no_c-1),
                           100*(factor*bin_models.max(axis=0)/figure_no_c-1), color="0.3", alpha=.12)
        lower.axhline(0, color="0.6", lw=1)
        upper.set(xscale="log", yscale="log", ylim=(7e-4, 2),
                  title=rf"$10^{{{mass-.1:.1f}}}$-$10^{{{mass+.1:.1f}}}\,h^{{-1}}M_\odot$, $z=0$")
        lower.set(xscale="log", xlim=(.04, 1.34), ylim=(-60, 25), xlabel=r"$r/R_{200c}$")
        for ax in (upper, lower):
            ax.axvline(1, color="0.6", lw=1, ls=":")
            ax.grid(alpha=.18)
            ax.tick_params(which="both", direction="in", top=True, right=True)
        for xx, qr, qn, qp, qs in zip(x, figure_no_c, q_native, q_paper, simulation):
            output_rows.append((column, mass, xx, qr, qn, qp, qs, 100*(qn/qr-1), 100*(qp/qr-1)))
    axes[0,0].set_ylabel(r"$(r/R_{200c})^3\,n_e/n_{200}$")
    axes[1,0].set_ylabel("Difference / Lee22 [%]", fontsize=16)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, fontsize=12.5, frameon=False)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUT/f"halfdome_vs_lee22_figure5.{suffix}", dpi=220)
    plt.close(fig)
    with (OUT/"comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "log10_mass_hinv", "x", "figure_no_c_x3_ne_n200", "halfdome_native_x3_ne_n200",
                         "halfdome_paper_x3_ne_n200", "figure_simulation_x3_ne_n200", "native_percent", "paper_percent"])
        writer.writerows(output_rows)
    (OUT/"audit.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
