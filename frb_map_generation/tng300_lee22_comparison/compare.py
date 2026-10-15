#!/usr/bin/env python3
"""Validate completed catalogues and plot matched PDFs plus Ralf's broader reference."""

from __future__ import annotations

import csv
import json
import re
import shutil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

from preflight import GENERATOR, HERE, OUTPUT, PROFILE, RALF, WINDOWS, sha256

COLORS = {"TNG300-1-Dark": "#1764ab", "HalfDome": "#d45b20"}
MASS_LABELS = [r"$1.48\times10^{13}\leq M_{200c}<10^{14}$",
               r"$10^{14}\leq M_{200c}<9.28\times10^{14}$",
               r"$1.48\times10^{13}\leq M_{200c}<9.28\times10^{14}$"]


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def distribution(values, edges):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Invalid DM values")
    positive = values[values > 0]
    counts = np.histogram(positive, edges)[0]
    pdf = counts / (len(positive) * np.diff(edges)) if len(positive) else np.zeros(len(edges) - 1)
    under = np.count_nonzero((values > 0) & (values < edges[0]))
    over = np.count_nonzero(values > edges[-1])
    zeros = np.count_nonzero(values == 0)
    assert int(counts.sum()) + under + over + zeros == len(values)
    metrics = dict(n_rays=len(values), positive_count=len(positive), hit_fraction=len(positive) / len(values),
                   zero_fraction=zeros / len(values), mean_all=float(np.mean(values)),
                   std_all=float(np.std(values)), mean_positive=float(np.mean(positive)),
                   median_positive=float(np.median(positive)), underflow_positive=int(under), overflow=int(over),
                   plotted_positive_probability=float(np.sum(pdf * np.diff(edges))))
    return pdf, counts, metrics


def load(path):
    with h5py.File(path, "r") as handle:
        for key, expected in (("mass_definition", "M200c"), ("halo_boundary", "spherical"),
                              ("concentration", "none"), ("normalization", "xgpaint_ne2d"),
                              ("n0_pivot", "mcut"), ("profile_sha256", sha256(PROFILE))):
            assert decode(handle.attrs[key]) == expected, f"Wrong profile provenance: {key}"
        assert handle.attrs["complete"] and handle.attrs["sphere_r200c"] == 1.0
        assert handle.attrs["source_redshift"] == 1.0 and handle.attrs["seed"] == 42
        assert handle.attrs["direct_quadrature_max_relative_error"] < 1e-6
        data = {key: handle[key][:] for key in ("dm_pc_cm3", "frb_pixel_ring_1based", "window_label",
                "window_effective_min_msun", "window_effective_max_msun", "window_halo_count",
                "pdf_bin_edges_pc_cm3", "pdf_count")}
    assert data["dm_pc_cm3"].shape == (3, 120000)
    assert [decode(s) for s in data["window_label"]] == [s[0] for s in WINDOWS]
    np.testing.assert_array_equal(data["window_effective_min_msun"], [s[1] for s in WINDOWS])
    np.testing.assert_array_equal(data["window_effective_max_msun"], [s[2] for s in WINDOWS])
    np.testing.assert_allclose(data["dm_pc_cm3"][0] + data["dm_pc_cm3"][1], data["dm_pc_cm3"][2], atol=1e-9, rtol=1e-12)
    assert data["window_halo_count"][0] + data["window_halo_count"][1] == data["window_halo_count"][2]
    return data


def save_figure(figure, stem):
    for extension in ("pdf", "png", "svg"):
        figure.savefig(OUTPUT / "plots" / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    paths = {"HalfDome": OUTPUT / "halfdome_direct/halo_dm.h5", "TNG300-1-Dark": OUTPUT / "tng/halo_dm.h5"}
    data = {name: load(path) for name, path in paths.items()}
    hd, tng = data["HalfDome"], data["TNG300-1-Dark"]
    np.testing.assert_array_equal(hd["frb_pixel_ring_1based"], tng["frb_pixel_ring_1based"])
    np.testing.assert_array_equal(hd["pdf_bin_edges_pc_cm3"], tng["pdf_bin_edges_pc_cm3"])
    edges = hd["pdf_bin_edges_pc_cm3"]
    centers = np.sqrt(edges[:-1] * edges[1:])
    results, bin_rows, statistics = {}, [], []
    for name, entry in data.items():
        results[name] = []
        for index, (label, lower, upper) in enumerate(WINDOWS):
            pdf, counts, metrics = distribution(entry["dm_pc_cm3"][index], edges)
            np.testing.assert_array_equal(counts, entry["pdf_count"][index])
            metrics.update(simulation=name, window=label, mass_min_msun=lower, mass_max_msun=upper,
                           halo_count=int(entry["window_halo_count"][index]))
            statistics.append(metrics)
            results[name].append((pdf, counts, metrics))
            for left, right, value, count in zip(edges[:-1], edges[1:], pdf, counts):
                bin_rows.append(dict(simulation=name, window=label, dm_left=float(left), dm_right=float(right),
                                     count=int(count), conditional_pdf_per_pc_cm3=float(value)))
    (OUTPUT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 19, "axes.titlesize": 17,
                         "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
                         "axes.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(2, 3, figsize=(18, 8), sharex="col", sharey="row",
                                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.18})
    distances = []
    for index, label in enumerate(MASS_LABELS):
        top, bottom = axes[:, index]
        for name in ("TNG300-1-Dark", "HalfDome"):
            pdf, counts, metrics = results[name][index]
            top.plot(centers, np.where(pdf > 0, pdf, np.nan), color=COLORS[name], lw=2, label=name)
        hp, hc, hm = results["HalfDome"][index]
        tp, tc, tm = results["TNG300-1-Dark"][index]
        delta = np.full_like(hp, np.nan)
        valid = (hc >= 10) & (tc >= 10)
        delta[valid] = 100 * (hp[valid] / tp[valid] - 1)
        bottom.plot(centers, delta, color=COLORS["HalfDome"], lw=1.5)
        bottom.axhline(0, color=".4", lw=1)
        bottom.set_ylim(-100, 100)
        bottom.set_yticks([-100, 0, 100])
        above = np.isfinite(delta) & (delta > 100)
        bottom.scatter(centers[above], np.full(np.count_nonzero(above), 95), marker="^", s=20, color=COLORS["HalfDome"])
        top.set(xscale="log", yscale="log", xlim=(0.1, 10000), ylim=(1e-7, 0.08), title=label)
        top.text(.04, .06, f"Hit fraction\nTNG {tm['hit_fraction']:.1%}\nHD   {hm['hit_fraction']:.1%}",
                 transform=top.transAxes, fontsize=13)
        bottom.set_xlabel(r"$\mathrm{DM}_{\rm halo}\ [\mathrm{pc\,cm^{-3}}]$")
        for axis in (top, bottom):
            axis.grid(alpha=.15, which="major")
            axis.tick_params(which="both", direction="in", top=True, right=True)
        test = ks_2samp(hd["dm_pc_cm3"][index], tng["dm_pc_cm3"][index])
        distances.append(dict(window=WINDOWS[index][0], ks_distance_all_rays=float(test.statistic),
                              mean_percent_hd_minus_tng=100 * (hm["mean_all"] / tm["mean_all"] - 1),
                              halo_abundance_percent_hd_minus_tng=100 * (hm["halo_count"] / tm["halo_count"] - 1)))
    axes[0, 0].set_ylabel(r"$p(\mathrm{DM}_{\rm halo}\mid\mathrm{DM}_{\rm halo}>0)$")
    axes[1, 0].set_ylabel(r"$\Delta p/p_{\rm TNG}$ [%]")
    axes[0, 0].legend(frameon=False)
    figure.suptitle(r"Lee22 no-c $\quad r\leq R_{200c}\quad z_s=1\quad [M_{200c}]=M_\odot$", y=.99, fontsize=20)
    save_figure(figure, "tng300_dark_vs_halfdome_supported_mass_pdfs")

    redshifts = np.load(RALF / "redshifts_DMhalo.npy")
    indices = np.flatnonzero(np.isclose(redshifts, 1., rtol=0, atol=1e-8))
    assert len(indices) == 1
    reference_path = RALF / "DMhalo_r200_mass13to14.npy"
    reference = np.load(reference_path, mmap_mode="r")[int(indices[0])]
    rpdf, _, rmetrics = distribution(reference, edges)
    figure, axis = plt.subplots(figsize=(10, 6.7))
    axis.plot(centers, np.where(rpdf > 0, rpdf, np.nan), color=".15", lw=2.1,
              label=r"Ralf TNG hydro: $10^{13}$--$10^{14}\,M_\odot$")
    for name in ("TNG300-1-Dark", "HalfDome"):
        pdf = results[name][0][0]
        axis.plot(centers, np.where(pdf > 0, pdf, np.nan), color=COLORS[name], lw=2,
                  label=name + r": $1.48\times10^{13}$--$10^{14}\,M_\odot$")
    axis.set(xscale="log", yscale="log", xlim=(.1, 5000), ylim=(1e-7, .08),
             xlabel=r"$\mathrm{DM}_{\rm halo}\ [\mathrm{pc\,cm^{-3}}]$",
             ylabel=r"$p(\mathrm{DM}_{\rm halo}\mid\mathrm{DM}_{\rm halo}>0)$",
             title=r"Ralf reference: different mass cuts $\quad z_s=1$")
    axis.legend(frameon=False, fontsize=14)
    axis.grid(alpha=.15)
    save_figure(figure, "ralf_reference_different_mass_cuts")

    for filename, rows in (("statistics.csv", statistics), ("pdf_bins.csv", bin_rows)):
        with (OUTPUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = dict(status="TNG_and_HalfDome_supported_mass_comparison_complete_Ralf_context_only",
                  checks={"same_mass_edges": "pass", "same_sightlines": "pass", "same_dm_edges": "pass",
                          "disjoint_window_dm_additivity": "pass", "saved_histogram_counts": "pass"},
                  statistics=statistics, comparisons=distances,
                  ralf_reference=dict(mass_min_msun=1e13, mass_max_msun=1e14, metrics=rmetrics,
                                      exact_mass_match=False, sha256=sha256(reference_path)),
                  catalogue_sha256={name: sha256(path) for name, path in paths.items()},
                  limitations=["Ralf's broad mass bin cannot be rebinned to the strict supported interval.",
                               "FoF/SUBFIND TNG halos differ from Lee22's Rockstar calibration catalogue.",
                               "Single periodic TNG volume and one observer; finite-box and observer variance are not measured.",
                               "Snapshot-centre lightcone with full halo spheres, not Voronoi gas-cell ray tracing.",
                               "Same profile code in each simulation's native cosmology; smooth mean profiles without halo gas scatter.",
                               "The established profile continues inward of its 0.04 R200c fitted radial limit.",
                               "The XGPaint-native normalization is the established successful convention, not independent physical validation."])
    resources = {}
    for name, relative in (("halfdome_direct", "halfdome_direct/paint_resources.txt"),
                           ("tng_direct", "tng/paint_resources.txt"),
                           ("lightcone", "lightcone_resources.txt")):
        text = (OUTPUT / relative).read_text()
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        status = re.search(r"Exit status:\s*(\d+)", text)
        assert status and int(status.group(1)) == 0
        resources[name] = dict(peak_rss_GiB=int(rss.group(1)) / 1024**2, exit_status=0)
    report["resources"] = resources
    receipts = [json.loads(path.read_text()) for path in (OUTPUT / "tng_snapshots").glob("*selected.json")]
    assert len(receipts) == 50
    report["download"] = dict(snapshots=len(receipts),
                              validated_transfer_bytes=sum(field["bytes"] for record in receipts for field in record["downloads"]),
                              retained_snapshot_bytes=sum(record["selected_bytes"] for record in receipts),
                              transfer_note="Successful input receipts; excludes discarded bytes from interrupted attempts.")
    frozen = OUTPUT / "code"
    frozen.mkdir(exist_ok=True)
    sources = [*HERE.glob("*.py"), *HERE.glob("*.jl"), HERE / "README.md"]
    for source in sources:
        shutil.copy2(source, frozen / source.name)
    report["source_sha256"] = {source.name: sha256(source) for source in sources}
    upstream = frozen / "upstream"
    upstream.mkdir(exist_ok=True)
    for source in (PROFILE, GENERATOR):
        shutil.copy2(source, upstream / source.name)
    report["upstream_source_sha256"] = {source.name: sha256(source) for source in (PROFILE, GENERATOR)}
    report["used_painter_sha256"] = {}
    for name, path in paths.items():
        used_painter = path.parent / "paint_used.jl"
        with h5py.File(path, "r") as handle:
            assert sha256(used_painter) == decode(handle.attrs["painter_sha256"])
            assert sha256(GENERATOR) == decode(handle.attrs["reused_generator_sha256"])
        shutil.copy2(used_painter, frozen / f"paint_{name}_used.jl")
        report["used_painter_sha256"][name] = sha256(used_painter)
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(distances, indent=2))
    print(f"Plots and validation saved under {OUTPUT}")


if __name__ == "__main__":
    main()
