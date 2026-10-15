#!/usr/bin/env python3
"""Match all three catalogues at 10^13 <= M200c/Msun < 10^14.

Run with TNG_LEE22_CASE=mass13to14. --prepare writes only the common ray/window
configuration; painting uses the existing local run_tng.py. The final figures
show signed PDF differences on -20..20% axes, without clipping the saved values
or adding out-of-range arrows. Lee22 is extrapolated below its mass-fit limit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare import decode, distribution
from preflight import CASE, GENERATOR, HERE, OUTPUT, PROFILE, RALF, ROOT, sha256

PREVIOUS = ROOT / "frb_map_generation/outputs/tng300_lee22_1r200c_20260921"
COLORS = {"Ralf": "#252525", "TNG": "#1764ab", "HalfDome": "#d45b20"}
LABELS = {"Ralf": "Ralf IllustrisTNG300", "TNG": "Lee22 TNG300-1-Dark", "HalfDome": "Lee22 HalfDome"}


def prepare():
    """Copy the original exact sky directions, changing only the mass window."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "ray_configuration.h5"
    source = PREVIOUS / "halfdome_direct/halo_dm.h5"
    if target.exists():
        raise FileExistsError(f"Preserving existing ray configuration: {target}")
    with h5py.File(source, "r") as old, h5py.File(target, "w") as new:
        for key in ("frb_pixel_ring_1based", "frb_ra_rad", "frb_dec_rad"):
            new[key] = old[key][:]
        new["window_label"] = np.array(["mass13to14"], dtype=h5py.string_dtype())
        new["window_effective_min_msun"] = [1e13]
        new["window_effective_max_msun"] = [1e14]
        new.attrs.update(source_sha256=sha256(source), source_redshift=1.0,
                         seed=42, n_rays=120000, nside=4096,
                         mass_selection="physical Msun; includes Lee22 low-mass extrapolation")
    print(f"Prepared exact reference rays and 10^13--10^14 Msun window: {target}")


def load_painted(path):
    with h5py.File(path, "r") as handle:
        for key, expected in (("profile_sha256", sha256(PROFILE)), ("mass_definition", "M200c"),
                              ("halo_boundary", "spherical"), ("concentration", "none"),
                              ("normalization", "xgpaint_ne2d"), ("n0_pivot", "mcut")):
            assert decode(handle.attrs[key]) == expected, key
        assert handle.attrs["complete"] and handle.attrs["sphere_r200c"] == 1
        assert handle.attrs["mass_extrapolation_below_fit"]
        assert handle.attrs["source_redshift"] == 1 and handle.attrs["seed"] == 42
        assert handle.attrs["direct_quadrature_max_relative_error"] < 1e-6
        np.testing.assert_array_equal(handle["window_effective_min_msun"][:], [1e13])
        np.testing.assert_array_equal(handle["window_effective_max_msun"][:], [1e14])
        dm = handle["dm_pc_cm3"][:]
        assert dm.shape == (1, 120000)
        return dict(dm=dm[0], edges=handle["pdf_bin_edges_pc_cm3"][:],
                    pixels=handle["frb_pixel_ring_1based"][:], counts=handle["pdf_count"][:][0],
                    halo_count=int(handle["window_halo_count"][:][0]))


def percent_difference(numerator, denominator):
    """Keep every computed value; restrict only the displayed y axis."""
    p_num, n_num, _ = numerator
    p_den, n_den, _ = denominator
    valid = (n_num >= 10) & (n_den >= 10)
    delta = np.full(p_num.shape, np.nan)
    delta[valid] = 100 * (p_num[valid] / p_den[valid] - 1)
    return delta, valid


def validate_snapshot_overlap():
    """Require the retained calibrated halos to be identical in the new download."""
    total = 0
    for snapshot in range(50, 100):
        filename = f"snapshot_{snapshot:03d}_selected.h5"
        with h5py.File(PREVIOUS / "tng_snapshots" / filename, "r") as old, \
                h5py.File(OUTPUT / "tng_snapshots" / filename, "r") as new:
            old_keep = old["mass_msun"][:] < 1e14
            new_keep = new["mass_msun"][:] >= float(old.attrs["mass_min_msun"])
            for field in ("halo_id", "mass_msun", "position_comoving_mpc", "r200c_physical_mpc"):
                np.testing.assert_array_equal(old[field][:][old_keep], new[field][:][new_keep])
            total += int(old_keep.sum())
    return dict(snapshots=50, identical_overlap_halos=total,
                fields="halo_id, mass, position, radius", status="pass")


def style_axis(axis, residual=False):
    axis.set_xscale("log")
    axis.set_xlim(.1, 10000)
    axis.grid(alpha=.15)
    axis.tick_params(which="both", direction="in", top=True, right=True)
    if residual:
        axis.set_ylim(-20, 20)
        axis.set_yticks([-20, -10, 0, 10, 20])
        axis.axhline(0, color=".45", lw=1)
    else:
        axis.set_yscale("log")
        axis.set_ylim(1e-7, .08)
        axis.set_ylabel(r"$p(\mathrm{DM}_{\rm halo}\mid\mathrm{DM}_{\rm halo}>0)$")


def save_figure(figure, stem):
    for extension in ("pdf", "png", "svg"):
        figure.savefig(OUTPUT / "plots" / f"{stem}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def write_report(report):
    lines = [
        "# Matched 10^13--10^14 Msun comparison", "",
        "All three curves use 1e13 <= M200c/Msun < 1e14, source redshift 1, and a spherical R200c boundary.",
        "The two painted catalogues retain the original 120,000 random directions and seed 42.",
        "Ralf's existing 120,000-ray array supplies the hydro reference; its ray paths are not matched individually.", "",
        "- [Three-way PDFs and all pairwise percentages](plots/lee22_ralf_tngdark_mass13to14.pdf)",
        "- [HalfDome versus TNG-Dark](plots/lee22_halfdome_tngdark_mass13to14.pdf)",
        "- [Full histogram values and signed percentages](pdf_and_percentages.csv)",
        "- [Statistics and provenance](comparison.json)", "",
        "Both percentage panels use 100*(p_numerator/p_denominator - 1), with the denominator named on the axis.",
        "The Ralf panel shows HalfDome/Ralf and TNG-Dark/Ralf; the Dark panel shows HalfDome/TNG-Dark.",
        "The displayed range is -20% to +20%, with no arrows or replacement markers.",
        "Values outside this range remain unchanged in the plotted lines and CSV and are clipped only by the axes.",
        "Ratios require at least ten rays in both histogram bins. The PDFs condition on DM > 0.", "",
        "## Results", "",
        "DM means have units pc cm^-3. The all-ray mean includes rays that intersect no selected halo.", "",
        "| Catalogue | Mean, all rays | Mean, positive rays | Hit fraction |",
        "|---|---:|---:|---:|",
    ]
    for name in ("Ralf", "TNG", "HalfDome"):
        stats = report["statistics"][name]
        lines.append(f"| {LABELS[name]} | {stats['mean_all']:.5f} | {stats['mean_positive']:.5f} | {stats['hit_fraction']:.5%} |")
    lines.extend([
        "", "## Physics and calibration", "",
        "The fixed Lee22 no-concentration profile, mcut pivot, physical redshift scaling, and XGPaint-native normalization are unchanged.",
        "DM is the electron-density integral through the finite halo chord, divided by (1+z_h), with no contribution outside R200c.",
        "Each simulation retains its native cosmology. No extra baryon multiplier, fitted rescaling, or additional gas scatter is introduced.",
        "**The requested lower mass cut requires extrapolation below the Lee22 fit limit of 1e13 Msun/h**, approximately 1.47--1.48e13 physical Msun.",
        "The existing mass power laws are continued down to 1e13 physical Msun; this is not a claim of calibration in that interval.",
        "Mass cuts now match Ralf exactly. FoF/SUBFIND versus calibration-catalogue differences, periodic TNG structure, and unmatched individual ray paths remain.",
        "The original calibrated-range outputs are retained separately.", "",
        "## Validation and resources", "",
        f"Profile SHA256: `{report['profile_sha256']}`.",
        f"All 50 snapshots passed overlap checks: {report['unchanged_snapshot_overlap']['identical_overlap_halos']:,} retained halos have exactly identical IDs, masses, positions, and radii.",
        "Both painters passed 100 direct-chord quadrature/density checks, including M200c=1e13 Msun.",
        "Saved histograms match the per-ray arrays; painted sightlines, mass edges, and DM edges match exactly.",
        "The newly added mass interval gives nonnegative extra per-ray DM, within floating-point tolerance.", "",
        "| Stage | Peak resident memory [GiB] |", "|---|---:|",
    ])
    for stage, resource in report["resources"].items():
        lines.append(f"| {stage.replace('_', ' ')} | {resource['peak_rss_GiB']:.3f} |")
    lines.extend([
        "", ("Painting and lightcone construction ran locally. The catalogue-transfer stage used idark under a 4 GiB address-space ceiling; no PBS painting job was submitted."
             if "cluster_catalogue_transfer" in report["resources"] else "Painting and lightcone construction ran locally; no PBS painting job was submitted."),
        "The Julia painting ceiling was 12 GiB. Successful snapshot checksums and source copies are retained.", "",
        "## Files changed", "",
        "In `frb_map_generation/tng300_lee22_comparison/`:", "",
        "- Added `compare_mass13to14.py`: common-ray configuration, exact mass-cut validation, signed ratios, figures, tables, and this report.",
        "- Edited `preflight.py`: explicit mass13to14 case and extrapolation metadata.",
        "- Edited `download_tng.py`: safe stdin authentication for the remote transfer and configured mass-selection description.",
        "- Edited `build_lightcone.py`: describe the configured interval accurately in its validation error.",
        "- Edited `paint_tng.jl`: configurable output/ray paths, general mass-window masks, extrapolation metadata, and the additional low-mass quadrature checks.",
        "- Edited `run_tng.py`: pass the output root and optional common-ray configuration to Julia.",
        "- Edited `README.md`: document this separate case, its calibration limitation, and plotting conventions.",
        "", "The upstream Lee22 profile and original HalfDome generator were not edited.",
    ])
    (OUTPUT / "RUN_REPORT.md").write_text("\n".join(lines) + "\n")


def compare():
    overlap_validation = validate_snapshot_overlap()
    paths = {"TNG": OUTPUT / "tng/halo_dm.h5", "HalfDome": OUTPUT / "halfdome_direct/halo_dm.h5"}
    painted = {name: load_painted(path) for name, path in paths.items()}
    np.testing.assert_array_equal(painted["TNG"]["pixels"], painted["HalfDome"]["pixels"])
    assert len(np.unique(painted["TNG"]["pixels"])) == 120000
    edges = painted["TNG"]["edges"]
    np.testing.assert_array_equal(edges, painted["HalfDome"]["edges"])
    centers = np.sqrt(edges[:-1] * edges[1:])
    redshifts = np.load(RALF / "redshifts_DMhalo.npy")
    indices = np.flatnonzero(np.isclose(redshifts, 1.0, atol=1e-8, rtol=0))
    assert len(indices) == 1
    reference_path = RALF / "DMhalo_r200_mass13to14.npy"
    reference = np.load(reference_path, mmap_mode="r")[indices[0]]
    assert reference.shape == (120000,)
    values = {name: item["dm"] for name, item in painted.items()}
    values["Ralf"] = reference
    results = {name: distribution(dm, edges) for name, dm in values.items()}
    for name, item in painted.items():
        np.testing.assert_array_equal(results[name][1], item["counts"])

    pairs = [("HalfDome", "Ralf"), ("TNG", "Ralf"), ("HalfDome", "TNG")]
    residuals = {f"{a}_over_{b}": percent_difference(results[a], results[b]) for a, b in pairs}
    (OUTPUT / "plots").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 18,
                         "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 14,
                         "axes.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    title = r"$10^{13}\leq M_{200c}/M_\odot<10^{14}\quad r\leq R_{200c}\quad z_s=1$"
    for three_way in (False, True):
        ratios = [2.9, 1, 1] if three_way else [3, 1]
        figure, axes = plt.subplots(len(ratios), 1, figsize=(11, 10 if three_way else 8),
                                    sharex=True, gridspec_kw={"height_ratios": ratios, "hspace": .10})
        top = axes[0]
        names = ("Ralf", "TNG", "HalfDome") if three_way else ("TNG", "HalfDome")
        for name in names:
            pdf = results[name][0]
            top.plot(centers, np.where(pdf > 0, pdf, np.nan), color=COLORS[name], lw=2, label=LABELS[name])
        top.set_title(title, pad=12)
        top.legend(frameon=False, loc="lower left")
        style_axis(top)
        if three_way:
            for name in ("TNG", "HalfDome"):
                delta, _ = residuals[f"{name}_over_Ralf"]
                axes[1].plot(centers, delta, color=COLORS[name], lw=1.5)
            axes[1].set_ylabel(r"$\Delta p/p_{\rm Ralf}$ [%]")
        axes[-1].plot(centers, residuals["HalfDome_over_TNG"][0], color=COLORS["HalfDome"], lw=1.5)
        axes[-1].set_ylabel(r"$\Delta p/p_{\rm Dark}$ [%]")
        for axis in axes[1:]:
            style_axis(axis, residual=True)
        axes[-1].set_xlabel(r"$\mathrm{DM}_{\rm halo}\ [\mathrm{pc\,cm^{-3}}]$")
        figure.text(.51, .008, "Lee22 includes extrapolation below its calibrated mass range", ha="center", fontsize=12)
        figure.subplots_adjust(bottom=.105 if three_way else .13)
        save_figure(figure, "lee22_ralf_tngdark_mass13to14" if three_way else "lee22_halfdome_tngdark_mass13to14")

    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        row = dict(dm_left=left, dm_right=right)
        for name in values:
            row[f"pdf_{name}"] = results[name][0][index]
            row[f"count_{name}"] = int(results[name][1][index])
        for pair, (delta, valid) in residuals.items():
            row[f"percent_{pair}"] = float(delta[index]) if valid[index] else ""
        rows.append(row)
    with (OUTPUT / "pdf_and_percentages.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # With exactly the same rays and geometry, new minus previous per-ray DM
    # measures the contribution of the newly included lower mass interval.
    extension = {}
    for name, subdir in (("TNG", "tng"), ("HalfDome", "halfdome_direct")):
        with h5py.File(PREVIOUS / subdir / "halo_dm.h5", "r") as old:
            np.testing.assert_array_equal(painted[name]["pixels"], old["frb_pixel_ring_1based"][:])
            added = values[name] - old["dm_pc_cm3"][0]
            assert added.min() > -1e-8
            extension[name] = dict(added_mass_min_msun=1e13,
                                   added_mass_max_msun=float(old["window_effective_min_msun"][0]),
                                   added_mean_dm=float(added.mean()),
                                   fraction_of_new_mean=float(added.mean() / values[name].mean()))
    report = dict(status="complete", mass_min_msun=1e13, mass_max_msun=1e14,
                  profile_sha256=sha256(PROFILE),
                  source_redshift=1, seed=42, n_rays=120000, sphere_r200c=1,
                  percentage_definition="100 * (p_numerator / p_denominator - 1)",
                  display_percent_limits=[-20, 20], out_of_range_markers=False,
                  saved_percentages_clipped=False, minimum_bin_counts_both=10,
                  statistics={name: result[2] for name, result in results.items()},
                  halo_counts={name: item["halo_count"] for name, item in painted.items()},
                  added_lower_mass_interval=extension,
                  mass_cut_matched_to_ralf=True, low_mass_lee22_extrapolation=True,
                  unchanged_snapshot_overlap=overlap_validation,
                  catalogue_sha256={name: sha256(path) for name, path in paths.items()},
                  ralf_array_sha256=sha256(reference_path),
                  limitations=["Lee22 is extrapolated below 1e13/h physical Msun.",
                               "Matching mass cuts does not match individual halos or gas-cell ray paths.",
                               "TNG uses the public FoF/SUBFIND catalogue rather than the calibration Rockstar catalogue.",
                               "One periodic TNG box and one observer; cosmic variance is not measured."])
    report["percentage_pairs"] = {
        pair: dict(plotted_bins=int(valid.sum()), above_20=int(np.count_nonzero(delta > 20)),
                   below_minus20=int(np.count_nonzero(delta < -20)))
        for pair, (delta, valid) in residuals.items()}
    report["resources"] = {}
    for stage, filename in (("halfdome_painting", "halfdome_direct/paint_resources.txt"),
                            ("tng_painting", "tng/paint_resources.txt"),
                            ("lightcone", "lightcone_resources.txt"),
                            ("cluster_catalogue_transfer", "cluster_transfer_resources.txt")):
        resource_file = OUTPUT / filename
        if not resource_file.exists() and stage == "cluster_catalogue_transfer":
            continue  # The documented workflow can also download locally.
        resource_text = resource_file.read_text()
        status = re.search(r"Exit status:\s*(\d+)", resource_text)
        rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", resource_text)
        assert status and int(status.group(1)) == 0 and rss
        report["resources"][stage] = dict(peak_rss_GiB=int(rss.group(1)) / 1024**2, exit_status=0)
    receipts = [json.loads(path.read_text()) for path in (OUTPUT / "tng_snapshots").glob("*selected.json")]
    assert len(receipts) == 50
    report["download"] = dict(snapshots=50, validated_transfer_bytes=sum(
        field["bytes"] for receipt in receipts for field in receipt["downloads"]),
        note="Successful extraction receipts; excludes interrupted-transfer bytes.")
    frozen = OUTPUT / "code"
    frozen.mkdir(exist_ok=True)
    sources = [*HERE.glob("*.py"), *HERE.glob("*.jl"), HERE / "README.md"]
    for source in sources:
        shutil.copy2(source, frozen / source.name)
    report["source_sha256"] = {source.name: sha256(source) for source in sources}
    for name, path in paths.items():
        used = path.parent / "paint_used.jl"
        with h5py.File(path, "r") as handle:
            assert sha256(used) == decode(handle.attrs["painter_sha256"])
            assert sha256(GENERATOR) == decode(handle.attrs["reused_generator_sha256"])
        shutil.copy2(used, frozen / f"paint_{name}_used.jl")
    for source in (PROFILE, GENERATOR):
        shutil.copy2(source, frozen / source.name)
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report(report)
    print(json.dumps(report["statistics"], indent=2))
    print(f"Saved matched mass13to14 figures and signed percentages: {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if CASE != "mass13to14":
        parser.error("Set TNG_LEE22_CASE=mass13to14 for this explicitly extrapolated case.")
    prepare() if args.prepare else compare()


if __name__ == "__main__":
    main()
