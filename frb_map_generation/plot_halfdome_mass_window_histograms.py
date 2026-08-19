#!/usr/bin/env python3
"""Plot foreground-halo DM PDFs from the histogram-only Julia HDF5 output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


UPPER_LABELS = (
    "m1e10_to_1e12",
    "m1e10_to_1e13",
    "m1e10_to_1e14",
    "m1e10_to_1e15",
    "m1e10_to_1e16",
)
TO_1E14_LABELS = (
    "m1e10_to_1e14",
    "m1e11_to_1e14",
    "m1e12_to_1e14",
    "m1e13_to_1e14",
)
LOWER_LABELS = (
    "m1e12_to_1e16",
    "m1e13_to_1e16",
    "m1e14_to_1e16",
    "m1e15_to_1e16",
)


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _window_bin_matrix(values: np.ndarray, nwindow: int, nbin: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape == (nwindow, nbin):
        return array
    if array.shape == (nbin, nwindow):
        return array.T
    raise ValueError(f"{name} has shape {array.shape}; expected {(nwindow, nbin)}")


def _mass_label(lower: float, upper: float) -> str:
    lo = int(round(np.log10(lower)))
    hi = int(round(np.log10(upper)))
    return rf"$10^{{{lo}}}\leq M_{{\rm halo}}/M_\odot<10^{{{hi}}}$"


def _plot_group(
    output_path: Path,
    requested_labels: tuple[str, ...],
    group_description: str,
    labels: list[str],
    requested_min: np.ndarray,
    requested_max: np.ndarray,
    halo_counts: np.ndarray,
    centers: np.ndarray,
    density: np.ndarray,
    n_rays: int,
    source_redshift: float,
) -> bool:
    index_by_label = {label: index for index, label in enumerate(labels)}
    available_labels = tuple(label for label in requested_labels if label in index_by_label)
    if not available_labels:
        print(f"Skipping {group_description}: none of its mass windows are stored.")
        return False
    missing = [label for label in requested_labels if label not in index_by_label]
    if missing:
        print(f"{group_description}: skipping unavailable windows {missing}")

    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(available_labels)))
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=110, constrained_layout=True)
    for label, color in zip(available_labels, colors):
        index = index_by_label[label]
        positive = density[index] > 0.0
        legend = (
            f"{_mass_label(requested_min[index], requested_max[index])} "
            f"($N_{{halo}}$={int(halo_counts[index]):,})"
        )
        ax.plot(
            centers[positive],
            density[index, positive],
            color=color,
            linewidth=2.6,
            label=legend,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(centers[0]), float(centers[-1]))
    ax.set_xlabel(r"Foreground-halo DM [pc cm$^{-3}$]", fontsize=15)
    ax.set_ylabel(r"$p({\rm DM}\mid z)$", fontsize=15)
    ax.set_title(
        rf"HalfDome {group_description}, $z_{{src}}={source_redshift:g}$"
        + f"\nSame {n_rays:,} rays in every curve; zero-DM rays are outside log bins",
        fontsize=13,
    )
    ax.tick_params(axis="both", which="both", labelsize=12, direction="in")
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=10, frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_catalog_histogram(
    output_path: Path,
    edges: np.ndarray,
    counts: np.ndarray,
    xlabel: str,
    title: str,
    *,
    log_x: bool,
) -> None:
    step_x = np.repeat(edges, 2)[1:-1]
    step_y = np.repeat(counts, 2)
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=110, constrained_layout=True)
    positive = step_y > 0
    ax.plot(step_x[positive], step_y[positive], color="navy", linewidth=2.2)
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Number of foreground halos per bin", fontsize=15)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", which="both", labelsize=12, direction="in")
    ax.grid(alpha=0.22, which="both")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_catalog_histogram_csv(
    output_path: Path, edges: np.ndarray, counts: np.ndarray, coordinate: str
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["bin_index", f"{coordinate}_left", f"{coordinate}_right", f"{coordinate}_center", "count"]
        )
        centers = (
            np.sqrt(edges[:-1] * edges[1:])
            if coordinate == "mass_msun"
            else 0.5 * (edges[:-1] + edges[1:])
        )
        for index, center in enumerate(centers):
            writer.writerow([index + 1, edges[index], edges[index + 1], center, int(counts[index])])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_hdf5", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_path = args.input_hdf5.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        labels = _decode_strings(h5["window_label"][:])
        edges = np.asarray(h5["pdf_bin_edges_pc_cm3"][:], dtype=float)
        centers = np.asarray(h5["pdf_bin_centers_pc_cm3"][:], dtype=float)
        requested_min = np.asarray(h5["window_requested_min_msun"][:], dtype=float)
        requested_max = np.asarray(h5["window_requested_max_msun"][:], dtype=float)
        effective_min = np.asarray(h5["window_effective_min_msun"][:], dtype=float)
        effective_max = np.asarray(h5["window_effective_max_msun"][:], dtype=float)
        halo_counts = np.asarray(h5["window_halo_count"][:], dtype=np.int64)
        counts = _window_bin_matrix(h5["pdf_count"][:], len(labels), len(centers), "pdf_count")
        density = _window_bin_matrix(
            h5["pdf_density_per_pc_cm3"][:], len(labels), len(centers), "pdf_density_per_pc_cm3"
        )
        n_rays = int(h5.attrs.get("n_rays", h5.attrs.get("provenance_nfrb_actual", -1)))
        source_redshift = float(np.asarray(h5["source_redshift_grid"][:]).ravel()[0])
        per_ray_saved = bool(h5.attrs.get("per_ray_dm_saved", "dm_pc_cm3" in h5))
        diagnostic_names = (
            "foreground_halo_mass_bin_edges_msun",
            "foreground_halo_mass_histogram_count",
            "foreground_halo_redshift_bin_edges",
            "foreground_halo_redshift_histogram_count",
        )
        if all(name in h5 for name in diagnostic_names):
            foreground_mass_edges = np.asarray(
                h5["foreground_halo_mass_bin_edges_msun"][:], dtype=float
            )
            foreground_mass_counts = np.asarray(
                h5["foreground_halo_mass_histogram_count"][:], dtype=np.int64
            )
            foreground_redshift_edges = np.asarray(
                h5["foreground_halo_redshift_bin_edges"][:], dtype=float
            )
            foreground_redshift_counts = np.asarray(
                h5["foreground_halo_redshift_histogram_count"][:], dtype=np.int64
            )
        else:
            foreground_mass_edges = None
            foreground_mass_counts = None
            foreground_redshift_edges = None
            foreground_redshift_counts = None

    if n_rays <= 0:
        raise ValueError("Could not determine the common ray count from HDF5 metadata.")
    if per_ray_saved:
        print("Warning: input contains per-ray DM data; plotting only its stored histograms.")

    long_csv = output_dir / "foreground_mass_window_histogram_bins.csv"
    with long_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window_index",
                "label",
                "requested_min_msun",
                "requested_max_msun",
                "effective_min_msun",
                "effective_max_msun",
                "n_rays",
                "in_range_count",
                "bin_index",
                "bin_left_pc_cm3",
                "bin_right_pc_cm3",
                "bin_center_pc_cm3",
                "count",
                "density_per_pc_cm3",
            ]
        )
        for window_index, label in enumerate(labels):
            in_range = int(counts[window_index].sum())
            for bin_index, center in enumerate(centers):
                writer.writerow(
                    [
                        window_index + 1,
                        label,
                        requested_min[window_index],
                        requested_max[window_index],
                        effective_min[window_index],
                        effective_max[window_index],
                        n_rays,
                        in_range,
                        bin_index + 1,
                        edges[bin_index],
                        edges[bin_index + 1],
                        center,
                        int(counts[window_index, bin_index]),
                        density[window_index, bin_index],
                    ]
                )

    upper_png = output_dir / "foreground_mass_upper_limit_pdfs.png"
    to_1e14_png = output_dir / "foreground_mass_to_1e14_pdfs.png"
    lower_png = output_dir / "foreground_mass_lower_limit_pdfs.png"
    saved_plots = []
    if _plot_group(
        upper_png,
        UPPER_LABELS,
        "upper-limit mass windows",
        labels,
        requested_min,
        requested_max,
        halo_counts,
        centers,
        density,
        n_rays,
        source_redshift,
    ):
        saved_plots.append(upper_png)
    if _plot_group(
        to_1e14_png,
        TO_1E14_LABELS,
        r"mass windows ending at $10^{14}\,M_\odot$",
        labels,
        requested_min,
        requested_max,
        halo_counts,
        centers,
        density,
        n_rays,
        source_redshift,
    ):
        saved_plots.append(to_1e14_png)
    if _plot_group(
        lower_png,
        LOWER_LABELS,
        "lower-limit mass windows",
        labels,
        requested_min,
        requested_max,
        halo_counts,
        centers,
        density,
        n_rays,
        source_redshift,
    ):
        saved_plots.append(lower_png)

    if foreground_mass_edges is not None:
        foreground_count = int(foreground_redshift_counts.sum())
        mass_png = output_dir / "foreground_halo_mass_histogram.png"
        redshift_png = output_dir / "foreground_halo_redshift_histogram.png"
        mass_csv = output_dir / "foreground_halo_mass_histogram.csv"
        redshift_csv = output_dir / "foreground_halo_redshift_histogram.csv"
        _plot_catalog_histogram(
            mass_png,
            foreground_mass_edges,
            foreground_mass_counts,
            r"Foreground halo mass $M_{200c}$ [$M_\odot$]",
            rf"All HalfDome foreground halos with $0\leq z\leq {source_redshift:g}$",
            log_x=True,
        )
        _plot_catalog_histogram(
            redshift_png,
            foreground_redshift_edges,
            foreground_redshift_counts,
            "Foreground halo redshift",
            (
                f"Redshift distribution of {foreground_count:,} HalfDome foreground halos "
                rf"with $0\leq z\leq {source_redshift:g}$"
            ),
            log_x=False,
        )
        _write_catalog_histogram_csv(
            mass_csv, foreground_mass_edges, foreground_mass_counts, "mass_msun"
        )
        _write_catalog_histogram_csv(
            redshift_csv,
            foreground_redshift_edges,
            foreground_redshift_counts,
            "redshift",
        )
        saved_plots.extend((mass_png, redshift_png))
        print(f"Saved {mass_csv}")
        print(f"Saved {redshift_csv}")
    else:
        print("Input HDF5 has no foreground halo mass/redshift diagnostic histograms.")

    for path in saved_plots:
        print(f"Saved {path}")
    print(f"Saved {long_csv}")


if __name__ == "__main__":
    main()
