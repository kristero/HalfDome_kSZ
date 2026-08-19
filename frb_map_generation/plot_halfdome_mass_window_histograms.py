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
    "m1e10_to_1e13",
    "m1e10_to_1e14",
    "m1e10_to_1e15",
    "m1e10_to_1e16",
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
    labels: list[str],
    requested_min: np.ndarray,
    requested_max: np.ndarray,
    halo_counts: np.ndarray,
    centers: np.ndarray,
    density: np.ndarray,
    n_rays: int,
    source_redshift: float,
) -> None:
    index_by_label = {label: index for index, label in enumerate(labels)}
    missing = [label for label in requested_labels if label not in index_by_label]
    if missing:
        raise KeyError(f"Histogram file is missing windows: {missing}")

    colors = ("cornflowerblue", "purple", "darkorange", "black")
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=110, constrained_layout=True)
    for label, color in zip(requested_labels, colors, strict=True):
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
        rf"HalfDome foreground-halo mass windows, $z_{{src}}={source_redshift:g}$"
        + f"\nSame {n_rays:,} catalogue rays in every curve; zero-DM rays are outside log bins",
        fontsize=13,
    )
    ax.tick_params(axis="both", which="both", labelsize=12, direction="in")
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=10, frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    lower_png = output_dir / "foreground_mass_lower_limit_pdfs.png"
    _plot_group(
        upper_png,
        UPPER_LABELS,
        labels,
        requested_min,
        requested_max,
        halo_counts,
        centers,
        density,
        n_rays,
        source_redshift,
    )
    _plot_group(
        lower_png,
        LOWER_LABELS,
        labels,
        requested_min,
        requested_max,
        halo_counts,
        centers,
        density,
        n_rays,
        source_redshift,
    )
    print(f"Saved {upper_png}")
    print(f"Saved {lower_png}")
    print(f"Saved {long_csv}")


if __name__ == "__main__":
    main()
