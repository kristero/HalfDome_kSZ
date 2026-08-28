#!/usr/bin/env python3
"""Plot the Battaglia16 and Lee22 projected electron columns on one scale."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


RADIAL_NAME = "battaglia16_vs_lee2022_projected_electron_density_radial.csv"
PNG_NAME = "battaglia16_vs_lee2022_projected_electron_density_2d.png"
PDF_NAME = "battaglia16_vs_lee2022_projected_electron_density_2d.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a side-by-side, shared-color comparison of the physical "
            "projected electron column densities used by the FRB profiles."
        )
    )
    parser.add_argument("radial_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mass-msun", type=float, default=1.0e14)
    parser.add_argument("--redshift", type=float, default=0.5)
    parser.add_argument("--extent-r200c", type=float, default=3.0)
    parser.add_argument("--grid-points", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def validate(args: argparse.Namespace, table: np.ndarray) -> None:
    required = {
        "r_perp_over_r200c",
        "battaglia16_ne_column_cm2",
        "lee2022_no_concentration_ne_column_cm2",
    }
    names = set(table.dtype.names or ())
    missing = required - names
    if missing:
        raise ValueError(f"Missing radial-profile columns: {sorted(missing)}")
    if args.mass_msun <= 0 or not np.isfinite(args.mass_msun):
        raise ValueError("mass-msun must be finite and positive")
    if args.redshift <= 0 or not np.isfinite(args.redshift):
        raise ValueError("redshift must be finite and positive")
    if args.extent_r200c <= 0 or not np.isfinite(args.extent_r200c):
        raise ValueError("extent-r200c must be finite and positive")
    if args.grid_points < 64 or args.grid_points % 2:
        raise ValueError("grid-points must be an even integer of at least 64")


def radial_log_map(
    axis: np.ndarray,
    radii: np.ndarray,
    column: np.ndarray,
    extent_r200c: float,
) -> np.ma.MaskedArray:
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    projected_radius = np.hypot(xx, yy)
    inside = projected_radius <= extent_r200c
    if np.any(projected_radius == 0):
        raise ValueError("Use an even grid so no pixel is centered on R_perp=0")

    log_column = np.full(projected_radius.shape, np.nan, dtype=float)
    log_column[inside] = np.interp(
        np.log10(projected_radius[inside]),
        np.log10(radii),
        np.log10(column),
    )
    return np.ma.masked_invalid(log_column)


def main() -> None:
    args = parse_args()
    table = np.genfromtxt(args.radial_csv, delimiter=",", names=True)
    table = np.atleast_1d(table)
    validate(args, table)

    radii = np.asarray(table["r_perp_over_r200c"], dtype=float)
    battaglia = np.asarray(table["battaglia16_ne_column_cm2"], dtype=float)
    lee2022 = np.asarray(
        table["lee2022_no_concentration_ne_column_cm2"], dtype=float
    )
    if (
        radii.ndim != 1
        or radii.size < 64
        or np.any(~np.isfinite(radii))
        or np.any(radii <= 0)
        or np.any(np.diff(radii) <= 0)
    ):
        raise ValueError("Radial coordinate must be finite, positive, and increasing")
    for label, values in (
        ("Battaglia16", battaglia),
        ("Lee22", lee2022),
    ):
        if values.shape != radii.shape or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{label} column density contains invalid values")
    if radii[0] > args.extent_r200c / args.grid_points:
        raise ValueError("Radial table does not resolve the innermost plotted pixel")
    if radii[-1] < args.extent_r200c:
        raise ValueError("Radial table does not reach the requested aperture")

    axis = np.linspace(
        -args.extent_r200c,
        args.extent_r200c,
        args.grid_points,
        dtype=float,
    )
    battaglia_map = radial_log_map(
        axis, radii, battaglia, args.extent_r200c
    )
    lee_map = radial_log_map(
        axis, radii, lee2022, args.extent_r200c
    )
    finite = np.concatenate(
        (battaglia_map.compressed(), lee_map.compressed())
    )
    vmin, vmax = float(np.min(finite)), float(np.max(finite))

    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 6.1),
        constrained_layout=True,
    )
    titles = (
        "XGPaint Battaglia16",
        "Lee22 Appendix A2 (no concentration)",
    )
    maps = (battaglia_map, lee_map)
    image = None
    extent = (
        -args.extent_r200c,
        args.extent_r200c,
        -args.extent_r200c,
        args.extent_r200c,
    )
    for axis_object, values, title in zip(axes, maps, titles):
        image = axis_object.imshow(
            values,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        axis_object.add_patch(
            Circle(
                (0.0, 0.0),
                1.0,
                fill=False,
                color="white",
                linewidth=1.3,
                linestyle="--",
            )
        )
        axis_object.add_patch(
            Circle(
                (0.0, 0.0),
                args.extent_r200c,
                fill=False,
                color="white",
                linewidth=1.1,
            )
        )
        axis_object.text(
            0.04,
            0.05,
            r"dashed: $R_{200c}$"
            + "\n"
            + rf"edge: ${args.extent_r200c:g}R_{{200c}}$",
            transform=axis_object.transAxes,
            color="white",
            fontsize=10,
            ha="left",
            va="bottom",
        )
        axis_object.set(
            title=title,
            xlabel=r"$x/R_{200c}$",
            ylabel=r"$y/R_{200c}$",
            xlim=(-args.extent_r200c, args.extent_r200c),
            ylim=(-args.extent_r200c, args.extent_r200c),
            aspect="equal",
        )
        axis_object.tick_params(which="both", direction="in")

    if image is None:
        raise RuntimeError("No density image was created")
    colorbar = fig.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.92,
        pad=0.025,
    )
    colorbar.set_label(
        r"$\log_{10}\!\left[N_e/\mathrm{cm}^{-2}\right]$",
        fontsize=13,
    )
    colorbar.ax.tick_params(labelsize=10)
    fig.suptitle(
        "Physical projected electron column density; "
        + rf"$\log_{{10}}(M_{{200c}}/M_\odot)={np.log10(args.mass_msun):.2f}$, "
        + rf"$z_{{halo}}={args.redshift:.2f}$",
        fontsize=14,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / PNG_NAME
    pdf_path = args.output_dir / PDF_NAME
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved shared-color 2D consistency plot: {png_path}")
    print(f"Saved vector copy: {pdf_path}")


if __name__ == "__main__":
    main()
