#!/usr/bin/env python3
"""Plot the Battaglia16 and Lee22 projected electron columns on one scale."""

from __future__ import annotations

import sys
import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
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
    if args.redshift < 0 or not np.isfinite(args.redshift):
        raise ValueError("redshift must be finite and nonnegative")
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


GRID_RADIAL_NAME = "battaglia16_vs_lee2022_mass_redshift_grid_radial.csv"
GRID_SUMMARY_NAME = "battaglia16_vs_lee2022_mass_redshift_grid_summary.csv"
GRID_PDF_NAME = "battaglia16_vs_lee2022_mass_redshift_2d_grid.pdf"


def parse_grid_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot direct physical Battaglia16 and Lee22 projected-electron-column "
            "maps over a grid of M200c and halo redshift."
        )
    )
    parser.add_argument("radial_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--extent-r200c", type=float, default=3.0)
    parser.add_argument("--grid-points", type=int, default=240)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--grid-test", action="store_true")
    return parser.parse_args()


def validate_grid_table(args: argparse.Namespace, table: np.ndarray) -> None:
    required = {
        "log10_mass_msun",
        "mass_msun",
        "redshift",
        "r_perp_over_r200c",
        "inside_lee22_mass_fit",
        "battaglia16_ne_column_cm2",
        "lee2022_no_concentration_ne_column_cm2",
    }
    names = set(table.dtype.names or ())
    missing = required - names
    if missing:
        raise ValueError(f"Missing grid columns: {sorted(missing)}")
    if args.extent_r200c <= 0 or not np.isfinite(args.extent_r200c):
        raise ValueError("extent-r200c must be finite and positive")
    if args.grid_points < 64 or args.grid_points % 2:
        raise ValueError("grid-points must be an even integer of at least 64")


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _subset(table: np.ndarray, redshift: float, log_mass: float) -> np.ndarray:
    mask = np.isclose(table["redshift"], redshift, rtol=0.0, atol=1.0e-10)
    mask &= np.isclose(
        table["log10_mass_msun"], log_mass, rtol=0.0, atol=1.0e-10
    )
    subset = np.atleast_1d(table[mask])
    if subset.size < 64:
        raise ValueError(
            f"Too few radial samples for logM={log_mass:g}, z={redshift:g}: "
            f"{subset.size}"
        )
    order = np.argsort(subset["r_perp_over_r200c"])
    return subset[order]


def _summary_lookup(path):
    if path is None or not path.is_file():
        return {}
    table = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = {
        "log10_mass_msun",
        "redshift",
        "battaglia16_dm_obs_max_pc_cm3",
        "lee2022_dm_obs_max_pc_cm3",
    }
    if not required.issubset(set(table.dtype.names or ())):
        raise ValueError(f"Summary file has the wrong schema: {path}")
    return {
        (float(row["redshift"]), float(row["log10_mass_msun"])): (
            float(row["battaglia16_dm_obs_max_pc_cm3"]),
            float(row["lee2022_dm_obs_max_pc_cm3"]),
        )
        for row in table
    }


def _draw_reference_circles(axis: plt.Axes) -> None:
    for radius, linestyle, width in ((1.0, "--", 0.9), (1.34, ":", 0.9), (3.0, "-", 0.8)):
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                fill=False,
                color="white",
                linewidth=width,
                linestyle=linestyle,
                alpha=0.9,
            )
        )


def _plot_grid_page(
    table: np.ndarray,
    redshift: float,
    masses: np.ndarray,
    extent: float,
    grid_points: int,
    summary: dict[tuple[float, float], tuple[float, float]],
) -> plt.Figure:
    axis_values = np.linspace(-extent, extent, grid_points, dtype=float)
    rows = []
    ratio_limit = 0.0
    for log_mass in masses:
        subset = _subset(table, redshift, float(log_mass))
        radii = np.asarray(subset["r_perp_over_r200c"], dtype=float)
        battaglia = np.asarray(subset["battaglia16_ne_column_cm2"], dtype=float)
        lee2022 = np.asarray(
            subset["lee2022_no_concentration_ne_column_cm2"], dtype=float
        )
        battaglia_map = radial_log_map(axis_values, radii, battaglia, extent)
        lee_map = radial_log_map(axis_values, radii, lee2022, extent)
        ratio_map = radial_log_map(axis_values, radii, lee2022 / battaglia, extent)
        ratio_limit = max(
            ratio_limit,
            float(np.max(np.abs(ratio_map.compressed()))),
        )
        rows.append(
            (
                float(log_mass),
                bool(int(subset["inside_lee22_mass_fit"][0])),
                battaglia_map,
                lee_map,
                ratio_map,
            )
        )
    ratio_limit = max(ratio_limit, 1.0e-6)

    fig = plt.figure(
        figsize=(15.2, 2.55 * len(rows) + 1.2),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(
        len(rows),
        5,
        width_ratios=(1.0, 1.0, 0.045, 1.0, 0.045),
    )
    ratio_image = None
    extent_tuple = (-extent, extent, -extent, extent)
    absolute_cmap = copy.copy(plt.get_cmap("viridis"))
    ratio_cmap = copy.copy(plt.get_cmap("coolwarm"))
    absolute_cmap.set_bad("#eeeeee")
    ratio_cmap.set_bad("#eeeeee")

    for row_index, (log_mass, inside_fit, battaglia_map, lee_map, ratio_map) in enumerate(rows):
        battaglia_axis = fig.add_subplot(grid[row_index, 0])
        lee_axis = fig.add_subplot(grid[row_index, 1])
        absolute_color_axis = fig.add_subplot(grid[row_index, 2])
        ratio_axis = fig.add_subplot(grid[row_index, 3])

        absolute_values = np.concatenate(
            (battaglia_map.compressed(), lee_map.compressed())
        )
        vmin = float(np.min(absolute_values))
        vmax = float(np.max(absolute_values))
        for panel_axis, values in (
            (battaglia_axis, battaglia_map),
            (lee_axis, lee_map),
        ):
            absolute_image = panel_axis.imshow(
                values,
                origin="lower",
                extent=extent_tuple,
                interpolation="bilinear",
                cmap=absolute_cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            _draw_reference_circles(panel_axis)
        fig.colorbar(absolute_image, cax=absolute_color_axis)
        absolute_color_axis.tick_params(labelsize=7)
        if row_index == 0:
            absolute_color_axis.set_title(r"$\log_{10}N_e$", fontsize=9)

        ratio_image = ratio_axis.imshow(
            ratio_map,
            origin="lower",
            extent=extent_tuple,
            interpolation="bilinear",
            cmap=ratio_cmap,
            norm=TwoSlopeNorm(vmin=-ratio_limit, vcenter=0.0, vmax=ratio_limit),
            rasterized=True,
        )
        _draw_reference_circles(ratio_axis)
        fit_label = "inside Lee22 mass fit" if inside_fit else "mass extrapolation"
        battaglia_axis.set_ylabel(
            rf"$\log_{{10}}M_{{200c}}={log_mass:g}$"
            + "\n"
            + fit_label
            + "\n"
            + r"$y/R_{200c}$",
            fontsize=9,
        )
        dm_values = summary.get((float(redshift), float(log_mass)))
        if dm_values is not None:
            ratio_axis.text(
                0.03,
                0.04,
                rf"max DM$_{{obs}}$: B={dm_values[0]:.2g}, L={dm_values[1]:.2g}",
                transform=ratio_axis.transAxes,
                color="white",
                fontsize=7.5,
                ha="left",
                va="bottom",
                bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.45, "pad": 2.0},
            )
        for panel_axis in (battaglia_axis, lee_axis, ratio_axis):
            panel_axis.set(
                xlim=(-extent, extent),
                ylim=(-extent, extent),
                aspect="equal",
            )
            panel_axis.tick_params(which="both", direction="in", labelsize=8)
            if row_index < len(rows) - 1:
                panel_axis.set_xticklabels([])
            else:
                panel_axis.set_xlabel(r"$x/R_{200c}$", fontsize=9)
        lee_axis.set_yticklabels([])
        ratio_axis.set_yticklabels([])
        if row_index == 0:
            battaglia_axis.set_title("XGPaint Battaglia16", fontsize=11)
            lee_axis.set_title("Lee22 A2, no concentration", fontsize=11)
            ratio_axis.set_title(r"$\log_{10}(N_{e,\mathrm{Lee}}/N_{e,\mathrm{Battaglia}})$", fontsize=11)

    if ratio_image is None:
        raise RuntimeError("No ratio image was generated")
    ratio_color_axis = fig.add_subplot(grid[:, 4])
    ratio_colorbar = fig.colorbar(ratio_image, cax=ratio_color_axis)
    ratio_colorbar.set_label(r"$\log_{10}(N_{e,L}/N_{e,B})$", fontsize=10)
    ratio_color_axis.tick_params(labelsize=8)
    fig.suptitle(
        "Direct physical projected electron columns; "
        + rf"$z_{{halo}}={redshift:g}$; solid: $3R_{{200c}}$, dashed: $R_{{200c}}$, dotted: $1.34R_{{200c}}$",
        fontsize=13,
    )
    return fig


def grid_main() -> None:
    args = parse_grid_args()
    table = np.atleast_1d(np.genfromtxt(args.radial_csv, delimiter=",", names=True))
    validate_grid_table(args, table)
    redshifts = np.unique(np.asarray(table["redshift"], dtype=float))
    masses = np.unique(np.asarray(table["log10_mass_msun"], dtype=float))
    expected_rows = redshifts.size * masses.size
    if expected_rows == 0:
        raise ValueError("The projection-grid table is empty")
    summary_path = args.summary_csv
    if summary_path is None:
        candidate = args.radial_csv.with_name(GRID_SUMMARY_NAME)
        summary_path = candidate if candidate.is_file() else None
    summary = _summary_lookup(summary_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / GRID_PDF_NAME
    with PdfPages(pdf_path) as pdf:
        for redshift in redshifts:
            figure = _plot_grid_page(
                table,
                float(redshift),
                masses,
                args.extent_r200c,
                args.grid_points,
                summary,
            )
            png_path = args.output_dir / (
                "battaglia16_vs_lee2022_mass_redshift_2d_grid_"
                f"z{_tag(float(redshift))}.png"
            )
            figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            print(f"Saved direct-projection page: {png_path}")
    print(f"Saved multi-page direct-projection comparison: {pdf_path}")

if __name__ == "__main__":
    if "--grid-test" in sys.argv:
        grid_main()
    else:
        main()
