#!/usr/bin/env python3
"""
Analyze the host-halo mass versus FRB DM relation using the same XGPaint DM
profile model used by the Julia FRB scripts.

This Python script delegates the DM computation to a Julia backend that uses
XGPaint/HaloDMProfile, then plots histogram-density estimates of

    p(Integrated DM | z-slice, mass-bin)

versus integrated DM for user-chosen halo-mass bins.

It also supports an option to exclude host-halo gas by evaluating the DM
profile at an impact parameter of N * R_vir away from the halo center, with
N configurable.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shlex
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAX_FILENAME_COMPONENT_LENGTH = 240
ALL_REDSHIFT_LOG10_MASS_PDF_BINS = (
    (12.0, 13.0),
    (13.0, 14.0),
    (15.0, 16.0),
    (16.0, 17.0),
)


def parse_bin_edges(text: str) -> np.ndarray:
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if len(tokens) < 2:
        raise argparse.ArgumentTypeError("Need at least two comma-separated bin edges.")

    values = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(math.inf)
        else:
            try:
                values.append(float(token))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"Could not parse bin edge {token!r}.") from exc

    edges = np.asarray(values, dtype=float)
    if not np.all(np.diff(edges) > 0.0):
        raise argparse.ArgumentTypeError("Bin edges must be strictly increasing.")
    return edges


def format_edge(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1.0e4 or abs_value < 1.0e-2:
        return f"{value:.2e}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_mass_bin_label(mass_min: float, mass_max: float) -> str:
    upper = format_edge(mass_max) if np.isfinite(mass_max) else "inf"
    return f"[{format_edge(mass_min)}, {upper}) Msun"


def format_factor_for_filename(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def find_default_catalog(base_dir: Path) -> Path:
    batched_data_dir = base_dir / "batched_data"
    candidates: list[Path] = []
    for path in batched_data_dir.glob("*FRB_catalog*.h5"):
        try:
            if path.is_file():
                path.stat()
                candidates.append(path)
        except FileNotFoundError:
            continue

    candidates.sort(key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any FRB catalog .h5 files in {batched_data_dir}. "
            "Pass --catalog explicitly."
        )
    return candidates[-1]


def compact_filename_component(
    stem: str,
    suffix: str,
    extension: str,
    max_component_length: int = MAX_FILENAME_COMPONENT_LENGTH,
) -> str:
    base_suffix = f"_{suffix}" if suffix else ""
    full_name = f"{stem}{base_suffix}{extension}"
    if len(full_name) <= max_component_length:
        return full_name

    digest = hashlib.sha1(full_name.encode("utf-8")).hexdigest()[:12]
    marker = f"_h{digest}"
    remaining = max_component_length - len(marker) - len(extension)
    if remaining <= 0:
        raise ValueError("Filename length budget is too small to build a compact output filename.")

    suffix_part = base_suffix
    if len(suffix_part) >= remaining:
        suffix_budget = max(0, remaining - 16)
        suffix_part = suffix_part[:suffix_budget].rstrip("._-")

    stem_budget = remaining - len(suffix_part)
    trimmed_stem = stem[: max(1, stem_budget)].rstrip("._-")
    return f"{trimmed_stem}{marker}{suffix_part}{extension}"


def default_backend_table_path(
    catalog_path: Path,
    generate_field_from_scratch: bool,
    exclude_host: bool,
    host_exclusion_rvir_factor: float,
    field_nside: int,
) -> Path:
    mode_tag = f"fieldfromscratch_nside{field_nside}" if generate_field_from_scratch else "profileonly"
    host_tag = (
        f"hostexcluded_{format_factor_for_filename(host_exclusion_rvir_factor)}rvir"
        if exclude_host
        else "hostincluded_center"
    )
    suffix = f"xgpaint_{mode_tag}_{host_tag}"
    filename = compact_filename_component(catalog_path.stem, suffix, ".csv")
    return catalog_path.parent / "derived" / filename


def default_frb_map_path(catalog_path: Path, field_nside: int) -> Path:
    suffix = f"frb_overdensity_nside{field_nside}"
    filename = compact_filename_component(catalog_path.stem, suffix, ".fits")
    return catalog_path.parent / "derived" / filename


def resolve_dm_bin_edges(
    dm_values: np.ndarray,
    bin_count: int,
    dm_min: float | None,
    dm_max: float | None,
    percentile_low: float,
    percentile_high: float,
) -> np.ndarray:
    finite_dm = dm_values[np.isfinite(dm_values)]
    if finite_dm.size == 0:
        raise ValueError("No finite integrated DM values are available for building DM bins.")

    low = float(np.percentile(finite_dm, percentile_low)) if dm_min is None else float(dm_min)
    high = float(np.percentile(finite_dm, percentile_high)) if dm_max is None else float(dm_max)

    if high < low:
        raise ValueError(
            f"Integrated DM range is invalid: dm_max={high} is smaller than dm_min={low}."
        )
    if high == low:
        width = max(abs(high) * 1.0e-3, 1.0)
        high = low + width

    return np.linspace(low, high, bin_count + 1)


def make_axes_grid(panel_count: int) -> tuple[plt.Figure, np.ndarray]:
    ncols = 1 if panel_count == 1 else 2
    nrows = int(math.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), squeeze=False)
    return fig, axes.reshape(-1)


def derive_output_path(base_path: Path, suffix: str) -> Path:
    extension = base_path.suffix or ".jpg"
    filename = compact_filename_component(base_path.stem, suffix, extension)
    return base_path.with_name(filename)


def format_log10_mass_bin_label(log10_mass_min: float, log10_mass_max: float) -> str:
    return f"log10(M/Msun) in [{format_edge(log10_mass_min)}, {format_edge(log10_mass_max)})"


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    savefig_kwargs: dict[str, float] = {}
    if output_path.suffix.lower() not in {".pdf", ".svg", ".eps", ".ps"}:
        savefig_kwargs["dpi"] = 200
    fig.savefig(output_path, **savefig_kwargs)


def apply_log_dm_xaxis(axis: plt.Axes, dm_values: np.ndarray) -> None:
    positive_dm = dm_values[np.isfinite(dm_values) & (dm_values > 0.0)]
    if positive_dm.size == 0:
        raise ValueError("Need at least one positive integrated DM value to use a log-scaled x-axis.")

    xmin = float(np.min(positive_dm))
    xmax = float(np.max(positive_dm))
    if xmax <= xmin:
        xmax = xmin * 1.2

    axis.set_xscale("log")
    axis.set_xlim(xmin, xmax)


def build_log_dm_histogram_grid(dm_values: np.ndarray, bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    positive_dm = dm_values[np.isfinite(dm_values) & (dm_values > 0.0)]
    if positive_dm.size == 0:
        raise ValueError("Need at least one positive integrated DM value to build log-spaced DM bins.")

    xmin = float(np.min(positive_dm))
    xmax = float(np.max(positive_dm))
    if xmax <= xmin:
        xmax = xmin * 1.2

    dm_bin_edges = np.geomspace(xmin, xmax, bin_count + 1)
    dm_centers = np.sqrt(dm_bin_edges[:-1] * dm_bin_edges[1:])
    return dm_bin_edges, dm_centers


def compute_histogram_density(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    counts, _ = np.histogram(finite_values, bins=bin_edges, density=False)
    total = int(np.sum(counts))
    if total == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    bin_widths = np.diff(bin_edges)
    return counts.astype(float) / (total * bin_widths)


def save_overall_dm_pdf_plot(
    output_path: Path,
    dm_values: np.ndarray,
    dm_bin_edges: np.ndarray,
    dm_centers: np.ndarray,
    catalog_name: str,
    field_description: str,
    host_description: str,
) -> int:
    log_dm_bin_edges, log_dm_centers = build_log_dm_histogram_grid(dm_values, len(dm_bin_edges) - 1)
    density = compute_histogram_density(dm_values, log_dm_bin_edges)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.step(log_dm_centers, density, where="mid", linewidth=2.0, color="black")
    axis.set_title(
        "Integrated DM PDF across all redshifts and host-halo masses\n"
        f"Catalog: {catalog_name} | {field_description} | {host_description}"
    )
    axis.set_xlabel(r"Integrated DM [pc cm$^{-3}$]")
    axis.set_ylabel("Estimated p(Integrated DM)")
    apply_log_dm_xaxis(axis, dm_values)
    axis.grid(alpha=0.25)
    axis.text(
        0.98,
        0.95,
        f"N={dm_values.size}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    return int(dm_values.size)


def save_all_redshift_log10_mass_pdf_plot(
    output_path: Path,
    masses: np.ndarray,
    dm_values: np.ndarray,
    dm_bin_edges: np.ndarray,
    dm_centers: np.ndarray,
    catalog_name: str,
    field_description: str,
    host_description: str,
) -> list[tuple[float, float, int]]:
    log10_masses = np.log10(masses)
    log_dm_bin_edges, log_dm_centers = build_log_dm_histogram_grid(dm_values, len(dm_bin_edges) - 1)
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    mass_bin_counts: list[tuple[float, float, int]] = []

    for log10_mass_min, log10_mass_max in ALL_REDSHIFT_LOG10_MASS_PDF_BINS:
        mass_mask = (log10_masses >= log10_mass_min) & (log10_masses < log10_mass_max)
        selected_dm = dm_values[mass_mask]
        count = int(selected_dm.size)
        mass_bin_counts.append((log10_mass_min, log10_mass_max, count))

        density = compute_histogram_density(selected_dm, log_dm_bin_edges)
        label = f"{format_log10_mass_bin_label(log10_mass_min, log10_mass_max)}  N={count}"
        linestyle = "-" if count > 0 else "--"
        axis.step(
            log_dm_centers,
            density,
            where="mid",
            linewidth=1.8,
            linestyle=linestyle,
            label=label,
        )

    axis.set_title(
        "Integrated DM PDF across all redshifts, split by log10 halo-mass bin\n"
        f"Catalog: {catalog_name} | {field_description} | {host_description}"
    )
    axis.set_xlabel(r"Integrated DM [pc cm$^{-3}$]")
    axis.set_ylabel("Estimated p(Integrated DM | log10 mass bin)")
    apply_log_dm_xaxis(axis, dm_values)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    return mass_bin_counts


def save_probability_data(
    output_path: Path,
    dm_bin_edges: np.ndarray,
    z_bins: np.ndarray,
    mass_bins: np.ndarray,
    density_rows: list[np.ndarray],
    count_rows: list[np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        dm_bin_edges=dm_bin_edges,
        z_bins=z_bins,
        mass_bins=mass_bins,
        density_rows=np.asarray(density_rows, dtype=float),
        count_rows=np.asarray(count_rows, dtype=int),
    )


def parse_julia_command(values: list[str] | None) -> list[str]:
    if not values:
        return ["julia"]
    parts: list[str] = []
    for value in values:
        parts.extend(shlex.split(value))
    return parts


def run_julia_backend(
    julia_command: list[str],
    backend_script: Path,
    catalog_path: Path,
    backend_table_path: Path,
    frb_map_output_path: Path | None,
    dm_cache_file: Path,
    generate_field_from_scratch: bool,
    field_nside: int,
    exclude_host: bool,
    host_exclusion_rvir_factor: float,
) -> None:
    backend_table_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *julia_command,
        str(backend_script),
        f"catalog_path={catalog_path}",
        f"output_path={backend_table_path}",
        f"dm_cache_file={dm_cache_file}",
        f"generate_field_from_scratch={str(generate_field_from_scratch).lower()}",
        f"field_nside={field_nside}",
        f"exclude_host={str(exclude_host).lower()}",
        f"host_exclusion_rvir_factor={host_exclusion_rvir_factor}",
    ]
    if frb_map_output_path is not None:
        cmd.append(f"frb_map_output_path={frb_map_output_path}")
    subprocess.run(cmd, check=True)


def load_backend_table(table_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(table_path, delimiter=",", names=True, dtype=float)
    if data.size == 0:
        raise ValueError(f"Backend output table is empty: {table_path}")

    masses = np.atleast_1d(data["sample_mass"]).astype(float)
    redshifts = np.atleast_1d(data["sample_redshift"]).astype(float)
    dm_values = np.atleast_1d(data["dm_xgpaint"]).astype(float)

    valid = np.isfinite(masses) & np.isfinite(redshifts) & np.isfinite(dm_values) & (masses > 0.0)
    if not np.any(valid):
        raise ValueError(f"No finite rows were found in backend table: {table_path}")

    return masses[valid], redshifts[valid], dm_values[valid]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot p(Integrated DM | z-slice, mass-bin) using XGPaint-computed "
            "host-halo DM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Path to an FRB catalog .h5 file. If omitted, the most recent batched_data/*FRB_catalog*.h5 is used.",
    )
    parser.add_argument(
        "--mass-bins",
        type=parse_bin_edges,
        default=parse_bin_edges("1e13,3e13,1e14,3e14,1e15,inf"),
        help="Comma-separated halo-mass bin edges in Msun.",
    )
    parser.add_argument(
        "--z-bins",
        type=parse_bin_edges,
        default=parse_bin_edges("0.0,0.5,1.0,1.5,2.0,2.5"),
        help="Comma-separated redshift bin edges used to approximate p(Integrated DM|z).",
    )
    parser.add_argument(
        "--dm-bin-count",
        type=int,
        default=60,
        help="Number of integrated-DM bins for the histogram density estimate.",
    )
    parser.add_argument(
        "--dm-min",
        type=float,
        default=None,
        help=(
            "Minimum integrated DM on the x-axis. If omitted, it is estimated "
            "from the selected percentile range."
        ),
    )
    parser.add_argument(
        "--dm-max",
        type=float,
        default=None,
        help=(
            "Maximum integrated DM on the x-axis. If omitted, it is estimated "
            "from the selected percentile range."
        ),
    )
    parser.add_argument(
        "--dm-percentile-low",
        type=float,
        default=0.5,
        help=(
            "Lower percentile used for the automatic integrated-DM range when "
            "--dm-min is not set."
        ),
    )
    parser.add_argument(
        "--dm-percentile-high",
        type=float,
        default=99.5,
        help=(
            "Upper percentile used for the automatic integrated-DM range when "
            "--dm-max is not set."
        ),
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum number of FRBs required to draw a mass-bin curve inside a redshift slice.",
    )
    parser.add_argument(
        "--generate-field-from-scratch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Paint the XGPaint Healpix DM field from the catalog host halos and sample it back at the FRB positions, matching the tSZ_cross_FRB workflow. Disable this to use the faster profile-only evaluation mode.",
    )
    parser.add_argument(
        "--field-nside",
        type=int,
        default=4096,
        help="Healpix nside used when --generate-field-from-scratch is enabled.",
    )
    parser.add_argument(
        "--exclude-host",
        action="store_true",
        help="Exclude host-halo gas by sampling the DM field at an offset of N * R_vir away from each host center. In profile-only mode, the same offset is applied directly to the profile evaluation.",
    )
    parser.add_argument(
        "--host-exclusion-rvir-factor",
        type=float,
        default=3.0,
        help="Impact-parameter offset in units of R_vir when --exclude-host is used.",
    )
    parser.add_argument(
        "--julia-command",
        nargs="+",
        default=["julia"],
        help="Command used to launch Julia. For example: --julia-command julia or --julia-command wsl julia",
    )
    parser.add_argument(
        "--julia-backend",
        type=Path,
        default=None,
        help="Path to the Julia backend script. Defaults to the script shipped next to this file.",
    )
    parser.add_argument(
        "--dm-cache-file",
        type=Path,
        default=None,
        help="Path to the XGPaint DM interpolator cache used by the Julia backend.",
    )
    parser.add_argument(
        "--backend-table",
        type=Path,
        default=None,
        help="Optional CSV path for the Julia backend output table.",
    )
    parser.add_argument(
        "--force-backend-recompute",
        action="store_true",
        help="Recompute the Julia/XGPaint backend table even if it already exists.",
    )
    parser.add_argument(
        "--save-frb-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create and save the FRB overdensity Healpix map from the same sampled host catalog used by the tSZ_cross_FRB workflow.",
    )
    parser.add_argument(
        "--frb-map-output",
        type=Path,
        default=None,
        help="Optional FITS output path for the FRB overdensity map. If omitted, a file is written under batched_data/derived/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the plot file. If omitted, a JPG is written under "
            "batched_data/plots/."
        ),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=None,
        help="Optional .npz file that stores the histogram densities and counts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    catalog_path = args.catalog.resolve() if args.catalog is not None else find_default_catalog(base_dir)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    if args.dm_bin_count < 2:
        raise ValueError("--dm-bin-count must be at least 2.")
    if args.min_count < 1:
        raise ValueError("--min-count must be at least 1.")
    if args.field_nside < 1:
        raise ValueError("--field-nside must be positive.")
    if args.dm_percentile_high <= args.dm_percentile_low:
        raise ValueError("--dm-percentile-high must be larger than --dm-percentile-low.")
    if args.host_exclusion_rvir_factor < 0.0:
        raise ValueError("--host-exclusion-rvir-factor must be nonnegative.")
    if not args.save_frb_map and args.frb_map_output is not None:
        raise ValueError("--frb-map-output cannot be used together with --no-save-frb-map.")

    julia_command = parse_julia_command(args.julia_command)
    backend_script = (args.julia_backend or (base_dir / "compute_frb_dm_xgpaint_backend.jl")).resolve()
    if not backend_script.is_file():
        raise FileNotFoundError(f"Julia backend script not found: {backend_script}")

    dm_cache_file = (args.dm_cache_file or (base_dir / "cached_FRB_true_DM_Websky_cosmo.jld2")).resolve()
    backend_table_path = (
        args.backend_table.resolve()
        if args.backend_table is not None
        else default_backend_table_path(
            catalog_path,
            args.generate_field_from_scratch,
            args.exclude_host,
            args.host_exclusion_rvir_factor,
            args.field_nside,
        )
    )
    frb_map_output_path = None
    if args.save_frb_map:
        frb_map_output_path = (
            args.frb_map_output.resolve()
            if args.frb_map_output is not None
            else default_frb_map_path(catalog_path, args.field_nside)
        )

    needs_backend_run = args.force_backend_recompute or not backend_table_path.is_file()
    if frb_map_output_path is not None and not frb_map_output_path.is_file():
        needs_backend_run = True

    if needs_backend_run:
        run_julia_backend(
            julia_command,
            backend_script,
            catalog_path,
            backend_table_path,
            frb_map_output_path,
            dm_cache_file,
            args.generate_field_from_scratch,
            args.field_nside,
            args.exclude_host,
            args.host_exclusion_rvir_factor,
        )

    masses, redshifts, dm_values = load_backend_table(backend_table_path)

    dm_bin_edges = resolve_dm_bin_edges(
        dm_values,
        args.dm_bin_count,
        args.dm_min,
        args.dm_max,
        args.dm_percentile_low,
        args.dm_percentile_high,
    )
    dm_centers = 0.5 * (dm_bin_edges[:-1] + dm_bin_edges[1:])

    if args.output is None:
        field_tag = f"fieldfromscratch_nside{args.field_nside}" if args.generate_field_from_scratch else "profileonly"
        mode_tag = (
            f"{field_tag}_hostexcluded_{format_factor_for_filename(args.host_exclusion_rvir_factor)}rvir"
            if args.exclude_host
            else f"{field_tag}_hostincluded_center"
        )
        output_filename = compact_filename_component(
            catalog_path.stem,
            f"{mode_tag}_pdm_given_z_massbins",
            ".jpg",
        )
        output_path = catalog_path.parent / "plots" / output_filename
    else:
        output_path = args.output
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall_pdf_output_path = derive_output_path(output_path, "allz_allmass_integrated_dm_pdf")
    log10_mass_pdf_output_path = derive_output_path(output_path, "allz_log10massbins_integrated_dm_pdf")

    fig, axes = make_axes_grid(len(args.z_bins) - 1)
    field_description = (
        f"field generated from scratch at nside={args.field_nside}"
        if args.generate_field_from_scratch
        else "profile-only evaluation"
    )
    host_description = (
        f"host excluded at {args.host_exclusion_rvir_factor:g} * Rvir"
        if args.exclude_host
        else "host included at halo center"
    )
    fig.suptitle(
        "XGPaint host-halo integrated DM distribution vs halo mass\n"
        f"Catalog: {catalog_path.name} | {field_description} | {host_description}"
    )

    density_rows: list[np.ndarray] = []
    count_rows: list[np.ndarray] = []

    for z_index, (z_min, z_max) in enumerate(zip(args.z_bins[:-1], args.z_bins[1:])):
        axis = axes[z_index]
        z_mask = (redshifts >= z_min) & (redshifts < z_max)
        z_count = int(np.count_nonzero(z_mask))
        row_densities = np.full((len(args.mass_bins) - 1, len(dm_centers)), np.nan, dtype=float)
        row_counts = np.zeros(len(args.mass_bins) - 1, dtype=int)

        for mass_index, (mass_min, mass_max) in enumerate(zip(args.mass_bins[:-1], args.mass_bins[1:])):
            mass_mask = (masses >= mass_min) & (masses < mass_max)
            selected_dm = dm_values[z_mask & mass_mask]
            count = selected_dm.size
            row_counts[mass_index] = count

            if count < args.min_count:
                continue

            density, _ = np.histogram(selected_dm, bins=dm_bin_edges, density=True)
            row_densities[mass_index, :] = density
            label = f"{format_mass_bin_label(mass_min, mass_max)}  N={count}"
            axis.step(dm_centers, density, where="mid", linewidth=1.8, label=label)

        finite_density = row_densities[np.isfinite(row_densities)]
        if finite_density.size > 0:
            axis.set_ylim(0.0, 1.08 * float(np.max(finite_density)))
        axis.set_title(f"z in [{format_edge(z_min)}, {format_edge(z_max)})   N={z_count}")
        axis.set_xlabel(r"Integrated DM [pc cm$^{-3}$]")
        axis.set_ylabel("Estimated p(Integrated DM | z slice, mass bin)")
        axis.grid(alpha=0.25)
        if np.any(row_counts >= args.min_count):
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                f"No mass-bin subset reached min_count={args.min_count}",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )

        density_rows.append(row_densities)
        count_rows.append(row_counts)

    for axis in axes[len(args.z_bins) - 1 :]:
        axis.set_visible(False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(fig, output_path)
    plt.close(fig)

    overall_count = save_overall_dm_pdf_plot(
        overall_pdf_output_path,
        dm_values,
        dm_bin_edges,
        dm_centers,
        catalog_path.name,
        field_description,
        host_description,
    )
    log10_mass_bin_counts = save_all_redshift_log10_mass_pdf_plot(
        log10_mass_pdf_output_path,
        masses,
        dm_values,
        dm_bin_edges,
        dm_centers,
        catalog_path.name,
        field_description,
        host_description,
    )

    if args.output_data is not None:
        save_probability_data(
            args.output_data,
            dm_bin_edges,
            args.z_bins,
            args.mass_bins,
            density_rows,
            count_rows,
        )

    print(f"Loaded catalog: {catalog_path}")
    print(f"Backend table: {backend_table_path}")
    if frb_map_output_path is not None:
        print(f"FRB overdensity map: {frb_map_output_path}")
    print(f"Saved plot to: {output_path}")
    print(f"Saved all-redshift/all-mass integrated-DM PDF plot to: {overall_pdf_output_path}")
    print(
        "Saved all-redshift integrated-DM PDF split by requested log10 mass bins to: "
        f"{log10_mass_pdf_output_path}"
    )
    if args.output_data is not None:
        print(f"Saved histogram data to: {args.output_data}")
    print(f"Mode: {field_description}; {host_description}")
    print(f"All-redshift/all-mass integrated-DM PDF count: N={overall_count}")
    print("Requested all-redshift log10 mass-bin counts:")
    for log10_mass_min, log10_mass_max, count in log10_mass_bin_counts:
        print(f"  {format_log10_mass_bin_label(log10_mass_min, log10_mass_max)}: N={count}")
    print("Mass-bin counts by redshift slice:")
    for z_index, (z_min, z_max) in enumerate(zip(args.z_bins[:-1], args.z_bins[1:])):
        print(f"  z in [{format_edge(z_min)}, {format_edge(z_max)}):")
        for mass_index, (mass_min, mass_max) in enumerate(zip(args.mass_bins[:-1], args.mass_bins[1:])):
            label = format_mass_bin_label(mass_min, mass_max)
            count = int(count_rows[z_index][mass_index])
            print(f"    {label}: N={count}")


if __name__ == "__main__":
    main()
