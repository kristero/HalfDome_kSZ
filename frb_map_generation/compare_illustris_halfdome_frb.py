#!/usr/bin/env python3
"""Compare the legacy IllustrisTNG FRB-DM notebook with HalfDome catalogues.

The legacy notebook ``DM_halo_pdf_IllustrisTNG_5.ipynb`` plots *foreground*
halo-DM sums after cutting the masses of intervening halos.  The current
HalfDome COSMOS2020 catalogue stores one total line-of-sight DM and the *host*
halo mass.  Host-mass cuts can reproduce the layout of the legacy figures, but
they are not the same physical statistic.  This module keeps that distinction
explicit and supplies a separate comparison of the abundance-matching (SHMR)
prescriptions at identical virial halo mass.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COSMOS2020_CATALOG = REPO_ROOT / (
    "frb_map_generation/outputs/stellar_weighted_allz_frb_los_COSMO20_n10M/"
    "stellar_weighted_frb_los_allredshifts_zsrcmin0p0_zsrcmax6p0_"
    "cosmos2020_alpha1p0_nside4096_nfrb10000000_seed42_hosts.csv"
)
DEFAULT_LEGACY_REDSHIFTS = (
    REPO_ROOT / "frb_map_generation/outputs/redshifts_DMhalo.npy"
)
DEFAULT_LEGACY_ALL_DM = REPO_ROOT / (
    "frb_map_generation/outputs/DMhalo_r200_all_ralf_konietzka.npy"
)
DEFAULT_PUBLIC_V2_Z1 = (
    REPO_ROOT / "haloDM/Konietzka2025_DMmap_halomass10_v2.hdf5"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "frb_catalog_comparison_outputs"

PUBLIC_V2_Z1_BYTES = 80_006_944
PUBLIC_V2_Z1_MD5 = "830033d20ee0b024f1b9ee90f80831bc"

DM_EDGES = np.logspace(np.log10(0.1), np.log10(3.0e4), 300)

# Ordered to match the two plots in DM_halo_pdf_IllustrisTNG_5.ipynb.
UPPER_LIMIT_WINDOWS = OrderedDict(
    [
        ("all", (None, None)),
        ("mass10to12", (1.0e10, 1.0e12)),
        ("mass10to13", (1.0e10, 1.0e13)),
        ("mass10to14", (1.0e10, 1.0e14)),
        ("mass10to15", (1.0e10, 1.0e15)),
    ]
)

LOWER_LIMIT_WINDOWS = OrderedDict(
    [
        ("all", (None, None)),
        ("mass10to14", (1.0e10, 1.0e14)),
        ("mass11to14", (1.0e11, 1.0e14)),
        ("mass12to14", (1.0e12, 1.0e14)),
        ("mass13to14", (1.0e13, 1.0e14)),
    ]
)

WINDOW_LABELS = {
    "all": "all hosts",
    "mass10to12": r"$10^{10}<M_{\rm host}/M_\odot<10^{12}$",
    "mass10to13": r"$10^{10}<M_{\rm host}/M_\odot<10^{13}$",
    "mass10to14": r"$10^{10}<M_{\rm host}/M_\odot<10^{14}$",
    "mass10to15": r"$10^{10}<M_{\rm host}/M_\odot<10^{15}$",
    "mass11to14": r"$10^{11}<M_{\rm host}/M_\odot<10^{14}$",
    "mass12to14": r"$10^{12}<M_{\rm host}/M_\odot<10^{14}$",
    "mass13to14": r"$10^{13}<M_{\rm host}/M_\odot<10^{14}$",
}

UPPER_LIMIT_COLORS = {
    "all": "mediumblue",
    "mass10to12": "cornflowerblue",
    "mass10to13": "purple",
    "mass10to14": "orange",
    "mass10to15": "gold",
}

LOWER_LIMIT_COLORS = {
    "all": "black",
    "mass10to14": "orange",
    "mass11to14": "turquoise",
    "mass12to14": "limegreen",
    "mass13to14": "green",
}


# Shuntov et al. (2022) / COSMOS2020 central-SHMR parameters.
# Columns: z_min, z_max, log10(M1), log10(Mstar0), beta, delta, gamma.
COSMOS2020_PARAMS = np.array(
    [
        [0.2, 0.5, 12.629, 10.855, 0.487, 0.935, 1.939],
        [0.5, 0.8, 12.793, 10.927, 0.502, 0.802, 3.132],
        [0.8, 1.1, 12.730, 11.013, 0.454, 1.109, 1.925],
        [1.1, 1.5, 12.673, 10.967, 0.393, 0.746, 0.335],
        [1.5, 2.0, 12.787, 11.040, 0.410, 0.716, 1.312],
        [2.0, 2.5, 13.097, 11.254, 0.495, 0.668, 1.077],
        [2.5, 3.0, 12.627, 10.920, 0.393, 0.274, 0.446],
        [3.0, 3.5, 12.820, 11.067, 0.465, 0.354, 0.741],
        [3.5, 4.5, 13.638, 12.222, 0.551, 1.557, 3.149],
        [4.5, 5.5, 13.547, 12.105, 0.567, 1.427, 3.225],
    ],
    dtype=float,
)

H0_COSMOS2020 = 70.0
H0_HALFDOME = 68.0
OMEGA_M_HALFDOME = 0.31


@dataclass(frozen=True)
class CatalogSlice:
    """A finite redshift slice from the HalfDome FRB host catalogue."""

    redshift: np.ndarray
    halo_mass_msun: np.ndarray
    stellar_mass_msun: np.ndarray
    dm_pc_cm3: np.ndarray
    z_target: float
    z_half_width: float
    rows_scanned: int
    source_path: Path

    @property
    def size(self) -> int:
        return int(self.dm_pc_cm3.size)

    @property
    def z_bounds(self) -> tuple[float, float]:
        return self.z_target - self.z_half_width, self.z_target + self.z_half_width


@dataclass(frozen=True)
class HistogramCurve:
    key: str
    mass_min: float | None
    mass_max: float | None
    selected_count: int
    plotted_count: int
    counts: np.ndarray
    density: np.ndarray
    dm_values: np.ndarray


def _concatenate(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=float)
    return np.concatenate(parts)


def load_catalog_redshift_slice(
    catalog_path: str | Path = DEFAULT_COSMOS2020_CATALOG,
    *,
    z_target: float = 1.0,
    z_half_width: float = 0.05,
    chunksize: int = 500_000,
    max_rows: int | None = None,
) -> CatalogSlice:
    """Stream the large FRB CSV and retain only a narrow host-redshift slice."""

    path = Path(catalog_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HalfDome FRB catalogue not found: {path}")
    if z_half_width <= 0.0:
        raise ValueError("z_half_width must be positive.")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive.")

    required = [
        "host_redshift",
        "host_halo_mass_msun",
        "host_stellar_mass_msun",
        "dm_pc_cm3",
    ]
    z_parts: list[np.ndarray] = []
    mass_parts: list[np.ndarray] = []
    stellar_mass_parts: list[np.ndarray] = []
    dm_parts: list[np.ndarray] = []

    z_min = z_target - z_half_width
    z_max = z_target + z_half_width
    rows_scanned = 0

    reader = pd.read_csv(path, usecols=required, chunksize=chunksize)
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - rows_scanned
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        rows_scanned += len(chunk)
        z = chunk["host_redshift"].to_numpy(dtype=float, copy=False)
        mass = chunk["host_halo_mass_msun"].to_numpy(dtype=float, copy=False)
        stellar_mass = chunk["host_stellar_mass_msun"].to_numpy(
            dtype=float, copy=False
        )
        dm = chunk["dm_pc_cm3"].to_numpy(dtype=float, copy=False)

        keep = (
            np.isfinite(z)
            & (z >= z_min)
            & (z < z_max)
            & np.isfinite(mass)
            & (mass > 0.0)
            & np.isfinite(stellar_mass)
            & (stellar_mass > 0.0)
            & np.isfinite(dm)
        )
        if np.any(keep):
            z_parts.append(np.asarray(z[keep], dtype=float))
            mass_parts.append(np.asarray(mass[keep], dtype=float))
            stellar_mass_parts.append(np.asarray(stellar_mass[keep], dtype=float))
            dm_parts.append(np.asarray(dm[keep], dtype=float))

        if max_rows is not None and rows_scanned >= max_rows:
            break

    result = CatalogSlice(
        redshift=_concatenate(z_parts),
        halo_mass_msun=_concatenate(mass_parts),
        stellar_mass_msun=_concatenate(stellar_mass_parts),
        dm_pc_cm3=_concatenate(dm_parts),
        z_target=float(z_target),
        z_half_width=float(z_half_width),
        rows_scanned=int(rows_scanned),
        source_path=path,
    )
    if result.size == 0:
        raise ValueError(
            f"No finite rows found in {z_min:g} <= host_redshift < {z_max:g}."
        )
    return result


def mass_window_mask(
    masses: np.ndarray,
    mass_min: float | None,
    mass_max: float | None,
) -> np.ndarray:
    mask = np.isfinite(masses) & (masses > 0.0)
    if mass_min is not None:
        mask &= masses >= mass_min
    if mass_max is not None:
        mask &= masses < mass_max
    return mask


def histogram_density(
    values: np.ndarray,
    edges: np.ndarray = DM_EDGES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return counts and density with NumPy's ``density=True`` convention."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    counts, _ = np.histogram(finite, bins=edges, density=False)
    total = int(counts.sum())
    if total == 0:
        density = np.zeros(len(edges) - 1, dtype=float)
    else:
        density = counts.astype(float) / (total * np.diff(edges))
    return counts, density


def build_catalog_curves(
    catalog_slice: CatalogSlice,
    windows: Mapping[str, tuple[float | None, float | None]],
    *,
    edges: np.ndarray = DM_EDGES,
) -> OrderedDict[str, HistogramCurve]:
    curves: OrderedDict[str, HistogramCurve] = OrderedDict()
    for key, (mass_min, mass_max) in windows.items():
        selected = mass_window_mask(
            catalog_slice.halo_mass_msun, mass_min, mass_max
        )
        dm_values = catalog_slice.dm_pc_cm3[selected]
        counts, density = histogram_density(dm_values, edges)
        curves[key] = HistogramCurve(
            key=key,
            mass_min=mass_min,
            mass_max=mass_max,
            selected_count=int(selected.sum()),
            plotted_count=int(counts.sum()),
            counts=counts,
            density=density,
            dm_values=dm_values,
        )
    return curves


def summarize_catalog_curves(
    catalog_slice: CatalogSlice,
    curves: Mapping[str, HistogramCurve],
    *,
    edges: np.ndarray = DM_EDGES,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | None]] = []
    for key, curve in curves.items():
        plotted = curve.dm_values[
            np.isfinite(curve.dm_values)
            & (curve.dm_values >= edges[0])
            & (curve.dm_values <= edges[-1])
        ]
        row: dict[str, float | int | str | None] = {
            "window": key,
            "mass_min_msun": curve.mass_min,
            "mass_max_msun": curve.mass_max,
            "selected_count": curve.selected_count,
            "plotted_count": curve.plotted_count,
            "fraction_of_redshift_slice_percent": (
                100.0 * curve.selected_count / catalog_slice.size
            ),
        }
        if plotted.size:
            row.update(
                {
                    "mean_dm": float(np.mean(plotted)),
                    "median_dm": float(np.median(plotted)),
                    "std_dm": float(np.std(plotted, ddof=1))
                    if plotted.size > 1
                    else np.nan,
                    "p2p5_dm": float(np.percentile(plotted, 2.5)),
                    "p97p5_dm": float(np.percentile(plotted, 97.5)),
                }
            )
        else:
            row.update(
                {
                    "mean_dm": np.nan,
                    "median_dm": np.nan,
                    "std_dm": np.nan,
                    "p2p5_dm": np.nan,
                    "p97p5_dm": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_catalog_curves(
    catalog_slice: CatalogSlice,
    windows: Mapping[str, tuple[float | None, float | None]],
    colors: Mapping[str, str],
    *,
    xlim: tuple[float, float],
    output_path: str | Path | None = None,
    edges: np.ndarray = DM_EDGES,
) -> tuple[plt.Figure, OrderedDict[str, HistogramCurve], pd.DataFrame]:
    """Reproduce the legacy visual layout using catalogue *host* mass cuts."""

    curves = build_catalog_curves(catalog_slice, windows, edges=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, ax = plt.subplots(
        figsize=(13, 7.5),
        dpi=110,
        constrained_layout=True,
    )

    for key, curve in curves.items():
        positive = curve.density > 0.0
        label = f"{WINDOW_LABELS[key]}  (N={curve.selected_count:,})"
        if np.any(positive):
            ax.plot(
                centers[positive],
                curve.density[positive],
                linewidth=3,
                color=colors[key],
                label=label,
                zorder=2 if key == "mass10to14" else 0,
            )
        else:
            ax.plot([], [], linewidth=3, color=colors[key], label=label)

    z_min, z_max = catalog_slice.z_bounds
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.tick_params(
        axis="both", direction="in", which="major", reset=True, labelsize=14
    )
    ax.tick_params(
        axis="both", direction="in", which="minor", reset=True, labelsize=14
    )
    ax.minorticks_on()
    ax.set_xlabel(r"DM [pc cm$^{-3}$]", fontsize=16)
    ax.set_ylabel(r"$p({\rm DM}\mid z)$", fontsize=16)
    ax.set_title(
        "HalfDome COSMOS2020 catalogue: host-mass selection\n"
        rf"${z_min:.2f}\leq z_{{\rm host}}<{z_max:.2f}$ (not foreground-mass decomposition)",
        fontsize=13,
    )
    ax.legend(fontsize=12, framealpha=1.0, bbox_to_anchor=(1.05, 1))
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    summary = summarize_catalog_curves(catalog_slice, curves, edges=edges)
    return fig, curves, summary


def extract_legacy_dm_at_redshift(
    dm_path: str | Path = DEFAULT_LEGACY_ALL_DM,
    redshifts_path: str | Path = DEFAULT_LEGACY_REDSHIFTS,
    *,
    z_target: float = 1.0,
) -> np.ndarray:
    """Extract one redshift row from a legacy ``(Nz, Nrays)`` or transposed array."""

    dm_path = Path(dm_path)
    redshifts_path = Path(redshifts_path)
    if not dm_path.is_file():
        raise FileNotFoundError(f"Legacy DM array not found: {dm_path.resolve()}")
    if not redshifts_path.is_file():
        raise FileNotFoundError(
            f"Legacy redshift array not found: {redshifts_path.resolve()}"
        )

    redshifts = np.asarray(np.load(redshifts_path), dtype=float)
    dm = np.load(dm_path, mmap_mode="r")
    matches = np.flatnonzero(
        np.round(redshifts, 2) == np.round(float(z_target), 2)
    )
    if matches.size == 0:
        raise ValueError(f"z={z_target} is absent from {redshifts_path}.")
    index = int(matches[0])

    if dm.ndim != 2:
        raise ValueError(f"Expected a 2-D legacy DM array, got shape {dm.shape}.")
    if dm.shape[0] == redshifts.size:
        result = dm[index, :]
    elif dm.shape[1] == redshifts.size:
        result = dm[:, index]
    else:
        raise ValueError(
            f"Cannot identify redshift axis: DM shape={dm.shape}, Nz={redshifts.size}."
        )
    return np.asarray(result, dtype=float)


def _distribution_statistics(values: np.ndarray, edges: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[
        np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    ]
    if values.size == 0:
        return {
            "count": 0,
            "mean_dm": np.nan,
            "median_dm": np.nan,
            "std_dm": np.nan,
            "p2p5_dm": np.nan,
            "p97p5_dm": np.nan,
        }
    return {
        "count": int(values.size),
        "mean_dm": float(np.mean(values)),
        "median_dm": float(np.median(values)),
        "std_dm": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
        "p2p5_dm": float(np.percentile(values, 2.5)),
        "p97p5_dm": float(np.percentile(values, 97.5)),
    }


def compare_all_halo_dm(
    reference_dm: np.ndarray,
    catalog_slice: CatalogSlice,
    *,
    output_path: str | Path | None = None,
    edges: np.ndarray = DM_EDGES,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Compare the locally available Ralf/TNG halo PDF and HalfDome slice."""

    ref_counts, ref_density = histogram_density(reference_dm, edges)
    user_counts, user_density = histogram_density(catalog_slice.dm_pc_cm3, edges)
    centers = np.sqrt(edges[:-1] * edges[1:])

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ref_keep = ref_density > 0.0
    user_keep = user_density > 0.0
    ax.plot(
        centers[ref_keep],
        ref_density[ref_keep],
        color="mediumblue",
        linewidth=2.5,
        label=f"Local Ralf/Konietzka TNG halo-only, z=1 (N={ref_counts.sum():,})",
    )
    z_min, z_max = catalog_slice.z_bounds
    ax.plot(
        centers[user_keep],
        user_density[user_keep],
        color="black",
        linewidth=2.5,
        label=(
            f"HalfDome catalogue, {z_min:.2f}<=z<{z_max:.2f} "
            f"(N={user_counts.sum():,})"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0, 5.0e3)
    ax.set_ylabel(r"$p({\rm DM}\mid z)$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)

    valid_ratio = ref_density > 0.0
    percent = np.full_like(ref_density, np.nan)
    percent[valid_ratio] = (
        100.0
        * (user_density[valid_ratio] - ref_density[valid_ratio])
        / ref_density[valid_ratio]
    )
    ratio_ax.semilogx(centers[valid_ratio], percent[valid_ratio], color="black")
    ratio_ax.axhline(0.0, color="0.4", linewidth=1.0)
    ratio_ax.set_xlabel(r"DM [pc cm$^{-3}$]")
    ratio_ax.set_ylabel("PDF diff. [%]")
    ratio_ax.grid(alpha=0.25, which="both")
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")

    ref_stats = _distribution_statistics(reference_dm, edges)
    user_stats = _distribution_statistics(catalog_slice.dm_pc_cm3, edges)
    summary_rows = []
    for statistic in ["mean_dm", "median_dm", "std_dm", "p2p5_dm", "p97p5_dm"]:
        ref_value = float(ref_stats[statistic])
        user_value = float(user_stats[statistic])
        summary_rows.append(
            {
                "statistic": statistic,
                "illustristng_local_ralf_array": ref_value,
                "halfdome": user_value,
                "percent_difference_vs_illustristng_local_ralf_array": (
                    100.0 * (user_value - ref_value) / ref_value
                    if np.isfinite(ref_value) and ref_value != 0.0
                    else np.nan
                ),
            }
        )

    widths = np.diff(edges)
    total_variation = 0.5 * np.sum(np.abs(user_density - ref_density) * widths)
    summary = pd.DataFrame(summary_rows)
    summary["total_variation_distance"] = total_variation
    summary.attrs["illustristng_count"] = int(ref_stats["count"])
    summary.attrs["halfdome_count"] = int(user_stats["count"])

    bin_table = pd.DataFrame(
        {
            "dm_bin_left": edges[:-1],
            "dm_bin_right": edges[1:],
            "dm_bin_center": centers,
            "illustristng_local_ralf_count": ref_counts,
            "halfdome_count": user_counts,
            "illustristng_local_ralf_density": ref_density,
            "halfdome_density": user_density,
            "pdf_percent_difference_vs_illustristng_local_ralf_array": percent,
        }
    )
    return fig, summary, bin_table


def _cosmos2020_row(z: float) -> np.ndarray:
    if z < COSMOS2020_PARAMS[0, 0]:
        return COSMOS2020_PARAMS[0].copy()
    for row in COSMOS2020_PARAMS:
        if row[0] <= z < row[1]:
            return row.copy()
    if np.isclose(z, COSMOS2020_PARAMS[-1, 1]):
        return COSMOS2020_PARAMS[-1].copy()
    raise ValueError("COSMOS2020 SHMR is tabulated only through z=5.5.")


def cosmos2020_log_mhalo_from_log_mstar(
    log_mstar: np.ndarray,
    *,
    z: float,
) -> np.ndarray:
    """Evaluate the COSMOS2020 inverse central SHMR in physical solar masses."""

    row = _cosmos2020_row(float(z))
    _, _, log_m1, log_mstar0, beta, delta, gamma = row
    log_m1 += np.log10(H0_COSMOS2020 / H0_HALFDOME)
    log_mstar0 += 2.0 * np.log10(H0_COSMOS2020 / H0_HALFDOME)
    x = 10.0 ** (np.asarray(log_mstar, dtype=float) - log_mstar0)
    return log_m1 + beta * np.log10(x) + x**delta / (1.0 + x ** (-gamma)) - 0.5


def cosmos2020_mstar_from_mvir(mvir_msun: np.ndarray, *, z: float) -> np.ndarray:
    """Numerically invert the COSMOS2020 relation on a dense monotonic grid."""

    mvir = np.asarray(mvir_msun, dtype=float)
    if np.any(~np.isfinite(mvir)) or np.any(mvir <= 0.0):
        raise ValueError("All virial halo masses must be finite and positive.")

    log_mstar_grid = np.linspace(3.0, 14.0, 100_000)
    log_mvir_grid = cosmos2020_log_mhalo_from_log_mstar(log_mstar_grid, z=z)
    order = np.argsort(log_mvir_grid)
    log_mvir_sorted = log_mvir_grid[order]
    log_mstar_sorted = log_mstar_grid[order]
    log_target = np.log10(mvir)
    if log_target.min() < log_mvir_sorted[0] or log_target.max() > log_mvir_sorted[-1]:
        raise ValueError("Requested halo mass is outside the COSMOS2020 inversion grid.")
    return 10.0 ** np.interp(log_target, log_mvir_sorted, log_mstar_sorted)


def moster2010_mstar_from_mvir(
    mvir_msun: np.ndarray,
    *,
    scatter_fit: bool = False,
) -> np.ndarray:
    """Scatter-free Moster et al. (2010) central SHMR used in the comparison.

    ``scatter_fit=True`` selects the paper's parameters obtained when fitting a
    model with 0.15-dex scatter; it does not draw random scatter realizations.
    """

    mvir = np.asarray(mvir_msun, dtype=float)
    if np.any(~np.isfinite(mvir)) or np.any(mvir <= 0.0):
        raise ValueError("All virial halo masses must be finite and positive.")
    if scatter_fit:
        log_m1, norm, beta, gamma = 11.899, 0.02817, 1.068, 0.611
    else:
        log_m1, norm, beta, gamma = 11.884, 0.02820, 1.057, 0.556
    m1 = 10.0**log_m1
    ratio = mvir / m1
    return 2.0 * norm * mvir / (ratio ** (-beta) + ratio**gamma)



def moster2013_mstar_from_mvir(
    mvir_msun: np.ndarray,
    *,
    z: float | np.ndarray,
) -> np.ndarray:
    """Evaluate the repository's previous/default Moster et al. (2013) SHMR.

    The HalfDome catalogue-generation code historically evaluates this relation
    at the supplied physical halo mass. Here both mappings receive the same
    numerical virial halo masses, so this is a mapping-only comparison rather
    than a halo-mass-function comparison.
    """

    mvir, redshift = np.broadcast_arrays(
        np.asarray(mvir_msun, dtype=float), np.asarray(z, dtype=float)
    )
    if np.any(~np.isfinite(mvir)) or np.any(mvir <= 0.0):
        raise ValueError("All virial halo masses must be finite and positive.")
    if np.any(~np.isfinite(redshift)) or np.any(redshift <= -1.0):
        raise ValueError("z must be finite and greater than -1.")

    z_fraction = redshift / (1.0 + redshift)
    log_m1 = 11.590 + 1.195 * z_fraction
    norm = 0.0351 - 0.0247 * z_fraction
    beta = 1.376 - 0.826 * z_fraction
    gamma = 0.608 + 0.329 * z_fraction
    m1 = 10.0**log_m1
    ratio = mvir / m1
    return 2.0 * norm * mvir / (ratio ** (-beta) + ratio**gamma)

def m200c_to_mvir_nfw(
    m200c_msun: np.ndarray,
    redshift: np.ndarray,
) -> np.ndarray:
    """Convert physical M200c to Bryan-Norman Mvir as in catalogue generation."""

    m200c, z = np.broadcast_arrays(
        np.asarray(m200c_msun, dtype=float), np.asarray(redshift, dtype=float)
    )
    if (
        np.any(~np.isfinite(m200c))
        or np.any(m200c <= 0.0)
        or np.any(~np.isfinite(z))
        or np.any(z <= -1.0)
    ):
        raise ValueError("M200c and redshift inputs must be finite and physical.")

    omega_m_z = OMEGA_M_HALFDOME * (1.0 + z) ** 3 / (
        OMEGA_M_HALFDOME * (1.0 + z) ** 3 + 1.0 - OMEGA_M_HALFDOME
    )
    x_bn = omega_m_z - 1.0
    delta_vir_critical = 18.0 * np.pi**2 + 82.0 * x_bn - 39.0 * x_bn**2
    concentration = 5.71 * (m200c * (H0_HALFDOME / 100.0) / 2.0e12) ** (
        -0.084
    ) * (1.0 + z) ** (-0.47)

    def nfw_mass_fraction(value: np.ndarray) -> np.ndarray:
        return np.log1p(value) - value / (1.0 + value)

    f_c = nfw_mass_fraction(concentration)
    target = delta_vir_critical / 200.0
    lo = np.full(m200c.shape, 0.05, dtype=float)
    hi = np.full(m200c.shape, 20.0, dtype=float)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        residual = (
            nfw_mass_fraction(concentration * mid) / f_c / mid**3 - target
        )
        lo = np.where(residual > 0.0, mid, lo)
        hi = np.where(residual > 0.0, hi, mid)
    radius_ratio = 0.5 * (lo + hi)
    mass_ratio = nfw_mass_fraction(concentration * radius_ratio) / f_c
    return m200c * mass_ratio


def catalog_row_abundance_matching_comparison(
    catalog_slice: CatalogSlice,
) -> pd.DataFrame:
    """Compare stored catalogue Mstar with SHMRs at each row's converted Mvir."""

    mvir = m200c_to_mvir_nfw(
        catalog_slice.halo_mass_msun, catalog_slice.redshift
    )
    stored = catalog_slice.stellar_mass_msun
    cosmos = cosmos2020_mstar_from_mvir(mvir, z=catalog_slice.z_target)
    moster13 = moster2013_mstar_from_mvir(mvir, z=catalog_slice.redshift)
    moster10 = moster2010_mstar_from_mvir(mvir, scatter_fit=False)
    return pd.DataFrame(
        {
            "host_redshift": catalog_slice.redshift,
            "m200c_msun": catalog_slice.halo_mass_msun,
            "mvir_msun": mvir,
            "log10_mvir_msun": np.log10(mvir),
            "mstar_catalog_msun": stored,
            "mstar_cosmos2020_formula_msun": cosmos,
            "mstar_moster2013_msun": moster13,
            "mstar_moster2010_scatter_free_msun": moster10,
            "catalog_vs_cosmos2020_formula_percent": 100.0
            * (stored - cosmos)
            / cosmos,
            "catalog_vs_moster2013_percent": 100.0
            * (stored - moster13)
            / moster13,
            "catalog_vs_moster2010_scatter_free_percent": 100.0
            * (stored - moster10)
            / moster10,
        }
    )


def summarize_catalog_abundance_rows(
    rows: pd.DataFrame,
    *,
    log_mass_edges: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scalar and fixed-Mvir-bin percentage summaries for catalogue rows."""

    percent_columns = OrderedDict(
        [
            (
                "COSMOS2020 stored-vs-formula self-check",
                "catalog_vs_cosmos2020_formula_percent",
            ),
            ("catalogue vs Moster+13", "catalog_vs_moster2013_percent"),
            (
                "catalogue vs Moster+10 scatter-free",
                "catalog_vs_moster2010_scatter_free_percent",
            ),
        ]
    )
    summary_rows = []
    for label, column in percent_columns.items():
        values = rows[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        summary_rows.append(
            {
                "comparison": label,
                "count": values.size,
                "mean_percent": float(np.mean(values)),
                "median_percent": float(np.median(values)),
                "p16_percent": float(np.percentile(values, 16.0)),
                "p84_percent": float(np.percentile(values, 84.0)),
                "p2p5_percent": float(np.percentile(values, 2.5)),
                "p97p5_percent": float(np.percentile(values, 97.5)),
            }
        )

    if log_mass_edges is None:
        log_mass_edges = np.arange(12.5, 15.5001, 0.1)
    log_mass_edges = np.asarray(log_mass_edges, dtype=float)
    log_mass = rows["log10_mvir_msun"].to_numpy(dtype=float)
    bin_index = np.digitize(log_mass, log_mass_edges) - 1
    binned_rows = []
    for index in range(len(log_mass_edges) - 1):
        keep = bin_index == index
        if not np.any(keep):
            continue
        row: dict[str, float | int] = {
            "log10_mvir_left": float(log_mass_edges[index]),
            "log10_mvir_right": float(log_mass_edges[index + 1]),
            "count": int(np.sum(keep)),
            "median_log10_mvir": float(np.median(log_mass[keep])),
        }
        for _, column in percent_columns.items():
            values = rows.loc[keep, column].to_numpy(dtype=float)
            row[f"{column}_median"] = float(np.median(values))
            row[f"{column}_p16"] = float(np.percentile(values, 16.0))
            row[f"{column}_p84"] = float(np.percentile(values, 84.0))
        binned_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(binned_rows)


def plot_catalog_row_abundance_comparison(
    binned: pd.DataFrame,
    catalog_slice: CatalogSlice,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot row-level catalogue percentage differences in common Mvir bins."""

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = 10.0 ** binned["median_log10_mvir"].to_numpy(dtype=float)
    series = [
        (
            "catalog_vs_moster2013_percent",
            "Moster+13 previous default",
            "tab:blue",
        ),
        (
            "catalog_vs_moster2010_scatter_free_percent",
            "Moster+10 sensitivity case",
            "tab:orange",
        ),
        (
            "catalog_vs_cosmos2020_formula_percent",
            "COSMOS2020 stored-vs-formula check",
            "0.35",
        ),
    ]
    for column, label, color in series:
        median = binned[f"{column}_median"].to_numpy(dtype=float)
        lower = binned[f"{column}_p16"].to_numpy(dtype=float)
        upper = binned[f"{column}_p84"].to_numpy(dtype=float)
        ax.semilogx(x, median, linewidth=2.0, color=color, label=label)
        ax.fill_between(x, lower, upper, color=color, alpha=0.15)
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    z_min, z_max = catalog_slice.z_bounds
    ax.set_title(
        "Actual HalfDome catalogue rows at common converted virial mass\n"
        f"{z_min:.2f} <= host redshift < {z_max:.2f}"
    )
    ax.set_xlabel(r"Converted virial halo mass $M_{\rm vir}$ [$M_\odot$]")
    ax.set_ylabel(
        r"$(M_{\star,\rm catalog}-M_{\star,\rm relation})/"
        r"M_{\star,\rm relation}$ [%]"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def abundance_matching_comparison(
    *,
    z: float = 1.0,
    log_mvir_min: float = 12.5,
    log_mvir_max: float = 15.5,
    sample_count: int = 500,
) -> pd.DataFrame:
    """Compare stellar masses assigned at the same virial halo masses."""

    log_mvir = np.linspace(log_mvir_min, log_mvir_max, sample_count)
    mvir = 10.0**log_mvir
    cosmos = cosmos2020_mstar_from_mvir(mvir, z=z)
    moster2013 = moster2013_mstar_from_mvir(mvir, z=z)
    moster = moster2010_mstar_from_mvir(mvir, scatter_fit=False)
    moster_scatter_fit = moster2010_mstar_from_mvir(mvir, scatter_fit=True)
    return pd.DataFrame(
        {
            "redshift": z,
            "log10_mvir_msun": log_mvir,
            "mvir_msun": mvir,
            "mstar_cosmos2020_msun": cosmos,
            "mstar_moster2013_msun": moster2013,
            "mstar_moster2010_scatter_free_msun": moster,
            "mstar_moster2010_0p15dex_fit_msun": moster_scatter_fit,
            "cosmos2020_vs_moster2013_percent": 100.0
            * (cosmos - moster2013)
            / moster2013,
            "cosmos2020_vs_moster_scatter_free_percent": 100.0
            * (cosmos - moster)
            / moster,
            "cosmos2020_vs_moster_0p15dex_fit_percent": 100.0
            * (cosmos - moster_scatter_fit)
            / moster_scatter_fit,
        }
    )


def plot_abundance_matching_comparison(
    comparison: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot SHMR values and the requested percentage difference at fixed mass."""

    z = float(comparison["redshift"].iloc[0])
    mvir = comparison["mvir_msun"].to_numpy()
    fig, (ax, diff_ax) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )
    ax.loglog(
        mvir,
        comparison["mstar_cosmos2020_msun"],
        linewidth=2.3,
        label="COSMOS2020 / Shuntov+22",
    )
    ax.loglog(
        mvir,
        comparison["mstar_moster2013_msun"],
        linewidth=2.0,
        label="Moster+13 (repository's previous default)",
    )
    ax.loglog(
        mvir,
        comparison["mstar_moster2010_scatter_free_msun"],
        linewidth=1.7,
        linestyle=":",
        label="Moster+10, scatter-free sensitivity case",
    )
    ax.loglog(
        mvir,
        comparison["mstar_moster2010_0p15dex_fit_msun"],
        linewidth=1.7,
        linestyle="--",
        label="Moster+10, 0.15-dex fit parameters",
    )
    ax.set_ylabel(r"Abundance-matched $M_\star$ [$M_\odot$]")
    ax.set_title(rf"Same-$M_{{\rm vir}}$ abundance-matching comparison at $z={z:g}$")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)

    diff_ax.semilogx(
        mvir,
        comparison["cosmos2020_vs_moster2013_percent"],
        linewidth=2.0,
        label="vs Moster+13 previous default",
    )
    diff_ax.semilogx(
        mvir,
        comparison["cosmos2020_vs_moster_scatter_free_percent"],
        linewidth=1.7,
        linestyle=":",
        label="vs Moster+10 scatter-free",
    )
    diff_ax.semilogx(
        mvir,
        comparison["cosmos2020_vs_moster_0p15dex_fit_percent"],
        linewidth=1.7,
        linestyle="--",
        label="vs 0.15-dex fit params",
    )
    diff_ax.axhline(0.0, color="0.4", linewidth=1.0)
    diff_ax.set_xlabel(r"Virial halo mass $M_{\rm vir}$ [$M_\odot$]")
    diff_ax.set_ylabel(
        r"$(M_{\star,{\rm C20}}-M_{\star,{\rm ref}})/"
        r"M_{\star,{\rm ref}}$ [%]"
    )
    diff_ax.grid(alpha=0.25, which="both")
    diff_ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def abundance_matching_anchor_table(
    comparison: pd.DataFrame,
    log_mvir_values: Iterable[float] = (12.5, 13.0, 14.0, 15.0),
) -> pd.DataFrame:
    log_grid = comparison["log10_mvir_msun"].to_numpy()
    numeric_columns = [
        "mstar_cosmos2020_msun",
        "mstar_moster2013_msun",
        "mstar_moster2010_scatter_free_msun",
        "mstar_moster2010_0p15dex_fit_msun",
        "cosmos2020_vs_moster2013_percent",
        "cosmos2020_vs_moster_scatter_free_percent",
        "cosmos2020_vs_moster_0p15dex_fit_percent",
    ]
    rows = []
    for value in log_mvir_values:
        row: dict[str, float] = {
            "redshift": float(comparison["redshift"].iloc[0]),
            "log10_mvir_msun": float(value),
            "mvir_msun": float(10.0**value),
        }
        for column in numeric_columns:
            row[column] = float(
                np.interp(value, log_grid, comparison[column].to_numpy())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_public_v2_z1(
    path: str | Path = DEFAULT_PUBLIC_V2_Z1,
    *,
    verify_md5: bool = False,
) -> dict[str, object]:
    """Validate the installed public catalogue without treating it as legacy data."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Public v2 z=1 catalogue not found: {path}")
    size = path.stat().st_size
    if size != PUBLIC_V2_Z1_BYTES:
        raise ValueError(f"Unexpected file size {size}; expected {PUBLIC_V2_Z1_BYTES}.")

    md5 = None
    if verify_md5:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        md5 = digest.hexdigest()
        if md5 != PUBLIC_V2_Z1_MD5:
            raise ValueError(f"MD5 mismatch: {md5} != {PUBLIC_V2_Z1_MD5}.")

    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        if keys != ["DMvalues", "redshifts"]:
            raise ValueError(f"Unexpected public-v2 keys: {keys}")
        dm_shape = tuple(handle["DMvalues"].shape)
        redshift_shape = tuple(handle["redshifts"].shape)
        version = str(handle.attrs.get("version", ""))

    return {
        "path": str(path),
        "bytes": size,
        "md5": md5,
        "version": version,
        "keys": keys,
        "dm_shape": dm_shape,
        "redshift_shape": redshift_shape,
        "is_legacy_mass_split_replacement": False,
    }


def run_analysis(
    *,
    catalog_path: str | Path = DEFAULT_COSMOS2020_CATALOG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    z_target: float = 1.0,
    z_half_width: float = 0.05,
    chunksize: int = 500_000,
    max_rows: int | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_slice = load_catalog_redshift_slice(
        catalog_path,
        z_target=z_target,
        z_half_width=z_half_width,
        chunksize=chunksize,
        max_rows=max_rows,
    )

    upper_fig, _, upper_summary = plot_catalog_curves(
        catalog_slice,
        UPPER_LIMIT_WINDOWS,
        UPPER_LIMIT_COLORS,
        xlim=(1.0, 5.0e3),
        output_path=output_dir / "halfdome_host_mass_upper_limit_pdfs.png",
    )
    plt.close(upper_fig)
    lower_fig, _, lower_summary = plot_catalog_curves(
        catalog_slice,
        LOWER_LIMIT_WINDOWS,
        LOWER_LIMIT_COLORS,
        xlim=(1.0, 1.0e4),
        output_path=output_dir / "halfdome_host_mass_lower_limit_pdfs.png",
    )
    plt.close(lower_fig)

    upper_summary.to_csv(
        output_dir / "halfdome_host_mass_upper_limit_summary.csv", index=False
    )
    lower_summary.to_csv(
        output_dir / "halfdome_host_mass_lower_limit_summary.csv", index=False
    )

    legacy_dm = extract_legacy_dm_at_redshift(z_target=z_target)
    compare_fig, dm_summary, dm_bins = compare_all_halo_dm(
        legacy_dm,
        catalog_slice,
        output_path=output_dir / "illustristng_vs_halfdome_all_halo_dm.png",
    )
    plt.close(compare_fig)
    dm_summary.to_csv(output_dir / "all_halo_dm_percent_differences.csv", index=False)
    dm_bins.to_csv(output_dir / "all_halo_dm_pdf_bin_differences.csv", index=False)

    abundance = abundance_matching_comparison(z=z_target)
    abundance_fig = plot_abundance_matching_comparison(
        abundance,
        output_path=output_dir / "same_mass_abundance_matching_percent_difference.png",
    )
    plt.close(abundance_fig)
    abundance.to_csv(
        output_dir / "same_mass_abundance_matching_percent_difference.csv",
        index=False,
    )
    anchors = abundance_matching_anchor_table(abundance)
    anchors.to_csv(
        output_dir / "same_mass_abundance_matching_anchor_masses.csv", index=False
    )

    catalog_abundance_rows = catalog_row_abundance_matching_comparison(catalog_slice)
    catalog_abundance_summary, catalog_abundance_binned = (
        summarize_catalog_abundance_rows(catalog_abundance_rows)
    )
    catalog_abundance_fig = plot_catalog_row_abundance_comparison(
        catalog_abundance_binned,
        catalog_slice,
        output_path=(
            output_dir
            / "catalog_rows_same_mass_abundance_matching_percent_difference.png"
        ),
    )
    plt.close(catalog_abundance_fig)
    catalog_abundance_summary.to_csv(
        output_dir / "catalog_rows_abundance_matching_summary.csv", index=False
    )
    catalog_abundance_binned.to_csv(
        output_dir / "catalog_rows_abundance_matching_by_mass.csv", index=False
    )

    return {
        "catalog_slice": catalog_slice,
        "upper_summary": upper_summary,
        "lower_summary": lower_summary,
        "dm_summary": dm_summary,
        "abundance_matching": abundance,
        "abundance_anchors": anchors,
        "catalog_abundance_summary": catalog_abundance_summary,
        "catalog_abundance_binned": catalog_abundance_binned,
        "output_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create HalfDome FRB-DM and same-mass SHMR comparisons."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_COSMOS2020_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--z-target", type=float, default=1.0)
    parser.add_argument("--z-half-width", type=float, default=0.05)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional input-row limit for a quick smoke test.",
    )
    parser.add_argument(
        "--validate-public-v2",
        action="store_true",
        help="Also verify the installed public z=1 HDF5 and its MD5.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.validate_public_v2:
        print(validate_public_v2_z1(verify_md5=True))
    result = run_analysis(
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        z_target=args.z_target,
        z_half_width=args.z_half_width,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )
    catalog_slice: CatalogSlice = result["catalog_slice"]  # type: ignore[assignment]
    print(
        f"Selected {catalog_slice.size:,} catalogue rows from "
        f"{catalog_slice.rows_scanned:,} scanned rows."
    )
    print(f"Outputs: {result['output_dir']}")
    print("\nSame-Mvir abundance-matching anchor table:")
    print(result["abundance_anchors"].to_string(index=False))  # type: ignore[union-attr]
    print(
        "\nCaveat: the two catalogue mass-window plots cut host mass; the legacy "
        "notebook cuts the masses of foreground halos contributing to each DM."
    )


if __name__ == "__main__":
    main()
