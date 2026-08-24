#!/usr/bin/env python3
"""Direct IllustrisTNG/HalfDome histogram comparisons used by the notebook."""

from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


H5_NAME = "halfdome_uniform_fixedz_foreground_mass_histograms.h5"
SUMMARY_NAME = "halfdome_uniform_fixedz_foreground_mass_histograms_summary.csv"
SOURCE_REDSHIFTS = (1.0, 2.0, 3.0, 4.0)
DEFAULT_WINDOW = "m1e10_to_1e16"  # Full total over every resolved HalfDome halo.
EXPECTED_RAYS = 120_000
MIN_PERCENT_BIN_COUNT = 10
FIXED_200C_REDSHIFT = 1.0
FIXED_200C_NSIDE = 4096
FIXED_200C_APERTURE = 3.0

TNG_FILES = {
    "all": "DMhalo_r200_all.npy",
    "m1e10_to_1e12": "DMhalo_r200_mass10to12.npy",
    "m1e10_to_1e13": "DMhalo_r200_mass10to13.npy",
    "m1e10_to_1e14": "DMhalo_r200_mass10to14.npy",
    "m1e10_to_1e15": "DMhalo_r200_mass10to15.npy",
    "m1e11_to_1e14": "DMhalo_r200_mass11to14.npy",
    "m1e12_to_1e14": "DMhalo_r200_mass12to14.npy",
    "m1e13_to_1e14": "DMhalo_r200_mass13to14.npy",
}


def _tag(value: float) -> str:
    # Cluster run tags always retain one decimal place (1.0 -> 1p0).
    return f"{float(value):.1f}".replace(".", "p")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class DirectComparison:
    """Load validated histogram products and make the five requested plots."""

    def __init__(self, project_root: str | Path | None = None):
        candidate = Path(project_root) if project_root is not None else Path("/home/cbllover/HalfDome")
        if not (candidate / "frb_catalog_comparison_outputs").exists():
            candidate = Path.cwd().resolve()
        self.project_root = candidate
        self.hd_base = (
            candidate
            / "frb_catalog_comparison_outputs"
            / "halfdome_uniform_fixedz_foreground_mass_histograms"
        )
        self.hd_200c_base = (
            candidate
            / "frb_catalog_comparison_outputs"
            / "halfdome_m200c_bins_m200m_profiles_r200c_z1_120k_foreground_mass_histograms"
        )
        self.tng_base = (
            candidate
            / "frb_catalog_comparison_outputs"
            / "haloDM-20260819T104507Z-1-001"
            / "haloDM"
        )
        self.output_dir = candidate / "tng_halfdome_direct_comparison" / "cluster_HalfDome_pdfs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tng_redshifts = np.load(self.tng_base / "redshifts_DMhalo.npy")
        self.tng_arrays = {
            label: np.load(self.tng_base / filename, mmap_mode="r")
            for label, filename in TNG_FILES.items()
        }

    def run_dir(
        self,
        source_redshift: float,
        nside: int,
        aperture_r200: float,
        *,
        validated_200c: bool = False,
    ) -> Path:
        name = (
            f"zsrc{_tag(source_redshift)}_nside{int(nside)}_nrays{EXPECTED_RAYS}"
            "_allhalos"
        )
        if not np.isclose(aperture_r200, 1.0):
            radius_name = "r200c" if validated_200c else "r200"
            name += f"_{radius_name}x{_tag(aperture_r200)}"
        if validated_200c:
            name += "_m200mprofile"
        base = self.hd_200c_base if validated_200c else self.hd_base
        return base / f"{name}_seed42"

    def load_halfdome(
        self,
        source_redshift: float,
        nside: int = 8192,
        aperture_r200: float = 3.0,
        window_label: str = DEFAULT_WINDOW,
        *,
        validated_200c: bool = False,
    ) -> dict:
        run_dir = self.run_dir(
            source_redshift, nside, aperture_r200, validated_200c=validated_200c
        )
        path = run_dir / H5_NAME
        if not path.exists():
            raise FileNotFoundError(f"Missing completed HalfDome run: {path}")

        with h5py.File(path, "r") as h5:
            labels = [_decode(value) for value in h5["window_label"][:]]
            if window_label not in labels:
                raise KeyError(f"{window_label!r} unavailable in {path}; labels={labels}")
            index = labels.index(window_label)
            edges = np.asarray(h5["pdf_bin_edges_pc_cm3"][:], dtype=float)
            centers = np.asarray(h5["pdf_bin_centers_pc_cm3"][:], dtype=float)
            density = np.asarray(h5["pdf_density_per_pc_cm3"][:], dtype=float)
            counts = np.asarray(h5["pdf_count"][:], dtype=np.int64)
            if density.shape[0] != len(labels):
                density = density.T
                counts = counts.T

            actual_z = float(h5.attrs["provenance_source_redshift"])
            actual_nside = int(h5.attrs["provenance_nside"])
            actual_rays = int(h5.attrs["n_rays"])
            actual_aperture = float(
                h5.attrs.get(
                    "provenance_halo_extension_r200_multiplier",
                    h5.attrs["provenance_dm_aperture_r200_multiplier"],
                )
            )
            mass_definition = _decode(h5.attrs.get("provenance_halo_mass_definition", ""))
            radius_definition = _decode(h5.attrs.get("provenance_halo_radius_definition", ""))
            reference_density = _decode(h5.attrs.get("provenance_halo_reference_density", ""))
            overdensity = int(h5.attrs.get("provenance_halo_overdensity", -1))
            selection_definition = _decode(
                h5.attrs.get("provenance_mass_window_selection_definition", "")
            )
            xgpaint_mass_definition = _decode(
                h5.attrs.get("provenance_xgpaint_input_mass_definition", "")
            )
            xgpaint_radius_definition = _decode(
                h5.attrs.get("provenance_xgpaint_aperture_radius_definition", "")
            )
            xgpaint_profile_mass_definition = _decode(
                h5.attrs.get("provenance_xgpaint_profile_input_mass_definition", "")
            )
            xgpaint_profile_mass_dataset = _decode(
                h5.attrs.get("provenance_xgpaint_profile_input_mass_dataset", "")
            )
            aperture_mass_definition = _decode(
                h5.attrs.get("provenance_aperture_mass_definition", "")
            )
            aperture_mass_dataset = _decode(
                h5.attrs.get("provenance_aperture_mass_dataset", "")
            )
            aperture_geometry_owner = _decode(
                h5.attrs.get("provenance_aperture_geometry_owner", "")
            )
            xgpaint_internal_radius_definition = _decode(
                h5.attrs.get("provenance_xgpaint_profile_internal_radius_definition", "")
            )
            xgpaint_default_theta_max_used = bool(
                h5.attrs.get("provenance_xgpaint_default_theta_max_used", True)
            )
            xgpaint_paint_function_used = bool(
                h5.attrs.get("provenance_xgpaint_paint_function_used", True)
            )
            requested_min = float(h5["window_requested_min_msun"][index])
            requested_max = float(h5["window_requested_max_msun"][index])
            effective_min = float(h5["window_effective_min_msun"][index])
            effective_max = float(h5["window_effective_max_msun"][index])

        expected = (float(source_redshift), int(nside), EXPECTED_RAYS, float(aperture_r200))
        actual = (actual_z, actual_nside, actual_rays, actual_aperture)
        if not (
            np.isclose(actual_z, source_redshift)
            and actual_nside == int(nside)
            and actual_rays == EXPECTED_RAYS
            and np.isclose(actual_aperture, aperture_r200)
        ):
            raise ValueError(f"Provenance mismatch for {path}: actual={actual}, expected={expected}")
        if validated_200c:
            actual_200c = (
                mass_definition,
                radius_definition,
                reference_density,
                overdensity,
                selection_definition,
                xgpaint_mass_definition,
                xgpaint_radius_definition,
                xgpaint_profile_mass_definition,
                xgpaint_profile_mass_dataset,
                aperture_mass_definition,
                aperture_mass_dataset,
                aperture_geometry_owner,
                xgpaint_internal_radius_definition,
                xgpaint_default_theta_max_used,
                xgpaint_paint_function_used,
            )
            expected_200c = (
                "M200c", "R200c", "critical", 200,
                "M200c", "M200m", "R200c",
                "M200m", "halo_mass_m200m",
                "M200c", "halo_mass_m200c", "generator", "R200m",
                False, False,
            )
            if actual_200c != expected_200c:
                raise ValueError(
                    f"HalfDome file is not a validated M200c/R200c product: "
                    f"actual={actual_200c}, expected={expected_200c}, path={path}"
                )

        zero_fraction = 1.0 - float(counts[index].sum()) / actual_rays
        summary_path = run_dir / SUMMARY_NAME
        if summary_path.exists():
            with summary_path.open(newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["label"] == window_label]
            if len(rows) == 1:
                zero_fraction = float(rows[0]["zero_fraction"])

        return {
            "run_dir": run_dir,
            "edges": edges,
            "centers": centers,
            "pdf": density[index],
            "counts": counts[index],
            "zero_fraction": zero_fraction,
            "source_redshift": actual_z,
            "nside": actual_nside,
            "aperture_r200": actual_aperture,
            "mass_definition": mass_definition,
            "radius_definition": radius_definition,
            "reference_density": reference_density,
            "overdensity": overdensity,
            "requested_min_msun": requested_min,
            "requested_max_msun": requested_max,
            "effective_min_msun": effective_min,
            "effective_max_msun": effective_max,
        }

    def validate_inputs(self) -> None:
        """Check all 16 complete-catalogue runs used by the diagnostics."""
        for redshift in SOURCE_REDSHIFTS:
            for nside in (2048, 8192):
                for aperture in (1.0, 3.0):
                    self.load_halfdome(redshift, nside, aperture, DEFAULT_WINDOW)

    def validate_fixed_200c_input(self) -> None:
        """Validate only the requested z=1, NSIDE=4096, 3R200c product."""
        self.load_halfdome(
            FIXED_200C_REDSHIFT,
            FIXED_200C_NSIDE,
            FIXED_200C_APERTURE,
            DEFAULT_WINDOW,
            validated_200c=True,
        )

    def tng_values(self, label: str, source_redshift: float) -> np.ndarray:
        array = self.tng_arrays[label]
        matches = np.flatnonzero(np.isclose(self.tng_redshifts, source_redshift))
        if len(matches) != 1:
            raise ValueError(f"TNG redshift {source_redshift} has {len(matches)} matches")
        index = int(matches[0])
        if array.shape[0] == len(self.tng_redshifts):
            return np.asarray(array[index], dtype=float)
        if array.shape[1] == len(self.tng_redshifts):
            return np.asarray(array[:, index], dtype=float)
        raise ValueError(f"TNG array {label} has incompatible shape {array.shape}")

    @staticmethod
    def histogram_from_values(
        values: np.ndarray, edges: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        counts, _ = np.histogram(values, bins=edges)
        density = np.zeros(len(edges) - 1, dtype=float)
        if counts.sum() > 0:
            density = counts / (counts.sum() * np.diff(edges))
        zero_fraction = float(np.count_nonzero(values == 0.0)) / len(values)
        return density, counts, zero_fraction

    @staticmethod
    def percent_difference(
        numerator_pdf: np.ndarray,
        denominator_pdf: np.ndarray,
        numerator_counts: np.ndarray,
        denominator_counts: np.ndarray,
    ) -> np.ndarray:
        """Return 100*(numerator-denominator)/denominator in supported bins."""
        result = np.full_like(denominator_pdf, np.nan, dtype=float)
        supported = (
            np.isfinite(numerator_pdf)
            & np.isfinite(denominator_pdf)
            & (denominator_pdf > 0.0)
            & (numerator_counts >= MIN_PERCENT_BIN_COUNT)
            & (denominator_counts >= MIN_PERCENT_BIN_COUNT)
        )
        result[supported] = (
            100.0
            * (numerator_pdf[supported] - denominator_pdf[supported])
            / denominator_pdf[supported]
        )
        return result

    @staticmethod
    def draw_pdf(ax, centers: np.ndarray, density: np.ndarray, **kwargs) -> None:
        keep = np.isfinite(density) & (density > 0.0)
        if np.any(keep):
            ax.plot(centers[keep], density[keep], **kwargs)

    @staticmethod
    def format_pdf_axis(
        ax, xlim=(0.1, 10_000.0), *, show_xlabel=True, show_ylabel=True
    ) -> None:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=10)
        ax.minorticks_on()
        ax.grid(alpha=0.20, which="both")
        if show_xlabel:
            ax.set_xlabel(r"DM [pc cm$^{-3}$]", fontsize=12)
        if show_ylabel:
            ax.set_ylabel(r"$p(\mathrm{DM}\mid z,\,\mathrm{DM}>0)$", fontsize=12)

    @staticmethod
    def format_percent_axis(
        ax, xlim=(0.1, 10_000.0), *, ylabel, show_xlabel=True, show_ylabel=True
    ) -> None:
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=25.0, linscale=0.8)
        ax.set_xlim(*xlim)
        ax.axhline(0.0, color="0.35", linewidth=1.0, zorder=1)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=9)
        ax.minorticks_on()
        ax.grid(alpha=0.18, which="both")
        if show_xlabel:
            ax.set_xlabel(r"DM [pc cm$^{-3}$]", fontsize=11)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=10)

    def _finish(self, fig, filename: str, show: bool) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.25, facecolor="white")
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved {path}")
        return path

    def _plot_mass_windows(
        self,
        entries,
        filename,
        title,
        xmax,
        show,
        *,
        nside=8192,
        validated_200c=False,
        radius_label="3R200",
    ) -> Path:
        reference = self.load_halfdome(
            1.0, nside, 3.0, DEFAULT_WINDOW, validated_200c=validated_200c
        )
        columns = len(entries)
        fig = plt.figure(figsize=(4.15 * columns, 8.2), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, columns, height_ratios=(3.0, 1.35), hspace=0.06)
        missing = []
        for column, (display, tng_label, hd_label, _color) in enumerate(entries):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            tng_pdf, tng_counts, _ = self.histogram_from_values(
                self.tng_values(tng_label, 1.0), reference["edges"]
            )
            self.draw_pdf(
                pdf_ax, reference["centers"], tng_pdf,
                color="#264653", linestyle="-", linewidth=2.5,
            )
            hd = self.load_halfdome(
                1.0, nside, 3.0, hd_label, validated_200c=validated_200c
            )
            if np.any(hd["pdf"] > 0.0):
                self.draw_pdf(
                    pdf_ax, hd["centers"], hd["pdf"],
                    color="#d1495b", linestyle="--", linewidth=2.5,
                )
                difference = self.percent_difference(
                    hd["pdf"], tng_pdf, hd["counts"], tng_counts
                )
                keep = np.isfinite(difference)
                percent_ax.plot(
                    hd["centers"][keep], difference[keep],
                    color="#6a4c93", linewidth=1.8,
                )
            else:
                missing.append(display)
                pdf_ax.text(
                    0.5, 0.12, "HalfDome unavailable\nat catalogue resolution",
                    transform=pdf_ax.transAxes, ha="center", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
                )
                percent_ax.text(
                    0.5, 0.5, "No matched\npercentage",
                    transform=percent_ax.transAxes, ha="center", va="center", fontsize=10,
                )

            self.format_pdf_axis(
                pdf_ax, (0.1, xmax), show_xlabel=False, show_ylabel=(column == 0)
            )
            self.format_percent_axis(
                percent_ax, (0.1, xmax),
                ylabel=r"$(p_{HD}-p_{TNG})/p_{TNG}$ [%]",
                show_ylabel=(column == 0),
            )
            pdf_ax.tick_params(labelbottom=False)
            pdf_ax.set_title(display, fontsize=12, pad=8)

        fig.suptitle(title, fontsize=17)
        fig.legend(
            handles=[
                Line2D([0], [0], color="#264653", linestyle="-", linewidth=2.7,
                       label="IllustrisTNG"),
                Line2D([0], [0], color="#d1495b", linestyle="--", linewidth=2.7,
                       label=f"HalfDome total/matched window, {radius_label}"),
            ],
            fontsize=12, framealpha=1.0, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(
            0.5, -0.015,
            f"Percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center", fontsize=10,
        )
        if missing:
            print("HalfDome empty/unplotted: " + ", ".join(missing))
        return self._finish(fig, filename, show)

    def plot_mass_upper(self, show: bool = True) -> Path:
        entries = [
            ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16", "black"),
            (r"$10^{10}\!-\!10^{12}\,M_\odot$", "m1e10_to_1e12", "m1e10_to_1e12", "black"),
            (r"$10^{10}\!-\!10^{13}\,M_\odot$", "m1e10_to_1e13", "m1e10_to_1e13", "black"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14", "black"),
            (r"$10^{10}\!-\!10^{15}\,M_\odot$", "m1e10_to_1e15", "m1e10_to_1e15", "black"),
        ]
        return self._plot_mass_windows(
            entries, "tng_vs_halfdome_3r200_upper_mass_windows.png",
            r"IllustrisTNG versus HalfDome 3R200: upper mass limits, $z_s=1$",
            5000.0, show,
        )

    def plot_mass_to_1e14(self, show: bool = True) -> Path:
        entries = [
            ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16", "black"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14", "black"),
            (r"$10^{11}\!-\!10^{14}\,M_\odot$", "m1e11_to_1e14", "m1e11_to_1e14", "black"),
            (r"$10^{12}\!-\!10^{14}\,M_\odot$", "m1e12_to_1e14", "m1e12_to_1e14", "black"),
            (r"$10^{13}\!-\!10^{14}\,M_\odot$", "m1e13_to_1e14", "m1e13_to_1e14", "black"),
        ]
        return self._plot_mass_windows(
            entries, "tng_vs_halfdome_3r200_mass_windows_to_1e14.png",
            r"IllustrisTNG versus HalfDome 3R200: windows ending at $10^{14}M_\odot$, $z_s=1$",
            10_000.0, show,
        )

    def plot_mass_upper_fixed_200c(self, show: bool = True) -> Path:
        entries = [
            ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16", "black"),
            (r"$10^{10}\!-\!10^{12}\,M_\odot$", "m1e10_to_1e12", "m1e10_to_1e12", "black"),
            (r"$10^{10}\!-\!10^{13}\,M_\odot$", "m1e10_to_1e13", "m1e10_to_1e13", "black"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14", "black"),
            (r"$10^{10}\!-\!10^{15}\,M_\odot$", "m1e10_to_1e15", "m1e10_to_1e15", "black"),
        ]
        return self._plot_mass_windows(
            entries,
            "tng_vs_halfdome_z1_nside4096_3r200c_upper_m200c_windows.png",
            r"IllustrisTNG versus HalfDome: $M_{200c}$ upper limits, "
            r"$z_s=1$, NSIDE=4096, $R_{max}=3R_{200c}$",
            5000.0,
            show,
            nside=FIXED_200C_NSIDE,
            validated_200c=True,
            radius_label="3R200c",
        )

    def plot_mass_to_1e14_fixed_200c(self, show: bool = True) -> Path:
        entries = [
            ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16", "black"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14", "black"),
            (r"$10^{11}\!-\!10^{14}\,M_\odot$", "m1e11_to_1e14", "m1e11_to_1e14", "black"),
            (r"$10^{12}\!-\!10^{14}\,M_\odot$", "m1e12_to_1e14", "m1e12_to_1e14", "black"),
            (r"$10^{13}\!-\!10^{14}\,M_\odot$", "m1e13_to_1e14", "m1e13_to_1e14", "black"),
        ]
        return self._plot_mass_windows(
            entries,
            "tng_vs_halfdome_z1_nside4096_3r200c_m200c_windows_to_1e14.png",
            r"IllustrisTNG versus HalfDome: $M_{200c}$ windows ending at "
            r"$10^{14}M_\odot$, $z_s=1$, NSIDE=4096, $R_{max}=3R_{200c}$",
            10_000.0,
            show,
            nside=FIXED_200C_NSIDE,
            validated_200c=True,
            radius_label="3R200c",
        )

    def plot_redshift(self, show: bool = True) -> Path:
        fig = plt.figure(figsize=(17, 8.2), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=(3.0, 1.35), hspace=0.06)
        for column, redshift in enumerate(SOURCE_REDSHIFTS):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            hd = self.load_halfdome(redshift, 8192, 3.0, DEFAULT_WINDOW)
            tng_pdf, tng_counts, tng_zero = self.histogram_from_values(
                self.tng_values("all", redshift), hd["edges"]
            )
            self.draw_pdf(
                pdf_ax, hd["centers"], tng_pdf,
                color="#264653", linestyle="-", linewidth=2.5,
            )
            self.draw_pdf(
                pdf_ax, hd["centers"], hd["pdf"],
                color="#d1495b", linestyle="--", linewidth=2.5,
            )
            difference = self.percent_difference(
                hd["pdf"], tng_pdf, hd["counts"], tng_counts
            )
            keep = np.isfinite(difference)
            percent_ax.plot(
                hd["centers"][keep], difference[keep],
                color="#6a4c93", linewidth=1.8,
            )
            self.format_pdf_axis(
                pdf_ax, show_xlabel=False, show_ylabel=(column == 0)
            )
            self.format_percent_axis(
                percent_ax,
                ylabel=r"$(p_{HD}-p_{TNG})/p_{TNG}$ [%]",
                show_ylabel=(column == 0),
            )
            pdf_ax.tick_params(labelbottom=False)
            pdf_ax.set_title(
                rf"$z_s={redshift:g}$" + "\n"
                + f"zero: TNG {100*tng_zero:.2f}%, HD {100*hd['zero_fraction']:.2f}%",
                fontsize=11,
            )
            print(
                f"z={redshift:g}: TNG zero={100*tng_zero:.2f}%, "
                f"HalfDome zero={100*hd['zero_fraction']:.2f}%"
            )

        fig.suptitle(
            r"Redshift evolution: IllustrisTNG versus HalfDome total DM "
            r"(3R200, NSIDE=8192)", fontsize=17,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color="#264653", linestyle="-", linewidth=2.7,
                       label="IllustrisTNG total halo DM"),
                Line2D([0], [0], color="#d1495b", linestyle="--", linewidth=2.7,
                       label="HalfDome total DM (all resolved halos), 3R200"),
            ],
            fontsize=12, framealpha=1.0, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(
            0.5, -0.015,
            f"Percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center", fontsize=10,
        )
        return self._finish(fig, "tng_vs_halfdome_3r200_redshift_evolution.png", show)

    def plot_resolution(self, show: bool = True) -> Path:
        fig = plt.figure(figsize=(17, 8.2), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=(3.0, 1.35), hspace=0.06)

        for column, redshift in enumerate(SOURCE_REDSHIFTS):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            hd2048 = self.load_halfdome(redshift, 2048, 3.0, DEFAULT_WINDOW)
            hd8192 = self.load_halfdome(redshift, 8192, 3.0, DEFAULT_WINDOW)

            self.draw_pdf(
                pdf_ax, hd2048["centers"], hd2048["pdf"], color="#277da1",
                linestyle="-", linewidth=2.5,
                label="HalfDome total DM, NSIDE=2048",
            )
            self.draw_pdf(
                pdf_ax, hd8192["centers"], hd8192["pdf"], color="#f8961e",
                linestyle="--", linewidth=2.5,
                label="HalfDome total DM, NSIDE=8192",
            )
            percent = self.percent_difference(
                hd8192["pdf"], hd2048["pdf"],
                hd8192["counts"], hd2048["counts"],
            )
            percent_ax.plot(
                hd2048["centers"], percent, color="#7b2cbf", linewidth=2.0,
            )

            self.format_pdf_axis(
                pdf_ax, show_xlabel=False, show_ylabel=(column == 0),
            )
            self.format_percent_axis(
                percent_ax, show_xlabel=True, show_ylabel=(column == 0),
                ylabel=r"$(p_{8192}-p_{2048})/p_{2048}$ [%]",
            )
            pdf_ax.tick_params(labelbottom=False)
            pdf_ax.set_title(
                rf"$z_s={redshift:g}$" + "\n"
                + f"zero: 2048 {100*hd2048['zero_fraction']:.2f}%, "
                + f"8192 {100*hd8192['zero_fraction']:.2f}%",
                fontsize=11,
            )

        fig.suptitle(
            r"HalfDome total-DM NSIDE comparison (3R200, all resolved halos)",
            fontsize=17,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color="#277da1", linestyle="-", linewidth=2.7,
                       label="HalfDome total DM, NSIDE=2048"),
                Line2D([0], [0], color="#f8961e", linestyle="--", linewidth=2.7,
                       label="HalfDome total DM, NSIDE=8192"),
            ],
            fontsize=12, framealpha=1.0, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(
            0.5, -0.015,
            f"Percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center", fontsize=10,
        )
        return self._finish(fig, "halfdome_3r200_nside_resolution_comparison.png", show)

    def plot_aperture(self, show: bool = True) -> Path:
        fig = plt.figure(figsize=(17, 8.2), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, 4, height_ratios=(3.0, 1.35), hspace=0.06)

        for column, redshift in enumerate(SOURCE_REDSHIFTS):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            hd1 = self.load_halfdome(redshift, 8192, 1.0, DEFAULT_WINDOW)
            hd3 = self.load_halfdome(redshift, 8192, 3.0, DEFAULT_WINDOW)

            self.draw_pdf(
                pdf_ax, hd1["centers"], hd1["pdf"], color="#d62828",
                linestyle="-", linewidth=2.5,
                label="HalfDome total DM, 1R200",
            )
            self.draw_pdf(
                pdf_ax, hd3["centers"], hd3["pdf"], color="#2a9d8f",
                linestyle="--", linewidth=2.5,
                label="HalfDome total DM, 3R200",
            )
            percent = self.percent_difference(
                hd3["pdf"], hd1["pdf"], hd3["counts"], hd1["counts"],
            )
            percent_ax.plot(
                hd1["centers"], percent, color="#6a4c93", linewidth=2.0,
            )

            self.format_pdf_axis(
                pdf_ax, show_xlabel=False, show_ylabel=(column == 0),
            )
            self.format_percent_axis(
                percent_ax, show_xlabel=True, show_ylabel=(column == 0),
                ylabel=r"$(p_{3R}-p_{1R})/p_{1R}$ [%]",
            )
            pdf_ax.tick_params(labelbottom=False)
            pdf_ax.set_title(
                rf"$z_s={redshift:g}$" + "\n"
                + f"zero: 1R {100*hd1['zero_fraction']:.2f}%, "
                + f"3R {100*hd3['zero_fraction']:.2f}%",
                fontsize=11,
            )

        fig.suptitle(
            r"HalfDome total-DM halo-extension comparison (NSIDE=8192, all resolved halos)",
            fontsize=17,
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color="#d62828", linestyle="-", linewidth=2.7,
                       label="HalfDome total DM, 1R200"),
                Line2D([0], [0], color="#2a9d8f", linestyle="--", linewidth=2.7,
                       label="HalfDome total DM, 3R200"),
            ],
            fontsize=12, framealpha=1.0, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.955),
        )
        fig.text(
            0.5, -0.015,
            f"Percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center", fontsize=10,
        )
        return self._finish(fig, "halfdome_1r200_vs_3r200.png", show)

    def run_all(self, show: bool = True) -> list[Path]:
        self.validate_inputs()
        return [
            self.plot_mass_upper(show=show),
            self.plot_mass_to_1e14(show=show),
            self.plot_redshift(show=show),
            self.plot_resolution(show=show),
            self.plot_aperture(show=show),
        ]

    def run_fixed_200c(self, show: bool = True) -> list[Path]:
        """Generate only the requested z=1, NSIDE=4096, 3R200c comparisons."""
        self.validate_fixed_200c_input()
        return [
            self.plot_mass_upper_fixed_200c(show=show),
            self.plot_mass_to_1e14_fixed_200c(show=show),
        ]
