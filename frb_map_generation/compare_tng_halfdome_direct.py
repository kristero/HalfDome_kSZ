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
ORIGINAL_3R200_NSIDE = 8192

TNG_FILES = {
    "lss_total": "DMall_IGM_halo.npy",
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
    """Load validated histogram products and make the requested comparison plots."""

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
            / "frb_map_generation"
            / "outputs"
        )
        self.hd_previous_base = (
            candidate
            / "frb_catalog_comparison_outputs"
            / "r200c_fix"
        )
        self.tng_base = (
            candidate
            / "frb_catalog_comparison_outputs"
            / "haloDM-20260819T104507Z-1-001"
            / "haloDM"
        )
        self.output_dir = candidate / "tng_halfdome_direct_comparison" / "cluster_HalfDome_pdfs"
        self.bfc_reference_dir = (
            candidate
            / "frb_catalog_comparison_outputs"
            / "analytical_bfc_torkamani2026"
        )
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
        previous_m200m_profile: bool = False,
    ) -> Path:
        if validated_200c and previous_m200m_profile:
            raise ValueError("A run cannot be both corrected and pre-correction")
        name = (
            f"zsrc{_tag(source_redshift)}_nside{int(nside)}_nrays{EXPECTED_RAYS}"
            "_allhalos"
        )
        if validated_200c or previous_m200m_profile:
            # The dedicated cluster PBS workflows always write an explicit
            # r200cx<multiplier> tag, including the 1R200c companion product.
            name += f"_r200cx{_tag(aperture_r200)}"
        elif not np.isclose(aperture_r200, 1.0):
            name += f"_r200x{_tag(aperture_r200)}"
        if validated_200c:
            name += "_m200cprofile"
        elif previous_m200m_profile:
            name += "_m200mprofile"
        if validated_200c:
            base = self.hd_200c_base
        elif previous_m200m_profile:
            base = self.hd_previous_base
        else:
            base = self.hd_base
        return base / f"{name}_seed42"

    def load_halfdome(
        self,
        source_redshift: float,
        nside: int = 8192,
        aperture_r200: float = 3.0,
        window_label: str = DEFAULT_WINDOW,
        *,
        validated_200c: bool = False,
        previous_m200m_profile: bool = False,
        run_name: str | None = None,
    ) -> dict:
        run_dir = self.run_dir(
            source_redshift,
            nside,
            aperture_r200,
            validated_200c=validated_200c,
            previous_m200m_profile=previous_m200m_profile,
        )
        if run_name is not None:
            # Explicit directory name under the same base, e.g. a repeat run
            # with a suffix; all provenance checks below still apply.
            run_dir = run_dir.parent / run_name
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
            profile_angular_support = _decode(
                h5.attrs.get("provenance_profile_angular_support", "")
            )
            catalog_mass_dataset = _decode(
                h5.attrs.get("provenance_catalog_mass_dataset", "")
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
        if validated_200c or previous_m200m_profile:
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
            if validated_200c:
                expected_200c = (
                    "M200c", "R200c", "critical", 200,
                    "M200c", "M200c", "R200c",
                    "M200c", "halo_mass_m200c",
                    "M200c", "halo_mass_m200c", "generator", "R200c",
                    False, False,
                )
                product_name = "validated M200c/R200c"
            else:
                expected_200c = (
                    "M200c", "R200c", "critical", 200,
                    "M200c", "M200m", "R200c",
                    "M200m", "halo_mass_m200m",
                    "M200c", "halo_mass_m200c", "generator", "R200m",
                    False, False,
                )
                product_name = "pre-correction M200m-profile/R200c-aperture"
            if actual_200c != expected_200c:
                raise ValueError(
                    f"HalfDome file is not the requested {product_name} product: "
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
            "profile_angular_support": profile_angular_support,
            "catalog_mass_dataset": catalog_mass_dataset,
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

    def validate_previous_workflow_input(self) -> None:
        """Validate the matching pre-correction z=1, NSIDE=4096 run."""
        previous = self.load_halfdome(
            FIXED_200C_REDSHIFT,
            FIXED_200C_NSIDE,
            FIXED_200C_APERTURE,
            DEFAULT_WINDOW,
            previous_m200m_profile=True,
        )
        expected_support = (
            "generator exact angular filter at angular_size(3.0*R200c); "
            "XGPaint compute_theta_max and paint! bypassed"
        )
        if previous["profile_angular_support"] != expected_support:
            raise ValueError(
                "The comparison file is not the requested pre-correction workflow: "
                f"profile_angular_support={previous['profile_angular_support']!r}, "
                f"expected={expected_support!r}, run={previous['run_dir']}"
            )
        if previous["catalog_mass_dataset"] != "halo_mass_m200c":
            raise ValueError(
                "The previous and corrected mass windows use different catalogue "
                f"mass coordinates: dataset={previous['catalog_mass_dataset']!r}, "
                f"run={previous['run_dir']}"
            )

    def validate_original_3r200_input(self) -> None:
        """Validate the original internal-XGPaint 3R200 comparison product."""
        original = self.load_halfdome(
            FIXED_200C_REDSHIFT,
            ORIGINAL_3R200_NSIDE,
            FIXED_200C_APERTURE,
            DEFAULT_WINDOW,
        )
        expected_support = "XGPaint compute_theta_max with explicit mult=3.0"
        if original["profile_angular_support"] != expected_support:
            raise ValueError(
                "The third curve is not the original internal 3R200 workflow: "
                f"profile_angular_support={original['profile_angular_support']!r}, "
                f"expected={expected_support!r}, run={original['run_dir']}"
            )
        if original["catalog_mass_dataset"] != "halo_mass_m200c":
            raise ValueError(
                "The original comparison does not use the M200c catalogue mass "
                f"coordinate: dataset={original['catalog_mass_dataset']!r}, "
                f"run={original['run_dir']}"
            )

    def validate_fixed_200m_input(self) -> None:
        """Backward-compatible alias; the older run was not a verified R200m run."""
        self.validate_previous_workflow_input()

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
        ax.set_yscale("linear")
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

        # Let constrained layout reserve a dedicated title row.  Forcing y>1
        # placed the main title on top of the per-panel titles in saved PNGs.
        fig.suptitle(title, fontsize=17)
        fig.legend(
            handles=[
                Line2D([0], [0], color="#264653", linestyle="-", linewidth=2.7,
                       label="IllustrisTNG"),
                Line2D([0], [0], color="#d1495b", linestyle="--", linewidth=2.7,
                       label=f"HalfDome total/matched window, {radius_label}"),
            ],
            fontsize=12, framealpha=1.0, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, 0.39),
        )
        fig.text(
            0.5, -0.015,
            f"Linear percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center", fontsize=10,
        )
        if missing:
            print("HalfDome empty/unplotted: " + ", ".join(missing))
        return self._finish(fig, filename, show)

    def _plot_halfdome_radius_definition_windows(
        self, entries, filename, title, xmax, show
    ) -> Path:
        self.validate_fixed_200c_input()
        self.validate_previous_workflow_input()
        columns = len(entries)
        fig = plt.figure(figsize=(4.15 * columns, 8.2), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, columns, height_ratios=(3.0, 1.35), hspace=0.06)

        for column, (display, window_label) in enumerate(entries):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            corrected = self.load_halfdome(
                FIXED_200C_REDSHIFT,
                FIXED_200C_NSIDE,
                FIXED_200C_APERTURE,
                window_label,
                validated_200c=True,
            )
            previous = self.load_halfdome(
                FIXED_200C_REDSHIFT,
                FIXED_200C_NSIDE,
                FIXED_200C_APERTURE,
                window_label,
                previous_m200m_profile=True,
            )
            same_edges = (
                corrected["edges"].shape == previous["edges"].shape
                and np.allclose(corrected["edges"], previous["edges"], rtol=0.0, atol=0.0)
            )
            if not same_edges:
                raise ValueError(
                    "Corrected and pre-correction products use different "
                    f"PDF bin edges for {window_label}"
                )

            self.draw_pdf(
                pdf_ax,
                previous["centers"],
                previous["pdf"],
                color="#277da1",
                linestyle="-",
                linewidth=2.5,
            )
            self.draw_pdf(
                pdf_ax,
                corrected["centers"],
                corrected["pdf"],
                color="#d1495b",
                linestyle="--",
                linewidth=2.5,
            )
            difference = self.percent_difference(
                corrected["pdf"],
                previous["pdf"],
                corrected["counts"],
                previous["counts"],
            )
            keep = np.isfinite(difference)
            percent_ax.plot(
                corrected["centers"][keep],
                difference[keep],
                color="#6a4c93",
                linewidth=1.8,
            )

            self.format_pdf_axis(
                pdf_ax, (0.1, xmax), show_xlabel=False, show_ylabel=(column == 0)
            )
            self.format_percent_axis(
                percent_ax,
                (0.1, xmax),
                ylabel=r"$(p_{corrected}-p_{previous})/p_{previous}$ [%]",
                show_ylabel=(column == 0),
            )
            pdf_ax.tick_params(labelbottom=False)
            pdf_ax.set_title(display, fontsize=12, pad=8)

        fig.suptitle(title, fontsize=17)
        fig.legend(
            handles=[
                Line2D(
                    [0], [0], color="#277da1", linestyle="-", linewidth=2.7,
                    label="HalfDome pre-correction M200m-routed profile",
                ),
                Line2D(
                    [0], [0], color="#d1495b", linestyle="--", linewidth=2.7,
                    label="HalfDome consistent M200c profile + external 3R200c",
                ),
            ],
            fontsize=12,
            framealpha=1.0,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.39),
        )
        fig.text(
            0.5,
            -0.015,
            f"Linear percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center",
            fontsize=10,
        )
        fig.text(
            0.5,
            -0.04,
            "Both use M200c bins and an external 3R200c aperture; only the profile mass "
            "routed into XGPaint changes from M200m to M200c.",
            ha="center",
            fontsize=9,
        )
        return self._finish(fig, filename, show)

    def _plot_three_way_mass_windows(
        self, entries, filename, title, xmax, show
    ) -> Path:
        """Plot TNG and two HalfDome workflows with both differences versus TNG."""
        self.validate_fixed_200c_input()
        self.validate_original_3r200_input()
        columns = len(entries)
        fig = plt.figure(figsize=(4.15 * columns, 8.2), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, columns, height_ratios=(3.0, 1.35), hspace=0.06)
        missing = []

        for column, (display, tng_label, hd_label) in enumerate(entries):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            corrected = self.load_halfdome(
                FIXED_200C_REDSHIFT,
                FIXED_200C_NSIDE,
                FIXED_200C_APERTURE,
                hd_label,
                validated_200c=True,
            )
            original = self.load_halfdome(
                FIXED_200C_REDSHIFT,
                ORIGINAL_3R200_NSIDE,
                FIXED_200C_APERTURE,
                hd_label,
            )
            if not (
                corrected["edges"].shape == original["edges"].shape
                and np.allclose(corrected["edges"], original["edges"], rtol=0.0, atol=0.0)
            ):
                raise ValueError(f"HalfDome PDF edges differ for {hd_label}")

            tng_pdf, tng_counts, _ = self.histogram_from_values(
                self.tng_values(tng_label, FIXED_200C_REDSHIFT), corrected["edges"]
            )
            self.draw_pdf(
                pdf_ax, corrected["centers"], tng_pdf,
                color="#264653", linestyle="-", linewidth=2.7,
            )
            available = False
            for product, color, linestyle in (
                (corrected, "#d1495b", "--"),
                (original, "#277da1", ":"),
            ):
                if not np.any(product["pdf"] > 0.0):
                    continue
                available = True
                self.draw_pdf(
                    pdf_ax, product["centers"], product["pdf"],
                    color=color, linestyle=linestyle, linewidth=2.5,
                )
                difference = self.percent_difference(
                    product["pdf"], tng_pdf, product["counts"], tng_counts
                )
                keep = np.isfinite(difference)
                percent_ax.plot(
                    product["centers"][keep], difference[keep],
                    color=color, linestyle=linestyle, linewidth=1.9,
                )
            if not available:
                missing.append(display)
                pdf_ax.text(
                    0.5, 0.12, "HalfDome unavailable\nat catalogue resolution",
                    transform=pdf_ax.transAxes, ha="center", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
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
                       label="HalfDome corrected: M200c profile + external 3R200c, NSIDE=4096"),
                Line2D([0], [0], color="#277da1", linestyle=":", linewidth=2.7,
                       label="HalfDome original: M200c profile + internal 3R200, NSIDE=8192"),
            ],
            fontsize=11, framealpha=1.0, loc="lower center", ncol=3,
            bbox_to_anchor=(0.5, 0.39),
        )
        fig.text(
            0.5, -0.015,
            f"Both linear percentage curves use TNG as the denominator and require at least "
            f"{MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
            ha="center", fontsize=10,
        )
        fig.text(
            0.5, -0.04,
            "The original all-halo 3R200 product is available locally only at NSIDE=8192; "
            "the corrected product is NSIDE=4096.",
            ha="center", fontsize=9,
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

    def plot_three_way_mass_upper_fixed_200c(self, show: bool = True) -> Path:
        entries = [
            ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16"),
            (r"$10^{10}\!-\!10^{12}\,M_\odot$", "m1e10_to_1e12", "m1e10_to_1e12"),
            (r"$10^{10}\!-\!10^{13}\,M_\odot$", "m1e10_to_1e13", "m1e10_to_1e13"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14"),
            (r"$10^{10}\!-\!10^{15}\,M_\odot$", "m1e10_to_1e15", "m1e10_to_1e15"),
        ]
        return self._plot_three_way_mass_windows(
            entries,
            "tng_vs_halfdome_corrected_vs_original_3r200_upper_m200c_windows.png",
            r"TNG versus corrected and original HalfDome workflows: "
            r"upper $M_{200c}$ limits, $z_s=1$",
            5000.0,
            show,
        )

    def plot_three_way_mass_to_1e14_fixed_200c(self, show: bool = True) -> Path:
        entries = [
            ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14"),
            (r"$10^{11}\!-\!10^{14}\,M_\odot$", "m1e11_to_1e14", "m1e11_to_1e14"),
            (r"$10^{12}\!-\!10^{14}\,M_\odot$", "m1e12_to_1e14", "m1e12_to_1e14"),
            (r"$10^{13}\!-\!10^{14}\,M_\odot$", "m1e13_to_1e14", "m1e13_to_1e14"),
        ]
        return self._plot_three_way_mass_windows(
            entries,
            "tng_vs_halfdome_corrected_vs_original_3r200_to_1e14_m200c_windows.png",
            r"TNG versus corrected and original HalfDome workflows: "
            r"$M_{200c}$ windows ending at $10^{14}M_\odot$, $z_s=1$",
            10_000.0,
            show,
        )

    def plot_published_bfc_vs_tng_lss_z0p7(self, show: bool = True) -> Path:
        """Compare the published BFC Figure 6 curve with the matching total TNG LSS PDF."""
        source_redshift = 0.7
        curve_path = self.bfc_reference_dir / "figure6_bfc_z0p7_digitized.csv"
        if not curve_path.exists():
            raise FileNotFoundError(f"Missing published BFC reference curve: {curve_path}")
        curve = np.genfromtxt(curve_path, delimiter=",", names=True)
        bfc_dm = np.asarray(curve["dm_pc_cm3"], dtype=float)
        bfc_pdf = np.asarray(curve["pdf_per_pc_cm3"], dtype=float)
        if not (
            len(bfc_dm) > 2
            and np.all(np.isfinite(bfc_dm))
            and np.all(np.isfinite(bfc_pdf))
            and np.all(np.diff(bfc_dm) > 0.0)
            and np.all(bfc_pdf >= 0.0)
        ):
            raise ValueError(f"Invalid BFC reference curve: {curve_path}")

        edges = np.linspace(300.0, 1450.0, 81)
        centers = 0.5 * (edges[1:] + edges[:-1])
        tng_pdf, tng_counts, _ = self.histogram_from_values(
            self.tng_values("lss_total", source_redshift), edges
        )
        bfc_on_bins = np.interp(centers, bfc_dm, bfc_pdf, left=np.nan, right=np.nan)
        difference = np.full_like(tng_pdf, np.nan)
        supported = (
            np.isfinite(bfc_on_bins)
            & (tng_pdf > 0.0)
            & (tng_counts >= MIN_PERCENT_BIN_COUNT)
        )
        difference[supported] = 100.0 * (
            bfc_on_bins[supported] - tng_pdf[supported]
        ) / tng_pdf[supported]

        fig = plt.figure(figsize=(9.2, 8.0), dpi=110, layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=(3.0, 1.25), hspace=0.06)
        pdf_ax = fig.add_subplot(grid[0, 0])
        percent_ax = fig.add_subplot(grid[1, 0], sharex=pdf_ax)
        pdf_ax.plot(
            centers, tng_pdf, color="#264653", linewidth=2.6,
            label="IllustrisTNG total LSS (local array)",
        )
        pdf_ax.plot(
            bfc_dm, bfc_pdf, color="#e76f51", linestyle="--", linewidth=2.6,
            label="BFC analytical model (published Figure 6, digitized)",
        )
        percent_ax.plot(
            centers[supported], difference[supported],
            color="#6a4c93", linewidth=2.0,
        )
        percent_ax.axhline(0.0, color="0.35", linewidth=1.0)

        pdf_ax.set_xlim(300.0, 1450.0)
        pdf_ax.set_ylim(bottom=0.0)
        pdf_ax.set_ylabel(r"$p_{LSS}(\mathrm{DM})$ [$(\mathrm{pc}\,\mathrm{cm}^{-3})^{-1}$]", fontsize=12)
        pdf_ax.tick_params(labelbottom=False, direction="in", labelsize=10)
        pdf_ax.grid(alpha=0.2)
        pdf_ax.legend(fontsize=11, framealpha=1.0, loc="upper right")
        percent_ax.set_xlim(300.0, 1450.0)
        percent_ax.set_xlabel(r"DM [pc cm$^{-3}$]", fontsize=12)
        percent_ax.set_ylabel(r"$(p_{BFC}-p_{TNG})/p_{TNG}$ [%]", fontsize=11)
        percent_ax.tick_params(direction="in", labelsize=10)
        percent_ax.grid(alpha=0.2)
        fig.suptitle(
            r"Published BFC analytical total-LSS PDF versus IllustrisTNG, $z_s=0.7$",
            fontsize=16,
        )
        fig.text(
            0.5, -0.015,
            "BFC curve digitized from Torkamani et al. (2026), Figure 6; "
            "published interval retained without renormalization.",
            ha="center", fontsize=9,
        )
        fig.text(
            0.5, -0.04,
            f"Linear percentage panel uses TNG as denominator and requires at least "
            f"{MIN_PERCENT_BIN_COUNT} TNG samples per bin.",
            ha="center", fontsize=9,
        )
        return self._finish(
            fig, "torkamani2026_bfc_figure6_vs_tng_total_lss_z0p7.png", show
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

    def plot_halfdome_corrected_vs_previous_upper(self, show: bool = True) -> Path:
        entries = [
            ("Total: all resolved halos", "m1e10_to_1e16"),
            (r"$10^{10}\!-\!10^{12}\,M_\odot$", "m1e10_to_1e12"),
            (r"$10^{10}\!-\!10^{13}\,M_\odot$", "m1e10_to_1e13"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14"),
            (r"$10^{10}\!-\!10^{15}\,M_\odot$", "m1e10_to_1e15"),
        ]
        return self._plot_halfdome_radius_definition_windows(
            entries,
            "halfdome_z1_nside4096_consistent_m200c_vs_previous_upper_windows.png",
            r"HalfDome: consistent $M_{200c}$/$3R_{200c}$ versus previous workflow, "
            r"upper $M_{200c}$ limits, $z_s=1$, NSIDE=4096",
            5000.0,
            show,
        )

    def plot_halfdome_corrected_vs_previous_to_1e14(self, show: bool = True) -> Path:
        entries = [
            ("Total: all resolved halos", "m1e10_to_1e16"),
            (r"$10^{10}\!-\!10^{14}\,M_\odot$", "m1e10_to_1e14"),
            (r"$10^{11}\!-\!10^{14}\,M_\odot$", "m1e11_to_1e14"),
            (r"$10^{12}\!-\!10^{14}\,M_\odot$", "m1e12_to_1e14"),
            (r"$10^{13}\!-\!10^{14}\,M_\odot$", "m1e13_to_1e14"),
        ]
        return self._plot_halfdome_radius_definition_windows(
            entries,
            "halfdome_z1_nside4096_consistent_m200c_vs_previous_to_1e14.png",
            r"HalfDome: consistent $M_{200c}$/$3R_{200c}$ versus previous workflow, "
            r"$M_{200c}$ windows ending at $10^{14}M_\odot$, "
            r"$z_s=1$, NSIDE=4096",
            10_000.0,
            show,
        )

    def plot_halfdome_3r200c_vs_3r200m_upper(self, show: bool = True) -> Path:
        """Backward-compatible alias; the previous product was not verified as R200m."""
        return self.plot_halfdome_corrected_vs_previous_upper(show=show)

    def plot_halfdome_3r200c_vs_3r200m_to_1e14(self, show: bool = True) -> Path:
        """Backward-compatible alias; the previous product was not verified as R200m."""
        return self.plot_halfdome_corrected_vs_previous_to_1e14(show=show)

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
            fontsize=12, framealpha=1.0, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, 0.39),
        )
        fig.text(
            0.5, -0.015,
            f"Linear percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
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
            fontsize=12, framealpha=1.0, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, 0.39),
        )
        fig.text(
            0.5, -0.015,
            f"Linear percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
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
            fontsize=12, framealpha=1.0, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, 0.39),
        )
        fig.text(
            0.5, -0.015,
            f"Linear percentage panels use bins containing at least {MIN_PERCENT_BIN_COUNT} rays in both PDFs.",
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
