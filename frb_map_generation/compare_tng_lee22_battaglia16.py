#!/usr/bin/env python3
"""Compare a provenance-validated Lee22 FRB PDF with TNG and Battaglia16."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .compare_tng_halfdome_direct import (
    EXPECTED_RAYS,
    FIXED_200C_APERTURE,
    FIXED_200C_NSIDE,
    FIXED_200C_REDSHIFT,
    H5_NAME,
    DirectComparison,
)


LEE22_RUN_NAME = (
    "zsrc1p0_nside4096_nrays120000_allhalos_"
    "r200cx3p0_m200c_lee2022_tablea2_noconc_seed42"
)
LEE22_OUTPUT_BASE_NAME = (
    "halfdome_lee2022_tablea2_noconcentration_z1_120k_"
    "foreground_mass_histograms"
)
PREVIOUS_BATTAGLIA_RUN_NAME = (
    "zsrc1p0_nside4096_nrays120000_allhalos_"
    "r200cx3p0_m200cprofile_seed42"
)

UPPER_ENTRIES = (
    ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16"),
    (r"$10^{10}-10^{12}\,M_\odot$", "m1e10_to_1e12", "m1e10_to_1e12"),
    (r"$10^{10}-10^{13}\,M_\odot$", "m1e10_to_1e13", "m1e10_to_1e13"),
    (r"$10^{10}-10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14"),
    (r"$10^{10}-10^{15}\,M_\odot$", "m1e10_to_1e15", "m1e10_to_1e15"),
)
TO_1E14_ENTRIES = (
    ("Total: TNG all / HD all resolved", "all", "m1e10_to_1e16"),
    (r"$10^{10}-10^{14}\,M_\odot$", "m1e10_to_1e14", "m1e10_to_1e14"),
    (r"$10^{11}-10^{14}\,M_\odot$", "m1e11_to_1e14", "m1e11_to_1e14"),
    (r"$10^{12}-10^{14}\,M_\odot$", "m1e12_to_1e14", "m1e12_to_1e14"),
    (r"$10^{13}-10^{14}\,M_\odot$", "m1e13_to_1e14", "m1e13_to_1e14"),
)
ALL_REQUIRED_WINDOWS = tuple(
    dict.fromkeys(entry[2] for entry in UPPER_ENTRIES + TO_1E14_ENTRIES)
)


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _attrs(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs.items())


def _profile(attrs: dict[str, Any]) -> str:
    return _text(attrs.get("provenance_dm_profile", "")).strip().lower()


def _is_requested_lee22(path: Path) -> bool:
    try:
        attrs = _attrs(path)
        return (
            _profile(attrs) == "lee2022"
            and np.isclose(float(attrs["provenance_source_redshift"]), 1.0)
            and int(attrs["provenance_nside"]) == FIXED_200C_NSIDE
            and int(attrs["n_rays"]) == EXPECTED_RAYS
            and np.isclose(
                float(attrs["provenance_halo_extension_r200_multiplier"]),
                FIXED_200C_APERTURE,
            )
        )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def expected_lee22_h5(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    return (
        root
        / "frb_catalog_comparison_outputs"
        / LEE22_OUTPUT_BASE_NAME
        / LEE22_RUN_NAME
        / H5_NAME
    )


def resolve_lee22_h5(
    project_root: str | Path, explicit_path: str | Path | None = None
) -> Path:
    """Resolve exactly one genuinely Lee22-signed z=1/4096/3R200c product."""
    root = Path(project_root).expanduser().resolve()
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.is_dir():
            direct = candidate / H5_NAME
            matches = [direct] if direct.is_file() else list(candidate.rglob(H5_NAME))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one {H5_NAME} below {candidate}; found {len(matches)}"
                )
            candidate = matches[0]
        if not candidate.is_file():
            raise FileNotFoundError(f"Lee22 histogram file does not exist: {candidate}")
        if not _is_requested_lee22(candidate):
            raise ValueError(
                "Selected file is not Lee22 z=1, NSIDE=4096, 120k, 3R200c: "
                f"profile={_profile(_attrs(candidate))!r}, path={candidate}"
            )
        return candidate

    expected = expected_lee22_h5(root)
    if expected.is_file():
        if not _is_requested_lee22(expected):
            raise ValueError(
                f"Expected Lee22 path contains profile={_profile(_attrs(expected))!r}: "
                f"{expected}"
            )
        return expected

    search_root = root / "frb_catalog_comparison_outputs"
    matches = [path for path in search_root.rglob(H5_NAME) if _is_requested_lee22(path)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Multiple valid Lee22 products found; set LEE22_H5 explicitly:\n  "
            + "\n  ".join(map(str, matches))
        )

    cluster_source = (
        "/lustre/work/kristero10/frb_data/"
        f"{LEE22_OUTPUT_BASE_NAME}/"
    )
    local_base = search_root / LEE22_OUTPUT_BASE_NAME
    raise FileNotFoundError(
        "No provenance-valid Lee22 histogram product is present locally.\n"
        f"Expected local file:\n  {expected}\n"
        "Copy the completed cluster run, for example:\n"
        "  rsync -av kristero10@idark.ipmu.jp:"
        f"{cluster_source} {local_base}/"
    )


def load_lee22_product(path: str | Path) -> dict[str, Any]:
    """Load all Lee22 mass windows after checking the scientific contract."""
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs.items())
        labels = tuple(_text(value) for value in handle["window_label"][:])
        edges = np.asarray(handle["pdf_bin_edges_pc_cm3"][:], dtype=float)
        centers = np.asarray(handle["pdf_bin_centers_pc_cm3"][:], dtype=float)
        density = np.asarray(handle["pdf_density_per_pc_cm3"][:], dtype=float)
        counts = np.asarray(handle["pdf_count"][:], dtype=np.int64)
        requested_min = np.asarray(handle["window_requested_min_msun"][:], dtype=float)
        requested_max = np.asarray(handle["window_requested_max_msun"][:], dtype=float)
        effective_min = np.asarray(handle["window_effective_min_msun"][:], dtype=float)
        effective_max = np.asarray(handle["window_effective_max_msun"][:], dtype=float)

    if density.shape[0] != len(labels):
        density, counts = density.T, counts.T
    expected_shape = (len(labels), len(centers))
    if density.shape != expected_shape or counts.shape != expected_shape:
        raise ValueError(
            f"Unexpected Lee22 histogram shapes: density={density.shape}, "
            f"counts={counts.shape}, expected={expected_shape}, path={path}"
        )

    required_attrs = {
        "provenance_dm_profile": "lee2022",
        "provenance_lee2022_concentration_mode": "none",
        "provenance_dm_model_family": (
            "lee2022_table_a2_no_concentration_m200c_profile_owned_los_v2"
        ),
        "provenance_lee2022_density_table": "Appendix A, Table A2",
        "provenance_halo_mass_definition": "M200c",
        "provenance_halo_radius_definition": "R200c",
        "provenance_halo_reference_density": "critical",
        "provenance_xgpaint_profile_input_mass_definition": "M200c",
        "provenance_xgpaint_profile_input_mass_dataset": "halo_mass_m200c",
        "provenance_aperture_geometry_owner": "generator",
        "provenance_sightline_catalog_path": "",
        "provenance_lee2022_los_integrator": (
            "profile-owned explicit Lee22 GNFW integrand with QuadGK"
        ),
        "provenance_dm_cache_profile_signature": (
            "lee2022_table_a2_no_concentration_v2|M200c|R200c|alpha=1|"
            "gamma=-0.3|XH=0.76|profile_owned_los|observer=1/(1+z)"
        ),
    }
    for key, expected in required_attrs.items():
        actual = _text(attrs.get(key, ""))
        if actual != expected:
            raise ValueError(
                f"Lee22 provenance mismatch for {key}: actual={actual!r}, "
                f"expected={expected!r}, path={path}"
            )
    numeric_contract = {
        "provenance_source_redshift": FIXED_200C_REDSHIFT,
        "provenance_nside": FIXED_200C_NSIDE,
        "n_rays": EXPECTED_RAYS,
        "provenance_halo_extension_r200_multiplier": FIXED_200C_APERTURE,
        "provenance_frb_seed": 42,
    }
    for key, expected in numeric_contract.items():
        actual = float(attrs.get(key, np.nan))
        if not np.isclose(actual, float(expected)):
            raise ValueError(
                f"Lee22 provenance mismatch for {key}: actual={actual}, "
                f"expected={expected}, path={path}"
            )
    if int(attrs.get("provenance_catalog_streamed_halos", -1)) != int(
        attrs.get("provenance_catalog_total_halos", -2)
    ):
        raise ValueError(f"Lee22 product did not stream the full catalogue: {path}")
    missing = [label for label in ALL_REQUIRED_WINDOWS if label not in labels]
    if missing:
        raise KeyError(f"Lee22 product is missing mass windows {missing}: {path}")

    windows = {}
    for index, label in enumerate(labels):
        windows[label] = {
            "edges": edges,
            "centers": centers,
            "pdf": density[index],
            "counts": counts[index],
            "requested_min_msun": float(requested_min[index]),
            "requested_max_msun": float(requested_max[index]),
            "effective_min_msun": float(effective_min[index]),
            "effective_max_msun": float(effective_max[index]),
            "zero_fraction": 1.0 - float(counts[index].sum()) / EXPECTED_RAYS,
        }
    return {"path": path, "attrs": attrs, "windows": windows}


class Lee22TngBattagliaComparison:
    """Three-way z=1 comparison with linear differences relative to TNG."""

    def __init__(
        self,
        project_root: str | Path = "/home/cbllover/HalfDome",
        *,
        lee22_h5: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.direct = DirectComparison(self.project_root)
        self.lee22_h5 = resolve_lee22_h5(self.project_root, lee22_h5)
        self.lee22 = load_lee22_product(self.lee22_h5)
        self.output_dir = (
            self.project_root
            / "tng_halfdome_direct_comparison"
            / "cluster_HalfDome_pdfs"
            / "lee22_vs_tng_vs_battaglia16_best"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.previous_battaglia_h5 = (
            self.project_root
            / "frb_map_generation"
            / "outputs"
            / PREVIOUS_BATTAGLIA_RUN_NAME
            / H5_NAME
        )
        self._validate_battaglia()

    def _validate_battaglia(self) -> None:
        self.direct.validate_fixed_200c_input()
        attrs = _attrs(self.previous_battaglia_h5)
        dm_profile = _profile(attrs)
        profile_repr = _text(attrs.get("provenance_profile", ""))
        if dm_profile not in ("", "battaglia16"):
            raise ValueError(
                f"Previous-best file has unexpected profile={dm_profile!r}: "
                f"{self.previous_battaglia_h5}"
            )
        if dm_profile != "battaglia16" and "BattagliaTauProfile" not in profile_repr:
            raise ValueError(
                "Previous-best file is not provenance-identifiable as Battaglia16: "
                f"{self.previous_battaglia_h5}"
            )

    def validate_inputs(self) -> None:
        for label in ALL_REQUIRED_WINDOWS:
            lee = self.lee22["windows"][label]
            battaglia = self.direct.load_halfdome(
                FIXED_200C_REDSHIFT,
                FIXED_200C_NSIDE,
                FIXED_200C_APERTURE,
                label,
                validated_200c=True,
            )
            if not np.array_equal(lee["edges"], battaglia["edges"]):
                raise ValueError(
                    f"Lee22 and Battaglia16 use different PDF bin edges for {label}"
                )

    def audit_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label in ALL_REQUIRED_WINDOWS:
            lee = self.lee22["windows"][label]
            battaglia = self.direct.load_halfdome(
                1.0, 4096, 3.0, label, validated_200c=True
            )
            rows.append(
                {
                    "window": label,
                    "lee22_requested_min_msun": lee["requested_min_msun"],
                    "lee22_requested_max_msun": lee["requested_max_msun"],
                    "lee22_effective_min_msun": lee["effective_min_msun"],
                    "lee22_effective_max_msun": lee["effective_max_msun"],
                    "lee22_zero_fraction": lee["zero_fraction"],
                    "battaglia16_zero_fraction": battaglia["zero_fraction"],
                    "max_abs_lee22_minus_battaglia16_pdf": float(
                        np.max(np.abs(lee["pdf"] - battaglia["pdf"]))
                    ),
                }
            )
        return rows

    def save_audit_csv(self) -> Path:
        rows = self.audit_rows()
        path = self.output_dir / "lee22_vs_battaglia16_mass_window_audit.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _plot(self, entries, stem: str, title: str, xmax: float) -> tuple[Path, Path]:
        columns = len(entries)
        fig = plt.figure(figsize=(4.25 * columns, 8.8), dpi=110, layout="constrained")
        grid = fig.add_gridspec(
            3, columns, height_ratios=(3.0, 1.35, 0.50), hspace=0.06
        )

        for column, (display_label, tng_label, hd_label) in enumerate(entries):
            pdf_ax = fig.add_subplot(grid[0, column])
            percent_ax = fig.add_subplot(grid[1, column], sharex=pdf_ax)
            lee = self.lee22["windows"][hd_label]
            battaglia = self.direct.load_halfdome(
                1.0, 4096, 3.0, hd_label, validated_200c=True
            )
            tng_pdf, tng_counts, _ = self.direct.histogram_from_values(
                self.direct.tng_values(tng_label, 1.0), lee["edges"]
            )

            self.direct.draw_pdf(
                pdf_ax, lee["centers"], tng_pdf,
                color="#264653", linewidth=2.8, zorder=2,
            )
            self.direct.draw_pdf(
                pdf_ax, battaglia["centers"], battaglia["pdf"],
                color="#277da1", linestyle="--", linewidth=2.8, zorder=3,
            )
            self.direct.draw_pdf(
                pdf_ax, lee["centers"], lee["pdf"],
                color="#d1495b", linestyle="-", linewidth=1.8,
                marker="o", markersize=2.6, markevery=18, zorder=4,
            )

            for product, color, linestyle, linewidth, marker in (
                (battaglia, "#277da1", "--", 2.6, None),
                (lee, "#d1495b", "-", 1.6, "o"),
            ):
                difference = self.direct.percent_difference(
                    product["pdf"], tng_pdf, product["counts"], tng_counts
                )
                keep = np.isfinite(difference)
                percent_ax.plot(
                    product["centers"][keep], difference[keep],
                    color=color, linestyle=linestyle, linewidth=linewidth,
                    marker=marker, markersize=2.3 if marker else 0, markevery=18,
                )

            self.direct.format_pdf_axis(
                pdf_ax, (0.1, xmax), show_xlabel=False, show_ylabel=(column == 0)
            )
            self.direct.format_percent_axis(
                percent_ax, (0.1, xmax),
                ylabel=r"$(p_{model}-p_{TNG})/p_{TNG}$ [%]",
                show_ylabel=(column == 0),
            )
            pdf_ax.tick_params(labelbottom=False)
            pdf_ax.set_title(display_label, fontsize=12, pad=8)

        legend_ax = fig.add_subplot(grid[2, :])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=(
                Line2D([0], [0], color="#264653", linewidth=2.8, label="IllustrisTNG"),
                Line2D(
                    [0], [0], color="#d1495b", linewidth=1.8, marker="o",
                    markersize=4, label="HalfDome Lee22 Table A2, no concentration",
                ),
                Line2D(
                    [0], [0], color="#277da1", linewidth=2.8, linestyle="--",
                    label="HalfDome Battaglia16 previous best",
                ),
            ),
            loc="upper center", ncol=3, fontsize=10.5, framealpha=1.0,
        )
        legend_ax.text(
            0.5, 0.02,
            "Both HalfDome models: M200c bins and masses, external 3R200c, "
            "z_source=1, NSIDE=4096, 120,000 uniform rays. "
            "Percentage axes are linear and use TNG as the denominator.",
            ha="center", va="bottom", fontsize=9.2,
        )
        fig.suptitle(title, fontsize=17)

        png_path = self.output_dir / f"{stem}.png"
        pdf_path = self.output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        return png_path, pdf_path

    def run_all(self) -> dict[str, Path]:
        self.validate_inputs()
        upper_png, upper_pdf = self._plot(
            UPPER_ENTRIES,
            "tng_vs_lee22_vs_battaglia16_upper_m200c_windows",
            r"IllustrisTNG versus Lee22 and Battaglia16: upper $M_{200c}$ limits",
            5_000.0,
        )
        to14_png, to14_pdf = self._plot(
            TO_1E14_ENTRIES,
            "tng_vs_lee22_vs_battaglia16_to_1e14_m200c_windows",
            r"IllustrisTNG versus Lee22 and Battaglia16: windows ending at $10^{14}M_\odot$",
            10_000.0,
        )
        return {
            "upper_png": upper_png,
            "upper_pdf": upper_pdf,
            "to_1e14_png": to14_png,
            "to_1e14_pdf": to14_pdf,
            "audit_csv": self.save_audit_csv(),
        }
