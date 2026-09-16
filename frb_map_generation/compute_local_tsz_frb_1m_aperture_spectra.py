#!/usr/bin/env python3
"""Measure local HalfDome tSZ/FRB spectra for sparse fixed-z FRB rays.

The tSZ input is the internally painted full-sky Compton-y map plus its strict
key-value provenance record. Each FRB input is the per-ray HDF5
output of generate_halfdome_z1_dm_mass_windows.jl. The sparse estimator is

    DM_hat(p) = (N_pix/N_FRB) [DM_i - <DM>]

at sampled pixels and zero elsewhere. Its auto-spectrum is saved before and
after subtracting the analytic catalogue shot noise <q^2>/nbar. No beam,
instrumental noise, or pixel-window deconvolution is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SO_B40_EDGES = np.concatenate((np.arange(80, 7881, 200), [7979]))
EXPECTED_Z_SOURCE = 1.0
EXPECTED_NFRB = 1_000_000
EXPECTED_NSIDE = 4096


@dataclass(frozen=True)
class FRBSpec:
    label: str
    profile: str
    aperture_r200c: float
    path: Path


@dataclass
class FRBData:
    spec: FRBSpec
    pixels: np.ndarray
    dm: np.ndarray
    attrs: dict[str, object]


def decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_provenance(path: Path) -> dict[str, str]:
    """Read the generator's deliberately simple key=value provenance format."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing tSZ provenance: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Malformed tSZ provenance line {line_number}: {raw_line!r}")
        key, value = line.split("=", 1)
        if not key:
            raise ValueError(f"Empty key in tSZ provenance line {line_number}.")
        values[key] = value
    return values


def validate_tsz_provenance(
    provenance_path: Path,
    map_path: Path,
    nside: int,
) -> dict[str, str]:
    values = read_provenance(provenance_path)
    expected = {
        "observable": "thermal SZ Compton-y",
        "profile_mass_definition": "M200c",
        "profile_mass_units": "physical Msun",
        "catalog_mass_dataset": "halo_mass_m200c",
        "aperture_radius_definition": "R200c",
        "ordering": "RING",
        "map_units": "dimensionless Compton-y",
        "beam": "none",
        "instrumental_noise": "none",
        "mask": "none",
        "catalog_truncated": "false",
    }
    for key, wanted in expected.items():
        found = values.get(key)
        if found != wanted:
            raise ValueError(
                f"tSZ provenance {key}={found!r}; expected {wanted!r}: {provenance_path}"
            )
    if not values.get("profile_label", "").startswith("Battaglia12 fiducial"):
        raise ValueError(f"tSZ map is not the fiducial Battaglia12 profile: {provenance_path}")
    if "complete resolved catalogue range" not in values.get("mass_selection", ""):
        raise ValueError(f"tSZ map does not use the complete halo-mass range: {provenance_path}")
    if not np.isposinf(float(values.get("maximum_halo_redshift_requested", "nan"))):
        raise ValueError(f"tSZ map was redshift-truncated: {provenance_path}")
    if int(values.get("nside", -1)) != nside:
        raise ValueError(f"tSZ provenance NSIDE does not equal {nside}: {provenance_path}")
    total = int(values.get("catalog_total_rows", -1))
    scanned = int(values.get("catalog_rows_scanned", -2))
    selected = int(values.get("halos_selected", 0))
    if total <= 0 or scanned != total or selected <= 0:
        raise ValueError(
            "tSZ provenance does not prove a complete catalogue scan: "
            f"total={total}, scanned={scanned}, selected={selected}."
        )
    selected_zmin = float(values.get("selected_redshift_min", "nan"))
    selected_zmax = float(values.get("selected_redshift_max", "nan"))
    if not np.isfinite(selected_zmax) or selected_zmax <= EXPECTED_Z_SOURCE:
        raise ValueError(
            "The complete-lightcone tSZ map must contain halos beyond the FRB "
            f"source plane z=1; selected_redshift_max={selected_zmax}."
        )
    cache_zmin = float(values.get("profile_redshift_cache_min", "nan"))
    cache_zmax = float(values.get("profile_redshift_cache_max", "nan"))
    if (
        not np.isfinite(selected_zmin)
        or not np.isfinite(cache_zmin)
        or not np.isfinite(cache_zmax)
        or selected_zmin < cache_zmin
        or selected_zmax > cache_zmax
    ):
        raise ValueError(
            "The selected tSZ halo redshifts are not contained in the profile cache: "
            f"selected=[{selected_zmin}, {selected_zmax}], "
            f"cache=[{cache_zmin}, {cache_zmax}]."
        )
    aperture = float(values.get("aperture_r200c_multiplier", "nan"))
    if not np.isfinite(aperture) or aperture <= 0:
        raise ValueError(f"Invalid tSZ R200c aperture in {provenance_path}: {aperture}")
    recorded_map = Path(values.get("output_map", "")).expanduser().resolve()
    if recorded_map != map_path:
        raise ValueError(
            f"tSZ provenance belongs to {recorded_map}, not requested map {map_path}."
        )
    return values


def parse_frb_spec(values: list[str]) -> FRBSpec:
    label, profile, aperture_text, path_text = values
    aperture = float(aperture_text)
    if not np.isfinite(aperture) or aperture <= 0:
        raise ValueError(f"Invalid aperture for {label}: {aperture_text}")
    return FRBSpec(label, profile.lower(), aperture, Path(path_text).expanduser().resolve())


def extract_dm_window(dataset: h5py.Dataset, index: int, nwindow: int) -> np.ndarray:
    if dataset.ndim != 2:
        raise ValueError(f"dm_pc_cm3 must be 2D, found {dataset.shape}.")
    if dataset.shape[0] == nwindow:
        return np.asarray(dataset[index, :], dtype=np.float64)
    if dataset.shape[1] == nwindow:
        return np.asarray(dataset[:, index], dtype=np.float64)
    raise ValueError(f"Cannot identify the window axis in dm_pc_cm3={dataset.shape}.")


def load_frb(spec: FRBSpec, nside: int) -> FRBData:
    if not spec.path.is_file():
        raise FileNotFoundError(f"Missing FRB ray HDF5: {spec.path}")
    with h5py.File(spec.path, "r") as handle:
        required = {"dm_pc_cm3", "frb_pixel_ring_1based", "window_label", "source_redshift_grid"}
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"{spec.path} is missing: {', '.join(missing)}")
        attrs = {key: decode(value) for key, value in handle.attrs.items()}
        labels = [str(decode(value)) for value in handle["window_label"][:]]
        if "all" not in labels:
            raise ValueError(f"{spec.path} has no all-mass window.")
        dm = extract_dm_window(handle["dm_pc_cm3"], labels.index("all"), len(labels))
        pixels = np.asarray(handle["frb_pixel_ring_1based"][:], dtype=np.int64) - 1
        z_source = np.asarray(handle["source_redshift_grid"][:], dtype=float)

    if z_source.size != 1 or not np.isclose(z_source[0], EXPECTED_Z_SOURCE, atol=1e-12, rtol=0):
        raise ValueError(f"{spec.path} is not fixed at z_source=1: {z_source}")
    if len(dm) != EXPECTED_NFRB or len(pixels) != EXPECTED_NFRB:
        raise ValueError(
            f"{spec.path} has {len(dm)} DM values and {len(pixels)} pixels; "
            f"both must equal {EXPECTED_NFRB}."
        )
    if not np.all(np.isfinite(dm)) or np.any(dm < 0):
        raise ValueError(f"{spec.path} has non-finite or negative DM.")
    npix = hp.nside2npix(nside)
    if np.any(pixels < 0) or np.any(pixels >= npix):
        raise ValueError(f"{spec.path} has pixels outside NSIDE={nside}.")
    if np.unique(pixels).size != pixels.size:
        raise ValueError(f"{spec.path} does not contain unique ray pixels.")

    nside_attr = attrs.get("provenance_nside")
    if nside_attr is not None and int(nside_attr) != nside:
        raise ValueError(f"{spec.path} has NSIDE={nside_attr}, expected {nside}.")
    aperture = attrs.get(
        "provenance_halo_extension_r200_multiplier",
        attrs.get("provenance_dm_aperture_r200_multiplier"),
    )
    if aperture is None or not np.isclose(float(aperture), spec.aperture_r200c, atol=1e-12, rtol=0):
        raise ValueError(f"{spec.path} aperture={aperture}; expected {spec.aperture_r200c} R200c.")
    expected_attrs = {
        "halo_mass_definition": "M200c",
        "aperture_radius_definition": "R200c",
        "healpix_ordering": "RING",
    }
    for name, expected in expected_attrs.items():
        if attrs.get(name) != expected:
            raise ValueError(f"{spec.path} has {name}={attrs.get(name)!r}; expected {expected!r}.")
    if not bool(attrs.get("per_ray_dm_saved", False)):
        raise ValueError(f"{spec.path} does not contain per-ray DM.")
    return FRBData(spec, pixels, dm, attrs)


def validate_inputs(inputs: list[FRBData]) -> None:
    if not inputs:
        raise ValueError("At least one --frb-input is required.")
    labels = [item.spec.label for item in inputs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"FRB labels must be unique: {labels}")
    for item in inputs[1:]:
        if not np.array_equal(inputs[0].pixels, item.pixels):
            raise ValueError(
                f"Ray pixels differ between {inputs[0].spec.label} and {item.spec.label}. "
                "Use the same NSIDE, N, and seed."
            )


def validate_common_catalog(inputs: list[FRBData], tsz_provenance: dict[str, str]) -> None:
    tsz_catalog_text = tsz_provenance.get("catalogue")
    if not tsz_catalog_text:
        raise ValueError("tSZ provenance has no catalogue path.")
    tsz_catalog = Path(tsz_catalog_text).expanduser().resolve()
    for data in inputs:
        frb_catalog_text = data.attrs.get("provenance_catalog_path")
        if not frb_catalog_text:
            raise ValueError(f"{data.spec.path} has no provenance_catalog_path attribute.")
        frb_catalog = Path(str(frb_catalog_text)).expanduser().resolve()
        if frb_catalog != tsz_catalog:
            raise ValueError(
                f"Catalogue mismatch: tSZ uses {tsz_catalog}, while "
                f"{data.spec.label} uses {frb_catalog}."
            )


def load_tsz(path: Path, nside: int) -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing tSZ map: {path}")
    y_map = np.asarray(
        hp.read_map(path, field=0, nest=False, dtype=np.float64),
        dtype=np.float64,
    )
    if hp.get_nside(y_map) != nside:
        raise ValueError(f"tSZ map NSIDE={hp.get_nside(y_map)}, expected {nside}.")
    if not np.all(np.isfinite(y_map)) or np.any(y_map < 0):
        raise ValueError(
            "tSZ input must be a finite, non-negative, complete Compton-y map, "
            "not a mask or a mean-subtracted temperature map."
        )
    mean_y = float(np.mean(y_map))
    y_map -= mean_y
    return y_map, mean_y


def sparse_dm_map(data: FRBData, nside: int) -> tuple[np.ndarray, np.ndarray, float]:
    npix = hp.nside2npix(nside)
    q = data.dm - np.mean(data.dm)
    estimator = np.zeros(npix, dtype=np.float32)
    estimator[data.pixels] = np.asarray(q * (npix / q.size), dtype=np.float32)
    nbar_sr = q.size / (4 * np.pi)
    shot_cl = float(np.mean(q * q) / nbar_sr)
    return estimator, q, shot_cl


def log_edges(lmin: int, lmax: int, requested_bins: int) -> np.ndarray:
    edges = np.unique(
        np.rint(np.exp(np.linspace(np.log(lmin), np.log(lmax + 1), requested_bins + 1))).astype(int)
    )
    if edges[0] > lmin:
        edges = np.insert(edges, 0, lmin)
    if edges[-1] <= lmax:
        edges = np.append(edges, lmax + 1)
    return edges


def arithmetic_bands(
    ell: np.ndarray,
    arrays: dict[str, np.ndarray],
    lmin: int,
    requested_bins: int,
) -> dict[str, np.ndarray]:
    output: dict[str, list[float]] = {
        "ell_min": [],
        "ell_max": [],
        "ell_effective": [],
        **{name: [] for name in arrays},
    }
    edges = log_edges(lmin, int(ell[-1]), requested_bins)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (ell >= lower) & (ell < upper)
        if not np.any(selected):
            continue
        weights = 2 * ell[selected].astype(float) + 1
        output["ell_min"].append(float(lower))
        output["ell_max"].append(float(upper - 1))
        output["ell_effective"].append(float(np.average(ell[selected], weights=weights)))
        for name, values in arrays.items():
            output[name].append(float(np.average(values[selected], weights=weights)))
    return {name: np.asarray(values) for name, values in output.items()}


def so_bands(ell: np.ndarray, cl_yy: np.ndarray) -> dict[str, np.ndarray]:
    output = {name: [] for name in ("ell_min", "ell_max", "ell_effective", "cl_yy", "dl_yy")}
    for index, (lower, next_edge) in enumerate(zip(SO_B40_EDGES[:-1], SO_B40_EDGES[1:])):
        upper = next_edge if index == len(SO_B40_EDGES) - 2 else next_edge - 1
        selected = (ell >= lower) & (ell <= upper)
        if not np.any(selected):
            raise ValueError(f"No multipoles in SO bin [{lower}, {upper}].")
        weights = 2 * ell[selected].astype(float) + 1
        dl = ell[selected] * (ell[selected] + 1) * cl_yy[selected] / (2 * np.pi)
        if np.any(~np.isfinite(dl)) or np.any(dl <= 0):
            raise ValueError(f"Invalid tSZ spectrum in SO bin [{lower}, {upper}].")
        output["ell_min"].append(float(lower))
        output["ell_max"].append(float(upper))
        output["ell_effective"].append(float(np.average(ell[selected], weights=weights)))
        output["cl_yy"].append(float(np.average(cl_yy[selected], weights=weights)))
        # Exact existing SO-emulator convention: weighted geometric mean of D_ell.
        output["dl_yy"].append(float(10 ** np.average(np.log10(dl), weights=weights)))
    return {name: np.asarray(values) for name, values in output.items()}


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    names = list(columns)
    if len({len(columns[name]) for name in names}) != 1:
        raise ValueError(f"Column lengths disagree for {path}.")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))


def percent(other: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.full_like(reference, np.nan, dtype=float)
    scale = float(np.nanmax(np.abs(reference)))
    valid = np.isfinite(other) & np.isfinite(reference) & (np.abs(reference) > 1e-8 * scale)
    result[valid] = 100 * (other[valid] - reference[valid]) / reference[valid]
    return result


DISPLAY_NAMES = {
    "battaglia16_r3": r"Battaglia16 DM, $3R_{200c}$",
    "battaglia16_r5": r"Battaglia16 DM, $5R_{200c}$",
    "lee22_r3": r"Lee22 DM, $3R_{200c}$",
    "lee22_r5": r"Lee22 DM, $5R_{200c}$",
}
COLORS = {
    "battaglia16_r3": "mediumblue",
    "battaglia16_r5": "cornflowerblue",
    "lee22_r3": "darkorange",
    "lee22_r5": "goldenrod",
}


def style_for(label: str) -> dict[str, object]:
    return {
        "label": DISPLAY_NAMES.get(label, label),
        "color": COLORS.get(label),
        "linestyle": "--" if label.endswith("_r5") else "-",
        "linewidth": 2.2,
    }


def style_axes(ax, ylabel: str, log_y: bool = True) -> None:
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.22)
    ax.tick_params(direction="in", which="both", top=True, right=True)


def save_plots(
    output_dir: Path,
    so: dict[str, np.ndarray],
    bands: dict[str, np.ndarray],
    labels: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    ax.plot(so["ell_effective"], so["dl_yy"], color="firebrick", linewidth=2.6)
    style_axes(ax, r"$D_\ell^{yy}$")
    ax.set_title("Battaglia12 tSZ auto-spectrum: SO 40-bin convention")
    fig.savefig(output_dir / "tsz_auto_so_binned40.png", dpi=180)
    plt.close(fig)

    ell = bands["ell_effective"]
    prefactor = ell * (ell + 1) / (2 * np.pi)
    fig, ax = plt.subplots(figsize=(10.5, 6.7), constrained_layout=True)
    for label in labels:
        dl = prefactor * bands[f"cl_dm_corrected_{label}"]
        keep = np.isfinite(dl) & (dl > 0)
        ax.plot(ell[keep], dl[keep], **style_for(label))
    style_axes(ax, r"Shot-noise-subtracted $D_\ell^{\rm DM\,DM}$  [$(\mathrm{pc\,cm^{-3}})^2$]")
    ax.set_title(r"FRB halo-DM auto-spectra: $10^6$ uniform rays at $z_s=1$")
    ax.legend(fontsize=10, ncol=2)
    fig.savefig(output_dir / "frb_dm_auto_log_binned.png", dpi=180)
    plt.close(fig)

    cross_dl = {
        label: prefactor * bands[f"cl_y_x_dm_{label}"]
        for label in labels
    }
    all_positive = all(np.all(values[np.isfinite(values)] > 0) for values in cross_dl.values())
    fig, ax = plt.subplots(figsize=(10.5, 6.7), constrained_layout=True)
    for label, values in cross_dl.items():
        ax.plot(ell, values, **style_for(label))
    style_axes(
        ax,
        r"$D_\ell^{y\times\rm DM}$  [$\mathrm{pc\,cm^{-3}}$]",
        log_y=all_positive,
    )
    if not all_positive:
        finite_abs = np.concatenate([np.abs(value[np.isfinite(value)]) for value in cross_dl.values()])
        ax.set_yscale("symlog", linthresh=max(1e-20, float(np.max(finite_abs)) * 1e-4))
    ax.set_title(r"Battaglia12 $y\times$ FRB halo-DM cross-spectra")
    ax.legend(fontsize=10, ncol=2)
    fig.savefig(output_dir / "tsz_x_frb_dm_cross_log_binned.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True, constrained_layout=True)
    for profile, color in (("battaglia16", "mediumblue"), ("lee22", "darkorange")):
        label3, label5 = f"{profile}_r3", f"{profile}_r5"
        if label3 not in labels or label5 not in labels:
            continue
        axes[0].plot(
            ell,
            percent(bands[f"cl_dm_corrected_{label5}"], bands[f"cl_dm_corrected_{label3}"]),
            color=color,
            linewidth=2,
            label="Battaglia16" if profile == "battaglia16" else "Lee22",
        )
        axes[1].plot(
            ell,
            percent(bands[f"cl_y_x_dm_{label5}"], bands[f"cl_y_x_dm_{label3}"]),
            color=color,
            linewidth=2,
            label="Battaglia16" if profile == "battaglia16" else "Lee22",
        )
    axes[0].set_ylabel(r"DM auto: $(5R-3R)/(3R)$  [%]")
    axes[1].set_ylabel(r"$y\times$DM: $(5R-3R)/(3R)$  [%]")
    axes[1].set_xlabel(r"Multipole $\ell$")
    for ax in axes:
        ax.set_xscale("log")
        ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
        ax.grid(True, which="both", alpha=0.22)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend()
    fig.suptitle(r"Projected halo aperture: $3R_{200c}$ versus $5R_{200c}$")
    fig.savefig(output_dir / "r200c_aperture_percent_differences.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
    for label in labels:
        correlation = bands[f"r_y_dm_{label}"]
        keep = np.isfinite(correlation)
        ax.plot(ell[keep], correlation[keep], **style_for(label))
    ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
    ax.axhline(1, color="0.6", linestyle=":", linewidth=1)
    ax.axhline(-1, color="0.6", linestyle=":", linewidth=1)
    style_axes(ax, r"$r_\ell^{y,\rm DM}$", log_y=False)
    ax.set_title("Scale-dependent tSZ-FRB correlation coefficient")
    ax.legend(fontsize=10, ncol=2)
    fig.savefig(output_dir / "tsz_frb_dm_correlation_coefficient.png", dpi=180)
    plt.close(fig)


def load_class_sz_reference(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load the explicitly unit-labelled CLASS-SZ Battaglia12 reference."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing CLASS-SZ tSZ reference: {path}")
    with np.load(path, allow_pickle=False) as source:
        required = {"ell", "dl_yy_1h", "dl_yy_2h", "dl_yy_total", "metadata_json"}
        missing = required.difference(source.files)
        if missing:
            raise ValueError(f"CLASS-SZ reference is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(source[name]) for name in required if name != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"].item()))

    ell = np.asarray(arrays["ell"], dtype=float)
    if ell.ndim != 1 or ell.size < 2 or not np.all(np.diff(ell) > 0):
        raise ValueError("CLASS-SZ reference needs increasing one-dimensional multipoles")
    for name in ("dl_yy_1h", "dl_yy_2h", "dl_yy_total"):
        values = np.asarray(arrays[name], dtype=float)
        if values.shape != ell.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid CLASS-SZ reference array: {name}")
        arrays[name] = values
    if not np.allclose(
        arrays["dl_yy_total"],
        arrays["dl_yy_1h"] + arrays["dl_yy_2h"],
        rtol=1e-10,
        atol=0.0,
    ):
        raise ValueError("CLASS-SZ total is inconsistent with its 1h + 2h components")
    arrays["ell"] = ell
    if metadata.get("pressure_profile") != "Battaglia12 (CLASS-SZ B12)":
        raise ValueError("CLASS-SZ reference does not identify the Battaglia12 profile")
    return arrays, metadata


def save_three_panel_comparison(
    output_dir: Path,
    so: dict[str, np.ndarray],
    bands: dict[str, np.ndarray],
    class_sz: dict[str, np.ndarray],
    tsz_aperture_r200c: float,
    aperture_r200c: int = 3,
) -> None:
    """Save the compact tSZ, DM, and y x DM comparison requested for the paper."""
    battaglia = f"battaglia16_r{aperture_r200c}"
    lee = f"lee22_r{aperture_r200c}"
    for label in (battaglia, lee):
        for prefix in ("dl_dm_corrected_", "dl_y_x_dm_"):
            if prefix + label not in bands:
                raise KeyError(f"Missing three-panel input: {prefix + label}")

    ell = bands["ell_effective"]
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 6.2), constrained_layout=True)

    axes[0].plot(
        so["ell_effective"],
        1e12 * so["dl_yy"],
        color="firebrick",
        linewidth=2.6,
        label=rf"HalfDome B12, ${tsz_aperture_r200c:g}R_{{200c}}$",
    )
    axes[0].plot(
        class_sz["ell"],
        1e12 * class_sz["dl_yy_total"],
        color="black",
        linestyle="--",
        linewidth=2.3,
        label="CLASS-SZ B12, 1h+2h\n(native x_outSZ=4)",
    )
    axes[0].set_ylabel(r"$10^{12}D_\ell^{yy}$")
    axes[0].set_title("tSZ auto-spectrum")

    profile_styles = (
        (battaglia, "mediumblue", "Battaglia16"),
        (lee, "darkorange", "Lee22"),
    )
    for label, color, display in profile_styles:
        dm = bands[f"dl_dm_corrected_{label}"]
        positive = np.isfinite(dm) & (dm > 0)
        axes[1].plot(
            ell[positive],
            dm[positive],
            color=color,
            linewidth=2.4,
            label=display,
        )
    axes[1].set_ylabel(r"$D_\ell^{\rm DM\,DM}$ [$(\mathrm{pc\,cm^{-3}})^2$]")
    axes[1].set_title(rf"FRB halo-DM auto-spectrum, ${aperture_r200c}R_{{200c}}$")

    cross_values = []
    for label, color, display in profile_styles:
        cross = bands[f"dl_y_x_dm_{label}"]
        cross_values.append(cross)
        finite = np.isfinite(cross)
        axes[2].plot(
            ell[finite],
            cross[finite],
            color=color,
            linewidth=2.4,
            label=display,
        )
    axes[2].set_ylabel(r"$D_\ell^{y\times\rm DM}$ [$\mathrm{pc\,cm^{-3}}$]")
    axes[2].set_title(rf"tSZ $\times$ FRB halo-DM, ${aperture_r200c}R_{{200c}}$")

    all_cross_positive = all(
        np.all(values[np.isfinite(values)] > 0) for values in cross_values
    )
    if all_cross_positive:
        axes[2].set_yscale("log")
    else:
        finite_abs = np.concatenate(
            [np.abs(values[np.isfinite(values)]) for values in cross_values]
        )
        axes[2].set_yscale(
            "symlog",
            linthresh=max(1e-20, float(np.max(finite_abs)) * 1e-4),
        )

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(80, 8200)
        axis.set_xlabel(r"Multipole $\ell$")
        axis.grid(True, which="both", alpha=0.22)
        axis.tick_params(direction="in", which="both", top=True, right=True)
        axis.legend(frameon=False, fontsize=10, loc="best")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    figure_stem = f"tsz_frb_three_panel_class_sz_{aperture_r200c}r200c"
    fig.savefig(output_dir / f"{figure_stem}.png", dpi=220)
    fig.savefig(output_dir / f"{figure_stem}.pdf")
    plt.close(fig)


def validate_requested_cases(specs: list[FRBSpec]) -> None:
    expected = {
        ("battaglia16_r3", "battaglia16", 3.0),
        ("battaglia16_r5", "battaglia16", 5.0),
        ("lee22_r3", "lee2022", 3.0),
        ("lee22_r5", "lee2022", 5.0),
    }
    actual = {(item.label, item.profile, item.aperture_r200c) for item in specs}
    if actual != expected:
        raise ValueError(f"This fixed comparison requires {sorted(expected)}; received {sorted(actual)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tsz-map",
        type=Path,
        required=True,
        help="Full-sky RING, beamless/noiseless Compton-y FITS map.",
    )
    parser.add_argument(
        "--tsz-provenance",
        type=Path,
        required=True,
        help="Key-value provenance written beside the internally painted tSZ map.",
    )
    parser.add_argument(
        "--class-sz-spectrum",
        type=Path,
        help="Optional CLASS-SZ Battaglia12 reference created by the companion script.",
    )
    parser.add_argument(
        "--frb-input",
        nargs=4,
        action="append",
        metavar=("LABEL", "PROFILE", "APERTURE_R200C", "H5"),
        required=True,
        help="Repeat for each per-ray HDF5 product.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nside", type=int, default=EXPECTED_NSIDE)
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--niter", type=int, default=0)
    parser.add_argument("--log-bins", type=int, default=55)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.nside != EXPECTED_NSIDE:
        raise ValueError(f"This fixed comparison requires NSIDE={EXPECTED_NSIDE}.")
    if not int(SO_B40_EDGES[-1]) <= args.lmax <= 3 * args.nside - 1:
        raise ValueError(f"lmax must be in [{SO_B40_EDGES[-1]}, {3 * args.nside - 1}].")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = output_dir / "tsz_frb_1m_z1_r200c_aperture_spectra.h5"
    if output_h5.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_h5}. Pass --overwrite to replace it.")

    specs = [parse_frb_spec(values) for values in args.frb_input]
    validate_requested_cases(specs)
    inputs = [load_frb(spec, args.nside) for spec in specs]
    validate_inputs(inputs)
    labels = [item.spec.label for item in inputs]

    tsz_path = args.tsz_map.expanduser().resolve()
    tsz_provenance_path = args.tsz_provenance.expanduser().resolve()
    tsz_provenance = validate_tsz_provenance(tsz_provenance_path, tsz_path, args.nside)
    validate_common_catalog(inputs, tsz_provenance)
    class_sz = None
    class_sz_metadata = None
    class_sz_path = None
    if args.class_sz_spectrum is not None:
        class_sz_path = args.class_sz_spectrum.expanduser().resolve()
        class_sz, class_sz_metadata = load_class_sz_reference(class_sz_path)
        reference_provenance = Path(class_sz_metadata["tsz_map_provenance"]).resolve()
        if reference_provenance != tsz_provenance_path:
            raise ValueError(
                "CLASS-SZ reference was not generated from this tSZ provenance: "
                f"{reference_provenance} != {tsz_provenance_path}"
            )
        theory_parameters = class_sz_metadata.get("parameters", {})
        theory_h = float(theory_parameters.get("h", "nan"))
        expected_theory_bounds = {
            "M_min": float(tsz_provenance["selected_mass_min_msun"]) * theory_h,
            "M_max": float(tsz_provenance["selected_mass_max_msun"]) * theory_h,
            "z_min": float(tsz_provenance["selected_redshift_min"]),
            "z_max": float(tsz_provenance["selected_redshift_max"]),
        }
        for name, expected in expected_theory_bounds.items():
            found = float(theory_parameters.get(name, "nan"))
            if not np.isclose(found, expected, rtol=1e-10, atol=0.0):
                raise ValueError(
                    f"Stale CLASS-SZ reference: {name}={found}, expected {expected} "
                    "from the current tSZ provenance"
                )
    print(f"Loading and centering tSZ map: {tsz_path}", flush=True)
    y_map, mean_y = load_tsz(tsz_path, args.nside)
    print(f"Computing tSZ alm: lmax={args.lmax}, niter={args.niter}", flush=True)
    y_alm = hp.map2alm(
        y_map,
        lmax=args.lmax,
        iter=args.niter,
        pol=False,
        use_pixel_weights=False,
    )
    del y_map

    ell = np.arange(args.lmax + 1)
    cl_yy = np.asarray(hp.alm2cl(y_alm, lmax=args.lmax), dtype=float)
    spectra = {"cl_yy": cl_yy}
    diagnostics: dict[str, dict[str, float]] = {}
    for data in inputs:
        label = data.spec.label
        print(f"Computing sparse DM alm and cross-spectrum: {label}", flush=True)
        dm_map, q, shot_cl = sparse_dm_map(data, args.nside)
        dm_alm = hp.map2alm(
            dm_map,
            lmax=args.lmax,
            iter=args.niter,
            pol=False,
            use_pixel_weights=False,
        )
        del dm_map
        observed = np.asarray(hp.alm2cl(dm_alm, lmax=args.lmax), dtype=float)
        cross = np.asarray(hp.alm2cl(y_alm, dm_alm, lmax=args.lmax), dtype=float)
        del dm_alm
        spectra[f"cl_dm_observed_{label}"] = observed
        spectra[f"cl_dm_shot_{label}"] = np.full_like(observed, shot_cl)
        spectra[f"cl_dm_corrected_{label}"] = observed - shot_cl
        spectra[f"cl_y_x_dm_{label}"] = cross
        diagnostics[label] = {
            "dm_mean_pc_cm3": float(np.mean(data.dm)),
            "dm_std_pc_cm3": float(np.std(data.dm)),
            "dm_min_pc_cm3": float(np.min(data.dm)),
            "dm_max_pc_cm3": float(np.max(data.dm)),
            "nbar_per_sr": float(q.size / (4 * np.pi)),
            "analytic_shot_cl_pc2_cm6": shot_cl,
        }
        del q
    del y_alm

    so = so_bands(ell, cl_yy)
    yy_log = arithmetic_bands(ell, {"cl_yy": cl_yy}, 2, args.log_bins)["cl_yy"]
    bands = arithmetic_bands(
        ell,
        {name: values for name, values in spectra.items() if name != "cl_yy"},
        2,
        args.log_bins,
    )
    for label in labels:
        denominator = np.sqrt(np.maximum(yy_log * bands[f"cl_dm_corrected_{label}"], 0))
        correlation = np.full_like(denominator, np.nan)
        valid = denominator > 0
        correlation[valid] = bands[f"cl_y_x_dm_{label}"][valid] / denominator[valid]
        bands[f"r_y_dm_{label}"] = correlation
        prefactor = bands["ell_effective"] * (bands["ell_effective"] + 1) / (2 * np.pi)
        bands[f"dl_dm_corrected_{label}"] = (
            prefactor * bands[f"cl_dm_corrected_{label}"]
        )
        bands[f"dl_y_x_dm_{label}"] = prefactor * bands[f"cl_y_x_dm_{label}"]
    for profile in ("battaglia16", "lee22"):
        bands[f"dm_auto_5r_minus_3r_percent_{profile}"] = percent(
            bands[f"cl_dm_corrected_{profile}_r5"],
            bands[f"cl_dm_corrected_{profile}_r3"],
        )
        bands[f"y_x_dm_5r_minus_3r_percent_{profile}"] = percent(
            bands[f"cl_y_x_dm_{profile}_r5"],
            bands[f"cl_y_x_dm_{profile}_r3"],
        )

    with h5py.File(output_h5, "w") as handle:
        unbinned = handle.create_group("unbinned")
        unbinned["ell"] = ell
        for name, values in spectra.items():
            unbinned[name] = values
        so_group = handle.create_group("tsz_so_binned40")
        for name, values in so.items():
            so_group[name] = values
        log_group = handle.create_group("dm_cross_log_binned")
        for name, values in bands.items():
            log_group[name] = values
        if class_sz is not None:
            class_group = handle.create_group("class_sz_tsz_reference")
            for name, values in class_sz.items():
                class_group[name] = values
            class_group.attrs["source_path"] = str(class_sz_path)
            class_group.attrs["metadata_json"] = json.dumps(class_sz_metadata)
        handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["tsz_map"] = str(tsz_path)
        handle.attrs["tsz_provenance"] = str(tsz_provenance_path)
        handle.attrs["tsz_map_mean_removed"] = mean_y
        handle.attrs["tsz_contract"] = (
            "internally painted Battaglia12; all resolved halos; complete lightcone; "
            "physical M200c; external R200c aperture; no beam/noise/mask"
        )
        handle.attrs["tsz_contract_validation"] = (
            "map numerical checks plus strict generated-provenance validation"
        )
        handle.attrs["tsz_aperture_r200c"] = float(
            tsz_provenance["aperture_r200c_multiplier"]
        )
        handle.attrs["tsz_selected_halo_count"] = int(tsz_provenance["halos_selected"])
        handle.attrs["tsz_selected_redshift_max"] = float(
            tsz_provenance["selected_redshift_max"]
        )
        handle.attrs["frb_source_redshift"] = EXPECTED_Z_SOURCE
        handle.attrs["frb_ray_count"] = EXPECTED_NFRB
        handle.attrs["frb_sightlines"] = "same unique uniform pixels in every case"
        handle.attrs["frb_foreground"] = "all resolved M200c halos, 0 <= z_halo <= 1"
        handle.attrs["frb_auto_shot_noise"] = "analytic <(DM-mean DM)^2>/nbar subtracted"
        handle.attrs["cross_shot_noise_subtraction"] = "none (no additive cross bias)"
        handle.attrs["nside"] = args.nside
        handle.attrs["lmax"] = args.lmax
        handle.attrs["niter"] = args.niter
        handle.attrs["beam_applied"] = False
        handle.attrs["instrumental_noise_added"] = False
        handle.attrs["pixel_window_deconvolved"] = False
        handle.attrs["tsz_binning"] = (
            "SO B40: edges 80:200:7880 plus 7979; (2ell+1)-weighted geometric D_ell"
        )
        handle.attrs["dm_cross_binning"] = (
            f"{args.log_bins} requested log ell bins; (2ell+1)-weighted arithmetic C_ell"
        )
        for data in inputs:
            group = handle.create_group(f"frb_inputs/{data.spec.label}")
            group.attrs["path"] = str(data.spec.path)
            group.attrs["profile"] = data.spec.profile
            group.attrs["aperture_r200c"] = data.spec.aperture_r200c
            for source_name in (
                "provenance_catalog_path",
                "provenance_dm_cache_file",
                "provenance_dm_model_family",
                "provenance_foreground_valid_halo_count",
                "provenance_catalog_streamed_halos",
            ):
                if source_name in data.attrs:
                    group.attrs[source_name] = data.attrs[source_name]
            for name, value in diagnostics[data.spec.label].items():
                group.attrs[name] = value

    write_csv(output_dir / "tsz_auto_so_binned40.csv", so)
    write_csv(output_dir / "frb_dm_and_tsz_cross_log_binned.csv", bands)
    save_plots(output_dir, so, bands, labels)
    if class_sz is not None:
        save_three_panel_comparison(
            output_dir,
            so,
            bands,
            class_sz,
            tsz_aperture_r200c=float(tsz_provenance["aperture_r200c_multiplier"]),
            aperture_r200c=3,
        )

    provenance = {
        "result_hdf5": str(output_h5),
        "tsz_map": str(tsz_path),
        "tsz_provenance": str(tsz_provenance_path),
        "tsz_generation": tsz_provenance,
        "tsz_mean_removed": mean_y,
        "nside": args.nside,
        "lmax": args.lmax,
        "niter": args.niter,
        "beam": "none",
        "instrumental_noise": "none",
        "pixel_window_deconvolution": False,
        "class_sz_tsz_reference": None if class_sz_path is None else str(class_sz_path),
        "class_sz_metadata": class_sz_metadata,
        "frb_inputs": [
            {
                "label": data.spec.label,
                "profile": data.spec.profile,
                "aperture_r200c": data.spec.aperture_r200c,
                "path": str(data.spec.path),
                **diagnostics[data.spec.label],
            }
            for data in inputs
        ],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved spectra and plots under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
