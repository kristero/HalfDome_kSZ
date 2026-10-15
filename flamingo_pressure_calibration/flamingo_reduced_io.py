"""Plumbing for FLAMINGO *reduced* snapshots streamed from the COSMA data service.

Reduced snapshots (https://dataweb.cosma.dur.ac.uk:8443/flamingo/snapshots/snapshot_partial.html)
contain every particle inside R100c of a subset of haloes: in each 0.05-dex bin of
M200c above 1e13 Msun, all haloes if the bin holds fewer than 200, otherwise a random
200.  SOAP flags them in ``SOAP/IncludedInReducedSnapshot``.  Everything in this file
is data access and caching; the science (profiles, fits, stacks) lives in the notebook.

All returned quantities are *physical* (Mpc, Msun, Pa, K, m^-3).  Conversion factors
are taken from the file attributes, never from assumed little-h powers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.constants import Boltzmann, c as C_LIGHT, electron_mass, physical_constants

MPC_CM = 3.08567758e24
MSUN_G = 1.98841e33
MPC_M = 3.085677581491367e22
SIGMA_T = physical_constants["Thomson cross section"][0]
Y_TO_PRESSURE_VOLUME = electron_mass * C_LIGHT**2 / SIGMA_T   # Pa m^3 per Mpc^2-free y: Pe*V = y * me c^2 / sigma_T

# Verified directory layout on the service (2026-09-25).
SIMULATIONS = {
    "L2p8_m9": "FLAMINGO/L2p8_m9/L2p8_m9",   # 2.8 Gpc fiducial hydro, snapshots 0000-0078 (z=0 at 0078)
    "L1_m9": "FLAMINGO/L1_m9/L1_m9",         # 1 Gpc fiducial hydro (no reduced snapshots on the service)
}

# Spherical-overdensity apertures kept from SOAP.  SOAP was run on the *full* snapshot, so
# ComptonY in the 50_crit and 5xR_500_crit apertures is valid even though the reduced
# snapshot only holds particles inside R100c.
APERTURES = ["2500_crit", "1000_crit", "500_crit", "200_crit", "100_crit", "50_crit", "5xR_500_crit"]

# name -> (SOAP dataset, divisor applied after the physical-CGS conversion; None = raw integers)
CATALOGUE_COLUMNS = {
    "halo_index": ("InputHalos/HaloCatalogueIndex", None),
    "is_central": ("InputHalos/IsCentral", None),
    "n_bound": ("InputHalos/NumberOfBoundParticles", None),
    "host_index": ("SOAP/HostHaloIndex", None),
    "centre_mpc": ("InputHalos/HaloCentre", MPC_CM),
    "m200c_msun": ("SO/200_crit/TotalMass", MSUN_G),
    "m500c_msun": ("SO/500_crit/TotalMass", MSUN_G),
    "m100c_msun": ("SO/100_crit/TotalMass", MSUN_G),
    "gas_mass_200c_msun": ("SO/200_crit/GasMass", MSUN_G),
    "hot_gas_mass_200c_msun": ("SO/200_crit/HotGasMass", MSUN_G),
    "stellar_mass_200c_msun": ("SO/200_crit/StellarMass", MSUN_G),
    "gas_temperature_200c_k": ("SO/200_crit/GasTemperature", 1.0),
    "dm_concentration_200c": ("SO/200_crit/DarkMatterConcentration", 1.0),
}
for _ap in APERTURES:
    CATALOGUE_COLUMNS[f"r_{_ap}_mpc"] = (f"SO/{_ap}/SORadius", MPC_CM)
    CATALOGUE_COLUMNS[f"y_{_ap}_mpc2"] = (f"SO/{_ap}/ComptonY", MPC_CM**2)
    CATALOGUE_COLUMNS[f"y_noagn_{_ap}_mpc2"] = (f"SO/{_ap}/ComptonYWithoutRecentAGNHeating", MPC_CM**2)
    CATALOGUE_COLUMNS[f"ngas_{_ap}"] = (f"SO/{_ap}/NumberOfGasParticles", None)


# --------------------------------------------------------------------------- service
def connect(user=None, password=None):
    """Open the COSMA hdfstream service.  Anonymous access works for FLAMINGO."""
    import hdfstream
    return hdfstream.open("cosma", "/", user=user, password=password)


_DOWNLOADED = {"bytes": 0, "limit": None}


def install_byte_counter(limit_bytes=None):
    """Count HTTP payload bytes in this process (and stop above ``limit_bytes``); returns the counter dict."""
    import requests
    _DOWNLOADED["limit"] = limit_bytes
    if getattr(requests.sessions.Session, "_fio_counted", False):
        return _DOWNLOADED
    original_send = requests.sessions.Session.send

    def send(session, request, **kwargs):
        kwargs.setdefault("timeout", (30, 900))
        wanted_stream = kwargs.get("stream", False)
        kwargs["stream"] = True
        response = original_send(session, request, **kwargs)
        raw_read = response.raw.read

        def read(amt=None, *args, **read_kwargs):
            body = raw_read(amt, *args, **read_kwargs)
            if body:
                _DOWNLOADED["bytes"] += len(body)
                if _DOWNLOADED["limit"] and _DOWNLOADED["bytes"] > _DOWNLOADED["limit"]:
                    raise RuntimeError(f"download cap of {_DOWNLOADED['limit'] / 1e9:.1f} GB reached")
            return body

        response.raw.read = read
        if not wanted_stream:
            response.content
        return response

    requests.sessions.Session.send = send
    requests.sessions.Session._fio_counted = True
    return _DOWNLOADED


def retry(function, *args, attempts=5, what="request", **kwargs):
    """Retry transient service failures with backoff (never retries a reached download cap)."""
    for attempt in range(attempts):
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "download cap" in str(exc) or attempt == attempts - 1:
                raise
            delay = min(120, 10 * 2**attempt)
            print(f"  retry {attempt + 1}/{attempts - 1} for {what} in {delay} s: {exc!r}"[:300], flush=True)
            time.sleep(delay)


def soap_path(base, snapshot):
    return f"{base}/SOAP-HBT/halo_properties_{snapshot:04d}.hdf5"


def reduced_snapshot_path(base, snapshot):
    return f"{base}/snapshots_reduced/flamingo_{snapshot:04d}.hdf5"


def scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def physical_factor(dataset):
    return scalar(dataset.attrs["Conversion factor to physical CGS (including cosmological corrections)"])


def cosmology(soap):
    """Cosmology attributes of a SOAP file.  Omega_m in SWIFT excludes massive neutrinos."""
    attrs = dict(soap["Cosmology"].attrs)
    omega_m_total = scalar(attrs["Omega_m"]) + scalar(attrs.get("Omega_nu_0", [0.0]))
    return {
        "redshift": scalar(attrs["Redshift"]),
        "a": scalar(attrs["Scale-factor"]),
        "h": scalar(attrs["h"]),
        "Omega_b": scalar(attrs["Omega_b"]),
        "Omega_m_total": omega_m_total,
        "baryon_fraction": scalar(attrs["Omega_b"]) / omega_m_total,
        "H0_km_s_mpc": scalar(attrs["H [internal units]"]),
    }


def recently_heated_filter(soap):
    """SOAP's definition of 'recently AGN-heated' gas (used for the *WithoutRecentAGNHeating fields)."""
    attrs = dict(soap["Parameters/RecentlyHeatedGasFilter"].attrs)
    return {"a_limit": scalar(attrs["a_limit"]), "Tmin_K": scalar(attrs["Tmin_in_K"]),
            "Tmax_K": scalar(attrs["Tmax_in_K"]), "delta_time_Myr": scalar(attrs["delta_time_in_Myr"])}


def read_full_column(dataset, chunk=25_000_000, progress=True):
    """Read a whole 1-D SOAP column in chunks (hundreds of millions of rows for L2p8)."""
    n = dataset.shape[0]
    out = np.empty(n, dtype=dataset.dtype)
    starts = range(0, n, chunk)
    if progress:
        try:
            from tqdm.auto import tqdm
            starts = tqdm(starts, desc=f"reading {dataset.name.split('/')[-1]}", unit="chunk")
        except ImportError:
            pass
    for start in starts:
        stop = min(start + chunk, n)
        out[start:stop] = dataset[start:stop]
    return out


def read_rows(dataset, rows, divisor=None):
    """Fancy-index a SOAP column at sorted rows (one multi-slice request per 16M rows)."""
    values = np.asarray(retry(lambda: dataset[rows], what=dataset.name))
    if divisor is None:
        return values
    return values.astype(np.float64) * (physical_factor(dataset) / divisor)


# --------------------------------------------------------------------------- catalogue
def flagged_rows(soap, snap, progress=True):
    """Rows flagged IncludedInReducedSnapshot, found through the shared 64^3 cell index.

    SOAP rows are sorted by the snapshot's top-level cells, and a flagged halo's centre lies
    in a cell that holds some of its gas in the reduced snapshot. Scanning the flag only in
    cells with gas reads ~6% of the catalogue (~90 MB instead of 1.6 GB; 9151/9151 at z=0).
    Falls back to the full column if the cell index is missing.
    """
    flag = soap["SOAP/IncludedInReducedSnapshot"]
    try:
        gas_counts = np.asarray(snap["Cells/Counts/PartType0"][...])
        sub_counts = np.asarray(soap["Cells/Counts/Subhalos"][...])
        sub_offsets = np.asarray(soap["Cells/OffsetsInFile/Subhalos"][...])
        same_grid = np.allclose(np.asarray(soap["Cells/Centres"][...]), np.asarray(snap["Cells/Centres"][...]))
    except Exception:  # noqa: BLE001
        same_grid = False
    if same_grid:
        selected = np.flatnonzero((gas_counts > 0) & (sub_counts > 0))
        slices = [np.s_[int(sub_offsets[i]):int(sub_offsets[i] + sub_counts[i])] for i in selected]
        values = np.asarray(retry(flag.request_slices, slices, what="IncludedInReducedSnapshot"))
        rows = np.concatenate([np.arange(s.start, s.stop) for s in slices])
        return np.sort(rows[values != 0]), {"method": "cell_index", "rows_scanned": int(len(values)), "rows_total": int(flag.shape[0])}
    values = read_full_column(flag, progress=progress)
    return np.flatnonzero(values), {"method": "full_column", "rows_scanned": int(len(values)), "rows_total": int(len(values))}


def load_reduced_haloes(remote, base, snapshot, cache_dir, rebuild=False, progress=True, log=None):
    """All haloes flagged IncludedInReducedSnapshot, with the SOAP columns above.

    The flagged rows are located through the cell index (~90 MB, see ``flagged_rows``), the
    SOAP columns are read at those rows (15-40 s each), and the result is cached as
    ``<cache_dir>/<sim>_<snapshot>_reduced_haloes.npz`` plus a JSON receipt.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{base.split('/')[-1]}_{snapshot:04d}"
    npz = cache_dir / f"{tag}_reduced_haloes.npz"
    receipt = npz.with_suffix(".json")
    if npz.exists() and receipt.exists() and not rebuild:
        data = dict(np.load(npz))
        data["meta"] = json.loads(receipt.read_text())
        return data
    started = time.monotonic()
    soap = remote[soap_path(base, snapshot)]
    cosmo = cosmology(soap)
    rows, lookup = flagged_rows(soap, remote[reduced_snapshot_path(base, snapshot)], progress=progress)
    data = {"row": rows}
    if log:
        log(f"snapshot {snapshot}: {len(rows)} flagged haloes via {lookup['method']}; reading {len(CATALOGUE_COLUMNS)} SOAP columns")
    for number, (name, (path, divisor)) in enumerate(CATALOGUE_COLUMNS.items(), 1):
        column_start = time.monotonic()
        try:
            data[name] = read_rows(soap[path], rows, divisor)
        except KeyError:
            print(f"  (column {path} missing in this SOAP file; skipped)")
        if log:
            log(f"snapshot {snapshot}: column {number}/{len(CATALOGUE_COLUMNS)} {path} in {time.monotonic() - column_start:.1f} s")
    data["is_central"] = data["is_central"].astype(bool)
    meta = {"simulation_base": base, "snapshot": snapshot, "soap": soap_path(base, snapshot),
            "reduced_snapshot": reduced_snapshot_path(base, snapshot), "cosmology": cosmo,
            "recently_heated_filter": recently_heated_filter(soap),
            "n_subhaloes_total": int(soap["SOAP/IncludedInReducedSnapshot"].shape[0]), "n_included": int(len(rows)),
            "flag_lookup": lookup,
            "n_included_central": int(data["is_central"].sum()),
            "built_seconds": time.monotonic() - started}
    np.savez_compressed(npz, **data)
    receipt.write_text(json.dumps(meta, indent=2) + "\n")
    data["meta"] = meta
    return data


def central_table(data):
    """Central haloes only, as a pandas DataFrame sorted by decreasing M200c."""
    import pandas as pd
    keep = data["is_central"]
    columns = {k: v[keep] for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim == 1}
    table = pd.DataFrame(columns)
    for axis, name in enumerate("xyz"):
        table[f"centre_{name}_mpc"] = data["centre_mpc"][keep, axis]
    table["log10_m200c"] = np.log10(table["m200c_msun"])
    return table.sort_values("m200c_msun", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- particles
class ReducedSnapshot:
    """One reduced snapshot opened through swiftsimio, with a reusable spatial-mask template."""

    def __init__(self, remote, base, snapshot, soap=None):
        import warnings
        import swiftsimio as sw
        self.base, self.snapshot = base, snapshot
        self.file = remote[reduced_snapshot_path(base, snapshot)]
        # reduced snapshots lack Cells/MinPositions; swiftsimio pads the region by 0.1 cell, which is fine
        warnings.filterwarnings("ignore", message="Snapshot does not contain Cells/MinPositions")
        self.template = sw.mask(self.file)
        self.a = float(self.template.metadata.a)
        self.redshift = float(self.template.metadata.redshift)
        self.box_mpc = np.asarray(self.template.metadata.boxsize.to_physical().to_value("Mpc"))
        soap = soap if soap is not None else remote[soap_path(base, snapshot)]
        self.heating_filter = recently_heated_filter(soap)

    def _mask(self, centre_mpc, half_size_mpc):
        import swiftsimio as sw
        import unyt
        centre = np.asarray(centre_mpc, dtype=float)
        region = sw.cosmo_array(np.stack([centre - half_size_mpc, centre + half_size_mpc], axis=1) / self.a,
                                unyt.Mpc, comoving=True, scale_factor=self.a, scale_exponent=1)
        # SWIFTMask's dynamic __getattr__ recurses inside copy.copy; copy the state directly.
        mask = object.__new__(type(self.template))
        mask.__dict__.update(self.template.__dict__)
        mask.constrain_spatial(region)
        return mask

    def cutout(self, centre_mpc, radius_mpc):
        """Gas inside a sphere of ``radius_mpc`` (physical) around ``centre_mpc`` (physical).

        Returns a dict of numpy arrays: relative coordinates ``xyz_mpc`` (periodic-wrapped),
        ``r_mpc``, particle Compton ``y_mpc2`` (physical Mpc^2), ``temperature_k``, electron
        density ``ne_m3``, ``mass_msun``, ``density_msun_mpc3``, ``hsml_mpc``, electron pressure
        ``pe_pa`` (= n_e k_B T), ``last_agn_a`` and the SOAP ``recently_heated`` flag.
        """
        import swiftsimio as sw
        mask = self._mask(centre_mpc, radius_mpc)
        ranges = np.asarray(mask.gas)
        n_cells = int(np.sum(ranges[:, 1] - ranges[:, 0])) if ranges.size else 0
        data = sw.load(self.file, mask=mask)
        gas = data.gas
        xyz = np.asarray(gas.coordinates.to_physical().to_value("Mpc")) - np.asarray(centre_mpc)
        xyz -= self.box_mpc * np.rint(xyz / self.box_mpc)
        r = np.linalg.norm(xyz, axis=1)
        inside = r <= radius_mpc
        temperature = np.asarray(gas.temperatures.to_value("K"))[inside]
        last = np.asarray(gas.last_agnfeedback_scale_factors)[inside]
        f = self.heating_filter
        recently = (last >= f["a_limit"]) & (temperature >= f["Tmin_K"]) & (temperature <= f["Tmax_K"])
        ne = np.asarray(gas.electron_number_densities.to_physical().to_value("m**-3"))[inside]
        out = {
            "xyz_mpc": xyz[inside], "r_mpc": r[inside],
            "y_mpc2": np.asarray(gas.compton_yparameters.to_physical().to_value("Mpc**2"))[inside],
            "temperature_k": temperature, "ne_m3": ne,
            "mass_msun": np.asarray(gas.masses.to_value("Msun"))[inside],
            "density_msun_mpc3": np.asarray(gas.densities.to_physical().to_value("Msun/Mpc**3"))[inside],
            "hsml_mpc": np.asarray(gas.smoothing_lengths.to_physical().to_value("Mpc"))[inside],
            "last_agn_a": last, "recently_heated": recently,
            "pe_pa": ne * Boltzmann * temperature,
            # SWIFT's total thermal gas pressure (all species); Pe/P ~ 0.52 for ionised primordial gas.
            "p_total_pa": np.asarray(gas.pressures.to_physical().to_value("Pa"))[inside],
            # Index (in the HBT catalogue, = SOAP halo_index) of the subhalo the particle is bound to; -1 if unbound.
            "group_nr_bound": np.asarray(gas.group_nr_bound)[inside].astype(np.int64),
            "n_loaded_from_cells": np.int64(n_cells),
        }
        return out


def load_halo_particles(snap: ReducedSnapshot, halo, cache_dir, radius_mpc=None, rebuild=False):
    """Cutout for one catalogue row (a pandas Series or dict), cached under cache_dir/particles.

    Default radius is the halo's R100c: the reduced snapshot holds nothing beyond it.
    """
    cache = Path(cache_dir) / "particles"
    cache.mkdir(parents=True, exist_ok=True)
    radius = float(halo["r_100_crit_mpc"] if radius_mpc is None else radius_mpc)
    tag = f"{snap.base.split('/')[-1]}_{snap.snapshot:04d}_halo{int(halo['halo_index'])}_r{radius:.4f}"
    path = cache / f"{tag}.npz"
    if path.exists() and not rebuild:
        return dict(np.load(path))
    centre = np.array([halo["centre_x_mpc"], halo["centre_y_mpc"], halo["centre_z_mpc"]], dtype=float)
    part = snap.cutout(centre, radius)
    part["radius_mpc"] = np.float64(radius)
    part["halo_index"] = np.int64(halo["halo_index"])
    np.savez_compressed(path, **part)
    return part


# --------------------------------------------------------------------------- profiles
def shell_pressure_from_y(sum_y_mpc2, inner_mpc, outer_mpc):
    """Volume-mean electron pressure [Pa] of a shell from the summed particle Compton y.

    Particle y = sigma_T/(m_e c^2) * P_e * V, so sum(y) * m_e c^2 / sigma_T = integral P_e dV.
    """
    volume_m3 = 4 * np.pi / 3 * (np.asarray(outer_mpc) ** 3 - np.asarray(inner_mpc) ** 3) * MPC_M**3
    return np.asarray(sum_y_mpc2) * MPC_M**2 * Y_TO_PRESSURE_VOLUME / volume_m3


def spherical_profile(part, r200c_mpc, edges_x, use_agn_filter=True):
    """Shell-mean electron pressure in units of x = r/R200c; returns a dict of per-shell arrays."""
    x = part["r_mpc"] / r200c_mpc
    y = np.where(part["recently_heated"], 0.0, part["y_mpc2"]) if use_agn_filter else part["y_mpc2"]
    counts = np.histogram(x, edges_x)[0]
    sum_y = np.histogram(x, edges_x, weights=y)[0]
    order = np.searchsorted(edges_x, x, side="right") - 1
    med_h = np.full(len(counts), np.nan)
    for k in range(len(counts)):
        sel = order == k
        if sel.any():
            med_h[k] = np.median(part["hsml_mpc"][sel]) / r200c_mpc
    pressure = shell_pressure_from_y(sum_y, edges_x[:-1] * r200c_mpc, edges_x[1:] * r200c_mpc)
    return {"x_inner": edges_x[:-1], "x_outer": edges_x[1:], "x_mid": np.sqrt(edges_x[:-1] * edges_x[1:]),
            "count": counts, "sum_y_mpc2": sum_y, "pe_pa": pressure, "median_hsml_x": med_h}


def projected_profile(part, r200c_mpc, edges_x, axis=2, use_agn_filter=True):
    """Area-mean Compton y in projected annuli (R in units of R200c) along one box axis.

    Every particle of the cutout is used, i.e. the line of sight is truncated at the
    cutout sphere.  Particle y is deposited at the particle centre (no kernel smoothing).
    """
    y = np.where(part["recently_heated"], 0.0, part["y_mpc2"]) if use_agn_filter else part["y_mpc2"]
    plane = np.delete(part["xyz_mpc"], axis, axis=1) / r200c_mpc
    big_r = np.linalg.norm(plane, axis=1)
    area_mpc2 = np.pi * np.diff(edges_x**2) * r200c_mpc**2
    counts = np.histogram(big_r, edges_x)[0]
    sum_y = np.histogram(big_r, edges_x, weights=y)[0]
    return {"x_inner": edges_x[:-1], "x_outer": edges_x[1:], "x_mid": np.sqrt(edges_x[:-1] * edges_x[1:]),
            "count": counts, "y": sum_y / area_mpc2}


def y_map(part, r200c_mpc, axis=2, half_extent_x=1.5, npix=128, use_agn_filter=True):
    """Compton-y image (dimensionless) of the cutout, pixels of size 2*half_extent_x/npix R200c."""
    y = np.where(part["recently_heated"], 0.0, part["y_mpc2"]) if use_agn_filter else part["y_mpc2"]
    plane = np.delete(part["xyz_mpc"], axis, axis=1) / r200c_mpc
    edges = np.linspace(-half_extent_x, half_extent_x, npix + 1)
    image = np.histogram2d(plane[:, 0], plane[:, 1], bins=edges, weights=y)[0]
    pixel_area_mpc2 = (2 * half_extent_x / npix * r200c_mpc) ** 2
    return image.T / pixel_area_mpc2, edges


def p200_pa(m200c_msun, r200c_mpc, baryon_fraction):
    """Thermal normalisation P200 = G M200c * 200 rho_crit * f_b / (2 R200c), as in XGPaint/Battaglia."""
    from scipy.constants import G
    mass = np.asarray(m200c_msun) * 1.988409870698051e30
    radius = np.asarray(r200c_mpc) * MPC_M
    rho200 = mass / (4 * np.pi * radius**3 / 3)
    return G * mass * rho200 * baryon_fraction / (2 * radius)


# --------------------------------------------------------------------------- outputs
def output_redshifts(remote, base, cache_dir, n_max=200):
    """snapshot -> redshift for every SOAP output of a run; cached as JSON in cache_dir."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{base.split('/')[-1]}_output_redshifts.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text()).items()}
    names = [n for n in remote[base + "/SOAP-HBT"] if n.startswith("halo_properties_")]
    table = {}
    for name in sorted(names)[:n_max]:
        snapshot = int(name.split("_")[-1].split(".")[0])
        table[snapshot] = round(cosmology(remote[soap_path(base, snapshot)])["redshift"], 4)
    path.write_text(json.dumps(table, indent=1) + "\n")
    return table
