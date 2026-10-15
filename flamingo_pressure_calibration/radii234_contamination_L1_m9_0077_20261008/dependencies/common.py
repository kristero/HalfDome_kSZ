"""Shared configuration, paths, logging, atomic I/O, constants and halo normalisations.

Every stage imports this module; see SPEC.md section 9 for the contract. All lengths in
Mpc, masses in Msun, pressures in Pa unless a name says otherwise. Positions and smoothing
lengths are comoving (``*_cmpc``), everything else physical.
"""
from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import zlib

import numpy as np
from scipy.constants import G as G_SI, c as C_LIGHT, electron_mass, physical_constants, proton_mass

# --------------------------------------------------------------------------- constants (SI unless noted)
MPC_M = 3.085677581491367e22
MPC_CM = 3.08567758e24            # FLAMINGO's internal length unit in cm (the conversion attributes use it)
MSUN_KG = 1.988409870698051e30    # as flamingo_reduced_io.p200_pa
MSUN_G = 1.98841e33               # SWIFT's solar mass: FLAMINGO/SOAP masses are multiples of it
M_E = electron_mass               # as the legacy Y_TO_PRESSURE_VOLUME
M_P = proton_mass
SIGMA_T = physical_constants["Thomson cross section"][0]
ELECTRON_FRACTION = 0.5176
Y_TO_PRESSURE_VOLUME = M_E * C_LIGHT**2 / SIGMA_T     # Pa m^3 per m^2 of particle y (y_i = sigma_T/(m_e c^2) P_e V_i)

# XGPaint (profiles_tau.jl, ne2d) turns rho into n_e = 0.9 rho / (m_e + nH_ne m_H + nHe_ne 4 m_H) with the
# CODATA 2018 constants of PhysicalConstants.jl; the same values are used here so that n_e round-trips exactly.
M_E_XGP = 9.1093837015e-31
M_P_XGP = 1.67262192369e-27
X_H = 0.76
NH_NE = 2 * X_H / (X_H + 1)
NHE_NE = (1 - X_H) / (2 * (1 + X_H))
MU_XGP_KG = (M_E_XGP + NH_NE * M_P_XGP + NHE_NE * 4 * M_P_XGP) / 0.9
MU_XGP_MSUN = MU_XGP_KG / (MSUN_G * 1e-3)   # in the solar mass of the catalogue, so n_e * MU / rho_crit is consistent

APERTURES = ["2500_crit", "1000_crit", "500_crit", "200_crit", "100_crit", "50_crit", "5xR_500_crit"]

# --------------------------------------------------------------------------- configuration
DEFAULTS = {
    "selection": {"log_m_min": 12.5, "fine_bin_dex": 0.05, "modes": ["random_block", "complete"],
                  "random_block_k": 2, "random_block_fraction": 0.02, "random_block_offset": "random",
                  "random_cell_fraction": 0.02, "complete_log_m_min": 14.3, "max_complete_per_fine_bin": 50,
                  "seed": 20260929},
    "radial": {"x_inner": 0.01, "x_outer": 5.0, "n_log_bins": 54, "sphere_r200c": 5.0,
               "hsml_log10_edges": [-3.0, 1.0, 80], "extra_edges": [2.0, 3.0, 4.0]},
    "particles": {"cell_padding_fraction": 0.1, "workers": 8, "tasks": 16, "checkpoint_every_cells": 50,
                  "read_from": "auto", "membership_field": "auto", "cells_per_batch": 8,
                  "hdf5_chunk_cache_mb": 64, "max_open_chunks": 2},
    # fallback when a snapshot lacks Parameters/Gravity:*_baryon_softening (FLAMINGO m9 values, internal Mpc)
    "softening": {"comoving_mpc": 0.0223, "max_physical_mpc": 0.0057},
    "merge": {"max_median_check_y200": 1e-3},
}


def _deep_update(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path) -> dict:
    """config.json with the defaults above filled in; relative roots resolve against the config's directory."""
    path = Path(path).resolve()
    cfg = _deep_update(copy.deepcopy(DEFAULTS), json.loads(path.read_text()))
    for key in ("output_root", "data_root"):
        value = str(cfg.get(key, ""))
        remote = ":" in value.split("/")[0]                    # e.g. hdfstream:FLAMINGO
        if value and not remote and not os.path.isabs(value):
            cfg[key] = str((path.parent / value).resolve())
    cfg["_config_path"] = str(path)
    return cfg


def enabled_sims(cfg) -> list:
    return [name for name, sim in cfg["simulations"].items() if sim.get("enabled", True)]


def sim_cfg(cfg, sim) -> dict:
    try:
        return cfg["simulations"][sim]
    except KeyError:
        raise KeyError(f"simulation {sim!r} is not in the config ({sorted(cfg['simulations'])})") from None


def selection_cfg(cfg, sim=None) -> dict:
    """cfg['selection'] with the simulation's own 'selection' overrides (e.g. random_block_fraction or random_block_k
    for the 1 Gpc resolution runs, whose 31.25 cMpc cells make 2^3-cell blocks small) deep-merged over it."""
    out = copy.deepcopy(cfg["selection"])
    if sim is not None:
        _deep_update(out, copy.deepcopy(sim_cfg(cfg, sim).get("selection", {})))
    return out


def radial_edges(cfg) -> np.ndarray:
    """Shell edges in x = r/R200c: 0, then n_log_bins+1 log-spaced edges from x_inner to x_outer, with every
    `extra_edges` value (default 2, 3, 4: the outer edges of the r2, r3 and baseline ranges) placed exactly on the
    grid: the nearest log edge moves onto it when it lies within 0.35 of a log-bin width, else it is inserted.
    A numeric range x_max then ends exactly at its label (SPEC section 5)."""
    r = cfg["radial"]
    lo, hi, n = float(r["x_inner"]), float(r["x_outer"]), int(r["n_log_bins"])
    edges = np.geomspace(lo, hi, n + 1)
    width = np.log(hi / lo) / n
    for value in sorted(float(v) for v in (r.get("extra_edges") or [])):
        if not lo < value < hi or np.any(edges == value):
            continue
        i = int(np.argmin(np.abs(np.log(edges / value))))
        if 0 < i < len(edges) - 1 and abs(np.log(edges[i] / value)) <= 0.35 * width:
            edges[i] = value
        else:
            edges = np.sort(np.r_[edges, value])
    return np.r_[0.0, edges]


def hsml_log_edges(cfg) -> np.ndarray:
    lo, hi, n = cfg["radial"]["hsml_log10_edges"]
    return np.linspace(float(lo), float(hi), int(n) + 1)


def seed_for(cfg, sim, snap, stream) -> list:
    """Deterministic numpy seed sequence for (seed, sim, snapshot, stream); the seed is the simulation's own
    selection seed when simulations.<sim>.selection overrides it (selection_cfg)."""
    return [int(selection_cfg(cfg, sim)["seed"]), zlib.crc32(str(sim).encode()), int(snap), int(stream)]


# --------------------------------------------------------------------------- paths
def paths(cfg, sim=None, snap=None) -> dict:
    root = Path(cfg["output_root"])
    out = {"root": root, "fits_dir": root / "fits", "reports_dir": root / "reports",
           "figures_dir": root / "figures", "logs_dir": root / "logs"}
    if sim is not None and snap is not None:
        tag = f"{sim}_{int(snap):04d}"
        particles_dir = root / "particles" / tag
        out.update({
            "tag": tag,
            "catalogue_npz": root / "catalogue" / f"{tag}_haloes.npz",
            "catalogue_json": root / "catalogue" / f"{tag}_haloes.json",
            "particles_dir": particles_dir,
            "task_npz": lambda k: particles_dir / f"task_{int(k):04d}.npz",
            "task_done": lambda k: particles_dir / f"task_{int(k):04d}.done.json",
            "task_checkpoint": lambda k: particles_dir / f"task_{int(k):04d}.checkpoint.npz",
            "products_npz": root / "products" / f"{tag}_profiles.npz",
            "products_json": root / "products" / f"{tag}_profiles.json",
        })
        for key in ("catalogue_npz", "products_npz"):
            out[key].parent.mkdir(parents=True, exist_ok=True)
        particles_dir.mkdir(parents=True, exist_ok=True)
    for key in ("fits_dir", "reports_dir", "figures_dir", "logs_dir"):
        out[key].mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- logging and atomic I/O
def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def plural(n, singular, plural_form=None) -> str:
    """'1 halo', '2 haloes', '1 stack', '3 stacks' (report and export wording)."""
    n = int(n)
    return f"{n} {singular if n == 1 else (plural_form or singular + 's')}"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# flock errors of filesystems without lock support (Lustre without flock/localflock, NFS without lockd)
_NO_LOCK_ERRNOS = {errno.ENOSYS, errno.ENOLCK, errno.EOPNOTSUPP, errno.EINVAL, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}


def log_event(path, echo=True, **fields):
    """Append one JSON line {"utc": ..., **fields} to `path` and echo it to stdout.

    The line is written with ONE O_APPEND write under an exclusive flock; where the filesystem does not support
    flock (ENOSYS, ENOLCK, EOPNOTSUPP) the same single append is made unlocked (short lines are not interleaved
    by O_APPEND writes on local and parallel filesystems in practice)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"utc": utc_now(), **fields}, default=json_default)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in _NO_LOCK_ERRNOS:
                raise
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)
    if echo:
        print(line, flush=True)
    return line


def lock_test(directory) -> dict:
    """Whether flock works in `directory` (preflight: the stages' JSON-line logs use it, with an unlocked fallback)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".lock_test.{os.getpid()}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return {"directory": str(directory), "flock": True}
    except OSError as exc:
        return {"directory": str(directory), "flock": False, "errno": exc.errno, "error": os.strerror(exc.errno or 0),
                "fallback": "unlocked single O_APPEND writes" if exc.errno in _NO_LOCK_ERRNOS else "none: log_event raises"}
    finally:
        os.close(fd)
        try:
            path.unlink()
        except OSError:
            pass


def _temporary(path):
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def atomic_savez(path, compress=False, **arrays):
    """np.savez to a temporary file in the same directory, fsync, then rename over `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    with temporary.open("wb") as stream:
        (np.savez_compressed if compress else np.savez)(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    temporary.write_text(json.dumps(value, indent=1, default=json_default, allow_nan=True) + "\n")
    os.replace(temporary, path)
    return path


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fingerprint(*arrays) -> str:
    """Short hash of array contents (dtype, shape and bytes), to tie stage outputs to one catalogue."""
    digest = hashlib.sha1()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:20]


# --------------------------------------------------------------------------- attributes and cosmology
def scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def attr_text(value) -> str:
    value = np.asarray(value).reshape(-1)[0] if np.ndim(value) else value
    return value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)


def physical_factor(dataset) -> float:
    """CGS factor of a dataset, including the a-scaling (the conversion used for all physical quantities)."""
    return scalar(dataset.attrs["Conversion factor to physical CGS (including cosmological corrections)"])


def cosmology_from_attrs(attrs) -> dict:
    """Cosmology group attributes. SWIFT's Omega_m excludes massive neutrinos (flamingo_reduced_io.cosmology)."""
    attrs = dict(attrs)
    get = lambda key, default=np.nan: scalar(attrs[key]) if key in attrs else default    # noqa: E731
    omega_nu_0 = get("Omega_nu_0", 0.0)
    omega_m_total = get("Omega_m") + omega_nu_0
    return {"redshift": get("Redshift"), "a": get("Scale-factor"), "h": get("h"), "omega_b": get("Omega_b"),
            "omega_m": get("Omega_m"), "omega_cdm": get("Omega_cdm"), "omega_nu_0": omega_nu_0,
            "omega_lambda": get("Omega_lambda"), "omega_m_total": omega_m_total,
            "baryon_fraction": get("Omega_b") / omega_m_total,
            "H_internal": get("H [internal units]"), "critical_density_internal": get("Critical density [internal units]")}


def baryon_fraction(attrs) -> float:
    return cosmology_from_attrs(attrs)["baryon_fraction"]


def recently_heated(last_agn_a, temperature_k, heating) -> np.ndarray:
    """SOAP's recently-AGN-heated gas: last >= a_limit & Tmin <= T <= Tmax.

    The thresholds are Python floats, so a float32 array is compared in float32 exactly as in
    stack_products.stream_haloes (pass the arrays in their stored dtype to reproduce it).
    """
    a_limit = float(heating["a_limit"])
    tmin = float(heating["tmin_k"] if "tmin_k" in heating else heating["Tmin_K"])
    tmax = float(heating["tmax_k"] if "tmax_k" in heating else heating["Tmax_K"])
    return (last_agn_a >= a_limit) & (temperature_k >= tmin) & (temperature_k <= tmax)


# --------------------------------------------------------------------------- halo normalisations
def p200_pa(m200c_msun, r200c_mpc, baryon_fraction):
    """P200 = G M200c (200 rho_crit) f_b / (2 R200c), exactly flamingo_reduced_io.p200_pa."""
    mass = np.asarray(m200c_msun) * MSUN_KG
    radius = np.asarray(r200c_mpc) * MPC_M
    rho200 = mass / (4 * np.pi * radius**3 / 3)
    return G_SI * mass * rho200 * baryon_fraction / (2 * radius)


def rho_crit_from_halo(m200c_msun, r200c_mpc):
    """Critical density [Msun/Mpc^3, physical] implied by SOAP's own M200c and R200c."""
    return np.asarray(m200c_msun) / (200 * 4 * np.pi / 3 * np.asarray(r200c_mpc) ** 3)


def shell_pressure_from_y(sum_y_mpc2, inner_mpc, outer_mpc):
    """Volume-mean electron pressure [Pa] of a shell from summed particle Compton y (flamingo_reduced_io)."""
    volume_m3 = 4 * np.pi / 3 * (np.asarray(outer_mpc) ** 3 - np.asarray(inner_mpc) ** 3) * MPC_M**3
    return np.asarray(sum_y_mpc2) * MPC_M**2 * Y_TO_PRESSURE_VOLUME / volume_m3


def softening_physical_mpc(a, comoving_mpc, max_physical_mpc):
    return min(float(comoving_mpc) * float(a), float(max_physical_mpc))
