#!/usr/bin/env python3
"""Compute a CLASS-SZ Battaglia12 tSZ reference for the HalfDome lightcone.

The halo-model mass and redshift limits are read from the provenance of the
internally painted HalfDome Compton-y map.  CLASS-SZ reports its tSZ spectra as
``1e12 * D_ell``; this script stores both that native convention and the
dimensionless ``D_ell`` used by the map-spectrum analysis.

The CLASS-SZ pressure transform uses its native ``x_outSZ`` convention.  This
is deliberately recorded rather than described as an R200c aperture: it is
not geometrically identical to the external R200c cut used by the map painter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# HalfDome simulation cosmology (arXiv:2407.17462), with massless neutrinos.
H = 0.6774
OMEGA_B = 0.0486
OMEGA_C = 0.2603
SIGMA8 = 0.8159
N_S = 0.9667


def read_provenance(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Malformed provenance line {line_number}: {raw_line!r}")
        key, value = line.split("=", 1)
        values[key] = value
    return values


def require_float(values: dict[str, str], key: str) -> float:
    try:
        value = float(values[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {key!r} in tSZ provenance") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in tSZ provenance: {value}")
    return value


def validate_map_provenance(values: dict[str, str]) -> None:
    expected = {
        "observable": "thermal SZ Compton-y",
        "profile_mass_definition": "M200c",
        "profile_mass_units": "physical Msun",
        "catalog_mass_dataset": "halo_mass_m200c",
        "catalog_truncated": "false",
        "beam": "none",
        "instrumental_noise": "none",
    }
    for key, wanted in expected.items():
        found = values.get(key)
        if found != wanted:
            raise ValueError(f"tSZ provenance {key}={found!r}; expected {wanted!r}")
    if not values.get("profile_label", "").startswith("Battaglia12 fiducial"):
        raise ValueError("The tSZ map does not use the fiducial Battaglia12 profile")
    if "complete resolved catalogue range" not in values.get("mass_selection", ""):
        raise ValueError("The tSZ map does not use the complete resolved mass range")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsz-provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ell-max", type=float, default=8192.0)
    parser.add_argument("--x-out-sz", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    provenance_path = args.tsz_provenance.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    if args.ell_max < 100:
        raise ValueError("ell-max must be at least 100")
    if args.x_out_sz <= 0:
        raise ValueError("x-out-sz must be positive")

    provenance = read_provenance(provenance_path)
    validate_map_provenance(provenance)
    mass_min_physical = require_float(provenance, "selected_mass_min_msun")
    mass_max_physical = require_float(provenance, "selected_mass_max_msun")
    redshift_min = require_float(provenance, "selected_redshift_min")
    redshift_max = require_float(provenance, "selected_redshift_max")
    if not 0 < mass_min_physical < mass_max_physical:
        raise ValueError("Invalid selected physical-mass bounds in tSZ provenance")
    if not 0 <= redshift_min < redshift_max:
        raise ValueError("Invalid selected redshift bounds in tSZ provenance")

    # The CLASS-SZ halo integral uses Msun/h, while the map painter receives
    # physical Msun.  Match the same resolved catalogue mass interval.
    mass_min_class_sz = mass_min_physical * H
    mass_max_class_sz = mass_max_physical * H

    from classy_sz import Class

    parameters = {
        "omega_b": OMEGA_B * H**2,
        "omega_cdm": OMEGA_C * H**2,
        "h": H,
        "n_s": N_S,
        "sigma8": SIGMA8,
        "N_ncdm": 0,
        "N_ur": 3.046,
        "cosmo_model": 0,
        "output": "tSZ_1h,tSZ_2h",
        "ell_min": 10.0,
        "ell_max": float(args.ell_max),
        "dell": 0.0,
        "dlogell": 0.025,
        "z_min": redshift_min,
        "z_max": redshift_max,
        "M_min": mass_min_class_sz,
        "M_max": mass_max_class_sz,
        "mass_function": "T08M200c",
        "pressure_profile": "B12",
        "concentration_parameter": "D08",
        "x_outSZ": float(args.x_out_sz),
        "ndim_masses": 200,
        "ndim_redshifts": 100,
        "n_m_pressure_profile": 100,
        "n_z_pressure_profile": 100,
        "N_samp_fftw": 4096,
        "use_fft_for_profiles_transform": 1,
        "redshift_epsabs": 1.0e-40,
        "redshift_epsrel": 1.0e-3,
        "mass_epsabs": 1.0e-40,
        "mass_epsrel": 1.0e-3,
        "hm_consistency": 0,
    }

    solver = Class()
    # Some CLASS-SZ wheels insert massive-neutrino defaults during
    # construction.  Remove them before requesting the massless simulation.
    for key in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
        solver.pars.pop(key, None)
    solver.set(parameters)
    try:
        solver.compute()
        result = solver.cl_sz()
        ell = np.asarray(result["ell"], dtype=float)
        dl_1h_x1e12 = np.asarray(result["1h"], dtype=float)
        dl_2h_x1e12 = np.asarray(result["2h"], dtype=float)
    finally:
        solver.struct_cleanup()
        solver.empty()

    for name, values in {
        "ell": ell,
        "dl_yy_1h_x1e12": dl_1h_x1e12,
        "dl_yy_2h_x1e12": dl_2h_x1e12,
    }.items():
        if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid CLASS-SZ array: {name}")
    if not np.all(np.diff(ell) > 0):
        raise ValueError("CLASS-SZ multipoles are not strictly increasing")
    if np.any(dl_1h_x1e12 < 0) or np.any(dl_2h_x1e12 < 0):
        raise ValueError("CLASS-SZ returned a negative tSZ bandpower")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "class_sz_version": importlib.metadata.version("classy_sz"),
        "observable": "thermal SZ Compton-y auto-spectrum",
        "spectrum_convention": "D_ell = ell(ell+1) C_ell / (2pi)",
        "native_class_sz_output_units": "1e12 * dimensionless D_ell",
        "pressure_profile": "Battaglia12 (CLASS-SZ B12)",
        "halo_model_components": "1h and 2h",
        "mass_function": "Tinker08 M200c",
        "mass_units_class_sz": "Msun/h",
        "mass_bounds_source": "selected physical-M200c bounds in tSZ map provenance",
        "radial_cut_convention": "CLASS-SZ native x_outSZ; not an R200c aperture",
        "tsz_map_provenance": str(provenance_path),
        "parameters": parameters,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ell=ell,
        dl_yy_1h=dl_1h_x1e12 * 1.0e-12,
        dl_yy_2h=dl_2h_x1e12 * 1.0e-12,
        dl_yy_total=(dl_1h_x1e12 + dl_2h_x1e12) * 1.0e-12,
        dl_yy_1h_x1e12=dl_1h_x1e12,
        dl_yy_2h_x1e12=dl_2h_x1e12,
        dl_yy_total_x1e12=dl_1h_x1e12 + dl_2h_x1e12,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Saved CLASS-SZ Battaglia12 reference: {output_path}")


if __name__ == "__main__":
    main()
