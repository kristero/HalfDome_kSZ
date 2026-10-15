#!/usr/bin/env python3
"""CLASS-SZ tSZ halo-model spectra that separate cosmology, pressure profile and halo coverage.

All runs: Tinker08 M200c mass function, 3-D pressure truncated at x_outSZ = 4 R200c (as the maps),
D_ell = ell(ell+1)C_ell/2pi of Compton-y. Pressure profiles: Battaglia+12 (class_sz defaults) or the
FLAMINGO L2p8_m9 joint fits, whose nine coefficients use the same convention as class_sz
(M200c in physical Msun over 1e14, beta the Battaglia exponent, alpha = 1, gamma = -0.3).

Runs
  hd_b12            HalfDome cosmology, Battaglia+12, HalfDome catalogue mass/redshift bounds
  hd_flamingo       HalfDome cosmology, FLAMINGO joint fit (with SOAP shells), same bounds
  hd_flamingo_ps    HalfDome cosmology, FLAMINGO particle-shells-only fit, same bounds
  fl_flamingo       FLAMINGO cosmology, FLAMINGO joint fit, HalfDome bounds       -> cosmology factor
  fl_b12            FLAMINGO cosmology, Battaglia+12, HalfDome bounds             -> cosmology factor (B12)
  fl_flamingo_all   FLAMINGO cosmology, FLAMINGO joint fit, 1e10-5e15 Msun/h, z < 5  -> halo coverage
  fl_b12_all        FLAMINGO cosmology, Battaglia+12, 1e10-5e15 Msun/h, z < 5

Run with the class_sz environment (one classy_sz process at a time, GPU disabled):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kn18001/.local/lib/python3.11/site-packages \
  PATH_TO_CLASS_SZ_DATA=/home/kn18001/class_sz_data_directory /home/kn18001/.conda/envs/class_sz/bin/python class_sz_references.py [name ...]
"""
import json
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HALFDOME = dict(h=0.6774, Omega_b=0.0486, Omega_c=0.2603, sigma8=0.8159, n_s=0.9667, m_nu=0.0)
FLAMINGO = dict(h=0.681, Omega_b=0.0486, Omega_m=0.306, sigma8=0.807, n_s=0.967, m_nu=0.06)   # D3A fiducial
HD_BOUNDS = dict(M_min=4458834966504.787, M_max=2573960654045765.0, z_min=0.002141595, z_max=3.8554277)   # Msun/h
ALL_BOUNDS = dict(M_min=1e10, M_max=5e15, z_min=0.002, z_max=5.0)


def b12_parameters(toml_file):
    c = tomllib.loads(Path(toml_file).read_text())
    assert c["alpha_amp"] == 1.0 and c["gamma_amp"] == -0.3 and all(
        c[k] == 0.0 for k in ("alpha_alpha_m", "alpha_alpha_z", "gamma_alpha_m", "gamma_alpha_z"))
    p = {"P0_B12": c["P0_amp"], "xc_B12": c["x_c_amp"], "beta_B12": c["beta_amp"],
         "alpha_m_P0_B12": c["P0_alpha_m"], "alpha_m_xc_B12": c["x_c_alpha_m"], "alpha_m_beta_B12": c["beta_alpha_m"],
         "alpha_z_P0_B12": c["P0_alpha_z"], "alpha_z_xc_B12": c["x_c_alpha_z"], "alpha_z_beta_B12": c["beta_alpha_z"]}
    # the (disabled in this source) break model: make any break a no-op
    p.update({"alphap_m_P0_B12": p["alpha_m_P0_B12"], "alphap_m_xc_B12": p["alpha_m_xc_B12"],
              "alphap_m_beta_B12": p["alpha_m_beta_B12"]})
    return p


def cosmology_parameters(c):
    h = c["h"]
    pars = {"omega_b": c["Omega_b"] * h**2, "h": h, "n_s": c["n_s"], "sigma8": c["sigma8"], "cosmo_model": 0}
    if c["m_nu"] > 0:
        omega_nu = c["m_nu"] / 93.14
        pars.update({"N_ncdm": 1, "m_ncdm": c["m_nu"], "N_ur": 2.0328, "omega_cdm": (c["Omega_m"] - c["Omega_b"]) * h**2 - omega_nu})
    else:
        pars.update({"N_ncdm": 0, "N_ur": 3.046, "omega_cdm": c["Omega_c"] * h**2})
    return pars


def run(name, cosmology, bounds, profile, out_dir):
    from classy_sz import Class
    pars = {**cosmology_parameters(cosmology), "output": "tSZ_1h,tSZ_2h",
            "ell_min": 10.0, "ell_max": 8192.0, "dell": 0.0, "dlogell": 0.05, **bounds,
            "mass_function": "T08M200c", "pressure_profile": "B12", "concentration_parameter": "D08",
            "x_outSZ": 4.0, "ndim_masses": 200, "ndim_redshifts": 100,
            "n_m_pressure_profile": 100, "n_z_pressure_profile": 100, "N_samp_fftw": 4096, "use_fft_for_profiles_transform": 1,
            "redshift_epsabs": 1e-40, "redshift_epsrel": 1e-4, "mass_epsabs": 1e-40, "mass_epsrel": 1e-4, "hm_consistency": 0,
            **profile}
    c = Class()
    if cosmology["m_nu"] == 0:
        for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
            c.pars.pop(k, None)
    c.set(pars)
    t0 = time.time()
    c.compute()
    dt = time.time() - t0
    r = c.cl_sz()
    ell = np.asarray(r["ell"], float)
    h1, h2 = np.asarray(r["1h"], float) * 1e-12, np.asarray(r["2h"], float) * 1e-12
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), name=name, parameters=pars, compute_seconds=dt,
                convention="D_ell of dimensionless Compton-y (class_sz native 1e12 removed); x_outSZ = 4 R200c")
    np.savez(out_dir / f"class_sz_{name}.npz", ell=ell, dl_1h=h1, dl_2h=h2, dl_total=h1 + h2, metadata_json=json.dumps(meta, sort_keys=True))
    i = int(np.argmin(np.abs(ell - 3000)))
    print(f"{name}: {dt:.1f} s; 1e12 D_ell(3000) 1h {1e12 * h1[i]:.4f} 2h {1e12 * h2[i]:.5f}", flush=True)
    c.struct_cleanup()
    c.empty()


if __name__ == "__main__":
    out = HERE / "class_sz"
    out.mkdir(exist_ok=True)
    main = b12_parameters(HERE / "flamingo_joint_fit.toml")
    ps = b12_parameters(HERE / "flamingo_joint_fit_particle_shells.toml")
    runs = [("hd_b12", HALFDOME, HD_BOUNDS, {}), ("hd_flamingo", HALFDOME, HD_BOUNDS, main),
            ("hd_flamingo_ps", HALFDOME, HD_BOUNDS, ps), ("fl_flamingo", FLAMINGO, HD_BOUNDS, main),
            ("fl_b12", FLAMINGO, HD_BOUNDS, {}),
            ("fl_flamingo_all", FLAMINGO, ALL_BOUNDS, main), ("fl_b12_all", FLAMINGO, ALL_BOUNDS, {})]
    wanted = set(sys.argv[1:])                      # run names on the command line; all runs by default
    for name, cosmo, bounds, profile in runs:
        if not wanted or name in wanted:
            run(name, cosmo, bounds, profile, out)
    print("ALL DONE", flush=True)
