#!/usr/bin/env python3
"""CLASS-SZ Battaglia12 tSZ references with the 3-D pressure-profile truncation x_outSZ = 1..5
(units of R200c), HalfDome cosmology and the HalfDome lightcone catalogue mass/redshift bounds.
Companion to the cluster-painted spherical-truncation radius series."""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

H = 0.6774; OMEGA_B = 0.0486; OMEGA_C = 0.2603; SIGMA8 = 0.8159; N_S = 0.9667
M_MIN = 4458834966504.787; M_MAX = 2573960654045765.0   # halo_mass_m200c bounds, Msun/h
Z_MIN = 0.002141595; Z_MAX = 3.8554277

def run(x_out, output):
    from classy_sz import Class
    pars = {"omega_b": OMEGA_B*H**2, "omega_cdm": OMEGA_C*H**2, "h": H, "n_s": N_S, "sigma8": SIGMA8,
            "N_ncdm": 0, "N_ur": 3.046, "cosmo_model": 0, "output": "tSZ_1h,tSZ_2h",
            "ell_min": 10.0, "ell_max": 8192.0, "dell": 0.0, "dlogell": 0.05,
            "z_min": Z_MIN, "z_max": Z_MAX, "M_min": M_MIN, "M_max": M_MAX,
            "mass_function": "T08M200c", "pressure_profile": "B12", "concentration_parameter": "D08",
            "x_outSZ": float(x_out), "ndim_masses": 200, "ndim_redshifts": 100,
            "n_m_pressure_profile": 100, "n_z_pressure_profile": 100, "N_samp_fftw": 4096, "use_fft_for_profiles_transform": 1,
            "redshift_epsabs": 1e-40, "redshift_epsrel": 1e-3, "mass_epsabs": 1e-40, "mass_epsrel": 1e-3, "hm_consistency": 0}
    c = Class()
    for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
        c.pars.pop(k, None)
    c.set(pars); t0 = time.time(); c.compute(); dt = time.time() - t0
    r = c.cl_sz(); ell = np.asarray(r["ell"], float); h1 = np.asarray(r["1h"], float) * 1e-12; h2 = np.asarray(r["2h"], float) * 1e-12
    md = dict(created_utc=datetime.now(timezone.utc).isoformat(), x_outSZ=x_out, parameters=pars, compute_seconds=dt,
              convention="D_ell = ell(ell+1)C_ell/2pi of dimensionless Compton-y (class_sz native 1e12 factor removed)",
              truncation="3-D pressure profile truncated at x_outSZ R200c (class_sz x_outSZ)")
    np.savez(output, ell=ell, dl_1h=h1, dl_2h=h2, dl_total=h1 + h2, metadata_json=json.dumps(md, sort_keys=True))
    i = int(np.argmin(np.abs(ell - 3000)))
    print(f"x_out={x_out}: saved {output} in {dt:.1f} s; 1e12 D_ell(3000) 1h={1e12*h1[i]:.4f} 2h={1e12*h2[i]:.5f}", flush=True)
    c.struct_cleanup(); c.empty()

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for x in (1, 2, 3, 4, 5):
        run(x, out / f"class_sz_tsz_xout{x}.npz")
