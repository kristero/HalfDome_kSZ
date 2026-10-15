#!/usr/bin/env python3
"""CLASS-SZ Battaglia12 tSZ with the WebSky simulation setup (Stein+2020: Omega_m=0.31, Omega_b=0.049, h=0.68,
sigma8=0.81, n_s=0.965; halos above ~1e12 Msun/h to z=4.6; pressure integrated to 4 R200c), to separate the
catalogue/cosmology difference from the truncation-radius difference when comparing HalfDome with WebSky."""
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
H = 0.68; OMEGA_M = 0.31; OMEGA_B = 0.049; SIGMA8 = 0.81; N_S = 0.965
def run(x_out, m_min, output):
    from classy_sz import Class
    pars = {"omega_b": OMEGA_B*H**2, "omega_cdm": (OMEGA_M-OMEGA_B)*H**2, "h": H, "n_s": N_S, "sigma8": SIGMA8,
            "N_ncdm": 0, "N_ur": 3.046, "cosmo_model": 0, "output": "tSZ_1h,tSZ_2h",
            "ell_min": 10.0, "ell_max": 8192.0, "dell": 0.0, "dlogell": 0.05,
            "z_min": 0.005, "z_max": 4.6, "M_min": m_min, "M_max": 5e15,
            "mass_function": "T08M200c", "pressure_profile": "B12", "concentration_parameter": "D08",
            "x_outSZ": float(x_out), "ndim_masses": 200, "ndim_redshifts": 100,
            "n_m_pressure_profile": 100, "n_z_pressure_profile": 100, "N_samp_fftw": 4096, "use_fft_for_profiles_transform": 1,
            "redshift_epsabs": 1e-40, "redshift_epsrel": 1e-3, "mass_epsabs": 1e-40, "mass_epsrel": 1e-3, "hm_consistency": 0}
    c = Class()
    for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"): c.pars.pop(k, None)
    c.set(pars); t0 = time.time(); c.compute(); dt = time.time() - t0
    r = c.cl_sz(); ell = np.asarray(r["ell"], float); h1 = np.asarray(r["1h"], float)*1e-12; h2 = np.asarray(r["2h"], float)*1e-12
    np.savez(output, ell=ell, dl_1h=h1, dl_2h=h2, dl_total=h1+h2, metadata_json=json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), parameters=pars, compute_seconds=dt, setup="WebSky cosmology and halo range", x_outSZ=x_out), sort_keys=True))
    i = int(np.argmin(np.abs(ell-3000))); print(f"websky-setup x_out={x_out} M_min={m_min:.1e}: 1e12 D(3000) 1h={1e12*h1[i]:.4f} 2h={1e12*h2[i]:.5f} ({dt:.0f} s)", flush=True)
    c.struct_cleanup(); c.empty()
if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    run(4, 1.0e12, out / "class_sz_tsz_websky_setup_xout4_mmin1e12.npz")
    run(4, 4.458834966504787e12 * 0.68 / 0.6774, out / "class_sz_tsz_websky_setup_xout4_halfdome_mmin.npz")   # same physical mass floor as HalfDome
