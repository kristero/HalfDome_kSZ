#!/usr/bin/env python3
"""CLASS-SZ halo-model references for the truncation comparison.

tSZ: Battaglia12 pressure profile (class_sz 'B12'), x_outSZ = 4 (3-D truncation at 4 R200c).
kSZ: Battaglia16 AGN gas density (class_sz 'B16', mode 'agn'), electrons truncated at 4 R200c,
     f_free = 0.9 and mu_e matched to XGPaint's ne2d composition (X_H = 0.76).
Mass and redshift bounds are the HalfDome lightcone catalogue bounds (M200c, Msun/h).
"""
import json, sys, time
from datetime import datetime, timezone
import numpy as np

H = 0.6774; OMEGA_B = 0.0486; OMEGA_C = 0.2603; SIGMA8 = 0.8159; N_S = 0.9667
# catalogue bounds (halo_mass_m200c in Msun/h; redshift)
M_MIN = 4458834966504.787; M_MAX = 2573960654045765.0
Z_MIN = 0.002141595; Z_MAX = 3.8554277
X_OUT = 4.0
# XGPaint ne2d composition: mass per free electron for X_H = 0.76 fully ionized H+He, incl. m_e
XH = 0.76
MU_E = (0.000544617 + 2*XH/(XH+1) + (1-XH)/(2*(1+XH))*4)   # in proton masses: m_e + nH/ne*mH + nHe/ne*4mH
F_FREE = 0.9

def run(output, kind, x_out):
    from classy_sz import Class
    common = {
        "omega_b": OMEGA_B*H**2, "omega_cdm": OMEGA_C*H**2, "h": H, "n_s": N_S, "sigma8": SIGMA8,
        "N_ncdm": 0, "N_ur": 3.046, "cosmo_model": 0,
        "ell_min": 10.0, "ell_max": 8192.0, "dell": 0.0, "dlogell": 0.05,
        "z_min": Z_MIN, "z_max": Z_MAX, "M_min": M_MIN, "M_max": M_MAX,
        "mass_function": "T08M200c", "concentration_parameter": "D08",
        "ndim_masses": 200, "ndim_redshifts": 100,
        "redshift_epsabs": 1e-40, "redshift_epsrel": 1e-3, "mass_epsabs": 1e-40, "mass_epsrel": 1e-3,
        "hm_consistency": 0,
    }
    if kind == "tsz":
        pars = dict(common, output="tSZ_1h,tSZ_2h", pressure_profile="B12", x_outSZ=float(x_out),
                    n_m_pressure_profile=100, n_z_pressure_profile=100,
                    use_fft_for_profiles_transform=1, N_samp_fftw=4096)
    else:
        pars = dict(common, output="kSZ_kSZ_1h,kSZ_kSZ_2h", gas_profile="B16", gas_profile_mode="agn",
                    x_out_truncated_density_profile_electrons=float(x_out),
                    f_free=F_FREE,   # mu_e is not an input parameter in this class_sz build (fixed 1.14 vs XGPaint 1.137)
                    n_m_density_profile=100, n_z_density_profile=100,
                    use_fft_for_profiles_transform=0)
    c = Class()
    for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
        c.pars.pop(k, None)
    c.set(pars)
    t0 = time.time(); c.compute(); dt = time.time() - t0
    r = c.cl_sz() if kind == "tsz" else c.cl_ksz()
    ell = np.asarray(r["ell"], float); h1 = np.asarray(r["1h"], float); h2 = np.asarray(r["2h"], float)
    extra = {}
    if kind == "ksz":
        zs = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
        extra["vrms2_over_c2_at_z"] = {f"{z}": float(c.get_vrms2_at_z(z)) for z in zs}
        extra["f_free"] = float(c.get_f_free()); extra["mu_e"] = float(c.get_mu_e())
    md = {"created_utc": datetime.now(timezone.utc).isoformat(), "kind": kind, "parameters": pars,
          "compute_seconds": dt, "spectrum_convention": "class_sz native: tSZ = 1e12 * ell(ell+1)C_ell/2pi (Compton-y); kSZ = ell(ell+1)C_ell/2pi in (Delta T/T)^2 (to be confirmed against the painted-map Poisson term)",
          "truncation": f"3-D sphere of {x_out} R200c (class_sz x_out)", "xgpaint_mu_e_for_reference": MU_E, **extra}
    np.savez(output, ell=ell, dl_1h=h1, dl_2h=h2, dl_total=h1 + h2, metadata_json=json.dumps(md, sort_keys=True))
    print(f"saved {output} ({kind}) in {dt:.1f} s; ell {ell[0]:.0f}-{ell[-1]:.0f}, n={ell.size}", flush=True)
    print("  1h[ell~3000] =", np.interp(3000, ell, h1), " 2h =", np.interp(3000, ell, h2), flush=True)
    if extra: print("  ", extra, flush=True)
    c.struct_cleanup(); c.empty()

if __name__ == "__main__":
    out_dir = sys.argv[1]
    kinds = sys.argv[2:] or ["tsz", "ksz"]
    for kind in kinds:
        run(f"{out_dir}/class_sz_{kind}_x{X_OUT:g}_reference.npz", kind, X_OUT)
