#!/usr/bin/env python3
"""Second CLASS-SZ batch (run only after run_class_sz.py has finished; one classy_sz process at a time):
  1. precision convergence of the B12 reference: finer pressure-profile tables (n_z/n_m/n_l), FFTLog sampling and
     z/M integration grids; tests whether CLASS-SZ's low-ell (< 100) 1-halo deficit and its 0.5-0.9 % higher y_ell
     relative to the exact 3-D transform are numerical;
  2. CLASS-SZ's own 3-D P/P200 accessor evaluated with the Lee22 parameters (and use_broken_pressure active) at the
     profile test points, and the helper get_1e6xdy_from_battaglia_pressure_at_x_z_and_m200c (h and argument order).
Outputs: OUT/class_sz/<name>.npz, OUT/profiles/class_sz_3d_probe.json"""
import faulthandler, json, sys, time
from pathlib import Path
import numpy as np

faulthandler.enable()
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
import run_class_sz as R  # noqa: E402
sys.path.insert(0, C.LEE22_MODELS_DIR)
import lee22_models as LM  # noqa: E402

HIPREC = {"N_samp_fftw": 8192, "n_m_pressure_profile": 300, "n_z_pressure_profile": 400, "n_l_pressure_profile": 600,
          "ndim_redshifts": 400, "ndim_masses": 400, "redshift_epsrel": 1e-4, "mass_epsrel": 1e-4}


def probe_3d():
    from classy_sz import Class
    h = C.PAINT["h"]
    out = {}
    for piece in ("low", "high"):
        p = C.lee22_class_sz_pars(h, piece)
        kw = dict(m_max=C.LEE_CLIP_HINV) if piece == "low" else dict(m_min=C.LEE_CLIP_HINV)
        pars = C.class_sz_base(C.PAINT, **kw); pars.update(p)
        c = Class()
        for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
            c.pars.pop(k, None)
        c.set(pars); c.compute()
        rows = []
        for mphys in (1e13, 5e13, 1e14, 5e14, 1e15, 3e15):
            mh = mphys * h
            if (piece == "low") != (mh < C.LEE_CLIP_HINV):
                continue
            for z in (0.1, 0.5, 1.0, 2.0):
                for x in (0.05, 0.3, 1.0, 2.0, 3.9):
                    v = c.get_pressure_P_over_P_delta_at_x_M_z_b12_200c(
                        x, mh, z, 0., p["P0_B12"], p["xc_B12"], p["beta_B12"], p["alpha_m_P0_B12"], p["alpha_m_xc_B12"],
                        p["alpha_m_beta_B12"], p["alpha_z_P0_B12"], p["alpha_z_xc_B12"], p["alpha_z_beta_B12"],
                        1e14, 0., 0., 0., 0., 0., 0., 1.0, -0.3)
                    ref = LM.lee_Pbar(x, mh, z, which="noc", shape_clip_hinv=C.LEE_CLIP_HINV) / C.PE_OVER_PTH
                    rows.append(dict(M=mphys, z=z, x=x, class_sz=v, lee22_over_0p5176=float(ref), ratio=v / float(ref)))
        out[piece] = rows
        if piece == "low":
            # helper with its Python argument order (z, m, x) -> C (x, z, m); and B12-default parameters in this instance
            try:
                a = c.get_1e6xdy_from_battaglia_pressure_at_x_z_and_m200c(0.5, 1e14, 0.5)
                b = c.get_1e6xdy_from_battaglia_pressure_at_x_z_and_m200c(0.5, 0.5, 1e14)
                out["get_1e6xdy_helper"] = dict(call_z_m_x=a, call_x_z_m=b, note="C signature is (x, z, m); python wrapper "
                                                "passes (z, m, x) through unchanged; the C body also divides m by h twice")
            except Exception as e:  # noqa: BLE001
                out["get_1e6xdy_helper"] = repr(e)
        c.struct_cleanup(); c.empty()
    return out


if __name__ == "__main__":
    P = C.PAINT
    specs = [dict(name="b12_paint_all_hiprec", cosmo=P, prof=dict(C.b12_class_sz_pars(), **HIPREC), extract="b12"),
             dict(name="lee22_paint_low_hiprec", cosmo=P, prof=dict(C.lee22_class_sz_pars(P["h"], "low"), **HIPREC),
                  m_max=C.LEE_CLIP_HINV, extract="lee22_low"),
             # low-z diagnostic (added 2026-09-29 continuation): the ell < 150 1-halo is dominated by z < 0.02, where
             # CLASS-SZ's ln(1+z) pressure-table spacing (first nodes z = 0.0021, 0.016) is coarsest
             dict(name="b12_paint_zmin0p02", cosmo=P, prof=C.b12_class_sz_pars(), z_min=0.02),
             dict(name="b12_paint_zmin0p02_hiprec", cosmo=P, prof=dict(C.b12_class_sz_pars(), **HIPREC), z_min=0.02)]
    for s in specs:
        R.run_one(s)
    t0 = time.time()
    res = probe_3d()
    for piece in ("low", "high"):
        r = np.array([x["ratio"] for x in res[piece]])
        print(piece, "CLASS-SZ 3-D accessor / (Lee22 P_e/P200 / 0.5176): min", r.min(), "max", r.max())
    print("helper", res.get("get_1e6xdy_helper"))
    json.dump(res, open(Path(C.OUT) / "profiles" / "class_sz_3d_probe.json", "w"), indent=1, default=float)
    print("done", time.time() - t0)
