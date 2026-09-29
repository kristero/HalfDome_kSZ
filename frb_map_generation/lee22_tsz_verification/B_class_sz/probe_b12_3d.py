#!/usr/bin/env python3
"""CLASS-SZ's own 3-D Battaglia12 pressure accessors against lee22_models (= XGPaint's B12), added in the 2026-09-29
continuation (run after run_class_sz_extra.py; one classy_sz process at a time).

  get_pressure_P_over_P_delta_at_x_M_z_b12_200c(x, m[Msun/h], z)   -> P0 (x/xc)^-0.3 (1+x/xc)^-beta   (the THERMAL fit shape)
  get_P_delta_at_m_and_z_b12(m[Msun/h], z)                            -> P_200 [eV cm^-3] = 0.5176 x Battaglia thermal P200
so CLASS-SZ's electron pressure is P_e(r) = P_200 x P/P_200. Compared with lee22_models.b12_Pe_over_P200 x Cosmo.P200
(XGPaint reproduction: P_e = 0.5176 P0 P200 g(x), P200 = 200 G M rho_cr f_b / (2 R200), physical R200, f_b = Omega_b/Omega_m).
Paint cosmology (h = 0.68, Omega_b = 0.049, Omega_c = 0.261, massless neutrinos).
Output: OUT/profiles/class_sz_b12_3d_probe.json
"""
import faulthandler, json, sys
from pathlib import Path
import numpy as np

faulthandler.enable()
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
import common as C  # noqa: E402
sys.path.insert(0, C.LEE22_MODELS_DIR)
import lee22_models as LM  # noqa: E402

EV = 1.602176634e-12      # erg

if __name__ == "__main__":
    from classy_sz import Class
    P = C.PAINT; h = P["h"]
    pars = C.class_sz_base(P); pars.update(C.b12_class_sz_pars())
    c = Class()
    for k in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
        c.pars.pop(k, None)
    c.set(pars); c.compute()
    cos = LM.Cosmo("paint", P["Omega_b"] + P["Omega_c"], P["Omega_b"], h)
    rows = []
    for lm in (13.0, 13.5, 14.0, 14.5, 15.0):
        mphys = 10 ** lm
        for z in (0.1, 0.5, 1.0, 2.0):
            p200_cs = c.get_P_delta_at_m_and_z_b12(mphys * h, z)                 # eV cm^-3 (electron)
            p200_th = cos.P200(mphys, z) / EV                                      # eV cm^-3 (thermal Battaglia)
            r200_cs = c.get_r_delta_of_m_delta_at_z(200, mphys * h, z)            # Mpc/h (physical)
            r200_py = cos.r_delta(mphys, z) / LM.MPC * h
            for x in (0.02, 0.05, 0.1, 0.3, 1.0, 2.0, 3.9):
                shape_cs = c.get_pressure_P_over_P_delta_at_x_M_z_b12_200c(x, mphys * h, z)
                shape_py = float(LM.b12_Pth_over_P200(x, mphys, z))
                pe_cs = p200_cs * shape_cs
                pe_py = float(LM.b12_Pe_over_P200(x, mphys, z)) * p200_th
                rows.append(dict(log10M200c_msun=lm, z=z, x=x, shape_class_sz=shape_cs, shape_xgpaint=shape_py,
                                 shape_ratio=shape_cs / shape_py, P200_class_sz_eV_cm3=p200_cs,
                                 P200_class_sz_over_0p5176_P200th=p200_cs / (C.PE_OVER_PTH * p200_th),
                                 Pe_class_sz_keV_cm3=pe_cs * 1e-3, Pe_xgpaint_keV_cm3=pe_py * 1e-3, Pe_ratio=pe_cs / pe_py,
                                 R200_class_sz_over_python=r200_cs / r200_py))
    sr = np.array([r["shape_ratio"] for r in rows]); pr = np.array([r["Pe_ratio"] for r in rows])
    rr = np.array([r["R200_class_sz_over_python"] for r in rows])
    summ = dict(shape_ratio_min=sr.min(), shape_ratio_max=sr.max(), Pe_ratio_min=pr.min(), Pe_ratio_max=pr.max(),
                R200_ratio_min=rr.min(), R200_ratio_max=rr.max(), f_b_class_sz=c.get_f_b(), Omega_m=c.Omega_m(),
                note="P_e ratio residual = CLASS-SZ rho_crit(z) (includes radiation) vs matter+Lambda in lee22_models, "
                     "and CLASS-SZ constants (2.61051e-18 carries 0.5176)")
    print(json.dumps(summ, indent=1, default=float))
    json.dump(dict(summary=summ, rows=rows), open(Path(C.OUT) / "profiles" / "class_sz_b12_3d_probe.json", "w"), indent=1,
              default=float)
    c.struct_cleanup(); c.empty()
