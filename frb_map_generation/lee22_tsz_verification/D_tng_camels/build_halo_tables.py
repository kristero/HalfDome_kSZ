#!/usr/bin/env python
"""Step 2a: per-halo dimensionless profiles on a common x = r/R200c grid, from
  (P) the CAMELS precomputed illstack profiles (27 CV boxes, old snapshots 010-033, z = 2.00-0), and
  (D) our direct computation from raw snapshots (direct/direct_CV_<b>_<new>.npz, subset of boxes/redshifts).

Dimensionless quantities (Lee22 definitions, PHYSICAL units, CAMELS cosmology Om 0.3, Ob 0.049, h 0.6711):
  nbar  = n_e / n200,  n200 = 200 rho_crit(z) f_b / (X_H m_p)
          (P): n_e = rho_gas / (mu_e m_p), full ionisation, mu_e = 1.1364;  (D): n_e = x_e X_H rho/m_p (ElectronAbundance)
  Pbar  = P_e / P200,  P200 = 200 G M200 rho_crit f_b / (2 R200)
          (P): P_e = 0.5176 P_th (full ionisation; validated to 0.2 %);  (D): P_e = n_e k_B T with x_e
  rhob  = rho_gas / (f_b rho_crit(z))        (Battaglia16 normalisation)
  fgas  = M_gas(<R200c) / (f_b M200c),  fe = N_e(<R200c) mu_e m_p / (f_b M200c)  (electron-equivalent)
  Yhat  = Y_sph(<R200c) / (P200 (4 pi/3) R200^3) = 3 int_0^1 Pbar x^2 dx   (sigma_T/m_e c^2 cancels)
Shell averages are converted to point values with the local log slope (profile_tools.point_values) before
interpolation; enclosed quantities use shell sums (partial shell by volume).
Emulated Lee22 bookkeeping (D only): nbar_emul = lee_ne/lee_n200_code, Pbar_emul = lee_pe/lee_P200_code
(comoving shells, physical rho_crit) - expected = truth x a^3 and x a^2.

Output: halo_tables/precomputed_halos.npz, halo_tables/direct_halos.npz
Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python build_halo_tables.py
"""
import glob
import os

import numpy as np

import camels_common as cc
import profile_tools as pt

XG = cc.x_grid(31, 0.03, 3.0)
LOGM_MIN = 12.5
TDIR = os.path.join(cc.OUTDIR, "halo_tables")
os.makedirs(TDIR, exist_ok=True)
C = cc.CAMELS


def precomputed_table():
    rec = {k: [] for k in ("box", "old", "z", "M200", "R200", "ID", "nbar", "Pbar", "rhob", "Tmw", "fgas", "Yhat",
                           "Yhat_2p5", "fgas_2p5")}
    q = np.exp(np.log(1e6) / 25)
    for box in range(27):
        for old in range(10, 34):
            fn = os.path.join(cc.PROFDIR, f"CV_{box}", f"IllustrisTNG_CV_{box}_{old:03d}.hdf5")
            if not os.path.exists(fn):
                continue
            pc = cc.load_precomputed(box, old)
            z = pc["z"]
            rc = C.rho_crit(z)
            sel = np.where(np.log10(pc["M200"]) >= LOGM_MIN)[0]
            for k in sel:
                M, R = pc["M200"][k], pc["R200"][k]
                xc = pc["r_kpc"] / R
                xe = pc["edges_kpc"] / R
                ok = (pc["ncell"][k] > 0) & (pc["rho"][k] > 0)
                nb = cc.ne_n200_full_ion(pc["rho"][k], z)
                Pb = cc.PE_OVER_PTH * pc["Pth"][k] / C.P200(M, z)
                rb = pc["rho"][k] / (C.fb * rc)
                rec["nbar"].append(pt.to_grid(xc, pt.point_values(xc, nb, q, ok), XG))
                rec["Pbar"].append(pt.to_grid(xc, pt.point_values(xc, Pb, q, ok & (Pb > 0)), XG))
                rec["rhob"].append(pt.to_grid(xc, pt.point_values(xc, rb, q, ok), XG))
                rec["Tmw"].append(pt.to_grid(xc, np.where(ok, pc["Tmw"][k], np.nan), XG))
                rec["fgas"].append(pt.enclosed(xe, np.where(ok, rb / 200.0, 0.0), 1.0))
                rec["fgas_2p5"].append(pt.enclosed(xe, np.where(ok, rb / 200.0, 0.0), 2.5) / 2.5 ** 3)
                rec["Yhat"].append(pt.enclosed(xe, np.where(ok, Pb, 0.0), 1.0))
                rec["Yhat_2p5"].append(pt.enclosed(xe, np.where(ok, Pb, 0.0), 2.5))
                for key, v in (("box", box), ("old", old), ("z", z), ("M200", M), ("R200", R), ("ID", pc["ID"][k])):
                    rec[key].append(v)
        print(f"box {box}: {len(rec['M200'])} halo-snapshots so far", flush=True)
    out = {k: np.array(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(TDIR, "precomputed_halos.npz"), xg=XG, **out)
    return out


def direct_table():
    files = sorted(glob.glob(os.path.join(cc.DIRECTDIR, "direct_CV_*_*.npz")))
    keys = ("box", "new", "z", "M200", "R200", "ID", "nbar_xe", "nbar_fullion", "nbar_xe_nosf", "Pbar_xe", "Pbar_th",
            "Pbar_xe_nosf", "rhob", "nbar_emul", "Pbar_emul", "fgas", "fe", "fsf", "Yhat", "Yhat_th", "Yhat_nosf",
            "nbar_xe_volw")
    rec = {k: [] for k in keys}
    for fn in files:
        d = np.load(fn)
        b = int(os.path.basename(fn).split("_")[2])
        new = int(os.path.basename(fn).split("_")[3].split(".")[0])
        z, a, h = float(d["z"]), float(d["a"]), float(d["h"])
        rc = C.rho_crit(z)
        M200 = d["M200_code"] * 1e10 / h
        R200 = d["R200_code"] * a / h
        xe = d["x_edges"]
        xc = np.sqrt(xe[2:] * xe[1:-1])        # log bins 1..40 (bin 0 = centre sphere x < 0.01)
        q = xe[2] / xe[1]
        xl = np.sqrt(d["lee_xlo"][0, 1:] * d["lee_xhi"][0, 1:])
        ql = d["lee_xhi"][0, 1] / d["lee_xlo"][0, 1]
        for j in np.where(np.log10(M200) >= LOGM_MIN)[0]:
            M, R = M200[j], R200[j]
            re = xe * R * cc.KPC
            vol = 4 * np.pi / 3 * (re[1:] ** 3 - re[:-1] ** 3)
            n200 = C.n200(z)
            P200 = C.P200(M, z)
            prof = {
                "nbar_xe": d["true_Ne"][j] / vol / n200,
                "nbar_fullion": d["true_Mgas"][j] * cc.MSUN / (cc.MU_E * cc.M_P) / vol / n200,
                "nbar_xe_nosf": (d["true_Ne"][j] - d["true_Ne_sf"][j]) / vol / n200,
                "Pbar_xe": d["true_PeV"][j] / vol / P200,
                "Pbar_th": cc.PE_OVER_PTH * d["true_PthV"][j] / vol / P200,
                "Pbar_xe_nosf": (d["true_PeV"][j] - d["true_PeV_sf"][j]) / vol / P200,
                "rhob": d["true_Mgas"][j] * cc.MSUN / vol / (C.fb * rc),
            }
            okn = d["true_n"][j] > 0
            for key, v in prof.items():
                ok = okn[1:] & (v[1:] > 0)
                rec[key].append(pt.to_grid(xc, pt.point_values(xc, v[1:], q, ok), XG))
            # volume-weighted cell average (sum n_e V / sum V) as an estimator check
            vw = d["true_Ne"][j] / np.where(d["true_V"][j] > 0, d["true_V"][j] * cc.KPC ** 3, np.nan) / n200
            rec["nbar_xe_volw"].append(pt.to_grid(xc, vw[1:], XG))
            ne_e = d["lee_ne"][j] / float(d["lee_n200_code"])
            pe_e = d["lee_pe"][j] / d["lee_P200_code"][j]
            okl = d["lee_npart"][j] > 0
            rec["nbar_emul"].append(pt.to_grid(xl, pt.point_values(xl, ne_e[1:], ql, okl[1:] & (ne_e[1:] > 0)), XG))
            rec["Pbar_emul"].append(pt.to_grid(xl, pt.point_values(xl, pe_e[1:], ql, okl[1:] & (pe_e[1:] > 0)), XG))
            mb = C.fb * M
            rec["fgas"].append(d["in200_Mgas"][j] / mb)
            rec["fe"].append(d["in200_Ne"][j] * cc.MU_E * cc.M_P / cc.MSUN / mb)
            rec["fsf"].append(d["in200_Mgas_sf"][j] / max(d["in200_Mgas"][j], 1e-30))
            vr = P200 * 4 * np.pi / 3 * (R * cc.KPC) ** 3
            rec["Yhat"].append(d["in200_PeV"][j] / vr)
            rec["Yhat_th"].append(cc.PE_OVER_PTH * d["in200_PthV"][j] / vr)
            rec["Yhat_nosf"].append((d["in200_PeV"][j] - d["in200_PeV_sf"][j]) / vr)
            for key, v in (("box", b), ("new", new), ("z", z), ("M200", M), ("R200", R), ("ID", int(d["ID"][j]))):
                rec[key].append(v)
        print(f"{os.path.basename(fn)}: z={z:.3f}, {np.sum(np.log10(M200) >= LOGM_MIN)} halos >= 10^{LOGM_MIN}", flush=True)
    out = {k: np.array(v) for k, v in rec.items()}
    np.savez_compressed(os.path.join(TDIR, "direct_halos.npz"), xg=XG, **out)
    return out


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["direct", "precomputed"]
    if "direct" in what:
        direct_table()
    if "precomputed" in what:
        precomputed_table()
