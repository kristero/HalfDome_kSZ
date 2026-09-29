#!/usr/bin/env python3
"""Tabulate the per-(z, M, ell) halo-model ingredients of kSZ (tau_l, 3R200c sphere) and tSZ (y_l, 4R200c full-LOS
aperture) for Lee22 literal / Lee22 xgpnative / Battaglia16 density and Lee22 / Battaglia12 pressure.

usage: run_budget.py {validation,physical} [--nz N] [--nm N] [--procs P]
  validation: halo range of the painted HalfDome maps, 6.557e12 <= M200c/Msun <= 3.785e15, 0.00214 <= z <= 3.855
  physical:   1e11 <= M200c/Msun <= 1e16, 0.005 <= z <= 4 (Lee22 fits extrapolated beyond z = 2 and 1e13-10^14.8/h)
Output: <OUT>/tables_<run>.npz
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import szhm as S  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget")
DENSITY_MODELS = ("lee22_literal", "b16")          # xgpnative = constant x literal, applied in analysis
PRESSURE_MODELS = ("lee22p", "b12")
ELL_SPECIAL = np.array([30, 100, 300, 1000, 2000, 3000, 5000, 8000], float)
RUNS = {"validation": dict(m_lo=6.55711024486e12, m_hi=3.78523625595e15, z_lo=0.00214159489, z_hi=3.855427742),
        "physical": dict(m_lo=1e11, m_hi=1e16, z_lo=0.005, z_hi=4.0)}

G = {}


def worker(iz):
    z = G["z"][iz]
    M = G["M"]
    ell = G["ell"]
    cosmo = S.PAINT_COSMO
    r200_mpc = S.r200c_mpc(cosmo, M, z)
    DA = G["chi"][iz] / (1 + z)
    theta = r200_mpc / DA                              # small-angle theta200c [rad]
    q = ell[None, :] * theta[:, None]                  # (nM, nell)
    r200_cm = r200_mpc * S.MPC_CM
    out = {}
    for name in DENSITY_MODELS:
        amp, shape = S.density_shape(name, cosmo, M, z)
        u3 = S.sphere_ft(shape, S.KSZ_SPHERE, q)
        out["tau_" + name] = S.SIGMA_T * (amp * r200_cm * theta ** 2)[:, None] * u3
        out["Ne3_" + name] = amp * r200_cm ** 3 * S.enclosed(shape, S.KSZ_SPHERE)
        out["Ne1_" + name] = amp * r200_cm ** 3 * S.enclosed(shape, S.FRB_SPHERE)
    for name in PRESSURE_MODELS:
        P0, xc, beta = S.pressure_params(name, cosmo, M, z)
        AP = P0 * S.p200_cgs(cosmo, M, z)
        Yt = S.aperture_hankel(G["table"], xc, beta, S.TSZ_APERTURE, q)
        Y0 = S.aperture_hankel(G["table"], xc, beta, S.TSZ_APERTURE, np.zeros((len(M), 1)), n=4096)[:, 0]
        pref = S.SIGMA_T / S.ME_C2 * AP * r200_cm * theta ** 2
        out["y_" + name] = pref[:, None] * Yt
        out["y0_" + name] = pref * Y0
        # thermal-energy proxy: integral of P_e over the R200c sphere [erg]
        sh = S.RowShape("lee22", xc=xc, beta=beta)
        out["Eint1_" + name] = AP * r200_cm ** 3 * S.enclosed(sh, 1.0)
    return iz, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=tuple(RUNS))
    ap.add_argument("--nz", type=int, default=160)
    ap.add_argument("--nm", type=int, default=0)
    ap.add_argument("--nell", type=int, default=60)
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ksz-sphere", type=float, default=S.KSZ_SPHERE)
    ap.add_argument("--dlog10m", type=float, default=0.02)
    a = ap.parse_args()
    S.KSZ_SPHERE = a.ksz_sphere
    r = RUNS[a.run]
    nm = a.nm or int(np.ceil(np.log10(r["m_hi"] / r["m_lo"]) / a.dlog10m)) + 1
    lnM = np.linspace(np.log(r["m_lo"]), np.log(r["m_hi"]), nm)
    M = np.exp(lnM)
    z = np.geomspace(r["z_lo"], r["z_hi"], a.nz)
    ell = np.unique(np.concatenate([np.round(np.geomspace(10, 12000, a.nell)), ELL_SPECIAL]))
    cosmo = S.PAINT_COSMO
    t0 = time.time()
    k, p = cosmo.power()
    print("CAMB done, sigma8 check", float(np.sqrt(cosmo._sigma2_R(k, p, 8 / cosmo.h))), flush=True)
    chi = cosmo.chi(z)
    Hc = cosmo.H_over_c(z)                        # 1/Mpc
    v2, D, f = S.vlos2_kms2(cosmo, z)
    dn = np.array([S.tinker08_dndlnm(cosmo, M, zz) for zz in z])       # (nz, nM) comoving Mpc^-3
    bias = np.array([S.tinker10_bias(cosmo, M, zz) for zz in z])
    print("mass function done {:.1f}s".format(time.time() - t0), flush=True)
    G.update(z=z, M=M, ell=ell, chi=chi, table=S.ProjectedGNFWTable())
    print("projected table done {:.1f}s".format(time.time() - t0), flush=True)
    res = {}
    with mp.Pool(a.procs) as pool:
        for n_done, (iz, out) in enumerate(pool.imap_unordered(worker, range(len(z)))):
            for key, val in out.items():
                if key not in res:
                    res[key] = np.empty((len(z),) + val.shape)
                res[key][iz] = val
            if n_done % 20 == 0:
                print("  {} / {} z done, {:.1f}s".format(n_done + 1, len(z), time.time() - t0), flush=True)
    meta = dict(run=a.run, grid=r, nz=len(z), nm=nm, nell=len(ell), cosmology=dict(
        h=cosmo.h, omega_m=cosmo.omega_m, omega_b=cosmo.omega_b, sigma8=cosmo.sigma8, n_s=cosmo.n_s),
        ksz_sphere_r200c=S.KSZ_SPHERE, tsz_aperture_r200c=S.TSZ_APERTURE, frb_sphere_r200c=S.FRB_SPHERE,
        xgpnative_factor=S.XGPNATIVE_FACTOR, seconds=time.time() - t0,
        units=dict(tau_l="sr (angular FT of tau)", y_l="sr", Ne="electrons", Eint1="erg (int P_e dV inside R200c)",
                   dn="comoving Mpc^-3 per ln M", chi="comoving Mpc", Hc="1/Mpc", v2="(km/s)^2 1-D linear"))
    kk, pp = cosmo.power()
    np.savez_compressed(OUT / "tables_{}{}.npz".format(a.run, a.tag), z=z, M=M, ell=ell, chi=chi, Hc=Hc, v2=v2, D=D, f=f,
                        dn=dn, bias=bias, k_lin=kk, p_lin0=pp, meta=json.dumps(meta), **res)
    print("saved; total {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
