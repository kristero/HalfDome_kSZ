#!/usr/bin/env python3
"""Fine truncation scan of the catalogue one-halo kSZ (B16, XGPaint conventions; see onehalo_variants.py for the validated
formulae). Saves per-cell amplitudes so any combination (truncation X, WebSky-style uniform-sphere compensation, mass/angle
cut, n_e normalisation) can be formed afterwards:
  a_halo[cell, iX, ell]   sphere-truncated profile transform for X in XS
  a_sph[cell, iv, ell]    uniform-sphere transforms (mass M200m or M200c, mean density 3 or 4 rho_m), XGPaint electrons/kg
  Ne[cell, iX]            electrons inside X R200c
Runs with 3 worker processes."""
import sys
import numpy as np
import h5py
from multiprocessing import Pool
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/painting_geometry")
from onehalo_variants import (HIST, OUT, b16_params, r200c_m, dA, MPC, rho_c, NE_PER_KG, f_b, gnfw, simpson_weights,
                              SIGMA_T, MSUN, Om, m200m_from_m200c)
from scipy.special import spherical_jn

XS = np.arange(0.5, 4.01, 0.25)
SPH = (("M200m", 3.0), ("M200m", 4.0), ("M200c", 3.0), ("M200c", 4.0))
ELLS = np.unique(np.concatenate([np.exp(np.linspace(np.log(30.0), np.log(8192.0), 80)),
                                 [300.0, 500.0, 600.0, 1000.0, 2000.0, 3000.0, 5000.0]]))


def one_cell(args):
    lm, z = args
    M = 10**lm
    P0, xc, al, be, ga = b16_params(M, z)
    R = r200c_m(M, z)
    d = dA(z) * MPC
    th = np.arctan(R / d)
    A = NE_PER_KG * f_b * rho_c(z) * P0
    n = int(np.clip(ELLS.max() * th * 4.0 / 0.25, 2048, 65536))
    n = (n // 32 + 1) * 32 + 1
    x = np.linspace(0, 4.0, n)
    hx = np.zeros(n)
    hx[1:] = x[1:] ** 2 * gnfw(x[1:], xc, al, be, ga)
    S = np.sinc(np.outer(ELLS * th, x) / np.pi)
    pref = SIGMA_T / d**2 * 4 * np.pi * R**3 * A
    a_h = np.zeros((len(XS), len(ELLS)))
    Ne = np.zeros(len(XS))
    for iX, X in enumerate(XS):
        j = int(round(X / 4.0 * (n - 1)))
        w = simpson_weights(j + 1, x[1] - x[0])
        a_h[iX] = pref * (S[:, : j + 1] @ (hx[: j + 1] * w))
        Ne[iX] = 4 * np.pi * R**3 * A * (hx[: j + 1] * w).sum()
    M200m = m200m_from_m200c(M, z)
    rho_m = Om * rho_c(0) * (1 + z) ** 3
    a_s = np.zeros((len(SPH), len(ELLS)))
    for iv, (mdef, od) in enumerate(SPH):
        Mc = M200m if mdef == "M200m" else M
        Rs = (Mc * MSUN / (4 * np.pi / 3 * od * rho_m)) ** (1 / 3)
        q = ELLS * Rs / d
        a_s[iv] = SIGMA_T * f_b * Mc * MSUN * NE_PER_KG / d**2 * 3 * spherical_jn(1, q) / q
    return a_h, a_s, Ne, th, M200m / M, R, d


def main():
    with h5py.File(HIST, "r") as f:
        le = f["logm_edges"][:]; ze = f["z_edges"][:]; cnt = f["count"][:]; v2 = f["sum_vlos_over_c_squared"][:]
    cells = np.argwhere(cnt > 0)
    lmz = [(0.5 * (le[im] + le[im + 1]), 0.5 * (ze[iz] + ze[iz + 1])) for iz, im in cells]
    # interleave so the three workers share slow low-z cells
    with Pool(3) as p:
        res = p.map(one_cell, lmz, chunksize=8)
    a_h = np.array([r[0] for r in res]); a_s = np.array([r[1] for r in res]); Ne = np.array([r[2] for r in res])
    th = np.array([r[3] for r in res]); mm = np.array([r[4] for r in res])
    R = np.array([r[5] for r in res]); dd = np.array([r[6] for r in res])
    lm = np.array([q[0] for q in lmz]); zz = np.array([q[1] for q in lmz])
    np.savez(OUT + "/onehalo_scan_amplitudes.npz", ells=ELLS, XS=XS, sph=np.array(["%s_%g" % s for s in SPH]),
             a_halo=a_h, a_sph=a_s, Ne=Ne, theta200=th, m200m_over_m200c=mm, R200c_m=R, dA_m=dd, logm=lm, z=zz,
             count=cnt[cells[:, 0], cells[:, 1]], sum_v2=v2[cells[:, 0], cells[:, 1]])
    print("saved", a_h.shape, a_s.shape)


if __name__ == "__main__":
    main()
