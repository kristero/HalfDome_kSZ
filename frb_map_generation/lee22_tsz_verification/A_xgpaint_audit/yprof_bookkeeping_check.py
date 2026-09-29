"""Workstream A, goal 4: numerical re-enactment of group_particles examples/y_prof.cpp (and ne_prof.cpp) bookkeeping
in raw IllustrisTNG snapshot units, for a synthetic halo with a known PHYSICAL pressure/density profile.

TNG units (www.tng-project.org/data/docs/specifications, fetched 2026-09-29 into group_particles_fetch/):
  Coordinates, GroupPos, Group_R_Crit200 : ckpc/h (comoving)       Masses, Group_M_Crit200 : 1e10 Msun/h
  InternalEnergy : (km/s)^2 (specific, physical)                    ElectronAbundance : n_e/n_H
y_prof.cpp (GitHub master 6c6d486, identical to the 2021-06-25 PR #1 version):
  l.85-87   shells from 0.03 to 2.5 x Group_R_Crit200, i.e. in ckpc/h
  l.57-67   shell_vol in (ckpc/h)^3 ; l.101 logR from comoving Rsq
  l.125-126 Y = 4 x XH/(1+3XH+4XH x) (gamma-1) m e  = P_e x V   [1e10 Msun/h (km/s)^2], physical energy
  l.140     pressure /= shell_vol  -> E / V_comoving
  l.160-172 GNewton = 4.30091e4 kpc (km/s)^2/(1e10 Msun); rho_crit = 2.775e-8 (Omega0/a^3 + OmegaLambda) = PHYSICAL
  l.181-183 P200 = 100 G m rho_crit f_b / r with r = Group_R_Crit200 (comoving)
ne_prof.cpp: l.113-121 N_e = x XH m / m_p ; l.135 e_density /= shell_vol (comoving); n200 is not in the public code.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit")
h, Om, OL, Ob = 0.6774, 0.3089, 0.6911, 0.0486
G = 4.30091e4              # kpc (km/s)^2 / (1e10 Msun)   (y_prof.cpp l.160)
RHO0 = 2.775e-8            # 1e10 Msun/h / (kpc/h)^3      (y_prof.cpp l.161)
XH = 0.76
MP = 0.6774 * 1.6726219 / 1.98847 * 1e-67   # proton mass in 1e10 Msun/h (ne_prof.cpp l.118)


def shape_P(x):   # arbitrary smooth truth shape (B12-like)
    s = x / 0.5
    return s ** -0.3 * (1 + s) ** -4.35


def shape_n(x):
    s = x / 0.3
    return s ** -0.3 * (1 + s) ** -3.0


out = {}
for z in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
    a = 1 / (1 + z)
    rho_c = RHO0 * (Om / a ** 3 + OL)            # physical, 1e10 Msun/h per (physical kpc/h)^3
    M = 1.0e4                                   # 1e14 Msun/h in code units
    R_phys = (3 * M / (4 * np.pi * 200 * rho_c)) ** (1 / 3)   # physical kpc/h
    R_code = R_phys / a                          # Group_R_Crit200 [ckpc/h]
    P200_true = 100 * G * M * rho_c * (Ob / Om) / R_phys
    P200_code = 100 * G * M * rho_c * (Ob / Om) / R_code       # y_prof.cpp l.183
    n200_true = 200 * rho_c * (Ob / Om) / (XH * MP)             # printed eq. 9, physical (h-units)
    # shells as in y_prof.cpp l.85-87 (comoving) -> physical edges r = a * r_code
    N = 128
    lo, hi = np.log(0.03 * R_code), np.log(2.5 * R_code)
    edges_code = np.exp(np.linspace(lo, hi, N))                 # shell boundaries idx>=1 (l.64-66)
    ratios_P, ratios_n = [], []
    for i in range(1, N):
        r1c, r2c = edges_code[i - 1], edges_code[i]
        r1, r2 = a * r1c, a * r2c                               # physical kpc/h
        E = quad(lambda r: P200_true * shape_P(r / R_phys) * 4 * np.pi * r * r, r1, r2, epsrel=1e-12)[0]   # sum of P_e V
        Ne = quad(lambda r: n200_true * shape_n(r / R_phys) * 4 * np.pi * r * r, r1, r2, epsrel=1e-12)[0]  # electrons
        V_code = 4 * np.pi / 3 * (r2c ** 3 - r1c ** 3)          # l.66, (ckpc/h)^3
        V_phys = 4 * np.pi / 3 * (r2 ** 3 - r1 ** 3)
        pbar_code = (E / V_code) / P200_code                     # l.140 and l.183
        pbar_true = (E / V_phys) / P200_true
        nbar_code = (Ne / V_code) / n200_true                    # ne_prof.cpp l.135 with a physical n200
        nbar_true = (Ne / V_phys) / n200_true
        ratios_P.append(pbar_code / pbar_true)
        ratios_n.append(nbar_code / nbar_true)
    rp, rn = np.array(ratios_P), np.array(ratios_n)
    out[str(z)] = {"a": a, "Pbar_code_over_true_mean": rp.mean(), "Pbar_spread": float(rp.max() - rp.min()),
                   "a2": a ** 2, "nbar_code_over_true_mean": rn.mean(), "a3": a ** 3,
                   "R200_code_ckpch": R_code, "R200_phys_kpch": R_phys}
    print(f"z={z}: (P/P200)_code/true = {rp.mean():.12f} (a^2 = {a**2:.12f}), (n/n200)_code/true = {rn.mean():.12f} (a^3 = {a**3:.12f})")
json.dump(out, open(OUT / "yprof_bookkeeping_check.json", "w"), indent=1)
