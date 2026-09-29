#!/usr/bin/env python3
"""Numerical check of the unit bookkeeping of leanderthiele/group_particles examples/y_prof.cpp and
examples/ne_prof.cpp (public code; PR #1 by GitHub user bkklee). A uniform gas sphere of known
physical density/temperature is expressed in TNG code units (ckpc/h, 1e10 Msun/h, (km/s)^2), passed
through the code's formulas (shell volume in code units; P200 = 100 G m rho_crit(z) Ob/Om / r with
m=Group_M_Crit200 [1e10 Msun/h], r=Group_R_Crit200 [ckpc/h], rho_crit(z) = 2.775e-8 E^2 [1e10 Msun/h (kpc/h)^-3]),
and compared with the physically correct ratios. n200 is NOT in the public code; two natural variants are tested."""
import numpy as np
h, Om, Ob = 0.6774, 0.3089, 0.0486
G_code = 4.30091e4                     # kpc (km/s)^2 / (1e10 Msun), physical, no h (as in y_prof.cpp)
rho_c0_code = 2.775e-8                 # 1e10 Msun/h / (kpc/h)^3
mprot_code = 0.6774 * 1.6726219 / 1.98847 * 1e-67   # 1e10 Msun/h (ne_prof.cpp)
mp_msun = 1.6726219e-24 / 1.98847e33
XH, gam = 0.76, 5 / 3
for z in (0.0, 0.5, 1.0, 2.0):
    a = 1 / (1 + z); E2 = Om * (1 + z) ** 3 + 1 - Om
    rho_c_phys = 2.775e2 * h ** 2 * E2           # Msun / kpc^3 (physical)
    M = 1e14                                     # Msun
    R = (3 * M / (4 * np.pi * 200 * rho_c_phys)) ** (1 / 3)   # physical kpc
    P200_true = 100 * 4.30091e-6 * M * rho_c_phys * Ob / Om / R          # Msun (km/s)^2 / kpc^3
    n200_true = 200 * rho_c_phys * Ob / Om / (XH * mp_msun)             # kpc^-3 physical
    # a gas shell: physical volume V (kpc^3), gas mass Mg (Msun), u (km/s)^2, x_e
    V, Mg, u, xe = 1.0e6, 3.0e9, 5.0e5, 1.158
    ne_true = xe * XH * Mg / mp_msun / V
    Pe_true = 4 * xe * XH / (1 + 3 * XH + 4 * XH * xe) * (gam - 1) * Mg * u / V
    # code units
    m_code, V_code = Mg * h / 1e10, V * (h / a) ** 3
    M_code, R_code = M * h / 1e10, R * h / a
    ne_code = xe * XH * m_code / mprot_code / V_code                    # per (ckpc/h)^3
    Pe_code = 4 * xe * XH / (1 + 3 * XH + 4 * XH * xe) * (gam - 1) * m_code * u / V_code
    P200_code = 100 * G_code * M_code * rho_c0_code * E2 * Ob / Om / R_code
    n200_hunits = 200 * rho_c0_code * E2 * Ob / Om / (XH * mprot_code)  # (kpc/h)^-3 physical, h-units
    n200_phys_noh = n200_true                                            # kpc^-3 physical
    print(f"z={z}: (Pe/P200)_code/(true) = {Pe_code/P200_code/(Pe_true/P200_true):.5f}  [(1+z)^-2 = {(1+z)**-2:.5f}]   "
          f"(ne/n200)_code/(true): n200 in h-units {ne_code/n200_hunits/(ne_true/n200_true):.5f} [(1+z)^-3={(1+z)**-3:.5f}], "
          f"n200 in kpc^-3 {ne_code/n200_phys_noh/(ne_true/n200_true):.5f} [h^-3 (1+z)^-3={h**-3*(1+z)**-3:.5f}]")
