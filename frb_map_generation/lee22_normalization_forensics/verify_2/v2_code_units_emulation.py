"""verify_2 / test T4: emulate the public group_particles y_prof.cpp / ne_prof.cpp bookkeeping on a
synthetic halo with KNOWN physical n_e and P_e, written in TNG code units, and compare the code's
(P/P200, n/n200) with the true physical ratios. Formulas transcribed from the fresh clone of
github.com/leanderthiele/group_particles (HEAD 6c6d486, 2024-04-20; y_prof.cpp lines 118-126, 160-183;
ne_prof.cpp lines 52-62, 113-121; shell_vol identical in both).

Synthetic halo: uniform sphere, M200c = 10^13.61 Msun/h, fully ionised (x = 1.157 = n_e/n_H for X_H=0.76),
gas mass = f_b M200c inside R200c, T = 10^7 K. Physical truth computed in cgs.
"""
import numpy as np

h, Om, Ob, OL = 0.6774, 0.3089, 0.0486, 0.6911
XH, gam = 0.76, 5 / 3
mp, msun, kpc, G, kB = 1.6726219e-24, 1.98847e33, 3.0856775814913673e21, 6.67430e-8, 1.380649e-16
H0s = 100 * h * 1e5 / (1e3 * kpc)

out = []
for z in (0.0, 0.5, 1.0, 2.0):
    a = 1 / (1 + z)
    E2 = Om / a ** 3 + OL
    rho_c = 3 * H0s ** 2 * E2 / (8 * np.pi * G)                     # g/cm^3 physical
    M = 10 ** 13.61 / h * msun                                        # g
    R = (3 * M / (4 * np.pi * 200 * rho_c)) ** (1 / 3)                # cm physical
    Mgas = Ob / Om * M
    x_e = 1 + 2 * (1 - XH) / (4 * XH)                                  # n_e/n_H fully ionised
    Vphys = 4 / 3 * np.pi * R ** 3
    ne_true = x_e * XH * Mgas / mp / Vphys                            # cm^-3
    mu = 4 / (3 + 5 * XH + 4 * XH * (x_e - 1) * 0) if False else 4 / (1 + 3 * XH + 4 * XH * x_e)
    T = 1e7
    u = kB * T / ((gam - 1) * mu * mp)                                # specific internal energy, cm^2/s^2
    Pe_true = ne_true * kB * T
    P200_true = 200 * G * M * rho_c / (2 * R) * Ob / Om
    n200_true = 200 * rho_c / (XH * mp) * Ob / Om
    # ---- code units -------------------------------------------------------------
    m_code = Mgas / msun / 1e10 * h                                   # 1e10 Msun/h
    e_code = u / 1e10                                                 # (km/s)^2
    R_code = R / kpc * h / a                                          # ckpc/h  (Group_R_Crit200)
    M_code = M / msun / 1e10 * h                                      # 1e10 Msun/h (Group_M_Crit200)
    Y = 4.0 * x_e * XH / (1.0 + 3.0 * XH + 4.0 * XH * x_e) * (gam - 1.0) * m_code * e_code
    shell = 4.0 * np.pi / 3.0 * R_code ** 3                           # sphere = sum of shells, (ckpc/h)^3
    P_code = Y / shell
    GN, rho_c0 = 4.30091e4, 2.775e-8
    P200_code = 100.0 * GN * M_code * rho_c0 * (Om / a ** 3 + OL) * Ob / Om / R_code
    mprot = 0.6774 * 1.6726219 / 1.98847 * 1e-67
    ne_code = x_e * XH * m_code / mprot / shell                       # electrons per (ckpc/h)^3
    # n200 is NOT in the public code; two readings:
    n200_code_h = 200 * rho_c0 * (Om / a ** 3 + OL) * Ob / Om / (XH * mprot)   # per (kpc/h)^3 physical
    n200_kpc = n200_true * kpc ** 3                                            # per kpc^3 physical
    rP = (P_code / P200_code) / (Pe_true / P200_true)
    rn_h = (ne_code / n200_code_h) / (ne_true / n200_true)
    rn_k = (ne_code / n200_kpc) / (ne_true / n200_true)
    out.append(f"z={z:.1f}: (P/P200)code/true = {rP:.5f} (a^2 = {a**2:.5f});  (n/n200)code/true: n200 in (kpc/h)^-3 "
               f"{rn_h:.5f} (a^3={a**3:.5f}), n200 in kpc^-3 {rn_k:.5f} (a^3/h^3={a**3/h**3:.5f})")
print("\n".join(out))
open("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_2/T4_code_units_emulation.txt", "w").write("\n".join(out) + "\n")
