#!/usr/bin/env python3
"""Rough TNG300 halo counts per Lee22 0.2-dex M200c bin (h^-1 Msun) at z = 0, 1, 2: Tinker+08 mass function at
Delta_m = 200/Omega_m(z) (i.e. M200c), Eisenstein & Hu (1998) no-wiggle P(k), Planck15/TNG cosmology, V = (205 h^-1 Mpc)^3.
Purpose: whether the 10^13.13 / 10^13.83 h^-1 Msun rows of Fig. 2d/2e are inside Lee22's fitted bins (> 29 halos) at z = 1, 2.
Calibration anchor: Lee22 Fig. 2 shows 177 and 65 halos in the lowest-c quartile of 13.2-13.4 and 14.0-14.2 at z = 0."""
import numpy as np
from scipy.integrate import quad

OM, OB, H, NS, S8 = 0.3089, 0.0486, 0.6774, 0.9667, 0.8159
V = 205.0 ** 3
RHO_M = 2.775e11 * OM            # h^2 Msun / Mpc^3 -> (h^-1 Msun)/(h^-1 Mpc)^3


def T_eh_nw(k):                  # k in h/Mpc
    omh2, obh2 = OM * H * H, OB * H * H
    th = 2.7255 / 2.7
    s = 44.5 * np.log(9.83 / omh2) / np.sqrt(1 + 10 * obh2 ** 0.75)
    ag = 1 - 0.328 * np.log(431 * omh2) * OB / OM + 0.38 * np.log(22.3 * omh2) * (OB / OM) ** 2
    kk = k * H
    gam = OM * H * (ag + (1 - ag) / (1 + (0.43 * kk * s) ** 4))
    q = k * th ** 2 / gam
    L0 = np.log(2 * np.e + 1.8 * q); C0 = 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q * q)


def growth(z):
    E = lambda a: np.sqrt(OM / a ** 3 + 1 - OM)
    D = lambda a: 2.5 * OM * E(a) * quad(lambda x: 1 / (x * E(x)) ** 3, 0, a)[0]
    return D(1 / (1 + z)) / D(1.0)


k = np.geomspace(1e-4, 1e3, 6000)
P0 = k ** NS * T_eh_nw(k) ** 2


def sigma(R, norm=1.0):
    x = k * R
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(norm * np.trapz(P0 * W ** 2 * k ** 2, k) / (2 * np.pi ** 2))


NORM = (S8 / sigma(8.0)) ** 2
TAB = np.array([[200, .186, 1.47, 2.57, 1.19], [300, .200, 1.52, 2.25, 1.27], [400, .212, 1.56, 2.05, 1.34],
                [600, .218, 1.61, 1.87, 1.45], [800, .248, 1.87, 1.59, 1.58], [1200, .255, 2.13, 1.51, 1.80],
                [1600, .260, 2.30, 1.46, 1.97], [2400, .260, 2.53, 1.44, 2.24], [3200, .260, 2.66, 1.41, 2.44]])


def dndlnM(M, z):
    omz = OM * (1 + z) ** 3 / (OM * (1 + z) ** 3 + 1 - OM)
    dm = 200 / omz
    ld = np.log10(TAB[:, 0])
    A, a, b, c = (np.interp(np.log10(dm), ld, TAB[:, i]) for i in range(1, 5))
    alpha = 10 ** (-(0.75 / np.log10(dm / 75)) ** 1.2)
    A *= (1 + z) ** -0.14; a *= (1 + z) ** -0.06; b *= (1 + z) ** -alpha
    g = growth(z)
    R = lambda m: (3 * m / (4 * np.pi * RHO_M)) ** (1 / 3)
    s = g * sigma(R(M), NORM)
    eps = 1e-3
    dlns = (np.log(g * sigma(R(M * (1 + eps)), NORM)) - np.log(g * sigma(R(M * (1 - eps)), NORM))) / (2 * eps)
    f = A * ((s / b) ** -a + 1) * np.exp(-c / s ** 2)
    return f * RHO_M / M * (-dlns) * M / M * 1.0  # dn/dlnM = f rho/M |dln sigma/dlnM|


def count(lo, hi, z):
    lm = np.linspace(lo, hi, 21)
    vals = [dndlnM(10 ** l, z) for l in lm]
    return V * np.trapz(vals, lm * np.log(10))


if __name__ == "__main__":
    print("sigma8 check:", sigma(8.0, NORM))
    print("bin (h^-1 Msun)   N(z=0)  N(z=0.2)  N(z=0.5)  N(z=1)  N(z=1.5)  N(z=2)")
    for lo in np.arange(13.0, 14.6, 0.2):
        print("  %.1f-%.1f  " % (lo, lo + .2) + "  ".join("%7.0f" % count(lo, lo + .2, z) for z in (0, .2, .5, 1, 1.5, 2)))
