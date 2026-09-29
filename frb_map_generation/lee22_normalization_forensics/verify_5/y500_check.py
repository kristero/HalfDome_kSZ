#!/usr/bin/env python3
"""verify_5: independent Y_sph(<R500c) of the Lee22 no-c electron-pressure fit vs Arnaud+10 UPP (electron pressure),
z = 0 and 0.3, TNG cosmology (h=0.6774); M200c -> M500c with an NFW of Duffy+08 (full, M200c) concentration."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from astropy.cosmology import FlatLambdaCDM
h = 0.6774; cos = FlatLambdaCDM(H0=100 * h, Om0=0.3089, Ob0=0.0486, Tcmb0=2.7255)
fb = 0.0486 / 0.3089
G, MSUN, MPC, KEV = 6.6743e-8, 1.98841e33, 3.0856775814913673e24, 1.602176634e-9
def rc(z): return cos.critical_density(z).to("g/cm3").value
def nfw_m(r_over_rs): return np.log1p(r_over_rs) - r_over_rs / (1 + r_over_rs)
out = []
for z in (0.0, 0.3):
    for lmh in (14.0, 14.5, 15.0):
        M200 = 10 ** lmh / h
        c200 = 5.71 * (M200 / (2e12 / h)) ** -0.084 * (1 + z) ** -0.47      # Duffy+08 full sample, M200c
        R200 = (3 * M200 * MSUN / (4 * np.pi * 200 * rc(z))) ** (1 / 3)
        rs = R200 / c200
        f = lambda R: M200 * nfw_m(R / rs) / nfw_m(c200) * MSUN / (4 / 3 * np.pi * R ** 3) - 500 * rc(z)
        R500 = brentq(f, 0.2 * R200, R200)
        M500 = 500 * rc(z) * 4 / 3 * np.pi * R500 ** 3 / MSUN
        # Lee22 Table 7 (Mcut pivot, h^-1), P200 = 200 G M rho_c fb / (2 R200)
        mc = 10 ** 13.60 / h; rr = M200 / mc; rsm = min(M200, 10 ** 14.8 / h) / mc
        P0 = 2.8 * (rr ** 1.08 if rr < 1 else rr ** 0.80) * (1 + z) ** -1.89
        xc = 2.10 * rsm ** -0.35 * (1 + z) ** 0.53; be = 9.4 * rsm ** -0.06 * (1 + z) ** 0.60
        P200 = 200 * G * M200 * MSUN * rc(z) * fb / (2 * R200)
        Ylee = quad(lambda r: 4 * np.pi * r * r * P0 * P200 * (r / R200 / xc) ** -0.3 * (1 + r / R200 / xc) ** -be, 0, R500, limit=200)[0]
        # Arnaud+10 UPP (electron pressure), h70 = h/0.7
        h70 = h / 0.7; E = cos.efunc(z)
        P500 = 1.65e-3 * E ** (8 / 3) * (M500 / (3e14 / h70)) ** (2 / 3 + 0.12) * h70 ** 2 * KEV   # erg cm^-3
        Pu0, c5, ga, al, bu = 8.403 * h70 ** -1.5, 1.177, 0.3081, 1.0510, 5.4905
        pu = lambda x: Pu0 / ((c5 * x) ** ga * (1 + (c5 * x) ** al) ** ((bu - ga) / al))
        Yupp = quad(lambda r: 4 * np.pi * r * r * P500 * pu(r / R500), 0, R500, limit=200)[0]
        out.append((z, lmh, c200, M500 / M200, Ylee / Yupp))
        print("z=%.1f log10(M200c h)=%.1f c200=%.2f M500/M200=%.3f  Y500(Lee22)/Y500(UPP)=%.3f" % out[-1])
