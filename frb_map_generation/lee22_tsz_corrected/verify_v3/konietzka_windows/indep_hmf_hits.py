#!/usr/bin/env python3
"""Independent (CAMB + own Tinker08) estimate of TNG300 cluster counts and lightcone hit fractions (verify_v3).
Does not import the repo engine. Writes a JSON to the verify output dir."""
import json
from pathlib import Path

import camb
import numpy as np
from scipy import integrate, interpolate

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_verify_v3_20261001/konietzka_windows")
h, Om, Ob, s8, ns = 0.6774, 0.3089, 0.0486, 0.8159, 0.9667
OL = 1 - Om
L_BOX = 205.0 / h
V_BOX = L_BOX ** 3
C_KMS = 299792.458
RHO_CRIT0 = 2.775e11 * h ** 2          # Msun / Mpc^3
RHO_M0 = Om * RHO_CRIT0

pars = camb.CAMBparams()
pars.set_cosmology(H0=100 * h, ombh2=Ob * h * h, omch2=(Om - Ob) * h * h, mnu=0.0, omk=0)
pars.InitPower.set_params(ns=ns, As=2.1e-9)
pars.set_matter_power(redshifts=[0.0], kmax=50.0)
pars.NonLinear = camb.model.NonLinear_none
res = camb.get_results(pars)
kh, _, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=40, npoints=2000)
k = kh * h                      # 1/Mpc
P = pk[0] / h ** 3              # Mpc^3


def sigma_R(R):
    R = np.atleast_1d(R)
    x = np.outer(R, k)
    W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(integrate.simpson(P * W ** 2 * k ** 2 / (2 * np.pi ** 2), x=k, axis=1))


norm = s8 / sigma_R(8 / h)[0]


def sigma_M(M):
    return norm * sigma_R((3 * M / (4 * np.pi * RHO_M0)) ** (1 / 3))


def E(z):
    return np.sqrt(Om * (1 + z) ** 3 + OL)


def growth(z):
    # LCDM linear growth (Heath integral), normalised to 1 at z=0
    def D(zz):
        a = 1 / (1 + zz)
        f = lambda aa: 1 / (aa * E(1 / aa - 1)) ** 3
        return 2.5 * Om * E(zz) * integrate.quad(f, 1e-6, a)[0]
    return D(z) / D(0.0)


def chi(z):
    return C_KMS / (100 * h) * integrate.quad(lambda zz: 1 / E(zz), 0, z)[0]


TD = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200.])
TA = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
Ta = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
Tb = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
Tc = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
spl = {n: interpolate.CubicSpline(np.log10(TD), v) for n, v in (("A", TA), ("a", Ta), ("b", Tb), ("c", Tc))}


def dndlnM_200c(M, z):
    """Tinker08 with Delta_m = 200/Omega_m(z); M is M200c [Msun]. (Treat M200c directly as the Tinker mass at
    Delta_m, as is standard.)"""
    Omz = Om * (1 + z) ** 3 / E(z) ** 2
    Dm = 200 / Omz
    ld = np.log10(Dm)
    A0, a0, b0, c0 = (float(spl[n](ld)) for n in "Aabc")
    alpha = 10 ** (-(0.75 / np.log10(Dm / 75)) ** 1.2)
    A, a, b, c = A0 * (1 + z) ** -0.14, a0 * (1 + z) ** -0.06, b0 * (1 + z) ** -alpha, c0
    lm = np.log(M)
    s = sigma_M(M) * growth(z)
    dlnsinv = -np.gradient(np.log(s), lm)
    f = A * ((s / b) ** (-a) + 1) * np.exp(-c / s ** 2)
    return f * RHO_M0 / M * dlnsinv


def r200c_com(M, z):
    rc = RHO_CRIT0 * E(z) ** 2
    return (3 * M / (4 * np.pi * 200 * rc)) ** (1 / 3) * (1 + z)


def window(z, lo, hi, n=400):
    lm = np.linspace(np.log(lo), np.log(hi), n)
    M = np.exp(lm)
    dn = dndlnM_200c(M, z)
    N = integrate.trapezoid(dn, lm)
    area = integrate.trapezoid(dn * np.pi * r200c_com(M, z) ** 2, lm)
    return N, area


SNAPS = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5])


def intervals(z_s, mid="z"):
    if mid == "z":
        mids = np.concatenate([[0.0], [chi(0.5 * (a + b)) for a, b in zip(SNAPS[:-1], SNAPS[1:])], [np.inf]])
    else:
        c = np.array([chi(z) for z in SNAPS])
        mids = np.concatenate([[0.0], 0.5 * (c[1:] + c[:-1]), [np.inf]])
    cs = chi(z_s)
    return [(z, a, min(b, cs)) for z, a, b in zip(SNAPS, mids[:-1], mids[1:]) if min(b, cs) > a]


out = {}
A_HI = 0.944
for lab, (lo, hi) in {"1e14_1e15": (1e14, 1e15), "9e14_9e15": (9e14, 9e15), "1e15_1e16": (1e15, 1e16),
                      "ge1e14": (1e14, 1e16), "ge1e15": (1e15, 1e16), "1e13_1e14": (1e13, 1e14),
                      "1e13_9e14": (1e13, 9e14), "1e13_1e15": (1e13, 1e15), "9e14_1e15": (9e14, 1e15)}.items():
    r = {}
    for z in (0.0, 0.5, 1.0):
        N, a = window(z, lo, hi)
        r[f"Nbox_tinker_z{z}"] = N * V_BOX
    for z_s in (0.1, 0.5, 1.0):
        for mid in ("z", "chi"):
            nb = sum(window(z, lo, hi)[1] * (b - a) for z, a, b in intervals(z_s, mid))
            r[f"hit_snap_{mid}mid_zs{z_s}_tinker"] = 1 - np.exp(-nb)
            r[f"hit_snap_{mid}mid_zs{z_s}_x0.944"] = 1 - np.exp(-A_HI * nb)
            r[f"nbar_snap_{mid}mid_zs{z_s}_tinker"] = nb
        zz = np.linspace(0, z_s, 101)
        dchi = np.gradient(np.array([chi(x) for x in zz]))
        nbc = sum(window(x, lo, hi, 200)[1] * d for x, d in zip(zz, dchi))
        r[f"hit_cont_zs{z_s}_tinker"] = 1 - np.exp(-nbc)
    out[lab] = r
    print(lab, {kk: (round(v, 2) if "Nbox" in kk else round(100 * v, 4)) for kk, v in r.items() if "nbar" not in kk})
print("box lengths to z_s", {z: chi(z) / L_BOX for z in (0.1, 0.5, 1.0)}, "chi", {z: chi(z) for z in (0.1, 0.5, 1.0)})
print("sigma8 check", sigma_M(4 / 3 * np.pi * (8 / h) ** 3 * RHO_M0))
(OUT / "indep_hmf_hits.json").write_text(json.dumps(out, indent=1))
