#!/usr/bin/env python3
"""Independent (Python) catalogue one-halo kSZ for the HalfDome B16 painting, and its sensitivity to
truncation, mass/angle cuts and the WebSky uniform-sphere compensation (ksz_debug/painting_geometry, 2026-10-02).

tau_ell(M, z) for a spherically truncated profile is the 3-D Fourier transform of n_e at k_perp = ell/d_A:
    a_ell = sigma_T / d_A^2 * 4 pi R^3 A int_0^X x^2 g(x) sinc(ell theta200 x) dx,   theta200 = atan(R200c/d_A)
which is exactly what the chord-limited painter + flat-sky Hankel transform compute. XGPaint B16 conventions:
    rho_gas = f_b rho_c(z) P0 gNFW(x; xc, alpha, beta, gamma) (beta used directly as the outer slope),
    n_e = 0.9 rho_gas / (m_e + nH/ne m_p + nHe/ne 4 m_p), X_H = 0.76.
Projected truncation (XGPaint default): full-LOS projected profile cut at theta = 4 theta200, 2-D Hankel transform.
C_ell = (1/4pi) sum_cells sum(v_los/c)^2 |a_ell|^2.
"""
import sys
import numpy as np
import h5py
from scipy.special import j0, spherical_jn

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/painting_geometry"
HIST = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5"

# ---------------------------------------------------------------- cosmology (XGPaint get_cosmology, h=0.68, Neff=3.046)
h, Ob, Oc = 0.68, 0.049, 0.261
Om = Ob + Oc
Tcmb, Neff = 2.7255, 3.046
OG = 4.48131e-7 * Tcmb**4 / h**2
ON = Neff * OG * (7 / 8) * (4 / 11) ** (4 / 3)
OR = OG + ON
OL = 1 - Om - OR
f_b = Ob / Om
E = lambda z: np.sqrt(OR * (1 + z) ** 4 + Om * (1 + z) ** 3 + OL)
C_KMS = 299792.458
DH = C_KMS / (100 * h)                       # Mpc
_zg = np.linspace(0, 6, 120001)
_ie = 1 / E(_zg)
_chi = DH * np.concatenate([[0], np.cumsum(0.5 * (_ie[1:] + _ie[:-1]) * np.diff(_zg))])
chi = lambda z: np.interp(z, _zg, _chi)
dA = lambda z: chi(z) / (1 + z)              # Mpc

MPC = 3.0856775814913673e22                  # m
MSUN = 1.98847e30                            # kg (XGPaint M_sun)
G = 6.67430e-11
SIGMA_T = 6.6524587321e-29
M_P = 1.67262192369e-27
M_E = 9.1093837015e-31
H0_SI = 100 * h * 1e3 / MPC
rho_c = lambda z: 3 * (H0_SI * E(z)) ** 2 / (8 * np.pi * G)          # kg/m^3
xH = 0.76
FACTOR = M_E + 2 * xH / (xH + 1) * M_P + (1 - xH) / (2 * (1 + xH)) * 4 * M_P
NE_PER_KG = 0.9 / FACTOR                     # XGPaint ne2d electrons per kg of gas


def r200c_m(M, z):
    """physical R200c [m] for M [Msun]"""
    return (M * MSUN / (4 * np.pi / 3 * 200 * rho_c(z))) ** (1 / 3)


def b16_params(M, z):
    m = M / 1e14
    z1 = 1 + z
    P0 = 4.0e3 * m**0.29 * z1**-0.66
    xc = 0.5
    al = 0.88 * m**-0.03 * z1**0.19
    be = 3.83 * m**0.04 * z1**-0.025
    ga = -0.2
    return P0, xc, al, be, ga


def gnfw(x, xc, al, be, ga):
    xb = x / xc
    with np.errstate(divide="ignore", invalid="ignore"):
        return xb**ga * (1 + xb**al) ** (-(be + ga) / al)


def simpson_weights(n, dx):
    w = np.ones(n)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    return w * dx / 3


def sphere_profile_transform(M, z, ells, Xs=(1.0, 2.0, 3.0, 4.0)):
    """a_ell for spheres X in Xs (all <= 4) for one (M, z); also total electrons N_e(<X)."""
    P0, xc, al, be, ga = b16_params(M, z)
    R = r200c_m(M, z)
    d = dA(z) * MPC
    th = np.arctan(R / d)
    A = NE_PER_KG * f_b * rho_c(z) * P0                             # m^-3
    kmax = ells.max() * th * 4.0
    n = int(np.clip(kmax / 0.25, 2048, 65536))
    n = (n // 8 + 1) * 8 + 1                                        # n-1 divisible by 8 -> X=1,2,3 on grid, even intervals
    x = np.linspace(0, 4.0, n)
    hx = np.zeros(n)
    hx[1:] = x[1:] ** 2 * gnfw(x[1:], xc, al, be, ga)
    kx = np.outer(ells * th, x)
    S = np.sinc(kx / np.pi)                                         # sin(kx)/(kx)
    pref = SIGMA_T / d**2 * 4 * np.pi * R**3 * A
    out = {}
    Ne = {}
    for X in Xs:
        j = int(round(X / 4.0 * (n - 1)))
        w = simpson_weights(j + 1, x[1] - x[0])
        out[X] = pref * (S[:, : j + 1] * (hx[: j + 1] * w)).sum(1)
        Ne[X] = 4 * np.pi * R**3 * A * (hx[: j + 1] * w).sum()
    return out, Ne, th, R, d


_U = np.linspace(0, 1, 6001)


def projected_profile_transform(M, z, ells, X=4.0, lmax_r200=1e5):
    """XGPaint default: full-LOS projected profile (0..1e5 R200c) cut at theta = X theta200; 2-D Hankel."""
    P0, xc, al, be, ga = b16_params(M, z)
    R = r200c_m(M, z)
    d = dA(z) * MPC
    th = np.arctan(R / d)
    A = NE_PER_KG * f_b * rho_c(z) * P0
    # projected dimensionless profile on a log grid in x
    xs = np.geomspace(1e-5, X, 400)
    proj = np.empty_like(xs)
    for i, xx in enumerate(xs):
        umax = np.arcsinh(lmax_r200 / xx)
        u = _U * umax
        r = xx * np.cosh(u)
        f = gnfw(r, xc, al, be, ga) * r                             # dl = x cosh(u) du = r du
        # integrate on a log-ish u grid; the integrand decays fast so a uniform u grid of 6001 points over asinh(1e5/x) is fine
        proj[i] = 2 * np.trapz(f, u)
    kmax = ells.max() * th * X
    n = int(np.clip(kmax / 0.25, 2048, 65536))
    n = (n // 2) * 2 + 1
    x = np.linspace(0, X, n)
    px = np.empty(n)
    px[1:] = np.exp(np.interp(np.log(x[1:]), np.log(xs), np.log(proj)))
    px[0] = proj[0]
    w = simpson_weights(n, x[1] - x[0])
    J = j0(np.outer(ells * th, x))
    a = SIGMA_T * A * R * th**2 * 2 * np.pi * (J * (x * px * w)).sum(1)
    return a


def m200m_from_m200c(M, z):
    """NFW conversion with Duffy+08 c200c(M, z) (full sample)."""
    c = 5.71 * (M * h / 2e12) ** -0.084 * (1 + z) ** -0.47
    rho_m = Om * rho_c(0) * (1 + z) ** 3
    R = r200c_m(M, z)
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    # solve M(<r)/(4/3 pi r^3) = 200 rho_m for r = s R
    lo, hi = 1.0, 3.0
    for _ in range(60):
        s = 0.5 * (lo + hi)
        Mr = M * mu(c * s) / mu(c)
        dens = Mr * MSUN / (4 * np.pi / 3 * (s * R) ** 3)
        if dens > 200 * rho_m:
            lo = s
        else:
            hi = s
    return M * mu(c * s) / mu(c)


def main():
    with h5py.File(HIST, "r") as f:
        le = f["logm_edges"][:]
        ze = f["z_edges"][:]
        cnt = f["count"][:]           # (nz, nm) in python order
        v2 = f["sum_vlos_over_c_squared"][:]
    ells = np.exp(np.linspace(np.log(30.0), np.log(8192.0), 80))
    ells = np.unique(np.concatenate([ells, [300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0]]))
    cells = np.argwhere(cnt > 0)
    print("cells", len(cells), "halos", cnt.sum())
    nl = len(ells)
    Xs = (1.0, 2.0, 3.0, 4.0)
    res = {}
    keys = ["sph1", "sph2", "sph3", "sph4", "proj4", "comp3_m200m", "comp4_m200m", "comp3_m200c", "comp3_m200m_ne088"]
    per_cell = {k: np.zeros((len(cells), nl)) for k in keys}
    meta = np.zeros((len(cells), 8))
    do_proj = "--no-proj" not in sys.argv
    for ci, (iz, im) in enumerate(cells):
        lm = 0.5 * (le[im] + le[im + 1])
        z = 0.5 * (ze[iz] + ze[iz + 1])
        M = 10**lm
        w = v2[iz, im] / (4 * np.pi)
        a, Ne, th, R, d = sphere_profile_transform(M, z, ells, Xs)
        for X in Xs:
            per_cell["sph%d" % X][ci] = w * a[X] ** 2
        if do_proj:
            per_cell["proj4"][ci] = w * projected_profile_transform(M, z, ells) ** 2
        # WebSky compensation: uniform sphere with the halo mass at mean density 3 rho_m (or 4 rho_m), same electrons per kg
        M200m = m200m_from_m200c(M, z)
        rho_m = Om * rho_c(0) * (1 + z) ** 3
        for key, Mc, odens, nek in (("comp3_m200m", M200m, 3.0, NE_PER_KG), ("comp4_m200m", M200m, 4.0, NE_PER_KG),
                                    ("comp3_m200c", M, 3.0, NE_PER_KG), ("comp3_m200m_ne088", M200m, 3.0, 0.88 / M_P)):
            Rs = (Mc * MSUN / (4 * np.pi / 3 * odens * rho_m)) ** (1 / 3)
            Ns = f_b * Mc * MSUN * nek
            q = ells * Rs / d
            Ws = 3 * spherical_jn(1, q) / q
            a_s = SIGMA_T * Ns / d**2 * Ws
            # halo profile: for the ne088 variant rescale the halo profile to the same 0.88/m_p convention
            a_h = a[4.0] * (nek / NE_PER_KG)
            per_cell[key][ci] = w * (a_h - a_s) ** 2
        meta[ci] = (lm, z, cnt[iz, im], v2[iz, im], th, Ne[4.0] / (f_b * M * MSUN * NE_PER_KG),
                    Ne[1.0] / (f_b * M * MSUN * NE_PER_KG), M200m / M)
        if ci % 1000 == 0:
            print(ci, lm, z, flush=True)
    np.savez(OUT + "/onehalo_variants_percell.npz", ells=ells, meta=meta, cells=cells,
             **{k: v for k, v in per_cell.items()})
    T2 = 2.7255e6**2
    pre = ells * (ells + 1) / (2 * np.pi) * T2
    for k in keys:
        tot = per_cell[k].sum(0) * pre
        print(k, " ".join("ell%d=%.4f" % (L, tot[np.argmin(abs(ells - L))]) for L in (300, 1000, 3000, 5000)))


if __name__ == "__main__":
    main()
