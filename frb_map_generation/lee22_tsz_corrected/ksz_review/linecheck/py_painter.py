#!/usr/bin/env python3
"""Independent numpy/healpy re-implementation of paint_ksz_websky.jl for the review (2026-10-02, ksz_review/linecheck).
No XGPaint, no Julia: own cosmology (flat LCDM + radiation as XGPaint.get_cosmology), own B16 / Lee22 electron densities written
from the formulas, own chord integrals (Gauss-Legendre in l = L t^2), own M200m -> M200c (NFW c200m = 7), own cut and
compensation, healpy query_disc for the discs.

Writes <out>/py_<tag>_<variant>.npy (Delta T/T maps, RING) and <out>/py_<tag>_halos.npz (per-halo analytic integrals
int Delta T/T dOmega = -v/c sigma_T N_e(<X R200c)/d_A^2 [+ v/c sigma_T N_comp/d_A^2], and the pixel-sampled per-halo sums).
Variants: b16_phys, lee22corr_phys, lee22pipe_phys, b16_websky, lee22corr_websky, b16_ws_plain (B16 1.5 R200c on the WebSky
selection and mass, no 1.1116, no compensation: the reference for test (ii))."""
import argparse
import time

import h5py
import healpy as hp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--catalog", required=True)
ap.add_argument("--max-rows", type=int, default=0)
ap.add_argument("--nside", type=int, default=1024)
ap.add_argument("--no-comp", action="store_true")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

# ------------------------------------------------------------------ constants (CODATA 2018 / IAU, as XGPaint & the painter)
SIGMA_T = 6.6524587321e-29; M_P = 1.67262192369e-27; M_E = 9.1093837015e-31
MPC = 3.0856775814913673e22; G = 6.67430e-11; C_KMS = 299792.458; MSUN = 1.98847e30
H = 0.6774; OB = 0.0486; OC = 0.2603; OM = OB + OC; FB = OB / OM
OG = 4.48131e-7 * 2.7255**4 / H**2; OR = OG * (1 + 3.046 * 7 / 8 * (4 / 11)**(4 / 3)); OL = 1 - OM - OR
H0 = 100 * H * 1e3 / MPC                                    # s^-1
RHO_C0 = 3 * H0**2 / (8 * np.pi * G)                       # kg m^-3
XH = 0.76
M_PER_E = M_E + 2 * XH / (XH + 1) * M_P + (1 - XH) / (2 * (1 + XH)) * 4 * M_P
NE_XGP = 0.9 / M_PER_E                                      # electrons per kg of gas, XGPaint ne2d
NE_WS = (1 - 0.24 / 2) / M_P                                # WebSky (1 - Yp/2)/m_p
WS_SCALE = NE_WS / NE_XGP
SHAPE_CLIP = 10**14.8 / H; MCUT = 10**13.61 / H
ARCMIN = np.pi / 180 / 60


def E(z):
    zp = 1 + z
    return np.sqrt(OR * zp**4 + OM * zp**3 + OL)


zg = np.linspace(0, 6, 600001)
inv = 1 / E(zg)
chig = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zg))]) * (C_KMS / (100 * H)) * MPC   # m, comoving


def d_A(z):
    return np.interp(z, zg, chig) / (1 + z)


def rho_c(z):
    return RHO_C0 * E(z)**2


def r200c(M, z):                                            # M in Msun -> physical m
    return (3 * M * MSUN / (4 * np.pi * 200 * rho_c(z)))**(1 / 3)


# M200c/M200m for NFW c200m = 7 (Omega_m(z) without radiation, as the painter's omega_m_z)
def _mu(x):
    return np.log1p(x) - x / (1 + x)


zr = np.linspace(0, 5, 5001)
omz = OM * (1 + zr)**3 / (OM * (1 + zr)**3 + 1 - OM)
lo = np.full_like(zr, 1e-3); hi = np.full_like(zr, 7.0); tgt = _mu(7.0) / (7.0**3 * omz)
for _ in range(100):
    mid = 0.5 * (lo + hi); up = _mu(mid) / mid**3 > tgt
    lo = np.where(up, mid, lo); hi = np.where(up, hi, mid)
ratio_r = _mu(0.5 * (lo + hi)) / _mu(7.0)

# ------------------------------------------------------------------ electron-density shapes: f(x) with amplitude n_amp [m^-3]
# n_e(r) = n_amp * (x/xc)^g * (1 + (x/xc)^al)^(-ex),  x = r / R200c


def b16_params(M, z):
    m = M / 1e14; z1 = 1 + z
    P0 = 4.0e3 * m**0.29 * z1**-0.66; al = 0.88 * m**-0.03 * z1**0.19; be = 3.83 * m**0.04 * z1**-0.025
    g = np.full_like(m, -0.2); xc = np.full_like(m, 0.5)
    n_amp = NE_XGP * FB * rho_c(z) * P0                     # 0.9/m_per_e * f_b rho_c P0
    return n_amp, xc, al, (be + g) / al, g                  # XGPaint exponent (beta+gamma)/alpha, beta = 3.83... raw


def broken(Ms, s_lo, s_hi):
    r = Ms / MCUT
    return np.where(r < 1, r**s_lo, r**s_hi)


def lee_params(M, z, kind):
    z1 = 1 + z; Ms = np.minimum(M, SHAPE_CLIP)
    n0 = 6.8 * (M / MCUT)**0.68 * z1**-2.11
    xc = 7.9 * z1**-0.67 * broken(Ms, 0.47, -0.45)
    bp = 19.5 * z1**-0.31 * broken(Ms, 0.70, -0.18)
    if kind == "pipe":                                       # xgpaint_ne2d: P0 = 200 n0, XGPaint electrons
        n_amp = NE_XGP * FB * rho_c(z) * 200 * n0
    else:                                                    # literal eq. 9 x (h_TNG (1+z))^3 X_H^2
        n_amp = (0.6774 * z1)**3 * XH**2 * n0 * 200 * rho_c(z) * FB / (XH * M_P)
    return n_amp, xc, np.ones_like(M), bp, np.full_like(M, -0.3)


def shape(x, xc, al, ex, g):
    s = x / xc
    return s**g * (1 + s**al)**(-ex)


TG, WG = np.polynomial.legendre.leggauss(160)
TG = 0.5 * (TG + 1); WG = 0.5 * WG


def chord(x, X, xc, al, ex, g):
    """2 int_0^L f(sqrt(x^2 + l^2)) dl, L = sqrt(X^2 - x^2), l = L t^2 (all arrays of one length)."""
    L = np.sqrt(np.maximum(X**2 - x**2, 0.0))
    out = np.empty_like(x)
    for s in range(0, x.size, 20000):
        e = slice(s, s + 20000)
        l = L[e, None] * TG[None, :]**2
        r = np.sqrt(x[e, None]**2 + l**2)
        f = shape(r, xc[e, None], al[e, None], ex[e, None], g[e, None])
        out[e] = 2 * np.sum(f * 2 * L[e, None] * TG[None, :] * WG[None, :], axis=1)
    return out


UG, VG = np.polynomial.legendre.leggauss(400)


def n_e_volume(X, xc, al, ex, g):
    """int_0^X 4 pi x^2 f(x) dx in log x from 1e-9 X to X."""
    lo_, hi_ = np.log(1e-9 * X), np.log(X)
    u = 0.5 * (hi_ - lo_)[:, None] * (UG[None, :] + 1) + lo_[:, None]
    x = np.exp(u)
    f = shape(x, xc[:, None], al[:, None], ex[:, None], g[:, None])
    return 0.5 * (hi_ - lo_) * np.sum(4 * np.pi * x**3 * f * VG[None, :], axis=1)


# ------------------------------------------------------------------ catalogue
t0 = time.time()
with h5py.File(a.catalog, "r") as f:
    n = f["redshift"].shape[0] if a.max_rows == 0 else min(a.max_rows, f["redshift"].shape[0])
    pos = f["Position"][:n].astype(np.float64); vel = f["Velocity"][:n].astype(np.float64)
    mc = f["halo_mass_m200c"][:n].astype(np.float64) / H; mm = f["halo_mass_m200m"][:n].astype(np.float64) / H
    z = f["redshift"][:n].astype(np.float64)
keep = np.isfinite(mc) & (mc > 0) & np.isfinite(mm) & (mm > 0) & np.isfinite(z) & (z > 0) & np.all(np.isfinite(pos), 1) & np.all(np.isfinite(vel), 1)
pos, vel, mc, mm, z = pos[keep], vel[keep], mc[keep], mm[keep], z[keep]
N = mc.size
r = np.linalg.norm(pos, axis=1); u = pos / r[:, None]
v = np.sum(pos * vel, 1) / r / C_KMS
dA = d_A(z)
R200 = r200c(mc, z); th200 = np.arctan(R200 / dA)
mws = mm * np.interp(z, zr, ratio_r)
R200w = r200c(mws, z); th200w = np.arctan(R200w / dA)
cut = (mws > 1e13) & (th200w > 0.5 * ARCMIN) & (np.log10(mws) >= 12) & (np.log10(mws) <= 15.7)
rhom = OM * RHO_C0 * (1 + z)**3
Rc = (mm * MSUN / (4 * np.pi * rhom))**(1 / 3)
Ncomp = FB * mm * MSUN * NE_WS
ncomp = Ncomp / (4 * np.pi / 3 * Rc**3)
thc = np.arctan(Rc / dA)
thmax_p = 4.0 * th200
thmax_w = np.where(cut, 1.5 * th200w if a.no_comp else np.maximum(1.5 * th200w, thc), 0.0)
thmax = np.minimum(np.maximum(thmax_p, thmax_w), np.pi)
print("halos %d  websky-selected %d  z<0.1 %d  vlos rms %.3f km/s" % (N, cut.sum(), np.sum(z < 0.1), np.sqrt(np.mean((v * C_KMS)**2))), flush=True)

# per-variant parameters: (name, X, mass, params, scale, websky)
PB = b16_params(mc, z); PBw = b16_params(mws, z)
PLp = lee_params(mc, z, "pipe"); PLc = lee_params(mc, z, "corr"); PLcw = lee_params(mws, z, "corr")
VARS = [("b16_phys", 4.0, R200, th200, PB, 1.0, False), ("lee22corr_phys", 4.0, R200, th200, PLc, 1.0, False),
        ("lee22pipe_phys", 4.0, R200, th200, PLp, 1.0, False), ("b16_websky", 1.5, R200w, th200w, PBw, WS_SCALE, True),
        ("lee22corr_websky", 1.5, R200w, th200w, PLcw, 1.0, True), ("b16_ws_plain", 1.5, R200w, th200w, PBw, 1.0, True)]

# analytic per-halo integrals of Delta T / T over the sky (flat: sigma_T N / d_A^2)
analytic = {}
for name, X, R, th, P, sc, ws in VARS:
    n_amp, xc, al, ex, g = P
    Ne = n_amp * R**3 * n_e_volume(np.full(N, X), xc, al, ex, g)
    val = -v * sc * SIGMA_T * Ne / dA**2
    if ws:
        val = np.where(cut, val, 0.0)
        if name != "b16_ws_plain" and not a.no_comp:
            val = val + np.where(cut, v * SIGMA_T * Ncomp / dA**2, 0.0)
    analytic[name] = val
analytic["comp_only"] = np.where(cut, v * SIGMA_T * Ncomp / dA**2, 0.0)
print("analytic integrals done %.0f s" % (time.time() - t0), flush=True)

# ------------------------------------------------------------------ painting
nside = a.nside; npix = hp.nside2npix(nside); omega = 4 * np.pi / npix
maps = {name: np.zeros(npix) for name, *_ in VARS}
psum = {name: np.zeros(N) for name, *_ in VARS}
npx = np.zeros(N, dtype=np.int64)


def flush(pixl, hal, thl):
    if not pixl:
        return
    p = np.concatenate(pixl); hi_ = np.concatenate(hal); th = np.concatenate(thl)
    for name, X, R, thv, P, sc, ws in VARS:
        if ws:
            m = cut[hi_] & (th < X * thv[hi_])
        else:
            m = th < X * thv[hi_]
        pv = np.zeros(p.size)
        if m.any():
            hh = hi_[m]; x = th[m] / thv[hh]
            n_amp, xc, al, ex, g = P
            tau = SIGMA_T * n_amp[hh] * R[hh] * chord(x, X, xc[hh], al[hh], ex[hh], g[hh])
            pv[m] = -v[hh] * sc * tau
        if ws and name != "b16_ws_plain" and not a.no_comp:
            b = dA[hi_] * th
            mc_ = cut[hi_] & (b < Rc[hi_])
            hh = hi_[mc_]
            pv[mc_] += v[hh] * SIGMA_T * ncomp[hh] * 2 * np.sqrt(Rc[hh]**2 - b[mc_]**2)
        np.add.at(maps[name], p, pv)
        np.add.at(psum[name], hi_, pv * omega)
    np.add.at(npx, hi_, 1)


pixl, hal, thl, nel = [], [], [], 0
for i in range(N):
    pix = hp.query_disc(nside, u[i], thmax[i] * (1 + 1e-7), inclusive=False)
    if pix.size == 0:
        continue
    vx, vy, vz = hp.pix2vec(nside, pix)
    th = np.arccos(np.clip(vx * u[i, 0] + vy * u[i, 1] + vz * u[i, 2], -1, 1))
    k = th < thmax[i]
    if not k.any():
        continue
    pixl.append(pix[k]); hal.append(np.full(k.sum(), i)); thl.append(th[k]); nel += k.sum()
    if nel > 400_000:
        flush(pixl, hal, thl); pixl, hal, thl, nel = [], [], [], 0
    if i % 20000 == 0:
        print("  halo %d / %d  %.0f s" % (i, N, time.time() - t0), flush=True)
flush(pixl, hal, thl)
for name in maps:
    np.save("%s/py_%s_%s.npy" % (a.out, a.tag, name), maps[name])
np.savez("%s/py_%s_halos.npz" % (a.out, a.tag), z=z, mc=mc, mm=mm, mws=mws, v=v, cut=cut, th200=th200, th200w=th200w, thc=thc,
         thmax=thmax, npix=npx, omega=omega, **{"an_" + k: val for k, val in analytic.items()}, **{"ps_" + k: val for k, val in psum.items()})
print("pixel updates %d  done %.0f s" % (npx.sum(), time.time() - t0), flush=True)
