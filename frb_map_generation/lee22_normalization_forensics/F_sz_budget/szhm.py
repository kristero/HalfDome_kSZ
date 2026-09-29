#!/usr/bin/env python3
"""Halo-model kSZ / tSZ / FRB-halo-DM budget for the Lee22 normalization forensics (subdir F_sz_budget).

Profiles (all physical, cgs internally):
  density  lee22_literal    n_e = n0 g(x) n200,  n200 = 200 rho_cr(z) f_b / (X_H m_p)    (Lee22 eq. 9 as printed,
                            Table 8 no-concentration fit, M_cut pivot for n0, x'_c/beta' frozen above 10^14.8/h Msun,
                            exactly as frb_map_generation/lee2022_frb_dm_profile.jl with n0_pivot=:mcut)
           lee22_xgpnative  0.6016316602183194 x literal (XGPaint ne2d reading, production PDF variant)
           b16              XGPaint HaloDMProfile (Battaglia16 AGN rho_gas fit): n_e = 0.9 P0 gNFW f_b rho_cr / m_per_e
  pressure b12              XGPaint Battaglia16ThermalSZProfile defaults (= Battaglia12 AGN): P_e = 0.5176 P0 P200 f(x)
           lee22p           Lee22 Table 7 no-c fit: P_e = P0 P200 f(x), shape frozen above 10^14.8/h Msun
                            (as lee2022_tsz_pressure_profile.jl)
Truncations, matching the painted maps:
  kSZ  3-D sphere of radius 3 R200c (line of sight included)      -> 3-D Fourier transform of the truncated profile
  tSZ  full line of sight inside an angular aperture of 4 R200c    -> Hankel transform of the projected profile
  FRB  3-D sphere of radius R200c                                   -> electron count inside R200c
Halo model:
  C_l^1h = int dz c chi^2/H int dlnM dn/dlnM W(z) |X_l(M,z)|^2
  C_l^2h = int dz c chi^2/H W(z) [int dlnM dn/dlnM b X_l]^2 P_lin((l+1/2)/chi, z)
  kSZ: X_l = tau_l = sigma_T N_e(k = l/D_A) / D_A^2,  W = T_CMB^2 <v_los^2>(z)/c^2,
       <v_los^2> = (1/3) (a H f D)^2 int dk P_lin(k, 0) / (2 pi^2)      (linear theory, no halo-scale smoothing)
  tSZ: X_l = y_l = int d^2theta y(theta) e^{i l.theta},  W = 1
Mass function Tinker08 (M200c), bias Tinker10, CAMB linear P(k) (via halo_model_dm_pdf.py).
"""
import sys

import numpy as np
from scipy import special, interpolate

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/analytic_pdf_torkamani2026")
from halo_model_dm_pdf import (Cosmology, tinker08_dndlnm, tinker10_bias, r200c_mpc,  # noqa: E402
                               MPC_CM, PC_CM, MSUN_G, M_P_G, C_KMS, RHO_CRIT0_H2)

_trapz = getattr(np, "trapezoid", None) or np.trapz
SIGMA_T = 6.6524587321e-25         # cm^2
M_E_G = 9.1093837015e-28           # g
ME_C2 = 8.1871057769e-7            # erg
G_CGS = 6.67430e-8
T_CMB_UK = 2.7255e6
X_H = 0.76
M_PER_E = M_E_G + (2 * X_H / (1 + X_H)) * M_P_G + ((1 - X_H) / (2 * (1 + X_H))) * 4 * M_P_G   # XGPaint ne2d
XGPNATIVE_FACTOR = 0.9 * X_H * M_P_G / M_PER_E                                                # 0.6016316602
KSZ_SPHERE = 3.0
TSZ_APERTURE = 4.0
FRB_SPHERE = 1.0

# painting cosmology of the kSZ / tSZ comparison (settings.toml of lee22_pdf_ksz_comparison)
PAINT_COSMO = Cosmology("paint contract", omega_m=0.31, omega_b=0.049, h=0.68, sigma8=0.8159, n_s=0.9667)


# ----------------------------------------------------------------------------------------------- profiles
def rho_crit_cgs(cosmo, z):
    return RHO_CRIT0_H2 * cosmo.h ** 2 * cosmo.E(z) ** 2 * MSUN_G / MPC_CM ** 3


def _bpl(ratio, lo, hi):
    return np.where(ratio < 1, ratio ** lo, ratio ** hi)


def lee22_density_params(cosmo, M, z):
    """Table 8 (arXiv) = Table A2 (MNRAS) no-concentration density fit; M physical Msun (array)."""
    h = cosmo.h
    mcut = 10 ** 13.61 / h
    ms = np.minimum(M, 10 ** 14.8 / h)
    n0 = 6.8 * (M / mcut) ** 0.68 * (1 + z) ** -2.11
    xc = 7.9 * (1 + z) ** -0.67 * _bpl(ms / mcut, 0.47, -0.45)
    bp = 19.5 * (1 + z) ** -0.31 * _bpl(ms / mcut, 0.70, -0.18)
    return n0, xc, bp


class RowShape:
    """Row-sliceable dimensionless shape f(x): params are (nM,) arrays; kind in {'lee22', 'b16'}.
    lee22 (and any alpha=1, gamma=-0.3 GNFW): f = (x/xc)^-0.3 (1 + x/xc)^-beta
    b16 (XGPaint generalized_nfw):             f = (x/xc)^-0.2 (1 + (x/xc)^alpha)^(-(beta - 0.2)/alpha)"""

    def __init__(self, kind, **par):
        self.kind, self.par = kind, {k: np.asarray(v, float) for k, v in par.items()}
        self.n = len(next(iter(self.par.values())))

    def __call__(self, x, rows=None):
        a, b = rows if rows is not None else (0, self.n)
        p = {k: v[a:b].reshape((b - a,) + (1,) * (x.ndim - 1)) for k, v in self.par.items()}
        u = x / p["xc"]
        if self.kind == "lee22":
            return u ** -0.3 * (1 + u) ** -p["beta"]
        if self.kind == "b16":
            return u ** -0.2 * (1 + u ** p["alpha"]) ** (-(p["beta"] - 0.2) / p["alpha"])
        raise ValueError(self.kind)


def density_shape(name, cosmo, M, z):
    """(amplitude [cm^-3] (nM,), RowShape) for the density models; n_e(r) = amp * f(r/R200c)."""
    fb = cosmo.omega_b / cosmo.omega_m
    rc = rho_crit_cgs(cosmo, z)
    if name.startswith("lee22"):
        n0, xc, bp = lee22_density_params(cosmo, M, z)
        amp = n0 * 200 * rc * fb / (X_H * M_P_G) * (XGPNATIVE_FACTOR if name == "lee22_xgpnative" else 1.0)
        return amp, RowShape("lee22", xc=xc, beta=bp)
    if name == "b16":
        m = M / 1e14
        z1 = 1 + z
        amp = 0.9 * 4.0e3 * m ** 0.29 * z1 ** -0.66 * fb * rc / M_PER_E
        return amp, RowShape("b16", xc=0.5 * np.ones_like(m), alpha=0.88 * m ** -0.03 * z1 ** 0.19,
                             beta=3.83 * m ** 0.04 * z1 ** -0.025)
    raise ValueError(name)


def pressure_params(name, cosmo, M, z):
    """(P0_eff, xc, beta) with P_e = P0_eff P200 (x/xc)^-0.3 (1 + x/xc)^-beta."""
    z1 = 1 + z
    if name == "b12":
        m = M / 1e14
        return (0.5176 * 18.1 * m ** 0.154 * z1 ** -0.758, 0.497 * m ** -0.00865 * z1 ** 0.731,
                4.35 * m ** 0.0393 * z1 ** 0.415)
    if name == "lee22p":
        h = cosmo.h
        mcut = 10 ** 13.60 / h
        r = M / mcut
        rs = np.minimum(M, 10 ** 14.8 / h) / mcut
        return 2.8 * _bpl(r, 1.08, 0.80) * z1 ** -1.89, 2.10 * rs ** -0.35 * z1 ** 0.53, 9.4 * rs ** -0.06 * z1 ** 0.60
    raise ValueError(name)


def p200_cgs(cosmo, M, z):
    """Battaglia P200 = 200 G M rho_cr f_b / (2 R200) [erg cm^-3], R200 physical."""
    r200 = r200c_mpc(cosmo, M, z) * MPC_CM
    return 200 * G_CGS * M * MSUN_G * rho_crit_cgs(cosmo, z) * (cosmo.omega_b / cosmo.omega_m) / (2 * r200)


# ------------------------------------------------------------------------- projected pressure shape table
class ProjectedGNFWTable:
    """Y(u; beta) = 2 int_0^inf (u^2+t^2)^(-0.15) (1 + sqrt(u^2+t^2))^(-beta) dt  (alpha=1, gamma=-0.3);
    Sigma(x) = x_c Y(x/x_c; beta) is the full-LOS projection of (x/x_c)^-0.3 (1+x/x_c)^-beta in R200c units."""

    def __init__(self, lnu=None, beta=None, n_t=1200):
        lnu = np.linspace(np.log(1e-8), np.log(20.0), 700) if lnu is None else lnu
        beta = np.linspace(2.5, 42.0, 317) if beta is None else beta
        t_nodes, t_w = _gl(n_t)
        lt_lo, lt_hi = np.log(1e-13), np.log(1e5)
        lt = t_nodes * (lt_hi - lt_lo) + lt_lo
        w = (lt_hi - lt_lo) * t_w
        t = np.exp(lt)
        out = np.empty((len(lnu), len(beta)))
        for i, lu in enumerate(lnu):
            r = np.sqrt(np.exp(2 * lu) + t ** 2)
            base = r ** -0.3 * t * w
            out[i] = 2 * (base[None, :] * np.exp(-beta[:, None] * np.log1p(r)[None, :])).sum(1)
        self.lnu, self.beta, self.lnY = lnu, beta, np.log(out)
        self.spl = interpolate.RectBivariateSpline(lnu, beta, self.lnY, kx=3, ky=3)

    def __call__(self, u, beta):
        u = np.asarray(u, float)
        b = np.broadcast_to(beta, u.shape)
        lu = np.log(np.clip(u, np.exp(self.lnu[0]), None))
        if lu.max() > self.lnu[-1] + 1e-9 or b.min() < self.beta[0] or b.max() > self.beta[-1]:
            raise ValueError("projected table range exceeded: {} {} {}".format(lu.max(), b.min(), b.max()))
        return np.exp(self.spl.ev(lu.ravel(), b.ravel())).reshape(u.shape)


# ------------------------------------------------------------------------------------------ transforms
_GL_CACHE = {}


def _gl(n, order=32):
    """n-node composite Gauss-Legendre rule on [0, 1] (n/order panels of `order` nodes; plain GL for n <= order).
    numpy's leggauss is O(n^3) (eigenvalue solve), so large rules are built from panels and cached."""
    key = (n, order)
    if key not in _GL_CACHE:
        if n <= order:
            t, w = np.polynomial.legendre.leggauss(n)
            _GL_CACHE[key] = (0.5 * (t + 1), 0.5 * w)
        else:
            p = int(np.ceil(n / order))
            t, w = np.polynomial.legendre.leggauss(order)
            t = 0.5 * (t + 1) / p; w = 0.5 * w / p
            edges = np.arange(p) / p
            _GL_CACHE[key] = ((edges[:, None] + t[None, :]).ravel(), np.tile(w, p))
    return _GL_CACHE[key]


def n_nodes_for(qX, base=512, cap=1 << 15):
    n = int(max(base, 3.0 * qX + 64))
    return int(min(cap, 1 << int(np.ceil(np.log2(n)))))


def sphere_ft(shape, X, q, n=None, chunk=8):
    """u(q) = 4 pi int_0^X x^2 f(x) j0(q x) dx for each row of q (nM, nq); nodes x = X s^2 (GL in s)."""
    nM, nq = q.shape
    out = np.empty((nM, nq))
    for a in range(0, nM, chunk):
        b = min(nM, a + chunk)
        nn = n_nodes_for(q[a:b].max() * X) if n is None else n
        s, w = _gl(nn)
        x = X * s ** 2
        jac = 2 * X * s * w
        fx = shape(np.broadcast_to(x, (b - a, nn)), rows=(a, b))
        integ = 4 * np.pi * x ** 2 * jac * fx
        qx = q[a:b, :, None] * x[None, None, :]
        out[a:b] = np.einsum("mqn,mn->mq", np.sinc(qx / np.pi), integ)
    return out


def aperture_hankel(table, xc, beta, X, q, n=None, chunk=8):
    """Ytilde(q) = 2 pi int_0^X x Sigma(x) J0(q x) dx, Sigma(x) = xc Y(x/xc; beta) (full LOS), nodes x = X s^2."""
    nM, nq = q.shape
    out = np.empty((nM, nq))
    for a in range(0, nM, chunk):
        b = min(nM, a + chunk)
        nn = n_nodes_for(q[a:b].max() * X) if n is None else n
        s, w = _gl(nn)
        x = X * s ** 2
        jac = 2 * X * s * w
        sig = xc[a:b, None] * table(x[None, :] / xc[a:b, None], np.broadcast_to(beta[a:b, None], (b - a, nn)))
        integ = 2 * np.pi * x * jac * sig
        qx = q[a:b, :, None] * x[None, None, :]
        out[a:b] = np.einsum("mqn,mn->mq", special.j0(qx), integ)
    return out


def enclosed(shape, X, n=4096):
    """4 pi int_0^X x^2 f(x) dx per row."""
    s, w = _gl(n)
    x = X * s ** 2
    jac = 2 * X * s * w
    return (4 * np.pi * x ** 2 * jac * shape(np.broadcast_to(x, (shape.n, n)))).sum(1)


# ---------------------------------------------------------------------------------------- cosmology bits
def growth_rate(cosmo, z, dz=1e-3):
    """f = dlnD/dlna by centred differences of the engine's growth D(z)."""
    z = np.atleast_1d(np.asarray(z, float))
    zp, zm = z + dz, np.maximum(z - dz, 0.0)
    return -(np.log(cosmo.growth(zp)) - np.log(cosmo.growth(zm))) / (np.log1p(zp) - np.log1p(zm))


def vlos2_kms2(cosmo, z):
    """Linear-theory 1-D velocity variance (km/s)^2 at z: (a H f D)^2/3 int dk P(k,0)/(2 pi^2)."""
    k, p = cosmo.power()
    s = _trapz(p / (2 * np.pi ** 2), k)            # Mpc^2 (z=0)
    z = np.atleast_1d(np.asarray(z, float))
    D = cosmo.growth(z)
    f = growth_rate(cosmo, z)
    aH = 100 * cosmo.h * cosmo.E(z) / (1 + z)      # km/s/Mpc
    return (aH * f * D) ** 2 * s / 3.0, D, f
