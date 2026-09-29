#!/usr/bin/env python3
"""Gas-density models of the HalfDome FRB sphere products, re-implemented in Python for the Lee22 redshift test.

Everything here mirrors frb_map_generation/lee2022_frb_dm_profile.jl and XGPaint (dev checkout,
/home/kn18001/.julia/dev/XGPaint/src/profiles_tau.jl, model.jl):

  Lee22 no-c (arXiv Table 8 = repo "Table A2"), n0 pivot = M_cut, shape clip at 10^14.8/h Msun:
      n0  = 6.8  (M/Mcut)^0.68 (1+z)^-2.11
      xc  = 7.9  (1+z)^-0.67 {(M/Mcut)^0.47  | (M/Mcut)^-0.45}
      b'  = 19.5 (1+z)^-0.31 {(M/Mcut)^0.70  | (M/Mcut)^-0.18}      Mcut = 10^13.61/h Msun
      f(x)= (x/xc)^-0.3 (1 + x/xc)^-b'
      literal (eq. 9 as printed):  n_e = n0 f(x) n200,   n200 = 200 rho_cr(z) f_b / (X_H m_p)
      xgpnative:                   0.9 X_H m_p / m_per_e = 0.6016... times literal
  Battaglia16 as XGPaint paints it (BattagliaTauProfile defaults):
      rho_gas = P0 gnfw(x) f_b rho_cr(z),  n_e = 0.9 rho_gas / m_per_e,
      gnfw = (x/xc)^g (1+(x/xc)^a)^(-(b+g)/a),  P0 = 4000 m^0.29 (1+z)^-0.66, xc = 0.5,
      a = 0.88 m^-0.03 (1+z)^0.19, b = 3.83 m^0.04 (1+z)^-0.025, g = -0.2,  m = M/1e14 Msun
  m_per_e = m_e + 2X_H/(1+X_H) m_p + (1-X_H)/(2(1+X_H)) 4 m_p  (XGPaint ne2d, X_H = 0.76).

Units: M in physical Msun (M200c), radii in physical Mpc/m, n_e physical m^-3, DM pc cm^-3.
"""
import numpy as np

# constants exactly as XGPaint / Unitful (CODATA 2018, UnitfulAstro)
G_SI = 6.67430e-11
M_P = 1.67262192369e-27
M_E = 9.1093837015e-31
MSUN = 1.98847e30
PC_M = 3.0856775814913673e16
MPC_M = PC_M * 1e6
KM = 1e3
C_KMS = 299792.458
X_H = 0.76
M_PER_E = M_E + (2 * X_H / (X_H + 1)) * M_P + ((1 - X_H) / (2 * (1 + X_H))) * 4 * M_P
XGP_NATIVE_FACTOR = 0.9 * X_H * M_P / M_PER_E            # 0.6016316602...
M2_TO_PC_CM3 = 1e-4 / (PC_M * 1e2)                        # 1 m^-2 = 1e-4 cm^-2, / (pc in cm)


class Cosmo:
    """Flat LCDM; radiation optional (XGPaint's get_cosmology adds photons + 3.046 massless nu)."""

    def __init__(self, name, omega_m, omega_b, h, radiation=False, tcmb=2.7255, neff=3.046):
        self.name, self.omega_m, self.omega_b, self.h = name, omega_m, omega_b, h
        if radiation:
            og = 4.48131e-7 * tcmb ** 4 / h ** 2
            self.omega_r = og + neff * og * (7 / 8) * (4 / 11) ** (4 / 3)
        else:
            self.omega_r = 0.0
        self.omega_l = 1 - omega_m - self.omega_r
        zz = np.linspace(0, 8, 160001)
        ez = self.E(zz)
        inv = 1 / ez
        chi = np.concatenate([[0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(zz))]) * C_KMS / (100 * h)
        self._zz, self._chi = zz, chi
        self.f_b = omega_b / omega_m

    def E(self, z):
        z = np.asarray(z, float)
        return np.sqrt(self.omega_r * (1 + z) ** 4 + self.omega_m * (1 + z) ** 3 + self.omega_l)

    def chi_mpc(self, z):
        return np.interp(z, self._zz, self._chi)

    def rho_crit_si(self, z):
        H = 100 * self.h * self.E(z) * KM / MPC_M          # s^-1
        return 3 * H ** 2 / (8 * np.pi * G_SI)              # kg m^-3

    def r200c_m(self, mass_msun, z):
        return np.cbrt(np.asarray(mass_msun) * MSUN / (4 * np.pi / 3 * 200 * self.rho_crit_si(z)))


HALFDOME_XGP = Cosmo("HalfDome/XGPaint (Oc=0.261, Ob=0.049, h=0.68, +radiation)", 0.31, 0.049, 0.68, radiation=True)
PLANCK15_TNG = Cosmo("Planck15/TNG (Om=0.3089, Ob=0.0486, h=0.6774)", 0.3089, 0.0486, 0.6774, radiation=False)


# ------------------------------------------------------------------ Lee22 no-concentration (Table 8)
def lee22_params(mass_msun, z, h, clip_hinv=10 ** 14.8):
    mass = np.asarray(mass_msun, float)
    z = np.asarray(z, float)
    mcut = 10 ** 13.61 / h
    ms = np.minimum(mass, clip_hinv / h)
    r = mass / mcut
    rs = ms / mcut
    opz = 1 + z
    n0 = 6.8 * r ** 0.68 * opz ** -2.11
    xc = 7.9 * opz ** -0.67 * np.where(rs < 1, rs ** 0.47, rs ** -0.45)
    bp = 19.5 * opz ** -0.31 * np.where(rs < 1, rs ** 0.70, rs ** -0.18)
    return n0, xc, bp


def lee22_shape(x, xc, bp):
    s = x / xc
    return s ** -0.3 * (1 + s) ** (-bp)


def lee22_n200_m3(cosmo, z):
    return 200 * cosmo.rho_crit_si(z) * cosmo.f_b / (X_H * M_P)


def lee22_literal_ne_m3(cosmo, x, mass, z):
    n0, xc, bp = lee22_params(mass, z, cosmo.h)
    return n0 * lee22_shape(x, xc, bp) * lee22_n200_m3(cosmo, z)


# ------------------------------------------------------------------ Battaglia16 (XGPaint defaults)
def b16_params(mass_msun, z):
    m = np.asarray(mass_msun, float) / 1e14
    opz = 1 + np.asarray(z, float)
    P0 = 4.0e3 * m ** 0.29 * opz ** -0.66
    xc = 0.5 * m ** 0.0 * opz ** 0.0
    al = 0.88 * m ** -0.03 * opz ** 0.19
    be = 3.83 * m ** 0.04 * opz ** -0.025
    ga = -0.2 + 0 * m
    return P0, xc, al, be, ga


def gnfw(x, xc, al, be, ga):
    s = x / xc
    return s ** ga * (1 + s ** al) ** (-(be + ga) / al)


def b16_ne_m3(cosmo, x, mass, z):
    P0, xc, al, be, ga = b16_params(mass, z)
    return 0.9 * P0 * gnfw(x, xc, al, be, ga) * cosmo.f_b * cosmo.rho_crit_si(z) / M_PER_E


# ------------------------------------------------------------------ enclosed electrons inside the R200c sphere
_LX = np.linspace(np.log(1e-9), 0.0, 6001)
_X = np.exp(_LX)


def _trap(y, lx):
    return np.sum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(lx), axis=-1)


def lee22_mean_nbar(mass, z, h):
    """Volume-mean <n_e/n200> inside R200c of the literal Lee22 no-c fit: 3 int_0^1 n0 f(x) x^2 dx."""
    n0, xc, bp = lee22_params(mass, z, h)
    n0, xc, bp = (np.asarray(a)[..., None] for a in (n0, xc, bp))
    return 3 * _trap(n0 * lee22_shape(_X, xc, bp) * _X ** 3, _LX)


def b16_mean_rho(mass, z):
    """Volume-mean <rho_gas/(f_b rho_cr)>/200 inside R200c of XGPaint's B16 = gas mass / (f_b M200c)."""
    P0, xc, al, be, ga = (np.asarray(a)[..., None] for a in b16_params(mass, z))
    return 3 * _trap(P0 * gnfw(_X, xc, al, be, ga) * _X ** 3, _LX) / 200


def electrons_in_r200(model, cosmo, mass, z):
    """Number of free electrons inside the R200c sphere (dimensionless count)."""
    mass = np.asarray(mass, float)
    if model == "lee22_literal":
        return cosmo.f_b * mass * MSUN / (X_H * M_P) * lee22_mean_nbar(mass, z, cosmo.h)
    if model == "lee22_xgpnative":
        return XGP_NATIVE_FACTOR * electrons_in_r200("lee22_literal", cosmo, mass, z)
    if model == "b16":
        return 0.9 * cosmo.f_b * mass * MSUN / M_PER_E * b16_mean_rho(mass, z)
    if model == "fb_full":
        # control: all cosmic baryons of the halo inside R200c, fully ionized H+He (n_e = (1+X_H)/2 rho/m_p)
        return cosmo.f_b * mass * MSUN * (1 + X_H) / (2 * M_P) * np.ones_like(np.asarray(z, float))
    raise ValueError(model)


def ne_m3(model, cosmo, x, mass, z):
    if model == "lee22_literal":
        return lee22_literal_ne_m3(cosmo, x, mass, z)
    if model == "lee22_xgpnative":
        return XGP_NATIVE_FACTOR * lee22_literal_ne_m3(cosmo, x, mass, z)
    if model == "b16":
        return b16_ne_m3(cosmo, x, mass, z)
    raise ValueError(model)


def chord_mean_dm(model, cosmo, x, mass, z, n_gl=400):
    """g = DM_sphere/(2 sqrt(1-x^2)) [pc cm^-3], observer frame (same quantity as the Julia table)."""
    from scipy import integrate
    r200 = cosmo.r200c_m(mass, z)
    L = np.sqrt(1 - x ** 2)
    f = lambda l: ne_m3(model, cosmo, np.sqrt(x ** 2 + l ** 2), mass, z)
    val, _ = integrate.quad(f, 0.0, L, epsrel=1e-11, epsabs=0, limit=500, points=[min(x, L * 0.5)])
    return val / L * r200 * M2_TO_PC_CM3 / (1 + z)
