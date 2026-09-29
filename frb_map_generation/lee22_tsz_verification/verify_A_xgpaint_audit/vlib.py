"""Verifier A: my own (independent of lee22_models.py and XGPaint) implementation of the painted tSZ profiles.
Units: SI internally for y; masses physical Msun; R200c physical; rho_crit(z) physical.
Pressure fits (as printed):
  B12 (Battaglia+12 AGN): P_th/P200 = P0 (x/xc)^-0.3 [1+x/xc]^-beta, P0=18.1 m^0.154 (1+z)^-0.758, xc=0.497 m^-0.00865 (1+z)^0.731,
      beta=4.35 m^0.0393 (1+z)^0.415, m = M200c/(1e14 Msun physical);  P_e = 0.5176 P_th.
  Lee22 Table 7 no-c: P_e/P200 = P0 (x/xc)^-0.3 [1+x/xc]^-beta, M in h^-1 Msun, Mcut=10^13.60, P0=2.8 (M/Mcut)^{1.08|0.80} (1+z)^-1.89,
      xc=2.10 (Ms/Mcut)^-0.35 (1+z)^0.53, beta=9.4 (Ms/Mcut)^-0.06 (1+z)^0.60, Ms=min(M,10^14.8 h^-1 Msun).
  P200 = 200 G M rho_cr(z) f_b / (2 R200c);  y = sigma_T/(m_e c^2) * P200 * R200c * p0 * 2 int_0^1e5 g(sqrt(X^2+l^2)) dl.
Cosmology: flat, Om=0.31, Ob=0.049, h=0.68, radiation as Cosmology.jl/XGPaint get_cosmology (Tcmb 2.7255, Neff 3.04 for the B12 model
default, 3.046 for the Lee22 model); rad=False drops radiation (Lee22-paper convention)."""
import numpy as np
from numpy.polynomial.legendre import leggauss

G = 6.67430e-11; MSUN = 1.98847e30; MPC = 3.0856775814913673e22; C = 299792458.0
SIGMA_T = 6.6524587321e-29; ME = 9.1093837015e-31
H = 0.68; OM = 0.31; OB = 0.049; FB = OB / OM


class Cos:
    def __init__(self, neff=3.04, rad=True, tcmb=2.7255):
        og = 4.48131e-7 * tcmb ** 4 / H ** 2
        self.orad = (og + neff * og * (7 / 8) * (4 / 11) ** (4 / 3)) if rad else 0.0
        self.ol = 1 - OM - self.orad
        zz = np.linspace(0, 6, 60001)
        ez = self.E(zz)
        f = 1 / ez
        chi = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(zz))]) * C / (100e3 * H) * MPC / MPC  # Mpc
        self._z, self._chi = zz, chi * 1.0

    def E(self, z):
        z = np.asarray(z, float)
        return np.sqrt(OM * (1 + z) ** 3 + self.orad * (1 + z) ** 4 + self.ol)

    def rho_crit(self, z):  # kg/m^3
        H0 = 100e3 * H / MPC
        return 3 * (H0 * self.E(z)) ** 2 / (8 * np.pi * G)

    def chi_mpc(self, z):
        return np.interp(z, self._z, self._chi)

    def dA_mpc(self, z):
        return self.chi_mpc(z) / (1 + np.asarray(z, float))

    def r200_m(self, m_msun, z):
        return (3 * m_msun * MSUN / (4 * np.pi * 200 * self.rho_crit(z))) ** (1 / 3)


def b12_params(m_msun, z):
    m = np.asarray(m_msun, float) / 1e14; zp = 1 + np.asarray(z, float)
    p0 = 0.5176 * 18.1 * m ** 0.154 * zp ** -0.758
    xc = 0.497 * m ** -0.00865 * zp ** 0.731
    beta = 4.35 * m ** 0.0393 * zp ** 0.415
    return p0, xc, beta


def lee_params(m_msun, z):
    mh = np.asarray(m_msun, float) * H; zp = 1 + np.asarray(z, float)
    r = mh / 10 ** 13.60
    rs = np.minimum(mh, 10 ** 14.8) / 10 ** 13.60
    p0 = 2.8 * np.where(r < 1, r ** 1.08, r ** 0.80) * zp ** -1.89
    xc = 2.10 * rs ** -0.35 * zp ** 0.53
    beta = 9.4 * rs ** -0.06 * zp ** 0.60
    return p0, xc, beta


def g3d(x, xc, beta):
    s = x / xc
    return s ** -0.3 * (1 + s) ** (-beta)


_U, _W = leggauss(400)


def los(X, xc, beta, L=1e5):
    """2 int_0^L g(sqrt(X^2+l^2)) dl, vectorised; l = X sinh(u), split u range into 8 panels of 400-point Gauss-Legendre."""
    X = np.atleast_1d(np.asarray(X, float))
    xc = np.broadcast_to(xc, X.shape); beta = np.broadcast_to(beta, X.shape)
    umax = np.arcsinh(L / X)
    edges = np.linspace(0, 1, 9)
    out = np.zeros_like(X)
    for a, b in zip(edges[:-1], edges[1:]):
        u = (a + (b - a) * (_U[:, None] + 1) / 2) * umax[None, :]
        w = (b - a) / 2 * _W[:, None] * umax[None, :]
        r = X[None, :] * np.cosh(u)
        out += np.sum(w * g3d(r, xc[None, :], beta[None, :]) * X[None, :] * np.cosh(u), axis=0)
    return 2 * out


def y_of_X(model, m_msun, z, X, cos):
    p0, xc, beta = (b12_params if model == "b12" else lee_params)(m_msun, z)
    rho = cos.rho_crit(z)
    r200 = cos.r200_m(m_msun, z)
    p200 = 200 * G * m_msun * MSUN * rho * FB / (2 * r200)  # Pa (physical)
    return SIGMA_T / (ME * C ** 2) * p200 * r200 * p0 * los(X, xc, beta)


_U96, _W96 = leggauss(96)


def los_fast(X, xc, beta, L=1e5):
    """Same integral as los() with 3 panels x 96 Gauss-Legendre nodes (agrees with los() to ~1e-14 on test cases)."""
    X = np.atleast_1d(np.asarray(X, float))
    xc = np.broadcast_to(xc, X.shape); beta = np.broadcast_to(beta, X.shape)
    umax = np.arcsinh(L / X)
    out = np.zeros_like(X)
    for a, b in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1)):
        u = (a + (b - a) * (_U96[:, None] + 1) / 2) * umax[None, :]
        w = (b - a) / 2 * _W96[:, None] * umax[None, :]
        r = X[None, :] * np.cosh(u)
        out += np.sum(w * g3d(r, xc[None, :], beta[None, :]) * r, axis=0)
    return 2 * out
