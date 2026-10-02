"""Independent re-implementation (numpy, no XGPaint) of the quantities paint_ksz_websky.jl uses, for a one-halo prediction of
the idark kSZ maps (review 2026-10-02). Cosmology mirrors XGPaint.get_cosmology (flat LCDM + photons + 3.046 massless nu)."""
import numpy as np
from scipy.optimize import brentq

C_KMS = 299792.458
MPC = 3.0856775814913673e22          # m
MSUN = 1.98847e30                     # kg (XGPaint.M_sun)
G_SI = 6.67430e-11
MP = 1.67262192369e-27
ME = 9.1093837015e-31
SIGT = 6.6524587321e-29
TCMB_UK = 2.7255e6
XH = 0.76
M_PER_E = ME + 2 * XH / (XH + 1) * MP + (1 - XH) / (2 * (1 + XH)) * 4 * MP   # XGPaint ne2d mass per free electron
YP = 0.24
EPM_XGP = 0.9 / M_PER_E                 # electrons per kg of gas, XGPaint ne2d
EPM_WS = (1 - YP / 2) / MP              # electrons per kg of gas, WebSky eq. 3.15
ARCMIN = np.pi / 180 / 60


class Cosmo:
    def __init__(self, h, Ob, Oc, Tcmb=2.7255, Neff=3.046, radiation=True):
        self.h, self.Ob, self.Oc = h, Ob, Oc
        self.Om = Ob + Oc
        og = 4.48131e-7 * Tcmb ** 4 / h ** 2
        self.Or = og * (1 + Neff * 7 / 8 * (4 / 11) ** (4 / 3)) if radiation else 0.0
        self.OL = 1 - self.Om - self.Or
        self.fb = Ob / self.Om
        self.H0 = 100 * h * 1e3 / MPC                       # s^-1
        self.rhoc0 = 3 * self.H0 ** 2 / (8 * np.pi * G_SI)    # kg m^-3
        zz = np.concatenate([np.linspace(0, 0.1, 20001)[:-1], np.linspace(0.1, 6, 120001)])
        Ez = self.E(zz)
        chi = np.concatenate([[0], np.cumsum(0.5 * (1 / Ez[1:] + 1 / Ez[:-1]) * np.diff(zz))]) * C_KMS / (100 * h)  # Mpc
        self._z, self._chi = zz, chi

    def E(self, z):
        z = np.asarray(z, float)
        return np.sqrt(self.Or * (1 + z) ** 4 + self.Om * (1 + z) ** 3 + self.OL)

    def chi(self, z):           # comoving distance, Mpc
        return np.interp(z, self._z, self._chi)

    def DA(self, z):            # angular diameter distance, Mpc
        return self.chi(z) / (1 + np.asarray(z, float))

    def rhoc(self, z):          # kg m^-3, physical
        return self.rhoc0 * self.E(z) ** 2

    def R200c(self, M, z):      # Mpc physical, M in Msun
        return (3 * np.asarray(M, float) * MSUN / (4 * np.pi * 200 * self.rhoc(z))) ** (1 / 3) / MPC

    def theta200c(self, M, z):  # radians (atan as XGPaint.angular_size)
        return np.arctan(self.R200c(M, z) / self.DA(z))


def mnfw(x):
    return np.log1p(x) - x / (1 + x)


def ratio_c7_table(Om, c=7.0, zmax=5.0, n=5001):
    """M200c/M200m for an NFW halo with c200m = c, Omega_m(z) without radiation (as the painter)."""
    zt = np.linspace(0, zmax, n)
    out = np.empty(n)
    for i, z in enumerate(zt):
        omz = Om * (1 + z) ** 3 / (Om * (1 + z) ** 3 + 1 - Om)
        # mean density inside x (units of r_s) relative to that inside c: (m(x)/x^3)/(m(c)/c^3) = 1/Omega_m(z) at R200c
        f = lambda x: mnfw(x) / x ** 3 - mnfw(c) / (c ** 3 * omz)
        x = brentq(f, 1e-4, c, xtol=1e-14, rtol=1e-14)
        out[i] = mnfw(x) / mnfw(c)
    return zt, out


# ---------------------------------------------------------------- profiles: 3-D electron density [m^-3] as a function of x = r/R200c
def b16_params(M, z):
    m = M / 1e14
    z1 = 1 + z
    P0 = 4e3 * m ** 0.29 * z1 ** -0.66
    a = 0.88 * m ** -0.03 * z1 ** 0.19
    b = 3.83 * m ** 0.04 * z1 ** -0.025
    return P0, 0.5, a, b, -0.2


def ne_b16(cos, x, M, z, epm):
    """XGPaint BattagliaTauProfile: rho_gas = f_b rho_crit(z) P0 (x/xc)^g [1+(x/xc)^a]^(-(b+g)/a); n_e = epm rho_gas."""
    P0, xc, a, b, g = b16_params(M, z)
    xb = x / xc
    shape = xb ** g * (1 + xb ** a) ** (-(b + g) / a)
    return epm * cos.fb * cos.rhoc(z) * P0 * shape


def lee22_params(M, z, h, shape_clip):
    """Lee+22 Appendix A2 no-concentration fit, n0 pivot M_cut (eq. 12 form), shape (x_c, beta') frozen above shape_clip."""
    z1 = 1 + z
    mcut = 10 ** 13.61 / h
    ms = min(M, shape_clip)
    r = M / mcut
    rs = ms / mcut
    n0 = 6.8 * r ** 0.68 * z1 ** -2.11
    xc = 7.9 * z1 ** -0.67 * (rs ** 0.47 if rs < 1 else rs ** -0.45)
    bp = 19.5 * z1 ** -0.31 * (rs ** 0.70 if rs < 1 else rs ** -0.18)
    return n0, xc, bp


def ne_lee22corr(cos, x, M, z, shape_clip):
    """literal eq. 9 density x (0.6774 (1+z))^3 X_H^2: n_e = n0 f(x) n200 with n200 = 200 rho_crit f_b/(X_H m_p),
    f = (x/x_c)^-0.3 (1 + x/x_c)^-beta'."""
    n0, xc, bp = lee22_params(M, z, cos.h, shape_clip)
    n200 = 200 * cos.rhoc(z) * cos.fb / (XH * MP)
    corr = (0.6774 * (1 + z)) ** 3 * XH ** 2
    xb = x / xc
    return corr * n0 * n200 * xb ** -0.3 * (1 + xb) ** (-bp)


def ne_lee22pipe(cos, x, M, z, shape_clip):
    """normalization=:xgpaint_ne2d (x 0.9 X_H m_p / m_per_e = 0.602 on the literal density), redshift_scaling=:physical."""
    corr = (0.6774 * (1 + z)) ** 3 * XH ** 2
    return ne_lee22corr(cos, x, M, z, shape_clip) / corr * (0.9 * XH * MP / M_PER_E)


def tau_ell_cell(cos, M, z, ells, model, X, nx=None, comp_m200m=None, shape_clip=np.inf, scale=1.0):
    """Exact flat-sky Fourier transform of the projected, spherically truncated profile (= 3-D FT):
    tau_ell = sigma_T / D_A^2 int_0^{X R200c} 4 pi r^2 n_e(r) j0(k r) dr, k = ell / D_A.
    Optional WebSky compensation: minus a uniform sphere with f_b M200m of gas at mean density 3 rho_m(z), n_e with (1-Yp/2)."""
    R = cos.R200c(M, z) * MPC            # m
    DA = cos.DA(z) * MPC                  # m
    if nx is None:   # Simpson on a uniform grid with k R200c X / nx < 0.02 at the largest ell (min 801, max 40001 points)
        nx = int(min(40001, max(801, 50 * np.max(ells) * R / DA * X)))
        nx += 1 - nx % 2
    x = np.linspace(0.0, X, nx)
    x[0] = 1e-12
    if model == 'b16_xgp':
        ne = ne_b16(cos, x, M, z, EPM_XGP)
    elif model == 'b16_ws':
        ne = ne_b16(cos, x, M, z, EPM_WS)
    elif model == 'lee22corr':
        ne = ne_lee22corr(cos, x, M, z, shape_clip)
    elif model == 'lee22pipe':
        ne = ne_lee22pipe(cos, x, M, z, shape_clip)
    else:
        raise ValueError(model)
    ne = ne * scale
    k = np.asarray(ells, float) / DA
    kr = np.outer(k, x * R)
    j0 = np.sinc(kr / np.pi)
    f = 4 * np.pi * (x * R) ** 2 * ne * j0
    f[:, 0] = 0.0
    # Simpson on the uniform grid
    w = np.ones(nx); w[1:-1:2] = 4; w[2:-1:2] = 2
    Ne = (f * w).sum(1) * (x[1] - x[0]) * R / 3
    tau = SIGT * Ne / DA ** 2
    if comp_m200m is not None:
        rhom = cos.Om * cos.rhoc0 * (1 + z) ** 3
        R3 = (comp_m200m * MSUN / (4 * np.pi * rhom)) ** (1 / 3)
        Nsph = cos.fb * comp_m200m * MSUN * EPM_WS
        q = k * R3
        W = np.where(q < 1e-3, 1 - q ** 2 / 10, 3 * (np.sin(q) - q * np.cos(q)) / np.maximum(q, 1e-30) ** 3)
        tau = tau - SIGT * Nsph * W / DA ** 2
    return tau


def dl_from_cl(ells, cl):
    ells = np.asarray(ells, float)
    return ells * (ells + 1) / (2 * np.pi) * TCMB_UK ** 2 * cl
