"""Lee et al. 2022 (arXiv:2205.01710v1) density/pressure fits, Battaglia 2012 pressure, Battaglia 2016
density (XGPaint convention) and observational anchors, in plain numpy (CGS unless noted).

Conventions (as printed in Lee22, TeX lines 357-411 and Tables 1, 3, 7, 8):
  x = r / R200c, R200c physical, rho_cr(z) = 3 H0^2/(8 pi G) [Om (1+z)^3 + OL]
  n200 = 200 rho_cr f_b / (X_H m_p),  P200 = 200 G M200 rho_cr f_b / (2 R200)
  nbar = n_e/n200 = n0 (x/xc')^gamma [1 + (x/xc')^alpha]^(-beta'),  alpha = 1, gamma = -0.3  (same for P)
  extended form: A = A0 (1+z)^az (c/10)^ac (M/Mcut)^{am | am'}; masses M200c in h^-1 Msun.
All no-c parameters use the common Mcut pivot (eq. 12 as printed), exactly like the production
HalfDome option n0_pivot=:mcut. Shape parameters can be frozen above the fitted mass range
(shape_clip_hinv, default 10^14.8 h^-1 Msun, as in production).
"""
import numpy as np

# ---------------------------------------------------------------- constants (CGS)
G = 6.67430e-8
M_P = 1.67262192369e-24
M_E = 9.1093837015e-28
MSUN = 1.98847e33
MPC = 3.0856775814913673e24
KPC = MPC / 1e3
PC = MPC / 1e6
KEV = 1.602176634e-9
KMS_PER_MPC_S = 1e5 / MPC           # 1 km/s/Mpc in s^-1
X_H = 0.76
MU = 4.0 / (3.0 + 5.0 * X_H)          # fully ionised H+He mean molecular weight, 0.5882
MU_E = 2.0 / (1.0 + X_H)              # mass per free electron / m_p (fully ionised)
M_PER_E_XGP = M_E + (2 * X_H / (X_H + 1)) * M_P + ((1 - X_H) / (2 * (1 + X_H))) * 4 * M_P  # XGPaint ne2d
PE_OVER_PTH = 0.5176                  # XGPaint / Battaglia12 electron-to-thermal pressure


class Cosmo:
    def __init__(self, name, om, ob, h):
        self.name, self.om, self.ob, self.h = name, om, ob, h
        self.fb = ob / om

    def E2(self, z):
        return self.om * (1 + z) ** 3 + 1 - self.om

    def rho_crit(self, z):
        """physical critical density [g cm^-3]"""
        H0 = 100 * self.h * KMS_PER_MPC_S
        return 3 * H0 ** 2 / (8 * np.pi * G) * self.E2(z)

    def r_delta(self, m_msun, z, delta=200.0):
        """physical radius [cm] enclosing mean density delta*rho_crit(z)"""
        return (3 * m_msun * MSUN / (4 * np.pi * delta * self.rho_crit(z))) ** (1 / 3)

    def n200(self, z):
        return 200 * self.rho_crit(z) * self.fb / (X_H * M_P)

    def P200(self, m_msun, z):
        r = self.r_delta(m_msun, z)
        return 200 * G * m_msun * MSUN * self.rho_crit(z) * self.fb / (2 * r)

    def kT200_keV(self, m_msun, z):
        r = self.r_delta(m_msun, z)
        return G * m_msun * MSUN * MU * M_P / (2 * r) / KEV


TNG = Cosmo("TNG300 (Planck15)", 0.3089, 0.0486, 0.6774)
HALFDOME = Cosmo("HalfDome XGPaint", 0.31, 0.049, 0.68)


def gnfw(x, xc, beta, gamma=-0.3, alpha=1.0):
    """Lee22 / Battaglia12 form: (x/xc)^gamma [1+(x/xc)^alpha]^(-beta)"""
    s = x / xc
    return s ** gamma * (1 + s ** alpha) ** (-beta)


def _bpl(ratio, below, above):
    return np.where(ratio < 1, ratio ** below, ratio ** above)


# ---------------------------------------------------------------- Lee22 fits (M in h^-1 Msun)
def lee_n_noc_params(mh, z, shape_clip_hinv=10 ** 14.8):
    """arXiv Table 8 (= MNRAS Table A2): no concentration, BPL in xc' and beta'."""
    mcut = 10 ** 13.61
    r = np.asarray(mh, float) / mcut
    rs = np.minimum(mh, shape_clip_hinv) / mcut
    zp = 1 + np.asarray(z, float)
    n0 = 6.8 * r ** 0.68 * zp ** -2.11
    xc = 7.9 * zp ** -0.67 * _bpl(rs, 0.47, -0.45)
    beta = 19.5 * zp ** -0.31 * _bpl(rs, 0.70, -0.18)
    return n0, xc, beta


def lee_P_noc_params(mh, z, shape_clip_hinv=10 ** 14.8):
    """arXiv Table 7 (= MNRAS Table A1): no concentration, BPL in P0."""
    mcut = 10 ** 13.60
    r = np.asarray(mh, float) / mcut
    rs = np.minimum(mh, shape_clip_hinv) / mcut
    zp = 1 + np.asarray(z, float)
    P0 = 2.8 * _bpl(r, 1.08, 0.80) * zp ** -1.89
    xc = 2.10 * rs ** -0.35 * zp ** 0.53
    beta = 9.4 * rs ** -0.06 * zp ** 0.60
    return P0, xc, beta


def tng_mean_c(mh, z):
    """Repo proxy through the two TNG mean Klypin concentrations quoted in Lee22 sec 2.3."""
    return 5.65 * (np.asarray(mh, float) / 10 ** 13.1) ** np.log10(4.53 / 5.65) * (1 + np.asarray(z, float)) ** -0.47


def lee_n_c_params(mh, z, c=None, shape_clip_hinv=10 ** 14.8):
    """arXiv Table 3: with concentration, BPL in xc' and beta' (paper's preferred density fit)."""
    if c is None:
        c = tng_mean_c(mh, z)
    mcut = 10 ** 13.75
    r = np.asarray(mh, float) / mcut
    rs = np.minimum(mh, shape_clip_hinv) / mcut
    zp = 1 + np.asarray(z, float)
    c10 = c / 10
    n0 = 15.7 * r ** 0.87 * zp ** -2.09 * c10 ** 0.63
    xc = 2.2 * zp ** -0.74 * c10 ** -1.37 * _bpl(rs, -0.06, -1.45)
    beta = 7.5 * zp ** -0.39 * c10 ** -1.11 * _bpl(rs, 0.24, -1.10)
    return n0, xc, beta


def lee_P_c_params(mh, z, c=None, shape_clip_hinv=10 ** 14.8):
    """arXiv Table 1: with concentration, BPL in P0 (paper's preferred pressure fit)."""
    if c is None:
        c = tng_mean_c(mh, z)
    mcut = 10 ** 13.64
    r = np.asarray(mh, float) / mcut
    rs = np.minimum(mh, shape_clip_hinv) / mcut
    zp = 1 + np.asarray(z, float)
    c10 = c / 10
    P0 = 6.0 * _bpl(r, 1.09, 0.76) * zp ** -1.38 * c10 ** 1.00
    xc = 1.02 * rs ** -0.29 * zp ** -0.33 * c10 ** -1.24
    beta = 6.3 * rs ** 0.01 * zp ** -0.07 * c10 ** -0.82
    return P0, xc, beta


def lee_nbar(x, mh, z, which="noc", **kw):
    n0, xc, beta = (lee_n_noc_params if which == "noc" else lee_n_c_params)(mh, z, **kw)
    return n0 * gnfw(x, xc, beta)


def lee_Pbar(x, mh, z, which="noc", **kw):
    P0, xc, beta = (lee_P_noc_params if which == "noc" else lee_P_c_params)(mh, z, **kw)
    return P0 * gnfw(x, xc, beta)


# ---------------------------------------------------------------- Battaglia 2012 / 2016 (M in physical Msun)
def b12_Pth_over_P200(x, m_msun, z):
    """Battaglia+2012 AGN feedback fit (XGPaint defaults): thermal pressure / P200."""
    m = np.asarray(m_msun, float) / 1e14
    zp = 1 + np.asarray(z, float)
    P0 = 18.1 * m ** 0.154 * zp ** -0.758
    xc = 0.497 * m ** -0.00865 * zp ** 0.731
    beta = 4.35 * m ** 0.0393 * zp ** 0.415
    return P0 * gnfw(x, xc, beta)


def b12_Pe_over_P200(x, m_msun, z):
    return PE_OVER_PTH * b12_Pth_over_P200(x, m_msun, z)


def b16_rho_over_fb_rhocrit(x, m_msun, z):
    """XGPaint BattagliaTauProfile (profiles_tau.jl): rho_gas / (f_b rho_crit(z)).
    rho = P0 (x/xc)^gamma [1+(x/xc)^alpha]^(-(beta+gamma)/alpha)."""
    m = np.asarray(m_msun, float) / 1e14
    zp = 1 + np.asarray(z, float)
    P0 = 4.0e3 * m ** 0.29 * zp ** -0.66
    xc = 0.5
    alpha = 0.88 * m ** -0.03 * zp ** 0.19
    beta = 3.83 * m ** 0.04 * zp ** -0.025
    gamma = -0.2
    s = x / xc
    return P0 * s ** gamma * (1 + s ** alpha) ** (-(beta + gamma) / alpha)


def b16_nbar(x, m_msun, z):
    """XGPaint ne2d electron density (0.9 rho_gas / m_per_e) in units of Lee22 n200."""
    return 0.9 * X_H * M_P / (200 * M_PER_E_XGP) * b16_rho_over_fb_rhocrit(x, m_msun, z)


# ---------------------------------------------------------------- NFW mass conversion
def nfw_m_ratio(c200, delta_to=500.0):
    """M_Delta/M200c and R_Delta/R200c for an NFW halo with concentration c200."""
    from scipy.optimize import brentq
    mu = lambda y: np.log(1 + y) - y / (1 + y)
    # mean density inside s=R/R200: 200 * mu(c s)/mu(c) / s^3 = delta_to
    f = lambda s: 200 * mu(c200 * s) / mu(c200) / s ** 3 - delta_to
    s = brentq(f, 1e-3, 1.0 if delta_to >= 200 else 10.0)
    return mu(c200 * s) / mu(c200), s


def duffy08_c200c(m_msun, z, h):
    return 5.71 * (m_msun * h / 2e12) ** -0.084 * (1 + z) ** -0.47


# ---------------------------------------------------------------- observations
def arnaud10_upp_P(r_over_r500, m500_msun, z, cosmo, mass_dependence=True):
    """Arnaud et al. 2010 (A&A 517, A92) eqs. 12-13 and 7-8: electron pressure P = n_e kT [keV cm^-3].
    Masses are Y_X-based (HSE-calibrated) M500."""
    h70 = cosmo.h / 0.7
    x = np.asarray(r_over_r500, float)
    ez = np.sqrt(cosmo.E2(z))
    mm = m500_msun / (3e14 / h70)
    P500 = 1.65e-3 * ez ** (8 / 3) * mm ** (2 / 3) * h70 ** 2
    P0, c500, g, a, b = 8.403 * h70 ** -1.5, 1.177, 0.3081, 1.0510, 5.4905
    p = P0 / ((c500 * x) ** g * (1 + (c500 * x) ** a) ** ((b - g) / a))
    if mass_dependence:
        ap = 0.12
        app = 0.10 - (ap + 0.10) * (x / 0.5) ** 3 / (1 + (x / 0.5) ** 3)
        p = p * mm ** (ap + app)
    return P500 * p
