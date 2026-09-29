"""Shared formulas for the Lee22 gas-fraction forensics (subdir D_gas_fractions).

Everything here is written from the paper (Lee et al. 2022, arXiv:2205.01710, TeX source in
frb_map_generation/outputs/lee22_fig5_audit_20260922/reference/source/sample631.tex) and from the
XGPaint source (profiles_tau.jl, model.jl); it does not import the production Julia code. The
production conventions it mirrors (M_cut pivot for every no-c parameter, XGPaint-native factor
0.9 X_H m_p / m_per_e) are validated against the Julia-tabulated HalfDome sphere profiles in
compute_lee22_gas_fractions.py.

Definitions
-----------
Lee22 eq. 9 (as printed):   n200 = 200 rho_cr(z) f_b / (X_H m_p),   f_b = Omega_b/Omega_m
Lee22 eq. 10:               nbar(x) = n0 (x/xc)^gamma [1 + (x/xc)^alpha]^(-beta'),  alpha=1, gamma=-0.3
Physical electron density:  n_e = K(z) * nbar(x) * n200          (K = variant factor, 1 = literal)

Free-electron ("ionized-gas-equivalent") fraction inside radius X R200c:
    f_e(<X) = N_e(<X) / [ f_b M200c (1+X_H)/(2 m_p) ]
            = K * (2/(X_H (1+X_H))) * 3 * n0 * int_0^X g(x) x^2 dx          (for X = 1)
i.e. the gas mass of fully ionized H+He gas (mu_e = 2/(1+X_H), n_e = rho (1+X_H)/(2 m_p)) that carries
those electrons, in units of the cosmic baryon share f_b M200c. If part of the gas is neutral (TNG's
star-forming cells; Konietzka+25 set their electrons to zero), the true gas mass is larger than f_e by
1/(ionized mass fraction), so f_e is a lower limit on M_gas/(f_b M).
"""
import numpy as np
from scipy import integrate, optimize

X_H = 0.76
MU_E_IONIZED = 2.0 / (1.0 + X_H)                      # gas mass per free electron / m_p, fully ionized H+He
F_E_PER_NBAR = MU_E_IONIZED / X_H                     # = 2/(X_H(1+X_H)) = 1.49522
XGP_NATIVE_FACTOR = 0.6016316602183194                # 0.9 X_H m_p / m_per_e (LEE22 fig5 audit)
# recompute XGP_NATIVE_FACTOR from the XGPaint constants as a cross-check
_MP = 1.67262192369e-27
_ME = 9.1093837015e-31
M_PER_E = _ME + (2 * X_H / (X_H + 1)) * _MP + ((1 - X_H) / (2 * (1 + X_H))) * 4 * _MP   # profiles_tau.jl ne2d
assert abs(0.9 * X_H * _MP / M_PER_E - XGP_NATIVE_FACTOR) < 1e-15

# ---------------------------------------------------------------- cosmologies
TNG = dict(name="TNG300 (Planck15)", om=0.3089, ob=0.0486, h=0.6774)
HALFDOME = dict(name="HalfDome", om=0.31, ob=0.049, h=0.68)


def E2(cosmo, z, radiation=False):
    """E^2(z) for flat LCDM. radiation=True reproduces XGPaint.get_cosmology (Tcmb=2.7255, Neff=3.046)."""
    om, h = cosmo["om"], cosmo["h"]
    orad = 0.0
    if radiation:
        og = 4.48131e-7 * 2.7255 ** 4 / h ** 2
        orad = og + 3.046 * og * (7 / 8) * (4 / 11) ** (4 / 3)
    ol = 1 - om - orad
    z = np.asarray(z, float)
    return om * (1 + z) ** 3 + orad * (1 + z) ** 4 + ol


# ---------------------------------------------------------------- Lee22 parameter sets (masses in h^-1 Msun)
def _bpl(mu, below, above):
    mu = np.asarray(mu, float)
    return np.where(mu < 1, mu ** below, mu ** above)


def lee22_noc_params(log10m_h, z, shape_clip_log10m_h=np.inf):
    """arXiv Table 8 (= repo 'Table A2'): no concentration, BPL in x'_c and beta', common M_cut pivot
    (the production n0_pivot=:mcut convention; LEE22 fig5 audit shows it reproduces Fig. 5 to 2-3%)."""
    mcut = 13.61
    mu = 10 ** (np.asarray(log10m_h, float) - mcut)
    mu_s = 10 ** (np.minimum(log10m_h, shape_clip_log10m_h) - mcut)
    opz = 1 + np.asarray(z, float)
    n0 = 6.8 * mu ** 0.68 * opz ** -2.11
    xc = 7.9 * opz ** -0.67 * _bpl(mu_s, 0.47, -0.45)
    beta = 19.5 * opz ** -0.31 * _bpl(mu_s, 0.70, -0.18)
    return n0, xc, beta


def tng_mean_concentration(log10m_h, z):
    """Lee22 Sec. 2.3 means (5.65 at 10^13.1, 4.53 at 10^14.1 h^-1 Msun, z=0) with a Duffy08-like (1+z)^-0.47."""
    return 5.65 * 10 ** ((np.asarray(log10m_h, float) - 13.1) * np.log10(4.53 / 5.65)) * (1 + np.asarray(z, float)) ** -0.47


def lee22_c_params(log10m_h, z, c=None):
    """arXiv Table 3: with concentration, BPL in x'_c and beta', M_cut = 10^13.75 h^-1 Msun."""
    if c is None:
        c = tng_mean_concentration(log10m_h, z)
    mu = 10 ** (np.asarray(log10m_h, float) - 13.75)
    opz = 1 + np.asarray(z, float)
    c10 = np.asarray(c, float) / 10
    n0 = 15.7 * mu ** 0.87 * opz ** -2.09 * c10 ** 0.63
    xc = 2.2 * opz ** -0.74 * c10 ** -1.37 * _bpl(mu, -0.06, -1.45)
    beta = 7.5 * opz ** -0.39 * c10 ** -1.11 * _bpl(mu, 0.24, -1.10)
    return n0, xc, beta


def gshape(x, xc, beta, gamma=-0.3, alpha=1.0):
    s = np.asarray(x, float) / xc
    return s ** gamma * (1 + s ** alpha) ** (-beta)


def shape_moment(xc, beta, X, gamma=-0.3, alpha=1.0, x_lo=0.0):
    """int_{x_lo}^{X} g(x) x^2 dx (x = r/R200c)."""
    f = lambda lx: gshape(np.exp(lx), xc, beta, gamma, alpha) * np.exp(3 * lx)
    lo = np.log(max(x_lo, 1e-12))
    val, _ = integrate.quad(f, lo, np.log(X), epsabs=0, epsrel=1e-11, limit=400)
    return val


def lee22_nbar_volume_mean(n0, xc, beta, X=1.0):
    """Volume mean of nbar = n_e/n200 inside X R200c: 3/X^3 * n0 * int_0^X g x^2 dx."""
    return 3 * n0 * shape_moment(xc, beta, X) / X ** 3


# ---------------------------------------------------------------- NFW overdensity conversion
def _mnfw(y):
    return np.log1p(y) - y / (1 + y)


def nfw_r_over_r200(c200, delta=500.0):
    """x = R_delta/R200c and M_delta/M200c for an NFW halo of concentration c200 (critical overdensities)."""
    target = delta / 200.0
    fun = lambda x: _mnfw(c200 * x) / (_mnfw(c200) * x ** 3) - target
    x = optimize.brentq(fun, 1e-3, 1.0, xtol=1e-14)
    return x, _mnfw(c200 * x) / _mnfw(c200)


# ---------------------------------------------------------------- Battaglia16 as implemented in XGPaint
def b16_xgpaint_params(m_msun, z):
    """BattagliaTauProfile defaults (profiles_tau.jl); M in physical Msun, 1e14 Msun pivot.
    rho_gas = f_b rho_crit(z) P0 (x/xc)^gamma [1+(x/xc)^alpha]^(-(beta+gamma)/alpha)."""
    m = np.asarray(m_msun, float) / 1e14
    opz = 1 + np.asarray(z, float)
    P0 = 4.0e3 * m ** 0.29 * opz ** -0.66
    xc = 0.5
    alpha = 0.88 * m ** -0.03 * opz ** 0.19
    beta = 3.83 * m ** 0.04 * opz ** -0.025
    gamma = -0.2
    return P0, xc, alpha, beta, gamma


def b16_gas_fraction(m_msun, z, X=1.0):
    """M_gas(<X R200c) / (f_b M200c) of XGPaint's Battaglia16 gas density (before the 0.9 factor)."""
    P0, xc, alpha, beta, gamma = b16_xgpaint_params(m_msun, z)
    f = lambda lx: ((np.exp(lx) / xc) ** gamma * (1 + (np.exp(lx) / xc) ** alpha) ** (-(beta + gamma) / alpha)
                    * np.exp(3 * lx))
    val, _ = integrate.quad(f, np.log(1e-12), np.log(X), epsabs=0, epsrel=1e-11, limit=400)
    return 3 * P0 * val / 200.0


def b16_free_electron_fraction(m_msun, z, X=1.0):
    """XGPaint's electrons (0.9 rho_gas / m_per_e) over the fully ionized cosmic share f_b M (1+X_H)/(2 m_p)."""
    return b16_gas_fraction(m_msun, z, X) * 0.9 * (2 * _MP / (1 + X_H)) / M_PER_E
