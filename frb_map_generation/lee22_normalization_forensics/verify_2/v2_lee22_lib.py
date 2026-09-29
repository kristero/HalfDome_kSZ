"""Independent re-implementation (verify_2) of the Lee+22 (arXiv:2205.01710) fits used for the
C3 adversarial check. Written from the arXiv TeX source only (Tables 1, 3, 7, 8; eqs. 9-12):

  g(x) = A0 (x/xc)^gamma [1 + (x/xc)^alpha]^(-beta),  alpha = 1, gamma = -0.3
  A = A0 (1+z)^az (c/10)^ac x (M/Mcut)^am  (M<Mcut) or (M/Mcut)^am' (M>Mcut); M = M200c [h^-1 Msun]

Cosmology: TNG300 / Planck15 (Om=0.3089, Ob=0.0486, h=0.6774). No repository code is imported.
"""
import numpy as np
from scipy import integrate

OM, OB, H = 0.3089, 0.0486, 0.6774
OL = 1.0 - OM
FB = OB / OM
XH = 0.76
MP_G = 1.67262192e-24
MSUN_G = 1.98847e33
MPC_CM = 3.0856775814913673e24
PC_CM = 3.0856775814913673e18
C_KMS = 299792.458
G_CGS = 6.67430e-8

# (A0, am, am', az, ac) ; am' = None -> single power law
TABLE8_N = dict(n0=(6.8, 0.68, None, -2.11, 0.0), xc=(7.9, 0.47, -0.45, -0.67, 0.0),
                beta=(19.5, 0.70, -0.18, -0.31, 0.0), log10Mcut=13.61)
TABLE7_P = dict(P0=(2.8, 1.08, 0.80, -1.89, 0.0), xc=(2.10, -0.35, None, 0.53, 0.0),
                beta=(9.4, -0.06, None, 0.60, 0.0), log10Mcut=13.60)
TABLE3_N = dict(n0=(15.7, 0.87, None, -2.09, 0.63), xc=(2.2, -0.06, -1.45, -0.74, -1.37),
                beta=(7.5, 0.24, -1.10, -0.39, -1.11), log10Mcut=13.75)
TABLE1_P = dict(P0=(6.0, 1.09, 0.76, -1.38, 1.00), xc=(1.02, -0.29, None, -0.33, -1.24),
                beta=(6.3, 0.01, None, -0.07, -0.82), log10Mcut=13.64)


def E2(z):
    return OM * (1 + z) ** 3 + OL


def param(p, mh, z, c=10.0, mcut=None):
    A0, am, amp, az, ac = p
    x = mh / mcut
    slope = am if (amp is None or x < 1) else amp
    return A0 * x ** slope * (1 + z) ** az * (c / 10.0) ** ac


def shape(table, mh, z, c=10.0):
    """Return (amp, xc, beta) for a table; mh in h^-1 Msun."""
    mcut = 10 ** table["log10Mcut"]
    keys = [k for k in table if k not in ("xc", "beta", "log10Mcut")]
    amp = param(table[keys[0]], mh, z, c, mcut)
    xc = param(table["xc"], mh, z, c, mcut)
    beta = param(table["beta"], mh, z, c, mcut)
    return amp, xc, beta


def gnfw(x, amp, xc, beta, gamma=-0.3, alpha=1.0):
    u = x / xc
    return amp * u ** gamma * (1 + u ** alpha) ** (-beta)


def vol_mean(table, mh, z, c=10.0, xmax=1.0, xmin=0.0):
    """3/xmax^3 int_xmin^xmax x^2 g(x) dx : volume-mean of the fitted dimensionless profile."""
    amp, xc, beta = shape(table, mh, z, c)
    f = lambda lx: np.exp(3 * lx) * gnfw(np.exp(lx), amp, xc, beta)
    lo = np.log(max(xmin, 1e-8))
    val, _ = integrate.quad(f, lo, np.log(xmax), limit=400, epsabs=0, epsrel=1e-10)
    return 3.0 * val / xmax ** 3


# Battaglia+12 AGN pressure fit (P_th / P200), M in 1e14 Msun (physical), for comparison only
B12 = dict(P0=(18.1, 0.154, -0.758), xc=(0.497, -0.00865, 0.731), beta=(4.35, 0.0393, 0.415))


def b12_vol_mean(m_msun, z, xmax=1.0):
    m14 = m_msun / 1e14
    P0 = B12["P0"][0] * m14 ** B12["P0"][1] * (1 + z) ** B12["P0"][2]
    xc = B12["xc"][0] * m14 ** B12["xc"][1] * (1 + z) ** B12["xc"][2]
    be = B12["beta"][0] * m14 ** B12["beta"][1] * (1 + z) ** B12["beta"][2]
    f = lambda lx: np.exp(3 * lx) * gnfw(np.exp(lx), P0, xc, be)
    val, _ = integrate.quad(f, np.log(1e-8), np.log(xmax), limit=400, epsrel=1e-10)
    return 3.0 * val / xmax ** 3
