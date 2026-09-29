"""Independent (verify_3) implementation of the Lee et al. 2022 (arXiv:2205.01710) no-concentration
fits, written directly from the TeX source (sample631.tex, Tables 7 and 8 = appendix tables
'table:7' and 'table:8'; eqs. 9-11), plus helpers for the C4 adversarial checks.

Conventions
  * masses M200c in h^-1 Msun (as in the paper), pivot = Mcut (the audit showed the Mcut pivot
    reproduces Fig. 5; the eq. 11 "10^14 Msun" pivot does not)
  * x = r / R200c
  * nhat = <n_e/n200>, phat = <P_e/P200> with the printed P200, n200 (eq. 9)
  * T200 defined by k T200 = G M200 mu m_p / (2 R200) with mu = 4/(3+5 X_H) (fully ionised H+He)
    so that P200 = (200 f_b rho_cr / (mu m_p)) k T200  (the B12 thermal-pressure normalisation)
"""
import numpy as np
from scipy import integrate

X_H = 0.76
MU = 4.0 / (3.0 + 5.0 * X_H)          # 0.588235
MU_E = 2.0 / (1.0 + X_H)              # 1.136364
PE_OVER_PTH = MU / MU_E               # 0.517647

# TNG300 (Planck15) cosmology, Lee22 Sec. 2.1
OM, OB, H, OL = 0.3089, 0.0486, 0.6774, 0.6911
FB = OB / OM

# physical constants (CGS)
G_CGS = 6.67430e-8
MSUN_G = 1.98847e33
MPC_CM = 3.0856775814913673e24
M_P_G = 1.67262192369e-24
M_E_G = 9.1093837015e-28
KEV_ERG = 1.602176634e-9
SIGMA_T = 6.6524587321e-25
C_CMS = 2.99792458e10
RHO_CRIT0_H2 = 2.77536627e11          # Msun Mpc^-3 h^-2

# ---- Lee22 Table 8 (density, no concentration), Table 7 (pressure, no concentration)
# each entry: (A0, am, am_prime, az); am_prime=None means same slope on both sides of Mcut
TABLE8 = dict(n0=(6.8, 0.68, None, -2.11), xc=(7.9, 0.47, -0.45, -0.67),
              beta=(19.5, 0.70, -0.18, -0.31), log10Mcut=13.61)
TABLE7 = dict(P0=(2.8, 1.08, 0.80, -1.89), xc=(2.10, -0.35, None, 0.53),
              beta=(9.4, -0.06, None, 0.60), log10Mcut=13.60)
ALPHA, GAMMA = 1.0, -0.3


def _param(entry, m_h, z, log10mcut):
    a0, am, amp, az = entry
    mr = m_h / 10.0 ** log10mcut
    slope = am if (amp is None or mr < 1.0) else amp
    return a0 * mr ** slope * (1.0 + z) ** az


def density_params(m_h, z):
    t = TABLE8
    return (_param(t["n0"], m_h, z, t["log10Mcut"]), _param(t["xc"], m_h, z, t["log10Mcut"]),
            _param(t["beta"], m_h, z, t["log10Mcut"]))


def pressure_params(m_h, z):
    t = TABLE7
    return (_param(t["P0"], m_h, z, t["log10Mcut"]), _param(t["xc"], m_h, z, t["log10Mcut"]),
            _param(t["beta"], m_h, z, t["log10Mcut"]))


def gnfw(x, a0, xc, beta):
    u = np.asarray(x, float) / xc
    return a0 * u ** GAMMA * (1.0 + u ** ALPHA) ** (-beta)


def dlng_dlnx(x, xc, beta):
    u = np.asarray(x, float) / xc
    return GAMMA - beta * ALPHA * u ** ALPHA / (1.0 + u ** ALPHA)


def nhat(x, m_h, z):
    return gnfw(x, *density_params(m_h, z))


def phat(x, m_h, z):
    return gnfw(x, *pressure_params(m_h, z))


def phat_prime(x, m_h, z):
    """d phat / dx"""
    p0, xc, beta = pressure_params(m_h, z)
    return gnfw(x, p0, xc, beta) * dlng_dlnx(x, xc, beta) / np.asarray(x, float)


# ---- cosmology helpers (physical units, no h unless stated)
def E(z):
    return np.sqrt(OM * (1 + z) ** 3 + OL)


def rho_crit(z):
    """physical critical density, Msun / Mpc^3"""
    return RHO_CRIT0_H2 * H ** 2 * E(z) ** 2


def r200c_mpc(m_msun, z):
    return (3.0 * m_msun / (4.0 * np.pi * 200.0 * rho_crit(z))) ** (1.0 / 3.0)


def kT200_keV(m_msun, z):
    r_cm = r200c_mpc(m_msun, z) * MPC_CM
    return G_CGS * m_msun * MSUN_G * MU * M_P_G / (2.0 * r_cm) / KEV_ERG


def n200_cm3(z):
    rc = rho_crit(z) * MSUN_G / MPC_CM ** 3
    return 200.0 * rc * FB / (X_H * M_P_G)


def P200_keV_cm3(m_msun, z):
    rc = rho_crit(z) * MSUN_G / MPC_CM ** 3
    r_cm = r200c_mpc(m_msun, z) * MPC_CM
    return 200.0 * G_CGS * m_msun * MSUN_G * rc / (2.0 * r_cm) * FB / KEV_ERG


# ---- NFW
def m_nfw(x, c):
    """M(<x R200)/M200 for NFW with concentration c"""
    f = lambda y: np.log1p(y) - y / (1.0 + y)
    return f(c * np.asarray(x, float)) / f(c)


def duffy08_c200c(m_h, z):
    """Duffy+08 full sample, 200c, M in h^-1 Msun"""
    return 5.71 * (m_h / 2e12) ** (-0.084) * (1 + z) ** (-0.47)


def lee22_klypin_c_z0(m_h):
    """log-linear interpolation of the two mean concentrations quoted in Lee22 Sec. 2.2 (z=0):
    c = 5.65 at 10^13.0-13.2 and 4.53 at 10^14.0-14.2 h^-1 Msun (bin centres 13.1, 14.1)"""
    s = np.log10(4.53 / 5.65) / 1.0
    return 5.65 * 10.0 ** (s * (np.log10(m_h) - 13.1))


def vol_int(f, a, b, n=4000):
    """int_a^b f(x) 3 x^2 dx on a log grid"""
    lx = np.linspace(np.log(a), np.log(b), n)
    x = np.exp(lx)
    return integrate.simpson(f(x) * 3.0 * x ** 3, x=lx)


def nhat_hse(x, m_h, z, c, pressure_scale=1.0, fnt=0.0):
    """electron density (in units of the printed n200) that the Lee22 pressure fit can hold in
    hydrostatic equilibrium in an NFW potential (gas self-gravity neglected):
      rho_gas = -(dP_tot/dr) / (G M(<r)/r^2), P_tot = P_th/(1-fnt), P_th = P_e/(mu/mu_e)
      => nhat_HSE = -(X_H/(2 mu)) x^2 phat'(x) / m(x) / (1-fnt)
    pressure_scale multiplies phat (e.g. (1+z)^2 to undo the a^2 bookkeeping error)."""
    return -(X_H / (2.0 * MU)) * x ** 2 * pressure_scale * phat_prime(x, m_h, z) / m_nfw(x, c) / (1.0 - fnt)
