"""Independent re-implementation (verify_1) of Lee et al. 2022 no-concentration density fit
(arXiv:2205.01710 Table 8, TeX sample631.tex lines 640-650) and a minimal halo model for
the mean / per-crossing halo DM inside the R200c sphere.

Written from the TeX source only; does not import any repository module.
"""
import numpy as np
from scipy import integrate

# ---- constants (CODATA / IAU) ----
G_CGS = 6.67430e-8
MP_G = 1.67262192e-24
MSUN_G = 1.98847e33
MPC_CM = 3.0856775814913673e24
PC_CM = 3.0856775814913673e18
C_KMS = 299792.458
XH = 0.76

# ---- TNG300 / Planck15 cosmology (Lee22 Sec 2.1) ----
H = 0.6774
OM = 0.3089
OB = 0.0486
OL = 0.6911
FB = OB / OM


def E2(z):
    return OM * (1 + z) ** 3 + OL  # Lee22 eq. 5 (no radiation)


def rho_crit_cgs(z):
    H0 = 100 * H * 1e5 / MPC_CM  # s^-1
    return 3 * H0 ** 2 / (8 * np.pi * G_CGS) * E2(z)


def r200c_mpc(M_msun, z):
    """physical R200c [Mpc] for M200c in physical Msun"""
    return (3 * M_msun * MSUN_G / (4 * np.pi * 200 * rho_crit_cgs(z))) ** (1 / 3) / MPC_CM


def n200_cgs(z):
    """Lee22 eq. 9 as printed: 200 rho_cr Omega_b/(X_H m_p Omega_m) [cm^-3]"""
    return 200 * rho_crit_cgs(z) * FB / (XH * MP_G)


# ---- Table 8 (no concentration, BPL in x'c and beta'), log10 Mcut = 13.61 h^-1 Msun ----
T8 = dict(n0=(6.8, 0.68, 0.68, -2.11), xc=(7.9, 0.47, -0.45, -0.67), beta=(19.5, 0.70, -0.18, -0.31))
LOG_MCUT_T8 = 13.61


def _bpl(par, Mh, z, logMcut):
    A0, am, amp, az = par
    s = np.asarray(Mh, float) / 10 ** logMcut
    return A0 * (1 + z) ** az * np.where(s < 1, s ** am, s ** amp)


def lee22_params(Mh, z, table=T8, logMcut=LOG_MCUT_T8):
    """Mh = M200c in h^-1 Msun"""
    return (_bpl(table['n0'], Mh, z, logMcut), _bpl(table['xc'], Mh, z, logMcut),
            _bpl(table['beta'], Mh, z, logMcut))


def gnfw(x, n0, xc, beta, gamma=-0.3, alpha=1.0):
    u = x / xc
    return n0 * u ** gamma * (1 + u ** alpha) ** (-beta)


def mean_nbar_inside(Mh, z, xmax=1.0):
    """volume-mean <n_e/n200> inside x<xmax: 3/xmax^3 int_0^xmax nbar x^2 dx; integrate in ln x."""
    n0, xc, b = lee22_params(Mh, z)
    f = lambda t: gnfw(np.exp(t), n0, xc, b) * np.exp(3 * t)
    val, err = integrate.quad(f, np.log(1e-8), np.log(xmax), limit=400, epsabs=0, epsrel=1e-10)
    return 3 * val / xmax ** 3


MGAS_PER_NBAR = 2.0 / ((1 + XH) * XH)  # mu_e/X_H: M_gas/(f_b M200) per unit <n_e/n200>


def disc_mean_dm(M_msun, z, nbar_mean):
    """<DM>_disc (observer frame, pc cm^-3) over the R200c disc for a sphere-truncated profile.
    = N_e(<R200)/(pi R200^2)/(1+z) = <n_e>_vol * 4/3 R200 /(1+z)."""
    R = r200c_mpc(M_msun, z) * MPC_CM / PC_CM  # pc
    return nbar_mean * n200_cgs(z) * 4.0 / 3.0 * R / (1 + z)


def chord_dm(M_msun, z, x, n0, xc, b, nl=400):
    """DM(x) (observer frame) of chord inside sphere at impact x=b/R200c; vectorized over x."""
    R = r200c_mpc(M_msun, z) * MPC_CM / PC_CM
    x = np.atleast_1d(x)
    L = np.sqrt(np.clip(1 - x ** 2, 0, None))
    # Gauss-Legendre in l on [0, L] with substitution l = L*s^2 to handle cusp near x->0 at l=0
    s, w = np.polynomial.legendre.leggauss(nl)
    s = 0.5 * (s + 1); w = 0.5 * w
    l = L[:, None] * s[None, :] ** 2
    dl = L[:, None] * 2 * s[None, :]
    r = np.sqrt(x[:, None] ** 2 + l ** 2)
    r = np.maximum(r, 1e-12)
    integ = (gnfw(r, n0, xc, b) * dl * w[None, :]).sum(1)
    return 2 * R / (1 + z) * integ * n200_cgs(z)
