"""Independent re-implementation (verify_4) of the pieces needed to test claim C5.

Written from scratch from the Lee+22 TeX (arXiv:2205.01710, Table 8, eqs. 7-11) and the
Tinker+08 paper; it does NOT import any repository module.

Units: masses in physical Msun, lengths in physical Mpc unless stated, n_e in cm^-3,
DM in pc cm^-3.
"""
import math
import numpy as np
from scipy import integrate, interpolate

# --- constants (SI) -------------------------------------------------------
G_SI = 6.67430e-11
MSUN_KG = 1.98847e30
MP_KG = 1.67262192369e-27
ME_KG = 9.1093837015e-31
MPC_M = 3.0856775814913673e22
PC_M = 3.0856775814913673e16
C_KMS = 299792.458
X_H = 0.76

# XGPaint ne2d conversion: n_e = 0.9 * rho_gas / m_per_e with m_per_e = m_e + 2 m_p/(1+X_H);
# Lee22 eq. 9: n200 uses 1/(X_H m_p).  Ratio xgpnative / literal:
XGP_OVER_LITERAL = 0.9 * X_H * MP_KG / (ME_KG + 2.0 * MP_KG / (1.0 + X_H))


class Cosmo:
    """Flat LCDM, no radiation (radiation is < 1e-3 of E^2 for z < 2)."""

    def __init__(self, h, om, ob, sigma8=None, ns=None, name=""):
        self.h, self.om, self.ob, self.sigma8, self.ns, self.name = h, om, ob, sigma8, ns, name
        self.ol = 1.0 - om
        self.fb = ob / om
        H0_SI = 100.0 * h * 1e3 / MPC_M
        # critical density today in Msun / Mpc^3 (physical)
        self.rho_c0 = 3 * H0_SI ** 2 / (8 * math.pi * G_SI) / MSUN_KG * MPC_M ** 3
        self.rho_m0 = om * self.rho_c0  # comoving mean matter density
        zg = np.linspace(0.0, 6.0, 60001)
        integrand = 1.0 / self.E(zg)
        chi = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zg))])
        chi *= C_KMS / (100.0 * h)
        self._chi = interpolate.interp1d(zg, chi, kind="cubic")
        self._z_of_chi = interpolate.interp1d(chi, zg, kind="cubic")

    def E(self, z):
        z = np.asarray(z, dtype=float)
        return np.sqrt(self.om * (1 + z) ** 3 + self.ol)

    def rho_crit(self, z):
        return self.rho_c0 * self.E(z) ** 2

    def chi(self, z):
        return self._chi(z)

    def c_over_H(self, z):  # Mpc
        return C_KMS / (100.0 * self.h * self.E(z))

    def r200c(self, m, z):  # physical Mpc
        return (3.0 * np.asarray(m) / (4 * math.pi * 200.0 * self.rho_crit(z))) ** (1.0 / 3.0)

    def om_z(self, z):
        return self.om * (1 + z) ** 3 / self.E(z) ** 2


TNG = Cosmo(0.6774, 0.3089, 0.0486, 0.8159, 0.9667, "TNG300 (Planck15)")
HALFDOME = Cosmo(0.68, 0.31, 0.049, None, None, "HalfDome")


# --- Lee+22 no-concentration density fit (Table 8), Mcut pivot ---------------
LOG10_MCUT = 13.61  # h^-1 Msun


def lee22_params(m_msun, z, h):
    mr = np.asarray(m_msun, dtype=float) * h / 10 ** LOG10_MCUT
    zf = 1.0 + np.asarray(z, dtype=float)
    low = mr < 1.0
    n0 = 6.8 * mr ** 0.68 * zf ** -2.11
    xc = 7.9 * np.where(low, mr ** 0.47, mr ** -0.45) * zf ** -0.67
    beta = 19.5 * np.where(low, mr ** 0.70, mr ** -0.18) * zf ** -0.31
    return n0, xc, beta


def lee22_nbar(x, n0, xc, beta, gamma=-0.3, alpha=1.0):
    u = x / xc
    return n0 * u ** gamma * (1.0 + u ** alpha) ** (-beta)


def n200_cm3(cosmo, z):
    """Lee22 eq. 9: n200 = 200 rho_cr(z) (Ob/Om) / (X_H m_p), physical, cm^-3."""
    rho = 200.0 * cosmo.rho_crit(z) * cosmo.fb  # Msun / Mpc^3
    return rho * MSUN_KG / MPC_M ** 3 / (X_H * MP_KG) * 1e-6


def lee22_volume_mean_nbar(m_msun, z, h):
    """<nbar>_V inside R200c = 3 int_0^1 x^2 nbar dx (literal reading, dimensionless)."""
    n0, xc, beta = lee22_params(m_msun, z, h)
    f = lambda x: 3 * x * x * lee22_nbar(x, n0, xc, beta)
    val, err = integrate.quad(f, 0.0, 1.0, limit=200, epsabs=0, epsrel=1e-10)
    return val


def lee22_electrons_literal(cosmo, m_msun, z):
    """Total electrons inside the R200c sphere (literal), returned as N_e * m_p / Msun
    (i.e. in 'proton-mass-units of Msun' to avoid 1e70 numbers)."""
    I = lee22_volume_mean_nbar(m_msun, z, cosmo.h)
    return cosmo.fb * m_msun / X_H * I


def lee22_chord_dm_literal(cosmo, x, m_msun, z):
    """Observer-frame DM [pc cm^-3] of a ray at impact x = b/R200c through the R200c sphere."""
    if x >= 1.0:
        return 0.0
    n0, xc, beta = lee22_params(m_msun, z, cosmo.h)
    L = math.sqrt(1.0 - x * x)
    f = lambda l: lee22_nbar(math.sqrt(x * x + l * l), n0, xc, beta)
    if x > 0:
        pts = [min(x, 0.5 * L)] if x < L else None
        val, _ = integrate.quad(f, 0.0, L, points=pts, limit=200, epsabs=0, epsrel=1e-9)
    else:
        val, _ = integrate.quad(f, 0.0, L, limit=200, epsabs=0, epsrel=1e-9)
    r200_pc = float(cosmo.r200c(m_msun, z)) * 1e6
    return 2.0 * r200_pc * n200_cm3(cosmo, z) * val / (1.0 + z)


# --- Tinker+08 M200c mass function with CAMB linear P(k) ------------------
_T08_DELTA = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200], float)
_T08_A = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
_T08_a = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
_T08_b = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
_T08_c = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
_spl = {k: interpolate.CubicSpline(np.log10(_T08_DELTA), v) for k, v in
        (("A", _T08_A), ("a", _T08_a), ("b", _T08_b), ("c", _T08_c))}


class Tinker08:
    def __init__(self, cosmo, kmax=200.0):
        import camb
        pars = camb.CAMBparams()
        h = cosmo.h
        pars.set_cosmology(H0=100 * h, ombh2=cosmo.ob * h * h, omch2=(cosmo.om - cosmo.ob) * h * h,
                           mnu=0.0, omk=0.0)
        pars.InitPower.set_params(ns=cosmo.ns, As=2.1e-9)
        pars.set_matter_power(redshifts=[0.0], kmax=kmax)
        pars.NonLinear = camb.model.NonLinear_none
        res = camb.get_results(pars)
        kh, _, pk = res.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=2000)
        s8 = res.get_sigma8_0()
        pk = pk[0] * (cosmo.sigma8 / s8) ** 2
        self.k = kh * h  # 1/Mpc
        self.pk = pk / h ** 3  # Mpc^3
        self.cosmo = cosmo
        # growth factor D(z)/D(0), flat LCDM integral (Heath 1977)
        zg = np.linspace(0, 6, 601)
        D = np.array([self._D_unnorm(z) for z in zg])
        self._D = interpolate.interp1d(zg, D / D[0], kind="cubic")
        lnR = np.linspace(np.log(0.01), np.log(60.0), 800)
        sig = np.array([self._sigma_R(math.exp(r)) for r in lnR])
        self._lnsig = interpolate.CubicSpline(lnR, np.log(sig))

    def _D_unnorm(self, z):
        c = self.cosmo
        a = 1.0 / (1 + z)
        f = lambda ap: 1.0 / (ap * math.sqrt(c.om / ap ** 3 + c.ol)) ** 3
        val, _ = integrate.quad(f, 1e-8, a, epsrel=1e-10)
        return 2.5 * c.om * math.sqrt(c.om / a ** 3 + c.ol) * val

    def _sigma_R(self, R):
        kr = self.k * R
        W = 3 * (np.sin(kr) - kr * np.cos(kr)) / kr ** 3
        return math.sqrt(np.trapz(self.k ** 3 * self.pk * W ** 2 / (2 * math.pi ** 2), np.log(self.k)))

    def growth(self, z):
        return float(self._D(z))

    def sigma(self, m, z):
        R = (3 * np.asarray(m) / (4 * math.pi * self.cosmo.rho_m0)) ** (1 / 3)
        return np.exp(self._lnsig(np.log(R))) * self.growth(z)

    def dndlnm(self, m, z):
        """Comoving Mpc^-3, m = M200c physical Msun."""
        c = self.cosmo
        m = np.asarray(m, float)
        delta_m = 200.0 / c.om_z(z)
        ld = math.log10(delta_m)
        A0, a0, b0, c0 = (float(_spl[k](ld)) for k in ("A", "a", "b", "c"))
        alpha = 10 ** (-(0.75 / math.log10(delta_m / 75.0)) ** 1.2)
        zz = min(z, 3.0)
        A = A0 * (1 + zz) ** -0.14
        a = a0 * (1 + zz) ** -0.06
        b = b0 * (1 + zz) ** -alpha
        R = (3 * m / (4 * math.pi * c.rho_m0)) ** (1 / 3)
        lnR = np.log(R)
        sig = np.exp(self._lnsig(lnR)) * self.growth(z)
        dlnsig_dlnR = self._lnsig(lnR, 1)
        dlnsiginv_dlnm = -dlnsig_dlnR / 3.0
        f = A * ((sig / b) ** (-a) + 1.0) * np.exp(-c0 / sig ** 2)
        return f * c.rho_m0 / m * dlnsiginv_dlnm
