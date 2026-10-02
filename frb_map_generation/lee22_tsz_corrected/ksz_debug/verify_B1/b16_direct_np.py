"""Independent numpy implementation of the B16 tau ChordMean quantity g(theta, M, z) that XGPaint caches
(ChordMeanProfile(BattagliaTauProfile(Omega_c=0.261, Omega_b=0.049, h=0.68), 4)), plus an independent replica
of the cached interpolator (tensor-product natural cubic spline of log g on the cache grid; Interpolations'
Cubic(Line(OnGrid())) == natural spline). Written for the B1 verification (2026-10-02)."""
import numpy as np
import h5py
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

# ---------------- constants (CODATA2018 as in PhysicalConstants.jl) ----------------
G = 6.67430e-11
ME = 9.1093837015e-31
MP = 1.67262192369e-27
SIGMA_T = 6.6524587321e-29
MSUN = 1.98847e30
MPC = 3.0856775814913673e22
C_KMS = 299792.458

H = 0.68; OB = 0.049; OC = 0.261; OM = OB + OC
TCMB = 2.7255; NEFF = 3.046
OG = 4.48131e-7 * TCMB**4 / H**2
OR = OG + NEFF * OG * (7 / 8) * (4 / 11) ** (4 / 3)
OL = 1 - OM - OR
FB = OB / OM
XSPH = 4.0


def Ez(z):
    z = np.asarray(z, float)
    return np.sqrt(OM * (1 + z) ** 3 + OR * (1 + z) ** 4 + OL)


def comoving_mpc(z):
    z = np.atleast_1d(np.asarray(z, float))
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        out[i] = quad(lambda t: 1 / Ez(t), 0, zi, epsabs=0, epsrel=1e-12)[0]
    return out * C_KMS / (100 * H)


def dA_mpc(z):
    return comoving_mpc(z) / (1 + np.atleast_1d(np.asarray(z, float)))


def rho_crit(z):  # kg/m^3
    Hz = 100 * H * 1e3 / MPC * Ez(z)  # 1/s
    return 3 * Hz**2 / (8 * np.pi * G)


def params(M, z):
    m = M / 1e14; z1 = 1 + z
    P0 = 4.0e3 * m**0.29 * z1**-0.66
    xc = 0.5
    al = 0.88 * m**-0.03 * z1**0.19
    be = 3.83 * m**0.04 * z1**-0.025
    ga = -0.2
    return P0, xc, al, be, ga


def r200c_m(M, z):
    return np.cbrt(M * MSUN / (4 * np.pi / 3 * 200 * rho_crit(z)))


def theta200(M, z, dA=None):
    if dA is None:
        dA = dA_mpc(z)
    return np.arctan(r200c_m(M, z) / (dA * MPC))


def amplitude(M, z):
    """sigma_T * 0.9/factor * P0 * f_b * rho_crit * r200c  (dimensionless tau per unit LOS integral in R200c)."""
    P0, *_ = params(M, z)
    xH = 0.76
    nH_ne = 2 * xH / (xH + 1); nHe_ne = (1 - xH) / (2 * (1 + xH))
    factor = ME + nH_ne * MP + nHe_ne * 4 * MP
    return SIGMA_T * 0.9 * P0 * FB * rho_crit(z) * r200c_m(M, z) / factor


def gnfw(x, xc, al, be, ga):
    xb = x / xc
    return xb**ga * (1 + xb**al) ** (-(be + ga) / al)


_GL_U, _GL_W = np.polynomial.legendre.leggauss(400)


def chord_integral(x, al, be, xc=0.5, ga=-0.2, X=XSPH, nsplit=4):
    """Integral of gnfw over the chord |l| < sqrt(X^2 - x^2) at projected radius x (vectorised in x).
    l = x sinh(u), r = x cosh(u), dl = r du; Gauss-Legendre on nsplit equal pieces of [0, asinh(L/x)]."""
    x = np.atleast_1d(np.asarray(x, float))
    out = np.zeros_like(x)
    ok = (x > 0) & (x < X)
    xv = x[ok]
    L = np.sqrt((X - xv) * (X + xv))
    umax = np.arcsinh(L / xv)
    tot = np.zeros_like(xv)
    for k in range(nsplit):
        a = umax * k / nsplit; b = umax * (k + 1) / nsplit
        u = 0.5 * (b - a)[:, None] * (_GL_U[None, :] + 1) + a[:, None]
        r = xv[:, None] * np.cosh(u)
        tot += 0.5 * (b - a) * np.sum(_GL_W[None, :] * gnfw(r, xc, al, be, ga) * r, axis=1)
    out[ok] = 2 * tot
    return out


def g_direct(theta, M, z, dA=None):
    """ChordMean g = A * chord_integral(x) / (2L)   (the quantity XGPaint caches)."""
    th200 = theta200(M, z, dA)
    _, xc, al, be, ga = params(M, z)
    x = np.atleast_1d(theta) / th200
    L = np.sqrt((XSPH - x) * (XSPH + x))
    return amplitude(M, z) * chord_integral(x, al, be, xc, ga) / (2 * L)


def tau_direct(theta, M, z, dA=None):
    """Painted tau inside the 4 R200c sphere = A * chord integral."""
    th200 = theta200(M, z, dA)
    _, xc, al, be, ga = params(M, z)
    x = np.atleast_1d(theta) / th200
    return amplitude(M, z) * chord_integral(x, al, be, xc, ga)


# ---------------- cache ----------------
class CacheReplica:
    def __init__(self, path):
        with h5py.File(path, "r") as f:
            def rng(k):
                s = f[k][()]
                return np.linspace(s["start"], s["stop"], int(s["len"]))
            self.logth = rng("prof_logthetas"); self.zs = rng("prof_redshift"); self.logM = rng("prof_logMs")
            y = f["prof_y"][()]  # h5py order (logM, z, logtheta)
        self.logy = np.log(y)
        self.zspline = CubicSpline(self.zs, self.logy, axis=1, bc_type="natural")

    def node_values(self):
        return self.logth, self.zs, self.logM, self.logy

    def eval_z(self, z):
        """log g on the (logM, logtheta) grid at redshift z."""
        return self.zspline(z)  # (128, 512)

    def eval(self, theta, M, z, slab=None):
        if slab is None:
            slab = self.eval_z(z)
        row = CubicSpline(self.logM, slab, axis=0, bc_type="natural")(np.log10(M))  # (512,)
        return np.exp(CubicSpline(self.logth, row, bc_type="natural")(np.log(theta)))
