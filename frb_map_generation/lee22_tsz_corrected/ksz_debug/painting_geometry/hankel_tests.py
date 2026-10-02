"""(2) Hankel normalisation test of poisson_mass_split.jl's discrete scheme:
a_ell = 2 pi sum_j f(theta_j) J0(ell theta_j) theta_j^2 dln, lntheta = range(log(1e-9), log(theta_max), 3000).
(a) Gaussian f = exp(-theta^2/2s^2): exact a_ell = 2 pi s^2 exp(-ell^2 s^2/2) (theta_max >> s).
(b) B16 chord-limited 4 R200c profile: discrete Hankel vs the exact 3-D sinc transform used in onehalo_variants.py.
(c) C_ell normalisation: Poisson C_ell = (N/4pi) a_ell^2 checked with a HEALPix simulation of Gaussian blobs."""
import sys
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/painting_geometry")
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from onehalo_variants import *

def discrete_hankel(f, theta_max, ells, theta_min=1.0181517217181816e-9, n=3000):
    lnt = np.linspace(np.log(max(theta_min, 1e-9)), np.log(theta_max), n)
    dln = lnt[1] - lnt[0]
    t = np.exp(lnt)
    ft = f(t)
    return np.array([2 * np.pi * np.sum(ft * j0(L * t) * t**2 * dln) for L in ells])

ells = np.array([30, 100, 300, 1000, 3000, 5000, 8192.0])
print("(a) Gaussian test")
for s_arcmin in (0.2, 1.0, 5.0):
    s = s_arcmin / 60 * np.pi / 180
    a = discrete_hankel(lambda t: np.exp(-t**2 / (2 * s**2)), 8 * s, ells)
    ex = 2 * np.pi * s**2 * np.exp(-ells**2 * s**2 / 2)
    print("  s=%.1f arcmin  discrete/exact:" % s_arcmin, " ".join("%.6f" % r for r in a / ex))

print("(b) B16 sphere-4 profile: discrete Hankel (Julia scheme) / 3-D sinc transform")
for (M, z) in ((2e13, 0.1), (1e14, 0.5), (5e14, 1.0), (3e13, 2.0), (1e15, 0.05)):
    P0, xc, al, be, ga = b16_params(M, z)
    R = r200c_m(M, z); d = dA(z) * MPC; th = np.arctan(R / d)
    A = NE_PER_KG * f_b * rho_c(z) * P0
    def tau(t):
        out = np.zeros_like(t)
        for i, tt in enumerate(t):
            x = tt / th
            if x >= 4: continue
            L = np.sqrt(16 - x**2)
            out[i] = SIGMA_T * A * R * 2 * quad(lambda l: gnfw(np.hypot(x, l), xc, al, be, ga), 0, L, limit=200, epsabs=0, epsrel=1e-9)[0]
        return out
    a_d = discrete_hankel(tau, 4 * th, ells)
    a_s, Ne, *_ = sphere_profile_transform(M, z, ells, (4.0,))
    print("  M=%.0e z=%.2f theta200=%.2f arcmin:" % (M, z, th * 180 / np.pi * 60), " ".join("%.5f" % r for r in a_d / a_s[4.0]))
