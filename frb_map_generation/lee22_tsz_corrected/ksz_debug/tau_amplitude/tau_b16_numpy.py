"""Independent numpy/scipy implementation of the Battaglia (2016) AGN Delta=200c projected optical depth tau(theta),
full line of sight and gas inside the 4 R200c sphere, compared with XGPaint (xgpaint_tau_dump.csv from dump_xgpaint_tau.jl).

Conventions (independent of XGPaint):
  rho_gas(r) = f_b rho_crit(z) rho0 (x/xc)^gamma [1 + (x/xc)^alpha]^(-(beta+gamma)/alpha),  x = r / R200c (physical)
  B16 Table 2 AGN feedback: rho0 = 4000 m^0.29 (1+z)^-0.66, alpha = 0.88 m^-0.03 (1+z)^0.19, beta = 3.83 m^0.04 (1+z)^-0.025,
  xc = 0.5, gamma = -0.2, m = M200c / 1e14 Msun.
  n_e = f_free rho_gas / (mu_e m_p), mu_e m_p = m_e + 2 m_p/(1+X_H)   (fully ionised H+He, X_H = 0.76); f_free = 0.9 as XGPaint.
  tau(R) = sigma_T int n_e dl (physical), R = d_A theta.
"""
import numpy as np
from scipy.integrate import quad
import csv, sys

# CODATA 2018 / IAU constants (independent of XGPaint's Unitful)
G = 6.67430e-11
MPC = 3.0856775814913673e22
MSUN = 1.98847e30
C_KMS = 299792.458
SIGMA_T = 6.6524587321e-29
M_P = 1.67262192369e-27
M_E = 9.1093837015e-31
X_H = 0.76

h, Om, Ob = 0.68, 0.31, 0.049
Tcmb, Neff = 2.7255, 3.046
og = 2.4728e-5 / h**2 * (Tcmb / 2.7255) ** 4         # Omega_gamma (Stefan-Boltzmann)
OR = og * (1 + Neff * 7 / 8 * (4 / 11) ** (4 / 3))
OL = 1 - Om - OR
fb = Ob / Om


def E(z):
    return np.sqrt(Om * (1 + z) ** 3 + OR * (1 + z) ** 4 + OL)


def rho_crit(z):  # kg/m^3, physical
    H = 100 * h * E(z) * 1e3 / MPC  # s^-1
    return 3 * H ** 2 / (8 * np.pi * G)


def d_A(z):  # Mpc
    chi = quad(lambda zz: 1 / E(zz), 0, z, epsabs=0, epsrel=1e-12)[0] * C_KMS / (100 * h)
    return chi / (1 + z)


def r200c(M, z):  # Mpc physical
    return (3 * M * MSUN / (4 * np.pi * 200 * rho_crit(z))) ** (1 / 3) / MPC


def b16_params(M, z, model="AGN"):
    m = M / 1e14
    zp = 1 + z
    if model == "AGN":
        rho0 = 4000 * m ** 0.29 * zp ** -0.66
        al = 0.88 * m ** -0.03 * zp ** 0.19
        be = 3.83 * m ** 0.04 * zp ** -0.025
    else:  # shock heating
        rho0 = 1.9e4 * m ** 0.09 * zp ** -0.95
        al = 0.70 * m ** -0.017 * zp ** 0.27
        be = 4.43 * m ** 0.005 * zp ** 0.037
    return rho0, 0.5, al, be, -0.2


def shape(x, xc, al, be, ga, sign=+1):
    """sign=+1: exponent -(beta+gamma)/alpha (XGPaint, asymptotic slope -beta); sign=-1: -(beta-gamma)/alpha as printed in
    B16 eq. (A1); sign=0: Stein+20 eq. (3.4) literal, exponent -beta."""
    u = x / xc
    if sign == 0:
        return u ** ga * (1 + u ** al) ** (-be)
    return u ** ga * (1 + u ** al) ** (-(be + sign * ga) / al)


def los_integral(x, pars, L=np.inf, sign=+1):
    """int_{-L}^{L} shape(sqrt(x^2+y^2)) dy in R200c units."""
    _, xc, al, be, ga = pars
    f = lambda y: shape(np.sqrt(x * x + y * y), xc, al, be, ga, sign)
    if np.isinf(L):
        # split to help quad with the core and the power-law tail
        s = quad(f, 0, max(x, 1.0), epsabs=0, epsrel=1e-11, limit=400)[0]
        s += quad(f, max(x, 1.0), 1e5, epsabs=0, epsrel=1e-11, limit=400)[0]
        return 2 * s
    if x >= L:
        return 0.0
    Lc = np.sqrt(L * L - x * x)
    return 2 * quad(f, 0, Lc, epsabs=0, epsrel=1e-11, limit=400)[0]


def mass_per_electron():
    return M_E + 2 * M_P / (1 + X_H)


def tau_profile(x, M, z, X=None, f_free=0.9, sign=+1, model="AGN"):
    pars = b16_params(M, z, model)
    rc = rho_crit(z)
    R = r200c(M, z) * MPC
    I = los_integral(x, pars, np.inf if X is None else X, sign)
    sigma_gas = pars[0] * I * fb * rc * R   # kg/m^2
    return SIGMA_T * f_free * sigma_gas / mass_per_electron()


if __name__ == "__main__":
    dump = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/tau_amplitude/xgpaint_tau_dump.csv"
    rows = list(csv.DictReader(open(dump)))
    worst = {}
    print(f"Omega_R = {OR:.6e}, Omega_L = {OL:.10f}, f_b = {fb:.6f}, mu_e m_p / m_p = {mass_per_electron()/M_P:.6f}")
    print(f"{'M':>8} {'z':>4} {'x':>5} | {'np full':>11} {'xgp/np':>9} | {'np 4R':>11} {'xgpdir/np':>9} {'cache/np':>9} | 4R/full")
    for r in rows:
        M, z, x = float(r["M_msun"]), float(r["z"]), float(r["x"])
        pars = b16_params(M, z)
        chk = {
            "rho_crit": float(r["rho_crit_kgm3"]) / rho_crit(z) - 1,
            "r200c": float(r["r200c_mpc"]) / r200c(M, z) - 1,
            "dA": float(r["dA_mpc"]) / d_A(z) - 1,
            "theta200": float(r["theta200_rad"]) / np.arctan(r200c(M, z) / d_A(z)) - 1,
            "P0": float(r["P0"]) / pars[0] - 1, "alpha": float(r["alpha"]) / pars[2] - 1, "beta": float(r["beta"]) / pars[3] - 1,
        }
        tf = tau_profile(x, M, z)
        ts = tau_profile(x, M, z, X=4.0)
        chk["tau_full"] = float(r["tau_full"]) / tf - 1
        chk["tau_sphere_direct"] = float(r["tau_sphere_direct"]) / ts - 1
        chk["tau_cache"] = float(r["tau_cache_x_chord"]) / ts - 1
        for k, v in chk.items():
            worst[k] = max(worst.get(k, 0), abs(v))
        if x in (0.2, 1.0, 3.5):
            print(f"{M:8.1e} {z:4.1f} {x:5.2f} | {tf:11.4e} {float(r['tau_full'])/tf:9.6f} | {ts:11.4e} "
                  f"{float(r['tau_sphere_direct'])/ts:9.6f} {float(r['tau_cache_x_chord'])/ts:9.6f} | {ts/tf:.4f}")
    print("max |XGPaint/numpy - 1| over all rows:")
    for k, v in worst.items():
        print(f"  {k:18s} {v:.2e}")
