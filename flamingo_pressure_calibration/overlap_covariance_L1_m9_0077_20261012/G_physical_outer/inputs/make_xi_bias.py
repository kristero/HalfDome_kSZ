"""Physical inputs for the outer-profile models of Test G (z = 0, FLAMINGO L1_m9 cosmology).

  xi_lin(r), xi_nl(r)   linear and halofit matter correlation functions from CLASS P(k, z = 0)
  sigma(M), nu(M)       top-hat variance of the linear field, nu = delta_c / sigma
  b(M200c)              Tinker et al. (2010) halo bias for Delta = 200 rho_crit = (200 / Omega_m) rho_mean
  rho_mean              mean matter density, so the cosmic mean electron density is Omega_m in the
                        E_geo normalisation (n_e mu / (rho_crit f_b))

Cosmology: FLAMINGO D3A (h 0.681, Omega_m 0.306, Omega_b 0.0486, sigma8 0.807, n_s 0.967); neutrinos are
treated as massless with Omega_cdm adjusted so that Omega_m = 0.306 (a < 1% effect on xi and b here).
Run with the class_sz environment:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/kn18001/.local/lib/python3.11/site-packages \
  PATH_TO_CLASS_SZ_DATA=/home/kn18001/class_sz_data_directory /home/kn18001/.conda/envs/class_sz/bin/python make_xi_bias.py
Output: xi_bias_z0.npz (radii in Mpc, masses in Msun, no h).
"""
import json
from pathlib import Path

import numpy as np
from scipy import integrate

HERE = Path(__file__).resolve().parent
H, OM, OB, S8, NS = 0.681, 0.306, 0.0486, 0.807, 0.967
DELTA_C = 1.686
RHO_CRIT0 = 2.77536627e11 * H ** 2          # Msun / Mpc^3
RHO_MEAN = OM * RHO_CRIT0


def tinker10_bias(nu, delta_mean):
    y = np.log10(delta_mean)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B, b = 0.183, 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4
    return 1 - A * nu ** a / (nu ** a + DELTA_C ** a) + B * nu ** b + C * nu ** c


def xi_from_pk(k, pk, r, damp=0.02):
    """xi(r) = 1/(2 pi^2) int k^2 P(k) j0(kr) dk on a fine log-k grid, Gaussian damping at k >> 1/damp."""
    lnk = np.log(k)
    out = np.empty(len(r))
    for i, rr in enumerate(r):
        kr = k * rr
        integrand = k ** 3 * pk * np.sinc(kr / np.pi) * np.exp(-(k * damp) ** 2)
        out[i] = integrate.simpson(integrand, x=lnk) / (2 * np.pi ** 2)
    return out


def main():
    from classy_sz import Class
    c = Class()
    for key in ('T_ncdm', 'deg_ncdm', 'm_ncdm', 'omega_ncdm', 'Omega_ncdm'):   # class_sz defaults clash with N_ncdm = 0
        c.pars.pop(key, None)
    c.set({'omega_b': OB * H ** 2, 'omega_cdm': (OM - OB) * H ** 2, 'h': H, 'n_s': NS, 'sigma8': S8,
           'N_ncdm': 0, 'N_ur': 3.046, 'output': 'mPk', 'P_k_max_1/Mpc': 200.0, 'z_max_pk': 0.0,
           'non_linear': 'halofit'})
    c.compute()
    k = np.geomspace(1e-4, 150.0, 6000)                  # 1/Mpc
    pk_lin = np.array([c.pk_lin(kk, 0.0) for kk in k])    # Mpc^3
    pk_nl = np.array([c.pk(kk, 0.0) for kk in k])
    sigma8_check = float(c.sigma8())
    c.struct_cleanup(); c.empty()

    r = np.geomspace(0.05, 60.0, 240)                     # Mpc
    xi_lin = xi_from_pk(k, pk_lin, r)
    xi_nl = xi_from_pk(k, pk_nl, r)
    mass = np.geomspace(1e12, 1e16, 161)                  # M200c, Msun
    R_L = (3 * mass / (4 * np.pi * RHO_MEAN)) ** (1 / 3)
    lnk = np.log(k)
    sigma = np.empty(len(mass))
    for i, R in enumerate(R_L):
        x = k * R
        W = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
        sigma[i] = np.sqrt(integrate.simpson(k ** 3 * pk_lin * W ** 2, x=lnk) / (2 * np.pi ** 2))
    nu = DELTA_C / sigma
    delta_mean = 200.0 / OM
    bias = tinker10_bias(nu, delta_mean)
    np.savez(HERE / 'xi_bias_z0.npz', r_mpc=r, xi_lin=xi_lin, xi_nl=xi_nl, k_per_mpc=k, pk_lin=pk_lin, pk_nl=pk_nl,
             m200c_msun=mass, sigma=sigma, nu=nu, bias_tinker10=bias, rho_mean_msun_mpc3=RHO_MEAN,
             rho_crit_msun_mpc3=RHO_CRIT0, omega_m=OM,
             meta=json.dumps({'h': H, 'Omega_m': OM, 'Omega_b': OB, 'sigma8_input': S8, 'sigma8_class': sigma8_check,
                              'n_s': NS, 'neutrinos': 'massless (N_ur 3.046); Omega_cdm = Omega_m - Omega_b',
                              'bias': 'Tinker+10, Delta_mean = 200/Omega_m', 'xi_damping_mpc': 0.02}))
    for M in (1e13, 1e14, 1e15):
        i = np.argmin(abs(mass - M))
        print(f'M200c {M:.0e}: nu {nu[i]:.2f}, b {bias[i]:.2f}')
    for rr in (1.0, 2.0, 5.0, 10.0):
        j = np.argmin(abs(r - rr))
        print(f'r {r[j]:.2f} Mpc: xi_lin {xi_lin[j]:.3f}, xi_nl {xi_nl[j]:.3f}')
    print('sigma8 check', sigma8_check)


if __name__ == '__main__':
    main()
