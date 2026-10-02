# Independent one-halo kSZ calculator (no XGPaint): Battaglia16 AGN rho_gas, 3D truncation at 4 R200c,
# tau_ell = sigma_T / d_A^2 * int_0^{4R} 4 pi r^2 n_e(r) j0(ell r / d_A) dr  (exact flat-sky projection of a truncated sphere),
# C_ell = (1/4pi) sum_cells sum (v_los/c)^2 |tau_ell|^2 using the painter's (log10 M, z) histogram.
# Purpose: quantify the cosmology/h convention (WebSky h=0.68 Om=0.31 Ob=0.049 vs TNG h=0.6774 Om=0.3089 Ob=0.0486).
import h5py, numpy as np
from scipy.integrate import cumulative_trapezoid
G = 6.67430e-11; MPC = 3.0856775814913673e22; MSUN = 1.98847e30; SIGT = 6.6524587321e-29
MP = 1.67262192369e-27; ME = 9.1093837015e-31; KMS = 1e3
xH = 0.76; nH_ne = 2 * xH / (xH + 1); nHe_ne = (1 - xH) / (2 * (1 + xH)); M_PER_E = ME + nH_ne * MP + nHe_ne * 4 * MP
hist = h5py.File('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5', 'r')
cnt = hist['count'][:]; v2 = hist['sum_vlos_over_c_squared'][:]   # (z, logM) in python order
le = hist['logm_edges'][:]; ze = hist['z_edges'][:]
iz, im = np.nonzero(cnt)
lm_cat = 0.5 * (le[im] + le[im + 1])          # log10(M_h / 0.68)
zc = 0.5 * (ze[iz] + ze[iz + 1]); w = v2[iz, im]
ells = np.array([300., 1000., 2000., 3000., 5000.])

def cosmo(h, Om, radiation=True, Neff=3.046, Tcmb=2.7255):
    Or = 0.0
    if radiation:
        OG = 4.48131e-7 * Tcmb**4 / h**2; Or = OG * (1 + Neff * 7 / 8 * (4 / 11)**(4 / 3))
    zg = np.linspace(0, 4.2, 8401)
    E = np.sqrt(Om * (1 + zg)**3 + Or * (1 + zg)**4 + 1 - Om - Or)
    chi = 2997.92458 / h * np.concatenate([[0], cumulative_trapezoid(1 / E, zg)])   # Mpc
    return dict(h=h, Om=Om, zg=zg, E=E, chi=chi)

def cl_onehalo(c, Ob, mass_h_div, fb_override=None, dA_cosmo=None, rhoc_cosmo=None, nr=600):
    dA_c = dA_cosmo or c; rc_c = rhoc_cosmo or c
    fb = fb_override if fb_override is not None else Ob / c['Om']
    M = 10**lm_cat * 0.68 / mass_h_div              # Msun
    Ez = np.interp(zc, rc_c['zg'], rc_c['E'])
    Hz = 100 * rc_c['h'] * Ez * KMS / MPC          # 1/s
    rhoc = 3 * Hz**2 / (8 * np.pi * G)             # kg/m^3
    R200 = (3 * M * MSUN / (4 * np.pi * 200 * rhoc))**(1 / 3)   # m
    dA = np.interp(zc, dA_c['zg'], dA_c['chi']) / (1 + zc) * MPC   # m
    m = M / 1e14; z1 = 1 + zc
    P0 = 4.0e3 * m**0.29 * z1**-0.66; xc = 0.5
    al = 0.88 * m**-0.03 * z1**0.19; be = 3.83 * m**0.04 * z1**-0.025; ga = -0.2
    x = np.geomspace(1e-5, 4.0, nr)[None, :]       # r / R200
    xb = x / xc
    gnfw = xb**ga * (1 + xb**al[:, None])**(-(be[:, None] + ga) / al[:, None])
    ne = 0.9 * fb * rhoc[:, None] * P0[:, None] * gnfw / M_PER_E   # 1/m^3
    r = x * R200[:, None]
    out = np.zeros(len(ells))
    for i, L in enumerate(ells):
        k = L / dA
        kr = k[:, None] * r
        j0 = np.sinc(kr / np.pi)
        integ = 4 * np.pi * r**2 * ne * j0
        tl = SIGT / dA**2 * np.trapz(integ, r, axis=1)
        out[i] = np.sum(w * tl**2) / (4 * np.pi)
    return out

T2 = 2.7255e6**2
D = lambda cl: ells * (ells + 1) * cl / (2 * np.pi) * T2
WS = cosmo(0.68, 0.31); TNG = cosmo(0.6774, 0.3089); TNGnr = cosmo(0.6774, 0.3089, radiation=False)
WS_h_TNG_Om = cosmo(0.68, 0.3089); TNG_h_WS_Om = cosmo(0.6774, 0.31)
base = cl_onehalo(WS, 0.049, 0.68)
rows = [
 ('A painter: WebSky cosmo (h .68 Om .31 Ob .049), M/0.68', base),
 ('B mass /0.6774 only (profiles in WebSky cosmo)', cl_onehalo(WS, 0.049, 0.6774)),
 ('C1 f_b only -> 0.0486/0.3089', cl_onehalo(WS, 0.049, 0.68, fb_override=0.0486 / 0.3089)),
 ('C2 d_A only -> TNG (h .6774, Om .3089)', cl_onehalo(WS, 0.049, 0.68, dA_cosmo=TNG)),
 ('C3 rho_crit(z) only -> TNG', cl_onehalo(WS, 0.049, 0.68, rhoc_cosmo=TNG)),
 ('C4 h only (0.6774) everywhere, Om .31 Ob/Om .049/.31', cl_onehalo(TNG_h_WS_Om, 0.049, 0.6774)),
 ('C5 Om only (0.3089, Ob .0486), h .68, M/0.68', cl_onehalo(WS_h_TNG_Om, 0.0486, 0.68)),
 ('D full TNG (h .6774 Om .3089 Ob .0486), M/0.6774, radiation', cl_onehalo(TNG, 0.0486, 0.6774)),
 ('E full TNG, no radiation in d_A/E(z)', cl_onehalo(TNGnr, 0.0486, 0.6774)),
]
print('independent one-halo (B16 AGN, 3D sphere 4 R200c, sinc transform), D_ell [uK^2] at ell =', ells.astype(int))
for lab, cl in rows:
    print('%-62s' % lab, ' '.join('%.4f' % v for v in D(cl)), '  ratio', ' '.join('%.4f' % v for v in cl / base))
# radial convergence check
cl2 = cl_onehalo(WS, 0.049, 0.68, nr=1500)
print('convergence nr=1500 / nr=600:', ' '.join('%.5f' % v for v in cl2 / base))
