"""Validate the independent verify_3 Lee22 density formula against the production HalfDome
sphere-profile table (lee22_noconc_xgpnative, divided by 0.6016316602183194 = literal).
Table cosmology: Omega_c=0.261, Omega_b=0.049, h=0.68 (flat). Masses in the table are Msun;
Lee22 masses are h^-1 Msun with h=0.68 (model.cosmo.h in lee2022_frb_dm_profile.jl)."""
import numpy as np, h5py, json, os
from scipy import integrate
import lee22_c4_lib as L

TAB = "/home/cbllover/HalfDome/frb_map_generation/outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5"
OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_3"
XGP_OVER_LIT = 0.6016316602183194
h, om, ob = 0.68, 0.31, 0.049
fb = ob / om
PC_CM = 3.0856775814913673e18


def rho_c(z):
    return L.RHO_CRIT0_H2 * h ** 2 * (om * (1 + z) ** 3 + 1 - om)


def g_literal(logm, z, x):
    m = 10 ** logm
    r200 = (3 * m / (4 * np.pi * 200 * rho_c(z))) ** (1 / 3)          # Mpc physical
    n200 = 200 * rho_c(z) * L.MSUN_G / L.MPC_CM ** 3 * fb / (L.X_H * L.M_P_G)
    mh = m * h
    out = []
    for xi in x:
        lmax = np.sqrt(1 - xi ** 2)
        # substitution l = lmax * sin(t) is not needed; use log-spaced quad
        f = lambda l: L.nhat(np.sqrt(xi ** 2 + l ** 2), mh, z)
        val, _ = integrate.quad(f, 0, lmax, limit=400, epsabs=0, epsrel=1e-10)
        dm = 2 * r200 * L.MPC_CM / PC_CM / (1 + z) * n200 * val
        out.append(dm / (2 * lmax))
    return np.array(out)


with h5py.File(TAB) as f:
    lm = f["log10_m200c_msun"][:]
    zz = f["redshift"][:]
    xx = f["x_r200c"][:]
    g = f["lee22_noconc_xgpnative/chord_mean_dm_pc_cm3"]   # h5py shape (700,100,200) = (logM, z, x)
    res = []
    for im, iz in [(300, 10), (400, 50), (450, 99), (500, 30), (380, 0), (550, 70)]:
        tab = g[im, iz, :] / XGP_OVER_LIT
        sel = np.arange(5, 190, 20)
        mine = g_literal(lm[im], zz[iz], xx[sel])
        rel = mine / tab[sel] - 1
        res.append(dict(log10m_msun=float(lm[im]), z=float(zz[iz]),
                        max_abs_rel=float(np.max(np.abs(rel))), median_rel=float(np.median(rel))))
        print(res[-1])
json.dump(res, open(os.path.join(OUT, "validation_vs_halfdome_table.json"), "w"), indent=1)
