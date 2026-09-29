#!/usr/bin/env python3
"""Validation of the Python profiles in szhm.py against the repo's Julia/XGPaint sphere tables and self-consistency
checks of the transforms. Writes validate_profiles.json in the output directory."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import szhm as S  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget")
TABLE = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5")
cosmo = S.PAINT_COSMO
res = {}

# 1. chord-mean DM inside the R200c sphere against the Julia table (XGPaint cosmology Om=0.31, Ob=0.049, h=0.68)
with h5py.File(TABLE, "r") as f:
    lm = f["log10_m200c_msun"][:]
    zz = f["redshift"][:]
    xx = f["x_r200c"][:]
    tabs = {"lee22_xgpnative": f["lee22_noconc_xgpnative/chord_mean_dm_pc_cm3"][:],
            "b16": f["b16/chord_mean_dm_pc_cm3"][:],
            "r200_lee": f["lee22_noconc_xgpnative/r200c_mpc"][:]}
print("table shape", tabs["b16"].shape)
t, w = S._gl(400)
rng = np.random.default_rng(1)
worst = {}
for model in ("lee22_xgpnative", "b16"):
    tab = tabs[model]
    errs = []
    for trial in range(60):
        im = rng.integers(0, len(lm)); iz = rng.integers(0, len(zz))
        if trial < 6:
            im = [200, 400, 460, 461, 560, 699][trial]
        M = np.array([10 ** lm[im]]); z = zz[iz]
        amp, shape = S.density_shape(model, cosmo, M, z)
        r200 = S.r200c_mpc(cosmo, M, z)[0] * S.MPC_CM
        for ix in (0, 50, 120, 180, 199):
            x = xx[ix]
            L = np.sqrt(1 - x ** 2)
            l = L * t
            r = np.sqrt(x ** 2 + l ** 2)[None, :]
            col = 2 * (shape(r) * L * w).sum() * amp[0] * r200 / S.PC_CM / (1 + z)   # DM through the chord
            g = col / (2 * L)
            # h5py order: (logM, z, x) per the table reader in halo_model_dm_pdf.py
            ref = tab[im, iz, ix]
            errs.append(abs(g / ref - 1))
    worst[model] = float(np.max(errs))
res["chord_mean_vs_julia_table_max_rel_err"] = worst
print("chord mean vs Julia table:", worst)

# 2. sphere 3-D FT equals the Hankel transform of the chord-projected sphere profile
M = np.array([1e13, 1e14, 1e15]); z = 0.5
amp, shape = S.density_shape("lee22_literal", cosmo, M, z)
q = np.array([[0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]] * 3)
u3 = S.sphere_ft(shape, 3.0, q)
# direct: projected chord profile Sigma(x) = 2 int_0^sqrt(9-x^2) f dl, then 2 pi int x Sigma J0
from scipy import special
s, ws = S._gl(4000)
x = 3.0 * s ** 2; jac = 6.0 * s * ws
tl, wl = S._gl(400)
sig = np.empty((3, len(x)))
for i in range(3):
    L = np.sqrt(np.maximum(9 - x ** 2, 0))
    r = np.sqrt(x[:, None] ** 2 + (L[:, None] * tl[None, :]) ** 2)
    fr = shape(np.broadcast_to(r[None], (3,) + r.shape))[i]
    sig[i] = 2 * (fr * wl[None, :]).sum(1) * L
u2 = np.array([[(2 * np.pi * x * jac * sig[i] * special.j0(qq * x)).sum() for qq in q[i]] for i in range(3)])
res["sphere_ft_vs_projected_hankel_max_rel_err"] = float(np.max(np.abs(u3 / u2 - 1)[:, :5]))
print("sphere FT vs projected Hankel (q<=10):", np.abs(u3 / u2 - 1).max(1), "all q:", np.abs(u3 / u2 - 1))

# 3. projected GNFW table against direct LOS quadrature, and the Hankel aperture transform at q=0
tab = S.ProjectedGNFWTable()
errs = []
for (u, beta) in [(1e-6, 4.0), (1e-3, 4.35), (0.1, 5.0), (1.0, 9.4), (3.0, 20.0), (8.0, 4.0), (0.05, 36.0), (15.0, 3.0)]:
    tt, ww = S._gl(20000)
    lt = np.log(1e-14) + tt * (np.log(1e6) - np.log(1e-14)); wt = ww * (np.log(1e6) - np.log(1e-14))
    T = np.exp(lt); r = np.sqrt(u ** 2 + T ** 2)
    direct = 2 * (r ** -0.3 * (1 + r) ** -beta * T * wt).sum()
    errs.append(abs(tab(np.array([u]), beta)[0] / direct - 1))
res["projected_table_max_rel_err"] = float(np.max(errs))
print("projected table errs", errs)

# 4. aperture Hankel at q=0 equals 2 pi int_0^4 x Sigma dx; compare full-LOS cylinder with sphere for B12
M = np.array([1e13, 1e14, 1e15]); z = 0.3
P0, xc, beta = S.pressure_params("b12", cosmo, M, z)
Y0 = S.aperture_hankel(tab, xc, beta, 4.0, np.zeros((3, 1)))[:, 0]
sh = S.RowShape("lee22", xc=xc, beta=beta)
V4 = S.enclosed(sh, 4.0)
Vinf = S.enclosed(sh, 1e4, n=20000)
res["b12_cylinder4_over_sphere4_volume_integral"] = (Y0 / V4).tolist()
res["b12_cylinder4_over_total"] = (Y0 / Vinf).tolist()
print("B12: cylinder(4R200) / sphere(4R200) pressure volume integral", Y0 / V4, " cylinder/total", Y0 / Vinf)

# 5. XGPaint amplitude factors and M_gas(<R200c) of the literal Lee22 fit (item 7 of the task context)
res["xgpnative_factor"] = S.XGPNATIVE_FACTOR
lmh = np.array([13.2, 13.61, 14.0])
M = 10 ** lmh / cosmo.h
fg = {}
for z in (0.0, 0.5, 1.0, 2.0):
    amp, shape = S.density_shape("lee22_literal", cosmo, M, z)
    n200 = 200 * S.rho_crit_cgs(cosmo, z) * (cosmo.omega_b / cosmo.omega_m) / (S.X_H * S.M_P_G)
    mean_nbar = (amp / n200) * S.enclosed(shape, 1.0) / (4 * np.pi / 3)
    fg[str(z)] = dict(volume_mean_ne_over_n200=mean_nbar.tolist(),
                      Mgas_over_fb_M200_mu_e=(mean_nbar * (2 / (1 + S.X_H)) / S.X_H).tolist())
res["lee22_literal_enclosed"] = dict(log10_M_h=lmh.tolist(), by_z=fg)
print(json.dumps(fg, indent=1))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "validate_profiles.json").write_text(json.dumps(res, indent=1))
