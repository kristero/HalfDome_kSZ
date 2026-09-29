"""Validate the independent Lee22 chord DM (common.py) against the HalfDome sphere table."""
import h5py, numpy as np, math
from common import HALFDOME, lee22_chord_dm_literal, XGP_OVER_LITERAL
f = h5py.File('/home/cbllover/HalfDome/frb_map_generation/outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5', 'r')
lm = f['log10_m200c_msun'][:]; zz = f['redshift'][:]; xs = f['x_r200c'][:]
g = f['lee22_noconc_xgpnative/chord_mean_dm_pc_cm3']
r200 = f['lee22_noconc_xgpnative/r200c_mpc']
print('XGP_OVER_LITERAL =', repr(XGP_OVER_LITERAL))
rng = np.random.default_rng(1)
worst = 0; worst_r = 0
for trial in range(60):
    j = rng.integers(300, 560)  # log10 M 12.0 .. 14.6
    i = rng.integers(0, 100); k = rng.integers(0, 200)
    m = 10 ** lm[j]; z = zz[i]; x = xs[k]
    ref = g[j, i, k] * 2 * math.sqrt(1 - x * x)
    mine = XGP_OVER_LITERAL * lee22_chord_dm_literal(HALFDOME, x, m, z)
    worst = max(worst, abs(mine / ref - 1))
    worst_r = max(worst_r, abs(float(HALFDOME.r200c(m, z)) / r200[j, i] - 1))
print('max |mine/table - 1| over 60 random (M,z,x) nodes =', worst)
print('max |R200c mine/table - 1| =', worst_r)
