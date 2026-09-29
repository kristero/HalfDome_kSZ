"""verify_1 check 1: (a) validate own Lee22 chord DM against the repo Julia table (HalfDome cosmology),
(b) literal Lee22 enclosed gas fraction vs M and z (TNG cosmology), (c) integral of the paper's own
Figure 5 simulation curves (z=0, lowest-concentration quartile) and no-c fit curves.
"""
import numpy as np, h5py, importlib, sys, json
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/verify_1')
import lee22_verify_lib as L

ROOT = '/home/cbllover/HalfDome/frb_map_generation/'
OUT = ROOT + 'outputs/lee22_normalization_forensics_20260928/verify_1/'
lines = []
P = lambda s: (print(s), lines.append(s))

# ---------------- (a) validation against Julia table in HalfDome cosmology ----------------
L.H, L.OM, L.OB = 0.68, 0.31, 0.049
L.OL = 1 - L.OM
L.FB = L.OB / L.OM
f = h5py.File(ROOT + 'outputs/analytic_halo_pdf_20260925/tables/halfdome_sphere_profiles.h5', 'r')
lm = f['log10_m200c_msun'][:]; zz = f['redshift'][:]; xx = f['x_r200c'][:]
g = f['lee22_noconc_xgpnative/chord_mean_dm_pc_cm3']
R_tab = f['lee22_noconc_xgpnative/r200c_mpc']
maxdev = 0.0
P('# (a) own literal chord-mean DM / (Julia xgpnative table / 0.6016316602183194)  [HalfDome cosmology]')
for im in [100, 350, 450, 500, 550, 650]:
    for iz in [0, 30, 60, 99]:
        M = 10 ** lm[im]; z = zz[iz]
        n0, xc, b = L.lee22_params(M * L.H, z)
        sel = np.arange(0, 200, 13)
        x = xx[sel]
        dm = L.chord_dm(M, z, x, n0, xc, b, nl=600)
        gmine = dm / (2 * np.sqrt(1 - x ** 2))
        gtab = g[im, iz, sel] / 0.6016316602183194
        r = gmine / gtab
        rr = L.r200c_mpc(M, z) / R_tab[im, iz]
        maxdev = max(maxdev, np.max(np.abs(r - 1)))
        P(f'logM={lm[im]:.3f} z={z:.3f}  ratio min/max = {r.min():.6f}/{r.max():.6f}  R200 ratio {rr:.6f}')
P(f'max |ratio-1| = {maxdev:.2e}')

# ---------------- (b) literal enclosed gas fraction, TNG cosmology ----------------
importlib.reload(L)
P('\n# (b) literal Lee22 (Table 8, eq. 9 as printed): <n_e/n200>(<R200c) and M_gas/(f_b M200) = 1.4952 <nbar>')
P('log10(M h)   z    <nbar>   Mgas/(fb M200)')
res = {}
for lmh in [12.83, 13.0, 13.3, 13.61, 13.9, 14.0, 14.5]:
    for z in [0.0, 0.5, 1.0, 1.5, 2.0]:
        nb = L.mean_nbar_inside(10 ** lmh, z)
        res[(lmh, z)] = nb
        P(f'{lmh:6.2f}  {z:4.1f}  {nb:7.4f}  {nb * L.MGAS_PER_NBAR:6.3f}')
P('effective exponent k in Mgas ~ (1+z)^k between z=0 and z=2, and z=0..1:')
for lmh in [12.83, 13.0, 13.3, 13.61, 13.9, 14.0, 14.5]:
    k2 = np.log(res[(lmh, 2.0)] / res[(lmh, 0.0)]) / np.log(3)
    k1 = np.log(res[(lmh, 1.0)] / res[(lmh, 0.0)]) / np.log(2)
    P(f'  log10(M h)={lmh:.2f}: k(0-2) = {k2:.3f}, k(0-1) = {k1:.3f}')

# self-similar physical check: what the printed n200 would imply if gas = f_b * rho_matter at 200 rho_c
P(f'<nbar> for M_gas = f_b M200 exactly would be {1 / L.MGAS_PER_NBAR:.4f}')

# ---------------- (c) paper Figure 5 curves (x^3 n_e/n200 at z=0) ----------------
import csv
rows = list(csv.DictReader(open(ROOT + 'outputs/lee22_fig5_audit_20260922/figure5_vector_curves.csv')))
P('\n# (c) Figure 5 vector curves: <nbar>(<R200c) = 3 int x^3 nbar dln x (x<1); below x_min extrapolated')
P('    as x^3 nbar ~ x^{2.7} (gamma=-0.3 inner slope) ; M_gas/(fb M200) = 1.4952 <nbar>')
for panel, lab in [('0', '13.2-13.4'), ('1', '13.8-14.0')]:
    for curve in ['simulation', 'without_c_bpl', 'with_c_bpl', 'without_c_no_bpl', 'battaglia16']:
        d = [(float(r['x']), float(r['x3_ne_over_n200'])) for r in rows if r['panel'] == panel and r['curve'] == curve]
        d = np.array(sorted(d))
        # de-duplicate
        _, iu = np.unique(d[:, 0], return_index=True); d = d[iu]
        x, y = d[:, 0], d[:, 1]
        m = x <= 1.0
        lx = np.log(x[m]); yy = y[m]
        # add endpoint at x=1 by log-linear interpolation
        if x[m].max() < 1.0:
            y1 = np.exp(np.interp(0.0, np.log(x), np.log(y)))
            lx = np.append(lx, 0.0); yy = np.append(yy, y1)
        main = np.trapz(yy, lx)
        inner = yy[0] / 2.7  # int_0^{x0} y0 (x/x0)^2.7 dlnx
        nb = 3 * (main + inner)
        P(f'panel {lab} {curve:18s}: x range {x.min():.4f}-{x.max():.3f}; <nbar>={nb:.3f} (inner extrap {3*inner:.4f}); Mgas/(fb M)={nb * L.MGAS_PER_NBAR:.2f}')
open(OUT + 'check1_gas_fractions_fig5.txt', 'w').write('\n'.join(lines) + '\n')
