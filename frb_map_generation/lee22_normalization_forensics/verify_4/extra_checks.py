"""verify_4 extra checks: (1) per-shell C(z) with the TNG-Dark halo population instead of Tinker08;
(2) Macquart-relation check of Konietzka total DM (redshift fidelity of the TNG side);
(3) test of the untested PDF-shape inference with a scatter-augmented C(z)-corrected model;
(4) mass-window-unit sensitivity (Msun vs Msun/h) of the Poisson hit fraction."""
import json, math, numpy as np
from common import TNG, XGP_OVER_LITERAL, Tinker08, MSUN_KG, MP_KG, MPC_M, X_H
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_4/'
KDIR = '/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/'
d = np.load(OUT + 'tngdark_crossings.npz'); nr = int(d['n_rays']); ray = d['ray']; zc = d['z']; lit = d['dm_literal']
zsg = np.load(KDIR + 'redshifts_DMhalo.npy'); K = np.load(KDIR + 'DMhalo_r200_mass13to14.npy')
res = {}
# (1) C(z) per shell, TNG-Dark population
Kall = K.mean(axis=1)[:10]; dK = np.diff(np.concatenate([[0], Kall]))
dL = np.array([lit[(zc > zs - 0.1 + 1e-12) & (zc <= zs + 1e-12)].sum() / nr for zs in zsg[:10]])
C = dK / dL; zm = zsg[:10] - 0.05
p = np.polyfit(np.log(1 + zm), np.log(C), 1)
zx = math.exp((math.log(XGP_OVER_LITERAL) - p[1]) / p[0]) - 1
wC = dK.sum() / dL.sum()
print('(1) per-shell C = dK/dlit with TNG-Dark halo population:')
print('   ', ' '.join(f'{a:.2f}:{b:.3f}' for a, b in zip(zm, C)))
print(f'    fit C = {math.exp(p[1]):.3f}(1+z)^{p[0]:.2f}; crosses 0.6016 at z = {zx:.3f}; z_s=1 weighted <C> = {wC:.3f}')
res['C_tngdark'] = dict(zmid=zm.tolist(), C=C.tolist(), A=math.exp(p[1]), k=p[0], zcross=zx, zs1_weighted=wC)

# (2) Macquart check: <DM_all> vs  c/H0 * n_e0 * int (1+z)/E dz with n_e0 = rho_b0 (1 - Y/2)/m_p  (full ionisation, Y=0.24)
c = TNG
Kt = np.load(KDIR + 'DMall_IGM_halo.npy').mean(axis=1)
ne0 = c.ob * c.rho_c0 * MSUN_KG / MPC_M ** 3 / MP_KG * (X_H + (1 - X_H) / 2) * 1e-6  # cm^-3 (He fully ionised)
zg = np.linspace(0, 5.5, 55001); integ = (1 + zg) / c.E(zg)
cum = np.concatenate([[0], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(zg))])
mac = ne0 * (299792.458 / (100 * c.h)) * 1e6 * np.interp(zsg, zg, cum)
fd = Kt / mac
pf = np.polyfit(np.log(1 + zsg), np.log(fd), 1)
print(f'(2) Konietzka <DM_total>/Macquart(f_d=1): z_s=0.1 {fd[0]:.3f}, 0.5 {fd[4]:.3f}, 1.0 {fd[9]:.3f}, 2.0 {fd[19]:.3f}, 5.5 {fd[-1]:.3f};'
      f' log-slope vs (1+z): {pf[0]:+.3f}')
res['macquart'] = dict(zs=zsg.tolist(), ratio=fd.tolist(), slope=pf[0])

# (3) PDF-shape inference test
A = json.load(open(OUT + 'analytic_mean_dm.json'))['fit_z<1.0']
corr = lit * A['A'] * (1 + zc) ** A['k']
xgp = XGP_OVER_LITERAL * lit
edges = np.concatenate([[0], np.logspace(0, np.log10(3000), 60)])
cen = np.sqrt(edges[1:] * np.maximum(edges[:-1], 1e-3))
rng = np.random.default_rng(7)
g = rng.standard_normal(len(lit))
def stats(per_cross, zs, kk):
    m = zc <= zs + 1e-12
    dm = np.bincount(ray[m], weights=per_cross[m], minlength=nr)
    h = dm > 0; hk = kk > 0
    pk = np.histogram(kk[hk], edges)[0]; px = np.histogram(dm[h], edges)[0]
    good = (cen > 3) & (cen < 1000) & (pk >= 50)
    shape = np.mean(np.abs((px[good] / h.sum()) / (pk[good] / hk.sum()) - 1))
    q10, q90 = np.quantile(dm[h], [0.1, 0.9])
    return shape, q90 / q10, (dm[h] < 10).mean(), dm[h].mean()
print('(3) PDF shape [mean|p/pK-1| over 3-1000], width q90/q10, frac(DM<10 | hit), mean per hit')
tab = {}
for zs_i, zs in ((0, 0.1), (2, 0.3), (4, 0.5), (9, 1.0)):
    kk = K[zs_i]; hk = kk > 0
    q10, q90 = np.quantile(kk[hk], [0.1, 0.9])
    line = f'  z_s={zs}: Konietzka width {q90/q10:5.1f} f<10 {(kk[hk]<10).mean():.3f} | '
    for name, base in (('xgp', xgp), ('corr', corr)):
        for s in (0.0, 0.4, 0.6):
            pc = base * np.exp(s * g - 0.5 * s * s)
            sh, w, f10, mh = stats(pc, zs, kk)
            tab[f'{zs}_{name}_s{s}'] = dict(shape=sh, width=w, f10=f10, mean_hit=mh)
            line += f'{name} s={s}: {sh:.3f} w={w:4.1f} f<10={f10:.3f} m={mh:5.1f}; '
    print(line)
res['shape_test'] = tab

# (4) Poisson hit fraction for window in Msun vs Msun/h (Tinker08)
t08 = Tinker08(c)
def hit(zs, mlo, mhi):
    dz = 0.005; zmid = np.arange(dz / 2, zs, dz); tot = 0
    lg = np.linspace(math.log10(mlo), math.log10(mhi), 41); lm = 0.5 * (lg[1:] + lg[:-1]); dl = (lg[1] - lg[0]) * math.log(10)
    for z in zmid:
        mm = 10 ** lm; th = c.r200c(mm, z) / (c.chi(z) / (1 + z))
        tot += c.chi(z) ** 2 * c.c_over_H(z) * np.sum(t08.dndlnm(mm, z) * math.pi * th ** 2 * dl) * dz
    return 1 - math.exp(-tot)
for zs_i, zs in ((0, 0.1), (9, 1.0)):
    print(f'(4) z_s={zs}: Konietzka hit {np.mean(K[zs_i]>0):.4f}; Poisson Tinker08 [1e13,1e14) Msun {hit(zs,1e13,1e14):.4f}; '
          f'[1e13,1e14)/h {hit(zs,1e13/c.h,1e14/c.h):.4f}')
json.dump(res, open(OUT + 'extra_checks.json', 'w'), indent=1, default=float)
