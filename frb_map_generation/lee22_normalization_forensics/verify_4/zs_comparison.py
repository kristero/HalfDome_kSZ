"""Compare Konietzka TNG300 (hydro, gas cells) with Lee22 xgpnative painted on the TNG300-1-Dark
FoF lightcone (verify_4 tracing) at source redshifts z_s = 0.1 ... 1.0.  Also per-crossing
statistics in redshift shells, and C(z)-corrected variants."""
import json, math, numpy as np
from common import XGP_OVER_LITERAL
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_4/'
KDIR = '/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/'
d = np.load(OUT + 'tngdark_crossings.npz')
nr = int(d['n_rays']); ray = d['ray']; zc = d['z']; lit = d['dm_literal']; xgp = XGP_OVER_LITERAL * lit
zsg = np.load(KDIR + 'redshifts_DMhalo.npy'); K = np.load(KDIR + 'DMhalo_r200_mass13to14.npy')
A = json.load(open(OUT + 'analytic_mean_dm.json'))
fitA, fitk = A['fit_z<1.0']['A'], A['fit_z<1.0']['k']
corr = lit * fitA * (1 + zc) ** fitk  # literal x C_TNG(z) fitted from Konietzka shells (analytic model)
rows = []
print(' zs | Konietzka: hit  mean_all mean_hit median_hit | TNGDark-xgp: hit  mean_all mean_hit median_hit | K/X all  K/X hit | Dark-corr mean_hit')
edges = np.concatenate([[0], np.logspace(0, np.log10(3000), 60)])
pdfs = {}
for j, zs in enumerate(zsg[:10]):
    m = zc <= zs + 1e-12
    dmx = np.bincount(ray[m], weights=xgp[m], minlength=nr)
    dmc = np.bincount(ray[m], weights=corr[m], minlength=nr)
    k = K[j]
    hk, hx, hc = k > 0, dmx > 0, dmc > 0
    row = dict(zs=float(zs), K_hit=hk.mean(), K_all=k.mean(), K_hit_mean=k[hk].mean(), K_med=np.median(k[hk]),
               X_hit=hx.mean(), X_all=dmx.mean(), X_hit_mean=dmx[hx].mean(), X_med=np.median(dmx[hx]),
               C_hit_mean=dmc[hc].mean(), C_med=np.median(dmc[hc]))
    row['K_over_X_all'] = row['K_all'] / row['X_all']; row['K_over_X_hit'] = row['K_hit_mean'] / row['X_hit_mean']
    rows.append(row)
    print(f"{zs:3.1f} | {row['K_hit']:.4f} {row['K_all']:7.2f} {row['K_hit_mean']:7.2f} {row['K_med']:7.2f} | "
          f"{row['X_hit']:.4f} {row['X_all']:7.2f} {row['X_hit_mean']:7.2f} {row['X_med']:7.2f} | "
          f"{row['K_over_X_all']:.3f} {row['K_over_X_hit']:.3f} | {row['C_hit_mean']:7.2f}")
    if round(zs, 1) in (0.1, 0.3, 0.5, 1.0):
        pk = np.histogram(k[hk], edges)[0] / hk.sum(); px = np.histogram(dmx[hx], edges)[0] / hx.sum(); pc = np.histogram(dmc[hc], edges)[0] / hc.sum()
        pdfs[f'{zs:.1f}'] = dict(edges=edges.tolist(), konietzka=pk.tolist(), tngdark_xgp=px.tolist(), tngdark_corr=pc.tolist())
        # quantiles
        qs = [0.1, 0.25, 0.5, 0.75, 0.9]
        print('     quantiles K  :', np.round(np.quantile(k[hk], qs), 1))
        print('     quantiles Xgp:', np.round(np.quantile(dmx[hx], qs), 1))
        print('     quantiles Cor:', np.round(np.quantile(dmc[hc], qs), 1))
        # PDF shape: mean |p_model/p_K - 1| over bins 3..1000 with >=50 K counts
        cen = np.sqrt(edges[1:] * np.maximum(edges[:-1], 1e-3))
        nk = np.histogram(k[hk], edges)[0]
        good = (cen > 3) & (cen < 1000) & (nk >= 50)
        mx = np.mean(np.abs(px[good] / pk[good] - 1)); mc = np.mean(np.abs(pc[good] / pk[good] - 1))
        print(f'     PDF shape mean|p/pK-1| (3-1000, per-hit normalised): xgp {mx:.3f}, corrected {mc:.3f}')
        pdfs[f'{zs:.1f}'].update(shape_xgp=mx, shape_corr=mc)

# per-crossing statistics by shell (single-crossing proxy): Konietzka positive increments per shell
print('\nshell   K: frac>0  median(dDM|>0)  mean | Dark-xgp crossings: median  mean | Dark-literal median')
shell = []
prev = np.zeros(nr)
for j, zs in enumerate(zsg[:10]):
    inc = K[j] - prev; prev = K[j].copy()
    pos = inc > 0
    m = (zc > zs - 0.1 + 1e-12) & (zc <= zs + 1e-12)
    s = dict(z_lo=float(zs - 0.1), z_hi=float(zs), K_frac=pos.mean(), K_med=np.median(inc[pos]), K_mean=inc[pos].mean(),
             X_med=np.median(xgp[m]), X_mean=xgp[m].mean(), L_med=np.median(lit[m]), n_cross=int(m.sum()))
    shell.append(s)
    print(f"{s['z_lo']:.1f}-{s['z_hi']:.1f}  {s['K_frac']:.4f} {s['K_med']:7.2f} {s['K_mean']:7.2f} | {s['X_med']:7.2f} {s['X_mean']:7.2f} | {s['L_med']:7.2f}  (n={s['n_cross']})")
json.dump(dict(cumulative=rows, shells=shell, pdfs=pdfs, C_fit=dict(A=fitA, k=fitk)), open(OUT + 'zs_comparison.json', 'w'), indent=1, default=float)
