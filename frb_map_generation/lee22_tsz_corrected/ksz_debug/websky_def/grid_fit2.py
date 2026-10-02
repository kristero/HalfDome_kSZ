"""Refined grid: truncation radius in R200c or R200m units, compensation overdensity/mass, n_e convention."""
import numpy as np, sys, itertools
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
cos = Cosmo()
d = np.load(OUT + 'websky_halo_hist.npz'); le, ze, n, v2 = d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c']
zr, rr = d['z_table'], d['r200c_over_r200m']
ws = np.loadtxt(OUT + 'stein2020_fig6_vector_halo.csv', delimiter=','); ws = ws[ws[:, 0] > 10]
Lfit = np.array([300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]); ell = Lfit.astype(float)
wsv = np.array([ws[np.abs(np.log(ws[:, 0]/L)) < 0.1, 1].mean() for L in Lfit])
pre = ell*(ell+1)/2/np.pi*2.7255e6**2
zt, rat = m200c_over_m200m_table(cos)
m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
m200c_of = lambda LM, Z: 10**LM
ARC = np.pi/180/60
def sel(LM, Z):
    M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3)
    return (M > 1e13) & (R200/cos.DA(Z) > 0.5*ARC)
def report(tag, kw):
    cl, _ = onehalo_cl(cos, le, ze, n, v2, ell, select=sel, **kw)
    r = pre*cl/wsv
    print('%-48s %s | rms %.3f mean %.3f' % (tag, ' '.join('%.3f' % x for x in r), np.sqrt(np.mean(np.log(r)**2)), np.exp(np.mean(np.log(r)))), flush=True)
    return r
print('ells', list(Lfit)); print('WebSky', ' '.join('%.4f' % x for x in wsv))
for Xc in [1.0, 1.25, 1.5, 1.75, 2.0]:
    for ne in ['websky', 'xgp']:
        report(f'X={Xc} R200c ne={ne} comp D3 M200m', dict(X=Xc, ne_conv=ne, comp=3.0, comp_mass_fn=m200m_of))
for Xm in [1.0, 1.25, 1.5, 2.0]:
    for ne in ['websky', 'xgp']:
        report(f'X={Xm} R200m ne={ne} comp D3 M200m', dict(X_fn=lambda LM, Z, Xm=Xm: Xm/np.interp(Z, zr, rr), ne_conv=ne, comp=3.0, comp_mass_fn=m200m_of))
for D in [1.0, 2.0, 4.0]:
    report(f'X=1.5 R200c ne=websky comp D{D} M200m', dict(X=1.5, ne_conv='websky', comp=D, comp_mass_fn=m200m_of))
report('X=1.5 R200c ne=websky comp D3 M200c', dict(X=1.5, ne_conv='websky', comp=3.0, comp_mass_fn=m200c_of))
report('X=1.5 R200c ne=websky NO comp', dict(X=1.5, ne_conv='websky'))
report('X=1.0 R200m ne=websky NO comp', dict(X_fn=lambda LM, Z: 1.0/np.interp(Z, zr, rr), ne_conv='websky'))
