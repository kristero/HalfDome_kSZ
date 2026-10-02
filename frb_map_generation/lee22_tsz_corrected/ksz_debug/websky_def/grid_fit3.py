import numpy as np, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
import healpy as hp
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
cos = Cosmo()
d = np.load(OUT + 'websky_halo_hist.npz'); le, ze, n, v2 = d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c']
ws = np.loadtxt(OUT + 'stein2020_fig6_vector_halo.csv', delimiter=','); ws = ws[ws[:, 0] > 10]
Lfit = np.array([300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]); ell = Lfit.astype(float)
wsv = np.array([ws[np.abs(np.log(ws[:, 0]/L)) < 0.1, 1].mean() for L in Lfit])
pw2 = hp.pixwin(4096, lmax=6000)[Lfit]**2
pre = ell*(ell+1)/2/np.pi*2.7255e6**2
zt, rat = m200c_over_m200m_table(cos); m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
ARC = np.pi/180/60
def sel(LM, Z):
    M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3); return (M > 1e13) & (R200/cos.DA(Z) > 0.5*ARC)
def rep(tag, kw):
    cl, _ = onehalo_cl(cos, le, ze, n, v2, ell, select=sel, **kw)
    for pwf in (False, True):
        r = pre*cl*(pw2 if pwf else 1)/wsv
        print('%-44s pixwin=%d %s | rms %.3f' % (tag, pwf, ' '.join('%.3f' % x for x in r), np.sqrt(np.mean(np.log(r)**2))), flush=True)
print('ells', list(Lfit))
for X in [1.5, 2.0, 2.5, 3.0, 4.0]:
    for ne in ['websky', 'xgp']:
        rep(f'X={X} ne={ne} comp D3 matched-electrons', dict(X=X, ne_conv=ne, comp=3.0, comp_mass_fn=m200m_of, comp_match=True))
for X in [1.5, 2.0]:
    rep(f'X={X} ne=websky comp D3 f_b M200m', dict(X=X, ne_conv='websky', comp=3.0, comp_mass_fn=m200m_of))
