"""Degeneracy scan: truncation X vs compensation-sphere overdensity Delta, n_e = (1-Y/2) rho/m_p, WebSky M200c histogram."""
import numpy as np, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
cos = Cosmo()
d = np.load(OUT + 'websky_halo_hist.npz'); le, ze, n, v2 = d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c']
ws = np.loadtxt(OUT + 'stein2020_fig6_vector_halo.csv', delimiter=','); ws = ws[ws[:, 0] > 10]
Lfit = np.array([300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000]); ell = Lfit.astype(float)
wsv = np.array([ws[np.abs(np.log(ws[:, 0]/L)) < 0.1, 1].mean() for L in Lfit])
pre = ell*(ell+1)/2/np.pi*2.7255e6**2
zt, rat = m200c_over_m200m_table(cos); m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
ARC = np.pi/180/60
def sel(LM, Z):
    M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3); return (M > 1e13) & (R200/cos.DA(Z) > 0.5*ARC)
print('ells', list(Lfit))
for X in [2.0, 3.0, 4.0]:
    for Dl in [3.0, 10.0, 30.0, 100.0]:
        for match in [False, True]:
            cl, _ = onehalo_cl(cos, le, ze, n, v2, ell, select=sel, X=X, ne_conv='websky', comp=Dl, comp_mass_fn=m200m_of, comp_match=match)
            r = pre*cl/wsv
            print('X=%.1f Delta=%5.0f %-14s %s | rms %.3f' % (X, Dl, 'matched' if match else 'f_b*M200m', ' '.join('%.3f' % x for x in r),
                  np.sqrt(np.mean(np.log(r)**2))), flush=True)
