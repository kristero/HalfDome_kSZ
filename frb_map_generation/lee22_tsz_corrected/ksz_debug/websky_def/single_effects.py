"""Single-change effects on our HalfDome one-halo term, and the WebSky best-fit model restricted to z<3.9."""
import numpy as np, h5py, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
cos = Cosmo()
D = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/'
h = h5py.File(D + 'maps/halo_mass_redshift_histogram.h5', 'r')
HD = (h['logm_edges'][:], h['z_edges'][:], h['count'][:].T, h['sum_vlos_over_c_squared'][:].T)
d = np.load(OUT + 'websky_halo_hist.npz')
Ls = np.array([300, 1000, 2000, 3000, 5000]); ell = Ls.astype(float); pre = ell*(ell+1)/2/np.pi*2.7255e6**2
zt, rat = m200c_over_m200m_table(cos); m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
ARC = np.pi/180/60
def sel(LM, Z):
    M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3); return (M > 1e13) & (R200/cos.DA(Z) > 0.5*ARC)
def run(cat, **kw):
    cl, _ = onehalo_cl(cos, *cat, ell, **kw); return pre*cl
base = run(HD)
for name, kw in [('cut only', dict(select=sel)), ('compensation only (D3, f_b M200m)', dict(comp=3.0, comp_mass_fn=m200m_of)),
                 ('truncation 1.5 R200c only', dict(X=1.5)), ('truncation 2.0 R200c only', dict(X=2.0)),
                 ('no 0.9 only', dict(ne_conv='websky')), ('trunc 1.5 + comp', dict(X=1.5, comp=3.0, comp_mass_fn=m200m_of))]:
    print('HalfDome %-36s ratio to ours at %s: %s' % (name, list(Ls), ' '.join('%.3f' % v for v in run(HD, **kw)/base)))
zc = 0.5*(d['z_edges'][1:]+d['z_edges'][:-1])
best = dict(select=sel, comp=3.0, comp_mass_fn=m200m_of, X=1.5, ne_conv='websky')
WS = (d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c'])
WSz = (d['logm_edges'], d['z_edges'], d['count_m200c']*(zc < 3.9), d['sum_vr_over_c_sq_m200c']*(zc < 3.9))
a, b = run(WS, **best), run(WSz, **best)
print('WebSky best model z<3.9 / all z:', ' '.join('%.3f' % v for v in b/a))
a, b = run(WS), run(WSz)
print('WebSky ours-conventions z<3.9 / all z:', ' '.join('%.3f' % v for v in b/a))
