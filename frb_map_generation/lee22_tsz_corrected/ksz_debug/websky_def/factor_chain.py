"""Sequential factor chain from our HalfDome B16 one-halo term to the WebSky Fig. 6 halo curve."""
import numpy as np, h5py, sys
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
cos = Cosmo()
D = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/'
h = h5py.File(D + 'maps/halo_mass_redshift_histogram.h5', 'r')
HD = (h['logm_edges'][:], h['z_edges'][:], h['count'][:].T, h['sum_vlos_over_c_squared'][:].T)
d = np.load(OUT + 'websky_halo_hist.npz'); WS = (d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c'])
ws = np.loadtxt(OUT + 'stein2020_fig6_vector_halo.csv', delimiter=','); ws = ws[ws[:, 0] > 10]
Ls = np.array([300, 1000, 2000, 3000, 5000]); ell = Ls.astype(float)
wsv = np.array([ws[np.abs(np.log(ws[:, 0]/L)) < 0.1, 1].mean() for L in Ls])
pre = ell*(ell+1)/2/np.pi*2.7255e6**2
zt, rat = m200c_over_m200m_table(cos); m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
ARC = np.pi/180/60
def sel(LM, Z):
    M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3); return (M > 1e13) & (R200/cos.DA(Z) > 0.5*ARC)
S = h5py.File(D + 'ksz_poisson_mass_split.h5', 'r')
spec = np.load(D + 'spectra/ksz_spectra.npz'); lm = spec['ell']; Dmap = lm*(lm+1)/2/np.pi*spec['cl_b16']*2.7255e6**2
Dmap_b = np.array([Dmap[(lm > L*0.9) & (lm < L*1.1)].mean() for L in Ls])
steps = [('ours: HalfDome cat, X=4, ne xgp(0.9), no cut, no comp', HD, dict()),
         ('+ WebSky catalogue (Tinker-matched M200c via c=7, 2LPT v)', WS, dict()),
         ('+ cut M200c>1e13 & theta200c>0.5\'', WS, dict(select=sel)),
         ('+ compensation (uniform sphere, 3 rho_m, f_b M200m)', WS, dict(select=sel, comp=3.0, comp_mass_fn=m200m_of)),
         ('+ truncation 4 -> 1.5 R200c', WS, dict(select=sel, comp=3.0, comp_mass_fn=m200m_of, X=1.5)),
         ('+ n_e = (1-Y/2) rho/m_p (drop XGPaint 0.9)', WS, dict(select=sel, comp=3.0, comp_mass_fn=m200m_of, X=1.5, ne_conv='websky'))]
print('ell                         ' + ' '.join('%8d' % L for L in Ls))
print('%-62s' % 'HalfDome B16 map (anafast, as in task)' + ' '.join('%8.4f' % v for v in Dmap_b))
print('%-62s' % 'WebSky Fig.6 halo (exact vector, +-10% ell avg)' + ' '.join('%8.4f' % v for v in wsv))
prev = None; rows = {}
for name, cat, kw in steps:
    cl, _ = onehalo_cl(cos, *cat, ell, **dict(kw)); Dv = pre*cl
    f = '' if prev is None else ' step x ' + ' '.join('%5.2f' % v for v in Dv/prev)
    print('%-62s' % name + ' '.join('%8.4f' % v for v in Dv) + '  /WS ' + ' '.join('%5.2f' % v for v in Dv/wsv) + f)
    prev = Dv; rows[name] = Dv
# same WebSky conventions on the HalfDome catalogue
for name, kw in [('HalfDome cat + all WebSky conventions (cut, comp, X=1.5, no 0.9)', dict(select=sel, comp=3.0, comp_mass_fn=m200m_of, X=1.5, ne_conv='websky')),
                 ('HalfDome cat + WebSky conventions but keep 0.9', dict(select=sel, comp=3.0, comp_mass_fn=m200m_of, X=1.5))]:
    cl, _ = onehalo_cl(cos, *HD, ell, **dict(kw)); Dv = pre*cl
    print('%-62s' % name + ' '.join('%8.4f' % v for v in Dv) + '  /WS ' + ' '.join('%5.2f' % v for v in Dv/wsv))
np.savez(OUT + 'factor_chain.npz', ell=ell, websky=wsv, map=Dmap_b, **{k[:40]: v for k, v in rows.items()})
