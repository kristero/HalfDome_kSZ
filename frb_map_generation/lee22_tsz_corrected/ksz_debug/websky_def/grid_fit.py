"""Grid of WebSky-convention one-halo models on the WebSky (or HalfDome) catalogue histogram vs the exact Fig. 6 halo curve."""
import numpy as np, h5py, sys, itertools
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
import healpy as hp
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
which = sys.argv[1] if len(sys.argv) > 1 else 'websky'
cos = Cosmo()
if which == 'websky':
    d = np.load(OUT + 'websky_halo_hist.npz'); le, ze, n, v2 = d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c']
else:
    D = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/'
    h = h5py.File(D + 'maps/halo_mass_redshift_histogram.h5', 'r')
    le, ze = h['logm_edges'][:], h['z_edges'][:]; n = h['count'][:].T; v2 = h['sum_vlos_over_c_squared'][:].T
ws = np.loadtxt(OUT + 'stein2020_fig6_vector_halo.csv', delimiter=','); ws = ws[ws[:, 0] > 10]
Lfit = np.array([300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000])
wsv = np.array([ws[np.abs(np.log(ws[:, 0]/L)) < 0.1, 1].mean() for L in Lfit])
ell = Lfit.astype(float)
pw2 = hp.pixwin(4096, lmax=6000)[Lfit]**2
pre = ell*(ell+1)/2/np.pi*2.7255e6**2
zt, rat = m200c_over_m200m_table(cos)
m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
ARC = np.pi/180/60
def mk_sel(thmin):
    def s(LM, Z):
        M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3)
        return (M > 1e13) & (R200/cos.DA(Z) > thmin*ARC)
    return s
rows = []
for X, ne, comp, th in itertools.product([1.5, 2.0, 2.5, 3.0, 4.0], ['xgp', 'websky'], [None, 3.0], [0.5, 0.25]):
    kw = dict(select=mk_sel(th), X=X, ne_conv=ne)
    if comp: kw.update(comp=comp, comp_mass_fn=m200m_of)
    cl, _ = onehalo_cl(cos, le, ze, n, v2, ell, **kw)
    Dm = pre*cl
    for pwf in [False, True]:
        r = Dm*(pw2 if pwf else 1)/wsv
        rows.append((X, ne, comp, th, pwf, r))
        print('X=%.1f ne=%-6s comp=%-4s theta>%.2f pixwin=%d  ratio model/WS at %s: %s  | rms log %.3f  mean %.3f' % (
            X, ne, comp, th, pwf, list(Lfit), ' '.join('%.2f' % x for x in r), np.std(np.log(r)), np.exp(np.mean(np.log(r)))), flush=True)
best = sorted(rows, key=lambda t: np.sqrt(np.mean(np.log(t[5])**2)))[:10]
print('\nBest 10 by rms |log ratio| (amplitude AND shape):')
for t in best: print(t[:5], ' '.join('%.2f' % x for x in t[5]), 'rms %.3f' % np.sqrt(np.mean(np.log(t[5])**2)))
best = sorted(rows, key=lambda t: np.std(np.log(t[5])))[:10]
print('\nBest 10 by shape only (std log ratio):')
for t in best: print(t[:5], ' '.join('%.2f' % x for x in t[5]), 'std %.3f mean %.3f' % (np.std(np.log(t[5])), np.exp(np.mean(np.log(t[5])))))
np.save(OUT + f'grid_fit_{which}.npy', np.array(rows, dtype=object), allow_pickle=True)
