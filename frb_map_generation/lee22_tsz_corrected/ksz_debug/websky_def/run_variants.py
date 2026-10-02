"""One-halo kSZ under candidate WebSky conventions, for a given (M200c, z) histogram (HalfDome or WebSky)."""
import numpy as np, h5py, sys, json
sys.path.insert(0, '/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_corrected/ksz_debug/websky_def')
from onehalo import *
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
which = sys.argv[1] if len(sys.argv) > 1 else 'halfdome'
cos = Cosmo()
if which == 'halfdome':
    D = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/'
    h = h5py.File(D + 'maps/halo_mass_redshift_histogram.h5', 'r')
    le, ze = h['logm_edges'][:], h['z_edges'][:]; n = h['count'][:].T; v2 = h['sum_vlos_over_c_squared'][:].T
else:
    d = np.load(OUT + 'websky_halo_hist.npz')
    le, ze, n, v2 = d['logm_edges'], d['z_edges'], d['count_m200c'], d['sum_vr_over_c_sq_m200c']
    if which == 'websky_z4':      # restrict to the HalfDome redshift range z<4
        zc = 0.5*(ze[1:]+ze[:-1]); n = n*(zc < 4.0)[None, :]; v2 = v2*(zc < 4.0)[None, :]
ell = np.unique(np.round(np.geomspace(30, 8192, 90)))
zt, rat = m200c_over_m200m_table(cos)
m200m_of = lambda LM, Z: 10**LM/np.interp(Z, zt, rat)
m200c_of = lambda LM, Z: 10**LM
ARC = np.pi/180/60
def sel_ws(LM, Z):
    M = 10**LM; R200 = (3*M/(4*np.pi*200*cos.rhoc(Z)))**(1/3)
    return (M > 1e13) & (R200/cos.DA(Z) > 0.5*ARC)
def sel_m(LM, Z): return 10**LM > 1e13
V = {
 'V0_ours':                     dict(),
 'V1_cut_M13':                  dict(select=sel_m),
 'V1_cut_M13_theta0.5':         dict(select=sel_ws),
 'V2_cut+comp_D3_M200m':        dict(select=sel_ws, comp=3.0, comp_mass_fn=m200m_of),
 'V2_cut+comp_D4_M200m':        dict(select=sel_ws, comp=4.0, comp_mass_fn=m200m_of),
 'V2_cut+comp_D3_M200c':        dict(select=sel_ws, comp=3.0, comp_mass_fn=m200c_of),
 'V2_nocut+comp_D3_M200m':      dict(comp=3.0, comp_mass_fn=m200m_of),
 'V3_cut_ne_websky(no0.9)':     dict(select=sel_ws, ne_conv='websky'),
 'V3_cut+comp_D3_M200m_no0.9':  dict(select=sel_ws, comp=3.0, comp_mass_fn=m200m_of, ne_conv='websky'),
 'V4_cut_printedexp':           dict(select=sel_ws, exp_form='printed'),
 'V4_cut_printedexp_no0.9':     dict(select=sel_ws, exp_form='printed', ne_conv='websky'),
 'V5_cut_meanbaryon':           dict(select=sel_ws, ne_conv='meanb'),
 'V6_cut_X1':                   dict(select=sel_ws, X=1.0),
 'V6_cut_X2':                   dict(select=sel_ws, X=2.0),
 'V6_cut_X6':                   dict(select=sel_ws, X=6.0),
 'V7_cut+comp_D3_M200m_printedexp_no0.9': dict(select=sel_ws, comp=3.0, comp_mass_fn=m200m_of, exp_form='printed', ne_conv='websky'),
}
T2 = 2.7255e6**2; pre = ell*(ell+1)/2/np.pi*T2
ws = np.loadtxt(OUT + 'stein2020_fig6_vector_halo.csv', delimiter=',')
ws = ws[ws[:, 0] > 10]
def wsb(L, w=0.1):   # WebSky halo D_ell averaged over |ln ell - ln L| < w
    m = np.abs(np.log(ws[:, 0]/L)) < w; return ws[m, 1].mean()
res = {}
Ls = [300, 1000, 2000, 3000, 5000]
print('%-42s' % which + ' '.join('%9s' % ('l=%d' % L) for L in Ls))
print('%-42s' % 'WebSky Fig6 halo (bin-avg)' + ' '.join('%9.4f' % wsb(L) for L in Ls))
for name, kw in V.items():
    cl, _ = onehalo_cl(cos, le, ze, n, v2, ell, **dict(kw))
    D = pre*cl; res[name] = D
    vals = [np.interp(np.log(L), np.log(ell), D) for L in Ls]
    print('%-42s' % name + ' '.join('%9.4f' % v for v in vals) + '   ratio/WS ' + ' '.join('%5.2f' % (v/wsb(L)) for v, L in zip(vals, Ls)))
np.savez(OUT + f'onehalo_variants_{which}.npz', ell=ell, **res)
