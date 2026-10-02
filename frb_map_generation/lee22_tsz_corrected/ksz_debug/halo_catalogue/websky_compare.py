"""HalfDome vs WebSky halo catalogues (full-catalogue histograms): M200m HMF vs Tinker08, v_los rms, and the B16 one-halo
kSZ power of the WebSky catalogue painted exactly like HalfDome (same a^2 table), with WebSky's M200c convention (NFW c200m = 7)
and halo-map selection (M200c > 1e13 Msun, theta200c > 0.5'). Usage: python websky_compare.py <websky_halo_hist.npz>"""
import sys
from common import *
from tinker08 import Tinker
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):          # onehalo_effects prints its own report on import
    from onehalo_effects import a2_at, cl_from, report, LIDX, ell, base, TH, lt, nc, vc, d, ws_mask, r7z
w = np.load(sys.argv[1])
le = w['logm200m_edges']; ze = w['z_edges']; lmc = 0.5 * (le[1:] + le[:-1]); zc = 0.5 * (ze[1:] + ze[:-1])
N = w['count']; V2 = w['sum_vlos2_kms2']; V3 = w['sum_v3d2_kms2']
print('WebSky halos in histogram', N.sum(), 'of', int(w['N']), ' RTHmax', float(w['RTHmax']), ' redshiftbox', float(w['redshiftbox']))
print('chi max of halos [Mpc]:', w['chi_edges_mpc'][np.nonzero(w['chi_count'])[0][-1] + 1])
# 1) HMF vs Tinker08 M200m (WebSky cosmology) and vs HalfDome M200m counts
t = Tinker('websky'); dlog = le[1] - le[0]
chi_e = np.array([t.chi(z) for z in ze]); dV = 4 * np.pi / 3 * np.diff(chi_e ** 3)
NT = np.zeros_like(N)
for j, z in enumerate(zc):
    NT[:, j] = t.dndlnm(10 ** lmc * 0.68, z, 200, 'mean') * np.log(10) * dlog * dV[j]
hd_n_m = d['hist_n_m']; hd_v2_m = d['hist_v2_m']; hd_v3_m = d['hist_v3_m']   # HalfDome grid: logm 11-16 step 0.025, z 0-4
shells = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]
mb = [12.0, 12.5, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0, 15.5]
print('\nM200m [Msun]: WebSky / Tinker08 (WebSky cosmo)  |  HalfDome / WebSky counts (full sky, same z shell)')
print('shell     ' + ' '.join(f'{a:.2f}-{b:.2f}' for a, b in zip(mb[:-1], mb[1:])))
for z1, z2 in shells:
    jz = (zc > z1) & (zc < z2); jh = jz[:hd_n_m.shape[1]]
    r1 = []; r2 = []
    for a, b in zip(mb[:-1], mb[1:]):
        im = (lmc > a) & (lmc < b)
        nw = N[np.ix_(im, jz)].sum(); nt = NT[np.ix_(im, jz)].sum(); nh = hd_n_m[np.ix_(im, jh)].sum()
        r1.append(f'{nw / nt:5.3f}' if nw > 100 else '  -  '); r2.append(f'{nh / nw:5.3f}' if nw > 100 and nh > 100 else '  -  ')
    print(f'{z1:.1f}-{z2:.1f} WS/T ' + '     '.join(r1)); print(f'        HD/WS ' + '     '.join(r2))
# 2) velocities
print('\nv_los rms [km/s] by z (all halos M200m > 10^13 Msun): WebSky | HalfDome ; |v|/v_los')
for z1, z2 in [(0, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0)]:
    jz = (zc > z1) & (zc < z2); jh = jz[:hd_n_m.shape[1]]; im = lmc > 13.0
    vw = np.sqrt(V2[np.ix_(im, jz)].sum() / N[np.ix_(im, jz)].sum()); vh = np.sqrt(hd_v2_m[np.ix_(im, jh)].sum() / hd_n_m[np.ix_(im, jh)].sum())
    print(f'  z {z1}-{z2}: {vw:6.1f} | {vh:6.1f}   ratio^2 HD/WS = {(vh / vw) ** 2:.3f}   |v|/vlos WS {np.sqrt(V3[np.ix_(im, jz)].sum() / V2[np.ix_(im, jz)].sum()):.3f}')
print('v_los rms by mass at z 0.2-1.0: WebSky | HalfDome')
for a in [13.0, 13.5, 14.0, 14.5]:
    jz = (zc > 0.2) & (zc < 1.0); jh = jz[:hd_n_m.shape[1]]; im = (lmc > a) & (lmc < a + 0.5)
    print(f'  logM200m {a}-{a + 0.5}: {np.sqrt(V2[np.ix_(im, jz)].sum() / N[np.ix_(im, jz)].sum()):.1f} | {np.sqrt(hd_v2_m[np.ix_(im, jh)].sum() / hd_n_m[np.ix_(im, jh)].sum()):.1f}')
# 3) one-halo kSZ of the WebSky catalogue painted like HalfDome
print('\nOne-halo B16 (4 R200c sphere) D_ell [uK^2] at ell = 300 1000 2000 3000 5000; ratio to HalfDome base')
r7 = lambda z: np.log10(m200c_over_m200m_c7(np.array([z]), 0.31)[0])
def cl_ws(shift, mask=None, v2=V2, n=N):
    cl = np.zeros(len(ell))
    for j in range(len(zc)):
        im = np.nonzero(n[:, j] > 0)[0]
        if len(im) == 0: continue
        lm = lmc[im] + shift(zc[j]); wgt = v2[im, j] / C ** 2
        keep = lm >= lt[0]                      # table starts at 10^12.0125 Msun; lower masses contribute negligibly
        if mask is not None: wgt = wgt * mask(lm, j)
        cl += (a2_at(lm[keep], j) * wgt[keep]).sum(1)
    return cl / (4 * np.pi)
ws_all = report('WebSky cat, M200c=c7(M200m), all halos M200c>=10^12', cl_ws(r7), base)
ws_m13 = report('WebSky cat, c7, M200c>=1e13 only', cl_ws(r7, lambda lm, j: (lm >= 13.0).astype(float)), base)
ws_cut = report('WebSky cat, c7, WebSky cuts (M200c>=1e13, theta200c>0.5\')', cl_ws(r7, ws_mask), base)
hd_cut = report('HalfDome cat, c7 M200c from M200m, WebSky cuts', cl_from(hd_n_m, hd_v2_m, shift=r7z, mask=ws_mask), base)
# WebSky catalogue with HalfDome velocities rescaled per z (v^2 ratio per z cell)
vw_z = V2[lmc > 13.0].sum(0) / np.maximum(N[lmc > 13.0].sum(0), 1)
vh_z = hd_v2_m[lmc > 13.0].sum(0) / np.maximum(hd_n_m[lmc > 13.0].sum(0), 1)
fac = np.ones(len(zc)); nzh = len(vh_z); fac[:nzh] = np.where(vw_z[:nzh] > 0, vh_z / np.maximum(vw_z[:nzh], 1e-9), 1)
report('WebSky cat, c7, WebSky cuts, v^2 rescaled to HalfDome(z)', cl_ws(r7, ws_mask, v2=V2 * fac[None, :]), base)
dig = np.loadtxt('/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/websky_halo_stein2020_fig6_digitized_binned.csv', delimiter=',')
print('\nDigitized WebSky Halo (Stein+20 Fig 6): ' + ' '.join(f'ell {L:.0f}: {np.interp(np.log(L), np.log(dig[:, 0]), dig[:, 1]):.3f}' for L in (1000, 2000, 3000, 5000)))
print('WebSky-catalogue one-halo (cuts) / digitized: ' + ' '.join(f'ell {ell[i]:.0f}: {dl(ell[i], ws_cut[i]) / np.interp(np.log(ell[i]), np.log(dig[:, 0]), dig[:, 1]):.3f}' for i in LIDX[1:]))
print('HalfDome base one-halo / WebSky-catalogue one-halo (cuts): ' + ' '.join(f'ell {ell[i]:.0f}: {base[i] / ws_cut[i]:.3f}' for i in LIDX))
np.savez(OUTD + 'websky_vs_halfdome_onehalo.npz', ell=ell, cl_halfdome_base=base, cl_ws_all=ws_all, cl_ws_m13=ws_m13, cl_ws_cuts=ws_cut, cl_hd_c7_cuts=hd_cut)
# 4) isolate velocity vs abundance: HalfDome counts with WebSky <v_los^2>(M200m, z) per cell (coarse 0.25 dex fallback)
nzh = hd_n_m.shape[1]
def coarse_mean(n, v2, nzcells):
    out = np.zeros((n.shape[0], nzcells))
    for j in range(nzcells):
        for a in np.arange(11.0, 16.0, 0.25):
            im = (lmc >= a) & (lmc < a + 0.25); nn = n[im, j].sum()
            out[im, j] = v2[im, j].sum() / nn if nn > 0 else np.nan
        bad = ~np.isfinite(out[:, j]); out[bad, j] = v2[:, j].sum() / max(n[:, j].sum(), 1)
    return out
vws = coarse_mean(N[:, :nzh], V2[:, :nzh], nzh); vhd = coarse_mean(hd_n_m, hd_v2_m, nzh)
hd_wsvel = report('HalfDome cat (c7 M200c, WebSky cuts) with WebSky <v^2>(M,z)', cl_from(hd_n_m, hd_n_m * vws, shift=r7z, mask=ws_mask), base)
report('  same, relative to HalfDome velocities', cl_from(hd_n_m, hd_n_m * vws, shift=r7z, mask=ws_mask), hd_cut)
report('HalfDome cat (catalogue M200c, no cuts) with WebSky <v^2>(M200m bin,z)', cl_from(hd_n_m, hd_n_m * vws, shift=lambda z: np.log10(d['acc_ratio'][min(int(z / 0.1), 39)] / max(d['acc_n'][min(int(z / 0.1), 39)], 1))), cl_from(hd_n_m, hd_v2_m, shift=lambda z: np.log10(d['acc_ratio'][min(int(z / 0.1), 39)] / max(d['acc_n'][min(int(z / 0.1), 39)], 1))))
ws_hdvel = report('WebSky cat (c7, cuts) with HalfDome <v^2>(M,z)', cl_ws(r7, ws_mask, v2=np.concatenate([N[:, :nzh] * vhd, V2[:, nzh:]], 1)), base)
report('  same, relative to WebSky velocities', cl_ws(r7, ws_mask, v2=np.concatenate([N[:, :nzh] * vhd, V2[:, nzh:]], 1)), ws_cut)
print('\nv_los rms [km/s] per log M200m bin (WebSky | HalfDome) at z 0-0.5 and 0.5-1:')
for z1, z2 in ((0, 0.5), (0.5, 1.0)):
    jz = (zc > z1) & (zc < z2); jh = jz[:nzh]
    print(f'  z {z1}-{z2}: ' + '; '.join(f'{a}: {np.sqrt(V2[np.ix_((lmc >= a) & (lmc < a + 0.5), jz)].sum() / max(N[np.ix_((lmc >= a) & (lmc < a + 0.5), jz)].sum(), 1)):.0f} | {np.sqrt(hd_v2_m[np.ix_((lmc >= a) & (lmc < a + 0.5), jh)].sum() / max(hd_n_m[np.ix_((lmc >= a) & (lmc < a + 0.5), jh)].sum(), 1)):.0f}' for a in (12.5, 13.0, 13.5, 14.0, 14.5, 15.0)))
