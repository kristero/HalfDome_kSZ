"""Effect of halo-catalogue properties on the HalfDome B16 one-halo kSZ power (catalogue Poisson term), using the per-cell
a^2(ell; M200c, z) table (validated against ksz/poisson_mass_split.jl) and the full-catalogue histograms."""
from common import *
t = load_table(); ell = t['ell']; A2 = t['a2']; TH = t['th']            # A2 (ell, logm 148, z 188)
lt = t['logm']; zt = t['zc']; nzt = len(zt)
d = np.load(OUTD + 'halfdome_catalogue_pass.npz'); tk = np.load(OUTD + 'tinker08_expected_counts.npz')
le = d['logm_edges']; ze = d['z_edges']; lmc = 0.5 * (le[1:] + le[:-1]); zc = 0.5 * (ze[1:] + ze[:-1]); nz = len(zc)
LIDX = [list(ell).index(L) for L in (300.0, 1000.0, 2000.0, 3000.0, 5000.0)]
def a2_at(logm, iz):
    """a^2 for each ell at masses logm (array) in z-cell iz (log-interp in mass, clipped to table)."""
    la = np.log(np.maximum(A2[:, :, iz], 1e-300))
    x = np.clip(logm, lt[0], lt[-1])
    return np.exp(np.array([np.interp(x, lt, la[k]) for k in range(len(ell))]))     # (ell, n)
def cl_from(nh, v2sum, shift=None, mask=None):
    """nh, v2sum on the pass grid (logm 11-16, z 0-4, (km/s)^2). shift(z) -> dlog10M added (e.g. M200m -> M200c)."""
    cl = np.zeros(len(ell))
    for j in range(nz):
        im = np.nonzero(nh[:, j] > 0)[0]
        if len(im) == 0: continue
        lm = lmc[im] + (shift(zc[j]) if shift is not None else 0.0)
        w = v2sum[im, j] / C ** 2
        if mask is not None: w = w * mask(lm, j)
        cl += (a2_at(lm, j) * w).sum(1)
    return cl / (4 * np.pi)
def report(name, cl, base=None):
    s = ' '.join(f'{dl(ell[i], cl[i]):.4f}' for i in LIDX)
    r = '' if base is None else '  ratio ' + ' '.join(f'{cl[i] / base[i]:.3f}' for i in LIDX)
    print(f'{name:58s} {s}{r}'); return cl
print('D_ell [uK^2] at ell = 300 1000 2000 3000 5000')
nc, vc = d['hist_n_c'], d['hist_v2_c']; nm_, vm_ = d['hist_n_m'], d['hist_v2_m']
base = report('base: catalogue M200c (= painter one-halo)', cl_from(nc, vc))
# h = 0.6774 instead of 0.68 in Msun/h -> Msun
report('mass / 0.6774 instead of / 0.68', cl_from(nc, vc, shift=lambda z: np.log10(0.68 / 0.6774)), base)
# abundance -> Tinker08 M200c (complete range logM >= 13.0), keep v^2 mean per (0.25 dex, z cell)
def tinker_replace(NT, lo=13.0):
    n2 = nc.copy(); v2 = vc.copy()
    for j in range(nz):
        for a in np.arange(lo, 15.75, 0.25):
            im = (lmc > a) & (lmc < a + 0.25)
            nn = nc[im, j].sum()
            vmean = vc[im, j].sum() / nn if nn > 0 else (vc[:, j].sum() / max(nc[:, j].sum(), 1))
            n2[im, j] = NT[im, j]; v2[im, j] = NT[im, j] * vmean
    return n2, v2
for key in ['NT_c_halfdome', 'NT_c_websky']:
    n2, v2 = tinker_replace(tk[key]); report(f'abundance -> Tinker08 M200c ({key[5:]} cosmology), M>=1e13', cl_from(n2, v2), base)
# M200c from catalogue M200m with WebSky's NFW c200m = 7 conversion
r7z = lambda z: np.log10(m200c_over_m200m_c7(np.array([z]))[0])
c7 = report('M200c := M200m x [c200m=7 NFW] (WebSky convention)', cl_from(nm_, vm_, shift=r7z), base)
# self-consistency: M200m hist with the catalogue's own mean ratio per 0.1 z
zb = d['zb']; rmean = d['acc_ratio'] / np.maximum(d['acc_n'], 1)
own = lambda z: np.log10(rmean[min(int(z / 0.1), len(rmean) - 1)])
report('M200c := M200m x <catalogue M200c/M200m>(z) (check)', cl_from(nm_, vm_, shift=own), base)
# Duffy08 conversion from M200m (iterate c(M200c))
def duffy_shift_fn(lm_m, z):
    Mh_m = 10 ** lm_m * HPAINT; Mc = Mh_m * 0.85
    for _ in range(5):
        Mc = Mh_m * m200c_over_m200m_duffy(Mc, z)
    return np.log10(Mc / Mh_m)
cl = np.zeros(len(ell))
for j in range(nz):
    im = np.nonzero(nm_[:, j] > 0)[0]
    if len(im) == 0: continue
    lm = lmc[im] + duffy_shift_fn(lmc[im], zc[j]); cl += (a2_at(lm, j) * vm_[im, j] / C ** 2).sum(1)
report('M200c := M200m x [Duffy08 NFW]', cl / (4 * np.pi), base)
# WebSky halo-map selection: M200c > 1e13 Msun and theta200c > 0.5 arcmin
TH_CUT = 0.5 / 60 * np.pi / 180
def ws_mask(lm, j):
    th = np.exp(np.interp(np.clip(lm, lt[0], lt[-1]), lt, np.log(TH[:, j])))
    return ((lm >= 13.0) & (th > TH_CUT)).astype(float)
report('WebSky cuts (M200c>=1e13 Msun, theta200c>0.5\')', cl_from(nc, vc, mask=ws_mask), base)
report('WebSky mass cut only (M200c>=1e13 Msun)', cl_from(nc, vc, mask=lambda lm, j: (lm >= 13.0).astype(float)), base)
report('WebSky angle cut only (theta200c>0.5\')', cl_from(nc, vc, mask=lambda lm, j: ws_mask(np.full_like(lm, 99.0), j) * 0 + (np.exp(np.interp(np.clip(lm, lt[0], lt[-1]), lt, np.log(TH[:, j]))) > TH_CUT)), base)
report('c200m=7 M200c + WebSky cuts', cl_from(nm_, vm_, shift=r7z, mask=ws_mask), base)
# velocity: linear-theory 1-D rms (halo-smoothed, 1e13.5 Msun/h) instead of catalogue v^2 per z
from linear_velocity import vrms
from tinker08 import Tinker
tw = Tinker('websky')
vl = np.array([vrms(tw, z, 10 ** 13.5)[0] for z in zc])
vcat = np.sqrt(vc.sum(0) / np.maximum(nc.sum(0), 1))
vc_lin = vc * (vl ** 2 / np.maximum(vcat ** 2, 1))[None, :]
report('v_los^2 -> linear 1-D rms(z) (WebSky cosmo, R_L(10^13.5))', cl_from(nc, vc_lin), base)
np.savez(OUTD + 'vlos_rms_vs_linear.npz', zc=zc, v_cat=vcat, v_lin_1d=vl)
for z in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0):
    j = int(z / 0.025); print(f'  z={zc[j]:.3f}: catalogue v_los rms {vcat[j]:.0f} km/s, linear 1-D {vl[j]:.0f} km/s')
# mass-bin and redshift split of base at ell 1000 / 3000
for L in (1000.0, 3000.0):
    k = list(ell).index(L)
    print(f'ell={L:.0f}: one-halo fraction by z: ' + ' '.join(
        f'{a}-{b}:{cl_from(nc, vc, mask=lambda lm, j, a=a, b=b: float(a <= zc[j] < b) * np.ones_like(lm))[k] / base[k]:.3f}'
        for a, b in ((0, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 4))))
