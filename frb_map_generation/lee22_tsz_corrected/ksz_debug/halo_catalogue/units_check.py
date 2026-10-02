"""Units check: HMF of halo_mass_m200c read as Msun/h vs as Msun, vs Tinker08 M200c; M_RFoF vs M200m; particle mass."""
from common import *
from tinker08 import Tinker
import h5py
f = h5py.File('/home/cbllover/HalfDome/lightcone_100.hdf5', 'r')
N = f['redshift'].shape[0]; idx = slice(0, N, 97)
z = f['redshift'][idx]; mc = f['halo_mass_m200c'][idx]; mm = f['halo_mass_m200m'][idx]; lr = f['log10_m_rfof'][idx]; L = f['Length'][idx]
print('sample', len(z))
mp = 10 ** lr / L
print('M_rfof / Length [Msun/h?]: median %.4g, 1-99%%: %.4g %.4g' % (np.median(mp), *np.percentile(mp, [1, 99])))
print('expected particle mass 2.775e11*0.3089*(3750/6144)^3 = %.4g Msun/h' % (2.775e11 * 0.3089 * (3750 / 6144) ** 3))
for z1, z2 in [(0, 0.3), (0.5, 0.7), (1, 1.2), (2, 2.3), (3, 3.5)]:
    s = (z > z1) & (z < z2)
    for a, b in [(13.0, 13.3), (13.8, 14.2), (14.5, 15.5)]:
        q = s & (np.log10(mm) > a) & (np.log10(mm) < b)
        if q.sum() < 20: continue
        print(f'z {z1}-{z2} logM200m[h-1] {a}-{b}: N={q.sum():6d} <M200m/M_rfof>={np.median(mm[q] / 10 ** lr[q]):.3f}  <M200c/M200m>={np.median(mc[q] / mm[q]):.3f} (16-84%: {np.percentile(mc[q] / mm[q], 16):.3f}-{np.percentile(mc[q] / mm[q], 84):.3f})')
    # NFW predictions
    zz = 0.5 * (z1 + z2)
    for lmh in (13.15, 14.0):
        Mh_c = 10 ** lmh * 0.8
        for _ in range(5): Mh_c = 10 ** lmh * m200c_over_m200m_duffy(Mh_c, zz)[0]
        print(f'   NFW M200c/M200m at z={zz:.2f}, M200m=10^{lmh}: c200m=7 -> {m200c_over_m200m_c7(np.array([zz]), 0.3089)[0]:.3f}; Duffy08 -> {Mh_c / 10 ** lmh:.3f}')
# monotonic mapping? M200c vs M200m scatter at fixed z
s = (z > 0.5) & (z < 0.52)
r = mc[s] / mm[s]; print('at z~0.51 scatter of M200c/M200m (std of log10):', np.std(np.log10(r)), ' rank corr M200m vs M200c:', np.corrcoef(np.argsort(np.argsort(mm[s])), np.argsort(np.argsort(mc[s])))[0, 1])
# HMF read as Msun instead of Msun/h
t = Tinker('halfdome')
d = np.load(OUTD + 'halfdome_catalogue_pass.npz'); le = d['logm_edges']; lmc = 0.5 * (le[1:] + le[:-1]); ze = d['z_edges']; zc = 0.5 * (ze[1:] + ze[:-1])
for z1, z2 in [(0.25, 0.5), (0.75, 1.0)]:
    jz = np.nonzero((zc > z1) & (zc < z2))[0]
    dV = 4 * np.pi / 3 * (t.chi(z2) ** 3 - t.chi(z1) ** 3)
    for a, b in [(13.25, 13.5), (14.0, 14.25), (14.5, 14.75)]:
        im = (lmc > a) & (lmc < b); nh = d['hist_n_c'][np.ix_(im, jz)].sum()
        # catalogue value in Msun/h: log10 Mcat = lmc + log10 0.68 (painter /0.68)
        lcat = np.linspace(a, b, 21) + np.log10(HPAINT)
        zm = 0.5 * (z1 + z2)
        nt_h = np.trapz(t.dndlnm(10 ** lcat, zm, 200, 'crit') * np.log(10), lcat) * dV            # catalogue as Msun/h
        nt_msun = np.trapz(t.dndlnm(10 ** (lcat + np.log10(t.p['h'])), zm, 200, 'crit') * np.log(10), lcat) * dV  # catalogue as Msun
        print(f'z {z1}-{z2} log10(M200c_cat/0.68) {a}-{b}: HalfDome/Tinker = {nh / nt_h:.3f} (Msun/h reading)  {nh / nt_msun:.3f} (Msun reading)')
