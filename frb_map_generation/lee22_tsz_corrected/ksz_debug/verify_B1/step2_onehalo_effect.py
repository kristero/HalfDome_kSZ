"""B1 verification, step 2: effect of the low-z cache error on the catalogue one-halo kSZ power.
Per-halo (actual z, actual v_los) for z < 0.03, fine (dz = 0.0025, dlogM = 0.025) bins for 0.03 <= z < 0.1, and the
production (log10 M, z) histogram (bin centres) for z >= 0.1.  Profiles: (a) production cache (natural-spline replica,
checked against the Julia interpolator), (b) direct numpy chord integral.  Full-sky Legendre transform:
tau_ell = 2 pi int tau(theta) P_ell(cos theta) sin theta dtheta;  C_ell = (1/4pi) sum (v/c)^2 tau_ell^2."""
import sys, time
sys.path.insert(0, '.')
import numpy as np, h5py
from scipy.special import eval_legendre
from scipy.interpolate import CubicSpline
import b16_direct_np as b

b._GL_U, b._GL_W = np.polynomial.legendre.leggauss(100)
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/verify_B1/'
CACHE = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/caches/tau_b16_sphere4p0_chordmean_interpolator.jld2'
HIST = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halo_mass_redshift_histogram.h5'
CAT = '/home/cbllover/HalfDome/lightcone_100.hdf5'
ELLS = np.array([30, 61, 124, 253, 500, 1000])
NU = 800
U, WU = np.polynomial.legendre.leggauss(NU); U = 0.5 * (U + 1); WU = 0.5 * WU   # u in [0,1]
X = b.XSPH
XG = X * (1 - U**2); DXG = 2 * X * U * WU                                          # x = X(1-u^2), dx = 2Xu du
LG = np.sqrt((X - XG) * (X + XG))
cr = b.CacheReplica(CACHE)
t0 = time.time()

# z -> d_A table (fine), interpolated (cubic) for speed
ztab = np.concatenate([np.linspace(1e-4, 0.2, 4001), np.linspace(0.2005, 4.1, 3000)])
dAtab = b.dA_mpc(ztab)
dA_of = CubicSpline(ztab, dAtab)


def tau_ell(M, z, which, slab=None, row=None):
    """Returns tau_ell (len(ELLS)) for the painted tau inside 4 R200c; which = 'direct' or 'cache'."""
    dA = float(dA_of(z))
    th200 = float(b.theta200(M, z, dA))
    th = XG * th200
    if which == 'direct':
        _, xc, al, be, ga = b.params(M, z)
        tau = b.amplitude(M, z) * b.chord_integral(XG, al, be, xc, ga)
    else:
        if row is None:
            row = CubicSpline(cr.logM, slab, axis=0, bc_type='natural')(np.log10(M))
        g = np.exp(CubicSpline(cr.logth, row, bc_type='natural')(np.log(th)))
        tau = g * 2 * LG
    w = 2 * np.pi * tau * np.sin(th) * th200 * DXG
    ct = np.cos(th)
    return np.array([np.sum(w * eval_legendre(l, ct)) for l in ELLS])


# ------------------------------------------------ low-z halos from the catalogue tail
with h5py.File(CAT, 'r') as f:
    N = f['redshift'].shape[0]
    tail0 = N - 3_000_000
    ztail = f['redshift'][tail0:]
    i0 = tail0 + np.argmax(ztail < 0.1)
    assert np.all(f['redshift'][i0:] < 0.1) and f['redshift'][i0 - 1] >= 0.1
    zc = f['redshift'][i0:].astype(float)
    pos = f['Position'][i0:].astype(float); vel = f['Velocity'][i0:].astype(float)
    mass = f['halo_mass_m200c'][i0:].astype(float) / 0.68
vlos_c = np.sum(pos * vel, axis=1) / np.linalg.norm(pos, axis=1) / b.C_KMS
v2 = vlos_c**2
print(f'z<0.1 halos: {len(zc)}  z range {zc.min():.5f}-{zc.max():.5f}  logM {np.log10(mass).min():.2f}-{np.log10(mass).max():.2f}'
      f'  rms v_los {np.sqrt(v2.mean())*b.C_KMS:.1f} km/s', flush=True)

res = {}
# (1) per-halo z < 0.03
sel = np.where(zc < 0.03)[0]
cl_lo = {'direct': np.zeros(len(ELLS)), 'cache': np.zeros(len(ELLS))}
cl_lo_by_z = {}  # contributions by z slices
zslices = [(0, 0.01), (0.01, 0.0206), (0.0206, 0.03), (0.03, 0.05), (0.05, 0.1)]
for s in zslices:
    cl_lo_by_z[s] = {'direct': np.zeros(len(ELLS)), 'cache': np.zeros(len(ELLS))}
for k, i in enumerate(sel):
    M, z = mass[i], zc[i]
    slab = cr.eval_z(z)
    for which in ('direct', 'cache'):
        t = tau_ell(M, z, which, slab=slab)
        c = v2[i] * t**2 / (4 * np.pi)
        cl_lo[which] += c
        for s in zslices:
            if s[0] <= z < s[1]:
                cl_lo_by_z[s][which] += c
print(f'per-halo z<0.03 done ({len(sel)} halos) {time.time()-t0:.0f}s', flush=True)

# (2) fine bins 0.03 <= z < 0.1
zedges_f = np.arange(0.03, 0.1 + 1e-9, 0.0025); medges = np.arange(12.0, 15.7 + 1e-9, 0.025)
m2 = zc >= 0.03
H2n, _, _ = np.histogram2d(zc[m2], np.log10(mass[m2]), bins=[zedges_f, medges])
H2v, _, _ = np.histogram2d(zc[m2], np.log10(mass[m2]), bins=[zedges_f, medges], weights=v2[m2])
for iz in range(len(zedges_f) - 1):
    z = 0.5 * (zedges_f[iz] + zedges_f[iz + 1])
    slab = cr.eval_z(z)
    ims = np.where(H2n[iz] > 0)[0]
    rows = CubicSpline(cr.logM, slab, axis=0, bc_type='natural')(0.5 * (medges[ims] + medges[ims + 1]))
    for j, im in enumerate(ims):
        M = 10**(0.5 * (medges[im] + medges[im + 1]))
        for which in ('direct', 'cache'):
            t = tau_ell(M, z, which, row=rows[j])
            c = H2v[iz, im] * t**2 / (4 * np.pi)
            cl_lo[which] += c
            for s in zslices:
                if s[0] <= z < s[1]:
                    cl_lo_by_z[s][which] += c
print(f'fine bins 0.03-0.1 done {time.time()-t0:.0f}s', flush=True)

# (3) histogram z >= 0.1 (cache replica; direct agrees to < 0.3% there -- spot-checked below)
with h5py.File(HIST, 'r') as f:
    cnt = f['count'][()]; sv2 = f['sum_vlos_over_c_squared'][()]; le = f['logm_edges'][()]; ze = f['z_edges'][()]
# h5py order (z, logM)
assert cnt.shape == (len(ze) - 1, len(le) - 1)
print('histogram halos z<0.1:', cnt[ze[:-1] < 0.0999].sum(), ' catalogue:', len(zc),
      ' sum v2 hist/cat:', sv2[ze[:-1] < 0.0999].sum() / v2.sum(), flush=True)
cl_hi = np.zeros(len(ELLS)); spot = []
for iz in range(len(ze) - 1):
    if ze[iz] < 0.0999:
        continue
    z = 0.5 * (ze[iz] + ze[iz + 1])
    ims = np.where(cnt[iz] > 0)[0]
    if len(ims) == 0:
        continue
    slab = cr.eval_z(z)
    rows = CubicSpline(cr.logM, slab, axis=0, bc_type='natural')(0.5 * (le[ims] + le[ims + 1]))
    for j, im in enumerate(ims):
        M = 10**(0.5 * (le[im] + le[im + 1]))
        t = tau_ell(M, z, 'cache', row=rows[j])
        cl_hi += sv2[iz, im] * t**2 / (4 * np.pi)
        if iz in (4, 5, 8, 40) and im % 20 == 0:
            spot.append((z, np.log10(M), tau_ell(M, z, 'direct')[0] / t[0] - 1, tau_ell(M, z, 'direct')[3] / t[3] - 1))
print(f'histogram z>=0.1 done {time.time()-t0:.0f}s', flush=True)

T2 = 2.7255e6**2
D = lambda cl: ELLS * (ELLS + 1) * cl / (2 * np.pi) * T2
lines = []
lines.append('ell: ' + ' '.join(f'{l:>9d}' for l in ELLS))
tot_c = cl_hi + cl_lo['cache']; tot_d = cl_hi + cl_lo['direct']
lines.append('D_ell 1h total, cache  [uK^2]: ' + ' '.join(f'{v:9.4g}' for v in D(tot_c)))
lines.append('D_ell 1h total, direct [uK^2]: ' + ' '.join(f'{v:9.4g}' for v in D(tot_d)))
lines.append('cache/direct (total 1h):        ' + ' '.join(f'{v:9.4f}' for v in tot_c / tot_d))
lines.append('direct/cache (total 1h):        ' + ' '.join(f'{v:9.4f}' for v in tot_d / tot_c))
lines.append('fraction of direct total from z<0.1: ' + ' '.join(f'{v:9.4f}' for v in cl_lo['direct'] / tot_d))
for s in zslices:
    c = cl_lo_by_z[s]
    lines.append(f'z in [{s[0]},{s[1]}): (cache - direct)/direct_total = ' + ' '.join(f'{v:+9.4f}' for v in (c['cache'] - c['direct']) / tot_d)
                 + '   share of direct total: ' + ' '.join(f'{v:8.4f}' for v in c['direct'] / tot_d))
lines.append('spot checks z>=0.1 (z, logM, direct/cache-1 at ell=30, 253): ' + '; '.join(f'({a:.4f},{m:.2f},{c:+.1e},{d:+.1e})' for a, m, c, d in spot))
txt = '\n'.join(lines)
print(txt)
open(OUT + 'step2_onehalo_effect.txt', 'w').write(txt + '\n')
np.savez(OUT + 'step2_onehalo_effect.npz', ells=ELLS, cl_hi=cl_hi, cl_lo_cache=cl_lo['cache'], cl_lo_direct=cl_lo['direct'])
