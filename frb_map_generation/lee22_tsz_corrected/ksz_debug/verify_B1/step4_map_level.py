"""B1 step 4: map-level effect of the low-z cache error.  Paint the error map
dMap = -sum_{z<0.05} (v_los/c) [tau_cache(theta) - tau_direct(theta)] at NSIDE 512 (per halo, actual z, M, v),
subtract it from the production B16 map (NSIDE 4096 degraded to 512) and compare power spectra.
Includes the two-halo/cross terms that the one-halo estimate (step 2) omits."""
import sys, time
sys.path.insert(0, '.')
import numpy as np, h5py, healpy as hp
from scipy.interpolate import CubicSpline
import b16_direct_np as b

b._GL_U, b._GL_W = np.polynomial.legendre.leggauss(100)
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/verify_B1/'
CACHE = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/caches/tau_b16_sphere4p0_chordmean_interpolator.jld2'
CAT = '/home/cbllover/HalfDome/lightcone_100.hdf5'
MAP = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_20261002/maps/halfdome_ksz_b16_sphere4p0_nside4096.fits'
NS = 512; ZCUT = 0.05; LMAX = 700
X = b.XSPH
NX = 1200
xg = np.concatenate([np.geomspace(1e-5, 0.5, 400, endpoint=False), np.linspace(0.5, X, NX - 400, endpoint=False)])
Lg = np.sqrt((X - xg) * (X + xg))
cr = b.CacheReplica(CACHE)
t0 = time.time()

with h5py.File(CAT, 'r') as f:
    N = f['redshift'].shape[0]
    tail0 = N - 3_000_000
    zt = f['redshift'][tail0:]
    i0 = tail0 + np.argmax(zt < ZCUT)
    zc = f['redshift'][i0:].astype(float)
    pos = f['Position'][i0:].astype(float); vel = f['Velocity'][i0:].astype(float)
    mass = f['halo_mass_m200c'][i0:].astype(float) / 0.68
assert zc.max() < ZCUT
r = np.linalg.norm(pos, axis=1); uvec = pos / r[:, None]
vlos_c = np.sum(pos * vel, axis=1) / r / b.C_KMS
print(f'{len(zc)} halos z<{ZCUT}', flush=True)

dmap = np.zeros(hp.nside2npix(NS))
direct_lowz = np.zeros(hp.nside2npix(NS))
for i in range(len(zc)):
    M, z = mass[i], zc[i]
    th200 = float(b.theta200(M, z))
    _, xc, al, be, ga = b.params(M, z)
    tau_d = b.amplitude(M, z) * b.chord_integral(xg, al, be, xc, ga)
    g_c = cr.eval(xg * th200, M, z)
    tau_c = g_c * 2 * Lg
    thmax = X * th200
    pix = hp.query_disc(NS, uvec[i], thmax, inclusive=False)
    if len(pix) == 0:
        continue
    pv = np.array(hp.pix2vec(NS, pix)).T
    th = np.arccos(np.clip(pv @ uvec[i], -1, 1))
    xpix = th / th200
    ok = xpix < X
    dmap[pix[ok]] -= vlos_c[i] * np.interp(xpix[ok], xg, tau_c - tau_d)
    direct_lowz[pix[ok]] -= vlos_c[i] * np.interp(xpix[ok], xg, tau_d)
print(f'painted error map {time.time()-t0:.0f}s', flush=True)

m4096 = hp.read_map(MAP, dtype=np.float64)
order = 'RING'
with __import__('astropy.io.fits', fromlist=['open']).open(MAP) as hdul:
    order = hdul[1].header.get('ORDERING', 'RING')
mprod = hp.ud_grade(m4096, NS, order_in='RING', order_out='RING')   # read_map returns RING (nest=False)
del m4096
print(f'loaded + degraded production map (ordering {order}) {time.time()-t0:.0f}s', flush=True)

cl_prod = hp.anafast(mprod, lmax=LMAX)
cl_fix = hp.anafast(mprod - dmap, lmax=LMAX)
cl_err = hp.anafast(dmap, lmax=LMAX)
cl_lowz_direct = hp.anafast(direct_lowz, lmax=LMAX)
ell = np.arange(LMAX + 1)
T2 = 2.7255e6**2
bins = [(20, 40), (41, 81), (82, 165), (166, 340), (341, 660)]
lines = [f'production B16 map (degraded to NSIDE {NS}) vs map with the z<{ZCUT} cache error removed',
         'ell bin      D_prod[uK^2]  D_fixed[uK^2]  prod/fixed   D_err[uK^2]  D_lowz_direct[uK^2]']
for lo, hi in bins:
    s = slice(lo, hi + 1)
    w = ell[s] * (ell[s] + 1) / (2 * np.pi) * T2
    dp, df, de, dl = (np.mean(w * c[s]) for c in (cl_prod, cl_fix, cl_err, cl_lowz_direct))
    lines.append(f'{lo:4d}-{hi:<4d}   {dp:12.5g} {df:13.5g} {dp/df:11.4f} {de:13.4g} {dl:15.4g}')
txt = '\n'.join(lines); print(txt)
open(OUT + 'step4_map_level.txt', 'w').write(txt + '\n')
np.savez(OUT + 'step4_map_level.npz', ell=ell, cl_prod=cl_prod, cl_fix=cl_fix, cl_err=cl_err, cl_lowz_direct=cl_lowz_direct)
