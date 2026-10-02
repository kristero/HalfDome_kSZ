"""(2c) C_ell normalisation: random Gaussian blobs (amplitude +-v) on a HEALPix map vs C_ell = (1/4pi) sum v^2 a_ell^2 pixwin^2."""
import numpy as np, healpy as hp
rng = np.random.default_rng(1)
nside = 1024; N = 3000; s = 8.0 / 60 * np.pi / 180
v = rng.normal(0, 1, N)
cost = rng.uniform(-1, 1, N); phi = rng.uniform(0, 2 * np.pi, N)
m = np.zeros(hp.nside2npix(nside))
vec = np.array([np.sqrt(1 - cost**2) * np.cos(phi), np.sqrt(1 - cost**2) * np.sin(phi), cost]).T
for i in range(N):
    ip = hp.query_disc(nside, vec[i], 6 * s)
    pv = np.array(hp.pix2vec(nside, ip)).T
    th = np.arccos(np.clip(pv @ vec[i], -1, 1))
    m[ip] += v[i] * np.exp(-th**2 / (2 * s**2))
cl = hp.anafast(m, lmax=2000)
ell = np.arange(2001)
pred = np.sum(v**2) / (4 * np.pi) * (2 * np.pi * s**2 * np.exp(-ell**2 * s**2 / 2))**2 * hp.pixwin(nside, lmax=2000)**2
for lo, hi in ((50, 150), (150, 400), (400, 800), (800, 1200)):
    k = slice(lo, hi)
    print("ell %d-%d  measured/predicted = %.4f" % (lo, hi, cl[k].sum() / pred[k].sum()))
