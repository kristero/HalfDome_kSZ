import numpy as np, healpy as hp, time
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/websky_def/'
t0 = time.time()
m = hp.read_map(OUT + 'websky_ksz_total_v0.0.fits', dtype=np.float32)   # muK
print('read', time.time()-t0, 'rms muK', np.sqrt(np.mean(m.astype(np.float64)**2)), 'mean', m.mean())
cl = hp.anafast(m, lmax=6143, use_pixel_weights=False, iter=0)
print('anafast', time.time()-t0)
np.savez(OUT + 'websky_total_ksz_cl.npz', cl_muK2=cl, note='anafast of public WebSky ksz.fits (v0.0, 27-Jan-2020), NSIDE 4096, no pixel-window correction, iter=0')
