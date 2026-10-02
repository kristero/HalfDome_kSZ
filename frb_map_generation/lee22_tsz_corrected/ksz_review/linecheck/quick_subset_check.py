#!/usr/bin/env python3
"""Preliminary Julia-vs-python check of the subset run (same metrics as compare.py, one run)."""
import sys, numpy as np, healpy as hp
O = sys.argv[1]; om = 4 * np.pi / hp.nside2npix(1024)
H = np.load(O + '/py/py_subset_halos.npz')
for v in ("b16_phys", "lee22corr_phys", "lee22pipe_phys", "b16_websky", "lee22corr_websky"):
    J = hp.read_map('%s/julia_subset/maps/halfdome_ksz_%s_nside1024.fits' % (O, v), dtype=np.float64, verbose=False)
    P = np.load('%s/py/py_subset_%s.npy' % (O, v))
    d = J - P
    print('%-17s sumJ %.6e sumP %.6e analytic %.6e | J/P-1 %.2e  J/A-1 %.2e | rms(d)/rms(P) %.2e  max|d|/max|P| %.2e  sumJ2/sumP2-1 %.2e  noise %.2e'
          % (v, J.sum() * om, P.sum() * om, H['an_' + v].sum(), J.sum() / P.sum() - 1, J.sum() * om / H['an_' + v].sum() - 1,
             np.sqrt(np.mean(d**2) / np.mean(P**2)), np.abs(d).max() / np.abs(P).max(), np.sum(J**2) / np.sum(P**2) - 1,
             np.sqrt(np.sum((H['ps_' + v] - H['an_' + v])**2)) / abs(H['an_' + v].sum())))
