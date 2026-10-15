"""Round-3 check: objective (a) sigma = max(SD/mean, 5%) on fit-grid strict shells to 4R: where the max sits and the
outward rise (max beyond the minimum / minimum) per bin.  Own numpy (cache read directly)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'cache'))
V = json.load(open(os.path.join(CACHE, 'index.json')))['variants']; tab = dict(np.load(os.path.join(CACHE, 'halo_table.npz')))
fe = np.load(os.path.join(CACHE, 'edges_fit.npy')); me = tab['mass_edges']
c = np.load(os.path.join(CACHE, 'counts_fit.npy')); h = np.load(os.path.join(CACHE, 'hsml_fit.npy'))
for f, key in (('P', 'P|geometric|removed|all|all|fit|P200'), ('E', 'E|geometric|removed|all|all|fit|rhoc_fb')):
    A = np.load(os.path.join(CACHE, V[key]['file'])); out = []
    for k in range(5, 15):
        rows = np.flatnonzero((tab['logm200c'] >= me[k]) & (tab['logm200c'] < me[k + 1])); w = tab['weight'][rows]
        x = fe[:-1]
        with np.errstate(invalid='ignore'):
            ok = ((x[None] >= np.maximum(0.04, 4 * tab['eps_over_r200c'][rows][:, None])) & (c[rows] >= 20) & (x[None] >= 2 * h[rows])).all(0)
        ok &= fe[1:] <= 4 + 1e-9
        F = A[rows]; m = (w[:, None] * F).sum(0) / w.sum(); var = (w[:, None] * (F - m)**2).sum(0) / w.sum()
        sd = np.sqrt(var / (1 - (w**2).sum() / w.sum()**2)); s = np.maximum(sd / m, 0.05)[ok]; xs = x[ok]
        i0 = int(np.argmin(s)); outer = s[i0:].max(); inner = s[:i0 + 1].max()
        out.append((f'{me[k]:.1f}-{me[k+1]:.1f}', round(float(s[i0]), 3), round(float(xs[i0]), 3), round(float(outer / s[i0]), 2),
                    round(float(inner / s[i0]), 2), round(float(s.max() / s[i0]), 2)))
    print(f, 'bin, min, x_min, outward rise (max beyond min / min), inward rise (max inside min / min), max/min')
    for o in out: print('  ', o)
    print(f, 'outward-rise range', min(o[3] for o in out), max(o[3] for o in out), ' max/min range', min(o[5] for o in out), max(o[5] for o in out))
