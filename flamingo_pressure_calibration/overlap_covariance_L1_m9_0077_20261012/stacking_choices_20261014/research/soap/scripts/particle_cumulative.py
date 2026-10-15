"""Particle-based cumulative gas quantities from the study's owner sums (read-only inputs).

Output: ../particle_cumulative.npz with, on the X200 (67 shells to 4 R200c) and X500 (40 shells to 6 R500c) grids,
  shells200_inc [n, 3 phases, 4 q, 67]  (sum over all 17 owners = every gas particle in the shell = inclusive)
  shells200_exc [n, 3, 4, 67]           (owner 0 = gas bound to the central subhalo = SOAP 'exclusive')
  and the same for 500; q = (count, mass, y, electrons).
"""
import json, numpy as np
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/'
OUT = STUDY + 'stacking_choices_20261014/research/soap/'
schema = json.load(open(STUDY + 'schema.json'))['names']
Q = ['count', 'mass', 'y', 'electrons']
qi = [schema['quantities'].index(q) for q in Q]
d = np.load(STUDY + 'measurements_owner.npz')
res = {'quantities': np.array(Q), 'phases': np.array(schema['phases'])}
for g in ('200', '500'):
    a = d['owner' + g]                                   # [n, 17, 3, 16, S]
    a = a[:, :, :, qi, :]
    res['shells%s_inc' % g] = a.sum(axis=1)
    res['shells%s_exc' % g] = a[:, 0]
    res['x' + g] = d['x' + g]
    del a
np.savez_compressed(OUT + 'particle_cumulative.npz', **res)
print({k: v.shape for k, v in res.items()})
