"""Particle-halo entries (gas particles within 4 R200c, any owner/phase) per halo from measurements_owner.npz,
extrapolated to complete central populations with the stored inclusion weights."""
import json, os, zipfile
import numpy as np
from numpy.lib import format as npf
HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012'
schema = json.load(open(os.path.join(STUDY, 'schema.json')))
z = np.load(os.path.join(STUDY, 'measurements_owner.npz'), mmap_mode='r')
own = z['owner200']                       # [n, 17, 3, 16, 67] f8 (2.0 GB in RAM)
x200 = z['x200']
cnt = own[:, :, :, 0, :]                  # count quantity
del own
cum = cnt.sum(axis=(1, 2)).cumsum(axis=1)  # [n, 67] cumulative count to each X200 edge
S = np.load(os.path.join(STUDY, 'selected_halos.npz'))
w = 1.0 / S['inclusion_probability']
logm = np.log10(S['m200c_msun'])
edges = np.arange(12.5, 15.51, 0.2)
radii = {'1R': 1.0, '2R': 2.0, '3R': 3.0, '4R': 4.0}
idx = {k: int(np.argmin(np.abs(x200[1:] - r))) for k, r in radii.items()}
out = {'x200_edges_used': {k: float(x200[1:][i]) for k, i in idx.items()}, 'bins': []}
tot = {k: 0.0 for k in radii}
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (logm >= lo) & (logm < hi)
    row = {'lo': round(lo, 1), 'hi': round(hi, 1), 'n_sample': int(m.sum()), 'n_population_weighted': float(w[m].sum())}
    for k, i in idx.items():
        row[f'mean_entries_{k}'] = float(np.average(cum[m, i], weights=w[m]))
        row[f'population_entries_{k}'] = float((cum[m, i] * w[m]).sum())
        tot[k] += row[f'population_entries_{k}']
    row['sample_entries_4R'] = float(cum[m, idx['4R']].sum())
    out['bins'].append(row)
out['population_entries_total'] = tot
out['sample_entries_4R_total'] = float(cum[:, idx['4R']].sum())
for thr in (12.5, 13.0, 13.5, 14.0):
    m = logm >= thr
    out[f'population_entries_4R_above_{thr}'] = float((cum[m, idx['4R']] * w[m]).sum())
    out[f'population_haloes_above_{thr}'] = float(w[m].sum())
json.dump(out, open(os.path.join(HERE, 'entries_estimate.json'), 'w'), indent=1)
for r in out['bins']:
    print(r['lo'], r['hi'], r['n_sample'], round(r['n_population_weighted']), round(r['mean_entries_4R']), f"{r['population_entries_4R']:.3e}")
print('sample entries 4R', f"{out['sample_entries_4R_total']:.3e}")
for k, v in out.items():
    if k.startswith('population_'):
        print(k, v if not isinstance(v, float) else f'{v:.4e}')
