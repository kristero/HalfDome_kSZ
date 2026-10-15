"""Time to build per-halo profile variants and stacks from the saved per-halo shell sums (measurements_owner.npz etc.)
with the study's analysis library, i.e. the cost of a new stacking choice when no particle pass is needed."""
import json, os, sys, time, resource
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012'
sys.path.insert(0, STUDY)
import numpy as np
from analysis import newdata, variants
HERE = os.path.dirname(os.path.abspath(__file__))
t0 = time.perf_counter(); data = newdata.load(); t_load = time.perf_counter() - t0
cases = [('pressure', dict(estimator='geometric', agn='removed', gas_selection='all')),
         ('pressure', dict(estimator='geometric', agn='kept', gas_selection='all')),
         ('electrons', dict(estimator='geometric', agn='removed', gas_selection='all')),
         ('pressure', dict(estimator='Q_V', agn='removed', gas_selection='all')),
         ('pressure', dict(estimator='geometric', agn='removed', gas_selection='partition_owned')),
         ('pressure', dict(estimator='geometric', agn='removed', gas_selection='all', grid='x500'))]
out = {'load_seconds': t_load, 'cases': []}
S = np.load(os.path.join(STUDY, 'selected_halos.npz'))
logm = np.log10(S['m200c_msun']); w = 1 / S['inclusion_probability']
for field, kw in cases:
    t0 = time.perf_counter()
    prof = variants.profile(data, field, **kw)
    t1 = time.perf_counter()
    vals = prof.values
    # stack in 15 bins of 0.2 dex with inclusion weights
    for lo in np.arange(12.5, 15.5, 0.2):
        m = (logm >= lo) & (logm < lo + 0.2)
        v = vals[m]; ww = w[m]
        good = np.isfinite(v)
        np.nansum(np.where(good, v, 0) * ww[:, None], axis=0) / np.maximum((good * ww[:, None]).sum(axis=0), 1e-300)
    t2 = time.perf_counter()
    out['cases'].append({'field': field, **{k: str(v) for k, v in kw.items()}, 'profile_seconds': t1 - t0, 'stack_15_bins_seconds': t2 - t1,
                         'shape': list(vals.shape)})
out['peak_rss_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
json.dump(out, open(os.path.join(HERE, 'bench_restack.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
