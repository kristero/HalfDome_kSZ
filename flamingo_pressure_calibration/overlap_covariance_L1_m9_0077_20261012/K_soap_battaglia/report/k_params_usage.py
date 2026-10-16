"""Evaluate any Test K law (k_params.json) with k_fit_core; works for every parametrisation incl. brk_all."""
import json, numpy as np, sys
sys.path.insert(0, '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/K_soap_battaglia')
import k_fit_core as K
P = json.load(open('/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/K_soap_battaglia/results/k_params.json'))
def law(key):
    f = P['fits'][key]; form = 'B12' if f['field'] in ('P', 'Pagn') else 'B16'
    if not f.get('fit_status', {}).get('usable', True):
        print('warning:', key, f['fit_status']['note'], file=sys.stderr)   # e.g. G|R1|free: not converged, degenerate
    L = K.Law(form, f['parametrization']); return L, np.array(f['theta'])
L, th = law('P|R3|orig')
x = np.geomspace(0.05, 3, 50)
print(K.law_profile(L, th, x, 1e14, 0.5))         # P_e/P200 at M200c = 1e14 Msun, z = 0.5
p = {q: float(np.ravel(v)[0]) for q, v in K.law_profile_params(L, th, 1e14, 0.5).items()}
print(4 * np.pi * K.cumulative(1.0, L.form, *(p[q] for q in K.QNAMES)))   # Ytilde(<R200c)
