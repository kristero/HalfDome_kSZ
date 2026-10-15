"""Model (NFW) mapping of SOAP SO radii to x = R_Delta/R200c at z = 0 and z = 2 (FLAMINGO cosmology), checked at
z = 0 against the measured study medians. Output ../nfw_aperture_mapping.json."""
import json
import numpy as np
from scipy.optimize import brentq
OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
STUDY = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/'
OM, OL = 0.30461099999999997, 0.693922          # SOAP Cosmology attributes (0077)


def om_z(z):
    e2 = OM * (1 + z)**3 + OL + (1 - OM - OL) * (1 + z)**2
    return OM * (1 + z)**3 / e2, e2


def deltas(z):
    omz, e2 = om_z(z)
    xx = omz - 1
    return {'2500_crit': 2500, '1000_crit': 1000, '500_crit': 500, '200_crit': 200, '100_crit': 100,
            'BN98': 18 * np.pi**2 + 82 * xx - 39 * xx**2, '200_mean': 200 * omz, '50_crit': 50}


def m(y):
    return np.log1p(y) - y / (1 + y)


def x_of_delta(delta, c):
    return brentq(lambda x: 200 * m(c * x) / m(c) / x**3 - delta, 1e-3, 50)


H = np.load(STUDY + 'selected_halos.npz')
audit = json.load(open(OUT + 'soap_audit_part1.json'))
res = {'note': 'NFW model with c200c as given; 5xR_500_crit = 5 x(500c). BN98 Delta_c = 18 pi^2 + 82 x - 39 x^2, x = Omega_m(z) - 1.'}
for z in (0.0, 2.0):
    d = deltas(z)
    res[f'z{z:g}'] = {'Delta_c': {k: float(v) for k, v in d.items()}, 'Omega_m_z': float(om_z(z)[0]),
                      'R200c_scale_vs_z0_at_fixed_M200c': float(om_z(z)[1]**(-1 / 3)), 'x': {}}
    for c in (3.0, 4.0, 5.0, 7.0):
        xs = {k: x_of_delta(v, c) for k, v in d.items()}
        xs['5xR_500_crit'] = 5 * xs['500_crit']
        res[f'z{z:g}']['x'][f'c{c:g}'] = {k: round(v, 4) for k, v in xs.items()}
# z = 0 check: measured median x per bin vs NFW with the bin's median SOAP c200c
check = {}
for ap in ('2500_crit', '1000_crit', '500_crit', '100_crit', 'BN98', '200_mean', '50_crit', '5xR_500_crit'):
    rows = []
    for row in audit['aperture_x_by_bin']['SO/' + ap]:
        lo, hi = row['bin']
        sel = (np.log10(H['m200c_msun']) >= lo) & (np.log10(H['m200c_msun']) < hi)
        c = float(np.median(H['concentration_200c'][sel]))
        dz = deltas(0.0)
        xm = 5 * x_of_delta(dz['500_crit'], c) if ap == '5xR_500_crit' else x_of_delta(dz[ap], c)
        rows.append({'bin': [lo, hi], 'median_c200c': round(c, 3), 'measured_median_x': round(row['median'], 4), 'nfw_x': round(xm, 4)})
    check[ap] = rows
res['z0_check_against_measured'] = check
json.dump(res, open(OUT + 'nfw_aperture_mapping.json', 'w'), indent=1)
print(json.dumps(res['z0'], indent=0)[:1500]); print(json.dumps(res['z2'], indent=0)[:1500])
for ap, rows in check.items():
    print(ap, [(r['bin'][0], r['median_c200c'], r['measured_median_x'], r['nfw_x']) for r in rows[::3]])
