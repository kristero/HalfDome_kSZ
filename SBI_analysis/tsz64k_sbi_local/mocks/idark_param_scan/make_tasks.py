"""One-parameter scans for the degeneracy figures: each of the nine parameters at 10, 35, 65 and 90 % of its prior
range, the other eight at the Battaglia+12 values, painted with the frozen engine (paint_mocks.jl) with the B12
observation's noise seeds. One job (folder p<k>) per parameter, four cases each.

    python make_tasks.py        writes p0..p8/task.toml and cases.json
"""
import json
from pathlib import Path

NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
LOW = [1.0, 0.025, 2.8, -0.6, -1.0, -0.2, -6.0, -1.5, -0.5]
HIGH = [60.0, 4.0, 16.0, 1.5, 0.4, 0.4, 0.5, 3.0, 2.0]
B12 = [18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]
QUANTILES = (0.10, 0.35, 0.65, 0.90)
OUT = '/home/kristero10/tsz64k_param_scan_20261001/out'
HERE = Path(__file__).resolve().parent
cases = {}
for k, name in enumerate(NAMES):
    lines = [f'out = "{OUT}"', '']
    for q in QUANTILES:
        theta = list(B12)
        theta[k] = LOW[k] + q * (HIGH[k] - LOW[k])
        label = f'{name}_q{int(round(100 * q)):02d}'
        cases[label] = dict(parameter=name, index=k, quantile=q, value=theta[k], theta=theta)
        lines += ['[[cases]]', f'label = "{label}"', f'theta = {json.dumps(theta)}', 'seeds = [[39001001, 39001002]]', '']
    (HERE / f'p{k}').mkdir(exist_ok=True)
    (HERE / f'p{k}' / 'task.toml').write_text('\n'.join(lines))
(HERE / 'cases.json').write_text(json.dumps(dict(reference='B12 = mocks/pass1_B12/B12', quantiles=QUANTILES,
                                                 names=NAMES, low=LOW, high=HIGH, b12=B12, cases=cases), indent=1) + '\n')
print(len(cases), 'cases')
