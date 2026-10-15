"""Painting tasks for the Fisher forecast at Battaglia+12 (fisher_forecast_b12.py), run on idark with the frozen
tSZ_64k engine (paint_mocks.jl, the copy used for the one-parameter scans; sha256 31671c42...).

  noise/n01..n08   the B12 sky with 50 new SO noise split-seed pairs each (950001/950002 ... 950799/950800): 400
                   independent noise realizations of the B12 observation for the covariance of the 7,900 multipoles
                   (the 21 local draws of mocks/pass1_B12/B12 used seeds 39001001/2 and 910001..910040)
  deriv/d1..d4     B12 +- 2 h for the eight non-amplitude parameters (h = 0.5 % of the prior width, the step of
                   mocks/fisher_b12), noise free: the second step for a step-size check / Richardson extrapolation

    python make_tasks.py ROOT        writes the task.toml files and cases.json under this folder, outputs to ROOT
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = sys.argv[1] if len(sys.argv) > 1 else '/home/kristero10/tsz64k_fisher_b12_20261002'
PLAN = json.loads((HERE.parent / 'mocks' / 'fisher_b12' / 'plan.json').read_text())
NAMES, B12, STEP = PLAN['names'], PLAN['fiducial'], PLAN['step']
NOISE_JOBS, PAIRS_PER_JOB, SEED0 = 8, 50, 950000
DERIV_GROUPS = [(1, 2), (3, 4), (5, 6), (7, 8)]          # parameter indices painted together in one catalogue pass

cases = dict(noise={}, deriv={})
for k in range(1, NOISE_JOBS + 1):
    first = SEED0 + 2 * PAIRS_PER_JOB * (k - 1) + 1
    seeds = [[first + 2 * j, first + 2 * j + 1] for j in range(PAIRS_PER_JOB)]
    name = f'n{k:02d}'
    out = f'{ROOT}/noise_out/{name}'
    (HERE / 'noise' / name).mkdir(parents=True, exist_ok=True)
    (HERE / 'noise' / name / 'task.toml').write_text('\n'.join([
        f'out = "{out}"', '', '[[cases]]', 'label = "B12"', f'theta = {json.dumps(B12)}',
        f'seeds = {json.dumps(seeds)}', '']))
    cases['noise'][name] = dict(out=out, seeds=seeds)
for k, group in enumerate(DERIV_GROUPS, start=1):
    lines = [f'out = "{ROOT}/deriv_out"', '']
    for i in group:
        for sign, tag in ((+1, 'p2'), (-1, 'm2')):
            theta = list(B12)
            theta[i] = B12[i] + sign * 2 * STEP[i]
            label = f'{NAMES[i]}_{tag}'
            lines += ['[[cases]]', f'label = "{label}"', f'theta = {json.dumps(theta)}', '']
            cases['deriv'][label] = dict(index=i, sign=sign, step=2 * STEP[i], theta=theta, job=f'd{k}')
    (HERE / 'deriv' / f'd{k}').mkdir(parents=True, exist_ok=True)
    (HERE / 'deriv' / f'd{k}' / 'task.toml').write_text('\n'.join(lines))
all_seeds = [s for c in cases['noise'].values() for p in c['seeds'] for s in p]
assert len(set(all_seeds)) == len(all_seeds) == 2 * NOISE_JOBS * PAIRS_PER_JOB
assert not set(all_seeds) & ({39001001, 39001002} | set(range(910001, 910041)))
(HERE / 'cases.json').write_text(json.dumps(dict(root=ROOT, fiducial=B12, names=NAMES, step=STEP, **cases), indent=1) + '\n')
print(len(cases['noise']), 'noise jobs,', len(all_seeds) // 2, 'pairs;', len(cases['deriv']), 'derivative cases')
