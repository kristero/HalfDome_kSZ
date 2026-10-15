"""Collect the finished rows of a partial (still running) tSZ_64k production into the dataset layout.

Inputs are minimal because the link to this machine is slow: the collaborator's batch_plan.json
(every row's parameters, seeds, cache grid) and, per finished row, masked_clean_cl.npy and
masked_noisy_cross_cl.npy (Globus verifies their checksums in transit). The plan is checked
against the frozen design (SHA-256 and every row's theta and seeds). Only complete batches
whose production log says "ran"/"done" are used. Output: rows_partial/ with the files of
pipeline/manage.py collect. The rows are NOT a design prefix: analyse with a random split.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent / 'tSZ_64k_flat_prior_SO_baseline_deproj0'
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
NPY_BYTES = 63920                      # 7980 float64 + 80-byte header


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def rebin40(values):
    edges = np.r_[np.arange(80, 7881, 200), 7980]
    return np.stack([np.average(values[:, (ELL >= a) & (ELL < b)], axis=1, weights=2 * ELL[(ELL >= a) & (ELL < b)] + 1)
                     for a, b in zip(edges[:-1], edges[1:])], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--delivery', type=Path, default=HERE / 'data' / 'delivery')
    p.add_argument('--out', type=Path, default=HERE / 'data' / 'rows_partial')
    args = p.parse_args()
    design = json.loads((BUNDLE / 'design' / 'design_manifest.json').read_text())
    design_theta = np.load(BUNDLE / 'design' / 'theta_design_65536.npy')
    design_seeds = np.load(BUNDLE / 'design' / 'noise_seeds_65536.npy')
    plan = json.loads((args.delivery / 'batch_plan.json').read_text())
    for name, digest in plan['design_sha256'].items():
        assert digest == sha256(BUNDLE / 'design' / name) == design['sha256'][name], f'plan design hash mismatch: {name}'
    assert plan['rows'] == 65536 and plan['batch_size'] == 4 and not plan['deferred_rows']
    finished = set()
    for f in glob.glob(str(args.delivery / 'logs' / 'tsz64k_prod_*.out')):
        finished.update(int(m.group(1)) for m in re.finditer(r'^batch (\d+): (ran|done) \(', open(f).read(), re.M))
    rows, grids, clean, noisy = [], [], [], []
    missing = 0
    for b in sorted(finished):
        batch = plan['batches'][b]
        assert batch['id'] == b
        folder = args.delivery / 'batches' / f'{b:05d}'
        paths = [(folder / c['label'] / 'masked_clean_cl.npy', folder / c['label'] / 'masked_noisy_cross_cl.npy') for c in batch['cases']]
        if not all(x.exists() and x.stat().st_size == NPY_BYTES and y.exists() and y.stat().st_size == NPY_BYTES for x, y in paths):
            missing += 1; continue
        for case, (cpath, npath) in zip(batch['cases'], paths):
            row = int(case['row'])
            assert np.array_equal(case['theta'], design_theta[row]) and case['seeds'] == design_seeds[row].tolist(), f'row {row} differs from design'
            c, n = np.load(cpath), np.load(npath)
            assert c.shape == n.shape == (7980,) and np.isfinite(c).all() and np.isfinite(n).all() and (c[ELL] > 0).all()
            rows.append(row); grids.append('x'.join(map(str, case['nodes'])))
            clean.append(c[ELL] * DL); noisy.append(n[ELL] * DL)
    order = np.argsort(rows)
    rows = np.asarray(rows)[order]; clean = np.asarray(clean)[order]; noisy = np.asarray(noisy)[order]
    assert len(np.unique(rows)) == len(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / 'clean_dl_unbinned.npy', clean); np.save(args.out / 'noisy_dl_unbinned.npy', noisy)
    np.save(args.out / 'clean_dl.npy', rebin40(clean)); np.save(args.out / 'noisy_dl.npy', rebin40(noisy))
    np.save(args.out / 'theta.npy', design_theta[rows]); np.save(args.out / 'noise_seeds.npy', design_seeds[rows])
    np.save(args.out / 'row_id.npy', rows.astype(np.int64)); np.save(args.out / 'ell_unbinned.npy', ELL)
    meta = dict(rows=int(len(rows)), partial=True, finished_batches_in_logs=len(finished), finished_batches_not_yet_copied=missing,
                source='tsz_64k_flat_prior_SO_baseline_deproj0_20260926 (collaborator run, in progress)',
                cache_grids={g: grids.count(g) for g in sorted(set(grids))},
                parameter_order=design['parameter_order'], lower=design['lower'], upper=design['upper'],
                units='D_ell = ell(ell+1)C_ell/2pi of Compton-y; ell = 80..7979 unbinned', binning='40 bins of 200 multipoles, weights 2l+1',
                unique_noise_seeds=int(len(np.unique(design_seeds[rows].ravel()))), row_id_min=int(rows.min()), row_id_max=int(rows.max()))
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
