"""Inputs of the 'raw' variant of merlin_compression.py: no binning and no compression (runs on idark).

All 7,900 multipoles (ell = 80..7979) of the production split cross spectrum of every row, each transformed as the
Merlin PCA inputs are (merlin_compression.fit_asinh): centre m_ell = training median, scale c_ell = 1.4826 x MAD of
(noisy - clean) over the training rows (the noise level at that multipole), z = arcsinh((x - m) / c), then the
z-score with the training mean and sd of z. The transform is fitted on the first FIT_ROWS rows of the training pool
and then frozen, so a row's features do not change when more rows are added for larger training sets.

Reads (idark)
  ~/tsz64k_beamfix_20260930/upload/stageKK/bNNN/   relay.py uploads: rows.npy, noisy/<row>.npy (production C_ell,
                                                   ell 0..7979, checksummed), clean_dl_unbinned.npy (D_ell 80..7979)
  results_final/split.npz, data/rows_64440/{row_id,theta,noisy_dl}.npy, meta.json     split, parameters, check
  the B12 observation (analyze_64k.TEST256 Battaglia12.npz) and its 21 SO noise draws (mocks/pass1_B12/B12)
Writes results_merlin_unbinned_asinh_p30/features_raw.npz (the layout of features_merlin.npz; the pool is the
longest available prefix of the training order) and raw_features.json (rows, checks, transform summary).

    python raw_features.py [--fit-rows 32000]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
UPLOAD = Path.home() / 'tsz64k_beamfix_20260930' / 'upload'
OUT = HERE / 'results_merlin_unbinned_asinh_p30'
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
MAD_TO_SD = 1.4826


def rebin40(dl):
    w = 2 * ELL + 1
    return np.stack([np.sum(dl[..., a - 80:b - 80] * w[a - 80:b - 80], axis=-1) / np.sum(w[a - 80:b - 80])
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=-1)


def gather():
    """{design row: (noisy D_ell (7900,), clean D_ell (7900,))} of every uploaded batch."""
    rows = {}
    for batch in sorted(UPLOAD.glob('stage*/b*')):
        if not (batch / 'UPLOADED').exists():
            continue
        batch_rows = np.load(batch / 'rows.npy')
        clean = np.load(batch / 'clean_dl_unbinned.npy')
        for i, r in enumerate(batch_rows.tolist()):
            cl = np.load(batch / 'noisy' / f'{r:05d}.npy')
            assert cl.shape == (7980,) and np.isfinite(cl).all()
            rows[r] = (cl[ELL] * DL, clean[i])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fit-rows', type=int, default=32000)
    args = p.parse_args()
    import analyze_64k as A
    t0 = time.time()
    split = np.load(HERE / 'results_final' / 'split.npz')
    order, test = split['order'], split['test']
    row_id = np.load(HERE / 'data' / 'rows_64440' / 'row_id.npy')
    theta = np.load(HERE / 'data' / 'rows_64440' / 'theta.npy')
    noisy40 = np.load(HERE / 'data' / 'rows_64440' / 'noisy_dl.npy')
    meta = json.loads((HERE / 'data' / 'rows_64440' / 'meta.json').read_text())
    available = gather()
    assert all(int(row_id[p]) in available for p in test), 'test rows missing from the uploads'
    n_pool = 0                                            # the longest available prefix of the training order
    while n_pool < len(order) and int(row_id[order[n_pool]]) in available:
        n_pool += 1
    assert n_pool >= args.fit_rows, (n_pool, args.fit_rows)
    pool = order[:n_pool]

    def stack(positions, which):
        return np.stack([available[int(row_id[p])][which] for p in positions])
    x_pool, x_held = stack(pool, 0), stack(test, 0)
    # the uploads are the spectra behind the dataset's band powers
    check = float(np.max(np.abs(rebin40(np.r_[x_pool[:2000], x_held]) - noisy40[np.r_[pool[:2000], test]])
                         / (np.abs(noisy40[np.r_[pool[:2000], test]]) + 1e-30)))
    assert check < 1e-9, check
    fit = slice(0, args.fit_rows)
    clean_fit = stack(pool[fit], 1)
    center = np.median(x_pool[fit], axis=0)
    residual = x_pool[fit] - clean_fit
    scale = MAD_TO_SD * np.median(np.abs(residual - np.median(residual, axis=0)), axis=0)
    assert np.all(scale > 0)
    z_fit = np.arcsinh((x_pool[fit] - center) / scale)
    z_mean, z_sd = z_fit.mean(0), z_fit.std(0)

    def transform(x):
        return ((np.arcsinh((np.asarray(x, dtype=np.float64) - center) / scale) - z_mean) / z_sd).astype(np.float32)
    obs = np.load(A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations' / 'Battaglia12.npz')
    draws = np.stack([np.load(q)[ELL] * DL for q in sorted((HERE / 'mocks' / 'pass1_B12' / 'B12').glob('noisy_cross_*.npy'))])
    n = x_pool.shape[1]
    np.savez(OUT / 'features_raw.npz', x_pool=transform(x_pool), x_held=transform(x_held),
             x_obs=transform(obs['noisy_dl_unbinned'][None]), x_draws=transform(draws),
             theta_pool=theta[pool], theta_held=theta[test], pool=pool, held=test,
             low=np.array(meta['lower']), high=np.array(meta['upper']), s_mean=np.zeros(n), s_sd=np.ones(n),
             s_center=center, s_scale=scale, z_mean=z_mean, z_sd=z_sd, fit_rows=args.fit_rows,
             input_transform='asinh')
    report = dict(pool_rows=int(n_pool), test_rows=int(len(test)), fit_rows=args.fit_rows,
                  noisy40_check_max_rel=check, scale_range=[float(scale.min()), float(scale.max())],
                  pool_z_abs_max=float(np.abs(transform(x_pool[fit])).max()), seconds=time.time() - t0,
                  source=str(UPLOAD), transform='z = (arcsinh((x - median) / (1.4826 MAD(noisy - clean))) - mean) / sd')
    (OUT / 'raw_features.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
