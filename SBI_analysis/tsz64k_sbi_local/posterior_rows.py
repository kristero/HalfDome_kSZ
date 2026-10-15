"""Per-test-row posterior summaries for the mean + width analysis (runs on idark).

For every model, the posterior of each of the 3,000 test rows is sampled with exactly the settings of
sbi_final.row_metrics (1,000 samples per row, torch.manual_seed(1), numpy default_rng(1)), so the posterior means
reproduce the Pearson r stored in the model's metrics (checked and recorded). Per row and parameter it keeps the truth,
the posterior mean and median and the 2.5/16/84/97.5 % quantiles.

Model specs: label=kind:path[:variant]
    ens     a train_ensemble.py folder (features.npz, estimator_seed*.pt, train_seed*.json); metrics.json next to it
    merlin  a merlin_compression.py output folder and its variant (merlin, pca or moped); metrics_<variant>.json

    python posterior_rows.py --out DIR [--threads 8] [--b12] [--b12-samples N] [--truncate] [--ranks] [--seed-means]
                             label=kind:path[:variant] ...
Writes DIR/<label>.npz (truth, mean, median, q025, q16, q84, q975 as (rows, 9) float32; low, high (9,)) and
DIR/<label>.json (the model, the r check, timing); with --b12 also DIR/<label>_b12.npy, B12_SAMPLES posterior samples
(theta units, float32) for the B12 observation x_obs, drawn as sbi_final.cmd_evaluate / merlin_compression evaluate
do (torch.manual_seed(2), default_rng(2)). --truncate (for networks trained without the prior box,
MERLIN_PARAM_SPACE=linear) also stores the share of each row's samples inside the box ('inside') and the summaries of
those samples alone (tmean, tq025, tq16, tmedian, tq84, tq975): the posterior truncated to the prior. --ranks stores
the SBC rank of every truth among its row's samples ('rank', 0..SAMPLES). --seed-means also stores each seed's own
posterior mean of every row ('seed_mean', (seeds, rows, 9); each seed alone, SAMPLES draws, torch.manual_seed(1),
default_rng(1)), for the scatter of r between trainings; with --truncate also 'seed_tmean', the in-box means.
--b12-samples sets the number of B12 draws (default B12_SAMPLES). Existing outputs are skipped.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SAMPLES = 1000                         # posterior draws per test row, as row_metrics
B12_SAMPLES = 20000                    # posterior draws for the B12 observation, as sbi_final.cmd_evaluate


def load(kind, path, variant, threads):
    """The ensemble (train_ensemble.Ensemble interface) and the metrics recorded for it."""
    if kind == 'ens':
        from train_ensemble import Ensemble
        ens = Ensemble(path, threads)
        metrics = json.loads((path / 'metrics.json').read_text())
    elif kind == 'merlin':
        import merlin_compression as MC
        ens = MC.ensemble(path, variant, threads)
        metrics = json.loads((path / f'metrics_{variant}.json').read_text())
    else:
        raise ValueError(f'unknown model kind {kind!r}')
    return ens, metrics


def seed_means(ens, samples, truncate=False):
    """{'seed_mean': (seeds, rows, 9)} posterior means of every test row under each seed of the ensemble alone; with
    truncate also 'seed_tmean', the means of the samples inside the prior box (NaN where none is)."""
    import torch
    nets, means, tmeans = ens.nets, [], []
    try:
        for net in nets:
            ens.nets = [net]
            torch.manual_seed(1)
            s, _, _ = ens.posterior(ens.f['x_held'], samples, rng=np.random.default_rng(1))
            means.append(s.mean(0))
            if truncate:
                inside = np.all((s >= ens.low) & (s <= ens.high), axis=2)
                tmeans.append(np.nanmean(np.where(inside[..., None], s, np.nan), axis=0))
    finally:
        ens.nets = nets
    return dict(seed_mean=np.stack(means), **(dict(seed_tmean=np.stack(tmeans)) if truncate else {}))


def summarize_rows(ens, samples, truncate=False, ranks=False):
    """Posterior summaries of every test row: dict of (rows, 9) arrays (with truncate, also the in-box summaries)."""
    import torch
    x, truth = ens.f['x_held'], ens.f['theta_held']
    torch.manual_seed(1)                                   # the draws of sbi_final.row_metrics
    s, _, _ = ens.posterior(x, samples, rng=np.random.default_rng(1))       # (samples, rows, 9)
    q = np.quantile(s, [.025, .16, .5, .84, .975], axis=0)
    out = dict(truth=truth, mean=s.mean(0), q025=q[0], q16=q[1], median=q[2], q84=q[3], q975=q[4])
    if ranks:
        out['rank'] = np.sum(s < truth[None], axis=0)          # (rows, 9): samples below the truth
    if truncate:
        inside = np.all((s >= ens.low) & (s <= ens.high), axis=2)          # (samples, rows): all nine in the box
        kept = np.where(inside[..., None], s, np.nan)
        tq = np.nanquantile(kept, [.025, .16, .5, .84, .975], axis=0)
        out.update(inside=inside.mean(0), tmean=np.nanmean(kept, axis=0), tq025=tq[0], tq16=tq[1], tmedian=tq[2],
                   tq84=tq[3], tq975=tq[4])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--b12', action='store_true', help='also save B12 posterior samples (<label>_b12.npy)')
    p.add_argument('--truncate', action='store_true', help='also summarize the samples inside the prior box')
    p.add_argument('--ranks', action='store_true', help='also store the SBC rank of each truth')
    p.add_argument('--seed-means', action='store_true', help="also store each seed's posterior means")
    p.add_argument('--b12-samples', type=int, default=B12_SAMPLES, help='number of B12 posterior draws')
    p.add_argument('models', nargs='+')
    args = p.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        label, rest = spec.split('=', 1)
        kind, path, *variant = rest.split(':')
        need_rows = not (args.out / f'{label}.npz').exists()
        need_b12 = args.b12 and not (args.out / f'{label}_b12.npy').exists()
        if not (need_rows or need_b12):
            print(f'{label}: done before', flush=True)
            continue
        t0 = time.time()
        ens, metrics = load(kind, Path(path), variant[0] if variant else None, args.threads)
        if need_b12:
            torch.manual_seed(2)
            samples, _, _ = ens.posterior(ens.f['x_obs'][:1], args.b12_samples, rng=np.random.default_rng(2))
            np.save(args.out / f'{label}_b12.npy', samples[:, 0].astype(np.float32))
            print(f'{label}: {len(samples)} B12 samples', flush=True)
        if not need_rows:
            continue
        rows = summarize_rows(ens, SAMPLES, args.truncate, args.ranks)
        if args.seed_means:
            rows.update(seed_means(ens, SAMPLES, args.truncate))
        r = [float(np.corrcoef(rows['truth'][:, i], rows['mean'][:, i])[0, 1]) for i in range(9)]
        diff = float(np.max(np.abs(np.array(r) - np.array(metrics['pearson_r']))))
        np.savez(args.out / f'{label}.npz', low=ens.low, high=ens.high,
                 **{k: np.asarray(v, dtype=np.float32) for k, v in rows.items()})
        record = dict(label=label, kind=kind, path=str(path), variant=variant[0] if variant else None,
                      rows=int(len(rows['truth'])), samples=SAMPLES, pearson_r=r,
                      pearson_r_metrics=metrics['pearson_r'], max_abs_r_difference_to_metrics=diff,
                      seconds=time.time() - t0)
        (args.out / f'{label}.json').write_text(json.dumps(record, indent=1) + '\n')
        print(f'{label}: {record["rows"]} rows in {record["seconds"]:.0f} s; max |r - metrics r| = {diff:.2e}',
              flush=True)


if __name__ == '__main__':
    main()
