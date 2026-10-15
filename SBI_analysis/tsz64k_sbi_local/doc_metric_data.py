"""Per-row metric data of one or more trained ensembles for the pipeline explainer figures (runs on idark).

For each model it repeats sbi_final.row_metrics exactly (1,000 draws per test row, torch.manual_seed(1),
default_rng(1)) and keeps what the stored metrics summarize:
  lp (rows,)              log posterior density of the truth, theta units (mean = metrics mean_log_prob_theta)
  lp_seeds (seeds, rows)  the same under each seed alone
  ecp, alpha (101,)       the TARP expected-coverage curve (tarp68 / tarp95 = its values at 0.68 / 0.95)
  rank (rows, 9) int16    SBC rank of each truth among the row's draws
  mean, q025, q16, q84, q975 (rows, 9)   posterior summaries; truth (rows, 9); low, high (9,)
  seed_mean (seeds, rows, 9)             each seed's own posterior means (1,000 draws, the same seeding)
  examples (draws, k, 9), example_index (k,)   the draws of a few test rows (EXAMPLES)
  features (seeds, rows, F)              Merlin only: the learned head outputs of each seed for the test rows
and with --b12-samples N also N draws of the B12 observation (torch.manual_seed(2), default_rng(2)).
Hard check: the mean of lp equals the stored metrics; the TARP differences are recorded in <label>.json.

    python doc_metric_data.py --out DIR [--threads 16] [--b12-samples N] label=kind:path[:variant] ...
kinds: ens (a train_ensemble.py folder with metrics.json) or merlin (merlin_compression.py folder and variant).
Writes DIR/<label>.npz and DIR/<label>[_b12].npy.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import posterior_rows as PR

EXAMPLES = [1, 2, 9]                   # test rows 65 (3.4 x B12), 72 (0.04 x B12) and 167 (180 x B12)


def per_seed_log_prob(ens, truth, x):
    """(seeds, rows) log density of the truth under each seed alone, theta units."""
    from train_ensemble import to_u, log_jacobian_theta_per_u
    u_t = to_u(truth, ens.low, ens.high)
    nets, out = ens.nets, []
    try:
        for net in nets:
            ens.nets = [net]
            out.append(ens.log_prob_u(u_t[None], x)[0] - log_jacobian_theta_per_u(u_t, ens.low, ens.high))
    finally:
        ens.nets = nets
    return np.stack(out)


def head_features(ens, x):
    """(seeds, rows, F) Merlin head outputs of each seed, or None for a network without an embedding head."""
    import torch
    out = []
    for net in ens.nets:
        head = getattr(net, 'embedding_net', None) or getattr(net, '_embedding_net', None)
        if head is None or isinstance(head, torch.nn.Identity):
            return None
        head.eval()
        with torch.no_grad():
            out.append(head(torch.as_tensor(x, dtype=torch.float32)).numpy())
    return np.stack(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--b12-samples', type=int, default=0)
    p.add_argument('models', nargs='+')
    args = p.parse_args()
    import torch
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references
    from sbi.utils.metrics import l2
    from train_ensemble import to_u, log_jacobian_theta_per_u
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.models:
        label, rest = spec.split('=', 1)
        kind, path, *variant = rest.split(':')
        if (args.out / f'{label}.npz').exists():
            print(f'{label}: done before', flush=True)
            continue
        t0 = time.time()
        ens, metrics = PR.load(kind, Path(path), variant[0] if variant else None, args.threads)
        x, truth = ens.f['x_held'], ens.f['theta_held']
        # sbi_final.row_metrics, step by step
        torch.manual_seed(1)
        s, _, _ = ens.posterior(x, PR.SAMPLES, rng=np.random.default_rng(1))         # (draws, rows, 9)
        u_t = to_u(truth, ens.low, ens.high)
        lp = ens.log_prob_u(u_t[None], x)[0] - log_jacobian_theta_per_u(u_t, ens.low, ens.high)
        torch.manual_seed(1)
        truth_t = torch.tensor(truth, dtype=torch.float32)
        references = get_tarp_references(truth_t)
        ecp, alpha = _run_tarp(torch.tensor(s, dtype=torch.float32), truth_t, references, l2, None, True)
        ecp, alpha = np.ravel(ecp.numpy()), np.ravel(alpha.numpy())
        c68, c95 = np.interp([.68, .95], alpha, ecp)
        checks = dict(lp=float(abs(lp.mean() - metrics['mean_log_prob_theta'])),
                      tarp68=float(abs(c68 - metrics['tarp68'])), tarp95=float(abs(c95 - metrics['tarp95'])))
        assert checks['lp'] < 1e-4, (label, checks)          # TARP is only reported (it depends on the draws)
        q = np.quantile(s, [.025, .16, .84, .975], axis=0)
        out = dict(lp=lp, lp_seeds=per_seed_log_prob(ens, truth, x), ecp=ecp, alpha=alpha,
                   rank=np.sum(s < truth[None], axis=0).astype(np.int16), mean=s.mean(0), q025=q[0], q16=q[1],
                   q84=q[2], q975=q[3], truth=truth, low=ens.low, high=ens.high,
                   seed_mean=PR.seed_means(ens, PR.SAMPLES)['seed_mean'],
                   examples=s[:, EXAMPLES].astype(np.float32), example_index=np.array(EXAMPLES))
        features = head_features(ens, x)
        if features is not None:
            out['features'] = features.astype(np.float32)
        np.savez(args.out / f'{label}.npz', **{k: (v.astype(np.float32) if isinstance(v, np.ndarray) and
                                                   v.dtype == np.float64 and v.ndim > 1 else v)
                                                for k, v in out.items()})
        if args.b12_samples:
            torch.manual_seed(2)
            b12, _, _ = ens.posterior(ens.f['x_obs'][:1], args.b12_samples, rng=np.random.default_rng(2))
            np.save(args.out / f'{label}_b12.npy', b12[:, 0].astype(np.float32))
        (args.out / f'{label}.json').write_text(json.dumps(dict(label=label, spec=spec, checks=checks,
                                                                seconds=time.time() - t0), indent=1) + '\n')
        print(f'{label}: lp {lp.mean():.3f} (stored {metrics["mean_log_prob_theta"]:.3f}), checks {checks}, '
              f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
