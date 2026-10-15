"""SBI on the 65,536-row flat-prior tSZ dataset, on this machine's CPU.

Data: a collected dataset folder of tSZ_64k_flat_prior_SO_baseline_deproj0 (rows_N/ with
clean_dl_unbinned.npy, noisy_dl_unbinned.npy, clean_dl.npy, noisy_dl.npy, theta.npy, meta.json).
Compression, as in the 256-row diagnostic: 'moped_fixed' (the frozen Battaglia+12
finite-difference MOPED matrix, 7900 raw D_ell -> 9 numbers, then a training-only signed
asinh) is primary; 'bins40' (40 band powers, per-bin asinh, z-scored) is the control.
Networks: MAF neural posterior estimators (sbi >= 0.27), one per training size and seed.
Split: training prefixes of the Sobol design (balanced) and a held-out aligned block at the
end of the design. Observations: Battaglia12 (in-family control, truth known) and three
FLAMINGO y-maps (out of family: effective gNFW parameters, not FLAMINGO truth).
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
TEST256 = HERE.parent / 'tsz_diagnostic_256_accelerated_20260922'
NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])


# ----------------------------------------------------------------------------- compression
def fit_transform(method, clean_train, noisy_train):
    """Fit the compression on training rows only. Returns a state dict and metadata."""
    if method == 'moped_fixed':
        context = np.load(TEST256 / 'reference' / 'compression_context.npz')
        matrix = context['Battaglia12_rcond1e-06']
        assert matrix.shape == (7900, 9)
        raw = noisy_train @ matrix
        scale = np.maximum(np.median(np.abs(raw), axis=0), 1e-30)
        y = np.arcsinh(raw / scale)
        state = dict(method=method, matrix=matrix, scale=scale, center=y.mean(0), spread=np.maximum(y.std(0), 1e-8))
        meta = dict(input_features=7900, retained=9,
                    derivative_method='frozen B12 finite differences (256-row test compression_context.npz)',
                    coordinate_transform='linear raw unbinned MOPED, then training-only signed asinh')
        return state, meta
    if method == 'bins40':
        scale = np.maximum(np.median(np.abs(clean_train), axis=0), 1e-30)
        y = np.arcsinh(noisy_train / scale)
        state = dict(method=method, scale=scale, center=y.mean(0), spread=np.maximum(y.std(0), 1e-8))
        return state, dict(input_features=40, retained=40, coordinate_transform='per-bin training-only signed asinh, z-scored')
    raise ValueError(method)


def project(state, noisy, chunk=8192):
    """Apply a fitted compression to (n, features) noisy spectra, in chunks (memory)."""
    noisy = np.asarray(noisy)
    out = np.empty((len(noisy), len(state['scale'])), dtype=np.float32)
    for start in range(0, len(noisy), chunk):
        block = np.asarray(noisy[start:start + chunk], dtype=np.float64)
        raw = block @ state['matrix'] if state['method'] == 'moped_fixed' else block
        out[start:start + chunk] = (np.arcsinh(raw / state['scale']) - state['center']) / state['spread']
    return out


# ----------------------------------------------------------------------------- plotting
def corner(samples, low, high, truth=None, title='', path=None, color='C0'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.1 * d, 2.1 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis('off'); continue
            if i == j:
                ax.hist(samples[:, i], bins=40, range=(low[i], high[i]), color=color, alpha=.8, density=True)
                if truth is not None: ax.axvline(truth[i], color='k', lw=1.2)
                ax.set_yticks([])
            else:
                ax.hist2d(samples[:, j], samples[:, i], bins=40, range=[(low[j], high[j]), (low[i], high[i])], cmap='Blues')
                if truth is not None: ax.plot(truth[j], truth[i], 'k+', ms=10, mew=1.5)
                ax.set_ylim(low[i], high[i])
            ax.set_xlim(low[j], high[j])
            if i == d - 1: ax.set_xlabel(NAMES[j], fontsize=9)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(NAMES[i], fontsize=9)
            elif j > 0: ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if path: fig.savefig(path, dpi=110); plt.close(fig)
    return fig


def summarize(samples, low, high, truth=None):
    q = np.quantile(samples, [.025, .16, .5, .84, .975], axis=0)
    out = dict(mean=samples.mean(0).tolist(), median=q[2].tolist(), q16=q[1].tolist(), q84=q[3].tolist(),
               q025=q[0].tolist(), q975=q[4].tolist(), width68_over_prior=((q[3] - q[1]) / (high - low)).tolist())
    if truth is not None:
        out['truth'] = np.asarray(truth).tolist()
        out['truth_in_68'] = ((truth >= q[1]) & (truth <= q[3])).tolist()
        out['truth_in_95'] = ((truth >= q[0]) & (truth <= q[4])).tolist()
        out['rank_of_truth'] = np.mean(samples < truth, axis=0).tolist()
        out['abs_error_of_mean_over_prior_width'] = (np.abs(samples.mean(0) - truth) / (high - low)).tolist()
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True, help='collected rows_N folder')
    p.add_argument('--observations', type=Path, default=TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--sizes', nargs='+', type=int, default=[8192, 32768, 61440], help='training prefixes (rows 0..N-1)')
    p.add_argument('--heldout-start', type=int, default=61440, help='first row of the held-out block')
    p.add_argument('--heldout-rows', type=int, default=512, help='held-out rows evaluated (from the block start)')
    p.add_argument('--split', choices=['prefix', 'random'], default='prefix',
                   help='prefix: train on design prefixes, hold out the aligned block at --heldout-start; '
                        'random: seeded random split of whatever rows exist (for a partial delivery)')
    p.add_argument('--heldout-fraction', type=float, default=0.05); p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--methods', nargs='+', default=['moped_fixed', 'bins40'])
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2])
    p.add_argument('--hidden', type=int, default=64); p.add_argument('--transforms', type=int, default=5)
    p.add_argument('--batch', type=int, default=256); p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=500); p.add_argument('--stop-after', type=int, default=30)
    p.add_argument('--heldout-samples', type=int, default=1000); p.add_argument('--obs-samples', type=int, default=8000)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--skip-labels', nargs='*', default=['Battaglia12_raw16384_sensitivity', 'Battaglia12_raw4096_sensitivity'])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform
    from sbi.neural_nets import posterior_nn
    torch.set_num_threads(args.threads)

    meta = json.loads((args.dataset / 'meta.json').read_text())
    low, high = np.array(meta['lower']), np.array(meta['upper']); width = high - low
    assert meta['parameter_order'] == NAMES
    theta = np.load(args.dataset / 'theta.npy')
    n = len(theta)
    noisy_unbinned = np.load(args.dataset / 'noisy_dl_unbinned.npy', mmap_mode='r')
    clean_unbinned = np.load(args.dataset / 'clean_dl_unbinned.npy', mmap_mode='r')
    clean40, noisy40 = np.load(args.dataset / 'clean_dl.npy'), np.load(args.dataset / 'noisy_dl.npy')
    assert noisy_unbinned.shape == (n, 7900) and noisy40.shape == (n, 40)
    assert np.all((theta >= low) & (theta <= high))
    if args.split == 'prefix':
        heldout = np.arange(args.heldout_start, min(args.heldout_start + args.heldout_rows, n))
        pool = np.arange(args.heldout_start)
    else:
        perm = np.random.default_rng(args.split_seed).permutation(n)
        k = int(round(args.heldout_fraction * n))
        heldout = np.sort(perm[:k])[:args.heldout_rows]; pool = np.sort(perm[k:])
    sizes = [s for s in args.sizes if s <= len(pool)] or [len(pool)]
    if sizes[-1] < len(pool) and args.split == 'random': sizes.append(len(pool))
    assert len(heldout) > 0, 'no held-out rows'
    prior_rmse = np.sqrt(np.mean(((theta[heldout] - (low + high) / 2) / width) ** 2, axis=0))

    observations = {}
    for path in sorted(args.observations.glob('*.npz')):
        if path.stem in args.skip_labels: continue
        with np.load(path) as o:
            observations[path.stem] = dict(unbinned=o['noisy_dl_unbinned'].copy(), binned=o['noisy_dl'].copy())
    print(f'dataset {args.dataset}: {n} rows; split {args.split}; sizes {sizes}; {len(heldout)} held-out rows; observations {list(observations)}', flush=True)

    prior = BoxUniform(torch.tensor(low, dtype=torch.float32), torch.tensor(high, dtype=torch.float32))
    (args.out / 'run_config.json').write_text(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=1) + '\n')
    np.savez(args.out / 'split.npz', heldout=heldout, pool=pool)
    results_path = args.out / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    for size in sizes:
        train = pool[:size]
        for method in args.methods:
            unbinned = method == 'moped_fixed'
            state, cmeta = fit_transform(method, clean_unbinned[train] if unbinned else clean40[train],
                                         noisy_unbinned[train] if unbinned else noisy40[train])
            x_train = project(state, noisy_unbinned[train] if unbinned else noisy40[train])
            x_held = project(state, noisy_unbinned[heldout] if unbinned else noisy40[heldout])
            x_obs = {k: project(state, v['unbinned' if unbinned else 'binned'][None, :])[0] for k, v in observations.items()}
            np.savez(args.out / f'transform_{method}_{size}.npz', **{k: v for k, v in state.items() if k != 'method'})
            for seed in args.seeds:
                tag = f'{method}_{size}_{seed}'
                if any(r['tag'] == tag for r in results):
                    print('done already:', tag); continue
                torch.manual_seed(seed); np.random.seed(seed)
                started = time.monotonic()
                builder = posterior_nn(model='maf', hidden_features=args.hidden, num_transforms=args.transforms,
                                       z_score_x='none', z_score_theta='independent')
                inference = NPE(prior=prior, density_estimator=builder, device='cpu', show_progress_bars=False)
                inference.append_simulations(torch.tensor(theta[train], dtype=torch.float32), torch.tensor(x_train))
                estimator = inference.train(training_batch_size=min(args.batch, size // 4), learning_rate=args.lr,
                                            validation_fraction=0.1, stop_after_epochs=args.stop_after,
                                            max_num_epochs=args.epochs, show_train_summary=False)
                summary = {k: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for k, v in inference.summary.items()}
                posterior = inference.build_posterior(estimator)
                torch.save(estimator.state_dict(), args.out / f'estimator_{tag}.pt')
                # held-out: posterior mean error, coverage and ranks
                means, cov68, cov95, ranks, w68 = [], [], [], [], []
                for k, i in enumerate(heldout):
                    s = posterior.sample((args.heldout_samples,), x=torch.tensor(x_held[k]), show_progress_bars=False).numpy()
                    a, b = np.quantile(s, [.16, .84], axis=0); a95, b95 = np.quantile(s, [.025, .975], axis=0)
                    means.append(s.mean(0)); cov68.append((theta[i] >= a) & (theta[i] <= b)); cov95.append((theta[i] >= a95) & (theta[i] <= b95))
                    ranks.append(np.mean(s < theta[i], axis=0)); w68.append((b - a) / width)
                means = np.array(means); ranks = np.array(ranks)
                rmse = np.sqrt(np.mean(((means - theta[heldout]) / width) ** 2, axis=0))
                np.savez_compressed(args.out / f'heldout_{tag}.npz', rows=heldout, truth=theta[heldout], mean=means, ranks=ranks, width68=np.array(w68))
                np.save(args.out / f'training_rows_{method}_{size}.npy', train)
                # observations
                obs_out = {}
                for label, xo in x_obs.items():
                    torch.manual_seed(seed + 100000)
                    s = posterior.sample((args.obs_samples,), x=torch.tensor(xo), show_progress_bars=False).numpy()
                    np.save(args.out / f'posterior_{label}_{tag}.npy', s.astype(np.float32))
                    truth = B12 if label.startswith('Battaglia12') else None
                    obs_out[label] = summarize(s, low, high, truth)
                    obs_out[label]['nearest_training_feature_distance'] = float(np.min(np.linalg.norm(x_train - xo, axis=1)))
                    obs_out[label]['training_feature_rms_nearest_neighbour_distance'] = float(np.median(
                        np.sort(np.linalg.norm(x_train[:2048, None, :] - x_train[None, :2048, :], axis=2), axis=1)[:, 1]))
                    corner(s, low, high, truth, title=f'{label}: {method}, {size} training rows, seed {seed}',
                           path=args.out / f'corner_{label}_{tag}.png')
                result = dict(tag=tag, method=method, size=size, seed=seed, seconds=time.monotonic() - started,
                              epochs=summary.get('epochs_trained'), best_validation_loss=summary.get('best_validation_loss'),
                              heldout=dict(rows=len(heldout), rmse_over_width=rmse.tolist(), rmse_over_prior_mean_rmse=(rmse / prior_rmse).tolist(),
                                           coverage68=np.mean(cov68, axis=0).tolist(), coverage95=np.mean(cov95, axis=0).tolist(),
                                           median_width68_over_prior=np.median(w68, axis=0).tolist()),
                              observations=obs_out, compression=cmeta, network=dict(model='maf', hidden=args.hidden, transforms=args.transforms))
                results.append(result)
                results_path.write_text(json.dumps(results, indent=1) + '\n')
                print(tag, f'{result["seconds"]:.0f} s, epochs {result["epochs"]}, rmse/prior', np.round(rmse / prior_rmse, 3).tolist(), flush=True)
    (args.out / 'prior_mean_rmse_over_width.json').write_text(json.dumps(dict(heldout_rows=len(heldout), value=prior_rmse.tolist()), indent=1))
    print('finished; results in', results_path)


if __name__ == '__main__':
    main()
