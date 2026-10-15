"""Calibration diagnostics for an analyze_64k.py results folder, with the sbi package's diagnostics.

For the largest training size of each compression method it rebuilds the trained MAF posteriors (one per
network seed) and their equal-weight mixture (what sbi's EnsemblePosterior samples), and on the held-out rows:
  * SBC: ranks from sbi.diagnostics.sbc._run_sbc (the rank step of run_sbc), sbi.diagnostics.check_sbc,
    sbi.analysis.sbc_rank_plot (histogram and CDF)
  * TARP (joint coverage): sbi.diagnostics.tarp._run_tarp (the step of run_tarp), check_tarp
  * truth vs posterior mean, 3x3 panels with Pearson r (style of plot_so_true_mean_correlations.py)
  * Pearson r vs training-set size (all sizes in this and any --extra-results folders)
Posterior samples are drawn once per network with per-row rejection to the prior box (identical to
DirectPosterior.sample), because sbi's batched sampler redraws every row until the leakiest row is done.
Outputs go to <results>/diagnostics/.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analyze_64k as A

LABELS = [r'$P_0$', r'$x_{\rm c}$', r'$\beta$', r'$\alpha_{m,P_0}$', r'$\alpha_{m,x_{\rm c}}$', r'$\alpha_{m,\beta}$',
          r'$\alpha_{z,P_0}$', r'$\alpha_{z,x_{\rm c}}$', r'$\alpha_{z,\beta}$']
PARAM_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f', '#17becf']
METHOD_LABEL = dict(moped_fixed='MOPED (fixed B12 compression, 9 numbers)', bins40='40 band powers')


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.linewidth': 0.6,
                         'savefig.bbox': 'tight', 'savefig.dpi': 200})


def pearson_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) == 0 or np.std(y[m]) == 0: return float('nan')
    return float(np.corrcoef(x[m], y[m])[0, 1])


def load_dataset(path):
    meta = json.loads((path / 'meta.json').read_text())
    return dict(meta=meta, low=np.array(meta['lower']), high=np.array(meta['upper']), theta=np.load(path / 'theta.npy'),
                clean_u=np.load(path / 'clean_dl_unbinned.npy'), noisy_u=np.load(path / 'noisy_dl_unbinned.npy'),
                clean_b=np.load(path / 'clean_dl.npy'), noisy_b=np.load(path / 'noisy_dl.npy'))


def rebuild(results, run, data, prior):
    """Posterior of one saved network, with the compression refitted exactly as in training."""
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    method, size, seed = run['method'], run['size'], run['seed']
    train = np.load(results / f'training_rows_{method}_{size}.npy')
    unb = method == 'moped_fixed'
    clean, noisy = (data['clean_u'], data['noisy_u']) if unb else (data['clean_b'], data['noisy_b'])
    state, _ = A.fit_transform(method, clean[train], noisy[train])
    xt = torch.tensor(A.project(state, noisy[train])); tt = torch.tensor(data['theta'][train], dtype=torch.float32)
    builder = posterior_nn(model='maf', hidden_features=run['network']['hidden'], num_transforms=run['network']['transforms'],
                           z_score_x='none', z_score_theta='independent')
    net = builder(tt, xt)
    net.load_state_dict(torch.load(results / f'estimator_{method}_{size}_{seed}.pt')); net.eval()
    return net, state


def sample_in_prior(net, xs, n, low, high, max_per_call=100_000, max_draws_per_row=20_000_000):
    """n posterior samples per condition, restricted to the prior box, as DirectPosterior.sample does.

    Memory is bounded by max_per_call flow samples per call. Phase 1 draws n for every row (in groups);
    phase 2 tops up, one row at a time and in bounded chunks, only the rows whose flow puts mass outside
    the box. Returns (n, rows, dim) float32 and the in-prior acceptance rate of each row.
    """
    xs = torch.as_tensor(xs, dtype=torch.float32)
    rows, dim = xs.shape[0], len(low)
    out = np.empty((n, rows, dim), dtype=np.float32)
    filled = np.zeros(rows, dtype=int); drawn = np.zeros(rows); accepted = np.zeros(rows)

    def draw(row_ids, d):
        with torch.no_grad():
            s = net.sample((d,), condition=xs[row_ids]).numpy()          # (d, len(row_ids), dim)
        ok = np.all((s >= low) & (s <= high), axis=2)
        for k, row in enumerate(row_ids):
            acc = s[ok[:, k], k]
            take = min(n - filled[row], len(acc))
            out[filled[row]:filled[row] + take, row] = acc[:take]
            filled[row] += take; drawn[row] += d; accepted[row] += len(acc)

    group = max(1, max_per_call // n)
    for start in range(0, rows, group):
        draw(np.arange(start, min(start + group, rows)), n)
    for row in np.flatnonzero(filled < n):
        while filled[row] < n and drawn[row] < max_draws_per_row:
            rate = max(accepted[row] / drawn[row], 1e-4)
            draw(np.array([row]), int(min(max_per_call, np.ceil(1.3 * (n - filled[row]) / rate))))
    if np.any(filled < n):
        raise RuntimeError(f'{int(np.sum(filled < n))} rows have fewer than {n} samples inside the prior')
    return out, accepted / drawn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=None, help='default: the dataset recorded in run_config.json')
    p.add_argument('--methods', nargs='+', default=['moped_fixed', 'bins40'])
    p.add_argument('--posterior-samples', type=int, default=1000)
    p.add_argument('--extra-results', nargs='*', type=Path, default=[], help='more results folders for r vs training size')
    p.add_argument('--threads', type=int, default=10)
    p.add_argument('--only-correlation', action='store_true', help='only redraw the r vs training size figure')
    args = p.parse_args()
    torch.set_num_threads(args.threads); style()
    from sbi.utils import BoxUniform
    from sbi.diagnostics import check_sbc, check_tarp
    from sbi.diagnostics.sbc import _run_sbc              # sbi's rank computation (what run_sbc calls after sampling)
    from sbi.diagnostics.tarp import _run_tarp, get_tarp_references   # likewise for run_tarp
    from sbi.utils.metrics import l2
    from sbi.analysis.plot import sbc_rank_plot

    cfg_path = args.results / 'run_config.json'
    dataset = args.dataset or Path(json.loads(cfg_path.read_text())['dataset'])
    if not args.only_correlation:
        data = load_dataset(dataset)
        low, high = data['low'], data['high']
        prior = BoxUniform(torch.tensor(low, dtype=torch.float32), torch.tensor(high, dtype=torch.float32))
    out = args.results / 'diagnostics'; out.mkdir(exist_ok=True)
    runs = json.loads((args.results / 'results.json').read_text())
    obs_dir = A.TEST256 / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
    b12_obs = np.load(obs_dir / 'Battaglia12.npz')
    summary = dict(results=str(args.results), dataset=str(dataset))
    N = args.posterior_samples

    for method in ([] if args.only_correlation else args.methods):
        mruns = [r for r in runs if r['method'] == method]
        if not mruns: continue
        size = max(r['size'] for r in mruns)
        chosen = sorted((r for r in mruns if r['size'] == size), key=lambda r: r['seed'])
        held = np.load(args.results / f"heldout_{method}_{size}_{chosen[0]['seed']}.npz")['rows']
        thetas = torch.tensor(data['theta'][held], dtype=torch.float32)
        unb = method == 'moped_fixed'
        per_seed, labels, accept = [], [], {}
        for r in chosen:
            t0 = time.monotonic()
            net, state = rebuild(args.results, r, data, prior)
            xs = A.project(state, (data['noisy_u'] if unb else data['noisy_b'])[held])
            x_b12 = A.project(state, b12_obs['noisy_dl_unbinned' if unb else 'noisy_dl'][None])
            torch.manual_seed(r['seed'])
            samples, rate = sample_in_prior(net, xs, N, low, high)
            b12, _ = sample_in_prior(net, x_b12, 4 * N, low, high)
            per_seed.append((samples, b12[:, 0])); labels.append(f"seed {r['seed']}")
            accept[labels[-1]] = dict(median=float(np.median(rate)), min=float(rate.min()), rows_below_10pct=int(np.sum(rate < 0.1)))
            print(f'{method}_{size}', labels[-1], f'sampled in {time.monotonic() - t0:.0f} s; in-prior acceptance',
                  accept[labels[-1]], flush=True)
        if len(per_seed) > 1:          # equal-weight mixture of the seeds' posteriors, as sbi's EnsemblePosterior
            rng = np.random.default_rng(0)
            half = N // len(per_seed)
            mix = np.concatenate([s[:half] for s, _ in per_seed], axis=0)
            mix = np.take_along_axis(mix, np.argsort(rng.random(mix.shape[:2]), axis=0)[:, :, None], axis=0)
            mixb = np.concatenate([b[: (4 * N) // len(per_seed)] for _, b in per_seed], axis=0)
            per_seed.append((mix, mixb)); labels.append(f'ensemble of {len(chosen)} seeds')
        tag = f'{method}_{size}'
        msum = dict(size=size, heldout_rows=int(len(held)), in_prior_acceptance=accept, networks={})
        ranks_all, tarp_all = [], []
        for (samples, b12), label in zip(per_seed, labels):
            t0 = time.monotonic()
            post_samples = torch.tensor(samples)
            n_here = post_samples.shape[0]
            ranks = _run_sbc(thetas, torch.zeros(len(held), 1), post_samples, 'marginals', False)
            dap = post_samples[0]
            check = check_sbc(ranks, thetas, dap, num_posterior_samples=n_here)
            torch.manual_seed(1)
            ecp, alpha = _run_tarp(post_samples, thetas, get_tarp_references(thetas), l2, None, True)
            atc, ks_tarp = check_tarp(ecp, alpha)
            mean, std = samples.mean(0), samples.std(0)
            r_vals = [pearson_r(data['theta'][held][:, i], mean[:, i]) for i in range(9)]
            msum['networks'][label] = dict(
                sbc_ks_pvalues=check['ks_pvals'].numpy().tolist(), sbc_c2st_ranks=check['c2st_ranks'].numpy().tolist(),
                sbc_c2st_dap=check['c2st_dap'].numpy().tolist(), tarp_atc=float(atc), tarp_ks_pvalue=float(ks_tarp),
                pearson_r_truth_vs_mean=r_vals, posterior_samples_per_row=n_here, seconds=time.monotonic() - t0)
            np.savez_compressed(out / f"heldout_{tag}_{label.replace(' ', '_')}.npz", rows=held, truth=data['theta'][held],
                                mean=mean, std=std, ranks=ranks.numpy(), ecp=ecp.numpy(), alpha=alpha.numpy(), b12_samples=b12)
            ranks_all.append(ranks.numpy()); tarp_all.append((ecp.numpy(), alpha.numpy(), label, atc, ks_tarp))
            print(tag, label, f"{time.monotonic() - t0:.0f} s; SBC KS p-values", np.round(check['ks_pvals'].numpy(), 3).tolist(),
                  f"; TARP ATC {float(atc):.3f} (KS p {float(ks_tarp):.2g})", flush=True)

            # ---------------- truth vs posterior mean, 3x3 ----------------
            fig, axes = plt.subplots(3, 3, figsize=(18 / 2.54, 17 / 2.54)); axes = axes.ravel()
            truth = data['theta'][held]
            for i, ax in enumerate(axes):
                ax.errorbar(truth[:, i], mean[:, i], yerr=std[:, i], fmt='none', ecolor=PARAM_COLORS[0], alpha=0.10, elinewidth=0.45, zorder=1)
                ax.scatter(truth[:, i], mean[:, i], s=9, color=PARAM_COLORS[0], alpha=0.7, edgecolor='none', zorder=2)
                ax.errorbar([A.B12[i]], [b12[:, i].mean()], yerr=[b12[:, i].std()], fmt='*', color='k', ms=9, elinewidth=1, zorder=4,
                            label='Battaglia+12 observation' if i == 0 else None)
                ax.plot([low[i], high[i]], [low[i], high[i]], color='k', lw=0.8, ls=':', zorder=0)
                ax.axhline((low[i] + high[i]) / 2, color='gray', lw=0.6, ls='--', zorder=0)
                ax.set_xlim(low[i], high[i]); ax.set_ylim(low[i], high[i])
                ax.set_title(rf"{LABELS[i]}  $r={r_vals[i]:.2f}$", pad=2)
                ax.set_xlabel('True'); ax.set_ylabel('Posterior mean'); ax.grid(True, alpha=0.25, lw=0.5)
            axes[0].legend(loc='upper left', frameon=False, fontsize=7)
            fig.suptitle(f'Truth vs posterior mean on {len(held)} held-out simulations: {METHOD_LABEL[method]}, '
                         f'N = {size:,}, {label}\n(error bars: posterior std; dashed: prior mean)', fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"truth_vs_mean_{tag}_{label.replace(' ', '_')}.png"); plt.close(fig)

        # ---------------- SBC rank histograms (sbi) for the ensemble, CDFs for all networks ----------------
        n_bins = 20
        n_last = per_seed[-1][0].shape[0]
        fig, ax = sbc_rank_plot(ranks_all[-1], n_last, num_bins=n_bins, plot_type='hist', parameter_labels=LABELS,
                                ranks_labels=[labels[-1]], num_cols=3, figsize=(12, 9.5))
        fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.90)
        fig.suptitle(f'SBC rank histograms (sbi.analysis.sbc_rank_plot): {METHOD_LABEL[method]}, N = {size:,}, {labels[-1]}\n'
                     f'{len(held)} held-out simulations; grey band: 99 % range for a calibrated posterior; '
                     f'U shape = overconfident, one-sided excess = biased', fontsize=10)
        fig.savefig(out / f'sbc_ranks_hist_{tag}.png'); plt.close(fig)
        colors3 = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(labels)]
        fig, ax = sbc_rank_plot(ranks_all, n_last, plot_type='cdf', parameter_labels=LABELS, ranks_labels=labels,
                                colors=colors3, num_cols=3, figsize=(12, 10))
        fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.88)
        handles = [plt.Line2D([], [], color=c, lw=2) for c in colors3] + [plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3)]
        fig.legend(handles, labels + ['99 % band for a calibrated posterior'], loc='upper center', ncol=len(labels) + 1,
                   bbox_to_anchor=(0.5, 0.945), frameon=False, fontsize=9)
        fig.suptitle(f'SBC rank CDFs (sbi.analysis.sbc_rank_plot): {METHOD_LABEL[method]}, N = {size:,}, {len(held)} held-out simulations',
                     fontsize=10, y=0.985)
        fig.savefig(out / f'sbc_ranks_cdf_{tag}.png'); plt.close(fig)

        # ---------------- TARP expected coverage (sbi) ----------------
        fig, ax = plt.subplots(figsize=(10 / 2.54, 9.5 / 2.54))
        ax.plot([0, 1], [0, 1], 'k:', lw=0.9, label='calibrated')
        for (ecp, alpha, label, atc, ks), color in zip(tarp_all, ['#1f77b4', '#ff7f0e', '#2ca02c']):
            c68, c95 = np.interp([0.68, 0.95], np.ravel(alpha), np.ravel(ecp))
            ax.plot(alpha, ecp, color=color, lw=1.4, label=f'{label}: 68 % level covers {100 * c68:.0f} %, 95 % covers {100 * c95:.0f} %')
            msum['networks'][label]['tarp_coverage_at_68'] = float(c68); msum['networks'][label]['tarp_coverage_at_95'] = float(c95)
        ax.set_xlabel('credibility level'); ax.set_ylabel('expected coverage')
        ax.set_title(f'TARP joint coverage (sbi TARP, ATC {float(tarp_all[-1][3]):+.3f} for the ensemble)\n{METHOD_LABEL[method]}, N = {size:,}', fontsize=9)
        ax.text(0.97, 0.03, 'below the diagonal: overconfident\nabove: underconfident', ha='right', va='bottom', fontsize=7, transform=ax.transAxes)
        ax.legend(frameon=False, fontsize=6.5, loc='upper left'); ax.grid(True, alpha=0.25, lw=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        fig.tight_layout(); fig.savefig(out / f'tarp_{tag}.png'); plt.close(fig)
        summary[method] = msum

    # ---------------- Pearson r vs training size (posterior means saved by analyze_64k, seeds averaged) ----------------
    # Each results folder is its own series (its own held-out set); lines join sizes only within one folder.
    folders = [args.results] + list(args.extra_results)
    markers = [dict(marker='o', mfc=None), dict(marker='o', mfc='white'), dict(marker='s', mfc='white')]
    for method in args.methods:
        series = []
        for folder in folders:
            fruns = [r for r in json.loads((folder / 'results.json').read_text()) if r['method'] == method]
            points = {}
            for size in sorted({r['size'] for r in fruns}):
                means, truth = [], None
                for r in (x for x in fruns if x['size'] == size):
                    h = np.load(folder / f"heldout_{method}_{size}_{r['seed']}.npz"); means.append(h['mean']); truth = h['truth']
                m = np.mean(means, axis=0)
                points[size] = ([pearson_r(truth[:, i], m[:, i]) for i in range(9)], len(truth))
            if points: series.append((folder, points))
        if not series: continue
        fig, ax = plt.subplots(figsize=(13 / 2.54, 8 / 2.54))
        for k, (folder, points) in enumerate(series):
            sizes = sorted(points); mk = markers[min(k, len(markers) - 1)]
            for i in range(9):
                ax.plot(sizes, [points[s][0][i] for s in sizes], lw=1, ms=3.5, color=PARAM_COLORS[i],
                        marker=mk['marker'], mfc=mk['mfc'] or PARAM_COLORS[i], label=LABELS[i] if k == 0 else None)
        handles, labels = ax.get_legend_handles_labels()
        for k, (folder, points) in enumerate(series):
            mk = markers[min(k, len(markers) - 1)]
            n_held = sorted({v[1] for v in points.values()})
            handles.append(plt.Line2D([], [], color='k', lw=1, marker=mk['marker'], mfc=mk['mfc'] or 'k'))
            labels.append(f"{folder.name} ({'/'.join(map(str, n_held))} held-out)")
        ax.set_xscale('log'); ax.set_ylim(0, 1); ax.set_xlabel('Training set size')
        ax.set_ylabel(r'Pearson $r(\theta_{\rm true}, \bar{\theta}_{\rm post})$')
        ax.set_title(f'{METHOD_LABEL[method]}: held-out correlation vs training size', fontsize=9)
        ax.grid(True, which='both', alpha=0.25, lw=0.5); ax.legend(handles, labels, ncol=3, frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(out / f'correlation_vs_size_{method}.png'); plt.close(fig)
        summary.setdefault('correlation_vs_size', {})[method] = {f'{folder.name}:{s}': points[s][0] for folder, points in series for s in points}
    if args.only_correlation:
        previous = json.loads((out / 'diagnostics.json').read_text()) if (out / 'diagnostics.json').exists() else {}
        previous['correlation_vs_size'] = summary.get('correlation_vs_size', {}); summary = previous
    (out / 'diagnostics.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
