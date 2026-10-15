#!/usr/bin/env python3
"""Publication figures from verified inference, or saved scale-cut results only.

--existing-only can be run without idark or any trained model. It does not
produce placeholder truth-mean/corner figures. The default requires completed,
hash-verified idark inference before computing new scientific diagnostics.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from diagnostic_metrics import PARAMETERS, LABELS, resample_pearson

STYLE = {'font.size': 17, 'axes.labelsize': 19, 'axes.titlesize': 20,
         'xtick.labelsize': 15, 'ytick.labelsize': 15,
         'legend.fontsize': 15, 'pdf.fonttype': 42, 'ps.fonttype': 42}
COLORS = ['#2364AA', '#E38B2C', '#5AA7A2', '#9A55A3', '#555555']
MODEL_ORDER = ['original_linear', 'band50_linear', 'original_logit', 'band50_logit', 'lmin580_linear']
MODEL_NAMES = ['Original unbounded', 'Band50 unbounded', 'Original bounded',
               'Band50 bounded', r'$\ell\geq580$ unbounded']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item))
    Path(path).write_text(json.dumps(value, indent=2, default=encode, allow_nan=False)+'\n')


def table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, root, name):
    fig.savefig(root/'figures'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(root/'figures'/f'{name}.png', bbox_inches='tight', dpi=180)
    plt.close(fig)


def existing_results(root):
    data = json.loads((root/'evidence/summary.json').read_text())
    methods = ['original_merlin', 'merlin_band50_full', 'merlin_lmin580']
    names = ['Original', 'Band50', r'Band50, $\ell\geq580$']
    rows = []
    arms = ['simulation', 'mask', 'combined']
    for arm in arms:
        records = [data['subsets'][arm]['held_out'][name] for name in methods]
        for name, record in zip(methods, records):
            rows.append(dict(arm=arm, method=name, **{key: record[key] for key in
                ['cases', 'truth_rms', 'agreement_rms', 'width_cost',
                 'worst_parameter_width_cost', 'truth_fraction68', 'truth_fraction95']}))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, metric, ylabel in zip(axes, ['truth_rms', 'agreement_rms', 'width_cost'],
                               ['Truth error / baseline scale', 'Spread / baseline scale', '68% width ratio']):
        for k, method in enumerate(methods):
            values = [data['subsets'][arm]['held_out'][method][metric] for arm in arms]
            ax.plot(np.arange(3), values, 'o-', color=COLORS[k], label=names[k], lw=2, ms=8)
        if metric == 'width_cost': ax.axhline(1.25, color='.5', ls=':', lw=1.2)
        ax.set_xticks(np.arange(3), ['Simulation', 'Mask', 'Combined'])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .89))
    save(fig, root, 'saved_scale_cut_comparison')
    table(root/'data/saved_scale_cut_comparison.csv', rows)
    # Demonstrate widths using the actual archived 100/12345 spectrum. This
    # linear averaging is not a test of inference with retrained 25/100 models.
    spectra = root/'evidence/baseline_spectra.npz'
    if spectra.exists():
        with np.load(spectra) as d:
            ell, noisy, clean = d['ell'], d['noisy'], d['clean']
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
        shown = (ell >= 1080) & (ell < 1580)
        for ax, width in zip(axes, [25, 50, 100]):
            edges = np.arange(80, 7981, width)
            values = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                inside = (ell >= lo) & (ell < hi)
                weights = 2*ell[inside]+1
                values.append(np.average(noisy[inside], weights=weights))
            ax.plot(ell[shown], noisy[shown], color='.6', lw=.8, alpha=.7, label='Saved noisy spectrum')
            ax.plot(ell[shown], clean[shown], color='k', lw=2, label='Clean spectrum')
            ax.stairs(values, edges, baseline=None, color=COLORS[1], lw=2.4, label='Weighted bands')
            ax.set_xlim(1080, 1580)
            lo, hi = np.min(noisy[shown]), np.max(noisy[shown])
            ax.set_ylim(lo-.05*(hi-lo), hi+.05*(hi-lo))
            ax.set_title(rf'$\Delta\ell={width}$')
            ax.set_xlabel(r'$\ell$')
            ax.grid(alpha=.15)
        axes[0].set_ylabel(r'$D_\ell$')
        axes[0].legend(frameon=False, fontsize=13)
        fig.tight_layout()
        save(fig, root, 'band_width_illustration')
    return rows


def load_verified(root):
    marker = root/'INFERENCE_COMPLETE.json'
    if not marker.exists():
        raise RuntimeError('No completed idark inference; use --existing-only for archived results')
    manifest = json.loads(marker.read_text())
    for name, expected in manifest['outputs'].items():
        assert sha(root/name) == expected, f'Inference output changed: {name}'
    arrays = {name: dict(np.load(root/'data'/f'{name}.npz')) for name in MODEL_ORDER}
    reference = arrays['original_linear']
    for d in arrays.values():
        for key in ['truth', 'design_ids', 'held_positions', 'low', 'high']:
            np.testing.assert_array_equal(d[key], reference[key])
    return arrays


def test_figures(root, arrays):
    truth = arrays['original_linear']['truth']
    means = {name: d['mean'] for name, d in arrays.items()}
    bootstrap = resample_pearson(truth, means)
    metrics = {name: json.loads((root/f'metrics_{name}.json').read_text()) for name in MODEL_ORDER}
    density_parity = []
    for space in ['linear', 'logit']:
        previous = json.loads((root/'evidence'/f'prior_final_test_{space}.json').read_text())['methods']
        for method, old_name in [('original', 'original_merlin'), ('band50', 'merlin_band50_full')]:
            name = f'{method}_{space}'
            difference = metrics[name]['mean_log_prob_truth']-previous[old_name]['mean_log_prob_truth']
            assert abs(difference) < 1e-4, f'Reserved-test truth density changed: {name}, {difference}'
            density_parity.append(dict(model=name, mean_log_prob_difference=difference))
    rows, paired = [], []
    for name, boot in bootstrap.items():
        r = metrics[name]['pearson']
        interval = np.quantile(boot, [.025, .975], axis=0)
        metrics[name]['pearson_row_bootstrap95'] = interval
        base = 'original_logit' if name.endswith('logit') else 'original_linear'
        width = arrays[name]['quantiles'][3]-arrays[name]['quantiles'][1]
        base_width = arrays[base]['quantiles'][3]-arrays[base]['quantiles'][1]
        assert np.all(width > 0) and np.all(base_width > 0)
        width_ratio = np.exp(np.mean(np.log(width/base_width), axis=0))
        metrics[name]['width68_ratio_geomean'] = width_ratio
        metrics[name]['width68_ratio_all_parameters'] = float(np.exp(np.mean(np.log(width/base_width))))
        delta_boot = boot-bootstrap[base]
        delta_interval = np.quantile(delta_boot, [.025, .975], axis=0)
        for j, parameter in enumerate(PARAMETERS):
            rows.append(dict(model=name, parameter=parameter, pearson=r[j],
                pearson_lo=interval[0, j], pearson_hi=interval[1, j],
                width68_ratio_geomean=width_ratio[j],
                **{key: metrics[name][key][j] for key in
                   ['normalized_rmse', 'normalized_bias', 'regression_slope',
                    'normalized_regression_intercept', 'coverage68', 'coverage95',
                    'median_width68_prior', 'rms_mean_mcse_prior',
                    'maximum_mean_mcse_prior', 'half_sample_pearson_difference']}))
            paired.append(dict(model=name, reference=base, parameter=parameter,
                delta_pearson=r[j]-metrics[base]['pearson'][j],
                delta_lo=delta_interval[0, j], delta_hi=delta_interval[1, j]))
    table(root/'data/truth_mean_metrics.csv', rows)
    table(root/'data/paired_pearson_changes.csv', paired)
    write_json(root/'TRUTH_MEAN_SUMMARY.json', dict(
        cases=1500, parameter_order=PARAMETERS, models=metrics,
        correlation='Pearson between distinct true theta and posterior means across test rows',
        scope='Conditional on halo100/mask12345, varied parameter points and SO noise',
        previous_reserved_test_truth_density_parity=density_parity,
        bootstrap='1000 paired row resamples; descriptive test-cohort intervals, not cosmic variance'))
    for space in ['linear', 'logit']:
        fig, axes = plt.subplots(3, 3, figsize=(17, 14))
        chosen = [f'original_{space}', f'band50_{space}']
        for j, ax in enumerate(axes.flat):
            for name, color in zip(chosen, COLORS[:2]):
                ax.scatter(truth[:, j], arrays[name]['mean'][:, j], s=12, alpha=.3,
                           color=color, edgecolors='none', rasterized=True)
            low = arrays[chosen[0]]['low'][j]
            high = arrays[chosen[0]]['high'][j]
            ax.plot([low, high], [low, high], 'k--', lw=1.4)
            ax.set_title(LABELS[j])
            ax.set_xlabel('True value')
            ax.set_ylabel('Posterior mean')
            ax.text(.03, .97, 'r = '+', '.join(f"{metrics[n]['pearson'][j]:.3f}" for n in chosen),
                    transform=ax.transAxes, va='top', fontsize=13,
                    bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
            ax.grid(alpha=.12)
        handles = [Line2D([], [], color=c, marker='o', linestyle='', label=t)
                   for c, t in zip(COLORS[:2], ['Original', 'Band50'])]
        fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .96))
        save(fig, root, f'truth_mean_{space}')
    for metric, ylabel in [('pearson', 'Truth-mean Pearson r'),
                           ('normalized_rmse', 'RMSE / prior width'),
                           ('normalized_bias', 'Mean bias / prior width'),
                           ('width68_ratio_geomean', '68% width / matching original'),
                           ('coverage68', '68% marginal coverage')]:
        fig, ax = plt.subplots(figsize=(16, 6))
        for k, name in enumerate(MODEL_ORDER):
            values = np.asarray(metrics[name][metric])
            x = np.arange(9)+(k-2)*.13
            if metric == 'pearson':
                interval = np.asarray(metrics[name]['pearson_row_bootstrap95'])
                # Percentile intervals need not straddle the point estimate.
                ax.vlines(x, interval[0], interval[1], color=COLORS[k], lw=1.3)
            ax.plot(x, values, 'o', color=COLORS[k], ms=7, label=MODEL_NAMES[k])
        if metric == 'coverage68': ax.axhline(.68, color='k', ls='--', lw=1.2)
        if metric == 'width68_ratio_geomean': ax.axhline(1, color='k', ls='--', lw=1.2)
        if metric == 'normalized_bias': ax.axhline(0, color='k', ls='--', lw=1.2)
        ax.set_xticks(np.arange(9), LABELS)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.15)
        ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.25), frameon=False)
        fig.tight_layout()
        save(fig, root, metric)
    # Save every evaluated mean and uncertainty; no full 1,500-row draw cube.
    records = []
    for name, d in arrays.items():
        for i in range(len(truth)):
            for j, parameter in enumerate(PARAMETERS):
                records.append(dict(model=name, design_id=int(d['design_ids'][i]),
                    held_position=int(d['held_positions'][i]), parameter=parameter,
                    truth=d['truth'][i, j], mean=d['mean'][i, j],
                    mean_mcse=d['mean_mcse'][i, j], q16=d['quantiles'][1, i, j],
                    q84=d['quantiles'][3, i, j]))
    table(root/'data/test_case_means.csv', records)


def contour_thresholds(histogram, sample_count):
    """68/95% mass thresholds relative to all draws, including off-axis tails."""
    density = gaussian_filter(histogram.astype(float), 1, mode='reflect')/sample_count
    sorted_density = np.sort(density.ravel())[::-1]
    mass = np.cumsum(sorted_density)
    if mass[-1] < .95:
        raise ValueError('Plot range misses too much posterior mass for a 95% contour')
    levels = [sorted_density[np.searchsorted(mass, p)] for p in [.95, .68]]
    return density, np.unique(levels)


def corner(root, draws, truth, indices, names, colors, filename):
    """Histogram contours smooth the DISPLAY by one bin; spectra are unchanged."""
    dim = len(indices)
    size = 24 if dim == 9 else 13
    fig, axes = plt.subplots(dim, dim, figsize=(size, size), squeeze=False)
    ranges = []
    for j in indices:
        limits = np.array([np.quantile(d[:, j], [.002, .998]) for d in draws])
        lo = min(limits[:, 0].min(), truth[j])
        hi = max(limits[:, 1].max(), truth[j])
        padding = .04*(hi-lo)
        ranges.append((lo-padding, hi+padding))
    for row, j in enumerate(indices):
        for col, k in enumerate(indices):
            ax = axes[row, col]
            if col > row:
                ax.set_visible(False)
                continue
            if row == col:
                for draw, color in zip(draws, colors):
                    counts, edges = np.histogram(draw[:, j], bins=60, range=ranges[row])
                    density = gaussian_filter1d(counts.astype(float), 1)/(len(draw)*np.diff(edges))
                    ax.plot((edges[:-1]+edges[1:])/2, density, color=color, lw=2)
                ax.axvline(truth[j], color='k', ls='--', lw=1.3)
                ax.set_xlim(ranges[row])
                ax.set_yticks([])
            else:
                for draw, color in zip(draws, colors):
                    hist, xe, ye = np.histogram2d(draw[:, k], draw[:, j], bins=55,
                                                 range=[ranges[col], ranges[row]])
                    density, levels = contour_thresholds(hist, len(draw))
                    ax.contour((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2,
                               density.T, levels=levels, colors=[color], linewidths=1.5)
                ax.axvline(truth[k], color='k', ls='--', lw=1)
                ax.axhline(truth[j], color='k', ls='--', lw=1)
                ax.plot(truth[k], truth[j], 'k+', ms=8)
                ax.set_xlim(ranges[col]); ax.set_ylim(ranges[row])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            if row != col:
                ax.yaxis.set_major_locator(MaxNLocator(3))
            else:
                # A locator installed after set_yticks([]) restores density
                # ticks and lets their labels spill into adjacent panels.
                ax.set_yticks([])
            if row == dim-1:
                ax.set_xlabel(LABELS[k], fontsize=23 if dim == 9 else 26)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0 and row > 0:
                ax.set_ylabel(LABELS[j], fontsize=23 if dim == 9 else 26)
            elif row != col:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=16 if dim == 9 else 20)
    handles = [Line2D([], [], color=c, lw=2, label=n) for c, n in zip(colors, names)]
    handles.append(Line2D([], [], color='k', ls='--', label='Truth'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.97, .96), frameon=False,
               fontsize=20 if dim == 9 else 22)
    fig.subplots_adjust(left=.075, right=.98, bottom=.075, top=.98, hspace=.10, wspace=.16)
    save(fig, root, filename)


def corner_figures(root):
    examples = json.loads((root/'corner_examples.json').read_text())['cases']
    arrays = {name: dict(np.load(root/'data'/f'corners_{name}.npz')) for name in MODEL_ORDER[:4]}
    reference = arrays['original_linear']
    for d in arrays.values():
        np.testing.assert_array_equal(d['truth'], reference['truth'])
        np.testing.assert_array_equal(d['indices'], reference['indices'])
        assert d['draws'].shape == reference['draws'].shape
    for i, example in enumerate(examples):
        truth = reference['truth'][i]
        for space in ['linear', 'logit']:
            names = [f'original_{space}', f'band50_{space}']
            title = 'unbounded' if space == 'linear' else 'bounded'
            corner(root, [arrays[n]['draws'][:, i] for n in names], truth, list(range(9)),
                   ['Original '+title, 'Band50 '+title], COLORS[:2], f'corner_example{i+1}_{space}')
        corner(root, [arrays[n]['draws'][:, i] for n in MODEL_ORDER[:4]], truth, [0, 1, 2, 6],
               MODEL_NAMES[:4], COLORS[:4], f'corner_example{i+1}_compact')
    write_json(root/'CORNER_PLOT_DEFINITION.json', dict(
        samples=reference['draws'].shape[0], parameters=PARAMETERS, cases=examples, contours=[.68, .95],
        range='Union of each displayed model central 99.6% range, padded, with truth included',
        smoothing='One histogram bin for display only; all inference uses unsmoothed posterior draws',
        mass='Contour mass referenced to all samples, including samples outside displayed axes',
        warning='Two seeded-random cases and an explicitly selected original-model stress case; not population evidence'))


def main(args):
    root = args.root
    (root/'figures').mkdir(exist_ok=True)
    (root/'data').mkdir(exist_ok=True)
    plt.rcParams.update(STYLE)
    existing_results(root)
    if args.existing_only:
        write_json(root/'PREPARATION_STATUS.json', dict(
            status='Prepared locally; idark unavailable; no new inference executed or submitted',
            ready=['existing scale-cut figures', 'band-width illustration', 'inference code', 'plotting code'],
            pending=['1500-case truth-mean evaluation', 'posterior corner examples', 'completed supplement PDF']))
        return
    arrays = load_verified(root)
    test_figures(root, arrays)
    corner_figures(root)
    products = sorted((root/'figures').glob('*'))+sorted((root/'data').glob('*.csv'))
    products += [root/'TRUTH_MEAN_SUMMARY.json', root/'CORNER_PLOT_DEFINITION.json']
    write_json(root/'PLOTS_COMPLETE.json', dict(
        inference_marker_sha256=sha(root/'INFERENCE_COMPLETE.json'),
        outputs={str(p.relative_to(root)): sha(p) for p in products}))
    # A preparation marker must not survive a completed scientific run.
    (root/'PREPARATION_STATUS.json').unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--existing-only', action='store_true')
    main(parser.parse_args())
