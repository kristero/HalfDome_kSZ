"""Corner figures and a summary table for one analyze_64k.py results folder.

Reads, from the folder given by --results (written by analyze_64k.py):
  results.json                                one record per (method, training size, network seed) with the
                                              held-out metrics (rows, coverage68, rmse_over_prior_mean_rmse)
  posterior_<obs>_<method>_<size>_<seed>.npy  posterior samples, shape (n_samples, 9), columns in NAMES order,
                                              for obs = Battaglia12 and the three FLAMINGO labels
Only the largest training size of --method is used; the samples of its network seeds are pooled.

Writes into the same folder:
  fig_battaglia12_<method>_<size>.png  Battaglia+12 control: pooled posterior with the true parameters marked
  fig_flamingo_<method>_<size>.png     the three FLAMINGO effective-parameter posteriors, Battaglia+12 marked
  summary_<method>_<size>.md           medians and 68 % intervals, truth recovery, held-out RMSE and coverage
                                       (the table is also printed)

Run:
  python plot_posteriors.py --results results_prelim_A --method moped_fixed

corner() and FLAMINGO are also used by train_ensemble.py and merlin_compression.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator

# Parameter names in the column order of every posterior sample array: the Battaglia+12 gNFW pressure profile
# amplitude P0, core scale xc and outer slope beta, and their mass (alpha_m) and redshift (alpha_z) exponents
NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta',
         'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
# Axis labels, same order as NAMES
LABELS = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_m^{P_0}$', r'$\alpha_m^{x_c}$', r'$\alpha_m^{\beta}$',
          r'$\alpha_z^{P_0}$', r'$\alpha_z^{x_c}$', r'$\alpha_z^{\beta}$']
# Battaglia+12 values: the truth of the Battaglia12 control observation
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
# Lower and upper bounds of the flat prior; the corner axes and histograms span this full range
LOW = np.array([1.0, 0.025, 2.8, -0.6, -1.0, -0.2, -6.0, -1.5, -0.5])
HIGH = np.array([60.0, 4.0, 16.0, 1.5, 0.4, 0.4, 0.5, 3.0, 2.0])
# FLAMINGO observations: (file label, legend name, colour)
FLAMINGO = [('L1_m9', 'FLAMINGO L1_m9 (fiducial)', '#1f77b4'),
            ('fgas-8sigma', 'FLAMINGO fgas-8sigma (strong feedback)', '#d62728'),
            ('Mstar-1sigma', 'FLAMINGO Mstar-1sigma', '#2ca02c')]


def density_levels(h, fractions=(0.68, 0.95)):
    """Highest-density thresholds of a histogram.

    h         : array of non-negative bin weights, any shape (here a smoothed 2-D histogram)
    fractions : fractions of the total weight to enclose
    Returns a list with one threshold per fraction, sorted ascending (the largest fraction gives the lowest
    threshold). The bins with h >= threshold are the fewest bins that hold at least that fraction of the total.
    """
    # rank the bins from the highest to the lowest and accumulate their share of the total
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    # the threshold is the value of the first bin at which the accumulated share reaches the fraction
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)


def corner(sets, truth=None, title='', path=None, bins=30, smooth=1.3, truth_label='Battaglia+12 values',
           prior_band=False, region_legend=False, width_percent=False, ranges=None, note=None):
    """Corner plot of one or more 9-parameter posteriors over the full flat-prior range, saved to `path`.

    Lower triangle: filled 68 % (dark) and 95 % (light) highest-density regions of the smoothed 2-D histograms.
    Diagonal: 1-D density histograms with the median (line) and central 68 % interval (band), the flat prior
    density 1/width and a grey bar under the prior's central 68 % interval. Above each diagonal panel, one line per
    posterior gives the median (+/- 68 % interval) and the 68 % width as a percentage of the prior's 68 % width.

    sets   : list of (samples, label, colour); samples has shape (n_samples, 9), columns in NAMES order
    truth  : optional array of shape (9,), drawn as dashed lines (diagonal) and stars (2-D panels)
    title  : figure title
    path   : output image file (saved at 90 dpi)
    bins   : histogram bins per axis over [LOW, HIGH]
    smooth : standard deviation, in bins, of the Gaussian that smooths the 2-D histograms before the levels are
             found; 0 uses the raw histograms. Smoothing broadens posteriors that span only a few bins.
    truth_label   : legend text of the truth marker
    prior_band    : draw the grey bar under the prior's central 68 % interval (off by default: it cluttered the
                    panels)
    region_legend : explain the shaded 68 / 95 % regions in the legend (off by default)
    width_percent : append the 68 % width as a percentage of the prior's 68 % width to the 1-D texts (off)
    ranges        : optional (9, 2) axis ranges replacing the prior box, for posteriors that are not confined to
                    it (networks trained without the logit bound); the prior bounds are then drawn dotted and the
                    histogram bins keep the prior-box bin width
    note          : optional text block under the legend (upper right)
    Returns None.
    """
    n_par = len(NAMES)
    extended = ranges is not None
    lo, hi = (LOW, HIGH) if not extended else (np.asarray(ranges, float)[:, 0], np.asarray(ranges, float)[:, 1])
    nb = [max(bins, int(round(bins * (hi[k] - lo[k]) / (HIGH[k] - LOW[k])))) for k in range(n_par)]
    multi = len(sets) > 1                          # overlaid posteriors get lighter fills and smaller text
    fig, axes = plt.subplots(n_par, n_par, figsize=(18, 18))
    for i in range(n_par):                         # row i: parameter i on the y axis (and on the diagonal)
        for j in range(n_par):                     # column j: parameter j on the x axis
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            diag_top = 0.0                         # tallest 1-D histogram in this diagonal panel
            for k, (samples, label, color) in enumerate(sets):
                if i == j:
                    # 1-D marginal: density histogram, median line and central 68 % band
                    q16, q50, q84 = np.quantile(samples[:, i], [.16, .5, .84])
                    heights, _, _ = ax.hist(samples[:, i], bins=nb[i], range=(lo[i], hi[i]), density=True,
                                            histtype='step', color=color, lw=1.6, label=label)
                    ax.axvspan(q16, q84, color=color, alpha=0.10 if multi else 0.18, lw=0)
                    ax.axvline(q50, color=color, lw=1.1)
                    # posterior 68 % width over the flat prior's central 68 % width (0.68 of the range)
                    ratio = (q84 - q16) / (0.68 * (HIGH[i] - LOW[i]))
                    # one text line per posterior, stacked above the panel with the first posterior on top
                    text = fmt(q16, q50, q84) + (f'  [{100 * ratio:.0f}%]' if width_percent else '')
                    ax.text(0.5, 1.03 + 0.12 * (len(sets) - 1 - k), text,
                            color=color, fontsize=8.5 if multi else 9.5, ha='center', va='bottom',
                            transform=ax.transAxes)
                    diag_top = max(diag_top, float(np.max(heights)))
                else:
                    # 2-D marginal of (parameter j, parameter i); .T puts parameter i on the rows for contour
                    counts, x_edges, y_edges = np.histogram2d(samples[:, j], samples[:, i], bins=[nb[j], nb[i]],
                                                              range=[(lo[j], hi[j]), (lo[i], hi[i])])
                    # Gaussian smoothing with sigma = `smooth` bins. It broadens posteriors that span only a few
                    # bins, so the levels below enclose 68 / 95 % of the smoothed histogram, not of the samples.
                    smoothed = gaussian_filter(counts.T, smooth)
                    x_mid = 0.5 * (x_edges[1:] + x_edges[:-1])
                    y_mid = 0.5 * (y_edges[1:] + y_edges[:-1])
                    try:
                        # thresholds [95 %, 68 %], ascending; an edge just above the peak closes the inner band
                        levels = density_levels(smoothed, (0.95, 0.68))
                        fill_edges = levels + [smoothed.max() * 1.01]
                        fill_colors = [to_rgba(color, 0.12 if multi else 0.25),
                                       to_rgba(color, 0.30 if multi else 0.55)]
                        ax.contourf(x_mid, y_mid, smoothed, levels=fill_edges, colors=fill_colors)
                        ax.contour(x_mid, y_mid, smoothed, levels=levels, colors=[color],
                                   linewidths=[0.8, 1.3])
                    except ValueError:
                        pass                       # levels not increasing (e.g. one occupied bin): no regions
            if i == j:
                # the flat prior: density 1/width over the range and a grey bar under its central 68 % interval
                width = HIGH[i] - LOW[i]
                peak = max(diag_top, 1 / width)
                ax.hlines(1 / width, LOW[i], HIGH[i], color='0.35', lw=1.6, zorder=3,
                          label='prior (flat)' if i == 0 else None)
                if prior_band:
                    ax.fill_between([LOW[i] + 0.16 * width, HIGH[i] - 0.16 * width], 0, 0.06 * peak * 1.15,
                                    color='0.45', alpha=0.8, lw=0, zorder=3,
                                    label='prior central 68 %' if i == 0 else None)
                ax.set_ylim(0, 1.15 * peak)
                if extended:
                    for bound in (LOW[i], HIGH[i]):
                        ax.axvline(bound, color='0.35', lw=1.0, ls=':', zorder=3)
            elif extended:
                ax.add_patch(plt.Rectangle((LOW[j], LOW[i]), HIGH[j] - LOW[j], HIGH[i] - LOW[i], fill=False,
                                           ec='0.35', lw=1.0, ls=':', zorder=3))
            if truth is not None:
                if i == j:
                    ax.axvline(truth[i], color='k', lw=1, ls='--')
                else:
                    ax.plot(truth[j], truth[i], marker='*', color='k', ms=10, mec='w', mew=0.6)
            # axes span the prior range (or `ranges`); tick labels only on the bottom row and the first column
            ax.set_xlim(lo[j], hi[j])
            if i != j:
                ax.set_ylim(lo[i], hi[i])
            else:
                ax.set_yticks([])
            if i == n_par - 1:
                ax.set_xlabel(LABELS[j], fontsize=15)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[i], fontsize=15)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
            if i != j:
                ax.yaxis.set_major_locator(MaxNLocator(4, prune='both'))
    # legend: the entries of the top-left panel (posteriors, prior) with longer prior texts, then explanations
    handles, labels = axes[0, 0].get_legend_handles_labels()
    longer = {'prior (flat)': ('prior (flat; dotted: its bounds)' if extended else
                               'prior (flat; the 2-D axes span its full range)'),
              'prior central 68 %': 'prior central 68 % interval (grey bar)'}
    labels = [longer.get(text, text) for text in labels]
    if truth is not None:
        handles.append(plt.Line2D([], [], color='k', ls='--', marker='*'))
        labels.append(truth_label)
    if region_legend:
        handles.append(plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.3))
        labels.append('posterior: 68 % (dark) and 95 % (light) regions')
    handles.append(plt.Line2D([], [], color='none'))
    labels.append('above each 1-D panel: median (+/- 68 % interval)'
                  + ("  [68 % width as % of the prior's]" if width_percent else ''))
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.97, 0.93), fontsize=14, frameon=False)
    if note:
        fig.text(0.585, 0.745, note, fontsize=11.5, va='top', ha='left', color='0.15', linespacing=1.35)
    fig.suptitle(title, fontsize=17, x=0.5, y=0.985)
    # fixed grid, so long tick or axis labels cannot squeeze the panels
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.045, top=0.93, wspace=0.07, hspace=0.07)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fmt(q16, q50, q84):
    """Format a median and its 68 % interval as 'median (+upper / -lower)'; 2 decimals if |median| >= 1, else 3."""
    digits = 2 if abs(q50) >= 1 else 3
    return f'{q50:.{digits}f} (+{q84 - q50:.{digits}f} / -{q50 - q16:.{digits}f})'


def main():
    """Command-line entry point: the two corner figures and the summary table of one method (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results', type=Path, required=True, help='analyze_64k.py results folder')
    parser.add_argument('--method', default='moped_fixed', help='compression method in results.json, e.g. bins40')
    args = parser.parse_args()
    results = json.loads((args.results / 'results.json').read_text())
    # the largest training size of this method; its network seeds are pooled
    runs = [r for r in results if r['method'] == args.method]
    size = max(r['size'] for r in runs)
    best = [r for r in runs if r['size'] == size]
    seeds = [r['seed'] for r in best]
    held = best[0]['heldout']
    # held-out calibration per parameter, averaged over the seeds: 68 % coverage and RMSE / prior-mean RMSE
    cov = np.mean([r['heldout']['coverage68'] for r in best], axis=0)
    rmse = np.mean([r['heldout']['rmse_over_prior_mean_rmse'] for r in best], axis=0)

    def load(label, seed):
        """Posterior samples of observation `label` from network `seed`, shape (n_samples, 9)."""
        return np.load(args.results / f'posterior_{label}_{args.method}_{size}_{seed}.npy')

    # samples of all seeds pooled, per observation
    pooled = {label: np.concatenate([load(label, seed) for seed in seeds])
              for label in ['Battaglia12'] + [f[0] for f in FLAMINGO]}
    corner([(pooled['Battaglia12'], f'posterior (seeds {seeds} pooled)', '#1f77b4')], truth=B12,
           title=f'Battaglia+12 control: posterior from the SO-like spectrum ({args.method}, {size} training '
                 'rows); diagonal: median and 68 % interval',
           path=args.results / f'fig_battaglia12_{args.method}_{size}.png')
    corner([(pooled[label], name, color) for label, name, color in FLAMINGO], truth=B12,
           title=f'FLAMINGO y-maps: effective Battaglia+12 parameters ({args.method}, {size} training rows); '
                 'diagonal: median and 68 % interval',
           path=args.results / f'fig_flamingo_{args.method}_{size}.png')

    header = ('| parameter | prior | held-out RMSE / prior-mean RMSE | held-out 68 % coverage | B12 truth '
              '| B12 posterior | B12 68 % width / prior | truth in 68 % | '
              + ' | '.join(name for _, name, _ in FLAMINGO) + ' |')
    lines = [f'# Preliminary SBI on {size} training rows ({args.method})', '',
             f'Held-out rows: {held["rows"]}. Network seeds: {seeds}. '
             'Posteriors of the seeds are pooled below.', '',
             header,
             '|---|---|---|---|---|---|---|---|' + '---|' * len(FLAMINGO)]
    # 16th, 50th and 84th percentiles per observation, shape (3, 9)
    quant = {label: np.quantile(samples, [.16, .5, .84], axis=0) for label, samples in pooled.items()}
    b12_quant = quant['Battaglia12']
    for i in range(len(NAMES)):
        inside = b12_quant[0, i] <= B12[i] <= b12_quant[2, i]
        # Battaglia12 68 % width over the flat prior's central 68 % width
        width_ratio = (b12_quant[2, i] - b12_quant[0, i]) / (0.68 * (HIGH[i] - LOW[i]))
        cells = [LABELS[i], f'[{LOW[i]:g}, {HIGH[i]:g}]', f'{rmse[i]:.2f}', f'{cov[i]:.2f}', f'{B12[i]:g}',
                 fmt(*b12_quant[:, i]), f'{width_ratio:.2f}', 'yes' if inside else 'no']
        cells += [fmt(*quant[label][:, i]) for label, _, _ in FLAMINGO]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'RMSE below 1 means the posterior mean beats the prior mean on held-out simulations.',
              'Coverage near 0.68 means the 68 % intervals are calibrated on held-out simulations.',
              'FLAMINGO posteriors are effective gNFW parameters at the HalfDome cosmology, not FLAMINGO truth.']
    (args.results / f'summary_{args.method}_{size}.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
