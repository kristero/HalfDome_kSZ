"""Merlin and whitened PCA with and without the prior bound: test-set accuracy, width and calibration per parameter.

Bounded: the networks of results_merlin_unbinned_asinh_p30 (flows in logit space, u = logit((theta - low) / width),
so every posterior sample lies inside the prior box). Unbounded: results_merlin_unbinned_asinh_p30_lin (the same
inputs, rows and training, flows on u = (theta - low) / width, MERLIN_PARAM_SPACE=linear). Both on the 3,000 test
rows with 1,000 samples per row (sbi_final.row_metrics). With posterior_rows/<model>_lin.npz (posterior_rows.py
--truncate) also the unbounded posteriors truncated to the prior box (only the samples with all nine parameters inside).

Measures, per parameter: Pearson r of truth and posterior mean; RMSE of the posterior mean over the RMSE of the
prior centre; median 68 % width over the flat-posterior width (0.6827 of the range); share of rows whose truth lies
inside the marginal 16-84 % interval (0.68 for a calibrated posterior).

    python bounded_vs_unbounded.py
Writes results_corrected_inputs/fig_bounded_vs_unbounded.png and bounded_vs_unbounded.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_corrected_inputs'
FLAT68 = 0.6827
TEX = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_m^{P_0}$', r'$\alpha_m^{x_c}$', r'$\alpha_m^{\beta}$',
       r'$\alpha_z^{P_0}$', r'$\alpha_z^{x_c}$', r'$\alpha_z^{\beta}$']
INK, INK2, GRID, SURFACE = '#222222', '#6b6b6b', '#e6e6e3', '#fcfcfb'
COLOUR = dict(merlin='#4a3aa7', pca='#1baf7a')
NAME = dict(merlin='Merlin', pca='PCA (whitened)')


def from_metrics(m):
    """The four measures of one metrics file (sbi_final.row_metrics)."""
    return dict(r=np.array(m['pearson_r']), rmse=np.array(m['rmse_over_prior_rmse']),
                width=np.array(m.get('median_width68_over_flat', np.array(m['median_width68_over_prior']) / FLAT68)),
                coverage=np.array(m['marginal_coverage68']))


def from_rows(rows, prefix='t'):
    """The four measures from per-row summaries (posterior_rows.py), with the truncated (in-box) statistics."""
    truth, low, high = rows['truth'].astype(float), rows['low'], rows['high']
    width = high - low
    mean, q16, q84 = (rows[f'{prefix}{k}'].astype(float) for k in ('mean', 'q16', 'q84'))
    good = np.all(np.isfinite(mean), axis=1)                 # rows with at least one sample inside the box
    truth, mean, q16, q84 = truth[good], mean[good], q16[good], q84[good]
    prior_rmse = np.sqrt(np.mean(((truth - (low + high) / 2) / width) ** 2, axis=0))
    return dict(r=np.array([np.corrcoef(truth[:, i], mean[:, i])[0, 1] for i in range(9)]),
                rmse=np.sqrt(np.mean(((mean - truth) / width) ** 2, axis=0)) / prior_rmse,
                width=np.median((q84 - q16) / (FLAT68 * width), axis=0),
                coverage=np.mean((truth >= q16) & (truth <= q84), axis=0), rows=int(good.sum()))


def load():
    """{(model, version): measures} and the scalar metrics, for version in bounded / unbounded / truncated."""
    sets, scalars = {}, {}
    for v in ('merlin', 'pca'):
        for version, folder in (('bounded', 'results_merlin_unbinned_asinh_p30'),
                                ('unbounded', 'results_merlin_unbinned_asinh_p30_lin')):
            m = json.loads((HERE / folder / f'metrics_{v}.json').read_text())
            sets[(v, version)] = from_metrics(m)
            scalars[(v, version)] = dict(log_p=m['mean_log_prob_theta'], tarp68=m['tarp68'], tarp95=m['tarp95'])
        p = HERE / 'posterior_rows' / f'{v}_lin.npz'
        if p.exists():
            rows = dict(np.load(p))
            if 'tmean' in rows:
                sets[(v, 'truncated')] = from_rows(rows)
                scalars[(v, 'truncated')] = dict(inside_median=float(np.median(rows['inside'])),
                                                 inside_mean=float(np.mean(rows['inside'])))
    return sets, scalars


def figure(sets, scalars, plt):
    from matplotlib.lines import Line2D
    panels = [('r', 'Pearson r (truth, posterior mean)', None),
              ('rmse', 'RMSE / prior-centre RMSE', 1.0),
              ('width', 'median 68 % width / flat width', 1.0),
              ('coverage', 'marginal 68 % coverage (calibrated: 0.68)', 0.68)]
    versions = [v for v in ('bounded', 'unbounded', 'truncated') if any(k[1] == v for k in sets)]
    style = dict(bounded=dict(mfc='full', marker='o', ms=7.5), unbounded=dict(mfc='none', marker='o', ms=7.5),
                 truncated=dict(mfc='none', marker='D', ms=6))
    order = [(m, v) for m in ('merlin', 'pca') for v in versions if (m, v) in sets]
    step = 0.8 / len(order)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12.5, 13), sharex=True, gridspec_kw=dict(hspace=0.12))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(9)
    for ax, (key, label, reference) in zip(axes, panels):
        for n, (model, version) in enumerate(order):
            s = style[version]
            colour = COLOUR[model]
            ax.plot(x - 0.4 + step * (n + 0.5), sets[(model, version)][key], ls='none', marker=s['marker'], ms=s['ms'],
                    mfc=colour if s['mfc'] == 'full' else SURFACE, mec=colour, mew=1.6, zorder=3)
        if reference is not None:
            ax.axhline(reference, color=INK, lw=0.9, ls=(0, (1, 1.5)), zorder=1)
        for i in range(1, 9):
            ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=0)
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis='y', color=GRID, lw=0.7)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_ylabel(label, color=INK, fontsize=10)
        means = '    '.join(f"{NAME[m].split(' ')[0]}: " + ', '.join(f"{v} {np.mean(sets[(m, v)][key]):.3f}"
                                                                for mm, v in order if mm == m) for m in ('merlin', 'pca'))
        ax.set_title(f'mean over the nine \u2014 {means}', color=INK2, fontsize=8.5, loc='left')
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(TEX, fontsize=11)
    handles = [Line2D([], [], ls='none', marker=style[v]['marker'], ms=style[v]['ms'], mew=1.6, mec=COLOUR[m],
                      mfc=COLOUR[m] if style[v]['mfc'] == 'full' else SURFACE) for m, v in order]
    names = dict(bounded='bounded (logit space, inside the prior)', unbounded='unbounded (linear space)',
                 truncated='unbounded, truncated to the prior box')
    labels = [f'{NAME[m]}, {names[v]}' for m, v in order]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=9.5, frameon=False,
               labelcolor=INK)
    lines = []
    for m in ('merlin', 'pca'):
        b, u = scalars[(m, 'bounded')], scalars[(m, 'unbounded')]
        lines.append(f"{NAME[m]}: log p of the truth {b['log_p']:.2f} bounded vs {u['log_p']:.2f} unbounded; "
                     f"TARP 68/95 % {b['tarp68']:.2f}/{b['tarp95']:.2f} vs {u['tarp68']:.2f}/{u['tarp95']:.2f}")
    fig.suptitle('Merlin and whitened PCA with and without the prior bound: 3,000 test rows, 61,440 training sims, '
                 'asinh inputs\n' + '\n'.join(lines), color=INK, fontsize=10.5, x=0.06, ha='left', y=0.995)
    fig.savefig(OUT / 'fig_bounded_vs_unbounded.png', dpi=130, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def figure_b12(plt):
    """B12 1-D marginals of Merlin bounded vs unbounded on axes past the prior box (prior bounds dotted)."""
    import plot_posteriors as PP
    bounded = np.load(HERE / 'obs_posteriors' / 'B12__Merlin.npy')
    unbounded = np.load(HERE / 'obs_posteriors_lin' / 'B12__Merlin.npy')
    width = PP.HIGH - PP.LOW
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.patch.set_facecolor(SURFACE)
    for i, ax in enumerate(axes.flat):
        lo, hi = PP.LOW[i] - 0.35 * width[i], PP.HIGH[i] + 0.35 * width[i]
        bins = np.linspace(lo, hi, 61)
        ax.hist(bounded[:, i], bins=bins, density=True, histtype='stepfilled', color=COLOUR['merlin'], alpha=0.35,
                lw=0, label='bounded (logit space)')
        ax.hist(unbounded[:, i], bins=bins, density=True, histtype='step', color=COLOUR['merlin'], lw=1.8,
                label='unbounded (linear space)')
        for bound in (PP.LOW[i], PP.HIGH[i]):
            ax.axvline(bound, color=INK2, lw=1, ls=':')
        ax.hlines(1 / width[i], PP.LOW[i], PP.HIGH[i], color=INK2, lw=1.2, label='flat prior')
        ax.axvline(PP.B12[i], color=INK, lw=1, ls='--', label='B12 truth')
        outside = np.mean((unbounded[:, i] < PP.LOW[i]) | (unbounded[:, i] > PP.HIGH[i]))
        ax.set_title(f'{TEX[i]}   (unbounded: {100 * outside:.0f} % outside)', color=INK, fontsize=10, loc='left')
        ax.set_yticks([])
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=9.5, frameon=False,
               labelcolor=INK)
    fig.suptitle('Battaglia+12 mock, Merlin: the same posterior with and without the prior bound (dotted: prior bounds)',
                 color=INK, fontsize=11.5, x=0.06, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / 'fig_b12_bounded_vs_unbounded_marginals.png', dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sets, scalars = load()
    figure(sets, scalars, plt)
    figure_b12(plt)
    record = {f'{m}_{v}': {k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in s.items()}
              for (m, v), s in sets.items()}
    record['scalars'] = {f'{m}_{v}': s for (m, v), s in scalars.items()}
    (OUT / 'bounded_vs_unbounded.json').write_text(json.dumps(record, indent=1) + '\n')
    for (m, v), s in sets.items():
        print(f'{m:6s} {v:9s} r {np.mean(s["r"]):.3f}  rmse {np.mean(s["rmse"]):.3f}  width {np.mean(s["width"]):.3f}  '
              f'coverage68 {np.mean(s["coverage"]):.3f}')
    print(record['scalars'])


if __name__ == '__main__':
    main()
