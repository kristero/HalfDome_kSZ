"""Correlation, mean and width analysis of the corrected input transforms (bug hunt of 2026-09-30, reruns 2026-10-01).

Everything heavy ran on idark; this script reads only the small result files fetched from there.

Two questions, per parameter:
  correlation  does the posterior mean follow the truth?  Pearson r of truth and posterior mean over the test rows
  mean + width is the mean right and has the posterior narrowed?  RMSE of the posterior mean (relative to the RMSE of
               the prior centre, 1 = no better than guessing the centre), and the median 68 % width relative to the
               68 % width of a flat posterior over the prior (0.6827 of the range; 1 = nothing learned)
A decent correlation with a width near 1 is the case "the mean is decent but nothing much was learned".

Sources (relative to this folder):
  results_final/<variant>/N<size>/metrics.json           64k flat prior, median asinh scale (the original study)
  results_final_noise/<variant>/N<size>/metrics.json     the same rows and settings, noise-level asinh scale (rerun)
  results_merlin_unbinned_p30/metrics_<v>.json            all multipoles: Merlin, PCA, MOPED; linear inputs (before)
  results_merlin_unbinned_asinh_p30/metrics_<v>.json      the same with asinh inputs (after); both with early stop 30
  results_merlin_unbinned_asinh_p30_N<size>/metrics_<v>.json, b12_samples_<v>.npy
                                                          the asinh-input comparison trained on the first <size> pool
                                                          rows (MERLIN_TRAIN_ROWS; the nested sets of sbi_final.py)
  posterior_rows/<label>_b12.npy                          B12 posterior samples of the 40-band and MOPED models
  beamfix/fetched/results/series.json                     staged beam-corrected series, both asinh scales
  posterior_rows/<label>.npz                              per-test-row posterior summaries (posterior_rows.py)
Missing sources are skipped, so the script can be rerun as results arrive.

Writes results_corrected_inputs/:
  fig_correlation_vs_N.png, fig_rmse_vs_N.png, fig_width_vs_N.png   per parameter against training-set size for
                                  the five corrected-input compressions: 40 bins, MOPED, PCA, Merlin, no binning
  fig_b12_corner_merlin_pca_61440.png   B12 posterior, Merlin vs whitened PCA at 61,440 rows
  fig_b12_corner_merlin_vs_N.png, fig_b12_corner_pca_vs_N.png   B12 posterior against training-set size
  fig_mean_width_summary.png      the three measures averaged over the nine parameters
  fig_compressions_common_N.png   per parameter at the largest training-set size common to the five
  fig_obs_corner_<obs>.png        per observation (B12, Lee22, FLAMINGO): Merlin vs whitened PCA at 61,440 rows
  fig_obs_marginals_<obs>.png     per observation: the 1-D posteriors of all five compressions
  fig_staged_beam.png             staged series: production vs beamed vs no noise, both asinh scales
  fig_mean_width_binned.png       posterior mean and 68 % interval against the truth, binned over the prior
  fig_mean_width_rows_<label>.png the same per test row (a random subset), one model per figure
  summary.json, summary.md        the numbers behind the figures

    python corrected_inputs_analysis.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_corrected_inputs'
FLAT68 = 0.6827                                   # 68 % width of a flat posterior, in units of the prior range
PARAMS = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
LABELS = [r'$P_0$', r'$x_{\rm c}$', r'$\beta$', r'$\alpha_{m,P_0}$', r'$\alpha_{m,x_{\rm c}}$', r'$\alpha_{m,\beta}$',
          r'$\alpha_{z,P_0}$', r'$\alpha_{z,x_{\rm c}}$', r'$\alpha_{z,\beta}$']

# Colour follows the entity in every figure (validated with the dataviz validator, light surface, all pairs):
# blue / orange / aqua / violet pass; red only appears where orange does not (staged figure), with markers as the
# secondary encoding. The input transform is never colour: solid filled = corrected inputs, dashed open = before.
# The five compressions (corrected inputs only): blue / aqua / violet / red / yellow validated all-pairs (the red-aqua
# pair is in the CVD warn band, so every series also has its own marker). The staged figure reuses red for beamed noise
# and yellow for half-depth; MOPED is not drawn there.
COLOR = dict(bins40='#2a78d6', moped='#eda100', merlin='#4a3aa7', pca='#1baf7a', raw='#e34948',
             clean='#1baf7a', beam='#e34948', half='#eda100')
MARKER = dict(bins40='o', moped='s', merlin='*', pca='P', raw='^', clean='D', beam='^', half='v')
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BEFORE = dict(ls=(0, (4, 2)), lw=1.3, mfc=SURFACE)          # old input transform
AFTER = dict(ls='-', lw=1.8)                                # corrected input transform


# ----------------------------------------------------------------------------- loading

def width_over_flat(m):
    """Median 68 % width over the flat-posterior width, per parameter (older metrics store only the width over the
    prior range)."""
    if 'median_width68_over_flat' in m:
        return np.array(m['median_width68_over_flat'])
    return np.array(m['median_width68_over_prior']) / FLAT68


def measures(m):
    """The three per-parameter measures of one metrics dict: r, RMSE ratio, width over flat (each (9,))."""
    return dict(r=np.array(m['pearson_r']), rmse=np.array(m['rmse_over_prior_rmse']), width=width_over_flat(m))


def convergence(folder, variant):
    """{N: metrics} of one variant of a convergence study folder (missing folder: empty)."""
    out = {}
    for p in sorted((HERE / folder / variant).glob('N*/metrics.json')):
        name = p.parent.name[1:]
        if name.isdigit():
            out[int(name)] = json.loads(p.read_text())
    return out


def load_all():
    """All series of the figures, as {key: dict(label, entity, after, N: [...], metrics: [...])}."""
    series = {}
    for folder, after in (('results_final_noise', True),):              # corrected inputs only (user, 2026-10-01)
        scale = 'noise-level asinh'
        for variant, entity, name in (('bins40_noisy', 'bins40', '40 bins'),
                                      ('bins40_clean', 'clean', '40 bins, no noise'),
                                      ('moped_noisy', 'moped', 'MOPED (9, all multipoles)')):
            byN = convergence(folder, variant)
            if byN:
                series[f'{variant}{"_noise" if after else ""}'] = dict(
                    label=name, entity=entity, after=after, N=sorted(byN),
                    metrics=[byN[n] for n in sorted(byN)])
    # all-multipole compressions at 61,440 rows
    points = {}
    for folder, after in (('results_merlin_unbinned_asinh_p30', True),):     # corrected (asinh) inputs only
        for v in ('merlin', 'pca', 'moped'):
            p = HERE / folder / f'metrics_{v}.json'
            if p.exists():
                points[(v, after)] = json.loads(p.read_text())
    # the asinh-input Merlin / PCA / MOPED comparison against training-set size (61,440: the full-pool run)
    folders = {int(p.name.split('_N')[-1]): p for p in HERE.glob('results_merlin_unbinned_asinh_p30_N*')}
    folders[61440] = HERE / 'results_merlin_unbinned_asinh_p30'
    for v, entity, name in (('merlin', 'merlin', 'Merlin (whitened PCA -> learned head), all multipoles'),
                            ('pca', 'pca', 'plain PCA (whitened), all multipoles'),
                            ('raw', 'raw', 'no binning: all 7,900 multipoles -> learned head')):
        byN = {n: json.loads((f / f'metrics_{v}.json').read_text()) for n, f in folders.items()
               if (f / f'metrics_{v}.json').exists()}
        if byN:
            series[f'{v}_asinh'] = dict(label=name, entity=entity, after=True, N=sorted(byN),
                                        metrics=[byN[n] for n in sorted(byN)], folders={n: folders[n] for n in byN})
    staged = None
    p = HERE / 'beamfix' / 'fetched' / 'results' / 'series.json'
    if p.exists():
        staged = json.loads(p.read_text())
    rows = {}
    for p in sorted((HERE / 'posterior_rows').glob('*.npz')):
        rows[p.stem] = dict(np.load(p))
    return series, points, staged, rows


# ----------------------------------------------------------------------------- plotting helpers

def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
                         'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def style(entity, after, markersize=5):
    """Line and marker style: colour = entity, solid filled = corrected inputs, dashed open = before."""
    base = dict(color=COLOR[entity], marker=MARKER[entity], ms=markersize, mec=COLOR[entity])
    base.update(AFTER if after else BEFORE)
    return base


def grid(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


N_TICKS = [2000, 4000, 8000, 16000, 32000, 61440]


def n_axis(ax):
    """Log x axis labelled at the training-set sizes of the study."""
    from matplotlib.ticker import NullFormatter, NullLocator
    ax.set_xscale('log')
    ax.set_xticks(N_TICKS)
    ax.set_xticklabels(['2k', '4k', '8k', '16k', '32k', '61k'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def legend_below(fig, axes, ncol=3):
    """One legend for the figure, below the panels (handles of the first axis that has any)."""
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=ncol, frameon=False, fontsize=7)
            return


# ----------------------------------------------------------------------------- figures against N

MEASURES = dict(r=('Pearson r (truth, posterior mean)', None),
                rmse=('RMSE of the posterior mean / RMSE of the prior centre', 1.0),
                width=('median 68 % width / flat-posterior width', 1.0))
YLIM = dict(r=(0.2, 0.9), rmse=(0.4, 1.05), width=(0.15, 1.05))   # shared by all panels, so the levels compare
SHORT = dict(r='Pearson r', rmse='RMSE / prior-centre RMSE', width='68 % width / flat')   # per-panel axis labels


def plot_vs_N(series, points, key, path, plt):
    """3 x 3 panels: one measure per parameter against N for every convergence series; the all-multipole compressions
    at 61,440 rows as markers (after only, to keep the panels readable; before/after is fig_compressions)."""
    ylabel, reference = MEASURES[key]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.set_ylim(*YLIM[key])
        for name, s in series.items():
            if s['entity'] == 'clean':
                continue                       # the no-noise bound is in fig_staged_beam (aqua = PCA here)
            values = [measures(m)[key][i] for m in s['metrics']]
            ax.plot(s['N'], values, label=s['label'], **style(s['entity'], s['after']))
        for v, name in (('merlin', 'Merlin, all multipoles, linear inputs (61,440 only)'),
                        ('pca', 'plain PCA, all multipoles, linear inputs (61,440 only)')):
            m = points.get((v, False))
            if m is not None:
                ax.plot([61440], [measures(m)[key][i]], ls='none', label=name,
                        **{k: val for k, val in style(v, False, 7).items() if k not in ('ls', 'lw')})
        if reference is not None:
            ax.axhline(reference, color=INK2, lw=0.8, ls=':')
        n_axis(ax)
        ax.set_title(LABELS[i])
        grid(ax)
    for ax in axes[-1]:
        ax.set_xlabel('training simulations N')
    for ax in axes[:, 0]:
        ax.set_ylabel(SHORT[key], fontsize=8)
    fig.suptitle(f'{ylabel} against training-set size (3,000 test rows, 3-seed ensembles; dotted: no information)'
                 if reference is not None else f'{ylabel} against training-set size (3,000 test rows, 3-seed ensembles)',
                 fontsize=10)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(series, points, path, plt):
    """1 x 3 panels: the three measures averaged over the nine parameters."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, key in zip(axes, MEASURES):
        ylabel, reference = MEASURES[key]
        for name, s in series.items():
            if s['entity'] == 'clean':
                continue
            ax.plot(s['N'], [measures(m)[key].mean() for m in s['metrics']], label=s['label'],
                    **style(s['entity'], s['after']))
        for v, name in (('merlin', 'Merlin, linear inputs (61,440 only)'), ('pca', 'PCA, linear inputs (61,440 only)')):
            m = points.get((v, False))
            if m is not None:
                ax.plot([61440], [measures(m)[key].mean()], ls='none', label=name,
                        **{k: val for k, val in style(v, False, 7).items() if k not in ('ls', 'lw')})
        if reference is not None:
            ax.axhline(reference, color=INK2, lw=0.8, ls=':')
        n_axis(ax)
        ax.set_xlabel('training simulations N')
        ax.set_title(f'{ylabel}\n(mean over the 9 parameters)')
        grid(ax)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_compressions(series, points, path, plt):
    """Per parameter at the largest training-set size that all five compressions have: r (top) and width over flat
    (bottom) for 40 bins, MOPED, PCA, Merlin and no binning (corrected inputs)."""
    keys = [k for k in ('bins40_noisy_noise', 'moped_noisy_noise', 'pca_asinh', 'merlin_asinh', 'raw_asinh')
            if k in series]
    if not keys:
        return
    common = sorted(set.intersection(*(set(series[k]['N']) for k in keys)))
    if not common:
        return
    N = common[-1]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    x = np.arange(9)
    offsets = np.linspace(-0.28, 0.28, len(keys))
    for k, dx in zip(keys, offsets):
        s = series[k]
        mm = measures(s['metrics'][s['N'].index(N)])
        for ax, key in zip(axes, ('r', 'width')):
            ax.plot(x + dx, mm[key], ls='none', label=s['label'],
                    **{kk: v for kk, v in style(s['entity'], True, 8).items() if kk not in ('ls', 'lw')})
    axes[1].axhline(1.0, color=INK2, lw=0.8, ls=':')
    axes[0].set_ylabel('Pearson r (truth, posterior mean)')
    axes[1].set_ylabel('median 68 % width / flat-posterior width')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(LABELS)
    for ax in axes:
        grid(ax)
    fig.suptitle(f'All compressions at {N:,} training simulations, corrected inputs (dotted: nothing learned)',
                 fontsize=10)
    legend_below(fig, axes, ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_staged(staged, path, plt):
    """Staged beam-corrected series (mean over parameters): r, width over flat and the paired log p change against
    production noise, for both asinh scales."""
    if staged is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    names = dict(bins40_noisy=('bins40', 'production noise'), bins40_beam=('beam', 'beamed noise'),
                 bins40_half=('half', 'beamed, half-depth splits'), bins40_clean=('clean', 'no noise'))
    for scale_key, paired_key, after in (('series_noise', 'paired_log_p_noise', True),):   # corrected inputs only
        for variant, (entity, label) in names.items():
            byN = staged.get(scale_key, {}).get(variant, {})
            if not byN:
                continue
            Ns = sorted(int(n) for n in byN)
            scale = 'noise-level' if after else 'median'
            st = style(entity, after)
            axes[0].plot(Ns, [np.mean(byN[str(n)]['pearson_r']) for n in Ns], label=f'40 bins, {label}',
                         **st)
            axes[1].plot(Ns, [np.mean(width_over_flat(byN[str(n)])) for n in Ns], **st)
            pair = staged.get(paired_key, {}).get(f'{variant}-bins40_noisy', {})
            if pair:
                Np = sorted(int(n) for n in pair)
                axes[2].errorbar(Np, [pair[str(n)]['mean'] for n in Np], [pair[str(n)]['sem'] for n in Np], **st)
    axes[2].axhline(0, color=INK2, lw=0.8, ls=':')
    titles = ['Pearson r (mean over parameters)', '68 % width / flat (mean over parameters)',
              'log p change vs production noise [nats]\n(paired, same test rows; bars: row sampling only)']
    for ax, t in zip(axes, titles):
        n_axis(ax)
        ax.set_xlabel('training simulations N')
        ax.set_title(t)
        grid(ax)
    legend_below(fig, axes, ncol=4)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- mean + width per test row

def unit(rows, key):
    """A per-row quantity of a posterior_rows file in prior units (0 = lower bound, 1 = upper bound)."""
    return (rows[key] - rows['low']) / (rows['high'] - rows['low'])


ROW_MODELS = dict(noise_bins40_noisy=('bins40', True, '40 bins'),
                  noise_moped=('moped', True, 'MOPED (9, all multipoles)'),
                  pca_asinh=('pca', True, 'plain PCA (whitened), all multipoles'),
                  merlin_asinh=('merlin', True, 'Merlin (whitened PCA -> learned head)'),
                  raw_32000=('raw', True, 'no binning: 7,900 multipoles -> learned head (N = 32,000)'))


def plot_binned(rows, path, plt, models=('noise_bins40_noisy', 'noise_moped', 'pca_asinh', 'merlin_asinh', 'raw_32000'),
                bins=10):
    """Per parameter: in bins of the true value, the median posterior mean (line) and the median 68 % interval (band),
    in prior units. A posterior that learned nothing sits on the flat-prior reference (0.5, band 0.16-0.84); a perfect
    one on the diagonal with a vanishing band."""
    present = [m for m in models if m in rows]
    if not present:
        return
    edges = np.linspace(0, 1, bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for i, ax in enumerate(axes.flat):
        ax.axhspan(0.16, 0.84, color=GRID, alpha=0.6, lw=0)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=':')
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
        for name in present:
            entity, after, label = ROW_MODELS[name]
            r = rows[name]
            t, mu = unit(r, 'truth')[:, i], unit(r, 'mean')[:, i]
            lo, hi = unit(r, 'q16')[:, i], unit(r, 'q84')[:, i]
            which = np.digitize(t, edges[1:-1])
            med = np.array([np.median(mu[which == b]) for b in range(bins)])
            band_lo = np.array([np.median(lo[which == b]) for b in range(bins)])
            band_hi = np.array([np.median(hi[which == b]) for b in range(bins)])
            st = style(entity, after, 4)
            ax.fill_between(centres, band_lo, band_hi, color=COLOR[entity], alpha=0.16 if after else 0.08, lw=0)
            ax.plot(centres, med, label=label, **st)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(LABELS[i])
        grid(ax)
    for ax in axes[-1]:
        ax.set_xlabel('true value (prior units)')
    for ax in axes[:, 0]:
        ax.set_ylabel('posterior mean (prior units)')
    fig.suptitle('Posterior mean (line) and median 68 % interval (band) against the truth, 61,440 training simulations\n'
                 'grey: flat-prior reference (nothing learned), diagonal: perfect recovery', fontsize=10)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rows(rows, name, path, plt, n=250):
    """Per parameter, for n random test rows: the posterior mean and its 68 % interval against the truth."""
    entity, after, label = ROW_MODELS[name]
    r = rows[name]
    pick = np.random.default_rng(0).choice(len(r['truth']), size=min(n, len(r['truth'])), replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for i, ax in enumerate(axes.flat):
        t, mu = unit(r, 'truth')[pick, i], unit(r, 'mean')[pick, i]
        lo, hi = unit(r, 'q16')[pick, i], unit(r, 'q84')[pick, i]
        ax.axhspan(0.16, 0.84, color=GRID, alpha=0.6, lw=0)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8)
        ax.errorbar(t, mu, [mu - lo, hi - mu], ls='none', color=COLOR[entity], alpha=0.35, lw=0.7, capsize=0)
        ax.plot(t, mu, ls='none', marker='o', ms=2.5, color=COLOR[entity])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(LABELS[i])
        grid(ax)
    for ax in axes[-1]:
        ax.set_xlabel('true value (prior units)')
    for ax in axes[:, 0]:
        ax.set_ylabel('posterior mean +- 68 % (prior units)')
    fig.suptitle(f'{label}: {len(pick)} random test rows, 61,440 training simulations\n'
                 'grey band: flat-prior 68 % interval, diagonal: perfect recovery', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def row_table(rows):
    """Per model and parameter, from the per-row summaries: r, bias and RMSE of the mean (prior units), median width
    over flat, the fraction of rows narrower than half the flat width, and the 68 / 95 % coverage."""
    table = {}
    for name, r in rows.items():
        t, mu = unit(r, 'truth'), unit(r, 'mean')
        width = (unit(r, 'q84') - unit(r, 'q16')) / FLAT68
        cov68 = np.mean((r['truth'] >= r['q16']) & (r['truth'] <= r['q84']), axis=0)
        cov95 = np.mean((r['truth'] >= r['q025']) & (r['truth'] <= r['q975']), axis=0)
        table[name] = dict(pearson_r=[float(np.corrcoef(t[:, i], mu[:, i])[0, 1]) for i in range(9)],
                           bias=(mu - t).mean(0).tolist(), rmse=np.sqrt(((mu - t) ** 2).mean(0)).tolist(),
                           median_width_over_flat=np.median(width, 0).tolist(),
                           fraction_narrower_than_half_flat=(width < 0.5).mean(0).tolist(),
                           coverage68=cov68.tolist(), coverage95=cov95.tolist())
    return table


# ----------------------------------------------------------------------------- B12 corner plots

# training-set size as an ordinal light -> dark ramp (the documented blue sequential steps 250 / 400 / 550 / 700)
N_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
CORNER_N = [2000, 8000, 32000, 61440]


def b12_samples(path, n=20000):
    """Up to n B12 posterior samples (theta units) from a .npy file, or None if it is missing."""
    if not path.exists():
        return None
    s = np.load(path)
    return s[np.random.default_rng(0).permutation(len(s))[:n]] if len(s) > n else s


def plot_b12_corners(series):
    """B12 corners: the four corrected models at 61,440 rows; Merlin and PCA against training-set size."""
    import plot_posteriors as PP
    sets = []
    full = HERE / 'results_merlin_unbinned_asinh_p30'
    for path, label, colour in ((full / 'b12_samples_merlin.npy', 'Merlin (all multipoles, asinh inputs)', COLOR['merlin']),
                                (full / 'b12_samples_pca.npy', 'plain PCA (all multipoles, asinh inputs)', COLOR['pca']),
                                (HERE / 'posterior_rows' / 'noise_moped_b12.npy', 'MOPED (noise-level scale)', COLOR['moped']),
                                (HERE / 'posterior_rows' / 'noise_bins40_noisy_b12.npy', '40 bins (noise-level scale)',
                                 COLOR['bins40'])):
        samples = b12_samples(path)
        if samples is not None:
            sets.append((samples, label, colour))
    if sets:
        two = [x for x in sets if x[1].startswith(('Merlin', 'plain PCA'))]
        if len(two) == 2:
            PP.corner(two, truth=PP.B12, title='Battaglia+12 SO mock, 61,440 training simulations, 3-seed ensembles, '
                      'all multipoles, asinh inputs: Merlin vs plain PCA', path=OUT / 'fig_b12_corner_merlin_pca_61440.png')
    for v, name in (('merlin', 'Merlin'), ('pca', 'plain PCA')):
        s = series.get(f'{v}_asinh')
        if not s:
            continue
        sets = []
        for N, colour in zip(CORNER_N, N_RAMP):
            folder = s['folders'].get(N)
            samples = b12_samples(folder / f'b12_samples_{v}.npy') if folder else None
            if samples is not None:
                sets.append((samples, f'{name}, N = {N:,}', colour))
        if len(sets) > 1:
            PP.corner(sets, truth=PP.B12, title=f'Battaglia+12 SO mock: {name} (all multipoles, asinh inputs) against '
                      'training-set size (colour: N, light = small)', path=OUT / f'fig_b12_corner_{v}_vs_N.png')


# ----------------------------------------------------------------------------- observation posteriors

OBS = dict(B12=('Battaglia+12 SO mock', 'Battaglia+12 values'),
           Lee22_corrected=('Lee22 parameters, pressure (1+z)^2-corrected: frozen-engine SO mock',
                            'Lee22 parameter point (truth)'),
           Lee22_printed=('Lee22 parameters as printed: frozen-engine SO mock', 'Lee22 parameter point (truth)'),
           # the FLAMINGO hydro tSZ maps themselves (flamingo_tsz, FLAMINGO cosmology) through the same beam, mask and
           # SO noise; the mark is the 9-parameter spectral fit of flamingo_cosmology_refit, not a profile measurement
           L1_m9=('FLAMINGO L1_m9 (fiducial) hydro tSZ map, same beam, mask and SO noise',
                  'gNFW spectral fit to the FLAMINGO map (reference, not a truth)'),
           **{'Mstar-1sigma': ('FLAMINGO M*-1sigma hydro tSZ map, same beam, mask and SO noise',
                               'gNFW spectral fit to the FLAMINGO map (reference, not a truth)'),
              'fgas-8sigma': ('FLAMINGO fgas-8sigma hydro tSZ map, same beam, mask and SO noise',
                              'gNFW spectral fit to the FLAMINGO map (reference, not a truth)')},
           Lee22_full=('Full corrected Lee22 model map on HalfDome, the FLAMINGO observation operator',
                       'Lee22 9-parameter point (reference, not a truth)'))
OBS_MODELS = [('Merlin', 'merlin', 'Merlin'), ('raw', 'raw', 'no binning (7,900 multipoles)'), ('40_bins', 'bins40', '40 bins'),
              ('PCA', 'pca', 'PCA (whitened)'), ('MOPED', 'moped', 'MOPED')]
CORNER_MODELS = ('Merlin', 'PCA')            # the best two (user, 2026-10-01): Merlin and whitened PCA, 61,440 rows
FLAMINGO_FITS = HERE.parent / 'flamingo_linear_prior' / 'cluster_results' / 'inputs' / 'flamingo_fits.json'


RELAXED = HERE / 'obs_posteriors_lin'       # Merlin / PCA trained without the prior box (MERLIN_PARAM_SPACE=linear)
NICE = ['P0', 'xc', 'β', 'α_m,P0', 'α_m,xc', 'α_m,β', 'α_z,P0', 'α_z,xc', 'α_z,β']   # short names for figure notes


def obs_mark(name, summary, fits):
    """An observation's marker: its truth, the FLAMINGO spectral fit, or (full Lee22 map) the 9-parameter Lee22 point."""
    info = summary['observations'][name]
    if info['truth'] is not None:
        return np.array(info['truth'])
    if name == 'Lee22_full' and 'Lee22_corrected' in summary['observations']:
        return np.array(summary['observations']['Lee22_corrected']['truth'])
    return np.array(fits[name]) if name in fits else None


def relaxed_ranges(sets, PP, pad=0.04, cap=1.0):
    """(9, 2) axis ranges: the prior box, widened on each side the central 99 % of the samples pass, by `pad` prior
    widths beyond those samples and at most `cap` prior widths beyond the box."""
    width = PP.HIGH - PP.LOW
    q_lo, q_hi = np.quantile(np.concatenate([s for s, _, _ in sets]), [0.005, 0.995], axis=0)
    lo = np.where(q_lo < PP.LOW, np.maximum(q_lo - pad * width, PP.LOW - cap * width), PP.LOW)
    hi = np.where(q_hi > PP.HIGH, np.minimum(q_hi + pad * width, PP.HIGH + cap * width), PP.HIGH)
    return np.stack([lo, hi], axis=1)


def plot_obs_relaxed():
    """Per observation: the corner of Merlin and whitened PCA trained without the prior box (obs_posteriors_lin),
    axes widened to the samples, the prior bounds dotted, and the share of samples outside the box. Returns the
    numbers ({observation: {model: outside shares, medians}})."""
    import plot_posteriors as PP
    if not (RELAXED / 'summary.json').exists():
        return {}
    summary = json.loads((RELAXED / 'summary.json').read_text())
    fits = json.loads(FLAMINGO_FITS.read_text())['corrected_parameters'] if FLAMINGO_FITS.exists() else {}
    table = {}
    for name in summary['observations']:
        title, mark_label = OBS.get(name, (name, 'truth'))
        sets = [(np.load(RELAXED / f'{name}__{key}.npy'), label, COLOR[entity]) for key, entity, label in OBS_MODELS
                if key in CORNER_MODELS and (RELAXED / f'{name}__{key}.npy').exists()]
        if not sets:
            continue
        lines, table[name] = ['samples beyond a prior bound (↑ above, ↓ below; ≥ 2 % shown):'], {}
        for s, label, _ in sets:
            below, above = s < PP.LOW, s > PP.HIGH
            out = below | above
            table[name][label] = dict(outside_any=float(out.any(1).mean()), outside_per_parameter=out.mean(0).tolist(),
                                      below_per_parameter=below.mean(0).tolist(), above_per_parameter=above.mean(0).tolist(),
                                      median=np.median(s, 0).tolist(), q16=np.quantile(s, .16, 0).tolist(),
                                      q84=np.quantile(s, .84, 0).tolist())
            entries = sorted([(share, f'{NICE[i]} {arrow}{100 * share:.0f} %') for i in range(9)
                              for share, arrow in ((above[:, i].mean(), '↑'), (below[:, i].mean(), '↓')) if share >= 0.02],
                             reverse=True)
            lines.append(f'{label}: {100 * out.any(1).mean():.0f} % beyond any bound')
            for start in range(0, len(entries), 4):                  # four entries per line
                lines.append('    ' + ',  '.join(text for _, text in entries[start:start + 4]))
        PP.corner(sets, truth=obs_mark(name, summary, fits), truth_label=mark_label, smooth=1.0,
                  title=f'{title}: Merlin vs whitened PCA, no prior bound, 61,440 training sims',
                  path=OUT / f'fig_obs_corner_relaxed_{name}.png', ranges=relaxed_ranges(sets, PP),
                  note='\n'.join(lines))
    return table


def plot_obs(plt):
    """Per observation: a corner of Merlin and whitened PCA, and the 1-D marginals of all five compressions."""
    import plot_posteriors as PP
    folder = HERE / 'obs_posteriors'
    if not (folder / 'summary.json').exists():
        return
    summary = json.loads((folder / 'summary.json').read_text())
    fits = json.loads(FLAMINGO_FITS.read_text())['corrected_parameters'] if FLAMINGO_FITS.exists() else {}
    for name, info in summary['observations'].items():
        title, mark_label = OBS.get(name, (name, 'truth'))
        mark = obs_mark(name, summary, fits)
        loaded = {}
        for key, entity, label in OBS_MODELS:
            path = folder / f'{name}__{key}.npy'
            if path.exists():
                loaded[key] = (np.load(path), label, COLOR[entity])
        sets = [loaded[k] for k in CORNER_MODELS if k in loaded]
        if sets:
            PP.corner(sets, truth=np.array(mark) if mark is not None else None, truth_label=mark_label,
                      title=f'{title}: Merlin vs whitened PCA, 61,440 training sims',
                      path=OUT / f'fig_obs_corner_{name}.png')
        # 1-D marginals of all five compressions (line style as the secondary encoding)
        fig, axes = plt.subplots(3, 3, figsize=(11, 8.5))
        styles = ['-', (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1))]
        for i, ax in enumerate(axes.flat):
            for (key, entity, label), ls in zip(OBS_MODELS, styles):
                if key in loaded:
                    ax.hist(loaded[key][0][:, i], bins=40, range=(PP.LOW[i], PP.HIGH[i]), density=True,
                            histtype='step', color=COLOR[entity], ls=ls, lw=1.5, label=label)
            ax.axhline(1 / (PP.HIGH[i] - PP.LOW[i]), color=INK2, lw=0.8, ls=':', label='flat prior')
            if mark is not None:
                ax.axvline(mark[i], color=INK, lw=1, ls='--', label=mark_label)
            ax.set_xlim(PP.LOW[i], PP.HIGH[i])
            ax.set_yticks([])
            ax.set_title(LABELS[i])
            grid(ax)
        fig.suptitle(f'{title}\n1-D posteriors of all five compressions, corrected inputs', fontsize=10)
        legend_below(fig, axes, ncol=4)
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(OUT / f'fig_obs_marginals_{name}.png', dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------- report

FIVE = (('bins40_noisy_noise', '40 bins'), ('moped_noisy_noise', 'MOPED (9, all multipoles)'),
        ('pca_asinh', 'plain PCA (whitened), all multipoles'), ('merlin_asinh', 'Merlin (whitened PCA -> learned head)'),
        ('raw_asinh', 'no binning (7,900 multipoles -> learned head)'))


def write_summary(series, points, staged, rows, table):
    """summary.json (all numbers) and summary.md: the five corrected-input compressions at 61,440 rows (no binning at
    its largest training set) and at the common 32,000 rows."""
    summary = dict(convergence={k: dict(label=series[k]['label'], N=series[k]['N'],
                                        r=[measures(m)['r'].tolist() for m in series[k]['metrics']],
                                        rmse=[measures(m)['rmse'].tolist() for m in series[k]['metrics']],
                                        width=[measures(m)['width'].tolist() for m in series[k]['metrics']],
                                        log_p=[m['mean_log_prob_theta'] for m in series[k]['metrics']])
                                for k, _ in FIVE if k in series},
                   per_row=table)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    lines = ['# Corrected input transforms: 40 bins vs MOPED vs plain PCA vs Merlin vs no binning', '',
             'r: Pearson r of truth and posterior mean. RMSE: of the posterior mean over the RMSE of the prior centre '
             '(1 = no better than the centre). width: median 68 % width over the flat-posterior width (1 = nothing '
             'learned). log p: mean log posterior density of the true parameters (joint, 9 parameters; higher is '
             'better). TARP: expected coverage of the 68 / 95 % regions. 3,000 test rows, 3-seed ensembles; inputs '
             'asinh-scaled at the noise level.', '']
    for N in (61440, 32000):
        entries = []
        for key, name in FIVE:
            s = series.get(key)
            if not s:
                continue
            if N in s['N']:
                entries.append((name, s['metrics'][s['N'].index(N)]))
            elif N == 61440:                       # not trained at the full pool yet: its largest training set
                entries.append((f'{name} (N = {max(s["N"]):,})', s['metrics'][-1]))
        lines += [f'## {N:,} training simulations', '', '| model | log p | TARP 68 | TARP 95 | mean r | mean RMSE | '
                  'mean width |', '|---|---|---|---|---|---|---|']
        for name, m in entries:
            mm = measures(m)
            lines.append(f"| {name} | {m['mean_log_prob_theta']:.2f} | {m['tarp68']:.2f} | {m['tarp95']:.2f} | "
                         f"{mm['r'].mean():.3f} | {mm['rmse'].mean():.3f} | {mm['width'].mean():.3f} |")
        lines.append('')
        for key in ('r', 'rmse', 'width'):
            lines += [f'### {key}', '', '| model | ' + ' | '.join(PARAMS) + ' | mean |', '|---|' + '---|' * 10]
            for name, m in entries:
                mm = measures(m)
                lines.append(f'| {name} | ' + ' | '.join(f'{x:.2f}' for x in mm[key]) + f' | {mm[key].mean():.3f} |')
            lines.append('')
    if table:
        lines += ['## Per-row coverage of the 68 % / 95 % intervals and fraction of rows narrower than half the flat '
                  'width', '', '| model | coverage68 (mean) | coverage95 (mean) | narrower than half flat (mean) |',
                  '|---|---|---|---|']
        for name, t in table.items():
            lines.append(f"| {ROW_MODELS[name][2]} | {np.mean(t['coverage68']):.3f} | {np.mean(t['coverage95']):.3f} | "
                         f"{np.mean(t['fraction_narrower_than_half_flat']):.3f} |")
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    plt = setup()
    series, points, staged, rows = load_all()
    for key, name in (('r', 'correlation'), ('rmse', 'rmse'), ('width', 'width')):
        plot_vs_N(series, points, key, OUT / f'fig_{name}_vs_N.png', plt)
    plot_summary(series, points, OUT / 'fig_mean_width_summary.png', plt)
    plot_compressions(series, points, OUT / 'fig_compressions_common_N.png', plt)
    plot_staged(staged, OUT / 'fig_staged_beam.png', plt)
    plot_b12_corners(series)
    plot_obs(plt)
    relaxed = plot_obs_relaxed()
    if relaxed:
        (OUT / 'relaxed_summary.json').write_text(json.dumps(relaxed, indent=1) + '\n')
    plot_binned(rows, OUT / 'fig_mean_width_binned.png', plt)
    for name in rows:
        if name in ROW_MODELS:
            plot_rows(rows, name, OUT / f'fig_mean_width_rows_{name}.png', plt)
    table = row_table({k: v for k, v in rows.items() if k in ROW_MODELS})     # the five compressions
    write_summary(series, points, staged, rows, table)
    print(f'series: {sorted(series)}; compressions: {sorted(points)}; per-row models: {sorted(rows)} -> {OUT}')


if __name__ == '__main__':
    main()
