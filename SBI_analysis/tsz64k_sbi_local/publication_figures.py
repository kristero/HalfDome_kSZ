"""Publication figures: the unbounded Merlin, whitened PCA and MOPED posteriors (61,440 training simulations).

  fig_pub_corner_B12          posteriors of the Battaglia+12 SO mock (input values marked)
  fig_pub_corner_FLAMINGO     posteriors of the FLAMINGO L1_m9 (fiducial) tSZ map through the same beam, mask and SO
                              noise; marked: the nine coefficients fitted to the stacked FLAMINGO pressure profiles
                              (L2p8_m9, fiducial physics, joint fit z = 0-3; halfdome_tsz_comparison/flamingo_joint_fit.toml)
  fig_pub_corner_B12_merlin_bounded_vs_unbounded   Merlin on the B12 mock, trained inside the prior box (logit
                              space) and without it (linear space)
  fig_pub_corner_B12_fisher_merlin   the Fisher forecast at B12 (all 7,900 multipoles, times the prior box;
                              fisher_forecast_b12.py, run on idark, results in results_fisher_b12/) against Merlin on
                              the B12 mock, bounded and unbounded; fig_pub_corner_B12_fisher_merlin_pca adds the Fisher
                              forecast of Merlin's own inputs (its 75 PCA components)
  fig_pub_corner_B12_fisher_merlin_beam[_pca]   the same with the beam-consistent noise (beamfix) in the forecast
                              (results_fisher_b12_beam/) and in Merlin, whose inputs are whitened with the covariance
                              of 420 beam-consistent B12 noise realizations (merlin_w420_prepare.py,
                              results_merlin_unbinned_beam_w420_asinh_p30[_lin])
  fig_pub_sbc                 simulation-based calibration on the 3,000 test simulations: ECDF of the rank of the
                              truth among 1,000 posterior samples minus the uniform ECDF, per parameter, with the 95 %
                              simultaneous band (Kolmogorov-Smirnov); fig_pub_sbc_bounded the same for the networks
                              trained inside the prior box
  fig_pub_dataset_size        mean over the nine parameters of Pearson r, RMSE / prior-centre RMSE and median 68 %
                              width / flat width against the number of training simulations
  fig_pub_dataset_size_<measure>  the same per parameter
Networks: results_merlin_unbinned_asinh_p30_lin (MERLIN_PARAM_SPACE=linear; noise-level asinh inputs; MOPED on the
noise-level features of moped_noise_features.py), the size series results_merlin_unbinned_asinh_p30_lin_N<N>.
Posterior samples: obs_posteriors_lin/, SBC ranks: posterior_rows/sbc_<variant>[_lin].npz (posterior_rows.py --ranks).
Each figure is written as PDF (vector) and PNG (300 dpi) to results_publication/.

    python publication_figures.py [--only fisher] [--fisher-dir DIR] [--out DIR]     --only fisher: the Fisher corners
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results_publication'
LOW = np.array([1.0, 0.025, 2.8, -0.6, -1.0, -0.2, -6.0, -1.5, -0.5])
HIGH = np.array([60.0, 4.0, 16.0, 1.5, 0.4, 0.4, 0.5, 3.0, 2.0])
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
TEX = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_m^{P_0}$', r'$\alpha_m^{x_c}$', r'$\alpha_m^{\beta}$',
       r'$\alpha_z^{P_0}$', r'$\alpha_z^{x_c}$', r'$\alpha_z^{\beta}$']
METHODS = [('Merlin', 'merlin', 'Merlin', '#4a3aa7'), ('PCA', 'pca', 'PCA', '#1baf7a'), ('MOPED', 'moped', 'MOPED', '#eda100')]
INK, INK2, GRID = '#1f1f1f', '#6b6b6b', '#e8e8e5'
LIN = HERE / 'results_merlin_unbinned_asinh_p30_lin'
FLAT68 = 0.6827
SIZES = [2000, 4000, 8000, 16000, 32000, 61440]
FISHER = HERE / 'results_fisher_b12'                   # fisher_forecast_b12.py output (idark_fisher/, run on idark)
FISHER_GREY = '#5f5f5c'
PROFILE_FIT = HERE.parent.parent / 'flamingo_pressure_calibration' / 'halfdome_tsz_comparison' / 'flamingo_joint_fit.toml'
KEYS = ['P0_amp', 'x_c_amp', 'beta_amp', 'P0_alpha_m', 'x_c_alpha_m', 'beta_alpha_m', 'P0_alpha_z', 'x_c_alpha_z',
        'beta_alpha_z']                                    # the engine's parameter order


def profile_fit():
    """The nine coefficients of the stacked FLAMINGO pressure-profile fit, in the engine's order."""
    values = {}
    for line in PROFILE_FIT.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = (part.strip() for part in line.split('=', 1))
            values[key] = float(value)
    return np.array([values[k] for k in KEYS])


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 9, 'axes.linewidth': 0.6,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.fontsize': 8,
                         'legend.frameon': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})
    return plt


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', pad_inches=0.02)


def ranges_for(sets, pad=0.04, cap=0.6):
    """Axis ranges: the prior box, widened on each side the central 99 % of the samples pass (at most `cap` widths)."""
    width = HIGH - LOW
    q_lo, q_hi = np.quantile(np.concatenate(sets), [0.005, 0.995], axis=0)
    lo = np.where(q_lo < LOW, np.maximum(q_lo - pad * width, LOW - cap * width), LOW)
    hi = np.where(q_hi > HIGH, np.minimum(q_hi + pad * width, HIGH + cap * width), HIGH)
    return lo, hi


def union_ranges(sets):
    """Axis ranges covering ranges_for of each set separately (pooling lets a large set hide a small set's tails)."""
    los, his = zip(*(ranges_for([s]) for s in sets))
    return np.min(los, axis=0), np.max(his, axis=0)


def hpd_levels(h, fractions=(0.95, 0.68)):
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat) / flat.sum()
    return sorted(flat[np.searchsorted(cum, f)] for f in fractions)


DEFAULT_STYLE = dict(fill68=0.22, fill95=None, lines=(0.55, 0.9), ls='-', diag_fill=None, diag_lw=1.0)


def corner(samples, mark, mark_label, name, plt, smooth=(0.8, 1.2), title=None, ranges=None):
    """A plain 9 x 9 corner: 68 % (shaded + line) and 95 % (line) regions, 1-D densities, the marker, prior bounds.

    samples: (s, label, colour) or (s, label, colour, style) per posterior, s (n, 9). style overrides DEFAULT_STYLE:
    fill68 / fill95 the alpha of the shaded 68 % / 95 % regions (None: none), lines the widths of the 95 % and 68 %
    contour lines (None: no lines), ls their line style, diag_fill the alpha of the shaded 1-D density (None: none),
    diag_lw the width of its line (0: none). smooth: the Gaussian smoothing (in bins of 1/44 of the axis range) of the
    1-D and 2-D histograms. title: an optional legend title. The prior-bounds legend entry appears only when an axis
    extends past the prior box (otherwise the axes are the box). ranges: (lo, hi) axis ranges (default: ranges_for of
    all samples pooled; with sets of very different size pass e.g. union_ranges, or the big set hides a small set's
    tails)."""
    from scipy.ndimage import gaussian_filter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from matplotlib.colors import to_rgba
    samples = [(s, label, colour, {**DEFAULT_STYLE, **(rest[0] if rest else {})})
               for s, label, colour, *rest in samples]
    lo, hi = ranges_for([s for s, *_ in samples]) if ranges is None else ranges
    if mark is not None:                                   # a marker outside the axes (e.g. past a prior bound) shown
        pad = 0.04 * (HIGH - LOW)
        lo, hi = np.where(mark < lo, mark - pad, lo), np.where(mark > hi, mark + pad, hi)
    n = 9
    fig, axes = plt.subplots(n, n, figsize=(7.2, 7.2))
    bins = 44
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                top = 0
                for s, label, colour, style in samples:
                    h, edges = np.histogram(s[:, i], bins=bins, range=(lo[i], hi[i]), density=True)
                    h = gaussian_filter(h, smooth[0])
                    x = 0.5 * (edges[1:] + edges[:-1])
                    if style['diag_fill'] is not None:
                        ax.fill_between(x, h, color=to_rgba(colour, style['diag_fill']), lw=0)
                    if style['diag_lw']:
                        ax.plot(x, h, color=colour, lw=style['diag_lw'], ls=style['ls'])
                    top = max(top, h.max())
                ax.set_ylim(0, 1.1 * top)
                ax.set_yticks([])
                for bound in (LOW[i], HIGH[i]):
                    if lo[i] < bound - 1e-12 or hi[i] > bound + 1e-12:
                        ax.axvline(bound, color=INK2, lw=0.5, ls=':')
                if mark is not None:
                    ax.axvline(mark[i], color=INK, lw=0.7, ls='--')
            else:
                for s, label, colour, style in samples:
                    h, xe, ye = np.histogram2d(s[:, j], s[:, i], bins=bins, range=[(lo[j], hi[j]), (lo[i], hi[i])])
                    h = gaussian_filter(h.T, smooth[1])
                    xm, ym = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                    try:
                        l95, l68 = hpd_levels(h)
                    except (ValueError, IndexError):            # no samples in this panel's range
                        continue
                    if not l95 < l68:
                        continue
                    if style['fill95'] is not None:
                        ax.contourf(xm, ym, h, levels=[l95, l68], colors=[to_rgba(colour, style['fill95'])])
                    if style['fill68'] is not None:
                        ax.contourf(xm, ym, h, levels=[l68, h.max() * 1.01], colors=[to_rgba(colour, style['fill68'])])
                    if style['lines']:
                        ls = style['ls'] if isinstance(style['ls'], str) else [style['ls']] * 2   # dash tuples per level
                        ax.contour(xm, ym, h, levels=[l95, l68], colors=[colour], linewidths=list(style['lines']),
                                   linestyles=ls)
                ax.add_patch(plt.Rectangle((LOW[j], LOW[i]), HIGH[j] - LOW[j], HIGH[i] - LOW[i], fill=False,
                                           ec=INK2, lw=0.5, ls=':'))
                if mark is not None:
                    ax.plot(mark[j], mark[i], marker='*', ms=5, color=INK, mec='white', mew=0.4, zorder=5)
                ax.set_ylim(lo[i], hi[i])
            ax.set_xlim(lo[j], hi[j])
            # at most three ticks, none within 10 % of an edge (rotated labels of neighbouring panels collide there)
            margin = 0.1 * (hi[j] - lo[j])
            ticks = MaxNLocator(3).tick_values(lo[j], hi[j])
            ax.set_xticks([t for t in ticks if lo[j] + margin < t < hi[j] - margin])
            if i != j:
                margin = 0.1 * (hi[i] - lo[i])
                ticks = MaxNLocator(3).tick_values(lo[i], hi[i])
                ax.set_yticks([t for t in ticks if lo[i] + margin < t < hi[i] - margin])
            if i == n - 1:
                ax.set_xlabel(TEX[j])
                for t in ax.get_xticklabels():
                    t.set_rotation(45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(TEX[i])
            else:
                ax.set_yticklabels([])
            ax.tick_params(top=False, right=False)
    handles = []
    for _, _, colour, style in samples:
        if style['lines']:
            handles.append(Line2D([], [], color=colour, lw=1.4, ls=style['ls']))
        else:                                                   # shaded only: a swatch of its 68 % shade
            handles.append(Patch(fc=to_rgba(colour, style['fill68'] or style['fill95'] or 0.5), ec='none'))
    labels = [label for _, label, _, _ in samples]
    if mark is not None:
        handles.append(Line2D([], [], color=INK, ls='--', lw=0.8, marker='*', ms=6, mec='white', mew=0.4))
        labels.append(mark_label)
    if np.any(lo < LOW - 1e-12) or np.any(hi > HIGH + 1e-12):
        handles.append(Line2D([], [], color=INK2, ls=':', lw=0.8))
        labels.append('prior bounds')
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.97, 0.95), fontsize=9, title=title,
               title_fontsize=9, alignment='left')
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.99, wspace=0.06, hspace=0.06)
    save(fig, name)
    plt.close(fig)


def sbc(tag, title_free_name, plt):
    """ECDF of the fractional SBC rank minus the uniform ECDF, per parameter, with the 95 % simultaneous KS band."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    loaded = []
    for key, variant, label, colour in METHODS:
        p = HERE / 'posterior_rows' / f'sbc_{variant}{tag}.npz'
        if p.exists():
            d = np.load(p)
            loaded.append((d['rank'] / 1000.0, label, colour))
    if not loaded:
        return None
    rows = len(loaded[0][0])
    band = 1.358 / np.sqrt(rows)                    # 95 % simultaneous (asymptotic Kolmogorov-Smirnov)
    z = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.6), sharex=True, sharey=True)
    worst = {}
    for i, ax in enumerate(axes.flat):
        ax.fill_between(z, -band, band, color='#d9d8d4', lw=0)
        ax.axhline(0, color=INK2, lw=0.5)
        for ranks, label, colour in loaded:
            ecdf = np.searchsorted(np.sort(ranks[:, i]), z, side='right') / rows
            ax.plot(z, ecdf - z, color=colour, lw=1.0)
            worst.setdefault(label, []).append(float(np.max(np.abs(ecdf - z))))
        ax.text(0.04, 0.9, TEX[i], transform=ax.transAxes, fontsize=9, va='top')
        ax.set_xlim(0, 1)
        ax.tick_params(top=False, right=False)
    lim = max(0.06, 1.15 * max(max(v) for v in worst.values()))
    axes[0, 0].set_ylim(-lim, lim)
    for ax in axes[-1]:
        ax.set_xlabel('fractional rank')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'ECDF $-$ uniform')
    handles = [Line2D([], [], color=c, lw=1.4) for _, _, c in loaded] + [Patch(color='#d9d8d4')]
    labels = [label for _, label, _ in loaded] + ['95 % simultaneous band']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8.5)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.09, top=0.94, wspace=0.08, hspace=0.08)
    save(fig, title_free_name)
    plt.close(fig)
    return dict(rows=rows, band=band, max_abs_ecdf_difference=worst)


def measures(m):
    return dict(r=np.array(m['pearson_r']), rmse=np.array(m['rmse_over_prior_rmse']),
                width=np.array(m.get('median_width68_over_flat', np.array(m['median_width68_over_prior']) / FLAT68)))


def size_series():
    """{variant: {N: measures}} of the unbounded networks."""
    series = {}
    for _, variant, _, _ in METHODS:
        byN = {}
        for N in SIZES:
            folder = LIN if N == 61440 else HERE / f'results_merlin_unbinned_asinh_p30_lin_N{N}'
            p = folder / f'metrics_{variant}.json'
            if p.exists():
                byN[N] = measures(json.loads(p.read_text()))
        series[variant] = byN
    return series


def dataset_size(series, plt):
    from matplotlib.ticker import FixedLocator, NullLocator
    panels = [('r', 'Pearson $r$ (truth, posterior mean)', None), ('rmse', 'RMSE / prior-centre RMSE', 1.0),
              ('width', 'median 68 % width / flat width', 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (key, label, ref) in zip(axes, panels):
        for _, variant, name, colour in METHODS:
            byN = series.get(variant, {})
            if byN:
                N = sorted(byN)
                ax.plot(N, [byN[n][key].mean() for n in N], color=colour, lw=1.2, marker='o', ms=3, label=name)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=0.6, ls=':')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(FixedLocator(SIZES))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(['2k', '4k', '8k', '16k', '32k', '61k'])
        ax.set_xlabel('training simulations')
        ax.set_ylabel(label)
        ax.tick_params(top=False, right=False)
        ax.grid(True, color=GRID, lw=0.5)
    axes[2].legend(loc='upper right')                      # the empty upper part of the width panel
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_pub_dataset_size')
    plt.close(fig)
    for key, label in (('r', 'Pearson $r$'), ('rmse', 'RMSE / prior-centre RMSE'), ('width', 'median 68 % width / flat')):
        fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.6), sharex=True)
        for i, ax in enumerate(axes.flat):
            for _, variant, name, colour in METHODS:
                byN = series.get(variant, {})
                if byN:
                    N = sorted(byN)
                    ax.plot(N, [byN[n][key][i] for n in N], color=colour, lw=1.0, marker='o', ms=2.5, label=name)
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(FixedLocator(SIZES))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels(['2k', '4k', '8k', '16k', '32k', '61k'])
            ax.text(0.04, 0.9, TEX[i], transform=ax.transAxes, fontsize=9, va='top')
            ax.tick_params(top=False, right=False)
            ax.grid(True, color=GRID, lw=0.5)
        for ax in axes[-1]:
            ax.set_xlabel('training simulations')
        for ax in axes[:, 0]:
            ax.set_ylabel(label)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=8.5)
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.09, top=0.94, wspace=0.25, hspace=0.12)
        save(fig, f'fig_pub_dataset_size_{key}')
        plt.close(fig)


# the Fisher-versus-Merlin corners: (Fisher run folder, bounded and unbounded Merlin B12 samples, figure name)
FISHER_VERSIONS = {
    # production noise; Merlin whitened with the OAS covariance of 20 B12 draws (results_merlin_unbinned_asinh_p30[_lin])
    'production': (FISHER, HERE / 'obs_posteriors' / 'B12__Merlin.npy', HERE / 'obs_posteriors_lin' / 'B12__Merlin.npy',
                   'fig_pub_corner_B12_fisher_merlin'),
    # beam-consistent noise (beamfix) for the Fisher forecast and for Merlin; Merlin whitened with the covariance model
    # of 420 beam-consistent B12 noise realizations (merlin_w420_prepare.py)
    'beam_w420': (HERE / 'results_fisher_b12_beam',
                  HERE / 'results_merlin_unbinned_beam_w420_asinh_p30' / 'b12_samples_merlin.npy',
                  HERE / 'results_merlin_unbinned_beam_w420_asinh_p30_lin' / 'b12_samples_merlin.npy',
                  'fig_pub_corner_B12_fisher_merlin_beam'),
}


def fisher_corners(plt, version='production'):
    """The Fisher forecast at B12 (Gaussian likelihood of all 7,900 multipoles about the input values, times the prior
    box) against the Merlin posteriors of the B12 mock, bounded and unbounded, for one FISHER_VERSIONS entry; the _pca
    figure adds the Fisher forecast of Merlin's own PCA inputs. Returns the numbers behind the figures."""
    folder, bounded, unbounded, name = FISHER_VERSIONS[version]
    f = np.load(folder / 'fisher_forecast_b12.npz')
    report = json.loads((folder / 'fisher_forecast_b12.json').read_text())
    # the Fisher matrix calibrated on held-out noise realizations (a lower bound on the information), else the model's
    key = 'boxed_forecast_every_multipole_calibrated' if 'boxed_forecast_every_multipole_calibrated' in f.files \
        else 'boxed_forecast'
    fisher = (f[key].astype(np.float64), r'Fisher $\times$ prior, all multipoles', FISHER_GREY,
              dict(fill68=0.55, fill95=0.22, lines=None, diag_fill=0.4, diag_lw=0))
    pca = (f['boxed_forecast_Merlin_75_PCA_inputs'].astype(np.float64), r"Fisher $\times$ prior, Merlin's PCA inputs",
           FISHER_GREY, dict(fill68=None, lines=(0.5, 0.8), ls=(0, (3, 1.5)), diag_lw=0.8))
    merlin = [(np.load(bounded), 'Merlin, bounded', '#4a3aa7', dict(fill68=None, lines=(0.6, 1.0))),
              (np.load(unbounded), 'Merlin, unbounded', '#e34948', dict(fill68=None, lines=(0.6, 1.0)))]
    # the same axes in both figures, each posterior's own extent (the 200,000 Fisher samples inside the prior box would
    # otherwise hide the unbounded Merlin's tails past the bounds)
    ranges = union_ranges([s for s, *_ in [fisher, pca] + merlin])
    corner([fisher] + merlin, B12, 'input values', name, plt, ranges=ranges)
    corner([fisher, pca] + merlin, B12, 'input values', f'{name}_pca', plt, ranges=ranges)
    numbers = {}
    for s, label, *_ in [fisher, pca] + merlin:
        q = np.quantile(s, [0.16, 0.5, 0.84], axis=0)
        numbers[label] = dict(median=q[1].tolist(), q16=q[0].tolist(), q84=q[2].tolist(),
                              width68_over_prior=((q[2] - q[0]) / (HIGH - LOW)).tolist(),
                              inside_prior=float(np.mean(np.all((s >= LOW) & (s <= HIGH), axis=1))))
    numbers['fisher_settings'] = dict(samples=key, derivatives=report['derivatives'], residuals=report['residuals'],
                                      model=report['model'], merlin_samples=[str(bounded), str(unbounded)])
    return numbers


def main():
    import argparse
    global OUT, FISHER
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['fisher'], default=None)
    parser.add_argument('--fisher-dir', type=Path, default=FISHER, help='fisher_forecast_b12.py output folder')
    parser.add_argument('--out', type=Path, default=OUT, help='figure folder')
    args = parser.parse_args()
    OUT, FISHER = args.out, args.fisher_dir
    FISHER_VERSIONS['production'] = (FISHER,) + FISHER_VERSIONS['production'][1:]
    if args.only == 'fisher':
        plt = setup()
        OUT.mkdir(exist_ok=True)
        numbers = {v: fisher_corners(plt, v) for v, (folder, bounded, unbounded, _) in FISHER_VERSIONS.items()
                   if (folder / 'fisher_forecast_b12.npz').exists() and bounded.exists() and unbounded.exists()}
        (OUT / 'publication_numbers_fisher.json').write_text(json.dumps(numbers, indent=1) + '\n')
        print('written: the Fisher corners of', list(numbers), '+ publication_numbers_fisher.json')
        return
    plt = setup()
    OUT.mkdir(exist_ok=True)
    record = {}
    for obs, name, mark, mark_label in (('B12', 'fig_pub_corner_B12', B12, 'input values'),
                                        ('L1_m9', 'fig_pub_corner_FLAMINGO', profile_fit(),
                                         'FLAMINGO stacked pressure profiles')):
        samples = [(np.load(HERE / 'obs_posteriors_lin' / f'{obs}__{key}.npy'), label, colour)
                   for key, _, label, colour in METHODS if (HERE / 'obs_posteriors_lin' / f'{obs}__{key}.npy').exists()]
        corner(samples, mark, mark_label, name, plt)
        record[obs] = {label: dict(median=np.median(s, 0).tolist(), q16=np.quantile(s, .16, 0).tolist(),
                                   q84=np.quantile(s, .84, 0).tolist()) for s, label, _ in samples}
    # Merlin on the B12 mock, trained inside the prior box (logit space) and without it (linear space)
    pair = [(np.load(HERE / 'obs_posteriors' / 'B12__Merlin.npy'), 'Merlin, bounded', '#4a3aa7'),
            (np.load(HERE / 'obs_posteriors_lin' / 'B12__Merlin.npy'), 'Merlin, unbounded', '#e34948')]
    corner(pair, B12, 'input values', 'fig_pub_corner_B12_merlin_bounded_vs_unbounded', plt)
    record['fisher'] = {v: fisher_corners(plt, v) for v, (folder, bounded, unbounded, _) in FISHER_VERSIONS.items()
                        if (folder / 'fisher_forecast_b12.npz').exists() and bounded.exists() and unbounded.exists()}
    record['sbc_unbounded'] = sbc('_lin', 'fig_pub_sbc', plt)
    record['sbc_bounded'] = sbc('', 'fig_pub_sbc_bounded', plt)
    series = size_series()
    dataset_size(series, plt)
    record['dataset_size'] = {v: {str(N): {k: x.tolist() for k, x in m.items()} for N, m in byN.items()}
                              for v, byN in series.items()}
    (OUT / 'publication_numbers.json').write_text(json.dumps(record, indent=1) + '\n')
    print('written:', sorted(p.name for p in OUT.iterdir()))
    for v, byN in series.items():
        print(v, {N: round(float(m['r'].mean()), 3) for N, m in byN.items()})


if __name__ == '__main__':
    main()
