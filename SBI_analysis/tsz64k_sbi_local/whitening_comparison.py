"""Merlin (and whitened PCA) with the beam-consistent noise and a fiducial-free noise-only whitening.

Four input sets, the same 61,440 training rows, 3,000 test rows, networks and seeds:
  production noise, B12 whitening        results_merlin_unbinned_asinh_p30            (the published inputs)
  production noise, noise-only whitening results_merlin_unbinned_nnwhite_asinh_p30
  beam-consistent noise, B12 whitening   results_merlin_unbinned_beam_asinh_p30       (+ size series _N<N>)
  beam-consistent noise, noise-only      results_merlin_unbinned_beam_nnwhite_asinh_p30
each trained inside the prior box (flows in logit space, 'bounded') and without it (MERLIN_PARAM_SPACE=linear, the
same folders with _lin, 'unbounded'; idark_jobs/whiten_lin.pbs, merlin_linear.pbs for the production set).
B12 whitening: the covariance of the 20 B12 noise draws at the fixed B12 sky (signal x noise plus noise x noise).
Noise-only whitening: the covariance of the regenerated noise x noise spectra of the 61,440 training-pool rows
(no signal term, no fiducial). Both: OAS-shrunk correlation times the measured variances (merlin_beam_prepare.py).

  fig_pub_merlin_beam_whitening     (a, b) test log p of the truth for the four sets, Merlin and PCA, bounded and
                                    unbounded, error bars the standard error over the three seeds (the unbounded
                                    flow density over all of R^9; light dashes: the same posteriors truncated to the
                                    prior box, log q - log(share inside)); (c) test log p
                                    against the number of training simulations, production vs beam-consistent noise
                                    (B12 whitening, bounded); (d) the noise scatter of the 20 B12 draws over the
                                    noise-only scatter of the pool, in bands (noise_scatter_bands.py, run on idark)
  fig_pub_r_whitening               Pearson r of truth and posterior mean per parameter on the 3,000 test rows,
                                    beam-consistent noise, B12 vs noise-only whitening, bounded and unbounded; lower
                                    panels the difference r(noise-only) - r(B12) with 1 sigma error bars: a paired
                                    bootstrap over the test rows combined with the scatter between training seeds
  fig_pub_corner_B12_merlin_beam_whitening[_unbounded]   Merlin on the B12 mock, beam-consistent noise: B12
                                    whitening against noise-only whitening (bounded; _unbounded)
  fig_pub_corner_B12_merlin_beam_nnwhite_bounded_vs_unbounded   Merlin on the B12 mock, beam-consistent noise with
                                    noise-only whitening, bounded against unbounded
  fig_pub_corner_FLAMINGO[_pixwin]_merlin_beam_whitening[_unbounded]   the FLAMINGO L1_m9 map (obs_beam_posteriors.py,
                                    beam-consistent; _pixwin: without the NSIDE 4096 pixel window), B12 vs noise-only
                                    whitening, marked with the stacked FLAMINGO pressure-profile fit
Per-row posteriors and 300,000-draw B12 posteriors: posterior_rows_whitening/<set>_<space>_<variant>[.npz|_b12.npy]
(posterior_rows.py --seed-means, idark_jobs/whiten_rows.pbs and whiten_lin.pbs).
Writes results_publication/ (PDF and PNG) and results_publication/whitening_comparison.json.

    python whitening_comparison.py
"""
import json

import numpy as np

import publication_figures as PF

HERE = PF.HERE
INK, INK2 = PF.INK, PF.INK2
SETS = [('prod_b12', 'production', 'B12', 'results_merlin_unbinned_asinh_p30'),
        ('prod_nn', 'production', 'noise-only', 'results_merlin_unbinned_nnwhite_asinh_p30'),
        ('beam_b12', 'beam', 'B12', 'results_merlin_unbinned_beam_asinh_p30'),
        ('beam_nn', 'beam', 'noise-only', 'results_merlin_unbinned_beam_nnwhite_asinh_p30')]
SPACES = [('bounded', ''), ('unbounded', '_lin')]
MODELS = [('merlin', 'Merlin', '#4a3aa7'), ('pca', 'PCA', '#1baf7a')]
ROWS = HERE / 'posterior_rows_whitening'
OBS_BEAM = HERE / 'obs_posteriors_beam'                 # obs_beam_posteriors.py (FLAMINGO L1_m9)
BANDS = HERE / 'results_merlin_unbinned_beam' / 'inputs' / 'noise_scatter_bands.json'
WHITENING = HERE / 'results_merlin_unbinned_beam' / 'inputs' / 'prepare_whitening.json'
BOOTSTRAP = 4000
DR_LIMIT = 0.035                        # y range of the r difference panels; points beyond as arrows
SMOOTH = (1.0, 1.6)                     # corner smoothing for the 300,000-draw posteriors (bins)
LOGP = r'test $\langle\log p(\theta_\mathrm{true})\rangle$'


def metrics(folder, variant):
    return json.loads((HERE / folder / f'metrics_{variant}.json').read_text())


def table():
    """{(set, space, variant): the test-set and B12 numbers}."""
    rows = {}
    for key, _, _, folder in SETS:
        for space, suffix in SPACES:
            for variant, _, _ in MODELS:
                m = metrics(folder + suffix, variant)
                rows[(key, space, variant)] = dict(
                    folder=folder + suffix, test_log_p=m['mean_log_prob_theta'],
                    test_log_p_seed_sem=m['test_log_prob_err'], test_log_p_single_seeds=m['test_log_prob_seeds'],
                    test_log_p_rows_sem=m['mean_log_prob_theta_sem'], r=float(np.mean(m['pearson_r'])),
                    rmse_over_prior=float(np.mean(m['rmse_over_prior_rmse'])),
                    width68_over_flat=float(np.mean(m['median_width68_over_flat'])),
                    coverage68=float(np.mean(m['marginal_coverage68'])), tarp68=m['tarp68'], tarp95=m['tarp95'],
                    b12_log_p=m['b12_log_prob_truth'], b12_log_p_seed_sem=m['b12_log_prob_truth_err'],
                    b12_log_p_21_draws=m['b12_log_prob_truth_draws'],
                    b12_log_p_21_draws_sem=m['b12_log_prob_truth_draws_sem'],
                    b12_width68_over_flat=float(np.mean(m['b12_width68_over_prior']) / PF.FLAT68),
                    r_per_parameter=m['pearson_r'])
                npz = ROWS / f'{key}_{space}_{variant}.npz'
                if space == 'unbounded' and npz.exists():
                    # the posterior truncated to the prior box: log q - log(share of q inside), row by row
                    # (a row with none of its 1,000 draws inside is floored at half a draw)
                    inside = np.load(npz)['inside'].astype(np.float64)
                    rows[(key, space, variant)].update(
                        inside_mean=float(inside.mean()), rows_nothing_inside=int(np.sum(inside == 0)),
                        test_log_p_in_box=float(m['mean_log_prob_theta']
                                                - np.mean(np.log(np.maximum(inside, 0.5 / 1000)))))
    return rows


def size_series():
    """{(variant, noise): {N: (log p, seed sem)}} for the B12-whitened production and beam-consistent inputs."""
    series = {}
    for variant, _, _ in MODELS:
        for noise, base in (('production', 'results_merlin_unbinned_asinh_p30'),
                            ('beam', 'results_merlin_unbinned_beam_asinh_p30')):
            series[(variant, noise)] = {N: (lambda m: (m['mean_log_prob_theta'], m['test_log_prob_err']))(
                metrics(base if N == 61440 else f'{base}_N{N}', variant)) for N in PF.SIZES}
    return series


def pearson(truth, mean):
    """Pearson r per parameter along axis -2 (rows): truth, mean (..., rows, 9)."""
    t = truth - truth.mean(-2, keepdims=True)
    m = mean - mean.mean(-2, keepdims=True)
    return (t * m).sum(-2) / np.sqrt((t * t).sum(-2) * (m * m).sum(-2))


def r_difference(a, b, rng):
    """r(b) - r(a) per parameter for two posterior_rows outputs of the same test rows, with its errors.

    Rows: paired bootstrap over the test rows (the same resampled rows for both). Seeds: from each seed's own posterior
    means, sqrt(var_a / n + var_b / n) (sample variances over the n seeds; the two sets are independent trainings).
    """
    assert np.array_equal(a['truth'], b['truth']), 'the two sets do not share the test rows'
    truth = a['truth'].astype(np.float64)
    ma, mb = a['mean'].astype(np.float64), b['mean'].astype(np.float64)
    ra, rb = pearson(truth, ma), pearson(truth, mb)
    boot = np.empty((BOOTSTRAP, 9))
    for k in range(BOOTSTRAP):
        idx = rng.integers(0, len(truth), len(truth))
        boot[k] = pearson(truth[idx], mb[idx]) - pearson(truth[idx], ma[idx])
    sa = pearson(truth[None], a['seed_mean'].astype(np.float64))      # (seeds, 9)
    sb = pearson(truth[None], b['seed_mean'].astype(np.float64))
    seeds = np.sqrt(sa.var(0, ddof=1) / len(sa) + sb.var(0, ddof=1) / len(sb))
    rows = boot.std(0, ddof=1)
    # the mean over the nine parameters (correlated between parameters: from the same resamples and seeds)
    mean_rows = boot.mean(1).std(ddof=1)
    mean_seeds = np.sqrt(sa.mean(1).var(ddof=1) / len(sa) + sb.mean(1).var(ddof=1) / len(sb))
    return dict(r_a=ra, r_b=rb, diff=rb - ra, err_rows=rows, err_seeds=seeds, err=np.hypot(rows, seeds),
                r_a_seeds=sa, r_b_seeds=sb, mean_diff=float((rb - ra).mean()),
                mean_err=float(np.hypot(mean_rows, mean_seeds)), mean_err_rows=float(mean_rows),
                mean_err_seeds=float(mean_seeds))


def style_axes(axes, tags):
    for ax, tag in zip(axes, tags):
        ax.tick_params(top=False, right=False)
        ax.grid(True, color=PF.GRID, lw=0.5)
        ax.text(0.03, 0.97, f'({tag})', transform=ax.transAxes, va='top', fontsize=8)


def figure(rows, series, plt):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    # (a, b) the four input sets, bounded and unbounded, on one scale
    values = [r[k] for r in rows.values() for k in ('test_log_p', 'test_log_p_in_box') if k in r]
    limits = (min(values) - 0.25, max(values) + 0.25)
    for ax, (space, _) in zip(axes[0], SPACES):
        for k, (variant, name, colour) in enumerate(MODELS):
            for g, noise in enumerate(('production', 'beam')):
                for w, whitening in enumerate(('B12', 'noise-only')):
                    key = [s for s, n, wh, _ in SETS if n == noise and wh == whitening][0]
                    r = rows[(key, space, variant)]
                    x = g + (w - 0.5) * 0.36 + (k - 0.5) * 0.12
                    ax.errorbar(x, r['test_log_p'], yerr=r['test_log_p_seed_sem'], fmt='o' if w == 0 else 's',
                                ms=4, color=colour, mfc=colour if w == 0 else 'white', mew=1.0, elinewidth=0.8,
                                capsize=0)
                    if 'test_log_p_in_box' in r:          # the same posterior truncated to the prior box
                        ax.plot(x, r['test_log_p_in_box'], marker='_', ms=8, mew=1.4, color=colour, alpha=0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['production noise', 'beam-consistent noise'])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(*limits)
        ax.set_ylabel(LOGP)
        ax.set_title(f'{space} posteriors', fontsize=8.5)
    handles = [Line2D([], [], ls='none', marker='o', ms=4, color=INK2),
               Line2D([], [], ls='none', marker='s', ms=4, color=INK2, mfc='white')] + \
              [Line2D([], [], ls='none', marker='o', ms=4, color=c) for _, _, c in MODELS]
    handles.append(Line2D([], [], ls='none', marker='_', ms=8, mew=1.4, color=INK2, alpha=0.55))
    fig.legend(handles, ['B12 whitening', 'noise-only whitening'] + [n for _, n, _ in MODELS] +
               ['unbounded, truncated to the prior box'], loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5,
               fontsize=7, handletextpad=0.2, columnspacing=1.2)
    # (c) training-set size
    ax = axes[1, 0]
    for variant, name, colour in MODELS:
        for noise, ls in (('production', '-'), ('beam', (0, (3, 1.5)))):
            byN = series[(variant, noise)]
            ax.plot(PF.SIZES, [byN[N][0] for N in PF.SIZES], color=colour, lw=1.1, ls=ls, marker='o', ms=2.5,
                    mfc=colour if noise == 'production' else 'white', mew=0.8)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(PF.SIZES))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(['2k', '4k', '8k', '16k', '32k', '61k'])
    ax.set_xlabel('training simulations')
    ax.set_ylabel(LOGP)
    handles = [Line2D([], [], color=c, lw=1.1) for _, _, c in MODELS] + \
              [Line2D([], [], color=INK2, lw=1.1, ls='-'), Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 1.5)))]
    ax.legend(handles, [n for _, n, _ in MODELS] + ['production noise', 'beam-consistent noise'], loc='upper left',
              bbox_to_anchor=(0.06, 1.0), fontsize=7)
    # (d) what the B12 whitening adds: the signal x noise scatter at the B12 sky
    ax = axes[1, 1]
    bands = json.loads(BANDS.read_text())['bands']
    ell = np.array([0.5 * (b['ell_lo'] + b['ell_hi']) for b in bands])
    ax.plot(ell, [b['noisy_over_nn'] for b in bands], color=INK, lw=1.1, label='B12 draws, total noise')
    ax.plot(ell, [b['nn_b12_over_nn'] for b in bands], color=INK2, lw=0.9, ls=(0, (3, 1.5)),
            label=r'B12 draws, noise $\times$ noise')
    ax.axhline(1.0, color=INK2, lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_xlim(80, 7980)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylim(0.93, 1.78)
    ax.set_ylabel('noise s.d. / noise-only s.d.')
    ax.legend(loc='upper right', fontsize=7)
    style_axes(axes.flat, 'abcd')
    fig.tight_layout(w_pad=1.2, h_pad=1.0, rect=(0, 0, 1, 0.965))
    PF.save(fig, 'fig_pub_merlin_beam_whitening')
    plt.close(fig)


def figure_r(diffs, plt):
    """Per-parameter r, beam-consistent noise, B12 (filled) vs noise-only (open) whitening, and their difference."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
    x = np.arange(9)
    lo = min(min(d['r_a'].min(), d['r_b'].min()) for d in diffs.values())
    for col, (space, _) in enumerate(SPACES):
        top, bottom = axes[0, col], axes[1, col]
        for k, (variant, name, colour) in enumerate(MODELS):
            d = diffs[(space, variant)]
            off = (k - 0.5) * 0.3
            top.plot(x + off - 0.05, d['r_a'], ls='none', marker='o', ms=4, color=colour)
            top.plot(x + off + 0.05, d['r_b'], ls='none', marker='s', ms=4, color=colour, mfc='white', mew=1.0)
            on = np.abs(d['diff']) <= DR_LIMIT
            bottom.errorbar((x + off)[on], d['diff'][on], yerr=d['err'][on], fmt='o', ms=3.5, color=colour,
                            elinewidth=0.8, capsize=0)
            for i in np.flatnonzero(~on):                      # off scale: an arrow at the edge, the value beside it
                sign = np.sign(d['diff'][i])
                bottom.plot(x[i] + off, sign * 0.88 * DR_LIMIT, marker='^' if sign > 0 else 'v', ms=4.5,
                            color=colour)
                bottom.text(x[i] + off + 0.14, sign * 0.88 * DR_LIMIT, f"{d['diff'][i]:+.3f} $\\pm$ {d['err'][i]:.3f}",
                            fontsize=5.5, va='center', ha='left', color=INK2)
        top.set_ylim(lo - 0.04, 1.0)
        top.set_title(f'{space} posteriors', fontsize=8.5)
        bottom.axhline(0, color=INK2, lw=0.6, ls=':')
        bottom.set_ylim(-DR_LIMIT, DR_LIMIT)
        bottom.set_xticks(x)
        bottom.set_xticklabels(PF.TEX)
        for i in range(1, 9):
            for ax in (top, bottom):
                ax.axvline(i - 0.5, color=PF.GRID, lw=0.5, zorder=0)
    axes[0, 0].set_ylabel(r'Pearson $r$')
    axes[1, 0].set_ylabel(r'$r_\mathrm{noise\,only} - r_\mathrm{B12}$')
    handles = [Line2D([], [], ls='none', marker='o', ms=4, color=INK2),
               Line2D([], [], ls='none', marker='s', ms=4, color=INK2, mfc='white')] + \
              [Line2D([], [], ls='none', marker='o', ms=4, color=c) for _, _, c in MODELS]
    axes[0, 1].legend(handles, ['B12 whitening', 'noise-only whitening'] + [n for _, n, _ in MODELS],
                      loc='lower left', fontsize=7, ncol=2, handletextpad=0.2, columnspacing=0.8)
    for ax, tag, (y, va) in zip(axes.flat, 'abcd', [(0.96, 'top')] * 2 + [(0.04, 'bottom')] * 2):
        ax.tick_params(top=False, right=False)
        ax.grid(True, axis='y', color=PF.GRID, lw=0.5)
        ax.text(0.02, y, f'({tag})', transform=ax.transAxes, va=va, fontsize=8)
    fig.tight_layout(w_pad=1.2, h_pad=0.6)
    PF.save(fig, 'fig_pub_r_whitening')
    plt.close(fig)


def flamingo(plt):
    """Corners of the FLAMINGO L1_m9 posteriors (obs_beam_posteriors.py), B12 vs noise-only whitening, beam-consistent
    noise, bounded and unbounded, as observed and without the pixel window; marked with the stacked-profile fit.
    Returns their marginals and which parameters of the fit fall inside the 68 % intervals."""
    mark = PF.profile_fit()
    record = {}
    for obs, tag, note in (('L1_m9', '', ''), ('L1_m9_pixwin', '_pixwin', ', pixel window removed')):
        for space, suffix in (('bounded', ''), ('unbounded', '_unbounded')):
            files = {s: OBS_BEAM / f'{obs}__{s}_{space}_merlin.npy' for s in ('beam_b12', 'beam_nn')}
            if not all(f.exists() for f in files.values()):
                continue
            x = {s: np.load(f).astype(np.float64) for s, f in files.items()}
            names = dict(beam_b12=('B12 whitening', '#4a3aa7'), beam_nn=('noise-only whitening', '#e34948'))
            PF.corner([(x[s], *names[s]) for s in ('beam_b12', 'beam_nn')], mark, 'FLAMINGO stacked pressure profiles',
                      f'fig_pub_corner_FLAMINGO{tag}_merlin_beam_whitening{suffix}', plt, smooth=SMOOTH,
                      title=f'Merlin, FLAMINGO L1_m9{note},\nbeam-consistent noise, {space}')
            for s, v in x.items():
                q = np.quantile(v, [0.16, 0.5, 0.84], axis=0)
                record[f'{obs}_{s}_{space}'] = dict(
                    q16=q[0].tolist(), median=q[1].tolist(), q84=q[2].tolist(),
                    profile_fit_inside68=((mark >= q[0]) & (mark <= q[2])).tolist(),
                    inside_prior=float(np.mean(np.all((v >= PF.LOW) & (v <= PF.HIGH), axis=1))))
    if record:
        record['profile_fit'] = mark.tolist()
    return record


def load_rows(label):
    return dict(np.load(ROWS / f'{label}.npz'))


def main():
    plt = PF.setup()
    PF.OUT.mkdir(exist_ok=True)
    rows, series = table(), size_series()
    figure(rows, series, plt)
    rng = np.random.default_rng(12345)
    diffs = {(space, variant): r_difference(load_rows(f'beam_b12_{space}_{variant}'),
                                            load_rows(f'beam_nn_{space}_{variant}'), rng)
             for space, _ in SPACES for variant, _, _ in MODELS}
    for (space, variant), d in diffs.items():                 # the ensemble r must be the evaluated one
        for key, s in (('r_a', 'beam_b12'), ('r_b', 'beam_nn')):
            folder = dict((k, f) for k, _, _, f in SETS)[s] + dict(SPACES)[space]
            stored = np.array(metrics(folder, variant)['pearson_r'])
            assert np.max(np.abs(d[key] - stored)) < 2e-3, (space, variant, s, d[key], stored)
    figure_r(diffs, plt)
    b12 = {(s, space): np.load(ROWS / f'{s}_{space}_merlin_b12.npy').astype(np.float64)
           for s in ('prod_b12', 'beam_b12', 'beam_nn') for space, _ in SPACES}
    names = dict(beam_b12=('B12 whitening', '#4a3aa7'), beam_nn=('noise-only whitening', '#e34948'))
    for space, suffix in (('bounded', ''), ('unbounded', '_unbounded')):
        pair = [(b12[(s, space)], *names[s]) for s in ('beam_b12', 'beam_nn')]
        PF.corner(pair, PF.B12, 'input values', f'fig_pub_corner_B12_merlin_beam_whitening{suffix}', plt,
                  smooth=SMOOTH, title=f'Merlin, beam-consistent noise,\n{space} posteriors')
    PF.corner([(b12[('beam_nn', 'bounded')], 'bounded', '#4a3aa7'),
               (b12[('beam_nn', 'unbounded')], 'unbounded', '#e34948')], PF.B12, 'input values',
              'fig_pub_corner_B12_merlin_beam_nnwhite_bounded_vs_unbounded', plt, smooth=SMOOTH,
              title='Merlin, beam-consistent noise,\nnoise-only whitening')
    flamingo_record = flamingo(plt)
    whitening = json.loads(WHITENING.read_text())
    record = dict(
        flamingo_l1_m9=flamingo_record,
        sets={f'{s}_{space}_{v}': r for (s, space, v), r in rows.items()},
        size_series={f'{v}_{noise}': {str(N): dict(test_log_p=lp, seed_sem=e) for N, (lp, e) in byN.items()}
                     for (v, noise), byN in series.items()},
        r_beam_noise_only_minus_b12={f'{space}_{v}': {k: np.asarray(x).tolist() for k, x in d.items()}
                                     for (space, v), d in diffs.items()},
        b12_21_draws_note='for the B12-whitened sets 20 of the 21 draws built the whitening covariance, for the '
                          'noise-only sets none did: not a like-for-like comparison of the whitenings',
        unbounded_log_p_note='test_log_p of the unbounded networks is the untruncated flow density (mass outside '
                             'the prior box counts against it); test_log_p_in_box is the posterior truncated to the '
                             'box (1,000 draws per row for the inside share)',
        r_note=f'paired bootstrap over the 3,000 test rows ({BOOTSTRAP} resamples) in quadrature with the seed term '
               'sqrt(var_a/3 + var_b/3) of the per-seed r (each seed alone, 1,000 draws per row)',
        noise_only_covariance=whitening['noise_only'],
        pca_components={k: s['pca']['n_pca'] for k, s in whitening['sets'].items()},
        b12_covariance_shrinkage=whitening['sets']['beam_b12']['covariance']['oas_correlation_shrinkage'],
        b12_marginals={f'{s}_{space}': dict(samples=int(len(x)), median=np.median(x, 0).tolist(),
                                            q16=np.quantile(x, .16, 0).tolist(), q84=np.quantile(x, .84, 0).tolist(),
                                            inside_prior=float(np.mean(np.all((x >= PF.LOW) & (x <= PF.HIGH), 1))))
                       for (s, space), x in b12.items()})
    (PF.OUT / 'whitening_comparison.json').write_text(json.dumps(record, indent=1) + '\n')
    print(f'{"":30s} log p (seed sem)  1-seed  in-box   r      rmse   width  cov68  tarp68  inside')
    for (s, space, v), r in rows.items():
        print(f"{s + ' ' + space + ' ' + v:30s} {r['test_log_p']:6.2f} ({r['test_log_p_seed_sem']:.2f})  "
              f"{np.mean(r['test_log_p_single_seeds']):6.2f}  {r.get('test_log_p_in_box', np.nan):6.2f}  "
              f"{r['r']:.3f}  {r['rmse_over_prior']:.3f}  {r['width68_over_flat']:.3f}  {r['coverage68']:.3f}  "
              f"{r['tarp68']:.3f}  {r.get('inside_mean', np.nan):.3f}")
    for (space, v), d in diffs.items():
        print(f'r(noise-only) - r(B12), beam, {space} {v}: ' +
              ' '.join(f'{x:+.3f}±{e:.3f}' for x, e in zip(d['diff'], d['err'])) +
              f"   mean {d['mean_diff']:+.4f}±{d['mean_err']:.4f} (rows {d['mean_err_rows']:.4f}, "
              f"seeds {d['mean_err_seeds']:.4f})")


if __name__ == '__main__':
    main()
