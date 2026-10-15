"""Build the pipeline explainer page (results_doc/index.html + results_doc/fig/).

Code excerpts are read from the source files by line range at build time, so they are verbatim and their line
numbers are the current ones; result tables are filled from the stored metrics json files. Figures come from
doc_figures.py (results_doc/*.png) and a few publication figures (results_publication/).

    python doc_build.py
"""
import html
import json
import re
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent / 'tSZ_64k_flat_prior_SO_baseline_deproj0'
OUT = HERE / 'results_doc'
FIG = OUT / 'fig'
SERIES = HERE / 'beamfix' / 'fetched' / 'results' / 'series.json'
MISSING = []


# ----------------------------------------------------------------------------------------------- helpers
def code(path, ranges, lang='python', note=None):
    """A verbatim excerpt of `path` (relative to the repo, or absolute) over line ranges [(a, b), ...]."""
    p = Path(path)
    p = p if p.is_absolute() else HERE / p
    lines = p.read_text().splitlines()
    parts = []
    for k, (a, b) in enumerate(ranges):
        if k:
            parts.append('    ...')
        parts.extend(lines[a - 1:b])
    shown = p.relative_to(HERE.parent.parent) if str(p).startswith(str(HERE.parent.parent)) else p.name
    where = ', '.join(f'{a}–{b}' for a, b in ranges)
    head = f'<div class="code-head"><span class="file">{html.escape(str(shown))}</span>' \
           f'<span class="lines">lines {where}</span></div>'
    body = f'<pre><code class="language-{lang}">{html.escape(chr(10).join(parts))}</code></pre>'
    tail = f'<p class="code-note">{note}</p>' if note else ''
    return f'<figure class="code">{head}{body}{tail}</figure>'


def fig(name, caption, source='doc', wide=False):
    src = (OUT / name) if source == 'doc' else (HERE / 'results_publication' / name)
    if not src.exists():
        MISSING.append(name)
        return f'<figure class="fig missing"><div class="plate">figure pending: {name}</div>' \
               f'<figcaption>{caption}</figcaption></figure>'
    FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, FIG / name)
    cls = 'fig wide' if wide else 'fig'
    return f'<figure class="{cls}"><div class="plate"><img src="fig/{name}" alt="{html.escape(strip_tags(caption))}" ' \
           f'loading="lazy"></div><figcaption>{caption}</figcaption></figure>'


def strip_tags(s):
    return re.sub(r'<[^>]+>', '', s)[:300]


def mjson(path):
    return json.loads(Path(path).read_text())


def minus(text):
    """ASCII minus before a digit (at the start or after a space or bracket) as a typographic minus."""
    return re.sub(r'(^|[\s(\[])-(?=\d)', '\\1\u2212', str(text))


def table(header, rows, cls='data'):
    head = ''.join(f'<th>{h}</th>' for h in header)
    body = ''.join('<tr>' + ''.join(f'<td>{minus(c)}</td>' for c in r) + '</tr>' for r in rows)
    return f'<div class="table-wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def callout(title, text, kind='note'):
    return f'<aside class="callout {kind}"><p class="callout-title">{title}</p>{text}</aside>'


def section(sid, number, title, body, kicker=None):
    k = f'<p class="kicker">{kicker}</p>' if kicker else ''
    num = f'<span class="num">{number}</span>' if number else ''
    return f'<section id="{sid}">{k}<h2>{num}{title}</h2>{body}</section>'


def sub(sid, title, body):
    return f'<section class="sub" id="{sid}"><h3>{title}</h3>{body}</section>'


def P(*paras):
    return ''.join(f'<p>{minus(p)}</p>' for p in paras)


def M(tex):
    """display math (< and > as TeX commands, so the HTML parser does not take them for tags)"""
    tex = tex.replace('<', r'\lt ').replace('>', r'\gt ')
    return f'<div class="math">\\[{tex}\\]</div>'


def dot(m):
    return f'<span class="dot {m}"></span>'


# ----------------------------------------------------------------------------------------------- numbers
def metrics_beam():
    """The headline set: beam-consistent noise, 61,440 training rows, bounded networks."""
    out = {}
    for key, folder, name in (('merlin', 'results_merlin_unbinned_beam_nnwhite_asinh_p30', 'merlin'),
                              ('pca', 'results_merlin_unbinned_beam_nnwhite_asinh_p30', 'pca'),
                              ('merlin_b12', 'results_merlin_unbinned_beam_asinh_p30', 'merlin'),
                              ('pca_b12', 'results_merlin_unbinned_beam_asinh_p30', 'pca')):
        out[key] = mjson(HERE / folder / f'metrics_{name}.json')
    full = HERE / 'doc_metric_data' / 'beam_bins40_N61440_metrics.json'   # the final stage, fetched from idark
    if full.exists():
        out['bins40'], out['bins40_N'] = mjson(full), 61440
        return out
    series = mjson(SERIES)['series_noise']['bins40_beam']
    n40 = max(int(n) for n in series)
    out['bins40'] = series[str(n40)]
    out['bins40_N'] = n40
    return out


def metrics_production():
    return dict(merlin=mjson(HERE / 'results_merlin_unbinned_asinh_p30' / 'metrics_merlin.json'),
                pca=mjson(HERE / 'results_merlin_unbinned_asinh_p30' / 'metrics_pca.json'),
                bins40=mjson(HERE / 'results_final_noise' / 'bins40_noisy' / 'N61440' / 'metrics.json'))


def interval_score_of(label_paths):
    """68 % interval score from per-row quantiles, the first existing file of label_paths (or None)."""
    width = np.array([59, 3.975, 13.2, 2.1, 1.4, 0.6, 6.5, 4.5, 2.5])
    for p in label_paths:
        if p.exists():
            d = np.load(p)
            t, lo, hi = d['truth'].astype(float), d['q16'].astype(float), d['q84'].astype(float)
            s = (hi - lo) + 2 / 0.32 * (lo - t) * (t < lo) + 2 / 0.32 * (t - hi) * (t > hi)
            return float(np.mean(s / width))
    return None


def metric_set():
    """'beam' once the beam-consistent 40-band per-row data exists, else 'production' (all three complete)."""
    return 'beam' if (HERE / 'doc_metric_data' / 'beam_bins40.npz').exists() else 'production'


def strength_of_test_rows():
    split = np.load(HERE / 'results_final' / 'split.npz')
    clean40 = np.load(HERE / 'data' / 'rows_64440' / 'clean_dl.npy')
    obs = HERE.parent / 'tsz_diagnostic_256_accelerated_20260922' / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
    b12 = np.load(obs / 'Battaglia12.npz')['clean_dl']
    return np.log10(np.clip(clean40[split['test'], 9] / b12[9], 1e-12, None))


STRENGTH_BINS = [(-9, -2, 'below 0.01×'), (-2, -1, '0.01–0.1×'), (-1, 0, '0.1–1×'), (0, 1, '1–10×'), (1, 2, '10–100×'),
                 (2, 3, '100–1000×'), (3, 9, 'above 1000×')]


def input_collapse_table():
    """Spread of the network inputs of faint training rows: PCA/Merlin components vs a 40-band feature."""
    d = np.load(HERE / 'doc_figure_data' / 'beam_nn.npz')
    theta = np.load(HERE / 'data' / 'rows_64440' / 'theta.npy')
    clean40 = np.load(HERE / 'data' / 'rows_64440' / 'clean_dl.npy')
    noisy40 = np.load(HERE / 'data' / 'rows_64440' / 'noisy_dl.npy')
    order = np.load(HERE / 'results_final' / 'split.npz')['order'][:len(d['theta'])]
    assert np.allclose(theta[order], d['theta'])
    obs = HERE.parent / 'tsz_diagnostic_256_accelerated_20260922' / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
    b12 = np.load(obs / 'Battaglia12.npz')['clean_dl']
    nm = mjson(HERE / 'results_final_noise' / 'noise_model.json')
    a = np.array(nm['N_band']) / np.sqrt(np.array(nm['nu_band']))
    st = np.log10(np.clip(clean40[order, 9] / b12[9], 1e-12, None))
    y40 = np.arcsinh(noisy40[order] / a)
    z40 = (y40 - y40.mean(0)) / y40.std(0)
    rows = []
    for lo, hi, name in STRENGTH_BINS[:4]:
        sel = (st >= lo) & (st < hi)
        rows.append([name, f'{int(sel.sum()):,}', ' '.join(f'{v:.3f}' for v in d['z_noisy'][sel].std(0)),
                     f'{z40[sel][:, 9].std():.3f}'])
    return table(['signal relative to B12', 'rows (first 4,000 of the pool)', 'Merlin/PCA inputs, components 1–6 (sd)',
                  '40-band input, band 10 (sd)'], rows, cls='plain')



def qa_totals():
    """Totals of the staged beam-fix QA over all stages in series.json."""
    qa = mjson(SERIES)['qa']
    return dict(stages=len(qa), testable=sum(v['noise_slope']['testable_rows'] for v in qa.values()),
                b2_rows=sum(v['b2_approximation_max_err_over_noise_rms']['rows'] for v in qa.values()),
                b2_worst=max(v['b2_approximation_max_err_over_noise_rms']['max'] for v in qa.values()),
                hashed=sum(v['noise_hash_equal_production']['checked'] for v in qa.values()),
                hash_equal=sum(v['noise_hash_equal_production']['equal'] for v in qa.values()))


PRIOR_WIDTH = np.array([59, 3.975, 13.2, 2.1, 1.4, 0.6, 6.5, 4.5, 2.5])


def per_row(mset):
    """{method: per-row dict} of doc_metric_data for a metric set (only the methods present)."""
    D = HERE / 'doc_metric_data'
    labels = dict(bins40='beam_bins40' if mset == 'beam' else 'prod_bins40',
                  pca='beam_nn_pca' if mset == 'beam' else 'prod_pca',
                  merlin='beam_nn_merlin' if mset == 'beam' else 'prod_merlin')
    return {m: {k: v.astype(np.float64) if v.dtype.kind == 'f' else v for k, v in np.load(D / f'{l}.npz').items()}
            for m, l in labels.items() if (D / f'{l}.npz').exists()}


def pearson(t, m):
    t = t - t.mean(-2, keepdims=True)
    m = m - m.mean(-2, keepdims=True)
    return (t * m).sum(-2) / np.sqrt((t * t).sum(-2) * (m * m).sum(-2))


def row_interval_score(d):
    t, lo, hi = d['truth'], d['q16'], d['q84']
    s = (hi - lo) + 2 / 0.32 * (lo - t) * (t < lo) + 2 / 0.32 * (t - hi) * (t > hi)
    return (s / PRIOR_WIDTH).mean(1)


def pairs_table(mset, boot=2000):
    """Paired differences between compressions on the same test rows: mean r, log p, interval score, with errors."""
    rows = per_row(mset)
    rng = np.random.default_rng(7)
    out = []
    for a, b, name in (('bins40', 'merlin', 'Merlin − 40 bins'), ('pca', 'merlin', 'Merlin − PCA'),
                       ('bins40', 'pca', 'PCA − 40 bins')):
        if a not in rows or b not in rows:
            continue
        A, B = rows[a], rows[b]
        assert np.array_equal(A['truth'], B['truth'])
        t = A['truth']
        n = len(t)
        dr = (pearson(t, B['mean']) - pearson(t, A['mean'])).mean()
        bs = []
        for _ in range(boot):
            i = rng.integers(0, n, n)
            bs.append((pearson(t[i], B['mean'][i]) - pearson(t[i], A['mean'][i])).mean())
        sa, sb = pearson(t[None], A['seed_mean']).mean(1), pearson(t[None], B['seed_mean']).mean(1)
        e_r = np.hypot(np.std(bs, ddof=1), np.sqrt(sa.var(ddof=1) / 3 + sb.var(ddof=1) / 3))
        dl = B['lp'] - A['lp']
        la, lb = A['lp_seeds'].mean(1), B['lp_seeds'].mean(1)
        e_l = np.hypot(dl.std(ddof=1) / np.sqrt(n), np.sqrt(la.var(ddof=1) / 3 + lb.var(ddof=1) / 3))
        di = row_interval_score(B) - row_interval_score(A)
        out.append([name, f'{dr:+.3f} ± {e_r:.3f}', f'{dl.mean():+.2f} ± {e_l:.2f}',
                    f'{di.mean():+.3f} ± {di.std(ddof=1) / np.sqrt(n):.3f}'])
    return table(['paired difference', 'mean Pearson r', 'test log p', 'interval score 68 % (lower is better)'], out,
                 cls='data') + P('The interval-score error is the paired test-row error only; per-seed quantiles, '
                                 'needed for a seed term, were not stored.')


def contrib_numbers(mset):
    """Per brightness bin: mean log p per method and the bin's share of the mean log p gap to 40 bins."""
    rows = per_row(mset)
    st = strength_of_test_rows()
    n = len(st)
    out, info = [], {}
    for lo, hi, name in STRENGTH_BINS:
        sel = (st >= lo) & (st < hi)
        means = {m: rows[m]['lp'][sel].mean() for m in rows}
        r = [name, f'{int(sel.sum()):,}'] + [f'{means[m]:.2f}' if m in means else '—' for m in ('bins40', 'merlin', 'pca')]
        for m in ('merlin', 'pca'):
            r.append(f'{sel.sum() / n * (means["bins40"] - means[m]):+.2f}' if 'bins40' in means and m in means else '—')
        out.append(r)
        if 'merlin' in rows and 'bins40' in rows:
            d40, dm = rows['bins40'], rows['merlin']
            rm = lambda d: np.mean(np.sqrt(np.mean(((d['mean'][sel] - d['truth'][sel]) / PRIOR_WIDTH) ** 2, 0)))
            wd = lambda d: np.mean(np.median((d['q84'][sel] - d['q16'][sel]) / (0.6827 * PRIOR_WIDTH), 0))
            cv = lambda d: np.mean((d['truth'][sel] >= d['q16'][sel]) & (d['truth'][sel] <= d['q84'][sel]))
            dlp = dm['lp'][sel] - d40['lp'][sel]
            info[name] = dict(dM=float(dlp.mean()), dMse=float(dlp.std(ddof=1) / np.sqrt(sel.sum())),
                              n=int(sel.sum()), rmse40=rm(d40), rmseM=rm(dm), w40=wd(d40), wM=wd(dm), cov40=cv(d40),
                              covM=cv(dm), lp40=means['bins40'], lpM=means['merlin'], minM=dm['lp'][sel].min(),
                              frac_better40=float(np.mean(d40['lp'][sel] > dm['lp'][sel])))
    tot = ['all test rows', f'{n:,}'] + [f'{rows[m]["lp"].mean():.2f}' if m in rows else '—' for m in ('bins40', 'merlin', 'pca')]
    for m in ('merlin', 'pca'):
        tot.append(f'{rows["bins40"]["lp"].mean() - rows[m]["lp"].mean():+.2f}' if 'bins40' in rows and m in rows else '—')
    out.append(tot)
    head = ['signal at ℓ ≈ 1980 relative to B12', 'test rows', f'{dot("bins40")}40 bins', f'{dot("merlin")}Merlin',
            f'{dot("pca")}PCA', 'share of the gap: 40 bins − Merlin', '40 bins − PCA']
    return table(head, out), info


def unbounded_stats():
    """Bounded vs unbounded per network type, over the beam (B12, noise-only) and production sets."""
    wj = mjson(HERE / 'results_publication' / 'whitening_comparison.json')['sets']
    res = {}
    for v in ('merlin', 'pca'):
        dr, cb, cu, out = [], [], [], []
        for key, folder in (('beam_nn', 'results_merlin_unbinned_beam_nnwhite_asinh_p30'),
                            ('beam_b12', 'results_merlin_unbinned_beam_asinh_p30'), ('prod_b12', 'results_merlin_unbinned_asinh_p30')):
            b, u = mjson(HERE / folder / f'metrics_{v}.json'), mjson(HERE / f'{folder}_lin' / f'metrics_{v}.json')
            dr.append(abs(np.mean(u['pearson_r']) - np.mean(b['pearson_r'])))
            cb.append(np.mean(b['marginal_coverage68'])); cu.append(np.mean(u['marginal_coverage68']))
            out.append(1 - wj[f'{key}_unbounded_{v}']['inside_mean'])
        res[v] = dict(dr=max(dr), cb=(min(cb), max(cb)), cu=(min(cu), max(cu)), out=(min(out), max(out)))
    return res


def fmt(x, nd=3):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def row_of(name, m, isc=None, seed_err=True, note='', err=None):
    lp = m['mean_log_prob_theta']
    err = m.get('test_log_prob_err') if err is None else err
    seed_err = seed_err or err is not None
    lp_s = f'{lp:.2f}' + (f' ± {err:.2f}' if err is not None and seed_err else '')
    cov = m.get('marginal_coverage68')
    return [name + note, lp_s, fmt(np.mean(m['pearson_r'])), fmt(np.mean(m['rmse_over_prior_rmse'])),
            fmt(np.mean(m['median_width68_over_flat'])), fmt(np.mean(cov)) if cov is not None else '—',
            f"{m['tarp68']:.3f} / {m['tarp95']:.3f}", fmt(isc)]


NAMES_DOC = dict(bins40='40 bins', merlin='Merlin', pca='PCA')
HEADER = ['compression · whitening', 'log p', 'r', 'RMSE / prior', 'width / flat', 'cov. 68 %', 'TARP 68 / 95 %',
          'interval score']


# ----------------------------------------------------------------------------------------------- page
def build():
    mb, mp = metrics_beam(), metrics_production()
    D = HERE / 'doc_metric_data'
    PRW = HERE / 'posterior_rows_whitening'
    isc = dict(merlin=interval_score_of([D / 'beam_nn_merlin.npz', PRW / 'beam_nn_bounded_merlin.npz']),
               pca=interval_score_of([D / 'beam_nn_pca.npz', PRW / 'beam_nn_bounded_pca.npz']),
               merlin_b12=interval_score_of([D / 'beam_b12_merlin.npz', PRW / 'beam_b12_bounded_merlin.npz']),
               pca_b12=interval_score_of([D / 'beam_b12_pca.npz', PRW / 'beam_b12_bounded_pca.npz']),
               bins40=interval_score_of([D / 'beam_bins40.npz']))
    isc_p = dict(merlin=interval_score_of([D / 'prod_merlin.npz', PRW / 'prod_b12_bounded_merlin.npz']),
                 pca=interval_score_of([D / 'prod_pca.npz', PRW / 'prod_b12_bounded_pca.npz']),
                 bins40=interval_score_of([D / 'prod_bins40.npz', HERE / 'posterior_rows' / 'noise_bins40_noisy.npz']))
    n40 = mb['bins40_N']
    n40_note = '' if n40 == 61440 else f' <span class="flag">N = {n40:,}</span>'
    qa = qa_totals()
    ub = unbounded_stats()
    lp40 = np.load(HERE / 'doc_metric_data' / 'prod_bins40.npz')['lp_seeds'].mean(1)
    seed_err_40p = float(lp40.std(ddof=1) / np.sqrt(3))
    mset = metric_set()
    contrib, rinfo = contrib_numbers(mset)
    headline = table(HEADER, [
        row_of(f'{dot("merlin")}Merlin · noise-only', mb['merlin'], isc['merlin']),
        row_of(f'{dot("pca")}PCA · noise-only', mb['pca'], isc['pca']),
        row_of(f'{dot("bins40")}40 bins', mb['bins40'], isc['bins40'], seed_err=False, note=n40_note),
        row_of(f'{dot("merlin")}Merlin · B12', mb['merlin_b12'], isc['merlin_b12']),
        row_of(f'{dot("pca")}PCA · B12', mb['pca_b12'], isc['pca_b12'])])
    production = table(HEADER, [
        row_of(f'{dot("merlin")}Merlin · B12', mp['merlin'], isc_p['merlin']),
        row_of(f'{dot("pca")}PCA · B12', mp['pca'], isc_p['pca']),
        row_of(f'{dot("bins40")}40 bins', mp['bins40'], isc_p['bins40'], err=seed_err_40p)])
    wj = mjson(HERE / 'results_publication' / 'whitening_comparison.json')
    dr = wj['r_beam_noise_only_minus_b12']
    pw = mjson(HERE / 'results_merlin_unbinned_beam' / 'inputs' / 'prepare_whitening.json')

    S = []
    # ------------------------------------------------------------------ overview
    S.append(section('overview', '', 'What this document explains', P(
        'We infer nine parameters of the generalised NFW electron-pressure profile from the thermal '
        'Sunyaev–Zel\'dovich (tSZ) power spectrum that the Simons Observatory (SO) would measure. A neural '
        'posterior estimator learns \\(p(\\theta\\mid x)\\) from 64,440 simulated skies painted on the HalfDome '
        'lightcone. This page follows one simulated spectrum through every step: painting, noise, compression, '
        'the network, and the test-set metrics used to compare three ways of compressing the spectrum before it '
        'reaches the network.',
        'The three compressions differ only in what the network is given. <b>40 bins</b> averages the 7,900 '
        'multipoles into 40 band powers. <b>PCA</b> whitens all multipoles with a noise covariance and keeps '
        'the leading principal components of the whitened noise-free spectra (75 or 83). <b>Merlin</b> '
        '(Wernersson, Franco-Abellan &amp; Cañas-Herrera 2026, arXiv:2609.24946) feeds the same components through '
        'a learned compression head that outputs 18 summary numbers; the original trains the head with ratio '
        'estimators, here it is trained jointly with the same flow as the other two, so that only the compression '
        'differs. '
        'All three end in the same type of normalising flow, trained on the same 61,440 simulations and tested on '
        'the same 3,000 held-out simulations.') +
        pipeline_svg() +
        callout('Headline numbers', P(
            'Beam-consistent noise, 61,440 training simulations, three-seed ensembles, posteriors bounded by the '
            'prior box. The ± on log p is the standard error over the three training seeds; the standard error '
            'over the 3,000 test rows is about 0.09 for Merlin and PCA and about 0.065 for 40 bins. Whitening applies '
            'to Merlin and PCA only.') + headline +
            P('Merlin is best on the marginal measures (correlation with the truth, error, width, interval score). '
              'The 40-band network has the highest log p of the full nine-parameter posterior but the weakest '
              'marginal constraints; section 12 shows where the two disagree.'), 'key')))
    # ------------------------------------------------------------------ 1 design
    S.append(section('design', '1', 'Parameters, prior and simulation design', P(
        'Each simulation draws nine parameters from a flat prior box: the pressure amplitude \\(P_0\\), core '
        'scale \\(x_c\\) and outer slope \\(\\beta\\) of the gNFW profile, each with a mass slope '
        '\\(\\alpha_m\\) and a redshift slope \\(\\alpha_z\\). The 65,536 parameter points are a scrambled Sobol '
        'sequence, so any power-of-two prefix is itself a balanced design. Two SO noise seeds per row are '
        'derived from a hash of the row index. 1,096 rows were lost in the production run, leaving 64,440.',
        'A fixed random permutation (seed 0) sets aside 3,000 test rows; a second permutation (seed 1) orders the '
        'remaining 61,440 rows, and every training set of size \\(N\\) is the first \\(N\\) rows of that order. '
        'Smaller training sets are therefore always contained in larger ones, which makes the size curves in '
        'section 12 nested.',
        'The nine parameters set the gas pressure of every halo with \\(M_{200c}\\ge10^{12}M_\\odot\\) in the '
        'HalfDome lightcone (85.2 million halos) through the generalised NFW profile of Battaglia et al. (2012), as '
        'implemented in XGPaint. Each of the three profile parameters scales with halo mass and redshift; the '
        'inner and middle slopes are fixed at \\(\\gamma=-0.3\\) and \\(\\alpha=1\\). The electron pressure is '
        '0.5176 of the thermal pressure, and its line-of-sight integral gives the Compton-\\(y\\) signal. Only gas '
        'inside a sphere of radius \\(4R_{200c}\\) is counted, so the projected profile falls to zero at '
        '\\(4R_{200c}\\).') +
        M(r'\frac{P_{\rm th}(x)}{P_{200c}}=P_0\Big(\frac{x}{x_c}\Big)^{\gamma}\Big[1+\Big(\frac{x}{x_c}\Big)^{\alpha}\Big]^{-\beta},'
          r'\quad x=\frac{r}{R_{200c}},\qquad A=A_0\Big(\frac{M_{200c}}{10^{14}M_\odot}\Big)^{\alpha_m^{A}}(1+z)^{\alpha_z^{A}},'
          r'\ A\in\{P_0,x_c,\beta\}') +
        M(r'P_{200c}=\frac{G M_{200c}\,200\,\rho_{\rm crit}(z)\,f_b}{2R_{200c}},\qquad P_e=0.5176\,P_{\rm th},\qquad '
          r'y=\frac{\sigma_T}{m_e c^2}\int P_e\,dl') +
        table(['', '\\(P_0\\)', '\\(x_c\\)', '\\(\\beta\\)', '\\(\\alpha_m^{P_0}\\)', '\\(\\alpha_m^{x_c}\\)',
               '\\(\\alpha_m^{\\beta}\\)', '\\(\\alpha_z^{P_0}\\)', '\\(\\alpha_z^{x_c}\\)', '\\(\\alpha_z^{\\beta}\\)'],
              [['prior low', '1', '0.025', '2.8', '−0.6', '−1', '−0.2', '−6', '−1.5', '−0.5'],
               ['prior high', '60', '4', '16', '1.5', '0.4', '0.4', '0.5', '3', '2'],
               ['B12', '18.1', '0.497', '4.35', '0.154', '−0.00865', '0.0393', '−0.758', '0.731', '0.415']]) +
        fig('step_profile.png', '(a) The pressure profile at the B12 parameters for three halo masses at z = 0 '
            '(solid) and z = 1 (dashed). (b) The effect of moving one amplitude to 10 % (dashed) or 90 % (dotted) of '
            'its prior range, at \\(M_{200c}=10^{14}M_\\odot\\), z = 0.5.') +
        code(BUNDLE / 'design' / 'design_tests.py', [(9, 21)]) +
        code('sbi_final.py', [(204, 211)]) +
        M(r'p(\theta)=\prod_{i=1}^{9}\frac{\mathbb 1[\theta_i^{\min}\le\theta_i\le\theta_i^{\max}]}{\theta_i^{\max}-\theta_i^{\min}},'
          r'\qquad \log p(\theta) = -\sum_i \log w_i = -12.90 \text{ inside the box}') +
        fig('step_design.png', '(a) Two of the nine parameters in prior units: training rows (grey, a subset) and '
            'the 3,000 test rows (red) fill the box evenly; the star is the Battaglia et al. (2012) point used as '
            'the reference mock (B12). (b) The flat prior spans more than ten decades of tSZ power at ℓ ≈ 1980; B12 sits at '
            'the 29th percentile. (c) Noise-free 40-band spectra of 40 random training rows and of B12.')))
    # ------------------------------------------------------------------ 2 spectra
    S.append(section('spectra', '2', 'From a painted map to a measured spectrum', P(
        'For every parameter point the engine paints a Compton-\\(y\\) map from the HalfDome halo catalogue at '
        'NSIDE 8192, smooths it with a 2′ Gaussian beam in harmonic space, and evaluates it at NSIDE 4096. The '
        'map is multiplied by a fixed apodised cap mask with \\(f_{\\rm sky}=0.4\\). Two independent SO noise '
        'maps (baseline, deprojection 0) are added to two copies of the masked signal, and the stored spectrum is '
        'the cross spectrum of the two noisy copies. Crossing two independent noise realisations removes the noise '
        'bias, but the result can be negative where noise dominates.',
        'All spectra are kept as \\(D_\\ell=\\ell(\\ell+1)C_\\ell/2\\pi\\) for \\(\\ell=80\\dots7979\\): 7,900 numbers '
        'per simulation. No mode-coupling correction is applied; training spectra and observations go through '
        'the same operator, so the inference does not need one.') +
        code(BUNDLE / 'engine' / 'engine.jl', [(10, 18), (20, 30), (34, 45)], lang='julia') +
        M(r'\hat C_\ell=\frac{1}{2\ell+1}\sum_m a^{(1)}_{\ell m}a^{(2)*}_{\ell m}'
          r'= C^{ss}_\ell + C^{sn}_\ell + C^{nn}_\ell,\qquad m_i = M\,(s_B + n_i)') +
        P('Here \\(C^{ss}\\) is the masked signal spectrum, \\(C^{sn}\\) the signal–noise cross terms and '
          '\\(C^{nn}\\) the cross of the two noise maps. The signal term \\(C^{ss}\\) is stored separately for '
          'every simulation, which the next step relies on.') +
        fig('step_spectra.png', '(a) The B12 mock: noise-free masked spectrum (black), the stored spectrum with '
            'the production noise (red) and with beam-consistent noise (blue). (b) The noise terms as an rms over '
            '100 multipoles: production noise grows above ℓ ≈ 5000 because the noise map was added without the '
            'beam; beam-consistent noise falls with the beam. (c) The same spectrum averaged into the 40 bands used by the '
            '40-bin compression.', wide=True)))
    # ------------------------------------------------------------------ 3 beam
    S.append(section('beam', '3', 'Beam-consistent noise', P(
        'The SO noise curves are beam-deconvolved, so on a map they belong under the same beam as the signal. '
        'The production engine beamed the signal but added the noise without the beam, which overstates the noise '
        'power by \\(1/B_\\ell^2\\) in the noise × noise term and by \\(1/B_\\ell\\) in the signal × noise term. '
        'Instead of repainting 64,440 maps, the noise maps were regenerated from their stored seeds (the same '
        'Gaussian realisations, up to floating-point round-off that depends on the CPU), masked, and crossed to give '
        '\\(C^{nn}\\) for every row. With \\(C^{ss}\\) stored, the signal–noise term follows, and the beam is applied to each noise term '
        'separately:') +
        M(r'C^{\rm beam}_\ell = C^{ss}_\ell + B_\ell\,\big(C^{\rm noisy}_\ell - C^{ss}_\ell - C^{nn}_\ell\big) + B_\ell^2\,C^{nn}_\ell') +
        code('beamfix/regen_noise.jl', [(168, 172), (192, 197)], lang='julia') +
        code('beamfix/beamfix.py', [(305, 311), (314, 327)]) +
        P('Because the pixel hashes differ by round-off (none of the {h:,} hashed rows matched bitwise), each pair '
          'is checked statistically: the regression slope of \\(C^{{\\rm noisy}}-C^{{ss}}\\) on the regenerated '
          '\\(C^{{nn}}\\) at \\(\\ell\\ge6000\\) is 1 for the right pair and 0 for an unrelated one. All {t:,} testable '
          'rows of the {st} stages passed. Using \\(B^2C^{{nn}}\\) instead of crossing beamed noise maps is accurate '
          'to {b2:.2f} % of the band noise rms at worst ({b2r} rows checked against the exact beamed cross). '
          'The excerpt also returns a half-depth variant used only in the staged 40-band study.'.format(
              h=qa['hashed'], t=qa['testable'], st=qa['stages'], b2=100 * qa['b2_worst'], b2r=qa['b2_rows'])) +
        fig('step_beam.png', '(a) The beam transfer function and its square, which multiply the signal–noise and '
            'noise–noise terms, and the NSIDE 4096 pixel window (relevant only for observed maps, section 13). '
            '(b) The band noise rms after the correction relative to the production noise: unchanged at low ℓ, '
            '13 % of it in the last band.')))
    # ------------------------------------------------------------------ 4 bins40
    S.append(section('bins40', '4', 'Compression A: 40 band powers', P(
        'The simplest compression averages the 7,900 multipoles into 40 bands of 200 multipoles (the last band '
        'is 100 wide), weighting each multipole by its number of modes \\(2\\ell+1\\).') +
        code('train_ensemble.py', [(715, 729)]) +
        M(r'D_b=\frac{\sum_{\ell\in b}(2\ell+1)\,D_\ell}{\sum_{\ell\in b}(2\ell+1)}') +
        P('Band powers range over many decades across the prior, so a linear z-score would squeeze most '
          'simulations, B12 included, into a sliver near zero. Each band therefore goes through a signed '
          '\\(\\operatorname{asinh}\\) at the noise level before the z-score: linear within one noise unit, '
          'logarithmic beyond. The noise scale comes from a per-band noise model fitted to the training rows '
          'themselves: around a noise-free power \\(S_b\\) the cross spectrum scatters with variance '
          '\\((2S_bN_b+N_b^2)/\\nu_b\\), so the mean squared residual is a straight line in \\(S_b\\).') +
        code('sbi_final.py', [(112, 126), (137, 147)]) +
        code('train_ensemble.py', [(125, 129), (148, 154)]) +
        M(r'x_b=\frac{\operatorname{asinh}(D_b/a_b)-\overline{y}_b}{\sigma_{y_b}},\qquad a_b=\frac{N_b}{\sqrt{\nu_b}}') +
        fig('step_bins40.png', 'Production-noise band powers. (a) Their spread over the training pool and the noise scale '
            '\\(a_b\\). (b) The asinh is linear below one noise unit and logarithmic above. (c) Network input for '
            'band 10: a linear z-score piles almost every row onto one value; the asinh spreads them over O(1).') +
        callout('Different input conventions', P(
            'The 40-band asinh acts on the band power directly, with an analytic noise scale. The PCA and Merlin '
            'inputs (sections 5–7) use a centred asinh with a robust noise scale measured on the PCA components. '
            'Choosing the noise-level scale moved the 40-band test log p from −4.61 to −2.39 nats.'))))
    # ------------------------------------------------------------------ 5 whitening
    S.append(section('whitening', '5', 'Compressions B and C, step 1: whitening', P(
        'PCA and Merlin start from all 7,900 multipoles. Neighbouring multipoles are correlated by the mask and '
        'their noise differs by orders of magnitude across ℓ, so the spectra are first whitened with a noise '
        'covariance \\(C=LL^\\top\\): \\(\\tilde x=L^{-1}x\\) has approximately unit, uncorrelated noise. Two '
        'covariances were compared.',
        '<b>B12 whitening</b> uses the 20 SO noise draws of the B12 mock (the observation\'s own draw left out). '
        'It contains the signal × noise term at the B12 sky, so it depends on a fiducial model. <b>Noise-only '
        'whitening</b> uses the regenerated noise × noise spectra of all 61,440 training rows. It needs no '
        'fiducial, and its 61,440 samples resolve the correlation between neighbouring multipoles that 20 draws '
        'cannot.',
        'Both use the same estimator: the correlation matrix of the standardised draws is shrunk towards the '
        'identity with the Oracle Approximating Shrinkage (OAS) rule, renormalised to a unit diagonal, and '
        'multiplied by the measured variances.') +
        code('merlin_beam_prepare.py', [(76, 98)]) +
        code('merlin_beam_prepare.py', [(217, 226)]) +
        M(r'\hat R=(1-\rho)\,S_z+\rho\,\tfrac{\operatorname{tr}S_z}{p}\,I,\qquad C=D\,R\,D,\quad D=\operatorname{diag}(\sigma_\ell),'
          r'\qquad C^{\rm beam}_{\ell\ell^\prime}=C^{nn}_{\ell\ell^\prime}B_\ell^2B_{\ell^\prime}^2') +
        table(['covariance', 'samples', 'OAS shrinkage ρ', 'mean lag-1 correlation'], [
            ['B12 noise draws', '20', f"{pw['sets']['beam_b12']['covariance']['oas_correlation_shrinkage']:.3f}", '≈ 0 (shrunk away)'],
            ['noise-only, first 1,000 rows', '1,000', f"{pw['noise_only']['estimate_from_1000_rows']['shrinkage_1000']:.3f}",
             f"{pw['noise_only']['estimate_from_1000_rows']['lag1_correlation_1000']:.3f}"],
            ['noise-only, all training rows', '61,440', f"{pw['noise_only']['oas_correlation_shrinkage']:.3f}",
             f"{pw['noise_only']['estimate_from_1000_rows']['lag1_correlation_61440']:.3f}"]]) +
        fig('step_whitening.png', '(a) Noise standard deviation per multipole of the two covariances (beam-consistent '
            'noise). (b) Mean correlation between multipoles ℓ and ℓ+k: 20 draws cannot resolve it, 61,440 can. '
            '(c) The noise of the B12 mock after whitening, against a unit Gaussian. (d, e) A 40 × 40 block of each '
            'correlation matrix (ℓ = 2000–2039).', wide=True) +
        fig('fig_pub_merlin_beam_whitening.png', 'Effect of noise model and whitening on Merlin and PCA. (a, b) '
            'Test log p for the four input sets, bounded and unbounded posteriors; the light dashes are the '
            'unbounded posteriors truncated to the prior box, whose higher log p comes with worse coverage (68 % '
            'coverage 0.56–0.58 after truncation; section 9 gives the untruncated values). (c) Test log p against the '
            'number of training simulations for production and beam-consistent noise, B12 whitening. (d) What B12 '
            'whitening adds, for production noise: the scatter of the 20 B12 noise draws over the noise-only '
            'scatter, in bands. The excess, about \\(\\sqrt{1+2S/N}\\), is the signal × noise term; it is visible '
            'wherever the B12 signal is a noticeable fraction of the noise power (ℓ ≲ 5000 here, further for '
            'beam-consistent noise).', source='pub', wide=True)))
    # ------------------------------------------------------------------ 6 pca
    S.append(section('pca', '6', 'Compressions B and C, step 2: principal components', P(
        'The whitened noise-free spectra of the training pool are decomposed into principal components. Their '
        '7,900 × 7,900 covariance is accumulated in chunks, its eigenvectors give the principal axes, and the '
        'singular values of the centred data matrix are \\(S_k=\\sqrt{n\\lambda_k}\\). Following Merlin\'s '
        'original code, of the top 500 components those whose share of the summed singular values exceeds '
        '0.001 % are kept: 75 with B12 whitening, 83 with noise-only whitening. Every spectrum, noisy or not, is '
        'then projected as \\(s=(L^{-1}x)\\,V\\).') +
        code('merlin_beam_prepare.py', [(101, 114)]) +
        M(r'\Sigma_w=\tfrac1n\textstyle\sum_j w_jw_j^\top-\bar w\bar w^\top=U\Lambda U^\top,\quad w_j=L^{-1}x^{\rm clean}_j;'
          r'\qquad \text{keep } k \text{ if } 100\,S_k\big/\textstyle\sum_{k^\prime\le500}S_{k^\prime}>10^{-3}') +
        fig('step_pca.png', '(a) Share of the summed singular values per component, with the 0.001 % cut. '
            '(b) The first six principal components as functions of ℓ (noise-only whitening).', wide=True)))
    # ------------------------------------------------------------------ 7 asinh
    S.append(section('asinh', '7', 'Compressions B and C, step 3: asinh standardisation', P(
        'The projections still span many decades over the prior, and a plain z-score is dominated by the '
        'brightest rows. Each component is centred on its pool median, divided by a robust noise scale (1.4826 × '
        'the median absolute deviation of the noise part \\(s^{\\rm noisy}-s^{\\rm clean}\\)), passed through '
        '\\(\\operatorname{asinh}\\) and z-scored. The same parameters are applied to test rows and observations.') +
        code('merlin_compression.py', [(152, 178)]) +
        M(r'x_j=\frac{\operatorname{asinh}\!\big((s_j-m_j)/c_j\big)-\bar z_j}{\sigma_{z_j}},\qquad '
          r'c_j=1.4826\,\operatorname{median}\big|r_j-\operatorname{median}(r_j)\big|,\quad r=s^{\rm noisy}-s^{\rm clean}') +
        fig('step_asinh.png', 'Network inputs for three PCA components over 4,000 training rows (beam-consistent '
            'noise, noise-only whitening): a linear z-score (grey) leaves most rows in a narrow spike, the asinh '
            'standardisation (colour) spreads the brighter rows over O(1). In component 1 the faintest 46 % of rows '
            'still pile up near −1 (see the box below). (d) The transform of component 1: linear within one noise '
            'unit of the pool median (grey band), logarithmic beyond; the faint skies sit near −1,300.', wide=True) +
        callout('A weakness for faint skies', P(
            'The asinh is centred on the pool median, a sky about 14 times brighter than B12 at ℓ ≈ 1980. A sky much '
            'fainter than B12 has projections close to zero, which lie about 1,300 and 2,100 noise units from the '
            'centre in the first two components (24 to 270 in components 3–6), in the logarithmic part of the asinh. '
            'There a tenfold change in a faint sky moves the input by only a few thousandths of its unit scale at '
            '0.01–0.1× B12, and by about \\(10^{-5}\\) below 0.01×. The asinh is monotonic, so the ordering by '
            'brightness survives (rank correlation with brightness up to 0.8 even for the faintest skies), but the '
            'differences become tiny. The 40-band asinh is centred on zero, so a band power well above the noise maps '
            'to about \\(\\ln(2D_b/a_b)\\): skies a factor of ten apart stay \\(\\ln 10\\) apart down to a few hundredths '
            'of B12 (B12 is about 90 noise units in band 10); below that the band is noise-dominated. The spread of '
            'the inputs over the faint rows among the first 4,000 training rows:') +
            input_collapse_table() +
            P('The Merlin/PCA columns use beam-consistent noise with noise-only whitening, the 40-band column the '
              'production band powers; for the faintest rows the 40-band spread is mostly noise. The compression is '
              'clear below 0.1× B12. Whether it explains the faint-sky part of the log p gap (section 12) is a '
              'hypothesis; centring the PCA asinh at zero, like the 40-band transform, would test it.'), 'warn')))
    # ------------------------------------------------------------------ 8 networks
    params = {k: mjson(HERE / 'results_merlin_unbinned_beam_nnwhite_asinh_p30' / f'{k}_seed1.json')['parameters']
              for k in ('merlin', 'pca')}
    S.append(section('networks', '8', 'The networks: one flow, three ways in', P(
        'How a network learns a posterior: every training pair \\((\\theta, x)\\) is a draw from the prior and the '
        'simulator. Minimising the average of \\(-\\log q_\\phi(\\theta\\mid x)\\) over such pairs minimises the expected '
        'Kullback–Leibler divergence between the true posterior and \\(q_\\phi\\), so a flexible enough \\(q_\\phi\\) '
        'approaches \\(p(\\theta\\mid x)\\) for every \\(x\\) at once. This is amortised neural posterior estimation (NPE): '
        'after training, a posterior for a new spectrum costs one network evaluation.') +
        M(r'\mathbb E_{p(\theta)\,p(x\mid\theta)}\big[-\log q_\phi(\theta\mid x)\big]'
          r'=\mathbb E_{p(x)}\,\mathrm{KL}\big(p(\theta\mid x)\,\big\|\,q_\phi(\theta\mid x)\big)+\text{const}') +
        P('The density estimator is an sbi masked autoregressive flow (MAF) with 5 transforms of 128 hidden units. '
          'It turns a standard Gaussian into \\(q_\\phi\\) through a chain of invertible maps; each map shifts and '
          'scales one parameter at a time, with coefficients predicted from the parameters before it and from the '
          'conditioning input, so the density stays exact and cheap to evaluate. The three compressions differ only '
          'in that conditioning input. 40 bins and PCA condition the flow directly on their 40 or 75–83 standardised '
          'inputs. Merlin first passes the PCA components through a learned head (two hidden layers of 2048 and 512 '
          'units, an output of 18 features, twice the number of parameters, and a batch normalisation) that is '
          'trained jointly with the flow as its embedding network.') +
        network_svg(params) +
        code('merlin_compression.py', [(343, 354)]) +
        code('merlin_compression.py', [(371, 379)]) +
        fig('merlin_features.png', 'What Merlin\'s head learns (beam-consistent noise, noise-only whitening, seed 1): '
            'correlation of each of the 18 learned features (rows, grouped by the parameter they follow most) with '
            'each parameter (columns) over the 3,000 test rows. The features are not tied to single parameters: most '
            'mix several, and no correlation exceeds about 0.6. The head passes the flow a compact, nonlinear '
            'summary, and the flow separates the parameters.')))
    # ------------------------------------------------------------------ 9 logit
    S.append(section('logit', '9', 'Where the flow lives: bounded and unbounded parameter spaces', P(
        'The flow models transformed parameters \\(u\\), not \\(\\theta\\) itself. By default each parameter is '
        'mapped to its position \\(t\\) in the prior range and then through the logit. The flat prior becomes a '
        'logistic distribution on the real line and every sample maps back inside the box ("bounded"). Densities '
        'in \\(\\theta\\) need the Jacobian of the map. Setting <code>MERLIN_PARAM_SPACE=linear</code> instead uses '
        '\\(u=t\\), whose Jacobian is constant, so posterior mass can extend past the prior bounds ("unbounded").') +
        code('train_ensemble.py', [(60, 83)]) +
        M(r'u=\log\frac{t}{1-t},\ t=\frac{\theta-\theta^{\min}}{w};\qquad '
          r'\log p_\theta(\theta\mid x)=\log q_u(u\mid x)-\sum_i\log\Big|\frac{d\theta_i}{du_i}\Big|,\quad '
          r'\frac{d\theta}{du}=w\,\sigma(u)\,\sigma(-u)') +
        fig('step_logit.png', '(a) The logit map and its log-Jacobian, which diverges at the box edges. (b) The '
            'bounded Merlin posterior of \\(\\alpha_m^{P_0}\\) for the B12 mock in \\(u\\), against the logistic '
            'prior. (c) The same marginal in \\(\\theta\\) for the bounded and the unbounded network; dotted lines '
            'are the prior bounds, the dashed line the input value.') +
        callout('Bounded or unbounded', P(
            f"Unbounded networks keep the mean Pearson r almost unchanged (Merlin within {ub['merlin']['dr']:.3f}, "
            f"PCA within {ub['pca']['dr']:.3f}) and give narrower intervals, but their mean 68 % coverage drops to "
            f"{ub['merlin']['cu'][0]:.3f}–{ub['merlin']['cu'][1]:.3f} for Merlin and {ub['pca']['cu'][0]:.3f}–"
            f"{ub['pca']['cu'][1]:.3f} for PCA, and {100 * ub['merlin']['out'][0]:.0f}–{100 * ub['merlin']['out'][1]:.0f} % "
            f"(Merlin) and {100 * ub['pca']['out'][0]:.0f}–{100 * ub['pca']['out'][1]:.0f} % (PCA) of their probability "
            f"lies outside the prior box. The bounded networks are close to calibrated: Merlin's mean 68 % coverage is "
            f"{ub['merlin']['cb'][0]:.3f}–{ub['merlin']['cb'][1]:.3f} (slightly low for \\(P_0\\) and "
            f"\\(\\alpha_m^{{P_0}}\\)), while bounded PCA ({ub['pca']['cb'][0]:.3f}–{ub['pca']['cb'][1]:.3f}) and 40 bins "
            f"are slightly conservative. The bounded networks are used for all comparisons."))))
    # ------------------------------------------------------------------ 10 training
    S.append(section('training', '10', 'Training and ensembles', P(
        'Merlin and PCA are trained with the loop below: a seed-dependent 20 % of the pool is held for '
        'validation and AdamW minimises the mean \\(-\\log q(u\\mid x)\\). PyTorch\'s ReduceLROnPlateau (patience 10) '
        'cuts the learning rate tenfold on the eleventh epoch without a relative improvement of \\(10^{-4}\\), and '
        'training stops after 30 epochs without an improvement larger than \\(10^{-6}\\), keeping the best '
        'validation state. These patiences were set with <code>MERLIN_PATIENCE=30 MERLIN_LR_PATIENCE=10</code> '
        '(the script defaults are 5 and 3). The 40-band networks were trained with sbi\'s built-in trainer, whose '
        'settings differ (table). Each configuration is trained with three seeds, and the three posteriors are '
        'combined as an equal-weight mixture.') +
        code('merlin_compression.py', [(398, 403), (409, 435)]) +
        code('train_ensemble.py', [(264, 266)]) +
        table(['', '40 bins (sbi NPE.train)', 'Merlin and PCA (merlin_compression loop)'], [
            ['optimiser', 'Adam, learning rate 7×10⁻⁴, no schedule',
             'AdamW, 5×10⁻⁴, ×0.1 after 11 epochs without a relative gain of 10⁻⁴'],
            ['batch / validation share', '512 / 10 %', '256 / 20 %'],
            ['gradient clipping', 'norm 5 (sbi default)', 'none'],
            ['early stopping', '30 epochs without gain', '30 epochs without gain'],
            ['epochs at the full training set', '≈ 340–440 (beam-consistent), 380–500 (production)', '≈ 75–110']]) +
        fig('step_training.png', 'Validation loss per epoch for the three seeds (beam-consistent noise, noise-only '
            'whitening). Dots mark the kept best epoch; faint vertical lines the learning-rate drops. Both networks '
            'overfit after their best epoch (the validation loss rises by 0.3–0.8 nats); early stopping keeps the '
            'best state.') +
        code('train_ensemble.py', [(343, 357), (375, 376)]) +
        M(r'q_{\rm ens}(u\mid x)=\frac13\sum_{k=1}^{3}q_k(u\mid x),\qquad \log q_{\rm ens}=\operatorname{logsumexp}_k\log q_k-\log 3') +
        fig('step_ensemble.png', '(a) Why the mixture helps: three seed posteriors and their equal-weight mixture; '
            'a truth in the tail of one seed still has density under the mixture. (b) Test log p of each single '
            'seed (open) and of the ensemble (filled): the ensemble is 0.7–1.7 nats higher (0.7–1.1 for Merlin, '
            '1.1–1.7 for PCA). (c) Pearson r per parameter for single seeds and the ensemble (Merlin).')))
    # ------------------------------------------------------------------ 11 metrics
    set_note = ('beam-consistent noise, with noise-only whitening for Merlin and PCA' if mset == 'beam'
                else 'production noise, with B12 whitening for Merlin and PCA (all three compressions have per-row '
                     'data for this set; the beam-consistent 40-band per-row data is pending)')
    pr = per_row(mset)
    bulk = ', '.join(f'{100 * np.mean(np.abs(pr[m]["lp"]) < 3):.0f} % for {NAMES_DOC[m]}' for m in ('bins40', 'merlin', 'pca') if m in pr)
    m11 = []
    m11.append(P(
        'Every metric is computed on the same 3,000 test simulations. For each test row the ensemble draws 999 '
        'posterior samples (333 per seed; torch and numpy seeds fixed), which give the posterior mean and '
        'quantiles; the log density of the true parameters is evaluated directly. The function below produces '
        'all stored metrics. It also records the edge-mass ratio, the share of samples in the outer 1 % of each '
        f'prior range divided by its flat-prior value of 2 %, a check for pile-ups at the box edges. The figures '
        f'in this section use the {set_note}.') +
        code('sbi_final.py', [(327, 353)]))
    m11.append(sub('m-logp', '11.1 Test log p', P(
        'The mean log posterior density of the true parameters, in \\(\\theta\\) units. It is a proper scoring '
        'rule for the full nine-dimensional posterior: it rewards concentrating probability where the truth is, '
        'jointly in all parameters. A flat posterior scores −12.90. Two uncertainties are worth separating. The '
        'standard error over the test rows (about 0.09 for Merlin and PCA, 0.07 for 40 bins) is the uncertainty '
        'of the value for this particular '
        'test set. The seed standard error, from the three seeds\' own log p, measures the scatter between '
        'trainings and is the one quoted in the tables. For differences between methods on the same test rows the '
        'paired errors of section 11.8 apply.') +
        M(r'\langle\log p\rangle=\frac1N\sum_{j=1}^{N}\Big[\log q_{\rm ens}(u_j\mid x_j)-\log\Big|\frac{d\theta}{du}\Big|_{u_j}\Big],'
          r'\qquad \mathrm{SEM}_{\rm seeds}=\frac{\operatorname{sd}_k(\langle\log p\rangle_k)}{\sqrt3}') +
        code('merlin_compression.py', [(490, 495)]) +
        fig(f'metric_logp_{mset}.png', '(a) Log p of the truth per test row: the bulk of the rows lies between '
            f'about −3 and +3 ({bulk}), and a second group sits around −11 (mainly Merlin and PCA), 1–2 nats above '
            'the flat-prior value −12.9. (b) Mean log p in bins of the signal strength of the test sky (209–570 rows '
            'per bin): the second group are faint skies.', wide=True)))
    m11.append(sub('m-r', '11.2 Pearson r and RMSE', P(
        'Pearson r measures how well the posterior mean tracks the truth across the test set, parameter by '
        'parameter. The RMSE of the posterior mean is quoted relative to the RMSE of always guessing the prior '
        'centre, so 1 means nothing was learned. Both reward point accuracy and ignore the posterior width.') +
        M(r'r_i=\operatorname{corr}_j(\theta_{ji},\mu_{ji}),\qquad '
          r'\mathrm{RMSE}_i=\frac{\sqrt{\langle(\mu_{ji}-\theta_{ji})^2\rangle_j}}{\sqrt{\langle(c_i-\theta_{ji})^2\rangle_j}},\ c_i=\tfrac12(\theta_i^{\min}+\theta_i^{\max})') +
        fig(f'metric_r_{mset}.png', 'Truth against posterior mean over the 3,000 test rows for the best-constrained '
            '(\\(\\alpha_z^{P_0}\\), top) and worst-constrained (\\(\\alpha_m^{\\beta}\\), bottom) parameter. Where '
            'the data say little, the posterior mean is pulled towards the prior centre (dotted), which flattens the '
            'cloud and lowers r.', wide=True)))
    m11.append(sub('m-width', '11.3 Posterior width', P(
        'The median over test rows of the central 68 % interval (16th to 84th percentile), divided by 0.6827 of '
        'the prior range. A value near 1 means the data did not narrow that parameter; a flat posterior scores '
        '0.68/0.6827 = 0.996.') +
        M(r'W_i=\operatorname{median}_j\frac{q_{84,ji}-q_{16,ji}}{0.6827\,w_i}') +
        fig(f'metric_width_{mset}.png', '(a) Distribution over test rows of the 68 % width relative to flat, averaged '
            'over the nine parameters. (b) Median width per parameter.')))
    m11.append(sub('m-cov', '11.4 Marginal coverage', P(
        'A posterior can be narrow because it is good or because it is overconfident. The share of test rows '
        'whose true value lies inside the central 68 % (or 95 %) interval checks this: a calibrated posterior gives '
        '0.68 (0.95), within a 95 % binomial range of ±0.017 around 0.68 and ±0.008 around 0.95 for 3,000 rows.') +
        fig(f'metric_coverage_{mset}.png', '(a) 68 % intervals of 60 test rows with the truth (tick); red intervals '
            'miss it. (b) 68 % (filled) and 95 % (open) coverage per parameter; the grey band is the 95 % binomial '
            'range around 0.68.')))
    m11.append(sub('m-tarp', '11.5 TARP: coverage of the joint posterior', P(
        'Marginal coverage checks one parameter at a time. TARP (Tests of Accuracy with Random Points) checks the '
        'joint posterior: for each test row a reference point is drawn uniformly within the range of the test '
        'truths, and the fraction of posterior samples closer to it than the truth is recorded, with Euclidean '
        'distances after scaling each parameter to that range. For a calibrated posterior these fractions are '
        'uniform, and the expected coverage curve lies on the diagonal. The stored numbers are the curve at '
        'credibility 0.68 and 0.95.') +
        M(r'f_j=\frac1n\sum_{s}\mathbb 1\big[d(\theta^{(s)}_j,\theta^{\rm ref}_j)<d(\theta_j,\theta^{\rm ref}_j)\big],\qquad '
          r'\mathrm{ECP}(\alpha)=\frac1N\sum_j\mathbb 1[f_j<\alpha]') +
        fig(f'metric_tarp_{mset}.png', '(a) TARP expected coverage against credibility level for the three '
            'compressions; the diagonal is perfect calibration, curves above it are conservative. (b) The difference '
            'from the diagonal; the grey band is the 95 % Kolmogorov–Smirnov band ±0.025, and the dotted lines mark '
            'the stored levels 0.68 and 0.95.')))
    m11.append(sub('m-sbc', '11.6 Simulation-based calibration', P(
        'Simulation-based calibration (SBC) ranks the true value of each parameter among its 999 posterior '
        'samples. If the posterior is calibrated, the true value is just another draw and the ranks are uniform. '
        'The empirical CDF of the ranks minus the uniform CDF makes departures visible: an S-shape means the '
        'posterior is too wide or too narrow, a one-sided bulge means it is biased. The grey band is the 95 % '
        'Kolmogorov–Smirnov band \\(\\pm1.358/\\sqrt{3000}=\\pm0.025\\).') +
        code('posterior_rows.py', [(76, 82)]) +
        fig(f'metric_sbc_{mset}.png', 'Left: rank histograms of the true \\(\\beta\\) (flat = calibrated). Right: '
            'ECDF minus uniform for all nine parameters per compression. A hump below then above zero means ranks '
            'cluster in the middle: the posterior is too wide (conservative).', wide=True)))
    m11.append(sub('m-is', '11.7 Interval score', P(
        'Width and coverage pull in opposite directions, so neither alone ranks two methods. The interval score '
        'combines them into one proper score for the central 68 % interval: the width, plus a penalty '
        'proportional to how far the truth falls outside. Lower is better; a flat posterior scores about 0.84 '
        'in prior units. It is computed from the per-row quantiles, so it is available for every network whose '
        'per-row posteriors were stored.') +
        M(r'\mathrm{IS}_\alpha=(u-l)+\frac{2}{\alpha}(l-\theta)\,\mathbb 1[\theta<l]+\frac{2}{\alpha}(\theta-u)\,\mathbb 1[\theta>u],\qquad \alpha=0.32') +
        code('doc_figures.py', interval_score_lines())))
    m11.append(sub('m-paired', '11.8 Comparing two models on the same test rows', P(
        'Differences between two models are smaller than the test-set scatter of either, but much of that scatter '
        'is shared because both see the same 3,000 test rows. The difference in r is therefore bootstrapped with '
        'the same resampled rows for both models, and the scatter between training seeds is added in quadrature '
        'from each seed\'s own posterior means. Log p and interval-score differences use the per-row differences '
        'in the same way.') +
        code('whitening_comparison.py', [(115, 139)]) +
        M(r'\Delta r_i=r_i^{(b)}-r_i^{(a)},\qquad \sigma_i^2=\operatorname{var}_{\rm boot}(\Delta r_i^\ast)+'
          r'\frac{\operatorname{var}_k r^{(a)}_{ik}}{3}+\frac{\operatorname{var}_k r^{(b)}_{ik}}{3}') +
        P(f"Applied to noise-only against B12 whitening (beam-consistent noise), Merlin gains "
          f"\\(\\Delta r={dr['bounded_merlin']['mean_diff']:+.4f}\\pm{dr['bounded_merlin']['mean_err']:.4f}\\) "
          f"averaged over the nine parameters (bounded), positive for all nine; PCA shows no consistent change "
          f"(\\({dr['bounded_pca']['mean_diff']:+.4f}\\pm{dr['bounded_pca']['mean_err']:.4f}\\)).") +
        fig('fig_pub_r_whitening.png', 'Per-parameter r for B12 and noise-only whitening (beam-consistent noise) '
            'and their paired difference with 1σ errors, bounded and unbounded posteriors.', source='pub', wide=True)))
    S.append(section('metrics', '11', 'How the test metrics are computed', ''.join(m11)))
    # ------------------------------------------------------------------ 12 comparison
    rk = {k: rinfo.get(k, {}) for k in ('0.01–0.1×', '0.1–1×', 'above 1000×')}
    set_label = ('beam-consistent noise, noise-only whitening' if mset == 'beam'
                 else 'production noise, B12 whitening for Merlin and PCA; the beam-consistent 40-band per-row data is pending')
    regimes = ''
    lead_bins = [k for k, v in rinfo.items() if v['dM'] > 2 * v['dMse']]
    level_bins = [k for k, v in rinfo.items() if abs(v['dM']) <= 2 * v['dMse']]
    leads = ', '.join(lead_bins) + ' B12' if lead_bins else 'no brightness'
    level = (f"; the two are level at {', '.join(level_bins)}" if level_bins else '')
    if all(rk.values()):
        a, b, c = rk['0.01–0.1×'], rk['0.1–1×'], rk['above 1000×']
        regimes = P(
            f"Three regimes stand out. <b>Faint skies</b>, below 0.1× B12: Merlin and PCA stay close to the prior "
            f"value of −12.9, while 40 bins reaches {a['lp40']:.1f} at 0.01–0.1×; section 7 shows how the PCA "
            f"inputs of these skies are compressed into tiny differences. <b>Skies of 0.1–1× B12</b> carry the "
            f"largest share of the gap. Here Merlin's marginals are better than those of 40 bins (RMSE "
            f"{b['rmseM']:.3f} against {b['rmse40']:.3f} of the prior width, 68 % width {b['wM']:.2f} against "
            f"{b['w40']:.2f}), yet 40 bins gives the higher log p for {100 * b['frac_better40']:.0f} % of these "
            f"rows: its posterior follows the joint structure of the nine parameters, including their narrow "
            f"degeneracies, more closely, which log p rewards and the one-parameter metrics do not see. <b>The "
            f"brightest skies</b>, above 1000× B12: Merlin's intervals are much narrower (width {c['wM']:.2f} "
            f"against {c['w40']:.2f}) but contain the truth only {100 * c['covM']:.0f} % of the time at 68 %, and "
            f"log p punishes the misses hard (worst row {c['minM']:.1f}). Merlin's log p is clearly higher (by "
            f"more than twice the paired error) only at {leads}{level}.",
            'These are observations, not established causes. The training recipes also differ (40 bins trained '
            'for several hundred epochs with sbi\'s trainer, Merlin and PCA for about 100), and the log p curves '
            'have not levelled off with the number of simulations, so log p reacts more to these differences than '
            'the marginal metrics. A Merlin run with a zero-centred asinh and a matched trainer would test how much '
            'of the gap remains.')
    S.append(section('comparison', '12', 'Merlin, PCA and 40 bins compared', P(
        'All numbers are on the same 3,000 test rows, for bounded three-seed ensembles. The figures use '
        'beam-consistent noise: Merlin and PCA with noise-only whitening, 40 bins the beam-consistent band powers.'
        + (f' The 40-band run at the full 61,440 rows is still in progress; its points are at N = {n40:,}, and its '
           f'interval score needs per-row data that is not in yet.' if n40 != 61440 else '')) +
        fig('compare_scorecard_beam.png', 'All metrics at the largest training set, one panel each.', wide=True) +
        fig('compare_sizes_beam.png', 'Metrics against the number of training simulations (B12 whitening for the '
            'Merlin and PCA curves; squares: noise-only whitening at 61,440).', wide=True) +
        fig('compare_params_beam.png', 'Pearson r per parameter' + (f' (40 bins at N = {n40:,}).' if n40 != 61440 else '.'),
            wide=True) +
        P(f'Paired differences on the same test rows, with the errors of section 11.8 ({set_label}):') +
        pairs_table(mset) +
        sub('paradox', 'Why log p and the marginal metrics disagree', P(
            'The 40-band network has the highest mean log p but the weakest marginal metrics. Splitting the per-row '
            f'log p by the brightness of the test sky ({set_label}) shows where the gap comes from; the last two '
            'columns give each bin\'s share of the difference in the mean:') + contrib + regimes) +
        sub('production', 'The same comparison with the production noise', P(
            'Before the beam correction all three compressions were trained on the production spectra (Merlin and '
            'PCA with B12 whitening). The ranking is the same.') + production +
            fig('compare_sizes_production.png', 'Metrics against training-set size, production noise.', wide=True))))
    # ------------------------------------------------------------------ 13 observations
    S.append(section('observations', '13', 'Posteriors for single observations', P(
        'A mock observation made with the production recipe goes through the same steps as a training row. Its '
        'spectrum is made beam-consistent with its own noise spectrum; a real SO map would already be '
        'beam-consistent and would enter at the whitening step. The spectrum is then whitened, projected, '
        'standardised with the stored parameters, and passed to the ensemble. Rebuilding the B12 mock this way '
        'reproduces its stored network input exactly, which checks the chain.',
        'The excerpt also builds a pixel-window variant for the FLAMINGO map. That map is a pixel average at '
        'NSIDE 4096, while the training signal is evaluated at pixel centres, so the variant divides its signal '
        'term by \\(p_\\ell^2\\) and its signal × noise term by \\(p_\\ell\\) (approximate under the mask).') +
        code('obs_beam_posteriors.py', [(68, 71), (78, 88)]) +
        fig('obs_b12_three.png', 'Posterior of the B12 mock for the three compressions (beam-consistent noise, '
            'bounded, 300,000 draws each); the star marks the input values.', wide=True) +
        fig('fig_pub_corner_FLAMINGO_pixwin_merlin_beam_whitening.png', 'The FLAMINGO fiducial map (L1_m9, pixel '
            'window removed) through Merlin with B12 and noise-only whitening; the star marks the gNFW fit to the '
            'stacked FLAMINGO pressure profiles.', source='pub', wide=True)))
    # ------------------------------------------------------------------ 14 reproduce
    S.append(section('reproduce', '14', 'Reproducing the numbers', P(
        'All analysis ran on the idark cluster; only results were downloaded. The scripts below are in '
        '<code>SBI_analysis/tsz64k_sbi_local</code>.') +
        table(['step', 'script', 'output'], [
            ['beam-consistent noise', '<code>beamfix/regen_noise.jl</code>, <code>beamfix/beamfix.py</code>', 'staged 40-band series'],
            ['Merlin/PCA inputs', '<code>merlin_beam_prepare.py</code>', 'whitening, PCA, features per input set'],
            ['training, evaluation', '<code>merlin_compression.py</code> (Merlin, PCA; run with <code>MERLIN_PATIENCE=30 '
             'MERLIN_LR_PATIENCE=10</code> and the results folder\'s <code>MERLIN_TAG</code>, e.g. '
             '<code>_beam_nnwhite_asinh_p30</code>), <code>train_ensemble.py</code> (40 bins)', 'metrics_*.json'],
            ['per-row posteriors', '<code>posterior_rows.py</code>, <code>doc_metric_data.py</code>', 'ranks, seed means, per-row log p'],
            ['observations', '<code>obs_beam_posteriors.py</code>', 'B12 and FLAMINGO posteriors'],
            ['comparisons and figures', '<code>whitening_comparison.py</code>, <code>doc_figures.py</code>, <code>doc_build.py</code>', 'this page']])))
    return S


def interval_score_lines():
    lines = (HERE / 'doc_figures.py').read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('def interval_score'))
    end = start + 1
    while end < len(lines) and (lines[end].startswith(' ') or not lines[end].strip()):
        end += 1
    while not lines[end - 1].strip():
        end -= 1
    return [(start + 1, end)]


# ----------------------------------------------------------------------------------------------- diagrams
def pipeline_svg():
    """The pipeline as an SVG flow diagram in three lanes (theme colours through CSS classes)."""
    top = [(10, ['Sobol design', '65,536 θ in the prior box']), (200, ['Paint y map', 'NSIDE 8192 → 2′ beam → 4096']),
           (390, ['Mask + 2 SO noise splits', 'cross spectrum, ℓ 80–7979']), (580, ['Beam-consistent noise', 'C_ss + B C_sn + B² C_nn']),
           (770, ['Split', '3,000 test · 61,440 training'])]
    p = ['<svg viewBox="0 0 980 430" role="img" aria-label="Pipeline from design to metrics" class="diagram">',
         '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="arrowhead"/></marker></defs>']

    def box(x, y, w, h, l1, l2, cls='box'):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="node {cls}"/>')
        p.append(f'<text x="{x + w / 2}" y="{y + (h / 2 - 3 if l2 else h / 2 + 5)}" class="t1">{html.escape(l1)}</text>')
        if l2:
            p.append(f'<text x="{x + w / 2}" y="{y + h / 2 + 14}" class="t2">{html.escape(l2)}</text>')

    def path(d):
        p.append(f'<path d="{d}" class="edge" marker-end="url(#arr)"/>')
    for k, (x, (l1, l2)) in enumerate(top):
        box(x, 16, 175, 56, l1, l2)
        if k:
            path(f'M{x - 15},44 L{x - 2},44')
    # a bus from the split down the right edge and across to the three lanes
    path('M857,72 L857,96 L40,96 L40,140 L58,140')
    path('M40,140 L40,214 L58,214')
    path('M40,214 L40,288 L58,288')
    box(60, 116, 300, 48, '40 band powers', 'asinh at the noise level → z-score', 'bins40')
    box(60, 190, 300, 48, 'Whiten L⁻¹x → PCA (75–83)', 'centred asinh → z-score', 'pca')
    box(60, 264, 140, 48, 'same PCA inputs', '', 'pca')
    box(222, 264, 196, 48, 'Merlin head', '2048 → 512 → 18 features', 'merlin')
    path('M200,288 L220,288')
    path('M360,140 L610,140 L610,180 L640,180')
    path('M360,214 L640,214')
    path('M418,288 L610,288 L610,248 L640,248')
    box(642, 164, 300, 100, 'Masked autoregressive flow', 'q(u | x), 5 × 128, logit u')
    box(642, 296, 300, 44, '3 seeds → equal-weight ensemble', '')
    path('M792,264 L792,294')
    box(300, 366, 380, 50, 'Test metrics on 3,000 rows', 'log p · r · RMSE · width · coverage · TARP · SBC')
    path('M792,340 L792,391 L682,391')
    p.append('<text x="372" y="133" class="lab">A</text><text x="372" y="207" class="lab">B</text>'
             '<text x="430" y="281" class="lab">C</text>')
    p.append('</svg>')
    return f'<figure class="fig wide diagram-wrap">{"".join(p)}<figcaption>The pipeline. Compression A feeds 40 band '\
           f'powers to the flow, B the whitened principal components, and C passes the same components through '\
           f'Merlin\'s learned head first. Everything else is shared.</figcaption></figure>'


def network_svg(params):
    p = ['<svg viewBox="0 0 980 300" role="img" aria-label="The three network variants" class="diagram">',
         '<defs><marker id="arr2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
         '<path d="M0,0 L8,4 L0,8 z" class="arrowhead"/></marker></defs>']
    rows = [(20, '40 bins', 'bins40', [('40 inputs', 96)], '≈ 209 k parameters'),
            (110, 'PCA', 'pca', [('75–83 inputs', 110)], f'{params["pca"]:,} parameters'),
            (200, 'Merlin', 'merlin', [('75–83 inputs', 110), ('2048, ReLU', 96), ('512, ReLU', 96),
                                        ('18 + BatchNorm', 118)], f'{params["merlin"]:,} parameters')]
    for y, name, cls, blocks, count in rows:
        p.append(f'<text x="16" y="{y + 25}" class="t1 left {cls}-ink">{name}</text>')
        x = 100
        for label, w in blocks:
            p.append(f'<rect x="{x}" y="{y}" width="{w}" height="40" rx="5" class="node {cls}"/>')
            p.append(f'<text x="{x + w / 2}" y="{y + 25}" class="t2">{label}</text>')
            p.append(f'<line x1="{x + w}" y1="{y + 20}" x2="{x + w + 24}" y2="{y + 20}" class="edge" marker-end="url(#arr2)"/>')
            x += w + 26
        p.append(f'<rect x="{x}" y="{y}" width="250" height="40" rx="5" class="node box"/>')
        p.append(f'<text x="{x + 125}" y="{y + 25}" class="t2">MAF, 5 × 128 → q(u | context)</text>')
        p.append(f'<text x="{x + 125}" y="{y + 60}" class="t2">{count}</text>')
    p.append('</svg>')
    return f'<figure class="fig wide diagram-wrap">{"".join(p)}<figcaption>The three variants share the flow; only '\
           f'its conditioning input differs. Merlin\'s head is trained together with the flow. Parameter counts for '\
           f'noise-only whitening (83 inputs) from the training records; the 40-band count from the network '\
           f'definition.</figcaption></figure>'


# ----------------------------------------------------------------------------------------------- page shell
CSS = r'''
/* Layout: a reading column with a sticky contents rail on wide screens; figures may run wider than the text. */
:root {
  --bg: #f6f7f9; --surface: #ffffff; --ink: #18212c; --ink2: #4d5967; --rule: #dce1e7; --accent: #1d5b86;
  --code-bg: #eef1f5; --code-ink: #1e2a38; --kw: #7a3e9d; --str: #1f6f43; --com: #6b7684; --num: #a4500d;
  --merlin: #4a3aa7; --pca: #138a60; --bins40: #b77b00; --flag-bg: #fff2cc; --flag-ink: #6b4a00;
  --key-bg: #eef3fa; --plate: #ffffff;
  --display: "Spectral", "Georgia", serif; --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f141b; --surface: #151c25; --ink: #e5eaf0; --ink2: #a6b1be; --rule: #2a3442; --accent: #86b9e3;
  --code-bg: #1a2230; --code-ink: #dbe3ec; --kw: #c79ae6; --str: #86d2a3; --com: #8995a3; --num: #f0a868;
  --merlin: #9a8ff0; --pca: #45cf98; --bins40: #f2b733; --flag-bg: #3a3014; --flag-ink: #f5d68a;
  --key-bg: #172232; --plate: #f4f5f7; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0f141b; --surface: #151c25; --ink: #e5eaf0; --ink2: #a6b1be; --rule: #2a3442; --accent: #86b9e3;
  --code-bg: #1a2230; --code-ink: #dbe3ec; --kw: #c79ae6; --str: #86d2a3; --com: #8995a3; --num: #f0a868;
  --merlin: #9a8ff0; --pca: #45cf98; --bins40: #f2b733; --flag-bg: #3a3014; --flag-ink: #f5d68a;
  --key-bg: #172232; --plate: #f4f5f7; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font-family: var(--body); font-size: 17px; line-height: 1.6; }
.shell { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0 48px; max-width: 1240px; margin: 0 auto;
         padding-inline: 20px; padding-block: 28px 80px; }
@media (min-width: 1100px) { .shell { grid-template-columns: 230px minmax(0, 1fr); } }
nav.toc { font-size: 14px; line-height: 1.45; }
@media (min-width: 1100px) { nav.toc { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 24px);
  align-self: start; max-height: calc(100vh - 48px); overflow-y: auto; } }
nav.toc details > summary { cursor: pointer; font-weight: 600; color: var(--ink2); letter-spacing: .04em;
  text-transform: uppercase; font-size: 12px; margin-bottom: 8px; }
nav.toc ol { list-style: none; padding: 0; margin: 0; display: grid; gap: 4px; }
nav.toc a { color: var(--ink2); text-decoration: none; display: grid; grid-template-columns: 2.2em 1fr; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--accent); }
nav.toc .n { font-variant-numeric: tabular-nums; color: var(--ink2); opacity: .7 }
main { min-width: 0; }
header.title { padding-block: 8px 28px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; }
header.title .eyebrow { font-size: 12.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink2); margin: 0 0 10px; }
h1 { font-family: var(--display); font-weight: 600; font-size: clamp(30px, 4.4vw, 44px); line-height: 1.12;
     margin: 0 0 14px; text-wrap: balance; }
header.title p.lede { font-size: 19px; color: var(--ink2); max-width: 46em; margin: 0; }
h2 { font-family: var(--display); font-weight: 600; font-size: 28px; line-height: 1.2; margin: 56px 0 14px;
     text-wrap: balance; display: flex; gap: 14px; align-items: baseline; }
h2 .num { font-family: var(--mono); font-size: 15px; color: var(--accent); font-weight: 500; min-width: 1.8em; }
h3 { font-family: var(--display); font-weight: 600; font-size: 21px; margin: 36px 0 8px; }
section > p, section.sub > p, aside p, .callout > div { max-width: 44em; }
p { margin: 0 0 14px; }
a { color: var(--accent); }
code { font-family: var(--mono); font-size: .86em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
.kicker { font-size: 13px; text-transform: uppercase; letter-spacing: .07em; color: var(--ink2); margin: 48px 0 0; }
figure { margin: 22px 0; }
figure.code { max-width: 52em; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--code-bg); }
.code-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; padding: 7px 12px;
  font-family: var(--mono); font-size: 12.5px; color: var(--ink2); border-bottom: 1px solid var(--rule); background: var(--surface); }
.code-head .file { color: var(--ink); word-break: break-all; }
figure.code pre { margin: 0; padding: 12px 14px; overflow-x: auto; font-size: 13px; line-height: 1.5; }
figure.code pre code { background: none; padding: 0; color: var(--code-ink); font-size: 13px; white-space: pre; }
.code-note { font-size: 13.5px; color: var(--ink2); padding: 6px 12px 10px; margin: 0; }
.hljs-keyword, .hljs-built_in, .hljs-literal { color: var(--kw) } .hljs-string { color: var(--str) }
.hljs-comment { color: var(--com); font-style: italic } .hljs-number { color: var(--num) }
.hljs-title, .hljs-title.function_ { color: var(--accent) }
figure.fig { max-width: 52em; } figure.fig.wide { max-width: 100%; }
figure.fig .plate { background: var(--plate); border: 1px solid var(--rule); border-radius: 8px; padding: 10px; }
figure.fig img { display: block; width: 100%; height: auto; }
figure.missing .plate { padding: 28px; text-align: center; color: #6b4a00; background: #fff2cc; font-family: var(--mono); font-size: 13px; }
figcaption { font-size: 14.5px; color: var(--ink2); margin-top: 8px; max-width: 52em; line-height: 1.5; }
.math { overflow-x: auto; max-width: 52em; padding: 4px 0; }
.table-wrap { overflow-x: auto; margin: 16px 0 20px; max-width: 100%; }
table { border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--ink2); font-size: 13px; letter-spacing: .02em; }
table.data td:not(:first-child), table.data th:not(:first-child) { text-align: right; white-space: nowrap; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; vertical-align: 1px; }
.dot.merlin { background: var(--merlin) } .dot.pca { background: var(--pca) } .dot.bins40 { background: var(--bins40) }
.flag { background: var(--flag-bg); color: var(--flag-ink); font-size: 12px; padding: 1px 6px; border-radius: 4px; margin-left: 6px; }
aside.callout { border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px 6px; margin: 24px 0; background: var(--surface); max-width: 100%; }
aside.callout.key { background: var(--key-bg); }
aside.callout.warn { border-color: var(--bins40); }
.callout-title { font-weight: 600; font-size: 14px; letter-spacing: .05em; text-transform: uppercase; color: var(--ink2); margin-bottom: 8px; }
svg.diagram { width: 100%; height: auto; display: block; }
.diagram-wrap .node { fill: var(--surface); stroke: var(--rule); stroke-width: 1.4; }
.diagram-wrap .node.bins40 { stroke: var(--bins40); stroke-width: 2 } .diagram-wrap .node.pca { stroke: var(--pca); stroke-width: 2 }
.diagram-wrap .node.merlin { stroke: var(--merlin); stroke-width: 2 }
.diagram-wrap .edge { stroke: var(--ink2); stroke-width: 1.3; fill: none } .diagram-wrap .arrowhead { fill: var(--ink2) }
.diagram-wrap text { fill: var(--ink); text-anchor: middle; font-family: var(--body) }
.diagram-wrap text.t1 { font-size: 15px; font-weight: 600 } .diagram-wrap text.t2 { font-size: 13px; fill: var(--ink2) }
.diagram-wrap text.left { text-anchor: start } .diagram-wrap text.lab { font-size: 12.5px; fill: var(--ink2); text-anchor: start; font-style: italic }
.diagram-wrap .bins40-ink { fill: var(--bins40) } .diagram-wrap .pca-ink { fill: var(--pca) } .diagram-wrap .merlin-ink { fill: var(--merlin) }
.diagram-wrap .plate { background: none; border: 0; padding: 0 }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (max-width: 640px) { body { font-size: 16px } h2 { font-size: 23px } figure.code pre code { font-size: 12px } }
'''

TOC = [('overview', '', 'What this explains'), ('design', '1', 'Prior and design'), ('spectra', '2', 'Map to spectrum'),
       ('beam', '3', 'Beam-consistent noise'), ('bins40', '4', '40 band powers'), ('whitening', '5', 'Whitening'),
       ('pca', '6', 'Principal components'), ('asinh', '7', 'asinh standardisation'), ('networks', '8', 'Networks'),
       ('logit', '9', 'Bounded and unbounded'), ('training', '10', 'Training and ensembles'),
       ('metrics', '11', 'Test metrics'), ('comparison', '12', 'Comparison'), ('observations', '13', 'Observations'),
       ('reproduce', '14', 'Reproducing')]


def page(sections):
    toc = ''.join(f'<li><a href="#{sid}"><span class="n">{n}</span><span>{t}</span></a></li>' for sid, n, t in TOC)
    return f'''<title>HalfDome tSZ Inference Pipeline</title>
<meta name="description" content="How the Merlin, PCA and 40-band posteriors and their test metrics are computed, step by step.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=Spectral:wght@500;600&display=swap">
<style>{CSS}</style>
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }}, svg: {{ fontCache: 'global' }} }};</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js" defer></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/julia.min.js"></script>
<div class="shell">
<nav class="toc" aria-label="Contents"><details open><summary>Contents</summary><ol>{toc}</ol></details></nav>
<main>
<header class="title">
<p class="eyebrow">HalfDome · SO tSZ power spectrum · neural posterior estimation</p>
<h1>HalfDome tSZ Inference Pipeline</h1>
<p class="lede">How a painted tSZ map becomes a posterior on nine pressure-profile parameters, and how the test metrics
that compare Merlin, whitened PCA and 40 band powers are computed, with the code that does each step.</p>
</header>
{''.join(sections)}
</main></div>
<script>document.addEventListener('DOMContentLoaded', () => {{ try {{ document.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el)); }} catch (e) {{}} }});</script>
'''


def main():
    sections = build()
    OUT.mkdir(exist_ok=True)
    (OUT / 'index.html').write_text(page(sections))
    size = sum(f.stat().st_size for f in FIG.glob('*')) / 1e6
    print(f'wrote {OUT / "index.html"} ({(OUT / "index.html").stat().st_size / 1e3:.0f} kB), figures {size:.1f} MB')
    if MISSING:
        print('pending figures:', ', '.join(MISSING))


if __name__ == '__main__':
    main()
