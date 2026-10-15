"""Where the degeneracies come from: one-parameter scans of the SO tSZ spectrum around Battaglia+12.

Each of the nine parameters at 10, 35, 65 and 90 % of its prior range with the other eight at the B12 values, painted
with the frozen 64k engine (mocks/idark_param_scan, paint_mocks.jl: raw NSIDE 8192 -> 2 arcmin beam -> NSIDE 4096,
f_sky 0.4 mask, SO baseline Deproj-0 noise with the B12 observation's seeds 39001001/39001002, lmax 7979), next to the
B12 mock itself (mocks/pass1_B12/B12) and its 21 noise draws (the noise scatter; the sky is the same in every SBI
simulation, so this is the whole scatter).

Figures (results_param_scan/)
  fig_param_scan_binned.png      3 x 3, one panel per parameter: the 40 band powers the SBI sees (noisy markers, the
                                 same maps without noise as lines) and their ratio to B12 without noise, with the
                                 +-1 sigma noise band of one band power
  fig_param_scan_unbinned.png    the same at every multipole ell = 80..7979 (the Merlin / PCA / no-binning inputs)
  fig_param_scan_degeneracies.png  the local responses at B12 (central differences of the Fisher mocks
                                 mocks/fisher_b12, steps 0.5 % of the prior width; P0 analytic, D ~ P0^2) in noise
                                 units: their shapes (unit norm), the cosine similarity of every pair (|cos| -> 1: the
                                 data cannot tell the two apart near B12), the local (Fisher) errors of the nine
                                 eigen-combinations and of each parameter with the other eight fixed or marginalized,
                                 in prior widths. Noise: the 21 B12 draws (band sigma smoothed over 5 bands; per
                                 multipole over 51, and x1.6 for the measured lag-1 correlation of neighbouring
                                 multipoles, 0.55); 'every multipole, smoothed' removes the derivatives' structure
                                 below 100 in ell (the imprint of the one HalfDome sky every simulation shares)
  param_scan_summary.json        the numbers behind the degeneracy figure

    python param_scan_figures.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCAN = HERE / 'mocks' / 'idark_param_scan'
FISHER = HERE / 'mocks' / 'fisher_b12'
B12_DIR = HERE / 'mocks' / 'pass1_B12' / 'B12'
OUT = HERE / 'results_param_scan'
LMAX = 7979
ELL = np.arange(LMAX + 1)
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES = np.r_[np.arange(80, 7881, 200), 7980]              # the SBI's 40 bands
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
SBI = slice(80, LMAX + 1)                                  # the multipoles the SBI uses
NOISY = 'noisy_cross_39001001_39001002.npy'

INK, INK2, GRID, SURFACE, BAND = '#222222', '#6b6b6b', '#e6e6e3', '#fcfcfb', '#c9c8c3'
RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']        # 10 / 35 / 65 / 90 % of the prior range, light -> dark
GROUP_COLOURS = ['#2a78d6', '#1baf7a', '#4a3aa7']          # amplitude, mass slope, redshift slope within a group
TEX = [r'$P_0$', r'$x_c$', r'$\beta$', r'$\alpha_m^{P_0}$', r'$\alpha_m^{x_c}$', r'$\alpha_m^{\beta}$',
       r'$\alpha_z^{P_0}$', r'$\alpha_z^{x_c}$', r'$\alpha_z^{\beta}$']


def bin40(cl):
    """The SBI's 40 band powers (D_ell, (2 ell + 1) weights) of a C_ell array starting at ell = 0."""
    dl = np.asarray(cl)[:LMAX + 1] * DL
    w = 2 * ELL + 1
    return np.array([np.sum(dl[a:b] * w[a:b]) / np.sum(w[a:b]) for a, b in zip(EDGES[:-1], EDGES[1:])])


def running_median(x, half=25):
    """Running median over +-half samples (smooths the 21-draw scatter estimate of single multipoles)."""
    padded = np.pad(x, half, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)


def load():
    """The scan cases, the B12 reference with its noise scatter, and the local derivatives at B12."""
    meta = json.loads((SCAN / 'cases.json').read_text())
    ref = dict(clean=np.load(B12_DIR / 'masked_clean_cl.npy')[:LMAX + 1], noisy=np.load(B12_DIR / NOISY)[:LMAX + 1])
    draws = np.array([np.load(p)[:LMAX + 1] for p in sorted(B12_DIR.glob('noisy_cross_*.npy'))])
    sigma_band = running_median(np.array([bin40(d) for d in draws]).std(0, ddof=1), half=2)
    sigma_ell = running_median((draws * DL).std(0, ddof=1)[SBI])
    scans, missing = {k: [] for k in range(9)}, []
    for label, case in meta['cases'].items():
        folder = SCAN / 'out' / label
        if (folder / NOISY).exists():
            scans[case['index']].append(dict(q=case['quantile'], value=case['value'],
                                             clean=np.load(folder / 'masked_clean_cl.npy')[:LMAX + 1],
                                             noisy=np.load(folder / NOISY)[:LMAX + 1]))
        else:
            missing.append(label)
    for k in scans:
        scans[k].sort(key=lambda c: c['q'])
    plan = json.loads((FISHER / 'plan.json').read_text())
    derivative = []                                            # d C_ell / d theta_k (masked, beamed, no noise)
    for k, name in enumerate(plan['names']):
        if name == 'P0':
            derivative.append(2 * ref['clean'] / plan['fiducial'][0])
        else:
            plus = np.load(FISHER / f'{name}_p' / 'masked_clean_cl.npy')[:LMAX + 1]
            minus = np.load(FISHER / f'{name}_m' / 'masked_clean_cl.npy')[:LMAX + 1]
            derivative.append((plus - minus) / (2 * plan['step'][k]))
    return meta, ref, sigma_band, sigma_ell, scans, np.array(derivative), missing


def style_axes(ax, xlabels=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, lw=0.7)
    ax.tick_params(colors=INK2, labelsize=8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
    ax.set_xscale('log')
    ax.set_xlim(75, 8100)
    ax.set_xticks([100, 300, 1000, 3000])
    ax.set_xticklabels(['100', '300', '1000', '3000'] if xlabels else [])
    ax.minorticks_off()


def panel_title(meta, k):
    values = ', '.join(f"{c['value']:.3g}" for c in sorted((c for c in meta['cases'].values() if c['index'] == k),
                                                          key=lambda c: c['quantile']))
    return f"{TEX[k]} = {values}   (B12: {meta['b12'][k]:.3g})"


def figure_scan(meta, ref, sigma_band, sigma_ell, scans, plt, unbinned):
    """3 x 3 panels: spectrum (top, log) and ratio to B12 without noise (bottom), each parameter varied alone."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(17, 15.5))
    fig.patch.set_facecolor(SURFACE)
    outer = GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.16, top=0.905, bottom=0.06, left=0.05, right=0.99)
    if unbinned:
        x = ELL[SBI]
        ref_clean, ref_noisy = (ref['clean'] * DL)[SBI], (ref['noisy'] * DL)[SBI]
        sigma = sigma_ell
    else:
        x = CENTRES
        ref_clean, ref_noisy = bin40(ref['clean']), bin40(ref['noisy'])
        sigma = sigma_band
    for k in range(9):
        inner = outer[k].subgridspec(2, 1, height_ratios=[1.55, 1], hspace=0.06)
        ax, rx = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        curves = [(c, RAMP[i]) for i, c in enumerate(scans[k])] + [(None, INK)]
        for case, colour in curves:
            if case is None:
                clean, noisy = ref_clean, ref_noisy
            elif unbinned:
                clean, noisy = (case['clean'] * DL)[SBI], (case['noisy'] * DL)[SBI]
            else:
                clean, noisy = bin40(case['clean']), bin40(case['noisy'])
            lw = 1.6 if case is None else 1.3
            if unbinned:
                trace = np.where(noisy > 0, noisy, np.nan)        # log axis: negative multipoles leave gaps
                ax.plot(x, 1e12 * trace, color=colour, lw=0.3, alpha=0.45, zorder=1)
                ax.plot(x, 1e12 * clean, color=colour, lw=lw, zorder=3, ls='--' if case is None else '-')
                rx.plot(x, clean / ref_clean, color=colour, lw=lw, zorder=3, ls='--' if case is None else '-')
            else:
                ax.plot(x, 1e12 * clean, color=colour, lw=lw, zorder=2, ls='--' if case is None else '-')
                pos = noisy > 0
                ax.plot(x[pos], 1e12 * noisy[pos], ls='none', marker='o', ms=2.6, color=colour, zorder=3)
                rx.plot(x, clean / ref_clean, color=colour, lw=lw, zorder=2, ls='--' if case is None else '-')
                rx.plot(x, noisy / ref_clean, ls='none', marker='o', ms=2.4, color=colour, zorder=3)
        # +-1 sigma noise of one band (one multipole) around B12 without noise
        lo = np.clip(1 - sigma / ref_clean, 1e-3, None)
        rx.fill_between(x, lo, 1 + sigma / ref_clean, color=BAND, alpha=0.75, lw=0, zorder=1)
        ratios = np.concatenate([(bin40(c['clean']) if not unbinned else (c['clean'] * DL)[SBI]) / ref_clean
                                 for c in scans[k]] + [np.ones(1)])
        finite = ratios[np.isfinite(ratios) & (ratios > 0)]
        top, bottom = max(1.25, finite.max() * 1.15), min(0.8, finite.min() / 1.15)
        rx.set_yscale('log')
        rx.set_ylim(bottom, top)
        rx.axhline(1, color=INK, lw=0.6, zorder=2)
        ax.set_yscale('log')
        ax.set_ylim(2e-3 if not unbinned else 5e-3, None)
        style_axes(ax, xlabels=False)
        style_axes(rx)
        ax.set_title(panel_title(meta, k), color=INK, fontsize=10.5, loc='left')
        if k % 3 == 0:
            ax.set_ylabel(r'$D_\ell$ [$10^{-12}$]', color=INK, fontsize=9.5)
            rx.set_ylabel('ratio to B12', color=INK, fontsize=9.5)
        if k >= 6:
            rx.set_xlabel(r'multipole $\ell$', color=INK, fontsize=9.5)
        from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
        span = np.log10(top / bottom)                      # decades shown: fewer ticks for wide ranges
        rx.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5) if span < 1.2 else (1, 3) if span < 2.2 else (1,)))
        rx.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        rx.yaxis.set_minor_formatter(NullFormatter())
    # one legend for the figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=c, lw=2) for c in RAMP] + [Line2D([], [], color=INK, lw=1.6, ls='--'),
                                                              Patch(color=BAND, alpha=0.75)]
    labels = ['10 % of the prior range', '35 %', '65 %', '90 %', 'B12 (all nine at the B12 values)',
              f"±1σ noise of one {'multipole' if unbinned else 'band power'} (21 noise draws)"]
    if unbinned:
        handles.insert(5, Line2D([], [], color=INK2, lw=0.6, alpha=0.6))
        labels.insert(5, 'thin: noisy, every multipole (gaps: negative)')
    else:
        handles.insert(5, Line2D([], [], color=INK2, ls='none', marker='o', ms=4))
        labels.insert(5, 'markers: noisy band powers (the SBI inputs)')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=4, fontsize=9.5, frameon=False,
               labelcolor=INK)
    what = 'every multipole ℓ = 80–7979, with noise' if unbinned else 'the 40 band powers, with noise'
    fig.suptitle(f'One parameter at a time across its prior range, the other eight at B12 ({what}): beamed, masked, '
                 f'SO noise; lines without noise', color=INK, fontsize=12.5, x=0.05, ha='left', y=0.99)
    name = 'fig_param_scan_unbinned.png' if unbinned else 'fig_param_scan_binned.png'
    fig.savefig(OUT / name, dpi=120, facecolor=SURFACE)
    plt.close(fig)


CORRELATION = 1.6        # measured band sigma / its independent-multipole expectation (lag-1 correlation 0.55)
FLAT_SD = 1 / np.sqrt(12)                                  # sd of a flat prior, in prior widths


def smooth_rows(x, half):
    """Running mean of each row over +-half samples (edges padded with the edge value)."""
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.array([np.convolve(np.pad(r, half, mode='edge'), kernel, mode='valid') for r in x])


def degeneracy_numbers(ref, sigma_band, sigma_ell, derivative, width):
    """Whitened local responses (per 10 % of the prior range), their cosine similarities, and the local Fisher errors
    (in prior widths) of each parameter (others fixed / marginalized) and of the eigen-combinations."""
    per_ell = (derivative * DL)[:, SBI]
    vectors = {'40 band powers': np.array([bin40(d) for d in derivative]) / sigma_band,
               'every multipole': per_ell / (CORRELATION * sigma_ell),
               'every multipole, smoothed': smooth_rows(per_ell, 50) / (CORRELATION * sigma_ell)}
    out = {}
    for name, v in vectors.items():
        r = v * (0.1 * width)[:, None]                         # response to a change of 10 % of the prior range
        norm = np.linalg.norm(r, axis=1)
        j = v * width[:, None]                                 # per prior width
        fisher = j @ j.T
        eigen = np.linalg.eigh(fisher)
        out[name] = dict(response=r, norm=norm, cosine=(r @ r.T) / np.outer(norm, norm),
                         conditional=1 / np.sqrt(np.diag(fisher)), marginal=np.sqrt(np.diag(np.linalg.inv(fisher))),
                         eigen_sigma=1 / np.sqrt(eigen[0][::-1]), eigen_vectors=eigen[1][:, ::-1].T)
    return out


def figure_degeneracies(numbers, plt):
    """Top: unit-norm whitened response shapes (40 bands) by group. Bottom: their cosine similarities, the local
    errors of the nine eigen-combinations, and each parameter's local error with the others fixed or marginalized."""
    from matplotlib.colors import LinearSegmentedColormap
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.5), gridspec_kw=dict(height_ratios=[1, 1.2], hspace=0.36,
                                                                         wspace=0.3))
    fig.patch.set_facecolor(SURFACE)
    groups = [('amplitude', (0, 3, 6)), ('core size', (1, 4, 7)), ('outer slope', (2, 5, 8))]
    b = numbers['40 band powers']
    for ax, (title, members) in zip(axes[0], groups):
        for colour, k in zip(GROUP_COLOURS, members):
            ax.plot(CENTRES, b['response'][k] / b['norm'][k], color=colour, lw=1.8, marker='o', ms=3,
                    label=f"{TEX[k]}  (10 % of its range: {b['norm'][k]:.0f}\u03c3 in all)")
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.7)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xscale('log')
        ax.set_xlim(150, 8100)
        ax.set_xticks([200, 500, 1000, 2000, 5000])
        ax.set_xticklabels(['200', '500', '1000', '2000', '5000'])
        ax.minorticks_off()
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.set_title(f'{title}: value, mass slope, redshift slope', color=INK, fontsize=10.5, loc='left')
        ax.set_xlabel(r'multipole $\ell$ (band centre)', color=INK, fontsize=9.5)
        ax.legend(fontsize=8.5, frameon=False, labelcolor=INK, loc='lower center')
    axes[0, 0].set_ylabel('response / band noise, unit length', color=INK, fontsize=9.5)
    cmap = LinearSegmentedColormap.from_list('div', ['#0d366b', '#3987e5', '#86b6ef', '#f0efec', '#f2a3a2', '#e34948',
                                                     '#8f1f1e'])
    ax = axes[1, 0]
    c = b['cosine']
    image = ax.imshow(c, cmap=cmap, vmin=-1, vmax=1)
    for i in range(9):
        for j in range(9):
            if i != j:
                ax.text(j, i, f'{c[i, j]:+.2f}', ha='center', va='center', fontsize=7.5,
                        color='white' if abs(c[i, j]) > 0.7 else INK)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(TEX, fontsize=9)
    ax.set_yticklabels(TEX, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('cosine similarity of the whitened responses (40 bands)\n\u00b11: the same change of the spectrum, '
                 'up to scale', color=INK, fontsize=10, loc='left')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    styles = {'40 band powers': ('#3987e5', 'o'), 'every multipole': ('#0d366b', 's'),
              'every multipole, smoothed': ('#86b6ef', 'D')}
    ax = axes[1, 1]
    for name, (colour, marker) in styles.items():
        ax.plot(np.arange(1, 10), numbers[name]['eigen_sigma'], color=colour, marker=marker, ms=6, lw=1.4, label=name)
    ax.axhline(FLAT_SD, color=INK, lw=0.9, ls='--')
    ax.text(9.1, FLAT_SD * 1.15, 'flat prior sd', color=INK2, fontsize=8, ha='right')
    ax.set_yscale('log')
    ax.set_xticks(range(1, 10))
    ax.set_xlabel('eigen-combination of the nine parameters (best measured first)', color=INK, fontsize=9.5)
    ax.set_ylabel('local error, prior widths', color=INK, fontsize=9.5)
    ax.set_title('how many combinations the spectrum measures\n(Fisher at B12; below the dashed line: better than the prior)',
                 color=INK, fontsize=10, loc='left')
    ax.legend(fontsize=8.5, frameon=False, labelcolor=INK, loc='upper left')
    ax = axes[1, 2]
    y = np.arange(9)
    ax.scatter(b['conditional'], y, color=INK, marker='|', s=120, lw=2, label='40 bands, other eight fixed', zorder=3)
    for offset, (name, (colour, marker)) in zip((-0.22, 0.0, 0.22), styles.items()):
        ax.scatter(numbers[name]['marginal'], y + offset, color=colour, marker=marker, s=34, zorder=3,
                   label=f'{name}, marginalized')
    ax.axvline(FLAT_SD, color=INK, lw=0.9, ls='--')
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels(TEX, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('local error, prior widths (dashed: flat prior sd)', color=INK, fontsize=9.5)
    ax.set_title('each parameter alone vs marginalized over\nthe other eight (Fisher at B12)', color=INK, fontsize=10,
                 loc='left')
    ax.legend(fontsize=7.8, frameon=False, labelcolor=INK, loc='lower left', bbox_to_anchor=(0.0, -0.42), ncol=2)
    for a in (axes[1, 1], axes[1, 2]):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, lw=0.7)
        a.tick_params(colors=INK2, labelsize=8.5)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
    fig.suptitle('Where the degeneracies come from: local responses of the SBI spectrum at B12 (Fisher mocks, steps '
                 '0.5 % of the prior width), in units of the SO noise', color=INK, fontsize=12.5, x=0.05, ha='left', y=0.995)
    fig.savefig(OUT / 'fig_param_scan_degeneracies.png', dpi=120, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(exist_ok=True)
    meta, ref, sigma_band, sigma_ell, scans, derivative, missing = load()
    figure_scan(meta, ref, sigma_band, sigma_ell, scans, plt, unbinned=False)
    figure_scan(meta, ref, sigma_band, sigma_ell, scans, plt, unbinned=True)
    width = np.array(meta['high']) - np.array(meta['low'])
    numbers = degeneracy_numbers(ref, sigma_band, sigma_ell, derivative, width)
    figure_degeneracies(numbers, plt)
    summary = dict(missing=missing, names=meta['names'],
                   **{key: dict(norm_sigma_per_10pc=v['norm'].tolist(), cosine=v['cosine'].round(4).tolist(),
                                conditional_prior_widths=v['conditional'].tolist(),
                                marginal_prior_widths=v['marginal'].tolist(), eigen_sigma=v['eigen_sigma'].tolist(),
                                eigen_vectors=v['eigen_vectors'].round(3).tolist()) for key, v in numbers.items()})
    (OUT / 'param_scan_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('missing scan cases:', missing or 'none')
    print('written:', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
