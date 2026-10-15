"""Figures of the pipeline explainer (results_doc/): one per pipeline step, from local data and doc_figure_data/.

    python doc_figures.py [--only NAME ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import publication_figures as PF

HERE = PF.HERE
OUT = HERE / 'results_doc'
DATA = HERE / 'doc_figure_data'
OBS = HERE.parent / 'tsz_diagnostic_256_accelerated_20260922' / 'cluster_snapshot' / 'diagnostic_256' / 'observations'
ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
INK, INK2, GRID = PF.INK, PF.INK2, PF.GRID
C = dict(merlin='#4a3aa7', pca='#1baf7a', bins40='#eda100', moped='#8a8a85', red='#e34948', blue='#2a78d6')
SCALE = 1e12                                               # D_ell in units of 10^-12


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)


def rebin40(x):
    w = 2 * ELL + 1
    return np.stack([np.sum(x[..., a - 80:b - 80] * w[a - 80:b - 80], axis=-1) / np.sum(w[a - 80:b - 80])
                     for a, b in zip(EDGES[:-1], EDGES[1:])], axis=-1)


def b12_spectra():
    """The B12 observation at each step (D_ell, ell 80..7979)."""
    d = np.load(OBS / 'Battaglia12.npz')
    nn = np.load(DATA / 'B12_39001001_39001002.npy')[ELL] * DL
    bl = np.load(DATA / 'beam_bl.npy')[ELL]
    clean, noisy = d['clean_dl_unbinned'], d['noisy_dl_unbinned']
    beam = clean + bl * (noisy - clean - nn) + bl ** 2 * nn
    return dict(clean=clean, noisy=noisy, nn=nn, sn=noisy - clean - nn, beam=beam, bl=bl)


def running_rms(x, width=100):
    """rms over a sliding window of `width` multipoles."""
    kernel = np.ones(width) / width
    return np.sqrt(np.convolve(x ** 2, kernel, mode='same'))


def fig_spectra(plt):
    """Steps 2-4: the B12 spectrum noise-free, observed (production) and beam-consistent; the noise terms; 40 bands."""
    s = b12_spectra()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axes[0]
    ax.plot(ELL, s['noisy'] * SCALE, color=C['red'], lw=0.3, alpha=0.75, label='observed (production noise)')
    ax.plot(ELL, s['beam'] * SCALE, color=C['blue'], lw=0.3, alpha=0.75, label='beam-consistent noise')
    ax.plot(ELL, s['clean'] * SCALE, color=INK, lw=1.0, label='noise-free (masked)')
    ax.set_xlim(80, 7980)
    ax.set_ylim(-0.25, 0.62)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$D_\ell\;[10^{-12}]$')
    ax.legend(loc='upper right', fontsize=6.5)
    ax = axes[1]
    sn, nn, bl = s['sn'], s['nn'], s['bl']
    ax.plot(ELL, s['clean'] * SCALE, color=INK, lw=1.0, label='signal')
    ax.plot(ELL, running_rms(sn + nn) * SCALE, color=C['red'], lw=1.0, label=r'production: $C^{sn}+C^{nn}$')
    ax.plot(ELL, running_rms(bl * sn + bl ** 2 * nn) * SCALE, color=C['blue'], lw=1.0,
            label=r'beam-consistent: $B C^{sn}+B^2 C^{nn}$')
    ax.plot(ELL, running_rms(nn) * SCALE, color=INK2, lw=0.8, ls=(0, (3, 1.5)), label=r'$C^{nn}$ alone')
    ax.set_yscale('log')
    ax.set_xlim(80, 7980)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'rms over 100 multipoles $[10^{-12}]$')
    ax.legend(loc='lower left', fontsize=6, handlelength=1.5)
    ax = axes[2]
    b40, c40 = rebin40(s['beam']), rebin40(s['clean'])
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    ax.plot(ELL, s['beam'] * SCALE, color=C['blue'], lw=0.3, alpha=0.3, label='per multipole')
    ax.step(EDGES, np.r_[b40, b40[-1]] * SCALE, where='post', color=C['bins40'], lw=1.3, label='40 band powers')
    ax.plot(mid, c40 * SCALE, ls='none', marker='o', ms=2.2, color=INK, label='noise-free bands')
    for e in EDGES[::5]:
        ax.axvline(e, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(80, 7980)
    ax.set_ylim(-0.25, 0.62)
    ax.set_xlabel(r'$\ell$')
    ax.legend(loc='upper right', fontsize=6.5)
    for ax, tag, (x, ha) in zip(axes, 'abc', [(0.03, 'left'), (0.97, 'right'), (0.03, 'left')]):
        ax.tick_params(top=False, right=False)
        ax.grid(True, color=GRID, lw=0.5)
        ax.text(x, 0.96, f'({tag})', transform=ax.transAxes, ha=ha, va='top', fontsize=8)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'step_spectra')
    plt.close(fig)



LOW, HIGH = PF.LOW, PF.HIGH
WIDTH = HIGH - LOW
B12 = PF.B12
TEX = PF.TEX
ROWS = HERE / 'data' / 'rows_64440'


def tag_axes(axes, positions=None):
    """Panel letters; positions: per panel (x, ha) at the top or (x, ha, y, va)."""
    for k, ax in enumerate(np.ravel(axes)):
        pos = positions[k] if positions else (0.03, 'left')
        x, ha, y, va = (pos + (0.96, 'top'))[:4] if len(pos) == 2 else pos
        ax.tick_params(top=False, right=False)
        ax.grid(True, color=GRID, lw=0.5)
        ax.text(x, y, f'({"abcdefghij"[k]})', transform=ax.transAxes, ha=ha, va=va, fontsize=8)


def fig_design(plt):
    """Step 1: the Sobol design in prior units, the test rows, and the range of signals it spans."""
    theta = np.load(ROWS / 'theta.npy')
    clean40 = np.load(ROWS / 'clean_dl.npy')
    split = np.load(HERE / 'results_final' / 'split.npz')
    test, order = split['test'], split['order']
    b12_40 = np.load(OBS / 'Battaglia12.npz')['clean_dl']
    u = (theta - LOW) / WIDTH
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax = axes[0]
    from matplotlib.lines import Line2D
    ax.scatter(u[order[:4000], 0], u[order[:4000], 2], s=0.6, color=INK2, alpha=0.5, lw=0)
    ax.scatter(u[test, 0], u[test, 2], s=0.8, color=C['red'], alpha=0.7, lw=0)
    ax.plot((B12[0] - LOW[0]) / WIDTH[0], (B12[2] - LOW[2]) / WIDTH[2], marker='*', ms=9, color=INK, mec='white', mew=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel(r'$P_0$ in prior units'); ax.set_ylabel(r'$\beta$ in prior units')
    handles = [Line2D([], [], ls='none', marker='o', ms=2.5, color=INK2), Line2D([], [], ls='none', marker='o', ms=2.5, color=C['red']),
               Line2D([], [], ls='none', marker='*', ms=6, color=INK)]
    ax.legend(handles, ['training rows (4,000 of 61,440)', 'test rows (3,000)', 'B12'], loc='lower center',
              bbox_to_anchor=(0.5, 1.0), fontsize=6, ncol=1, handletextpad=0.2, frameon=False)
    ax = axes[1]
    band = 9                                                      # ell 1880-2080
    ratio = clean40[:, band] / b12_40[band]
    bins = np.linspace(-7, 7, 71)
    ax.hist(np.log10(np.clip(ratio, 1e-12, None)), bins=bins, color=C['blue'], alpha=0.75, lw=0)
    ax.axvline(0, color=INK, lw=0.9, ls='--')
    pct = 100 * np.mean(ratio < 1)
    ax.text(0.04, 0.80, f'B12 is at the\n{pct:.0f}th percentile', transform=ax.transAxes, fontsize=6.5, color=INK)
    ax.set_xlabel(r'$\log_{10}\,D_\ell / D_\ell^{\rm B12}$  ($\ell\approx 1980$)')
    ax.set_ylabel('rows')
    ax = axes[2]
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    rng = np.random.default_rng(3)
    for r in rng.choice(order, 40, replace=False):
        ax.plot(mid, clean40[r] * SCALE, color=INK2, lw=0.5, alpha=0.5)
    ax.plot(mid, b12_40 * SCALE, color=INK, lw=1.6, label='B12')
    ax.set_yscale('log'); ax.set_ylim(1e-6, 1e3); ax.set_xlim(80, 7980)
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'noise-free $D_\ell\;[10^{-12}]$, 40 bands')
    ax.legend(loc='lower left', fontsize=6.5)
    tag_axes(axes, [(0.03, 'left', 0.04, 'bottom'), (0.97, 'right'), (0.97, 'right')])
    fig.tight_layout(w_pad=0.8)
    save(fig, 'step_design')
    plt.close(fig)


def fig_beam(plt):
    """Step 3: the beam and pixel window, and how much the beam-consistent correction shrinks the band noise."""
    bl = np.load(DATA / 'beam_bl.npy')[ELL]
    pw = np.load(HERE / 'data' / 'pixwin_nside4096.npy')[ELL]
    qa = json.loads((HERE / 'beamfix' / 'fetched' / 'results' / 'series.json').read_text())['qa']['1']
    ratio = np.array(qa['band_noise_rms_ratio_beam_over_production'])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    ax = axes[0]
    ax.plot(ELL, bl, color=C['blue'], lw=1.2, label=r"$B_\ell$ (2$'$ beam): multiplies $C^{sn}$")
    ax.plot(ELL, bl ** 2, color=C['blue'], lw=1.2, ls=(0, (3, 1.5)), label=r'$B_\ell^2$: multiplies $C^{nn}$')
    ax.plot(ELL, pw ** 2, color=INK2, lw=1.0, ls=':', label=r'$p_\ell^2$ (NSIDE 4096 pixel window)')
    ax.set_xlim(80, 7980); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\ell$')
    ax.legend(loc='lower left', fontsize=6.5)
    ax = axes[1]
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    ax.plot(mid, ratio, color=C['blue'], marker='o', ms=2.5, lw=1.0)
    ax.axhline(1, color=INK2, lw=0.6, ls=':')
    ax.set_xlim(80, 7980); ax.set_ylim(0, 1.08)
    ax.set_xlabel(r'$\ell$ (band centre)')
    ax.set_ylabel('noise rms ratio\n(beam-consistent / production)')
    allqa = json.loads((HERE / 'beamfix' / 'fetched' / 'results' / 'series.json').read_text())['qa']
    testable = sum(v['noise_slope']['testable_rows'] for v in allqa.values())
    worst = max(np.max(np.abs(np.array(v['noise_slope'].get('testable_median_abs_dev', 0)))) for v in allqa.values())
    sl = qa['noise_slope']
    ax.text(0.04, 0.08, f"noise-pair check, all stages: {testable:,} rows,\nmedian |slope $-$ 1| $\\leq$ {worst:.4f}; "
            f"wrong pair:\n{sl['wrong_pair_slope_mean']:+.4f} $\\pm$ {sl['wrong_pair_slope_sd']:.3f}",
            transform=ax.transAxes, fontsize=6.3, color=INK, va='bottom')
    tag_axes(axes, [(0.97, 'right'), (0.97, 'right')])
    fig.tight_layout(w_pad=1.2)
    save(fig, 'step_beam')
    plt.close(fig)


def fig_bins40(plt):
    """Step 4: the 40-band powers span many decades; the signed asinh at the noise scale brings them to O(1)."""
    noisy40 = np.load(ROWS / 'noisy_dl.npy')
    order = np.load(HERE / 'results_final' / 'split.npz')['order']
    nm = json.loads((HERE / 'results_final_noise' / 'noise_model.json').read_text())
    scale = np.array(nm['N_band']) / np.sqrt(np.array(nm['nu_band']))
    pool = noisy40[order]
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(width_ratios=[1.15, 0.85, 1]))
    ax = axes[0]
    q = np.quantile(pool, [0.01, 0.16, 0.5, 0.84, 0.99], axis=0)
    ax.fill_between(mid, np.clip(q[0], 1e-30, None), q[4], color=C['bins40'], alpha=0.25, lw=0, label='1-99 % of pool')
    ax.fill_between(mid, np.clip(q[1], 1e-30, None), q[3], color=C['bins40'], alpha=0.55, lw=0, label='16-84 %')
    ax.plot(mid, q[2], color=C['bins40'], lw=1.2, label='median')
    ax.plot(mid, scale, color=INK, lw=1.0, ls=(0, (3, 1.5)), label=r'noise scale $N_b/\sqrt{\nu_b}$')
    ax.set_yscale('log'); ax.set_xlim(80, 7980); ax.set_ylim(1e-20, 1e-7)
    ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'band power $D_b$ (signed cross)')
    ax.legend(loc='lower left', fontsize=6, ncol=1)
    ax = axes[1]
    y = np.linspace(-30, 30, 400)
    ax.plot(y, np.arcsinh(y), color=C['bins40'], lw=1.4, label=r'asinh$(D/a)$')
    ax.plot(y, y, color=INK2, lw=0.8, ls=':', label='linear')
    big = np.where(np.abs(y) > 1.5, y, np.nan)
    ax.plot(big, np.sign(big) * np.log(2 * np.abs(big)), color=INK2, lw=0.8, ls=(0, (3, 1.5)), label=r'sign$(y)\,\ln 2|y|$')
    ax.set_xlim(-30, 30); ax.set_ylim(-5, 5)
    ax.set_xlabel(r'$y = D_b / a_b$'); ax.legend(loc='lower right', fontsize=6)
    ax = axes[2]
    b = 9
    z_lin = (pool[:, b] - pool[:, b].mean()) / pool[:, b].std()
    ya = np.arcsinh(pool[:, b] / scale[b]); z_as = (ya - ya.mean()) / ya.std()
    bins = np.linspace(-4, 4, 81)
    ax.hist(np.clip(z_lin, -4, 4), bins=bins, color=INK2, alpha=0.6, lw=0, label='z-score of $D_b$')
    ax.hist(z_as, bins=bins, color=C['bins40'], alpha=0.7, lw=0, label='z-score of asinh')
    ax.set_yscale('log'); ax.set_xlabel(f'network input, band {b + 1} ($\\ell$ {EDGES[b]}-{EDGES[b + 1]})')
    ax.legend(loc='upper left', fontsize=6)
    tag_axes(axes, [(0.97, 'right'), (0.03, 'left'), (0.97, 'right')])
    fig.tight_layout(w_pad=0.8)
    save(fig, 'step_bins40')
    plt.close(fig)


def fig_logit(plt):
    """Step 9: the logit map of the prior box, its Jacobian, and one B12 marginal in u and theta, bounded/unbounded."""
    i = 3                                                          # alpha_m^P0
    bounded = np.load(HERE / 'posterior_rows_whitening' / 'beam_nn_bounded_merlin_b12.npy')[:, i]
    unbounded = np.load(HERE / 'posterior_rows_whitening' / 'beam_nn_unbounded_merlin_b12.npy')[:, i]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    ax = axes[0]
    t = np.linspace(1e-4, 1 - 1e-4, 400)
    ax.plot(t, np.log(t / (1 - t)), color=C['merlin'], lw=1.3, label=r'$u=\mathrm{logit}\,t$, $t=(\theta-\theta_{\min})/w$')
    ax.plot(t, np.log(t * (1 - t)), color=INK2, lw=1.0, ls=(0, (3, 1.5)), label=r'$\log|d t/du| = \log t(1-t)$')
    ax.set_xlim(0, 1); ax.set_ylim(-9, 9)
    ax.set_xlabel(r'$t$ (position in the prior range)'); ax.legend(loc='upper left', fontsize=6)
    ax = axes[1]
    ub = np.log((bounded - LOW[i]) / (HIGH[i] - bounded))
    ax.hist(ub, bins=80, range=(-8, 8), density=True, color=C['merlin'], alpha=0.6, lw=0, label='bounded posterior in $u$')
    uu = np.linspace(-8, 8, 400)
    ax.plot(uu, np.exp(uu) / (1 + np.exp(uu)) ** 2, color=INK2, lw=1.0, label='flat prior in $u$ (logistic)')
    ax.axvline(np.log((B12[i] - LOW[i]) / (HIGH[i] - B12[i])), color=INK, ls='--', lw=0.8)
    ax.set_ylim(0, 0.42); ax.set_xlabel(r'$u$ for $\alpha_m^{P_0}$'); ax.set_yticks([]); ax.legend(loc='upper right', fontsize=6)
    ax = axes[2]
    lo, hi = LOW[i] - 0.4 * WIDTH[i], HIGH[i] + 0.4 * WIDTH[i]
    ax.hist(bounded, bins=90, range=(lo, hi), density=True, color=C['merlin'], alpha=0.55, lw=0, label='bounded (logit $u$)')
    ax.hist(unbounded, bins=90, range=(lo, hi), density=True, histtype='step', color=C['red'], lw=1.2, label='unbounded (linear $u$)')
    for bnd in (LOW[i], HIGH[i]):
        ax.axvline(bnd, color=INK2, lw=0.8, ls=':')
    ax.axvline(B12[i], color=INK, ls='--', lw=0.8)
    inside = np.mean((unbounded >= LOW[i]) & (unbounded <= HIGH[i]))
    ax.set_xlabel(r'$\alpha_m^{P_0}$'); ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=6, title=f'unbounded: {100 * inside:.0f} % inside', title_fontsize=6)
    tag_axes(axes, [(0.97, 'right', 0.04, 'bottom'), (0.03, 'left'), (0.03, 'left')])
    fig.tight_layout(w_pad=0.8)
    save(fig, 'step_logit')
    plt.close(fig)


def fig_training(plt):
    """Step 10: validation loss per epoch of the three seeds (Merlin, PCA; beam-consistent, noise-only whitening)."""
    folder = HERE / 'results_merlin_unbinned_beam_nnwhite_asinh_p30'
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2), sharey=False)
    for ax, (variant, name) in zip(axes, (('merlin', 'Merlin'), ('pca', 'PCA'))):
        for k, seed in enumerate((1, 2, 3)):
            h = json.loads((folder / f'{variant}_seed{seed}.json').read_text())['history']
            ep = [e['epoch'] for e in h]; vl = [e['val_loss'] for e in h]
            ax.plot(ep, vl, color=C[variant], lw=1.0, alpha=[1, 0.7, 0.45][k], label=f'seed {seed}')
            best = int(np.argmin(vl))
            ax.plot(ep[best], vl[best], marker='o', ms=3, color=C[variant])
            drops = [e['epoch'] for e0, e in zip(h[:-1], h[1:]) if e['lr'] < e0['lr']]
            for d in drops:
                ax.axvline(d, color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel('epoch'); ax.set_ylabel('validation loss (mean $-\\log q$ in $u$)')
        ax.set_title(name, fontsize=8.5)
        ax.legend(loc='upper right', fontsize=6.5)
    tag_axes(axes, [(0.03, 'left', 0.04, 'bottom'), (0.03, 'left', 0.04, 'bottom')])
    fig.tight_layout(w_pad=1.2)
    save(fig, 'step_training')
    plt.close(fig)


def fig_ensemble(plt):
    """Step 10: the ensemble mixture beats every single seed in log p; per-seed vs ensemble Pearson r."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    ax = axes[0]
    x = np.linspace(-4, 4, 400)
    g = lambda m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    seeds = [g(-0.6, 0.55), g(0.2, 0.5), g(0.7, 0.6)]
    for k, q in enumerate(seeds):
        ax.plot(x, q, color=C['merlin'], lw=0.9, alpha=[1, 0.7, 0.45][k], label='seeds' if k == 0 else None)
    ax.plot(x, np.mean(seeds, axis=0), color=INK, lw=1.5, label='equal-weight mixture')
    ax.axvline(1.3, color=C['red'], ls='--', lw=0.8, label='a truth in a tail')
    ax.set_ylim(0, 1.45 * max(q.max() for q in seeds))
    ax.set_yticks([]); ax.set_xlabel(r'$u$'); ax.legend(loc='upper left', fontsize=6)
    ax = axes[1]
    labels = []
    for k, (folder, name) in enumerate((('results_merlin_unbinned_beam_nnwhite_asinh_p30', 'noise-only'),
                                        ('results_merlin_unbinned_beam_asinh_p30', 'B12'))):
        for j, variant in enumerate(('merlin', 'pca')):
            m = json.loads((HERE / folder / f'metrics_{variant}.json').read_text())
            xpos = 2 * k + j
            ax.plot([xpos] * 3, m['test_log_prob_seeds'], ls='none', marker='o', ms=3, mfc='white', color=C[variant])
            ax.plot(xpos, m['mean_log_prob_theta'], marker='D', ms=4.5, color=C[variant])
            labels.append(f"{'Merlin' if variant == 'merlin' else 'PCA'}\n{name}")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel(r'test $\langle\log p(\theta_{\rm true})\rangle$')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], ls='none', marker='o', mfc='white', color=INK2, ms=3), Line2D([], [], ls='none', marker='D', color=INK2, ms=4)],
              ['single seed', 'ensemble'], loc='lower left', fontsize=6)
    ax = axes[2]
    d = np.load(HERE / 'posterior_rows_whitening' / 'beam_nn_bounded_merlin.npz')
    truth = d['truth'].astype(float)
    r_ens = PFpearson(truth, d['mean'].astype(float))
    r_seed = PFpearson(truth[None], d['seed_mean'].astype(float))
    xx = np.arange(9)
    for k in range(3):
        ax.plot(xx + (k - 1) * 0.12, r_seed[k], ls='none', marker='o', ms=2.5, mfc='white', color=C['merlin'])
    ax.plot(xx, r_ens, ls='none', marker='D', ms=3.5, color=C['merlin'])
    ax.set_xticks(xx); ax.set_xticklabels(TEX, fontsize=6.5)
    ax.set_ylabel('Pearson $r$ (Merlin, noise-only)')
    tag_axes(axes, [(0.97, 'right'), (0.97, 'right'), (0.03, 'left')])
    fig.tight_layout(w_pad=0.8)
    save(fig, 'step_ensemble')
    plt.close(fig)


def PFpearson(truth, mean):
    t = truth - truth.mean(-2, keepdims=True)
    m = mean - mean.mean(-2, keepdims=True)
    return (t * m).sum(-2) / np.sqrt((t * t).sum(-2) * (m * m).sum(-2))



METRIC_DATA = HERE / 'doc_metric_data'
FLAT68 = 0.6827
NAMES = {'bins40': '40 bins', 'pca': 'PCA', 'merlin': 'Merlin'}


def method_rows(noise='beam'):
    """{method: per-row dict (truth, mean, q16, q84, q025?, q975?, rank?, seed_mean?, lp?)} for the comparison.

    beam: Merlin and PCA with noise-only whitening (doc_metric_data or posterior_rows_whitening) and the staged
    40-band network (doc_metric_data/beam_bins40); production: the production-noise networks."""
    src = {}
    if noise == 'beam':
        src['bins40'] = [METRIC_DATA / 'beam_bins40.npz']
        src['pca'] = [METRIC_DATA / 'beam_nn_pca.npz', HERE / 'posterior_rows_whitening' / 'beam_nn_bounded_pca.npz']
        src['merlin'] = [METRIC_DATA / 'beam_nn_merlin.npz', HERE / 'posterior_rows_whitening' / 'beam_nn_bounded_merlin.npz']
    else:
        src['bins40'] = [METRIC_DATA / 'prod_bins40.npz', HERE / 'posterior_rows' / 'noise_bins40_noisy.npz']
        src['pca'] = [METRIC_DATA / 'prod_pca.npz', HERE / 'posterior_rows_whitening' / 'prod_b12_bounded_pca.npz']
        src['merlin'] = [METRIC_DATA / 'prod_merlin.npz', HERE / 'posterior_rows_whitening' / 'prod_b12_bounded_merlin.npz']
    out = {}
    for m, paths in src.items():
        for p in paths:
            if p.exists():
                d = dict(np.load(p))
                d = {k: (v.astype(np.float64) if v.dtype.kind == 'f' else v) for k, v in d.items()}
                d['source'] = p.name
                out[m] = d
                break
    return out


def fig_metric_r(plt, noise='beam'):
    """Pearson r and RMSE: truth against posterior mean for the best and worst parameter, three compressions."""
    rows = method_rows(noise)
    params = [6, 5]                                       # alpha_z^P0 (best constrained), alpha_m^beta (worst)
    fig, axes = plt.subplots(2, len(rows), figsize=(7.2, 4.6), sharex='row', sharey='row')
    for j, (m, d) in enumerate(rows.items()):
        for i, k in enumerate(params):
            ax = axes[i, j]
            t, mu = d['truth'][:, k], d['mean'][:, k]
            ax.hexbin(t, mu, gridsize=34, extent=(LOW[k], HIGH[k], LOW[k], HIGH[k]), cmap='Greys', mincnt=1,
                      linewidths=0, bins='log')
            ax.plot([LOW[k], HIGH[k]], [LOW[k], HIGH[k]], color=C[m], lw=1.0)
            ax.axhline(0.5 * (LOW[k] + HIGH[k]), color=INK2, lw=0.7, ls=':')
            r = np.corrcoef(t, mu)[0, 1]
            rmse = np.sqrt(np.mean((mu - t) ** 2)) / np.sqrt(np.mean((0.5 * (LOW[k] + HIGH[k]) - t) ** 2))
            ax.text(0.04, 0.96, f'$r$ = {r:.3f}\nRMSE/prior = {rmse:.2f}', transform=ax.transAxes, va='top',
                    fontsize=6.5, color=INK, bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.5))
            ax.set_xlim(LOW[k], HIGH[k]); ax.set_ylim(LOW[k], HIGH[k])
            if i == 0:
                ax.set_title(NAMES[m], fontsize=8.5, color=C[m])
            if j == 0:
                ax.set_ylabel(f'posterior mean of {TEX[k]}')
            ax.set_xlabel(f'true {TEX[k]}')
            ax.tick_params(top=False, right=False)
    fig.tight_layout(w_pad=0.5, h_pad=0.8)
    save(fig, f'metric_r_{noise}')
    plt.close(fig)


def fig_metric_width(plt, noise='beam'):
    """68 % width over the flat-posterior width: per-row distributions and medians per parameter."""
    rows = method_rows(noise)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    bins = np.linspace(0, 1.25, 51)
    for m, d in rows.items():
        w = ((d['q84'] - d['q16']) / (FLAT68 * WIDTH)).mean(axis=1)          # row average over the nine
        ax.hist(w, bins=bins, histtype='step', color=C[m], lw=1.3, label=f'{NAMES[m]} (median {np.median(w):.2f})')
    ax.axvline(1, color=INK2, lw=0.8, ls=':')
    ax.set_xlabel('68 % width / flat width, mean over the 9'); ax.set_ylabel('test rows')
    ax.legend(loc='upper left', fontsize=6.3)
    ax = axes[1]
    x = np.arange(9)
    for j, (m, d) in enumerate(rows.items()):
        med = np.median((d['q84'] - d['q16']) / (FLAT68 * WIDTH), axis=0)
        ax.bar(x + (j - 1) * 0.27, med, width=0.25, color=C[m], label=NAMES[m])
    ax.axhline(1, color=INK2, lw=0.8, ls=':')
    ax.set_xticks(x); ax.set_xticklabels(TEX, fontsize=7)
    ax.set_ylabel('median width / flat'); ax.set_ylim(0, 1.12)
    ax.legend(loc='upper center', fontsize=6.3, ncol=3)
    tag_axes(axes, [(0.97, 'right'), (0.02, 'left', 0.04, 'bottom')])
    fig.tight_layout(w_pad=1.0)
    save(fig, f'metric_width_{noise}')
    plt.close(fig)


def fig_metric_coverage(plt, noise='beam'):
    """Coverage: 68 % intervals of 60 test rows against the truth; empirical 68 / 95 % coverage per parameter."""
    rows = method_rows(noise)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axes[0]
    d, k = rows['merlin'], 0
    idx = np.argsort(d['truth'][:60, k])
    for n, r in enumerate(idx):
        lo, hi, t = d['q16'][r, k], d['q84'][r, k], d['truth'][r, k]
        inside = lo <= t <= hi
        ax.plot([n, n], [lo, hi], color=C['merlin'] if inside else C['red'], lw=1.4, alpha=0.85)
        ax.plot(n, t, marker='_', color=INK, ms=4, mew=1.0)
    share = np.mean((d['truth'][:60, k] >= d['q16'][:60, k]) & (d['truth'][:60, k] <= d['q84'][:60, k]))
    ax.set_xlabel('60 test rows, sorted by the true $P_0$'); ax.set_ylabel(r'$P_0$: 68 % interval and truth')
    ax.text(0.04, 0.96, f'Merlin: {share:.0%} contain the truth', transform=ax.transAxes, va='top', fontsize=6.5)
    ax.set_xticks([])
    ax = axes[1]
    x = np.arange(9)
    for j, (m, d) in enumerate(rows.items()):
        c68 = np.mean((d['truth'] >= d['q16']) & (d['truth'] <= d['q84']), axis=0)
        ax.plot(x + (j - 1) * 0.2, c68, ls='none', marker='o', ms=4, color=C[m], label=f'{NAMES[m]}, 68 %')
        if 'q025' in d:
            c95 = np.mean((d['truth'] >= d['q025']) & (d['truth'] <= d['q975']), axis=0)
            ax.plot(x + (j - 1) * 0.2, c95, ls='none', marker='s', ms=3.5, mfc='white', color=C[m])
    ax.axhline(0.68, color=INK2, lw=0.8, ls=':'); ax.axhline(0.95, color=INK2, lw=0.8, ls=':')
    band = 1.96 * np.sqrt(0.68 * 0.32 / 3000)
    ax.axhspan(0.68 - band, 0.68 + band, color=GRID, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(TEX, fontsize=7); ax.set_ylim(0.58, 1.0)
    ax.set_ylabel('share of truths inside')
    ax.legend(loc='center right', fontsize=6, ncol=1, title='squares: 95 %', title_fontsize=6)
    tag_axes(axes, [(0.97, 'right', 0.04, 'bottom'), (0.02, 'left', 0.04, 'bottom')])
    fig.tight_layout(w_pad=1.0)
    save(fig, f'metric_coverage_{noise}')
    plt.close(fig)


def fig_metric_sbc(plt, noise='beam'):
    """SBC: rank histograms for one parameter and ECDF minus uniform for all nine, per compression."""
    rows = {m: d for m, d in method_rows(noise).items() if 'rank' in d}
    if not rows:
        return
    n_draws = 999
    fig, axes = plt.subplots(1, 1 + len(rows), figsize=(7.2, 2.35), sharey=False)
    ax = axes[0]
    k = 2                                                       # beta
    edges = np.linspace(0, n_draws + 1, 21)
    for m, d in rows.items():
        h, _ = np.histogram(d['rank'][:, k], bins=edges)
        ax.step(edges, np.r_[h, h[-1]], where='post', color=C[m], lw=1.2, label=NAMES[m])
    expect = len(next(iter(rows.values()))['rank']) / 20
    ax.axhline(expect, color=INK2, lw=0.8, ls=':')
    ax.set_xlabel(r'rank of the true $\beta$ among 999 draws'); ax.set_ylabel('test rows per bin')
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.6 * (hi - lo))
    ax.legend(loc='upper right', fontsize=5.5, ncol=1, handlelength=1.2)
    band = 1.358 / np.sqrt(3000)
    grid = np.linspace(0, 1, 201)
    for ax, (m, d) in zip(axes[1:], rows.items()):
        frac = d['rank'] / n_draws
        for i in range(9):
            ecdf = np.searchsorted(np.sort(frac[:, i]), grid, side='right') / len(frac)
            ax.plot(grid, ecdf - grid, color=C[m], lw=0.8, alpha=0.75)
        ax.axhspan(-band, band, color=GRID, zorder=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_ylim(-0.06, 0.06); ax.set_xlabel('rank / draws')
        ax.set_title(NAMES[m], fontsize=8, color=C[m])
    axes[1].set_ylabel('ECDF $-$ uniform')
    for a in axes:
        a.tick_params(top=False, right=False); a.grid(True, color=GRID, lw=0.5)
    fig.tight_layout(w_pad=0.6)
    save(fig, f'metric_sbc_{noise}')
    plt.close(fig)



SERIES = HERE / 'beamfix' / 'fetched' / 'results' / 'series.json'


def mjson(path):
    return json.loads(Path(path).read_text())


def size_series(noise='beam'):
    """{method: {N: metrics dict}} along the training-set size."""
    out = {'bins40': {}, 'pca': {}, 'merlin': {}}
    if noise == 'beam':
        series = mjson(SERIES)['series_noise']['bins40_beam']
        out['bins40'] = {int(N): m for N, m in series.items()}
        full = METRIC_DATA / 'beam_bins40_N61440_metrics.json'     # the final stage, fetched from idark
        if full.exists() and 61440 not in out['bins40']:
            out['bins40'][61440] = mjson(full)
        base = 'results_merlin_unbinned_beam_asinh_p30'
    else:
        for N in (2000, 4000, 8000, 16000, 32000, 61440):
            p = HERE / 'results_final_noise' / 'bins40_noisy' / f'N{N}' / 'metrics.json'
            if p.exists():
                out['bins40'][N] = mjson(p)
        base = 'results_merlin_unbinned_asinh_p30'
    for N in (2000, 4000, 8000, 16000, 32000, 61440):
        folder = HERE / (base if N == 61440 else f'{base}_N{N}')
        for v in ('pca', 'merlin'):
            p = folder / f'metrics_{v}.json'
            if p.exists():
                out[v][N] = mjson(p)
    return out


def interval_score(d, alpha=0.32, lo='q16', hi='q84'):
    """Mean interval score of the central (1 - alpha) intervals over the prior width (lower is better)."""
    t, l, u = d['truth'], d[lo], d[hi]
    score = (u - l) + 2 / alpha * (l - t) * (t < l) + 2 / alpha * (t - u) * (t > u)
    return float(np.mean(score / WIDTH))


def fig_compare_sizes(plt, noise='beam'):
    """log p, r, RMSE/prior and width/flat against the number of training simulations, three compressions."""
    from matplotlib.ticker import FixedLocator, NullLocator
    series = size_series(noise)
    keys = [('mean_log_prob_theta', r'test $\langle\log p\rangle$'), ('pearson_r', 'mean Pearson $r$'),
            ('rmse_over_prior_rmse', 'mean RMSE / prior'), ('median_width68_over_flat', 'mean width / flat')]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for ax, (key, label) in zip(axes, keys):
        for m in ('bins40', 'pca', 'merlin'):
            byN = series[m]
            if not byN:
                continue
            N = sorted(byN)
            y = [np.mean(byN[n][key]) for n in N]
            ax.plot(N, y, color=C[m], lw=1.2, marker='o', ms=2.5, label=NAMES[m])
        if noise == 'beam' and key != 'mean_log_prob_theta' or noise == 'beam':
            for v in ('pca', 'merlin'):
                m = mjson(HERE / 'results_merlin_unbinned_beam_nnwhite_asinh_p30' / f'metrics_{v}.json')
                ax.plot(61440, np.mean(m[key]), marker='s', ms=4, mfc='white', color=C[v], ls='none')
        ax.set_xscale('log')
        ticks = [2000, 8000, 32000, 61440]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(['2k', '8k', '32k', '61k'])
        ax.set_xlabel('training simulations'); ax.set_title(label, fontsize=7.5)
        ax.tick_params(top=False, right=False, labelsize=6.5); ax.grid(True, color=GRID, lw=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C[m], lw=1.2, marker='o', ms=2.5) for m in ('bins40', 'pca', 'merlin')]
    labels = [NAMES[m] for m in ('bins40', 'pca', 'merlin')]
    if noise == 'beam':
        handles.append(Line2D([], [], ls='none', marker='s', ms=4, mfc='white', color=INK2))
        labels.append('noise-only whitening (61k)')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=len(labels), fontsize=6.5)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.95))
    save(fig, f'compare_sizes_{noise}')
    plt.close(fig)


def fig_compare_scorecard(plt, noise='beam'):
    """All metrics at the full training set, one small panel each (arrows: the better direction)."""
    rows = method_rows(noise)
    series = size_series(noise)
    full = {m: series[m][max(series[m])] for m in series if series[m]}
    if noise == 'beam':                                    # Merlin and PCA with noise-only whitening
        for v in ('pca', 'merlin'):
            full[v] = mjson(HERE / 'results_merlin_unbinned_beam_nnwhite_asinh_p30' / f'metrics_{v}.json')
    panels = [('test log p', lambda m, d: np.mean(full[m]['mean_log_prob_theta']), 'up'),
              ('Pearson r', lambda m, d: np.mean(full[m]['pearson_r']), 'up'),
              ('RMSE / prior', lambda m, d: np.mean(full[m]['rmse_over_prior_rmse']), 'down'),
              ('width / flat', lambda m, d: np.mean(full[m]['median_width68_over_flat']), 'down'),
              ('interval score 68 %', lambda m, d: interval_score(d) if d is not None else np.nan, 'down'),
              ('TARP 68 %', lambda m, d: full[m]['tarp68'], 'target 0.68')]
    methods = [m for m in ('bins40', 'pca', 'merlin') if m in full]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 1.9))
    for ax, (title, func, better) in zip(axes, panels):
        vals = [func(m, rows.get(m)) for m in methods]
        good = [v for v in vals if np.isfinite(v)]
        for j, (m, v) in enumerate(zip(methods, vals)):
            if not np.isfinite(v):
                ax.text(np.mean(good), j, 'pending', ha='center', va='center', fontsize=6, color=INK2, style='italic')
                continue
            ax.plot(v, j, marker='o', ms=5, color=C[m])
            ax.text(v, j + 0.28, f'{v:.3f}' if abs(v) < 10 else f'{v:.2f}', ha='center', fontsize=6, color=INK)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([NAMES[m] for m in methods] if ax is axes[0] else [])
        ax.set_ylim(-0.6, len(methods) - 0.2)
        good = [v for v in vals if np.isfinite(v)]
        pad = 0.25 * (max(good) - min(good) + 1e-3)
        ax.set_xlim(min(good) - pad, max(good) + pad)
        if better == 'target 0.68':
            ax.axvline(0.68, color=INK2, lw=0.7, ls=':')
        ax.set_title(f"{title}\n{'higher is better' if better == 'up' else 'lower is better' if better == 'down' else better}", fontsize=6.8)
        ax.tick_params(top=False, right=False, labelsize=6); ax.grid(True, axis='x', color=GRID, lw=0.5)
    fig.tight_layout(w_pad=0.4)
    save(fig, f'compare_scorecard_{noise}')
    plt.close(fig)


def fig_compare_params(plt, noise='beam'):
    """Per-parameter Pearson r of the three compressions at the full training set."""
    rows = method_rows(noise)
    r_of = {m: PFpearson(d['truth'], d['mean']) for m, d in rows.items()}
    label_of = {m: NAMES[m] for m in rows}
    if 'bins40' not in r_of and noise == 'beam':           # the staged series' stored r until the per-row data exists
        series = size_series('beam')['bins40']
        n = max(series)
        r_of = {'bins40': np.array(series[n]['pearson_r']), **r_of}
        label_of['bins40'] = f'40 bins (N = {n:,})'
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    x = np.arange(9)
    for j, m in enumerate(r_of):
        ax.bar(x + (j - (len(r_of) - 1) / 2) * 0.27, r_of[m], width=0.25, color=C[m], label=label_of[m])
    ax.set_xticks(x); ax.set_xticklabels(TEX)
    ax.set_ylabel('Pearson $r$'); ax.set_ylim(0, 1)
    ax.legend(loc='upper center', ncol=3, fontsize=6.5)
    ax.tick_params(top=False, right=False); ax.grid(True, axis='y', color=GRID, lw=0.5)
    fig.tight_layout()
    save(fig, f'compare_params_{noise}')
    plt.close(fig)



def matplotlib_null():
    from matplotlib.ticker import NullLocator
    return NullLocator()


WHITE_NAMES = dict(beam_b12=('B12 whitening (20 draws)', C['merlin']), beam_nn=('noise-only (61,440 rows)', C['red']))


def fig_whitening(plt):
    """Step 5: the two whitening covariances and what they do to the B12 spectrum."""
    d = {k: dict(np.load(DATA / f'{k}.npz')) for k in ('beam_b12', 'beam_nn')}
    fig = plt.figure(figsize=(7.2, 4.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.45, wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    for k, (name, col) in WHITE_NAMES.items():
        ax.plot(ELL, d[k]['sd'] * SCALE, color=col, lw=0.8, label=name)
    ax.set_yscale('log'); ax.set_xlim(80, 7980); ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r'noise sd $[10^{-12}]$')
    ax.legend(loc='lower left', fontsize=5.8)
    axes = [ax]
    ax = fig.add_subplot(gs[0, 1])
    lags = np.arange(len(d['beam_nn']['corr_lags']))
    for k, (name, col) in WHITE_NAMES.items():
        ax.plot(lags[1:], d[k]['corr_lags'][1:], color=col, marker='o', ms=2.2, lw=0.9)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel(r'lag $k$ (multipoles)'); ax.set_ylabel(r'mean corr($\ell$, $\ell+k$)')
    axes.append(ax)
    ax = fig.add_subplot(gs[0, 2])
    for k, (name, col) in WHITE_NAMES.items():
        r = d[k]['white_noisy_b12'] - d[k]['white_clean_b12']
        ax.hist(r, bins=np.linspace(-5, 5, 81), histtype='step', color=col, lw=1.1, density=True)
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color=INK2, lw=0.8, ls=':', label='unit Gaussian')
    ax.set_xlabel('whitened noise, B12 mock'); ax.set_yticks([])
    ax.set_ylim(0, 0.62)
    ax.legend(loc='upper left', fontsize=5.6)
    axes.append(ax)
    for j, k in enumerate(('beam_b12', 'beam_nn')):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(d[k]['corr_block'][:40, :40], cmap='RdBu_r', vmin=-1, vmax=1, extent=(2000, 2040, 2040, 2000))
        ax.set_title(WHITE_NAMES[k][0], fontsize=7, color=WHITE_NAMES[k][1])
        ax.set_xlabel(r'$\ell$'); ax.set_ylabel(r"$\ell'$")
        ax.set_xticks([2000, 2020, 2040]); ax.set_yticks([2000, 2020, 2040])
        ax.tick_params(labelsize=6)
        axes.append(ax)
    cax = fig.add_subplot(gs[1, 2])
    cax.set_axis_off()
    cb = fig.colorbar(im, ax=cax, fraction=0.6, aspect=12, location='left')
    cb.set_label('noise correlation', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    for k, ax in enumerate(axes):
        ax.tick_params(top=False, right=False)
        if k < 3:
            ax.grid(True, color=GRID, lw=0.5)
        ax.text(0.97 if k in (1, 2) else 0.03, 0.96, f'({"abcde"[k]})', transform=ax.transAxes,
                ha='right' if k in (1, 2) else 'left', va='top', fontsize=8,
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=1) if k >= 3 else None)
    save(fig, 'step_whitening')
    plt.close(fig)


def fig_pca(plt):
    """Step 6: the PCA spectrum with the 0.001 % cut and the first components."""
    d = {k: dict(np.load(DATA / f'{k}.npz')) for k in ('beam_b12', 'beam_nn')}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    for k, (name, col) in WHITE_NAMES.items():
        sv = d[k]['singular_values']
        share = 100 * sv / sv.sum()
        n = int(d[k]['kept'].sum())
        ax.plot(np.arange(1, 151), share[:150], color=col, lw=1.1, label=f'{name}: {n} kept')
    ax.axhline(1e-3, color=INK, lw=0.8, ls='--')
    ax.text(150, 1.4e-3, '0.001 % cut', ha='right', fontsize=6.5)
    ax.set_yscale('log'); ax.set_xlabel('component $k$'); ax.set_ylabel(r'$100\,S_k/\sum S$  [%]')
    ax.legend(loc='upper right', fontsize=6)
    ax = axes[1]
    V = d['beam_nn']['V6']
    for j in range(6):
        v = V[:, j] / np.max(np.abs(V[:, j]))
        ax.plot(ELL, v - 2.2 * j, color=C['red'] if j % 2 == 0 else INK2, lw=0.7)
        ax.text(8100, -2.2 * j, f'$V_{j + 1}$', fontsize=7, va='center')
    ax.set_xlim(80, 7980); ax.set_ylim(-2.2 * 5 - 1.3, 2.4); ax.set_yticks([]); ax.set_xlabel(r'$\ell$')
    ax.set_title('first six components (noise-only whitening)', fontsize=7.5)
    tag_axes(axes, [(0.03, 'left', 0.04, 'bottom'), (0.02, 'left', 0.98, 'top')])
    axes[1].grid(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'step_pca')
    plt.close(fig)


def fig_asinh(plt):
    """Step 7: linear z-scores against the asinh standardisation for three PCA components; the transform."""
    d = dict(np.load(DATA / 'beam_nn.npz'))
    comps = [0, 2, 5]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    bins = np.linspace(-4, 4, 81)
    for ax, j in zip(axes[:3], comps):
        sn = d['s_noisy'][:, j]
        zl = (sn - sn.mean()) / sn.std()
        ax.hist(np.clip(zl, -4, 4), bins=bins, color=INK2, alpha=0.55, lw=0, label='linear z-score')
        ax.hist(d['z_noisy'][:, j], bins=bins, color=C['merlin'], alpha=0.7, lw=0, label='asinh standardised')
        ax.set_yscale('log'); ax.set_xlabel(f'input, component {j + 1}')
        if j == 0:
            ax.set_ylabel('training rows')
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi * 30)
            ax.legend(loc='upper right', fontsize=5.6)
    ax = axes[3]
    j = 0
    y = (d['s_noisy'][:, j] - d['s_center'][j]) / d['s_scale'][j]
    yy = np.linspace(-3e3, 2e9, 400000)
    ax.plot(yy, (np.arcsinh(yy) - d['z_mean'][j]) / d['z_sd'][j], color=INK2, lw=0.8, zorder=1)
    ax.scatter(y, d['z_noisy'][:, j], s=2, color=C['merlin'], alpha=0.5, lw=0, zorder=2)
    ax.set_xscale('symlog', linthresh=1); ax.set_xlim(-3e3, 2e9)
    ax.set_xticks([-1e3, 0, 1e3, 1e6, 1e9]); ax.set_xticklabels(['$-10^3$', '0', '$10^3$', '$10^6$', '$10^9$'], fontsize=6)
    ax.xaxis.set_minor_locator(matplotlib_null())
    ax.set_xlabel('$(s-m)/c$, component 1'); ax.set_ylabel('network input')
    ax.axvspan(-1, 1, color=GRID, zorder=0)
    ax.text(0.05, 0.92, 'grey: within one\nnoise unit (linear)', transform=ax.transAxes, fontsize=6, va='top')
    tag_axes(axes, [(0.03, 'left'), (0.03, 'left'), (0.03, 'left'), (0.97, 'right', 0.04, 'bottom')])
    fig.tight_layout(w_pad=0.6)
    save(fig, 'step_asinh')
    plt.close(fig)



def signal_strength():
    """log10 of each test row's noise-free band-10 power (ell 1880-2080) over the B12 value, in test-row order."""
    split = np.load(HERE / 'results_final' / 'split.npz')
    clean40 = np.load(ROWS / 'clean_dl.npy')
    b12 = np.load(OBS / 'Battaglia12.npz')['clean_dl']
    return np.log10(np.clip(clean40[split['test'], 9] / b12[9], 1e-12, None))


def fig_metric_logp(plt, noise='beam'):
    """Test log p: per-row distributions (doc_metric_data lp) and the mean against training-set size."""
    rows = {m: d for m, d in method_rows(noise).items() if 'lp' in d}
    if not rows:
        return
    from matplotlib.ticker import FixedLocator, NullLocator
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    bins = np.linspace(-25, 25, 101)
    for m, d in rows.items():
        ax.hist(np.clip(d['lp'], -25, 25), bins=bins, histtype='step', color=C[m], lw=1.2,
                label=f'{NAMES[m]}: {d["lp"].mean():.2f} / {np.median(d["lp"]):.2f}')
        ax.axvline(d['lp'].mean(), color=C[m], lw=0.8, ls=(0, (3, 1.5)))
    ax.axvline(-12.90, color=INK2, lw=0.8, ls=':')
    ax.set_xlim(-22, 8)
    ax.text(-13.2, ax.get_ylim()[1] * 0.55, 'flat prior', fontsize=6, color=INK2, ha='right')
    ax.set_xlabel(r'$\log p(\theta_{\rm true}\mid x)$ per test row [nats]'); ax.set_ylabel('test rows')
    ax.legend(loc='upper left', fontsize=5.6, title='mean / median', title_fontsize=5.6)
    ax = axes[1]
    strength = signal_strength()
    edges = np.array([-9, -2, -1, 0, 1, 2, 3, 9])
    centres = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    counts = np.histogram(strength, bins=edges)[0]
    for m, d in rows.items():
        means = [d['lp'][(strength >= a) & (strength < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(centres, means, color=C[m], lw=1.2, marker='o', ms=3, label=NAMES[m])
    ax.axhline(-12.90, color=INK2, lw=0.8, ls=':')
    ax.text(-2.4, -12.5, 'flat prior', fontsize=6, color=INK2, ha='left')
    ax.set_xticks(centres); ax.set_xticklabels(['<-2', '-2..-1', '-1..0', '0..1', '1..2', '2..3', '>3'], fontsize=6)
    ax.set_xlabel(r'$\log_{10}$ signal / B12 signal ($\ell\approx1980$)'); ax.set_ylabel(r'mean $\log p$ in the bin')
    ax.set_ylim(-13.5, 1.5)
    ax.legend(loc='center right', fontsize=6)
    tag_axes(axes, [(0.97, 'right'), (0.03, 'left')])
    fig.tight_layout(w_pad=1.0)
    save(fig, f'metric_logp_{noise}')
    plt.close(fig)


def fig_metric_tarp(plt, noise='beam'):
    """TARP expected coverage against credibility for the three compressions."""
    rows = {m: d for m, d in method_rows(noise).items() if 'ecp' in d}
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axes[0]
    for m, d in rows.items():
        ax.plot(d['alpha'], d['ecp'], color=C[m], lw=1.3, label=NAMES[m])
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=':')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel(r'credibility level $\alpha$'); ax.set_ylabel('expected coverage')
    ax.legend(loc='upper left', fontsize=6.3)
    ax = axes[1]
    for m, d in rows.items():
        ax.plot(d['alpha'], d['ecp'] - d['alpha'], color=C[m], lw=1.3)
    band = 1.358 / np.sqrt(3000)
    ax.axhspan(-band, band, color=GRID, zorder=0)
    ax.axhline(0, color=INK2, lw=0.6)
    for a in (0.68, 0.95):
        ax.axvline(a, color=INK2, lw=0.6, ls=':')
    ax.set_xlim(0, 1); ax.set_xlabel(r'credibility level $\alpha$'); ax.set_ylabel(r'coverage $-\ \alpha$')
    tag_axes(axes, [(0.97, 'right', 0.04, 'bottom'), (0.03, 'left')])
    fig.tight_layout(w_pad=1.2)
    save(fig, f'metric_tarp_{noise}')
    plt.close(fig)


def fig_merlin_features(plt, label='beam_nn_merlin'):
    """Correlation of Merlin's 18 learned features with the nine parameters over the test rows (seed 1)."""
    p = METRIC_DATA / f'{label}.npz'
    if not p.exists():
        p = METRIC_DATA / 'prod_merlin.npz'                 # the production-noise network until the beam one exists
    if not p.exists():
        return
    d = np.load(p)
    if 'features' not in d.files:
        return
    f = d['features'][0].astype(float)                      # seed 1: (rows, 18)
    t = d['truth'].astype(float)
    corr = np.corrcoef(np.c_[f, t].T)[:f.shape[1], f.shape[1]:]
    # order features by the parameter they correlate with most
    best = np.argmax(np.abs(corr), axis=1)
    order = np.lexsort((-np.abs(corr).max(1), best))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(corr[order], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(9)); ax.set_xticklabels(TEX, fontsize=7)
    ax.set_yticks(range(f.shape[1])); ax.set_yticklabels([f'{k + 1}' for k in order], fontsize=6)
    ax.set_ylabel('learned feature'); ax.set_xlabel('parameter')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.set_label('correlation over test rows', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.tick_params(top=False, right=False)
    fig.tight_layout()
    save(fig, 'merlin_features')
    plt.close(fig)


def fig_obs_b12_three(plt):
    """B12 posterior of the three compressions (beam-consistent noise, bounded) as a corner plot."""
    files = dict(bins40=METRIC_DATA / 'beam_bins40_b12.npy',
                 pca=HERE / 'posterior_rows_whitening' / 'beam_nn_bounded_pca_b12.npy',
                 merlin=HERE / 'posterior_rows_whitening' / 'beam_nn_bounded_merlin_b12.npy')
    have = {m: p for m, p in files.items() if p.exists()}
    if len(have) < 2:
        return
    samples = [(np.load(p).astype(float), NAMES[m], C[m]) for m, p in have.items()]
    old = PF.OUT
    PF.OUT = OUT
    try:
        PF.corner(samples, PF.B12, 'input values', 'obs_b12_three', plt, smooth=(1.0, 1.6),
                  title='B12 mock, beam-consistent noise,\nbounded posteriors')
    finally:
        PF.OUT = old
    # publication_figures.save wrote pdf + png into OUT; keep the png only
    pdf = OUT / 'obs_b12_three.pdf'
    if pdf.exists():
        pdf.unlink()



def gnfw(x, theta, M, z):
    """P_th / P_200c (thermal pressure; P_e = 0.5176 P_th) of the engine's gNFW profile (XGPaint Battaglia16ThermalSZProfile): each of P0, x_c, beta is
    A (M / 1e14 Msun)^alpha_m (1 + z)^alpha_z; alpha = 1 and gamma = -0.3 fixed; x = r / R_200c."""
    P0, xc, beta = (theta[k] * (M / 1e14) ** theta[3 + k] * (1 + z) ** theta[6 + k] for k in range(3))
    return P0 * (x / xc) ** -0.3 * (1 + x / xc) ** -beta


def fig_profile(plt):
    """Step 1: the gNFW pressure profile at B12 for three masses and two redshifts; one parameter varied at a time."""
    x = np.logspace(-2, np.log10(4), 300)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    for M, col in ((1e13, C['blue']), (1e14, INK), (1e15, C['red'])):
        for z, ls in ((0, '-'), (1, (0, (3, 1.5)))):
            ax.plot(x, gnfw(x, B12, M, z), color=col, lw=1.1, ls=ls,
                    label=f'$M_{{200c}}=10^{{{int(np.log10(M))}}}\\,M_\\odot$' if z == 0 else None)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-2, 4)
    ax.set_xlabel(r'$x=r/R_{200c}$'); ax.set_ylabel(r'$P_{\rm th}/P_{200c}$')
    ax.legend(loc='lower left', fontsize=6, title='solid z = 0, dashed z = 1', title_fontsize=6)
    ax = axes[1]
    ax.plot(x, gnfw(x, B12, 1e14, 0.5), color=INK, lw=1.4, label='B12')
    for k, col in ((0, C['merlin']), (1, C['pca']), (2, C['bins40'])):
        for frac, ls in ((0.1, (0, (3, 1.5))), (0.9, ':')):
            th = B12.copy(); th[k] = LOW[k] + frac * WIDTH[k]
            ax.plot(x, gnfw(x, th, 1e14, 0.5), color=col, lw=1.0, ls=ls,
                    label=f'{TEX[k]} at 10 % / 90 % of its range' if frac == 0.1 else None)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-2, 4); ax.set_ylim(1e-4, 3e2)
    ax.set_xlabel(r'$x=r/R_{200c}$'); ax.set_title(r'$M_{200c}=10^{14}\,M_\odot$, $z=0.5$', fontsize=7.5)
    ax.legend(loc='lower left', fontsize=5.8)
    tag_axes(axes, [(0.97, 'right'), (0.97, 'right')])
    fig.tight_layout(w_pad=1.0)
    save(fig, 'step_profile')
    plt.close(fig)


FIGURES = dict(spectra=fig_spectra, design=fig_design, beam=fig_beam, bins40=fig_bins40, logit=fig_logit,
               training=fig_training, ensemble=fig_ensemble, metric_r=fig_metric_r, metric_width=fig_metric_width,
               metric_coverage=fig_metric_coverage, metric_sbc=fig_metric_sbc, compare_sizes=fig_compare_sizes,
               compare_scorecard=fig_compare_scorecard, compare_params=fig_compare_params, whitening=fig_whitening,
               pca=fig_pca, asinh=fig_asinh, metric_logp=fig_metric_logp, metric_tarp=fig_metric_tarp,
               merlin_features=fig_merlin_features, obs_b12_three=fig_obs_b12_three, profile=fig_profile)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', nargs='*', default=None)
    args = p.parse_args()
    plt = PF.setup()
    for name, func in FIGURES.items():
        if args.only and name not in args.only:
            continue
        func(plt)
        print('written', name, flush=True)


if __name__ == '__main__':
    main()
