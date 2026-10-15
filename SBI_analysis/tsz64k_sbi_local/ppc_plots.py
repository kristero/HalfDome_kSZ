"""Posterior-predictive SO mocks: Battaglia+12 versus the parameters recovered by the logit-space NPE ensembles.

  ppc_plots.py task --results results_logit --out mocks/pass2.toml [--points noisy:map clean:map noisy:median]
      writes the paint_mocks.jl task for the recovered parameter sets, each with the fitted observation's
      noise seeds (39001001, 39001002), so its SO mock differs from the observation only through the signal
  ppc_plots.py plot --results results_logit --b12 mocks/pass1_B12/B12 --recovered mocks/pass2
      fig_ppc_so_mock.png   SO mock (signal + noise cross spectrum) of B12 and of the noisy-inference estimates
      fig_ppc_noise_free.png  clean spectra of B12 and of the noise-free-inference estimate (+ the noisy one)
Error bars are the scatter of each 40-bin D_ell over the 20 extra SO noise realizations of the B12 sky.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ELL = np.arange(80, 7980)
DL = ELL * (ELL + 1) / (2 * np.pi)
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
OBS_SEEDS = (39001001, 39001002)
B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
NAMES = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc', 'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
COLORS = {'noisy:map': '#2a78d6', 'clean:map': '#eb6834', 'noisy:median': '#1baf7a', 'clean:median': '#1baf7a'}
MARKERS = {'noisy:map': 'o', 'clean:map': 's', 'noisy:median': 'D', 'clean:median': 'D'}
DESCR = {'noisy:map': 'MAP of the SO-mock posterior', 'noisy:median': 'marginal medians of the SO-mock posterior',
         'clean:map': 'MAP of the noise-free posterior', 'clean:median': 'marginal medians of the noise-free posterior'}


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                         'savefig.bbox': 'tight', 'savefig.dpi': 200})


def label_of(point):
    return point.replace(':', '_')


def rebin(cl):
    dl = np.asarray(cl)[ELL] * DL
    return np.array([np.average(dl[(ELL >= a) & (ELL < b)], weights=2 * ELL[(ELL >= a) & (ELL < b)] + 1) for a, b in zip(EDGES[:-1], EDGES[1:])])


def centres():
    return np.array([0.5 * (a + b - 1) for a, b in zip(EDGES[:-1], EDGES[1:])])


def ell_ticks(ax, ticks=(100, 200, 500, 1000, 2000, 5000)):
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()


def log_markers(ax, x, y, marker, color, size, label=None, zorder=5):
    """Band powers on a log axis: positive ones filled, negative ones (noise-dominated bins) open at |D_ell|."""
    pos = y > 0
    ax.plot(x[pos], y[pos], marker, ls='none', ms=size, color=color, mec='white', mew=0.6, zorder=zorder, label=label)
    if np.any(~pos):
        ax.plot(x[~pos], -y[~pos], marker, ls='none', ms=size, mfc='white', mec=color, mew=1.0, zorder=zorder)
    return int(np.sum(~pos))


def recovered_points(results, points):
    out = {}
    for point in points:
        kind, which = point.split(':')
        rec = json.loads((results / kind / 'recovered.json').read_text())['Battaglia12']
        out[point] = np.array(rec[which])
    return out


def cmd_task(args):
    pts = recovered_points(args.results, args.points)
    lines = [f'out = "{args.paint_out.resolve()}"']
    for point, theta in pts.items():
        lines += ['', '[[cases]]', f'label = "{label_of(point)}"', 'theta = [' + ', '.join(repr(float(v)) for v in theta) + ']',
                  f'seeds = [[{OBS_SEEDS[0]}, {OBS_SEEDS[1]}]]']
    args.out.write_text('\n'.join(lines) + '\n')
    print(args.out.read_text())


def param_text(theta):
    return ', '.join(f'{n}={v:.3g}' for n, v in zip(['P0', 'xc', 'β', 'αm,P0', 'αm,xc', 'αm,β', 'αz,P0', 'αz,xc', 'αz,β'], theta))


def cmd_plot(args):
    style()
    b12_dir, rec_dir = args.b12, args.recovered
    obs_file = b12_dir / f'noisy_cross_{OBS_SEEDS[0]}_{OBS_SEEDS[1]}.npy'
    others = sorted(p for p in b12_dir.glob('noisy_cross_*.npy') if p != obs_file)
    b12_obs = rebin(np.load(obs_file)); b12_clean = rebin(np.load(b12_dir / 'masked_clean_cl.npy'))
    b12_clean_unbinned = np.load(b12_dir / 'masked_clean_cl.npy')[ELL] * DL
    realizations = np.stack([rebin(np.load(p)) for p in others])
    sigma = realizations.std(0, ddof=1)                      # per-bin scatter of the B12 SO mock over new noise draws
    x = centres()
    points = [p for p in args.points if (rec_dir / label_of(p)).is_dir()]
    thetas = recovered_points(args.results, points)
    rec = {p: dict(obs=rebin(np.load(rec_dir / label_of(p) / f'noisy_cross_{OBS_SEEDS[0]}_{OBS_SEEDS[1]}.npy')),
                   clean=rebin(np.load(rec_dir / label_of(p) / 'masked_clean_cl.npy')),
                   clean_unbinned=np.load(rec_dir / label_of(p) / 'masked_clean_cl.npy')[ELL] * DL) for p in points}
    scale = 1e12
    report = dict(noise_realizations_for_sigma=len(others), chi2_of_b12_realizations_vs_clean=None, points={})
    chi_b12 = np.sum(((realizations - b12_clean) / sigma) ** 2, axis=1)
    report['chi2_of_b12_realizations_vs_clean'] = dict(mean=float(chi_b12.mean()), bins=len(x))

    # ---------------- figure 1: SO mocks (noisy inference) ----------------
    noisy_pts = [p for p in points if p.startswith('noisy')]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(16 / 2.54, 13 / 2.54), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
    ax.set_yscale('log', nonpositive='clip')
    ax.errorbar(x, np.abs(b12_obs) * scale, yerr=sigma * scale, fmt='none', ecolor=INK, elinewidth=0.9, capsize=0, zorder=4)
    negative = log_markers(ax, x, b12_obs * scale, 'o', INK, 4, 'Battaglia+12: SO mock that was fitted (signal + noise cross)', zorder=4)
    ax.plot(x, b12_clean * scale, color=INK, lw=1.2, zorder=3, label='Battaglia+12: clean (masked)')
    for k, p in enumerate(noisy_pts):
        c = COLORS[p]; off = 1 + 0.012 * (k + 1)
        resid = (rec[p]['obs'] - b12_obs) / sigma
        negative += log_markers(ax, x * off, rec[p]['obs'] * scale, MARKERS[p], c, 4.5,
                                f'{DESCR[p]}: SO mock, same noise ($\\chi^2$ = {np.sum(resid ** 2):,.0f} for {len(x)} bins)')
        ax.plot(x, rec[p]['clean'] * scale, color=c, lw=1.2, zorder=2, label=f'{DESCR[p]}: clean')
        axr.plot(x * off, resid, MARKERS[p], ms=4, color=c, mec='white', mew=0.6, zorder=3)
        chi2 = float(np.sum(resid ** 2))
        report['points'][p] = dict(theta=thetas[p].tolist(), chi2_so_mock_vs_observation=chi2, bins=len(x),
                                   max_abs_pull=float(np.max(np.abs(resid))))
    ax.set_xscale('log'); ax.set_ylabel(r'$D_\ell\ [10^{-12}]$ (40 bins)')
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title('SO mock of Battaglia+12 and of the recovered parameters (logit-space NPE ensemble, SO-mock inference)', fontsize=9)
    axr.axhline(0, color=INK2, lw=0.8)
    axr.axhspan(-1, 1, color='0.92', zorder=0, lw=0)
    axr.set_ylabel(r'(recovered $-$ fitted) / $\sigma$'); axr.set_xlabel(r'$\ell$'); ell_ticks(axr)
    axr.grid(True, color=GRID, lw=0.5); axr.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    if negative:
        handles.append(plt.Line2D([], [], marker='o', ls='none', ms=4, mfc='white', mec=INK2, mew=1.0))
        labels.append(r'open marker: negative band power (noise-dominated), $|D_\ell|$ shown')
    handles.append(plt.Rectangle((0, 0), 1, 1, color='0.92')); labels.append(f'lower panel: $\\pm1\\sigma$, with $\\sigma$ from {len(others)} new SO noise draws of the B12 sky')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=1, frameon=False, fontsize=7)
    fig.savefig(args.results / 'fig_ppc_so_mock.png'); plt.close(fig)

    # ---------------- figure 2: noise-free spectra (noise-free inference, with the noisy one for reference) ----------------
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(16 / 2.54, 13 / 2.54), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
    ax.plot(ELL, b12_clean_unbinned * scale, color=INK, lw=1.4, label='Battaglia+12 (truth)', zorder=4)
    order = [p for p in points if p.startswith('clean')] + [p for p in points if p.startswith('noisy')]
    for p in order:
        c = COLORS[p]; ls = '-' if p.startswith('clean') else (0, (4, 2))
        ax.plot(ELL, rec[p]['clean_unbinned'] * scale, color=c, lw=1.3, ls=ls, label=DESCR[p], zorder=3)
        ratio = 100 * (rec[p]['clean_unbinned'] / b12_clean_unbinned - 1)
        axr.plot(ELL, ratio, color=c, lw=1.1, ls=ls)
        report['points'].setdefault(p, dict(theta=thetas[p].tolist()))
        report['points'][p]['clean_percent_difference'] = dict(median_abs=float(np.median(np.abs(ratio))), max_abs=float(np.max(np.abs(ratio))))
    band = 100 * sigma / np.abs(b12_clean)
    axr.fill_between(x, -band, band, color='0.9', lw=0, zorder=0, label=r'SO noise $\pm1\sigma$ per 40-bin band')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel(r'$D_\ell\ [10^{-12}]$ (masked, clean)')
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title('Noise-free spectra: Battaglia+12 and the recovered parameters', fontsize=9)
    axr.axhline(0, color=INK2, lw=0.8); axr.set_ylabel('recovered / B12 $-$ 1 [%]'); axr.set_xlabel(r'$\ell$'); ell_ticks(axr)
    axr.set_ylim(-100, 40)
    axr.grid(True, color=GRID, lw=0.5); axr.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    fig.legend(handles + h2, labels + l2, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=7)
    fig.savefig(args.results / 'fig_ppc_noise_free.png'); plt.close(fig)
    (args.results / 'ppc.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    for p in points:
        print(p, param_text(thetas[p]))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('task'); a.add_argument('--results', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    a.add_argument('--paint-out', type=Path, default=Path('mocks/pass2')); a.add_argument('--points', nargs='+', default=['noisy:map', 'clean:map'])
    a = sub.add_parser('plot'); a.add_argument('--results', type=Path, required=True); a.add_argument('--b12', type=Path, required=True)
    a.add_argument('--recovered', type=Path, required=True); a.add_argument('--points', nargs='+', default=['noisy:map', 'clean:map', 'noisy:median'])
    args = p.parse_args()
    dict(task=cmd_task, plot=cmd_plot)[args.cmd](args)


if __name__ == '__main__':
    main()
