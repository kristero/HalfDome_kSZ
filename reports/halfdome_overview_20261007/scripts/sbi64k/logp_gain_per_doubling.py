"""Test log p against training-set size and the log p gained per doubling of N, from stored metrics only.

Reads (nothing is retrained or re-evaluated):
  SBI_analysis/tsz64k_sbi_local/results_final/bins40_noisy/N*/metrics.json          40 bands, production noise,
                                                                                     median-scale asinh (first run)
  SBI_analysis/tsz64k_sbi_local/results_final_noise/bins40_noisy/N*/metrics.json    the same, noise-level asinh
  SBI_analysis/tsz64k_sbi_local/beamfix/fetched/results/series.json                 40 bands, beam-consistent noise
                                                                                     (series_noise/bins40_beam)
  SBI_analysis/tsz64k_sbi_local/results_merlin_unbinned_beam_asinh_p30[_N<N>]/metrics_{merlin,pca}.json
                                                                                     Merlin and PCA, beam-consistent
                                                                                     noise, B12 whitening
  SBI_analysis/tsz64k_sbi_local/results_524k_moped/N*/metrics.json                  historical 524k narrow-prior
                                                                                     set, fixed noise maps, MOPED
Writes fig/3_sbi64k/14_logp_gain_per_doubling.png and prints the numbers (also saved as JSON next to this script).

    LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python logp_gain_per_doubling.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path('/home/cbllover/HalfDome')
SBI = REPO / 'SBI_analysis' / 'tsz64k_sbi_local'
OUT = REPO / 'reports' / 'halfdome_overview_20261007' / 'fig' / '3_sbi64k' / '14_logp_gain_per_doubling.png'
NUMBERS = Path(__file__).with_name('logp_gain_per_doubling.json')
SIZES = (2000, 4000, 8000, 16000, 32000, 61440)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
C40, CPCA, CMER, CMOPED = '#eda100', '#1baf7a', '#4a3aa7', '#8a8a85'   # the project's colours (doc_figures.C)


def mj(path):
    return json.loads(Path(path).read_text())


def series():
    """{label: (N array, log p array, style dict)} for every curve."""
    out = {}
    for folder, label, ls in (('results_final', '40 bins, production noise, median-scale asinh', ':'),
                              ('results_final_noise', '40 bins, production noise, noise-level asinh', '--')):
        N = [n for n in SIZES if (SBI / folder / 'bins40_noisy' / f'N{n}' / 'metrics.json').exists()]
        lp = [mj(SBI / folder / 'bins40_noisy' / f'N{n}' / 'metrics.json')['mean_log_prob_theta'] for n in N]
        out[label] = (np.array(N, float), np.array(lp), dict(color=C40, ls=ls, marker='o', mfc='white'))
    beam = mj(SBI / 'beamfix' / 'fetched' / 'results' / 'series.json')['series_noise']['bins40_beam']
    N = [8000, 16000, 32000, 61440]                       # the doublings of the staged series (it also has 24k, 40k...)
    out['40 bins, beam-consistent noise (staged)'] = (np.array(N, float),
                                                      np.array([beam[str(n)]['mean_log_prob_theta'] for n in N]),
                                                      dict(color=C40, ls='-', marker='o'))
    for v, label, colour, marker in (('pca', 'PCA, beam-consistent, B12 whitening', CPCA, 's'),
                                     ('merlin', 'Merlin, beam-consistent, B12 whitening', CMER, 'D')):
        lp = []
        for n in SIZES:
            folder = SBI / ('results_merlin_unbinned_beam_asinh_p30' + ('' if n == 61440 else f'_N{n}'))
            lp.append(mj(folder / f'metrics_{v}.json')['mean_log_prob_theta'])
        out[label] = (np.array(SIZES, float), np.array(lp), dict(color=colour, ls='-', marker=marker))
    prep = mj(SBI / 'results_524k_moped' / 'prepare.json')
    N = prep['sizes']
    lp = [mj(SBI / 'results_524k_moped' / f'N{n}' / 'metrics.json')['mean_log_prob_theta'] for n in N]
    out['524k narrow prior, fixed noise, MOPED (other dataset)'] = (np.array(N, float), np.array(lp),
                                                                     dict(color=CMOPED, ls='-', marker='v'))
    return out


def main():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.2,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6, 'axes.edgecolor': INK2,
                         'xtick.color': INK2, 'ytick.color': INK2, 'axes.labelcolor': INK, 'text.color': INK,
                         'axes.unicode_minus': False, 'savefig.dpi': 200})
    S = series()
    numbers = {}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.7), gridspec_kw=dict(wspace=0.28))
    for label, (N, lp, st) in S.items():
        other = label.startswith('524k')
        # local gain per doubling between consecutive points, plotted at their geometric mean
        gain = np.diff(lp) / np.diff(np.log2(N))
        mid = np.sqrt(N[1:] * N[:-1])
        slope = float(np.polyfit(np.log2(N), lp, 1)[0])            # log-linear gain over the whole range
        numbers[label] = dict(N=N.tolist(), log_p=lp.tolist(), gain_per_doubling_local=gain.tolist(),
                              gain_per_doubling_loglinear=slope)
        kw = dict(color=st['color'], ls=st['ls'], marker=st['marker'], ms=4.5, lw=1.3,
                  mfc=st.get('mfc', st['color']), mec=st['color'])
        if not other:
            a1.plot(N, lp, label=label, **kw)
        a2.plot(mid, gain, label=f'{label}: {slope:.2f} / doubling overall', **kw)
    a1.axhline(-12.8975, color=INK2, lw=0.8, ls='-.')
    a1.text(2100, -12.6, 'flat prior: -12.90', fontsize=7, color=INK2)
    a1.set_xscale('log')
    a1.set_xlabel('training simulations N')
    a1.set_ylabel('mean log p of the true parameters (nats)')
    a1.set_title('(a) 64k flat prior: 3,000 fixed test rows, 3-seed ensembles', fontsize=8.5)
    # one shared legend below the panels (the same line styles in (a) and (b))
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xscale('log')
    a2.set_xlabel('training simulations N (geometric mean of the two sizes)')
    a2.set_ylabel('log p gained per doubling of N (nats)')
    a2.set_title('(b) local gain per doubling; grey: 524k set (495 test rows)', fontsize=8.5)
    a2.legend(loc='upper center', bbox_to_anchor=(-0.17, -0.17), ncol=2, frameon=False, fontsize=6.8)
    for ax in (a1, a2):
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.savefig(OUT, bbox_inches='tight')
    NUMBERS.write_text(json.dumps(numbers, indent=1) + '\n')
    for label, d in numbers.items():
        print(f"{label}: overall {d['gain_per_doubling_loglinear']:.2f}; local "
              + ', '.join(f'{g:.2f}' for g in d['gain_per_doubling_local']))
    print('written', OUT)


if __name__ == '__main__':
    main()
