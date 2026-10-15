"""Fisher information against the multipole averaging scale (section 5, Merlin and 50-multipole bands).

Reads only the saved Fisher forecast of the beam-consistent B12 spectrum,
SBI_analysis/tsz64k_sbi_local/results_fisher_b12_beam/fisher_forecast_b12.json (fisher_forecast_b12.py, run on idark
2026-10-02), and plots the geometric mean over the nine parameters of the marginal Fisher sd / prior width for
  - uniform bands of width 2..100 multipoles (2l+1 weights; 'bands of N'),
  - the derivatives low-passed to periods above P multipoles ('information_by_period'),
with the every-multipole, Merlin-PCA-input and 40-SBI-band values as references. No new computation.

    python fisher_band_scale.py   -> ../../fig/5_merlin/11_fisher_vs_band_width.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SRC = REPO / 'SBI_analysis/tsz64k_sbi_local/results_fisher_b12_beam/fisher_forecast_b12.json'
OUT = HERE.parents[1] / 'fig/5_merlin/11_fisher_vs_band_width.png'

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e8e8e5'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def gmean(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))


def main():
    r = json.loads(SRC.read_text())
    F = r['fisher']
    widths = [2, 5, 10, 20, 50, 100]
    band = [gmean(F[f'bands of {w}']['marginal_sd_prior_widths']) for w in widths]
    periods = [3, 5, 10, 20, 50, 200]
    low = [gmean(r['information_by_period'][f'periods above {p}']) for p in periods]
    every = gmean(F['every multipole']['marginal_sd_prior_widths'])
    pca = gmean(F['Merlin 75 PCA inputs']['marginal_sd_prior_widths'])      # key name kept; n_pca = 86 here
    sbi40 = gmean(F['40 SBI bands']['marginal_sd_prior_widths'])
    smooth = gmean(F['every multipole, derivatives smoothed']['marginal_sd_prior_widths'])

    plt.rcParams.update({'font.size': 12, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
    ax.set_facecolor('#fcfcfb')
    fig.patch.set_facecolor('#fcfcfb')
    ax.plot(widths, band, '-o', color=BLUE, lw=2, ms=8, mec='#fcfcfb', mew=2, zorder=3,
            label='uniform bands of width Δℓ (2ℓ+1 weights)')
    ax.plot(periods, low, '-s', color=ORANGE, lw=2, ms=8, mec='#fcfcfb', mew=2, zorder=3,
            label='derivatives low-passed: periods above P multipoles')
    ax.plot([100, 200], [band[-1], sbi40], ls=(0, (2, 2)), color=BLUE, lw=1.5, zorder=2)
    ax.plot([200], [sbi40], 'o', color='#fcfcfb', mec=BLUE, mew=2, ms=8, zorder=3)
    ax.annotate(f'40 SBI bands (\u0394\u2113 = 200): {sbi40:.2f}', (200, sbi40), xytext=(36, 2.45), color=INK,
                fontsize=10, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    for y, text in ((every, 'every multipole (7,900)'), (pca, "Merlin's 86 whitened PCA inputs (w420)")):
        ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(1.75, y * 1.04, f'{text}: {y:.2f}', color=INK2, fontsize=10, va='bottom', ha='left')
    ax.annotate(f'\u0394\u2113 = 50: {band[4]:.2f}', (50, band[4]), xytext=(20, 1.15), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.annotate(f'P = 50: {low[4]:.2f}', (50, low[4]), xytext=(62, 0.40), color=INK, fontsize=10,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.6, 260)
    ax.set_ylim(0.1, 3.2)
    ax.set_xticks([2, 3, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(['2', '3', '5', '10', '20', '50', '100', '200'])
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
    ax.set_yticklabels(['0.1', '0.2', '0.5', '1', '2'])
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_xlabel('averaging scale in multipoles (Δℓ or period cut P)')
    ax.set_ylabel('Fisher marginal sd / prior width\n(geometric mean over 9 parameters)')
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    ax.set_title('Beam-consistent B12 Fisher forecast: information lost by averaging in ℓ',
                 fontsize=12, color=INK, loc='left')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    print(OUT, dict(every=every, pca=pca, sbi40=sbi40, smooth50=smooth, band=dict(zip(widths, band)),
                    lowpass=dict(zip(periods, low))))


if __name__ == '__main__':
    main()
