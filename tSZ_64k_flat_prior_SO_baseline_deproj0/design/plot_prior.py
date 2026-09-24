"""Corner plot of the 65,536-row flat-prior Sobol design (prior_65536.png).

Diagonal: 1-D marginals of all rows (64 bins). Lower triangle: the first 2,048
rows, a balanced Sobol prefix, so the space filling stays visible; the text
reports how evenly all 65,536 rows fill a 32 x 32 grid in every pair. The
orange cross is the Battaglia+12 fiducial, shown for scale only.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {'P0': r'$P_0$', 'xc': r'$x_c$', 'beta': r'$\beta$',
          'alpha_m_P0': r'$\alpha_m^{P_0}$', 'alpha_m_xc': r'$\alpha_m^{x_c}$',
          'alpha_m_beta': r'$\alpha_m^{\beta}$', 'alpha_z_P0': r'$\alpha_z^{P_0}$',
          'alpha_z_xc': r'$\alpha_z^{x_c}$', 'alpha_z_beta': r'$\alpha_z^{\beta}$'}
B12 = dict(P0=18.1, xc=0.497, beta=4.35, alpha_m_P0=0.154, alpha_m_xc=-0.00865,
           alpha_m_beta=0.0393, alpha_z_P0=-0.758, alpha_z_xc=0.731, alpha_z_beta=0.415)
# Reference palette: series slot 1 (blue), slot 2 (orange), light blue fill, chrome.
BLUE, ORANGE, FILL = '#2a78d6', '#eb6834', '#b7d3f6'
SURFACE, INK, INK2, MUTED, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#c3c2b7'
BINS_1D, GRID, PREFIX = 64, 32, 2048


def main():
    manifest = json.loads((HERE / 'design_manifest.json').read_text())
    theta = np.load(HERE / manifest['files']['theta_design'])
    names, low, high = manifest['parameter_order'], manifest['lower'], manifest['upper']
    count, dim = theta.shape
    unit = (theta - np.array(low)) / (np.array(high) - np.array(low))
    cells = [np.histogram2d(unit[:, j], unit[:, i], bins=GRID, range=[[0, 1], [0, 1]])[0]
             for i in range(dim) for j in range(i)]
    cell_min, cell_max = int(min(c.min() for c in cells)), int(max(c.max() for c in cells))

    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2,
                         'ytick.labelcolor': INK2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'axes.labelcolor': INK})
    fig, axes = plt.subplots(dim, dim, figsize=(14, 14), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.985, wspace=0.08, hspace=0.08)
    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                counts, edges = np.histogram(theta[:, i], bins=BINS_1D, range=(low[i], high[i]))
                ax.stairs(counts, edges, fill=True, color=FILL, linewidth=0)
                ax.stairs(counts, edges, color=BLUE, linewidth=1.2)
                ax.axvline(B12[names[i]], color=ORANGE, linewidth=1.2)
                ax.set_ylim(0, 1.6 * count / BINS_1D)
                ax.set_yticks([])
                ax.set_xlim(low[i], high[i])
            else:
                ax.scatter(theta[:PREFIX, j], theta[:PREFIX, i], s=1.2, color=BLUE, alpha=0.75,
                           linewidths=0, rasterized=True)
                ax.plot(B12[names[j]], B12[names[i]], marker='X', markersize=8, color=ORANGE,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
                ax.set_xlim(low[j], high[j])
                ax.set_ylim(low[i], high[i])
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=2.5, pad=1.5, labelsize=7)
            # Edge ticks would collide with the neighbouring panel; the table lists the bounds.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            if i != j:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            if i == dim - 1:
                ax.set_xlabel(LABELS[names[j]], fontsize=11)
                ax.tick_params(axis='x', labelrotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[names[i]], fontsize=11)
            else:
                ax.set_yticklabels([])

    left = 0.36
    fig.text(left, 0.955, 'Flat prior: 65,536-row scrambled Sobol design', fontsize=17, color=INK, weight='bold')
    per_cell = f'exactly {cell_min}' if cell_min == cell_max else f'{cell_min}-{cell_max}'
    notes = [f'{count:,} rows, nine independent uniforms in the physical parameters (no rejection),',
             'scipy.stats.qmc.Sobol(d=9, scramble=True, seed=20260920).random_base2(16).',
             f'Diagonal: all rows; every 1-D marginal has exactly {count // BINS_1D:,} rows in each of {BINS_1D} bins.',
             f'Lower panels: the first {PREFIX:,} rows (a balanced Sobol prefix). All {count:,} rows put',
             f'{per_cell} rows in every cell of a {GRID} x {GRID} grid, for all 36 parameter pairs.',
             'Rows 0-255 are the validated 256-row test design; two SO baseline Deproj-0',
             'noise seeds per row, unique across all 131,072 splits.']
    for k, line in enumerate(notes):
        fig.text(left, 0.93 - 0.0175 * k, line, fontsize=10.5, color=INK2)

    table_x, header_y = 0.50, 0.79
    columns = [('Parameter', table_x, 'left'), ('Lower', table_x + 0.12, 'right'),
               ('Upper', table_x + 0.19, 'right'), ('B12 fiducial', table_x + 0.29, 'right')]
    for title, x, align in columns:
        fig.text(x, header_y, title, fontsize=10.5, color=INK, weight='bold', ha=align)
    for k, name in enumerate(names):
        y = header_y - 0.02 * (k + 1)
        values = [LABELS[name], f'{low[k]:g}', f'{high[k]:g}', f'{B12[name]:g}']
        for (title, x, align), value in zip(columns, values):
            fig.text(x, y, value, fontsize=11 if title == 'Parameter' else 10.5,
                     color=INK if title == 'Parameter' else INK2, ha=align)

    fig.legend(handles=[
        Line2D([], [], color=BLUE, linewidth=1.2, label=f'1-D marginal, all rows ({BINS_1D} bins)'),
        Line2D([], [], color=BLUE, marker='o', markersize=3, linewidth=0, label=f'first {PREFIX:,} rows'),
        Line2D([], [], color=ORANGE, marker='X', markersize=8, markeredgecolor=SURFACE,
               linewidth=1.2, label='Battaglia+12 fiducial (reference only)')],
        loc='upper left', bbox_to_anchor=(0.60, 0.565), frameon=False, fontsize=10, labelcolor=INK2)
    out = HERE / 'prior_65536.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print('Saved', out, f'(2-D cell counts {cell_min}-{cell_max})')


if __name__ == '__main__':
    main()
