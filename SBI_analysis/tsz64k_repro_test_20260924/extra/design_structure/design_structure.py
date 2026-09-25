"""Quantify the visible structure of the 65,536-row scrambled Sobol design.

Compares the design with independent uniform draws of the same size:
correlations, 2-D t-values of the Sobol nets, cell-count dispersion on
g x g grids (1 = as uneven as random, 0 = perfectly even), and subsets taken
by row stride. Writes design_structure.json and design_structure.png.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

design_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
manifest = json.loads((design_dir / 'design_manifest.json').read_text())
names = manifest['parameter_order']
lo, hi = np.asarray(manifest['lower']), np.asarray(manifest['upper'])
theta = np.load(design_dir / manifest['files']['theta_design'])
unit = qmc.Sobol(d=9, scramble=True, seed=20260920).random_base2(16)
assert np.allclose(lo + (hi - lo) * unit, theta, rtol=0, atol=1e-12), 'design is not the Sobol draw'
pairs = list(combinations(range(9), 2))
rng = np.random.default_rng(12345)
iid = rng.random(unit.shape)


def dispersion(points, g, i, j):
    """var/mean of cell counts on a g x g grid; about 1 for independent uniform draws."""
    counts = np.histogram2d(points[:, i], points[:, j], bins=g, range=[[0, 1], [0, 1]])[0].ravel()
    expected = len(points) / g ** 2
    return float(counts.var() / expected / (1 - 1 / g ** 2))


def t_value(points, i, j):
    """Smallest t for which the 2-D projection is a (t, m, 2)-net in base 2."""
    m = int(np.log2(len(points)))
    assert 2 ** m == len(points)
    for t in range(m + 1):
        if all(np.unique(np.floor(points[:, i] * 2 ** a).astype(np.int64) * 2 ** (m - t - a)
                         + np.floor(points[:, j] * 2 ** (m - t - a)).astype(np.int64),
                         return_counts=True)[1].min() == 2 ** t and
               len(np.unique(np.floor(points[:, i] * 2 ** a).astype(np.int64) * 2 ** (m - t - a)
                             + np.floor(points[:, j] * 2 ** (m - t - a)).astype(np.int64))) == 2 ** (m - t)
               for a in range(m - t + 1)):
            return t
    return m


def max_abs_corr(points):
    c = np.corrcoef(points.T)
    return float(np.max(np.abs(c[np.triu_indices(9, 1)])))


report = dict(generator='scipy.stats.qmc.Sobol(d=9, scramble=True, seed=20260920).random_base2(16)')
grids = [2, 4, 8, 16, 32, 64, 128, 256]
for count in (2048, 65536):
    s, r = unit[:count], iid[:count]
    entry = dict(max_abs_correlation=dict(sobol=max_abs_corr(s), independent=max_abs_corr(r),
                                          independent_expected_scale=1 / np.sqrt(count)))
    entry['t_values'] = {f'{names[i]}|{names[j]}': t_value(s, i, j) for i, j in pairs}
    entry['dispersion'] = {}
    for g in grids:
        if count / g ** 2 < .5:
            continue
        ds = [dispersion(s, g, i, j) for i, j in pairs]
        dr = [dispersion(r, g, i, j) for i, j in pairs]
        worst = int(np.argmax(ds))
        entry['dispersion'][g] = dict(points_per_cell=count / g ** 2, sobol_median=float(np.median(ds)),
                                      sobol_max=float(max(ds)),
                                      sobol_max_pair=f'{names[pairs[worst][0]]}|{names[pairs[worst][1]]}',
                                      sobol_pairs_above_1=int(sum(d > 1 for d in ds)),
                                      independent_median=float(np.median(dr)))
    report[f'first_{count}_rows'] = entry

# Subsets by stride: what a careless train/validation split would see.
strides = {}
for k in (2, 3, 4, 5, 8, 10, 16):
    for offset in (0, 1):
        sub = unit[offset::k]
        d1 = max(float(np.histogram(sub[:, i], bins=16, range=(0, 1))[0].var() / (len(sub) / 16) / (1 - 1 / 16))
                 for i in range(9))
        d2 = max(dispersion(sub, 8, i, j) for i, j in pairs)
        halves = [float(np.mean(sub[:, i] < .5)) for i in range(9)]
        worst = int(np.argmax([abs(v - .5) for v in halves]))
        strides[f'rows[{offset}::{k}]'] = dict(rows=len(sub), max_1d_dispersion_16bins=d1,
                                              max_2d_dispersion_8x8=d2, max_abs_correlation=max_abs_corr(sub),
                                              most_lopsided_parameter=names[worst],
                                              fraction_in_lower_half=halves[worst])
random_subset = unit[np.sort(rng.choice(len(unit), 6554, replace=False))]
strides['random 10% of rows'] = dict(
    rows=len(random_subset),
    max_1d_dispersion_16bins=max(float(np.histogram(random_subset[:, i], bins=16, range=(0, 1))[0].var()
                                       / (len(random_subset) / 16) / (1 - 1 / 16)) for i in range(9)),
    max_2d_dispersion_8x8=max(dispersion(random_subset, 8, i, j) for i, j in pairs),
    max_abs_correlation=max_abs_corr(random_subset))
report['subsets'] = strides
(out_dir / 'design_structure.json').write_text(json.dumps(report, indent=1))

# ------------------------------------------------------------------ figure
label = {'P0': r'$P_0$', 'xc': r'$x_c$', 'beta': r'$\beta$', 'alpha_m_P0': r'$\alpha_m^{P_0}$',
         'alpha_m_xc': r'$\alpha_m^{x_c}$', 'alpha_m_beta': r'$\alpha_m^{\beta}$',
         'alpha_z_P0': r'$\alpha_z^{P_0}$', 'alpha_z_xc': r'$\alpha_z^{x_c}$', 'alpha_z_beta': r'$\alpha_z^{\beta}$'}
first = unit[:2048]
show = sorted(pairs, key=lambda p: (-dispersion(first, 32, *p), -dispersion(first, 64, *p)))[:3]
blue, orange, ink, muted, grid_c = '#2f6fd6', '#e0672a', '#1d1d1f', '#5f6368', '#c4c8ce'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': muted, 'axes.labelcolor': ink,
                     'xtick.color': muted, 'ytick.color': muted})
fig = plt.figure(figsize=(14, 9), facecolor='white')
outer = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.3], wspace=.36, hspace=.5,
                         left=.05, right=.985, top=.845, bottom=.07)
for column, (i, j) in enumerate(show):
    for row_index, (points, colour, title) in enumerate([(first, blue, 'Sobol design, rows 0-2,047'),
                                                         (iid[:2048], orange, '2,048 independent uniform draws')]):
        ax = fig.add_subplot(outer[row_index, column])
        x, y = lo[i] + (hi[i] - lo[i]) * points[:, i], lo[j] + (hi[j] - lo[j]) * points[:, j]
        if row_index == 0:  # the 16 x 16 grid in which every Sobol cell holds exactly 8 points
            for edge in np.linspace(0, 1, 17)[1:-1]:
                ax.axvline(lo[i] + (hi[i] - lo[i]) * edge, color=grid_c, lw=.75, zorder=0)
                ax.axhline(lo[j] + (hi[j] - lo[j]) * edge, color=grid_c, lw=.75, zorder=0)
        ax.scatter(x, y, s=2.2, color=colour, linewidths=0, rasterized=True, zorder=2)
        ax.set_xlim(lo[i], hi[i]); ax.set_ylim(lo[j], hi[j])
        ax.set_xlabel(label[names[i]]); ax.set_ylabel(label[names[j]], labelpad=1)
        ax.tick_params(labelsize=8, length=2)
        ax.set_title(title, fontsize=9.5, color=ink, loc='left')
ax = fig.add_subplot(outer[0, 3])
for count, style in ((2048, '-'), (65536, '--')):
    entry = report[f'first_{count}_rows']['dispersion']
    g = np.array(sorted(entry))
    ax.plot(g, [entry[k]['sobol_max'] for k in g], style, color=blue, lw=2, marker='o', ms=4.5,
            label=f'Sobol, most uneven of 36 pairs, {count:,} rows')
ax.axhline(1, color=orange, lw=2, label='independent uniform draws')
ax.set_xscale('log', base=2)
ax.set_xticks(grids); ax.set_xticklabels([f'1/{g}' for g in grids], fontsize=8)
ax.set_ylim(-.06, 2.9)
ax.set_xlabel('grid cell width / parameter range')
ax.set_ylabel('cell-count variance / random expectation')
ax.grid(color=grid_c, lw=.6); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc='upper left')
ax.set_title('0 = every cell holds exactly its share', fontsize=10, color=ink, loc='left')
stride = report['subsets']['rows[0::10]']
k = names.index(stride['most_lopsided_parameter'])
ax = fig.add_subplot(outer[1, 3])
edges = lo[k] + (hi[k] - lo[k]) * np.linspace(0, 1, 17)
ax.hist(lo[k] + (hi[k] - lo[k]) * unit[0::10, k], bins=edges, color=blue, alpha=.55,
        label=f'every 10th row, rows[::10] ({len(unit[0::10]):,})')
ax.hist(lo[k] + (hi[k] - lo[k]) * random_subset[:, k], bins=edges, histtype='step', color=orange, lw=2,
        label=f'random 10% of rows ({len(random_subset):,})')
ax.set_xlim(lo[k], hi[k])
ax.set_xlabel(label[names[k]]); ax.set_ylabel('rows per bin')
ax.legend(fontsize=8, frameon=False, loc='upper right')
ax.set_title('Taking every 10th row is not a fair subset', fontsize=10, color=ink, loc='left')
fig.text(.05, .96, 'The Sobol design is deliberately not random', fontsize=15, weight='bold', color=ink)
fig.text(.05, .918, 'Top: rows 0-2,047 (as in the prior plot) for the three pairs with the strongest pattern. In each '
         'of these panels every cell of the grey 16 x 16 grid holds exactly 8 points;\nthe checkerboards appear only '
         'at 1/32 of the range. Bottom: independent draws, with clumps and holes. With all 65,536 rows every pair '
         'is exact down to 1/128.', fontsize=9.5, color=muted, va='top')
fig.savefig(out_dir / 'design_structure.png', dpi=110)
print(json.dumps({k: v for k, v in report.items() if k != 'subsets'}, indent=1)[:6000])
print(json.dumps(report['subsets'], indent=1))
