"""One-page figure of the idark reproduction test (results/reproduction_test.png)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXTRA = Path(__file__).resolve().parent
sys.path.insert(0, str(EXTRA))
from run_extra import CFG, DL, ELL, WHERE, common, manage  # noqa: E402

BLUE, ORANGE, GREEN, VERMILION = '#0072B2', '#E69F00', '#009E73', '#D55E00'   # Okabe-Ito
INK, MUTED, GRID = '#1d1d1f', '#5f6368', '#e3e5e8'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left'})
summary = json.loads((EXTRA / 'results' / 'summary.json').read_text())
noise = EXTRA / 'noise_check'
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4), facecolor='white')
fig.subplots_adjust(left=.055, right=.985, top=.865, bottom=.075, wspace=.27, hspace=.42)

# (a) same seeds
ax = axes[0, 0]
labels, sets = [], [json.loads((WHERE['repro'] / 'repro_result.json').read_text()),
                    json.loads((EXTRA / 'results' / 'repro_batch040.json').read_text())]
names = [('masked_clean_cl.npy', 'masked clean', BLUE, 'o'), ('unmasked_clean_cl.npy', 'full-sky clean', ORANGE, 's'),
         ('masked_noisy_cross_cl.npy', 'noisy split cross', GREEN, '^')]
position = 0
for result in sets:
    for label, row in result['rows'].items():
        for k, (key, name, colour, marker) in enumerate(names):
            ax.scatter(position + (k - 1) * .18, max(row[key], 1e-17), color=colour, marker=marker, s=34,
                       zorder=3, label=name if position == 0 else None)
        labels.append(str(int(label)))
        position += 1
ax.axhline(1e-6, color=VERMILION, lw=1.4, ls='--', label='tolerance 1e-6')
ax.set_yscale('log'); ax.set_ylim(1e-17, 1e-4)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.set_xlabel('row of the 256-row test (rows 160-163 include two refined-cache rows)')
ax.set_ylabel('max |new - test| / signal, l = 80..7979')
ax.grid(color=GRID, lw=.6, axis='y'); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc='upper left', ncol=2)
ax.set_title('Same seeds: rerun reproduces the test')

# (b) noise amplitude
ax = axes[0, 1]
bins = np.loadtxt(noise / 'bins.csv', delimiter=',', ndmin=2)
ratio = np.loadtxt(noise / 'auto_ratio.csv', delimiter=',', ndmin=2)
centre = (bins[:, 0] + bins[:, 1]) / 2
band = np.sqrt(2 * bins[:, 3]) / bins[:, 2] / np.sqrt(len(ratio))
ax.fill_between(centre, 1 - 2 * band, 1 + 2 * band, color=BLUE, alpha=.15, lw=0, label='expected 2-sigma range of the mean')
ax.plot(centre, ratio.mean(axis=0), 'o-', color=BLUE, ms=3.5, lw=1.5, label=f'mean of {len(ratio)} realizations')
ax.axhline(1, color=MUTED, lw=.8)
ax.set_xlabel('multipole l (40 bins of 200)'); ax.set_ylabel('measured noise power / SO baseline Deproj-0 N_l')
ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc='upper right')
ax.set_title('Noise amplitude matches the SO table')

# (c) independence
ax = axes[0, 2]
allpairs = np.loadtxt(noise / 'allpairs_lowl_z.csv', delimiter=',')
full = np.concatenate([np.loadtxt(noise / f'pair_z_{k}.csv', delimiter=',', ndmin=2).ravel() for k in
                       ('same_row_splits', 'consecutive_rows', 'train_vs_test_seed_same_row')])
edges = np.linspace(-5, 5, 51)
ax.hist(allpairs, bins=edges, density=True, color=BLUE, alpha=.45,
        label=f'all {len(allpairs) // 3:,} pairs of realizations, l = 80-679 ({len(allpairs):,} values)')
ax.hist(full, bins=edges, density=True, histtype='step', lw=1.8, color=ORANGE,
        label=f'split 1 x split 2, row r x row r+1, new x test seed; all 40 bins ({len(full):,})')
grid = np.linspace(-5, 5, 400)
ax.plot(grid, np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2, label='independent: standard normal')
control = summary['noise_seeds']['same_seed_twice']['min_z']
ax.text(.02, .97, f'same seed drawn twice: z >= {control:,.0f} (off scale)', transform=ax.transAxes,
        fontsize=8.5, color=VERMILION, va='top')
ax.set_xlabel('cross-spectrum z-score per l bin'); ax.set_ylabel('density')
ax.set_ylim(0, .62); ax.legend(fontsize=7.6, frameon=False, loc='upper left', bbox_to_anchor=(0, .92))
ax.set_title('Different seeds: independent noise')

# (d) production residuals
ax = axes[1, 0]
res = np.load(EXTRA / 'results' / 'residuals.npz')
z_new, z_old = res['new'] / res['sigma'], res['test'] / res['sigma']
same = [np.corrcoef(a, b)[0, 1] for a, b in zip(z_new, z_old)]
between = [np.corrcoef(z_new[i], z_new[j])[0, 1] for i in range(len(z_new)) for j in range(i)]
sigma = 1 / np.sqrt(z_new.shape[1] - 1)
ax.axhspan(-2 * sigma, 2 * sigma, color=MUTED, alpha=.12, lw=0, label='2-sigma range for independent noise')
jitter = np.random.default_rng(1).uniform(-.18, .18, len(between))
ax.scatter(np.zeros(len(same)) + np.linspace(-.15, .15, len(same)), same, color=BLUE, s=22, zorder=3,
           label='same row: new seeds vs test seeds')
ax.scatter(1 + jitter, between, color=ORANGE, s=12, alpha=.8, zorder=3, label='two different rows, new seeds')
ax.set_xticks([0, 1]); ax.set_xticklabels([f'{len(same)} rows (0-15)', f'{len(between)} row pairs'])
ax.set_xlim(-.6, 1.6); ax.set_ylim(-1, 1)
ax.set_ylabel('correlation of (noisy - clean) over 40 bins')
ax.grid(color=GRID, lw=.6, axis='y'); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, loc='upper right')
ax.set_title('Production rows: noise differs from the test')

# (e) profile cache accuracy
ax = axes[1, 1]
_, theta, _ = manage.load_design()
tasks = sorted(int(p.name) for p in WHERE['audits'].glob('[0-9]*'))
results, _ = manage.collect_audits(CFG, theta, [r for t in tasks for r in range(256 * t, 256 * (t + 1))])
edges = np.geomspace(1e-5, 1e-2, 46)
for (low, high, colour, name) in ((0, 256, ORANGE, 'rows 0-255 (the test rows)'),
                                  (256, 70000, BLUE, 'new rows (256-511, 32768-33023, 65280-65535)')):
    values = [v['max_relative_visible'] for r, v in results.items() if low <= r < high]
    ax.hist(values, bins=edges, histtype='step' if low == 0 else 'stepfilled', lw=1.8, color=colour,
            alpha=1 if low == 0 else .45, label=f'{name}: {len(values)}')
ax.axvline(.004, color=VERMILION, lw=1.4, ls='--', label='accuracy target 0.4%')
ax.set_xscale('log'); ax.set_xlabel('max relative error of the painted profile cache vs direct integration')
ax.set_ylabel('rows'); ax.legend(fontsize=8, frameon=False, loc='upper left')
los = summary['profiles']['independent_los']
worst_los = max(v['max_relative_visible'] for v in los.values() if isinstance(v, dict))
ax.text(.02, .70, f"independent SciPy line-of-sight integration\nvs the engine's direct columns: max rel. diff {worst_los:.1e}",
        transform=ax.transAxes, fontsize=8.3, color=INK, va='top')
ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.set_title('Profiles: every audited row within 0.4%')

# (f) spot checks
ax = axes[1, 2]
ax.axhline(1e-2, color=VERMILION, lw=1.4, ls='--', label='target: 1% at every l')
for group, (spot, colours) in enumerate(zip(summary['profiles']['spots'], ((BLUE, ORANGE), (GREEN, MUTED)))):
    for record, colour in zip(spot['rows'], colours):
        row = record['row']
        folder = EXTRA / 'spots' / f'spot{group}'
        c = np.load(folder / f'{row:05d}_candidate' / 'masked_clean_cl.npy')[ELL] * DL
        r = np.load(folder / f'{row:05d}_reference' / 'masked_clean_cl.npy')[ELL] * DL
        ax.plot(ELL, np.maximum(np.abs(c / r - 1), 1e-12), color=colour, lw=.7,
                label=f"row {row} ({'x'.join(map(str, record['nodes']))} vs 2x): max "
                      f"{record['masked_clean_cl.npy']['max_fractional_Dell']:.1e}")
ax.set_yscale('log'); ax.set_ylim(1e-10, 1e-1)
ax.set_xlabel('multipole l'); ax.set_ylabel('|clean D_l, audited cache / 2x finer cache - 1|')
ax.grid(color=GRID, lw=.6); ax.set_axisbelow(True)
ax.legend(fontsize=7.6, frameon=False, loc='upper right', ncol=1)
ax.set_title('Full-sky spectra: cache grid is converged')

fig.text(.055, .955, 'Reproduction test of the 65,536-row bundle on idark', fontsize=15, weight='bold', color=INK)
fig.text(.055, .915, 'Fresh GitHub clone (tag tsz64k-flat-prior-v1) and a fresh Julia 1.12.2 install from the internet; '
         'XGPaint from GitHub (tag validated-tsz256-20260922). All maps from the full 85,224,251-halo catalogue.',
         fontsize=9.5, color=MUTED)
out = EXTRA / 'results' / 'reproduction_test.png'
fig.savefig(out, dpi=110)
print(out)
