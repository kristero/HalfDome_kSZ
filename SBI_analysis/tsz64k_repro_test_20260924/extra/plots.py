"""Figures that document every check of the idark reproduction test (results/plots/*.png)."""
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXTRA = Path(__file__).resolve().parent
sys.path.insert(0, str(EXTRA))
from run_extra import CFG, DL, ELL, TEST, WHERE, common, manage  # noqa: E402

OUT = EXTRA / 'results' / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, VERMILION, SKY, PURPLE = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9', '#CC79A7'
INK, MUTED, GRID = '#1d1d1f', '#5f6368', '#e3e5e8'
plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlesize': 10.5, 'axes.titlelocation': 'left',
                     'axes.titlecolor': INK, 'legend.fontsize': 8, 'legend.frameon': False})
summary = json.loads((EXTRA / 'results' / 'summary.json').read_text())
residual_check = json.loads((EXTRA / 'results' / 'residual_check.json').read_text())
_, THETA, SEEDS = manage.load_design()
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
CENTRE = (EDGES[:-1] + EDGES[1:] - 1) / 2
TABLE = np.loadtxt(common.SO_DIR / 'SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt', comments='#')
NOISE = np.zeros(7980)
NOISE[TABLE[:, 0].astype(int)] = TABLE[:, 1]


def test_dir(row):
    return TEST / 'diagnostic_256' / 'batches' / f'{row // 4:03d}' / f'{row:05d}'


def dl(folder, name):
    return np.load(folder / name)[ELL] * DL


def binned(values):
    return manage.rebin_unbinned(values[None, :], ELL)[0]


def expected_sigma(full_sky_signal_cl, w4=residual_check['w4']):
    """Analytic scatter of the binned noisy-minus-clean D_l (see residual_check.py)."""
    var = w4 * (2 * full_sky_signal_cl[ELL] * NOISE[ELL] + NOISE[ELL] ** 2) / (2 * ELL + 1) * DL ** 2
    out = []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = (ELL >= a) & (ELL < b)
        w = (2 * ELL[sel] + 1) / np.sum(2 * ELL[sel] + 1)
        out.append(np.sqrt(np.sum(w ** 2 * var[sel])))
    return np.array(out)


def style(ax, title, grid='both'):
    ax.set_title(title)
    if grid:
        ax.grid(color=GRID, lw=.6, axis=grid)
    ax.set_axisbelow(True)


def finish(fig, name, title, subtitle):
    fig.text(.05, .962, title, fontsize=15, weight='bold', color=INK)
    fig.text(.05, .928, subtitle, fontsize=9.5, color=MUTED, va='top')
    fig.savefig(OUT / name, dpi=110, facecolor='white')
    plt.close(fig)
    print(OUT / name)


def checklist(ax, title, lines):
    ax.axis('off')
    ax.set_title(title)
    y = .96
    for ok, text in lines:
        ax.text(0, y, '✓' if ok else '✗', color=GREEN if ok else VERMILION, fontsize=13,
                weight='bold', transform=ax.transAxes, va='top')
        ax.text(.07, y - .005, text, fontsize=8.8, color=INK, transform=ax.transAxes, va='top', wrap=True)
        y -= .12 if '\n' not in text else .17


def rss_and_minutes(folder):
    status = json.loads((folder / 'status.json').read_text())
    text = (folder / 'time.txt').read_text() if (folder / 'time.txt').exists() else ''
    match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', text)
    return status['seconds'] / 60, (int(match.group(1)) / 2 ** 20 if match else np.nan)


def new_axes(rows=2, cols=3, height=9.2):
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, height), facecolor='white')
    fig.subplots_adjust(left=.055, right=.985, top=.855, bottom=.07, wspace=.28, hspace=.44)
    return fig, axes


# ================================================================== 1. code: same seeds -> same numbers
fig, axes = new_axes()
groups = [(WHERE['repro'] / 'batch000', range(0, 4), BLUE, 'rows 0-3 (manage.py repro-check)'),
          (EXTRA / 'repro_batches' / 'batch040', range(160, 164), GREEN, 'rows 160-163 (two on the finer grid)')]
ax = axes[0, 0]
for folder, rows, colour, name in groups:
    for k, row in enumerate(rows):
        ax.plot(ELL, dl(folder / f'{row:05d}', 'masked_clean_cl.npy'), color=colour, lw=1.1,
                label=f'rerun, {name}' if k == 0 else None)
        ax.plot(ELL[::160], dl(test_dir(row), 'masked_clean_cl.npy')[::160], 'o', mfc='none', mec=INK, ms=3.2, mew=.7,
                label='256-row test (every 160th l)' if (k == 0 and colour == BLUE) else None)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(top=ax.get_ylim()[1] * 1e4)
ax.set_xlabel('multipole l'); ax.set_ylabel('masked clean D_l of Compton-y')
ax.legend(loc='upper right', fontsize=7.4)
style(ax, 'Rerun lies on the test, row by row')
for ax, name, metric_label in ((axes[0, 1], 'masked_clean_cl.npy', 'clean (masked; full-sky dashed)'),
                               (axes[0, 2], 'masked_noisy_cross_cl.npy', 'noisy split cross spectrum')):
    for folder, rows, colour, gname in groups:
        for k, row in enumerate(rows):
            new, ref = np.load(folder / f'{row:05d}' / name)[ELL], np.load(test_dir(row) / name)[ELL]
            clean_ref = np.load(test_dir(row) / 'masked_clean_cl.npy')[ELL]
            scale = np.abs(ref) + (clean_ref if name != 'masked_clean_cl.npy' else 0)
            ax.plot(ELL, np.maximum(np.abs(new - ref) / scale, 1e-18), color=colour, lw=.5, alpha=.8,
                    label=gname if k == 0 else None)
            if name == 'masked_clean_cl.npy':
                full_new = np.load(folder / f'{row:05d}' / 'unmasked_clean_cl.npy')[ELL]
                full_ref = np.load(test_dir(row) / 'unmasked_clean_cl.npy')[ELL]
                ax.plot(ELL, np.maximum(np.abs(full_new / full_ref - 1), 1e-18), color=colour, lw=.5, ls='--', alpha=.6)
    ax.axhline(1e-6, color=VERMILION, lw=1.4, ls='--', label='pass threshold 1e-6')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-18, 1e-3)
    ax.set_xlabel('multipole l'); ax.set_ylabel('|rerun - test| / signal')
    ax.legend(loc='upper left')
results_rows = [json.loads((WHERE['repro'] / 'repro_result.json').read_text())['rows'],
                json.loads((EXTRA / 'results' / 'repro_batch040.json').read_text())['rows']]
worst_clean = max(max(v['masked_clean_cl.npy'], v['unmasked_clean_cl.npy']) for rows in results_rows for v in rows.values())
worst_noisy = max(v['masked_noisy_cross_cl.npy'] for rows in results_rows for v in rows.values())
style(axes[0, 1], f'Clean spectra agree to {worst_clean:.0e} (round-off)')
style(axes[0, 2], f'Noisy spectra agree to {worst_noisy:.0e}')
ax = axes[1, 0]
reference = json.loads((common.REFERENCE / 'audits.json').read_text())['rows']
new_audits, _ = manage.collect_audits(CFG, THETA, range(256))
for grid, colour in (([256, 128, 64], BLUE), ([512, 256, 128], ORANGE)):
    rows = [r for r in range(256) if reference[r]['nodes'] == grid]
    ax.scatter([reference[r]['max_relative_visible'] for r in rows], [new_audits[r]['max_relative_visible'] for r in rows],
               s=14, color=colour, label=f"{'x'.join(map(str, grid))}: {len(rows)} rows", zorder=3)
ax.plot([1e-5, 1e-2], [1e-5, 1e-2], color=MUTED, lw=.8, zorder=1, label='identical')
ax.axvline(.004, color=VERMILION, lw=1, ls=':'); ax.axhline(.004, color=VERMILION, lw=1, ls=':', label='0.4% target')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1e-5, 1e-2); ax.set_ylim(1e-5, 1e-2)
ax.set_xlabel('test: max cache error vs direct integration'); ax.set_ylabel('rerun: same quantity')
ax.legend(loc='upper left')
style(ax, 'Rows 0-255: same cache grid, same audit numbers')
ax = axes[1, 1]
ax.axis('off'); ax.set_title('Noise maps: SHA-256 of every split is identical')
ax.text(0, .97, 'row  split   rerun               256-row test', family='monospace', fontsize=8.6, color=MUTED,
        transform=ax.transAxes, va='top')
y = .905
for folder, rows, colour, _ in groups:
    for row in rows:
        new_hash = common.load_toml(folder / f'{row:05d}' / 'observation.toml')['noise_sha256']
        old_hash = common.load_toml(test_dir(row) / 'observation.toml')['noise_sha256']
        for split in (0, 1):
            same = new_hash[split] == old_hash[split]
            ax.text(0, y, f'{row:>4}   {split + 1}    {new_hash[split][:16]}  {old_hash[split][:16]}', family='monospace',
                    fontsize=8.6, color=INK, transform=ax.transAxes, va='top')
            ax.text(.93, y, '✓' if same else '✗', color=GREEN if same else VERMILION, fontsize=11,
                    weight='bold', transform=ax.transAxes, va='top')
            y -= .056
try:
    sources_ok = len(common.verify_sources()['frozen_sha256']) == 39
except RuntimeError:
    sources_ok = False
env_ok = any('verify_env: OK' in p.read_text(errors='replace') for p in WHERE['logs'].glob('*.OU'))
job_folders = ([WHERE['repro'] / 'batch000', EXTRA / 'repro_batches' / 'batch040'] + sorted(WHERE['batches'].iterdir())
               + [EXTRA / 'spots' / f'spot{g}' for g in (0, 1)])
halo_counts = {common.load_toml(f / 'batch.toml')['selected_halos'] for f in job_folders}
rng_ok = all(json.loads(p.read_text())['checks']['operator_probe_exact'][k]
             for p in (WHERE['repro'] / 'repro_result.json', EXTRA / 'results' / 'repro_batch040.json')
             for k in ('rng_uniform', 'rng_normal', 'xgpaint_sha256', 'operator_sha256', 'mask_pixel_sha256'))
checklist(axes[1, 2], 'Exact-identity checks', [
    (sources_ok, '39 frozen files match their SHA-256 (38 byte-identical\nto the test; Manifest.toml: only XGPaint\'s source line)'),
    (env_ok, 'Julia 1.12.2, 12 package versions, 17 XGPaint files\n(XGPaint from GitHub, tree 3b57cacb): verify_env OK'),
    (rng_ok, 'MersenneTwister fingerprint, XGPaint, operator and\nmask hashes identical to the test\'s operator probe'),
    (halo_counts == {common.SELECTED_HALOS},
     f'{common.SELECTED_HALOS:,} halos selected in all {len(job_folders)} full-catalogue jobs'),
    (summary['code_same_seeds']['rows_0_3']['passed'] and summary['code_same_seeds']['rows_160_163']['passed'],
     'repro-check (rows 0-3) and test batch 40 (rows 160-163)\npass the 1e-6 criterion'),
    (summary['code_same_seeds']['compare_test256']['audits']['worst_metric_relative_difference'] == 0,
     'Audits of rows 0-255: 256/256 same grid,\nlargest metric difference 0.0')])
finish(fig, 'fig1_same_seeds_reproduce_the_test.png', 'Test 1: the pushed code reproduces the 256-row test',
       'Fresh GitHub clone (tag tsz64k-flat-prior-v1) and a fresh Julia install, rerun with the test\'s parameters, '
       'noise seeds and cache grids, on the full 85,224,251-halo catalogue. Round-off from threaded summation is ~1e-13.')

# ================================================================== 2. noise seeds
fig, axes = new_axes()
noise = EXTRA / 'noise_check'
bins = np.loadtxt(noise / 'bins.csv', delimiter=',', ndmin=2)
ratio = np.loadtxt(noise / 'auto_ratio.csv', delimiter=',', ndmin=2)
nreal = len(ratio)
ax = axes[0, 0]
unit = SEEDS.astype(float) / 2 ** 63
test_seeds = np.load(common.REFERENCE / 'noise_seeds.npy').astype(float) / 2 ** 63
ax.scatter(unit[:, 0], unit[:, 1], s=.25, color=BLUE, alpha=.35, linewidths=0, rasterized=True)
ax.scatter(test_seeds[:, 0], test_seeds[:, 1], s=6, color=ORANGE, linewidths=0)
ax.plot([], [], 'o', color=BLUE, ms=3, label='65,536 rows of this dataset')
ax.plot([], [], 'o', color=ORANGE, ms=3, label='256 rows of the test')
design = summary['noise_seeds']['design']
ax.text(.03, .03, f"{design['unique']:,} distinct seeds of {design['splits']:,}\n{design['overlap_with_test_seeds']} shared "
        f"with the test; {design['rows_with_equal_splits']} rows with split 1 = split 2", transform=ax.transAxes,
        fontsize=8.4, color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
ax.set_xlabel('split-1 seed / 2^63'); ax.set_ylabel('split-2 seed / 2^63')
ax.legend(loc='upper right', framealpha=.9, frameon=True, edgecolor='none')
style(ax, 'Seeds: SHA-256 derived, no pattern, no repeats', grid=None)
ax = axes[0, 1]
model = np.array([np.sum((2 * np.arange(a, b) + 1) * NOISE[a:b] * np.arange(a, b) * (np.arange(a, b) + 1) / (2 * np.pi))
                  / np.sum(2 * np.arange(a, b) + 1) for a, b in zip(EDGES[:-1], EDGES[1:])])
ax.plot(ELL, NOISE[ELL] * DL, color=INK, lw=1.3, label='SO LAT baseline, ILC Deproj-0 (column 2)', zorder=3)
ax.plot(CENTRE, ratio.mean(0) * model, 'o', color=BLUE, ms=4, label=f'measured, mean of {nreal} realizations', zorder=4)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('multipole l'); ax.set_ylabel('noise D_l')
ax.legend(loc='upper left')
style(ax, 'Every realization follows the SO noise curve')
ax = axes[0, 2]
single = np.sqrt(2 * bins[:, 3]) / bins[:, 2]
for r in ratio:
    ax.plot(CENTRE, 100 * (r - 1), color=SKY, lw=.4, alpha=.12)
ax.plot(CENTRE, 200 * single, color=MUTED, lw=1, ls='--', label='expected 2-sigma, one realization')
ax.plot(CENTRE, -200 * single, color=MUTED, lw=1, ls='--')
ax.plot(CENTRE, 100 * (ratio.mean(0) - 1), 'o-', color=BLUE, ms=3, lw=1.3, label=f'mean of {nreal} (within +-0.03%)')
ax.axhline(0, color=INK, lw=.7)
ax.plot([], [], color=SKY, lw=1, label=f'each of the {nreal} realizations')
ax.set_xscale('log'); ax.set_ylim(-2.6, 2.6)
ax.set_xlabel('multipole l'); ax.set_ylabel('measured power / SO N_l - 1 (%)')
ax.legend(loc='upper right')
style(ax, 'Noise amplitude is exactly the SO table')
ax = axes[1, 0]
allpairs = np.loadtxt(noise / 'allpairs_lowl_z.csv', delimiter=',')
first_bin = allpairs[:nreal * (nreal - 1) // 2]
matrix = np.full((nreal, nreal), np.nan)
position = 0
for j in range(nreal):
    for i in range(j):
        matrix[i, j] = matrix[j, i] = first_bin[position]
        position += 1
cmap = copy.copy(plt.get_cmap('RdBu_r')); cmap.set_bad(INK)
image = ax.imshow(matrix, cmap=cmap, vmin=-4, vmax=4, interpolation='nearest')
for edge in (255.5,):
    ax.axhline(edge, color=INK, lw=.8); ax.axvline(edge, color=INK, lw=.8)
ax.set_xticks([128, 320]); ax.set_xticklabels(['this dataset,\nrows 0-127 x 2 splits', 'test seeds,\nrows 0-63 x 2'])
ax.set_yticks([128, 320]); ax.set_yticklabels(['this dataset', 'test'], rotation=90, va='center')
bar = fig.colorbar(image, ax=ax, fraction=.046, pad=.03); bar.set_label('cross-spectrum z, l = 80-279')
style(ax, f'All {len(first_bin):,} pairs uncorrelated (diagonal: self)', grid=None)
ax = axes[1, 1]
full = np.concatenate([np.loadtxt(noise / f'pair_z_{k}.csv', delimiter=',', ndmin=2).ravel()
                       for k in ('same_row_splits', 'consecutive_rows', 'train_vs_test_seed_same_row')])
edges = np.linspace(-5.5, 5.5, 45)
width = edges[1] - edges[0]
grid = np.linspace(-5.5, 5.5, 400)
for values, colour, name, kind in ((allpairs, BLUE, f'all pairs, l 80-679 ({len(allpairs):,})', 'stepfilled'),
                                   (full, ORANGE, f'split, neighbour, new-vs-test\npairs, all l ({len(full):,})', 'step')):
    ax.hist(values, bins=edges, histtype=kind, color=colour, alpha=.5 if kind == 'stepfilled' else 1, lw=1.8, label=name)
    ax.plot(grid, len(values) * width * np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=.9)
ax.plot([], [], color=INK, lw=.9, label='standard normal')
ax.set_yscale('log'); ax.set_ylim(.5, 3e7)
tail = summary['noise_seeds']['all_pairs_low_ell']['all_low_bins']
ax.text(.03, .97, f"|z| > 4: {tail['n_abs_gt_4']} seen, {len(allpairs) * 6.334e-5:.0f} expected\n|z| > 5: 0 seen, "
        f"{len(allpairs) * 5.733e-7:.1f} expected\nsame seed twice: z = "
        f"{summary['noise_seeds']['same_seed_twice']['min_z']:.0f}", transform=ax.transAxes, va='top',
        fontsize=8.4, color=INK)
ax.set_xlabel('cross-spectrum z-score'); ax.set_ylabel('count per bin')
ax.legend(loc='upper right', fontsize=7.6)
style(ax, 'Different seeds give independent maps')
ax = axes[1, 2]
res = np.load(EXTRA / 'results' / 'residuals_normalized.npz')
ax.scatter(res['z_test'].ravel(), res['z_new'].ravel(), s=9, color=BLUE, alpha=.55, linewidths=0)
ax.axhline(0, color=MUTED, lw=.7); ax.axvline(0, color=MUTED, lw=.7)
lim = 4.2
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
corr = residual_check['corr_same_row_new_vs_test']
ax.text(.03, .97, f"16 rows x 40 bins; z = (noisy - clean) / expected sigma\nstd {residual_check['z_new']['std']:.2f} "
        f"(new), {residual_check['z_test']['std']:.2f} (test); expected 1\nper-row correlation "
        f"{corr['mean']:+.2f} +- {corr['sem']:.2f}; independent: 0", transform=ax.transAxes, va='top', fontsize=8.4,
        color=INK, bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
ax.set_xlabel('residual z with the test\'s noise seeds'); ax.set_ylabel('residual z with this dataset\'s seeds')
style(ax, 'Production rows 0-15: new noise, same statistics')
finish(fig, 'fig2_noise_seeds_are_independent.png', 'Test 2: every row gets its own SO baseline Deproj-0 noise',
       f'{nreal} full-resolution realizations (lmax 7979) drawn with the engine\'s own code, Healpix.synalm with '
       'MersenneTwister(seed), for rows 0-127 of this dataset and rows 0-63 of the test (two splits each).\n'
       'The regenerated maps of production rows 0-3 and 65532-65535 are bit-identical (SHA-256) to the maps the '
       'production jobs used.')

# ================================================================== 3. profiles
fig, axes = new_axes()
row = 65370
probes = np.loadtxt(WHERE['audits'] / f'{row // 256:04d}' / f'{row:05d}' / 'probes_256.csv', delimiter=',').reshape(64, 39, 6)
visible = probes[:, :, 3] > 1e-5 * probes[:, :, 5]
errors = np.where(visible, np.abs(probes[:, :, 4] / probes[:, :, 3] - 1), 0)
worst = int(np.argmax(errors.max(axis=1)))
halos = [worst] + [h for h in (0, 36, 63) if h != worst][:3]
los = np.load(EXTRA / 'results' / 'los_records.npy')
colours = [VERMILION, BLUE, GREEN, PURPLE]
ax = axes[0, 0]
for h, colour in zip(halos, colours):
    x, direct, estimate, central = probes[h, :, 2], probes[h, :, 3], probes[h, :, 4], probes[h, :, 5]
    name = f"log M = {probes[h, 0, 0]:.2f}, z = {probes[h, 0, 1]:.3g}" + (' (largest error)' if h == worst else '')
    ax.plot(x, direct / central, color=colour, lw=1.3, label=name)
    ax.plot(x, estimate / central, 'o', mfc='none', mec=colour, ms=3.5, mew=.8)
    mine = los[(los[:, 0] == row) & (los[:, 1] == h)]
    ax.plot(probes[h, mine[:, 2].astype(int), 2], mine[:, 4], 'x', color=INK, ms=6, mew=1.2)
ax.plot([], [], 'o', mfc='none', mec=MUTED, label='painted cache (256x128x64)')
ax.plot([], [], 'x', color=INK, label='independent SciPy integration')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-9, 3)
ax.set_xlabel('projected radius / R200c'); ax.set_ylabel('column / central column')
ax.legend(loc='lower left', fontsize=7.4)
style(ax, f'Row {row} (hardest in 1,024): profiles, lines = direct')
ax = axes[0, 1]
ax.axhspan(-.4, .4, color=MUTED, alpha=.12, lw=0, label='0.4% accuracy target')
for h, colour in zip(halos, colours):
    x = probes[h, :, 2]
    rel = 100 * (probes[h, :, 4] / probes[h, :, 3] - 1)
    keep = visible[h]
    ax.plot(x[keep], rel[keep], 'o-', color=colour, ms=3, lw=1)
ax.set_xscale('log'); ax.set_ylim(-.6, .6)
ax.set_xlabel('projected radius / R200c'); ax.set_ylabel('cache / direct integration - 1 (%)')
ax.legend(loc='lower left')
style(ax, f'Same row: cache error <= {100 * errors.max():.2f}% at all radii')
ax = axes[0, 2]
tasks = sorted(int(p.name) for p in WHERE['audits'].glob('[0-9]*'))
results, _ = manage.collect_audits(CFG, THETA, [r for t in tasks for r in range(256 * t, 256 * (t + 1))])
edges = np.geomspace(1e-5, 1e-2, 46)
for low, high, colour, name, kind in ((256, 70000, BLUE, 'new rows', 'stepfilled'), (0, 256, ORANGE, 'rows 0-255', 'step')):
    values = [v['max_relative_visible'] for r, v in results.items() if low <= r < high]
    ax.hist(values, bins=edges, histtype=kind, lw=1.8, color=colour, alpha=.5 if kind == 'stepfilled' else 1,
            label=f'{name}: {len(values)}')
ax.axvline(.004, color=VERMILION, lw=1.4, ls='--', label='0.4% target')
ax.set_xscale('log'); ax.set_xlabel('max cache error per row (2,496 direct points each)'); ax.set_ylabel('rows')
ax.legend(loc='upper left')
style(ax, f"All {summary['profiles']['resolved']:,} audited rows accepted")
ax = axes[1, 0]
rel = np.abs(los[:, 4] / los[:, 5] - 1)
rel = rel[los[:, 5] > 1e-10]
ax.hist(np.maximum(rel, 1e-17), bins=np.geomspace(1e-17, 1e-5, 49), color=BLUE, alpha=.6)
ax.axvline(1e-7, color=VERMILION, lw=1.4, ls='--', label='criterion of the test (1e-7)')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(.5, 1e8)
ax.set_xlabel('|SciPy / engine - 1| per point (exact agreement at 1e-17)'); ax.set_ylabel('points')
ax.text(.03, .97, f"{len(rel):,} points: 1,024 rows x 64 halos\nx 3-4 radii; largest difference {rel.max():.1e}",
        transform=ax.transAxes, va='top', fontsize=8.6, color=INK)
ax.legend(loc='upper right', labels=["test's criterion 1e-7"])
style(ax, 'Independent line-of-sight integration agrees')
ax = axes[1, 1]
ax.axhline(1e-2, color=VERMILION, lw=1.4, ls='--', label='target: 1% at every l')
for group, (spot, pair_colours) in enumerate(zip(summary['profiles']['spots'], ((VERMILION, BLUE), (GREEN, PURPLE)))):
    for record, colour in zip(spot['rows'], pair_colours):
        folder = EXTRA / 'spots' / f'spot{group}'
        c = dl(folder / f"{record['row']:05d}_candidate", 'masked_clean_cl.npy')
        r = dl(folder / f"{record['row']:05d}_reference", 'masked_clean_cl.npy')
        ax.plot(ELL, np.maximum(np.abs(c / r - 1), 1e-12), color=colour, lw=.6,
                label=f"row {record['row']} ({'x'.join(map(str, record['nodes']))}): max "
                      f"{record['masked_clean_cl.npy']['max_fractional_Dell']:.1e}")
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-11, 1e-1)
ax.set_xlabel('multipole l'); ax.set_ylabel('|D_l, audited cache / 2x finer cache - 1|')
ax.legend(loc='upper left', fontsize=7.4)
style(ax, 'Full-sky spectra: the cache grid has converged')
ax = axes[1, 2]
test_grids, new_grids = summary['profiles']['cache_grids_rows_0_255'], summary['profiles']['cache_grids_new_rows']
labels = ['256x128x64', '512x256x128', '1024x512x256']
x = np.arange(len(labels))
for offset, (grids, colour, name) in zip((-.18, .18), ((test_grids, ORANGE, 'rows 0-255'), (new_grids, BLUE, 'new rows'))):
    total = sum(grids.values())
    share = [100 * grids.get(k, 0) / total for k in labels]
    bars = ax.bar(x + offset, share, width=.34, color=colour, label=f'{name} ({total})')
    for bar, value in zip(bars, share):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f'{value:.1f}%', ha='center', fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 100)
ax.set_xlabel('cache grid needed to reach the 0.4% target'); ax.set_ylabel('share of rows (%)')
ax.legend(loc='upper right')
style(ax, 'New prior regions need the same refinement', grid='y')
finish(fig, 'fig3_profiles_are_accurate.png', 'Test 3: the Battaglia+12 pressure profiles are accurate',
       'Per-row audit of 1,024 rows (0-511, 32768-33023, 65280-65535): painted cache vs direct integration of the '
       '4 R200c gas sphere, an independent SciPy integration,\nand full-sky spectra of the four spot rows (the test\'s '
       'rule: two largest cache errors, faintest, brightest) with a 2x finer cache.')

# ================================================================== 4. production path end to end
fig, axes = new_axes()
produced = sorted(int(p.name) for p in WHERE['rows'].iterdir())
first, new_rows = [r for r in produced if r < 256], [r for r in produced if r >= 256]
ax = axes[0, 0]
for row in first:
    for name, style_ in (('masked_clean_cl.npy', '-'), ('unmasked_clean_cl.npy', '--')):
        new, ref = dl(WHERE['rows'] / f'{row:05d}', name), dl(test_dir(row), name)
        ax.plot(ELL, np.maximum(np.abs(new / ref - 1), 1e-18), color=BLUE, lw=.4, alpha=.6, ls=style_)
ax.plot([], [], color=BLUE, label='masked clean, 16 rows'); ax.plot([], [], color=BLUE, ls='--', label='full-sky clean')
ax.axhline(1e-6, color=VERMILION, lw=1.4, ls='--', label='pass threshold 1e-6')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-18, 1e-3)
ax.set_xlabel('multipole l'); ax.set_ylabel('|production / test - 1|')
ax.legend(loc='upper left')
style(ax, 'Rows 0-15 via the batch plan: clean = test')
ax = axes[0, 1]
ax.axhspan(-2, 2, color=MUTED, alpha=.08, lw=0, label='expected 2 sigma')
ax.axhspan(-1, 1, color=MUTED, alpha=.16, lw=0, label='expected 1 sigma')
for row, marker in ((5, 'o'), (12, 's')):
    folder = WHERE['rows'] / f'{row:05d}'
    clean = binned(dl(folder, 'masked_clean_cl.npy'))              # equal to the test's clean spectrum
    sigma = expected_sigma(np.load(folder / 'unmasked_clean_cl.npy'))
    for source, colour, name in ((folder, BLUE, "this dataset's seeds"), (test_dir(row), ORANGE, "the test's seeds")):
        z = (binned(dl(source, 'masked_noisy_cross_cl.npy')) - clean) / sigma
        ax.plot(CENTRE, z, marker, color=colour, ms=3.8, mfc=colour if marker == 'o' else 'none', mew=1,
                label=f'row {row}, {name}')
ax.axhline(0, color=INK, lw=.7)
ax.set_xscale('log'); ax.set_ylim(-4.5, 5.2)
ax.set_xlabel('multipole l'); ax.set_ylabel('(noisy - clean) / expected sigma, 40 bins')
ax.legend(loc='upper left', ncol=2, fontsize=7.2)
style(ax, 'Rows 5 and 12: two independent noise draws each')
batches = summary['production']['batches']
meta = summary['production']['collect']
compare = summary['code_same_seeds']['compare_test256']
maps = summary['noise_seeds']['check_maps_equal_production_maps']
git = lambda *a: subprocess.run(['git', '-C', str(common.BUNDLE), *a], stdout=subprocess.PIPE,  # noqa: E731
                                universal_newlines=True).stdout.strip()
clone_clean = git('status', '--porcelain', '--', '.') == ''
at_tag = git('rev-parse', 'HEAD') == git('rev-parse', 'tsz64k-flat-prior-v1^{commit}') == meta['bundle_git']['commit']
worst_bins = max(compare['clean_spectra_bins40']['worst_masked'], compare['clean_spectra_bins40']['worst_unmasked'])
checklist(axes[0, 2], 'End-to-end checks', [
    (all(b['returncode'] == 0 for b in batches.values()),
     f"{len(batches)} production batches ({sum(len(b['rows']) for b in batches.values())} rows): return code 0"),
    (meta['rows'] == 16 and meta['independent_noise_realizations'] == 32,
     'collect --rows 16: parameters and seeds equal the design;\nchecksums, mask, halo count; 32 distinct noise maps'),
    (compare['passed'], f'compare-test256: 256 audits identical; clean spectra\nof rows 0-15 within {worst_bins:.0e} '
                        '(40 bins, tolerance 1e-6)'),
    (all(maps.values()), 'noise maps of rows 0-3 and 65532-65535 equal\nthe independently regenerated maps (SHA-256)'),
    (clone_clean, 'fresh clone unchanged after all runs (git status clean)'),
    (at_tag, f"runs used commit {meta['bundle_git']['commit'][:7]} = tag tsz64k-flat-prior-v1")])
ax = axes[1, 0]
for k, row in enumerate(new_rows):
    folder = WHERE['rows'] / f'{row:05d}'
    colour = [BLUE, GREEN, PURPLE, ORANGE, SKY, VERMILION, INK, MUTED][k % 8]
    ax.plot(ELL, dl(folder, 'masked_clean_cl.npy'), color=colour, lw=1, label=f'row {row}')
for row in first:
    ax.plot(ELL, dl(WHERE['rows'] / f'{row:05d}', 'masked_clean_cl.npy'), color=GRID, lw=.8, zorder=0)
ax.plot([], [], color=GRID, label='rows 0-15')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('multipole l'); ax.set_ylabel('masked clean D_l')
ax.legend(loc='lower center', ncol=3, fontsize=7.2)
style(ax, 'New rows 508-511 and 65532-65535')
ax = axes[1, 1]
z_new_rows = []
for row in new_rows:
    folder = WHERE['rows'] / f'{row:05d}'
    residual = binned(dl(folder, 'masked_noisy_cross_cl.npy') - dl(folder, 'masked_clean_cl.npy'))
    z_new_rows.append(residual / expected_sigma(np.load(folder / 'unmasked_clean_cl.npy')))
z_new_rows = np.concatenate(z_new_rows)
edges = np.linspace(-4, 4, 25)
ax.hist(res['z_new'].ravel(), bins=edges, density=True, histtype='step', lw=1.8, color=BLUE,
        label=f'rows 0-15, this dataset\'s seeds (std {np.std(res["z_new"]):.2f})')
ax.hist(z_new_rows, bins=edges, density=True, color=GREEN, alpha=.45,
        label=f'new rows (std {np.std(z_new_rows):.2f})')
ax.plot(grid := np.linspace(-4, 4, 300), np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1,
        label='expected: standard normal')
ax.set_xlabel('(noisy - clean) / expected sigma, per bin'); ax.set_ylabel('density')
ax.set_ylim(0, .78); ax.legend(loc='upper left', fontsize=7.4)
style(ax, 'Noise level in production is as expected')
ax = axes[1, 2]
jobs = [('repro 0-3', WHERE['repro'] / 'batch000'), ('test b40', EXTRA / 'repro_batches' / 'batch040')]
jobs += [(f'batch {int(p.name)}', p) for p in sorted(WHERE['batches'].iterdir())]
jobs += [(f'spot {g}', EXTRA / 'spots' / f'spot{g}') for g in (0, 1)]
minutes, memory = zip(*[rss_and_minutes(folder) for _, folder in jobs])
bars = ax.bar(range(len(jobs)), minutes, color=[MUTED, MUTED] + [BLUE] * (len(jobs) - 4) + [GREEN, GREEN])
for bar, gib in zip(bars, memory):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .4, f'{gib:.0f}\nGiB', ha='center', fontsize=7.4,
            color=INK)
ax.set_xticks(range(len(jobs))); ax.set_xticklabels([name for name, _ in jobs], rotation=35, ha='right', fontsize=8)
ax.set_ylim(0, 32); ax.set_ylabel('minutes on 26 cores (labels: peak memory)')
style(ax, f'Every 4-row job: {min(minutes):.0f}-{max(minutes):.0f} min, <= {np.nanmax(memory):.0f} GiB '
          f'(64 GB requested)', grid='y')
finish(fig, 'fig4_production_path_end_to_end.png', 'Test 4: the production path works end to end',
       'Audits -> batch plan -> manage.py run (PBS, separate jobs) -> collect, all from the fresh clone. Rows 0-15 have '
       'the test\'s parameters with this dataset\'s seeds;\nrows 508-511 and 65532-65535 are new. Expected sigma: '
       'w4 (2 C_signal N + N^2) / (2l+1) per l for the row\'s own signal and the SO N_l, binned like the data.')
