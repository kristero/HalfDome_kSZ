"""Summarize the idark reproduction test into results/summary.json (run on the login node)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

EXTRA = Path(__file__).resolve().parent
sys.path.insert(0, str(EXTRA))
from run_extra import CFG, DL, ELL, HASH_TEST_ROWS, HASH_TRAIN_ROWS, TEST, WHERE, common, manage  # noqa: E402

load_json = lambda path: json.loads(Path(path).read_text())  # noqa: E731
SPECTRA = ('masked_clean_cl.npy', 'unmasked_clean_cl.npy', 'masked_noisy_cross_cl.npy')


def test_row(row):
    return TEST / 'diagnostic_256' / 'batches' / f'{row // 4:03d}' / f'{row:05d}'


def worst(result):
    return max(max(v[name] for name in SPECTRA) for v in result['rows'].values())


def peak_rss_gib(folder):
    text = (folder / 'time.txt').read_text() if (folder / 'time.txt').exists() else ''
    match = re.search(r'Maximum resident set size \(kbytes\): (\d+)', text)
    return round(int(match.group(1)) / 2 ** 20, 1) if match else None


summary = {}
# ------------------------------------------------------------------ 1. code, same seeds as the test
repro = load_json(WHERE['repro'] / 'repro_result.json')
rep040 = load_json(EXTRA / 'results' / 'repro_batch040.json')
compare = load_json(WHERE['root'] / 'compare_test256.json')
rows_done = sorted(int(p.name) for p in WHERE['rows'].iterdir())
first = [r for r in rows_done if r < 256]
clean_full = {}
for name in SPECTRA[:2]:
    clean_full[name] = max(float(np.max(np.abs(np.load(WHERE['rows'] / f'{r:05d}' / name)[ELL] /
                                               np.load(test_row(r) / name)[ELL] - 1))) for r in first)
summary['code_same_seeds'] = dict(
    rows_0_3=dict(passed=repro['passed'], worst_spectrum_metric=worst(repro),
                  rng_fingerprint_identical=repro['checks']['operator_probe_exact']['rng_uniform']
                  and repro['checks']['operator_probe_exact']['rng_normal'],
                  noise_maps_bitwise_identical=all(v['noise_sha256_bitwise_equal_informational']
                                                   for v in repro['rows'].values())),
    rows_160_163=dict(passed=rep040['passed'], worst_spectrum_metric=worst(rep040),
                      cache_grids=['x'.join(map(str, v['nodes'])) for v in rep040['rows'].values()],
                      noise_maps_bitwise_identical=all(v['noise_maps_bitwise_equal'] for v in rep040['rows'].values())),
    compare_test256=compare,
    production_rows_with_test_parameters=dict(
        rows=first, note='new SO noise seeds; clean spectra must equal the test at every ell',
        max_relative_clean_difference_all_ell=clean_full))

# ------------------------------------------------------------------ 2. noise seeds
_, theta, seeds = manage.load_design()
test_seeds = np.load(common.REFERENCE / 'noise_seeds.npy')
check = common.load_toml(EXTRA / 'noise_check' / 'noise_check.toml')
hashes = check['map_pixel_sha256']
train_match = {r: common.load_toml(WHERE['rows'] / f'{r:05d}' / 'observation.toml')['noise_sha256']
               == [hashes[f'train_{r}_1'], hashes[f'train_{r}_2']] for r in HASH_TRAIN_ROWS
               if (WHERE['rows'] / f'{r:05d}').exists()}
test_match = {r: common.load_toml(common.REFERENCE / 'batch000' / f'{r:05d}' / 'observation.toml')['noise_sha256']
              == [hashes[f'test_{r}_1'], hashes[f'test_{r}_2']] for r in HASH_TEST_ROWS}
# Residuals normalized by their analytic expected scatter (residual_check.py). A pooled
# per-bin scale is not enough: each row's own signal sets its signal x noise variance.
residuals = load_json(EXTRA / 'results' / 'residual_check.json')
all_seeds = seeds.ravel()
summary['noise_seeds'] = dict(
    design=dict(splits=int(all_seeds.size), unique=int(np.unique(all_seeds).size),
                overlap_with_test_seeds=int(np.intersect1d(all_seeds, test_seeds.ravel()).size),
                rows_with_equal_splits=int(np.sum(seeds[:, 0] == seeds[:, 1]))),
    realizations_checked=check['realizations'], auto_power_z=check['auto_power_z'],
    auto_power_ratio_mean_per_bin_range=[min(check['auto_power_ratio_mean_per_bin']),
                                         max(check['auto_power_ratio_mean_per_bin'])],
    cross_pairs_full_ell=check['pairs'], all_pairs_low_ell=check['all_pairs_low_ell'],
    same_seed_twice=dict(bitwise_identical=check['same_seed_twice_bitwise_identical'],
                         min_z=check['same_seed_twice_min_z']),
    check_maps_equal_production_maps=train_match, check_maps_equal_test_maps=test_match,
    production_residuals=residuals)

# ------------------------------------------------------------------ 3. profiles
tasks = sorted(int(p.name) for p in WHERE['audits'].glob('[0-9]*'))
rows = [r for t in tasks for r in range(256 * t, 256 * (t + 1))]
results, unresolved = manage.collect_audits(CFG, theta, rows)


def grids(selection):
    counts = {}
    for r in selection:
        key = 'x'.join(map(str, results[r]['nodes']))
        counts[key] = counts.get(key, 0) + 1
    return counts


new_rows = [r for r in results if r >= 256]
summary['profiles'] = dict(
    audit_tasks=tasks, rows_audited=len(rows), resolved=len(results), unresolved=unresolved,
    cache_grids_rows_0_255=grids([r for r in results if r < 256]), cache_grids_new_rows=grids(new_rows),
    max_relative_visible_new_rows=max(results[r]['max_relative_visible'] for r in new_rows),
    max_absolute_over_central_new_rows=max(results[r]['max_absolute_over_central'] for r in new_rows),
    max_area_weighted_L1_new_rows=max(results[r]['max_sampled_radial_L1_relative_error'] for r in new_rows),
    independent_los=load_json(EXTRA / 'results' / 'independent_los.json'),
    spots=[load_json(EXTRA / 'results' / f'spot{g}.json') for g in (0, 1)])

# ------------------------------------------------------------------ 4. production runs and collection
batches = {}
for folder in sorted(WHERE['batches'].iterdir()):
    status = load_json(folder / 'status.json')
    request = load_json(folder / 'request.json')
    batches[folder.name] = dict(returncode=status['returncode'], minutes=round(status['seconds'] / 60, 1),
                                peak_rss_gib=peak_rss_gib(folder), host=status['host'],
                                rows=[c['row'] for c in request['cases']],
                                grids=['x'.join(map(str, c['nodes'])) for c in request['cases']])
meta_path = WHERE['dataset'] / f'rows_{len(first)}' / 'meta.json'
summary['production'] = dict(batches=batches, collect=load_json(meta_path) if meta_path.exists() else None)
common.atomic_json(EXTRA / 'results' / 'summary.json', summary)
print(json.dumps(summary, indent=1, default=str)[:12000])
