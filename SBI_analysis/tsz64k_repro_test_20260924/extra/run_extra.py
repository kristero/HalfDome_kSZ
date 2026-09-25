"""Extra checks of the idark reproduction test (outside the bundle; nothing in it is modified).

  noise         draw 384 SO noise realizations with the engine's own code and test them
  repro-batch   rerun another batch of the 256-row test with its seeds and cache grids
  select-spots  pick four new audited rows by the test's rule and write two spot tasks
  spot          full-sky clean spectra of two rows with the audited cache and a 2x finer one
  los           independent SciPy line-of-sight integration vs the engine's direct columns
  report        collect every result into results/summary.json

Everything runs through the bundle's pipeline/common.py (same launcher, environment
scrubbing, frozen-source check) and a fresh GitHub clone of the bundle.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import subprocess
import sys

import numpy as np

EXTRA = Path(__file__).resolve().parent
ROOT = EXTRA.parent
BUNDLE = ROOT / 'HalfDome_kSZ' / 'tSZ_64k_flat_prior_SO_baseline_deproj0'
TEST = Path('/lustre/work/kristero10/tsz_diagnostic_256_accelerated_20260922')
sys.path.insert(0, str(BUNDLE / 'pipeline'))
import common  # noqa: E402
import manage  # noqa: E402

CFG = common.load_config(BUNDLE / 'pipeline' / 'config.toml')
WHERE = manage.layout(CFG)
ELL, DL = manage.ELL, manage.DL_FACTOR
HASH_TRAIN_ROWS = [0, 1, 2, 3, 65532, 65533, 65534, 65535]
HASH_TEST_ROWS = [0, 1, 2, 3]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_json(path, value)
    print(json.dumps(value, indent=1)[:4000])


# ----------------------------------------------------------------------------- noise

def cmd_noise(args):
    common.verify_sources()
    folder = EXTRA / 'noise_check'
    folder.mkdir(parents=True, exist_ok=True)
    _, theta, seeds = manage.load_design()
    test_seeds = np.load(common.REFERENCE / 'noise_seeds.npy')
    spec = dict(theta=theta[0].tolist(), train_rows=list(range(128)),
                train_seeds=[seeds[r].tolist() for r in range(128)], eng_rows=list(range(64)),
                eng_seeds=[test_seeds[r].tolist() for r in range(64)],
                hash_maps=[[f'train_{r}_{s + 1}', str(int(seeds[r][s]))] for r in HASH_TRAIN_ROWS for s in (0, 1)]
                + [[f'test_{r}_{s + 1}', str(int(test_seeds[r][s]))] for r in HASH_TEST_ROWS for s in (0, 1)])
    (folder / 'noise_spec.toml').write_text(''.join(f'{k} = {common.toml_value(v)}\n' for k, v in spec.items()))
    command = common.build_command(CFG, folder, args.threads)
    engine = str(common.ENGINE / 'engine.jl')
    assert command.count(engine) == 1
    command[command.index(engine)] = str(EXTRA / 'noise_check.jl')
    env, removed = common.build_environment(CFG, folder, args.threads)
    env['TSZ64K_ENGINE'] = str(common.ENGINE)
    common.atomic_json(folder / 'command.json', dict(argv=command, removed_environment=removed))
    with (folder / 'run.log').open('w') as log:
        code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    common.atomic_json(folder / 'status.json', dict(returncode=code, utc=common.now_utc()))
    if code:
        raise SystemExit(f'noise check failed ({code}); see {folder / "run.log"}')


# ----------------------------------------------------------------------------- same-seed reproduction

def compare_batch(task, folder, reference):
    """The acceptance of manage.py repro-check, for any batch of the 256-row test."""
    tolerance, passed = 1e-6, True
    timing = common.load_toml(folder / 'batch.toml')
    probe, ref_probe = common.load_toml(folder / 'operator_probe.toml'), common.load_toml(reference / 'operator_probe.toml')
    exact = ['julia_version', 'healpix_version', 'hdf5_version', 'xgpaint_sha256', 'operator_sha256',
             'adapter_sha256', 'beam_source_sha256', 'rng_uniform', 'rng_normal', 'mask_pixel_sha256']
    checks = dict(selected_halos=timing['selected_halos'],
                  operator_probe_exact={key: probe[key] == ref_probe[key] for key in exact},
                  operator_probe_bitwise_informational={key: probe[key] == ref_probe[key]
                                                        for key in ('noise_pixel_sha256', 'beam_pixel_sha256')})
    passed &= timing['selected_halos'] == common.SELECTED_HALOS and all(checks['operator_probe_exact'].values())
    rows = {}
    for case in task['cases']:
        new_dir, ref_dir = folder / case['label'], reference / case['label']
        observation, ref_observation = common.load_toml(new_dir / 'observation.toml'), common.load_toml(ref_dir / 'observation.toml')
        clean_ref = np.load(ref_dir / 'masked_clean_cl.npy')
        record = dict(nodes=case['nodes'], split_seeds=observation['split_seeds'],
                      split_seeds_equal=observation['split_seeds'] == ref_observation['split_seeds'],
                      mask_sha256_equal=observation['mask_sha256'] == common.MASK_SHA256 == ref_observation['mask_sha256'],
                      noise_maps_bitwise_equal=observation['noise_sha256'] == ref_observation['noise_sha256'])
        for name in ('masked_clean_cl.npy', 'unmasked_clean_cl.npy', 'masked_noisy_cross_cl.npy'):
            new, ref = np.load(new_dir / name), np.load(ref_dir / name)
            record[name] = manage.spectrum_metrics(new, ref, clean_ref if name != 'unmasked_clean_cl.npy' else ref)
        record['passed'] = bool(record['split_seeds_equal'] and record['mask_sha256_equal'] and
                                all(record[n] < tolerance for n in ('masked_clean_cl.npy', 'unmasked_clean_cl.npy',
                                                                    'masked_noisy_cross_cl.npy')))
        passed &= record['passed']
        rows[case['label']] = record
    return dict(passed=bool(passed), tolerance=tolerance, checks=checks, rows=rows, utc=common.now_utc())


def cmd_repro_batch(args):
    batch = int(args.arg)
    reference = TEST / 'diagnostic_256' / 'batches' / f'{batch:03d}'
    task = json.loads((reference / 'request.json').read_text())
    folder = EXTRA / 'repro_batches' / f'batch{batch:03d}'
    print(common.launch(task, folder, args.threads, CFG), flush=True)
    write_json(EXTRA / 'results' / f'repro_batch{batch:03d}.json', compare_batch(task, folder, reference))


# ----------------------------------------------------------------------------- spots

def audited_new_rows():
    _, theta, _ = manage.load_design()
    rows, unresolved = [], {}
    for task_folder in sorted(WHERE['audits'].glob('[0-9]*')):
        for folder in sorted(p for p in task_folder.glob('[0-9]*') if p.is_dir()):
            row = int(folder.name)
            result, reason = manage.audit_row(folder)
            if result is None:
                unresolved[row] = reason
                continue
            probes = np.loadtxt(folder / f"probes_{result['nodes'][0]}.csv", delimiter=',')
            rows.append(dict(row=row, nodes=result['nodes'], error=result['max_relative_visible'],
                             norm=float(np.sqrt(np.mean(probes[:, 3] ** 2))), theta=theta[row].tolist()))
    return rows, unresolved


def cmd_select_spots(args):
    """The test's rule: two largest post-refinement probe errors plus low/high direct-column norm."""
    rows, _ = audited_new_rows()
    rows = [r for r in rows if r['row'] >= 256]
    by_error = sorted(rows, key=lambda r: -r['error'])[:2]
    rest = [r for r in rows if r not in by_error]
    chosen = by_error + [min(rest, key=lambda r: r['norm']), max(rest, key=lambda r: r['norm'])]
    for group in (0, 1):
        cases = []
        for r in chosen[2 * group:2 * group + 2]:
            cases.append(dict(label=f"{r['row']:05d}_candidate", row=r['row'], theta=r['theta'], nodes=r['nodes']))
            cases.append(dict(label=f"{r['row']:05d}_reference", row=r['row'], theta=r['theta'],
                              nodes=[2 * n for n in r['nodes']]))
        write_json(EXTRA / 'spots' / f'spot{group}.json', dict(mode='maps', cases=cases, output_nsides=[4096]))
    write_json(EXTRA / 'spots' / 'selection.json',
               dict(rule='two largest post-refinement probe errors plus low/high direct-column norm (new rows)',
                    rows=[{k: v for k, v in r.items() if k != 'theta'} for r in chosen]))


def cmd_spot(args):
    group = int(args.arg)
    task = json.loads((EXTRA / 'spots' / f'spot{group}.json').read_text())
    folder = EXTRA / 'spots' / f'spot{group}'
    print(common.launch(task, folder, args.threads, CFG), flush=True)
    results = []
    for candidate, reference in zip(task['cases'][::2], task['cases'][1::2]):
        record = dict(row=candidate['row'], nodes=candidate['nodes'], reference_nodes=reference['nodes'])
        for kind in ('masked_clean_cl.npy', 'unmasked_clean_cl.npy'):
            c = np.load(folder / candidate['label'] / kind)[ELL] * DL
            r = np.load(folder / reference['label'] / kind)[ELL] * DL
            assert np.all(r > 0)
            record[kind] = dict(max_fractional_Dell=float(np.max(np.abs(c / r - 1))),
                                relative_Dell_l2=float(np.linalg.norm(c - r) / np.linalg.norm(r)))
        record['passed'] = record['masked_clean_cl.npy']['max_fractional_Dell'] < .01
        results.append(record)
    write_json(EXTRA / 'results' / f'spot{group}.json', dict(target='< 1% per-ell clean D_ell, l = 80..7979',
                                                            passed=all(r['passed'] for r in results), rows=results))


# ----------------------------------------------------------------------------- independent LOS

def direct_column(x, xc, beta, outer=4.):
    """Verbatim from independent_los.py of the 256-row test."""
    from scipy.integrate import quad
    end = np.sqrt((outer - x) * (outer + x))
    if x == 0:
        scale = 2 * xc ** .3 * end ** .7 / .7
        core = min(1., (xc / max(beta, 1.) / end) ** .7)
        breaks = np.unique(np.r_[0., core * np.geomspace(1e-8, 1, 10), np.geomspace(core, 1., 20), 1.])
        f = lambda t: np.exp(-beta * np.log1p(end * t ** (1 / .7) / xc))  # noqa: E731
        return scale * sum(quad(f, a, b, epsabs=1e-13, epsrel=1e-10)[0] for a, b in zip(breaks[:-1], breaks[1:]))
    peak = -.3 * np.log(x / xc) - beta * np.log1p(x / xc)
    core = min(end, np.sqrt(x * (x + xc) / max(beta, 1.)))
    breaks = np.unique(np.r_[0., np.geomspace(max(core * 1e-5, 1e-200), end, 32)])

    def integrand(distance):
        radius = np.hypot(x, distance)
        return np.exp(-.3 * np.log(radius / xc) - beta * np.log1p(radius / xc) - peak)
    value = sum(quad(integrand, a, b, epsabs=1e-13, epsrel=1e-10)[0] for a, b in zip(breaks[:-1], breaks[1:]))
    return np.exp(np.log(2 * value) + peak) if value > 0 else 0.


def los_row(item):
    """The test's 4 halos x 4 radii, plus all 64 probe halos at 3 radii >= 0.015 R200."""
    row, folder, p, nodes = item
    probes = np.loadtxt(Path(folder) / f'probes_{nodes[0]}.csv', delimiter=',').reshape(64, 39, 6)
    out = []
    for index in range(64):
        halo = probes[index]
        mass, z = 10. ** halo[0, 0], halo[0, 1]
        xc = p[1] * (mass / 1e14) ** p[4] * (1 + z) ** p[7]
        beta = p[2] * (mass / 1e14) ** p[5] * (1 + z) ** p[8]
        test_set = index in (0, 21, 42, 63)
        samples = (14, 24, 35, 38) if test_set else (24, 35, 38)
        central = direct_column(0, xc, beta)
        for sample in samples:
            independent = direct_column(halo[sample, 2], xc, beta) / central
            julia = halo[sample, 3] / halo[sample, 5]
            out.append((row, index, sample, test_set, independent, julia))
    return out


def cmd_los(args):
    _, theta, _ = manage.load_design()
    items = []
    for task_folder in sorted(WHERE['audits'].glob('[0-9]*')):
        for folder in sorted(p for p in task_folder.glob('[0-9]*') if p.is_dir()):
            result, _ = manage.audit_row(folder)
            if result is not None:
                items.append((int(folder.name), str(folder), theta[int(folder.name)].tolist(), result['nodes']))
    with Pool(args.threads) as pool:
        records = [r for chunk in pool.map(los_row, items, chunksize=4) for r in chunk]
    array = np.array([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in records], dtype=float)
    np.save(EXTRA / 'results' / 'los_records.npy', array)

    def stats(mask):
        independent, julia = array[mask, 4], array[mask, 5]
        visible = julia > 1e-10
        return dict(points=int(mask.sum()), rows=int(len(np.unique(array[mask, 0]))),
                    max_relative_visible=float(np.max(np.abs(independent[visible] / julia[visible] - 1))),
                    max_absolute_ratio_error=float(np.max(np.abs(independent - julia))))
    new, test_set = array[:, 0] >= 256, array[:, 3] == 1
    result = dict(scope='Normalized spherical column shapes; independent SciPy direct-distance integration '
                        '(independent_los.py of the 256-row test)',
                  test_selection_rows_0_255=stats(~new & test_set), test_selection_new_rows=stats(new & test_set),
                  all_64_halos_new_rows=stats(new), all_64_halos_rows_0_255=stats(~new))
    result['passed'] = all(v['max_relative_visible'] < 1e-7 and v['max_absolute_ratio_error'] < 1e-8
                           for k, v in result.items() if isinstance(v, dict))
    write_json(EXTRA / 'results' / 'independent_los.json', result)


def cmd_profile_checks(args):
    """One job instead of three (keeps the cluster load low): LOS check, then both spot groups."""
    cmd_los(args)
    for group in ('0', '1'):
        args.arg = group
        cmd_spot(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['noise', 'repro-batch', 'select-spots', 'spot', 'los', 'profile-checks'])
    parser.add_argument('--arg')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    dict(noise=cmd_noise, **{'repro-batch': cmd_repro_batch, 'select-spots': cmd_select_spots,
                             'profile-checks': cmd_profile_checks},
         spot=cmd_spot, los=cmd_los)[args.command](args)


if __name__ == '__main__':
    main()
