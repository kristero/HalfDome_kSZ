"""Produce the 65,536-row flat-prior tSZ dataset with SO baseline Deproj-0 noise.

Commands, in the order they are normally used (see README.md):

  repro-check      rerun rows 0-3 of the validated 256-row test and compare
  prepare-audits   write the per-row cache-accuracy audit tasks
  audit            run one audit task (one scheduler array element)
  prepare-batches  check every audit and write the batch plan (4 rows per batch)
  run              run production batches (static array range, list, or pool)
  status           progress, failures, claims, timing
  reset            set a failed or interrupted batch/audit aside so it can rerun
  compare-test256  compare rows 0-255 with the 256-row test (same parameters)
  collect          verify every row and assemble the dataset arrays

No command redraws, drops or replaces a parameter row. Failed work is kept
for inspection and only rerun after an explicit reset.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUNDLE, DESIGN, MASK_SHA256, REFERENCE, ROW_FILES, SELECTED_HALOS,  # noqa: E402
                    PreviousFailure, TaskFailed, atomic_json, launch, load_config, load_toml,
                    now_utc, sha256_file, source_manifest, verify_sources)

ELL = np.arange(80, 7980)
DL_FACTOR = ELL * (ELL + 1) / (2 * np.pi)
trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def rebin_unbinned(values, ell):
    """40 bins of 200 multipoles (last to 7979), weights 2l+1: same as the 256-row test."""
    edges = np.r_[np.arange(80, 7881, 200), 7980]
    return np.stack([np.average(values[:, (ell >= a) & (ell < b)], axis=1,
                                weights=2 * ell[(ell >= a) & (ell < b)] + 1)
                     for a, b in zip(edges[:-1], edges[1:])], axis=1)


# ----------------------------------------------------------------------------- layout

def layout(cfg):
    root = cfg['run_root']
    return dict(root=root, tasks=root / 'tasks', audits=root / 'audits', batches=root / 'batches',
                rows=root / 'rows', plan=root / 'batch_plan.json', audit_summary=root / 'audit_summary.json',
                repro=root / 'repro', attempts=root / 'attempts', dataset=root / 'dataset',
                logs=root / 'logs', manifest=root / 'run_manifest.json')


def load_design():
    manifest = json.loads((DESIGN / 'design_manifest.json').read_text())
    theta = np.load(DESIGN / manifest['files']['theta_design'])
    seeds = np.load(DESIGN / manifest['files']['noise_seeds'])
    assert theta.shape == (manifest['count'], 9) and seeds.shape == (manifest['count'], 2)
    return manifest, theta, seeds


def bundle_git_state():
    try:
        commit = subprocess.run(['git', '-C', str(BUNDLE), 'rev-parse', 'HEAD'], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(BUNDLE), 'status', '--porcelain', '--', '.'],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.strip()
        return dict(commit=commit, uncommitted_changes=bool(dirty))
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, uncommitted_changes=None)


def ensure_run_root(cfg):
    """Create run_root once; refuse to mix designs or source versions in it."""
    where = layout(cfg)
    where['logs'].mkdir(parents=True, exist_ok=True)
    design = json.loads((DESIGN / 'design_manifest.json').read_text())
    record = dict(count=design['count'], design_sha256=design['sha256'],
                  frozen_sha256=source_manifest()['frozen_sha256'])
    if where['manifest'].exists():
        previous = json.loads(where['manifest'].read_text())
        if any(previous[key] != record[key] for key in record):
            raise SystemExit(f"{where['root']} was started with a different design or source set; "
                             'use a fresh run_root')
        return previous
    record.update(created_utc=now_utc(), bundle_git=bundle_git_state(), config=str(cfg['config_path']))
    atomic_json(where['manifest'], record)
    return record


def write_once(path, value):
    """Write a JSON plan file, or confirm an identical one already exists."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise SystemExit(f'{path} exists with different content; inspect it rather than replace it')
        return False
    atomic_json(path, value)
    return True


# ----------------------------------------------------------------------------- audits

def audit_task(cfg, index, theta):
    size = cfg['audit_rows_per_task']
    rows = range(index * size, min((index + 1) * size, len(theta)))
    if not rows:
        raise SystemExit(f'Audit task {index} is outside the design')
    return dict(mode='audit', cases=[dict(label=f'{row:05d}', row=row, theta=theta[row].tolist())
                                     for row in rows])


def audit_task_count(cfg, count):
    return math.ceil(count / cfg['audit_rows_per_task'])


def cmd_prepare_audits(cfg, args):
    verify_sources()
    ensure_run_root(cfg)
    _, theta, _ = load_design()
    where = layout(cfg)
    where['tasks'].mkdir(exist_ok=True)
    total = audit_task_count(cfg, len(theta))
    written = sum(write_once(where['tasks'] / f'audit_{index:04d}.json', audit_task(cfg, index, theta))
                  for index in range(total))
    print(f'{total} audit tasks of {cfg["audit_rows_per_task"]} rows in {where["tasks"]} ({written} new).')
    print(f'Array range for the audit stage: 0-{total - 1}')


def cmd_audit(cfg, args):
    _, theta, _ = load_design()
    where = layout(cfg)
    path = where['tasks'] / f'audit_{args.task:04d}.json'
    if not path.exists():
        raise SystemExit(f'{path} is missing: run prepare-audits first')
    task = json.loads(path.read_text())
    if task != audit_task(cfg, args.task, theta):
        raise SystemExit(f'{path} does not match the design and audit_rows_per_task in the config')
    threads = args.threads or cfg['audit_threads']
    result = launch(task, where['audits'] / f'{args.task:04d}', threads, cfg, dry_run=args.dry_run)
    print(json.dumps(result, indent=2) if args.dry_run else f'audit task {args.task}: {result}')


def audit_row(folder):
    """Accepted cache nodes for one row, or the reason it is unresolved.

    Same acceptance as the 256-row test: the engine's direct LOS probe
    (<=0.4% visible relative and absolute/central error) and, from the saved
    probes, an area-weighted radial L1 error <= 0.4% for every probed halo.
    """
    path = folder / 'audit.toml'
    if not path.exists():
        return None, 'no audit.toml'
    record = load_toml(path)
    last = record['attempts'][-1]
    if not record['accepted']:
        return None, 'LOS accuracy target not met up to 1024x512x256 cache nodes'
    nodes = [int(value) for value in last['nodes']]
    probes = np.loadtxt(folder / f'probes_{nodes[0]}.csv', delimiter=',').reshape(64, 39, 6)
    errors = []
    for halo in probes:
        x, direct, estimate = halo[:, 2], halo[:, 3], halo[:, 4]
        error = trapezoid(abs(estimate - direct) * x, x) / trapezoid(direct * x, x)
        if not np.isfinite(error) or error > .004:
            return None, f'area-weighted radial cache error {error:.4g} > 0.004'
        errors.append(float(error))
    return dict(nodes=nodes, attempts=len(record['attempts']),
                max_relative_visible=float(last['max_relative_visible']),
                max_absolute_over_central=float(last['max_absolute_over_central']),
                relative_l2=float(last['relative_l2']),
                max_sampled_radial_L1_relative_error=max(errors)), None


def collect_audits(cfg, theta, rows=None):
    where = layout(cfg)
    size = cfg['audit_rows_per_task']
    results, unresolved, checked_tasks = {}, {}, {}
    for row in (range(len(theta)) if rows is None else rows):
        index = row // size
        if index not in checked_tasks:
            folder = where['audits'] / f'{index:04d}'
            status = folder / 'status.json'
            ok = status.exists() and json.loads(status.read_text())['returncode'] == 0
            if ok:
                request = json.loads((folder / 'request.json').read_text())
                ok = request == audit_task(cfg, index, theta)
            checked_tasks[index] = ok
        if not checked_tasks[index]:
            unresolved[row] = 'audit task not completed successfully'
            continue
        result, reason = audit_row(where['audits'] / f'{index:04d}' / f'{row:05d}')
        if result is None:
            unresolved[row] = reason
        else:
            results[row] = result
    return results, unresolved


def cmd_prepare_batches(cfg, args):
    verify_sources()
    ensure_run_root(cfg)
    manifest, theta, seeds = load_design()
    where = layout(cfg)
    results, unresolved = collect_audits(cfg, theta)
    nodes = {}
    for result in results.values():
        key = 'x'.join(map(str, result['nodes']))
        nodes[key] = nodes.get(key, 0) + 1
    summary = dict(rows=len(theta), resolved=len(results), cache_nodes=nodes,
                   unresolved={str(row): reason for row, reason in sorted(unresolved.items())},
                   max_relative_visible=max((r['max_relative_visible'] for r in results.values()), default=None),
                   max_sampled_radial_L1_relative_error=max(
                       (r['max_sampled_radial_L1_relative_error'] for r in results.values()), default=None))
    atomic_json(where['audit_summary'], summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'unresolved'}, indent=2))
    if unresolved:
        print(f'{len(unresolved)} rows unresolved; first ten:', dict(list(sorted(unresolved.items()))[:10]))
        if not args.defer_unresolved:
            raise SystemExit('Not writing a batch plan. Finish or reset the audits; rows that fail the '
                             'accuracy target need a decision from the dataset owner (see README). '
                             '--defer-unresolved plans the resolved rows and lists the rest.')
    ordered = sorted(results)
    size = cfg['batch_size']
    batches = []
    for start in range(0, len(ordered), size):
        cases = [dict(label=f'{row:05d}', row=row, theta=theta[row].tolist(), nodes=results[row]['nodes'],
                      seeds=[int(value) for value in seeds[row]]) for row in ordered[start:start + size]]
        batches.append(dict(id=len(batches), mode='maps', cases=cases, output_nsides=[4096]))
    plan = dict(batch_size=size, rows=len(ordered), deferred_rows=sorted(unresolved),
                design_sha256=manifest['sha256'], batches=batches)
    write_once(where['plan'], plan)
    print(f'{len(batches)} batches of up to {size} rows in {where["plan"]}.')
    print(f'Array range with --batches-per-task K: 0-{math.ceil(len(batches) / args.batches_per_task) - 1} '
          f'for K={args.batches_per_task}')


# ----------------------------------------------------------------------------- production

def load_plan(cfg):
    path = layout(cfg)['plan']
    if not path.exists():
        raise SystemExit(f'{path} is missing: run prepare-batches first')
    return json.loads(path.read_text())


def batch_folder(cfg, batch_id):
    return layout(cfg)['batches'] / f'{batch_id:05d}'


def finalize_rows(cfg, task, folder):
    """Checksum each row and link it into run_root/rows (idempotent)."""
    rows = layout(cfg)['rows']
    rows.mkdir(exist_ok=True)
    batch_status = json.loads((folder / 'status.json').read_text())
    timing = load_toml(folder / 'batch.toml')
    if timing['selected_halos'] != SELECTED_HALOS:
        raise RuntimeError(f'{folder}: selected {timing["selected_halos"]} halos, expected {SELECTED_HALOS}')
    for case in task['cases']:
        source = folder / case['label']
        if not (source / 'status.json').exists():
            atomic_json(source / 'status.json', dict(
                row=case['row'], returncode=0, batch=task['id'], theta=case['theta'], nodes=case['nodes'],
                split_seeds=case['seeds'], selected_halos=timing['selected_halos'], batch_status=batch_status,
                sha256={name: sha256_file(source / name) for name in ROW_FILES}))
        link = rows / case['label']
        try:
            link.symlink_to(os.path.relpath(source, rows), target_is_directory=True)
        except FileExistsError:
            if link.resolve() != source.resolve():
                raise RuntimeError(f'Row path collision: {link}')


def repro_passed(cfg):
    path = layout(cfg)['repro'] / 'repro_result.json'
    return path.exists() and json.loads(path.read_text()).get('passed') is True


def selected_batches(args, count):
    if args.batches:
        ids = sorted({int(value) for value in args.batches.split(',') if value.strip()})
    elif args.array_index is not None:
        start = args.array_index * args.batches_per_task
        ids = list(range(start, min(start + args.batches_per_task, count)))
    elif args.pool:
        ids = list(range(count))
    else:
        raise SystemExit('Choose --array-index, --batches or --pool')
    outside = [value for value in ids if not 0 <= value < count]
    if outside:
        raise SystemExit(f'Batch ids outside 0..{count - 1}: {outside[:5]}')
    return ids


def cmd_run(cfg, args):
    plan = load_plan(cfg)
    if not args.dry_run and not args.skip_repro_gate and not repro_passed(cfg):
        raise SystemExit('The reproduction check has not passed in this run_root (manage.py repro-check). '
                         'Use --skip-repro-gate only if you know why.')
    ids = selected_batches(args, len(plan['batches']))
    threads = args.threads or cfg['threads']
    started, completed, failures = time.monotonic(), 0, []
    for batch_id in ids:
        if args.stop_after_hours and (time.monotonic() - started) / 3600 > args.stop_after_hours:
            print(f'Stopping before batch {batch_id}: --stop-after-hours reached', flush=True)
            break
        if args.pool and args.max_batches and completed >= args.max_batches:
            break
        task, folder = plan['batches'][batch_id], batch_folder(cfg, batch_id)
        if args.pool and ((folder / 'claim').exists() or (folder / 'status.json').exists()):
            status = folder / 'status.json'
            if status.exists() and json.loads(status.read_text())['returncode'] == 0:
                finalize_rows(cfg, task, folder)
            continue
        try:
            result = launch(task, folder, threads, cfg, dry_run=args.dry_run)
        except FileExistsError:
            print(f'batch {batch_id}: claimed by another worker, skipped', flush=True)
            continue
        except (PreviousFailure, TaskFailed) as error:
            print(f'batch {batch_id}: {error}', flush=True)
            failures.append(batch_id)
            continue
        if args.dry_run:
            print(json.dumps(dict(batch=batch_id, **result), indent=2) if isinstance(result, dict)
                  else f'batch {batch_id}: {result}')
            break
        finalize_rows(cfg, task, folder)
        completed += result == 'ran'
        print(f'batch {batch_id}: {result} ({time.monotonic() - started:.0f} s elapsed)', flush=True)
    if failures:
        raise SystemExit(f'Failed batches: {failures}')


# ----------------------------------------------------------------------------- status and reset

def folder_state(folder):
    status = folder / 'status.json'
    if status.exists():
        code = json.loads(status.read_text())['returncode']
        return ('done' if code == 0 else 'failed'), code
    if (folder / 'claim').exists():
        owner = folder / 'claim' / 'owner.json'
        return 'claimed', json.loads(owner.read_text()) if owner.exists() else {}
    if (folder / 'request.json').exists():
        return 'interrupted', None
    return 'pending', None


def cmd_status(cfg, args):
    where = layout(cfg)
    _, theta, _ = load_design()
    report = dict(run_root=str(where['root']))
    audits = {}
    for index in range(audit_task_count(cfg, len(theta))):
        state, _ = folder_state(where['audits'] / f'{index:04d}')
        audits.setdefault(state, []).append(index)
    report['audit_tasks'] = {state: len(ids) for state, ids in audits.items()}
    for state in ('failed', 'claimed', 'interrupted'):
        if audits.get(state):
            report[f'audit_tasks_{state}'] = audits[state][:50]
    if where['plan'].exists():
        plan = load_plan(cfg)
        states, seconds, claims = {}, [], {}
        for task in plan['batches']:
            folder = batch_folder(cfg, task['id'])
            state, detail = folder_state(folder)
            states.setdefault(state, []).append(task['id'])
            if state == 'done':
                seconds.append(json.loads((folder / 'status.json').read_text())['seconds'])
            elif state == 'claimed':
                claims[task['id']] = detail
        pending = len(states.get('pending', [])) + len(states.get('interrupted', [])) + len(states.get('failed', []))
        report['batches'] = {state: len(ids) for state, ids in states.items()}
        report['rows_in_plan'] = plan['rows']
        report['deferred_rows'] = len(plan['deferred_rows'])
        report['rows_done'] = sum(len(plan['batches'][i]['cases']) for i in states.get('done', []))
        for state in ('failed', 'interrupted'):
            if states.get(state):
                report[f'batches_{state}'] = states[state][:50]
        if claims:
            report['claimed_batches'] = {str(key): value for key, value in list(claims.items())[:50]}
        if seconds:
            median = float(np.median(seconds))
            report['median_batch_seconds'] = median
            report['remaining_batch_hours'] = round(median * pending / 3600, 1)
    report['repro_check_passed'] = repro_passed(cfg)
    print(json.dumps(report, indent=2))


def set_aside(cfg, folder, label):
    state, _ = folder_state(folder)
    if state == 'done':
        raise SystemExit(f'{folder} completed successfully; refusing to reset it')
    if state == 'pending':
        return False
    target = layout(cfg)['attempts'] / f'{label}_{now_utc().replace(":", "").replace("+", "_")}'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(folder), str(target))
    print(f'{folder} ({state}) moved to {target}')
    return True


def cmd_reset(cfg, args):
    where = layout(cfg)
    targets = []
    if args.batches:
        targets += [('batch', int(value)) for value in args.batches.split(',') if value.strip()]
    if args.audit_tasks:
        targets += [('audit', int(value)) for value in args.audit_tasks.split(',') if value.strip()]
    if args.all_failed or args.all_interrupted or args.stale_claims_hours is not None:
        wanted = set()
        if args.all_failed:
            wanted.add('failed')
        if args.all_interrupted:
            wanted.add('interrupted')
        groups = [('audit', where['audits'])] + ([('batch', where['batches'])] if where['plan'].exists() else [])
        for kind, parent in groups:
            if not parent.exists():
                continue
            for folder in sorted(parent.iterdir()):
                if not (folder.is_dir() and folder.name.isdigit()):
                    continue
                state, _ = folder_state(folder)
                stale = (state == 'claimed' and args.stale_claims_hours is not None and
                         time.time() - (folder / 'claim').stat().st_mtime > 3600 * args.stale_claims_hours)
                if state in wanted or stale:
                    targets.append((kind, int(folder.name)))
    if not targets:
        raise SystemExit('Nothing selected (use --batches, --audit-tasks, --all-failed, '
                         '--all-interrupted or --stale-claims-hours)')
    for kind, number in targets:
        folder = batch_folder(cfg, number) if kind == 'batch' else where['audits'] / f'{number:04d}'
        if (folder / 'claim').exists() and args.stale_claims_hours is None and not args.force:
            raise SystemExit(f'{folder} is claimed by a (possibly running) worker; use --force if it is dead')
        set_aside(cfg, folder, f'{kind}_{number:05d}')


# ----------------------------------------------------------------------------- checks against the test

def spectrum_metrics(new, ref, clean_ref):
    """Largest per-ell difference over ell=80..7979, relative to the signal scale."""
    scale = np.maximum(np.abs(ref[ELL]) + np.abs(clean_ref[ELL]), np.finfo(float).tiny)
    return float(np.max(np.abs(new[ELL] - ref[ELL]) / scale))


def cmd_repro_check(cfg, args):
    """Rerun batch 000 of the 256-row test (rows 0-3, its seeds and cache nodes) and compare."""
    where = layout(cfg)
    ensure_run_root(cfg)
    task = json.loads((REFERENCE / 'batch000' / 'task.json').read_text())
    folder = where['repro'] / 'batch000'
    if args.run:
        result = launch(task, folder, args.threads or cfg['threads'], cfg, dry_run=args.dry_run)
        if args.dry_run:
            print(json.dumps(result, indent=2))
            return
    if not (folder / 'status.json').exists():
        raise SystemExit(f'No reproduction run in {folder}; use --run')
    if json.loads((folder / 'status.json').read_text())['returncode'] != 0:
        raise SystemExit(f'The reproduction run failed; see {folder / "run.log"}')
    tolerance, checks, passed = 1e-6, {}, True
    timing = load_toml(folder / 'batch.toml')
    checks['selected_halos'] = timing['selected_halos']
    passed &= timing['selected_halos'] == SELECTED_HALOS
    probe, reference_probe = load_toml(folder / 'operator_probe.toml'), load_toml(REFERENCE / 'batch000' / 'operator_probe.toml')
    exact = ['julia_version', 'healpix_version', 'hdf5_version', 'xgpaint_sha256', 'operator_sha256',
             'adapter_sha256', 'beam_source_sha256', 'rng_uniform', 'rng_normal', 'mask_pixel_sha256']
    checks['operator_probe_exact'] = {key: probe[key] == reference_probe[key] for key in exact}
    passed &= all(checks['operator_probe_exact'].values())
    # libsharp may pick a different SIMD code path on other CPUs: report only.
    checks['operator_probe_bitwise_informational'] = {key: probe[key] == reference_probe[key]
                                                      for key in ('noise_pixel_sha256', 'beam_pixel_sha256')}
    rows = {}
    for case in task['cases']:
        new_dir, ref_dir = folder / case['label'], REFERENCE / 'batch000' / case['label']
        observation, reference_observation = load_toml(new_dir / 'observation.toml'), load_toml(ref_dir / 'observation.toml')
        clean_ref = np.load(ref_dir / 'masked_clean_cl.npy')
        record = dict(mask_sha256_equal=observation['mask_sha256'] == MASK_SHA256 == reference_observation['mask_sha256'],
                      split_seeds_equal=observation['split_seeds'] == reference_observation['split_seeds'],
                      noise_sha256_bitwise_equal_informational=observation['noise_sha256'] == reference_observation['noise_sha256'])
        for name in ('masked_clean_cl.npy', 'unmasked_clean_cl.npy', 'masked_noisy_cross_cl.npy'):
            new, ref = np.load(new_dir / name), np.load(ref_dir / name)
            record[name] = spectrum_metrics(new, ref, clean_ref if name != 'unmasked_clean_cl.npy' else ref)
        record['passed'] = bool(record['mask_sha256_equal'] and record['split_seeds_equal'] and
                                all(record[name] < tolerance for name in
                                    ('masked_clean_cl.npy', 'unmasked_clean_cl.npy', 'masked_noisy_cross_cl.npy')))
        passed &= record['passed']
        rows[case['label']] = record
    result = dict(passed=bool(passed), tolerance=tolerance, checks=checks, rows=rows, utc=now_utc(),
                  metric='max over ell=80..7979 of |new-ref| / (|ref| + masked clean ref)',
                  note='Differences at the 1e-14 level are expected from threaded summation order.')
    atomic_json(where['repro'] / 'repro_result.json', result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit('Reproduction check FAILED: do not start production; send repro_result.json and run.log')


def cmd_compare_test256(cfg, args):
    """Rows 0-255 have the test's parameters: audits and clean spectra must agree."""
    where = layout(cfg)
    _, theta, _ = load_design()
    reference = json.loads((REFERENCE / 'audits.json').read_text())
    rows = range(len(reference['rows']))
    results, _ = collect_audits(cfg, theta, rows)
    audit_report = dict(compared=0, node_mismatches=[], worst_metric_relative_difference=0.0, not_available=[])
    for row in rows:
        if row not in results:
            audit_report['not_available'].append(row)
            continue
        ref, new = reference['rows'][row], results[row]
        audit_report['compared'] += 1
        if ref['nodes'] != new['nodes']:
            audit_report['node_mismatches'].append(row)
        for key in ('max_relative_visible', 'max_absolute_over_central', 'relative_l2'):
            difference = abs(new[key] - ref[key]) / max(abs(ref[key]), 1e-300)
            audit_report['worst_metric_relative_difference'] = max(audit_report['worst_metric_relative_difference'], difference)
    spectra = dict(compared=0, worst_masked=0.0, worst_unmasked=0.0, not_available=0)
    bins = dict(masked=np.load(REFERENCE / 'masked_clean_dl_bins40.npy'),
                unmasked=np.load(REFERENCE / 'unmasked_clean_dl_bins40.npy'))
    for row in rows:
        folder = where['rows'] / f'{row:05d}'
        if not (folder / 'status.json').exists():
            spectra['not_available'] += 1
            continue
        spectra['compared'] += 1
        for kind, name in (('masked', 'masked_clean_cl.npy'), ('unmasked', 'unmasked_clean_cl.npy')):
            new = rebin_unbinned((np.load(folder / name)[ELL] * DL_FACTOR)[None, :], ELL)[0]
            difference = float(np.max(np.abs(new / bins[kind][row] - 1)))
            spectra['worst_' + kind] = max(spectra['worst_' + kind], difference)
    tolerance = 1e-6
    passed = (audit_report['compared'] > 0 and not audit_report['node_mismatches']
              and audit_report['worst_metric_relative_difference'] < tolerance
              and spectra['worst_masked'] < tolerance and spectra['worst_unmasked'] < tolerance)
    audit_report['not_available'] = len(audit_report['not_available'])
    result = dict(passed=bool(passed), tolerance=tolerance, audits=audit_report, clean_spectra_bins40=spectra,
                  utc=now_utc())
    atomic_json(where['root'] / 'compare_test256.json', result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit('Rows 0-255 disagree with the 256-row test beyond the tolerance')


# ----------------------------------------------------------------------------- collection

def cmd_collect(cfg, args):
    """Verify rows 0..N-1 and write memory-mappable arrays (float64 D_ell, ell=80..7979)."""
    verify_sources()
    where = layout(cfg)
    manifest, theta, seeds = load_design()
    count = args.rows or len(theta)
    missing = [row for row in range(count) if not (where['rows'] / f'{row:05d}' / 'status.json').exists()]
    if missing and not args.allow_missing:
        raise SystemExit(f'{len(missing)} of {count} rows are not complete (first: {missing[:10]}); '
                         'use --allow-missing to write only the completed rows, listed explicitly')
    skipped = set(missing)
    rows = [row for row in range(count) if row not in skipped]
    if not rows:
        raise SystemExit('No completed rows to collect')
    output = Path(args.out) if args.out else where['dataset'] / f'rows_{count}'
    temporary = output.with_name(output.name + '.tmp')
    if output.exists():
        raise SystemExit(f'{output} exists; remove or rename it first')
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    kinds = [('clean_dl_unbinned', 'masked_clean_cl.npy'), ('noisy_dl_unbinned', 'masked_noisy_cross_cl.npy')]
    if args.include_unmasked:
        kinds.append(('unmasked_clean_dl_unbinned', 'unmasked_clean_cl.npy'))
    arrays = {key: np.lib.format.open_memmap(temporary / f'{key}.npy', mode='w+', dtype='<f8',
                                              shape=(len(rows), len(ELL))) for key, _ in kinds}
    noise_hashes = set()
    for position, row in enumerate(rows):
        folder = where['rows'] / f'{row:05d}'
        status = json.loads((folder / 'status.json').read_text())
        assert status['returncode'] == 0 and status['row'] == row
        assert status['theta'] == theta[row].tolist(), f'row {row}: theta differs from the design'
        assert status['split_seeds'] == seeds[row].tolist(), f'row {row}: seeds differ from the design'
        assert status['selected_halos'] == SELECTED_HALOS
        for name, expected in status['sha256'].items():
            assert sha256_file(folder / name) == expected, f'row {row}: {name} changed after it was written'
        observation = load_toml(folder / 'observation.toml')
        assert observation['output_nside'] == 4096 and observation['ell_max'] == 7979
        assert observation['noise_beam_applied'] is False and observation['split_Nell_multiplier'] == 1.0
        assert observation['split_seeds'] == seeds[row].tolist() and observation['mask_sha256'] == MASK_SHA256
        assert len(observation['noise_sha256']) == 2
        noise_hashes.update(observation['noise_sha256'])
        for key, name in kinds:
            cl = np.load(folder / name)
            assert cl.shape == (7980,) and np.isfinite(cl).all(), f'row {row}: bad {name}'
            if name != 'masked_noisy_cross_cl.npy':
                assert np.all(cl >= 0), f'row {row}: negative clean spectrum'
            arrays[key][position] = cl[ELL] * DL_FACTOR
        if position % 4096 == 0:
            print(f'collected {position}/{len(rows)} rows', flush=True)
    if len(noise_hashes) != 2 * len(rows):
        raise SystemExit('Repeated SO noise realization across rows')
    for key, _ in kinds:
        arrays[key].flush()
    binned = {}
    for key in ('clean_dl_unbinned', 'noisy_dl_unbinned'):
        binned[key.replace('_unbinned', '')] = np.concatenate(
            [rebin_unbinned(np.asarray(arrays[key][start:start + 4096]), ELL)
             for start in range(0, len(rows), 4096)])
    np.save(temporary / 'theta.npy', theta[rows])
    np.save(temporary / 'row_id.npy', np.asarray(rows, dtype=np.int64))
    np.save(temporary / 'noise_seeds.npy', seeds[rows])
    np.save(temporary / 'ell_unbinned.npy', ELL)
    for key, value in binned.items():
        np.save(temporary / f'{key}.npy', value)
    meta = dict(rows=len(rows), requested_prefix=count, missing_rows=missing,
                parameter_order=manifest['parameter_order'], lower=manifest['lower'], upper=manifest['upper'],
                units='D_ell = ell(ell+1)C_ell/2pi of Compton-y, dimensionless; ell = 80..7979 unbinned',
                binning='40 bins of 200 multipoles (last to 7979), weights 2l+1',
                spectra=dict(clean_dl_unbinned='masked clean signal (fsky=0.4 cap, 2 arcmin beam)',
                             noisy_dl_unbinned='signed cross spectrum of two masked signal+noise splits, '
                                               'SO baseline Deproj-0, two seeds per row',
                             unmasked_clean_dl_unbinned='full-sky clean signal (only with --include-unmasked)'),
                mask_sha256=MASK_SHA256, independent_noise_realizations=len(noise_hashes),
                design_sha256=manifest['sha256'], created_utc=now_utc(), bundle_git=bundle_git_state())
    atomic_json(temporary / 'meta.json', meta)
    if args.npz:
        del arrays
        with (temporary / 'dataset.npz').open('wb') as stream:
            np.savez(stream, row_id=np.asarray(rows), theta=theta[rows], ell_unbinned=ELL,
                     clean_dl_unbinned=np.load(temporary / 'clean_dl_unbinned.npy', mmap_mode='r'),
                     noisy_dl_unbinned=np.load(temporary / 'noisy_dl_unbinned.npy', mmap_mode='r'),
                     clean_dl=binned['clean_dl'], noisy_dl=binned['noisy_dl'], lower=np.asarray(manifest['lower']),
                     upper=np.asarray(manifest['upper']), parameter_order=np.asarray(manifest['parameter_order']),
                     noise_seeds=seeds[rows])
    temporary.replace(output)
    print(f'Wrote {len(rows)} rows to {output}' + (f' ({len(missing)} missing rows listed in meta.json)' if missing else ''))


# ----------------------------------------------------------------------------- CLI

def cmd_show(cfg, args):
    """Print one value for the submit scripts."""
    if args.what in ('run_root', 'julia', 'julia_depot'):
        print(cfg[args.what])
    elif args.what == 'audit_tasks':
        print(audit_task_count(cfg, load_design()[0]['count']))
    elif args.what == 'batches':
        print(len(load_plan(cfg)['batches']))
    elif args.what == 'production_tasks':
        print(math.ceil(len(load_plan(cfg)['batches']) / args.batches_per_task))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'config.toml'))
    commands = parser.add_subparsers(dest='command', required=True)

    repro = commands.add_parser('repro-check', help='rerun and compare rows 0-3 of the 256-row test')
    repro.add_argument('--run', action='store_true', help='run the batch (about 20 min on 26 cores, 40 GiB)')
    repro.add_argument('--threads', type=int)
    repro.add_argument('--dry-run', action='store_true', help='print the Julia command and exit')

    commands.add_parser('prepare-audits', help='write the audit task files')
    audit = commands.add_parser('audit', help='run one audit task')
    audit.add_argument('--task', type=int, required=True)
    audit.add_argument('--threads', type=int)
    audit.add_argument('--dry-run', action='store_true')

    prepare = commands.add_parser('prepare-batches', help='check audits and write the batch plan')
    prepare.add_argument('--defer-unresolved', action='store_true')
    prepare.add_argument('--batches-per-task', type=int, default=4, help='only used to print the array range')

    run = commands.add_parser('run', help='run production batches')
    run.add_argument('--array-index', type=int, help='run batches [index*K, (index+1)*K)')
    run.add_argument('--batches-per-task', type=int, default=4, help='K for --array-index')
    run.add_argument('--batches', help='comma-separated batch ids')
    run.add_argument('--pool', action='store_true', help='claim any unstarted batch, in order')
    run.add_argument('--max-batches', type=int, help='with --pool: stop after this many new batches')
    run.add_argument('--stop-after-hours', type=float, help='do not start a batch after this many hours')
    run.add_argument('--threads', type=int)
    run.add_argument('--skip-repro-gate', action='store_true')
    run.add_argument('--dry-run', action='store_true')

    commands.add_parser('status', help='progress report')
    reset = commands.add_parser('reset', help='move failed or interrupted work to attempts/ for a rerun')
    reset.add_argument('--batches')
    reset.add_argument('--audit-tasks')
    reset.add_argument('--all-failed', action='store_true')
    reset.add_argument('--all-interrupted', action='store_true')
    reset.add_argument('--stale-claims-hours', type=float, help='claims older than this are from dead jobs')
    reset.add_argument('--force', action='store_true', help='also reset explicitly listed claimed work')

    commands.add_parser('compare-test256', help='compare rows 0-255 with the 256-row test')
    collect = commands.add_parser('collect', help='assemble the dataset')
    collect.add_argument('--rows', type=int, help='collect the prefix 0..N-1 (default: all rows)')
    collect.add_argument('--allow-missing', action='store_true')
    collect.add_argument('--include-unmasked', action='store_true')
    collect.add_argument('--npz', action='store_true', help='also write one dataset.npz like the 256-row test')
    collect.add_argument('--out')

    show = commands.add_parser('show', help='print a value used by the submit scripts')
    show.add_argument('what', choices=['run_root', 'julia', 'julia_depot', 'audit_tasks', 'batches',
                                       'production_tasks'])
    show.add_argument('--batches-per-task', type=int, default=4)

    args = parser.parse_args()
    cfg = load_config(args.config)
    handlers = {'repro-check': cmd_repro_check, 'prepare-audits': cmd_prepare_audits, 'audit': cmd_audit,
                'prepare-batches': cmd_prepare_batches, 'run': cmd_run, 'status': cmd_status, 'reset': cmd_reset,
                'compare-test256': cmd_compare_test256, 'collect': cmd_collect, 'show': cmd_show}
    handlers[args.command](cfg, args)


if __name__ == '__main__':
    main()
