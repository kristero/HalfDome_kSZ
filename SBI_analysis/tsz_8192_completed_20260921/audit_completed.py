"""Validate fetched controls and summarize completed resolution experiments."""
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'tsz_8192_validation_recovery_20260921'
EXTREME = ROOT.parent / 'tsz_8192_noise_extremes_20260921'
PARAMETERS = ['P0', 'xc', 'beta', 'alpha_m_P0', 'alpha_m_xc',
              'alpha_m_beta', 'alpha_z_P0', 'alpha_z_xc', 'alpha_z_beta']


def read(path):
    return json.loads(path.read_text())


def main():
    report = read(RUN / 'results/report.json')
    snapshot = read(ROOT / 'cluster_snapshot.json')
    assert report['completed'] == report['requested'] == 86
    assert report['analysis_errors'] == [] and len(report['anchors']) == 2
    manifest = read(RUN / 'manifest.json')
    checked_sources, absent_sources = [], []
    for name, expected in manifest['source_sha256'].items():
        path = RUN / name
        if not path.exists():
            absent_sources.append(name)
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        checked_sources.append(name)
    spectra = 0
    seeds, noise_hashes, masks, parity = [], [], set(), []
    statuses = []
    for root in [RUN, EXTREME]:
        plan = read(root / 'plan.json')
        for task in plan['tasks']:
            path = root / 'controls' / f"{task['id']:03d}"
            status = read(path / 'status.json')
            assert status['returncode'] == 0, (task['id'], status)
            statuses.append(dict(task=task['id'], label=task['label'], seconds=status['seconds']))
            meta = tomllib.loads((path / 'control.toml').read_text())
            masks.add(meta['mask_sha256'])
            assert meta['output_nside'] == 4096 and meta['beam_arcmin'] == 2
            assert meta['split_Nell_multiplier'] == 1 and not meta['noise_beam_applied']
            for draw in meta['noise_draws']:
                seeds.extend(draw['split_seeds'])
                noise_hashes.extend([draw['noise1_sha256'],draw['noise2_sha256']])
                parity.append(draw['map_alm_parity'])
            for filename in path.rglob('*.npy'):
                data = np.load(filename)
                assert data.shape == (7980,) and np.isfinite(data).all(), filename
                spectra += 1
            remote_check = snapshot['noise_checks'][root.name][str(task['id'])]
            assert remote_check['draws'] == task.get('noise_draws', 0)
    assert len(seeds) == len(set(seeds)) == len(set(noise_hashes))
    assert len(masks) == 1 and max(parity) < 1e-10
    plan = read(RUN / 'plan.json')
    ell = np.arange(7980, dtype=float)
    factor = ell * (ell + 1) / (2 * np.pi)
    repeats = []
    for task in plan['tasks']:
        if 'repeats_task' not in task:
            continue
        a = np.load(RUN / 'controls' / f"{task['repeats_task']:03d}" / 'masked_clean_cl.npy') * factor
        b = np.load(RUN / 'controls' / f"{task['id']:03d}" / 'masked_clean_cl.npy') * factor
        relative = float(np.linalg.norm(a[80:] - b[80:]) / np.linalg.norm(a[80:]))
        assert relative < 1e-10
        repeats.append(dict(label=task['label'], relative_dl_norm=relative))
    summary = dict(completed_controls=len(statuses), locally_checked_finite_spectra=spectra,
                   remotely_checked_noise_spectra=sum(v['finite_spectra']
                       for rows in snapshot['noise_checks'].values() for v in rows.values()),
                   verified_sources=checked_sources, absent_local_sources=absent_sources,
                   unique_split_seeds=len(seeds), unique_noise_hashes=len(set(noise_hashes)),
                   mask_hashes=list(masks), max_map_alm_noise_parity=max(parity),
                   repeats=repeats, anchors=[], benchmarks=[],
                   extremes=read(EXTREME / 'results/extreme_noise.json')['records'])
    for anchor in report['anchors']:
        rows = []
        for cut in anchor['cuts']:
            sensitivity = [r for r in cut['sensitivity'] if 'distance' in r]
            rows.append(dict(ell_max=cut['ell_max'], frozen=cut['frozen_moped_distance'],
                local=cut['spherical_moped_distance'],
                reference_8192_16384=cut.get('reference16384_spherical_moped_distance'),
                trace_fraction=cut['information_trace_fraction'],
                mode_fractions=cut['identified_mode_information_fractions'],
                rank=cut['spherical_moped']['rank'],
                sensitivity_min=min(r['distance'] for r in sensitivity),
                sensitivity_max=max(r['distance'] for r in sensitivity),
                sensitivity_ranks=sorted(set(r['rank'] for r in sensitivity)),
                covariance_identity_error=cut['spherical_moped']['covariance_identity_error'],
                shrinkage=cut['spherical_moped']['oas_shrinkage']))
        summary['anchors'].append(dict(name=anchor['anchor'],
            derivative_changes=dict(zip(PARAMETERS,anchor['derivative_step_relative_change'])),cuts=rows))
    for row in report['benchmarks']:
        if row['task_id'] in [0,78,79,80]:
            summary['benchmarks'].append(row)
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    for anchor in summary['anchors']:
        print(anchor['name'],'derivative changes:',anchor['derivative_changes'])
        for cut in anchor['cuts']:
            print(cut)
    for row in summary['benchmarks']:
        print('BENCHMARK',row)
    print('VERIFIED',len(statuses),'controls;',spectra,'finite spectra; missing local sources:',absent_sources)


if __name__ == '__main__':
    main()
