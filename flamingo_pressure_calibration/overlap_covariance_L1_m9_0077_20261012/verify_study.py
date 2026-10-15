"""Gates G1-G7 of the overlap/covariance study (plan Step 1). Local; run after control.py fetch.

  python -B verify_study.py            all gates whose inputs exist (others: 'not_run'), -> verification.json
  python -B verify_study.py --g1       G1 only; (re)writes input_hashes.json
  python -B verify_study.py --manifest also writes SHA256SUMS and ARTIFACTS.json for the fetched products

G1 frozen sources = radii234 executed_sources.json and final_source_manifest.json (SHA256);
G2 grids: old edges an exact subset; G3 census vs brute force and own-satellite lists = radii234 (census_g3.json,
metadata_plan.json); G4 dataset probe (probe_g4.json + prepare's re-check); G5 one-cell gate; G6 one-file gate;
G7 after the run: legacy sums np.array_equal radii234's; the phase split reproduces y_noagn; SOAP Y/Y_noAGN at
R200c, R500c, 5 R500c and ROSAT L_X at R500c (|median fractional difference| < 1e-4, 16th and 84th percentiles
within +-0.1%, >= 98.5% of haloes within 0.3%); bound gas counts never exceed SOAP's; the 45 SWIFT-check
particle-ID hashes equal radii234's; new blocks sum to the legacy totals.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = Path(os.environ.get('OVERLAP_STUDY_DIR', str(HERE))).resolve()
sys.path.insert(0, str(HERE / 'dependencies'))
sys.path.insert(0, str(HERE))
RADII234 = HERE.parent / 'radii234_contamination_L1_m9_0077_20261008'
FROZEN = ['pilot_physics.py', 'measure.py', 'models.py', 'projection_model.py', 'swift_reader.py', 'check_fresh_mask.py',
          'remote_control.py', 'dependencies/common.py', 'dependencies/flamingo_io.py', 'dependencies/catalogue_stage.py']
NEW = ['study_config.json', 'grids.py', 'census.py', 'prepare_study.py', 'measure_overlap.py', 'extract_study.py', 'geometry.py',
       'verify_study.py', 'control.py', 'probe_g4.py']
INPUTS = {'inputs/radii234_selected_halos.npz': 'selected_halos.npz', 'inputs/radii234_metadata_plan.json': 'metadata_plan.json',
          'inputs/radii234_particle_hashes.json': 'particle_hashes.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def g1():
    executed = json.loads((RADII234 / 'executed_sources.json').read_text())
    final = json.loads((RADII234 / 'final_source_manifest.json').read_text())
    frozen = {n: {'sha256': sha(HERE / n), 'equals_radii234_executed': sha(HERE / n) == executed.get(n),
                  'equals_radii234_final': sha(HERE / n) == final.get(n)} for n in FROZEN}
    inputs = {n: {'sha256': sha(HERE / n), 'equals_radii234_original': sha(HERE / n) == sha(RADII234 / original)}
              for n, original in INPUTS.items()}
    record = {'frozen_sources': frozen, 'inputs': inputs,
              'new_sources': {n: sha(HERE / n) for n in NEW if (HERE / n).exists()},
              'dry_run_catalogue': {str(HERE.parent / 'sensitivity_projects_L1_m9_0077_20261009/flamingo_centrals_metadata.npz'):
                                    sha(HERE.parent / 'sensitivity_projects_L1_m9_0077_20261009/flamingo_centrals_metadata.npz')}}
    passed = all(v['equals_radii234_executed'] and v['equals_radii234_final'] for v in frozen.values()) and \
        all(v['equals_radii234_original'] for v in inputs.values())
    record['passed'] = passed
    (HERE / 'input_hashes.json').write_text(json.dumps(record, indent=1) + '\n')
    return {'status': 'passed' if passed else 'FAILED', 'frozen_equal': sum(v['equals_radii234_executed'] for v in frozen.values()),
            'frozen_total': len(frozen), 'inputs_equal': all(v['equals_radii234_original'] for v in inputs.values())}


def g2():
    import grids as G
    report = G.g2_report()
    import pilot_physics as P
    passed = np.array_equal(G.X200[G.OLD_INDEX], P.EDGES)
    return {'status': 'passed' if passed else 'FAILED', **report}


def load_json(name):
    path = STUDY / name
    return json.loads(path.read_text()) if path.exists() else None


def g3():
    census, plan = load_json('census_g3.json'), load_json('metadata_plan.json')
    if census is None:
        return {'status': 'not_run', 'reason': 'census_g3.json not fetched'}
    passed = bool(census['overlapper_sets_equal'] and census['own_satellite_lists_equal_radii234']
                  and (plan is None or plan.get('own_satellite_lists_equal_radii234')))
    return {'status': 'passed' if passed else 'FAILED', **census}


def g4():
    probe, plan = load_json('probe_g4.json'), load_json('metadata_plan.json')
    if probe is None:
        return {'status': 'not_run'}
    checks = {'xray_not_auxiliary': probe['auxiliary_probe_fields'] == [],
              'rosat_column_2': probe['top_named_columns'].get('XrayLuminosities', [None] * 3)[2] == 'ROSAT',
              'xray_chunk_equals_top_small_cell': probe['small_cell']['fields']['XrayLuminosities']['values_identical'],
              'xray_chunk_equals_top_halo_cell': probe['halo_cell_slice']['xray_chunk_vs_top']['values_identical'],
              'ids_chunk_equal_top': probe['halo_cell_slice']['ids_chunk_vs_top_identical'],
              'sfr_sign_convention': 'If positive, star formation rates' in probe['top_fields']['StarFormationRates']['attrs']['Description'],
              'y_zero_where_sfr_positive': probe['halo_cell_slice']['y_zero_where_sfr_positive']}
    if plan is not None:
        checks['prepare_recheck'] = plan['g4_recheck']['passed']
    return {'status': 'passed' if all(checks.values()) else 'FAILED', **checks}


def gate_file_status(name):
    record = load_json(name)
    if record is None:
        return {'status': 'not_run'}
    return {'status': 'passed' if record.get('passed') else 'FAILED',
            **{k: v for k, v in record.items() if k not in ('gate',)},
            **({'gate_haloes': len(record['gate']), 'gate_all_passed': all(g['passed'] for g in record['gate'])} if 'gate' in record else {})}


def fractional_summary(value, reference):
    valid = reference > 0
    f = value[valid] / reference[valid] - 1
    p16, p50, p84 = np.percentile(f, [16, 50, 84])
    within = float(np.mean(np.abs(f) <= 3e-3))
    passed = abs(p50) < 1e-4 and -1e-3 <= p16 and p84 <= 1e-3 and within >= .985
    return {'passed': bool(passed), 'haloes': int(valid.sum()), 'zero_reference': int((~valid).sum()),
            'median': float(p50), 'p16': float(p16), 'p84': float(p84), 'fraction_within_0p3pct': within,
            'beyond_0p3pct': int(np.sum(np.abs(f) > 3e-3)), 'max_abs': float(np.max(np.abs(f)))}


def g7(radii234_measurements):
    needed = ['measurements_legacy.npz', 'measurements_owner.npz', 'measurements_excision.npz', 'measurements_partition.npz',
              'apertures.npz', 'selected_halos.npz', 'overlap_tables.npz', 'particle_hashes.json', 'complete.json']
    missing = [n for n in needed if not (STUDY / n).exists()]
    if missing:
        return {'status': 'not_run', 'missing': missing}
    import grids as G
    import measure_overlap as MO
    import pilot_physics as P
    legacy = np.load(STUDY / 'measurements_legacy.npz')['sums']
    owner = np.load(STUDY / 'measurements_owner.npz')
    ap = np.load(STUDY / 'apertures.npz')
    halos = dict(np.load(STUDY / 'selected_halos.npz'))
    tables = dict(np.load(STUDY / 'overlap_tables.npz'))
    out = {}
    n_old = int(json.loads((STUDY / 'study_config.json').read_text())['old_sample']['rows'][1])
    if radii234_measurements is not None and Path(radii234_measurements).exists():
        old = np.load(radii234_measurements)['sums']
        out['legacy_sums_array_equal_radii234'] = bool(np.array_equal(legacy[:n_old], old))
        out['legacy_rows_compared'] = int(len(old))
    total = legacy.sum(axis=1)
    o200 = owner['owner200']
    qi = MO.QUANTITIES.index
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b).max(axis=-1, keepdims=True), 1e-300)))  # noqa: E731
    phase_y = G.to_old(o200[:, :, 1:, qi('y')].sum(axis=(1, 2)))
    out['phase_split_y_noagn_max_relative'] = rel(phase_y, total[:, P.SUM_NAMES.index('y_noagn')])
    out['phase_split_reproduces_y_noagn'] = out['phase_split_y_noagn_max_relative'] < 1e-12
    shared = {}
    for q in ('count', 'mass', 'y', 'electrons', 'electron_kT', 'volume', 'gas_pressure_volume'):
        shared[q] = rel(G.to_old(o200[:, :, :, qi(q)].sum(axis=(1, 2))), total[:, P.SUM_NAMES.index(q)])
    out['owner_vs_legacy_max_relative'] = shared
    out['owner_count_exact'] = bool(np.array_equal(G.to_old(o200[:, :, :, 0].sum(axis=(1, 2))), total[:, 0]))
    reduced = np.stack([o200[:, :, :, qi(q)].sum(axis=(1, 2)) for q in MO.REDUCED], axis=1)
    exc = np.load(STUDY / 'measurements_excision.npz')['excision']
    part = np.load(STUDY / 'measurements_partition.npz')['partition']
    out['excision_sums_max_relative'] = max(rel(exc[:, v].sum(axis=(1, 2, 3)), reduced) for v in range(exc.shape[1]))
    out['partition_sums_max_relative'] = rel(part.sum(axis=(1, 2, 3)), reduced)
    out['excision_partition_dtype'] = [str(exc.dtype), str(part.dtype)]
    apert = ap['apertures']
    soap = {}
    pairs = {'Y_R200c': (apert[:, 0, 0, 2], 'y_200_mpc2'), 'Ynoagn_R200c': (apert[:, 0, 0, 3], 'y_noagn_200_mpc2'),
             'Y_R500c': (apert[:, 1, 0, 2], 'y_500_mpc2'), 'Ynoagn_R500c': (apert[:, 1, 0, 3], 'y_noagn_500_mpc2'),
             'Y_5R500c': (apert[:, 2, 0, 2], 'y_5x500_mpc2'), 'Ynoagn_5R500c': (apert[:, 2, 0, 3], 'y_noagn_5x500_mpc2'),
             'LX_ROSAT_R500c': (apert[:, 1, 0, 4], ('xray_500_erg_s', 2)),
             'LXnoagn_ROSAT_R500c': (apert[:, 1, 0, 5], ('xray_noagn_500_erg_s', 2))}
    for name, (value, key) in pairs.items():
        reference = halos[key[0]][:, key[1]] if isinstance(key, tuple) else halos[key]
        soap[name] = fractional_summary(value, reference)
    out['soap_comparisons'] = soap
    if 'xray_restframe_500_erg_s' in halos:      # diagnostic only (the gate uses the observer-frame column)
        out['diagnostic_LX_ROSAT_R500c_vs_restframe'] = fractional_summary(apert[:, 1, 0, 4], halos['xray_restframe_500_erg_s'][:, 2])
    bound = apert[:, 3, 1, 0]
    out['central_bound_count_never_exceeds_soap'] = bool(np.all(bound <= halos['bound_gas_count']))
    ov = ap['overlapper']
    out['overlapper_bound_count_never_exceeds_soap'] = bool(np.all(ov[:, 0, 1, 0] <= tables['ov_bound_gas_count']))
    new_hashes = json.loads((STUDY / 'particle_hashes.json').read_text())
    old_hashes = json.loads((STUDY / 'inputs' / 'radii234_particle_hashes.json').read_text())
    out['swift_hashes_equal'] = new_hashes == old_hashes
    out['swift_hashes_compared'] = len(old_hashes)
    diag = ap['diagnostics']
    names = MO.DIAGNOSTICS
    out['star_forming_with_nonzero_y'] = int(diag[:, names.index('star_forming_with_nonzero_y')].sum())
    out['membership_not_in_soap'] = int(diag[:, names.index('membership_not_in_soap')].sum())
    out['hs_overflow'] = int(diag[:, [names.index(k) for k in names if 'flow' in k]].sum())
    checks = [out.get('legacy_sums_array_equal_radii234', False), out['phase_split_reproduces_y_noagn'],
              max(shared.values()) < 1e-12, out['owner_count_exact'], out['excision_sums_max_relative'] < 1e-12,
              out['partition_sums_max_relative'] < 1e-12, all(v['passed'] for v in soap.values()),
              out['central_bound_count_never_exceeds_soap'], out['overlapper_bound_count_never_exceeds_soap'],
              out['swift_hashes_equal'], out['star_forming_with_nonzero_y'] == 0]
    out['status'] = 'passed' if all(checks) else 'FAILED'
    return out


def manifest():
    files = sorted(p for p in STUDY.iterdir() if p.is_file() and p.suffix in ('.npz', '.json', '.py', '.md', '.csv', '.png')
                   and p.name not in ('SHA256SUMS', 'ARTIFACTS.json'))
    (STUDY / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    (STUDY / 'ARTIFACTS.json').write_text(json.dumps({p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files}, indent=1) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--g1', action='store_true')
    parser.add_argument('--manifest', action='store_true')
    parser.add_argument('--radii234-measurements', default=str(RADII234 / 'measurements.npz'))
    args = parser.parse_args()
    if args.g1:
        print(json.dumps(g1(), indent=1))
        return
    result = {'G1_frozen_sources': g1(), 'G2_grids': g2(), 'G3_census': g3(), 'G4_dataset_probe': g4(),
              'G5_one_cell_gate': gate_file_status('one_cell_gate.json'), 'G6_one_file_gate': gate_file_status('gate_file.json'),
              'G7_after_run': g7(args.radii234_measurements)}
    result['all_run_gates_passed'] = all(v['status'] != 'FAILED' for v in result.values())
    result['gates_not_run'] = [k for k, v in result.items() if isinstance(v, dict) and v.get('status') == 'not_run']
    (STUDY / 'verification.json').write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(json.dumps({k: (v['status'] if isinstance(v, dict) else v) for k, v in result.items()}, indent=1))
    if args.manifest:
        manifest()


if __name__ == '__main__':
    main()
