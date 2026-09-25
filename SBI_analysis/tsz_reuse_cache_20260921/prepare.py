"""Create an isolated, finite performance experiment; never submit a dataset."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'tsz_8192_validation_recovery_20260921'


def main():
    for directory in ('inputs', 'results', 'plots'):
        (ROOT / directory).mkdir(exist_ok=True)
    for name in ('fullsky_test.jl', 'spherical_truncation_profiles.jl',
                 'external_dependencies.json'):
        target = ROOT / name
        if not target.exists():
            shutil.copy2(PREVIOUS / name, target)
    old = json.loads((PREVIOUS / 'plan.json').read_text())
    cases = []
    for label in ('Battaglia12', 'FL_L1_m9', 'compact', 'extended_shallow'):
        task = next(t for t in old['tasks'] if t['label'] == label and t['nside'] == 8192)
        cases.append(dict(label=label, theta=task['theta'], reference_id=task['id']))
    tasks = []

    def add(label, mode, nodes=(512, 256, 128), selected=None, **extra):
        tasks.append(dict(id=len(tasks), label=label, mode=mode, nodes=list(nodes),
                          cases=cases if selected is None else selected,
                          nside=8192, threads=26, **extra))

    # The same four skies, same process, same hardware allocation in each arm.
    add('serial_four_catalogue_passes', 'reread')
    add('read_once_four_painters', 'read_once')
    add('shared_geometry_four_profiles', 'fused', pixel_targets=[4096])
    for label, nodes in (
        ('half_theta', (256, 256, 128)),
        ('half_logz', (512, 128, 128)),
        ('half_logmass', (512, 256, 64)),
        ('half_all_axes', (256, 128, 64)),
        ('quarter_all_axes', (128, 64, 32)),
    ):
        add(label, 'fused', nodes)
    # Independent fresh-process repeat to expose scheduling/cache variability.
    add('shared_geometry_repeat', 'fused')
    add('shared_geometry_two_profiles', 'fused', selected=cases[:2])
    add('single_profile_optimized', 'fused', selected=cases[:1])
    # Only an accuracy reference, not a proposed 16384 dataset.
    for case in (cases[3], cases[0]):
        add(case['label'] + '_reference16384', 'fused', selected=[case],
            pixel_targets=[8192, 4096], reference_only=True)
        tasks[-1]['nside'] = 16384
    plan = dict(tasks=tasks, ell_min=80, ell_max=7979, output_nside=4096,
                beam_arcmin=2., fsky=.4, mask_seed=12345,
                diagnostic_256_submitted=False,
                accuracy_budget_sigma=.1,
                accuracy_budget_interpretation='Engineering accuracy target, not a prior exclusion',
                scope='Performance and rendering controls; no new training rows')
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Prepared {len(tasks)} finite controls, no submission.')


if __name__ == '__main__':
    main()
