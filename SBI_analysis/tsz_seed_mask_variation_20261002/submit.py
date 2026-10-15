#!/usr/bin/env python3
"""Submit exactly ten lightcones plus their dependent analysis, and record every job immediately."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def queue_order():
    """Include both node pools, preferring the pool with room for a 26-CPU job now.

    mini and mini_B both use GroupA; mini2 uses GroupB. The returned allocation
    keeps five paint jobs in the higher-priority mini queue, two in mini2 (its
    per-user running limit), and three in mini_B. The scheduler chooses the nodes.
    """
    nodes = json.loads(subprocess.check_output(['pbsnodes', '-a', '-F', 'json'], text=True))['nodes']
    free_b = any(n.get('resources_available', {}).get('host_group') == 'GroupB'
                 and not any(s in n.get('state', '') for s in ('offline', 'down', 'unknown'))
                 and n['resources_available'].get('ncpus', 0) - n.get('resources_assigned', {}).get('ncpus', 0) >= 26
                 for n in nodes.values())
    if free_b:
        return ['mini2', 'mini', 'mini2', 'mini', 'mini', 'mini', 'mini', 'mini_B', 'mini_B', 'mini_B']
    return ['mini', 'mini2', 'mini', 'mini2', 'mini', 'mini', 'mini', 'mini_B', 'mini_B', 'mini_B']


def main(root):
    root = root.resolve()
    if (root / 'jobs.json').exists():
        raise RuntimeError('jobs.json exists: inspect those jobs before resubmitting anything')
    subprocess.run([sys.executable, str(root / 'code/analyze.py'), '--root', str(root), '--preflight'], check=True)
    plan = json.loads((root / 'plan.json').read_text())
    assert len(plan['simulation_seeds']) == 10 and len(plan['cases']) == 19
    jobs = dict(paint={}, analysis=None, queues={})

    def persist():
        (root / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')

    def submit(script, variables, name, queue, dependency=None):
        cmd = ['qsub', '-q', queue, '-N', name, '-o', str(root / 'logs'), '-v', variables]
        if dependency:
            cmd += ['-W', 'depend=afterok:' + ':'.join(dependency)]
        cmd.append(str(root / 'code' / script))
        result = subprocess.check_output(cmd, text=True, cwd=root).strip()
        assert result.endswith('.idark'), result
        return result

    for seed, queue in zip(plan['simulation_seeds'], queue_order()):
        jobs['paint'][str(seed)] = submit('paint.pbs', f'RUN_ROOT={root},SIM_SEED={seed}', f'tsz_s{seed}', queue)
        jobs['queues'][str(seed)] = queue
        persist()
    jobs['analysis'] = submit('analyze.pbs', f'RUN_ROOT={root}', 'tsz_seed_stats', 'mini', list(jobs['paint'].values()))
    persist()
    print(json.dumps(jobs, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
