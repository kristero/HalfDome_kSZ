#!/usr/bin/env python3
"""Submit one job to a free eligible idark node, across the known queue groups.

This script runs on the login node and performs scheduler inspection only.
Inference runs in PBS. Repeated invocation returns the existing job identifier.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

ROOT = Path('/lustre/work/kristero10/tsz_truth_mean_validation_20261006')
QUEUES = {'mini2': ('GroupB', 2), 'mini': ('GroupA', 6), 'mini_B': ('GroupA', 16)}


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def gib(value):
    value = str(value).lower()
    for unit, scale in [('tb', 1024), ('gb', 1), ('mb', 1/1024),
                        ('kb', 1/1024**2), ('b', 1/1024**3)]:
        if value.endswith(unit): return float(value[:-len(unit)])*scale
    return float(value)/1024**3


def main(root, task='inference'):
    if root.resolve() != ROOT:
        raise ValueError('Use only the new experiment root')
    with (root/'submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = root/'jobs.json'
        state = json.loads(ledger.read_text()) if ledger.exists() else dict(jobs=[])
        existing = [j for j in state['jobs'] if j.get('task', 'inference') == task]
        if existing:
            print(json.dumps(existing, indent=2))
            return
        if len(state['jobs']) >= 20:
            raise RuntimeError('20-submission limit reached')
        if task == 'inference' and (root/'INFERENCE_COMPLETE.json').exists():
            print('Inference already complete; no submission')
            return
        if task == 'publication':
            assert (root/'INFERENCE_COMPLETE.json').exists()
        # A preparation record must be present before any scheduler mutation.
        assert (root/'LOCAL_CHECKS.json').exists()
        jobs = json.loads(command(['qstat', '-f', '-F', 'json'])).get('Jobs', {})
        own = [j for j in jobs.values() if j.get('Job_Owner', '').split('@')[0] == 'kristero10'
               and j.get('job_state') not in ['F', 'E']]
        if len(own) >= 20:
            raise RuntimeError('20-job active limit reached; do not submit')
        nodes = json.loads(command(['pbsnodes', '-a', '-F', 'json']))['nodes']
        options = []
        for node_name, node in nodes.items():
            if any(flag in node.get('state', '') for flag in ['down', 'offline', 'unknown']): continue
            available = node.get('resources_available', {})
            assigned = node.get('resources_assigned', {})
            cpus = int(available.get('ncpus', 0))-int(assigned.get('ncpus', 0))
            memory = gib(available.get('mem', '0gb'))-gib(assigned.get('mem', '0gb'))
            if cpus < 26 or memory < 16: continue
            for queue, (group, limit) in QUEUES.items():
                count = sum(j.get('queue') == queue for j in own)
                if available.get('host_group') == group and count < limit:
                    options.append((cpus, memory, queue, node_name,
                                    available.get('host', node_name.split('.')[0])))
        if not options:
            raise RuntimeError('No eligible free 26-CPU/16-GB node; nothing submitted')
        cpus, memory, queue, node_name, host = max(options)
        job_id = command(['qsub', '-q', queue, '-l',
                          f'select=1:ncpus=26:mpiprocs=1:mem=16gb:host={host}',
                          '-o', str(root/'logs'), str(root/('truth_mean.pbs' if task == 'inference'
                                                          else 'publication_corners.pbs'))]).strip()
        state['jobs'].append(dict(id=job_id, task=task, queue=queue, host=host, node=node_name,
                      free_cpus_at_submission=cpus, free_memory_gib_at_submission=memory,
                      submitted_utc=datetime.now(timezone.utc).isoformat()))
        ledger.write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--task', choices=['inference', 'publication'], default='inference')
    args = parser.parse_args()
    main(args.root, args.task)
