#!/usr/bin/env python3
"""Lightweight PBS submission/monitoring: at most 20 jobs, with work confined to nodes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from common import ROOT, SEEDS, write_json

LIMITS = {'mini':6, 'mini2':2, 'mini_B':16}
GROUPS = {'mini':'GroupA', 'mini2':'GroupB', 'mini_B':'GroupA'}
PREFERRED = ['mini2', 'mini', 'mini', 'mini_B', 'mini2', 'mini', 'mini', 'mini_B', 'mini']


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def size_gib(value):
    value = str(value).lower()
    for unit, factor in [('tb',1024.),('gb',1.),('mb',1/1024.),('kb',1/1024.**2),('b',1/1024.**3)]:
        if value.endswith(unit):
            return float(value[:-len(unit)]) * factor
    return float(value) / 1024.**3


def available():
    jobs = json.loads(command(['qstat','-f','-F','json'])).get('Jobs',{})
    nodes = json.loads(command(['pbsnodes','-a','-F','json']))['nodes']
    counts = dict.fromkeys(LIMITS,0)
    for job in jobs.values():
        if job.get('Job_Owner','').split('@')[0] == 'kristero10' and job.get('job_state') not in ['F','E']:
            queue=job.get('queue')
            if queue in counts:
                counts[queue]+=1
    slots = {'GroupA':0, 'GroupB':0}
    for node in nodes.values():
        if any(flag in node.get('state','') for flag in ['offline','down','unknown']):
            continue
        resources=node.get('resources_available',{})
        assigned=node.get('resources_assigned',{})
        group=resources.get('host_group')
        cpus=int(resources.get('ncpus',0))-int(assigned.get('ncpus',0))
        mem=size_gib(resources.get('mem','0gb'))-size_gib(assigned.get('mem','0gb'))
        if group in slots:
            slots[group]+=max(0,min(cpus//26,int(mem//64)))
    return jobs,counts,slots


def submit(root, task, queue, state):
    if len(state['jobs']) >= 20:
        raise RuntimeError('The authorized 20-submission limit has been reached')
    script='paint.pbs' if isinstance(task,int) else 'pipeline.pbs'
    variables=f'RUN_ROOT={root}'
    if isinstance(task,int):
        variables+=f',SIM_SEED={task}'
    job=command(['qsub','-q',queue,'-v',variables,'-o',str(root/'logs'),str(root/'code'/script)]).strip()
    state['jobs'].append(dict(id=job,task=str(task),queue=queue,submitted_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    write_json(root/'jobs.json',state)
    print('SUBMITTED',job,task,queue,flush=True)


def run(root, once=False):
    lock=open(root/'dispatcher.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((root/'jobs.json').read_text()) if (root/'jobs.json').exists() else dict(jobs=[])
    while True:
        if (root/'COMPLETE.json').exists():
            write_json(root/'status.json',dict(state='complete',jobs=len(state['jobs'])))
            return
        try:
            jobs,counts,slots=available()
            failed=[]
            for entry in state['jobs']:
                task=entry['task']
                marker=root/('COMPLETE.json' if task=='pipeline' else f'results/seed_{task}/VALIDATED.json')
                if marker.exists() or entry['id'] in jobs:
                    continue
                finished=json.loads(command(['qstat','-x','-f','-F','json',entry['id']])).get('Jobs',{})
                record=finished.get(entry['id'],{})
                if record.get('job_state')=='F':
                    failed.append(dict(job=entry['id'],task=task,exit_status=record.get('Exit_status'),
                                       reason='Job finished without its required validation/completion marker'))
            if failed:
                write_json(root/'status.json',dict(state='failed',failures=failed,jobs=len(state['jobs'])))
                print('FAILED',failed,flush=True)
                return
            submitted={entry['task'] for entry in state['jobs']}
            for index,seed in enumerate(SEEDS[1:]):
                if str(seed) in submitted:
                    continue
                preferred=PREFERRED[index]
                options=[preferred]+[q for q in LIMITS if q!=preferred]
                queue=next((q for q in options if counts[q]<LIMITS[q] and slots[GROUPS[q]]>0),None)
                if queue is None:
                    continue
                submit(root,seed,queue,state)
                counts[queue]+=1;slots[GROUPS[queue]]-=1
            ready=all((root/f'results/seed_{seed}/VALIDATED.json').exists() for seed in SEEDS)
            if ready and 'pipeline' not in submitted:
                queue=next((q for q in ['mini2','mini','mini_B'] if counts[q]<LIMITS[q] and slots[GROUPS[q]]>0),None)
                if queue:
                    submit(root,'pipeline',queue,state)
            write_json(root/'status.json',dict(state='running' if ready else 'painting',
                validated_seeds=[s for s in SEEDS if (root/f'results/seed_{s}/VALIDATED.json').exists()],
                jobs=len(state['jobs']),free_26cpu_64gb_slots=slots,
                updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            print('SCHEDULER',str(error),flush=True)
            write_json(root/'status.json',dict(state='waiting_for_scheduler',reason=str(error),jobs=len(state['jobs'])))
        if once:
            return
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--once',action='store_true')
    args=p.parse_args();run(args.root,args.once)
