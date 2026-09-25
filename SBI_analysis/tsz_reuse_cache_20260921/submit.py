"""Freeze and submit numerical tests only. No 256-dataset submission path."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
REMOTE = PurePosixPath('/lustre/work/kristero10') / ROOT.name


def write_pbs(name, queue, cpus, memory, command, wall='23:59:00'):
    value = f'''#!/bin/bash
#PBS -N tsz_speed
#PBS -q {queue}
#PBS -l select=1:ncpus={cpus}:mem={memory}gb
#PBS -l walltime={wall}
#PBS -j oe
set -euo pipefail
cd {REMOTE}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS={cpus} MKL_NUM_THREADS=1
/home/anaconda3/bin/python3 {command}
'''
    (ROOT/name).write_bytes(value.encode())  # PBS requires LF, even on Windows.


def remote_submit():
    from run import verify_sources, write_json
    verify_sources()
    journal = ROOT/'submission.json'
    if journal.exists():
        raise RuntimeError('Submission journal already exists; never resubmit implicitly')
    state = dict(utc=datetime.now(timezone.utc).isoformat(),gate=None,workers=[],
                 reference=None,analysis=None,diagnostic_256_submitted=False)
    write_json(journal,state)

    def submit(file,name,depend=None,variables=None):
        command = ['qsub','-N',name,'-o',str(ROOT/(name+'.log'))]
        if depend:command += ['-W','depend='+depend]
        if variables:command += ['-v',variables]
        value = subprocess.check_output(command+[str(ROOT/file)],text=True).strip()
        assert value.split('.')[0].isdigit(), value
        return value

    state['gate'] = submit('gate.pbs','tsz_speed_gate');write_json(journal,state)
    for shard in range(4):
        state['workers'].append(submit('worker.pbs',f'tsz_speed_{shard}',
            'afterok:'+state['gate'],f'TSZ_SHARD={shard}'))
        write_json(journal,state)
    # Speed experiments take scheduling priority over the expensive reference.
    state['reference'] = submit('reference.pbs','tsz_pixel_ref','afterok:'+':'.join(state['workers']))
    write_json(journal,state)
    state['analysis'] = submit('analysis.pbs','tsz_speed_report',
        'afterany:'+':'.join(state['workers']+[state['reference']]))
    write_json(journal,state)
    print(json.dumps(state,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--submit',action='store_true')
    parser.add_argument('--remote',action='store_true')
    args=parser.parse_args()
    if args.remote:
        remote_submit();return
    write_pbs('gate.pbs','mini2',2,8,'run.py --gate',wall='00:30:00')
    write_pbs('worker.pbs','mini',26,64,'run.py --shard "$TSZ_SHARD"')
    write_pbs('reference.pbs','mini',26,96,'run.py --references')
    write_pbs('analysis.pbs','mini2',2,8,'analyze.py',wall='01:00:00')
    files = sorted([p for p in ROOT.iterdir() if p.suffix in ('.py','.jl','.pbs')]+
                   [ROOT/'plan.json',ROOT/'external_dependencies.json']+list((ROOT/'inputs').glob('*')))
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(dict(sha256=hashes,
        scope='Numerical controls only; 256 dataset held'),indent=2)+'\n')
    if not args.submit:
        print('Frozen locally; use --submit for authorized numerical tests.');return
    existing = subprocess.run(['ssh','idark','test','-e',str(REMOTE/'manifest.json')])
    if existing.returncode != 1:
        raise RuntimeError('Remote root exists or connection check failed')
    with tarfile.open(ROOT/'source.tar.gz','w:gz') as archive:
        for path in files+[ROOT/'manifest.json']:
            archive.add(path,arcname=path.relative_to(ROOT).as_posix())
    subprocess.run(['ssh','idark','mkdir','-p',str(REMOTE)],check=True)
    subprocess.run(['scp',str(ROOT/'source.tar.gz'),'idark:'+str(REMOTE/'source.tar.gz')],check=True)
    subprocess.run(['ssh','idark','tar','-xzf',str(REMOTE/'source.tar.gz'),'-C',str(REMOTE)],check=True)
    result = subprocess.run(['ssh','idark','/home/anaconda3/bin/python3',str(REMOTE/'submit.py'),'--remote'],
                            text=True,capture_output=True)
    print(result.stdout);print(result.stderr,file=sys.stderr)
    result.check_returncode()
    (ROOT/'results/submission.json').write_text(result.stdout)


if __name__=='__main__':
    main()
