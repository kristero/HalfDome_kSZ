"""Freeze, deploy and submit only resolution preflight jobs, once.

Run locally: python submit.py --submit
Four workers pull the finite list using per-control locks. The independent
256-row diagnostic is deliberately absent from every submission path.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
REMOTE=PurePosixPath('/lustre/work/kristero10/tsz_8192_validation_20260920')
CAMPAIGN=Path('/lustre/work/kristero10/flamingo_tsz_comparison_20260914')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def remote_submit():
    """Runs on idark via stdin; save each accepted job before the next qsub."""
    global REMOTE
    REMOTE=Path(REMOTE)
    import fcntl
    import numpy as np
    import toml
    from worker import atomic_json
    with (REMOTE/'submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=json.loads((REMOTE/'manifest.json').read_text())
        for name,sha in manifest['source_sha256'].items():
            if digest(REMOTE/name)!=sha:raise RuntimeError('Deployment checksum mismatch: '+name)
        origin=Path('/lustre/work/kristero10/so_unbinned_moped_524k_20260914/final_export/unbinned_moped_transform.npz')
        assert digest(origin)==digest(REMOTE/'inputs/frozen_unbinned_moped_transform.npz')
        prepared=Path('/lustre/work/kristero10/tsz_spherical_diagnostic_256_prepared_20260920')
        previous=np.load(prepared/'noise_seeds.npy').ravel()
        plan=json.loads((REMOTE/'plan.json').read_text())
        seeds=[62000000+t['id']*10000+2*i+s for t in plan['tasks']
               for i in range(t['noise_draws']) for s in [1,2]]
        assert len(set(seeds))==len(seeds) and not set(seeds).intersection(map(int,previous))
        assert 12345 not in seeds
        dependencies={}
        for path in sorted((CAMPAIGN/'code/halfdome/tSZ_visuals').glob('*.jl')):
            dependencies[str(path)]=digest(path)
        for path in [CAMPAIGN/'code/process_maps.jl',CAMPAIGN/'runtime/julia_env/Project.toml',
                     CAMPAIGN/'runtime/julia_env/Manifest.toml']:
            dependencies[str(path)]=digest(path)
        atomic_json(REMOTE/'external_dependencies.json',dict(sha256=dependencies,
            frozen_moped_origin=str(origin),unique_noise_seeds=len(seeds),prepared_seed_overlap=0,
            runtime_python=sys.version))
        journal=REMOTE/'submission.json'
        state=json.loads(journal.read_text()) if journal.exists() else dict(
            utc=datetime.now(timezone.utc).isoformat(),root=str(REMOTE),gate=None,workers=[],analysis=None,
            diagnostic_256_submitted=False)
        def submit(pbs,name,dependency=None):
            command=['qsub','-N',name,'-o',str(REMOTE/(name+'.log'))]
            if dependency:command+=['-W','depend='+dependency]
            command.append(str(REMOTE/pbs))
            job=subprocess.check_output(command,text=True).strip()
            if not job.split('.')[0].isdigit():raise RuntimeError('Unexpected qsub response: '+job)
            return job
        if state['gate'] is None:
            state['gate']=submit('gate.pbs','tsz8192_gate');atomic_json(journal,state)
        while len(state['workers'])<4:
            job=submit('worker.pbs','tsz8192_w'+str(len(state['workers'])+1),'afterok:'+state['gate'])
            state['workers'].append(job);atomic_json(journal,state)
        if state['analysis'] is None:
            state['analysis']=submit('analyze.pbs','tsz8192_report','afterany:'+':'.join(state['workers']))
            atomic_json(journal,state)
        print(json.dumps(state,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true')
    parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    if args.remote:
        sys.path.insert(0,str(REMOTE));remote_submit();return
    if not args.submit:raise SystemExit('Use --submit for the explicitly authorized preflight jobs.')
    subprocess.run([sys.executable,str(ROOT/'verify_gate.py')],check=True)
    files=sorted([p for p in ROOT.iterdir() if p.suffix in ['.py','.jl','.pbs','.md']]+
                 [ROOT/'plan.json']+list((ROOT/'inputs').glob('*')))
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files if p.is_file()}
    manifest=dict(utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        scope='80 finite controls; independent 256-row dataset not submitted')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Never overwrite source underneath submitted jobs.
    existing=subprocess.run(['ssh','idark','test','-e',str(REMOTE/'manifest.json')])
    if existing.returncode==0:raise RuntimeError('Remote snapshot exists; use its saved submission journal, do not redeploy.')
    if existing.returncode!=1:raise RuntimeError('Cannot verify remote root')
    with tarfile.open(ROOT/'source.tar.gz','w:gz') as archive:
        for path in files+[ROOT/'manifest.json']:
            if path.is_file():archive.add(path,arcname=str(path.relative_to(ROOT)))
    subprocess.run(['ssh','idark','mkdir','-p',str(REMOTE)],check=True)
    subprocess.run(['scp',str(ROOT/'source.tar.gz'),'idark:'+str(REMOTE/'source.tar.gz')],check=True)
    subprocess.run(['ssh','idark','tar','-xzf',str(REMOTE/'source.tar.gz'),'-C',str(REMOTE)],check=True)
    result=subprocess.run(['ssh','idark','/home/anaconda3/bin/python3',str(REMOTE/'submit.py'),'--remote'],
                          text=True,capture_output=True)
    print(result.stdout);print(result.stderr,file=sys.stderr)
    result.check_returncode()
    state=json.loads(result.stdout)
    (ROOT/'results/submission.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
