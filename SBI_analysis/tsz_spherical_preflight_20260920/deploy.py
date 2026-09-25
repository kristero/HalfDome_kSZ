"""Snapshot isolated test sources and submit only explicitly requested PBS tests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parent
REMOTE='/lustre/work/kristero10/tsz_spherical_preflight_20260920'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--submit',choices=['resolution.pbs']);args=p.parse_args()
    shutil.copy2(ROOT.parents[1]/'truncation_comparison/spherical_truncation_profiles.jl',ROOT/'spherical_truncation_profiles.jl')
    shutil.copy2(ROOT.parent/'tsz_guardrail_study/inputs/cases.json',ROOT/'inputs/cases.json')
    files=[p for p in ROOT.iterdir() if p.suffix in ['.py','.jl','.pbs','.md']]+[ROOT/'inputs/cases.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'inputs/deployed_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with tarfile.open(ROOT/'source.tar.gz','w:gz') as tar:
        for p in files+[ROOT/'inputs/deployed_sha256.json']:tar.add(p,arcname=str(p.relative_to(ROOT)))
    subprocess.run(['ssh','idark','mkdir','-p',REMOTE],check=True)
    subprocess.run(['scp',str(ROOT/'source.tar.gz'),'idark:'+REMOTE+'/source.tar.gz'],check=True)
    subprocess.run(['ssh','idark','tar','-xzf',REMOTE+'/source.tar.gz','-C',REMOTE],check=True)
    if args.submit:
        job=subprocess.check_output(['ssh','idark','qsub','-o',REMOTE+'/scheduler_resolution.log',REMOTE+'/'+args.submit],text=True).strip()
        result=dict(job_id=job,pbs=args.submit,source_sha256=hashes)
        (ROOT/'results'/('submission_'+job.split('.')[0]+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(job)
