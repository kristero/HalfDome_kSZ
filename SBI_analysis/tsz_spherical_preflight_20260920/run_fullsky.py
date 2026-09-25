"""Run clean spherical resolution controls without touching the completed 8k."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=Path('/lustre/work/kristero10/flamingo_tsz_comparison_20260914')
KEYS=['P0_amp','x_c_amp','beta_amp','P0_alpha_m','x_c_alpha_m',
      'beta_alpha_m','P0_alpha_z','x_c_alpha_z','beta_alpha_z']


def run(case,nside,factor):
    out=ROOT/'fullsky'/case/f'nside{nside}_grid{factor}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text())['returncode']==0:
        return
    request=json.loads((CAMPAIGN/'preflight/metadata_manifest.json').read_text())
    old=request['halfdome_reference']['simulation_request']['command']
    settings=dict(a.split('=',1) for a in old if '=' in a and not a.startswith('--'))
    theta=json.loads((ROOT/'inputs/cases.json').read_text())[case]
    settings.update({'battaglia_'+k:format(v,'.17g') for k,v in zip(KEYS,theta)})
    noise=CAMPAIGN/'code/halfdome/other_sims/SO'
    settings.update(output_dir=str(out/'raw'),cache_dir=str(out/'cache'),nside=str(nside),
        halfdome_path='/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5',
        enforce_battaglia_guardrails='false',model_exists='false',reuse_existing_cache='false',
        baseline_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt'),
        goal_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt'))
    runtime=CAMPAIGN/'runtime'
    threads=os.environ.get('PREFLIGHT_THREADS','4')
    env=dict(os.environ,FLAMINGO_CAMPAIGN=str(CAMPAIGN),
        HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'),PREFLIGHT_OUTPUT=str(out),
        PREFLIGHT_GRID_FACTOR=str(factor),JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
        LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS=threads,MKL_NUM_THREADS='1',
        HDF5_USE_FILE_LOCKING='FALSE',JULIA_PKG_PRECOMPILE_AUTO='0')
    cmd=[str(runtime/'julia-1.12.2/bin/julia'),'--startup-file=no','--threads='+threads,
         '--project='+str(runtime/'julia_env'),str(ROOT/'fullsky_test.jl')]
    cmd += [k+'='+v for k,v in settings.items()]
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        code=subprocess.call(['/usr/bin/time','-v','-o',str(out/'time.txt')]+cmd,
                             env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/'status.json').write_text(json.dumps(dict(returncode=code,seconds=time.monotonic()-start,
        command=cmd,theta=theta,job=os.environ.get('PBS_JOBID')),indent=2)+'\n')
    print(case,nside,factor,code,time.monotonic()-start,flush=True)
    if code:raise RuntimeError(f'{case}/{nside} failed; inspect {out}/run.log')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',default=['Battaglia12'])
    p.add_argument('--nsides',nargs='+',type=int,default=[4096,8192,16384])
    p.add_argument('--factor',type=int,default=1);args=p.parse_args()
    for case in args.cases:
        for nside in args.nsides:run(case,nside,args.factor)
