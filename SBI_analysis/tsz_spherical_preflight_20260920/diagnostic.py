"""Deadline-aware experimental flat-prior rows; preserve failures and identities."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from design_tests import design,noise_seed,LOW,HIGH,NAMES
from run_fullsky import CAMPAIGN,KEYS
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'diagnostic_256'


def atomic_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def _run_row(row,theta,threads):
    out=RUN/'rows'/f'{row:05d}';out.mkdir(parents=True,exist_ok=True)
    settings=dict(a.split('=',1) for a in json.loads((CAMPAIGN/'preflight/metadata_manifest.json').read_text())['halfdome_reference']['simulation_request']['command'] if '=' in a and not a.startswith('--'))
    noise=CAMPAIGN/'code/halfdome/other_sims/SO'
    settings.update(output_dir=str(out/'raw'),cache_dir=str(out/'cache'),nside='4096',
        interpolator_pad='256',
        halfdome_path='/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5',
        enforce_battaglia_guardrails='false',model_exists='false',reuse_existing_cache='false',
        baseline_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt'),
        goal_noise_path=str(noise/'SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt'))
    settings.update({'battaglia_'+k:format(v,'.17g') for k,v in zip(KEYS,theta)})
    sources=['diagnostic_row.jl','fullsky_test.jl','spherical_truncation_profiles.jl']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}
    seeds=np.load(RUN/'noise_seeds.npy')[row].tolist()
    assert seeds==[noise_seed(row,i,'engineering') for i in [1,2]]
    identity=dict(theta=theta.tolist(),settings=settings,source_sha256=hashes,split_seeds=seeds,
        purpose='experimental pipeline diagnostic, not production-certified')
    identifier=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    marker=out/'status.json'
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous['identity_sha256']!=identifier:raise RuntimeError('Refusing reuse after model/seed/source change')
        if previous['returncode']==0:return
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    for name in sources:shutil.copy2(ROOT/name,snapshot/name)
    atomic_json(out/'request.json',identity)
    runtime=CAMPAIGN/'runtime'
    env=dict(os.environ,FLAMINGO_CAMPAIGN=str(CAMPAIGN),HALFDOME_SOURCE_DIR=str(CAMPAIGN/'code/halfdome'),
        PREFLIGHT_OUTPUT=str(out),PREFLIGHT_GRID_FACTOR='1',DIAGNOSTIC_SPLIT_SEEDS=','.join(map(str,seeds)),
        JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',
        LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS='1',HDF5_USE_FILE_LOCKING='FALSE')
    cmd=[str(runtime/'julia-1.12.2/bin/julia'),'--startup-file=no','--threads='+str(threads),
         '--project='+str(runtime/'julia_env'),str(snapshot/'diagnostic_row.jl')]
    cmd += [k+'='+v for k,v in settings.items()]
    start=time.monotonic()
    with (out/'run.log').open('w') as log:
        code=subprocess.call(['/usr/bin/time','-v','-o',str(out/'time.txt')]+cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(marker,dict(row=row,returncode=code,seconds=time.monotonic()-start,
        identity_sha256=identifier,job=os.environ.get('PBS_JOBID')))
    print(row,code,time.monotonic()-start,flush=True)


def run_row(row,theta,threads):
    import fcntl
    locks=RUN/'locks';locks.mkdir(parents=True,exist_ok=True)
    with (locks/f'{row:05d}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _run_row(row,theta,threads)


def _summarize(count):
    rows=[];clean=[];noisy=[];ids=[]
    ell=np.arange(7980);edges=np.r_[np.arange(80,7881,200),7980]
    def bins(cl):
        dl=cl[:7980]*ell*(ell+1)/(2*np.pi)
        return np.array([np.average(dl[a:b],weights=2*ell[a:b]+1) for a,b in zip(edges[:-1],edges[1:])])
    for row in range(count):
        folder=RUN/'rows'/f'{row:05d}';marker=folder/'status.json'
        status=json.loads(marker.read_text()) if marker.exists() else dict(row=row,returncode=None,
            status='started_incomplete' if (folder/'request.json').exists() else 'not_started')
        rows.append(status)
        if status['returncode']==0:
            clean.append(bins(np.load(folder/'masked_clean_cl.npy')))
            noisy.append(bins(np.load(folder/'masked_noisy_cross_cl.npy')));ids.append(row)
    out=RUN;out.mkdir(exist_ok=True)
    with (out/'dataset.tmp').open('wb') as stream:
        np.savez(stream,row_id=ids,theta=np.load(RUN/'theta_design.npy')[ids],clean_dl=clean,noisy_dl=noisy,
            lower=LOW,upper=HIGH,parameter_order=NAMES)
    (out/'dataset.tmp').replace(out/'dataset.npz')
    atomic_json(out/'summary.json',dict(requested=count,completed=len(ids),rows=rows,
        geometry='sphere4; experimental point painter at raw/output4096',
        inference='Small-sample pipeline and information diagnostic; cannot certify nine-parameter coverage',
        production_certified=False))


def summarize(count):
    import fcntl
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _summarize(count)


def prepare(count):
    RUN.mkdir(parents=True,exist_ok=True)
    theta=design(count)
    seeds=np.array([[noise_seed(i,j,'engineering') for j in [1,2]] for i in range(count)],dtype=np.int64)
    assert len(np.unique(seeds))==2*count
    manifest=dict(count=count,parameter_order=NAMES,lower=LOW.tolist(),upper=HIGH.tolist(),
        prior='independent physical uniforms, no rejection',geometry='sphere4',
        raw_nside=4096,output_nside=4096,beam_fwhm_arcmin=2.,fsky=.4,mask_seed=12345,
        cosmo_h=.68,cosmo_omegab=.049,cosmo_omegac=.261,mass_min_msun=1e12,
        noise_master_seed=20260920,noise_stream='engineering',noise_split_Nell_multiplier=1.,
        cl_lmax=7979,cl_niter=0,cache_grid_factor=1,painting='experimental point painter',
        cache_grid_nodes=[512,256,128],cache_coordinates=['log_theta','log_z','log10_mass'],
        cache_angular_padding=256,cache_theta_min_rad=1.0181517217181794e-11,
        cache_quantity='Physical amplitude times chord-mean pressure',cache_value_floor=1e-300,
        retained_spectra='Per-row masked clean and signed split-cross C_ell, ell=0..7979',
        moped_input=dict(ell_min=80,ell_max=7979,features=7900,binned=False),
        known_limitations=['compact-halo pixel aliasing','inherited angular interpolation floor',
                          'corrected production renderer and nine-parameter calibration remain unvalidated'],
        validation_rows=count//4,training_pool_rows=count-count//4,
        split_rule='last quarter held out; all preprocessing fitted on training only',
        production_certified=False,submission_authorized=False,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['diagnostic.py','diagnostic_row.jl','fullsky_test.jl','spherical_truncation_profiles.jl','design_tests.py']})
    marker=RUN/'manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=manifest:
        if (RUN/'rows').exists() and any((RUN/'rows').iterdir()):
            raise RuntimeError('Rows exist: use a fresh run root after source changes')
        shutil.copy2(marker,RUN/'manifest.previous.json')
    atomic_json(marker,manifest)
    np.save(RUN/'theta_design.npy',theta);np.save(RUN/'noise_seeds.npy',seeds)
    heldout=np.arange(count)>=count-count//4
    np.save(RUN/'heldout_design.npy',heldout)
    print(f'Prepared {count} rows at {RUN}; no simulation or scheduler submission.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=256)
    p.add_argument('--threads',type=int,default=26);p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--stop-new-rows-utc');args=p.parse_args();RUN=args.run_root.resolve()
    if not 0<=args.worker<args.workers:p.error('Require 0 <= worker < workers')
    if args.prepare_only:prepare(args.count);raise SystemExit(0)
    if not (RUN/'manifest.json').exists():raise RuntimeError('Run --prepare-only first')
    manifest=json.loads((RUN/'manifest.json').read_text())
    if manifest['count']!=args.count:raise RuntimeError('Count differs from prepared manifest')
    for name,expected in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Prepared source changed: '+name)
    deadline=datetime.fromisoformat(args.stop_new_rows_utc) if args.stop_new_rows_utc else None
    if not args.summarize_only:
        for row,theta in enumerate(np.load(RUN/'theta_design.npy')):
            if row%args.workers!=args.worker:continue
            if deadline and datetime.now(timezone.utc)>=deadline:break
            run_row(row,theta,args.threads)
            summarize(args.count)
    summarize(args.count)
