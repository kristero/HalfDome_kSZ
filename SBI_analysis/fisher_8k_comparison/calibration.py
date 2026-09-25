"""Small matched map calibration for the completed independent-noise 8k data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
FIDUCIAL = np.array([18.1,.497,4.35,.154,-.00865,.0393,-.758,.731,.415])
EDGES = np.append(np.arange(80,7881,200),7980)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path)
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def bin_cl(cl):
    ell=np.arange(7980)
    dl=cl*ell*(ell+1)/(2*np.pi)
    return np.array([np.average(dl[a:b],weights=2*ell[a:b]+1) for a,b in zip(EDGES[:-1],EDGES[1:])])


def initialize(args):
    if args.root.exists():
        raise ValueError('Use a fresh calibration root')
    sys.path.insert(0,str(args.dataset/'code'))
    from worker import verify_frozen
    from prior import JointPrior
    from noise_seeds import split_seeds
    source=verify_frozen(args.dataset)
    done=json.loads((args.dataset/'dataset/complete.json').read_text())
    for name,expected in done['files'].items():
        assert digest(args.dataset/'dataset'/name)==expected,name
    assert done['rows']==8192
    prior=JointPrior(source['prior'])
    assert prior.contains(FIDUCIAL)
    # A small step in the BROAD prior's units; the second step is twice this.
    steps=(prior.high-prior.low)*.002
    tasks=[]
    seeds=np.array([split_seeds(2**30+i) for i in range(args.noise+1)],dtype=np.int64)
    train_seeds=np.load(args.dataset/'dataset/noise_split_seeds.npy')
    all_seeds=np.concatenate([seeds.ravel(),train_seeds.ravel(),[22446,22447]])
    assert len(np.unique(all_seeds))==len(all_seeds)
    # Start covariance batches first so their cached fiducial spectra also
    # validate consistency before most parameter variations finish.
    for first in range(0,args.noise,args.batch):
        ids=list(range(first,min(first+args.batch,args.noise)))
        tasks.append(dict(kind='noise',theta=FIDUCIAL.tolist(),noise_indices=ids,
                          split_seeds=seeds[ids].tolist()))
    tasks.append(dict(kind='observation',theta=FIDUCIAL.tolist(),
                      noise_indices=[args.noise],split_seeds=seeds[-1:].tolist()))
    for j in range(9):
        for multiple in (-2,-1,1,2):
            theta=FIDUCIAL.copy();theta[j]+=multiple*steps[j]
            assert prior.contains(theta),(j,multiple)
            tasks.append(dict(kind='derivative',parameter=j,multiple=multiple,
                              theta=theta.tolist(),noise_indices=[],split_seeds=[]))
    args.root.mkdir(parents=True)
    for directory in ('tasks','scratch','logs','code'):
        (args.root/directory).mkdir()
    for path in HERE.iterdir():
        if path.suffix in ('.py','.jl','.pbs'):
            shutil.copy2(path,args.root/'code'/path.name)
    run=json.loads((args.dataset/'run_config.json').read_text())
    config=dict(dataset=str(args.dataset.resolve()),dataset_manifest_sha256=digest(args.dataset/'manifest.json'),
        source_code_sha256=source['code_sha256'],run=run,prior=source['prior'],steps=steps.tolist(),
        noise_count=args.noise,tasks=tasks,noise_seed_namespace_start=2**30,
        code_sha256={p.name:digest(p) for p in (args.root/'code').iterdir()},
        scope='fixed sky and mask; independent split noise; raw 40-bin D_ell; frozen 8k painter')
    write_json(args.root/'manifest.json',config)
    print('Initialized',len(tasks),'tasks with',args.noise,'independent covariance pairs',flush=True)


def work(args):
    import toml
    root=args.root.resolve()
    config=json.loads((root/'manifest.json').read_text())
    source=Path(config['dataset'])
    assert digest(source/'manifest.json')==config['dataset_manifest_sha256']
    for name,expected in config['source_code_sha256'].items():
        assert digest(source/'code'/name)==expected,name
    for name,expected in config['code_sha256'].items():
        assert digest(root/'code'/name)==expected,name
    sys.path.insert(0,str(source/'code'))
    from prior import KEYS
    campaign=Path(config['run']['campaign'])
    metadata=json.loads((campaign/'preflight/metadata_manifest.json').read_text())
    original=metadata['halfdome_reference']['simulation_request']['command']
    base=dict(p.split('=',1) for p in original if '=' in p and not p.startswith('--'))
    runtime=campaign/'runtime'
    for i,task in enumerate(config['tasks']):
        if i%args.workers!=args.worker:
            continue
        out=root/'tasks'/f'task{i:03d}'
        if (out/'complete.json').exists():
            marker=json.loads((out/'complete.json').read_text())
            assert marker['manifest_sha256']==digest(root/'manifest.json')
            assert marker['spectra_sha256']==digest(out/'spectra.npz')
            continue
        out.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'task{i:03d}_',dir=root/'scratch') as temp:
            temp=Path(temp)
            settings=dict(base,halfdome_path=config['run']['catalogue'],output_dir=str(temp/'raw'),
                cache_dir=str(temp/'cache'),enforce_battaglia_guardrails='false',
                baseline_noise_path=str(campaign/'code/halfdome/other_sims/SO/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt'),
                goal_noise_path=str(campaign/'code/halfdome/other_sims/SO/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt'))
            settings.update({k:format(v,'.17g') for k,v in zip(KEYS,task['theta'])})
            (out/'seeds.toml').write_text(toml.dumps(dict(pairs=task['split_seeds'])))
            env={k:v for k,v in os.environ.items() if not k.startswith(('TSZ_','BATTAGLIA_'))}
            env.update(HALFDOME_SOURCE_DIR=str(campaign/'code/halfdome'),FLAMINGO_CAMPAIGN=str(campaign),
                CAL_OUTPUT=str(out),CAL_FROZEN_CODE=str(source/'code'),CAL_SEEDS_FILE=str(out/'seeds.toml'),
                JULIA_DEPOT_PATH=str(runtime/'depot')+':/home/kristero10/.julia',JULIA_PKG_PRECOMPILE_AUTO='0',
                HDF5_USE_FILE_LOCKING='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                OMP_NUM_THREADS=str(args.threads),LD_LIBRARY_PATH=str(runtime/'julia-1.12.2/lib/julia')+':'+env.get('LD_LIBRARY_PATH',''))
            command=[str(runtime/'julia-1.12.2/bin/julia'),'--startup-file=no',f'--threads={args.threads}',
                '--project='+str(runtime/'julia_env'),str(root/'code/calibrate_maps.jl')]
            command += [k+'='+v for k,v in settings.items()]
            started=time.monotonic()
            print('Starting',i,task['kind'],flush=True)
            with (out/'run.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,timeout=7200)
            clean=np.load(out/'clean.npy')
            assert clean.shape==(7980,) and np.isfinite(clean).all()
            parsed=toml.load(out/'parameters.toml')
            for key,value in zip(KEYS,task['theta']):
                assert parsed[key[len('battaglia_'):]]==value,key
            noisy=np.array([np.load(out/f'noisy_{k+1}.npy') for k in range(len(task['split_seeds']))])
            assert np.isfinite(noisy).all()
            operator=toml.load(out/'complete.toml')
            assert operator['split_seeds']==task['split_seeds']
            np.savez(out/'spectra.npz',clean=bin_cl(clean),
                noisy=np.array([bin_cl(x) for x in noisy]).reshape(-1,40),theta=task['theta'])
            write_json(out/'complete.json',dict(manifest_sha256=digest(root/'manifest.json'),
                spectra_sha256=digest(out/'spectra.npz'),seconds=time.monotonic()-started,
                job_id=os.environ.get('PBS_JOBID'),operator=operator,task=task,command=command))
            print('Completed',i,'seconds',round(time.monotonic()-started),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('init','work'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915'))
    p.add_argument('--noise',type=int,default=32)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--threads',type=int,default=26)
    args=p.parse_args()
    if args.noise<3 or args.batch<1 or not 0<=args.worker<args.workers:
        p.error('Invalid ensemble, batch or worker settings')
    (initialize if args.stage=='init' else work)(args)


if __name__=='__main__':
    main()
