"""Five independent idark workers: matched-volume maps, PDFs and power spectra."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import h5py
import numpy as np
from numpy.polynomial.chebyshev import chebval2d
import physics as P
import painting as M
import field_statistics as S

ROOT=Path(__file__).resolve().parent
SIDE=250.
FITS=None
SEM_FITS=None
REGIONS=None
ORIGINS=None


def cases():
    result=[]
    for fit in [1,2,3,4]:
        for cut in [1,2,3,4]:
            result.append(dict(name='fit%d_cut%d'%(fit,cut),p_fit=fit,e_fit=fit,p_cut=cut,e_cut=cut))
    for cut in [1,2,3,4]:
        result.append(dict(name='battaglia_cut%d'%cut,p_fit=2,e_fit=3,p_cut=cut,e_cut=cut,baseline=True))
    base=dict(p_fit=2,e_fit=3,p_cut=4,e_cut=3)
    result.append(dict(base,name='main_hybrid'))
    result.append(dict(base,name='no_other_hybrid',component='no_other'))
    result.append(dict(base,name='normalization_control',normalization_control=True))
    result.append(dict(base,name='sem_weight_control',sem_weights=True))
    for seed in [42,43]:
        result.append(dict(base,name='paired_scatter_%d'%seed,scatter_seed=seed))
    result.append(dict(base,name='high_mass_cut_15p1',maximum_logmass=15.1))
    for pop in ['halfdome','flamingo']:
        for case in result:
            yield pop,case
    for i in range(3):
        yield 'halfdome',dict(base,name='mass_mapping_alternative_%d'%i,mapping_alternative=i)


def ensure_regions(pop):
    """One streamed catalogue pass, shared through a globally atomic mkdir lock."""
    output=ROOT/('region_halos_'+pop+'.h5')
    with region_lock(pop) as acquired:
        if not acquired:
            return
        source=ROOT/('halos_'+pop+'.h5')
        groups=[{k:[] for k in ['mass','position','velocity','row']} for _ in range(8)]
        with h5py.File(source,'r') as f:
            box=float(f.attrs['box_mpc'])
            origins=np.array(list(itertools.product([box*.25-SIDE/2,box*.75-SIDE/2],repeat=3)))
            n=len(f['mass'])
            for start in range(0,n,1_000_000):
                stop=min(start+1_000_000,n)
                arrays={k:f[k][start:stop] for k in groups[0]}
                radius=P.scales(arrays['mass'],P.TARGET)[0]*5
                pos=arrays['position']
                for r,origin in enumerate(origins):
                    take=np.all((pos+radius[:,None]>=origin)&(pos-radius[:,None]<origin+SIDE),axis=1)
                    for key,values in arrays.items():
                        groups[r][key].append(values[take])
            attrs=dict(f.attrs)
        temporary=output.with_name(output.name+'.%d.temporary'%os.getpid())
        with h5py.File(temporary,'w') as f:
            f.create_dataset('origins',data=origins)
            for r,group in enumerate(groups):
                target=f.create_group(str(r))
                for key,values in group.items():
                    target.create_dataset(key,data=np.concatenate(values),compression='gzip',compression_opts=1)
            for key,value in attrs.items():
                f.attrs[key]=value
            f.attrs['source_catalogue_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        temporary.replace(output)


@contextmanager
def region_lock(pop):
    """Lustre can mount flock as node-local; mkdir remains cluster-global."""
    output=ROOT/('region_halos_'+pop+'.h5')
    lock=ROOT/('regions_'+pop+'.lockdir')
    started=time.time()
    while True:
        if output.exists():
            yield False
            return
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.time()-started>600:
                raise RuntimeError('Region cache lock timed out; inspect '+str(lock/'owner.json'))
            time.sleep(.5)
    try:
        P.save_json(lock/'owner.json',dict(pid=os.getpid(),host=os.uname().nodename))
        yield True
    finally:
        (lock/'owner.json').unlink()
        lock.rmdir()


def load_regions(pop):
    ensure_regions(pop)
    with h5py.File(ROOT/('region_halos_'+pop+'.h5'),'r') as f:
        origins=f['origins'][:]
        regions=[{k:f[str(r)][k][:] for k in ['mass','position','velocity','row']} for r in range(8)]
    return origins,regions


def mapping_alternative(catalogue,index):
    d=json.loads((ROOT/'mass_mapping.json').read_text())
    grid=np.load(ROOT/'mass_mapping_validation.npz')
    x=np.interp(np.log10(catalogue['mass']*P.TARGET['h']),grid['z0_logm200c'],grid['grid_raw_logmass'])
    coeff=np.asarray(d['alternatives'][index]['coefficients'])
    assigned=x+chebval2d((x-14.2)/1.6,np.full_like(x,-1.),coeff)
    mass=10**assigned/P.TARGET['h']
    keep=(mass>=1e13)&(mass<10**15.5)
    result={k:v[keep] for k,v in catalogue.items()}
    result['mass']=mass[keep]
    return result,dict(excluded_from_fixed_cohort=int((~keep).sum()),
        max_absolute_mass_change_percent=float(np.max(np.abs(mass/catalogue['mass']-1))*100),
        scope='Numerical reconstruction sensitivity on original fixed cohort; not a calibrated confidence bound.')


def region_task(task):
    pop,case,r=task
    started=time.time()
    cat=REGIONS[pop][r]
    case=dict(case)
    # Measured FLAMINGO closure uses native source cosmology. The control
    # puts it in the common target normalization; HalfDome does the reverse.
    case['source_cosmology']=(pop=='flamingo') != bool(case.get('normalization_control',False))
    mapping_info={}
    if 'mapping_alternative' in case:
        cat,mapping_info=mapping_alternative(cat,case['mapping_alternative'])
    if 'maximum_logmass' in case:
        take=np.log10(cat['mass'])<case['maximum_logmass']
        cat={k:v[take] for k,v in cat.items()}
    fits=SEM_FITS if case.get('sem_weights') else FITS
    sums,cube,info=M.region_fields(cat,fits,case,ORIGINS[pop][r])
    result=dict(pop=pop,case=case,region=r,sums=sums,statistics={},info=info,mapping=mapping_info)
    result['info']['halo_mass_histogram']=np.histogram(np.log10(cat['mass']),bins=np.arange(13.,15.6,.2))[0].tolist()
    for q,name in [(0,'pressure'),(1,'electrons')]:
        values=cube[q].astype(float)/(SIDE/256)**3
        if q==0:
            values*=P.Y_TO_P_VOLUME
        h,a=S.probability(values,name)
        hc,ac=S.probability(S.rebin_cube(values),name)
        result['statistics'][name]=dict(pdf=h,atoms=a,coarse_pdf=hc,coarse_atoms=ac,
            mean=float(values.mean()),rms=float(np.sqrt(np.mean(values**2))))
    observations=P.map_observables(sums,(SIDE/512)**2)
    for q,name in enumerate(['y','tau','dm','deltaT']):
        values=observations[q]
        h,a=S.probability(values,name)
        k,power,counts=S.power2d(values,SIDE)
        result['statistics'][name]=dict(pdf=h,atoms=a,power=power,
            mean=float(values.mean()),rms=float(np.sqrt(np.mean(values**2))))
    result.update(k=k,power_counts=counts,wall_seconds=time.time()-started)
    return result


def save_case(pop,case,results):
    output=ROOT/'cases'/(pop+'_'+case['name']+'.h5')
    output.parent.mkdir(exist_ok=True)
    with h5py.File(output,'w') as f:
        f.create_dataset('region_sums',data=np.stack([r['sums'] for r in results]),compression='gzip',compression_opts=1)
        f.create_dataset('origins',data=ORIGINS[pop])
        group=f.create_group('statistics')
        for name in results[0]['statistics']:
            for key in results[0]['statistics'][name]:
                group.create_dataset(name+'_'+key,data=[r['statistics'][name][key] for r in results])
        group.create_dataset('k_per_mpc',data=results[0]['k'])
        group.create_dataset('power_mode_counts',data=results[0]['power_counts'])
        f.attrs.update(population=pop,case=json.dumps(case),side_mpc=SIDE,map_n=512,grid_n=256,
            gas_velocity='halo bulk peculiar velocity',particle_grid_assignment='conservative NGP-matched finite halo spheres')
        f.attrs['region_info']=json.dumps([r['info'] for r in results])
        f.attrs['mapping_sensitivity']=json.dumps([r['mapping'] for r in results])
    P.save_json(output.with_suffix('.json'),dict(completed=True,population=pop,case=case,
        summed_task_wall_seconds=sum(r['wall_seconds'] for r in results),bytes=output.stat().st_size,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ['physics.py','painting.py','run_comparisons.py','fits.json']}))


def whole_task(task):
    pop,case,start,stop=task
    with h5py.File(ROOT/('halos_'+pop+'.h5'),'r') as f:
        cat={k:f[k][start:stop] for k in ['mass','position','velocity','row']}
        side=float(f.attrs['box_mpc'])
    case=dict(case,source_cosmology=(pop=='flamingo'))
    sums,_,info=M.region_fields(cat,FITS,case,np.zeros(3),side=side,map_n=1024,whole=True)
    return sums,info


def whole_overviews(workers):
    for pop in ['halfdome','flamingo']:
        with h5py.File(ROOT/('halos_'+pop+'.h5'),'r') as f:
            n=len(f['mass']);side=float(f.attrs['box_mpc'])
        for case in [dict(name='battaglia_cut4',p_fit=2,e_fit=3,p_cut=4,e_cut=4,baseline=True),
                     dict(name='main_hybrid',p_fit=2,e_fit=3,p_cut=4,e_cut=3)]:
            tasks=[(pop,case,start,min(start+1_000_000,n)) for start in range(0,n,1_000_000)]
            total=np.zeros((3,1024,1024));expected=np.zeros(2)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for sums,info in pool.map(whole_task,tasks):
                    total+=sums;expected+=info['integrated_full_spheres']
            errors=np.array([total[0].sum(),total[1].sum()])/expected-1
            assert np.max(np.abs(errors))<1e-9
            output=ROOT/('overview_'+pop+'_'+case['name']+'.h5')
            with h5py.File(output,'w') as f:
                f.create_dataset('sums',data=total,compression='gzip',compression_opts=1)
                f.attrs.update(box_mpc=side,pixels=1024,halo_count=n,case=json.dumps(case),periodic=True)
            P.save_json(output.with_suffix('.json'),dict(completed=True,spherical_integral_relative_errors=errors.tolist(),
                bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest()))


def main():
    global FITS,SEM_FITS,REGIONS,ORIGINS
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--workers',type=int,default=26)
    args=parser.parse_args()
    started=time.time()
    assert (ROOT/'fits_sd_ready.json').exists()
    FITS=json.loads((ROOT/'fits.json').read_text())
    SEM_FITS=json.loads((ROOT/'fits_sem_weighting_control.json').read_text())
    from verify_numerics import run
    P.save_json(ROOT/('worker_%d_numerical_gate.json'%args.worker),run(ROOT,FITS))
    ORIGINS={};REGIONS={}
    for pop in ['halfdome','flamingo']:
        ORIGINS[pop],REGIONS[pop]=load_regions(pop)
    own=[task for i,task in enumerate(cases()) if i%5==args.worker]
    P.save_json(ROOT/('worker_%d_plan.json'%args.worker),dict(cases=[dict(population=p,**c) for p,c in own]))
    tasks=[(pop,case,r) for pop,case in own for r in range(8)]
    grouped={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(region_task,tasks):
            key=(result['pop'],result['case']['name'])
            grouped.setdefault(key,[]).append(result)
            if len(grouped[key])==8:
                save_case(result['pop'],next(c for p,c in own if (p,c['name'])==key),grouped.pop(key))
                P.save_json(ROOT/('worker_%d_progress.json'%args.worker),dict(last_completed=key,wall_seconds=time.time()-started))
    if args.worker==4:
        whole_overviews(args.workers)
    P.save_json(ROOT/('worker_%d_complete.json'%args.worker),dict(completed=True,case_count=len(own),
        wall_seconds=time.time()-started,max_parent_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))


if __name__=='__main__':
    main()
