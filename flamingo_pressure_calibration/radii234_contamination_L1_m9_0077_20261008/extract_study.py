"""One resumable full-snapshot pass, membership decomposition and SWIFT checks.

Workers save compact file-level accumulators; the parent merges them. No raw
particle cutout is saved. Completed file products survive an interrupted run.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback
import h5py
import numpy as np
import psutil

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'dependencies'))
import common as C
import flamingo_io as F
import pilot_physics as P
import measure as M
from prepare_study import ROOT,SNAPSHOT,MEMBERSHIP,FIELDS


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def load_inputs():
    plan=json.loads((HERE/'metadata_plan.json').read_text())
    halo=dict(np.load(HERE/'selected_halos.npz'))
    return plan,halo,plan['baryon_fraction']


def initialize():
    global PLAN,H,FB
    sys.dont_write_bytecode=True
    resource.setrlimit(resource.RLIMIT_AS,(5*1024**3,5*1024**3))
    PLAN,H,FB=load_inputs()


def map_shape():return (15,2,2,3,4,3,M.NPIX,M.NPIX)


def read_file(file_id, gate_only=False):
    started=time.perf_counter();cpu=time.process_time()
    path=HERE/'file_products'/f'file_{file_id:03d}.npz'
    receipt=path.with_suffix('.json')
    if path.exists() and receipt.exists() and not gate_only:
        return json.loads(receipt.read_text())
    snap=F.Snapshot(F.Store(ROOT),SNAPSHOT,77,read_from='chunks',chunk_cache_bytes=8*1024**2)
    cells=[int(c) for c in H['cells'] if snap.cell_file[c]==file_id]
    if gate_only:cells=cells[:1]
    maps=np.zeros(map_shape());views=np.zeros((15,11,3,4,3,M.NB))
    parts={};ids={};candidate_rows=0
    membership_path=MEMBERSHIP/f'membership_0077.{file_id}.hdf5'
    with h5py.File(membership_path,'r',rdcc_nbytes=8*1024**2) as member_file:
        membership=member_file['PartType0/GroupNr_bound']
        assert len(membership)==len(snap._chunk(file_id)['PartType0/Masses'])
        for cell in cells:
            raw=snap.read_cell_raw(cell,FIELDS)
            data={name:raw[name].astype(np.float64)*snap.factor(name) for name in F.GAS_FIELDS}
            data.update(ParticleIDs=raw['ParticleIDs'],Hydrogen=raw['SmoothedElementMassFractions'][:,0].astype(float),
                        Helium=raw['SmoothedElementMassFractions'][:,1].astype(float),
                        Pressures=raw['Pressures'].astype(float)*C.physical_factor(snap.dataset('Pressures'))/10)
            assert all(np.isfinite(v).all() for v in data.values())
            assert np.all(data['Masses']>0)&np.all(data['Densities']>0)
            assert np.all(data['ComptonYParameters']>=0)&np.all(data['ElectronNumberDensities']>=0)
            lo=int(snap.cell_offset[cell]);size=int(snap.counts[cell])
            owner=np.asarray(membership[lo:lo+size],dtype=np.int64)
            candidate_rows+=size
            k=int(np.searchsorted(H['csr_cells'],cell))
            assert H['csr_cells'][k]==cell
            for h in H['csr_halo'][H['csr_ptr'][k]:H['csr_ptr'][k+1]]:
                h=int(h);radius=H['r_200_crit_mpc'][h]
                delta=data['Coordinates']-H['centre_cmpc'][h]
                delta-=snap.box_size_cmpc*np.rint(delta/snap.box_size_cmpc)
                inside=np.einsum('ij,ij->i',delta,delta)<=(4*radius/PLAN['scale_factor'])**2
                selected={name:v[inside] for name,v in data.items()}
                sat=H['satellite_ids'][H['satellite_ptr'][h]:H['satellite_ptr'][h+1]]
                one={'centre_cmpc':H['centre_cmpc'][h],'radius':radius,'mass':H['m200c_msun'][h],
                     'id':H['halo_index'][h],'satellites':sat,'a':PLAN['scale_factor'],'box':snap.box_size_cmpc}
                item=M.accumulate(selected,owner[inside],one,FB,PLAN['recent_agn_filter'],PLAN['rotation_matrices'])
                if gate_only:
                    empty=M.accumulate({name:v[:0] for name,v in selected.items()},owner[:0],one,FB,PLAN['recent_agn_filter'],PLAN['rotation_matrices'])
                    assert empty['sums'].dtype==np.float64
                    for key in ('sums','hs_hist','projected'):
                        joined=np.zeros_like(empty[key]);joined+=empty[key];joined+=item[key]
                        assert np.array_equal(joined,item[key]),key
                    reference,_=P.accumulate(selected,one['centre_cmpc'],radius,one['a'],one['box'],one['mass'],FB,PLAN['recent_agn_filter'])
                    for j,name in enumerate(P.SUM_NAMES):
                        assert np.allclose(item['sums'][:,j].sum(axis=0),reference[name],rtol=2e-10,atol=1e-25),name
                    assert np.allclose(item['maps'][0,2,:,0].sum(axis=0),reference['y_map'],rtol=2e-10,atol=1e-25)
                    assert np.allclose(item['maps'][0,2,:,2].sum(axis=0),reference['gas_map']*800*np.pi/3,rtol=2e-10,atol=1e-25)
                if h not in parts:
                    parts[h]={'sums':np.zeros(item['sums'].shape,dtype=np.float64),
                              'hs_hist':np.zeros(item['hs_hist'].shape,dtype=np.uint32),
                              'projected':np.zeros(item['projected'].shape,dtype=np.float64)}
                for key in parts[h]:parts[h][key]+=item[key]
                b=int(H['mass_bin'][h]);maps[b,0]+=item['maps']
                if H['disturbed_proxy'][h]:maps[b,1]+=item['maps']
                views[b]+=item['view_profiles']
                if h in PLAN['swift_check_indices']:ids.setdefault(h,[]).append(item['ids'])
            del raw,data,owner
    snap.close()
    rows=np.array(sorted(parts),dtype=np.int64)
    arrays={'halo_rows':rows,'map_sums':maps,'view_profile_sums':views}
    arrays.update({key:np.array([parts[int(h)][key] for h in rows]) for key in ('sums','hs_hist','projected')})
    id_array=np.concatenate([np.concatenate(ids[h]) for h in sorted(ids)]) if ids else np.empty(0,dtype=np.uint64)
    arrays.update(id_halo_rows=np.array(sorted(ids)),id_ptr=np.r_[0,np.cumsum([sum(len(x) for x in ids[h]) for h in sorted(ids)])],ids=id_array)
    result={'file':int(file_id),'candidate_rows':candidate_rows,'halo_parts':len(rows),
            'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-started,
            'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'gate_only':gate_only}
    if gate_only:
        result['selected_particle_entries']=int(arrays['sums'][:,:,0].sum())
        assert result['selected_particle_entries']>0, 'Choose a non-empty cell for the gate'
        result['all_shell_estimators_and_original_maps_match_reference']=True
        result['empty_then_populated_cell_merge_passed']=True
        return result
    temporary=path.with_suffix('.temporary.npz')
    np.savez_compressed(temporary,**arrays);temporary.replace(path)
    result['product_bytes']=path.stat().st_size
    write_json(receipt,result)
    return result


def monitor(stop,record,limit):
    process=psutil.Process();usage={}
    while not stop.wait(1):
        tree=[process]+process.children(recursive=True);rss=0
        for p in tree:
            try:
                rss+=p.memory_info().rss;t=p.cpu_times();usage[p.pid]=max(usage.get(p.pid,0),t.user+t.system)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        record['peak_tree_rss_bytes']=max(record.get('peak_tree_rss_bytes',0),rss)
        record['sampled_cpu_seconds']=sum(usage.values())
        if rss>limit:
            for p in tree[1:]:
                try:p.terminate()
                except psutil.NoSuchProcess:pass
            os.kill(process.pid,signal.SIGTERM)


def merge_products(files,halo):
    n=len(halo['halo_index'])
    sums=np.zeros((n,4,len(P.SUM_NAMES),M.NB))
    hs=np.zeros((n,M.NB,len(M.HS_EDGES)-1),dtype=np.uint32)
    projected=np.zeros((n,2,3,4,3,M.NB))
    maps=np.zeros(map_shape());views=np.zeros((15,11,3,4,3,M.NB));ids={}
    for file_id in sorted(files):
        with np.load(HERE/'file_products'/f'file_{file_id:03d}.npz') as p:
            rows=p['halo_rows'];sums[rows]+=p['sums'];hs[rows]+=p['hs_hist'];projected[rows]+=p['projected']
            maps+=p['map_sums'];views+=p['view_profile_sums']
            for i,h in enumerate(p['id_halo_rows']):
                ids.setdefault(int(h),[]).append(p['ids'][p['id_ptr'][i]:p['id_ptr'][i+1]])
    hmedian=M.smoothing_upper_median(hs);del hs
    hashes={str(h):P.hash_ids(np.concatenate(parts)) for h,parts in ids.items()}
    del ids
    total=sums.sum(axis=1)
    all_sums={name:total[:,i] for i,name in enumerate(P.SUM_NAMES)}
    profiles=P.physical_profiles(all_sums,halo,PLAN['baryon_fraction'])
    # Component profiles share the all-gas halo normalization and geometric volume.
    volume=4*np.pi/3*halo['r_200_crit_mpc'][:,None]**3*np.diff(P.EDGES**3)
    rho_c=profiles['critical_density_msun_mpc3']
    pressure=sums[:,:,P.SUM_NAMES.index('y')]*P.electron_mass*P.c**2/P.SIGMA_T/(volume[:,None]*P.MPC_M)
    pressure/=profiles['p200_pa'][:,None,None]
    density=sums[:,:,P.SUM_NAMES.index('electrons')]*P.MU_XGP_MSUN/(volume[:,None]*rho_c[:,None,None]*FB)
    gas=sums[:,:,P.SUM_NAMES.index('mass')]/(volume[:,None]*rho_c[:,None,None]*FB)
    np.savez_compressed(HERE/'measurements.npz',sums=sums,hsml_median_upper_x=hmedian,
        pressure_components=pressure,electron_components=density,gas_components=gas,
        projected_profiles=projected,edges=P.EDGES,**profiles)
    np.savez_compressed(HERE/'projection_stacks.npz',map_sums=maps,view_profile_sums=views,
        bin_counts=np.bincount(halo['mass_bin'],minlength=15),
        disturbed_counts=np.bincount(halo['mass_bin'][halo['disturbed_proxy']],minlength=15),
        radii=M.RADII,rotations=PLAN['rotation_matrices'],extent=M.EXTENT)
    write_json(HERE/'particle_hashes.json',hashes)
    return hashes


def swift_checks():
    # Use the prior independently implemented loader with this study's new inputs.
    import swift_reader as R
    R.PLAN,R.HALOS,R.FB=PLAN,H,FB
    receipts=[];arrays=[];hashes=[]
    for done,h in enumerate(PLAN['swift_check_indices']):
        _,item=R.swift_halo(h)
        receipts.append({key:item[key] for key in ('candidate_rows','cpu_seconds','wall_seconds')})
        arrays.append(np.stack([item[name] for name in P.SUM_NAMES]));hashes.append(P.hash_ids(item['ids']))
        write_json(HERE/'progress.json',{'phase':'SWIFT checks','done':done+1,'total':len(PLAN['swift_check_indices'])})
    np.savez_compressed(HERE/'swiftsimio_profiles.npz',sums=arrays,halo_rows=PLAN['swift_check_indices'],edges=P.EDGES)
    full=np.load(HERE/'measurements.npz')['sums'].sum(axis=1)[PLAN['swift_check_indices']]
    arrays=np.array(arrays);relative=np.abs(full-arrays)/np.maximum(np.maximum(np.abs(full),np.abs(arrays)),1e-100)
    direct=json.loads((HERE/'particle_hashes.json').read_text())
    matching=[direct[str(h)]==hashes[i] for i,h in enumerate(PLAN['swift_check_indices'])]
    assert all(matching)
    assert np.array_equal(full[:,0],arrays[:,0])
    result={'all_particle_hashes_match':all(matching),'checked_halos':len(matching),
            'max_relative_shell_difference':{name:float(relative[:,i].max()) for i,name in enumerate(P.SUM_NAMES)},'receipts':receipts}
    write_json(HERE/'swift_checks.json',result)
    return result


def main():
    global PLAN,H,FB
    PLAN,H,FB=load_inputs();cfg=json.loads((HERE/'study_config.json').read_text())
    assert not (HERE/'complete.json').exists()
    assert PLAN['logical_bytes_one_direct_pass']+PLAN['swift_logical_bytes_upper_bound']<300e9
    os.nice(10);started=time.perf_counter()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('180-minute stop')))
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(MemoryError('40-GiB RSS stop')))
    signal.alarm(cfg['wall_limit_minutes']*60)
    record={'started_UTC':datetime.now(timezone.utc).isoformat(),'workers':cfg['workers'],
            'RAM_limit_GiB':cfg['RAM_limit_GiB'],'wall_limit_minutes':cfg['wall_limit_minutes'],
            'logical_read_bytes':PLAN['logical_bytes_one_direct_pass']+PLAN['swift_logical_bytes_upper_bound'],
            'physical_NFS_bytes_measured':False,'scheduler_jobs_submitted':0,'completed':False}
    stop=threading.Event();watch=threading.Thread(target=monitor,args=(stop,record,cfg['RAM_limit_GiB']*1024**3),daemon=True);watch.start()
    (HERE/'file_products').mkdir(exist_ok=True)
    try:
        snap=F.Snapshot(F.Store(ROOT),SNAPSHOT,77,read_from='chunks',chunk_cache_bytes=1024**2)
        files=sorted(int(x) for x in np.unique(snap.cell_file[H['cells']]))
        snap.close();receipts=[]
        if not (HERE/'measurements.npz').exists():
            with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
                todo=iter(files);pending={pool.submit(read_file,next(todo)) for _ in range(min(cfg['workers'],len(files)))}
                while pending:
                    future=next(as_completed(pending));pending.remove(future)
                    receipt=future.result();receipts.append(receipt)
                    write_json(HERE/'progress.json',{'phase':'full-snapshot extraction','done':len(receipts),'total':len(files),'elapsed_seconds':time.perf_counter()-started})
                    print(json.dumps({'file_done':receipt['file'],'completed_files':len(receipts),'wall_seconds':time.perf_counter()-started}),flush=True)
                    next_file=next(todo,None)
                    if next_file is not None:pending.add(pool.submit(read_file,next_file))
            write_json(HERE/'file_receipts.json',receipts)
            merge_products(files,H)
        checks=swift_checks()
        stop.set();watch.join(timeout=2)
        record.update(completed=True,wall_seconds=time.perf_counter()-started,
            main_cpu_seconds=time.process_time(),children_cpu_seconds=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            SWIFT_checks_passed=checks['all_particle_hashes_match'])
        write_json(HERE/'complete.json',record);print(json.dumps(record,indent=2),flush=True)
    except BaseException as error:
        stop.set();write_json(HERE/'failed.json',{'error':str(error),'traceback':traceback.format_exc(),'resources':record});raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
