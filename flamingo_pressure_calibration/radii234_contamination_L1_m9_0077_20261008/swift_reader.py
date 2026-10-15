"""Bounded full-file and SWIFT extraction in periodic 4 R200c spheres.

Runtime worker, RAM and wall limits are explicit constants. Gas values are
discarded after physical shell sums, particle-ID hashes and coarse maps.
No production cache or full-particle export is used by the analysis entrypoint.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE / "dependencies"))
import common as C
import flamingo_io as F
import pilot_physics as P
from prepare_study import ROOT, SNAPSHOT, FIELDS
from check_fresh_mask import fresh_mask as make_mask

WORKERS = 8
RAM_LIMIT = 32 * 1024**3
LOGICAL_READ_LIMIT = 320 * 10**9
WALL_LIMIT_SECONDS = 5400


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def load_inputs():
    plan=json.loads((HERE/'metadata_plan.json').read_text())
    with np.load(HERE/'selected_halos.npz') as saved:
        halos={name:saved[name] for name in saved.files}
    cosmology=plan['cosmology']
    scalar=lambda name:float(np.asarray(cosmology[name]).ravel()[0])
    fb=scalar('Omega_b')/(scalar('Omega_m')+scalar('Omega_nu_0'))
    return plan,halos,fb


def worker_init():
    global PLAN, HALOS, FB, SNAP
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    PLAN,HALOS,FB=load_inputs()
    SNAP=None


def direct_file(file_id):
    global SNAP
    started=time.perf_counter(); cpu=time.process_time()
    if SNAP is None:
        SNAP=F.Snapshot(F.Store(ROOT),SNAPSHOT,77,read_from='chunks',chunk_cache_bytes=16*1024**2)
    cells=[int(cell) for cell in HALOS['cells'] if int(SNAP.cell_file[cell])==file_id]
    pieces=[]; rows=0
    for cell in cells:
        raw=SNAP.read_cell_raw(cell,FIELDS)
        data={name:raw[name].astype(np.float64)*SNAP.factor(name) for name in F.GAS_FIELDS}
        data['ParticleIDs']=raw['ParticleIDs']
        data['Hydrogen']=raw['SmoothedElementMassFractions'][:,0].astype(np.float64)
        data['Helium']=raw['SmoothedElementMassFractions'][:,1].astype(np.float64)
        attrs=dict(SNAP.dataset('Pressures').attrs)
        data['Pressures']=raw['Pressures'].astype(np.float64)*C.physical_factor(SNAP.dataset('Pressures'))/10
        rows+=len(data['Masses'])
        for halo in HALOS['pair_halo'][HALOS['pair_cell']==cell]:
            h=int(halo)
            result,_=P.accumulate(data,HALOS['centre_cmpc'][h],HALOS['r_200_crit_mpc'][h],
                PLAN['scale_factor'],SNAP.box_size_cmpc,HALOS['m200c_msun'][h],FB,PLAN['recent_agn_filter'],
                rotations=PLAN['rotation_matrices'] if h in PLAN['rotation_halo_indices'] else None)
            if h not in PLAN['swift_check_indices']:
                # Particle-ID verification is repeated for representative cutouts.
                # Smoothing/radius samples remain exact for every halo.
                result['ids']=np.empty(0,dtype=np.uint64)
            pieces.append((h,result))
            del _

        del raw,data
    # HDF5 allocates one raw-data cache per dataset; release it between files.
    SNAP.close()
    SNAP=None
    return {'file':file_id,'pieces':pieces,'rows':rows,'cpu_seconds':time.process_time()-cpu,
            'wall_seconds':time.perf_counter()-started,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}


def swift_halo(h):
    import swiftsimio as sw
    started=time.perf_counter(); cpu=time.process_time()
    mask=make_mask(PLAN['snapshot_path'],HALOS['centre_cmpc'][h],4*HALOS['r_200_crit_mpc'][h],PLAN['scale_factor'])
    data=sw.load(PLAN['snapshot_path'],mask=mask)
    gas=data.gas
    values={
        # SWIFT labels raw internal lengths as symbolic Mpc; use exact header CGS.
        'Coordinates':np.asarray(gas.coordinates.astype(np.float64).to_comoving())*
            float(np.asarray(PLAN['field_schema']['Coordinates']['attributes'][F.COMOVING_ATTR]).ravel()[0])/P.MPC_CM,
        'Masses':gas.masses.astype(np.float64).to_value('g')/P.MSUN_G,
        'Densities':gas.densities.astype(np.float64).to_physical().to_value('g/cm**3')/(P.MSUN_G/P.MPC_CM**3),
        'ComptonYParameters':gas.compton_yparameters.astype(np.float64).to_physical().to_value('cm**2')/P.MPC_CM**2,
        'ElectronNumberDensities':gas.electron_number_densities.astype(np.float64).to_physical().to_value('cm**-3')*P.MPC_CM**3,
        'SmoothingLengths':np.asarray(gas.smoothing_lengths.astype(np.float64).to_comoving())*
            float(np.asarray(PLAN['field_schema']['SmoothingLengths']['attributes'][F.COMOVING_ATTR]).ravel()[0])/P.MPC_CM,
        'Temperatures':gas.temperatures.astype(np.float64).to_value('K'),
        'LastAGNFeedbackScaleFactors':np.asarray(gas.last_agnfeedback_scale_factors),
        'ParticleIDs':np.asarray(gas.particle_ids),
        'Hydrogen':np.asarray(gas.smoothed_element_mass_fractions.hydrogen),
        'Helium':np.asarray(gas.smoothed_element_mass_fractions.helium),
        'Pressures':gas.pressures.astype(np.float64).to_physical().to_value('Pa'),
    }
    box=float(PLAN['header']['BoxSize'][0])
    result,selected=P.accumulate(values,HALOS['centre_cmpc'][h],HALOS['r_200_crit_mpc'][h],
        PLAN['scale_factor'],box,HALOS['m200c_msun'][h],FB,PLAN['recent_agn_filter'])
    # The two portable cutouts are small local-unit/loading checks. Completeness
    # is established independently above, against the original full snapshot.
    chosen={int(np.argmin(HALOS['m200c_msun'])),int(np.argmin(abs(np.log10(HALOS['m200c_msun'])-14.4)))}
    # Keep only profiles, hashes and maps; no particle cutouts are exported.
    result['candidate_rows']=len(values['Masses'])
    result['cpu_seconds']=time.process_time()-cpu
    result['wall_seconds']=time.perf_counter()-started
    return h,result


def combine(pieces,n_halos):
    sums={name:np.zeros((n_halos,len(P.EDGES)-1)) for name in P.SUM_NAMES}
    sums['aperture']=np.zeros((n_halos,3))
    sums['gas_map']=np.zeros((n_halos,64,64));sums['y_map']=np.zeros((n_halos,64,64))
    smoothing=np.full((n_halos,len(P.EDGES)-1),np.nan)
    ids=[[] for _ in range(n_halos)];radii=[[] for _ in range(n_halos)];hsml=[[] for _ in range(n_halos)]
    for h,result in pieces:
        for name in sums:sums[name][h]+=result[name]
        ids[h].append(result['ids']);radii[h].append(result['radius_samples']);hsml[h].append(result['hsml_samples'])
    hashes=[]
    for h in range(n_halos):
        hashes.append(P.hash_ids(np.concatenate(ids[h])))
        r=np.concatenate(radii[h]);s=np.concatenate(hsml[h])
        for k,(lo,hi) in enumerate(zip(P.EDGES[:-1],P.EDGES[1:])):
            take=(r>=lo)&(r<hi)
            if take.any():smoothing[h,k]=np.median(s[take])
    sums['hsml_median_x']=smoothing
    return sums,hashes


def monitor_resources(stop,record):
    process=psutil.Process()
    cpu={}
    while not stop.wait(.5):
        processes=[process]+process.children(recursive=True)
        rss=0
        for child in processes:
            try:
                rss+=child.memory_info().rss
                times=child.cpu_times();cpu[child.pid]=max(cpu.get(child.pid,0),times.user+times.system)
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        record['peak_tree_rss_bytes']=max(record.get('peak_tree_rss_bytes',0),rss)
        record['sampled_tree_cpu_seconds']=sum(cpu.values())
        if rss>RAM_LIMIT:
            for child in processes[1:]:
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            os.kill(os.getpid(),signal.SIGTERM)

