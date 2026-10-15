"""One streamed full-gas pass: conservative maps, volume grids and gas PDFs.

Four workers read disjoint snapshot files. Raw particles never leave STRW.
The deposits are particle-centre (NGP) estimates; resolution checks use the
same assignment and physical pixel/cell size for measured and painted fields.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import resource
import threading
import time
import h5py
import numpy as np
import psutil
import physics as P

ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path('/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077')
SOAP = SNAPSHOT.parent.parent/'SOAP-HBT/halo_properties_0077.hdf5'
FIELDS = ['Coordinates','Velocities','Masses','Densities','ElectronNumberDensities','ComptonYParameters']
SIDE = 250.
ORIGINS = np.array(list(itertools.product([125.,625.],repeat=3)))
MAP_N = 512
GRID_N = 256
WHOLE_N = 1024
PE_EDGES = np.linspace(-23,-9,281)
NE_EDGES = np.linspace(-12,3,301)


def physical_factor(dataset):
    return float(np.asarray(dataset.attrs['Conversion factor to physical CGS (including cosmological corrections)']).ravel()[0])


def extract_halos():
    """Export distinct FLAMINGO centrals in the common physical mass interval."""
    with h5py.File(SOAP,'r') as f:
        central = f['InputHalos/IsCentral'][:] == 1
        d = f['SO/200_crit/TotalMass']
        mass = d[:] * physical_factor(d) / (P.MSUN_KG*1000)
        rows = np.flatnonzero(central & (mass >= 1e13) & (mass < 10**15.5))
        # Sequential column reads avoid a very large HDF5 fancy selection.
        position = f['InputHalos/HaloCentre'][:][rows] * physical_factor(f['InputHalos/HaloCentre']) / P.MPC_CM
        velocity = f['BoundSubhalo/CentreOfMassVelocity'][:][rows] * physical_factor(f['BoundSubhalo/CentreOfMassVelocity']) / 1e5
    with h5py.File(ROOT/'halos_flamingo.h5','w') as f:
        for key,values in dict(mass=mass[rows],position=position,velocity=velocity,row=rows).items():
            f.create_dataset(key,data=values,compression='gzip',compression_opts=1)
        f.attrs.update(box_mpc=1000.,redshift=0.,mass_definition='SOAP M200c',mass_unit='physical Msun',
            position_unit='physical Mpc',velocity_unit='peculiar km/s',min_mass_msun=1e13,max_mass_msun=10**15.5)


def worker(task):
    number,files = task
    started = time.time()
    cpu = time.process_time()
    whole = np.zeros((3,WHOLE_N,WHOLE_N),float)
    maps = np.zeros((8,3,MAP_N,MAP_N),float)
    grids = np.zeros((8,3,GRID_N,GRID_N,GRID_N),np.float32)
    hist = np.zeros((2,300),float)
    # Pressure has 280 bins; the last 20 slots are explicitly unused.
    atom = np.zeros((2,3),float)  # zero, positive-underflow, positive-overflow volume
    totals = np.zeros(6,float)  # gas rows, Y, n_e V, n_e v_z V, particle volume, gas mass
    logical = stored = 0
    for path in files:
        with h5py.File(path,'r',rdcc_nbytes=32*1024**2) as f:
            gas = f['PartType0']
            ds = {k:gas[k] for k in FIELDS}
            conv = {k:physical_factor(d) for k,d in ds.items()}
            logical += sum(d.size*d.dtype.itemsize for d in ds.values())
            stored += sum(d.id.get_storage_size() for d in ds.values())
            n = len(ds['Masses'])
            for start in range(0,n,262144):
                stop = min(start+262144,n)
                raw = {k:d[start:stop] for k,d in ds.items()}
                pos = raw['Coordinates'] * conv['Coordinates'] / P.MPC_CM
                mass = raw['Masses'].astype(float) * conv['Masses'] / (P.MSUN_KG*1000)
                rho = raw['Densities'].astype(float) * conv['Densities'] / ((P.MSUN_KG*1000)/P.MPC_CM**3)
                volume = mass/rho
                y = raw['ComptonYParameters'].astype(float) * conv['ComptonYParameters'] / P.MPC_CM**2
                ne = raw['ElectronNumberDensities'] * conv['ElectronNumberDensities']
                vel = raw['Velocities'][:,2].astype(float) * conv['Velocities'] / 1e5
                ne_volume = ne * volume
                weights = [y,ne_volume,ne_volume*vel]
                totals += [len(pos),y.sum(),ne_volume.sum(),np.sum(ne_volume*vel),volume.sum(),mass.sum()]
                pixel = np.minimum((np.mod(pos[:,:2],1000.)/1000*WHOLE_N).astype(int),WHOLE_N-1)
                key = pixel[:,0]*WHOLE_N+pixel[:,1]
                for q,w in enumerate(weights):
                    whole[q] += np.bincount(key,weights=w,minlength=WHOLE_N**2).reshape(WHOLE_N,WHOLE_N)
                pe = y / volume * P.Y_TO_P_VOLUME
                for q,values,edges in [(0,pe,PE_EDGES),(1,ne,NE_EDGES)]:
                    positive = values > 0
                    logs = np.log10(values[positive])
                    hist[q,:len(edges)-1] += np.histogram(logs,bins=edges,weights=volume[positive])[0]
                    atom[q] += [volume[~positive].sum(),volume[positive][logs<edges[0]].sum(),volume[positive][logs>edges[-1]].sum()]
                low,high = pos.min(axis=0),pos.max(axis=0)
                for region,origin in enumerate(ORIGINS):
                    if np.any(high < origin) or np.any(low >= origin+SIDE):
                        continue
                    local = pos-origin
                    take = np.all((local>=0)&(local<SIDE),axis=1)
                    if not take.any():
                        continue
                    local = local[take]
                    xy = np.minimum((local[:,:2]/SIDE*MAP_N).astype(int),MAP_N-1)
                    map_key = xy[:,0]*MAP_N+xy[:,1]
                    xyz = np.minimum((local/SIDE*GRID_N).astype(int),GRID_N-1)
                    grid_key = (xyz[:,0]*GRID_N+xyz[:,1])*GRID_N+xyz[:,2]
                    for q,w in enumerate(weights):
                        maps[region,q] += np.bincount(map_key,weights=w[take],minlength=MAP_N**2).reshape(MAP_N,MAP_N)
                        np.add.at(grids[region,q].ravel(),grid_key,w[take].astype(np.float32))
            P.save_json(ROOT/('worker_%d.progress.json'%number),dict(file=path.name,gas_rows=int(totals[0]),wall_seconds=time.time()-started))
    output = ROOT/('partial_%d.h5'%number)
    with h5py.File(output,'w') as f:
        f.create_dataset('whole_sums',data=whole)
        f.create_dataset('region_sums',data=maps)
        f.create_dataset('grid_sums',data=grids,compression='gzip',compression_opts=1,chunks=(1,1,32,32,32))
        f.create_dataset('particle_volume_hist',data=hist)
        f.create_dataset('particle_volume_atoms',data=atom)
        f.create_dataset('totals',data=totals)
    receipt = dict(worker=number,files=len(files),gas_rows=int(totals[0]),logical_bytes=logical,
        dataset_stored_bytes=stored,wall_seconds=time.time()-started,cpu_seconds=time.process_time()-cpu,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    P.save_json(ROOT/('worker_%d.receipt.json'%number),receipt)
    return receipt


def guard(stop,started,record):
    parent = psutil.Process()
    while not stop.wait(5):
        rss = 0
        for process in [parent]+parent.children(recursive=True):
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        record['peak_tree_RSS_bytes'] = max(rss,record.get('peak_tree_RSS_bytes',0))
        if rss > 24*2**30 or time.time()-started > 4*3600:
            P.save_json(ROOT/'failed.json',dict(reason='memory or four-hour wall guard',rss=rss))
            for child in parent.children(recursive=True):
                child.terminate()
            os._exit(2)


def merge(receipts):
    paths = [ROOT/('partial_%d.h5'%i) for i in range(4)]
    totals = np.zeros(6)
    output = ROOT/'measured_flamingo.h5'
    with h5py.File(output,'w') as out:
        for key in ['whole_sums','region_sums','particle_volume_hist','particle_volume_atoms','totals']:
            value = None
            for path in paths:
                with h5py.File(path,'r') as f:
                    a = f[key][:]
                    value = a if value is None else value+a
            out.create_dataset(key,data=value,compression='gzip',compression_opts=1)
            if key == 'totals':
                totals = value
        target = out.create_dataset('grid_sums',(8,3,GRID_N,GRID_N,GRID_N),dtype='f4',
            chunks=(1,1,32,32,32),compression='gzip',compression_opts=1)
        for region in range(8):
            for q in range(3):
                value = np.zeros((GRID_N,GRID_N,GRID_N),np.float32)
                for path in paths:
                    with h5py.File(path,'r') as f:
                        value += f['grid_sums'][region,q]
                target[region,q] = value
        out.create_dataset('region_origins_mpc',data=ORIGINS)
        out.create_dataset('pressure_log10_edges_pa',data=PE_EDGES)
        out.create_dataset('electrons_log10_edges_cm3',data=NE_EDGES)
        out.attrs.update(box_mpc=1000.,region_side_mpc=SIDE,grid_n=GRID_N,map_n=MAP_N,whole_n=WHOLE_N,
            deposit_quantities='Y [Mpc^2], n_e V [cm^-3 Mpc^3], n_e v_z V [cm^-3 km/s Mpc^3]',
            assignment='particle-centre NGP; all gas, native ionization, actual peculiar gas velocity')
    assert int(totals[0]) == 5_271_821_058
    with h5py.File(output,'r') as f:
        check = np.sum(f['whole_sums'][0],dtype=float)/totals[1]-1
        assert abs(check)<1e-10
    P.save_json(ROOT/'extraction_checks.json',dict(expected_gas_rows=5_271_821_058,
        actual_gas_rows=int(totals[0]),whole_y_conservation_relative_error=float(check),
        raw_snapshot_modified=False,rotation_averaging=False))
    # Only verified redundant partials from this study are removed.
    removed = sum(p.stat().st_size for p in paths)
    for path in paths:
        assert path.parent.resolve()==ROOT.resolve() and path.name.startswith('partial_')
        path.unlink()
    return dict(gas_rows=int(totals[0]),logical_read_bytes=sum(r['logical_bytes'] for r in receipts),
        worker_cpu_seconds=sum(r['cpu_seconds'] for r in receipts),redundant_partials_removed_bytes=removed)


def main():
    assert not (ROOT/'extraction_complete.json').exists(), 'Preserve completed extraction.'
    started = time.time()
    record = dict(workers=4,memory_guard_GiB=24,wall_guard_hours=4)
    stop = threading.Event()
    monitor = threading.Thread(target=guard,args=(stop,started,record),daemon=True)
    monitor.start()
    extract_halos()
    files = sorted(SNAPSHOT.glob('flamingo_0077.*.hdf5'))
    assert len(files)==64
    with ProcessPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(worker,[(i,files[i::4]) for i in range(4)]))
    record.update(merge(receipts))
    stop.set();monitor.join()
    record.update(completed=True,wall_seconds=time.time()-started,physical_network_bytes_measured=False)
    P.save_json(ROOT/'extraction_complete.json',record)


if __name__=='__main__':
    main()
