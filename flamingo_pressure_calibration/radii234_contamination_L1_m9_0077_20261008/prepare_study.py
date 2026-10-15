"""Select 300 random lower-mass centrals/bin and budget a single 4R read.

Only catalogue columns, particle-dataset attributes and cell indexes are read.
HBT membership files are paired with snapshot chunks by their file number;
SOAP writes membership in exactly the original snapshot particle order.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import psutil
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'dependencies'))
import common as C
import flamingo_io as F
from catalogue_stage import sphere_cells,csr_map
from check_fresh_mask import fresh_mask
from orientations import viewing_rotations

ROOT='/net/hypernova/data2/FLAMINGO'
RUN='L1000N1800/HYDRO_FIDUCIAL'
SNAPSHOT=RUN+'/snapshots/flamingo_0077/flamingo_0077.hdf5'
SOAP=RUN+'/SOAP-HBT/halo_properties_0077.hdf5'
MEMBERSHIP=Path(ROOT)/RUN/'SOAP-HBT/membership_0077'
FIELDS=tuple(F.GAS_FIELDS)+('ParticleIDs','SmoothedElementMassFractions','Pressures')


def plain(value):
    if isinstance(value,bytes):return value.decode(errors='replace')
    if hasattr(value,'tolist'):return plain(value.tolist())
    if isinstance(value,(list,tuple)):return [plain(x) for x in value]
    return value


def main():
    started=time.perf_counter()
    assert not (HERE/'metadata_plan.json').exists()
    cfg=json.loads((HERE/'study_config.json').read_text())
    source=np.load(HERE.parent/'mass12p5_15p5_r4_preflight_L1_m9_0077_20261008/candidate_metadata.npz')
    logm=np.log10(source['m200c_msun'])
    rng=np.random.default_rng(cfg['seed']);chosen=[];populations=[]
    for lo,hi in zip(cfg['mass_edges_log10_Msun'][:-1],cfg['mass_edges_log10_Msun'][1:]):
        group=np.flatnonzero((logm>=lo)&(logm<hi))
        populations.append(len(group))
        chosen.extend(rng.choice(group,300,replace=False) if lo<14.5 else group)
    chosen=np.sort(chosen)
    rows=source['soap_row'][chosen]
    snapshot=F.Snapshot(F.Store(ROOT),SNAPSHOT,77,read_from='chunks',chunk_cache_bytes=1024**2)
    soap=F.Soap(F.Store(ROOT),SOAP)
    halos=soap.columns(names=('halo_index','centre_mpc','m200c_msun','r_200_crit_mpc','ngas_200_crit',
                             'gasmass_200_crit_msun','y_200_crit_mpc2','y_noagn_200_crit_mpc2'),rows=rows)
    halos['bound_mass_msun']=soap.read('BoundSubhalo/TotalMass',rows,target=C.MSUN_G)
    halos['bound_gas_mass_msun']=soap.read('BoundSubhalo/GasMass',rows,target=C.MSUN_G)
    halos['bound_gas_count']=soap.read('BoundSubhalo/NumberOfGasParticles',rows)
    com=soap.read('BoundSubhalo/CentreOfMass',rows,target=C.MPC_CM)
    delta=com-halos['centre_mpc'];delta-=snapshot.box_size_cmpc*snapshot.a*np.rint(delta/(snapshot.box_size_cmpc*snapshot.a))
    halos['centre_offset_R200c']=np.linalg.norm(delta,axis=1)/halos['r_200_crit_mpc']
    # Find all own satellites globally; do not assume they lie inside a SOAP cell cut.
    selected_rows=np.sort(rows)
    def keep(host):
        pos=np.searchsorted(selected_rows,host['host'])
        pos=np.minimum(pos,len(selected_rows)-1)
        return (host['host']>=0)&(selected_rows[pos]==host['host'])
    sat_rows=soap.scan({'host':('SOAP/HostHaloIndex',None)},keep,chunk_rows=500000)
    sat_host=soap.read('SOAP/HostHaloIndex',sat_rows)
    sat_halo=np.searchsorted(rows,sat_host)
    assert np.array_equal(rows[sat_halo],sat_host)
    sat_ids=soap.read('InputHalos/HaloCatalogueIndex',sat_rows)
    sat_mass=soap.read('BoundSubhalo/TotalMass',sat_rows,target=C.MSUN_G)
    sat_centre=soap.read('InputHalos/HaloCentre',sat_rows,target=C.MPC_CM)
    d=sat_centre-halos['centre_mpc'][sat_halo]
    d-=snapshot.box_size_cmpc*snapshot.a*np.rint(d/(snapshot.box_size_cmpc*snapshot.a))
    sat_radius=np.linalg.norm(d,axis=1)/halos['r_200_crit_mpc'][sat_halo]
    ratios=sat_mass/halos['bound_mass_msun'][sat_halo]
    sat_ptr=np.r_[0,np.cumsum(np.bincount(sat_halo,minlength=len(rows)))]
    order=np.lexsort((sat_ids,sat_halo))
    halos.update(satellite_ptr=sat_ptr,satellite_ids=sat_ids[order])
    for radius in (2,3,4):
        inside=sat_radius<=radius
        largest=np.zeros(len(rows));np.maximum.at(largest,sat_halo[inside],ratios[inside])
        halos[f'max_satellite_bound_mass_ratio_R{radius}']=largest
        halos[f'satellite_count_R{radius}']=np.bincount(sat_halo[inside],minlength=len(rows))
    halos['disturbed_proxy']=(halos['max_satellite_bound_mass_ratio_R2']>=.25)|(halos['centre_offset_R200c']>.07)
    halos['centre_cmpc']=halos['centre_mpc']/snapshot.a
    ph,pc=sphere_cells(halos['centre_cmpc'],4*halos['r_200_crit_mpc']/snapshot.a,
        snapshot.cell_size_cmpc,snapshot.dimension,snapshot.lookup,snapshot.box_size_cmpc,.1*snapshot.cell_size_cmpc)
    cells,ptr,index=csr_map(ph,pc)
    halos.update(pair_halo=ph,pair_cell=pc,cells=snapshot.read_order(cells),
                 csr_cells=cells,csr_ptr=ptr,csr_halo=index,
                 mass_edges=cfg['mass_edges_log10_Msun'])
    lm=np.log10(halos['m200c_msun'])
    halos['mass_bin']=np.searchsorted(halos['mass_edges'],lm,side='right')-1
    schema={};bpp=0
    for name in FIELDS:
        ds=snapshot.dataset(name);nb=ds.dtype.itemsize*int(np.prod(ds.shape[1:]));bpp+=nb
        schema[name]={'dtype':str(ds.dtype),'shape':list(ds.shape),'attributes':{k:plain(v) for k,v in ds.attrs.items()},'bytes_per_particle':nb}
    # Verify every membership chunk exists and has the same row count.
    membership_sources=[]
    for file_id in np.unique(snapshot.cell_file[cells]):
        file_id=int(file_id);path=MEMBERSHIP/f'membership_0077.{file_id}.hdf5'
        with h5py.File(path,'r') as f:
            ds=f['PartType0/GroupNr_bound']
            assert len(ds)==len(snapshot._chunk(file_id)['PartType0/Masses'])
            membership_sources.append({'file':file_id,'path':str(path),'shape':list(ds.shape),'dtype':str(ds.dtype)})
    checks=[];swift_rows=[]
    for b in range(15):
        group=np.flatnonzero(halos['mass_bin']==b)
        order=group[np.argsort(lm[group])]
        checks.extend(order[[0,len(order)//2,-1]])
    checks=sorted(set(int(h) for h in checks))
    for h in checks:
        mask=fresh_mask(str(Path(ROOT)/SNAPSHOT),halos['centre_cmpc'][h],4*halos['r_200_crit_mpc'][h],snapshot.a)
        ranges=np.asarray(mask.gas);swift_rows.append(int(np.sum(ranges[:,1]-ranges[:,0])))
    # Include the original z view, then ten direction/roll-balanced views.
    rotation=viewing_rotations();turns=[0,1,2,3,0,1,2,3,0,2]
    for i,k in enumerate(turns):
        angle=k*np.pi/2
        roll=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rotation[i]=roll@rotation[i]
    unique_rows=int(snapshot.counts[cells].sum())
    plan={'simulation':'L1_m9','snapshot':77,'redshift':snapshot.redshift,'scale_factor':snapshot.a,
          'snapshot_path':str(Path(ROOT)/SNAPSHOT),'soap_path':str(Path(ROOT)/SOAP),
          'halo_count':len(rows),'sample_per_bin':np.bincount(halos['mass_bin'],minlength=15).tolist(),
          'population_per_bin':populations,'mass_edges_log10_msun':list(halos['mass_edges']),
          'selected_cells':len(cells),'selected_candidate_rows':unique_rows,
          'candidate_halo_particle_comparisons':int(snapshot.counts[pc].sum()),
          'bytes_per_particle_with_membership':bpp+8,'logical_bytes_one_direct_pass':unique_rows*(bpp+8),
          'swift_check_indices':checks,'swift_check_rows':swift_rows,'swift_logical_bytes_upper_bound':sum(swift_rows)*108,
          'header':{k:plain(v) for k,v in snapshot.header.items()},
          'cosmology':{k:plain(v) for k,v in snapshot.file['Cosmology'].attrs.items()},
          'field_schema':schema,'baryon_fraction':snapshot.cosmology['baryon_fraction'],
          'softening_physical_mpc':min(snapshot.a*snapshot.softening_comoving_mpc,snapshot.softening_max_physical_mpc),
          'recent_agn_filter':soap.heating_filter(),'rotation_matrices':np.concatenate([np.eye(3)[None],rotation]).tolist(),
          'membership_sources':membership_sources,'membership_order_contract':'SOAP flamingo-hbt group_membership.py writes original snapshot file/order; GroupNr_bound equals InputHalos/HaloCatalogueIndex, not SOAP row',
          'disturbed_proxy':'max bound satellite/central mass ratio within 2R >=0.25 OR bound-central COM offset/R >0.07; snapshot proxy, not a measured merger history',
          'particle_values_read':False,'wall_seconds':time.perf_counter()-started,
          'resources':{'available_memory_GiB':psutil.virtual_memory().available/1024**3,'load':list(os.getloadavg()),'host':__import__('socket').getfqdn()}}
    np.savez_compressed(HERE/'selected_halos.npz',**halos)
    (HERE/'metadata_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ['halo_count','sample_per_bin','selected_cells','logical_bytes_one_direct_pass','swift_logical_bytes_upper_bound','candidate_halo_particle_comparisons','resources','wall_seconds']},indent=2))
    snapshot.close();soap.close()


if __name__=='__main__':main()
