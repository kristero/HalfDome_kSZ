"""Quick remote read-only listing (names, dtype, shape, Description, a-exponent, physical flag) of SOAP files,
compared with the full fiducial z=0 inventory."""
import json
import rc
OUT = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'
SRC = r'''import os; os.nice(19)
import h5py, json, numpy as np, time, glob
res={}
base='/net/hypernova/data2/FLAMINGO/L1000N1800/'
res['strong_agn_soap_files']=sorted(os.path.basename(p) for p in glob.glob(base+'HYDRO_STRONG_AGN/SOAP-HBT/*'))[:200]
res['fid_soap_halo_props']=sorted(os.path.basename(p) for p in glob.glob(base+'HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_*.hdf5'))
res['fid_soap_sizes']={os.path.basename(p):os.path.getsize(p) for p in glob.glob(base+'HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_*.hdf5')}
for tag,path in (('strong_agn0077',base+'HYDRO_STRONG_AGN/SOAP-HBT/halo_properties_0077.hdf5'),('fid0037',base+'HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_0037.hdf5')):
    t0=time.time()
    f=h5py.File(path,'r'); ds={}
    def visit(name,obj):
        if isinstance(obj,h5py.Dataset):
            a=obj.attrs
            ds[name]=[str(obj.dtype),list(obj.shape),(lambda v: v.decode(errors='replace') if isinstance(v,bytes) else str(v))(a.get('Description','')),
                      float(np.ravel(a.get('a-scale exponent',[np.nan]))[0]),float(np.ravel(a.get('Value stored as physical',[np.nan]))[0])]
    f.visititems(visit)
    hdr={k:(v.tolist() if hasattr(v,'tolist') else str(v)) for k,v in f['Header'].attrs.items() if k in ('Redshift','Scale-factor','RunName','NumSubhalos_Total','SnapshotDate')}
    flt={k:(v.tolist() if hasattr(v,'tolist') else str(v)) for k,v in f['Parameters/RecentlyHeatedGasFilter'].attrs.items()}
    res[tag]={'path':path,'bytes':os.path.getsize(path),'datasets':ds,'header':hdr,'agn_filter':flt,'seconds':time.time()-t0}
print(json.dumps(res, default=lambda o: o.decode(errors="replace") if isinstance(o, bytes) else str(o)))
'''
obj, dt, raw = rc.run(SRC, timeout=900)
obj['wall_s'] = dt
json.dump(obj, open(OUT + 'inventory_raw_quick.json', 'w'))
full = json.load(open(OUT + 'inventory_raw_fid0077.json'))['datasets']
for tag in ('strong_agn0077', 'fid0037'):
    ds = obj[tag]['datasets']
    a, b = set(full), set(ds)
    print(tag, obj[tag]['header'], obj[tag]['bytes'] / 1e9, 'GB', len(ds), 'datasets', round(obj[tag]['seconds'], 1), 's')
    print('  only in fid0077:', sorted(a - b)[:40])
    print('  only in', tag, ':', sorted(b - a)[:40])
    diff = [k for k in a & b if full[k]['dtype'] != ds[k][0] or full[k]['shape'][1:] != ds[k][1][1:]]
    print('  dtype/shape differences:', diff[:20])
    print('  agn filter', obj[tag]['agn_filter'])
print(obj['strong_agn_soap_files'])
print(obj['fid_soap_halo_props'][:3], len(obj['fid_soap_halo_props']))
print('wall', dt)
