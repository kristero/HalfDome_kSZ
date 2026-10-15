"""Remote read-only inventory of SOAP-HBT halo_properties files on STRW maas (headers/attributes only)."""
import json, sys, time
import rc

OUTDIR = '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/stacking_choices_20261014/research/soap/'

SRC = r'''import os; os.nice(19)
import h5py, json, numpy as np, time
t0=time.time()
path=%r
f=h5py.File(path,'r')
def conv(v):
    if isinstance(v,bytes): return v.decode(errors='replace')
    if isinstance(v,np.ndarray):
        if v.dtype.kind in 'SO': return [x.decode(errors='replace') if isinstance(x,bytes) else str(x) for x in v.ravel().tolist()][:50]
        return v.ravel().tolist()[:50]
    if isinstance(v,(np.generic,)): return v.item()
    return str(v)
dsets={}; groups={}
def visit(name,obj):
    if isinstance(obj,h5py.Dataset):
        dsets[name]={'dtype':str(obj.dtype),'shape':list(obj.shape),'attrs':{k:conv(v) for k,v in obj.attrs.items()},
                     'chunks':list(obj.chunks) if obj.chunks else None,'compression':obj.compression,'storage_bytes':int(obj.id.get_storage_size())}
    else:
        groups[name]={'attrs':{k:conv(v) for k,v in obj.attrs.items()},'n':len(obj.keys())}
f.visititems(visit)
meta={}
for g in ('Header','Cosmology','Parameters','Parameters/RecentlyHeatedGasFilter','Code','Units','SWIFT/Header'):
    if g in f: meta[g]={k:conv(v) for k,v in f[g].attrs.items()}
print(json.dumps({'path':path,'file_bytes':os.path.getsize(path),'groups':groups,'datasets':dsets,'meta':meta,'elapsed':time.time()-t0}))
'''

if __name__ == '__main__':
    tag, path = sys.argv[1], sys.argv[2]
    obj, dt, raw = rc.run(SRC % path, timeout=600)
    obj['wall_including_ssh_s'] = dt
    with open(OUTDIR + f'inventory_raw_{tag}.json', 'w') as fh:
        json.dump(obj, fh)
    print(tag, len(obj['datasets']), 'datasets', len(obj['groups']), 'groups', 'remote', round(obj['elapsed'], 2), 's, total', round(dt, 2))
