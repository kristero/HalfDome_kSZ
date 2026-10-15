"""Read-only metadata probe on STRW maas: cell index (Counts/Offsets/Files) of L1_m9 0077 and per-field stored bytes,
chunk shapes and filters from HDF5 headers (dataset.id.get_storage_size reads no particle data). Writes nothing remotely."""
import json, sys
sys.path.insert(0, '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012')
from remote_control import remote

SRC = r'''
import os; os.nice(19)
import json, time
import h5py, numpy as np
t0 = time.time()
base = '/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL'
top = base + '/snapshots/flamingo_0077/flamingo_0077.hdf5'
res = {}
with h5py.File(top, 'r') as f:
    c = f['Cells']
    res['cells_keys'] = sorted(c.keys())
    res['counts'] = c['Counts/PartType0'][...].astype(np.int64).tolist()
    res['offsets_in_file'] = c['OffsetsInFile/PartType0'][...].astype(np.int64).tolist()
    res['files'] = c['Files/PartType0'][...].astype(np.int64).tolist()
    res['cell_meta'] = {k: (np.asarray(v).tolist() if np.asarray(v).size < 10 else 'big') for k, v in c['Meta-data'].attrs.items()} if 'Meta-data' in c else None
def fields(path, group='PartType0'):
    out = {}
    with h5py.File(path, 'r') as f:
        g = f[group]
        for k in sorted(g.keys()):
            d = g[k]
            plist = d.id.get_create_plist()
            filt = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
            out[k] = {'shape': list(d.shape), 'dtype': str(d.dtype), 'chunks': list(d.chunks) if d.chunks else None,
                      'stored_bytes': int(d.id.get_storage_size()), 'filters': filt}
    return out
res['chunk0'] = fields(base + '/snapshots/flamingo_0077/flamingo_0077.0.hdf5')
res['chunk56'] = fields(base + '/snapshots/flamingo_0077/flamingo_0077.56.hdf5')
res['membership0'] = fields(base + '/SOAP-HBT/membership_0077/membership_0077.0.hdf5')
res['elapsed_s'] = time.time() - t0
print(json.dumps(res))
'''
out = remote(SRC, python='/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python', timeout=300)
obj, _ = json.JSONDecoder().raw_decode(out.strip())
json.dump(obj, open('maas_cell_index_and_field_storage.json', 'w'))
print('elapsed', obj['elapsed_s'], 'cells', len(obj['counts']), 'sum counts', sum(obj['counts']))
for name in ('chunk0', 'chunk56', 'membership0'):
    print('==', name)
    for k, v in obj[name].items():
        n = v['shape'][0]
        print(f"  {k:34s} {v['dtype']:8s} {str(v['chunks']):16s} stored {v['stored_bytes']/n:6.2f} B/p filters {v['filters']}")
