"""Read-only: HDF5 filter parameters (cd_values) of the L1_m9 0077 chunk-0 gas datasets on maas (header metadata only)."""
import json, sys
sys.path.insert(0, '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012')
from remote_control import remote
SRC = r'''
import os; os.nice(19)
import json, h5py
p = '/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/snapshots/flamingo_0077/flamingo_0077.0.hdf5'
out = {}
with h5py.File(p, 'r') as f:
    for k, d in f['PartType0'].items():
        pl = d.id.get_create_plist()
        out[k] = [[int(x) if not isinstance(x, (tuple, bytes)) else (list(map(int, x)) if isinstance(x, tuple) else x.decode()) for x in pl.get_filter(i)] for i in range(pl.get_nfilters())]
        t = d.id.get_type()
        try:
            out[k].append({'type_size': t.get_size(), 'precision': t.get_precision(), 'offset': t.get_offset()})
        except Exception as e:
            out[k].append({'type_err': str(e)})
print(json.dumps(out))
'''
out = remote(SRC, python='/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python', timeout=120)
obj, _ = json.JSONDecoder().raw_decode(out.strip())
json.dump(obj, open('maas_filter_parameters.json', 'w'), indent=1)
for k, v in obj.items():
    print(k, v)
