"""Scan .npz files under flamingo_pressure_calibration and list member arrays (shape, dtype) without loading data."""
import json, os, sys, zipfile
import numpy as np
from numpy.lib import format as npf

ROOT = '/home/cbllover/HalfDome/flamingo_pressure_calibration'
SKIP = ('stacking_choices_20261014', 'rehearsal_work', '__pycache__', 'results_smoke')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npz_schema_scan.json')

def members(path):
    res = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename
            if not name.endswith('.npy'):
                continue
            with z.open(info) as f:
                try:
                    version = npf.read_magic(f)
                    if version == (1, 0):
                        shape, fortran, dtype = npf.read_array_header_1_0(f)
                    else:
                        shape, fortran, dtype = npf.read_array_header_2_0(f)
                    res[name[:-4]] = [list(shape), str(dtype)]
                except Exception as e:
                    res[name[:-4]] = ['err', str(e)]
    return res

out = {}
for dp, dn, fn in os.walk(ROOT):
    if any(s in dp for s in SKIP):
        continue
    for f in fn:
        if f.endswith('.npz'):
            p = os.path.join(dp, f)
            sz = os.path.getsize(p)
            if sz < 50_000:
                continue
            try:
                out[os.path.relpath(p, ROOT)] = {'bytes': sz, 'members': members(p)}
            except Exception as e:
                out[os.path.relpath(p, ROOT)] = {'bytes': sz, 'error': str(e)}
json.dump(out, open(OUT, 'w'), indent=1, sort_keys=True)
print(len(out), 'files ->', OUT)
