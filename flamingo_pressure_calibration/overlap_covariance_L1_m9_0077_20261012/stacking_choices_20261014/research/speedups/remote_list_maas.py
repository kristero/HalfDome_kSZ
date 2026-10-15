"""Read-only listing of saved products under /home/nagainis/HalfDome* on STRW maas (no writes, no loops over data)."""
import json, sys
sys.path.insert(0, '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012')
from remote_control import remote

SRC = r'''
import os; os.nice(19)
import json, glob, time
t0 = time.time()
res = {}
roots = sorted(glob.glob('/home/nagainis/HalfDome*'))
res['roots'] = roots
files = []
dirs = {}
for root in roots:
    for dp, dn, fn in os.walk(root):
        # skip venvs and caches
        dn[:] = [d for d in dn if d not in ('venv', '__pycache__', '.git', 'site-packages', 'lib', 'include', 'bin', 'share')]
        tot = 0
        for f in fn:
            p = os.path.join(dp, f)
            try:
                s = os.lstat(p).st_size
            except OSError:
                continue
            tot += s
            if f.endswith(('.npz', '.npy', '.hdf5', '.h5', '.fits')) and s > 200000:
                files.append([p, s, time.strftime('%Y-%m-%d', time.gmtime(os.lstat(p).st_mtime))])
        dirs[dp] = [len(fn), tot]
res['data_files'] = files
# summarise per top-level study dir
agg = {}
for dp, (n, tot) in dirs.items():
    parts = dp.split('/')
    key = '/'.join(parts[:6])
    a = agg.setdefault(key, [0, 0])
    a[0] += n; a[1] += tot
res['dir_totals'] = agg
try:
    st = os.statvfs('/home/nagainis')
    res['home_free_GB'] = st.f_bavail * st.f_frsize / 1e9
except Exception as e:
    res['home_free_GB'] = str(e)
res['elapsed_s'] = time.time() - t0
print(json.dumps(res))
'''
out = remote(SRC, timeout=120)
obj, _ = json.JSONDecoder().raw_decode(out.strip())
json.dump(obj, open('maas_listing.json', 'w'), indent=1)
print('roots', obj['roots'], 'files', len(obj['data_files']), 'elapsed', obj['elapsed_s'])
for k, v in sorted(obj['dir_totals'].items()):
    print(f"{v[0]:6d} files {v[1]/1e9:8.3f} GB  {k}")
