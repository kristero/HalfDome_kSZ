"""Local helpers for Test K on STRW maas (transport: the study's remote_control.remote, Windows OpenSSH alias strw-maas).

  python -B k_maas.py upload            upload Test K's code (k_*.py, run_test_k.py, write_cluster_run.py,
                                        CLUSTER_RUN.json) into the runtime mirror, SHA256-verified on maas
                                        (the full cluster_tests.py mirror timed out for Test I; same approach as Test I)
  python -B k_maas.py probe             load, free memory, /tmp space, python modules, the other jobs
  python -B k_maas.py poll [--max N]    poll cluster_tests.py status every 5 minutes until test_K has a done or
                                        failed record (or N polls); one line per poll in cluster_poll.txt
  python -B k_maas.py scratch [--delete]  size of /tmp/nagainis_halfdome_testK (and delete it)
  python -B k_maas.py fetchdir --dir results_smoke   copy K_soap_battaglia/<dir> from the runtime tree (SHA256 checked;
                                        files under work/ skipped)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
from remote_control import remote  # noqa: E402

CFG = json.loads((STUDY / "study_config.json").read_text())
PY = CFG["remote_python"]
RUNTIME = "/home/nagainis/HalfDome_runtime/overlap_covariance_tests"
REMOTE_DIR = RUNTIME + "/HalfDome/flamingo_pressure_calibration/" + STUDY.name + "/" + HERE.name
SCRATCH = "/tmp/nagainis_halfdome_testK"
CODE = ["k_common.py", "k_fit_core.py", "k_read.py", "k_fits.py", "k_selftest.py", "run_test_k.py", "write_cluster_run.py", "k_starts_run2.json",
        "CLUSTER_RUN.json"]


def upload():
    out = {}
    for name in CODE:
        p = HERE / name
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        res = remote(f'''import base64, hashlib
from pathlib import Path
d = Path({REMOTE_DIR!r}); d.mkdir(parents=True, exist_ok=True)
t = d / {name!r}
if t.is_symlink():
    t.unlink()
t.write_bytes(base64.b64decode({base64.b64encode(data).decode()!r}))
print(hashlib.sha256(t.read_bytes()).hexdigest())''', python=PY, timeout=300).strip()
        assert res == digest, (name, res, digest)
        out[name] = digest
    print(json.dumps({"uploaded": out, "remote_dir": REMOTE_DIR}, indent=1))


def probe():
    print(remote(f'''import json, os, shutil, sys, psutil, importlib.util
st = shutil.disk_usage('/tmp')
home = shutil.disk_usage('/home/nagainis')
out = {{'load': os.getloadavg(), 'cpus': os.cpu_count(), 'available_GiB': round(psutil.virtual_memory().available / 2**30, 1),
       'tmp_free_GB': round(st.free / 1e9, 1), 'home_fs_free_GB': round(home.free / 1e9, 1),
       'modules': {{m: bool(importlib.util.find_spec(m)) for m in ('numpy', 'scipy', 'h5py', 'psutil')}},
       'scratch_exists': os.path.exists({SCRATCH!r}), 'python': sys.version.split()[0]}}
import numpy, scipy, h5py
out['versions'] = {{'numpy': numpy.__version__, 'scipy': scipy.__version__, 'h5py': h5py.__version__}}
procs = []
for p in psutil.process_iter(['username', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
    try:
        if p.info['username'] == 'nagainis' and 'python' in (p.info['name'] or ''):
            procs.append(' '.join(p.info['cmdline'] or [])[:160])
    except Exception:
        pass
out['own_python_processes'] = procs[:20]
import subprocess
q = subprocess.run(['du', '-sb', '/home/nagainis'], capture_output=True, text=True, timeout=120).stdout.split()
out['home_used_GB'] = round(int(q[0]) / 1e9, 2) if q else None
print(json.dumps(out, indent=1))''', python=PY, timeout=300))


def poll(max_polls):
    log = HERE / "cluster_poll.txt"
    py = "/home/cbllover/miniconda3/envs/halfdome/bin/python"
    for i in range(max_polls):
        line = ""
        try:
            out = subprocess.run([py, "-B", str(STUDY / "cluster_tests.py"), "status"], capture_output=True, text=True,
                                 timeout=240, cwd=str(STUDY)).stdout
            d = json.loads(out)
            j = d["jobs"].get("test_K", {})
            tail = j.get("log_tail", "").strip().splitlines()[-1:] or [""]
            line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} {'ALIVE' if j.get('alive') else 'ENDED'} load {d['load']} avail "
                    f"{d['available_GiB']} rss {j.get('tree_RSS_GiB')} min {j.get('elapsed_min')} | {tail[0][:200]}")
            finished = (not j.get("alive")) and (j.get("done") or j.get("failed"))
            if finished:
                line += f" | done={j.get('done')} failed={j.get('failed')}"
        except Exception as exc:   # noqa: BLE001
            finished = False
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} poll error {exc!r}"[:400]
        with open(log, "a") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        if finished:
            return
        time.sleep(300)


def scratch(delete):
    print(remote(f'''import json, os, shutil
from pathlib import Path
p = Path({SCRATCH!r})
size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0
n = sum(1 for f in p.rglob('*') if f.is_file()) if p.exists() else 0
if {bool(delete)!r} and p.exists():
    shutil.rmtree(p)
print(json.dumps({{'path': str(p), 'files': n, 'bytes': size, 'deleted': {bool(delete)!r} and n >= 0, 'exists_after': p.exists()}}))''',
                 python=PY, timeout=600))


def fetchdir(sub):
    src = REMOTE_DIR + "/" + sub
    listing = json.loads(remote(f'''import hashlib, json
from pathlib import Path
root = Path({src!r})
print(json.dumps({{str(p.relative_to(root)): [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()]
                   for p in root.rglob('*') if p.is_file() and 'work' not in p.relative_to(root).parts}} if root.exists() else {{}}))''',
                            python=PY, timeout=900))
    target = HERE / sub
    n = 0
    for rel, (size, digest) in sorted(listing.items()):
        path = target / rel
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray()
        for offset in range(0, size, 12 * 1024 ** 2):
            data += base64.b64decode(remote(f'''import base64
with open({src + "/" + rel!r}, 'rb') as f:
    f.seek({offset}); print(base64.b64encode(f.read(12 * 1024 ** 2)).decode())''', python=PY, timeout=600))
        assert hashlib.sha256(bytes(data)).hexdigest() == digest, rel
        path.write_bytes(bytes(data))
        n += 1
    print(json.dumps({"dir": sub, "files": len(listing), "fetched": n, "MiB": round(sum(v[0] for v in listing.values()) / 2 ** 20, 2)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["upload", "probe", "poll", "scratch", "fetchdir"])
    ap.add_argument("--dir", default="results_smoke")
    ap.add_argument("--max", type=int, default=96)
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args()
    if a.action == "upload":
        upload()
    elif a.action == "probe":
        probe()
    elif a.action == "poll":
        poll(a.max)
    elif a.action == "fetchdir":
        fetchdir(a.dir)
    else:
        scratch(a.delete)


if __name__ == "__main__":
    main()
