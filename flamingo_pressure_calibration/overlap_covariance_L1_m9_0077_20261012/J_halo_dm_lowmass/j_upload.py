"""Targeted, SHA256-verified upload of Test J's code and inputs into the maas runtime mirror (local helper).

Why: the full 'cluster_tests.py mirror' re-uploads ~250 MB (every test's referenced files) through 12 MB ssh chunks; on
2026-10-15 it failed after 25.6 min when outbound ssh dropped (CLUSTER_RUN.json mirror_attempts).  Tests I and K used the
same targeted approach.  This helper uses the same transport (remote_control.remote: Windows OpenSSH alias strw-maas)
and the same runtime layout (RUNTIME/HalfDome/<path relative to /home/cbllover/HalfDome>), and writes nothing outside
the runtime tree.  It uploads only the files Test J needs that are missing or different on maas:
  * every J_halo_dm_lowmass/*.py and CLUSTER_RUN.json;
  * every input listed in CLUSTER_RUN.json 'inputs' (inputs/j0_*, Test F's profiles, Test H's truth, engine deps, ...).
A remote path that is a symlink into the real study tree and has the right SHA256 is left alone; one with the wrong
content is replaced by a real file (as cluster_tests.py mirror does).  Each file goes through RUNTIME/upload_J.tmp in
8 MB base64 chunks, is SHA256-checked on maas and then moved into place.  Writes post/upload_J.json.
Usage: python -B j_upload.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
LOCAL_REPO = Path("/home/cbllover/HalfDome")
sys.path.insert(0, str(STUDY))
from remote_control import remote  # noqa: E402

PY = json.loads((STUDY / "study_config.json").read_text())["remote_python"]
RUNTIME = "/home/nagainis/HalfDome_runtime/overlap_covariance_tests"
MIRROR = RUNTIME + "/HalfDome"
CHUNK = 8 * 1024**2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wanted():
    spec = json.loads((HERE / "CLUSTER_RUN.json").read_text())
    files = {p.resolve() for p in HERE.glob("*.py")} | {(HERE / "CLUSTER_RUN.json").resolve()}
    files |= {Path(r["path"]).resolve() for r in spec["inputs"]}
    files |= {(STUDY / "cluster_job.py").resolve()}
    return sorted(p for p in files if p.is_file() and p.is_relative_to(LOCAL_REPO))


def remote_state(rels):
    out = remote(f'''import hashlib, json, os
from pathlib import Path
m = Path({MIRROR!r})
res = {{}}
for rel in json.loads({json.dumps(rels)!r}):
    p = m / rel
    if p.exists():
        res[rel] = [hashlib.sha256(p.read_bytes()).hexdigest(), p.is_symlink()]
    else:
        res[rel] = None
print(json.dumps(res))''', python=PY, timeout=600)
    return json.loads(out.strip().splitlines()[-1])


def upload(path, rel):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    tmp = RUNTIME + "/upload_J.tmp"
    remote(f'''from pathlib import Path
Path({tmp!r}).write_bytes(b"")''', python=PY, timeout=120)
    for off in range(0, len(data), CHUNK):
        chunk = base64.b64encode(data[off:off + CHUNK]).decode()
        for attempt in range(3):
            try:
                remote(f'''import base64
with open({tmp!r}, "ab") as f:
    f.write(base64.b64decode({chunk!r}))''', python=PY, timeout=600)
                break
            except Exception:                              # noqa: BLE001 - retry the chunk after truncating back
                if attempt == 2:
                    raise
                remote(f'''import os
os.truncate({tmp!r}, {off})''', python=PY, timeout=120)
                time.sleep(10)
    res = remote(f'''import hashlib, os
from pathlib import Path
tmp = Path({tmp!r}); dst = Path({MIRROR!r}) / {rel!r}
assert hashlib.sha256(tmp.read_bytes()).hexdigest() == {digest!r}, "upload corrupted"
assert dst.resolve().is_relative_to(Path({RUNTIME!r}).resolve()) or dst.is_symlink(), dst
dst.parent.mkdir(parents=True, exist_ok=True)
if dst.is_symlink():
    dst.unlink()
os.replace(tmp, dst)
print(hashlib.sha256(dst.read_bytes()).hexdigest())''', python=PY, timeout=300)
    assert res.strip().splitlines()[-1] == digest, rel
    return digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = wanted()
    rels = [str(p.relative_to(LOCAL_REPO)) for p in files]
    state = remote_state(rels)
    todo, ok = [], []
    for p, rel in zip(files, rels):
        s = state.get(rel)
        if s is not None and s[0] == sha(p):
            ok.append(rel)
        else:
            todo.append((p, rel, "missing" if s is None else ("symlink_differs" if s[1] else "differs")))
    print(json.dumps({"files": len(files), "already_identical": len(ok),
                      "to_upload": [(r, why, round(p.stat().st_size / 1e6, 2)) for p, r, why in todo]}, indent=1))
    log = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "files_checked": len(files), "identical": ok, "uploaded": []}
    if not a.dry_run:
        # the remote jobs/ directory needs the wrapper (cluster_tests.run uses RUNTIME/jobs/cluster_job.py)
        for p, rel, why in todo:
            t0 = time.time()
            digest = upload(p, rel)
            log["uploaded"].append({"path": rel, "reason": why, "bytes": p.stat().st_size, "sha256": digest,
                                    "seconds": round(time.time() - t0, 1)})
            print("uploaded", rel, round(time.time() - t0, 1), "s", flush=True)
        jobs = remote(f'''import hashlib
from pathlib import Path
j = Path({RUNTIME!r}) / "jobs" / "cluster_job.py"
print(hashlib.sha256(j.read_bytes()).hexdigest() if j.exists() else "missing")''', python=PY, timeout=120)
        log["jobs_cluster_job_sha256_remote"] = jobs.strip().splitlines()[-1]
        log["jobs_cluster_job_sha256_local"] = sha(STUDY / "cluster_job.py")
        (HERE / "post").mkdir(exist_ok=True)
        (HERE / "post" / "upload_J.json").write_text(json.dumps(log, indent=1) + "\n")
        print(json.dumps({k: v for k, v in log.items() if k != "identical"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
