"""Test J2 cluster helper (local; transport = remote_control.remote through the Windows OpenSSH alias strw-maas, as
cluster_tests.py and j_upload.py).  Writes nothing on maas outside the runtime tree RUNTIME and J2's node-local scratch.

  python -B j2_cluster.py upload [--dry-run]   SHA256-checked targeted upload of J2's code and inputs (j_upload.upload)
  python -B j2_cluster.py check [--out F]      read-only: remote SHA256 of every file J2 reads vs local (post_j2/remote_check.json,
                                               or F relative to J_halo_dm_lowmass/)
  python -B j2_cluster.py fetch [--dir D]      RUNTIME/.../J_halo_dm_lowmass/<D> (default results_j2; without work/) ->
                                               local <D>/, plus the wrapper records jobs/test_J_{receipt,done,failed}.json,
                                               test_J.log (D = results_mecke: the J1 Mecke revision run, 2026-10-16)
  python -B j2_cluster.py cleanup              delete the node-local scratch /tmp/nagainis_halfdome_testJ (J2's bases)
  python -B j2_cluster.py record JSON [--key K]  append a run record to CLUSTER_RUN.json[K] (default 'j2_maas_runs';
                                               'j1_mecke_runs' for J1's Mecke revision run)
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
sys.path.insert(0, str(HERE))
from remote_control import remote  # noqa: E402

import j_upload as JU  # noqa: E402

PY = JU.PY
RUNTIME = JU.RUNTIME
MIRROR = JU.MIRROR
REL = str(HERE.relative_to(LOCAL_REPO))
SCRATCH = "/tmp/nagainis_halfdome_testJ"
READ_BY_J2 = [HERE / "inputs" / "j2_fitdata.npz", HERE / "inputs" / "j2_fitdata_meta.json",
              HERE / "inputs" / "j0_profiles_x200.npz", HERE / "results" / "truth_summary.json",
              HERE / "results" / "truth.npz",
              HERE / "results" / "sens.npz", HERE / "results" / "stacks.npz",
              STUDY / "I_fitting_a2ii" / "results" / "fits35.json",
              STUDY / "I_fitting_a2ii" / "results" / "ladder_selection_summary.json",
              STUDY / "I_fitting_a2ii" / "results" / "tables" / "I2_ladder_models.csv",
              STUDY / "G_physical_outer" / "results" / "prepare.npz",
              STUDY / "G_physical_outer" / "inputs" / "xi_bias_z0.npz"] + \
    sorted((STUDY / "G_physical_outer").glob("g_*.py")) + sorted((STUDY / "F_observables").glob("f_*.py"))


def code_files():
    return sorted(HERE.glob("*.py")) + [HERE / "inputs" / "j2_fitdata.npz", HERE / "inputs" / "j2_fitdata_meta.json"]


def rels(paths):
    return [str(Path(p).resolve().relative_to(LOCAL_REPO)) for p in paths]


def cmd_upload(dry):
    files = code_files()
    rr = rels(files)
    state = JU.remote_state(rr)
    todo = [(p, r, "missing" if state.get(r) is None else "differs") for p, r in zip(files, rr)
            if state.get(r) is None or state[r][0] != JU.sha(p)]
    print(json.dumps({"files": len(files), "to_upload": [(r, why) for _, r, why in todo]}, indent=1))
    log = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "checked": len(files), "uploaded": []}
    if dry:
        return
    for p, r, why in todo:
        t0 = time.time()
        digest = JU.upload(p, r)
        log["uploaded"].append({"path": r, "reason": why, "bytes": p.stat().st_size, "sha256": digest,
                                "seconds": round(time.time() - t0, 1)})
        print("uploaded", r, flush=True)
    (HERE / "post_j2").mkdir(exist_ok=True)
    (HERE / "post_j2" / "upload_J2.json").write_text(json.dumps(log, indent=1) + "\n")


def cmd_check(out_rel="post_j2/remote_check.json"):
    files = [p for p in READ_BY_J2 + code_files() if Path(p).exists()]
    rr = rels(files)
    state = JU.remote_state(rr)
    out = {"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    for p, r in zip(files, rr):
        s = state.get(r)
        out["files"][r] = {"local": JU.sha(p), "remote": None if s is None else s[0],
                           "remote_is_symlink": None if s is None else s[1]}
    out["all_equal"] = all(v["local"] == v["remote"] for v in out["files"].values())
    out["differ"] = [r for r, v in out["files"].items() if v["local"] != v["remote"]]
    target = HERE / out_rel
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"all_equal": out["all_equal"], "differ": out["differ"]}, indent=1))


def cmd_fetch(dirname="results_j2"):
    assert dirname in ("results_j2", "results_mecke", "results_mecke_checks"), dirname
    source = f"{MIRROR}/{REL}/{dirname}"
    listing = json.loads(remote(f'''import hashlib, json
from pathlib import Path
root = Path({source!r})
print(json.dumps({{str(p.relative_to(root)): [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()]
                   for p in root.rglob('*') if p.is_file() and 'work' not in p.relative_to(root).parts}} if root.exists() else {{}}))
''', python=PY, timeout=900).strip().splitlines()[-1])
    target = HERE / dirname
    got = 0
    for rel, (size, digest) in sorted(listing.items()):
        path = target / rel
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray()
        for off in range(0, size, 8 * 1024**2):
            data += base64.b64decode(remote(f'''import base64
with open({source + '/' + rel!r}, 'rb') as f:
    f.seek({off}); print(base64.b64encode(f.read(8 * 1024 ** 2)).decode())''', python=PY, timeout=600))
        assert hashlib.sha256(bytes(data)).hexdigest() == digest, rel
        path.write_bytes(bytes(data))
        got += 1
    for name in ("test_J_receipt.json", "test_J_done.json", "test_J_failed.json", "test_J.log"):
        text = remote(f'''from pathlib import Path
p = Path({RUNTIME!r}) / 'jobs' / {name!r}
print(p.read_text(errors='replace') if p.exists() else '')''', python=PY, timeout=300)
        if text.strip():
            (target / name).write_text(text)
    print(json.dumps({"files": len(listing), "fetched": got, "MiB": round(sum(v[0] for v in listing.values()) / 2**20, 2)}))


def cmd_cleanup():
    out = remote(f'''import json, shutil, subprocess
from pathlib import Path
p = Path({SCRATCH!r})
size = int(subprocess.run(['du', '-sb', str(p)], capture_output=True, text=True).stdout.split()[0]) if p.exists() else 0
if p.exists():
    shutil.rmtree(p)
print(json.dumps({{'removed': {SCRATCH!r}, 'bytes': size, 'exists_after': p.exists()}}))''', python=PY, timeout=300)
    print(out.strip())


def cmd_record(js, key="j2_maas_runs"):
    assert key in ("j2_maas_runs", "j1_mecke_runs"), key
    spec = json.loads((HERE / "CLUSTER_RUN.json").read_text())
    spec.setdefault(key, []).append(json.loads(js))
    (HERE / "CLUSTER_RUN.json").write_text(json.dumps(spec, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("upload", "check", "fetch", "cleanup", "record"))
    ap.add_argument("payload", nargs="?")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dir", default="results_j2")
    ap.add_argument("--out", default="post_j2/remote_check.json")
    ap.add_argument("--key", default="j2_maas_runs")
    a = ap.parse_args()
    if a.action == "upload":
        cmd_upload(a.dry_run)
    elif a.action == "check":
        cmd_check(a.out)
    elif a.action == "fetch":
        cmd_fetch(a.dir)
    elif a.action == "cleanup":
        cmd_cleanup()
    else:
        cmd_record(a.payload, a.key)


if __name__ == "__main__":
    main()
