"""READ-ONLY provenance check of Test H's code on STRW maas (local helper; nothing is written on maas).

Hashes (sha256) every H_sensitivity/*.py in the maas runtime mirror and every local H_sensitivity/*.py, and records
which are equal, which differ, and which exist on one side only.  Writes post/remote_code_check.json, the input of
verify_h.py gate G8 (compute modules equal the code that ran; the check is current, i.e. no local .py changed after
it).  Run it AFTER the last local code edit and BEFORE verify_h.py.
Transport: Windows OpenSSH alias strw-maas (ProxyJump strw), Python on stdin, as cluster_tests.py / remote_control.py.
Usage: python -B remote_code_check.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SSH = "/mnt/c/Windows/System32/OpenSSH/ssh.exe"
ROOT = ("/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/flamingo_pressure_calibration/"
        "overlap_covariance_L1_m9_0077_20261012/H_sensitivity")
PROG = f'''import hashlib, json
from pathlib import Path
root = Path({ROOT!r})
print(json.dumps({{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}}))
'''


def main():
    r = subprocess.run([SSH, "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "strw-maas", "python3", "-"],
                       input=PROG, text=True, capture_output=True, timeout=300)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    remote = json.loads(r.stdout.strip().splitlines()[-1])
    local = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob("*.py"))}
    out = {"checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
           "remote_root": ROOT, "remote": remote, "local": local,
           "equal": sorted(f for f in local if remote.get(f) == local[f]),
           "differ": sorted(f for f in local if f in remote and remote[f] != local[f]),
           "local_only": sorted(f for f in local if f not in remote),
           "remote_only": sorted(f for f in remote if f not in local),
           "note": "read-only; written by remote_code_check.py (round-3 fix, 2026-10-14) after the last local code "
                   "edit; the compute modules (h_basis ... h_truth, run_test_h.py) must be in 'equal'"}
    (HERE / "post" / "remote_code_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("checked_utc", "equal", "differ", "local_only", "remote_only")}, indent=1))


if __name__ == "__main__":
    main()
