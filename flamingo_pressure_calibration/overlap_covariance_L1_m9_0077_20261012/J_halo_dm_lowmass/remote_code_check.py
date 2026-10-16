"""READ-ONLY provenance check of Test J's code on STRW maas (local helper; nothing is written on maas).

Hashes (sha256) every J_halo_dm_lowmass/*.py in the maas runtime mirror and every local J_halo_dm_lowmass/*.py, and
records which are equal, which differ and which exist on one side only.  The compute modules (COMPUTE) must be equal:
they are the code that produced results/.  Writes post/remote_code_check.json (input of verify_j.py gate G11).
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
        "overlap_covariance_L1_m9_0077_20261012/J_halo_dm_lowmass")
COMPUTE = ["j_common.py", "j_prepare.py", "j_basis.py", "j_fallback.py", "j_obs.py", "j_pdf.py", "j_truth.py",
           "j_sens.py", "run_test_j.py"]
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
           "compute_modules": COMPUTE,
           "compute_modules_equal": all(remote.get(f) == local.get(f) for f in COMPUTE)}
    (HERE / "post").mkdir(exist_ok=True)
    (HERE / "post" / "remote_code_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("checked_utc", "equal", "differ", "local_only", "remote_only",
                                          "compute_modules_equal")}, indent=1))


if __name__ == "__main__":
    main()
