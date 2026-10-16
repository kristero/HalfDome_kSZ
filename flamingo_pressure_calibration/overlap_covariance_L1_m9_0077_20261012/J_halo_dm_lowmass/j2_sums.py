"""Checksum manifests of Test J2 (local, read-only except for the two manifests and their log).

  python -B j2_sums.py j1    update, in Test J1's SHA256SUMS, only the lines of the files that J1 and J2 share and that
                            J2 changed after J1's manifest was written (CLUSTER_RUN.json: J2's run records appended;
                            run_test_j.py: the '--j2' hook added at 19:20 on 2026-10-15, before J2's first maas run).
                            Every change is logged with the old and the new digest in post_j2/sums_update_j2.json.
                            J1's own products (results/, post/, figures/, j_*.py) are never re-hashed here.
  python -B j2_sums.py j2    write SHA256SUMS_J2 over J2's files (code, inputs, results_j2/ without work/, results_j2_run1/,
                            post_j2/, figures_j2/, the smoke counterparts, verification_j2.json, cluster_poll_j2.txt,
                            CLUSTER_RUN.json); run it last, after the final verify_j2.py.
Format: '<sha256>  ./<path relative to J_halo_dm_lowmass>' (sha256sum -c compatible from this directory).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = ("CLUSTER_RUN.json", "run_test_j.py")
J1_RUN_SHA = {"run_test_j.py": ("719caec92cefca8c297f9c089d00ed6dfe5ccbc7d195f4c141c8180e09878e3c",
                                "post/remote_code_check.json: the version J1 ran on maas")}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def j1():
    path = HERE / "SHA256SUMS"
    lines = path.read_text().splitlines()
    log_p = HERE / "post_j2" / "sums_update_j2.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else {"updates": []}
    out, changed = [], []
    for line in lines:
        if not line.strip():
            continue
        dig, rel = line.split("  ", 1)
        name = rel[2:] if rel.startswith("./") else rel
        if name in SHARED and (HERE / name).exists():
            new = sha(HERE / name)
            if new != dig:
                changed.append({"file": name, "old": dig, "new": new})
                line = f"{new}  {rel}"
        out.append(line)
    if changed:
        path.write_text("\n".join(out) + "\n")
        log["updates"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "changed": changed})
    log["shared_files"] = list(SHARED)
    log["J1_lines_updated_before_this_log"] = ["run_test_j.py"]      # 20:28 EEST, before j2_sums.py existed
    log["J1_lines_updated"] = sorted({c["file"] for u in log["updates"] for c in u["changed"]} |
                                     set(log["J1_lines_updated_before_this_log"]))
    log["J1_run_versions"] = J1_RUN_SHA
    log["note"] = ("J1's SHA256SUMS covers two files that J2 shares and changed: run_test_j.py (the '--j2' hook, added "
                   "2026-10-15 19:20 EEST before J2's first maas run; J1 itself ran with sha 719caec..., "
                   "post/remote_code_check.json; the line was regenerated at 20:28 EEST by the J2 session) and "
                   "CLUSTER_RUN.json (J2's run records j2_maas_runs appended).  Only these lines are updated; every "
                   "other line of J1's manifest is J1's original digest.")
    log_p.write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps({"changed": changed}, indent=1))


def j2():
    pats = ["cluster_poll_j2.txt", "CLUSTER_RUN.json", "figures_j2/*", "figures_j2_smoke/*", "inputs/j2_fitdata*",
            "j2_*.py", "post_j2/*", "post_j2_smoke/*", "results_j2/**/*", "results_j2_smoke/**/*",
            "results_j2_run1/*", "results_j2_run2/*", "run_test_j2.py", "verification_j2.json", "verify_j2.py"]
    files = set()
    for pat in pats:
        for p in HERE.glob(pat):
            rel = p.relative_to(HERE)
            if p.is_file() and "work" not in rel.parts and "__pycache__" not in rel.parts:
                files.add(rel)
    lines = [f"{sha(HERE / r)}  ./{r}" for r in sorted(files, key=str)]
    (HERE / "SHA256SUMS_J2").write_text("\n".join(lines) + "\n")
    print(f"SHA256SUMS_J2: {len(lines)} files")


if __name__ == "__main__":
    {"j1": j1, "j2": j2}[sys.argv[1]]()
