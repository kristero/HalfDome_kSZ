"""Poll Test J2 (wrapper job test_J) on STRW maas every 5 minutes through cluster_tests.py status (read-only); append
one line per poll to cluster_poll_j2.txt; exit when the wrapper has written test_J_done.json (exit 0) or
test_J_failed.json (exit 3), or after --max-hours (exit 4).  It terminates on the wrapper's status files only (never on
a process-name match).  Usage: python -B j2_poll.py [--interval 300] [--max-hours 4] [--log cluster_poll_j2.txt]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = "/home/cbllover/miniconda3/envs/halfdome/bin/python"


def status():
    r = subprocess.run([PY, "-B", "cluster_tests.py", "status"], cwd=ROOT, capture_output=True, text=True, timeout=240)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=300)
    ap.add_argument("--max-hours", type=float, default=4)
    ap.add_argument("--log", default="cluster_poll_j2.txt")
    a = ap.parse_args()
    t_end = time.time() + 3600 * a.max_hours
    log = HERE / a.log
    while time.time() < t_end:
        try:
            s = status()
            j = s["jobs"].get("test_J", {})
            tail = j.get("log_tail", "").strip().splitlines()
            line = {"local_time": time.strftime("%Y-%m-%d %H:%M:%S"), "load": s["load"],
                    "available_GiB": s["available_GiB"], "alive": j.get("alive"), "tree_RSS_GiB": j.get("tree_RSS_GiB"),
                    "elapsed_min": j.get("elapsed_min"), "done": j.get("done"), "failed": j.get("failed"),
                    "last_log": tail[-1][:200] if tail else ""}
        except Exception as exc:                          # noqa: BLE001 - transient ssh failures: keep polling
            line = {"local_time": time.strftime("%Y-%m-%d %H:%M:%S"), "error": repr(exc)[:300]}
            j = {}
        with open(log, "a") as fh:
            fh.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if j.get("done"):
            return 0
        if j.get("failed"):
            return 3
        time.sleep(a.interval)
    return 4


if __name__ == "__main__":
    sys.exit(main())
