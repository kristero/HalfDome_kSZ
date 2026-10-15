"""Poll the maas job test_I every 5 minutes (cluster_tests.py status) until it is no longer alive; one line per poll
in I_fitting_a2ii/cluster_poll.txt; at most 72 polls (6 h).  Local helper (no effect on the job)."""
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/home/cbllover/miniconda3/envs/halfdome/bin/python"
log = HERE / "cluster_poll.txt"
for i in range(72):
    try:
        out = subprocess.run([PY, "-B", str(HERE.parent / "cluster_tests.py"), "status"], capture_output=True, text=True,
                             timeout=240, cwd=str(HERE.parent)).stdout
        d = json.loads(out)
        j = d["jobs"].get("test_I", {})
        tail = j.get("log_tail", "").strip().splitlines()[-1:] or [""]
        line = (f"{time.strftime('%H:%M:%S')} {'ALIVE' if j.get('alive') else 'ENDED'} load {d['load']} avail "
                f"{d['available_GiB']} rss {j.get('tree_RSS_GiB')} min {j.get('elapsed_min')} | {tail[0][:200]}")
        if not j.get("alive"):
            line += f" | done={j.get('done')} failed={j.get('failed')}"
        with open(log, "a") as fh:
            fh.write(line + "\n")
        if not j.get("alive"):
            break
    except Exception as exc:                      # noqa: BLE001
        with open(log, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} poll error {exc!r}\n")
    time.sleep(300)
