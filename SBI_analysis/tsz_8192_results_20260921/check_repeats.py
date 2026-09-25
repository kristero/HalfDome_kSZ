"""Compare completed recovery spectra with their retained original controls.

Noise draws have new seeds in recovery and must not be compared elementwise.
The clean D_ell arrays have identical physical inputs and should agree to
floating-point precision. A 1e-10 relative-norm tolerance allows harmless
thread-order differences without accepting any scientific resolution error.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "tsz_8192_validation_20260920"
NEW = ROOT.parent / "tsz_8192_validation_recovery_20260921"


def main():
    plan = json.loads((NEW / "plan.json").read_text())
    ell = np.arange(7980, dtype=float)
    weight = ell * (ell + 1) / (2 * np.pi)
    records = []
    for task in plan["tasks"]:
        if "repeats_task" not in task:
            continue
        current = NEW / "controls" / f"{task['id']:03d}"
        marker = current / "status.json"
        if not marker.exists() or json.loads(marker.read_text())["returncode"] != 0:
            records.append(dict(task_id=task["id"], status="pending"))
            continue
        previous = OLD / "controls" / f"{task['repeats_task']:03d}"
        a = np.load(previous / "masked_clean_cl.npy") * weight
        b = np.load(current / "masked_clean_cl.npy") * weight
        assert a.shape == b.shape == (7980,)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        relative = float(np.linalg.norm(a[80:] - b[80:]) / np.linalg.norm(a[80:]))
        assert relative < 1e-10, (task["id"], relative)
        records.append(dict(task_id=task["id"], repeats=task["repeats_task"],
                            label=task["label"], status="passed",
                            relative_dl_norm=relative))
    result = dict(utc=datetime.now(timezone.utc).isoformat(),
                  tolerance=1e-10, records=records)
    (ROOT / "results/recheck_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
