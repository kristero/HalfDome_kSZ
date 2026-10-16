"""K1 provenance check (local, small): the per-cell products of selected outputs re-read on maas with the final code
(results_k1check/cells, run 3) against the cells used by the fits (results/cells, K1 of run 1 with an earlier,
unrecorded version of k_read.py / k_common.py).

  python -B k1_check_compare.py [--check DIR] [--results DIR] [--out FILE]
Every array of cells_<snap>.npz is compared: identical (bitwise, NaN = NaN) or the maximum absolute and relative
difference.  comp_err_pub / comp_err_pert (32-node quadrature error of the published and a perturbed law) use
k_fit_core.cumulative, whose large-s branch was corrected for run 3, so tiny differences are expected there only.
The cells_<snap>.json summaries are compared for the cell labels and N; their 'usable_G' lists follow the gas-sphere
rule of the code version that wrote them (k_fits applies the current rule at fit time).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"identical": False, "shape": [list(a.shape), list(b.shape)], "dtype": [str(a.dtype), str(b.dtype)]}
    if a.dtype.kind in "fc":
        same = np.array_equal(a, b, equal_nan=True)
    else:
        same = np.array_equal(a, b)
    out = {"identical": bool(same)}
    if not same and a.dtype.kind in "fc":
        fin = np.isfinite(a) & np.isfinite(b)
        out["nan_pattern_equal"] = bool(np.array_equal(np.isfinite(a), np.isfinite(b)))
        d = np.abs(a[fin].astype(float) - b[fin].astype(float))
        out["max_abs_diff"] = float(d.max()) if d.size else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            r = d / np.abs(b[fin].astype(float))
        out["max_rel_diff"] = float(np.nanmax(r)) if r.size else 0.0
        out["n_diff"] = int(np.sum(d > 0))
    elif not same:
        out["n_diff"] = int(np.sum(a != b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default=str(HERE / "results_k1check"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "results" / "k1_provenance_check.json"))
    a = ap.parse_args()
    chk, ref = Path(a.check), Path(a.results)
    out = {"description": __doc__.strip().splitlines()[0], "outputs": {}}
    man = chk / "run_manifest.json"
    if man.exists():
        inv = json.loads(man.read_text())["invocations"][-1]
        out["check_run"] = {k: inv.get(k) for k in ("host", "started", "finished", "stages", "snaps", "workers", "exit_code",
                                                     "code_sha256", "stage_seconds")}
    all_core_identical = True
    for f in sorted((chk / "cells").glob("cells_*.npz")):
        A, B = np.load(f), np.load(ref / "cells" / f.name)
        res = {}
        keys = sorted(set(A.files) | set(B.files))
        for k in keys:
            if k not in A.files or k not in B.files:
                res[k] = {"identical": False, "missing_in": "check" if k not in A.files else "fit cells"}
                continue
            res[k] = compare(A[k], B[k])
        core = [k for k in keys if not k.startswith("comp_err")]
        ident = all(res[k]["identical"] for k in core)
        all_core_identical &= ident
        ja = json.loads(f.with_suffix(".json").read_text())
        jb = json.loads((ref / "cells" / f.with_suffix(".json").name).read_text())
        lab_a = [(c["label"], c["N"]) for c in ja["cells"]]
        lab_b = [(c["label"], c["N"]) for c in jb["cells"]]
        usable_g_same = [c1["usable_G"] == c2["usable_G"] for c1, c2 in zip(ja["cells"], jb["cells"])]
        out["outputs"][f.stem] = {"arrays_compared": len(keys), "all_arrays_except_comp_err_identical": ident,
                                  "non_identical": {k: v for k, v in res.items() if not v["identical"]},
                                  "cell_labels_and_N_identical": lab_a == lab_b, "n_cells": len(lab_a),
                                  "json_usable_G_lists_identical": int(sum(usable_g_same)),
                                  "json_usable_G_lists_differ": int(len(usable_g_same) - sum(usable_g_same))}
    zc, zr = chk / "z0_study_rows.npz", ref / "z0_study_rows.npz"
    if zc.exists() and zr.exists():
        A, B = np.load(zc), np.load(zr)
        res = {k: compare(A[k], B[k]) if k in B.files else {"identical": False} for k in A.files}
        out["z0_study_rows"] = {"arrays_compared": len(res), "all_identical": all(v["identical"] for v in res.values()),
                                "non_identical": {k: v for k, v in res.items() if not v["identical"]}}
        all_core_identical &= out["z0_study_rows"]["all_identical"]
    out["all_core_arrays_identical"] = bool(all_core_identical and out["outputs"])
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if k != "outputs" else {s: {kk: vv for kk, vv in o.items() if kk != "non_identical"} for s, o in v.items()})
                      for k, v in out.items() if k != "check_run"}, indent=1))


if __name__ == "__main__":
    main()
