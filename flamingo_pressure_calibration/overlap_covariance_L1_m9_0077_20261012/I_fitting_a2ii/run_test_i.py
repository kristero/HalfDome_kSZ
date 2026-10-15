#!/usr/bin/env python
"""Test I driver: fitting the A2-ii painting stacks (B12 / B16 refits and the Lee22 parametrisation ladder), evaluated
on Test H's fine A2-ii basis against the A2-ii truth.

  python -B run_test_i.py --smoke                local check (Test F's smoke configuration through G's / H's smoke
                                                 products; DM PDF limited with I_PDF_LIMIT)
  python -B run_test_i.py --full --workers 8     the full test (on STRW maas through cluster_tests.py run I)
Options: --stages fits35,finefits,ladder,lvariants,evaluate (default all, in this order; the supplement stages
suppbasis,supp (run 2) and phfits,pheval (run 3) run only when named); --scratch DIR (node-local directory holding the fine A2-ii basis: a hard link of
Test H's in run 1, the rebuilt basis in the supplement); --cleanup-scratch (after every stage succeeded: delete DIR -
Test I's own scratch only; run 1's version also deleted Test H's scratch, see attempts/run1_20261014/); --h-results DIR
(Test H's results; default H_sensitivity/results).
Each stage runs in its own Python process (i_g35.py: Test G's configuration on the 35-shell grid; i_fine.py: Test
H's configuration on the fine grid), because Test H patches Test F's grid on the imported module object.
Stages
  fits35    I1 B12 / B16 fits on the 35-shell grid, objectives a / d / L, 6 R_fit; I5 B12 / B16 variant refits
  finefits  I1 B12 / B16 objective (a) on the fine 67-shell grid (grid dependence)
  ladder    I2 Test G's Lee22 ladder (g_selection) on the A2-ii stacks
  lvariants I5 selected Lee22 model and PL3 refitted to the stacking-choice variants
  evaluate  I1 / I3 / I5 observables on the fine A2-ii basis (det, S1; DM PDF passes) and sensitive-part scores
  suppbasis (supplement, i_supp.py) Test H's fine A2-ii basis rebuilt into --scratch (run 1 deleted Test H's scratch)
  supp      (supplement, i_supp.py) I1 objective L with objective-(a) bins where no L fit exists; I3 at every R_cut;
            run-1 parity paintings; outputs in results/supp/
  phfits    (run 3, i_perhalo.py) the per-halo resolution-rule variant refitted on its OWN resolved shells (runs 1-2
            fitted it on the strict shells, where it equals A2-ii): B12 / B16 objective (a) and the selected Lee22
            model + PL3, R_fit 2 and 4; strict-mask refits as parity with run 1; results/perhalo/phfits.json
  pheval    (run 3, i_perhalo.py) i_fine.stage_evaluate on those paintings, the A2-ii and per-halo-rule fine stacks
            themselves, and three run-1 parity paintings; outputs in results/perhalo/
Outputs: results/ (fits35.json, finefits.json, ladder_selection_summary.json, ladder_selection.npz, lvariants.json,
evaluate.npz, evaluate_items.json, tables/*.csv, run_manifest.json, SHA256SUMS; intermediates in results/work/).
Exit status: 0 = completed; 1 = a stage failed (its log tail in run_manifest.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("fits35", "finefits", "ladder", "lvariants", "evaluate", "suppbasis", "supp", "phfits", "pheval")
DEFAULT = ("fits35", "finefits", "ladder", "lvariants", "evaluate")
SCRIPT = {"fits35": "i_g35.py", "ladder": "i_g35.py", "lvariants": "i_g35.py", "finefits": "i_fine.py",
          "evaluate": "i_fine.py", "suppbasis": "i_supp.py", "supp": "i_supp.py", "phfits": "i_perhalo.py",
          "pheval": "i_perhalo.py"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--cleanup-scratch", action="store_true")
    ap.add_argument("--h-results", default=None)
    ap.add_argument("--results", default=None)
    a = ap.parse_args(argv)
    here = Path(__file__).resolve().parent
    root = Path(os.environ.get("STUDY_ROOT", str(here.parent))).resolve()
    smoke = a.smoke
    results = Path(a.results).resolve() if a.results else here / ("results_smoke" if smoke else "results")
    hres = Path(a.h_results).resolve() if a.h_results else root / "H_sensitivity" / ("results_smoke" if smoke else "results")
    results.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, I_MODE="smoke" if smoke else "full", I_RESULTS=str(results), I_H_RESULTS=str(hres),
               STUDY_ROOT=str(root), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    workers = 1 if smoke else max(1, a.workers)
    stages = DEFAULT if a.stages == "all" else tuple(s.strip() for s in a.stages.split(","))
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}")
    mpath = results / "run_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"invocations": []}
    inv = {"mode": env["I_MODE"], "workers": workers, "study_root": str(root), "results": str(results),
           "h_results": str(hres), "scratch": a.scratch, "argv": sys.argv[1:], "python": sys.version.split()[0],
           "host": platform.node(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    inv["h_inputs_sha256"] = {n: sha256(hres / n) for n in ("truth.npz", "truth_summary.json", "sens.npz")
                              if (hres / n).exists()}
    manifest["invocations"].append(inv)
    failed = False
    t_all = time.perf_counter()
    logs = results / "work" / "stage_logs"
    logs.mkdir(parents=True, exist_ok=True)
    for st in stages:
        t0 = time.perf_counter()
        cmd = [sys.executable, "-B", str(here / SCRIPT[st]), st, "--workers", str(workers)]
        if SCRIPT[st] in ("i_fine.py", "i_supp.py", "i_perhalo.py") and a.scratch:
            cmd += ["--scratch", a.scratch]
        print(time.strftime("%H:%M:%S"), "=== stage", st, " ".join(cmd), flush=True)
        with open(logs / f"{st}.log", "w") as fh:
            p = subprocess.Popen(cmd, env=env, cwd=str(here), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                print(f"[{st}] {line}", end="", flush=True)
            rc = p.wait()
        rec = {"returncode": rc, "wall_seconds": time.perf_counter() - t0,
               "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2,
               "status": "ok" if rc == 0 else "failed"}
        if rc != 0:
            failed = True
            rec["log_tail"] = (logs / f"{st}.log").read_text(errors="replace")[-4000:]
        inv["stages"][st] = rec
        mpath.write_text(json.dumps(manifest, indent=1) + "\n")
        if rc != 0 and st in ("fits35", "ladder", "suppbasis", "phfits"):
            print("stage", st, "failed; later stages depend on it - stopping", flush=True)
            break
    if a.cleanup_scratch and not failed:
        removed = []
        for d in (a.scratch,):                       # Test I's own scratch only (never Test H's)
            if d and Path(d).exists() and "testH" not in Path(d).name:
                shutil.rmtree(d)
                removed.append(d)
        inv["scratch_removed"] = removed
        inv["scratch_removed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inv["wall_seconds"] = time.perf_counter() - t_all
    sums = []
    for p in sorted(results.rglob("*")):
        rel = p.relative_to(results)
        if p.is_file() and "work" not in rel.parts and p.name not in ("SHA256SUMS", "run_manifest.json") \
                and not p.name.startswith("test_I"):
            sums.append(f"{sha256(p)}  {rel}")
    (results / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    inv["exit"] = 1 if failed else 0
    inv["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    print(time.strftime("%H:%M:%S"), f"=== done in {inv['wall_seconds']:.0f} s; exit {inv['exit']}", flush=True)
    return inv["exit"]


if __name__ == "__main__":
    sys.exit(main())
