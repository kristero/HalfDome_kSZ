#!/usr/bin/env python
"""Test G driver: physically motivated outer components, calibrated per field (pressure on tSZ; electrons on kSZ
and the DM PDF), with a Lee22-style parametrisation ladder.

  python -B run_test_g.py --smoke                 Test F's smoke configuration (2 calibrated bins, 25 haloes per bin,
                                                  every 25th lightcone redshift shell, 12 multipoles, 2 folds),
                                                  R_fit / R_cut in {2, 4}; 1 worker; local check (< 10 min)
  python -B run_test_g.py --full [--workers 4]    the full test (10 calibrated bins, 2,988 haloes, 106 multipoles,
                                                  8 folds, R_fit / R_cut in {1, 1.5, 2, 3, 4})
Options: --stages inputs,kernels,prepare,parity,fits,evaluate,selection,cv (default all, in this order; each stage
         reads the earlier stages' outputs from the results directory, so a failed stage can be rerun alone),
         --results DIR (default G_physical_outer/results, or results_smoke).
Environment: STUDY_ROOT = the study directory (default: the parent of this file's directory).

Stages
  inputs     full: copy Test F's products (profiles, truth, importance, models) to results/f_base;
             smoke: regenerate them with Test F's own stages (smoke mode) in results_smoke/f_base
  kernels    Test F's per-shell lightcone basis (set T) rebuilt in f_base/work (F's work/ was deleted on maas)
  prepare    normalisation checks, 2-halo templates, temperatures, splashback scales, p_keep, S1/S2 tables
  parity     rebuilt basis and DM-PDF machinery vs Test F's truth; the published-fallback PDF slot
  fits       every field model x fit target x objective x R_fit; sigma_d; AIC/BIC
  evaluate   observables of every fit x R_cut x target (det, S1, S2), DM-PDF passes, tSZ x DM combinations
  selection  Lee22-style parametrisation ladder (reduced chi^2, BIC, spatial CV; fewest parameters) and the
             observables of the selected minimal models
  cv         spatial-block cross-validation of the per-field candidate choices
Outputs: compact npz / csv / json in the results directory (big intermediates in work/ and f_base/work/, not
checksummed); results/run_manifest.json (stage times, peak RSS, inputs read with SHA256), results/SHA256SUMS.
Exit status: 0 = completed; 1 = a stage raised (traceback in run_manifest.json); 2 = completed but the parity gate
failed (inspect parity.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

STAGES = ("inputs", "kernels", "prepare", "parity", "fits", "evaluate", "selection", "cv")


def _audit(opened: set):
    def hook(event, args):
        if event == "open" and args:
            path, mode = args[0], (args[1] if len(args) > 1 else None)
            if isinstance(path, (str, bytes, os.PathLike)) and (mode is None or not any(c in str(mode) for c in "wax+")):
                try:
                    opened.add(os.path.abspath(os.fsdecode(path)))
                except Exception:      # noqa: BLE001 - never let the audit hook fail a run
                    pass
    return hook


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)
    here = Path(__file__).resolve().parent
    os.environ["G_MODE"] = "smoke" if args.smoke else "full"
    os.environ["G_WORKERS"] = "1" if args.smoke else str(max(1, args.workers))
    os.environ.setdefault("STUDY_ROOT", str(here.parent))
    if args.results:
        os.environ["G_RESULTS"] = str(Path(args.results).resolve())
    opened: set = set()
    sys.addaudithook(_audit(opened))
    sys.path.insert(0, str(here))
    import g_common as GC
    stages = STAGES if args.stages == "all" else tuple(s.strip() for s in args.stages.split(","))
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    mpath = GC.RESULTS / "run_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.setdefault("invocations", [])
    inv = {"mode": GC.MODE, "workers": GC.WORKERS, "study_root": str(GC.ROOT), "results": str(GC.RESULTS),
           "command": " ".join([Path(sys.executable).name, "-B", Path(__file__).name] + list(argv or sys.argv[1:])),
           "python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    try:
        import numpy
        import scipy
        import matplotlib  # noqa: F401  (imported by Test F's frozen engine modules)
        inv["packages"] = {"numpy": numpy.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    except Exception as exc:                     # noqa: BLE001
        inv["packages"] = {"error": repr(exc)}
    manifest["invocations"].append(inv)
    failed, parity_failed = False, False
    t_all = time.perf_counter()
    for stage in stages:
        t0 = time.perf_counter()
        GC.log(f"=== stage {stage} ({GC.MODE}, {GC.WORKERS} workers)")
        rec = {}
        try:
            if stage == "inputs":
                import g_prepare as M
                M.inputs()
            elif stage == "kernels":
                basis = GC.FBASE / "work" / "basis_T.npz"
                if GC.SMOKE and basis.exists():
                    rec["note"] = "smoke: basis_T built by the inputs stage (Test F's smoke kernels), reused"
                else:
                    import f_kernels
                    rec["timing"] = f_kernels.build(sets=("T",), workers=GC.WORKERS)
            elif stage == "prepare":
                import g_prepare as M
                M.main()
            elif stage == "parity":
                import g_parity as M
                res = M.main()
                rec["passed"] = res["passed"]
                parity_failed = not res["passed"]
            elif stage == "fits":
                import g_fits as M
                M.main()
            elif stage == "evaluate":
                import g_evaluate as M
                M.main()
            elif stage == "selection":
                import g_selection as M
                M.main()
            elif stage == "cv":
                import g_cv as M
                M.main()
            rec["status"] = "ok"
        except Exception:                        # noqa: BLE001 - record and continue; exit 1 at the end
            failed = True
            rec["status"] = "failed"
            rec["traceback"] = traceback.format_exc()
            GC.log(f"stage {stage} FAILED\n{rec['traceback']}")
        rec.update(wall_seconds=time.perf_counter() - t0,
                   peak_rss_self_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                   peak_rss_largest_child_GiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2)
        inv["stages"][stage] = rec
        GC.write_json(mpath, manifest)
    inv["wall_seconds"] = time.perf_counter() - t_all
    prefixes = tuple(str(Path(p).resolve()) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix})
    inputs = []
    files = set()
    for p in opened:
        if "__pycache__" in p:                     # byte code read from an existing cache -> its source file
            q = Path(p)
            p = str(q.parent.parent / (q.name.split(".")[0] + ".py"))
        files.add(p)
    for p in sorted(files):
        if not os.path.isfile(p) or p.startswith(prefixes) or p.startswith(str(here)) or p.startswith(str(GC.RESULTS)):
            continue
        if p.startswith(("/proc", "/sys", "/dev", "/etc", "/usr")) or "/.cache/" in p:
            continue
        inputs.append({"path": p, "relative_to_study_root": os.path.relpath(p, GC.ROOT), "bytes": os.path.getsize(p),
                       "sha256": sha256(p)})
    inv["inputs_read"] = inputs
    inv["g_inputs_read"] = [{"path": p, "sha256": sha256(p)} for p in sorted(opened)
                            if os.path.isfile(p) and p.startswith(str(here / "inputs"))]
    sums = []
    for p in sorted(GC.RESULTS.rglob("*")):
        rel = p.relative_to(GC.RESULTS)
        if p.is_file() and "work" not in rel.parts and p.name not in ("SHA256SUMS", "run_manifest.json"):
            sums.append(f"{sha256(p)}  {rel}")
    (GC.RESULTS / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    inv["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inv["exit"] = 1 if failed else (2 if parity_failed else 0)
    GC.write_json(mpath, manifest)
    GC.log(f"=== done in {inv['wall_seconds']:.0f} s; exit {inv['exit']}; manifest {mpath}")
    return inv["exit"]


if __name__ == "__main__":
    sys.exit(main())
