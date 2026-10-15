#!/usr/bin/env python
"""Test F driver: observable-driven calibration of the FLAMINGO L1_m9 z = 0 pressure / electron profiles.

  python -B run_test_f.py --smoke                 tiny subset (2 mass bins, 50 haloes, 20 bootstrap draws,
                                                  every 25th lightcone redshift shell, 12 multipoles, 2 CV
                                                  folds, 1 worker); local check, < 5 min
  python -B run_test_f.py --full [--workers 4]    the full test (10 calibrated bins, 2,988 haloes, 1000
                                                  bootstrap draws, 421 shells, 106 multipoles, 8 CV folds)
Options: --stages profiles,kernels,validate,truth,importance,fits,engine,cv (default: all, in this order),
         --results DIR (default F_observables/results, or results_smoke).
Environment: STUDY_ROOT = the study directory (default: the parent of this file's directory).  No absolute
path is hard-coded: earlier studies and the engine inputs are reached by the analysis library and
kernels/engine.py relative to their own location (STUDY_ROOT/.. and STUDY_ROOT/../..).

Stages
  profiles    per-halo P/P200 and n_e profiles on the 35-shell F grid (4 gas/AGN selections), resolution
              masks, per-bin truth moments, spatial CV folds                         -> profiles.npz
  kernels     exact per-shell lightcone basis kernels (validation set V, production set T)  -> work/basis_*.npz
  validate    in-basis machinery vs the engine (moments, Jacobians, DM quantiles and their Jacobians)
  truth       scatter-inclusive truth (per-halo second moments; per-halo DM PDF), controls, lognormal
              amplitude models, Mecke quantile Jacobians                              -> truth.npz
  importance  F1 importance of every shell and mass bin; weights (objective d) and metric K (objective e)
  fits        F2-F4 on the full sample: every method, deterministic and scatter-aware predictions, DM PDF of
              every model, profile residuals; controls (AGN kept, A2-i, A2-ii)         -> tables/F_*.csv
  engine      engine-exact check of the basis representation and of the flux-matched logistic edge
  cv          F5 spatial-block cross-validation (out-of-fold predictions)              -> tables/F5_*.csv
Outputs (compact npz / csv / json, no figures) in the results directory; big intermediates in
<results>/work (not needed for the figures).  Figures and report tables: make_figures.py (local).
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
from pathlib import Path

STAGES = ("profiles", "kernels", "validate", "truth", "importance", "fits", "engine", "cv")


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
    os.environ["F_MODE"] = "smoke" if args.smoke else "full"
    os.environ["F_WORKERS"] = "1" if args.smoke else str(max(1, args.workers))
    os.environ.setdefault("STUDY_ROOT", str(here.parent))
    if args.results:
        os.environ["F_RESULTS"] = str(Path(args.results).resolve())
    opened: set = set()
    sys.addaudithook(_audit(opened))
    sys.path.insert(0, str(here))
    import f_common as FC                                       # configuration from the environment
    stages = STAGES if args.stages == "all" else tuple(s.strip() for s in args.stages.split(","))
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    manifest = {"mode": FC.MODE, "workers": FC.WORKERS, "study_root": str(FC.ROOT), "results": str(FC.RESULTS),
                "command": " ".join([Path(sys.executable).name, "-B", Path(__file__).name] + list(argv or sys.argv[1:])),
                "python": sys.version.split()[0], "platform": platform.platform(), "started_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    try:
        import numpy, scipy, matplotlib                          # noqa: E401  (matplotlib: imported by frozen modules)
        manifest["packages"] = {"numpy": numpy.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    except Exception as exc:                                     # noqa: BLE001
        manifest["packages"] = {"error": repr(exc)}
    t_all = time.perf_counter()
    for stage in stages:
        t0 = time.perf_counter()
        FC.log(f"=== stage {stage} ({FC.MODE}, {FC.WORKERS} workers)")
        result = None
        if stage == "profiles":
            import f0_profiles as M
            M.main()
        elif stage == "kernels":
            import f_kernels as M
            result = M.build(sets=("V", "T"), workers=FC.WORKERS)
        elif stage == "validate":
            import f1_validate as M
            result = M.main()
            if result is not None and not result.get("passed_all", False):
                manifest["validation_failed"] = True
        elif stage == "truth":
            import f1_truth as M
            M.main()
        elif stage == "importance":
            import f1_importance as M
            M.main()
        elif stage == "fits":
            import f2_run as M
            M.main()
        elif stage == "engine":
            import f4_engine_check as M
            M.main()
        elif stage == "cv":
            import f5_cv as M
            M.main()
        manifest["stages"][stage] = {
            "wall_seconds": time.perf_counter() - t0,
            "peak_rss_self_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
            "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2}
        if stage == "validate" and result is not None:
            manifest["stages"][stage]["passed"] = result.get("passed")
        FC.write_json(FC.RESULTS / "run_manifest.json", manifest)
    manifest["wall_seconds"] = time.perf_counter() - t_all
    # inputs read outside F_observables (and outside the Python installation)
    prefixes = tuple(str(Path(p).resolve()) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix})
    code = str(here)
    inputs = []
    for p in sorted(opened):
        if not os.path.isfile(p) or p.startswith(prefixes) or p.startswith(code) or p.startswith(str(FC.RESULTS)):
            continue
        if p.startswith(("/proc", "/sys", "/dev", "/etc", "/usr")):
            continue
        inputs.append({"path": p, "relative_to_study_root": os.path.relpath(p, FC.ROOT), "bytes": os.path.getsize(p),
                       "sha256": sha256(p)})
    manifest["inputs_read"] = inputs
    # checksums of the compact outputs
    sums = []
    for p in sorted(FC.RESULTS.rglob("*")):
        if p.is_file() and "work" not in p.relative_to(FC.RESULTS).parts and p.name not in ("SHA256SUMS", "run_manifest.json"):
            sums.append(f"{sha256(p)}  {p.relative_to(FC.RESULTS)}")
    (FC.RESULTS / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    FC.write_json(FC.RESULTS / "run_manifest.json", manifest)
    FC.log(f"=== done in {manifest['wall_seconds']:.0f} s; manifest {FC.RESULTS / 'run_manifest.json'}")
    return 0 if not manifest.get("validation_failed") else 2


if __name__ == "__main__":
    sys.exit(main())
