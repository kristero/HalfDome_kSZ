#!/usr/bin/env python
"""Test H driver: sensitivity of tSZ, kSZ and FRB DM at the A2-ii painting target, at the best stored radial
resolution (67 X200 shells to 4 R200c + 8 extension shells to 12 R200c).

  python -B run_test_h.py --smoke                 Test F's smoke configuration (2 calibrated bins, F's 25-halo smoke
                                                  rows per bin, every 25th lightcone redshift shell, 12 multipoles);
                                                  1 worker; local check
  python -B run_test_h.py --full [--workers 8]    the full test (10 calibrated bins, 2,988 haloes, 106 multipoles)
Options: --stages prepare,basis,truth,sens,params,beyond,binning (default all, in this order; each stage reads the
         earlier stages' outputs from the results directory), --results DIR, --scratch DIR (the fine bases, ~2.4 GB,
         go there instead of results/work/g/f_base/work, which becomes a symlink; on maas: a /tmp directory).
Environment: STUDY_ROOT = the study directory (default: the parent of this file's directory).
H0 (h_inputs.py, local, reads the stacking cache) must have been run first: inputs/h0_profiles_x200.npz.

Stages
  prepare  fine per-halo profiles and per-bin truth moments in Test F's profiles format (results/work/g/f_base)
  basis    H1: Test F's in-basis kernels on the fine grid, sets FP (parity), HA (A2-ii), HM (all gas)
  truth    H2: truths, DM PDF, Mecke Jacobians; parity gates against Test F (results/parity.json)
  sens     H3: importance and contribution maps, cumulative signal vs truncation radius, radii containing
           50-99%, profile tolerance per shell, per-mass shares
  params   H6: B12/B16 fits on the fine grid, parameter sensitivities, Fisher, fit influence and fit-transfer
           kernels, 200 paired bootstrap refits
  beyond   H4: painting beyond 4 R200c (zero / B12-B16 extrapolation / log-linear extrapolation), double counting
  binning  H5: mass-binning propagation (6 schemes, piecewise and interpolated painting, R_cut 1-4)
Outputs: compact npz / csv / json in the results directory (big intermediates in work/, not checksummed);
results/run_manifest.json (stage times, peak RSS, inputs read with SHA256), results/SHA256SUMS.
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

STAGES = ("prepare", "basis", "truth", "sens", "params", "beyond", "binning")


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
    ap.add_argument("--scratch", default=None, help="directory for the fine bases (node-local disk on maas)")
    args = ap.parse_args(argv)
    here = Path(__file__).resolve().parent
    os.environ["H_MODE"] = "smoke" if args.smoke else "full"
    os.environ["H_WORKERS"] = "1" if args.smoke else str(max(1, args.workers))
    os.environ.setdefault("STUDY_ROOT", str(here.parent))
    if args.results:
        os.environ["H_RESULTS"] = str(Path(args.results).resolve())
    if args.scratch:
        os.environ["H_SCRATCH"] = str(Path(args.scratch).resolve())
    opened: set = set()
    sys.addaudithook(_audit(opened))
    sys.path.insert(0, str(here))
    import h_common as HC
    stages = STAGES if args.stages == "all" else tuple(s.strip() for s in args.stages.split(","))
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; choose from {STAGES}")
    mpath = HC.RESULTS / "run_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest.setdefault("invocations", [])
    inv = {"mode": HC.MODE, "workers": HC.WORKERS, "study_root": str(HC.ROOT), "results": str(HC.RESULTS),
           "command": " ".join([Path(sys.executable).name, "-B", Path(__file__).name] + list(argv or sys.argv[1:])),
           "python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    try:
        import numpy
        import scipy
        inv["packages"] = {"numpy": numpy.__version__, "scipy": scipy.__version__}
    except Exception as exc:                     # noqa: BLE001
        inv["packages"] = {"error": repr(exc)}
    inv["h0_inputs_sha256"] = sha256(HC.H0_PROFILES) if HC.H0_PROFILES.exists() else None
    inv["scratch"] = HC.SCRATCH
    manifest["invocations"].append(inv)
    failed, parity_failed = False, False
    t_all = time.perf_counter()
    for stage in stages:
        t0 = time.perf_counter()
        HC.log(f"=== stage {stage} ({HC.MODE}, {HC.WORKERS} workers)")
        rec = {}
        n_mem = len(HC.MEM_LOG)
        HC.mem(f"stage {stage} start")
        try:
            if stage == "prepare":
                import h_prepare as M
                M.main()
            elif stage == "basis":
                import h_truth as M
                rec["timing"] = M.basis_stage()
            elif stage == "truth":
                import h_truth as M
                res = M.main()
                rec["parity_passed"] = res["passed"]
                parity_failed = not res["passed"]
            elif stage == "sens":
                import h_sens as M
                M.main()
            elif stage == "beyond":
                import h_beyond as M
                M.main()
            elif stage == "binning":
                import h_binning as M
                M.main()
            elif stage == "params":
                import h_params as M
                M.main()
            rec["status"] = "ok"
        except Exception:                        # noqa: BLE001 - record and continue; exit 1 at the end
            failed = True
            rec["status"] = "failed"
            rec["traceback"] = traceback.format_exc()
            HC.log(f"stage {stage} FAILED\n{rec['traceback']}")
        rec.update(wall_seconds=time.perf_counter() - t0,
                   peak_rss_self_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                   peak_rss_largest_child_GiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2)
        rec["memory_checkpoints"] = HC.MEM_LOG[n_mem:]
        inv["stages"][stage] = rec
        HC.write_json(mpath, manifest)
        HC.release()                           # return the stage's freed heap to the OS before the next stage
    inv["wall_seconds"] = time.perf_counter() - t_all
    prefixes = tuple(str(Path(p).resolve()) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix})
    inputs = []
    files = set()
    for p in opened:
        if "__pycache__" in p:
            q = Path(p)
            p = str(q.parent.parent / (q.name.split(".")[0] + ".py"))
        files.add(p)
    for p in sorted(files):
        if not os.path.isfile(p) or p.startswith(prefixes) or p.startswith(str(here)) or p.startswith(str(HC.RESULTS)):
            continue
        if p.startswith(("/proc", "/sys", "/dev", "/etc", "/usr")) or "/.cache/" in p:
            continue
        inputs.append({"path": p, "relative_to_study_root": os.path.relpath(p, HC.ROOT), "bytes": os.path.getsize(p),
                       "sha256": sha256(p)})
    inv["inputs_read"] = inputs
    inv["h_inputs_read"] = [{"path": p, "sha256": sha256(p)} for p in sorted(opened)
                            if os.path.isfile(p) and p.startswith(str(here / "inputs"))]
    sums = []
    for p in sorted(HC.RESULTS.rglob("*")):
        rel = p.relative_to(HC.RESULTS)
        if p.is_file() and "work" not in rel.parts and p.name not in ("SHA256SUMS", "run_manifest.json") \
                and not p.name.startswith("test_H"):
            sums.append(f"{sha256(p)}  {rel}")
    (HC.RESULTS / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    inv["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    inv["exit"] = 1 if failed else (2 if parity_failed else 0)
    HC.write_json(mpath, manifest)
    HC.log(f"=== done in {inv['wall_seconds']:.0f} s; exit {inv['exit']}; manifest {mpath}")
    return inv["exit"]


if __name__ == "__main__":
    sys.exit(main())
