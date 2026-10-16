#!/usr/bin/env python
"""Test J2 driver: fits of the halo-component definitions D0-D3 (B16 per mass bin with objectives a / d / L, the Lee22
ladder of Test G) on the 14 calibrated bins 12.7-15.5, the low-mass calibration with Test I's recipe, and the exact
evaluation of every painting against Test J1's truth.

  python -B run_test_j2.py --smoke                 local check (J1's smoke bins 13.1-13.3 and 14.5-14.7, 25 haloes per
                                                   bin, Test F's smoke sky; J1's smoke truth re-run in j1base)
  python -B run_test_j2.py --full --workers 8 --scratch /tmp/nagainis_halfdome_testJ
On maas through:  cluster_tests.py run J --workers 8 --rss-stop 24 --driver-args="--j2 --scratch ..."
(run_test_j.py hands every invocation with --j2 to this driver; Test J1's own behaviour is unchanged).
Stages (default all, in this order; each runs in its own Python process):
  j1base   Test J1's stages prepare, basis, fallback (and truth in smoke) re-run with run_test_j.py into
           <results>/j1base: per-halo T1 profiles, the two 14-bin fine bases (JA: A2-ii, JB: host-bound) and the
           fallback pieces (bases in --scratch; J1 deleted its own scratch after its final fetch)
  fits     j2_fits.py fits    (Test G configuration, 35-shell grid): B16 / B12 per bin, RC variant, Test I parity
  ladder   j2_fits.py ladder  Lee22 ladder per definition (+ D0 parity run on Test I's 10 bins)
  eval     j2_eval.py         (Test J configuration): every painting, det and S1, DM PDF passes, errors vs J1 truth
  mecke    j_mecke_atom.py    (only when named; J1 revision 2026-10-16): J1's Mecke quantile Jacobians of D0-D3 with
                              the whole DM = 0 atom, after j1base; outputs mecke_atom.npz / mecke_atom_summary.json /
                              mecke_parity.json in --mecke-out (default --results; e.g. results_mecke)
  mecke_checks  j_mecke_atom.py --checks  (only when named): finite-difference test of J1's truth-mean Mecke maps
                              by radial band and the bootstrap noise of the per-halo subsample maps, after j1base;
                              outputs in --mecke-out (e.g. results_mecke_checks)
Inputs: inputs/j2_fitdata.npz (j2_inputs.py, local), inputs/j0_profiles_x200.npz, Test J1's results/ (truth_summary,
sens) and Test I's results (parity).  Outputs: results_j2/ (results_j2_smoke/), run_manifest.json, SHA256SUMS.
Exit status: 0 = completed; 1 = a stage failed (later stages that need it are skipped); 2 = completed, a parity gate failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("j1base", "fits", "ladder", "eval")
EXTRA_STAGES = ("mecke", "mecke_checks")   # run only when named in --stages (not part of "all")


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
    ap.add_argument("--results", default=None)
    ap.add_argument("--mecke-out", default=None, help="output directory of the stages mecke / mecke_checks "
                    "(default: --results)")
    a = ap.parse_args(argv)
    here = Path(__file__).resolve().parent
    root = Path(os.environ.get("STUDY_ROOT", str(here.parent))).resolve()
    smoke = a.smoke
    results = Path(a.results).resolve() if a.results else here / ("results_j2_smoke" if smoke else "results_j2")
    (results / "tables").mkdir(parents=True, exist_ok=True)
    workers = 1 if smoke else max(1, a.workers)
    j1base = results / "j1base"
    env = dict(os.environ, J_MODE="smoke" if smoke else "full", J2_RESULTS=str(results), STUDY_ROOT=str(root),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stages = STAGES if a.stages == "all" else tuple(s.strip() for s in a.stages.split(","))
    bad = [s for s in stages if s not in STAGES + EXTRA_STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}")
    mpath = results / "run_manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"invocations": []}
    inv = {"mode": "smoke" if smoke else "full", "workers": workers, "study_root": str(root), "results": str(results),
           "scratch": a.scratch, "argv": list(argv if argv is not None else sys.argv[1:]),
           "python": sys.version.split()[0], "host": platform.node(),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stages": {}}
    ins = [here / "inputs" / ("j2_fitdata_smoke.npz" if smoke else "j2_fitdata.npz"), here / "inputs" /
           "j0_profiles_x200.npz", here / "results" / "truth_summary.json", here / "results" / "sens.npz",
           here / "results" / "stacks.npz", root / "I_fitting_a2ii" / "results" / "fits35.json",
           root / "I_fitting_a2ii" / "results" / "tables" / "I2_ladder_models.csv",
           root / "G_physical_outer" / "results" / "f_base" / "profiles.npz",
           root / "G_physical_outer" / "results" / "prepare.npz"]
    inv["inputs_sha256"] = {str(p.relative_to(root)): sha256(p) for p in ins if p.exists()}
    inv["code_sha256"] = {p.name: sha256(p) for p in sorted(here.glob("*.py"))}
    manifest["invocations"].append(inv)
    logs = results / "work" / "stage_logs"
    logs.mkdir(parents=True, exist_ok=True)
    failed, parity_failed = False, False
    t_all = time.perf_counter()
    for st in stages:
        if failed:
            inv["stages"][st] = {"status": "skipped (an earlier stage failed)"}
            continue
        t0 = time.perf_counter()
        if st == "j1base":
            stg = "prepare,basis,fallback" + (",truth" if smoke else "")
            cmd = [sys.executable, "-B", str(here / "run_test_j.py"), "--smoke" if smoke else "--full", "--workers",
                   str(workers), "--results", str(j1base), "--stages", stg]
            if a.scratch:
                cmd += ["--scratch", a.scratch]
            e = dict(env)
        elif st in ("fits", "ladder"):
            cmd = [sys.executable, "-B", str(here / "j2_fits.py"), st, "--workers", str(workers)]
            e = dict(env)
        else:
            cmd = [sys.executable, "-B", str(here / ("j_mecke_atom.py" if st.startswith("mecke") else "j2_eval.py"))]
            cmd += ["--checks"] if st == "mecke_checks" else []
            mout = Path(a.mecke_out).resolve() if a.mecke_out else results
            e = dict(env, J_RESULTS=str(j1base), J_WORKERS=str(workers), MECKE_OUT=str(mout))
            if a.scratch:
                e["J_SCRATCH"] = str(Path(a.scratch).resolve())
        print(time.strftime("%H:%M:%S"), "=== stage", st, " ".join(cmd), flush=True)
        with open(logs / f"{st}.log", "w") as fh:
            p = subprocess.Popen(cmd, env=e, cwd=str(here), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                print(f"[{st}] {line}", end="", flush=True)
            rc = p.wait()
        rec = {"returncode": rc, "wall_seconds": time.perf_counter() - t0, "status": "ok" if rc in (0,) else "failed",
               "peak_rss_largest_child_GiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2}
        if st in ("j1base", "mecke") and rc == 2:
            rec["status"] = f"ok ({'J1' if st == 'j1base' else 'mecke'} parity gate reported failure)"
            parity_failed = True
        elif rc != 0:
            failed = True
            rec["log_tail"] = (logs / f"{st}.log").read_text(errors="replace")[-4000:]
        inv["stages"][st] = rec
        mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    for name in ("fits_parity.json", "ladder_parity.json", "eval_parity.json", "mecke_parity.json"):
        p = results / name
        if p.exists():
            par = json.loads(p.read_text())
            inv.setdefault("parity", {})[name] = par
            if name in ("eval_parity.json", "mecke_parity.json") and par and not par.get("passed", True):
                parity_failed = True
    inv["wall_seconds"] = time.perf_counter() - t_all
    sums = []
    for p in sorted(results.rglob("*")):
        rel = p.relative_to(results)
        if p.is_file() and "work" not in rel.parts and p.name not in ("SHA256SUMS", "run_manifest.json") \
                and not p.name.startswith("test_J"):
            sums.append(f"{sha256(p)}  {rel}")
    (results / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    inv["exit"] = 1 if failed else (2 if parity_failed else 0)
    inv["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mpath.write_text(json.dumps(manifest, indent=1) + "\n")
    print(time.strftime("%H:%M:%S"), f"=== J2 done in {inv['wall_seconds']:.0f} s; exit {inv['exit']}", flush=True)
    return inv["exit"]


if __name__ == "__main__":
    sys.exit(main())
