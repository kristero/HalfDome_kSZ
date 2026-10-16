#!/usr/bin/env python
"""Test L1 driver: where does the halo end?  Truncation radii from the bound gas (FLAMINGO L1_m9, z = 0).

  python -B run_test_l.py --smoke     2 bins (13.1-13.3, 14.5-14.7), 20 bootstrap draws -> results_smoke/, figures_smoke/
  python -B run_test_l.py --full      14 calibrated bins 12.7-15.5, 200 paired bootstrap draws -> results/, figures/
Options: --stages inputs,stacks,radii,global,codex,extrap,figures,numbers,verify (default: all in this order; 'inputs' is
         shared by both modes and writes inputs/).
Local only (stack-level work on the per-halo cache; 2 threads, < 0.5 GB RSS, about 1 min in all): nothing of L1 needs
STRW maas.  The observable-optimal single cut is computed here on the exact additive (1-halo / linear) terms only
(global.json: observable_optimal); with two-halo terms, total spectra and quantiles (exact painting of per-bin and global
cuts) it is Test L2.
Run with: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH=<halfdome env>/lib <halfdome env>/bin/python -B run_test_l.py

Stages
  inputs    l_inputs.py   partition complement (p10, p11), central/host-bound owned counts (streamed), SOAP R200m
  stacks    l_stacks.py   per-bin stacks + 200 paired bootstrap rows, merged groups (J's 400-particle rule), fractions
  radii     l_radii.py    every truncation rule per bin and field, unit conversions, inside fractions
  global    l_global.py   signal-weighted radii, mass dependence, exact additive terms from J's truncation curves
  codex     l_codex.py    Codex HalfDome shell cross-spectra, FLAMINGO/HalfDome ratios, redshift/mass shares, pilot
  extrap    l_extrap.py   the region R_t - 4 R200c that the painted law extrapolates into (added 2026-10-16 after the
                          user's clarification: R_t is the FIT range; HalfDome still paints the fitted law to 4 R200c):
                          shares of the 4R content beyond R_t for all A2-ii gas, bound gas and the published B12/B16
                          shapes, the mean-DM bracket, and the map shares beyond x_t
  figures   l_figures.py
  numbers   l_numbers.py  the numbers quoted in the report, each with its source file and field
  verify    verify_l.py   gates -> verification.json (verification_smoke.json)
Outputs: results/run_manifest.json (stage times, peak RSS, files read with SHA256), SHA256SUMS (written by l_sums.py).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

STAGES = ("inputs", "stacks", "radii", "global", "codex", "extrap", "figures", "numbers", "verify")
MODULES = {"inputs": "l_inputs", "stacks": "l_stacks", "radii": "l_radii", "global": "l_global", "codex": "l_codex", "extrap": "l_extrap",
           "figures": "l_figures", "numbers": "l_numbers", "verify": "verify_l"}


def _audit(opened: set):
    def hook(event, args):
        if event == "open" and args:
            path, mode = args[0], (args[1] if len(args) > 1 else None)
            if isinstance(path, (str, bytes, os.PathLike)) and (mode is None or not any(c in str(mode) for c in "wax+")):
                try:
                    opened.add(os.path.abspath(os.fsdecode(path)))
                except Exception:      # noqa: BLE001
                    pass
    return hook


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    # Test L2 (STRW maas, 2026-10-16): cluster_tests.py always starts this driver; '--l2' hands the call to
    # run_test_l2.py.  Without '--l2' everything below is Test L1's driver, unchanged.
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--l2" in argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_test_l2
        return run_test_l2.main([x for x in argv if x != "--l2"])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--stages", default="all")
    a = ap.parse_args(argv)
    os.environ["L_MODE"] = "smoke" if a.smoke else "full"
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "2"
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    stages = STAGES if a.stages == "all" else tuple(s.strip() for s in a.stages.split(","))
    opened: set = set()
    sys.addaudithook(_audit(opened))
    import l_common as LC
    man_path = LC.RESULTS / "run_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"invocations": []}
    inv = {"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "mode": LC.MODE, "stages": list(stages),
           "host": platform.node(), "python": sys.version.split()[0], "argv": sys.argv, "stage_seconds": {},
           "stage_peak_rss_gb": {}, "status": "running"}
    rc = 0
    for s in stages:
        t0 = time.perf_counter()
        try:
            LC.log(f"stage {s}")
            importlib.import_module(MODULES[s]).main()
        except Exception:      # noqa: BLE001
            inv["traceback"] = traceback.format_exc()
            inv["failed_stage"] = s
            rc = 1
            break
        finally:
            inv["stage_seconds"][s] = round(time.perf_counter() - t0, 2)
            inv["stage_peak_rss_gb"][s] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
    inv["status"] = "completed" if rc == 0 else "failed"
    inv["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    keep = sorted(p for p in opened if p.startswith(str(LC.FLAM)) and Path(p).is_file()
                  and "/L_halo_truncation/" not in p and "__pycache__" not in p and not p.endswith(".py"))
    inv["inputs_read"] = {str(Path(p).relative_to(LC.FLAM)): sha256(p) for p in keep}
    manifest["invocations"].append(inv)
    man_path.write_text(json.dumps(manifest, indent=1))
    LC.log(f"done rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
