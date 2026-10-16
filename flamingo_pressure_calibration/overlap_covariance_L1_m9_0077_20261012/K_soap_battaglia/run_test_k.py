#!/usr/bin/env python
"""Test K driver: Battaglia B12 (pressure) / B16 (gas density) laws refitted to FLAMINGO L1_m9 HYDRO_FIDUCIAL SOAP-HBT
aperture integrals at all 71 outputs (0007-0077), forward-modelled at every halo's own aperture radii.

  python -B run_test_k.py --full --workers 8 --scratch /tmp/nagainis_halfdome_testK [--cleanup-scratch]   (maas)
  python -B run_test_k.py --smoke [--workers 2] [--scratch DIR]       maas smoke: outputs 0077 and 0037, first
                                                                      --limit rows (default 3,000,000), 4 bootstraps
  python -B run_test_k.py --smoke --synthetic                         local smoke without SOAP: synthetic haloes from
                                                                      the published laws through the same reduction
Options: --stages k0,k1,k2,k3 (default all, in order); --snaps 77,76 (subset of outputs); --results DIR;
         --boot orig,other,check (bootstrap refits); --starts FILE (extra K3 start vectors); --keep-halo 0|1.
Stages
  k0  inventory: headers, cosmology, AGN filter and column metadata of the 71 files; smoke read of 0077
  k1  one worker per output: read the selected centrals, normalise, per-halo arrays -> scratch, per-cell reduction ->
      results/cells/cells_<snap>.npz/.json (k_read.py)
  k2  per-cell free fits (k_fits.stage_k2) -> results/k2_cell_fits.json
  k3  global fits, bootstrap, published-law scores, predictions, parameter file (k_fits.stage_k3)
Outputs: results/ (compact; per-halo arrays only in the scratch), results/run_manifest.json, results/SHA256SUMS.
Exit status: 0 = completed; 1 = a stage raised (traceback in run_manifest.json); 3 = some outputs failed in k1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

STAGES = ("k0", "k1", "k2", "k3")
HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def peak_rss_gib():
    return max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 2 ** 20


def synthetic_halos(snap, z, rng, n_low=4000):
    """Fake per-halo arrays (k_read.derive format) drawn from the published laws, for the local smoke."""
    import numpy as np
    import k_common as KC
    import k_fit_core as K
    lm = []
    for b in range(len(KC.MASS_EDGES) - 1):
        n = max(3, int(n_low * 10 ** (-0.55 * b) * (1 + z) ** -1.5))
        if KC.MASS_EDGES[b] > 15.3 - 0.4 * z:
            break
        lm.append(rng.uniform(KC.MASS_EDGES[b], KC.MASS_EDGES[b + 1], n))
    lm = np.concatenate(lm) * np.log(10)
    n = len(lm)
    R = 1.6 * np.exp((lm - 14 * np.log(10)) / 3) / (1 + z)
    frac = np.array([0.30, 0.47, 0.65, 1.0, 1.35, 1.34, 1.66, 1.8, 3.3])
    x = np.empty((n, KC.N_AP))
    x[:, :KC.N_SO] = frac * np.exp(0.06 * rng.standard_normal((n, 1)) + 0.02 * rng.standard_normal((n, KC.N_SO)))
    x[:, 3] = 1.0
    x[:, KC.N_SO:] = np.array(KC.KPC) / 1e3 / R[:, None]
    out = {"soap_row": np.arange(n), "lnM": lm, "M500c": np.exp(lm) * 0.7, "conc": np.full(n, 5.0),
           "Nbound": np.full(n, 1000), "R200c": R, "x": x, "valid": np.ones((n, KC.N_AP), bool)}
    for q, form, field in (("Yn", "B12", "P"), ("Mg", "B16", "G")):
        law = K.Law(form, "orig")
        p = law.phi(law.start(), lm, np.full(n, np.log1p(z)))
        amp = p["amp"] * np.exp(0.2 * rng.standard_normal(n) - 0.02)
        out[q] = K.FIELD_NORM[field] * K.cumulative(x, form, amp[:, None], p["xc"][:, None], p["alpha"][:, None],
                                                     p["beta"][:, None], p["gamma"][:, None]) * np.exp(0.05 * rng.standard_normal((n, KC.N_AP)))
    out["Yn"][:, KC.N_SO:] = np.nan
    out["Ya"] = out["Yn"] * 1.003
    out["Ngas"] = (np.exp(lm)[:, None] / 1e10 * out["Mg"] * 0.158 * 0.9).astype(np.int64)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="full run (cluster_tests.py always passes it)")
    ap.add_argument("--smoke", action="store_true", help="smoke run; takes precedence over --full")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--snaps", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--cleanup-scratch", action="store_true")
    ap.add_argument("--results", default=None)
    ap.add_argument("--keep-halo", type=int, default=1, help="write per-halo arrays to the scratch (1) or not (0)")
    ap.add_argument("--resume", action="store_true", help="k3: re-use nominal fits and bootstrap chunks in results/work/")
    ap.add_argument("--boot", default=None, help="k3 bootstrap refits 'orig,other,check' (default 100,40,20)")
    ap.add_argument("--starts", default=None, help="k3: JSON file of extra start vectors per configuration (k_fits.extra_starts)")
    a = ap.parse_args()
    if not (a.full or a.smoke):
        ap.error("give --full or --smoke")
    os.environ["K_MODE"] = "smoke" if a.smoke else "full"
    if a.workers:
        os.environ["K_WORKERS"] = str(a.workers)
    if a.results:
        os.environ["K_RESULTS"] = str(Path(a.results).resolve())
    if a.scratch:
        os.environ["K_SCRATCH"] = str(Path(a.scratch).resolve())
    if a.resume:
        os.environ["K_RESUME"] = "1"
    if a.boot:
        os.environ["K_BOOT"] = a.boot
    if a.starts:
        sp = Path(a.starts)
        os.environ["K_STARTS"] = str(sp if sp.is_absolute() else (Path.cwd() / sp).resolve())
    sys.path.insert(0, str(HERE))
    import numpy as np
    import k_common as KC
    import k_read as KR
    import k_fits as KF
    KC.RESULTS.mkdir(parents=True, exist_ok=True)
    stages = [s for s in a.stages.split(",") if s]
    bad = set(stages) - set(STAGES)
    if bad:
        ap.error(f"unknown stages {bad}")
    snaps = tuple(int(s) for s in a.snaps.split(",")) if a.snaps else (KC.SMOKE_SNAPS if a.smoke else KC.SNAPS)
    limit = a.limit if a.limit is not None else (3_000_000 if a.smoke and not a.synthetic else None)
    man_path = KC.RESULTS / "run_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"invocations": []}
    inv = {"argv": sys.argv, "host": platform.node(), "python": sys.version, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "mode": os.environ["K_MODE"], "workers": KC.WORKERS, "stages": stages, "snaps": list(snaps), "limit": limit,
           "scratch": str(KC.SCRATCH), "results": str(KC.RESULTS), "stage_seconds": {},
           "starts_file": os.environ.get("K_STARTS"), "boot": os.environ.get("K_BOOT"),
           "starts_sha256": (sha256(os.environ["K_STARTS"]) if os.environ.get("K_STARTS") and Path(os.environ["K_STARTS"]).exists() else None),
           "ranges": {r: {"so": list(v["so"]), "sphere_xmax": v["xmax"]} for r, v in KC.RANGES.items()}, "code_sha256": {
               p.name: sha256(p) for p in sorted(HERE.glob("k_*.py")) + [HERE / "run_test_k.py"]}}
    code = 0
    try:
        for st in stages:
            t0 = time.time()
            KC.log(f"stage {st} start")
            if st == "k0":
                if a.synthetic:
                    inv["k0"] = "skipped (synthetic)"
                else:
                    inv0 = KR.k0_inventory(snaps)
                    inv0["smoke_read_0077"] = KR.k0_smoke_read(77, limit)
                    KC.write_json(KC.RESULTS / "k0_inventory.json", inv0)
            elif st == "k1":
                order = sorted(snaps, reverse=True)
                if a.synthetic:
                    rng = np.random.default_rng(KC.SEED)
                    status = []
                    for s, z in zip(order, (0.0, 2.0, 1.0, 0.5)):
                        h = synthetic_halos(s, z, rng)
                        arr, summary = KR.reduce_cells(h, s, z, KC.SEED)
                        (KC.RESULTS / "cells").mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(KC.RESULTS / "cells" / f"cells_{s:04d}.npz", **arr)
                        KC.write_json(KC.RESULTS / "cells" / f"cells_{s:04d}.json", {"snap": s, "Redshift": z, "synthetic": True, "cells": summary})
                        status.append({"snap": s, "ok": True, "n_selected": int(len(h["lnM"]))})
                else:
                    from multiprocessing import get_context
                    from functools import partial
                    fn = partial(KR.k1_file, results=str(KC.RESULTS), scratch=str(KC.SCRATCH) if a.keep_halo else None,
                                 limit=limit, keep_halo=bool(a.keep_halo))
                    if KC.WORKERS > 1:
                        with get_context("fork").Pool(KC.WORKERS, maxtasksperchild=1) as pool:
                            status = []
                            for res in pool.imap_unordered(fn, order):
                                status.append(res)
                                KC.log(f"K1 {res['snap']:04d} ok={res['ok']} n={res.get('n_selected')} {res['seconds']:.0f} s")
                    else:
                        status = []
                        for s in order:
                            status.append(fn(s))
                            KC.log(f"K1 {s:04d} ok={status[-1]['ok']} {status[-1]['seconds']:.0f} s")
                status = sorted(status, key=lambda d: d["snap"])
                KC.write_json(KC.RESULTS / "k1_status.json", status)
                if not all(s["ok"] for s in status):
                    code = 3
            elif st == "k2":
                KF._CACHE.clear()
                for f in ("P", "G"):
                    for r in KC.RANGES:
                        KF.dataset(f, r)
                KF.stage_k2(KC.WORKERS)
            elif st == "k3":
                KF._CACHE.clear()
                for f in ("P", "G", "Pagn", "Gcore"):
                    for r in KC.RANGES:
                        KF.dataset(f, r)
                KF.stage_k3(KC.WORKERS)
            inv["stage_seconds"][st] = time.time() - t0
            inv.setdefault("peak_rss_GiB_after", {})[st] = peak_rss_gib()
            KC.log(f"stage {st} done in {inv['stage_seconds'][st]:.0f} s")
    except Exception:   # noqa: BLE001
        inv["traceback"] = traceback.format_exc()[-6000:]
        print(inv["traceback"], flush=True)
        code = 1
    if a.cleanup_scratch and code == 0 and KC.SCRATCH.exists() and "nagainis_halfdome_testK" in str(KC.SCRATCH):
        size = sum(p.stat().st_size for p in KC.SCRATCH.rglob("*") if p.is_file())
        shutil.rmtree(KC.SCRATCH)
        inv["scratch_removed"] = {"path": str(KC.SCRATCH), "bytes": size, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    inv["exit_code"] = code
    inv["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest["invocations"].append(inv)
    KC.write_json(man_path, manifest)
    sums = []
    for p in sorted(KC.RESULTS.rglob("*")):
        if p.is_file() and "work" not in p.relative_to(KC.RESULTS).parts and p.name != "SHA256SUMS" and not p.name.startswith("test_K"):
            sums.append(f"{sha256(p)}  {p.relative_to(KC.RESULTS)}")
    (KC.RESULTS / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
