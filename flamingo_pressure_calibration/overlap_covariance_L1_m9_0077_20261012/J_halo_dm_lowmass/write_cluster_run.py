"""Write CLUSTER_RUN.json, the deployment manifest of Test J's --full run on STRW maas.

Input list = (files opened outside J_halo_dm_lowmass by the --smoke run, from results_smoke/run_manifest.json, with
smoke-only Test F products swapped for the full ones) + (the full-mode inputs the smoke run does not read: Test F's full
profiles, Test H's truth for the parity gates, the J0 export) + (Test F's deployment list and the engine dependencies,
which Test J runs through F's code) + (Test G's modules imported read-only).
Local helper: python -B write_cluster_run.py [--maas-run JSON]   (--maas-run appends a run record)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FULL_ONLY = [ROOT / "F_observables" / "results" / "profiles.npz",
             ROOT / "H_sensitivity" / "results" / "truth.npz", ROOT / "H_sensitivity" / "results" / "truth_summary.json",
             HERE / "inputs" / "j0_profiles_x200.npz", HERE / "inputs" / "j0_meta.json", ROOT / "metadata_plan.json"]
CODE = ["j_common.py", "j_prepare.py", "j_basis.py", "j_fallback.py", "j_obs.py", "j_pdf.py", "j_truth.py", "j_sens.py",
        "run_test_j.py"]
G_CODE = [ROOT / "G_physical_outer" / n for n in ("g_common.py", "g_pdf.py", "g_physics.py")]
SCRATCH = "/tmp/nagainis_halfdome_testJ"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def audited(manifest_path):
    m = json.loads(Path(manifest_path).read_text())
    out = set()
    for inv in m.get("invocations", []):
        for item in inv.get("inputs_read", []):
            p = Path(item["path"])
            if "__pycache__" in p.parts:
                p = p.parent.parent / (p.name.split(".")[0] + ".py")
            if "results_smoke" in p.parts and p.parent.parent.name == "F_observables":
                p = ROOT / "F_observables" / "results" / p.name
            out.add(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maas-run", default=None, help="JSON string of a run record to append")
    ap.add_argument("--update-run", default=None, help="JSON string: fields merged into the last run record")
    ap.add_argument("--mirror-attempt", default=None, help="JSON string of a mirror (upload) record to append")
    a = ap.parse_args()
    smoke = audited(HERE / "results_smoke" / "run_manifest.json")
    fspec = json.loads((ROOT / "F_observables" / "CLUSTER_RUN.json").read_text())
    via_f = {Path(r["path"]) for r in fspec["files_read_outside_F_observables"]}
    via_f |= {ROOT / "F_observables" / c.split("/")[-1] for c in fspec["code_to_deploy"]}
    via_f |= {Path(d) for d in json.loads((ROOT / "kernels_cluster_deps.json").read_text())["engine_dependency_files"]}
    files = sorted({p for p in smoke | set(FULL_ONLY) | via_f | set(G_CODE)
                    if p.is_file() and not p.is_relative_to(HERE / "results_smoke") and "results_smoke" not in p.parts})
    rows = [{"path": str(p), "relative_to_STUDY_ROOT": os.path.relpath(p, ROOT), "bytes": p.stat().st_size,
             "sha256": sha(p), "read_in_smoke": p in smoke, "full_only": p in FULL_ONLY,
             "via_test_F_or_engine_deps": p in via_f and p not in smoke} for p in files]
    old = json.loads((HERE / "CLUSTER_RUN.json").read_text()) if (HERE / "CLUSTER_RUN.json").exists() else {}
    runs = old.get("maas_runs", [])
    if a.maas_run:
        runs.append(json.loads(a.maas_run))
    if a.update_run and runs:
        runs[-1].update(json.loads(a.update_run))
    mirrors = old.get("mirror_attempts", [])
    if a.mirror_attempt:
        mirrors.append(json.loads(a.mirror_attempt))
    out = {
        "purpose": "Test J1 (halo component of FRB DM under three definitions + low-mass calibration 12.7-13.5 + exact "
                   "bound-only kSZ): full run on STRW maas; compute stages only (J0 export, figures, report tables and "
                   "verification are made locally)",
        "command": f"cd $STUDY_ROOT && STUDY_ROOT=$STUDY_ROOT nice -n 19 python -B J_halo_dm_lowmass/run_test_j.py --full "
                   f"--workers 8 --scratch {SCRATCH} [--stages ...]   (through cluster_tests.py run J --workers 8 "
                   f"--rss-stop 32 --driver-args=\"--scratch {SCRATCH}\", which adds nice 19, ionice 7 and the "
                   f"load/memory guards)",
        "scratch": f"{SCRATCH} (maas node-local /tmp, 20 GB): the two 14-bin fine bases (~1.0 GB each), their C10 "
                   f"f_pdf files and the fallback pieces (~0.1 GB); results/work/g/f_base/work is a symlink to it. It is "
                   f"ours and is deleted after the final fetch of Test J (record in maas_runs)",
        "maas_runs": runs,
        "mirror_attempts": mirrors,
        "STUDY_ROOT_on_maas": "/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/"
                              "flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012",
        "smoke_command": "cd $STUDY_ROOT/J_halo_dm_lowmass && python -B run_test_j.py --smoke   (local, 1 worker, ~3 min)",
        "local_preparation": "python -B J_halo_dm_lowmass/j_inputs.py   (J0: export from stacking_choices_20261014/cache and "
                             "the particle-pass owned counts into J_halo_dm_lowmass/inputs/, mirrored to maas)",
        "stages": "prepare, basis, fallback, truth, sens (default all, in order); --stages a,b,... reruns some",
        "exit_status": "0 = completed; 1 = a stage raised (traceback in results/run_manifest.json); 2 = completed but a "
                       "parity gate failed (results/parity.json, results/fallback_parity.json)",
        "code_to_deploy": [f"J_halo_dm_lowmass/{c}" for c in CODE],
        "reads_test_F_and_G_modules": "F_observables/f_common.py, f_basis.py, f_kernels.py, f_pdf.py, f1_importance.py "
                                      "(+ f_fit, f_methods); G_physical_outer/g_common.py (constants), g_pdf.py (fast "
                                      "deposit) - imported read-only, configured through environment variables "
                                      "(F_RESULTS / G_RESULTS under J_halo_dm_lowmass/results/work/g); the fine grid and "
                                      "the calibrated support are set on the imported f_common module object in the J "
                                      "process only; f_kernels.basis_grid is wrapped during DM-PDF passes to cut the "
                                      "fallback haloes at 1/2/4 R200c",
        "writes": "J_halo_dm_lowmass/results/ (compact npz/csv/json; per-halo profiles in results/work/, not copied back) "
                  f"and {SCRATCH}",
        "inputs": rows,
        "inputs_total_MB": round(sum(r["bytes"] for r in rows) / 1e6, 1),
        "expected_resources_workers_8": {
            "basis": "2 sets x 14 bins (Test H: 3 sets x 10 bins, 371 s, parent 12.5 GiB, child 2.1 GiB); running sums "
                     "keep the parent at ~2.5 GiB",
            "dm_pdf": "about 0.035 s per per-halo entry (8 workers; Test H 15,130 entries in 641 s) plus ~100 s per pass: "
                      "truth ~1.5 h (incl. 4 Mecke passes), sens ~0.5 h",
            "wall_total": "about 2.5-3 h at nice 19 with 8 workers",
            "rss": "tree RSS at the DM-PDF forks ~ 9 x parent (<= 0.5 GiB) + 8 x ~1 GiB; basis build the largest; "
                   "rss-stop 32 GiB (Test H attempt 2's basis stage passed under 32 GiB)"},
        "copy_back": "J_halo_dm_lowmass/results/ without work/ (cluster_tests.py fetch J)",
    }
    (HERE / "CLUSTER_RUN.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(rows), "inputs,", out["inputs_total_MB"], "MB")


if __name__ == "__main__":
    main()
