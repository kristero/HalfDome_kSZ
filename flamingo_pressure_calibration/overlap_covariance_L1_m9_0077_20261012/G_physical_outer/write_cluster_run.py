"""Write CLUSTER_RUN.json, the deployment manifest of Test G's --full run on STRW maas.

Input list = (files opened outside G_physical_outer by the --smoke run, from results_smoke/run_manifest.json, with
byte-code caches replaced by their sources) + (the full-mode inputs that the smoke run does not read: Test F's maas
products, Test B's per-halo arrays, Test G's xi/bias table) + (files opened by any local full-mode dev run given with
--dev-manifest).  Local helper: python -B write_cluster_run.py [--dev-manifest PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = Path("/home/cbllover/HalfDome")
FULL_ONLY = [ROOT / "F_observables" / "results" / n for n in
             ("profiles.npz", "truth.npz", "truth_summary.json", "importance_F.npz", "models_main.npz")] + \
            [ROOT / "B_covariance" / "results" / "profiles_fit.npz", ROOT / "B_covariance" / "results" / "halo_table.npz",
             HERE / "inputs" / "xi_bias_z0.npz", ROOT / "metadata_plan.json"]
CODE = ["g_common.py", "g_physics.py", "g_models.py", "g_fit.py", "g_prepare.py", "g_parity.py", "g_pdf.py", "g_fits.py",
        "g_evaluate.py", "g_selection.py", "g_cv.py", "run_test_g.py"]
PARTICLE_PRODUCTS = {"measurements_owner.npz", "measurements_excision.npz", "measurements_partition.npz",
                     "measurements_legacy.npz", "apertures.npz", "selected_halos.npz", "overlap_tables.npz",
                     "census_pairs.npz", "census_population.npz", "soap_extra_columns.npz"}


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
            out.add(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-manifest", default=None)
    a = ap.parse_args()
    smoke = audited(HERE / "results_smoke" / "run_manifest.json")
    dev = audited(a.dev_manifest) if a.dev_manifest else set()
    # Test G runs Test F's code paths: everything Test F's deployment manifest lists, and the engine's dependencies
    fspec = json.loads((ROOT / "F_observables" / "CLUSTER_RUN.json").read_text())
    via_f = {Path(r["path"]) for r in fspec["files_read_outside_F_observables"]}
    via_f |= {ROOT / "F_observables" / c.split("/")[-1] for c in fspec["code_to_deploy"]}
    deps = json.loads((ROOT / "kernels_cluster_deps.json").read_text())["engine_dependency_files"]
    via_f |= {Path(d) for d in deps}
    files = sorted({p for p in smoke | dev | set(FULL_ONLY) | via_f
                    if p.is_file() and not p.is_relative_to(HERE / "results_smoke")})
    rows = []
    for p in files:
        rows.append({"path": str(p), "relative_to_STUDY_ROOT": os.path.relpath(p, ROOT),
                     "bytes": p.stat().st_size, "sha256": sha(p), "read_in_smoke": p in smoke,
                     "listed_via_test_F_or_engine_deps": p in via_f and p not in smoke and p not in dev,
                     "particle_product_symlinked_on_maas": p.name in PARTICLE_PRODUCTS and p.parent == ROOT})
    out = {
        "purpose": "Test G (physical outer components, per-field calibration, Lee22-style parametrisation ladder): full run "
                   "on STRW maas; compute stages only (figures and report tables are made locally)",
        "command": "cd $STUDY_ROOT && STUDY_ROOT=$STUDY_ROOT nice -n 19 python -B G_physical_outer/run_test_g.py --full "
                   "--workers 8   (through cluster_tests.py run G --workers 8 --rss-stop 32, which adds nice 19, ionice "
                   "and the load/memory guards)",
        "maas_runs": [
            {"attempt": 1, "workers": 4, "rss_stop_GiB": 24, "launched": "2026-10-13 14:11 EEST", "stages": "all",
             "outcome": "stopped on request (user: use 8 workers) at 14:35 EEST during evaluate (PDF pass p2 64/274); "
                        "inputs, kernels (504 s), prepare, parity (exact) and fits (162 s) completed and were kept in "
                        "results/, with the evaluate p1 PDF checkpoint (work/pdf_eval_p1.npz)",
             "archived": "G_physical_outer/attempts/attempt1_4workers_20261013/ (receipt, log, failed.json, run_manifest)"},
            {"attempt": 2, "workers": 8, "rss_stop_GiB": 32, "launched": "2026-10-13 14:36 EEST",
             "stages": "evaluate,selection,cv (python -B cluster_tests.py run G --workers 8 --rss-stop 32 "
                       "--driver-args=\"--stages evaluate,selection,cv\")",
             "load_at_launch": "25.4 / 27.1 (1/5 min) on os.cpu_count() = 48; 178.6 GiB available",
             "note": "evaluate reuses the p1 checkpoint (same specification digest); results/run_manifest.json gets a "
                     "second invocation record"}],
        "STUDY_ROOT_on_maas": "/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/"
                              "flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012",
        "smoke_command": "cd $STUDY_ROOT/G_physical_outer && python -B run_test_g.py --smoke   (local, 1 worker)",
        "stages": "inputs, kernels, prepare, parity, fits, evaluate, selection, cv (default all, in order); --stages "
                  "a,b,... reruns some; each stage reads the earlier stages' outputs from results/",
        "exit_status": "0 = completed; 1 = a stage raised (traceback in results/run_manifest.json, later stages still "
                       "attempted); 2 = completed but the parity gate failed (results/parity.json)",
        "code_to_deploy": [f"G_physical_outer/{c}" for c in CODE],
        "reads_test_F_modules": "F_observables/f_common.py, f_fit.py, f_methods.py, f_basis.py, f_kernels.py, f_pdf.py, "
                                "f2_run.py (+ f0_profiles.py, f1_truth.py, f1_importance.py in --smoke) - imported "
                                "read-only; their outputs go to G_physical_outer/results/f_base",
        "writes": "G_physical_outer/results/ only (compact npz/csv/json; big intermediates in results/work and "
                  "results/f_base/work - the rebuilt Test F basis, ~0.1 GB, not copied back)",
        "layout_requirement": "as Test F (CLUSTER_RUN.json there): STUDY_ROOT/{analysis,kernels,F_observables,"
                              "B_covariance,G_physical_outer}, the earlier studies one level up, frb_map_generation two "
                              "levels up (engine inputs)",
        "inputs": rows,
        "inputs_total_MB": round(sum(r["bytes"] for r in rows) / 1e6, 1),
        "expected_resources_workers_4": {
            "wall_minutes_by_stage_local_estimate_4_workers": {"inputs": 0.1, "kernels": 5, "prepare": 0.3, "parity": 1,
                                                               "fits": 1.5, "evaluate": 15, "selection": 7, "cv": 3},
            "maas_attempt1_4_workers_minutes": {"kernels": 8.4, "prepare": 0.03, "parity": 0.5, "fits": 2.7,
                                                "evaluate_p1_pdf": 5.1},
            "wall_total": "about 50 min locally; allow 1.5-2 h at nice 19 on maas",
            "peak_memory_GB": "attempt 1 (4 workers): peak tree RSS 4.67 GiB; 8 workers: about 3 + 8 x 1.5 GB at most "
                              "(rss-stop 32 GiB); keep >= 10 GB available"},
        "copy_back": "G_physical_outer/results/ without work/ and f_base/work/ (cluster_tests.py fetch G)",
    }
    (HERE / "CLUSTER_RUN.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(rows), "inputs,", out["inputs_total_MB"], "MB")


if __name__ == "__main__":
    main()
