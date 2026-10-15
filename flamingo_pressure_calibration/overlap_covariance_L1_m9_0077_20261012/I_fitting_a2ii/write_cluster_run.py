"""Write CLUSTER_RUN.json, the deployment manifest of Test I's --full run on STRW maas (cluster_tests.py mirrors every
absolute local path named here, plus the .py / .json files of the test directory).
Local helper: python -B write_cluster_run.py [--maas-run JSON]   (--maas-run appends a run record)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ["i_common.py", "i_g35.py", "i_fine.py", "i_supp.py", "i_perhalo.py", "run_test_i.py"]
LOCAL_INPUTS = [HERE / "inputs" / "i0_variants.npz", HERE / "inputs" / "i0_variants.json",
                HERE / "inputs" / "i0_perhalo_x200.npz", HERE / "inputs" / "i0_perhalo_x200.json"]
READ_ON_MAAS = {
    "Test G 35-shell per-halo profiles (A2-ii and all gas)": "G_physical_outer/results/f_base/profiles.npz",
    "Test G prepare products (ebar, kappa_P)": "G_physical_outer/results/prepare.npz",
    "Test B halo table (SOAP concentration_200c for the c-split cells)": "B_covariance/results/halo_table.npz",
    "Test G linear bias / xi table": "G_physical_outer/inputs/xi_bias_z0.npz",
    "Test H fine per-halo profiles": "H_sensitivity/results/work/g/f_base/profiles.npz",
    "Test H fine A2-ii basis (run 1: Test H's scratch, hard-linked into Test I's scratch; run 2: rebuilt by i_supp "
    "suppbasis with Test H's code into Test I's scratch)": "/tmp/nagainis_halfdome_testH/f_base_work/basis_HA.npz (run 1); "
    "/tmp/nagainis_halfdome_testI/f_base_work/basis_HA.npz (runs 2 and 3; run 3 re-uses run 2's rebuild)",
    "Test H truth": "H_sensitivity/results/truth.npz, H_sensitivity/results/truth_summary.json",
    "Test H A2-ii importance maps": "H_sensitivity/results/sens.npz",
    "Test H DM-PDF fallback slot": "H_sensitivity/results/work/pdf_fallback_h.npz",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maas-run", default=None)
    a = ap.parse_args()
    path = HERE / "CLUSTER_RUN.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    spec = {
        "purpose": "Test I (fitting the A2-ii painting stacks: B12/B16 refits, the Lee22 parametrisation ladder, "
                   "stacking-choice propagation), evaluated on Test H's fine A2-ii basis: full run on STRW maas; "
                   "figures, I4 tables and the report section are made locally",
        "command": "run 1: cd $STUDY_ROOT && nice -n 19 python -B I_fitting_a2ii/run_test_i.py --full --workers 8 "
                   "--scratch /tmp/nagainis_halfdome_testI --cleanup-scratch   (through cluster_tests.py run I "
                   "--workers 8 --rss-stop 16 --driver-args=\"--scratch /tmp/nagainis_halfdome_testI --cleanup-scratch\"); "
                   "run 2 (supplement): cluster_tests.py run I --workers 8 --rss-stop 32 --driver-args=\"--stages "
                   "suppbasis,supp --scratch /tmp/nagainis_halfdome_testI\" (no cleanup); run 3 (per-halo "
                   "resolution rule on its own shells): cluster_tests.py run I --workers 8 --rss-stop 16 "
                   "--driver-args=\"--stages phfits,pheval --scratch /tmp/nagainis_halfdome_testI\" (no cleanup)",
        "scratch": "run 1: /tmp/nagainis_halfdome_testI held a hard link of Test H's basis_HA.npz; run 1's driver "
                   "(attempts/run1_20261014/run_test_i.py) deleted /tmp/nagainis_halfdome_testH and "
                   "/tmp/nagainis_halfdome_testI at its end (--cleanup-scratch; recorded in results/run_manifest.json "
                   "scratch_removed) - Test H's scratch should not have been deleted.  The current driver deletes only "
                   "Test I's own scratch and only with --cleanup-scratch.  run 2: i_supp suppbasis rebuilt basis_HA.npz "
                   "into /tmp/nagainis_halfdome_testI/f_base_work (left in place for re-use; delete when the study is "
                   "closed)",
        "STUDY_ROOT_on_maas": "/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012",
        "smoke_command": "cd $STUDY_ROOT/I_fitting_a2ii && I_PDF_LIMIT=16 python -B run_test_i.py --smoke",
        "stages": "fits35, finefits, ladder, lvariants, evaluate (default all, in order; each in its own process); "
                  "supplement stages suppbasis, supp (i_supp.py; run 2) and phfits, pheval (i_perhalo.py; run 3), "
                  "only when named; inputs/i0_perhalo_x200.npz is made locally by i_perhalo_inputs.py",
        "exit_status": "0 = completed; 1 = a stage failed (log tail in results/run_manifest.json)",
        "code_to_deploy": [f"I_fitting_a2ii/{n}" for n in CODE] + ["I_fitting_a2ii/write_cluster_run.py"],
        "reads_modules_read_only": "F_observables/f_*.py, G_physical_outer/g_*.py (g_common, g_selection, g_physics, g_models, "
                                   "g_pdf), H_sensitivity/h_*.py (h_common, h_fits, h_pdf, h_params, h_truth, h_basis), "
                                   "analysis/*, kernels/engine.py - configured through environment variables; nothing is "
                                   "written outside I_fitting_a2ii/results and the scratch directory",
        "inputs_local_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in LOCAL_INPUTS],
        "inputs_read_on_maas": READ_ON_MAAS,
        "test_H_basis_HA_sha256": "3a7c7f4cae9b483aa949fc1ecd263b390835127f9926121edb9057f83e7a9f1b",
        "maas_runs": old.get("maas_runs", []),
    }
    if a.maas_run:
        spec["maas_runs"].append(json.loads(a.maas_run))
    path.write_text(json.dumps(spec, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
