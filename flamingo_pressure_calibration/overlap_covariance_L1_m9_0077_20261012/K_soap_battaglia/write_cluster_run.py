"""Write CLUSTER_RUN.json, the deployment manifest of Test K's run on STRW maas (cluster_tests.py mirrors every
absolute local path named here, plus the .py / .json files of the test directory).
Local helper: python -B write_cluster_run.py [--maas-run JSON]   (--maas-run appends a run record)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = ["k_common.py", "k_fit_core.py", "k_read.py", "k_fits.py", "k_selftest.py", "run_test_k.py", "k_starts_run2.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maas-run", default=None)
    a = ap.parse_args()
    path = HERE / "CLUSTER_RUN.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    spec = {
        "purpose": "Test K (Battaglia B12 pressure / B16 gas-density laws refitted to FLAMINGO L1_m9 HYDRO_FIDUCIAL "
                   "SOAP-HBT aperture integrals at all 71 outputs, forward-modelled at every halo's own radii): reads "
                   "(K0, K1) and fits (K2, K3) on STRW maas; comparisons, parity checks, figures and the report are "
                   "made locally from the fetched results",
        "command": "cd $STUDY_ROOT && nice -n 19 python -B K_soap_battaglia/run_test_k.py --full --workers 8 "
                   "--scratch /tmp/nagainis_halfdome_testK [--stages k0,k1,k2,k3]   (through cluster_tests.py run K "
                   "--workers 8 --rss-stop 24 --driver-args=\"--scratch /tmp/nagainis_halfdome_testK\", which adds "
                   "nice 19, ionice 7 and the load/memory guards)",
        "smoke_command": "maas: cluster_tests.py run K --workers 2 --driver-args=\"--smoke --scratch "
                         "/tmp/nagainis_halfdome_testK_smoke --results K_soap_battaglia/results_smoke\" (outputs 0077 "
                         "and 0037, first 3,000,000 rows); local without data: python -B run_test_k.py --smoke "
                         "--synthetic --workers 2 --results results_smoke_syn",
        "scratch": "/tmp/nagainis_halfdome_testK (maas node-local /tmp): per-halo arrays halo_<snap>.npz (float32) of "
                   "the selected centrals, for re-reduction without re-reading; ours; deleted after the final fetch "
                   "(recorded in maas_runs)",
        "STUDY_ROOT_on_maas": "/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012",
        "stages": "k0 inventory, k1 reads + per-cell reduction (one worker per output), k2 per-cell fits, k3 global fits "
                  "+ bootstrap (default all, in order)",
        "run3_commands": ["cluster_tests.py run K --workers 3 --driver-args=\"--stages k1 --snaps 77,37,8 --results "
                          "K_soap_battaglia/results_k1check --keep-halo 0\"  (K1 provenance check with the final code; "
                          "compared with results/cells by k1_check_compare.py)",
                          "cluster_tests.py run K --workers 8 --rss-stop 24 --driver-args=\"--stages k2,k3 --boot 100,20,10 "
                          "--starts K_soap_battaglia/k_starts_run2.json --scratch /tmp/nagainis_halfdome_testK/full\""],
        "exit_status": "0 = completed; 1 = a stage raised (traceback in results/run_manifest.json); 3 = an output failed in k1",
        "code_to_deploy": [f"K_soap_battaglia/{c}" for c in CODE],
        "code_sha256": {c: sha(HERE / c) for c in CODE},
        "reads_on_maas_read_only": {
            "SOAP-HBT": "/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_00{07..77}.hdf5 "
                        "(columns of k_common.COLUMNS over the row span of the selected centrals)",
            "study haloes (z = 0 parity rows)": "selected_halos.npz (study product symlink, soap_row only)",
        },
        "writes": "K_soap_battaglia/results/ (cells/cells_<snap>.npz|json, k0_inventory.json, k1_status.json, "
                  "k2_cell_fits.json, k3_fits.json, k3_boot.npz, k3_predictions.npz, k_params.json, z0_study_rows.npz, "
                  "run_manifest.json, SHA256SUMS; work/ = per-fit json, not fetched) and the scratch directory",
        "budget": "reduced results < 300 MB (expected ~100 MB); remote home quota 3.9 GB shared with Test J: per-halo "
                  "arrays only in /tmp",
        "maas_runs": old.get("maas_runs", []),
    }
    if a.maas_run:
        spec["maas_runs"].append(json.loads(Path(a.maas_run).read_text()))
    path.write_text(json.dumps(spec, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
