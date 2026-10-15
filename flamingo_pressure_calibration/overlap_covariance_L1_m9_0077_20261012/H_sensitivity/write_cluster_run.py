"""Write CLUSTER_RUN.json, the deployment manifest of Test H's --full run on STRW maas.

Input list = (files opened outside H_sensitivity by the --smoke run, from results_smoke/run_manifest.json, with
byte-code caches replaced by their sources, smoke-only Test F products swapped for the full ones) + (the full-mode
inputs the smoke run does not read: Test F's maas products) + (Test F's deployment list and the engine dependencies,
which Test H runs through F's code) + (Test G's modules imported read-only) + Test H's own inputs (H0 export).
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
FULL_ONLY = [ROOT / "F_observables" / "results" / n for n in
             ("profiles.npz", "truth.npz", "truth_summary.json", "importance_F.npz")] + \
            [HERE / "inputs" / "h0_profiles_x200.npz", HERE / "inputs" / "h0_meta.json", ROOT / "metadata_plan.json"]
CODE = ["h_common.py", "h_prepare.py", "h_basis.py", "h_pdf.py", "h_truth.py", "h_sens.py", "h_fits.py", "h_params.py",
        "h_beyond.py", "h_binning.py", "run_test_h.py"]
G_CODE = [ROOT / "G_physical_outer" / n for n in ("g_common.py", "g_pdf.py", "g_physics.py")]


MEMORY_FIX = {
    "diagnosis": "attempt 2 was stopped by the wrapper at 'own RSS 32.4 GiB above 32.0 GiB' in 'truth', right after the "
                 "moment observables, i.e. when f_pdf forked its 8 workers for the first DM-PDF pass. The guard sums the "
                 "RSS of every process in the tree; a forked worker's RSS includes all of the parent's resident pages it "
                 "shares copy-on-write, so the sum is ~ (1 + 8) x parent RSS + the workers' private memory. The parent "
                 "then held the three fine bases (3 x 0.737 GB: W_y, W_tau [421 z, 10 bins, 75 shells, 106 ell] and the "
                 "K tensors), the importance Jacobians, and heap left over from the basis build in the same process "
                 "(basis parent peak 12.5 GiB): ~3.5-3.6 GiB x 9 = 32.4 GiB. The real (PSS) footprint was ~5-6 GB.",
    "fix": ["truth: one basis in memory at a time (FP: parity; HA, HM: production truths and the cross-basis checks), "
            "each deleted before the next; none at the DM-PDF / Mecke forks (f_pdf reads only g_E, gbar_E lazily from "
            "the basis file)",
            "sens / beyond / binning: the basis, Jacobians and 2-halo prefix sums are deleted before the DM-PDF forks",
            "params: fits -> DM-PDF / Mecke passes -> bootstrap pool, all before the basis and the H3 maps are loaded; "
            "only the imp__a2ii__* maps of sens.npz are read",
            "h_common.release(): gc + glibc malloc_trim(0) before every fork and between stages; h_common.mem() logs "
            "the parent RSS / PSS at every fork point (results/run_manifest.json: memory_checkpoints per stage)",
            "attempt 3 reruns prepare,truth,sens,params,beyond,binning in a fresh process and reuses the attempt-2 "
            "bases in /tmp/nagainis_halfdome_testH (sha256 / zip CRC / finiteness checked on maas; the basis code is "
            "unchanged), so no basis-build heap is carried into 'truth'"],
    "evidence": {"local_smoke_profile_4_workers": "H_sensitivity/attempts/attempt2_rss_20261014/local_memory_profile_smoke_4workers.json "
                 "(tree RSS <= 2.1 GiB, tree PSS <= 1.4 GiB, parent <= 0.28 GiB; parity passed)",
                 "local_full_parent_baseline": "H_sensitivity/attempts/attempt2_rss_20261014/local_full_parent_baseline.json "
                 "(full-mode parent with every truth-stage input, sky, grid, column setup: 0.20 GiB RSS)",
                 "extrapolation": "full-mode tree at a fork ~ 9 x 0.2 GiB + 8 x private worker memory (G's DM-PDF "
                 "workers peaked at 0.94 GiB on the 35-shell grid; the fine-grid column arrays add < 0.2 GB) -> "
                 "<= ~10 GiB, well below the 24 GiB target; float32 and memmapped bases were not needed"},
    "rss_stop_GiB": 24, "workers": 8, "nice": 19}
REUSED_BASES = {
    "directory": "/tmp/nagainis_halfdome_testH/f_base_work (maas node-local, ours)",
    "built_by": "attempt 2 (basis stage completed, 371 s, 8 workers)",
    "sha256": {"basis_FP.npz": "0f5409d4339e792a507be44ab8952914fedb30d6dbf44bc380df49d0c087681d",
               "basis_HA.npz": "3a7c7f4cae9b483aa949fc1ecd263b390835127f9926121edb9057f83e7a9f1b",
               "basis_HM.npz": "636e98b3ebd45616783596270461a69eefdf63461d883b939b5492582e1f7582"},
    "bytes_each": 737228069, "checks": "zip CRC ok, 59 arrays, W_y [421, 10, 75, 106], K [10, 75, 75, 106], all finite"}
FETCH_CORRECTED = ("2026-10-14 16:29 EEST, cluster_tests.py fetch H: 23 files, 75.49 MiB, results/SHA256SUMS verified "
                   "locally. Remote and local copies of every compute module (h_basis, h_beyond, h_binning, h_common, "
                   "h_fits, h_inputs, h_params, h_pdf, h_prepare, h_sens, h_truth, run_test_h.py) identical (sha256). "
                   "Local-only post-processing files differ from the mirrored copies or exist only locally: "
                   "make_figures.py, report_numbers.py, write_cluster_run.py, verify_h.py (edited after the fetch), "
                   "report_digest.py and h_post.py (local only). [Corrected 2026-10-14 ~21:00 EEST: the first version "
                   "named only make_figures.py / report_digest.py.]")
POST_FIX = {
    "when": "2026-10-14 17:25-21:10 EEST",
    "what_happened": "after the fetch, an interrupted fix agent edited h_sens.py and h_truth.py locally at 17:25 "
                     "(fallback-inside-calibrated-bin maps and unsigned fine-vs-F maxima); that code never ran. The two "
                     "files were restored from attempts/pre_fix_20261014/ (sha256 0feed023... and 25f09d87..., equal "
                     "to the maas copies that produced results/); the edit is kept, unapplied, as "
                     "attempts/pre_fix_20261014/postrun_patch_1725_NOT_RUN.diff.txt",
    "remote_code_check": "post/remote_code_check.json (read-only sha256 of the remote mirror, 2026-10-14 ~20:55 EEST): "
                         "all compute modules equal",
    "no_rerun": "no maas rerun: the round-2 verifier reproduced every numerical result; the quantities the patch "
                "would have added are exact functions of the fetched arrays and are computed locally by h_post.py "
                "(post/h_post.json, post/H_post_*.csv): calibrated-halo-only importance shares from truth, truthcal "
                "and the truncation-curve zero points; per-shell DM-quantile fallback from truth.npz Jqfb; unsigned "
                "fine-vs-F truth maxima from results/truth.npz and F's truth.npz. Only the per-shell split of the "
                "quadratic maps in bin 15.3-15.5 would need the fine bases; it is given as a labelled estimate",
    "scratch": "/tmp/nagainis_halfdome_testH no longer exists on maas (deleted after Test I; checked 2026-10-14 "
               "~20:45 EEST); a rerun of truth/sens/params/beyond/binning would need the basis stage again (371 s)"}

POST_FIX_R3 = {
    "when": "2026-10-14 ~21:20-22:00 EEST (round-3 verification fixes)",
    "what_changed": "text, labels and local post-processing only; no maas rerun and no change to results/ (the round-3 "
                    "verifier reproduced every number from the fetched arrays). h_post.py: band labels state the edges "
                    "used (0.2335 / 0.5185 R200c = Test F's shell edges) plus an alternative band set with the fine edges "
                    "nearest below 0.25 / 0.5 R200c; binning_family_rms_pct_Rcut4 now separates det binning-only, S1 "
                    "binning-only (d_O) and S1 total error; diagonal (own-shell) fraction with both normalisations; "
                    "top-bin attribution; post/H_post_bands8.csv repeats the maas 8-band table (results/tables/H3_bands.csv, "
                    "edges 0.1051 / 0.2335 / 0.5185 labelled 0.1 / 0.25 / 0.5) with the edges used. make_figures.py: H5 panel (b) titled as the TOTAL S1 error; mass-share figure "
                    "wording. verify_h.py: gates G8_remote_code_check_is_current, G9 binning and band-label checks. "
                    "remote_code_check.py (new, local only): the read-only sha256 check of the maas mirror",
    "remote_code_check": "post/remote_code_check.json, rerun by remote_code_check.py after the last local code edit "
                         "(read-only; time in its checked_utc field); the compute modules are equal, the local-only "
                         "post-processing files (incl. verify_h.py) are listed under differ / local_only"}


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
                p = ROOT / "F_observables" / "results" / p.name           # the full run reads F's maas products
            out.add(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maas-run", default=None, help="JSON string of a run record to append")
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
    for r in runs:
        if r.get("attempt") == 3:
            r["fetched"] = FETCH_CORRECTED
            r["scratch_after_fetch"] = ("kept for Test I at the time (2.1 GB in /tmp/nagainis_halfdome_testH); deleted "
                                        "later (absent on 2026-10-14 ~20:45 EEST)")
    out = {
        "purpose": "Test H (sensitivity of tSZ / kSZ / FRB DM at the A2-ii painting target, best radial resolution: "
                   "67 X200 shells to 4 R200c + 8 extension shells to 12 R200c): full run on STRW maas; compute stages "
                   "only (figures and report tables are made locally; H0 export from the stacking cache is local)",
        "command": "cd $STUDY_ROOT && STUDY_ROOT=$STUDY_ROOT nice -n 19 python -B H_sensitivity/run_test_h.py --full "
                   "--workers 8 --scratch /tmp/nagainis_halfdome_testH [--stages ...]   (through cluster_tests.py run H "
                   "--workers 8 --rss-stop 24 --driver-args=\"--scratch /tmp/nagainis_halfdome_testH --stages ...\", "
                   "which adds nice 19, ionice and the load/memory guards)",
        "scratch": "the three fine bases (~2.4 GB) do not fit the 3.9 GB home quota on maas: they go to the node-local "
                   "/tmp/nagainis_halfdome_testH (20 GB disk), symlinked from results/work/g/f_base/work. After attempt 3 "
                   "the directory was kept for Test I and has since been deleted (absent on 2026-10-14 ~20:45 EEST)",
        "maas_runs": runs,
        "STUDY_ROOT_on_maas": "/home/nagainis/HalfDome_runtime/overlap_covariance_tests/HalfDome/"
                              "flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012",
        "smoke_command": "cd $STUDY_ROOT/H_sensitivity && python -B run_test_h.py --smoke   (local, 1 worker, ~1.5 min)",
        "local_preparation": "python -B H_sensitivity/h_inputs.py   (H0: export from stacking_choices_20261014/cache "
                             "into H_sensitivity/inputs/, mirrored to maas)",
        "stages": "prepare, basis, truth, sens, params, beyond, binning (default all, in order); --stages a,b,... reruns "
                  "some; each stage reads the earlier stages' outputs from results/",
        "exit_status": "0 = completed; 1 = a stage raised (traceback in results/run_manifest.json, later stages still "
                       "attempted); 2 = completed but the parity gate failed (results/parity.json)",
        "code_to_deploy": [f"H_sensitivity/{c}" for c in CODE],
        "reads_test_F_and_G_modules": "F_observables/f_common.py, f_basis.py, f_fit.py, f_kernels.py, f_pdf.py, "
                                      "f1_importance.py (+ f_methods via f_fit); G_physical_outer/g_common.py, "
                                      "g_pdf.py, g_physics.py - imported read-only and configured through environment "
                                      "variables (F_RESULTS / G_RESULTS under H_sensitivity/results/work/g); the fine "
                                      "grid is set on the imported f_common module object in the H process only",
        "writes": "H_sensitivity/results/ (compact npz/csv/json; the fallback PDF slot and the fine profiles in "
                  "results/work/, not copied back) and the scratch directory (the fine bases; kept for Test I, which deletes it)",
        "layout_requirement": "as Tests F and G: STUDY_ROOT/{analysis,kernels,F_observables,G_physical_outer,"
                              "H_sensitivity}, the earlier studies one level up, frb_map_generation two levels up "
                              "(engine inputs)",
        "inputs": rows,
        "inputs_total_MB": round(sum(r["bytes"] for r in rows) / 1e6, 1),
        "expected_resources_workers_8": {
            "local_estimates": "basis (3 sets, 75 shells): 1,330 CPU-s locally for the 3 sets; DM-PDF passes "
                               "~0.5-0.6 CPU-s per entry, Mecke ~7 CPU-s per entry; ~45k entries and ~1.8k Mecke "
                               "entries in total",
            "wall_total": "about 1.5-2.5 h at nice 19 with 8 workers",
            "peak_memory_GB": "after the attempt-2 fix: at every fork (DM-PDF passes, bootstrap pool) the parent holds no "
                              "fine basis (measured locally in full mode: 0.20 GiB with every truth-stage input, sky, "
                              "grid and column setup loaded); tree RSS (the guard's sum, which counts the parent's "
                              "copy-on-write pages once per forked worker) ~ 9 x 0.2 + 8 x (<= 1 GB private per worker) "
                              "<= ~10 GiB; parent-only peaks (no workers alive) <= ~3 GB in 'sens' (one basis + 2-halo "
                              "prefix sums); basis build (not rerun in attempt 3): parent 12.5 GiB, largest child 2.1 GiB; "
                              "rss-stop 24 GiB"},
        "copy_back": "H_sensitivity/results/ without work/ (cluster_tests.py fetch H)",
        "memory_fix_after_attempt_2": MEMORY_FIX,
        "reused_bases_attempt_3": REUSED_BASES,
        "post_fix_20261014": POST_FIX,
        "post_fix_round3_20261014": POST_FIX_R3,
    }
    (HERE / "CLUSTER_RUN.json").write_text(json.dumps(out, indent=1) + "\n")
    print(len(rows), "inputs,", out["inputs_total_MB"], "MB")


if __name__ == "__main__":
    main()
