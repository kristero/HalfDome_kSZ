"""Export restart provenance and a live production snapshot; do not alter workers."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile

import h5py


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "code"))
    from prior import digest, write_json
    from worker import verify_frozen
    from restart import live_jobs, completed_files
    manifest = verify_frozen(root)
    run = json.loads((root / "run_config.json").read_text())
    gate = json.loads((root / "preflight/quality_gate.json").read_text())
    assert gate["passed"] and gate["manifest_sha256"] == digest(root / "manifest.json")
    retirement = json.loads((root / "audit/previous_run_retirement.json").read_text())
    reuse = json.loads((root / "audit/reuse_import.json").read_text())
    submission = json.loads((root / "audit/production_submission.json").read_text())
    assert submission["phase"] == "released"
    assert completed_files(Path(run["previous_run"])) == retirement["completed_files_after"]
    assert not live_jobs(Path(run["previous_run"]))
    jobs = live_jobs(root)
    states = {state:sum(j["job_state"] == state for j in jobs.values()) for state in ("R", "Q", "H", "E")}
    generated = []
    for path in sorted((root / "chunks").glob("chunk_*/row_*.h5")):
        if ".tmp." in path.name:
            continue
        with h5py.File(path, "r") as row:
            if "reuse_provenance_json" not in row.attrs:
                status = json.loads(row.attrs["status_json"])
                generated.append(dict(file=str(path.relative_to(root)), sha256=digest(path),
                                      seconds=status["elapsed_seconds"], peak_rss_GiB=status["peak_rss_GiB"],
                                      column_error=status["numerics"]["maximum_column_relative_error"],
                                      noise_seed_row_id=status["noise"]["seed_row_id"]))
    snapshot = dict(utc=datetime.now(timezone.utc).isoformat(), states=states,
                    jobs={job_id:{key:j.get(key) for key in ("Job_Name", "job_state", "queue", "comment")}
                          for job_id,j in jobs.items()}, target_rows=run["count"], reused_rows=reuse["reused_count"],
                    newly_generated_rows_observed=len(generated), generated_rows=generated,
                    manifest_sha256=digest(root / "manifest.json"), old_jobs_remaining=0)
    write_json(root / "production_status.json", snapshot)
    audit = json.loads((root / "audit.json").read_text())
    handoff = dict(run_root=str(root), target_rows=run["count"], parameter_order=manifest["parameter_order"],
                   prior_config=manifest["prior"], normalization=audit["normalizing_mass_Z"],
                   normalization_standard_error=audit["normalizing_mass_standard_error"],
                   prior_factory="code/sbi_prior.py: ExtendedPrior(config, normalization)",
                   source_old_row="design/source_old_row.npy; -1 means a fresh proposal",
                   noise_seed_row_ids="design/noise_seed_row_ids.npy",
                   noise_mode="independent_per_row; old seeds preserved for selected old points",
                   dataset_export="dataset/ after all chunks complete", observations="observations/",
                   old_model_usable_as_new_prior_posterior=False, training_started=False)
    write_json(root / "training_handoff.json", handoff)
    maximum_error = max(case["numerics"]["maximum_column_relative_error"] for case in gate["cases"])
    regression = max(value for case in gate["cases"] for value in
                     case.get("full_map_regression_max_error_over_peak", {}).values())
    text = """# Broad linear-prior production restart

Snapshot: {utc}. Cluster root: `{root}`.

The broad nine-parameter bounds and every joint cut are unchanged. All base
coordinates are linear-uniform; accepted marginals remain correlated and
nonuniform. FLAMINGO fits only annotate plots and check coverage.

## Preservation and reuse

- Previous completed profiles preserved: {preserved}; their file hashes are unchanged.
- Completed profiles imported with identical theta, spectra and noise: {reused}.
- Target size: {count}. Simulations still required immediately after import: {remaining}.
- The full old design was thinned with probability P0*xc/240 before inspecting
  completion. The new design contains {old_points} such points and {fresh_points}
  fresh points, targeting the same conditional density. Unavailable selected
  old points are generated normally. No unselected old profiles were deleted.

## Checks

- Six fresh NSIDE4096 preflight maps passed, with independent SO splits.
- Maximum native-versus-scaled column error: {column_error:.3g}.
- Largest historical clean-spectrum regression error relative to its peak: {regression:.3g}.
- Noise seeds checked through the 524288-row configuration, including reuse,
  the fresh namespace, preflight and observation seeds.
- Prefix stability and conditional-density consistency checks passed.
- A denser 33x33 diagnostic finds {dense_count} design points slightly above the
  original finite-Y200 grid bound; maximum ratio {dense_max:.8g}, versus 30.
  The original 9x9 sampling rule is retained exactly; no extra rejection is applied.

## Production snapshot

{running} running, {queued} queued, {held} held by dependencies. Concurrency cap:
{concurrency} (6 mini + 16 mini_B). {generated} newly generated rows were observed
in addition to the imported rows at this snapshot. This is a started production
run, not a completed 8192-row dataset or newly trained SBI model.

Production worker IDs and commands: `audit/production_submission.json`.
Collector: `{collector}`. It waits for every terminal worker chain.

## Outputs and source

Plots: `plots/prior_all_parameters`, `plots/joint_prior_all_parameters`, and
`plots/joint_prior`, each as PNG and PDF. Exact values are in
`plots/prior_and_fits.csv`. Markers are cosmology-corrected effective clean-spectrum
fits, not posterior confidence bounds. The original old SBI bounds are dashed.

Sources and checks are included in the review archive. `changed_files.txt` lists
new/adapted and unchanged copied sources. To scale, use `prepare.py --count 524288`
with a fresh root; do not edit a frozen design. Noise IDs and the first 8192
parameter rows remain prefix stable.
""".format(utc=snapshot["utc"], root=root, preserved=reuse["previous_completed_files"],
           reused=reuse["reused_count"], count=run["count"], remaining=run["count"]-reuse["reused_count"],
           old_points=manifest["reuse_design"]["old_source_points"], fresh_points=manifest["reuse_design"]["fresh_points"],
           column_error=maximum_error, regression=regression,
           dense_count=audit["denser_Y_grid"]["outside_9x9_gate_bounds"], dense_max=audit["denser_Y_grid"]["max"],
           running=states["R"], queued=states["Q"], held=states["H"], generated=len(generated),
           concurrency=submission["concurrency"], collector=submission["collector_job_id"])
    (root / "RUN_REPORT.md").write_text(text)
    paths = []
    for name in ("code", "inputs", "design", "preflight", "observations", "audit", "tools"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and ".tmp." not in p.name)
    for name in ("manifest.json", "run_config.json", "noise_seed_check.json", "audit.json",
                 "linear_joint_sbi_prior.pkl", "RUN_REPORT.md", "training_handoff.json",
                 "production_status.json", "changed_files.txt"):
        paths.append(root / name)
    artifacts = {str(path.relative_to(root)):digest(path) for path in sorted(set(paths))}
    write_json(root / "review_manifest.json", artifacts)
    archive_path = root / "review_artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(set(paths)):
            archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(root / "review_manifest.json", arcname="review_manifest.json")
    (root / "review_artifacts.sha256").write_text(digest(archive_path) + "  review_artifacts.tar.gz\n")
    print(json.dumps(dict(snapshot=snapshot["utc"], states=states, reused=reuse["reused_count"],
                          generated=len(generated), artifacts=len(artifacts), archive_sha256=digest(archive_path)), indent=2))


if __name__ == "__main__":
    main()
