# Files created or updated for the 21 September audit

All paths below are relative to this report directory.

## New audit and orchestration source files

- `analyze_extreme_noise.py`
- `build_report.py`
- `check_repeats.py`
- `collect_results.py`
- `deploy_recovery.py`
- `extreme_analysis.pbs`
- `fetch_recovery.py`
- `partition_worker.py`
- `plot_audit.py`
- `prepare_recovery.py`
- `publish_report.py`
- `queue_extreme_analysis.py`
- `queue_extreme_noise.py`
- `repair_submission.py`
- `run_stock_audit.py`
- `stock_cache_profiles.jl`
- `stock_probe.jl`
- `test_deployment.py`
- `test_partition.py`

## Generated artifacts and copied sources

- `REPORT.md`, `plots/`, `results/`, and the dated PDF: report, evidence and figures.
- `inputs/stock_profiles*.jl`: snapshots of original XGPaint commit 5dd0b57.
- `../tsz_8192_validation_recovery_20260921/`: frozen physical producers, revised task plan, disjoint workers and PBS scripts.
- `../tsz_8192_noise_extremes_20260921/`: frozen producers and two additional matched-noise controls.
- `../tsz_spherical_preflight_20260920/results/fullsky.json` and its plots: regenerated from the completed independent resolution controls.

The recovered jobs use byte-identical physical producer files. This audit changed job orchestration and added analysis/probes; it did not change prior bounds or the simulation physics.
