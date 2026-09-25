# Files created for the reuse/cache experiments

All implementation changes are additions under `SBI_analysis/tsz_reuse_cache_20260921/`. No existing simulator or prepared-dataset source was edited. Frozen numerical copies are identified in README.md.

## Source and scheduler files

- `analysis.pbs`
- `analyze.py`
- `benchmark.jl`
- `edge_extension/analyze.py`
- `edge_extension/entrypoint.jl`
- `edge_extension/gate.jl`
- `edge_extension/gate.pbs`
- `edge_extension/prepare_submit.py`
- `edge_extension/run.py`
- `edge_extension/smooth_exterior.jl`
- `edge_extension/worker.pbs`
- `fetch.py`
- `followup/baseline.pbs`
- `followup/baseline_cache.py`
- `followup/baseline_cache_v2.py`
- `followup/baseline_v2.pbs`
- `followup/cache_probe.jl`
- `followup/launch.py`
- `followup/prepare_submit.py`
- `followup/probe.pbs`
- `fullsky_test.jl`
- `gate.jl`
- `gate.pbs`
- `pixel_progress.py`
- `pixel_windows.py`
- `prepare.py`
- `publish_report.py`
- `reference.pbs`
- `report_progress.py`
- `run.py`
- `snapshot_cluster.py`
- `spherical_truncation_profiles.jl`
- `submit.py`
- `work_balance/analyze.py`
- `work_balance/balanced_painter.jl`
- `work_balance/entrypoint.jl`
- `work_balance/gate.jl`
- `work_balance/gate.pbs`
- `work_balance/prepare_submit.py`
- `work_balance/run.py`
- `work_balance/worker.pbs`
- `worker.pbs`

## Main reports and measured evidence

- `README.md`: scope, physics, memory, cache grids, commands.
- `REPORT.md`, `results/progress.json`: completed numerical checks and full-catalogue controls.
- `PIXEL_PROGRESS.md`, `results/pixel_progress.json`: finite pixel-quadrature comparison.
- `followup/baseline_cache.json`: default-cache error in unbinned MOPED units.
- `followup/probe/cache_probe.toml`: 67,584 scalar cache comparisons.
- `edge_extension/README.md`: analytic explanation of the exterior derivative kink.
- `plots/`: figures with concise labels and large fonts.
- `manifest.json`, subdirectory manifests and `fetch_manifest.json`: source/product SHA256 evidence.
- Submission journals and per-control request/status/timing files: cluster provenance.
