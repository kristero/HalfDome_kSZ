# Files created or edited in this isolated study

No existing calibration code, previous study outputs or raw simulation files were edited.

Latest report-finalization changes:

- `make_reports.py`: Final report reflow, physical-variable explanations, resource limits and atomic PDF saves.
- `build_delivery.py`: Missing-accounting handling, one log per submitted job, gallery links and delivery hashes.
- `run_local_report.ps1`: Windows/WSL memory guard for report and QA stages; separate local logs.
- `verify_final_reports.py`: Frozen scientific hashes, final PDF hashes/page counts and full-page render sheets.
- `finish_delivery.py`: Final QA status, cluster-receipt preservation and study-only cleanup.
- `README.md`: Final scope, accounting definitions and cluster-verification limitations.

Full source inventory for the isolated study:

Source files:

- `aggregate_and_plot.py`
- `aggregate_recovered.py`
- `build_delivery.py`
- `bundle_idark.py`
- `cluster_control.py`
- `compact_strw.py`
- `emergency_local_cleanup.sh`
- `extract_strw.py`
- `field_statistics.py`
- `finalize_control.py`
- `finish_delivery.py`
- `halo_union_strw.py`
- `make_reports.py`
- `painting.py`
- `physics.py`
- `plot_measured_preview.py`
- `prepare_idark.py`
- `refine_figures.py`
- `repair_plotting_source.py`
- `run_aggregate.pbs`
- `run_compare.pbs`
- `run_comparisons.py`
- `run_local_report.ps1`
- `run_prepare.pbs`
- `run_refine_figures.pbs`
- `run_refit.pbs`
- `run_report.pbs`
- `run_report_retry.pbs`
- `run_retry0.pbs`
- `sync_metadata.py`
- `unpack_delivery.py`
- `verify_final_reports.py`
- `verify_numerics.py`

Documentation: `README.md`, `FILES_CHANGED.md`.

Generated deliverables: `index.html`, `figures/`, `output/pdf/`, statistics CSV/JSON and SHA-256 manifests.

Executed version-1 scientific source remains under `executed_code_v1/`. `aggregate_recovered.py` preserves the plotting source recovered after an interrupted local edit.
