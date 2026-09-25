# Files added or updated for completed validation

The earlier partial-results PDF and its scripts are retained.

## New report and audit sources

- `audit_completed.py`
- `build_report.py`
- `fetch_completed.py`
- `plot_completed.py`
- `prepare_8192_diagnostic.py`
- `publish_completed.py`
- `test_prepared.py`
- `verify_analysis.py`

## New prepared package

`../tsz_diagnostic_256_8192_prepared_20260921/`:

- Revised `diagnostic.py`: raw8192, atomic row claims, preserved failures, frozen-input checks and one final collector.
- Revised `diagnostic_row.jl`: raw8192 assertion, validated lean allocation and a load-only guard.
- Revised `diagnostic.pbs` and `diagnostic_analysis.pbs`: four explicit workers, one dependent collector/analysis; not submitted.
- `fullsky_test.jl` and `spherical_truncation_profiles.jl`: byte-identical copies from the completed validation.
- `diagnostic_analysis.py`, `unbinned_moped.py`, `design_tests.py`, `run_fullsky.py`, `stage_observations.py`, `test_unbinned_moped.py`: copied support and analysis sources.
- `diagnostic_256/manifest.json`: updated resolution, source hashes and preparation provenance.
- Saved theta, noise seeds, held-out mask and three FLAMINGO observation artifacts: preserved byte-for-byte.
- `README.md`: scope and remaining numerical limits.

## Retrieved/generated evidence

- Completed task/analysis products refreshed in the recovery and extreme-noise directories.
- This directory: cluster reports, audits, five figure pairs, REPORT.md and artifact manifest.
- Final standalone report: output/pdf/tsz_completed_validation_20260921.pdf.
- No simulations or SBI training were launched.
