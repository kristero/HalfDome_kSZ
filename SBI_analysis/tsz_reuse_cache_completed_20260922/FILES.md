# Files created or updated on 22 September 2026

All new analysis source is in this directory. Frozen simulator/producer code was not edited in this action.

## New source files

- `analyze_completed.py`
- `build_report.py`
- `cluster_evidence.py`
- `make_plots.py`
- `publish_and_cleanup.py`

## New results and documents

- `REPORT.md`: completed report, from the same content as the eight-page PDF.
- `results.json`, `spectra.npz`: recomputed comparisons and plotting data.
- `timings.csv`, `errors.csv`, `resolution.csv`, `pixel_quadrature.csv`: machine-readable tables.
- `plots/`: five PNG and five editable SVG figures.
- `cluster_audit.json`: empty queue and cluster-side checksum verification.
- `cleanup.json`: per-archive hashes and reasons for removal or retention.
- `qa.json`: final PDF and result validation.
- `delivery_manifest.json`, `publication.json`: deliverable hashes and cluster verification.
- `../../output/pdf/tsz_completed_performance_20260922.pdf`: final PDF (local); `report.pdf` inside the cluster bundle.

## Existing experiment paths affected

- `../tsz_reuse_cache_20260921/controls/`, `edge_extension/controls/`, `work_balance/controls/`, `results/`, `plots/`: refreshed local copies of completed cluster products via the existing fetch.py; 312 products verified.
- `../tsz_reuse_cache_20260921/fetch_manifest.json`: new complete fetch manifest; also saved on the cluster.
- Duplicate transfer archives listed in cleanup.json were deleted only after every member matched an extracted file. Distinct archived versions were retained.

No production launch, new prior cut or noise-seed change is performed by these reporting scripts.
