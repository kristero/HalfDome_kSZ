# File inventory

All changes are confined to the new directory
`SBI_analysis/tsz_truth_mean_validation_20261006/`.
Earlier run code, models, figures and PDFs are unchanged.

- `diagnostic_metrics.py`: physical truth-mean statistics, paired Pearson case
  bootstrap and numerical gates.
- `evaluate_truth_mean.py`: idark inference over the reserved test rows,
  bounded/unbounded parity checks, 580 primary control, corner-case selection,
  physical posterior draws and source/output SHA-256 provenance.
- `plot_truth_mean.py`: completed-run truth-mean/metric/corner plots; separate
  existing-only mode for archived scale-cut results and bandwidth illustration.
- `submit_idark.py`: one-job ledger, free-node selection across eligible queues,
  active-job guard.
- `truth_mean.pbs`: 26 CPUs, 16 GB, two-hour compute-node execution.
- `build_supplement.py`: one combined 44-page PDF, updated reader guide,
  rendered equations, previous seed/mask evidence and new truth-mean tests.
- `publication_corners.py`, `publication_corners.pbs`: compute-node formatting
  from saved posterior samples, larger text and corrected diagonal density ticks.
- `finalize_artifacts.py`: local/cluster completion and SHA-256 verification.
- `README.md`: contracts, physics, interpretation, limitations and run commands.
- `FILES_CREATED.md`: this inventory.
- `evidence/summary.json`, `prior_final_test_linear.json`,
  `prior_final_test_logit.json`, `baseline_spectra.npz`: copied archived inputs.
- `evidence/frozen_input_hashes.json`: earlier verified hashes for all fifteen
  compared network files and the known frozen loader/original feature inputs.
- `data/saved_scale_cut_comparison.csv`: tabulation of saved held-out sky metrics.
- `figures/saved_scale_cut_comparison.png` and `.pdf`: archived-method comparison.
- `figures/band_width_illustration.png` and `.pdf`: actual archived spectrum
  averaged with widths25,50,100; illustration only, not new model results.
- `LOCAL_CHECKS.json`: local preparation checks and SHA-256 provenance.
- `PREPARATION_STATUS.json`: historical preparation status before cluster access
  was restored; current completion is recorded in `RUN_STATUS.json`.

Completed scientific outputs include INFERENCE_COMPLETE.json,
INFERENCE_VERIFIED.json, PLOTS_COMPLETE.json, PUBLICATION_COMPLETE.json,
TRUTH_MEAN_SUMMARY.json, CORNER_PLOT_DEFINITION.json, five test summaries,
four corner draw files with three cases each, metrics CSVs and 18 figure pairs.
The final PDF is in `output/pdf/`; PAGE_MANIFEST.json identifies its source
pages. PDF_REVIEW.json, CLUSTER_VERIFIED.json and VERIFIED.json record final
review and delivery. `source_snapshot/` and
`evidence/initial_plots_complete.json` preserve the original inference/plot
versions before the publication formatting pass. Both job ledgers and useful
resource logs are retained. Obsolete transfer archives, review PNGs and
bytecode are removed after verification.
