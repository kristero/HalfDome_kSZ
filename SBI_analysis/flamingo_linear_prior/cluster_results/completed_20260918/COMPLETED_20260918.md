# Broad linear-prior 8192-row HalfDome tSZ dataset: completed

Downloaded 2026-09-18 from `/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915` (collector wrote `dataset/complete.json` on 2026-09-18 07:39 cluster time).
No jobs of this run remain in the queue. The earlier log-uniform "extended" root
(`halfdome_flamingo_extended_8192_20260915`) was retired on 2026-09-15 with 273 rows and never completed;
its 663 thinned design points were carried into this run (23 with imported spectra, 640 regenerated).

Contents: `dataset/` (theta, masked_clean_dl40, masked_noisy_cross_dl40, unmasked_clean_dl40, bin_edges,
validation_split, source_old_row, noise seeds), `observations/` (HalfDome B12 reference and three FLAMINGO
targets through the same binning), `logs/` (68 PBS logs), `plots/` (prior figures from preparation),
run metadata, and the new summary figures `summary_spectra.{png,pdf}`, `summary_design_and_run.{png,pdf}`
(`make_dataset_summary_plots.py`, statistics in `summary.json`).

Key numbers: 8192 rows (843 validation), 8169 maps generated in this run, median 261 s per map,
602 worker-hours (about 15.7k core-hours), 2.9% of noisy cross bins negative, all clean bins positive.
All four targets sit between the 17th and 37th percentile of the design in every bin, so the design
brackets them with margin on both sides; the design median is 5 times above the targets at ell ~ 3000
because the linear-uniform base puts most weight on large P0.
