# Broad linear-prior production restart

Snapshot: 2026-09-15T14:42:48.446199+00:00. Cluster root: `/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915`.

The broad nine-parameter bounds and every joint cut are unchanged. All base
coordinates are linear-uniform; accepted marginals remain correlated and
nonuniform. FLAMINGO fits only annotate plots and check coverage.

## Preservation and reuse

- Previous completed profiles preserved: 273; their file hashes are unchanged.
- Completed profiles imported with identical theta, spectra and noise: 23.
- Target size: 8192. Simulations still required immediately after import: 8169.
- The full old design was thinned with probability P0*xc/240 before inspecting
  completion. The new design contains 663 such points and 7529
  fresh points, targeting the same conditional density. Unavailable selected
  old points are generated normally. No unselected old profiles were deleted.

## Checks

- Six fresh NSIDE4096 preflight maps passed, with independent SO splits.
- Maximum native-versus-scaled column error: 2.89e-15.
- Largest historical clean-spectrum regression error relative to its peak: 6.19e-16.
- Noise seeds checked through the 524288-row configuration, including reuse,
  the fresh namespace, preflight and observation seeds.
- Prefix stability and conditional-density consistency checks passed.
- A denser 33x33 diagnostic finds 2 design points slightly above the
  original finite-Y200 grid bound; maximum ratio 30.034271, versus 30.
  The original 9x9 sampling rule is retained exactly; no extra rejection is applied.

## Production snapshot

12 running, 10 queued, 43 held by dependencies. Concurrency cap:
22 (6 mini + 16 mini_B). 12 newly generated rows were observed
in addition to the imported rows at this snapshot. This is a started production
run, not a completed 8192-row dataset or newly trained SBI model.

Production worker IDs and commands: `audit/production_submission.json`.
Collector: `597857.idark`. It waits for every terminal worker chain.

## Outputs and source

Plots: `plots/prior_all_parameters`, `plots/joint_prior_all_parameters`, and
`plots/joint_prior`, each as PNG and PDF. Exact values are in
`plots/prior_and_fits.csv`. Markers are cosmology-corrected effective clean-spectrum
fits, not posterior confidence bounds. The original old SBI bounds are dashed.

Sources and checks are included in the review archive. `changed_files.txt` lists
new/adapted and unchanged copied sources. To scale, use `prepare.py --count 524288`
with a fresh root; do not edit a frozen design. Noise IDs and the first 8192
parameter rows remain prefix stable.
