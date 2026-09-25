# Two-parameter fixed-noise compression comparison

This is a conditional diagnostic for P0 and beta, with the seven remaining
Battaglia pressure parameters fixed to Battaglia12. It is not the earlier
nine-parameter marginalization experiment and not an independent-noise forecast.

## Inputs and identity

- Raw cluster data: `/lustre/work/kristero10/two_param_P0_beta_32k/y100`.
- The user confirmed that these simulations use the first 32,768 data rows of
  `battaglia_sobol_P0_beta_524288.csv` from Downloads. The script checks its SHA256,
  named columns, fixed parameters and every filename's split/local/global labels.
  The mapping is global row = `(split - 1) * 128 + local`, one-based.
- Priors, matching that design: P0 = [1.81, 34.39], beta = [3.48, 5.22].
  These are supplied bounds, never estimated from sample extrema.
- Six-product matrices are C_ell on ell = 0..7979. Rows 1 and 2 are respectively
  masked clean and masked baseline-deproj0 cross spectra. The existing 2 arcmin
  beam is retained, not applied again.
- All filenames say seed 12345. The strong shared high-ell noisy-minus-clean
  residual pattern supports a fixed-noise interpretation. Names alone do not
  prove the original RNG implementation; full original run provenance is absent.
- The corner observation is the separately generated Battaglia12 spectrum with
  the same mask and noise seed 12345. Its raw file and simulation-completion
  record are copied under the isolated experiment's `inputs/`.

## Computation

1. Preserve signed values, convert C_ell to D_ell = ell(ell+1) C_ell / (2 pi),
   and average with weights 2 ell + 1 in Delta-ell = 200 bins, ell = 80..7979.
   The first and last bins contain only multipoles in that range. There are 40 bins.
2. Reserve the last 1,000 global rows for testing. Exclude out-of-prior rows,
   shuffle the remaining pool once with seed 42, and take nested training sizes:
   256, 512, 1024, 2048, 4096, 8192, 16384, 24576, max. With no excluded rows,
   max is 31,768. Each size includes a 10% validation subset; only 90% fit weights.
3. Fit the scale s (lower median absolute value per bin), asinh(x/s), and its
   standardization on optimization rows only. Apply the exact saved transform
   to validation, held-out spectra and Battaglia12. There is no log10 or clipping.
4. Compare all 40 transformed bins with two MOPED components. For each size,
   fit a local quadratic mean around Battaglia12 using at most 2,048 optimization
   rows. Fit the conditional mean of the paired noisy-minus-clean residual too.
   Estimate covariance after removing this fitted residual mean, correct the
   regression degrees of freedom, and shrink 5% toward its diagonal.
   A whitened derivative SVD provides two linear compression directions.
   Diagnostics test the retained local mean-Fisher matrix and half-neighborhood
   derivative stability. Compression is refitted at every N, without test leakage.
5. Train a separate MAF for each method and N, with 64 hidden features, 6 transforms,
   patience 60 and an epoch cap of 1000. Restore the best-validation weights before
   saving the estimator, state dictionary, prior and validation summary.
6. Draw 2,000 samples per held-out profile and 10,000 at Battaglia12. Cache each
   profile. Sampling has finite proposal/time budgets; failures are reported, not
   hidden by wider priors, clipped samples or silent removal from the summary.

The MOPED covariance here is a fitted residual metric in transformed coordinates.
Because the simulation noise is reused, it is NOT a covariance of repeated SO
noise realizations at fixed theta. The local fitted mean-Fisher preservation
test does not prove preservation of the full simulator likelihood. Held-out
RMSE and correlations determine whether this compression works in this dataset.
Fixed-noise corner widths must not be presented as independent-noise constraints.

## Metrics and plots

For parameter j and held-out profile i, error = posterior_mean - theta_true:

- RMSE/prior_j = sqrt(mean_i((error_ij / prior_width_j)^2)).
- RMSE/std_j = sqrt(mean_i((error_ij / posterior_std_ij)^2)). Each profile is
  normalized before averaging, not by dividing mean RMSE by mean standard deviation.
- Combined curves average squared normalized errors over both profiles and
  parameters before taking the square root. They do not average parameter RMSEs.
- Pearson r compares posterior mean and truth across the identical held-out rows.
- Largest-N true-versus-mean panels include r and posterior-standard-deviation bars.
- Battaglia12 corners compare bins40 and MOPED at minimum, middle and maximum N.

All plots are saved as PNG, JPG and PDF. Cases are labelled fixed-noise diagnostics.
The summary requires all held-out profiles in both methods, with valid caches.

## Cluster commands

The staged code is independent of the cluster repository checkout. Review
`so_two_param_compression_convergence.env` before submitting. The submission
script checks CSV identity, inputs and plotting imports before sending jobs.

```bash
ROOT=/lustre/work/kristero10/two_param_compression_convergence_32k_v1
export COMPARISON_CONFIG="$ROOT/code/so_two_param_compression_convergence.env"
bash "$ROOT/code/submit_so_two_param_compression_convergence.sh"
```

This submits one preparation job, 18 individual training/evaluation jobs in four
dependency lanes, and one summary job. There are no PBS arrays and no more than
four concurrent training jobs from this submission. Jobs request mini,
26 CPUs and 23:59:00; preparation needs 4 GB, while training and summary request
32 GB. Do not launch a duplicate submission into the same root.

```bash
cat "$ROOT/submission_jobs.txt"
qstat -u "$USER"
ls -lt "$ROOT/logs/"
tail -F "$ROOT/logs/<chosen-job-log>.log"

# Regenerate summary without retraining after runs are complete:
python3 "$ROOT/code/run_so_two_param_compression_convergence.py" summarize --root "$ROOT"

# Retry one incomplete run using its saved model and completed profile caches:
qsub -v "COMPARISON_CONFIG=$COMPARISON_CONFIG,STAGE=run,METHOD=moped,N_TRAIN=31768" \
  "$ROOT/code/run_so_two_param_compression_convergence.pbs"
```

Results: `$ROOT/N<size>/<bins40-or-moped>/`; summary: `$ROOT/summary/`.
Successful final output includes `$ROOT/summary/summary_complete.json`.
The preparation and individual completion markers are not a completed comparison.

From Windows PowerShell, after successful completion:

```powershell
scp -O -r idark:/lustre/work/kristero10/two_param_compression_convergence_32k_v1/summary `
  //wsl.localhost/myrootfs/home/cbllover/HalfDome/SBI_analysis/outputs/two_param_compression_convergence_32k_v1/
```

Local numerical tests use only NumPy and do not require training. The synthetic
end-to-end smoke covers training, checkpoint restoration, evaluation and plots,
but is not a scientific validation of the real 32k data.

## Comparison with nine-parameter results (2026-09-13)

`compare_so_fixed_two_vs_nine.py` reads the finished two-parameter summary, the
nine-parameter best-validation compression reference, and optionally the earlier
nine-parameter 40-bin convergence sweep. It performs no training or resampling.
It checks run completion/experiment IDs, test truth labels, missing or repeated
metric rows, and saved training/test splits before producing plots. The old
paired-index comparison script is not appropriate: these are DIFFERENT Sobol
designs and different test populations, not the same simulations with fewer
posterior columns.

The cross-experiment P0/beta plots use COMMON reference widths 32.58 and 1.74,
respectively, from the new two-parameter saved prior. The old nine-parameter
inference priors are slightly narrower; their native normalization is retained
in `comparison_metrics.csv`. Re-expressing errors in common units does not change
the trained prior or make the experiments controlled/paired. The aggregate
two-parameter plots in the original `summary/` remain unchanged.

The nine-parameter compression reference is explicitly PRELIMINARY: 489 of 495
eligible test profiles shared across the earlier methods. Six difficult cases
are missing, so those metrics may be optimistic. It contributes isolated points
at N=513042, not an invented MOPED convergence curve. The legacy 40-bin sweep
uses 100 tests and a different training recipe; best-validation snapshot selection
was not recorded there. The two-parameter runs use all 1000 held-out profiles,
with seven nuisance parameters fixed and reused noise. Near-perfect correlations
in that problem do not establish independent-noise posterior calibration.

Submit just the light plotting/comparison job, after staging the script and PBS:

```bash
ROOT=/lustre/work/kristero10/two_param_compression_convergence_32k_v1
qsub -o "$ROOT/logs/comparison_two_vs_nine_pbs.out" \
  "$ROOT/comparison_code_20260913/run_so_fixed_two_vs_nine_comparison.pbs"
```

This job uses one CPU, 8 GB and at most 30 minutes on mini; it does not load SBI
or the density estimators. Its default output is
`$ROOT/comparison_with_nine_20260913/`. It saves PNG/JPG/PDF correlation, RMSE/common
prior-range and RMS-standardized-error comparisons for P0 and beta; all-nine-
parameter correlation/native-prior-RMSE panels; and separate largest-N true-vs-mean
comparisons for 40 bins and MOPED. `comparison_complete.json` records input hashes,
the numerical checks and the scientific limitations. The complete input metric
table and the largest-run table are saved alongside the plots.

Local copies are placed under
`SBI_analysis/outputs/two_param_compression_convergence_32k_v1/` in separate
`summary/` and `comparison_with_nine_20260913/` folders. The plotting program's
strict audit needs the original run metadata/splits; the rendered plots and
exported metric tables can be inspected locally without any pickled model.
