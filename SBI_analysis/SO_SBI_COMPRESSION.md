# Full-data nine-parameter SO compression comparison

This is a new, controlled experiment, not a modification of your existing
trained estimators. It uses masked SO baseline deproj0 only, the existing
signed 40-bin D_ell product, and all nine varied Battaglia parameters.
No simulations are regenerated. No beam, noise, masking, flooring, or log10
operation is applied to the prepared spectra.

## Run on idark

From `/home/kristero10/HalfDome_kSZ`:

```bash
bash SBI_analysis/submit_so_sbi_compression_comparison.sh --dry-run
bash SBI_analysis/submit_so_sbi_compression_comparison.sh
```

This submits **five ordinary jobs**, without PBS arrays: one preparation,
three MAF training/evaluation jobs, and one dependent summary. At most three
run concurrently. Each requests `mini`, one node, 26 CPUs, 64 GB RAM, and
23:59:00. Training runtime is not guaranteed to fit in 24 hours.
Choose a working SBI Python environment with `PYTHON=/path/to/env/bin/python`.
The runtime preflight does not repair an incompatible NumPy/Numba/SBI stack.

Defaults for both prepared inputs are under:

```
/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/
  so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz
  so_masked_no_noise_ell80_7979_sbi_run.npz
```

Both are needed for the paired residual fit. `PREPARED_DATASET` and
`CLEAN_DATASET` override these paths. If the clean prepared file is missing,
prepare that product from the existing raw spectra and their metadata (not
from a CSV-order assumption), then pass the new file as `CLEAN_DATASET`:

```bash
python3 SBI_analysis/prepare_adrian_so_sbi_case_datasets.py \
  --input-dir /lustre/work/kristero10/adrian_dataset \
  --products masked_no_noise \
  --output-dir /lustre/work/kristero10/so_compression_clean_input
```

Check that its bin range, weighting, prior, theta, and Sobol identities match
the noisy input. Preparation refuses misaligned products rather than sorting
one array independently or subtracting different simulations.

Configuration is saved once in `experiment.json`. For another experiment,
use a new output directory, for example:

```bash
PCA_COMPONENTS=20 \
COMPRESSION_ROOT=/lustre/work/kristero10/adrian_9param_compression_pca20 \
bash SBI_analysis/submit_so_sbi_compression_comparison.sh
```

All train/evaluate jobs read the saved experiment, not changed environment
hyperparameters. Defaults: PCA=9, MOPED local rows=20000, shrinkage=0.05,
MAF hidden features=64, transforms=6, batch=1024, patience=20, max epochs=200,
2000 posterior samples per test row. The installed SBI MAF builder's other
defaults are identical between methods; this does not reuse an old MAF with
potentially different architecture or preprocessing.

## Statistical design

The last 500 original array rows are held out before any fitting. Rows outside
the saved BoxUniform prior are explicitly excluded everywhere, as requested;
neither bounds nor theta values are changed. The local input contains 524288
rows, of which 10751 are outside the saved prior. `experiment.json` reports the
actual remaining training and test counts. Ten percent of the non-held-out,
in-prior pool is a common validation set. PCA, MOPED, scaling, and regression
use only the optimization rows. `N_train` includes that validation set, as in
the previous dataset-size runs.

On the checked local files this is 513042 training-plus-validation rows,
461737 optimization rows, 51305 validation rows, and 495 eligible held-out
rows. No held-out row is moved into training to replace an excluded row.

The common input coordinates are:

```
s = lower_median(abs(x_optimization), axis=0), clamped to 1e-30
u = asinh(x / s)
y = (u - mean(u_optimization)) / std(u_optimization)
```

The lower median matches `torch.median`. Negative cross-spectra are retained.
Training-only affine standardization stabilizes covariance estimation and
does not discard information. The three inputs are:

1. `bins40`: all 40 coordinates, the uncompressed control.
2. `pca`: the leading nine eigenvectors of the optimization covariance of y.
   PCA selects variance, not parameter information; discarded low-variance
   directions can be informative. Retained variance is reported.
3. `moped`: up to nine local noise-whitened mean-sensitivity directions.
   All nine parameters are still inferred even if the compression rank is
   smaller than nine. No nuisance parameters are fixed.

MOPED uses the nearest 20000 optimization theta vectors to Battaglia12,
measuring distance in prior-range units. A 55-coefficient local quadratic
fits each transformed clean bin and each paired residual
`r = y_noisy - y_clean`. Their sum estimates the noisy conditional mean and
its derivatives J. Fitting the residual mean matters because nonlinear
asinh does not preserve zero-mean additive noise. The scatter of residuals
about this fit estimates local conditional noise, avoiding the changing
clean signal's covariance across the prior. It is **not** a 64-seed ensemble
covariance and not an exact fixed-theta covariance measurement.

The covariance is shrunk toward its own diagonal, retaining bin variances:
`C = 0.95 C_sample + 0.05 diag(C_sample)`. For `C = L L^T`, SVD of
`L^-1 J = U S V^T` gives weights `B = L^-T U`. Thus `B^T C B = I`, and
`J^T B B^T J` equals the fitted mean Fisher matrix `J^T C^-1 J` for retained
directions. This is a rotation of linear MOPED; unresolved singular modes
are dropped rather than assigned artificial information. Final compressed
coordinates are standardized on optimization rows before NPE.
The sample covariance accounts for the fitted residual mean's regression
degrees of freedom.

This is a **local approximate** compression. No covariance-derivative score
is included; cross-spectrum noise can depend on theta, and the likelihood
can be non-Gaussian. Broad nine-dimensional neighborhoods also introduce
regression error. Saved diagnostics include covariance condition, clean-fit
error relative to noise, and derivative changes using half the local rows.
Large changes call for a neighborhood/shrinkage sensitivity study or trusted
finite-difference derivatives. The local Fisher identity alone is not proof
of physically lossless compression. See
[Heavens et al.](https://arxiv.org/abs/astro-ph/9911102) and
[Alsing & Wandelt](https://arxiv.org/abs/1712.00012).

## Outputs and interpretation

Root: `/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0`.

- Root: immutable experiment settings, shared row identities and splits,
  fitted transforms, compressed full arrays, PCA/MOPED diagnostics.
- `bins40/`, `pca/`, `moped/`: density estimator pickle and state dict,
  prior, captured `show_train_summary=True` stdout, parsed best validation
  performances (higher is better), and training completion metadata.
- Each `evaluation/profiles/row*.npz`: posterior samples, truth, moments,
  prior-normalized error, pull, coverage, and raw prior acceptance.
- `summary/`: per-profile/per-parameter/aggregate CSV tables, PNG and JPG
  RMSE/prior, RMS standardized-error, Pearson-coefficient, posterior-width,
  coverage, and true-versus-mean plots. Pearson coefficients are printed on
  the nine-panel true-versus-mean comparisons. GetDist optionally adds the
  same **held-out synthetic observation** corner plot, with true markers.
  This example is selected near Battaglia12 in theta; it is explicitly not
  labelled as a Battaglia12 observation.

RMSE/prior is `sqrt(mean(((posterior_mean - truth)/prior_width)^2))`.
RMSE/std means `sqrt(mean(((posterior_mean - truth)/posterior_std)^2))`,
not mean(RMSE)/mean(std). Each is reported per parameter; aggregate RMS uses
all parameter/profile errors after normalization. Correlation is Pearson
across the shared test rows for each parameter. Better RMSE with maintained
coverage is stronger evidence of improvement than narrower contours alone.
Single-seed training and finite posterior sampling still have uncertainty;
this first comparison does not establish statistically significant gains.

## Resume or plot locally

Sampling is checkpointed per profile and limited to 200000 proposals or
120 seconds (checked between network calls). Failed rows are reported and
prevent a selective-subset comparison. Resubmission reuses completed models
and test rows; it retries failed ones, but cannot cure an out-of-support
observation by waiting. Training interrupted before its completion marker
restarts; completed density estimators are saved before evaluation.

```bash
# One resumed method, without resubmitting the other jobs:
qsub -v COMPRESSION_STAGE=run,COMPRESSION_METHOD=moped \
  SBI_analysis/run_so_sbi_compression_comparison.pbs

# Rerun plots only, no training or sampling:
python3 SBI_analysis/run_so_sbi_compression_comparison.py summarize

# Live log (use the filename printed/listed for your job):
ls /lustre/work/kristero10/adrian_9param_compression_baseline_deproj0/logs/
```

For local plotting download `experiment.json`, `shared.npz`,
`pca_diagnostics.npz`, `moped_diagnostics.npz`, and each method's
`training_complete.json` plus complete `evaluation/` folder. Then run:

```bash
python SBI_analysis/run_so_sbi_compression_comparison.py summarize \
  --output-root /path/to/downloaded/experiment
```

This needs NumPy, Matplotlib, and optionally GetDist, not Torch/SBI or the
large original raw dataset. Add `--skip-corner` to omit GetDist.
