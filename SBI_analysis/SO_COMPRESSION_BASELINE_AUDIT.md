# Why the new 40-bin baseline is worse

Audit date: 2026-09-10. No estimators were retrained or modified in this audit.

## Verified comparison

The old result is in `convergence_tests/last100_param_metrics.csv`, for
`masked_baseline_noise_cross_deproj0`, N=523788. Its P0 Pearson coefficient
was 0.950884 over the last 100 profiles. The compression summary instead
uses 489 of 495 eligible test profiles. Six difficult observations were
omitted because at least one method failed sampling; this was explicitly
approved as a preliminary, potentially optimistic comparison.

The new audit compares exactly the same **97 observations** available in
both analyses. It checks all nine truth values and the prior normalization,
recomputes pulls, and independently calculates Pearson r with NumPy.

| Model | P0 r | beta r | P0 RMSE/prior | beta RMSE/prior |
|---|---:|---:|---:|---:|
| Old 40-bin MAF | 0.94819 | 0.63396 | 0.09659 | 0.21737 |
| New bins40 | 0.75581 | 0.39672 | 0.19807 | 0.25760 |
| PCA | 0.94076 | 0.61593 | 0.10181 | 0.22116 |
| MOPED | 0.98745 | 0.81463 | 0.04886 | 0.16322 |

The truth arrays match exactly. A test-row change, swapped labels, or a
different formula for Pearson r does not explain the degradation.
The new summary is not reusing your earlier trained 40-bin model.

## Training was substantially different

Architecture was inspected from the saved estimators, not inferred from
filenames. The old trainer calls `train()` without a batch-size override;
the installed cluster SBI 0.22 train signature has default batch size **50**.

| Setting | Old N523788 | New bins40 |
|---|---:|---:|
| MAF hidden features | 50 | 64 |
| Autoregressive transforms | 5 | 6 |
| Batch size | 50 | 1024 |
| Early-stopping patience | 60 | 20 |
| Epoch limit | SBI default, effectively unlimited | 200 |
| Epochs actually trained | 452 | 110 |
| Saved best epoch | not independently recovered | 90 |
| Printed best validation performance | 21.2621 | 11.7287 |
| Training plus validation rows | 523788 | 513042 |
| Input dimensions | 40 | 40 |

The old run used about 4.3 million optimizer updates versus about 50,000
for the new bins40 run, approximately 86 times as many. Counts are estimates
from rows, batches and epochs; exact loader rounding is immaterial here.
An epoch count is not an equal optimization budget when batch sizes differ.
Increasing batch size without scaling the training budget changed the
experiment materially. Early stopping can end training in a poor plateau;
the phrase "successfully converged" does not establish posterior accuracy.

This is the strongest identified explanation, **not a controlled causal
proof** that changing only batch size fixes everything. Architecture,
random splits, early stopping and exclusion of out-of-prior rows also changed.
The validation sets/target distributions differ, so scores are supporting
diagnostics rather than an exactly matched validation comparison.

The old input had external `asinh(x/s)` followed by SBI's internal per-bin
standardization (confirmed in the saved embedding network). The new input
has train-only external asinh and standardization, with SBI x-standardization
disabled. Thus the old model was not trained on wholly unstandardized x.
The new identity projection includes another affine standardization, which
is approximately identity and does not remove spectral information.

A separate previously confirmed SBI 0.22 bug affected capped PCA/MOPED runs:
the final weights, rather than best-validation weights, were returned at the
epoch limit. The September 9 corrected models explicitly restore the best
weights. New bins40 stopped early and was already restored correctly; that
bug does **not** explain its poor correlation.

## Exactly what each method computes

All methods use the same signed, beamed/masked baseline-deproj0 40-bin D_ell
input and infer all nine parameters. The beam and mask are not applied again.
Rows outside the saved inference prior are excluded, as requested. The last
500 array indices are held out before fitting anything; 495 are in prior.
The remaining 513042 rows split into 461737 optimization and 51305 validation
rows. The same split is used for all methods.

For each bin, fitted on optimization rows only:

```
s = max(lower_median(abs(x)), 1e-30)
u = asinh(x / s)
y = (u - mean(u)) / max(std(u, ddof=1), 1e-8)
```

- **bins40** keeps all 40 y coordinates.
- **PCA** projects y onto the leading nine eigenvectors of its optimization
  covariance. Each projected coordinate is standardized again. These nine
  axes retain 99.9998566% of the variance, not necessarily that fraction of
  parameter information. The bins are strongly redundant. Rotation and
  rescaling make weak directions easier for a finite neural network to use;
  compression does not create physical information.
- **MOPED** uses the nearest 20000 optimization theta vectors to Battaglia12,
  with distances in prior-width units. A quadratic in nine parameters has
  55 coefficients. Fits to clean spectra and noisy-minus-clean residuals
  estimate the transformed noisy mean and its derivatives J. Residual mean
  subtraction is necessary because asinh(noisy)-asinh(clean) need not have
  zero mean. The covariance uses those detrended residuals, with a regression
  degrees-of-freedom correction and 5% shrinkage toward its diagonal.

For C=L L^T and SVD(L^-1 J)=U S V^T, the MOPED weights are B=L^-T U.
The nine retained coordinates are B^T(y-mean_fid), then standardized using
optimization rows. This preserves the *fitted local mean Fisher matrix*;
it does not prove exact global preservation for this nonlinear,
signal-dependent-noise simulator. No separate 64-profile ensemble is used.
The saved warning says the clean quadratic fit error exceeds the paired
noise scatter in some bins. This limits a precision Fisher interpretation.
See the assumptions in [Heavens et al.](https://arxiv.org/abs/astro-ph/9911102).

All three use a trained MAF and prior-restricted direct samples, with 2000
accepted draws per completed observation. No MCMC, clipping or truth-based
posterior correction is used. Posterior mean and sample standard deviation
are evaluated separately for each observation and each parameter.

```
Pearson_j = corr(theta_true[:, j], posterior_mean[:, j])
RMSE_prior_j = sqrt(mean(((posterior_mean[:, j] - truth[:, j]) / prior_width[j])**2))
RMSE_std_j = sqrt(mean(((posterior_mean[:, j] - truth[:, j]) / posterior_std[:, j])**2))
```

The last expression is **not** mean(RMSE)/mean(std). Aggregate RMS pools
the squared normalized errors over observations AND parameters before taking
the square root. Pearson r is not a coverage probability or "95% accuracy".
All three current methods under-cover their nominal 68% intervals: aggregate
coverage on the preliminary 489-profile set is approximately 55%, 56%, 59%.
Sharper MOPED contours alone are not proof of a calibrated posterior.

## Recommended controlled follow-up

Keep the new verified rows, saved prior and held-out split. Do not restore
out-of-prior simulations merely to reproduce the old sample count.
Train all three methods with a common stronger budget, first reproducing
the old MAF settings (50 features, 5 transforms, batch 50, patience 60,
explicit generous epoch cap). Retain best-validation checkpoints and compare
on the same independent test rows. Repeat training seeds before attributing
small differences to compression. If using larger batches for speed, compare
optimizer steps and convergence, not only epochs.

The current code already exposes the necessary options; use a **new output
root**. No retraining has been submitted by this audit. Finite sampling
failures must remain visible; do not silently drop difficult profiles from
a final result. MOPED remains better than the old model on this shared subset,
but the very large gain relative to the new weak bins40 baseline is not a
clean measurement of compression alone.

## Local artifacts and commands

```bash
python SBI_analysis/audit_so_compression_baseline.py
```

The audit saves `baseline_audit/old_vs_compression_matched_rows.png`, PDF,
matched-row CSVs, indices and `audit.json`, alongside the original `summary/`.
The original summary plots are not overwritten.

Open `SBI_analysis/moped_gnfw_local.ipynb` with the `HalfDome (clean)` kernel.
Its portable model bundle is in
`SBI_analysis/outputs/compression_bestval_20260909/moped_bundle/`.
The notebook uses all nine editable parameters with Battaglia12 defaults.
It saves spectra, transformed observation, simulator command/source hashes,
posterior samples, diagnostics and PNG/PDF corner plots in a new
`SBI_analysis/outputs/moped_local_profiles/<request_id>/` directory.

The local model import is tested against 32 cluster log probabilities.
Generation additionally needs the full HalfDome catalogue, SO noise curve,
working Julia/XGPaint environment and adequate RAM. A model smoke test on an
existing training row is not a newly generated profile or held-out validation.

The available local Julia is
`/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia`, with project
`/home/kn18001/.julia/environments/v1.12` and depot `/home/kn18001/.julia`.
The notebook defaults to this tested environment. Its XGPaint constructor
passed all nine pressure-parameter checks. The older `julia_env/Manifest.toml`
in the checkout points at a missing cluster directory, and its bundled
XGPaint does not support all nine editable parameters; it is not used.
No project dependencies or pressure source files were edited.
GetDist 1.7.7 was installed in the existing isolated HalfDome Python
environment; no NumPy, SBI or Torch versions were upgraded.

The actual loaded XGPaint source hashes are recorded for every new simulation.
The historical simulator-library revision used for the original 524288 rows
has not been independently hash-matched: SSH became unavailable during that
additional check. Matching the wrapper/settings and passing a constructor
test are not proof of exact historical numerical equivalence. For a final
calibration claim, compare to the original source snapshot or reproduce a
known training-row signal before trusting new-profile accuracy. Full map
generation was not run here; the current WSL RAM allocation is about 31 GiB.

To export again on the cluster, with a fresh output directory:

```bash
ROOT=/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0_bestval_20260909
DATA=/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow/so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz
python3 SBI_analysis/export_so_moped_bundle.py --root "$ROOT" --dataset "$DATA" --output "$ROOT/moped_bundle_new"
```

The explicit dataset override handles the cluster's `/misc/home` versus
`/home` mount aliases. Export validates theta, IDs, bounds and cached contexts
before packaging; this is a lightweight operation, not another SBI run.
