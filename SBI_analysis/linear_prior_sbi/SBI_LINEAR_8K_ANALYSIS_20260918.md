# SBI on the broad linear-prior 8192-row HalfDome tSZ dataset

Date: 2026-09-18. All computation on idark under `/lustre/work/kristero10/sbi_linear_8k_20260918/`
(code in `code/`, results in `analysis/`, figures in `analysis/figures/`); only figures, tables and JSON
summaries were copied to `SBI_analysis/linear_prior_sbi/cluster_results/`. Dataset:
`/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915` (see `flamingo_linear_prior/cluster_results/completed_20260918/`).

## 1. Independent noise per row: verified

`code/verify_independent_noise.py` (`cluster_results/noise_independence_check.{png,json}`):

- `dataset/noise_split_seeds.npy` equals the SHA256 recipe `halfdome-so-v1|12345|ROW_ID|SPLIT` (first 8 bytes,
  big-endian, >> 1) for every stored row ID; all 16384 seeds distinct; observation seeds 22446/22447 absent.
- Every packed row's provenance (`status_json.noise`) carries the design seeds for its row (0 mismatches),
  split 1 != split 2 in every row, and the 16384 noise-map pixel SHA256s are all distinct (each row received its
  own two noise maps); one mask hash. 23 rows carry the imported-spectrum provenance of the retired run (same seeds).
- Statistics: the row mean of (noisy cross - clean) per bin is consistent with zero (max |z| = 2.2 over 40 bins);
  the correlation between residual vectors of random row pairs in the noise-dominated bins is -0.006 +- 0.36.
  A shared noise realization would give a common n1 n2 term (|z| >> 1 at high ell, correlation near 1).

## 2. Setup of the NPE analysis (`code/sbi_linear_prior_pipeline.py`, sbi 0.22, CPU)

- Test set: the 843 rows of the dataset's validation split. Training pool: the other 7349 rows, shuffled once
  (seed 20260918); nested training sizes N = 256, 512, 1024, 2048, 4096, 7349. 10% of each N is sbi's
  early-stopping validation set; compressions use only the remaining 90% (optimization rows).
- Observable: the 40-bin masked noisy split-cross D_ell; coordinates y = standardized asinh(x/s) as in the earlier
  SO compression work. Methods: `bins40` (all 40 coordinates), `pca` (9 leading components of the optimization
  covariance, 99.96% of the variance), `moped` (noise-whitened local mean-sensitivity directions around
  Battaglia12 from the paired clean/noisy rows; 9 components at every N; quadratic local mean model for
  N >= 1024, linear for N = 256 and 512; local rows = min(2048, optimization rows)).
- Prior: the joint conditional prior of the dataset (`code/sbi_prior.py: ExtendedPrior`), used both for NPE
  and for exact rejection of posterior samples outside the support. No BoxUniform.
- Density estimators: MAF and NSF (5 transforms, 50 hidden features, NSF 8 bins); batch 32 (N <= 512), 64
  (N <= 2048), 100 otherwise; learning rate 5e-4; early stopping after 80 epochs without validation improvement
  (cap 4000); best-validation weights kept. Per (method, N) the estimator with the higher validation log-prob
  is used for the summary and corner plots: MAF won at every point (NSF is drawn as dashed lines).
- Metrics on the 843 test rows: posterior mean from 1000 accepted samples; RMSE of the posterior mean divided
  by the prior range, per parameter and averaged; Pearson r(truth, posterior mean) per parameter and pooled over
  the nine standardized parameters; bootstrap standard errors. Median rejection acceptance 0.42-0.68.
- Compute: three PBS jobs (one per method), about 2.5 h each on 26 CPUs (training 12 estimators each, 190-1200 s
  per estimator).

## 3. Convergence with dataset size (`figures/convergence_rmse.{png,pdf}`, `convergence_pearson.{png,pdf}`, `convergence_metrics.csv`)

Mean RMSE / prior range (pooled Pearson r), MAF:

| N | 40 bins | PCA (9) | MOPED |
|---|---|---|---|
| 256 | 0.201 (0.54) | 0.203 (0.52) | 0.191 (0.59) |
| 1024 | 0.198 (0.56) | 0.197 (0.57) | 0.186 (0.62) |
| 4096 | 0.187 (0.62) | 0.191 (0.60) | 0.172 (0.69) |
| 7349 | 0.182 (0.65) | 0.188 (0.62) | 0.167 (0.71) |

- MOPED is the best compression at every N and improves fastest; 40 bins beats PCA(9) from N ~ 2048 on;
  PCA discards informative low-variance directions. None of the curves has flattened at N = 7349: RMSE
  still falls by 3-4% per doubling, so a larger training set would still help.
- Best-constrained parameters at N = 7349 (MOPED RMSE / range, r): alpha_m(P0) 0.12 (0.84), alpha_z(P0) 0.16
  (0.78), P0 0.21 (0.68), xc 0.19 (0.60), beta 0.20 (0.55). Weakest: alpha_m(beta) 0.17 (0.39),
  alpha_z(beta) 0.16 (0.42): the mass and redshift dependence of the outer slope is barely recovered from the
  power spectrum with SO noise. For reference, a posterior mean equal to the prior mean gives RMSE / range
  of roughly 0.2-0.3 for these marginals, so values near 0.2 mean weak constraints.

## 4. Posteriors for the observations (`figures/corner_*.{png,pdf}`, `corner_posterior_summary.json`)

N = 7349, MAF, 10000 accepted samples per method; observations use the fixed SO noise splits 22446/22447 of the
FLAMINGO comparison (independent of every training-noise seed). The Lee22 no-c observation was produced on the
cluster by the same operator (`code/process_fits_map.jl`, job 598123) from the uploaded Lee22 electron-pressure
Compton-y map (h = 0.68, 4 R200c projected aperture, all z), binned identically.

- Battaglia12 (true values known): all three methods contain the truth, but the constraints are weak and
  prior-shaped for most parameters; xc and beta sit on a strong degeneracy (xc-beta panel) whose low end holds
  the truth, so their 1-D marginals peak well above the true 0.50 and 4.35 (posterior means 1.4-1.6 and 9.5-10.9).
  MOPED gives the tightest P0 (16 +- 9 against 18.1); 40 bins and PCA 20-24 +- 12-13.
- Lee22 no-c (outside the family, no reference values): P0 ~ 10-20, beta ~ 10, xc ~ 1.5, and a strongly
  negative alpha_z(P0) ~ -2.7, i.e. the fit absorbs Lee22's steep redshift decline of the pressure normalization.
- FLAMINGO L1_m9, fgas-8sigma, Mstar-1sigma: dashed lines are the effective clean-spectrum fits from the prior
  preparation (not truths); the posteriors of the three methods are mutually consistent and the fits lie inside
  the 68-95% regions.

These are NPE posteriors without coverage calibration (no SBC yet); treat widths as indicative.

## 5. Files

Local: `sbi_linear_prior_pipeline.py`, `verify_independent_noise.py`, `process_fits_map.jl`, `run_lee22_observation.py`,
`run_sbi_stage.pbs`, `run_lee22_observation.pbs`, `so_sbi_compression.py` (copy), `cluster_results/`.
Cluster job IDs: 598123 (Lee22 observation), 598124/598125/598126 (bins40/pca/moped).
