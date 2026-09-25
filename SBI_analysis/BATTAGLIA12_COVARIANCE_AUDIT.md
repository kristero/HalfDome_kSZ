# Battaglia12 covariance audit, 2026-09-20

The previous Fisher-versus-SBI overlay is not a matched likelihood comparison.
This was rechecked against local source code, saved arrays and artifact hashes.
No new physical simulations or production SBI trainings have run in this audit.
Both Windows and WSL SSH attempts to idark timed out; the availability of the
collaborator's independent-noise data could not be checked remotely.

## What differs in the archived results

| Ingredient | Latest Fisher pilot | Archived 524k SBI |
|---|---|---|
| Noise | 16 disjoint pairs of random split maps at fixed Battaglia12 | Same noise maps reused across parameter rows |
| First bandpower | ell 80–199 | ell 80–279 |
| Priors | Collaborator's new saved bounds | Historical saved inference bounds |
| Observation | Held-out noise root 20001 | Fixed-noise root 12345 |
| Covariance | OAS-shrunk conditional bandpower covariance | NPE learns conditional scatter implicitly |

Legacy MOPED used a local quadratic fit to nearby clean and noisy training
rows, followed by a regularized covariance of detrended paired residuals.
Those residuals across changing parameter values are not independent noise
realizations at fixed parameters. Legacy PCA diagonalized the covariance of
asinh-standardized training spectra across the prior. That covariance includes
signal variation and is not a conditional noise covariance.

The pilot's 16 independent realizations remain a preliminary noise estimate:
OAS correlation shrinkage is 0.969493. The 37 derivative spectra are a different
input: fiducial plus positive/negative steps at two step sizes for nine
parameters. They estimate the response of the mean, not random noise scatter.
The earlier archive labelled 64 adjacent root seeds as independent; the split
seed audit established shared splits. Only the corrected disjoint 16-pair pilot
is used here.

## Required common calculation

Let x be the 40 signed masked pseudo-D_ell bandpowers, mu(theta) their mean,
J = dmu/dtheta at Battaglia12, and C = Cov(x | theta_B12), with the same fixed
sky, beam and mask. A linear compression y = A^T x has

```
J_y = A^T J
C_y = A^T C A
F_y = J_y^T C_y^-1 J_y
score_y = J_y^T C_y^-1 A^T (x_obs - mu_B12)
```

The matrices have different dimensions, but derive from ONE conditional C.
Normalization after compression belongs in A. Re-estimating OAS separately in
each compressed space can change the statistical model and is deliberately
avoided. PCA's training covariance chooses A; it is never substituted for C_y.

The new shared helper `so_fisher_compression.py` computes this propagation,
checks that compression cannot increase mean Fisher information, and saves
the projected covariance, derivatives, Fisher matrix and observed score.
For MOPED, both F and score agree with the full data; therefore the local
Gaussian likelihood ratios, and hence posteriors with identical priors, agree.
PCA is allowed to lose information and gets its own projected Fisher posterior.
The MOPED identity is local with fixed covariance, as in
[Heavens, Jimenez and Lahav (2000)](https://arxiv.org/abs/astro-ph/9911102).

All nine parameters are varied in the order
`P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc,
alpha_z_beta`. The fiducial is
`[18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]`.
Each normalization or shape coefficient evolves as
`q(M,z) = A_q (M200c / 1e14 Msun)^alpha_m,q (1+z)^alpha_z,q`.
The 37 clean spectra give central finite differences at 1% and 2% of prior
width, combined by Richardson extrapolation. The local Gaussian likelihood
is multiplied by the original hard box prior. Importance integration removes
its proposal penalty; weak parameter directions are not assigned artificial
finite errors by flooring Fisher eigenvalues.

## Physical limits of the Fisher approximation

For independent splits the cross spectrum contains signal-signal,
signal-noise and noise-noise terms. On an ideal full sky with equal per-split
noise N_ell, its Gaussian variance is proportional to
`2 S_ell^2 + 2 S_ell N_ell + N_ell^2` if sky realizations are also randomized.
Conditioning on the same deterministic signal removes the `2 S_ell^2`
changing-sky term, but leaves signal-dependent noise scatter. The actual
pipeline estimates the masked bandpower covariance from maps instead of
applying this illustrative full-sky formula or an extra f_sky factor.

Freezing C at Battaglia12 gives the mean-information Fisher `J^T C^-1 J`.
A parameter-dependent Gaussian likelihood also contains
`0.5 Tr(C^-1 C_,i C^-1 C_,j)`, which is not computed by the available clean
derivatives. The prepared independent-noise NPE can learn changing covariance
and nonlinear mean responses. Shared physical inputs therefore make the
comparison meaningful, but do not promise identical Fisher and SBI contours.
The saved forecast is conditional on the fixed halo lightcone and mask;
changing-sky variance is not included.

## Battaglia12 plots available now

The output directory is `outputs/battaglia12_covariance_audit_20260920/`:

- `battaglia12_sbi_corner.png` and `.pdf`: 40-bin, 40-bin MOPED and unbinned
  MOPED, each using the existing 524k density estimator and 10,000 posterior
  draws for exactly the same fixed-noise Battaglia12 observation.
- `battaglia12_sbi_intervals.png` and `.pdf`: medians with 68% thick bars and
  95% thin bars. Dotted vertical lines show Battaglia12.
- `battaglia12_intervals.csv`: all nine parameter summaries.
- `audit.json`: checked source hashes, noise/bin/prior mismatch and numerical
  information-preservation checks. `complete.json` hashes generated artifacts.

These are archived fixed-noise SBI diagnostics, not new independent-noise
results. PCA's saved estimator is not in the local export and could not be
retrieved from the cluster, so no PCA posterior is plotted here. The updated
pipeline includes PCA for future matched training. All prior plots and models
were preserved.

On the real 16-pair pilot arrays, the newly recomputed MOPED/full Fisher
relative difference is 2.44e-16 and the observed-score difference is 4.37e-16.
This verifies compression algebra; it does not establish covariance convergence
or agreement with the fixed-noise NPE.

## Code changes and validation

Added `so_fisher_compression.py` and `plot_battaglia12_covariance_audit.py`.
Updated `run_so_nine_independent_comparison.py` to prepare/train/evaluate PCA,
save per-method likelihood arrays and compare the PCA Fisher posterior with
PCA SBI. The new PCA uses standardized raw D_ell so its projection is linear;
the legacy asinh-PCA model is a different transform and is not reused.
Updated `submit_so_nine_independent_comparison.py` to submit the methods recorded
in the experiment; `run_so_nine_independent_stage.pbs` restores access to the
cluster's user-site Torch/SBI installation. Updated
`test_so_nine_independent_comparison.py` and `SO_NINE_INDEPENDENT_FISHER.md`.
This report is also new.

Twelve numerical/contract tests pass; they include disjoint seeds, held-out
observation exclusion, train-only preprocessing, conditional covariance
propagation, MOPED likelihood-ratio invariance, PCA information loss and hard
prior behavior. A separate synthetic workflow is under
`outputs/battaglia12_matched_software_test_20260920/`; preparation, all three
MAF trainings with best-validation weights restored, all eight held-out
observations and the fiducial observation per method, hard-prior Fisher
integration and all summary plots completed. Its two-epoch cap is a software
test, not scientific validation or converged training. The local environment
uses SBI 0.27.0; the cluster's SBI 0.22 environment was not re-exercised because
SSH was unavailable. The PBS script passed shell syntax checking and the
submission dry run produced three method jobs and a dependent summary job.
The real-data diagnostic corner and interval plots were visually inspected.

To finish the physical comparison, supply the collaborator's completed
`nine_param/prepared/dataset.npz` and generation `experiment.json`, restore
cluster access, and follow `SO_NINE_INDEPENDENT_FISHER.md`. The manifest checks
must pass before calibration or training; fixed-noise models cannot be made
independent-noise models by attaching a new covariance afterward.
