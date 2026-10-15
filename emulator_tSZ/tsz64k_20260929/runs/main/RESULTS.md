# 64k noiseless tSZ emulator

Trained on the complete 65,536-row design: 49,152 training, 6,554 validation,
9,830 untouched test rows. Selected **log_space_ensemble** using validation
log-spectrum RMSE only. All source rows were verified against the frozen design
and the two archived source pieces. No noisy spectra were used as targets.

## Held-out results

| Metric | Value |
|---|---:|
| Median absolute percentage error | 0.440853% |
| 95th percentile absolute percentage error | 1.92607% |
| 99th percentile absolute percentage error | 3.84168% |
| Fractional RMSE, all test bins | 2.05282% |
| RMSE in log10 D | 0.0064241 dex |
| RMSE in linear D | 1.51127e-08 |
| Mean signed percentage bias | 0.0279521% |
| Worst individual bin error | 250.853% |
| Minimum per-bin linear R-squared | 0.982549 |
| Test-bin predictions within 1% | 81.5585% |
| Test-bin predictions within 5% | 99.4652% |
| Median reference covariance distance | 83.2276 |
| 95th percentile reference covariance distance | 417758 |
| Independent B12 reference covariance distance | 4.56041 |
| Independent B12 fractional RMSE | 0.845366% |

Engineering percentage gate: **PASS**.
Provisional MCMC accuracy gate: **FAIL**.

At the independent Battaglia12 point, the fractional RMSE is
0.845%, but its reference covariance distance
is 4.560. This is a distance in the 40-dimensional
spectrum space, not a measured parameter bias in sigma units. It exceeds the
declared local accuracy budget, so this version is not ready for the intended
MCMC comparison despite its good typical percentage accuracy.

The worst fractional-RMSE test profile is row 57962,
with RMSE 170.504% and reference covariance distance
5.397e-09. It is an extremely faint spectrum;
its large relative discrepancy has negligible absolute size on this reference
noise scale. Both rankings are retained so that this difference is visible.

The latter uses explicit budgets epsilon < 0.1 at B12, < 0.3 at every local
external point, and a 95th-percentile test reference epsilon < 0.3. These are
diagnostic accuracy budgets, not a posterior coverage certificate. The 64-draw
OAS covariance comes from the older fixed B12 signal and is only a reference
scale for the wide prior and updated painter. It excludes cosmic variance,
model discrepancy, and parameter-dependent signal-noise effects. It is not
divided by 64. Large distances at extremely bright prior corners should not be
read as candidate-specific posterior significances.

Linear RMSE and R-squared are dominated by the brightest spectra in this very
wide prior. Fractional RMSE, log RMSE, per-bin and parameter-slice diagnostics
are needed alongside them. Worst-profile panels rank whole-profile fractional
RMSE and reference covariance distance separately; all parameters and row IDs
are in metrics_by_profile.csv. The overall worst individual bin is preserved
in test_metrics.json and the per-profile max_ape_pct column.

## Physics and model

The pressure amplitude P0 multiplies the Compton-y map linearly, while its
power spectrum scales as P0 squared. The network therefore learns
log10(D/P0**2) from xc, beta and the six mass/redshift exponents, using log10
xc and log10 beta as input features. The final prediction restores P0**2 and
returns positive linear 40-bin D_ell = ell(ell+1) C_ell/(2 pi).

Targets are the same masked, 2-arcmin-beam clean spectra as the 64k SBI data,
with mode-count weights over ell=80..7979, and the saved wide nine-parameter
prior. Training uses per-bin standardized log targets and residual MLPs.
All normalizers are fitted only to training rows. Validation selects epochs,
network width, and the optional log-space ensemble; test rows do not select
any of these. Independent B12 and its finite-difference neighbours are used
only as external checks after training, never to tune this run.

## Figures and reusable artifacts

- figures/loss_curves.png: training and validation histories.
- figures/errors_by_multipole.png: percentiles, bias and both RMSE measures.
- figures/worst_fractional_profiles.png: six worst fractional-RMSE spectra.
- figures/worst_covariance_profiles.png: six worst reference-noise distances.
- figures/representative_profiles.png: ranked 0/25/50/75/95/99-percentile cases.
- figures/battaglia12_and_local_checks.png: independent B12 and local checks.
- figures/errors_by_parameter.png: error trends across all nine parameters.
- figures/error_distributions.png and prediction_parity.png: full test errors.

Every figure also has a vector PDF. emulator.pt is the reusable artifact;
model.py provides Emulator.predict(theta). split_indices.npz and provenance.json
record the split and sources. test_predictions.npz stores every test prediction.
The CPU single-profile timing here was 0.652 ms, with four Torch
threads and the selected artifact loaded; this is not an MCMC runtime.

No MCMC has been run. Good percentage performance alone does not establish
that the emulator is accurate enough for your conditional-noise likelihood.
