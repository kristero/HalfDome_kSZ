# Scale and sky robustness of the SO Battaglia12 posterior

The primary inference uses the unbounded, linear-parameter ensemble. The baseline is simulation 100 with training mask 12345. Every observation uses the same two split-noise draws of the single SO deproj0 baseline prescription, with the trained cross-spectrum and Gaussian-beam correction.

Winners are selected using 6 simulations × 5 masks only. Validation uses 4 unused simulations at the baseline mask, 5 unused masks on simulation 100, and the unused 4 × 5 joint grid. The full 10 × 10 results are descriptive and include discovery cases.

| Test | Selected method | Held-out truth RMS | Held-out scatter | Width ratio |
|---|---|---:|---:|---:|
| simulation | merlin_band50_full | 1.169 | 0.050 | 1.247 |
| mask | merlin_band50_full | 1.162 | 0.098 | 1.179 |
| combined | merlin_band50_full | 1.162 | 0.097 | 1.226 |

Truth displacement and scatter use the original baseline 68% half width as a fixed unit. Width ratio is the geometric mean of the selected/original 68% interval widths for the same skies and parameters. The per-parameter table must be checked: an average can hide a poorly recovered parameter. Truth inclusion on these correlated skies is descriptive, not repeated-noise frequentist coverage.

The Pareto front reports the tradeoff without requiring one preferred cost function. The selected compromise adds the truth/scatter norm to log(width ratio)/log(1.25), with no reward for narrowing. Alternative winners under 10%, 25%, and 50% width budgets appear in selection.json. A broad prior-coverage screen is a diagnostic, not certification.

## Physics and limitations

The thermal SZ spectrum depends on the integrated electron pressure. Changing the simulation changes halo abundances and structure; changing a large sky cap changes which rare, bright halos enter and also changes mode coupling. Crossed masks and simulations can interact. combined_variance shows the exact balanced decomposition of posterior-median variation into simulation, mask, and interaction terms. Overlapping masks are spatial variations, not independent sky realizations.

Cuts remove complete 50-multipole bands and require retraining. Full-range band averaging retains every multipole but removes fine spectral structure. MOPED projects nine parameter derivatives through a chosen covariance. Adding a discovery-sky covariance downweights directions sensitive to sky variation; a smooth high-ell covariance inflation reduces small-scale weight without discarding a range. Local Fisher preservation concerns the chosen covariance and binned inputs only; it does not guarantee a globally calibrated SBI posterior. See [MOPED](https://arxiv.org/abs/astro-ph/9911102) and [nuisance-hardened compression](https://arxiv.org/abs/1903.01473).

Changing the covariance used to construct weights changes the projection only: the unaugmented SBI candidates retain the original fixed-sky training distribution. Their posteriors do not automatically marginalize over new-sky uncertainty. The augmented candidate is the one that explicitly introduces approximate sky variation into training.

The augmented candidate adds discovery B12 signal-ratio templates across the parameter prior. This is an approximation: it assumes those relative variations transfer across pressure parameters and leaves the SO noise residual unchanged. Real unused B12 skies and a fixed-sky prior test check two aspects of this approximation; neither validates arbitrary cosmologies. The unbounded flow is trained inside the original prior only. Outside-prior mass is reported per case and parameter and is not calibrated extrapolation.

compression_scale_response plots a signed cumulative derivative-response norm. It can be non-monotonic and must not be interpreted as an additive Fisher-information fraction.

All non-winning candidates have 12,000 posterior samples per observation; original and selected candidates have 90,000 samples when final=true. prior_validation.csv uses 1,500 unused parameter rows with 600 samples each. input_sha256.json freezes sources; COMPLETE.json hashes the delivered report. No claim of bias removal is warranted if held-out truth or agreement fails to improve, or if the width cost is unacceptable.

The final ranking also screens joint random-reference distance ranks at the 68% and 95% levels, with the same 0.10 and 0.06 tolerances. The user-confirmed final selection applies a hard 25% cap on the geometric mean of the case- and parameter-wise 68% width ratios. Among qualified candidates that improve BOTH truth displacement and sky agreement on discovery skies, it minimizes their quadrature discrepancy within this budget. The preliminary soft width-penalty score is superseded for this final report. If none qualifies within the budget, the original method is retained and explicitly flagged as no qualifying improvement in selection.json. The worst individual-parameter width increase is reported separately. These two finite diagnostics do not establish accuracy over every parameter or sky.
