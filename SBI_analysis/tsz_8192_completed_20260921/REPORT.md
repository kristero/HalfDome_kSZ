## Completed tSZ validation

2026-09-21 12:45 UTC

All 86 recovery controls and both additional matched-noise controls finished with successful saved exit statuses. Both cluster analyses completed. Raw NSIDE8192 remains the appropriate candidate for the full-range diagnostic; NSIDE4096 is not equivalent near Battaglia12. NSIDE16384 remains an accuracy reference only.

This is numerical validation on fixed HalfDome skies, not a calibration of a nine-parameter SBI posterior. No 256-row generation or training job has been submitted. A separate 8192 preparation now preserves the existing 256 parameter rows and their independent noise seeds.

| Verified evidence | Result |
| --- | --- |
| Completed controls | 88 / 88; no failed controls |
| Source identities | 31 frozen source/input hashes verified |
| Spectra | 202 checked locally; 1044 noise spectra checked on idark |
| Noise identity | 1064 unique split seeds and pixel hashes; one common mask |
| Six independent repeat maps | All pass; largest relative D-ell difference 7.71e-16 |
| Local reproduction of cluster analysis | Maximum relative result difference 2.20e-12 |

The main completed result: at ell=80..7979, the local spherical-model MOPED measures 4096-to-8192 shifts of 9.21 conditional noise units at Battaglia12 and 1.58 at the historical FLAMINGO-fit anchor. The old frozen compressor gives 4.16 and 0.83. A compressor can discard a numerical difference; a smaller old-compressor shift does not establish accuracy of a new inference model.

Using 16384 as a reference, the corresponding 8192 shifts are 0.378 and 0.0204. Therefore 8192 is much closer to the reference, but the B12 full-range result does not meet an illustrative 0.1-noise-unit accuracy target. This target is a declared numerical budget, not a physical constant, and 16384 is not the exact continuum.

The scheduler queue was empty when checked. PBS historical job queries did not provide retained exit records; completion is established from all saved task statuses, complete analysis products, hashes and reproduced calculations.

## Resolution and multipole cutoff

Unbinned individual D-ell inputs throughout. The local MOPED weights are recomputed at each cutoff. The old-compressor curves restrict its existing weights. The dotted 0.1 line is an illustrative accuracy budget.

![resolution_moped](plots/resolution_moped.png)

| ell max | 4096/8192 B12 | 4096/8192 FL fit | 8192/16384 B12 |
| --- | --- | --- | --- |
| 1000 | 0.229 | 0.032 | 0.0071 |
| 1500 | 0.378 | 0.064 | 0.0122 |
| 2000 | 0.824 | 0.110 | 0.0214 |
| 3000 | 2.843 | 0.528 | 0.0796 |
| 4000 | 6.325 | 1.277 | 0.2417 |
| 5000 | 8.650 | 1.499 | 0.3566 |
| 6000 | 9.273 | 1.582 | 0.3816 |
| 7979 | 9.206 | 1.577 | 0.3781 |

At ell<=2000, 4096 is less discrepant (0.824 at B12 and 0.110 at the FL fit), but it is not automatically below a stringent numerical budget. At ell<=3000, 8192 is within 0.080 of the B12 reference. The full-range B12 discrepancy develops mainly between ell=2000 and 6000.

The 2 arcmin beam transmits power fractions 0.783, 0.376, 0.111 and 0.0205 at ell=2000, 4000, 6000 and 7979. Intermediate scales still matter. The pressure column is integrated before painting; the unresolved step is sampling that projected profile at pixel centres. Beam smoothing after incorrect sampling cannot restore the true halo flux.

All three raw resolutions are beam-smoothed and synthesized to the same output NSIDE4096 before the same mask. Raw painting resolution and the resolution of the final band-limited output map are different choices.

## Information retained and derivative reliability

Left: trace of the prior-width-scaled, regularized local Fisher matrix relative to the full range. Right: minimum retained information over its five default full-range directions. These are local diagnostics, not fractions of total nonlinear nine-parameter information.

![information_cutoffs](plots/information_cutoffs.png)

At ell<=2000, the trace retains 27.8% at B12 and 22.8% at the FL fit, while the least-retained direction keeps only 0.394% and 0.136%. Lowering ellmax can therefore lose a parameter combination much more severely than its trace suggests. At ell<=6000, trace retention is 99.86% and 99.72%, and every default retained direction keeps at least 97.9% and 97.2%. A 6000 cutoff is worth testing in SBI; it has not been silently adopted.

Central differences use 0.5% and 0.25% of each prior width. The P0 derivative is analytic. Singular values are shown relative to the largest; a relative threshold is a numerical choice, not a count of independently measured physical parameters.

![derivatives_and_modes](plots/derivatives_and_modes.png)

Largest step-halving changes are 0.128% at B12 and 0.0589% at the FL fit. At full range, retaining 5, 8 or 9 local directions changes the B12 resolution distance only from 9.205 to 9.295 and the FL distance from 1.577 to 1.611. This supports the resolution conclusion without claiming all nine parameters are well constrained.

Covariance estimation uses 64 independent SO noise draws and validates compressed scatter on another 64. With 7900 inputs the fitted OAS covariance is strongly regularized: about 98% toward a diagonal in standardized coordinates. Information fractions depend on that approximation. Distances use single-observation scatter, not the uncertainty on the mean of 64 mocks. Cosmic variance, foreground residuals and model discrepancy are excluded.

## Extreme profiles and interpolation

Each extreme now uses its own 128 paired noise realizations and the frozen unbinned compressor. The two curves show covariance-half sensitivity; the actual signal and corresponding signal-noise terms are included.

![extreme_noise](plots/extreme_noise.png)

| Extreme | 4096/8192 relative spectrum norm | Full-range compressed shift |
| --- | --- | --- |
| Compact / faint | 266% | 0.0000457 noise units |
| Extended / bright | 0.0203% | 30.70 noise units |

The compact test is extremely faint: a large percentage error has negligible weight at this noise level. It must not be rejected solely because its core is compact or its percentage error is large. Conversely, the bright shallow model is sufficiently precise that a small relative spectrum change matters. Its alternate covariance half gives 30.22 noise units, so the large discrepancy is not caused by choosing one covariance half.

These compare 4096 with 8192. They do not establish the residual error of 8192 for either extreme. There is no completed 16384 or full-catalogue pixel-integrated reference for the bright shallow case in this batch. Compactness alone and a universal Y200 ratio cannot replace observable-level accuracy tests.

| Interpolation refinement at fixed 8192 | Relative D-ell norm change |
| --- | --- |
| Battaglia12 | 1.88e-7 |
| FLAMINGO-fit anchor | 3.41e-7 |
| Compact | 4.62e-8 |
| Extended / shallow | 5.44e-6 |

The default cache is 512 x 256 x 128 in log(theta), log(z), log10(M). Doubling every axis changes these four spectra little compared with their raw-pixel-resolution changes. This supports retaining the default grid for the diagnostic. The very bright case still needs noise-weighted cache-error assessment; a small percentage alone is not a precision guarantee.

All pressure profiles retain finite support inside 4 R200. This is a physical model definition. The positive cache floor and normalized, bounded LOS quadrature are numerical measures, and are not pressure-amplitude renormalizations.

## Measured performance and useful optimizations

Matched signal-preparation phases only: cache, allocation, catalogue painting, harmonic transform, beam and synthesis. Full job wall times differ in their noise workload. These are single measurements, not a controlled throughput scaling curve.

![performance](plots/performance.png)

| Configuration | Signal time | CPU hours |
| --- | --- | --- |
| Standard allocation, 26 threads | 17.09 min | 7.41 |
| Lean allocation, 26 threads | 15.49 min | 6.71 |
| Lean allocation, 13 threads | 20.35 min | 4.41 |

The two lean controls reproduce the standard clean spectrum to better than 7e-16 in relative D-ell norm. Their complete clean-job walltimes are 16.52 and 21.23 minutes; peak RSS is 14.83 and 14.75 GiB. Standard clean derivative controls are around 21 GiB. Removing the unused raw temporary map eliminates a 6 GiB allocation at 8192. Noise generation adds its own memory, so the clean-job RSS is not a full noisy-row memory guarantee.

Thirteen threads use 34.3% fewer signal CPU-hours than the lean 26-thread test. Two concurrent 13-thread workers could therefore improve maps per allocated CPU, if memory, bandwidth and the scheduler permit. This concurrency has not been benchmarked on one node; no 1.5x throughput gain is claimed as measured. The prepared package retains the tested default of 26 threads.

Catalogue reading and painting still dominate (893 seconds in the lean 26-thread run); cache construction takes 12 seconds. The next useful work is reusing catalogue geometry, reducing repeated coordinate calculations and measuring painting lock contention. Increasing cache resolution or optimizing scalar integration alone does not address the main walltime cost.

For orientation, 256 rows at the lean clean-job time imply about 70.5 worker-hours, or 17.6 hours with four continuously running workers, before noise, training, queue delay and parameter-dependent runtime. This is an extrapolation, not a deadline promise. The prepared four-worker 23:59 jobs are a diagnostic scheduling choice; failed or unfinished rows remain visible.

## Prepared 256-row run and remaining limits

A fresh, unsubmitted package now uses raw NSIDE8192, output NSIDE4096, the same 2 arcmin beam and fsky=0.4 mask. The existing 256 theta rows, 512 split seeds, 192/64 training/held-out division, and three processed FLAMINGO observations are copied unchanged. MOPED reads ell=80..7979 directly, without 40-bin input compression.

The two numerical projection files are byte-identical to the validated campaign. The allocation block is copied from the successful lean control. The launcher replaces node-local flock with atomic directory claims and disjoint row ownership; only one collector runs after all workers. It verifies frozen input hashes, preserves failed row identities and refuses training on an incomplete successful subset. Launcher tests use mocked subprocesses; no new full catalogue row has been generated by this preparation.

| Independent physical uniforms | P0 | xc | beta |
| --- | --- | --- | --- |
| Amplitude | 1 to 60 | 0.025 to 4 | 2.8 to 16 |
| Mass exponent | -0.6 to 1.5 | -1 to 0.4 | -0.2 to 0.4 |
| Redshift exponent | -6 to 0.5 | -1.5 to 3 | -0.5 to 2 |

The completed tests support proceeding with an explicitly diagnostic 8192 dataset after authorization. They do not certify a larger science release. Remaining scientific work: full-catalogue pre-beam or pixel-integrated painting; an 8192 accuracy reference for bright/extreme rows; broader prior fidelity checks; and held-out SBI calibration including shuffled-label and prior-only baselines. The earlier sparse-halo pre-beam and CLASS-SZ comparisons are not new completed full-catalogue tests in this batch.

Historical guardrails: the earlier two 1200-second pilot timeouts were real. Fast central-column probes did not contradict slow underflow-tail cache nodes. Normalized integration and the safe positive cache floor address those numerical failures; beta<=50 and the old relative-size/Y200 envelopes were conservative cuts, not proven physical boundaries. The finite sphere also changes the untruncated-energy assumption. The extended prior must be assessed with the repaired operator, not by dropping all checks from original XGPaint.

Files added: fetch_completed.py, audit_completed.py, verify_analysis.py, plot_completed.py, build_report.py, prepare_8192_diagnostic.py and test_prepared.py, plus evidence JSON, figures and report. The separate tsz_diagnostic_256_8192_prepared_20260921 directory contains the revised launcher, PBS scripts, frozen numerical files, preserved designs and observations. No cluster dataset or SBI training submission was made.

Evidence roots: SBI_analysis/tsz_8192_completed_20260921; tsz_8192_validation_recovery_20260921; tsz_8192_noise_extremes_20260921. Historical failure provenance: tsz_8192_results_20260921/GUARDRAIL_PROVENANCE.md. The earlier report is retained as a dated partial-results snapshot.

