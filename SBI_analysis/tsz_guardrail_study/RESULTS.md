# Guardrail study: measured results

Updated UTC: 2026-09-20T10:13:18.001807+00:00

**The active 8k run is unchanged. The extended distribution plotted here is an analytic candidate, not a certified production prior.**

Completed full-map experiments: 36/36; failed: 0; pending: 0.

## Main metric

Use the covariance-whitened error in the original 40 clean bandpowers. The reference covariance comes from 64 independent SO split-noise realizations at fixed Battaglia12. It does not include cosmic variance or pressure-model discrepancy. Away from that signal it is a reference precision scale, not the exact candidate posterior uncertainty.

Accuracy targets 0.05, 0.1 and 0.3 correspond to different explicit numerical budgets. At 0.1 the local identifiable Fisher bias is at most 0.1 standard deviations and the Gaussian mean-shift KL divergence is at most 0.005 nats. These implications, assumptions and derivations are in [METHODS.md](METHODS.md).

## Prior volume

Of 262,144 proposals, 2.198% pass the production prior. Within the same rectangle, 66.301% pass the finite-energy condition. In the extended rectangle the analytic acceptance is 67.377%.

Independent quadrature gives analytic acceptance 67.38506%. All 5,761 accepted production-prior test points also pass the analytic candidate. This does not certify the numerical accuracy of the newly admitted points.

![All-parameter prior comparison](plots/prior_comparison_all_parameters.png)

Grey: exact frozen 8k design. Purple: analytic extended candidate. Dashed boundaries: 8k rectangle. Gold: original SBI bounds. Black and colored lines: Battaglia12 and effective FLAMINGO fits.

## Quadrature and LOS endpoint

The joint stress audit contains 2,275 probes, with 0 tolerance failures. The maximum evolved beta tested is 2536.1; the maximum evaluation count is 357. Tightening tolerances changes successful results by at most 1.82e-12 relatively.

Doubling the LOS endpoint changes columns inside 4R200 by at most 0.0444 over these cases. The maximum including unpainted radius 1000R200 is 0.0466. There are 75 correctly recorded floating-point underflows in the successful log-integral tests. These do not by themselves establish map accuracy.

![Quadrature and endpoint diagnostics](plots/quadrature_and_los.png)

## Complete-catalogue interpolation tests

| Case | Historical vs refined log-z, reference sigma | Coarse vs refined log-z, reference sigma | Historical maximum fractional difference |
|---|---:|---:|---:|
| amP0_low_2 | 0.046062 | 4.9236e-06 | 0.20501% |
| amxc_low_2 | 2.1131 | 0.001649 | 0.032478% |
| azbeta_high_2 | 0.0099204 | 1.0941e-06 | 0.039305% |
| azP0_low_2 | 0.010913 | 6.3369e-07 | 0.044188% |
| azxc_high_2 | 0.66575 | 0.02514 | 0.00076888% |
| Battaglia12 | 0.0089114 | 4.9727e-06 | 0.014069% |
| beta_steep | 1.9813e-08 | 1.3326e-10 | 0.001737% |
| combined_tails | 0.002594 | 1.6675e-09 | 15.773% |
| FL_L1_m9 | 0.097377 | 1.6131e-05 | 0.11436% |
| xc_low_1 | 1.5048e-05 | 2.441e-10 | 4.6217% |

These are full 85,224,251-halo maps with the same physical pressure model, beam, mask and painting radius. They isolate interpolation changes while retaining the historical pixel-center painter.

![Full-map interpolation convergence](plots/full_map_interpolation_convergence.png)

## LOS endpoint on occupied catalogue support

The 122,200 additional probes use 876 catalogue-hull vertices and 64 occupied-bin means for each tested parameter vector. The largest L-to-2L change is 2.9295e-06; its next 2L-to-4L change is 2.385e-07.

The much larger 4.44% cache-domain result above occurs at M=1e12 Msun, z=5, outside the occupied catalogue range. These domains must not be conflated. The occupied-support probes are empirical column tests, not a complete-map bound or a guarantee between all tested points.

## Measured resource use

| Operator | Time range [s] | Largest peak RSS [GiB] |
|---|---:|---:|
| historical1 | 308--338 | 10.27 |
| logz1 | 310--332 | 10.25 |
| logz2 | 538--586 | 12.27 |
| pixel8192_2 | 1087--1126 | 19.32 |
| pixel16384_2 | 3347--3350 | 55.85 |

These isolated study processes use eight threads. They measure clean map experiments, including their stated diagnostics; they are not a throughput forecast for the complete noisy 8k production pipeline. Runtime budgets are computational choices, not astrophysical boundaries.

## Pixel sampling

The exact single-halo random-position bound is E[F_pix^2]/F^2 >= max(1, Omega_pix/A_eff), where A_eff=(integral y)^2/integral y^2. It explains rare-hit variance and gives a physically defined resolution diagnostic. It is a bound on flux squared, not on every multipole of a masked catalogue.

![Effective-area bound](plots/effective_area_bound.png)

| Case | 4096 vs 8192 raw sampling, reference sigma | Maximum fractional bandpower difference |
|---|---:|---:|
| Battaglia12 | 9.4099 | 17.502% |
| combined_tails | 0.014624 | 47.149% |
| FL_L1_m9 | 1.8595 | 4.7852% |
| xc_low_1 | 0.0091867 | 136.57% |

Further refinement for Battaglia12: 4096 to 16384 gives 9.7894 reference sigma and maximum bandpower change 18.289%. The 8192 to 16384 step gives 0.38313 reference sigma; maximum relative bandpower change 0.66912%.

Further refinement for FL_L1_m9: 4096 to 16384 gives 1.882 reference sigma and maximum bandpower change 4.8403%. The 8192 to 16384 step gives 0.025997 reference sigma; maximum relative bandpower change 0.052543%.

![Further pixel refinement](plots/full_map_pixel_refinement.png)


As an identifiable one-parameter example, treating the 16384 map as synthetic data and fitting only P0 with the 4096 template gives P0/P0_input=0.991868, or a shift -8.116 in the conditional power-amplitude standard deviation. This is not a nine-parameter posterior result and does not assert that 16384 is the continuum reference.


For this paired full-map test the denser raw map is smoothed with the same harmonic beam and band limit, then synthesized at 4096 before the original mask and binning. Agreement at two resolutions alone is not a proof of the continuum limit.

## Interpretation of the requested tails

All six requested directions have nonzero analytic support in the candidate rectangle. They are not all unconditionally safe to paint with the historical renderer. The numerical gate must depend on the observable error and amplitude, rather than universal xc/beta or Y200/B12 ratios. At fixed other parameters and fixed covariance, the clean-spectrum error scales as P0^2; therefore a measured complete error epsilon_ref implies P0_max=P0_ref sqrt(budget/epsilon_ref).

The new rectangle is P0 [1,60], xc [0.025,4], beta [2.8,16], alpha_m_P0 [-0.6,1.5], alpha_m_xc [-1,0.4], alpha_m_beta [-0.2,0.4], alpha_z_P0 [-6,0.5], alpha_z_xc [-1.5,3], alpha_z_beta [-0.5,2]. Its endpoints define an exploration window; they are not claimed physical singularities.

The old upper Y ratio 30 has no universal pressure-only physical derivation. FIRAS mean-y measurements provide a separate empirical test of total thermal energy; they are not silently inserted into the sampling prior. See the independent mean-y artifacts when available.

## Remaining evidence

A usable production revision requires the point-specific fidelity gate to pass, with adequate reference convergence and the intended covariance. Missing pixel or LOS evidence is an explicit failure of certification, not permission to assume safety. The existing 8k run has not been restarted, changed or relabelled.

Pending planned maps: none.

Failed planned maps: none.

## Independent mean-y diagnostic

These comparisons are not imposed as prior exclusions. The numbers refer to a conservative positive halo contribution evaluated with approximate catalogue histogram integration. They omit additional positive components and are not a measurement of total cosmic y.

| Distribution | Above 5.2e-6 | Above 15e-6 |
|---|---:|---:|
| production (8192 points) | 24.768% | 6.433% |
| analytic_candidate (8192 points) | 46.008% | 32.214% |

![Independent thermal-energy diagnostic](plots/independent_mean_y_check.png)

Across 65 case checks, maximum change on histogram refinement is 0.5839%; maximum radial-quadrature refinement change is 1.554e-11%.

## Numerical identity check

Thirty independent finite-energy checks against 70-digit arithmetic give a maximum normalized quadrature error 7.11e-15. The installed SciPy special-function path has maximum error 1.76e-12; the detailed version-specific comparison is retained in audit/finite_integral_validation.json.
