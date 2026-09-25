# FLAMINGO pressure-prior pilot: completed results

**The full independent rectangular prior is not ready for a larger production run.** It contains divergent or cutoff-sensitive pressure combinations, extreme thermal-content excursions, and compact profiles that the current pixel sampling cannot resolve. The fitted spectra below establish what this pilot actually reproduces.

All numerical science calculations and full maps were run on the cluster. Clean spectra were fitted; the identical cached SO noise was compared afterward. The original HalfDome operators, old inference bounds, MOPED weights and trained SBI bundle were preserved.

[All nine plot sets in one vector PDF](plots/tsz_prior_pilot_plots.pdf) | [Parameter table](parameters.csv) | [Clean/noisy spectra](fit_spectra.csv)

## Full-map fit quality

| FLAMINGO target | RMS fractional residual | Maximum bin residual | 2% RMS / 5% maximum target |
|---|---:|---:|---|
| L1_m9 | 2.310% | 7.575% | not met |
| fgas-8sigma | 1.899% | 5.884% | not met |
| Mstar-1sigma | 1.753% | 6.593% | not met |

These values use repainted Nside4096 spectra over all 40 bins (ell 80–7979). RMS is sqrt(mean((fit/target-1)^2)); the optimization objective uses log residuals. They are effective matches on the HalfDome catalogue, not unique FLAMINGO feedback measurements or SBI posteriors.

The supplied FLAMINGO L1 maps are lensed and integrated to z=3 ([product documentation](https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/integrated_lightcones.html#integrated-thermal-sz-maps)). The painted HalfDome catalogue extends to z=3.855428. Cosmology, halo/gas modelling, lensing and realization differences remain in this comparison. The fitted pressure evolution can absorb these differences, so its redshift exponents should not be interpreted as isolated feedback measurements.

![Clean fits](plots/01_clean_flamingo_fits.png)

Scale dependence of the same fits (no refitting or removal of bins):

| Target | RMS for ell>=500 | Maximum for ell>=500 | Worst full-range bin centre |
|---|---:|---:|---:|
| L1_m9 | 1.874% | 6.024% | 198.02 |
| fgas-8sigma | 1.441% | 5.602% | 388.27 |
| Mstar-1sigma | 1.085% | 4.841% | 198.02 |

Additional ell>=1000 and ell>=2000 metrics are saved in audit/fit_scale_metrics.json. These are descriptive residuals; the pilot does not have a covariance-weighted goodness-of-fit or a multi-realization test assigning the residuals uniquely to feedback or sample variance.

## Fitted parameters and tested ranges

| Parameter | Lower | Upper | Battaglia12 | L1_m9 fit | fgas-8sigma fit | Mstar-1sigma fit |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 1 | 60 | 18.1 | 19.8639 | 15.709 | 34.1998 |
| xc | 0.1 | 4 | 0.497 | 1.37885 | 1.37728 | 1.37051 |
| beta | 2.8 | 16 | 4.35 | 7.27354 | 7.00167 | 8.66182 |
| alpha_m_P0 | -0.2 | 1.5 | 0.154 | 0.00403154 | 0.228737 | 0.21116 |
| alpha_m_xc | -0.6 | 0.4 | -0.00865 | -0.370845 | -0.212388 | -0.345442 |
| alpha_m_beta | -0.2 | 0.4 | 0.0393 | -0.0100063 | 0.0909572 | -0.00778554 |
| alpha_z_P0 | -4.5 | 0.5 | -0.758 | -3.2032 | -3.24449 | -3.83123 |
| alpha_z_xc | -1.5 | 2 | 0.731 | 1.27151 | 1.46324 | 1.57943 |
| alpha_z_beta | -0.5 | 1.5 | 0.415 | 0.712278 | 0.773872 | 0.833874 |

P0 and xc were drawn uniformly in their logarithms; the other coordinates were drawn uniformly. Alpha=1 and gamma=-0.3 remain fixed. CSV/JSON files preserve full precision.

- L1_m9: min beta on the interpolation grid = 6.999; within 2% of a prior edge in unit coordinates: none.
- fgas-8sigma: min beta on the interpolation grid = 4.609; within 2% of a prior edge in unit coordinates: none.
- Mstar-1sigma: min beta on the interpolation grid = 8.409; within 2% of a prior edge in unit coordinates: none.

A weak 0.005 unit-cube regularizer selects representatives among spectral degeneracies. Some proposal optimizers stop at their evaluation limit; the measured full-map residual, not that stopping flag, determines the fit-quality label. A boundary fit does not establish that its parameter is constrained.

## tSZ extremes and combinations

![tSZ range](plots/02_tsz_with_extremes.png)

| Explicit full-map stress test | Outcome | Minimum / maximum clean-power ratio to Battaglia12 |
|---|---|---|
| extreme_xc_high | full_map_completed | 155.84 / 19609 |
| extreme_beta_low | full_map_completed | 3.2613 / 21.803 |
| extreme_beta_high | full_map_completed | 0.00033384 / 0.01262 |
| extreme_wide_steep | full_map_completed | 17.909 / 33.652 |
| extreme_compact_steep_evolving | failed | No accepted full-map spectrum; inspect its recorded run status/log |
| extreme_mass_redshift_amplitude | full_map_completed | 4.5875 / 455.6 |

P0=1 and P0=60 are exact amplitude rescalings at fixed other parameters: y scales as P0 and clean power as P0 squared. Their power ratios are 0.00305241 and 10.9887. Shape/evolution examples use new full maps when they complete; failed examples remain explicit.

## Whole-prior audit

| Condition | Interpolation grid: 65,536 draws | Actual catalogue: 65,536 draws |
|---|---:|---:|
| Divergent untruncated LOS | 1285 (1.9608%) | 32 (0.0488%) |
| Divergent infinite-volume pressure integral | 22073 (33.6807%) | 10268 (15.6677%) |

4419 draws (6.743%) lose more than 1% of the central column beyond the current LOS endpoint somewhere on the diagnostic grid. The finite Y200 ratio across sampled points spans 8.956e-19 to 4.508e+05. These extreme ratios are diagnostics, not measured exclusion thresholds.

All 512 corners were also evaluated: 192 have divergent LOS columns on the interpolation grid and 64 at actual catalogue support. The 18 single edges and 144 pair edges are summarized separately.

![Prior support](plots/04_prior_support_and_conditions.png)

![Edges and pairs](plots/05_edges_and_combinations.png)

Concrete combinations behind the range:

| Example | M200c [Msun] | z | Y200 / Battaglia12 | Passes optional slope screen |
|---|---:|---:|---:|---|
| sobol_largest_Y200 | 5.0119e+15 | 5.0 | 450815 | True |
| sobol_smallest_Y200 | 1e+12 | 5.0 | 8.95573e-19 | True |
| corners_largest_admitted_Y200 | 5.0119e+15 | 5.0 | 815888 | True |

All nine coordinates for these cases are in [extreme_parameter_examples.json](audit/extreme_parameter_examples.json). The largest admitted corner combines high P0 and xc with mass/redshift growth of both, while decreasing beta toward high mass/redshift. It passes outer-slope convergence yet has an enormous finite pressure integral. This demonstrates why the slope screen alone is insufficient.

## Physics of the limits

The model has P_th/P200 = P0 (x/xc)^(-0.3) (1+x/xc)^(-beta), with x=r/R200c. Each of P0, xc and beta evolves as A0 (M200c/1e14 physical Msun)^alpha_m (1+z)^alpha_z. Electron pressure is 0.5176 times thermal pressure, and Compton y is sigma_T/(m_e c^2) times its LOS integral.

Consequently the outer slope is beta(M,z)+0.3: LOS convergence needs beta>0.7, while a finite untruncated volume integral needs beta>2.7. A pivot beta0 above those thresholds does not ensure they hold at every mass and redshift. The 4R200 projected painting cutoff does not truncate the three-dimensional profile; the original LOS endpoint is 1e5 R200.

For steep profiles, (1+x/xc)^(-beta) approaches exp(-beta*x/xc) in the inner region. The characteristic pressure extent therefore scales roughly as xc/beta, not xc alone. The finite integrated pressure scales as P0*xc^3 times a beta-dependent integral. This explains both the amplitude/size/slope degeneracy and why combinations can be much more extreme than any single parameter edge.

![LOS cutoff](plots/07_line_of_sight_cutoff.png)

The fixed projected painting radius imposes a separate extent limit. At M200c=1e14 Msun and z=0.5:

| Profile | Fraction of untruncated projected flux retained inside 4R200 |
|---|---:|
| Battaglia12 | 97.041% |
| fit_L1_m9 | 99.432% |
| fit_fgas-8sigma | 99.142% |
| fit_Mstar-1sigma | 99.874% |
| xc_high | 44.942% |

These fractions compare intrinsic profile integrals with the model's untruncated total; they are not measured C_ell biases or observed missing gas fractions. The diagnostic is undefined when the untruncated total diverges. Wider profiles require a physical extent choice and a painting-radius convergence test as well as a pixel-resolution test. See audit/projected_cutoff.json.

## Numerical resolution

The pressure checks below compare production interpolation with independently validated scaled LOS quadrature. They include the exact archived radiation contribution to H(z). They are unweighted diagnostic-profile errors, not measured C_ell biases. Extreme high-mass/low-redshift grid points need not be occupied by the catalogue.

| Case | All-grid profile-error p95 | z>=0.1 profile-error p95 |
|---|---:|---:|
| reference_Battaglia12 | 77.843% | 0.00079869% |
| fit_L1_m9_iter2 | 22.113% | 0.00015683% |
| fit_fgas-8sigma_iter3 | 14.591% | 0.00018812% |
| fit_Mstar-1sigma_iter4 | 23.665% | 0.00015388% |

A bounded quadrature diagnostic found 2 native evaluation-budget hits in 926 probes; all 2 occurred beyond the painted 4R200 disc. At those points, the rescaled integral needed [19, 19] evaluations. Regular native/rescaled values agreed to 1.137e-13 relative. The timeouts therefore expose a numerical precision problem in very faint grid tails, not a physical exclusion of those pressure parameters. The temporary beta<=40 proposal-search cap is a pilot workaround, not a proposed physical upper bound on beta(M,z). See audit/quadrature_cost_summary.json and code/diagnose_quadrature.jl.

The archived Battaglia12 reference reveals which interpolation errors predate the wider prior. Separate direct-quadrature errors and zero-versus-positive-reference counts are saved in interpolation_summary.json.

![Interpolation](plots/08_interpolation_accuracy.png)

Pixel sampling at M200c=1e14 Msun and z=0.5:

| Profile | Nside | Median sampled/continuous flux | 16th–84th percentile |
|---|---:|---:|---:|
| Battaglia12 | 4096 | 0.99347 | 0.96417–1.0567 |
| Battaglia12 | 8192 | 0.99788 | 0.99479–1.0059 |
| compact_steep_evolving | 4096 | 1.514e-06 | 1.5939e-09–0.014395 |
| compact_steep_evolving | 8192 | 0.0004795 | 2.1398e-05–0.057143 |

These are 32 reproducible actual HEALPix placements per case, not a converged population mean or a full-spectrum resolution test. Very narrow profiles are sensitive to where their cores land within pixels. An ensemble mean over uniform random placements can be unbiased in principle, but these examples do not establish convergence of rare bright alignments or of the power spectrum. A normalized Gaussian beam preserves total flux and cannot restore flux missed at painting.

![Pixel sampling](plots/09_pixel_sampling_limits.png)

## Lee22 and noise comparisons

The [Lee22 pressure formula, Table 1 and Eq.12](https://arxiv.org/html/2205.01710) adds concentration dependence and an amplitude mass break. The displayed c=4.5 example uses its electron-pressure normalization and converted break-mass pivot. Grey radial shading marks its fitted radial interval; other regimes are extrapolations. The calibration approximately spans 1e13–1e14 h^-1 Msun and z<=2. It is a literature reference, not an independent FLAMINGO measurement.

![Pressure profiles](plots/06_pressure_profiles_and_lee22.png)

![Shared SO noise](plots/03_shared_so_noise.png)

The exact seed-12345 mask and both SO noise arrays were verified for every completed full map. Signal-noise cross terms differ between signal maps even when the noise pixels are identical. No noisy statistic was fitted.

Two independently painted maps differing only in P0 also verified clean C_ell proportional to P0 squared to 2.33e-14 maximum relative error. The noisy cross-spectrum identity, including its terms linear in signal amplitude and the unchanged noise-noise term, agreed to 3.21e-15 of peak noisy power.

## Recommendation before a larger run

1. Treat the current rectangle as an exploration envelope. The unrestricted product distribution is not a production-ready prior.
2. Apply a coupled outer-slope condition. The optional implemented screen requires min_grid beta>=2.8 and a conservative missing-column bound <=1%. This is a mathematical/numerical condition, not an observational prior. An explicitly motivated three-dimensional pressure truncation is an alternative, but it changes the forward model and requires new validation.
3. Constrain integrated pressure and profile extent jointly using the reference family and appropriate halo data. A prior in integrated Y200 and a profile-size coordinate is easier to control than independent P0, xc and beta. Spectral fits alone do not determine that physical prior.
4. Validate a stable pressure-quadrature/interpolation implementation and pixel-integrated or demonstrably converged halo painting over the retained range. The present pilot intentionally preserved historical operators.
5. Generate a valid new training set and retrain/validate SBI for the changed prior. Recheck whether the existing 40-bin MOPED compression retains sufficient information across the new range; changing bounds alone cannot extend the old trained posterior.

The optional outer-slope screen retains 42348 / 65,536 draws (64.618%), yet its retained finite-Y200 ratios still span 8.956e-19–4.508e+05. Passing that screen alone therefore does not establish production readiness.

## Resources and reproducibility

17 full maps passed the operator/parameter/hash checks. Maximum measured RSS among completed maps was 11.282 GiB; median per-map elapsed time was 294.3 s. At that pilot throughput, 1,000 serial maps would take about 81.7 hours, or 2125 allocated core-hours at 26 CPUs. This estimate includes the pilot's clean/noisy spectra and profile checks, excludes proposal optimization and failed attempts, and is not a large-job benchmark.

5 candidate maps did not complete; their exit codes and elapsed times are retained in resource_usage.json. The completed-map throughput estimate is not applicable to unrestricted draws containing these failures. The final jobs used mini2 after mini filled; the default PBS queue remains mini.

The retained interpolation caches occupy 2.125 GiB. The original FLAMINGO inputs, catalogue and caches remain on the cluster. Repainted pixel maps were reduced to spectra without retaining their pixel arrays. PNG, vector PDF, CSV and JSON artifacts are transferred with SHA256 verification.

Each retained interpolation cache occupies about 128.01 MiB. Keeping one per model would require about 125.0 GiB for 1,000 models, before other outputs. A larger campaign needs an explicit cache-retention policy; the RAM measurement alone is not its storage budget.

Created source files, purpose and execution commands are listed in [README](code/README.md). The complete file inventory is changed_files.txt. Cluster verification is in verification.json and transferred-file verification is in local_verification.json. No original pipeline or trained-model files were edited.
