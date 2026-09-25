
## Spherical tSZ: tests and code changes

Updated 2026-09-20 15:29 UTC. Deadline for overnight work: 21 September 2026, 08:00 Europe/Berlin (06:00 UTC).

<b>Decision at this checkpoint:</b> the finite spherical projection is numerically verified over the extended parameter box. This does not yet certify a flat-prior production dataset: full-sky resolution controls are running, and a production-quality treatment of unresolved haloes is still required.

Cluster job 598439 runs six clean spherical controls: Battaglia12 and the historical FLAMINGO fiducial fit, each at raw NSIDE 4096, 8192 and 16384. All outputs use the same NSIDE 4096 observation grid, 2 arcmin Gaussian beam and mask. Their 40-bin resolution diagnostic is separate from the newly specified unbinned MOPED input. Job 598448 automatically analyzes the completed or partial results after that job exits.

<b>Latest instruction:</b> prepare a 256-row diagnostic, but do not run it. The design, independent noise seeds, 192/64 train-test split, generation and analysis scripts are prepared. No 256-row generation or inference job has been submitted.

The first new spherical 4096 control processed all 85,224,251 haloes in 958.3 seconds on four CPUs. This is a clean-spectrum benchmark, not a complete noisy training-row time. The current queue also contains the separately submitted Battaglia12 Fisher workflow; none of those jobs is changed here.

Main findings: direct Python and Julia columns agree within 7.3e-13; 2-D and 3-D integrated pressure agree within 5e-14; CLASS-SZ pressure shapes agree within 4.3e-15. Pre-beam painting conserves halo flux in sparse HEALPix tests, while point sampling can strongly mismeasure compact haloes. A naive FFT acceleration failed and is not adopted.

This document separates completed scalar tests, sparse-pixel experiments, historical full-sky measurements and new running full-sky controls. Only the last category measures the proposed spherical map operator directly.


## The physical projection and its amplitude

Write x = r/R200c, q = x/xc, and p(x) = q^gamma (1 + q^alpha)^(-beta_raw), with alpha = 1 and gamma = -0.3. Each of P0, xc and beta_raw evolves as its pivot amplitude times (M/1e14 Msun)^alpha_m (1+z)^alpha_z. The catalogue halo_mass_m200c is in Msun/h and is divided by h = 0.68 before profile evaluation.

For projected radius b/R200c = x and spherical outer radius X = 4, the correct column is J(x) = 2 integral[0,sqrt(X^2-x^2)] p(sqrt(x^2+l^2)) dl. It is exactly zero at and beyond X. The old projected-disc model integrated to l = 1e5, admitting gas outside the physical sphere even when the map was cut at projected radius 4 R200c.

The observed halo field is y(theta) = A(M,z) J(theta/theta200), where A = (sigma_T / (m_e c^2)) x 0.5176 x [G M 200 rho_crit(z) f_b / 2] x P0(M,z). The pressure scale contains 1/R200c and the physical LOS element contributes R200c; these factors cancel. The electron-pressure factor, baryon fraction, mass units and P0 normalization are retained.

The prepared map operator retains the archived painter cosmology h = 0.68, Omega_b = 0.049 and Omega_c = 0.261. These are not the exact native HalfDome catalogue cosmology, nor FLAMINGO cosmology. The CLASS-SZ spectrum comparison uses the same painter values; previously processed FLAMINGO observations retain their own cosmology. No cosmology rescaling is introduced in this diagnostic.

<b>The old wrapper obtained A indirectly:</b> evaluate the old projected profile at x = 1 and divide by its old LOS integral. Algebraically the integral cancels. Numerically this calls precisely the integral that may underflow, hit an old slope restriction, or take too long. It can create 0/0 despite a finite spherical model.

<b>The revised tSZ wrapper obtains A directly</b> from XGPaint.prepare_profile_slice(inner, mass, z).amplitude. A regression test replaced the old LOS routine with an exception: all 12 shallow/steep/compact amplitude checks still passed. The generic tau/DM fallback was retained; it is not newly certified for arbitrary extreme parameters.


## Numerical normalization of the LOS integral

Physical pressure normalization and numerical quadrature normalization are different. The physical A above must not be divided out of the final map. Numerical normalization temporarily rescales an integrand to a convenient magnitude and restores the factor exactly afterward.

For x &gt; 0, use l = x sinh(u), r = x cosh(u), and dl = r du. The column becomes J = 2 integral exp(g(u)) du, with g = log[r p(r)]. Choose a finite interval peak c = max g, evaluate K = integral exp(g-c) du, and return J = exp(log(2K)+c). The normalized integrand is bounded by one; neither an arbitrary 1e9 multiplier nor a minimum observable pressure is required.

At x = 0, use t = log r instead. Then dl = r dt and the integral extends from t = -infinity to log X. The central integrand behaves as exp[(1+gamma)t]; gamma = -0.3 makes it integrable. The code never evaluates an infinite central pressure directly.

With XGPaint conventions, log[r p(r)] = (1+gamma) log r - gamma log xc - [(beta_internal+gamma)/alpha] log(1+exp(alpha log(r/xc))). The code uses a stable log1p-exp evaluation. Critically, beta_internal = alpha beta_raw - gamma; the two beta definitions must not be interchanged.

The peak is found from interval endpoints and, when it lies in the interval, r/xc = [(1+gamma)/(beta_internal-1)]^(1/alpha). Thus the numerical scaling responds to the actual profile. Rescaling cannot cure a genuinely divergent integral, but all central columns here are finite and the outer radius is finite.

Restoring the scale means that P0 still multiplies y and P0 squared still multiplies the power spectrum. Halos are not forced to have equal integrated Y. Unit-Y profiles appear only in the sampling diagnostic to make relative pixel errors comparable.


## What passed, and what finite means

Finite spherical support removes the need for the infinite-radius energy condition beta_raw &gt; 2.7. That inequality describes an extrapolation to infinity, not the thermal energy inside 4 R200c. A flat, independent beta box is mathematically available in the finite-radius model.

The outer radius remains a physical model assumption. Extending it from 4 to 8 R200c at fixed 3D amplitude increases Y by 6.7% for the pivot B12 shape, by a factor 4.64 for the shallow-beta test and by a factor 2.73 for the extended-core test. These are model changes, not numerical convergence failures.

Finite and positive do not mean astrophysically plausible. The Cartesian cache-corner test includes mass-redshift combinations not occupied by the catalogue. Its largest test column was y about 1.23e4: this is an extreme mathematical stress test, not evidence that such a physical halo is allowed. Physical plausibility and observable fidelity must be assessed on occupied catalogue support and against data.

A true column below floating-point representability contributes negligibly by itself. The observed amplitude bound on the full corner grid is about 135. The absolute 1e-300 cache floor, even summed over 85 million haloes and multiplied by the maximum chord 8, is below 7e-292 in y. However, interpolation between a floor and non-negligible cells can create larger errors; this separate effect must be tested and is not excused by the tiny floor.

CLASS-SZ checks use physical M200c converted to Msun/h for its interface, h = 0.68 and the same B12 exponents. Its saved spherical halo-model spectrum uses 1h + 2h terms and is stored in dimensionless y units after division by 1e12. Differences between a theoretical mass function and the fixed HalfDome catalogue are not automatically painting errors.


## Independent pressure-profile check

![Lines: the analytical B12-family pressure formula. Open circles: CLASS-SZ evaluations. The pivot mass and redshift are shown; the numerical comparison also spans other masses and redshifts. The shallow shape is an intentional numerical stress test.](plots/classsz_pressure_comparison.png)

This agreement checks parameter conventions, physical mass conversion and radial shape. It does not validate halo abundances, map discretization, feedback physics or an SBI posterior. The historical FLAMINGO fit is an effective parameter set from the earlier model, not a refit after spherical truncation.


## CLASS-SZ spectrum context

![Same painter cosmology and 2 arcmin beam: new unmasked HalfDome spherical control versus CLASS-SZ one-halo plus two-halo spectrum with outer radius 4 R200c. The map curve is the completed 4096 point-painted control; its resolution convergence is still under test.](plots/classsz_spectrum_comparison.png)

The theoretical curve integrates a mass function and bias model, while HalfDome paints a particular halo catalogue. Their difference combines abundance, clustering, catalogue selection and rendering effects. This is an independent scale and normalization check, not an equality test or a fitted cosmology correction.


## Before the beam versus before the pixels

The pressure is already integrated along the LOS before the map is made. The missing operation is spatial integration or smoothing over an angular pixel. Evaluating the correct continuous column only at pixel centres can miss a core or hit it unusually closely. Applying a beam afterward smooths that incorrect set of pixel amplitudes; it does not restore lost flux.

The independent pre-beam reference convolves the continuous spherical column with a normalized 2 arcmin Gaussian in the tangent plane. Its radial kernel contains exp[-(theta-t)^2/(2 sigma^2)] I0e(theta t/sigma^2). Positive quadrature is used, with refinement and integrated-flux checks. Beam wings extend beyond the physical gas sphere; the convolved image is not cut again at 4 R200c.

![Actual sparse HEALPix samples at 32 random sky centres per point-sampling case. Left: mean and position scatter of integrated Y. Right: local Fourier-power errors for pre-beam sampling; shaded range is across centres. These are isolated-halo tangent-plane tests, not a masked full-sky certification.](plots/pixel_prebeam_test.png)

Pre-beam flux errors are at the level of about 1e-7 in the continuous convolution check, and positional flux scatter is very small. At NSIDE 4096, the worst tested local power errors are about 0.015% at ell = 4000, 0.155% at 6000 and 1.72% at 8000. Their importance for the full data vector depends on halo weights, noise covariance and correlations.


## Extreme profiles and interpolation limits

![Solid: finite spherical columns; dashed: the same profiles after the 2 arcmin beam. Each curve is normalized to unit integrated Y for this numerical comparison only. The underlying parameter prior retains its physical amplitude.](plots/raw_and_prebeam_profiles.png)

Nested HEALPix child pixels were also tested. Four-times and sixteen-times finer linear sampling approach a pixel integral, but a sufficiently compact core can still be missed even at the latter setting. The parent-pixel Fourier response is retained in that test; it is not removed by dividing an arbitrarily sampled map by a generic pixel window.

The cache stores the chord mean J/(2L), interpolates its logarithm, and restores the exact geometric 2L after interpolation. This avoids trying to interpolate a moving zero at the sphere boundary. An extended cache test checks 10,000 off-grid points per model/refinement, including the inherited angular lower limit. That limit can matter for very compact cores even if NSIDE is increased.

Six auxiliary cache tests completed. Their compact case retained a maximum central-normalized error of 1.387% at both refinements. However, the code audit found angular padding 128 in that test, versus 256 in the prepared map; its minimum angle was 100 times larger. Its cosmology also differed slightly. The 1.387% value must not be assigned to the prepared painter. A matched-settings compact check is reported on the following page.

The attempted Pixell radial-FFT pre-convolution produced centre errors and did not converge robustly under the tested settings. It remains an explicitly failed acceleration experiment. No negative-pressure clipping or unvalidated central repair was silently adopted in production.


## Interpolator changes: counts versus coordinates

The default number of cache nodes did not increase. Log-z moves nodes toward low redshift, where angular halo scales vary rapidly. Doubling every axis uses eight times as many cells and was a convergence experiment, not an automatically adopted default. Changing the interpolation coordinate is distinct from increasing HEALPix NSIDE.

Actual angular padding is 256 in both the guarded baseline and prepared run: theta_min = 1.01815e-11 radians. The earlier auxiliary test used padding 128, theta_min = 1.01815e-9 radians. The audit now records that mismatch explicitly.

The matched compact check uses the actual padding and painter cosmology. Maximum central-normalized errors are 0.04014% (grid factor 1), 0.04014% (grid factor 2). These are pointwise profile errors, not spectrum or posterior errors.

The older guarded LOS implementation already had a sinh transform and peak scaling. The new spherical work adapts those ideas to the finite chord, obtains the pressure amplitude directly, and removes the short-chord approximation. The spherical tolerance is 1e-10, compared with 1e-12 in the older guarded LOS implementation; both use quadrature order 9.

The current spherical quadrature uses QuadGK's default evaluation limit and does not inspect its returned error estimate. The older guarded routine imposed 4096 evaluations and checked that estimate. Independent reference tests passed, but explicit bounded-work/error reporting remains a production implementation check. NUMERICAL_CHANGE_AUDIT.md records these distinctions and the exact source history.


## What NSIDE costs, after the actual beam

![Preserved full-sky cylindrical controls: same fine interpolation grid, fixed 2 arcmin beam, mask and output grid. The new spherical controls are a separate running experiment. Error bars are summarized with the saved conditional split-noise covariance; cosmic variance is excluded.](plots/resolution_historical.png)

Historical eight-thread clean timings were 551 s at 4096, 1,119 s at 8192 and 3,347 s at 16384 for B12, with peak memory about 12.1, 19.2 and 55.5 GiB. One Float64 map alone occupies 1.5, 6 and 24 GiB respectively. The tested operator fixes the harmonic band limit and returns to 4096, so transform cost is not inferred from a simple NSIDE-cubed rule.


## Why noise does not settle the resolution question

For a 2 arcmin FWHM Gaussian, B_l squared is about 0.783 at ell = 2000, 0.376 at 4000, 0.111 at 6000 and 0.0205 at 7979. Thus the highest multipoles are strongly suppressed. But the historical 4096 discrepancy was strongest in useful intermediate multipoles, not only at the noisy high-ell end.

Using the preserved 40-bin conditional noise covariance, the historical B12 distance sqrt(deltaD^T C^-1 deltaD) was 9.79 for 4096 versus 16384 and 0.383 for 8192 versus 16384. For the historical FLAMINGO fit it was 1.88 and 0.026. A chosen 0.1 numerical-error allowance is a declared accuracy target, not a physical law. These old-cylinder numbers cannot certify the new sphere or the full extended prior.

<b>New spherical result:</b> the matched 4096 and 8192 B12 controls now give a 9.410 distance in that same reference metric and a maximum bandpower difference of 17.50%. Their four-CPU times are 958.3 and 2558.5 seconds, with peak RSS 9.43 and 19.91 GiB. This is a beam-smoothed spectrum discrepancy, not a measured nine-parameter bias or an unbinned-MOPED uncertainty. The 16384 spherical control remains pending.

<b>4096 remains a plausible output resolution.</b> The evidence does not support declaring the historical point painter adequate just because the beam is 2 arcmin. A corrected renderer must first recover flux before discretization, then demonstrate acceptable error in the retained beam-smoothed, masked bandpowers. Merely raising NSIDE also cannot guarantee that every compact halo in the extended prior is resolved.

Crossing two independent noise realizations removes their mean noise bias: E[(s+n1)(s+n2)] = signal power. It leaves variance. For a fixed sky and nu independent real modes, Var(cross) = [C(N1+N2)+N1N2]/nu; an ensemble of random skies adds the cosmic-variance term 2C squared/nu. Shared foreground residuals remain in the cross mean.

The numerical noise controls reproduced the means and variances to roughly one percent. Existing mocks assign the tabulated N_l to each split. If that table denotes a full-survey coadd, two equal observing halves instead have about 2N_l each, under the white-noise time-scaling assumption. A real SO mock must state this choice and include correlated residuals where appropriate; the existing dataset convention was not silently changed.


## The flat candidate and inference controls

![Blue: independent uniforms in all nine physical parameter values. Grey: the completed guarded 8k empirical density; dotted lines mark that run's box boundaries. The FLAMINGO marker is its historical effective fit, not a newly fitted spherical constraint or a sampling weight.](plots/flat_prior_comparison.png)

The candidate uses no rejection, Gaussian weights or log-uniform amplitudes. A Sobol design is a deterministic space-filling representation of the declared continuous uniform density; it is not an independent random sample. The prefix is stable when increasing the target count from 8192 to 524288.


## Inference and the overnight decision

A one-parameter P0 control used the exact spectrum scaling D(P0) = (P0/18.1)^2 D(B12), a saved fixed covariance, and the corresponding rank-one sufficient MOPED statistic. Noiseless inversion recovered the labels exactly. A 1,024-example SBI fit recovered the four tested posterior means with RMS error 0.249% of the prior width. Shuffling parameter-spectrum pairs raised that error to 32.65% and returned nearly observation-independent means around 30.5.

This demonstrates observation dependence, but the uncertainty check fails: the learned 68% intervals are 7.1, 7.9, 15.8 and 28.7 times wider than the exact intervals at the four truths. Small mean error is insufficient for calibration. This is a stringent one-parameter software control using fixed covariance, not a measurement of bias in the existing nine-parameter model. Flat priors alone do not repair inference accuracy or create information about weak parameters.

<b>Prepared size: 256 rows, not submitted.</b> The user chose preparation only. The cluster root is /lustre/work/kristero10/tsz_spherical_diagnostic_256_prepared_20260920. Four optional 26-CPU workers can divide the rows into groups of 64; starts depend on PBS capacity and user limits. This is an experimental spherical 4096 point-painting diagnostic, with its known numerical limitations explicitly retained in the manifest.

At the measured four-CPU clean-row time, 256 clean rows alone correspond to about 68 worker-hours. Additional CPU allocations could shorten wall time, but the complete new noisy-row throughput is not yet measured. No overnight completion time is promised while the 256-row run is unsubmitted.

Prepared analysis uses 192 training-pool and 64 held-out rows, nested sizes 64/128/192, two training seeds, 40-bin/PCA/unbinned-regression-MOPED comparisons, shuffled controls and the three previously processed FLAMINGO observations. MOPED takes 7,900 individual D_ell values at ell=80..7979. Source snapshots, saved design arrays, row locks, atomic status files and explicit failure reporting prevent silent row dropping or relabelling. A separate synthetic software check does not count as scientific validation.

Outstanding gates: full-sky spherical resolution and cache refinement; a tested production treatment of compact haloes; joint-extreme maps with all failures retained; actual independent-noise row timing; round-trip data/seed/operator checks; P0-beta controls; nine-parameter held-out recovery, shuffled/prior-only comparisons, repeated-noise coverage and SBC plus information-sensitive tests. The cluster follow-up produces the resolution plots automatically even if the parent exits with a failure, making partial outcomes visible.


## Posterior means are not a calibration test

![P0-only software control: posterior mean and central 68% interval for paired and shuffled training. The diagonal marks correct means. The exact likelihood intervals are substantially narrower than the paired SBI intervals, as quantified in the preceding page.](plots/sbi_amplitude_control.png)

The control deliberately supplies a sufficient one-dimensional statistic. Failure to recover its uncertainty cannot be attributed to nine-parameter degeneracy or lossy compression. It motivates explicit likelihood controls and held-out coverage checks before interpreting apparent precision or increasing the simulation count.


## Unbinned MOPED and the resolution decision

MOPED now reads the original per-row masked clean and signed cross C_ell arrays, converts ell=80..7979 to D_ell and compresses all 7,900 coordinates. The FLAMINGO observations retain the same multipoles. Rebinning is used only as a parity check and for separate 40-bin/PCA comparisons; no multipoles are reconstructed from bins.

The signed-asinh scaling is fitted on optimization rows only. A small local linear regression estimates clean-spectrum derivatives, and OAS shrinkage regularizes the paired-residual covariance. This is explicitly approximate regression-MOPED. The larger established pipeline uses a 55-term quadratic fit requiring at least 600 optimization rows, which the 256-row pilot cannot supply.

A row-space Woodbury solve applies the exact OAS-plus-ridge covariance inverse without allocating a 7,900-square matrix. Dense-reference errors were below 2.5e-16. A 7,900-feature regression test recovered nine within-bin shape modes whose 40-bin average was constant. The compressed covariance identity error was 2.9e-15. A synthetic full analysis also executed successfully.

<b>Current decision:</b> do not accept the raw 4096 point painter for precision nine-parameter constraints over this prior on the present evidence. A 4096 output map may still work with a validated pre-beam or pixel-integrated renderer. Neither the successful scalar LOS tests nor a finer interpolation grid repairs missed pixel flux.

The inference-specific test must pass matched-resolution spectra through one fixed fitted unbinned compressor, compare their compressed errors with its noise covariance, and assess shifts in identifiable parameter combinations and held-out posteriors. Fitting a new compressor independently at every resolution may hide rendering changes. Mock recovery with the same flawed renderer for both training and testing cannot reveal a shared simulator bias.


## Files, provenance and references

<b>Earlier source edit:</b> truncation_comparison/spherical_truncation_profiles.jl. Changes: stable finite-chord quadrature in transformed coordinates; explicit internal-beta convention; removal of the tiny-chord approximation; direct prepared tSZ amplitude. This update audits that implementation and changes the prepared analysis to unbinned MOPED.

<b>New source directory:</b> SBI_analysis/tsz_spherical_preflight_20260920. projection.py, profile_tests.py, validate_columns.jl and corner_tests.jl test LOS physics. interpolation_test.jl checks the cache. classsz_test.py supplies independent pressure and halo-model references. pixel_test.py implements positive pre-beam and subpixel tests. fft_prebeam_test.jl and check_fft.py retain the failed acceleration experiment. design_tests.py and inference_control.py test design, noise and inference controls.

fullsky_test.jl and run_fullsky.py implement isolated spherical point-painting controls. resolution.pbs, analyze_resolution.pbs, deploy.py, fetch.py and cluster_capacity.py operate and inspect cluster tests. analyze_fullsky.py, plot_results.py and build_report.py generate the deliverables. Numerical JSON/TOML files and exact source hashes are retained alongside the figures; changed_files.txt provides the full inventory.

diagnostic.py, diagnostic_row.jl and diagnostic.pbs prepare and implement the unsubmitted 256-row run. diagnostic_analysis.py and diagnostic_analysis.pbs provide its data checks and inference comparisons. prepare_cluster.py uploads the design without submission; verify_prepared.py checks saved arrays, source identities, installed cluster imports and the absence of generated rows or diagnostic jobs. smoke_analysis.py exercises the analysis on a separate synthetic fixture only.

unbinned_moped.py implements the memory-bounded covariance solve; test_unbinned_moped.py checks it against dense OAS and within-bin information modes. stage_observations.py retains original FLAMINGO multipoles. NUMERICAL_CHANGE_AUDIT.md records the detailed code comparison and limits.

Primary references: HEALPix pixel-window definition, https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm ; HEALPix conventions, https://healpix.sourceforge.io/html/intro_HEALPix_conventions.htm ; CLASS-SZ source, https://github.com/CLASS-SZ/class_sz ; CLASS-SZ pressure-profile notebooks, https://github.com/CLASS-SZ/notebooks . Local XGPaint pressure normalization is checked against its prepared-profile implementation. This report records measured tests rather than treating package import or a queued job as scientific validation.

