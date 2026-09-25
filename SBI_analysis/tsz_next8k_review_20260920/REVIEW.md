
## tSZ priors, rendering and the next 8k dataset

Technical review | 20 September 2026

Decision supported by the evidence: use the HalfDome spherical-truncation update as the physical starting point, retain the requested wide physical-uniform ranges for testing, and validate the renderer and inference controls before starting the next diagnostic dataset. No new production run was submitted in this review.

| Finding | Consequence |
| --- | --- |
| The continuous tSZ column is already computed before mapping. | The unresolved-halo problem is pixel sampling of that column. A map cross-spectrum does not repair it. |
| A 2 arcmin beam makes the highest bins relatively unimportant. | Measured errors at ell roughly 2000-6000 remain important in the existing noise metric. |
| Matched clean tests give a 2.02-2.03 time ratio for raw 8192 versus 4096. | This is not a measurement of the proposed spherical, noisy production pipeline. |
| MOPED beats a prior-mean prediction on held-out simulations. | Prior influence remains; flat priors alone cannot remove degeneracies or guarantee calibrated inference. |


### Verified versions and scope

HalfDome branch cluster was fetched and its local HEAD matches origin at 10947ce. The user-confirmed spherical update is 3522b71. The public XGPaint cluster branch remains at 5dd0b57; the sphere is implemented in the HalfDome wrapper, not in that upstream line-of-sight routine.

Two cluster SSH attempts timed out. Calculations here use current fetched source and preserved outputs from completed cluster experiments. They are a fresh analysis of those outputs, not fresh cluster timings or a verification of the live cluster checkout.

The completed 8,192-row dataset, its seeds, trained models and earlier study outputs were left intact. The tables below explicitly distinguish historical production, separate experiments, and proposed changes.


## Where tSZ is computed, and what FRB does differently


### Current forward calculation

XGPaint first evolves the gNFW pressure parameters with mass and redshift, then integrates electron pressure along a continuous line of sight. It stores the projected profile in an interpolation cache. The painter evaluates that profile at HEALPix pixel centres, adds the haloes, and only then applies the Gaussian beam. Independent noise maps, the common mask, cross-power estimation, 40-bin compression and SBI follow.

y(θ) = [σT/(mec²)] ∫ Pe(r) dl

mA = W [B * y + nA],   mB = W [B * y + nB]

A compact halo can fall between pixel centres or have its central value overweighted. The resulting flux error exists before either noise realization is drawn. Smoothing a wrongly sampled map cannot reconstruct missed halo flux; independent-noise cross-correlation preserves the same erroneous clean signal in both maps.


### The FRB comparison

The FRB DM calculation evaluates halo columns along selected sightlines, so it does not require a pixel average for the DM value on a ray. However, compare_takahashi_sightlines.py::sample_y reads a HEALPix tSZ map, filters its harmonics with an annular filter and beam, then samples that map at the source pixels. Its tSZ side therefore has the same map-representation issue.


### Useful alternatives to test

Pre-beam or pixel-integrated painting: project each halo continuously, convolve with the 2 arcmin beam before sampling, or integrate it over pixels. This is an implementation option to test at 4096; it is not implemented by this review.

Direct harmonics: transform each axisymmetric profile and sum its harmonic contribution at the actual halo positions. This retains inter-halo cross terms. Summing only individual halo powers gives a Poisson/one-halo reference, not the full masked map statistic. A naive direct harmonic sum over the full catalogue is expensive and needs an independently benchmarked acceleration.

HEALPix provides sampling, transforms and pixel windows; its pixel window does not infer a continuous compact halo that the painter never represented. See the HEALPix discretisation reference [R1].


## Spherical truncation: geometry and implementation

A projected cut b < X R200c with a long line-of-sight integral includes gas outside the sphere. If the intended gas domain is r < X R200c, the line-of-sight endpoint must depend on the impact parameter. This is a change in the physical model, not simply a faster integration method.

x = b/R200c,   L(x) = √(X² - x²),   ysph(x) = 2A ∫0L(x) p(√(x² + u²)) du

The column is exactly zero for x ≥ X. The default comparison radius is X = 4. The projected integral must satisfy 2π ∫ x ysph(x) dx = 4πA ∫ r²p(r) dr, with both outer bounds X and consistent dimensional factors.


### What commit 3522b71 implements

truncation_comparison/spherical_truncation_profiles.jl introduces ChordMeanProfile. Rather than log-interpolating a column that goes to zero at the sphere edge, it caches the positive chord mean g = ysph/(2L). The painter restores the exact factor 2X√[1 - (θ/θmax)²]. This separates the known edge geometry from the smooth quantity being interpolated.


### Remaining dependency exposed by source review

The wrapper currently recovers the normalization using inner(θ200, M, z) / old_LOS_integral(1). The ratio is algebraically correct when both evaluations are reliable, but it unnecessarily invokes the old long-column integrator. With the previous stable override it can encounter the β > 0.7 assertion; extreme underflow can also make a ratio unsafe.

Required before production: obtain the physical amplitude directly from the compatible inner prepared-profile API, or factor that normalization into a shared tested function. Use a normalized positive quadrature for broad spherical profiles. These changes are proposed here, not silently applied to the wrapper.

After convolution the observed halo extends beyond its physical sphere. Pre-beam painting must include beam wings and apply the beam exactly once. Cutting the convolved profile again at 4 R200 would discard signal.


## Numerical changes already made for the wider prior


### 1. Normalized line-of-sight quadrature - used in the completed 8k

flamingo_linear_prior/stable_los.jl replaces a difficult direct long-column integration by u = asinh(l/x), so r = x cosh(u). For α = 1 and γ = -0.3, the transformed positive integrand has log shape:

log f(u) = 0.7 log r + 0.3 log xc - β log(1 + r/xc).

The code subtracts its peak log amplitude before quadrature and restores the normalization afterwards. For β > 0.7 the stationary radius is r* = 0.7 xc/(β - 0.7), clamped to the integration interval. This avoids asking the integrator to resolve extremely small absolute values against an inappropriate scale. The historical integration endpoint and map definition were retained in that run.


### 2. Positive cache floor - used in the completed 8k

A floor max(minimum positive cache value × 10-6, nextfloat(0)) prevents log(0) when profiles underflow. This removes a numerical failure; it does not prove the modified faint tail has zero effect on the observable. A convergence test must bound that effect. True zeros outside a sphere should be represented by the analytic support factor, not a positive physical tail.


### 3. Separate follow-up implementations

tsz_guardrail_study/map_experiment.jl tests log-redshift interpolation, grid refinement and raw map resolution independently. tsz_beta_flat_followup_20260920 generalizes the finite-LOS normalization for β ≤ 0.7 by placing the maximum at the upper endpoint, tests finite spheres, and constructs a continuous correlated beta prior with flat marginals.

Those study implementations did not retroactively change the completed 8k dataset. Nor do their profile-level tests certify the current spherical wrapper through the full noisy map pipeline. No change to the physical baryon fraction normalization is proposed; pressure normalization and physical M200c units must remain consistent with the pinned XGPaint model.


## Guardrails: physical statements versus engineering cuts

| Old restriction | Reason and disposition for the next model |
| --- | --- |
| Evolved beta between 2.8 and 50 on the full cache rectangle | 2.8 was a margin above the untruncated finite-energy threshold 2.7; 50 is not a physical singularity. A finite sphere does not need either outer-tail convergence cut. |
| (xc/beta)/(xc/beta)B12 between 0.4 and 8 | A compactness proxy, not a measured HEALPix error. Replace with flux and bandpower convergence of the renderer; do not reject compact physical profiles merely because point painting fails. |
| Y200/Y200,B12 between 0.003 and 30 | An exploration/dynamic-range choice, not a universal energy law. Test finite integrated pressure directly; retain observational or thermodynamic bounds only as explicit scientific assumptions. |
| Missing central long-LOS tail at most 1 percent | A conservative bound combining extreme xc and beta values, sometimes from different locations. For a sphere, gas outside X is excluded by definition; vary X as model sensitivity. |
| 1200-second row timeout | An operational limit. A timeout must remain a recorded failure or be rerun; dropping it changes the prior. |

For p(r) ∝ r-0.3(1 + r/xc)-β, the untruncated outer pressure slope is -(β + 0.3). Total integrated thermal energy requires β > 2.7; a column to infinity requires β > 0.7. A finite-radius sphere removes both outer-infinity divergences, while the fixed central r-0.3 cusp remains integrable.

The unused cache corner test checks the entire interpolation rectangle, including high-mass/high-redshift combinations absent from the catalogue. Those cells must be numerically computable if the cache builds them, but their hypothetical infinite-radius energy need not constrain a catalogue-only physical prior. Changing the catalogue later requires checking its support again.

Finite radius does not establish that every pressure amplitude is astrophysically realistic. Very shallow profiles can place most thermal energy near the chosen outer boundary. Treat that radius and pressure normalization as explicit model assumptions rather than describing all formerly excluded cases as impossible.

Measured finite-radius examples at M = 1013 M⊙ and z = 2 illustrate the distinction: increasing the sphere from 4 to 16 R200 gives energy ratios 1.032, 7.474 and 20.612 for evolved β = 6.269, 1.617 and 0.644. The last two profiles are finite inside each sphere, but their prediction depends strongly on the chosen outer radius.


## Flat priors and the proposed exploration ranges

For each of P0, xc and beta, q(M,z) = q0 (M/1014 M⊙) αm(1+z)αz. The amplitudes and exponents below are the nine sampled parameters, in the established order.

| Parameter | Battaglia12 | Completed 8k box | Extended test box |
| --- | --- | --- | --- |
| P0 | 18.1 | [1, 60] | [1, 60] |
| xc | 0.497 | [0.1, 4] | [0.025, 4] |
| beta0 | 4.35 | [2.8, 16] | [2.8, 16] |
| alpha_m_P0 | 0.154 | [-0.2, 1.5] | [-0.6, 1.5] |
| alpha_m_xc | -0.00865 | [-0.6, 0.4] | [-1, 0.4] |
| alpha_m_beta | 0.0393 | [-0.2, 0.4] | [-0.2, 0.4] |
| alpha_z_P0 | -0.758 | [-4.5, 0.5] | [-6, 0.5] |
| alpha_z_xc | 0.731 | [-1.5, 2] | [-1.5, 3] |
| alpha_z_beta | 0.415 | [-0.5, 1.5] | [-0.5, 2] |

The completed 8k used a uniform proposal in physical values followed by joint guards. Its accepted density was constant on the surviving joint support, but its one-dimensional marginals were not uniform. About 2.20 percent of that particular proposal survived; the earlier 1.70 percent figure belongs to a different base proposal and must not be reused for this dataset.

Preferred candidate to test: independent physical uniforms in all nine extended ranges, with the intended finite spherical gas radius. Finite integrals permit this mathematically. Full numerical stability, catalogue-scale costs and observable accuracy across the rectangle are still unverified. No further narrowing or FLAMINGO-fit weighting was introduced here.

Alternative if untruncated finite energy is retained: the separate continuous beta sampler has flat one-dimensional marginals but correlated joint support. Its 48³ whole-cell construction was validated on the actual catalogue hull, with 131,072 draws and no energy violations. This is not a fully independent rectangular beta prior or a complete new nine-parameter production configuration.

FLAMINGO overlays remain effective spectrum fits, not nine independent pressure measurements. The new spherical model requires new clean fits before those coordinates can be interpreted with it.


## Beam suppression and where sampling errors matter

![Same refined cache grid; raw 4096 and 8192 compared with raw 16384. All clean comparisons include the existing 2 arcmin beam and fsky = 0.4 mask. The FLAMINGO curve is the HalfDome effective-fit model, not the hydrodynamic map. Noise power is shown separately from the bandpower uncertainty.](plots/beam_noise_resolution.png)

Bℓ² = exp[-ℓ(ℓ+1)σb²],   σb = (2 arcmin in radians)/√(8 ln 2).

The retained power is 0.783 at ℓ = 2000, 0.376 at 4000, 0.111 at 6000 and 0.0205 at 7979. These are power factors, not the beam amplitudes. The convention agrees with healpy gauss_beam [R2].

For B12, the last-bin raw-4096 discrepancy is 18.29 percent of the signal but only 0.0527 of that bin’s noise standard deviation. The largest noise-scaled discrepancy is instead 3.24 at ℓ ≈ 4180. A large fractional error in a nearly invisible bin is therefore a poor numerical guardrail.


## An accuracy metric tied to the actual statistic

For two implementations of the same physical model, form the difference ΔD of the same 40 beam-smoothed, masked bandpowers and evaluate:

ε = √(ΔDTC-1ΔD).

A proposed allowance ε = 0.1 is an explicit precision budget, not a law of physics. Report sensitivity to 0.05 and 0.3. For equal-covariance Gaussian likelihoods, the mean-shift KL divergence is ε²/2; in a local linear identifiable model, the Fisher-metric parameter shift cannot exceed ε. Those statements motivate the metric, but do not certify a nonlinear SBI posterior.

| Maximum ell | B12 ε4096 | B12 ε8192 | B12 amplitude information kept |
| --- | --- | --- | --- |
| 1079 | 0.271 | 0.008 | 15.22% |
| 2079 | 0.877 | 0.020 | 24.40% |
| 4079 | 7.283 | 0.272 | 87.97% |
| 6079 | 9.730 | 0.381 | 99.95% |
| 7979 | 9.789 | 0.383 | 100.00% |

Across all bins, ε4096 is 9.79 for B12 and 1.88 for the FLAMINGO fit; ε8192 is 0.383 and 0.0260 respectively. Thus raw 8192 strongly reduces the measured error, but B12 still exceeds the proposed 0.1 budget. Raw 16384 is a comparison reference, not a proven continuum solution.

Discarding bins above ℓ = 6079 loses only 0.045 percent of the calculated B12 amplitude information, yet leaves ε4096 = 9.73. The dominant sampling problem has already entered lower multipoles. The information calculation uses ∂D/∂ln P0 = 2D with all other parameters fixed; it is not a nine-parameter information or posterior forecast.


### Limits of the covariance

C comes from 64 independent split-noise realizations on one fixed B12 sky, using shrinkage for its 40-bin correlations. It excludes cosmic variance, foreground/model discrepancy and variation in the underlying halo catalogue. Its use for the FLAMINGO fit is a common reference metric. A new-run certification must check covariance sampling uncertainty and relevant signal dependence. Do not label the ε values as a detection significance or a measured posterior bias.


## Measured NSIDE cost: compare identical cache grids

| Raw NSIDE / grid | B12 time | FL fit time | B12 / FL peak GiB |
| --- | --- | --- | --- |
| 4096 / old | 5.50 min | 5.63 min | 10.26 / 10.27 |
| 4096 / fine | 9.19 min | 9.30 min | 12.10 / 12.12 |
| 8192 / fine | 18.64 min | 18.77 min | 19.19 / 19.28 |
| 16384 / fine | 55.78 min | 55.83 min | 55.52 / 55.85 |

![Saved eight-thread clean cluster experiments. The fine cache takes about 4.3-4.4 minutes independently of raw NSIDE. These tests return the final map to NSIDE 4096 and use the same harmonic bandlimit.](plots/resolution_costs.png)

Matched result: raw 8192 takes 2.02-2.03 times the total clean-test time and 1.59 times the process peak memory of raw 4096 on the fine grid. The earlier approximately 3.4 ratio compared 8192/fine with 4096/old and mixed interpolation-grid cost with resolution cost.

Doubling NSIDE increases raw pixel count fourfold. A float64 map is 1.5 GiB at 4096 and 6 GiB at 8192. Keeping the final beam-smoothed/noise map at 4096 avoids making every downstream map four times larger. The physical beam and observed angular information are unchanged; the purpose is more faithful rendering.

The completed noisy 26-thread production had a 261.45-second median and 602.39 worker-hours for 8,169 generated maps plus 23 imported spectra. Those settings differ from this clean experiment. New spherical, pre-beam and full-noise runtime forecasts require the matched pilot in P07.


## What the completed SBI analysis actually establishes

![Normalized held-out RMSE for the completed guarded prior. The black control always predicts the training-pool parameter mean and never reads the observation. MAF results are taken from the preserved convergence tables.](plots/prior_only_baseline.png)

| Estimator / control | Mean RMSE / range | Pooled correlation | Test rows |
| --- | --- | --- | --- |
| Prior-mean control | 0.2174 | 0.403 | 843 |
| 40 bins, MAF | 0.1819 | 0.646 | 842 |
| MOPED, MAF | 0.1666 | 0.714 | 843 |
| PCA, MAF | 0.1876 | 0.618 | 843 |

MOPED improves the mean normalized RMSE by about 23.4 percent over the constant prior-mean predictor, so the model has learned some dependence on the spectra. But even a constant prediction gives pooled correlation 0.403 when parameters with different marginal means are concatenated. Its within-parameter correlation is undefined. Pooled correlation alone is therefore not evidence of successful parameter recovery.

The result is compatible with weak directions and broad degeneracies in a nine-parameter fit to one power spectrum. A posterior mean displaced from B12 need not be a network failure if that truth sits along a broad prior-sensitive degeneracy. Flat priors are the requested design choice, but this observation alone does not prove they will eliminate inference bias.

The 40-bin table includes 842 of 843 test rows after its finite-result and minimum-sample filtering. The next analysis must record every failed posterior rather than omit it without an outcome. New tests should use per-parameter metrics, known-truth recovery, density scores and a shuffled-pair training control.


## Noise cross-spectra and realistic SO observations

Two maps of the same sky with independent noise can be cross-correlated to remove the mean noise auto-bias. This is a legitimate observing strategy. For ideal independent Gaussian noise, E[CAB] = S; for a fixed signal and ν effective modes the approximate conditional variance is:

Var(CAB | s) = [S(NA + NB) + NANB]/ν.

Consequently, S < N per mode does not imply that a bandpower is uninformative: many modes reduce its uncertainty. The actual mask couples modes, which is why the measured pipeline covariance is used for the numerical comparison rather than the ideal formula alone.


### Two full-depth draws are not two half-survey splits

The existing mock draws each noise map with the tabulated residual Nℓ. For purely instrumental white noise, splitting a fixed total observing time into equal halves instead gives approximately 2Nℓ in each half. In the noise-dominated limit its cross variance is four times that of two independent full-depth draws. This is a choice of mock definition, not a reason to alter the finished dataset.

A post-component-separation y-noise table can include residual sky foregrounds. They are generally shared between observing splits and cannot be made independent just by choosing different random seeds. Their cross power can remain in the mean, and their scaling need not follow observing time. Do not simply double the full ILC table without identifying its components. The SO forecast is context [R3]; the exact mock contract is the saved table plus the simulation code.


### Independent rows and FLAMINGO parity

The prior follow-up audited all 16,384 row/split seeds as unique and separate from the two observation seeds. That establishes the intended old mock construction; it does not establish that it includes every real-SO residual or sky-to-sky fluctuation.

For the next run, preserve independent noise between rows and use the same declared beam, mask, noise convention, bins and compression for HalfDome and FLAMINGO. Fit clean FLAMINGO spectra separately from noise. Hydrodynamic y maps contain diffuse gas and cosmology differences that a finite-radius halo model may not reproduce; a good pipeline can correctly expose that mismatch.


## Before 8k: projection and interpolation tests

These are blockers for declaring the new spherical production pipeline ready. Existing cylinder-model results are useful controls, not a substitute for exercising the intended new operator.


### P01 - Freeze the complete forward model

Pin HalfDome and XGPaint sources, spherical radius, physical M200c/h conversion, cosmology, pressure-electron factor, catalogue, redshift range, raw/output NSIDE, beam, mask, noise prescription and bin edges.

Acceptance: One machine-readable manifest; intentional differences from the old run enumerated. Do not mix the truncation comparison h=0.6774 with the SBI h=0.68.

Evidence now: Source review completed; new-run manifest not frozen


### P02 - Spherical LOS and integrated-Y identities

Compare chord projection with independent quadrature, a uniform-pressure sphere, and 2pi integral b*y(b) db versus the 3D pressure-volume integral. Probe centre, grazing chords and outside support.

Acceptance: Agreement at 1e-8 relative where well scaled; absolute error bounds near zero; exact zero outside the sphere. This is a numerical identity tolerance.

Evidence now: B12 wrapper and separate finite-radius tests exist; broad-prior integrated-Y suite required


### P03 - Remove hidden dependence on the old LOS integral

Use the inner model's prepared amplitude rather than inner(theta200)/old_column. Exercise beta<=0.7, tiny xc, high beta and all cache corners.

Acceptance: No old beta assertion, 0/0 normalization, NaN or quadrature stall. Any underflow is bounded in observable units.

Evidence now: Issue identified in wrapper; proposed production change not applied


### P04 - Interpolation and boundary reconstruction

Cache positive chord-mean values, multiply by the exact chord outside interpolation, compare linear-z and log-z grids and two refinements over adversarial points plus at least 10000 random columns.

Acceptance: Observable error plus measured reference change below chosen budget on full-map controls; convergence near the moving sphere boundary documented.

Evidence now: Old-cylinder log-z tests passed selected points; spherical extended-prior tests pending


## Before 8k: rendering, resources and the prior


### P05 - Flux-conserving treatment of compact haloes

Compare point painting, pixel-integrated or pre-beam painting at 4096, and raw 8192/16384 controls. Randomize subpixel halo positions. Apply the beam once; include its wings beyond the physical sphere.

Acceptance: Integrated Y and retained multipoles stable under position/resolution changes; no pixel hit/miss bias. Full-map numerical budget passes at tested points.

Evidence now: Old painter resolution defect measured; alternative renderer not implemented


### P06 - Joint-extreme and catalogue-scale fidelity

Use B12, updated clean FLAMINGO fits, each tail, and 16-32 joint stress cases; compare clean spectra with the same beam/mask at two resolutions, with selected 16384 references.

Acceptance: No accepted point silently dropped; failures resolved or support/density explicitly redesigned. A sphere 4 versus sphere 8 test is model sensitivity, not numerical convergence.

Evidence now: Previous cylinder tests available; new spherical model pending


### P07 - Matched timing and memory pilot

Measure cold/warm cache, painting, transforms, noise, masking, compression and I/O separately at 4096 and 8192 with identical settings, threads and repeat count.

Acceptance: Per-row p50/p95/p99 and peak RSS recorded; memory-limited concurrency and 524288-row storage estimated from measurements.

Evidence now: Matched clean eight-thread timings exist; complete proposed pipeline pending


### D01 - Flat-prior density and failure accounting

Declare independent physical uniforms or a correlated flat-marginal density. Test all nine marginals, pair coverage, normalization and sampler/log_prob agreement. Keep sampling independent of FLAMINGO fit quality.

Acceptance: No undocumented rejection, timeout filtering or Gaussian/log weighting. Exact ranges retained unless a documented support revision is chosen.

Evidence now: Continuous correlated beta sampler verified; full nine-parameter spherical prior not certified


## Before 8k: data and inference-interface integrity


### D02 - Noise splits and observation parity

Unique row/split seeds; independent train/test/observation seeds; matched beam/noise conventions; split auto/cross spectra and shared foreground residual tests.

Acceptance: Means/covariances agree with the declared mock. Decide idealized table draws versus physical observing splits; do not silently double the full ILC table.

Evidence now: Old independent-noise design verified; new operator parity pending


### D03 - SBI serialization, units and row alignment

Round-trip known cases through stored clean/noisy spectra, signed asinh transform, compression and inference; verify parameter order, D_l/C_l factors and array indexing.

Acceptance: Exact row labels and reconstruction within numerical precision; negative cross bins retained. Every test row has a recorded outcome.

Evidence now: New-pipeline regression suite pending


### D04 - Compression fit isolation and rank

Fit scaling/PCA/MOPED only on optimization rows; compare finite-difference and local-regression derivatives, covariance whitening and singular values at B12 and additional anchors.

Acceptance: No test/observation leakage; no hidden regularization manufacturing nine constrained directions. bins40 retained as a reference.

Evidence now: Previous local Fisher rank six is a warning, not a rank certificate for the new model


### D05 - Resume and scale reproducibility

Generate the same small prefix with count 8192 and 524288; simulate interruption/resume, failed rows and imported outputs.

Acceptance: Identical prefix theta and seed IDs; checksums prevent reuse across geometry/beam changes; writes are atomic and no incomplete row counts as success.

Evidence now: Old workflow tested; new geometry identity must be added

A changed physical sphere or renderer invalidates reuse of old clean spectra as new-model rows. Parameter vectors, catalogue data and other unchanged inputs may be reused with provenance, but output reuse requires the complete forward-model identity to match. Hold-out copies must be grouped by clean sky/parameter setting when repeated noise draws are added.


## Small-pilot inference controls, then diagnostic 8k


### I01 - Noiseless positive controls

Fit P0 only and P0-beta at fixed remaining parameters against direct likelihood/grid controls; add noise progressively using a small isolated pilot.

Acceptance: Known truths recovered in identifiable settings; disagreement traced before nine-parameter production.

Evidence now: Required new pilot


### I02 - Prior-only and shuffled-data controls

Evaluate the constant prior-mean predictor and prior density; train the same architecture after shuffling parameter-spectrum pairings.

Acceptance: Correctly paired data improve held-out density scores and identifiable parameter recovery over both controls. Weak parameters may legitimately remain prior-like.

Evidence now: Prior-mean baseline computed in this review; shuffled training pending


### I03 - Repeated-noise truth recovery

Use 16-32 known parameter settings spanning interior and tails, with 16-32 independent noise realizations each. Keep all copies of a clean sky in the same split.

Acceptance: Report conditional bias/coverage per truth and parameter, with binomial intervals; no focus solely on B12 or pooled correlation.

Evidence now: Planned for new diagnostic dataset


### I04 - SBC plus information-sensitive validation

Run at least 256 prior-predictive SBC cases and posterior predictive checks; assess conditional coverage, observation dependence and posterior-to-prior information gain.

Acceptance: Calibration assessed with finite-sample/simultaneous uncertainty; SBC alone is insufficient because a prior-only predictor can pass it.

Evidence now: Not done for the completed 8k analysis

Uniform SBC ranks alone are insufficient: a method that always returns the prior can pass unconditional SBC while ignoring observations. Pair calibration with proper held-out density scores, likelihood-positive controls, conditional coverage and observation dependence.


## What the diagnostic dataset must decide


### I05 - Nested learning curves and estimator robustness

Use fixed held-out truths and nested training sizes 256,512,1024,2048,4096 and the remaining pool; compare bins40 and MOPED with multiple training seeds.

Acceptance: Improvement survives seed variation and exceeds prior/shuffled controls; report failures and per-parameter metrics with uncertainty.

Evidence now: Old curves exist; rerun required after model/prior changes


### I06 - FLAMINGO is an out-of-family observation test

Reprocess with exactly the new observation operator, refit clean effective parameters if desired, and check spectral support/posterior predictive residuals before interpreting parameters.

Acceptance: No claim of parameter coverage using effective FLAMINGO gNFW fits as truths. Cosmology/diffuse-gas/model discrepancies are distinguished from network error.

Evidence now: Old observations exist; new-model comparison pending


### I07 - Decision to scale

Combine physics, rendering, prior, noise, calibration, information and resource results from the diagnostic 8k run.

Acceptance: Launch larger production only after declared numerical budgets, inference controls and operational cost limits pass; extra samples cannot repair a wrong forward model or unidentifiable statistic.

Evidence now: Future decision


### Recommended sequence

First: repair the spherical normalization dependency in an isolated implementation, test its integral identities and broad-prior columns, then compare a flux-conserving 4096 renderer with raw 8192 and selected 16384 controls. No currently measured resolution is an automatic certificate for every extended-prior point.

Next: freeze the physical-uniform density, complete the small inference controls and benchmark the entire noisy pipeline. Use 8,192 as the diagnostic count, with prefix-stable parameter/noise IDs and 524,288 as an adjustable larger target. The repeated-noise suite should be reserved explicitly within that design or budgeted as additional noise processing of saved clean maps or harmonics, not silently counted twice.

Then: assess conditional bias, calibration, prior information and learning curves before scaling. If nine parameters remain weakly identified, investigate the statistic or scientific model; more training examples alone do not create information absent from the observation.


## Files created, reproduction and evidence sources

All new analysis code and small outputs live under SBI_analysis/tsz_next8k_review_20260920/. No existing production source or historical result was edited for this review.

| New file | Purpose |
| --- | --- |
| analyze.py | Recompute beam/noise comparisons, matched timing tables and the prior-mean inference baseline from saved outputs. |
| test_plan.json | Editable structured specification of the 19 tests, pass criteria, current evidence and stages. |
| build_report.py | Generate this report, REVIEW.md and PRE_RUN_TESTS.md from measurements and the test plan. |
| finalize.py | Hash consumed inputs and delivered outputs; check document text and record page inspection. |
| results/ and plots/ | Machine-readable numbers, bandpower CSV and three presentation-ready figures in PNG/PDF. |
| inputs/ | Pinned public XGPaint source snapshots used in the review. |
| artifact_manifest.json and changed_files.txt | Hashes, input provenance and inventory for this isolated review. |


### Reproduction

Run python analyze.py in an environment with NumPy and Matplotlib. Its default entrypoint refreshes pinned public source snapshots over HTTPS. Run python3 build_report.py in the WSL environment containing ReportLab and DejaVu fonts. Source data are read from the preserved relative repository paths, so this review is reproducible within the existing workspace rather than a standalone copy of the large simulations. After inspecting rendered pages, run python3 finalize.py to refresh hashes and the file inventory.


### Local evidence

[L1] flamingo_linear_prior: prior.py, prior.json, stable_los.jl, paint_row.jl, independent_noise.jl and cluster_results/completed_20260918/.[L2] tsz_guardrail_study: METHODS.md, RESULTS.md, map_experiment.jl, audit/noise_covariance.npz and maps/{Battaglia12,FL_L1_m9}/.[L3] tsz_beta_flat_followup_20260920: RESULTS.md, continuous_flat.py and results/.[L4] linear_prior_sbi: SBI_LINEAR_8K_ANALYSIS_20260918.md, sbi_linear_prior_pipeline.py and cluster_results/figures/convergence_metrics.csv.[L5] truncation_comparison/spherical_truncation_profiles.jl and frb_map_generation/compare_takahashi_sightlines.py.

The manifest records exact source and measurement hashes. Statements about older tests are based on those preserved outputs; unperformed tests are explicitly marked pending in the plan.


## References and interpretation limits

[R1] HEALPix: Discretisation of Functions on the Sphere

Sampling rationale and pixel-count convention.

[HEALPix: Discretisation of Functions on the Sphere](https://healpix.sourceforge.io/html/intro_Discretisation_Functions_on.htm)

[R2] healpy: gauss_beam

Gaussian beam amplitude versus power convention.

[healpy: gauss_beam](https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.gauss_beam.html)

[R3] Simons Observatory Collaboration, Science goals and forecasts (2019)

Primary forecast context; the local saved residual table defines this particular mock.

[Simons Observatory Collaboration, Science goals and forecasts (2019)](https://arxiv.org/abs/1808.07445)

[R4] HalfDome spherical-truncation update, commit 3522b71

User-confirmed wrapper implementation reviewed here.

[HalfDome spherical-truncation update, commit 3522b71](https://github.com/kristero/HalfDome_kSZ/commit/3522b71)

[R5] XGPaint public cluster source, commit 5dd0b57

Pinned upstream profile integration and amplitude definitions.

[XGPaint public cluster source, commit 5dd0b57](https://github.com/kristero/XGPaint.jl/tree/5dd0b57cae243598cef9608de77f6807689db712)


### What is established

The published source implements continuous projection before point painting. The fetched HalfDome wrapper supplies chord-limited spherical geometry. Saved matched-grid experiments quantify a large reduction in the existing sampling discrepancy at raw 8192, and their timings establish the quoted clean-run cost ratios. The completed SBI results improve on a matched prior-mean baseline.


### What remains to establish

The new spherical model, full independent flat rectangle, improved compact-halo renderer and noise/observation contract have not jointly passed a full-sky preflight. The report is a source audit, measured reanalysis and staged test specification, not a claim that the next 8k job is ready to submit.

