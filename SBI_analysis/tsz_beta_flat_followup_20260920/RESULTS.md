# Beta priors, outer pressure radii, pixel sampling and observing splits

This is a separate experiment under `/lustre/work/kristero10/tsz_beta_flat_followup_20260920`. It does not replace the 8192-row production dataset, prior, noise seeds or map operator. All generated figures are provided as PNG and vector PDF.

## What the unused cache corner test means

The interpolation cache spans physical masses from 10^12 to 10^15.7 Msun and redshifts from 0.001 to 5. The selected catalogue occupies a smaller, correlated region: log10(M/Msun) approximately 12.8167--15.5781 and z approximately 0.0021--3.8555. A rectangular cache therefore includes combinations absent from the catalogue.

The stress case called `joint_size_max` has parameters, in the recorded nine-parameter order:

```
P0, xc, beta, alpha_m_P0, alpha_m_xc, alpha_m_beta,
alpha_z_P0, alpha_z_xc, alpha_z_beta

35.5682488168, 3.62424920513, 9.99346818402,
0.760150257591, -0.977621512488, 0.0892194559798,
-3.10822514631, 2.98911710875, -0.381467691157
```

At M=10^12 Msun and z=5, it gives xc=69254.2281 and beta=3.34532073. At projected radius 4R200, doubling the LOS half-length from 10^5 to 2x10^5 R200 changes the column by 4.44119%. This is a deliberately extreme extrapolation, not an actual catalogue halo.

The separate occupied-support experiment evaluated 65 shapes at 876 catalogue convex-hull vertices and 64 occupied-bin representatives, at two projected radii: 122200 tests. The worst first-doubling change was 2.92953e-6 in fractional units, or 0.000292953%; the next doubling changed it by 2.38500e-7. This is empirical evidence for those tests, not a proof that the nonlinear LOS error reaches its maximum at a hull vertex. Evolved power-law parameter extrema, unlike LOS errors, can be bounded exactly using the convex hull in log mass and log(1+z).

The cache-corner result should not become a blanket astrophysical exclusion. Nevertheless, a cache must either compute those nodes reliably or be redesigned: an unused halo coordinate can still enter the interpolation stencil near occupied support. Old evidence is retained in `../tsz_guardrail_study/audit/los_catalogue_support.json`.

## What beta convergence actually requires

Using the raw Battaglia exponent, the pressure shape is

P/P200 = P0 q^(-0.3) (1+q)^(-beta), q=r/(xc R200),

with beta(M,z)=beta0 (M/10^14 Msun)^alpha_m_beta (1+z)^alpha_z_beta.

The outer slope is therefore -(beta+0.3), not -beta. The central cusp is integrable both in volume and along the central LOS.

At large radius, the thermal-energy integrand r^2 P scales as r^(1.7-beta). Total energy extrapolated to infinity is finite iff beta>2.7. The infinite-LOS integral is finite iff beta>0.7. Equality gives logarithmic divergence. These conditions apply to extrapolation to infinity, not to a profile confined to a finite sphere or the current finite cylindrical painting volume.

Physics does not prescribe a Gaussian beta prior. A constant joint density inside a restricted support generally has nonuniform marginals because the available cross-sectional volume varies with each parameter. The old size and Y guards add further distortions.

For the new comparison, the beta rectangle is beta0 in [2.8,16], alpha_m_beta in [-0.2,0.4], alpha_z_beta in [-0.5,2]. A 262144-point scrambled Sobol scan, seed 20260920, gives:

| Requirement | Fraction of beta rectangle accepted |
|---|---:|
| Infinite-energy convergence over full cache | 67.3744% |
| Infinite-energy convergence over actual catalogue hull | 85.1257% |
| Finite-radius mathematical integrability | 100% |

The third row is an analytic integrability statement for positive finite parameters, not a full numerical or astrophysical certification. The fraction with beta<=0.7 somewhere is 1.74408% on the cache and 0.0423431% on the catalogue hull.

![Beta priors](plots/beta_priors.png)

The gray distribution is the actual existing 8k design, including its original joint guards and its narrower alpha_z_beta upper bound of 1.5. Orange and blue apply only the stated infinite-energy support condition to the extended beta rectangle. Green is the independent flat target for a finite-radius model. No FLAMINGO weighting or Gaussian/lognormal weighting was used.

There are two distinct routes to flat marginals:

1. **Retain untruncated-energy support, introduce correlations.** Under the full cache condition, exact flat marginals across the entire requested range are impossible: even beta0=16 and alpha_z_beta=2 cannot support alpha_m_beta greater than log(16*1.001^2/2.7)/log(100)=0.38681219 at M=10^12 Msun, z=0.001. The interval up to 0.4 has no allowed combinations. On actual catalogue support, iterative proportional fitting on 48^3 cell centres converged in 15 iterations to marginal relative error 6.34e-9. That demonstrates a discrete correlated construction; it is not yet a continuous sampler with guaranteed support throughout each cell.
2. **Choose a finite physical outer radius.** The beta rectangle can then remain independent and uniform without the infinite-energy cut. The radius becomes part of the physical forward model. Positive finite pressure alone does not establish consistency with total thermal-energy observations, gas mass, feedback energetics or pixel accuracy.

![Correlated flat marginal construction](plots/correlated_flat_beta.png)

White regions have zero probability. The displayed joint distributions have equal one-dimensional cell probabilities, despite the correlations. This construction does not restore the old size/Y cuts or certify a new nine-dimensional production prior.

The follow-up `continuous_flat.py` strengthens this into a continuous sampler. It retains only cells for which the minimum over every point of the parameter cell and every catalogue-hull vertex exceeds 2.7. This minimum is analytic: use the lower beta0, the lower alpha_z_beta, and the mass-exponent endpoint selected by the sign of log(M/10^14 Msun). Since log beta is affine in log mass and log(1+z), the hull vertices bound all archived catalogue points.

Iterative proportional fitting on these whole allowed cells converged in 26 iterations. All three continuous physical-value marginals are uniform to relative tolerance 9.37e-11. The minimum allowed-cell beta is 2.70004049; 131072 draws gave zero violations and minimum beta 2.71005. The density integrates to 1 within floating-point precision. The retained cells occupy 83.2203% of the rectangle, compared with the approximately 85.1257% exact catalogue-support volume. The small additional excluded boundary volume is a discretization choice; none of the three marginal parameter ranges is narrowed.

![Continuous flat beta marginals](plots/continuous_flat_marginals.png)

![Continuous correlated beta density](plots/continuous_flat_correlations.png)

The `CorrelatedFlatBetaPrior` class exposes `sample` and normalized `log_prob`, in the physical parameter order beta0, alpha_m_beta, alpha_z_beta. It changes the joint prior density, not the pressure formula. Its physical support is restricted to the archived catalogue domain, not arbitrary mass/redshift extrapolation. It has not been installed into production, combined with the other six parameters' numerical checks, or certified for the existing interpolation/pixel renderer. This is a concrete route to the requested flat marginals without choosing a new outer pressure radius.

## Computed profiles that fail the infinite-energy condition

All six non-beta coefficients remain at Battaglia12 values. At M=10^13 Msun and z=2, the examples are:

| Case | beta0 | alpha_m_beta | alpha_z_beta | Evolved beta | E(<16R200)/E(<4R200) |
|---|---:|---:|---:|---:|---:|
| Battaglia12 | 4.35 | 0.0393 | 0.415 | 6.26895 | 1.03232 |
| Shallow outskirts | 2.8 | 0 | -0.5 | 1.61658 | 7.47427 |
| Very shallow outskirts | 2.8 | 0.4 | -0.5 | 0.643572 | 20.6118 |

Each shallow example is a positive, monotonically declining pressure profile. Calling it universally nonphysical would be too strong. Its untruncated isolated-halo extrapolation has infinite energy; this does not prevent calculation of a finite model.

![Pressure and energy](plots/pressure_and_energy.png)

Finite spherical projection uses LOS half-length sqrt(Rout^2-Rperp^2) for Rperp<Rout, and zero beyond Rout. We tested Rout/R200=4,8,16. For comparison, the current operator uses LOS half-length 10^5 R200 and paints only Rperp<4R200. A projected cutoff is not a spherical pressure cutoff. In these plots, every model is shown within the same projected 4R200 aperture; the 8R200 and 16R200 spheres still contain pressure outside that aperture.

![Projected profiles](plots/projected_profiles.png)

At projected R200, changing the sphere from 4R200 to 16R200 changes y by factors 1.00297, 1.31973 and 1.94506 respectively. For the shallowest case, the current long-LOS value is 10.57698 times the 4R200-sphere value. Doubling the long LOS changes it by another 8.39528%. This illustrates why simply removing the beta assertion while retaining an arbitrary very long endpoint would create a cutoff-dependent model.

The upper beta tail behaves differently: it is finite and increasingly compact. A separate plot evaluates beta0=2.8,4.35,8,16 at M=10^14 Msun, z=0.5, with all eight other coefficients fixed at Battaglia12. beta0=16 is not excluded by a divergence theorem; pixel fidelity must be measured for that compact profile and its amplitude.

![Upper beta tail](plots/high_beta_profiles.png)

Rout=4,8,16 are sensitivity experiments, not empirically established physical boundaries. A sharp spherical edge can also introduce harmonic structure. Selecting a truncation or smooth taper for a production dataset requires validation against pressure outskirts and a numerical map-convergence check. P0 rescales energy and y linearly but does not remove the fractional cutoff dependence or cure an infinite-radius divergence.

## Numerical implementation and distinction from existing fixes

The newly written `study.py` transforms ell=x sinh(u), so r=x cosh(u), and normalizes the positive integrand before quadrature. The transformed log-integrand is

g(log r)=0.7 log r + 0.3 log xc - beta log(1+r/xc).

For beta>0.7 its maximum is at r=0.7xc/(beta-0.7), clamped to the integration interval. For beta<=0.7 it increases towards the finite endpoint, which is the correct normalization point. The new endpoint branch allows the finite integral to be computed; it does not reinterpret a divergent infinite integral as converged.

The energy quadrature similarly uses log radius and remains valid below beta=2.7. Its independent check uses the incomplete beta function with a finite upper endpoint and arbitrary precision, where a negative second parameter is allowed.

The chronology is important:

| Change | Status |
|---|---|
| Normalized finite-LOS quadrature for beta>0.7 | Already part of existing 8k production |
| Positive cache floor protected against floating-point underflow | Already part of existing 8k production |
| Linear-z interpolation changed to log-z, with refinement tests | Previous separate guardrail study only |
| Fixed size/Y cuts replaced by explicit analytic support and observable-error checks | Separate candidate only; no production replacement |
| Denser raw HEALPix sampling | Previous controlled map tests only |
| Finite spherical radii and beta<=0.7 endpoint branch | This separate study only |

The positive-floor fix matters because multiplying the smallest positive cache value by 10^-6 can itself produce zero; log(0) then contaminates interpolation. Neither that fix nor normalization changes the intended finite pressure integral. The finite-radius models do change the physical integral.

Machine-readable checks are in `results/julia_verification.toml` and `results/independent_validation.json`. Julia QuadGK is compared against Python QUADPACK at 48 columns. The frozen production override is explicitly tested, including its expected beta<=0.7 assertion. A further 12 LOS checks integrate directly in ell using 60-digit arithmetic; nine energy checks use an independent special function. These are profile-level checks, not full-sky validation of a finite-radius dataset.

## Aliasing and the resolution comparison

The painter evaluates profiles at pixel centres. That is generally different from the area-averaged value of a narrow halo within a pixel. When unresolved high-frequency structure has already folded into the sampled map, smoothing that map afterwards cannot reconstruct the original continuous profile. A pixel-window correction is also not a universal inverse of point-sampling errors. HEALPix defines the pixelized signal through pixel averaging: [pixel-window documentation](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm).

The statement that healpy contains no anti-aliasing functionality would be inaccurate. Current healpy has `harmonic_ud_grade`, which band-limits before resampling and can handle beam/pixel-window transfer. It can prevent additional downgrade aliasing, but does not infer missing subpixel structure from an already undersampled map. [Official comparison](https://healpy.readthedocs.io/en/stable/healpy_harmonic_ud_grade_comparison.html). The checked cluster environment has healpy 1.16.5 and lacks this function. The production painter itself uses Julia Healpix.jl.

The previous experiment painted at higher raw NSIDE, applied the same 2-arcmin harmonic beam and band limit, synthesized back to NSIDE4096, then used the same mask and bins. The output observable was preserved while testing initial sampling. Accurate subpixel integration or a validated beam-convolved halo painter are alternative remedies; neither has been deployed here.

The tested number is 8192, not 8912. Pixel-area angular scales are approximately 0.859, 0.429 and 0.215 arcmin for NSIDE4096,8192,16384. Previous full-map results:

| Case | Max bandpower change, 4096 vs 16384 | Max change, 8192 vs 16384 | Whitened 8192 vs 16384 distance |
|---|---:|---:|---:|
| Battaglia12 | 18.2887% | 0.669118% | 0.383135 |
| FLAMINGO fiducial fitted profile | 4.84030% | 0.0525434% | 0.0259968 |

Whitened distance means sqrt(delta_D^T C^-1 delta_D), using the previously specified fixed-sky, idealized split-noise covariance; it is not a full SO observational or posterior significance. The 0.1 allowance is a declared numerical-error budget, not a physical constant. NSIDE8192 improves accuracy substantially but Battaglia12 fails that chosen step-size budget. NSIDE16384 is not proven to be the continuum reference. Peak memory was approximately 19.3 GiB at raw8192 and 55.9 GiB at raw16384 for these clean eight-thread experiments, not the complete noisy production workflow.

## Noise cross-spectra and real observing splits

For maps a=s+nA and b=s+nB, independent zero-mean noises give E[C_AB]=C_s. With correlated noises, the expectation includes N_AB. Real observing programmes use split-map spectra: the [ACT data release](https://lambda.gsfc.nasa.gov/product/act/actpol_maps_info.html) explicitly provides four splits used for power spectra. SO-style mocks can use this technique when the split noise and shared components are represented consistently.

The current Julia implementation draws two independent maps from the same baseline standard-ILC table, each with `split_Nell_multiplier=1.0`. These are two realizations of one prescription, not two alternative observing scenarios. For instrumental noise alone, equal observing-time halves have 2N_full each, so the current choice would correspond to two independent full-depth maps rather than two halves of one full-depth survey.

The noise table was checked byte-for-byte against the official [SO component-separated table](https://raw.githubusercontent.com/simonsobs/so_noise_models/master/LAT_comp_sep_noise/v3.1.0/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt). It is an effective Compton-y spectrum, not purely instrumental noise: the forecast combines SO and Planck and includes foreground effects through component separation. Residual common sky emission and reused Planck noise need not be independent across SO splits. Therefore multiplying the entire table by two is not a sufficient physical correction. [SO README](https://raw.githubusercontent.com/simonsobs/so_noise_models/master/LAT_comp_sep_noise/v3.1.0/README.txt), [SO forecast, Sections II.5 and VII.2](https://arxiv.org/html/1808.07445v2).

For independent Gaussian modes and a fixed signal realization, the conditional cross-spectrum variance is

Var(C_AB | s) = [S(NA+NB) + NA NB]/nu.

Random Gaussian signal realizations add 2S^2/nu. A real masked, non-Gaussian tSZ sky also requires mask coupling and non-Gaussian covariance; replacing nu by a simple fsky approximation is not a complete treatment.

We tested 30000 realizations for each of nine cases, with 201 independent real modes and seed 20260921. All mean residuals were within 1.95 Monte Carlo standard errors; the largest variance discrepancy was 1.25%. Equal half-depth splits raise the conditional standard deviation relative to two full-depth splits by sqrt((4SN+4N^2)/(2SN+N^2)), between sqrt(2) and 2. A shared residual of power 0.5 was recovered as a cross-spectrum excess 0.49977 when S=N=1; it did not cancel.

![Noise split check](plots/noise_split_validation.png)

Adding two independent noise draws to the same already-noisy observed map also leaves its original shared noise in the cross-spectrum. Independent data splits must come from independent noise contributions in the observations. The existing dataset is a valid idealized statistical experiment under its stated noise model; these checks do not establish a faithful full SO observing pipeline, and no existing noise implementation was changed.

## Files and reproduction

New source files: `study.py`, `finite_los_test.jl`, `validate.py`, `render.py`, `continuous_flat.py`, `run.pbs`; this report is also new. Inputs are frozen copies under `inputs/`; numeric outputs and validation records are under `results/`; PNG/PDF figures are under `plots/`. Failed test-driver logs are retained separately from final successful validation.

On the cluster, from this experiment directory:

```
qsub run.pbs
```

The job requests one CPU and 4GB in `tiny`. `TEST_STAGE=julia` skips the completed Python study and reruns Julia, independent validation and rendering. The scripts use the existing campaign environment without installing packages or modifying production dependencies. `changed_files.txt` and `artifact_manifest.json` inventory this separate experiment.

## Final verification record

The main profile, beta-support and noise experiments ran in the Python stage of PBS job 598374 on the cluster. Two Julia test-driver issues (dependency import and local scope) and a plotting-keyword incompatibility with the older cluster Matplotlib were fixed; the diagnostic logs are retained. The follow-up queue was saturated; job 598434 was cancelled and the short scalar validation was completed directly on idark with one thread and a 90-second timeout. Plotting and the continuous-prior construction used separate 60-second limits. No production job was changed.

Final Julia record:

```toml
max_supported_production_difference = 6.661338147750939e-16
julia_version = "1.12.2"
passed = true
max_python_julia_relative_difference = 1.7763568394002505e-15
production_assertions = 4
probes = 48
```

Independent 60-digit quadrature maximum LOS relative error: 1.1554e-15; maximum energy relative error: 1.3628e-15. All 77 production hashes were unchanged. The continuous sampler was also rebuilt and checked on the cluster, with 0 sampled support violations.
