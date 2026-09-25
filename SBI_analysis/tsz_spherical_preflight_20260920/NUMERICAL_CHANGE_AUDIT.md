# Numerical changes from the guarded linear prior to the spherical diagnostic

This audit distinguishes the prior measure, physical projection, interpolation
and sky pixelization. Removing a prior cut does not fix an inaccurate renderer.
The prepared 256-row run remains unsubmitted.

## What changed, beyond prior bounds

| Component | Guarded linear-prior implementation | Prepared spherical diagnostic | Interpretation |
|---|---|---|---|
| Pressure support | LOS integral to 100,000 R200c, then a projected disc cut at 4 R200c | LOS stops at the sphere chord sqrt(16-x²) | Physical model change; adopted from HalfDome commit 3522b71 |
| Cached quantity | Projected column | Column divided by its chord length; multiply by exact chord after interpolation | Avoid interpolating the moving zero at the sphere edge; already part of 3522b71 |
| Redshift coordinate | Linear z, 0.001..5 | Log z over the same endpoints | Numerical change concentrating nodes at low z |
| Default grid | 512 angle × 256 redshift × 128 mass | Same node counts | No default increase in total grid resolution |
| Angular/mass coordinates | log(theta), log10(M/Msun), angular padding 256 | Same coordinates, padding and M=10^12..10^15.7 Msun | No new angular central repair |
| Refined test grid | Separate comparison | 1024 × 512 × 256 tested | Eight times as many cells; not the prepared diagnostic default |
| Interpolation kernel | Cubic B-spline of logarithmic values | Same cubic B-spline, now of chord-mean values | No different polynomial order |
| tSZ amplitude | Original pressure amplitude; first sphere wrapper inferred it by dividing two old projected quantities | Direct XGPaint.prepare_profile_slice(...).amplitude | Avoid redundant old-LOS evaluation and possible 0/0; physical P0 normalization retained |
| Finite-chord integration | Initial sphere wrapper used direct LOS and fixed multiplier 1e9 | sinh LOS coordinate, log integrand, profile-dependent peak scaling; log-radius central integral | Numerical stability over compact and steep shapes |
| Chord edge | Initial wrapper approximated short chords when L<1e-6 | sqrt((X-x)(X+x)); integrate all positive chords | Removes an arbitrary approximation threshold |
| Positive cache floor | max(minimum_positive×1e-6, nextfloat(0)) for nonpositive entries | Reject nonfinite/negative cells, then floor positive underflow tails at 1e-300 | Floating-point device; not an astrophysical prior or proof of interpolation fidelity |
| Angular sampling | Pixel-centre evaluation | Pixel-centre evaluation | Small-halo aliasing is still present |
| Beam | Harmonic 2 arcmin beam after raw painting | Same; all resolution controls output at 4096 | Pre-beam quadrature is a separate tested prototype, not deployed here |

The old linear-prior `stable_los.jl` already used l=x sinh(u), peak-scaled
quadrature and strictly positive cache cleanup. Those are not entirely new
inventions in this preflight. The new work adapts stable integration to a finite
sphere and removes the old-amplitude dependency and old slope assertion.

The earlier guarded LOS routine asserted raw beta>0.7 and used rtol=1e-12,
quadrature order 9, an explicit 4096-evaluation cap and an error-estimate check.
The finite-sphere routine uses rtol=1e-10 and order 9 (as did the initial sphere
wrapper). It currently uses QuadGK's default evaluation limit and discards the
returned error estimate. Independent convergence/reference tests are therefore
essential; a bounded production quadrature with explicit error reporting remains
an implementation check to resolve. These settings must not be described as
identical quadrature policies.

## Exact normalization

For x=impact parameter/R200c and X=4,

    J(x) = 2 integral_0^sqrt(X²-x²) p(sqrt(x²+l²)) dl.

At x>0 set l=x sinh(u), r=x cosh(u), dl=r du. If g=log[r p(r)] and
c=max(g) over the finite interval, compute K=integral exp(g-c) du and restore
J=exp(log(2K)+c). At x=0 use log(r), which is integrable for gamma=-0.3.
This numerical normalization never changes the physical amplitude A in y=A J.
In particular y remains proportional to P0, and its power to P0².

The beta convention is explicit: beta_internal=alpha*beta_raw-gamma. With
alpha=1 and gamma=-0.3, the 3D shape is q^-0.3 (1+q)^-beta_raw. Finite pressure
support permits low evolved beta without an infinite-radius energy divergence.
This makes a flat independent beta prior mathematically possible; it does not
prove every pressure profile is astrophysically realistic. The choice X=4 is a
model assumption whose sensitivity has been tested separately.

## Interpolator evidence

The earlier complete-catalogue cylindrical experiment isolated the redshift
coordinate change from sky pixelization. For the low alpha_m_xc example, the
historical-to-refined-log-z spectrum difference was 2.113 in the saved conditional
40-bin noise metric; coarse-log-z to refined-log-z was 0.001649. For the high
alpha_z_xc example those numbers were 0.66575 and 0.02514. This motivates log-z;
it is not evidence that all joint extremes have converged.

In the earlier auxiliary spherical cache tests, doubling all three node counts reduced the
B12 maximum pointwise relative error from 0.923% to 0.262%. The compact test
retained a maximum central-normalized error of about 1.387% at both resolutions
because the same minimum-angle clamp remained. This is a deliberately broad
off-grid profile test, not a 1.387% spectrum or parameter error. **Correction:**
that auxiliary test used angular padding 128 and theta_min=1.01815e-9 rad,
whereas the prepared map uses padding 256 and theta_min=1.01815e-11 rad.
The auxiliary result cannot be assigned directly to the prepared map. At fixed
node count the padding-256 grid also covers a wider range with coarser log-angle
spacing. The test cosmology used Omega_b=0.0486, Omega_c=0.2603, h=0.68;
full map controls retain Omega_b=0.049, Omega_c=0.261, h=0.68. A separate compact
test with the exact prepared padding and cosmology is recorded in
`results/interpolation_matched.json`; both coarse and doubled grids use those
same endpoints. The parameterization of `interpolation_test.jl` now records
the padding, angular endpoints and cosmology rather than leaving them implicit.

The matched compact case has maximum central-normalized errors 0.04013975%
(default) and 0.04013974% (doubled grid), with 214/10,000 angle-clamped probes
at either refinement. The cache contains 6,851 and 50,491 underflow cells,
respectively, on its very broad angular support; all tested columns inside the
painting sphere were finite. This supersedes using the auxiliary 1.387% value
for the prepared settings. It validates this case, not all nine-parameter
combinations or the observable/pixel response.

The 1e-300 floor is chosen to make logarithms representable; its negligible
direct signal contribution does not bound errors interpolating across floored
and non-negligible cells. Such cells and angular-floor hits are recorded.

## Is 4096 enough for nine-parameter SBI?

**Do not accept the present raw 4096 point painter for precision constraints
over this prior on the current evidence.** This is different from declaring
that an output map at 4096 is fundamentally insufficient.

The new spherical B12 full-catalogue comparison (four CPUs, same coarse log-z
cache, beam, mask and output grid) has completed at 4096 and 8192:

| Raw NSIDE | Wall time | Peak resident memory |
|---|---:|---:|
| 4096 | 958.3 s | 9.43 GiB |
| 8192 | 2558.5 s | 19.91 GiB |

Their maximum fractional 40-bin difference is 17.50%; the covariance-whitened
distance is 9.410 using the saved fixed-B12 split-noise covariance. These are
beam-smoothed, masked spectra. The covariance omits cosmic variance and model
discrepancy. This is not a measured 9.410-sigma parameter bias, nor an unbinned
MOPED posterior result. The 16384 spherical control is still pending.

The beam cannot reconstruct flux missed before pixelization. Positive
continuous pre-beam convolution worked well in isolated-halo tests, but its
full-catalogue implementation has not been validated. Increasing raw NSIDE is
another possible convergence route; 8192 is not certified for every joint
extreme either. A finite interpolation cache error is a separate issue again.

For the actual unbinned MOPED analysis the relevant validation is to propagate
matched-resolution spectra through the *same fixed fitted transform*, estimate
their compressed noise covariance, and check resolution-induced shifts in the
identifiable parameter combinations and held-out posteriors. Re-fitting an
unrelated compressor at each resolution could conceal a rendering change.
Fixed-simulator mock recovery alone also cannot reveal a systematic error that
is shared by both training and mock observations.

## Unbinned MOPED correction

The prepared analysis now reads all 7900 individual signed D_ell values at
ell=80..7979 from each row's retained raw C_ell file. It does not reconstruct
multipoles from 40 bins. FLAMINGO observations include the same unbinned arrays.
The 40-bin and PCA paths are separate comparisons.

Signed-asinh scaling is fit on optimization rows only, as in the existing
unbinned analysis convention. A local linear regression supplies derivatives
and paired residuals supply an OAS-shrunk noise covariance. This 256-row pilot
cannot support the existing larger pipeline's 55-term quadratic regression,
which requires at least 600 optimization rows. The compression is therefore
explicitly labelled approximate regression-MOPED.

`unbinned_moped.py` evaluates the OAS inverse with a small row-space Woodbury
solve. It is algebraically the same covariance operation, not an approximation
by binning or discarding multipoles. Dense-oracle relative errors were below
2.5e-16. A 7900-feature test recovered nine constructed within-bin shape modes
whose weighted 40-bin signal was identically constant. Its compressed covariance
identity error was 2.9e-15 and Fisher identity error 1.1e-15. The full analysis
also passed a synthetic execution check; none of these tests is a physical
nine-parameter inference validation.

## Files changed in this update

- `diagnostic_analysis.py`: full-spectrum loading, rebin parity, unbinned MOPED routing.
- `unbinned_moped.py`: new memory-bounded OAS inverse and MOPED basis.
- `diagnostic.py`: explicit unbinned-input contract in the prepared manifest.
- `stage_observations.py`: retain the original FLAMINGO multipoles in prepared observations.
- `test_unbinned_moped.py`, `smoke_analysis.py`: algebra, within-bin information and execution checks.
- `verify_prepared.py`: cluster source and observation-dimension verification.
- `analyze_fullsky.py`: report available matched-resolution pairs before 16384 finishes.
- `interpolation_test.jl`: explicit padding/cosmology and separate matched compact tests.
- `PREPARED_256.md`, this audit and report builder: current scientific and operational specification.
- `deploy.py`, `prepare_cluster.py`: include the revised documentation in cluster preparation.
- `finalize.py`: updated validation status and artifact inventory.

The spherical projection implementation itself is not changed in this update;
its previous edits are audited above against the initial spherical wrapper and
the guarded linear-prior baseline.
