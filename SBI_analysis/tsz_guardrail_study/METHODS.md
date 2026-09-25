# Evidence for tSZ prior guardrails

This study keeps the active 8,192-row dataset immutable. Its reference is
`halfdome_flamingo_linear_8192_20260915`, manifest SHA-256
`994333174a0c5a1e636a5cc065d000464ea3b4ddee2bb0082ccb4cfdd70cc600`.
The new directory contains an analytic candidate prior and a separate numerical
admissibility test. An analytic candidate is **not** a certified usable simulation.

## The metric to use

Let **d** contain the original 40 masked, beam-smoothed, clean tSZ bandpowers,
with the original multipole boundaries and `(2 ell + 1)` weighting. For a
candidate implementation and a converged reference, measure

\[
\epsilon_{\rm num}^2=(\mathbf d_{\rm candidate}-\mathbf d_{\rm reference})^T
C^{-1}(\mathbf d_{\rm candidate}-\mathbf d_{\rm reference}).
\]

This compares numerical error with the precision of the observable actually
used for inference. A fractional profile error is unsuitable as the main gate:
it can be large where the signal contributes negligibly, and small errors can
matter for a very bright signal. A core smaller than a pixel is also not an
automatic exclusion: the integrated signal can be measurable, and an accurate
pixel integral can represent a source without resolving its interior.

For a Gaussian likelihood with common covariance, the KL divergence caused by
a shift of the mean is `epsilon_num^2 / 2`. With derivative matrix J, define
`A=C^(-1/2) J` and `F=J^T C^(-1) J`. In the identifiable Fisher subspace,

\[
\delta\theta^T F\delta\theta
=\delta d^T C^{-1/2} A(A^TA)^+A^TC^{-1/2}\delta d
\leq\epsilon_{\rm num}^2.
\]

The middle matrix is an orthogonal projector. Thus an allowance of 0.1 limits
the local shift of any resolved parameter combination to 0.1 of its standard
deviation, under these assumptions. It also corresponds to 0.005 nats of
Gaussian mean-shift KL divergence. This is an explicit accuracy target, **not a
constant of astrophysics**. Results are evaluated at 0.05, 0.1 and 0.3 to expose
the sensitivity to that choice. A finite experiment cannot remove the need to
declare an accuracy budget.

Linear compression, including MOPED, cannot increase this distance if its
covariance is propagated consistently. Testing all 40 bins avoids hiding an
error in directions discarded by a particular set of MOPED weights. The Fisher
argument does not certify nonlinear SBI, weakly identified parameters, or
posterior truncation at a prior boundary. Those require inference calibration.
The use of parameter bias as a systematic-error budget is consistent with
[Amara & Refregier (2008)](https://arxiv.org/abs/0710.5171); the numerical budget
and its application here are our analysis choices.

The current diagnostic covariance uses 64 archived independent SO split-noise
realizations of the same fixed Battaglia12 signal and catalogue. An OAS
coefficient regularizes the correlation matrix; the unbiased measured variance
of every band is preserved exactly. This prevents high-noise bands from imposing
an artificial variance floor on well-measured bands. Two hundred bootstrap
resamples measure sensitivity to the finite noise ensemble. It excludes cosmic variance, cosmology uncertainty,
pressure-model discrepancy and emulator error. Away from Battaglia12 it defines
a **reference precision scale**, not the exact posterior uncertainty of that
candidate: signal-noise covariance changes with signal amplitude and shape.
This distinction must remain in a paper caption and in any reported sigma units.

## The profile and exact mathematical conditions

The code uses physical M200c in solar masses, with pivot mass `1e14 Msun`:

\[
A(M,z)=A_0(M/10^{14}M_\odot)^{\alpha_{m,A}}(1+z)^{\alpha_{z,A}},\qquad
P_{\rm th}/P_{200}=P_0q^{-0.3}(1+q)^{-\beta},\quad q=r/(x_cR_{200}).
\]

`beta` here is the raw Battaglia exponent: the asymptotic slope is
`-(beta+0.3)`. Confusing it with an outer-slope parameter shifts all convergence
conditions by 0.3. The fixed electron conversion is `Pe=0.5176 Pth`. The
profile family follows [Battaglia et al. (2012)](https://arxiv.org/abs/1109.3711).

* Positive amplitudes P0 and xc give positive pressure. The fixed central cusp
  is integrable: the central LOS behaves as `integral r^(-0.3) dr`, and the
  central thermal-energy integral as `integral r^(1.7) dr`.
* At infinity the LOS integral behaves as `integral r^(-beta-0.3) dr` and
  converges if and only if `beta>0.7`.
* The total, **untruncated** thermal-energy integral behaves as
  `integral r^(1.7-beta) dr` and converges if and only if `beta>2.7`.
* A finite-radius energy integral exists below 2.7. Therefore the last condition
  is an isolated, untruncated-halo modelling assumption; it is not required for
  a halo whose physical pressure is truncated at a finite radius.

For beta greater than 2.7, the finite spherical integral to R200 is

\[
I_{200}=P_0x_c^3 B_{1/(1+x_c)}(2.7,\beta-2.7).
\]

The dimensional prefactor cancels in the same-mass, same-redshift ratio to
Battaglia12. Ratios of 0.003 or 30 do **not** appear as singularities in this
expression. Nor does a beta upper bound of 50.

The logarithm of any evolved parameter is affine in `log(M/1e14)` and
`log(1+z)`. Its extrema on a rectangular domain therefore occur at corners;
on the actual catalogue they occur on its convex hull. This gives an exact
test for beta and for power-law size ratios. It does **not** make a 9 by 9
Y200 test a continuous-domain guarantee, because the incomplete beta function
is nonlinear in the evolving shape parameters.

The analytic candidate conservatively requires finite untruncated energy
throughout the declared interpolation domain, `log10 M=[12,15.7]`,
`z=[0.001,5]`. This domain comes from the archived cache, not from a fitted
FLAMINGO contour. The separate convex-hull calculation quantifies how much
less restrictive a requirement on occupied catalogue points would be. Requiring
physical validity at unused cache corners is conservative and must not be
described as an observational constraint.

Parameter combinations matter. For beta much larger than 2.7, the untruncated
dimensionless energy approaches
`Gamma(2.7) P0 xc^3 beta^(-2.7)`. Consequently its approximate evolution
depends on `alpha_m_P0 + 3 alpha_m_xc - 2.7 alpha_m_beta` and the analogous
redshift combination. Changing xc or beta can be offset by amplitude or other
evolution parameters. This asymptote is not a uniformly valid replacement for
the finite-radius integral, especially for very broad profiles. It explains
why separate per-parameter limits and a fixed size ratio can exclude valid
compensating combinations.

## Audit of each existing cut

| Existing restriction | What can actually justify it? | Treatment in the revision |
|---|---|---|
| beta at least 2.8 | An untruncated profile requires beta greater than 2.7. The additional 0.1 margin is discretionary. | Use the strict mathematical inequality, state the untruncated assumption, and measure LOS convergence separately. |
| beta at most 50 | No mathematical singularity occurs at 50. Steep profiles can underflow far outside the painted region. | Remove as a physical bound; test normalized quadrature, cache interpolation and map fidelity at steep examples. |
| size ratio at least 0.4 | Pixel sampling can alias compact profiles; xc/beta relative to another profile contains neither angular-diameter distance nor pixel geometry. | Replace the proxy with resolution convergence of the observable. Do not reject a faint compact profile only because its relative error is large. |
| size ratio at most 8 | Very broad profiles can make a finite painting radius consequential. Width alone does not bound spectral error. | Report radial-truncation sensitivity as a model-definition issue; use direct numerical convergence for the fixed definition. |
| Y200/B12 at least 0.003 | Low thermal signal can be poorly constrained; this is not a positivity or convergence failure. | Remove as a mathematical/physical hard cut. Keep prior-dominated inference distinct from invalid simulation. |
| Y200/B12 at most 30 | A thermal-energy budget could constrain pressure only after assumptions about gas mass, confinement, nonthermal support and feedback are supplied. | Remove this unsupported universal number. Report Y ratios as diagnostics; do not claim every positive finite profile is a realistic hydrostatic halo. |
| central missing LOS fraction at most 1% | The finite LOS integral should converge to its intended reference at the required observable accuracy. A central-column percentage alone is insufficient. | Compare endpoint L and 2L (and further refinements where needed); require full-observable convergence in the numerical gate. |
| finite cache values and positive logarithm arguments | Floating-point representability and the logarithmic interpolator require these. | Keep hard numerical checks. Underflowed irrelevant tails need a safe floor, not an astrophysical beta ceiling. |
| 1,200 seconds per production row | A scheduling/throughput allowance, not a pressure law. | Record runtime and peak memory; failures stop the assigned row. Do not redraw until a fast parameter happens to appear. |

No pressure-only theorem supplies a universal upper Y200 ratio. For example,
imposing hydrostatic equilibrium would infer a gas density from
`rho_g=-r^2/(G M(<r)) dP/dr`. That introduces a mass profile, nonthermal-pressure
assumptions and a gas-mass constraint. It is a different, explicitly specified
astrophysical prior; silently interpreting the old factor 30 that way would be
incorrect. Likewise, [Lee et al. (2022)](https://arxiv.org/abs/2205.01710) find
concentration dependence and broken mass scaling in hydrodynamical simulations.
Their fitted relations support testing a broader model family, but do not
establish universal hard boundaries for this nine-parameter, single-power-law
model. The FLAMINGO markers are effective clean-spectrum fits, including the
previous approximate cosmology correction; they are not measured posterior
constraints on nine independently identifiable pressure parameters.

An independent observational alternative is the sky-averaged Compton y. The
original [Fixsen et al. (1996)](https://arxiv.org/abs/astro-ph/9605054) FIRAS
limit is `|mean y|<15e-6` at 95% confidence. A foreground-method study by
[Sabyr et al. (2025)](https://arxiv.org/abs/2508.04593) reports a pixel-method
limit `mean y<8.3e-6`. The subsequent
[Fabbian et al. analysis](https://arxiv.org/abs/2512.03038) gives
`mean y=(1.2 +/- 2.0)e-6`, or an upper limit `5.2e-6` at 95% confidence.
These provide data-supported checks on integrated thermal energy, with explicit
foreground and statistical assumptions. None is a mathematical pressure bound.

`mean_y.py` measures the model catalogue contribution using independently built
80 by 96 and 160 by 192 histograms in log mass and log redshift. It compares
radial integration orders 96 and 192. For a deliberately conservative diagnostic
it also uses only pressure inside a 3D sphere of radius 4R200, multiplying by
`sin(4 theta200)/(4 theta200)` to bound the conversion from flat angular area
to spherical area. This is a lower contribution than the full projected cylinder;
missing halos, diffuse gas and exterior pressure add positive signal. The bound
is subject to the measured histogram approximation. The approximate full-cylinder
prediction uses an infinite LOS and is labeled separately. These FIRAS checks
are **not applied to the simulation prior** by default, so exploratory FLAMINGO
models are not silently excluded using real-universe data.

## Numerical experiments and what they establish

`audit.py` evaluates 262,144 reproducible Sobol proposals. An initial 675-probe
QUADPACK audit is followed by a 2,275-probe joint stress audit: the specified
examples, six extreme accepted combinations and 32 additional accepted joint
combinations, including all four interpolation-domain corners. The transform
is `l=R sinh(u)` with log-integrand
normalization at its maximum. Tightened tolerances test integration accuracy;
doubled LOS endpoints test a different source of error. Evaluation counts and
wall times are recorded, including underflow in faint, distant tails.

The Julia experiment additionally compares the production stable quadrature
against the native finite-LOS integrand with a bounded evaluation budget. Its
45 probes per instrumented map include painted and unpainted radii. An unconverged native
result is diagnostic evidence, not ground truth. Raw CSVs retain evaluation
counts and native error estimates.

The largest endpoint effect in the cache-domain stress test occurs at an
unused cache corner, M=1e12 Msun and z=5. It must not be described as the
largest error of a halo actually present in the catalogue. `los_support.py`
separately checks the actual catalogue hull vertices and 64 occupied-bin means
at L, 2L and 4L. These probes measure occupied-support behavior, but do not
guarantee a maximum because the relative endpoint error is nonlinear in shape.

`map_experiment.jl` paints the complete selected catalogue (85,224,251 halos)
using the archived cosmology, mass conversion, 4R200 painting radius, finite
LOS endpoint, beam and deterministic mask. It compares:

1. The historical 512 by 256 by 128 grid in log angle, **linear z**, log mass.
2. A 512 by 256 by 128 grid in log angle, **log z**, log mass.
3. A 1024 by 512 by 256 grid in those same logarithmic coordinates.

The coordinate change resolves the rapid low-z variation in angular size with
the same number of grid nodes. It changes the numerical approximation to the
same continuous pressure model. It does not alter the active production code.
Agreement of two log-z grids is an empirical interpolation check, not proof
that HEALPix point sampling or the physical pressure model is converged.

The archived angular grid spans log theta approximately [-25.31045, 11.49494]
in radians. Its very large outer endpoint is far outside the 4R200 painting
region and helps explain why unpainted cache tails underflow. The painter also
clamps separations at the lower interpolation boundary. Our grid-refinement
comparison preserves these boundaries; a complete reference protocol must also
check boundary sensitivity if a newly admitted profile has appreciable signal
there. Neither finite cache values nor finer grid spacing alone establishes that.

`pixel_experiment.py` places individual halos at 256 reproducible random
full-sky positions at NSIDE 4096 and 8192. It compares flux and the mean squared
Fourier amplitude with a continuous projected-profile reference at two masses
and redshifts. The radial integral is checked at orders 256 and 512; its omitted
central area has an explicit upper bound. Projection-table values are checked
against direct finite-LOS integration. It never extrapolates the table.

For unresolved sources, unbiased mean flux does not imply unbiased power:
`E[F_pix^2]=E[F_pix]^2+Var(F_pix)`. The power test therefore occurs before
averaging pixel positions. Standard errors and extreme flux values are stored;
256 placements can still miss rare central hits. These are single-halo,
flat-sky Fourier diagnostics using actual HEALPix pixel centers. They do not
include mask coupling, correlated halo positions or full-sky spectral
convergence. In particular, they cannot substitute for the full-observable
pixel component of the future evidence gate.

There is also a useful analytic resolution diagnostic that does not rely on
rare events appearing in a small Monte Carlo sample. For a nonnegative circular
profile, let `F=integral y dOmega` and define

\[
A_{\rm eff}=\frac{(\int y\,d\Omega)^2}{\int y^2\,d\Omega},\qquad
N_{\rm eff}=A_{\rm eff}/\Omega_{\rm pix}.
\]

Place its center uniformly on the sphere and let
`F_pix=Omega_pix sum_p y_p`. Rotational invariance and
`N_pix Omega_pix=4pi` give `E[F_pix]=F` exactly. In `E[F_pix^2]`, the sum of
diagonal terms is `Omega_pix integral y^2 dOmega`; all off-diagonal terms are
nonnegative. Jensen's inequality supplies a second lower bound. Therefore

\[
\frac{\mathbb E[F_{\rm pix}^2]}{F^2}\geq
\max(1,N_{\rm eff}^{-1}).
\]

This is an analytic bound for point sampling, not a fitted size threshold.
For example, `N_eff=0.1` guarantees a mean squared flux at least ten times the
continuous squared flux. P0 cancels from N_eff, while angular distance, the
full profile shape and pixel area enter correctly. A measured finite-sample
mean below the bound indicates that the Monte Carlo sample has not captured
the rare-hit tail adequately; it does not invalidate the bound. When N_eff
exceeds one, the bound becomes uninformative and cannot certify convergence.
This is a statement about individual-halo **flux squared**. It approximates
low-multipole power for an angularly unresolved halo; it is not a lower bound
on every multipole of the masked full-catalogue spectrum. The main statistical
error metric remains necessary, particularly for faint compact halos.

The [HEALPix pixel-window documentation](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm)
defines pixelized signals as pixel averages and gives an approximate harmonic
window for that operation. The historical painter samples at pixel centers.
Multiplying by the standard pixel window afterwards does not recover signal
lost or aliased by sampling a sharply peaked profile. Applying the Gaussian
beam after painting likewise does not generally undo aliasing already present
in the raw map.

Changing the painting radius from 4R200 to 8R200 is **not** a pure convergence
test of the original simulator. It adds pressure that the original map
definition excludes. Such sensitivity is relevant for interpreting FLAMINGO,
but must be reported separately from numerical interpolation and pixel errors.

## Candidate extension and conditions for a future usable prior

The proposed **exploration** rectangle is:

| Parameter | 8k rectangle | Analytic candidate rectangle |
|---|---|---|
| P0 | [1, 60] | [1, 60] |
| xc | [0.1, 4] | [0.025, 4] |
| beta | [2.8, 16] | [2.8, 16] |
| alpha_m_P0 | [-0.2, 1.5] | [-0.6, 1.5] |
| alpha_m_xc | [-0.6, 0.4] | [-1.0, 0.4] |
| alpha_m_beta | [-0.2, 0.4] | [-0.2, 0.4] |
| alpha_z_P0 | [-4.5, 0.5] | [-6.0, 0.5] |
| alpha_z_xc | [-1.5, 2.0] | [-1.5, 3.0] |
| alpha_z_beta | [-0.5, 1.5] | [-0.5, 2.0] |

These new outer edges are finite experimental coverage choices in the six
requested directions. No mathematical singularity establishes, for example,
xc=0.025 or alpha_z_xc=3 as a physical limit. Calling these numbers empirically
proved endpoints would repeat the problem with the original guardrails.
Single-axis examples span intermediate levels, and `combined_tails` changes
all six requested directions together. The beta=128 stress test and pressure
amplitude controls lie outside the candidate rectangle deliberately.

The base measure is uniform in all nine physical values. `analytic_prior.py`
conditions it on positivity, finiteness and the stated finite-energy support.
It does not apply Gaussian, log-normal or FLAMINGO-centered weights. Conditional
marginals of beta and its evolution coefficients need not be uniform.

For a future numerically screened dataset the definition should be

\[
\pi(\theta)\propto 1_{\rm rectangle}(\theta)
1_{\rm analytic}(\theta)1_{\rm fidelity}(\theta).
\]

`evidence_gate.py` defines the last indicator conservatively. It requires
parameter, covariance and operator provenance, and full-observable tests for
interpolation, pixelization and LOS endpoint. It sums measured error distances
and reference-refinement changes, so cancellation between error sources cannot
produce a spurious pass. Missing evidence fails closed. It has **not** been
used to declare the entire extended rectangle safe. Reference convergence
must itself be established over the regimes being admitted.

The practical response to an interpolation failure is to improve interpolation,
not exclude an otherwise necessary reference model. There is also a useful
amplitude-dependent way to admit faint compact profiles. At fixed other eight
parameters, the clean map scales as P0 and every clean bandpower as P0 squared.
For a fixed reference covariance and a complete numerical-error measurement
epsilon_ref at amplitude P0_ref,

\[
P_{0,\max}^{\rm num}=P_{0,\rm ref}
\sqrt{\epsilon_{\rm allowed}/\epsilon_{\rm ref}}.
\]

This replaces a universal compactness exclusion with a measured amplitude
envelope: large relative discretization error can be acceptable for a signal
whose absolute error is below the declared accuracy budget. It is implemented
by `amplitude_ceiling` but is not evaluated from interpolation-only evidence
and then mislabeled a complete safety limit. If the covariance is updated with
signal amplitude, the simple square-root formula must be replaced by the
corresponding covariance-dependent calculation. Uniform volume sampling inside
such an envelope will naturally induce nonuniform marginals without any
Gaussian weighting.

After improving an implementation, repeat the accuracy test. A pixel failure
may require pixel integration or
convolving the continuous source before sampling, with the same physical
truncation implemented consistently. Those are future simulator changes and
must be validated before producing a larger dataset. They cannot be repaired
by relabelling arbitrary size/Y cuts as physics.

The files in `audit/` and the generated `RESULTS.md` distinguish completed
measurements from pending cluster work. The all-parameter figure explicitly
labels the purple distribution as an analytic candidate with fidelity pending;
it is not a plot of a fully numerically certified prior.
