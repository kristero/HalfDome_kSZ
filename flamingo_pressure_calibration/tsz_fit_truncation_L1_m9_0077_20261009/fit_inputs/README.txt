Pressure-fit inputs: fit extent 1, 2, 3, 4 R200c; all profiles available to 4R
============================================================================

What changed
------------
This new directory contains the 1R pressure fits, paired bootstrap coefficients,
common-support shape tables and fit/projection diagnostics for the tSZ study.
The original extraction and 2/3/4R results were read, never edited. No particles
were loaded, downloaded or remeasured. All numerical fits used one CPU thread.

Files created here
------------------
prepare_fit_inputs.py    readable 1R fitting and positive-curve export
plot_fit_diagnostics.py  coefficients, covariance samples, 3D and projected fits
verify_fit_inputs.py     independent input/reuse/support checks + heldout 1R
fit_parameters.npz      coefficients[bin,fitR,parameter], bootstrap[bin,R,128,3]
profile_shapes.npz      F=Pe/P200 on [fitR,logmass,x]; see schema below
coefficients.csv        absolute P0,xc,beta with bootstrap p16/p50/p84 and RMS
common_support.json     strict-resolution and coefficient-stability decisions
fit_input_summary.json  objective, units, constraints, provenance, resource use
derived_profile_constraints.npz  F at x=.1,.5,1,3,4 and dimensionless Y(<4R)
projection_metrics.json common4R map residuals and integrated biases
heldout_1R.json          original 20% halo holdout, same RNG stream, N>=20
verification.json       numerical checks and original-input SHA256 fingerprints
figure_manifest.json    hashes of 12 PNGs and their matching SVG files
figure_contact_*.jpg    visual-review contact sheets
figures/                publication-sized profile, coefficient and map figures

How to run
----------
Use the validated miniconda Python in WSL, from this directory:
  /home/cbllover/miniconda3/bin/python prepare_fit_inputs.py
  /home/cbllover/miniconda3/bin/python plot_fit_diagnostics.py
  /home/cbllover/miniconda3/bin/python verify_fit_inputs.py

Fit definition
--------------
x = r/R200c is dimensionless. F(x) = Pe/P200, where Pe is physical electron
pressure and P200 = G M200c (200 rho_c) fb / (2 R200c). The pressure convention
matches the previous XGPaint/Battaglia pipeline:
  F(x) = 0.5176 P0 (x/xc)^(-0.3) (1+x/xc)^(-beta).
P0 is an amplitude coefficient, xc a dimensionless transition scale and beta
sets the outer decline; the asymptotic outer slope is -(beta+0.3). P0 alone is
not the pressure at x=0 or x=1. At x << xc, the measurable inner amplitude is
approximately 0.5176 P0 xc^0.3, so coefficients compensate each other.

Each shell prediction is a volume average, not a value at the shell center.
Every stack is the arithmetic mean over the same selected central halos.
Fit residual = [log(predicted_shell)-log(measured_mean)] / max(std/mean,0.05).
Here std is the between-halo sample standard deviation. It is not the error of
the mean and is not a full radial covariance. Lower masks are unchanged from
the previous extraction: resolution, particle-count and smoothing criteria
are applied before changing the upper fitting radius.

Primary parameter bounds are unchanged: P0=[.001,10000], xc=[.01,10],
beta=[.5,20]. Stable means solver success, log-parameter bound distance >1e-5
and weighted log-Jacobian condition <1e6. A separate wide-bound 1R fit uses
P0=[1e-5,1e6], xc=[1e-3,100], beta=[.05,100]; it is a sensitivity control,
never an automatic replacement of the primary fit.

Support and mass policy -- essential for consumers
-------------------------------------------------
The common all-four experiment uses ten bins covering log10 physical mass
13.5--15.5, and stops calibration at the largest measured logmass 15.388243.
The 13.3--13.5 bin has only FOUR strict shells within 1R and fails the minimum
SIX-shell gate. It uses the SAME published fallback in every fit-radius case,
even though its 2/3/4R fits were supported previously. Lower bins are diagnostic
and also use the shared published fallback. Three lowest diagnostic 1R fits
hit coefficient bounds; they cannot be promoted to calibration.

Within a supported source 0.2-dex mass bin, the dimensionless measured shape is
constant with mass. P200 and R200c still use each halo's actual physical mass.
There is no interpolation between bin pivots in this experiment. Use the
actual mass and fit_parameters.npz to select the bin and evaluate the analytic
formula directly. Below, above or outside supported calibration, evaluate the
published B12 coefficients at the ACTUAL mass. Never interpolate across sharp
bin/support boundaries in the convenience grid.

profile_shapes.npz schema:
  fit_radii = [1,2,3,4]
  grid_logmass = linspace(12,16,201)
  x = geomspace(1e-5,4,1024)
  profiles.shape = (4,201,1024)
  common_support.shape = (15,), false bins0..4, true bins5..14
  mass_edges.shape = (16,)
  calibrated_logmass_max = largest measured physical logmass
  extra_case_names, extra_profiles = (6,201,1024):
    no_other_fit4, published, Y4_matched_fit1, Y4_matched_fit2,
    Y4_matched_fit3, wide_bounds_fit1
Y4-matched cases multiply each shorter-fit curve by a constant so its integrated
4R pressure equals the 4R-fitted curve. These artificial normalization controls
isolate Fourier shape from total pressure budget. no_other removes gas bound to
OTHER halos; it is not a unique gas-ownership model or a physical two-halo term.

Bootstrap and extrapolation
---------------------------
The 128 whole-halo resamples use seed 20261008+bin_index, exactly pairing 1R
with stored 2/3/4R draws. Each draw is refitted with its own empirical diagonal
weights. Failed/bound-limited draws remain NaN in their original positions.
All 128 draws pass in every common-support bin for all four fit extents.
The top bin contains only SIX halos. These bootstrap intervals describe the
conditional mean-fit sensitivity to those halos, not cosmic variance or an
intrinsic independent distribution of paintable coefficients.

1R fitting does not measure the pressure outside 1R. Evaluating to 4R explicitly
extrapolates a gNFW form. The low-mass common bin resolves only x>=0.321; even
F(.1) is an inward extrapolation there. The 14.5--14.7 example resolves x>=.0896
and the six-halo high-mass bin resolves x>=.04035. We retain these distinctions
when showing derived parameter constraints.

The wide-bound test leaves supported 1R Y(<4R) unchanged to <1.4e-8 relative,
and a different initial guess reproduces supported coefficients to <1.9e-8.
That checks optimization/prior-bound sensitivity for the CURRENT point fits;
it does not validate the extrapolated physical profile.

Projection diagnostics
----------------------
All curves in projected_profiles_fixed4_* and all model maps are projected
through the SAME 4R sphere. The measured target is the equal-halo mean of ten
rotated finite-4R particle-center maps; these camera views are correlated.
Models use the matching 64x64 grid, 3x3 subpixel averaging and 128-node LOS
quadrature. The full maps include unresolved central lines of sight and are
diagnostic. Resolved projected profiles start at max(.25, first_edge+.089) R.
The displayed model-minus-stack residual can contain asymmetry, substructure,
environment and gNFW mismatch; it is not a direct isolated two-halo map.

Physical interpretation
-----------------------
Extending a fit adds outer radial information, while forcing a single smooth
gNFW family to accommodate gas belonging to the host, satellites, neighboring
halos and unbound environment. It can move P0 and xc together and change the
central approximation. More gas does not create independent radial samples
or guarantee a more accurate one-halo pressure model. A fit-range comparison
with fixed 4R integration isolates this model response; a separate fixed-fit
truncation comparison isolates the loss of pressure volume. The full 4x4
fit-radius/truncation matrix reveals their interaction.
