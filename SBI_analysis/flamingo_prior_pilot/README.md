# FLAMINGO prior pilot

This folder tests the proposed nine-parameter pressure prior, fits the **clean**
FLAMINGO spectra, and checks the parameters with the original full-resolution
HalfDome painter. Noise is compared separately, as requested. Final measured
results and the production recommendation are in cluster_results/RUN_REPORT.md locally
and RUN_REPORT.md in the cluster workspace.

Cluster workspace: /lustre/work/kristero10/flamingo_prior_pilot_20260914

Immutable reference: /lustre/work/kristero10/flamingo_tsz_comparison_20260914

No further FLAMINGO map downloads are needed. The original inference prior,
MOPED weights, trained SBI bundle and archived operators are not edited.

## Proposed ranges

| Parameter | Lower | Upper | Distribution |
|---|---:|---:|---|
| P0 | 1 | 60 | uniform in log |
| xc | 0.1 | 4 | uniform in log |
| beta | 2.8 | 16 | uniform |
| alpha_m_P0 | -0.2 | 1.5 | uniform |
| alpha_m_xc | -0.6 | 0.4 | uniform |
| alpha_m_beta | -0.2 | 0.4 | uniform |
| alpha_z_P0 | -4.5 | 0.5 | uniform |
| alpha_z_xc | -1.5 | 2 | uniform |
| alpha_z_beta | -0.5 | 1.5 | uniform |

The explicit pilot contract is prior.json. These are generation ranges, not a
measured posterior or an approved replacement production prior.

Battaglia12 reference, in the table's order:

    [18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415]

## Pressure and tSZ physics

With x = r/R200c, the archived painter uses

    P_th/P200 = P0(M,z) (x/xc(M,z))^(-0.3) [1+x/xc(M,z)]^(-beta(M,z))
    A(M,z) = A0 (M200c / 1e14 Msun)^alpha_m (1+z)^alpha_z
    P_e = 0.5176 P_th
    y(theta) = sigma_T/(m_e c^2) integral P_e dl

Catalogue masses in Msun/h are divided by h=0.68 before the model receives
**physical** M200c. The pivot is physical 1e14 Msun.
P200 = G M200c (200 rho_crit) (Omega_b/Omega_m)/(2 R200c).
The fixed shape exponents are alpha=1 and gamma=-0.3.

The asymptotic outer pressure slope is beta(M,z)+0.3. The untruncated central
LOS integral converges only for beta(M,z)>0.7; pressure integrated over
infinite three-dimensional volume requires beta(M,z)>2.7. Finite Y200
remains defined for positive beta, even when those infinite limits diverge.

The painter cuts off the **projected** halo at 4R200c but integrates each
line of sight to 1e5 R200c. The projected cutoff does not impose a spherical
pressure truncation. The audit checks the column missed beyond the endpoint.

Y200 ratios compare the finite integral of (P_th/P200) x^2 over 0<x<1 at
the **same mass and redshift**, so dimensional prefactors cancel. Ratios
0.01 and 100 are diagnostic markers, not observational exclusion limits.
Central-column and finite-volume integrals are checked against independent
quadrature/incomplete-beta expressions. A separately validated LOS spline
also checks steep cases where production quadrature can underflow.

## Inputs and exact observation operator

The existing FLAMINGO maps are L1_m9, fgas-8sigma and Mstar-1sigma. Their
dimensionless Compton y includes the supplied lensing and shell rotations
to z<=3. The full historical HalfDome lightcone, including z>3, is retained.
Effective spectral fits can absorb catalogue, cosmology, redshift-support,
diffuse-gas and projection differences as well as feedback.

Completed candidates use the original Nside4096 RING pipeline:

- Gaussian beam FWHM 2 arcmin and the original default beam harmonic limit.
- Spherical cap support fsky=0.4 and 60 arcmin cosine apodization.
- Original SO baseline/deprojection-0 noise table.
- Root seed 12345 and split seeds 22446/22447.
- Identical cached mask and noise pixels, checked by SHA256.
- Spectrum lmax7979, niter0, and the original 40 weighted bins.

The clean pseudo-spectrum is formed from WBy. The noisy split cross-spectrum
uses W(By+n1) and W(By+n2). There is no fsky division, mask deconvolution,
beam deconvolution or additional beam on noise. Noise is excluded from fitting.
Plots show D_ell = ell(ell+1) C_ell/(2pi) in dimensionless y-squared units;
the tilde denotes the masked pseudo-spectrum. Identical noise maps need not
produce identical noisy-minus-clean spectra because signal-noise cross terms
depend on the signal.

## Fitting

The proposal engine is a new numerical catalogue calculation, not an
extrapolation of the old emulator. It integrates projected profiles within
4R200 using a flat-sky Hankel transform and squared cap weights at halo
centres. Nearby massive haloes are retained individually. Pixel sampling,
mask boundaries and halo-halo terms are approximated.

The objective is an equal-weight sum of squared logarithmic residuals in
40 clean bins. A weak unit-cube regularizer chooses a representative among
degenerate parameters. This is not a covariance-weighted likelihood and
does not supply parameter uncertainties.

Every proposal is repainted. Its measured full-map response updates the
next local proposal. Reported RMS/maximum residuals use full-map spectra,
with working targets of 2% RMS and 5% maximum, retaining the lowest bin.
Optimizer stopping status and prior-bound contacts remain explicit.

After the first unrestricted trial, follow-up steps are restricted to
0.12 in each unit-cube coordinate around the last map. New proposal searches
require 1.3<=beta(M,z)<=40 on the interpolation domain. These are numerical
search restrictions, not changes to the audited prior. A 20-minute watchdog
bounds each full map; timed-out candidates remain range-limit findings.

## Prior and numerical diagnostics

The audit includes 65,536 scrambled Sobol draws, all 512 corners, 18 single
edges, 144 pair-edge combinations and the reference. It checks the full
interpolation domain and the actual catalogue. A convex hull of all selected
(log M, log(1+z)) pairs gives exact power-law extrema over actual haloes.

The production interpolator is checked at 240 mass/redshift/radius points.
The archived Battaglia12 cache receives the same check. Results distinguish
the whole grid from z>=0.1. Independent LOS integration checks both production
quadrature and interpolation. These unweighted profile errors are not
power-spectrum errors.

A sparse HEALPix test samples actual pixel centres at Nside4096 and 8192
for 32 reproducible halo placements. Sampled halo flux is compared with
the continuous spherical integral over the projected profile. This isolates
subpixel flux loss and placement scatter; it does not replace full-map
resolution convergence. A normalized beam cannot restore missed total flux.

The optional screen in conditional_prior.py requires min beta>=2.8 and a
conservative central-column tail below 1%. The beta floor provides a margin
above infinite-volume convergence. This correlates beta0 with its mass and
redshift exponents, so the screened prior is not nine independent uniforms.
It does not impose observed thermal-energy bounds or ensure numerical
resolution. Its retention fraction and remaining Y200 range are reported.

The projected-cutoff diagnostic separately integrates the fraction of the
untruncated profile's total flux inside the painted 4R200 disc. It includes
the portions of outer spherical shells intersecting that cylinder. Fractions
are undefined for divergent untruncated totals and are not C_ell biases.
This identifies profile-extent choices that need a painting-radius test.

## Lee22 reference

The pressure panels use [Lee22 arXiv v1, Table 1 and Eq.12](https://arxiv.org/html/2205.01710),
including the amplitude mass break and a displayed fixed concentration c=4.5.
The break mass is converted from h^-1 Msun using TNG h=0.6774. Its
normalization is electron pressure, so no extra 0.5176 factor is applied.
The shaded radial interval is 0.04–1.34 R200; extrapolation is identified.
This is a literature-formula reference, not downloaded halo pressure data.
Existing FRB density runs retain Battaglia12 tSZ pressure and do not
independently constrain a Lee22 pressure prior.

## Files

| File | Purpose |
|---|---|
| prior.json, prior_model.py | Ranges, units, pressure integrals and binning |
| prepare_pilot.py | Catalogue quadrature and prior/edge/corner audit |
| forward_proposals.py | Numerical proposal model and bounded fitting |
| paint_candidate.jl | Archived full-map operators and profile checks |
| run_map.py | Execution, watchdog, resume and resource receipts |
| run_pilot.py, collect_pilot.py | Disjoint fit/extreme jobs and merge |
| prepare_guarded_proposals.py, seed_related_feedback.py | Guarded follow-up proposals and exact amplitude warm starts |
| secant_proposal.py, refine_fiducial.py, run_refine_fiducial.pbs | Interpolation between completed fits, followed by a new full-map check |
| conditional_prior.py, diagnose_ranges.py | Mathematical screen and independent tests |
| diagnose_quadrature.jl | Bounded native/scaled LOS cost comparison; does not replace production code |
| plot_pilot.py | Nine PNG/PDF sets and a combined vector PDF |
| verify_pilot.py, fetch_results.py, summarize_report.py | Provenance, tables, verified retrieval and measured report |
| requirements_diagnostics.txt | Private Astropy compatibility pin; shared runtime is unchanged |
| run_prepare.pbs, run_pilot.pbs, run_split_pilot.pbs | PBS preparation and forward runs |
| run_early_checks.pbs, run_diagnostics.pbs, run_guarded_proposals.pbs | PBS checks, plots, guarded proposals and verification |

## Execution and outputs

PBS scripts use mini, 26 CPUs, 64 GB, application thread settings, and
walltime <=23:59:00. The early-check job has a 15-minute limit.
Python is /home/anaconda3/bin/python3. Julia uses the archived 1.12.2
runtime/project/depot to preserve random-number behavior.

Set PILOT and CAMPAIGN with qsub -v. For split runs, set PILOT_ROLE to
fiducial, other_feedback or extremes. Diagnostics depend on all three
jobs succeeding. split_jobs.json records the actual cluster job IDs.

Fetch verified artifacts with:

    python SBI_analysis/flamingo_prior_pilot/fetch_results.py

The bundle excludes maps, the original catalogue and interpolation caches.
parameters.csv contains all nine parameters; fit_spectra.csv contains
clean/noisy target and fitted spectra. verification.json records cluster
checks; local_verification.json records successful retrieval and checksums.
resource_usage.json contains measured timings and memory. Use the completed
report, not job submission or proposal-only fits, as readiness evidence.
