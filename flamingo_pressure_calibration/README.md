# FLAMINGO L1_m9 pressure calibration

Status: the cluster analysis is complete. The verified
final report is exported to `cluster_results/reports/RUN_REPORT.md`; scientific
completion is recorded by `cluster_results/complete.json`, not by submission.

User contract (2026-09-21): compute on the cluster; pressure only; use only
fiducial L1_m9; assess its complete available mass and redshift range; compare
measured profiles and fitted XGPaint gNFW profiles with current HalfDome
Battaglia12 at z = 0, 1, 2, 3 and log10(M200c / physical Msun) =
11, 12, 13, 14, 15, 15.5. Explicitly mark unresolved and absent bins.

Cluster output root: `/lustre/work/kristero10/flamingo_pressure_20260921`.
All scientific computations, data extraction, fitting and plotting run there.
This directory contains readable source and small, verified result exports.

## Scientific definitions

The target is spherical electron pressure, Pe = ne kB T, in physical units.
The pressure normalization is P200 = G M200c (200 fb rho_crit)/(2 R200c).
The current XGPaint family uses Pe/P200 = 0.5176 P0 q^-0.3 (1+q)^-beta,
q = r/(xc R200c). P0, xc and beta each have amplitude, mass exponent and
redshift exponent: A = A0 (M200c/1e14 Msun)^am (1+z)^az.
The stored beta is the Battaglia exponent; the asymptotic slope is -beta-0.3.

HalfDome is an N-body simulation with painted pressure, not a hydrodynamical
pressure measurement. The comparison baseline is its existing Battaglia12
family. A fit cannot validate mass/redshift regions absent from FLAMINGO or
below its numerical resolution. Such model predictions are extrapolations.

## Step record

1. Read the existing XGPaint pressure code and prior FLAMINGO spectral reports.
   Confirmed that prior map-spectrum fits are not direct radial-profile fits.
2. User selected L1_m9 only and the current Battaglia12 HalfDome comparison.
3. Connected to `idark` as `kristero10`. Confirmed the FLAMINGO data service
   responds from the cluster and lists 78 full L1_m9 snapshots, z=15 to z=0.
4. Inspected cluster Python installations. The default Python 3.8 does not
   support the current hdfstream client. A project-owned Python 3.11 virtual
   environment isolates its dependencies from existing cluster analyses.
5. The first environment installation attempted to compile the latest SciPy
   against an incompatible cluster Fortran toolchain and failed. Installation
   was changed to explicit compatible binary-wheel versions, without modifying
   any existing environment. The setup log preserves the failure and retry.
6. Live inventory measured 4.354589892 GB for the mass columns of all 78
   catalogues. The conservative full-analysis download estimate is 20.855 GB.
   A streamed HTTP byte counter and 60 GB cap enforce the user's limit.
7. Catalogue job 598563.idark started extraction in the mini queue, using
   26 allocated CPUs and two simultaneous server clients. The first four
   requested redshifts completed successfully.
8. Particle job 598574.idark exposed a SWIFTMask copy.copy recursion issue
   before particle extraction. Replaced Python copy discovery with an explicit
   shallow state copy of the unconstrained mask; immutable cell metadata is
   reused and each halo receives a fresh spatial selection. The failure log
   is retained with the retry record.
9. All 78 catalogues completed without extraction failures. The queued retry
   was made dependent on successful catalogue completion; it then started.
10. Independent pressure-unit, quadrature, analytic-gradient and synthetic-fit
    checks passed in the PBS allocation. Particle extraction validates the
    measured integrals and produces the fine radial profiles.
11. Completed 89 particle profiles and fitted 2358 shell means from all usable
    mass/redshift cells. Held-out RMS fractional error is 23.9%, compared with
    31.1% for Battaglia12 in the same FLAMINGO normalization. Both fit starts
    agree; all nine directions have numerical rank and no search bound is hit.
12. Checked 288 evaluations against the cluster's actual Julia XGPaint:
    relative shape error below 3.2e-15 and normalization difference 0.0021%.
    The report preserves software failures, fixes and scheduling details.
13. Generated the 24 requested comparisons, individual redshift PDFs/PNGs,
    coverage and residual plots, model exports, step log and checksum manifest.

## Reproduction and model export

`config.json` contains the scientific cuts and sampling settings.
`deploy.py` copies the source and submits the catalogue job on the first run.
`run_particles.pbs` extracts particle profiles, fits, verifies XGPaint, plots,
and writes the report; submit it with an `afterok` dependency on the catalogue
job. Both PBS scripts use the mini queue and 26 allocated CPUs.
`fetch_results.py` transfers the completed compact artifacts and verifies their
SHA256 manifest. It does not download simulation snapshots.

For subsequent painting, `load_xgpaint_calibration.jl` loads
`fits/xgpaint_parameters.toml` into the existing XGPaint constructor. Consult
`fits/calibrated_model.json` first: the global gNFW fit may have substantial
residuals even inside the sampled domain. This run does not repaint HalfDome.

Further steps and numerical outcomes are recorded under the cluster output
root and copied here with the final report. Job submission is not completion.

## Sources

- https://arxiv.org/abs/2604.24324
- https://github.com/jchelly/flamingo-docs
- https://arxiv.org/html/2312.08277v2
- `../frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl`
- `../frb_map_generation/compute_pressure_profile_grid.jl`

Large particle cutouts are not retained after profiles and validation summaries
have been safely written. Compact halo selections, shell sums, metadata,
checksums, code, fit results and the step report are retained for reproducibility.

## Interactive notebook: L2p8_m9 reduced snapshots (2026-09-25)

`FLAMINGO_L2p8_reduced_pressure_profiles.ipynb` streams the 2.8 Gpc fiducial run
from the COSMA data service (anonymous `hdfstream`) and works only with central
haloes flagged `SOAP/IncludedInReducedSnapshot`, i.e. those whose gas is present
out to R100c in `snapshots_reduced/`. It shows single haloes (y maps, 3D and
projected profiles), fits a plain gNFW (P0, xc, beta; optionally alpha, gamma;
no mass/redshift/concentration dependence), and stacks ~5e13 Msun haloes
(combined 3D profile with bootstrap errors, stacked y map and projected profile,
per-halo fit scatter). Plumbing lives in `flamingo_reduced_io.py`; downloads are
cached under `notebook_cache/` (gitignored, snapshot-78 catalogue cache built on
idark at `/lustre/work/kristero10/flamingo_reduced_cache/`).

Kernel: `FLAMINGO (swiftsimio+hdfstream)`, the venv `/home/cbllover/miniconda3/envs/flamingo`
(numpy 2.2, swiftsimio 12.1.4, hdfstream 0.0.28). The `halfdome` env cannot host
swiftsimio without moving to numpy >= 2.1, so it was left untouched.

## Cluster run prepared for 2026-09-28: `l2p8_reduced_calibration/`

Nine-coefficient fit to every central in the L2p8_m9 reduced snapshots (ten outputs, z = 0 to 3),
with small-scale (1 kernel length, free gamma, free alpha) and large-radius (SOAP shells to 5 R500c)
extensions. See `l2p8_reduced_calibration/README.md` for the Monday commands and the budget.

## Preliminary stacks and nine-coefficient fit (local, 2026-09-27)

`FLAMINGO_L2p8_preliminary_stacks.ipynb` (kernel "FLAMINGO (swiftsimio+hdfstream)"):
stacked Compton-y maps and profiles in 0.2-dex bins of M200c at z = 0, 0.5, 1, 1.5, 2, 3,
a Battaglia-form fit (free P0, xc, beta; alpha = 1, gamma = -0.3) of every stack against
Battaglia+12, and one joint fit of all nine Battaglia+12 coefficients to every stack.

- `stack_products.py` streams the data (run once; incremental: a later run with a larger
  selection downloads only the new haloes). Default subsample: random centrals per bin until
  0.5 M gas particles (10-200 haloes), bins up to 10^15.2 Msun, about 1.5 GB for six snapshots.
  Output: `notebook_cache/stacks/L2p8_m9_NNNN_stack_products.npz` (+ .json receipt, .jsonl log).
- `flamingo_stacks.py` holds the stacking, the shell-averaged model with its analytic Jacobian,
  the per-stack and joint fits and the bootstrap. The notebook only plots and tabulates.
- Results: `notebook_cache/preliminary_stack_fits.json`, `notebook_cache/xgpaint_L2p8_preliminary.toml`
  (joint fit including the SOAP shells beyond R100c; XGPaint key convention, loads with
  `load_xgpaint_calibration.jl`), `notebook_cache/xgpaint_L2p8_preliminary_particle_shells_only.toml`
  (the degenerate particle-shells-only variant), figures in `figures_preliminary/`.
- Fitted shells: from max(2 median smoothing lengths, 4 softenings, 0.05 R200c) to each halo's R100c.
  Per-stack fits use particle shells only; the joint fit adds SOAP's R100c-R50c and R50c-5R500c shells,
  because without them xc and beta slide along their degeneracy (xc 2.4, beta 9.5) and the model falls
  a factor ~1.8 below the measured pressure beyond R100c.
- Radial extent: the reduced snapshots hold every particle within R100c (1.35 R200c, median; FLAMINGO
  documentation), checked by particle Y inside R100c against SOAP's SO/100_crit/ComptonY (full snapshot):
  median |difference| 4e-4, maximum 0.7%, all 4,459 stored haloes. Beyond R100c only SOAP apertures
  (R50c = 1.8 R200c, 5 R500c = 3.3 R200c) and the full-snapshot cells of section 7 have data.
- Resolution-cut sensitivity: fitting from 1 instead of 2 median smoothing lengths adds 173 shells
  (114 from 10^13-10^13.6 groups, where the main model lies 48% above them) and gives P0 3.62, xc 1.69,
  beta 7.02, alpha_m(P0) 0.78, alpha_m(xc) -0.33; Y(<4 R200c) changes by -5% (10^13), -3% (10^14) and
  +3.5% (10^15) at z = 0. The 2h cut binds only below ~10^14 at z <= 1.5 (4 softenings bind at z >= 2).
  The resolved radius itself should come from L1_m8 against L1_m9 (both have reduced snapshots).

The cluster run in `l2p8_reduced_calibration/` (every reduced-snapshot central, ten redshifts)
supersedes these numbers once it has run.
- Sections 6-7 (added 2026-09-27): stacked projected-pressure maps with the joint model painted on the
  same pixels and their difference, for a mass and a redshift sequence; and extrapolation tests outside
  the fit: z = 0 stacks above 10^15.2 Msun, z = 4 (snapshot 14), and haloes from full-snapshot cells
  (`fullsnap_products.py`, incremental, 8 cells each at z = 0 and 1) down to 10^12.4 Msun and out to 3 R200c.
  Stacks above `FIT_LOG_M_MAX` = 10^15.2 never enter a fit. Bootstrap draws are seeded per (snapshot, bin).

## HalfDome painted with the FLAMINGO fit, against FLAMINGO's tSZ map (2026-09-27)

`halfdome_tsz_comparison/` (notebook `HalfDome_vs_FLAMINGO_tSZ.ipynb`, kernel "FLAMINGO"):

- `paint_halfdome_flamingo_tsz.jl`: all 85.2 M HalfDome haloes (M200c 6.6e12-3.8e15 Msun, z < 3.86),
  XGPaint `Battaglia16ThermalSZProfile` with the joint fit (`flamingo_joint_fit.toml`) and the
  particle-shells-only variant, gas inside 4 R200c spheres, NSIDE 4096, both maps in one 57-minute pass
  (20 threads). XGPaint's pressure equals the fitted formula to 4e-15. Maps in `output/` (1.6 GB each).
- `download_flamingo_tsz_map.py --lightcone K`: the Yang+26 lensed integrated map of L2p8_m9
  (`integrated_maps/yang26/lightconeK_shells/lensed_tSZ_rot_same_rot.hdf5`, 1.6 GB, resumable).
- `class_sz_references.py`: Tinker08 halo-model spectra separating cosmology, profile and halo coverage.
- `compute_comparison_products.py` (halfdome env): anafast (lmax 8192), one-point PDF, patches.
  Painted maps are point samples (no pixel window); the FLAMINGO spectrum is divided by the pixel window.

Figures: `spectra_halfdome_vs_flamingo.png` (as painted, no correction) and
`spectra_halfdome_vs_flamingo_cosmology_shifted.png` (HalfDome shifted to FLAMINGO's cosmology, nothing
else corrected), three maps each; maps on the cmocean `thermal` scale. The cosmology shift multiplies
each binned HalfDome spectrum by F_p(ell) = D_ell(FLAMINGO cosmology) / D_ell(HalfDome cosmology) from
CLASS-SZ with the same profile p, Tinker08 M200c, 4 R200c truncation and the HalfDome halo range
(integration tolerance 1e-4; 1e-3 left 0.8% single-ell glitches). Ratio to the FLAMINGO lightcone-0 map
(bands nearest each ell; 1e12 D_ell = 1.00, 1.49, 1.64 at ell = 1000, 2000, 3000):

| ell | FLAMINGO fit, as painted | FLAMINGO fit, cosmology-shifted (F) | B12, as painted | B12, cosmology-shifted (F) |
|---|---|---|---|---|
| 300 | 1.11 | 1.01 (0.908) | 0.79 | 0.72 (0.914) |
| 1000 | 0.99 | 0.90 (0.907) | 0.70 | 0.64 (0.915) |
| 2000 | 0.98 | 0.88 (0.907) | 0.69 | 0.63 (0.917) |
| 3000 | 0.95 | 0.86 (0.907) | 0.69 | 0.63 (0.919) |
| 5000 | 0.91 | 0.83 (0.906) | 0.71 | 0.65 (0.921) |

- The painted FLAMINGO-fit/B12 ratio matches CLASS-SZ's to 2% at ell = 300-3000 (the painting is right).
- Not corrected: in its own cosmology the painted HalfDome map has 12% (fit) / 10% (B12) less power
  than the Tinker08 halo model at ell = 1000-3000; the Tinker08 halo model with the fit in FLAMINGO
  cosmology over all haloes gives 1.04, 1.01, 0.98 of the map at ell = 1000, 2000, 3000. Halo coverage
  (haloes outside the HalfDome range) is worth +0.5% (ell = 3000) to +3.5% (ell = 300).
  The particle-shells-only fit gives the same spectrum to 1% (mean y 10% lower).
- Mean y: FLAMINGO 1.61e-6 (all gas), HalfDome 1.13e-6 (FLAMINGO fit), 1.01e-6 (B12). The one-point PDF
  agrees above y ~ 1e-5; FLAMINGO's peak sits 0.25 dex higher (diffuse gas and haloes below the catalogue).
- ell < 300: HalfDome is 10-25% above lightcone 0 but inside the spread of FLAMINGO's own skies
  (lightcone 1 / lightcone 0 = 1.24 at ell = 100, 1.05 at 300, 1.00 +- 0.01 at ell >= 1000).
  ell > 5000: pixel-window correction (x1.15 at 5000) and point-sampling aliasing dominate.
