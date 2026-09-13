# Handoff validation, 2026-09-11

Completed locally before packaging:

- Both shipped CSVs have exactly 32,768 rows and the nine required columns.
- The nine-parameter table varies all nine columns inside the stated generation bounds.
- Only P0 and beta vary in the two-parameter table; the other seven equal Battaglia12.
- All 131,072 actual split seeds across both datasets are unique.
- Twelve Python unit tests cover the adjacent-row seed-collision
  regression, cross-dataset seed collisions, inherited environment overrides,
  signed binning, resume behavior, corruption detection and the combiner.
  They also check all four products, 524288-row seed namespaces, Sobol prefix
  extension, original-CSV block extraction and mislabeled row offsets.
- The worker/combiner integration test uses a mocked simulator, not real sky spectra.
- A PBS dry run produced separate jobs and the requested dependency lanes without
  submitting generation jobs. Shell syntax was checked.
- The pinned Julia 1.12.2 dependencies were installed and the vendored XGPaint
  loaded successfully. Actual small-map tests at NSIDE=16, lmax=31 passed for
  same-seed reproducibility, different-seed noise maps and the fixed mask.
- The Julia CSV parser was checked against the first and last rows of BOTH designs.
  Every row of both 32k designs also passes the physical pressure guardrails.
- The original local HalfDome HDF5 layout was checked: Position, halo_mass_m200c
  and redshift have consistent lengths, with 85,224,251 halos.

Not performed as part of this handoff: full-resolution NSIDE=4096 generation
of the two 32k datasets, a complete physics validation on another machine,
posterior training or SBI calibration. Run the full-resolution smoke job on the
collaborator's cluster before scaling up. Catalogue layout checks do not replace
verification of the catalogue's provenance and units.

## Saved diagnostic plots

All five plots are supplied as PNG and PDF in `validation_plots/`:

1. `01_seed_reuse_vs_independent`: actual noise-map correlations for repeated
   seeds, stride-one seeds and the new stream allocation. Repeated seeds give
   correlation one; stride one visibly reuses adjacent split maps.
2. `02_noise_map_comparison`: same mask and noise power, repeated versus new
   row seeds, with a shared color scale.
3. `03_noise_power_and_products`: full native baseline/goal deproj0/2 curves
   and empirical harmonic power divided by input power over the test range.
4. `04_signed_cross_noise_spectra`: repeated-seed curves coincide, whereas
   independent rows have distinct signed noise-only cross spectra.
5. `05_threaded_painter_regression`: serial versus old threaded and ring-locked
   accumulation for 1000 deliberately overlapping toy halo profiles.

The noise test uses 72 real Julia/HEALPix noise draws (including controls),
NSIDE=128 and lmax=255, the supplied SO curves, and the same seeded mask recipe.
Across 48 independent test maps the largest absolute sampled-pixel correlation
was 0.0387. Mean harmonic power/input power was 1.00253, 0.99941, 0.99918 and
1.00155 for baseline0, baseline2, goal0 and goal2 respectively. All were within
the analytic five-sigma full-sky Gaussian-noise sampling tolerance. These tests
use unmasked alms for power validation; they do not assume diagonal mask covariance.
The mask correlation calculation excludes zero-mask pixels.

The painter stress test reproduced lost contributions in the original threaded
painter. Ring locks agree with the serial reference at rtol=1e-12. The plotted
old error is NOT an estimate of the error in a full HalfDome catalogue map.
Floating-point summation order can still differ across thread schedules.

## Audit fixes and remaining assumptions

- Reserved row/product/dataset seed namespaces prevent split reuse when scaling
  N or adding goal/deproj2. This replaces the original bundle's stride-two plan.
- Per-ring locking removes a real shared-pixel race in the supplied HEALPix
  painter. The override is separate from the original vendored source snapshot.
- The wrapper honors the explicit beam CLI setting instead of the included
  driver's unconditional fallback environment value. Default remains 2 arcmin.
- Noise tables must contain exactly ell=80..7979 with finite nonnegative values
  in the correct columns; missing/duplicate/invalid multipoles cannot be silently
  dropped by the otherwise permissive Julia loader.
- Actual theta, seeds, beam and painter are checked before accepting an output.
  Design hashes and global-row offsets must match; theta stays float64 on disk.
- Smoke failures return nonzero, shared submission/combination operations are
  locked, and stale completion markers cannot survive interrupted recombination.
- User prior bounds are checked in Python; physical Julia guardrails remain on
  without accidentally enforcing a second hard-coded prior for new designs.

The physical contract is unchanged: one fixed halo lightcone, one apodized mask,
signal smoothed once by the 2 arcmin beam, native N_ell applied to each independent
noise split, masked cross C_ell without fsky/beam deconvolution, signed D_ell
with 2ell+1 weighted Delta-ell=200 bins. The sky realization is not resampled.
Noise products are separate draws, not a jointly correlated multifrequency ILC
simulation. More data alone cannot change either of these physical assumptions.

The user confirmed the existing PER-SPLIT N_ell convention on 2026-09-11. No
factor-of-two change is applied. The headers do not independently specify the
split/coadd distinction, so this is an explicit modeling choice. Confirm that the beam convention matches
the intended observable; this package does not add a second beam to noise.
The full-resolution smoke job is still required for runtime/memory and a real
catalogue output check. No claim is made that these checks prove all mocks correct.

The immutable source/data inventory is in SHA256SUMS. Verify it before editing
cluster.env/config.json. Dependencies and the catalogue are external to the archive.
