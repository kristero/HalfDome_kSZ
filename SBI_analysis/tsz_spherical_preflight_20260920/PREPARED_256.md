# Prepared 256-row diagnostic - NOT SUBMITTED

The user requested preparation only. Do not submit generation or training jobs
until the user asks to start them. Existing resolution jobs 598439 and 598448
are separate, previously authorized numerical tests.

Prepared cluster run root:
`/lustre/work/kristero10/tsz_spherical_diagnostic_256_prepared_20260920`

Code root:
`/lustre/work/kristero10/tsz_spherical_preflight_20260920`

## Data contract

- 256 immutable nine-parameter Sobol design rows, independent uniforms in
  physical values, no rejection or FLAMINGO weighting.
- Parameter order and bounds are in `manifest.json`; beta amplitudes and both
  beta exponents are flat as well.
- 192 training-pool rows and 64 held-out rows. Noise seeds are independent per
  row and split, with no reuse of the observation stream.
- Spherical pressure support: 4 R200c; alpha=1 and gamma=-0.3. Physical M200c is
  catalogue M200c/h divided by h=0.68; original catalogue and selection remain.
- Experimental raw/output NSIDE4096 point painter, 2 arcmin beam applied once,
  fsky=0.4, original cap-mask seed 12345, ellmax7979. Each row retains the full
  clean and signed cross C_ell arrays. MOPED reads the 7,900 individual D_ell
  values at ell=80..7979 directly, without binning. The 40 bins are separate
  baseline/PCA comparisons only.
- Each noise split gets the original tabulated N_ell, without another beam.
  This preserves the existing idealized mock convention; it is not automatically
  equivalent to two equal observing halves of a full-survey noise table.
- No reuse of old cylindrical spectra under new spherical parameter labels.
- Cache: 512 x 256 x 128 nodes in log angle, log z and log10 mass; angular
  padding 256, theta_min=1.01815e-11 rad; log-cubic interpolation of chord mean,
  with an absolute 1e-300 floor. These settings are recorded in the manifest.

## Explicit limitations

This is a pipeline diagnostic, not a production-quality data release. The
point painter has known compact-halo aliasing; pre-beam positive quadrature was
tested separately, but no corrected full-catalogue renderer has been adopted.
The inherited angular interpolation floor is retained and documented. The
diagnostic can reveal numerical and inference failures but cannot certify
the new 8k dataset, or a 524k expansion, by itself.

## Prepared operation

`diagnostic.py --prepare-only` writes the immutable design, seeds, held-out mask
and manifest without generating maps or submitting anything. Generation reads
those saved arrays, rather than regenerating a version-dependent Sobol design.

`diagnostic.pbs` requests mini / 26 CPUs / 64 GiB / 23:59:00. After explicit
authorization, up to four workers could share the 256 rows using
DIAGNOSTIC_WORKER=0..3 and DIAGNOSTIC_WORKERS=4. Row locks prevent duplication;
atomic status files and a locked summary preserve incomplete/failed rows.
Changing the prepared physics or source hashes requires a fresh run root.

`diagnostic_analysis.pbs` is also unsubmitted. It checks all rows and labels,
retains negative cross bins, and stops if rows are missing. It prepares
train-only signed-asinh scaling, 40-bin, PCA and unbinned locally regressed MOPED
comparisons at 64, 128 and 192 rows, two training seeds, and shuffled controls
at the largest size. MOPED regression singular values are reported as a
compression diagnostic, not as proof of physical parameter identifiability.
Held-out normalized RMSE and marginal 68% coverage are saved. With only 64
held-out rows, neither precise coverage nor full SBC can be claimed.

`unbinned_moped.py` applies the OAS covariance inverse through a small
row-space system instead of allocating a 7,900-square covariance. Dense
reference checks verify that this produces the same covariance solve; it
does not discard or bin multipoles. Every full-spectrum row and FLAMINGO
observation is also rebinned and checked against its stored 40-bin counterpart.
The small pilot uses a local linear derivative regression, not the 55-term
quadratic fit that the larger existing pipeline requires at least 600 rows for.

See `NUMERICAL_CHANGE_AUDIT.md` for the exact grid, coordinate, projection,
quadrature and painting changes and their current validation limits.

The template accepts larger `--count` values. The 8192/524288 prefix and seed
uniqueness were tested, but the numerical production gate remains false.
No start time or completion date is promised while the run is unsubmitted.
