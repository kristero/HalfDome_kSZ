# Lee22 / Battaglia16 numerical validation

These are diagnostic tests, not a replacement production run. They do not
modify XGPaint, the density normalization, the original interpolation caches,
or the original maps.

## Cluster execution

The 2026-09-11 submission is **596898.idark**, queue **mini2**, one node with
26 CPUs, 32 GB RAM, and walltime 23:59:00. The runner's default queue remains
mini; mini2 was selected explicitly because an eligible node was idle.

- Source snapshot:
  `/home/kristero10/HalfDome_kSZ/frb_map_generation/cluster_tests/lee22_numerical_20260910_v1`
- Cluster results:
  `/lustre/work/kristero10/frb_data/lee22_numerical_tests_20260910_v1`
- Local download directory:
  `frb_map_generation/outputs/cluster_lee22_numerical_tests_20260910_v1`
- Checkout HEAD at submission: `28c8f92161752a0336bb920f0ab1b10c576f4d1c`.
  The checkout includes uncommitted files: the result's `source_sha256.txt`,
  not HEAD alone, identifies the actual uploaded source snapshot.

From the cluster, inspect progress with:

```bash
OUT=/lustre/work/$USER/frb_data/lee22_numerical_tests_20260910_v1
qstat -f 596898.idark
cat "$OUT/stage.txt"
tail -n 40 "$OUT/logs/job_596898.idark.log"
```

Completion requires `stage.txt` to contain `complete`, `exit_status.txt` to
contain `0`, and the expected numerical reports and plots to exist. Passing
syntax/import checks does not establish numerical or scientific correctness.

## What is tested, and why

### Direct profiles and angular-cache interpolation

`audit_lee22_profile_cache.jl` compares the existing angular interpolation
tables with direct line-of-sight integrals of the same profiles. Mass means
physical M200c in solar masses; the catalogue's Msun/h values are divided by
h = 0.68. Halo DM is converted to observer frame with 1/(1+z_halo).

The catalogue is scanned completely. Its 29,130,459 valid foreground halos
at 0 <= z <= 1 are counted. Pointwise interpolation tests use all 774 halos
below z=0.02 and 60 mass-stratified representatives at 0.02 <= z < 0.1.
This 834-object diagnostic sample is NOT used to paint the science maps.

The audit also compares segmented and ordinary LOS quadrature and integrates
the electron profiles into an ionized-gas-equivalent mass. A suspicious gas
budget is a normalization diagnostic, not permission to rescale the fit.

### Radius-scaled table prototype

`test_radius_scaled_dm_cache.jl` tabulates the logarithm of a dimensionless
projected profile as a function of log(R_perp/R200c), halo redshift, and
log10(M200c/Msun). The physical density/radius amplitude and observer-frame
factor are evaluated at each test halo's actual mass and redshift. This avoids
interpolating over a rapidly changing angular size at very low redshift.

Two grid resolutions test convergence. The mass grid includes the Lee22
broken-power-law transition. Held-out cases include nearby catalogue halos
and 500 random mass/redshift/radius points, with projected radii through 5R200c.
The existing angular geometry is retained so geometry and interpolation
changes are not mixed. Tables are experimental HDF5 outputs, not installed
production caches. The report checks a 1% relative-error target for direct
columns above 1e-5 pc cm^-3 and also reports maximum absolute errors.

### One-million-ray sampling and complete-map spectra

`test_cluster_dm_sampling.py` uses the existing NSIDE4096, source-z=1,
projected-3R200c maps. Both contain the same full resolved foreground halo
population. It draws one million distinct pixels for each of 16 matched
random seeds per density model. No beam, instrumental noise, or pixel-window
deconvolution is introduced. All transforms use lmax=8192 and zero iterations.

Let P be the number of sky pixels, n the number of distinct sampled pixels,
and q the DM minus the known complete-map mean. The zero-filled map contains
(P/n)q at selected pixels. Its expected spectrum is

```text
E[C_sparse] = a C_full + b (4 pi/n) Var_full(DM)
a = P(n-1) / [n(P-1)]
b = (P-n) / (P-1).
```

The code compares ordinary Poisson subtraction with the finite-population
estimate `(C_sparse - b N_sample)/a`, where
`N_sample = (4 pi/n) mean(q_sample^2)`. An exhaustive 12-pixel test verifies
this sampling identity. The known full-map mean isolates this effect; it is
not assumed to be observable from a sparse real FRB catalogue. Four additional
draws per model test sample-mean subtraction with an explicitly approximate
correction.

Signed negative estimates are retained. The plots distinguish the scatter of
one realization from the standard error of the 16-realization mean. Percentage
residual axes are linear and are not clipped to an arbitrary percentage range.
The full-map references still contain any existing profile-cache errors;
sampling validation alone cannot validate the underlying painted sky.

## Main output plots

- `current_production_profile_comparison.png`: direct and cached columns.
- `physical_density_and_column_check.png`: 3D density and projected column.
- `radius_scaled_cache_validation.png`: old versus prototype interpolation errors.
- `cluster_sampling_comparison.png`: full-map and sparse-ray spectra, residuals,
  and sampling scatter.

Machine-readable CSV, NPZ, and JSON diagnostics accompany these plots. The
radius experiment checks 3R200c and 5R200c columns, but this batch does not
repaint complete 5R200c maps or recompute tSZ auto/cross spectra. The aperture
is projected; the existing LOS integration is not changed into a spherical
cutoff at the aperture radius.

## Completed results: job 596898.idark

The job ran on `ansys1` and finished at 2026-09-11 21:11:26 UTC with exit
status 0 and stage `complete`, 8 minutes 17 seconds after the script started.
All four PNG plots and their CSV/JSON/NPZ diagnostics were downloaded and
the 23 entries in `artifact_sha256.txt` verified locally. Large original maps
and experimental HDF5 interpolation tables remain on the cluster; the latter
are not required to inspect the downloaded validation results.

### Interpolation result

For the actual catalogue halo at zero-based row 85224238,
M200c=1.002107226e14 Msun and z=0.0044367313, the old angular cache at projected
3R200c gives 4.317 times the direct Battaglia16 DM and 4.985 times the direct
Lee22 DM. Thus the low-redshift cache error affects both profiles.

Maximum relative errors of the radius-scaled prototype on the 4,670 test
points per model are:

| Model | Coarser grid | Finer grid |
| --- | ---: | ---: |
| Battaglia16 | 0.1174% | 0.02886% |
| Lee22 | 0.6325% | 0.19046% |

Both grids pass the stated 1% target. This validates the sampled columns, not
a full-sky map painted with the new table. Independent segmented LOS
quadrature differs from the original integral by at most 7.12e-12 relative
on its test grid, so it does not explain the large interpolation discrepancies.

### Sampling result

The last band spans ell=7043..8192 with effective ell=7631.97. Values below
are D_ell in (pc cm^-3)^2. The quoted uncertainty is the standard error of the
mean of 16 realizations, not the uncertainty of one catalogue.

| Model | Complete map | Old Poisson-subtracted mean | Finite-population mean +/- SEM | One-realization standard deviation |
| --- | ---: | ---: | ---: | ---: |
| Battaglia16 | 3562.5 | -10053.3 | 3388.0 +/- 463.3 | 1853.1 |
| Lee22 | 3956.0 | -69479.0 | 4802.1 +/- 7705.6 | 30822.4 |

The corrected highest-band means are 0.38 and 0.11 SEM from their complete-map
references, respectively. Lee22's apparent agreement there is not a precise
measurement: its one-realization scatter is 7.79 times the signal. In this
binning the scatter first exceeds the signal near effective ell=5640. One
Lee22 bin near ell=4848 remains 3.34 empirical SEM below its reference; do not
claim all-bin convergence from only 16 realizations or treat correlated bands
as independent statistical tests. The exact finite-population identity is
separately verified by exhaustive enumeration of a small sky.

The broad Lee22 excess remains in the complete maps: Lee22/Battaglia16 is
about 3.05 near ell=3000 and 1.11 in the last band. These complete-map spectra
still use the old caches; their ratio is not a validated prediction after the
interpolation correction. Negative sparse estimates must remain visible;
filtering them out can make a noisy last positive point look like an upturn.

### Physical interpretation and remaining limitation

Source z=1 does not mean every contributing halo has z=1. In the implemented
direct profiles at M200c=1e14 Msun and projected radius 0.04R200c, the Lee22/B16
column ratio is 2.58 at halo z=0.1, 1.25 at z=0.5, and 0.643 at z=1. Therefore
a comparison of central density only at z=1 cannot establish the ordering of
the light-cone power spectra.

The current Lee22 profile at M200c=1e14 Msun, z=0 integrates to an
ionized-gas-equivalent mass of 3.270 f_b M200c within R200c; B16 gives 0.631.
The fitted radial shell 0.04..1R200c alone gives 3.267 for Lee22, so simply
excluding the innermost extrapolation would not resolve this discrepancy.
This is a normalization/convention warning, not a proof that f_b M is a strict
upper bound for every individual halo, and not a reason to rescale the fit
arbitrarily. Check an author reference evaluation before claiming Lee22's
physical amplitude is validated.

The next production validation should use the numerically tested radius
table to repaint both models with identical halo and aperture selections,
compare complete-map spectra first, and then test sparse-ray estimates.
No such production replacement or repaint was performed in this diagnostic job.
