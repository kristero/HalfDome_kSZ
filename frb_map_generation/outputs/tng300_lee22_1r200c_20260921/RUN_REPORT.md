# TNG300-1-Dark and HalfDome: Lee22 no-c within one R200c

Completed locally on 2026-09-21. The matched TNG/HalfDome comparison is complete.
Ralf's saved hydro arrays are included as a contextual comparison with explicitly
different mass cuts; an exact three-way mass-matched comparison is unavailable.
No cluster job was submitted.

## Figures and data

- [Matched probability densities and percentage differences](plots/tng300_dark_vs_halfdome_supported_mass_pdfs.pdf)
- [Ralf hydro reference with different mass cuts](plots/ralf_reference_different_mass_cuts.pdf)
- PNG and SVG copies accompany both PDF figures.
- [Summary statistics](statistics.csv), [histogram counts and densities](pdf_bins.csv),
  and [validation and source hashes](comparison.json).
- Per-ray results: [TNG](tng/halo_dm.h5) and [HalfDome](halfdome_direct/halo_dm.h5).
  These retain all 120,000 sightlines, including zero-DM rays.
- [Numerical interpolation audit](numerical_audit.json),
  [lightcone provenance](lightcone_provenance.json), and [geometry validation](geometry_validation.json).

## Results

The mean values below include zero-DM sightlines and have units pc cm^-3.
Masses are physical solar masses. Lower bounds are inclusive; upper bounds exclusive.
The figures show conditional densities p(DM | DM > 0), so the hit fraction is
reported independently. Similar conditional PDFs need not imply identical hit rates.

| M200c interval [Msun] | TNG mean DM | HalfDome mean DM | 100(HD/TNG - 1) | TNG hit fraction | HD hit fraction |
|---|---:|---:|---:|---:|---:|
| 1.4762326542663123e13 to 1e14 | 62.4884 | 64.5926 | +3.3673% | 41.8642% | 43.6208% |
| 1e14 to 9.278784477649915e14 | 45.4698 | 42.4016 | -6.7477% | 11.8267% | 11.8967% |
| 1.4762326542663123e13 to 9.278784477649915e14 | 107.9581 | 106.9942 | -0.8929% | 48.5667% | 50.1908% |

There are 12,200,520 selected TNG halo images and 13,174,022 selected HalfDome
halos in the lightcones. These counts describe the two realizations and their
native cosmological volumes; they are not volume-normalized mass functions.
The same profile gives closely overlapping conditional PDFs, while the
high-mass TNG sample has a larger mean DM. Opposite signed differences in the
two disjoint mass bins partially cancel in the combined mean. These results
do not establish a universal percent-level agreement or a statistical significance.

Ralf's nearest available bin is 1e13 <= M200c < 1e14 Msun. Its all-ray mean is
72.9079 pc cm^-3, positive-ray mean 138.6477 pc cm^-3, and hit fraction 52.5850%.
The painted lower-bin positive-ray means are 149.2646 (TNG) and 148.0773
(HalfDome) pc cm^-3. The different lower mass threshold affects the PDF and hit
rate, so these differences cannot be attributed solely to gas-profile physics.
No mass-matched percentage residual to Ralf is claimed.

## Physics and selection

The SHA256 of the reused Lee22 source is
`1fe10765ba8e38ae42c0cf1d20630babff0b8b10aa2a41e1a617bdbc50ae8806`.
It matches the fixed implementation recorded for the successful 2026-09-17 run.
The profile file and existing HalfDome generator were not edited.

The model uses Table A2 without concentration, `normalization=xgpaint_ne2d`,
`n0_pivot=mcut`, and physical redshift scaling. Its XGPaint mapping retains
P0=200*n0 and beta_XGPaint=alpha*beta_prime-gamma. The native gas-to-electron
normalization is applied once; there is no fitted rescaling to Ralf and no
second baryon-fraction multiplier on the DMO mass.

Each intersected halo contributes the finite spherical chord

```text
DM_halo(b) = 2/(1+z_h) * integral[0, sqrt(R200c^2-b^2)] n_e(sqrt(b^2+l^2)) dl,
```

with the established unit conversion to pc cm^-3 and zero contribution for
b >= R200c. Contributions add along a sightline. No diffuse IGM, host, Milky Way,
or additional halo-to-halo gas scatter is included. As in the fixed code, the
smooth profile continues inward of the fitted radial limit 0.04 R200c.

Lee22's calibrated mass interval is 1e13 <= M200c/(Msun/h) < 10^14.8.
Converting with each simulation's h and intersecting the intervals gives the
common bounds used above. TNG uses h=0.6774, Omega_m=0.3089, Omega_b=0.0486;
HalfDome uses h=0.68, Omega_m=0.31, Omega_b=0.049. Native cosmologies determine
critical densities, profile pivots, radii, and distances.

## Numerical evaluation and checks

Both final calculations evaluate the existing spherical chord directly.
An attempted native-TNG angular cache had a 54.3% error at a tested low-redshift
grazing chord (M=8e14 Msun, z=0.03, b/R200c=0.99). It was not used for the final
results. The historical HalfDome cache gives an all-ray combined mean 1.3161%
higher than direct integration on exactly the same halos and rays. Thus the
profile physics and normalization are unchanged, while interpolation error is
removed consistently from both final calculations.

Checks passed:

- 80 direct-versus-tighter-quadrature and XGPaint density-mapping checks per
  cosmology; maximum tested relative differences 5.66e-15 (HD), 6.11e-15 (TNG).
- Identical 120,000 unique NSIDE 4096 directions, seed 42, source redshift 1;
  these are random sightlines, not simulated host galaxies.
- Exact common physical mass edges and histogram edges; per-ray low+high=all.
- Saved histogram counts reproduce the per-ray catalogues. One positive TNG
  ray falls below 0.1 pc cm^-3 in the low/combined bins; it is counted separately.
  There is no overflow above the histogram limit 30,000 pc cm^-3.
- Compact snapshot checksums, mass/radius consistency at approximately 0.012%,
  and exact position/radius prefix-read agreement with a full snapshot-99 control.
- Periodic shell pruning agrees with exhaustive enumeration for 2,654 test
  images; adjacent test shells have no duplicates. Distance interpolation agrees
  with quadrature to 1.04e-11 relative error in the tested samples.
- The PDF panels were visually inspected. Percentage residuals require at
  least ten rays per histogram bin in both samples; upper triangles mark
  residuals exceeding the displayed +100% range. No Poisson significance is
  assigned to correlated rays or repeated structures.

The full TNG lightcone uses public FoF/SUBFIND M200c halos from snapshots 50--99,
one seed-42 uniform observer, and fixed periodic orientation. Shell boundaries
are halfway between snapshot comoving distances. Positions/masses use the
nearest snapshot; the gas profile uses continuous lightcone redshift. Halo
centres select shell membership, with full spheres assigned to selected halos.
This is a snapshot-based halo lightcone, not Ralf's Voronoi gas-cell ray tracing.
The halo finder differs from the Rockstar catalogue used to calibrate Lee22.
Finite box size, periodic repetitions, and observer variance remain unquantified.

## Resources and retained products

| Local production stage | Wall time | Peak resident memory |
|---|---:|---:|
| TNG lightcone construction | 30.65 s | 0.109 GiB |
| TNG direct painting | 34.11 s | 1.014 GiB |
| HalfDome direct painting | 131.55 s | 1.127 GiB |

Julia jobs used 20 threads and a hard 12 GiB virtual-memory ceiling. The cached
HalfDome control peaked at 1.347 GiB. Successful input receipts account for
5.664 GB transferred, excluding discarded interrupted attempts. Compact
snapshots retain 6.40 MB; the TNG lightcone is 391.92 MB. Redundant raw downloads
and the unused 134.22 MB cache were removed. Earlier scientific outputs remain.
The API key was used for authentication and is absent from code and products.

## Files added

All new workflow files are in `frb_map_generation/tng300_lee22_comparison/`:

| File | Purpose |
|---|---|
| preflight.py | Verify the fixed-profile provenance, mass support, inputs, resource estimates |
| download_tng.py | Authenticated, resumable field download and compact validated catalogues |
| build_lightcone.py | Periodic snapshot-shell lightcone with traceable halo IDs |
| paint_tng.jl | Reuse the fixed profile and accumulator for direct spherical chords |
| run_tng.py | Bounded local execution for TNG and HalfDome with resource/source records |
| run_halfdome.py | Reproduce the historical cached control on common mass selections |
| validate_geometry.py | Check shell geometry and cosmological distance interpolation |
| audit_numerics.py | Quantify the cached-versus-direct HalfDome differences |
| compare.py | Validate outputs and produce publication figures, tables, and hashes |
| README.md | Physical conventions, limitations, and executable workflow |

All new results and this report are under
`frb_map_generation/outputs/tng300_lee22_1r200c_20260921/`.
The `code/` directory freezes the workflow, exact painters used, and unchanged
upstream profile/generator. Source differences between the two painter runs are
a progress-message rename and a TNG-only HDF5 completion-attribute read fix;
the profile and accumulation mathematics are identical. `SHA256SUMS` records
the retained artifacts.

## Primary sources

- [Lee et al. 2022](https://arxiv.org/abs/2205.01710)
- [Official TNG300-1-Dark data and cosmology](https://www.tng-project.org/data/downloads/TNG300-1-Dark/)
- [TNG field definitions and units](https://www.tng-project.org/data/docs/specifications/)
- [TNG API](https://www.tng-project.org/data/docs/api/)
- [Lee calibration catalogue access](https://aust427.github.io/releases.html)
- [Konietzka et al. ray-tracing methodology](https://arxiv.org/html/2507.07090v2)
