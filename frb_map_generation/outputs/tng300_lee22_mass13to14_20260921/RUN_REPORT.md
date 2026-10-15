# Matched 10^13--10^14 Msun comparison

All three curves use 1e13 <= M200c/Msun < 1e14, source redshift 1, and a spherical R200c boundary.
The two painted catalogues retain the original 120,000 random directions and seed 42.
Ralf's existing 120,000-ray array supplies the hydro reference; its ray paths are not matched individually.

- [Three-way PDFs and all pairwise percentages](plots/lee22_ralf_tngdark_mass13to14.pdf)
- [HalfDome versus TNG-Dark](plots/lee22_halfdome_tngdark_mass13to14.pdf)
- [Full histogram values and signed percentages](pdf_and_percentages.csv)
- [Statistics and provenance](comparison.json)

Both percentage panels use 100*(p_numerator/p_denominator - 1), with the denominator named on the axis.
The Ralf panel shows HalfDome/Ralf and TNG-Dark/Ralf; the Dark panel shows HalfDome/TNG-Dark.
The displayed range is -20% to +20%, with no arrows or replacement markers.
Values outside this range remain unchanged in the plotted lines and CSV and are clipped only by the axes.
Ratios require at least ten rays in both histogram bins. The PDFs condition on DM > 0.

## Results

DM means have units pc cm^-3. The all-ray mean includes rays that intersect no selected halo.

| Catalogue | Mean, all rays | Mean, positive rays | Hit fraction |
|---|---:|---:|---:|
| Ralf IllustrisTNG300 | 72.90788 | 138.64768 | 52.58500% |
| Lee22 TNG300-1-Dark | 70.91944 | 137.83254 | 51.45333% |
| Lee22 HalfDome | 73.59388 | 136.94009 | 53.74167% |

## Physics and calibration

The fixed Lee22 no-concentration profile, mcut pivot, physical redshift scaling, and XGPaint-native normalization are unchanged.
DM is the electron-density integral through the finite halo chord, divided by (1+z_h), with no contribution outside R200c.
Each simulation retains its native cosmology. No extra baryon multiplier, fitted rescaling, or additional gas scatter is introduced.
**The requested lower mass cut requires extrapolation below the Lee22 fit limit of 1e13 Msun/h**, approximately 1.47--1.48e13 physical Msun.
The existing mass power laws are continued down to 1e13 physical Msun; this is not a claim of calibration in that interval.
Mass cuts now match Ralf exactly. FoF/SUBFIND versus calibration-catalogue differences, periodic TNG structure, and unmatched individual ray paths remain.
The original calibrated-range outputs are retained separately.

## Validation and resources

Profile SHA256: `1fe10765ba8e38ae42c0cf1d20630babff0b8b10aa2a41e1a617bdbc50ae8806`.
All 50 snapshots passed overlap checks: 115,962 retained halos have exactly identical IDs, masses, positions, and radii.
Both painters passed 100 direct-chord quadrature/density checks, including M200c=1e13 Msun.
Saved histograms match the per-ray arrays; painted sightlines, mass edges, and DM edges match exactly.
The newly added mass interval gives nonnegative extra per-ray DM, within floating-point tolerance.

| Stage | Peak resident memory [GiB] |
|---|---:|
| halfdome painting | 1.003 |
| tng painting | 0.949 |
| lightcone | 0.110 |
| cluster catalogue transfer | 0.290 |

Painting and lightcone construction ran locally. The catalogue-transfer stage used idark under a 4 GiB address-space ceiling; no PBS painting job was submitted.
The Julia painting ceiling was 12 GiB. Successful snapshot checksums and source copies are retained.

## Files changed

In `frb_map_generation/tng300_lee22_comparison/`:

- Added `compare_mass13to14.py`: common-ray configuration, exact mass-cut validation, signed ratios, figures, tables, and this report.
- Edited `preflight.py`: explicit mass13to14 case and extrapolation metadata.
- Edited `download_tng.py`: safe stdin authentication for the remote transfer and configured mass-selection description.
- Edited `build_lightcone.py`: describe the configured interval accurately in its validation error.
- Edited `paint_tng.jl`: configurable output/ray paths, general mass-window masks, extrapolation metadata, and the additional low-mass quadrature checks.
- Edited `run_tng.py`: pass the output root and optional common-ray configuration to Julia.
- Edited `README.md`: document this separate case, its calibration limitation, and plotting conventions.

The upstream Lee22 profile and original HalfDome generator were not edited.
