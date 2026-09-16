# Battaglia16 at 1R200c versus 3R200c versus IllustrisTNG

Completed on 2026-09-16. A new HalfDome halo-DM histogram product was painted on the idark cluster with the external projected aperture set to 1R200c; everything else matches the historical 3R200c product used for the 2026-09-15 publication figures.

## Open the results

- Main figure (same layout as `halo_pdf_b16_tng_upper_mass_limits`, now with three curves): `frb_map_generation/outputs/publication_comparison_20260916_1r200c/plots/halo_pdf_b16_3r200c_1r200c_tng_upper_mass_limits.png` (+ `.svg`).
- Companion figure with the windows ending at 1e14 Msun: `.../plots/halo_pdf_b16_3r200c_1r200c_tng_to_1e14.png` (+ `.svg`).
- Two-page vector PDF: `output/pdf/halfdome_b16_1r200c_vs_3r200c_vs_tng_20260916.pdf`.
- Per-bin audit table: `.../analysis/halo_pdf_aperture_comparison_bins.csv`; captions: `.../figure_captions.json`.
- New 1R200c product: `frb_map_generation/outputs/zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200cprofile_seed42/` (HDF5, summary CSV, provenance, generator plots, cluster log). Cluster copies: `/lustre/work/kristero10/frb_data/halfdome_m200c_r200c_z1_120k_foreground_mass_histograms{,_3cpu}/<same name>/`.

## What was run

Cluster job `597992.idark` (queue `mini2`, 3 CPUs, ~5 minutes) executed `frb_map_generation/run_halfdome_z1_nside4096_1r200c_mass_histograms.pbs`, a copy of the 3R200c PBS script with `HALO_EXTENSION_R200_MULTIPLIER=1.0` and run tag `r200cx1p0`. It reused the committed generator on the cluster (`generate_halfdome_z1_dm_mass_windows.jl`, same content as local git HEAD), the same 120k unique uniform rays on the z=1 plane (seed 42, NSIDE 4096), the complete `lightcone_100.hdf5` catalogue (85,224,251 rows, 29,130,459 foreground halos), the same shared XGPaint Battaglia16 DM cache (identical sha256), the same M200c mass windows and the same 300 log bin edges over [0.1, 30000] pc cm^-3. Only the generator-owned angular filter changed: rays contribute when theta <= angular_size(1.0*R200c) instead of 3.0*R200c. The cached DM profile itself is aperture-independent, so no re-painting of the profile cache was needed. The PBS post-check confirmed the M200c/R200c provenance attributes and multiplier 1.0.

The 26-CPU submission on `mini` (597990) could not start because every GroupA node was full; it was deleted after the 3-CPU job on the free GroupB node completed.

## Results

Total window (1e10-1e16, effectively 7.327e12-3.8e15 Msun), 120k rays:

| Aperture | Ray/halo intersections | Zero-DM ray fraction | Mean DM [pc cm^-3] | Std DM |
|---|---:|---:|---:|---:|
| 3R200c | 1,141,198 | 0.0002 | 203.9 | 153.7 |
| 1R200c | 126,665 | 0.3527 | 102.6 | 143.4 |

The intersection count drops by ~9x, as expected for a 3x smaller angular radius. With 1R200c, 35% of rays intersect no resolved halo at all (86% in the 1e10-1e13 window, 46% in 1e13-1e14).

Interpretation notes:

- The PDFs are normalized over in-range rays only, as before. For 1R200c that means the plotted curve describes the 65% of rays with a positive halo DM; the delta at DM=0 is not drawn. Compare zero fractions before reading the percentage panels as an overall agreement measure.
- The 1R200c curve has a sharp lower cutoff near 22 pc cm^-3: a single resolution-floor halo (7.3e12 Msun) grazed at exactly R200c yields at least that DM under the Battaglia16 profile, whereas at 3R200c the same halo contributes down to ~3.5 pc cm^-3. TNG contains halos far below the HalfDome floor and extends to arbitrarily low DM.
- Above ~100 pc cm^-3 the 1R200c and 3R200c curves nearly coincide and both track TNG; the percentage panels show 1R200c slightly below TNG there while 3R200c sits 40-50% above at 200-500 pc cm^-3.
- Everything else from the 2026-09-15 captions still applies: historical projected-aperture products, original LOS prescriptions, not the later spherical-cut test; TNG is Ralf Konietzka's catalogue; no rebinning or smoothing.

## Code changes

- Added `frb_map_generation/run_halfdome_z1_nside4096_1r200c_mass_histograms.pbs` (also uploaded to the cluster).
- `frb_map_generation/compare_tng_halfdome_direct.py`: `run_dir` now always writes the explicit `r200cx<mult>` tag for validated M200c/R200c products, so aperture 1.0 resolves to the new directory. Existing 3R200c and legacy paths are unchanged.
- `frb_map_generation/make_publication_comparisons.py`: new `halo_pdf_aperture_figures` and an `--aperture-only` flag.

Regenerate with:

```bash
/home/cbllover/miniconda3/envs/halfdome/bin/python -m frb_map_generation.make_publication_comparisons --aperture-only --output frb_map_generation/outputs/publication_comparison_20260916_1r200c --pdf output/pdf/halfdome_b16_1r200c_vs_3r200c_vs_tng_20260916.pdf
```
