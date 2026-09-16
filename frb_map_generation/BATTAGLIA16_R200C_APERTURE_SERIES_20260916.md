# Battaglia16 aperture series: 1-5 R200c versus IllustrisTNG

Completed on 2026-09-16, extending `BATTAGLIA16_1R200C_COMPARISON_20260916.md`. Four more HalfDome halo-DM histogram products were painted on the idark cluster: 2, 4 and 5 R200c, plus a repeat of the 3R200c configuration as a consistency check against the historical 2026-08-26 product.

## Open the results

- Main figure (layout of `halo_pdf_b16_tng_upper_mass_limits`, seven curves): `frb_map_generation/outputs/publication_comparison_20260916_apertures/plots/halo_pdf_b16_r200c_apertures_tng_upper_mass_limits.png` (+ `.svg`).
- Companion with the windows ending at 1e14 Msun: `.../plots/halo_pdf_b16_r200c_apertures_tng_to_1e14.png` (+ `.svg`).
- Two-page vector PDF: `output/pdf/halfdome_b16_r200c_apertures_vs_tng_20260916.pdf`.
- Tables: `.../analysis/halo_hit_percentages.csv` (share of rays through at least one resolved halo, per window and aperture), `.../analysis/halo_pdf_aperture_series_bins.csv` (per-bin audit), `.../analysis/repeat_3r200c_consistency.json`.
- Products: `frb_map_generation/outputs/zsrc1p0_nside4096_nrays120000_allhalos_r200cx{1p0,2p0,4p0,5p0}_m200cprofile_seed42/` and `..._r200cx3p0_m200cprofile_seed42_repeat20260916/`. Cluster copies under `/lustre/work/kristero10/frb_data/halfdome_m200c_r200c_z1_120k_foreground_mass_histograms_3cpu/`.

## What was run

Cluster job `597994.idark` (queue `mini2`, 3 CPUs on the only node with free cores) executed `frb_map_generation/run_halfdome_z1_nside4096_r200c_apertures_mass_histograms.pbs`. That script loops over `APERTURE_MULTIPLIERS="2.0 3.0:_repeat20260916 4.0 5.0"` and calls the unchanged generic workflow once per aperture with the same settings as the 3R200c and 1R200c products: 120k unique uniform rays on the z=1 plane (seed 42, NSIDE 4096), the complete `lightcone_100.hdf5` catalogue, the shared Battaglia16 M200c XGPaint DM cache, M200c windows and 300 log bins over [0.1, 30000] pc cm^-3. Only the generator-owned angular filter theta <= angular_size(N*R200c) changes. Each product passed the M200c/R200c provenance assertions with its own multiplier. Each aperture took about five minutes.

## Percentage of rays through at least one resolved halo

Each top panel of the figures prints this number per aperture for its mass window (1 - zero fraction; the zero-DM rays sit in an undrawn delta because the PDFs are normalized over in-range rays only).

| Window [Msun] | TNG | 1R200c | 2R200c | 3R200c | 4R200c | 5R200c |
|---|---:|---:|---:|---:|---:|---:|
| Total (1e10-1e16) | 100.0% | 64.7% | 98.2% | 100.0% | 100.0% | 100.0% |
| 1e10-1e13 | 100.0% | 13.7% | 44.5% | 73.1% | 90.2% | 97.3% |
| 1e10-1e14 | 100.0% | 60.1% | 97.2% | 100.0% | 100.0% | 100.0% |
| 1e10-1e15 | 100.0% | 64.7% | 98.2% | 100.0% | 100.0% | 100.0% |
| 1e13-1e14 | 52.6% | 53.8% | 95.1% | 99.8% | 100.0% | 100.0% |

Total-window summary statistics over all 120k rays (zero-DM rays included), from the generator summary files:

| Aperture | Ray/halo intersections | Mean DM [pc cm^-3] | Std DM |
|---|---:|---:|---:|
| 1R200c | 126,665 | 102.6 | 143.4 |
| 2R200c | 507,125 | 166.4 | 151.1 |
| 3R200c | 1,141,198 | 203.9 | 153.7 |
| 4R200c | 2,029,966 | 228.9 | 155.3 |
| 5R200c | 3,173,483 | 246.9 | 156.2 |

Observations:

- The hit fraction saturates quickly: 2R200c already reaches 98% of rays in the total window, 3R200c 99.98%, and 4-5R200c 100%. The 1e10-1e13 window (effectively 7.3e12-1e13 Msun) is the exception, rising from 14% at 1R200c to 97% at 5R200c. For 1e13-1e14 HalfDome exceeds TNG's 52.6% at every aperture above 1R200c because TNG's window is the same M200c range but its ray sample is a different realization.
- The mean DM keeps growing with aperture (103, 166, 204, 229, 247 pc cm^-3) because the Battaglia16 profile still carries gas far outside R200c; the high-DM tail above ~300 pc cm^-3 is nearly aperture independent.
- Each aperture has a characteristic lower DM cutoff set by a resolution-floor halo grazed at the aperture edge: about 22 (1R), 7 (2R), 3.5 (3R), 2 (4R) and 1.3 (5R) pc cm^-3. Below 1e-3 in p(DM) the wide apertures show a straight power-law shelf from these grazing hits.
- In the percentage panels, 1R200c and 2R200c sit at or below TNG for DM > 100 pc cm^-3, while 3-5R200c are 50-120% above TNG at 200-500 pc cm^-3. Any aperture choice also changes the zero-DM fraction, so the positive-DM PDFs alone do not fix the preferred aperture.

## Consistency check

The repeated 3R200c run and the historical 3R200c product share bin edges and give bit-identical histogram counts in every window (largest per-bin count difference 0; identical zero fractions and identical intersection count 1,141,198), although the repeat used the newer committed generator revision, 3 threads instead of 26, and a fresh job. The dotted repeat curve therefore lies exactly under the dashed 3R200c curve in the figures. Details: `analysis/repeat_3r200c_consistency.json`.

## Code changes

- Added `frb_map_generation/run_halfdome_z1_nside4096_r200c_apertures_mass_histograms.pbs` (uploaded to the cluster).
- `frb_map_generation/compare_tng_halfdome_direct.py`: `load_halfdome` accepts `run_name=` to load a suffixed directory such as the repeat run; all provenance checks still apply.
- `frb_map_generation/make_publication_comparisons.py`: `halo_pdf_aperture_series_figures` and the `--aperture-series-only` flag.

Regenerate with:

```bash
/home/cbllover/miniconda3/envs/halfdome/bin/python -m frb_map_generation.make_publication_comparisons --aperture-series-only --output frb_map_generation/outputs/publication_comparison_20260916_apertures --pdf output/pdf/halfdome_b16_r200c_apertures_vs_tng_20260916.pdf
```
