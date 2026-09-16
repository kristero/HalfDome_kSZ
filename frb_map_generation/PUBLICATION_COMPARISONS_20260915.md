# Readable PDF and finite-FRB comparisons

Completed locally on 2026-09-15. Existing individual sightlines and halo-PDF products were reused; no halo painting, XGPaint modification, or cluster submission was needed.

## Open the results

- Combined, 11-page vector PDF: `/home/cbllover/HalfDome/output/pdf/halfdome_publication_comparisons_20260915.pdf`.
- Individual PNG and editable SVG figures: `/home/cbllover/HalfDome/frb_map_generation/outputs/publication_comparison_20260915/plots/`.
- Updated top sections of `DM_halo_pdf_HalfDome_catalog-Copy1.ipynb` and `FRB_tSZ_Takahashi_comparison.ipynb` contain the rendered figures. Existing sections were preserved.
- Figure captions and scientific qualifications: `figure_captions.json` under the new output root. Long explanations are outside the figures; labels, ticks and legends use large fonts.

The PDF pages are:

1. Battaglia16 versus TNG: total and upper mass limits.
2. Battaglia16 versus TNG: total and windows ending at 1e14 solar masses.
3. The first comparison with Lee22 added.
4. The second comparison with Lee22 added.
5. Planck: 71, 10k and 100k sightlines versus Takahashi and the previous full-map prediction.
6. ACT: 31, 10k and 100k sightlines versus Takahashi and the previous full-map prediction.
7. Battaglia16 versus the digitized Medlock & Nagai BP reference.
8. Lee22 without concentration versus the same reference.
9. Lee22 with concentration versus the same reference, explicitly a diagnostic test.
10. Verification of matched source-redshift distributions.
11. Small-catalogue resampling diagnostic alongside the selected catalogue's jackknife interval.

## What was implemented

### Real, nested individual sightlines

The parent directory is `frb_map_generation/outputs/takahashi_100k_20260914`. Each of its 100k rows already contains a source position, source redshift, individually computed halo DM for each density model, and annular tSZ samples around that position. No mean-map DM was substituted for an individual ray.

Using selection seed `20260915`, the new analysis independently constructs a Planck and an ACT sample:

- Choose 10,000 parent rows without replacement, with balanced counts among the 71 or 31 observed-redshift entries.
- Choose one of those rows per observed entry for the 71- or 31-source catalogue. The small catalogue is therefore nested in the 10k catalogue; both are nested in the existing 100k parent. All density models use the same selected positions.
- Reserve the other 90,000 rows to estimate the mean halo DM at each observed redshift, separately for each density model. These calibration rows do not overlap either smaller sample.
- Retain sources with zero foreground intersections. There are 20 such sources in the Planck 71-source selection and 7 in the ACT 31-source selection.

The 71/31 source redshifts reproduce the observed list exactly, not just a smoothed approximation to its histogram. Their mean redshifts are respectively 0.274607 and 0.323455. The 10k sample uses weights `1/(G*n_g)`, where `G` is the number of observed entries and `n_g` is the selected count at that entry, so every entry has exactly weight `1/G`. Positions remain the saved random full-sky positions, not the actual observed positions and not simulated host-halo placements.

### Cross-correlation and jackknife

For source `i` and angular bin `b`, the measured contribution is

`q_i(b) = [DM_i - mean_DM_calibration(z_i)] * annular_y_i(b)`.

The annular y samples come from the mean-subtracted tSZ map. With normalized source weights `a_i`, the estimate is `w(b) = sum_i a_i*q_i(b)`. For the 71/31 samples the weights are equal.

For every source, delete it and renormalize the remaining weights:

`w_minus_i = [w - a_i*q_i] / (1-a_i)`.

The full jackknife covariance is

`C_JK = (N-1)/N * sum_i [(w_minus_i - mean(w_minus)) outer (w_minus_i - mean(w_minus))]`.

The plotted error is `sqrt(diag(C_JK))`. Thus the 71-source result has 71 explicit delete-one replicates and the 31-source result has 31. The 10k samples also use explicit delete-one-source jackknifes; their errors are not rescaled from the 100k error bars. Cross-model and cross-angular-bin covariance is retained in the saved matrix.

The independent 90k mean-DM calibration is held fixed during this jackknife. In particular, the code does not estimate mean DM from the single retained source at each observed redshift, which would incorrectly force every residual to zero. Calibration uncertainty is not propagated.

The 100k curves and errors are copied unchanged from the previous analysis. That estimator used the unbiased within-redshift covariance, its own fitted stratum means and a stratified delete-one jackknife. It targets the same mean correlation but is not algebraically identical to the independently calibrated smaller-sample estimator. This difference is recorded in the provenance rather than silently changing the previous result.

### Physics retained from the parent products

For the cross-correlation, DM includes resolved foreground halos individually between observer and each source, with a spherical 3R200c cutoff and the observer-frame redshift factor. The parent calculation scanned all 85,224,251 catalogue rows, rather than choosing a fraction of foreground halos. It retained 74,907,259 halos in the redshift range needed by the sources, up to source redshift 2.148, with the additional foreground condition applied separately for each ray.

tSZ retains Battaglia12 pressure and the full available halo light cone, including halos beyond the FRB source. Its existing projected extent is unchanged. The y field is filtered with the parent Planck 10-arcmin or ACT 1.6-arcmin beam for the Takahashi comparison. No new noise or survey masks were added.

The plotted correlation therefore responds both to individual halo intersections and to angular correlations with surrounding pressure. A rare massive foreground can dominate a small sample, so more sources do not guarantee monotonic changes in the measured signal.

The historical halo-PDF products are deliberately different from the later spherical-cut correlation test: the PDFs retain the original projected 3R200c aperture and original line-of-sight prescriptions, at z=1, NSIDE=4096, 120k rays. Reformatting those figures does not recompute their physics. Both sets use their saved M200c selections; these two aperture conventions must not be conflated.

### PDF normalization and percentages

The old histogram bins and original density normalization are preserved: counts divided by the number counted in the stored `[0.1, 30000]` pc cm^-3 range and the bin width. This is a positive-DM, finite-range PDF, not a representation of the point probability at DM=0. No new mass selections were synthesized by subtraction, and no smoothing was applied.

The lower panels show linear `100*(HD-TNG)/TNG`, only where both counts are at least 10. The total HalfDome curve uses the full stored resolved mass window, not an upper cutoff at 1e14 solar masses. The approximate effective lower mass limit is 7.327e12 solar masses. The 1e10-1e12 HalfDome window therefore has no positive-DM PDF and no defined percentage difference; it is not drawn as an artificial -100% curve.

For the cross-correlation figures, lower-panel percentages instead use the previous full-map prediction as denominator. They are shown only where its absolute value exceeds 1% of its peak absolute amplitude; noisy, near-zero observed points are not used as percentage denominators.

## Results and interpretation

For illustration, in the 10-17.8 arcmin bin the new measurements are below. Both the values and jackknife standard errors are in units of `1e-5 pc cm^-3`.

| Selection | N | Battaglia16 | Lee22, no concentration |
|---|---:|---:|---:|
| Planck | 71 | 1.521 +/- 0.748 | 5.792 +/- 3.334 |
| Planck | 10,000 | 1.237 +/- 0.120 | 4.340 +/- 0.477 |
| Planck | 100,000 | 1.780 +/- 0.161 | 7.094 +/- 0.813 |
| ACT | 31 | 0.431 +/- 0.444 | 0.980 +/- 1.332 |
| ACT | 10,000 | 3.221 +/- 1.365 | 14.528 +/- 7.058 |
| ACT | 100,000 | 1.778 +/- 0.165 | 7.075 +/- 0.830 |

The small catalogues fluctuate substantially; the 10k result is not necessarily closer to the full-map mean than the 71/31 realization. A separate set of 5,000 redshift-matched 71/31 resamples, drawn from the held-out 10k pool, demonstrates strongly skewed conditional sampling distributions. The diagnostic retains all draws, including the rare tails, and does not replace the requested jackknife bars.

Important limitations:

- Source-delete-one jackknife is not a spatial survey jackknife and cannot reveal rare halos absent from a selected sample. Its interval is not a guarantee of coverage of the full-sky prediction.
- The resampling diagnostic is conditional on a single simulated sky and its finite 10k pool; it is not an ensemble of independent cosmologies.
- Neither error estimate includes host or diffuse/intergalactic DM, instrumental noise, survey masks, calibration uncertainty or full cosmic variance. The comparison remains a halo-only partial prediction, not an exact observational likelihood.
- Each saved covariance spans 72 quantities: two filters, three density models, twelve angular bins. The 71/31-source matrices have rank at most 70/30; do not directly invert them for a joint 72-dimensional fit.
- Takahashi values and their observational errors are approximate digitizations, not an author-provided data vector or covariance matrix.
- The Isabel reference is the previously digitized best-fit angular correlation from Medlock & Nagai Figure 5, not a digitized three-dimensional electron-density profile. Its source population is at z=2. The reference figures use unbeamed HalfDome curves but retain the observed source-redshift distributions, so they are qualitative overlays, not matched-kernel parameter tests. Their narrow gray reference band represents digitization uncertainty, not a model posterior.
- The concentration-dependent Lee22 extrapolation retains the existing gas-budget caveat and is labeled `Lee22 + c (test)`; it is not promoted to a validated physical model.

## Saved analysis and checks

Under `frb_map_generation/outputs/publication_comparison_20260915/`:

- `catalogues/nested_sightlines.h5`: chosen parent indices, source coordinates/redshifts, halo DM, weights, 90k calibration, all jackknife replicates, covariance matrices and diagnostic resamples. Example group: `planck/n71/`; covariance array order is stored as an attribute.
- `analysis/source_count_comparison.csv`: point estimates, standard errors, old full-map values and error-method labels for every source count, model, filter and angular bin.
- `analysis/source_count_provenance.json`: random seed, sample sizes, calibration and error limitations. Parent-file hashes are also stored in the HDF5 catalogue.
- `analysis/halo_pdf_comparison_bins.csv`: original histogram counts, PDF values, zero fractions and masked percentage differences.
- `analysis/validation_checks.json`: independent checks of nesting, calibration disjointness, redshift weights and saved jackknifes.

Validation performed:

- Algebraic delete-one estimates checked against explicitly deleting each row on a toy weighted sample; equal-weight covariance checked against sample covariance divided by N.
- Every real 71/31 replicate independently recomputed and checked. Three representative replicates per 10k sample independently recomputed; the complete stored covariance checked against all saved replicates.
- Exact observed-redshift matching, nesting, disjoint calibration, original histogram count preservation, source-position checksums and complete parent halo-catalogue scan checked.
- All 11 PDF pages rendered and visually inspected; revised layouts checked again for clipping and title collisions.
- Only the three newly added notebook code cells were executed, successfully. Historical notebook sections and their outputs were preserved, not rerun or newly validated.

## Files added or edited

Added `frb_map_generation/compare_takahashi_source_counts.py` for subset selection, estimation, jackknife, provenance and tests; added `frb_map_generation/make_publication_comparisons.py` for all figure exports; added this report. Edited the two notebooks named above by adding self-contained sections. Added `tmp/execute_publication_sections_20260915.py` as a validation helper that executes only the new notebook cells. No profile-generation scripts or XGPaint files were changed.

To check the saved estimates and regenerate figures, from `/home/cbllover/HalfDome` in the existing Python environment:

```bash
python frb_map_generation/compare_takahashi_source_counts.py --self-test
python frb_map_generation/compare_takahashi_source_counts.py --check-output
python -m frb_map_generation.make_publication_comparisons
```

The figure command refreshes only this new export set. To create a different random selection, use the analysis script with a new `--output` directory and explicit `--seed`; it refuses to replace an existing saved catalogue.
