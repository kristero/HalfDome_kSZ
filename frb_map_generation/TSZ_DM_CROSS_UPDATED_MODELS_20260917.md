# tSZ x FRB-DM cross-correlation with the updated Battaglia16 and Lee22 implementations

Date: 17 September 2026. Compares the HalfDome halo-only prediction, computed with the density
implementations validated against IllustrisTNG on 16-17 September (finite spherical boundary,
XGPaint-native Lee22 normalization), with

- Takahashi et al. 2025, [arXiv:2511.02155v2](https://arxiv.org/abs/2511.02155v2), Figure 13
  (Planck MILCA, 71 localized FRBs, 10' beam; ACT, 31 FRBs, 1.6' beam), and
- Medlock & Nagai 2026, [arXiv:2608.06455v1](https://arxiv.org/abs/2608.06455), Figure 5
  (Baryon Pasting, "BP", maximum-likelihood curves for the ACT, Planck MILCA and Planck NILC
  panels; the paper places every FRB at z = 2 and shows un-beamed model curves).

Figures (PNG/PDF/SVG) under `outputs/tsz_dm_cross_updated_20260917/plots/`:

- `takahashi_fig13_updated_models` - Planck and ACT panels, observed redshifts, y beamed: the inside-R200c
  curves with, as dotted lines, the previous implementation of each (the 2026-09-14 products:
  Battaglia16 and legacy Lee22, both with the 3R200c sphere).
- `takahashi_fig13_updated_models_vs_3r200c` - the same with the recomputed updated models to 3R200c instead.
- `medlock_fig5_updated_models` - ACT, Planck MILCA, Planck NILC panels, all sources at z = 2, no beam.
- `updated_vs_previous_diagnostic` - the new products against the 2026-09-14 sightline products.

Two-page vector book: `output/pdf/halfdome_tsz_dm_cross_updated_models_20260917.pdf`.
Tables: `outputs/tsz_dm_cross_updated_20260917/analysis/annulus_tables.md` (and `.csv`).

## 1. What was recomputed and what was reused

Reused unchanged from the 2026-09-14 test (`TAKAHASHI_100K_SIGHTLINES.md`):

- the 100,000 random source directions (NSIDE 4096 pixel centres, seed 20260914) and their
  stratified assignments of the 71 Planck and 31 ACT observed redshifts (mean z 0.275 and 0.323);
- the annular Compton-y samples at those positions (Battaglia12 pressure, full lightcone, XGPaint's
  projected 4R200c footprint), for the 12 logarithmic annuli 1-1000 arcmin, with the 10' Planck
  beam, the 1.6' ACT beam, or no beam applied to y only;
- the estimator: within each observed-redshift stratum the unbiased sample covariance of the
  halo-only DM and the annular y, averaged over strata with equal weight; delete-one-source
  jackknife errors. These errors measure finite-source sampling on one simulated sky only.

New: the per-source halo DM for six models (`sample_halfdome_updated_sightlines.jl`), plus a third
source plane with every source at z = 2 for the Medlock & Nagai kernel. One pass over the
85,224,251 catalogue rows (74,907,259 halos with 0 < z <= 2.148) took 56 s on 20 local threads.

| label | density fit | electron normalization | gas counted | purpose |
|---|---|---|---|---|
| `b16_sphere1` | Battaglia16 (XGPaint parameters) | XGPaint `ne2d` | inside the R200c sphere | the implementation that matches TNG within R200 |
| `b16_sphere3` | Battaglia16 | XGPaint `ne2d` | inside 3 R200c | the previous cross-correlation convention |
| `lee22_noconc_sphere1` | Lee22 Table A2, no concentration | XGPaint-native (P0 = 200 n0, `ne2d`), M_cut pivot, fit-range shape clip | inside R200c | the Lee22 implementation that matches TNG |
| `lee22_noconc_sphere3` | same | same | inside 3 R200c | extrapolation beyond the 0.04-1.34 R200c fit range |
| `lee22_pref_sphere1` | Lee22 Table 3 + TNG-mean concentration | same reading | inside R200c | diagnostic only (not plotted in the main figures) |
| `lee22_legacy_sphere3` | Lee22 Table A2 | previous reading: literal eq. 9, 1e14 pivot, no clip | inside 3 R200c | regression against the 2026-09-14 product |

"Inside the sphere" means the chord-limited line of sight of the like-for-like TNG comparison
(`LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md`): a ray at impact parameter b receives the integral of
the 3-D electron density over the chord of half-length sqrt(X^2 R200c^2 - b^2) only, so the DM goes
to zero continuously at the edge. The previous cross-correlation already used this construction at
3 R200c; the 1 R200c products are new.

## 2. Validation

- Cache interpolation: the radius-coordinate cache (dimensionless chord-mean shape on a 321 x 65 x
  131 grid in log b/R200c, z, log M) reproduces direct quadrature to <= 0.06 % on 100 random points
  per model. The first attempt failed at 1.24 % at one point next to the fit-range shape-clip mass
  (log10 M = 14.964 vs. the clip at 14.9675); the clip mass is now a grid node, like the two M_cut
  pivots (`radius_scaled_dm_cache.jl`).
- Independent cross-check: 60 points per model against `chord_dm_pc_cm3` in
  `lee2022_frb_dm_profile.jl`, the profile-owned chord integral used for the TNG comparison, which
  shares no code with the cache's shape/amplitude split: <= 0.06 %.
- The XGPaint-native reading through the plain Lee22 model (normalization factor 0.602 on the
  printed eq. 9) and through the XGPaint `get_params` wrapper (P0 = 200 n0, XGPaint's own `ne2d`
  constants) agree to 1e-9.
- Regression: the recomputed `b16_sphere3` and `lee22_legacy_sphere3` DM vectors reproduce the
  2026-09-14 `battaglia16` and `lee22_legacy` vectors ray by ray to 3e-12 of the peak DM.
- Zero-hit fractions: 75 % (Planck plane) and 73 % (ACT plane) of the rays intersect no halo within
  R200c; 23 % and 20 % intersect none within 3 R200c; at z = 2, 26 % and 0.002 %.

## 3. Results against Takahashi et al. 2025

Annulus means of w_yDM in 1e-5 pc cm^-3 (HalfDome +- jackknife; observed values are approximate
digitizations with correlated errors). Full tables in `analysis/annulus_tables.md`.

Planck MILCA (10' beam, 71 redshifts), bins above the paper's 10' cut:

| annulus ['] | observed | B16 inside R200c | Lee22 no-c inside R200c | B16 to 3R200c | Lee22 no-c to 3R200c | Lee22 previous, 3R200c |
|---|---:|---:|---:|---:|---:|---:|
| 10-17.8 | 4.28 +- 1.61 | 1.28 +- 0.15 | 5.62 +- 0.95 | 1.78 +- 0.16 | 6.71 +- 0.98 | 7.09 +- 0.81 |
| 17.8-31.6 | 2.48 +- 1.30 | 0.66 +- 0.07 | 2.93 +- 0.44 | 1.01 +- 0.08 | 3.74 +- 0.45 | 3.97 +- 0.39 |
| 31.6-56.2 | 1.08 +- 0.87 | 0.29 +- 0.03 | 1.27 +- 0.15 | 0.52 +- 0.03 | 1.82 +- 0.16 | 1.96 +- 0.15 |
| 56.2-100 | 1.21 +- 0.36 | 0.11 +- 0.01 | 0.44 +- 0.04 | 0.25 +- 0.01 | 0.78 +- 0.04 | 0.86 +- 0.04 |
| 100-178 | 0.67 +- 0.35 | 0.04 +- 0.01 | 0.16 +- 0.02 | 0.10 +- 0.01 | 0.31 +- 0.02 | 0.36 +- 0.02 |

ACT (1.6' beam, 31 redshifts), first bins above the paper's 1.78' cut:

| annulus ['] | observed | B16 inside R200c | Lee22 no-c inside R200c | B16 to 3R200c | Lee22 no-c to 3R200c |
|---|---:|---:|---:|---:|---:|
| 1.78-3.16 | 3.79 +- 4.58 | 4.27 +- 0.33 | 15.28 +- 1.83 | 5.15 +- 0.35 | 17.06 +- 1.87 |
| 3.16-5.62 | 2.21 +- 1.98 | 3.16 +- 0.29 | 12.37 +- 1.75 | 3.98 +- 0.31 | 14.05 +- 1.80 |
| 5.62-10 | 2.65 +- 2.93 | 2.14 +- 0.25 | 9.19 +- 1.58 | 2.84 +- 0.26 | 10.66 +- 1.62 |
| 10-17.8 | 5.37 +- 3.07 | 1.26 +- 0.15 | 5.63 +- 0.97 | 1.78 +- 0.16 | 6.77 +- 1.00 |
| 17.8-31.6 | 2.57 +- 2.13 | 0.65 +- 0.07 | 2.89 +- 0.42 | 1.01 +- 0.08 | 3.72 +- 0.44 |

What the numbers say:

1. Battaglia16 inside R200c is a factor 3-4 below the Planck measurement at 10-56' and a factor
   ~10 below at 56-100'. Counting gas to 3 R200c raises it by 1.4x at 10-18' and 2.3x at 56-100',
   still well below. At the ACT small angles (1.8-10') Battaglia16 lies within the observational
   errors (all within 0.5 sigma of the digitized points).
2. Lee22 no-c inside R200c reproduces the Planck points at 10-56' (0.2-0.8 sigma) and falls below
   them beyond 56' (2 sigma at 56-100'). It overshoots the ACT points at 1.8-10' by 2-5 sigma of
   the digitized diagonal errors. Its signal is 3.4-4.5x Battaglia16 in every bin (Section 5).
3. Indicative amplitude fits (observed = A x model, diagonal digitized errors only, not a
   likelihood; Takahashi's covariance is not available): Planck A = 4.5 +- 0.9 for Battaglia16
   inside R200c and 1.00 +- 0.22 for Lee22 inside R200c; ACT A = 1.08 +- 0.48 and 0.28 +- 0.12.
   Takahashi et al. quote 2.01 +- 0.50 (Planck) and 1.23 +- 0.82 (ACT) relative to their HMx
   prediction. Read these as "which scale of gas normalization the data prefer", not as fits.
4. Boundary dependence: the fraction of the 3R200c signal that comes from gas inside R200c is
   0.72-0.80 at theta <= 18' and decreases to 0.45 at 56-100' for Battaglia16 (0.84-0.88 and 0.57
   for Lee22). The outer gas matters most on the large scales where both models are lowest.
5. Updated versus previous Lee22 reading (both to 3 R200c): the mean DM per ray drops by 15 %
   (the expected 1.42 x 0.602 = 0.855), but the cross-correlation drops by only 5-7 % at
   theta <= 30' and 10-15 % at theta >= 100'. The cross signal is carried by the most massive
   halos, where the fit-range shape clip (x_c and beta' frozen above 9.28e14 Msun) keeps the
   profile more extended than the unclipped legacy fit and partly compensates the lower
   normalization.

## 4. Results against Medlock & Nagai 2026

The BP curves in their Figure 5 assume every FRB at z = 2 and no beam ("We do not include an FRB
redshift source kernel in our calculations, which effectively assumes all FRB sources are at
z = 2"), integrate halo masses 1e13-1e16 Msun over z = 0.01-2 out to 5 R200m, and are fit
separately to each data set (Sharma et al. 2026 CHIME x Planck NILC with DM cuts; Takahashi et
al. ACT, Planck MILCA, Planck NILC). The HalfDome curves in `medlock_fig5_updated_models` therefore
use the z = 2 source plane and un-beamed y, so the source kernel and the beam treatment are
matched; the halo mass floor (6.8e12 Msun), the pressure model and the outer boundary are not.

At the annulus centres (1e-5 pc cm^-3):

| annulus ['] | BP ACT | BP MILCA | BP NILC | B16 inside R200c | Lee22 no-c inside R200c | B16 to 3R200c |
|---|---:|---:|---:|---:|---:|---:|
| 1.78-3.16 | 3.47 | - | - | 6.89 +- 0.35 | 20.9 +- 1.9 | 9.45 +- 0.37 |
| 5.62-10 | 0.67 | - | - | 2.68 +- 0.25 | 10.5 +- 1.6 | 3.96 +- 0.26 |
| 10-17.8 | 0.20 | 0.71 | 1.52 | 1.49 +- 0.15 | 6.21 +- 0.98 | 2.31 +- 0.16 |
| 17.8-31.6 | 0.04 | 0.26 | 0.55 | 0.74 +- 0.07 | 3.14 +- 0.43 | 1.24 +- 0.08 |
| 31.6-56.2 | 0.02 | 0.07 | 0.12 | 0.32 +- 0.03 | 1.36 +- 0.15 | 0.63 +- 0.03 |

With the same z = 2 kernel, Battaglia16 inside R200c is 2x the ACT-panel BP fit at 2-3' and
4x at 6-10', 2x the MILCA-panel fit and equal to the NILC-panel fit at 10-18', and above all three
fits by 3-15x beyond 30'. Lee22 inside R200c is a further factor 3-4 higher. Interpretation
(not tested here): the data were taken with sources at z ~ 0.3, so modelling them with sources at
z = 2 places about 4x more foreground halo column per source into the model (mean Battaglia16
DM inside R200c per ray: 95 versus 25 pc cm^-3), and the BP fits
compensate with a low gas normalization (strong feedback). HalfDome at z = 2 with a fixed
Battaglia16 gas content is correspondingly higher than those fits, and the same HalfDome model with
the observed redshifts (Section 3) is below the same data.

## 5. Why Lee22 is 4x Battaglia16 here although both matched TNG

The like-for-like TNG comparison used z_s = 1 rays, so its halos were at z ~ 0.3-0.8. The
Takahashi samples have mean redshifts 0.27 and 0.32, so the halos that dominate the cross signal
are at z < 0.3. The Lee22 fit scales its normalization as n0 ~ (1+z)^-2.11, Battaglia16 does
not, and with the reading that matches TNG at z_s = 1 the two fits diverge at low redshift.
Enclosed gas fraction relative to the cosmic baryon fraction, f_gas(< X R200c)/f_b, from the
profile-owned 3-D density with XGPaint's electron-per-mass convention for both models
(`enclosed_gas_fraction_table.jl`; full table in `analysis/enclosed_gas_fraction_table.txt`):

| z | M200c [Msun] | B16, X = 1 | Lee22 no-c, X = 1 | B16, X = 3 | Lee22 no-c, X = 3 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1e13 | 0.55 | 0.90 | 1.41 | 4.06 |
| 0.1 | 1e14 | 0.70 | 2.36 | 1.56 | 4.47 |
| 0.1 | 1e15 | 0.83 | 2.94 | 1.61 | 3.81 |
| 0.3 | 1e14 | 0.69 | 1.47 | 1.53 | 2.64 |
| 0.5 | 1e14 | 0.68 | 0.97 | 1.51 | 1.69 |
| 1.0 | 1e14 | 0.65 | 0.42 | 1.45 | 0.68 |
| 2.0 | 1e14 | 0.61 | 0.13 | 1.35 | 0.19 |

Inside R200c the Lee22 fit holds 1.5-3.4 times the cosmic baryon share for M >= 1e14 Msun at
z <= 0.3, which is not a physical gas content and is not what TNG contains at those redshifts;
Battaglia16 stays at 0.55-0.83 f_b at all redshifts. The Lee22 cross-correlation for the
Takahashi samples is therefore not a validated prediction: the reading that reproduces TNG at
z ~ 0.3-0.8 gives 2-3.5x the Battaglia16 gas at z <= 0.3 and 0.2-0.65x at z >= 1. The
z = 2 plane averages over both regimes (mean DM per ray: Lee22/B16 = 1.4 at z = 2 versus 2.4 for
the Planck plane). Settling the redshift dependence needs a TNG comparison at a low-redshift
source plane (z_s ~ 0.3), which the PDF generator can produce with `ZSOURCE=0.3`, and a
corresponding TNG catalogue at that redshift.

## 6. Limits that carry over

Halo-only partial prediction (no diffuse IGM, host or Milky Way DM), no survey noise, masks,
component-separation effects or covariance likelihood; equal source weights; the observed values
are approximate digitizations with correlated errors; HalfDome halos below 6.8e12 Msun are absent;
sources are random directions, not host halos. The Lee22 fits are extrapolated below their
1e13 h^-1 Msun mass floor and, for the 3 R200c products, beyond their 1.34 R200c radial range.

## 7. Files

- `sample_halfdome_updated_sightlines.jl` (new): the six-model sightline pass with the z = 2
  plane, per-model cache validation, the profile-owned chord cross-check and the XGPaint-route
  check. Output `outputs/tsz_dm_cross_updated_20260917/rays/individual_dm_updated.h5` (+
  `_provenance.txt`), caches under `cache/`.
- `radius_scaled_dm_cache.jl` (edited): Lee22 amplitude now multiplies the model's normalization
  reading and redshift-scaling option (identity for the legacy literal reading); the shape-clip
  mass is a grid node.
- `compare_updated_sightlines.py` (new): estimates, jackknife covariances, regression check,
  figures and tables. Run from the repository root:
  `python frb_map_generation/compare_updated_sightlines.py all`.
- `enclosed_gas_fraction_table.jl` (new): the Section 5 table.
- `FRB_tSZ_Takahashi_comparison.ipynb`: new final section displaying these figures and tables.
- Smoke-test and production logs under `outputs/tsz_dm_cross_updated_20260917/logs/`.
