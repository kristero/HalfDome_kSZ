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

- `takahashi_fig13_updated_models` - Planck and ACT panels, observed redshifts, y beamed. For each density
  model the three truncations computed on the same rays: gas inside the R200c sphere (solid), inside the
  3R200c sphere (dotted) and the old truncation, i.e. the full-line-of-sight column inside an angular
  aperture of R200c, a cylinder (dash-dotted).
- `takahashi_fig13_updated_models_vs_previous` - the inside-R200c curves with the 2026-09-14 products
  (3R200c sphere; legacy Lee22 reading) as the previous implementation.
- `takahashi_fig13_lee22_calibrated_range` - Lee22 (R200c sphere) with all resolved halos versus only the
  halos inside its calibration ranges, Battaglia16 under both selections as reference.
- `takahashi_fig13_y_lee22p`, `takahashi_fig13_y_b12` - 100k curves and the mean of ten 71/31-FRB realizations,
  for the Lee22-pressure and the Battaglia12 Compton-y maps (the other map dotted for reference).
- `takahashi_fig13_realizations_y_lee22p`, `takahashi_fig13_realizations_y_b12` - the ten realizations
  individually, with the median and 16-84 % band of 1000 realizations and the observations.
- `cl_yy_lee22_pressure_vs_b12` - auto power spectra of the two y maps.
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

New: the per-source halo DM for ten models (`sample_halfdome_updated_sightlines.jl`), plus a third
source plane with every source at z = 2 for the Medlock & Nagai kernel. One pass over the
85,224,251 catalogue rows (74,907,259 halos with 0 < z <= 2.148) took about 60 s on 20 local threads.

| label | density fit | electron normalization | gas counted | purpose |
|---|---|---|---|---|
| `b16_sphere1` | Battaglia16 (XGPaint parameters) | XGPaint `ne2d` | inside the R200c sphere | the implementation that matches TNG within R200 |
| `b16_sphere3` | Battaglia16 | XGPaint `ne2d` | inside 3 R200c | the previous cross-correlation convention |
| `b16_projected1` | Battaglia16 | XGPaint `ne2d` | full LOS (1e5 R200c) inside an angular aperture of R200c (cylinder) | the old truncation of the PDF products |
| `lee22_noconc_sphere1` | Lee22 Table A2, no concentration | XGPaint-native (P0 = 200 n0, `ne2d`), M_cut pivot, fit-range shape clip | inside R200c | the Lee22 implementation that matches TNG |
| `lee22_noconc_sphere3` | same | same | inside 3 R200c | extrapolation beyond the 0.04-1.34 R200c fit range |
| `lee22_noconc_projected1` | same | same | full LOS inside an angular aperture of R200c (cylinder) | the old truncation, same fit and reading |
| `lee22_noconc_sphere1_calib` | same | same | inside R200c; only halos with 1.5e13 <= M200c < 9.3e14 Msun and z <= 2 | the fit used only where it was calibrated |
| `b16_sphere1_calib` | Battaglia16 | XGPaint `ne2d` | inside R200c; same halo selection | reference for the selection itself |
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

## 5b. The old truncation on the same rays (cylinder versus sphere)

`b16_projected1` and `lee22_noconc_projected1` reproduce the convention of the PDF products before the
fix: XGPaint's full-line-of-sight column (to 1e5 R200c) credited to every ray inside an angular aperture
of R200c, and nothing outside it. Compared with the R200c sphere on the same rays (Planck plane;
values in 1e-5 pc cm^-3, ratios to the sphere in brackets):

| annulus ['] | B16 cylinder | B16 sphere R200c | B16 sphere 3R200c | Lee22 cylinder | Lee22 sphere R200c | Lee22 sphere 3R200c |
|---|---:|---:|---:|---:|---:|---:|
| 10-17.8 | 1.53 (1.19) | 1.28 | 1.78 | 6.20 (1.10) | 5.62 | 6.71 |
| 17.8-31.6 | 0.81 (1.23) | 0.66 | 1.01 | 3.31 (1.13) | 2.93 | 3.74 |
| 31.6-56.2 | 0.38 (1.29) | 0.29 | 0.52 | 1.50 (1.18) | 1.27 | 1.82 |
| 56.2-100 | 0.15 (1.33) | 0.11 | 0.25 | 0.55 (1.24) | 0.44 | 0.78 |

The cylinder adds 35 % to the mean halo DM per ray for both models (Battaglia16 24.5 -> 33.1, Lee22
59.1 -> 80.6 pc cm^-3) and 10-35 % to the cross-correlation, more at large angles where the outer gas
matters. It sits between the two spheres everywhere: it contains the gas beyond R200c along the ray but
not the gas at b > R200c that the 3R200c sphere adds. The ACT plane gives the same ratios within 0.02.

## 5b2. Lee22 used only where it was calibrated

`lee22_noconc_sphere1_calib` keeps the R200c sphere (radii inside the 0.04-1.34 R200c fit range) and
counts only halos with 1e13 <= M200c h/Msun < 10^14.8 (1.47e13-9.28e14 Msun) and z <= 2, the mass and
redshift ranges of the Lee22 fit. `b16_sphere1_calib` applies the same selection to Battaglia16. The
removed halos are the resolved halos below 1.47e13 Msun (the bulk of the catalogue by number), the 868
foreground halos above 9.28e14 Msun, and the few above z = 2. Values in 1e-5 pc cm^-3:

| annulus ['] | observed Planck | Lee22 all halos | Lee22 calibrated | ratio | B16 all halos | B16 calibrated | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10-17.8 | 4.28 +- 1.61 | 5.62 | 3.19 +- 0.26 | 0.57 | 1.28 | 0.83 | 0.65 |
| 17.8-31.6 | 2.48 +- 1.30 | 2.93 | 1.66 +- 0.14 | 0.57 | 0.66 | 0.42 | 0.64 |
| 31.6-56.2 | 1.08 +- 0.87 | 1.27 | 0.79 +- 0.06 | 0.62 | 0.29 | 0.20 | 0.69 |
| 56.2-100 | 1.21 +- 0.36 | 0.44 | 0.33 +- 0.02 | 0.75 | 0.11 | 0.09 | 0.79 |
| 100-178 | 0.67 +- 0.35 | 0.16 | 0.13 +- 0.02 | 0.83 | 0.04 | 0.03 | 0.83 |

The selection removes 13 % of the mean halo DM per ray (Planck plane: 59.1 -> 51.6 pc cm^-3 for Lee22,
24.5 -> 20.0 for Battaglia16) but 40 % of the cross-correlation at theta <= 56' and 10-25 % beyond
100', for both models alike (ratios 0.57-0.63 versus 0.64-0.72 at small angles). The small-angle loss is
therefore not a property of the Lee22 fit but of the selection: the excluded halos above 9.28e14 Msun
carry the largest Compton-y and dominate the one-halo part of the signal, while the many excluded
low-mass halos carry little y. Lee22 loses slightly more than Battaglia16 because its DM per halo rises
faster with mass (n0 ~ M^0.68 against P0 ~ M^0.29).

Against the Planck points the calibrated Lee22 curve is within 0.7 sigma at 10-56' and 2.4 sigma low at
56-100'; against ACT it is 1.2-2.9 sigma high at 1.8-10' and within 0.8 sigma at 10-56'. Indicative
amplitudes (observed = A x model, diagonal digitized errors): Planck A = 1.80 +- 0.37 (all halos:
1.00 +- 0.22), ACT A = 0.43 +- 0.19 (all halos: 0.28 +- 0.12). The redshift caveat of Section 5 is
unchanged by this selection: z <= 2 is the fitted range, but the fit's low-redshift normalization exceeds
the cosmic baryon budget inside R200c, and the Takahashi halos are at z < 0.3.

## 5c. Why Battaglia16 is nearly self-similar and Lee22 is not

Both fits are written in self-similar variables: radius in R200c and density in units of the critical
density at the halo's redshift (Battaglia16 through `rho_gas = P0 f(x) f_b rho_cr(z)`, Lee22 through
`n200 propto rho_cr(z)`). A halo whose gas fraction and shape do not depend on mass or redshift is a single
curve in these variables. The two fits differ in how far their parameter power laws depart from that.

Battaglia16 (XGPaint parameters, m = M200c/1e14 Msun): `P0 = 4e3 m^0.29 (1+z)^-0.66`,
`alpha = 0.88 m^-0.03 (1+z)^0.19`, `beta = 3.83 m^0.04 (1+z)^-0.025`, `x_c = 0.5`, `gamma = -0.2`.
The amplitude changes by 3.8x over 1e13-1e15 Msun and by 0.48x from z = 0 to 2, but the shape moves the
other way: the outer slope `-(beta+gamma)/alpha` steepens from -3.5 at 1e13 to -4.9 at 1e15 Msun and
flattens with redshift, so the gas inside R200c changes much less than P0. The enclosed gas fraction
(Section 5 table) is 0.55-0.83 f_b over two decades in mass (M^0.09) and moves by less than 20 % between
z = 0.05 and 2. In the (n_e/n200)(r/R200c)^3 plot the curve peaks at 1.5-3 R200c with amplitude 0.15 in
every panel. This is by construction: the Battaglia simulations have gas fractions that rise gently with
mass and hardly evolve, and the fitted exponents encode only those small departures.

Lee22 no-concentration fit (M_cut = 10^13.61 h^-1 Msun): `n0 = 6.8 (M/M_cut)^0.68 (1+z)^-2.11`,
`x_c = 7.9 (1+z)^-0.67 B(M; 0.47, -0.45)`, `beta' = 19.5 (1+z)^-0.31 B(M; 0.70, -0.18)`, `alpha = 1`,
`gamma = -0.3`. Three things break the self-similarity:

1. **Redshift.** `(1+z)^-2.11` on the amplitude is a factor 0.23 at z = 1 and 0.10 at z = 2 relative to
   z = 0, in units that already scale with `rho_cr(z)`. Nothing in the shape compensates (x_c and beta'
   also shrink with redshift, which removes gas from the outskirts rather than adding it), so the enclosed
   fraction at 1e14 Msun falls from 2.4 f_b at z = 0.1 to 0.13 f_b at z = 2. TNG's gas fractions at fixed
   M200c are nearly constant in redshift, so this exponent most likely absorbs a unit convention of the
   fit (comoving versus physical densities, or rho_cr(0) versus rho_cr(z)), the open question of
   `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`.
2. **Mass.** The amplitude grows as M^0.68 while the shape parameters follow broken power laws whose
   slopes change sign at M_cut: below it the core radius and the cutoff steepness grow with mass
   (x_c ~ M^0.47, beta' ~ M^0.70), above it both shrink (M^-0.45, M^-0.18). The enclosed fraction goes
   from 0.9 f_b at 1e13 to 2.9 f_b at 1e15 Msun at z = 0.1, a factor 3 against Battaglia16's 1.5, and the
   curve shapes in the grid change visibly from row to row.
3. **Shape.** With x_c ~ 8 and beta' ~ 20 the profile is a shallow x^-0.3 power law that is cut off by
   `(1 + x/x_c)^-19.5`. The cutoff scale lies far outside the fitted radii (0.04-1.34 R200c), so x_c and
   beta' are strongly degenerate and only their combination is constrained; the logarithmic slope steepens
   quickly from -2.5 at R200c to -5.7 at 3 R200c. That is why the volume-weighted Lee22 curve peaks
   inside R200c and falls off steeply beyond it, while Battaglia16, with x_c = 0.5, has reached its
   asymptotic slope by R200c and keeps a broad distribution out to 3 R200c.

The practical consequence is the one seen in Sections 3 and 5: the two fits agree only in a narrow
redshift window (z ~ 0.3-0.8 for the XGPaint-native reading), and any comparison that weights other
redshifts, such as the z ~ 0.3 Takahashi samples or the z = 2 Medlock & Nagai kernel, separates them.

## 5d. Compton-y from the Lee22 no-concentration pressure fit

A second full-lightcone y map was painted with the Lee22 electron-pressure fit (arXiv v1 Table 7 =
MNRAS Table A1; `lee2022_tsz_pressure_profile.jl`, option `--tsz-profile=lee2022_noconc` of the tSZ
painter). Everything else equals the Battaglia12 map: all 85,224,251 halos, NSIDE 4096, XGPaint's
projected 4R200c aperture, no beam. The fit is P_e/P200 = P0 (x/x_c)^-0.3 [1 + x/x_c]^-beta with the
Battaglia P200 = 200 G M200 rho_cr f_b / (2 R200) and

| parameter | Lee22 Table 7 (M_cut = 10^13.60 h^-1 Msun) | Battaglia12 (m = M/1e14 Msun) |
|---|---|---|
| P0 | 2.8 (M/M_cut)^1.08 [< M_cut], ^0.80 [> M_cut], (1+z)^-1.89 | 18.1 m^0.154 (1+z)^-0.758 |
| x_c | 2.10 (M/M_cut)^-0.35 (1+z)^0.53 | 0.497 m^-0.00865 (1+z)^0.731 |
| beta | 9.4 (M/M_cut)^-0.06 (1+z)^0.60 | 4.35 m^0.0393 (1+z)^0.415 |

XGPaint paints 0.5176 P_th, so the wrapper passes P0/0.5176 (Lee22 fits the electron pressure) and
beta + 0.3 (XGPaint's exponent convention); x_c and beta are frozen above the fit-range mass, as for
the density fit. A self-test reproduces an independent eq. 9-10 Compton-y integral to 1e-6. The map
took 810 s on 20 threads; its mean y is 4.56e-7 against 1.04e-6 for Battaglia12 (0.44x). The annular
y samples at the 100k positions were recomputed for this map with the same filters
(`cross_finite_source_realizations.py sample-y`), and the cross-correlations below use them with the
unchanged per-source DM vectors.

## 5e. What a 71- or 31-FRB measurement looks like on this sky

`cross_finite_source_realizations.py` draws, for each survey plane, realizations that use exactly the
observed number of FRBs: one random ray per observed redshift (71 for Planck, 31 for ACT), disjoint
between realizations. Each realization is evaluated with the estimator of the full test,
w_b = (1/G) sum_g (D_g - <D>_g)(Y_gb - <Y>_gb), where the stratum means come from all 100k rays (the
analogue of the DM-z relation and the random-position y mean of the observational estimator). Ten
realizations are shown individually; 1000 more (also disjoint) give the distribution of a single
measurement. Battaglia12 y, R200c sphere, all halos, in 1e-5 pc cm^-3:

| plane, DM model | annulus ['] | 100k (ensemble mean) | mean of 1000 | median | 16-84 % of one realization | fraction above the mean |
|---|---|---:|---:|---:|---|---:|
| Planck, Battaglia16 | 10-17.8 | 1.28 | 1.29 +- 0.19 | 0.37 | 0.06 to 1.39 | 18 % |
| Planck, Battaglia16 | 31.6-56.2 | 0.29 | 0.30 +- 0.04 | 0.09 | -0.06 to 0.43 | 23 % |
| Planck, Lee22 | 10-17.8 | 5.62 | 5.74 +- 1.27 | 1.21 | 0.24 to 5.14 | 14 % |
| Planck, Lee22 | 31.6-56.2 | 1.27 | 1.33 +- 0.21 | 0.29 | -0.11 to 1.52 | 18 % |
| ACT, Battaglia16 | 10-17.8 | 1.26 | 1.56 +- 0.42 | 0.23 | -0.11 to 1.21 | 15 % |
| ACT, Lee22 | 10-17.8 | 5.63 | 7.69 +- 2.85 | 0.64 | -0.11 to 4.25 | 12 % |

The estimator is unbiased: the mean over 1000 realizations agrees with the 100k value within its
error. But the distribution of a single 71- or 31-source measurement is extremely skewed. Its median
is 0.2-0.3 of the mean at theta <= 30', only 12-23 % of realizations exceed the mean, and the standard
deviation of one realization is 5-15 times the mean (ACT, Lee22, 5.6-10': 145 against 9.2). The reason
is the halo-only DM itself: 75 % of the rays intersect no halo inside R200c, and the cross-correlation at
small angles is carried by the few sightlines through massive, high-y clusters. A set of 71 or 31
sightlines usually contains none of them and then sits far below the ensemble mean; occasionally it
contains one and then lies far above (single realizations reach 4x the mean at 8' in the figure). The
ten realizations plotted happen to contain no such sightline, so their average is 0.5-0.6 of the mean
for Planck and 0.2-0.5 for ACT; that is sampling, not a bias.

Two consequences for the comparison with Takahashi et al. First, a halo-only ensemble-mean curve is
not what a single 71-source measurement is expected to look like; the median curve and the 16-84 %
band are the relevant reference, and the observed Planck points lie inside or above that band for
both density models. Second, the jackknife error of a real 71-source measurement cannot capture the
missing rare sightlines, as already noted in `PUBLICATION_COMPARISONS_20260915.md`. These
realizations sample directions on one fixed sky; they contain no observational noise, mask, host or
IGM scatter.

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
