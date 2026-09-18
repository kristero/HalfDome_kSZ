# Full-sky tSZ x FRB-DM comparison with the updated implementations ("cluster" method, local run)

Date: 18 September 2026. The three figure sets requested on 18 September, all against the digitized
Takahashi et al. 2025 Figure 13 points ([arXiv:2511.02155v2](https://arxiv.org/abs/2511.02155v2);
Planck MILCA, 71 localized FRBs, 10' beam; ACT, 31 FRBs, 1.6' beam):

1. `fullsky_takahashi_fig13` - the full-sky (map x map) prediction, no sightlines: Battaglia12
   pressure x Battaglia16 density, Lee22 pressure x Lee22 density, and Lee22 pressure x Lee22 density
   with both restricted to the halos inside the Lee22 calibration ranges.
2. `realizations_selected_battaglia`, `realizations_selected_lee22` - realizations with the observed
   numbers of FRBs (71 Planck, 31 ACT): the mean of 1000, the two highest, and the eight that fit the
   Takahashi points best; the Lee22 figure has one row for all halos and one for the calibrated ranges
   (both the y map and the DM).
3. `fullsky_power_spectra_three_column` - tSZ auto, FRB-DM auto and tSZ x FRB-DM power spectra of the
   same full-sky maps, for the three (pressure, density) pairs.

Everything ran locally on 20 threads (the 13 September products of this kind were computed on idark;
the method is identical, `paint_halfdome_observed_source_dm_maps.jl` -> `compare_halfdome_takahashi.py`).
Figures (PNG/PDF/SVG) and tables are under `outputs/tsz_dm_fullsky_20260918/`; vector book
`output/pdf/halfdome_tsz_dm_fullsky_20260918.pdf`.

## 1. Maps

All maps: NSIDE 4096 (0.86' pixels), the complete HalfDome `lightcone_100` catalogue (85,224,251 rows),
physical M200c (catalogue Msun/h divided by 0.68), no beam, no noise, no mask.

Compton-y (XGPaint projected convention, 4 R200c angular aperture, full line of sight; unchanged from the
earlier y maps):

| key | pressure fit | halos | mean y |
|---|---|---|---|
| `b12` | Battaglia12 fiducial (`paint_halfdome_battaglia12_tsz_map.jl`, 2026-09-09) | all, all z | 1.04e-6 |
| `lee22p` | Lee22 no-c Table 7 through the XGPaint mapping P0/0.5176, beta+0.3 (2026-09-17) | all, all z | 4.56e-7 |
| `lee22p_calib` | same | 1e13 <= M200c/(h^-1 Msun) < 10^14.8 (1.47e13-9.28e14 Msun), z <= 2 | 3.97e-7 (27,007,042 halos; max y 1.65e-4 against 1.64e-3 with all halos) |

Halo DM (`paint_halfdome_kernel_weighted_dm_maps.jl`, new): the validated radius-coordinate caches of the
17 September sightline test (`b16_sphere1`, `lee22_noconc_sphere1`; grid and profile-owned chord checks
<= 0.04 %), finite spherical boundary at R200c with the chord-limited line of sight, and for each survey
the observed-redshift kernel of 13 September (`kernels/planck_sources.csv`, `act_sources.csv`, equal source
weights, sha256 34e75a77... and 17ecfa60...):

    DM_survey(n) = sum_halo w_survey(z_halo) DM_halo(n),
    w_survey(z_halo) = fraction of the survey's sources with z_source >= z_halo.

Because the cross-correlation is linear in the DM map, <y DM_survey> over the full sky equals the average
over the observed source redshifts of the full-sky <y DM(z_source)>: the ensemble mean of the stratified
sightline estimator (one stratum per observed redshift) without any finite-sightline sampling noise, and
without any dependence on where the sources are placed. It is not a realization of 71 or 31 FRBs. Six maps
(three density models x two kernels) were painted in one catalogue pass (379 s for the 74,907,259 halos with 0 < z <= 2.148; 2.8e8 pixel updates per map):

| model | density fit | halos | mean DM Planck kernel | mean DM ACT kernel |
|---|---|---|---|---|
| `b16_sphere1` | Battaglia16 (XGPaint), ne2d electrons | all resolved, 0 < z <= 2.148 | 24.19 | 26.83 |
| `lee22_noconc_sphere1` | Lee22 Table A2 no-c, XGPaint-native reading (P0 = 200 n0), M_cut pivot, shape clip | same | 58.50 | 62.86 |
| `lee22_noconc_sphere1_calib` | same | 1.47e13 <= M200c/Msun < 9.28e14, z <= 2 | 51.12 | 54.93 |

## 2. Validation

- Painter self-test (`--rows=85000001:85012000 --self-test-pixels=400`, 12,000 halos at z = 0.07-0.14
  incl. the most massive low-z clusters): every painted value on 500 random pixels x 6 maps (2194 non-zero
  entries) equals a brute-force sum over all halos of the weighted, selected column to the last bit
  (worst relative difference 0.0; `analysis/selftest/`). The brute-force sum shares no ring/disc code with
  the painter.
- Radius caches rebuilt and re-validated for this run (`analysis/*_cache_check.csv`,
  `*_profile_owned_chord_check.csv`): worst 0.028 % (Battaglia16) and 0.042 % (Lee22) against the direct
  integrals, the same as on 17 September.
- Map means against the 100k individual sightlines of 17 September (same density models, same kernels
  as strata; `analysis/map_mean_vs_100k_rays.csv`): the six map means are 0.7-1.4 standard errors below the
  weighted ray means (24.19 vs 24.54 +- 0.25 pc cm^-3 for Battaglia16/Planck; 58.50 vs 59.09 +- 0.70 for
  Lee22/Planck). The six differences share the same 100k rays, so they are one correlated fluctuation, not
  six independent ones.
- Spectra: `map2alm` with lmax 8192 and 3 Jacobi iterations for every map; every y x DM cross-spectrum
  passed the Cauchy-Schwarz bound against the two auto-spectra. Pixel windows are not deconvolved
  (the maps are profile values at pixel centres, as in the earlier FRB power-spectrum products).

## 3. Full-sky prediction against Takahashi (figure set 1)

`plots/fullsky_takahashi_fig13` (values: `analysis/fullsky_takahashi_annuli.csv`). The smooth curves are
w(theta) = sum_l (2l+1)/(4 pi) C_l^{yDM} B_l P_l(cos theta) with the survey beam on y only; the markers are the
exact solid-angle means over the paper's annuli. Units 1e-5 pc cm^-3; the observed values are the digitized
Figure 13 points with symmetrized errors; annuli below the paper's angular cut (10' Planck, 1.8' ACT) are shaded.

| annulus | observed | Battaglia12 y x Battaglia16 DM | Lee22 y x Lee22 DM | Lee22 x Lee22, calibrated range |
|---|---|---|---|---|
| Planck 10-17.8' | 4.28 +- 1.61 | 1.20 (-1.9 sigma) | 7.85 (+2.2 sigma) | 3.51 (-0.5 sigma) |
| Planck 17.8-31.6' | 2.48 +- 1.30 | 0.65 (-1.4) | 4.19 (+1.3) | 1.82 (-0.5) |
| Planck 31.6-56.2' | 1.08 +- 0.87 | 0.29 (-0.9) | 1.59 (+0.6) | 0.76 (-0.4) |
| Planck 56.2-100' | 1.21 +- 0.36 | 0.11 (-3.0) | 0.41 (-2.2) | 0.25 (-2.7) |
| Planck 100-178' | 0.67 +- 0.35 | 0.04 (-1.8) | 0.14 (-1.5) | 0.09 (-1.7) |
| ACT 1.8-3.2' | 3.79 +- 4.58 | 4.00 (+0.0) | 19.90 (+3.5) | 10.92 (+1.6) |
| ACT 3.2-5.6' | 2.21 +- 1.98 | 2.96 (+0.4) | 16.48 (+7.2) | 8.50 (+3.2) |
| ACT 5.6-10' | 2.65 +- 2.93 | 1.96 (-0.2) | 12.15 (+3.2) | 5.77 (+1.1) |
| ACT 10-17.8' | 5.37 +- 3.07 | 1.18 (-1.4) | 7.80 (+0.8) | 3.43 (-0.6) |
| ACT 17.8-31.6' | 2.57 +- 2.12 | 0.64 (-0.9) | 4.11 (+0.7) | 1.78 (-0.4) |
| ACT 31.6-56.2' | 0.86 +- 1.36 | 0.29 (-0.4) | 1.56 (+0.5) | 0.76 (-0.1) |
| ACT 56.2-100' | 2.58 +- 1.49 | 0.11 (-1.7) | 0.40 (-1.5) | 0.25 (-1.6) |

- Battaglia12 x Battaglia16 sits at 0.28 of the Planck 10-17.8' point (-1.9 sigma) and 0.26 at 17.8-31.6', but
  within 0.4 sigma of the three ACT points at 1.8-10'. Compared with the 13 September full-sky product
  (Battaglia12 y x Battaglia16 DM inside the 3 R200c sphere: 1.70 at Planck 10-17.8', 4.88 at ACT 1.8-3.2') the
  R200c boundary lowers the prediction by 30 % at 10-18' and 18 % at 1.8-3.2'.
- Lee22 x Lee22 with all halos overshoots: +2.2 sigma at Planck 10-17.8' and +3.2 to +7.2 sigma at ACT 1.8-10',
  where it is 5-7 times the observed values; it crosses the data near 15-40'.
- Restricting both Lee22 maps to the calibrated ranges removes 55 % of the small-angle signal (0.45 at l ~ 100-300
  in the cross-spectrum, Section 5) although it removes only 13 % of the mean DM and 13 % of the mean y: the
  excluded halos are the clusters above 9.28e14 Msun, which dominate y and, through the Lee22 (1+z)^-2.11
  normalization, the low-z halo DM. The calibrated-range prediction is within 0.5 sigma of the three Planck points
  at 10-56' and within 0.6 sigma of ACT at 10-56', but still 1.1-3.2 sigma above ACT at 1.8-10'.
- Beyond 56' every model falls 1.5-3 sigma below the data (observed 1.21 +- 0.36 at 56-100' on Planck against
  0.1-0.4 for the models). The maps contain the two-halo clustering of the R200c gas, so this is gas outside R200c
  or non-halo contributions that the halo-only prediction does not have, or the paper's large-angle systematics.
- The full-sky values agree with the 100k-sightline estimates of 17 September where both exist (Battaglia16
  with Battaglia12 y at Planck 10-17.8': 1.20 here, 1.28 +- 0.3 from the sightlines), as they must: the map
  estimate is the sightline estimator's ensemble mean.

## 4. Realizations with the observed number of FRBs (figure set 2)

`plots/realizations_selected_battaglia` (Battaglia12 y x Battaglia16 DM) and `plots/realizations_selected_lee22`
(Lee22 y x Lee22 DM with all halos; both maps restricted to the calibrated ranges). Method
(`cross_finite_source_realizations.py analyze-pairs`): from the 100k individual sightlines of 17 September, 1000
disjoint realizations per survey with exactly 71 (Planck) or 31 (ACT) sources, one random ray per observed
redshift; each realization's cross-correlation is the stratified estimator with the stratum means taken from all
100k rays (the analogue of the DM-z relation and of the random-position y mean). The same 1000 ray sets are used
for every pair, so the pairs differ only through the gas models. Shown: the mean of the 1000, the 100k ensemble
mean (dotted), the two realizations with the highest peak above the paper's angular cut, and the eight remaining
realizations with the lowest diagonal chi^2 against the digitized points above the cut (8 Planck annuli, 11 ACT
annuli; symmetrized digitized errors; a ranking, not a goodness-of-fit test). The y axis is linear up to 10 and
logarithmic above (1e-5 pc cm^-3); `analysis/finite_source_pairs_selected.csv` lists every plotted curve.

First analysis annulus (Planck 10-17.8', ACT 1.8-3.2'), units 1e-5 pc cm^-3:

| pair | survey | 100k mean | mean of 1000 | median | 16-84 % | highest realization | share of the mean from the highest | chi^2 of the mean / best realization | realizations with chi^2 below the mean |
|---|---|---|---|---|---|---|---|---|---|
| Battaglia | Planck | 1.28 | 1.29 | 0.37 | 0.06-1.39 | 162 | 13 % | 28.4 / 7.3 | 18 % |
| Battaglia | ACT | 4.27 | 4.71 | 1.01 | 0.25-4.12 | 667 | 14 % | 7.9 / 3.3 | 7 % |
| Lee22, all halos | Planck | 9.12 | 9.79 | 1.12 | 0.26-5.73 | 3576 | 37 % | 31.6 / 2.0 | 41 % |
| Lee22, all halos | ACT | 23.3 | 34.3 | 2.54 | 0.82-12.6 | 16004 | 47 % | 334 / 2.8 | 92 % |
| Lee22, calibrated range | Planck | 3.73 | 3.62 | 1.01 | 0.23-4.49 | 344 | 10 % | 19.8 / 3.2 | 10 % |
| Lee22, calibrated range | ACT | 12.4 | 12.2 | 2.34 | 0.72-11.4 | 694 | 6 % | 23.9 / 3.1 | 83 % |

- The two highest realizations of every pair are sets that contain one sightline through the core of a massive
  cluster: their small-angle values are 40-700 times the observed ones for Battaglia and 1000-4000 times for Lee22
  with all halos (0.16 pc cm^-3 at ACT 1.8-3.2'); the second-highest is 2-8 times lower than the highest. With
  the calibrated ranges the Lee22 outliers fall to the Battaglia level (344 and 694), because the clusters above
  9.28e14 Msun, where both Lee22 fits are extrapolated, are exactly the halos that produce them.
- The median realization is 0.1-0.3 of the ensemble mean and only 10-19 % of the realizations exceed the mean:
  a 71- or 31-source measurement on this sky usually sits well below the halo-model mean and occasionally far above
  it. The mean of the 1000 realizations reproduces the 100k mean to 1-10 % except for Lee22 with all halos on ACT,
  where the single highest realization contributes 47 % of the mean (the 1000 sets use 31,000 of the 100k rays,
  so one extreme ray weighs three times more than in the 100k estimate).
- Realizations that fit the data well exist for every pair (chi^2 = 2-3.3 for the best against 7.9-334 for the
  mean). They are sets without a cluster sightline whose small-angle values are 2-5 (1e-5 pc cm^-3): for Battaglia
  and for the calibrated-range Lee22 these are ordinary realizations near the 84th percentile; for Lee22 with all
  halos on ACT 92 % of the realizations fit better than the mean, i.e. the mean is dominated by the extrapolated
  clusters while a typical 31-source sample does not contain one.
- Battaglia's ensemble mean is itself a fair description of the ACT points (chi^2 7.9 for 11 annuli) and misses
  Planck mainly through the 56-100' annulus (-3 sigma) and the 10-17.8' annulus (-1.9 sigma), where the data are
  above every halo-only curve.

## 5. Power spectra (figure set 3)

`plots/fullsky_power_spectra_three_column` (binned values: `analysis/fullsky_binned_spectra.csv`; ratios:
`analysis/fullsky_summary.json`). D_l = l(l+1) C_l / 2 pi, 12 logarithmic bins per decade, l = 30-8192, monopoles
removed, no beam, pixel window not deconvolved; the DM spectra are shown for the Planck kernel (solid) and the ACT
kernel (dotted). Shaded l > N_side.

| ratio | l ~ 100 | l ~ 300 | l ~ 1000 | l ~ 3000 |
|---|---|---|---|---|
| C^yy Lee22 / Battaglia12 | 2.40 | 2.70 | 1.57 | 0.84 |
| C^yy Lee22 calibrated range / Lee22 all halos | 0.23 | 0.23 | 0.37 | 0.55 |
| C^DD Lee22 / Battaglia16 (Planck kernel) | 16.7 | 14.2 | 8.6 | 3.6 |
| C^DD Lee22 calibrated / all (Planck kernel) | 0.73 | 0.77 | 0.90 | 0.93 |
| C^yD Lee22 x Lee22 / Battaglia12 x Battaglia16 (Planck kernel) | 6.4 | 6.4 | 3.9 | 2.0 |
| C^yD Lee22 calibrated / all (Planck kernel) | 0.45 | 0.47 | 0.67 | 0.82 |
| C^yD ACT kernel / Planck kernel (Battaglia pair) | 1.01 | 1.04 | 1.10 | 1.15 |

- The Lee22 pressure map has 2.4-2.7 times the Battaglia12 y power at l = 100-300 and less above l ~ 2500, the
  redistribution towards massive low-z clusters already seen on 17 September (P0 proportional to M^1.08 above M_cut).
  77 % of that low-l y power comes from halos outside the calibrated ranges, i.e. from the clusters above
  9.28e14 Msun where the fit is extrapolated.
- The Lee22 DM auto-spectrum is 14-17 times Battaglia16 at l = 100-300 (3.7-4.1 in amplitude), while the mean DM
  is only 2.4 times higher: the extra power is the mass and redshift dependence of the Lee22 normalization,
  which puts far more of the DM into rare massive halos (Section 5c of the 17 September report). Removing the
  halos outside the calibrated ranges lowers it by only 23-27 %.
- In the cross-spectrum the two effects multiply: Lee22 x Lee22 is 6.4 times Battaglia12 x Battaglia16 at l <= 300
  and 2.0 times at l ~ 3000; the calibrated-range pair is 0.45-0.47 of the all-halo pair at l <= 300, so the
  Lee22 x Lee22 ratio to Battaglia drops to about 3 there.
- The ACT kernel (31 redshifts, mean z 0.32) gives 1-4 % more DM power and cross power than the Planck kernel
  (71, mean z 0.27) at l <= 300 and 10-15 % more at l >= 1000: the higher-redshift sources see more, smaller halos.

## 6. Limits

- Halo-only partial prediction: no diffuse IGM or host-galaxy DM, no survey mask, noise or
  component-separation residuals; the y maps keep XGPaint's projected 4 R200c footprint while the DM
  uses the R200c sphere (the like-for-like TNG implementation).
- Equal source weights approximate the paper's weights; the digitized error bars are correlated between
  annuli, so the chi^2 values only rank realizations, they are not a goodness-of-fit test.
- The Lee22 fits are extrapolated above 9.28e14 Msun and below 1.47e13 Msun in the all-halo maps; the
  calibrated-range maps drop those halos entirely instead of modelling them differently.

## 7. Files

- `frb_map_generation/paint_halfdome_kernel_weighted_dm_maps.jl` - kernel-weighted full-sky DM painter
  (updated implementations, several models and kernels per pass, brute-force self-test)
- `frb_map_generation/paint_halfdome_battaglia12_tsz_map.jl` - new `--minimum-halo-mass-msun`,
  `--maximum-halo-mass-msun` options (calibrated-range y map)
- `frb_map_generation/fullsky_tsz_dm_comparison.py` - stages `spectra`, `check`, `plot`
- `frb_map_generation/cross_finite_source_realizations.py` - `sample-y --ykey`, `analyze-pairs`,
  `plot-selected`
- `frb_map_generation/run_fullsky_tsz_dm_20260918.sh` - the local production chain
- `frb_map_generation/run_fullsky_tsz_dm_cluster.pbs` - the idark chain (stages y, dm, post); products under
  `/lustre/work/kristero10/frb_data/fullsky_20260918`, small products copied to `outputs/tsz_dm_fullsky_20260918/cluster_results/`
- `outputs/tsz_dm_fullsky_20260918/{maps,spectra,analysis,plots,logs,kernels}` (maps and spectra are
  not git-tracked)

## 8. Addendum 18 September (afternoon): beams made explicit, percentage panels, distinguishable realizations, idark run

Requested changes and where they are:

- Every tSZ-side quantity carries the survey's Gaussian beam: 10' FWHM for Planck, 1.6' for ACT, applied to y only
  (FRB positions are not smeared). The Takahashi-type figures always had these beams; their titles now say so.
  The three-column spectra figure now exists in a beamed version, `fullsky_power_spectra_three_column_beamed`:
  C_l^yy B_l^2 and C_l^{yDM} B_l with the Planck beam on the Planck-kernel curves and the ACT beam on the ACT-kernel
  curves (the DM auto-spectrum has no beam). The unbeamed figure is kept.
- Percentage-difference panels (-100 to +100 %) under every panel: `fullsky_takahashi_fig13_residuals` and
  `realizations_selected_{battaglia,lee22}_residuals` show (model - observed) / |observed| per annulus with the
  observed 1 sigma as a grey band; a value beyond +-100 % is drawn as an open triangle at the panel edge (with the
  number for the model curves), so nothing is clipped silently. The spectra figure shows each pair relative to the
  Lee22 x Lee22 all-halo pair (the largest, so every curve stays inside the range) and, in grey, the beam
  suppression alone (B_l^2 - 1 and B_l - 1): a percentage panel relative to Battaglia would leave the range
  (Lee22 x Lee22 is up to 6.4 times Battaglia12 x Battaglia16).
- The ten shown realizations each have their own colour and marker (`BEST_STYLE`, `OUTLIER_STYLE`), the two
  highest as crimson/dark-red lines with x/+ markers, the eight best fits ranked 1-8 in the legend.
- Reduced version requested afterwards (`realizations_best3_{battaglia,lee22}`, stage `plot-simple`): only the
  Takahashi points and the three best-fitting realizations, linear axis, percentage panel below, large labels;
  this is the version in the PDF book. The full ten-realization figures remain in `plots/`.
- Computations on idark (`run_fullsky_tsz_dm_cluster.pbs`, working root
  `/lustre/work/kristero10/frb_data/fullsky_20260918`, code snapshot = commit f0a1626): stage `y` repaints the two
  Lee22 pressure maps, stage `dm` paints the six kernel-weighted DM maps (with the brute-force self-test first) and
  the 10-model individual sightlines, stage `post` (dependent on both) samples the annular y at the 100k positions,
  computes the spectra, the map-mean check, the pair realizations and all figures. The Battaglia12 y map is the
  13 September cluster repaint (`battaglia12_full_lightcone_repaint.fits`, sha256 9ea83f55...; the local map used
  above has sha256 1213ee84..., a different painting of the same model), the 100k positions and Battaglia12
  annular samples are the 14 September cluster products. The cluster XGPaint fork is the same commit (5dd0b57)
  with the same syntax patches in `profiles.jl`.

Cluster run (PBS jobs 598127 `y`, 598128 `dm`, 598129 `post`; 8 CPUs, 48 GB each; queue `mini`, started at once):

| stage | node | wall time | content |
|---|---|---|---|
| y | ansys19 | 32 min | Lee22 pressure self-test; all-halo map 1014 s; calibrated-range map 458 s |
| dm | ansys20 | 69 min | two caches; brute-force self-test (500 pixels x 6 maps, 2168 non-zero entries, worst relative difference 0.0); six maps 936 s; ten sightline caches + 100k x 3 planes x 10 models scan 141 s |
| post | ansys19 | 47 min | 72 annular y syntheses at the 100k positions (two Lee22 maps); nine `map2alm`; check; pair realizations; all figures |

Cluster products against the local run of the morning (`analysis/*` in
`outputs/tsz_dm_fullsky_20260918/cluster_results/`): the six map means are identical to 1e-13, the annulus values
agree to 1.2e-10 relative, the 1000 realizations select the same outlier and best-fit indices with identical chi^2,
and the binned spectrum ratios agree to four digits. The Julia 1.6 / 1.12 and the two independent Battaglia12
paintings therefore make no visible difference; the cluster figures are the ones delivered
(`cluster_results/.../plots/*_residuals.*`, `*_beamed.*`).

Reading the new panels:

- Full-sky percentages (Planck 10-17.8', 17.8-31.6', 31.6-56.2'): Battaglia12 x Battaglia16 -72, -74, -73 %;
  Lee22 x Lee22 +83, +69, +47 %; calibrated range -18, -27, -30 %. On ACT (1.8-3.2', 3.2-5.6', 5.6-10'):
  Battaglia +5, +34, -26 %; Lee22 +424, +646, +358 % (edge triangles); calibrated +188, +285, +118 %. Beyond 56'
  every model is 65-95 % below the data on both surveys.
- Realizations: the two highest are 100-1800 % above the data (Battaglia) and up to 1.6e4 % (Lee22 all halos) in
  every annulus of the paper's range; the eight best fits scatter within about +-60 % of the observed values, i.e.
  inside the observed 1 sigma band, with no systematic sign, while the mean of 1000 lies at -70 % (Battaglia, Planck
  beyond 10'), +80 to -90 % (Lee22 all halos, from small to large angles) and -10 to -90 % (Lee22 calibrated).
- Beamed spectra: the 10' Planck beam removes half of the tSZ auto power at l ~ 700 and half of the cross power at
  l ~ 1000; the 1.6' ACT beam does the same at l ~ 4400 and ~ 6300, i.e. only inside the pixel-limited range. The
  model-to-model ratios are unchanged by the beams, so the percentage panels equal those of the unbeamed figure:
  Battaglia12 x Battaglia16 sits 80-85 % below Lee22 x Lee22 in the cross power at l < 1000, the calibrated pair
  45-55 % below.
