## Radial grid, radius and normalisation scaling, and radial range (group `grids_scaling`)

**Reference** (the same in every group, id `ref`):
- key `{field}|geometric|removed|all|all|fit|{default norm}`, i.e. the geometric estimator (for T_e and K the ratio-of-sums `weighted` estimator, the only one cached for those fields), recently AGN-heated gas removed, all gas, all phases;
- the 35-shell fit grid;
- inclusion-weighted mean;
- `strict` rule (min_count 20, softening x4, smoothing length x2, min_edge 0.04);
- scheme M200c_0.2, all haloes, n_boot = 200 with the core's per-bin seeds.

**How the variants were compared.**
- Every variant was brought to reference units on the fit grid and compared through paired bootstrap replicates.
- Band ratios are the median over resolved shells. A "range" means min-max over the 10 calibrated bins 13.5-15.5, unless other bins are named.
- Observable impacts come from `sc.impact_from_stacks` on the calibrated bins. They are first order, using Test F's Jacobians. The truncation costs in §3 use `sc.observable_impact` with the profile removed beyond the cut.
- All numbers are in `results.json`; key paths are given in brackets. `report_tables.py` prints every table below from that file. A few statements are recomputed by the read-only scripts `section_checks.py`, `sliver_error_check.py`, `verify/v3_sigma_outward.py` and `verify/v4_sigma_profile.py`, named where used.

**Corrections to earlier versions of this section.** Three rounds of independent verification found the following.
- **4R growth rates** (first round). These were wrong by a factor of about 1.9. `sc.integral(rows, 'fit', 3.6342)` drops the 3.30-3.63 shell because the stored edge is 3.634241. All cuts now use exact stored edges.
- **R_fit = 1.34 on x200** (second round). A 0.011 edge tolerance had fitted x200 to its 1.3508 edge instead of 1.34, which added one 0.0035-dex shell. x200 is now fitted to exactly 1.34. Only the old grid, which has no 1.34 edge, is fitted to 1.3508. The 1.3508 fit is kept as a diagnostic row. Several signs and every R_fit = 1.34 x200 impact changed (§1).
- **Fit-sensitivity explanation** (second and third rounds). The per-shell sigma is not roughly equal across shells. It is roughly U-shaped in radius. The electron outward rise is 3.4-8.4, not 3.4-11.3: the 11.3 is max/min in 15.3-15.5, where the maximum is a core shell (0.047-0.056 R200c), not the outskirts. The inner-to-outer shell-count ratio rises by a similar fraction in every bin, so it does not explain the mass-dependent sign (§1).
- **Ranges, bins and significances** (first and second rounds). Several were over-generalised or misattributed: grid-fit significances, the sign of the N_e change, paint-bias ranges, the bin sets behind conversion-factor statistics and correlations, the reach of 6 R500c, the remap worst shells and the conversion precision.
- **Truncation costs** (first round). These lacked the calibrated-bins-only caveat and the second-order correction for the quadratic observables.
- **x500_halo method floor** (first round). The per-halo "method floor" quoted for electrons came from outer shells not covered by every halo (§2). This is now fixed.

### Bottom line

| Choice | Recommendation for painting HalfDome | Main evidence |
|---|---|---|
| Storage grid | Store per-halo **linear sums** (y, n_e V, m, n_e kT, V, counts) on one fine **nested** grid that contains every edge you will ever cut, fit or compare at. Never store ratios. | Rebinning to any sub-grid is exact to <= 1.6e-15, including every bootstrap replicate. Regridding to non-nested edges has a maximum per-shell error of 2.3% (PCHIP of the cumulative), 6.9% (power-law cumulative), 6.7% (log-log interpolation) and 12.5% (constant density). |
| Stacking / fitting grid | One fixed grid with no sliver shells and exact edges at every R_fit/R_cut: the 35-shell **fit** grid (0.037-0.079 dex shells), identical for all snapshots. | x200's thin shells (down to 0.001 dex) leave 1-5 strict-mask holes per bin. At R_fit = 4, switching the fit from the fit grid to x200 moves Y_model(<4R) by -0.24 to -1.41% and N_e,model(<4R) by +1.18 to -4.37% (the sign depends on mass), and kSZ 1h at ell = 100 by -1.92 ± 0.09%. At R_fit <= 2 the observables move by <= 0.35%. One 0.0035-dex boundary sliver (1.34-1.3508 R200c) alone moves Y_model(<4R) by up to 1.42% and N_e,model(<4R) by up to 1.11%. |
| Radius unit | **R200c**: HalfDome is an M200c catalogue and XGPaint paints r/R200c. | The cross-bin spread in R500c units equals the R200c one to within 0.022 in ln F. R500c with P_e,500 lowers the within-bin pressure scatter only inside about 0.4 R200c (0.220 vs 0.307 at 0.2 R200c; 0.382 vs 0.241 at 1 R200c). An R500c stack painted at M200c biases tSZ at ell = 3000 by -1.54 ± 0.54% and mean y by +1.57 ± 0.35%. |
| Normalisation | **P200 and n_e mu_XGP/(rho_c f_b)**, exactly as in the painter. | Per-halo conversions match the analytic factors to <= 1.4e-9 (2.2e-16 for pressure). With bin-mean conversion, the normalisation alone moves tSZ by <= 0.70%. |
| Radial range | Measure to >= 4 R200c (more if affordable), **fit to 4R and use a hard 4R cut**. | The budget is still growing at 4R: d ln Y/d ln r = 0.12-0.55 and d ln N_e/d ln r = 0.59-0.83 over 3.63-4 R200c. Cutting the calibrated bins at 2R costs tSZ ell = 100 -7.5% (first order; -7.4 / -7.2% at second order), kSZ 1h ell = 100 -39.6 / -38.7% (second order; -44.5% at first order) and mean DM -22.8%. A 2R cut of all engine bins costs mean DM -37.1%. |

### 1. Radial grid

**The choice.** There are two decisions:
- which shells to sum particles into (storage);
- which shells to stack and fit on.

All six cached grids derive from two storage grids, X200 and X500 [derived_summaries.grid_widths_dex]:

| Grid | Shells | Range | Notes |
|---|---|---|---|
| fit (ref) | 35 | 0-4 R200c | radii234 log shells (0.069 dex) to 0.84R, then exact edges at 1, 1.15, 1.34, 1.5, 1.75, 2, 2.18, 2.5, 3, 3.30, 3.63, 4. Shells are 0.037-0.079 dex wide (minimum d ln x = 0.086; 2-2.18R is the thinnest). |
| x200 | 67 | 0-4 R200c | Storage grid: the union of the radii234, Lee22 and fit edges. Slivers go down to 0.0010 dex (d ln x = 0.0024, at 0.3205-0.3212 R200c). |
| lee22 | 27 | 0.04-1.34 R200c | Lee et al. 2022 log shells. |
| old | 36 | 0-4 R200c | radii234 grid, with a sliver at 0.98-1 R200c. It has no 1.34, 1.5, 1.75 or 2.5 edge; its nearest edge to 1.34 is 1.3508. |
| x500 | 40 | 0-6 R500c | Storage grid, in R500c units. |
| b24 | 30 | 0.01-3 R500c | Braspenning+24 shells. |

![grids](groups/grids_scaling/figures/grid_layout.png)

**Rebinning consistency, verified** [rebinning].
- A geometric shell value is a sum divided by the shell volume, so a coarse shell is the volume-weighted mean of its fine shells.
- This holds per halo. Because the inclusion-weighted mean is linear, it also holds for the stack and for every paired bootstrap replicate.

| Check | Max deviation |
|---|---|
| P, E, G per halo: x200 -> fit / lee22 / old; x500 -> b24 | <= 8.9e-16 |
| T_e per halo, electron-weighted (n_e V) rebin | <= 7.8e-16 |
| K E^(2/3) / T_e per halo, constant on all grids | <= 6.7e-16 |
| Mean stack + 200 replicates, x200 rebinned vs fit (P, E, G) | <= 1.6e-15 |
| Stacked T_e: volume-weighted rebin of the x200 stack | 0.56% |
| Stacked T_e: rebin weighted by the stacked electrons | 0.13% |
| Medians after stacking, rebinned (calibrated bins) | P 0.7-2.8% (6.9% at 15.1-15.3), E 0.4-3.4% |

**Ratio fields and non-linear statistics.** T_e, K, the median and the percentiles do not rebin after stacking. Rebin the sums first, then form the ratios.

**Non-nested regridding** depends on what is assumed inside each source shell [rebinning.remap_accuracy]. The test is old -> fit and lee22 -> fit, where the true fit-grid stack is known. Errors are over the resolved shells of the calibrated bins:

| Remap method | max per-shell error, P (old / lee22) | max, E (old / lee22) | median |
|---|---|---|---|
| Monotone PCHIP of the cumulative in ln x (**used**) | 2.3% / 1.5% | 1.4% / 1.6% | <= 0.07% |
| Power-law cumulative (ln C linear in ln x) | 5.3% / 6.8% | 2.3% / 6.9% | <= 0.32% |
| Log-log interpolation of shell values (not conservative) | 5.7% / 6.7% | 3.0% / 5.0% | <= 0.60% |
| Constant density per source shell | 12.5% / 3.8% | 7.8% / 2.6% | <= 1.01% |

The worst shells are in the top bins:
- old -> fit: 1.5-2 R200c in 15.3-15.5 for PCHIP, constant density and power-law cumulative; 3.63-4 R200c in 15.3-15.5 for log-log interpolation (both fields).
- lee22 -> fit: 0.040-0.047 R200c in 15.3-15.5 for PCHIP and power-law cumulative; 1.15-1.34 R200c in 15.1-15.3 for constant density and log-log interpolation.

**Remapped band ratios** against ref [derived_summaries.band_ratio_ranges_calibrated_bins]:
- lee22: 0.998-1.004 for every field and bin.
- old: 0.997-1.003, including T_e and K.
- x200: exactly 1 for P, E and G. T_e and K are within 0.995-1.005, the error of remapping a stacked ratio.
- No grid changes an observable: the old and x200 impacts are 0.00%.
- lee22 stops at 1.34R. Its range beyond counts as missing: a 0.41 missing share for kSZ 1h at ell = 100 and 0.58 for mean DM.

**Resolution masks differ by grid** [resolution_masks]:
- The first strict-resolved edge is the same on fit and old. It runs from 0.321 R200c at 13.5-13.7 down to 0.040 at 15.1-15.5.
- x200 has 1-5 **holes** per calibrated bin. These are thin shells that fail the 20-particle rule in some halo, for example 0.3205-0.3212, 0.0764-0.0766 and 1.34-1.3508 R200c.
- On R500c grids, the "every halo resolved" condition is set by the least concentrated halo. Converted with the bin-mean R500c/R200c, the first x500 edge is:
  - 0.426 against 0.321 (13.5-13.7);
  - 0.198 against 0.145 (14.1-14.3);
  - 0.135 against 0.090 (14.5-14.7);
  - within 0.02 R200c of the fit-grid edge in the other seven bins.

![profiles on grids](groups/grids_scaling/figures/grid_profiles.png)

**Fit sensitivity to the grid.**
- **Fits.** B12 for pressure and B16 for electrons, with objective (a) of Tests B/F: diagonal, sigma = max(SD/mean, 5%). Strict shells, calibrated bins, 50 paired replicates.
- **Parity.** The fit-grid parameters reproduce Test F's to 4.7e-6.
- **Fit range.** R_fit = 1.34 is exact on the fit, x200 and lee22 grids. The old grid has no 1.34 edge and is fitted to its 1.3508 edge, so its R_fit = 1.34 row includes the 1.34-1.3508 shell. The diagnostic row "x200 to 1.3508" fits x200 to the same 1.3508 edge, to isolate that one sliver [grid_fits.cases.*_Rfit1.34].
- **What is compared.** Model-vs-model on the fit grid with a 4R cut [grid_fits.cases.*.vs_fit; derived_summaries.grid_fit_sensitivity].
  - Third column: the largest |d ln param| over parameters and bins, with the parameter it belongs to. In brackets is the largest significance of any parameter in any bin, which can belong to another parameter or bin.
  - Fourth column: the range over bins of d ln of the model integral to 4R. In brackets is the largest significance over bins.

| Field, R_fit | Grid | max abs d ln(param) (parameter; max sigma of any parameter) | d ln(model integral <4R), range over bins (max sigma) | Observable impact, % (paired SD) |
|---|---|---|---|---|
| P, 1.34 | x200 | 5.3% (xc; <= 3.2) | -1.05..+0.83% (2.8) | tSZ ell=100 +0.09±0.09, mean y -0.14±0.08 |
| P, 1.34 | x200 to 1.3508 (diagnostic) | 5.3% (xc; <= 2.2) | -1.05..+0.04% (2.5) | tSZ ell=100 -0.20±0.07, mean y -0.27±0.09 |
| P, 1.34 | old (to 1.3508) / lee22 | 0.8% / 1.1% (xc; <= 1.8 / 5.2) | -0.36..+0.08% (1.7) / -0.15..+0.20% (3.4) | tSZ ell=100 -0.05±0.05 / +0.02±0.04 |
| P, 2 | x200 / old | 1.9% / 1.4% (xc; <= 1.8 / 4.6) | -0.43..+0.32% (1.8) / -0.30..+0.32% (4.3) | tSZ ell=100 -0.01±0.07 / +0.11±0.06 |
| P, 4 | x200 | 7.5% (xc; <= 9.0) | -1.41..-0.24% (7.3) | tSZ ell=100 -0.71±0.08, ell=1000 -0.25±0.06, ell=3000 +0.00±0.05, mean y -0.71±0.04 |
| P, 4 | old | 1.1% (P0; <= 4.3) | -0.20..+0.18% (3.1) | tSZ ell=100 -0.08±0.04 |
| E, 1.34 | x200 | 4.8% (alpha; <= 5.7) | -0.33..+0.80% (5.6) | kSZ ell=100 +0.32±0.06, mean DM +0.25±0.04, DM var +0.10±0.04 |
| E, 1.34 | x200 to 1.3508 (diagnostic) | 4.8% (alpha; <= 6.8) | -0.69..+0.80% (6.2) | kSZ ell=100 -0.39±0.06, mean DM -0.06±0.04, DM var -0.22±0.04 |
| E, 1.34 | old (to 1.3508) / lee22 | 0.6% / 3.9% (alpha; <= 3.3 / 6.0) | -0.31..+0.25% (2.3) / -0.05..+0.29% (9.2) | kSZ ell=100 +0.04±0.06 / +0.18±0.02 |
| E, 2 | x200 / old | 2.4% / 2.0% (rho0; <= 7.6 / 12.8) | -0.64..+0.50% (5.9) / -0.89..+0.29% (11.7) | kSZ ell=100 +0.11±0.04 / +0.12±0.04 |
| E, 4 | x200 | 13.0% (rho0; <= 25.3) | -4.37..+1.18% (25.3) | kSZ ell=100 -1.92±0.09, ell=3000 +0.45±0.03, mean DM -0.14±0.03, DM var -0.28±0.04 |
| E, 4 | old | 6.0% (rho0; <= 24.8) | -0.60..+0.16% (17.4) | kSZ ell=100 -0.18±0.03 |

- **R_fit <= 2, all grids** (diagnostic row excluded). Model integrals change by <= 1.05% to 4R (worst case P, R_fit = 1.34, x200) and <= 0.46% to 2R. Observables change by <= 0.35% (kSZ 1h ell = 1000, E, R_fit = 1.34, x200) [derived_summaries.grid_fit_sensitivity_Rfit_le_2].
- **One boundary sliver** [derived_summaries.sliver_effect_x200_Rfit1.34]. Adding the single 1.34-1.3508 R200c shell (0.0035 dex) to the x200 fit at R_fit = 1.34 changes the fits as follows.
  - Y_model(<4R) moves by up to 1.42% (-1.42% at 15.1-15.3, 3.6σ). On its own this flips that bin from +0.83% to -0.59% against the fit grid.
  - N_e,model(<4R) moves by up to 1.11% (-1.11% at 13.7-13.9, 7.6σ).
  - The impacts shift by -0.28% (tSZ ell = 100), -0.71% (kSZ 1h ell = 100) and -0.31% (mean DM).
  - The sliver is not strict-resolved in 13.5-13.7, so nothing changes there.
- **R_fit = 4, x200.** N_e,model(<2R) changes by -2.09 to +0.78%.
- **Sign of the N_e change at R_fit = 4 (x200), per bin** [derived_summaries.grid_fit_sensitivity."E_Rfit4:x200".per_bin<4]. The change in N_e,model(<4R) is:
  - +1.03% (12σ) at 13.5-13.7 and +1.18% (8σ) at 13.7-13.9;
  - +0.09% (0.5σ) at 13.9-14.1;
  - -1.21% to -4.37% (5-25σ) from 14.1 upward.
- **Pressure at R_fit = 4 (x200).** Y_model(<4R) falls in every bin, by 0.24-1.41%.

![fit sensitivity](groups/grids_scaling/figures/grid_fit_sensitivity.png)

**Why.** Objective (a) is a diagonal chi^2 whose per-shell sigma is the halo-to-halo SD/mean, floored at 5%.
- **Sigma is far from uniform.** On the fit grid's strict shells to 4R it runs from 0.05 to 3.5 for P and from 0.05 to 0.57 for E [derived_summaries.objective_a_sigma_strict_shells_to_4R]. Within one bin it is roughly U-shaped in radius (`verify/v3_sigma_outward.py`, `verify/v4_sigma_profile.py`):
  - it is lowest in a shell between 0.38 and 1.15 R200c;
  - its largest value beyond that minimum is 8.6-20 (P) and 3.4-8.4 (E) times the minimum;
  - it is also raised in the core: 0.28-0.65 (P) and 0.30-0.56 (E) in the innermost strict shell.
  - The outer side is not monotone in every bin. In 14.7-15.1 its largest value is at 2.5-3 R200c and in 15.3-15.5 at 1.5-1.75 R200c, with a dip beyond. In the other bins it is in the last one or two shells (3.30-4 R200c).
  - In 15.3-15.5 the electron maximum (0.564, at 0.047-0.056 R200c) lies in the core, not the outskirts: the outer peak is 0.421 and the outermost shell 0.229. That bin's max/min of 11.3 is therefore not an outward rise.
- **Sigma does not shrink when a shell is split.** An x200 sub-shell has 0.69-1.49 (P) and 0.78-1.67 (E) times the sigma of its parent fit shell, with a median of 1.01 for both (`section_checks.py`). Each extra shell therefore adds about a full shell's weight, and the density of shells per ln r acts as an implicit weight on each radial range.

On x200, compared with the fit grid [derived_summaries.strict_shell_counts_fit_vs_x200]:
- there are 1.7-2.2 times as many strict-resolved shells inside R200c (7-20 become 12-44);
- there are about 1.5 times as many beyond R200c (11 become 16-17). The extra edges at 1.03, 1.18, 1.35, 1.58, 1.86 and 2.56 R200c split six fit shells, adding shells of 0.0035-0.026 dex at 1-2.6 R200c.

At R_fit <= 2 the extra weight moves the model integrals by <= 1.05% and the observables by <= 0.35%. At R_fit = 4 the 3-parameter shapes cannot follow both the core and the 2-4R tail, so the compromise moves:
- **Pressure.** Y_model(<4R) falls in every bin, by 0.24-1.41%.
- **Electrons.** The sign depends on mass (see above).
- **The inner-to-outer ratio does not explain the sign.** The ratio of strict shells inside to beyond R200c rises by a similar fraction in every bin, +18% to +42% [derived_summaries.inner_to_outer_relative_change_range]. Examples:
  - +18% at 13.5-13.7 (0.64 -> 0.75) and +29% at 13.7-13.9 (0.73 -> 0.94), where N_e,model(<4R) rises;
  - +35% at 13.9-14.1 (1.00 -> 1.35), where it changes by only +0.09%;
  - +29% at 14.1-14.3 (1.09 -> 1.41) and 14.5-14.7 (1.36 -> 1.76), where it falls.
- **Hypothesis, not tested.** The cause may be which shells are added. In 13.5-13.9 the resolved range starts only at 0.25-0.32 R200c, so the shells added at 1-2.6 R200c, where B16 misfits, weigh more in the fit than in bins resolved to <= 0.17 R200c. The sliver diagnostic above shows that a single added shell at 1.34 R200c can move N_e,model(<4R) by 1.1%.

The paired differences are far larger than the bootstrap noise, so this is a systematic effect of the grid, not noise. They reach 7.3σ in the pressure model integrals (9.0σ in the pressure parameters) and 25.3σ for electrons.

Resolved sliver shells (< 0.02 dex) do not carry systematically larger bootstrap errors of the stacked mean. Their fractional error is 0.99 (P) and 1.02 (E) times that of the parent fit shell in the median. It scatters more widely, though: 0.64-1.45 (P) and 0.48-1.57 (E), against 0.82-1.23 and 0.78-1.37 for the wider split shells (`sliver_error_check.py`). The red crosses in the profile figure mark x200 shells that fail the strict rule.

**Recommendation.**
- Store sums on a fine nested grid. This can be x200-like, or a log grid of about 0.02-0.03 dex augmented with every required edge. The slivers this creates are harmless for storage, as long as you never stack or fit on them. The 0.02-0.03 dex spacing is a suggestion; it was not tested here.
- Stack and fit on the fixed 35-shell fit grid.
- If the grid is ever changed, refit; do not reuse old parameters.

### 2. Radius scaling and normalisation

**Constants** [normalisation_checks]. Recomputing Y(<R200c), Y(<4R200c), Y(<R500c), N_e(<R200c) and N_e(<4R200c) from the normalised profiles times these scales reproduces the halo-table integrals to <= 5.4e-9. The cached x500 arrays match the analytic per-halo factors to <= 1.4e-9.

| Scale | Definition | Notes |
|---|---|---|
| P200 | G M200c (200 rho_c) f_b / (2 R200c); f_b = 0.158823 | Thermal (total-gas) scale. For ionised primordial gas P_e = (mu/mu_e) P_th, with mu/mu_e = 2(1+X)/(3+5X) = 0.5176 for X = 0.76 (0.59/1.14 = 0.5175 with rounded values): the prefactor of the coded B12 shape. rho_c comes from each halo's own M200c and R200c, and its median differs from 2.775e11 h^2 by 2e-5. |
| B24 P_e,500 | n_e,500 kT500, with n_e,500 = 500 f_B rho_c/(mu_e m_p), kT500 = mu m_p G M500c/(2 R500c), f_B = 0.159 | Electron pressure. P_e,500/P200 has median 1.395 (5-95%: 1.193-1.504). It is proportional to q^2 = (R500c/R200c)^2 with constant (500/200)^2 (f_B/f_b)(mu/mu_e) = 3.2382, to within 1.06%. The deviation is rho_c jitter from the lossy-stored M500c/R500c. The within-bin correlation of ln(P_e,500/P200) with ln q is 0.994-0.999 [derived_summaries.Pe500_over_P200_vs_q2]. |
| self_similar_500 | G M500c (500 rho_c) f_b/(2 R500c) | Equals 1.930 x P_e,500(B24) = (f_b/f_B)(mu_e/mu). |
| A10 P500 | 1.65e-3 E(z)^(8/3) [M500/(3e14 h70^-1 Msun)]^(2/3) h70^2 keV cm^-3, with h70 = 0.9729 and E(0) = 1 | A10/B24 has median 1.101, which equals 0.175/0.159 = 1.1006. My interpretation: this is consistent with a WMAP-era f_b = 0.175. The per-halo range 1.0958-1.1057 is not physics: B24 uses rho_c from each halo's lossy-stored M500c/R500c (rho_c spread up to 0.35%), while A10 uses M500 only. A10's extra (M500/3e14)^0.12 is not in the cached norm; it is derived here as variant `x500_A10M`. |
| Electrons | E = n_e mu_XGP/(rho_c f_b), with mu_XGP = (m_e + 2 m_p/1.76)/0.9; B24 uses n_e/n_e,500 | The ratio is a constant, 554.7 (within 0.54%). For electrons only the radius unit can change the shape. |

**Self-similarity across mass.**
- **Metric.** SD over bins of ln F at fixed radius [self_similarity.*.common_bins_spread_sd_lnF].
- **Bins.** The bins of 13.5-15.1 that are resolved in every scaling of the field at that radius, with at least 4 required. That is 5 bins at 0.2R, 6 at 0.3R and 8 from 0.5R outward.
  - At 0.2R: 13.9-14.1 and 14.3-15.1. The 13.5-13.9 bins are unresolved on the fit grid, and 14.1-14.3 is unresolved in the R500c scalings.
  - At 0.3R: 13.9-15.1. 13.5-13.7 is unresolved on the fit grid, and 13.7-13.9 is unresolved in the R500c scalings.
- **Radii.** Radii are R200c-equivalent: R500c profiles are read at r/R500c = x/0.656.
- **Figure.** selfsim_metrics shows the same bin set; its open markers are the tabulated values.

| x (R200c-equiv.) | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| bins used | 5 | 6 | 8 | 8 | 8 | 8 | 8 | 8 |
| P: R200c, P200 (ref) | 0.167 | 0.104 | 0.138 | 0.064 | 0.128 | 0.258 | 0.314 | 0.379 |
| P: R200c, P_e,500 | 0.167 | 0.103 | 0.143 | 0.067 | 0.124 | 0.255 | 0.312 | 0.376 |
| P: R500c, P200 | 0.168 | 0.106 | 0.146 | 0.069 | 0.111 | 0.239 | 0.300 | 0.383 |
| P: R500c, P_e,500 (B24) | 0.169 | 0.106 | 0.152 | 0.073 | 0.107 | 0.236 | 0.299 | 0.382 |
| P: R500c, A10 incl. M^0.12 | 0.077 | 0.038 | 0.097 | 0.137 | 0.234 | 0.368 | 0.432 | 0.514 |
| E: R200c / R500c | 0.302 / 0.304 | 0.227 / 0.230 | 0.284 / 0.292 | 0.159 / 0.170 | 0.054 / 0.059 | 0.142 / 0.129 | 0.168 / 0.158 | 0.128 / 0.128 |
| Stack noise (median bootstrap fractional error, P ref) | 0.015 | 0.012 | 0.007 | 0.008 | 0.013 | 0.025 | 0.036 | 0.062 |

- **R200c against R500c.** The ref spread and the R500c/P_e,500 spread differ by at most 0.022 in ln F: 0.258 vs 0.236 at 1.5R and 0.128 vs 0.107 at 1R [derived_summaries.self_similarity_P].
- **Mass slope, ref.** The pressure slope d ln F/d log10 M200c over 13.5-15.1 is +0.41 (0.2R), +0.22 (0.5R), 0.00 (0.7R), -0.25 (1R), -0.53 (1.5R) and -0.76 (3R).
- **Mass slope, R500c/P_e,500.** At the same radii it is +0.42, +0.24, +0.03, -0.20, -0.48 and -0.77.
- **Neither scaling removes the mass trend.** At the tabulated radii the spread is 6.1-18.9 times the stack noise for ref and 4.8-19.5 times for R500c/P_e,500 (`section_checks.py`).
- **A10 including M^0.12.** It flattens the core (spread 0.04-0.10 at 0.2-0.5R). It roughly doubles the spread at 0.7-1 R200c (1.9-2.2 x the B24 value) and raises it 1.3-1.6 times beyond R200c.

**Halo-to-halo scatter within a bin** (median SD ln F over the same bins) [self_similarity.*.common_bins_within_bin_sd_ln_median]:

| x | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| P: R200c, P200 | 0.307 | 0.225 | 0.131 | 0.146 | 0.241 | 0.389 | 0.526 | 0.735 |
| P: R200c, P_e,500 | 0.248 | 0.165 | 0.117 | 0.183 | 0.287 | 0.414 | 0.533 | 0.747 |
| P: R500c, P200 | 0.273 | 0.182 | 0.117 | 0.189 | 0.320 | 0.441 | 0.550 | 0.745 |
| P: R500c, P_e,500 | 0.220 | 0.135 | 0.143 | 0.246 | 0.382 | 0.482 | 0.583 | 0.767 |
| E: R200c / R500c | 0.250 / 0.230 | 0.177 / 0.149 | 0.102 / 0.103 | 0.092 / 0.150 | 0.148 / 0.205 | 0.225 / 0.266 | 0.284 / 0.308 | 0.359 / 0.373 |

Median particle counts per shell are similar on both grids: 606 (fit) and 717 (x500) at 1R in 13.5-13.7 [counts_context_median_per_shell]. Shot noise therefore does not drive the difference.

**Why.** R500c/R200c and P_e,500/P200 encode concentration [derived_summaries.conversion_factors].
- **Scatter of the conversion factors.** Within a bin in 13.5-15.1, SD ln(R500c/R200c) = 0.034-0.043 and SD ln(P_e,500/P200) = 0.067-0.086. Over all 10 calibrated bins the ranges are 0.021-0.043 and 0.040-0.086.
- **Correlation with the core.** ln(R500c/R200c) correlates with ln P in the 0.20-0.23 R200c shell.
  - Over the bins where that shell is strict-resolved, rho = +0.66 to +0.84 (13.9-15.1). Over all eight resolved calibrated bins (13.9-15.5) it is +0.64 to +0.99; the 15.1-15.5 bins have N = 21 and 6 [derived_summaries.conversion_factors.corr_lnq_lnP_0.20-0.23_resolved_bins_only].
  - The shell is unresolved in 13.5-13.7 and 13.7-13.9 (first strict edges 0.321 and 0.274 R200c). The values there, +0.52 and +0.59, include unresolved shells.
- **Correlation in the outskirts.** At 1.34-1.5 R200c the correlation is -0.35 to -0.19 (13.5-15.1), or -0.41 to -0.05 over all 10 bins.
- **Physical picture.** Concentrated haloes have more core pressure and less outer pressure in R200c units. Scaling by R500c absorbs this in the core but adds it outside.
  - With P_e,500 the within-bin pressure scatter falls at 0.2-0.3 R200c and rises from 0.5 R200c outward, so the crossover is at about 0.4 R200c.
  - With P200 it is lower up to 0.5 R200c and higher from 0.7 R200c.
  - For electrons it is lower at 0.2-0.3 R200c, equal at 0.5 R200c (0.103 in R500c units, 0.102 in R200c units) and higher from 0.7 R200c.
- **Where tSZ lives.** In Test F's importance for the calibrated bins, 63-64% of the tSZ signal at ell = 100, 1000 and 3000 lies at 0.23-1 R200c. There the R200c scatter has its minimum (0.13-0.15 at 0.5-0.7 R200c) and, from 0.5 R200c outward, is below the R500c/P_e,500 scatter. A further 18-32% lies inside 0.23 R200c. There R500c scaling has the lower scatter at 0.2 R200c (0.220 vs 0.307); smaller radii are resolved in too few bins to compare (`section_checks.py`).

![self-similar profiles](groups/grids_scaling/figures/selfsim_profiles.png)
![self-similarity metrics](groups/grids_scaling/figures/selfsim_metrics.png)

**Painting an R500c-scaled stack at M200c** (`x500_paint`).
- **Method.** The x500 B24 stack is converted with the bin's mean R500c/R200c and mean P_e,500/P200. This is what an M200c-only catalogue would have to do. The result is PCHIP-remapped to the fit grid.
  - Shells beyond 6 R500c at the bin-mean ratio, 3.87-3.90 R200c, are empty. The last fit shell, 3.63-4 R200c, is therefore missing in every calibrated bin and counts as no change. Its missing share is 0.021 for mean y, 0.037 for kSZ 1h at ell = 100 and 0.069 for mean DM [derived_summaries.fit_grid_coverage_by_6R500c].
  - B24, A10 (without M^0.12) and self-similar-500 give identical results, since their normalisations differ by constants.
- **Check.** The per-halo conversion of the same X500 sums (`x500_halo`) reproduces ref to 0.996-1.010 per band and to <= 0.01% in every observable [variants.grids_scaling:x500_halo].
  - Fit shells beyond 6 R500c of any halo in the bin are left empty. The first empty shell is set by 6 x the bin's minimum R500c/R200c [derived_summaries.fit_grid_coverage_by_6R500c_summary]. It starts at 2.5 R200c in 14.1-14.3 and 14.5-14.7; at 3.0 R200c in 13.5-13.9, 14.3-14.5 and 14.7-15.1; at 3.30 R200c in 13.9-14.1; and at 3.63 R200c in 15.1-15.5.
  - The missing share is 0.06 for mean y, 0.12 for kSZ 1h at ell = 100 and 0.19 for mean DM.
  - The first version quoted a -0.51% (kSZ) and -0.32% (DM) electron "method floor". That came from averaging those uncovered shells over only the covered, more concentrated haloes.

Band ratios to ref, range over the calibrated bins [derived_summaries.band_ratio_ranges_calibrated_bins]:

| Variant | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 R200c |
|---|---|---|---|---|---|---|---|---|
| P, R500c+B24 painted | 0.955-0.993 | 0.953-0.993 | 0.979-0.993 | 1.004-1.037 | 1.022-1.096 | 1.003-1.120 | 0.997-1.133 | 0.998-1.055 |
| P, R500c+P200 (radius only) | 0.983-0.997 | 0.973-0.996 | 0.978-0.993 | 0.994-1.013 | 1.017-1.044 | 1.001-1.058 | 0.992-1.061 | 0.988-1.032 |
| P, R200c+P_e,500 (normalisation only) | 0.970-0.996 | 0.975-0.996 | 0.993-0.998 | 1.003-1.015 | 1.003-1.025 | 0.999-1.026 | 0.999-1.020 | 0.998-1.030 |
| E, R500c painted | 0.987-1.001 | 0.986-0.999 | 0.987-0.996 | 0.997-1.010 | 1.005-1.022 | 1.004-1.028 | 0.995-1.025 | 0.993-1.013 |

In band medians, the painted pressure is therefore:
- 0.7-4.7% low inside 0.25 R200c;
- -0.3% to +13.3% at 1-3 R200c, and +2.2 to +9.6% in the 1-1.5 R200c band alone.

Observable impacts (%, paired SD) [variants.*.impact]:

| Variant | tSZ ell=100 | tSZ ell=1000 | tSZ ell=3000 | mean y | kSZ ell=100 | kSZ ell=3000 | mean DM | DM var |
|---|---|---|---|---|---|---|---|---|
| R500c+B24 painted | +1.43±1.50 | +0.04±0.77 | -1.54±0.54 | +1.57±0.35 | +0.91±0.33 | -0.27±0.19 | +0.34±0.12 | -0.01±0.21 |
| R500c, A10 incl. M^0.12 | +1.98±1.60 | +0.48±0.84 | -1.25±0.59 | +1.87±0.38 | - | - | - | - |
| R500c+P200 | -0.23±0.84 | -0.92±0.37 | -1.78±0.24 | +0.46±0.16 | +0.91±0.33 | -0.27±0.19 | +0.34±0.12 | -0.02±0.21 |
| R200c+P_e,500 | +0.29±0.65 | -0.23±0.36 | -0.70±0.27 | +0.48±0.13 | - | - | - | - |

**Remap-method dependence** [remap_method_sensitivity; derived_summaries.remap_method_sensitivity]. The raw paint impact depends on how the R500c shells are redistributed onto R200c shells. Removing each method's own error gives method-independent numbers. That error is measured by the per-halo check, where the truth is known. "Corrected" means F_paint x F_ref / F_halo, on the shells covered by every halo.

| Method | per-halo check: tSZ ell=3000 / mean y / kSZ ell=100 / mean DM | raw paint: same four | corrected paint: same four |
|---|---|---|---|
| PCHIP (used) | +0.01 / -0.00 / +0.00 / -0.00 | -1.54 / +1.57 / +0.91 / +0.34 | -1.54 / +1.43 / +0.78 / +0.26 |
| power-law cumulative | -0.46 / -0.03 / -0.08 / -0.00 | -1.99 / +1.55 / +0.86 / +0.35 | -1.52 / +1.42 / +0.78 / +0.26 |
| constant density | -1.48 / -0.12 / -0.50 / -0.07 | -3.08 / +1.46 / +0.43 / +0.29 | -1.62 / +1.38 / +0.69 / +0.23 |
| log-log points | +0.64 / +0.55 / +1.19 / +0.35 | -0.79 / +2.25 / +2.24 / +0.73 | -1.42 / +1.50 / +0.85 / +0.28 |

The paired SDs are about 0.54 (tSZ ell = 3000), 0.31-0.35 (y), 0.29-0.33 (kSZ) and 0.10-0.12 (DM).
- **Method-independent result.** After correction all four methods agree: tSZ ell = 3000 -1.42 to -1.62%, mean y +1.38 to +1.50%, kSZ +0.69 to +0.85%, mean DM +0.23 to +0.28%.
- **Missing share.** On these common shells it is 0.06 (y), 0.12 (kSZ) and 0.19 (DM). The raw paint numbers have 0.02, 0.04 and 0.07 (the 3.63-4 R200c shell only).
- **PCHIP is validated.** Its own per-halo error is <= 0.01%, so the PCHIP numbers above stand.

The b24 grid adds a range limit at 3 R500c, 1.93-1.95 R200c at the bin-mean ratio. Beyond it shells count as missing: a 0.45 missing share for mean DM and 0.30 for kSZ at ell = 100.

![R500c stack painted at M200c](groups/grids_scaling/figures/r500_stack_painted_at_m200c.png)
![conversion factors](groups/grids_scaling/figures/conversion_factors.png)

**Recommendation.**
- Stack and paint in r/R200c with P200 and rho_c f_b, the same constants as XGPaint.
- Store M500c, R500c and P_e,500 per halo for literature comparisons; this is cheap.
- Do not paint R500c-scaled stacks onto an M200c catalogue. The bias in mean y is +1.57 ± 0.35% with PCHIP and +1.38 to +1.50% after removing each remap method's own error, so it is robust to the remap method.

### 3. Radial range and truncation

**Three different radii:**
1. **Where the stack is measured.** The extraction sphere is max(4 R200c, 6 R500c).
   - The x200 grid stops at 4 R200c.
   - The x500 grid reaches 6 R500c, which is 3.87-3.90 R200c on the bin average in the calibrated bins.
   - 6 R500c exceeds 4 R200c for only 13-19% of the haloes in each bin of 13.5-15.1 (inclusion-weighted). In 15.1-15.3 it is 5% and in 15.3-15.5 it is 3 of 6.
   - That is 15.9% of the 13.5-15.5 population and 51% of the 12.5-15.5 population, because low-mass haloes are more concentrated. The unweighted sample fraction is 26.5% of the 4,588 haloes [derived_summaries.fraction_6R500c_gt_4R200c].
   - No unbiased stack therefore exists beyond 4 R200c.
2. **Fit range.** The shells used to fit B12/B16 or the Test G models.
3. **Painting cut.** The radius at which XGPaint zeroes the profile.

Tests F and G set (2) = (3) = 4R.

**Cumulative budgets** [cumulative_fractions_fit_grid]. Values are % of the value inside 4 R200c that lies outside x (stacked mean, all gas, bootstrap SD):

| Bin | Y beyond 1 / 1.34 / 2 / 3 R200c | N_e beyond 1 / 1.34 / 2 / 3 R200c |
|---|---|---|
| 13.5-13.7 | 60.3±1.3 / 46.9±1.6 / 29.7±1.9 / 13.9±1.6 | 78.8±0.3 / 66.9±0.4 / 46.2±0.4 / 21.7±0.3 |
| 14.1-14.3 | 41.4±0.7 / 29.4±0.7 / 16.8±0.7 / 7.4±0.6 | 64.9±0.3 / 52.6±0.4 / 35.5±0.4 / 17.2±0.3 |
| 14.5-14.7 | 35.5±0.4 / 24.9±0.5 / 14.3±0.4 / 6.2±0.3 | 60.4±0.2 / 49.1±0.2 / 34.0±0.3 / 16.9±0.2 |
| 14.9-15.1 | 30.4±0.8 / 20.5±0.8 / 11.0±0.7 / 4.0±0.3 | 57.1±0.4 / 46.4±0.5 / 32.3±0.5 / 15.8±0.3 |
| 15.3-15.5 | 29.5±2.8 / 20.8±2.1 / 10.6±1.3 / 3.7±0.7 | 58.1±1.0 / 48.5±0.8 / 33.8±1.5 / 15.8±1.0 |

The per-halo spread is wide: at 14.5-14.7 the fraction of Y beyond R200c has q16/q50/q84 = 0.25/0.32/0.42.

**The budget does not converge at 4R.** Over the last fit-grid shell, 3.634-4 R200c (exact stored edges) [derived_summaries.growth_at_4R]:
- d ln Y/d ln r = 0.55±0.09 (13.5-13.7) falling to 0.13±0.01 (14.9-15.1); 0.12-0.55 over all calibrated bins;
- d ln N_e/d ln r = 0.83±0.01 falling to 0.61±0.02; 0.59-0.83 over all calibrated bins;
- that last shell alone holds 1.1-5.2% of Y(<4R) and 5.7-8.0% of N_e(<4R);
- the stacked electron density falls as r^-1.4 to r^-2.2 between the 3-3.30 and 3.63-4 shells.

The 4R cut is therefore a definition, not a convergence radius. The halo model's 2-halo term must be consistent with it. Test G found that a linear 2-halo term matches the 3-4R electrons to within ±10% at 10^13.9-14.9 (quoted).

In R500c units (M500c bins), Y(<R500c)/Y(<5R500c) = 0.297±0.006 at 13.5-13.7, rising to 0.539±0.011 at 14.9-15.1. A10's universal shape gives 0.557 (computed here). Groups have far more pressure outside R500c than A10 [cumulative_R500c_units_M500c_bins].

![cumulative](groups/grids_scaling/figures/cumulative_budgets.png)
![outside fraction](groups/grids_scaling/figures/outside_fraction_vs_mass.png)

**First-order cost of cutting the reference stack of the 10 calibrated bins** [truncation_first_order]. Uncalibrated haloes are held fixed, so these are costs of a partial cut.
- The calibrated bins carry 98.6% of the tSZ ell = 100 importance, 90.7% of kSZ 1h ell = 100, 70% of mean y, 58.4% of mean DM and 75.6% of DM var.
- The numbers are exact for mean y and mean DM, for this partial cut only.

| R_cut | tSZ ell=100 / 1000 / 3000 | mean y | kSZ 1h ell=100 / 1000 / 3000 | mean DM | DM var |
|---|---|---|---|---|---|
| 1 | -35.9 / -19.5 / -7.4 | -27.8 | -97.9 / -73.8 / -31.1 | -40.8 | -44.4 |
| 1.34 | -19.1 / -8.5 / -1.8 | -19.1 | -73.9 / -51.2 / -14.7 | -33.7 | -28.3 |
| 2 | -7.5 / -2.3 / +0.0 | -10.1 | -44.5 / -26.1 / -2.0 | -22.8 | -13.4 |
| 3 | -2.8 / -0.5 / +0.1 | -4.1 | -19.2 / -8.8 / +1.1 | -10.9 | -4.5 |

Two corrections to these numbers, at R_cut = 2:
- **Second-order estimates for the quadratic observables** (calibrated bins). Two limits are given: one coherent amplitude, (1 - removed)^2 - 1; and independent bins. The results are:
  - tSZ ell = 100: -7.4 / -7.2%;
  - kSZ 1h ell = 100: -39.6 / -38.7%;
  - kSZ 1h ell = 1000: -24.4 / -23.0%;
  - DM var: -12.9 / -12.7%.
  - The first-order -44.5% for kSZ therefore overstates the loss. For the 1-halo term the halo-level exact change lies closer to zero still, because the removed fraction varies between haloes (the mean of (1 - f)^2 exceeds (1 - mean f)^2).
- **Cutting all 16 Test F engine bins** (fallback profiles outside 13.5-15.5, first order):
  - mean DM -37.1% (exact for a linear observable);
  - mean y -14.1%;
  - tSZ ell = 100 -7.9%;
  - kSZ 1h ell = 100 -49.2% (second order -43.1 / -42.7%);
  - kSZ 1h ell = 3000 -13.4%;
  - DM var -16.9%.
  - A 2R cut on all painted HalfDome haloes therefore costs more than the calibrated-bin numbers suggest.

**Comparison with Test F.** Test F's exact engine (fitted models, objective (a)) agrees qualitatively. Its (R_fit, R_cut) = (4, 2) cell gives family RMS errors of 5.1 / 25.7 / 22.0% for tSZ / kSZ / DM, and the (4, 3) cell 2.8 / 10.1 / 9.5%. These are RMS values over the observables of each family and include model error. They are not directly comparable with single-ell values. For example, Test F's kSZ 25.7% should be set against the first-order -26.1% at ell = 1000 here, not against -44.5% at ell = 100.

![truncation](groups/grids_scaling/figures/truncation_first_order.png)

**Summary of Tests C, F and G** (quoted, not recomputed) [quoted_truncation_tests]:
- **Test C (edge shape).** At 4R a 0.05-dex logistic edge changes pressure C_ell by <= 0.21% (ell = 100-2e4). Electrons are about 5 times more edge-sensitive (DM-field C_ell -1.1% at ell = 1000). Wider windows or 1-2R cuts move C_ell by 5-20% through flux redistribution.
- **Test F (fit and cut radius).** The only Pareto-optimal (R_fit, R_cut) is (4, 4), chosen by all 8 CV folds. Family errors fall from 4.1 / 3.2 / 2.2 / 3.9% at (2, 4) to 2.2 / 1.1 / 1.3 / 1.8% at (4, 4).
- **Test G (per-field models).** Electrons need the 4R cut: a 3R cut costs 8-10% in kSZ/DM. For pressure, R_fit = 3 and 4 are equivalent, but R_cut must stay at 4R.

The budgets explain these results qualitatively:
- The edge shape barely matters because the outermost shell holds only 1.1-5.2% of Y(<4R) (5.7-8.0% of N_e).
- Cuts below 4R are costly because 29.5-60.3% of Y(<4R) and 56.5-78.8% of N_e(<4R) lie beyond R200c (calibrated bins, `section_checks.py`).

**Recommendation.**
- Measure to at least 4 R200c. For future snapshots, extend the extraction to about 6 R200c if the I/O budget allows, to test the 1-halo/2-halo split.
- Fit to 4R and use a hard 4R cut. Handle the 2-halo term consistently in the halo model.
- Never use the Lee22 (1.34R) or B24 (3 R500c, 1.93-1.95 R200c) ranges for DM or kSZ.

### 4. Clumping and SPH volumes

Band medians over resolved shells [clumping]:

| Band (R200c), bin 14.5-14.7 | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 |
|---|---|---|---|---|---|---|---|---|
| C_P = <P_e^2>_V/<P_e>_V^2 | 1.04 | 1.05 | 1.14 | 1.32 | 1.68 | 2.52 | 5.19 | 19.7 |
| C_rho (all phases) | 15 | 7.6 | 7.0 | 15 | 105 | 689 | 2881 | 10225 |
| f_V = sum(m/rho)/V_shell | 0.964 | 0.974 | 0.973 | 0.963 | 0.944 | 0.919 | 0.884 | 0.804 |
| P_QV / P_geo (ratio of stacks) | 1.035 | 1.025 | 1.027 | 1.038 | 1.058 | 1.085 | 1.125 | 1.233 |

The pattern is the same at 13.5-13.7 (f_V from 0.956 down to 0.829, C_P from 1.12 up to 6.4) and at 14.9-15.1 (f_V down to 0.791, C_P up to 29). Over the calibrated bins, f_V is 0.956-0.975 inside 0.5R and 0.791-0.829 at 3-4R.

**Why SPH-volume weighting differs.**
- **The geometric estimator** divides linear particle sums by the true shell volume, so it is the volume average that tSZ and DM need. Clumps carry real y and electrons and are not a bias for it.
- **Q_V** instead divides by the sum of SPH volumes m_i/rho_i. In band medians over the calibrated bins, these tile only 95.6-97.5% of the shell inside 0.5 R200c and 79.1-82.9% at 3-4 R200c, so Q_V reads high by about 1/f_V.
  - The likely cause, an interpretation not tested here, is that kernel-smoothed densities of diffuse particles next to dense clumps are overestimated.
  - Q_V with the y-based particle pressure equals Q_V to 1e-5.
- **Mass weighting** is biased by clumping exactly: G_QM/G_geo = C_rho/f_V per halo, verified to 6.7e-16. Beyond R200c the band medians of C_rho are 91-10,225 over the calibrated bins, because star-forming and cold gas is in the 'all' phase.
- **C_P** measures how far X-ray- or mass-weighted pressures are biased.

![clumping](groups/grids_scaling/figures/clumping_and_volume_filling.png)

### 5. What to do for ~100 snapshots

1. In the particle pass, store per-halo linear sums only, never ratios:
   - y (with the recent-AGN flag split), n_e V, m, n_e kT, SPH volume and particle counts, plus the smoothing-length histograms;
   - on one fine nested R200c grid to >= 4 R200c (ideally about 6) that contains every edge needed: 1, 1.34, 1.5, 1.75, 2, 2.5, 3, 4 and the fit-grid edges.
   - Every later grid, fit range and cut is then exact, for one pass per snapshot.
   - Use the exact stored edge values in every later cut, with no tolerance snapping. A rounded edge silently drops a shell, as happened here at 3.6342. A loose tolerance silently adds one, as happened here when 1.34 snapped to 1.3508.
2. Store M500c and R500c per halo. Add an X500 sum block only if literature comparisons need it. The per-halo conversion reproduces the R200c stack to within 1% per band (0.996-1.010) on the shells every halo covers.
3. Stack and fit on the 35-shell fit grid in r/R200c with P200 and rho_c f_b, identical across snapshots. Use exactly the painter's constants (f_b, mu_XGP).
4. Fit to 4 R200c and paint with a hard 4 R200c cut, for every painted halo.
5. Rebin sums before forming T_e or K. Never rebin medians or stacked ratios.