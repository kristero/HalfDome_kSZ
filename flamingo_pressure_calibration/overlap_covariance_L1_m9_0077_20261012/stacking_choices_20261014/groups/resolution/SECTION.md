## Resolution criteria for 1D stacked profiles (group `resolution`)

**Data**
- FLAMINGO L1_m9 at z = 0: 4,588 centrals with inclusion weights.
- Fit grid, 35 shells:
  - a central 0-0.025 R200c sphere;
  - 22 shells of 0.069 dex from 0.025 to 0.837 R200c;
  - one 0.077-dex shell from 0.837 to 1 R200c;
  - 11 shells of 0.037-0.079 dex from 1 to 4 R200c, with edges at 1, 1.15, 1.34, 1.5, 1.75, 2, 2.18, 2.5, 3, 3.30, 3.63 and 4 R200c.
- Stacks are inclusion-weighted means. Errors come from 200 stratified bootstrap replicates, paired across variants.

**Files.** Everything is in `groups/resolution/`. Field paths in brackets refer to `results.json`.
- `web_variants.json`: 41 variants, 1,365 records, 2.48 MB, every variant x field x bin present.
- `figures/`: 12 PNGs.
- Scripts:
  - `res_common.py`: settings and masks;
  - `run_resolution.py`: masks, stacks, 1,920 B12/B16 fit jobs x 201 fits, impacts; 348 s;
  - `run_redshift.py`;
  - `assemble_results.py`;
  - `make_web.py`;
  - `make_figures.py` (`regen_caption_figs.py` redraws fig1, fig4 and fig9 after the caption corrections);
  - `section_checks.py`, `section_checks_extra.py`, `section_checks_final.py`: recompute the numbers this text quotes that are not stored directly in `results.json` (outputs `section_checks.json`, `section_checks_extra.json`, `section_checks_final.json`).

### 0. Bottom line for painting tSZ / kSZ / DM

1. **"Minimum particles per bin" is not the limiting criterion at FLAMINGO m9 resolution.**
   - The SPH smoothing-length test binds in every bin from 12.7 to 15.1. The test requires the inner edge to be >= 2 x the shell's upper-median h.
     - It binds alone in 13.1-15.1.
     - It binds jointly with the count test in 12.7-13.1.
     - In 15.1-15.5 the 0.04 R200c floor binds instead.
   - In the first shell the h test lets in, in 13.1-15.1, the per-bin minimum over haloes is 30-61 gas particles and the median is 93-199 [`per_setting.ref.summary.innermost_count_min/median`].
   - In the calibrated bins (13.5-15.5), any minimum count from 0 to 20, or the 32 used for L2p8, gives the same mask and identical observables. Outside them:
     - 0 / 5 / 10 particles keep 12.5-12.7 from 3.30 R200c, where the reference keeps nothing;
     - 32 moves 12.7-12.9 from 2.18 to 2.50 R200c and 13.1-13.3 from 1.00 to 1.15 R200c [`report_numbers.shell_shift_mc*`].
2. **Thresholds of 100 particles or more throw away resolved shells and bias the painted pressure high.**
   - They move the first kept edge outward in the calibrated bins:
     - 100: by 0-3 shells;
     - 200: by 1-5 shells;
     - 500: by 4-14 shells.
   - With 4R fits, tSZ C_ell(3000) rises by:
     - +1.00 +- 0.09% (100);
     - +2.69 +- 0.26% (200);
     - +7.05 +- 0.72% (500).
   - **The 500 value is incomplete:** bin 13.5-13.7 keeps a single shell (2.5-3.0 R200c), so it has no fit and enters as "no change". That bin holds 2.0% of the tSZ(3000) importance, 13.7% of the kSZ 1h(1000) importance and 25% of the mean-DM importance.
3. **Per-halo masks and unmasked cores change the observables more than any threshold from 0 to 200 particles.**
   - The largest change over the ten observables is +9.35% for per-halo masks (4R fits; +14.6% with 2R fits) and +4.59% for unmasked cores (4R; +11.27% with 2R). For 200 particles it is +2.69% (4R; -1.15% with 2R) [`section_checks_final.json`, `max_abs_impact`].
   - For a single observable the gap can be small: for tSZ(3000) the unmasked 4R fit gives -2.83 +- 0.12%, about the same size as the +2.69 +- 0.26% from 200 particles. A 500-particle threshold moves tSZ(3000) more than per-halo masks do (+7.05% against +4.38%).
   - Never average only the resolved haloes of a shell (per-halo masks). With 4R fits, against the reference fit: kSZ 1h(1000) +6.10 +- 0.97%, DM variance +9.35 +- 1.53%.
   - Never fit unmasked cores. With 2R fits: kSZ 1h(1000) +10.69 +- 0.35%, DM mean +7.29 +- 0.27%.
   - Replace "every halo resolved" (strict) by ">= 95% of the bin's inclusion weight resolved" (frac95), with every halo still averaged.
     - Strict tightens as N grows; frac95 does so much less (over six random seeds the median subset edge moves in 10-11 of 23 tests for strict and in 0-3 of 23 for frac95).
     - The switch changes every one of the ten first-order observables by <= 0.48% (2R and 4R fits).

### 1. What the choice is

A halo-shell counts as resolved when three conditions hold:
- **count:** at least `min_count` gas particles (all owners and phases);
- **softening:** inner edge x_in >= max(min_edge, softening_factor x eps / R200c), with eps = 5.7 physical kpc at z = 0;
- **smoothing length:** x_in >= hsml_factor x the upper-median h a / R200c of the shell's particles.

The reference is min_count 20, softening factor 4, hsml factor 2, min_edge 0.04 and the strict rule.

The rule decides which shells are kept and what enters the mean:

| Rule | Shell kept if | What enters the mean |
|---|---|---|
| strict (reference) | resolved in every halo of the bin | all haloes |
| frac95 / frac84 (local) | resolved in >= 95% / 84% of the inclusion weight | all haloes |
| diagnostic (core) | geometry passes for every halo, >= 200 particles pooled, >= 50% of haloes non-empty | all haloes |
| per_halo | at least one halo resolved | only the resolved halo-shells |
| per_halo_f50 (local) | >= 50% of the weight resolved | only the resolved halo-shells |
| none | always | all haloes |

**How to read the results**
- Every rule except the per-halo ones leaves the stacked mean unchanged, because every halo enters every shell.
  - Their stack ratio to the reference is exactly 1 with zero paired error in every bin and band [`per_setting.*.stack_band_ratios`].
  - They act only through which shells the B12 (pressure) / B16 (electron) fit sees. The painted profile is that fit, extrapolated inward.
- Each setting is therefore measured on four things:
  1. the fitted parameters;
  2. the fit integrals Y(<R200c), Y(<4R) and N_e(<4R) against the resolution-independent particle sums;
  3. the first-order observables against the reference-mask fit [`impact_fit*R_vs_ref_fit`];
  4. the first-order observables against the measured stack over all shells [`impact_fit*R_vs_truth`, the painting error].
- Whenever a setting leaves a calibrated bin with fewer than 6 shells (analysis.fitting.MIN_SHELLS), that bin has no fit and counts as "no change".
  - Its importance share is stored as `missing_share` and listed in `per_setting.*.impacts.missing_fit_bins`.
  - Only the >= 500 particle setting (and the identical count-only 500) triggers this in the calibrated range: 13.5-13.7 with 4R fits, 13.5-13.9 with 2R fits.

### 2. Which criterion binds at z = 0 (reference) [`per_setting.ref.masks`, `report_numbers`]

| Bin log M200c | N | First kept edge [R200c] | Binding | Kept shells <= 4R / <= 2R | Particles in innermost kept shell (min / median) | First edge with count >= 20 / h / softening alone |
|---|---|---|---|---|---|---|
| 12.5-12.7 | 321 | none | count (no shell passes) | 0 / 0 | - | none / 3.30 / 0.076 |
| 12.7-12.9 | 332 | 2.18 | count + h | 5 / 0 | 22 / 93 | 2.18 / 2.18 / 0.065 |
| 12.9-13.1 | 322 | 1.50 | count + h | 8 / 2 | 38 / 122.5 | 1.50 / 1.50 / 0.056 |
| 13.1-13.3 | 317 | 1.00 | h | 11 / 5 | 30 / 134 | 0.713 / 1.00 / 0.047 |
| 13.3-13.5 | 308 | 0.519 | h | 15 / 9 | 38 / 109 | 0.442 / 0.519 / 0.040 |
| 13.5-13.7 | 326 | 0.321 | h | 18 / 12 | 32 / 93 | 0.274 / 0.321 / 0.034 |
| 13.7-13.9 | 324 | 0.274 | h | 19 / 13 | 57 / 147.5 | 0.199 / 0.274 / 0.034 |
| 13.9-14.1 | 319 | 0.170 | h | 22 / 16 | 41 / 110 | 0.145 / 0.170 / 0.029 |
| 14.1-14.3 | 327 | 0.145 | h | 23 / 17 | 44 / 159 | 0.105 / 0.145 / 0.025 |
| 14.3-14.5 | 317 | 0.105 | h | 25 / 19 | 47 / 151 | 0.090 / 0.105 / 0.025 |
| 14.5-14.7 | 895 | 0.090 | h | 26 / 20 | 41 / 199 | 0.076 / 0.090 / 0.025 |
| 14.7-14.9 | 326 | 0.065 | h | 28 / 22 | 61 / 164.5 | 0.056 / 0.065 / 0.025 |
| 14.9-15.1 | 127 | 0.056 | h | 29 / 23 | 56 / 195 | 0.040 / 0.056 / 0.025 |
| 15.1-15.3 | 21 | 0.040 | min_edge | 31 / 25 | 70 / 154 | 0 / 0.034 / 0.025 |
| 15.3-15.5 | 6 | 0.040 | min_edge | 31 / 25 | 137 / 298 | 0 / 0.025 / 0.025 |

Notes on the table:
- **12.5-12.7 keeps nothing.** The h test passes only from 3.30 R200c, which leaves two shells, and both fail the count test: one halo has 19 particles in 3.30-3.63 R200c and a different halo has 16 in 3.63-4 R200c. (Twelve haloes have < 20 particles in 3.0-3.30, but that shell already fails the h test.) [recomputed in `section_checks_final.json`, `bin_12.5-12.7`; `counts_profiles.12.5-12.7.count_min`]
- **The minimum count of >= 32 holds only in 12.9-13.1 and 13.3-15.5.** It is 30 in 13.1-13.3 and 22 in 12.7-12.9.
- **Figures:**
  - `groups/resolution/figures/fig2_particle_counts_per_shell.png` shows the median, 16-84% range and minimum particles per shell.
  - `groups/resolution/figures/fig2b_inner_edge_over_smoothing_length.png` shows x_in / h.

**Why h binds before the count**
- h is the SPH kernel scale. The Wendland C2 kernel used by SWIFT has support 1.936 h (snapshot metadata), so gas structure within about 2h is a kernel average, not a resolved profile.
- In 13.1-15.1 the count-alone edge lies 1-2 fit shells inside the h-alone edge (e.g. 0.274 against 0.321 at 13.5-13.7): one or two shells inside the h edge still hold >= 20 particles in every halo.
- Softening never binds at z = 0. 4 eps/R200c is 0.0081-0.074 over all haloes and 0.0081-0.034 in the calibrated bins [`report_numbers.four_eps_over_R200c_range_*`]. The softening-alone edge lies inside the h-alone edge in every bin except 15.3-15.5, where both are 0.025 and the 0.04 floor binds.

### 3. Minimum particles per halo-shell

First kept edge [R200c] in the calibrated bins [`per_setting.*.summary.first_edge`]. Shifts in shells against the reference are in `report_numbers.shell_shift_*`.

| Setting | 13.5-13.7 | 13.7-13.9 | 13.9-14.1 | 14.1-14.3 | 14.3-14.5 | 14.5-14.7 | 14.7-14.9 | 14.9-15.1 | 15.1-15.3 | 15.3-15.5 | Shift [shells] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference (20) = 0 = 5 = 10 = 32 | 0.321 | 0.274 | 0.170 | 0.145 | 0.105 | 0.090 | 0.065 | 0.056 | 0.040 | 0.040 | 0 |
| >= 50 | 0.442 | 0.274 | 0.199 | 0.170 | 0.123 | 0.105 | 0.065 | 0.056 | 0.040 | 0.040 | 0-2 |
| >= 100 | 0.519 | 0.377 | 0.274 | 0.199 | 0.145 | 0.123 | 0.090 | 0.076 | 0.047 | 0.040 | 0-3 (3 at 13.5-13.7 and 13.9-14.1) |
| >= 200 | 0.713 | 0.519 | 0.321 | 0.274 | 0.199 | 0.170 | 0.123 | 0.090 | 0.065 | 0.047 | 1-5 |
| >= 500 | 2.50 | 0.837 | 0.519 | 0.377 | 0.274 | 0.233 | 0.170 | 0.123 | 0.090 | 0.076 | 4-14 (14 at 13.5-13.7; 6-7 in 13.7-14.9) |
| >= 20, other criteria off | 0.274 | 0.199 | 0.145 | 0.105 | 0.090 | 0.076 | 0.056 | 0.040 | 0 | 0 | -4 to -1 |

**The count criterion on its own**, with softening, h and min_edge switched off [`report_numbers.n_kept_mconly*`]:
- 0 particles keeps all 35 shells, an unmasked core identical to "none".
- 5 particles keeps 22-35 shells in the calibrated bins (5-35 over all bins).
- 10 particles keeps 20-35 shells in the calibrated bins (5-35 over all bins).
- 20 particles gives the same edges as h factor 1 in every bin up to 15.1 (see section 4 for why).

**At >= 500 particles the mask has gaps.** In 13.7-13.9 and 13.9-14.1 some of the thinner shells beyond 1 R200c fail the count while wider shells around them pass [`per_setting.resolution:mc500.masks.*.contiguous`].
- In 13.7-13.9 the setting keeps 6 of the 12 shells between 0.837 and 4 R200c (0.837-1, 1.15-1.34, 1.5-1.75, 1.75-2, 2.18-2.5 and 2.5-3 R200c; 4 of them inside 2R).
- In 13.9-14.1 it keeps every shell from 0.519 R200c except 2-2.18 R200c.
- The 4R fit at 13.7-13.9 therefore rests on 6 scattered shells [`section_checks_final.json`, `mc500_noncontiguous`].

**Why raising the threshold biases the painted pressure high**
- Raising the threshold drops resolved shells (in the calibrated bins: 0.040-0.52 R200c for >= 100 particles, 0.040-0.71 R200c for >= 200). The B12 core shape is then set by the outer shells, and its inward extrapolation overshoots the measured, flatter, resolved pressure.
- For >= 500 particles, the 4R fit at 13.7-13.9 is 2.00 +- 0.54 times the reference fit at < 0.1 R200c and 1.64 +- 0.28 times at 0.1-0.25 R200c (medians over the shells of each band) [`per_setting.resolution:mc500.fits.P.4R.13.7-13.9.band_ratio_vs_ref`].
- For 100 particles the same band medians are 1.055 +- 0.019 and 1.037 +- 0.012.
- Fig: `groups/resolution/figures/fig1_innermost_radius_vs_mass.png` and `fig8_kept_shells_per_setting.png`.

### 4. Softening, smoothing length, minimum edge

**Softening (factor 0, 2, 3, 6)**
- It never changes a mask in any of the 15 bins, and the observables are identical.

**h factor**
- **Factor 1, and factor 0 (criterion off)**, move the edge 1-2 shells inward:
  - 2 shells in 13.7-13.9, 14.1-14.3 and 14.9-15.1;
  - 1 shell in the other bins from 13.5 to 14.9;
  - 0 in 15.1-15.5.
  - Factors 0 and 1 give identical masks because at factor 1 the h test no longer limits anything. Its edge (0.145 / 0.123 / 0.076 / 0.065 / 0.047 / 0.040 / 0.025 / 0.025 in the eight bins 13.5-15.1) lies 3-5 shells inside the count >= 20 edge (0.274 / 0.199 / 0.145 / 0.105 / 0.090 / 0.076 / 0.056 / 0.040). The count test therefore sets the edge both with factor 1 and with the test off. In 15.1-15.5 the 0.04 floor sets it in both cases [recomputed in `section_checks_final.json`, `h1_vs_count20`].
  - With 4R fits the observables change by -0.45 +- 0.04% (tSZ 3000) to +0.17 +- 0.03% (DM variance). With 2R fits they change by -0.04% to +0.50 +- 0.02% (DM variance).
- **Factor 3** moves the edge outward by 2-4 shells in 13.5-15.3:
  - 4 at 13.5-13.7;
  - 3 in 13.9-14.9;
  - 2 in 13.7-13.9 and 14.9-15.3;
  - 0 at 15.3-15.5.
  - With 4R fits, tSZ(3000) rises by +1.47 +- 0.14% and kSZ 1h(3000) by +0.95 +- 0.10%.

**min_edge**
- The 0.04 R200c floor is the binding criterion in 15.1-15.5.
- **0** changes only those two bins (edges 0.034 / 0.025): tSZ(1000) -0.25 +- 0.09%.
- **0.1** cuts the bins from 14.5 upward at 0.105, removing 1-6 shells: tSZ(1000) +1.17 +- 0.35% and tSZ(3000) +1.05 +- 0.21%.

### 5. The masking rule

#### (a) The strict rule depends on N [`n_dependence_of_strict`]

I drew 200 random subsets of 30, 100 and 300 haloes per bin (only sizes smaller than the bin).
- Across 23 (bin, size) tests in 13.5-15.1, the median strict edge differs from the full-sample edge in **10 of 23** tests.
- The frac95 edge differs in **2 of 23** (13.5-13.7 with 30 haloes; 14.9-15.1 with 100).
- These counts depend on the random draws. Repeating the test with five other seeds gives 10-11 of 23 for strict and 0-3 of 23 for frac95 [`section_checks_extra.json`, `n_dependence_seeds`]. The contrast between the rules holds for every seed; the exact frac95 count does not.

Examples (13.1-13.3 is an exploratory bin, outside the 23 tests):

| Bin | Strict, full sample | Strict, subsets (median) | frac95, full | frac95, subsets |
|---|---|---|---|---|
| 14.5-14.7 (N = 895) | 0.090 | 300: 0.076; 100: 0.076; 30: 0.065 | 0.065 | 0.065 for all sizes |
| 13.7-13.9 (N = 324) | 0.274 | 300: 0.274; 100: 0.233; 30: 0.233 | 0.233 | 0.233 for all sizes |
| 13.1-13.3 (N = 317) | 1.00 | 300: 1.00; 100: 0.837; 30: 0.713 | 0.713 | 0.713 for all sizes |

Why this matters:
- "Every halo" is an extreme-value statistic. Its threshold is set by the single worst halo, which moves outward as more haloes are added.
- Across about 100 snapshots N per bin will vary, so strict masks would shift for statistical, not physical, reasons.
- frac95 against strict, 4R fits [`per_setting.resolution:frac95.summary.impact_fit4R_vs_ref_fit`]:
  - tSZ(3000): -0.47 +- 0.04%
  - kSZ 1h(1000): +0.14 +- 0.01%
  - DM variance: +0.20 +- 0.03%
  - The largest change over all ten observables is -0.48 +- 0.04% (tSZ 1h, ell = 3000). With 2R fits it is +0.46 +- 0.02% (DM variance).
- Its painting error for tSZ(3000) is +0.76 +- 0.15%, against +1.25 +- 0.15% for strict.

#### (b) Per-halo masks are biased [`per_halo_bias`, `per_halo_bias_by_fraction`; figures `fig3_per_halo_bias_vs_fraction.png`, `fig10_per_halo_band_ratios.png`]

The table gives the per-halo mean divided by the all-halo mean in the partially resolved shells of the calibrated bins. Each entry is the median over the shells in its class, not a single-shell value.

| Weight fraction resolved | Shells | P | n_e | rho_gas | T_e | K |
|---|---|---|---|---|---|---|
| 0.9 to < 1 | 14 | 1.017 | 1.016 | 1.016 | 1.002 | 0.989 |
| 0.8-0.9 | 1 | 1.082 | 1.068 | 1.066 | 1.025 | 0.966 |
| 0.6-0.8 | 6 | 1.137 | 1.124 | 1.123 | 1.019 | 0.916 |
| 0.4-0.6 | 2 | 1.313 | 1.278 | 1.279 | 1.041 | 0.871 |
| 0.2-0.4 | 4 | 1.419 | 1.406 | 1.389 | 1.029 | 0.805 |
| 0.05-0.2 | 5 | 1.765 | 1.742 | 1.721 | 1.024 | 0.681 |
| < 0.05 | 18 | 2.32 (P shells range 0.70-11.3) | 2.52 | 2.81 | 1.034 (shells range 0.46-1.20) | 0.527 |

The per-halo / reference ratio per band, as the median over the band's kept shells [`per_setting.resolution:per_halo.stack_band_ratios`; paired-bootstrap SD in brackets; * = the band holds a single kept shell]:

| Bin | P, < 0.1 R200c | n_e, < 0.1 R200c | P, 0.1-0.25 R200c | n_e, 0.1-0.25 R200c |
|---|---|---|---|---|
| 13.5-13.7 | 1.495 (0.078)* | 3.07 (0.12)* | 2.84 (0.81) | 2.81 (0.78) |
| 13.7-13.9 | 1.64 (0.48) | 2.09 (0.27) | 1.38 (0.07) | 1.38 (0.05) |
| 13.9-14.1 | 5.87 (2.11) | 6.65 (2.63) | 1.017 (0.005) | 1.017 (0.005) |
| 14.1-14.3 | 1.28 (0.14) | 1.30 (0.14) | 1.000 | 1.000 |
| 14.3-14.5 | 1.33 (0.11) | 1.23 (0.09) | 1.000 | 1.000 |
| 14.5-14.7 | 1.049 (0.005) | 1.041 (0.004) | 1.000 | 1.000 |
| 14.7-14.9 | 1.001 (0.001) | 1.001 (0.001) | 1.000 | 1.000 |
| 14.9-15.5 | 1.000 | 1.000 | 1.000 | 1.000 |

The 0.1-0.25 entries, including the 1.000 ones, are band medians too, and single shells inside the band are more biased:
- The largest single-shell P ratio in the band is 4.65 at 13.5-13.7 (0.105-0.123 R200c, 0.3% of the weight resolved), 1.90 at 13.7-13.9 and 1.34 at 13.9-14.1 (both also at 0.105-0.123 R200c).
- In 14.1-14.3 the band holds two partially resolved shells among five: 0.105-0.123 and 0.123-0.145 R200c, with 96% and 99% of the weight resolved and P ratios 1.023 and 1.007 [`section_checks_final.json`, `per_halo_partial_shells_0.1_0.25`].

Bands >= 0.25 R200c:
- In the calibrated bins every band median is exactly 1 (P, n_e, rho_gas, T_e and K). This does not mean every shell there is fully resolved.
  - In 13.5-13.7 the 0.25-0.5 band holds two partially resolved shells: 0.233-0.274 R200c (69% of the weight resolved, P ratio 1.127) and 0.274-0.321 (94%, 1.026). The band median is 1 because the other three shells are fully resolved.
  - In 13.7-13.9 the 0.233-0.274 shell is 99.7% resolved, ratio 1.002 [`section_checks_extra.json`, `per_halo_shells_ge025_not_one`].
- In the exploratory bins (12.5-13.5) the band medians at 0.25-1 R200c run from 1.00 (13.3-13.5, 0.5-1) up to 2.98 (P) and 6.77 (n_e), both at 12.5-12.7 in the 0.25-0.5 band. For example, P is 1.31 at 13.1-13.3 and 1.16 at 13.3-13.5 in the 0.25-0.5 band.

**Why**
- Both the count test and the h test pass first in haloes whose central gas is denser: more particles per shell, and a smaller h, since h is proportional to (m / rho)^(1/3).
- The mask is therefore correlated with the averaged quantity. The per-halo mean is E[F | dense core], not E[F].
- So density and pressure come out high and entropy low. T_e is nearly unbiased in the class medians (+0.2 to +4%), although single shells with < 5% of the weight resolved range from 0.46 to 1.20.
- HalfDome paints every halo, so the target is the all-halo mean.

**Consequences**
- **Painting the per-halo stack itself** [`impact_stack`; shells with no resolved halo, 0.1-0.7% of the importance, count as no change]:
  - tSZ(3000): +2.91 +- 0.56%
  - DM variance: +3.69 +- 0.89%
- **Fitting it:**
  - ln P0 moves by +1.78 +- 0.94 at 13.5-13.7 with the 4R fit (+2.58 +- 1.62 with 2R). Each shift alone is below 2 sigma.
  - Against the reference fit (4R): tSZ(3000) +4.38 +- 0.62%, kSZ 1h(1000) +6.10 +- 0.97%, DM mean +2.71 +- 0.46%, DM variance +9.35 +- 1.53%.
  - With 2R fits the changes are larger: tSZ(3000) +6.30 +- 0.74%, kSZ 1h(1000) +12.91 +- 2.29%, DM mean +7.31 +- 1.41%, DM variance +13.96 +- 2.53%.
- **Restricted to shells with >= 50% kept** (per_halo_f50), there is still a bias. With 4R fits:
  - kSZ 1h: +0.72 +- 0.04% (1000) and +0.83 +- 0.06% (3000);
  - DM variance: +1.16 +- 0.07%.
  - With 2R fits the DM variance changes by +1.91 +- 0.06%.

#### (c) Unmasked cores ('none') and the diagnostic rule [`inner_decomposition`, `none_inner`, `counts_profiles`; figure `fig4_unmasked_inner_shells.png`]

The table describes the shells the reference rejects. Each value is given at x = 0.027 / 0.052 R200c (fit shells 1 / 5). "Extrapolation" means the reference B12 / B16 fit to 2 R200c, extrapolated inward.

| Bin | P stack / extrapolation | P with recent-AGN gas kept / removed | n_e stack / extrapolation | Hot (X-ray phase) n_e / extrapolation | Cold/SF share of n_e | Hot share of the n_e excess | Haloes with zero gas particles | Median particles per halo-shell | Median x_in / h | Share of the stack's Y(<R200c) in rejected shells |
|---|---|---|---|---|---|---|---|---|---|---|
| 13.5-13.7 | 0.55 / 0.61 | 1.08 / 1.00 | 7.40 / 2.47 | 1.72 / 1.49 | 0.77 / 0.40 | 0.11 / 0.33 | 54% / 38% | 0 / 1 | 0 / 0.63 | 17.2% |
| 13.9-14.1 | 0.65 / 0.96 | 1.006 / 1.00 | 3.07 / 1.87 | 1.84 / 1.74 | 0.40 / 0.07 | 0.41 / 0.84 | 23% / 2% | 2 / 5 | 0.79 / 1.12 | 6.7% |
| 14.3-14.5 | 0.73 / 0.99 | 1.016 / 1.00 | 1.79 / 1.36 | 1.61 / 1.35 | 0.10 / 0.01 | 0.77 / 0.95 | 3% / 0% | 5 / 22 | 1.13 / 1.67 | 3.0% |
| 14.9-15.1 | 0.87 / 0.97 | 1.000 / 1.000 | 1.42 / 1.03 | 1.40 / 1.02 | 0.02 / 0.005 | 0.94 / 0.82 | 0% / 0% | 29 / 129 | 1.81 / 2.97 | 0.8% |

Column sources:
- All ratio, share and median columns are in `inner_decomposition.<bin>`: `P_over_ref2R_extrapolation`, `P_agn_kept_over_removed`, `E_over_ref2R_extrapolation`, `E_xray_over_ref2R_extrapolation`, `E_cold_sf_share`, `E_excess_hot_share`, `count_median`, `xh_median`. A median x_in / h of 0 means the median halo has no particle in that shell.
- "Zero gas particles" = `counts_profiles.<bin>.count_zero_fraction`. The fraction of haloes with no gas other than recently AGN-heated gas (`no_nonagn_gas_fraction`) is 56% / 38% and 24% / 3% in the first two bins.
- The Y share is `none_inner.<bin>.share_Y200_in_unkept`.

**What the rejected shells contain**
- **Shot noise and kernel averaging.**
  - In 13.5-13.7 every rejected shell inside 0.105 R200c has a median of 0-4 particles per halo at a median x_in / h <= 1.07.
  - In 13.9-14.1 the shells at 0.025-0.056 R200c have medians of 2-5 particles at x_in / h = 0.79-1.12.
  - There the per-halo profile is shot noise plus a kernel average.
  - At 14.3-14.5 the medians at x = 0.027 / 0.052 are 5 / 22 particles at 1.1 / 1.7 h: better sampled, but still inside the 2h kernel support.
  - At 14.9-15.1 they are 29 / 129 particles at 1.8 / 3.0 h. At x = 0.027 the median halo is still inside 2h. At x = 0.052 the median halo passes the h test and 99.2% of the inclusion weight is resolved; that shell is rejected only because the strict rule requires every halo, and frac95 keeps it [`resolved_fraction_per_shell.ref.14.9-15.1`].
- **Pressure lies below the B12 cusp in the inner rejected shells, but not everywhere.** [`inner_decomposition.<bin>.P_over_ref2R_extrapolation`; `section_checks_final.json`, `P_inner`]
  - At x = 0.027 and 0.052 the stack is below the extrapolation in all four tabulated bins (0.55-0.99).
  - In 13.5-13.7 the stack is below it only at 0.025-0.105 R200c. It is 2-12% above it at 0.105-0.321 R200c, and 1.40 times the extrapolation in the central 0-0.025 sphere.
  - In 13.9-14.1 it is below at 0-0.056 R200c and 2-10% above at 0.056-0.170 R200c.
  - In 13.7-13.9 (not tabulated) it is 0.2-3% above at 0.090-0.233 R200c. In 14.1-14.3 (not tabulated) it is 1-4% above at 0.047-0.145 R200c, so there x = 0.052 already lies above the extrapolation (1.010).
- **Removing recently AGN-heated gas does not explain the deficit at 0.025-0.1 R200c.** [`inner_decomposition.<bin>.P_agn_kept_over_removed`; `section_checks_final.json`, `P_inner`]
  - In 13.5-13.7 at x = 0.027, keeping that gas raises P by 8%. P / extrapolation goes from 0.552 to 0.597, which closes about 10% of the 45% deficit. At x = 0.052 it changes nothing (deficit 39%).
  - In 13.7-13.9 it raises P by 5% at x = 0.027 (closing 5% of a 50% deficit) and by 0% at 0.052. In 13.9-14.1 it raises P by 0.6% and 0%.
  - Outside the central sphere it never raises P by more than 8% in any calibrated bin.
  - In the central 0-0.025 sphere it raises P by:
    - 73% at 13.5-13.7, where the stack already lies at 1.40 times the extrapolation;
    - 59% at 13.7-13.9, closing 84% of a 41% deficit;
    - 28% at 13.9-14.1, closing 39% of a 41% deficit;
    - 19% at 14.1-14.3 and 10% at 14.3-14.5;
    - <= 4% from 14.5 upward.
  - Where the deficit remains, it is in the hot phase itself: the hot-phase P is 0.548 / 0.605 of the extrapolation at x = 0.027 / 0.052 in 13.5-13.7.
- **n_e rises above the B16 extrapolation.** Which phase carries the excess depends on radius and mass [`section_checks_final.json`, `ne_inner`]:
  - Cold / star-forming gas, whose density is set by the subgrid equation of state, is most of n_e in the central 0-0.025 sphere from 13.5 to 14.3 (82%, 82%, 67% and 54% of n_e). At x = 0.027 it is most of n_e only in 13.5-13.9 (77% and 65%).
  - It also supplies most of the central-sphere excess over the extrapolation up to 14.5: the hot share of that excess is 0.12-0.32 in 13.5-14.5, 0.49 at 14.5-14.7 (about half each) and 0.67-0.90 from 14.7 upward.
  - Integrated over volume, over the rejected shells inside R200c where n_e exceeds the extrapolation, the excess is mostly hot gas above the fitted form. The hot share is 0.60 (13.5-13.7), 0.76 (13.7-13.9) and 0.82-0.93 in 13.9-15.5 (in 15.3-15.5 only the central sphere exceeds the extrapolation).
  - Hot n_e alone is 1.35-1.84 times the extrapolation at x = 0.027 and 0.052 in 13.5-14.5. In 14.5-15.1 it is 1.40-1.60 times at x = 0.027 and 1.02-1.17 times at x = 0.052.
  - So at x of about 0.03-0.05 the rejected gas in the tabulated bins is denser and has lower pressure (hence is cooler) than the inward extrapolation of the outer-shell fits predicts. At 14.9-15.1, x = 0.052, both differences are only 3%.
  - Whether this is physical (cool-core-like centres) or numerical (kernel smoothing and over-cooling at low particle number) cannot be decided from one resolution level. A comparison with a higher-resolution run (e.g. FLAMINGO L1_m8) would settle it; that was not done here.

**Effects of fitting all shells ('none', identical to count-only 0)**
- **2R fits:**
  - ln rho0 of B16 rises by +3.70 +- 0.37 at 13.5-13.7.
  - Against the reference fit: kSZ 1h(1000) +10.69 +- 0.35%, DM mean +7.29 +- 0.27%, DM variance +8.49 +- 0.24%, tSZ(3000) -1.47 +- 0.12%.
- **4R fits:**
  - kSZ 1h(1000) +3.85 +- 0.13%, DM mean +2.28 +- 0.09%, DM variance +4.59 +- 0.16%.
  - tSZ(3000) falls by -2.83 +- 0.12%.
- **The diagnostic rule** keeps 0.040 R200c outward in all calibrated bins. With 4R fits it gives kSZ 1h(1000) +1.51 +- 0.06% and DM variance +2.25 +- 0.10% (2R fits: DM variance +3.51 +- 0.09%).

### 6. Fits and integrated signal [`per_setting.*.fits`; figures `fig5_fit_parameter_shifts_2R.png`, `fig5_fit_parameter_shifts_4R.png`, `fig6_fit_integrals_vs_particle_sums.png`]

The stack's particle sums over all shells give Y and N_e independently of the kernel. They therefore test whether the inward extrapolation conserves the integrated signal.

Ranges below are fit integral / particle sum - 1 [%] over the calibrated bins that have a fit.

| Setting | Y(<R200c), 2R fit | Y(<R200c), 4R fit | Y(<4R), 4R fit | N_e(<4R), 4R fit |
|---|---|---|---|---|
| reference | -1.19 to +0.05 (13.5-13.7: -0.54) | -1.30 to +0.68 | -5.07 to +0.10 | -4.02 to +4.39 |
| >= 95% resolved | -1.19 to +0.12 | -1.30 to +0.40 | -5.07 to -0.02 | -4.13 to +4.39 |
| >= 100 particles | -1.20 to +0.08 | -1.29 to +2.70 | -3.55 to +0.37 | -3.69 to +3.68 |
| h factor 3 | -1.29 to +0.11 | -1.29 to +5.68 | -2.09 to +0.37 | -3.44 to +3.63 |
| >= 500 particles (2R: no fit at 13.5-13.9; 4R columns: no fit at 13.5-13.7) | -1.42 to +1.00 | -1.35 to +21.95 (+21.95 at 13.7-13.9) | -1.06 to +6.38 | -3.61 to +4.17 |
| no mask | -1.69 to +0.04 | -1.45 to -0.12 | -6.07 to -0.61 | -2.95 to +11.05 |
| per-halo | -1.19 to +10.07 | -1.30 to +9.83 | -1.20 to +14.52 | -2.76 to +9.67 |

What the table shows:
- **Y(<R200c) is robust.** With the reference mask and the 2R fit it is conserved to <= 1.2%, even though 17% of the stack's Y(<R200c) sits in rejected shells at 13.5-13.7 [`none_inner.13.5-13.7.share_Y200_in_unkept`].
- **The h-limited mask is not the accuracy bottleneck.**
  - The reference B12 4R fit has 12.7-17.8% rms residuals in 13.5-15.3 (6.8% at 15.3-15.5). The 2R fit has 0.3-1.5% in 13.5-15.1 (3.9% and 5.5% in 15.1-15.3 and 15.3-15.5) [`fits.P.*.rms_percent`].
  - With the reference mask, Y(<R200c) of the 4R fit is within -1.30 to +0.68%. So the Y(<4R) deficit of up to 5.07% (13.5-13.7) arises at 1-4 R200c, where every shell is resolved. It reflects the B12 form in the outskirts, not the mask.
- **Removing inner shells makes Y(<R200c) of the 4R fit overshoot:** by up to +2.70% (>= 100, 13.5-13.7), +5.68% (h factor 3, 13.5-13.7) and +21.95% (>= 500, 13.7-13.9, a fit to 6 scattered shells from 0.837 R200c outward; section 3). The 2R fits of the same settings stay within -1.42 to +1.00%.
- **Unmasked cores make the n_e integral overshoot** by +11% at 13.5-13.7.
- **Parameter shifts can be large while observables move only a few percent**, because P0, x_c and beta are degenerate. For example, mc500 at 13.7-13.9 (4R): d ln P0 = +0.96 +- 0.35, d ln x_c = -0.53 +- 0.18.

### 7. Observable impact (first order, calibrated bins 13.5-15.5, B12/B16 fits to 4 R200c) [`per_setting.*.summary.impact_fit4R_vs_ref_fit`; figure `fig7_observable_impact.png`]

The table gives dO/O [%] against the reference-mask fit, +- the paired-bootstrap 1-sigma error.

| Setting | tSZ C_ell 100 | tSZ C_ell 1000 | tSZ C_ell 3000 | kSZ 1h 1000 | kSZ 1h 3000 | DM mean | DM variance |
|---|---|---|---|---|---|---|---|
| 0 / 5 / 10 / 32 particles; softening factor 0, 2, 3, 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| >= 50 particles | +0.07 +- 0.01 | +0.11 +- 0.01 | +0.21 +- 0.03 | -0.04 +- 0.02 | +0.24 +- 0.02 | -0.08 +- 0.01 | +0.10 +- 0.03 |
| >= 100 particles | +0.45 +- 0.05 | +0.64 +- 0.06 | +1.00 +- 0.09 | -0.06 +- 0.04 | +0.51 +- 0.07 | -0.12 +- 0.01 | +0.25 +- 0.07 |
| >= 200 particles | +1.11 +- 0.11 | +1.62 +- 0.15 | +2.69 +- 0.26 | +0.37 +- 0.12 | +1.87 +- 0.19 | -0.11 +- 0.02 | +1.29 +- 0.21 |
| >= 500 particles (*) | +2.63 +- 0.27 | +4.02 +- 0.39 | +7.05 +- 0.72 | +0.42 +- 0.20 | +2.03 +- 0.30 | (0.00 +- 0.04, not a measurement) | +1.80 +- 0.33 |
| missing importance share for (*) | 1.0% | 1.0% | 2.0% | 13.7% | 19.0% | 25.1% | 10.7% |
| h factor 1 (= 0) | -0.25 +- 0.03 | -0.33 +- 0.03 | -0.45 +- 0.04 | +0.15 +- 0.02 | +0.09 +- 0.03 | +0.06 +- 0.00 | +0.17 +- 0.03 |
| h factor 3 | +0.59 +- 0.06 | +0.90 +- 0.09 | +1.47 +- 0.14 | +0.08 +- 0.06 | +0.95 +- 0.10 | -0.12 +- 0.02 | +0.59 +- 0.11 |
| min edge 0 | -0.23 +- 0.08 | -0.25 +- 0.09 | -0.12 +- 0.04 | -0.01 +- 0.00 | 0.00 | 0.00 | -0.02 +- 0.00 |
| min edge 0.1 | +1.06 +- 0.30 | +1.17 +- 0.35 | +1.05 +- 0.21 | +0.19 +- 0.01 | +0.18 +- 0.01 | +0.03 +- 0.00 | +0.41 +- 0.02 |
| >= 95% resolved | -0.21 +- 0.02 | -0.31 +- 0.03 | -0.47 +- 0.04 | +0.14 +- 0.01 | +0.14 +- 0.02 | +0.04 +- 0.00 | +0.20 +- 0.03 |
| >= 84% resolved | -0.33 +- 0.03 | -0.47 +- 0.04 | -0.69 +- 0.05 | +0.15 +- 0.02 | +0.10 +- 0.03 | +0.06 +- 0.00 | +0.18 +- 0.03 |
| diagnostic | -0.50 +- 0.04 | -0.72 +- 0.05 | -1.12 +- 0.08 | +1.51 +- 0.06 | +1.47 +- 0.07 | +0.56 +- 0.02 | +2.25 +- 0.10 |
| per-halo, >= 50% kept | -0.08 +- 0.04 | -0.08 +- 0.06 | -0.02 +- 0.09 | +0.72 +- 0.04 | +0.83 +- 0.06 | +0.20 +- 0.01 | +1.16 +- 0.07 |
| per-halo | +2.09 +- 0.43 | +2.64 +- 0.46 | +4.38 +- 0.62 | +6.10 +- 0.97 | +5.73 +- 0.85 | +2.71 +- 0.46 | +9.35 +- 1.53 |
| no mask | -1.55 +- 0.12 | -2.15 +- 0.13 | -2.83 +- 0.12 | +3.85 +- 0.13 | +2.23 +- 0.10 | +2.28 +- 0.09 | +4.59 +- 0.16 |

(*) Bin 13.5-13.7 keeps one shell (2.5-3.0 R200c), so it has no fit and enters as "no change". The row below gives its share of each observable's importance [`per_setting.resolution:mc500.summary.impact_fit4R_vs_ref_fit.*.missing_share`].
- The >= 500 numbers therefore leave out the change in 13.5-13.7, whose size and sign are not measured. For kSZ and DM, where that bin carries 11-25% of the importance, the row is incomplete, not a bound.
- The DM-mean entry carries no information.
- With 2R fits, bins 13.5-13.7 and 13.7-13.9 have no fit (0 and 4 kept shells within 2R). The missing shares are then 6% (tSZ 3000), 30% (kSZ 1000) and 47% (DM mean).

Error sizes:
- In this table (4R fits against the reference fit), outside the two per-halo rows the paired 1-sigma errors are <= 0.72%. The largest are for >= 500 particles: 0.72% for tSZ(3000) (total and 1h), then 0.48% (y_mean) and 0.39% (tSZ 1000) [`report_numbers.largest_impact_errors_non_per_halo_fit4R_vs_ref_fit`].
- The per-halo row has errors of 0.43-1.53%; the per-halo >= 50% row has 0.01-0.09%.

**Painting error against the measured stack (all shells, 4R fit)** [`impact_fit4R_vs_truth`]:

| Setting | tSZ 1000 | tSZ 3000 | kSZ 1h 1000 | DM mean | DM variance |
|---|---|---|---|---|---|
| reference | +1.27 +- 0.22 | +1.25 +- 0.15 | +2.60 +- 0.14 | +0.55 +- 0.09 | +0.30 +- 0.08 |
| >= 95% resolved | +0.95 +- 0.21 | +0.76 +- 0.15 | +2.73 +- 0.14 | +0.59 +- 0.09 | +0.47 +- 0.08 |
| >= 100 particles | +1.92 +- 0.24 | +2.28 +- 0.20 | +2.51 +- 0.15 | +0.38 +- 0.08 | +0.56 +- 0.13 |
| h factor 3 | +2.18 +- 0.25 | +2.76 +- 0.24 | +2.63 +- 0.16 | +0.37 +- 0.08 | +0.88 +- 0.16 |
| per-halo | +3.89 +- 0.49 | +5.87 +- 0.67 | +8.17 +- 0.83 | +3.11 +- 0.43 | +7.76 +- 1.08 |
| no mask | -0.93 +- 0.27 | -1.67 +- 0.18 | +6.34 +- 0.20 | +2.87 +- 0.14 | +4.02 +- 0.13 |

- The "truth" here includes the measured but unresolved inner shells, so it is not truth inside the resolution limit.
- The comparison is still useful:
  - Settings that drop resolved shells overshoot the measured tSZ(3000) by more than the reference does: +2.28% (>= 100) and +2.76% (h factor 3), against +1.25%.
  - Unmasked fits overshoot the measured n_e-driven observables: kSZ 1h(1000) +6.34% and DM mean +2.87%, against +2.60% and +0.55% for the reference.

### 8. Grid dependence [`grid_dependence`, `grid_gaps_strict`]

- In the calibrated bins the strict first edge on the R200c grids (fit, radii234 'old', lee22, x200) differs by at most 0.045 dex, which is less than one 0.069-dex fit shell [`section_checks_extra.json`, `grid_dex_vs_fit`].
  - The largest differences are 0.247 against 0.274 at 13.7-13.9 (lee22 and x200) and 0.099 against 0.090 at 14.5-14.7 (lee22).
  - fit and old agree exactly from 13.1 upward. 13.5-13.7 is 0.321 on fit, old and x200, and 0.320 on lee22.
  - In the exploratory bins the difference reaches 0.099 dex (13.1-13.3 on lee22: 0.796 against 1.00), and lee22 keeps no shell in 12.5-13.1.
- **The finest grid shows that a count threshold is grid dependent.**
  - On x200 (median 0.035 dex, but with sliver shells as narrow as 0.001 dex next to the exact edges), the strict mask has 41 gaps beyond the first kept edge, in 14 bins. With the reference settings the fit grid has none (at >= 500 particles the fit grid has gaps too; section 3).
  - 38 of these gaps fail only the count test. All 41 are 0.001-0.026 dex wide.
- The h test does not depend on shell width. This is another reason to let h, not a count, set the limit, and to avoid sliver shells or merge them before testing counts.

### 9. Redshift dependence [`redshift`; figure `fig9_redshift_dependence.png`]

**Softening.** eps = min(22.3 a ckpc, 5.7 pkpc).
- At fixed M200c, eps/R200c grows by 1.20 / 1.46 / 2.08 / 2.67 / 2.65 at z = 0.5 / 1 / 2 / 3 / 4 [`redshift.analytic_L1m9_bins.*.eps_over_R_factor_vs_z0`; self-similar estimate].
- At z = 3 the 4 eps/R200c limit for 13.5-13.7 is 0.091 R200c. The first fit-grid edge outside it is 0.105.

**Measured on L2p8_m9 stacks** [`redshift.L2p8_empirical`]: same particle mass; bins subsampled to <= 200 haloes (10-200 per bin; 10-12 in the most massive bin at each z); h from 0.05-dex histograms. The trends below use the frac95 rule.
- **The h test is among the binding criteria in 57 of the 59 (bin, z) cases.**
  - The two exceptions are 15.0-15.2 at z = 0 and z = 0.5 (10 haloes each), where the 0.04 floor binds.
  - With the strict rule the count is 56 of 59. At z = 2 in 13.8-14.0 the strict edge (0.155) lies outside every single-criterion edge (h alone: 0.115), so only the combination of criteria binds [`section_checks_extra.json`, `L2p8`; `section_checks_final.json`, `L2p8`].
- **4 eps/R200c itself never sets the first edge.** In L2p8 the softening test includes the 0.04 floor. That combined test binds only at the floor: alone at 15.0-15.2 (z = 0, 0.5), and tied with the count and h tests at 14.4-14.6 (z = 2) and 14.8-15.0 (z = 1). This holds for both rules.
- The first kept edge moves *inward* with redshift:
  - 13.4-13.6: 0.371 (z = 0), 0.321 (z = 0.5, 1), 0.277 (z = 1.5), 0.207 (z = 2), 0.179 (z = 3, 4);
  - 14.0-14.2: 0.134 (z = 0), 0.115 (z = 0.5-1.5), 0.100 (z = 2, 3; 10 haloes at z = 3).
- In the shell starting at 0.115 R200c in 13.4-13.6, the median particle count rises from 5 (z = 0) to 35 (z = 4), and the median x_in/h from 1.15 to 2.30.
- Interpretation (not tested here): at fixed mass, high-z haloes hold more gas in their inner regions, presumably because feedback has had less time to remove it.

So masks must be recomputed per snapshot. Freezing the z = 0 masks would discard resolved shells at high z.

### 10. Recommendations for processing ~100 FLAMINGO snapshots and painting HalfDome

1. **Keep the smoothing-length test as the primary criterion,** with inner edge >= 2 x the shell's upper-median h.
   - It is the physically meaningful limit (kernel support 1.936 h), and it does not depend on shell width.
   - It binds at every mass from 12.7 to 15.1 at z = 0.
   - In L2p8 (bins subsampled to <= 200 haloes) it is among the binding criteria in 57 of 59 (bin, z) cases up to z = 4 with frac95, and in 56 of 59 with strict. The exceptions are the 0.04 floor at 15.0-15.2 (z = 0 and 0.5) and, under strict, a combined-criterion edge at z = 2 in 13.8-14.0.
   - Factor 1 changes observables by <= 0.45% with 4R fits and <= 0.50% with 2R fits, because the count test then sets the edge. Factor 3 throws away resolved shells (tSZ(3000) +1.47 +- 0.14%).
2. **Keep min_count = 20 (or 32) and softening factor 4 as cheap safety floors.**
   - In the calibrated bins at z = 0 they change no mask: min_count 0-32 and softening factors 0-6 all give the reference mask.
   - Keep the 0.04 R200c floor too, but it is not inert. It binds in 15.1-15.5, and removing it moves those edges to 0.034 / 0.025 R200c (tSZ(1000) -0.25 +- 0.09%).
   - Do not raise min_count to >= 100. It discards resolved 0.04-0.52 R200c shells and biases the painted tSZ(3000) high: +1.00% at 100, +2.69% at 200, +7.05% at 500. The 500 value excludes 13.5-13.7, which has no fit.
3. **Use frac95 instead of strict, and average all haloes.**
   - It is much more stable against the number of haloes per bin: over six seeds the median subset edge differs in 0-3 of 23 tests, against 10-11 of 23 for strict.
   - It keeps 0-2 extra shells in the calibrated bins and changes all ten observables by <= 0.48%.
4. **Never use per-halo masks.**
   - With 4R fits they change kSZ 1h(1000) by +6.10% and DM variance by +9.35% (2R fits: +12.91% and +13.96%).
   - Even restricted to shells with >= 50% kept, they bias kSZ 1h by +0.7-0.8% and the DM variance by +1.16% (4R fits; +1.91% with 2R fits).
   - **Never fit unmasked cores** (2R fits: kSZ 1h(1000) +10.69%, DM mean +7.29%).
   - Do not use the pooled 'diagnostic' rule either: DM variance +2.25% (4R fits).
5. **Store per snapshot the per-halo shell counts and smoothing-length histograms**, so the rule can be re-applied without another particle pass. Recompute masks per snapshot: at fixed mass the first kept edge moves inward with z (L2p8, frac95, 13.4-13.6: 0.37 R200c at z = 0, 0.18 at z = 3).
6. **Report the missing-bin share whenever a setting leaves fewer than 6 shells in a calibrated bin.** Otherwise the first-order impacts silently treat that bin as unchanged.
7. **Treat the unresolved core as a modelling uncertainty, not as data.** At z = 0 it lies inside 0.17-0.32 R200c in 13.5-14.1 and inside 0.04-0.15 R200c above that.
   - Its share of the stack's Y(<R200c) is 17% at 13.5-13.7, 15% at 13.7-13.9, 5-7% in 13.9-14.3 and <= 3.0% from 14.3 upward. The 2R fits conserve Y(<R200c) to <= 1.2%.
   - For kSZ and DM, the main open question is the n_e excess there. Integrated over the rejected shells it is hot gas above the B16 form (hot share 0.60-0.93). In the central 0-0.025 R200c sphere, cold / star-forming gas supplies most of it up to 10^14.5 and about half at 14.5-14.7. A higher-resolution comparison (L1_m8) is the way to settle it.

### 11. Web variants (`web_variants.json`)

Contents (41 variants):
- **ref:** P, E, G, T_e, K, identical to the reference records of the other groups.
- **31 setting variants:** min_count 0/5/10/32/50/100/200/500 (8); count-only 0/5/10/20/50/100/200/500 (8); softening 0/2/3/6 (4); h 0/1/3 (3); min_edge 0/0.1 (2); and the rules frac95, frac84, diagnostic, per_halo, per_halo_f50, none (6).
  - Mask-only settings carry the reference values with their own `resolved` flags (P, E).
  - The two per-halo settings carry their own stacks (P, E, G, T_e, K).
- **9 'fit4_*' variants:** the B12/B16 4R fits for ref, 0, 100, 500, h 1, h 3, frac95, per_halo and none (P, E).
  - Bins without a fit are present as empty records (null values, resolved = 0), so every variant has all 15 bins.

### 12. Corrections relative to the version the final verifier checked

**Text**
- h factor 0 = 1: the reason is now that at factor 1 the h edge lies 3-5 shells inside the count >= 20 edge in 13.5-15.1, so the count test sets the edge in both cases. The old reason, "the count test binds at the same shells as x >= 1 h", was wrong. (The verifier's suggested wording said "1-4 shells"; the recomputation from the cache gives 3-5, in `section_checks_final.json`, `h1_vs_count20`.)
- 12.5-12.7: both shells that pass the h test (from 3.30 R200c) fail the count test, in two different haloes (19 and 16 particles). Previously only the outermost shell was given.
- Recommendation on safety floors: min_edge 0.04 is no longer called inert. It binds in 15.1-15.5, and removing it changes tSZ(1000) by -0.25 +- 0.09%.
- Per-halo bands:
  - The bands >= 0.25 R200c are band medians equal to 1. The partially resolved shells inside them (13.5-13.7: 69% / 94% resolved, ratios 1.127 / 1.026) are now given.
  - The exploratory-bin range is now stated (1.00 to 2.98 for P and to 6.77 for n_e).
  - The 0.1-0.25 entries are labelled as medians, and the largest single-shell ratios inside that band are given (P 4.65 / 1.90 / 1.34 at 13.5-13.7 / 13.7-13.9 / 13.9-14.1).
- Per-halo < 0.1 R200c band: all calibrated bins are now listed (P up to 5.87 +- 2.11 and n_e up to 6.65 +- 2.63 at 13.9-14.1), together with the 0.1-0.25 band. Previously the list was selective. The 2R per-halo impacts are added.
- AGN:
  - The old text said removal "explains at most 8%" of the low pressure. Instead, keeping the gas raises P by 8% at x = 0.027 in 13.5-13.7, which closes about 10% of the 45% deficit.
  - The text no longer says the gas "matters only in the central sphere": outside it the gas raises P by up to 8%.
  - The central-sphere effect is now given for every bin: 73 / 59 / 28 / 19 / 10% and <= 4% from 14.5. In 13.5-13.7 the stack there is 1.40 times the extrapolation (stored ratio 1.4049; the verifier wrote 1.41).
- The pressure deficit is now limited to where it holds: 0.025-0.105 R200c in 13.5-13.7. The bins 13.7-13.9 and 14.1-14.3, where P lies above the extrapolation further in, are added.
- n_e phases:
  - Cold/SF gas dominates n_e in the central sphere up to 14.3. It supplies most of the central-sphere excess up to 14.5 (about half at 14.5-14.7).
  - The excess is mostly hot only when integrated over volume (0.60 / 0.76 / 0.82-0.93). The lower end 0.82 is 0.8246 at 14.5-14.7; the verifier's "0.83" is a rounding slip.
  - Hot n_e / extrapolation ranges are now given per mass range.
- Shot noise: at 14.9-15.1, x = 0.052, the median halo passes the h test (x_in / h = 2.97) and 99.2% of the weight is resolved. The old text placed it inside the 2h kernel support.
- frac95 bound: <= 0.48% (was 0.47%). The "2 of 23" subset result depends on the seed (0-3 of 23 over six seeds; strict 10-11).
- L2p8: h is among the binding criteria in 57 of 59 cases (frac95) or 56 of 59 (strict), not "at every z <= 4". The bins are subsampled to <= 200 haloes, and the softening test there includes the 0.04 floor.
- Fit grid: the description now lists the central sphere, the 0.077-dex 0.837-1 shell and every edge beyond 1 R200c.
- The bottom-line claim "the masking rule matters more than the threshold" is now limited to per-halo masks and unmasked cores against thresholds of 0-200, with the largest changes quoted. For tSZ(3000) the unmasked 4R fit and 200 particles are about the same size.
- >= 500 particles: the fit-grid mask is not contiguous in 13.7-14.1 (now in section 3). The +21.95% Y(<R200c) at 13.7-13.9 comes from a fit to 6 scattered shells.
- h factor 1 and per_halo_f50: the 2R-fit impacts are added next to the 4R ones (h factor 1: <= 0.50%; per_halo_f50: DM variance +1.91%).
- Table rounding: innermost medians 122.5 / 147.5 / 164.5 are given exactly (were 122 / 148 / 165).
- Rounding: DM variance per-halo +9.35% (was "+9.4%"); diagnostic DM variance +2.25% (was "+2.3%").
- ">= 500 numbers are lower bounds" was replaced by "incomplete" (the missing bin's sign is unknown).
- Ranges that held for one subset are now labelled: the count-only 10-particle range (calibrated bins), shot-noise counts per bin, band ratios as band medians with their errors, ln P0 shifts with their errors (each < 2 sigma).
- Grid dependence is now quantified: <= 0.045 dex in the calibrated bins.
- The z = 3 softening limit for 13.5-13.7 is 0.091, not 0.105 (0.105 is the next grid edge).
- Web variants: 31 setting variants, not 30.
- The painting-error table gains an h factor 3 row. The comparison is now stated against the reference's own error.

**Figures (redrawn with `regen_caption_figs.py`)**
- Fig4 title: P lies below the extrapolation at 0.025-0.056 R200c in every bin shown (to 0.105 R200c in 13.5-13.7). It lies 2-12% above it at 0.105-0.32 R200c in 13.5-13.7 and 0.056-0.17 R200c in 13.9-14.1, and at 1.40 times the extrapolation in the central sphere of 13.5-13.7 (below it there in the other bins shown). The previous title put "up to 12% above" at 0.056-0.32 R200c for 13.5-14.1 as a whole, which is wrong at 0.056-0.105 R200c in 13.5-13.7.
- Fig9a: the dashed lines are labelled max(0.04, 4 eps/R200c) (self-similar L1 estimate).
- Fig1b: the note now says that every curve with first edge 0 is drawn on the centre line (0 everywhere; 5 from 14.7, 10 from 14.9, 20 and 50 from 15.1).

**Earlier revision (kept)**
- Shell shifts: 100 / 200 / 500 particles move the edge 0-3 / 1-5 / 4-14 shells; h factor 1 moves it 1-2 shells inward; h factor 3 moves it 2-4 shells outward.
- Particle minima: 30-61 in 13.1-15.1; >= 32 only in 12.9-13.1 and 13.3-15.5.
- The 0-20 equivalence holds only in the calibrated bins; 32 was added as an actual setting.
- Count-only 5 particles keeps 22-35 shells (calibrated bins).
- Impact errors reach 0.72%.
- The >= 500 impacts carry missing shares, and 2R fits are missing at 13.5-13.9.
- 4 eps/R200c is 0.0081-0.074 (0.0081-0.034 calibrated).
- The "no gas" column uses true zero-particle fractions.
- The rejected-shell table columns are stored in `inner_decomposition`.
- Fig7 flags the missing bin. Fig1 and fig8 include the 32-particle setting.
