## Where does the halo end? Truncation radii from the bound gas (Test L1)

*FLAMINGO L1_m9, z = 0 (snapshot 0077).*
- 14 calibrated 0.2-dex M200c bins, 12.7-15.5. Test J's stack-level 400-particle rule applies in 12.7-13.5.
- Recently AGN-heated gas is removed. The geometric shell estimator is used on the fine X200 grid (67 shells out to 4 R200c).
- Error ranges are 16-84% of 200 paired, stratified halo bootstraps (stackcore seeds 20261012 + 1000 x bin index).
- Paths are relative to `L_halo_truncation/`. Every number comes from the file and field given in brackets.
- `results/report_numbers.json` registers 6,901 numbers with their source file and field. Gate G8 re-resolves each entry from its source. G8 checks the registry against the result files only; it does not read this text.
- The numeric tables of this report (L1.0 global radius and shares beyond x_t, L1.1, L1.2 with the global-versus-per-bin table, L1.4 shell shares and map ratios) are pasted verbatim from `results/tables/L1_report_tables.json`. `l_numbers.py` formats them from the JSON with 3 decimals. Gate G12 re-formats them with independent code and checks that each of the 55 rows occurs in this text [`verification.json`: `gates.G12_report_tables`].
- Estimates are labelled ESTIMATE.
- Notation (full definitions in L1.5): x = r/R200c. P = P_e/P200 with P200 = G M200c 200 rho_c f_b / (2 R200c); E = n_e mu_XGP / (rho_c f_b). Both are dimensionless.
- Cosmology: FLAMINGO D3A, h = 0.681, Omega_m = 0.306 (neutrinos included), f_b = 0.158823. R200m and R500c are each halo's own SOAP radii.

**What the radius is for (user clarification, 2026-10-16).**
- R_t only decides which data constrain the halo-profile fit: all gas inside R_t, as in observations.
- HalfDome still paints the fitted analytic law out to 4 R200c, extrapolated beyond R_t. Nothing here proposes painting only inside R_t.
- A "share inside R_t" below is what a hard cut would keep; no hard cut is planned. One minus that share is the part of the 4R-painted signal that comes from the extrapolation.

**How the request is interpreted. Both readings are kept.**
- **PRIMARY, B32:** in each bin and for each field, the per-shell bound fraction of the stacked profile falls to 0.32.
  - The fields are electron pressure P (tSZ) and free electrons E (kSZ, FRB DM).
  - The fraction is f(x) = stacked host-bound gas / stacked gas. Beyond that radius more than 68% ("one sigma") of the gas is not bound to the halo.
  - Host-bound means HBT-bound to the central or to its own satellites. The variant B32cb counts central-bound gas only.
  - Inside R_t the fit uses all gas.
- **VARIANT, C68:** the radius that encloses 68% of the bound gas inside 4 R200c.

### L1.0 Short answer

1. **B32 radius per bin** [`results/radii.json`: `bins.<bin>.rules.B32.{P,E}`; `results/tables/L1_report_bins.csv`]:
   - Over 10^12.9-10^15.3 it is nearly mass-independent: 1.35-1.45 R200c for pressure and 1.37-1.48 R200c for electrons.
   - In 12.7-12.9 pressure gives 1.25 and electrons 1.46.
   - The 6-halo top bin 15.3-15.5 gives P 1.70 [1.69, 2.71] and E 1.71 [1.45, 2.45]. There, host-bound satellites keep f near 0.32 out to about 1.7 R200c. Central-bound only gives 1.33 (P) and 1.34 (E).
   - Bootstrap half-widths below 10^15.3 are 0.006-0.042 R200c, median 0.011 [`results/global.json`: `B32_bootstrap_halfwidths`]. The widest bin is 15.1-15.3: E 1.462 [1.420, 1.504] (±0.042) and P ±0.030.
   - In the sample, every bin crosses 0.32 exactly once.
2. **Nesting does not matter.** The partition block has no owner axis, so the cache cannot form "host-bound AND A2-ii".
   - Primary: host-bound / all gas. It is nested, because host-bound gas is a subset of all gas.
   - Also computed: host-bound / A2-ii (an upper bound) and the exact per-halo-shell bracket.
   - All of them give the same B32 to within 0.0017 R200c in 13 bins and 0.0053 in the top bin [radii.json: B32, B32u, B32lo, B32hi].
   - The reason: inside 1.5 R200c, A2-ii gas equals all gas to within 0.061% (P) and 0.12% (E) per stacked shell [global.json: `a2ii_vs_all_inside_1.5R`]. The fit data are therefore all gas, which is the literature convention.
3. **Global fit radius x_t = 1.43 R200c, one value for both fields.** This is the signal-weighted mean of R_t(M), with weights equal to Test J's D0 importance shares [global.json: `global.B32.{P,E}.{R,R200m,R500c}.family`, `global.B32.common`]:

   | | R200c | R200m | R500c |
   |---|---|---|---|
   | P | 1.429 [1.424, 1.491] | 0.856 | 2.212 |
   | E | 1.435 [1.432, 1.439] | 0.856 | 2.213 |
   | common | 1.432 | 0.856 | 2.212 |

   Four alternative weightings or objectives built on the same per-bin B32 radii give the same value. They are not independent checks: all four re-weight or re-score the same 14 radii with Test J's same D0 curves. Their agreement shows that the weighting barely matters because R_t(M) is flat (CV 5.1% (P) / 2.2% (E)):
   - Truncated-target weights give 1.432 (P) and 1.428 (E) [`truncated_target_weights.B32.*.R_weighted`].
   - Single-observable weights span 1.419-1.452 (P 1.421-1.451, E 1.419-1.452). The extremes are y x DM (E side) at ell = 1000 and the median DM [`global_sensitivity.B32.both_fields_single_observable`].
   - Leaving one bin out gives 1.413-1.442 (P) and 1.431-1.438 (E) [`global_sensitivity.B32.*.leave_one_bin_out`].
   - The **observable-optimal single cut** on the exact additive terms is 1.431 for P (RMS 0.17%), 1.435 for E (0.57%) and 1.435 for both (0.46%) [`observable_optimal.B32.{P,E,common}.x_opt / rms_at_opt`]. At the signal-weighted radius the RMS is 0.18% / 0.57% / 0.48%. The largest single-term deviation is 0.96% at the common optimum and 1.19% (mean DM) at the common x_t = 1.432 [`observable_optimal.B32.common.max_abs_dev_at_opt`, `.dev_at_signal_weighted.dm_mean`].
4. **What the fit radius leaves to the extrapolation** (new, L1.2):
   - **If the painted law beyond x_t followed all A2-ii gas** (D0), these are the parts of the 4R-painted 1-halo/linear terms that come from beyond x_t [`results/extrap.json`: `observable_level.D0_share_beyond_xt`]:

     | term | share from beyond x_t |
     |---|---|
     | mean y | 0.288 |
     | tSZ 1-halo, ell = 100 / 1000 / 3000 | 0.113 / 0.061 / 0.017 |
     | mean DM | 0.615 |
     | DM variance | 0.278 |
     | kSZ 1-halo, ell = 100 / 1000 / 3000 | 0.569 / 0.482 / 0.315 |
     | DM-field C_ell 1-halo, ell = 1000 | 0.357 |

     The tSZ 1-halo shares carry no fallback-halo term, because Test J stores no `trunc_all__D0__yy1h`. The fallback haloes (the 9 lightcone haloes above 10^15.388 Msun, painted with the published laws) hold 0.12-0.63% / 0.60-1.75% / 0.28-0.69% of the 1-halo C_ell^yy at ell = 100 / 1000 / 3000. With that term the shares beyond x_t become 0.1125-0.1131 / 0.0597-0.0604 / 0.0167-0.0168 (ESTIMATE) [`observable_level.yy1h_fallback_ESTIMATE`]. Rounded to 3 decimals only ell = 1000 moves, from 0.061 to 0.060. The other terms include the fallback exactly.

   - **If only bound gas were painted beyond x_t**, the mean-DM share from beyond x_t is 0.117. That painting totals 0.437 x D0 = 1.226 x D3 [`observable_level.mean_dm_bracket.global`].
   - **The published B12/B16 maps** (Codex, ESTIMATE) take 0.060 (tSZ), 0.405 (kSZ) and 0.284 (DM) of their 4R power near ell ≈ 1025 from beyond x_t. With FLAMINGO A2-ii gas in our engine the shares are 0.063 / 0.512 / 0.378 [`maps_share_beyond_xt`].
   - The tSZ power spectrum is therefore mostly set by the data (0.02-0.11 from beyond x_t); the mean y less so (0.29).
   - For kSZ and DM, 0.28-0.61 of the painted 1-halo/linear terms (all-gas reference) comes from beyond x_t. The data constrain that part only through the law's outer-shape parameters.
5. **Recommendation:**
   - Fit the halo profile to all gas inside x_t = 1.43 R200c, for both fields, and paint the fitted law to 4 R200c. This is B32 with one global radius; the per-bin B32 radii are the variant.
   - Use B50 (1.20 / 1.24 R200c) and Req (1.20 / 1.33) as shorter fit ranges, to test how stable the extrapolation is.
   - Drop C68.
   - Score every fitted law against the bracket [bound gas, all A2-ii gas] beyond x_t (L1.2).

### L1.1 Truncation radius per mass bin

All radii are in R200c with bootstrap 16-84% ranges; N is the number of haloes [radii.json; tables/L1_report_bins.csv; stacks_summary.json `bins.<bin>.N`]. The last column is the share of the A2-ii stack's Y(<4R) (P) and N_e(<4R) (E) that lies inside each bin's B32 radius [radii.json `rules.B32.<f>.inside_a2ii`].

| bin | N | B32 P | B32 E | B32cb P / E | B50 P / E | C68 P / E | inside B32, P / E |
|---|---|---|---|---|---|---|---|
| 12.7-12.9 | 332 | 1.251 [1.235, 1.275] | 1.463 [1.451, 1.475] | 1.245 / 1.417 | 0.979 / 1.241 | 1.046 / 1.261 | 0.238 / 0.210 |
| 12.9-13.1 | 322 | 1.373 [1.360, 1.383] | 1.465 [1.456, 1.474] | 1.370 / 1.441 | 1.139 / 1.287 | 1.039 / 1.246 | 0.330 / 0.225 |
| 13.1-13.3 | 317 | 1.421 [1.412, 1.432] | 1.482 [1.476, 1.490] | 1.411 / 1.458 | 1.238 / 1.304 | 1.028 / 1.226 | 0.415 / 0.265 |
| 13.3-13.5 | 308 | 1.432 [1.423, 1.443] | 1.463 [1.456, 1.471] | 1.428 / 1.450 | 1.256 / 1.289 | 0.996 / 1.173 | 0.502 / 0.319 |
| 13.5-13.7 | 326 | 1.445 [1.433, 1.455] | 1.446 [1.438, 1.454] | 1.433 / 1.431 | 1.267 / 1.274 | 0.934 / 1.113 | 0.608 / 0.386 |
| 13.7-13.9 | 324 | 1.443 [1.433, 1.456] | 1.434 [1.426, 1.443] | 1.424 / 1.417 | 1.256 / 1.256 | 0.873 / 1.048 | 0.676 / 0.445 |
| 13.9-14.1 | 319 | 1.455 [1.442, 1.466] | 1.428 [1.419, 1.437] | 1.415 / 1.401 | 1.254 / 1.243 | 0.808 / 0.977 | 0.739 / 0.500 |
| 14.1-14.3 | 327 | 1.443 [1.431, 1.456] | 1.419 [1.410, 1.428] | 1.387 / 1.380 | 1.236 / 1.226 | 0.776 / 0.941 | 0.762 / 0.526 |
| 14.3-14.5 | 317 | 1.427 [1.417, 1.436] | 1.419 [1.411, 1.425] | 1.385 / 1.381 | 1.225 / 1.219 | 0.752 / 0.922 | 0.774 / 0.545 |
| 14.5-14.7 | 895 | 1.398 [1.391, 1.405] | 1.398 [1.392, 1.404] | 1.357 / 1.354 | 1.191 / 1.189 | 0.707 / 0.879 | 0.791 / 0.562 |
| 14.7-14.9 | 326 | 1.407 [1.393, 1.420] | 1.405 [1.393, 1.415] | 1.339 / 1.342 | 1.181 / 1.178 | 0.692 / 0.862 | 0.800 / 0.578 |
| 14.9-15.1 | 127 | 1.352 [1.336, 1.381] | 1.369 [1.351, 1.391] | 1.327 / 1.326 | 1.138 / 1.143 | 0.663 / 0.827 | 0.817 / 0.588 |
| 15.1-15.3 | 21 | 1.426 [1.398, 1.457] | 1.462 [1.420, 1.504] | 1.404 / 1.411 | 1.192 / 1.189 | 0.679 / 0.838 | 0.847 / 0.627 |
| 15.3-15.5 | 6 | 1.695 [1.690, 2.712] | 1.706 [1.453, 2.451] | 1.335 / 1.342 | 1.124 / 1.131 | 0.631 / 0.921 | 0.868 / 0.644 |

**Units.** In 12.9-15.3 the B32 radius is [radii.json `rules.B32.<f>.R200m/R500c`]:
- P: 0.818-0.868 R200m and 2.08-2.25 R500c;
- E: 0.828-0.884 R200m and 2.12-2.27 R500c.

Neither alternative unit is clearly more mass-independent. The coefficient of variation is P 5.1 / 4.8 / 5.0% and E 2.2 / 2.0 / 1.8% in R200c / R200m / R500c [global.json `global.B32.<f>.<unit>.family.cv`]. R200c stays the unit.

**Non-monotonic cases.** Only two bins cross 0.32 more than once in the bootstrap [`non_monotonic_fraction_boot`]:
- 14.9-15.1, P: 4% of replicates;
- 15.3-15.5: P 38.5%, E 24.5%.

**Bound shares by band** (host-bound / A2-ii, bands 0-1 / 1-2 / 2-3 / 3-4 R200c) [global.json `band_bound_shares`]:

| field | 0-1 | 1-2 | 2-3 | 3-4 |
|---|---|---|---|---|
| E | 0.84-0.95 | 0.31-0.39 | 0.04-0.15 (top bin 0.27) | 0.010-0.070 (top bin 0.095) |
| P | 0.69-0.96 | 0.20-0.41 | 0.02-0.20 (top bin 0.28) | 0.007-0.20 |

- In 12.7-13.1, 25-31% of the pressure inside R200c is not host-bound (host-bound shares 0.694 and 0.748). That gas is still fitted, because the fit uses all gas.
- Above 10^14.3, the per-shell host-bound pressure fraction between 2 and 3 R200c is 0.11-0.30, while the central-bound fraction there is only 0.02-0.07 [`per_shell_fraction_ranges`].
- That bound gas is therefore mostly bound to satellites. This is why f stays below 0.32 there, though not far below it.

### L1.2 The region the painting extrapolates into (new, after the clarification)

The data inside x_t = 1.4320 R200c fix the fitted law; the region from x_t to 4 R200c is painted by extrapolation. Inside x_t every reference equals the measured all-gas stack. Beyond x_t there are three references:
- **upper:** the painted A2-ii gas (Test J's D0);
- **lower:** host-bound gas only (Test J's D3);
- **published:** the B12/B16 law that HalfDome paints now. Only its shape enters: it is rescaled to the same content inside x_t, as a law fitted inside x_t would be.

The table below is at stack level, for the global x_t [`results/extrap.json`: `bins.<bin>.<f>.global`; `tables/L1_extrapolation.csv`]:
- s = the share of the 4R content that lies between x_t and 4 R200c when the inside is matched;
- t = the position of the published shape between bound gas (0) and A2-ii gas (1).

| bin | P: A2-ii | P: all gas | P: bound | P: B12 shape | t_B12 | E: A2-ii | E: all gas | E: bound | E: B16 shape | t_B16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 12.7-12.9 | 0.711 | 0.770 | 0.076 | 0.352 | 0.194 | 0.797 | 0.813 | 0.175 | 0.532 | 0.250 |
| 13.1-13.3 | 0.581 | 0.611 | 0.089 | 0.321 | 0.290 | 0.750 | 0.760 | 0.163 | 0.509 | 0.299 |
| 13.5-13.7 | 0.397 | 0.441 | 0.071 | 0.289 | 0.570 | 0.619 | 0.639 | 0.123 | 0.484 | 0.535 |
| 13.9-14.1 | 0.268 | 0.302 | 0.052 | 0.259 | 0.948 | 0.499 | 0.527 | 0.091 | 0.458 | 0.832 |
| 14.5-14.7 | 0.202 | 0.230 | 0.046 | 0.215 | 1.101 | 0.428 | 0.468 | 0.089 | 0.418 | 0.951 |
| 14.9-15.1 | 0.168 | 0.188 | 0.037 | 0.187 | 1.169 | 0.396 | 0.442 | 0.084 | 0.390 | 0.969 |
| 15.1-15.3 | 0.152 | 0.177 | 0.028 | 0.175 | 1.215 | 0.381 | 0.433 | 0.083 | 0.376 | 0.974 |
| family-weighted | 0.240 | – | 0.050 | 0.224 | – | 0.551 | – | 0.112 | 0.460 | – |

All 14 bins are in the CSV. The family-weighted row uses the D0 family weights [`family_weighted.<f>.global`]; its 16-84% ranges are ±0.003 or narrower.

**What it shows.**
- **Electrons.** The B16 shape lies inside the bracket [bound, A2-ii] in all 14 bins (t = 0.25-0.97).
  - It sits near the bound end below 10^13.5 (t = 0.25-0.39) and near A2-ii from 10^13.9 upwards (t = 0.83-0.97; 0.70 in the top bin) [`family_weighted.E.global.t_pub_range*`; `bins.<bin>.E.global.t_pub`].
  - The extrapolated region holds 8-18% of the lower reference (all gas inside x_t, bound gas beyond), but 38-80% of the A2-ii content [`s_hb_range`, `s_a2ii_range`].
- **Pressure.** The B12 shape lies inside [bound, A2-ii] in 8 of 14 bins.
  - In 14.1-15.3 it lies slightly above the A2-ii gas (t = 1.02-1.21), but below all gas. All 14 bins lie inside [bound, all gas] [`n_bins_pub_in_bracket`, `n_bins_pub_in_bracket_all`].
  - Below 10^13.5 the FLAMINGO pressure is much more extended than B12: 50-71% of the A2-ii Y(<4R) lies beyond x_t, against 30-35% for the B12 shape.
- **Normalisation inside x_t** (not a shape quantity) [`n_pub`]:
  - From 10^13.9 upwards, the published laws hold 0.89-0.93 (B12) and 0.92-0.96 (B16) of FLAMINGO's content inside x_t.
  - In 12.7-12.9 they hold 3.2 (B12) and 3.0 (B16) times FLAMINGO's content.

**Mean DM, exact** [`observable_level.mean_dm_bracket.global`]:
- The bracket runs from a lower reference (all gas inside x_t plus bound gas beyond it) to D0 (upper).
- Lower reference = 0.437 x D0 = 1.226 x D3, with D3/D0 = 0.356.
- The share from beyond x_t is 0.615 for the upper reference and 0.117 for the lower one.
- With the per-bin B32 radii instead: lower reference 0.440 x D0, shares 0.610 / 0.113 [`.perbin`].
- Quadratic terms (variance, C_ell) of this composite profile need the exact painting (L2/L3).

**Global versus per-bin fit radius.** The data-constrained part barely changes. The table gives the share of each 4R-painted term that the data inside the radius fix, i.e. one minus the share from beyond it [global.json `additive_terms.<O>.B32.{T_global,T_perbin}`; middle column: 1 - extrap.json `observable_level.D0_share_beyond_xt`]:
- "field radius" applies the field's own family-weighted radius to every bin: 1.4292 R200c for the P terms and 1.4348 R200c for the E terms.
- "common x_t" applies x_t = 1.4320 R200c, as everywhere else in L1.2. It is the complement of the L1.0 item 4 table, e.g. kSZ 1-halo at ell = 100: 1 - 0.569 = 0.431.
- The tSZ term carries no fallback-halo term (L1.0 item 4).

| term | field radius (P 1.4292 / E 1.4348) | common x_t = 1.4320 | per-bin B32 radii |
|---|---|---|---|
| mean y | 0.711 | 0.712 | 0.711 |
| tSZ 1-halo, ell = 1000 | 0.939 | 0.939 | 0.938 |
| kSZ 1-halo, ell = 100 | 0.432 | 0.431 | 0.429 |
| mean DM | 0.386 | 0.385 | 0.390 |

The all-gas content inside the per-bin B32 radii already exceeds the whole bound component out to 4R [`perbin_over_D3`]:
- mean DM: 1.095 x D3;
- DM variance: 1.143 x D3;
- kSZ 1-halo: 1.192 / 1.183 / 1.179 x D3 at ell = 100 / 1000 / 3000.

A law that also extrapolates all gas therefore paints far more than the bound halo for kSZ and DM. L3 has to decide this against the bracket.

### L1.3 Alternative rules

All values are in R200c [radii.json; global.json `global.<rule>`]. Literature rules:
- MDK15 is More, Diemer & Kravtsov (2015, ApJ 810, 36) Eq. 7.
- CLg is a gas-only proxy for the closure radius of Ayromlou, Nelson & Pillepich (2023, MNRAS 524, 5391).
- SH_K0 is the entropy maximum, which we use as the proxy for the accretion shock. Using the entropy maximum as the locator is our choice, not a definition taken from the papers below.
  - Lau, Nagai, Avestruz, Nelson & Vikhlinin (2015, ApJ 806, 68) find that the shock location depends on the mass accretion rate.
  - Shi (2016, MNRAS 461, 1804) finds that the shock tracks the dark-matter splashback radius only for gas with adiabatic index near 5/3 and moderate accretion rates.

| rule | per bin | family-weighted (P / E) | verdict |
|---|---|---|---|
| B50 (bound share 0.50) | 0.98-1.30 | 1.200 / 1.243 | inner variant of the fit range |
| B32cb (central-bound) | 1.25-1.46 | 1.372 / 1.404 | 0.003-0.068 below B32 in 12.7-15.3 (0.36 below in the top bin) |
| C68 (68% of the bound gas inside R) | P 1.05 → 0.63, E 1.26 → 0.83 | 0.750 / 1.025 (CV 13.2 / 12.4%) | strongly mass-dependent; too small to constrain the outer slope; dropped |
| C90 (bound-component convergence) | E 1.41-2.07 | 1.174 / 1.572 | within 0.012 of D3's R90 of the mean DM (1.40-2.06) [`d3_convergence.own_bin`] |
| Req (A2-ii gas inside R = bound gas inside 4R) | P 1.05-1.26, E 1.25-1.39 | 1.202 / 1.328 | budget-matched inner fit range |
| SB (steepest log slope of all gas) | E, well defined in 13.3-15.3: 2.245 → 1.279 (1.33 → 0.77 R200m) | 1.644 / 1.866 (includes the shallow low-mass bins) | close to MDK15 below 10^13.7 (SB/MDK15 = 1.06, 1.005); 0.76-0.81 x MDK15 in 14.5-15.3 |
| SH_TM / SH_K (steepest drop of T_M or entropy K) | interior minima found but all shallow (depth < 0.3); entropy minima interior only below 10^13.3 | – | no shock signature |
| SH_K0 (entropy maximum) | 2.31-3.09, only at ≥ 10^14.5 | – | well outside the bound halo |
| 2H2 (all gas = 2 x linear field) | E 1.66-2.72; P 0.46-3.30 (ESTIMATE) | 2.416 / 2.298 | the field dominates from about 2 R200c, beyond B32 |
| MDK15 (dark-matter splashback, reference) | 2.19 → 1.62 | 1.850 / 1.985 | (ESTIMATE) |
| CLg (gas-only closure proxy) | 3.47 → 1.27 | 1.488 / 1.782 | (ESTIMATE) |

- The bound-fraction rules are all nearly mass-independent. Their CV (P / E) is 5.1% / 2.2% for B32, 2.5% / 2.6% for B32cb and 3.7% / 3.3% for B50, against 13.2% / 12.4% for C68 [`global.<rule>.<f>.R.family.cv`].
- B32 lies inside the dark-matter splashback estimate MDK15 in every bin below 10^15.3. In the top bin it is 1.70 against 1.62.
- SB, 2H2 and MDK15 mark where the profile steepens or meets the field, not where the bound gas ends. They serve as upper limits on a fit range.

### L1.4 Codex analyses

**Shell sensitivities.** The table gives the share of C_ell from the shells 0-1 / 1-1.5 / 1.5-2 / 2-3 / 3-4 R200c [codex.json `halfdome_maps.<f>.77.common_shell_fractions`, `engine.<f>.100.band_importance_shares`]. It compares Codex's HalfDome B12/B16 maps at ell ≈ 77 with our engine (FLAMINGO A2-ii stacks on the same lightcone) at ell = 100.

| | HalfDome B12/B16 (ell ≈ 77) | engine, FLAMINGO A2-ii (ell = 100) |
|---|---|---|
| tSZ | 0.802 / 0.118 / 0.045 / 0.027 / 0.008 | 0.821 / 0.109 / 0.037 / 0.024 / 0.009 |
| kSZ | 0.483 / 0.178 / 0.119 / 0.141 / 0.079 | 0.362 / 0.178 / 0.133 / 0.189 / 0.138 |
| DM | 0.524 / 0.172 / 0.110 / 0.125 / 0.070 | 0.447 / 0.178 / 0.118 / 0.152 / 0.104 |

FLAMINGO's outskirts weigh kSZ and DM more heavily than B16 does.

**FLAMINGO / HalfDome map power near ell = 1025** [codex.json `flamingo_over_halfdome_1025.<f>.<sim>.ratio_4R`]. FLAMINGO's lightcone maps contain all gas; HalfDome is painted with B12/B16 to 4R.

| | L1_m9 | L2p8 |
|---|---|---|
| tSZ | 1.315 | 1.428 |
| kSZ | 2.462 | 2.587 |
| DM | 1.248 | 1.256 |

- These ratios remain the reference comparison. FLAMINGO's maps also hold field gas beyond 4R, low-mass haloes that are not painted, and different inner profiles. The excess therefore says nothing about the fit range.
- A hard cut of HalfDome's maps at x_t would imply ratios of 1.40 / 4.13 / 1.74 (`ratio_B32_ESTIMATE`). **This is not the painting plan** and is kept for reference only.
- Under the plan, the maps change through the normalisation inside x_t (`n_pub`, L1.2) and through the extrapolated shape.

**Particle pilot** (one halo per run, at z ≈ 0.21; the pilot includes AGN-heated gas) [codex.json `pilot.codex.<sim>`]:
- L1_m9 (log M = 15.29): f_hb first crosses 0.32 at 1.20 and last at 2.27 R200c. There are 3 crossings, because a satellite bump reaches 0.36 at 2.18.
- L2p8 (log M = 15.37): first crossing 1.36, last 2.50.
- The central-only crossings are 1.20 / 1.36, close to our pressure B32cb in 15.1-15.5 (1.33-1.40).
- The pilot's cumulative host-bound / all-gas fraction is 0.923 / 0.934 inside 1R and 0.668 / 0.760 inside 4R.
- Ours inside 4R, for 14.9-15.1 / 15.1-15.3 / 15.3-15.5, is 0.735 / 0.760 / 0.722, with per-halo 16-84% ranges 0.685-0.798 / 0.710-0.798 / 0.662-0.789. The L1_m9 pilot halo (host-bound/all pressure inside 4R = 0.668, log M 15.29) lies below the 16-84% range of its own bin 15.1-15.3, at about its 10th percentile (2 of 21 haloes lower), so it is a low-bound-fraction example rather than a typical one [`pilot.ours_pressure`].

**Where the signal is (for the redshift extension).** HalfDome maps at ell ≈ 1025 [codex.json `halfdome_maps.<f>.{redshift,mass}_fractions.1025`]:
- Redshift: z = 0.1-1 carries 0.31 + 0.33 + 0.19 of the tSZ, 0.22 + 0.30 + 0.29 of the kSZ and 0.31 + 0.33 + 0.24 of the DM (z_s = 1).
- Mass: 10^14-10^15 carries 0.22 + 0.47 of the tSZ (10^15-10^15.5: 0.25). 10^13-10^14.5 carries 0.20 + 0.31 + 0.31 of the kSZ and 0.16 + 0.29 + 0.33 of the DM.
- FLAMINGO L1_m9's kSZ map takes 0.91 of its ell ≈ 77 power from z = 2-3 (0.21 at ell ≈ 1025) [`redshift_shares_maps.L1_m9.ksz`]. We suspect an artefact and suggest asking Codex.

### L1.5 Definitions and equations

**Profiles** (dimensionless; $x=r/R_{200c}$):

$$P=\frac{P_e}{P_{200}},\quad P_{200}=\frac{G\,M_{200c}\,200\rho_c f_b}{2R_{200c}},\qquad E=\frac{n_e\,\mu_{\rm XGP}}{\rho_c f_b},\qquad G_{\rm gas}=\frac{\rho_{\rm gas}}{\rho_c f_b}$$

- $P_e$: electron pressure [Pa]; $n_e$: free-electron density [m^-3]; $\rho_{\rm gas}$: gas mass density [kg m^-3].
- $G$: Newton's constant.
- $M_{200c}$ [kg] and $R_{200c}$ [m]: the mass and radius at 200 times the critical density $\rho_c$ (z = 0) [kg m^-3].
- $f_b=\Omega_b/\Omega_m$ = 0.158823.
- $\mu_{\rm XGP}=(m_e+2m_p/1.76)/0.9$: XGPaint's gas mass per electron [kg], with $m_e$ and $m_p$ the electron and proton masses.
- $F_{h,s}$: the volume average of halo $h$ in fine shell $s=[x_s,x_{s+1}]$; $V_s=\tfrac{4\pi}{3}(x_{s+1}^3-x_s^3)$.

**Selections.** Recently AGN-heated gas is removed everywhere.
- all: every gas particle.
- a2ii: the A2-ii painting partition.
- hb: host-bound.
- cb: central-bound.
- a2iia2i: A2-ii and not bound to another painted halo.
- p10 / p11: the A2-ii complement (gas closer, in r/R200c, to a painted neighbour) that is not bound / is bound to another painted halo.
- all = a2ii + p10 + p11 (gate G0).

**Stack and bootstrap.**

$$\bar F_{b,s}^{(r)}=\frac{\sum_{h\in b}W_{rh}F_{h,s}}{\sum_{h\in b}W_{rh}}$$

- $W_{0h}=w_h$: the inclusion weight.
- $W_{rh}=n_{rh}w_h$ for $r=1..200$, where $n_{rh}$ is the multiplicity of halo $h$ in the $r$-th stratified resample. The resamples are paired across all quantities.

**Bound fraction of a group $g$ of fine shells.** Single shells are used at ≥ 10^13.5; in 12.7-13.5, Test J's merged groups of at least 400 stacked bound particles.

$$f_g=\frac{\sum_{s\in g}V_s\bar B_s}{\sum_{s\in g}V_s\bar R_s},\qquad x_g=\frac34\,\frac{x_e^4-x_a^4}{x_e^3-x_a^3}$$

- $\bar B$: the stacked bound quantity (hb or cb).
- $\bar R$: the stacked reference gas (all or a2ii).
- $x_g$: the volume-weighted mean radius of the group $[x_a,x_e]$.

The nested bracket per halo-shell is

$$\max(0,hb-p_{10})\le hb\cap a2ii\le\min(hb,a2iia2i).$$

**Threshold rules B32 / B50** ($\theta$ = 0.32 / 0.50):

$$R_t=\exp\!\Big[\ln x_i+\frac{f_i-\theta}{f_i-f_{i+1}}(\ln x_{i+1}-\ln x_i)\Big]$$

- $i$ is the last point with $f_i\ge\theta>f_{i+1}$, so $f<\theta$ at every later point. The search starts at $x\ge0.3$.
- The first crossing and the number of crossings are also recorded.
- A replicate with $f\ge\theta$ at the last point ($x$ = 3.82) is censored and enters the percentiles as 4.

**Cumulative and budget rules.**

$$X(<R)=\sum_{x_{s+1}\le x_k}V_sF_s+F_k\tfrac{4\pi}{3}(R^3-x_k^3)\quad (x_k\le R<x_{k+1})$$

- C68 / C90 / C95: $X_{hb}(<R)=\ell X_{hb}(<4)$, with $\ell$ = 0.68 / 0.90 / 0.95.
- Req: $X_{a2ii}(<R_{eq})=X_{hb}(<4)$.
- $X$ is in units of $P_{200}R_{200c}^3$ (P) or $\rho_cf_bR_{200c}^3/\mu_{\rm XGP}$ (E).

**Steepest slope SB.**
- At each $u_0$, a local quadratic regression in $u=\ln x$ minimises $\sum_iw_i[\ln\bar F_i-a_0-\eta(u_i-u_0)-a_2(u_i-u_0)^2]^2$ with $w_i=e^{-(u_i-u_0)^2/2\sigma^2}$. Here $\sigma$ = 0.15 (0.10 and 0.25 as checks), and the slope is $\eta=d\ln\bar F/d\ln x$.
- SB is the location of the minimum slope $\eta_{\min}$ in $[\max(0.7,x_h),3.6]$, where $x_h$ is Test J's hydrodynamic-resolution radius.
- SB is flagged **shallow** (ill-defined) when its depth $\delta=\min(\eta_{\rm lo},\eta_{\rm hi})-\eta_{\min}<0.3$, with $\eta_{\rm lo},\eta_{\rm hi}$ the slopes at the two range ends. A minimum therefore counts only if it is at least 0.3 steeper than the steeper of the two ends.
- SH_TM / SH_K apply the same rule in $[\max(1,x_h),3.6]$ to the mass-weighted temperature $T_M$ and to the entropy $K=k_BT_en_e^{-2/3}$. Here $k_B$ is Boltzmann's constant and $T_e$ the electron temperature; the stacks are geometric means in units of $T_{200}$ and $K_{200}$.
- SH_K0 is the first downward crossing of $d\ln K/d\ln x=0$, i.e. the entropy peak.

**Two-halo floor** (2H2 / 2H1: the radius where $F_{\rm all}/F_{2h}$ falls to 2 / 1):

$$E_{2h,h}=\bar E[1+b(M_h)\langle\xi_{\rm lin}\rangle_s],\qquad P_{2h,h}=\frac{\langle P_e\rangle+\langle bP_e\rangle b(M_h)\langle\xi_{\rm lin}\rangle_s}{P_{200,h}}$$

- $\bar E=\kappa_E\Omega_m$ = 0.3348: the cosmic mean in E units, with $\kappa_E$ = 1.094 (Test J's E_COSMIC / $\Omega_m$).
- $b$: the Tinker+10 bias.
- $\langle\xi_{\rm lin}\rangle_s$: the 32-point shell average of CLASS's z = 0 linear correlation function at $xR_{200c,h}$.
- $\langle P_e\rangle=\sum_hw_hY_h/V_{\rm box}$ = 7.01e-18 Pa, and $\langle bP_e\rangle/\langle P_e\rangle$ = 3.03 [stacks_summary.json `pressure_means_ESTIMATE`; ESTIMATE].
- $Y_h=P_{200,h}R_{200c,h}^3\sum_sV_sP^{a2ii}_{h,s}$ [Pa m^3]: the halo's A2-ii pressure volume inside 4R.
- $V_{\rm box}=(1000\,{\rm Mpc})^3$. The sums run over all 4,588 haloes ≥ 10^12.5.

**References** (ESTIMATE, not measured):
- MDK15: $R_{\rm sp}/R_{200m}=0.81(1+0.97e^{-\nu_{200m}/2.44})$, with $\nu_{200m}=\delta_c/\sigma(M_{200m})$ and $M_{200m}=M_{200c}(R_{200m}/R_{200c})^3\Omega_m$. $\delta_c$ is the linear collapse threshold and $\sigma(M)$ the rms linear density fluctuation in a sphere of mass $M$ (G's CLASS table).
- CLg: $\sum_{x<R}V_sG_{{\rm gas},s}=800\pi/3$, i.e. the gas inside R equals $f_bM_{200c}$.

**Units per row.**
- $R^{(200m)}=R\langle R_{200c}/R_{200m}\rangle_W$ and $R^{(500c)}=R\langle R_{200c}/R_{500c}\rangle_W$, from each halo's own SOAP radii.
- Cosmology: FLAMINGO D3A, h = 0.681, $\Omega_m$ = 0.306 (neutrinos included), $f_b$ = 0.158823, box 1000 Mpc.

**Signal-weighted radius.**

$$\bar R=\sum_bw_bR_{t,b},\qquad {\rm CV}=\sqrt{\textstyle\sum_bw_b(R_{t,b}-\bar R)^2}/\bar R$$

- $w_b$: Test J's D0 importance share of bin $b$, renormalised over the 14 bins and averaged over each field's family of observables.
- P family: tSZ $C_\ell^{yy}$ at ell = 100 / 1000 / 3000, mean y, and y x DM (P side) at ell = 1000.
- E family: kSZ 1-halo at ell = 1000, kSZ total at ell = 100 / 1000 / 3000, DM mean / variance / median, DM-field $C_\ell$ at ell = 1000, and y x DM (E side).
- "Common" is the mean of the P and E values.

**Additive terms** (exact at the fine edges; linear in $R^3$ between them):

$$T_O(R_b)=\frac{O_{\rm fb}+\sum_bo_b(R_b)}{O_{\rm fb}+\sum_bo_b(4)}$$

- $o_b(R)$: Test J's per-bin "own" truncation curve, i.e. bin $b$'s 1-halo or linear term with its haloes cut at $R$ (C14, HalfDome lightcone).
- $O_{\rm fb}$: the fallback haloes' term (the 9 lightcone haloes above $10^{15.388}\,M_\odot$, painted with the published laws and never cut), from Test J's `trunc_all__D0__<O>` at $R=0$. For $C_\ell^{yy}$ 1-halo J stores no such curve, so $O_{\rm fb}=0$ there and $T$ is a share of the calibrated haloes' term only.
- Fallback bracket for $C_\ell^{yy}$ 1-halo (ESTIMATE): with the fallback, the share beyond $x_t$ is $(1-T)(1-\phi)$, where $\phi$ is the fallback's share of the 4R term. Its lower bound $\phi_{\rm lo}$ is Test H's importance share of the $\ge10^{15.5}$ row (H `results/sens_summary.json`: `shares['a2ii__yy1h@<ell>']['>=15.5']`); haloes in $10^{15.388}$-$10^{15.5}$ sit in H's 15.3-15.5 row, so this bound is low. Its upper bound is $\phi_{\rm hi}=Y_0/(Y_0+\sum_bo_b(4))$, where $Y_0$ is J's `trunc_all__D0__yytot` at $R=0$: the total (1-halo plus 2-halo) $C_\ell^{yy}$ of the fallback haloes alone, which bounds their 1-halo term from above.
- Observable-optimal cut: $x_{\rm opt}=\arg\min_x\sqrt{N_F^{-1}\sum_{O\in F}[T_O(x)/T_O(R_{t,b})-1]^2}$, searched on a grid with step 1e-4 over 1-2.5 R200c. $N_F$ is the number of terms in family $F$: 4 (P), 6 (E) or 10 (common).

**Extrapolated region** (stack level):

$$\rho_q=\frac{X_q(<4)-X_q(<x_t)}{X_{a2ii}(<x_t)},\quad s_q=\frac{\rho_q}{1+\rho_q},\quad \rho_{\rm pub}=\frac{X_{\rm pub}(<4)-X_{\rm pub}(<x_t)}{X_{\rm pub}(<x_t)},\quad t=\frac{\rho_{\rm pub}-\rho_{hb}}{\rho_{a2ii}-\rho_{hb}},\quad n_{\rm pub}=\frac{X_{\rm pub}(<x_t)}{X_{a2ii}(<x_t)}$$

- $q$ is a2ii (upper), hb (lower) or all. $s_q$ is a share, not a shell index.
- The published shell averages are exact: $F^{\rm pub}_{h,s}=3[C_h(x_{s+1})-C_h(x_s)]/(x_{s+1}^3-x_s^3)$, with $C_h(x)=\int_0^xF^{\rm pub}x'^2dx'$ in closed form (Test K's `k_fit_core.cumulative`).
- B12 at z = 0: $P^{\rm pub}=0.5176P_0q^{-0.3}(1+q)^{-\beta}$ with $q=x/x_c$, $P_0=18.1m^{0.154}$, $x_c=0.497m^{-0.00865}$, $\beta=4.35m^{0.0393}$.
- B16 at z = 0: $E^{\rm pub}=\rho_0q^{-0.2}(1+q^\alpha)^{-(\beta-0.2)/\alpha}$ with $q=x/0.5$, $\rho_0=4000m^{0.29}$, $\alpha=0.88m^{-0.03}$, $\beta=3.83m^{0.04}$.
- $m=M_{200c}/10^{14}M_\odot$.

The mean-DM lower reference is

$$L=O_{\rm fb}+\sum_b[o^{D0}_b(x_t)+o^{D3}_b(4)-o^{D3}_b(x_t)],$$

- The share from beyond $x_t$ is $\sum_b[o^{D3}_b(4)-o^{D3}_b(x_t)]/L$.
- D3 is Test J's host-bound electrons out to 4R.

### L1.6 Choice of rule and recommendation

- **B32 is kept as the fit-range rule.**
  - It follows the user's definition directly.
  - It is nearly mass-independent, lies inside the MDK15 splashback estimate in every bin below 10^15.3, and is insensitive to nesting.
  - A single global value reproduces the per-bin radii's additive terms with an RMS of 0.17-0.57% per family. The largest single-term deviation is 1.19% (mean DM at the common x_t) (L1.0 item 3).
- **The evidence does not strongly favour another rule.**
  - C68 depends strongly on mass and is too small (0.75 R200c for P) to constrain the outer slope.
  - The shape- and field-based edges lie at 1.6-2.4 R200c, where the field already dominates.
- **Fit range for L3:** R_fit = x_t = 1.43 R200c, with all gas inside it. To see how the extrapolation responds, also try shorter ranges, 1.20-1.24 (B50, Req-P) and 1.33 (Req-E), and longer ones up to the SB/MDK15 values (about 1.6-2.0).
- **Painting:** the fitted law is painted to 4 R200c.
  - For kSZ and DM, 0.28-0.61 of the painted 1-halo/linear terms comes from R > x_t if the law follows all gas there.
  - At stack level, the region beyond x_t holds 8-18% of the lower reference (bound gas beyond x_t) and 38-80% of the A2-ii content.
- **L3 must therefore:**
  - quantify, by bootstrap and Fisher, how well the inside data constrain the outer parameters (beta, x_c, alpha);
  - report t for each fitted law and probe;
  - choose, for kSZ and DM, between the halo-component reference (bound gas beyond x_t) and D0, in line with the user's goal of painting the halo component and not the field.
- **Redshift extension:** z = 0.1-1, with the same rule.

**Figures** (`figures/`, PNG and SVG):
- Main: `L1_bound_fraction_profiles`, `L1_radii_bound_rules`, `L1_extrapolation_bracket`.
- Also: `L1_radii_alternatives`, `L1_alternative_diagnostics`, `L1_inside_fractions`, `L1_observable_retention`, `L1_halfdome_map_retention` (fit radius marked; beyond it the law is extrapolated and still painted), `L1_pilot_comparison`, `L1_signal_weights`.

### L1.7 What was run, and checks

**Runs.** Everything ran locally: stack-level work on the cache, 2 threads. No fine basis, truth or DM-PDF pass was needed, so STRW maas was not used (no `CLUSTER_RUN.json`; `cluster_tests.py` has no 'L' entry yet).
- Full run, 2026-10-16 15:38:22-15:39:04: 42 s, peak RSS 0.38 GB [results/run_manifest.json `invocations[0]`].
- Later stages were re-run at 15:40, 16:07-16:12 (first fix round) and 16:25-16:29 (second round: new stage `extrap`, relabelled figures). Each run took ≤ 56 s and ≤ 0.31 GB [`invocations[1..6]`]. Smoke mode was re-run alongside.
- This round changed text, checks and one figure legend; no result changed.
  - Stages global, codex, extrap, figures and numbers ran at 17:04:27-17:04:39 (12 s, ≤ 0.31 GB) [`invocations[7]`]. The verify stage then ran with `L_REPORT_TEXT` set to this text [`invocations[8]`].
  - Every value in radii.json, stacks_summary.json, global.json, codex.json and extrap.json equals the previous round's, as do every CSV table and every array in extrap.npz.
  - extrap.json only gains `observable_level.yy1h_fallback_ESTIMATE`. report_numbers.json gains its 15 entries (6,886 → 6,901). `tables/L1_report_tables.json` is new.
  - The electron row of `L1_extrapolation_bracket` now has its own legend, which names the published B16 shape.
  - Smoke mode was re-run alongside, with identical values.

**Gates** [`verification.json`; all 13 passed]:

| gate | check | result |
|---|---|---|
| G0 | partition closure all = a2ii + p10 + p11; library A2-ii = cache; no complement inside R200c; host-bound counts = Test J's J0 | ≤ 1.4e-14; equal; none; equal |
| G1 | stacks vs stackcore.stack_bin; bootstrap rows; merged groups vs Test J | ≤ 7.3e-15; identical; equal |
| G2 | 0 ≤ f ≤ 1; fraction definitions coincide inside R200c; nested-bracket order; cb ≤ hb | hold |
| G3 | rule functions on synthetic profiles | pass (crossing 3.2e-5, slope 9.8e-15) |
| G4 | B32 crosses in every bin and field; B50 ≤ B32 | pass |
| G5 | fallback + Σ own curves at 4R = J's trunc_all | ≤ 2.2e-16 |
| G6 | Codex shell matrices sum to the spectra; CSV band fractions; pilot cumulative fractions | ≤ 9.0e-9; ≤ 3.1e-10; 2.2e-16 |
| G7 | smoke vs full central values | identical (115 values) |
| G8 | report-number registry: each entry re-resolved from its source field (registry against files only; G8 does not read this text) | 6,901 entries |
| G9 | earlier tests, cache, analysis/, particle products, cluster_tests.py | none modified |
| G10 | 9 observable optima inside the grid; vectorised = scalar interpolation | pass |
| G11 | published shells vs 48-point engine quadrature; shares at 4R; D3 closure; mean-DM lower reference in [D3, D0]; s recomputed; s_hb, s_a2ii ≤ s_all (201 rows) | ≤ 1.1e-14; 0; 2.2e-16; holds; exact; 0 violations |
| G12 | report tables re-formatted from the JSON by code independent of `l_numbers.py` = `tables/L1_report_tables.json`; every row found verbatim in this text | 7 tables, 55 rows; 0 mismatches; 0 missing |

*Post-verification text corrections (coordinator, 2026-10-16): pressure B32 range 1.35-1.45; B12-shape share 30-35%; B32cb CV 2.5% / 2.6%; pilot comparison 1.33-1.40; the L1_m9 pilot halo is at about the 10th percentile of its bin. Numbers from results/radii.json, extrap.json, global.json.*
