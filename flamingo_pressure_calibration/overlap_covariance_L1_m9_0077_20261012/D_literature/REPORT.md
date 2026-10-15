# Test D: literature recipes on the same haloes (FLAMINGO L1_m9, z = 0)

Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md` (Test D, "Literature corrections", "AGN-heated gas").
Full run on STRW maas (`run_test_d.py --full`, 2 workers, nice 19, 1,899 s wall, peak tree RSS 3.8 GiB, all gates passed).
Figures and tables were made locally from `results/`. The tSZ C_ell at ell = 100/1000/3000 were recomputed locally with a
cancellation-free radial kernel (`results_local/`, see "Numerics"); every number below uses those values.

## What a literature recipe would do to HalfDome (plain language)

Each recipe was run on the same 4,588 FLAMINGO haloes with its own conventions (statistic, halo weighting, scaling, mass
bins, particle estimator and selection, fit range and weighting, truncation). It was turned into a mean painted profile per
0.2-dex M200c bin of HalfDome's catalogue and propagated through the `kernels/` lightcone engine.

Changes are relative to our reference:
- all gas, satellites kept, recent-AGN gas removed;
- geometric shell pressure;
- inclusion-weighted mean in 0.2-dex M200c bins;
- R200c / P200 scaling;
- B12 fit to 2 R200c with the GLS objective (c);
- hard cut at 4 R200c.

| Rank | Recipe | tSZ: yy 1h ell = 100 / 1000 / 3000; 2h ell = 100; Y(<4R) | kSZ / DM | Main ingredient(s) |
|---|---|---|---|---|
| 1 | B24 with X-ray (Q_X) weighting | +80 / +42 / +23%; +87%; +50% | - | L_X-weighted particle pressure (Shapley +152% at ell = 1000), partly cancelled by the median (-104%) |
| 2 | A10 universal profile | +29 / +5 / -4%; -8%; -8% | - | universality with alpha_P = 0.12 (+40 / +13 / +2%); median (-4 to -11%); R500c scaling (-3%); 5 R500c cut (-2% on Y) |
| 3 | B24 with mass (Q_M) weighting | +29 / +16 / +8%; +20%; +12% | - | mass-weighted particle pressure (+22 to +66%); median (-9 to -51%) |
| 4 | B16 density, converted to electrons | (tau: -13 / -13 / -8%) | kSZ 1h ell = 1000 -12.6%; DM variance -10%; mean DM -5.3%; DM quantiles -4.9 to -6.4%; N_e(<4R) -8.7% | gas density painted with XGPaint's mean molecular weight (-9% on N_e, -14% on kSZ) |
| 5 | B12 | +5.6 / +4.4 / +4.5%; +5.2%; +2.1% | - | Y200 weighting (+1.6 to +3.4%); internal-variance fit over 0.01-3 R200c (+1 to +4%) |
| 6 | B24 with volume (Q_V) weighting | +0.6 / -1.9 / -3.1%; -5.5%; -3.2% | - | ingredients cancel: Q_V on X-ray gas (+5 to +9%) against the median (-3 to -11%), R500c scaling and 0.5-dex M500c bins |
| 7 | Lee22 (fit to 1.34 R200c, h^-1 Msun bins) | -1.3 / -2.3 / -2.0%; -1.3%; -0.7% | - | both ingredients <= 2.4% at ell >= 1000 |

What drives the large shifts:
- **B24 particle weighting.** L_X weighting (and, less, mass weighting) selects dense, X-ray-bright gas: satellite haloes and
  clumps, whose pressure beyond R200c is several times the volume average. Volume weighting (Q_V) is close to our
  geometric estimator: +2-6% inside R200c, +9-11% at 2 R200c.
- **A10.** One universal shape is too extended for massive clusters and too concentrated for groups. It boosts the
  large-scale 1-halo power and removes 8-14% of the 2-halo power.
- **B16 (DM / kSZ).** The bias is mostly a convention: painting B16's gas density with XGPaint's mu_XGP (= 1.112 x the fully
  ionised mu_e) gives 10% fewer electrons. FLAMINGO's actual electron content offsets 1.7% of that.
- **Mean DM moves less than N_e** because the engine's calibrated mass range (log M200c >= 13.5) carries only 59% of the mean
  DM. The published B16 fallback below that mass is unchanged in every variant.
- **Recent-AGN gas** changes no observable by more than 0.26% in any recipe.

Figures: `figures/fig02_recipe_totals.png`, `figures/fig06_cl_ratio_vs_ell.png`.

![totals](figures/fig02_recipe_totals.png)

## Recipes as implemented

Each recipe switches ingredient groups from the reference. Shapley values are computed over all 2^K subsets.

**B12 (Battaglia+12)**
- *Literature convention:* Y200-weighted stacks of P/P200 in R200c; fits weighted by the internal variance per radial bin. No
  fit range is stated (30 log bins over 0.01-3 R200, data to 2.73 R200). Maps are cut at 6 R500 with a Gaussian taper
  (Sect. 5.3; the printed width "80 x R500" is read as a typo and s = 0.1 x_t used, as in C_sigmoid).
- *Groups switched:*
  - W: equal -> Y200 weighting;
  - P: AGN removed -> all gas;
  - F: GLS to 2R -> diagonal internal-variance fit over 0.01-3 R200c;
  - T: 4 R200c hard cut -> Gaussian taper beyond 6 R500c.

**B16 (Battaglia 2016, App. A)**
- *Literature convention:* gas density, gNFW with xc = 0.5 and gamma = -0.2 fixed, same radial grid as B12, no truncation
  stated.
- *Groups switched:*
  - W: Y200 weighting (B12 convention; **assumed**, not stated in B16);
  - P: n_e (AGN removed) -> gas density / mu_XGP (all gas);
  - F: as B12.
  - Truncation is kept at 4 R200c.

**A10 (Arnaud+10)**
- *Literature convention:* median P/P500 vs r/R500 over 0.03-1 R500 (X-ray) and 1-4 R500 (simulations); GNFW
  (P0, c500, gamma, alpha, beta) fitted in log-log, weighted by the dispersion; Y to 5 R500.
- *Groups switched:*
  - Q: mean -> median;
  - S: R200c/P200 -> R500c/P500 (A10 formula);
  - M: 0.2-dex bins -> one universal stack of 1e14-1e15 M500c haloes scaled by (M500/3e14 h70^-1)^0.12;
  - E: geometric -> X-ray-like n_e T_X inside R500c (L_X-weighted T), geometric beyond;
  - F: B12 GLS to 2R -> gNFW5 over 0.03-4 R500c weighted by the 16-84% dispersion;
  - T: 4 R200c hard cut -> hard cut at 5 R500c.

**B24 (Braspenning+24)**
- *Literature convention:* medians in 0.5-dex M500c bins (1e13-1e15.5); 30 log bins over 0.01-3 R500c; excludes star-forming
  gas, T < 1e5 K and gas heated by AGN in the last 15 Myr; Q_V / Q_X / Q_M particle weightings; P500 with f_B = 0.159,
  mu = 0.59, mu_e = 1.14.
- *Groups switched:*
  - Q: median;
  - S: R500c / P_e,500 (B24);
  - M: 0.5-dex M500c bins;
  - E: geometric all gas -> Q_V, Q_X or Q_M over X-ray-selected gas (three variants);
  - F: gNFW5 over 0.01-3 R500c, dispersion-weighted (B24 publish medians only; this painting fit is our construction).
  - Truncation is kept at 4 R200c.

**Lee22 (for completeness)**
- *Literature convention:* 27 log shells over 0.04-1.34 R200c; lognormal likelihood with the full covariance; 0.2-dex bins in
  h^-1 Msun from 1e13 h^-1; cells need > 29 haloes.
- *Groups switched:*
  - M: physical-Msun bins -> h^-1 Msun bins (h = 0.681) with the > 29-halo rule;
  - F: GLS + sigma_d to 2R -> full-covariance GLS (no sigma_d) on the Lee22 grid.

**Mapping to HalfDome.**
- Every recipe profile is painted per halo in units of R200c:
  - R500c-scaled profiles use each halo's own R500c/R200c and P500/P200;
  - the universal A10 profile is multiplied by (M500/M_piv)^0.12;
  - truncation windows are per halo (6 or 5 R500c = 3.3-4.2 R200c).
- Profiles are then averaged with inclusion weights over the haloes of each 0.2-dex M200c engine bin.
- The engine paints calibrated profiles for log M200c 13.5-15.388 and the published B12/B16 fallback elsewhere (as in every
  earlier test).

**Checks on the inputs.**
- The reference GLS fits (objective c) reproduce quicklook q4 at 2 R200c to |dln theta| = 1.6e-9 (bins with log M200c >= 14.5).
- All B24 M500c bins and the A10 universal range are complete for the M200c selection 12.5-15.5: M500c < M200c for every
  halo (maximum ratio 0.88), and the most massive central has log M200c = 15.388. The 15.0-15.5 M500c bin holds only 15
  haloes.
- Lee22's two highest h^-1 bins (17 and 2 haloes) are excluded by its cell-size rule. Their haloes are painted with the
  14.8-15.0 h^-1 bin.

**Radial range each recipe actually gets.** 0.01 R500c (0.0065 R200c) is never resolved. First resolved shell, with the
library masks and a common 0.04 R200c floor:
- M200c bins 13.3-13.5 ... 15.3-15.5: 0.52, 0.32, 0.27, 0.17, 0.14, 0.105, 0.090, 0.065, 0.056, 0.040, 0.040 R200c;
- B24 M500c bins: 0.081 (13.0-13.5, diagnostic mask), 0.37, 0.21, 0.098, 0.067 R500c;
- A10 universal stack: 0.21 R500c (fit range 0.21-4 R500c);
- Lee22 bins: 0.10-0.42 R200c (fits end at 1.34 R200c).

The full table is `report_tables/t7_fit_ranges.md`.

**Unresolved cores.**
- The free-gamma gNFW5 fits extrapolate the local slope into the unresolved core.
- When the free fit was not accepted (bound or ill-conditioned), gamma was fixed to A10's 0.3081, as Planck 2013 did. This
  applied to 121 of 898 bin fits; 24 fits remain not accepted (mostly Q_X subsets).
- Splices replace each recipe profile by the reference below the first resolved shell. This changes the full-recipe C_ell by
  <= 0.7%, except A10's 2-halo term (1.6-3.7%), so the totals above come from resolved radii.

**Total change and the part from unresolved radii.** Each cell is the total change in %, with the part from unresolved
radii in brackets.

| Recipe | 1h ell = 100 | 1h ell = 1000 | 1h ell = 3000 | 2h ell = 100 | 2h ell = 1000 |
|---|---|---|---|---|---|
| B12 | +5.6 (+0.29) | +4.4 (+0.47) | +4.5 (+0.65) | +5.2 (+0.35) | +4.5 (+0.44) |
| A10 | +29.4 (+0.11) | +5.0 (+0.33) | -3.8 (+1.11) | -8.4 (+1.58) | -14.2 (+2.44) |
| B24_QV | +0.6 (-0.14) | -1.9 (-0.33) | -3.1 (-0.71) | -5.5 (-0.36) | -6.5 (-0.45) |
| B24_QX | +79.9 (+0.19) | +41.9 (+0.36) | +23.2 (+0.40) | +86.6 (-0.01) | +48.5 (-0.09) |
| B24_QM | +28.7 (-0.06) | +15.6 (-0.10) | +8.3 (-0.32) | +19.6 (-0.31) | +9.0 (-0.41) |
| Lee22 | -1.3 (-0.14) | -2.3 (-0.18) | -2.0 (-0.27) | -1.3 (-0.14) | -1.1 (-0.16) |

For B16 the unresolved part is <= 0.2% for kSZ and DM.

## Ingredient waterfalls

Exact Shapley values (% of the reference observable) are averaged over all orderings of the groups and sum to the total
(residual 2.8e-14 %).
- Figures: `figures/fig01_waterfall_<recipe>.png` and the heat map `figures/fig03_shapley_heatmap.png`.
- Every group's one-at-a-time and last-in values are in `report_tables/t2_shapley_<recipe>.md`, as
  "Shapley (first-in / last-in)".

**B12.**
- Y200 weighting: +1.4% on Y(<R200c); +1.6 / +2.4 / +3.4% on yy 1h at ell = 100 / 1000 / 3000. Heavier haloes in a bin have
  more concentrated profiles, so the core rises by 3-5% (fig05).
- Internal-variance fit to 3 R200c: +3.9% at ell = 100, +1.1% at ell = 3000. It raises the 1.5-3 R200c extrapolation.
- Taper beyond 6 R500c: +1.4% on the 2-halo term at ell = 100. Its Gaussian tail adds 0.5% painted Y beyond 4 R200c.
- All gas instead of AGN removed: 0.0%.

**B16.**
- Field ingredient: -8.7 to -8.9% on N_e, -13.9% on kSZ 1h at ell = 1000, -5.2% on mean DM, -7.3% on DM Q99.9.
- Electrons / (gas / mu_XGP) = 1.086-1.097 per engine bin, which decomposes as:
  - mu_XGP / mu_e(full ionisation, X_H = 0.76) = 1.1116, i.e. -10.0%;
  - FLAMINGO's fully ionised electrons per rho/mu_e: 0.982-0.990 (composition);
  - actual / fully ionised electrons: 0.994-0.999 (ionisation);
  - net: -8.5 to -9.1%.
- With mu_e for full ionisation instead of mu_XGP, the field ingredient would be +1.7%.
- Y200 weighting (assumed): +0.2 to +2%.
- Fit to 3 R200c: <= 0.7%.

**A10.**
- Universality is the dominant ingredient: +40.4 / +13.2 / +2.2% on yy 1h at ell = 100 / 1000 / 3000; +3.1% and -6.6% on 2h at
  ell = 100 / 1000; -4.3% on mean y.
- The universal stack is dominated in number by log M200c ~ 14.1-14.3 haloes. Its painted profile (fig04, fig05) relative to
  each bin's own:
  - 14.9-15.1: +38% at 2 R200c;
  - 13.5-13.7: +137% at 0.1 R200c and -60% at 2 R200c.
  - So alpha_P = 0.12 and one shape cannot follow FLAMINGO's mass trend.
- The median lowers the outskirts: -7.4% on Y(<4R), -11.1% on 2h at ell = 100.
- R500c scaling: -2.8 to -3.0% on the 1-halo term.
- The 5 R500c cut: -2.1% of Y(<4R).
- The X-ray n_e T_X estimator and the gNFW5 fit: each <= 1.5%.
- FLAMINGO's universal fit is P0 = 2.29, c500 = 0.98, alpha = 1.27, beta = 5.29, gamma = 0.55 (A10's P500 with the 0.12
  correction, h = 0.681). It lies 13% below the published A10 profile (8.40 h70^-3/2, 1.177, 1.051, 5.49, 0.308) at 0.5 R500.

**B24.**
- Shared ingredients:
  - the median: -7.6% on Y(<4R), -11.4% on 2h at ell = 100;
  - R500c scaling: -0.2 / -0.9% on the same two;
  - 0.5-dex M500c bins: -0.8 / -2.2%. These put a wide bin's haloes on the population-weighted (low-mass) mean; the
    13.5-13.7 engine bin draws 83% of its haloes from the 13.0-13.5 M500c bin.
- Estimators (E):
  - Q_V: +3.4 to +8.7%;
  - Q_M: +14 to +66%;
  - Q_X: +36 to +591% (with mean stacks Q_X alone gives +563% at ell = 100; with medians +90%).
  - The opposite-signed Q and E bars of the Q_X waterfall share this strong median x Q_X interaction.
- gNFW5 painting fit: -2 to +7%.

**Lee22.**
- The h^-1 binning and the 1.34 R200c full-covariance fit each move tSZ by <= 2.4% at ell >= 1000.
- At ell = 100 they interact (Shapley -15.8 / +14.5%). The 1.34 R fit alone gives +30% because it then also fits the 21- and
  6-halo top bins, which Lee22's cell-size rule excludes.

Profiles (0.1-2 R200c, z = 0): `figures/fig04_profile_ratios.png`; per-ingredient decompositions in
`figures/fig05_profile_shapley_<recipe>.png`.

![waterfalls](figures/fig03_shapley_heatmap.png)

## The plan's predictions

Measured on resolved shells, log M200c 13.3-15.1; bootstrap 16-84% bands are in fig07.

| Prediction (plan) | Measured | Verdict |
|---|---|---|
| median/mean 0 to -4% inside 1R | -7.4% to +1.9% (most shells 0 to -5%) | confirmed within ~3 points; single shells reach -5 to -7% |
| median/mean -6% to -41% beyond 1.5R | 1.5-3 R200c: -5.5 to -38%; 3-4 R200c: -24 to -59% | confirmed to 3 R200c; the deficit at 3-4 R200c exceeds the prediction |
| Y200 weighting +1-10% inside 0.5R | +0.1 to +6.5% (largest at low mass) | confirmed (slightly below +1% in the most massive bins) |
| Q_V +3-6% inside 1R | +2.1 to +6.4% (X-ray-selected Q_V +2.2 to +6.9%) | confirmed |
| Q_V +9-13% at 2R | +9.0 to +10.9% | confirmed (lower half) |
| Q_V larger beyond 2R (+10-29%) | +10 to +31% over 2-4 R200c (X-ray-selected up to +40%) | confirmed |

## Positive controls

**(i) Our aperture sums vs SOAP**, all 4,588 haloes. The X500 grid has exact edges at 0.15 and 1 R500c; X200 at 1 R200c.
Details are in `report_tables/t4_soap.md` and fig08.

| Quantity | Median offset | 16-84% half-width | Within 0.3% | Within 0.3%, >= 1000 gas particles (2,825 haloes) |
|---|---|---|---|---|
| Y(<R500c), all gas | +2.0e-5 | +-5.7e-4 | 98.5% | - |
| Y(<R500c), no recent AGN | +1.8e-5 | +-5.7e-4 | 98.5% | 99.96% |
| Gas mass (<R500c) | +1.8e-5 | +-6.7e-4 | 97.8% | 99.7% |
| Mass-weighted T(<R500c), all gas | -5e-7 | +-5e-4 | 99.4% | 100% |
| Mass-weighted T(<R500c), no recent AGN | -4.6e-6 | +-5e-4 | 99.5% | 100% |
| Mass-weighted T, without cool gas and recent AGN | -3.1e-5 | +-5e-4 | 99.3% | - |
| Mass-weighted T, core-excised (0.15-1 R500c), all three selections | -7e-6 to -1.0e-5 | - | 99.4% | - |
| Mass-weighted T(<R200c), all gas | +4e-6 | - | 99.6% | - |

- The "without cool gas and recent AGN" comparison is approximate: our X-ray phase also drops star-forming gas with
  T >= 1e5 K.
- The shell sums reproduce SOAP's SO apertures at the float32 level of SOAP's temperature columns.

**L_X weighting vs spectroscopic-like temperature.** Our L_X-weighted T_X (X-ray-selected gas) compared with SOAP's
SpectroscopicLikeTemperatureWithoutRecentAGNHeating:
- R500c: +8.8% median (16-84%: +5.2 to +26.5%);
- core-excised: +8.2%;
- R200c: +13.9%;
- well-resolved haloes: +6.9% (+4.8 to +10.5%);
- by mass: about +20-30% at M500c ~ 1e12.5-13 and +6-8% above 1e14.

So the X-ray weighting is not spectroscopic-like. B24 equate the two, but T_X is about 7% higher in clusters and 20-30%
higher in groups. The mass-weighted T of the X-ray phase is -2.2% (+-10%) from the spectroscopic-like value.

**(ii) B24's statements** (medians in B24's bins, fig09):
- **"Pressure nearly weighting-independent": only partly reproduced.** It holds roughly for Q_M in the inner region, not for
  Q_X, and not near or beyond R500c.
  - Q_M/Q_V: +0.8 to +10% at 0.15-0.5 R500c (mostly +2 to +7%), +3 to +13% at 0.5-1 R500c, +5 to +70% at 1-3 R500c.
  - Q_X/Q_V: +0.1 to +18% at 0.15-0.5 R500c (mostly +4 to +15%), +8 to +28% at 0.5-1 R500c, +15 to +200% at 1-3 R500c.
- **"Removals negligible": reproduced above log M500c = 13.5, not in low-mass group cores.**
  - Q_V with the X-ray selection vs all gas: <= 0.23% for log M500c >= 13.5.
  - The AGN cut alone: <= 0.31% everywhere.
  - In the 1e13-1e13.5 bin, removing cold and star-forming gas raises Q_V by up to 17% at 0.15-0.5 R500c and up to 190% at
    0.08-0.15 R500c.

## Recently AGN-heated gas (paired table)

Each pair flips only the AGN selection: removed vs kept for the reference, B12, B16 and A10; for B24, the X-ray selection
with or without the recent-AGN gas.

| Pair | yy 1h ell = 100 / 1000 / 3000 | yy 2h ell = 100 | Y(<R200c) / Y(<4R) | kSZ 1h ell = 1000 | mean DM / DM Q50 / Q99.9 | max abs dln theta |
|---|---|---|---|---|---|---|
| reference P, kept vs removed | -0.000 / +0.000 / +0.000% | +0.001% | +0.000 / +0.001% | - | - | 0.06% |
| reference n_e, kept vs removed | - | - | - | +0.000% | +0.000 / +0.000 / +0.000% | 0.37% |
| B12 (all gas) vs AGN removed | -0.001 / -0.001 / -0.001% | -0.002% | -0.000 / -0.001% | - | - | 0.13% |
| B16 (all gas) vs AGN removed | - | - | - | -0.001% | -0.000 / -0.000 / -0.000% | 0.11% |
| A10 (all gas) vs AGN removed | +0.042 / +0.066 / +0.082% | +0.031% | +0.026 / +0.018% | - | - | 4.1% |
| B24_QV X-ray + AGN vs X-ray | +0.001 / +0.001 / +0.001% | +0.010% | +0.001 / +0.007% | - | - | 0.54% |
| B24_QX X-ray + AGN vs X-ray | +0.217 / +0.103 / +0.070% | +0.255% | +0.076 / +0.165% | - | - | 6.8% |
| B24_QM X-ray + AGN vs X-ray | +0.208 / +0.076 / -0.047% | +0.011% | -0.016 / +0.002% | - | - | 222% |

- The largest observable change is 0.26%.
- Parameters of the degenerate gNFW5 fits can move a lot (B24_QM: 222%) while the profile moves by <= 0.2%.
- Core profiles (fig10): keeping AGN gas raises the innermost, unresolved shell (x < 0.025 R200c) by +73 / +28 / +4 / +0.6%
  at log M200c 13.5-13.7 / 13.9-14.1 / 14.5-14.7 / 14.9-15.1. Resolved shells move by <= 0.2%.

## Numerics: a cancellation in the frozen radial kernel (found and corrected locally)

- **Where it fails.** The engine's frozen kernel uses an 8-term series for q < 0.2 and closed-form antiderivatives for
  q >= 0.2. For q >= 0.2 on the innermost nodes (q x << 1) the closed form cancels to noise: on 60 inner nodes its weights
  are off by up to 7e9 x the largest weight, against 50-digit mpmath.
- **When it matters.** It is harmless for B12/B16-type cores (regular profiles: 3.5e-7). It matters for the steep unresolved
  cusps that free-gamma gNFW5 fits produced in a few calibration bins (gamma up to ~2 in the 1e13-1e14 M500c bins fitted in
  R200c units).
- **The fix.** `td_stable.py` applies the engine's own guard rule to every q (series per interval wherever q x <= 1.5). It
  matches mpmath to 2.9e-13.
- **The recomputation.** `td_fixcl.py` re-evaluated every pressure variant at ell = 100/1000/3000 (65 s, 2 workers):
  - unaffected variants agree with the maas values to 1.4e-6, and the P2 gate to 2e-9;
  - 33 of 156 painted variants have a transform-error indicator >= 1e-3;
  - only 4 changed by more than 0.1%, all intermediate B24 subsets with M500c bins and gNFW fits in R200c units. B24_QX
    `Q+M+E+F` changed by +4009% at ell = 1000, and the `M+F` subset shared by the three B24 variants by +58%.
- **Not affected:**
  - none of the full recipes (indicators <= 6e-4);
  - Y, N_e, mean y, DM moments and quantiles;
  - kSZ / tau spectra (B16 profiles, indicators < 1e-3).
- **Which values are used.**
  - The Shapley values above use the corrected C_ell. The largest corrections were in the B24_QX decomposition at
    ell = 1000; B24_QV / QM moved by up to 3-4 points.
  - The maas-run values are in `results/shapley_summary.json`; the corrected ones in
    `results_local/shapley_corrected_summary.json`.
  - The C_ell-vs-ell curves of the four affected subsets at other multipoles were not recomputed and are not used.
- **Recommendation:** any engine use with steep cores (Test F's flexible families) should adopt this guard.

Figure: `figures/fig11_numerics.png`.

## Checks

| Check | Result |
|---|---|
| Cache vs measurements_legacy.npz (maas, 4,588 haloes) | max rel 1.1e-14 |
| Test D estimators vs analysis.variants.profile (400 haloes); resolution masks | 3.8e-16; identical |
| Reference GLS vs quicklook q4, objective (c) at 2 R200c, bins >= 14.5 | max abs dln theta 1.6e-9 |
| Engine P2 / E4 at 4 R200c vs C_sigmoid | 3.7e-6 / 7.8e-6 |
| TableProfile vs FitProfile | 3.3e-16 / 4.4e-16 |
| Shapley efficiency (maas; corrected) | 1.6e-13 % / 2.8e-14 % |
| Result files vs MANIFEST hashes | 18/18 data files equal (run.log appended after hashing) |
| Stable kernel vs mpmath | 2.9e-13 |
| Unaffected variants, frozen vs stable kernel | 1.4e-6 |
| P2 gate | 2.0e-9 |
| Local smoke test (50 haloes, all code paths) | 58 s, all gates passed |
| SOAP positive controls | above (median offsets <= 3e-5) |
| Figures | all 23 PNGs inspected; overlaps fixed |

All values are in `verification.json`. Template subtraction was not used, so the later `geometry.make_G` fix (zero-amplitude
templates) does not affect Test D.

## Caveats

**Halo model** (see `kernels/REPORT.md`).
- Deterministic mean profiles per cell, with an assumed redshift response; Limber, linear 2-halo term.
- The calibrated support (log M200c 13.5-15.388) carries 97-99% of yy but only 59% of mean DM, 51-89% of the DM quantiles and
  75% of the DM variance. The published fallback dilutes all density effects.

**Recipe choices that are ours.**
- B16's Y200 weighting (assumed; its Shapley value is shown separately: +0.2 to +2%).
- B24's gNFW5 painting fit.
- The A10 universal sample (1e14-1e15 M500c) and its X-ray-like estimator (n_e x L_X-weighted T, which is not
  spectroscopic-like: +7% in clusters).
- Per-bin fits instead of B12/B16's published mass power laws.
- The B12 taper width (printed "80 x R500" taken as a typo).

**Resolution.**
- Recipe inner limits (0.01-0.04 R500c) are never reached.
- Free-gamma fits extrapolate the resolved slope; gamma was fixed in 121 bin fits, and 24 fits are not accepted.
- Unresolved radii contribute <= 0.7% to the full-recipe C_ell (A10 2-halo: 3.7%), from the splices.

**Interactions.** Shapley values are exact but can be dominated by interactions: B24_QX (median x Q_X) and Lee22 at
ell = 100 (fit x cell-size rule). One-at-a-time and last-in values are reported for every group.

**X-ray luminosities.**
- Q_X, T_X and the A10 estimator weight by the particle XrayLuminosities (ROSAT band). These are +0.3% higher than SOAP's
  recomputed L_X(<R500c) in the median (16-84%: +0.06 to +0.76%), with 148 haloes above 5% (verification.json
  G7_assessment); the difference is independent of the phase cuts.
- So the weights are defined by the stored particle field, not by SOAP's calculator. This is a definitional caveat for every
  L_X-weighted quantity, not an extraction error.

**Sampling and uncertainties.**
- Below 1e14.5 Msun the sample is stratified (inclusion-weighted stacks); the 15.0-15.5 M500c bin has 15 haloes.
- No bootstrap uncertainty was propagated to the recipe observables (that needs engine runs per replicate). The prediction
  ratios carry bootstrap 16-84% bands.

**Environment.** maas ran scipy 1.15.2; local runs used 1.17.1. Both carry the same CODATA 2022 sigma_T / m_e.

## Files, runtimes, reproduction

**Cluster run (maas).**
- Command: `STUDY_ROOT=<study> nice -n 19 python -B D_literature/run_test_d.py --full --workers 2`, launched through
  `../cluster_tests.py run D` (polite wrapper `../cluster_job.py`).
- Scripts: `run_test_d.py` with `d_common.py`, `td_cache.py`, `td_recipes.py`, `td_engine.py`, `td_controls.py`,
  `td_shapley.py`. The deployment list is in `CLUSTER_RUN.json`.
- Timings: cache 72 s, fits 149 s, engine 1,669 s (167 variants in 3 batches), controls 5 s, Shapley 1 s.
- Outputs: `results/`.

**Local post-processing** (light):
- `td_fixcl.py` (65 s), `td_postshap.py`;
- `make_figures.py` (23 figures, PNG 150 dpi + SVG, in `figures/`; tables in `report_tables/`);
- `make_verification.py` (`verification.json`, `SHA256SUMS`).
- Outputs: `results_local/`.

**Smoke test:** `run_test_d.py --smoke` (58 s, `results_smoke/`).

**Key products:**
- `results/shapley.csv`, `results/observables.csv` (every variant);
- `results/tables.npz` (painted tables and Y);
- `results/fits.json`, `results/fit_parameters.csv`, `results/stacks.npz`;
- `results/controls.json`, `results/agn_paired.csv`, `results/splices.csv`, `results/profile_shapley.npz`;
- `results_local/shapley_corrected.csv`, `results_local/numerics.json`.

(Written by the coordinator from the agent's hand-back; the harness blocked the agent from writing `.md` files.)
