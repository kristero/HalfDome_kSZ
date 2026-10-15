## Which gas and which haloes enter the stacked profiles (group `selection`)

### What was measured

**Reference.** Every variant is compared with the same reference stack:
- geometric estimator (sum over particles divided by the geometric shell volume);
- recently AGN-heated gas removed;
- all gas in the sphere, all phases;
- fit grid (35 shells to 4 R200c);
- inclusion-weighted mean;
- strict resolution mask. A shell is used only if every halo of the bin passes in it. A halo passes when the shell holds >= 20 gas particles (all owners and phases) and the shell's inner edge is >= max(0.04, 4 eps/R200c) and >= 2 x the shell's upper-median smoothing length h a/R200c;
- M200c bins of 0.2 dex;
- 200 halo bootstraps with the core's per-bin seeds, stratified by design stratum. The replicates are therefore paired between variants.

T_e and K use the core's `weighted` estimator, which is the only one the cache holds for them.

**Row subsets (the halo-sample choice).** These reuse the full bin's resampling matrix, restricted to the subset's rows. This keeps them paired with the reference.

**What is reported for each variant.**
- **Band ratio.** In each radial band, the median over strict-resolved shells of F_variant/F_ref. Its error is the SD of the same median over the paired replicates. Field: `results.json: variants.<id>.band_ratio`. Single-shell ratios are in `shell_ratio`.
  - A band has a value only where the bin has strict-resolved shells in it. The <0.1 band exists only in 14.5-15.5 and the 0.1-0.25 band only in 13.9-15.5, so "cal" ranges for these two bands cover only those bins.
- **Integral ratio.** The ratio of the volume integrals of the stacked profiles over 0-1, 1-2, 2-4, 0-2 and 0-4 R200c, for example Y(<4R) and N_e(<4R). Field: `integral_ratio`.
- **Observable impact.** First-order changes from `sc.impact_from_stacks`, with paired-bootstrap errors. Field: `impact`.
  - The change is dO/O = k sum imp[b,s] delta[b,s], with k = 2 for C_ell, kSZ and the DM variance and k = 1 for means and quantiles. This is exact for the linear observables (mean y, mean DM) and first order for the others. For large changes the first order overstates the quadratic observables: a uniform -50% change gives -100% x share to first order, but -75% x share exactly.
  - **Only the 10 calibrated bins (10^13.5-15.5) are perturbed.** The 12.7-13.5 haloes, which HalfDome paints (down to M200c = 6.6e12 Msun) with the published fallback profile, do not respond, and neither do haloes above 10^15.5.
  - The calibrated bins carry 98.6 / 98.5 / 97.5% of the tSZ importance (ell = 100 / 1000 / 3000), 70% of mean y, 90.7 / 82.6 / 63.0% of kSZ 1h, 58.4% of mean DM, 75.6% of the DM variance and 49.9% of the DM 16th percentile. Most of the rest sits in the 12.7-13.5 bins: 41.6% of mean DM and 17.4% of kSZ at ell = 1000. Bins above 10^15.5 hold at most 0.6% of any of these observables (tSZ at ell = 1000) (`text_support_v2.importance_share`).
  - Every impact below is therefore the effect on the calibrated bins only. For the global choices (sections 1-3), a labelled first-order estimate with all painted bins perturbed is also given (`text_support_v2.all_painted_bins_estimate`). The inner shells of the 12.7-13.5 bins are unresolved, so these estimates are rough.
  - Each impact also has a `resolved_only` version, which leaves unresolved shells unchanged.
  - The significance of each impact (|value| / error) is in `text_support.impact_significance`. The per-bin contributions are in `impact.per_bin_pct`, `text_support.clean_per_bin` and `text_support_v2.clean_per_bin_groups`.

**How ranges are quoted.**
- Ranges marked "cal" run over the nine bins 13.5-15.3.
- The 15.3-15.5 bin holds only 6 haloes. It is quoted separately as "top" (`results.json: summary`).
- Bins below 10^13.5 are exploratory. The strict mask passes nothing in 12.5-12.7. In 12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5 it passes only shells from 2.18 / 1.5 / 1.0 / 0.52 R200c outward (`results.json: bins`).

**Outputs.** All are under `groups/selection/`:
- `compute_selection.py` writes `results.json` and `web_variants.json` (30 variants, 1,560 records, 2.8 MB).
- `summarize.py` adds the `summary` block and writes `summarize_output.txt`.
- `text_support.py` adds the `text_support` block:
  - shell-level extremes;
  - where the heated gas sits;
  - significances;
  - paired differences;
  - the SOAP-only electron conversion.
- `text_support_v2.py` adds the `text_support_v2` block:
  - the per-halo split of the top bin's heated gas;
  - the importance share of the calibrated bins;
  - the all-painted-bins estimates;
  - the low-mass cold/SF shares;
  - the A2-ii single-shell extremes;
  - the ownership groups;
  - the largest option per choice;
  - the CLEAN per-bin groups.
- `make_figures.py` makes the 12 PNG figures.

**Runtimes.**
- `compute_selection.py`: 68.5 s (`results.json: verification.runtime_s`).
- `text_support.py`: 3.4 s.
- `text_support_v2.py`: 1.8 s.
- The runtimes of `summarize.py` and `make_figures.py` were not recorded.

An independent verifier recomputed 54 headline numbers from the cache arrays with its own code and found no mismatch (`verify/v2_independent.json`).

### Recommendation at a glance

| Choice | Options measured | Main effect (first order, calibrated bins 10^13.5-15.5) | Recommendation for painting HalfDome |
|---|---|---|---|
| Recently AGN-heated gas | removed (ref), kept, AGN-heated only | Kept: at most +0.016% on any observable (tSZ, ell = 3000), and at most 0.044% of Y(<R200c) (13.5-15.3). Estimate with all painted bins: at most +0.021% (mean y). | Remove it, as FLAMINGO's own y maps do. The choice is immaterial. |
| Gas phase | all (ref), X-ray phase, cold/star-forming only, central-bound X-ray | X-ray only: kSZ 1h -1.64 / -1.31 / -0.89%, mean DM -0.77%, tSZ at most 0.0015%. Estimate with all painted bins: kSZ (ell = 1000) -2.6%, mean DM -2.2%, DM variance -4.5%. Central-bound X-ray is much larger (kSZ -84 / -60 / -23%, mean DM -35%), but almost all of that comes from dropping unbound and satellite gas, an ownership effect (section 4). | Keep all phases. |
| Electron count | SWIFT n_e (ref), fully ionised H+He, gas density / mu_XGP | Gas/mu_XGP: N_e(<4R) -8.5 to -8.8% (13.5-15.3), kSZ -14.1% (ell = 1000), mean DM -5.0%. Estimate with all painted bins: kSZ (ell = 1000) -16.7%, mean DM -8.2%. | Use SWIFT ElectronNumberDensities in XGPaint units. Never paint gas density with mu_XGP. |
| Ownership | 16 selections: bound membership, overlapper removal, A2-i, A2-ii, excision | A2-ii: band medians inside 1.5R within 0.02% (single shells within 0.12%); 3-4R pressure band -38 to -50%; mean DM -3.2%. Largest: bound-only membership, mean DM -35.3% (central-bound) and -34.1% (host-bound). | Use the A2-ii partition among painted haloes (M200c >= 6.6e12), keeping own satellites and unbound gas. |
| Halo sample | all (ref), CLEAN, overlapped, in-1R, annulus, CLEAN at mu >= 0.03 / 0.3 | CLEAN: tSZ at ell = 3000 +4.2 ± 1.0%, mean DM -1.9 ± 0.2%; tSZ at ell = 100 / 1000 +0.8 ± 1.6 / +1.7 ± 1.8% (consistent with zero). Other subsets move more: overlapper inside 1R, tSZ at ell = 3000 -17.0 ± 1.8%; CLEAN at mu >= 0.03, +15.3 ± 1.5%; overlapper in 1-2R only, mean DM +11.4 ± 0.7%. | Use all centrals with their inclusion weights. Do not select clean haloes. |

Figures:
- `groups/selection/figures/observable_impact.png`: every variant on six observables;
- `groups/selection/figures/integrated_ratio_vs_mass.png`: how the main neighbour-handling options depend on mass.

---

### 1. Recently AGN-heated gas

**What it is.** FLAMINGO's AGN heat a few gas particles stochastically, raising them by Delta T_AGN, about 8.8e7 K in this run.

SOAP flags a particle as recently heated when both conditions hold:
- its last AGN event was within 15 Myr (a_last >= 0.998956);
- its temperature lies in [0.1, 10^0.3] x Delta T_AGN, that is 8.8e6-1.76e8 K.

The pipeline audit estimates that float16 storage of the scale factor stretches the window to about 17.5 Myr. FLAMINGO's lightcone y maps and SOAP's `ComptonYWithoutRecentAGNHeating` remove this gas.

**Options.**
- removed (the reference);
- kept;
- the heated phase alone, shown on the website as the variant "AGN-heated gas only".

**Measured effect.** Sources: `agn_share_of_total`, `variants.selection:agn_kept`, `text_support.agn_share_innermost_P`, `text_support.agn_heated_location` and `text_support_v2.top_bin_agn_per_halo`. Figure: `groups/selection/figures/agn_heated_share.png`.

| Quantity | 13.5-13.7 | 14.1-14.3 | 14.7-14.9 | 15.1-15.3 | Exploratory 12.5-12.7 |
|---|---|---|---|---|---|
| Heated share of Y(<R200c) | 0.044% | 0.0085% | 0.0017% | 0.0004% | 1.59% |
| Heated share of Y(<4R200c) | 0.025% | 0.0067% | 0.0019% | 0.0009% | 0.22% |
| Heated share of the innermost shell (x = 0.019, unresolved) | 42% | 16% | 1.6% | 0.43% | 15% (maximum over shells: 41%, at x = 0.027) |
| Fraction of the heated gas's own Y(<4R) that lies inside 0.1 R200c | 60 ± 8% | 65 ± 6% | 51 ± 6% | 31 ± 7% | 65 ± 23% |
| Fraction of the heated gas's own Y(<4R) that lies at 2-4 R200c | 28 ± 8% | 17 ± 5% | 30 ± 5% | 54 ± 7% | 30 ± 24% |

**Resolved shells.**
- **Pressure share.** The heated share of the pressure is at most 0.032% as a band median (the 3-4R band, 13.5-15.5). Single strict-resolved shells reach 0.054% (13.5-13.7, x = 3.16).
- **2-4R share.** Integrated over 2-4R, the heated gas is 0.004-0.024% of the pressure (`summary.agn_share_of_total.P.integral_pct['2-4'].cal`, 44-237 ppm).
- **Electrons.** The heated share is at most 0.0023% as a band median. Single strict-resolved shells: at most 0.005% in 13.5-15.3, but 0.17% in one shell of the 6-halo top bin (x = 0.071).

**Kept vs removed, profiles** (`text_support.agn_kept_extremes`).

| Field | Largest band median (13.5-15.3) | Largest single resolved shell (13.5-15.3) | Largest single resolved shell, top bin (N = 6, x = 0.071) |
|---|---|---|---|
| P | +3.2e-4 (13.5-13.7, 3-4R; +317 ± 170 ppm) | +5.4e-4 (13.5-13.7, x = 3.16) | +4.2e-4 |
| E | +2.3e-5 | +4.9e-5 | +1.7e-3 |
| T_e | +4.1e-4 | +5.2e-4 (x = 3.16) | -8.7e-4 |
| K | +4.3e-4 | +4.5e-4 (x = 3.16) | -1.0e-3 |

**Kept vs removed, observables** (all shells, calibrated bins):
- tSZ C_ell: +0.005 / +0.009 / +0.016% at ell = 100 / 1000 / 3000;
- kSZ 1h: +0.003 / +0.004 / +0.005%;
- mean DM: +0.001%;
- DM variance: +0.011%.

The paired errors are 5.9-7.4% of these values. Counting only resolved shells, every observable moves by at most +0.002%.

*Estimate with all painted bins perturbed:* tSZ +0.007 / +0.010 / +0.019%, mean y +0.021%, kSZ 1h +0.004 / +0.005 / +0.007%, mean DM +0.002% and DM variance +0.014%.

**Why.**
- **Groups (10^13.5-14.9).**
  - 51-71% of the heated gas's Y(<4R) lies inside 0.1 R200c, next to the central black hole, and 15-30% lies at 2-4R.
  - These innermost shells are unresolved: the strict mask starts at 0.065-0.32 R200c in these bins.
  - There, gas at about 10^8 K is a large share of a shell's pressure: 42% of the innermost shell at 13.5-13.7.
- **Clusters (10^14.9-15.3).** About half of the heated Y(<4R) (48-54%) lies at 2-4 R200c, and only 31-33% inside 0.1 R200c.
  - Our reading is that in clusters most of the heated gas inside the sphere was heated by AGN in neighbours and satellites, out in the outskirts.
  - This is inferred from where the gas sits; the cache does not record which black hole heated a particle.
- **The 15.3-15.5 bin (6 haloes, all with weight 1).** 81 ± 16% of its heated Y(<4R) lies inside 0.1 R200c. This comes from central heating in two of the six haloes:
  - they hold 48% and 46% of the bin's heated Y inside 0.1 R200c;
  - each has most of its own heated Y(<4R) inside 0.1 R200c (98% and 91%).
  - Of the other four, two hold 3% and 4% of the bin's heated Y inside 0.1R. One has heated gas only outside 0.1R, and one has none within 4R.
- **Integrals and resolved shells, calibrated bins.** In the calibrated bins the heated gas is at most:
  - 5.4e-4 of the pressure in any strict-resolved shell;
  - 4.4e-4 of Y(<R200c);
  - 2.5e-4 of Y(<4R).

  There are two exceptions. In one resolved shell of the 6-halo top bin (x = 0.071), keeping the gas raises E by 1.7e-3 and G by 1.9e-3. In the exploratory 12.5-12.7 bin, the heated gas is 1.59% of Y(<R200c).

**Recommendation.** Remove the gas, which matches FLAMINGO's y maps and SOAP Y_noAGN. In the calibrated bins the choice moves no observable by more than 0.016%; the estimate with all painted bins is at most 0.021%. It can be made purely on convenience.
- **For the feedback variants**, apply each run's own SOAP `RecentlyHeatedGasFilter`: Delta T_AGN differs between runs (SOAP audit).
- **The cost** is reading LastAGNFeedbackScaleFactors and Temperatures.
- **For electrons** the cut is only a convention borrowed from pressure.

### 2. Gas phase (X-ray-selected vs all gas; cold and star-forming gas)

**What it is.**
- **X-ray phase:** star-formation rate (SFR) = 0, T >= 1e5 K, and not AGN-heated. This is the B24 / X-ray-analysis selection.
- **Cold/star-forming (cold/SF) phase:** SFR > 0 or T < 1e5 K, and not AGN-heated.
- The reference equals X-ray phase + cold/SF phase.

A "central-bound X-ray gas" variant mimics X-ray analyses that drop both cold gas and gas outside the central subhalo.

**Measured effect.** Sources: `cold_sf_share_of_total`, `variants.selection:xray`, `text_support.cold_sf_innermost_pct`, `text_support.cb_xray_minus_central_bound`, `text_support_v2.low_mass_cold_sf` and `text_support_v2.all_painted_bins_estimate`. Figure: `groups/selection/figures/gas_phase.png`.

| Cold/SF share of the stack | Pressure | Free electrons | Gas mass |
|---|---|---|---|
| Inside R200c (cal) | 0-0.005% | 0.15-1.70% (falls with mass) | 0.63-2.80% |
| 2-4 R200c annulus (cal) | 0.005-0.030% | 1.96-2.38% | 2.71-3.24% |
| 3-4R band (band median, cal) | 0.011-0.044% | 2.86-3.35% | 3.79-4.35% |
| Whole 0-4 R200c (cal) | at most 0.013% | 0.78-1.66% | 1.32-2.45% |
| Innermost shell (unresolved), 13.5-13.7 / 14.7-14.9 | 0.86% / 0.012% | 82% / 15% | 88% / 61% |
| Exploratory 12.5-12.7, inside R200c | 0.53% | 27% | 37% |

In the painted fallback bins 12.7-13.5, the cold/SF phase holds 3.1-17% of N_e(<R200c) and 2.0-5.1% of N_e(<4R). The share is largest at 12.7-12.9 (`text_support_v2.low_mass_cold_sf`).

**Effect of keeping only the X-ray phase.**
- **Observables (calibrated bins).**
  - tSZ changes by at most 0.0015%.
  - kSZ 1h changes by -1.64 ± 0.01 / -1.31 ± 0.01 / -0.89 ± 0.01% (ell = 100 / 1000 / 3000).
  - Mean DM falls by 0.77 ± 0.01%, the DM variance by 1.255% and the DM 16th percentile by 0.97%.
- **Estimate with all painted bins perturbed.** The X-ray cut removes 2.0-5.1% of N_e(<4R) in 12.7-13.5. Including these bins, the first-order estimate is:
  - kSZ 1h -2.3 / -2.6 / -3.9%;
  - mean DM -2.23%;
  - DM variance -4.52%.
- **Temperature.** The electron-weighted T_e of the X-ray phase is higher than that of all gas (band medians, cal):
  - by 0.10-1.6% inside 2R;
  - by 0.9-1.2% at 2-3R;
  - by 2.6-3.1% at 3-4R.
- **Central-bound X-ray gas.**
  - Y(<4R) falls by 25-52% and N_e(<4R) by 49-69%.
  - The first-order observable changes are tSZ -36 / -20 / -9%, kSZ -84 / -60 / -23% and mean DM -35%. Mean DM is linear in the profile, so its value is exact for the calibrated bins. The tSZ and kSZ values are first order: changes this large are overstated (a uniform -50% gives -100% to first order, but -75% exactly), so they show direction and scale only.
  - Almost all of this is the loss of unbound and satellite gas, so the variant is nearly identical to central-bound with all phases (section 4).
  - For tSZ the two agree to better than 0.001 percentage points.
  - For electrons the X-ray cut removes a further:
    - 0.34 / 0.42 / 0.51 points of kSZ;
    - 0.11 points of mean DM;
    - 0.67 points of DM variance.

    For example, kSZ at ell = 1000 is -60.46% against -60.05%, and the DM variance is -37.07% against -36.39%. These paired differences are well above their errors, which are at most 0.01 points.

**Why.**
- **Pressure.** Measured: in 13.5-15.5 the cold/SF phase carries at most 0.044% of the pressure in any resolved band (band median). In 13.5-15.3 it carries at most 0.013% of Y(<4R).
  - Each particle adds N_e k T to the geometric electron-pressure sum (N_e = its number of free electrons), whatever its density. Per unit mass, gas below 1e5 K therefore contributes at most about 1% of what gas at 1e7 K does, and about 0.1% of what gas at 1e8 K does. This is an order-of-magnitude estimate from the temperature ratio.
  - The cold/SF gas is also a small mass fraction: 0.63-2.80% of the gas inside R200c (cal).
  - Star-forming gas has y = 0 in the particle output: none of the 503,296 star-forming particles in the spheres has non-zero y (methods survey).
- **Electrons.** The cold/SF phase holds 0.78-1.66% of N_e(<4R) in 13.5-15.3, 2.0-5.1% in 12.7-13.5 and 7.3% in 12.5-12.7. It has two parts with different reliability:
  - *Non-star-forming gas below 1e5 K* is partly ionised. Its n_e comes from the ionisation tables the simulation uses for cooling.
  - *Star-forming gas* sits on FLAMINGO's subgrid effective equation of state, so its temperature, and hence its tabulated n_e, depends on the model rather than on resolved physics.

  The cache does not separate the two parts. This caveat matters mainly where cold/SF gas is a large share:
  - 82% of the electrons in the innermost, unresolved 13.5-13.7 shell;
  - 17% of N_e(<R200c) in 12.7-12.9.

  In the resolved bands (band medians, cal) its share is 0.10-1.43% inside 2R, 0.87-1.23% at 2-3R and 2.86-3.35% at 3-4R. Two interpretations, neither tested:
  - Inside R200c, the cold/SF gas is the ISM of the central galaxy and of satellites.
  - Beyond 2R, it is cool infalling clumps and filament gas. This would explain why its 3-4R share is 2.9-3.4% in every bin from 13.5 to 15.5.
- **Temperature.** Removing the cold electrons raises T_e slightly, most at large radii.

**Recommendation.** Make no phase cut for either field.
- **tSZ.** The cut is irrelevant: at most 0.0015% in the calibrated bins.
- **kSZ and DM.** In 13.5-15.3, cutting the cold phase removes 0.78-1.66% of N_e(<4R). In the calibrated bins this lowers kSZ by 0.9-1.6% and mean DM by 0.77%.
- **Bound on star-forming n_e.** The X-ray-only variant bounds the model uncertainty of star-forming n_e in the calibrated bins. Even if every cold/SF electron were spurious, kSZ there would move by at most 1.64% and mean DM by 0.77%. The variant also drops the partly ionised non-star-forming cold gas, so this is a generous bound.
- **Low-mass haloes.** This bound does not cover the painted haloes as a whole. Low-mass haloes hold more cold/SF electrons (3.1-17% of N_e(<R200c) in 12.7-13.5, 27% in 12.5-12.7). With all painted bins perturbed, the first-order estimate is kSZ (ell = 1000) -2.6%, mean DM -2.2% and DM variance -4.5%.
- **Where the X-ray selection belongs.** It belongs in X-ray-observable recipes (B24), not in painting.

### 3. Electron counting

**What it is.** XGPaint paints electrons in units of n_e mu_XGP/(rho_c f_b) and divides by mu_XGP = (m_e + 2m_p/(1+0.76))/0.9 = 1.2632 m_p. The options are:
- the stack built from SWIFT `ElectronNumberDensities` (E, the reference);
- full ionisation computed from each particle's own H and He masses, n_e = (m/m_p)(X_H + X_He/2) (E_fullion);
- the gas-density stack in the same units (G). This is what painting a gas-density profile, such as B16, as electrons amounts to.

A fourth case, a single global hydrogen fraction, is derived exactly from G. This is all that SOAP-only data allow: SOAP stores no electron number and no hydrogen mass or fraction (SOAP audit). With one global X, n_e = rho (1+X)/(2 m_p), so E_X = G x mu_XGP x (1+X)/2 shell by shell (`text_support.soap_only_conversion`).

**Measured effect.** Sources: `variants.selection:e_fullion`, `selection:e_gas_muxgp`, `text_support.soap_only_conversion` and `text_support_v2.all_painted_bins_estimate`. Figure: `groups/selection/figures/electron_counting.png`.
- Values are variant / SWIFT - 1, as cal ranges of band medians.
- The <0.1 band is resolved only in 14.5-15.3 and the 0.1-0.25 band only in 13.9-15.3; the other bands cover all nine cal bins.
- The last column is the volume integral.

| Band [R200c] | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 | N_e(<4R) |
|---|---|---|---|---|---|---|---|---|---|
| Fully ionised, per-particle H and He | +0.53 to +1.06% | +0.03 to +0.57% | -0.05 to +0.41% | -0.05 to +0.23% | 0.00 to +0.16% | +0.15 to +0.19% | +0.24 to +0.44% | +0.64 to +0.77% | +0.20 to +0.43% |
| Gas / mu_XGP | -6.1 to -4.9% | -7.8 to -6.9% | -8.6 to -7.9% | -9.0 to -8.5% | -9.1 to -8.8% | -9.0 to -8.8% | -8.9 to -8.7% | -8.5 to -8.4% | -8.75 to -8.49% |
| Fully ionised, global X = 0.76 (SOAP-like) | +4.3 to +5.8% | +2.5 to +3.5% | +1.6 to +2.4% | +1.1 to +1.8% | +1.0 to +1.4% | +1.1 to +1.3% | +1.3 to +1.5% | +1.7 to +1.9% | +1.43 to +1.72% |
| Fully ionised, global X = 0.745 | +3.4 to +4.9% | +1.6 to +2.6% | +0.7 to +1.5% | +0.3 to +0.9% | +0.1 to +0.6% | +0.2 to +0.5% | +0.4 to +0.7% | +0.8 to +1.0% | +0.57 to +0.86% |
| Fully ionised, X_eff = 0.733 (nulls N_e(<4R) at z = 0, mean of the per-bin values 0.730-0.735) | +2.8 to +4.1% | +0.9 to +1.9% | +0.05 to +0.82% | -0.41 to +0.22% | -0.54 to -0.11% | -0.43 to -0.20% | -0.29 to -0.02% | +0.13 to +0.32% | -0.11 to +0.17% |

**Observables (calibrated bins).**
- **Full ionisation (per particle):** kSZ 1h +0.45 / +0.40 / +0.35%, mean DM +0.19%, DM variance +0.64%.
- **Gas/mu_XGP:** kSZ 1h -15.6 / -14.1 / -10.4%, mean DM -5.03%, DM variance -12.3%, DM 16th percentile -4.31%. The paired errors are at most 0.02%.
- **Estimate with all painted bins perturbed:**
  - Gas/mu_XGP: kSZ 1h -17.0 / -16.7 / -15.7%, mean DM -8.2%, DM variance -15.2%, DM 16th percentile -8.5%.
  - Full ionisation: kSZ 1h +0.65 / +0.78 / +1.27%, mean DM +0.62%, DM variance +1.62%.
- **Lower masses (N_e(<4R)):**
  - painted fallback bins 12.7-13.5: full ionisation +0.54 to +1.57%, gas/mu_XGP -7.3 to -8.4%;
  - exploratory 12.5-12.7: +2.4% and -6.4%.

**Why.**
- **Gas/mu_XGP.** The undercount comes from the 1/0.9 factor in mu_XGP.
  - In the hot 1-1.5 R200c band, the measured mass per SWIFT electron is 1.148-1.153 m_p, against mu_XGP = 1.263 m_p (`text_support.soap_only_conversion['band_1-1.5']`).
  - That mass per electron implies X_H = 0.735-0.743 for fully ionised, metal-free gas.
  - Dividing gas density by mu_XGP therefore undercounts electrons by about 9% in that band.
  - N_e(<4R) is low by 8.5-8.8% in 13.5-15.5, by 7.3-8.4% in 12.7-13.5 and by 6.4% in 12.5-12.7.
  - The deficit is smaller in the resolved cores: -4.9 to -6.1% in the <0.1 band (14.5-15.3) and -6.9 to -7.8% at 0.1-0.25R (13.9-15.3). This is probably because partly neutral gas lowers SWIFT's n_e there too (an interpretation).
- **Full ionisation.** It counts every H and He electron, so it is high wherever gas is partly neutral. That is presumably cold cores, satellite ISM and cool clumps at 3-4R.
  - Integrated over 0-4R, full ionisation exceeds SWIFT n_e by 0.20-0.43% (13.5-15.3). Equivalently, SWIFT n_e is 0.20-0.42% below full ionisation.
  - In 191 halo-shells the gas holds no free electrons at all according to SWIFT (`results.json: electron_zero_shells`). None of these is a strict-resolved shell of a calibrated bin (`text_support.electron_zero_shells`).
  - In several 0.25-1 R200c bands of 13.9-15.3, the SWIFT band medians are up to 0.05% above full H+He ionisation (the largest is 15.1-15.3, 0.25-0.5R). This is probably metal electrons (an interpretation).
- **A single global X.** It cannot follow the radial change of ionisation and metallicity.
  - X = 0.76, the primordial value, overestimates N_e(<4R) by 1.4-1.7%. This agrees with the SOAP audit's +1.4-1.7% for GasMass at >= 10^13.5 in SO apertures.
  - Even the effective X_eff = 0.733 that nulls N_e(<4R) leaves residuals (band medians):
    - +2.8 to +4.1% in the <0.1 band, which is resolved only in 14.5-15.3;
    - -0.5 to -0.1% at 1-1.5R.
  - X_eff is not a physical hydrogen fraction. It absorbs the partly neutral gas and the metals. It was fitted to these same stacks at z = 0 only.

**Recommendation.**
- Calibrate electrons on SWIFT `ElectronNumberDensities`, expressed in XGPaint units.
- Never paint a gas-density fit converted with mu_XGP. This was the convention error behind the -12.6% kSZ in Test D.
- **If per-particle element masses exist but n_e does not** (for example, downsampled particle data), compute full ionisation from each particle's H and He masses. It is accurate to +0.2 to +0.4% on N_e(<4R) (13.5-15.3) and to +0.5 to +1.1% in the resolved <0.1 R200c band (14.5-15.3).
- **At SOAP-only redshifts**, there is no electron number and no hydrogen mass. The conversion then needs one global factor:
  - Do not use full ionisation at X = 0.76 (+1.4 to +1.7%).
  - Use the effective factor mu_XGP (1+X_eff)/2, with X_eff measured from particle data at that redshift (0.733 at z = 0).
  - At z = 0 this gives -0.11 to +0.17% on N_e(<4R) in 13.5-15.3, an in-sample value. It leaves +2.8 to +4.1% in the resolved <0.1 R200c band (14.5-15.3). The unresolved cores of lower-mass haloes were not tested.

### 4. Ownership: which particles in the sphere count

**What it is.** A particle inside 4 R200c of a halo can be:
- bound to the central (HBT+ membership);
- bound to an own satellite;
- unbound;
- bound to another distinct halo, of any mass.

Spheres of neighbouring haloes overlap beyond about 1-2 R200c. The 16 selections tested are:
- **Bound membership:** central-bound (SOAP ExclusiveSphere-like) and host-bound (central + own satellites, FoF-like).
- **Neighbour membership:** no gas bound to other haloes; central-bound + unbound.
- **Overlapper removal:** remove the bound gas of structures of >= 3 / 10 / 30% of M200c reaching inside 2R. Two variants restrict this: one keeps own satellites, the other removes only overlappers that reach inside 1R.
- **A2-i:** remove gas bound to other painted haloes (M200c >= 6.6e12 Msun).
- **A2-ii (the painting partition):** each particle goes to the painted halo in which it has the smallest r/R200c.
- **A2-ii + A2-i.**
- **Excision:** cut spheres of kappa = 0.5 / 1 / 2 R200c,j around overlappers and renormalise by the retained volume; a halo-shell that keeps less than 50% of its volume is left out of the stack. A further variant excises only around neighbours (kappa = 1).

**Where the stacked signal comes from** (`summary.owner_split_integral_share_pct`, cal ranges; figure `groups/selection/figures/owner_split.png`).

| Share of the stack | Inside R200c | 1-2 R200c | 2-4 R200c | 0-4 R200c |
|---|---|---|---|---|
| P: bound to the central | 91-96% | 33-38% | 1.8-3.3% | 48-75% |
| P: own satellites | 0.1-0.8% | 1.0-4.2% | 3.3-12.6% | 1.0-3.0% |
| P: unbound | 3.9-8.7% | 59-64% | 74-90% | 23-49% |
| P: bound to other haloes | 0.0% | 0.0% | 5.0-13.2% | 1.1-2.2% |
| E: bound to the central | 92-95% | 30-35% | 1.8-2.6% | 32-51% |
| E: own satellites | 0.3-1.1% | 1.3-4.1% | 1.9-5.7% | 1.5-3.3% |
| E: unbound | 4.5-7.1% | 62-66% | 82-92% | 43-65% |
| E: bound to other haloes | 0.0% | 0.1-0.3% | 4.0-11.0% | 1.9-3.5% |

**Measured effect** (`summary.variants.<id>.integral_dev_pct` and `impact_pct`). Integrals are cal ranges; impacts are on the 10 calibrated bins. Figures:
- `groups/selection/figures/ownership_band_heatmap_P.png`, `groups/selection/figures/ownership_band_heatmap_E.png`;
- `groups/selection/figures/ownership_membership_ratio.png`, `groups/selection/figures/ownership_neighbour_ratio.png`;
- `groups/selection/figures/profiles_ownership_example.png`.

| Selection | Y(<R200c) | Y(2-4R) | Y(<4R) | N_e(<4R) | tSZ ell 100 / 1000 / 3000 | kSZ 1h ell 1000 | mean DM |
|---|---|---|---|---|---|---|---|
| Central-bound only | -4.2 to -8.8% | -96.7 to -98.2% | -25.0 to -51.7% | -49.3 to -68.3% | -36.3 / -19.8 / -9.0% | -60.0% | -35.3% |
| Host-bound | -3.9 to -8.7% | -84.9 to -94.8% | -24.0 to -50.3% | -46.8 to -66.8% | -32.7 / -18.0 / -8.3% | -57.7% | -34.1% |
| Central-bound + unbound | -0.10 to -0.76% | -8.3 to -23.2% | -2.3 to -4.4% | -3.4 to -6.5% | -4.24 / -1.93 / -0.73% | -3.49% | -2.52% |
| No gas bound to other haloes | 0.00% | -5.0 to -13.2% | -1.1 to -2.2% | -1.9 to -3.5% | -0.70 / -0.12 / +0.03% | -1.17% | -1.35% |
| Remove overlappers, mu >= 0.03 | -0.10 to -0.76% | -1.4 to -9.1% | -0.7 to -2.3% | -0.8 to -2.6% | -3.29 / -1.79 / -0.76% | -1.84% | -0.83% |
| Remove overlappers, mu >= 0.1 | -0.09 to -0.72% | -0.5 to -8.4% | -0.4 to -2.3% | -0.6 to -2.0% | -3.12 / -1.73 / -0.75% | -1.68% | -0.75% |
| Remove overlappers, mu >= 0.3 | 0.00 to -0.34% | 0.0 to -6.2% | 0.0 to -2.0% | 0.0 to -1.6% | -2.17 / -1.05 / -0.43% | -1.14% | -0.56% |
| Remove neighbour overlappers (keep satellites) | 0.00% | 0.0 to -1.5% | 0.0 to -0.26% | 0.0 to -0.17% | -0.03 / -0.01 / +0.005% | -0.03% | -0.03% |
| Remove overlappers reaching 1R | -0.09 to -0.72% | 0.0 to -0.7% | -0.2 to -1.1% | -0.2 to -1.0% | -2.55 / -1.51 / -0.76% | -0.95% | -0.28% |
| A2-i | 0.00% | -4.8 to -13.1% | -1.1 to -2.1% | -1.1 to -3.1% | -0.69 / -0.11 / +0.03% | -0.71% | -0.93% |
| **A2-ii (painting partition)** | **0.00%** | **-20.2 to -29.3%** | **-2.3 to -7.4%** | **-5.1 to -8.3%** | **-1.60 / -0.35 / +0.04%** | **-2.71%** | **-3.23%** |
| A2-ii + A2-i | 0.00% | -20.2 to -29.3% | -2.3 to -7.4% | -5.1 to -8.3% | -1.61 / -0.36 / +0.04% | -2.72% | -3.24% |
| Excision, kappa = 0.5 | -0.5 to -1.7% | -0.2 to -5.1% | -0.6 to -2.0% | -0.3 to -1.1% | -5.23 / -4.58 / -3.09% | -1.11% | -0.33% |
| Excision, kappa = 1 | -1.2 to -2.7% | -0.4 to -10.3% | -1.2 to -3.9% | -0.9 to -2.7% | -7.22 / -5.56 / -3.26% | -2.85% | -0.97% |
| Excision, kappa = 2 | -0.2 to +1.0% | -0.7 to -14.4% | -0.8 to -4.5% | -1.2 to -3.6% | -3.95 / -1.53 / +2.25% | -3.09% | -1.66% |
| Excision of neighbours only, kappa = 1 | 0.00% | 0.0 to -1.6% | 0.0 to -0.23% | 0.0 to -0.20% | -0.04 / -0.01 / +0.004% | -0.03% | -0.02% |

The bound-only rows change the quadratic observables by up to -83% (kSZ 1h at ell = 100). These are first-order values, which overstate changes this large (see "What was measured"). Their mean-DM values are linear and exact for the calibrated bins.

**A2-ii in detail.**
- **Inside 1 R200c** no strict-resolved shell changes at all.
- **At 1-1.5 R200c:**
  - the band medians change by at most 0.0095% (P) and 0.019% (E and G), in 14.9-15.1 (`text_support.A2ii_inside_1p5R`);
  - single strict-resolved shells change by at most -0.053% (P), -0.114% (E) and -0.116% (G), all in the shell at x = 1.42 of 14.9-15.1 (`text_support_v2.A2ii_single_shells`).
- **Outside 1.5 R200c** (band medians, cal):
  - at 1.5-2R it removes 0.24-0.81% of the pressure and 0.17-1.18% of the electrons;
  - at 2-3R it removes 3.6-7.4% of the pressure and 2.4-10.9% of the electrons;
  - at 3-4R it removes 37.8-49.6% of the pressure and 17.6-35.6% of the electrons.
- **Observables** (calibrated bins, paired errors):
  - tSZ -1.602 ± 0.068 / -0.354 ± 0.027 / +0.038 ± 0.010%;
  - kSZ 1h -6.76 ± 0.19 / -2.71 ± 0.12 / +0.24 ± 0.03%;
  - mean DM -3.23 ± 0.11%, DM variance -1.58 ± 0.05%, DM 16th percentile -5.52 ± 0.20%.
- **Resolved-only.** The `resolved_only` impacts equal these values: the removed gas sits in resolved shells.
- **If the 12.7-13.5 haloes were also partitioned.** A first-order estimate gives mean DM -4.29%, DM 16th percentile -7.59% and kSZ (ell = 1000) -3.27%. Those haloes are painted with the fallback profile, so this applies only if that profile is built with the same partition.

**The excisions are sensitive to the top bin.** For kappa = 1, the tSZ change at ell = 1000 is -5.56 ± 0.90%. Of this, -2.34 points come from the 6-halo 15.3-15.5 bin (`text_support.exc_all_k1_tSZ1000_per_bin`).

**Why.**
- **Unbound gas.** The unbound atmosphere carries most of the signal beyond R200c:
  - pressure: 59-64% at 1-2R and 74-90% at 2-4R;
  - electrons: 62-66% at 1-2R and 82-92% at 2-4R.

  Every free electron in it contributes to y, to the kSZ and to the DM, whatever its binding state. The bound-only selections (central-bound, host-bound) remove it, losing 24-52% of Y(<4R) and 47-68% of N_e(<4R). This is not a "contamination" correction: those selections define a different quantity, a subhalo's own gas.
- **Gas bound to other haloes.** It is only 1.1-2.2% of Y(<4R) and 1.9-3.5% of N_e(<4R). None of its pressure lies inside R200c, and almost all of it lies at 2-4R: 5.0-13.2% of the 2-4R pressure, against 0.01-0.05% at 1-2R.
  - Part of it belongs to haloes below the painting threshold, which HalfDome does not paint. Compare no gas bound to other haloes (N_e(<4R) -1.9 to -3.5%) with A2-i (-1.1 to -3.1%).
  - Removing it for all haloes (the "no gas bound to other haloes" variant) therefore loses gas that nobody paints.
- **Overlapper removal and excision.** The variants that move the observables are the seven that remove own-satellite gas: removal at mu >= 0.03 / 0.1 / 0.3, removal of overlappers reaching 1R, and excision at kappa = 0.5 / 1 / 2.
  - They lower tSZ at ell = 100 by 2.2-7.2% (`text_support_v2.ownership_groups`).
  - The neighbour-only versions remove no satellite gas. "Keep satellites" moves tSZ at ell = 100 by -0.034 ± 0.013% and neighbour-only excision by -0.035 ± 0.013%. Both change every observable by at most 0.09%.
  - 997 of the 1,013 haloes whose overlapper reaches inside 1R have no neighbouring overlapper (`text_support.sample_classes`), so in practice that overlapper is an own satellite.
  - In the painting, satellites are not painted separately, so their gas must stay with the host.
  - Excision also cuts host gas around the satellite: SYNTHESIS Test A found a 6-16% overshoot in Y(<1R).
  - The retained-volume renormalisation makes the result depend on kappa non-monotonically. tSZ at ell = 3000 is -3.1% at kappa = 0.5, -3.3% at kappa = 1 and +2.3% at kappa = 2.
- **A2-ii.** HalfDome sums the painted spheres of neighbouring haloes. A FLAMINGO particle that lies within 4R of two painted haloes is in both "all-gas" stacks, so painting those stacks paints it twice.
  - A2-ii gives every particle to exactly one painted halo: the one in which it has the smallest r/R200c.
  - Inside R200c the halo's own centre always won in these stacks, so no resolved shell inside 1R changes. At 1-1.5R single shells change by at most 0.12%.
  - Beyond 2R it removes the overlap with painted neighbours: 38-50% of the 3-4R pressure (band medians).
  - A2-i removes only the bound gas of painted neighbours, not their unbound atmospheres. It therefore captures only a fraction of the overlap: Y(2-4R) -4.8 to -13.1%, against -20 to -29% for A2-ii.
  - Adding A2-i to A2-ii changes nothing: the impacts agree within 0.03 points.

**Recommendation.** Use the A2-ii partition among the painted haloes (M200c >= 6.6e12 Msun). Keep all phases, own satellites and unbound gas, and give gas that is closer in r/R200c to a painted neighbour to that neighbour.
- **What it costs.** A2-ii needs only the painted-halo catalogue: centres and R200c. It needs no HBT+/SOAP membership, which makes it cheap for ~100 snapshots.
- **What it gains.** It removes the double-counted 2-4R gas. In the calibrated bins this shifts mean DM by -3.2% and the DM 16th percentile by -5.5%. SYNTHESIS Test A also found that it makes 4R pressure fits adequate: sigma_d falls from 8-14% to 0.2-3.9%.
- **What to avoid.** Do not use bound-only membership (SOAP ExclusiveSphere-like), overlapper removal or excision for painting.

### 5. Halo sample (CLEAN vs overlapped haloes)

**What it is.** The reference uses all 4,588 centrals with their inclusion weights. The variants restrict each bin to a class:
- **CLEAN** (class_main = 0): no structure of >= 10% of M200c reaches inside 2 R200c;
- **overlapped** (class_main > 0);
- **overlapper inside 1R** (class_main = 1);
- **overlapper in the 1-2R annulus only** (class_main = 2);
- **CLEAN with stricter or looser thresholds:** mu >= 0.03 or mu >= 0.3.

**Composition.**
- CLEAN haloes carry 66-81% of the inclusion weight in 13.5-15.3, but only 2 of the 6 haloes in 15.3-15.5 (`subset_composition`).
- The weight shares of the other subsets in 13.5-15.3 are:
  - overlapped: 19-34%;
  - overlapper inside 1R: 10-24%;
  - annulus only: 9-11%;
  - CLEAN at mu >= 0.03: 22-29%;
  - CLEAN at mu >= 0.3: 91-100%.

**Measured effect** (`summary.variants`, `text_support.impact_significance`, `text_support.clean_per_bin`, `text_support_v2.clean_per_bin_groups`; figure `groups/selection/figures/sample_overlap_classes.png`). Integrals are cal ranges; impacts are on the 10 calibrated bins.

| Subset | Y(<R200c) | Y(2-4R) | N_e(<4R) | tSZ ell 100 / 1000 / 3000 | kSZ 1h ell 100 / 1000 / 3000 | mean DM | DM variance |
|---|---|---|---|---|---|---|---|
| CLEAN | +0.5 to +2.1% | -18.4 to -1.0% | -4.2 to -0.5% | +0.80 ± 1.57 / +1.72 ± 1.82 / **+4.16 ± 1.00%** | -4.54 ± 0.37 / -2.52 ± 0.34 / +0.68 ± 0.27% | **-1.93 ± 0.20%** | -0.30 ± 0.25% |
| CLEAN at mu >= 0.03 | +3.3 to +6.8% | -30.0 to -10.6% | -7.4 to -0.3% | +6.6 ± 1.9 / +9.9 ± 2.2 / +15.3 ± 1.5% | -5.40 ± 0.63 / -2.03 ± 0.55 / +4.13 ± 0.52% | -3.01 ± 0.38% | +2.88 ± 0.46% |
| CLEAN at mu >= 0.3 | -1.3 to +0.2% | -13.7 to 0.0% | -3.6 to 0.0% | -1.64 ± 0.94 / -0.75 ± 1.1 / -0.04 ± 0.55% | -3.39 ± 0.25 / -2.56 ± 0.23 / -0.82 ± 0.14% | -1.52 ± 0.13% | -1.38 ± 0.14% |
| Overlapped | -4.2 to -1.0% | +1.9 to +44.7% | +2.0 to +8.7% | -0.30 ± 1.35 / -5.03 ± 1.4 / -10.1 ± 1.3% | +9.70 ± 0.77 / +5.07 ± 0.69 / -1.52 ± 0.54% | +3.97 ± 0.41% | +0.58 ± 0.50% |
| Overlapper inside 1R | -6.6 to -3.0% | -34.9 to +10.2% | -8.2 to +2.2% | -5.68 ± 1.69 / -10.5 ± 1.8 / -17.0 ± 1.8% | +1.02 ± 1.06 / -0.63 ± 1.02 / -4.53 ± 0.78% | +0.74 ± 0.57% | -4.24 ± 0.72% |
| Overlapper in 1-2R only | -3.2 to +7.8% | +33.9 to +156% | +12.3 to +25.2% | +7.83 ± 2.03 / +3.75 ± 1.9 / +2.62 ± 1.9% | +28.6 ± 1.3 / +18.3 ± 1.1 / +5.49 ± 0.80% | +11.4 ± 0.70% | +11.4 ± 0.70% |

**Which CLEAN shifts are significant** (against the halo-bootstrap errors).
- **Significant:**
  - tSZ at ell = 3000 (4.2 sigma);
  - mean DM (9.5 sigma);
  - kSZ at ell = 100 and 1000 (12 and 7 sigma);
  - mean y (-2.05 ± 0.29%, 7 sigma).
- **Consistent with zero:**
  - the tSZ shifts at ell = 100 (0.5 sigma) and ell = 1000 (0.9 sigma);
  - the DM variance (1.2 sigma).
- **Marginal:** kSZ at ell = 3000 (2.5 sigma).
- **Where the ell = 100 shift comes from.** The net +0.80% is a cancellation: -1.37 points from 13.5-14.9 against +2.17 points from 14.9-15.5. The +2.17 is the sum of:
  - +1.15 from 14.9-15.1;
  - +0.16 from 15.1-15.3;
  - +0.86 from the 15.3-15.5 bin, which holds 2 CLEAN haloes.

  Without the top bin the shift is -0.06%.
- **ell = 3000.** The +4.16% comes mostly from 14.3-15.1 (+3.99 points). It does not depend on the top bin: +4.30% without it.

**Why.**
- **CLEAN haloes differ from the population at fixed M200c** (band medians, cal):
  - their inner pressure is higher, by 1.8-9.5% in the <0.1 band (resolved only in 14.5-15.3) and by 2.4-7.3% at 0.1-0.25R (13.9-15.3);
  - at 1.5-3R their pressure is lower (P -3.0 to -20.1%) and they have less gas (G -2.4 to -10%).
- **Overlapped haloes are the mirror image.**
  - For an overlapper inside 1R, the inner pressure is lower: by 20-29% in the <0.1 band (14.5-15.3) and 16-28% at 0.1-0.25R (13.9-15.3).
  - The annulus class has more gas at 1-4R: G +2.4 to +12.9% at 1-1.5R and +16 to +80% at 1.5-4R.
- **The inner difference is selection, not contamination** (SYNTHESIS Test A). A satellite inside R200c inflates M200c, so the host's own atmosphere is normalised to too large a mass. Removing the satellite's gas does not lift the deficit; expressing the mass in host-only units removes it.
- **The outer difference is real neighbour gas.** The A2-ii partition, not a sample cut, handles it (section 4).
- **The CLEAN definition is arbitrary, and the bias follows it.** At ell = 3000 the bias is:
  - +15.3 ± 1.5% at mu >= 0.03;
  - +4.16 ± 1.00% at mu >= 0.1;
  - -0.04 ± 0.55% at mu >= 0.3.

  A bias that depends on an arbitrary threshold is a property of the sample, not of the gas.
- **HalfDome paints every halo,** overlapped ones included, in the proportion the inclusion weights reproduce.

**Recommendation.** Calibrate on all centrals with their inclusion weights. Do not select clean haloes and do not subtract overlappers.
- Calibrating on CLEAN haloes would bias tSZ at ell = 3000 by +4.2 ± 1.0% and mean DM by -1.9 ± 0.2%, and lower kSZ 1h by 2.5-4.5% at ell = 100-1000.
- Its tSZ shifts at ell = 100 and 1000 (+0.8 ± 1.6 and +1.7 ± 1.8%) are not significant. At ell = 100 the shift is a cancellation between 13.5-14.9 (-1.37 points) and 14.9-15.5 (+2.17 points), and it is -0.06% without the 6-halo top bin.
- With the stricter mu >= 0.03 threshold the tSZ bias at ell = 3000 reaches +15.3 ± 1.5%.

### Caveats

- **First-order impacts, calibrated bins only.** All impacts are first order, from Test F's importance Jacobians, and only the calibrated bins 13.5-15.5 respond.
  - Those bins carry 97.5-98.6% of the tSZ importance but only 58% of mean DM, 83% of kSZ at ell = 1000 and 76% of the DM variance.
  - The painted fallback bins 12.7-13.5 carry most of the rest. For choices applied to every painted halo, the all-painted-bins estimates are larger: for example, mean DM -8.2% instead of -5.0% for gas/mu_XGP.
  - For large changes in the quadratic observables, the first order overstates the effect: a uniform -50% gives -100% instead of -75%. For the bound-only and central-bound X-ray selections (tSZ and kSZ changes of up to -84%), these values show direction and scale only. Mean y and mean DM are linear and exact for the calibrated bins.
- **Bootstrap errors.** The errors are halo-bootstrap SDs. SYNTHESIS Test B's spatial jackknife is 1.11x the bootstrap per shell at 2-4R and 1.16-1.26x for pressure parameters. The quoted errors of effects beyond 2R are therefore lower limits.
- **Redshift.** Everything is at z = 0 in L1_m9. The AGN-heated and cold/SF shares, the A2-ii overlap and X_eff will change with redshift and feedback variant, so they should be re-measured per snapshot, which is cheap.
- **Star-forming gas.** Its n_e is model-dependent (effective equation of state). The X-ray-only variant bounds its effect in the calibrated bins: 1.6% on kSZ and 0.8% on mean DM. With all painted bins it is larger, about 2.6% on kSZ at ell = 1000 and 2.2% on mean DM (first-order estimate), because low-mass haloes hold more cold/SF electrons.
- **Where the heated gas sits.** The location fractions come from the stacked profiles. Attributing the outer heated gas to satellite and neighbour AGN is an inference.
- **Small bins.** The 15.3-15.5 bin has 6 haloes (2 CLEAN), and the 15.1-15.3 bin has 21. Bins below 10^13.5 are exploratory.
- **Website resolution flags.** On the website, subset records flag resolution using the subset's own rows. Over the six subset variants and all 15 bins, 50 shells per field (23 of them in the calibrated bins 13.5-15.5) are therefore shown as resolved although they fail the full-bin strict mask that `results.json` band ratios use (`verify/v6b_extra.json`).
