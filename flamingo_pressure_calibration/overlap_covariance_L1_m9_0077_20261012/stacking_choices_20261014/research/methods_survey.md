## Ways to obtain 1D stacked profiles

**Scope.** This section lists every route from FLAMINGO data to a 1D stacked profile of electron pressure P_e, electron density n_e, gas density rho_gas or temperature T. For each route it gives the input parameters, the cost, and what it changes when the profile is used to paint tSZ, kSZ and FRB DM onto the HalfDome catalogue.

**Data and sources.** Unless stated otherwise, numbers are for FLAMINGO L1_m9 HYDRO_FIDUCIAL at z = 0 (4,588 centrals). Sources are marked as follows:
- **[MN]**: new numbers in `research/methods/methods_numbers.json`, produced by `methods_numbers.py` from the saved particle-pass products (14 s on one core).
- **A-G**: the study's earlier tests (`SYNTHESIS.md` and the per-test `REPORT.md` files).
- **[SOAP]**: a read-only listing of `SOAP-HBT/halo_properties_0077.hdf5` on maas (`soap_0077_group_listing.json`).
- **Papers**: each was checked on arXiv on 2026-10-14.

### Summary

- **Six decisions define a stacked profile.** They are the data source, the per-halo estimator, the gas selection, the radial coordinate and normalisation, the stacking statistic with its mass binning, and the post-processing (Table 1).
- **One combination is unbiased by construction for painting:**
  - the geometric volume average of linear particle sums (sum y, sum n_e V or sum m, divided by the geometric shell volume);
  - stacked as an inclusion-weighted mean of the normalised per-halo profiles;
  - in narrow bins of the painter's own mass definition;
  - normalised exactly as XGPaint normalises.
- **Every other common estimator or statistic measures something else** (sizes are in Sections 3-4):
  - SPH-volume weighting reads +3% to +25% high, growing with radius.
  - Mass and X-ray weighting read 1.1 to 20 times high.
  - Medians fall 5-55% below the mean beyond 1.3 R200c.
  - Y200 weighting raises the core by 2-8%.
- **SPH-kernel deposition and particle-centre binning differ by at most 1.2%** in every shell that the resolution mask accepts (log M200c >= 13.5). The large differences, up to +-78%, occur only in shells the mask already rejects. The mask's "inner edge >= 2 smoothing lengths" rule is what makes the two methods agree.
- **Projected profiles need a deep line of sight.** A +-1 R200c slab loses 18-26% of the projected pressure and 28-35% of the projected electrons at R200c.
- **Onion-peeling deprojection is accurate inside R200c.** It recovers the 3D shells there to <= 1.2% (P_e) and <= 2.8% (n_e), even with its outer boundary at 2 R200c. It piles the excluded outer gas into the last shell (x2.05-2.53).
- **Fastest routes, in order:**
  1. **Saved products.** This project's z = 0 products rebuild any estimator, gas selection, statistic or sub-grid in seconds.
  2. **SOAP apertures.**
     - SOAP gives exact cumulative Compton Y, Doppler B and gas mass at about 8 distinct radii between about 0.27 and 3.35 R200c.
     - It covers every SOAP snapshot and every feedback and cosmology variant. Reading one column for all 15.3M rows takes 122 MB, with no particle reading.
     - Its limits: no radius inside 0.27 or beyond 3.35 R200c, and no electron-number column.
  3. **Reduced snapshots** stop at R100c, about 1.35 R200c.
  4. **The 1% downsampled snapshot** is too sparse for per-halo profiles below about 10^14.3 Msun.

### 1. The six decisions (Table 1)

| # | Decision | Options (literature and FLAMINGO products) | This project | Largest measured effect of an alternative |
|---|---|---|---|---|
| 1 | Data source | full-snapshot particles; reduced snapshots; 1% downsampled snapshot; SOAP apertures; lightcone or projected maps; published fits and emulators | full-snapshot particles (saved linear sums) | Section 2 |
| 2 | Per-halo estimator | geometric; SPH-volume Q_V; mass-weighted Q_M; X-ray-weighted Q_X, n_e kT_X and T_sl; SPH-kernel deposition; synthetic X-ray spectroscopy | geometric | B24's Q_X recipe: yy 1-halo +80% at ell = 100 (D) |
| 3 | Gas selection | phase cuts (recent AGN, star-forming, T < 1e5 K); ownership (sphere, FoF, bound, host-bound, exclusive, partition); centre; extraction radius | all gas with own satellites; A2-ii partition; recent-AGN gas removed; most-bound-particle centre; r_ext = max(4 R200c, 6 R500c) | a 2R cut: kSZ -26%, DM -22% (F); A2-ii vs all gas: Y(<4R) -2.4 to -7.4% (A); AGN cut <= 0.04% (F) |
| 4 | Radius, normalisation, grid, mask | R200c, R500c, R200m, BN98, splashback; P200 and P500 variants; 27-67 log shells; masks on softening, smoothing length, particle number and occupancy | R200c / P200; 67 shells to 4 R200c; 3-part mask | A10 universal shape with M^0.12: +29% (D); B16 gas density with mu_XGP: electrons -9% (D) |
| 5 | Statistic and binning | mean, median, log-mean, aggregate, Y-weighted; bins 0.05-0.5 dex; M200c or M500c; Msun or h^-1 Msun; minimum haloes per bin | inclusion-weighted mean in 0.2-dex M200c bins | median: -39 to -55% at 3.6-4 R200c [MN] |
| 6 | Post-processing | fit form (B12/B16, gNFW, two-component); fit range and weights; truncation shape; scatter model | B12/B16 plus an outer term, fitted to 4R; hard cut at 4R; lognormal scatter | a 2R fit is 31-37% low at 2-4R (B) |

### 2. Data sources (where the particles or integrals come from)

| Source | Content | Radial reach | Availability | Cost | Use |
|---|---|---|---|---|---|
| Full snapshots (STRW maas) | all gas fields: ComptonYParameters, ElectronNumberDensities, Temperatures, Densities, SmoothingLengths, XrayLuminosities, StarFormationRates; membership files | any | fiducial L1_m9 at z = 15, 2, 1.5, 1, 0.8, 0.5, 0.3, 0.2, 0.15, 0; feedback variants only at 0074/0077 (+0000), without membership; HYDRO_PLANCK at 0077 | 545 GB per snapshot (65 files), 5.27e9 gas particles, 51 B/particle stored and 140 B logical. Our pass: 66 min wall, 6 workers, 5.5 CPU-h, 308 GB logical, 2.2e9 rows (42% of the gas), decompression-bound | calibration |
| Saved pass products (this study) | per-halo linear sums [4,588 x 17 owners x 3 phases x 16 quantities x 67 shells], X500 grid, partition, excision, apertures, smoothing-length histograms | 4 R200c (6 R500c) | z = 0, fiducial only | seconds | every z = 0 choice in Sections 3-6 |
| SOAP-HBT catalogues | SO integrals at 9 radii; fixed-radius spheres; projected apertures | about 0.27-3.35 R200c, cumulative only | fiducial outputs 0007-0077; feedback variants at 0017, 0027, 0037, 0047, 0057, 0067, 0075, 0077; cosmology variants (SOAP only) | 36.7 GB per file, 15.3M rows; one float64 column is 122 MB | integrals; redshift and variant trends |
| Reduced snapshots (COSMA service) | all particles within R100c of <= 200 haloes per 0.05 dex above 1e13 Msun | R100c, about 1.35 R200c | L2p8_m9 (all 79 outputs), L1_m8 | 485M gas particles at z = 0 (L2p8); local link 0.3-6 MB/s | inner shape vs z |
| Downsampled snapshot (maas, z = 0) | 1% of the gas: Coordinates, Masses, Velocities, ComptonYParameters | any | z = 0 | 1% of the particle I/O | aggregate y stacks of massive haloes only |
| Lightcone HEALPix maps | per-shell ComptonY and DM (kSZ in the native maps) | projected, includes the 2-halo term | L1_m9: 60 shells (Delta z = 0.05) to z = 3 at NSIDE 4096 | 1.61 GB per map per shell (project memory note) | closure tests |
| `mcdonald_snapshots_reduced` (z = 0) | no gas | - | - | - | unusable |
| Published fits and emulators | parametric profiles | - | - | free | fallback below the calibrated mass range |

### 3. (a) Per-halo 3D estimators

**Formulae.**
- Pressure per particle: P_e,i = y_i m_e c^2 / (sigma_T V_i).
- SPH volume: V_i = m_i / rho_i.
- Shell volume: V_shell = (4 pi/3) R^3 (x_out^3 - x_in^3).

| Estimator | Measures | Extra inputs and parameters | Cost | Bias relative to the geometric estimator (FLAMINGO z = 0) | Pros | Cons | For painting |
|---|---|---|---|---|---|---|---|
| **Geometric** (sum over particles in the shell, divided by V_shell); particle-centre binning | Volume average of a linear quantity. Its line-of-sight integral is y, tau or DM | ComptonYParameters (or n_e V, or m); centre; R_Delta; grid edges | one pass (Section 2) | reference. Particle Y(<R500c) matches SOAP to a median 2.0e-5; 98.5% of haloes agree within 0.3% (D) | linear: merges and rebins exactly; conserves Y and N_e | shot noise in sparse shells, so it needs a mask | **use** |
| **SPH-volume Q_V** = sum q V_i / sum V_i (B24, eq. 8) | volume-weighted mean over the particles' own volumes | Densities | same pass | +3-5% inside 1R, +6-7% at 1-2R, +13-14% at 2-3R, +18-25% at 3-4R. Through the fits: C_ell^yy +6-8%, mean DM +4.3% (B) | comparable with B24 | the summed SPH volumes fill only 0.94-0.97 of the shell at 0.09-1 R200c, 0.85-0.86 at 2.5-3 R200c and 0.77-0.79 at 3.6-4 R200c (log M200c >= 13.9) [MN] | no |
| **Mass-weighted Q_M** (B24, eq. 10) | mass-weighted mean; SOAP's GasTemperature is mass-weighted | Masses | same | pressure +13-38% at 0.5-1R, x1.6-2.6 at 1-2R, x7-20 at 2-3R; density x7-10 at 0.1-0.5R and x300-390 at 1-2R (B). B24's Q_M recipe painted: yy +29/+16/+8% at ell = 100/1000/3000 (D) | correct for T_mw | dominated by dense clumps | not for P_e or n_e |
| **X-ray-weighted Q_X** (B24 fiducial, eq. 9); **n_e x T_X**; **spectroscopic-like T_sl** (weights rho m T^-3/4; Mazzotta et al. 2004) | what X-ray observations see | XrayLuminosities (ROSAT 0.5-2 keV); an X-ray phase cut | +24 B/particle | Q_X pressure +17-36% at 0.1-0.5R, x1.3-2.4 at 0.5-1R, x3-10 at 1-2R. n_e kT_X: within 1% inside 0.5R, +10-42% at 2-3R (B). The particle L_X-weighted T_X is +8.8% above SOAP's T_sl at R500c, and +20-30% in groups (D). B24's Q_X recipe painted: yy +80/+42/+23% (D) | like-for-like with X-ray data | strongly biased high for tSZ | no |
| **Synthetic X-ray spectroscopy** with a hydrostatic R500 (Le Brun et al. 2014; McCarthy et al. 2014, 2017) | biased X-ray observables | spectral code, instrument response | much higher (spectral fits per bin) | not computed | like-for-like with X-ray samples | measures hydrostatic- and spectroscopy-biased quantities | no |
| **SPH-kernel deposition** onto shells or a 3D grid (Wendland C2, support H = 1.936492 h in SWIFT) | same as geometric, with each particle spread over its kernel | SmoothingLengths; padding >= H beyond r_ext | 67 x 96 x 68 transfer matrix built in 1.6 s; 3D grids cost ~N_part (H/cell)^3 | see the table below | correct where the kernel exceeds the shell; standard for maps | needs h and padding | for maps and cores only |
| **Mesh-code cell sums** (TNG Voronoi cells, Lee22; ART AMR, Nagai et al. 2007) | the geometric estimator for mesh codes | cell volumes | - | - | - | - | for literature comparison only |

**Kernel deposition vs particle-centre binning [MN].** This is an estimate, not a particle re-read. It uses the stored per-shell smoothing-length histograms, places particles at volume quantiles of their shell, and gives each particle the shell's mean value.

| log M200c | First shell the mask accepts (coarse grid, R200c) | Largest change in accepted shells, P_e / n_e | Change in rejected inner shells, P_e | Last shell 3.63-4 R200c, P_e (edge effect) |
|---|---|---|---|---|
| 12.9-13.1 | 1.75 | 2.4% / 4.2% | -78% to +78% | -27% |
| 13.5-13.7 | 0.42 | 1.1% / 1.1% | -7% to +53% | -18% |
| 13.9-14.1 | 0.19 | 1.2% / 0.9% | -5% to +22% | -13% |
| 14.3-14.5 | 0.13 | 1.1% / 0.8% | -5% to -1% | -9% |
| 14.7-14.9 | 0.059 | 0.9% / 0.6% | -6% to -2% | -6% |
| 14.9-15.5 | 0.059 | 0.7% / 0.6% | -4% to -1% | -5% |

- In every accepted shell, 0% of the particles have a kernel support H larger than the shell's inner radius. One shell further in, up to 15% do. The mask therefore removes exactly the radii where binning by particle centre and by kernel disagree.
- The last-shell deficit is an artefact of the extraction radius. Kernel deposition needs particles collected to r_ext + H.

**Temperature definitions** (all available from the saved sums):
- T_mw = sum m T / sum m;
- T_V = sum V T / sum V;
- T_X (L_X-weighted);
- SOAP's T_sl and GasComptonYTemperature (Compton-y-weighted);
- T_e = P_e / (n_e k) from the two geometric profiles. This is the electron-number-weighted temperature, the one that links the painted pressure and electron profiles.
- Measured: T_X/T_mw = 1.24 at 2.5 R200c (B). T_e/T200 falls from 0.43 to 0.30 at 1.5-2R, and from 0.34 to 0.10 at 3-4R, between 10^13.6 and 10^15.35 Msun (G).

**Clumping.**
- Clumping factors are C_rho = <rho^2>/<rho>^2 and C_P = <P^2>/<P>^2. C_P is 1.04 at 0.1R and 2.8 at 1.75R (B).
- Clump-robust estimators (per-shell medians of particle values, clump excision; Nagai & Lau 2011) suit X-ray comparisons. They are wrong for tSZ and DM, because clumps carry real y and electrons.
- Excising overlappers overshoots Y(<1R) by 6-16% on real haloes (A).

### 4. (a) Stacking statistics

**How the comparison was made [MN].**
- Same haloes, geometric estimator, all gas, recent-AGN gas removed, 300 paired halo bootstraps.
- Entries are the ratio to the inclusion-weighted mean, minus 1, in %. Each entry is the range over the bins 13.5-13.7, 13.9-14.1, 14.3-14.5, 14.7-14.9 and 14.9-15.5, using only the shells the mask accepts.
- An aggregate (particle) stack in scaled units, which normalises each particle by its host's scale, equals the M200c-weighted mean, because V_shell scales as M. The raw ratio sum y / sum (V P200) equals the M^(5/3)-weighted mean.

| Statistic | Field | 0.09-0.28 | 0.28-0.61 | 0.61-1 | 1-2 | 2-3 | 3-4 R200c |
|---|---|---|---|---|---|---|---|
| median | P_e | -6 to +2 | -3 to +1 | -3 to 0 | -17 to +1 | -34 to -13 | -55 to -25 |
| median | n_e | -7 to 0 | -5 to +1 | -1 to +1 | -6 to 0 | -13 to -5 | -17 to -5 |
| log-mean exp<ln f> | P_e | -8 to -3 | -2 to -1 | -1 | -15 to -3 | -27 to -16 | -49 to -23 |
| log-mean | n_e | -5 to -2 | -2 to 0 | -1 to 0 | -4 to -1 | -7 to -4 | -10 to -5 |
| aggregate = M-weighted | P_e | 0 to +2 | 0 to +1 | 0 to +1 | -1 to 0 | -1 to 0 | -4 to +1 |
| raw aggregate = M^(5/3)-weighted | P_e | 0 to +4 | -1 to +1 | 0 to +1 | -2 to +1 | -2 to +1 | -7 to +2 |
| Y200-weighted (B12) | P_e | +2 to +8 | 0 to +5 | 0 to +3 | -2 to +2 | -2 to +2 | -2 to +3 |
| Y200-weighted | n_e | +1 to +7 | 0 to +6 | 0 to +4 | -1 to +3 | 0 to +2 | 0 to +1 |

- **Statistical precision of the mean.** With about 320 haloes per 0.2-dex bin, the bootstrap 16-84% half-width of the pressure mean is 0.8-1.3% at 0.6-1R, 5-9% at 2.5-3R and 9-19% at 3.6-4R. For electrons it is 0.5-0.9% at 0.6-1R and 1.7-3.2% at 2.5-4R.
- **Medians are biased beyond 1.3 R200c.** The pressure median/mean ratio at 3.6-4R (0.45-0.61) lies far outside these errors.
- **Low mass breaks medians and log-means (12.9-13.1).**
  - 52-58% of haloes have no gas particle in the shells at 0.025-0.087 R200c, so the median there is 0.
  - A log-mean that drops the empty shells is biased high: +41% for electrons at 0.09-0.13 R200c.
  - Y200 weighting is +18 to +31% at every radius.
- **Painting a median instead of the mean** costs -7.4 to -7.6% of Y(<4R) and about -11% of the 2-halo term at ell = 100 (D).
- **B12's Y200 weighting** adds +1.6/+2.4/+3.4% to the yy 1-halo term at ell = 100/1000/3000 (D).
- **A single mean profile carries no halo-to-halo scatter.** It leaves the 1-halo tSZ 1.5% (ell = 100) to 7.3% (ell = 8000) low, so a separate lognormal scatter model is needed (F).

### 5. (b) Radius, normalisation and mass binning

**Radius scaling: R200c vs R500c [MN].**
- Within-bin scatter sd(ln f) and the mass slope d ln<f>/d log10 M200c are measured on shells matched in physical radius (R500c/R200c = 0.646-0.674). Each profile uses its own normalisation (P200, or B24's P_e,500).
- Fits cover the 0.2-dex bins 13.5-14.9 plus the pooled bin 14.9-15.5.

| Shell (~R200c) | sd ln P_e: R200c / R500c | sd ln n_e: R200c / R500c | d ln<P_e>/d log M: R200c / R500c | d ln<n_e>/d log M: R200c / R500c |
|---|---|---|---|---|
| ~0.26 | 0.27 / 0.17 | 0.22 / 0.19 | +0.52 / +0.51 | +0.89 / +0.88 |
| ~0.56 | 0.12 / 0.19 | 0.09 / 0.11 | +0.11 / +0.13 | +0.40 / +0.41 |
| ~1.0 | 0.25 / 0.40 | 0.15 / 0.21 | -0.28 / -0.22 | -0.06 / -0.01 |
| ~1.7 | 0.44 / 0.54 | 0.25 / 0.29 | -0.59 / -0.58 | -0.33 / -0.32 |

- **R500c scaling reduces the core scatter; R200c is tighter everywhere beyond about 0.5 R200c.**
- **Neither scaling removes the mass trend.** The profile shape changes with mass by +-0.5 in ln per dex in either radius unit. A universal shape (A10) therefore fails: +29% on yy at ell = 100 (D).
- **R200m and splashback.**
  - Outskirts are more self-similar in mean-density units. Diemer & Kravtsov 2014 find the steepest DM slope at about 1-1.2 R200m, and Lau et al. 2015 find gas outskirts self-similar in mean-density units.
  - Here R200m is about 1.63-1.70 R200c (NFW estimate), and Test G found the gas transition at x_t = 2.1-2.4 R200c (1.1-1.6 R200m).
  - This would need a new particle binning, because only X200 and X500 grids are stored.
  - The painter's mass definition should set the grid: HalfDome uses M200c, so R200c.

**Normalisation conventions (all verified).** A convention mismatch acts as a systematic error.

| Convention | Definition | Source |
|---|---|---|
| P200 | G M200 200 rho_c f_b / (2 R200) | Battaglia et al. 2012 (arXiv:1109.3711) |
| P500 (N07) | n_g,500 k T500 with mu = 0.59 and mu_e = 1.14; 1.45e-11 erg cm^-3 (M500 / 1e15 h^-1 Msun)^(2/3) E(z)^(8/3) | Nagai et al. 2007 (arXiv:astro-ph/0703661) |
| P500 (A10) | 1.65e-3 E(z)^(8/3) (M500 / 3e14 h70^-1)^(2/3) h70^2 keV cm^-3, times (M500/3e14)^(0.12 + alpha'_P(x)) | Arnaud et al. 2010 (arXiv:0910.1234) |
| P_e,500 (B24) | f_B = 0.159, mu = 0.59, mu_e = 1.14 | Braspenning et al. 2024 (arXiv:2312.08277) |
| rho_gas / rho_crit(z) | B16 density | Battaglia 2016 (arXiv:1607.02442) |
| n200 | 200 rho_cr f_b / (X_H m_p) | Lee et al. 2022 (arXiv:2205.01710) |
| XGPaint electrons | n_e mu_XGP / (rho_c f_b); cosmic mean 0.335 = 1.094 Omega_m | G |

- Painting B16's gas density with mu_XGP gives 9% fewer electrons, kSZ -12.6% and mean DM -5.3% (D).
- Self-similar redshift scaling (E(z)^(8/3) for P500) is a convention. B12 and Lee22 fit (1+z)^alpha_z on top of it.

**Mass binning.**
- **Bin width.** In 0.2-dex bins the implicit weighting (mean vs M-weighted) matters at <= 2% (-4% at 3.6-4R in 13.5-13.7) [MN]. B24's 0.5-dex M500c bins cost -0.8% of Y(<4R) and -2.2% of the 2-halo term at ell = 100. 83% of the haloes in the 13.5-13.7 engine bin come from the 13.0-13.5 M500c bin (D).
- **Minimum haloes per bin.** Lee22 require more than 29 haloes per bin, which drops our two top h^-1 bins (17 and 2 haloes) (D). Kugel et al. 2023 use adaptive bins with at least 10 objects.
- **Stratified samples need inclusion weights.** Here the sample has 300 random haloes per 0.2 dex below 10^14.5 and all haloes above.
- **Bin-centre vs mean mass.** Assigning the stack to the mean of log M (or fitting a continuous mass model, as Test G's ladder does) avoids the bin-centre bias that a steep mass dependence would otherwise produce.

**Secondary properties.**
- Lee22 stack by concentration quartiles. Diemer & Kravtsov 2014 and Lau et al. 2015 bin by accretion rate.
- In FLAMINGO the core amplitude correlates with ln c200c (+0.41 to +0.90). It anti-correlates with the satellite mass fraction (-0.38 to -0.57). The 2-4R amplitude correlates with the neighbour mass within 4R (+0.34 to +0.51) (`shell_correlations_20261013/physical_drivers.json`).
- This is useful only if the painted catalogue carries the property. HalfDome's RFoF catalogue does not, and concentration terms did not help painting (G).

### 6. (c) Particle selection, centring, resolution and extraction radius

| Choice | Options | Inputs and parameters | Measured effect | Recommendation |
|---|---|---|---|---|
| Recently AGN-heated gas | keep; remove | SOAP filter [SOAP attributes]: last AGN event a >= 0.998956 (15 Myr at z = 0) and T within 10^-1 to 10^0.3 of Delta T_AGN = 8.80e7 K (8.80e6-1.756e8 K) | every observable <= 0.04% (F) and <= 0.26% in any literature recipe (D). When the gas is kept, only the innermost unresolved shell moves: +73 / +28 / +4 / +0.6% at log M 13.5-13.7 / 13.9-14.1 / 14.5-14.7 / 14.9-15.1 | remove (as FLAMINGO's y maps do); immaterial |
| Star-forming and cold gas | keep; remove SFR > 0; remove T < 1e5 K | StarFormationRates, Temperatures | all 503,296 star-forming particles in the spheres have y = 0 (apertures diagnostics). The B24 X-ray selection vs all gas changes Q_V by <= 0.23% for log M500c >= 13.5, but by up to +17% (0.15-0.5 R500c) and +190% (0.08-0.15 R500c) in 1e13-1e13.5 cores (D) | keep (no y anyway); matters only for densities in group cores |
| Wind particles | - | - | FLAMINGO's gas output has no separate decoupled-wind class (assumption, not checked); in TNG, wind particles are stored with the stars | - |
| Ownership | all gas in the sphere (SOAP SO convention); FoF; bound to the central (HBT+/SUBFIND); host-bound (central + own satellites); exclusive spheres; partition between painted neighbours (A2-ii) | membership files (GroupNr_bound), SOAP HostHaloIndex, painted-neighbour list (M200c >= 6.6e12) | A2-ii equals all gas to <= 1.4% inside 2R. Beyond 2R it removes 2.4-7.4% of Y(<4R) and 5-7.6% of N_e(<4R) that would otherwise be painted twice. Removing all satellite gas moves P0 by -7 to -26%. Clean-only samples bias yy by +0.7 / +2.1 / +4.1 / +7.9% at ell = 100 / 1000 / 3000 / 8000 (A) | all gas with own satellites, A2-ii |
| Centre | most-bound particle (HBT+/SOAP HaloCentre); shrinking-sphere centre of mass (B12); most-bound black hole (McCarthy et al. 2025); BCG or X-ray peak | SOAP InputHalos/HaloCentre, BoundSubhalo/CentreOfMass | the bound centre of mass lies 0.087 R200c from the most-bound particle (median; 84th percentile 0.15-0.16, 99th 0.31-0.35), at every mass (`selected_halos.npz`). The effect on the profiles was not tested | match the painted catalogue's centre definition |
| Resolution mask | inner edge >= max(0.04, 4 eps/R); >= 20 particles; inner edge >= 2 x upper-median h (this project); >= 16% of haloes with >= 1 particle (B24); > 29 haloes per bin (Lee22); convergence radius (Power et al. 2003, N-body) | softening 5.7 physical kpc = 0.0027-0.017 R200c from 10^15.2 to 10^12.5; per-shell counts and h histograms | first accepted shell: 0.52 R200c at 13.3-13.5 down to 0.040 at 15.1-15.5 (D). Median N_gas(<R200c) per halo: 207 (12.9-13.1), 2,239 (13.5-13.7), 8,460 (13.9-14.1), 24,817 (14.3-14.5), 65,490 (14.7-14.9) [MN]. Unresolved radii carry <= 0.7% of the full-recipe C_ell (D). Changing 2h to 1h moved Y(<4R) by -5% / -3% / +3.5% at 10^13 / 10^14 / 10^15 in the L2p8 pilot (project memory note, 2026-09-27) | keep the 3-part mask; extrapolate the fitted form inward |
| Extraction radius and truncation | r_ext; hard cut; sigmoid or Gaussian taper (B12: 6 R500); Y to 5 R500 (A10) | r_ext = max(4 R200c, 6 R500c) | cut = fit = 4R is the only Pareto-optimal choice. A 2R cut costs tSZ -5%, kSZ -26%, DM -22% (F). A 0.05-dex sigmoid differs from a hard cut by <= 0.21% (C) | 4 R200c |

### 7. (d) Projected and observable-space profiles

**Projection depth [MN].**
- Projected profiles are computed from the stacked 3D means, integrating along the line of sight within |l| < D. Values are the fraction of the full projection through the 4 R200c sphere, over log M200c 13.5-15.5.

| Projected radius | P_e: D = 1 / 2 / 3 R200c | n_e: D = 1 / 2 / 3 R200c | Share from 3D r > 2 R200c: P_e / n_e |
|---|---|---|---|
| 0.1 R200c | 0.94-0.99 / 0.990-0.999 / >= 0.997 | 0.85-0.96 / 0.96-0.99 / 0.988-0.997 | 0.1-1.0% / 0.8-3.9% |
| 0.5 R200c | 0.87-0.94 / 0.97-0.99 / 0.991-0.998 | 0.76-0.88 / 0.93-0.97 / 0.979-0.990 | 0.9-3.1% / 3.3-7.4% |
| 1.0 R200c | 0.74-0.82 / 0.93-0.96 / 0.977-0.991 | 0.65-0.72 / 0.88-0.91 / 0.964-0.970 | 5.6-9.8% / 12-15% |
| 2.0 R200c | 0.49-0.54 / 0.78-0.84 / 0.95-0.97 | 0.45-0.47 / 0.75-0.77 / 0.94-0.94 | - |

- **Cylinder vs sphere.** Through the 4 R200c sphere, Y_cyl(<0.65 R200c ~ R500c) / Y_sph = 1.23-1.54, and N_e,cyl / N_e,sph = 1.39-2.01 over log M200c 15.5 down to 13.6.
- **Real maps add more.** Lightcone maps also contain gas beyond 4 R200c and the 2-halo term, so map-based annuli always exceed the 1-halo 3D stack. The excess must be modelled (Hill et al. 2018, arXiv:1706.03753).
- **Pixel scale (estimate).** NSIDE 4096 pixels are 0.86 arcmin, which is 98 / 236 / 323 kpc at z = 0.1 / 0.3 / 0.5. NSIDE 16384 pixels are 0.21 arcmin: 24 / 59 / 81 kpc. For comparison, R200c is 0.72 Mpc at 10^13.6 Msun and 0.97 Mpc at 10^14 Msun (z = 0).

**Deprojection (onion peeling) [MN].**
- The test case: noiseless, spherical; annuli on the fine 67-shell grid; the outer boundary X_out set inside the true 4 R200c extent of the gas. Values are recovered over true, minus 1.

| X_out | P_e, shells inside 0.42 R200c | P_e, 0.42-1 R200c | P_e, 1-1.34 R200c | n_e, inside 0.42 R200c | n_e, 0.42-1 R200c | Last shell |
|---|---|---|---|---|---|---|
| 4 R200c (no truncation) | 0 | 0 | 0 | 0 | 0 | exact |
| 3 R200c | <= +0.03% | +0.0-0.2% | +0.2-0.5% | <= +0.1% | +0.1-0.4% | x1.6-1.9 |
| 2 R200c | <= +0.2% | +0.1-1.2% | +2.0-3.9% | <= +0.9% | +0.4-2.8% | x2.05-2.53 (1.75-2R) |
| 1.34 R200c | <= +0.9% | +0.5-6.5% | x1.54-1.77 (this is the last shell) | <= +3.3% | +1.6-13.5% | n_e: x1.85-2.10 (1-1.34R) |

- **Discretisation alone.** Piecewise-constant shells on 16 coarse annuli produce -6.6% to +9.6% shell-to-shell oscillations even without truncation.
- **What real data add.** Noise amplification, beam deconvolution (Planck Int. V used Croston et al. 2006 with regularisation and a 10 arcmin beam) and asphericity.
- **Verdict.** Deprojection is unnecessary in simulations, where 3D sums exist.

**Observable-space forward modelling.**
- **Method.** CAP filters with a disc of radius theta_d minus a ring out to sqrt(2) theta_d, a beam (ACT DR5 150 GHz, or 1.3-2.1 arcmin), galaxy selection, miscentring and the 2-halo term.
- **Examples.**
  - Amodeo et al. 2021 (arXiv:2009.05558) and Schaan et al. 2021 (arXiv:2009.05557) used CAP radii of 1-6 arcmin.
  - McCarthy et al. 2025 (arXiv:2410.19905): FLAMINGO Doppler-b lightcone maps at native NSIDE 16384, down-graded to 8192. Three Delta z = 0.05 shells converge, and satellites add up to 20% to the kSZ.
  - Siegel et al. 2026 (arXiv:2608.00164): FLAMINGO lightcone tSZ compared with DESI LRG stacks.
- **Use.** This yields closure tests of the painted HalfDome sky, not 3D calibration targets: the mixture of masses, satellites, 2-halo gas and beam cannot be inverted uniquely.

### 8. (e) Catalogue shortcuts, saved products and models

**SOAP-HBT aperture content at z = 0 [SOAP].**
- **Spherical-overdensity groups.** SO/{2500_crit, 1000_crit, 500_crit, 200_crit, 100_crit, 50_crit, 200_mean, BN98, 5xR_500_crit}. Each holds ComptonY (with and without recent AGN heating), DopplerB, GasMass, HotGasMass, GasTemperature (mass-weighted; variants without cool gas and/or recent AGN), GasComptonYTemperature, SpectroscopicLikeTemperature (with and without AGN), XRayLuminosity (observer frame, rest frame, photon), Thermal and Kinetic energies, Concentration, MassFractionSatellites and External, and SORadius.
  - Core-excised (0.15 R500c) variants exist only for 500_crit.
  - There is no electron-number property.
  - Properties are masked for haloes with fewer than 100 bound particles.
  - ComptonY is stored with lossy DMantissa9 compression, and our particle sums still match it to about 1e-5.
- **Fixed-radius spheres.** InclusiveSphere (all particles) and ExclusiveSphere (bound only) at 10, 30, 50, 100, 300, 500, 1000 and 3000 kpc. They hold GasMass and mass-weighted GasTemperature, but no ComptonY or DopplerB.
- **Projected apertures.** ProjectedAperture at 10-100 kpc, in 3 projections, holds gas mass only.
- **No radial profiles** of any kind are stored.

**Where SOAP SO apertures fall, in R200c units.**
- These are NFW estimates with each bin's median SOAP c200c (`soap_aperture_radii_nfw_estimate.json`). The estimate reproduces the measured R500c/R200c to <= 0.005.

| log M200c | 2500c | 1000c | 500c (measured) | 100c | BN98 | 200m | 50c | 5 R500c |
|---|---|---|---|---|---|---|---|---|
| 12.5-13.0 | 0.32 | 0.50 | 0.67 | 1.33 | 1.32 | 1.63 | 1.76 | 3.35 |
| 13.5-14.0 | 0.29 | 0.47 | 0.66 | 1.35 | 1.34 | 1.67 | 1.81 | 3.27 |
| 14.5-15.0 | 0.28 | 0.46 | 0.65 | 1.36 | 1.35 | 1.68 | 1.83 | 3.26 |

- **What SOAP gives.** Differences of consecutive cumulative values form about 7 coarse shells between about 0.3 and 3.3 R200c, from 8 distinct radii (100c and BN98 coincide).
- **What it cannot give.**
  - no shape inside about 0.3 R200c, where 3-11% of the tSZ sits in the core below 0.1R (kernels);
  - nothing beyond 3.35 R200c;
  - all particles in the sphere, so no ownership options;
  - gas mass instead of electrons (a fully-ionised mu_e must be assumed);
  - no AGN-phase separation for gas mass.
- **What it is good for.** It is the cheapest exact integral constraint (Y, N_e via gas mass) for redshift and feedback-variant trends, where full snapshots or membership files are missing. Kugel et al. 2023 (arXiv:2306.05492) calibrated FLAMINGO on aperture gas fractions alone, using f_gas(<R500c) with b_HSE ~ N(0.74, 0.10), and McCarthy et al. 2017 (arXiv:1603.02702) on Y(<5 r500).

**All shortcut methods.**

| Method | Measures | Inputs | Cost | Pros | Cons | For painting |
|---|---|---|---|---|---|---|
| SOAP SO aperture differences | coarse shells of Y, Doppler B, gas mass | SOAP SO columns | 122 MB per column (15.3M rows); no particles | exact integrals; every snapshot and variant | about 8 radii; 0.3-3.35 R200c; no electrons or ownership | integral constraints and z/variant trends |
| SOAP fixed spheres and projected apertures | gas mass and T_mw within 10-3000 kpc | SOAP | same | physical-radius checks | no ComptonY; radii fixed in kpc | minor |
| Reduced snapshots | particle profiles to R100c | COSMA hdfstream; SOAP IncludedInReducedSnapshot | 485M gas particles at z = 0 (L2p8) | many redshifts, larger box | stops at about 1.35 R200c; at most 200 haloes per 0.05 dex | inner shape vs z |
| 1% downsampled snapshot | aggregate y and mass stacks | Coordinates, Masses, ComptonYParameters | 1% of the I/O | cheap | expected share of haloes with >= 20 particles per coarse shell: 0 inside 0.6 R200c at 13.5-13.7 and <= 17% beyond; per-halo profiles impossible below about 10^14.3; no n_e or T fields [MN] | no |
| Saved pass products (this study) | everything in Sections 3-6 at z = 0 | study npz files | seconds | any estimator, selection, statistic or sub-grid | z = 0 fiducial only | first choice at z = 0 |
| Halo-model fits to stacks (Amodeo et al. 2021; Hill et al. 2018; Pandey et al. 2022, arXiv:2108.01601) | effective GNFW/OBB parameters | projected stacks, beam, mass distribution, 2-halo model | MCMC | works on data | degenerate; tied to the assumed form | no |
| Published fits and emulators: B12, B16, A10, Lee22; HMx (Mead et al. 2020, arXiv:2005.00009); picasso (Keruzore et al. 2024, arXiv:2408.17445) | parametric profiles | M, z (and c) | free | smooth in M and z | inherit the source simulation and its conventions. On HalfDome vs the FLAMINGO-calibrated reference: B12 +4.4-5.6% yy, A10 +29% at ell = 100, Lee22 -0.5 to -2.3% (D). picasso: percent-level bias, ~20% scatter at 0.1-1 R500c | fallback below the calibrated mass (B12/B16 now) |
| Baryon pasting and baryonification (Osato & Nagai 2023, arXiv:2201.02632; Shaw et al. 2010, arXiv:1006.1945; Bode et al. 2009, arXiv:0905.3748; Schneider & Teyssier 2015, arXiv:1510.06034; Anbajagane et al. 2024, arXiv:2409.03822) | analytic HSE gas, or displaced particles on N-body | M, c, particles, feedback parameters | Osato & Nagai: all-sky halo-based map 170 min on 320 cores; lightcone particle-based 62 min on 224 cores | physical; handles asphericity and diffuse kSZ gas | needs calibration itself; halo-based tSZ within ~10% of the halo model, particle-based ~20% lower | alternative painter that would use these stacks as targets |

### 9. What the main papers did

| Paper (arXiv) | Data | Estimator | Statistic | Radii | Normalisation | Selection cuts | Mass bins | Resolution criteria |
|---|---|---|---|---|---|---|---|---|
| Nagai, Kravtsov & Vikhlinin 2007 (astro-ph/0703661) | ART AMR, 16 clusters (6 relaxed), 5e13-2e15 h^-1 Msun; Chandra | gas cells in shells (weighting not stated) | mean and scatter of log | r/r500; fits 0.5-2 r500 (simulations), <~ r500 (Chandra) | P500 = n_g,500 k T500 | relaxed subsample | one stack | peak resolution 5-7 h^-1 kpc |
| Arnaud et al. 2010, A10 (0910.1234) | REXCESS (31 of 33) + 3 simulation sets | deprojected n_e x PSF-corrected T; simulations with mass-weighted T | median | 0.03-1 R500 (X-ray), 1-4 R500 (simulations); Y to 5 R500 | A10 P500 x M^0.12 | representative sample | one universal stack, 1e14-1e15 M500 | >= 15 clusters per radius; simulations renormalised +10% beyond 0.4 R500 |
| Battaglia et al. 2012, B12 (1109.3711) | GADGET-2 SPH, 165 h^-1 Mpc, 256^3, AGN | spherical shells (estimator not stated) | Y_Delta-weighted average of P/P_Delta; internal-variance weights | x = r/R200; fit range not stated (>~ 2 R200 down-weighted); maps: all gas to 6 R500 with a Gaussian taper | P200 | substructure kept; FoF b = 0.2; shrinking-sphere centre | power law in M200 (P0 18.1, xc 0.497, beta 4.35 at 1e14); M500 > 4.2e13 for C_ell | eps 20 h^-1 kpc, 32 neighbours |
| Battaglia 2016, B16 (1607.02442) | as B12 | gas density in shells | "stacked average" (not specified) | x = r/R200 (range not stated) | rho/rho_crit | not stated | power laws (rho0 4.0e3 M^0.29, alpha 0.88, beta 3.83; AGN model) | as B12 |
| Le Brun et al. 2014 (1312.5462) | cosmo-OWLS | synthetic X-ray spectra in 3D bins (10-20 within r500) | median, 68% | r/r500,hse | rho_crit, S500 | T < 1e5 K and substructure-bound gas excluded; centre = most-bound particle | groups 5.75e13-1.54e14, clusters 2.5e14-1e15, resampled to observed medians | M500 >= 1e13 |
| Le Brun et al. 2015 (1501.05666) | cosmo-OWLS synthetic tSZ skies | Planck-like aperture fluxes | LBG stacks | Y(<r500), Y(<5 r500) | Y-M | LBG selection | stellar and halo mass | - (finds Y(<r500) biased up to x10 at 1e13) |
| McCarthy et al. 2014 (1312.5341) | cosmo-OWLS | P from synthetic X-ray n and T; tSZ maps by SPH-kernel smoothing | median, 68% | r/r500,hse | P500,hse | (maps: all gas) | groups 5.8e13-1.5e14, clusters 2.5e14-1e15 | - |
| McCarthy et al. 2017, BAHAMAS (1603.02702) | BAHAMAS | synthetic X-ray profiles; calibrated on hot-gas fractions and the GSMF | median (profiles); mean and median (Y) | 10-20 bins within r500; Y(<5 r500) | r500,hse | T < 1e5 K and satellite-bound gas excluded | groups 5.25e13-2e14 (526), clusters > 2e14 (166) | - |
| Lee et al. 2022, Lee22 (2205.01710) | TNG300 (also SIMBA) | cell sums V(bin)^-1 sum V_i A_i | mean, lognormal likelihood with covariance (Hartlap) | 27 log bins 0.04-1.34 R200 (1.01, 1.76 tested) | P200, n200 | not stated | 0.2 dex in h^-1 Msun from 1e13 h^-1; concentration quartiles; z <= 2 | > 29 haloes per bin |
| Braspenning et al. 2024, B24 (2312.08277) | FLAMINGO L2p8_m9 (L1_m9 for variants) | Q_V, Q_X (fiducial), Q_M | median, 16-84% | 30 log bins 0.01-3 R500c | P500 (f_B 0.159, mu 0.59, mu_e 1.14) | no SF gas, T > 1e5 K, no AGN heating in the last 15 Myr; VELOCIraptor + SOAP | 0.5 dex M500c, 1e13-1e15.5 | shown where >= 16% of haloes have a particle |
| Schaye et al. 2023 (2306.04024) | FLAMINGO | SOAP apertures; on-the-fly HEALPix lightcone maps (tSZ, kSZ, DM, X-ray, lensing) | - | SO and fixed apertures | - | - | - | m9 baryon mass ~1e9 Msun |
| Kugel et al. 2023 (2306.05492) | FLAMINGO calibration runs (32 per resolution) | aperture f_gas(<R500c); no profiles | median, 16-84% | R500c | b_HSE ~ N(0.74, 0.10) for X-ray masses | VELOCIraptor | adaptive bins with >= 10 objects | M500c >= 10^13.5 |
| Kay et al. 2024 (2404.08539) | FLAMINGO massive clusters | pressure profiles | median, GNFW fit with gamma fixed at 0.31 | - | P500 | massive clusters | - | - |
| Planck Int. V 2013 (1207.4061) | 62 ESZ clusters | y annuli (0.25 R500) deconvolved (10 arcmin beam) and deprojected | mean (median within 5%) | to about 3 R500 | A10 P500 x M^0.12 | XMM-observed ESZ | one stack | beam-limited |
| Amodeo et al. 2021 (2009.05558) | ACT + Planck around CMASS | CAP on beam-convolved projected GNFW/OBB models | stack | 1-6 arcmin | - | CMASS galaxies | mass-weighted over CMASS | beam 1.3-2.1 arcmin |
| McCarthy et al. 2025 (2410.19905) | FLAMINGO lightcone kSZ | Doppler-b HEALPix maps, CAP, ACT beam | velocity-weighted stack | CAP radii | - | stellar-mass cut fitted to lensing; satellites included | - | - |
| Diemer & Kravtsov 2014 (1401.1216) | N-body | DM density | median (mean checked) | r/R200m | rho_m | isolated haloes | peak height nu and accretion rate Gamma | >= 1000 particles |
| **This project** | L1_m9 z = 0, 4,588 centrals | geometric (16 quantities stored) | inclusion-weighted mean + lognormal scatter | 67 shells to 4 R200c (40 to 6 R500c) | P200; XGPaint electron units | all gas; own satellites; A2-ii; recent-AGN gas removed | 0.2 dex M200c 12.5-15.5; 300 random per bin below 10^14.5, all above | x >= max(0.04, 4 eps/R); >= 20 particles; >= 2 h |

The machine-readable versions are `literature_table.json` (18 rows) and `methods_catalogue.json` (31 methods, with the fields name, measures, inputs, cost, pros, cons, literature, relevance_for_painting, key_numbers, status_in_this_project).

### 10. Ranking for painting HalfDome

1. **Calibrate on full-snapshot particles** with the geometric estimator, the inclusion-weighted mean, 0.2-dex M200c bins, P200 and XGPaint units, all gas with own satellites, the A2-ii partition, recent-AGN gas removed, extraction and cut at 4 R200c, and the 3-part resolution mask.
   - At z = 0 this costs seconds from the saved products. A new snapshot needs one more particle pass on maas per redshift (estimate: about 1 h, scaled from the z = 0 pass); the fiducial run has full snapshots at z = 0.15, 0.2, 0.3, 0.5, 0.8, 1, 1.5 and 2.
2. **Use SOAP SO apertures** for what full snapshots cannot reach:
   - integral (Y, gas mass) trends with redshift at the 71 SOAP outputs (0007-0077);
   - the feedback and cosmology variants, which have no membership files and full snapshots only at z = 0 and 0.15.
   Treat SOAP as constraints on the integrals of an already-shaped profile, not as a profile source.
3. **Use reduced snapshots (L2p8, L1_m8)** only to check the redshift and resolution dependence of the inner shape (<= 1.35 R200c).
4. **Use lightcone maps and CAP stacks** as closure tests of the painted sky.
5. **Avoid as calibration inputs:** Q_V, Q_M, Q_X, medians, log-means, Y-weighted stacks, A10-type universal shapes, and fits converted with a different mu. Each is a measured systematic of 2-80% (Sections 3-4 and D).

### 11. Files (all in `stacking_choices_20261014/research/methods/`)

- `methods_numbers.py`, `methods_numbers.json`, `methods_numbers.npz`: Sections 3-7 numbers, the kernel transfer matrix, coarse edges.
- `methods_catalogue.json`: the 31 methods; built by `build_methods_catalogue.py`.
- `literature_table.json`: Section 9.
- `soap_0077_group_listing.json`: read-only SOAP group and dataset listing and the recently-heated-gas filter attributes.
- `soap_aperture_radii_nfw_estimate.json`: the SO-aperture radius table.
- `methods_section_markdown.txt`: this section as plain text.

### Caveats

- **The kernel-deposition comparison is an estimate.** It uses particles at shell volume quantiles with the shell-mean value, and smoothing-length histograms coarsened 4x. It is not a re-read with true positions and values.
- **The projection and onion-peeling tests** use the stacked 3D mean profiles. They assume spherical symmetry, no noise and no gas beyond 4 R200c.
- **SOAP aperture radii other than R500c and 5 R500c are NFW estimates.**
- **The radius-scaling comparison** uses shells matched in physical radius only approximately (0.24-0.30, 0.51-0.64, 0.91-1.14 and 1.60-2.02 R200c for the R500c shells).
- **Pixel scales** use a flat Omega_m = 0.306, h = 0.681 approximation.
- **Unverified statements, flagged in the text:**
  - the absence of a decoupled-wind gas class in FLAMINGO (assumed);
  - B16's statistic, range and selection (not stated in the paper);
  - B12's fit range (not stated).
- **Numbers marked "project memory note"** come from earlier sessions' notes, not from this study's files.