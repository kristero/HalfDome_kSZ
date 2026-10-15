## The input parameters (choices) of the stacking pipeline

This section lists every choice in our own code that changes the 1D stacked profiles. That covers the maas particle pass (`census.py`, `prepare_study.py`, `extract_study.py`, `measure_overlap.py`), the `analysis/` library, and the stacking and fitting code of Tests B, F and G. Each choice is compared with the earlier pipelines (L2p8 reduced, `flamingo_stacks.py`/fullsnap, radii234).

**How to read the tables**
- Paths are relative to the study root `overlap_covariance_L1_m9_0077_20261012/`. A leading `../` means `flamingo_pressure_calibration/`.
- **Local** means the alternative can be computed from the stored products without reading particles again. **New pass** means it needs a new particle read.
- Measured effects come from `stacking_choices_20261014/research/choices/quantify_choices.json`, unless another source is named. I recomputed those numbers from the stored products. The recomputed per-halo profiles equal Test F's `profiles.npz` to 1.7e-15.
- Ranges cover the calibrated bins 13.5-15.5 (log10 M200c/Msun) and resolved (base) shells only, unless stated otherwise.
- The machine-readable catalogue has 69 entries: `research/choices/choices_catalogue.json`. The same entries as a plain list are in `choices_catalogue_list.json`.

### What the stored products can and cannot do

**Computable locally:**
- **Radial grids:** any sub-grid of the 68 X200 edges (to 4 R200c) or the 41 X500 edges (to 6 R500c).
- **Gas selections:** any union of the 17 owner classes and 3 phases. The phases are recent-AGN, X-ray-selected and other.
- **Quantities:** 16 per shell. From them come the geometric, Q_V, Q_M, Q_X and n_e T_X estimators, the T_M, T_V and T_X temperatures, and clumping factors.
- **Stacking:** per-halo profiles are stored, so any statistic, any halo weights and any mass bins are possible.
- **Resolution masks:** any threshold, from the stored particle counts and smoothing-length histograms.
- **Normalisations:** P200, B24, A10 and self-similar-500.
- **Overlap handling:** A2-i, A2-ii, excision and templates. These exist on X200 only and carry only count, mass, volume, y and electrons.

**Needs a new pass:**
- anything beyond 4 R200c or 6 R500c, and any new shell edge;
- another centre, another recent-AGN window, or a temperature cut;
- FOF membership or another painting threshold;
- R200m scaling;
- kernel-weighted deposition;
- any other snapshot or simulation.

### 1. Sample and catalogue

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Run, snapshot, halo finder | `study_config.json:2-8` | L1_m9 HYDRO_FIDUCIAL, snapshot 0077 (z = 0), SOAP-HBT + HBT membership | none | other z (full snapshots 0037-0074 on maas); feedback variants (0074/0077 only, no membership files) | One snapshot cannot give z or feedback trends. |
| Central definition | `census.py:270-271` | SOAP/HostHaloIndex < 0 | same as IsCentral (0 disagreements, `census_summary.json`) | - | HalfDome paints distinct haloes only. |
| Mass / radius | `census.py:274-276`, `prepare_study.py:41` | M200c/R200c; M500c/R500c stored | R500c scaling (X500 grid); host-only mass (Test A) | M200m, Mvir, BN98 | Sets the self-similar scaling. R500c/R200c median 0.656 (5-95%: 0.606-0.682). |
| Mass range | `study_config.json:9-26`; `census.py:321-324` | 12.5-15.5; calibrated 13.5-15.5 with log M <= 15.388 (`F_observables/f_common.py:146-154`) | any sub-range | below 12.5 | No bin below 13.3 passes the strict mask. |
| Sampling per bin | `study_config.json:27-37` | 300 random per 0.2 dex below 14.5; all above | sub-samples | more haloes | N = 308-332 per bin below 14.5; 895 / 326 / 127 / 21 / 6 above. |
| Stratified enlargement | `study_config.json:56-70`; `census.py:228-258` | +213 annulus-only haloes, so each (bin, class) has >= 50 | drop them (`role == 1`) | other strata | Makes the halo weights non-uniform within a bin. |
| Census class | `study_config.json:38-50`; `census.py:161-166` | overlapper: mu >= 0.1 reaching inside 2 R200c; classes clean / reach1R / annulus-only | mu 0.03 / 0.3; own satellites, neighbours (`newdata.py:259-302`) | mu < 0.03, reach > 2R | Calibrating on clean haloes only changes yy by +0.7 to +7.9% (SYNTHESIS 2.2). |
| **Inclusion weights** | `prepare_study.py:284-295`; `analysis/newdata.py:242-244` | 1/inclusion probability of the (bin, class) stratum; 1 above 14.5 | equal; x Y200 (`D_literature/td_recipes.py:133-139`) | - | Equal weights read high in 13.5-14.5 (largest per-bin deviation): pressure 0.4-0.8% inside 1R, 2.1-5.3% at 1-2R, 3.4-9.4% at 2-4R; electrons 1.7-4.3% at 2-4R. |
| Mass floor, painting threshold | `study_config.json:88-91`; `census.py:39-42` | tree floor 9.4e10 Msun; painted = host M200c >= 6.6e12 | - | another threshold | Defines owner codes 14/15 and the A2 rules. |

### 2. Extraction geometry

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| **Extraction sphere** | `study_config.json:95-97`; `prepare_study.py:281-282` | r <= max(4 R200c, 6 R500c). owner200 uses r <= 4 R200c (`measure_overlap.py:177`); owner500 uses r <= 6 R500c (`:181-182`) | any outer edge <= 4 R200c or <= 6 R500c | 5 R200c, the 2-halo regime | 6 R500c > 4 R200c for 26.5% of haloes; r_ext/R200c max 4.24. The 2-4R shells carry 37% of mean DM and 64% of DM Q16. |
| Centre | `census.py:272` | InputHalos/HaloCentre (halo-finder centre) | none (the centre-of-mass offset is stored) | centre of mass, gas or SZ peak | Mis-centring flattens the inner stack. abs(CoM - centre)/R200c: median 0.087, 90th percentile 0.184, 63% > 0.07. |
| Periodic and sphere test | `extract_study.py:108-120`; `grids.py:59-61` | minimum image; exact test; outer edge inclusive | - | - | Correctness only; bit-identical to radii234 (gates G5/G7). |
| Cell padding | `study_config.json:98`; `prepare_study.py:310-314` | 0.1 cell (3.1 Mpc) | - | smaller or larger | Completeness against read volume: 10,247 cells, 2.198e9 candidate rows. |
| Units | `measure_overlap.py:178-181`; `prepare_study.py:381` | comoving coordinates x a; physical R200c; softening = min(a eps_c, eps_max) = 0.0057 Mpc | - | - | No effect at z = 0 (a = 1); matters for any z > 0 pass. |

### 3. Particle selection

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Recent-AGN definition | `dependencies/flamingo_io.py:623-630`; `measure_overlap.py:196` | SOAP filter: a_last >= 0.998956 (15 Myr) and 8.80e6 <= T <= 1.756e8 K | - | other windows or temperature bands | a_last is stored as float16 (HalfFloat filter, `probe_g4.json`). The stored values that pass near a = 1 are 0.99902, 0.99951 and 1.0, so the effective window is about 17.5 Myr. This is an estimate from float16 arithmetic; SOAP sees the same stored values. |
| **Recent-AGN gas removed or kept** | `analysis/variants.py:159-167, 320` | removed, for both pressure and electrons | `agn='kept'` | - | Kept/removed on resolved shells: pressure <= 0.05%, electrons <= 0.17%. Removing it from n_e is a convention borrowed from pressure. |
| Star-forming / cold gas | `measure_overlap.py:197-198` | kept (phase "other") | X-ray phase only | T or density cuts | Star-forming gas has y = 0: none of the 503,296 star-forming particles has non-zero y. |
| X-ray phase | `study_config.json:102-110`; `extract_study.py:131` | SFR <= 0, T >= 1e5 K, not AGN; ROSAT L_X (column 2) | Q_X, n_e T_X, T_X | eROSITA bands, other T cuts | Used only in X-ray recipes, which bias yy by +80/+42/+23% (D). |
| **Ownership classes** | `measure_overlap.py:46-57, 132-155`; `prepare_study.py:148-151` | HBT bound membership mapped to 17 owner codes | any union of codes | FOF membership; another painted threshold | Stacked y shares: see the ownership table below. |
| Gas selection used | `analysis/variants.py:60-61, 92-124` | "all" for calibration; A2-ii for painting | no_other, host_bound, central_bound, remove_overlappers(mu, reach, flavour), A2-i, A2-ii | - | Central-bound only would drop 62-70% of the 1-2R y. "all" paints the 2-4R neighbour gas twice. |
| A2-i | `analysis/variants.py:110-116, 198-220` | remove owner 14 plus neighbour-overlapper codes whose overlappers are painted (bound-gas majority 0.5) | owner-14-only | - | Removes gas bound to other painted haloes but not their unbound atmospheres. |
| **A2-ii partition** | `measure_overlap.py:217-226` | gas goes to a painted neighbour if min_n abs(r - d_n)/R200c,n < r/R200c,c | `partition_owned` (+ not other-painted) | other distance metrics | Removes 2.4-7.4% of Y(<4R) and 5-7.6% of N_e(<4R) (SYNTHESIS 1). |
| Excision, templates | `analysis/variants.py:33-46, 246-262, 529-567` | not used for painting | variants, kappa 0.5/1/2, member sets, min_retained 0.5 | other kappa | Excision overshoots Y(<1R) by 6-16%; templates leave a +3.5-5% residual (A). |

How the stacked y (AGN removed) splits between owners, bins 13.5-15.5:

| Radius | Central-bound | Own satellites | Bound to other haloes | Unbound |
|---|---|---|---|---|
| 0.1-1R | 84-96% | 0.1-4.2% | 0% | 4-12% |
| 1-2R | 30-38% | 1.2-8.9% | 0.01-0.05% | 59-64% |
| 2-4R | 1.8-5.8% | 3.8-14.4% | 5.0-13.2% (painted hosts 4.8-12.9%) | 72-89% |

### 4. Quantities and estimators

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Pressure source | `measure_overlap.py:75, 190-193` | particle ComptonYParameters; P_e = y/V m_e c^2/sigma_T | n_e k T; SPH gas pressure | T_e not equal to T | Our aperture sums reproduce SOAP's Y to about 1e-5 (D). |
| **Electron count** | `measure_overlap.py:189-191, 228-231` | ElectronNumberDensities x m/rho | fully ionised H+He (aux200/aux500); gas density / mu_XGP | metal electrons | n_e / fully ionised: 97.0-99.7% (< 0.1R), 99.2-100.2% (0.1-1R), 99.2-99.8% (2-4R). Painting gas density with mu_XGP gives about 9-10% fewer electrons (D). |
| **Pressure estimator** | `analysis/variants.py:57-58, 361-386` | geometric: sum y / V_shell | Q_V, Q_V_y, Q_M, Q_X, n_e T_X | - | Q_V is +3-5% inside 1R and +18-25% at 3-4R; Q_M is x1.6-2.6 at 1-2R; Q_X is +17-36% at 0.1-0.5R (B). |
| **Volume estimator** | `analysis/variants.py:266-269` versus `:366-373` | geometric shell volume | sum of SPH volumes m/rho | kernel-deposited volume | Summed SPH volume / geometric volume: 95-98% inside 1R, 91-95% at 1-2R, 76-91% at 2-4R. |
| Deposition | `measure_overlap.py:183-184, 206-209` | at the particle centre (point) | - | kernel deposition | Exact for integrals; the 2h cut removes the shells it affects. |
| Density estimators | `analysis/variants.py:387-400` | geometric | Q_V; Q_M (gas) | - | Mass-weighted densities are x7-10 at 0.1-0.5R and x300-390 at 1-2R (B). |
| Temperature diagnostics | `analysis/variants.py:432-451` | not in the stacks | T_M, T_V, T_X, clumping factors | T_e not equal to T | Feed the X-ray recipes and Test G's outer temperature. |
| Order of operations | `analysis/variants.py:7-8, 333` | rebin linear sums first, then form ratios | ratio of summed sums | - | Exact on any sub-grid. |

### 5. Radial grids

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| X200 storage grid | `grids.py:14` | 67 shells over 0-4 R200c: union of the radii234 edges, Lee22's 27 shells and {1.5, 1.75, 2.5} | any exact sub-grid (`newdata.py:150-168`) | new edges; beyond 4R | Every fit or cut radius must be a stored edge. |
| X500 storage grid | `grids.py:15` | 40 shells over 0-6 R500c (B24's 30 shells plus extra edges) | B24 30-shell grid; any sub-grid | new edges | R500c-scaled literature comparisons. |
| F fit grid | `F_observables/f_common.py:86-103` | 35 shells to 4R: radii234 log edges (0.069 dex) inside 0.84R, plus 1, 1.15, 1.34, 1.5, 1.75, 2, 2.18, 2.5, 3, 3.30, 3.63, 4 | other sub-grids | - | Every requested R_fit and R_cut is an exact edge; no slivers. |
| Lee22 grid | `grids.py:18` | 27 log shells over 0.04-1.34 R200c | stored | - | On this grid the GLS covariance is near-singular (adjacent rho 0.90-0.94, condition number 220-1,500). |
| radii234 grid | `pilot_physics.py:24` | 36 shells | stored, bit-identical | - | Legacy parity. |
| Model extension shells | `F_observables/f_common.py:104-107` | 4-12 R200c, models only | - | measured profiles beyond 4R | Lets painted models extend past the data. |
| Smoothing-length histogram | `measure_overlap.py:73, 233-240` | h a/R in 384 log bins over 1e-4 to 1e2 (0.0156 dex), all gas | any quantile; exact rebinning | per-owner histograms | Upper median is conservative by <= 0.0156 dex; under/overflow 0. L2p8 uses 0.05-dex bins and the median at the bin centre. |

### 6. Normalisations

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Pressure: P200 | `pilot_physics.py:30-34`; `analysis/newdata.py:199-201` | G M200c (200 rho_c) f_b / (2 R200c), from the halo's own M and R | B24 P_e,500 (f_B 0.159, mu 0.59, mu_e 1.14); A10; self-similar-500 (`newdata.py:203-230`); `normalisation='auto'` (`variants.py:276-277`) | - | XGPaint paints P/P200 against r/R200c. rho_c from the halo versus from the cosmology differs by 1.08e-5. |
| Density units | `pilot_physics.py:17-19`; `analysis/variants.py:306-316` | electrons: n_e mu_XGP/(rho_c f_b) with mu_XGP = 1.2632 m_p; gas: rho/(rho_c f_b) | B24 n_e,500; any constant | - | Must match XGPaint's tau convention. |
| Baryon fraction | `analysis/newdata.py:103` | f_b = 0.0486/0.306 = 0.158823 | B24 fixed 0.159 (0.11% different) | - | Constant factor; must match the painter. |
| Scaling variable | `analysis/variants.py:266-269` | r/R200c (X200) or r/R500c (X500) | either | r/R200m, physical r | Self-similarity and the catalogue convention. |

### 7. Resolution criteria (`analysis/masks.py:18-20` DEFAULTS)

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Minimum inner edge | `analysis/masks.py:42` | x >= 0.04 | any value | - | Binding only in 15.1-15.5. With 0.01 the first shell moves from 0.040 to 0.034 / 0.025 there. |
| Softening | `analysis/masks.py:42`; `prepare_study.py:381` | x >= 4 eps/R, eps = 0.0057 Mpc | any factor | - | Never binds; factor 2.8 changes nothing. |
| **Minimum particles per halo-shell** | `analysis/masks.py:43` | >= 20 gas particles (all owners) in every halo | any value | - | **20 versus 32: identical masks in every bin.** With 100, the first shell moves 1-2 shells outward (13.5-13.7: 0.321 to 0.519). L2p8 and fullsnap use 32; fullsnap counts pooled particles. |
| **Smoothing-length factor** | `analysis/masks.py:44-45`; `analysis/io.py:240-248` | x_inner >= 2 x the upper median of h a/R in that shell | any factor or quantile | - | Binds in every supported bin. With factor 1 the first shell moves 1-2 shells inward and 13.1-13.3 becomes supported; with factor 3 it moves 1-2 shells outward and 13.3-13.5 loses support. |
| Strict / diagnostic / base / supported | `analysis/masks.py:46-53` | strict = every halo of the bin resolved; supported = >= 6 strict shells inside 2R; otherwise the diagnostic mask | quantile rules (`resolved_fraction` is returned) | - | "Every halo" gets stricter as N grows. In unsupported bins 12.5-13.3 the diagnostic mask starts at 0.105-0.233, inside the first shell of the supported 13.3-13.5 bin (0.519). There the median is up to 98% below the mean in 0.1-1R. Not used for painting. |
| **Per-halo versus common mask** | `analysis/masks.py:43-46`; `F_observables/f_fit.py:108` | common mask per bin; mean over all haloes | per-halo masks | - | Averaging only the resolved haloes in partially resolved shells (0.04-0.32 R200c, bins 13.5-15.1) gives +1.6% to +104%: denser haloes pass first. This is the L2p8 rule (`../l2p8_reduced_calibration/fit_stage.py:49-56`). |
| Fit-shell mask | `analysis/masks.py:56-62`; `analysis/fitting.py:33` | base, outer edge <= cut, mean > 0, >= 6 shells | any | - | Zero shells are excluded, never log-filled. fullsnap requires only 3 shells. |

First resolved inner edge with the current rules (F grid, in R200c):

| Bin | 13.5-13.7 | 13.7-13.9 | 13.9-14.1 | 14.1-14.3 | 14.3-14.5 | 14.5-14.7 | 14.7-14.9 | 14.9-15.1 | 15.1-15.5 |
|---|---|---|---|---|---|---|---|---|---|
| First edge | 0.321 | 0.274 | 0.170 | 0.145 | 0.105 | 0.090 | 0.065 | 0.056 | 0.040 |

### 8. Stacking statistic and weights

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| **Statistic** | `analysis/stacking.py:33-44`; `F_observables/f0_profiles.py:25-28` | inclusion-weighted arithmetic mean of per-halo P/P200 and n_e units (mean of ratios) | median, log-mean (`stacking.py:29-30, 47-54`), weighted median, Y200 weighting, ratio of sums | - | The painted map is a sum over haloes, so the mean is what the observables need. Median/mean - 1 for pressure: -3 to -56% at 2-4R and -24 to +6% at 1-2R. Log-mean: -4 to -49% at 2-4R. Electron median: -18 to +2% at 2-4R. Y200 weighting: +1.6 to +3.4% on yy (D). |
| Scatter, N_eff | `analysis/stacking.py:33-44` | reliability-weighted SD; N_eff = 1/sum w^2 | percentiles, log-covariance | - | N_eff = 300-325 per bin below 14.5. |
| Bootstrap | `analysis/stacking.py:66-79`; `F_observables/f_common.py:113-114` | 1,000 halo bootstraps (F); 50 draws for the shrinkage intensity | stratified bootstrap; delta method; 4^3 jackknife | - | Jackknife errors are 1.16-1.26x the bootstrap; beyond 2R the bootstrap intervals are lower limits. radii234 used 128 draws, L2p8 200. |
| Covariance shrinkage | `analysis/covariance.py:173-197, 279-338`; `F_observables/f_fit.py:126-166` | shrink towards the pooled correlation of neighbouring bins; lam = 1 if N < 3p; eigenvalue floor 1e-10 | factor model; Higham projection | - | Used by the GLS fits only. |
| CV folds | `F_observables/f0_profiles.py:22, 114-123` | 4^3 blocks of 250 Mpc, 8 folds | other block or fold sizes | - | Out-of-fold validation. |

### 9. Mass binning

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Bin width and edges | `study_config.json:9-26` | 0.2 dex in M200c [Msun] | any edges, because per-halo profiles and weights are stored: 0.1 dex (about 150 haloes per bin below 14.5), 0.4 dex, Lee22 h^-1 bins (`analysis/io.py:307-318`), B24 0.5-dex M500c bins | - | Narrow bins track the mass trend; the top bins hold only 127 / 21 / 6 haloes. L2p8 uses 0.25-dex bins centred on multiples of 0.25 (`../l2p8_reduced_calibration/common.py:178-180`). |
| Top-bin pooling | `B_covariance/bcommon.py:12-13, 103-104` | B pools >= 14.9 (154 haloes, log M trend); F and G fit 15.1-15.3 (N = 21) and 15.3-15.5 (N = 6) one by one | either | - | The two conventions differ by 0.5-1.3% in C_ell^yy at 2R (B). |
| Mass given to a stack | `F_observables/f_fit.py:109`; `kernels/engine.py:332-345` | bin centre (mass-trend fits); painted shape constant within a bin | population-weighted mean, or the model averaged over the bin's haloes (as L2p8 and fullsnap do) | - | Population-weighted mean log M lies 0.004-0.026 dex below the bin centre (0.047 dex in 15.3-15.5). |

### 10. Fitting and truncation (after stacking)

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Family | `../radii234_.../models.py:9-17`; `analysis/families.py`; `G_physical_outer/g_models.py` | B12 / B16; recommended: plus physical outer terms (G) | gnfw5, env, spline, Lee22 broken power laws | - | B12/B16 fit the stack only to 2R (sigma_d about 12% at 4R). |
| **R_fit, R_cut** | `F_observables/f_common.py:109-111`; `G_physical_outer/g_common.py:83-84` | 4R / 4R | 1, 1.34, 1.5, 1.75, 2, 2.5, 3, 4 | > 4R | Y(<2R)/Y(<4R) = 70-90% and N_e(<2R)/N_e(<4R) = 54-69% (stacked, resolved shells). A 2R cut costs tSZ -5%, kSZ -26%, DM -22% (F). |
| Truncation shape | `F_observables/f_fit.py:204-224` | hard cut | 0.05-dex logistic, centred or flux-matched | - | <= 0.21% on C_ell at 4R (C). |
| Objective, error floor | `analysis/fitting.py:193-205, 33, 95-97` | (a) diagonal sigma = max(SD/mean, 5%) | (b), (c) GLS + sigma_d, (d)-(g) | - | GLS beyond 2R follows the tail (P0 x1.45 at 4R). fullsnap uses hypot(err/mean, 3%) (`../fullsnap_calibration/profiles.py:793`). |

### 11. Performance knobs

| Choice | Where | Current | Local alternatives | New pass | Why it matters / measured effect |
|---|---|---|---|---|---|
| Fields read, bytes per particle | `prepare_study.py:35-36, 326-357`; `study_config.json:99` | 13 datasets + membership = 140 B logical | - | A lean pass for P + n_e + AGN flag + resolution + ownership needs 64 B logical (46%). It drops 36 B of element fractions, 24 B of X-ray luminosities, IDs, SFR and Pressures. | The pass is decompression-bound. Every chunk (1,048,576 rows) of a dataset that is read is decompressed whole. |
| Workers and politeness | `study_config.json:112, 119-135`; `extract_study.py:329-333` | 6 workers, load-adaptive; nice 19, ionice 7 | - | more workers when the host is idle | Run record (`complete.json`): 66.4 min wall, 5.47 CPU-h, 307.8 GB logical, peak tree RSS 8.8 GB. |
| Chunk cache, reopening | `study_config.json:116, 138`; `extract_study.py:48-50, 166-177` | 40 MiB per dataset; reopen files every 16 cells | - | - | The cache holds one chunk of the largest dataset (36 MiB). |
| Memory and time guards | `study_config.json:113-115, 139-143`; `extract_study.py:76-78, 336-369` | RLIMIT_AS 8 GiB, RSS stop 48 GiB, 12% free-memory floor, 300-min wall limit | - | - | Protect the shared host; G6 measured 1.52 GiB per worker. |
| Read budget, batch, read path | `study_config.json:101`; `extract_study.py:51, 157` | 488 GB cap; 1M-particle batches; read the chunk files directly | - | - | No effect on results (all sums are linear). |

### Where the earlier pipelines differ from this one

| Item | This study | L2p8 reduced | flamingo_stacks / fullsnap | radii234 |
|---|---|---|---|---|
| Outer extent | 4 R200c / 6 R500c | R100c (particles exist only there) | R100c / 5 R200c | 4 R200c |
| Grid | 67 X200 + 40 X500 shells | 28 log shells over 0.03-1.6 R200c | 31 to 1.6 R200c / 54 log shells over 0.01-5 R200c | 36 shells |
| Membership | 17 owner classes | none (all gas) | none / "exclude_other_haloes" (= our `no_other`) | 4 components |
| Electrons | yes | no | fullsnap yes | yes |
| Particle count cut | 20 per halo-shell, every halo | 32 per halo-shell, per halo; >= 100 in R200c | 32 pooled over the stack | 20 |
| Smoothing-length cut | 2 x upper median per halo-shell, common mask | 2 x median at the bin centre, per halo | 2 x pooled median per 0.05-dex fine bin, "outward of last failure" | as ours |
| Halo weights | inclusion weights | equal | equal | equal (correct for its uniform sample) |
| Mass bins | 0.2 dex | 0.25 dex, rounded centres | 0.2 dex + 0.02-dex nodes | 0.2 dex |
| Error floor | max(SD/mean, 5%), >= 6 shells | none; equal weight per (z, mass) group | hypot(err/mean, 3%), >= 3 shells | 5% |

What these differences do (measured here):
- Per-halo masks bias partially resolved shells by +1.6% to +104%.
- Equal weights with an enlarged sample bias 2-4R by up to +9.4%.
- An R100c extent cannot support the 4R fit that the observables need.
- `no_other` also removes gas bound to small haloes that are never painted.
- The 32-versus-20 particle threshold has no effect.

**Documentation drift (no effect on results).** `STEP1_README.md` gives RLIMIT_AS 7 GiB, a 64 GiB RSS stop, an 80 MiB cache and 8 workers. The run actually used 8 GiB, 48 GiB, 40 MiB and 6 workers (`study_config.json`, `complete.json`). The docstring at `extract_study.py:3-4` still says 80 MiB.

### Which choices matter most (measured effect on the stacks)

1. Outer extent and truncation radius.
2. Estimator: geometric versus SPH-volume, mass- or X-ray-weighted.
3. Statistic: mean versus median.
4. Ownership against painted neighbours (A2-ii).
5. Halo weights when the sample is stratified.
6. Common versus per-halo resolution mask.
7. Smoothing-length factor, which sets the first resolved shell.

These change little: the minimum particle count (20 or 32), the softening factor, min_edge (except 15.1-15.5), the recent-AGN cut (<= 0.05% pressure, <= 0.17% electrons on resolved shells) and cell padding.