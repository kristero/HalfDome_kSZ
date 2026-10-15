# Shell estimators, temperature definitions, stacking statistics and bootstrap errors (group `estimators`)

## Bottom line for painting HalfDome

| Choice | Use for tSZ/kSZ/DM painting and fitting | Do not use | What the wrong choice does to HalfDome (calibrated bins 13.5-15.5; mean y and mean DM exact, C_ell, kSZ 1h and DM variance first order) |
|---|---|---|---|
| Shell estimator for P_e, n_e | **geometric**: particle y (or N_e) summed over the shell, divided by the geometric shell volume | Q_V (B24 SPH-volume weighting), Q_M, Q_X, n_e kT_X | Q_V: tSZ C_ell +8.29/+7.11/+6.50% (ell = 100/1000/3000), kSZ 1h +14.73/+10.99/+5.90%, mean DM +6.06%, mean y +4.42%. Q_M: tSZ +211% at ell = 100, mean y +187%. Q_X: tSZ +1721%, mean y +1353% |
| Temperature | **T_e** = sum N_e kT / sum N_e for anything tied to the painted P_e and n_e (P_e = n_e k T_e holds to 4e-4 for 99.84% of resolved halo-shells; maximum deviation 1.0e-3) | T_X and T_V for tSZ work. None of the cached temperatures is the y-weighted temperature that relativistic tSZ needs | At 3-4 R200c (bins 13.5-15.1): T_X/T_e = 1.38-1.64 (falling with mass) and T_V/T_e = 1.02-1.47 (rising with mass) |
| Stacking statistic | **inclusion-weighted mean** sum w F / sum w (the Hajek, i.e. ratio, form of the Horvitz-Thompson estimator) | median, geometric mean (log-mean), Y200 weighting, trimmed mean; unweighted mean below 10^14.5 | median: tSZ -5.0 +- 1.8% and kSZ 1h -6.9 +- 0.3% (ell = 100), mean DM -3.2%. Log-mean: -8.3%, -5.2%, -2.2%. Y200-weighted: tSZ +1.7/+2.5/+3.5%. 5% trimmed: -3.05%, -2.96%, -1.29% |
| Errors | 13.5-15.1: paired stratified halo bootstrap inside 1.5 R200c; beyond 1.5 R200c scale it by the measured jackknife/bootstrap ratio (band medians 1.09-1.18 at 1.5-3 R200c) or take the larger of the two. 15.1-15.5: the larger of the unstratified bootstrap and the 4^3-block spatial jackknife | the stratified bootstrap alone in the two top bins (strata of 17/2/2 and 2/3/1 haloes) and beyond 1.5 R200c | In 15.3-15.5 the stratified bootstrap is smaller than the unstratified one by up to x1.71 (P), x1.89 (n_e), x1.88 (gas density) and x1.84 (T_e), and smaller than the jackknife by up to x1.84 (P) and x2.04 (n_e). In 13.5-15.1 the jackknife exceeds it by band medians of 1.09-1.18 at 1.5-3 R200c and by up to x1.28 (P) and x1.39 (n_e) in single bins at 2-3 R200c |

## 0. Setup and how to read the numbers

**Reference (`ref`, identical in every group).**
- Key `sc.make_key(field, grid='fit')`: geometric estimator, recently AGN-heated gas removed, all gas, all phases, 35-shell fit grid.
- Normalisations: P/P200, n_e mu_XGP/(rho_c f_b), T/T200, K/K200.
- Inclusion-weighted mean; strict resolution rule (>= 20 particles, 4 softening lengths, 2 h, 0.04 R200c); 0.2-dex M200c bins.
- 200 stratified bootstrap draws with the core's per-bin seeds. Every variant uses the same haloes and the same resamples, so ratio errors are **paired**.

**Band ratio.** For each mass bin and radial band (<0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3, 3-4 R200c), this is the median of variant/reference over the shells that pass the strict rule. Its error is the SD of the same median over the 200 paired replicates. All values quoted per band are such band medians, not single-shell values.
- Location: `results.json -> band_ratios[variant][field][bin]`. `ratio_all` repeats the calculation with unresolved shells included.
- Tables give the **range over the 8 bins 13.5-15.1** (N = 127-895 haloes; 317-327 (N_eff 309-319) below 10^14.5). The two top bins (N = 21 and 6) are quoted separately where they differ.
- **Inner bands are resolved in fewer bins.** <0.1 R200c passes the strict rule only in the 3 bins 14.5-15.1, and 0.1-0.25 R200c only in the 6 bins 13.9-15.1. Ranges in those rows are over those bins.
- Largest paired errors over the 8 bins:
  - Q_V: <= 0.0064 (P) and <= 0.0053 (n_e) per band, both at 3-4 R200c in 14.9-15.1.
  - n_e kT_X: <= 0.043.
  - Statistics at 3-4 R200c, P: 0.071 (median), 0.069 (geometric mean), 0.085 (5% trimmed mean), 0.042 (Y200-weighted) and 0.027 (unweighted mean); all in 13.5-13.7 except the unweighted mean (14.1-14.3).

**Observable impact.** `sc.impact_from_stacks` uses the 10 calibrated bins x 35 shells and Test F's importance Jacobians.
- **Mean y and mean DM** are linear in the stacked profile (k = 1), so their changes are exact.
- **The tSZ C_ell, kSZ 1h and DM variance** are quadratic (k = 2). They also depend on where in radius and mass the change sits, so their changes are first-order only.
- Errors are the SD over the paired replicates. Values are in `results.json -> impact`.
- `dO_over_O_resolved_only` treats unresolved shells as unchanged.

**Scripts.**
- `compute_estimators.py` (100 s in compute.log, 44 s in the verifier's rerun; peak RAM 0.27 GB).
- `owner_sums.py` (6-7 s; peak RAM 0.36 GB, rechecked 2026-10-14 with an identical output). It streams STUDY/measurements_owner.npz and splits the particle sums by phase and ownership. The split sums reproduce the cache to <= 3e-11 (`diagnostics.mass_weighting_decomposition.unit_conversion_check_max_rel_dev`).
- `make_figures.py`. Figures are in groups/estimators/figures/.
- `recheck_numbers.py`, `check_section_round3.py` and `check_section_round4.py` (read-only) print the per-bin values behind the ranges in this section, including the identity extremes, the bootstrap ratios per field and the Q_M, Q_X and n_e kT_X mass trends. `check_ntx_weighting.py` (read-only, under 1 s) checks the n_e kT_X weighting in 1.2.

## 1. Shell estimator: how one halo's shell value is computed

### 1.1 The choice
A shell holds gas particles i with:
- mass m_i, SPH density rho_i and SPH volume V_i = m_i/rho_i;
- electron number N_e,i and stored Compton y_i;
- ROSAT 0.5-2 keV luminosity L_X,i.

The particle electron pressure is p_e,i = y_i m_e c^2 / (sigma_T V_i). The options:
- **geometric** (reference): F = sum_i q_i / V_shell. For P_e, sum y_i m_e c^2/sigma_T / V_shell; for n_e, sum N_e,i / V_shell.
- **Q_V** (B24 eq. 8): sum V_i q_i / sum V_i. For P_e this is sum N_e,i k T_i / sum V_i.
- **Q_V_y**: Q_V built from the particle y, i.e. sum y_i (m_e c^2/sigma_T) / sum V_i.
- **Q_M** (B24 eq. 10): sum m_i q_i / sum m_i. For P_e, sum m p_e / sum m; for the gas density, sum m rho / sum m.
- **Q_X** (B24 fiducial): sum L_X p_e / sum L_X over the X-ray phase (SFR <= 0, T >= 1e5 K, not recently AGN-heated).
- **n_e kT_X**: the geometric (mass-conserving) n_e times the L_X-weighted T_X of the X-ray phase. It isolates only the temperature-weighting part of the X-ray route. A real X-ray deprojection also uses an emission-measure-weighted density, proportional to sqrt(<n_e^2>) and so biased high by about sqrt(C_rho) of the emitting gas; this variant does not include that factor (not computed here).

### 1.2 Measured effect on the pressure stack (ratio to geometric; groups/estimators/figures/fig1_pressure_estimators.png)

| r/R200c | Q_V (= Q_V_y) | Q_M | Q_X | n_e kT_X |
|---|---|---|---|---|
| <0.1 (3 bins) | 1.029-1.035 | 1.028-1.035 | 1.067-1.076 | 0.990-0.993 |
| 0.1-0.25 (6 bins) | 1.025-1.035 | 1.034-1.083 | 1.087-1.191 | 0.992-0.998 |
| 0.25-0.5 | 1.027-1.043 | 1.055-1.206 | 1.126-1.512 | 0.990-1.015 |
| 0.5-1 | 1.037-1.045 | 1.106-1.377 | 1.313-2.197 | 0.977-1.048 |
| 1-1.5 | 1.056-1.058 | 1.36-1.84 | 2.08-4.72 | 0.943-1.117 |
| 1.5-2 | 1.082-1.088 | 2.07-4.18 | 3.5-22.7 | 0.924-1.215 |
| 2-3 | 1.120-1.130 | 5.9-15.9 | 16.5-115 | 1.043-1.383 |
| 3-4 | 1.183-1.251 | 22.7-46.7 | 117-429 | 1.419-1.603 |

- **Mass trends, Q_V.** Inside 3 R200c the Q_V range over the 8 bins is <= 0.016 per band. At 3-4 R200c Q_V rises with mass, from 1.183 (13.5-13.7) to 1.251 (14.9-15.1), because f falls with mass there (1.6(b)).
- **Mass trends, Q_M and Q_X.**
  - At 0.25-1 R200c both excesses grow with mass over 13.7-15.1. Q_M rises monotonically in both bands: from 1.055 (13.7-13.9) to 1.206 (14.9-15.1) at 0.25-0.5 R200c, and from 1.11 (13.5-13.7) to 1.38 (14.9-15.1) at 0.5-1 R200c. Q_X rises monotonically at 0.5-1 R200c, from 1.31 (13.7-13.9) to 2.20 (14.9-15.1); 13.5-13.7 has Q_X = 1.36 +- 0.05. At 0.25-0.5 R200c Q_X has one dip, from 1.166 (13.7-13.9) to 1.126 (13.9-14.1), and then rises to 1.512 (14.9-15.1); 14.1-14.3 and 14.3-14.5 are equal within errors (1.211 +- 0.036 and 1.200 +- 0.014).
  - Inside 0.25 R200c there is no consistent trend. At <0.1 R200c (14.5-15.1) Q_M = 1.028-1.035 and Q_X = 1.067-1.076, not ordered by mass. At 0.1-0.25 R200c neither rises over 13.9-14.5 (Q_M 1.039, 1.037, 1.034, errors 0.002-0.003; Q_X 1.088, 1.088, 1.087, errors 0.006-0.007); both rise only above 10^14.5, to 1.083 and 1.191 in 14.9-15.1.
  - In 15.3-15.5 (N = 6) both are lower again at 0.5-1 R200c (1.23 and 1.50).
- **Mass trends, n_e kT_X.** In every resolved shell of the 10 calibrated bins this ratio equals the per-halo T_X/T_e averaged with weights w P_e (w the inclusion weight), to <= 1.7e-5 (`check_ntx_weighting.py`). It is therefore not the ratio of the stacked temperatures in section 2: at 3-4 R200c it is 1.419-1.603 over the 8 bins, against T_X/T_e = 1.377-1.643.
  - Inside 0.25 R200c it is 0.990-0.998 in every resolved main bin.
  - In 13.5-14.5 it is > 1 everywhere beyond 0.25 R200c (1.001-1.603) and rises with radius.
  - At 1-2 R200c it is < 1 in 14.9-15.5: 0.94 (1-1.5 R200c) and 0.92 (1.5-2 R200c) in 14.9-15.1, and 0.84-0.90 (1-1.5) and 0.81-0.85 (1.5-2) in the two top bins. At 2-3 R200c it is < 1 only in 15.1-15.3 (0.83); 14.9-15.1 has 1.043 and 15.3-15.5 has 1.029 there.
  - 14.5-14.7 stays > 1 beyond 0.5 R200c (1.006-1.476), and 14.7-14.9 is 0.995-1.001 at 0.5-2 R200c.
  - At 3-4 R200c it is > 1 in every bin: 1.42-1.60 in the 8 bins, 1.19 and 1.38 in the top bins.
- **Q_V_y vs Q_V.** Q_V_y equals Q_V to the identity scatter in 1.6(a): per halo-shell P_QV/P_QV_y = 0.99963-1.00039 (0.1-99.9 percentiles over the 100,164 resolved halo-shells of all 15 bins; extremes 0.99899-1.00083).

### 1.3 Densities (groups/estimators/figures/fig3_density_estimators.png)

| r/R200c | n_e: Q_V/geometric | gas: Q_M/geometric |
|---|---|---|
| 0.25-0.5 | 1.027-1.044 | 6.1-24.8 |
| 0.5-1 | 1.037-1.044 | 15-52 |
| 1-1.5 | 1.057-1.059 | 107-199 |
| 2-3 | 1.122-1.134 | 1,690-2,850 |
| 3-4 | 1.199-1.263 | 7,150-10,500 |

The table gives band medians of the ratio of the stacked profiles. Per halo-shell the gas-density ratio is exactly C_rho/f (1.6(a)); the stacked ratio is a ratio of means and is not C_rho/f itself. The gas-density Q_V ratio equals the n_e one to <= 1e-3.

### 1.4 What each stack integrates to (`diagnostics.integral_ratios`, bins 13.5-15.5)

The table compares the volume integral of each stacked profile with that of the geometric stack. All shells are included, unresolved ones too.

| Estimator | Y or N_e (<1 R200c) | (<2 R200c) | (<4 R200c) |
|---|---|---|---|
| Q_V, pressure | 1.031-1.053 | 1.038-1.060 | 1.051-1.088 |
| Q_V, electrons | 1.032-1.062 | 1.044-1.067 | 1.091-1.112 |
| Q_M, pressure | 1.10-1.26 | 1.34-1.86 | 3.7-5.1 |
| Q_X, pressure | 1.25-1.77 | 2.0-12.1 | 15-50 |
| n_e kT_X | 0.952-1.044 | 0.930-1.092 | 0.948-1.214 |

The geometric shell integral reproduces each halo's own particle Y(<R200c) to 1.000000, for all 4,588 haloes (`diagnostics.check_Y200_own_vs_shell_integral`; the residual is a constant 5.9e-9 for every halo).

### 1.5 Observable impact of painting each estimator (dO/O in %, paired-bootstrap SD)

| Variant | tSZ ell = 100 | 1000 | 3000 | mean y | kSZ 1h ell = 100 | 1000 | 3000 | mean DM | DM variance |
|---|---|---|---|---|---|---|---|---|---|
| Q_V | +8.29 +-0.03 | +7.11 +-0.02 | +6.50 +-0.02 | +4.42 +-0.02 | +14.73 +-0.04 | +10.99 +-0.04 | +5.90 +-0.03 | +6.06 +-0.02 | +8.82 +-0.04 |
| Q_M (P) | +211 +-6 | +66 +-1 | +22 +-1 | +187 +-5 | - | | | | |
| Q_X (P) | +1721 +-148 | +320 +-26 | +42 +-9 | +1353 +-118 | - | | | | |
| n_e kT_X (P) | -0.83 +-0.25 | +0.48 +-0.15 | +0.65 +-0.09 | +6.52 +-0.10 | - | | | | |

- **Linear vs quadratic.** Q_V raises the pressure profile by 2.5-4.3% inside 0.5 R200c but by 18-25% at 3-4 R200c (band medians).
  - Mean y is linear in the profile and rises by +4.42%.
  - The tSZ C_ell is quadratic, so it rises by about twice the profile change: +8.29% at ell = 100, where the most massive haloes weigh most (14.5-15.5 give 6.4 of the 8.3 points), and less at ell = 3000 (+6.50%). Interpretation: the outer radii, where Q_V is highest, weigh more at low ell, and the inner radii dominate at ell = 3000.
  - The kSZ 1h at ell = 100 (+14.73%) against mean DM (+6.06%) shows the same pattern.
- **Resolved shells only.** Q_V is +8.07/+6.66/+5.73% for tSZ and +5.93% for mean DM.
- **Q_M and Q_X** are far outside the first-order regime. Only their mean-y numbers are exact; the C_ell numbers only show that the change is catastrophic.
- **n_e kT_X partly cancels in C_ell.** Its per-bin contributions to the ell = 100 tSZ change are positive for 13.5-14.9 and negative for 14.9-15.5 (`impact['estimators:nT_X'].P.tSZ_yy_total_ell100.per_bin`), giving -0.83 +- 0.25% in total. Mean y still rises by 6.52%.

### 1.6 Why

**(a) Conservation and identities.**
- **What the geometric estimator conserves.** It conserves each halo's particle Y and N_e exactly: sum_s F_s V_s = sum over particles of q_i (1.000000 above).
- **What the stack estimates.** The inclusion-weighted mean sum w F / sum w of these profiles, with w the inverse inclusion probabilities of the sample design, is the Hajek (ratio) form of the Horvitz-Thompson estimator of the population-mean profile. It is design-consistent, not an exact conservation. The population-mean profile is the quantity that mean y and mean DM are linear in. Every weighted estimator averages over a different measure, so its volume integral is not Y or N_e (table 1.4).
- **Identities built from the same particle sums.** These hold to rounding (0.1-99.9 percentiles over resolved halo-shells of all bins, `diagnostics.identities`):
  - P_QV_y = P_geom / f and n_e,QV = n_e,geom / f, with f = sum V_i / V_shell. f from n_e and f from the gas density agree to 6 digits.
  - Per halo-shell, gas-density Q_M = <rho>_V C_rho, with C_rho = <rho^2>_V/<rho>_V^2. Hence Q_M/geometric = C_rho/f.
- **Identities that compare y with N_e kT.** Three ratios compare the stored ComptonY with N_e kT built from the stored electron density, temperature, mass and density: P_QV/P_QV_y, P_geom/(n_e k T_e) and the n_e kT_X identity.
  - Their 0.1-99.9 percentile range is 0.99961-1.00039. 157 of the 100,164 resolved halo-shells (0.16%) deviate by more than 4e-4, and the largest deviation is 1.0e-3 (P_geom/(n_e k T_e) spans 0.99917-1.00101; `recheck_numbers.json`).
  - The scatter falls as N^-1/2 with the particle count N in the halo-shell. The median |dev| x sqrt(N) is 6.3-7.1e-4 for every N range from 20-100 up to > 10^4 (`diagnostics.identity_noise`).
  - Estimate: this implies a per-particle mismatch of about 0.9-1.0e-3 between y and N_e kT. float32 rounding would give about 1e-7.
  - Hypothesis, not tested: lossy compression of the stored particle fields.
  - Hence **P_e = n_e k T_e holds to 4e-4 for 99.84% of resolved halo-shells, with a maximum deviation of 1.0e-3, not exactly**. This is negligible for painting.

**(b) Q_V reads high because the SPH volumes do not fill the shell** (groups/estimators/figures/fig2_filling_factor_clumping.png, `diagnostics.filling_factor`).

| r/R200c | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 |
|---|---|---|---|---|---|---|---|---|
| mean f (resolved bins of 13.5-15.5) | 0.964-0.970 | 0.963-0.975 | 0.956-0.973 | 0.956-0.966 | 0.943-0.949 | 0.916-0.923 | 0.878-0.888 | 0.791-0.829 |

The <0.1 and 0.1-0.25 columns are over the 5 and 8 resolved bins (14.5-15.5 and 13.9-15.5).

- **f depends mainly on r/R200c.** Inside 3 R200c it varies by <= 0.017 across the bins in each band. At 3-4 R200c it falls with mass, from 0.829 (13.5-13.7) to 0.791 (14.9-15.1). It is 0.823 at 3-4 R200c in the exploratory 12.7-12.9 bin. The halo-to-halo 16-84% range is 0.77-0.84 at 3-4 R200c (14.3-14.5, the band shaded in fig2a; 14.5-14.7 gives the same).
- **1/<f> explains Q_V.** It reproduces the stacked Q_V/geometric ratio to a median 0.10-0.22% per bin (maximum 0.6-2.9%).
- **Not recently AGN-heated gas.** Keeping that gas gives the same pooled median f (resolved halo-shells of the 10 calibrated bins) to 4 digits in every band (0.8124 at 3-4 R200c).
- **Not mainly clumping.** At fixed band, per halo-shell f anticorrelates only weakly with ln C_rho (Spearman -0.18 to -0.34). Smooth halo-shells (C_rho < 1.5) still have a median f = 0.898 at 2-3 R200c (n = 768), against 0.879 for C_rho >= 100 (n = 6,454). At 3-4 R200c the values are 0.857 (n = 31) and 0.810 (n = 7,696). So the smooth/clumpy difference is 0.02-0.05 out of a pooled median deficit 1 - f of 0.12 (2-3 R200c) to 0.19 (3-4 R200c).
- **Not a kernel-scale density-gradient bias.** The median h/r falls from 0.33 (<0.1 R200c) to 0.063 (3-4 R200c) while the deficit grows, and Spearman(f, h/r) is -0.16 to +0.23.
- **Interpretation, not tested.** The SPH density is a kernel sum that includes the particle's own mass. sum m_i/rho_i equals the volume only for a regular, glass-like particle arrangement. In the disordered outskirts, rho_i is biased high and V_i low.
- **Consequence.** Q_V measures P/f, not P.

**(c) Q_M: dense cold/star-forming and substructure-bound gas (a minority of the mass) carries most of sum(m rho) at every radius and most of sum(m p_e) beyond 2 R200c; the smooth hot gas alone is still biased, and inside 1 R200c it gives most of the pressure excess** (groups/estimators/figures/fig11_mass_weighting_decomposition.png, `diagnostics.mass_weighting_decomposition`).

- **Exact identity.** P_QM/P_QV = <rho P>_V/(<rho>_V <P>_V) = 1 + r sqrt((C_rho - 1)(C_P - 1)), with SPH-volume moments and r the SPH-volume correlation coefficient of rho and P.
- **Clumping and correlation (14.5-14.7, by band from <0.1 to 3-4 R200c, halo medians, `diagnostics.clumping`):**
  - median C_rho = 1.03, 1.08, 1.97, 3.56, 23, 244, 1,500, 6,200;
  - median C_P = 1.03, 1.03, 1.08, 1.18, 1.39, 1.73, 2.44, 7.3;
  - rho-P correlation r = 0.37, 0.33, 0.22, 0.16, 0.07, 0.04, 0.05, 0.07.
- **Who carries the mass-weighted sums.** The split by phase and ownership answers this. Cold/SF means star-forming or T < 1e5 K. Substructure-bound means bound to any subhalo other than the central. Ranges are over the 8 bins 13.5-15.1.

| r/R200c | cold/SF: share of gas mass | cold/SF: share of sum(m rho) | cold/SF: share of sum y | substructure-bound: share of gas mass | substructure-bound: share of sum y | substructure-bound: share of sum(m p_e) |
|---|---|---|---|---|---|---|
| 0.25-0.5 | 0.6-2.4% | 81-96% | <= 3.7e-5 | 0.2-1.4% | 0.01-0.4% | 0.04-1.2% |
| 0.5-1 | 0.5-1.8% | 91-98% | <= 4.1e-5 | 0.6-1.9% | 0.09-1.5% | 0.9-8.5% |
| 1-1.5 | 0.6-1.5% | 98.4-99.4% | <= 5.8e-5 | 1.5-3.6% | 0.7-3.8% | 11-29% |
| 1.5-2 | 1.0-1.4% | 99.5-99.7% | <= 7.9e-5 | 2.2-6.9% | 1.4-6.3% | 35-52% |
| 2-3 | 1.6-1.9% | 99.7-99.9% | <= 1.2e-4 | 3.6-13% | 3.3-15% | 58-78% |
| 3-4 | 3.8-4.4% | 99.8-99.9% | <= 4.4e-4 | 9.1-18% | 12-33% | 83-89% |

The table below gives Q_M / geometric with both stacks built over the same remaining gas (8 bins 13.5-15.1).

| r/R200c | gas density: all gas | gas density: without cold/SF | gas density: smooth hot only | pressure: all gas | pressure: without cold/SF | pressure: without substructure-bound | pressure: smooth hot only |
|---|---|---|---|---|---|---|---|
| 0.25-0.5 | 6.1-24.8 | 1.07-1.22 | 1.07-1.21 | 1.055-1.21 | 1.07-1.21 | 1.06-1.20 | 1.07-1.21 |
| 0.5-1 | 15-52 | 1.12-1.40 | 1.10-1.32 | 1.11-1.38 | 1.12-1.38 | 1.11-1.30 | 1.12-1.31 |
| 1-1.5 | 107-199 | 1.20-1.74 | 1.18-1.55 | 1.36-1.84 | 1.23-1.83 | 1.20-1.47 | 1.20-1.46 |
| 2-3 | 1,690-2,850 | 1.95-8.7 | 1.63-4.9 | 5.9-15.9 | 2.0-14.6 | 1.8-8.0 | 1.8-7.9 |
| 3-4 | 7,150-10,500 | 4.3-18 | 2.8-7.9 | 22.7-46.7 | 5.4-38.9 | 4.5-13.2 | 3.5-12.8 |

Here 'smooth hot only' means central-bound or unbound X-ray-phase gas. The gas that is not smooth hot (cold/SF or substructure-bound) is 10-19% of the gas mass at 3-4 R200c.

- **Gas density.** At 0.25-4 R200c cold/SF gas holds only 0.5-4.4% of the gas mass but 81-99.9% of sum(m rho), because each such particle sits at an enormous density. Removing it brings the gas-density Q_M/geometric from 6-10,500 down to 1.07-18 over the same bands.
- **Pressure.** Cold/SF gas carries <= 4.4e-4 of the shell's volume-integrated y.
  - In groups it still supplies up to 74% of sum(m p_e) at 3-4 R200c (13.5-13.7); in 14.7-15.1 this falls to 19-23%.
  - Inside 1.5 R200c the smooth hot gas carries most of sum(m p_e): 91-99.9% at 0.25-1 R200c and 71-88% at 1-1.5 R200c (47-64% at 1.5-2 R200c, 22-41% at 2-3 R200c and 10-16% at 3-4 R200c).
  - In all 8 bins, groups and clusters alike, substructure-bound gas holds 12-33% of y at 3-4 R200c but 83-89% of sum(m p_e).
  - Removing the substructure-bound gas lowers the 3-4 R200c pressure Q_M/geometric from 22.7-46.7 to 4.5-13.2.
- **The smooth hot gas alone is still biased.** Its pressure Q_M/geometric is 1.07-1.31 at 0.25-1 R200c, 1.8-7.9 at 2-3 R200c and 3.5-12.8 at 3-4 R200c. In log terms that is 0.84-1.35 times the excess of all gas at 0.25-1 R200c (so at least 84% of it), 31-75% at 2-3 R200c and 40-69% at 3-4 R200c. Its gas-density Q_M/geometric is 2.8-7.9 at 3-4 R200c (12-22% of the log-excess of all gas). Per halo-shell this residual is (1 + r sqrt((C_rho - 1)(C_P - 1)))/f of that gas: its own clumping (median C_rho 2.1-5.3 at 3-4 R200c) and density-pressure correlation, divided by f.
- **The mechanism.** Mass weighting reads the density and pressure of the gas that carries sum(m rho) and sum(m p_e), not the volume average that y and DM see. Two parts contribute. A minority of the gas mass at high density (cold/SF 0.5-4.4%, substructure-bound up to 18%) dominates sum(m rho) at every radius and sum(m p_e) beyond 2 R200c. The clumping and density-pressure correlation of the smooth hot gas also contribute. For pressure they give at least 84% of the log-excess at 0.25-1 R200c, 31-75% at 2-3 R200c and 40-69% at 3-4 R200c.

**(d) Q_X weights by L_X ~ rho^2 Lambda(T) over the X-ray phase.** It therefore favours dense gas even more strongly than Q_M; at 3-4 R200c it reads 117-429 x geometric. n_e kT_X inherits the P_e-weighted T_X/T_e (1.2; section 2 explains T_X/T_e).

### 1.7 Recommendation (estimator)
- **Paint and fit the geometric estimator for P_e and n_e.** It integrates back to each halo's particle Y and N_e exactly, and only its stack estimates the population-mean profile that mean y and mean DM are linear in.
- **Never calibrate on Q_V.** Q_V (B24 eq. 8) puts +8.3% into the tSZ C_ell at ell = 100.
- **Comparing with B24 needs more than f.** Per halo-shell Q_V = geometric / f converts the estimator (f table in 1.6(b)). A like-for-like comparison also needs B24's other choices (research/methods_survey.md):
  - their fiducial estimator is Q_X, not Q_V;
  - they stack the median (with 16-84%), not the mean;
  - they use X-ray-phase gas only (no star-forming gas, T > 1e5 K, no AGN heating in the last 15 Myr);
  - they use 0.5-dex M500c bins and 30 log bins of r/R500c (0.01-3 R500c).
  The median alone is 0.51-0.72 x the mean for P at 3-4 R200c (`band_ratios['estimators:median'].P`), so dividing by f does not give a like-for-like comparison.
- **Never use Q_M, Q_X or n_e kT_X as calibration inputs.** n_e kT_X looks harmless in the C_ell (-0.83 +- 0.25% at ell = 100) only because positive per-bin contributions from 13.5-14.9 and negative ones from 14.9-15.5 cancel; mean y is still +6.52%.

## 2. Temperature definitions (groups/estimators/figures/fig4_temperature_definitions.png, fig10_tx_decomposition.png)

**The options:**
- T_M = sum m T / sum m;
- T_V = sum V T / sum V;
- T_X = sum L_X T / sum L_X over the X-ray phase;
- T_e = sum N_e kT / sum N_e (reference);
- <P>/(k<n_e>): the temperature implied by the two stacked reference profiles.

| r/R200c | T_M/T_e | T_V/T_e | T_X/T_e | <P>/(k<n_e>) / <T_e> |
|---|---|---|---|---|
| <0.1 (3 bins) | 0.976-0.983 | 1.007-1.010 | 0.985-0.990 | 1.015-1.024 |
| 0.1-0.25 (6 bins) | 0.985-0.992 | 1.002-1.010 | 0.988-0.997 | 0.992-1.020 |
| 0.25-0.5 | 0.990-0.995 | 0.998-1.012 | 0.987-1.017 | 0.990-1.008 |
| 0.5-1 | 0.993-0.997 | 0.991-1.010 | 0.977-1.050 | 0.995-1.004 |
| 1-1.5 | 0.995-0.997 | 0.993-1.037 | 0.943-1.116 | 1.016-1.033 |
| 1.5-2 | 0.995-0.996 | 1.001-1.112 | 0.917-1.215 | 1.044-1.069 |
| 2-3 | 0.993-0.995 | 1.009-1.265 | 0.984-1.391 | 1.094-1.163 |
| 3-4 | 0.990-0.992 | 1.020-1.470 | 1.377-1.643 | 1.140-1.342 |

Top bins: T_X/T_e is 0.82-0.90 at 1-1.5 R200c and 0.82 at 1.5-2 R200c, and T_V/T_e is 1.65-1.68 at 3-4 R200c. Paired errors over the 8 bins are <= 0.003 for T_M, <= 0.042 for T_V, <= 0.038 for T_X and <= 0.103 for <P>/(k<n_e>); the T_V, T_X and <P>/(k<n_e>) maxima are at 3-4 R200c.

**Why T_X differs from T_e: the L_X weighting, not the phase cut.** The ratio splits exactly into two factors: T_X/T_e = (T_e of the X-ray phase / T_e of all gas) x (T_X / T_e of the X-ray phase). The table is over the 8 bins 13.5-15.1.

| r/R200c | stacked kT_e [keV] | T_X/T_e | phase cut | L_X weighting | L_X weighting, central-bound + unbound gas only | share of X-ray-phase L_X from substructure-bound gas |
|---|---|---|---|---|---|---|
| <0.1 (3 bins) | 5.1-8.4 | 0.985-0.990 | 1.004-1.005 | 0.981-0.985 | 0.982-0.985 | 0.02-0.1% |
| 0.25-0.5 | 0.94-5.2 | 0.987-1.017 | 1.001-1.016 | 0.986-1.001 | 0.988-1.003 | 0.2-1.7% |
| 0.5-1 | 0.66-3.8 | 0.977-1.050 | 1.002-1.012 | 0.975-1.038 | 0.983-1.040 | 0.7-12% |
| 1-1.5 | 0.41-2.5 | 0.943-1.116 | 1.004-1.010 | 0.940-1.105 | 0.962-1.102 | 7-33% |
| 1.5-2 | 0.30-1.7 | 0.917-1.215 | 1.006-1.009 | 0.912-1.204 | 0.951-1.204 | 14-59% |
| 2-3 | 0.23-1.16 | 0.984-1.391 | 1.010-1.012 | 0.974-1.376 | 1.002-1.392 | 34-77% |
| 3-4 | 0.19-0.63 | 1.377-1.643 | 1.026-1.031 | 1.335-1.601 | 1.318-1.683 | 67-88% |

- **The phase cut contributes little.** Excluding star-forming, T < 1e5 K and AGN-heated gas changes the electron-weighted temperature by only 0.1-3.1%. Cool, partly neutral gas has few electrons per unit mass, so it barely enters T_e anyway.
- **The L_X weighting inside the X-ray phase gives the rest.** Its sign follows the gas temperature (fig10 panel e). Over the resolved bin-bands of the 10 calibrated bins 13.5-15.5:
  - in all 21 with kT_e < 0.7 keV the factor is 1.038-1.601;
  - in all 35 with kT_e >= 1.5 keV it is 0.812-1.008 (0.912-1.008 in the 22 of these that lie in 13.5-15.1);
  - in the 17 between 0.7 and 1.5 keV it ranges from 0.825 to 1.293.
- **Interpretation, not tested here.** The ROSAT 0.5-2 keV emissivity per unit emission measure rises steeply with T below about 1 keV, and is flat or falling above about 2 keV. Combined with the rho^2 weighting, L_X picks the hottest gas in cool shells and the densest, cooler gas in hot ones.
- **Substructure does not set the L_X weighting in groups.** Gas bound to other subhaloes emits 67-88% of the X-ray-phase L_X at 3-4 R200c. Yet the L_X weighting computed on central-bound + unbound gas alone is as large or larger: 1.63-1.68 against 1.55-1.60 for all gas in 13.5-14.1 at 3-4 R200c (1.32-1.68 against 1.34-1.60 over the 8 bins).
- **Substructure changes T_X directly only in clusters.** T_X(all)/T_X(smooth) is within 0.4% of 1 at 1-3 R200c in 13.5-14.1 (0.999-1.004). In 14.3-15.1 the substructure gas lowers T_X by up to 5% at 1-3 R200c (0.948-0.994) and raises it by 5-8% at 3-4 R200c (1.046-1.078).
- **T_M < T_e.** Mass weighting gives relatively more weight to cool, partly neutral gas, which has fewer electrons per unit mass. T_M is within 1% of T_e beyond 0.25 R200c (0.990-0.997); it is 0.985-0.992 at 0.1-0.25 R200c and 0.976-0.983 inside 0.1 R200c.
- **T_V reads high in cluster outskirts.** At 3-4 R200c T_V/T_e rises with mass, from 1.02 +- 0.02 (13.5-13.7) to 1.47 +- 0.04 (14.9-15.1), and is 1.65-1.68 in the top bins. Interpretation, not tested: SPH-volume weighting up-weights low-density gas, which is hotter there.
- **<P>/(k<n_e>) vs <T_e>.** <P>/(k<n_e>) is the n_e-weighted mean of the per-halo T_e, so it differs from <T_e> by the halo-to-halo covariance of n_e and T_e. That covariance is positive in the outskirts: 1.09-1.16 at 2-3 R200c and 1.14-1.34 at 3-4 R200c. At 3-4 R200c the values run from 1.14 +- 0.02 (14.9-15.1) to 1.34 +- 0.10 (13.5-13.7), with 1.24-1.32 in 13.7-14.9.

**Recommendation (temperature).**
- **Use T_e (electron-weighted).** Use it for entropy, for consistency checks of the painted P_e and n_e, and for any temperature tied to them. A painter of the mean P_e and mean n_e profiles implicitly assumes <P>/(k<n_e>), which exceeds the stacked T_e by 14-34% at 3-4 R200c in the 8 bins (5-14% in the top bins).
- **T_X and T_V** are for comparison with X-ray work only.
- **Relativistic tSZ.** The first-order correction needs the Compton-y-weighted temperature sum(y T)/sum(y). None of the cached temperatures is that quantity, so store sum(y T) (section 5).

## 3. Stacking statistic: how haloes are combined (groups/estimators/figures/fig5_statistics_vs_mean.png, fig6_lognormal_test.png, fig7_median_geomean_heatmap.png)

**Options:**
- inclusion-weighted mean (reference);
- unweighted mean;
- inclusion-weighted median;
- geometric mean exp<ln F> (log-mean; haloes with F <= 0 are dropped);
- Y200-weighted mean (B12);
- 5% trimmed mean.

Literature, from research/methods_survey.md:
- B24, A10, Le Brun+14 and McCarthy+14/17 use medians (McCarthy+17 also the mean for Y).
- B12 uses Y_Delta weighting.
- Nagai+07 use the mean and scatter of log.
- Lee22 stack the mean and use a lognormal likelihood with a covariance.

**Ratio to the inclusion-weighted mean, pressure P_e (8 bins 13.5-15.1):**

| r/R200c | median | geometric mean | 5% trimmed | Y200-weighted | unweighted |
|---|---|---|---|---|---|
| <0.1 (3 bins) | 0.952-0.971 | 0.904-0.927 | 0.961-0.985 | 1.024-1.033 | 1.000 |
| 0.1-0.25 (6 bins) | 0.940-0.996 | 0.935-0.950 | 0.973-0.996 | 1.020-1.035 | 0.997-1.000 |
| 0.25-0.5 | 0.974-1.011 | 0.967-0.984 | 0.991-0.999 | 1.012-1.059 | 0.997-1.005 |
| 0.5-1 | 0.982-0.999 | 0.983-0.993 | 0.995-0.998 | 0.998-1.038 | 1.000-1.005 |
| 1-1.5 | 0.931-0.979 | 0.933-0.958 | 0.962-0.983 | 0.986-1.024 | 1.000-1.019 |
| 1.5-2 | 0.840-0.937 | 0.873-0.905 | 0.920-0.964 | 0.987-1.026 | 1.000-1.042 |
| 2-3 | 0.701-0.840 | 0.762-0.830 | 0.806-0.905 | 0.980-1.045 | 1.000-1.083 |
| 3-4 | 0.510-0.724 | 0.580-0.751 | 0.707-0.869 | 0.984-1.051 | 1.000-1.073 |

**Electron density n_e:**

| r/R200c | median | geometric mean | 5% trimmed | Y200-weighted | unweighted |
|---|---|---|---|---|---|
| 0.25-0.5 | 0.951-1.009 | 0.967-0.993 | 0.983-1.002 | 1.007-1.064 | 0.997-1.004 |
| 0.5-1 | 0.994-1.004 | 0.984-0.997 | 0.996-1.001 | 0.997-1.046 | 1.000-1.004 |
| 1-1.5 | 0.972-0.997 | 0.979-0.988 | 0.985-0.998 | 0.992-1.023 | 1.000-1.009 |
| 2-3 | 0.887-0.934 | 0.937-0.962 | 0.938-0.982 | 0.992-1.011 | 1.000-1.041 |
| 3-4 | 0.839-0.951 | 0.913-0.947 | 0.933-0.968 | 0.995-1.016 | 1.000-1.030 |

T_e and K are less affected: at 3-4 R200c the median gives 0.786-0.914 (T_e) and 0.918-0.966 (K), and the geometric mean 0.844-0.896 and 0.945-0.962.

**Observable impact (dO/O %, paired SD):**

| Statistic | tSZ ell = 100 | 1000 | 3000 | mean y | kSZ 1h ell = 100 | 1000 | 3000 | mean DM | DM variance |
|---|---|---|---|---|---|---|---|---|---|
| median | -5.04 +-1.82 | -2.83 +-1.43 | -1.78 +-0.81 | -5.53 +-0.20 | -6.89 +-0.29 | -4.54 +-0.23 | -1.61 +-0.19 | -3.15 +-0.14 | -3.38 +-0.19 |
| geometric mean | -8.30 +-0.34 | -7.26 +-0.25 | -6.64 +-0.20 | -5.12 +-0.14 | -5.21 +-0.10 | -3.91 +-0.08 | -2.19 +-0.05 | -2.16 +-0.05 | -3.41 +-0.06 |
| 5% trimmed | -3.05 +-0.26 | -2.15 +-0.22 | -1.62 +-0.18 | -2.83 +-0.16 | -2.96 +-0.10 | -1.90 +-0.08 | -0.68 +-0.06 | -1.29 +-0.06 | -1.44 +-0.06 |
| Y200-weighted | +1.69 +-0.20 | +2.55 +-0.21 | +3.50 +-0.22 | +1.11 +-0.15 | +1.29 +-0.16 | +1.86 +-0.16 | +2.51 +-0.13 | +0.64 +-0.10 | +2.30 +-0.12 |
| unweighted | +0.36 +-0.03 | +0.35 +-0.03 | +0.16 +-0.05 | +0.71 +-0.07 | +1.30 +-0.08 | +1.09 +-0.07 | +0.31 +-0.05 | +0.64 +-0.05 | +0.58 +-0.05 |

The median's tSZ changes are uncertain (-5.04 +- 1.82% at ell = 100, -2.83 +- 1.43% at ell = 1000); its mean-y, kSZ and DM changes are well determined.

**Why.**
- **Median and geometric mean: the halo-to-halo distribution of F is skewed** (`diagnostics.distribution`, fig6; P, 8 bins).
  - Inside 1.5 R200c the lognormal expectation works. mean/geomean equals exp(sd_ln^2/2) to <= 0.1% at 0.25-1 R200c (for P, 1.008-1.018 vs 1.008-1.018 at 0.5-1 R200c), and to <= 0.8% inside 0.25 R200c and at 1-1.5 R200c (largest 0.75%, at <0.1 R200c in 14.5-14.7), although ln P is somewhat skewed there (skewness -0.75 to +0.54).
  - The geometric mean is therefore low there by about the scatter term. With sd_ln P = 0.40-0.46 inside 0.1 R200c, that is 7-10%.
  - Beyond 2 R200c, ln P is positively skewed (skewness 0.40-1.15 at 2-3 R200c and 0.32-0.97 at 3-4 R200c) and sd_ln reaches 0.74-0.87 at 3-4 R200c.
  - At 3-4 R200c, mean/geomean is 1.33-1.72, against the lognormal prediction of 1.32-1.47. A 4th-order cumulant expansion, exp(k2/2 + k3/6 + k4/24), recovers 1.33-1.68. mean/median is 1.38-1.96.
  - So a 'median (or log-mean) + lognormal scatter' model under-predicts the mean in the outskirts, even with the measured sd_ln.
  - The halo model's mean y and mean DM are linear in the population mean (n(M) <Y>), so the mean is the right target.
- **Y200 weighting.** Exactly, Y-weighted/mean = 1 + corr(Y200, F) cv(Y200) cv(F) per shell.
  - Inside 0.5 R200c, F correlates with the halo's own Y200 (corr 0.27-0.68), so the weighting raises the profile by 1.2-5.9%.
  - Beyond 1.5 R200c the correlation is weak (-0.11 to +0.17), but cv(F) is large there. The Y200-weighted P therefore still scatters between 0.980 and 1.051 at 2-4 R200c (paired errors 0.007-0.042), with no consistent sign across mass; for example 1.045 +- 0.016 (2-3 R200c) and 1.051 +- 0.023 (3-4 R200c) in 13.7-13.9.
  - Overall it re-weights each bin towards high-Y haloes, which steepens the inner profile and lifts the high-ell tSZ most (+3.50% at ell = 3000).
- **Trimmed mean.** It cuts the high tail of the positively skewed outskirts: -13 to -29% for P at 3-4 R200c.
- **Unweighted mean.** The sample design oversamples the CONT-annulus class (class 2: an overlapper with mu >= 0.1 that reaches only the 1-2 R200c annulus): 15.3-15.8% unweighted against 9.7-10.0% weighted in 13.5-14.5. The CONT-1R class is not oversampled (19.4-24.5% unweighted, 22.6-24.2% weighted).
  - The unweighted P is higher at 2-3 R200c by up to +8.3% (14.1-14.3: 1.083 +- 0.009; 1.031-1.083 over 13.5-14.5), so the over-represented haloes evidently carry more pressure there.
  - Above 10^14.5 all haloes are included with equal weight, so the two means are identical. The tSZ change therefore comes only from 13.5-14.5, which is why it is only +0.36% at ell = 100.
  - In wide bins the unweighted mean also shifts the mean mass by +0.041 to +0.059 dex (0.5-dex bins 12.5-14.5), because the design takes ~300 haloes per 0.2 dex regardless of the mass function. For example, P in 13.5-14.0 is +5.4 +- 0.3% at 0.25-0.5 R200c (`diagnostics.unweighted_vs_weighted_M200c_0.5`).

**Recommendation (statistic).**
- **Stack the inclusion-weighted mean and paint it.**
- **Model halo-to-halo scatter separately**, if needed for the 1-halo terms, around that mean. Do not substitute a median or log-mean as the profile. At ell = 100 they change the tSZ C_ell by -5.0 +- 1.8% (median) and -8.3 +- 0.3% (log-mean), mean y by -5.5% and -5.1%, and mean DM by -3.2% and -2.2%. A lognormal centred on the median does not recover the mean beyond 2 R200c.
- **Never drop the inclusion weights below 10^14.5.**

## 4. Bootstrap errors (groups/estimators/figures/fig8_bootstrap_errors.png, `results.bootstrap`)

| Quantity, 8 bins 13.5-15.1 (top bins 15.1-15.3, 15.3-15.5) | 0.25-0.5 | 0.5-1 | 1-1.5 | 2-3 | 3-4 R200c |
|---|---|---|---|---|---|
| relative error of the mean P | 0.65-1.7% (2.4, 2.5%) | 0.54-1.4% (2.8, 4.6%) | 1.2-2.7% (5.5, 11.9%) | 2.8-7.1% (9.2, 13.1%) | 6.2-14.4% (18.5, 21.6%) |
| relative error of the mean n_e | 0.45-1.5% (1.7, 1.8%) | 0.34-1.0% (1.8, 2.4%) | 0.68-1.6% (4.0, 5.8%) | 1.1-2.6% (5.4, 6.0%) | 1.7-3.0% (7.0, 7.7%) |
| P: bootstrap SD / (sd/sqrt(N_eff)) | 0.90-1.06 (0.90, 0.54) | 0.95-1.06 (0.91, 0.58) | 0.92-1.02 (0.94, 0.82) | 0.73-1.04, median 0.85 (0.92, 0.82) | 0.90-1.05 (0.94, 0.88) |
| P: unstratified / stratified bootstrap | 0.90-1.18 (1.21, 1.71) | 0.91-1.07 (1.11, 1.60) | 0.93-1.13 (1.08, 1.12) | 0.97-1.32 (1.06, 1.09) | 0.89-1.12 (0.99, 1.04) |
| n_e: unstratified / stratified bootstrap | 0.91-1.16 (1.09, 1.60) | 0.90-1.09 (1.13, 1.89) | 0.93-1.17 (1.04, 1.14) | 0.93-1.33 (1.09, 1.14) | 0.92-1.14 (1.01, 1.08) |
| P: 4^3-block jackknife / bootstrap | 0.82-1.16, median 0.96 (1.11, 1.84) | 0.83-1.08, median 0.94 (1.09, 1.71) | 0.94-1.04, median 1.00 (1.08, 1.22) | 0.91-1.28, median 1.11 (1.41, 1.22) | 0.98-1.15, median 1.03 (1.36, 1.14) |
| n_e: 4^3-block jackknife / bootstrap | 0.83-1.07, median 0.92 (1.11, 1.83) | 0.84-1.10, median 0.98 (1.07, 2.04) | 0.92-1.08, median 1.01 (1.06, 1.24) | 0.88-1.39, median 1.18 (1.36, 1.23) | 0.93-1.17, median 1.05 (1.37, 1.23) |
| P: SD of median / SD of mean | 1.07-1.44 | 0.90-1.36 | 0.82-1.14 | 0.39-0.89 | 0.19-0.71 |
| N_eff for a 3% error of the mean P (estimate, N_eff^-1/2 scaling) | 27-68 | 17-42 | 101-166 | 349-1,400 | 980-7,300 |
| N_eff for a 3% error of the mean n_e (estimate) | 12-83 | 7-37 | 27-47 | 74-123 | 129-275 |

Medians are over the 8 bins. Noise estimates (from `recheck_numbers.json`): a 200-draw bootstrap SD is uncertain by about 5%, and a 64-block jackknife SD by about 9%, so single-bin ratios scatter by about +-0.1.

- **Inside 1.5 R200c the bootstrap SD agrees with the analytic sd/sqrt(N_eff).** The ratio is 0.90-1.11 there (band medians 0.93-1.01 for P). At 1.5-3 R200c the stratified bootstrap is below the analytic value (band medians 0.85-0.90 for P and 0.83-0.89 for n_e, down to 0.73).
- **The median is noisier in the inner halo.** Its SD is 1.07-1.44 x that of the mean at 0.25-0.5 R200c and 1.00-1.36 inside 0.25 R200c, around the Gaussian sqrt(pi/2) = 1.25.
- **The median is quieter in the outskirts, but biased.** Beyond 2 R200c its SD is 0.19-0.89 x that of the mean, but it is biased low by 16-49% (section 3).
- **Stratification lowers the error estimate modestly in 13.5-15.1.** The unstratified/stratified ratio is 0.88-1.32 for P and 0.90-1.33 for n_e over all bands. Its band medians are 1.01-1.10 for P (largest at 0.1-0.25 R200c) and 0.95-1.14 for n_e (largest at 2-3 R200c).
- **Stratification fails in the top bins.** The design strata are 17/2/2 haloes in 15.1-15.3 and 2/3/1 in 15.3-15.5, so a stratum of one or two haloes adds little or no bootstrap variance.
  - In 15.3-15.5 the unstratified bootstrap exceeds the stratified one by up to x1.71 for P (0.25-0.5 R200c), x1.89 for n_e (0.5-1 R200c), x1.88 for gas density (0.5-1 R200c) and x1.84 for T_e (0.1-0.25 R200c). In 15.1-15.3 the factor is up to 1.22-1.27.
  - The jackknife is 1.14-1.84 x the stratified bootstrap for P in 15.3-15.5 and 1.23-2.04 x for n_e; in 15.1-15.3 it is 1.08-1.41 (P) and 1.04-1.37 (n_e).
  - With N = 21 and 6 haloes, any SD estimate is itself uncertain by about 16% and 32% (1/sqrt(2(N-1)), estimate).
- **Spatial correlation raises the errors beyond 1.5 R200c in the main bins.** Inside 1.5 R200c the jackknife/bootstrap band medians are 0.90-1.00 (P) and 0.89-1.01 (n_e), so the bootstrap is adequate there. At 1.5-2 and 2-3 R200c the band medians are 1.09 and 1.11 (P) and 1.15 and 1.18 (n_e), with 6-7 of the 8 bins above 1. Single bins reach x1.28 (P, 14.1-14.3) and x1.39 (n_e, 14.7-14.9) at 2-3 R200c. At 3-4 R200c the medians are 1.03 (P) and 1.05 (n_e).
  - Interpretation, not tested: neighbouring haloes share outskirts gas and large-scale environment, so their outer profiles are correlated, and resampling haloes as independent understates the error.
  - At 1.5-3 R200c the stratified bootstrap is thus the smallest of the four error estimates (analytic, unstratified, jackknife).
- **The outskirts are statistics-limited.** At z = 0 with 127-895 haloes per main bin, the mean pressure at 3-4 R200c is known to 6.2-14.4%. A 3% error would need N_eff of about 1,000-7,300 per bin (estimate).

**Recommendation (errors).**
- **13.5-15.1, inside 1.5 R200c:** use the paired stratified bootstrap of the inclusion-weighted mean.
- **13.5-15.1, beyond 1.5 R200c:** scale the bootstrap SD by the measured jackknife/bootstrap band median (P: 1.09, 1.11, 1.03 at 1.5-2, 2-3, 3-4 R200c; n_e: 1.15, 1.18, 1.05), or use max(bootstrap, jackknife) per shell. The max is biased slightly high, because each jackknife SD carries about 9% noise.
- **15.1-15.5:** use the larger of the unstratified bootstrap and the 4^3-block jackknife, and treat the error itself as uncertain by 16-32%.
- **For outskirt precision,** use all haloes of each snapshot above the calibrated mass, not 300 per bin, or pool adjacent snapshots through a smooth mass-redshift fit.

## 5. For the ~100-snapshot campaign: what to compute and store

- **One particle pass per snapshot that stores the linear per-particle sums.** Keep them per halo, per ownership class, per phase (recent AGN / X-ray / other) and per shell, as measure_overlap.py already does (it also stores per-shell histograms of h/R200c and h/R500c). Its 16 quantities are: count, m, V = m/rho, y, N_e, N_e kT, P_gas V, m p_e, V p_e^2, m T, V T, m rho, L_X, L_X p_e, L_X T and y^2. Every estimator and temperature in this group is a ratio of these sums:
  - geometric = sum q / V_shell;
  - Q_V(P) = sum N_e kT / sum V, and Q_V_y = sum y / sum V;
  - Q_M(P) = sum m p_e / sum m, and Q_M(rho) = sum m rho / sum m;
  - Q_X = sum L_X p_e / sum L_X;
  - T_X = sum L_X T / sum L_X, T_M = sum m T / sum m, T_V = sum V T / sum V, T_e = sum N_e kT / sum N_e;
  - C_rho = sum m rho sum V / (sum m)^2, and C_P from sum V p_e^2, sum y and sum V.
- **Statistics come for free.** Every stacking statistic and error method needs only these per-halo profiles plus the inclusion weights, design strata and halo positions.
- **Size.** At z = 0 the x200 block is 2,007 MB uncompressed and 80 MB compressed for 4,588 haloes (measurements_owner.npz, owner200).
- **Add sum(y T).** Add sum(y T^2) too if second-order relativistic tSZ is wanted. They are not stored now, and they are what the y-weighted temperature needed for relativistic tSZ is built from.
- **Do not store estimator-level profiles in place of the sums.** Ratios cannot be re-binned or merged exactly.

## 6. Changes after verification

**First round.**
1. **T_X > T_e.** The cause is corrected from the phase cut, which gives 0.1-3.1%, to the L_X weighting. New diagnostic `T_X_decomposition` and new fig10.
2. **Q_M.** "Dense particles carry most of the mass" is replaced by the measured decomposition (new `mass_weighting_decomposition`, fig11). The gas-density formula is corrected to Q_M = <rho>_V C_rho, with Q_M/geometric = C_rho/f.
3. **T_M.** "Within 1% beyond 0.1 R200c" is corrected to "beyond 0.25 R200c".
4. **P = n_e k T_e.** "Exactly" is corrected. The scatter is attributed to an N^-1/2 per-particle y vs N_e kT mismatch, not float32 rounding (new `identity_noise`).
5. **Linearity.** Only mean y and mean DM are linear in the profile. The inclusion-weighted mean estimates the population mean; it does not conserve anything exactly.
6. **Stored sums and temperature advice.** The list of sums is corrected. The relativistic-tSZ advice is replaced by "store sum(y T)".
7. **Lee22.** The attribution of the log-mean is removed, including from the web description of the geometric-mean variant.
8. **Figures.** Five-bin colour, line-style and marker encoding in fig2, fig3 and fig8a. Fig4 and fig2 titles fixed. Sections 3-4 completed.

**Second round (text only; results.json and the web records are unchanged).**
1. **Q_X mass trend.** 1.31 belongs to 13.7-13.9, not 13.5-13.7 (1.36 +- 0.05).
2. **n_e kT_X below 1.** Values per band and bin: 0.94/0.92 in 14.9-15.1, 0.84-0.90 and 0.81-0.85 in the top bins, 0.995-1.001 in 14.7-14.9 at 0.5-2 R200c; in 13.5-14.5 it is > 1 everywhere beyond 0.25 R200c.
3. **Errors.** The unstratified/stratified factor is given per field (P 1.71, n_e 1.89, gas density 1.88, T_e 1.84 in 15.3-15.5). The jackknife excess is given in full (1.84 for P and 2.04 for n_e in 15.3-15.5; band medians 1.09-1.18 at 1.5-3 R200c in the main bins). The recommendation now scales the main-bin bootstrap beyond 1.5 R200c.
4. **Q_M mechanism.** "A few % of the gas mass" is replaced by 0.5-4.4% (cold/SF) and up to 18% (substructure-bound). The smooth hot gas alone still gives 40-69% of the pressure log-excess at 3-4 R200c.
5. **P = n_e k T_e.** "To 4e-4" is now "to 4e-4 for 99.84% of resolved halo-shells, maximum 1.0e-3" (extremes from `recheck_numbers.py`).
6. **n_e kT_X.** It is described as the temperature-weighting part of the X-ray route only, here and in the web label and description.
7. **B24 comparison.** Division by f is not enough: B24 use Q_X, the median, X-ray-phase gas and M500c/R500c bins.
8. **Same kinds of error elsewhere.**
   - Inner bands are resolved in only 3 and 6 bins, which is now stated.
   - The Q_V and f mass trends at 3-4 R200c are now stated.
   - The f deficit is 0.12-0.19, not 0.10-0.19.
   - The lognormal match is <= 0.1% only at 0.25-1 R200c (<= 0.8% elsewhere inside 1.5 R200c; 0.7% before the re-verification round below).
   - Y200 weighting is not inert beyond 1.5 R200c (0.980-1.051).
   - The substructure effects on T_X are attributed to the right bins.
   - The "clusters" label on the all-bin substructure shares is corrected.
   - Bootstrap/analytic and median-noise ranges now cover all bands.
   - N per bin is 127-895, not ~320.
   - Paired errors: Q_V 0.0064/0.0053, T_X 0.038, statistics per method.
   - The 34% <P>/(k<n_e>) excess now carries its +-0.10 error.
   - Section 5 no longer asks for smoothing-length histograms that are already stored.
   - "Hajek" replaces "Horvitz-Thompson" for the ratio mean. The shared `ref` web description is left unchanged because `ref` is identical in every group.
   - The fig2a band bin is now quoted as 14.3-14.5, matching the figure.

**Last round (text only; results.json and the web records are unchanged).**
1. **N per bin.** Below 10^14.5 the main bins have N = 317-327 (N_eff 309-319), not 308-327.
2. **f table.** The 0.1-0.25 R200c column is over 8 resolved bins (13.9-15.5), not 7.
3. **Q_M and Q_X mass trends.** The growth with mass over 13.7-15.1 is now stated for 0.25-1 R200c only (Q_X with one dip at 0.25-0.5 R200c, 1.166 to 1.126 at 13.9-14.1). Inside 0.25 R200c there is no consistent trend.
4. **n_e kT_X.** "Beyond R200c it is < 1 in clusters" is replaced by the per-bin values: < 1 at 1-2 R200c in 14.9-15.5, at 2-3 R200c only in 15.1-15.3, and > 1 beyond 0.5 R200c in 14.5-14.7. "Equal to T_X/T_e of the stacks" is replaced by the P_e-weighted mean of the per-halo T_X/T_e (new `check_ntx_weighting.py`), which differs from the stacked-temperature ratio of section 2.

**Re-verification round (text and the fig11 title; results.json and the web records are unchanged).**
1. **Q_M mechanism.** The dense minority (cold/SF and substructure-bound gas) carries most of sum(m rho) at every radius, but most of sum(m p_e) only beyond 2 R200c. Inside 1.5 R200c the smooth hot gas carries 71-99.9% of sum(m p_e), and at 0.25-1 R200c its own Q_M gives 0.84-1.35 times the pressure log-excess of all gas. The heading of 1.6(c), its mechanism bullet and the fig11 title now say so (`verify/v4_check.py`).
2. **Lognormal match.** Inside 0.25 R200c and at 1-1.5 R200c, mean/geomean matches exp(sd_ln^2/2) to <= 0.8% (largest 0.75%), not <= 0.7% (`verify/v4_lognormal.py`, own recomputation from the cache).
