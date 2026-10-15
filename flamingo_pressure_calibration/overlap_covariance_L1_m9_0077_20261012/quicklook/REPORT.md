# Step 0.3 quick look: covariance, GLS, AGN and membership on the radii234 haloes

FLAMINGO L1_m9, snapshot 0077 (z = 0); 4,375 centrals of the radii234 extraction (`radii234_contamination_L1_m9_0077_20261008/measurements.npz`), 36 shells to 4 R200c, four membership components. Default field: electron pressure with recently AGN-heated gas removed (`y_noagn`); all-gas pressure is the control. Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md`, step 0.3 (plus a scope extension for observable-driven weighting). Library: `../analysis/` (README there). No particle data were read; nothing outside `analysis/` and `quicklook/` was written.

## Main results

* **Regression gate passed.** With radii234's stored normalised inputs, objective (a) reproduces all 540 saved fits and 270 saved 128-draw bootstrap series bit for bit. Profiles rebuilt from the linear sums differ from the stored arrays at the ULP level (<= 1.2e-15), which moves the fits by at most 1.42e-06 (2-point Jacobian) and 1.30e-05 (analytic Jacobian), i.e. <= 7.58e-04 bootstrap sigma at an unchanged minimum. 14.5-14.7, all gas, to 2R: P0 15.813, xc 0.7623, beta 5.6557; to 4R: 17.952, 0.6430, 5.1113.
* **Correlation structure (Lee22 Fig. 1 analogue).** Above ~10^13.9 Msun the per-halo ln P_e correlation has two blocks: shells inside ~0.5-0.6 R200c are strongly mutually correlated, the shells outside likewise, and the two blocks are weakly anti-correlated (Fig. 2). The radius where shells decorrelate from the innermost resolved shell moves inwards with mass (pressure: 13.5-13.7 1.09, 13.7-13.9 0.90, 13.9-14.1 0.62, 14.1-14.3 0.60, 14.3-14.5 0.55, 14.5-14.7 0.53, 14.7-14.9 0.52, 14.9-15.1 0.49, 15.1-15.3 0.42, 15.3-15.5 0.44 R200c). Below 10^13.9 the resolved range starts at 0.27-2.6 R200c and one positive block dominates (weak anti-correlation between the innermost resolved shells and > R200c at 13.5-13.9). A single exponential in ln x (a, L) does not describe it (off-diagonal RMS 0.15-0.32 for bins >= 13.5 with N >= 100); the top two eigenvectors carry 68-80% of the standardised variance.
* **Stability.** Permutation tests of adjacent 0.2-dex bins (residualised, standardised ln F, common resolved shells, 500 permutations): pressure differs significantly (min-p < 0.05) in 3/13 pairs (12.7-12.9 | 12.9-13.1, 13.7-13.9 | 13.9-14.1, 14.5-14.7 | 14.7-14.9; 0.65 expected by chance), electrons in 8/13 and gas in 8/13: the density correlation structure drifts measurably with mass; the pressure structure changes clearly only across the 10^13.9 boundary (p = 0.002; the other two pairs are marginal). For fitting it hardly matters: refitting a bin with its neighbour's correlation matrix (N >= 3p cases) shifts the 2R pressure parameters by a median 0.16/0.12/0.16 sigma (P0/xc/beta; max 0.63 sigma). The top bins (N = 21, 6) are rank-deficient (N < number of shells) and their tests have little power.
* **GLS versus the current diagonal fit (pressure, no AGN, bins with N >= 100).** To 2 R200c the objectives agree within a few per cent: GLS - diagonal median +3.8/-4.5/-2.7% in P0/xc/beta (range -6.0 to +7.7% in P0), i.e. 2.1 paired-bootstrap sigma in the median (max 4.2), along the known P0-xc-beta degeneracy; the B12 form fits the 2R stack nearly within the statistical errors (chi2/nu at sigma_d = 0: 0.78-3.62; sigma_d = 0.0-0.8%). To 4 R200c the fits diverge: median +35.1/-40.7/-15.6% (P0 up to +62.5%). There chi2/nu at sigma_d = 0 is 4.4-18.2 and the explicit discrepancy is sigma_d = 7-14%: the white sigma_d term then dominates the covariance and turns (c) into an almost equal-weight fit in ln F, which follows the flat 2-4R outskirts and degrades 0.3-2R (Item 7). Beyond 2R the B12 shape, not the weighting, is the limitation.
* **Objective (b)** (diagonal error on the mean, no floor) is indistinguishable from (a): a diagonal fit is invariant to a common rescaling of the errors, and SD/mean exceeds the 5% floor on almost every shell. The current fit is therefore already the inverse-variance diagonal fit; what (a) ignores is the shell-shell correlation.
* **AGN (paired, same haloes and draws).** Removing recently heated gas changes fitted pressure parameters by <= 0.19% (supported bins, objectives a and c, 2R and 4R), Y(<4R) by <= 0.21% and Y(<1R) by <= 1.49% (largest at 12.5-12.7), but the unresolved core Y(<0.047R) by -27% to -0.07%, and the innermost shell (x < 0.025) by up to -44% (13.5-13.7). Every strictly resolved shell changes by <= 0.23%.
* **Membership controls, objective (c), bins with N >= 100.** To 2R, removing other-halo bound gas moves P0 by -0.42 to +0.00% and also removing own-satellite gas by -6.4 to -0.5%; to 4R by -29 to -6% and -26 to -7% (with compensating xc/beta). Host-bound-only profiles cannot be fitted by B12 to 2R (1/11 full-sample fits accepted; beta runs to its bound) because the bound gas truncates near R200c.
* **Scope extension (observable-driven weighting; toy importance plateau 0.3-1.5 R200c).** With B12, upweighting the important shells lets the fit extend to 3 R200c with a lower importance-weighted RMS than the current 2R fit (14.5-14.7: 6.0% -> 4.8%; 13.9-14.1: 5.5% -> 3.7%), but not to 4R without degrading the inner and 1-2R shells; GLS + sigma_d is the worst at 4R (11.7%, 8.7%). The two-component family B12 + A (x/2)^-s fitted to 4R keeps the RMS at <= 1.7% inside 2R and <= 4.3% at 2-4R: the model form, not the weighting, limits how far the fit can extend (Item 7).

## Verification

| Check | Result |
|---|---|
| Regression gate (item 1) | passed |
| Library tests (`analysis/tests/test_results.json`) | 18/18 passed |
| Inputs equal radii234 delivery-manifest hashes; earlier studies unchanged | passed |
| q4 fits reproduced bit for bit by the final library (after the scope extension) | passed |
| sigma_d solutions satisfy chi2/nu = 1 | max |chi2/nu - 1| = 1.03e-10 |

Gaussian mocks (known covariance from the 14.5-14.7 ln-scatter, 21 shells to 2R, N = 300, 200 mocks, 100 replicates each): GLS + shrinkage 68% coverage analytic 0.68, 0.69, 0.69, bootstrap 0.69, 0.69, 0.69 (P0, xc, beta); unshrunk bootstrap-covariance GLS analytic 0.61, 0.64, 0.65. RMS ln-error shrunk GLS 0.048, 0.040, 0.022 vs diagonal (a) 0.052, 0.046, 0.025.
Permutation null: false-positive rates at alpha = 0.05 frobenius_rms 0.070, delta_a 0.070, delta_lnL 0.055, max_angle_deg 0.045, combined_minp 0.070 (N 300/300) and frobenius_rms 0.060, delta_a 0.055, delta_lnL 0.045, max_angle_deg 0.055, combined_minp 0.080 (N 300/21); power against bin 9 vs bin 12 correlations 0.66 (min-p).
Shrinkage (bootstrap-variance intensity) reduces the correlation-matrix error by 65% at N = 10, 55% at N = 20, 45% at N = 40, 36% at N = 63, 24% at N = 120, 10% at N = 300, 5% at N = 900; the analytic Schaefer-Strimmer intensity overestimates lambda for the lognormal profiles and makes the error worse at N >= 300, so it is not used.

## 1. Regression gate

| Route | Max relative parameter deviation | Notes |
|---|---|---|
| stored normalised arrays, 2-point Jacobian (exact radii234 call) | 0 | 540/540 fits and 270/270 bootstrap series bit-identical |
| profiles rebuilt by `analysis.io`, 2-point Jacobian | 1.42e-06 | chi2 at new vs saved solution equal to 8.09e-12 |
| profiles rebuilt, analytic Jacobian | 1.30e-05 | <= 7.58e-04 bootstrap sigma |
| rebuilt-profile bootstraps (128 legacy rows) | 1.17e-04 | accepted-draw pattern identical in 270/270 series |

Fits compared: 15 bins x 4 selections x 3 fields x 2/3/4R (540 fits). The stored `measurements.npz` edges were computed on maas; edge 15 differs by one ULP from a local recomputation of `pilot_physics.EDGES`. `tables/regression_gate_fits.csv`, `tables/regression_gate_bootstraps.csv`.

## 2. Correlation matrices of per-halo ln F

Strict resolution mask (every halo resolved: inner edge >= max(0.04, 4 softening/R200c), >= 20 particles, >= 2 smoothing lengths), raw ln F (Lee22 convention; mass-residualised versions are saved). Fields: pressure without recent AGN, all-gas pressure, electrons, gas. Lee22's bins converted with h = 0.681 and weighted by inclusion weights (population/sample of each stratum):

| Lee22 bin | Physical log10 M200c | N (by stratum) | N_eff | First resolved x | Resolved shells to 1.35R |
|---|---|---|---|---|---|
| 10^13.2-10^13.4 h^-1 Msun | 13.367-13.567 | 313 (13.3-13.5: 196, 13.5-13.7: 117) | 301 | 0.442 | 7 |
| 10^13.8-10^14.0 h^-1 Msun | 13.967-14.167 | 299 (13.9-14.1: 184, 14.1-14.3: 115) | 281 | 0.145 | 15 |

At 10^13.4 h^-1 Msun only x >= 0.442 is resolved for the Lee22-bin haloes (0.5185 for the whole 13.3-13.5 stratum); the thin 0.98-1.0 shell is unresolved there. Lee22's full 0.04-1.34 R200c range is resolved only for >= 10^15.1 Msun (from 0.0555 R200c at 14.9-15.1).

![Lee22-style correlation, pressure](figures/fig2_lee22_style_correlation_pressure_noagn.png)

![All bins, pressure](figures/fig2_correlation_grid_pressure_noagn.png)

Grids for all-gas pressure, electrons and gas: `figures/fig2_correlation_grid_{pressure,electrons,gas}.png`; matrices in `products/correlation_matrices.npz`.

## 3. Stability across adjacent bins

| Pair | N_a/N_b | Shells | Frobenius RMS (null 95%) | Angle deg (null 95%) | p Frob | p a | p lnL | p angle | p min-p (pressure) | p min-p (electrons) |
|---|---|---|---|---|---|---|---|---|---|---|
| 12.7-12.9 vs 12.9-13.1 | 300/300 | 4 | 0.009 (0.012) | 3.9 (8.3) | 0.156 | 0.136 | 0.012 | 0.531 | 0.036 | 0.002 |
| 12.9-13.1 vs 13.1-13.3 | 300/300 | 8 | 0.022 (0.036) | 1.5 (4.7) | 0.263 | 1.000 | 0.283 | 0.900 | 0.465 | 0.509 |
| 13.1-13.3 vs 13.3-13.5 | 300/300 | 11 | 0.026 (0.055) | 5.0 (6.4) | 0.465 | 1.000 | 0.545 | 0.164 | 0.293 | 0.012 |
| 13.3-13.5 vs 13.5-13.7 | 300/300 | 15 | 0.079 (0.095) | 7.8 (8.5) | 0.104 | 1.000 | 0.238 | 0.090 | 0.246 | 0.002 |
| 13.5-13.7 vs 13.7-13.9 | 300/300 | 19 | 0.060 (0.099) | 8.5 (11.7) | 0.419 | 0.980 | 0.984 | 0.160 | 0.375 | 0.156 |
| 13.7-13.9 vs 13.9-14.1 | 300/300 | 20 | 0.201 (0.105) | 35.2 (17.0) | 0.002 | 0.166 | 0.004 | 0.002 | 0.002 | 0.002 |
| 13.9-14.1 vs 14.1-14.3 | 300/300 | 23 | 0.072 (0.104) | 12.1 (13.7) | 0.377 | 0.900 | 0.505 | 0.098 | 0.226 | 0.002 |
| 14.1-14.3 vs 14.3-14.5 | 300/300 | 24 | 0.087 (0.103) | 10.3 (14.5) | 0.162 | 0.196 | 0.100 | 0.216 | 0.212 | 0.002 |
| 14.3-14.5 vs 14.5-14.7 | 300/895 | 26 | 0.057 (0.082) | 3.5 (10.4) | 0.455 | 0.579 | 0.623 | 0.862 | 0.750 | 0.092 |
| 14.5-14.7 vs 14.7-14.9 | 895/326 | 27 | 0.081 (0.083) | 4.4 (10.2) | 0.066 | 0.030 | 0.014 | 0.713 | 0.036 | 0.002 |
| 14.7-14.9 vs 14.9-15.1 | 326/127 | 29 | 0.071 (0.128) | 10.4 (15.7) | 0.792 | 0.874 | 0.900 | 0.317 | 0.591 | 0.275 |
| 14.9-15.1 vs 15.1-15.3 | 127/21 | 30 | 0.266 (0.314) | 14.2 (43.8) | 0.170 | 0.295 | 0.212 | 0.760 | 0.317 | 0.178 |
| 15.1-15.3 vs 15.3-15.5 | 21/6 | 32 | 0.577 (0.667) | 55.9 (82.0) | 0.152 | 0.313 | 0.659 | 0.479 | 0.321 | 0.022 |

Per bin (pressure, residualised): (a, L) with 16-84% bootstrap ranges, off-diagonal RMS of the model, fraction of variance in the top two eigenvectors:

| Bin | N | Shells | a | L | Model RMS | Top-2 fraction | Decorrelation radius |
|---|---|---|---|---|---|---|---|
| 12.7-12.9 | 300 | 4 | 0.88 [0.80, 0.92] | 0.44 [0.29, 0.80] | 0.003 | 0.99 | - |
| 12.9-13.1 | 300 | 8 | -1.00 [-1.00, -1.00] | 7.23 [6.55, 8.01] | 0.017 | 0.98 | - |
| 13.1-13.3 | 300 | 11 | -1.00 [-1.00, -1.00] | 5.25 [4.80, 5.90] | 0.026 | 0.96 | - |
| 13.3-13.5 | 300 | 15 | -1.00 [-1.00, -1.00] | 3.82 [3.52, 4.10] | 0.066 | 0.89 | - |
| 13.5-13.7 | 300 | 19 | -0.51 [-0.77, -0.32] | 1.92 [1.59, 2.32] | 0.151 | 0.80 | 1.09 |
| 13.7-13.9 | 300 | 20 | -0.56 [-0.81, -0.39] | 1.95 [1.63, 2.37] | 0.174 | 0.79 | 0.899 |
| 13.9-14.1 | 300 | 23 | -0.33 [-0.40, -0.24] | 1.03 [0.93, 1.12] | 0.268 | 0.72 | 0.617 |
| 14.1-14.3 | 300 | 24 | -0.35 [-0.42, -0.28] | 0.97 [0.88, 1.07] | 0.296 | 0.72 | 0.595 |
| 14.3-14.5 | 300 | 26 | -0.24 [-0.29, -0.18] | 0.84 [0.76, 0.92] | 0.275 | 0.68 | 0.547 |
| 14.5-14.7 | 895 | 27 | -0.28 [-0.32, -0.24] | 0.92 [0.87, 0.97] | 0.313 | 0.71 | 0.529 |
| 14.7-14.9 | 326 | 29 | -0.12 [-0.17, -0.08] | 0.77 [0.70, 0.82] | 0.311 | 0.68 | 0.516 |
| 14.9-15.1 | 127 | 30 | -0.12 [-0.21, -0.05] | 0.78 [0.68, 0.89] | 0.324 | 0.69 | 0.49 |
| 15.1-15.3 | 21 | 32 | 0.11 [-0.03, 0.21] | 0.54 [0.43, 0.72] | 0.394 | 0.72 | 0.424 |
| 15.3-15.5 | 6 | 32 | -0.25 [-0.56, -0.04] | 0.76 [0.39, 1.20] | 0.573 | 0.83 | 0.441 |

The (a, L) model is inadequate wherever the inner block is resolved (RMS 0.15-0.39; 0.57 for N = 6) and degenerate where only a few outer shells are resolved (a at its -1 bound for 12.9-13.5), so Delta a and Delta ln L are weak statistics; the Frobenius distance and the principal angles carry the test. Neighbour-correlation refits (`tables/neighbour_covariance_shift.csv`, N >= 3p cases): pressure 2R median |Delta ln theta| 0.8%, 0.6%, 0.4%, 4R 1.4%, 1.6%, 0.6%; electrons 2R 0.4%, 0.1%, 0.1%. Top bins: 15.1-15.3 (N = 21) and 15.3-15.5 (N = 6) have fewer haloes than shells, so their sample correlations are rank-deficient (rank <= 20 and 5); permutation p-values remain valid but have little power, and their GLS fits use the pooled correlation (lambda = 1).

![Stability](figures/fig3_stability.png)

## 4. Fits at seven cut-offs: diagonal versus GLS

Cut-offs are the nearest existing edges 1, 1.3508, 1.5845, 1.8587, 2, 3, 4 R200c (stand-ins for 1/1.34/1.5/1.75/2/3/4; the coming 69-shell extraction has the exact edges). Supported bins 13.3-15.5. Objectives: (a) diagonal sigma = max(SD/mean, 5%) (current); (b) diagonal bootstrap error on ln mean; (c) GLS with the 1000-draw bootstrap Cov[ln mean], shrunk towards the leave-one-out pooled correlation of the other bins with N >= 3p (lambda from 50 resamples; lambda = 1 when N < 3p), plus sigma_d solved for chi2/nu = 1. 500 paired halo-bootstrap replicates (the first 128 identical to radii234's), each re-estimating its own covariance, lambda and sigma_d. Pressure without recent AGN:

| Bin | Cut | Obj | P0 [16, 84] | xc | beta | sigma_d | lambda | chi2/nu (sd = 0) | N_eff P0/xc/beta |
|---|---|---|---|---|---|---|---|---|---|
| 13.5-13.7 | 1 | a | 2.66 [2.42, 2.91] | 1.532 [1.334, 1.779] | 5.92 [5.37, 6.60] | - | - | - | 6.5/6.7/6.8 |
| 13.5-13.7 | 1 | b | 2.66 [2.42, 2.90] | 1.532 [1.338, 1.787] | 5.92 [5.36, 6.60] | - | - | - | 6.6/6.8/6.8 |
| 13.5-13.7 | 1 | c | 2.77 [2.52, 3.02] | 1.437 [1.264, 1.669] | 5.67 [5.18, 6.29] | 0.00456 | 0.00758 | 1.94 | 7.2/7.3/7.4 |
| 13.5-13.7 | 2 | a | 2.58 [2.39, 2.74] | 1.614 [1.488, 1.776] | 6.14 [5.83, 6.53] | - | - | - | 9.7/9.8/10.1 |
| 13.5-13.7 | 2 | b | 2.58 [2.39, 2.74] | 1.612 [1.488, 1.777] | 6.13 [5.83, 6.53] | - | - | - | 9.6/9.8/10.1 |
| 13.5-13.7 | 2 | c | 2.71 [2.52, 2.86] | 1.511 [1.404, 1.655] | 5.89 [5.61, 6.27] | 0.00562 | 0.021 | 2.56 | 7.2/8.0/7.9 |
| 13.5-13.7 | 3 | a | 2.87 [2.70, 3.06] | 1.365 [1.270, 1.473] | 5.50 [5.25, 5.75] | - | - | - | 10.5/11.0/11.2 |
| 13.5-13.7 | 3 | b | 2.88 [2.70, 3.06] | 1.358 [1.270, 1.468] | 5.48 [5.25, 5.75] | - | - | - | 10.5/11.0/11.3 |
| 13.5-13.7 | 3 | c | 3.22 [3.01, 3.50] | 1.188 [1.081, 1.285] | 5.09 [4.82, 5.33] | 0.0323 | 0.0366 | 3.8 | 11.1/12.2/12.7 |
| 13.5-13.7 | 4 | a | 3.21 [3.05, 3.64] | 1.169 [0.995, 1.248] | 5.00 [4.55, 5.18] | - | - | - | 10.6/11.1/11.5 |
| 13.5-13.7 | 4 | b | 3.25 [3.07, 3.66] | 1.153 [0.991, 1.238] | 4.96 [4.55, 5.17] | - | - | - | 10.6/11.1/11.6 |
| 13.5-13.7 | 4 | c | 4.35 [4.04, 6.05] | 0.857 [0.610, 0.922] | 4.32 [3.70, 4.48] | 0.0775 | 0.0804 | 4.35 | 11.8/13.3/14.1 |
| 13.9-14.1 | 1 | a | 6.86 [6.39, 7.34] | 1.168 [1.070, 1.289] | 6.21 [5.87, 6.64] | - | - | - | 8.8/8.4/8.4 |
| 13.9-14.1 | 1 | b | 6.84 [6.39, 7.34] | 1.174 [1.069, 1.289] | 6.23 [5.86, 6.64] | - | - | - | 8.6/8.3/8.3 |
| 13.9-14.1 | 1 | c | 7.28 [6.79, 7.71] | 1.079 [1.002, 1.181] | 5.91 [5.64, 6.26] | 0.00302 | 0.153 | 1.87 | 7.1/8.6/8.8 |
| 13.9-14.1 | 2 | a | 6.93 [6.51, 7.35] | 1.155 [1.076, 1.240] | 6.18 [5.94, 6.46] | - | - | - | 11.5/10.8/10.8 |
| 13.9-14.1 | 2 | b | 6.92 [6.52, 7.35] | 1.158 [1.076, 1.239] | 6.18 [5.94, 6.46] | - | - | - | 11.3/10.8/10.7 |
| 13.9-14.1 | 2 | c | 7.49 [7.01, 7.84] | 1.049 [0.998, 1.125] | 5.83 [5.66, 6.09] | 0.00704 | 0.211 | 2.57 | 6.6/8.3/8.9 |
| 13.9-14.1 | 3 | a | 7.57 [7.17, 8.00] | 1.016 [0.960, 1.074] | 5.68 [5.50, 5.87] | - | - | - | 12.7/11.8/11.7 |
| 13.9-14.1 | 3 | b | 7.56 [7.16, 7.99] | 1.019 [0.959, 1.075] | 5.69 [5.49, 5.87] | - | - | - | 12.7/11.7/11.6 |
| 13.9-14.1 | 3 | c | 8.59 [8.10, 9.01] | 0.855 [0.819, 0.904] | 5.13 [5.00, 5.31] | 0.0265 | 0.253 | 4.46 | 11.2/12.1/12.0 |
| 13.9-14.1 | 4 | a | 8.82 [8.32, 9.29] | 0.827 [0.787, 0.874] | 5.02 [4.88, 5.19] | - | - | - | 13.3/12.5/12.4 |
| 13.9-14.1 | 4 | b | 8.79 [8.31, 9.30] | 0.832 [0.785, 0.873] | 5.04 [4.87, 5.19] | - | - | - | 13.4/12.4/12.3 |
| 13.9-14.1 | 4 | c | 12.91 [11.90, 13.82] | 0.542 [0.511, 0.585] | 4.11 [4.00, 4.26] | 0.122 | 0.286 | 7.6 | 15.6/17.5/17.4 |
| 14.5-14.7 | 1 | a | 15.95 [15.50, 16.47] | 0.751 [0.723, 0.779] | 5.60 [5.47, 5.72] | - | - | - | 12.1/11.2/10.9 |
| 14.5-14.7 | 1 | b | 15.95 [15.50, 16.47] | 0.751 [0.723, 0.779] | 5.60 [5.47, 5.72] | - | - | - | 12.0/11.0/10.8 |
| 14.5-14.7 | 1 | c | 15.62 [15.26, 16.15] | 0.777 [0.745, 0.797] | 5.72 [5.57, 5.82] | 8.79e-04 | 0.028 | 1.41 | 11.1/11.6/11.6 |
| 14.5-14.7 | 2 | a | 15.81 [15.37, 16.27] | 0.762 [0.743, 0.783] | 5.66 [5.57, 5.75] | - | - | - | 14.7/13.4/13.6 |
| 14.5-14.7 | 2 | b | 15.82 [15.38, 16.26] | 0.762 [0.743, 0.783] | 5.65 [5.57, 5.74] | - | - | - | 14.6/13.2/13.2 |
| 14.5-14.7 | 2 | c | 16.03 [15.69, 16.49] | 0.749 [0.730, 0.766] | 5.60 [5.52, 5.67] | 0.00761 | 0.0331 | 3.62 | 11.1/14.6/14.9 |
| 14.5-14.7 | 3 | a | 16.69 [16.26, 17.16] | 0.707 [0.688, 0.724] | 5.40 [5.32, 5.47] | - | - | - | 15.9/14.2/14.3 |
| 14.5-14.7 | 3 | b | 16.72 [16.25, 17.15] | 0.706 [0.688, 0.724] | 5.40 [5.33, 5.47] | - | - | - | 15.8/14.0/14.1 |
| 14.5-14.7 | 3 | c | 18.73 [18.23, 19.20] | 0.606 [0.592, 0.620] | 4.97 [4.91, 5.03] | 0.0591 | 0.0432 | 11 | 16.8/18.7/18.5 |
| 14.5-14.7 | 4 | a | 17.95 [17.53, 18.61] | 0.643 [0.620, 0.656] | 5.11 [5.01, 5.17] | - | - | - | 16.2/14.6/14.7 |
| 14.5-14.7 | 4 | b | 17.94 [17.53, 18.64] | 0.644 [0.620, 0.656] | 5.12 [5.01, 5.17] | - | - | - | 16.1/14.4/14.5 |
| 14.5-14.7 | 4 | c | 25.50 [24.70, 26.32] | 0.422 [0.409, 0.434] | 4.17 [4.11, 4.23] | 0.141 | 0.0757 | 18.2 | 18.5/20.7/20.8 |
| 14.9-15.1 | 1 | a | 16.85 [15.20, 18.57] | 0.810 [0.724, 0.917] | 6.16 [5.72, 6.66] | - | - | - | 14.3/12.5/11.9 |
| 14.9-15.1 | 1 | b | 16.86 [15.20, 18.56] | 0.809 [0.724, 0.917] | 6.15 [5.72, 6.65] | - | - | - | 14.5/12.7/12.1 |
| 14.9-15.1 | 1 | c | 15.77 [14.54, 16.91] | 0.870 [0.793, 0.964] | 6.42 [6.04, 6.85] | 0.00196 | 0.235 | 1.54 | 10.5/10.6/10.9 |
| 14.9-15.1 | 2 | a | 17.18 [15.70, 18.70] | 0.786 [0.731, 0.855] | 6.04 [5.81, 6.34] | - | - | - | 18.1/15.7/15.5 |
| 14.9-15.1 | 2 | b | 17.18 [15.70, 18.69] | 0.787 [0.731, 0.856] | 6.05 [5.81, 6.34] | - | - | - | 18.4/16.3/16.0 |
| 14.9-15.1 | 2 | c | 16.85 [15.81, 18.04] | 0.784 [0.738, 0.823] | 6.00 [5.79, 6.15] | 0.00331 | 0.23 | 1.69 | 13.5/14.5/15.2 |
| 14.9-15.1 | 3 | a | 18.17 [16.78, 19.72] | 0.725 [0.678, 0.773] | 5.75 [5.56, 5.93] | - | - | - | 19.7/16.8/16.8 |
| 14.9-15.1 | 3 | b | 18.17 [16.77, 19.70] | 0.725 [0.677, 0.773] | 5.75 [5.55, 5.93] | - | - | - | 20.0/17.4/17.4 |
| 14.9-15.1 | 3 | c | 18.64 [17.51, 20.34] | 0.695 [0.637, 0.723] | 5.60 [5.35, 5.71] | 0.0329 | 0.25 | 3.47 | 15.6/20.8/20.8 |
| 14.9-15.1 | 4 | a | 20.73 [19.11, 22.40] | 0.608 [0.574, 0.647] | 5.18 [5.05, 5.33] | - | - | - | 20.7/18.3/18.1 |
| 14.9-15.1 | 4 | b | 20.73 [19.12, 22.37] | 0.607 [0.575, 0.646] | 5.18 [5.05, 5.32] | - | - | - | 21.0/18.8/18.7 |
| 14.9-15.1 | 4 | c | 22.97 [21.47, 24.79] | 0.516 [0.489, 0.543] | 4.74 [4.64, 4.85] | 0.084 | 0.321 | 5.42 | 18.7/21.5/21.0 |

All bins, fields and cut-offs: `tables/fits_cutoffs.csv` (with cross-fit parameters, analytic errors, Hartlap factors), shifts relative to the 2R fit `tables/fits_cutoff_shifts_vs_2R.csv`, GLS minus diagonal `tables/fits_gls_minus_diagonal.csv`.

![Parameters versus cut-off](figures/fig4_cutoff_parameters_pressure_noagn.png)

![GLS minus diagonal](figures/fig4_gls_minus_diagonal_pressure.png)

![Influence rows](figures/fig4_influence_14p5-14p7_pressure_noagn.png)

Reading the influence rows: (a) is driven by 0.4-0.8 R200c (the peak of 1/CV^2 and the curvature of B12) and by the innermost resolved shells; GLS spreads the influence and puts large weight on the end shells (the least redundant ones under strong correlation). Effective number of shells (participation ratio) at 2R: (a) 7, 10, 10, 11, 11, 13, 14, 15, 16, 16, 9; (c) 5, 8, 9, 8, 11, 10, 13, 12, 14, 16, 14 (bins 13.3-15.5, mean over parameters).

Cross-fitting (covariance from one random half, mean from the other, 10 splits, delta-method covariance) moves the 2R GLS parameters by -5.0 to +3.0% and the 4R ones by -19 to +21%: no systematic covariance-mean feedback at 2R; at 4R the spread reflects the sigma_d-dominated, nearly flat objective. Analytic GLS errors (incl. sigma_d) / bootstrap SD: 0.88-1.16 at 2R, 1.1-4.6 at 4R (sigma_d is a model-error term that halo resampling does not reproduce). Hartlap factors (N - p - 2)/(N - 1) are 0.91-0.98 for N = 300 (reported only; they cannot move the point estimates at fixed sigma_d).

## 5. Paired AGN table (y_noagn versus y)

| Bin | N | dY(<0.047R) % | dY(<1R) % | dY(<2R) % | dY(<4R) % | max |dP| resolved % | innermost shell % |
|---|---|---|---|---|---|---|---|
| 12.5-12.7 | 300 | -13.82 | -1.488 | -0.490 | -0.206 | - | -16.7 |
| 12.7-12.9 | 300 | -8.51 | -0.453 | -0.170 | -0.062 | 0 | -10.0 |
| 12.9-13.1 | 300 | -10.00 | -0.236 | -0.099 | -0.073 | 0.139 | -12.1 |
| 13.1-13.3 | 300 | -27.41 | -0.260 | -0.145 | -0.095 | 0.231 | -30.6 |
| 13.3-13.5 | 300 | -14.60 | -0.058 | -0.040 | -0.028 | 0.0512 | -22.4 |
| 13.5-13.7 | 300 | -25.29 | -0.046 | -0.027 | -0.027 | 0.0624 | -44.3 |
| 13.7-13.9 | 300 | -12.05 | -0.024 | -0.015 | -0.013 | 0.0159 | -38.5 |
| 13.9-14.1 | 300 | -4.10 | -0.012 | -0.008 | -0.008 | 0.0188 | -21.8 |
| 14.1-14.3 | 300 | -2.54 | -0.009 | -0.007 | -0.007 | 0.0154 | -17.1 |
| 14.3-14.5 | 300 | -1.33 | -0.006 | -0.005 | -0.006 | 0.0133 | -9.0 |
| 14.5-14.7 | 895 | -0.60 | -0.003 | -0.002 | -0.003 | 0.00771 | -4.1 |
| 14.7-14.9 | 326 | -0.28 | -0.002 | -0.002 | -0.002 | 0.00844 | -1.6 |
| 14.9-15.1 | 127 | -0.10 | -0.001 | -0.001 | -0.001 | 0.00999 | -0.6 |
| 15.1-15.3 | 21 | -0.07 | -0.000 | -0.000 | -0.001 | 0.00846 | -0.4 |
| 15.3-15.5 | 6 | -0.07 | -0.001 | -0.001 | -0.001 | 0.0416 | -0.4 |

Parameter changes (no-AGN minus all gas, %, full sample with 16-84% of paired replicates), objective (a) and (c), 2R and 4R:

| Bin | Cut | Obj | dln P0 % | dln xc % | dln beta % |
|---|---|---|---|---|---|
| 13.3-13.5 | 2 | a | +0.163 [+0.019, +0.300] | -0.186 [-0.343, -0.019] | -0.092 [-0.172, -0.005] |
| 13.3-13.5 | 2 | c | +0.072 [-0.019, +0.223] | -0.109 [-0.266, +0.022] | -0.056 [-0.140, +0.022] |
| 13.3-13.5 | 4 | a | +0.122 [+0.021, +0.233] | -0.106 [-0.202, -0.014] | -0.035 [-0.068, -0.001] |
| 13.3-13.5 | 4 | c | +0.103 [+0.032, +0.190] | -0.075 [-0.130, -0.027] | -0.017 [-0.026, -0.005] |
| 13.5-13.7 | 2 | a | +0.017 [-0.002, +0.035] | -0.022 [-0.049, +0.004] | -0.013 [-0.030, +0.003] |
| 13.5-13.7 | 2 | c | +0.024 [+0.001, +0.031] | -0.033 [-0.041, -0.001] | -0.020 [-0.024, -0.001] |
| 13.5-13.7 | 4 | a | -0.011 [-0.032, +0.004] | +0.020 [-0.000, +0.045] | +0.014 [+0.003, +0.027] |
| 13.5-13.7 | 4 | c | -0.043 [-0.085, +0.002] | +0.053 [+0.005, +0.090] | +0.029 [+0.006, +0.045] |
| 13.7-13.9 | 2 | a | +0.016 [-0.006, +0.034] | -0.027 [-0.052, +0.003] | -0.017 [-0.033, +0.001] |
| 13.7-13.9 | 2 | c | +0.022 [-0.013, +0.044] | -0.020 [-0.047, +0.017] | -0.008 [-0.025, +0.012] |
| 13.7-13.9 | 4 | a | +0.006 [-0.007, +0.016] | -0.010 [-0.024, +0.005] | -0.005 [-0.013, +0.003] |
| 13.7-13.9 | 4 | c | -0.012 [-0.019, +0.000] | +0.009 [-0.002, +0.014] | +0.004 [-0.000, +0.006] |
| 13.9-14.1 | 2 | a | +0.007 [+0.003, +0.012] | -0.009 [-0.015, -0.003] | -0.005 [-0.009, -0.002] |
| 13.9-14.1 | 2 | c | +0.003 [+0.001, +0.006] | -0.004 [-0.007, -0.001] | -0.002 [-0.004, -0.001] |
| 13.9-14.1 | 4 | a | +0.000 [-0.003, +0.004] | +0.002 [-0.003, +0.006] | +0.002 [-0.000, +0.004] |
| 13.9-14.1 | 4 | c | -0.001 [-0.006, +0.004] | +0.004 [-0.001, +0.008] | +0.003 [+0.001, +0.005] |
| 14.1-14.3 | 2 | a | -0.001 [-0.003, +0.002] | +0.002 [-0.002, +0.005] | +0.002 [-0.001, +0.004] |
| 14.1-14.3 | 2 | c | -0.005 [-0.013, +0.002] | +0.014 [-0.003, +0.023] | +0.010 [-0.001, +0.016] |
| 14.1-14.3 | 4 | a | -0.002 [-0.004, +0.001] | +0.003 [+0.000, +0.006] | +0.003 [+0.001, +0.004] |
| 14.1-14.3 | 4 | c | -0.007 [-0.012, -0.003] | +0.009 [+0.004, +0.014] | +0.005 [+0.003, +0.007] |
| 14.3-14.5 | 2 | a | -0.003 [-0.005, -0.001] | +0.003 [-0.000, +0.006] | +0.002 [-0.000, +0.004] |
| 14.3-14.5 | 2 | c | +0.004 [-0.003, +0.007] | -0.009 [-0.009, +0.004] | -0.006 [-0.005, +0.003] |
| 14.3-14.5 | 4 | a | -0.005 [-0.007, -0.004] | +0.007 [+0.005, +0.009] | +0.004 [+0.003, +0.005] |
| 14.3-14.5 | 4 | c | -0.002 [-0.004, +0.001] | +0.003 [+0.000, +0.006] | +0.002 [+0.001, +0.003] |
| 14.5-14.7 | 2 | a | -0.000 [-0.001, -0.000] | +0.001 [+0.000, +0.001] | +0.001 [+0.000, +0.001] |
| 14.5-14.7 | 2 | c | -0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.001] | +0.000 [-0.000, +0.001] |
| 14.5-14.7 | 4 | a | -0.002 [-0.003, -0.002] | +0.003 [+0.003, +0.004] | +0.002 [+0.002, +0.002] |
| 14.5-14.7 | 4 | c | -0.003 [-0.004, -0.003] | +0.004 [+0.004, +0.005] | +0.002 [+0.002, +0.003] |
| 14.7-14.9 | 2 | a | -0.000 [-0.001, +0.000] | -0.000 [-0.001, +0.001] | +0.000 [-0.000, +0.001] |
| 14.7-14.9 | 2 | c | -0.001 [-0.002, +0.001] | -0.001 [-0.002, +0.002] | -0.001 [-0.001, +0.002] |
| 14.7-14.9 | 4 | a | -0.001 [-0.002, -0.001] | +0.002 [+0.002, +0.003] | +0.001 [+0.001, +0.002] |
| 14.7-14.9 | 4 | c | -0.003 [-0.004, -0.002] | +0.004 [+0.003, +0.005] | +0.002 [+0.002, +0.003] |
| 14.9-15.1 | 2 | a | -0.000 [-0.000, -0.000] | +0.001 [+0.000, +0.001] | +0.000 [+0.000, +0.001] |
| 14.9-15.1 | 2 | c | +0.001 [-0.000, +0.001] | -0.002 [-0.002, +0.000] | -0.001 [-0.001, +0.000] |
| 14.9-15.1 | 4 | a | -0.001 [-0.001, -0.001] | +0.002 [+0.001, +0.002] | +0.001 [+0.001, +0.001] |
| 14.9-15.1 | 4 | c | -0.001 [-0.001, -0.000] | +0.001 [+0.001, +0.002] | +0.001 [+0.001, +0.001] |
| 15.1-15.3 | 2 | a | -0.000 [-0.000, -0.000] | +0.001 [+0.000, +0.001] | +0.000 [+0.000, +0.001] |
| 15.1-15.3 | 2 | c | +0.004 [-0.000, +0.002] | -0.001 [-0.001, +0.002] | +0.000 [-0.000, +0.001] |
| 15.1-15.3 | 4 | a | -0.001 [-0.002, -0.001] | +0.002 [+0.001, +0.003] | +0.001 [+0.001, +0.002] |
| 15.1-15.3 | 4 | c | +0.001 [-0.001, +0.001] | +0.002 [-0.000, +0.003] | +0.002 [+0.000, +0.002] |
| 15.3-15.5 | 2 | a | -0.013 [-0.023, +0.000] | +0.009 [-0.000, +0.015] | +0.003 [-0.000, +0.005] |
| 15.3-15.5 | 2 | c | -0.052 [-0.058, -0.001] | +0.041 [-0.000, +0.042] | +0.018 [-0.001, +0.017] |
| 15.3-15.5 | 4 | a | -0.012 [-0.021, -0.000] | +0.007 [+0.000, +0.011] | +0.002 [+0.000, +0.003] |
| 15.3-15.5 | 4 | c | -0.015 [-0.042, +0.004] | +0.005 [-0.004, +0.020] | -0.000 [-0.003, +0.004] |

![AGN](figures/fig5_agn_paired.png)

## 6. Membership controls (objective c, paired)

| Bin | Field | Selection | Cut | dln par1 % | dln par2 % | dln beta % | dY(<cut) % | accepted |
|---|---|---|---|---|---|---|---|---|
| 13.3-13.5 | pressure_noagn | no_other | 2 | -0.4 [-0.6, -0.2] | +0.6 [+0.3, +0.9] | +0.4 [+0.2, +0.5] | -0.01 | 1.00 |
| 13.5-13.7 | pressure_noagn | no_other | 2 | -0.1 [-0.1, -0.1] | +0.2 [+0.1, +0.2] | +0.1 [+0.1, +0.2] | -0.01 | 1.00 |
| 13.7-13.9 | pressure_noagn | no_other | 2 | -0.1 [-0.2, -0.0] | +0.2 [+0.1, +0.3] | +0.1 [+0.0, +0.2] | -0.01 | 1.00 |
| 13.9-14.1 | pressure_noagn | no_other | 2 | -0.0 [-0.1, -0.0] | +0.1 [+0.0, +0.1] | +0.0 [+0.0, +0.1] | -0.00 | 1.00 |
| 14.1-14.3 | pressure_noagn | no_other | 2 | -0.2 [-0.2, -0.1] | +0.3 [+0.1, +0.3] | +0.2 [+0.1, +0.2] | -0.01 | 1.00 |
| 14.3-14.5 | pressure_noagn | no_other | 2 | -0.1 [-0.1, -0.0] | +0.2 [+0.1, +0.2] | +0.1 [+0.1, +0.1] | -0.01 | 1.00 |
| 14.5-14.7 | pressure_noagn | no_other | 2 | -0.1 [-0.2, -0.1] | +0.2 [+0.2, +0.3] | +0.2 [+0.1, +0.2] | -0.01 | 1.00 |
| 14.7-14.9 | pressure_noagn | no_other | 2 | +0.0 [-0.1, +0.0] | -0.0 [-0.0, +0.2] | -0.0 [-0.0, +0.1] | -0.01 | 1.00 |
| 14.9-15.1 | pressure_noagn | no_other | 2 | -0.1 [-0.3, -0.1] | +0.1 [+0.2, +0.4] | +0.1 [+0.1, +0.3] | -0.01 | 1.00 |
| 15.1-15.3 | pressure_noagn | no_other | 2 | +0.3 [-0.0, +0.3] | -0.2 [-0.3, +0.1] | -0.1 [-0.1, +0.0] | -0.00 | 1.00 |
| 15.3-15.5 | pressure_noagn | no_other | 2 | +3.9 [+0.6, +3.7] | -2.4 [-2.2, -0.3] | -0.9 [-0.8, -0.0] | -0.00 | 1.00 |
| 13.3-13.5 | pressure_noagn | no_satellites_or_other | 2 | -6.4 [-8.7, -4.0] | +9.2 [+5.6, +12.1] | +5.8 [+3.5, +7.6] | -0.44 | 1.00 |
| 13.5-13.7 | pressure_noagn | no_satellites_or_other | 2 | -1.7 [-2.4, -0.9] | +3.3 [+2.0, +4.5] | +2.5 [+1.6, +3.3] | -0.48 | 1.00 |
| 13.7-13.9 | pressure_noagn | no_satellites_or_other | 2 | -4.8 [-6.0, -2.9] | +7.8 [+4.8, +9.4] | +5.3 [+3.3, +6.4] | -0.61 | 1.00 |
| 13.9-14.1 | pressure_noagn | no_satellites_or_other | 2 | -3.2 [-4.4, -1.9] | +6.3 [+4.0, +8.3] | +4.6 [+2.9, +6.0] | -1.12 | 1.00 |
| 14.1-14.3 | pressure_noagn | no_satellites_or_other | 2 | -2.4 [-3.8, -1.6] | +6.0 [+4.2, +7.8] | +4.7 [+3.3, +5.8] | -1.53 | 1.00 |
| 14.3-14.5 | pressure_noagn | no_satellites_or_other | 2 | -0.8 [-2.7, -0.3] | +2.7 [+2.2, +5.2] | +2.2 [+1.9, +3.8] | -1.17 | 1.00 |
| 14.5-14.7 | pressure_noagn | no_satellites_or_other | 2 | -1.4 [-1.8, -1.0] | +3.1 [+2.5, +3.8] | +2.3 [+2.0, +2.8] | -1.13 | 1.00 |
| 14.7-14.9 | pressure_noagn | no_satellites_or_other | 2 | -0.5 [-1.4, -0.2] | +1.7 [+1.3, +3.2] | +1.6 [+1.3, +2.6] | -1.59 | 1.00 |
| 14.9-15.1 | pressure_noagn | no_satellites_or_other | 2 | -1.2 [-3.7, -0.6] | +2.8 [+1.7, +5.3] | +2.3 [+1.4, +3.7] | -1.19 | 1.00 |
| 15.1-15.3 | pressure_noagn | no_satellites_or_other | 2 | -3.6 [-4.5, -0.7] | +3.7 [+0.5, +5.3] | +2.2 [+0.4, +3.4] | -0.58 | 1.00 |
| 15.3-15.5 | pressure_noagn | no_satellites_or_other | 2 | -7.8 [-15.9, +4.9] | -7.8 [-13.9, +5.1] | -8.0 [-9.7, +1.0] | -4.85 | 1.00 |
| 13.3-13.5 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -38.15 | 0.00 |
| 13.5-13.7 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -31.56 | 0.00 |
| 13.7-13.9 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -26.55 | 0.00 |
| 13.9-14.1 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -22.88 | 0.00 |
| 14.1-14.3 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -21.20 | 0.00 |
| 14.3-14.5 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -19.30 | 0.00 |
| 14.5-14.7 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -18.50 | 0.00 |
| 14.7-14.9 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -18.53 | 0.19 |
| 14.9-15.1 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -19.13 | 0.27 |
| 15.1-15.3 | pressure_noagn | host_bound | 2 | not accepted | not accepted | not accepted | -16.42 | 0.14 |
| 15.3-15.5 | pressure_noagn | host_bound | 2 | -45.8 [-75.2, -19.2] | +86.2 [+55.3, +122.2] | +71.4 [+47.2, +94.8] | -21.66 | 0.93 |
| 13.3-13.5 | pressure_noagn | no_other | 4 | -29.1 [-56.4, -16.7] | +22.2 [+12.5, +41.9] | +6.5 [+3.6, +11.5] | -3.05 | 1.00 |
| 13.5-13.7 | pressure_noagn | no_other | 4 | -6.2 [-14.1, -4.1] | +6.9 [+4.6, +14.2] | +3.4 [+2.3, +6.2] | -1.46 | 1.00 |
| 13.7-13.9 | pressure_noagn | no_other | 4 | -11.2 [-19.6, -6.1] | +9.6 [+5.2, +17.1] | +3.6 [+1.9, +6.6] | -2.20 | 1.00 |
| 13.9-14.1 | pressure_noagn | no_other | 4 | -16.9 [-20.0, -13.1] | +17.7 [+13.9, +21.1] | +8.0 [+6.3, +9.4] | -2.00 | 1.00 |
| 14.1-14.3 | pressure_noagn | no_other | 4 | -13.9 [-16.6, -10.6] | +15.2 [+11.7, +17.9] | +7.1 [+5.4, +8.3] | -1.86 | 1.00 |
| 14.3-14.5 | pressure_noagn | no_other | 4 | -11.3 [-13.4, -9.2] | +13.3 [+10.9, +15.8] | +6.4 [+5.2, +7.5] | -1.67 | 1.00 |
| 14.5-14.7 | pressure_noagn | no_other | 4 | -13.0 [-14.2, -11.5] | +15.0 [+13.3, +16.4] | +7.0 [+6.2, +7.7] | -1.71 | 1.00 |
| 14.7-14.9 | pressure_noagn | no_other | 4 | -10.4 [-11.4, -8.7] | +12.7 [+11.0, +14.4] | +6.2 [+5.5, +7.2] | -1.27 | 1.00 |
| 14.9-15.1 | pressure_noagn | no_other | 4 | -9.2 [-10.2, -7.3] | +12.8 [+10.2, +14.0] | +6.8 [+5.5, +7.6] | -1.13 | 1.00 |
| 15.1-15.3 | pressure_noagn | no_other | 4 | -3.4 [-6.4, +0.9] | +0.6 [-3.0, +6.5] | -0.5 [-2.4, +3.5] | -1.32 | 1.00 |
| 15.3-15.5 | pressure_noagn | no_other | 4 | -3.1 [-9.6, +1.5] | +0.6 [-3.3, +6.1] | -0.4 [-2.5, +2.3] | -0.77 | 1.00 |
| 13.3-13.5 | pressure_noagn | no_satellites_or_other | 4 | -26.4 [-54.2, -14.8] | +22.0 [+13.3, +41.9] | +7.5 [+4.9, +12.4] | -4.46 | 1.00 |
| 13.5-13.7 | pressure_noagn | no_satellites_or_other | 4 | -7.2 [-16.8, -3.5] | +9.2 [+5.3, +17.7] | +5.2 [+3.3, +8.2] | -2.76 | 1.00 |
| 13.7-13.9 | pressure_noagn | no_satellites_or_other | 4 | -9.4 [-18.2, -3.8] | +10.4 [+5.6, +18.1] | +5.1 [+3.3, +8.2] | -3.78 | 1.00 |
| 13.9-14.1 | pressure_noagn | no_satellites_or_other | 4 | -11.1 [-14.3, -7.2] | +14.7 [+10.7, +17.9] | +8.1 [+6.3, +9.5] | -3.92 | 1.00 |
| 14.1-14.3 | pressure_noagn | no_satellites_or_other | 4 | -10.1 [-14.2, -6.4] | +13.9 [+10.1, +17.7] | +7.9 [+6.2, +9.4] | -4.27 | 1.00 |
| 14.3-14.5 | pressure_noagn | no_satellites_or_other | 4 | -11.3 [-13.7, -8.6] | +15.1 [+12.3, +18.0] | +8.3 [+7.1, +9.5] | -4.08 | 1.00 |
| 14.5-14.7 | pressure_noagn | no_satellites_or_other | 4 | -16.0 [-17.5, -14.1] | +20.5 [+18.6, +22.3] | +10.6 [+9.8, +11.4] | -4.28 | 1.00 |
| 14.7-14.9 | pressure_noagn | no_satellites_or_other | 4 | -7.1 [-8.7, -5.1] | +11.4 [+9.4, +13.6] | +6.8 [+5.9, +7.9] | -4.29 | 1.00 |
| 14.9-15.1 | pressure_noagn | no_satellites_or_other | 4 | -6.9 [-8.3, -4.7] | +11.4 [+9.1, +13.2] | +6.9 [+5.7, +7.9] | -3.39 | 1.00 |
| 15.1-15.3 | pressure_noagn | no_satellites_or_other | 4 | +0.0 [-5.3, +3.2] | +3.4 [-2.7, +10.0] | +3.2 [-1.2, +7.0] | -2.32 | 1.00 |
| 15.3-15.5 | pressure_noagn | no_satellites_or_other | 4 | -36.7 [-33.8, -2.7] | +17.8 [-2.6, +18.1] | +4.9 [-2.2, +6.6] | -6.62 | 1.00 |
| 13.3-13.5 | pressure_noagn | host_bound | 4 | +86.4 [+13.6, +141.5] | +83.9 [+48.7, +140.0] | +95.8 [+79.4, +121.5] | -60.05 | 1.00 |
| 13.5-13.7 | pressure_noagn | host_bound | 4 | +41.3 [-5.4, +54.9] | +69.6 [+55.8, +124.7] | +84.3 [+74.3, +118.7] | -50.33 | 0.98 |
| 13.7-13.9 | pressure_noagn | host_bound | 4 | -48.8 [-68.6, -25.8] | +140.8 [+113.1, +169.1] | +114.2 [+95.6, +134.8] | -42.23 | 0.95 |
| 13.9-14.1 | pressure_noagn | host_bound | 4 | -95.3 [-98.6, -77.6] | +189.1 [+162.1, +195.4] | +144.5 [+123.6, +150.9] | -36.24 | 0.73 |
| 14.1-14.3 | pressure_noagn | host_bound | 4 | -68.2 [-87.6, -46.2] | +141.8 [+110.2, +174.2] | +107.6 [+85.7, +132.3] | -32.69 | 0.91 |
| 14.3-14.5 | pressure_noagn | host_bound | 4 | -50.7 [-59.0, -36.7] | +108.2 [+89.4, +121.7] | +80.7 [+68.2, +91.3] | -29.82 | 1.00 |
| 14.5-14.7 | pressure_noagn | host_bound | 4 | -25.8 [-31.3, -19.8] | +71.7 [+64.2, +78.6] | +55.7 [+51.0, +60.0] | -28.18 | 1.00 |
| 14.7-14.9 | pressure_noagn | host_bound | 4 | -39.3 [-48.8, -31.4] | +92.0 [+78.7, +107.2] | +70.4 [+61.3, +80.7] | -27.17 | 1.00 |
| 14.9-15.1 | pressure_noagn | host_bound | 4 | -49.1 [-57.0, -36.5] | +109.0 [+86.8, +124.0] | +84.1 [+68.2, +95.7] | -26.50 | 1.00 |
| 15.1-15.3 | pressure_noagn | host_bound | 4 | not accepted | not accepted | not accepted | -24.02 | 0.35 |
| 15.3-15.5 | pressure_noagn | host_bound | 4 | -72.1 [-88.9, -23.3] | +86.2 [+54.4, +109.0] | +57.6 [+40.4, +70.6] | -27.85 | 0.99 |
| 13.3-13.5 | electrons | no_other | 2 | +0.4 [+0.3, +0.5] | -0.2 [-0.2, -0.1] | +0.2 [+0.2, +0.2] | -0.09 | 1.00 |
| 13.5-13.7 | electrons | no_other | 2 | +0.1 [+0.0, +0.1] | +0.1 [+0.1, +0.1] | +0.1 [+0.1, +0.1] | -0.09 | 1.00 |
| 13.7-13.9 | electrons | no_other | 2 | -0.1 [-0.1, -0.1] | +0.3 [+0.2, +0.3] | +0.0 [+0.0, +0.1] | -0.08 | 1.00 |
| 13.9-14.1 | electrons | no_other | 2 | -0.3 [-0.4, -0.3] | +0.4 [+0.3, +0.4] | +0.0 [+0.0, +0.0] | -0.08 | 1.00 |
| 14.1-14.3 | electrons | no_other | 2 | -0.4 [-0.5, -0.3] | +0.4 [+0.3, +0.4] | +0.0 [+0.0, +0.0] | -0.07 | 1.00 |
| 14.3-14.5 | electrons | no_other | 2 | -0.0 [-0.2, +0.1] | +0.1 [+0.1, +0.2] | +0.1 [+0.1, +0.1] | -0.07 | 1.00 |
| 14.5-14.7 | electrons | no_other | 2 | -0.0 [-0.1, +0.0] | +0.1 [+0.1, +0.2] | +0.1 [+0.1, +0.1] | -0.09 | 1.00 |
| 14.7-14.9 | electrons | no_other | 2 | -0.2 [-0.3, +0.0] | +0.2 [+0.1, +0.3] | +0.1 [+0.1, +0.1] | -0.08 | 1.00 |
| 14.9-15.1 | electrons | no_other | 2 | -0.4 [-0.6, -0.1] | +0.3 [+0.2, +0.4] | +0.1 [+0.1, +0.2] | -0.11 | 1.00 |
| 15.1-15.3 | electrons | no_other | 2 | -0.1 [-0.5, +0.1] | +0.2 [+0.1, +0.3] | +0.1 [+0.0, +0.1] | -0.06 | 1.00 |
| 15.3-15.5 | electrons | no_other | 2 | -4.7 [-4.5, -1.1] | +1.3 [+0.4, +1.3] | -0.2 [-0.1, +0.1] | -0.06 | 1.00 |
| 13.3-13.5 | electrons | no_satellites_or_other | 2 | -0.4 [-0.8, -0.0] | -0.8 [-1.0, -0.5] | +0.5 [+0.3, +0.6] | -1.62 | 1.00 |
| 13.5-13.7 | electrons | no_satellites_or_other | 2 | -0.9 [-1.2, -0.7] | +0.6 [+0.3, +1.1] | +0.3 [+0.2, +0.4] | -1.19 | 1.00 |
| 13.7-13.9 | electrons | no_satellites_or_other | 2 | -1.3 [-1.7, -1.0] | +1.8 [+1.3, +2.3] | +0.3 [+0.2, +0.4] | -1.07 | 1.00 |
| 13.9-14.1 | electrons | no_satellites_or_other | 2 | -2.7 [-3.5, -2.0] | +3.2 [+2.5, +4.0] | +0.6 [+0.6, +0.8] | -1.38 | 1.00 |
| 14.1-14.3 | electrons | no_satellites_or_other | 2 | -3.5 [-4.4, -2.7] | +3.3 [+2.6, +4.0] | +0.8 [+0.7, +1.0] | -1.80 | 1.00 |
| 14.3-14.5 | electrons | no_satellites_or_other | 2 | -3.4 [-3.7, -2.1] | +3.0 [+2.1, +3.2] | +1.0 [+0.9, +1.2] | -1.67 | 1.00 |
| 14.5-14.7 | electrons | no_satellites_or_other | 2 | -0.8 [-1.1, -0.6] | +1.4 [+1.2, +1.6] | +1.0 [+0.9, +1.1] | -1.73 | 1.00 |
| 14.7-14.9 | electrons | no_satellites_or_other | 2 | -0.6 [-0.9, -0.2] | +1.1 [+0.9, +1.4] | +1.1 [+0.9, +1.3] | -2.24 | 1.00 |
| 14.9-15.1 | electrons | no_satellites_or_other | 2 | -0.1 [-1.4, +0.7] | +0.9 [+0.5, +1.5] | +1.2 [+0.9, +1.4] | -2.24 | 1.00 |
| 15.1-15.3 | electrons | no_satellites_or_other | 2 | +1.4 [-2.4, +1.0] | +0.3 [+0.1, +2.0] | +1.2 [+0.7, +1.4] | -1.50 | 1.00 |
| 15.3-15.5 | electrons | no_satellites_or_other | 2 | +47.0 [-23.1, +37.8] | -16.6 [-12.9, +3.6] | -1.0 [-3.4, +0.4] | -6.04 | 1.00 |
| 13.3-13.5 | electrons | host_bound | 2 | +133.7 [+131.1, +136.8] | +1.7 [+1.0, +2.5] | +75.6 [+74.8, +76.5] | -45.35 | 1.00 |
| 13.5-13.7 | electrons | host_bound | 2 | +43.7 [+42.0, +45.3] | +15.7 [+14.3, +17.2] | +53.2 [+52.4, +54.1] | -41.76 | 1.00 |
| 13.7-13.9 | electrons | host_bound | 2 | +16.8 [+14.9, +18.8] | +37.0 [+35.2, +38.9] | +41.7 [+41.0, +42.7] | -37.61 | 1.00 |
| 13.9-14.1 | electrons | host_bound | 2 | -27.1 [-30.1, -24.1] | +72.0 [+70.5, +73.6] | +29.6 [+28.9, +30.4] | -34.05 | 1.00 |
| 14.1-14.3 | electrons | host_bound | 2 | -40.6 [-43.1, -38.2] | +72.9 [+71.8, +74.1] | +26.7 [+26.0, +27.4] | -31.94 | 1.00 |
| 14.3-14.5 | electrons | host_bound | 2 | -49.1 [-50.5, -46.2] | +65.7 [+64.4, +66.7] | +25.1 [+24.7, +25.6] | -29.56 | 1.00 |
| 14.5-14.7 | electrons | host_bound | 2 | -46.6 [-47.6, -45.8] | +59.0 [+58.3, +59.7] | +25.8 [+25.5, +26.0] | -28.76 | 1.00 |
| 14.7-14.9 | electrons | host_bound | 2 | -39.8 [-41.5, -38.5] | +49.0 [+47.7, +50.5] | +25.6 [+25.2, +26.1] | -28.39 | 1.00 |
| 14.9-15.1 | electrons | host_bound | 2 | -35.9 [-38.2, -32.8] | +42.3 [+40.3, +44.1] | +24.6 [+24.2, +25.5] | -28.61 | 1.00 |
| 15.1-15.3 | electrons | host_bound | 2 | -30.0 [-38.4, -20.8] | +35.5 [+31.1, +41.2] | +22.0 [+20.7, +24.1] | -25.75 | 1.00 |
| 15.3-15.5 | electrons | host_bound | 2 | -34.1 [-80.3, -20.1] | +18.4 [+18.1, +40.1] | +12.8 [+11.4, +23.2] | -29.91 | 1.00 |
| 13.3-13.5 | electrons | no_other | 4 | +10.7 [+8.4, +12.7] | -4.3 [-5.4, -3.0] | +3.0 [+2.4, +3.5] | -2.20 | 1.00 |
| 13.5-13.7 | electrons | no_other | 4 | +5.6 [+4.9, +6.2] | -3.8 [-4.4, -3.2] | +2.0 [+1.8, +2.2] | -1.87 | 1.00 |
| 13.7-13.9 | electrons | no_other | 4 | +6.8 [+5.9, +7.7] | -6.3 [-7.3, -5.2] | +2.2 [+2.0, +2.5] | -2.10 | 1.00 |
| 13.9-14.1 | electrons | no_other | 4 | -2.0 [-3.1, -1.2] | +3.6 [+2.9, +4.5] | +0.7 [+0.6, +0.8] | -2.49 | 1.00 |
| 14.1-14.3 | electrons | no_other | 4 | -8.5 [-9.7, -7.0] | +6.7 [+5.8, +7.4] | +0.3 [+0.2, +0.4] | -2.51 | 1.00 |
| 14.3-14.5 | electrons | no_other | 4 | -10.1 [-11.3, -9.0] | +6.2 [+5.6, +6.8] | +0.6 [+0.5, +0.6] | -2.72 | 1.00 |
| 14.5-14.7 | electrons | no_other | 4 | -12.9 [-13.7, -12.1] | +6.7 [+6.3, +7.0] | +0.8 [+0.8, +0.8] | -3.01 | 1.00 |
| 14.7-14.9 | electrons | no_other | 4 | -9.5 [-10.3, -8.8] | +5.2 [+4.9, +5.6] | +0.9 [+0.9, +1.0] | -2.89 | 1.00 |
| 14.9-15.1 | electrons | no_other | 4 | -9.7 [-10.6, -8.6] | +5.2 [+4.7, +5.6] | +1.1 [+1.0, +1.2] | -3.32 | 1.00 |
| 15.1-15.3 | electrons | no_other | 4 | -5.2 [-7.1, -3.9] | +2.2 [+1.5, +3.7] | -0.1 [-0.3, +0.6] | -3.49 | 1.00 |
| 15.3-15.5 | electrons | no_other | 4 | -6.8 [-14.0, -2.4] | +2.2 [+1.5, +4.0] | +0.3 [-0.2, +1.0] | -3.01 | 1.00 |
| 13.3-13.5 | electrons | no_satellites_or_other | 4 | +9.0 [+6.5, +11.0] | -3.9 [-5.1, -2.6] | +3.1 [+2.5, +3.5] | -3.85 | 1.00 |
| 13.5-13.7 | electrons | no_satellites_or_other | 4 | +5.0 [+4.0, +5.7] | -3.5 [-4.2, -2.7] | +2.3 [+2.0, +2.6] | -3.33 | 1.00 |
| 13.7-13.9 | electrons | no_satellites_or_other | 4 | +5.4 [+4.2, +6.6] | -4.4 [-5.7, -3.0] | +2.6 [+2.3, +2.9] | -3.68 | 1.00 |
| 13.9-14.1 | electrons | no_satellites_or_other | 4 | -4.8 [-5.9, -3.6] | +6.3 [+5.3, +7.2] | +1.0 [+0.9, +1.2] | -4.32 | 1.00 |
| 14.1-14.3 | electrons | no_satellites_or_other | 4 | -9.0 [-10.9, -7.3] | +7.3 [+6.2, +8.4] | +1.0 [+0.8, +1.1] | -4.84 | 1.00 |
| 14.3-14.5 | electrons | no_satellites_or_other | 4 | -11.6 [-13.0, -10.0] | +7.5 [+6.7, +8.3] | +1.4 [+1.3, +1.5] | -5.48 | 1.00 |
| 14.5-14.7 | electrons | no_satellites_or_other | 4 | -15.0 [-15.9, -13.9] | +8.2 [+7.7, +8.6] | +1.6 [+1.6, +1.7] | -5.96 | 1.00 |
| 14.7-14.9 | electrons | no_satellites_or_other | 4 | -9.4 [-10.4, -8.2] | +5.7 [+5.1, +6.1] | +1.9 [+1.8, +2.0] | -6.21 | 1.00 |
| 14.9-15.1 | electrons | no_satellites_or_other | 4 | -9.3 [-10.6, -7.9] | +5.6 [+4.9, +6.1] | +2.0 [+1.9, +2.2] | -6.52 | 1.00 |
| 15.1-15.3 | electrons | no_satellites_or_other | 4 | -4.5 [-6.8, -2.7] | +2.9 [+1.8, +4.4] | +1.3 [+0.9, +1.9] | -5.91 | 1.00 |
| 15.3-15.5 | electrons | no_satellites_or_other | 4 | -7.0 [-31.4, +6.2] | +2.5 [-1.5, +9.0] | +2.6 [+1.1, +2.8] | -10.91 | 1.00 |
| 13.3-13.5 | electrons | host_bound | 4 | +188.5 [+170.5, +213.4] | +10.0 [+5.5, +13.5] | +80.5 [+77.9, +84.0] | -72.52 | 1.00 |
| 13.5-13.7 | electrons | host_bound | 4 | +91.4 [+82.8, +103.0] | +9.8 [+6.2, +12.8] | +65.2 [+63.2, +68.1] | -66.82 | 1.00 |
| 13.7-13.9 | electrons | host_bound | 4 | +72.7 [+65.9, +82.6] | +5.4 [+1.6, +8.5] | +58.3 [+56.5, +60.9] | -61.30 | 1.00 |
| 13.9-14.1 | electrons | host_bound | 4 | -1.6 [-6.7, +4.2] | +67.1 [+63.3, +70.3] | +44.4 [+42.9, +46.3] | -56.96 | 1.00 |
| 14.1-14.3 | electrons | host_bound | 4 | -49.2 [-54.5, -42.3] | +91.8 [+88.7, +94.8] | +34.6 [+33.5, +36.7] | -54.14 | 1.00 |
| 14.3-14.5 | electrons | host_bound | 4 | -76.8 [-81.1, -72.1] | +90.1 [+88.5, +92.5] | +29.9 [+29.4, +31.0] | -51.42 | 1.00 |
| 14.5-14.7 | electrons | host_bound | 4 | -96.6 [-99.2, -94.3] | +89.4 [+88.1, +90.8] | +30.1 [+29.5, +30.6] | -50.36 | 1.00 |
| 14.7-14.9 | electrons | host_bound | 4 | -79.9 [-83.0, -77.7] | +74.3 [+73.2, +76.3] | +30.8 [+30.3, +31.7] | -49.25 | 1.00 |
| 14.9-15.1 | electrons | host_bound | 4 | -75.4 [-80.1, -71.9] | +67.8 [+66.5, +70.7] | +31.0 [+30.4, +32.2] | -49.09 | 1.00 |
| 15.1-15.3 | electrons | host_bound | 4 | -55.0 [-64.5, -52.2] | +55.5 [+53.9, +77.2] | +28.8 [+28.1, +38.7] | -46.84 | 1.00 |
| 15.3-15.5 | electrons | host_bound | 4 | -124.3 [-163.2, -65.3] | +59.7 [+43.5, +72.7] | +23.7 [+20.6, +27.9] | -48.09 | 1.00 |

Parameters: P0/xc/beta for pressure, rho0/alpha/beta for electrons. Selections keep the all-gas halo normalisation and the full shell volume. `tables/membership_controls.csv` also lists objective (a).

![Membership](figures/fig6_membership_pressure_noagn.png)

## 7. Scope extension: observable-driven weighting and flexible families

New library capabilities (`analysis/README.md`): objectives (d) per-shell weights, (e) any PSD metric (e.g. K = sum_O lambda_O g_O g_O^T/O^2 with g = F dO/dF from `fitting.observable_metric`), (f) hybrid r^T C^-1 r + lambda r^T K r with K trace-normalised to C^-1, (g) appended extra residuals; families gNFW (5 free), B12 x (1 + natural cubic spline in ln x, ridge prior) and B12 + A (x/2)^-s; truncated shell averages; per-shell residuals, importance-weighted RMS and R_fit x R_cut grids. Demonstration with a TOY importance vector (plateau 0.3-1.5 R200c; replace with the kernels/ Jacobians), pressure without recent AGN:

| Bin | Family | Objective | R_fit | Accepted | RMS <1R % | RMS 1-2R % | RMS 2-4R % | Importance RMS % |
|---|---|---|---|---|---|---|---|---|
| 14.5-14.7 | b12 | a | 2 | True | 0.25 | 2.12 | 47.0 | 5.95 |
| 14.5-14.7 | b12 | c | 2 | True | 0.32 | 1.99 | 46.0 | 5.73 |
| 14.5-14.7 | b12 | d | 2 | True | 0.47 | 1.94 | 46.2 | 5.76 |
| 14.5-14.7 | b12 | f10 | 2 | True | 0.46 | 1.95 | 46.2 | 5.77 |
| 14.5-14.7 | b12 | a | 3 | True | 0.66 | 3.91 | 42.1 | 5.14 |
| 14.5-14.7 | b12 | c | 3 | True | 2.56 | 7.29 | 34.7 | 5.53 |
| 14.5-14.7 | b12 | d | 3 | True | 3.08 | 3.98 | 39.1 | 4.82 |
| 14.5-14.7 | b12 | f10 | 3 | True | 2.90 | 4.22 | 38.8 | 4.83 |
| 14.5-14.7 | b12 | a | 4 | True | 1.60 | 8.18 | 36.3 | 5.68 |
| 14.5-14.7 | b12 | c | 4 | True | 7.62 | 16.36 | 21.7 | 11.72 |
| 14.5-14.7 | b12 | d | 4 | True | 4.74 | 5.36 | 35.9 | 4.98 |
| 14.5-14.7 | b12 | f10 | 4 | True | 6.15 | 7.39 | 31.9 | 5.67 |
| 14.5-14.7 | b12env | a | 2 | False | 0.35 | 1.24 | 23.4 | 2.12 |
| 14.5-14.7 | b12env | f10 | 2 | False | 0.87 | 0.61 | 85.4 | 6.16 |
| 14.5-14.7 | b12env | a | 3 | True | 0.38 | 1.24 | 37.1 | 2.58 |
| 14.5-14.7 | b12env | f10 | 3 | True | 1.04 | 0.78 | 19.3 | 1.47 |
| 14.5-14.7 | b12env | a | 4 | True | 0.51 | 1.69 | 4.3 | 1.15 |
| 14.5-14.7 | b12env | f10 | 4 | True | 1.37 | 0.92 | 4.7 | 0.86 |
| 13.9-14.1 | b12 | a | 2 | True | 0.54 | 3.08 | 44.4 | 5.47 |
| 13.9-14.1 | b12 | c | 2 | True | 0.92 | 2.05 | 40.9 | 4.58 |
| 13.9-14.1 | b12 | d | 2 | True | 0.75 | 2.02 | 40.8 | 4.54 |
| 13.9-14.1 | b12 | f10 | 2 | True | 0.75 | 2.03 | 40.8 | 4.54 |
| 13.9-14.1 | b12 | a | 3 | True | 0.73 | 3.26 | 37.8 | 4.08 |
| 13.9-14.1 | b12 | c | 3 | True | 1.71 | 6.26 | 30.3 | 4.37 |
| 13.9-14.1 | b12 | d | 3 | True | 1.62 | 2.71 | 35.2 | 3.72 |
| 13.9-14.1 | b12 | f10 | 3 | True | 1.64 | 2.75 | 35.1 | 3.72 |
| 13.9-14.1 | b12 | a | 4 | True | 1.88 | 9.05 | 27.9 | 5.65 |
| 13.9-14.1 | b12 | c | 4 | True | 6.58 | 10.54 | 20.6 | 8.72 |
| 13.9-14.1 | b12 | d | 4 | True | 2.42 | 3.60 | 32.0 | 3.77 |
| 13.9-14.1 | b12 | f10 | 4 | True | 3.40 | 4.61 | 28.8 | 4.24 |
| 13.9-14.1 | b12env | a | 2 | False | 0.75 | 0.78 | 117.4 | 10.09 |
| 13.9-14.1 | b12env | f10 | 2 | True | 0.72 | 0.63 | 124.9 | 10.82 |
| 13.9-14.1 | b12env | a | 3 | True | 0.76 | 1.08 | 5.8 | 0.94 |
| 13.9-14.1 | b12env | f10 | 3 | True | 0.38 | 0.70 | 11.8 | 0.94 |
| 13.9-14.1 | b12env | a | 4 | True | 0.70 | 1.30 | 2.3 | 1.04 |
| 13.9-14.1 | b12env | f10 | 4 | True | 0.72 | 0.85 | 4.5 | 0.85 |

Best (R_fit, R_cut) per objective from the grids (model truncated at R_cut, evaluated on all resolved shells to 4R): 14.5-14.7 (a): R_fit 3, R_cut 4, 5.14%; 14.5-14.7 (c): R_fit 3, R_cut 4, 5.53%; 14.5-14.7 (d): R_fit 3, R_cut 4, 4.82%; 14.5-14.7 (f10): R_fit 3, R_cut 4, 4.83%; 13.9-14.1 (a): R_fit 3, R_cut 4, 4.08%; 13.9-14.1 (c): R_fit 3, R_cut 4, 4.37%; 13.9-14.1 (d): R_fit 3, R_cut 4, 3.72%; 13.9-14.1 (f10): R_fit 3, R_cut 4, 3.72%. With this toy weight the 2-3R shells still carry weight, so truncating at 2R costs ~25% importance-weighted RMS for every objective.

![Importance demo](figures/fig7_importance_demo_metrics.png)

![Importance residuals](figures/fig7_importance_demo_residuals.png)

Answer to the question posed: upweighting alone does not let a B12 fit extend to 4 R200c without degrading the important shells (1-2R RMS 1.9-3.1% for 2R fits, 3.6-9.1% for (a), (d) and lambda >= 10 hybrids fitted to 4R; inside 1R 0.25-0.92% versus 1.6-6.2%), although a 3R fit with importance or hybrid (lambda >= 10) weighting improves the importance-weighted RMS over the current 2R fit. A second, environment component (B12 + A (x/2)^-s, s free and signed) fitted to 3-4R keeps the important shells at the 2R-fit level. When the real kernels/ Jacobians are available, rerun q7 with `fitting.observable_metric`.

## Caveats

* The 36-shell grid lacks 1.34/1.5/1.75 R200c edges (nearest 1.3508, 1.5845, 1.8587 used) and Lee22's 27-shell grid; the coming 69-shell x200 / 40-shell x500 extraction fixes this, and `analysis.io.rebin` is ready for it.
* Bootstrap intervals describe halo sampling at fixed cosmic volume only (no spatial/cosmic variance; 2-4R spheres of neighbouring haloes overlap and share gas). The plan's 4^3 sub-volume jackknife is not applied here.
* sigma_d is a white, constant-fractional discrepancy. When it dominates (fits beyond ~2R) it decides the weighting of the GLS fit; a correlated or radius-dependent discrepancy model would give different 3-4R parameters. Analytic GLS errors include sigma_d, bootstrap intervals do not.
* Warm-started replicate fits under-represent strongly degenerate directions (e.g. B16 on 9 shells at 13.3-13.5: analytic errors >> bootstrap SD); treat those bootstrap intervals as lower limits.
* The (a, L) correlation model is a poor description (RMS 0.15-0.39); its Delta a/Delta ln L permutation p-values are reported but carry little weight. The pooled shrinkage target uses the (N-3)-weighted Fisher-z average of the other supported bins (uncovered inner shells of the top bins filled from the nearest covered shell); making it PSD by eigenvalue clipping changes its entries by 0.02-0.065 for pressure but up to 0.13 for electrons, whose structure drifts with mass. Later tests should pool neighbouring bins only (`covariance.nearest_correlation(method='higham')` is available).
* Electron and gas correlation structures change significantly between neighbouring bins; pooling them across the full mass range is a stronger approximation than for pressure.
* The scope-extension demonstration uses a toy importance vector, not the kernels/ Jacobians, and B12 + outer power law is a phenomenological environment term (it does not separate ownership; see Test A).
* AGN removal uses the radii234 `y_noagn` estimator (SOAP RecentlyHeatedGasFilter); electron and gas densities have no AGN split in this extraction.

## Files, runtimes and reproduction

| Item | Script | Products |
|---|---|---|
| 1 | `q1_regression_gate.py` | `products/regression_gate.json`, `tables/regression_gate_*.csv` |
| 2 | `q2_correlations.py` | `products/correlation_matrices.npz`, `products/correlation_summary.json`, `figures/fig2_*` |
| 3 | `q3_stability.py` | `products/stability.json`, `tables/stability_*.csv`, `tables/neighbour_covariance_shift.csv`, `figures/fig3_stability.*` |
| 4 | `q4_fits.py`, `q4_report.py` | `products/fits_cutoffs.npz`, `products/fits_membership.npz`, `products/fits_full_sample.json`, `products/pooled_correlation_targets.npz`, `tables/fits_*.csv`, `figures/fig4_*` |
| 5 | `q5_agn.py` | `products/agn_paired.{npz,json}`, `tables/agn_*.csv`, `figures/fig5_agn_paired.*` |
| 6 | `q6_membership.py` | `products/membership_*`, `tables/membership_controls.csv`, `figures/fig6_*` |
| 7 | `q7_importance_demo.py` | `products/importance_demo.json`, `tables/importance_demo*.csv`, `figures/fig7_*` |
| - | `verify.py`, `build_report.py`, `run_all.py` | `verification.json`, `SHA256SUMS`, this report |

Wall times on this machine (10 workers, single-threaded BLAS): tests 306 s, q4 444 s, q1 133 s, q2 15 s, q3 16 s, q4_report 10 s, q5 3 s, q6 4 s, q7 5 s.

Reproduce: `cd quicklook && /home/cbllover/miniconda3/envs/flamingo/bin/python -B run_all.py` (then `python -B verify.py products/session_start_inputs.sha256`, the input hashes recorded at session start). Product arrays: `fits_cutoffs.npz` `ln_parameters[bin, field, draw, cut, objective, parameter]` (draw 0 = full sample), `accepted`, `sigma_d`, `shrinkage_lambda`, `means[bin, field, draw, shell]`, influence rows `H[bin, field, cut, objective, parameter, shell]`, `n_eff`, `crossfit_ln_parameters`.

