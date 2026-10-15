# Test B: covariance, weightings, covariance-weighted fits and cut-offs (FLAMINGO L1_m9, z = 0)

Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md`, "Test B" (with "Literature corrections" and
"AGN-heated gas"). This supersedes the first pass in `../quicklook/REPORT.md` (radii234 data, 36 shells, nearest edges):
it uses the new particle-pass products, the exact cut-off edges, Lee22's exact 27-shell grid, the enlarged sample with
inclusion weights, AGN-removed electrons and the new estimators (Q_V, Q_M, Q_X, n_e kT_X).

**Run.** `run_test_b.py --full --workers 4` on STRW maas (nice 19), in two invocations:
- the first ran prepare, the CLEAN fits, templates, all fits and B1 (7,062 s of stages; 7,111 s wall including the
  failed B2 attempt), then stopped in B2 (see Verification);
- after the fix, `--stages b2,b3,jackknife,engine` reused the saved fit products (2,484 s of stages; 2,486 s wall,
  exit 0).

Figures, report tables, `verification.json` and this report were made locally from `results/` (`make_figures.py`).

**Conventions.**
- Sample: 4,588 centrals (the 4,375 radii234 haloes plus 213 annulus-only additions).
- Stacks: inclusion-weighted means.
- Pressure: the geometric estimator sum(y)/V_shell, with recently AGN-heated gas removed (default). Electrons: n_e V summed per shell (B16). AGN kept is the paired control.
- Fit grid: 35 shells (the radii234 edges to 0.84 R200c plus 1, 1.15, 1.34, 1.5, 1.75, 2, 2.18, 2.5, 3, 3.30, 3.63, 4), so every cut-off is an exact edge.
- Fit units: the 0.2-dex bins 13.5-13.7 ... 14.7-14.9, plus the pooled >= 14.9 sample (154 haloes) with a per-shell linear log M trend, evaluated at the mean mass of each of 14.9-15.1, 15.1-15.3 and 15.3-15.5.
- Uncertainties: 500 paired replicates, stratified by (bin, census class), so CLEAN, CONT and ALL are paired draw by draw. Cov[ln mean] is re-estimated from 1,000 inner resamples in every replicate.
- GLS objective (c): shrinkage toward the pooled correlation of the neighbouring bins, plus sigma_d solved for chi2/nu = 1.
- Observables: computed with the kernel engine and a hard cut at 4 R200c. Brackets are 16-84% of the replicates, propagated linearly with the kernel Jacobians around the exact change.

## Plain answers

1. **What the covariance looks like (B1).** Per-halo ln P_e on Lee22's exact grid has two regimes.
   - At 10^13.2-13.4 h^-1 Msun only 0.42-1.34 R200c is resolved for every halo. The 9 shells form one positive block
     (adjacent rho 0.89, mean across 0.5R +0.61), and a single exponential (a, L) fits it (RMS 0.04).
   - At 10^13.8-14.0 h^-1 Msun (0.15-1.34 R200c, 17 shells) there are two blocks:
     - inside ~0.6 R200c, shells are strongly correlated (mean rho 0.82 below 0.5R);
     - outside, they are moderately correlated (0.54);
     - the two blocks are anti-correlated (-0.22; fitted break 0.61 R200c, r_cross -0.27).
   - The (a, L) model fails there (RMS 0.33); the two-block model halves the residual (0.21).
   - Above 10^14 Msun the break moves inward with mass, from 0.6 to 0.4 R200c, and r_cross stays at -0.2 to -0.4.
   - Electrons look the same. CONT haloes have weaker inner-outer anti-correlation (-0.05 vs -0.27 for CLEAN).
   - The sampling covariance of ln(mean) that a GLS fit inverts is near-singular on Lee22's grid: adjacent rho 0.90-0.94
     and condition numbers 220-1,500, against 200-770 on the coarser fit grid.
2. **Is it stable between adjacent bins? (B2)** For pressure, mostly yes above 10^13.9.
   - The permutation null (1,000 relabellings at the original N) rejects equality for 3 of 11 pressure pairs, all below
     10^14.1 Msun (13.1-13.3 vs 13.3-13.5, 13.3-13.5 vs 13.5-13.7, 13.7-13.9 vs 13.9-14.1), where the resolved range
     moves inward and the two-block structure appears. The pressure pairs from 13.9 to 15.5 are consistent with one
     structure (p = 0.06-0.7).
   - For electrons it is no: 9 of 11 pairs differ (every pair from 13.1 to 14.9, and 15.1-15.3 vs 15.3-15.5 at p = 0.049).
   - Lee22's qualitative claim therefore holds for pressure at cluster masses, but not for electron density or across 10^14.
   - It hardly matters for the fit. Fitting each bin with its neighbour's correlation moves the parameters by a median of
     0.1-0.25 bootstrap sigma per cut-off (at most 0.9 sigma at 2R, and 1.4 sigma over all cut-offs). The observables move
     by <= 0.3% (C_ell^yy <= 0.22%, mean DM <= 0.06%).
3. **How much GLS changes the fit and the observables (B4).**
   - Up to 1.75-2 R200c, GLS and the current diagonal fit agree to a few per cent along the P0-xc-beta degeneracy. At 2R
     the median Delta ln P0 is +4% (range -5 to +8%). C_ell^yy changes by <= 0.2% at 2R and <= 2.1% at 1-1.34R; mean DM
     by -1.0% at 2R.
   - Beyond 2R the B12/B16 shapes cannot describe the stack: sigma_d rises from 0.3% (pressure, 2R) to 3.5% (3R) and 12%
     (4R). The white sigma_d term then dominates the covariance and turns GLS into a nearly unweighted ln F fit that
     follows the 2-4R tail. At 4R this gives P0 +38%, xc -42%, beta -19%, C_ell^yy -3 to -6% and kSZ -3.8%.
   - Agreement with Test F (F3: GLS is the worst weighting for the end observables). Test B's (c) - (a) shifts match
     Test F's own:
     - at 4R to <= 0.5 points (C_ell^yy -3.3 to -5.8% vs F -3.4 to -5.7%; kSZ at ell = 3000 -3.8% in both; DM variance
       -3.0 vs -2.9%);
     - for electrons at 2R to <= 0.1 points (mean DM -1.00 vs -0.99%; kSZ -0.8 to -1.8% vs F -0.8 to -1.7%).
   - F's comparison with the truth gives the sign. The diagonal fit is already low in C_ell^yy (-3.5 to -5.1% at 2R,
     -1.0 to -2.8% at 4R), and GLS moves it further down. So GLS is worse for the observables although it is the
     statistically efficient fit of the profile.
   - The reason is that the covariance weighting is only meaningful where the model is adequate. Beyond about 2R the
     choice of weighting becomes a choice of which radii to sacrifice, and the observables, not the covariance, should
     decide it.
   - One difference remains. At 2R, F's GLS lowers C_ell^yy by 0.5-1.3%, while Test B's moves it by <= 0.16%. The likely
     cause (not tested here) is that F fits 15.1-15.3 (N = 21) and 15.3-15.5 (N = 6) individually, with the pooled
     correlation forced, whereas Test B pools >= 14.9 (N = 154) with a log M trend. These bins dominate C_ell^yy, and they
     matter little for kSZ and DM, which agree.
   - The plain error-on-mean diagonal (b) equals (a) to <= 0.5% in C_ell^yy at every cut-off. Cross-fitting, a 2- or
     3-factor covariance, unshrunk correlations and neighbour correlations all move the 2R fit by at most 1.4 sigma
     (medians <= 0.45 sigma). The covariance model itself still matters for C_ell: a 2-factor covariance changes C_ell^yy
     by up to -2.5% at 2R.
4. **Which radii drive the fit.** For the diagonal fit, P0, xc and beta are driven by 0.1-1 R200c; xc and beta peak at
   0.5-0.6 R200c.
   - At 14.5-14.7 (fit to 2R): 0.5-1R carries 29-31% of |H| and 1-2R 13-24%; the effective number of shells is 13-15 of 20.
   - GLS spreads the influence to the least redundant end shells: the innermost resolved shells (0-0.1R) rise from 5% to
     13-19%, and 1.5-2R from 7-11% to 11-20%. N_eff drops from 14.6 to 10.5 for P0 and stays at about 13 for xc and beta.
   - At 4R GLS gives 3-4R (12-21%) at least as much weight as 0.25-0.5R (8-15%).
5. **Which cut-off is best supported.**
   - Pressure: the B12 form describes the stack within its statistical errors up to 2 R200c (sigma_d <= 0.7%, chi2/nu
     <= 3.4 in every bin). It breaks beyond (3R: sigma_d 1.9-5.8%).
   - Electrons (B16) already need sigma_d at small cut-offs: a median of 0.6% at 1R and 1.2% at 1.34R (up to 1.8%), and
     1.6% at 2R.
   - Lee22's 1.01, 1.34 and 1.76 analogues (cuts 1.0, 1.34, 1.75) are all supported, and they change C_ell^yy by
     <= 1.4% relative to 2R.
   - For DM the cut matters, because DM is 1-4R dominated: relative to the 2R fit, mean DM is +9.0% (1R), +4.6% (1.34R),
     +1.5% (1.75R), -2.1% (3R) and -0.4% (4R).
   - Fits to 2R extrapolate below the measured 2-4R shells, by a median of 31-37% (pressure and electrons, objectives (a)
     and (c)). They miss Y(<4R) by a median of -6% (pressure) and -8% (electrons). The electron fits of 13.5-13.9 are
     the exception: they overshoot the 2-4R shells by 14-31%.
   - So, of 1.0 / 1.34 / 1.5 / 1.75 / 2 / 3 / 4 R200c, the best-supported fit range is 2R for both fields. For pressure
     every cut-off up to 2R is adequate; for electrons sigma_d is 1.6% (median) at 2R and rises to 6% at 3R and 11% at 4R.
     Lee22's 1.34 lies inside this range.
   - The observables need the 2-4R gas, which only a form with an outer component can carry (Test F: fit and cut at 4R).
6. **B24 weightings versus the tSZ-correct estimator (B3).**
   - Ranges are over the resolved bands of the bins 13.5-15.1 (N >= 127). The two top bins (N = 21 and 6) are noisier
     (for example, Q_X is x52 at 1-2R in 15.3-15.5).
   - **Q_V** (SPH-volume denominator) reads high because the summed SPH volumes fill only 97% of the geometric volume
     inside 0.5R, 91% at 1.75R and 78% at 3.6R.
     - It is +3-5% inside 1R, +6-7% at 1-2R, +13-14% at 2-3R and +18-25% at 3-4R, nearly the same at every mass from 10^13.5.
     - Below 10^13.5 it grows as the shells approach the smoothing length (0.5-1R: +6% at 13.3-13.5, +33% at 12.5-12.7).
     - The plan's 3-6% (inside 1R) and 9-13% (at 2R) are confirmed. Through the fits it gives C_ell^yy +6 to +8%, and
       electron Q_V gives DM +4%.
   - **Q_M** is +13-38% at 0.5-1R, x1.6-2.6 at 1-2R and x7-20 at 2-3R: dense, pressured clumps dominate the mass weighting.
   - **Q_X** (ROSAT, X-ray phase) is +17-36% at 0.1-0.5R, x1.3-2.4 at 0.5-1R, x3-10 at 1-2R and x30-260 at 2-3R. Q_M fits
     fail beyond 2R and Q_X fits from 2R (B12 cannot follow them); at 2R Q_M raises C_ell^yy by +27 to +80%.
   - **n_e kT_X** is unbiased at 0.1-0.5R (within 1%) and within -2 to +5% at 0.5-1R. It rises to +10-42% at 2-3R because
     T_X > T_M (T_X/T_M 1.24 at 2.5R).
   - Mass- and X-ray-weighted densities are clumping-biased and are not used for DM: the mass-weighted gas density
     is x7-10 at 0.1-0.5R and x300-390 at 1-2R, and the X-ray-weighted n_X is x2.3-14 at 1-2R.
7. **B5 comparison grid (samples x cut-offs, GLS reference).**
   - Selection mainly reshapes the profile; it changes the integrals much less. At 2R:
     - CLEAN changes C_ell^yy by -0.2 to +3.3% and mean DM by -2.2%.
     - CONT-raw changes C_ell^yy by -4.5 to -10.4% (lower inner amplitude: P0 -22%) while mean y rises +4.6% and mean DM
       +5.0% (more 1-4R gas).
   - The subtraction methods (bound-gas removal, kappa = 1 excision, templates) do not recover CLEAN. The
     keep-own-satellites flavours equal CONT-raw on the common halo-shell set. Only 95 of the 1,979 overlappers are not
     own satellites (88 neighbouring centrals and 7 of their satellites).
   - Agreement with Test A: the CLEAN vs CONT inner difference is selection plus mass normalisation (merging hosts), not
     contamination.
     - Here it shows as a large change of amplitude and shape (CONT vs ALL: P0 -22%, xc +4%; Test A: inner B12
       amplitude -17 to -27%), while Y(<2R) moves only +3.9% (CONT) or -1.9% (CLEAN).
     - No subtraction method recovers CLEAN, in Test A or here, because the inner deficit is not made by the
       overlappers' gas (their real contamination sits at 1-2R, next to them).
     - So the CLEAN - ALL differences (C_ell^yy up to +3.3%, mean DM -2.2%) are a selection effect, not a correction
       to apply. ALL, the population that is painted, is the right calibration sample.
   - A2-i and A2-ii change nothing at <= 2R (observables <= 0.12%, Y(<2R) <= 0.2%). At 4R they raise C_ell^yy by up to
     5% and lower mean DM by up to 3.6%, because the removed neighbour gas sits at 2-4R.
   - The 4^3 sub-volume jackknife gives errors 1.16-1.26x the halo bootstrap for ALL pressure parameters (1.32-1.41x for
     CONT) and 1.03-1.11x for electrons. Per shell the inflation is about 1.0 inside 2R and 1.11 at 2-4R (shared gas
     between neighbours).
8. **AGN row.**
   - Removing recently AGN-heated gas changes nothing resolved: parameters <= 0.08%, Y(<2R) <= 0.01%, C_ell, DM and kSZ
     0.00%. This agrees with the other tests (every observable <= 0.04% in F, <= 0.26% for D's X-ray recipe;
     `../SYNTHESIS.md`, 2.7).
   - Only the unresolved cores (< 0.06 R200c) change, by -0.5% (14.9-15.1) to -42% (13.5-13.7) in pressure and up to -4% in
     electrons.

![Lee22 Fig. 1, pressure](figures/b1_lee22_fig1_pressure.png)

## Verification

The cluster and local checks are recorded in `verification.json`; all 18 passed.

| check | result | number |
|---|---|---|
| the 47 inputs used on maas equal the local files (SHA256) | passed | 47 files |
| fit-grid and 36-shell profiles agree on common shells | passed | max rel. diff 0.0e+00 |
| ALL stack = weighted CLEAN + CONT stacks | passed | 2.1e-15 |
| Q_V(y) / geometric x sum V_i / V_geo = 1 | passed | 7.8e-16 |
| guarded templates = library (where finite) | passed | max \|diff\| 0 |
| quick-look correlation matrices reproduced (radii234 rows) | passed | max \|diff\| 7.3e-14 |
| quick-look objective-(a) fits reproduced (14.5-14.9) | passed | max \|dln theta\| 3.0e-15 |
| sigma_d solutions give chi2/nu = 1 | passed | 2432 fits, max \|chi2/nu - 1\| 4.3e-10 |
| all geometric-estimator full-sample fits accepted | passed | 3234/3234 |
| replicate acceptance >= 0.9 in >= 99% of cells | passed | 1 of 2695 cells below 0.9 (CONT 15.3-15.5 trend target at 1R: 0.74) |
| fast replicate solver = library objectives | passed | 28083 refits, max 9.7e-05, median 1e-08 |
| engine reproduces the kernels' P4/E4 observables | passed | max rel. 2.2e-5 (106 multipoles) |
| every geometric fit set propagated | passed | Q_M beyond 2R, Q_X from 2R not fittable (see B3) |
| linearised vs exact engine changes | passed | 553 comparisons, median rel. diff 0.09 |
| permutation null false-positive rate (random halves) | passed | max 0.10 at alpha = 0.05 (40 trials) |
| B2 guard: degenerate samples recorded, never raised | passed | 32 undetermined cases, 20 rank-deficient |
| jackknife: no failed block or fit, all occupied blocks used | passed | 792 rows, 0 failures |
| cluster run: every stage ok | passed | 9 stages ok; rerun return code 0 (monitor receipt) |

**The b2 fix.** The first cluster run stopped in B2 with `LinAlgError: Eigenvalues did not converge` in `bin_task`.
- Cause: CLEAN 15.3-15.5 has N = 2. After the bin mean and the log M slope are removed, every residual is zero, so the
  per-shell standardisation divides by zero and the correlation matrix is all NaN.
- The guards:
  - samples with N < 4 or non-finite residuals are recorded as `undetermined`, with the reason;
  - N < p + 2 is computed but flagged `ok_rank_deficient`;
  - matrices are symmetrised before `eigvalsh`, and every decomposition is guarded;
  - a failed permutation is recorded, and p-values use the valid permutations only;
  - a built-in 2-halo self-test is stored in `b2_stability.json`.
- Undetermined cases:
  - the 15.3-15.5 CLEAN bin (pressure, electrons) and its pair with 15.1-15.3;
  - the bins and pairs below 13.1 Msun that have no shell resolved for every halo on the Lee22 grid.
- The rank-deficient bins (15.1-15.3, N = 21; 15.3-15.5, N = 6) are computed and flagged.

The jackknife and engine stages gained the same kind of guards (failed blocks/fits and batches are recorded, never
raised). None triggered.

**Upstream changes and checks.**
- `geometry.make_G` was patched upstream (2026-10-12 22:12) to return 0, not NaN, for zero-amplitude templates. The
  cluster used the patched file, and Test B's guarded template routine equals it.
- The analysis-library tests and `test_newdata` on the real products ran upstream and were not re-run here.

## B1: Lee22 Fig. 1 on the exact grid

Lee22's 27 log shells over 0.04-1.34 R200c are an exact sub-grid of the stored X200 edges. Only shells resolved for every
halo are used: inner edge >= max(0.04, 4 softening/R), >= 20 particles and >= 2 smoothing lengths. The correlations are
of raw per-halo ln F (inclusion-weighted; mass-residualised versions are saved).

Lee22's bins are 10^13.2-13.4 and 10^13.8-14.0 h^-1 Msun, converted with h = 0.681 to 13.367-13.567 and 13.967-14.167 Msun.

| field | Lee22 bin [h^-1 Msun] | sample | N | shells | first x | adjacent rho | mean rho x<0.5 | mean rho x>0.5 | mean rho across | fitted x_b | decorr. radius | (a, L) RMS | two-block RMS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure | 10^13.2-13.4 h^-1 | ALL | 321 | 9 | 0.416 | 0.89 | 0.84 | 0.79 | +0.61 | 0.54 | - | 0.04 | 0.13 |
| pressure | 10^13.2-13.4 h^-1 | CLEAN | 193 | 9 | 0.416 | 0.89 | 0.85 | 0.75 | +0.63 | 1.03 | - | 0.06 | 0.13 |
| pressure | 10^13.2-13.4 h^-1 | CONT | 128 | 9 | 0.416 | 0.90 | 0.83 | 0.83 | +0.64 | 0.54 | - | 0.05 | 0.10 |
| pressure | 10^13.8-14.0 h^-1 | ALL | 325 | 17 | 0.147 | 0.94 | 0.82 | 0.54 | -0.22 | 0.61 | 0.61 | 0.33 | 0.21 |
| pressure | 10^13.8-14.0 h^-1 | CLEAN | 201 | 17 | 0.147 | 0.93 | 0.80 | 0.53 | -0.27 | 0.61 | 0.60 | 0.34 | 0.21 |
| pressure | 10^13.8-14.0 h^-1 | CONT | 124 | 17 | 0.147 | 0.95 | 0.83 | 0.51 | -0.05 | 0.61 | 0.65 | 0.30 | 0.18 |
| electrons | 10^13.2-13.4 h^-1 | ALL | 321 | 9 | 0.416 | 0.90 | 0.85 | 0.84 | +0.75 | 0.54 | - | 0.04 | 0.08 |
| electrons | 10^13.2-13.4 h^-1 | CLEAN | 193 | 9 | 0.416 | 0.91 | 0.87 | 0.85 | +0.78 | 1.03 | - | 0.04 | 0.07 |
| electrons | 10^13.2-13.4 h^-1 | CONT | 128 | 9 | 0.416 | 0.88 | 0.80 | 0.83 | +0.69 | 0.54 | - | 0.05 | 0.09 |
| electrons | 10^13.8-14.0 h^-1 | ALL | 325 | 17 | 0.147 | 0.94 | 0.91 | 0.46 | +0.02 | 0.70 | 0.76 | 0.35 | 0.19 |
| electrons | 10^13.8-14.0 h^-1 | CLEAN | 201 | 17 | 0.147 | 0.94 | 0.92 | 0.45 | -0.01 | 0.70 | 0.76 | 0.36 | 0.20 |
| electrons | 10^13.8-14.0 h^-1 | CONT | 124 | 17 | 0.147 | 0.95 | 0.90 | 0.47 | +0.14 | 0.70 | 0.82 | 0.32 | 0.16 |

![Lee22 Fig. 1, electrons](figures/b1_lee22_fig1_electrons.png)

![All bins, pressure](figures/b1_bins_lee22grid_pressure.png)

**Sampling covariance of ln(mean F)** (1,000 stratified bootstraps; same haloes and shells to 1.34R). This is what a GLS
fit inverts; Lee22's grid is near-singular:

| field | Lee22 bin | grid | shells | first x | adjacent rho (median / max) | min eigenvalue | condition number |
|---|---|---|---|---|---|---|---|
| pressure | 10^13.2-13.4 | lee22 | 9 | 0.416 | 0.897 / 0.965 | 0.0192 | 353 |
| pressure | 10^13.2-13.4 | fit | 7 | 0.442 | 0.921 / 0.964 | 0.0150 | 367 |
| pressure | 10^13.8-14.0 | lee22 | 17 | 0.147 | 0.935 / 0.966 | 0.0062 | 1512 |
| pressure | 10^13.8-14.0 | fit | 13 | 0.170 | 0.919 / 0.967 | 0.0092 | 771 |
| electrons | 10^13.2-13.4 | lee22 | 9 | 0.416 | 0.904 / 0.942 | 0.0325 | 225 |
| electrons | 10^13.2-13.4 | fit | 7 | 0.442 | 0.929 / 0.945 | 0.0284 | 206 |
| electrons | 10^13.8-14.0 | lee22 | 17 | 0.147 | 0.943 / 0.977 | 0.0094 | 1099 |
| electrons | 10^13.8-14.0 | fit | 13 | 0.170 | 0.942 / 0.982 | 0.0103 | 752 |

**Comparison with the quick look (36-shell grid).**
- The quick-look matrices are reproduced to 7e-14 for the radii234 rows.
- On the new sample the structure is the same on both grids: the decorrelation radius falls from 1.15 R200c
  (13.5-13.7) to 0.4 R200c (15.1-15.5) on both grids, and the two-block break agrees within one shell.
- The quick look's "two blocks, weakly anti-correlated, structure moves inward with mass" is confirmed on Lee22's grid.

![Structure versus mass](figures/b1_structure_vs_mass.png)

## B2: stability between adjacent bins

**Method.**
- Residualised logs: per bin, ln F on the shells strictly resolved in both bins, minus the bin mean and a linear log10 M
  trend, standardised per shell.
- Permutation null: pool the two bins, reassign labels at the original N_a and N_b, 1,000 times.
- Metrics:
  - the Frobenius RMS of R_a - R_b;
  - |Delta a| and |Delta ln L| of the 2-parameter model;
  - the largest principal angle between the top-2 eigen-subspaces;
  - the two-block model's |Delta ln x_b|, |Delta r_cross|, |Delta a_inner| and |Delta a_outer|.
- Min-p combinations are calibrated on the same permutations.
- False-positive control: random halves of one bin reject at <= 10% at alpha = 0.05 over 40 trials, consistent with 5%.

| grid | sample | field | pairs tested | p < 0.05 | rank-deficient | undetermined | pairs that differ (min-p over all metrics) |
|---|---|---|---|---|---|---|---|
| lee22 grid | ALL | pressure | 11 | 3 | 2 | 3 | 13.1-13.3\|13.3-13.5; 13.3-13.5\|13.5-13.7; 13.7-13.9\|13.9-14.1 |
| lee22 grid | ALL | electrons | 11 | 9 | 2 | 3 | 13.1-13.3\|13.3-13.5; 13.3-13.5\|13.5-13.7; 13.5-13.7\|13.7-13.9; 13.7-13.9\|13.9-14.1; 13.9-14.1\|14.1-14.3; 14.1-14.3\|14.3-14.5; 14.3-14.5\|14.5-14.7; 14.5-14.7\|14.7-14.9; 15.1-15.3\|15.3-15.5 |
| lee22 grid | CLEAN | pressure | 10 | 2 | 1 | 4 | 13.5-13.7\|13.7-13.9; 13.7-13.9\|13.9-14.1 |
| lee22 grid | CLEAN | electrons | 10 | 9 | 1 | 4 | 13.1-13.3\|13.3-13.5; 13.3-13.5\|13.5-13.7; 13.5-13.7\|13.7-13.9; 13.7-13.9\|13.9-14.1; 13.9-14.1\|14.1-14.3; 14.1-14.3\|14.3-14.5; 14.3-14.5\|14.5-14.7; 14.5-14.7\|14.7-14.9; 14.7-14.9\|14.9-15.1 |
| fit grid | ALL | pressure | 13 | 3 | 2 | 1 | 12.9-13.1\|13.1-13.3; 13.7-13.9\|13.9-14.1; 14.5-14.7\|14.7-14.9 |
| fit grid | ALL | electrons | 13 | 8 | 2 | 1 | 13.1-13.3\|13.3-13.5; 13.3-13.5\|13.5-13.7; 13.7-13.9\|13.9-14.1; 13.9-14.1\|14.1-14.3; 14.1-14.3\|14.3-14.5; 14.3-14.5\|14.5-14.7; 14.5-14.7\|14.7-14.9; 15.1-15.3\|15.3-15.5 |

Lee22 grid, ALL (rank-deficient pairs have fewer haloes than shells; the test stays valid but has little power):

| field | adjacent bins | N_a/N_b | shells | Frobenius RMS (null 95%) | max angle deg (null 95%) | p min-p (library) | p min-p (+ two-block) | rank-deficient |
|---|---|---|---|---|---|---|---|---|
| pressure | 13.1-13.3 \| 13.3-13.5 | 317/308 | 4 | 0.021 (0.036) | 10.2 (10.5) | 0.005 | 0.005 |  |
| pressure | 13.3-13.5 \| 13.5-13.7 | 308/326 | 7 | 0.136 (0.074) | 4.7 (5.8) | 0.001 | 0.001 |  |
| pressure | 13.5-13.7 \| 13.7-13.9 | 326/324 | 11 | 0.092 (0.101) | 5.8 (5.2) | 0.045 | 0.068 |  |
| pressure | 13.7-13.9 \| 13.9-14.1 | 324/319 | 13 | 0.302 (0.114) | 11.4 (6.4) | 0.001 | 0.001 |  |
| pressure | 13.9-14.1 \| 14.1-14.3 | 319/327 | 16 | 0.087 (0.105) | 10.1 (14.7) | 0.190 | 0.206 |  |
| pressure | 14.1-14.3 \| 14.3-14.5 | 327/317 | 17 | 0.075 (0.099) | 17.3 (24.5) | 0.363 | 0.482 |  |
| pressure | 14.3-14.5 \| 14.5-14.7 | 317/895 | 20 | 0.065 (0.077) | 12.4 (28.3) | 0.319 | 0.125 |  |
| pressure | 14.5-14.7 \| 14.7-14.9 | 895/326 | 20 | 0.050 (0.071) | 18.5 (26.1) | 0.057 | 0.081 |  |
| pressure | 14.7-14.9 \| 14.9-15.1 | 326/127 | 24 | 0.052 (0.118) | 4.4 (26.5) | 0.707 | 0.547 |  |
| pressure | 14.9-15.1 \| 15.1-15.3 | 127/21 | 25 | 0.250 (0.283) | 10.7 (64.6) | 0.241 | 0.341 | yes |
| pressure | 15.1-15.3 \| 15.3-15.5 | 21/6 | 27 | 0.477 (0.572) | 48.4 (87.3) | 0.142 | 0.132 | yes |
| electrons | 13.1-13.3 \| 13.3-13.5 | 317/308 | 4 | 0.040 (0.037) | 19.9 (14.9) | 0.001 | 0.001 |  |
| electrons | 13.3-13.5 \| 13.5-13.7 | 308/326 | 7 | 0.117 (0.074) | 6.0 (6.4) | 0.001 | 0.001 |  |
| electrons | 13.5-13.7 \| 13.7-13.9 | 326/324 | 11 | 0.159 (0.088) | 5.5 (4.9) | 0.001 | 0.001 |  |
| electrons | 13.7-13.9 \| 13.9-14.1 | 324/319 | 13 | 0.291 (0.106) | 14.3 (5.8) | 0.001 | 0.001 |  |
| electrons | 13.9-14.1 \| 14.1-14.3 | 319/327 | 16 | 0.223 (0.102) | 11.7 (7.7) | 0.001 | 0.001 |  |
| electrons | 14.1-14.3 \| 14.3-14.5 | 327/317 | 17 | 0.187 (0.100) | 38.5 (21.2) | 0.001 | 0.001 |  |
| electrons | 14.3-14.5 \| 14.5-14.7 | 317/895 | 20 | 0.135 (0.076) | 32.2 (17.0) | 0.001 | 0.001 |  |
| electrons | 14.5-14.7 \| 14.7-14.9 | 895/326 | 20 | 0.113 (0.074) | 10.7 (10.6) | 0.004 | 0.006 |  |
| electrons | 14.7-14.9 \| 14.9-15.1 | 326/127 | 24 | 0.080 (0.126) | 6.4 (10.8) | 0.410 | 0.506 |  |
| electrons | 14.9-15.1 \| 15.1-15.3 | 127/21 | 25 | 0.310 (0.293) | 14.5 (25.2) | 0.087 | 0.147 | yes |
| electrons | 15.1-15.3 \| 15.3-15.5 | 21/6 | 27 | 0.566 (0.536) | 53.1 (73.9) | 0.048 | 0.049 | yes |

**Fitting a bin with its neighbour's correlation** (GLS, own variances, raw bootstrap correlation of bin i-1 or i+1):
- The parameters move by a median of 0.1-0.25 sigma per cut-off. The largest shifts are 0.9 sigma at 2R (pressure xc,
  13.9-14.1) and 1.4 sigma over all cut-offs (electron alpha, 14.7-14.9 with the pooled >= 14.9 correlation, cuts 1-1.5).
- Observables (cuts 2 and 4): C_ell^yy changes by <= 0.22%, mean y by <= 0.11%, mean DM by <= 0.06%, DM variance by
  <= 0.10%, the DM quantiles by <= 0.12%, kSZ by <= 0.17% and the DM-field C_ell by <= 0.29%.
- Shrinkage itself barely matters: unshrunk correlations change the observables by <= 0.05%.

![Stability](figures/b2_stability.png)

## B3: B24 weightings of the shell pressure (and T, n_e)

All ratios are to sum(y)/V_geo (or n_e V/V_geo), computed on the same halo-shells. They are ratios of volume-integrated
stacks per radial band (ALL, AGN gas removed); the electron density and the gas Q_M are ratios to their own geometric
estimators:

| estimator / geometric | bin | 0.1-0.5 | 0.5-1 | 1-2 | 2-3 | 3-4 R200c |
|---|---|---|---|---|---|---|
| Q_V | 13.9-14.1 | (1.032) | 1.038 | 1.068 | 1.136 | 1.210 |
| Q_V | 14.5-14.7 | 1.026 | 1.038 | 1.066 | 1.135 | 1.232 |
| Q_V | 14.9-15.1 | 1.028 | 1.039 | 1.065 | 1.138 | 1.250 |
| Q_M | 13.9-14.1 | (1.052) | 1.150 | 1.992 | 8.659 | 30.590 |
| Q_M | 14.5-14.7 | 1.096 | 1.260 | 2.046 | 11.709 | 39.500 |
| Q_M | 14.9-15.1 | 1.163 | 1.379 | 2.225 | 19.527 | 39.508 |
| Q_X | 13.9-14.1 | (1.114) | 1.437 | 5.931 | 31.216 | 113.972 |
| Q_X | 14.5-14.7 | 1.211 | 1.960 | 7.134 | 255.356 | 429.089 |
| Q_X | 14.9-15.1 | 1.363 | 2.239 | 9.661 | 169.159 | 400.924 |
| n_e kT_X | 13.9-14.1 | (1.000) | 1.028 | 1.122 | 1.360 | 1.589 |
| n_e kT_X | 14.5-14.7 | 0.999 | 1.006 | 1.027 | 1.186 | 1.481 |
| n_e kT_X | 14.9-15.1 | 0.990 | 0.978 | 0.938 | 1.103 | 1.448 |
| n_e Q_V | 13.9-14.1 | (1.032) | 1.039 | 1.071 | 1.143 | 1.226 |
| n_e Q_V | 14.5-14.7 | 1.027 | 1.039 | 1.070 | 1.141 | 1.244 |
| n_e Q_V | 14.9-15.1 | 1.028 | 1.039 | 1.069 | 1.147 | 1.266 |
| n_X = Q_X/kT_X (clumping-biased) | 13.9-14.1 | (1.115) | 1.408 | 4.320 | 19.674 | 61.388 |
| n_X = Q_X/kT_X (clumping-biased) | 14.5-14.7 | 1.223 | 2.071 | 7.277 | 116.022 | 263.802 |
| n_X = Q_X/kT_X (clumping-biased) | 14.9-15.1 | 1.389 | 2.355 | 13.572 | 152.223 | 254.437 |
| gas Q_M (clumping-biased) | 13.9-14.1 | (11.891) | 28.780 | 342.291 | 3281.190 | 9856.970 |
| gas Q_M (clumping-biased) | 14.5-14.7 | 6.559 | 19.756 | 343.458 | 3573.490 | 10326.100 |
| gas Q_M (clumping-biased) | 14.9-15.1 | 9.771 | 20.863 | 389.486 | 3635.440 | 10814.300 |

Values in parentheses are bands not resolved for every halo of the bin (the strict mask); they are not used in the text.

- **Q_V.** The ratio equals V_geo / sum V_i exactly (identity checked to 8e-16). The SPH-volume filling factor is
  0.97 at 0.1-0.5R, 0.96 at 0.84R, 0.91 at 1.75R, 0.86 at 2.5R and 0.78 at 3.6R (14.5-14.7). This is the whole Q_V bias.
- **Q_M, Q_X.** They are dominated by dense, X-ray-bright clumps:
  - pressure clumping <Pe^2>_V/<Pe>_V^2 is 1.04 at 0.1R, 1.43 at 0.84R and 2.8 at 1.75R;
  - B12 cannot fit Q_M beyond 2R or Q_X from 2R (the fits hit their bounds; these engine sets are omitted).
- **n_e kT_X.** It inherits T_X/T_M (1.01 inside 0.8R, 1.06 at 1.75R, 1.24 at 2.5R, 1.6 at 3.6R).
- **Caveat on L_X.** The particle ROSAT L_X summed inside R500c is +0.3% above SOAP's recomputed L_X in the median, with
  148 haloes > 5% (`../verification.json`, G7_assessment). The particle field is used, so Q_X and T_X carry this
  definitional caveat.

Observables of the weighting fits (fit to 2R, vs the geometric GLS fit, %):

| weighting | C_ell^yy ell=100 | ell=1000 | ell=3000 | mean y | mean DM | DM variance |
|---|---|---|---|---|---|---|
| Q_V | +7.88 [+7.74, +7.89] | +6.96 [+6.81, +6.97] | +6.32 [+6.14, +6.30] | +3.89 [+3.75, +3.87] | - | - |
| Q_V (y form) | +7.88 [+6.76, +9.15] | +6.96 [+5.56, +8.49] | +6.32 [+5.37, +7.37] | +3.89 [+3.68, +4.43] | - | - |
| Q_M | +80.36 [+78.96, +82.88] | +54.38 [+51.47, +54.53] | +27.16 [+25.29, +27.21] | +39.54 [+39.07, +40.51] | - | - |
| n_e kT_X | -2.80 [-3.05, -2.13] | +0.10 [-0.22, +0.81] | +0.78 [+0.58, +1.21] | +4.50 [+4.37, +4.59] | - | - |
| n_e Q_V (electrons) | - | - | - | - | +4.28 [+4.25, +4.30] | +7.11 [+7.09, +7.14] |

![Pressure weightings](figures/b3_pressure_weighting_bias.png)

![Density and temperature weightings](figures/b3_density_temperature_weightings.png)

![Bias versus mass](figures/b3_bias_vs_mass.png)

![Scatter by estimator](figures/b3_scatter_by_estimator.png)

## B4: covariance-weighted fit

**Objectives.**
- (a) the current diagonal fit, sigma = max(SD/mean, 5%);
- (b) a diagonal fit with the bootstrap error on ln mean;
- (c) GLS with:
  - the 1,000-draw bootstrap Cov[ln mean];
  - shrinkage toward the neighbour-pooled correlation (Fisher-z average, Higham projection), with lambda from 50
    resamples (median 0.21 for pressure);
  - the pooled correlation alone (lambda = 1) only if N_eff < 3p, which no ALL bin needed;
  - sigma_d solved for chi2/nu = 1.
- Hartlap factors are 0.80-0.98. They are reported only, because they cannot move the point estimates.

Medians [range] over the 8 fit units:

| field | cut | 100 dln par1 (c - a) | 100 dln par2 | 100 dln beta | sigma_d [%] (c) | chi2/nu at sigma_d = 0 | N_eff shells (a / c) |
|---|---|---|---|---|---|---|---|
| pressure | 1 | -1.1 [-6.4, +6.5] | +2.8 [-9.4, +7.8] | +1.9 [-6.0, +4.5] | 0.08 [0.00, 0.32] | 1.22 [0.67, 1.90] | 9.1 / 7.9 |
| pressure | 1.34 | +0.7 [-7.5, +5.5] | +0.5 [-6.2, +9.3] | +0.7 [-3.5, +5.5] | 0.16 [0.00, 0.42] | 1.30 [0.76, 3.00] | 10.8 / 8.5 |
| pressure | 1.5 | +1.8 [-6.0, +6.8] | -1.4 [-8.7, +6.2] | -0.6 [-5.4, +3.4] | 0.17 [0.00, 0.46] | 1.32 [0.69, 3.08] | 11.3 / 9.7 |
| pressure | 1.75 | +2.7 [-4.4, +7.3] | -2.8 [-9.5, +4.0] | -1.5 [-5.8, +1.9] | 0.20 [0.00, 0.58] | 1.43 [0.73, 2.91] | 11.7 / 9.7 |
| pressure | 2 | +4.2 [-5.2, +8.3] | -5.5 [-10.8, +5.9] | -3.5 [-6.6, +3.3] | 0.29 [0.00, 0.69] | 1.58 [0.89, 3.37] | 11.9 / 10.1 |
| pressure | 3 | +11.1 [+0.3, +15.3] | -13.8 [-18.2, -2.8] | -7.5 [-9.9, -2.1] | 3.48 [1.90, 5.83] | 3.92 [2.66, 11.49] | 12.9 / 15.2 |
| pressure | 4 | +37.5 [+11.4, +54.8] | -41.9 [-51.5, -16.7] | -19.4 [-22.0, -8.7] | 11.71 [8.42, 14.29] | 6.46 [3.88, 19.11] | 13.1 / 17.7 |
| electrons | 1 | +4.5 [-1.5, +9.4] | -4.5 [-7.3, +1.2] | +0.2 [+0.1, +0.5] | 0.62 [0.15, 0.86] | 4.78 [1.47, 8.86] | 8.3 / 6.5 |
| electrons | 1.34 | +6.2 [-1.4, +10.5] | -4.4 [-9.1, +1.3] | +0.8 [+0.2, +1.5] | 1.18 [0.11, 1.82] | 6.39 [1.29, 20.36] | 11.1 / 8.2 |
| electrons | 1.5 | +4.9 [-1.2, +10.3] | -4.2 [-7.8, +1.0] | +1.0 [+0.1, +1.6] | 1.39 [0.09, 2.36] | 6.25 [1.20, 29.84] | 11.5 / 8.9 |
| electrons | 1.75 | +4.0 [-1.1, +9.8] | -3.0 [-7.2, +0.5] | +0.6 [-0.4, +1.9] | 1.48 [0.34, 3.58] | 6.15 [2.72, 42.99] | 12.1 / 9.2 |
| electrons | 2 | +2.0 [-2.8, +9.5] | -1.5 [-6.4, +2.4] | +0.4 [-0.8, +2.1] | 1.60 [0.65, 5.01] | 8.27 [3.31, 47.04] | 12.4 / 9.7 |
| electrons | 3 | +5.8 [-2.7, +9.4] | -4.7 [-6.7, +2.1] | +0.3 [-1.9, +2.2] | 6.06 [2.90, 8.00] | 17.15 [12.17, 45.08] | 12.9 / 13.0 |
| electrons | 4 | +13.0 [-7.5, +31.2] | -10.6 [-18.6, +8.4] | -0.8 [-2.5, +0.8] | 11.29 [6.38, 16.28] | 36.29 [28.64, 114.94] | 13.3 / 15.4 |

Observables, (c) - (a) at the same cut (%):

|  | C_ell^yy ell=100 | ell=1000 | ell=3000 | mean y |
|---|---|---|---|---|
| (c) - (a), cut 1 | -2.10 [-2.15, -0.76] | -1.08 [-1.30, +0.12] | -0.79 [-1.07, -0.15] | -0.03 [-0.04, +0.31] |
| (c) - (a), cut 1.34 | -1.54 [-1.59, -0.16] | -1.29 [-1.46, +0.64] | -0.70 [-0.93, +0.35] | -0.06 [-0.17, +0.13] |
| (c) - (a), cut 1.5 | -0.95 [-1.11, +0.11] | -1.10 [-1.31, +0.66] | -0.58 [-0.85, +0.38] | +0.16 [-0.04, +0.23] |
| (c) - (a), cut 1.75 | -0.09 [-0.40, +0.52] | -0.39 [-0.63, +0.50] | -0.30 [-0.55, +0.28] | +0.39 [+0.14, +0.41] |
| (c) - (a), cut 2 | +0.14 [-0.23, +0.58] | -0.08 [-0.45, +0.49] | -0.16 [-0.54, +0.24] | +0.62 [+0.36, +0.61] |
| (c) - (a), cut 3 | -0.35 [-0.96, +0.11] | -1.28 [-1.74, -0.67] | -1.64 [-1.99, -1.18] | +0.45 [+0.33, +0.51] |
| (c) - (a), cut 4 | -3.32 [-4.40, -2.79] | -5.46 [-6.11, -4.74] | -5.82 [-6.22, -5.28] | -0.40 [-0.53, -0.34] |

|  | mean DM | median DM | DM variance | DM Q99.9 | kSZ 1h ell=3000 |
|---|---|---|---|---|---|
| (c) - (a), cut 1 | +0.81 [+0.74, +0.93] | +0.95 [+0.86, +1.10] | +0.87 [+0.79, +0.95] | +0.61 [+0.51, +0.61] | +0.45 [+0.35, +0.49] |
| (c) - (a), cut 1.34 | -0.07 [-0.10, -0.02] | -0.07 [-0.11, -0.01] | +0.11 [+0.04, +0.16] | +0.53 [+0.43, +0.53] | +0.05 [-0.02, +0.07] |
| (c) - (a), cut 1.5 | -0.44 [-0.48, -0.39] | -0.52 [-0.56, -0.45] | -0.17 [-0.27, -0.13] | +0.50 [+0.38, +0.50] | -0.17 [-0.25, -0.15] |
| (c) - (a), cut 1.75 | -0.70 [-0.76, -0.64] | -0.82 [-0.90, -0.75] | -0.41 [-0.52, -0.36] | +0.36 [+0.24, +0.39] | -0.43 [-0.52, -0.40] |
| (c) - (a), cut 2 | -1.00 [-1.08, -0.93] | -1.19 [-1.29, -1.10] | -0.74 [-0.85, -0.68] | +0.10 [-0.00, +0.13] | -0.82 [-0.90, -0.78] |
| (c) - (a), cut 3 | -1.33 [-1.44, -1.23] | -1.53 [-1.67, -1.40] | -1.62 [-1.70, -1.51] | -0.92 [-0.98, -0.88] | -2.06 [-2.16, -1.96] |
| (c) - (a), cut 4 | -0.40 [-0.53, -0.27] | -0.07 [-0.24, +0.10] | -3.02 [-3.11, -2.90] | -2.46 [-2.54, -2.38] | -3.83 [-3.95, -3.70] |

**Alternative covariance estimates** (median |Delta ln theta| vs (c)):
- cross-fitting (covariance from one half, mean from the other; 10 splits): 0.6-0.7% at 2R and 5.5% at 4R (pressure);
- 2- and 3-factor covariance models: 0.9-1.6% (2R, pressure) and <= 0.2% (electrons);
- with the factor models, C_ell^yy changes by -0.6 to -2.5% at ell = 100-3000 (2R, pressure; largest at ell = 300) and
  by <= 0.7% at 4R.

**Comparison with Test F** (`../F_observables/REPORT.md`, section 3). Test B's exact engine changes (c) - (a) are shown
next to Test F's own (c) - (a) and F's errors against the measured truth (R_cut = 4R, deterministic prediction, from
`F_observables/results/tables/F_errors_long.csv`; %):

| observable | fit 2R: B (c)-(a) | F (c)-(a) | F error vs truth (a) -> (c) | fit 4R: B (c)-(a) | F (c)-(a) | F error vs truth (a) -> (c) |
|---|---|---|---|---|---|---|
| C_ell^yy ell=100 | +0.14 | -1.25 | -3.98 -> -5.18 | -3.32 | -3.35 | -0.98 -> -4.30 |
| C_ell^yy ell=300 | -0.10 | -1.34 | -5.12 -> -6.39 | -3.87 | -3.43 | -2.74 -> -6.07 |
| C_ell^yy ell=1000 | -0.08 | -0.63 | -3.50 -> -4.11 | -5.46 | -5.06 | -1.86 -> -6.83 |
| C_ell^yy ell=3000 | -0.16 | -0.49 | -3.62 -> -4.08 | -5.82 | -5.70 | -2.77 -> -8.31 |
| mean y | +0.62 | +0.52 | -3.36 -> -2.86 | -0.40 | -0.35 | -0.45 -> -0.80 |
| kSZ 1h ell=1000 | -1.76 | -1.72 | +1.95 -> +0.20 | -3.26 | -3.15 | +0.83 -> -2.34 |
| kSZ 1h ell=3000 | -0.82 | -0.80 | -0.47 -> -1.27 | -3.83 | -3.80 | -1.78 -> -5.51 |
| mean DM | -1.00 | -0.99 | +1.57 -> +0.57 | -0.40 | -0.36 | +0.55 -> +0.19 |
| DM variance | -0.74 | -0.74 | -2.29 -> -3.01 | -3.02 | -2.90 | -1.54 -> -4.40 |
| DM Q99.9 | +0.10 | +0.13 | -2.68 -> -2.55 | -2.46 | -2.38 | -2.47 -> -4.80 |

- Test B reproduces F's GLS shifts at 4R and, for electrons, at 2R.
- At 2R, F's GLS lowers C_ell^yy by 0.5-1.3%, while Test B's changes it by <= 0.16%. The likely cause is the top two
  bins: F fits them individually (N = 21 and 6, pooled correlation forced), and Test B fits them through the pooled
  >= 14.9 trend. This was not tested.
- In F the diagonal fit is already low in C_ell^yy (and in kSZ at ell = 100 and 3000), so the downward GLS shifts
  increase those errors. This is why F finds GLS the worst weighting for the observables.
- At 2R GLS moves mean DM toward the truth (F's DM family error 2.2 -> 1.8%). At 4R every family gets worse.

**Influence** (14.5-14.7). The rows H = (J^T C^-1 J)^-1 J^T C^-1 show which radii drive each parameter.
- For the diagonal fits, xc and beta peak at 0.5-0.6 R200c, and P0 is anti-weighted there.
- GLS flattens the rows and puts weight on the least redundant shells: the innermost resolved shells and the outermost
  fitted shell.

![Influence bands](figures/b4_influence_bands.png)

![Influence rows](figures/b4_influence_pressure.png)

![Parameters vs cut, pressure](figures/b4_parameters_vs_cut_pressure.png)

![GLS - diagonal](figures/b4_gls_minus_diagonal.png)

![sigma_d and chi2](figures/b4_sigma_d_chi2.png)

## B5: comparison grid (samples x cut-offs, GLS reference)

Cut-off effect relative to the 2R fit (ALL, GLS, %):

|  | C_ell^yy ell=100 | ell=1000 | ell=3000 | mean y |
|---|---|---|---|---|
| cut 1 vs 2 | -0.04 [-0.69, +1.47] | -0.62 [-0.81, +0.28] | -0.47 [-0.65, +0.07] | +0.47 [+0.23, +1.24] |
| cut 1.34 vs 2 | -1.43 [-1.65, -0.17] | -1.21 [-1.26, +0.25] | -0.55 [-0.60, +0.30] | -0.55 [-0.66, -0.12] |
| cut 1.5 vs 2 | -1.22 [-1.34, -0.29] | -1.13 [-1.12, +0.21] | -0.46 [-0.51, +0.29] | -0.58 [-0.70, -0.36] |
| cut 1.75 vs 2 | -0.38 [-0.50, -0.10] | -0.42 [-0.40, +0.01] | -0.18 [-0.18, +0.11] | -0.40 [-0.48, -0.30] |
| cut 3 vs 2 | +0.50 [-0.05, +0.94] | -0.58 [-0.97, -0.14] | -1.17 [-1.47, -0.74] | +1.04 [+1.03, +1.31] |
| cut 4 vs 2 | -0.63 [-1.71, -0.09] | -3.84 [-4.56, -3.08] | -4.80 [-5.20, -4.21] | +1.97 [+1.96, +2.39] |

|  | mean DM | median DM | DM variance | DM Q99.9 | kSZ 1h ell=3000 |
|---|---|---|---|---|---|
| cut 1 vs 2 | +9.00 [+8.72, +9.36] | +10.92 [+10.58, +11.35] | +7.35 [+7.09, +7.66] | +2.38 [+2.27, +2.47] | +5.88 [+5.65, +6.12] |
| cut 1.34 vs 2 | +4.64 [+4.50, +4.81] | +5.70 [+5.53, +5.91] | +3.47 [+3.34, +3.62] | +1.35 [+1.28, +1.40] | +3.10 [+2.98, +3.21] |
| cut 1.5 vs 2 | +2.97 [+2.87, +3.10] | +3.68 [+3.55, +3.83] | +2.13 [+2.03, +2.21] | +0.96 [+0.90, +0.99] | +2.04 [+1.96, +2.11] |
| cut 1.75 vs 2 | +1.46 [+1.40, +1.52] | +1.82 [+1.74, +1.89] | +1.01 [+0.95, +1.04] | +0.52 [+0.48, +0.54] | +1.01 [+0.97, +1.05] |
| cut 3 vs 2 | -2.10 [-2.22, -1.96] | -2.67 [-2.84, -2.49] | -1.39 [-1.44, -1.27] | -1.28 [-1.30, -1.17] | -2.19 [-2.26, -2.09] |
| cut 4 vs 2 | -0.40 [-0.62, -0.18] | -0.39 [-0.69, -0.11] | -1.57 [-1.64, -1.42] | -2.37 [-2.42, -2.24] | -4.31 [-4.42, -4.16] |

Samples relative to ALL at the same cut (2R, %):

|  | C_ell^yy ell=100 | ell=1000 | ell=3000 | mean y |
|---|---|---|---|---|
| CLEAN | -0.20 [-0.86, +1.41] | +0.88 [+0.14, +3.05] | +3.34 [+2.88, +4.63] | -2.32 [-2.65, -1.96] |
| CONT | -4.53 [-5.07, -0.00] | -6.18 [-7.25, -2.48] | -10.44 [-11.22, -8.04] | +4.63 [+4.12, +5.18] |
| CONTc | -4.55 [-7.10, -2.99] | -6.29 [-9.12, -5.28] | -10.01 [-12.29, -8.91] | +5.04 [+4.38, +5.43] |
| CONT_bound_all | -10.82 [-12.91, -8.48] | -10.46 [-13.02, -9.12] | -11.36 [-13.74, -10.20] | +1.17 [+0.70, +1.72] |
| CONT_exc_all | -21.27 [-24.54, -19.15] | -21.69 [-24.07, -19.65] | -21.99 [-23.85, -20.02] | -1.96 [-2.49, -1.44] |
| CONT_tmpl_all | -10.91 [-13.17, -8.63] | -10.80 [-13.26, -9.37] | -11.80 [-13.94, -10.50] | +1.02 [+0.60, +1.60] |
| CONT_bound_keep | -4.56 [-7.10, -3.00] | -6.28 [-9.12, -5.28] | -9.98 [-12.27, -8.90] | +5.02 [+4.36, +5.41] |
| A2i | -0.02 [-0.03, -0.01] | -0.00 [-0.01, +0.00] | +0.01 [-0.00, +0.01] | -0.02 [-0.02, -0.01] |
| A2ii | -0.10 [-0.16, -0.09] | -0.02 [-0.07, -0.01] | +0.03 [-0.02, +0.03] | -0.12 [-0.16, -0.09] |
| AGNkept | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

|  | mean DM | median DM | DM variance | DM Q99.9 | kSZ 1h ell=3000 |
|---|---|---|---|---|---|
| CLEAN | -2.15 [-2.41, -1.91] | -2.77 [-3.07, -2.48] | -0.29 [-0.57, -0.03] | +1.18 [+0.97, +1.33] | +0.69 [+0.41, +0.97] |
| CONT | +4.96 [+4.43, +5.33] | +6.16 [+5.56, +6.65] | +2.05 [+1.34, +2.40] | -1.67 [-2.15, -1.39] | -0.73 [-1.42, -0.25] |
| CONTc | +4.96 [+4.42, +5.37] | +6.16 [+5.52, +6.69] | +2.13 [+1.39, +2.55] | -1.42 [-1.98, -1.01] | -0.65 [-1.38, -0.13] |
| CONT_bound_all | +1.58 [+1.18, +2.00] | +2.37 [+1.88, +2.91] | -2.25 [-2.85, -1.79] | -3.04 [-3.62, -2.61] | -2.24 [-2.89, -1.72] |
| CONT_exc_all | +0.75 [+0.33, +1.08] | +1.95 [+1.45, +2.38] | -6.56 [-7.29, -6.15] | -6.10 [-6.68, -5.74] | -6.00 [-6.74, -5.50] |
| CONT_tmpl_all | +1.43 [+1.03, +1.84] | +2.24 [+1.74, +2.76] | -2.68 [-3.28, -2.22] | -3.38 [-3.91, -2.93] | -2.59 [-3.24, -2.08] |
| CONT_bound_keep | +4.95 [+4.41, +5.35] | +6.14 [+5.50, +6.67] | +2.12 [+1.38, +2.54] | -1.40 [-1.96, -0.98] | -0.64 [-1.37, -0.12] |
| A2i | -0.02 [-0.02, -0.01] | -0.02 [-0.02, -0.02] | -0.01 [-0.01, -0.01] | +0.01 [+0.01, +0.01] | +0.00 [+0.00, +0.00] |
| A2ii | -0.08 [-0.10, -0.07] | -0.10 [-0.12, -0.09] | -0.06 [-0.07, -0.05] | +0.03 [+0.01, +0.03] | -0.01 [-0.02, -0.00] |
| AGNkept | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

Samples relative to ALL at 4R (pressure, %):

|  | C_ell^yy ell=100 | ell=1000 | ell=3000 | mean y |
|---|---|---|---|---|
| CLEAN | -0.77 [-1.39, +0.67] | +2.00 [+0.91, +3.13] | +4.74 [+3.99, +5.55] | -2.89 [-3.06, -2.38] |
| CONT | +1.48 [-0.44, +3.80] | -1.04 [-3.22, +0.95] | -6.41 [-8.17, -5.11] | +6.70 [+5.62, +6.78] |
| CONTc | -1.79 [-4.85, +0.18] | -4.02 [-6.88, -2.70] | -7.71 [-10.18, -6.77] | +6.70 [+5.61, +6.72] |
| CONT_bound_all | -9.14 [-12.60, -6.68] | -9.99 [-13.02, -8.21] | -11.47 [-13.88, -10.13] | +2.32 [+1.63, +2.65] |
| CONT_exc_all | -19.15 [-22.27, -16.31] | -19.09 [-22.11, -17.22] | -19.34 [-21.84, -17.78] | -1.38 [-1.97, -0.98] |
| CONT_tmpl_all | -9.50 [-12.97, -6.73] | -10.28 [-13.39, -8.50] | -11.85 [-14.22, -10.50] | +2.27 [+1.56, +2.59] |
| CONT_bound_keep | -1.76 [-4.81, +0.17] | -4.03 [-6.90, -2.65] | -7.72 [-10.16, -6.74] | +6.62 [+5.54, +6.64] |
| A2i | +0.88 [+0.41, +1.33] | +1.75 [+1.35, +2.01] | +1.70 [+1.48, +1.86] | -0.34 [-0.44, -0.30] |
| A2ii | +1.13 [+0.18, +1.88] | +4.35 [+3.16, +4.57] | +5.03 [+4.16, +5.08] | -0.66 [-1.25, -0.82] |
| AGNkept | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

Delta ln of the model Y(<2R) or N_e(<2R) relative to ALL (100 x, median [range] over the 8 units):

| sample | Y(<2R) P, fit 2R | Y(<2R) P, fit 4R | N_e(<2R) E, fit 2R | N_e(<2R) E, fit 4R |
|---|---|---|---|---|
| CLEAN | -1.94 [-3.14, +0.63] | -2.67 [-3.69, -0.43] | -1.39 [-1.90, -0.64] | -1.74 [-2.16, -0.27] |
| CONT-raw | +3.85 [-2.05, +5.98] | +6.70 [+1.47, +9.51] | +3.79 [+1.22, +5.09] | +3.86 [+0.61, +5.66] |
| CONT-raw, common halo-shells | +3.82 [-1.95, +6.74] | +6.77 [+0.35, +9.81] | +3.86 [+1.35, +5.26] | +3.75 [+0.87, +5.33] |
| CONT-bound, remove all | -0.40 [-5.34, +3.74] | +0.23 [-3.09, +5.63] | -0.24 [-2.73, +0.29] | -0.23 [-2.46, +0.48] |
| CONT-bound, keep own sat. | +3.81 [-1.96, +6.73] | +6.65 [+0.41, +9.81] | +3.82 [+1.35, +5.22] | +3.65 [+0.85, +5.38] |
| CONT-excised k=1, remove all | -7.76 [-13.45, -0.20] | -5.67 [-9.30, +0.70] | -3.07 [-7.35, -1.79] | -3.62 [-6.88, -0.83] |
| CONT-excised, keep own sat. | +3.81 [-2.01, +6.73] | +6.64 [+0.44, +9.81] | +3.82 [+1.35, +5.25] | +3.67 [+0.86, +5.38] |
| CONT-template, remove all | -0.70 [-5.45, +3.40] | +0.01 [-3.50, +5.25] | -0.48 [-3.36, +0.08] | -0.41 [-2.66, +0.25] |
| CONT-template, keep own sat. | +3.80 [-1.99, +6.73] | +6.66 [+0.48, +9.80] | +3.82 [+1.35, +5.24] | +3.66 [+0.80, +5.38] |
| A2-i | -0.01 [-0.03, +0.00] | +0.63 [-0.29, +0.95] | -0.02 [-0.04, -0.01] | -0.51 [-0.85, -0.31] |
| A2-ii | -0.07 [-0.20, -0.01] | +1.88 [+0.08, +2.93] | -0.08 [-0.15, -0.05] | -1.94 [-2.39, -1.47] |
| AGN-heated gas kept | +0.00 [+0.00, +0.01] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

The heat maps show every sample, weighting and objective at all seven cut-offs:
- top row: relative to ALL at the same cut-off;
- bottom row: relative to the same row at 2R;
- boxed cells are the references, and `*` marks cells beyond the colour range (Q_M and Q_X).

![yy](figures/b5_heatmap_yy.png)

![DM](figures/b5_heatmap_dm.png)

![kSZ and DM-field](figures/b5_heatmap_ksz_dmcl.png)

![y x DM](figures/b5_heatmap_ydm.png)

![Parameters](figures/b5_heatmap_parameters_pressure.png)

**Spatial jackknife.** 4^3 blocks of the 1 Gpc box, with every halo assigned by its centre. The whole GLS estimator is
recomputed with each block deleted, and the result is compared with the 500-replicate bootstrap SD. Ratios are medians
[25-75%] over the bins and parameters:

| sample | fit to 2R | fit to 3R | fit to 4R |
|---|---|---|---|
| ALL, pressure | 1.16 [1.07, 1.37] | 1.26 [1.19, 1.32] | 1.24 [1.19, 1.31] |
| CLEAN, pressure | 1.24 [1.08, 1.37] | 1.09 [1.04, 1.18] | 1.15 [1.04, 1.23] |
| CONT-raw, pressure | 1.32 [1.26, 1.38] | 1.34 [1.28, 1.40] | 1.41 [1.31, 1.47] |
| A2-ii, pressure | 1.17 [1.08, 1.32] | 1.30 [1.22, 1.39] | 1.36 [1.20, 1.53] |
| ALL, electrons | 1.11 [1.04, 1.21] | 1.06 [0.96, 1.13] | 1.03 [0.97, 1.06] |
| CLEAN, electrons | 1.07 [0.97, 1.13] | 0.93 [0.90, 0.96] | 0.93 [0.91, 1.02] |
| CONT-raw, electrons | 1.17 [1.10, 1.27] | 1.15 [1.09, 1.22] | 1.10 [1.03, 1.16] |
| A2-ii, electrons | 1.13 [1.04, 1.19] | 1.10 [1.04, 1.16] | 1.07 [0.99, 1.10] |

The jackknife SD of a 64-block jackknife is itself uncertain by about 18%. The per-shell ratios of the stacked ln mean
are 0.97 inside 1R, 1.02 at 1-2R and 1.11 (10-90%: 1.01-1.27) at 2-4R for pressure (electrons 0.95 / 1.05 / 1.12): haloes
share outskirts gas.

![Jackknife](figures/b5_jackknife.png)

## AGN-heated gas (paired control)

Kept minus removed (ALL, GLS, %):

|  | C_ell^yy ell=100 | ell=1000 | ell=3000 | mean y |
|---|---|---|---|---|
| kept - removed, cut 2 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| kept - removed, cut 4 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

|  | mean DM | median DM | DM variance | DM Q99.9 | kSZ 1h ell=3000 |
|---|---|---|---|---|---|
| kept - removed, cut 2 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| kept - removed, cut 4 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

The fitted parameters change by <= 0.08% and Y(<2R) by <= 0.01%. Only the unresolved cores change:

![AGN core ratio](figures/agn_core_ratio.png)

## Caveats

- **Fit adequacy.** B12/B16 do not describe the stack beyond about 2R. Every 3R and 4R GLS fit carries sigma_d 2-16%,
  and the parameters at those cut-offs mainly reflect how the weighting trades the inner shells against the 2-4R tail.
  Test F's outer-component form removes this.
- **Error assumptions.**
  - sigma_d is white and constant-fractional, and analytic errors include it while bootstrap intervals do not.
  - Bootstrap intervals assume independent haloes. The jackknife shows they are 10-40% too narrow for pressure
    parameters, so treat them as lower limits beyond 2R.
- **Observables.**
  - The engine applies the frozen halo-model kernels with an assumed redshift response.
  - Intervals on observable changes are linearised around the radii234 P4/E4 profiles (median deviation from the
    exact change: 9% of the change).
  - The pooled >= 14.9 trend extrapolates to 15.3-15.5 from 6 haloes (4 CONT).
  - Test F fits the two top bins individually. The 2R GLS shift of C_ell^yy differs by up to 1.4 points between the two
    tests; the cause suggested in B4 (the top bins) was not tested.
- **Sample definitions.**
  - CONT-template uses the CLEAN 2R GLS fit as the overlapper model, and the published B12/B16 below 10^13.5 Msun (1,298
    of 1,979 overlappers, most of them satellites whose amplitude is matched to their bound gas). Test A studies the
    subtraction methods in depth.
  - The common halo-shell set of the CONT methods excludes the inner shells of haloes with overlappers closer than about
    R_j, mostly major mergers.
- **Particle L_X.** The particle ROSAT L_X differs from SOAP's recomputed L_X (+0.3% median, 148 haloes > 5%). Q_X and
  T_X use the particle field.
- **Rank deficiency and resolution.**
  - 15.1-15.3 (N = 21) and 15.3-15.5 (N = 6) have fewer haloes than shells.
  - CLEAN 15.3-15.5 (N = 2) is undetermined.
  - Below 10^13.1 Msun no Lee22 shell is resolved for every halo.

## Files, runtimes and reproduction

**Code** (`B_covariance/`):
- `bcommon.py` (configuration, stacks, bootstraps), `bcorr.py` (correlation models), `bfast.py` (fast replicate
  refits);
- the compute stages `b0_prepare.py`, `b45_fits.py`, `b5_templates.py`, `b1_lee22.py`, `b2_stability.py`,
  `b3_weightings.py`, `b5_jackknife.py`, `b6_engine.py`;
- the driver `run_test_b.py` (`--smoke` / `--full`, `--stages`, `STUDY_ROOT`);
- local post-processing: `make_figures.py` (`b45_tables.py`, `b7_figures.py`, `b8_verify.py`, `build_report.py`);
- `write_cluster_run.py` and `CLUSTER_RUN.json` (deployment manifest).

**Products:**
- `results/` (cluster; npz/csv/json, run_manifest.json with per-stage times and input hashes);
- `tables/` (report tables); `figures/` (PNG 150 dpi + SVG); `verification.json`; `SHA256SUMS`;
- `results_smoke/`, `figures_smoke/`, `tables_smoke/` and `verification_smoke.json` (the --smoke test);
- `attempts/local_pre_cluster_20261012/` (superseded local partial run).

**Runtimes on maas (4 workers):**
- first invocation: prepare 86 s, CLEAN fits 572 s, templates 93 s, fits 6,306 s, B1 5 s;
- second invocation: B2 1,062 s, B3 7 s, jackknife 218 s, engine 1,197 s;
- peak tree RSS 4.0 GiB (first invocation) and 3.8 GiB (rerun), from the run monitor; the driver's own ru_maxrss peaked at 4.4 GiB (prepare);
- local post-processing: about 6 min.

**Reproduce:**
- cluster: `STUDY_ROOT=<study> python -B B_covariance/run_test_b.py --full --workers 4`;
- locally: `python -B B_covariance/make_figures.py`.
