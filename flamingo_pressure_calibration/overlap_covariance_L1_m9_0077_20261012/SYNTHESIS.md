# FLAMINGO L1_m9 (z = 0): what to paint for HalfDome - synthesis of Tests A-G

Study `overlap_covariance_L1_m9_0077_20261012`, 2026-10-12/13. Plan:
`/root/.claude/plans/there-has-been-multiple-indexed-lantern.md` (approved, extended by the user with Test F).

**Inputs.** One verified particle pass over FLAMINGO L1_m9 snapshot 0077 on STRW maas (4,588 centrals, 4R200c / 6R500c
spheres, owner x phase x quantity shell sums).

**Where the tests ran.** On maas, at nice 19 / lowest I/O priority, with at most 8 load-sized workers and memory guards.
Shared data were only read. Figures and reports were made locally.

Per-test reports:
- `A_overlap/REPORT.md` (overlap)
- `B_covariance/REPORT.md` (covariance, stability, weightings, cutoffs)
- `C_sigmoid/REPORT.md` (truncation shape)
- `D_literature/REPORT.md` (literature recipes)
- `F_observables/REPORT.md` (observable-driven calibration)
- `G_physical_outer/REPORT.md` (physical outer components, per-field calibration, Lee22-style parameter ladder; added
  2026-10-13 at the user's request); full report with the equations and the fitted profiles against the stacks:
  `G_physical_outer/report/Test_G_report.pdf`
- `kernels/REPORT.md` (radial importance)
- `quicklook/REPORT.md` (first covariance pass)

## 1. Recommendation for painting HalfDome

| Choice | Recommendation | Evidence |
|---|---|---|
| Gas | All gas, own satellites kept (literature "full signal"); recent-AGN-heated gas removed | HalfDome's RFoF catalogue never paints satellites, so their gas belongs to the host (A). The AGN cut changes every observable by <= 0.04-0.26% (A, D, F); only unresolved cores move. |
| Ownership against painted neighbours | Scaled-distance partition (A2-ii): gas closer in r/R200c to another painted halo (M200c >= 6.6e12) goes to that halo | Inside 2R it equals ALL to <= 1.4%, so the 2R calibration is unchanged. Beyond 2R it removes the 2.4-7.4% of Y(<4R) and 5-7.6% of N_e(<4R) otherwise painted twice. It makes 4R pressure fits adequate (sigma_d 8-14% -> 0.2-3.9%) (A). |
| Truncation | Hard spherical cut at 4 R200c for pressure and electrons | A 0.05-dex sigmoid changes tSZ C_ell by <= 0.2% at 4R; no ringing appears in C_ell or the box P(k) (C). Any cut below 4R loses real signal: a 2R cut costs tSZ -5%, kSZ -26%, DM -22% (F). |
| Fit radius | 4 R200c (cut = fit = 4R) | The only Pareto-optimal (R_fit, R_cut) over tSZ / kSZ / DM PDF / tSZ x DM. All 8 spatial-CV folds chose it. It halves today's errors, from 4.1 / 3.2 / 2.2 / 3.9% to 2.2 / 1.1 / 1.3 / 1.8% (out-of-fold 1.8 / 1.1 / 1.3 / 1.8%) (F). B12/B16 describe the stack within its errors only to 2R, but a 2R fit falls 31-37% short in the 2-4R shells that carry DM and kSZ (B). See 2.4. |
| Fit weighting | Current diagonal (halo SD/mean); importance-diagonal for the two-component profile | Upweighting important shells moves the errors by only -0.3..+0.7 points (F). GLS agrees with the diagonal fit to 2R (C_ell^yy within 0.2%). Beyond 2R its model-discrepancy term turns it into a nearly unweighted fit of the 2-4R tail: at 4R ln P0 rises by 0.38 (P0 x1.45, median), C_ell^yy -3 to -6%, kSZ -3.8% (B). It is the worst choice for the observables at every fit radius (F). |
| Profile form (minimum change) | B12 (pressure) / B16 (electrons) refitted to 4R: new parameters only, no XGPaint change | Costs some inner accuracy: the RMS inside 1R rises from 2.1 to 3.0% (pressure) and from 1.9 to 8.5% (electrons) relative to a 2R fit (F). |
| Profile form (better) | Per field, with physical outer terms (G). Pressure: P2m = B12 x a splashback-like transition + the linear 2-halo gas at a mass-dependent outer temperature (44 parameters). Electrons: E2gf = B16 x the transition + the parameter-free linear 2-halo term (41 parameters). Painted x p_keep(x, M), the fitted A2-ii/all ratio | tSZ 0.52%, kSZ 0.32%, DM 0.50% (S1, painting target; CV 0.38 / 0.33%). More accurate inside 2R than F's empirical outer power law (A (x/2)^-s, 50 parameters) and robust at any R_fit; the power law follows 2-4R electrons better (3.5% vs 10%). The terms were tested physically: transition at the splashback scale, 2-halo level within +-10% at 10^13.9-14.9, outer temperature matching the measured T_e at 3-4R (G). |
| Profile form (fewest parameters) | Lee22's gNFW with every parameter a broken power law in mass (common break at 10^14.1 Msun), plus the physical outer term with x_t(M): 14 parameters (pressure), 12 (electrons) | Selected by a Lee22-style ladder (fewest parameters with dBIC <= 6 and CV within the best's scatter); chi^2/nu 1.32 / 1.26 at 4R. Costs tSZ 1.14% (vs 0.52%) and kSZ 1.26% (vs 0.32%); DM unchanged (0.50%). Lee22's own break choices are rejected; concentration terms do not help painting (HalfDome has no concentrations) (G). |
| Scatter | Lognormal amplitude per painted halo; variance per bin from FLAMINGO's per-halo Y(<4R), N_e(<4R) (sigma_P^2 0.31 -> 0.007, sigma_E^2 0.045 -> 0.005 over 13.5-15.4) | One mean profile per bin leaves 1-halo tSZ 1.5-4% low. With the better profile, errors fall to 1.3 / 0.4 / 0.4 / 0.9% (F). Separate inner (Y(<1R)) and outer amplitudes do worse (tSZ 1.43% vs 0.52%): the core's scatter is mostly a change of shape about ~0.55 R200c, which Y(<1R) averages out (G and the shell correlation matrices). |
| Do not | Calibrate on clean haloes; subtract overlappers; use medians, Y200 weighting, X-ray- or mass-weighted pressure, the A10 universal shape, or B16 density converted with mu_XGP for electrons | (A, D) |

## 2. The questions, answered

### 2.1 Do the inner profiles matter most for tSZ / FRB DM?

**For tSZ: yes inside R200c, but not the core.** For DM: no.

Shares of the signal inside 1R / 1-2R / 2-4R R200c, from the measured FLAMINGO truth (F, kernels):

| Observable | Share inside 1R / 1-2R / 2-4R | Where it peaks |
|---|---|---|
| tSZ auto | 82 / 15 / 4% at ell = 100; 96 / 4 / 0% at ell = 3000 | 0.25-0.5 R200c; the core < 0.1R holds 3-11% |
| kSZ 1h | 46 / 30 / 25% at ell = 100; 67 / 26 / 7% at ell = 3000 | 0.5-1 R200c |
| mean DM | 32 / 31 / 37% | outskirts |
| DM Q16 | 5 / 31 / 64% | outskirts |
| DM Q50 | 18 / 37 / 45% | outskirts |
| DM Q99.9 | 80 / 11 / 8% | core (high-DM tail) |
| tSZ x DM | pressure side 75-95% inside 1R; electron side 56-95% (ell = 100 -> 3000) | |

- DM also depends on haloes below 10^13.5, where the published fallback is still used: 42-50% of mean DM and Q16-Q84.
- The current fit weighting peaks at 0.4-0.7R. That suits tSZ but under-weights what DM and kSZ need. This is why fitting
  to 4R improves them.

### 2.2 Overlap test (overlay = structure >= 10% of M200c reaching inside 2 R200c) - Test A

**Sample.** 1,650 of 4,588 centrals are overlapped: 1,013 with an overlapper reaching inside 1R, 637 in the 1-2R annulus
only. Every overlapper reaching inside 1R is an own satellite; neighbouring centrals matter only beyond 1R (88 of 1,979).

**Inside 1R.** Overlapped haloes are low (Y(<0.5R) -9 to -13%, inner B12 amplitude -17 to -27%). This is selection and mass
normalisation, not contamination:
- removing the overlappers' gas does not lift it;
- the deficit sits only in the shells around a central satellite;
- annulus-only haloes match the clean ones inside 0.71R;
- in host-only mass units (M200c minus the satellite) it disappears (-0.3 to +3.4%).

**1-2R.** Contamination is real and localised: shells next to an overlapper are +49% in pressure and +21% in electrons.
Injecting the overlapper gas into clean haloes reproduces 19 of the 25% excess.

**Subtraction recovers the clean profile with no method.**
- Bound-gas removal and templates (identical to 0.3%) leave the unbound atmospheres: a residual of +3.5-5%.
- Excision recovers injected gas to <= 0.9% (kappa = 1-2). On real haloes it overshoots by 6-16% in Y(<1R), because it also
  cuts host gas.
- Keeping own satellites (the literature convention) changes every method by <= 0.6%.
- Placebo excisions on clean haloes are unbiased (<= 0.45%) unless pointed at the nearest neighbour (-2 to -4% at kappa = 2).

**High-mass bins.** >= 14.9 is pooled (154 haloes, 50 overlapped); 15.1-15.5 is exploratory (27 haloes). The conclusions
hold there with wider intervals.

**Observables relative to painting ALL.**
- Clean only: yy +0.7 / +2.1 / +4.1 / +7.9% (ell = 100 / 1000 / 3000 / 8000); mean DM -2.2%.
- Overlapped haloes after subtraction (that subsample alone): yy -8 to -22%.
- A2-ii partition (measured to 4R): mean DM -3.3%, DM Q16 -5.8%.

So: paint ALL with A2-ii ownership; do not paint clean or subtracted profiles.

### 2.3 Covariance matrices, stability, weightings (Lee22 / B24) - Test B

Setup:
- Lee22's exact 27-shell grid for the correlation matrices.
- Fits on a 35-shell grid in which every cutoff is an exact edge.
- 500 bootstrap replicates, paired between CLEAN, CONT and ALL.
- Pressure is the geometric estimator sum(y)/V_shell with recently AGN-heated gas removed.

**Covariance (Lee22 Fig. 1).**
- **10^13.2-13.4 h^-1 Msun.** Only 0.42-1.34 R200c is resolved for every halo. These 9 shells form one positive block
  (adjacent rho 0.89), described by a single exponential correlation model.
- **10^13.8-14.0 h^-1 Msun (0.15-1.34 R200c): two blocks.**
  - Shells inside ~0.6 R200c are strongly correlated (mean rho 0.82).
  - Outer shells are moderately correlated (0.54).
  - The two blocks are anti-correlated (-0.22; break at 0.61 R200c).
  - Above 10^14 Msun the break moves inward to 0.4 R200c.
- **Electrons** look the same.
- **Overlapped haloes** have almost no inner-outer anti-correlation (-0.05, against -0.27 for clean ones).
- **Near-singular for GLS.** The covariance of ln(mean) that a GLS fit inverts is near-singular on Lee22's grid: adjacent
  rho 0.90-0.94, condition numbers 220-1,500.

**Stability between adjacent bins** (permutation null, 1,000 relabellings at the original N):
- **Pressure:** one structure from 10^13.9 to 10^15.5 (p = 0.06-0.7). The 3 of 11 pairs that differ all lie below 10^14.1,
  where the resolved range moves inward and the second block appears.
- **Electrons:** 9 of 11 pairs differ.
- **Verdict on Lee22's claim:** their qualitative stability claim holds only for pressure at cluster masses.
- **It does not matter for the calibration.** Fitting a bin with its neighbour's covariance moves the parameters by a
  median of 0.1-0.25 sigma and the observables by <= 0.3%.

**Covariance-weighted (GLS) fit vs the current diagonal fit.**
- **Up to 2R they agree:** P0 +4% (median), C_ell^yy within 0.2%, mean DM -1.0%. The plain error-on-mean diagonal equals the
  current one to <= 0.5%.
- **Beyond 2R the B12/B16 shapes no longer describe the stack.**
  - The model-discrepancy term sigma_d grows from 0.3% (2R) to 3.5% (3R) and 12% (4R).
  - It turns GLS into a nearly unweighted fit that follows the 2-4R tail: at 4R ln P0 rises by 0.38 (P0 x1.45, median),
    C_ell^yy -3 to -6%, kSZ -3.8%.
  - These shifts match Test F's own to <= 0.5 points. The diagonal fit is already low in C_ell^yy, so GLS moves it further
    from the truth.
- **Which radii drive each fit.**
  - The diagonal fit draws P0, xc and beta from 0.1-1R; xc and beta peak at 0.5-0.6R.
  - GLS shifts the influence to the least redundant end shells: the innermost resolved shells (0-0.1R) rise from 5% to
    13-19%, and 1.5-2R from 7-11% to 11-20%.
- **The covariance model matters at the 2% level:** a 2-factor covariance changes C_ell^yy by up to -2.5% at 2R.
- **One open difference.** At 2R, Test F's GLS lowers C_ell^yy by 0.5-1.3%, while Test B's moves it by <= 0.2%. The likely
  cause, not tested, is that F fits the two top bins (N = 21 and 6) individually, whereas B pools >= 14.9 with a mass trend.

**B24 weightings vs the tSZ-correct geometric estimator** (bins 13.5-15.1; via fits to 2R):
- **Q_V (volume-weighted).**
  - It reads high only because the summed SPH volumes fill 97% (0.1-0.5R), 91% (1.75R) and 78% (3.6R) of the geometric volume.
  - Bias: +3-5% inside 1R, +6-7% at 1-2R, +13-14% at 2-3R and +18-25% at 3-4R, at every mass above 10^13.5.
  - Through the fits: C_ell^yy +6 to +8%; for electrons, mean DM +4.3%.
  - In the full B24 recipe the medians offset this (Test D: net -2 to -3%).
- **Q_M (mass-weighted)** is +13-38% at 0.5-1R, x1.6-2.6 at 1-2R and x7-20 at 2-3R.
- **Q_X (X-ray-weighted)** is +17-36% at 0.1-0.5R, x1.3-2.4 at 0.5-1R and x3-10 at 1-2R.
- **Why Q_M and Q_X read high:** dense, X-ray-bright clumps dominate them. The pressure clumping factor is 1.04 at 0.1R and
  2.8 at 1.75R.
  - At 2R, Q_M raises C_ell^yy by +27 to +80%.
  - B12 cannot fit Q_M beyond 2R, nor Q_X from 2R.
  - B24's "pressure is nearly weighting-independent" therefore holds only roughly, for Q_M relative to Q_V in the inner
    region (Test D).
- **n_e kT_X (the X-ray route)** is unbiased inside 0.5R (within 1%) and -2 to +5% at 0.5-1R.
  - It is +10-42% at 2-3R, because T_X > T_M (1.24 at 2.5R).
  - Its 2R fit changes C_ell^yy by -2.8 to +0.8%.
- **Mass- and X-ray-weighted densities** are clumping-biased (mass-weighted: x7-10 at 0.1-0.5R, x300-390 at 1-2R). They must
  not be used for DM.

**Sample comparison at 2R** (relative to ALL):
- **CLEAN:** C_ell^yy -0.2 to +3.3%, mean DM -2.2%.
- **CONT-raw:** C_ell^yy -4.5 to -10.4% (inner amplitude: P0 -22%), while mean y rises +4.6% and mean DM +5.0% (more 1-4R
  gas).
- **No subtraction method recovers CLEAN.** The keep-own-satellites flavours equal CONT-raw, because only 95 of the 1,979
  overlappers are not own satellites.
  - This confirms Test A: the inner CLEAN-ALL difference is selection, not a correction to apply.
- **A2-i / A2-ii:** no change at <= 2R (<= 0.12%). In the GLS fits to 4R they raise C_ell^yy by up to 5% and lower mean DM
  by up to 3.6%, because the neighbour gas they remove sits at 2-4R.
- **Spatial jackknife (4^3 sub-volumes):**
  - Errors are 1.16-1.26x the halo bootstrap for ALL pressure parameters (1.32-1.41x for CONT) and 1.03-1.11x for
    electrons.
  - Per shell the factor is ~1.0 inside 2R and 1.11 at 2-4R, where neighbours share gas.
  - Bootstrap intervals beyond 2R are therefore lower limits.

### 2.4 Radial cutoffs 1.0 / 1.34 / 1.5 / 1.75 / 2 / 3 / 4 R200c - Tests B and F

These are two separate questions with different answers, and the answers do not conflict.

**Where do B12/B16 describe the stacked profile? Up to 2R (B).**
- Pressure is described within its statistical errors at every cutoff up to 2R (sigma_d <= 0.7%, chi2/nu <= 3.4 in every
  bin). Beyond, sigma_d is 1.9-5.8% at 3R and ~12% at 4R.
- Electrons need a small sigma_d at every cutoff: 0.6% at 1R, 1.2% at 1.34R, 1.6% at 2R, then 6% at 3R and 11% at 4R.
- Lee22's 1.01 / 1.34 / 1.76 (here 1.0 / 1.34 / 1.75) are all adequate and change C_ell^yy by <= 1.4% relative to 2R. The
  1.5 and 1.75 fits lie between the 1R and 2R fits.
- So 2R is the right basis for the profile *shape*.

**Which fit and cut give the right observables? Both 4R (F).**
- A 2R fit extrapolated outward falls 31-37% (median) below the measured 2-4R shells (B). It misses Y(<4R) by -6% and
  N_e(<4R) by -8%. Those shells carry 37% of mean DM, 64% of DM Q16 and 25% of the kSZ 1h at ell = 100.
- DM therefore depends on the fit range. Relative to the 2R fit, mean DM is +9.0% (1R), +4.6% (1.34R), +1.5% (1.75R), -2.1%
  (3R) and -0.4% (4R) (B). tSZ moves by <= 1.4%.
- Only (R_fit, R_cut) = (4, 4) is Pareto-optimal. No intermediate cutoff (1.34-1.75) beats it for any observable family (F).
- The cut matters most. Any cut below 4R removes real signal: a 2R cut costs tSZ -5%, kSZ -26%, DM -22% (F).

**How the two fit together.**
- **Plain B12/B16 fitted to 4R** gets the integrals right at some cost in inner accuracy (F3):
  - pressure RMS inside 1R rises from 2.1% to 3.0%, and at 1-2R from 3.7% to 8.8%;
  - electron RMS inside 1R rises from 1.9% to 8.5%.
  - tSZ hardly notices, because 1-2R holds only 4-15% of it.
- **The two-component profile** (B12/B16 + outer power law, fitted to 4R) removes the trade-off (F). Test G replaces the
  empirical outer power law by physical terms, calibrated per field (2.10).
  - Inside 1R it is as accurate as the 2R fit (pressure 2.2%, electrons 3.4%), and it follows 2-4R to <= 5%.
  - B12/B16 keep describing the inner 2R; the outer term carries the 2-4R gas.
- **A2-ii ownership** removes most of the neighbour gas that makes B12 inadequate at 4R: pressure sigma_d falls from 8-14% to
  0.2-3.9% (A).
  - Electrons improve less (3.4-8.1% above 10^13.9, no better below), so the electron profile still needs the outer term.

### 2.5 Sigmoid vs hard truncation - Test C

- **Pressure.** At 4R a 0.05-dex logistic changes C_ell by <= 0.21% for ell = 100-2e4 (<= 0.07% with a flux-matched edge;
  the centred one is up to 0.7% low at ell < 100, a flux change), and the box P(k) by <= 0.35%. Ringing appears only in single-halo transforms at k ~ 10/Mpc. The hard edge adds no identifiable high-ell
  power at 4R.
- **Wider windows (0.2 dex) or cuts at 1-2R** move C_ell by 5-20% through flux redistribution, not ringing.
- **Electrons** are ~5x more edge-sensitive (DM-field C_ell -1.1% at ell = 1000).
- **Recommendation:** keep the hard cut at 4R. If a smooth edge is wanted for FFT work, use a 0.05-dex flux-matched logistic.

### 2.6 Literature methodologies on the same haloes - Test D

Bias each recipe would put into HalfDome (yy 1-halo at ell = 100 / 1000 / 3000; Y within 4R):

| Rank | Recipe | Bias | Driver |
|---|---|---|---|
| 1 | B24 with X-ray weighting | +80 / +42 / +23%; +50% | L_X-weighted particle pressure |
| 2 | A10 | +29 / +5 / -4%; -8% | one universal shape with M^0.12 cannot follow FLAMINGO's mass trend; median; R500c scaling; 5 R500c cut |
| 3 | B24 with mass weighting | +29 / +16 / +8%; +12% | mass-weighted pressure |
| 4 | B16 (density) | kSZ -12.6%, mean DM -5.3%, DM quantiles -5 to -6% | painting gas density with XGPaint's mu_XGP gives 10% fewer electrons (a convention) |
| 5 | B12 | +4.4-5.6%; +2.1% | Y200 weighting; internal-variance fit to 3 R200c |
| 6 | B24 with volume weighting | -2 to -3% (2h -6%) | ingredients cancel |
| 7 | Lee22 | -0.5 to -2.3% | |

- Medians undershoot the mean by up to 38% at 1.5-3R and 59% at 3-4R.
- B24's volume weighting on its own raises C_ell^yy by +6 to +8% (B, 2.3); in the full recipe the medians offset it.
- SOAP positive controls: our aperture sums reproduce SOAP's Y and mass-weighted T to ~1e-5.

### 2.7 Recently AGN-heated gas (B24's 15-Myr cut), in every test

- It changes fitted parameters by <= 0.03-0.19% and every observable by <= 0.04% (F), <= 2e-5 (A) or <= 0.26% (D, B24
  X-ray recipe).
- Only unresolved cores of low-mass groups move: Y(<0.1R) -3.6% at 10^13.5-13.7; the innermost shell -7 to -44%.
- It is removed by default (as in FLAMINGO's own y maps); keeping it makes no practical difference.

### 2.8 Observable-driven calibration - Test F

- **Which bins matter:** see 2.1.
- **Best (R_fit, R_cut) = (4, 4)** for all four observable families, and in cross-validation.
- **Upweighting the high-contribution bins does not let B12/B16 extend further.** Fitting to 4R does, and the profile form is
  the remaining limit.
- **Other methods.**
  - Free-gamma gNFW and direct observable fits score best numerically, but by compensation (unresolved cusps, distorted
    profiles); they are not safe to paint.
  - The physically consistent best is B12/B16 + an outer power law fitted to 4R, plus lognormal scatter:
    1.3 / 0.4 / 0.4 / 0.9% on tSZ / kSZ / DM PDF / tSZ x DM.
- **Scatter floor.** Halo-to-halo scatter adds +1.5% (ell = 100) to +7.3% (ell = 8000) to the 1-halo tSZ. One mean profile per
  bin cannot reproduce that.

### 2.9 Satellites and HalfDome

- **Catalogue.** HalfDome's RFoF catalogue has no subhaloes. Satellites, typically ~5% of M200c each (a group mass in
  clusters), are part of their host's entry.
- **Calibration.** Their gas must therefore stay in the host's calibrated profile, as in every published fit.
- **Effect of removing it.** Removing the flagged overlappers' bound gas, in census proportions (A, "mix CLEAN + bound"),
  biases the painted population low: yy -2.9 / -1.6 / -0.8% at ell = 100 / 1000 / 3000, mean DM -0.9%.
  - For the overlapped subsample alone the bias is -8 to -17% in yy.
  - Removing all satellite gas, not only the flagged overlappers', moves the 4R fits further: P0 -7 to -26% (quick look).

### 2.10 Physical outer components, per field - Test G

- **Separate calibrations.** Pressure is chosen on tSZ, electrons on kSZ plus the DM PDF. Both still prefer
  R_fit = R_cut = 4 R200c, objective (a) and S1 scatter. Electrons need the 4R cut (3R costs 8-10% in kSZ/DM); for
  pressure R_fit 3 and 4 are equivalent, but the cut must stay at 4R.
- **Which physical terms hold.**
  - A splashback-like transition: x_t = 2.1-2.4 R200c (1.1-1.6 R200m) for electrons, with a steep gamma_t = 5.1.
  - The linear 2-halo term (no free parameters) matches the stacked 3-4R electrons within +-10% at 10^13.9-14.9 Msun.
    The stack is 16-26% above it in groups and 15-19% below it above 10^14.9.
  - Pressure needs the 2-halo gas at a mass-dependent temperature (T_out proportional to M^0.24). A self-similar outer
    temperature is 1.3-4.4 times too cold at 3-4R. The temperature law is checked only at 3-4R; inside R200c P2m's
    outer term carries 9-35% of the pressure, so the bound/outer split there is a fit decomposition.
  - Not supported: a universal infall term (an outer excess exists only below 10^14.1), and an accretion-shock cutoff
    inside 4R.
- **Normalisation correction.** In XGPaint's electron units (mu_XGP includes a 1/0.9), the cosmic mean electron density
  is 0.335 = 1.094 Omega_m, not Omega_m. Earlier 2-halo ratios quoted against Omega_m were 9% high (factor 0.915), and
  pressure-"implied" outer temperatures were a factor 1.96 low. Measured T_e/T200 is 0.43 -> 0.30 at 1.5-2R and
  0.34 -> 0.10 at 3-4R from 10^13.6 to 10^15.35.
- **Lee22-style parametrisation.** On the mass-binned stacks, Lee22's convention (chi^2 with the covariance of the mean)
  reproduces their quality numbers (3.7 / 5.4 against their 3.3 / 5.1). The fewest-parameter model that is as good as
  per-bin fits is a broken power law in mass for every gNFW parameter, with a common break at 10^14.1 Msun (Lee22:
  10^13.8), plus the physical outer term: 14 parameters for pressure, 12 for electrons.
- **Scatter.** One lognormal amplitude (S1) stays best. Inner and outer amplitudes (S2) are only weakly correlated
  (|rho| mostly < 0.2), but S2 under-predicts the 1-halo boost.

## 3. What was run, where, and how it was verified

**Pass.** One particle pass on maas: 66 min, 6 workers, nice 19; 308 GB logical read; peak 8.8 GiB of 188.
- Legacy sums are bit-identical to radii234 for all 4,375 old haloes.
- New blocks equal the legacy totals to 1e-14.
- Y vs SOAP: median 1e-5.
- Two documented SOAP-comparison exceptions:
  - Y(R500c) 98.47% within 0.3%, from low-gas haloes;
  - particle vs SOAP-recomputed L_X, +0.3% (definitional).

**Tests on maas** (polite wrapper):

| Test | Wall time | Peak memory | Status |
|---|---|---|---|
| A | 2.1 h | 7.3 GiB | done |
| D | 0.5 h | 3.8 GiB | done |
| F | 1.7 h | 8.7 GiB | done |
| B | 2.0 h (stopped in B2) + 0.7 h (rerun of the remaining stages) | 4.4 GiB | done |
| G | 0.4 h (4 workers; stopped to switch to 8 at the user's request) + 0.4 h (8 workers, remaining stages) | 7.0 GiB | done |

Each test passed its own gates; details are in each `verification.json`.

**Bugs found and fixed during the work:**
- **Zero-amplitude templates gave NaN** (`geometry.make_G`). Fixed before Test A's template step.
- **The frozen engine kernel loses precision for steep unresolved cusps.** A stable kernel is in
  `D_literature/td_stable.py`. It affects only intermediate Test D subsets; no Test F number changed.
- **Test B's stability stage crashed on a 2-halo bin.** CLEAN 15.3-15.5 has N = 2, so its residuals are all zero and its
  correlation matrix is NaN. Degenerate samples are now recorded as undetermined, and the remaining stages were rerun from
  the saved fits.
- **Worker memory grew to 11 GiB in the pass.** HDF5 held decompression buffers while files stayed open; reopening files
  every 16 cells plus a fixed mmap threshold brought it to 1.5 GiB.

## 4. Limits and next steps

- **Mass coverage.** The calibrated range is log M200c 13.5-15.4. It carries 97-99% of tSZ but only 58% of mean DM and
  50-89% of the DM quantiles. DM needs calibration below 10^13.5 (resolution-limited in L1_m9; L1_m8 would help).
- **Redshift.** All of this is z = 0 only, and the redshift response is assumed. A z ~ 0.5-1 snapshot is the next
  calibration step.
- **Closure tests not yet done:**
  - painting FLAMINGO's own haloes and comparing with its lightcone y/DM maps;
  - a hydro-to-DMO M200c mapping (a 5% mass offset is ~8% in Y and ~16% in C_ell^1h).
- **To adopt the recommendation:**
  - refit B12/B16 to 4R on A2-ii-owned all-gas profiles and export XGPaint parameters (minimum change);
  - then implement the per-field physical profiles of Test G (P2m, E2gf, times p_keep) and per-halo lognormal
    scatter in XGPaint.
- **Open after Test G:**
  - a smooth-in-mass version of P2m/E2 (broken power law plus the mass-dependent outer temperature) is untested;
  - the 2-halo normalisation trend with mass (-19% to +26%) is not modelled (linear xi, no halo exclusion);
  - a scatter model with a core-shape (pivot) component was not tried.

## 5. Files

- **Extraction and verification:** `STEP1_README.md`, `verification.json` (G1-G7 + `G7_assessment`), `measurements_*.npz`,
  `apertures.npz`, `selected_halos.npz`, `overlap_tables.npz`, `soap_extra_columns.npz`.
- **Cluster runner:** `cluster_tests.py` (mirror / run / status / fetch / stop), `cluster_job.py` (polite job wrapper),
  `kernels_cluster_deps.json`.
- **Tests:** `A_overlap/`, `B_covariance/`, `C_sigmoid/`, `D_literature/`, `F_observables/`, `G_physical_outer/`, each with `REPORT.md`,
  `figures/`, `results/`, `verification.json`, `SHA256SUMS`.
- **Shared code:** `analysis/` (library), `kernels/` (engine, importance, Jacobians).
