# Test F: observable-driven calibration (FLAMINGO L1_m9, z = 0)

**Question** (user, 2026-10-12):
- which radial bins carry tSZ, kSZ, DM and tSZ x DM;
- which fit and cut radius fit those bins best;
- whether upweighting them extends the truncation;
- which other method recovers the end observables best;
- and what HalfDome should paint.

**Run.** `run_test_f.py --full` on STRW maas (4 workers, nice 19, 5,997 s, peak 8.7 GiB, exit 0). Figures and the numerics
check were made locally from the fetched results. Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md`.

## Answers in brief

1. **Where the signal comes from** (share inside 1R / 1-2R / 2-4R R200c, measured FLAMINGO truth):

   | Observable | ell = 100 | ell = 3000 | Peak |
   |---|---|---|---|
   | tSZ auto | 82 / 15 / 4% | 96 / 4 / 0% | 0.25-0.5 R200c (33-43%) |
   | kSZ auto (1h) | 46 / 30 / 25% | 67 / 26 / 7% | 0.5-1 R200c |
   | tSZ x DM, pressure side | 75 / 18 / 7% | 95 / 5 / 0% | |
   | tSZ x DM, electron side | 56 / 25 / 18% | 95 / 6 / -2% | |

   - The DM PDF is ell-independent: its low quantiles come from the outskirts (Q16 5 / 31 / 64%, Q50 18 / 37 / 45%; mean DM
     32 / 31 / 37%).
   - Its tail is core-dominated (Q99.9 80 / 11 / 8%).
2. **Best (R_fit, R_cut): cut at 4 R200c and fit to 4 R200c.**
   - It is the only point on the Pareto front over the four observable families.
   - Every cross-validation training set chose it.
   - It stays best with AGN gas kept and under both painting-consistent (A2) gas selections.
   - It halves the errors of the current 2R fit.
3. **Upweighting the important bins does not extend what B12/B16 can do.**
   - At R_fit = 4 it changes the family errors by -0.3 to +0.7 points.
   - At R_fit = 2-3 it cannot recover the 2-4R gas that B12/B16 misses.
   - The profile form is the limit, not the weighting: B12/B16 plus an outer power law follows the measured mean to a few %
     at every radius.
4. **Best recovery of the observables.**
   - Numerically, the free-gamma gNFW fit and the direct observable fits score best (0.2-1%). Both get there by compensation:
     a biased mean profile stands in for the missing scatter, and gNFW also leans on an unconstrained core.
   - The physically consistent best is a correct mean profile (B12/B16 + outer power law, fitted to 4R) plus per-halo
     lognormal amplitude scatter: 1.3% (tSZ), 0.4% (kSZ), 0.4% (DM PDF), 0.9% (tSZ x DM).
5. **Paint** with a hard cut at 4R and a fit to 4R (see "Painting recommendation").
6. **Numerics** (the Test D kernel cancellation): no number in this report needed correction; the largest change is 6e-6.

## Method

- **Data.** The particle-pass products (4,588 centrals), built with `analysis.variants.profile`:
  - pressure Pe/P200 (geometric estimator) and electrons n_e mu/(rho_c f_b);
  - recent-AGN gas removed; all gas with satellites kept (the literature full signal);
  - controls: AGN kept, A2-i (remove gas bound to other painted haloes), A2-ii (partition owned).
- **Shells.** 35 shells to 4 R200c, a sliver-free sub-grid of X200 with exact edges at 1, 1.34, 1.5, 1.75, 2, 2.5, 3 and 4.
  Seven extension shells to 12R are used by models only.
- **Truth.** The measured per-halo profiles to 4R, propagated through the frozen lightcone halo model of `kernels/`.
  - Every engine observable is rewritten exactly in per-shell form.
  - The 1h terms and the DM variance use each bin's inclusion-weighted second moments (the mean of |y_ell,i|^2 over its
    haloes), so scatter counts. The DM PDF deposits every halo's own column.
  - Gate: per-shell evaluation vs the engine 1.3e-13 (observables), 4.9e-13 (Jacobians), 7.7e-12 (DM quantiles),
    1.1e-12 (quantile Jacobians).
  - Calibrated support is 13.5 <= log M <= 15.388 (the engine's). Outside it, the published B12/B16 fallback is identical in
    truth and models.
- **Models.** Every model is scored deterministically (one profile per bin, as painted today). It is also scored with
  scatter: lognormal amplitudes per bin with sigma from per-halo Y / N_e moments (apertures 1R and 4R), and with the measured
  second-moment correction.
- **Errors.** Delta O / O against the truth, and against the deterministic truth mean (the fit / cut effect alone). Family
  scores are RMS values:
  - over ell = 100, 300, 1000, 3000 for tSZ yy total, kSZ 1h and tSZ x DM total;
  - over the mean, variance and Q16 / 50 / 84 / 97.5 / 99.9 for DM.
- **Fits.** B12 (pressure) and B16 (electrons) at 8 R_fit x 6 R_cut, hard cut, with these objectives:
  - (a) current diagonal;
  - (c) GLS with shrinkage and sigma_d;
  - (d) importance-diagonal, with weights from F1;
  - (e) observable metric K = sum lambda_O J_O J_O^T / O^2;
  - (f) hybrid C^-1 + lambda K (lambda chosen on the sample: pressure 3, electrons 30);
  - plus a flux-matched 0.05-dex logistic edge for (a).
- **F4 methods:**
  - gNFW (5 free);
  - B12/B16 + A (x/2)^-s;
  - B12/B16 x spline;
  - the measured stack, and a stack partially pooled in mass;
  - a per-bin edge x_t matched to Y / N_e(<4R) or to an observable;
  - direct observable fits per bin, and a joint P+E fit with a smooth log-M trend (weak profile chi^2, eta = 0.1).
- **Cross-validation.** 4^3 blocks of the box grouped into 8 folds. Everything (truth, weights, covariances, lambda,
  configuration choice) is refitted on 7/8 and scored out-of-fold.

## 1. Which radial bins and haloes carry the signal (F1)

Normalised importance (sums to 100% over all haloes), evaluated at the measured truth. Band edges are the nearest F-grid
edges (0.105, 0.233 and 0.519 stand in for 0.1, 0.25 and 0.5). Columns are bands in R200c.

| Observable | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 | Calibrated share |
|---|---|---|---|---|---|---|---|---|---|
| yy total, ell 100 | 4 | 14 | 33 | 31 | 11 | 4 | 3 | 1 | 99% |
| yy total, ell 1000 | 7 | 20 | 38 | 26 | 7 | 2 | 1 | 0 | 98% |
| yy total, ell 3000 | 7 | 24 | 43 | 21 | 4 | 1 | 0 | 0 | 97% |
| kSZ 1h, ell 100 | 1 | 4 | 15 | 26 | 18 | 11 | 14 | 10 | 91% |
| kSZ 1h, ell 1000 | 1 | 5 | 19 | 29 | 18 | 11 | 11 | 6 | 83% |
| kSZ 1h, ell 3000 | 2 | 7 | 25 | 33 | 18 | 8 | 6 | 1 | 63% |
| mean DM | 0 | 2 | 10 | 19 | 17 | 14 | 20 | 17 | 58% |
| DM variance | 3 | 11 | 28 | 30 | 14 | 6 | 6 | 3 | 76% |
| DM Q16 / Q50 | 0 / 0 | 0 / 0 | 0 / 3 | 5 / 15 | 14 / 20 | 17 / 17 | 33 / 25 | 31 / 20 | 50 / 54% |
| DM Q99.9 | 6 | 21 | 35 | 17 | 7 | 4 | 5 | 4 | 89% |
| y x DM, P side, ell 1000 | 4 | 15 | 35 | 30 | 10 | 3 | 2 | 1 | 96% |
| y x DM, E side, ell 1000 | 3 | 11 | 30 | 31 | 14 | 6 | 4 | 1 | 95% |

**Haloes.**
- tSZ is carried by >= 10^14.9 at ell = 100 (59%), by 14.5-15.1 at ell = 1000 and by 14.3-14.9 at ell = 3000.
- Mean DM and Q16-Q84 come from 12.9-14.1, where 42-50% sits in the uncalibrated fallback below 10^13.5.
- Q99.9 comes from 14.3-14.9.
- kSZ at ell = 8000 is 68% fallback.

**Scatter.** The halo-to-halo scatter included in the truth adds to the deterministic mean profile:
- yy total: +1.5 / +3.3 / +3.2 / +4.1 / +7.3% at ell 100 / 300 / 1000 / 3000 / 8000;
- kSZ 1h: +1.4 to +2.2%;
- DM variance: +2.1%;
- Q99.9: +1.8%.

**Controls.**
- Keeping the recent-AGN gas changes every observable by <= 0.04%.
- A2-i lowers yy at ell = 100 by 0.7%, kSZ 1h at ell = 100 by 2.3% and mean DM by 0.9%.
- A2-ii lowers the same three by 1.5%, 6.7% and 3.2%.

Figures: `figures/F1_importance_bands.png`, `F1_importance_mass.png`, `F1_importance_profiles.png`. Tables:
`results/tables/F1_*.csv`.

![importance](figures/F1_importance_bands.png)

## 2. Fit radius x cut radius (F2)

Objective (a), the current diagonal. Each cell gives the family RMS |Delta O/O| against the truth, in %, as
tSZ / kSZ / DM / tSZ x DM.

| R_fit \ R_cut | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 22.8 / 54.6 / 39.2 / 36.8 | 5.9 / 24.8 / 20.4 / 12.9 | 3.8 / 7.7 / 5.3 / 5.2 | 3.3 / 8.7 / 8.2 / 2.0 |
| 2 (current) | 22.9 / 54.7 / 39.2 / 36.9 | 6.5 / 27.2 / 22.1 / 13.9 | 4.6 / 12.4 / 9.2 / 7.2 | **4.1 / 3.2 / 2.2 / 3.9** |
| 3 | 22.7 / 54.6 / 39.1 / 36.8 | 5.9 / 27.0 / 22.3 / 13.2 | 3.8 / 12.5 / 10.1 / 6.3 | 3.3 / 3.6 / 1.6 / 3.1 |
| 4 | 22.7 / 54.6 / 39.1 / 36.8 | 5.1 / 25.7 / 22.0 / 11.9 | 2.8 / 10.1 / 9.5 / 4.4 | **2.2 / 1.1 / 1.3 / 1.8** |

The full 8 x 6 grids for every objective are in `figures/F2_grid_{a,c,d,e,f}.png` and
`results/tables/F_model_scores.csv`.

- **Pareto front.** Over the four families it contains only (R_fit, R_cut) = (4, 4).
  - This is also the configuration chosen on the training data of all 8 CV folds.
  - It is the best for every family under the AGN-kept, A2-i and A2-ii truths. The exceptions are A2-i tSZ x DM and A2-ii
    tSZ, which prefer R_fit = 1 by <= 0.24 points.
- **The cut matters most.** Any cut below 4R removes real signal. At R_fit = 4:
  - a 3R cut: kSZ -10%, DM -9.5%;
  - a 2R cut: tSZ -5%, kSZ -26%, DM -22%.
- **Importance-weighted profile RMS** (resolved shells, weights = F1 importance) is also lowest at (4, 4). From (2, 4) to
  (4, 4):
  - pressure tSZ-weighted 5.6 -> 4.7%;
  - electrons kSZ-weighted 11.2 -> 8.3%;
  - electrons DM-weighted 16.1 -> 10.2%.
- **Single-observable optima** often sit where a signed error crosses zero (for example kSZ ell = 1000 at (2.5, 4): 0.10%; DM
  Q97.5 at (2, 4): 0.05%). These are not robust choices (`figures/F2_best_per_observable_a.md`).
- **Against the deterministic truth mean** (the fit / cut effect only):
  - tSZ is best at (1, 4), 0.6%: a B12 fitted inside R200c extrapolates well enough for tSZ. The same fit gives 10.6% for
    kSZ and 8.2% for DM.
  - (4, 4) gives 0.9 / 1.7 / 0.6 / 1.4%.
- **Edge shape** (engine-exact, objective (a) fits). The flux-matched 0.05-dex logistic edge:
  - equals the hard cut to <= 0.09% for tSZ;
  - lowers kSZ 1h and the DM-field C_ell by 0.7-1.1% at ell <= 1000;
  - moves the DM quantiles by <= 0.44%.

Figure: `figures/F2_pareto_a.png`.

![pareto](figures/F2_pareto_a.png)

## 3. Can upweighting the important bins extend the truncation? (F3)

Profile RMS on resolved shells and family errors against the truth. The pressure and electron columns give the RMS in
<1R / 1-2R / 2-4R; the last column is tSZ / kSZ / DM / tSZ x DM.

| Objective | R_fit/R_cut | Pressure RMS [%] | Electrons RMS [%] | Family errors [%] |
|---|---|---|---|---|
| (a) current | 2/4 | 2.1, 3.7, 41.0 | 1.9, 5.4, 36.2 | 4.1 / 3.2 / 2.2 / 3.9 |
| (a) current | 3/4 | 2.4, 4.4, 35.9 | 5.6, 6.9, 29.0 | 3.3 / 3.6 / 1.6 / 3.1 |
| (a) current | 4/4 | 3.0, 8.8, 28.2 | 8.5, 13.7, 21.6 | 2.2 / 1.1 / 1.3 / 1.8 |
| (c) GLS + sigma_d | 2/4 | 3.8, 6.8, 41.2 | 2.1, 5.5, 35.1 | 5.0 / 4.1 / 1.8 / 4.2 |
| (c) GLS + sigma_d | 4/4 | 5.8, 11.2, 21.4 | 11.3, 18.2, 17.3 | 6.5 / 3.0 / 2.6 / 5.8 |
| (d) importance-diagonal | 3/4 | 2.5, 5.3, 33.5 | 4.7, 7.5, 27.0 | 3.0 / 4.6 / 2.3 / 3.2 |
| (d) importance-diagonal | 4/4 | 3.4, 10.3, 24.7 | 7.7, 15.0, 19.5 | 2.4 / 1.8 / 1.5 / 2.7 |
| (e) metric K | 4/4 | 3.0, 8.8, 30.3 | 6.2, 13.0, 27.3 | 2.2 / 1.6 / 1.2 / 2.0 |
| (f) hybrid, lambda 3 / 30 | 3/4 | 2.5, 5.9, 34.2 | 4.3, 8.3, 32.2 | 2.9 / 4.7 / 2.2 / 3.2 |
| (f) hybrid, lambda 3 / 30 | 4/4 | 4.6, 12.3, 23.8 | 9.8, 11.9, 24.1 | 2.0 / 1.6 / 1.2 / 1.9 |
| B12/B16 + outer power law, (d) | 4/4 | 2.2, 3.2, 4.6 | 3.4, 3.9, 3.7 | 2.8 / 2.5 / 1.6 / 2.3 |

**Plain answer: no.** Upweighting the important shells does not let B12/B16 reach further; with every objective the
observables improve most by fitting to 4R.
- **At R_fit = 4**, importance weighting (d), the metric (e) or the hybrid (f) move the family errors by -0.3 to +0.7 points
  relative to the plain diagonal. tSZ improves (2.2 -> 2.0%, hybrid), DM barely (1.26 -> 1.20%), and kSZ gets worse
  (1.1 -> 1.6%).
- **At R_fit = 2-3** no weighting recovers the 2-4R gas: B12/B16 stays 31-42% low there.
- **GLS (c) is the worst at every R_fit**; its sigma_d term flattens the weighting.
- **The model form is the limit.** One extra outer component (B12/B16 + A (x/2)^-s, s free) follows the measured mean to
  <= 5% everywhere, including 2-4R, without degrading the inner shells.

Figures: `figures/F3_objectives.png`, `F3_lambda_scan.png`, `F6_profile_residuals.png`.

## 4. Other methods and the method x observable table (F4, F5)

Each cell gives the family RMS |Delta O/O| in %, as tSZ / kSZ / DM / tSZ x DM.
- **Deterministic:** one profile per bin, as painted today, scored in-sample.
- **Out-of-fold:** the same, scored in spatial CV.
- **Vs truth mean:** the profile and fit effect alone, without the scatter target.
- **+ lognormal 1R / 4R:** per-halo amplitude scatter, with sigma from Y and N_e inside 1R or 4R.

All fits and stacks use a cut at 4R.

| Method | Deterministic | Out-of-fold | Vs truth mean | + lognormal 1R | + lognormal 4R |
|---|---|---|---|---|---|
| B12/B16 (a), fit 2R (current) | 4.1 / 3.2 / 2.2 / 3.9 | 3.7 / 3.2 / 2.2 / 3.9 | 1.6 / 2.5 / 1.6 / 2.5 | 3.2 / 2.9 / 1.9 / 3.5 | 1.6 / 3.0 / 1.5 / 2.1 |
| B12/B16 (a), fit 4R | 2.2 / 1.1 / 1.3 / 1.8 | 1.8 / 1.1 / 1.3 / 1.8 | 0.9 / 1.7 / 0.6 / 1.4 | 1.4 / 1.1 / 1.0 / 1.5 | 2.0 / 2.6 / 0.7 / 2.5 |
| B12/B16 (e) metric K, 4R | 2.2 / 1.6 / 1.2 / 2.0 | 1.8 / 1.6 / 1.2 / 1.9 | 0.9 / 1.1 / 0.4 / 1.2 | 1.3 / 0.9 / 0.9 / 1.6 | 1.9 / 1.8 / 0.3 / 2.2 |
| B12/B16 (f) hybrid, 4R | 2.0 / 1.6 / 1.2 / 1.9 | 1.6 / 1.6 / 1.2 / 1.9 | 1.3 / 1.1 / 0.4 / 1.6 | 1.2 / 0.9 / 0.9 / 1.6 | 2.2 / 1.8 / 0.3 / 2.5 |
| gNFW, 5 free (a), 4R | 1.7 / 1.0 / 0.4 / 0.4 | 1.3 / 1.0 / 0.4 / 0.4 | 1.7 / 1.0 / 0.8 / 1.6 | 0.9 / 0.5 / 0.3 / 0.3 | 2.7 / 2.0 / 1.1 / 2.8 |
| gNFW, 5 free (d), 4R | 1.0 / 0.7 / 0.4 / 0.3 | 0.6 / 0.6 / 0.4 / 0.4 | 2.4 / 1.3 / 1.1 / 2.3 | 0.7 / 0.5 / 0.5 / 0.9 | 3.5 / 2.2 / 1.4 / 3.5 |
| B12/B16 + outer power law (a), 4R | 3.2 / 1.1 / 1.9 / 2.1 | 2.8 / 1.1 / 1.9 / 2.1 | 0.3 / 1.1 / 1.2 / 0.4 | 2.3 / 0.5 / 1.6 / 1.6 | 1.1 / 2.0 / 1.2 / 1.2 |
| **B12/B16 + outer power law (d), 4R** | 2.8 / 2.5 / 1.6 / 2.3 | 2.4 / 2.5 / 1.6 / 2.2 | **0.2 / 0.8 / 0.5 / 0.4** | 1.9 / 1.6 / 1.2 / 1.8 | **1.3 / 0.4 / 0.4 / 0.9** |
| B12/B16 x spline (a), 4R | 2.5 / 1.5 / 1.3 / 1.8 | 2.1 / 1.5 / 1.3 / 1.8 | 0.7 / 0.7 / 0.5 / 0.7 | 1.6 / 0.6 / 1.0 / 1.4 | 1.7 / 1.5 / 0.3 / 1.8 |
| Measured mean stack (ceiling) | 3.0 / 1.9 / 1.1 / 1.9 | 2.6 / 1.9 / 1.1 / 1.8 | 0 / 0 / 0 / 0 | 2.2 / 0.9 / 0.7 / 1.4 | 1.1 / 1.0 / 0.4 / 1.3 |
| Stack, partial pooling in mass | 6.6 / 2.1 / 1.1 / 2.8 | 6.7 / 2.1 / 1.1 / 2.8 | 4.2 / 0.2 / 0.1 / 1.1 | - | - |
| 2R fit + edge matched to Y/N_e(<4R) | 3.9 / 2.7 / 1.6 / 3.3 | 3.5 / 2.7 / 1.6 / 3.3 | 1.3 / 1.4 / 0.5 / 1.8 | 3.0 / 2.1 / 1.2 / 2.9 | 1.4 / 1.6 / 0.2 / 1.3 |
| 2R fit + edge matched to an observable | 4.8 / 3.4 / 8.2 / 2.4 | 4.4 / 3.4 / 8.3 / 2.3 | 2.7 / 5.1 / 7.9 / 0.6 | - | - |
| Direct observable fit, per bin | 0.3 / 0.7 / 0.4 / 1.1 | 0.6 / 0.7 / 0.5 / 1.1 | 3.3 / 2.1 / 1.0 / 2.4 | - | - |
| Direct observable fit, joint P+E log-M trend | 0.2 / 1.0 / 0.3 / 0.8 | 0.3 / 1.0 / 0.3 / 0.8 | 3.2 / 2.1 / 1.3 / 1.9 | - | - |

**Where the numbers come from.**
- The DM quantiles of the scatter variants come from the maas run for B12/B16 (a) 2R/4R, (c) 2R, the stack and the power
  law (a). For the other rows they come from a local light PDF pass, which reproduces the maas quantiles to 3.2e-8.
- The scatter-aware rows were not cross-validated.
- For deterministic predictions the out-of-fold numbers differ from the in-sample ones by <= 0.5 points, with the same
  ranking. The out-of-fold combination also adds the fold-to-fold variance to the 1h terms, a small positive boost.

**How to read this table.**
- **No single mean profile per bin can reproduce the truth.** The scatter in the truth (about 2-4% of tSZ 1h) is a floor for
  any deterministic painting with the correct mean; the measured stack itself is 3.0 / 1.9 / 1.1 / 1.9% off. Deterministic
  methods that score better do so by biasing the mean profile upward.
- **gNFW (5 free)** has the best deterministic scores at ell <= 3000. But its fitted inner slopes (gamma = 1.0-1.7 for
  pressure, log M < 14.7) are extrapolations into the unresolved core.
  - Above ell ~ 8000 it overshoots: +4% at ell = 8000, +25% at 2 x 10^4 (`F6_spectra_ratio.png`).
  - Against the truth mean it is worse than B12/B16.
- **Direct observable fits** reach 0.2-1% on the observables they were fitted to, also out-of-fold. But:
  - they distort the profiles (pressure RMS inside 1R of 7% per bin, 17% with the trend);
  - the joint trend fit breaks observables it was not fitted to (yy 2-halo at ell = 100 -13%, mean y -6%).
  - Use them as a diagnostic, not for painting.
- **Truncation as a calibrated parameter.**
  - It fails for pressure: the B12 fit to 2R falls so steeply that even an edge at 12R does not reach the measured
    Y(<4R) in any bin.
  - For electrons, edges matched to N_e(<4R) lie at 3.5-8 R200c (rising with mass). They improve DM (2.2 -> 1.6%) but less
    than refitting to 4R.
- **Partial pooling** (a quadratic trend in log M per shell) over-smooths pressure (RMS inside 1R 12%); it is not useful as
  configured.
- **B12/B16 x spline.** The ridge prior (sigma = 0.3 per knot) dominates. With objective (d)'s unit-sum weights the
  correction stays at 0, so this variant equals B12/B16 (d).
- **Joint pressure-electron calibration.** The matched (4, 4) fit gives tSZ x DM 1.8%; the joint direct fit gives 0.8%, with
  the distortions above.
- **Scatter-aware calibration.**
  - Lognormal amplitudes recover most of the scatter deficit: with the 4R aperture the stack goes from 3.0 to 1.1% for tSZ.
  - They cannot follow the ell dependence of the scatter boost (measured +1.7% at ell = 100 vs +7.4% at ell = 8000 in yy 1h).
  - The 1R aperture is too small for tSZ, and the 4R aperture over-boosts models whose mean is already high (gNFW).
  - The measured second-moment correction applied to the power-law tail (d) model gives 0.3 / 0.8 / 0.7 / 0.5%. That is what
    a correct mean plus correct scatter achieves.

Figures: `figures/F4_methods_deterministic.png`, `F4_methods_lognormal_1R.png`, `F5_cv.png`, `F6_spectra_ratio.png`.

## 5. Trade-offs

- **tSZ-optimal vs DM-optimal.**
  - Against the truth mean, a B12 fit restricted to < 1R is the best tSZ choice (0.6%), but it costs DM 8% and kSZ 11%.
  - A fit to 4R is near-optimal for both (tSZ 0.9%, DM 0.6%).
  - With B12/B16 the 4R fit pays in the profile: the pressure RMS in 1-2R rises from 3.7% to 8.8%. tSZ hardly notices,
    because 1-2R holds only 4-15% of its signal.
- **Cut.** A shorter cut is never better against this truth (2R costs tSZ 5% and DM 22%). DM and low-ell kSZ still gain
  signal per unit ln r at 4R (`F1_importance_profiles.png`), so a cut beyond 4R would add signal this halo-based truth does
  not contain.
- **GLS vs the diagonal fit.** GLS fits the statistically best-constrained shells (0.3-1R) and degrades exactly the
  observables that depend on 1-4R.

## Painting recommendation for HalfDome

1. **Gas: all gas, satellites kept, recent-AGN gas removed** (the literature full-signal convention).
   - The AGN-kept control changes nothing (<= 0.04%).
   - If neighbours are painted as well, the A2 selections lower the target by 0.7-1.5% (tSZ, ell = 100), 2-7% (kSZ,
     ell = 100) and 1-3% (mean DM); the choices below do not change.
2. **Cut: hard spherical cut at 4 R200c**, for pressure and electrons alike. A flux-matched 0.05-dex edge is equivalent for
   tSZ but costs about 1% of the kSZ / DM-field C_ell.
3. **Fit radius = cut radius = 4 R200c.**
   - With the existing B12 / B16 forms and the current diagonal weighting, this halves today's errors: from
     4.1 / 3.2 / 2.2 / 3.9% to 2.2 / 1.1 / 1.3 / 1.8% (tSZ / kSZ / DM / tSZ x DM); out-of-fold 1.8 / 1.1 / 1.3 / 1.8%.
   - **This is the minimum change.** It needs no XGPaint change, only new fit parameters.
4. **Better profile: B12 / B16 plus an outer power law A (x/2)^-s, fitted to 4R with importance-diagonal weights.**
   - It matches the measured mean profile to <= 2-5% at every resolved radius, and the mean-profile observables to <= 0.8%.
   - When painting, damp the power-law term inside ~0.1 R200c. It creates an unresolved cusp (s <= 1.8, <= 0.2% of the
     flux), which the engine's frozen kernel also mishandles (see Numerics).
5. **Add halo-to-halo scatter:** a lognormal amplitude per painted halo, with its variance per 0.2-dex bin from FLAMINGO's
   per-halo Y(<4R) and N_e(<4R).
   - From 13.5 to 15.4: sigma_P^2 = 0.31 -> 0.007, sigma_E^2 = 0.045 -> 0.005, correlation <Y N>/<Y><N> = 1.10 -> 1.005.
   - With item 4 this gives 1.3 / 0.4 / 0.4 / 0.9%.
   - Without scatter, any correct mean profile leaves tSZ 1h low by 1.5-4% and the DM variance and Q99.9 low by 2%.
6. **Avoid:**
   - GLS weighting for these profiles;
   - cuts below 4R;
   - free-gamma gNFW without an inner-slope cap;
   - edges calibrated to a single observable;
   - direct observable fits.

The table numbers come from `results/tables/F_model_scores.csv`; for the scatter rows the DM quantiles come from
`results_local/scatter_pdf_errors.csv`, as noted above.

## Numerics: the Test D kernel cancellation

Test D found that the frozen kernel loses precision on the innermost nodes for q >= 0.2. In this test the kernel integrated
only the basis reference shapes (interpolated measured means, inner slope clipped to >= -1), the kernels/ P4 / E4 fits and
the B12 / B16 fits of the engine check. Model profiles entered only through their shell averages.

- **Transform-error indicator** (`td_fixcl` recipe; max over q in [1e-3, 1e3] of |u_frozen/u_stable - 1|):
  - truth-shaped basis: <= 1.6e-4 for pressure (13.5-13.7, measured inner slope -2.7 clipped to -1), <= 4.7e-4 for
    electrons;
  - P4 / E4: <= 5.1e-6.
  - All are below Test D's 1e-3 threshold. The error is confined to the innermost shell (0-0.025 R200c); elsewhere it is
    <= 1.2e-10.
- **Recomputation** with `D_literature/td_stable.py` (local, 2 workers, 153 s; production basis rebuilt at 25 multipoles):
  - every spectrum of every model and of the truth changes by <= 6.0e-6 (yy 2h) and <= 1.5e-6 otherwise;
  - every relative error in this report moves by <= 5.9e-6;
  - the frozen local rebuild reproduces the maas values to 4.9e-8 over 17,655 values.
  - **No number needed correction.**
- **If the fitted profiles were pushed through the frozen engine directly** (they were not), the indicator would be:
  - gNFW pressure: 0.31 (a) and 1.47 (d), in all bins below 14.7;
  - B12/B16 + outer power law: 0.04-0.17, in the top bins.
  - Anyone propagating these profiles through the engine should use the stable kernel.
- **The innermost shell** of the models holds <= 0.12% (measured) to <= 0.45% (gNFW) of the 4R pressure flux.

## Verification (all passed; `verification.json`)

| Check | Result |
|---|---|
| maas run | exit 0; 5,997 s at 4 workers, nice 19; peak tree RSS 8.7 GiB; 24 / 24 result checksums verified locally |
| In-basis machinery vs engine (kernels/jacobians.npz) | observables 1.3e-13, Jacobians 4.9e-13 of kO, DM quantiles 7.7e-12, quantile Jacobians 1.1e-12 |
| Same gate, full-size local pre-check (before the move to maas) | observables <= 5e-15, quantiles 9e-14 |
| Smoke run (2 bins, 50 haloes) | passed, 43 s |
| DM PDF mean / variance vs moment forms | <= 4.9e-8 / 2.2e-6 |
| Mecke quantile-Jacobian sum rules | deterministic <= 4.1e-6; per-halo importance (64-halo subsample) <= 1% |
| Identity: stack x measured second moments = truth | 2.2e-16 |
| CV: pooled folds reproduce the full truth | 2.5e-14 |
| Basis vs engine-exact for smooth fits | <= 5.3e-4 (ell <= 3000), 7.5e-4 (ell = 8000); sigmoid / hard ratios agree to 8e-4 |
| Numerics (Test D) | above; local rebuild vs maas 4.9e-8; scatter-PDF supplement vs maas 3.2e-8 |
| Fit acceptance | (a) / (c) / (d) / (e) / (f) 160 / 160 each; lambda scan 1120 / 1120; gNFW 39 / 40; spline 40 / 40; power law 30 / 40 (flags kept, values used) |
| Figures | 17 PNG + SVG, every PNG inspected after regeneration |

## Limits

- **Halo model.** The engine's redshift response is assumed (z = 0 calibration only); flat-sky Limber halo model with a linear
  2h term; the DM PDF is halo-only and compound-Poisson (no clustering, no IGM); kSZ 2h is approximate (only 1h is scored).
- **Calibrated mass range.** The calibrated bins (13.5-15.388) carry 97-99% of tSZ (ell <= 3000), but only 63-91% of kSZ 1h,
  58% of mean DM, 76% of the DM variance and 50-89% of the DM quantiles. Errors relative to the totals are diluted by the
  shared published fallback, which this test cannot calibrate.
- **Truth definition.** The truth is the per-halo profiles to 4R, and the importance of DM and low-ell kSZ is still rising at
  4R. Unresolved cores (inside 0.04-0.32 R200c, depending on mass) are the measured smoothed values; they matter at
  ell >~ 8000 and for Q99.9. All fits exclude them.
- **Sampling.** The top bins (N = 21 and 6) are noisy. Lognormal scatter is amplitude-only. The scatter-aware variants were
  scored in-sample only.

## Files

**Driver and modules** (`F_observables/`):
- `run_test_f.py` (`--smoke`, `--full`), `f_common.py`;
- `f0_profiles.py`, `f_kernels.py`, `f_basis.py`, `f_pdf.py`;
- `f1_validate.py`, `f1_truth.py`, `f1_importance.py`;
- `f_fit.py`, `f_methods.py`, `f2_run.py`, `f4_engine_check.py`, `f5_cv.py`;
- local post-processing: `f7_numerics.py`, `f9_scatter_pdf.py`, `f8_verification.py`, `make_figures.py`;
- `CLUSTER_RUN.json`.

**maas results** (`results/`, 38 MB, `SHA256SUMS`):
- `profiles.npz`, `truth.npz`, `importance_F.npz`, `models_main.npz`, `model_spectra.npz`;
- `*_summary.json`, `validation_V.json`, `engine_check.json`, `cv_summary.json`, `run_manifest.json`;
- `tables/`: long-format errors, scores, fit records, CV, controls, F1.

**Local** (`results_local/`): `numerics.json`, `numerics_per_model.csv`, `scatter_pdf.json`, `scatter_pdf_errors.csv`.
The `work/` directory with rebuilt bases is not checksummed.

**Figures** (`figures/`): 17 PNG + SVG, plus Markdown tables (`F2_best_*.md`, `F3_objectives.md`, `F4_methods_*.md`,
`F5_cv.md`).

**Other:** `results_smoke/` (smoke test), `local_precheck/` (the full-size local gate run before the move to the cluster),
`verification.json`, `SHA256SUMS`.

(Written by the coordinator from the agent's hand-back; the harness blocked the agent from writing `.md` files.)
