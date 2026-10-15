# Test G: physical outer components, calibrated per field (FLAMINGO L1_m9, z = 0)

Added on 2026-10-13 at the user's request. The user asked for:
- a separate calibration for the electron pressure (tSZ) and the electron density (kSZ, FRB DM);
- an outer profile component that is physically motivated, not "a random addition of parameters";
- a Lee22-style comparison of parametrisations by reduced chi^2, with as few parameters as possible.

Inputs are Test B's per-halo profiles, Test F's observable engine and basis, and the linear correlation function and
Tinker10 bias of the FLAMINGO cosmology (`inputs/xi_bias_z0.npz`, CLASS). Code, results and figures were made by a
delegated agent; this report was written from its hand-back and checked against `results/` (section 9).

**Full report** with every equation, the fitted profiles against the stacks per mass bin, and all tables: [`report/Test_G_report.pdf`](report/Test_G_report.pdf) (built by `report/make_report_figures.py` and `report/build_report.py` from `results/`).

**Run.**
- Everything ran on STRW maas through `cluster_tests.py run G`, at nice 19 with ionice and the load and memory guards.
- First attempt: 4 workers. It was stopped during `evaluate` at the user's request; the completed stages (inputs, the
  basis, prepare, parity, fits) and the finished evaluate pass p1 were kept.
- Second attempt: `--stages evaluate,selection,cv` with 8 workers; 1,490 s, exit 0, peak tree RSS 7.0 GiB.
- Load at launch was 24-25 on 48 threads, and at least 175 GiB stayed free.
- Figures and tables were made locally from `results/` (`make_figures.py`). `verification.json`: `all_passed: true`.

## 1. Plain answers

1. **Separate calibration per field: done.** Pressure is chosen on tSZ, electrons on kSZ plus the DM PDF, and tSZ x DM
   is the cross-check.
   - The outer physics that works differs by field:
     - **Electrons:** a bound gNFW with a splashback-like transition, plus a parameter-free linear 2-halo term (E2).
     - **Pressure:** P = n k T needs an outer-gas temperature that depends on mass (P2m); a self-similar temperature fails.
   - Both fields still prefer R_fit = R_cut = 4 R200c, objective (a) and S1 scatter.
     - Electrons need the 4R cut: a 3R cut costs 8-10% in kSZ/DM, a 2R cut 20-24%.
     - Pressure is less sensitive: R_fit = 3 or 4 is equivalent (0.44-0.52% tSZ with S1), but R_cut must stay at 4R.
2. **Is the two-component profile Lee22's? No.**
   - **Lee22:** one gNFW (alpha = 1, gamma = -0.3) for P and n_e, fitted over 0.04-1.34 R200c in TNG300. Every parameter
     is a broken mass power law with concentration (and redshift) terms. There is no outer component.
   - **Test F:** B12/B16 + A (x/2)^-s was an empirical addition with two free parameters per bin.
   - **Test G:** replaces it with physically motivated terms and tests their assumptions:

     | Term | Verdict | Evidence |
     |---|---|---|
     | Linear 2-halo term (electrons) | supported, with a mass trend | stacked 3-4R electrons within +-10% for 10^13.9-14.9; 16-26% above it for 13.5-13.9, 15-19% below it for >= 14.9 |
     | Splashback-like transition | supported | x_t = 2.1-2.4 R200c (1.3-1.4 R200m) with gamma_t = 5.1; the steepest log slope moves inward with mass |
     | Global infall term (E3) | not supported | the per-bin version finds an outer excess (A_c = 1.3-2.7, slope 1.6-2.6) only below 10^14.1 and none above 10^14.5; globally the slope runs to its bound (3.9), where the term acts as an inner cusp, not infall |
     | Self-similar outer temperature (P2) | fails | 1.3-4.4 times too cold against the measured T_e/T200 at 3-4R; 2-4R residuals 17% |
     | Mass-dependent outer temperature (P2m) | works | tau proportional to M^-0.43 matches the measured T_e at 3-4R within -35% to +12% |
     | Accretion-shock cutoff | not detected inside 4R | pressure shows a transition only in P2m, at x_t ~ 2.0 R200c (the splashback scale, not the expected shock radius of 2.5-5 R200c) |

3. **Scatter: S1 stays.**
   - S2 was tested: an inner amplitude from per-halo integrals inside 1R, and an outer one from 1.5R to R_cut.
   - Its inputs confirm the motivation: sigma_in = 0.07-0.19 (P), 0.03-0.19 (E); sigma_out = 0.33-1.00 (P),
     0.13-0.34 (E); rho(in, out) = -0.36 to +0.32, mostly |rho| < 0.2.
   - But S2 under-predicts the 1-halo boost: P2m tSZ 1.43% with S2 against 0.52% with S1; E2 kSZ/DM 1.02/1.22% against
     0.32/0.50%.
   - The likely reason: Y(<1R) averages over the anti-correlated inner shells (Lee22's two blocks), so sigma_in
     underestimates the core scatter. The shell correlation matrices in the gallery
     (`observable_residual_maps_L1_m9_0077_20261012/shell_correlations_20261013/`) show this. Concentration moves
     pressure between the core and 0.5-1.5 R200c while nearly conserving the total, so the core's scatter is a change
     of shape, not of amplitude. A core-shape (pivot) scatter component would be the physical next step; it is not
     tested here.

## 2. Normalisations (checked against code and data, `results/prepare.json`)

- **Electron units.** E = n_e mu_XGP / (rho_c f_b), with mu_XGP = (m_e + m_p 2/1.76)/0.9 = 1.2632 m_p (XGPaint's 1/0.9
  factor).
  - The ratio of E to the gas density G is 1.076-1.099 inside 1R and 1.092-1.097 at 2-4R; ionised primordial gas gives
    1.107.
  - So the cosmic mean electron density is 1.094 Omega_m = 0.3348 in E units, not Omega_m = 0.306, and the 200 rho_c f_b
    equivalent is about 219, not 200. Stars are neglected.
- **Pressure units.** With P200 = G M (200 rho_c) f_b / (2R) and kT200 = 0.59 m_p G M / (2R) (Test B),
  P/P200 = 0.4671 (E/200)(T/T200). Measured: 0.4685-0.4727, i.e. T_e/T_M = 1.003-1.012.
- **Effect on the brief's numbers.** Its 2-halo ratios and implied temperatures are reproduced to within 2% (e.g.
  4.51/2.31/1.38 for 13.5-13.7), but with the corrected units:
  - the 2-halo ratios are 0.915 times the brief's;
  - the "implied" temperatures are 1.96 times the brief's. They are not gas temperatures either, because 1.5-3R holds
    1.5-4 times the 2-halo density.
- **Measured T_e/T200** (from the electron-weighted T, Test B), 10^13.6 to 10^15.35:

  | Band | 13.6 | 15.35 |
  |---|---|---|
  | 1.5-2R | 0.43 | 0.30 |
  | 2-3R | 0.35 | 0.17 |
  | 3-4R | 0.34 | 0.10 |

## 3. Models and parameter counts (10 bins; per-bin + global)

Notation:
- f_tr = [1 + (x/x_t)^4]^(-gamma_t/4), the transition;
- n2h = Ebar [1 + b(M) xi_lin(x R)], the bin mean over haloes;
- D = x^4 / (x^4 + 0.1^4), which damps G's own power-law terms inside 0.1 R200c.

**Electrons**

| Model | Form | Free parameters |
|---|---|---|
| E0 | B16 | 30 |
| E1 | B16 + A (x/2)^-s (s <= 3) | 50 |
| E2g6 | B16 f_tr + n2h, gamma_t = 6 | 40 |
| E2gf | as E2g6, gamma_t global | 41 |
| E3 | E2gf + Ebar A_inf (b/b_piv)^a (x/2)^-s_inf D | 44 |
| E3pb | per-bin A_c, s_c (diagnostic) | 61 |

**Pressure**

| Model | Form | Free parameters |
|---|---|---|
| P0 | B12 | 30 |
| P1 | B12 + A (x/2)^-s | 50 |
| P2g6 | B12 f_tr + 0.467/200 n2h tau0 (x/2)^-q D, gamma_t = 6 | 42 |
| P2gf | as P2g6, gamma_t global | 43 |
| P2i | P2gf with E3's infall in n_out (E3 values fixed) | 43 |
| P2s | P2gf with tau [1 + (x/x_sh)^8]^-1 | 44 |
| P2m | P2gf with tau0 (M/10^14.5)^-p_M | 44 |

**Ownership (p_keep).** Physical models are fitted to all gas. Painting uses the model times
p_keep(x, M) = 1 - f(M) / [1 + (x/x_h)^-n], fitted per field to the stacked A2-ii/all ratio.
- Pressure: f = 0.82, x_h = 3.44, n = 6.9; it matches the stacks to 1-4% per bin (22% in the N = 21 bin).
- Electrons: f = 0.50, x_h = 3.29, n = 6.4; it matches to 0.3-1.4%.
- Direct A2-ii fits were also run. E2 cannot fit them (2-4R RMS 42%), because the partition removes neighbour gas below
  the 2-halo level.

## 4. Pressure results (all-gas fits, objective a, R_fit = 4)

**Profile and information criteria.** RMS in %; dBIC with a common sigma_ref per bin (median 0.08%), n = 252 shells.

| Model | k | RMS <1R / 1-2R / 2-4R | sigma_d | dAIC / dBIC |
|---|---|---|---|---|
| P2m | 44 | 1.3 / 2.4 / 7.9 | 0.63% | 20 / 0 |
| P1 | 50 | 2.1 / 3.0 / 4.0 | 0.48% | 0 / 0.8 |
| P2g6 | 42 | 1.4 / 4.3 / 17.7 | 5.0% | 153 / 126 |
| P2gf | 43 | 1.8 / 4.1 / 17.5 | 5.4% | 176 / 152 |
| P2s | 44 | 1.8 / 4.0 / 17.4 | 5.5% | 195 / 174 |
| P0 | 30 | 3.0 / 8.8 / 28.2 | 14.6% | 942 / 873 |

**tSZ family RMS** at the painting target (model times p_keep against the A2-ii truth), R_cut = 4, in %:
- det: one profile per bin;
- S1 / S2: with scatter;
- vs mean: the deterministic prediction against the truth mean.

| Model | det | S1 | S2 | vs mean | CV out-of-fold S1 |
|---|---|---|---|---|---|
| P2m | 3.19 | 0.52 | 1.43 | 0.30 | 0.38 |
| P1 (a) | 3.15 | 0.59 | 2.25 | 0.36 | 0.44 |
| P1 (d) | 2.77 | 0.58 | | | 0.82 |
| P2gf | 3.01 | 0.58 | 2.09 | | |
| P0 | 2.04 | 1.31 | 1.24 | 1.06 | 1.58 |

- Against the all-gas truth: P2m S1 0.94%, P1 (d) S1 1.26% (Test F reported 1.3%).
- Best overall: objective (d) at R_fit 3 / R_cut 4 gives 0.44-0.49% for every P2 variant, a difference within the CV
  scatter.

**Physical parameters (all gas, a, 4R)**
- **P2m:** gamma_t = 2.70; tau0 = 0.35 at x = 2 and 10^14.5 Msun; q = 1.44; p_M = 0.43, so T_out is proportional to M^0.24.
  - tau at 3.5 R200c is 0.38 / 0.14 / 0.067 against the measured T_e/T200 of 0.34 / 0.18 / 0.10 (10^13.6 / 14.6 / 15.35).
  - x_t = 1.96-2.12 R200c up to 10^15.1 (1.08 at 15.1-15.3, 3.09 at 15.3-15.5).
  - The temperature law is checked only at 3-4R (`figures/G3_outer_temperature.png`). Inside 2R its power law
    (q = 1.44) runs hotter than the measured gas, by up to about 3 times at R200c in groups, and the outer term carries
    9-32% of the pressure at 0.5 R200c and 15-35% at R200c (most in the lowest bins). The profile is accurate, but inside
    R200c the bound/outer split is a fit decomposition, not a physical one.
- **P2gf:** tau0 = 0.115, q = 0.69, so tau(3.5) = 0.078 at all masses: it fails the temperature check. Its x_t reaches the
  20 R200c bound below 10^14.9, i.e. no transition inside 4R.
- **P2s:** x_sh = 5.0 R200c, outside the fitted range; with R_fit = 3 it runs to the bound. The shock is not detected.

## 5. Electron results (all gas, objective a, R_fit = 4)

**Profile and information criteria.** Median sigma_ref 1.5%.

| Model | k | RMS <1R / 1-2R / 2-4R | sigma_d | dAIC / dBIC |
|---|---|---|---|---|
| E3 | 44 | 1.0 / 1.9 / 10.0 | 2.3% | 0 / 0 |
| E2gf | 41 | 1.2 / 2.0 / 10.0 | 2.4% | 28 / 17.6 |
| E2g6 | 40 | 1.3 / 2.1 / 9.9 | 2.5% | 44 / 29.8 |
| E1 (Test F) | 50 | 2.8 / 3.5 / 6.2 | 2.0% | 249 / 270 |
| E1 (d) | 50 | 3.6 / 3.8 / 3.5 | | |
| E0 | 30 | 8.5 / 13.7 / 21.6 | 13% | 4391 / 4341 |

E3's best BIC comes from its unphysical inner cusp (section 1), so it is not adopted.

**kSZ / DM family RMS** at the painting target, R_cut = 4, in %:

| Model | det | S1 | S2 | vs mean | CV S1 |
|---|---|---|---|---|---|
| E2gf | 1.96 / 1.55 | 0.32 / 0.50 | 1.02 / 1.22 | 0.49 / 0.56 | kSZ 0.33 |
| E2g6 | 2.01 / 1.56 | 0.26 / 0.53 | | | 0.25 / 0.52 |
| E1 (d), Test F's recommendation | 2.31 / 1.55 | 0.37 / 0.54 | | | 0.37 / 0.54 |
| E3 | 1.18 / 1.00 | 1.02 / 0.43 | | | |
| E0 | 1.15 / 1.36 | 2.70 / 0.82 | | | |

**Robustness at other fit ranges.** E1 cannot be fitted below R_fit = 4: its free power law diverges in the unresolved
core (deterministic DM errors 77-387% for R_fit 1-3 at R_cut 4, painting target). E2 stays at 1.6-2.2% DM for R_fit 2-4.

**Physical parameters (all gas, a, 4R)**
- **E2gf:** gamma_t = 5.10.
  - x_t = 2.40 / 2.30 / 2.25 / 2.30 / 2.38 / 2.37 / 2.24 / 2.13 / 1.82 / 2.69 R200c from 10^13.6 to 10^15.35, i.e.
    1.07-1.60 R200m. That is inside the 1.0-1.5 R200m splashback band, except the N = 6 bin (R200m/R200c = 1.67-1.70).
  - The steepest log slope moves from 2.28 to 1.20 R200c (1.36 to 0.71 R200m) while nu runs from 1.4 to 3.6.
- **Measured/predicted 2-halo ratio at 3-4R:**
  - all gas: 1.26, 1.16, 1.07, 1.02, 1.02, 0.98, 0.91, 0.85, 0.81, 0.85;
  - A2-ii: 1.03 falling to 0.50-0.59.
- **E3 (global infall):** A_inf = 0.011, a = -2.3, s_inf = 3.9 (at the bound).
- **E3pb (per-bin infall):** A_c = 2.75 / 2.35 / 1.25 / 0.29 / 0.40 / 0.10 / ~0 above 10^14.7;
  s_c = 1.6 / 2.1 / 2.6 / 3.1 / 2.9 / 3.2. Not universal.

**tSZ x DM cross-check** (painting target, 4/4, S1):
- P2m x E2gf 0.57% (det 2.18%, vs mean 0.39%); P1 x E1 (d) 0.50%; P0 x E0 2.49%.
- Against the all-gas truth: P2m x E2gf 1.00%, P1 x E1 (d) 0.89% (Test F reported 0.9%).

## 6. Selection stage (Lee22-style ladder; fits to the stacks only)

**Setup.**
- The inner profile is Lee22's gNFW for both fields.
- Mass laws: per bin (PB), power law (PL), broken power law (BPL, a common break M_cut, free or Lee22's value),
  + concentration, + gamma.
- Outer options: O0 none; O1 the physical outer term (E: 0 parameters; P: tau0, q); O2 = O1 plus x_t as a mass power law
  (+2 parameters).
- Likelihood: bootstrap Cov[ln mean] with shrinkage, plus a common floor sigma_d set by the best per-bin reference (PB,
  or PBcell on the concentration-split cells) at chi^2/nu = 1. Also scored: the Lee22 convention and spatial CV.
- Hartlap factors: 0.76-0.98 for the mass cells (the N = 21 and 6 cells are excluded); 0.03-0.92 for the concentration
  cells.
- Floor sigma_d on the mass cells: P 0.28 / 0.26 / 0.49 / 1.12% and E 1.01 / 0.63 / 0.70 / 1.93% at R_fit 1.34 / 2 / 3 / 4.

**Lee22's chi^2 convention.** Lee22's best models on our concentration-split cells give chi^2/nu = 3.7 (P) and 5.4-5.5
(E) with the covariance of the mean, close to their 3.3 and 5.1. The per-halo-scatter convention gives 0.05-0.06. So
Lee22 used the covariance of the mean; that is the "L22" column of the tables.

**Selected models (mass cells).** The rule: the fewest parameters with dBIC <= 6, and CV within the best model's fold
scatter.
- R_fit 1.34: BPLall+O0, 10 parameters; chi^2/nu (floor) 1.43 (P) and 1.39 (E).
- R_fit 2-4: BPLall+O2, 14 parameters for P and 12 for E; chi^2/nu 1.32 (P) and 1.26 (E) at 4R.
- **Break mass.** The fitted break sits at 10^14.10-14.23 Msun for R_fit 2-4 (14.12 for P and 14.13 for E at 4R). Lee22's
  values convert to 10^13.77-13.78 Msun, about 0.35 dex lower.
- Lee22's own choices (a break in amplitude for P, in x_c and beta for E, M_cut fixed) are rejected by dBIC 190-340 at
  1.34R.

**Parsimony ladder at 4R** (best model at each k; chi^2/nu with the floor; `figures/G9_ladder_{P,E}_mass.png`):
- P: 15.5 (PL1, k 4) -> 6.1 (PL2+O0, 5) -> 3.65 (PL3+O1, 8) -> 1.86 (BPLs+O1, 11) -> 1.42 (BPLall+O1, 12)
  -> 1.32 (BPLall+O2, 14, dBIC 0) -> PB+O2 (34, dBIC 13.8).
- E: 20.7 (PL1+O1, 4) -> 3.92 (PL3+O1, 6) -> 2.09 (BPLa+O1, 8) -> 1.39 (PL3gM+O2, 10, dBIC 22.8)
  -> 1.26 (BPLall+O2, 12, dBIC 0) -> PB+O2 (32, dBIC 29.3).

**Accuracy against per-bin fits** (painting target, R_cut 4, R_fit 4; det / S1 / vs mean, in %):

| Field | Model | k | tSZ, or kSZ / DM |
|---|---|---|---|
| P | BPLall+O2 | 14 | 3.25 / 1.14 / 0.89 |
| P | PB+O2 | 34 | 4.61 / 2.12 / 2.07 |
| P | PL3+O2 | 10 | 5.86 / 8.77 / 8.80 |
| P | PL1+O2 | 8 | 7.43 / 4.79 / 4.73 |
| E | BPLall+O2 | 12 | det 1.13 / 1.45; S1 1.26 / 0.50 |
| E | PB+O2 | 32 | det 1.34 / 1.42; S1 1.22 / 0.46 |
| E | PL3+O2 | 8 | det 0.74 / 1.91 |
| E | PL1+O2 | 6 | det 1.23 / 2.22 |

- So the minimal BPL models lose nothing against the per-bin fits of the same likelihood.
- Fits restricted to 1.34R and extrapolated miss the outskirts: E kSZ about 12%, DM about 7.5%; P tSZ 3.4% det.
- The selection-stage fits use the GLS likelihood, where the noisy pressure outskirts carry little weight, and their
  pressure outer term uses a self-similar tau (the P2gf limitation). They are therefore less accurate than the per-field
  recommendations below (P2m: 0.52%, E2gf: 0.32/0.50% with S1).

**Concentration terms.**
- On concentration-split cells they matter. E at 4R: PL3+O2 5.1 -> PL3c3+O2 2.5 -> Lee22-best+O2 1.7. P: 7.6 -> 2.6 -> 2.1.
- No smooth model reaches dBIC <= 6 there (all >= 56), so the rule falls back to the per-cell reference.
- For painting they are not worth their parameters. HalfDome has no concentrations, and the concentration models
  painted at the bin-mean concentration describe the mass-bin stacks worse than the mass-only fit (E inside 1R: 9.7% vs
  4.9%; P at 2-4R: 30.5% vs 23.4%). Their effect is scatter, which S1 already models.

**Recommended minimal Lee22-style models:** BPLall+O2 at 4R (P 14 parameters, E 12), or BPLall at Lee22's 1.34R
(10 parameters).

## 7. Recommendation per field

**Pressure: P2m.** B12 times f_tr with x_t per bin, plus 0.467/200 n_2h tau0 (x/2)^-q (M/10^14.5)^-p_M; 44 parameters.
Fit to all gas at R_fit = R_cut = 4, objective (a), hard 4R cut, S1 scatter, painted times p_keep.
- Errors: tSZ 0.52% (S1; CV 0.38%), 0.30% vs the truth mean; profile RMS 1.3 / 2.4 / 7.9%; sigma_d 0.63%.
- It ties with Test F's P1 on BIC with 6 fewer parameters, and is more accurate inside 2R.
- It passes the P = n k T check only with the mass-dependent outer temperature. The self-similar version fails, and no
  shock is seen within 4R.
- The minimum change remains P0 refitted at 4R (1.31% with S1).

**Electrons: E2gf** (or the equivalent E2g6). B16 times f_tr plus the parameter-free linear 2-halo term; 41 parameters.
Same fit, cut, objective, scatter and painting.
- Errors: kSZ 0.32% and DM 0.50% (S1; CV kSZ 0.33%), 0.49 / 0.56% vs the truth mean; profile RMS 1.2 / 2.0 / 10.0%;
  sigma_d 2.4%.
- Against Test F's E1 (d) (0.37 / 0.54%): E2 is better inside 2R and robust at any R_fit; E1 follows 2-4R better (3.5% vs
  10%).
- The 2-4R gap is the 2-halo normalisation, off by -19% to +26% across mass.
- E3 is not adopted: its infall term is unphysical.

**Fewest parameters.** BPLall+O2 (P 14, E 12) costs tSZ 1.14% instead of 0.52% and kSZ 1.26% instead of 0.32%, with DM
unchanged (0.50%). Combining the smooth BPL mass law with the P2m outer temperature was not tested.

## 8. Caveats

- **2-halo term.** Stars are not removed from Ebar, which raises it by a few percent. It uses the linear xi, Tinker10 bias
  at Delta = 200/Omega_m evaluated at M200c, bin-mean templates and no halo exclusion inside R200c.
- **p_keep** is a population average.
- **z = 0 only.**
- **DM quantiles.** Exact DM quantiles were not computed for R_cut < 2 (the DM score there uses the mean and variance
  only). The S1/S2 PDFs were computed only for the listed subsets.
- **Bounds and damping.** E1/P1 s_env is bounded at 3 (Test F used 10; F's 4R fits have s <= 1.8). G's infall and tau
  terms are damped inside 0.1 R200c.
- **Selection stage.** See section 6: the GLS likelihood and the self-similar pressure tau.
- **Convergence.** 2 of 161 joint fits hit max_nfev (P2m at R_fit 1, objective d). Many R_fit <= 1.5 fits sit on bounds
  by construction.
- **Small samples.** The top bins have N = 21 and 6.
- **Restart.** The two attempts give identical results by construction (the stages are deterministic and read the saved
  outputs).

## 9. Runs and verification

| Run | Workers | Stages and wall time | Peak tree RSS |
|---|---|---|---|
| Attempt 1 (stopped) | 4 | kernels 521 s, prepare 2 s, parity 33 s, fits 162 s | 4.67 GiB |
| Attempt 2 | 8 | evaluate 951 s, selection 382 s, cv 154 s; total 1,490 s; exit 0 | 7.03 GiB |

- **Smoke test:** 168 s on 1 worker, 2.9 GB.
- **Parity on maas:**
  - the rebuilt basis and the DM quantiles equal Test F's bitwise (0.0);
  - E0/P0 shells equal F's grid_a models (0.0, 50 configurations), and the family RMS equals F's scores (0.0);
  - E1/P1 reproduce F's flex_env_a to 6e-6 / 4e-4.
- **Checks in this session:**
  - `verification.json`: `all_passed: true`;
  - `SHA256SUMS` (100 entries) and `results/SHA256SUMS` verify;
  - the quoted scores, information criteria, global parameters, the selection ladder and the break masses were compared
    with `results/tables/` (G_scores, G_aic_bic, G_fit_globals, G_selection_ladder, G_selection_models) and match.

## 10. Files

- **Code:** `run_test_g.py`, `g_common.py`, `g_physics.py`, `g_models.py`, `g_fit.py`, `g_prepare.py`, `g_parity.py`,
  `g_pdf.py`, `g_fits.py`, `g_evaluate.py`, `g_selection.py`, `g_cv.py`, `make_figures.py`, `write_cluster_run.py`.
- **Run records:** `CLUSTER_RUN.json` (8 workers plus the restart note), `verification.json`, `verification_smoke.json`,
  `SHA256SUMS`, `attempts/attempt1_4workers_20261013/`.
- **Results:** `results/` (33 MB from maas, with its own SHA256SUMS). Tables in `results/tables/`: `G_scores`,
  `G_errors_long`, `G_cross_scores`, `G_fit_records`, `G_fit_globals`, `G_aic_bic`, `G_cv_scores`,
  `G_cv_fold_parameters`, `G_selection_models`, `G_selection_ladder`, `G_selection_cells`, `G_selection_observables`.
- **Report tables and smoke products:** `tables/G_report_scores_{P,E}.csv`; `results_smoke/`, `figures_smoke/`,
  `tables_smoke/`.
- **Figures** (`figures/`, PNG + SVG):
  - `G1_normalisation`, `G2_two_halo_ratio`, `G3_outer_temperature`;
  - `G4_residuals_{E,P}_main_a_Rf{2,4}`, `G4_residuals_{E,P}_a2ii_a_Rf4`;
  - `G5_transition_vs_mass`, `G6_grid_{P_tSZ,E_kSZ,E_DM}`, `G7_scatter_{paint,all}_a_Rf4`, `G8_cv`;
  - `G9_ladder_{P,E}_{mass,massc}` (the parsimony ladders), `G10_pkeep`.
- **Changed outside `G_physical_outer/`:** `cluster_tests.py` (one TESTS entry, `'G': ('G_physical_outer', 'run_test_g.py')`)
  and `cluster_mirror_manifest.json` (rewritten by `cluster_tests.py mirror`, as on every mirror). On maas only our
  runtime tree was written; it has not been deleted.
