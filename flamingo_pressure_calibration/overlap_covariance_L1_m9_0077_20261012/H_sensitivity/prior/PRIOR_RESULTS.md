## What earlier tests already established (prior results for the sensitivity and fitting study)

All earlier products were only read. The scripts in `H_sensitivity/prior/` (p1-p7) rebuilt every number below from stored arrays and tables. Every quoted value is listed with its source in `H_sensitivity/prior/tables/prior_key_numbers.csv`.

**Common conventions**
- The truth is FLAMINGO L1_m9 at z = 0: per-halo profiles to 4 R200c, all gas, recently AGN-heated gas removed. It includes halo-to-halo scatter.
- x = r/R200c (dimensionless).
- Calibrated masses are 10^13.5-10^15.388 Msun. Below that, every model and the truth use the published B12/B16 profiles ("fallback").

### 0. Definitions used throughout

**Normalised importance** (Test F, `F_observables/f1_importance.py:80`):

$$I_{b,s}(O)=\frac{1}{k\,O}\,\frac{\partial O}{\partial \ln F_{b,s}}$$

- $O$: an observable (tSZ $C_\ell^{yy}$, kSZ 1-halo $C_\ell$, mean DM, ...) evaluated at the truth.
- $F_{b,s}$: the 3D profile (pressure $P_e/P_{200}$ or electron density $n_e\mu_{\rm XGP}/(\rho_c f_b)$, dimensionless) in mass bin $b$ and radial shell $s$.
- $k$: 2 for quadratic observables (auto spectra, DM variance), 1 for linear ones (means, quantiles, each side of a cross spectrum).
- With this $k$ the map sums to 1 over $(b,s)$. It is a local derivative, not an error budget.
- The DM-quantile maps (Mecke formula, 64-halo subsample) sum to 0.990-1.002, so their band shares here are renormalised. They differ from Test F's table by at most 0.34 points (`p1_summary.json`).

**Cumulative importance and enclosing radius** (`H_sensitivity/prior/prior_common.py: cumulative_radius`):

$$C_O(x)=\frac{\sum_{s:\,x_{s+1}\le x}I_s}{\sum_s I_s},\qquad R_f=\min\{x:\ C_O(x)\ge f\}$$

- $I_s=\sum_b I_{b,s}$ is summed over mass bins (or taken in a single bin).
- $x_s, x_{s+1}$ are the shell edges in R200c.
- $R_f$ (for example R90) is the radius in R200c that encloses a fraction $f$ of the sensitivity. It is interpolated linearly inside a shell.

**Importance density**, $dI/d\ln x|_s = I_s/\ln(x_{s+1}/x_s)$, is the sensitivity per logarithmic radius. Its value in the last shell (3.63-4 R200c) divided by its peak says whether the sensitivity is still at its maximum at the 4R data edge (1.00 = yes).

**First-order truncation:** $\Delta O/O\approx -k\sum_{s:\,x_s\ge R_{\rm cut}}I_s$. This is the linearised loss from a hard cut at $R_{\rm cut}$ (convention of `stackcore.observable_impact`).

**Family score** (`F_observables/f2_run.py:60`, `G_physical_outer/g_evaluate.py:53`):

$$\mathrm{RMS}_{\rm fam}=\Big[\tfrac{1}{n}\sum_{O\in{\rm fam}}\epsilon_O^2\Big]^{1/2},\qquad \epsilon_O=O_{\rm model}/O_{\rm truth}-1$$

- tSZ = $C_\ell^{yy}$ (1-halo + 2-halo) at ell = 100, 300, 1000, 3000.
- kSZ = $C_\ell^{\rm kSZ,1h}$ at the same ell.
- DM = mean, variance and the 16 / 50 / 84 / 97.5 / 99.9% quantiles of the halo DM PDF (source redshift $z_s=1$).
- tSZxDM = $C_\ell^{y\,{\rm DM}}$.
- "vs mean" scores the deterministic prediction against a truth built from the stacked mean profile only. It isolates the fit and cut effect from scatter.

### 1. Where the signal comes from: radial shares at the measured truth

Source: `F_observables/results/importance_F.npz`, 35 shells, all masses; recomputed in `tables/imp_bands_F.csv` and `tables/imp_radii_F.csv`. Band edges 0.1 / 0.25 / 0.5 are the nearest F-grid edges (0.105 / 0.234 / 0.519).

| Observable | <0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 | R50 / R90 / R95 [R200c] | edge/peak | calibrated share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tSZ yy, ell=100 | 3.7 | 13.9 | 33.2 | 30.8 | 10.8 | 3.7 | 2.5 | 1.4 | 0.51 / 1.34 / 1.81 | 0.10 | 99% |
| tSZ yy, ell=1000 | 6.5 | 20.0 | 37.7 | 25.7 | 6.9 | 1.9 | 1.0 | 0.3 | 0.39 / 1.00 / 1.30 | 0.02 | 98% |
| tSZ yy, ell=3000 | 7.4 | 24.4 | 42.8 | 21.1 | 3.6 | 0.6 | 0.1 | 0.0 | 0.33 / 0.78 / 0.97 | 0.00 | 97% |
| tSZ yy, ell=8000 | 12.3 | 34.4 | 42.4 | 11.0 | 0.3 | -0.2 | -0.2 | 0.0 | 0.25 / 0.54 / 0.65 | 0.00 | 91% |
| mean y | 1.5 | 6.8 | 22.1 | 29.9 | 16.8 | 8.9 | 8.8 | 5.3 | 0.81 / 2.38 / 3.04 | 0.40 | 70% |
| kSZ 1h, ell=100 | 0.8 | 3.9 | 15.3 | 25.8 | 18.1 | 11.5 | 14.2 | 10.4 | 1.10 / 3.04 / 3.50 | 0.83 | 91% |
| kSZ 1h, ell=1000 | 1.1 | 5.1 | 18.7 | 28.7 | 18.4 | 10.8 | 11.3 | 5.8 | 0.93 / 2.55 / 3.12 | 0.39 | 83% |
| kSZ 1h, ell=3000 | 1.6 | 7.3 | 25.0 | 33.3 | 17.8 | 8.4 | 5.8 | 0.9 | 0.72 / 1.76 / 2.17 | 0.01 | 63% |
| mean DM | 0.5 | 2.3 | 9.7 | 19.5 | 17.3 | 13.6 | 20.3 | 16.8 | 1.53 / 3.40 / 3.70 | **1.00** | 58% |
| DM variance | 2.9 | 10.5 | 28.3 | 29.7 | 13.7 | 6.4 | 5.7 | 2.7 | 0.62 / 1.85 / 2.49 | 0.18 | 76% |
| DM Q16 | 0.0 | 0.0 | 0.1 | 4.7 | 13.6 | 17.4 | 33.2 | 31.0 | 2.42 / 3.68 / 3.84 | **1.00** | 50% |
| DM Q50 | 0.0 | 0.1 | 2.9 | 15.2 | 19.9 | 16.7 | 24.9 | 20.3 | 1.85 / 3.50 / 3.75 | **1.00** | 54% |
| DM Q84 | 0.3 | 2.3 | 11.6 | 25.4 | 18.7 | 12.5 | 16.5 | 12.6 | 1.25 / 3.20 / 3.59 | 0.99 | 59% |
| DM Q99.9 | 6.1 | 21.6 | 35.6 | 17.2 | 7.0 | 4.0 | 4.9 | 3.6 | 0.37 / 1.80 / 2.69 | 0.27 | 89% |
| tSZxDM, P side, ell=1000 | 4.2 | 15.4 | 35.5 | 29.5 | 9.7 | 3.2 | 1.9 | 0.5 | 0.47 / 1.21 / 1.58 | 0.03 | 96% |
| tSZxDM, E side, ell=1000 | 2.8 | 11.0 | 30.0 | 31.0 | 13.6 | 5.8 | 4.3 | 1.4 | 0.59 / 1.61 / 2.10 | 0.07 | 95% |

Band shares are in %. "edge/peak" is $dI/d\ln x$ in the 3.63-4R shell divided by its peak.

**Reading the table**
- tSZ is not core-dominated: the peak is at 0.25-0.5 R200c, and <0.1R holds 4-7% for ell ≤ 3000.
- DM (mean, Q16-Q84) and low-ell kSZ are outskirt-dominated, and their sensitivity is still maximal at the 4R edge. A truncation radius for DM therefore cannot be shown to converge from the measured truth, which ends at 4R by construction.
- At high ell, outer shells carry negative importance (phase). This is why some shares are below 0 and inside-1R shares can exceed 100%.

**Finest stored radial resolution.** The kernels' 67-shell x200 Jacobians at the 4R B12/B16 fits (deterministic; `kernels/jacobians.npz`, recomputed in `tables/imp_x200_kernels.csv`) agree with the 35-shell truth bands to ≤ 4.2 points. For example:
- tSZ ell=1000: R90 = 1.02.
- kSZ ell=100: R90 = 2.85.
- mean DM: R90 = 3.34, peak in the last shell.

They differ in one place: at the fits the kSZ ell=100 density turns down beyond 2R (edge/peak 0.57), while in the all-gas truth it turns up again at 3-4R (`F_observables/figures/F1_importance_profiles.png`). That upturn is neighbour gas, which A2-ii removes. No truth-based importance on the 67-shell grid exists.

**Mass dependence** (`tables/imp_radii_F.csv`, `figures/prior_R90_by_mass.png`). R90 in R200c, from 10^13.5-13.7 to 10^15.3-15.5:

| Observable | R90 range |
|---|---|
| tSZ ell=1000 | 2.72 → 0.49 |
| tSZ ell=100 | 3.04 → 0.98 |
| kSZ ell=1000 | 3.22 → 0.59 |
| kSZ ell=100 | 3.36 → 1.19 |
| mean DM | 3.53 → 3.33 |
| DM Q16 | 3.74 → 3.83 |
| DM Q99.9 | 3.44 → 1.06 |

- Low-mass haloes need their outskirts for every probe.
- Share of each observable's importance carried by each calibrated bin (13.5-13.7 → 15.3-15.5, %):
  - tSZ ell=1000: 1.0, 2.1, 4.1, 7.6, 13.0, 17.5, 19.4, 17.2, 10.7, 5.9.
  - mean DM: 14.7, 12.9, 10.8, 8.1, 5.9, 3.3, 1.6, 0.8, 0.2, 0.1 (58% in total).
- The fallback below 10^13.5 carries 42% of mean DM and 50% of Q16. The dominant engine bin for Q16 is 12.9-13.1 (18%).

**Integrated budgets** (`tables/budget_bands_F.csv`):
- Y(<2R)/Y(<4R) = 70-90% and N_e(<2R)/N_e(<4R) = 54-69%. Both rise with mass.
- The enclosed integral is still growing at 4R: $d\ln I(<x)/d\ln x$ = 0.12-0.55 for Y and 0.59-0.83 for N_e. Here $I(<x)$ is the shell-summed Y or N_e inside x, and the slope is evaluated over the 3.63-4R shell.

### 2. Truncation radius and fit radius

**B12/B16 refits** (Test F, objective (a), one mean profile per bin, hard cut). Family RMS vs the truth in %, as tSZ / kSZ / DM / tSZxDM (`tables/grid_F_B12B16_long.csv`; heat maps in `figures/prior_fit_cut_grids_B12B16.png`):

| R_fit \ R_cut | 2 | 3 | 4 |
|---|---|---|---|
| 1 | 5.9 / 24.8 / 20.4 / 12.9 | 3.8 / 7.7 / 5.3 / 5.2 | 3.3 / 8.7 / 8.2 / 2.0 |
| 2 | 6.5 / 27.2 / 22.1 / 13.9 | 4.6 / 12.4 / 9.2 / 7.2 | 4.1 / 3.2 / 2.2 / 3.9 |
| 3 | 5.9 / 27.0 / 22.3 / 13.2 | 3.8 / 12.5 / 10.1 / 6.3 | 3.3 / 3.6 / 1.6 / 3.1 |
| 4 | 5.1 / 25.7 / 22.0 / 11.9 | 2.8 / 10.1 / 9.5 / 4.4 | **2.2 / 1.1 / 1.3 / 1.8** |

- (4, 4) is the only Pareto point, and all 8 spatial-CV folds chose it (F REPORT).
- Against the truth mean, (1, 4) is best for tSZ (0.6%) but costs kSZ 10.6% and DM 8.2%. (4, 4) gives 0.9 / 1.7 / 0.6 / 1.4%.
- **The scatter model changes the optimum** (R_cut = 4; `prior_key_numbers.csv`):
  - With lognormal amplitudes from Y/N_e(<4R) (S1), the best R_fit is 1 for tSZ (1.25% vs 1.96% at R_fit = 4) and 3 for kSZ (2.08%) and DM (0.22%).
  - With the measured second moments, tSZ is best at R_fit = 1 (0.74%), and kSZ and DM at 4.
  - In Test G (painting target, S1), P0 has tSZ 0.74 / 1.20 / 0.75 / 1.31% at R_fit = 1 / 2 / 3 / 4 (R_cut = 4).
  - So the "fit to 4R" conclusion is robust only for the deterministic and vs-mean scores.
- **The cut matters most** (R_fit = 4): a 3R cut costs kSZ 10.1% and DM 9.5%; a 2R cut costs tSZ 5.1%, kSZ 25.7% and DM 22.0%.
- **Edge shape** (Test C): a 0.05-dex logistic edge at 4R changes $C_\ell^{yy}$ by ≤ 0.21% (≤ 0.07% flux-matched). Electrons are about 5x more edge-sensitive (DM-field $C_\ell$ about -1%).

**How far B12/B16 describe the stack** (Test B GLS, model-discrepancy $\sigma_d$; `tables/b_cut_support_summary.csv`).

$\sigma_d$ solves $\mathbf r^{\rm T}(\mathbf C+\sigma_d^2\mathbf I)^{-1}\mathbf r=\nu$:
- $\mathbf r=\ln S-\ln\bar F$ are the log residuals on the fitted shells.
- $\mathbf C$ is the bootstrap covariance of $\ln\bar F$.
- $\nu$ is the number of degrees of freedom.

Median $\sigma_d$ over 8 fit units, in %:

| cut [R200c] | 1 | 1.34 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| pressure, all gas | 0.08 | 0.16 | 0.29 | 3.48 | 11.71 |
| electrons, all gas | 0.62 | 1.18 | 1.60 | 6.06 | 11.29 |
| **pressure, A2-ii** | 0.08 | 0.16 | 0.25 | 0.64 | **0.72** (max 3.95) |
| electrons, A2-ii | 0.62 | 1.18 | 1.52 | 3.82 | 6.98 |

- With the main A2-ii stack, B12 is statistically adequate for pressure out to 4R. The 2-4R neighbour gas was what broke it.
- On the A2-ii stacks, B12/B16 refitted at (4, 4) give 2.8 / 1.1 / 1.8 / 1.8% against the A2-ii truth, with profile RMS:
  - pressure <1R / 1-2R / 2-4R: 2.2 / 4.9 / 10.5%;
  - electrons: 6.9 / 9.4 / 12.2% (`tables/grid_F_controls_B12B16_long.csv`).
- Fits to 2R extrapolate 31-37% low in the 2-4R shells (all gas, B REPORT).

### 3. Which shells drive the fitted parameters

**Influence** (Test B; `tables/b_influence_summary.csv`). $H=(J^{\rm T}WJ)^{-1}J^{\rm T}W$:
- $J=\partial\ln S/\partial\ln\theta$ is the model Jacobian.
- $W$ holds the fit weights.
- The band share is $\sum|H_{ps}|$ over a band.

| Fit | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-2 | 2-4 |
|---|---|---|---|---|---|
| Objective (a) to 4R | 0.12-0.24 | 0.22-0.24 | 0.35-0.39 | 0.05-0.12 | 0.07-0.11 |
| GLS (c) to 4R | — | — | — | — | 0.15-0.27 |

- With objective (a), P0, xc and beta are driven by 0.1-1R even when fitting to 4R; 2-4R carries only 0.07-0.11 of the influence.
- The weighting is therefore misaligned with DM and kSZ, whose sensitivity peaks at 1-4R.

**Parameter precision** (bootstrap SD of ln θ, median; `tables/b_param_precision.csv`), objective (a), all gas:

| Parameter | cut 1R | cut 4R |
|---|---|---|
| P0 | 6.2% | 5.0% |
| xc | 8.8% | 4.6% |
| beta (pressure) | 5.9% | 2.4% |
| rho0 | — | 4.3% |
| alpha | — | 2.2% |
| beta (electrons) | — | 0.7% |

- Jackknife errors are 1.16-1.36x the bootstrap for pressure, and per shell 1.11x at 2-4R (B REPORT).

**Weighting** (Test F F3): upweighting the important shells (objectives d, e, f) moves the (4, 4) family errors by -0.3 to +0.7 points. GLS (c) is worst at every R_fit.

**No observable-vs-parameter Jacobian (dO/dθ) exists.**

### 4. Fitting: B12/B16, the physical outer terms, and the Lee22 ladder

**Test G, $R_{\rm fit}=R_{\rm cut}=4$, objective (a).** Painting target = A2-ii truth. Values are % as det / S1 / vs mean (`tables/model_comparison_4R_S1.csv`, `tables/grid_G_long.csv`, `tables/importance_weighted_profile_rms.csv`). The importance-weighted profile RMS (irms) weights each shell's residual by the family's importance (tSZ for pressure; kSZ and DM for electrons).

| Model (parameters) | Fitted to | tSZ, or kSZ | DM (S1) | irms (tSZ, or kSZ / DM) |
|---|---|---|---|---|
| B12 = P0 (30) | all gas × p_keep | 2.04 / 1.31 / 1.06 | — | 4.67 |
| B12 = P0 | A2-ii stack | 2.82 / 0.68 / 0.45 | — | 2.61 |
| P1 (50) | all gas × p_keep | 3.15 / 0.59 / 0.36 | — | 1.89 |
| **P2m (44)** | all gas × p_keep | 3.20 / **0.52** / 0.30 | — | 1.66 |
| P2m | A2-ii stack | 3.01 / 0.59 / 0.23 | — | 1.82 |
| Lee22 BPLall+O2 (14) | all gas × p_keep | 3.25 / 1.14 / 0.89 | — | — |
| B16 = E0 (30) | all gas × p_keep | 1.16 / 2.70 / 2.15 | 0.82 | 8.16 / 10.11 |
| B16 = E0 | A2-ii stack | 1.11 / 2.28 / 1.76 | 1.37 | 6.42 / 8.52 |
| E1 (50) | all gas × p_keep | 1.03 / 1.52 / 0.98 | 1.20 | 4.46 / 5.92 |
| **E2gf (41)** | all gas × p_keep | 1.96 / **0.32** / 0.49 | **0.50** | 2.19 / 2.74 |
| E2gf | A2-ii stack | 1.81 / 2.99 / 2.57 | 1.66 | 9.77 / 13.36 (2-4R RMS 41.9%) |
| Lee22 BPLall+O2 (12) | all gas × p_keep | 1.13 / 1.26 / 0.79 | 0.50 | — |

**Model forms**
- B12: $P=0.5176P_0(x/x_c)^{-0.3}(1+x/x_c)^{-\beta}$. B16: $E=\rho_0(x/0.5)^{-0.2}[1+(x/0.5)^\alpha]^{-(\beta-0.2)/\alpha}$.
- E2: $E=E_g f_{\rm tr}+\bar E[1+T_c(x)]$ with $f_{\rm tr}=[1+(x/x_t)^4]^{-\gamma_t/4}$.
- P2m: $P=P_gf_{\rm tr}+\tfrac{\kappa_P}{200}\bar E[1+T_c]\tau_0(x/2)^{-q}(M/10^{14.5})^{-p_M}D(x)$.
- Ownership correction: $p_{\rm keep}=1-f(M)/[1+(x/x_h)^{-n}]$.
- Symbols are defined in the equation list below.

**Reading the table**
- For pressure, fitting B12 directly to the A2-ii stack beats all gas × p_keep (vs mean 0.45% vs 1.06%).
- For electrons, the parameter-free 2-halo term in E2 lies above the A2-ii outskirts, so E2 can only be used as all gas × p_keep.

**R_fit dependence at R_cut = 4** (S1, painting target):
- P2m tSZ: 3.47 / 1.73 / 0.54 / 0.53 / 0.52% at R_fit = 1 / 1.5 / 2 / 3 / 4. It is robust from R_fit = 2.
- E2gf kSZ: 9.95 / 4.55 / 2.32 / 1.02 / 0.32%. It needs R_fit = 4.
- E1 cannot be fitted below R_fit = 4 (DM 119% at R_fit 2 with R_cut 2; G REPORT: 77-387%).

**Lee22-style ladder** (Test G, all-gas mass stacks, floor likelihood; `tables/lee22_ladder_selected.csv`, `lee22_named_models.csv`).

Inner profile: $F_{\rm in}=A(x/x_c)^{\gamma}(1+x/x_c)^{-\beta}$ with $\gamma=-0.3$. Each of $\ln A,\ln x_c,\ln\beta$ follows a broken power law in $\ln M$ with one common break $M_{\rm cut}$. Outer option O2 adds the 2-halo term and $x_t(M)$.

| Field | R_fit | Selected | k | χ²/ν (floor) | log10 M_cut |
|---|---|---|---|---|---|
| P | 1.34 | BPLall+O0 | 10 | 1.43 | 14.10 |
| P | 2 | BPLall+O1 | 12 | 1.55 (dBIC 5.0) | 14.09 |
| P | 3 / 4 | BPLall+O2 | 14 | 1.39 / 1.32 | 14.23 / 14.12 |
| E | 1.34 | BPLall+O0 | 10 | 1.39 | 14.14 |
| E | 2 / 3 / 4 | BPLall+O2 | 12 | 1.45 / 1.40 / 1.25 | 14.13 / 14.12 / 14.13 |

- **Lee22's own choices** (break fixed at their value, here 10^13.77-13.78 Msun; BPLa for P, BPLs for E) are rejected: dBIC 221-339 at 1.34R (O0), and 270-409 at 4R (O1).
- **Free γ:**
  - pressure PL3g+O2: dBIC 544 at 4R;
  - electrons PL3gM+O2 (γ as a mass law): k = 10, χ²/ν 1.38, dBIC 22.8. This is the closest unselected rung.
  - Test F's per-bin free-γ gNFW scored best numerically, but by compensation through unresolved cusps (up to +25% at ell = 2×10^4).
- **Concentration terms** (mass × c cells, 4R, O2):
  - pressure: PL3 7.57 → PL3c3 2.61 → L22best 2.13 (χ²/ν, floor likelihood);
  - electrons: 5.11 → 2.51 → 1.71;
  - none reach dBIC ≤ 6 (all ≥ 56), and HalfDome has no concentrations.
- **The likelihood convention changes the χ²/ν:** under the L22 convention the selected 4R models have χ²/ν 5.35 (P) and 5.09 (E). The floor $\sigma_d$ grows from 0.28% to 1.12% (P) and from 1.01% to 1.93% (E) between R_fit 1.34 and 4.
- **The ladder's pressure outer term uses a self-similar temperature** (fitted ln τ0 = -2.48, q = -0.38). So BPL combined with P2m's mass-dependent outer temperature was never tested.

### 5. Mass binning and binning choice

**Stability** (Test B2 permutation tests, Lee22 grid, all gas; `tables/binning_stability_pairs.csv`):
- Pressure: 3 of 11 adjacent pairs differ, all below 10^14.1 Msun.
- Electrons: 9 of 11 pairs differ.
- Fitting a bin with its neighbour's correlation moves the parameters by 0.1-0.25σ (median) and the observables by ≤ 0.3%.

**Scheme choice at the stack level** (stacking-choices workflow 2026-10-14, `mass_binning/work/res_schemes.json` and `res_topbins.json`; machine-readable results, no written report yet).

First-order impact of painting the 10 calibrated bins with each scheme's stacks, interpolated in log M, relative to 0.2-dex M200c bins; tSZ ℓ=1000 / kSZ ℓ=1000 / mean DM, %:

| Scheme | tSZ | kSZ | mean DM |
|---|---|---|---|
| 0.1 dex | -0.04 | -0.07 | -0.02 |
| 0.3 dex | -0.71 | -0.68 | +0.01 |
| 0.5 dex | -2.82 | -2.62 | -0.73 |
| M500c 0.2 dex | +3.88 | +0.10 | 0.00 |
| M500c 0.5 dex (B24-like) | +2.52 | -3.11 | -1.11 |
| Lee22 h^-1 bins | -0.19 | -0.51 | -0.11 |

**Top bins** (N = 21 and 6):
- Pooling 15.1-15.5 into one stack: tSZ ℓ=1000 -1.24 ± 0.42%.
- Using 14.9-15.1 for both: -2.24%.
- A 14.3-15.1 trend: +0.06%.
- Test B pooled ≥ 14.9 with a trend, while F and G fit the top bins one by one. This is the untested cause of the 0.5-1.3% $C_\ell^{yy}$ difference at 2R (B REPORT).

**Bin-centre bias:** the population-weighted mean log M lies 0.004-0.026 dex below the bin centre (0.047 dex in 15.3-15.5; pipeline audit).

### 6. Stacking choices (A2-ii main; others as variations)

**A2-ii (main stack)**
- Inside 2R it equals all gas to ≤ 1.4% (≤ 0.4% per band).
- It removes 2.4-7.4% of Y(<4R) and 5.1-7.6% of N_e(<4R) (A, neighbour_bookkeeping).
- p_keep at 4R is 0.38-0.40 for pressure and 0.56-0.71 for electrons (G).
- At the truth level, relative to all gas: yy ell=100 -1.5%, kSZ 1h ell=100 -6.7%, mean DM -3.2% (F REPORT).

**Other choices, measured at the stack level** (pipeline audit, B3, D):

| Choice | Effect |
|---|---|
| Median instead of mean | pressure -3 to -56% at 2-4R |
| Equal instead of inclusion weights | up to +9.4% at 2-4R |
| Per-halo instead of common resolution mask | +1.6% to +104% in partially resolved shells |
| Q_V estimator | +3-5% inside 1R, +18-25% at 3-4R |
| Y200 weighting | +1.6 / +2.4 / +3.4% on yy 1h at ell = 100 / 1000 / 3000 |
| Recent-AGN gas kept | ≤ 0.05% (pressure), ≤ 0.17% (electrons) on resolved shells |

Only A2-i, A2-ii and AGN-kept were propagated through B12/B16 refits (F controls, B5).

### Earlier figures worth showing
See the `figures` field.

### GAPS the new tests must fill
1. **Importance at the A2-ii truth.** All truth-based importance maps use all gas, and A2-ii removes 2-4R gas (p_keep(4R) = 0.38-0.40 for P). Recompute $I_{b,s}$ for the A2-ii main stack, scatter-inclusive.
2. **Finest radial resolution of the truth importance.** It exists only on the 35-shell F grid. The 67-shell x200 maps exist only as deterministic Jacobians at the all-gas 4R fits. Rebuild truth importance on x200 (67 shells) per mass bin, with R50 / R90 / R95 and dI/dln x.
3. **Beyond 4R and truncation convergence.** For mean DM and Q16 / Q50 / Q84, edge/peak = 0.99-1.00, and N_e(<x) still grows as $x^{0.6-0.8}$ at 4R. The truth stops at 4R (6R500c > 4R200c for only 26.5% of haloes). For the A2-ii target, quantify:
   - the halo-owned gas beyond 4R, via p_keep × the 2-halo term from the models' 4-12R extension shells;
   - the R_cut response above 4R.
   - Any measured profile beyond 4R needs a new particle pass.
4. **Parameter sensitivity of the observables.** No $\partial O/\partial\theta$ exists for the B12 (P0, xc, β) and B16 (ρ0, α, β) parameters, nor for the Lee22 / BPL and G globals, per mass bin. Also missing: propagation of the parameter covariance (bootstrap / jackknife) to observable uncertainties, and the degeneracy directions. Only influence rows (B4) and parameter bootstrap SDs exist.
5. **Impact of each radial bin on the fitted 1D profiles.** Influence exists only for B12/B16 with objectives (a)/(c) on 8 fit units, and on A2-ii only for GLS (c). Missing:
   - objective (a) on A2-ii;
   - leave-one-shell-out or shell-perturbation responses of the fitted profile and of the observables, for the G models and the Lee22 ladder.
6. **Binning choice through fits.** Scheme effects (0.1 / 0.3 / 0.5 dex, M500c, h^-1, top-bin pooling, bin-centre vs mean mass, step vs smooth assignment) exist only at the stack level and from a workflow without a written report. Propagate them through B12/B16, the G models and BPL refits on A2-ii, including the B-vs-F top-bin discrepancy (0.5-1.3%).
7. **Lee22 ladder on A2-ii.** The whole ladder (PB / PL / BPL, _L22, free γ, γ(M), +c, O0 / O1 / O2) was fitted to all-gas stacks with the GLS floor likelihood only, at R_cut = 4. Missing:
   - observables for every rung (only BPLall, PB, PL3 and PL1 were evaluated);
   - an R_fit × R_cut grid;
   - objective (a) and importance-weighted variants;
   - BPL combined with P2m's mass-dependent outer temperature.
8. **A2-ii-consistent electron outer term.** E2's parameter-free 2-halo term overshoots the A2-ii outskirts (2-4R RMS 42%). A direct A2-ii electron model, beyond multiplying by p_keep, is untested.
9. **Coupling between R_fit / R_cut and the scatter model.** The optimum moves from R_fit = 4 (deterministic, vs mean) to R_fit = 1-3 under S1 / measured second moments for B12/B16. Report the det, S1 and vs-mean scores side by side, and test whether the 4R conclusion holds on A2-ii with scatter.
10. **Finer R_fit and R_cut steps for the physical models.** G used 1, 1.5, 2, 3 and 4 only. Add 2.5, 3.3 and 3.63, and R_cut between 3 and 4.
11. **Low-mass coverage.** The fallback below 10^13.5 carries 42% of mean DM and 50% of Q16. Strict masks resolve 13.3-13.5 only from 0.52R. Quantify how the sensitivity in the uncalibrated range depends on the fallback, or use the diagnostic masks.
12. **Other stacking choices through the A2-ii fit chain.** Statistic, weights, mask rule, estimator and normalisation have only stack-level numbers. Propagate the main user choices as variations of the A2-ii pipeline.