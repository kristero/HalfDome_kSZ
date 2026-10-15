## Fitting the A2-ii stacks: B12/B16 and the Lee22 ladder (Test I)

### 0. Short answer
"How far must I fit, and where must I truncate?" Every number below is a family RMS error against the A2-ii truth. Two values are given for each: the S1 painting HalfDome would use, against the scatter-inclusive truth; and in brackets the profile-only error (det painting against the truth mean).

| probe | recommended model | R_fit [R200c] | R_cut [R200c] | error S1 (profile-only) | where the \|importance\| weight is (Test H) | R containing 90% of the observable, ℓ = 100/300/1000/3000 |
|---|---|---|---|---|---|---|
| tSZ $C_\ell^{yy}$ | B12 per mass bin, objective (a) | any of 1.34-4 (use 4) | 4 (3: +0.3 pp for (a)) | 0.68% (0.45%) | 90% inside 1.05-1.15, 99% inside 2.4-2.5 | 1.59 / 1.42 / 1.20 / 0.87 |
| kSZ 1-halo | B16 per bin, objective (d) | 3-4 | 4 | 0.69-0.98% (0.57-0.62%) | 16% at 2-4; 99% inside 3.8-4.0 | 3.21 / 3.10 / 2.61 / 1.37 |
| FRB DM PDF | B16 per bin, objective (d) | 4 (3: 0.76%) | 4 | 0.51% (0.49%) | 26% at 2-4; 90% inside 3.1-3.3 | DM mean 2.87; DM variance 1.91 |
| tSZ×DM | B12 + B16 as above | 4 | 4 | 0.99% (B12 side), 1.06% (B16 (d) side) | in between | E side: 2.34 / 2.08 / 1.47 / 0.78 |

Sources: `results/tables/I1_scores_final.csv` (f35 grid, R_cut = 4, columns `rms_*`, `prof_*`), `I4_importance_regions.csv` and `I4_observable_truncation_radii.csv`.

- **Weight radius vs observable radius.** The "\|importance\| weight" columns give radii of the weight, not of the observable. For a quadratic observable ($C_\ell$, DM variance; $k_O=2$), truncating at $x$ removes, to first order, twice the signed importance beyond $x$. So "90% of the tSZ weight is inside 1.05-1.15 R200c" does **not** mean that truncating there keeps 90% of $C_\ell^{yy}$. Truncating the truth at 1.15 R200c loses 22 / 16 / 11 / 3.3% of $C_\ell^{yy}$ at ℓ = 100 / 300 / 1000 / 3000. The last column gives the radius that keeps 90% of the observable itself.
- **S1 floor.** The tSZ (0.68%) and kSZ (0.69-0.98%) S1 errors are close to the S1 scatter-model floor. That floor is 0.48% / 0.62% as estimated; the exact value for the target profile is 0.48% / 0.60%. The DM error (0.51%) is instead set by the profile shape: profile-only 0.49%, against a floor of only 0.18% (exact 0.17%).
- **Truncation.** Truncating inside 4 R200c costs kSZ and DM 7-9% at R_cut = 3.
- **Compact Lee22 laws vs per-bin forms** (R_fit = R_cut = 4; `I4_compact_vs_perbin_Rf4Rc4.csv`):
  - **tSZ.** The compact laws are worse than B12 per bin (0.68%). BPLall+O2 gives 2.18% (+1.5 pp). PL1+O2, PL3+O2 and PL3g+O2 give 5.87 / 9.32 / 9.82% (+5.2 to +9.1 pp).
  - **kSZ.** PL3+O0, PL3g+O0 and BPLall+O0 give 1.13 / 1.09 / 1.31%. That beats B16 per bin with objective (a) (2.27%) and the ladder's per-bin PB+O0 (1.40%).
  - **DM.** BPLall+O0 (1.05%) beats B16 (a) (1.37%) and equals PB+O0 (1.05%). PL3+O0 and PL3g+O0 (1.83%) are worse than B16 (a).
  - **Only B16 with objective (d) is better than all of them** (kSZ 0.98%, DM 0.51%). Its advantage is largely in-sample (section 2).
- **Per-halo resolution rule.** It is not an acceptable stacking choice for this target. Fitted on its own resolved shells, it raises the low-mass cores and gives tSZ 6.56%, kSZ 13.0%, DM 14.6% (B12/B16 (a), R_fit = R_cut = 4; section 8).

### 1. What was done and where it ran
- **Target.** The A2-ii painting partition of FLAMINGO L1_m9 at z = 0 (snapshot 0077).
  - Each gas particle belongs to the painted halo (M200c ≥ 6.6e12 Msun) in which it has the smallest r/R200c.
  - Recently AGN-heated gas is removed.
  - Stacks are inclusion-weighted means in the 10 calibrated M200c bins 13.5-15.5 (0.2 dex; log10 Msun).
- **Fits.** On Test F's 35-shell fit grid; B12/B16 also on the fine 67-shell x200 grid, to check grid dependence.
- **Evaluation.**
  - Every painting was evaluated with Test H's fine A2-ii basis (67 shells to 4 R200c, plus 8 extension shells) against Test H's A2-ii truth.
  - The Test H inputs are truth.npz, truth_summary.json and sens.npz. Their SHA256 equal the current local Test H results (0f9261…, d3caaf…, 8dbe2d…) in all three runs (`results/run_manifest.json` invocations[0..2].h_inputs_sha256).
- **maas run 1** (stages fits35, finefits, ladder, lvariants, evaluate): 8 workers, nice 19; exit 0, 1538 s, peak tree RSS 2.82 GiB (`CLUSTER_RUN.json` maas_runs[0]).
  - Its driver option `--cleanup-scratch` also deleted Test H's maas scratch (/tmp/nagainis_halfdome_testH), which held Test H's fine basis. Deleting another test's scratch was a process error. The driver now deletes only Test I's own scratch.
  - Because the basis was gone, run 2 had to rebuild it.
- **maas run 2** (supplement, `i_supp.py`): 8 workers, nice 19; exit 0, 702 s, peak tree RSS 6.82 GiB (maas_runs[1]). It did three things:
  - Rebuilt the A2-ii basis with Test H's code. Its truth-mean observables equal Test H's to a max relative difference of 0.0 over 63 observables (`results/supp/basis_rebuild.json`).
  - Painted objective (L) and evaluated the ladder at every R_cut.
  - Re-evaluated run-1 paintings: 10 rows and 200 values agree to 0.0 relative difference.
- **maas run 3** (this fix; `i_perhalo.py`, stages phfits and pheval): 8 workers, nice 19; launched 21:53 EEST at load 24.2 on 48 cores with 178.5 GiB free; exit 0, 110 s, peak tree RSS 2.39 GiB (maas_runs[2]; `results/test_I_done.json`).
  - It refits the per-halo-rule variant on its own resolved shells and paints the per-halo-rule fine stack (section 8).
  - Inputs: `inputs/i0_perhalo_x200.npz`, built locally from the stacking-core cache by `i_perhalo_inputs.py`.
  - It reused run 2's basis (SHA256 2e1d…, `results/perhalo/basis_check.json`).
  - Parity: strict-shell refits equal run 1 to 0.0 (`results/perhalo/phfits.json` parity); six run-1 painting rows (124 values) reproduced to 0.0; the A2-ii stack painted det equals Test H's truth mean to 0.0 (`verification.json`).
- **Local in this fix:** everything else, a re-analysis in `report_i.py`. The pre-fix copy is in `attempts/pre_fix_verifier2_20261014/`; run 2's job records are in `attempts/run2_20261014/`.

### 2. Definitions
#### Notation and units
- $x=r/R_{200c}$ (dimensionless).
- $P=P_e/P_{200}$, with $P_{200}=G M_{200}(200\rho_c)f_b/(2R_{200})$.
- $E=n_e\mu_{XGP}/(\rho_c f_b)$, where:
  - $n_e$ is the electron number density;
  - $\mu_{XGP}=1.2632\,m_p$ is XGPaint's mass per electron (including its 1/0.9);
  - $\rho_c$ is the critical density at z = 0;
  - $f_b=\Omega_b/\Omega_m=0.1588$.
- $kT_{200}=0.59\,m_p\,G M_{200}/(2R_{200})$.
- With these definitions $P/P_{200}=\kappa_P\,(E/200)\,(T_e/T_{200})$, where $\kappa_P=0.59\,m_p/\mu_{XGP}=0.4671$ (`G_physical_outer/g_common.py:21,103`; `prepare.json` normalisation.KAPPA_P_code).
- $c$ = calibrated mass bin; $s$ = radial shell.
- $m_s$ = inclusion-weighted A2-ii stacked mean in shell $s$.
- $S_s(\theta)$ = exact volume average of a model $F(x;\theta)$ over shell $[x_s,x_{s+1}]$:
$$S_s=\frac{3}{x_{s+1}^3-x_s^3}\int_{x_s}^{x_{s+1}}F(x)\,x^2\,dx$$
  It is evaluated with 32-point Gauss-Legendre nodes per shell.

#### Resolution rules (stacking core)
A halo resolves shell $[x_s,x_{s+1}]$ when all three conditions hold:
- $x_s\ge\max(0.04,\,4\epsilon/R_{200c})$, with softening $\epsilon=0.0057$ Mpc;
- the shell holds at least 20 gas particles;
- $x_s\ge 2h_{med}/R_{200c}$, where $h_{med}$ is the upper-median SPH smoothing length of the shell's gas.

There are two rules:
- **Strict (A2-ii reference).** A shell is used for a bin only if every halo of the bin resolves it. All haloes enter every shell.
- **Per-halo.** Each halo enters only the shells it resolves. A shell is used if at least one halo resolves it.

#### B12 (pressure) and B16 (electrons), per mass bin
$$P(x)=0.5176\,P_0\,q^{-0.3}(1+q)^{-\beta},\qquad q=x/x_c$$
- $P_0$: amplitude in $P_{200}$ units.
- $x_c$: core radius in $R_{200c}$.
- $\beta$: outer slope.
- 3 per bin, so $k=30$ over 10 bins.

$$E(x)=\rho_0\,q^{-0.2}(1+q^{\alpha})^{-(\beta-0.2)/\alpha},\qquad q=x/0.5$$
- $\rho_0$: amplitude in E units.
- $\alpha$: sharpness of the transition.
- $\beta$: outer slope.
- $k=30$.

#### Objectives for B12/B16
All three sum over the resolved fit shells with $x_{s+1}\le R_{fit}$.

(a) Diagonal, with a 5% floor on the halo-to-halo scatter:
$$\chi^2_a=\sum_s\Big[\frac{\ln S_s-\ln m_s}{\sigma_s}\Big]^2,\qquad \sigma_s=\max(\mathrm{SD}_s/m_s,\ 0.05)$$
- $\mathrm{SD}_s$ is the reliability-weighted halo-to-halo standard deviation of the bin.

(d) Importance-weighted:
$$\chi^2_d=\sum_s w_s(\ln S_s-\ln m_s)^2$$
- $w_s$ is Test H's A2-ii family importance of the bin, summed into the fit shell and normalised to unit sum.
- Pressure uses the tSZ weights; electrons use (kSZ + DM)/2.
- **(d) is tuned in-sample.** The observable errors are exact (computed in the basis). But the (d) weights are the first-order responses of these same observables to these same truth profiles.
  - The linear response $k_O\sum I_O(M/m-1)$ reproduces the exact kSZ profile-only errors of the B16 paintings at R_fit 3-4 to ≤ 0.05 pp. Compare `lin_ksz1h_*` in `results/tables/I_scores.csv` with `error_vs_mean` in `I_errors_long.csv`; for example B16 (d) at R_fit 4, ℓ = 100: 0.618% linear vs 0.615% exact.
  - So (d)'s advantage on the kSZ, DM and tSZ errors, and on the score $S$, is largely in-sample.
  - How it performs on other observables, other redshifts or another truth is not tested.

(L) Lee22 lognormal likelihood with the covariance of the mean:
$$\chi^2_L=r^{T}\big(C_{L22}/h\big)^{-1}r,\qquad r_s=\ln S_s-\ln m_s,\qquad h=\frac{N-p-2}{N-1}$$
- $C_{L22}$ is the inclusion-weighted sample covariance of the haloes' $\ln F$, divided by $N_{eff}=1/\sum_i w_i^2$.
- $h$ is the Hartlap factor; $N$ = haloes in the bin; $p$ = shells fitted.
- Shells where any halo has $F\le0$ are dropped.
- Bins 15.1-15.3 and 15.3-15.5 have $N\le p+2$, so they have no L fit (`verification.json` I1_L_fits_missing_bins_per_field: 12 per field). The painting "L*" uses the objective-(a) fit of the same R_fit there.

#### Lee22 ladder
Run unchanged with Test G's `g_selection` on the A2-ii stacks.

Inner profile:
$$F_{in}(x)=A\,(x/x_c)^{\gamma}(1+x/x_c)^{-\beta}$$
- $A$ is the amplitude in P or E units.
- $\alpha=1$; $\gamma=-0.3$ unless freed.

Mass laws for each $Q\in\{A,x_c,\beta\}$ (in ln):
- per bin (PB);
- power law
$$\ln Q=\ln Q_0+a_m\ln(M/10^{14.5}M_\odot)$$
  - PL1: $A$ only; PL2: $A$ and $x_c$; PL3: all three.
- broken power law with slope $a_m$ below and $a'_m$ above a shared $M_{cut}$:
$$\ln Q=\ln Q_0+a_m^{(\prime)}\ln(M/M_{cut})$$
  - BPLa: $A$ broken; BPLs: $x_c$ and $\beta$ broken; BPLall: all three broken.
  - $M_{cut}$ is free, or fixed at Lee22's value (suffix _L22) converted with $h=0.6774$: log10 M_cut = 13.77 (P), 13.78 (E), 13.81 / 13.92 with c (`ladder_selection_summary.json` lee22_mcut_msun).
- concentration terms $+a_c\ln(c/10)$, with $c$ = SOAP concentration_200c, on concentration-split cells: PL1cA, PL3cA, PL3c3, and L22best (Lee22's preferred law plus c terms on all three).
- free gamma: $\gamma$ global (PL3g), or $\gamma=\gamma_0(M/10^{14.5})^{a_\gamma}$ (PL3gM).

Outer options (Test G):
- O0: none.
- O1 for E: $F=F_{in}+\bar E\,[1+b(M)\,\xi_{lin}(xR_{200c})]$ (no parameters).
- O1 for P:
$$F=F_{in}+\frac{\kappa_P}{200}\,\bar E\,[1+b\,\xi_{lin}]\;\tau_0\,(x/2)^{-q}\;D(x),\qquad D(x)=\frac{x^4}{x^4+x_D^4},\ x_D=0.1$$
- O2: as O1, but with $F_{in}$ multiplied by
$$f_{tr}=[1+(x/x_t)^4]^{-1.5},\qquad x_t=x_{t0}(M/10^{14.5})^{a_{xt}}$$

Symbols of the outer terms:
- $\bar E=\kappa_E\Omega_m=0.3348$ is the cosmic mean electron density in E units; $\kappa_E=1.094$ was measured at 2-4 R200c and $\Omega_m=0.306$ (`G_physical_outer/results/prepare.json` normalisation.Ebar_used).
- $b(M)$ is the Tinker+2010 halo bias for $\Delta=200/\Omega_m$.
- $\xi_{lin}$ is the CLASS linear matter correlation function at z = 0, averaged over the bin's haloes.
- $\tau_0$ is the outer electron temperature $T_e/T_{200}$ at $x=2$ (dimensionless).
- $q$ is the power-law index of $T_e/T_{200}$ with radius; $\tau_0$ and $q$ are 2 global parameters (P only).
- $D(x)$ damps the pressure outer term inside $x_D=0.1$ R200c.
- $x_{t0}$ is the transition radius at the pivot mass, in R200c; $a_{xt}$ is its mass slope. These 2 global parameters, with exponent $-1.5$, are a Diemer-Kravtsov steepening with $\gamma_t=6$.
- Code: `g_selection.py:262-279`, `g_models.py:30-35`.

Likelihoods:
- **stat:** bootstrap $\mathrm{Cov}[\ln\bar m]$ (1000 draws), shrunk to the neighbours' pooled correlation.
- **floor:** $C_{stat}+\sigma_d^2 I$. $\sigma_d$ is common to a (field, data set, R_fit) and solves $\sum r^T(C+\sigma_d^2I)^{-1}r=\nu$ for the per-bin reference (PB with its best outer option), so that the reference has $\chi^2/\nu=1$.
- **L22:** as objective (L).

Information criteria:
$$\mathrm{AIC}=\chi^2+2k,\qquad \mathrm{BIC}=\chi^2+k\ln n$$
- $\chi^2$ under the floor likelihood; $k$ = free parameters; $n$ = data points.
- $\Delta$BIC is relative to the best model of the same field, data set and R_fit.

Cross-validation (CV): predictive $\chi^2$ summed over $K=8$ spatial folds.

Selection rule: choose the fewest $k$ that satisfies both
$$\Delta\mathrm{BIC}\le6\quad\text{and}\quad \mathrm{CV}-\mathrm{CV}_{best}\le\sqrt{K}\,\mathrm{SD}_{folds}(\text{best-CV model})$$
If no model satisfies both, the lowest-BIC model is reported with rule_satisfied = False.

#### Painting, predictions and truth
- **Truncation:** a hard cut $F(x)\,\Theta(R_{cut}-x)$ on the fine shells.
- **det:** one profile per calibrated bin.
- **S1:** lognormal scatter of the whole profile's amplitude, $\langle F_sF_t\rangle=\mathcal M\,F_sF_t$, with:
  - $\mathcal M=\langle X^2\rangle/\langle X\rangle^2$ for $X=Y(<4R_{200c})$ (pressure) or $N_e(<4R_{200c})$ (electrons);
  - the cross term $\langle YN\rangle/(\langle Y\rangle\langle N\rangle)$;
  - all from the A2-ii per-halo integrals (`g_physics.py:191-211`).
  - S1 is what HalfDome would paint.
- **Pairing:** a pressure model is paired with the truth's mean electrons, and vice versa.
- **Truth (Test H):** scatter-inclusive, using every calibrated halo's own profile and second moments, including the shells that halo does not resolve. The **truth mean** is the A2-ii bin-mean profiles painted the same way. Both stop at 4 R200c: nothing beyond 4 R200c is in the target.
- **Fallback haloes:** outside the calibrated bins they are painted with the published B12/B16 profiles, identically in truth and every model (`F_observables/f_common.py:15`). The quoted relative errors are therefore diluted by their share (section 3).

#### Scores
Family RMS error:
$$\epsilon_{fam}=\sqrt{\tfrac{1}{n_{fam}}\sum_{O\in fam}(O_{model}/O_{ref}-1)^2}$$
- $n_{fam}$ is the number of members.
- tSZ: $C_\ell^{yy}$ total at ℓ = 100, 300, 1000, 3000.
- kSZ: 1-halo $C_\ell$ at the same ℓ.
- tSZ×DM: $C_\ell^{y,DM}$ total at the same ℓ.
- DM: mean, variance and the 16/50/84/97.5/99.9% quantiles of the halo DM PDF ($z_s=1$).
- **Headline error:** S1 painting with $O_{ref}$ = scatter-inclusive truth.
- **Profile-only error ("prof"):** det painting with $O_{ref}$ = truth mean. It isolates the profile-shape and truncation error from the scatter model.
- DM S1 PDFs were run only where $R_{cut}=R_{fit}$ or $R_{cut}=4$; elsewhere DM is det (marked "d" in fig_I1).

Decomposition (exact, per observable):
$$1+e_{S1}=(1+e_{det,mean})(1+f_{S1}),\qquad f_{S1}=\frac{O_{S1}(M)/O_{det}(M)}{O_{truth}/O_{truth\,mean}}-1$$
- $e_{S1}$ is the S1 error against the truth; $e_{det,mean}$ is the profile-only error.
- $f_{S1}$ is the error of the S1 scatter model: the ratio of the painted to the true scatter boost. It hardly depends on the painted profile $M$.
- Its median over the 105-106 R_cut = 4 paintings is the "S1 floor" (an estimate; `I4_s1_floor.csv`; `report_i.py:190-217`).
- Run 3 also gives its **exact value for the target profile itself**. When the A2-ii stack is painted with S1, $e_{det,mean}=0$, so $e_{S1}=f_{S1}(m)$ (`report_numbers.json` "I4|S1_error_of_target_profile_EXACT").

Importance (Test H):
$$I_O[c,s]=\frac{1}{k_O\,O}\frac{\partial O}{\partial\ln F_{c,s}}$$
- $k_O=2$ for quadratic observables and 1 for linear ones.
- $I_O$ is the fractional response of $O$ to a fractional change of the profile in bin $c$, shell $s$.
- Summed over all 16 engine bins and 67 shells it is 1, to within 0.990-1.000 (DM quantiles are nonlinear; `verification.json` signed_importance_sum_all_bins_minmax).

Family weight: $w_{fam}[c,s]$ is the mean of $|I_O[c,s]|$ over the family's members and ℓ = 100-3000, over the calibrated bins only.

Observable truncation curve (Test H `sens.npz` trunc_all__a2ii__*):
$$T_O(R_t)=\frac{O(\text{every calibrated halo of the truth truncated at }R_t)}{O(4R_{200c})}$$
- Fallback haloes are unchanged.
- $R_q$ is the first $R_t$ with $T_O\ge q$, interpolated linearly in $x$ between fine edges.
- The lost fraction is $1-T_O$. To first order it is $k_O\sum_{c,\,x_s\ge R_t}I_O[c,s]$ (column lost_pct_linear in `I4_observable_truncation_radii.csv`; `report_i.py:1067-1096`).
- The linear estimate overshoots when the loss is large. For example, kSZ at ℓ = 100 truncated at 1 R200c: 65% exact vs 93% linear; $C_\ell^{yy}$ at ℓ = 1000 truncated at 1.15: 11.2% vs 12.8%.

Sensitive-part score:
$$S_{fam}=\sqrt{\frac{\sum_{c,s}w_{fam}[c,s]\,d[c,s]^2}{\sum_{c,s}w_{fam}[c,s]}}$$
- $d=\ln(M/m)$ inside $R_{cut}$, with $M$ the painted model's fine-shell average and $m$ the A2-ii stack.
- $d=-1$ beyond $R_{cut}$ (removed signal).
- $S$ is dimensionless: 0.05 means a 5% RMS profile error where the family's signal comes from.
- The in-profile part is $S_{in}=\sqrt{(S^2-c)/(1-c)}$, where $c$ is the weight share beyond $R_{cut}$.
- $S$ is not an observable error: signed residuals at different radii and masses cancel in the observables.

Profile RMS of a fit (`I1_profile_rms_summary.csv`, Test F/G's `rms_percent`, `models.py:45`):
$$100\sqrt{\mathrm{mean}_s(S_s/m_s-1)^2}$$
- It runs over the fitted shells of one bin, unweighted.
- It is the RMS of model/stack − 1, not of $\ln$(model/stack). For example, B16 (a) at R_fit 4, bin 0: 12.72% (ratio − 1) vs 11.83% (ln).

### 3. Where the signal is
#### Where the |importance| weight is
Family weights (`results/tables/I4_importance_regions.csv`, `I4_calibrated_signal_share.csv`):

| family | x<0.3 | 0.3-0.7 | 0.7-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 | x at 50/90/95/99% of the weight: shell edge | interpolated | bins with most weight |
|---|---|---|---|---|---|---|---|---|---|---|
| tSZ | 0.334 | 0.377 | 0.110 | 0.071 | 0.023 | 0.014 | 0.004 | 0.42 / 1.15 / 1.50 / 2.50 | 0.40 / 1.05 / 1.39 / 2.39 | 14.3-15.5 (0.11-0.20 each) |
| kSZ | 0.091 | 0.263 | 0.159 | 0.183 | 0.106 | 0.109 | 0.054 | 0.91 / 2.50 / 3.30 / 4.00 | 0.91 / 2.48 / 3.06 / 3.77 | 13.5-14.7 (0.12-0.17 each) |
| DM | 0.128 | 0.214 | 0.112 | 0.146 | 0.107 | 0.150 | 0.108 | 1.15 / 3.30 / 3.63 / 4.00 | 1.03 / 3.07 / 3.51 / 3.90 | 13.5-14.5 (0.13-0.18 each) |
| tSZ×DM (E side) | 0.184 | 0.342 | 0.145 | 0.130 | 0.062 | 0.058 | 0.028 | 0.61 / 1.86 / 2.56 / 3.63 | 0.61 / 1.85 / 2.50 / 3.57 | 13.9-14.9 |

How to read the table:
- The "shell edge" radius is the outer edge of the fine shell in which the cumulative share first reaches q. The "interpolated" radius interpolates linearly inside that shell.
- **tSZ:** 90% of the \|importance\| weight is inside 1.05-1.15 R200c, mostly in haloes above 10^14.3 Msun. Truncating there would still lose 11-22% of $C_\ell^{yy}$ at ℓ = 100-1000; see the next table, and for a fitted B12 (a) at R_fit 4 the error is 7.5% at R_cut 1.5 and 3.2% at R_cut 2 (`I4_rcut_curves.csv`).
- **kSZ and DM:** the weight comes from 10^13.5-10^14.7 haloes, with 16% (kSZ) and 26% (DM) of it at 2-4 R200c.

#### How much of the observable a truncation keeps
Every calibrated halo of the truth truncated at $R_t$ (`I4_observable_truncation_radii.csv`, from Test H `sens.npz` trunc_all__a2ii__*):

| observable | ℓ | R50 | R90 | R95 | R99 | lost [%] at R_t = 1 / 1.15 / 1.5 / 2 / 3 |
|---|---|---|---|---|---|---|
| $C_\ell^{yy}$ | 100 | 0.72 | 1.59 | 2.06 | 3.25 | 29.4 / 22.0 / 11.6 / 5.3 / 1.5 |
| | 300 | 0.53 | 1.42 | 1.84 | 2.97 | 20.8 / 16.0 / 8.7 / 4.0 / 0.9 |
| | 1000 | 0.53 | 1.20 | 1.50 | 2.30 | 16.4 / 11.2 / 5.0 / 1.8 / 0.2 |
| | 3000 | 0.41 | 0.87 | 1.05 | 1.43 | 6.0 / 3.3 / 0.7 / −0.05 / −0.06 |
| kSZ 1-halo | 100 | 1.38 | 3.21 | 3.58 | 3.92 | 65.3 / 58.9 / 46.0 / 31.8 / 13.0 |
| | 300 | 1.29 | 3.10 | 3.52 | 3.90 | 61.2 / 55.2 / 42.8 / 29.3 / 11.4 |
| | 1000 | 1.00 | 2.61 | 3.14 | 3.79 | 49.9 / 43.5 / 31.2 / 19.0 / 6.0 |
| | 3000 | 0.48 | 1.37 | 1.63 | 2.00 | 22.1 / 16.4 / 7.2 / 1.0 / −1.1 |
| DM mean | – | 0.54 | 2.87 | 3.40 | 3.88 | 38.9 / 35.4 / 28.4 / 20.3 / 8.6 |
| DM variance | – | 0.57 | 1.91 | 2.53 | 3.57 | 29.2 / 24.2 / 15.9 / 9.0 / 2.8 |
| tSZ×DM (electron side cut) | 100 | 0.74 | 2.34 | 3.03 | 3.79 | 36.7 / 30.8 / 21.2 / 13.3 / 5.2 |
| | 1000 | 0.55 | 1.47 | 1.90 | 2.89 | 22.4 / 17.1 / 9.5 / 4.3 / 0.8 |
| | 3000 | 0.36 | 0.78 | 0.90 | 1.06 | 2.1 / −0.5 / −2.5 / −2.1 / −0.5 |

- A negative loss means truncating raises $O$ at that ℓ.
- These radii include the fallback haloes in $O$, so they refer to the total observable.

#### Caveat: calibrated bins only
- The weight shares above are normalised within the 10 calibrated bins.
- On Test H's signed importance, the calibrated bins carry only:
  - 0.570 of the mean DM;
  - 0.752 of the DM variance;
  - 0.47 / 0.52 / 0.57 / 0.69 / 0.88 of the DM 16/50/84/97.5/99.9% quantiles;
  - 0.975-0.987 of $C_\ell^{yy}$;
  - 0.63-0.90 of kSZ 1-halo;
  - 0.92-0.97 of tSZ×DM (ℓ 100-3000).
- The rest comes from fallback haloes (mostly M200c < 10^13.5), which are identical in model and truth. The DM and kSZ errors quoted here are therefore diluted by that share, and the fallback profiles are not tested by Test I.

#### Unresolved core
Sources: `I4_unresolved_core.csv`, `I4_unresolved_core_by_bin.csv`.
- Every calibrated bin has an unresolved core. The first strict-resolved fit shell starts at 0.321 / 0.274 / 0.170 / 0.145 / 0.105 / 0.090 / 0.065 / 0.056 / 0.040 / 0.040 R200c in bins 13.5-13.7 … 15.3-15.5. In the top two bins, 0.04 is the global inner limit.
- The shells inside are not fitted but are painted and scored.
- Summed over **all ten bins** they carry 2.69% (tSZ), 2.15% (kSZ) and 3.42% (DM) of the family weight.
- The three lowest bins (13.5-14.1) carry 0.57% (tSZ), 1.39% (kSZ) and 1.88% (DM).
- Per bin, the largest shares are 0.50% (tSZ, 14.5-14.7), 0.58% (kSZ, 13.7-13.9) and 0.73% (DM, 13.7-13.9).

### 4. I1: B12/B16 refits on the A2-ii stacks
**S1 errors against the truth**, R_cut = 4, in % (`I1_scores_final.csv`, column rms_*; `figures/fig_I1_rfit_rcut_family_rms.png`, where DM cells are S1 when the S1 PDF was run and det otherwise, marked "d"):

| R_fit | tSZ (a) | tSZ (d) | tSZ (L*) | kSZ (a) | kSZ (d) | kSZ (L*) | DM (a) | DM (d) | DM (L*) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.91 | 1.07 | 0.82 | 15.7 | 15.8 | 15.6 | 11.8 | 11.9 | 11.7 |
| 1.34 | 0.79 | 0.52 | 1.07 | 10.1 | 9.2 | 9.5 | 8.5 | 8.1 | 7.8 |
| 2 | 0.88 | 0.57 | 0.74 | 4.8 | 2.9 | 4.4 | 4.8 | 3.5 | 3.7 |
| 3 | 0.66 | 0.70 | 0.68 | 2.4 | 0.69 | 3.8 | 2.4 | 0.76 | 2.3 |
| 4 | 0.68 | 0.81 | 0.80 | 2.3 | 0.98 | 6.6 | 1.37 | 0.51 | 2.9 |

**Profile-only errors** (det against the truth mean), R_cut = 4, in % (`I1_scores_final.csv`, column prof_*; `figures/fig_I1_profile_vs_S1.png`):

| R_fit | tSZ (a) | tSZ (d) | tSZ (L*) | kSZ (a) | kSZ (d) | kSZ (L*) | DM (a) | DM (d) | DM (L*) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.48 | 0.76 | 0.40 | 14.9 | 15.0 | 14.8 | 11.8 | 11.9 | 11.6 |
| 1.34 | 0.70 | 0.32 | 1.06 | 9.3 | 8.5 | 8.8 | 8.4 | 8.0 | 7.8 |
| 2 | 0.83 | 0.38 | 0.65 | 4.2 | 2.4 | 3.7 | 4.8 | 3.4 | 3.6 |
| 3 | 0.45 | 0.35 | 0.23 | 1.8 | 0.57 | 3.2 | 2.4 | 0.71 | 2.2 |
| 4 | 0.45 | 0.49 | 0.66 | 1.8 | 0.62 | 6.0 | 1.30 | 0.49 | 2.9 |

**Truncation** (objective (a), R_fit = 4; `I4_rcut_curves.csv`). Errors at R_cut = 1 / 1.5 / 2 / 3 / 4:
- tSZ (S1): 20.4 / 7.5 / 3.2 / 1.0 / 0.7%.
- kSZ (S1): 52.1 / 34.5 / 22.3 / 7.1 / 2.3%.
- DM (det): 37.3 / 27.9 / 20.2 / 8.1 / 1.8%. DM S1 at R_cut 4 is 1.37%.

Notes on truncation:
- The profile-only errors at R_cut 3 (objective (a), R_fit 4) are tSZ 1.00%, kSZ 7.5%, DM 8.1% (I1_scores_final, R_cut 3).
- "R_cut 3: +0.3 pp" for tSZ holds for objective (a): 0.99 vs 0.68% at R_fit 4.
- For B12 (d) the R_cut 3 error is **lower** than at R_cut 4: 0.65 vs 0.70% at R_fit 3, and 0.60 vs 0.81% at R_fit 4. This is a cancellation between the removed outskirts and the in-profile error (section 5), not a better profile.

**Sensitive-part score** (objective (a), R_fit = 4; `fig_I1_rfit_rcut_score.png`):

| score | R_cut | tSZ | kSZ | DM |
|---|---|---|---|---|
| S | 4 | 3.3% | 8.0% | 11.6% |
| S | 3 | 6.8% | 24.3% | 34.5% |
| S_in | 3 | 3.2% | 7.3% | 11.0% |

Profile RMS of the fits (median per-bin RMS of model/stack − 1; `I1_profile_rms_summary.csv`):
- B12 (a): 0.56% at R_fit 1.34; 5.2% at R_fit 4 (bins 2.8-8.5%).
- B16 (a): 1.4% at R_fit 1.34; 7.7% at R_fit 4 (bins 3.7-14.6%).
- With 3 parameters per bin, neither form follows the A2-ii outskirts to 4 R200c shell by shell. The observable errors are nonetheless small, because the residuals change sign with radius (section 7).

**Objective (L)**:
- Gives $\chi^2/\nu$ = 2.3-12.5 (P) and 8.8-54 (E) per bin at R_fit 4 (report_numbers I1|*|L|Rf4|chi2nu_L_range).
- This is the covariance-of-the-mean convention, dominated by the high signal-to-noise inner shells. It fits the outskirts worst: kSZ 6.6% (profile-only 6.0%).

**Grid dependence:**
- Fine-grid (a) fits change the parameters by at most 3.4% (P) and 5.7% (E) at R_fit 4 (`I1_grid_dependence_params.csv`).
- Observables at R_fit = R_cut = 4, fine vs 35-shell fit: tSZ 0.665 vs 0.679%, kSZ 2.50 vs 2.27%, DM 1.83 vs 1.37%. A finer fit grid does not help.

**Parity with Test G:**
- Objective-(a) parameters equal Test G's A2-ii fits to 4.7e-8 over 100 fits.
- The family RMS differs from G's (35-shell basis, Test F truth) by ≤ 0.016 (tSZ), 0.021 (kSZ) and 0.005 (DM) percentage points for (a).
- For (d) the difference is ≤ 0.21 pp (kSZ 4/4: 1.49 vs 1.70%), because Test I's weights are Test H's A2-ii importance (`verification.json` I1_vs_G_family_rms_abs_diff_percentage_points).

### 5. What limits the accuracy: scatter-model floor, cancellations
**S1 floor.** Estimate from `I4_s1_floor.csv`: the median $f_{S1}$ over 105-106 paintings at R_cut = 4.
- Family RMS: tSZ 0.48%, kSZ 0.62%, DM 0.18%, tSZ×DM 0.72%.
- **Exact value for the target profile** (run 3, the A2-ii stack painted with S1): tSZ 0.484%, kSZ 0.601%, DM 0.171%, tSZ×DM 0.714%. The det painting of the same stack equals the truth mean to 0.0.
- Per observable (estimate):
  - kSZ: +0.47 to +0.80% at all ℓ;
  - tSZ: +0.77 / −0.53 / +0.24 / −0.05% at ℓ = 100 / 300 / 1000 / 3000;
  - DM mean: exactly 0, since S1 preserves the mean;
  - DM variance: +0.06%;
  - DM quantiles: −0.12 to +0.31%.

Consequences:
- R_cut = 4 errors are **not** pure profile-shape errors. For B12 (a) at 4/4 the profile-only error is 0.45% and the S1 error 0.68%.
- **tSZ and kSZ.** The recommended S1 errors (tSZ 0.68%, kSZ 0.69-0.98%) are close to the floor (0.48% / 0.60%).
- **DM.** The recommended B16 (d) error (0.51%) is 2.8 times the floor (0.18%). Its profile-only part (0.49%) is almost the whole error, so DM is limited by the profile shape, not by the scatter model.
- The floor can also cancel a profile error. For example, PL3g+O0 at R_fit 4 has kSZ S1 1.09% but profile-only 1.58%, because $f_{S1}>0$ offsets a profile that is low on average. Sections 4 and 7 therefore give both numbers.

**Non-monotone truncation curves** (`I4_rcut_nonmonotone.csv`). Threshold: an increase of more than 0.05 pp at larger R_cut.
- This happens in 7 ladder curves and 38 I1 curves (`report_numbers.json` I4|rcut_nonmonotone_count). One further I4 curve, of B12 (d) at R_fit 4, duplicates an I1 curve.
- Largest cases:
  - B16 (a/d/L*) at R_fit 1: kSZ 4.97-5.08% at R_cut 3 but 15.6-15.8% at R_cut 4; DM (det) 2.44-2.48% vs 11.6-11.9%.
  - B16 (a) at R_fit 1.34: kSZ 4.94% vs 10.06%.
  - B16 (L*) at R_fit 4: kSZ 3.79% vs 6.61%.
  - PL3g+O2 (pressure) at R_fit 2: tSZ 4.45% at R_cut 1.5 vs 13.3% at 4.
  - PL3+O2 (pressure) at R_fit 4: 4.35% vs 9.32%.
  - B12 (d) at R_fit 4: tSZ 0.60% (R_cut 3) vs 0.81% (R_cut 4).
  - BPLall+O2 (pressure) at R_fit 4: 1.91% vs 2.18%.
- Mechanism: removing the 3-4 R200c shells (a negative error) cancels a positive in-profile error (a model that overshoots there). The sensitive-part score always gets worse when R_cut decreases; for example B16 (a) at R_fit 1.34 has S_kSZ 26.0% at R_cut 3 and 15.5% at R_cut 4.
- A lower error at R_cut < 4 is therefore not a better profile. It is not a reason to truncate earlier.

### 6. I2: the Lee22 ladder on A2-ii
Mass bins; sources: `results/ladder_selection_summary.json`, `results/tables/I2_*.csv`, `figures/fig_I2_ladder_chi2_dbic_vs_k.png`. The figure names the selected models per panel; its points are offset horizontally by a factor of 0.86-1.16 per R_fit so that the coincident PB+O2 stars at k 34 are both visible.

| field | R_fit | A2-ii selected (k, χ²/ν floor) | all gas, Test G (k) | σ_d (A2-ii) |
|---|---|---|---|---|
| P | 1.34 | BPLall+O0 (10, 1.43) | BPLall+O0 (10) | 0.28% |
| P | 2 | PB+O2 (34, 1.00) | BPLall+O1 (12) | 0.24% |
| P | 3 | PB+O2 (34, 1.00) | BPLall+O2 (14) | 0.41% |
| P | 4 | BPLall+O2 (14, 1.40; ΔBIC 5.0) | BPLall+O2 (14) | 0.88% |
| E | 1.34 | BPLall+O0 (10, 1.38) | BPLall+O0 (10) | 1.01% |
| E | 2 | BPLall+O2 (12, 1.44) | BPLall+O2 (12) | 0.64% |
| E | 3 | BPLall+O2 (12, 1.13) | BPLall+O2 (12) | 2.11% |
| E | 4 | PL3g+O0 (7, 0.98) | BPLall+O2 (12) | 6.63% |

Parsimony ladder at R_fit 4: best χ²/ν (floor) per k (`I2_parsimony_ladder_A2ii_vs_allgas.csv`).
- **P:**
  - PL1+O0 (k 4) 16.3 → PL2+O0 (5) 4.36 → PL3+O0 (6) 4.04 → BPLa_L22+O0 (7) 2.97 → BPLs_L22+O0 (8) 2.63 → BPLs+O0 (9) 2.08.
  - BPLall+O0 (10) 1.57 → BPLall+O1 (12) 1.51 → **BPLall+O2 (14) 1.40, ΔBIC 5.0, selected** → PB+O0 (30) 1.23 → PB+O2 (34) 1.00, ΔBIC 0.
- **E:**
  - PL1+O0 (4) 5.90 → PL2+O0 (5) 1.66 → PL3+O0 (6) 1.47 (ΔBIC 117).
  - **PL3g+O0 (7) 0.98, ΔBIC 0, selected** → PL3gM+O0 (8) 0.97 (ΔBIC 1.6; fails CV).
  - BPLall_L22+O0 (9) 1.21 (ΔBIC 64) → BPLall+O0 (10) 1.04 (ΔBIC 27).
  - BPLall+O2 (12) 3.99 (ΔBIC 745) → PB+O0 (30) 1.00 (ΔBIC 109) → PB+O2 (32) 4.27.

**Two-halo outer term.** For the electrons at R_fit 4, every model with the linear two-halo outer term (O1/O2) has χ²/ν ≥ 3.99 (floor), against ≤ 1.47 for PL3/PL3g/BPLall/PB with O0. On A2-ii the outskirts sit below $\bar E[1+b\xi_{lin}]$ because neighbour gas is partitioned away. This is Test G's finding, here on every rung.

**Floor size.** For E at R_fit 4 the per-bin reference needs σ_d = 6.6% per shell to reach χ²/ν = 1. No gNFW law with any outer option follows the A2-ii electron outskirts to 4 R200c at the statistical precision. The selection of PL3g+O0 (k 7) is relative to that systematic floor.

**Other likelihoods** (R_fit 4; `I2_key_models_likelihoods.csv`). χ²/ν under stat / floor / L22:

| field | model | stat | floor | L22 |
|---|---|---|---|---|
| E | PL3g+O0 | 20.9 | 0.98 | 15.5 |
| E | PB+O0 | 23.5 | 1.00 | 32.3 |
| E | BPLall+O0 | 60.7 | 1.04 | 28.8 |
| E | PL3+O0 | 94.7 | 1.47 | 29.6 |
| P | PB+O2 | 2.17 | 1.00 | 5.13 |
| P | BPLall+O2 | 2.93 | 1.40 | 4.71 |

Without the floor, no model is acceptable at R_fit 4.

**Break mass** (`I2_mcut_summary.csv`; report_numbers I2|Mcut).
- Free log10 M_cut [Msun] of BPLall on mass cells: P 14.09-14.23, E 14.11-14.27. Over all BPL variants: P 13.91-14.23, E 14.08-14.46.
- Selected models: P 14.10 (R_fit 1.34) and 14.23 (R_fit 4); E 14.14 / 14.13 / 14.12 (R_fit 1.34 / 2 / 3).
- Lee22's converted values (13.77-13.92) are 0.2-0.5 dex lower.
- Fixed-M_cut (_L22) variants on mass cells: ΔBIC 158-415 (P) and 64-992 (E). The best one per R_fit has ΔBIC 158-189 (P) and 64-298 (E). None passes the BIC test.
- On concentration cells:
  - P: fixed-M_cut variants have ΔBIC ≥ 636.
  - E: the best per R_fit has ΔBIC 98 / 354 / 29 / 1.75 (R_fit 1.34 / 2 / 3 / 4).
  - L22best_L22+O0 (E, R_fit 4) has ΔBIC 1.75, so it **passes** the BIC test. It is rejected only by CV: CV − CV_best = 554 against a fold scatter of 90.

**Concentration cells.**
- P selects PBcell+O0/O2 (k 102-106) at every R_fit.
- E selects PBcell+O0 (R_fit 1.34) and PBcell+O2 (R_fit 2).
- At R_fit 3 and 4 no E model satisfies both criteria. The lowest-BIC L22best+O2 (k 14) and L22best+O0 (k 12) are reported (rule_satisfied = False); the free-M_cut L22best+O0 also fails CV (509 vs 90).
- HalfDome has no concentrations, so +c models are painted at the bin-mean concentration.

**What the free gamma does** (`I2_gamma_slopes.csv`; E, R_fit 4, pivot 10^14.5 Msun).

The local slope of the inner profile is
$$\frac{d\ln F}{d\ln x}=\gamma-\beta\frac{x/x_c}{1+x/x_c}$$
with $\gamma$, $\beta$, $x_c$ as in section 2 (`report_i.py:546-571`).

| model | γ | x_c | β | slope at x = 0.1 / 1 / 2 / 3 / 4 | γ − β |
|---|---|---|---|---|---|
| PL3g+O0 | +0.603 | 0.319 | 4.214 | −0.40 / −2.59 / −3.03 / −3.21 / −3.30 | −3.61 |
| PL3+O0 (γ = −0.3) | −0.3 | 0.680 | 3.735 | −0.78 / −2.52 / −3.09 / −3.34 / −3.49 | −4.04 |

- The free gamma does **not** steepen the outer profile: PL3g's slopes at 2-4 R200c are slightly shallower.
- What it does is change the core. With γ > 0 the profile peaks at $x_{peak}=x_c\gamma/(\beta-\gamma)$ = 0.11 (10^13.6), 0.053 (10^14.5) and 0.027 R200c (10^15.4), and decreases inwards. The core radius is smaller and falls faster with mass: $x_c$ goes from 0.70 at 10^13.6 to 0.15 at 10^15.4.
- In the kSZ-weighted residuals (section 7), PL3g roughly halves the A2-ii deficit at 0.3-0.7 R200c and reduces the excess at 1.5-3 R200c.
- The cost is the core: −7.5% (kSZ weight) and −13.2% (DM weight) at x < 0.3. About four fifths of this comes from the unresolved shells of the low-mass bins, where the weighted mean ln(model/stack) is −0.26 (kSZ) and −0.38 (DM). There the painted profile is an extrapolation of the γ > 0 form.
- For pressure, the free γ stays near −0.3 (−0.22 to −0.12) and changes little.

### 7. I3/I4: which model captures the sensitive part
**Errors at R_fit = R_cut = 4**, S1 (profile-only) and S. Sources: `I4_compact_vs_perbin_Rf4Rc4.csv`, `I4_model_table.csv`, `I3_ladder_observables_final.csv`; figures `fig_I4_model_table.png` and `fig_I4_error_score_vs_k.png`.

Pressure models, tSZ (the last column is the difference to B12 (a)):

| model (k) | tSZ S1 (prof) | S_tSZ | vs B12 (a) |
|---|---|---|---|
| B12 (a) (30) | 0.68 (0.45) | 3.3 | – |
| B12 (L*) (30) | 0.80 (0.66) | 3.9 | +0.12 pp |
| B12 (d) (30) | 0.81 (0.49) | 3.2 | +0.13 pp |
| PB+O2 (34) | 1.22 (0.98) | 4.1 | +0.54 pp |
| BPLall+O2 (14, selected) | 2.18 (1.99) | 6.6 | +1.50 pp |
| L22best+O2 (15, c cells, bin-mean c) | 2.75 (2.79) | 7.2 | +2.07 pp |
| PL1+O2 (8) | 5.87 (5.82) | 16.4 | +5.19 pp |
| PL3+O2 (10) | 9.32 (9.35) | 10.9 | +8.64 pp |
| PL3g+O2 (11) | 9.82 (9.85) | 11.2 | +9.14 pp |

Electron models:

| model (k) | kSZ S1 (prof) | DM S1 (prof) | S_kSZ / S_DM |
|---|---|---|---|
| B16 (d) (30) | 0.98 (0.62) | 0.51 (0.49) | 7.7 / 10.7 |
| PL3g+O0 (7, selected) | 1.09 (1.58) | 1.83 (1.87) | 10.4 / 15.4 |
| PL3+O0 (6) | 1.13 (1.36) | 1.83 (1.91) | 8.7 / 10.0 |
| BPLall+O0 (10) | 1.31 (1.48) | 1.05 (1.11) | 6.9 / 8.6 |
| PB+O0 (30) | 1.40 (1.56) | 1.05 (1.08) | 6.8 / 8.5 |
| L22best+O0 (12, c cells) | 1.62 (1.55) | 1.28 (1.36) | 7.7 / 9.0 |
| B16 (a) (30) | 2.27 (1.76) | 1.37 (1.30) | 8.0 / 11.6 |
| PL1+O2 (6) | 3.77 (3.35) | 2.54 (2.47) | 13.4 / 15.2 |
| B16 (L*) (30) | 6.61 (6.03) | 2.91 (2.85) | 9.2 / 11.9 |

**Compact laws against per-bin forms:**
- **Pressure.** Every compact law is worse than B12 per bin, by +1.5 pp (BPLall+O2) to +9.1 pp (PL3g+O2).
- **kSZ.** PL3g+O0, PL3+O0 and BPLall+O0 are better than both B16 (a) and the per-bin ladder law PB+O0, in S1 and in profile-only error.
- **DM.** BPLall+O0 is better than B16 (a) and equals PB+O0, but PL3+O0 and PL3g+O0 are worse than B16 (a).
- B16 (d) is best for both electron observables, with the in-sample caveat of section 2.

At R_fit 3, the best free-γ model, PL3gM+O0 (k 8), gives kSZ 1.56 (1.99) and DM 1.48 (1.59).

**Minimal adequate models** (`I4_minimal_adequate_profile_Rcut4.csv`). Criterion: R_cut = 4, and both the S1 and the profile-only error under the threshold.
- **tSZ ≤ 1%:** any B12 per-bin fit; the smallest R_fit is 1, with B12 (L*) at 0.82% (0.40%). **≤ 2%:** BPLall+O0 (k 10) at R_fit 1.34, 1.44% (1.44%).
- **kSZ ≤ 1%:** only B16 (d) at R_fit 3 and 4. **≤ 2%:** PL3+O0 (k 6) at R_fit 4, 1.13% (1.36%).
- **DM ≤ 1%:** only B16 (d) at R_fit 3 and 4. **≤ 2%:** PL3+O0 (k 6) at R_fit 4, 1.83% (1.91%).
- The looser table `I4_minimal_adequate_final.csv` allows R_cut < 4 and S1 only, so it also picks up cancellation cases (section 5).

**Electron fits that stop at R_fit = 1.34, painted to R_cut = 4** (`I4_Rfit134_extrapolated_to_Rcut4.csv`). kSZ / DM S1:

| model (k) | kSZ / DM | S_kSZ / S_DM |
|---|---|---|
| PL3g+O0 (7) | 2.73 / 2.00 | 11.1 / 14.9 |
| PL1+O0 (4) | 5.59 / 8.84 | 26.4 / 34.8 |
| L22best+O0 (12, c cells) | 6.41 / 3.08 | 20.2 / 24.0 |
| PL3+O0 (6) | 6.57 / 4.58 | 17.3 / 22.0 |
| BPLall+O0 (10) | 8.62 / 5.68 | 19.5 / 24.5 |
| PB+O0 (30) | 8.74 / 5.55 | 19.9 / 24.9 |
| B16 (d) | 9.23 / 8.11 | – |
| B16 (L*) | 9.48 / 7.83 | – |
| B16 (a) | 10.06 / 8.48 | – |

- The per-bin forms and PB/BPLall land at 8.6-10.1% kSZ.
- PL3g+O0 reaches 2.7% / 2.0% only because its extrapolated outskirts happen to compensate. Its S_kSZ = 11.1%, against 6.8-8.0% for R_fit = 4 fits. It is not a reliable shortcut.

**Where each model fails** (`I4_residual_by_region.csv`). Weight-averaged ln(model/stack) at R_fit = R_cut = 4, in %, by region of x (`figures/fig_I4_residual_profiles.png`).

kSZ weight:

| model | 0-0.3 | 0.3-0.7 | 0.7-1 | 1-1.5 | 1.5-2 | 2-3 | 3-4 |
|---|---|---|---|---|---|---|---|
| B16 (a) | −3.4 | +1.5 | −1.1 | −1.4 | +1.9 | +5.2 | +2.9 |
| B16 (d) | −2.0 | +1.3 | −1.5 | −2.2 | +0.4 | +2.7 | −1.0 |
| PB+O0 | +1.0 | −5.2 | −3.7 | +1.0 | +6.7 | +7.1 | −4.6 |
| PL3+O0 | −2.0 | −5.9 | −3.0 | +2.1 | +7.3 | +6.5 | −6.7 |
| BPLall+O0 | +0.7 | −5.0 | −3.6 | +0.9 | +6.5 | +6.7 | −5.3 |
| L22best+O0 | −0.4 | −4.2 | −2.3 | +1.7 | +6.3 | +5.9 | −5.7 |
| PL3g+O0 | −7.5 | −2.9 | −3.1 | −0.1 | +4.3 | +4.7 | −4.7 |
| PL1+O2 | +5.6 | −5.2 | −1.3 | +1.9 | −1.3 | −2.1 | +26.1 |

The electron models fall into three groups:
- **O0 gNFW laws (PB, PL3, BPLall, L22best)** share an S-shaped residual: −2 to −6% at 0.3-1 R200c, +6 to +7% at 1.5-3 R200c, and −5 to −7% at 3-4 R200c.
- **PL3g** reduces this pattern, at the cost of the core.
- **The per-bin B16 fits** have a different, smaller pattern. **PL1+O2** is low at 1.5-3 R200c and +26% at 3-4.

**Pressure** (tSZ weight):
- B12 (a), PB+O2 and BPLall+O2 are within ±0.8% inside 1 R200c, where 82% of the tSZ weight is.
- PL3+O2 and PL3g+O2 are +10.4 / +11.0% at x < 0.3, where tSZ has 33% of its weight. A single mass power law cannot follow the mass trend of the pressure core, which is why PL3/PL3g give 9-10% tSZ errors.
- The broken law BPLall fixes this (+0.8%).

The DM-weighted residuals follow the same pattern.

**Is the sensitive-part score an observable error?** No.
- For example, at 4/4 PB+O0 and BPLall+O0 have the lowest S for the electrons (6.8-6.9% / 8.5-8.6%), but B16 (d) has the lowest kSZ and DM errors.
- The signed residuals of different radii and masses cancel in the observables.
- S measures how far the profile is from the stack where the signal is. It bounds the error that remains if those cancellations change, for example with other cosmologies or other profiles.

### 8. I5: stacking choices propagated through the fit
B12/B16 objective (a), the selected ladder models and PL3 were refitted to each variant's stacks and painted against the A2-ii truth at R_cut = 4. Sources: `I5_variants_final.csv`, `I5_stack_differences.csv`, `I5_parameter_shifts_summary.csv`, `I5_perhalo_rule.csv`, `I5_perhalo_rule_bins.csv`; figures `fig_I5_stacking_choices.png`, `fig_I5_parameter_shifts.png`, `fig_I5_perhalo_rule.png`.

**Likelihood of the I5 refits.** The I5 refits of the Lee22 models use diagonal weights (the objective-(a) convention), because the variants have no bootstrap covariances. Their A2-ii reference rows therefore differ from I3's floor-likelihood fits:

| model | I5 A2-ii row (diagonal) | I3 (floor likelihood) |
|---|---|---|
| PL3g+O0, kSZ | 2.18% | 1.09% |
| PL3+O0, kSZ | 4.13% | 1.13% |
| BPLall+O2, tSZ | 1.66% | 2.18% |

Compare variants only with the I5 A2-ii row. The size of these differences also shows that, for compact laws, the observable error depends on the likelihood by factors of 2-4.

| stack used for the fit | B12 R_fit 4: tSZ | B16 R_fit 4: kSZ / DM | PL3g+O0 R_fit 4: kSZ / DM | B16 R_fit 2: kSZ / DM |
|---|---|---|---|---|
| A2-ii (reference) | 0.68 (0.45) | 2.27 / 1.37 (1.76 / 1.30) | 2.18 / 1.74 | 4.84 / 4.82 |
| all gas | 1.78 (1.53) | 6.64 / 3.80 | 6.51 / 2.98 | 4.94 / 4.88 |
| A2-i | 1.54 | 6.56 / 3.55 | 6.00 / 2.78 | 4.92 / 4.87 |
| unweighted mean | 0.82 | 3.34 / 1.98 | 2.99 / 1.83 | 5.69 / 5.29 |
| median | 2.64 (2.72) | 1.50 / 1.02 (2.01 / 1.04) | 1.71 / 2.76 | 3.38 / 4.61 |
| 0.5-dex bins | 2.95 | 2.88 / 2.81 | 2.76 / 1.53 | 4.57 / 6.07 |
| all gas, X-ray phase | 1.78 | 5.29 / 3.18 | 5.08 / 2.45 | 4.11 / 4.55 |
| per-halo rule, own shells (run 3) | 6.56 (6.38) | 13.00 / 14.56 (12.26 / 14.42) | 6.90 / 3.91 | 17.61 / 16.55 |

- Recently AGN-heated gas kept gives results identical to A2-ii: the stacks differ by ≤ 0.17% (`I5_stack_differences.csv`).
- The rows of `I5_stack_differences.csv` compare stacks on the A2-ii strict-resolved shells (column compared_on). The per-halo rule is compared separately on the shells only it resolves.

**Per-halo resolution rule (maas run 3).**
- **Why a new run.** Runs 1-2 fitted this variant only on the strict shells: `i_g35.stage_fits35` intersects every 0.2-dex variant's mask with the strict base mask, and `stage_lvariants` with strict_mask. On those shells every halo resolves the shell, so the per-halo stack equals A2-ii exactly (`inputs/i0_perhalo_x200.json`: max relative difference 0.0). That variant was therefore the reference by construction. It is flagged as such in `I5_variants_final.csv` (column note) and is no longer shown in fig_I5.
- **What run 3 changes.** It fits the variant on its own resolved shells: 26-31 of the 35 fit shells per bin instead of 18-31. In bins 13.5-13.7 / 13.7-13.9 / 13.9-14.1 the inner edge moves from 0.32 / 0.27 / 0.17 R200c to 0.09 / 0.065 / 0.040 (`I5_perhalo_rule_bins.csv`).
- **The added shells.** These shells, all at x < 0.33, average only the haloes that resolve them:
  - In bins 13.5-13.7, 13.9-14.1 and 14.1-14.3 that is as few as **one halo** (0.3% of the bin's inclusion weight).
  - Those haloes have dense, well-sampled cores. The electron stack there is up to $e^{2.42}\approx11$ times the A2-ii all-halo mean (bin 13.9-14.1; maximum ln ratio 1.43 / 0.82 / 2.42 in the three lowest bins).
  - On the fine grid these shells carry 2.7% (tSZ), 2.5% (kSZ) and 3.8% (DM) of the family weight.
- **The stack itself, painted without a fit.** It is the A2-ii mean wherever no halo resolves a shell. Against the truth mean (`results/perhalo/tables/I_errors_long.csv`, det):
  - $C_\ell^{yy}$: +0.96 / +1.10 / +1.73 / +3.76% at ℓ = 100 / 300 / 1000 / 3000;
  - kSZ: +1.38 / +1.53 / +1.97 / +3.13%;
  - DM: mean +0.50%, variance +5.0%, 99.9% quantile +7.7%;
  - family RMS (profile-only): tSZ 2.20%, kSZ 2.12%, DM 3.53% (`I5_perhalo_rule.csv`).
- **Fitted.** The 3-parameter forms follow the raised cores:
  - B16 (a) at R_fit 4 rises by up to ln 4.41 inside 0.33 R200c (bin 13.9-14.1) and changes by ≤ 0.25 in ln at 1-4 R200c.
  - B12 (a) at R_fit 4 rises by up to ln 2.93 inside 0.33, and in bin 13.9-14.1 it also changes by up to 1.48 in ln at 1-4 R200c (`I5_perhalo_rule_bins.csv`; fig_I5_perhalo_rule).
  - Resulting errors: tSZ 0.68 → 6.56%; kSZ 2.27 → 13.00%; DM 1.37 → 14.56%; at R_fit 2: tSZ 0.88 → 6.85, kSZ 4.84 → 17.61, DM 4.82 → 16.55%.
  - Lee22 laws (diagonal): PL3g+O0 R_fit 4 kSZ 2.18 → 6.90, DM 1.74 → 3.91%; BPLall+O2 (P) R_fit 4 tSZ 1.66 → 3.46%.
  - The one improvement, PL3+O2 (P) at R_fit 4 (3.02 → 1.08%), comes with a larger S (7.5 → 8.8%). It is a cancellation.
- **Conclusion.** For the A2-ii painting target use the strict rule: every halo enters every shell, and the fit uses only shells that all haloes resolve.
- **Caveat.** The truth's unresolved cores are themselves the simulated, resolution-limited profiles of all haloes. Whether a converged core lies closer to the per-halo stack cannot be tested with this simulation.

**Partition.**
- All gas and A2-i differ from A2-ii only beyond about 1.3 R200c: median |Δln| at 2-4 R200c is 0.23 / 0.16 (P) and 0.17 / 0.12 (E).
- At R_fit 2 the variants are equivalent (≤ 0.1 pp).
- At R_fit 4 the fits absorb the neighbour gas. B12 parameters shift by up to 0.16-0.20 in ln (P0, x_c), and B16 ρ0 by up to 0.29-0.38. The errors grow: tSZ 1.5-1.8%, kSZ 6.6%, DM 3.6-3.8%.
- So the partition matters precisely where kSZ and DM need the fit.

**Median.**
- Pressure: the median is biased, with tSZ 2.6-5.2% against 0.7-3.0% for the mean (B12 parameter shifts up to 0.34 in ln x_c).
- Electrons: the median lowers some errors, for example:

| model | kSZ: median vs mean |
|---|---|
| PL3+O0, R_fit 4 | 1.68 vs 4.13% |
| BPLall+O2, R_fit 2 | 1.78 vs 4.25% |
| B16, R_fit 4 | 1.50 vs 2.27% |

- This is a compensation, not a better profile:
  - S is larger for every median fit: B16 R_fit 4 S_kSZ 8.67 vs 7.95% and S_DM 12.39 vs 11.61%; PL3g S_kSZ 12.61 vs 11.20%.
  - The profile-only kSZ error of B16 at R_fit 4 is larger: 2.01 vs 1.76%.
  - For PL3g+O0 the DM error gets worse: 2.76 vs 1.74%.
- The median lies below the mean in the electron outskirts (median |Δln| 0.08 at 2-4 R200c) and partly cancels the models' excess at 1.5-3 R200c. The painting target is the mean.

**Mass binning.** 0.5-dex bins give tSZ 2.95% vs 0.68% and S_tSZ 10.9% vs 3.3%. The per-bin ladder laws (PB) are not defined on that binning.

### 9. Recommendations
See the table in section 0. The reasoning per probe follows.

1. **tSZ.**
   - Fit B12 per 0.2-dex bin to the A2-ii mean with objective (a) (or (d)).
   - R_fit is not critical: every R_fit in 1-4 gives S1 ≤ 1.1%, because 71% of the weight is inside 0.7 R200c of massive haloes.
   - Use R_cut = 4: 0.68% (0.45%). R_cut 3 gives 0.99% (1.00%) for (a); R_cut 2 gives 3.2%.
   - Do not truncate near the 90%-weight radius (1.05-1.15): the truth itself loses 11-22% of $C_\ell^{yy}$ at ℓ ≤ 1000 there.
   - Compact Lee22 laws cost +1.5 pp (BPLall+O2) to +9.1 pp (PL3g+O2), because one mass power law cannot follow the pressure core.
2. **kSZ.**
   - Fit B16 per bin with objective (d) to R_fit 3-4: 0.69-0.98% (0.57-0.62%). This is in-sample tuned to these observables (section 2).
   - With objective (a), use R_fit 4: 2.27% (1.76%).
   - R_fit ≥ 3 is needed for the per-bin forms (R_fit 2: 2.9-4.8%; 1.34: 9.2-10.1%).
   - Truncate at 4: R_cut 3 costs 7-8.5%. 90% of kSZ 1-halo needs 2.6-3.2 R200c at ℓ ≤ 1000.
3. **FRB DM.**
   - Fit B16 (d) to R_fit 4 and truncate at 4: 0.51% (0.49%). This is set by the profile shape, not by the scatter floor (0.18%).
   - R_fit 3 gives 0.76%; objective (a) at R_fit 4 gives 1.37% (1.30%).
   - 26% of the calibrated weight is at 2-4 R200c; R_cut 3 costs about 8%, and the truth's mean DM needs 2.87 R200c for 90%.
   - Only 57% of the mean DM comes from the calibrated bins: the fallback profiles below 10^13.5 Msun matter at least as much.
4. **Compact alternative for the electrons** (one law for all masses).
   - At R_fit 4 with the floor likelihood, PL3+O0 (k 6), PL3g+O0 (k 7) and BPLall+O0 (k 10) give kSZ 1.13 / 1.09 / 1.31%. All are better than B16 (a) (2.27%).
   - For DM only BPLall+O0 (1.05%) beats B16 (a) (1.37%); PL3 and PL3g give 1.83%. So BPLall+O0 is the compact choice if DM matters.
   - Do not add the O1/O2 two-halo term for A2-ii electrons at R_fit 4.
   - These errors depend on the fit likelihood (diagonal refits: 2.2-4.1% kSZ).
5. **Stacking.**
   - Fit the A2-ii inclusion-weighted mean in 0.2-dex bins with the **strict** resolution rule.
   - The per-halo rule biases the low-mass cores high and costs 6.6% tSZ, 13% kSZ and 14.6% DM with B12/B16 at R_fit 4.
   - AGN-kept gas is irrelevant (stacks ≤ 0.17%).
   - All-gas or A2-i stacks bias the R_fit-4 fits (kSZ 6.6%).
   - The median biases pressure; for electrons it lowers errors only by compensation.
   - 0.5-dex bins cost tSZ about 2.3 pp.
6. **Reporting.**
   - Quote S1 errors together with profile-only errors.
   - tSZ and kSZ at 0.5-1% are near the S1 scatter floor; DM is not.
   - Errors at R_cut < 4 can be lowered by cancellation.

### 10. Figures
All under `/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/I_fitting_a2ii/figures/`:
- `fig_I1_rfit_rcut_family_rms.png` (main): R_fit × R_cut grids per objective. DM cells are S1 where the S1 PDF was run, otherwise det, marked "d".
- `fig_I1_profile_vs_S1.png` (main): S1 vs truth and profile-only errors against R_fit at R_cut 4, with the S1 floor.
- `fig_I1_rfit_rcut_score.png`: sensitive-part score.
- `fig_I2_ladder_chi2_dbic_vs_k.png` (main): parsimony ladders, A2-ii vs all gas. The selected models are named in the panel titles. Points are offset horizontally per R_fit.
- `fig_I4_model_table.png` and `fig_I4_error_score_vs_k.png`: model × (R_fit, R_cut) errors and S against k.
- `fig_I4_rcut_curves.png` (main): error against R_cut.
- `fig_I4_residual_profiles.png` (main): residuals per bin, with the importance shown.
- `fig_I5_stacking_choices.png` (main): error by stack used for the fit. The per-halo row is fitted on its own shells; the column labels are rotated.
- `fig_I5_perhalo_rule.png` (main): per-halo vs strict stacks, the weight fraction of the haloes resolving each shell, and the ratio of the fitted B16/B12 profiles.
- `fig_I5_parameter_shifts.png`: B12/B16 parameter shifts by variant. The per-halo line sets the scale.