## Sensitivity of tSZ, kSZ and FRB DM (Test H: A2-ii painting target, 67 X200 shells to 4 R200c)

*Corrected version (round 3), 2026-10-14 ~21:30 EEST. Only labels, definitions, local post-processing and two figure titles changed. The numerical results are those of maas attempt 3 and are unchanged; every number here comes from the fetched results or from `H_sensitivity/post/h_post.json` (written by `h_post.py` from those results).*

### H.0 What was computed and where

- **Inputs.** FLAMINGO L1_m9, snapshot 0077 (z = 0): 4,588 central haloes. The ten calibrated stacks (10^13.5-10^15.5 Msun in 0.2 dex of M200c) are built from 326 / 324 / 319 / 327 / 317 / 895 / 326 / 127 / 21 / 6 FLAMINGO haloes (`results/prepare.json`, `post/h_post.json:halo_counts_calibrated_bins`).
- **Two kinds of haloes.** The observables are those of the HalfDome lightcone haloes painted by the engine. "Calibrated haloes" below means the lightcone haloes with $10^{13.5}\le M_{200c}\le10^{15.388}$ Msun, which are painted with the FLAMINGO stacks ($10^{15.388}$ is the most massive FLAMINGO halo). "Fallback haloes" are all other lightcone haloes, painted with the published B12/B16 profiles. The halo counts above are FLAMINGO haloes per stack, not lightcone haloes.
- **Stacks.** Per-halo geometric pressure and electron profiles, with recently AGN-heated gas removed, exported from the stacking-choices cache.
  - Selections: A2-ii, the painting partition (main), and all gas (comparison).
  - Grid: the 67 X200 shells to 4 R200c, plus 8 extension shells to 12 R200c.
- **Normalisation.**
  - Pressure is $P/P_{200c}$ (key `P|geometric|removed|A2ii|all|x200|P200`).
  - Electrons are $E=n_e\mu_{\rm XGP}/(\rho_{\rm crit}f_b)$, with $\mu_{\rm XGP}=(m_e+2m_p/1.76)/0.9$ the XGPaint mass per electron (key `...|rhoc_fb`). Both are dimensionless.
- **Where it ran.** Every heavy stage ran on STRW maas through `cluster_tests.py` (8 workers, nice 19, ionice 7): truths, Jacobians, DM-PDF and Mecke passes, fits, bootstrap, beyond-4R and binning, on the fine bases built by attempt 2.
- **No rerun for this fix.** Everything corrected here is a label, a definition, or an exact function of the fetched arrays (`h_post.py`, 1 s locally).

**Parity with Test F**, at <= 1.2e-12 (`results/parity.json`; `verification.json` G0-G2):
- moments 1.3e-14;
- importance maps summed into F's shells 1.3e-15;
- DM PDF 1.6e-13;
- quantile importance 1.6e-13;
- Mecke Jacobians 1.2e-12;
- deposit identity exactly 0.

**Effect of the finer grid itself.** The fine-grid truth differs from F's 35-shell truth of the same selection (A2-ii; `post/h_post.json:fine_vs_F_truth_absmax`) by at most:
- 4.0e-4 relative at ell <= 3000 (3.96e-4, DM-field 1-halo at ell = 55);
- 3.97e-4 at the selected multipoles (y x DM 1-halo at ell = 8000);
- 5.3e-4 over all 106 multipoles (y x DM 1-halo at ell = 2x10^4);
- 2.7e-4 for the scalar observables.

Sub-shell resolution is therefore negligible.

**DM PDF of the A2-ii truth** (halo DM, z_s = 1; `results/truth_summary.json:pdf.truth_a2ii`): Q16 / Q50 / Q84 / Q97.5 / Q99.9 = 99.8 / 183.2 / 328.9 / 575.6 / 1220.1 pc cm^-3; mean 217.1 pc cm^-3.

### H.1 Definitions (every equation, with symbols and units)

**Importance (F1 convention)**

$$I_O[b,s]=\frac{1}{k_O\,O}\,\frac{\partial O}{\partial \ln F_{b,s}},\qquad \sum_{b,s}I_O[b,s]=1 .$$

- $O$: an observable in its own units; the ratio is unit-free.
  - $C_\ell^{yy}$ is dimensionless.
  - The kSZ $C_\ell$ is dimensionless $(\Delta T/T_{\rm CMB})^2$.
  - DM moments are in pc cm$^{-3}$ (mean) or (pc cm$^{-3}$)$^2$ (variance); the DM-field $C_\ell$ is in (pc cm$^{-3}$)$^2$.
- $F_{b,s}$: the profile of engine mass bin $b$ in fine shell $s$ (dimensionless $P/P_{200c}$ or $E$). Every halo of the bin is scaled by $(1+\epsilon)$ in that shell.
  - In the calibrated bins 13.5-15.5 it is the inclusion-weighted mean FLAMINGO stack.
  - In the fallback bins (below $10^{13.5}$ and $\ge10^{15.5}$) it is the published B12/B16 profile.
  - Engine bin 15.3-15.5 holds both: calibrated haloes ($10^{15.3}$-$10^{15.388}$) and fallback haloes above $10^{15.388}$.
- $k_O$: the degree of homogeneity. It is 2 for quadratic observables (auto spectra, DM variance, DM-field $C_\ell$) and 1 for linear ones (means, DM quantiles, each side of tSZ$\times$DM).
- The sum rule is Euler's theorem. It holds to $1.1\times10^{-14}$ for every moment observable. The DM-quantile maps come from Mecke Jacobians of a 64-halo-per-bin subsample and sum to 0.990-1.000: Q99.9 0.990, Q16 0.997, Q50 0.998 (`verification.json` G3, threshold 0.02; `post/h_post.json:sum_rules`).

**Calibrated-halo share (exact)**

$$c_{\rm cal}=\frac{1}{k_O\,O}\frac{\partial O}{\partial\ln A_{\rm cal}}=\begin{cases}1-O_{\rm fb}/O & \text{linear and one-halo-type terms}\\[2pt] (O+O_{\rm cal}-O_{\rm fb})/(2O) & \text{terms with a 2-halo part}\\[2pt] 1-O(X_{\rm cal}{=}0)/O & \text{side }X\in\{P,E\}\text{ of }y\times{\rm DM}\end{cases}$$

- $A_{\rm cal}$: a common amplitude of all calibrated haloes' profiles.
- $O_{\rm cal}$: the observable with only the calibrated haloes (`truthcal`).
- $O_{\rm fb}$: the observable with only the fallback haloes (the truncation curve at $R_t=0$).
- $O(X_{\rm cal}=0)$: the stored one-side truncation curve at $R_t=0$.
- The fallback part inside the calibrated engine bins is $\sum_{b\in{\rm cal}}I_O[b,\cdot]-c_{\rm cal}$.

The 2-halo case follows from $O=O_{\rm cal}+O_{\rm fb}+2X_\ell$. The 1-halo terms of the two halo sets simply add, and the 2-halo cross term between them is

$$X_\ell=\sum_z\frac{P_{\rm lin}(k_\ell,z)}{V_z}\,a_z\,W_{\rm cal}(z,\ell)\,W_{\rm fb}(z,\ell),\qquad W_{\rm set}(z,\ell)=\sum_{h\in{\rm set},\,z}\frac{N_h\,b_h}{4\pi}\,\tilde u_h(\ell).$$

- $z$: a lightcone redshift shell of the engine. $V_z$ is its comoving volume (Mpc$^3$).
- $P_{\rm lin}(k_\ell,z)$: the linear matter power spectrum (Mpc$^3$) at $k_\ell=(\ell+1/2)/\chi_z$, with $\chi_z$ the comoving distance.
- $W_{\rm set}$: the 2-halo window of the calibrated or fallback haloes, summed over the engine's (z, M) cells of the shell.
  - $N_h$: the number of HalfDome haloes in the cell.
  - $b_h$: their Tinker linear bias (dimensionless).
  - $\tilde u_h(\ell)$: the Fourier transform at $\ell$ of the cell's projected y, $\tau$ or DM profile.
  - Units are such that $P_{\rm lin}W^2/V$ has the units of $O$.
- $a_z$:
  - 1 for tSZ;
  - $\sigma_{v,\rm lin}^2(z)/c^2$ for the approximate kSZ 2-halo term, with $\sigma_{v,\rm lin}$ the linear velocity dispersion;
  - $s_{\rm DM}(z)^2$ for the DM field, with $s_{\rm DM}$ the DM scale of the shell (0 beyond z_s = 1).
- Scaling the calibrated profiles by $A$ gives $O(A)=A^2O_{\rm cal}+O_{\rm fb}+2AX$. Hence $c_{\rm cal}=(O_{\rm cal}+X)/O$, which is the middle case.
- Checks (`verification.json` G9):
  - additivity over 1-halo + 2-halo to 1e-16;
  - for the 1-halo terms the split equals that bin's own signal from `trunc_own` to 5.5e-15;
  - $0\le$ fallback-inside $\le$ the 15.3-15.5 share for every observable.

**Contribution (signal decomposition)**

- Linear observables: $c_s=F_sK_s/O$.
- One-halo quadratic terms: $c_s=\sum_t K_{st}\langle F_sF_t\rangle/O$.
- $K_s$ (units of $O$) is the observable per unit profile in shell $s$, and $K_{st}$ is the corresponding kernel matrix of a calibrated bin.
- $\langle F_sF_t\rangle$ is the per-halo second moment within the bin (dimensionless).

By Euler's theorem the contribution of the calibrated haloes equals their importance exactly. In bins 13.5-15.3 this is the stored map. In engine bin 15.3-15.5 the stored map also contains the fallback haloes above $10^{15.388}$; the split is given below.

The own-shell (diagonal) part is small. The quoted number is

$$f_{\rm diag}=\frac{\sum_{c\in{\rm cal}}\sum_s K_{c,ss}\,\langle F_s^2\rangle_c}{O^{\rm 1h}_{\rm cal}},$$

where $c$ runs over the ten calibrated bins and $O^{\rm 1h}_{\rm cal}$ is the calibrated haloes' 1-halo term (`truthcal`). Values at ell = 100 / 300 / 1000 / 3000 (`results/sens_summary.json:self_fraction`; `post/h_post.json:self_fraction_a2ii`):
- tSZ 1-halo: 3.9 / 4.7 / 5.5 / 6.9%;
- kSZ 1-halo: 4.3 / 4.6 / 5.1 / 7.4%.

Normalised instead by the full 1-halo term $O^{\rm 1h}$ (all haloes), the same sums are 3.8-6.8% (tSZ) and 3.9-4.7% (kSZ) over ell = 100-3000.

The diagonal shrinks as the grid is refined, so a shell's signal is its full row sum over partner shells.

**Truncation curves**

$$T_O(R_t)=\frac{O(R_t)}{O(4R)},\qquad C_O(R_t)=\frac{O(R_t)-O(0)}{O(4R)-O(0)}$$

- $O(R_t)$: the observable with every calibrated halo cut at $R_t$ (in R200c); fallback haloes are unchanged.
- $O(0)$: no calibrated haloes.
- Loss means $1-T$ (a fraction of the total observable).
- $R_X$: the first radius where $C$ reaches $X$ (interpolated in $\ln x$).
- $R_X^{\rm conv}$: the smallest fine edge beyond which $|C-1|\le1-X$ everywhere.

**Profile tolerance**

$$\mathrm{tol}(x)=\frac{0.01}{|k_O\,i(x)\,0.1\ln 10|},\qquad i(x)=\frac{1}{\Delta\ln x_s}\sum_{b\in\{\text{calibrated engine bins 13.5-15.5}\}} I_O[b,s].$$

- This is the coherent fractional error of the stacks over 0.1 dex of radius, in all calibrated bins at once, that changes $O$ by 1%.
- $\Delta\ln x_s$ is the shell width.
- Removing the fallback haloes of bin 15.3-15.5 from $i(x)$ changes the tolerance minima by at most 0.9% (tSZ ell = 1000: 4.25% to 4.29%). This is exact for mean y, mean DM and the DM quantiles, and an estimate for the quadratic terms (`post/h_post.json:tolerance`).

**B12 and B16 forms** ($x=r/R_{200c}$)

$$P(x)=0.5176\,P_0\,q^{-0.3}(1+q)^{-\beta},\quad q=x/x_c;\qquad E(x)=\rho_0\,q^{-0.2}(1+q^{\alpha})^{-(\beta-0.2)/\alpha},\quad q=x/0.5.$$

- $P_0$, $\rho_0$: dimensionless amplitudes in units of $P_{200c}$ and $\rho_{\rm crit}f_b$.
- 0.5176: the electron-to-thermal pressure ratio $P_e/P_{\rm th}$.
- $x_c$: core radius in R200c.
- $\alpha$, $\beta$: dimensionless slopes.

**Objective (a)**

$$r_s=\frac{\ln S_s(\theta)-\ln m_s}{\sigma_s},\qquad \sigma_s=\max(\mathrm{SD}_s/m_s,\;5\%)$$

- The objective is minimised over the resolved shells with $x_{\rm hi}\le R_{\rm fit}$.
- $S_s$: the model's exact shell average.
- $m_s$: the stacked mean.
- $\mathrm{SD}_s$: the halo-to-halo SD.
- $\theta$: the vector of $\ln$ parameters.

**Parameter sensitivity and information**

$$g_O[c,k]=\frac{\partial\ln O}{\partial\theta_k}=\frac1O\sum_s\frac{\partial O}{\partial\ln F_{cs}}\frac{\partial\ln S_{cs}}{\partial\theta_k}$$

- $c$: calibrated bin.
- $\theta_k=\ln p_k$.
- Family Fisher-type matrix: $\mathcal I_{\rm fam}[c]=\frac1{n_O}\sum_{O\in{\rm fam}}g_Og_O^{\rm T}$ (3$\times$3), where $n_O$ is the number of observables in the family.
- A shift $\delta$ of $\ln\theta$ changes the family RMS by $\sqrt{\delta^{\rm T}\mathcal I\delta}$. The tolerance along an eigenvector with eigenvalue $\lambda$ is $0.01/\sqrt\lambda$.

**Fit influence and fit transfer**

$$H=(J^{\rm T}WJ)^{-1}J^{\rm T}W,\qquad T_O[c,s]=\sum_k\frac{\partial O}{\partial\theta_k}\frac{H[c,k,s]}{k_O\,O}$$

- $J=\partial\ln S/\partial\theta$ on the fit shells.
- $W={\rm diag}\,\sigma_s^{-2}$.
- $H[c,k,s]=\partial\theta_k/\partial\ln m_{cs}$.
- $T_O$ is compared with the direct kernel $D_O[c,s]=I_O[c,s]$.

**Scatter model S1**

$$\sigma^2_{Q}=\ln\frac{\langle X_Q^2\rangle}{\langle X_Q\rangle^2},\qquad X_{Q,i}=\sum_{s:\,x_{s+1}\le R_{\rm cut}}V_s\,F^{Q}_{i,s},\qquad V_s=\frac{4\pi}{3}\left(x_{s+1}^3-x_s^3\right)$$

- $Q\in\{P,E\}$; $i$ is a FLAMINGO halo of the bin, and $\langle\cdot\rangle$ the inclusion-weighted average over the bin's haloes.
- $F^{Q}_{i,s}$: the halo's own A2-ii profile in shell $s$, $P/P_{200c}$ or $E$ (dimensionless).
- $V_s$: the shell volume in units of $R_{200c}^3$; $x_s$ are the fine shell edges.
- $X_Q$ is therefore the self-similar scaled integral: $Y(<R_{\rm cut})/(P_{200c}R_{200c}^3)$ or $N_e(<R_{\rm cut})/(\rho_{\rm crit}f_bR_{200c}^3/\mu_{\rm XGP})$.
  - It is not the physical Compton-y volume or electron number.
  - Those would also carry the spread of $M_{200c}$ within the bin ($P_{200c}R_{200c}^3\propto M_{200c}^{5/3}$, $\rho_{\rm crit}f_bR_{200c}^3\propto M_{200c}$), and would give a larger $\sigma^2$.
- Cross term: $\rho_{PE}=\langle X_PX_E\rangle/(\langle X_P\rangle\langle X_E\rangle)$.
- The S1 painting uses the bin-mean profiles $\bar F$. It sets $\langle F_sF_t\rangle=e^{\sigma^2_Q}\bar F^Q_s\bar F^Q_t$ for each field and $\langle P_sE_t\rangle=\rho_{PE}\bar P_s\bar E_t$.
- Values at $R_{\rm cut}=4$ (`results/params.npz:s1`), from bin 13.5-13.7 to bin 15.3-15.5, broadly falling with mass:
  - $\sigma^2_P$ 0.107 to 0.006;
  - $\sigma^2_E$ 0.032 to 0.0034;
  - $\rho_{PE}$ 1.056 to 1.0035.

**Binning error (H5)**

$$e_O^{\rm det}=\frac{O_{\rm painted}^{\rm det}}{\bar O}-1,\qquad e_O^{\rm S1}=\frac{O_{\rm painted}^{\rm S1}}{O}-1,\qquad d_O=e_O(\text{scheme},R_{\rm cut})-e_O(\text{M200c\_0.2 piecewise},R_{\rm cut}),\qquad \mathrm{RMS}_{\rm fam}=\Big(\frac1{n}\sum_{O\in{\rm fam}}d_O^2\Big)^{1/2}.$$

- $O_{\rm painted}$: the observable when every calibrated halo is painted with a stack and cut at $R_{\rm cut}$. The stack is either its scheme bin's stack (piecewise) or one interpolated in log mass between bin means (interp).
  - det: no intrinsic scatter (the bin-mean profiles only).
  - S1: with the S1 scatter above, computed per scheme bin.
- $\bar O$: the truth with each engine bin's mean profile (truth mean). $O$: the scatter-inclusive truth (each halo's own profile).
- Three quantities are reported, each a family RMS in %:
  - **det binning-only**: RMS of $d_O$ with $e^{\rm det}$. At $R_{\rm cut}=4$ the reference painting equals the truth mean exactly ($e=0$ to 1.7e-14), so this is the RMS of $e^{\rm det}$ itself.
  - **S1 binning-only**: RMS of $d_O$ with $e^{\rm S1}$. Subtracting the M200c_0.2 S1 painting at the same cut cancels the S1 model error and the common truncation loss.
  - **S1 total**: RMS of $e^{\rm S1}$ itself, i.e. binning plus the S1 model error.
- Families:
  - tSZ = $C_\ell^{yy}$, kSZ = 1-halo, tSZxDM = total, each at ell = 100 / 300 / 1000 / 3000.
  - DM = mean, variance and Q16 / Q50 / Q84 / Q97.5 / Q99.9 ($n$ = 7); "DM moments" = mean and variance only.
  - The quantiles are evaluated for det at R_cut = 2 / 3 / 4 and for S1 at R_cut = 4 only.
- Sources:
  - `post/h_post.json:binning_family_rms_pct_Rcut4` (R_cut = 4); it reproduces the stored family RMS to 5e-11 points (G9).
  - `results/tables/H_report_binning_only.csv` (every R_cut, DM moments only).

**Double counting**

$$s_c(R_1,R_2)=\frac{\sum_{s\in[R_1,R_2]}V_s\,m^{\rm A2ii}_{c,s}}{\sum_{s\in[R_1,R_2]}V_s\,m^{\rm all}_{c,s}}$$

This is the volume-weighted fraction of all gas in the shell range that A2-ii assigns to the halo itself. $V_s$ is the shell volume and $m$ the stacked means of bin $c$ (P or E). Per-shell ratios are in `results/beyond.npz:share_a2ii_over_all__P/E`.

### H.2 Where the signal comes from (H3)

**Importance by radius band.** Values are % of the observable, A2-ii, all engine bins (`post/h_post.json:bands5`, single rounding).
- **Band edges.** Each band is a sum over whole fine shells. The inner edges are the grid edges 0.2335 and 0.5185 R200c (Test F's shell edges), shown to three decimals in the headers.
- "Peak x" is the radius (R200c, geometric shell centre) of the importance-density peak.
- "Calibrated haloes" is the exact share of the calibrated haloes ($c_{\rm cal}$).
- "Engine bins" is the share of the engine bins 13.5-15.5, including the fallback haloes above $10^{15.388}$.

| Observable | <0.233R | 0.233-0.519R | 0.519-1R | 1-2R | 2-4R | peak x | calibrated haloes | engine bins 13.5-15.5 |
|---|---|---|---|---|---|---|---|---|
| tSZ yy ell=100 | 17.7 | 33.6 | 31.0 | 14.6 | 3.1 | 0.57 | 98.1% | 98.6% |
| tSZ yy ell=1000 | 26.6 | 37.8 | 25.7 | 8.8 | 1.0 | 0.43 | 97.4% | 98.5% |
| tSZ yy ell=3000 | 31.8 | 42.8 | 21.1 | 4.2 | 0.1 | 0.37 | 97.1% | 97.5% |
| mean y | 8.5 | 22.6 | 30.6 | 26.3 | 12.1 | 0.82 | 69.3% | 69.3% |
| kSZ 1h ell=100 | 4.9 | 16.0 | 26.9 | 30.6 | 21.5 | 1.09 | 90.0% | 90.0% |
| kSZ 1h ell=1000 | 6.3 | 19.2 | 29.2 | 29.6 | 15.7 | 1.02 | 82.0% | 82.1% |
| kSZ 1h ell=3000 | 8.9 | 25.0 | 33.3 | 26.1 | 6.7 | 0.82 | 63.1% | 63.1% |
| mean DM | 2.9 | 10.0 | 20.1 | 31.9 | 35.1 | 3.81 | 57.0% | 57.0% |
| DM variance | 13.6 | 28.7 | 30.0 | 20.2 | 7.4 | 0.66 | 75.2% | 75.2% |
| DM Q16 | 0.0 | 0.1 | 4.6 | 32.5 | 62.5 | 3.81 | 47.1% | 47.1% |
| DM Q50 | 0.1 | 2.9 | 15.8 | 38.2 | 42.8 | 2.53 | 52.2% | 52.2% |
| DM Q99.9 | 27.8 | 35.6 | 17.1 | 10.9 | 7.7 | 0.28 | 88.4% | 88.5% |
| DM-field C ell=1000 | 8.4 | 22.9 | 31.0 | 26.8 | 10.9 | 0.87 | 82.5% | 82.6% |
| tSZxDM ell=1000, P side | 19.8 | 35.7 | 29.6 | 12.9 | 2.0 | 0.50 | 95.9% | 96.2% |
| tSZxDM ell=1000, e side | 13.9 | 30.2 | 31.2 | 19.5 | 5.2 | 0.61 | 95.0% | 95.3% |

DM-quantile rows sum to 0.997 / 0.998 / 0.990 (Mecke subsample; see H.1).

**The inner split depends on the edges.** The tSZ importance peaks at 0.37-0.57R, close to these edges, so moving them by one fine shell shifts several points between the inner three bands. With the fine edges nearest below 0.25 and 0.5 R200c (0.247 and 0.473; each shell assigned by its outer edge), the middle band is 0.28 dex wide instead of 0.35 dex (`post/h_post.json:bands5_alt_edges`):

| Observable | <0.247R | 0.247-0.473R | 0.473-1R |
|---|---|---|---|
| tSZ yy ell=100 | 19.4 | 27.2 | 35.7 |
| tSZ yy ell=1000 | 28.8 | 31.0 | 30.4 |
| tSZ yy ell=3000 | 34.6 | 35.5 | 25.7 |
| kSZ 1h ell=1000 | 7.1 | 15.2 | 32.4 |
| mean DM | 3.2 | 7.8 | 21.9 |
| DM variance | 15.0 | 23.2 | 34.2 |
| DM Q99.9 | 30.4 | 29.7 | 20.3 |

- The largest shift is 7.3 points (tSZ ell = 3000, middle band); the 1-2R and 2-4R columns do not change.
- Per-shell values at full resolution are in `results/tables/H3_importance_shells.csv` and `post/H_post_tolerance_shells.csv`.
- The maas run's 8-band table `results/tables/H3_bands.csv` labels its edges 0.1051 / 0.2335 / 0.5185 R200c as 0.1 / 0.25 / 0.5. `post/H_post_bands8.csv` repeats it with the edges used, each value recomputed from `results/sens.npz` (agreement 5e-9, the CSV's 8-digit rounding).

**Fallback haloes inside engine bin 15.3-15.5** (exact; `post/h_post.json:calibrated_shares`; `post/H_post_calibrated_shares.csv`).

| Observable | Share in engine bin 15.3-15.5 | From fallback-painted haloes above $10^{15.388}$ | Fallback fraction of that bin |
|---|---|---|---|
| tSZ yy ell=100 | 18.8% | 0.47 points | 2.5% |
| tSZ yy ell=1000 | 5.9% | 1.13 points | 19% |
| tSZ 1h ell=100 | — | 0.50 points | — |
| tSZ 1h ell=1000 | — | 1.14 points | — |
| kSZ 1h | — | <= 0.09 points | — |
| mean DM | — | 0.006 points | — |
| DM quantiles | — | <= 0.06 points | — |

**Mass shares** (`figures/H3_mass_shares_a2ii.png`, which splits bin 15.3-15.5 into its calibrated and fallback haloes; `sens_summary.json:shares`). Each share belongs to the lightcone haloes of a bin, painted with that bin's stack.
- tSZ at ell=1000 peaks in 14.5-14.9 (17.5 + 19.5%).
- At ell=100 the low-ell tSZ rests on three top bins, whose stacks are built from few FLAMINGO haloes:
  - 14.9-15.1 (stack of 127 FLAMINGO haloes): 30.3%;
  - 15.1-15.3 (stack of 21): 11.4%;
  - 15.3-15.5: 18.8% in total. Of this, 18.3 points come from the lightcone haloes with $10^{15.3}$-$10^{15.388}$ painted with the stack of 6 FLAMINGO haloes. The other 0.47 points come from the fallback-painted lightcone haloes above $10^{15.388}$ (`post/h_post.json:top_bin_15.3-15.5_split_yytot@100`).
- kSZ at ell=1000 is spread over 13.5-14.5 (10.9-14.6% per bin).
- 43.0% of mean DM and 52.5% of DM Q16 come from the fallback bins below $10^{13.5}$, which are not calibrated by FLAMINGO.

**Scatter.** Using per-halo second moments instead of the mean profile raises (`H_digest.json:scatter_boost_truth_over_truthmean_a2ii`):
- tSZ by 1.4 / 3.1 / 4.0% at ell = 100 / 1000 / 3000;
- kSZ 1h by 1.8 / 1.5 / 1.3% at the same ell;
- DM variance by 2.0%.

**A2-ii relative to all gas** (`H_digest.json:a2ii_over_main`):

| Observable | Change |
|---|---|
| yy ell=100 | -1.5% |
| kSZ 1h ell=100 | -6.7% |
| mean DM | -3.2% |
| DM Q16 | -5.6% |
| DM Q50 | -3.9% |

**Profile tolerance.** Values are per 0.1 dex from the calibrated engine bins (`results/sens_summary.json:tolerance`, reproduced in `post/h_post.json:tolerance`; figure `H3_tolerance_a2ii.png`).

| Observable | Tolerance per 0.1 dex |
|---|---|
| tSZ ell=1000 | minimum 4.25% at 0.43R; 4.2-4.9% over 0.3-0.6R; 4.86% at 0.3R; 4.36% at 0.5R; 4.83% at 0.6R; 8.6% at 1R; 17.5% at 0.1R; 23% at 1.5R; 56% at 2R; 278% at 3R |
| tSZ ell=100 | minimum 4.24% at 0.57R; 6.05% at 1R; 26% at 2R |
| kSZ 1h ell=1000 | minimum 5.32% at 0.94R; 7.1% at 0.5R; 8.4% at 2R; 14.9% at 3R; 23% at 3.8R |
| mean DM | 14.1-18.4% over 1-4R (minimum 14.1% at 3.8R) |
| DM Q50 | 11.9-12.9% over 1.5-4R; 21% at 1R |
| DM Q99.9 | minimum 8.9% at 0.28R |
| DM variance | minimum 6.0% at 0.66R |

**Coherent band tolerance** (`post/h_post.json:coherent_band_tolerance`).
- The calibrated stacks carry 20.3% of mean DM in 2-4R. Including the fallback haloes, which a stack error does not touch, the band carries 35.1%.
- A coherent 4.9% error of the calibrated stacks over the whole 2-4R band therefore changes mean DM by 1%.
- It would be 2.9% only if the published fallback below $10^{13.5}$ changed as well.

**Resolution.**
- The first resolved inner edge runs from 0.32R (13.5-13.7) to 0.04R (15.1-15.5) (`post/h_post.json:first_resolved_inner_edge`).
- Unresolved calibrated halo-shells carry the following (`post/h_post.json:unresolved`):

| Observable | % of total observable | % of calibrated-bin importance |
|---|---|---|
| tSZ ell = 100 | 1.8% | 1.8% |
| tSZ ell = 1000 | 3.6% | 3.7% |
| tSZ ell = 3000 | 6.4% | 6.6% |
| mean DM | 1.3% | 2.2% |
| DM Q99.9 | 8.8% | 10.0% |

### H.3 Truncation: radii and losses (A2-ii)

- R50 and R90 are first crossings; R95c and R99c are converged radii (in R200c).
- Losses are % of the total observable for a calibrated-halo cut at 1 / 1.5 / 2 / 3 / 3.63 R200c.
- Sources: `post/h_post.json:radii`, `post/H_post_radii.csv`, rounded once from full precision. DM quantiles come from the truth-mean DM-PDF passes.

| Observable | R50 | R90 | R95c | R99c | loss 1R | 1.5R | 2R | 3R | 3.63R |
|---|---|---|---|---|---|---|---|---|---|
| tSZ yy ell=100 | 0.73 | 1.59 | 2.18 | 3.30 | 29.4 | 11.6 | 5.35 | 1.48 | 0.40 |
| tSZ yy ell=300 | 0.53 | 1.43 | 1.86 | 3.00 | 20.8 | 8.67 | 4.01 | 0.92 | 0.22 |
| tSZ yy ell=1000 | 0.54 | 1.21 | 1.58 | 2.50 | 16.4 | 5.02 | 1.76 | 0.23 | 0.04 |
| tSZ yy ell=3000 | 0.42 | 0.87 | 1.15 | 1.50 | 5.97 | 0.68 | -0.05 | -0.06 | -0.02 |
| mean y | 0.78 | 2.14 | 3.00 | 3.63 | 26.1 | 14.2 | 8.06 | 2.38 | 0.66 |
| kSZ 1h ell=100 | 1.53 | 3.28 | 3.63 | 4.00 | 65.3 | 46.0 | 31.8 | 13.0 | 4.33 |
| kSZ 1h ell=1000 | 1.21 | 2.78 | 3.30 | 4.00 | 49.9 | 31.2 | 19.0 | 6.04 | 1.74 |
| kSZ 1h ell=3000 | 0.80 | 1.55 | 1.86 | 3.63 | 22.1 | 7.19 | 1.01 | -1.10 | -0.47 |
| kSZ 2h ell=100 | 1.83 | 3.49 | 4.00 | 4.00 | 69.1 | 53.7 | 40.1 | 18.0 | 6.30 |
| mean DM | 1.49 | 3.31 | 4.00 | 4.00 | 38.9 | 28.4 | 20.3 | 8.62 | 2.95 |
| DM variance | 0.80 | 2.17 | 3.00 | 4.00 | 29.2 | 15.9 | 9.04 | 2.79 | 0.82 |
| DM Q16 | 2.17 | 3.58 | 4.00 | 4.00 | 57.4 | 47.4 | 36.1 | 16.3 | 5.70 |
| DM Q50 | 1.71 | 3.40 | 4.00 | 4.00 | 46.4 | 34.3 | 24.5 | 10.4 | 3.56 |
| DM Q84 | 1.28 | 3.14 | 3.63 | 4.00 | 33.4 | 22.8 | 15.7 | 6.39 | 2.15 |
| DM Q97.5 | 0.93 | 2.77 | 3.63 | 4.00 | 23.3 | 14.8 | 9.84 | 3.89 | 1.30 |
| DM Q99.9 | 0.48 | 1.81 | 3.00 | 4.00 | 12.7 | 7.60 | 4.93 | 1.91 | 0.63 |
| DM-field C ell=1000 | 1.01 | 2.33 | 3.00 | 3.63 | 42.2 | 23.3 | 12.6 | 3.18 | 0.78 |
| tSZxDM ell=1000 | 0.71 | 1.65 | 2.18 | 3.00 | 29.3 | 12.3 | 5.40 | 0.94 | 0.18 |

- **Negative losses.** At ell >= 3000, cutting the outer gas raises the 1-halo spectrum slightly, because the outer shells' Fourier terms enter with negative sign. The cumulative curve C overshoots 1 by up to 3.7% (yy, ell=8000) and 23.6% (kSZ 1h, ell=8000) (`post/h_post.json:C_max`).
- **Cross spectrum sides.** On its own, the P side of tSZxDM at ell=1000 converges by R95c = 1.50R and the e side by 2.00R.
- **Per-halo DM-PDF check.** The per-halo passes confirm the truth-mean DM-quantile losses: Q16 at 2R is 35.96% per halo against 36.10% from the truth mean.
- **All gas vs A2-ii.** With all gas the losses are larger beyond 2R because neighbour gas sits at 2-4R.
  - For example, kSZ 1h ell=100 at 3R loses 17.4% against 13.0%.
  - A2-ii converges faster: yy ell=100 R95c is 2.18 against 2.5 for all gas.

**Mass dependence.** R90 is computed from each calibrated bin's own 1-halo or linear signal (`tables/H_report_radii_by_bin.csv`; figure `H3_radii_vs_mass_a2ii.png`). Values are in R200c, from 13.5-13.7 to 15.3-15.5:

| Observable | R90 (lowest -> highest mass bin) |
|---|---|
| tSZ 1h ell=1000 | 2.66 -> 0.53 |
| tSZ 1h ell=3000 | 1.98 -> 0.28 |
| kSZ 1h ell=1000 | 3.45 -> 0.54 |
| mean DM | 3.43 -> 3.03-3.13 (nearly mass-independent) |
| DM variance | 2.85 -> 1.51 |

Small haloes are unresolved at a given ell, so their weight follows the integrated profile.

### H.4 Beyond 4 R200c (H4, model-dependent)

The FLAMINGO truth stops at 4R. The table gives the changes if painting were extended to 5 / 6 / 8 / 12 R200c, for A2-ii (`H_digest.json:H4_changes`, `H4_dm_quantiles`; figure `H4_beyond_4R.png`).

| Observable | (ii) B12/B16 fit to 4R, extrapolated | (iii-a) power law fitted at 3-4R |
|---|---|---|
| yy ell=100 | +0.48 / +0.66 / +0.77 / +0.83% | +0.58 / +0.82 / +1.02 / +1.14% |
| yy ell=1000 | +0.03 / +0.03 / +0.03 / +0.03% | +0.03 / +0.04 / +0.03 / +0.03% |
| mean y | +0.93 / +1.39 / +1.80 / +2.04% | +1.17 / +1.83 / +2.52 / +3.05% |
| kSZ 1h ell=100 | +8.8 / +15.4 / +25.3 / +38.4% | +9.5 / +16.9 / +28.1 / +42.3% |
| kSZ 1h ell=1000 | +3.6 / +5.6 / +7.2 / +6.8% | +2.9 / +4.3 / +5.0 / +4.4% |
| mean DM | +7.7 / +14.3 / +25.1 / +41.7% | +7.2 / +13.3 / +23.7 / +39.7% |
| DM variance | +1.5 / +2.5 / +3.6 / +4.6% | +1.5 / +2.4 / +3.5 / +4.5% |
| DM Q16 | +15.4 / +28.7 / +51.2 / +86.0% | +14.2 / +26.8 / +48.2 / +82.0% |
| DM Q50 | +9.3 / +17.2 / +30.2 / +49.9% | +8.6 / +16.0 / +28.4 / +47.5% |
| DM Q99.9 | +1.5 / +2.8 / +4.9 / +7.9% | +1.5 / +2.7 / +4.7 / +7.7% |

tSZ at ell >= 1000 changes by at most 0.04% (0.036%, model iii-a at 6R).

- **Budgets.** Model (ii) adds, as the median over bins, 1.5 / 2.4 / 3.3 / 3.9% of Y(<4R) and 7.3 / 12.7 / 20.1 / 28.5% of N_e(<4R) (`H4_budget_added_median`).
- **Check of the shortcut.** At 12R the per-halo DM-PDF passes agree with the truth-mean ones to within 0.7 points (largest: Q16, 86.7% vs 86.0%).
- **Double counting.** This is measured only to 4R. Ranges over the ten calibrated bins (`beyond_summary.json:double_counting`; per-shell ranges from `results/beyond.npz`, `post/h_post.json:double_counting`):

  | Shells | Pressure, volume-weighted | Electrons, volume-weighted | Pressure, per shell | Electrons, per shell |
  |---|---|---|---|---|
  | 2-3R | 0.88-0.96 | 0.85-0.96 | 0.77-0.99 | 0.78-0.99 |
  | 3-4R | 0.45-0.66 | 0.62-0.82 | 0.28-0.80 | 0.55-0.87 |
  | 3.63-4R | 0.28-0.57 | 0.55-0.77 | 0.28-0.57 | 0.55-0.77 |

  - The halo-owned fraction falls with radius. At 3.63-4R, 43-72% of the pressure and 23-45% of the electrons are nearer another painted halo. For electrons the majority is still owned by the halo.
  - Extrapolation, not measured: if the trend continues, most of the pressure beyond 4R would belong to other painted haloes. For electrons this is not established.
  - A per-halo all-gas extension double counts the shared part. All gas with the 3-4R power law to 12R gives mean DM +99.6% and kSZ 1h ell=100 +140.8%.
- **Conclusion.** DM and low-ell kSZ are not converged at 4R. Any extension beyond 4R must keep a partition like A2-ii or a separate diffuse/2-halo component; a per-halo all-gas tail must not be used.

### H.5 Mass binning (H5)

Family RMS errors in % at R_cut = 4 (definitions and families in H.1; `post/h_post.json:binning_family_rms_pct_Rcut4`; figure `H5_binning.png`). DM includes the five quantiles; "DM mom." is mean + variance only.

**Binning-only error**, det painting (no scatter), against the truth mean. The reference scheme M200c_0.2 piecewise is exact, so these are the binning errors themselves (panel a):

| Scheme | tSZ | kSZ | DM | DM mom. | tSZxDM |
|---|---|---|---|---|---|
| M200c_0.1 | 0.02 | 0.03 | 0.05 | 0.04 | 0.03 |
| M200c_0.2 (engine bins) | 0 | 0 | 0 | 0 | 0 |
| M200c_0.3 | 1.46 | 0.96 | 0.68 | 0.92 | 1.29 |
| M200c_0.5 | 2.80 | 2.02 | 1.43 | 2.04 | 3.45 |
| M500c_0.2 | 3.04 | 0.33 | 0.33 | 0.15 | 0.98 |
| M500c_0.5 | 1.28 | 3.66 | 2.02 | 2.94 | 2.72 |

Interpolating in log M between bin means gives (tSZ / kSZ / DM / DM mom. / tSZxDM):
- M200c_0.1: 0.30 / 0.07 / 0.04 / 0.04 / 0.11;
- M200c_0.2: 0.28 / 0.47 / 0.22 / 0.34 / 0.44.

Interpolation is no better than piecewise here, because the engine uses one moment per 0.2-dex bin.

**With S1 scatter**, against the scatter-inclusive truth.
- The first five value columns are the binning-only error $d_O$: the scheme minus the M200c_0.2 S1 painting, so the S1 model error cancels.
- The last column repeats the total error of the S1 painting (binning plus S1 model; panel b).

| Scheme | tSZ | kSZ | DM | DM mom. | tSZxDM | total: tSZ / kSZ / DM / tSZxDM |
|---|---|---|---|---|---|---|
| M200c_0.1 | 0.01 | 0.03 | 0.06 | 0.04 | 0.04 | 0.48 / 0.63 / 0.20 / 0.73 |
| M200c_0.2 | 0 | 0 | 0 | 0 | 0 | 0.48 / 0.60 / 0.17 / 0.71 |
| M200c_0.3 | 1.31 | 0.88 | 0.63 | 0.86 | 1.19 | 1.52 / 0.65 / 0.64 / 1.01 |
| M200c_0.5 | 2.33 | 2.00 | 1.46 | 2.01 | 3.30 | 2.53 / 1.53 / 1.41 / 2.96 |
| M500c_0.2 | 3.01 | 0.26 | 0.33 | 0.11 | 1.01 | 3.10 / 0.38 / 0.44 / 1.62 |
| M500c_0.5 | 1.80 | 3.51 | 1.90 | 2.84 | 2.51 | 2.09 / 2.99 / 1.87 / 2.20 |

- The M200c_0.2 total (0.48 / 0.60 / 0.17 / 0.71%) is the error of the S1 scatter model itself.
- The total can be smaller than the binning-only error (for example kSZ for M200c_0.5: 1.53% against 2.00%), because the binning error and the S1 model error partly cancel. The binning-only column is the one that measures the binning choice.
- The binning-only error is about the same with and without scatter: M500c_0.2 tSZ is 3.04% (det) and 3.01% (S1); M200c_0.5 tSZ is 2.80% and 2.33%.

**Binning and truncation are separable.** The S1 binning-only error barely depends on R_cut (`tables/H_report_binning_only.csv`; H5 panels c-f; DM moments only): M200c_0.5 tSZ is 3.09% at 1R and 2.33% at 4R. Truncation dominates below 3R: the M200c_0.2 S1 painting itself has total errors of 4.4 / 23.6 / 15.9 / 11.4% at R_cut = 2 (tSZ / kSZ / DM moments / tSZxDM; `results/binning_summary.json`).

**Why M500c bins hurt tSZ.** This is an interpretation and was not tested.
- The stacks are normalised in r/R200c and P200c, and an M500c bin mixes haloes from neighbouring M200c bins at different concentrations.
- The stack-level first-order estimate of the earlier stacking-choices workflow was +3.9% for M500c 0.2 dex (`prior/`), consistent with this result.

### H.6 B12/B16 parameters (H6)

**Fits** (A2-ii, objective (a), fine grid; `params_summary.json:fits`). Per-bin RMS residual:

| Fit | R_fit = 4 | R_fit = 2 |
|---|---|---|
| Pressure (B12) | 2.8-7.5% | 0.4-5.6% |
| Electrons (B16) | 3.2-14% (worst at the edge bins 13.5-13.7 and 15.3-15.5) | 0.8-5.8% |

All fits are accepted.

**Painting errors** (family RMS, cut 4R, `tables/H_report_fit_painting.csv`; %):

| Fit, scatter | tSZ | kSZ | DM | tSZxDM |
|---|---|---|---|---|
| R_fit 4, det | 2.87 | 1.01 | 2.19 | 1.80 |
| R_fit 4, S1 | 0.67 | 2.50 | 1.83 | 1.48 |
| R_fit 2, det | 3.47 | 2.92 | 5.00 | 2.28 |
| R_fit 2, S1 | 0.86 | 4.92 | 4.82 | 1.03 |

The 2R B16 fit overshoots on A2-ii (S1): mean DM +4.9%, Q16 +8.9%, kSZ ell=1000 +7.3%.

**Coherent parameter sensitivity.** These are sums over calibrated bins of d ln O / d ln theta, R_fit 4 (`H_digest.json:H6_coherent_sensitivity`, derived from `params.npz g__*`).

| Observable | P0 or rho0 | x_c or alpha | beta |
|---|---|---|---|
| tSZ ell=100 | +1.96 | +5.13 | -6.44 |
| tSZ ell=300 | +1.95 | +4.63 | -5.50 |
| tSZ ell=1000 | +1.95 | +4.42 | -5.08 |
| tSZ ell=3000 | +1.94 | +3.85 | -4.07 |
| kSZ ell=1000 | +1.65 | +1.75 | -4.51 |
| mean DM | +0.58 | +0.41 | -2.05 |
| DM Q16 | +0.49 | +0.19 | -2.34 |
| DM Q99.9 | +0.89 | +1.41 | -1.76 |

A coherent 1% change in beta changes $C_\ell^{yy}$ by 4.1-6.4% (ell = 3000-100; 5.1% at ell = 1000). The outer slope beta is the most important parameter for every probe.

**Degeneracy** (`params_summary.json:fisher`; figure `H6_parameter_sensitivity.png`):
- tSZ constrains one combination. The eigenvector is about (-0.27 ln P0, -0.63 ln x_c, +0.73 ln beta), the same in every bin. The second eigenvalue is <= 3.7e-3 of the first.
- The tolerance along the constrained direction is 0.0064-0.10 in ln theta, against 0.14-4 along the second direction.
- For electrons, the constrained direction is mostly beta at low mass: (-0.34, -0.18, +0.92) for kSZ at 13.5-13.7. At high mass it mixes in alpha: (-0.26, -0.45, +0.85) at 14.5-14.7.

**Bootstrap** (200 paired halo resamples, `params_summary.json:bootstrap`):
- sigma(ln P0) is 3.0-5.3% below 10^14.9, rising to 7.8 / 12.7 / 35.8% in the top three bins.
- sigma(ln rho0) is 2.7-4.9%, rising to 7.4 / 11.5 / 60%.
- Correlations (pressure, both fit radii): ln P0 with ln x_c from -0.85 to -0.99; ln x_c with ln beta from +0.93 to +0.98.
- Observable SD from the R_fit 4 fits:
  - tSZ 0.90-2.10% (ell = 3000 / 100 / 1000 / 300: 0.90 / 1.08 / 1.17 / 2.10);
  - kSZ 0.37-0.48%;
  - mean DM 0.29%;
  - DM quantiles 0.18-0.46%;
  - tSZxDM 0.44-0.60%.
- Painting the bootstrap stacks without a fit gives about the same: tSZ 0.88-2.11%, kSZ 0.34-0.43%, mean DM 0.23%. Fitting adds no noise.
- The tSZ scatter is sampling-limited (inference). The stacks built from 21 and 6 FLAMINGO haloes paint lightcone haloes that carry 11.4% and 18.3% of $C_\ell^{yy}$ at ell = 100.

**Fit transfer vs direct.** Calibrated bins summed, regions <1R / 1-2R / 2-4R (`params_summary.json:transfer`; figure `H6_fit_transfer.png`).

| Observable | Direct kernel | Through fit, R_fit 4 | Through fit, R_fit 2 |
|---|---|---|---|
| tSZ ell=1000 | 0.890 / 0.086 / 0.009 | 0.899 / 0.054 / 0.021 | 0.871 / 0.103 / 0 |
| kSZ ell=1000 | 0.466 / 0.241 / 0.114 | 0.456 / 0.273 / 0.097 | 0.397 / 0.435 / 0 |
| mean DM | 0.182 / 0.186 / 0.203 | 0.135 / 0.301 / 0.142 | 0.077 / 0.514 / 0 |
| DM Q16 | 0.004 / 0.132 / 0.336 | -0.088 / 0.360 / 0.220 | -0.157 / 0.668 / 0 |

- The totals agree, because a uniform rescaling passes through the amplitude exactly.
- B16 moves the DM-relevant information from 2-4R into 1-2R.
- A change of the stack at 0.3-1R reaches DM through the fit with the wrong sign: the cumulative kernel dips to -0.2 for R_fit 2.
- A fit to 2R carries none of the 2-4R information, which holds 20% of mean DM and 34% of Q16 for the calibrated stacks.
- Finite-difference refits confirm the first-order transfer: kSZ +2% in 2-4R gives 0.394% by refit vs 0.390% predicted; G6 maximum relative difference 1.0%.

### H.7 Answer: to what radius to fit and truncate

| Probe | Truncate at | Fit to | Profile accuracy needed where (per 0.1 dex, all calibrated bins coherently) |
|---|---|---|---|
| tSZ C_ell^yy, ell >= 1000 | >= 2.5R (loss 0.64% at 2.5R, 0.23% at 3R, 1.76% at 2R); 4R is free | 2R is enough (S1 0.86% vs 0.67%) | 4.2-4.9% at 0.3-0.6R; 8.6% at 1R; 17% at 0.1R |
| tSZ, ell 100-300 and mean y | >= 3.6R (loss 0.40 / 0.22 / 0.66% at 3.63R; 1.48 / 0.92 / 2.38% at 3R) | 2-4R | 4.2-6% at 0.5-1R for ell=100; 13% at 1.5R |
| kSZ 1-halo, ell <= 1000 | 4R; not converged (R99c = 4R; loss at 3R 13.0% at ell=100, 6.0% at ell=1000) | 4R (a 2R fit gives +7.3% at ell=1000) | 5.3-8.4% at 0.5-2R (minimum at 0.94R); 15-23% at 3-3.8R |
| FRB DM mean, Q16-Q84 | 4R; not converged (loss at 3R: mean 8.6%, Q84 6.4%, Q50 10.4%, Q16 16.3%) | 4R (2R fit: mean +4.9%, Q16 +8.9%) | mean 14-18% over 1-4R; Q50 12-13% over 1.5-4R; 2-4R holds 35-63% (all bins) and 20-34% (calibrated stacks) |
| DM Q99.9, DM variance | >= 3R (loss 1.91% / 2.79%) | 2-4R | Q99.9: 8.9% at 0.28R; variance: 6.0% at 0.66R |
| tSZ x DM, ell=1000 | >= 3R (0.94%); P side converges by 1.5R, e side by 2R | 4R for electrons | P side 8.6% at 0.5R; e side 9.2% at 0.6R |

**Mass and binning.**
- Low-mass bins need larger radii in R200c units. For mean DM the 90% radius stays at 3.0-3.4R in every bin.
- Paint with the 0.2-dex M200c engine bins (or 0.1 dex). The binning alone costs (family RMS, det and S1 binning-only):
  - 0.6-1.5% for 0.3-dex M200c bins;
  - 1.4-3.5% for 0.5-dex M200c bins;
  - 3.0% in tSZ for M500c 0.2-dex binning, and 1.3-3.7% for M500c 0.5-dex bins.
- The low-ell tSZ depends on the three top bins. Their stacks are built from 127, 21 and 6 FLAMINGO haloes, and the lightcone haloes painted with them carry 30%, 11% and 18% of $C_\ell^{yy}$ at ell = 100.

**Parameters.**
- The B12 combination (x_c, beta) carries tSZ. P0 alone is a pure amplitude: d ln C / d ln P0 = 2 x the calibrated share.
- For kSZ and DM the B16 outer slope beta dominates, and a B16 fit restricted to 2R gives it the wrong 2-4R information.
- This supports Test F/G's choice of fit = cut = 4R on A2-ii, and G's separate physical outer term for electrons.

### Figures

All are under `H_sensitivity/figures/` (.png and .svg), regenerated in this fix:
- `H3_importance_by_mass_a2ii` (main): sensitivity maps by mass bin.
- `H3_importance_probes`: A2-ii vs all gas.
- `H3_cumulative_truncation_a2ii` (main) and `_main`.
- `H3_cumulative_by_mass_a2ii`, `H3_radii_vs_mass_a2ii`.
- `H3_mass_shares_a2ii`: bin 15.3-15.5 split into its calibrated and fallback haloes. The title now says "lightcone haloes painted with the FLAMINGO stacks".
- `H3_tolerance_a2ii`.
- `H3_contribution_split_a2ii`.
- `H4_beyond_4R` (main).
- `H5_binning` (main): panel (b) is now titled as the TOTAL S1 error (binning + S1 model); panels c-f are the S1 binning-only error.
- `H6_parameter_sensitivity`, `H6_fit_transfer` (main).