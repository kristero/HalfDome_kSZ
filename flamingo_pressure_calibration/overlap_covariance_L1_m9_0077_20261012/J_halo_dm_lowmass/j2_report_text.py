"""Static text blocks of the Test J2 report (definitions and equations); imported by j2_report.py.

Every symbol has one meaning in the whole report (halo index i, mass-bin index j, fit / fine shell s and t, stack shell
g; h_SPH the SPH smoothing length, H the Hartlap factor, h the Hubble parameter, b_T the Tinker bias, c_200 the
concentration, delta_s the fit residuals, eta_DM the DM family error, E_cos the cosmic mean electron density).  The @@key@@
fields are filled by j2_report.py with registered numbers.
"""

DEFINITIONS = r"""
### J2.2 Definitions and equations

**Indices used throughout.** $i$: a halo; $j$: a 0.2-dex mass bin, labelled by its range in $\log_{10}(M_{200c}/M_\odot)$ (e.g. 12.7-12.9); $s$, $t$: radial shells; $g$: a shell of the stack's own grid (defined under *Importance*); $\ell$: the multipole of an angular power spectrum.

**Profiles and targets.** $x=r/R_{200c}$ is the 3D radius $r$ in units of $R_{200c}$; $M\equiv M_{200c}$ (in $M_\odot$). The electron field $E$ and the pressure field $P$ are dimensionless:
$$E=\frac{n_e\,\mu_{\rm XGP}}{f_b\,\rho_c},\qquad P=\frac{P_e}{P_{200}},\qquad P_{200}=\frac{G\,M_{200c}\,200\,\rho_c\,f_b}{2R_{200c}}$$
- $n_e$: free-electron number density [cm$^{-3}$];
- $\mu_{\rm XGP}=1.2632\,m_p$: XGPaint's gas mass per electron ($m_p$: proton mass);
- $f_b=0.1588$: the cosmic baryon fraction; $\rho_c$: the critical density at $z=0$ [$M_\odot\,$Mpc$^{-3}$];
- $P_e$: the electron pressure; $G$: Newton's constant.

The targets are J1's stacks of D0-D3 in the 14 calibrated bins 12.7-15.5 (configuration C14): inclusion-weighted means over the haloes of each bin, treatment T1 (J1's merged $\geq400$-particle shells in 12.7-13.5). The haloes above $10^{15.388}\,M_\odot$ are painted with the published profiles, identically in truth and models.

**Fit forms** (per mass bin, three parameters each, so $k=42$ free parameters over 14 bins):
$$E(x)=\rho_0\Big(\frac{x}{0.5}\Big)^{-0.2}\Big[1+\Big(\frac{x}{0.5}\Big)^{\alpha}\Big]^{-(\beta-0.2)/\alpha}\qquad\text{(B16)}$$
$$P(x)=0.5176\,P_0\Big(\frac{x}{x_c}\Big)^{-0.3}\Big(1+\frac{x}{x_c}\Big)^{-\beta}\qquad\text{(B12)}$$
- $\rho_0$: B16 amplitude (E units); $\alpha$: sharpness of the B16 transition; $\beta$: outer logarithmic slope (both forms);
- $P_0$: B12 amplitude ($P_{200}$ units); $x_c$: B12 scale radius (in $R_{200c}$);
- 0.5176: converts thermal to electron pressure; 0.5 and the exponents $-0.2$, $-0.3$ are fixed.

**Shell averages and data.** A model enters a fit through its exact volume average over a fit shell $s=[x_s,x_{s+1}]$:
$$\bar F_s(\theta)=\frac{3}{x_{s+1}^3-x_s^3}\int_{x_s}^{x_{s+1}}F(x;\theta)\,x^2\,dx$$
- $F$: the fitted field ($E$ or $P$); $\theta$: its parameters; $x_s$, $x_{s+1}$: inner and outer shell edges (in $R_{200c}$);
- evaluated with 32 Gauss-Legendre nodes.

The data $m_s$ are the stacked means on Test F's 35 shells to 4 $R_{200c}$. Each is the volume-weighted mean of the fine X200 shells it contains, and equals Test F/G's 35-shell profile to @@rebin@@ relative.

**Fit masks.** A shell $s$ is fitted only if it passes the bin's mask and $x_{s+1}\leq R_{\rm fit}$ ($R_{\rm fit}$: the fit radius, in $R_{200c}$).
- *Bins 13.5-15.5:* the strict rule of Tests F-I. Every halo of the bin must resolve the shell: $x_s\geq\max(0.04,\,4\epsilon_{\rm soft}/R_{200c})$, at least 20 gas particles, and $x_s\geq2h_{\rm SPH}/R_{200c}$.
  - $\epsilon_{\rm soft}$: gravitational softening length; $h_{\rm SPH}$: the upper-median SPH smoothing length of the halo's gas particles in that shell.
- *Bins 12.7-13.5:* the strict rule trusts only shells from @@strict@@ $R_{200c}$ outward there. These bins use the stack-level rule **S400** instead: a shell is fitted if $x_s\geq\max(0.04,\,{\rm median}_i\,4\epsilon_{\rm soft}/R_{200c})$ and
$$\mathcal N_{j,s}=\sum_{i\in j}n^{\rm gas}_{i,s}\geq400$$
  - $\mathcal N_{j,s}$: gas particles of the selection in shell $s$, summed over the haloes $i$ of bin $j$; $n^{\rm gas}_{i,s}$: halo $i$'s owned particles in shell $s$, recently AGN-heated gas excluded. A2-ii counts are used for D0-D2 and pressure, host-bound counts for D3.
  - $1/\sqrt{400}=5\%$ is the Poisson error of the stacked particle sum.
  - S400 keeps shells inside the hydrodynamic-resolution radius $x_h$ (J1: the first fine edge at or beyond the median over the bin's haloes of $2h_{\rm SPH}/R_{200c}$; @@xh@@ $R_{200c}$ in 12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5). Inside $x_h$ only the integral of the gas is measured, not its shape. J1 showed that the core shape there moves every DM statistic by at most @@core@@%.
- *RC variant (bins 12.7-13.5 only):* fits the grid "$[0,x_h]$ + J1's merged T1 shells outside $x_h$", i.e. the resolved shape plus the measured core integral.
- *D3, every bin:* also requires $\mathcal N_{j,s}\geq400$ host-bound particles. This never binds above $10^{13.5}\,M_\odot$.

**Objectives** (sums over the fitted shells):
$$\chi^2_a=\sum_s\Big[\frac{\ln\bar F_s-\ln m_s}{\sigma_s}\Big]^2,\qquad\sigma_s=\max\big({\rm SD}_s/m_s,\,0.05\big)$$
$$\chi^2_d=\sum_s w_s\,(\ln\bar F_s-\ln m_s)^2$$
$$\chi^2_L=\boldsymbol\delta^{T}\big(C_{L22}/H\big)^{-1}\boldsymbol\delta,\qquad H=\frac{N_{\rm h}-p-2}{N_{\rm h}-1}$$
- ${\rm SD}_s$: reliability-weighted halo-to-halo standard deviation of the bin in shell $s$; 0.05: a 5% floor on the fractional error $\sigma_s$.
- $w_s$: J1's DM-family importance weight $W_{j,g}$ of the definition (defined under *Importance*), summed into fit shell $s$ and normalised to unit sum per bin.
  - Objective (d) is tuned in-sample: the weights are the first-order responses of the same DM statistics to the same truth (Test I).
  - 'dI' (D0 only): Test I's weight formula $w_s=\tfrac12w_s^{\rm kSZ}+\tfrac12w_s^{\rm DM}$ (kSZ-1-halo-family and DM-family weights, each normalised per bin first), applied here to J1's C14 importance maps and fitted in all 14 bins. These are not Test I's weights: Test I computed them from Test H's C10 maps (10 calibrated bins). In 13.5-15.5 the dI fits therefore differ from Test I's published fits by up to @@dimax@@% in the parameters (J2.4).
  - 'dIT' (D0 only): Test I's own published fits in 13.5-15.5 (`I_fitting_a2ii/results/fits35.json`: electrons E|d|4, pressure P|a|4), unchanged, plus the dI fits (electrons) and B12 (a) fits (pressure) in 12.7-13.5. This is the kSZ/tSZ recipe of Test I extended to the new bins without touching its calibrated bins.
- $\boldsymbol\delta$: the vector of the log residuals $\delta_s=\ln\bar F_s-\ln m_s$ over the $p$ fitted shells.
- $C_{L22}$: Lee22's covariance of the mean: the inclusion-weighted sample covariance of the haloes' $\ln F$ in those shells, divided by $N_{\rm eff}=1/\sum_iu_i^2$ ($u_i$: halo $i$'s inclusion weight, normalised to unit sum over the bin).
- $H$: the Hartlap factor; $N_{\rm h}$: number of haloes in the bin; $p$: number of fitted shells. Shells where any halo has $F\leq0$ are dropped from (L).
- (L) has no fit when $N_{\rm h}\leq p+2$ ($H\leq0$: the Hartlap factor is non-positive) or $p<@@pmin@@$ (too few shells; @@pmin@@ is the minimum of `analysis.fitting`). Such bins are painted with their (a) fit; that combination is called (L*). The bins concerned are listed in J2.3 [post_j2/J2_L_substitutions.json].
- All objectives are least squares in $\ln\theta$ with an analytic Jacobian (`analysis.fitting`, unchanged).

**Lee22 ladder** (Test G's `g_selection`, run unchanged per definition on its own stacks, electrons only). Inner profile (the B12 shape with a free amplitude and inner slope):
$$F_{\rm in}(x)=A\,(x/x_c)^{\gamma}(1+x/x_c)^{-\beta}$$
- $A$: amplitude (E units); $x_c$: scale radius (in $R_{200c}$); $\beta$: outer slope; $\gamma$: inner slope.
- $\gamma=-0.3$ unless freed. PL3g has one global $\gamma$; PL3gM has $\gamma=\gamma_0\,(M/10^{14.5}M_\odot)^{a_\gamma}$ ($\gamma_0$: value at $10^{14.5}\,M_\odot$; $a_\gamma$: its mass slope).

Mass laws, for each $\Pi\in\{A,x_c,\beta\}$:
- per bin (PB, $k=42$): $\Pi$ free in every bin;
- power law: $\ln\Pi=\ln\Pi_0+a_\Pi\ln(M/10^{14.5}M_\odot)$, with $\Pi_0$ the value at $10^{14.5}\,M_\odot$ and $a_\Pi$ the mass slope. PL1 applies it to $A$ only, PL2 to $A$ and $x_c$, PL3 to all three;
- broken power law: $\ln\Pi=\ln\Pi_0+a_\Pi\ln(M/M_{\rm cut})$ below and $\ln\Pi_0+a'_\Pi\ln(M/M_{\rm cut})$ above a break mass $M_{\rm cut}$ shared by all broken parameters ($\Pi_0$: value at $M_{\rm cut}$; $a_\Pi$, $a'_\Pi$: slopes below and above). BPLa breaks $A$, BPLs $x_c$ and $\beta$, BPLall all three.
  - $M_{\rm cut}$ is either free or fixed at Lee22's value ('_L22').
  - A free $M_{\rm cut}$ is bounded to $10^{12.9}$-$10^{15.3}\,M_\odot$, because the calibrated range now starts at $10^{12.7}\,M_\odot$ (G's lower bound was $10^{13.5}$).
  - The '_L22' values are $\log_{10}M_{\rm cut}=13.61-\log_{10}0.6774=13.78$ without and $13.75-\log_{10}0.6774=13.92$ with concentration terms (`g_selection.LEE22_MCUT`, `g_selection.H_TNG`). Lee22 quote $M_{\rm cut}$ in $h^{-1}M_\odot$ for IllustrisTNG, the simulation they calibrated on, so their $h=0.6774$ (IllustrisTNG's dimensionless Hubble parameter, $H_0=100\,h$ km s$^{-1}$ Mpc$^{-1}$) converts it to $M_\odot$. FLAMINGO's $h=0.681$ (`analysis/io.py` H_FLAMINGO) does not enter.
- PL1s: PL1 with the shape fixed by a first fit with per-bin amplitudes.

Concentration cells ('massc'): each bin is split into inclusion-weighted concentration quartiles when $N_{\rm h}\geq120$, or halves when $N_{\rm h}\geq40$.
- Terms $a_{c}\ln(c_{200}/10)$ are added to $\ln\Pi$, with $c_{200}$ the SOAP concentration of the halo and $a_c$ a free coefficient per parameter: on $A$ only (PL1cA, PL3cA) or on all three (PL3c3; L22best = Lee22's broken law with $c_{200}$ terms on all three).
- PBcell: $A$, $x_c$, $\beta$ free per cell.
- HalfDome has no concentrations, so the $c_{200}$ models are painted at the bin-mean $\ln c_{200}$.

Outer options:
- O0: none.
- O1: adds $E_{\rm cos}\,[1+b_{\rm T}(M)\,\xi_{\rm lin}(xR_{200c})]$, the linear two-halo electron density. $E_{\rm cos}=$ @@ebar@@ is the cosmic mean electron density in E units, $b_{\rm T}(M)$ the Tinker et al. halo bias, $\xi_{\rm lin}$ the linear matter correlation function at separation $xR_{200c}$ ($z=0$).
- O2: O1 with the inner part multiplied by a transition $[1+(x/x_t)^4]^{-1.5}$. Here $x_t=x_{t0}\,(M/10^{14.5}M_\odot)^{a_{xt}}$ is the truncation radius (in $R_{200c}$), $x_{t0}$ its value at $10^{14.5}\,M_\odot$ (bounded to 0.5-20) and $a_{xt}$ its mass slope (bounded to $\pm3$); both are global free parameters.
- O0 only for D1 and D2 (their profiles end at a boundary); O0, O1 and O2 for D0 and D3.

Likelihoods:
- **stat**: bootstrap covariance of $\ln m_s$ between shells (1000 draws, shrunk to the neighbours' pooled correlation).
- **floor**: stat $+\,\sigma_d^2\,\mathbb 1$, with $\mathbb 1$ the identity matrix and $\sigma_d$ the fractional per-shell systematic floor at which the per-bin reference (PB with its best outer option) reaches $\chi^2/\nu=1$ ($\nu$: degrees of freedom).
- **L22**: Lee22's convention, as objective (L); cells with any $F\leq0$ on the mask are dropped.

Criteria and selection:
- ${\rm AIC}=\chi^2+2k$ and ${\rm BIC}=\chi^2+k\ln n_{\rm d}$, with $\chi^2$ under the floor likelihood, $n_{\rm d}$ the number of data points and $k$ the number of free parameters. $\Delta$BIC (written dBIC in the tables) is relative to the best model of the same definition, cell set and $R_{\rm fit}$.
- Spatial cross-validation (CV): the predictive $\chi^2$ summed over $K=8$ spatial folds.
- **G's selection rule:** the fewest $k$ with $\Delta{\rm BIC}\leq6$ and ${\rm CV}-{\rm CV}_{\rm best}\leq\sqrt K\,{\rm SD}_{\rm folds}$, where ${\rm CV}_{\rm best}$ is the lowest CV and ${\rm SD}_{\rm folds}$ the standard deviation of the per-fold CV $\chi^2$ of that best-CV model. If no model passes both, the lowest-BIC model is taken (rule_satisfied = no).

**Painting and predictions.** Every fitted form is painted as its exact average on J1's 75 fine shells and hard-cut in 3D at $R_{\rm cut}$, the definition's boundary (D1: 1, D2: 2, D0 and D3: 4 $R_{200c}$). It is projected with J1's spherical-chord column, which equals HalfDome's SphericalChordDMProfile (J1 gate G8):
$${\rm DM}(x_\perp)=\frac{2R_{200c}}{1+z}\int_0^{\sqrt{R_{\rm cut}^2-x_\perp^2}}n_e\Big(\sqrt{x_\perp^2+\lambda^2}\Big)\,d\lambda$$
- $x_\perp$: projected impact parameter and $\lambda$: line-of-sight coordinate, both in $R_{200c}$ (with $R_{\rm cut}$ also in $R_{200c}$); $R_{200c}$ in physical pc and $n_e$ in physical cm$^{-3}$, so DM is in pc cm$^{-3}$; $z$: the halo's redshift ($1/(1+z)$ gives the observer-frame DM).

Two predictions are made for each painting:
- **det**: one profile per bin.
- **S1**: lognormal scatter of the whole profile's amplitude per bin:
$$\langle F_sF'_t\rangle=\mathcal M_{FF'}\,F_sF'_t,\qquad \mathcal M_{EE}=\frac{\langle N_e^2\rangle}{\langle N_e\rangle^2},\quad \mathcal M_{PP}=\frac{\langle Y^2\rangle}{\langle Y\rangle^2},\quad \mathcal M_{PE}=\frac{\langle YN_e\rangle}{\langle Y\rangle\langle N_e\rangle}$$
  - $F,F'\in\{E,P\}$: the two painted fields; $F_s$, $F'_t$: their bin-mean painted values in fine shells $s$ and $t$; $\langle F_sF'_t\rangle$: the mean over the bin's haloes of the product of the per-halo values.
  - $N_e$: a halo's electron count inside the definition's boundary (its selection); $Y$: its A2-ii electron pressure integrated inside $4R_{200c}$ (in $P_{200}R_{200c}^3$); $\langle\cdot\rangle$: the inclusion-weighted mean over the bin's haloes (T1 profiles).
  - In the DM PDF the S1 amplitude is a 9-node Gauss-Hermite lognormal mixture with log-variance $\sigma_{\ln}^2=\ln\mathcal M_{EE}$.
- An electron model is paired with the truth's mean A2-ii pressure (and a pressure model with the truth's electrons), as in Test I.

**Errors.** For an observable $O$: $e_O=O_{\rm model}/O_{\rm ref}-1$.
- Headline ('S1'): the S1 painting against J1's scatter-inclusive truth (per-halo profiles and second moments).
- **Profile-only** (in brackets): the det painting against J1's truth mean (the bin-mean stacks painted det).
- DM family error (RMS over the statistics of the halo DM PDF to sources at $z_s=1$):
$$\eta_{\rm DM}=\Big[\frac1{n_O}\sum_{O}e_O^2\Big]^{1/2},\qquad O\in\{\langle{\rm DM}\rangle,\ {\rm Var(DM)},\ Q_{16},Q_{50},Q_{84},Q_{97.5},Q_{99.9}\}$$
  - $n_O$: number of members (7). D1 has 6: its $Q_{16}$ lies in the atom of sightlines that cross no D1 sphere (@@atom@@% of sightlines), the same for every painting, so it is left out.
  - $\langle{\rm DM}\rangle$, ${\rm Var(DM)}$: mean and variance; $Q_{q}$: the $q$-th percentile of the halo-DM distribution.
- DM-field $C_\ell$, $y\times$DM, kSZ and tSZ families: the same RMS over $\ell$ = 100, 300, 1000, 3000 ($C_\ell$: angular power spectrum at multipole $\ell$; $y$: the Compton-$y$ map; DM field: the halo-DM map; tSZ: $C_\ell^{yy}$). The DM-field $C_\ell$, $y\times$DM and tSZ families use the total (1-halo + 2-halo) spectra; kSZ is scored separately for its 1-halo and its 2-halo term.
- **S1 floor** of a definition: $\eta_{\rm DM}$ of its own stack painted with S1, i.e. the error that the scatter model causes when the profile is exact. It is not a lower bound: a fitted profile's errors can partly cancel those of the scatter model, so a painting's S1 error can lie below the floor. For this reason the recommendation rule (below) also uses the profile-only error.

**Importance (J1).** $I_O[j,s]=\dfrac{\partial O/\partial\ln F_{j,s}}{k_O\,O}$: the fractional response of $O$ to a fractional change of bin $j$'s stack $F_{j,s}$ in fine shell $s$; $k_O=2$ for quadratic observables ($C_\ell$, variance), 1 otherwise.
- Family weight $W_{j,g}$: the mean of $|I_O|$ over the family's members (and over $100\leq\ell\leq3000$), summed over the fine shells $s$ inside stack shell $g$. The stack shells $g$ are the fine shells in 13.5-15.5 and J1's merged T1 shells in 12.7-13.5.
- The DM quantile maps are J1's Mecke maps with the whole DM = 0 atom (J1 revision of 2026-10-16, J1 J.2): every map of a quantile outside the atom sums to 1 within subsample noise. D1's $Q_{16}$ lies inside the atom and is left out of D1's DM family.

**Sensitive-part score** (Test I's formula):
$$S=\Big[\frac{\sum_{j,g}W_{j,g}\,d_{j,g}^2}{\sum_{j,g}W_{j,g}}\Big]^{1/2},\qquad d_{j,g}=\ln\frac{\bar F_{j,g}}{m_{j,g}}$$
- $\bar F_{j,g}$: volume average of the painted model of bin $j$ over shell $g$; $m_{j,g}$: the stack; $d_{j,g}=-1$ beyond $R_{\rm cut}$ (no weight there in J2, since every painting is cut at the boundary where the weight ends).
- $S$ is dimensionless: 0.05 is a 5% RMS profile error where the signal comes from.
- $S_{\rm DM}$ uses the DM-family weight; $S_{\rm mean}$, $S_{\rm var}$, $S_{Q50}$, ... use the $|I_O|$ of one statistic.
- The suffix 'fit' restricts the sums to the region $x\geq x_{{\rm m},j}$, where $x_{{\rm m},j}$ is the inner edge of the first shell of bin $j$'s 35-shell fit mask (S400 in 12.7-13.5, strict above; host-bound counts for D3). The region is the same for every painting of a definition, whatever objective was fitted.
  - (L) fits drop the shells where any halo has $F\leq0$, so in 12.7-13.5 they start further out than $x_{\rm m}$. Example, D0 at $R_{\rm fit}=4$: (L) from @@lfirst@@ $R_{200c}$ against mask edges @@mfirst@@ $R_{200c}$ in 12.7-12.9 / 12.9-13.1 / 13.1-13.3 / 13.3-13.5. For the (L*) paintings, $S_{\rm fit}$ therefore also scores shells their (L) fits did not use.
- $S$ is not an observable error: signed residuals at different radii and masses cancel in the observables.

**Candidate paintings** of a definition (recommendation rule, minimal adequate model, per-statistic choices): every B16 per-bin painting (every objective and $R_{\rm fit}$, the RC variants, and dIT for D0) and every mass-cell ladder law with a floor fit at the definition's ladder radii (D0: $R_{\rm fit}=4$; D1: 1; D2: 2; D3: 2 and 4). Since run 2 every one of these @@nlad@@ mass-cell laws is painted. The concentration-cell laws are not candidates: they would be painted at the bin-mean concentration, and HalfDome has no concentrations (they are reported separately in J2.5).

**Minimal adequate model:** the fewest $k$ among the candidates with both the S1 and the profile-only $\eta_{\rm DM}$ at or below the threshold.

**Recommendation rule** (data-driven, `j2_post.py`): among the candidate paintings of a definition, take the one minimising $\max(\eta_{\rm DM}^{\rm S1},\eta_{\rm DM}^{\rm profile})$. Using the maximum avoids preferring an S1 error that is low only because of cancellation against the scatter model. The compact alternative is the ladder law with $k\leq14$ that minimises the same quantity.
"""
