# Why the non-training seeds share a pressure-parameter offset

The dominant diagnosed effect is inference conditional on the fixed training sky, amplified by fine multipole structure. It is not an approximately 3× increase in physical pressure in the other simulations. All ten maps were painted with exactly the same Battaglia12 parameters. The other nine seeds' original unbounded posterior medians have P0 = 49.89 ± 2.32 and beta = 13.01 ± 0.21, where ± here describes the scatter of medians across nine skies, not a posterior uncertainty. Their offsets are similar, rather than exactly constant.

![Controlled spectral tests](plots/mechanism_summary.png)

## Evidence from controlled changes

These tests perturb the saved seed-100/mask-12345 observation. They are diagnostics, not new physical mock skies. The fixed Gaussian beam, SO baseline deproj0 cross-spectrum definition and trained feature transformation are preserved.

For each multipole ell, D_ell = ell(ell+1)C_ell/(2 pi), where C_ell is the masked angular cross-power. The saved training correction is

$$D_{\mathrm{obs},\ell}=D_{\mathrm{ss},\ell}
+B_\ell(D_{\mathrm{prod},\ell}-D_{\mathrm{ss},\ell}-D_{\mathrm{nn},\ell})
+B_\ell^2D_{\mathrm{nn},\ell}.$$

Here ss denotes the beam-smoothed clean signal, prod the original signal-plus-two-noise-draw cross-spectrum, nn the unfiltered noise–noise cross-spectrum, and B_ell the two-arcminute Gaussian harmonic beam. The middle term is the corrected signal–noise contribution. This is the trained harmonic correction approximation; it is not an exact commutation of masking and beam smoothing. All controls preserve this operator. At a fixed mask the nn term is identical for all simulation seeds, so its differences cancel; for changed masks we subtract its saved B_ell squared contribution separately.

| Change to seed 100 | P0 median | beta median | alpha_m,P0 median |
|---|---:|---:|---:|
| None | 12.23 | 9.66 | -0.04 |
| Mean of the nine other observations | 51.81 | 13.08 | 0.72 |
| Change only the clean signal | 43.22 | 11.52 | 1.18 |
| Change only the signal–noise term | 17.60 | 10.38 | -0.30 |
| Smooth the fractional clean-signal change, sigma_ell = 5 | 13.90 | 9.85 | 0.14 |
| Keep only its removed fine residual, sigma_ell = 5 | 40.09 | 11.70 | 0.97 |
| Keep only its removed fine residual, sigma_ell = 50 | 42.43 | 11.58 | 1.11 |
| Reverse the clean-signal change | 4.56 | 7.13 | -0.09 |
| Change amplitude alone by +0.742% in clean power | 12.43 | 9.71 | -0.04 |
| Generating Battaglia12 parameters | 18.10 | 4.35 | 0.154 |

The smooth sigma_ell = 5 control returns all nine individual simulations to P0 medians between 12.13 and 15.46. The complementary fine residual retains the upward shift. Smoothing the absolute change independently gives P0 = 12.59 at sigma_ell = 50, confirming that the result does not depend on smoothing a fractional ratio.

The mean of nine other skies still produces the shift. Averaging away their independent fluctuations cannot restore the particular training realization. Reversing the fine residual instead moves P0 to 4.67. This is a directional response, not an input-independent high-P0 prediction.

Both parameter-space implementations show the mechanism: the original bounded model gives P0 = 52.16 for the mean new observation, 11.67 for the smooth control and 44.50 for the fine residual. All three independently trained ensemble members give elevated P0 on the other simulations. All loaded modules are in evaluation mode, and the seed-100 log density is invariant to evaluation in a different batch to within 1.91e-06; BatchNorm updating is not responsible.

## Why different random skies can shift in the same direction

The network learns a posterior conditional on one fixed halo realization H_100 and one fixed training window W_12345:

$$q_\phi(\theta\mid D)\simeq p(\theta\mid D,H_{100},W_{12345}).$$

Here theta is the nine pressure parameters, D is the signed beam-corrected pseudo-power cross-spectrum, H is the halo lightcone and W is the apodized cap mask. Training varies theta and the SO noise, but does not marginalize over independent H or W. The intended general-sky inference instead requires integrating over these nuisance realizations. This differs from fitting a truly seed-marginalized simulator.

For independent sky fluctuations epsilon_i about an ensemble spectrum, every difference used in the comparison contains the same reference:

$$\Delta_i=\epsilon_i-\epsilon_{100},\qquad
\mathbb E[\Delta_i\mid\epsilon_{100}]=-\epsilon_{100}.$$

Thus all differences share the negative of seed 100's particular fluctuation. Their signs are not independent tests of a zero-mean parameter bias. Under independent, equal-covariance skies, distinct differences have unconditional cross-covariance equal to the covariance of the reference sky. Nonlinear inference can map their common spectral component into a common pressure-parameter direction.

The controlled smooth/fine tests identify that component in the rapidly varying multipole structure. A fixed catalogue and mask leave a repeatable pattern of halo positions, masses, redshifts and mask mixing in the clean spectrum. The trained fine-spectrum model uses that pattern. A different sky or window changes it, even though the actual pressure law is identical. Noise-only whitening does not model this additional sky/window variation. In the original retained whitened PCA coordinates, clean differences have norms 21.6–39.7. These are reference noise distances, not formal sigma significances or chi-square probabilities.

This is a conditional mismatch, not proof that all new observations lie outside global training support. Nearest-neighbour distances to the broad training pool sit at roughly the 22nd–30th percentiles of held-out training distances. The parameter prior spans alternative spectra that can resemble the new observation at incorrect parameters; global proximity alone does not ensure recovery of the true pressure law. General neural-SBI misspecification failures are discussed by [Cannon, Ward and Schmon](https://arxiv.org/abs/2209.01845) and explicit discrepancy modelling by [Ward et al.](https://arxiv.org/abs/2210.06564).

## Why P0, beta and the alpha parameters move together

The implemented Battaglia pressure shape is

$$\frac{P_e(r\mid M,z)}{P_{200}(M,z)}=
P_0(M,z)\left[\frac{x}{x_c(M,z)}\right]^{-0.3}
\left[1+\frac{x}{x_c(M,z)}\right]^{-\beta(M,z)},\quad x=\frac{r}{R_{200c}},$$

$$A(M,z)=A_0\left(\frac{M}{10^{14}M_\odot}\right)^{\alpha_{m,A}}
\,(1+z)^{\alpha_{z,A}},\qquad A\in\{P_0,x_c,\beta\}.$$

In the last equation the two scaling factors multiply. P_e is electron pressure; P_200 is the common mass/redshift-dependent pressure scale; x_c is a dimensionless radius scale; beta controls the outer decline. The alpha_m and alpha_z coordinates control how each of these three quantities changes with halo mass and redshift. The fixed inner slope is -0.3, the transition exponent is 1, and projection is inside a finite 4R_200c sphere, matching training. See [Battaglia et al.](https://arxiv.org/abs/1109.3711).

The painted Compton-y contribution along a sky direction is the physical pressure integral

$$y=\frac{\sigma_T}{m_ec^2}\int P_e\,dl_{\rm phys}.$$

Here sigma_T is the Thomson scattering cross section, m_e the electron mass, c the speed of light and dl_phys physical line-of-sight length. The integral is restricted to the halo's finite sphere and summed over the catalogue. M is physical M200c, R_200c encloses mean density 200 times the critical density, and z is redshift. Catalogue masses in solar masses divided by h are converted using h = 0.68, and the physical mass threshold is 10^12 solar masses. All available finite nonnegative redshifts are used; the interpolation cache spans z = 0.001–5. No low-redshift-only tSZ selection is applied.

At fixed other parameters, raising P0 increases pressure and raises the clean tSZ power as P0 squared. Raising beta suppresses pressure at finite radius, while raising x_c increases it. They can compensate. The mass and redshift exponents provide further compensations by changing which halo populations contribute most to each angular scale. Consequently a high inferred P0 alongside high beta and altered alpha values need not mean a proportionally larger observable tSZ amplitude.

For example, at fixed halo mass, redshift and radius, set u = x/x_c. Then

$$d\ln P_e=d\ln P_0+\left(0.3+\frac{\beta u}{1+u}\right)d\ln x_c
-\ln(1+u)\,d\beta.$$

The equation shows the compensating directions locally; it does not assert a positive P0–beta correlation in the full posterior. In the stored joint samples from non-training skies, the average x_c–beta correlation is 0.82, while P0–x_c is -0.39. The actual nine-dimensional correlations differ from comparing the directions of marginal-median offsets across observations.

This is why a pure-amplitude explanation fails. Compared with seed 100, other simulations average -6.14% in the low-ell block, -0.063% in the middle block and 0.742% at high ell. One seed has the opposite low-ell shift yet shows the same large P0 increase. A +0.742% clean-power amplitude change alone barely shifts P0. The local derivative matrix has a prior-scaled singular-value ratio of about 1.2e-05, so the full nine-coordinate linear inverse is ill-conditioned; its least-squares coordinate shifts are not reliable nonlinear estimates.

The alpha offsets also have different signs: alpha_m,P0 and alpha_m,beta increase, whereas alpha_m,xc, alpha_z,P0 and alpha_z,xc decrease. These coordinated shifts reflect a coupled fit, not a uniform rescaling of every exponent. The joint-draw correlation matrices are provided separately; marginal medians should not be combined and interpreted as a best-fitting physical profile.

## Mask and combined controls

Mask changes alter the clean spectrum, the signal–noise cross term, and the noise–noise term. The extension separates all three using the actual saved beam operator, rather than attributing all residual changes to signal–noise. For seed 100, clean-only mask changes give most of the P0 shift; noise-only changes do not. The smooth/fine result also holds for changed masks and for the mean of the 81 joint-only cases (nine non-training simulations crossed with nine non-training masks).

| Mean change | Actual P0 | Clean-only P0 | Smooth-only P0 | Fine-only P0 |
|---|---:|---:|---:|---:|
| Other simulations, fixed mask | 51.81 | 43.22 | 13.90 | 40.09 |
| Other masks, fixed seed 100 | 37.82 | 37.37 | 11.34 | 37.73 |
| Both changed | 47.09 | 43.01 | 13.30 | 41.27 |

## What this diagnosis does and does not solve

The existing retrained Delta-ell = 50 model is insensitive to these fine residual controls: P0 stays near 32.7 for the baseline and 32.8 for the mean new observation. This supports the earlier finding that banding reduces realization sensitivity. It does not establish unbiased marginal medians: even the original seed-100 beta median is about 9.7 instead of the generating 4.35, and the banded P0 median remains above 18.1. Broad nonlinear degeneracies and prior volume can displace medians even when the truth is in a high-density joint region.

For the original models, the mean new observation and fine-only control exclude truth at joint 95%, while the smooth controls restore the truth to joint 68%. For the banded models the truth remains inside joint 68% for these controls. Joint regions here use densities in physical theta coordinates, including the parameter-transform Jacobian. This distinction between median bias, joint truth containment and actual calibration must be retained.

The principled remedy is to train while varying catalogues and masks, then test on entire held-out catalogues/masks. A nuisance covariance or residual basis can provide an interim approximation, but it must enter the simulator/compression and retraining consistently. A revised MOPED could downweight the nuisance directions in C_eff = C_noise + C_sky, or project them out before optimizing the pressure score. The nuisance response changes with pressure parameters, and ten skies at one B12 point are insufficient to certify it throughout the nine-dimensional prior. Compressing every fine residual away also loses some real pressure information. The previous 25%-width-budget comparison remains the constraint/robustness tradeoff evidence; this follow-up diagnoses its mechanism rather than claiming a newly calibrated method.

Smoothing a change relative to the known B12 reference is a counterfactual test, not a deployable bias subtraction. It restores the seed-100-conditioned result, including its existing marginal offsets. Subtracting an ensemble offset fitted at the truth would not demonstrate unbiased inference for unknown parameters.

## Checks and reproducibility

- Three jobs ran on idark, with 26 CPUs each: two diagnostic jobs and one small completion job adding two mean-mask controls and generating the report. No repainting or retraining was needed. The source experiment and its trained model files remain frozen.
- The completion job finished the added inference controls but initially failed in report formatting. The formatting bug was corrected and report generation rerun successfully; existing posterior draws were preserved.
- Every catalogue uses the same physical M200c cut, h conversion, redshift selection, B12 parameters, 4R_200c truncation, beam and noise-map hashes. Catalogue selected counts are recorded in diagnostics.json; physical equivalence of raw simulation provenance beyond these stored contracts was not re-established here.
- Existing clean bandpowers reconstruct to 8.9e-16 maximum relative error; original input features reconstruct exactly for all ten fixed-mask seeds.
- Spectral factorization is checked algebraically; the same-mask noise–noise spectra are identical across simulations. Each posterior uses 9,000 equal-mixture draws from three trained networks. Monte Carlo fluctuations of diagnostic medians are small relative to the observed tens-of-units P0 changes; these runs are not precise estimates of sub-percent effects.
- The old 21 B12 noise draws give P0 medians 9.3–23.6, far below the common ~50 on new skies. Twenty of those draws built whitening and are conditional diagnostics, not independent calibration data.
- Empirical support tests use 600 reserved parameter/noise rows. They do not test calibration across independent halo skies. Posterior correlations use 10,000 stored joint samples per simulation.
- Checksums cover the consumed spectra, whitening, features, models and relevant source files; raw halo catalogues were not rehashed or reread, since their saved spectra were reused. Completion markers, job resource records, CSV tables and all PDF/PNG plots are included. No existing HalfDome maps or prior reports were overwritten.

Code: diagnose.py, extended_controls.py, make_report.py, audit_inputs.py, diagnose.pbs, extended_controls.pbs and finalize.pbs. All were created for this follow-up. Full generated file inventory is in FILES_CREATED.md.
