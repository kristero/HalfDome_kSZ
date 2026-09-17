# Lee22 electron-density profiles: implementation check and 1R200c comparison

Date: 2026-09-16. Reference: Lee, Coulton, Thiele & Ho, arXiv:2205.01710v1 (MNRAS 517, 420). Only v1 exists on arXiv; the appendix table the repository calls "Table A2" (MNRAS numbering) is Table 8 in the arXiv PDF.

## What the paper defines

- Eq. (9): `n200 = 200 rho_cr(z) (Omega_b/Omega_m) / (X_H m_p)`, with `rho_cr(z)` from eq. (5).
- Eq. (10): `n_e(x)/n200 = n0 (x/x_c')^gamma [1 + (x/x_c')^alpha]^(-beta')`, `x = r/R200c`, fixed `alpha = 1`, `gamma = -0.3`.
- Eq. (12): every parameter `A in {n0, x_c', beta'}` follows `A0 (1+z)^alpha_z (c/10)^alpha_c (M/M_cut)^(alpha_m or alpha_m')`, with a common `M_cut` (h^-1 Msun) for the three density parameters; the primed slope applies above `M_cut`. `c = R200/R_scale` (Rockstar/Klypin).
- Table 3 (best fit, concentration + mass break in `x_c'` and `beta'`): n0 (15.7, 0.87, -, -2.09, 0.63); x_c' (2.2, -0.06, -1.45, -0.74, -1.37); beta' (7.5, 0.24, -1.10, -0.39, -1.11); log10 M_cut = 13.75.
- Table 8 (no concentration, mass break in `x_c'` and `beta'`): n0 (6.8, 0.68, -, -2.11); x_c' (7.9, 0.47, -0.45, -0.67); beta' (19.5, 0.70, -0.18, -0.31); log10 M_cut = 13.61.
- Fit range: 27 radial bins over 0.04-1.34 R200, masses 1e13-10^14.8 h^-1 Msun (populated bins in Fig. 6 end near 10^14.2), 20 TNG300 snapshots to z=2. Cosmology TNG (h = 0.6774, Omega_b = 0.0486, Omega_m = 0.3089).

## What the repository implements (`frb_map_generation/lee2022_frb_dm_profile.jl`)

Checked line by line against the tables and equations:

| Item | Status |
|---|---|
| Functional form eq. (10), alpha = 1, gamma = -0.3 | correct |
| Table 3 and Table 8 coefficients, signs, M_cut values | correct |
| h conversion: `M_cut/h` compared with physical M200c, i.e. `M h / M_cut` | correct |
| Mass break continuity at M_cut | correct (self-test) |
| n200 from eq. (9) with `rho_cr(z)` (XGPaint `rho_crit`), X_H = 0.76 | correct as printed |
| Column `n0 * n200 * R200c(physical) * 2 int f dy`, `R200c` from M200c and `rho_cr(z)` | correct |
| Observer-frame factor 1/(1+z_halo) applied once | correct |
| Angular radius `theta/theta200c` with `atan(R200c/D_A)` | correct |
| LOS to 1e5 R200c (same effectively infinite bound XGPaint uses for Battaglia16) | consistent |
| Duffy08 concentration proxy with physical mass and `M h / 2e12` pivot | correct formula |
| n0 pivot for the no-concentration fit | inconsistent (see 1) |
| Overall normalization | physically inconsistent with TNG and Battaglia16 (see 2) |
| Concentration source | Duffy08 is 10-13% below the TNG means the paper quotes (see 3) |
| Preferred fit above the mass range | divergent LOS for M200c > about 2e15 Msun (see 4) |
| Redshift scaling | open question (see 5) |

### 1. n0 pivot of the no-concentration fit

The previous code used the eq. (11) pivot `1e14 Msun` for n0 (a physical mass, no h), while x_c' and beta' used M_cut. Table 8 is described as an eq. (10)+(12) fit, and eq. (12) states that the three density parameters share `M_cut`; a parameter without a second slope is a single power law about `M_cut`. The preferred-fit code already used the common pivot. With the common pivot the no-concentration amplitude rises by `(1e14 h / 10^13.61)^0.68 = 1.42`. At M = 10^13.75 h^-1 Msun, z = 0 the two fits then agree (n0 = 8.5 without concentration versus 9.2-10 for Table 3 at c = 4.3-4.9); with the 1e14 pivot they disagree by 35-40%. The corrected runs use the common pivot.

### 2. Normalization: the Omega_b/Omega_m factor

Direct quadrature of the literal profiles gives the gas mass inside R200c relative to the cosmic allotment `f_b M200c` (mu_e = 2/(1+X_H)):

| M200c [Msun], z | Battaglia16 (XGPaint) | Lee22 no-c literal | Lee22 best literal | no-c x f_b | best x f_b |
|---|---:|---:|---:|---:|---:|
| 3e13, 0 | 0.57 | 2.22 | 3.16 | 0.35 | 0.50 |
| 1e14, 0 | 0.63 | 3.27 | 4.80 | 0.52 | 0.76 |
| 3e14, 0 | 0.69 | 3.71 | 5.40 | 0.59 | 0.85 |
| 1e15, 0 | 0.74 | 3.93 | 10.6 | 0.62 | 1.67 |
| 1e14, 1 | 0.59 | 0.45 | 0.69 | 0.07 | 0.11 |

Read literally, both Lee22 fits put 2-5 times the cosmic baryon budget inside R200c at z = 0, which no TNG halo does. Multiplying the electron density by `f_b = Omega_b/Omega_m` gives 0.35-0.85 f_b, the range of TNG300 gas fractions rising with mass. The same factor is present in XGPaint's Battaglia16 density (`rho_gas = rho_fit f_b rho_crit`, flagged in XGPaint as the correction to Battaglia 2016), and the paper's own Figure 5 is consistent with this: the cyan Battaglia16 curve there matches XGPaint's Battaglia16 divided by f_b, not the physical XGPaint density, at every radius (x = 0.04: 0.0055 versus plotted about 0.004-0.006; x = 1: 0.90 versus plotted about 1). Our literal Lee22 curves also reproduce the plotted fits (x = 1: 0.72-1.2; x = 0.1: 0.008-0.028). So the paper's plotted `n_e/n200` equals `n_e,physical / (f_b n200)` for both the simulation data and the Battaglia16 reference, and the tabulated n0 must be multiplied by f_b to give physical electron densities. The corrected runs apply this factor. Without it, the Lee22 best fit gives 1566 pc cm^-3 at b = 0.01 R200c for a 1e14 Msun halo at z = 0.5 against 875 for Battaglia16, in contradiction with Figure 5, where the Lee22 fit lies below Battaglia16 in the inner halo.

### 3. Concentration proxy

HalfDome halos carry no concentrations, so a mean relation is unavoidable. Duffy08 (WMAP5 NFW c200c) gives 4.98 and 4.10 at 10^13.1 and 10^14.1 h^-1 Msun, z = 0; the paper quotes TNG means of 5.65 and 4.53 for those bins with its own Rockstar/Klypin definition. Through `(c/10)^alpha_c` the Duffy values lower n0 by 7%, raise x_c' by 21% and beta' by 17%. The corrected best-fit runs use the power law through the two quoted TNG means (`5.65 (M h / 10^13.1)^(-0.096)`) with Duffy's `(1+z)^(-0.47)` evolution, because TNG's evolution is not quoted. No concentration scatter is included in either case.

### 4. Extrapolation above the fitted mass range

Above M_cut, beta' of the best fit falls as `M^(-1.10)`, so the outer slope `beta' - gamma` drops below 1 for M200c above about 2e15 Msun at all redshifts (Duffy c): the projected column then diverges and a projected aperture cannot repair it. HalfDome has 14 foreground halos above 2e15 and 868 above the fit limit 10^14.8 h^-1 = 9.3e14 Msun. The corrected runs freeze the shape parameters `x_c'` and `beta'` at the fit limit for more massive halos (`--lee2022-shape-mass-clip=fit`), keeping the amplitude n0 and the concentration on their fitted power laws; the outer slope then stays at about 2. This affects only the 868 most massive foreground halos, whose rays populate the extreme DM tail. Even inside the fit range the best fit implies 1.7 f_b at the upper edge (Table 3 is poorly constrained there; Figure 6 shows no populated bins above 10^14.2 h^-1).

### 5. Redshift scaling (open)

n0 evolves as `(1+z)^(-2.09)` relative to `n200 proportional to rho_cr(z)`. Taken literally this makes the enclosed gas fraction at fixed M200c fall by a factor 4 to 7 between z = 0 and z = 1, unlike TNG, whose gas fractions at fixed mass are nearly constant. If the fitted densities were comoving values normalized by the z = 0 critical density, the physical density would carry an extra `(1+z)^3 / E^2(z)` (1.94 at z = 0.5, 2.52 at z = 1), and the fitted slope would combine a Battaglia16-like `-0.66` with `-1.33`, i.e. about `-2.0`, close to the tabulated `-2.09`. The pressure slopes (Table 1: -1.38, Table 7: -1.89 versus Battaglia12's -0.758) are less conclusive. Because this cannot be settled from the paper, the corrected runs keep the literal `rho_cr(z)` reading, and separate `(1+z)^3/E^2` hypothesis runs show the size of the effect. This is the point to confirm with the authors.

## Products (all local runs, 120k rays, z = 1, NSIDE 4096, projected 1R200c, complete lightcone)

`frb_map_generation/outputs/zsrc1p0_nside4096_nrays120000_allhalos_r200cx1p0_m200c_lee2022_<variant>_seed42/`, made with `run_lee2022_z1_nside4096_1r200c_variants_local.sh`; each provenance file records the option values, cache signature and model family.

Total window (7.3e12-3.8e15 Msun), all 120k rays including zero-DM rays; B16 = XGPaint Battaglia16 at the same 1R200c aperture:

| Variant | Ray/halo intersections | Rays with DM > 0 | Mean DM [pc cm^-3] | Std DM | Max DM |
|---|---:|---:|---:|---:|---:|
| Battaglia16 (reference) | 126,665 | 64.7% | 102.6 | 143.4 | 3625 |
| Lee22 no-c, corrected | 126,754 | 64.9% | 44.7 | 78.1 | 3841 |
| Lee22 best fit, corrected | 126,754 | 64.9% | 50.6 | 118.3 | 5509 |
| Lee22 no-c, corrected + (1+z)^3/E^2 hypothesis | 126,754 | 64.9% | 69.0 | 107.9 | 4381 |
| Lee22 best fit, corrected + (1+z)^3/E^2 hypothesis | 126,754 | 64.9% | 77.0 | 153.7 | 6679 |
| Lee22 no-c, literal (previous conventions) | 126,754 | 64.9% | 198.6 | 340.0 | 13952 |
| Lee22 best fit, literal + Duffy08 c | 126,754 | 64.9% | 313.0 | 681.0 | 31376 |

The 89 extra intersections of the Lee22 runs come from the Lee22 caches being evaluated at slightly different R200c floating-point roundings at the aperture edge; the zero fractions agree to 0.14%.

Reading of the figures:

- With the corrections, both Lee22 fits give about half the mean halo DM of Battaglia16 at 1R200c, and they lie a factor 2 to 5 below TNG around the PDF peak (their peak sits near 10 pc cm^-3, TNG's and Battaglia16's near 60 to 100). The no-concentration and best fits are nearly identical once the same conventions are applied: the concentration and mass break matter mainly in the high-DM tail (best fit is 20 to 50% higher above 1000 pc cm^-3).
- Since the fits reproduce TNG's own z = 0 profiles (Figure 5), the gap to Konietzka's TNG halo DM at z_s = 1 most plausibly comes from the redshift scaling (item 5): the hypothesis variants recover most of it (mean 69 to 77 versus 103 for Battaglia16) and put the halo gas fraction back near TNG values at z = 0.5 (0.64 f_b instead of 0.33 f_b at 1e14). Differences in how TNG's within-r200 DM was defined (spherical membership versus a projected aperture with the profile's long LOS) also enter and are not documented for the catalogue used here.
- The literal conventions (previous implementation) give 2 to 3 times the Battaglia16 mean DM and tails to 14,000 to 31,000 pc cm^-3; they imply several times the cosmic baryon budget inside R200c and should not be used for physical predictions.

## Figures

`frb_map_generation/outputs/publication_comparison_20260916_lee22_1r200c/plots/`:

- `halo_pdf_b16_lee22_1r200c_tng_upper_mass_limits` and `..._to_1e14`: TNG, Battaglia16 and the two corrected Lee22 fits.
- `..._sensitivity_...`: the same plus the literal-normalization variants (what the previous implementation would give at 1R200c).
- `..._zscaling_...`: corrected fits with and without the `(1+z)^3/E^2` hypothesis.
- PDF: `output/pdf/halfdome_b16_lee22_1r200c_vs_tng_20260916.pdf`; tables under `analysis/`.

## Code changes

- `lee2022_frb_dm_profile.jl`: option fields `normalization`, `n0_pivot`, `concentration_source`, `shape_clip_mass_msun`, `redshift_scaling`; TNG-mean concentration; option-aware cache signature, model family and provenance; self-test extended. Defaults reproduce the historical products bit for bit (regression value unchanged).
- `generate_halfdome_z1_dm_mass_windows.jl`: flags `--lee2022-normalization`, `--lee2022-n0-pivot`, `--lee2022-concentration-source`, `--lee2022-shape-mass-clip`, `--lee2022-redshift-scaling`; configuration printout and provenance.
- `run_halfdome_z1_mass_histograms_120k.pbs`: environment pass-through for the five options (defaults unchanged).
- `make_publication_comparisons.py`: `halo_pdf_lee22_figures` and `--lee22-1r200c-only`.
- The historical 3R200c Lee22 products and the earlier cross-correlation results were not touched; they used the literal conventions.

## Addendum (2026-09-16, after the like-for-like spherical comparison)

`LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md` compares the same fits with gas counted only inside the R200c sphere, which is what TNG's within-R200 catalogue does. In the 1e13-1e14 Msun window, where both catalogues contain the same halos (hit fractions 52.6% TNG versus 53.7% HalfDome), the mean DM of hit rays is 138.6 pc cm^-3 in TNG, 36-37 for the Lee22 fits with the Omega_b/Omega_m factor of item 2 (a factor 3.8 too low) and 230-234 without it (a factor 1.66 too high); Battaglia16 inside the sphere gives 92.9. Item 2 therefore over-corrects: the f_b reading cannot be right as implemented, but neither is the literal one. The recommendation stands that the normalization (definition of n200, and whether the fitted profiles are means or medians over halos) be confirmed with the authors before Lee22 is used for absolute predictions; until then the two readings bracket the answer and both are available as options (`--lee2022-normalization=literal|baryon_fraction`).

**Electron-count reading (2026-09-17).** A third option, `--lee2022-normalization=electron_count`, replaces the 1/(X_H m_p) of eq. 9 by the free electrons per unit mass of ionized H+He, (1+X_H)/(2 m_p), a factor X_H(1+X_H)/2 = 0.669 on the literal reading. In the like-for-like spherical test it gives 154-157 pc cm^-3 per hit ray in the 1e13-1e14 window against TNG's 138.6 (1.11-1.13), with tail fractions close to TNG's. It is now the recommended reading, pending confirmation of how n200 was defined in the fit; items 1, 3 and 4 of this note are unchanged.
