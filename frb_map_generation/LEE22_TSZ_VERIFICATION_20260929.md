# tSZ profile verification: XGPaint, CLASS-SZ, FLAMINGO, TNG (Battaglia12 and Lee22)

2026-09-29. Four workstreams, each re-derived by an adversarial verifier; verifier corrections are marked (corr.).
Products: `outputs/lee22_tsz_verification_20260929/{A_xgpaint_audit,B_class_sz,C_flamingo,D_tng_camels,verify_*,synthesis}/`.
Figures (in `outputs/lee22_tsz_verification_20260929/synthesis/`, with README): fig 1 `fig_tsz1_pressure_profiles.png`,
fig 2 `fig_tsz2_redshift_trend.png`, fig 3 `fig_tsz3_yy_power.png`, fig 4 `fig_tsz4_density.png` (PDFs alongside).

**Bottom line.** No change in XGPaint (or HalfDome) has changed the tSZ profiles: today's code repaints the stored B12,
Lee22 and Lee22-calibrated y maps to ≤ 6e-16, both interpolator caches equal current XGPaint at their nodes, the XGPaint
commits since 2026-09-24 only fix print statements/typos and add tests, and the new `tng_calibrated` option is density-only.
The painting is correct (XGPaint = independent LOS integral = CLASS-SZ conventions), apart from a low-z interpolation
error that matters only at ℓ < 50. What is wrong is the printed Lee22 pressure fit: against both the TNG model (CAMELS)
and FLAMINGO it falls as (1+z)^-2, the `group_particles` comoving-volume bookkeeping error M1, with the right z = 0
amplitude. Corrected, the Lee22 y power is 3.3x Battaglia12 and 2.3x FLAMINGO at ℓ ≥ 1000, now with FLAMINGO's ℓ shape;
most of that excess comes from Lee22's P0 extrapolated above its 10^14.8 h^-1 Msun fit limit.

## 1. Is the painting right?
- **Provenance** (A, vA). Four new discs (halos selected independently) repaint to ≤ 5.6e-16, and every precompile
  cache 09-02..09-29 holds source identical to HEAD. The committed `profiles.jl` before 07fc686 does not parse, so the
  09-09/09-18 maps came from an uncommitted tree; its CRC32c equals HEAD.
- **Profiles.** XGPaint direct y(θ) matches an independent LOS integral to about 1e-9; the production interpolators
  match to ≤ 9e-5 (B12) and 6e-4 (Lee22) for z ≥ 0.1. **Refuted as a general claim** (vA): the linear cache z grid
  is too coarse at low z (outskirt errors up to x35 for the 850 halos at z < 0.02; ≤ 1.6 % at the Lee22 shape clip).
  C_ℓ rises by 16 % and 1.2 % at ℓ = 10 and 50 (B12) and changes by < 1e-3 at ℓ ≥ 100.
- **Conventions** (B, vB). CLASS-SZ's B12 is XGPaint's B12: P_e = 0.5176 P_th, f_b = Ω_b/Ω_m, M200c, physical R200c and
  ρ_c(z), shape equal to 2e-16. Lee22 maps exactly onto CLASS-SZ's generalised B12 (4e-15). The 4R200c projected
  aperture keeps 88-100 % of Y.
- **Map level** (fig 3). At ℓ = 100, 300, 1000, 3000:
  - B12 map / CLASS-SZ = 0.98, 1.00, 0.92, 0.93;
  - B12 map / exact per-halo sum over its own catalogue = 1.00, 0.98, 0.99, 1.01 (Lee22 1.04, 0.99, 1.00, 1.00).

  The 7-8 % CLASS-SZ offset is catalogue vs Tinker08 abundance, a third of it set by the T08 reference cosmology
  (corr.). The maps are point samples: deconvolving the pixel window (as the 09-25 truncation series did) adds 6 % at
  ℓ = 3000. CLASS-SZ pitfalls: default precision is 2.4 % low at ℓ = 100 (21 % at ℓ = 30), and `cosmo_model` silently
  adds a 0.06 eV neutrino (+2.5 %).

## 2. B12 and Lee22 against simulations
Each simulation uses its own f_b and physical ρ_c(z), with R200c physical. Lee22 masses are in h^-1 Msun. P_e = n_e kT,
which is 0.5176 P_th for B12 and CAMELS. TNG = CAMELS IllustrisTNG CV, 27 boxes of 25 Mpc/h; FLAMINGO = L2p8_m9
preliminary stacks.

**Pressure** (fig 1; fig 2, top row).
- **TNG, 10^13.5-14 Msun.** Data / printed Lee22 = 1.02 (1+z)^(1.84 ± 0.06). F_P = truth/(printed x (1+z)^2) is 1.01,
  0.94, 0.96, 0.99 at z = 0-1.5, and 1.00 ± 0.05 in the verifier's independent stack.
- **FLAMINGO, M ≥ 10^13.6.** s_Lee22 - s_FL = -1.97 to -2.07. Lee22 x (1+z)^2 / FLAMINGO in Y(<R200c) = 1.21-1.25,
  flat in z = 0-2.
- **B12** tracks the redshift evolution of both simulations to ≤ 0.3 in the exponent at M ≥ 10^13.5, but is 2-3x TNG in
  the cores of 10^13.2 groups.
- **Emulation.** The Lee22 bookkeeping applied to CAMELS cells reproduces the printed trend (0.99 to 0.89); being an
  algebraic identity, it checks consistency only.

**Verdict: M1 holds for the pressure.** True P_e = printed x (1+z)^2 with the shape unchanged; F_P ≈ 1 in TNG and 1.2 in
FLAMINGO. This is established for 10^13.5-15 Msun. The 10^13-13.5 bin gives (1+z)^2.2, and TNG has no halos above
1.8e14 Msun.

**Density** (fig 2, bottom row; fig 4). The density fit needs (1+z)^3 plus a constant F_n.
- TNG, 10^13.5-14: exponent 3.0-3.1, F_n ≈ 5-5.7 (corr.: D's 6.3-6.5 used a mass-pooled x_e correction). At 10^13-13.5
  the exponent is 3.3 and F_n drifts with z (corr.).
- FLAMINGO, M ≥ 10^13.6: s_Lee22 - s_FL = -2.9 to -3.1, F_n ≈ 7.4.
- C(z) ∝ (1+z)^2.48 under-corrects by 0.4-0.6 in the exponent. Printed Lee22 puts 2-4 f_b of electrons inside R200c
  at z = 0.

## 3. y power (fig 3)
- **Lee22/B12, as painted.** Maps 2.44, 2.70, 1.57, 0.84 at ℓ = 100, 300, 1000, 3000; CLASS-SZ 2.97, 2.45, 1.61, 0.87.
- **Low-ℓ excess.** Above the clip P0 ∝ M^0.8 while the shape stays frozen, so per-halo Y/Y_B12 = 1.03, 1.52, 2.72 at
  6.3e14, 1e15, 2e15 h^-1 Msun (∝ M^0.84, corr.). Only 35 of the 1211 halos above the clip lie at z < 0.1 (corr.), and
  they carry about 95 % of the ℓ = 100 excess.
- **Inside the calibrated range**, with the same halos, Lee22/B12 = 1.28, 1.22, 0.84, 0.55 (corr.).
- **Production maps / FLAMINGO lc0** at ℓ = 300, 1000, 3000, 5000: B12 0.80, 0.72, 0.70, 0.72; Lee22 2.14, 1.16,
  0.60, 0.40 (C's 2 % lower B12 values used a map painted with h = 0.6774; the difference is painter cosmology, not
  truncation, corr.). In FLAMINGO's cosmology all HalfDome spectra drop by about 8 %.

## 4. If the Lee22 pressure is corrected
- **Profiles.** Lee22 x (1+z)^2 agrees with TNG to ≤ 14 % at x = 0.5 (10^13.7, z ≤ 1) and is 1.2x FLAMINGO, flat in z.
  Above the fit range, FLAMINGO's Y200 is 1.6-3.4x lower at z = 0.
- **Power.** CLASS-SZ C_ℓ rises by 1.48, 2.18, 3.90 at ℓ = 300, 1000, 3000. That gives 3.0-3.8x B12 on the catalogue,
  and the estimated map is 3.0, 2.4, 2.3, 2.25x FLAMINGO at ℓ = 300, 1000, 3000, 5000.
- **Source of the excess.** Halos above the fit limit carry 84 %, 73 % and 49 % of the excess over B12 at ℓ = 300,
  1000, 3000 (corr.); below 1e14 Msun the corrected 1-halo term is 1.2-1.4x B12.
- **Consequence.** M1 alone does not give a usable map. The extrapolation above 10^14.8 h^-1 Msun must also be replaced,
  for example by B12 (a hybrid map). This is an estimate: no corrected map has been painted.

| key number | value | source |
|---|---|---|
| stored maps vs today's code | ≤ 5.6e-16 | A, vA |
| B12 map / CLASS-SZ; / exact catalogue sum (ℓ 100, 300, 1000, 3000) | 0.98, 1.00, 0.92, 0.93; 1.00, 0.98, 0.99, 1.01 | B, vB |
| TNG P_e/P200 / printed Lee22 (10^13.5-14 Msun) | 1.02 (1+z)^(1.84±0.06); F_P = 1.00 ± 0.05 | D, vD |
| FLAMINGO s_Lee22 - s_FL, Y(<R200c) (B12 - FL) | -1.97 to -2.07 (-0.26 to +0.03) | C, vC |
| Lee22 x (1+z)^2 / FLAMINGO, Y(<R200c), z = 0-2 | 1.21-1.25 | C, vC |
| density exponent; F_n | TNG 3.0-3.1, 5-5.7; FLAMINGO 2.9-3.1, 7.4 | D (corr.), C |
| B12, Lee22 map / FLAMINGO lc0 (ℓ 300, 1000, 3000) | 0.80, 0.72, 0.70; 2.14, 1.16, 0.60 | synthesis |
| Lee22 x (1+z)^2 / as painted (CLASS-SZ); / FLAMINGO (estimate), ℓ 300, 1000, 3000 | 1.48, 2.18, 3.90; 3.0, 2.4, 2.3 | B, synthesis |
| low-z interpolation bias of C_ℓ (B12) | +16 % (ℓ 10), +1.2 % (ℓ 50), < 1e-3 (ℓ ≥ 100) | vA |

## 5. Caveats and open items
- **TNG.** This is the TNG model at CAMELS resolution and cosmology (h = 0.6711, σ8 = 0.8), not TNG300. There are 9
  halos above 1e14 Msun at z = 0 and 5 at z = 0.47, none above 1.8e14.
- **FLAMINGO.** The stacks are preliminary (8-12 halos per bin above 10^14.4); the full L2p8 run was never made.
  n_e = 0 in star-forming gas, so C's "no-SF" variant is a no-op (corr.); counting that gas shifts slopes by ≤ 0.08.
- **Lee22's pipeline** (Rockstar) is not public, so a^2 rests on `y_prof.cpp` and the two simulation trends. Asking
  B. K. K. Lee or L. Thiele would also address F_n.
- **Next steps.** Paint Lee22 x (1+z)^2 on idark, with and without B12 above 10^14.8 h^-1 Msun. Refine the low-z grid
  of the tSZ caches. Record the CLASS-SZ precision and pixel-window issues in `TSZ_TRUNCATION_RADIUS_SERIES_20260925.md`.
  Reconcile the density exponent 3.0-3.3 with C(z)'s 2.48 before using C(z) at z > 1.
