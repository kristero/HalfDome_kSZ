# The electron-density normalization of Lee et al. (2022): why equations 7 and 9 do not fit together, and what the codes do

Date: 2026-09-17. Repository commit at the time of writing: see `git log` (branch `cluster`).
Companion documents: `LEE22_IMPLEMENTATION_CHECK_20260916.md` (transcription check of the fits) and
`LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md` (comparison with TNG inside the R200c sphere).

## 1. Summary

Lee et al. (2022, arXiv:2205.01710, hereafter Lee22) fit generalized-NFW profiles to the electron density of
IllustrisTNG-300 halos and tabulate the fit amplitude `n0` in units of a reference density `n200`. Their eq. 7
computes the electron density of each gas cell correctly from the simulation's electron abundance. Their eq. 9
defines the reference density with the hydrogen mass fraction `X_H` in the *denominator*,
`n200 = 200 rho_cr Omega_b / (X_H m_p Omega_m)`. That number density is larger than the number of nucleons in the
gas it describes, so it cannot be an electron density, and it is not the form that eq. 7 would give for the same
gas. Since `n0` is only defined relative to whatever `n200` the fitting code used, a reader who uses eq. 9 as
printed inherits a constant factor.

The size of that factor is testable. With HalfDome's halo catalogue and the same spherical R200c membership as
the TNG within-R200 catalogue, eq. 9 as printed predicts 1.66 times TNG's own halo DM in the mass range the
fits cover; writing `n200` with `X_H` in the numerator (a factor `X_H^2 = 0.578`) reproduces TNG to 2-4% in the
mean and 1% in the median, and treating the profile exactly as XGPaint treats Battaglia16 (no Lee22-side hydrogen
factor, XGPaint's own electron conversion, a factor 0.602) reproduces it to 0-2%. A residual redshift-dependence problem
remains (Section 5) and should be put to the
authors together with the `X_H` question.

## 2. What the paper defines (arXiv v1, Section 2.3, page 4-5)

Quoted from the text extracted from the PDF; equation numbers are those of v1.

- Eq. 5: `rho_cr(z) = 3 H0^2 / (8 pi G) [Omega_m (1+z)^3 + Omega_Lambda]`.
- Eq. 6 (pressure per cell): `V_i P_e,i = x_i m_i epsilon_i 4 X_H (gamma - 1) / (1 + 3 X_H + 4 X_H x_i)`.
- Eq. 7 (electron number per cell): `V_i n_e,i = x_i m_i X_H / m_p`,
  "where x is the electron abundance, m the mass, epsilon the internal energy, X_H = 0.76 the primordial
  hydrogen mass fraction, gamma = 5/3 the adiabatic index, and m_p the proton mass."
- Eq. 8 (radial profile): `A_alpha = V^-1(b_alpha) sum_{i in b_alpha} V_i A_i`, i.e. a volume-weighted mean in each
  of 27 logarithmic radial bins between 0.04 and 1.34 R200.
- Eq. 9 (self-similar amplitudes): `P200 = 200 G M200 rho_cr Omega_b / (2 R200 Omega_m)` and
  `n200 = 200 rho_cr Omega_b / (X_H m_p Omega_m)`.
- Eq. 10: `n_e(x)/n200 = n0 (x/x_c')^gamma [1 + (x/x_c')^alpha]^(-beta')` with `x = r/R200`, fixed `alpha = 1`,
  `gamma = -0.3`; then "we calculate the average profile ... <n_e/n200>".

Halos come from a Rockstar catalogue of the gravity-only TNG300 run; the gas cells are those of the
hydrodynamical TNG300-1 run around those centres, so all gas inside 1.34 R200 (satellites included) enters the
mean.

## 3. Why eq. 7 is right and eq. 9, as printed, cannot be

**Eq. 7 is the correct electron count.** In IllustrisTNG the cell field `ElectronAbundance` is defined as
`x_e = n_e / n_H`, the electron number density relative to the *total* hydrogen number density (Nelson et al.
2019, data specifications). With `n_H = X_H rho / m_p`, the electron density is `n_e = x_e X_H rho / m_p`, and
multiplying by the cell volume gives eq. 7 exactly: `V_i n_e,i = x_i m_i X_H / m_p`. Eq. 6 is likewise the
standard TNG recipe for the electron pressure (`P_e = n_e k T` with `T = (gamma-1) epsilon mu m_p / k` and
`mu = 4 / (1 + 3 X_H + 4 X_H x_e)`). Both are consistent and physically normalized.

**Eq. 9's `n200` is not an electron density.** For gas of mass density `rho` the nucleon number density is
`rho / m_p` and the free-electron density of fully ionized H+He is

    n_e = rho / m_p * [X_H + (1 - X_H)/2] = rho (1 + X_H) / (2 m_p) = 0.88 rho / m_p      (X_H = 0.76),

which is also what eq. 7 gives with the ionized-gas electron abundance `x_e = (1 + X_H)/(2 X_H) = 1.158`. Every
electron belongs to a nucleon, so no electron density can exceed `rho / m_p`. The printed reference density,
`200 rho_cr f_b / (X_H m_p) = 1.32 * (200 rho_cr f_b / m_p)`, exceeds it by 32%. The pressure half of eq. 9, by
contrast, is the standard self-similar `P_Delta` of Battaglia et al. (2012, eq. 9 there) with `Delta = 200` and
the `Omega_b/Omega_m` factor. The natural explanation for the asymmetry is that `X_H` moved across the fraction
bar when eq. 9 was written down from the same quantities as eq. 7.

**Candidate readings of `n200` and the factor each implies relative to eq. 9 as printed:**

| Reading | `n200` | Factor on eq. 9 as printed | Meaning |
|---|---|---|---|
| A. as printed | `200 rho_cr f_b / (X_H m_p)` | 1 | not an electron density (32% more electrons than nucleons) |
| B. eq. 7 form, unit electron abundance | `200 rho_cr f_b X_H / m_p` | `X_H^2 = 0.578` | hydrogen electrons only, the direct analogue of eq. 7 with `x = 1` |
| C. ionized H+He electron count | `200 rho_cr f_b (1+X_H) / (2 m_p)` | `X_H (1+X_H)/2 = 0.669` | what XGPaint uses for Battaglia16 (Section 6) |
| D. an extra `Omega_b/Omega_m` | `f_b * (A)` | `0.158` | the "corrected" reading of `LEE22_IMPLEMENTATION_CHECK_20260916.md`, since retracted |
| E. XGPaint-native | none on the Lee22 side: `P0 = 200 n0`, XGPaint `ne2d` converts | `0.9 X_H m_p / m_per_e = 0.602` | Lee22 treated exactly like Battaglia16 in XGPaint (Section 6); no assumption about eq. 9 |

Because `n0 = <n_e/n200>` was fitted with the code's own `n200`, the fit is internally consistent whatever that
`n200` was; the question is only which one it was, and the paper's text does not allow deciding it. The
following test does.

## 4. Empirical test against TNG's own within-R200 halo DM

Setup (details in `LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md`): 120,000 rays to `z_s = 1` through the HalfDome
halo lightcone, gas counted only inside the R200c sphere of each halo (chord-limited line of sight, exactly the
membership rule of Ralf Konietzka's TNG within-R200 catalogue), M200c window `10^13 - 10^14 Msun`, which is
inside Lee22's fitted mass range (`10^13 - 10^14.8 h^-1 Msun`) and the only window where both catalogues
contain the same halos (HalfDome resolves nothing below `7.3e12 Msun`). Hit fractions agree to 2% (TNG 52.6%,
HalfDome 53.7%), so the geometry is matched and the residual is the gas profile alone.

| Model inside the R200c sphere | mean DM of hit rays [pc cm^-3] | ratio to TNG | median | % of hits below 3 / 10 / 30 |
|---|---|---|---|---|
| TNG within R200 | 138.6 | 1 | 80.8 | 0.57 / 4.5 / 20.1 |
| Lee22, reading A (as printed) | 230.4 / 234.0 | 1.66 / 1.69 | 138.5 | 0.14 / 1.6 / 10 |
| Lee22, reading B (`X_H^2`) | 133.1 / 135.2 | 0.96 / 0.98 | 80.1 | 0.42 / 4.3 / 20.4 |
| Lee22, reading C (`X_H(1+X_H)/2`) | 154.1 / 156.5 | 1.11 / 1.13 | 94.8 | 0.33 / 3.3 / 17 |
| Lee22, reading D (`x Omega_b/Omega_m`) | 36.4 / 37.0 | 0.26 / 0.27 | 22 | 4.8 / 26 / 61 |
| Lee22, reading E (XGPaint-native, `ne2d`) | 138.6 / 140.8 | 1.00 / 1.02 | 83.5 | 0.39 / 4.0 / 19.5 |
| Battaglia16 (XGPaint normalization) | 92.9 | 0.67 | 59.6 | 0.53 / 5.4 / 26 |

(The two Lee22 numbers are the no-concentration fit of Table A2 and the best fit of Table 3.) Readings B and E
reproduce the simulation the profiles were fitted to (E to 0-2%); reading A, the printed one, does not. Reading E makes no
assumption about eq. 9: it treats the Lee22 profile exactly as XGPaint treats Battaglia16 and lets XGPaint's electron
conversion act, which is the consistent choice when both profiles are used in the same pipeline. Figures:
`outputs/publication_comparison_20260916_sphere_1r200c/plots/halo_pdf_sphere_1r200c_b16_lee22_tng_{xh2_,efix_,sensitivity_,}{upper_mass_limits,to_1e14}.png`.

## 5. The part a constant factor does not fix: redshift dependence

Integrating each profile over the R200c sphere gives the gas mass it implies, here in units of the cosmic
allotment `f_b M200c` (electrons converted to mass with `mu_e = 2/(1+X_H)`; `n0` pivot `M_cut`, TNG-mean
concentration for the best fit):

| Model | 3e13 Msun: z = 0 / 0.5 / 1 | 1e14 Msun: z = 0 / 0.5 / 1 |
|---|---|---|
| Battaglia16 (XGPaint) | 0.56 / 0.53 / 0.51 | 0.63 / 0.61 / 0.59 |
| Lee22 no-c, reading A | 3.07 / 1.01 / 0.46 | 4.63 / 1.45 / 0.63 |
| Lee22 no-c, reading B (`X_H^2`) | 1.77 / 0.58 / 0.26 | 2.68 / 0.84 / 0.37 |
| Lee22 no-c, reading C | 2.05 / 0.68 / 0.31 | 3.10 / 0.97 / 0.42 |
| Lee22 no-c, reading D | 0.48 / 0.16 / 0.07 | 0.73 / 0.23 / 0.10 |

Two things follow. First, reading B matches TNG's DM at `z_s = 1` because rays to `z = 1` mostly cross halos at
`z = 0.3 - 0.8`, where the fit implies a TNG-like 0.6-0.9 `f_b` of gas inside R200c. Second, the same fit implies
1.8-2.7 `f_b` at `z = 0`, more baryons than exist, and 0.3-0.4 `f_b` at `z = 1`. The enclosed fraction of the fit
scales as `n0(z) propto (1+z)^-2.1` relative to `rho_cr(z)`, i.e. it falls by a factor 3.4 between `z = 0` and
`z = 1`, whereas Battaglia16's falls by 10% and TNG's gas fractions at fixed mass are, to our knowledge, nearly
constant. So the tabulated redshift exponent of `n0` most likely also carries a bookkeeping convention (for
instance comoving cell densities normalized with the `z = 0` critical density, which would contribute
`(1+z)^3/E^2(z)`; see item 5 of `LEE22_IMPLEMENTATION_CHECK_20260916.md`). No constant factor makes the implied
gas fraction both physical at `z = 0` and TNG-like at `z = 0.5`; reading D is right at `z = 0` and reading B at
`z = 0.5`. The DM test at `z_s = 1` is therefore evidence for reading B *at the redshifts that matter for
FRBs*, not proof that eq. 9 is the only issue.

## 6. How XGPaint normalizes the Battaglia16 electron density (for comparison)

Local checkout `/home/kn18001/.julia/dev/XGPaint` (XGPaint.jl v0.4.0, WebSky-CITA, commit 5dd0b57 (2026-07-13); `src/profiles.jl` carries uncommitted local edits in that checkout, the three other files cited below are at the commit). Line numbers refer to this checkout.

- `src/profiles_tau.jl:35`: `f_b = Omega_b / OmegaM`. The cosmic baryon fraction is a struct field, 0.158 for the
  HalfDome cosmology (`Omega_b = 0.049`, `Omega_c = 0.261`).
- `src/profiles_tau.jl:88-94` (`rho_2d`): the projected gas density is
  `rho_fit * f_b * rho_crit(z) * r200c`, with the comment "mistake in battaglia 2016: need f_b to convert from m
  to gas". So the Battaglia et al. (2016) fit is read as a density in units of the *matter* critical density and
  multiplied by `f_b` to make it a gas density. We have not re-derived that claim; what matters here is that the
  result is physically normalized (0.51-0.63 `f_b` inside R200c, Section 5).
- `src/profiles_tau.jl:97-106` (`ne2d`): electrons per unit gas mass are
  `1 / (m_e + (2 X_H/(1+X_H)) m_p + ((1-X_H)/(2(1+X_H))) 4 m_p) = (1+X_H) / (2 m_p + (1+X_H) m_e)`, the ionized
  H+He count of reading C, with `X_H = 0.76`; the result is multiplied by a hard-coded `0.9`, undocumented in the
  code (most plausibly the fraction of halo baryons in ionized gas rather than stars).
- `src/profiles_y.jl:63-65` (`generalized_nfw`): `x^gamma (1 + x^alpha)^(-(beta+gamma)/alpha)`, commented
  "correction to battaglia 2016 tau". Note the exponent convention: Lee22's eq. 10 uses `-beta'` directly, so a
  Lee22 `beta'` corresponds to an XGPaint `beta = alpha beta' - gamma` (`= beta' + 0.3` for `alpha = 1`,
  `gamma = -0.3`). This is why Lee22 cannot be dropped into XGPaint by changing `PowerLawParam` values alone:
  `beta' + 0.3` is not a power law in mass and redshift. It can be dropped in by overriding `get_params` instead:
  `Lee2022XGPaintDMProfile` (Section 7) does exactly that, and XGPaint's `compute_DM` then reproduces the Lee22 code to 2e-6.
- `src/profiles_y.jl:100` (`_nfw_profile_los_quadrature`): the line-of-sight integral runs over the coordinate
  along the ray from 0 to `zmax = 1e5` in units of R200c (a length, not a redshift), i.e. the projected column
  contains all gas out to effectively infinite radius.
- `src/profiles.jl:278` (`compute_θmax`): painting is cut on the sky at `mult * R200c` with `mult = 4`.
- `src/profiles_frb.jl:53-55` (`compute_DM`): `DM = N_e [m^-2] * M2_TO_PC_CM3 / (1 + z)`.

## 7. How this repository implements Lee22 and the readings above

All in `frb_map_generation/lee2022_frb_dm_profile.jl` unless stated (line numbers at commit 4f50f28 or later):

- `:396-435` (`lee2022_projected_electron_column_pc_cm3`) and `:614-620` (`halo_electron_density_m3` for the
  Lee22 family): `n200` is computed *as printed*,
  `200 rho_crit(z) / (hydrogen_mass_fraction * m_p) * (omega_b / omega_m)`, and multiplied by
  `lee2022_normalization_factor(model)`. `hydrogen_mass_fraction` defaults to 0.76 (`:263`).
- `:56` and `:184-197` (`LEE2022_NORMALIZATIONS`, `lee2022_normalization_factor`): the four readings as options,
  `:literal` (A, factor 1), `:hydrogen_count` (B, `X_H^2`), `:electron_count` (C, `X_H(1+X_H)/2`),
  `:baryon_fraction` (D, `Omega_b/Omega_m`), `:xgpaint_ne2d` (E, `P0 = 200 n0` in the
  XGPaint wrapper and XGPaint's `ne2d` does the electron conversion; factor `0.9 X_H m_p / m_per_e = 0.602`). The option name enters the cache signature, the model family and the
  provenance (`lee2022_normalization`, `lee2022_normalization_factor` in every product's `*_provenance.txt`).
- `:338-349` (`lee2022_dimensionless_density`): eq. 10 with the `-beta'` exponent, i.e. the paper's convention,
  not XGPaint's.
- `:588-612` (`Battaglia16DensityDMProfile`): XGPaint's own Battaglia16 3-D density, including `f_b rho_crit(z)`,
  the `ne2d` electron count and the 0.9, reproduced so that both families share one line-of-sight code path;
  the self-test (`:726`) checks it against XGPaint's `HaloDMProfile` to 1e-5.
- `:629-640` (`chord_dm_pc_cm3`) and `:658-700` (`SphericalChordDMProfile`): the line of sight limited to the chord
  inside the R200c sphere, used for the like-for-like comparison of Section 4.
- `generate_halfdome_z1_dm_mass_windows.jl:141` parses `--lee2022-normalization=literal|baryon_fraction|
  electron_count|hydrogen_count`; `:730` computes the aperture from M200c; `:1126` applies the angular selection
  and `:1132` the chord factor in spherical mode.
- `Lee2022XGPaintDMProfile` (same file, section "Lee22 inside XGPaint's own pipeline"): an `XGPaint.AbstractFRBProfile` whose
  `XGPaint.get_params` returns `beta = alpha beta' - gamma` and `P0 = N n0 200 m_per_e / (0.9 X_H m_p)`, so XGPaint's `rho_2d`,
  `ne2d` and `compute_DM` run unchanged on the Lee22 fits (`--dm-profile=lee2022_xgpaint`; self-test `run_lee2022_xgpaint_self_test`).
  The spherical products built this way are bin-for-bin identical to the Lee22-code products.
- Runner `run_spherical_z1_nside4096_1r200c_variants_local.sh`: variants `lee22_{noconc,pref}` (D),
  `lee22_{noconc,pref}_literalnorm` (A), `lee22_{noconc,pref}_efix` (C), `lee22_{noconc,pref}_xh2` (B). Products in
  `frb_map_generation/outputs/zsrc1p0_nside4096_nrays120000_allhalos_sphere1p0_m200c_<variant>_seed42/`.
- The other conventions used in all runs (`M_cut` pivot for the no-concentration `n0`, TNG-mean concentration,
  shape parameters frozen above `10^14.8 h^-1 Msun`) are described in `LEE22_IMPLEMENTATION_CHECK_20260916.md`;
  they are independent of the normalization question.

## 8. What to ask the authors

1. In the code that produced Tables 3 and A2, was `n200` computed as `200 rho_cr f_b / (X_H m_p)` (as printed in
   eq. 9), as `200 rho_cr f_b X_H / m_p`, or with the ionized-gas electron count? The three differ by factors of
   1, 0.578 and 0.669 in every predicted electron density.
2. Were the cell densities of eq. 7 and the `rho_cr` of eq. 9 both physical at each snapshot's redshift, or were
   comoving densities and/or the `z = 0` critical density used? This decides whether the tabulated
   `(1+z)^-2.1` evolution of `n0` is physical.
3. Were the profiles of eq. 8 averaged over all gas cells inside each radial shell (satellites and star-forming
   gas included), as the text implies?

## References

- Lee, B. K. K., Coulton, W. R., Thiele, L., Ho, S., 2022, "An exploration of the properties of cluster profiles
  for the thermal and kinetic Sunyaev-Zel'dovich effects", arXiv:2205.01710 (v1, 3 May 2022; published in
  MNRAS). Equation numbers in this note follow v1, Section 2.3.
- Battaglia, N., Bond, J. R., Pfrommer, C., Sievers, J. L., 2012, ApJ 758, 75 (pressure profile; definition of
  the self-similar `P_Delta`).
- Battaglia, N., 2016, JCAP 08, 058, "The tau of galaxy clusters" (electron-density gNFW fit used by XGPaint).
- Nelson, D. et al., 2019, Comput. Astrophys. Cosmol. 6, 2, "The IllustrisTNG simulations: public data release";
  field definitions at https://www.tng-project.org/data/docs/specifications/ (`ElectronAbundance` = n_e/n_H).
- XGPaint.jl, WebSky-CITA, https://github.com/WebSky-CITA/XGPaint.jl (local dev checkout v0.4.0, commit
  5dd0b57 (2026-07-13); files `src/profiles_tau.jl`, `src/profiles_y.jl`, `src/profiles_frb.jl`, `src/profiles.jl`).
- This repository (branch `cluster`): `frb_map_generation/lee2022_frb_dm_profile.jl`,
  `frb_map_generation/generate_halfdome_z1_dm_mass_windows.jl`,
  `frb_map_generation/run_spherical_z1_nside4096_1r200c_variants_local.sh`,
  `frb_map_generation/LEE22_IMPLEMENTATION_CHECK_20260916.md`,
  `frb_map_generation/LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md`.
