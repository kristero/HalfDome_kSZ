# Why the M ~ 1e14 density comparison does not look like Lee+2022 Figure 5

**AMENDMENT (2026-09-19, later the same day): the "divide by f_b" conclusion below is retracted.** A
reviewer correctly pointed out that Fig. 5 does not show Battaglia (2016) above the simulation/best-fit curve
at *every* radius — it shows a single crossover: cyan above at small r ("the profile parameters overestimated
n_e/n200 in the inner region, especially at low masses", Lee+2022 Sec. 5) and a good match at large r/large
mass. Checked directly against the actual figure (page 5/11 of the arXiv PDF) at the paper's own z = 0 and
both of its exact mass bins (3.0e13 and 1.17e14 Msun, from 10^13.2-13.4 and 10^13.8-14.0 h^-1 Msun with
h = 0.6774): **neither the physical (f_b-included) Battaglia16 nor Battaglia16/f_b reproduces this crossover.**
Physical B16 has Lee/TNG higher at *every* radius in both bins (opposite of the low-r part of Fig. 5); B16/f_b
overshoots even further, since the Lee/B16 ratio never reaches 1/f_b ≈ 6.3 at either mass, so B16/f_b also
never crosses below Lee. A separate, smaller, and better-motivated factor was found instead: our comparison's
`lee2022_normalization="xgpaint_ne2d"` reading of Lee's n0 (chosen for TNG-matching, see
`LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`) uses XGPaint's own electron-per-mass convention
(0.9 / m_per_e for ionized H+He, `lee2022_frb_dm_profile.jl` lines 731-733, 749-758), not Lee+2022's own
literal eq. 9 electron definition (1/(X_H m_p)). The two differ by a factor of exactly 1.662 (X_H(1+X_H)/(2·0.9)
inverted). Applying that 1.662x to Battaglia16 (not f_b's 6.3x) reproduces a crossover close to the
low-mass-panel's shape (~0.18 R200c) but not cleanly at the high-mass panel (crossover pushed to/below the
plotted range's left edge) — so this explains part, not all, of the effect. **Open**: the residual mass-dependence
means this is not a single constant-factor fix; a genuine shape (mass/redshift-scaling) difference between our
implementation and whatever Battaglia/Lee actually used for their own Fig. 5 curve remains unresolved. The
production comparison plots and power spectra are unaffected either way (Section 4 below still holds for that
part). See the conversation record for the full numeric tables at z=0 across masses 7.3e12-1e15 Msun.

Date: 2026-09-19. Prompted by a reviewer comment: at M ~ 1e14 Msun our Battaglia16-vs-Lee22 electron-density
comparison shows Lee22 with a higher amplitude than Battaglia16 at essentially every radius, while in Lee et
al. 2022 (arXiv:2205.01710, hereafter Lee22) Figure 5 the "Battaglia (2016)" reference curve is *bigger* than
the TNG/best-fit curve, especially at large r. The question was whether this is the exponent "typo" discussed
the day before (XGPaint's `generalized_nfw`, commented "correction to battaglia 2016 tau").

**Short answer: no, that exponent mapping is exact and not the cause. The cause is the f_b (baryon-fraction)
convention on the Battaglia16 curve, already identified in `LEE22_IMPLEMENTATION_CHECK_20260916.md` (Section 2)
and `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md` (Section 6): our comparison plots use the physically
normalized Battaglia16 density (which includes f_b, as a real gas density must), while Lee+2022's own Fig. 5
"Battaglia (2016)" curve is the same formula *without* that factor. Dividing our Battaglia16 by f_b reproduces
Lee+2022's Fig. 5 shape, including the large-r behaviour the reviewer is missing in our plots.**

## 1. The exponent mapping is exact, not approximate

XGPaint's `generalized_nfw(x, xc, α, β, γ) = x̄^γ (1 + x̄^α)^(-(β+γ)/α)` is evaluated with
`β = α·β_raw − γ` (`get_params`, both `Battaglia16ThermalSZProfile` and the Lee22 wrappers), where `β_raw` is
the literal Battaglia/Lee gNFW exponent as printed in the papers, `x̄^γ(1+x̄^α)^(-β_raw)`. Substituting:

    β + γ = (α β_raw − γ) + γ = α β_raw
    -(β + γ)/α = -β_raw

so `generalized_nfw(x, xc, α, β, γ) ≡ x̄^γ (1+x̄^α)^(-β_raw)` **identically, at every radius**, not just in the
large-x asymptote. This is an exact algebraic identity (verified here independently of the existing numerical
self-tests, which already checked it to 1e-5/2e-6 in `lee2022_frb_dm_profile.jl`). The "correction" comment
refers only to the bookkeeping needed to hand `generalized_nfw` an exponent that reproduces the literal formula;
it changes nothing about the profile shape and is not a source of error here.

## 2. The f_b factor is the whole effect

XGPaint's Battaglia16 density (`profiles_tau.jl`, `rho_2d`) multiplies the dimensionless gNFW shape by
`f_b = Omega_b/Omega_m` before converting to physical density ("mistake in battaglia 2016: need f_b to convert
from m to gas" — the shape fit is calibrated against the *matter* critical density, not the gas density, so an
external f_b factor is required to turn it into a gas/electron density). `LEE22_IMPLEMENTATION_CHECK_20260916.md`
already found that Lee+2022's own plotted "Battaglia (2016)" curve in Fig. 5 numerically matches XGPaint's
Battaglia16 **divided by** f_b, not the physical (f_b-included) density — i.e. Lee+2022 plotted the raw,
literal Battaglia gNFW shape without applying this factor, about 1/f_b ≈ 6.3× higher than a true gas density.
Lee22's own amplitude `n0` needs no such factor: their eq. 9 `n200` already carries an explicit
`Omega_b/Omega_m`, so `n_e = n0 · n200` is a physical density by construction (mass/normalization details in
`LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`).

Comparing our *physical* Battaglia16 against Lee22 is therefore not the same comparison as Fig. 5: our
Battaglia16 curve is deflated by f_b relative to the one Lee+2022 plotted, so Lee22 ends up above it almost
everywhere instead of below-then-converging.

## 3. Direct numeric check (this repository's own profile grid)

Using `outputs/projected_profiles_20260917/projected_profiles.csv` (`compute_projected_profile_grid.jl`), in
Lee+2022's own Fig. 5 units `(n_e/n200)(r/R200c)^3`, at M = 1e14 Msun, z = 0.1 (closest grid point to the
paper's z = 0):

| x = r/R200c | Lee22 no-c | Battaglia16, physical (ours) | Battaglia16 / f_b (Fig.5-style) |
|---:|---:|---:|---:|
| 0.04 | 0.0011 | 0.00074 | 0.0047 |
| 0.10 | 0.0132 | 0.0072 | 0.0454 |
| 0.40 | 0.220 | 0.068 | 0.429 |
| 1.00 | 0.536 | 0.140 | 0.885 |
| 1.34 | 0.523 | 0.154 | 0.975 |

`Battaglia16/f_b` sits **above** Lee22 at every radius and the two **converge** toward r = R200c (ratio
Lee22/(B16/f_b): 0.24 at x = 0.04 rising to 0.61 at x = 1) — the same qualitative shape as Fig. 5, and the
x = 0.04 and x = 1 values match the earlier note's digitized read-off of the actual figure (0.0055 and 0.90)
to about 5-15%. The physical Battaglia16 (our normal comparison convention) instead sits **below** Lee22 almost
everywhere, with the gap *growing* toward r = R200c (ratio Lee22/B16: 1.5 at x = 0.04 rising to 3.8 at x = 1)
— the opposite trend. Figure: `outputs/projected_profiles_20260917/fig5_reproduction_check.png`
(`plot_fig5_reproduction_check.py`), left/right panels at the two masses closest to Fig. 5's own bins.

## 4. What this does and does not mean

- Not a new bug, and not the exponent-mapping "typo": it is the same f_b/normalization question raised on
  16-17 September, now shown to also explain the qualitative *shape* mismatch against Fig. 5 at M ~ 1e14, not
  only the earlier-quantified amplitude/gas-fraction issue.
- Our production comparison plots and the cross-correlation/power-spectrum results are unaffected: they use the
  physically normalized (f_b-included) Battaglia16 throughout, which is the correct convention for an absolute
  DM/pressure prediction, and Lee22's own normalization (`xgpaint_ne2d` reading) was separately validated
  against TNG's absolute R200c-sphere DM (0-2% agreement; `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`,
  Section 4). Reproducing Fig. 5's specific curve was never a target of those comparisons.
- A residual, smaller, already-documented issue remains: even with f_b correctly applied, XGPaint's Battaglia16
  gives 0.67x TNG's own mean DM inside the R200c sphere in this same mass window (same Section 4 table), i.e.
  it undershoots real TNG halos by about a third — a genuine (if much smaller) shape/calibration question about
  the Battaglia16 fit itself, separate from the f_b bookkeeping addressed here.

## Files

- `frb_map_generation/plot_fig5_reproduction_check.py` (new)
- `frb_map_generation/outputs/projected_profiles_20260917/fig5_reproduction_check.{png,pdf,svg}` (not git-tracked)
- Companion notes: `LEE22_IMPLEMENTATION_CHECK_20260916.md`, `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md`
