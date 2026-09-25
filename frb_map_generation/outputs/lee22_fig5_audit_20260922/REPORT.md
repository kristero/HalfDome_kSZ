# HalfDome versus Lee22 Figure 5

Date: 2026-09-22. This audit evaluates the existing profile locally; no cluster
job, new lightcone painting, or change to the previous PDF products was made.

## Finding

There is a real, approximately 40% normalization discrepancy between the
**current PDF-matching HalfDome Lee22 model** and the **electron-density fit
defined by Lee22**. It is introduced by our `xgpaint_ne2d` choice, not by a
numerical integration error in XGPaint. The shape and parameter transcription
pass the independent checks below. This comparison provides no evidence that
Lee22's equation 9 is a typo.

The existing model remains useful as the specific empirical variant that
matched Ralf's DM PDFs. That agreement does not make it an unchanged evaluation
of Lee22's fitted electron density. These two scientific claims must be kept
separate.

## Matching the figure

Reference: [Lee et al. arXiv:2205.01710v1](https://arxiv.org/abs/2205.01710),
equations 9, 10 and 12, Figure 5, and Table 8 (the no-concentration fit called
Table A2 in the repository). Original vector figure and TeX were obtained from
the [arXiv source](https://arxiv.org/src/2205.01710v1). The source asset is named
`figures/fig_4.pdf` but is Figure 5 in the compiled paper. Direct journal access
was blocked, so these numbers refer specifically to the arXiv figure, not an
independently retrieved final journal figure.

The apples-to-apples **model comparison** is:

- Three-dimensional `x^3 n_e/n200`, with `x=r/R200c`. Neither projected column
  density nor observer-frame DM is the figure's ordinate.
- Snapshot redshift **z=0**, not a foreground lightcone ending at source z=1.
- Mass bins **13.2-13.4** and **13.8-14.0** in `log10(M200c/[h^-1 Msun])`.
  With HalfDome h=0.68 these are `[2.331,3.694]e13` and
  `[9.279e13,1.471e14]` physical Msun. The second figure bin extends above the
  previous PDF selection's physical `1e14 Msun` upper limit.
- The orange figure curve: **without concentration, with the broken mass
  power law in xc and beta**. The green curve includes concentration and the
  brown curve omits both concentration and the mass break; neither is the
  corresponding model for the existing no-c implementation.
- HalfDome predictions are evaluated at geometric bin centres. The shaded
  envelope samples the entire mass bin; it is not a confidence interval.
- All densities use the same **paper-defined n200** on the vertical axis.
  Dividing the current density by its own alternative reference normalization
  would conceal the physical amplitude change.

The original simulation curve represents the lowest Klypin-concentration
quartile of Rockstar halos. The current HalfDome lightcone contains no
Klypin-concentration field; it is also not a z=0 snapshot. No fabricated
concentrations or pseudo z=0 halo stack is used. Consequently this is an exact
definition-matched test of the painting prescription, with approximate
bin-centre predictions, **not a recreation of the original hydro stack**.
Its exact halo weights, unrounded fit parameters and covariance are unavailable.
This limitation does not hide the normalization offset: the mass-bin envelope
of the current no-c model remains well below the corresponding figure curve.

## Quantitative result

`halfdome_vs_lee22_figure5.pdf` is the comparison; `comparison.csv` contains
unclipped values. Errors below are unweighted mean absolute fractional
differences against 100 vector-curve vertices per panel, approximately uniform
in log radius over 0.0425-1.34 R200c. They are not statistical error bars.

| HalfDome prescription / reference curve | low-mass bin | high-mass bin |
|---|---:|---:|
| Paper normalization / Lee22 no-c+BPL curve | 3.21% | 2.02% |
| Current PDF normalization / Lee22 no-c+BPL curve | 37.91% low | 38.62% low |
| Paper normalization / figure's low-c simulation curve | 7.46% | 2.37% |
| Current PDF normalization / figure's low-c simulation curve | 37.79% low | 38.87% low |

The first row is the primary model-transcription comparison. The remaining
few percent cannot be assigned uniquely to rounded coefficients versus exact
mass-bin averaging without the original inputs. A shape-only diagnostic
allowing a small effective mass shift and a free amplitude reduces the maximum
residual to 0.13% and 0.062%; these fitted adjustments are **not** used in the
plot or quoted primary results. They show that the large discrepancy is not
a different radial shape. The 101-point mass-bin envelope for the current
normalization spans residuals from -48.5% to -25.5% in the low-mass panel and
-45.5% to -31.0% in the high-mass panel.

The dotted vertical line is 1R200c. The dotted HalfDome profile continuations
outside that radius only expose the full Lee22 fitted radial range up to
1.34R200c for comparison; **the production halo-DM calculation still includes
only gas inside the 1R200c sphere**. Truncating a spherical profile at R200c
does not change its local density at r<R200c. It does change a projected column,
which is a different observable.

## Origin of the factor

Write the dimensionless shape as

```
g(x) = (x/xc)^(-0.3) [1+x/xc]^(-beta_prime)
n200_paper = 200 f_b rho_crit(z)/(X_H m_p)
n_e_paper = n0 g(x) n200_paper
```

Here `f_b=Omega_b/Omega_m`, `X_H=0.76`; all no-c parameters use the common
`Mcut=10^13.61 h^-1 Msun` pivot. These are already an **electron-density fit**.

XGPaint instead accepts a **gas-density** amplitude, then computes

```
rho_gas = P0 g(x) f_b rho_crit(z)
n_e_XGP = 0.9 rho_gas/m_per_e
m_per_e = m_e + 2 m_p/(1+X_H)
```

Our PDF-matching option sets `P0=200 n0`. It therefore gives

```
n_e_current / n_e_paper = 0.9 X_H m_p/m_per_e
                      = 0.6016316602183194
```

This reduces density by **39.8368% at every radius, mass and redshift**.
It preserves the shape while imposing a new physical amplitude. To feed
Lee22's electron fit through the same gas-density interface faithfully, use

```
P0 = 200 n0 m_per_e/(0.9 X_H m_p)
   = 1.6621465692764925 * (200 n0)
```

The existing wrapper already performs that conversion with
`normalization=:literal, n0_pivot=:mcut`; this audit uses that option without
editing the profile. The exponent conversion
`beta_XGP = alpha*beta_prime - gamma` is also already correct. There is no
missing additional baryon fraction in equation 9.

**Correction to the earlier normalization note:** n200 is a defined reference
scale, not a claim that gas of density `200 f_b rho_crit` contains that number
of electrons. Equations 7 and 9 have different roles and are not contradictory
merely because their hydrogen factors differ. If a reference density is
changed by a factor C, its fitted dimensionless amplitude must change by 1/C
to preserve the physical profile. Holding the published n0 fixed while
replacing n200 changes the model. The earlier assertion that equation 9
"cannot be" correct is not established and is withdrawn by this audit.

## Why kSZ and the DM PDFs can still agree

The saved XGPaint/CLASS-SZ kSZ comparison constructs **BattagliaTauProfile**,
with Battaglia16 gas density and the two codes' free-electron conversion.
`tau_ksz_comparison/paint_maps.jl` and its README establish this directly.
That checks the adopted Battaglia prescription and projection pipeline; it
does not test whether a separate Lee22 n0 has been mapped to gas density with
the correct normalization.

Physically, `tau = sigma_T integral n_e dl` and
`DM_observed = integral n_e dl/(1+z)`. Both use the same electron density, so
a constant normalization change multiplies their halo amplitudes by the same
factor. The conversion routines can be correct while an input profile's
amplitude is wrong relative to its defining paper.

The HalfDome and TNG300-Dark painted PDFs share this same imposed profile.
Their mutual agreement is therefore not an independent test of its absolute
density normalization. Ralf's hydrodynamic PDF is an independent integrated
comparison, but a one-point, all-foreground distribution does not uniquely
determine a fixed-redshift, mass-selected radial density profile. The 0.602
variant was selected after comparisons with those PDFs; their agreement
supports that empirical variant, not the claim that it reproduces Figure 5.

For scale only, applying the paper normalization to the same rays would
multiply every modeled halo DM by **1.66215**. From the retained mass13to14
results, the positive-ray means would change from 136.94 to **227.61** for
HalfDome and from 137.83 to **229.10** for TNG-Dark, whereas Ralf stays at
138.65 pc cm^-3. These are exact linear rescaling predictions, **not new
painting runs**. Resolving that remaining model-versus-hydro tension requires
matched radial profiles/redshift slices and halo definitions. The available
broad-bin Ralf arrays cannot supply that test.

## Checks, resources, and retained files

- Actual HalfDome wrapper versus independent Table A2 formula: maximum relative
  difference **4.44e-15** over both cosmologies, three redshifts, both mass
  grids, and 181 radii.
- Wrapper versus direct local electron-density calculation: **3.77e-15**.
- Normalization ratio through 48 finite spherical chords: **4.44e-16**.
- Interpolation onto figure vertices contributes at most **0.0130%**.
- Log-coordinate inversion checked against major y ticks and an independent
  minor x tick; vector paths are used rather than raster digitization.
- Independent top-versus-middle-panel check: the recovered densities reproduce
  the figure's plotted fractional residuals within **1.11e-6** absolute fraction.
- Julia peak RSS was **749572 KiB = 0.715 GiB**. No cluster work was conducted.
- PDF rendered with Poppler and visually inspected. Prior profile and generator
  hashes checked; neither production source was modified.

New code: `frb_map_generation/lee22_fig5_audit/export_profiles.jl`,
`compare_fig5.py`, and `README.md`. This output directory contains the report,
PDF/PNG/SVG, actual Julia profile grid, extracted figure curves, comparison CSV,
JSON metrics, validation record, original figure/TeX, and checksum manifest.
Historical notes `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md` and
`WHAT_CHANGED_FOR_TNG_AGREEMENT_20260917.md` receive a short correction notice;
their earlier numerical history is retained. No earlier PDF/data products,
production profile, or XGPaint package files are edited.
