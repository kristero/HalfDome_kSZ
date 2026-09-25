# Isabel Medlock's Baryon-Pasting spectra on the Takahashi+25 figures

Date: 2026-09-25. Adds Isabel Medlock's BP halo-model tSZ x FRB-DM spectra to the real-data Takahashi+25
w_yDM(theta) figures (`fullsky_takahashi_fig13`, `fullsky_takahashi_fig13_residuals`), next to Battaglia12 x
Battaglia16, Lee22 x Lee22 and Lee22 x Lee22 (calibrated range), plus a per-parameter figure.

## 1. Input

`Isabel_Medlock_data/Dell_yDM_param_variations.csv` (not committed; supplied by I. Medlock, redistribution terms not
stated): D_ell^{y x DM} = ell(ell+1) C_ell / 2 pi [pc cm^-3] on 1000 log-spaced ell from 10 to 10^4, for the BP
fiducial gas parameters and one-at-a-time variations `<param>__<mult>` of epsilon (x0.25-4), fstar, Sstar, A_nt,
B_nt, gamma_nt (x0.5-2); the six `__1` columns are the identical fiducial, leaving 25 distinct curves. The meta file
gives the fiducial parameters, cosmology (H0 = 67.2, Om = 0.31, Ob = 0.049, sigma8 = 0.81), a halo mass grid
(1e13 - 10^15.5 Msun) and a redshift grid 0.01 - 2. It states neither the FRB source-redshift kernel nor a beam.

## 2. Treatment (`medlock_bp_spectra.py`, `fullsky_tsz_dm_comparison.py`)

- C_ell on integer ell: log-log interpolation inside ell = 10 - 10^4, power-law continuation of D_ell (fit on
  ell = 10 - 20) below ell = 10, zero above 10^4; monopole dropped by the Legendre sums.
- The survey Gaussian beam is applied to y (Planck 10', ACT 1.6'), as for the HalfDome curves, since the Takahashi
  measurement uses beam-smoothed y maps and the BP D_ell is a theory spectrum.
- w(theta) and exact annulus means with the same `angular_correlation` / `annular_correlation` as the HalfDome pairs.
- Main figures: the fiducial (reddish purple #9E4A8A, dash-dot, hexagons) and the min-max envelope over all 25 curves
  (shaded; the range of one-at-a-time variations, not an uncertainty band). Figure text states the kernel difference
  (HalfDome: observed FRB redshifts; Medlock: all FRBs at z = 2, assumed); panel titles give the data sample (71 / 31 FRBs). New figure `medlock_bp_parameter_variations_takahashi.{png,pdf,svg}`: one column per parameter, rows
  Planck / ACT, light to dark = low to high multiplier, fiducial dash-dot.
- Residual panels: annotation rows for values beyond +-100 % are stacked only among the curves actually clipped at that
  angle (`compact_text_slots`); where several curves are clipped at one angle the edge triangle is drawn neutral grey
  (`shared_edges`) and the coloured numbers carry the identity; in-range markers are no longer cut by the spines.

## 3. What the fiducial is and is not

The BP fiducial is not any of Medlock & Nagai's Fig. 5 maximum-likelihood curves (digitized in
`outputs/takahashi_100k_20260914/references/medlock_fig5_best_fit_digitized.csv`): unbeamed it is 2.6-8x the ACT fit
at 2-15' and 1.9-3.8x the Planck MILCA fit at 12-39' (the digitized traces are unresolved beyond that), and matches the
NILC fit only near 12' (0.9x); the ACT fit is also steeper (10x drop from 2' to 10' against 5x). The three published fits
differ from each other by 8x at 12', so no single parameter set is all three. It is therefore
labelled "fiducial", not "best fit". The source kernel is assumed to be Medlock & Nagai's all-sources-at-z = 2 (the
meta's redshift grid ends at z = 2); the HalfDome curves use the observed 71 / 31 Takahashi source redshifts, so the
two families are not kernel-matched. Scaling by the HalfDome Battaglia16 curves, a z = 2 kernel raises the prediction
by ~15-65 % relative to the observed-redshift kernels, so the Medlock rows below are biased high by roughly that
much (the halo mass range, halo boundary and cosmology also differ). No beam is stated in the meta; the file is
unbeamed by its shape (D_ell peaks at ell ~ 5000 with no beam roll-off), so applying the survey beam does not double it.

## 4. Generalized chi^2 against the real data (full jackknife covariance, bins above the paper's cut)

`analysis/takahashi25_chi2_halfdome_and_medlock.csv`; per-annulus values in `analysis/medlock_bp_takahashi_annuli.csv`
(fiducial rows/columns named `fiducial`, variations `<param>x<mult>`). The jackknife covariance is inverted directly: no
Hartlap-type debiasing (the number of jackknife regions is not in the delivered files), so model-to-model differences
are meaningful but absolute PTEs are not quoted.

| model | Planck MILCA (8 bins) | ACT (11 bins) |
|---|---:|---:|
| w = 0 (for context) | 21.85 | 9.71 |
| Battaglia12 y x Battaglia16 DM | 19.77 | 9.00 |
| Lee22 y x Lee22 DM | 57.71 | 119.42 |
| Lee22 y x Lee22 DM, calibrated range | 23.50 | 30.04 |
| Medlock BP fiducial | 20.85 | 13.62 |
| Medlock BP, range over the 25 curves | 20.62 - 21.38 | 8.43 (A_nt x0.5) - 19.28 (fstar x0.5) |

In Planck the BP variations are nearly indistinguishable above the 10' cut and every model, BP or HalfDome, falls below
the data beyond theta ~ 60' (halo-only predictions carry almost no signal there), which dominates chi^2. In ACT the lowest chi^2
are A_nt x0.5 and fstar x2 (8.4-8.9, comparable to Battaglia12 x Battaglia16 at 9.0, the fiducial being 13.6): both
simply lower the amplitude, which is also what a kernel matched to the observed z ~ 0.3 sources would do, so this is
not a constraint on the BP parameters. The Planck fiducial is barely better than w = 0 (20.85 vs 21.85).

## 5. Verification

Independent three-lens check (workflow `verify-medlock-overlay`, 2026-09-25): the annulus means of all 25 curves were
re-derived from the raw CSV with separate code (own Legendre recurrence, scipy `eval_legendre`, brute-force quadrature
over theta; mutually consistent to 2e-10) and match the repository's CSV to <= 1.6e-10; the chi^2 values reproduce to
3e-10. Construction choices are negligible: zeroing ell < 10 moves any annulus by < 0.05 sigma (Planck chi^2 +0.5),
lmax 8192 vs 10^4 by < 0.03 sigma (ACT), interpolation scheme by < 1e-4 sigma. HalfDome curves, Takahashi points and
`analysis/fullsky_takahashi_annuli.csv` are bit-identical to before this change.

## 6. Files

- `frb_map_generation/medlock_bp_spectra.py` (new; `python medlock_bp_spectra.py` runs its self-test)
- `frb_map_generation/fullsky_tsz_dm_comparison.py` (`medlock_prediction`, `draw_medlock`, `chi_square_table`,
  `plot_medlock_variations`, `compact_text_slots`; `draw_percent` accepts per-point `text_slot`)
- Outputs (not git-tracked): `outputs/tsz_dm_fullsky_20260918/plots/{fullsky_takahashi_fig13,
  fullsky_takahashi_fig13_residuals, medlock_bp_parameter_variations_takahashi}.*`, the two analysis CSVs above.
