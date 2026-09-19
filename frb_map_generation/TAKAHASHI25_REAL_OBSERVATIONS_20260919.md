# Replacing the digitized Takahashi+25 comparison with the real measurement + covariance

Date: 2026-09-19. The user obtained the actual Takahashi et al. 2025 binned w_yDM(theta)
measurements and full jackknife covariance directly from the authors (previously flagged as
"available on request" in `prepare_takahashi_observational_inputs.py`'s inventory and worked
around with a plot-digitized approximation, `digitize_frb_observational_figures.py`).

## 1. Input files and contract

`Takahashi25_data/xi_folder/xi_folder/` (not committed — obtained on request from the authors;
see Section 4), for each of Planck MILCA (71 FRBs), Planck NILC (71 FRBs), ACT (31 FRBs):

- `xi_angle_ne2001_varalp_{survey}.txt`: col1 = theta-bin mean [arcmin], col2 = nominal log-bin
  center, col3 = bin index 0-12, col4 = first term of Eq. (27) [pc/cm^3], (a).
- `xi_angle_rand3e+3_ne2001_varalp_{survey}.txt`: col1 = bin index, col2 = nominal center,
  col3 = second term of Eq. (27) [pc/cm^3], (b).
- `xi_angle_covjk_ne2001_varalp_{survey}.txt`: 13 header-block lines (index, center, reference
  value, jackknife sigma) then 91 lines for (i, j), i >= j: covariance [pc^2/cm^6] and correlation.
- Measurement (Fig. 9) = (a) - (b).

## 2. Processing (`prepare_takahashi25_real_observations.py`)

Parses and validates all three files per survey, checks internally that
`sqrt(diag(covariance)) == sigma` and `covariance / outer(sigma, sigma) == correlation` (both to
1e-6), and checks that bin indices 1-12 exactly match this repository's own fixed angular grid,
`compare_takahashi_sightlines.EDGES = np.logspace(0, 3, 13)` (12 bins, 1-1000 arcmin, 0.25 dex
each) — the author's bin 0 (theta ~ 0.75', below 1 arcmin) is an extra bin outside that range.
Output: `frb_map_generation/outputs/takahashi25_real_observations/{planck_milca,planck_nilc,act}.npz`
with all 13 bins, the full 13x13 covariance, and n_frb.

Detection is modest and physically reasonable: Planck MILCA peaks around 3.4 sigma at theta ~ 80',
mostly 1-2 sigma elsewhere; ACT (31 FRBs, noisier) mostly 1-2 sigma. Planck NILC is systematically
a bit weaker than MILCA and goes slightly negative beyond ~250' (consistent with being a noisier
component-separation pipeline at this signal level, not a sign of a processing error — the MILCA/
NILC agreement at small/intermediate theta, where the signal is detected, is good).

## 3. Consumers updated

- `fullsky_tsz_dm_comparison.py`, `observations(plane)`: now loads the real npz (all 13 bins;
  `annular_correlation` accepts arbitrary bin edges). Default Planck reading is MILCA
  (`TAKAHASHI25_SURVEY_KEY`), matching the earlier digitized default. Regenerated:
  `fullsky_takahashi_fig13.{png,pdf,svg}`, `fullsky_takahashi_fig13_residuals.{png,pdf,svg}`,
  `analysis/fullsky_takahashi_annuli.csv`, `analysis/fullsky_summary.json` (`plot` stage).
- `cross_finite_source_realizations.py`, `observed()` / `observed_bins()` / `chi_square()`: now
  loads the real npz restricted to bin indices 1-12 (matching `EDGES` exactly, verified in the
  loader). `chi_square` was upgraded from a diagonal, symmetrized-digitized-error approximation
  ("this ranks, it does not test") to a proper generalized chi^2 using the real jackknife
  covariance's inverse (bins above `PAPER_CUT` only), computed once per survey in `observed_bins`.
  Regenerated via `analyze-pairs`, `plot-selected`, `plot-simple` (new best-fit realizations
  selected under the real covariance; chi2_100k = 24.96/8 annuli (Planck), 36.34/11 annuli (ACT)
  — reduced chi2 ~ 3.1-3.3, i.e. the halo-only Battaglia/Lee22 predictions are in tension with the
  real measurement at roughly the same level the residual panels already show, not a new finding).
- `compare_halfdome_takahashi.py`, `make_publication_comparisons.py`, `plot_takahashi_sightlines.py`,
  `compare_updated_sightlines.py` still read the old digitized CSV; see the companion audit for
  whether each is a live deliverable that also needs updating, or superseded by the fullsky method.

## 4. Data provenance caveat

`Takahashi25_data/` is **not committed**: it was obtained on request from the paper's authors and
its redistribution terms were not stated. It is left untracked (confirmed not covered by
`.gitignore`, so `git add -A` would pick it up) pending the user's decision on whether/how to share
it (e.g. a private lustre path rather than the git history).

## 5. Files

- `prepare_takahashi25_real_observations.py` (new)
- `fullsky_tsz_dm_comparison.py`, `cross_finite_source_realizations.py` (edited)
- `frb_map_generation/outputs/takahashi25_real_observations/*.npz` (not git-tracked, regenerable)
