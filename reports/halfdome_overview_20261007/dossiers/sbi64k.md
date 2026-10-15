# Dossier: section 3, 64k SBI final results

Fragment: `sections/s3_sbi64k.html` (id `sbi64k`). Figures: `fig/3_sbi64k/` (16 PNG + 1 PDF). Helper scripts:
`scripts/sbi64k/` (all read stored results only; nothing retrained).

## Sources read (all under SBI_analysis/tsz64k_sbi_local unless noted; the whole folder is untracked in git)

- Explainer page v3: `results_doc/index.html` (built by `doc_build.py`, figures by `doc_figures.py`, per-row data
  `doc_metric_data/` written on idark by `doc_metric_data.py`). Artifact G6HNsrPWDCbTwvNASkM6Ar.
- `results_final/` (median-scale asinh, production noise): convergence_table.md, convergence.json, checks.json,
  validation.json, prepare.json, noise_model.json, fig_convergence/fig_correlation_vs_N/fig_checks/fig_validation.
- `results_final_noise/` (noise-level asinh): convergence_table.md, convergence.json (fit_at_bound True for all 5).
- `beamfix/fetched/results/series.json` (staged beam-consistent 40-band series, `series_noise`).
- `results_merlin_unbinned_*` metrics_{merlin,pca,moped}.json (production / beam / nnwhite / w420; _lin = unbounded).
- `results_publication/` publication_numbers.json, publication_numbers_fisher.json, whitening_comparison.json, figs.
- `results_fisher_b12_beam/fisher_forecast_b12.json` (calibrated all-multipole Fisher, beam noise).
- `results_logit/config_comparison.md`, `beta_bias.json`; `results_finished_rows/diagnostics/diagnostics.json`,
  `joint_hpd.json` (box-space NPE, 24,647 training rows, 768 held out).
- `results_524k_moped/` (historical narrow-prior 524k set, fixed noise maps, 495 test rows).
- Bundle `tSZ_64k_flat_prior_SO_baseline_deproj0/` (tag tsz64k-flat-prior-v1, b46ab65): README.md, engine/engine.jl,
  engine/benchmark.jl, engine/spherical_truncation_profiles.jl, design/design_tests.py, julia_env/Manifest.toml.
- XGPaint dev checkout `/home/kn18001/.julia/dev/XGPaint` (branch cluster, HEAD 1660e08; tag
  validated-tsz256-20260922 = commit 35f79ad, tree 3b57cac = tree of ce1d239; HEAD differs from tag only in
  test/runtests.jl).
- Memory notes: tsz64k-local-sbi-and-globus, tsz-sbi-pipeline-explainer-doc, tsz64k-beamfix-staged-run,
  fisher-b12-forecast-idark, tsz64k-bundle-and-collaborator-handoff.

## Verified facts (task list)

| claim | verdict | where |
|---|---|---|
| 64,440 rows (+1,096 idark validation) | yes | results_final/prepare.json (missing_rows 1096), validation.json rows 1096 |
| nine B12 params, flat priors | yes | bundle README table, design_tests.py L9-L10 |
| NSIDE 8192 painting, 2' beam | yes | engine.jl L10-L18 and asserts in run_maps (nside 8192, fwhm 2.) |
| f_sky 0.4 cap mask seed 12345 | yes | engine.jl L23 |
| two SO noise splits, cross spectrum | yes | engine.jl L32-L45 |
| ell 80-7979 | yes | README (collect), train_ensemble.py ELL |
| compressions 40 bins / MOPED / PCA / Merlin | yes | sbi_final.py, merlin_compression.py VARIANTS |
| logit vs linear NPE | yes | train_ensemble.to_u; MERLIN_PARAM_SPACE=linear |
| log p ~1.27 nats per doubling, no flattening | yes for results_final 40 bands (checks.json gain_per_doubling_loglinear 1.267); saturating fit L_inf ended at its upper bound (lp+20) in results_final (15.39 = -4.61+20) and in all results_final_noise variants (fit_at_bound True) | checks.json, convergence.json |
| TARP 71-73 / 96-97 | yes (results_final table, all N, all variants) | convergence_table.md |
| mean r 0.43 -> 0.55 | yes (0.432 -> 0.552, 40 bands production, median scale) | convergence_table.md |
| beam 40-band 61,440: log p -2.354, r 0.590, width 0.777, cov 0.721 | yes | doc_metric_data/beam_bins40_N61440_metrics.json; series.json |
| Merlin - 40 bins r +0.139, log p -0.61 | yes (recomputed: +0.139 ± 0.004, -0.61 ± 0.16; IS -0.130 ± 0.003) | scripts/sbi64k/final_calibration_numbers.json |
| boxed-Fisher comparison | yes: checks.json (40-band linear Fisher, expected log p +7.98 baseline / +8.71 goal vs NPE B12 -6.29); calibrated all-multipole Fisher corner (publication_numbers_fisher.json beam_w420) | |

## Final-version choice

Final = beam-consistent noise, 61,440 training rows, bounded (logit) 3-seed ensembles, 3,000 fixed test rows;
Merlin and PCA with noise-only whitening (fiducial-free); 40 bins = staged beam series final stage. This is the set of
explainer v3. Merlin w420 (-2.82) has the best log p of the Merlin variants but its whitening is built from 420 B12
noise draws (fiducial-dependent); it is used only for the Fisher corner (Fig 3.11).

## Numbers recomputed for the report (scripts/sbi64k)

- final_calibration_numbers.json: coverage68/95, interval score, SBC max |ECDF-U| (Merlin 3/9 params above the
  0.0248 band, 40 bins 9/9, PCA 9/9), TARP max deviation (40 bins +0.052 at 0.64; PCA +0.024; Merlin -0.011),
  paired differences (identical to results_doc section 12). Merlin has higher per-row log p than 40 bins on 45.3 % of
  test rows.
- final_calibration_numbers.json also holds: alpha_z_P0 bias/coverage by truth bin (Merlin [-1,0): -0.46 / 0.62;
  [0,0.5): -0.68 / 0.45), wide_rows (93 % of Merlin rows with mean width > 0.85 flat are fainter than 0.1x B12, 16 %
  of all rows are), coverage_by_brightness (Merlin 68 % coverage 0.69-0.71 below 1x B12, 0.64 above 1000x; P0 and
  alpha_m_P0 0.60-0.71 in most bins).
- b12_posterior_correlations.json: B12 posterior parameter-parameter correlations (Merlin/PCA 30,000 draws, 40 bins
  300,000) and the Fisher correlation matrix.
- logp_gain_per_doubling.json: local and log-linear gains; 524k set gains fall to 1.54, 1.23, 1.11 nats per doubling
  above 65k.
- 13_convergence_40band_noise_variants.png: results_final_noise/fig_convergence.png re-rendered locally from copies
  of its metrics.json (idark copy had missing minus glyphs); code unchanged (sbi_final.cmd_report).

## XGPaint

Pinned tag validated-tsz256-20260922 (Manifest L1976-L1982: git-tree-sha1 3b57cac). Edits the training set relies on:
f16a7f3 (2026-02-09, parametrised Battaglia16ThermalSZProfile keywords), e67a8a1 (2026-04-30, beta convention
beta = alpha*beta_raw - gamma replacing -gamma - alpha*beta_raw, and prepare/evaluate_profile_slice hooks),
35f79ad (restores N_logθ; tree identical to ce1d239, no numerical change). Not XGPaint: 4 R200c chord truncation
(bundle engine/spherical_truncation_profiles.jl), shared-geometry painter, mask/noise/beam (engine.jl), beam-consistent
noise correction (beamfix/), all SBI code.
Numerical effect of e67a8a1 not quantified in any file read; algebra: with generalized_nfw exponent -(beta+gamma)/alpha
(since 9f37b1e), the old conversion gives outer factor (1+x/xc)^(+beta_raw), a rising profile; so it is a correctness
fix, not a percent correction. 64k rows were all made after it.

## Open issues to state

- Simulation-limited: no flattening of log p up to 61,440 on the 64k prior.
- One fixed HalfDome sky (no sample variance); multipole fine structure is a fixed-sky fingerprint (Fisher info).
- Merlin overconfident on bright skies (>1000x B12, cov 0.64 per doc) and near prior on faint skies (asinh centring).
- Training recipes differ between 40 bins (sbi trainer, 340-436 epochs) and Merlin/PCA (75-110 epochs).
- Validation rows only evaluated with results_final (median scale) networks.
- alpha_z_P0 prior edge pull; FLAMINGO pixel window (other sections).

## Cross-references to other sections (read their ledes, 2026-10-07)

- s4_seedmask.html (section 4): other lightcones / masks move the Merlin posterior of B12 (P0 ~12 -> ~50, truth 18.1);
  band50 mitigation, 18-25 % wider at B12, ~46 % wider over the prior. Section 3 cites this as a caveat only.
- s5_merlin.html (section 5): Merlin pipeline, whitening (20 draws -> 420-draw lag model), band50.
- Section 3 therefore labels its metrics "on the training sky".

## Build

`python3 scripts/sbi64k/build_section.py` fills the 12 code excerpts verbatim from the current files (template
scripts/sbi64k/s3_template.html) and checks tags, images and the single <section>.
