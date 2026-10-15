# Dossier: Section 4, seed / mask / simulation impact (s4_seedmask.html, id="seedmask")

Written 2026-10-07. Notes for assembling and checking the section. All paths repo-relative to /home/cbllover/HalfDome.

## Sources read (in order)
- SBI_analysis/tsz_seed_mask_variation_20261002/ (README.md, final_results/RESULTS.txt, variation_summary.json,
  parameter_comparison.csv, validation.json, cluster_run/baseline_check.json, plan.json, paint_seed.jl, prepare_idark.py, analyze.py)
- SBI_analysis/tsz_robustness_methods_20261004/ (README.md, EXTENSIONS.md, final_results/INTERPRETATION.md,
  final_results/report/results.md, selection.json, joint_truth_linear.json, report.py, common.py, pipeline.py, joint_truth.py)
- SBI_analysis/tsz_seed_offset_diagnosis_20261005/ (README.md, REPORT.md, seed_offset_summary.csv, diagnostics.json,
  diagnose.py, extended_controls.py, plots/)
- SBI_analysis/tsz_joint_report_20261005/ (README.md, page_manifest.json, control_definition.json, pearson_pairs.csv,
  PEARSON_COMPLETE.json, baseline_pearson.py, figures/, output/pdf/HalfDome_Merlin_seed_mask_and_Pearson_report.pdf, 36 pages, text extracted)
- SBI_analysis/tsz_truth_mean_validation_20261006/ (README.md, PAGE_MANIFEST.json, figures/,
  output/pdf/HalfDome_Merlin_complete_seed_mask_truth_mean_report.pdf, 44 pages; pages 27-29 and 36-37 used here)
- tSZ_64k_flat_prior_SO_baseline_deproj0/ (README.md, design/design_manifest.json, reference/test256/base_command.json,
  engine/{engine.jl,benchmark.jl,balanced_painter.jl,spherical_truncation_profiles.jl})
- tSZ_visuals/run_halfdome_fullsky_so_noise.jl (random_apodized_cap_mask, identical to the bundle copy)
- XGPaint dev checkout /home/kn18001/.julia/dev/XGPaint: git log, tag validated-tsz256-20260922 -> 35f79ad,
  f16a7f3 (parametrized B12), e67a8a1 (beta convention + prepare_profile_slice), upstream/main 29ec48a.

## Seeds
- Simulation (lightcone) seeds: 100, 102, ..., 118 (Gaussian HalfDome lightcones, /lustre/work/ruizm/halos/lightcone_<s>.hdf5).
  Seed 120 (non-Gaussian) excluded. Selected halo counts 85,215,792 (108) .. 85,267,919 (102); seed 100 = 85,224,251 (diagnostics.json).
- Mask seeds: 12345 .. 12354; MersenneTwister(seed) draws only the cap centre; fsky 0.4 cap (radius acos(0.2)), 60 arcmin cosine apodization.
  Training mask = 12345 (SHA a6c3d64d...). Masks overlap.
- Noise: two split draws, seeds 39001001 and 39001002, fixed for every observation (B12 observation convention).
- Training set: lightcone_100 + mask 12345 for all 64,440 rows; per-row unique noise seeds (sha256 rule, master 20260920).
- Posterior RNG: seed 2 (first study, common random numbers); 7103 for joint HPD draws. Ensemble networks seeds 1,2,3.

## Key numbers (source)
- First study P0 medians: baseline 12.10; masks 23.70-45.92; sims 45.50-53.66 (tsz_seed_mask_variation_20261002/final_results/parameter_comparison.csv).
- Median scatter / baseline half width, P0: sims 1.163, masks 0.946 (RESULTS.txt). P0 mean width ratio 1.24 (sim), 1.32 (mask); xc 0.68/0.78 (variation_summary.json).
- Other nine seeds: P0 49.89 +- 2.32, beta 13.01 +- 0.21 (scatter of medians); seed 100 P0 12.15, beta 9.70 (seed_offset_summary.csv).
- Spectral blocks (other seeds vs 100): -6.14% (80-579), -0.063% (580-2979), +0.742% (2980-7979) (REPORT.md; diagnostics.json).
- Relative clean spectrum scatter below ell~580: 10.7% sims, 15.0% masks; <0.7% above ell~2980 (INTERPRETATION.md).
- Counterfactual table (REPORT.md): none 12.23; mean other 51.81; clean only 43.22; cross only 17.60; smooth5 13.90; fine5 40.09; fine50 42.43;
  reversed 4.56; +0.742% amplitude 12.43. Per-seed smooth 12.13-15.46, fine 33.89-42.39 (joint report p25). Reversed fine 4.67.
- Mask/combined table (REPORT.md): other sims 51.81/43.22/13.90/40.09; other masks 37.82/37.37/11.34/37.73; both 47.09/43.01/13.30/41.27.
- Bounded original: mean-new 52.16, smooth 11.67, fine 44.50. BatchNorm cohort invariance 1.91e-06. NN percentiles 22-30. Clean PCA norms 21.6-39.7.
  Derivative singular-value ratio 1.2e-05 (condition ~85,000).
- Held-out (INTERPRETATION.md / joint report p17): sim 1.746->1.169, 0.157->0.050, +24.7%/59.1%; mask 1.419->1.162, 0.459->0.098, +17.9%/55.4%;
  joint 1.660->1.162, 0.342->0.097, +22.6%/74.4%. Full grid R 1.620->1.150, A 0.444->0.092, G 1.215.
- Variance fractions: original 26.3/31.8/41.9 %, band50 12.4/73.1/14.4 % (sim/mask/interaction).
- Joint truth: original inside joint 68% on 1/100, 95% on 2/100; band50 100/100 at 68% (levels 0.135-0.294) (joint_truth_linear.json).
- Prior final test: original joint 64.1/94.3 %, band50 66.4/96.1 %; mean log density -3.013 -> -5.481.
- 580 cut: +28.7/+22.5/+27.0 % mean, worst 68.1/60.0/80.7 % (truth-mean README).
- Full-prior widths: band50 U +45.64 %, band50 B +46.41 %, 580 U +56.51 % (truth-mean README).
- Truth-mean Pearson P0 0.733/0.537/0.508; xc 0.790/0.694/0.645; beta 0.696/0.478/0.390.
- Pearson report (unbounded): P0-beta -0.337/-0.277/-0.317; P0-xc -0.645/-0.632/-0.642; xc-beta 0.813/0.791/0.804; largest |dr| 0.061+-0.0043;
  width ratios 1.012/0.999; bounded P0-beta -0.352 -> -0.297 / -0.330; inside-prior -0.361 -> -0.292 / -0.336.

## XGPaint
- No XGPaint source was edited for any seed/mask experiment. The painter is the frozen 64k engine, julia_env pins
  kristero/XGPaint.jl tag validated-tsz256-20260922 (= 35f79ad, tree identical to ce1d239 per its commit message).
- The nine variable B12 parameters come from fork commit f16a7f3 (parametrized Battaglia16ThermalSZProfile, PowerLawParam);
  e67a8a1 changed the beta convention (beta = alpha*beta_raw - gamma, exponent -(beta+gamma)/alpha, same profile values) and added
  prepare_profile_slice (amplitude) used by the project wrapper. At theta = B12 the profile equals upstream's hard-coded one.
- Project-side (NOT XGPaint): 4 R200c sphere = ChordMeanProfile in engine/spherical_truncation_profiles.jl; per-seed halo-count
  assertion in paint_seed.jl:218-222 replacing engine.jl:73.

## Caveats kept in the section
- Masks overlap, one noise pair: sensitivity, not coverage. Ten skies at one theta. Counterfactual inputs are hybrids, not mocks.
- Band50 (2l+1-weighted 50-multipole bands, 158 bandpowers) is NOT Gaussian sigma_l = 50 smoothing (that is the Pearson control).
- 18-29% (fixed-B12 seed/mask) vs 46-57% (full prior) width costs measure different things.

## XGPaint beta-convention history (checked in the dev checkout, 2026-10-07)
- upstream 29ec48a: beta = gamma - alpha*beta_raw, exponent (beta-gamma)/alpha -> outer factor (1+x^a)^(-beta_raw) (correct).
- f16a7f3: same conversion, now parametrized (correct).
- 71bc3fd / b924ab1 ("Adrian fixes", 2026-03-03): exponent -(beta+gamma)/alpha (tau correction) with beta = -gamma - alpha*beta_raw
  -> (1+x^a)^(+beta_raw) for tSZ (wrong sign).
- e67a8a1 (2026-04-30): beta = alpha*beta_raw - gamma -> (1+x^a)^(-beta_raw) again (correct); adds prepare_profile_slice.
- Tag validated-tsz256-20260922 = 35f79ad is after e67a8a1, so the 64k data and every seed/mask sky use the correct profile.
  Whether any earlier product used the intermediate state is outside this section (check Sections 1/3).

## Figure files (fig/4_seedmask/, copied by scripts/seedmask/copy_figures.py; PDF beside each PNG)
01_grid_design, 02_signal_variation, 03_first_study_variation, 04_posterior_simulation, 05_posterior_mask,
06_combined_P0_grid, 07_counterfactual_posterior_response, 08_mechanism_summary, 09_fingerprint_per_seed,
10_baseline_smoothing, 11_pearson_linear_delta, 12_held_tradeoffs, 13_variance_width, 14_joint_truth_linear,
15_saved_scale_cut_comparison, 16_width68_ratio_full_prior.
Section built by scripts/seedmask/build_section.py from scripts/seedmask/s4_template.html (code excerpts pulled verbatim).
