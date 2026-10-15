# Dossier: section 5, Merlin and 50-multipole bands

Fragment: `sections/s5_merlin.html` (id `merlin`), built by `scripts/merlin/build_section.py` (prose + verbatim
excerpts read from the current files by line range). Figures: `fig/5_merlin/` (15 copied, 1 regenerated from saved
data by `scripts/merlin/fisher_band_scale.py`).

## What Merlin is
- Method of Wernersson, Franco-Abellan & Canas-Herrera 2026, arXiv:2609.24946, Sect. 3.3, code merlin/preprocessing.py
  and merlin/network.py. Citation taken from the docstring of `SBI_analysis/tsz64k_sbi_local/merlin_compression.py:1-25`
  and the explainer page `results_doc/index.html`. The original paper/code is NOT on this machine (searched); only the
  project re-implementation exists.
- Adopted: whitening with a noise covariance, PCA of whitened noise-free spectra (torch.pca_lowrank q=500, keep
  S/sum S*100 > 0.001), learned head standardize -> 2048 -> ReLU -> 512 -> ReLU -> 9*N_feat (N_feat=2) -> BatchNorm1d,
  merlin training setup (AdamW 5e-4, batch 256, ReduceLROnPlateau x0.1 after 3, early stop 5, 20 % validation).
- Changed: head trained jointly with the sbi MAF (embedding net) instead of swyft marginal ratio estimators; one noise
  realization per simulation (original redraws Gaussian noise every epoch); fixed standardization instead of swyft's
  online layer; noise-level asinh input transform (project addition, 2026-10-01); patience 30/10 via env vars.

## Chain and code
1. Unbinned D_ell, ell = 80..7979 (7,900 values), 65,536 collaborator rows -> 64,440 analysed, pool 61,440 / test 3,000
   (`merlin_unbinned_prepare.py`, checks against 40 bands to 1e-9).
2. Beam-consistent spectra `C_ss + B (C_noisy - C_ss - C_nn) + B^2 C_nn` (`merlin_beam_prepare.py:71-73`), checks
   noisy40 2.6e-15, clean40 1.7e-15, beam40 0, beam_unbinned 0 (`results_merlin_unbinned_beam/inputs/prepare_whitening.json`).
3. Whitening:
   - 20 B12 draws, OAS on correlation, shrinkage 0.95167 (production), 0.95185 (beam) -> nearly diagonal.
   - noise-only: 61,440 pool C_nn (x B^2 B^2 for beam), shrinkage 0.17302, lag-1 corr 0.455; first 1,000 rows only:
     shrinkage 0.927, lag-1 0.040, sd agree to 1.5 %. B12 sd / noise-only sd at l=100,300,1000,2000,3000: 1.47, 1.66,
     1.19, 1.13, 1.26 (S x N term); 0.83/0.86 at 5000/7000 are 20-draw scatter (memory note + json).
   - w420: 20 + 400 beam-consistent B12 residuals; fisher_forecast_b12.covariance_model (running-mean variance +-25,
     Welch Toeplitz 64-lag Hann) + white floor 1e-3 x variance; rho_1..5 = 0.551, 0.049, 0.032, 0.026, 0.005;
     min eigenvalue of R after floor 0.00100 (before: S(pi) = min eig = 7.7e-7 in results_fisher_b12_beam json).
4. PCA: n_pca 75 (prod B12, beam B12), 82 (prod nn), 83 (beam nn), 86 (w420). First singular-value shares (prod B12)
   72.78, 11.63, 8.25, 3.43, 1.14 %; variance kept 0.9999999969.
5. asinh: z = asinh((s - median)/c), c = 1.4826 MAD(s_noisy - s_clean); scale range 4.10-6.94 (prod B12), 3.82-6.93
   (beam B12), 3.27-6.20 (beam nn), 2.23-4.77 (w420).
6. Network: head 2048/512/18 + BN, MAF 5 x 128, logit u. Params: Merlin 1,409,296 (75 in), 1,425,680 (83), 1,431,824
   (86); PCA 231,770 (75), 236,890 (83). Epochs 83-102 (asinh_p30), 89-98 (beam nn), 73-104 (w420).

## Results (metrics_merlin.json etc.)
| run | test log p | TARP 68/95 | mean r | width/flat | B12 log p(truth) |
|---|---|---|---|---|---|
| unbinned, linear inputs, patience 5 | -6.77 | .701/.964 | .635 | n/a | -10.23 |
| unbinned_p30, linear | -5.01 | .695/.961 | .662 | n/a | -8.70 |
| asinh_p30 (prod, B12-20) | -3.38+-0.06 | .674/.948 | .720 | .561 | -4.12+-0.37 |
| asinh_p30_lin | -3.59+-0.03 | .642/.957 | .721 | .502 | -3.68+-0.98 |
| nnwhite (prod, nn) | -3.19+-0.09 | .664/.949 | .723 | .556 | -2.34+-0.56 |
| beam B12-20 | -3.28+-0.05 | .675/.953 | .721 | .567 | -4.81+-0.32 |
| beam nn | -2.96+-0.05 | .675/.952 | .729 | .551 | -1.41+-0.26 |
| beam nn lin | -3.14+-0.05 | .635/.953 | .729 | .490 | -0.97+-0.48 |
| beam w420 | -2.82+-0.08 | .672/.953 | .728 | .547 | -1.81+-0.37 |
| beam w420 lin | -2.95+-0.05 | .635/.956 | .729 | .497 | -1.18+-0.05 |
PCA asinh_p30 -4.67/.677; MOPED -4.38/.583. Marginal 68 % coverage bounded .678/.675/.674 vs unbounded .631/.627/.629.
Size series asinh_p30 Merlin log p 2k..61k: -10.45 -9.60 -8.21 -6.33 -4.60 -3.38; r .582 .621 .662 .691 .709 .720.
Raw (no PCA, 32k): -7.21, r .542. Explainer headline (beam, noise-only): Merlin -2.96/.729/RMSE .663/width .551/cov .675/
IS .526; PCA -4.56/.684; 40 bins -2.35/.590/.787/.777/.721/IS .656. Paired Merlin-40 bins r +0.139+-0.004,
log p -0.61+-0.16, IS -0.130+-0.003; Merlin-PCA r +0.046+-0.003, log p +1.60+-0.21.
Fisher (beam w420 json): geometric-mean marginal sd/prior width: every multipole 0.146, Merlin 86 PCA 0.595,
bands 50 0.813, low-pass P>50 0.559, 40 SBI bands 1.663. B12 posterior 68 % width/prior mean: Fisher all 0.231,
Fisher PCA inputs 0.288, Merlin bounded 0.461, unbounded 0.432 (publication_numbers_fisher.json).
Derivative fine structure h vs 2h correlation 1.000 for all 8 non-P0 params (deterministic, fixed-sky).

## band50
- `tsz_robustness_methods_20261004/common.py:29,48-66`: EDGES arange(80,7981,50) -> 158 bands, 2l+1 weights;
  covariance A C A^T. `pipeline.py:127-129`: nn_cov = (A L)(A L)^T / fid fid^T, L the noise-only (beam nn, 83-PCA
  original) Cholesky factor; `pipeline.py:164-185`: Merlin branch whitens 158 bands with nn_cov (no PCA), asinh
  (fit_transform, same formula), MAF+head, 3 seeds, patience 30/10, linear space primary (`train_one`, 207-220);
  bounded companions `bounded.py`. lmin580 etc. in `merlin_checks.py` (C_nn whitening).
- The 'covariance': 'total_noise' entry in the recipe is not used for the Merlin branch (comment line 165).
- Dividing by fid is a no-op after whitening (Cholesky of D C D is D L).
- Sky study (held-out): truth-RMS reduction 33.0/18.1/30.0 %, median-scatter reduction 68.2/78.7/71.5 %, width +24.7/
  +17.9/+22.6 %, worst parameter +59.1/+55.4/+74.4 % (sim/mask/combined; INTERPRETATION.md). Discovery combined:
  original truth 1.560 scatter 0.498; band50 1.134, 0.077, width 1.199; lmin580 1.175, 0.147, 1.250.
  Truth inside band50 68 % joint HPD on all 100 skies; original excludes truth at 95 % on every held-out sky.
  Variance split original 26.3/31.8/41.9, band50 12.4/73.1/14.4 (sim/mask/interaction).
  Spectrum relative scatter below l~580: 10.7 % (sims), 15.0 % (masks); <0.7 % above l~2980.
- Seed offset (REPORT.md): other nine seeds P0 49.89+-2.32, beta 13.01+-0.21 (original unbounded); seed 100 P0 12.23,
  beta 9.66; smooth sigma_l=5 control returns P0 12.13-15.46; fine residual 40.09 (sigma 5), 42.43 (sigma 50);
  band50 P0 ~32.7 baseline, 32.8 mean new obs.
- Truth-mean (2026-10-06, 1,500 odd reserved rows, 9,000 draws): Pearson orig/band50/580 P0 .733/.537/.508,
  xc .790/.694/.645, beta .696/.478/.390; mean r .730/.584/.526; width geo-mean ratio 1.4564 (band50 lin),
  1.4641 (band50 logit), 1.5651 (580 lin; README 56.51 %); coverage68 band50 lin .632-.668, logit .683-.730;
  original lin .607-.665, logit .637-.714; mean log p truth orig lin -3.013, logit -2.874, band50 -5.481/-5.475,
  580 -6.282; outside prior orig .329, band50 .394, 580 .393; banded features reconstruct with max error 0.0.
- 40 bins (production, 61,440, 3,000 rows) mean r .584, per-parameter .51 .69 .48 .65 .65 .32 .79 .72 .45
  (results_corrected_inputs/summary.md) ~ band50 Merlin .537 .694 .478 .639 .637 .335 .778 .710 .451 (different row
  subsets: 3,000 vs 1,500 odd).
- smooth_clean_50 (joint report 2026-10-05): Gaussian sigma_l=50 smoothing of the fractional mean clean difference of
  sims 102-118, applied as input to the ORIGINAL frozen Merlin; posterior pair correlations P0-beta -0.337/-0.277/-0.317,
  width ratios 1.012 (sigma 5) / 0.999 (sigma 50). Not band50.

## XGPaint
No XGPaint edit in this topic. Training spectra from the frozen engine pinned to kristero/XGPaint.jl tag
validated-tsz256-20260922 (tree 3b57cac..., identical to commits ce1d239 and 35f79ad in the dev checkout). Beam
inconsistency fixed by beamfix lives in the project engine (`engine.jl:31-51`, noise added unbeamed,
"noise_beam_applied"=>false); 4R200c sphere is the project wrapper `spherical_truncation_profiles.jl:22`
(ChordMeanProfile <: XGPaint.AbstractGNFW). Merlin/band50 are pure Python post-processing of stored spectra.

## Open
- Zero-centred asinh for PCA/Merlin inputs (faint skies) untested; matched trainer vs 40 bins untested.
- No band-width sweep (25/100 only illustrated); 580 bounded companion never trained.
- Training on a single sky/mask; multi-sky training is the principled fix (not done).
- B12 truth log p is one observation with seed SEM 0.05-0.98: not a ranking metric.

## Figures (fig/5_merlin/, in page order)
01_whitening_20draw_vs_noiseonly (results_doc/step_whitening), 02_pca_components (step_pca), 03_asinh_inputs
(step_asinh), 04_head_feature_correlations (merlin_features), 05_training_curves (step_training), 06_bounded_logit_space
(step_logit), 07_whitening_test_logp (results_publication/fig_pub_merlin_beam_whitening, +pdf), 08_size_series_beam
(results_doc/compare_sizes_beam), 09_corner_b12_fisher_vs_merlin_w420 (fig_pub_corner_B12_fisher_merlin_beam, +pdf),
10_seed_offset_smooth_vs_fine (tsz_seed_offset_diagnosis_20261005/plots/mechanism_summary, +pdf),
11_fisher_vs_band_width (NEW, scripts/merlin/fisher_band_scale.py from results_fisher_b12_beam json),
12_band_operator_25_50_100 (truth_mean figures/band_width_illustration, +pdf), 13_robustness_tradeoff_combined
(robustness report/tradeoff_combined, +pdf), 14_truth_mean_pearson (+pdf), 15_width68_ratio (+pdf),
16_truth_vs_posterior_mean_unbounded (truth_mean_linear, downscaled to 1700 px, +pdf).
Large PNGs downscaled to 2000 px width with PIL; all under 1.5 MB.
