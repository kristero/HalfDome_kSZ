# Verification of sections/s5_merlin.html (5. Merlin and 50-multipole bands)

Checker pass, 2026-10-07. The fragment was edited in place; the builder `scripts/merlin/build_section.py` was NOT
updated (outside this checker's write scope), so re-running it would revert these fixes.

## 1. Code excerpts (12)
All 12 `<pre class="code">` blocks compared with `sed -n 'L1,L2p'` after HTML-unescaping: all verbatim.
One range was changed: merlin_compression.py 342-379 -> 341-379, so the excerpt includes the `mean` buffer that the
code-note discusses (39 lines). Prose line references checked: merlin_unbinned_prepare.py 194-212, 213-228;
merlin_beam_prepare.py 71-73; fisher_forecast_b12.py 63-64, 219-224; train_ensemble.py 60-82 -> 60-83 (the
log-Jacobian return is on 83), 359-376; merlin_compression.py 471-473, 53-55; pipeline.py 40, 78-84, 127-129, 165, 207-220;
common.py 48-66; w420 line 135; engine.jl 31-51; spherical_truncation_profiles.jl 22; Manifest.toml 1976-1982.
Fixed: docstring range "lines 1 to 25" -> "1 to 33" (the noise-redraw statement is on line 31).

## 2. Figures (16)
All exist, open, are separate files; 8 same-size copies are pixel-identical to their sources, the others are
downscaled copies of the cited sources; PDFs are byte-identical to the source PDFs. Each viewed against its caption.
Caption fixes: 5.2 (V2/V6 are not confined below l~1000), 5.6 (input set and "95 % inside"), 5.7c (PCA at 61,440
differs: -4.33 vs -4.67), 5.9 (all-multipole Fisher is the calibrated one, centred on the input values).
Note: 04_head_feature_correlations.png is 884 px wide (source resolution).

## 3. Numbers traced (about 90)
Against prepare.json / prepare_whitening.json / noise_scatter_bands.json / metrics_merlin.json and *_seed*.json of
all 12 Merlin runs, fisher_forecast_b12.json, publication_numbers_fisher.json, results_corrected_inputs/summary.md,
results_doc/index.html, discovery_ranking.csv, INTERPRETATION.md, REPORT.md, truth_mean_metrics.csv,
paired_pearson_changes.csv, TRUTH_MEAN_SUMMARY.json, README.md (truth mean), pearson_pairs.csv / pearson_linear.json,
obs_posteriors_beam/summary_bounded.json. Corrections:
- "averaged over bands the ratio is about 1.02 at l > 6000": unsupported; noise_scatter_bands.json gives 0.96-1.06 per
  100-multipole band above 6000, mean 1.00.
- "variances agree to 1.5 %": the stored quantity is the median |sd_1000/sd_61440 - 1| = 0.015 (standard deviations).
- B12 truth log p seed error "0.3 to 0.6": actual 0.26-0.56 for the bounded runs cited.
- whitening gain "0.3 to 0.5 nats": true for beam-consistent noise; production noise-only vs 20-draw is 0.19.
  Also in the conclusions ("0.2 to 0.5"). Single-seed spread "0.2 to 0.3" -> 0.17-0.28 (test_log_prob_seeds).
- first run date "2026-09-29" -> 2026-09-30 (networks/metrics dated 09-30; inputs 09-29/30).
- band50 audit "at least 32 rows" -> exactly 32 (pipeline.py:40).
- "compared 24 compressions with the original" -> 23 alternatives, 24 methods in all.
- PCA variance kept is of the whitened clean spectra.
All other checked values agree (OAS 0.9517/0.9519/0.173, lag-1 0.455/0.040, rho_1..5, S(pi) 7.7e-7, min eig 0.00100,
n_pca 75/82/83/86, scale ranges, parameter counts, epochs and seconds, every table cell of the run-history,
bounded/unbounded, headline, sky-test and truth-mean tables, Fisher 0.146/0.595/0.813/0.559/1.663 and band series,
0.231/0.288/0.461/0.432, seed-offset 49.89/13.01/12.23/9.66/12.13-15.46/40.09/42.43/12.43/32.7/32.8, discovery
1.560/0.498/1.134/0.077/1.199/1.175/0.147/1.250, smooth_clean_50 -0.317/-0.337/0.999).

## 4. Physics / statistics
- OAS equations: added S_z = Z^T Z / n and the unit-diagonal renormalisation (the code renormalises before C = D R D).
- ReduceLROnPlateau: corrected to PyTorch semantics (relative threshold 1e-4, reduction at the (lr_patience+1)-th bad
  epoch, i.e. 11th for lr_patience = 10; matches the explainer's "x0.1 after 11 epochs"); early stop uses absolute 1e-6.
- sqrt(1 + 2S/N): defined S, N and why there is no S^2 term (fixed signal map across draws).
- asinh scale c_j = 2-7: the old "reflect the noise in a projected direction" replaced by the derivation (c_j = 1 if the
  whitening covariance were the row's noise) and two inferred causes (S x N term; OAS-shrunk neighbour correlation,
  up to sqrt(S(0)) = 1.55 for the 20-draw whitening), flagged as inference.
- Spectral density S(omega) defined; S(0) = 2.41 added.
- Unbounded overconfidence: the "geometric" mechanism was stated as fact without a source; now flagged as a hypothesis,
  with the +-0.017 binomial range for 3,000 rows. The "decision ... direction-of-pull diagnostic" had no source;
  replaced by the docstring rationale (merlin_compression.py:53-55) and the explainer's "bounded used for all comparisons".
- TARP definition made precise (reference point uniform in the test-truth range, scaled distances) with citation.
- HPD and the joint truth test defined (INTERPRETATION.md); band50's P0 median 32.7 noted as itself above truth 18.1.
- "noise on any other sky" -> the fine structure biases a network on another sky (fine-residual controls).
- Fisher comparison: all-multipole curve is the calibrated Fisher (8 % wider), PCA-input curve the model Fisher
  (publication_figures.py:353-358).
- Coverage ranges for band50 differ between INTERPRETATION.md (63.3-66.9 / 68.8-72.7) and the 2026-10-06 truth-mean
  run (63.2-66.8 / 68.3-73.0); text now says the later dedicated run is used.

## 5. XGPaint
Verified in /home/kn18001/.julia/dev/XGPaint: tag validated-tsz256-20260922 -> commit 35f79ad; trees of 35f79ad and
ce1d239 are both 3b57cacb5f82...; Manifest pin at lines 1976-1982. engine.jl:31-51 adds unbeamed noise
("noise_beam_applied"=>false); ChordMeanProfile <: XGPaint.AbstractGNFW at spherical_truncation_profiles.jl:22 (project
code). Fixed: full repo-relative paths; "Neither changes any number" was wrong for the unbeamed noise (it does change
the production-noise rows; the beam-consistent rows are the corrected ones) - rewritten. No XGPaint edit in this topic.

## 6. HTML
One <section id="merlin" class="topic">, all tags balanced, only contract classes, no script/style, no em-dashes,
no bare ampersands, only https links, PNG images only.

## 7. Clarity
Added definitions: C^ss, C^noisy, C^nn, B_l; nats and test log p at first use; 40 bins (last band 100 wide, 2l+1
weights); HPD; half-sample Monte Carlo check; Hann window. The 40-band beam-consistent r (0.590) added beside the
production 0.584 in the band50 cross-check.

## Remaining concerns
- Prose is long (about 10,700 words including captions, tables and code-notes) against the 3,000-7,000 guideline.
- build_section.py still emits the pre-check text.
- (resolved in pass 2) the XGPaint tag message itself says "XGPaint used by the validated 256-row HalfDome tSZ test (2026-09-22)".

# Second checker pass (2026-10-07, "try again")

Independent re-check of the fragment as left by pass 1. Fragment edited in place again; build_section.py still not
updated (re-running it reverts both passes).

## Re-verified
- 12/12 excerpts verbatim (script compare after HTML-unescape); prose line refs re-read (docstring l.31, ff 63-64,
  219-224, 20-24; w420 l.135; te 60-83, 359-376; mc 53-55, 471-473; pipeline 78-84, 127-129, 164-168, 207-220;
  merlin_checks 54-58 uses local_model.npz C_nn = pipeline nn_cov; engine.jl 31-51; spherical_truncation_profiles.jl 22;
  Manifest 1976-1982).
- 16/16 figures viewed; 8 same-size copies pixel-identical (max diff 0), 8 PDFs byte-identical to sources.
- ~110 numbers recomputed from: 12 metrics_merlin/pca/moped.json (lp, err, TARP, mean r, width, coverage, B12 lp, seed
  ranges, epochs, seconds, parameters), 6 prepare.json, prepare_whitening.json, noise_scatter_bands.json (bands >= 6000
  are 100 wide, last 80; 0.955-1.058, mean 1.004), fisher_forecast_b12.json (geo-means 0.1462/0.2033/0.2872/0.3749/
  0.5201/0.8126/1.1781/1.6626/0.5948/0.1585; low-pass P50 0.5592; S_0 2.414, S_pi 7.71e-7; derivative corr >= 0.99999;
  step h = 0.5 % of width), publication_numbers_fisher.json (0.2306/0.2880/0.4612/0.4321), results_doc text (headline
  table, paired diffs, brightness split, 0.7-1.1 nats, trainer table, +-0.017, 33-34 %), summary.md, REPORT.md,
  discovery_ranking.csv (24 methods, width range 1.199-1.361), INTERPRETATION.md, truth_mean_metrics.csv,
  paired_pearson_changes.csv (bounded deltas within 0.008 of unbounded), metrics_*.json of the truth-mean run,
  pearson_pairs.csv / README (0.99867), summary_bounded.json checks, head-feature max |corr| 0.612 (doc_metric_data).

## Corrections made in pass 2
- Beam-fix checks (5.2): 2.6e-15 / 1.7e-15 are the uploaded PRODUCTION and clean per-multipole spectra rebinned to the
  stored 40 bands, not "the rebuilt spectra"; the rebuilt beam-consistent spectra match the staged run exactly
  (beam40 = beam_unbinned = 0) (merlin_beam_prepare.py 150-166).
- Beam-fix effect with noise-only whitening: 0.23 -> 0.22 nats (-3.185 -> -2.964).
- band50 audit: pass 1's "exactly 32 rows" was wrong; pipeline.py 40-42 audits 32 random rows PLUS the first four
  rows of each staged beam-fix batch.
- 400 new noise realisations: seed PAIRS 950001/950002 .. 950799/950800 (idark_fisher/make_tasks.py).
- Fig 5.12 caption: the orange steps are band averages of the NOISY spectrum (plot_truth_mean.py 96-104).
- Fig 5.8 caption: "none of the curves has levelled off" -> Merlin/PCA still rising; 40-band log p flat at the end.
- Fig 5.7d caption: describe both curves (total-noise excess vs noise x noise at 1).
- Calibrated Fisher defined properly (t = J^T C^-1 r on held-out draws, (n-10)/n Hartlap-type factor,
  F_cal = M Cov(t)^-1 M, lower bound), and the Merlin B12 widths labelled as the 420-draw whitening ("beam_w420").
- 5.10: the 49.89/13.01 shift is for the nine other LIGHTCONES at the training mask (REPORT.md), +-  is scatter of
  medians; "excludes truth at 95 %" is from the later 10x10 robustness grid.
- 5.10 Fisher: added that derivatives exist only for the training sky, so their sky dependence is inferred.
- 5.12: the l>=580 control also failed the joint calibration screen in the mask and combined arms.
- 5.13: joint log p -5.48 vs 40 bins -2.35/-2.39 is across different row subsets and spaces; added the original's
  -3.01/-2.87 (1,500 rows) vs -2.96 (3,000 rows) to show the subset effect is small.
- Lede and 5.14: softened causal wording ("the evidence indicates"; FLAMINGO/real-sky transfer "not expected, not
  tested"; "conditional upper bound" -> "hold conditional on the training sky and are optimistic for any other").
- Cross-reference "section 3" -> "section 3.4" (metric definitions) and explainer section 11.

## Remaining concerns
- Length about 10,800 words including captions, tables and code-notes (guideline 3,000-7,000).
- build_section.py emits the pre-check text.
- band50's whitening factor L comes from OLD/reference/whitening_pca.npz on idark (not local); that it is the 83-PCA
  beam noise-only model rests on the seed/mask README and INTERPRETATION.md, not on a local file read.
