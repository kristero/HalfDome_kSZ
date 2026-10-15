# Verification of section 3 (sections/s3_sbi64k.html), second pass, 2026-10-07

Independent re-check of the whole fragment (after the first checker pass, kept below). Only the fragment and this file
were changed; fig/3_sbi64k/ needed no change. As before, scripts/sbi64k/s3_template.html does NOT contain the fixes of
either pass; do not re-run build_section.py without porting them.

## Checks and results

1. Code excerpts: 12/12 verbatim against the current files after HTML-unescaping (XGPaint at
   /home/kn18001/.julia/dev/XGPaint, HEAD 1660e08). Line references inside the notes re-checked: engine.jl 23/34-37/40-43;
   profiles_y.jl 84; spherical_truncation_profiles.jl 125 (get_params) and 104 (prepare_profile_slice amplitude);
   sbi_final.py L328/L333/L336-L338/L345-L351, L681-L705, L868-L873; train_ensemble.py sample_u L334-L357 (999 draws =
   3 x 333); doc_figures.py interval_score L541-L545; constants EDGE 0.01, FLAT_WIDTH68 0.6827, UNIT_CLIP 1e-9. p.path
   function ranges correct to within a line or two.
2. Figures: 16 PNG + 1 PDF open, 1300-1900 px, all < 0.6 MB. 01-09, 12, 15, 16 and the PDF byte-identical to the cited
   sources; 10/11 the cited sources at 1900 px; 13 the documented re-render; 14 the report helper figure. Each viewed and
   matched to its caption (B12 68 % misses recomputed from metrics b12 q16/q84: Merlin and PCA miss xc, beta; 40 bins
   also alpha_z_P0; 40-bins P0 draws pile at 60, 11.7 % above 57 against 5.1 % flat). Fig 3.10 draw files confirmed
   (300,000 x 9 each).
3. Numbers traced (about 170): all scorecard, per-parameter, production, unbounded and w420 rows (metrics_*.json);
   seed log p and ensemble gains (0.66-1.09 Merlin, 1.07-1.68 PCA); 40-band seeds -3.144/-2.845/-2.665; training records
   (epochs 340/404/436; 73-110; AdamW 5e-4, batch 256, val 0.2, ReduceLROnPlateau patience 10; sbi Adam 7e-4, batch 512,
   VALIDATION_FRACTION 0.1); OAS shrinkage 0.952/0.173, lag-1 0.455; both convergence tables (every log p, r, TARP);
   checks.json (prior -12.8975 = -sum log w recomputed; Fisher eigen-widths; +7.98/+8.71; B12 -6.29, 21 draws; eigen-sd
   0.19-0.32 -> 0.12-0.29; network differences); convergence.json (L_inf = last + 20 everywhere); series.json (+0.345 +-
   0.023, clean-noisy +0.619 +- 0.031, 61,440 row -2.354/0.726); validation.json (33 ensembles, mean -0.039, -0.100 ...
   +0.027, SEM 0.065-0.144, KS p, TARP); row_id.npy (61 runs, rows 7,456-60,687); final_dataset.py + missing_batches.json
   (274 batches of 4); diagnostics.json (box-space 0.51, 0.84-0.86); config_comparison.md; beta_bias.json;
   fisher_forecast_b12.json (marginal sd ranges; coherence 1.0; 20 + 8 x 50 residuals); publication_numbers_fisher.json
   (mean widths 0.23/0.29/0.46/0.43, inside 0.7238); publication_numbers.json SBC; final_calibration_numbers.json (all
   values); b12_posterior_correlations.json; logp_gain_per_doubling.json; prior width ratio 1.815-7.604; Merlin/PCA small-N
   r (0.578/0.617, 0.535/0.575/0.605); Merlin size series with/without beam fix (|diff| <= 0.036 to 32k, +0.105 at
   61,440); worst Merlin row -34.74; results_doc section numbers (3, 5, 7, 9, 10, 11.8, 12); s4 P0 shifts (12 -> 45-54,
   24-46); seeds 131,072 distinct; tag tsz64k-flat-prior-v1 = b46ab65, bundle unmodified; mass cut (catalogue M / h >=
   mass_min = 1e12 Msun).
4. Physics/statistics re-derived: Pe/Pth = 2(X_H+1)/(5X_H+3) = 0.5176 for X_H = 0.76; P200c; logit Jacobian; NPE/KL; log p
   proper score; r invariance; prior-centre RMSE w/sqrt 12; flat interval score 0.84 (0.68 + 2 x 6.25 x 0.0128);
   binomial +-0.017/+-0.008; TARP against sbi 0.27 source (references Uniform(min, max) of truths, min-max scaling of
   samples/truths/references, num_bins = 3000 // 10, cumsum ECDF); KS 1.358/sqrt(3000) = 0.0248; SBC shape reading;
   beam algebra (engine overstates C^sn by 1/B, C^nn by 1/B^2); boxed-Fisher expected log p (-1/2 log det 2 pi Sigma - 9/2 -
   E log Z); Richardson (4 J_h - J_2h)/3; NSIDE 4096 pixel 0.859 arcmin; e^1.3 = 3.7.
5. XGPaint: f16a7f3 (keyword constructor, kept upstream beta = gamma - alpha beta_raw with exponent (beta-gamma)/alpha),
   9f37b1e (adrianbayer, upstream/adrian: exponent -(beta+gamma)/alpha and beta = -gamma - alpha beta), merged by b924ab1
   (old line 81 confirmed), e67a8a1 (beta = alpha beta_raw - gamma; introduced prepare/evaluate_profile_slice per git log
   -S), 35f79ad (tree 3b57cacb = ce1d239 tree = Manifest git-tree-sha1); HEAD differs only in test/runtests.jl; upstream
   main 29ec48a hard-codes B12 in get_params; LOS quadrature zmax 1e5. Not-XGPaint items confirmed project code.
6. HTML: one section#sbi64k.topic, balanced, allowed tags/classes only, no script/style, no em dashes, https links only.

## Corrections made in this pass

- Beta-degeneracy study: the chi^2 was described as "within chi^2 < 100 of the B12 observation"; beta_bias.py compares each
  row's P0-rescaled noise-free 40-band spectrum with the noise-free B12 spectrum (pulls in units of the sd of 20 noise
  draws; the noisy mock has chi^2 = 48). Rewritten. Also: the networks were MOPED logit-space ensembles; "4.38 (noisy) /
  4.48 (noise-free)" = noise-trained network on the noisy mock / noise-free-trained network on the noise-free spectrum;
  the "not a beam artefact" claim now cites the FWHM 1.8-2.2 arcmin input scan (beta median 11.31-11.12).
- SBC bounded vs unbounded: added that bounded PCA and MOPED in the same file also exceed the band (0.048, 0.046), so SBC
  separates the parametrisations only for Merlin (bounded Merlin max 0.032).
- Network size: "with the same slope in N" (only two sizes for MAF 256x8) -> similar 32k -> 61,440 step (+1.02 vs +0.90);
  "about half a doubling" -> half to two thirds (0.65-0.86 vs 1.27).
- Fisher correlation note: "calibration changes by < 0.01" holds for the seven tabulated pairs only (max 0.031 elsewhere).
- Per-parameter r gains given to three decimals (alpha_m_P0 +0.185 had been rounded to +0.19, overlapping the "largest"
  group).
- Faint skies: 1,300 / 2,100 noise units are for the first two principal components (24-270 on components 3-6).
- Fig 3.6 caption: "PCA touches its edge" -> +0.024, just inside the 0.025 band.
- Clarity: X_H and the 0.5176 factor; halo-mass units (M_sun/h divided by h before the 1e12 Msun cut); f_sky; 40-band
  noise-level asinh scale N_b/sqrt(nu_b) with N_b, nu_b defined (table row and 3.2); NSF; PCA acronym; Pearson r
  explanation (linear-invariant, collapse onto the prior centre); f16a7f3 keyword count (15 profile + 3 cosmology); the
  engine note no longer uses C^nn for the auto noise bias (C^nn is the noise cross spectrum later).

## Remaining concerns

- Fixes of both passes are not in scripts/sbi64k/s3_template.html.
- Prose is about 9,100 words excluding code and tables (guide 3,000-7,000); not cut, since the content is non-redundant.
- Non-uniform P0 of the 1,096 validation rows unexplained; EDQUOT cause of the lost batches only in project notes.
- The SBC file used for the bounded/unbounded statement (publication_numbers.json) is the production-noise set, not the
  final beam-consistent set.
- The scratchpad directory is shared with other concurrent sessions (a checker script was overwritten mid-run); results
  here were re-run with uniquely named scripts.

---

# Verification of section 3 (sections/s3_sbi64k.html), 2026-10-07

Checker pass over the fragment written by the section-3 author. Only the fragment and this file were changed; the
figure folder fig/3_sbi64k/ was checked but needed no change. NOTE: the author's template
scripts/sbi64k/s3_template.html (built by scripts/sbi64k/build_section.py) does NOT contain these fixes. Re-running
build_section.py would overwrite them; edit the fragment, or port the fixes to the template first.

## 1. Code excerpts (12)

All 12 `<pre class="code" data-src>` blocks match the current files byte for byte after HTML-unescaping (XGPaint paths
resolved to /home/kn18001/.julia/dev/XGPaint, HEAD 1660e08). Line references inside the code notes (engine.jl lines
23/34-37/40-43, profiles_y.jl line 84, sbi_final.py L328/L333/L336-L338/L345-L351, L681-L705, L868-L873,
doc_figures.py L541-L545, train_ensemble.py sample_u L334-L357) checked against the files: correct.
p.path line ranges (sbi_final, train_ensemble, merlin_compression, posterior_rows, doc_metric_data,
publication_figures) checked with grep -n: correct to within the last line or two of each function.

## 2. Figures (16 PNG + 1 PDF)

All exist and open; 01-09, 12, 15, 16 and the PDF are byte copies of the cited sources; 10 and 11 are the cited
sources downscaled to 1900 px; 13 is the documented local re-render; 14 is the report helper figure. Each was viewed
and matches its caption, with these caption fixes:
- Fig 3.10: draws were "30,000 Merlin/PCA, 300,000 40 bins"; the figure function (doc_figures.fig_obs_b12_three) reads
  posterior_rows_whitening/beam_nn_bounded_{merlin,pca}_b12.npy (300,000 each) and beam_bins40_b12.npy (300,000).
  PCA mean B12 width 0.54 -> 0.53 (b12_width68_over_prior mean 0.5348). Added: 40 bins also misses alpha_z^P0 at 68 %
  and its P0 marginal piles up at the upper edge.
- Fig 3.5: "Merlin sits at or slightly below 0.68" -> scatters around 0.68 (0.643 ... 0.716); added that Merlin's 95 %
  coverage 0.943 is below the +-0.008 binomial range.
- Fig 3.11: stated that the Fisher Gaussian is centred on the input values and the filled one uses the calibrated
  covariance (publication_figures.fisher_corners, key boxed_forecast_every_multipole_calibrated).

## 3. Numbers traced (about 120; all others in the fragment were left unchanged because they matched)

Matched (selection): Merlin/PCA nnwhite metrics (log p -2.964 +- 0.055, sem 0.089; r 0.7295/0.6838; RMSE, width,
coverage, TARP, per-parameter rows); 40-band beam metrics (-2.354, 0.5904, 0.7875, 0.7774, 0.7209, 0.7257/0.963;
epochs 340/404/436; Adam lr 7e-4, batch 512, val 10 %); Merlin training (AdamW 5e-4, batch 256, val 20 %, lr patience
10, epochs 89-98); B12-whitened beam and w420 rows; production rows (-3.38/-4.67, r 0.720/0.677); unbounded
(beam nnwhite lin: r 0.7288, cov 0.627, TARP 0.635/0.953; production lin: -3.59, width 0.502, cov 0.631);
convergence tables (results_final, results_final_noise: every quoted log p, r, TARP); checks.json (prior -12.8975,
slope 1.2675, differences +0.47+-0.27, +0.30+-0.38, -0.12+-0.21, +0.73+-0.23, +0.86+-0.22, +0.65+-0.21; Fisher
eigen-widths 5.2e-5 ... 10.7; NPE eigen sd 0.12-0.29; B12 log p -6.29 over 21 draws; +7.98/+8.71); convergence.json
(L_inf = last + 20 for all variants; fit_at_bound True in results_final_noise); series.json (+0.345 +- 0.023,
clean-noisy +0.619 +- 0.031, last-band noise ratio 0.1326); final_calibration_numbers.json (coverage95, IS, SBC
maxima and counts, TARP deviations, paired differences, alpha_z_P0 bins, wide rows, coverage by brightness) -
recomputation logic read and test-row order verified against results_final/split.npz (max |truth diff| 1.9e-6);
b12_posterior_correlations.json; fisher_forecast_b12.json (40-band sd 1.14-2.70, calibrated 0.094-0.234, smoothed
+-50 multipoles 0.725-1.779, coherence 1.0, 420 = 20 + 8 x 50 realisations); publication_numbers_fisher.json
beam_w420 (widths 0.23 [0.12-0.34], 0.29, 0.46, 0.43, inside 0.7238); publication_numbers.json sbc_unbounded Merlin
max 0.0483; diagnostics.json (box-space TARP 0.51, 0.84-0.86); config_comparison.md (-5.89 +- 0.13, 72/95, r 0.52);
beta_bias.json (25 rows, median 11.6, 4.3-15.8; conditional mode 4.38/4.48; 11.26 vs 11.23); validation.json
(33 ensembles, mean -0.039, range -0.100 ... +0.027; KS p 0.005 for P0, >= 0.99 others; TARP 73/97, 70/96, 72/97);
logp_gain_per_doubling.json; fig_correlation_prior_comparison.json (ratio 1.815-7.604); design (131,072 distinct
seeds; prior box); README (85,224,251 halos; NSIDE 8192 -> 4096; 2'); Manifest L1976-L1982 (3b57cacb).

Corrected or qualified:
- Ensemble gain "0.7 to 1.7 nats over single seeds" applied to Merlin/PCA only (results_doc s10); 40 bins gains 0.3 to
  0.8 (single seeds -3.14, -2.85, -2.66 from doc_metric_data/beam_bins40.npz lp_seeds; ensemble -2.35).
- Source of the 40-band single-seed values (was attributed to the metrics json, which does not hold them) and of the
  B12-whitened interval scores 0.534/0.574 (recomputed from doc_metric_data/beam_b12_{merlin,pca}.npz).
- "alpha_z^P0 and alpha_m^P0 are the best-measured parameters for every compression": false for 40 bins (alpha_m^P0 is
  4th there, r 0.659). Rewritten. Per-parameter r gains corrected: alpha_m^P0 +0.185, alpha_m^xc +0.173 (were inside
  "+0.19 to +0.23").
- Validation: "agreement within 0.04 nats over every ensemble" -> mean -0.04, single ensembles -0.10 ... +0.03;
  "statistically identical" -> consistent. "lost batches formed a contiguous block": they are 61 runs of consecutive
  rows (row_id.npy); cause of the P0 non-uniformity marked as not investigated.
- P0 shift on other skies: "about 12 to about 50" -> 45-54 (other lightcones) and 24-46 (other masks), per s4 lede.
- Last-step gain "0.95 to 2.1 nats" clarified as per doubling (32,000 -> 61,440 is a factor 1.92).
- Lost-row cause: 274 batches cited to final_dataset.py and data/final_status/missing_batches.json; the disk-quota
  cause is only in the project notes (memory), now labelled as such.
- Noise-only whitening "at least as good" restricted to Merlin; PCA change within errors stated.
- TARP/SBC paragraph: Merlin "calibrated jointly and per parameter" -> jointly calibrated, per parameter close
  (P0, alpha_m^P0 coverage 0.643/0.649; 3 SBC curves just outside).
- "posterior widths along the tight Fisher directions barely change with either" quantified from checks.json
  (0.19-0.32 -> 0.12-0.29 with N; 0.12-0.29 for MAF 256x8 / NSF); "the simulations are the limit" labelled a working
  conclusion.
- Validation open-issue bullet: evaluated with all five results_final variants (four 40-band + MOPED), not only two.

## 4. Physics / statistics

Checked: B12 gNFW form and P200c, Pe = 0.5176 Pth, y prefactor; logit map, Jacobian dtheta/du = w s(u) s(-u), flat
prior -sum log w = -12.90; NPE/KL statement; log p as proper score; Pearson r and RMSE definitions (prior-centre RMSE
w/sqrt(12)); width normalisation 0.6827 w; binomial +-0.017 (68 %) and +-0.008 (95 %) for 3,000 rows; TARP definition
against sbi 0.27 _run_tarp/get_tarp_references (z-scored to truth range, num_bins = 3000 // 10 = 300, ECP cumsum);
KS band 1.358/sqrt(3000) = 0.0248; SBC rank interpretation (hump below-then-above = too wide) matches the figure and
the >0.68 coverage of PCA/40 bins; interval score and flat value 0.84; ensemble logsumexp - log 3; saturating fit
L - A (N/1e4)^-alpha; boxed-Fisher expected log p formula; beam-correction algebra
C_beam = C_ss + B C_sn + B^2 C_nn (beamfix.correct). Added definitions: Deproj-0/ILC, NSIDE, B_ell and the C^ss,
C^sn, C^nn, C^noisy notation, Richardson derivatives.

## 5. XGPaint

git -C /home/kn18001/.julia/dev/XGPaint: tag validated-tsz256-20260922 (annotated, object bb9f74e) -> 35f79ad, tree
3b57cacb = tree of ce1d239; HEAD 1660e08 differs only in test/runtests.jl. f16a7f3 (2026-02-09) adds the keyword
constructor (15 keywords) and keeps upstream's beta = gamma - alpha*beta_raw. 9f37b1e (Adrian Bayer, 2026-02-27, on
upstream/adrian) changes generalized_nfw exponent (beta-gamma)/alpha -> -(beta+gamma)/alpha AND the conversion to
beta = -gamma - alpha*beta; merged into the fork by b924ab1 (2026-03-02, "addition of Adrian fixes"), giving the
outer factor (1+x/xc)^(+beta_raw). e67a8a1 (2026-04-30) sets beta = alpha*beta_raw - gamma (old line 81 confirmed in
the diff) and adds prepare/evaluate_profile_slice (profiles.jl L56-L66 incl. fill_profile_slice!) and the tSZ
prepare_profile_slice amplitude (profiles_y.jl L147). 35f79ad restores N_logθ, no numerical change. The aside now
names the b924ab1/9f37b1e origin of the wrong conversion (the author's text implied it was the f16a7f3 state) and
notes that the project wrapper ChordMeanProfile calls XGPaint.get_params (L125) and the fork's tSZ
prepare_profile_slice (projection_amplitude, L104), so it depends on e67a8a1. Not-XGPaint items (4 R200c chord
wrapper, painter, mask, noise, beam, beamfix, SBI code) confirmed as project code.

## 6. HTML

One <section id="sbi64k" class="topic">; all tags balanced (html.parser check); classes only from the contract (plus
td.num, span.src); no script/style; no em dashes; 12 excerpts each followed by p.code-note. Prose about 8,700 words
including captions (above the 3,000-7,000 guide; not cut, since nothing is redundant enough to drop safely).

## Remaining concerns

- Fixes are not in scripts/sbi64k/s3_template.html (see top).
- The cause of the non-uniform P0 of the 1,096 validation rows is unexplained.
- The EDQUOT (disk quota) cause of the lost batches is sourced only from the project memory notes, not from a log file
  in the repository.
- Fig 3.10 and the correlation table use different draw files (300,000 vs 30,000) of the same networks; correlations
  agree to the sampling error but are not from the identical draws.
