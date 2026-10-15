# Verification notes: Section 4 (s4_seedmask.html), adversarial check 2026-10-07

Checker scripts (temporary, scratchpad): s4chk/code.py (excerpt vs file/git), s4chk/htmlcheck.py.

## 1. Code excerpts (12)
All 12 `<pre class="code">` blocks match the current files verbatim (HTML-unescaped), including the two
XGPaint excerpts read with `git show 29ec48a:src/profiles_y.jl` and `git show validated-tsz256-20260922:src/profiles_y.jl`.
Code-note line references re-counted; fixed: extended_controls nn term "line 50" -> "lines 48-50";
baseline_pearson rebuild "79-84" -> "79-85". Added the missing code-note after the upstream get_params excerpt
(12 pre, 12 code-notes now).

## 2. Figures (16)
All 16 PNGs exist, 1800 px wide, < 0.3 MB, aspect equal to source; every PDF is byte-identical to its source.
Each image was viewed. Caption fixes:
- 4.2: right panel is median + 16-84% band over all 100 cells (report.py:237-240), not "mean and spread".
- 4.3: error bars are +/- 1 SD across observations (variation_summary.json width_ratio_sd), not the range.
- 4.6: training cell is NOT "the only cell close to the truth"; s100_m12353 (23.8) is as close. Reworded with
  posterior_intervals.csv numbers (next lowest 23.8, 26.0; others 29.8-53.8).
- 4.7: band50 range for the inputs shown is 32.65-33.24 (was 32.3-33.2).
- 4.11: next largest |dr| is 0.0404 (beta-alpha_m,P0), so "0.04 or less" -> "at most 0.041".
- 4.12: combined original sqrt(T^2+A^2) = 1.695 (was 1.694), mask 1.492; recipes span 0.99-1.26 at widths 1.18-1.41
  (summary.json subsets.*.held_out, the data the plotting code reads).

## 3. Numbers traced (> 60)
Confirmed (source): P0 12.10 (5.64-26.48), 45.50-53.66, 23.70-45.92, max shifts 3.99/3.25, all 28 table cells
(parameter_comparison.csv, variation_summary.json, RESULTS.txt); 1.6e-15/1.8e-15/0 and 0.29-0.30, 2.8e-4
(baseline_check.json, validation.json); halo counts 85,215,792-85,267,919 (diagnostics.json); 49.89+-2.32,
13.01+-0.21, 12.15/9.70 (seed_offset_summary.csv); -6.14/-0.063/+0.742 %, seed 106 +15.8 % (diagnostics.json);
whole counterfactual table, sigma 15/50/100 values, blocks 16.37/23.03/20.56 (counterfactual_posteriors.csv);
mask table, nn 8.30-13.52, clean 26.12-44.34, per-seed 12.13-15.46 / 33.89-42.39, absolute smooth 12.59
(extended_control_posteriors.csv); singular-value ratio 1.17e-5 (diagnostics.json, = condition ~85,000 p26);
held-out T/A/width/worst, full-grid 1.620->1.150, 0.444->0.092, 1.215 (evidence/study/summary.json, p17);
variance fractions 26.3/31.8/41.9 and 12.4/73.1/14.4 (recomputed from median_variance_parts);
joint truth 1/100, 2/100, band50 0.135-0.294, training 0.073 (joint_truth_linear.json); Pearson table, bounded
and inside-prior values -0.361/-0.292/-0.336 (pearson_pairs.csv, pearson_linear.json); smoothing -15.3 % @140,
-9.3 % @232, +0.76 % (recomputed from data/baseline_spectra.npz); 580 cut numbers and full-prior widths 1.456/1.464/1.565,
alpha_m,P0 2.35/2.38/2.54, P0 1.71/1.78/1.76, coverage 63.2-66.8 / 68.3-73.0 (truth_mean_metrics.csv, README);
log density -3.013+-0.136 -> -5.481+-0.097 (joint report p21); 12 jobs (EXTENSIONS.md, jobs files).
Corrected:
- Beam-approximation comparison: 0.29-0.30 is a norm in the FULL whitened coordinates (analyze.py:324), but was
  compared with retained-PCA norms 21.6-39.7. Now compared like-for-like: full whitened clean differences 46.7-57.2
  (lightcones), 33.2-73.2 (masks) from diagnostics.json spectral_metrics.full_noiseonly_clean_distance.
- "every counterfactual input stays within 32.3-33.2": false; all 104 band50 inputs give 32.12-34.20.
- 50 %-budget MOPED width: 37-40 % is INTERPRETATION (full set); page 20 says 35-39 % (held-out). Both now cited.
- Coverage in 4.8 (63.3-66.9 %) disagreed with 4.7 (63.2-66.8 %): both are real (5 Oct final test vs 6 Oct
  9,000-draw validation on the same rows); both now given with provenance.
- Lede: "P0 from about 12 to about 50" true only for lightcones; masks give 24-46. Fixed.
- Seed 120 exclusion source added (joint report page 7); network seeds 1,2,3 source added (plan.json).

## 4. Physics / statistics
- gNFW equation: B12 fits THERMAL pressure; XGPaint uses P_e = 0.5176 P_th (profiles_y.jl:128, amplitude :158 at
  the tag) with P200 = 200 rho_crit f_b G M / (2 R200), f_b = Omega_b/Omega_m. The section had labelled the B12 form
  P_e/P200 (copied from REPORT.md). Fixed; 0.5176 = 2(X_H+1)/(5X_H+3), X_H = 0.76, explained; cancels between skies.
- d ln P_e derivative re-derived: correct. Delta_i covariance statement correct. MOPED unit covariance and
  J^T C^-1 J preservation re-derived: correct. HPD/Jacobian logic correct. T, A, G formulas match report.py.
- Taper code-note was wrong ("multiplies the covariance by a factor rising to sqrt 10"): the covariance element
  (i,j) is multiplied by s_i s_j, so variance up to ~10x, s.d. up to ~sqrt 10. Fixed.
- Added definitions: joint calibration screen (random-reference / TARP ranks, tolerances 0.10/0.06, even rows),
  HPD level at first use, mean log density at truth with row SEM. Band50 change list now includes the refitted asinh.
- Ruled-out bullet: diagnose.py:99-102 does not assert the mass cut; now stated as recorded in contracts instead.
- Counterfactual note: "uses the known truth sky as its reference" reworded (it uses the training-sky clean
  spectrum at the known B12 truth).

## 5. XGPaint
Confirmed: tag validated-tsz256-20260922 = 35f79ad (tree = ce1d239 per message); upstream/main = 29ec48a;
HEAD 1660e08 differs from tag only in test/runtests.jl; f16a7f3 (2026-02-09) parametrisation with B12 defaults and
upstream beta convention; e67a8a1 (2026-04-30) beta = alpha*beta_raw - gamma and the prepare_profile_slice /
evaluate_profile_slice hooks; generalized_nfw exponent at tag line 65; Manifest pin lines 1976-1982; ChordMeanProfile
is project code (engine/spherical_truncation_profiles.jl), balanced_painter.jl:20-22, :30 correct.
Corrected history of the wrong-sign state: it was introduced by Adrian Bayer's 9f37b1e ("fixing ksz", 2026-02-27,
upstream WebSky-CITA `adrian` branch), which changed BOTH the exponent to -(beta+gamma)/alpha AND the tSZ conversion
to beta = -gamma - alpha*beta_raw (the section said the conversion was "kept"). It entered the fork through the
merge b924ab1 (2026-03-02, parents b459728 + 077b849). 71bc3fd ("Adrian fixes", 2026-03-03) is a side-branch commit,
NOT an ancestor of b924ab1 or of the tag (the section said b924ab1 added 71bc3fd). All verified with
git log --graph / merge-base / git show <rev>:src/profiles_y.jl.

## 6-7. HTML and clarity
One section#seedmask.topic, all tags closed, only contract classes, no script/style, no em-dashes.
Running prose (without code, tables, captions, path notes) ~6,900 words.

## Open / not fixed here
- scripts/seedmask/build_section.py regenerates the fragment from s4_template.html; the template does NOT contain
  these fixes (outside my write scope). Re-running it would revert them.
- Whether any pre-tag product (between b924ab1 and e67a8a1, i.e. 2026-03-02 to 2026-04-30) was painted with the
  wrong-sign tSZ profile is outside this section; Sections 1/3/5 should check.

---------------------------------------------------------------------------------------------------------------
# Second adversarial pass (re-run requested, 2026-10-07)

Independent re-check of the fragment as left by the first pass (file dated 14:07). Scratch scripts:
scratchpad s4chk2/code.py (excerpt vs current file / git revision), s4chk2/htmlchk.py (structure, classes, prose count).

## 1. Excerpts: 12/12 verbatim (HTML-unescaped), each 8-37 lines, each followed by a code-note. Every line number
quoted in the code-notes re-counted against the file (paint_seed 217/223/232-237/240-243/157/160-162/165/170/174/176/178-182,
upstream get_params 45-47/50 and generalized_nfw :34, tag :65/:128/:158, report.py 44/47/50/51, diagnose 126/141-145/95-96/99-102,
extended_controls 48-52/54, baseline_pearson 79-89/55/59-60, pipeline 154/158-162/164-168, joint_truth 46/48/49,
balanced_painter 20-22/30, engine.jl 60/73, base_command.json 11/32-36, Manifest 1976-1982, analyze.py 144-148/324,
report.py 89-100/237-240, common.py 48-54). All correct.

## 2. Figures: 16/16 exist, 1800 px wide, 0.10-0.27 MB, aspect = source, PDFs byte-identical to source PDFs; all viewed.
Plot content matches captions (checked against plotting code: diagnose.py:327-338 case list for 4.7, make_report.py:55-75
for 4.8, extended_controls.py:118-133 for 4.9, report_figures.py:61-77 for 4.12, report.py:228-245 for 4.2).
Caption fix: 4.8 "returns ... to the training-sky values" -> "close to ..., well within their 68% intervals" (alpha_m,P0
smooth medians 0.02-0.14 vs dotted -0.035). 4.15 "former 1.25 cap" now attributed to the 6 Oct report's own wording.

## 3. Numbers: > 90 re-traced (all values in the first-pass list re-derived from the files, plus): variation_summary.json
(all 28 table cells, 3.99/3.25, 0.07-0.33), parameter_comparison.csv (12.10, 5.64-26.48, 45.50-53.66, 23.70-45.92, 49.88),
validation.json (0.2904-0.3012, max 2.75e-4), diagnostics.json (46.70-57.16, 33.16-73.15, PCA 21.59-39.73, blocks
-6.137/-0.0626/+0.742 %, seed 106 +15.8 %, counts, SV ratio 1.17e-5, 8.9e-16), counterfactual_posteriors.csv (whole table,
Fig 4.7 band50 32.65-33.24 for the ten plotted cases, 104-input range 32.12-34.20, HPD 0.0723), extended_control_posteriors.csv,
posterior_intervals.csv (grid ranges), summary.json (held-out T/A/G/worst, hypot 1.753/1.492/1.695, recipe span 0.986-1.26 at
1.179-1.406, per-parameter widths, full grid 1.620/1.150/0.444/0.092/1.215, variance fractions recomputed 26.27/31.83/41.90
and 12.43/73.14/14.43), joint_truth_linear.json (1/100, 2/100, 0.1352-0.2938, 0.0732), pearson_pairs.csv / pearson_linear.json
(table, MCSE, -0.0404 next largest, medians, outside-prior, HPD 0.305/0.089, inside-prior -0.361/-0.292/-0.336, widths
1.0122/0.9987), baseline_spectra.npz (-15.28 % @140, -9.34 % @232, +0.763/+0.759 %), truth-mean README + truth_mean_metrics.csv,
plan.json (17 + 6 = 23 recipes, 61,440 / 64,440), design_manifest.json (bounds, Sobol seed 20260920, noise rule).
Corrected:
- Support percentile: new observations lie at 0.22-0.312 of the 600-reserved-row reference (support.grid_percentile);
  "22nd-30th" -> "22nd-31st" (reports round to 30), and the metric is now described (nearest-neighbour distance to the
  training pool in the standardised inputs).
- Bounded-model control values 11.67 / 44.50 are the sigma_ell = 50 cases (make_report.py:136); sigma now stated.
- Table 4.2 "noise of the 64,440 training rows" -> 61,440 training + 3,000 reserved test rows (plan.json).
- Redshift selection: "all catalogue redshifts" -> "all finite, non-negative ... (cache z = 0.001-5)" (diagnostics.json contracts).
- 50 % budget winners: mask arm is the weak lambda = 0.1 variant (selection.json width_budget_winners); now stated.

## 4. Physics / statistics
Re-derived: 0.5176 = 2(1+X_H)/(5X_H+3); P200 x R200 = G M 200 rho_c f_b / 2 matches XGPaint P_th_los; cap fraction
(1-cos r)/2; d ln P_e; Delta_i covariance; MOPED unit covariance and J^T C^-1 J; HPD Jacobian; taper s^2 <= 9.89 at the top
band; T/A/G vs report.py; TARP ECP description; calibration tolerances (final_report.py:44-46: every marginal and the joint
68/95 ranks, 0.10/0.06). Unsupported or wrong causal claims fixed:
- 4.5 claimed "the shifted solutions lie along the posterior's own degeneracy directions". Contradicted by the data: training-sky
  posterior has P0-beta -0.34 and P0-xc -0.65, yet the other skies move P0, beta and xc all upwards; and the posterior shape
  itself changes on new skies (mean over nine lightcones: P0-xc -0.390, P0-beta -0.040, xc-beta 0.825 vs -0.645/-0.336/0.813;
  posterior_correlations.json, 10,000 joint draws each, original_merlin final draws). Paragraph rewritten.
- 4.6 framed the Pearson test as "does the shape depend on the fine fingerprint" and concluded "the fine pattern moves the
  posterior along its existing degeneracy valley without reshaping it". The test only adds the SMOOTH part (fine pattern kept),
  so it cannot test the fine part; and the real skies do show reshaping (above). Question and conclusion rewritten.
- Aside 4.5: "displacement comes from the xc-beta degeneracy and prior volume" overstated; joint report p26 lists degeneracy,
  prior volume and density-estimator error with unseparated contributions. Reworded.
- "This is why cutting low or high multipoles cannot cure the effect" softened to what the block controls show (blocks do not
  add up; no cut recipe was selected in 4.7).
- "The fine pattern ... is averaged out inside each band" now notes band50 also changes whitening and training.
- Selection rule "fixed before any held-out sky was examined" conflicted with the stated partial non-blindness; now "evaluated
  on discovery skies only; no held-out sky used for tuning". Rule restated as min sqrt(T^2+A^2) subject to the constraints.

## 5. XGPaint: all hashes, dates, authors, parents and ancestry re-verified (f16a7f3 2026-02-09 defaults = B12; 9f37b1e
2026-02-27 adrianbayer, both exponent and conversion changed; 077b849 child of 9f37b1e; b924ab1 2026-03-02 parents
b459728 + 077b849; 71bc3fd 2026-03-03 not an ancestor of the tag; e67a8a1 2026-04-30 adds the slice hooks (profiles.jl)
and beta = alpha*beta_raw - gamma; tag = 35f79ad, message tree = ce1d239 (3b57cacb = Manifest git-tree-sha1); HEAD 1660e08
differs only in test/runtests.jl). paint.pbs of both studies runs the 64k bundle julia_env, so the pin applies. The sphere
wrapper takes its amplitude from XGPaint prepare_profile_slice (spherical_truncation_profiles.jl:102-105), so 0.5176 is applied.
No change needed except clearer wording of the history sentence.

## 6. HTML: one section#seedmask.topic, all tags closed, only contract classes, https links only, no script/style, no
em/en dashes. Prose (excluding code, tables, captions, path notes) ~7,500 words after the clarifications.

## 7. Clarity additions: B12 defined in the lede; Sobol sequence; deproj0 (SO LAT baseline ILC tSZ noise, no deprojection,
column 2); BatchNorm; MOPED introduced before the recipe table; grid index ordering in the diagnose code-note; remedy needs
repainting but no new N-body runs.

## Still open (unchanged)
- scripts/seedmask/build_section.py + s4_template.html would regenerate the fragment WITHOUT either pass's fixes (outside the
  allowed write scope). Do not re-run it.
- Whether pre-tag products (2026-03-02 to 2026-04-30) used the wrong-sign tSZ profile is for Sections 1/3/5/6.
