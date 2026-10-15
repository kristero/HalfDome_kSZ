# Mass binning and sample size for 1D stacked profiles (FLAMINGO L1_m9, z = 0)

All numbers come from `groups/mass_binning/results.json`; the field is named in brackets where it helps. Derived numbers (ranges with their field and bins, sums of per-bin contributions, pulls and significances, baseline conversions) are recomputed from results.json by `text_checks.py` and `text_checks_v3.py` and saved in `work/text_checks.json` and `work/text_checks_v3.json`; the last verification's recomputations are in `verify/v4_claims.py` and `verify/v4_spot.py` (outputs `verify/v4_*_out.txt`). Impacts are first-order changes, in per cent, propagated with `sc.impact_from_stacks` on the 10 calibrated bins (13.5-15.5) x 35 fit shells. Errors are paired-bootstrap 1 sigma unless stated otherwise. Radial values are band values: the median over a band's resolved fit shells. P is electron pressure, E (n_e) electron density.

## Bottom line

| Choice | Measured effect on what HalfDome would be painted with | Recommendation |
|---|---|---|
| Bin width in M200c | **Continuous painting, relative to the 0.2-dex reference painted the same way (metric C, linear interpolation):** <br>- 0.1 dex changes no observable by more than +0.52±0.09 % (kSZ, ell = 3000); its significant changes remove most of the reference's linear-interpolation chord error.<br>- 0.3 dex: tSZ -0.61 / -0.71 / -0.67 % (±0.90 / 0.88 / 0.61); kSZ (ell = 3000) -0.80±0.18 %; DM variance -0.51±0.16 %.<br>- 0.5 dex: tSZ -3.01 / -2.98 / -3.13 % (±1.43 / 1.50 / 0.90); kSZ (ell = 3000) -3.25±0.18 %; DM mean -0.62±0.08 %; DM variance -2.39±0.19 %.<br>**Interpolant (metric B, vs the truth):** with ln F linear in log M, even the 0.2-dex nodes paint kSZ (ell = 3000) 0.71±0.10 % low. With PCHIP the offset is -0.17±0.10 %. | Use 0.2-dex M200c bins and a curvature-aware mass interpolant (PCHIP in log M); a smooth parametric mass model is an untested alternative. 0.3 dex is acceptable if needed: with PCHIP its central values stay within 0.64 % of the truth on every observable (errors up to ±0.82 %). Avoid 0.5 dex: even with PCHIP it loses 1.46±0.19 % on kSZ (ell = 3000) and 0.79±0.19 % on DM variance, and its top node must be extrapolated to the haloes at 15.3-15.5. Its PCHIP tSZ loss (-2.22±1.35 / -1.60±1.32 / -1.19±0.78 %) is only 1.2-1.6 sigma. Place nodes at the inclusion-weighted mean log M200c. |
| Normalisation | Bin-mean P200 and R200c instead of each halo's own:<br>- 0.2 dex: tSZ +2.95±1.96 / +3.02±1.15 / +3.12±0.80 % (1.5 / 2.6 / 3.9 sigma), y +1.71 %, kSZ +1.56 / +1.22 / +0.76 %, DM mean +0.66 %, DM variance +0.98 %.<br>- 0.5 dex: tSZ +18.5 / +18.5 / +18.9 %. | Normalise each halo by its own P200 (T200, K200) and R200c. |
| Binning variable | Relative to the reference painted the same way:<br>- 0.2-dex M500c bins: tSZ +1.61±0.41 / +3.79±0.64 / +5.07±0.51 %.<br>- 0.5-dex M500c bins (B24-like): tSZ +1.88±1.11 / +2.38±1.25 / +2.53±1.16 % (1.7-2.2 sigma), kSZ -1.93 to -3.13 % (±0.25-0.32), DM mean -0.98±0.15 %.<br>- Lee22 h^-1 bins: all observables within 0.15 % (noise up to 1.4 %). | Bin in M200c, the mass HalfDome carries. Msun or h^-1 units make no difference once nodes sit at their mean masses. Avoid M500c. |
| Haloes per bin | Observable error from N haloes per calibrated bin is about 9.3 %/sqrt(N) for tSZ (ell = 1000) and 4.9-5.1 %/sqrt(N) for mean DM (N = 10-25). Outer-pressure band stacks of small subsamples have a strongly right-skewed sampling distribution. At 3-4 R200c the mean over 400 subsamples stays close to the full-bin value (-4.6 to +1.3 % at N = 10, standard error 2.1-4.5 %), but the median subsample is 14.0-19.9 % low at N = 10 and 4.0-11.5 % low at N = 25. A typical small sample therefore underestimates the outskirts. | At least 100 haloes per 0.2-dex bin; the current 317-327 per sampled calibrated bin (13.5-14.5) is better. Use complete census where bins are sparse. |
| Sparse top bins (21 + 6 haloes) | Carry 29.8 / 16.6 / 6.2 % of the total tSZ response.<br>Reference tSZ noise (census bootstrap): 1.77 / 1.99 / 1.24 % (all calibrated bins); 1.75 / 1.95 / 1.19 % from the census bins 14.5-15.5 (jackknife 1.81 / 1.95 / 1.14 %); 1.54 / 1.76 / 0.89 % from the top two alone.<br>Continuous painting relative to the reference:<br>- pooling the top two into one node: -0.85±0.95 / -1.30±0.70 / -0.48±0.25 %;<br>- dropping them: -4.08±2.00 % (last segment extrapolated) or -2.16±1.36 % (14.9-15.1 held flat) at ell = 100, 1.6-2.0 sigma;<br>- pooling everything from 14.9 upwards: -2.92±1.35 / -1.98±1.41 / -0.74±0.67 %. | Keep the census top bins as nodes, or pool the two into one node at its mean mass. Do not drop them, and do not paint them with the 14.9-15.1 or the 14.9-15.5 stack: these options lower tSZ by 2.0-4.1 % at ell <= 1000. Each is only 1.2-2.2 sigma, but all are low at every ell. Quote about 1.8-2.0 % (ell <= 1000) and 1.1-1.2 % (ell = 3000) tSZ calibration noise from the finite census. |

## 1. How the variants were compared

**Reference.** Key `sc.make_key(field, grid='fit')`; T_e and K use estimator 'weighted', the only one cached for them. Mean statistic, strict rule (min_count 20, softening_factor 4, hsml_factor 2, min_edge 0.04), M200c_0.2 bins, all haloes, n_boot 200. The 'ref' web records come from `sc.stack_bin` with the core's default seeds. The 75 ref records are identical to those of groups estimators, selection, resolution and grids_scaling (checked).

**Paired bootstrap across different bin sets.** Schemes, sub-bins and pooled bins contain different haloes, so all comparisons use one global bootstrap: 200 draws, seed 20261012 + 4242. A bin's replicate stack is the inclusion-weighted mean over its haloes with the global multiplicities.
- *Census design (new; used for every error in results.json).* The sampled bins 12.5-14.5 are resampled within mass_bin x class_main strata. The complete-census bins 14.5-15.5 (all weights 1) are resampled plainly within the 0.2-dex bin.
- *Why the core's convention is not used.* `sc.stack_bin(stratified=True)` also stratifies the census bins by class. There the strata hold 17/2/2 haloes (15.1-15.3) and 2/3/1 (15.3-15.5), and a one-halo stratum never varies.
- *Consistency check* (`analyses.bootstrap_consistency_summary`). With the core's strata, the global bootstrap reproduces `sc.stack_bin`'s per-bin errors to a ratio of 0.88-1.10 at 0.5-1 R200c (all 15 bins, 5 fields). Over all bands it is 0.82-1.22 (median 0.97 sampled, 1.01 census), consistent with the Monte Carlo noise of a 200-draw SD.
- *Effect of the census design.* At 0.5-1 R200c, census-design errors are larger by factors of 1.62 (P) and 1.93 (E) in 15.3-15.5, and 1.12 / 1.08 in 15.1-15.3.
- *Web records.* They keep the core convention, so the err of the top-bin web records (ref included) understates the sampling error (see issues).

**Band ratios.** For each band (<0.1, 0.1-0.25, 0.25-0.5, 0.5-1, 1-1.5, 1.5-2, 2-3, 3-4 R200c) the band value is the median, over the band's resolved fit shells, of variant / comparator. Its error is the SD of the same statistic over the paired replicates.

**Comparators.**
- Bin schemes: the reference interpolated to the bin's inclusion-weighted mean log M200c (ln F linear in log M).
- Normalisation variants: the own-normalised stack of the same haloes.
- Subsamples: the full bin.
- Top-bin treatments: the measured top-bin stacks.

**Observable metrics for bin schemes** (`analyses.scheme_impacts_*`).
- **A (at the reference's node masses).** The scheme's stacks are interpolated to the calibrated bins' mean masses and compared with the reference stacks. The reference is exact there by construction (closure check: |impact| <= 2e-15). This metric therefore charges every other scheme with node-placement error. It was the only metric in the previous version.
- **B (continuous painting vs truth).** Every halo of the 10 calibrated bins is painted with the scheme's node stacks interpolated to its own log M200c. The painted bin mean is compared with the true bin mean, i.e. the mean of the haloes' own normalised profiles (the reference stack). No scheme is exact by construction.
- **C.** Metric B of the scheme relative to metric B of the 0.2-dex reference, paired: the cost of the scheme when both are painted the same way. This is the `impacts` entry of each scheme variant in results.json.

Interpolants (`analyses.interpolant_test`): ln F linear in log M200c, or monotone cubic (PCHIP). Both extrapolate linearly beyond the end nodes.

**Caveat on the impact propagation.** The first-order Jacobians apply each painted bin-mean change as a uniform rescaling of every halo in that bin. Continuous painting redistributes the change within a bin (with a concave mass trend, haloes between nodes are painted low), and the quadratic observables (tSZ, kSZ, DM variance) weight the massive haloes of a bin more heavily. Metrics B and C therefore carry an extra second-order uncertainty that is not in the quoted errors.

## 2. Choice: bin width in M200c (0.1 / 0.2 / 0.3 / 0.5 dex)

**What it is.** One mean profile is assigned per mass interval, and the painter interpolates between the bins' nodes. Wider bins give fewer, more widely spaced nodes. Within a bin they also mix haloes whose normalised profiles differ.

**Measured effect.** Per cent, ± paired-bootstrap SD.

Metric C: continuous painting with ln F linear in log M, relative to the 0.2-dex reference painted the same way.

| Width | tSZ ell=100 | tSZ 1000 | tSZ 3000 | mean y | kSZ 100 | kSZ 1000 | kSZ 3000 | DM mean | DM var |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 dex | +0.05±0.19 | +0.01±0.24 | +0.32±0.18 | +0.19±0.04 | +0.29±0.05 | +0.41±0.06 | +0.52±0.09 | +0.12±0.03 | +0.41±0.08 |
| 0.3 dex | -0.61±0.90 | -0.71±0.88 | -0.67±0.61 | +0.10±0.23 | +0.08±0.19 | -0.26±0.19 | -0.80±0.18 | +0.11±0.13 | -0.51±0.16 |
| 0.5 dex | -3.01±1.43 | -2.98±1.50 | -3.13±0.90 | -0.99±0.15 | -1.34±0.27 | -2.26±0.17 | -3.25±0.18 | -0.62±0.08 | -2.39±0.19 |

Metric B: continuous painting vs the true bin means, linear and PCHIP interpolation.

| Width | interp | tSZ 100 | tSZ 1000 | tSZ 3000 | mean y | kSZ 1000 | kSZ 3000 | DM mean | DM var |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | linear | +0.05±0.39 | +0.02±0.42 | -0.08±0.22 | -0.02±0.06 | -0.09±0.05 | -0.19±0.05 | -0.00±0.03 | -0.12±0.05 |
| 0.2 (ref) | linear | +0.02±0.27 | +0.03±0.37 | -0.38±0.23 | -0.20±0.06 | -0.49±0.06 | -0.71±0.10 | -0.12±0.04 | -0.52±0.08 |
| 0.3 | linear | -0.56±0.90 | -0.64±0.85 | -1.02±0.63 | -0.09±0.28 | -0.74±0.21 | -1.50±0.20 | -0.01±0.16 | -1.03±0.16 |
| 0.5 | linear | -2.97±1.39 | -2.93±1.35 | -3.49±0.84 | -1.18±0.15 | -2.73±0.17 | -3.93±0.19 | -0.73±0.08 | -2.90±0.18 |
| 0.1 | PCHIP | -0.06±0.47 | -0.11±0.50 | -0.08±0.26 | +0.01±0.05 | -0.01±0.04 | -0.06±0.05 | +0.01±0.03 | -0.02±0.04 |
| 0.2 (ref) | PCHIP | -0.02±0.28 | -0.04±0.31 | -0.14±0.18 | -0.03±0.04 | -0.10±0.05 | -0.17±0.10 | -0.01±0.03 | -0.10±0.08 |
| 0.3 | PCHIP | -0.64±0.82 | -0.45±0.72 | -0.34±0.49 | +0.31±0.29 | +0.09±0.22 | -0.43±0.20 | +0.24±0.16 | -0.17±0.17 |
| 0.5 | PCHIP | -2.22±1.35 | -1.60±1.32 | -1.19±0.78 | -0.27±0.14 | -0.69±0.17 | -1.46±0.19 | -0.13±0.07 | -0.79±0.19 |

For comparison, metric A (the old metric) gives:
- 0.3 dex: tSZ -0.73±0.86 / -0.71±0.83 / -1.03±0.60, kSZ (ell = 3000) -1.40±0.20, DM variance -0.99±0.17;
- 0.5 dex: tSZ -2.92 / -2.82 / -3.25, kSZ (ell = 3000) -3.81±0.19, DM mean -0.73±0.08, DM variance -2.77±0.19;
- 0.1 dex: |impact| <= 0.09.

Metric A overstated the 0.3-dex cost on kSZ (ell = 3000) and DM variance by factors of 1.76 and 1.94 relative to metric C. The Lee22 control in section 4 shows why: its nodes are only shifted, yet metric A still charges it -0.85±0.20 % on kSZ at ell = 3000.

**What the continuous metric shows.**
- *The error is a chord error.* Linear segments in log M sit below a concave mass trend, and the painting error grows roughly as the node spacing squared. On kSZ (ell = 3000) against the truth it is -0.19 / -0.71 / -1.50 / -3.93 % for 0.1 / 0.2 / 0.3 / 0.5 dex.
- *PCHIP removes most of it.* A curvature-aware interpolant gives -0.06 / -0.17 / -0.43 / -1.46 % on the same observable.
- *0.5 dex still fails with PCHIP, but on kSZ and DM variance, not clearly on tSZ.* Against the truth with PCHIP, 0.5 dex loses 1.46±0.19 % on kSZ (ell = 3000, 7.6 sigma), 0.69±0.17 % on kSZ (ell = 1000) and 0.79±0.19 % on DM variance (4.0-4.3 sigma). Its tSZ loss (-2.22±1.35 / -1.60±1.32 / -1.19±0.78 %) is only 1.2-1.6 sigma, and mean y (-0.27±0.14 %) and DM mean (-0.13±0.07 %) are within 2 sigma. Where the losses come from (per_bin_pct; per-bin errors are not saved):
  - **kSZ and DM variance: 13.5-14.1,** where the mass trend is steep and concave (Fig 3). With linear interpolation (metric C), kSZ (ell = 3000) loses -0.92 / -0.59 / -1.03 % from 13.5-13.7 / 13.7-13.9 / 13.9-14.1, i.e. -2.54 of its -3.25 %. With PCHIP these bins give -0.37 / -0.37 / -0.50 % against the truth (-1.24 of -1.46 %; -0.33 / -0.34 / -0.44 % against the 0.2-dex PCHIP nodes). DM variance with PCHIP: -0.15 / -0.19 / -0.30 % (-0.65 of -0.79 %).
  - **DM mean: a linear-interpolation loss only.** With linear interpolation (metric C) DM mean loses -0.62±0.08 %, -0.52 % of it from 13.5-13.7. With PCHIP the 13.5-13.7 loss (-0.28 %) is partly offset by 13.9-14.3 (+0.13 / +0.09 %), leaving -0.13±0.07 % against the truth (-0.12±0.07 % against the 0.2-dex PCHIP nodes).
  - **tSZ: the 15.3-15.5 bin.** Its 0.5-dex node (15.0-15.5, mean log M 15.10) must be extrapolated to haloes at 15.3-15.5. At ell = 1000 this 6-halo bin contributes -2.19 % of the -2.98 % with linear interpolation (metric C). With PCHIP it still contributes -2.18 % against the truth (-2.13 % against the 0.2-dex PCHIP nodes). The PCHIP total (-1.60 %) is smaller in magnitude than this contribution because 14.9-15.3 add +0.74 % and 13.5-14.9 only -0.16 %.

**The wide-bin stack itself agrees with the fine bins at its mean mass, within noise** (`analyses.wide_vs_fine_summary`).
- *Selection.* The summary uses wide bins whose inclusion-weighted mean log M200c lies in [13.5, 15.5) and that contain at least two non-empty 0.1-dex sub-bins. That excludes 15.3-15.5 in the 0.2-dex scheme, whose only populated sub-bin is 15.3-15.4. The 'all' selection [12.5, 15.5) is also saved.
- *Additivity.* A wide bin's stack equals the summed-inclusion-weight average of its 0.1-dex sub-bin stacks exactly (max deviation 0 for P, E, T_e and K).
- *Same mean mass.* Compared with the 0.1-dex stacks interpolated to the same mean log M200c:
  - 0.5 dex (4 bins, 29 band values): median |dev| 1.18 % (P) and 0.80 % (E), RMS pull 0.75 / 0.92, no |pull| > 3. The maximum deviation, 11.5 % for P, is at 3-4 R200c with an error of 12.4 %.
  - 0.3 dex (7 bins, 50 values): median |dev| 0.80 % / 0.49 %, RMS pull 0.88 / 0.85.
  - 0.2 dex (9 bins, 65 values): median |dev| below 0.05 % (4-digit ratios), RMS pull 0.21 / 0.24.
- *What wide bins really cost.* The cost is the coarse sampling of a curved trend, plus resolved range. The strict rule needs every halo resolved, so the 0.5-dex bin 13.5-14.0 is resolved from 0.321 R200c, while its most massive 0.1-dex sub-bin is resolved from 0.170. For 14.0-14.5 the edges are 0.145 and 0.090.

**Residual mass trend of the normalised profile** (`analyses.mass_trend`; d ln F per 0.1 dex of M200c from neighbouring 0.2-dex bins, %, census-design errors).

| Bin | P 0.25-0.5 | P 0.5-1 | P 1-1.5 | P 2-3 | P 3-4 | E 0.25-0.5 | E 0.5-1 | E 1-1.5 | E 2-3 | E 3-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 13.5-13.7 | – | +8.9±0.4 | +0.7±0.6 | -10.0±1.5 | -13.6±3.4 | – | +12.4±0.4 | +5.0±0.4 | -4.9±0.6 | -6.3±0.8 |
| 13.7-13.9 | +9.9±0.4 | +4.2±0.3 | -3.6±0.6 | -10.5±1.6 | -13.8±4.4 | +14.4±0.5 | +8.4±0.3 | +0.3±0.3 | -6.2±0.6 | -4.9±0.9 |
| 13.9-14.1 | +6.4±0.4 | +1.1±0.3 | -4.4±0.7 | -8.9±1.3 | -9.5±3.0 | +11.0±0.4 | +4.6±0.2 | -1.9±0.3 | -4.7±0.5 | -3.4±0.9 |
| 14.3-14.5 | +2.6±0.3 | -1.6±0.2 | -4.8±0.6 | -5.4±1.1 | -5.5±2.5 | +6.1±0.3 | +0.7±0.1 | -2.7±0.3 | -2.1±0.5 | -0.6±0.7 |
| 14.7-14.9 | +0.2±0.5 | -1.9±0.4 | -4.5±0.8 | -5.7±2.1 | -14.1±2.6 | +2.6±0.3 | -0.2±0.2 | -2.3±0.5 | -1.9±0.8 | -2.4±0.8 |

**Physical reading.**
- *Self-similarity is broken in the inner halo.* At 0.1-0.5 R200c (slopes exist from 13.7 upwards at 0.25-0.5 R200c and from 14.1 upwards at 0.1-0.25 R200c) the normalised profiles rise with mass in every bin from 13.7 to 14.9: P by +0.2 to +9.9 % and n_e by +2.6 to +14.4 % per 0.1 dex. The largest slope is n_e at 0.25-0.5 R200c in 13.7-13.9 (+14.4±0.5 %), where P rises by +9.9±0.4 %. The smallest of these, P +0.2±0.5 % at 0.25-0.5 R200c in 14.7-14.9, is consistent with zero. In 14.9-15.1 only n_e still rises (+3.3±1.1 / +1.2±0.4 % at 0.1-0.25 / 0.25-0.5 R200c; P -0.2±1.5 / -1.0±0.7 %). At 0.5-1 R200c the rise is confined to lower masses: P rises only below 10^14.1 (+8.9 / +4.2 / +1.1 % in 13.5-14.1) and falls slightly above (-0.9 to -1.9 % in 14.1-14.9); n_e rises below 10^14.5 (+12.4 to +0.7 %). The pattern is consistent with feedback removing proportionally more gas from the inner regions of lower-mass haloes; this interpretation is not tested here.
- *Outskirts fall with mass.* At 2-4 R200c in 13.5-15.1, P falls by 4.9-14.1 % and n_e by 0.6-6.3 % per 0.1 dex; several n_e values above 10^14.1 are consistent with zero. In 15.1-15.3 no fall is detected (P +1.4±4.1 / -2.3±5.8 %, n_e +4.0±1.8 / +2.3±3.0 % at 2-3 / 3-4 R200c).
- *Steep and concave below 10^14.3.* At 0.25-1 R200c the curvature (second difference of ln F) is negative in every bin 13.5-14.3, for both P and n_e (-0.65 to -2.94 per dex^2). In 14.3-15.1 the slopes are smaller and the curvature changes sign from bin to bin (-1.46 to +2.62, 7 of 16 values positive); in the sparse 15.1-15.3 bin it reaches -6.59 and +3.73. Curvature errors are not saved. The steep, concave trend below 10^14.3 is why linear interpolation and wide nodes lose kSZ and DM there.
- *Inside a 0.2-dex bin it explains little scatter.* There sd(log M) is 0.053-0.058 dex in 13.5-15.1, and 0.044 and 0.025 in the top two bins. In 13.5-15.1 the trend explains at most 7.7 % (P) and 15.5 % (E) of the per-halo ln-variance, both at 0.5-1 R200c in 13.5-13.7 (`summary_ranges`); in 15.1-15.3 it explains at most 2.5 %.

**Integrated per-halo painting estimate** (`analyses.painting_assignment_estimate`). This is an ESTIMATE with integrated Y and N_e within 4 R200c: no radial shape, no 2-halo term and no velocities. Values are given relative to smooth painting through 0.1-dex nodes (y / DM / tSZ-1h / kSZ):
- smooth painting, 0.2 dex: y -0.14 %, DM -0.14 %, tSZ-1h +0.15 %, kSZ -0.19 %;
- smooth painting, 0.3 dex: +0.18 / +0.17 / +0.35 / +0.47 %;
- smooth painting, 0.5 dex: -0.49 / -0.51 / -3.84 / -0.74 %;
- step painting, 0.2 dex: <= 0.22 % in absolute value;
- step painting, 0.5 dex: y +0.92 %, tSZ-1h +1.06 %.

On the same baseline as metric C (converted to 'vs smooth 0.2 dex'), the estimate does not in general agree with metric C:
- *tSZ.* It reproduces the 0.5-dex tSZ-1h change: -3.98 % vs metric C -3.01±1.43 % (tSZ, ell = 100). At 0.3 dex, y, DM and tSZ-1h agree within 1.6 sigma.
- *Mean y, DM and low-ell kSZ at 0.5 dex.* It underestimates the losses: mean y -0.36 % vs -0.99±0.15 % (4.1 sigma), mean DM -0.37 % vs -0.62±0.08 % (3.2 sigma), low-ell kSZ -0.55 % vs -1.34±0.27 % at ell = 100 (3.0 sigma).
- *Low-ell kSZ at 0.3 dex.* It overestimates the change: +0.66 % vs +0.08±0.19 % (3.1 sigma).
- *Against the truth.* Its linear proxies at 0.5 dex are also too small: mean y -0.47 % vs metric B -1.18±0.15 %, mean DM -0.54 % vs -0.73±0.08 %.

A possible reason, not tested: the 4 R200c integral weights the outskirts, whose mass trend has the opposite sign, instead of each observable's radial weighting. Use metrics B and C, not this estimate.

**Recommendation.**
- Keep 0.2-dex M200c bins and interpolate between nodes with PCHIP in log M. Painted that way, the 0.2-dex nodes' central values are within 0.17 % of the true bin means on every observable here (only kSZ at ell = 1000, -0.10±0.05 %, is beyond 2 sigma). A smooth parametric mass model is an alternative that was not tested.
- 0.1 dex gains only the chord error, which PCHIP already removes, and halves the haloes per node.
- If many snapshots force wider bins, 0.3 dex with PCHIP keeps every central value within 0.64 % of the truth (the largest is tSZ at ell = 100, -0.64±0.82 %; the only change beyond 2 sigma is kSZ at ell = 3000, -0.43±0.20 %).
- Do not use 0.5 dex. Its significant PCHIP losses are kSZ (ell = 3000: -1.46±0.19 %; ell = 1000: -0.69±0.17 %) and DM variance (-0.79±0.19 %), and its top node must be extrapolated to the 15.3-15.5 haloes. Its tSZ loss is only 1.2-1.6 sigma.
- Always place a node at its bin's inclusion-weighted mean log M200c, never at the bin centre.

## 3. Choice: own-halo normalisation vs bin-mean normalisation

**What it is.** There are two ways to normalise:
- *Own (the reference).* Each halo's P_e(r) is divided by its own P200 at r/R200c,h.
- *Bin mean.* Profiles are stacked at fixed physical r/Rbar and divided by P200(Mbar), where Mbar = 10^<log M200c> (inclusion-weighted) and Rbar = R200c(Mbar).

**Method.** Per-halo x200 profiles (67 shells) are moved to Rbar units by monotone PCHIP interpolation of the cumulative profile in ln r.
- *Accuracy test* (`analyses.rescaling_test`, script `run_rebin_test.py`). On analytic gNFW and beta profiles with s = R_h/Rbar = 0.85-1.30 (s ≠ 1), the per-shell error is median 0.13 % (gNFW) and 0.18 % (beta), maximum 1.6 % and 0.80 %. For s = 0.95 and 1.05 the maximum is 1.2 % and 0.73 %. The tested s range covers the bins (s = 0.93-1.09 in 0.2-dex bins, 0.85-1.30 in 0.5-dex bins).
- *Exactness at s = 1.* Every fit-grid edge is also an x200 edge, so s = 1 reproduces the reference exactly (error <= 2e-10).
- *Comparison with the bias.* The median errors (0.13-0.18 %) are far below the measured P and E biases in the table below (+0.4 to +17.0 %). Single shells can carry up to 1.6 % of extra error.

**Measured effect** (`analyses.normalisation_own_vs_binmean`; bin-mean / own - 1, %).

| Bin (scheme) | sd ln s | P 0.25-0.5 | P 0.5-1 | P 1-1.5 | P 2-3 | E 0.5-1 | E 1-1.5 | T_e 0.5-1 | K 0.25-0.5 | K 0.5-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 13.5-13.7 (0.2) | 0.044 | +2.5±1.0 | +2.7±1.1 | +3.1±1.4 | +2.1±1.6 | +0.7±0.4 | +1.0±0.6 | +0.5±0.7 | -0.4±0.5 | -0.2±0.5 |
| 14.1-14.3 (0.2) | 0.044 | +2.1±1.1 | +2.3±1.3 | +1.7±1.5 | +2.1±1.6 | +0.7±0.6 | +0.4±0.7 | +0.5±0.7 | -0.4±0.4 | -0.1±0.3 |
| 14.5-14.7 (0.2) | 0.043 | +1.7±0.6 | +2.2±0.8 | +2.4±0.9 | +2.6±1.0 | +0.6±0.4 | +0.6±0.4 | +0.5±0.4 | -0.1±0.2 | 0.0±0.1 |
| 13.5-14.0 (0.5) | 0.106 | +16.3±1.0 | +17.0±1.0 | +16.4±1.3 | +13.2±1.8 | +5.3±0.4 | +5.2±0.5 | +2.5±0.5 | -2.4±0.5 | -2.0±0.4 |
| 14.0-14.5 (0.5) | 0.104 | +11.6±1.2 | +13.6±1.5 | +15.6±1.9 | +13.9±2.7 | +3.8±0.6 | +3.7±0.8 | +2.5±0.7 | -1.4±0.5 | -0.7±0.4 |
| 14.5-15.0 (0.5) | 0.095 | +8.4±0.9 | +11.9±1.0 | +15.1±1.3 | +12.7±1.6 | +2.8±0.4 | +3.7±0.5 | +2.5±0.4 | -0.1±0.2 | +0.1±0.1 |

**Observable impact.**
- *0.2 dex.* tSZ +2.95±1.96 / +3.02±1.15 / +3.12±0.80 % (1.5 / 2.6 / 3.9 sigma), y +1.71±0.36 %, kSZ +1.56±0.40 / +1.22±0.37 / +0.76±0.27 %, DM mean +0.66±0.16 %, DM variance +0.98±0.29 %.
- *0.5 dex.* Both normalisations are painted smoothly through the 0.5-dex nodes. tSZ +18.5±3.6 / +18.5±2.4 / +18.9±1.7 %, y +9.7±0.6 %, kSZ +7.0 / +6.2 / +4.7 %, DM mean +2.74±0.24 %, DM variance +5.20±0.48 %.

**Why (corrected).** Write u = ln s_h, with weighted mean 0 and variance sigma^2. The other quantities are:
- a, the mass power of the normalising scale in units of s (P200, T200 and K200: 2; rho_c f_b: 0);
- alpha = -dlnF/dlnx and gamma = d^2 lnF/dlnx^2 of the stack;
- beta(x) = dlnF_h/du at fixed x/R200c (= 3 dlnF/dlnM), the residual mass trend inside the bin, from a weighted least-squares fit per shell.

The bin-mean profile of a halo is ln G_h(x') = a u + ln F_h(x'/s). To second order in sigma:

bin-mean / own - 1 = exp(0.5 sigma^2 [(a+alpha)(a+alpha+2 beta) + gamma - 2 dbeta/dlnx]) - 1.

The terms of this formula:
- **(a+alpha)^2** is the pure Jensen (convexity) term.
- **2(a+alpha) beta** is the covariance of the rescaling with the residual mass trend. It can be negative.
- **gamma and dbeta/dlnx** are curvature terms.
- **beta^2 does not appear,** because the own-normalised stack carries the same factor and it cancels.

`analyses.normalisation_own_vs_binmean.jensen_decomposition` gives, per band, the measured value and three predictions: pure Jensen, this second-order form, and a numerical trend model in which each halo is rescaled exactly with ln F_h = c0(x) + beta(x) u_h.

| Bin, field, band | measured | pure Jensen | second order | trend model | a+alpha | beta |
|---|---|---|---|---|---|---|
| 13.5-14.0 P 0.25-0.5 | +16.3±1.0 | +8.7 | +16.2 | +16.5 | 3.85 | +1.34 |
| 13.5-14.0 P 0.5-1 | +17.0±1.0 | +11.1 | +16.6 | +17.1 | 4.33 | +0.75 |
| 13.5-14.0 E 0.25-0.5 | +4.6±0.4 | +0.9 | +4.6 | +4.3 | 1.24 | +1.86 |
| 13.5-14.0 E 0.5-1 | +5.3±0.4 | +1.5 | +5.2 | +5.1 | 1.61 | +1.29 |
| 13.5-14.0 T_e 0.5-1 | +2.5±0.5 | +4.3 | +2.4 | +2.4 | 2.75 | -0.53 |
| 13.5-14.0 K 0.25-0.5 | -2.4±0.5 | +2.0 | -2.5 | -2.3 | 1.86 | -1.75 |
| 13.5-14.0 K 0.5-1 | -2.0±0.4 | +1.7 | -2.1 | -1.9 | 1.73 | -1.39 |
| 14.5-15.0 P 0.5-1 | +11.9±1.0 | +12.1 | +10.7 | +11.5 | 5.04 | -0.25 |
| 14.5-15.0 E 0.5-1 | +2.8±0.4 | +2.8 | +2.7 | +2.5 | 2.45 | -0.01 |

How to read the table:
- *Pure Jensen works only where the residual trend is flat.* It matches where beta ≈ 0, e.g. P and E at 0.5-1 R200c in 14.5-15.0. In 13.5-14.0 it under-predicts P by a factor of 1.5-1.9 and E by 3.6-5.3, and over-predicts T_e (measured / pure = 0.58). Over all 160 band values of the 0.5-dex bins its RMS pull is 3.4 (max |pull| 10.0).
- *The second-order form and the trend model reproduce the bands within errors,* including K's negative bias. Over the 550 band values with errors (0.2- and 0.5-dex bins), the RMS pull is 0.31 / 0.46 (0.2 / 0.5 dex) for the second-order form, with max |pull| 1.9, and 0.44 / 0.61 for the trend model, with max |pull| 2.2.
- *Why K is negative.* Pressure, T_e and K are all scaled by an s^2 amplitude. For K, alpha < 0 (entropy rises outward), so a + alpha = 1.7-1.9 in 13.5-14.0. There K's residual trend is negative (beta_K = -1.4 to -1.8): within the bin, more massive haloes have higher normalised n_e (beta_E = +1.3 to +1.9) and lower normalised T_e (beta_T = -0.5). Since K ∝ T_e n_e^(-2/3), beta_K = beta_T - (2/3) beta_E, which gives -1.75 and -1.39, the fitted values. The negative covariance term then outweighs the positive convexity term.
- *Where pressure is hit hardest.* Pressure has the largest a + alpha (3.85-5.18 at 0.25-1 R200c in the 0.5-dex bins, against 1.24-2.62 for E, 2.37-2.75 for T_e and 0.84-1.86 for K). In 13.5-14.0 its positive inner trend (beta = +1.34 at 0.25-0.5 and +0.75 at 0.5-1 R200c) adds to the bias.
- *How the mean mass is defined matters a little.* Defining Mbar as the arithmetic mean mass instead of 10^<log M> changes P200(Mbar) by a further factor of 1.001-1.006 (0.2 dex) or 1.019-1.038 (0.5 dex) (Pbar_arith_over_geo).

**Recommendation.** Normalise every halo by its own P200 (T200, K200) and R200c before stacking. This is exactly what the painter does with each HalfDome halo. Bin-mean normalisation biases tSZ high by about 3 % even in 0.2-dex bins (2.6 and 3.9 sigma at ell = 1000 and 3000; 1.5 sigma at ell = 100), and by 18.5-18.9 % in 0.5-dex bins.

## 4. Choice: binning variable (M200c vs M500c; Msun vs h^-1 Msun)

**Scatter between the two masses** (`analyses.m500_vs_m200.per_M200c_bin`).
- *Size.* At fixed 0.2-dex M200c bin, sd(log M200c - log M500c) is 0.027-0.066 dex over the 15 bins. The minimum is at 15.1-15.3 and the maximum at 12.5-12.7; in the calibrated bins it is 0.027-0.056.
- *Origin.* Its correlation with log c200c is -0.895 (15.1-15.3) to -0.981 (15.3-15.5) over all bins, and -0.910 to -0.944 in 13.5-15.1. So M500c/M200c is almost entirely concentration.
- *Mixing between schemes.* An M500c_0.2 bin takes 60-76 % of its haloes from its most-populated M200c_0.2 bin (bins with >= 31 haloes; 86 % in the 7-halo top bin). Also, 83 % of the 13.5-13.7 M200c bin falls in the 13.0-13.5 M500c_0.5 bin.

**Which haloes move, and what that does to the stacks.**
- *Selection.* At fixed M500c, the steep mass function brings in more haloes from the low-M200c side. Those must have a high M500c/M200c, i.e. high concentration. The nine M500c_0.2 bins with 31-782 haloes whose mean M200c lies in the calibrated range have a mean log c200c higher by 0.008-0.050 dex than M200c bins at the same mean M200c; the 7-halo top bin shows no offset (-0.0002 dex).
- *Effect on the stacks* (`scheme_vs_reference_same_mass`, P, against the reference at the same mean M200c). In the three best-populated M500c_0.2 bins above 10^14.3 in mean M200c (14.1-14.3, 14.3-14.5 and 14.5-14.7 in M500c; 352-782 haloes; mean log M200c 14.35-14.74), central pressure (< 0.25 R200c) is higher by +4.4 to +8.0 % (errors 1.2-1.3 %), and pressure at 1-1.5 R200c is lower by 2.6-5.8 % (errors 1.2-1.4 %). Including the sparser bins 14.7-14.9 (147 haloes) and 14.9-15.1 (31 haloes), the ranges widen to +1.6 to +17.6 % and -2.4 to -6.6 %; the 14.7-14.9 core offsets (+1.6±2.6 and +2.2±1.9 %) are consistent with zero.
- *RMS pull of the stacks against the reference* (13.5-15.1): 2.43 (P) and 2.40 (E), with 10 and 8 bands at |pull| > 3. For 0.5-dex M500c bins: 2.97 and 2.95.

**Observable impact** (metric C, relative to the reference painted the same way; metric B in brackets).
- *0.2-dex M500c bins.* tSZ +1.61±0.41 / +3.79±0.64 / +5.07±0.51 % [B: +1.66 / +3.88 / +4.72], kSZ (ell = 3000) +0.77±0.23 %, DM mean +0.07±0.13 %, DM variance +0.74±0.19 %.
- *0.5-dex M500c bins (B24-like).* tSZ +1.88±1.11 / +2.38±1.25 / +2.53±1.16 %, kSZ -1.93±0.32 / -2.65±0.25 / -3.13±0.28 %, DM mean -0.98±0.15 %, DM variance -2.07±0.25 %.

**h^-1 units (Lee22) — corrected sign.**
- *Edge positions.* log M[h^-1 Msun] = log M[Msun] + log h, with log h = -0.167. So in Msun every Lee22 edge lies **0.167 dex above** its label: for example, 13.4-13.6 h^-1 Msun = 13.567-13.767 Msun, with mean log M200c 13.66. The nodes then sit 0.03-0.07 dex from the reference nodes (`lee22_bins_in_Msun`).
- *Stacks.* At the same mean M200c the stacks agree with the reference (RMS pull 0.84 for P and 0.75 for E, no |pull| > 3).
- *Observables, metric A.* Metric A charges Lee22 with kSZ (ell = 3000) -0.85±0.20 % and DM variance -0.63±0.17 %. That is the node-placement artefact of metric A.
- *Observables, metric C.* Continuous painting puts Lee22 within noise of the reference: tSZ -0.10±1.40 / -0.15±1.43 / -0.12±0.71 %, kSZ (ell = 3000) -0.11±0.18 %, DM variance -0.06±0.16 %.
- *Lee22's halo-count rule.* Their rule of more than 29 haloes per bin would drop 15.0-15.2 h^-1 (17 haloes) and 15.2-15.4 h^-1 (2).

**Recommendation.** Bin in M200c, the quantity HalfDome carries. Binning in M500c selects by concentration, even though the profiles are in R200c units. With 0.2-dex M500c bins it moves tSZ by +3.79±0.64 / +5.07±0.51 % at ell = 1000 / 3000 (+1.61±0.41 % at ell = 100). With 0.5-dex M500c bins tSZ moves by +1.9 to +2.5 % (1.7-2.2 sigma) and kSZ by -1.9 to -3.1 %. Msun or h^-1 Msun makes no difference, provided nodes are placed at their mean masses in the painter's units.

## 5. Choice: haloes per bin

**Per-halo amplitude** (subsampling: 400 random N-halo subsamples, finite-population corrected; `analyses.N_needed_subsampling`, detail bins 13.5-13.7, 13.9-14.1, 14.1-14.3, 14.5-14.7, 14.7-14.9).

| Field, band | sigma1 (range over detail bins) | N for 3 % | N for 5 % |
|---|---|---|---|
| P 0.5-1 R200c | 0.109 (13.9-14.1) - 0.175 (13.5-13.7) | 14-35 | – |
| P 2-3 R200c | 0.710 (14.1-14.3) - 1.227 (14.7-14.9) | 560-1673 | – |
| P 3-4 R200c | 1.377 (13.9-14.1) - 2.799 (13.5-13.7) | – | 759-3134 |
| E 0.5-1 R200c | 0.072 (14.1-14.3) - 0.181 (13.5-13.7) | 6-37 | – |
| E 2-3 R200c | 0.270 (13.5-13.7) - 0.423 (14.7-14.9) | 82-199 | – |
| E 3-4 R200c | 0.443 (13.5-13.7) - 0.511 (14.5-14.7) | – | 79-105 |

The analytic sigma1 (per-halo SD / mean) for P at 0.5-1 R200c is 0.12-0.20 over the calibrated bins 13.5-15.5.

**Skewed outer pressure.** The per-halo outer pressure has a heavy upper tail (plausibly from a few haloes with massive neighbours; this was not tested), so the stack of a small subsample has a right-skewed sampling distribution. The median subsample is low:

| P, median subsample bias | N = 10 | N = 25 | N = 50 |
|---|---|---|---|
| 2-3 R200c | -1.5 to -14.9 % | +0.1 to -9.0 % | +0.3 to -4.9 % |
| 3-4 R200c | -14.0 to -19.9 % | -4.0 to -11.5 % | +0.2 to -9.2 % |

The mean over the 400 subsamples stays much closer to the full-bin value (`sample_size.precision`, mean_bias; standard error = SD / sqrt(400)):
- *3-4 R200c.* -4.6 to +1.3 % at N = 10 (standard error 2.1-4.5 %, every bin within 2.1 sigma) and -2.1 to +1.5 % at N = 25.
- *2-3 R200c.* Four of the five detail bins are within 1.9 sigma at N = 10-50. In 14.7-14.9 the mean is also low: -5.3±1.8 % (N = 10), -3.2±1.1 % (N = 25) and -2.1±0.8 % (N = 50).

**Observable scatter** (`analyses.sample_size.observable_scatter`): 400 realisations with N random haloes in every calibrated bin that holds more than N.

| N | 10 | 25 | 50 | 100 | 200 | 300 |
|---|---|---|---|---|---|---|
| tSZ ell=1000 SD | 2.96 | 1.86 | 1.18 | 0.66 | 0.37 | 0.21 |
| tSZ ell=3000 | 3.82 | 2.53 | 1.60 | 0.98 | 0.58 | 0.34 |
| kSZ ell=3000 | 2.03 | 1.22 | 0.77 | 0.51 | 0.28 | 0.10 |
| mean DM | 1.61 | 0.97 | 0.63 | 0.42 | 0.23 | 0.08 |
| DM variance | 1.88 | 1.15 | 0.74 | 0.49 | 0.28 | 0.11 |

- *Scaling.* SD x sqrt(N) is 9.36 / 9.30 for tSZ (ell = 1000) and 5.08 / 4.87 for DM mean at N = 10 / 25.
- *Finite-population limit.* The subsampled bins hold 127-895 haloes (plus 15.1-15.3, with 21, at N = 10), and these SDs carry no finite-population correction. SD x sqrt(N) already falls at N = 50 (8.37 % for tSZ at ell = 1000, 4.49 % for DM mean) and further at N = 100 (6.56 / 4.16 %). The values from N = 50 upwards are therefore lower limits; Fig 10 marks N >= 100 with open markers.

**Strict-rule resolved range vs N.** The strict rule requires every halo to be resolved, so larger samples start further out. In 14.1-14.3 the median first resolved edge is 0.105 R200c at N = 10, 0.123 at N = 25-50 and 0.145 at N >= 100. In 14.5-14.7 it moves from 0.065 to 0.090 (N = 500).

**Precision of the per-halo scatter** (`scatter_precision`, five detail bins, finite-population-corrected `rel_sd_inf`).
- *Noise.* The SD of ln(Y_4R) (field P) is known to 28-38 % (relative, 1 sigma) at N = 10 and 10-17 % at N = 100. For ln(N_e,4R) (field E) the figures are 24-29 % and 8-11 %.
- *Bias.* The median estimate of sd ln(Y_4R) is low by 12-25 % at N = 10; at N = 100 it is -2.5 to +0.3 %. For E it is low by 8-16 % at N = 10 and by 0.5-1.2 % at N = 100.

**Covariance on Lee22's grid** (strict-resolved shells, p = 11-24).
- *At N = 30.* The mean SD of a correlation coefficient (finite-population corrected, `mean_sd_offdiag_inf`) is 0.12-0.14 (P) and 0.09-0.12 (E). The second eigenvector's |cos| with the full-sample one is 0.89-0.99 (P) and 0.95-0.99 (E). The Hartlap-corrected precision trace ratio has mean 1.04-1.22 and SD 0.18-0.45 (P), and mean 1.03-1.12 and SD 0.16-0.56 (E).
- *At N = 100.* Correlation SD (same correction) 0.063-0.078 (P) and 0.051-0.067 (E); |cos| 0.97-0.998 (P) and 0.988-0.997 (E); trace-ratio mean 0.98-1.05 (P) and 0.95-1.02 (E), SD 0.06-0.08 (P) and 0.05-0.06 (E).
- *Inference about Lee22's rule.* On their full 27-shell grid the Hartlap factor (N-p-2)/(N-1) is positive only for N > 29, which matches their more-than-29 rule. This is an inference, not a tested statement.

**Recommendation.**
- *Per bin.* Use at least 100 haloes per 0.2-dex bin. By the measured scalings this gives an estimated ≈0.9 % tSZ (ell = 1000) and ≈0.5 % mean-DM error (9.3 %/sqrt(N) and 4.9 %/sqrt(N)).
- *Current sample.* With 317-327 haloes per sampled calibrated bin, the sampled bins 13.5-14.5 contribute 0.15 / 0.20 / 0.33 % to the tSZ noise and 0.25 % to DM mean (census bootstrap, section 6).
- *Outer pressure.* Beyond 2 R200c, a 3 % per-bin precision for P is out of reach in L1_m9: at 2-3 R200c it needs 560-1673 haloes, more than each detail bin holds (319-895). Rely on a smooth fit across bins and radii there.
- *Covariances.* Use N >= 100 if the analysis needs the shell covariance (GLS).

## 6. Choice: the sparse top bins (15.1-15.3: 21 haloes; 15.3-15.5: 6)

**Why they matter** (`top_bins.bin_shares`). Response to a uniform 1 % change of one bin:
- *Top two bins.* They carry 29.8 / 16.6 / 6.2 % of the total tSZ response at ell = 100 / 1000 / 3000 (30.3 / 16.8 / 6.4 % of the calibrated-bin part). For kSZ (ell = 3000), DM mean and DM variance they carry only 0.2 / 0.3 / 1.6 %.
- *Sampled bins 13.5-14.5.* They carry 17.1 / 27.9 / 44.6 % of tSZ and 52-67 % of the kSZ and DM observables.

**Statistical error of the reference sample — corrected** (`top_bins.reference_sample_error_pct`).

| tSZ C_ell noise (%), ell = 100 / 1000 / 3000 | census bootstrap (this group) | core-convention bootstrap | jackknife |
|---|---|---|---|
| all calibrated bins | 1.77 / 1.99 / 1.24 | 1.22 / 1.50 / 0.97 | – |
| census bins 14.5-15.5 only | 1.75 / 1.95 / 1.19 | 1.18 / 1.45 / 0.89 | 1.81 / 1.95 / 1.14 |
| top two only | 1.54 / 1.76 / 0.89 | 1.01 / 1.35 / 0.69 | 1.70 / 1.85 / 0.93 |
| sampled bins 13.5-14.5 only | 0.15 / 0.20 / 0.33 | 0.15 / 0.20 / 0.33 | – |

- *The previous version quoted the core-convention row.* In the census bins it is too small by a factor of about 1.3-1.5.
- *kSZ and DM noise.* It is set by the sampled bins: all calibrated bins give kSZ 0.46 / 0.42 / 0.31 %, DM mean 0.26 % and DM variance 0.32 % (census design).
- *Per-band errors of the individual top stacks* (P, 0.5-1 R200c, census bootstrap; delete-one jackknife in brackets): 3.1 % (3.0 %) in 15.1-15.3 and 7.6 % (8.0 %) in 15.3-15.5.

**Treatments of the top bins.** Each treatment is a node set painted continuously. Results are relative to the reference painted the same way (`top_bins.continuous_painting`), in %.

| Treatment | tSZ 100 | tSZ 1000 | tSZ 3000 | kSZ 100 | DM var |
|---|---|---|---|---|---|
| top two pooled (27 haloes) as one node at its mean mass | -0.85±0.95 | -1.30±0.70 | -0.48±0.25 | +0.05±0.05 | -0.01±0.01 |
| >= 14.9 pooled (154) as one node | -2.92±1.35 | -1.98±1.41 | -0.74±0.67 | -0.13±0.08 | -0.03±0.04 |
| top two from a trend fit over 14.3-15.1 | -1.26±1.64 | -0.11±1.70 | +0.39±0.93 | -0.00±0.12 | +0.10±0.06 |
| top two dropped, last segment extrapolated | -4.08±2.00 | -2.44±2.11 | -0.94±1.20 | -0.28±0.16 | -0.05±0.08 |
| top two dropped, 14.9-15.1 held flat | -2.16±1.36 | -2.56±1.69 | -1.17±0.94 | -0.31±0.12 | -0.18±0.06 |

Step painting (each bin painted with one stack, vs the reference stacks) gives:
- top two painted with the 27-halo pooled stack: -1.20±0.86 / -1.24±0.55 / -0.36±0.15 %;
- top two painted with the 14.9-15.1 stack: -2.08±1.30 / -2.24±1.51 / -0.94±0.80 %.

Against the truth, the reference's own continuous painting is +0.02 / +0.03 / -0.38 % on tSZ.

**Recommendation** (only treatments tested above).
- *Keep or pool.* Keep the two census top bins as their own nodes, or pool them into one node at the pooled mean mass. Pooling shifts tSZ by -0.85 / -1.30 / -0.48 %, within 1.9 sigma of the paired error.
- *What to avoid.* Do not drop them: that costs -2.2 to -4.1 % at ell = 100 (1.6-2.0 sigma). Do not paint them with the 14.9-15.1 stack or a stack pooled from 14.9 upwards: that costs -2.0 to -2.9 % at ell <= 1000 (1.4-2.2 sigma). The four-node trend fit over 14.3-15.1 agrees within noise (-1.26±1.64 / -0.11±1.70 / +0.39±0.93 %), but it is noisier than keeping or pooling the bins, about as noisy as dropping them with 14.9-15.1 held flat (±1.36 / 1.69 / 0.94 %), and less noisy only than dropping them with an extrapolated last segment (±2.00 / 2.11 / 1.20 %).
- *Quote the noise.* Quote about 1.8-2.0 % (ell <= 1000) and 1.1-1.2 % (ell = 3000) tSZ calibration noise from the finite census. More L1_m9 haloes cannot reduce it, because every halo there is already used.
- *Untested suggestion.* Stacking adjacent snapshots, or adding a larger box for M200c > 10^15.1, would help. This was not tested here.

## 7. Figures (groups/mass_binning/figures/)
- fig01_scheme_band_ratios.png: band ratios of every scheme to the reference at the same mean M200c.
- fig02_observable_impacts.png: scheme impacts under metric A (left) and metric B (right). Metric C, used in the tables above, is not plotted.
- fig03_mass_trend.png: residual mass trend per 0.1 dex. The grey band marks the calibrated range 13.5-15.5; the 15.1-15.3 bin is not plotted.
- fig04_wide_vs_fine.png: 0.3- and 0.5-dex stacks against 0.1-dex stacks at the same mean mass.
- fig05_normalisation_smearing.png: bin-mean / own normalisation for P, E, T_e and K. Solid lines are the second-order prediction including the residual mass trend; dotted lines the pure Jensen term.
- fig06_m500_vs_m200.png: M500c/M200c against concentration, and the M500c-binned pressure offsets.
- fig07_precision_vs_N.png, fig08_N_needed.png, fig09_covariance_stability.png, fig10_observable_scatter_vs_N.png: sample size. In fig10 the open markers (N >= 100) are finite-population-limited lower limits.
- fig11_top_bins.png, fig12_bin_shares.png: top bins and observable shares.
- fig13_topbins_normalisation_impacts.png: top-bin treatments and bin-mean normalisation.
- fig14_interpolant_test.png: bin width x interpolant (linear vs PCHIP) against the truth.

## 8. What changed from the previous version
- **Bin width.** The cost is now measured by continuous painting (metrics B and C) plus an interpolant test. The old metric A favoured the reference's nodes: 0.3 dex on kSZ (ell = 3000) moves from -1.40 to -0.80±0.18 % (metric C), and the Lee22 artefact is gone.
- **Normalisation.** The pure-Jensen explanation is replaced by the second-order form with the residual mass trend (beta), which matches all bands including K. The alternative form exp(0.5 (a+alpha+beta)^2 sigma^2) - 1 keeps a beta^2 term that cancels in the correct form, omits the curvature terms, and gives K ≈ 0, while -2.4 % is measured.
- **Statistical errors.** They now use the census design. The reference tSZ noise is 1.77 / 1.99 / 1.24 % (not 1.22 / 1.50 / 0.97).
- **Lee22.** The edges lie +0.167 dex above their labels in Msun (the sign is corrected in the text and in the web description).
- **Corrected ranges.** sd(Δlog M) = 0.027-0.066 dex with r = -0.895 to -0.981. sigma1 (P, 0.5-1 R200c) = 0.109-0.175 (N for 3 %: 14-35); at 3-4 R200c 1.377-2.799 (N for 5 %: 759-3134).
- **Saved checks.** The wide-vs-fine selection is stated, the bootstrap-consistency check and the rescaling test are saved, and the trend_top web records carry their bins' own N.
- **Top bins.** The recommendations are restricted to the tested treatments.
- **Text revision after the second verification.** The integrated painting estimate is no longer said to agree with metric C. The mass-trend maximum is +14.4±0.5 % per 0.1 dex (n_e at 0.25-0.5 R200c in 13.7-13.9), and the trend ranges are given per field. Small-subsample outer-pressure stacks are described as right-skewed (low median, mean close to the full bin) instead of biased. The 0.5-dex failure is attributed separately for linear and PCHIP interpolation; with PCHIP, DM mean no longer counts as a failure. Significances are quoted for the 0.5-dex PCHIP tSZ loss, bin-mean normalisation and the top-bin options. Ranges that held for one field or a subset of bins now name it: covariance stability, scatter precision, M500c stack offsets and concentration offsets, the Jensen under-prediction factors and Pbar_arith_over_geo. A caveat on the first-order impact propagation is added.
- **Precision edits after the third verification.** The curvature range above 10^14.3 is restricted to 14.3-15.1, with the 15.1-15.3 values stated separately. The 0.5-dex PCHIP tSZ total is attributed to all bin groups (15.3-15.5, 14.9-15.3 and 13.5-14.9). The covariance correlation SDs are labelled as finite-population corrected.
