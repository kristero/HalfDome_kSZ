# Adversarial check of sections/s2_frb.html (section id "frb"), 2026-10-07

Two passes. Pass 2 (this one, "Try again") re-verified everything from scratch on top of pass 1's fixes and made the
corrections listed under "Pass 2". Pass 1's record is kept below.

## IMPORTANT for whoever rebuilds the section
All fixes (pass 1 and pass 2) were made directly in `sections/s2_frb.html`. `scripts/frb/s2_frb_template.html` and
`scripts/frb/copy_figures.py` were NOT edited (outside the checker's write scope). Running `build_section.py` or
`copy_figures.py` again would revert them. Before a rebuild:
- port the text fixes into the template (diff template vs fragment); the 12 code excerpts are unchanged, but data-src now
  uses `path:L1-L2` and XGPaint excerpts are pinned as `XGPaint.jl@1660e08/src/...` (build_section.py would need the same);
- in copy_figures.py change the source of `09_finite_frb_realizations` to
  `frb_map_generation/outputs/tsz_dm_fullsky_20260918/plots/realizations_selected_battaglia_residuals.png` (local, 2026-09-25).
- Note: the session scratchpad is shared with the other section checkers (s3's checker overwrote a generic
  `check_html.py`); pass-2 scripts live in `<scratchpad>/s2chk_x4/`.

## Pass 2: what was checked
1. Code excerpts: 12/12 verbatim at the cited lines (HTML-unescaped, exact string compare). XGPaint excerpts against
   /home/kn18001/.julia/dev/XGPaint at 1660e08 (branch cluster, clean tree). NOTE: `julia_env/dev/XGPaint.jl` in the repo is an
   older upstream-like copy and differs, hence the explicit `@1660e08` pin.
2. Figures: 14 PNG exist, open, 1279-2000 px, <= 1.0 MB; each viewed and matched to its caption; each equals its stated
   source (mean abs pixel diff 0 after the same resize; 12 is the 400 dpi render of the identical PDF); PDFs identical to source
   PDFs; 08 has no PDF (stated).
3. Numbers (~110 traced): whole z-series stats CSV (TNG mean/hit, hit %, three model ratios, '13to14' ratios); captions.txt,
   captions_tsz_mass.txt, captions_ksz.txt (all values used); the chi2 CSV (all 15 pairs used, p-values and Hartlap recomputed
   with scipy); yy_power_stdout (whole table incl. l~100 column, low-z fix ratios); finite_source_pairs_summary.json; map means
   (map_mean_vs_100k_rays.csv, corrected_spectra.npz metadata); DM-map provenance (85,224,251 rows, 3.2e-4/3.9e-4, 3.79e15,
   shape clip 9.28e14); halo_pdf_corrected_lee22_stats.csv (Fig 2.6 boxes, 78.8/72.4, 2-3 % and <=1 %); Lee22 doc v4 (0.80-0.84,
   27..41 %, finite-box ranges, F_P 0.99-1.07 / 1.21-1.25, 79,302, ~1,200, 92/85/59/39 %, KS, 83 %, 58 %, 0.33/0.53/0.47 vs
   0.01/0.05/0.21, 0.3+-0.6 %); analytic README sec. 1/3/4; WHAT_CHANGED sec. 2; B_class_sz and A_xgpaint_audit READMEs;
   TSZ_DM_FULLSKY sec. 4/8; TAKAHASHI25 notes; kernel CSVs; PBS defaults (seed 42, chunk 1e6, 300 edges 0.1-3e4, 12 windows;
   13 in the w14 runs); PAPER_CUT, BEAMS, MIN_COUNT=5, C(z) constants, self-test tolerances; every file:line pointer.
   Independent re-runs (read-only): TNG window identity residuals 1.14e-13..9.09e-13, decade hit % and mean/hit, cluster-decade
   and nominal hit % (all reproduced exactly); Poisson factors N/(1-e^-N) (1.008/1.043/1.069/1.075/1.074/1.071 HalfDome/TNG);
   low-DM shares below 10 pc cm^-3 from ray histograms + raw arrays (corrected -27.5/-31.5/-36.9/-38.4/-40.4/-40.6 %, B16
   -0.5..+1.3 % at z_s >= 0.5); Battaglia16 gas outside the sphere within a 1 R200c cylinder (26-30 %); 0.6016, 3.35/0.42/0.12,
   2.5/1.9 at z = 0.1/0.2, m_per_e = 1.137 m_p, 1 m^-2 = 3.24e-23 pc cm^-3, P_e/P_th = 0.88/1.70.
4. Physics/statistics: DM (1+z)^-1, chord formula and chord factor, chord-mean caching, Poisson inversion, compound Poisson
   CF, Hartlap definition and its effect on Delta chi2, Gaussian beam, Legendre annulus means, kernel-weighted ensemble mean,
   n200/P200, X_H factors: correct. Causal claims checked against the cited reports.
5. XGPaint: git show on 9f37b1e, 077b849, b924ab1 (merge of b459728 + 077b849), b459728, e67a8a1, 71646e2, ce1d239, 5dd0b57,
   07fc686, 35f79ad (git diff ce1d239 35f79ad empty), 1660e08, c609849; upstream/main 29ec48a for "before" code
   (rho_2d/ne2d, zmax=1e5, P_e 0.5176, no profiles_frb.jl, no fill_profile_slice!). B12 sign arithmetic checked
   (upstream -4.35, 9f37b1e +4.35, e67a8a1 -4.35); B16 exponent identical at all m, z. Project-vs-XGPaint attribution correct.
6. HTML: one <section id="frb" class="topic">, balanced, allowed tags/classes only, all img exist, no script/style, no em-dash.

## Pass 2: corrections made in the fragment
- data-src: `path:1-2` -> `path:L1-L2` (contract); XGPaint paths pinned `XGPaint.jl@1660e08/...`.
- Fig 2.8 caption: "one ACT source above z = 2" was wrong: FRB 20240304B (z = 2.148) and 20220610A (z = 1.016) are in BOTH
  kernels; "halos above z ~ 0.5 carry < 20 % weight" -> at most 11 % (Planck, 8/71) / 10 % (ACT, 3/31); 63/71, 28/31 below 0.5.
- Ray provenance: "record xgpaint_version=unknown" was half true: idark rays (09-29/30) say unknown, local rays (10-01) say
  0.4.0; neither records a commit (aside + open item).
- XGPaint aside (iv): the audit compared the LOCAL copies of the lookup tables (B12 built 2026-04-29, Lee22 2026-09-17);
  the 2026-09-30 maps were painted from idark copies (/lustre/...), not compared directly. Aside (i): added that the idark
  B12 table was not read (Lee22 doc sec. 1).
- Sphere aside: WHAT_CHANGED's "27% of the projected column at 1 R200c" is physically garbled (at b = R200c the whole column
  is outside the sphere); reworded as the fraction of gas inside a 1 R200c aperture that lies outside the sphere, checked
  here at 26-30 % for B16.
- Low-DM deficit: made explicit that 27..41 % refers to the fraction of hit rays below 10 pc cm^-3 (TNG 4.7-6.5 %), with
  the re-derived values; same in Open items. "about 8,000 rays hit" -> ~7,900 TNG / ~9,700 HalfDome.
- Takahashi bin 0: not "dropped" (observations() loads all 13 bins); it lies below both angular cuts.
- Fig 2.9: outliers "from a single massive halo next to one FRB" -> one sightline through a massive cluster core
  (TSZ_DM_FULLSKY sec. 4); provenance "2026-09-18 maps" -> 100k sightlines of 09-17 + annular y samples
  (takahashi_100k_20260914) from the pre-repaint B12 map.
- "TNG300's 278/152/49 clusters per snapshot" -> in the z = 0/0.5/1 snapshots. "The paper" -> Konietzka+25 (README sec. 1).
- ACT/Planck "neither survey separates Battaglia from hybrid/calibrated": quantified (Planck 0.9/0.6; ACT 5.2 (3.1 Hartlap)
  /1.1 (0.6)).
- "Lee22 as printed 2.1x FLAMINGO at large scales" -> 2.1-2.3x at l ~ 100-300. Lede "mostly from extrapolating" -> at l < 2000.
- "(1+z)^3, h^3, X_H^2 all come from the extraction code's units" -> inferred; measured CAMELS-TNG F_n (~5.5 vs 5.57) chose
  among candidates (Lee22 doc sec. 3).
- Battaglia16 parameter citation: the project constructs HaloDMProfile from BattagliaTauProfile (lee2022_frb_dm_profile.jl:903);
  defaults identical (profiles_tau.jl:28-33 = profiles_frb.jl:16-21). B16 exponent: unchanged at every mass and z.
- 71646e2 "every product postdates" -> every DM product (the B12 y table was built 2026-04-29 but does not use the constant).
- Halo-model code note: "minus its zero mode" -> minus its value at zero frequency (bias-weighted N-bar).
- Definitions added: ne2d/rho_2d, P_th and 0.5176 = n_e/(n_e+n_ion), F_P, M_cut and the P0 broken power law, CAMELS-TNG,
  stratified sightline estimator. Fig 2.10 "(0.000)" -> "(< 0.001)".

## Remaining concerns (not fixed)
- Template/copy script out of sync with the fragment (see top).
- Prose ~10.5k words excluding code (above the 3-7k guideline); not cut, to avoid losing verified content.
- R. Konietzka has not confirmed the window convention; ray provenance lacks the XGPaint commit (both open items in the text).
- The idark copies of the y lookup tables that painted the 2026-09-30 maps were never compared with the audited local copies.
- "y x DM chi2" and the realisations figure use different map vintages (09-30 repaint vs pre-repaint y samples); stated.

---
# Pass 1 record (kept)
## What was checked
1. Code excerpts: 12/12 match the current files verbatim at the cited lines (HTML-unescaped comparison;
   XGPaint excerpts against /home/kn18001/.julia/dev/XGPaint at 1660e08, clean tree, branch cluster).
2. Figures: 14 PNGs exist, open, are separate files and show what the captions say (each viewed). PDFs present for all but 08.
3. Numbers: about 70 traced. Re-ran read-only on the raw Konietzka arrays: identity residuals 1.14e-13 ... 9.09e-13, decade hit %
   6.60/29.65/46.33/53.46/56.25/57.66, 1e14-1e15 decade 2.011/8.306/11.726 %, nominal difference 0.065/0.231/0.263 %, decade
   mean/hit 77.54/87.29/101.99/112.42/117.35/120.81. Also checked: halo_pdf_zseries_stats.csv (whole table),
   captions.txt, captions_tsz_mass.txt, captions_ksz.txt, tsz_x_frbdm_corrected_chi2.csv (whole chi2 table, 92 rows),
   yy_power_stdout.txt (whole table + low-z fix), halo_pdf_corrected_lee22_stats.csv, map_mean_vs_100k_rays.csv,
   corrected_spectra.npz metadata (map means), DM-map provenance (85,224,251 rows, 3.785e15, cache errors 3.24e-4/3.85e-4),
   finite_source_pairs_summary.json, Takahashi npz (bins, S/N 3.35 at 80'), analytic-repro README + implied_gas_fraction.log,
   Lee22 doc v4 text, WHAT_CHANGED §2, A_xgpaint_audit README, B_class_sz README, generator/PBS defaults (seed 42, chunk 1e6,
   300 edges 0.1-3e4), line numbers of every cited function.
4. XGPaint: every commit hash/author/date/file in the asides checked with git show (9f37b1e, 077b849, b459728, b924ab1,
   e67a8a1, 71646e2, ce1d239, 5dd0b57, 07fc686, 35f79ad, 1660e08, c609849; upstream/main for "before" code).
5. HTML: one section, balanced tags, only allowed tags/classes, no script/style, no em-dashes.

## Corrections made (in the fragment)
- Fig 2.9 (finite-FRB realisations) was the idark 2026-09-18 plot, whose eight "best fit" realisations were selected with a
  diagonal chi2 against the old digitised points (TSZ_DM_FULLSKY_20260918.md l.126); the caption numbers (19.86/19.84/22.09/
  9.61/24,724/2,426; ACT 10.72/9.44) come from the local real-covariance summary (2026-09-19; the idark summary has 28.43/28.42).
  Replaced PNG+PDF with the local regenerated plot (plots/realizations_selected_battaglia_residuals, 2026-09-25); caption and
  source path updated.
- Planck "within -1.8 / +2.1 of w=0": max is 24.03 - 21.85 = +2.18 -> "+2.2" (doc's +2.1 is from rounded values).
  w=0 Planck p with Hartlap 0.0146 -> 0.015 (doc 0.014); added p = 0.005 without. ACT corrected pair p without Hartlap 0.017 added.
- Hartlap note said it "matters for absolute fit, not for differences": wrong, it scales Delta chi2 too (14.0 -> 8.4); fixed.
- "fit applied up to 3.8e15 Msun, roughly 0.8 dex beyond its fitted bins": 3.8e15 Msun = 10^15.4 h^-1 Msun is ~1.4 dex beyond
  the ~1e14 h^-1 end of the bins (0.8 dex is the hybrid threshold 10^14.8); fixed.
- Pipeline factor "3.35 times too high at z < 0.1" -> 2.5-3.35 (3.35 only at z = 0); Fig 2.8 caption "2-3 times" -> 3.35/2.5/1.9
  at z = 0/0.1/0.2, 1 near 0.5; chi2 bullet likewise.
- Fig 2.6 caption "all four densities" -> the three Lee22 densities (no Battaglia16 in that figure).
- C(z) formula: 0.284 (1+z)^2.48 -> 0.284 (1+min(z,2))^2.48 (as in the doc/code).
- "identity asserted before every figure" -> every panel of the z-series figures (only call is plot_pub_figures.py:115).
- "one catalogue pass paints the nine DM models" -> each painter run does one pass; the nine maps come from three runs (09-18/29/30).
- XGPaint aside 1: added fork commit b459728 ("f_b addition" in ne2d) and that the merge b924ab1 kept Bayer's single f_b in rho_2d
  (no double counting); clarified that the 6.4x-high curve is Lee22's drawn Battaglia16 curve.
- XGPaint aside 2: "idark runs used 5dd0b57" -> 5dd0b57 plus uncommitted syntax patches (5dd0b57 does not parse;
  TSZ_DM_FULLSKY §8, A_xgpaint_audit); local precompile caches = ce1d239 source. Same in Open items.
- XGPaint y aside: B12 bug made explicit ((1+x)^+4.35); ce1d239 also changed profile_grid defaults; added the audit's
  <=7e-14 rtol effect and the lookup-table equality.
- Note aside: the doc's "kSZ maps still use the previous density" is out of date (kSZ repainted with corrected density,
  e8934dc, 2026-10-02); tSZ_64k phrasing fixed.
- kSZ paragraph: the WebSky-halo check is a one-halo calculation, not a map; range 0.975-1.01.
- Lede: "density was up to 2.85 times too high" -> 2.85 times TNG's mean DM per hit; causal "ACT disfavours because y power
  2.3-3x FLAMINGO" reworded (ACT disfavours via small-angle y amplitude; FLAMINGO excess is a separate observation).
- Fig 2.7 caption: BFC defined, KS distance defined, F defined; panel (c) computed with upper edge 9e14 (<1 %).
- Clarity: Compton-y, TNG300, FLAMINGO, MILCA, ACT, Lee22 table numbering (MNRAS A2 / arXiv Table 7), "Lee22 doc", Poisson
  inversion N = -ln(1-f_hit), analytic-model amplitude S/dz defined.
- Added aside (derived here, flagged as such): mean per hit includes multiple crossings; N/(1-e^-N) from the tabulated hit
  fractions is 1.008/1.043/1.069/1.075/1.074/1.071 (HalfDome/TNG), so per crossing the corrected density is ~0.94-0.97 of TNG,
  consistent with the doc's 0.96-0.97 over all rays with TNG300 abundance.

## Remaining concerns (not fixed)
- Template/copy script out of sync with the fragment (see top).
- data-src uses "path:1-2" while s1/s3/s4/s6 use "path:L1-L2"; the assembler does not parse it, so harmless.
- Prose is long (~11.5k words incl. captions/tables, above the 3-7k guideline).
- R. Konietzka has not confirmed the window convention; xgpaint_version=unknown in ray provenance (both stated as open items).
