# Adversarial check of sections/s1_lee22.html (2026-10-07)

## What was checked
- 12/12 code excerpts: byte-for-byte against the current files after HTML-unescaping (XGPaint.jl/ = dev checkout
  /home/kn18001/.julia/dev/XGPaint, branch cluster, HEAD 1660e08). All match. Every file:line pointer in the prose was
  opened (lee2022_frb_dm_profile.jl 135/321/240-250/654-664/799-815/835-870, lee2022_tsz_pressure_profile.jl 9/38/41/43/66-79/94,
  paint_tsz_variants.jl 12/34/45-55/63-79/103-106/119-120/237/243, generate_halfdome_z1_dm_mass_windows.jl:187,
  run_lee22_corrected_cluster.pbs:63-65, tng_windows.py 9-28/31-34, ksz_models.jl 24-30, y_prof.cpp 85-86/139-140,
  ne_prof.cpp 134-135, XGPaint profiles_y.jl:128, profiles.jl:489).
- 16/16 figures: exist, single PNG files (1343-2000 px, <= 1.06 MB), PDFs byte-identical to the cited sources, each viewed;
  fig 14 pixel-identical to the source PNG. Fig 15 script re-read; its data reproduce yy_power_stdout.txt.
- XGPaint: git show of 9f37b1e, b459728, b924ab1 (merge result), e67a8a1, ce1d239, 29ec48a (upstream/main), plus project
  commits 2c6d3b6, 7f70174, 2771265, 6fa5373, 7b2692c, c19bd20, e4cb510, 7a8e57d, e8934dc, 3ba440c. All hashes, dates,
  authors, files and before/after code as described, with the corrections below.
- About 90 numbers traced to: doc v4 (lee22_calibration.html), hbook.txt, Fn_integrated.txt, check_beta_convention.txt,
  lee22_fits_vs_drawn.txt, battaglia_curves.txt, b16_own_convention.txt, b12_cache_sign_test.json, fig1_check.txt,
  fig2bc_check.txt, grid_check.json, frb_decades/report.txt, v4b_lowz_impact.json, v4b_cl_*.npz, corrected_spectra.npz,
  selftest_600152 and y_600153 logs, yy_power_stdout.txt, pub captions.txt, the four LEE22_*.md reports, FINAL_REPORT.md
  of the forensics, konietzka_check README, tau_ksz_comparison/README.md, the 2026-10-07 unit-test rerun log.

## Corrections made in the fragment
1. C(z) description: the slice ratios give (1+z)^2.57-2.60 (C0 0.271-0.274); 0.284 (1+z)^2.48 (rms 0.8 %) is the fit to the
   cumulative halo DM over z_s = 0.1-2 (FINAL_REPORT.md Sec. 4.2). The fragment had attributed 2.48 to the slices.
2. "Enclosed gas fraction falls by a factor 3.4 from z = 0 to 1": the cited note's own table gives 3.07 -> 0.46 and 4.63 -> 0.63 f_b
   (factor ~7; n0 alone 2^2.11 = 4.3). Fig. 1.1a agrees with ~7. Text now quotes the table and flags the note's 3.4.
3. B12 cache sign test: stored/correct is 1 + (0.4-13)e-9 over 20 nodes (not "1.000000002"); buggy/stored 1e29-1e40 (not 1e29-1e33);
   added that this is the local copy (idark copy not read), as doc v4 says.
4. Battaglia16 vs TNG in the B16 XGPaint aside: "within 2.5-3.5 %" mixed B12 pressure (1.025) with B16 density (1.035). Now
   +3.5 % / -28 % (fig1_check.txt 1.035, 0.722). The causal "this is why ... 6-9x below the drawn curve" now names all factors
   (fork f_b and 0.9, upstream electron conversion, exponent), not the fork edits alone.
5. Battaglia (2016) Fig. A1 check: "0.984-0.998" holds only for the four bins 7e13-2e14 Msun; added 0.995-1.010 and 1.02-1.08 for
   the heavier bins. Separated the exponent evidence (curve fit) from the f_b evidence (3.25 f_b gas mass).
6. profile_grid aside: ce1d239 also changed thread-local storage; the Lee22 cache's 512 angles come from the project's explicit
   N_logtheta=512 (log line 23), not from the ce1d239 default (256).
7. 9f37b1e: the Battaglia16 beta flip is in src/profiles_tau.jl (file named). "Mapping correct only at HEAD 1660e08" -> correct
   for the exponent carried since b924ab1.
8. Open item "+16 % at l = 10 (B12)": that value is the ratio of C_l summed over l +- 5 (reproduced from v4b_cl_b12.npz:
   1.1639); the single-multipole effect in the production B12 map is 1/0.8108 = 1.23 at l = 10, 1.009 at l = 50. Fig 1.14 caption
   now states the 5-multipole running mean and the band definition. Added a sentence that the per-l ratios scatter 0.63-1.03 below
   l ~ 20 and that the verifier's prediction matches multipole by multipole to 1e-4.
9. C4 code quote: `halo_mass_m200c / H_VALUE` is not a code string; now quotes line 237 and line 243 verbatim.
10. C3 status "since 2026-09-17" -> in the code from its first commit (3ba440c); the division was never absent.
11. C5: the M^-1.10 divergence argument applies to the concentration fit (Table 3); added that the production no-c fits fall only as
   M^-0.18 (density beta') and M^-0.06 (pressure beta), so the clip is an extrapolation guard there. "These 1,211 halos dominate the
   low-l y power" -> they carry most of the Lee22 excess over B12 (35 at z < 0.1 carry ~95 % of the l = 100 excess; TSZ report Sec. 3).
12. Lede and key result overstated: "both factors follow from the units of the public code and are confirmed by CAMELS-TNG and
   FLAMINGO". Now: (1+z) powers from the code (identity for P, inference for n), confirmed by both simulations' trends; h^3 X_H^2
   inferred (code units + hydrogen convention) and selected by CAMELS-TNG (FLAMINGO gives 7.3-7.5, not 5.57). "5.57 comes from code
   units" -> 1/h^3 from units, 1/X_H^2 from the n200 convention.
13. Konietzka windows: the identity alone fixes only the 10^13 edge; added the hit-fraction evidence (2.0/8.3/11.7 % vs HalfDome
   2.0/8.7/11.9 %, doc v4 Sec. 3) that fixes [10^A, 10^(B+1)).
14. Status note: added that the public y_prof.cpp uses FoF Group_R_Crit200 while the paper used Rockstar halos, so even the pressure
   a^2 being in the paper's tables is an inference backed by CAMELS/FLAMINGO (TSZ report Sec. 5).
15. Clarity: defined mu_e and mu (candidate table), CV (cosmic-variance set), FoF; FLAMINGO "less hot gas" -> "about 20 % lower
   group pressure" (what was measured); 1.1 now says the pressure wrapper also overrides the y evaluation helpers, not only get_params;
   Fig 1.2 caption explains the two orange lines of the bottom row; Fig 1.3 grey band is x > 1.34 only; low-z Y budget labelled as the
   B12 map (Lee22: 0.43 %); constant LEE2022_FIT_MASS_MAX_HINV_MSUN located (lee2022_frb_dm_profile.jl:18).

## Verified and left as is (selection)
- hbook table (all 24 entries), mprot 2e-16, rho_crit_0 2.7753e-8; beta-convention table (12 cells); F_P/F_n table = doc v4;
  candidate F_n values recomputed (3.22, 4.23, 4.81, 5.57, 6.36, 7.33, 5.47); 0.6016 = 0.9 X_H m_p / m_per_e with m_per_e = 1.137 m_p;
  (1e14 h/10^13.61)^0.68 = 1.42; readings A-E table (230.4, 36.4, 154.1, 133.1, 138.6, 92.9); FRB ratios 0.978-1.024, 2.849-1.034,
  1.511/1.423/1.350, errors; 9.09e-13, 13.5 %, 55 z_s; replica numbers (0.981/1.033, 1.018/1.010, 3.412/3.383, fig2bc all);
  grid 3.35/0.42/0.12, z_equal 0.4964; y-power table; 3.17 at l = 273; 85,224,251 / 79,302 halos; self-test values (1.29e-4, 0.0,
  3.7066745225537403, PASS line 4); low-z Y budget; Konietzka 1.55 -> 0.95; unit-test rerun 0.652/0.761/0.932/1.150.
- XGPaint: no Lee22 number depends on an XGPaint edit; the 0.9 (e67a8a1) is cancelled in the final density path (line 811);
  B12 conversion bug (9f37b1e -> e67a8a1) not in the B12 table used; low-z fix is project-side and uses e67a8a1's fill_profile_slice!.

## Remaining concerns (not fixable from the fragment)
- Prose is ~10,500 words including captions/tables, above the 3,000-7,000 guideline.
- Stale comments in code (10^14.8 "fit range", "20 TNG300 snapshots", "interpolate linearly in z") remain in repo files (not edited).
- The idark copy of the B12 table was never compared byte for byte.
- Density n200 step of Lee22 not public; Konietzka window convention unconfirmed.
- The EQ7_EQ9 note's "factor 3.4" text contradicts its own table (repo file, not edited).
16. Sec. 1.4 "2.22-3.93 f_b at z = 0" was computed with the legacy 1e14 n0 pivot (implementation check item 2, masses 3e13-1e15);
   added the M_cut-pivot values 3.07 / 4.63 f_b (EQ7 note Sec. 5) so the numbers are consistent with C1.

# Second adversarial pass (2026-10-07, rerun after "Try again")

## Re-checked independently
- 12/12 excerpts byte-identical to the current files (XGPaint.jl/ = /home/kn18001/.julia/dev/XGPaint, cluster@1660e08, clean).
  HTML: one <section id="lee22" class="topic">, all tags closed, only contract classes, https links only, no script/style,
  no em dashes. 16/16 figures exist, open, are single files (1343-2000 px, <= 1.06 MB), each viewed; 15 PDFs byte-identical
  to the cited sources (fig 14 has none; fig 15 is the report's own script).
- XGPaint: git show 9f37b1e (parent 29ec48a, adrian branch), b459728, b924ab1 (parents b459728 + 077b849), e67a8a1, ce1d239,
  upstream/main 29ec48a (exponent (β-γ)/α, β = γ - αβ, N_z=256 LinRange, P_e = 0.5176 P_th). Project commits 2c6d3b6, 7f70174,
  2771265, 6fa5373, 3ba440c, 7b2692c, c19bd20, e4cb510, 7a8e57d, e8934dc, 6648c4b.
- Numbers re-traced (~110): doc v4 text, hbook.txt, Fn_integrated.txt, check_beta_convention.txt, lee22_fits_vs_drawn.txt,
  battaglia_curves.txt, b16_own_convention.txt, b12_cache_sign_test.json (20 nodes: 1 + 3.7e-10..1.29e-8; buggy 2.3e-40..1.5e-29),
  fig1_check.txt, fig2bc_check.txt, grid_check.json, frb_decades/report.txt, v4b_lowz_impact.json, v4b_cl_b12.npz and
  corrected_spectra.npz (recomputed: ±5 sum 1.1639, single l=10 0.8108 vs prediction 0.8107, max |diff| 1.0e-4 at the 13 plotted l,
  min/max 0.634/1.033 at l = 4/9), selftest_600152 and y_600153 logs, yy_power_stdout.txt, pub captions.txt, the four LEE22_*.md
  reports, FINAL_REPORT.md Sec. 4.2, konietzka README Sec. 3, fig5 audit REPORT.md, completeness_critic/edits.json,
  tau_ksz_comparison/README.md, the 2026-10-07 unit-test rerun log.

## Corrections made in this pass
1. Sec. 1.1 "XGPaint's projection, interpolation and HEALPix painting are used unchanged" was wrong for the FRB rays and kSZ:
   both use project-side chord integrals through a sphere (SphericalChordDMProfile/chord_dm_pc_cm3,
   lee2022_frb_dm_profile.jl:679-743; ChordMeanProfile, truncation_comparison/spherical_truncation_profiles.jl), and the painting
   loops are project code calling XGPaint's interpolators and ring helpers. Only the tSZ maps use XGPaint's own LOS integral.
   Lede "through XGPaint" -> "written in XGPaint's profile conventions".
2. XGPaint dependence that was missing: line 811 has no f_b and no rho_cr, so the eq. (9) n200 of the final Lee22 density is
   completed by f_b rho_crit(z) R200c from 9f37b1e's rho_2d convention. kSZ (Lee22TauProfile, ksz_models.jl:12-21) uses XGPaint's
   rho_2d/ne2d directly; the FRB rays use a project copy (Battaglia16DensityDMProfile, lee2022_frb_dm_profile.jl:654-664, selected at
   generate_halfdome_z1_dm_mass_windows.jl:200-201). run_lee2022_xgpaint_self_test (835-870) tests both against the project eq. (9)
   column (666-675) via the same line-811 branch (reading B). Added to the 1.4 XGPaint box and the line-811 code note.
   (The first pass's dossier line "no Lee22 number depends on an XGPaint edit" is too strong for kSZ.)
3. Sec. 1.8 box: the 6-9x between painted and drawn Battaglia16 omitted the fork's 0.9 (e67a8a1). Now lists f_b, X_H m_p/m_per_e
   (upstream), 0.9 and the exponent; 1/(0.6016 f_b) = 10.6 at equal shape (f_b = 0.1573 from b16_own_convention.txt).
4. Sec. 1.8: printed-exponent Battaglia (2016) Fig. A1 range "0.575-0.92" -> "0.575-0.99 (median 0.78-0.82)"; the widest bin
   reaches 0.986 (b16_own_convention.txt).
5. Table row C5 "prevents a divergent LOS" contradicted Sec. 1.3.5 (production no-c fits do not diverge) -> shape frozen; would
   prevent divergence only for the concentration fit.
6. Table row C6 "factors 0.158 to 1" ignored C(z) -> constant readings 0.158-1; C(z) = 0.28-4.3 at z = 0-2 (unit-test log 4.3489).
7. Sec. 1.3.1 "disagreed by 35-40 % at M_cut" -> at 10^13.75 h^-1 Msun, the concentration fit's M_cut (Table 3), z = 0
   (LEE22_IMPLEMENTATION_CHECK item 1).
8. Sec. 1.3.5: 1,211 (full tSZ catalogue, TSZ report Sec. 3) and 868 (foreground halos of z_s = 1 rays, implementation check item 4)
   are different counts; now said explicitly.
9. Sec. 1.4: chi^2 = 22337/19 belongs to the constant fit of the 20 slices (357/18 for the power law), not to the cumulative C(z)
   fit; sentence rewritten.
10. Key result: F_P = 0.99-1.07 qualified to 10^13.5-14 Msun, z = 0-1.5 (lower bin gives 0.67-0.86); DM ratio qualified to
   10^13-10^14 Msun halos.
11. Status note: "three independent facts" -> three pieces of evidence, two of which both use the CAMELS-TNG gas; the z trend tests
   only a^3.
12. Clarity: Rockstar, FLAMINGO L2p8_m9 (2.8 Gpc fiducial run, flamingo_pressure_calibration/README.md), HMF, CLASS-SZ defined.

## Remaining concerns
- scripts/lee22/s1_lee22.template.html is stale: the section was edited directly in both passes. Re-running
  scripts/lee22/build_section.py would silently revert all corrections. Not fixable within this checker's write scope.
- Prose ~7,900 words (incl. asides; ~9,600 incl. captions/tables), above the 3,000-7,000 guideline.
- verify_v3/ (y_prof.cpp, ne_prof.cpp, hbook.py copies) is untracked in git; excerpts cite untracked files.
- Fig. 1.1 is a four-panel composite (historical summary); left as one file.
- Inherited open items: idark B12 table copy not compared byte for byte; Lee22 density n200 code not public; Konietzka window
  convention unconfirmed; stale code comments (10^14.8 fit range, "20 TNG300 snapshots", "interpolate linearly in z").
- Shared scratchpad: another session overwrote this checker's generic check_html.py mid-run; rechecks used uniquely named scripts.
