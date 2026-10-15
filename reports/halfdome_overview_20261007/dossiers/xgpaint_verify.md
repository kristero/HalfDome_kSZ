# Verification of sections/s6_xgpaint.html ("Appendix: XGPaint edits"), 2026-10-07, second pass

Second, independent adversarial pass ("try again"). It starts from the fragment as left by the first pass (14:19) and re-does
every check from scratch. Edits were made only to the fragment, to fig/6_xgpaint/02_tau_fork_over_upstream.png (regenerated)
and to this file. The first pass's notes are summarised at the end.

## What was checked (second pass)
- Code excerpts: all 12 `<pre class="code" data-src=...>` compared with the current files at the stated lines after
  HTML-unescaping (CRLF stripped for julia_env/dev): 12/12 verbatim, before and after the edits.
- Vendored fork copy SBI_analysis/so_32k_independent_noise/vendor/XGPaint/src = 35f79ad for 15/15 files (git hash-object vs
  git rev-parse); julia_env/dev profiles_tau.jl, profiles_y.jl, model.jl = 29ec48a; julia_env/dev = v0.4 (1902953) for 12/13
  files, util.jl differs. Trees ce1d239 = 35f79ad = 3b57cacb; tag object points at 35f79ad; manifests pin 3b57cac.
- Git: git show of f16a7f3, 9f37b1e, 077b849, b459728, b924ab1 (merge diff), e67a8a1, cbf5cbd, 71646e2, ce1d239, 5dd0b57,
  07fc686, 35f79ad, 1660e08, 71bc3fd; diffstats (6 files 570+/95-; e67a8a1 448+); reflog; every file:line reference in the
  ten xgpaint asides against 35f79ad numbering (profiles.jl l.56-67, 70-76, 79-113, 150, 152-172, 175-220, 222-273, 278-281,
  284, 358, 364-369, 373, 431-433; profiles_y.jl l.63-70, 84, 100-106, 147-221; profiles_tau.jl l.87-107, 106; model.jl l.8-23;
  profiles_frb.jl l.3) and the 5dd0b57 broken lines (l.81, 85, 294, 382-383).
- Figures: 10 PNGs exist and open; each looked at. 03/08/09 PNG+PDF byte-identical to truncation_comparison/spectra/*.
  10 is a five-window composite with no single-panel original in its plots directory (kept; caption fixed).
- Numbers traced (> 60): tau ratios 0.1451/0.3201/0.5690/1.2803/2.2761, z = 1.65; DM 58.7/183.5 and 63.4/111.5; f_b 0.158065;
  m_per_e 1.136908; 0.7916; 0.88; 0.809; 1.112; 0.6016; M2_TO_PC_CM3 3.2408e-23, ratio 1.5027e28; r200c_comoving 2.2e-16;
  sign-bug 1.75e25-2.69e44 (36 points); quadrature 6.76e-6 / 1.98e-5 / 7.8e-16 and audit 7e-14; audit A 6.8e-14, 4.4e-16,
  8.9e-5, 6.0e-4, CRC32c 8cfcf4e7 / 26851 B; verifier z-grid errors (recomputed from v1c_itp_vs_direct_grid.csv), 850 and
  79,302 halos (sum of v4b Y_budget N), C_l ratios and their definition, mean y 0.54 % / 0.43 %, 2.8e-8, dz 0.0196; kSZ step4
  1.1831/1.0386/0.9996/0.9989/0.9998; kSZ f_b table 0.370/14.96/0.429/0.199, 40.398 = 6.356^2 (binned_spectra_fb.csv), f_b
  0.0486/0.3089; budget 0.809; truncation 0.868..1.003, 0.987, profile ratios; class_sz 0.4 %, 0.3 %, mu_e 1.14;
  catalogue 4.46e12-2.57e15; B16 Fig 5 0.984-1.073 / 0.750-0.831; Lee22 wrapper 2e-6 / 1e-9 / 300 bins / 2e-16; 22 pc cm^-3;
  kSZ painter 527/509/5098/8343 s, 7,090,669,884, 79,302, 3.3e-16, 1.4e-16, 7.4e-11; tSZ painter self-test 300 halos, 2e-3;
  ~650 exact-table nodes (ln(4e5)/0.02); 64k engine node sets, 0.004 thresholds, dln z = 0.067; 137 mis-encoded lines.
- Physics: comoving-to-physical column (1+z)^2; f_b; m_per_e = m_e + 2 m_p/(1+X_H); WebSky 1 - Y_p/2; 0.5176 = mu/mu_e;
  Lee22 P0 = N n0 200 m_per_e/(0.9 X_H m_p); exponent conventions (upstream B16 (beta-gamma)/alpha with beta = -3.83 equals
  -(beta+gamma)/alpha with +3.83; 9f37b1e B12 beta = -gamma - alpha beta_B gives +beta_B); chord factor 2X sqrt(1-(t/tmax)^2) = 2L;
  observed DM 1/(1+z).
- HTML: one <section id="xgpaint" class="topic">, all tags closed, only contract classes, no script/style, all img src exist.

## Corrections made in this pass
1. Section 4.1 low-z errors: "between z = 0.02 and 0.05 it is at most 6 %" was wrong for Lee22: v1c_itp_vs_direct_grid.csv
   gives up to 5.05 % (B12) and 7.24 % (Lee22, x = 3.9, M = 10^14.9675) at z = 0.02. "At z = 0.01 ... a factor of about 9"
   made precise: |itp/direct - 1| up to 8.46 (Lee22) and 4.04 (B12).
2. Section 4.5 and Fig 6.10: the 300-of-300-bin histogram equality was presented as a test of XGPaint's get_params/ne2d/
   compute_DM pipeline. LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md l.46 says the histograms were made in spherical mode,
   where both products share the project chord quadrature (Battaglia16DensityDMProfile, generator l.200); they test the
   get_params mapping, while XGPaint's projected rho_2d/ne2d/compute_DM chain is tested only by the 5-point self-test. Rewritten.
3. Fig 6.10 caption: added the black IllustrisTNG curve and the Delta p/p_TNG panels, which the caption omitted.
4. Fig 6.7 caption: the Lee22 outliers are the four 10^15 Msun halos (z = 0.1, 0.5, 1, 2), one cell above the
   10^14.97 Msun shape clip (painted_profile_vs_independent_los.csv); defined "shape clip".
5. Section 4.1 and Fig 6.6: the quoted C_l ratios (1.16, 1.10, ...) are ratios of C_l averaged over l +- 5
   (v4b_lowz_impact.py l.68), not single-l values (single-l B12 at l = 10 is 1.23); stated in text and caption.
6. B12 sign bug "in no product": the evidence covers the tables behind the products in this report (audit A scope: the y maps),
   not every table built in March-April. Reworded in the aside, summary table and key aside; added a new piece of evidence:
   the production B12 cache stores "prof_logthetas", a key name introduced by e67a8a1 (upstream, b924ab1 and the loose
   intermediate profiles.jl write "prof_logθs"), written 2026-04-29 16:08 +0300, 17 h before the e67a8a1 commit.
7. Key aside: "agrees with upstream to 10^-15" -> 7e-14 or better (audit A measures the rtol change at <= 7e-14).
8. Summary table: "grid accuracy checked per painter" was unsupported -> "loaded tables unaffected, rebuilt default-grid tables
   not checked in general"; rtol cell 7.8e-16 -> 7e-14 (audit A value); "kSZ power x its square" -> per-halo.
9. ce1d239 aside: the 7.4e-11 kSZ rebuild comparison was offered as a check of rebuilt tables; both builds use the same
   ce1d239 defaults, so it is a reproducibility check only. Said so; named the default-grid tables (kSZ paint_ksz_websky.jl l.77,
   truncation build_truncation_caches.jl l.28; logs/build_caches.log shows N_logM = 128).
10. 07fc686: it deleted the broken line "N_logθ=256)" instead of restoring "N_logθ = length(logθs)" (it did not delete the
    correct line). 1660e08: one test added to an existing set plus one new test set, not "two test sets".
11. e67a8a1 (a): "This is a free-electron fraction" -> "It acts as a free-electron fraction" (intent undocumented, consistent
    with Open issues). (e): cleanup_negatives! was also rewritten for leading/all non-positive columns (upstream indexed out of
    bounds there); aside line refs l.150-170 -> l.152-172 and l.175-220 (l.150 is ce1d239's helper).
12. compute_DM code-note: the 1/(1+z) explanation gave only the frequency shift (which alone would give (1+z)^-2); added the
    (1+z) stretching of the delay.
13. FRB generator "not handled" aside: it normally loads an existing build_interpolator table (builds at l.911 only on overwrite)
    and re-interpolates on the table's linear z axis (l.948, l.984).
14. Clarity: Lee22 and IllustrisTNG defined at first use (Section 1); M_cut defined; FLAMINGO and f_NL named; "audit A" and
    "the verifier" defined with paths at first use; Fig 6.1 caption notes 1660e08.
15. Fig 6.2 regenerated with explicit y tick labels (0.15 ... 2); the log axis showed only "10^0". Done by running
    make_figures.fig_tau_ratio() unchanged through a wrapper that only sets FixedLocator/FixedFormatter on the y axis before
    save (wrapper kept in the session scratchpad; scripts/xgpaint/ was outside this pass's edit scope). Same data and markers.

## Remaining concerns
- idark copies (runtime/XGPaint, /home/kristero10/XGPaint) not re-read (no ssh attempted in this pass; the first pass's ssh
  timed out). Evidence: tag message, PBS comment, 7f5b73a, and the 2026-09-14 sha256 of src/XGPaint.jl, which identifies
  "e67a8a1 or later", not 35f79ad.
- Effect of the linear-z table error on the FRB DM PDFs and y x DM spectra is unmeasured; the truncation-comparison maps
  (Figs 6.8/6.9) also lack exact low-z tables (their sphere/projected ratios share the error).
- No accuracy check of default-grid (128 log M) tables against the finer grid exists.
- Origin of the 0.9 in ne2d undocumented.
- Fig 6.10 is a five-panel composite (no single-panel original exists); kept as one figure.
- Prose is ~8,900 words including captions, asides and code-notes (above the ~7,000 guide); not trimmed, to keep verified content.

## First pass (same day, 14:20), for reference
It had already corrected: the f_b power ratio (40.40 at every l by construction), "tightens" -> "relaxes" for eps() -> 1e-12,
the factor-35 low-z regime (z < 0.01), the kSZ low-z band, the 5dd0b57 broken-line count and reflog date, 71bc3fd's
PowerLawParam, the chord-mean wrapper's get_params, the Lee22 wrapper's dependency list, the f_b row of the summary table,
the CLASS-SZ f_free/0.4 %/0.3 % wording, the kSZ painter's slice calls, the copies table (two extra profiles.jl copies and the
FLAMINGO reference profiles_y.jl), the painter cosmology of Figs 6.3/6.8/6.9, the 64k engine notes, and many definitions.
