# Dossier: 1. Lee22 profile corrections (section s1_lee22.html, id="lee22")

Written 2026-10-07 for the HalfDome overview report. Notes on sources, decisions and numbers; the section is the product.

## Primary sources read
- `frb_map_generation/lee22_tsz_corrected/doc/lee22_calibration.html` (doc v4, commit 6648c4b, 2026-10-01): final statement.
- Verification outputs: `outputs/lee22_verify_v3_20261001/` (derivation/hbook.txt, Fn_integrated.txt, group_particles_provenance.txt,
  fig2fig5_content/*.txt, code_snippets/b12_cache_sign_test.json, completeness_critic/edits.json),
  `outputs/lee22_verify_v4_20261001/` (frb_decades/report.txt + csv, models_grid/*.txt|json, revision_audit/*).
- Earlier reports: `LEE22_IMPLEMENTATION_CHECK_20260916.md` (n0 pivot 1.42, shape clip, divergence above ~2e15 Msun),
  `LEE22_NORMALIZATION_EQ7_EQ9_NOTE_20260917.md` (readings A-E, 0.602 = 0.9 X_H m_p/m_per_e, spherical DM test),
  `LEE22_FIG5_SHAPE_CHECK_20260919.md`, `outputs/lee22_fig5_audit_20260922/REPORT.md` (38 % deficit of xgpaint_ne2d vs paper),
  `LEE22_NORMALIZATION_FORENSICS_20260929.md` (M1-M4, C(z)), `LEE22_TSZ_VERIFICATION_20260929.md` (F_P, FLAMINGO slopes, low-z).
- Code: `lee2022_frb_dm_profile.jl`, `lee2022_tsz_pressure_profile.jl`, `lee22_tsz_corrected/paint_tsz_variants.jl`,
  `test_bookkeeping_option.jl`, `tng_windows.py`, `doc/check_beta_convention.py`, `doc/fig_lee22_fig2_fig5_replica.py`,
  local copies of group_particles `y_prof.cpp`, `ne_prof.cpp` in `lee22_tsz_corrected/verify_v3/derivation/` (untracked copies,
  byte-identical to GitHub master per group_particles_provenance.txt).
- XGPaint dev checkout `/home/kn18001/.julia/dev/XGPaint` (branch cluster, HEAD 1660e08, clean): git log/show of 9f37b1e, b459728,
  b924ab1, e67a8a1, ce1d239, 07fc686, 35f79ad; `upstream/main` for comparison.

## Local rerun done for this report
- `scripts/lee22/test_bookkeeping_option_run_20261007.log`: rerun of `lee22_tsz_corrected/test_bookkeeping_option.jl` with Julia 1.12.2
  (depot stacked as scratchpad:/home/kn18001/.julia so no compile cache was written into the shared depot). PASS; worst relative errors
  2.2e-16 (a3) and 6.7e-16 (physical); F_n = 5.5698; table (1+z)^3/5.4 vs C(z): ratio 0.652/0.761/0.932/1.150 at z = 0/0.35/1/2.
- `scripts/lee22/plot_lowz_fix_ratio.py` -> fig 15 (reads saved spectra; numbers identical to analysis/yy_power_stdout.txt).

## XGPaint findings (stress in section)
- No XGPaint edit was made for Lee22. All Lee22 code lives in the repository (`lee2022_frb_dm_profile.jl` header says so).
- 9f37b1e (Adrian Bayer, upstream branch `adrian`, 2026-02-27; merged into the fork in b924ab1 2026-03-02): generalized_nfw exponent
  (β-γ)/α -> -(β+γ)/α; B16 rho_2d gets f_b and physical rho_crit, r200c; B16 β sign -3.83 -> 3.83; B12 conversion γ-αβ -> -γ-αβ
  (sign bug: exponent +β). Fixed in e67a8a1 (fork, 2026-04-30): β = αβ_raw - γ. B12 maps not affected (b12_cache_sign_test.json:
  stored/correct 1.000000002, stored/buggy ~1e-29..1e-33).
- Correction found while writing (2026-10-07): upstream main (29ec48a) already had the Battaglia16 exponent right, because it uses
  β = -3.83 with (β-γ)/α = -(|β|+γ)/α. 9f37b1e only re-expressed it (same numbers). What 9f37b1e changed numerically for B16 is the f_b
  and the units: upstream rho_2d = P0 LOS rho_crit(z)/(1+z)^3/h^2 * r200c(1+z), with ne2d dividing by m_per_e/h^2, i.e. the comoving
  column (1+z)^-2 times the physical one. Fork/upstream B16 column = 0.9 f_b (1+z)^2 (code reading, no report quantifies it). The doc v4
  sentence "XGPaint fixes both typos" is right about the current code but the exponent was never wrong in XGPaint.
- Consequence for Lee22 history: the first Lee22 code (2c6d3b6 2026-08-27 .. 7f70174 2026-08-30) used upstream's legacy
  `beta_standard = gamma - alpha*beta_prime` (comment: "(beta_standard - gamma)/alpha = -beta_prime", i.e. written for the old exponent)
  and called XGPaint._nfw_profile_los_quadrature; with the fork's -(β+γ)/α exponent that is (1+x)^{+(β'+0.6)} -> divergent LOS.
  2771265 (2026-08-31, "lee22 fix for exp DM values") replaced it with a profile-owned integrand with -β'. 6fa5373 (2026-09-17) then
  passed β_XG = αβ' - γ through get_params.
- b459728 (fork, "f_b addition" in ne2d) superseded at the merge b924ab1 (ne2d without f_b, rho_2d with f_b): f_b applied once.
- e67a8a1 (fork): ne2d multiplied by 0.9 (upstream main has no 0.9). Previous production Lee22 density factor 0.6016 contains it
  (0.9 X_H m_p/m_per_e); final :literal+:bookkeeping_physical cancels the 0.9 in get_params (P0 has 1/(0.9 X_H m_p)). B16 painted in
  HalfDome carries the 0.9. e67a8a1 also added prepare/evaluate/fill_profile_slice! (used by the low-z exact tables).
- profile_grid z grid LinRange(1e-3, 5, 256) (dz 0.0196) is upstream behaviour; ce1d239 changed only N_logM 256->128, logM_min 11->12,
  N_logtheta 512->256 defaults. Interpolation: cubic B-spline in log y (profiles.jl:489). Workaround project-side.
- P_e = 0.5176 P_th hard-coded (profiles_y.jl:128), upstream; Lee22 pressure wrapper divides P0 by 0.5176.
- 07fc686/35f79ad (2026-09-24): print-statement typos; maps repaint to <= 5.6e-16 (tSZ verification report).

## Numbers used (source)
- Final factors and 0.1795, 5.57: doc v4; test log; check (0.6774^3*0.76^2).
- M1 code identities: hbook.txt (a^3/h^3, a^2, ratios at z=0,0.47,1.05,1.48).
- Emulation, CAMELS F_P/F_n table, FLAMINGO slopes: doc v4 section 2-3; LEE22_TSZ_VERIFICATION_20260929.md.
- β' convention: outputs/lee22_corrected_20260930/doc/check_beta_convention.txt; lee22_fits_vs_drawn.txt.
- Replica ratios: verify_v4 models_grid/fig1_check.txt, fig2bc_check.txt; doc captions.
- Grid ratios: doc v4 Fig. 2d/2e captions; grid_check.json (pipe/cor n 3.351/0.419/0.124; z_equal 0.4964).
- TNG windows and FRB ratios: verify_v4 frb_decades/report.txt.
- Low-z: verify_A_xgpaint_audit/v4b_lowz_impact.json (Y budget), selftest_600152 log, yy_power_stdout.txt, y_600153 log (79,302).
- tSZ power: analysis/yy_power_stdout.txt (doc table), pub captions (band-centre variant).
- Painter vs HalfDome cosmology: tau_ksz_comparison/README.md (h 0.6774, Om 0.3089, Ob 0.0486); painters h 0.68, Ob 0.049, Oc 0.261;
  ~2 % B12/FLAMINGO effect (LEE22_TSZ_VERIFICATION_20260929.md sec. 3).

## Disagreements resolved
- Fig. 1a model/TNG at 0.3<x<1: doc caption "0.99 and 1.04" vs verify_v4 fig1_check "0.981 and 1.033": used verify_v4.
- y power band centres: doc table (ell 300: 3.03) vs pub caption (ell 273: 3.17): same maps, different bands; doc table used.
- Fit range: 10^14.8 in code constants/comments is the old misreading; paper bins end near 10^14 (10^14.2 at z<=0.2). Kept as the
  production clip threshold; stale comments at lee2022_tsz_pressure_profile.jl:9 and :94 ("20 TNG300 snapshots").
- Snapshot count: paper says "20 full snapshots", of which 10 have z <= 2 (completeness critic edit).
- paint_tsz_variants.jl:12 says caches "interpolate linearly in z"; the grid is linear (uniform) but XGPaint interpolates with a cubic
  B-spline in log y. Section says uniform grid + cubic B-spline.

## Open
- Lee22 density n200 code not public (a^3, 1/h^3, X_H^2 inferred); ask B. K. K. Lee / L. Thiele.
- Konietzka window convention to confirm with R. Konietzka.
- Mass range above ~10^14 h^-1 Msun extrapolated; hybrid threshold 10^14.8 is ~0.8 dex above fitted bins.
- Corrected density runs 27-41 % low below ~10 pc cm^-3 (about 5 % of hits).
- tSZ_64k dataset still on coarse low-z tables (doc v4 open item, as of 2026-10-01).
