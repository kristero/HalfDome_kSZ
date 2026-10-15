# Dossier: Appendix "XGPaint edits" (section s6_xgpaint.html, id="xgpaint")

Written 2026-10-07. Everything below was read or measured in this session; paths are absolute unless noted.

## Repositories and copies
- Fork dev checkout: /home/kn18001/.julia/dev/XGPaint, branch `cluster` at 1660e08, clean, in sync with origin/cluster
  (git@github.com:kristero/XGPaint.jl). upstream remote = WebSky-CITA/XGPaint.jl, upstream/main = 29ec48a (2025-08-30).
- Trees: ce1d239 = 35f79ad = 3b57cacb5f824269f1cae22e7fd2085788d14eee. 5dd0b57 befd21cc, 07fc686 cc958324, 1660e08 5bd5d512.
  `git diff --stat ce1d239 35f79ad` is empty; 35f79ad..1660e08 = test/runtests.jl only (+19 lines).
- Net fork vs upstream (29ec48a..1660e08): src/XGPaint.jl, model.jl, profiles.jl, profiles_frb.jl (new), profiles_tau.jl,
  profiles_y.jl, test/runtests.jl; 589 insertions, 95 deletions. Helpers (rho_crit, R_Delta, angular_size, util.jl) unchanged.
- Tag validated-tsz256-20260922: annotated tag object bb9f74ef, points at 35f79ad; message says tree identical to ce1d239,
  the runtime/XGPaint snapshot of flamingo_tsz_comparison_20260914 on idark.
- Local `main` = ce1d239 is "ahead 1" of origin/main (71646e2): ce1d239 itself is not on GitHub main, but its tree is on GitHub
  as 35f79ad (cluster, tag). Local branch `adrian` = 71bc3fd (2026-03-03, Kristers, "Adrian fixes"): ahead of upstream/adrian by 1,
  not contained in main or cluster.
- In-repo copies (hash-object comparison of every src/*.jl, CRLF stripped where present):
  - SBI_analysis/so_32k_independent_noise/vendor/XGPaint/src: 15/15 files identical to ce1d239/35f79ad (tracked in git, last
    touched by HalfDome commit 9047277 2026-09-13). Used as the verbatim source for fork excerpts in the section.
  - fnl_comparison/pair_100_120/pipeline_snapshot/XGPaint/src: 15/15 identical to ce1d239/35f79ad (untracked; its manifest
    points to idark /lustre/work/kristero10/fnl_pair_100_120_20260929/pair_100_120/pipeline_snapshot/XGPaint).
  - julia_env/dev/XGPaint.jl/src: 13 files (no healpixworkspace.jl, no profiles_frb.jl), CRLF line endings; 12/13 identical to
    upstream 1902953 (v0.4, 2025-08-26), util.jl differs (local edit: resol.numOfPixels instead of resol.npix and polar-ring
    ranges); profiles_tau.jl, profiles_y.jl, model.jl identical to 29ec48a. Untracked; julia_env/Manifest.toml points to
    idark /home/kristero10/XGPaint/XGPaint.jl, so this copy is not what julia_env resolves. Used for the upstream excerpts.
- Manifests: tSZ_64k_flat_prior_SO_baseline_deproj0/julia_env and SBI_analysis/tsz_local_profile_tests/julia_env pin
  git-tree-sha1 3b57cac, repo-rev validated-tsz256-20260922 (Julia 1.12.2). SBI_analysis/flamingo_tsz/runtime_env -> idark
  /lustre/work/kristero10/flamingo_tsz_comparison_20260914/runtime/XGPaint. julia_env -> idark /home/kristero10/XGPaint/XGPaint.jl.
  Default local env ~/.julia/environments/v1.12 -> /home/kn18001/.julia/dev/XGPaint.
- idark: ssh.exe timed out on 2026-10-07 (connect to idark.ipmu.jp port 22: Connection timed out), so idark copies were not re-read.
  Indirect evidence: SBI_analysis/flamingo_tsz/cluster_operator_probe.toml (2026-09-14) records sha256 of
  /home/kristero10/XGPaint/XGPaint.jl/src/XGPaint.jl = aadae01f..., equal to src/XGPaint.jl of e67a8a1..35f79ad (upstream and
  f16a7f3 give 0ed06d91...). frb_map_generation/lee22_tsz_corrected/ksz/run_ksz_websky_idark.pbs l.17 and commit 7f5b73a state
  the idark tSZ runtime uses "the same XGPaint 35f79ad". Tag message: runtime/XGPaint tree = ce1d239.

## Commit-by-commit (git show)
- f16a7f3 (2026-02-09): PowerLawParam + powerlaw_value (model.jl); 15 kwargs per profile (BattagliaTauProfile[Physical],
  Battaglia16ThermalSZProfile, BreakModel); get_params dispatches on the concrete type (generic AbstractGNFW fallback removed);
  profile_grid default logθ_min -16.5 -> -15.7; build_interpolator gains logM_max kwarg. Defaults reproduce the hard-coded numbers.
- 9f37b1e (adrianbayer, 2026-02-27): tau beta -3.83 -> +3.83; generalized_nfw exponent (β-γ)/α -> -(β+γ)/α; B12 β = -γ - αβ_raw
  (wrong sign, see below); rho_2d now P0 I f_b ρ_crit(z) r200c physical (was ρ_crit/(1+z)^3/h^2 × r200c(1+z)); ne2d factor drops
  /h^2; ρ_crit_comoving_h⁻² and r200c_comoving deleted from profiles_tau.jl. 077b849: comment only.
- b459728 (2026-03-02): f_b multiplied in ne2d (old units). b924ab1 merge took Adrian's ne2d (f_b once, in rho_2d) and set
  beta_amp default +3.83; B12 β = -γ - αβ_raw => exponent +β_raw (divergent) until e67a8a1.
- e67a8a1 (2026-04-30): B12 β = αβ_raw - γ (with comment); ne2d returns 0.9 * result/factor; profiles_frb.jl (AbstractFRBProfile,
  HaloDMProfile, BattagliaFRBProfile, object_size, compute_DM, M2_TO_PC_CM3 = 4.87e5 hard-coded); r200c_comoving in model.jl
  (exported); prepare/evaluate/fill_profile_slice!; profile_grid flattened (z, M) loop with up to 8×nthreads chunks; threaded
  transform_profile_grid! and cleanup_negatives!; replace_nonpositive_with_floor! (floor = 1e-6 × smallest positive) behind
  XGPAINT_CLEANUP_NONPOSITIVE (default true); non-finite check; logging; cache key prof_logθs -> prof_logthetas with a broken
  fallback "prof_log?s"; defaults still N_logM 256, N_logθ 512, logM_min 11, pad 256 (= the B12 production cache grid).
- cbf5cbd (2026-07-01, cluster): profile_grid N_logM 128, N_logθ 256, logM_min 12; build_interpolator N_logθ 256 pad 128;
  LOS quadrature rtol 1e-3 order 7.
- 71646e2 (2026-07-13): M2_TO_PC_CM3 = ustrip(u"pc*cm^-3", 1u"m^-2") = 3.2407792894443655e-23 (measured); 4.87e5 was 1.5027e28 x too large.
- ce1d239 (2026-07-13): profile_grid defaults N_logM 128, N_logtheta 256, logM_min 12; build_interpolator pad 128 (N_logtheta 512);
  _thread_storage_count() = maxthreadid(); cache-key fallback fixed (prof_logθs or error); LOS rtol eps() -> 1e-12 (order 9).
- 5dd0b57 merge: profiles.jl does not parse (3 broken lines); 07fc686 fixes syntax but drops N_logθ = length(logθs);
  35f79ad restores it (tree = ce1d239). 1660e08: tests (r200c_comoving, BattagliaFRBProfile compute_DM vs Unitful).

## Measured in this session (scripts/xgpaint/measure_xgpaint_edits.jl, Julia 1.12.2, 53 s)
Upstream 29ec48a functions transcribed verbatim into a module next to the fork; painter cosmology Ω_c 0.261, Ω_b 0.049, h 0.68.
- τ(fork)/τ(upstream) = 0.9 f_b (1+z)^2 exactly (max deviation 0.0 over M 1e13-1e15, z 0.01-3, x 0.1-3.5); f_b = 0.158065;
  ratio 0.1451 at z=0.01, 0.3201 at z=0.5, 0.5690 at z=1, 1.2803 at z=2, 2.2761 at z=3; equal to 1 at z = 1.65.
- DM: fork compute_DM vs the public-v0.4 fallback wrapper of generate_halfdome_z1_dm_mass_windows.jl l.80-107: same ratio.
  Example M=1e14, x=1: fork 58.7 pc/cm3 at z=0.5 (fallback 183.5), 63.4 at z=1 (111.5).
- Projected B12 y shape fork/upstream = 1 exactly; 3-D B12 and B16 shapes identical (max |ratio-1| = 0).
- b924ab1 sign bug: LOS bug/correct = 1.7e25 (1e13, z 0.1, x 0.1) to 2.7e44 (1e15, z 2, x 3.5); doc quotes 10^29-10^40.
- Quadrature: cbf5cbd (rtol 1e-3, order 7) vs HEAD: max 6.8e-6 (B12 y), 2.0e-5 (B16 τ); eps() vs 1e-12: <= 7.8e-16.
- r200c_comoving upstream vs fork: 2.2e-16. m_per_e/m_p = 1.136908; XGPaint n_e = 0.79162 ρ/m_p.

## Numbers taken from existing reports (source)
- Audit A (frb_map_generation/outputs/lee22_tsz_verification_20260929/A_xgpaint_audit/README.md): B12 production cache
  /home/cbllover/HalfDome/cached_tSZ_HalfDome_cosmo_base_cosmo_fid.jld2 built 2026-04-29, grid 512 θ × 256 z × 256 logM [11,15.7],
  pad 256 (= e67a8a1 defaults); cache values = current code to 6.8e-14; Lee22 cache bit-for-bit; precompile caches 2026-09-02..09-25
  record profiles.jl 26851 B CRC32c 8cfcf4e7 and profiles_y.jl 8708 B 2dca2a56 (= HEAD = ce1d239); 5dd0b57 never compiled;
  map repaint <= 4.4e-16; interpolator vs independent LOS: B12 8.9e-5, Lee22 6.0e-4; XGPaint direct vs independent 1.4e-10/2.2e-10;
  ρ_crit radiation +4e-5 (z 0.1), +8e-4 (z 2), +1.1e-3 (z 3).
- Verifier (verify_A_xgpaint_audit/README.md): 850 halos at z < 0.0206; z<0.01 factor 1-35 at x>=1 (7-9 % at x=0.1); z 0.02-0.05
  1-6 %; 79,302 halos z<0.1; mean y +0.54 % (B12) / +0.43 % (Lee22); C_l stored/exact B12 1.16/1.10/1.04/1.012/1.0008 at
  l=10/20/30/50/100, Lee22 1.037/1.029/1.010/1.002/0.9993; |Δ|<1e-3 for l>=100. Linear z grid Δz = 0.0196.
- kSZ low-z (frb_map_generation/outputs/lee22_ksz_debug_20261002/verify_B1/step4_map_level.txt): prod/fixed 1.1831 (l 20-40),
  1.0386 (41-81), 0.9996 (82-165), 0.9989 (166-340), 0.9998 (341-660).
- kSZ f_b (truncation_comparison/KSZ_FB_WEBSKY_CLASSSZ_20260919.md): rms ratio 6.356 = 1/f_b; power 27-40x = 1/f_b^2 = 40.1;
  l=1000: with f_b 0.370, without 14.96, CLASS-SZ 0.429, WebSky 0.199 uK^2.
- Truncation (truncation_comparison/TSZ_KSZ_TRUNCATION_COMPARISON_20260917.md): tSZ sphere/projected 0.987 (l 32) ... 1.000;
  kSZ 0.868 (32), 0.877 (81), 0.895 (330), 0.921 (840), 0.960 (2141), 0.983 (3417), 1.003 (5454); profile level B12 x=3 0.81-0.84,
  x=3.9 0.30-0.33, disc 0.972-0.991; B16 x=3 0.68-0.74, x=3.9 0.23-0.26, disc 0.875-0.935; self-test 1e-8, caches 1e-7.
- kSZ budget (frb_map_generation/outputs/lee22_ksz_debug_20261002/synthesis/budget.txt): XGPaint 0.9 vs WebSky (1-Y/2) = 0.809 in power.
- kSZ painter speed (HalfDome commits 77ac291, 7f5b73a): job 600626 paint 509 s, total 527 s vs 5098 s (600487, Julia 1.6) and 8343 s
  (600479); spectra <= 7.4e-11; single-thread byte-identical; 16 threads <= 3.3e-16.
- Lee22 wrapper (frb_map_generation/LIKE_FOR_LIKE_SPHERICAL_R200C_20260916.md l.46): compute_DM on wrapper = Lee22 code to 2e-6 (column),
  1e-9 (density); 0 of 300 bins differ; means to 2e-16. 0.9 X_H m_p/m_per_e = 0.6016316602 (lee22_fig5_audit_20260922/REPORT.md).
- Lee22 doc (frb_map_generation/lee22_tsz_corrected/doc/lee22_calibration.html l.89-95): β_XG = αβ' − γ reproduces Lee22 fit curves to
  0.3-3 %; B16 typos (no f_b, −(β−γ)/α) corrected by f_b ρ_crit ρ_fit and −(β+γ)/α; sign bug "would change y by 10^29-10^40".

## Open
- FRB DM ray generator and DM map painters still use build_interpolator caches on the linear z grid with no exact low-z tables;
  effect on DM PDFs / y x DM not measured in any file read.
- Origin of the 0.9 in ne2d is not documented in e67a8a1.
- idark copies not re-verified today.

## Section build
- Template scripts/xgpaint/section_template.html -> scripts/xgpaint/build_section.py inserts verbatim excerpts (CR stripped,
  HTML-escaped) -> sections/s6_xgpaint.html. Fork excerpts quoted from the git-tracked vendored copy
  SBI_analysis/so_32k_independent_noise/vendor/XGPaint/src (byte-identical to 35f79ad); upstream excerpts from
  julia_env/dev/XGPaint.jl/src (profiles_tau.jl, profiles_y.jl, model.jl identical to 29ec48a).
- Figures (fig/6_xgpaint/): 01 lineage, 02 tau ratio, 04 sign bug, 05/06 low-z (re-plotted from the verifier's saved data,
  split into two plots), 07 painted vs independent (re-plotted from audit A csv) are made by scripts/xgpaint/make_figures.py;
  03 (ksz f_b), 08/09 (truncation) copied with PDFs from truncation_comparison/spectra/; 10 downscaled copy of
  frb_map_generation/outputs/publication_comparison_20260916_sphere_1r200c/plots/halo_pdf_sphere_1r200c_b16_lee22_tng_xgpaint_to_1e14.png.
