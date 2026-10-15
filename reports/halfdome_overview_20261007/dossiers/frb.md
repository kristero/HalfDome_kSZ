# Dossier: Section 2, FRB DM PDFs and tSZ x FRB (section id "frb")

Fragment: `sections/s2_frb.html` (built from `scripts/frb/s2_frb_template.html` by `scripts/frb/build_section.py`, which
pulls all 12 code excerpts verbatim from the current files). Figures: `fig/2_frb/01..14_*.png` (+ PDF beside each except 08),
copied/downscaled by `scripts/frb/copy_figures.py` (12_yy_power_pub.png is a 400 dpi pdftoppm render of the vector PDF, the
source PNG was 958 px wide). Rebuild: `python3 scripts/frb/build_section.py` from anywhere.

## Story chosen
1. Rays -> hits -> R200c-sphere chord DM -> per-window histograms -> mean DM per hit (why per hit: TNG hydro halos ~0.8x as
   abundant as HalfDome gravity-only halos).
2. TNG window reading ('AtoB' = [10^A, 10^(B+1))) is the pivot of the FRB comparison; re-verified here from the raw arrays.
3. Results: corrected Lee22 flat 0.98-1.02; pipeline 2.85 -> 1.03 (cancellation, not agreement); B16 1.02 -> 0.86; clusters
   (1e14-1e15) corrected 1.04-1.11; low-DM deficit 27-41 % below ~10 pc/cm3.
4. Analytic halo model (Torkamani compound Poisson) reproduces the rays; TNG-cosmology run gives 0.95/1.01/1.03.
5. Cross-correlation estimator: kernel-weighted full-sky DM map = ensemble mean; Legendre transform with beam on y; exact
   annulus means; chi2 with real jackknife covariance; Hartlap 0.871/0.600.
6. Results: previous pair excluded; Planck halo-only models all ~ w=0 (large-angle signal missing); ACT disfavours full
   corrected Lee22 pair (dchi2 14 / 8.4) because of y amplitude; y power 2.3-3x FLAMINGO, from >1e15 halos (72 % at l=300).

## Facts verified (task list)
- 'massAtoB' = [10^A, 10^(B+1)): re-run read-only on `frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM/`
  (arrays 55 x 120000): max |DM(10to14)-DM(10to12)-DM(13to14)| = 1.14e-13, 2.27e-13, 4.55e-13, 9.09e-13, 4.55e-13, 4.55e-13
  at z_s 0.1/0.5/1/1.5/2/3. Hit % 1e13-1e14 decade 6.60/29.65/46.33/53.46/56.25/57.66; DM(10to14)-DM(10to13) 2.01/8.31/11.73;
  DM(10to15)-DM(10to14) 0.065/0.231/0.263. Exact mean/hit of decade 77.54/87.29/101.99/112.42/117.35/120.81 (binned CSV
  77.57/87.30/101.99/112.42/117.35/120.81).
- Corrected 0.98/1.00/1.02/1.02/1.02/1.01; B16 1.02 -> 0.86; pipeline 2.85 -> 1.03: `outputs/lee22_pub_figures_20261001/captions.txt`
  and `outputs/lee22_pdf_zseries_20261001/analysis/halo_pdf_zseries_stats.csv`.
- chi2 (Planck/ACT, no Hartlap) from `outputs/lee22_corrected_20260930/analysis/tsz_x_frbdm_corrected_chi2.csv`:
  b12 x b16_sphere1 20.07/9.03; b12 x lee22_noconc_phys_sphere1 20.23/11.54; lee22p_m1_b12hi x phys 21.01/14.21;
  lee22p_m1_calib x phys_calib 20.63/10.11; lee22p_m1 x phys 24.03/23.04; lee22p x lee22_noconc_sphere1 58.82/119.38; w=0 21.85/9.71.
- Hartlap: `plot_pub_figures.py:187` (71-8-2)/70 = 0.871, (31-11-2)/30 = 0.600; captions give chi2 -> chi2*h and p.
- y-power vs FLAMINGO L2p8_m9 lc0 (pixwin^2 divided): captions.txt (band centres 273/980/2931/5068) and yy_power_stdout.txt.
- tSZ mass split: captions_tsz_mass.txt.

## Discrepancies / superseded numbers (say which is final)
- `outputs/lee22_corrected_20260930/analysis/halo_pdf_corrected_stdout.txt` (2026-09-30 11:08) uses the '13to14' array as TNG
  (110.3/123.3/138.7). SUPERSEDED by `halo_pdf_corrected_lee22_stats.csv` (2026-10-01, decade: 77.6/87.3/102.0).
- WHAT_CHANGED_FOR_TNG_AGREEMENT_20260917.md table ("xgpaint_ne2d = 1.00 of TNG", 138.6 at z_s=1) is against the '13to14'
  array; against the true decade the same density is 1.36. Its own 2026-09-22 qualification already withdrew the eq.9 claim.
- Memory note "corrected mean/hit vs TNG -31..-22 %" was against '13to14'; superseded.
- yy ratio at l~300 for Lee22 x (1+z)^2: 3.03 (stdout, finer band) vs 3.17 (pub figure band centred on 273). Both correct.
- Finite-volume 99.9 % ranges: Lee22 doc v4 1.1-2.9 / 4.7-12.2 / 6.4-17 % vs analytic README 1.1-2.9 / 4.7-12.4 / 6.3-17.0 %.
  Used the doc (final v4).
- Figure 2.9 (finite-FRB realisations) uses the 2026-09-18 Battaglia pair maps (before the low-z repaint): chi2_100k 19.84
  vs the repainted-map 20.07. Stated in the caption.
- WHAT_CHANGED report calls the 0.9 in ne2d "XGPaint's 0.9 factor"; it is a FORK edit (e67a8a1, 2026-04-30), upstream has none.

## XGPaint (fork kristero/XGPaint.jl, branch cluster, /home/kn18001/.julia/dev/XGPaint, HEAD 1660e08)
- 9f37b1e (A. Bayer 2026-02-27, merged b924ab1 2026-03-02): rho_2d physical rho_crit + f_b; B16 beta sign and gNFW exponent
  -(b+g)/a (B16 shape unchanged: -3.63/alpha both ways); introduced B12 sign bug (beta = -gamma - alpha*beta).
- e67a8a1 (2026-04-30): ne2d `0.9 *`; B12 beta = alpha*beta_raw - gamma (bug fixed); profiles_frb.jl (HaloDMProfile,
  compute_DM) added; fill_profile_slice!/prepare_profile_slice added.
- 71646e2 (2026-07-13): M2_TO_PC_CM3 4.87e5 -> ustrip(u"pc*cm^-3", 1u"m^-2") (~3.24e-23). No effect on these products.
- ce1d239 (2026-07-13): LOS quadrature rtol eps() -> 1e-12. 07fc686/35f79ad: log prints + N_logtheta local; 1660e08 tests.
- Upstream (not fork): HealpixRingProfileWorkspace (c609849), P_e = 0.5176 P_th, LOS zmax = 1e5.
- Project-side (NOT XGPaint): spherical chord boundary (lee2022_frb_dm_profile.jl:710-745), Lee2022XGPaintDMProfile get_params
  (799-815, undoes 0.9/m_per_e), Lee22 pressure P0/0.5176 (lee2022_tsz_pressure_profile.jl:47-51), map_weights, ExactTable.
- Ray provenance records `xgpaint_version=unknown` (open item).

## Not used / caveats
- `Takahashi25_data/` not copied (redistribution unclear). Konietzka arrays not copied (only derived figures).
- `sample_halfdome_updated_sightlines.jl` has an uncommitted diff adding `lee22_noconc_phys_sphere3` (unused here).
- Halo kSZ vs WebSky mentioned in one paragraph only (captions_ksz.txt).
