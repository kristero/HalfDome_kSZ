# Why Konietzka's TNG300 halo DM implies about 2x more electrons than TNG-model halos (2026-09-30)

Code is in `frb_map_generation/lee22_tsz_corrected/konietzka_check/`. The CAMELS cell runs were idark PBS jobs 600119-600122, run in `/lustre/work/kristero10/lee22_konietzka_check_20260930`; the raw snapshots were deleted after processing.

**Bottom line.** No halo-boundary or electron convention explains the tension. The most likely cause is the catalogue's mass windows. The window algebra of the 120,000 rays shows that `DMhalo_r200_mass13to14` behaves as if its upper edge were about 9x the nominal value, i.e. it holds halos with 1e13 < M200c < ~9e14 Msun inside R200c spheres. Counting those clusters gives CAMELS-level gas fractions. It reproduces the factor, the shallow (1+z)^2.48 scaling, and the hit fractions at every z. This is inferred from the arrays and has not been confirmed. R. Konietzka should confirm the window and mass definitions.

## 1. What the sources define (quotes)
- **Konietzka+25, arXiv:2507.07090v2, Sec. III.7.** "we set the free electron density of star-forming gas to zero and calculate the free electron density of non-star-forming gas as ... n_e = f_eH x_H ρ_gas/m_p (1+z)^3", with ρ_gas and f_eH taken from "Density and ElectronAbundance".
- **Konietzka+25, eq. 11.** DM = ∫ n_e/(1+z)^2 dx over the comoving box path, so DM is in the observer frame.
- **Konietzka+25, halo boundary and mass.** "We define the virial radius as the radius of a sphere ... whose mean density is equal to 200 times the critical density". "We use the mass enclosed in a spherical overdensity of 200 times the critical density ... for the mass of the dark matter halo". Both quotes come from the host-exclusion (3 R_vir) and Milky-Way-selection sections.
- **The paper does not describe the halo-DM mass splits.** Our arrays are bare `.npy` files with no attributes. The author's notebook (`DM_halo_pdf_IllustrisTNG_5.ipynb`) says "Please do not distribute the catalogs prior to the official release date" and labels the windows only as "10^10 < M_halo/Msun < 10^12".
- **Hussaini, Connor, Konietzka+25, arXiv:2506.04186, Sec. IV.3.** This paper uses "Konietzka et al. 2025, in prep": "The halo component is defined as Voronoi cells where the dark matter density is 57 times that of the critical density ... approximately the local density of a dark matter halo at its virial radius (~r200)". It cites Artale+22 (arXiv:2102.01092), which uses "the comoving local dark matter density (ρdm) of each gas cell" (SubfindDMDensity) with "Haloes ... ρdm > 57 ρcrit,0".

## 2. What the arrays themselves show (`window_structure_and_hmf.json`)
- **The windows are not disjoint.** At z_s = 0.1, DM(10to13) + DM(13to14) − DM(10to14) is 55% of DM(13to14), and 57-68% at z_s = 0.3-1.5. The overlap is all-or-nothing per ray: 76% of the rays that hit 13to14 also carry their entire 13to14 DM in 10to13, 22% carry none of it, and 10to12 never contains it. For 10to12 against 12to14 the overlap is 23% of the DM.
- **The nominal 1e14-1e15 window is nearly empty.** DM(10to15) > DM(10to14) on only 0.06 / 0.23 / 0.26% of rays at z_s = 0.1 / 0.5 / 1. R200c spheres with M200c in 1e14-1e15 would be hit by 2.1 / 8.8 / 12% of rays.
- **A Tinker08 model with effective windows [lo, c·hi] fits all of this with c ≈ 9.** For c = 9 it predicts:
  - 0.07 / 0.21 / 0.23% for the 10to15 excess;
  - an overlap of 0.55 of the DM and 0.79 of the rays for 13to14, and 0.60 of the rays for 12to14 (observed 0.591).

  The fitted range is c = 8-10. The 22% of rays with no overlap carry 2.7x the DM per crossing, as expected if they are 1e14-1e15 clusters.

## 3. Test table
All values are relative to CAMELS R200c spheres with ElectronAbundance electrons, the nominal M200c window 1e13-1e14 Msun and Tinker08 abundance. Ne = electron sum over the window (sets the mean DM); area = summed projected area (sets the hit fraction). CAMELS cells come from 4 boxes (CV_0-3) at z = 0, 0.27, 0.47, 1.05, 1.48 (8-24 halos per z; bootstrap errors in `analysis_log.txt`). Window rows use the Tinker08 halo model per shell.

| candidate | Ne z=0.05 | 0.45 | 1.05 | 1.45 | area z=0.05 | 1.05 | fits? |
|---|---|---|---|---|---|---|---|
| **target: Konietzka / model** | **1.55** | **1.24** | **1.07** | **0.95** | **1.02** | **0.93** | |
| R200c, star-forming n_e = 0 (their convention) | 1.00 | 1.00 | 0.98 | 0.97 | 1 | 1 | no |
| R200m sphere | 1.84 | 1.36 | 1.13 | 1.06 | 2.71 | 1.24 | level and slope yes, hits no (x2.7) |
| R200m sphere + M200m window | 1.69 | 1.35 | 1.17 | 1.10 | 3.09 | 1.32 | hits no |
| 1.5 / 2 R200c sphere | 1.66 / 2.20 | 1.56 / 1.98 | 1.53 / 1.95 | 1.40 / 1.71 | 2.25 / 4 | 2.25 / 4 | no |
| all FoF member cells | 1.45 | 1.10 | 1.07 | 0.96 | 2.32 | 1.48 | hits no |
| ρ_DM > 57 ρcrit,0 comoving, FoF cells (Hussaini/Artale) | 0.88 | 0.67 | 0.62 | 0.61 | 1.47 | 0.97 | no (fewer electrons) |
| same with ρcrit(z) physical / threshold x0.5 | 0.91 / 1.19 | 0.90 / 0.92 | 0.97 / 0.88 | 0.90 / 0.82 | 1.52 / 1.95 | 1.33 / 1.25 | no |
| window read in M200m / M_FoF / M500c | 0.92 / 0.94 / 1.06 | 1.00 / 1.00 / 0.98 | 1.03 / 1.03 / 0.86 | 1.03 / 1.04 / 0.78 | 1.14 / 1.11 / 0.85 | 1.06 / 1.06 / 0.75 | no |
| window read in Msun/h | 1.07 | 0.98 | 0.85 | 0.76 | 0.85 | 0.73 | no |
| missing 1/(1+z) in DM | 1.05 | 1.45 | 2.05 | 2.45 | 1 | 1 | no (wrong trend; Konietzka total/Macquart is flat, 0.95) |
| TNG300 hydro abundance only (Pillepich+18b counts) | 0.75 | 0.78 | 0.82 | 0.82 | 0.75 | 0.82 | no |
| **window upper edge x9, Tinker08** | 1.72 | 1.43 | 1.15 | 1.06 | 1.24 | 1.07 | slope yes, level 10-20% high |
| **window upper edge x9, TNG300 hydro abundance** | **1.43** | **1.18** | **0.96** | **0.87** | **0.98** | **0.88** | **yes: Konietzka/prediction = 1.07 (1+z)^0.03; hits within ±7% in 13 of 15 shells** |

Result of the table:
- **Geometric definitions fail.** Every one that adds electrons (R200m, larger spheres, FoF) adds 1.5-4x more hit area, and the equal hit fractions exclude that. The local-DM-density definition removes electrons: it is 0.6-0.9x R200c while its area is 1.0-1.5x.
- **Only the window extension works.** Counting 1e14-9e14 halos raises the electrons by 1.4-1.7x at z < 0.5 and 1.1x at z > 1, but the area by only 1.03-1.24x, because clusters are rare and gas-rich. With TNG300's hydro abundance the level and slope of both the electrons and the hit fraction match.
- **The fit is robust.** It holds for c = 6-10 and for a cluster f_e cap of 0.80-0.95 (`sensitivity_c_and_cap.txt`: Konietzka/prediction 1.02-1.16, hits 0.99-1.05).
- **Consistency with HalfDome.** HalfDome's own hits (8.06% at z_s = 0.1) are 0.97x Tinker, which is consistent with this.
- **Resolution and box size do not explain the factor on their own.** CAMELS L25n256 agrees with the published TNG300-1 gas fractions of 0.56-0.60 f_b (forensics D).

Figure `fig_konietzka_check.png`:
- (a) implied f_e per shell;
- (b) hit fraction per shell relative to the model;
- (c) electrons vs area of each candidate at z = 0.

## 4. Recommendation and F_n (n_e = literal Lee22 x (1+z)^3 / F_n, 1e13-1e14 Msun, HMF-weighted, shells z < 1.5)
- **Calibrate to physical R200c electrons of TNG-model halos, not to `mass13to14`.** Use CAMELS, which agrees with the TNG300 literature.
  - F_n = 5.55 (1+z)^-0.05, i.e. a constant **F_n ≈ 5.4** (DM-weighted 5.43).
  - That is n_e ≈ 0.185 (1+z)^3 x literal, compared with C(z) = 0.284 (1+z)^2.48 now.
  - Per mass bin (workstream D): 5.1 for 13-13.5 and 5.7 for 13.5-14.
- **Konietzka's array under its own labels** (compared with a nominal 1e13-1e14 R200c model) needs F_n = 3.54 (1+z)^0.51 (DM-weighted 4.3), which is today's C(z). It fits only because it absorbs the cluster gas.
- **Konietzka reinterpreted** (window 1e13-9e14, TNG300 hydro abundance) needs F_n = 5.44 (1+z)^-0.10 (DM-weighted 5.2), within 5% of the CAMELS value. This closure supports the interpretation.
- **For the HalfDome-TNG PDF comparison:** confirm the window with R. Konietzka, then compare like with like. Use a HalfDome M200c window of 1e13 to ~1e15, and account for HalfDome being dark-matter-only: TNG300 hydro has ~25% fewer 1e13-1e14 halos than Tinker08.

Caveats:
- The window mechanism is inferred (a different mass or unit for the upper cut, or an off-by-one decade, would both fit), not read from code.
- CAMELS has 8-24 halos per z in 1e13-1e14 and only 9 halos above 1e14 at z = 0 across all 27 boxes, so the cluster f_e is capped at 0.88 (range tested 0.80-0.95).
- The TNG300 abundance factors come from Pillepich+18b counts at z = 0 and 1 only.

Reproduce: `konietzka_window_hmf.py` (local, ~5 min), `run_konietzka_defs.pbs` via `qsub -v BOX=b,SNAPS="090 080 074 060 052"` on idark, `fetch_defs.sh`, `analyze_konietzka_check.py`, `plot_konietzka_check.py`. Use the halfdome env with LD_LIBRARY_PATH set.
