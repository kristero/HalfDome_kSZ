# Analytic halo-model reproduction of Konietzka's TNG300 DM catalogue (2026-09-30)

Code: `frb_map_generation/lee22_tsz_corrected/analytic_tng_repro/`. It uses the Torkamani+26 compound-Poisson engine with clustering (`analytic_pdf_torkamani2026/halo_model_dm_pdf.py`) unchanged, with Planck15, Tinker08 M200c and Tinker10 bias. Figure: `fig_analytic_tng_repro.{png,pdf}`. Tables: `analysis/*.csv|json`.

## 1. What the papers compared
- **Torkamani+26 (2601.18784v2).**
  - Data: the **total** LSS DM PDF of Konietzka's TNG300-1 continuous catalogue (IGM + halos), "measured at z = 0.7, 1.5, 3, 4, 5". The TNG input "is already a fit itself".
  - Model: BFC hot gas, fully ionised, integrated to θ_max = **5 r200**, halos of 1e8-1e16 Msun/h, clustering from their eq. 46.
  - The mean is not predicted. It is added from their eq. 22 (Macquart with f_IGM).
  - Reported agreement:
    - z = 0.7 (Fig. 6): a prediction from the profile fit, which "matches ... very well".
    - Per-z MCMC fits: reduced χ² = 1.51 / 0.30 / 0.22 / 0.40 / 1.17.
    - "Good match down to z ≈ 0.3".
  - No halo-only or mass-window comparison.
- **Medlock+24 (2403.02313).** This is not a halo model.
  - Data: CAMELS-TNG 25 Mpc/h boxes, 10,000 total-DM sightlines. F = σ/μ √z = **0.182** at z = 0.5; <DM(z=1)> = 1022.6 ± 169.5.
  - The "excellent matches" of the Macquart p(Δ) are a citation of Macquart+20 and Zhang+21, not a test of their own.
- **Konietzka+25.**
  - Catalogue: 1.2e5 tilted rays n = (1, 5, 25) with 12 signed permutations through the periodic 205 Mpc/h box, snapshots z = 0, 0.1, ..., 0.5, 0.7, 1, and star-forming gas set to n_e = 0.
  - The DMhalo_r200 mass windows are not described in the paper.

## 2. Torkamani's test reproduced (total DM, model shifted to the TNG mean)

| z_s | TNG σ | BFC 5R200c (our run) σ / KS / rms% in ±1σ | unclustered KS | Macquart F = 0.182 KS / rms% | eq. 22 mean / TNG |
|---|---|---|---|---|---|
| 0.3 | 139 | 125 / 0.031 / 10.8 | 0.162 | 0.096 / 46 | 1.023 |
| 0.5 | 175 | 156 / 0.035 / 9.0 | 0.151 | 0.106 / 37 | 1.022 |
| 0.7 | 201 | 177 / 0.041 / 9.3 | 0.150 | 0.107 / 33 | 1.021 |
| 1.0 | 231 | 198 / 0.053 / 12.0 | 0.156 | 0.102 / 30 | 1.021 |

- **Torkamani's result is reproduced to about 10%.** At z = 0.7 their published Fig. 6 curve matches TNG to 3.5% rms (max 8.6%). Ours matches TNG to 9.3% rms and matches their curve to 10.4% rms. Our width is 10-14% narrow (F = 0.206 against 0.234).
- **The clustering term is essential.** It needed a taper at DM scales below 10-50 pc cm⁻³, because eq. 46 diverges when a ray crosses hundreds of 5 R200c discs. σ changes by less than 1e-5 when the taper scale changes.
- **Medlock's constant is not TNG300's.** TNG300 gives F = 0.245 at z = 0.5, 35% above Medlock's CAMELS value.
- **This test says little about halo interiors.** In TNG, the gas inside R200c carries 17% of the mean but 60% of the variance (z = 0.7). BFC places 83% of its halo DM between R200c and 5 R200c.

## 3. Halo-only (gas inside R200c): mean over all rays, model/TNG, and KS distance of the hit-ray distribution

| TNG array, reading | z_s | TNG mean | corr. Lee22 F_n=5.4 | F_n=5.57 | B16 | BFC | prev. Lee22 ×0.6016 |
|---|---|---|---|---|---|---|---|
| 13to14 read as [1e13, 9e14], TNG300 abundance | 0.1 | 9.35 | 0.99, 0.034 | 0.96, 0.025 | 0.82, 0.035 | 0.81, 0.069 | 2.79, 0.40 |
| | 0.5 | 43.74 | 1.00, 0.029 | 0.97, 0.023 | 0.81, 0.051 | 0.80, 0.081 | 1.84, 0.21 |
| | 1.0 | 72.91 | 1.00, 0.026 | 0.97, 0.020 | 0.79, 0.062 | 0.78, 0.092 | 1.36, 0.075 |
| 13to14 nominal [1e13, 1e14], Tinker08 | 0.1 / 1.0 | 9.35 / 72.91 | 0.67 / 0.80 | 0.65 / 0.78 | 0.64 / 0.69 | 0.53 / 0.60 | 1.89 / 1.03 |
| 'all' (≥1e9, no window), TNG300 abundance | 0.1 | 16.55 | 0.70, 0.31 | 0.67, 0.32 | 0.91, 0.13 | 0.58, 0.26 | 1.96, 0.20 |
| | 1.0 | 177.85 | 0.55, 0.40 | 0.53, 0.41 | 0.73, 0.16 | 0.43, 0.48 | 0.73, 0.37 |

Mean ratios and KS values at z_s = 0.5 and for the Tinker08-abundance 'all' runs are in `analysis/comparison_halo.csv` (for example 'all' B16 / Tinker08 = 0.94 at z_s = 1).

- **Reinterpreted window.**
  - The corrected Lee22 reproduces:
    - the level;
    - the flat redshift trend (panel h: 0.99-1.01 for F_n = 5.4 and 0.96-0.98 for 5.57, z_s = 0.1-1);
    - the median at z_s = 1 (83.4 for F_n = 5.4 and 80.8 for 5.57, against TNG's 80.8);
    - the hit fraction (model 3-6% low);
    - the shape: bin rms 6-13% over 10-1000 pc cm⁻³ at z_s ≥ 0.5, and 12-23% at z_s = 0.1, where only 10k rays hit.
  - It is 21-37% low at 1-10 pc cm⁻³, at the sphere edges.
  - Under the literal window it cannot fit: 0.67 → 0.80 with z_s.
  - The previous ×0.6016 matches only at z_s = 1 under the literal window.
- **'all' fails for every smooth profile.**
  - 58% of TNG's DM at z_s = 1 comes from halos below 1e13 (panel d).
  - TNG implies f_e(<R200c) = 0.33 / 0.53 / 0.47 at 1e10-11 / 1e11-12 / 1e12-13 (z_s = 1). CAMELS-TNG gives 0.26-0.51 at 1e12.5-13.
  - The corrected Lee22 gives 0.01 / 0.05 / 0.21, BFC gives 0.04-0.14 and B16 gives 0.2-0.4.
- **TNG300 abundance, measured geometrically.** A = -ln(1 - hit)/N̄_Tinker08 from the nested-window differences:
  - 0.41 / 0.66 / 0.77 / 0.83 for the decades from 1e9 to 1e13;
  - 0.81 for [1e13, 9e14], which matches the Pillepich+18b count composite of 0.79-0.80;
  - 0.02-0.03 for the nominal 1e14-1e15 window;
  - 0.84-1.17 for [9e14, 9e15].
- **Cross-check.** Our F_n = 5.57 and 5.4 runs equal the `lee22_corrected_20260930` phys and a3fn runs to all printed digits.

## 4. Finite volume (`analysis/finite_volume.json`)
- **Cluster counts in one TNG300 snapshot.** One snapshot (302.6 Mpc)³ holds 278 / 152 / 49 halos of 1e14-1e15 Msun at z = 0 / 0.5 / 1 (Tinker08 × 0.944). Pillepich+18b count 277 at z = 0. Scatter: Poisson 6 / 8 / 14%, cosmic variance 12 / 13 / 15%.
- **Expected hits.** A ray to z_s = 0.1 / 0.5 / 1 crosses 1.4 / 6.4 / 11.2 box lengths. Expected hits in a TNG300-snapshot lightcone: **2.0 / 8.3 / 11.6%**. Over 20,000 box realisations the 99.9% range is 1.1-2.9 / 4.7-12.4 / 6.3-17.0%, and the minimum is 0.95 / 3.6 / 5.2%.
- **Observed hits.** **0.065 / 0.23 / 0.26%**, i.e. 78 / 277 / 315 rays against 2,429 / 10,449 / 14,794 expected crossings. That is 30-40× below expectation.
- **Finite volume cannot explain the empty window. The ×9 window can.** [9e14, 9e15] predicts 0.066 / 0.196 / 0.218%, which is 0.98-1.21 of the observed values. HalfDome's lightcone has 651k such halos below z = 1.

## Conclusions
- **Is the reproduction a success?** Partly.
  - It succeeds for Torkamani's total-DM shape (about 10%) and mean (2%), and for the corrected Lee22 against '13to14' read as [1e13, 9e14] with TNG300's abundance (mean ≤ 3%, KS ≤ 0.034, z-trend flat).
  - It fails for the window-free 'all' array. There TNG's R200c gas in 1e10-1e13 halos is 2-30× that of Lee22 and BFC, and 1.1-1.8× that of B16.
- **Is the physical correction supported?** The correction n_e = literal × (h(1+z))³ X_H² (F_n = 5.57) is supported in the mass range it was calibrated on. It is not a model for low-mass TNG halos.
- **Caveats.**
  - The window reading is inferred, not confirmed by Konietzka.
  - The low-mass abundance and f_e come from the same rays.
  - The profile tables use the HalfDome h = 0.68 (below 1% effect).

Reproduce, in `analytic_tng_repro/`, with the halfdome env and LD_LIBRARY_PATH set:
1. `LEE2022_BOOKKEEPING_FN=5.4 julia-1.12.2 -t 8 tabulate_repro_sphere_profiles.jl tables/repro_sphere_profiles.h5` (75 s)
2. `mass_decomposition.py`
3. `run_repro_pdfs.py --workers 6` (81 runs, 10-120 s each)
4. `implied_gas_fraction.py`, `finite_volume.py`, `analyze_repro.py`, `plot_repro.py`
