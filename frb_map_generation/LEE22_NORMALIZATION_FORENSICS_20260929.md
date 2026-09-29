# Lee22 density normalization: what is wrong, what we corrected, what is open

Date: 2026-09-29. Concise version of `outputs/lee22_normalization_forensics_20260928/FINAL_REPORT.md` (13-agent investigation with
adversarial re-checks). Figures: `outputs/lee22_tngcal_20260929/report/lee22_fig{1,2,3}_*.png`
(script `lee22_tng_correction/plot_lee22_forensics_summary.py`, built only from the saved investigation outputs).

**Bottom line.** The Lee et al. (2022) density fit, as printed, is wrong in two ways: its amplitude falls with redshift about
(1+z)^2.5 too fast, and it is too high by a constant factor. Our production "XGPaint-native" Lee22 (0.6016 x the paper) matched
TNG's FRB halo-DM PDF only because 0.6016 is the right correction at halo redshift z ~ 0.35, which is where the z_s = 1 DM comes
from. The TNG-calibrated correction below removes the redshift error; the absolute electron content remains open.

## 1. Mistakes

| # | Where | What | Size | Evidence |
|---|---|---|---|---|
| M1 | Lee22 profile extraction | cell sums divided by **comoving** (ckpc/h)^3 shell volumes, then by the **physical** n200(z), P200(z); P200 also uses the comoving R200 | density x (1+z)^-3, pressure x (1+z)^-2 (code level); measured against TNG: density (1+z)^-(2.4-3.2) | Fig. 1b, 1c; public `group_particles` code (`examples/ne_prof.cpp`, `y_prof.cpp`: `shell_vol` from `Group_R_Crit200`, `rho_crit_0 (Omega0/a^3 + OmegaLambda)`) |
| M2 | Lee22 density fit | an extra, redshift-independent density factor F_n (not in the pressure) | F_n ~ 3-8; printed fit holds 2.1-5.4 f_b of gas inside R200c at z = 0 | Fig. 1a (gas fraction), 1c (T_e = 0.15 T200 from Lee22's own P/n), 1d |
| M3 | Lee22 figures | reference curves mis-normalized: Fig. 5 "Battaglia16" drawn without f_b, Fig. 2 "Battaglia12" is thermal, not electron, pressure | x 6.3 and x 1.9 | so the comparison in the paper could not reveal M1/M2 |
| M4 | ours (production) | `normalization=:xgpaint_ne2d` feeds P0 = 200 n0 to XGPaint, whose ne2d converts gas -> electrons | exactly 0.6016 x the paper, all r, M, z | a units choice, not physics; matches TNG only at z_s ~ 1 (Fig. 2) |
| m5 | ours (minor) | default `n0_pivot=:legacy_1e14` (production passes `:mcut`); `:comoving_hypothesis` option ruled out; Lee22 pressure P0 extrapolated above 10^14.8 h^-1 Msun | tSZ low-ell excess 1.7-2.7x Battaglia12 is this extrapolation | Fig. 3b |

Our transcription of the Lee22 tables is exact (6e-16 against the TeX, 2-3 % against the paper's Fig. 5 curves). Fig. 5's y-axis
is dimensionless (x^3 n_e/n200), so matching it never tested the absolute normalization.

## 2. The correction now in the code

`lee2022_frb_dm_profile.jl`: `redshift_scaling = :tng_calibrated` (with `normalization = :literal`),

    n_e = C(z) x n0 g(x) n200(z),   C(z) = 0.28418 (1+z)^2.4832,   held at its z = 2 value above z = 2,

fitted to Konietzka's TNG300 within-R200 halo DM of 1e13-1e14 Msun at source redshifts 0.1-2 (rms 0.8 %). Equivalent Table 8
edit: n0 A0 6.8 -> 1.93, alpha_z -2.11 -> +0.37 (x'c, beta' as printed). Relative to the production variant the density changes by
C/0.6016 = 0.47 (z = 0), 1.00 (z = 0.35), 2.6 (z = 1), 7.2 (z = 2). Generator: `--dm-profile=lee2022_xgpaint
--lee2022-normalization=literal --lee2022-redshift-scaling=tng_calibrated`; full-sky painter model `lee22_noconc_tngcal_sphere1`.
Unit test `lee22_tng_correction/test_tng_calibrated_option.jl` (factor exact to 4e-16 locally and on idark; existing cache
signatures unchanged).

What it fixes: the FRB halo DM against TNG at every source redshift (by construction, Fig. 2a) and where it comes from (Fig. 2b);
the kSZ excess over Battaglia16 becomes flat in ell (Fig. 3a). What it does not fix: it inherits Lee22's mass slope, so massive
halos at z < 0.5 still hold 1.3-1.6 f_b inside R200c (Fig. 1a, green); kSZ stays 2.6x Battaglia16 at ell = 3000; the pressure is
untouched.

## 3. Open questions

1. **F_n and the exponent.** Konietzka's DM implies F_n ~ 3 (0.89 f_b of halo electrons); TNG's published gas fractions, X-ray
   temperatures and hydrostatic balance imply 5-7 (Fig. 1d). Candidates 1/(h^3 X_H^2) = 5.6 or Omega_m/Omega_b = 6.4 (f_b twice).
   Needs Lee22's n200 code: ask B. K. K. Lee and L. Thiele.
2. **Konietzka catalogue definitions** (mass type, overlap, star-forming cells): it holds ~1.5x TNG's published halo gas, which sets
   the absolute level of C(z).
3. **Pressure a^2** is predicted by the code but not isolated in tSZ; compare Lee22's pressure with TNG pressure profiles.
4. **Mass dependence** of the correction: C(z) was calibrated on 1e13-1e14 Msun only (1e12-1e14 gives 0.25 (1+z)^3.2).

## 4. Results with the corrected Lee22

idark runs of 2026-09-29, root `/lustre/work/kristero10/frb_data/lee22_tngcal_20260929` (jobs 600040-600049; the first ray
jobs stopped after z_s = 1 because the post-run plotter needs Python >= 3.7, and were resubmitted with Python 3.8; all exits 0).
Local copies in `outputs/lee22_tngcal_20260929/{rays,spectra,logs}`; scripts `lee22_tng_correction/plot_tngcal_{pdfs,cross}.py`.

**Halo-DM PDF against TNG** (`outputs/lee22_tngcal_20260929/plots/halo_pdf_lee22_tngcal_vs_tng.png`; 120k rays, R200c sphere,
same rays at every z_s; stats in `analysis/halo_pdf_lee22_tngcal_stats.csv`). Mean DM per hit, 1e13-1e14 Msun:

| z_s | TNG | Battaglia16 | Lee22 x 0.6016 (previous) | Lee22 x C(z) |
|---|---|---|---|---|
| 0.1 | 110 | 79 (-28 %) | 221 (+100 %) | 117 (+6 %) |
| 0.5 | 123 | 82 (-33 %) | 163 (+32 %) | 124 (+1 %) |
| 1.0 | 139 | 93 (-33 %) | 139 (0 %) | 138 (-1 %) |

The correction removes the source-redshift dependence of the mismatch. What it leaves: rays with DM < 30 pc cm^-3 (grazing
chords) are 50-75 % below TNG for every Lee22 variant (a profile-shape effect near R200c, not normalization), and at z_s = 1 the
PDF shape is slightly worse than the previous variant (median bin |difference| 19 % vs 6.5 %; 33 % vs 81 % at z_s = 0.1). At
1e12-1e14 Msun (below the calibration range) the means agree (TNG 129, Lee22 x C(z) 132), but only 60 % of HalfDome rays hit a halo
against 91 % in TNG (catalogue incompleteness below 1e13), so that window is not a clean test. idark reproduces the local z_s = 1
products statistically (mean per hit 93.0/93.2 and 138.6/138.9 local/idark), not bitwise: Julia 1.6.2 and 1.12 draw different
directions from `MersenneTwister(42)`.

**tSZ x FRB DM against Takahashi+25** (`plots/tsz_x_frbdm_lee22_tngcal_takahashi.png`; chi^2 in
`analysis/tsz_x_frbdm_lee22_tngcal_chi2.csv`; real jackknife covariance above the paper's cut; the y maps are unchanged):

| pair | Planck chi^2 (8 bins) | ACT chi^2 (11 bins) |
|---|---|---|
| w = 0 | 21.9 | 9.7 |
| Battaglia12 y x Battaglia16 DM | 19.8 | 9.0 |
| Lee22 y x Lee22 DM x 0.6016 (previous) | 57.7 | 119.4 |
| **Lee22 y x Lee22 DM x C(z)** | **28.5** | **37.4** |
| Battaglia12 y x Lee22 DM x C(z) | 20.5 | 19.3 |
| Lee22 y x Battaglia16 DM | 20.7 | 10.9 |
| calibrated ranges only: previous / x C(z) | 23.5 / 20.6 | 30.0 / 12.8 |

The kernel-weighted halo DM drops from 58.5 to 40.0 pc cm^-3 (Planck kernel) and from 62.9 to 44.1 (ACT), because the observed
FRBs lie at low z, where C(z)/0.6016 < 1. The correction halves the Lee22 x Lee22 chi^2 for Planck and cuts it by a factor of 3.2 for
ACT. It is still disfavoured against Battaglia12 x Battaglia16 (Delta chi^2 = +8.7, +28). Neither Lee22 ingredient alone does
this (Lee22 y x B16 DM: +0.9, +1.9; B12 y x Lee22 DM x C(z): +0.7, +10.3); the product of the Lee22 y excess and the remaining Lee22
low-z density excess (massive halos at 1.3-1.6 f_b, Sec. 2) does. Restricted to the calibrated ranges, Lee22 x C(z) is as good as
Battaglia (20.6 / 12.8). The data alone are weak (w = 0 gives 21.9 / 9.7).
The tSZ side is verified separately in `LEE22_TSZ_VERIFICATION_20260929.md`.
