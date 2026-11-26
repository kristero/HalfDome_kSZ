# BE step 3 (be_compare) independent verification, 2026-11-26

## Verdict

All four checks pass. Every loss agrees within 1.05e-5 relative, every row value within 4e-6, and the split procedure
has converged.

## What I used

I wrote the code from three sources only:

- the `code/be_compare.py` module docstring;
- the T7x line in the `code/be_fit.py` docstring;
- the coordinator's brief.

No study function was read or copied. The model building blocks (electron law, T7 temperature law, row quadrature) are
my own functions from the first verification (`../scripts/verify_be.py`).

Where things are:

- Ran on STRW maas: `nice -n 19`, one thread, 56 s, RSS < 0.2 GB.
- Script: `step3/scripts/verify_step3.py`.
- Results: `step3/out/verify_step3_results.json` and `verify_step3_log.txt`. The laptop copies have identical SHA-256.
- Inputs, all read-only:
  - `out/be_compare.json` and `out/be_compare_rows.npz` (one__, full__, E__, template);
  - `out/be_fits_rows.npz` (d_P, d_E, wP, wE, region, block, lo, hi, kind, nhalo, P_err);
  - the five BD fit files under BD_ROOT `explore2h/…`;
  - the BD template files.

## Definitions

- x = r/R200c; M = 10^node of the row's BD block; z = the block's redshift; lm = ln(M/1e14); lz = ln(1+z).
- Each law quantity Q has ln Q = lnA_Q + am_Q lm + az_Q lz.
- Mass-break law (MB): ln Q gets an extra (dm_Q + kz_Q lz) S, with S = -0.3 ln(1 + exp(-(lm - lmb)/0.3)), for Q in
  {P0, x_c, beta}.
- Pressure law: P1 = 0.5176 P0 q^-0.3 (1 + q^alpha)^-beta, with q = x/x_c.
  - B family: alpha = alpha0, a saved constant.
  - A and MB families: alpha = 0.93935 + 0.29477 z.
- Electron law: E = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a_e]^(-(beta_e - 0.2)/a_e).
- Temperature laws, with T0' = exp(lnT0 + am_T lm + az_T lz), s = (x/x_cc)^2 and ln x_cc = c0 + bm lm:
  - T7: tau = T0' (1 + x/x_t)^-c · s/(1+s)
  - T7x: tau = T0' exp[-(x/x_e)^c_e] · s/(1+s)
  - In both, P1 = E tau/C_T, with C_T = 428.214.
- Full model: m = P1 + A_2h(z) t, where ln A_2h = a + b lz + c lz^2 and t is the saved `template` column.
- Row values: the volume mean over the shell, or 4 pi int_0^hi x^2 f dx for core and group rows. I integrate with
  48-point Gauss-Legendre in ln x.
- Losses:
  - L_P = sum over region 0 of wP ln^2(m/d_P)
  - L_out4 = sum over region 1 of v ln^2(m/d_P)
  - L_out6 = sum over regions 1 and 2 of v ln^2(m/d_P)
  - Outer weight: v = 1/(0.1^2 + e^2), with e = P_err/d_P.
  - L_E = sum over region 0 of wE ln^2(E/d_E)
- Outer RMS: shells in regions 1-2 with nhalo ≥ 100, banded by the shell's upper edge hi in (lo_b, hi_b], for bands
  1.5-2, 2-3, 3-4.01 and 4.01-6.01. RMS = sqrt(mean ln^2(m/d)).

## Check 1: existing BD fits with their saved parameters (law + A_2h t)

The saved law and amp equal the parameters in be_compare.json exactly.

### Losses: mine vs the study (be_compare.json)

| Fit (BD file) | L_P mine / study | rel | L_out4 mine = study | L_out6 mine = study |
|---|---|---|---|---|
| B6 (`corner…/runs/B_2h_rout6.json`) | 14.327135 / 14.327102 | 2.3e-6 | 1506.444767 (0) | 1868.809275 (0) |
| B4 (`final/final_curved075_ivar.json`) | 14.326446 / 14.326412 | 2.3e-6 | 1471.677106 (2e-16) | 1936.895946 (0) |
| A6 (`corner…/runs/A_2h_rout6.json`) | 12.335350 / 12.335319 | 2.5e-6 | 2904.862755 (1e-16) | 3735.759520 (1e-16) |
| A4 (`final/rec_held_curved.json`) | 12.337327 / 12.337296 | 2.5e-6 | 2635.684162 (2e-16) | 4193.553605 (0) |
| MB4 (`mass_weights/final/best_sbplzz_held.json`) | 1.999252 / 1.999250 | 8.9e-7 | 2700.769205 (2e-16) | 4119.057383 (0) |

### Losses: mine vs the values saved in the BD files

- Inner loss (B/A: `L_in`; MB4: `info.L_fit`): mine differs by 2.3e-6 (B6, B4), 2.5e-6 (A6, A4) and 8.9e-7 (MB4)
  relative.
- Outer loss (B6/A6: `L_out` against my L_out6; B4/A4: `L_out` against L_out4; MB4: `info.L_amp` against L_out4): mine
  differs by 2.3e-7, 1.6e-6, 2.5e-8, 1.7e-7 and 7.7e-7.

### Row values and other checks

- Rows against one__/full__: shells agree to 8e-15. Core and group integrals differ by at most 2.6e-6 (B), 4.0e-6 (A)
  and 2.5e-6 (MB).
- Outer all-mass RMS per band equals `outer_all_masses` to 1e-16, and n matches (186, 277, 238, 279). The RMS values are:

| Band | B6 | A6 | MB4 |
|---|---|---|---|
| 1.5-2 | 0.0905 | 0.1827 | 0.1570 |
| 2-3 | 0.0969 | 0.1768 | 0.1758 |
| 3-4.01 | 0.1173 | 0.1381 | 0.1885 |
| 4.01-6.01 | 0.1159 | 0.1857 | 0.2491 |

- `amp_by_z` equals exp(a + b lz + c lz^2) to 7e-16.
- My MB law reproduces MB4's saved `reference_points` (P0, x_c, beta, alpha at 6 mass/z points) exactly.
- The `template` column equals the BD template files exactly:
  - `flamingo_B32_full_hm_ex0.75.npy` for region 0;
  - `flamingo_x4_full_hm_ex0.75.npy` for region 1 (all 1025 rows matched by block, lo and hi);
  - `far_hm_ex0p75.npy` for region 2.

### Why L_P differs by about 2e-6

The residual comes entirely from the core and group integrals, where the study (and BD explore2h) use 12-point
Gauss-Legendre in x over [0, hi]. That rule has an error of about 3e-6 on those rows; shells and outer losses agree to
machine precision. The prefactor 0.5176 is the record convention: the exact (2+2X)/(3+5X) would raise L_P by about
2e-5.

## Check 2: electron law × temperature law + 2-halo

### Losses: mine vs the study (be_compare.json)

| Model | L_P mine / study (rel) | L_out4 rel | L_out6 rel | L_E mine / study (rel) |
|---|---|---|---|---|
| joint_T7x_rout6 | 7.676276 / 7.676277 (5.6e-8) | 2e-16 | 2e-16 | 15.816678 / 15.816513 (1.0e-5) |
| joint_T7_rout6 | 7.738602 / 7.738603 (7.9e-8) | 0 | 2e-16 | 15.834261 / 15.834094 (1.0e-5) |
| joint_T7x_rout4 (extra) | 7.672675 (5.8e-8) | 0 | 2e-16 | 15.816176 (1.0e-5) |
| joint_T7_rout4 (extra) | 7.735962 (8.1e-8) | 0 | 0 | 15.833625 (1.0e-5) |
| sep_T7x_rout6 (extra) | 9.085509 (6.2e-8) | 0 | 0 | 15.153339 (9.6e-6) |
| sep_T7_rout6 (extra) | 9.396441 (3.1e-7) | 1e-16 | 2e-16 | 15.153339 (9.6e-6) |

### Row values and other checks

- one__ rows: shells agree to 8e-15; integral rows differ by at most 2.0e-6.
- full__ rows: at most 2.0e-6.
- E__ rows: at most 9.3e-6, on the integral rows only (same quadrature cause as in check 1).
- Outer all-mass RMS equals the JSON to 2e-16, and n matches:

| Band | joint_T7x_rout6 | joint_T7_rout6 |
|---|---|---|
| 1.5-2 | 0.2068 | 0.2144 |
| 2-3 | 0.1863 | 0.2001 |
| 3-4.01 | 0.1413 | 0.1474 |
| 4.01-6.01 | 0.1149 | 0.1192 |

- `amp_by_z` matches to 4e-16.

## Check 3: convergence of the split procedure (joint_T7x_rout6)

**(a) Amplitude refit with the law held fixed.** I used least squares in ln over regions 1+2 with weights v, starting
from the reported amp = (-0.496861, 1.872408, -0.650236).

- Largest change in any amplitude parameter: 7.8e-8.
- Largest relative change in A_2h(z): 1.0e-7.
- L_out6 falls by 1.4e-10 (from 4278.8652611).

The amplitude is converged.

**(b) Law parameters with the amplitude held fixed.** The objective is L_P + L_E over region 0, with all 16 law
parameters free, starting from the reported point.

- At the reported point my objective is 23.4929545; the study's L_P + L_E is 23.4927895. The 1.65e-4 difference is the
  L_E quadrature difference above.
- Least squares (TRF, 4 evaluations) lowers it by **1.3e-8**, well under the 1e-4 limit.
- The largest parameter change is 4.9e-6.
- 30 random perturbations of size 1e-3 all raised the objective.

The reported point is a minimum of the law step given the amplitude.

## Check 4: parameter counts

Every model's counts.total_with_electrons equals both the sum of its parts and the number of saved parameters (plus the
9 electron-law parameters for the existing pressure fits):

| Family | total_with_electrons | Breakdown |
|---|---|---|
| B6, B4 | 22 | 9 + 10 + 3 |
| A6, A4 | 21 | 9 + 9 + 3 |
| MB4 | 28 | 9 + 16 + 3 |
| All T7 and T7x | 19 | 9 + 7 + 3 |
| H6 (xout6, rout6 and rout4) | 18 | 9 + 6 + 3 |

## Notes

- joint_T7_rout6 and joint_T7_rout4 again have c_out = 9.999999999999998: the T7 outer slope sits on its bound. This
  is the same as in the 1-halo T7 fits; check 3 covered only T7x.
- The existing fits' L_P reproduces the BD `L_in` only with the 12-point integral rule. In contrast, the BD record loss
  in `results/flamingo_residuals.json` matched accurate quadrature to 2e-15 (first verification). The two BD products
  therefore integrate the core and group rows differently. The effect is about 1e-6 relative, so it is harmless.
- Not verified: be_compare's `bands_inside`, `extension_one_halo` and `physicality` blocks, and the H6 rout fits.
