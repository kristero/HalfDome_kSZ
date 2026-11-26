# BE step 4 (be_fit4r) independent verification, 2026-11-26

## Verdict

All three checks pass in both weightings.

- The weights rebuild exactly.
- Every recomputed loss agrees within 2.3e-6 relative (L_E within 1.1e-5).
- The reported points are the minima: no search from the reported point or from 40 random starts finds anything lower
  by more than 1.4e-8.

## What I used and where it ran

I wrote the code from the `code/be_fit4r.py` module docstring and the coordinator's brief only. No study code was read
or copied. The building blocks (electron law, quadrature) come from my first verification (`../scripts/verify_be.py`),
here vectorized over rows. The vectorized rule reproduces the scalar rule to 3.3e-16.

- Ran on STRW maas: `nice -n 19`, one thread, 1001 s, RSS about 0.1 GB.
- Script: `step4/scripts/verify_step4.py`.
- Results: `step4/out/verify_step4_results.json` and `verify_step4_log.txt`. The laptop copies have identical SHA-256.
- Inputs, all read-only:
  - `out/be_fit4r.json` and `be_fit4r_rows.npz` (record weights);
  - `out/be_fit4r_parity.json` and `be_fit4r_parity_rows.npz` (parity weights);
  - `out/be_fits_rows.npz` and `out/be_fits.json`;
  - BD `results/flamingo_residuals.{json,npz}`.

## Definitions

- x = r/R200c; M = 10^node of the row's BD block; z = the block's redshift; lm = ln(M/1e14); lz = ln(1+z).
- Each law quantity Q has ln Q = lnA_Q + am_Q lm + az_Q lz.
- Mass-break law: ln Q gets an extra (dm_Q + kz_Q lz) S, with S = -0.3 ln(1 + exp(-(lm - lmb)/0.3)).
- Pressure law: P1 = 0.5176 P0 q^-0.3 (1 + q^alpha)^-beta, with q = x/x_c.
  - plaw_const: alpha = alpha0.
  - plaw_massbreak: alpha = 0.93935 + 0.29477 z.
- joint_T7x: P1 = E tau/C_T, with:
  - E = the electron law of the first 9 parameters;
  - tau = T0' exp[-(x/x_e)^c_e] s/(1+s), where T0' = exp(lnT0 + am_T lm + az_T lz), s = (x/x_cc)^2 and
    ln x_cc = c0 + bm lm.
- Full model: m = P1 + A_2h(z) t, where ln A_2h = a + b lz + c lz^2 and t is the saved template. The template is
  identical in both rows files and in step 3.
- Losses:
  - L_4R = sum over regions 0 and 1 of w ln^2(m/d_P); I also report its region-1 part.
  - L_P = sum over region 0 of wP ln^2(m/d_P)
  - L_E = sum over region 0 of wE ln^2(E/d_E)
  - L_out6 = sum over regions 1 and 2 of v ln^2(m/d_P), with v = 1/(0.1^2 + e^2) and e = P_err/d_P.
- Outer RMS: shells in regions 1-2 with nhalo ≥ 100, banded by hi in (lo_b, hi_b].

## Check 1: weights

**Region 0.** The weights are BD `B32__weight`, identical to wP in be_fits_rows.npz. They sum to 2701.0 (2700.9999999999995).

**Region 1, record.** These are BD `x4__weight` for rows with kind = shell and lo ≥ B32(block) - 1e-9, in file order.

- The selection gives exactly 1025 rows, and their block, lo and hi match the region-1 rows of be_fits_rows.npz in
  order.
- Sum: 44.99058226938062. weight_share.sum_w_region1 is 44.99058226938062 (relative difference 0).
- Region 1 carries 1.64 % of the L_4R weight.

**Region 1, parity.** w = (2701/1494) v_i / mean_region1(v).

- Sum: 1853.095716198126 = (2701/1494) × 1025. The study's value is 1853.0957161981255 (relative difference 2e-16).
- Region 1 carries 40.7 % of the weight.

PASS.

## Check 2: models recomputed from saved parameters (law + A_2h t)

| Weighting / model | L_4R mine / study (rel) | L_4R region 1 rel | L_P mine / study (rel) | L_E mine / study (rel) | L_out6 rel |
|---|---|---|---|---|---|
| record / plaw_const_4R | 15.206281 / 15.206248 (2.2e-6) | 2e-16 | 14.457332 / 14.457299 (2.3e-6) | — | 2e-16 |
| record / plaw_massbreak_4R | 2.705008 / 2.705006 (5.9e-7) | 4e-16 | 2.141204 / 2.141203 (7.4e-7) | — | 0 |
| record / joint_T7x_4R | 9.094424 / 9.094425 (1.5e-7) | 1e-16 | 7.426544 / 7.426545 (1.8e-7) | 16.260400 / 16.260230 (1.0e-5) | 0 |
| parity / plaw_const_4R | 35.836777 / 35.836754 (6.5e-7) | 4e-16 | 16.756688 / 16.756665 (1.4e-6) | — | 0 |
| parity / plaw_massbreak_4R | 19.610541 / 19.610540 (7.4e-8) | 2e-16 | 4.255544 / 4.255542 (3.4e-7) | — | 0 |
| parity / joint_T7x_4R | 39.861519 / 39.861519 (1.4e-8) | 0 | 7.857022 / 7.857022 (7.3e-8) | 27.683840 / 27.683533 (1.1e-5) | 1e-16 |

The region-1 parts agree to machine precision: 0.748949, 0.563804 and 1.667880 for record; 19.080089, 15.354998 and
32.004497 for parity.

- Rows against one__ and full__: shells agree to 8e-15, and core/group integrals differ by at most 3.6e-6.
- E__ rows differ by at most 9.9e-6.
- Outer all-mass RMS per band (1.5-2, 2-3, 3-4.01, 4.01-6.01) equals `outer_all_masses` to at most 1.1e-16 absolute,
  with identical counts.

All residual differences sit on core and group integral rows. The study uses 12-point Gauss-Legendre in x over
[0, hi] there, as in steps 2 and 3, which has an error of about 1e-6 to 1e-5. PASS.

## Check 3: optimality with all parameters free

Free parameters: the law plus the 3 amplitude parameters, 13 for plaw_const and 19 for joint_T7x. The objective is
L_4R, plus L_E for the joint model. For each model and weighting I ran three searches:

- least squares (TRF) from the reported point;
- Nelder-Mead from the reported point, 600 evaluations;
- 10 random starts, each parameter perturbed by a Gaussian with sigma 0.05, then least squares to convergence.

| Weighting / model | Objective at reported point, mine (study) | Local LS decrease | Nelder-Mead decrease | Best of 10 random starts minus reported | Starts lower by > 1e-3 |
|---|---|---|---|---|---|
| record / plaw_const_4R | 15.2062810 (15.2062483) | 9.4e-11 | 4.0e-11 | -9.5e-11 | 0 |
| record / joint_T7x_4R | 25.3548236 (25.3546547) | 1.4e-8 | 0 | -1.4e-8 | 0 |
| parity / plaw_const_4R | 35.8367771 (35.8367537) | 6.9e-11 | 8.6e-12 | -6.8e-11 | 0 |
| parity / joint_T7x_4R | 67.5453594 (67.5450515) | 8.4e-9 | 0 | -8.4e-9 | 0 |

- Every random start began between 38 and 4585 and converged to the reported solution: the end losses agree within
  1.2e-11, and the parameters lie within 6.2e-6 of the reported ones.
- The local least-squares steps moved the parameters by at most 5.3e-6.
- The constant offset between my objective and the study's (1.7e-4 for the joint fits) is the L_E quadrature difference
  on the integral rows noted in check 2. It is not a different optimum.

No search lowers the objective by more than 1e-3; the largest decrease is 1.4e-8. PASS.

## Notes

- The T7x fits have no active bounds (`at_bounds` = []). joint_T7_4R, which I did not verify, still reports c_out at
  its bound of 10 in both weightings.
- Region 1 carries only 1.6 % of the record weight. Under record weights the 4 R200c fit is therefore almost the
  inside-B32 fit; parity weighting gives region 1 about 41 %.
- Not verified: plaw_held_4R, joint_T7_4R, sep_T7x_4R and joint_H6_xout6_4R (only their weight sums are shared), and
  the Y200, extension, physicality and bands_inside blocks.
