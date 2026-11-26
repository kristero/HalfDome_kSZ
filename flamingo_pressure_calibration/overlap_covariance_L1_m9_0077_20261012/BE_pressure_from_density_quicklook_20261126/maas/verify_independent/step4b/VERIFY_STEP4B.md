# BE step 4b (be_fit4r --set flex / eflex, parity weights): independent verification, 2026-11-26

## Verdict

All three checks pass: every recomputed loss agrees with the study, and no search finds a lower minimum.

One physics flag: the free electron inner slope (the "eg" variants) is held at its upper bound, g = 1.

## Method

I wrote the code only from these sources:

- the be_fit.py docstring lines for T8xm, T8xz and T9x;
- the ELaw.point docstring;
- the be_fit4r.py docstring (--set flex / eflex);
- the coordinator's brief.

The building blocks reuse my step-4 vectorized evaluator: 48-point Gauss-Legendre in ln x, and the parity weights rebuilt in step 4.

- Ran on STRW maas: nice 19, one thread, 480 s.
- Script: `step4b/scripts/verify_step4b.py`.
- Results: `step4b/out/verify_step4b_results.json` and `verify_step4b_log.txt`. The laptop copies have identical SHA-256.

## Definitions

General electron law:

- E = n0 q^g (1 + q^a_e)^(-(beta_e + g)/a_e), with q = x/x_c.
- Production law: x_c = 0.5 and g = -0.2.
- "ec" variants: x_c = exp(ln_xc_e).
- "eg" variants: g = gamma_e.
- The extra constant is the 10th electron parameter.

Temperature law:

- tau = exp(lnT0 + am_T lm + az_T lz - (x/x_e)^c_e) s/(1+s).
- s = (x/x_cc)^2, with ln x_cc = lnxcc0 + bm_cc lm.
- ln x_e = ln_xe + em_xe lm (T8xm, T9x) + ez_xe lz (T8xz, T9x).
- P1 = E tau/C_T.

Model and losses:

- m = P1 + A_2h(z) t, with ln A_2h = a + b lz + c lz^2.
- L_4R: parity weights, regions 0 and 1.
- L_P: weights wP, region 0.
- L_E: weights wE, region 0.
- L_out6: weights v, regions 1 and 2.
- Outer RMS: shells in regions 1-2 with nhalo ≥ 100, banded by hi.
- For the sep models, the electron law is `be_fits.json` electron_law.params.

## Check 1: models recomputed from saved law/amp

| Model | L_4R mine / study (rel) | L_P rel | L_E rel | L_4R region 1, L_out6, outer RMS |
|---|---|---|---|---|
| joint_T9x_4R | 39.675238 / 39.675237 (2.6e-8) | 1.5e-7 | 1.4e-5 | ≤2e-16, same n |
| sep_T9x_4R | 67.616043 / 67.616053 (1.4e-7) | 9.8e-7 | 9.6e-6 | ≤1e-16, same n |
| joint_T7x_eg_4R | 41.002840 / 41.002842 (6.4e-8) | 2.8e-7 | 4.8e-7 | ≤2e-16, same n |
| joint_T8xm_ec_4R | 39.499351 / 39.499353 (4.7e-8) | 1.9e-7 | 7.2e-6 | ≤1e-16, same n |
| joint_T7x_ec_4R (extra) | 40.259326 / 40.259327 (2.3e-8) | 9.5e-8 | 7.4e-6 | ≤2e-16, same n |
| joint_T8xm_4R (extra) | 39.141897 / 39.141897 (1.0e-8) | 5.4e-8 | 1.1e-5 | ≤2e-16, same n |

- Rows (one__/full__/E__): shells agree to 8e-15. Core and group integrals differ by at most 1.2e-5 (sep_T9x) and 3e-6 elsewhere, from the study's 12-point rule in x.
- The template equals the step-4 template, and the parity region-1 weights sum to 1853.0957.
- Outer RMS from my code:
  - joint_T9x: 0.098, 0.130, 0.131, 0.157.
  - joint_T7x_ec: 0.108, 0.128, 0.123, 0.129.

**PASS.**

## Check 2: optimality, all parameters free (law + 3 amplitude), objective L_4R + L_E

| Model | Objective mine (study) | Local least-squares decrease | 5 random starts (sigma 0.05): best minus reported | Lower by > 1e-3 |
|---|---|---|---|---|
| joint_T9x_4R (21 par) | 64.2307370 (64.2303802) | 1.3e-8 | -1.3e-8 | 0 |
| joint_T7x_ec_4R (20 par) | 65.3494750 (65.3492891) | 4.0e-9 | -4.0e-9 | 0 |

- Every random start returned to the reported solution:
  - joint_T9x: initial losses 178-2899, final parameters within 6.9e-6.
  - joint_T7x_ec: initial losses 571-1669, final parameters within 1.3e-5.
- The offsets between my objective and the study's (3.6e-4 and 1.9e-4) come only from the L_E integral rule. They do not mean a different optimum.

**PASS.**

## Check 3: electron laws with one extra constant

**electron_law_c (free x_c):**

- L_E at the saved parameters: 14.297273 (mine) against 14.297187 (study), relative 6.1e-6.
- Refit from the 9-parameter law plus ln x_c = ln 0.5: it starts at 15.1533 and converges to 14.297273.
- The refit gives ln_xc_e = -1.03009 (saved -1.03008). All parameters lie within 1.4e-5 of the saved values. No bound is active.

**electron_law_g (free inner slope):**

- L_E at the saved parameters: 13.486477 against 13.486506, relative 2.1e-6.
- Refit with the study's bound g ≤ 1: it starts at 15.1533 and reproduces the saved solution, with g = 1 at the bound and parameters within 2.1e-6.

**PASS.**

## Physics flag: the inner-slope bound is binding

Without the bound, g runs to +3.35 and L_E drops to 12.864.

Local density slope of the saved g law at M = 1e14, z = 0, compared with the production law:

| Radius (R200c) | Saved g law | Production law |
|---|---|---|
| 0.005 | +0.39 | -0.38 |
| 0.01 | +0.21 | -0.46 |
| 0.02 | about 0 | -0.58 |
| 0.05 | -0.38 | -0.78 |
| 0.1 | -0.70 | -0.98 |

The "eg" variants (electron_law_g, joint_T7x_eg_4R, joint_T8xm_eg_4R) therefore:

- have a hollow electron core inside about 0.02 R200c;
- are shallower than production out to 0.1 R200c;
- get their L_E gain from an unphysical inner slope that sits at its bound.

The "ec" variants have no active bounds and are the cleaner one-parameter extension.

## Not verified

- joint_T8xz_4R, sep_T8xm_4R and joint_T8xm_eg_4R.
- The Y200, extension and physicality blocks of these outputs.
