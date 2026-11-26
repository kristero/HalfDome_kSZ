# BE step 5 (electron 2-halo template; pressure and electrons fitted to 4 R200c): independent verification, 2026-11-26

## Verdict

All four checks pass.

## What I used

I wrote the code from the docstrings of `be_twohalo_e.py`, `be_etemplate.py` and `be_fit4r_e.py`, plus the
coordinator's brief. No study code was read or copied.

What I took from `bd_twohalo` (imported read-only):

- cosmology;
- the Sheth-Mo-Tormen bias;
- the Sheth-Tormen dn/dlnM;
- comoving R200c;
- xi_lin;
- `b12_pe`, used only for the pressure self-check.

What I wrote myself:

- the Battaglia+16 electron neighbour profile;
- the normalisation ratio S;
- the exclusion integral;
- the row averaging.

The fits reuse my step-4 vectorized evaluator: 48-point Gauss-Legendre in ln x.

Run details:

- STRW maas, nice 19, one thread.
- Template check: 23 s, after a 1 min table set-up. Fits check: 282 s.
- Scripts: `step5/scripts/step5_template.py`, `step5_fits.py`.
- Results: `step5/out/step5_template.json`, `step5_fits.json` and the logs.
- The laptop copies have identical SHA-256.

## Check 1: electron 2-halo template, exclusion 0.75

**Formula (from the docstring).**

    t(r | M, z) = b(M) ∫ dlnM' n(M') b(M') S ∫_{|s| ≥ 0.75 (R200c + R200c')} d³s ξ_lin(s) F'(|r − s| | M')

- Neighbours run from 1e10 to 1e16 Msun.
- F' is truncated at 4 R200c'.
- Electrons use S = 1.
- F' is the B16 AGN profile in E units:
  - E' = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a]^(-(b−0.2)/a)
  - n0 = 4e3 m^0.29 (1+z)^-0.66
  - a = 0.88 m^-0.03 (1+z)^0.19
  - b = 3.83 m^0.04 (1+z)^-0.025
- The mean term replaces b b' ξ by 1 + b b' ξ.

**How I computed it.**

- The angular part is done analytically:
  ∫ d³s w(s) F'(|r − s|) = (2π/r) ∫ ds s w(s) ∫_{|r−s|}^{min(r+s, 4R')} du u F'(u).
- Mass integral: 96-point Gauss-Legendre in ln M'.
- s integral: 48 nodes per segment, split at s = r and s = 4R' − r.
- u integral: 16 nodes.
- Row averaging: 8 nodes per shell; 24 nodes on [0, hi] for core and group integrals.

**Electron result.** I compared 22 random rows with `etemplate_ex0p75.npy` and `etemplate_ex0p75+mean.npy`: 8 from
region 0, 7 from region 1, 5 from region 2, plus 2 extra core-integral rows.

| Template | Largest difference | Median difference |
|---|---|---|
| ex0p75 (no mean term) | 1.8e-4 | 5.5e-5 |
| ex0p75+mean | 1.7e-4 | 5.6e-5 |

**Pressure self-check of my exclusion integral.** I swapped in B12 neighbours with S = (M'/M)^(2/3). On 6 rows this
reproduces the BD pressure template (`template_P`) within 2.8e-4, median about 4e-5.

The residual of about 1e-4 is consistent with the study's coarser quadrature: the `template_excl` defaults are nr = 6
and ns = 64. It is the same size as my residual on the pressure template.

`etemplate_check.json` reports `max_rel_diff_vs_BD_pressure_template` = 0.0 on 60 rows. So the study's own field 'P'
check reproduces the BD template exactly.

Across all rows, the mean term raises the template by a factor of 1.034 to 1.434.

**PASS.**

## Check 2: weights

| Sum | Mine | `results.weights` | Rel |
|---|---|---|---|
| P, region 0 | 2700.9999999999995 | 2700.9999999999995 | 0 |
| P, region 1 (parity, v from P_err/d_P) | 1853.095716198126 | 1853.0957161981255 | 2e-16 |
| E, region 0 | 2700.9999999999995 | 2700.9999999999995 | 0 |
| E, region 1 (parity, v_E = 1/(0.01 + (E_err/d_E)²)) | 1853.0957161981257 | 1853.0957161981253 | 2e-16 |

**PASS.**

## Check 3: both-field losses from saved parameters

Model definitions:

- Pressure: P = P1h + A_P(z) t^P.
- Electrons: E = E1h + A_E(z) t^E.
- In both, ln A = a + b lz + c lz², where lz = ln(1+z).
- Templates: t^P = `template_P`; t^E = `template_E_ex0p75+mean` (the study's chosen electron template).

| Model | L_4R_P mine / study (rel) | L_4R_E mine / study (rel) | L_P rel | L_E mine / study (rel) | L_out6, L_out6_E |
|---|---|---|---|---|---|
| elaw_2h_ex0p75+mean | — | 52.109853 / 52.109577 (5.3e-6) | — | 35.163791 / 35.163514 (7.9e-6) | — / 1355.150912 |
| split_const | 35.836777 / 35.836754 (6.5e-7) | 52.109853 / 52.109577 (5.3e-6) | 1.4e-6 | 7.9e-6 | rel 0 / 0 |
| joint_T7x | 38.328091 / 38.328091 (2.1e-8) | 60.624457 / 60.624160 (4.9e-6) | 6.6e-8 | 44.266880 / 44.266583 (6.7e-6) | rel 2e-16 / 2e-16 |

Row values against `one__`, `full__`, `oneE__` and `fullE__`:

- Shells agree to 8e-15.
- Core and group integrals differ by at most 2.1e-6 for P and 1.4e-5 for E. As in earlier steps, that is the study's
  12-point integral rule in x.

Outer all-mass RMS per band:

- Both fields agree with the JSON to 1.4e-16 or better, with identical counts.
- joint_T7x, pressure: 0.080, 0.119, 0.124, 0.155.
- joint_T7x, electrons: 0.092, 0.093, 0.073, 0.134.

**PASS.**

## Check 4: optimality, all parameters free

Objectives:

- elaw_2h_ex0p75+mean: 9 + 3 parameters; objective L_4R_E.
- joint_T7x: 16 + 3 + 3 parameters; objective L_4R_P + L_4R_E.

| Model | Objective mine (study) | Local LS decrease | 5 random starts (σ 0.05): best minus reported | Lower by > 1e-3 |
|---|---|---|---|---|
| elaw_2h_ex0p75+mean | 52.1098535 (52.1095768) | 6.7e-9 | -6.7e-9 | 0 |
| joint_T7x | 98.9525489 (98.9522511) | 3.9e-9 | -3.9e-9 | 0 |

- elaw_2h_ex0p75+mean: random starts from losses 365-2201 all return to the reported point, within 7.9e-6 in every
  parameter.
- joint_T7x: random starts from losses 748-2966 all return, within 2.4e-5.
- My objectives sit 2.8e-4 to 3.0e-4 above the study's. That is the L_E integral-rule offset on the core rows, not a
  different optimum.

**PASS.**

## Not verified

- elaw_2h_ex0p75, split_massbreak, sep_T7x, joint_T9x and joint_H6 (only shared ingredients were checked).
- The `bands_inside`, `physicality_E` and `amp_by_z` blocks.
- The template on every row: I checked a random sample of 22 electron and 6 pressure rows.
