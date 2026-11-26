# BE independent verification (2026-11-26)

Independent check of the BE study ("electron pressure from an electron-density law plus a temperature model"),
FLAMINGO L1_m9, BD residual set. I wrote all code from the written definitions: the module docstrings of
`code/be_fit.py` and `code/be_sums.py`. I copied no study function. I opened no study code beyond those two
docstrings, so the explanations of small differences below come from replicating the documented rules myself.

- Where it ran: STRW maas, `nice -n 19`, one thread, peak RSS about 0.1 GB. Total time about 6 minutes.
- maas folder: `/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/`
  - `scripts/verify_be.py`: checks 1-6.
  - `scripts/diag_be.py`, `diag2_be.py`, `diag3_be.py`: diagnostics of the small differences.
  - `out/*.json`, `out/*_log.txt`: results. SHA-256 checksums are in `out/sha256_maas.txt`; the laptop copy has the same checksums.
- Inputs, all read-only: the study's `out/be_fits.json`, `be_fits2.json`, `be_fits_rows.npz`, `be_fits2_rows.npz`,
  `be_heldout.json` and `be_sums.npz`; BD `results/flamingo_residuals.{json,npz}`; and the BC atlas `atlas_E.npz`
  (sha256 8fa15c14…554e8), copied to `verify_independent/input/`.

Tolerances: model values and losses pass at a relative difference of 1e-3. Data identities pass at 1e-10.

## Symbols

- x = r/R200c; M = M200c in Msun; z = redshift; lm = ln(M/1e14 Msun); lz = ln(1+z).
- For each law quantity Q: ln Q = lnA_Q + am_Q lm + az_Q lz.
- P = P_e/P200c, with P200c = G M 200 rho_crit f_b/(2 R200c).
- E = n_e mu_XGP/(rho_crit f_b), with mu_XGP = (m_e + 2 m_p/1.76)/0.9.
- tau = T_e/T200, with T200 = 0.59 m_p G M/(2 R200c k_B).
- P_tot = total thermal pressure. Its hydrostatic integral runs out to x_out, and P_tot(x_out) = 0.
- f_nth = a (1+z)^b x^n, capped at 0.95.
- Every model is evaluated at M = 10^node and at z of the row's BD block.
- A shell row's model value is the volume mean over [lo, hi]. A core or group row's value is 4 pi int_0^hi x^2 f dx.
- Losses use region 0 only: L_P = sum_i wP_i ln^2(P_i/d_P,i) and L_E = sum_i wE_i ln^2(E_i/d_E,i).

My quadrature integrates in u = ln x with 48-point Gauss-Legendre. Integral rows use composite panels down to 1e-9 hi.
The integration is split where f_nth reaches its 0.95 cap.

The hydrostatic integral K int_x^x_out E m/s^2 ds uses 4000 log panels with 12 Gauss-Legendre points each, with exact
cumulants at the panel edges. Here m(s) is the NFW enclosed mass with the Duffy+08 concentration
c200c = 5.71 (M h/2e12)^-0.084 (1+z)^-0.47, and h = 0.681.

Convergence: quadrupling the panels and moving the lower limit changes P_tot by 1.4e-15. On 10 random rows, scipy
`quad` with a nested `quad` for P_tot agrees with my values to 2.9e-13 (H6) and 1.5e-14 (T7).

## Results

| # | Check | Mine | Study (file : field) | Rel. diff | Verdict |
|---|---|---|---|---|---|
| 1 | mu_XGP (scipy.constants) | 1.2632313926501406 | be_fits.json : constants.MU_XGP | 0 | PASS |
| 1 | C_T = 200 mu_XGP/(0.59 m_p) | 428.21403140682736 | constants.C_T | 0 | PASS |
| 1 | K_HSE = 0.01 (2/1.76) m_p/mu_XGP | 0.00899568869943655 | constants.K_HSE | 0 | PASS |
| 1 | PE_FRAC = (2+2X)/(3+5X), X = 0.76 | 0.5176470588235295 | constants.PE_FRAC | 0 | PASS |
| 1 | P = E tau/C_T, numerically in SI | — | — | 0 | PASS |
| 1 | dP_tot/dx = -0.01 g E m/x^2, numerically in SI | — | — | 1.1e-16 | PASS |
| 2 | plaw_held L_P | 12.079708 (exact PE_FRAC); 12.079686 (0.5176) | be_fits.json : results.plaw_held.L_P = 12.079656 | 4.4e-6 / 2.5e-6 | PASS |
| 2 | electron_law L_E | 15.153339 | results.electron_law.L_E = 15.153193 | 9.6e-6 | PASS |
| 2 | joint_T7 L_P / L_E | 7.7535523 / 15.880503 | results.joint_T7.L_P = 7.7535531 / L_E = 15.880337 | 1.1e-7 / 1.0e-5 | PASS |
| 2 | sep_H6 L_P / L_E | 9.241927 / 15.153339 | results.sep_H6.L_P = 9.241154 / L_E = 15.153193 | 8.4e-5 / 9.6e-6 | PASS |
| 2 | joint_H6 L_P / L_E | 5.656548 / 16.241180 | results.joint_H6.L_P = 5.656373 / L_E = 16.241044 | 3.1e-5 / 8.3e-6 | PASS |
| 2 | joint_H6_xout6 L_P / L_E | 5.637896 / 16.288382 | be_fits2.json : results.joint_H6_xout6.L_P = 5.637708 / L_E = 16.288249 | 3.3e-5 / 8.2e-6 | PASS |
| 2 | Y200 model/data, joint_H6, 6 blocks | 1.010290, 0.955068, 1.054938, 1.030547, 1.012951, 0.983365 | results.joint_H6.Y200[*].Y200_model_over_data = 1.010301, 0.955084, 1.054965, 1.030559, 1.012967, 0.983375 | 1.1e-5 to 2.5e-5 | PASS |
| 2 | Y200, joint_H6_xout6, 6 blocks | 1.014693 … 0.984889 | be_fits2.json : results.joint_H6_xout6.Y200 | 1.1e-5 to 2.5e-5 | PASS |
| 3 | Record law, BD coefficients on BD targets and weights | 12.0799522096442 | BD json : lsq_fixed_alpha.loss = 12.0799522096442 | 2e-15 | PASS |
| 3 | plaw_held refit from the study's point (my least squares) | 12.079686, max parameter shift 9.2e-5 | results.plaw_held.params | — | PASS: the study's point is the minimum |
| 4 | d_E = volume mean of sum_k S_E_all / sum_k W | — | be_fits_rows.npz : d_E | max 4.8e-15 | PASS |
| 4 | d_P = volume mean of sum_k S_P_all / sum_k W | — | be_fits_rows.npz : d_P | max 4.7e-15, all regions | PASS |
| 4 | U_E_a2ii/N against BC atlas prof_a2ii | 109 blocks, 7630 shells | atlas_E.npz : prof_a2ii | max 1.1e-15, median 2.2e-16 | PASS |
| 4 | wE from the documented formula | — | rows : wE | 0 (1425 rows); the 69 core rows have prio1 = 0, so wE = 0 | PASS |
| 5 | Held-out totals = sum of per-output entries | 12.914189, 15.720845, 13.435634, 15.210348, 11.927468, 8.749815, 6.638002; L_E 19.391115 | be_heldout.json : results[m].total_L_P_test / total_L_E_test | 0 | PASS |
| 5 | Leave-one-output-out refit of plaw_held (my own fits) | 12.914220 | results.plaw_held.total_L_P_test = 12.914189 | 2.4e-6 | PASS |
| 6 | tau = C_T d_P/d_E, clusters at z = 0 | 0.1 R: 1.23-1.55; 0.5 R: 0.73-0.91; 1 R: 0.46-0.54 | rows : tau (identical to C_T d_P/d_E) | — | PASS: plausible |

### Check 1 details

**P = E tau / C_T.** Start from P_e = n_e k_B T_e. Using G M/(2 R200c) = k_B T200/(0.59 m_p):

- P200c = 200 rho_crit f_b k_B T200/(0.59 m_p)
- P = n_e T_e 0.59 m_p/(200 rho_crit f_b T200) = E tau · 0.59 m_p/(200 mu_XGP) = E tau / C_T

**Hydrostatic equation.** Start from dP_tot/dr = -G M(<r) rho_gas/r^2, with:

- r = x R200c
- M(<r) = m(x) M200c
- rho_gas = g E rho_crit f_b

Dividing by P200c gives d(P_tot/P200c)/dx = -0.01 g E m/x^2.

With rho_gas = n_e 2 m_p/1.76, g = (2/1.76)/mu_XGP = 0.899569, so K = 0.01 g reproduces K_HSE exactly. I also checked
both formulas numerically in SI units with arbitrary halo values.

Note: if rho_gas included the electron mass, K would be 0.009 exactly. That is 4.8e-4 relative to the study's value,
a convention that f_nth absorbs.

### Check 2 details: region-0 model values, mine against P__/E__ in the rows files

| Model | P max / median rel. diff | E max / median rel. diff |
|---|---|---|
| plaw_held | 9.49e-5 / 9.09e-5 | — |
| electron_law (E__sep) | — | 1.0e-5 / 3e-16 |
| joint_T7 | 1.97e-6 / 4e-16 | 9.2e-6 / 3e-16 |
| sep_H6 | 9.09e-5 / 1.04e-5 | 1.0e-5 / 3e-16 |
| joint_H6 | 8.93e-5 / 1.08e-5 | 1.13e-5 / 3e-16 |
| joint_H6_xout6 | 8.89e-5 / 1.08e-5 | 1.15e-5 / 3e-16 |

Every difference is below 1e-4. The residuals have three explanations, each tested:

- **plaw_held:** the ratio mine/study is constant (shells: 1.0000909174, std 6e-16) and equals 0.517647/0.5176.
  The study, like the BD record, uses the literal prefactor **0.5176** for the pressure law of record, not
  (2+2X)/(3+5X). With 0.5176, my BD-coefficient loss reproduces the record to 2e-15.
- **Core and group integrals:** my replica of "12-point Gauss-Legendre in x over [0, hi]" reproduces the study's E
  integrals to 7.8e-16. Near x = 0 the integrand behaves like x^1.7-1.8, so that rule has an error of about 1e-5.
  My panelled ln-x rule is more accurate. Shell means agree to machine precision.
- **H6 models:** the study's hydrostatic pressure is slightly high, and the bias grows outward. The median difference
  goes from 0.8e-6 at x < 0.1 to 3.6e-5 at x = 1-1.6, with a maximum of 9e-5. The study's P_tot quadrature therefore
  carries a small systematic error. My P_tot is converged to 1.4e-15 and confirmed by nested `quad`.

The blocks used for the Y200 check are 77/12.9-13.1, 77/13.9-14.1, 77/14.9-15.1, 57/12.9-13.1, 57/13.9-14.1 and
37/12.9-13.1. Y200 = 4 pi int_0^1 x^2 P dx. For the data I used uniform cell values of sum_k S_P_all / sum_k W from
be_sums.npz.

### Check 3 details

- With the docstring's rounded alpha (0.9394 + 0.2948 z), L_P becomes 12.079781 instead of 12.079708. The study and I
  both use BD alpha_held = [0.93935, 0.29477].
- My least-squares refit from the study's parameters moves only lnA_P0, by -9.2e-5 = ln(0.5176/0.517647). The other
  eight parameters move by less than 1e-5.
- The BD record targets differ from BE's targets:
  - BD `B32__target_flamingo_Y` is the **a2ii (halo-component) stack**: it matches my a2ii volume means to 4.7e-15.
  - BE's d_P is the **all-gas stack**, as the BE docstring says.
  - The two differ in 162 outer rows (hi ≥ 0.837) by up to 7.85e-4.
  - Effect on the loss: the BD coefficients give L_P = 12.079952 on the BD targets and 12.079686 on the BE targets.
  - Comparing 12.0797 (BE) with 12.08 (BD) is therefore fair, although the targets differ very slightly.

### Check 4 details

- The region-0 row geometry is identical to the BD B32 rows: block, lo, hi and kind match exactly.
- wP equals `B32__weight` exactly; sum wP = sum wE = 2701.
- d_E also equals `be_sums.npz : resid_E_mean` to 1.8e-15.
- 105 of the 1494 rows have an edge that does not lie on a storage edge. For those I took partial overlaps with
  uniform cell values, as the docstring specifies, and the identity still holds to 5e-15.
- Atlas comparison: I mapped each snapshot to its redshift through the BD block list (77 → z = 0 … 17 → z = 3). Every
  atlas edge is also a storage edge, so every atlas shell is a union of storage cells, which I combined by volume.
  - All 109 blocks were compared, and BE N equals atlas n in every block.
  - This is the strongest data check, because the BC atlas came from a separate pipeline reading the same production
    files.
- In rows npz, the column `P_a2ii` holds the a2ii stack, not the all-gas stack, so the column labels are correct.

### Check 5 details

The joint models have no stored L_E total. Summing their per-output entries gives L_E_test = 17.790 (joint_T7) and
17.917 (joint_H6), against 19.391 for the electron law alone. That confirms the "17.8-17.9 vs 19.39" claim.

Held-out pressure ordering:

joint_H6 6.638 < joint_T7 8.750 < sep_H6 11.927 < plaw_held 12.914 < sep_T7 13.436 < sep_H3 15.210 < plaw_const 15.721.

My own leave-one-output-out refits of plaw_held match every per-output L_P_test to 4e-6 relative or better.

### Check 6 details

Region-0 shells containing the radius, blocks with node ≥ 14 (14.1-15.3) at z = 0:

| Radius | n blocks | tau range |
|---|---|---|
| 0.1 R200c | 4 | 1.23-1.55 |
| 0.5 R200c | 6 | 0.73-0.91 |
| 1 R200c | 6 | 0.46-0.54 |

The 14.1-14.5 blocks have core integrals, not shells, at 0.1 R200c, which is why only 4 blocks appear there.

From the stacks' storage cells, tau = C_T sum_k S_P_all / sum_k S_E_all for the 13.9-15.3 blocks:

| Radius | tau range |
|---|---|
| 0.1 R200c | 1.22-1.80 |
| 0.5 R200c | 0.73-0.96 |
| 1 R200c | 0.48-0.59 |

tau falls with radius, and at fixed x it falls with mass, as expected since T200 ∝ M^(2/3). The range matches the
expected 0.5-1.6, with small excursions: down to 0.46 at R200c for the most massive blocks, and up to 1.8 at
0.1 R200c for 13.9-14.1.

## Findings worth attention (none fails the brief)

1. **Shells straddling the f_nth cap (beyond B32 only).**
   - In region 1 there is one such shell per block, 109 in total. There the study's unsplit 12-point rule is off by
     up to **0.53 %** (joint_H6) and 0.25 % (joint_H6_xout6) from the accurate shell mean.
   - My unsplit 12-point replica reproduces the study's values to 1.1e-4, which confirms the cause.
   - These rows are not in the region-0 losses. They do enter the region-1 2-halo amplitude fits and the extension
     diagnostics.
   - Suggested fix: split the quadrature at x_cap = (0.95/(a(1+z)^b))^(1/n), or use more nodes.
   - I did not re-verify `be_twohalo.json`.
2. **P_tot quadrature bias near x_out.** For x_out = 4, the study's P_tot error grows to 4.2e-4 at 3.5-4 R200c,
   because P_tot → 0 there. It stays below 1.6e-4 elsewhere beyond B32. For x_out = 6 it stays below 1.5e-4 to
   4 R200c and reaches 2.7e-4 at 4-6 R200c.
3. **x_out = 4 profiles reach exactly zero at 4 R200c by construction.** joint_H6 `physicality.slope_3_4R_median` is
   -2239, and the region-2 law-only band rms is inf because ln 0 diverges. The x_out = 6 variant has almost the same
   L_P (5.638) and outer slopes from -5.3 to -3.0. That matters if the profile is painted to 4 R200c.
4. **joint_T7 has c_out = 9.999999999999998 with x_t = e^2.02 = 7.5.** The outer temperature decline sits on an
   apparent bound of 10, and c and x_t are degenerate.
5. **The record law uses the rounded 0.5176 prefactor.** Painting code must use 0.5176 with these coefficients, or
   expect 9.1e-5 higher pressure. The core and group integrals use a 12-point rule in x with about 1e-5 error. Both
   are negligible.
6. **The electron law has two local minima** in its start losses (15.153 and 15.449); the best one is kept.
   joint_H6_xout6 reaches the f_nth cap at x = 1.5 for z ≥ 1, and f_nth = 0.67 at R200c for z = 2.

## Not verified

- The T7 and H6 optimisations themselves: I checked their losses at the reported parameters, but not that those
  parameters are minima.
- The per-halo scatter f_E behind wE, which needs halo-level data. I recomputed wE only from the stored scatter.
- The 2-halo fits.
