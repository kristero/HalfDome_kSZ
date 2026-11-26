## Table 6. Step 3: existing pressure fits with free 2-halo amplitude vs electron-law models with the same 2-halo term
Source: `out/be_compare.json: results[name].{counts.total_with_electrons, L_P, L_E, L_out4, L_out6, bands_inside[group|all].rms, outer_all_masses[band].rms, amp_by_z}`. Existing fits use their saved parameters; their recomputed L_P equals the saved L_in (`recomputed_minus_saved`).

| model | parameters: electron + pressure + amplitude | L_P inside B32 | L_E | L_out to 4 R200c | L_out to 6 R200c | RMS inside: 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >= 14.3 | RMS beyond B32 (all masses): 1.5-2 / 2-3 / 3-4 / 4-6 R200c | A_2h at z = 0 / 1 / 3 |
|---|---|---|---|---|---|---|---|---|
| existing pressure fit, constant alpha, A_2h(z) free (amplitude to 6 R200c) | 9 + 10 + 3 = 22 | 14.33 | 15.15 | 1506 | 1869 | 0.581 / 0.269 / 0.072 / 0.088 | 0.090 / 0.097 / 0.117 / 0.116 | 0.52 / 1.69 / 2.82 |
| existing pressure fit, constant alpha, A_2h(z) free (to 4 R200c) | 9 + 10 + 3 = 22 | 14.33 | 15.15 | 1472 | 1937 | 0.581 / 0.269 / 0.072 / 0.088 | 0.090 / 0.093 / 0.113 / 0.135 | 0.48 / 1.76 / 2.92 |
| existing pressure fit, alpha(z) of record, A_2h(z) free (to 6 R200c) | 9 + 9 + 3 = 21 | 12.34 | 15.15 | 2905 | 3736 | 0.599 / 0.280 / 0.067 / 0.073 | 0.183 / 0.177 / 0.138 / 0.186 | 0.54 / 1.77 / 4.78 |
| existing pressure fit, alpha(z) of record, A_2h(z) free (to 4 R200c) | 9 + 9 + 3 = 21 | 12.34 | 15.15 | 2636 | 4194 | 0.599 / 0.280 / 0.067 / 0.073 | 0.158 / 0.142 / 0.178 / 0.260 | 0.50 / 1.88 / 6.20 |
| existing pressure fit with smooth mass break (16 law parameters), A_2h(z) free (to 4 R200c) | 9 + 16 + 3 = 28 | 2.00 | 15.15 | 2701 | 4119 | 0.148 / 0.089 / 0.035 / 0.031 | 0.157 / 0.176 / 0.188 / 0.249 | 0.47 / 1.73 / 5.79 |
| electron law x temperature law (exponential outer decline) + 2-halo, joint (to 6 R200c) | 9 + 7 + 3 = 19 | 7.68 | 15.82 | 3912 | 4279 | 0.247 / 0.081 / 0.072 / 0.067 | 0.207 / 0.186 / 0.141 / 0.115 | 0.61 / 1.63 / 2.34 |
| same, amplitude to 4 R200c | 9 + 7 + 3 = 19 | 7.67 | 15.82 | 3878 | 4343 | 0.247 / 0.081 / 0.072 / 0.067 | 0.203 / 0.183 / 0.154 / 0.132 | 0.63 / 1.63 / 2.03 |
| electron law x temperature law (exponential outer decline) + 2-halo, separate (to 6 R200c) | 9 + 7 + 3 = 19 | 9.09 | 15.15 | 6270 | 6815 | 0.244 / 0.083 / 0.080 / 0.073 | 0.274 / 0.270 / 0.174 / 0.146 | 0.53 / 1.42 / 1.45 |
| same, amplitude to 4 R200c | 9 + 7 + 3 = 19 | 9.08 | 15.15 | 6080 | 7453 | 0.243 / 0.084 / 0.080 / 0.073 | 0.268 / 0.256 / 0.184 / 0.243 | 0.50 / 1.26 / 0.76 |
| electron law x temperature law + 2-halo, joint (to 6 R200c) | 9 + 7 + 3 = 19 | 7.74 | 15.83 | 4172 | 4559 | 0.247 / 0.081 / 0.072 / 0.067 | 0.214 / 0.200 / 0.147 / 0.119 | 0.58 / 1.57 / 2.05 |
| same, amplitude to 4 R200c | 9 + 7 + 3 = 19 | 7.74 | 15.83 | 4130 | 4669 | 0.246 / 0.081 / 0.072 / 0.067 | 0.211 / 0.196 / 0.158 / 0.146 | 0.58 / 1.53 / 1.62 |
| electron law x temperature law + 2-halo, separate (to 6 R200c) | 9 + 7 + 3 = 19 | 9.40 | 15.15 | 5695 | 6221 | 0.240 / 0.088 / 0.079 / 0.073 | 0.254 / 0.253 / 0.175 / 0.143 | 0.53 / 1.42 / 1.34 |
| same, amplitude to 4 R200c | 9 + 7 + 3 = 19 | 9.39 | 15.15 | 5558 | 6683 | 0.240 / 0.088 / 0.079 / 0.073 | 0.249 / 0.242 / 0.183 / 0.218 | 0.50 / 1.28 / 0.74 |
| hydrostatic + core term + 2-halo, joint (to 6 R200c) | 9 + 6 + 3 = 18 | 5.56 | 16.30 | 42019 | 49273 | 0.227 / 0.096 / 0.060 / 0.065 | 0.887 / 0.307 / 0.464 / 0.563 | 1.02 / 2.89 / 5.59 |
| same, amplitude to 4 R200c | 9 + 6 + 3 = 18 | 5.55 | 16.31 | 39330 | 54753 | 0.233 / 0.095 / 0.060 / 0.065 | 0.716 / 0.315 / 0.714 / 0.824 | 1.32 / 3.74 / 8.55 |

Recomputed minus saved L_in of the existing fits: B6: 0.0e+00, B4: 0.0e+00, A6: -1.8e-15, A4: 0.0e+00, MB4: 2.2e-16.

Temperature law with exponential outer decline, joint + 2-halo to 6 R200c (`results.joint_T7x_rout6.params`): lnT0 = 0.7103, am_T = -0.1306, az_T = -0.1886, ln_xe = -0.2605, c_e = 0.9343, lnxcc0 = -2.734, bm_cc = -0.6047. Physicality: 109/109 bins decline monotonically, log slope at 3-4 R200c -6.1 [-8.1, -5.0].
