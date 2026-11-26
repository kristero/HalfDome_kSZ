## Table 1. Inside the cut: losses and RMS by mass group (`out/be_fits.json`, `out/be_fits2.json`: results[name].L_P, L_E, k_P, bands_P[group|all].rms, Y200)

| model | fit | pressure params | density + pressure params | L_P | L_E | RMS ln P: 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >=14.3 | Y200 model/data at 10^13, z=0 / 10^14, z=0 / 10^15, z=0 |
|---|---|---|---|---|---|---|---|
| pressure law of record, alpha(z) held | pressure law alone | 9 | 18 | 12.08 | 15.15 | 0.592 / 0.277 / 0.066 / 0.073 | 2.03 / 0.94 / 1.00 |
| pressure law, one constant alpha | pressure law alone | 10 | 19 | 14.09 | 15.15 | 0.574 / 0.266 / 0.072 / 0.088 | 1.95 / 0.94 / 1.06 |
| electron law x temperature, no core term | separate | 9 (shared) + 5 | 14 | 14.03 | 15.15 | 0.531 / 0.298 / 0.120 / 0.068 | 1.03 / 0.91 / 1.09 |
| electron law x temperature law | separate | 9 (shared) + 7 | 16 | 9.50 | 15.15 | 0.241 / 0.092 / 0.080 / 0.073 | 0.88 / 0.92 / 1.06 |
| electron law x temperature law, z-dependent core | separate | 9 (shared) + 8 | 17 | 9.41 | 15.15 | 0.217 / 0.103 / 0.082 / 0.071 | 0.93 / 0.92 / 1.05 |
| hydrostatic | separate | 9 (shared) + 3 | 12 | 13.24 | 15.15 | 0.539 / 0.295 / 0.089 / 0.067 | 1.24 / 1.01 / 1.07 |
| hydrostatic, free concentration | separate | 9 (shared) + 4 | 13 | 12.93 | 15.15 | 0.548 / 0.301 / 0.092 / 0.064 | 1.24 / 1.01 / 1.07 |
| hydrostatic + core term | separate | 9 (shared) + 6 | 15 | 9.24 | 15.15 | 0.274 / 0.112 / 0.072 / 0.079 | 0.97 / 0.99 / 1.06 |
| hydrostatic + core term, P = 0 at 3 R200c | separate | 9 (shared) + 6 | 15 | 8.94 | 15.15 | 0.249 / 0.113 / 0.074 / 0.077 | 0.96 / 0.98 / 1.07 |
| hydrostatic + core term, P = 0 at 6 R200c | separate | 9 (shared) + 6 | 15 | 9.43 | 15.15 | 0.293 / 0.113 / 0.071 / 0.081 | 0.97 / 0.99 / 1.06 |
| electron law x temperature law | joint | 9 (shared) + 7 | 16 | 7.75 | 15.88 | 0.245 / 0.084 / 0.072 / 0.067 | 0.94 / 0.92 / 1.06 |
| electron law x temperature law, z-dependent core | joint | 9 (shared) + 8 | 17 | 7.70 | 15.87 | 0.222 / 0.095 / 0.073 / 0.065 | 0.97 / 0.92 / 1.05 |
| hydrostatic | joint | 9 (shared) + 3 | 12 | 7.94 | 16.72 | 0.520 / 0.277 / 0.085 / 0.054 | 1.24 / 0.97 / 1.05 |
| hydrostatic + core term | joint | 9 (shared) + 6 | 15 | 5.66 | 16.24 | 0.198 / 0.102 / 0.064 / 0.065 | 1.01 / 0.96 / 1.05 |
| hydrostatic, smooth non-thermal fraction, 6 R200c | separate | 9 (shared) + 3 | 12 | 13.49 | 15.15 | 0.544 / 0.299 / 0.086 / 0.065 | 1.26 / 1.01 / 1.07 |
| hydrostatic + core, smooth non-thermal fraction, 6 R200c | separate | 9 (shared) + 6 | 15 | 9.14 | 15.15 | 0.293 / 0.114 / 0.063 / 0.075 | 0.97 / 0.99 / 1.06 |
| hydrostatic + core term, P = 0 at 6 R200c | joint | 9 (shared) + 6 | 15 | 5.64 | 16.29 | 0.209 / 0.100 / 0.062 / 0.066 | 1.01 / 0.96 / 1.05 |
| hydrostatic, smooth non-thermal fraction, 6 R200c | joint | 9 (shared) + 3 | 12 | 7.83 | 16.82 | 0.524 / 0.279 / 0.082 / 0.052 | 1.26 / 0.97 / 1.05 |
| hydrostatic + core, smooth non-thermal fraction, 6 R200c | joint | 9 (shared) + 6 | 15 | 5.44 | 16.27 | 0.209 / 0.101 / 0.058 / 0.062 | 1.02 / 0.96 / 1.05 |

Electron law alone (9 parameters): L_E = 15.15 (`results.electron_law.L_E`).

## Table 2. Leave one output out (`out/be_heldout.json`: results[name].total_L_P_test and per-redshift L_P_test)

| model | held-out L_P, sum of 10 outputs | z=0 | z=0.15 | z=0.2 | z=0.3 | z=0.5 | z=0.8 | z=1 | z=1.5 | z=2 | z=3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure law of record, alpha(z) held | 12.91 | 1.26 | 1.23 | 1.29 | 2.50 | 2.06 | 1.49 | 1.47 | 0.69 | 0.68 | 0.24 |
| pressure law, one constant alpha | 15.72 | 1.24 | 1.31 | 1.44 | 2.91 | 2.27 | 1.58 | 1.82 | 1.03 | 1.36 | 0.76 |
| separate: electron law x temperature law | 13.44 | 1.13 | 1.20 | 1.23 | 2.09 | 1.36 | 0.84 | 3.39 | 0.72 | 1.04 | 0.42 |
| separate: hydrostatic | 15.21 | 0.78 | 0.69 | 0.66 | 1.18 | 1.39 | 1.88 | 4.98 | 1.96 | 1.50 | 0.20 |
| separate: hydrostatic + core term | 11.93 | 0.81 | 0.69 | 0.64 | 1.09 | 0.99 | 1.01 | 4.41 | 1.03 | 0.97 | 0.28 |
| joint: electron law x temperature law | 8.75 | 0.54 | 0.44 | 0.97 | 1.63 | 1.06 | 0.72 | 1.36 | 0.62 | 0.96 | 0.46 |
| joint: hydrostatic + core term | 6.64 | 0.55 | 0.36 | 0.55 | 0.83 | 0.52 | 0.54 | 1.49 | 0.57 | 0.73 | 0.50 |
| electron law alone (held-out L_E) | 19.39 | 3.94 | 1.78 | 1.88 | 1.69 | 1.40 | 1.89 | 2.26 | 1.19 | 1.06 | 2.31 |

Held-out L_E of electron law x temperature law (joint), sum of 10 outputs: 17.79.

Held-out L_E of hydrostatic + core term (joint), sum of 10 outputs: 17.92.

## Table 3. 1-halo + 2-halo split fit (`out/be_twohalo.json`: results[<name>__rout<4|6>].L_P, amp_by_z, bands_outer[group|band].rms)

| model | amplitude shells to | L_P inside the cut | A_2h at z = 0 / 1 / 3 | outer RMS 1.5-2 / 2-3 / 3-4 / 4-6 R200c, 10^13.5-14.3 | same, >= 10^14.3 |
|---|---|---|---|---|---|
| pressure law, one constant alpha | 4 R200c | 14.33 | 0.48 / 1.76 / 2.92 | 0.05 / 0.09 / 0.12 / 0.11 | 0.05 / 0.09 / 0.12 / 0.19 |
| pressure law, one constant alpha | 6 R200c | 14.33 | 0.52 / 1.69 / 2.82 | 0.05 / 0.11 / 0.14 / 0.09 | 0.05 / 0.09 / 0.10 / 0.16 |
| pressure law of record, alpha(z) held | 4 R200c | 12.34 | 0.50 / 1.88 / 6.20 | 0.15 / 0.15 / 0.14 / 0.20 | 0.09 / 0.16 / 0.17 / 0.22 |
| pressure law of record, alpha(z) held | 6 R200c | 12.34 | 0.54 / 1.77 / 4.78 | 0.17 / 0.19 / 0.13 / 0.14 | 0.09 / 0.16 / 0.14 / 0.18 |
| separate: hydrostatic + core term | 4 R200c | 9.23 | 1.42 / 3.88 / 9.17 | 0.72 / 0.35 / 0.75 / 0.87 | 0.87 / 0.31 / 0.64 / 0.77 |
| separate: hydrostatic + core term | 6 R200c | 9.23 | 1.09 / 2.99 / 6.18 | 0.90 / 0.27 / 0.50 / 0.61 | 1.04 / 0.37 / 0.40 / 0.52 |
| separate: electron law x temperature law | 4 R200c | 9.39 | 0.50 / 1.28 / 0.74 | 0.25 / 0.27 / 0.21 / 0.20 | 0.14 / 0.17 / 0.16 / 0.19 |
| separate: electron law x temperature law | 6 R200c | 9.40 | 0.53 / 1.42 / 1.34 | 0.26 / 0.28 / 0.20 / 0.12 | 0.14 / 0.17 / 0.15 / 0.17 |
| joint: hydrostatic + core term | 4 R200c | 5.55 | 1.35 / 3.79 / 9.02 | 0.72 / 0.33 / 0.72 / 0.84 | 0.83 / 0.31 / 0.60 / 0.73 |
| joint: hydrostatic + core term | 6 R200c | 5.57 | 1.04 / 2.94 / 6.08 | 0.88 / 0.28 / 0.47 / 0.58 | 0.98 / 0.40 / 0.37 / 0.49 |
| joint: electron law x temperature law | 4 R200c | 7.74 | 0.58 / 1.53 / 1.62 | 0.20 / 0.21 / 0.19 / 0.12 | 0.12 / 0.16 / 0.14 / 0.14 |
| joint: electron law x temperature law | 6 R200c | 7.74 | 0.58 / 1.57 / 2.05 | 0.20 / 0.21 / 0.17 / 0.10 | 0.12 / 0.16 / 0.14 / 0.14 |

## Table 4. 1-halo law extended beyond B32 (`results[name].extension`, `physicality`)

Median law / halo-component stack and law / host-bound stack in 1.5-2, 2-3, 3-4 R200c; clusters >= 10^14.3 and 10^13.5-14.3. Physicality over 0.3-4 R200c for all 109 bins.

| model | clusters: law/halo | clusters: law/host-bound | 10^13.5-14.3: law/halo | monotonic decline (of 109) | log slope at 3-4 R200c, median [range] |
|---|---|---|---|---|---|
| pressure law of record, alpha(z) held | 0.84 / 0.60 / 0.41 | 5.74 / 4.42 / 3.40 | 0.89 / 0.69 / 0.39 | 109 | -5.8 [-8.4, -3.9] |
| pressure law, one constant alpha | 0.89 / 0.69 / 0.56 | 5.96 / 4.93 / 4.34 | 0.90 / 0.72 / 0.51 | 109 | -5.3 [-6.7, -4.0] |
| separate: electron law x temperature law | 1.03 / 0.84 / 0.71 | 6.75 / 5.60 / 5.30 | 1.18 / 1.14 / 1.09 | 109 | -4.5 [-6.6, -3.4] |
| joint: electron law x temperature law | 0.95 / 0.68 / 0.44 | 6.21 / 4.76 / 3.40 | 1.08 / 0.93 / 0.68 | 109 | -5.5 [-7.5, -4.3] |
| separate: hydrostatic + core term, P = 0 at 6 R200c | 0.11 / 0.12 / 0.16 | 0.74 / 0.83 / 1.19 | 0.14 / 0.17 / 0.25 | 109 | -3.7 [-5.3, -2.9] |
| joint: hydrostatic + core term, P = 0 at 6 R200c | 0.12 / 0.12 / 0.15 | 0.90 / 0.81 / 1.15 | 0.14 / 0.16 / 0.24 | 109 | -3.7 [-5.3, -3.0] |
| joint: hydrostatic + core, smooth non-thermal fraction, 6 R200c | 0.28 / 0.02 / 0.00 | 2.11 / 0.15 / 0.00 | 0.23 / 0.01 / 0.00 | 75 | -57.5 [-2201.9, 0.0] |

## Table 5. Fitted parameters of the two joint few-parameter models

- electron law x temperature law (joint, `be_fits.json: results.joint_T7.params`): lnA_n0 = 8.704, am_n0 = -0.3178, az_n0 = -0.7503, lnA_ae = -0.6165, am_ae = 0.354, az_ae = 0.5651, lnA_betae = 1.063, am_betae = 0.1501, az_betae = 0.187, lnT0 = 0.6883, am_T = -0.1313, az_T = -0.1936, ln_xt = 2.022, c_out = 10, lnxcc0 = -2.751, bm_cc = -0.6057
- hydrostatic + core term, P = 0 at 6 R200c (joint, `be_fits2.json: results.joint_H6_xout6.params`): lnA_n0 = 8.499, am_n0 = -0.2855, az_n0 = -0.5932, lnA_ae = -0.4944, am_ae = 0.3401, az_ae = 0.4565, lnA_betae = 1.115, am_betae = 0.1364, az_betae = 0.1179, a_nth = 0.2249, b_nth = 0.9946, n_nth = 2.72, lnxcc0 = -3.133, bm_cc = -0.7339, bz_cc = -0.207
- Non-thermal fraction of joint hydrostatic + core (6 R200c) at x = 0.5 / 1 / 1.5: z=0 0.03 / 0.22 / 0.68; z=1 0.07 / 0.45 / 0.95; z=2 0.10 / 0.67 / 0.95 (cap 0.95; `be_fits2.json: results.joint_H6_xout6.nth`). Share of fitted quadrature nodes at the cap: 0.000.
