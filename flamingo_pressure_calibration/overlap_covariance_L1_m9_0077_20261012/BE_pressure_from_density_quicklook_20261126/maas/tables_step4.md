## Table 7. Step 4: law + 2-halo amplitude fitted together on all shells to 4 R200c

### record weights beyond B32 (region 1 = 45.0 of 2746.0 weight, 1.6 %)  (`out/be_fit4r.json: results[name]`)

| model | parameters: electron + pressure + amplitude | L_4R (beyond-B32 part) | L_P inside B32 | L_E | L_out6 (step-3 outer weights) | RMS inside: 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >= 14.3 | RMS beyond B32: 1.5-2 / 2-3 / 3-4 / 4-6 R200c (4-6 not fitted) | A_2h at z = 0 / 1 / 3 | at bounds |
|---|---|---|---|---|---|---|---|---|---|
| pressure law, constant alpha | 9 + 10 + 3 = 22 | 15.21 (0.75) | 14.46 | 15.15 | 1721 | 0.582 / 0.269 / 0.072 / 0.095 | 0.084 / 0.099 / 0.118 / 0.135 | 0.49 / 1.81 / 3.02 | - |
| pressure law, alpha(z) of record | 9 + 9 + 3 = 21 | 13.21 (0.72) | 12.48 | 15.15 | 3431 | 0.598 / 0.278 / 0.066 / 0.070 | 0.130 / 0.135 / 0.180 / 0.240 | 0.58 / 1.81 / 6.23 | - |
| pressure law with smooth mass break | 9 + 16 + 3 = 28 | 2.71 (0.56) | 2.14 | 15.15 | 2954 | 0.146 / 0.093 / 0.033 / 0.032 | 0.102 / 0.143 / 0.187 / 0.220 | 0.58 / 1.78 / 5.59 | - |
| electron law x temperature law T7x, joint | 9 + 7 + 3 = 19 | 9.09 (1.67) | 7.43 | 16.26 | 8671 | 0.235 / 0.084 / 0.070 / 0.064 | 0.153 / 0.175 / 0.348 / 0.403 | 1.11 / 2.06 / 3.00 | - |
| electron law x temperature law T7, joint | 9 + 7 + 3 = 19 | 9.69 (1.91) | 7.79 | 16.08 | 6192 | 0.238 / 0.081 / 0.072 / 0.069 | 0.174 / 0.202 / 0.276 / 0.272 | 0.90 / 1.62 / 1.90 | c_out |
| electron law x temperature law T7x, electron law fixed | 9 + 7 + 3 = 19 | 11.68 (2.20) | 9.48 | 15.15 | 9487 | 0.221 / 0.093 / 0.079 / 0.072 | 0.172 / 0.200 / 0.363 / 0.400 | 1.14 / 1.87 / 2.23 | - |
| hydrostatic + core term, joint | 9 + 6 + 3 = 18 | 10.22 (2.74) | 7.48 | 17.12 | 18010 | 0.243 / 0.087 / 0.069 / 0.074 | 0.297 / 0.222 / 0.510 / 0.618 | 1.05 / 3.04 / 7.22 | - |

### parity weights beyond B32 (region 1 = 1853.1 of 4554.1 weight, 41 %)  (`out/be_fit4r_parity.json: results[name]`)

| model | parameters: electron + pressure + amplitude | L_4R (beyond-B32 part) | L_P inside B32 | L_E | L_out6 (step-3 outer weights) | RMS inside: 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >= 14.3 | RMS beyond B32: 1.5-2 / 2-3 / 3-4 / 4-6 R200c (4-6 not fitted) | A_2h at z = 0 / 1 / 3 | at bounds |
|---|---|---|---|---|---|---|---|---|---|
| pressure law, constant alpha | 9 + 10 + 3 = 22 | 35.84 (19.08) | 16.76 | 15.15 | 1448 | 0.453 / 0.217 / 0.102 / 0.126 | 0.078 / 0.085 / 0.103 / 0.152 | 0.48 / 1.79 / 3.45 | - |
| pressure law, alpha(z) of record | 9 + 9 + 3 = 21 | 37.12 (22.60) | 14.52 | 15.15 | 1789 | 0.551 / 0.252 / 0.074 / 0.068 | 0.083 / 0.107 / 0.114 / 0.176 | 0.52 / 1.64 / 4.61 | - |
| pressure law with smooth mass break | 9 + 16 + 3 = 28 | 19.61 (15.35) | 4.26 | 15.15 | 1431 | 0.171 / 0.131 / 0.054 / 0.046 | 0.069 / 0.088 / 0.106 / 0.175 | 0.49 / 1.57 / 4.26 | - |
| electron law x temperature law T7x, joint | 9 + 7 + 3 = 19 | 39.86 (32.00) | 7.86 | 27.68 | 1948 | 0.274 / 0.149 / 0.068 / 0.078 | 0.103 / 0.129 / 0.127 / 0.133 | 0.56 / 1.79 / 3.35 | - |
| electron law x temperature law T7, joint | 9 + 7 + 3 = 19 | 40.12 (32.18) | 7.93 | 27.95 | 1946 | 0.280 / 0.153 / 0.067 / 0.079 | 0.104 / 0.130 / 0.127 / 0.131 | 0.55 / 1.77 / 3.25 | c_out |
| electron law x temperature law T7x, electron law fixed | 9 + 7 + 3 = 19 | 83.50 (65.24) | 18.26 | 15.15 | 3647 | 0.160 / 0.145 / 0.089 / 0.103 | 0.165 / 0.162 / 0.156 / 0.155 | 0.69 / 1.82 / 2.86 | - |
| hydrostatic + core term, joint | 9 + 6 + 3 = 18 | 51.08 (38.14) | 12.95 | 29.63 | 2703 | 0.197 / 0.140 / 0.079 / 0.082 | 0.132 / 0.139 / 0.138 / 0.195 | 0.68 / 1.94 / 4.14 | - |

### Parity weights: more flexible temperature or electron laws (`out/be_fit4r_parity_flex.json`, `out/be_fit4r_parity_eflex.json`)

| model | law parameters (electron + temperature) | + amplitude | L_4R | L_P inside B32 | L_E | L_4R + L_E | RMS beyond B32: 1.5-2 / 2-3 / 3-4 / 4-6 R200c | at bounds |
|---|---|---|---|---|---|---|---|---|
| pressure law, constant alpha (+ 9-parameter electron law, fitted separately) | 9 + 10 = 19 | 3 | 35.84 | 16.76 | 15.15 | 50.99 | 0.078 / 0.085 / 0.103 / 0.152 | - |
| pressure law with mass break (+ electron law) | 9 + 16 = 25 | 3 | 19.61 | 4.26 | 15.15 | 34.76 | 0.069 / 0.088 / 0.106 / 0.175 | - |
| T7x, joint | 9 + 7 = 16 | 3 | 39.86 | 7.86 | 27.68 | 67.55 | 0.103 / 0.129 / 0.127 / 0.133 | - |
| T8xm (outer radius depends on mass), joint | 9 + 8 = 17 | 3 | 39.14 | 7.56 | 27.96 | 67.10 | 0.100 / 0.126 / 0.128 / 0.132 | - |
| T8xz (outer radius depends on z), joint | 9 + 8 = 17 | 3 | 40.90 | 7.59 | 24.87 | 65.77 | 0.105 / 0.133 / 0.129 / 0.152 | - |
| T9x (both), joint | 9 + 9 = 18 | 3 | 39.68 | 7.03 | 24.56 | 64.23 | 0.098 / 0.130 / 0.131 / 0.157 | - |
| T8xm, electron law fixed | 9 + 8 = 17 | 3 | 83.28 | 18.31 | 15.15 | 98.44 | 0.168 / 0.163 / 0.154 / 0.154 | - |
| T9x, electron law fixed | 9 + 9 = 18 | 3 | 67.62 | 9.92 | 15.15 | 82.77 | 0.144 / 0.167 / 0.150 / 0.169 | - |
| T7x + electron law with free x_c, joint | 10 + 7 = 17 | 3 | 40.26 | 9.81 | 25.09 | 65.35 | 0.108 / 0.128 / 0.123 / 0.129 | - |
| T8xm + electron law with free x_c, joint | 10 + 8 = 18 | 3 | 39.50 | 9.66 | 25.05 | 64.55 | 0.103 / 0.124 / 0.124 / 0.127 | - |
| T7x + electron law with free inner slope, joint | 10 + 7 = 17 | 3 | 41.00 | 9.31 | 22.67 | 63.67 | 0.104 / 0.126 / 0.127 / 0.133 | gamma_e |
| T8xm + electron law with free inner slope, joint | 10 + 8 = 18 | 3 | 40.62 | 9.11 | 22.82 | 63.44 | 0.101 / 0.124 / 0.128 / 0.132 | gamma_e |

Electron law alone with the extra constant (x_c): L_E = 14.30 (9 parameters: 15.15); ln_xc_e = -1.030.

Electron law alone with the extra constant (inner slope): L_E = 13.49 (9 parameters: 15.15); gamma_e = 1.000.
