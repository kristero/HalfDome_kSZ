## Table 8. Step 5: pressure and electrons both fitted to 4 R200c, each with its own 2-halo term (parity weights)

Source: `out/be_fit4r_e.json: results[name]`. Electron template used: ex0p75+mean (halo-model, Battaglia+16 neighbours, exclusion 0.75, with the uncorrelated mean term). Weights: pressure 2701 + 1853.1, electrons 2701 + 1853.1 (inside + beyond B32).

- Electron law + 2-halo alone, template ex0p75: L_4R,E = 53.23, electron loss inside B32 (step-2 weights) 35.87 (15.15 when fitted inside B32 only); RMS beyond B32 0.084 / 0.105 / 0.087 / 0.155; A_E(z = 0 / 1 / 3) = 0.82 / 1.63 / 4.76.
- Electron law + 2-halo alone, template ex0p75+mean: L_4R,E = 52.11, electron loss inside B32 (step-2 weights) 35.16 (15.15 when fitted inside B32 only); RMS beyond B32 0.084 / 0.102 / 0.084 / 0.107; A_E(z = 0 / 1 / 3) = 0.70 / 1.35 / 4.00.

| model | parameters (law + amplitudes) | L_4R,P | L_4R,E | total | inside B32: L_P / L_E | pressure RMS inside: 10^12.7-13.1 / 13.1-13.5 / 13.5-14.3 / >= 14.3 | electron RMS inside, same groups | pressure RMS beyond B32: 1.5-2 / 2-3 / 3-4 / 4-6 | electron RMS beyond B32, same bands |
|---|---|---|---|---|---|---|---|---|---|
| separate laws: electron law + 2-halo; pressure law (constant alpha) + 2-halo | 19 + 6 = 25 | 35.84 | 52.11 | 87.95 | 16.76 / 35.16 | 0.453 / 0.217 / 0.102 / 0.126 | 0.278 / 0.219 / 0.173 / 0.106 | 0.078 / 0.085 / 0.103 / 0.152 | 0.084 / 0.102 / 0.084 / 0.107 |
| separate laws: electron law + 2-halo; pressure law with mass break + 2-halo | 25 + 6 = 31 | 19.61 | 52.11 | 71.72 | 4.26 / 35.16 | 0.171 / 0.131 / 0.054 / 0.046 | 0.278 / 0.219 / 0.173 / 0.106 | 0.069 / 0.088 / 0.106 / 0.175 | 0.084 / 0.102 / 0.084 / 0.107 |
| electron law + 2-halo held fixed; temperature law T7x + pressure 2-halo | 16 + 6 = 22 | 62.12 | 52.11 | 114.23 | 19.35 / 35.16 | 0.185 / 0.166 / 0.112 / 0.113 | 0.278 / 0.219 / 0.173 / 0.106 | 0.116 / 0.147 / 0.141 / 0.136 | 0.084 / 0.102 / 0.084 / 0.107 |
| electron law x temperature law T7x, both 2-halo terms, joint | 16 + 6 = 22 | 38.33 | 60.62 | 98.95 | 12.03 / 44.27 | 0.276 / 0.173 / 0.096 / 0.085 | 0.345 / 0.222 / 0.167 / 0.092 | 0.080 / 0.119 / 0.124 / 0.155 | 0.092 / 0.093 / 0.073 / 0.134 |
| electron law x temperature law T9x (outer radius with M and z), joint | 18 + 6 = 24 | 38.37 | 59.49 | 97.86 | 11.71 / 43.05 | 0.259 / 0.156 / 0.092 / 0.098 | 0.350 / 0.224 / 0.167 / 0.085 | 0.078 / 0.121 / 0.126 / 0.168 | 0.094 / 0.093 / 0.072 / 0.136 |
| hydrostatic + core term (x_out 6), joint | 15 + 6 = 21 | 47.85 | 55.72 | 103.57 | 15.71 / 41.14 | 0.216 / 0.157 / 0.094 / 0.077 | 0.315 / 0.229 / 0.161 / 0.123 | 0.114 / 0.131 / 0.128 / 0.181 | 0.073 / 0.094 / 0.080 / 0.109 |
