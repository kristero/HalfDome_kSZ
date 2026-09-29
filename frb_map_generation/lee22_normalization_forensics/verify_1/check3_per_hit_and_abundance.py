"""verify_1 check 3: remove the halo-abundance normalization from the TNG/literal comparison.
C_hit(z) = TNG mean DM per hit ray / model mean DM per hit ray (model hit = 1-exp(-N)); independent of the
Tinker08 amplitude. Also per-slice hit-fraction ratio TNG/Tinker08 and its z trend.
Reads check2_slices.npz (own outputs only).
"""
import numpy as np
OUT = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_1/'
d = np.load(OUT + 'check2_slices.npz')
zlo, zhi = d['zlo'], d['zhi']; zm = 0.5 * (zlo + zhi)
lines = []
P = lambda s: (print(s), lines.append(s))
for units in ['phys', 'hinv']:
    for win in ['mass13to14', 'mass12to14']:
        N = d[f'model_{units}_{win}_N']; D = d[f'model_{units}_{win}_D']; Dfb = d[f'model_{units}_{win}_Dfb']
        hfm = 1 - np.exp(-N)
        per_hit_model = D / hfm; per_hit_fb = Dfb / hfm
        t_hit = d[f'tng_{win}_mean_hit']; t_hf = d[f'tng_{win}_hf']; t_med = d[f'tng_{win}_med_hit']
        Ch = t_hit / per_hit_model; Cfb = t_hit / per_hit_fb; Rhf = t_hf / hfm
        P(f'\n## {win} ({units})')
        for zr in [(0, 1), (0, 2)]:
            s = (zlo >= zr[0] - 1e-9) & (zhi <= zr[1] + 1e-9)
            k, lA = np.polyfit(np.log(1 + zm[s]), np.log(Ch[s]), 1)
            kf, lAf = np.polyfit(np.log(1 + zm[s]), np.log(Cfb[s]), 1)
            kh, lAh = np.polyfit(np.log(1 + zm[s]), np.log(Rhf[s]), 1)
            kt, lAt = np.polyfit(np.log(1 + zm[s]), np.log(t_hit[s]), 1)
            km, lAm = np.polyfit(np.log(1 + zm[s]), np.log(per_hit_model[s]), 1)
            ktm, _ = np.polyfit(np.log(1 + zm[s]), np.log(t_med[s]), 1)
            P(f' z{zr}: per-hit C = {np.exp(lA):.3f}(1+z)^{k:.3f}; per-hit TNG/fb-control ~ (1+z)^{kf:.3f}; '
              f'hit-frac TNG/T08 = {np.exp(lAh):.3f}(1+z)^{kh:.3f} (range {Rhf[s].min():.3f}-{Rhf[s].max():.3f})')
            P(f'        TNG mean/hit ~ (1+z)^{kt:.3f}, TNG median/hit ~ (1+z)^{ktm:.3f}; literal mean/hit ~ (1+z)^{km:.3f}')
        P(f'  literal/TNG per hit at z 0-0.1: {per_hit_model[0] / t_hit[0]:.2f}; at 0.9-1.0: {per_hit_model[9] / t_hit[9]:.2f}; '
          f'literal per-hit drop 0.05->0.95: {per_hit_model[0] / per_hit_model[9]:.2f}x, TNG: {t_hit[0] / t_hit[9]:.2f}x')
open(OUT + 'check3_per_hit_and_abundance.txt', 'w').write('\n'.join(lines) + '\n')
