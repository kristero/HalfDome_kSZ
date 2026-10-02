"""Synthesis: one consistent multiplicative budget of HalfDome map / WebSky-halo (Stein+20 Fig. 6) at ell = 1000, 3000, 5000.
Reads only the investigators' saved outputs; writes outputs/lee22_ksz_debug_20261002/synthesis/budget.{json,txt}."""
import json, numpy as np
O = '/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_ksz_debug_20261002/'
idx = [2, 4, 5]                       # ell_rep = 300,500,1000,2000,3000,5000 -> 1000,3000,5000
ells = [1000, 3000, 5000]
wct = json.load(open(O + 'websky_catalogue_test/onehalo_summary.json'))
S = {k: np.array(v)[idx] for k, v in wct['spectra_at_ell_rep'].items()}
curve = np.array(wct['websky_digitized'])[idx]            # CSV digitization (0.197/0.509/0.605)
fc = np.load(O + 'websky_def/factor_chain.npz', allow_pickle=True)
fk = {k: fc[k][[1, 3, 4]] for k in fc.files}               # ell 300,1000,2000,3000,5000 -> 1000,3000,5000
curve_exact = fk['websky']
hdmap = fk['map']
# abundance investigator (halo_catalogue/onehalo_effects.txt), base = HD one-halo 0.3624/0.6456/0.6491
hd_cut_c7 = np.array([0.3620, 0.6183, 0.5898])             # c200m=7 M200c + WebSky cuts
vel_hd_over_ws_cut = np.array([1.168, 1.156, 1.148])       # WebSky cat (c7, cuts) painted with HD <v^2>(M,z)
# painter investigator (painting_geometry/analysis_tables.txt): HD sph4, cut / comp (M200m, 3 rho_m) combos
p_base = np.array([0.3624, 0.6456, 0.6491]); p_cut = np.array([0.3520, 0.5982, 0.5704])
p_comp = np.array([0.2543, 0.5472, 0.6119]); p_cutcomp = np.array([0.2509, 0.5304, 0.5712])

R_total = hdmap / curve
f = {}
f['map_over_onehalo'] = hdmap / S['halfdome_b16_x4']
f['websky_cut_on_HD'] = S['halfdome_b16_x4'] / S['halfdome_b16_x4_sel_cell']
f['catalogue_at_fixed_cut'] = S['halfdome_b16_x4_sel_cell'] / S['websky_c7m_b16_x4_sel_cell']
f['  velocity_Nbody_vs_2LPT'] = vel_hd_over_ws_cut
f['  M200c_definition_HDown_vs_c7'] = S['halfdome_b16_x4_sel_cell'] / hd_cut_c7
f['  abundance_residual'] = hd_cut_c7 / (S['websky_c7m_b16_x4_sel_cell'] * vel_hd_over_ws_cut)
f['compensation_after_cut'] = S['websky_c7m_b16_x4_sel_cell'] / S['websky_c7m_b16_x4_sel_comp_websky']
f['residual_vs_curve'] = S['websky_c7m_b16_x4_sel_comp_websky'] / curve
trunc = fk['+ truncation 4 -> 1.5 R200c'] / fk['+ compensation (uniform sphere, 3 rho_m,']
ne09 = fk['+ truncation 4 -> 1.5 R200c'] / fk['+ n_e = (1-Y/2) rho/m_p (drop XGPaint 0.']
f['  truncation_4_vs_1p5R200c'] = 1 / trunc
f['  XGPaint_0p9_vs_WebSky_(1-Y/2)'] = ne09
f['  leftover'] = f['residual_vs_curve'] / (f['  truncation_4_vs_1p5R200c'] * f['  XGPaint_0p9_vs_WebSky_(1-Y/2)'])
top = ['map_over_onehalo', 'websky_cut_on_HD', 'catalogue_at_fixed_cut', 'compensation_after_cut', 'residual_vs_curve']
prod = np.prod([f[k] for k in top], axis=0)
# Shapley (order-averaged) split of cut vs compensation on HalfDome, painter numbers
cut_sh = np.exp(0.5 * (np.log(p_base / p_cut) + np.log(p_comp / p_cutcomp)))
comp_sh = np.exp(0.5 * (np.log(p_base / p_comp) + np.log(p_cut / p_cutcomp)))
lines = ['ell                                   ' + ''.join('%10d' % l for l in ells),
         'HD map D_ell [uK^2]                   ' + ''.join('%10.4f' % x for x in hdmap),
         'WebSky halo, CSV digitized            ' + ''.join('%10.4f' % x for x in curve),
         'WebSky halo, exact PDF vector         ' + ''.join('%10.4f' % x for x in curve_exact),
         'R_total = map / CSV curve             ' + ''.join('%10.3f' % x for x in R_total),
         'product of top-level factors          ' + ''.join('%10.3f' % x for x in prod),
         '', 'factor (contribution to R)            ' + ''.join('%10d' % l for l in ells) + '   | share of ln R_total']
for k, v in f.items():
    lines.append('%-38s' % k + ''.join('%10.3f' % x for x in v) + '   | ' + ' '.join('%6.0f%%' % (100 * np.log(x) / np.log(r)) for x, r in zip(v, R_total)))
lines += ['', 'order-averaged (Shapley) on HalfDome, painter numbers:',
          '%-38s' % 'cut' + ''.join('%10.3f' % x for x in cut_sh),
          '%-38s' % 'compensation' + ''.join('%10.3f' % x for x in comp_sh),
          '', 'reference points:',
          '%-38s' % 'WebSky cat, our painting / curve' + ''.join('%10.3f' % x for x in S['websky_c7m_b16_x4'] / curve),
          '%-38s' % 'WebSky cat, cut+comp / curve' + ''.join('%10.3f' % x for x in S['websky_c7m_b16_x4_sel_comp_websky'] / curve),
          '%-38s' % 'WebSky cat, full WebSky recipe/exact' + ''.join('%10.3f' % x for x in fk['+ n_e = (1-Y/2) rho/m_p (drop XGPaint 0.'] / curve_exact),
          '%-38s' % 'HD cut+comp / WS cut+comp (same recipe)' + ''.join('%10.3f' % x for x in (p_cutcomp / S['websky_c7m_b16_x4_sel_comp_websky']))]
txt = '\n'.join(lines); print(txt)
open(O + 'synthesis/budget.txt', 'w').write(txt + '\n')
json.dump({'ell': ells, 'R_total': R_total.tolist(), 'factors': {k.strip(): v.tolist() for k, v in f.items()},
           'shapley_cut': cut_sh.tolist(), 'shapley_comp': comp_sh.tolist()}, open(O + 'synthesis/budget.json', 'w'), indent=1)
