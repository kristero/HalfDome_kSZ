"""Diagnostics for the small model-value differences seen in verify_be.py (my definitions, read-only inputs)."""
import json
import numpy as np
SRC = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/scripts/verify_be.py'
txt = open(SRC).read().split('# ---------------------------------------------------------------- check 2')[0]
exec(txt)
J2 = json.load(open(OUT + 'be_fits2.json'))
out = {}
# 1. record law: constant ratio?
pp = np.asarray(J1['results']['plaw_held']['params'])
Pm, _ = model_rows('plaw_held', pp, 'plaw')
ratio = Pm / Z['P__plaw_held'][r0]
out['plaw_ratio_mine_over_study'] = dict(mean=float(ratio.mean()), std=float(ratio.std()),
                                         shells_mean=float(ratio[kind == 'shell'].mean()),
                                         shells_std=float(ratio[kind == 'shell'].std()),
                                         PEexact_over_0p5176=float(PE / 0.5176))
# loss with the rounded prefactor
LP_rounded = float(np.sum(wP * np.log(Pm * 0.5176 / PE / dP) ** 2))
out['plaw_LP_with_0p5176_prefactor'] = LP_rounded
# 2. H6: kink rows vs the rest
for name, J, ZZ in (('joint_H6', J1, Z), ('sep_H6', J1, Z), ('joint_H6_xout6', J2, Z2)):
    r = J['results'][name]; p = np.asarray(r['params']); xout = r['info']['xout']
    Pm, Em = model_rows(name, p, 'H6', xout)
    st = ZZ['P__' + name][r0]
    d = Pm / st - 1
    a, b, n = p[9:12]
    xcap = (0.95 / (a * (1 + zb) ** b)) ** (1 / n)
    kinkin = (kind == 'shell') & (lo < xcap) & (hi > xcap)
    shell = kind == 'shell'
    o = dict(n_kink_rows=int(kinkin.sum()),
             kink_rows_maxabs=float(np.max(np.abs(d[kinkin]))) if kinkin.any() else None,
             nonkink_shell_maxabs=float(np.max(np.abs(d[shell & ~kinkin]))),
             nonkink_shell_median=float(np.median(np.abs(d[shell & ~kinkin]))),
             worst_nonkink=None)
    s = np.where(shell & ~kinkin)[0]
    w = s[np.argmax(np.abs(d[s]))]
    o['worst_nonkink'] = dict(lo=float(lo[w]), hi=float(hi[w]), z=float(zb[w]), node=float(node[w]), d=float(d[w]))
    # trend with x: bins of shell mid radius
    xm = 0.5 * (lo + hi)
    tr = {}
    for x1, x2 in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.6)):
        ss = shell & ~kinkin & (xm >= x1) & (xm < x2)
        if ss.any():
            tr['%g-%g' % (x1, x2)] = dict(n=int(ss.sum()), median=float(np.median(d[ss])), maxabs=float(np.max(np.abs(d[ss]))))
    o['by_x_nonkink_signed'] = tr
    # how much of L_P difference is from kink rows
    LPm = np.sum(wP * np.log(Pm / dP) ** 2); LPs = np.sum(wP * np.log(st / dP) ** 2)
    o['LP_mine'] = float(LPm); o['LP_from_study_rows'] = float(LPs); o['LP_study_json'] = r['L_P']
    # capped fraction at the B32 cut range
    o['xcap_range'] = [float(xcap.min()), float(xcap.max())]
    out[name] = o
# 3. convergence of my own Ptot (panel count and lower limit) for one block at z=2
p = np.asarray(J1['results']['joint_H6']['params'])
i = int(np.where(zb == zb.max())[0][0]); M = 10 ** node[i]; z = zb[i]
xs = np.array([0.01, 0.1, 0.5, 1.0, 1.5, 3.0])
pa = Ptot(p[:9], M, z, 4.0)(xs); pb = Ptot(p[:9], M, z, 4.0, npan=16000, umin=np.log(1e-16))(xs)
out['Ptot_convergence_maxrel'] = float(np.max(np.abs(pa / pb - 1)))
print(json.dumps(out, indent=1))
json.dump(out, open(VER + 'out/diag_results.json', 'w'), indent=1)
# 4. electron weights from the documented formula (rows with prio1 = 0 have zero weight)
S = np.load(OUT + 'be_sums.npz')
fE = S['resid_E_frac_scatter']; prio = BN['B32__prio1']
sE = np.maximum(fE, 0.05)
wE_re = 2701.0 * (prio / sE ** 2) / np.sum(prio / sE ** 2)
pos = wE > 0
out['wE'] = dict(n_zero_weight=int((~pos).sum()), n_prio_zero=int((prio == 0).sum()),
                 zero_rows_agree=bool(np.all(wE_re[~pos] == 0)),
                 maxrel_positive=float(np.max(np.abs(wE_re[pos] / wE[pos] - 1))),
                 prio_values=np.unique(prio).tolist())
# 5. BD record target: a2ii or all-gas stack? and the BD loss with the record prefactor 0.5176
lab = {}
def vm(e, F, a, b, knd):
    if knd == 'shell':
        ov = (np.clip(e[1:], a, b) ** 3 - np.clip(e[:-1], a, b) ** 3) / 3
        return np.sum(F * ov) / ((b ** 3 - a ** 3) / 3)
    ov = (np.minimum(e[1:], b) ** 3 - np.minimum(e[:-1], b) ** 3) / 3
    return 4 * np.pi * np.sum(F * ov)
dPa2 = np.zeros(len(r0)); dPall = np.zeros(len(r0))
for i in range(len(r0)):
    b = blocks[blk[i]]; key = '%d__%s__' % (b['snapshot'], b['label']); e = S['%d__edges' % b['snapshot']]
    dPa2[i] = vm(e, S[key + 'S_P_a2ii'].sum(0) / S[key + 'W'].sum(), lo[i], hi[i], kind[i])
    dPall[i] = vm(e, S[key + 'S_P_all'].sum(0) / S[key + 'W'].sum(), lo[i], hi[i], kind[i])
tgt = BN['B32__target_flamingo_Y']
diff = np.abs(dPall / dPa2 - 1)
out['BD_target'] = dict(target_vs_a2ii_maxrel=float(np.max(np.abs(tgt / dPa2 - 1))),
                        target_vs_all_maxrel=float(np.max(np.abs(tgt / dPall - 1))),
                        n_rows_all_ne_a2ii=int(np.sum(diff > 1e-12)),
                        rows_all_ne_a2ii_kinds=sorted(set(kind[diff > 1e-12].tolist())),
                        rows_all_ne_a2ii_min_hi=float(hi[diff > 1e-12].min()) if np.any(diff > 1e-12) else None)
pp_bd = np.asarray(BJ['lsq_fixed_alpha']['coef'])
Pbd, _ = model_rows('plaw', pp_bd, 'plaw')
Pbd_r = Pbd * 0.5176 / PE
out['BD_loss'] = dict(record=BJ['lsq_fixed_alpha']['loss'],
                      mine_BDtarget_prefactor0p5176=float(np.sum(BN['B32__weight'] * np.log(Pbd_r / tgt) ** 2)),
                      mine_BDtarget_exactPE=float(np.sum(BN['B32__weight'] * np.log(Pbd / tgt) ** 2)),
                      mine_BEtarget_prefactor0p5176=float(np.sum(wP * np.log(Pbd_r / dP) ** 2)))
pps = np.asarray(J1['results']['plaw_held']['params'])
Ps, _ = model_rows('plaw', pps, 'plaw')
out['plaw_held_LP_prefactor0p5176'] = float(np.sum(wP * np.log(Ps * 0.5176 / PE / dP) ** 2))
# 6. is the residual 1e-5 on integrals the 12-point Gauss-Legendre rule in x over [0, hi]? (my replica of the documented rule)
xg, wg = leggauss(12)
pe = np.asarray(J1['results']['electron_law']['params'])
st = Z['E__sep'][r0]
dd = []
for i in np.where(kind != 'shell')[0]:
    lm = np.log(10 ** node[i] / 1e14); lz = np.log1p(zb[i])
    xx = 0.5 * hi[i] * (xg + 1)
    v = 4 * np.pi * 0.5 * hi[i] * np.sum(wg * xx ** 2 * E_law(xx, pe, lm, lz))
    dd.append(abs(v / st[i] - 1))
out['gl12_integral_rule_vs_study_E_maxrel'] = float(max(dd))
print(json.dumps(out, indent=1))
json.dump(out, open(VER + 'out/diag_results.json', 'w'), indent=1)
# 7. tau of the stacks on storage cells (all z=0 blocks with node >= 13.9), C_T sum_k S_P_all / sum_k S_E_all
t7 = {}
for b in blocks:
    if b['snapshot'] != 77 or b['log_mass_low'] < 13.9:
        continue
    key = '77__%s__' % b['label']; e = S['77__edges']
    tau_c = C_T * S[key + 'S_P_all'].sum(0) / S[key + 'S_E_all'].sum(0)
    t7[b['label']] = {('x%g' % xr): float(tau_c[np.searchsorted(e, xr, side='right') - 1]) for xr in (0.1, 0.5, 1.0)}
out['tau_stack_cells_z0'] = t7
print(json.dumps(out['tau_stack_cells_z0'], indent=1))
json.dump(out, open(VER + 'out/diag_results.json', 'w'), indent=1)
