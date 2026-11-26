"""Extra diagnostic: model values beyond B32 (regions 1 and 2, to 6 R200c) and row-file column labels. My definitions."""
import json
import numpy as np
SRC = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/scripts/verify_be.py'
txt = open(SRC).read().split('# ---------------------------------------------------------------- check 2')[0]
exec(txt)
J2 = json.load(open(OUT + 'be_fits2.json'))
out = {}
# row-file columns P_a2ii / E_a2ii vs my a2ii and all-gas stack means (region 0)
S = np.load(OUT + 'be_sums.npz')
def vm(e, F, a, b, knd):
    if knd == 'shell':
        ov = (np.clip(e[1:], a, b) ** 3 - np.clip(e[:-1], a, b) ** 3) / 3
        return np.sum(F * ov) / ((b ** 3 - a ** 3) / 3)
    ov = (np.minimum(e[1:], b) ** 3 - np.minimum(e[:-1], b) ** 3) / 3
    return 4 * np.pi * np.sum(F * ov)
allrows = np.arange(len(Z['region']))
cols = {}
for q in ('a2ii', 'all', 'hb'):
    for F in ('P', 'E'):
        v = np.zeros(len(allrows))
        for i in allrows:
            b = blocks[Z['block'][i]]; key = '%d__%s__' % (b['snapshot'], b['label']); e = S['%d__edges' % b['snapshot']]
            v[i] = vm(e, S[key + 'S_%s_%s' % (F, q)].sum(0) / S[key + 'W'].sum(), Z['lo'][i], Z['hi'][i], Z['kind'][i])
        cols[F + '_' + q] = v
for F in ('P', 'E'):
    out['col_%s_a2ii_vs_my_a2ii_maxrel' % F] = float(np.max(np.abs(Z['%s_a2ii' % F] / cols[F + '_a2ii'] - 1)))
    out['col_%s_a2ii_vs_my_all_maxrel' % F] = float(np.max(np.abs(Z['%s_a2ii' % F] / cols[F + '_all'] - 1)))
    out['col_%s_hb_vs_my_hb_maxrel' % F] = float(np.max(np.abs(Z['%s_hb' % F] / cols[F + '_hb'] - 1)))
    out['d_%s_vs_my_all_allregions_maxrel' % F] = float(np.max(np.abs(Z['d_%s' % F] / cols[F + '_all'] - 1)))
# switch the globals used by model_rows to ALL rows, then evaluate regions 1 and 2
blk = Z['block']; lo = Z['lo']; hi = Z['hi']; kind = Z['kind']
node = np.array([blocks[b]['node'] for b in blk]); zb = np.array([blocks[b]['z'] for b in blk])
reg = Z['region']
for name, J, ZZ in (('plaw_held', J1, Z), ('joint_T7', J1, Z), ('joint_H6', J1, Z), ('joint_H6_xout6', J2, Z2)):
    r = J['results'][name]; kindm = r['kind']; xout = r['info'].get('xout', 4.0) if 'xout' in r['info'] else 4.0
    Pm, Em = model_rows(name, r['params'], kindm, xout, rows=np.where(reg > 0)[0])
    st = ZZ['P__' + name][reg > 0]
    o = {}
    xm = 0.5 * (lo + hi)[reg > 0]
    for x1, x2 in ((1.0, 2.0), (2.0, 3.0), (3.0, 3.5), (3.5, 4.01), (4.01, 6.1)):
        s = (xm >= x1) & (xm < x2)
        if not s.any():
            continue
        both0 = (Pm[s] == 0) & (st[s] == 0)
        nz = ~both0
        d = np.abs(Pm[s][nz] / st[s][nz] - 1) if nz.any() else np.array([0.0])
        o['%g-%g' % (x1, x2)] = dict(n=int(s.sum()), n_both_zero=int(both0.sum()),
                                     maxrel=float(np.nanmax(d)), medrel=float(np.nanmedian(d)))
    out[name + '_beyond_B32'] = o
print(json.dumps(out, indent=1))
json.dump(out, open(VER + 'out/diag2_results.json', 'w'), indent=1)
