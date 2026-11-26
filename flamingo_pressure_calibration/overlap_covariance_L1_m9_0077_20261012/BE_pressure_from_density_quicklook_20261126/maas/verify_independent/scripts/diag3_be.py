"""Are the >1e-3 differences beyond B32 (H6 models) the shells that straddle the f_nth = 0.95 cap? My definitions."""
import json
import numpy as np
SRC = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/scripts/verify_be.py'
txt = open(SRC).read().split('# ---------------------------------------------------------------- check 2')[0]
exec(txt)
J2 = json.load(open(OUT + 'be_fits2.json'))
blk = Z['block']; lo = Z['lo']; hi = Z['hi']; kind = Z['kind']
node = np.array([blocks[b]['node'] for b in blk]); zb = np.array([blocks[b]['z'] for b in blk])
reg = Z['region']
out = {}
for name, J, ZZ in (('joint_H6', J1, Z), ('joint_H6_xout6', J2, Z2)):
    r = J['results'][name]; p = np.asarray(r['params']); xout = r['info']['xout']
    idx = np.where((reg == 1))[0]
    Pm, _ = model_rows(name, p, 'H6', xout, rows=idx)
    st = ZZ['P__' + name][idx]
    d = np.abs(Pm / st - 1)
    a, b, n = p[9:12]
    xcap = (0.95 / (a * (1 + zb[idx]) ** b)) ** (1 / n)
    kink = (lo[idx] < xcap) & (hi[idx] > xcap)
    # my value WITHOUT splitting at the kink, 12-point GL in x over the shell (replica of the documented rule)
    xg, wg = leggauss(12)
    nos = []
    cache = {}
    for j, i in enumerate(idx[kink]):
        M = 10 ** node[i]; z = zb[i]
        if blk[i] not in cache:
            cache[blk[i]] = Ptot(p[:9], M, z, xout)
        xx = 0.5 * (hi[i] - lo[i]) * xg + 0.5 * (hi[i] + lo[i])
        f = H_P(xx, p[:9], p[9:], M, z, xout, cache[blk[i]], 'H6')
        v = 0.5 * (hi[i] - lo[i]) * np.sum(wg * xx ** 2 * f) / ((hi[i] ** 3 - lo[i] ** 3) / 3)
        nos.append(abs(v / ZZ['P__' + name][i] - 1))
    out[name] = dict(n_region1=int(len(idx)), n_kink=int(kink.sum()),
                     kink_maxrel=float(d[kink].max()) if kink.any() else None,
                     nonkink_maxrel=float(d[~kink].max()),
                     nonkink_maxrel_x_lt_3p5=float(d[~kink & (hi[idx] < 3.5)].max()),
                     gl12_unsplit_vs_study_on_kink_rows_maxrel=float(max(nos)) if nos else None)
print(json.dumps(out, indent=1))
json.dump(out, open(VER + 'out/diag3_results.json', 'w'), indent=1)
