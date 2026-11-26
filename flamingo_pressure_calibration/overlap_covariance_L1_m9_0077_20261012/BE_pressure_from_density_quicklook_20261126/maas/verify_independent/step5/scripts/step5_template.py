"""Step 5 check 1: independent electron (and pressure) 2-halo template with halo exclusion, from the be_twohalo_e.py docstring.

t(r | M, z) = b(M) int dlnM' n(M') b(M') S(M', M) int_{|s| >= s_ex} d^3s xi_lin(s) F'(|r - s| | M'),
  s_ex = f_ex (R200c(M) + R200c(M')), f_ex = 0.75; F' truncated at 4 R200c(M'); neighbours 1e10-1e16 Msun;
  mean term: b b' xi -> 1 + b b' xi.
Electrons: F' = Battaglia+16 AGN gas profile in E units, E' = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a]^(-(b - 0.2)/a),
  n0 = 4e3 m^0.29 (1+z)^-0.66, a = 0.88 m^-0.03 (1+z)^0.19, b = 3.83 m^0.04 (1+z)^-0.025, m = M'/1e14, x = |r-s|/R200c(M');
  S = 1.  Pressure (self-check of my exclusion integral against the BD template): F' = B12 P_e/P200c (bd_twohalo.b12_pe),
  S = (M'/M)^(2/3).
Exclusion integral (mine): with u = |r - s| and the angular integral done analytically,
  int_{|s|>=s_ex} d^3s w(s) F'(|r-s|) = (2 pi / r) int_{s_lo}^{s_hi} ds s w(s) int_{|r-s|}^{min(r+s, 4R')} du u F'(u),
  s_lo = max(s_ex, r - 4R'), s_hi = r + 4R'.
Imported read-only from bd_twohalo: cosmology, Sheth-Mo-Tormen bias, Sheth-Tormen dn/dlnM, comoving R200c, xi_lin (and b12_pe
for the pressure self-check). Row value: shell volume mean over [lo, hi] (x = r/R200c(M)) or 4 pi int_0^hi x^2 t dx.
"""
import sys, json, time
import numpy as np
from numpy.polynomial.legendre import leggauss

t0 = time.time()
BDC = '/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118/code'
sys.path.insert(0, BDC)
import bd_twohalo
OUT = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/out/'
S5 = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/step5/'
BDJ = '/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118/results/flamingo_residuals.json'
tx = bd_twohalo.TwoHaloExcl('flamingo', (10.0, 16.0))
Z = np.load(OUT + 'be_fits_rows.npz')
blocks = json.load(open(BDJ))['blocks']
node = np.array([blocks[b]['node'] for b in Z['block']]); zb = np.array([blocks[b]['z'] for b in Z['block']])
lo = Z['lo']; hi = Z['hi']; kind = Z['kind']; reg = Z['region']
tE0 = np.load(OUT + 'etemplate_ex0p75.npy'); tE1 = np.load(OUT + 'etemplate_ex0p75+mean.npy')
tP = np.load(OUT + 'be_fit4r_e_rows.npz')['template_P']

FEX = 0.75; XT = 4.0
xm, wm = leggauss(96)
lnM = 0.5 * (np.log(1e16) - np.log(1e10)) * xm + 0.5 * (np.log(1e16) + np.log(1e10))
wlnM = 0.5 * (np.log(1e16) - np.log(1e10)) * wm
Mp = np.exp(lnM)
gs, gws = leggauss(48)
gu, gwu = leggauss(16)

def b16(x, M, z):
    m = M / 1e14
    n0 = 4e3 * m ** 0.29 * (1 + z) ** -0.66
    a = 0.88 * m ** -0.03 * (1 + z) ** 0.19
    b = 3.83 * m ** 0.04 * (1 + z) ** -0.025
    y = x / 0.5
    return n0 * np.exp(-0.2 * np.log(y) - (b - 0.2) / a * np.log1p(y ** a))

cache = {}
def mass_tables(z):
    if z not in cache:
        cache[z] = dict(R=tx.r200c(Mp, z), b=tx.bias(Mp, z), n=tx.dndlnm(Mp, z))
    return cache[z]

def t_both(r, M, z, field):
    """Return (without mean, with mean) in one pass."""
    tb = mass_tables(z)
    R = float(tx.r200c(np.array([M]), z)[0]); bM = float(tx.bias(np.array([M]), z)[0])
    corr = 0.0; mpart = 0.0
    for j in range(len(Mp)):
        Rp = tb['R'][j]
        s_lo = max(FEX * (R + Rp), r - XT * Rp, 0.0); s_hi = r + XT * Rp
        if s_lo >= s_hi:
            continue
        pts = [s_lo] + sorted(p for p in (r, XT * Rp - r) if s_lo < p < s_hi) + [s_hi]
        s = np.concatenate([0.5 * (pts[k + 1] - pts[k]) * gs + 0.5 * (pts[k + 1] + pts[k]) for k in range(len(pts) - 1)])
        ws = np.concatenate([0.5 * (pts[k + 1] - pts[k]) * gws for k in range(len(pts) - 1)])
        ua = np.abs(r - s); ub = np.minimum(r + s, XT * Rp)
        ok = ub > ua
        s, ws, ua, ub = s[ok], ws[ok], ua[ok], ub[ok]
        u = 0.5 * (ub - ua)[:, None] * gu + 0.5 * (ub + ua)[:, None]
        if field == 'E':
            Fp = b16(u / Rp, Mp[j], z); Sfac = 1.0
        else:
            Fp = tx.b12_pe(u.ravel() / Rp, Mp[j], z).reshape(u.shape); Sfac = (Mp[j] / M) ** (2.0 / 3.0)
        G = np.sum(0.5 * (ub - ua)[:, None] * gwu * u * Fp, 1)
        I_xi = 2 * np.pi / r * np.sum(ws * s * tx.xi(s, z) * G)
        I_1 = 2 * np.pi / r * np.sum(ws * s * G)
        corr += wlnM[j] * tb['n'][j] * tb['b'][j] * Sfac * I_xi
        mpart += wlnM[j] * tb['n'][j] * Sfac * I_1
    return bM * corr, bM * corr + mpart

xs8, ws8 = leggauss(8); xs24, ws24 = leggauss(24)
def row_template(i, field):
    M = 10 ** node[i]; z = zb[i]
    R = float(tx.r200c(np.array([M]), z)[0])
    if kind[i] == 'shell':
        x = 0.5 * (hi[i] - lo[i]) * xs8 + 0.5 * (hi[i] + lo[i]); w = 0.5 * (hi[i] - lo[i]) * ws8 * x ** 2
        norm = (hi[i] ** 3 - lo[i] ** 3) / 3
    else:
        x = 0.5 * hi[i] * (xs24 + 1); w = 0.5 * hi[i] * ws24 * x ** 2 * 4 * np.pi
        norm = 1.0
    vals = np.array([t_both(xx * R, M, z, field) for xx in x])
    return np.sum(w[:, None] * vals, 0) / norm

rng = np.random.default_rng(55)
pick = []
for rg, n in ((0, 8), (1, 7), (2, 5)):
    cand = np.where(reg == rg)[0]
    pick += list(rng.choice(cand, n, replace=False))
integ = np.where(kind != 'shell')[0]
pick += list(rng.choice(integ, 2, replace=False))      # make sure core/group integrals are covered
res = dict(rows_E=[], rows_P=[])
for i in pick:
    v0, v1 = row_template(int(i), 'E')
    d = dict(row=int(i), region=int(reg[i]), kind=str(kind[i]), lo=float(lo[i]), hi=float(hi[i]), node=float(node[i]), z=float(zb[i]),
             mine_ex=float(v0), study_ex=float(tE0[i]), rel_ex=float(v0 / tE0[i] - 1),
             mine_mean=float(v1), study_mean=float(tE1[i]), rel_mean=float(v1 / tE1[i] - 1))
    res['rows_E'].append(d)
    print('E', json.dumps(d), flush=True)
for i in list(rng.choice(np.where(reg == 0)[0], 2, replace=False)) + list(rng.choice(np.where(reg == 1)[0], 2, replace=False)) + \
         list(rng.choice(np.where(reg == 2)[0], 2, replace=False)):
    try:
        v0, _ = row_template(int(i), 'P')
        d = dict(row=int(i), region=int(reg[i]), kind=str(kind[i]), mine=float(v0), bd_template_P=float(tP[i]), rel=float(v0 / tP[i] - 1))
    except Exception as ex:
        d = dict(row=int(i), error=repr(ex), rel=float('nan'))
    res['rows_P'].append(d)
    print('P', json.dumps(d), flush=True)
for key in ('rel_ex', 'rel_mean'):
    a = np.abs([d[key] for d in res['rows_E']])
    res['E_max_abs_' + key] = float(a.max()); res['E_median_abs_' + key] = float(np.median(a))
res['P_max_abs_rel'] = float(np.max(np.abs([d['rel'] for d in res['rows_P']])))
chk = json.load(open(OUT + 'etemplate_check.json'))
res['etemplate_check_P_field'] = dict(max_rel_diff_vs_BD_pressure_template=chk.get('max_rel_diff_vs_BD_pressure_template'),
                                      n_rows=len(chk.get('rows', [])))
res['mean_over_ex_ratio_range_all_rows'] = [float((tE1 / tE0).min()), float((tE1 / tE0).max())]
res['seconds'] = time.time() - t0
json.dump(res, open(S5 + 'out/step5_template.json', 'w'), indent=1)
print('summary', json.dumps({k: v for k, v in res.items() if not k.startswith('rows')}), flush=True)
