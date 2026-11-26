"""Independent verification of BE step 4 (be_fit4r): 1-halo law + 2-halo term fitted in one loss to 4 R200c.
Written from the be_fit4r.py docstring and the coordinator's brief only (no study code read or copied).

Symbols: x = r/R200c; M = 10^node of the row's BD block [Msun]; z = block redshift; lm = ln(M/1e14); lz = ln(1+z).
  law quantity Q: ln Q = lnA_Q + am_Q lm + az_Q lz (+ (dm_Q + kz_Q lz) S for the mass break,
                  S = -0.3 ln(1 + exp(-(lm - lmb)/0.3)), Q in {P0, x_c, beta})
  pressure law P1 = 0.5176 P0 q^-0.3 (1 + q^alpha)^-beta, q = x/x_c; alpha = alpha0 (const) or 0.93935 + 0.29477 z (massbreak)
  electron law E = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a_e]^(-(beta_e - 0.2)/a_e); T7x: tau = T0' exp[-(x/x_e)^c_e] s/(1+s),
  T0' = exp(lnT0 + am_T lm + az_T lz), s = (x/x_cc)^2, ln x_cc = c0 + bm lm; P1 = E tau / C_T (C_T = 428.214)
  model m_i = P1_i + A_2h(z_i) t_i, ln A_2h = a + b lz + c lz^2, t = saved template column
  row value: volume mean over [lo, hi] (shells) or 4 pi int_0^hi x^2 f dx (core / group), 48-pt Gauss-Legendre in ln x
  L_4R = sum_{regions 0,1} w ln^2(m/d_P); region-1 part; L_P = sum_{region 0} wP ln^2(m/d_P); L_E = sum_{region 0} wE ln^2(E/d_E)
  weights: region 0 = BD B32__weight; region 1 'record' = BD x4__weight of shells with lo >= B32(block) - 1e-9 (file order);
           region 1 'parity' = (W0/N0) v_i / mean_region1(v), v = 1/(0.1^2 + e^2), e = P_err/d_P, W0 = 2701, N0 = 1494
  L_out6 = sum_{regions 1,2} v ln^2(m/d_P);  outer RMS: shells, region >= 1, nhalo >= 100, hi in (lo_b, hi_b]
"""
import json, time, sys
import numpy as np
from scipy.optimize import least_squares

t0 = time.time()
SRC = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/scripts/verify_be.py'
exec(open(SRC).read().split('# ---------------------------------------------------------------- check 2')[0])
S4 = VER + 'step4/'
PREF = 0.5176
reg = Z['region']; blkA = Z['block']; loA = Z['lo']; hiA = Z['hi']; kindA = Z['kind']
nodeA = np.array([blocks[b]['node'] for b in blkA]); zA = np.array([blocks[b]['z'] for b in blkA])
lmA = np.log(10 ** nodeA / 1e14); lzA = np.log1p(zA)
dPA = Z['d_P']; dEA = Z['d_E']; wPA = Z['wP']; wEA = Z['wE']; nhA = Z['nhalo']
eA = Z['P_err'] / dPA
vA = 1.0 / (0.1 ** 2 + eA ** 2)
N = len(reg); r0m = reg == 0; r1m = reg == 1; r12m = reg >= 1; r01m = reg <= 1
res = {}

# ---------------------------------------------------------------- vectorized row quadrature (same rule as row_integral)
NQ48, WQ48 = leggauss(48)
idxS = np.where(kindA == 'shell')[0]; idxI = np.where(kindA != 'shell')[0]
ua = np.log(loA[idxS]); ub = np.log(hiA[idxS])
XS = np.exp(0.5 * (ub - ua)[:, None] * NQ48 + 0.5 * (ua + ub)[:, None])
WS = (0.5 * (ub - ua))[:, None] * WQ48 * XS ** 3 / ((hiA[idxS] ** 3 - loA[idxS] ** 3) / 3)[:, None]
npan = 42
XI = []; WI = []
for i in idxI:
    e = np.linspace(np.log(hiA[i] * 1e-9), np.log(hiA[i]), npan + 1)
    A, B = e[:-1], e[1:]
    u = 0.5 * (B - A)[:, None] * NQ48 + 0.5 * (A + B)[:, None]
    x = np.exp(u)
    XI.append(x.ravel()); WI.append((4 * np.pi * (0.5 * (B - A))[:, None] * WQ48 * x ** 3).ravel())
XI = np.array(XI); WI = np.array(WI)

def rowvals(f):
    out = np.empty(N)
    out[idxS] = np.sum(WS * f(XS, idxS), 1)
    out[idxI] = np.sum(WI * f(XI, idxI), 1)
    return out

def c3(c, idx):
    return np.exp(c[0] + c[1] * lmA[idx] + c[2] * lzA[idx])[:, None]

def E_vec(pe):
    def f(x, idx):
        n0 = c3(pe[0:3], idx); ae = c3(pe[3:6], idx); be = c3(pe[6:9], idx)
        y = x / 0.5
        return n0 * np.exp(-0.2 * np.log(y) - (be - 0.2) / ae * np.log1p(y ** ae))
    return f

def T7x_vec(pe, pt):
    fE = E_vec(pe)
    def f(x, idx):
        lnT0, amT, azT, lnxe, ce, c0, bm = pt
        lm = lmA[idx][:, None]; lz = lzA[idx][:, None]
        s = (x / np.exp(c0 + bm * lm)) ** 2
        tau = np.exp(lnT0 + amT * lm + azT * lz - (x / np.exp(lnxe)) ** ce) * s / (1 + s)
        return fE(x, idx) * tau / C_T
    return f

def plaw_vec(p, family):
    def f(x, idx):
        lm = lmA[idx][:, None]; lz = lzA[idx][:, None]; z = zA[idx][:, None]
        lnP0 = p[0] + p[1] * lm + p[2] * lz; lnxc = p[3] + p[4] * lm + p[5] * lz; lnb = p[6] + p[7] * lm + p[8] * lz
        if family == 'const':
            alpha = p[9]
        elif family == 'held':
            alpha = AH[0] + AH[1] * z
        elif family == 'massbreak':
            alpha = AH[0] + AH[1] * z
            Sb = -0.3 * np.log1p(np.exp(-(lm - p[12]) / 0.3))
            lnP0 = lnP0 + (p[9] + p[13] * lz) * Sb
            lnxc = lnxc + (p[10] + p[14] * lz) * Sb
            lnb = lnb + (p[11] + p[15] * lz) * Sb
        q = x / np.exp(lnxc)
        return PREF * np.exp(lnP0) * np.exp(-0.3 * np.log(q) - np.exp(lnb) * np.log1p(q ** alpha))
    return f

def amp_of(a, lz):
    return np.exp(a[0] + a[1] * lz + a[2] * lz ** 2)

# self-test of the vectorized rule against the scalar row_integral used in steps 1-3 (a few rows)
pe_t = np.asarray(J1['results']['electron_law']['params'])
Ev = rowvals(E_vec(pe_t))
chk = [abs(Ev[i] / row_integral(lambda x, i=i: E_law(x, pe_t, lmA[i], lzA[i]), loA[i], hiA[i], kindA[i]) - 1)
       for i in list(idxS[::97]) + list(idxI[::11])]
res['vector_rule_vs_scalar_maxrel'] = float(max(chk))
print('vector self-test', res['vector_rule_vs_scalar_maxrel'], flush=True)

# ---------------------------------------------------------------- check 1: weights
B32 = np.asarray(BJ['cuts']['B32'])
xk = BN['x4__kind']; xb = BN['x4__block']; xl = BN['x4__lo']; xh = BN['x4__hi']; xw = BN['x4__weight']
sel = (xk == 'shell') & (xl >= B32[xb] - 1e-9)
idx1 = np.where(r1m)[0]
w0 = BN['B32__weight']
w1_rec = xw[sel]
w1_par = (2701.0 / 1494.0) * vA[idx1] / np.mean(vA[idx1])
c1 = dict(n_selected_x4=int(sel.sum()), n_region1=int(len(idx1)),
          region1_geometry_matches=bool(sel.sum() == len(idx1) and np.array_equal(xb[sel], blkA[idx1])
                                        and np.max(np.abs(xl[sel] - loA[idx1])) < 1e-12 and np.max(np.abs(xh[sel] - hiA[idx1])) < 1e-12),
          region0_equals_wP=bool(np.array_equal(w0, wPA[r0m])),
          sum_w_region0=float(w0.sum()), sum_w_region1_record=float(w1_rec.sum()), sum_w_region1_parity=float(w1_par.sum()),
          parity_expected=2701.0 / 1494.0 * 1025)
weights = {}
for wname, w1 in (('record', w1_rec), ('parity', w1_par)):
    w = np.zeros(N); w[r0m] = w0; w[idx1] = w1
    weights[wname] = w
JR = {'record': json.load(open(OUT + 'be_fit4r.json'))['results'],
      'parity': json.load(open(OUT + 'be_fit4r_parity.json'))['results']}
RR = {'record': np.load(OUT + 'be_fit4r_rows.npz'), 'parity': np.load(OUT + 'be_fit4r_parity_rows.npz')}
for wname in ('record', 'parity'):
    ws = JR[wname]['weight_share']
    c1[wname + '_study_sum_region0'] = ws['sum_w_region0']; c1[wname + '_study_sum_region1'] = ws['sum_w_region1']
    c1[wname + '_rel_region1'] = float(abs(weights[wname][idx1].sum() / ws['sum_w_region1'] - 1))
    c1[wname + '_rel_region0'] = float(abs(weights[wname][r0m].sum() / ws['sum_w_region0'] - 1))
    c1[wname + '_share_region1'] = float(weights[wname][idx1].sum() / weights[wname][r01m].sum())
    c1[wname + '_template_same_as_compare'] = bool(np.array_equal(RR[wname]['template'], np.load(OUT + 'be_compare_rows.npz')['template']))
res['check1'] = c1
print('check1', json.dumps(c1, indent=1), flush=True)
tA = RR['record']['template']

# ---------------------------------------------------------------- model + scores
def model(name, law, amp):
    law = np.asarray(law, float); amp = np.asarray(amp, float)
    if name.startswith('plaw_const'):
        one = rowvals(plaw_vec(law, 'const')); E = None
    elif name.startswith('plaw_held'):
        one = rowvals(plaw_vec(law, 'held')); E = None
    elif name.startswith('plaw_massbreak'):
        one = rowvals(plaw_vec(law, 'massbreak')); E = None
    elif name.startswith('joint_T7x'):
        one = rowvals(T7x_vec(law[:9], law[9:16])); E = rowvals(E_vec(law[:9]))
    return one, E, one + amp_of(amp, lzA) * tA

def scores(w, one, E, m):
    lr = np.log(m / dPA)
    out = dict(L_4R=float(np.sum(w[r01m] * lr[r01m] ** 2)), L_4R_region1=float(np.sum(w[r1m] * lr[r1m] ** 2)),
               L_P=float(np.sum(wPA[r0m] * lr[r0m] ** 2)), L_out6=float(np.sum(vA[r12m] * lr[r12m] ** 2)))
    if E is not None:
        out['L_E'] = float(np.sum(wEA[r0m] * np.log(E[r0m] / dEA[r0m]) ** 2))
    bands = {}
    for b1, b2 in ((1.5, 2.0), (2.0, 3.0), (3.0, 4.01), (4.01, 6.01)):
        s = (kindA == 'shell') & r12m & (nhA >= 100) & (hiA > b1) & (hiA <= b2)
        bands['%g-%g' % (b1, b2)] = dict(rms=float(np.sqrt(np.mean(lr[s] ** 2))), mean=float(np.mean(lr[s])), n=int(s.sum()))
    out['outer_all_masses'] = bands
    return out

res['check2'] = {}
for wname in ('record', 'parity'):
    for name in ('plaw_const_4R', 'plaw_massbreak_4R', 'joint_T7x_4R'):
        js = JR[wname][name]
        one, E, m = model(name, js['law'], js['amp'])
        sc = scores(weights[wname], one, E, m)
        d = dict(mine={k: v for k, v in sc.items() if k != 'outer_all_masses'})
        d['study'] = {k: js[k] for k in d['mine']}
        d['rel'] = {k: float(abs(d['mine'][k] / d['study'][k] - 1)) for k in d['mine']}
        rr = RR[wname]
        r1 = np.abs(one / rr['one__' + name] - 1); rf = np.abs(m / rr['full__' + name] - 1)
        d['one_rows'] = dict(max=float(r1.max()), max_shell=float(r1[idxS].max()), max_integral=float(r1[idxI].max()))
        d['full_rows_max'] = float(rf.max())
        if E is not None:
            d['E_rows_max'] = float(np.max(np.abs(E / rr['E__' + name] - 1)))
        bc = {}
        for k, v in sc['outer_all_masses'].items():
            sv = js['outer_all_masses'][k]
            bc[k] = dict(rms_mine=v['rms'], rms_study=sv['rms'], absdiff=abs(v['rms'] - sv['rms']), n_mine=v['n'], n_study=sv['n'])
        d['outer_bands'] = bc
        d['outer_bands_max_absdiff'] = float(max(b['absdiff'] for b in bc.values()))
        d['outer_bands_n_equal'] = bool(all(b['n_mine'] == b['n_study'] for b in bc.values()))
        res['check2'][wname + '|' + name] = d
        print(wname, name, json.dumps({k: v for k, v in d.items() if k != 'outer_bands'}), flush=True)

# ---------------------------------------------------------------- check 3: optimality with all parameters free
def make_resid(name, w):
    sw = np.sqrt(w[r01m]); swE = np.sqrt(wEA[r0m])
    nl = len(JR['record'][name]['law'])
    def r(theta):
        law, amp = theta[:nl], theta[nl:]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            one, E, m = model(name, law, amp)
            out = sw * np.log(m[r01m] / dPA[r01m])
            if E is not None:
                out = np.concatenate([out, swE * np.log(E[r0m] / dEA[r0m])])
        out = np.where(np.isfinite(out), out, 1e3)
        return out
    return r, nl

res['check3'] = {}
rng = np.random.default_rng(20261126)
for wname in ('record', 'parity'):
    for name in ('plaw_const_4R', 'joint_T7x_4R'):
        js = JR[wname][name]
        r, nl = make_resid(name, weights[wname])
        th0 = np.concatenate([js['law'], js['amp']])
        L0 = float(np.sum(r(th0) ** 2))
        Lstudy = js['L_4R'] + (js['L_E'] if name.startswith('joint') else 0.0)
        t1 = time.time()
        ls = least_squares(r, th0, x_scale='jac', max_nfev=40, xtol=1e-12, ftol=1e-13, gtol=1e-12)
        d = dict(objective_mine_at_reported=L0, objective_study=Lstudy,
                 local_LS_objective=float(2 * ls.cost), local_LS_decrease=float(L0 - 2 * ls.cost),
                 local_LS_max_param_change=float(np.max(np.abs(ls.x - th0))), local_LS_nfev=int(ls.nfev),
                 local_seconds=time.time() - t1)
        # Nelder-Mead polish from the reported point (a few hundred evaluations)
        from scipy.optimize import minimize
        nm = minimize(lambda th: float(np.sum(r(th) ** 2)), th0, method='Nelder-Mead',
                      options=dict(maxfev=600, xatol=1e-9, fatol=1e-12, initial_simplex=None))
        d['nelder_mead_decrease'] = float(L0 - nm.fun); d['nelder_mead_nfev'] = int(nm.nfev)
        starts = []
        for k in range(10):
            ths = th0 + rng.normal(size=th0.size) * 0.05
            Ls0 = float(np.sum(r(ths) ** 2))
            lsk = least_squares(r, ths, x_scale='jac', max_nfev=300, xtol=1e-11, ftol=1e-12, gtol=1e-11)
            starts.append(dict(L_start=Ls0, L_end=float(2 * lsk.cost), max_dist_from_reported=float(np.max(np.abs(lsk.x - th0))),
                               nfev=int(lsk.nfev), status=int(lsk.status)))
        Lmin = min(s['L_end'] for s in starts)
        d['random_starts'] = starts
        d['random_best_objective'] = Lmin
        d['random_best_minus_reported_mine'] = float(Lmin - L0)
        d['n_random_lower_by_more_than_1e-3'] = int(sum(s['L_end'] < L0 - 1e-3 for s in starts))
        d['seconds'] = time.time() - t1
        res['check3'][wname + '|' + name] = d
        print('check3', wname, name, json.dumps({k: v for k, v in d.items() if k != 'random_starts'}), flush=True)
        print('   starts', [(round(s['L_start'], 4), round(s['L_end'], 6), s['nfev']) for s in starts], flush=True)
        json.dump(res, open(S4 + 'out/verify_step4_results.json', 'w'), indent=1, default=float)

res['seconds'] = time.time() - t0
json.dump(res, open(S4 + 'out/verify_step4_results.json', 'w'), indent=1, default=float)
print('done', res['seconds'])
