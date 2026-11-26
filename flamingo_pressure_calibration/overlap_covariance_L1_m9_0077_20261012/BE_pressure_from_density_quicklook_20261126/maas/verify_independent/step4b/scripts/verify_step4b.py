"""Independent verification of BE step 4b (be_fit4r --set flex / eflex, parity weights).
Written from the docstrings (be_fit.py T8xm/T8xz/T9x line, ELaw.point, be_fit4r.py) and the coordinator's brief only.

Electron law (general): E = n0 q^g (1 + q^a_e)^(-(beta_e + g)/a_e), q = x/x_c; production x_c = 0.5, g = -0.2;
  'c' variant: x_c = exp(ln_xc_e); 'g' variant: g = gamma_e (10th electron parameter, before the temperature parameters).
  n0, a_e, beta_e: ln Q = lnA_Q + am_Q lm + az_Q lz, lm = ln(M/1e14), lz = ln(1+z), M = 10^node of the block.
Temperature T7x and extensions: tau = exp(lnT0 + am_T lm + az_T lz - (x/x_e)^c_e) s/(1+s), s = (x/x_cc)^2,
  ln x_cc = lnxcc0 + bm_cc lm, ln x_e = ln_xe + em_xe lm (T8xm, T9x) + ez_xe lz (T8xz, T9x); P1 = E tau / C_T.
Model m = P1 + A_2h(z) t, ln A_2h = a + b lz + c lz^2; L_4R (parity weights, regions 0+1), L_P (region 0, wP),
L_E (region 0, wE), L_out6 (regions 1+2, v), outer RMS (shells, region >= 1, nhalo >= 100, hi in (lo_b, hi_b]).
Row values: shell volume means / 4 pi int_0^hi x^2 f dx, 48-point Gauss-Legendre in ln x (as steps 1-4).
"""
import json, time
import numpy as np
from scipy.optimize import least_squares

t0 = time.time()
SRC4 = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/step4/scripts/verify_step4.py'
exec(open(SRC4).read().split('# ---------------------------------------------------------------- model + scores')[0])
S4B = VER + 'step4b/'
w = weights['parity']
JF = json.load(open(OUT + 'be_fit4r_parity_flex.json'))['results']
JE = json.load(open(OUT + 'be_fit4r_parity_eflex.json'))['results']
RF = np.load(OUT + 'be_fit4r_parity_flex_rows.npz'); RE = np.load(OUT + 'be_fit4r_parity_eflex_rows.npz')
pe_sep = np.asarray(J1['results']['electron_law']['params'], float)
res = dict(template_same=bool(np.array_equal(RF['template'], tA) and np.array_equal(RE['template'], tA)),
           weight_share_flex=JF['weight_share'], weight_share_eflex=JE['weight_share'],
           my_sum_w_region1=float(w[r1m].sum()))
ENAMES = ['lnA_n0', 'am_n0', 'az_n0', 'lnA_ae', 'am_ae', 'az_ae', 'lnA_betae', 'am_betae', 'az_betae']

def E_gen(pe, xc_ln=None, g=None):
    gg = -0.2 if g is None else g
    lxc = np.log(0.5) if xc_ln is None else xc_ln
    def f(x, idx):
        n0 = c3(pe[0:3], idx); ae = c3(pe[3:6], idx); be = c3(pe[6:9], idx)
        lq = np.log(x) - lxc
        return n0 * np.exp(gg * lq - (be + gg) / ae * np.log1p(np.exp(ae * lq)))
    return f

def build(names, vals, sep=False):
    """Return (fE, fP) closures from a named parameter vector."""
    d = dict(zip(names, vals))
    pe = pe_sep if sep else np.array([d[k] for k in ENAMES])
    fE = E_gen(pe, d.get('ln_xc_e'), d.get('gamma_e'))
    em = d.get('em_xe', 0.0); ez = d.get('ez_xe', 0.0)
    def fP(x, idx):
        lm = lmA[idx][:, None]; lz = lzA[idx][:, None]
        lnxe = d['ln_xe'] + em * lm + ez * lz
        s = (x / np.exp(d['lnxcc0'] + d['bm_cc'] * lm)) ** 2
        tau = np.exp(d['lnT0'] + d['am_T'] * lm + d['az_T'] * lz - (x / np.exp(lnxe)) ** d['c_e']) * s / (1 + s)
        return fE(x, idx) * tau / C_T
    return fE, fP

def evaluate(names, law, amp, sep=False):
    fE, fP = build(names, law, sep)
    E = rowvals(fE); one = rowvals(fP)
    m = one + amp_of(np.asarray(amp), lzA) * tA
    return one, E, m

def score(one, E, m):
    lr = np.log(m / dPA)
    out = dict(L_4R=float(np.sum(w[r01m] * lr[r01m] ** 2)), L_4R_region1=float(np.sum(w[r1m] * lr[r1m] ** 2)),
               L_P=float(np.sum(wPA[r0m] * lr[r0m] ** 2)), L_E=float(np.sum(wEA[r0m] * np.log(E[r0m] / dEA[r0m]) ** 2)),
               L_out6=float(np.sum(vA[r12m] * lr[r12m] ** 2)))
    bands = {}
    for b1, b2 in ((1.5, 2.0), (2.0, 3.0), (3.0, 4.01), (4.01, 6.01)):
        s = (kindA == 'shell') & r12m & (nhA >= 100) & (hiA > b1) & (hiA <= b2)
        bands['%g-%g' % (b1, b2)] = dict(rms=float(np.sqrt(np.mean(lr[s] ** 2))), mean=float(np.mean(lr[s])), n=int(s.sum()))
    out['outer_all_masses'] = bands
    return out

# ---------------------------------------------------------------- check 1
res['check1'] = {}
for J, R, name in ((JF, RF, 'joint_T9x_4R'), (JF, RF, 'sep_T9x_4R'), (JE, RE, 'joint_T7x_eg_4R'), (JE, RE, 'joint_T8xm_ec_4R'),
                   (JE, RE, 'joint_T7x_ec_4R'), (JF, RF, 'joint_T8xm_4R')):
    js = J[name]; sep = name.startswith('sep')
    one, E, m = evaluate(js['names'], js['law'], js['amp'], sep)
    sc = score(one, E, m)
    d = dict(mine={k: v for k, v in sc.items() if k != 'outer_all_masses'})
    d['study'] = {k: js[k] for k in d['mine']}
    d['rel'] = {k: float(abs(d['mine'][k] / d['study'][k] - 1)) for k in d['mine']}
    d['one_rows_max_shell'] = float(np.max(np.abs(one[idxS] / R['one__' + name][idxS] - 1)))
    d['one_rows_max_integral'] = float(np.max(np.abs(one[idxI] / R['one__' + name][idxI] - 1)))
    d['full_rows_max'] = float(np.max(np.abs(m / R['full__' + name] - 1)))
    d['E_rows_max'] = float(np.max(np.abs(E / R['E__' + name] - 1)))
    bd = {k: abs(v['rms'] - js['outer_all_masses'][k]['rms']) for k, v in sc['outer_all_masses'].items()}
    d['outer_rms_max_absdiff'] = float(max(bd.values()))
    d['outer_n_equal'] = bool(all(v['n'] == js['outer_all_masses'][k]['n'] for k, v in sc['outer_all_masses'].items()))
    d['outer_rms_mine'] = {k: round(v['rms'], 5) for k, v in sc['outer_all_masses'].items()}
    d['at_bounds'] = js.get('at_bounds'); d['counts'] = js.get('counts')
    res['check1'][name] = d
    print(name, json.dumps(d), flush=True)

# ---------------------------------------------------------------- check 3: electron laws with one extra constant
idx0 = np.where(r0m)[0]
res['check3'] = {}
for key, extra_name, extra0, ub in (('electron_law_c', 'ln_xc_e', np.log(0.5), np.inf), ('electron_law_g', 'gamma_e', -0.2, 1.0)):
    js = JE[key]
    names = js['names']
    def LE_of(vals):
        d = dict(zip(names, vals))
        E = rowvals(E_gen(np.array([d[k] for k in ENAMES]), d.get('ln_xc_e'), d.get('gamma_e')))
        return float(np.sum(wEA[r0m] * np.log(E[r0m] / dEA[r0m]) ** 2)), E
    LE_saved, E_saved = LE_of(js['params'])
    def r(vals):
        d = dict(zip(names, vals))
        E = rowvals(E_gen(np.array([d[k] for k in ENAMES]), d.get('ln_xc_e'), d.get('gamma_e')))
        out = np.sqrt(wEA[idx0]) * np.log(E[idx0] / dEA[idx0])
        return np.where(np.isfinite(out), out, 1e3)
    start = np.concatenate([pe_sep, [extra0]])
    L_start = float(np.sum(r(start) ** 2))
    lb = np.full(10, -np.inf); ubv = np.full(10, np.inf); ubv[9] = ub
    x_start = start.copy()
    if np.isfinite(ub):
        x_start[9] = min(x_start[9], ub - 1e-9)
    ls_b = least_squares(r, x_start, bounds=(lb, ubv), x_scale='jac', xtol=1e-12, ftol=1e-13, gtol=1e-12, max_nfev=2000)
    d = dict(L_E_mine_at_saved=LE_saved, L_E_study=js['L_E'], rel=float(abs(LE_saved / js['L_E'] - 1)),
             L_E_9par_study=js['L_E_9par'], L_E_at_start_9par_plus_const=L_start,
             refit_bounded_L_E=float(2 * ls_b.cost), refit_bounded_minus_study=float(2 * ls_b.cost - js['L_E']),
             refit_bounded_extra=float(ls_b.x[9]), saved_extra=float(js['params'][9]),
             refit_bounded_max_param_diff_vs_saved=float(np.max(np.abs(ls_b.x - np.asarray(js['params'])))),
             refit_bounded_nfev=int(ls_b.nfev), upper_bound_used=ub)
    if np.isfinite(ub):   # unbounded refit for information (where does the free inner slope go?)
        ls_u = least_squares(r, ls_b.x, x_scale='jac', xtol=1e-12, ftol=1e-13, gtol=1e-12, max_nfev=2000)
        d['refit_unbounded_L_E'] = float(2 * ls_u.cost); d['refit_unbounded_extra'] = float(ls_u.x[9])
        d['refit_unbounded_nfev'] = int(ls_u.nfev)
        # local slope of the saved 'g' law at small radii (z=0, node 14 block-like mass)
        dd = dict(zip(names, js['params']))
        ae = np.exp(dd['lnA_ae']); be = np.exp(dd['lnA_betae']); g = dd['gamma_e']
        xs = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
        qa = (xs / 0.5) ** ae
        d['saved_g_local_slope_M1e14_z0'] = dict(zip(['x%g' % x for x in xs], (g - (be + g) * qa / (1 + qa)).round(3).tolist()))
    # local slope of the 9-par production law for comparison
    ae9 = np.exp(pe_sep[3]); be9 = np.exp(pe_sep[6]); qa9 = (np.array([0.005, 0.01, 0.02, 0.05, 0.1]) / 0.5) ** ae9
    d['production_local_slope_M1e14_z0'] = (-0.2 - (be9 - 0.2) * qa9 / (1 + qa9)).round(3).tolist()
    res['check3'][key] = d
    print(key, json.dumps(d), flush=True)
json.dump(res, open(S4B + 'out/verify_step4b_results.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- check 2: optimality (all parameters free)
res['check2'] = {}
rng = np.random.default_rng(4242)
for J, name in ((JF, 'joint_T9x_4R'), (JE, 'joint_T7x_ec_4R')):
    js = J[name]; names = js['names']; nl = len(js['law'])
    sw = np.sqrt(w[r01m]); swE = np.sqrt(wEA[r0m])
    def r(th):
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            one, E, m = evaluate(names, th[:nl], th[nl:])
            out = np.concatenate([sw * np.log(m[r01m] / dPA[r01m]), swE * np.log(E[r0m] / dEA[r0m])])
        return np.where(np.isfinite(out), out, 1e3)
    th0 = np.concatenate([js['law'], js['amp']])
    L0 = float(np.sum(r(th0) ** 2))
    t1 = time.time()
    ls = least_squares(r, th0, x_scale='jac', max_nfev=40, xtol=1e-12, ftol=1e-13, gtol=1e-12)
    d = dict(objective_mine=L0, objective_study=js['L_4R'] + js['L_E'], local_decrease=float(L0 - 2 * ls.cost),
             local_max_param_change=float(np.max(np.abs(ls.x - th0))), local_nfev=int(ls.nfev))
    starts = []
    for k in range(5):
        ths = th0 + rng.normal(size=th0.size) * 0.05
        Ls = float(np.sum(r(ths) ** 2))
        lk = least_squares(r, ths, x_scale='jac', max_nfev=300, xtol=1e-11, ftol=1e-12, gtol=1e-11)
        starts.append(dict(L_start=Ls, L_end=float(2 * lk.cost), max_dist=float(np.max(np.abs(lk.x - th0))), nfev=int(lk.nfev)))
    d['random_starts'] = starts
    d['random_best_minus_reported'] = float(min(s['L_end'] for s in starts) - L0)
    d['n_lower_by_more_than_1e-3'] = int(sum(s['L_end'] < L0 - 1e-3 for s in starts))
    d['seconds'] = time.time() - t1
    res['check2'][name] = d
    print('check2', name, json.dumps(d), flush=True)
    json.dump(res, open(S4B + 'out/verify_step4b_results.json', 'w'), indent=1, default=float)

res['seconds'] = time.time() - t0
json.dump(res, open(S4B + 'out/verify_step4b_results.json', 'w'), indent=1, default=float)
print('done', res['seconds'])
