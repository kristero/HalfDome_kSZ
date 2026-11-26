"""Step 5 checks 2-4: weights, both-field 4 R200c losses from saved parameters, optimality (be_fit4r_e.py docstring).

Models: P_i = P1h_i + A_P(z_i) t^P_i, E_i = E1h_i + A_E(z_i) t^E_i, ln A_F = a_F + b_F lz + c_F lz^2.
  t^P = template_P, t^E = template_E_ex0p75+mean (be_fit4r_e_rows.npz).
  E1h = 9-parameter electron law; P1h = pressure law (split_const: constant alpha) or E1h x T7x temperature (joint_T7x).
Weights: P region 0 = wP, P region 1 = (W0/N0) v/<v>, v = 1/(0.1^2 + (P_err/d_P)^2);
         E region 0 = wE, E region 1 = (W0/N0) v_E/<v_E>, v_E = 1/(0.1^2 + (E_err/d_E)^2); W0 = 2701, N0 = 1494.
Losses: L_4R_P = sum_{0,1} w_P ln^2(P/d_P); L_4R_E = sum_{0,1} w_E ln^2(E/d_E); L_P, L_E = region-0 parts (step-2 weights);
  L_out6 = sum_{1,2} v ln^2(P/d_P), L_out6_E = sum_{1,2} v_E ln^2(E/d_E); outer RMS by band (shells, region >= 1, nhalo >= 100).
"""
import json, time
import numpy as np
from scipy.optimize import least_squares

t0 = time.time()
SRC4 = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/step4/scripts/verify_step4.py'
exec(open(SRC4).read().split('# ---------------------------------------------------------------- model + scores')[0])
S5 = VER + 'step5/'
JX = json.load(open(OUT + 'be_fit4r_e.json'))['results']
RX = np.load(OUT + 'be_fit4r_e_rows.npz')
tP = RX['template_P']; tE = RX['template_E_ex0p75+mean']
eE = Z['E_err'] / dEA
vE = 1.0 / (0.1 ** 2 + eE ** 2)
idx1 = np.where(r1m)[0]
wP4 = weights['parity'].copy()
wE4 = np.zeros(N); wE4[r0m] = wEA[r0m]; wE4[idx1] = (2701.0 / 1494.0) * vE[idx1] / np.mean(vE[idx1])
res = dict(template_P_equals_step4=bool(np.array_equal(tP, tA)),
           template_E_used=JX['electron_template_used'])
ws_ = JX['weights']
res['check2'] = dict(mine=dict(P_region0=float(wP4[r0m].sum()), P_region1=float(wP4[r1m].sum()),
                               E_region0=float(wE4[r0m].sum()), E_region1=float(wE4[r1m].sum())),
                     study={k: ws_[k] for k in ('P_region0', 'P_region1', 'E_region0', 'E_region1')})
res['check2']['rel'] = {k: float(abs(res['check2']['mine'][k] / res['check2']['study'][k] - 1)) for k in res['check2']['mine']}
print('check2', json.dumps(res['check2']), flush=True)

def bands(lr):
    out = {}
    for b1, b2 in ((1.5, 2.0), (2.0, 3.0), (3.0, 4.01), (4.01, 6.01)):
        s = (kindA == 'shell') & r12m & (nhA >= 100) & (hiA > b1) & (hiA <= b2)
        out['%g-%g' % (b1, b2)] = dict(rms=float(np.sqrt(np.mean(lr[s] ** 2))), mean=float(np.mean(lr[s])), n=int(s.sum()))
    return out

def model(name, law=None, amp=None, law_E=None, amp_E=None):
    """Return dict with one/full for P (if any) and oneE/fullE."""
    o = {}
    if name.startswith('elaw'):
        E1 = rowvals(E_vec(np.asarray(law_E)))
    elif name == 'split_const':
        E1 = rowvals(E_vec(np.asarray(law_E)))
        P1 = rowvals(plaw_vec(np.asarray(law), 'const'))
        o['one'] = P1; o['full'] = P1 + amp_of(np.asarray(amp), lzA) * tP
    elif name.startswith('joint_T7x'):
        law = np.asarray(law)
        E1 = rowvals(E_vec(law[:9]))
        P1 = rowvals(T7x_vec(law[:9], law[9:16]))
        o['one'] = P1; o['full'] = P1 + amp_of(np.asarray(amp), lzA) * tP
    o['oneE'] = E1; o['fullE'] = E1 + amp_of(np.asarray(amp_E), lzA) * tE
    return o

def scores(o):
    s = {}
    lE = np.log(o['fullE'] / dEA)
    s['L_4R_E'] = float(np.sum(wE4[r01m] * lE[r01m] ** 2)); s['L_E'] = float(np.sum(wEA[r0m] * lE[r0m] ** 2))
    s['L_out6_E'] = float(np.sum(vE[r12m] * lE[r12m] ** 2)); s['outer_all_masses_E'] = bands(lE)
    if 'full' in o:
        lP = np.log(o['full'] / dPA)
        s['L_4R_P'] = float(np.sum(wP4[r01m] * lP[r01m] ** 2)); s['L_P'] = float(np.sum(wPA[r0m] * lP[r0m] ** 2))
        s['L_out6'] = float(np.sum(vA[r12m] * lP[r12m] ** 2)); s['outer_all_masses'] = bands(lP)
    return s

res['check3'] = {}
for name in ('elaw_2h_ex0p75+mean', 'split_const', 'joint_T7x'):
    js = JX[name]
    if name.startswith('elaw'):
        o = model(name, law_E=js['params'], amp_E=js['amp_E'])
    elif name == 'split_const':
        o = model(name, law=js['law'], amp=js['amp'], law_E=js['law_E'], amp_E=js['amp_E'])
    else:
        o = model(name, law=js['law'], amp=js['amp'], amp_E=js['amp_E'])
    sc = scores(o)
    d = dict(mine={k: v for k, v in sc.items() if not k.startswith('outer')})
    d['study'] = {k: js.get(k) for k in d['mine']}
    d['rel'] = {k: (float(abs(d['mine'][k] / d['study'][k] - 1)) if d['study'][k] is not None else None) for k in d['mine']}
    for f in ('outer_all_masses', 'outer_all_masses_E'):
        if f in sc and f in js:
            d[f + '_max_rms_absdiff'] = float(max(abs(v['rms'] - js[f][k]['rms']) for k, v in sc[f].items()))
            d[f + '_n_equal'] = bool(all(v['n'] == js[f][k]['n'] for k, v in sc[f].items()))
            d[f + '_mine'] = {k: round(v['rms'], 5) for k, v in sc[f].items()}
    for key, arr in (('one', 'one__'), ('full', 'full__'), ('oneE', 'oneE__'), ('fullE', 'fullE__')):
        if key in o and (arr + name) in RX.files:
            rr = np.abs(o[key] / RX[arr + name] - 1)
            d['rows_' + key] = dict(max_shell=float(rr[idxS].max()), max_integral=float(rr[idxI].max()))
    res['check3'][name] = d
    print('check3', name, json.dumps(d), flush=True)
json.dump(res, open(S5 + 'out/step5_fits.json', 'w'), indent=1, default=float)

# ---------------------------------------------------------------- check 4: optimality
res['check4'] = {}
rng = np.random.default_rng(5555)
sP = np.sqrt(wP4[r01m]); sE = np.sqrt(wE4[r01m])
for name in ('elaw_2h_ex0p75+mean', 'joint_T7x'):
    js = JX[name]
    if name.startswith('elaw'):
        th0 = np.concatenate([js['params'], js['amp_E']])
        def r(th):
            with np.errstate(all='ignore'):
                o = model(name, law_E=th[:9], amp_E=th[9:12])
                out = sE * np.log(o['fullE'][r01m] / dEA[r01m])
            return np.where(np.isfinite(out), out, 1e3)
        Lst = js['L_4R_E']
    else:
        th0 = np.concatenate([js['law'], js['amp'], js['amp_E']])
        def r(th):
            with np.errstate(all='ignore'):
                o = model(name, law=th[:16], amp=th[16:19], amp_E=th[19:22])
                out = np.concatenate([sP * np.log(o['full'][r01m] / dPA[r01m]), sE * np.log(o['fullE'][r01m] / dEA[r01m])])
            return np.where(np.isfinite(out), out, 1e3)
        Lst = js['L_4R_P'] + js['L_4R_E']
    L0 = float(np.sum(r(th0) ** 2))
    t1 = time.time()
    ls = least_squares(r, th0, x_scale='jac', max_nfev=40, xtol=1e-12, ftol=1e-13, gtol=1e-12)
    d = dict(n_par=int(th0.size), objective_mine=L0, objective_study=Lst, local_decrease=float(L0 - 2 * ls.cost),
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
    res['check4'][name] = d
    print('check4', name, json.dumps(d), flush=True)
    json.dump(res, open(S5 + 'out/step5_fits.json', 'w'), indent=1, default=float)
res['seconds'] = time.time() - t0
json.dump(res, open(S5 + 'out/step5_fits.json', 'w'), indent=1, default=float)
print('done', res['seconds'])
