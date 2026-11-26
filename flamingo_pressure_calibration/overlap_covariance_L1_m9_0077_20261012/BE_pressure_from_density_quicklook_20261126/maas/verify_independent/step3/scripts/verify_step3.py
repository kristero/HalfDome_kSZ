"""Independent verification of BE step 3 (be_compare): existing BD 1h+2h pressure fits vs electron law x temperature law
with the same 2-halo term. Written from the be_compare.py / be_fit.py docstrings and the coordinator's brief only.

Symbols: x = r/R200c; M = M200c [Msun] = 10^node of the row's BD block; z = block redshift; lm = ln(M/1e14); lz = ln(1+z).
  law quantity Q: ln Q = lnA_Q + am_Q lm + az_Q lz  (+ (dm_Q + kz_Q lz) S for the mass-break law MB,
                  S = -0.3 ln(1 + exp(-(lm - lmb)/0.3)), Q in {P0, x_c, beta})
  pressure law: P1 = 0.5176 P0 q^-0.3 (1 + q^alpha)^-beta, q = x/x_c; alpha = alpha0 (B) or 0.93935 + 0.29477 z (A, MB)
  electron law: E = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a_e]^(-(beta_e - 0.2)/a_e)
  T7 : tau = T0' (1 + x/x_t)^-c        s/(1+s);  T7x: tau = T0' exp[-(x/x_e)^c_e] s/(1+s);
       T0' = exp(lnT0 + am_T lm + az_T lz), s = (x/x_cc)^2, ln x_cc = c0 + bm lm;   P1 = E tau / C_T
  2-halo: m = P1 + A_2h(z) t,  ln A_2h = a + b lz + c lz^2,  t = saved template per row (be_compare_rows.npz 'template')
  row value: shell volume mean over [lo, hi] or 4 pi int_0^hi x^2 f dx (core / group rows)
  L_P = sum_{region 0} wP ln^2(m/d_P); L_out4 = sum_{region 1} v ln^2(m/d_P); L_out6 = sum_{regions 1,2} v ln^2(m/d_P)
  v = 1/(0.1^2 + e^2), e = P_err/d_P;  L_E = sum_{region 0} wE ln^2(E/d_E)
  outer RMS: shells, region >= 1, nhalo >= 100, band by hi in (lo_b, hi_b]: sqrt(mean ln^2(m/d)), mean ln(m/d), n
"""
import json, time
import numpy as np
from scipy.optimize import least_squares

t0 = time.time()
SRC = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/verify_independent/scripts/verify_be.py'
exec(open(SRC).read().split('# ---------------------------------------------------------------- check 2')[0])
BDR = '/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118/'
S3 = VER + 'step3/'
JC = json.load(open(OUT + 'be_compare.json'))['results']
C = np.load(OUT + 'be_compare_rows.npz')
PREF = 0.5176          # literal prefactor of the record law (found in the first verification)

# all rows
reg = Z['region']; blkA = Z['block']; loA = Z['lo']; hiA = Z['hi']; kindA = Z['kind']
nodeA = np.array([blocks[b]['node'] for b in blkA]); zA = np.array([blocks[b]['z'] for b in blkA])
dPA = Z['d_P']; dEA = Z['d_E']; wPA = Z['wP']; wEA = Z['wE']; nhA = Z['nhalo']; PerrA = Z['P_err']
eA = PerrA / dPA
vA = 1.0 / (0.1 ** 2 + eA ** 2)
tA = C['template']
lmA = np.log(10 ** nodeA / 1e14); lzA = np.log1p(zA)
r0m = reg == 0; r1m = reg == 1; r12m = reg >= 1
res = dict(rows=dict(n=int(len(reg)), n_region=[int((reg == k).sum()) for k in range(3)],
                     template_min=float(tA.min()), template_n_nonpositive=int((tA <= 0).sum()),
                     M_vs_10node=float(np.max(np.abs(Z['M'] / 10 ** nodeA - 1))),
                     z_vs_block=float(np.max(np.abs(Z['z'] - zA)))))

def amp_of(a, lz):
    return np.exp(a[0] + a[1] * lz + a[2] * lz ** 2)

def law_qty(p, lm, lz, mb=None):
    lnP0 = p[0] + p[1] * lm + p[2] * lz
    lnxc = p[3] + p[4] * lm + p[5] * lz
    lnb = p[6] + p[7] * lm + p[8] * lz
    if mb is not None:
        dm, lmb, kz = mb
        Sb = -0.3 * np.log1p(np.exp(-(lm - lmb) / 0.3))
        lnP0 = lnP0 + (dm[0] + kz[0] * lz) * Sb
        lnxc = lnxc + (dm[1] + kz[1] * lz) * Sb
        lnb = lnb + (dm[2] + kz[2] * lz) * Sb
    return np.exp(lnP0), np.exp(lnxc), np.exp(lnb)

def pressure_law(x, p, lm, lz, z, family):
    if family == 'B':
        alpha = p[9]; mb = None
    elif family == 'A':
        alpha = AH[0] + AH[1] * z; mb = None
    elif family == 'MB':
        alpha = AH[0] + AH[1] * z
        mb = ((p[9], p[10], p[11]), p[12], (p[13], p[14], p[15]))
    P0, xc, beta = law_qty(p, lm, lz, mb)
    q = x / xc
    return PREF * P0 * np.exp(-0.3 * np.log(q) - beta * np.log1p(q ** alpha))

def T7x_tau(x, pt, lm, lz):
    lnT0, amT, azT, lnxe, ce, c0, bm = pt
    s = (x / np.exp(c0 + bm * lm)) ** 2
    return np.exp(lnT0 + amT * lm + azT * lz - (x / np.exp(lnxe)) ** ce) * s / (1 + s)

def rows_eval(f_of_i, idx):
    return np.array([row_integral(f_of_i(i), loA[i], hiA[i], kindA[i]) for i in idx])

def scores(one, E=None):
    m = one + amp_cur * tA
    lr = np.log(m / dPA)
    out = dict(L_P=float(np.sum(wPA[r0m] * lr[r0m] ** 2)), L_out4=float(np.sum(vA[r1m] * lr[r1m] ** 2)),
               L_out6=float(np.sum(vA[r12m] * lr[r12m] ** 2)))
    if E is not None:
        out['L_E'] = float(np.sum(wEA[r0m] * np.log(E[r0m] / dEA[r0m]) ** 2))
    bands = {}
    for b1, b2 in ((1.5, 2.0), (2.0, 3.0), (3.0, 4.01), (4.01, 6.01)):
        s = (kindA == 'shell') & r12m & (nhA >= 100) & (hiA > b1) & (hiA <= b2)
        bands['%g-%g' % (b1, b2)] = dict(rms=float(np.sqrt(np.mean(lr[s] ** 2))), mean=float(np.mean(lr[s])), n=int(s.sum()))
    out['outer_all_masses'] = bands
    return out, m

def compare(name, mine, one, m, E=None):
    js = JC[name]
    d = dict(mine={k: mine[k] for k in ('L_P', 'L_out4', 'L_out6', 'L_E') if k in mine})
    d['study'] = {k: js.get(k) for k in d['mine']}
    d['rel'] = {k: float(abs(d['mine'][k] / d['study'][k] - 1)) for k in d['mine']}
    rr1 = np.abs(one / C['one__' + name] - 1); rrf = np.abs(m / C['full__' + name] - 1)
    d['one_rows'] = dict(max=float(rr1.max()), median=float(np.median(rr1)),
                         max_shell=float(rr1[kindA == 'shell'].max()), max_integral=float(rr1[kindA != 'shell'].max()))
    d['full_rows'] = dict(max=float(rrf.max()), median=float(np.median(rrf)))
    if E is not None:
        rre = np.abs(E / C['E__' + name] - 1)
        d['E_rows'] = dict(max=float(rre.max()), median=float(np.median(rre)))
    band_cmp = {}
    for k, v in mine['outer_all_masses'].items():
        sv = js['outer_all_masses'][k]
        band_cmp[k] = dict(rms_mine=v['rms'], rms_study=sv['rms'], rms_absdiff=abs(v['rms'] - sv['rms']),
                           mean_absdiff=abs(v['mean'] - sv['mean']), n_mine=v['n'], n_study=sv['n'])
    d['outer_bands'] = band_cmp
    d['outer_bands_max_rms_absdiff'] = float(max(b['rms_absdiff'] for b in band_cmp.values()))
    d['outer_bands_n_equal'] = bool(all(b['n_mine'] == b['n_study'] for b in band_cmp.values()))
    zs = sorted(set(np.round(zA, 4)))
    d['amp_by_z_maxrel'] = float(max(abs(amp_of(js['amp'], np.log1p(z)) / js['amp_by_z']['%g' % z] - 1) for z in zs))
    d['counts'] = js['counts']
    return d

# ---------------------------------------------------------------- check 1: existing BD fits
existing = {
    'existing_B6': ('B', 'explore2h/corner_1h_vs_1h2h_20261120/runs/B_2h_rout6.json', 'law_best', 'amp_best'),
    'existing_B4': ('B', 'explore2h/final/final_curved075_ivar.json', 'law_best', 'amp_best'),
    'existing_A6': ('A', 'explore2h/corner_1h_vs_1h2h_20261120/runs/A_2h_rout6.json', 'law_best', 'amp_best'),
    'existing_A4': ('A', 'explore2h/final/rec_held_curved.json', 'law_best', 'amp_best'),
    'existing_MB4': ('MB', 'explore2h/mass_weights/final/best_sbplzz_held.json', 'law', 'amp'),
}
res['check1'] = {}
allidx = np.arange(len(reg))
for name, (fam, path, lk, ak) in existing.items():
    B = json.load(open(BDR + path))
    p = np.asarray(B[lk], float); a = np.asarray(B[ak], float)
    amp_cur = amp_of(a, lzA)
    one = rows_eval(lambda i: (lambda x, i=i: pressure_law(x, p, lmA[i], lzA[i], zA[i], fam)), allidx)
    mine, m = scores(one)
    d = compare(name, mine, one, m)
    d['params_equal_study_json'] = bool(np.allclose(p, JC[name]['law'], rtol=0, atol=0) and np.allclose(a, JC[name]['amp'], rtol=0, atol=0))
    # saved BD losses
    if 'L_in' in B:
        d['BD_saved_L_in'] = B['L_in']; d['BD_saved_L_out'] = B['L_out']
    else:
        d['BD_saved_L_in'] = B['info']['L_fit']; d['BD_saved_L_out'] = B['info']['L_amp']
    rout = 6 if 'rout6' in path or '_rout6' in name or name.endswith('6') else 4
    d['rel_L_P_vs_BD_L_in'] = float(abs(mine['L_P'] / d['BD_saved_L_in'] - 1))
    Lo = mine['L_out6'] if name.endswith('6') else mine['L_out4']
    d['rel_L_out_vs_BD_L_out'] = float(abs(Lo / d['BD_saved_L_out'] - 1))
    # with the exact electron fraction instead of 0.5176 (diagnostic)
    one_x = one * PE / PREF
    m_x = one_x + amp_cur * tA
    d['L_P_with_exact_PE'] = float(np.sum(wPA[r0m] * np.log(m_x[r0m] / dPA[r0m]) ** 2))
    if fam == 'MB':   # the saved reference points P0, x_c, beta
        rp = {}
        for key, v in B['reference_points'].items():
            lm_ = np.log(10 ** v['M'] / 1e14); lz_ = np.log1p(v['z'])
            P0, xc, be = law_qty(p, lm_, lz_, ((p[9], p[10], p[11]), p[12], (p[13], p[14], p[15])))
            rp[key] = float(max(abs(P0 / v['P0'] - 1), abs(xc / v['x_c'] - 1), abs(be / v['beta'] - 1),
                                abs((AH[0] + AH[1] * v['z']) / v['alpha'] - 1)))
        d['MB_reference_points_maxrel'] = rp
    res['check1'][name] = d
    print(name, json.dumps({k: v for k, v in d.items() if k != 'outer_bands'}, indent=1), flush=True)

# template column vs the BD template files (where the shapes allow a direct match)
tm = {}
for f in ('explore2h/run/templates/flamingo_B32_full_hm_ex0.75.npy', 'explore2h/run/templates/flamingo_x4_full_hm_ex0.75.npy',
          'explore2h/mass_weights/templates/far_hm_ex0p75.npy'):
    try:
        arr = np.load(BDR + f)
        tm[f] = dict(shape=list(arr.shape))
        if arr.ndim == 1 and arr.shape[0] == int(r0m.sum()):
            tm[f]['vs_region0_maxrel'] = float(np.max(np.abs(arr / tA[r0m] - 1)))
        if arr.ndim == 1 and arr.shape[0] == int((reg == 2).sum()):
            tm[f]['vs_region2_maxrel'] = float(np.max(np.abs(arr / tA[reg == 2] - 1)))
        if arr.ndim == 1 and f.endswith('x4_full_hm_ex0.75.npy'):
            # map region-1 rows to x4 rows by (block, lo, hi)
            key = {(int(b), round(float(l), 10), round(float(h), 10)): k for k, (b, l, h) in
                   enumerate(zip(BN['x4__block'], BN['x4__lo'], BN['x4__hi']))}
            ii = np.where(r1m)[0]
            kk = [key.get((int(blkA[i]), round(float(loA[i]), 10), round(float(hiA[i]), 10))) for i in ii]
            ok = [j for j, k in enumerate(kk) if k is not None]
            if ok:
                tm[f]['region1_matched'] = len(ok)
                tm[f]['vs_region1_maxrel'] = float(np.max(np.abs(arr[[kk[j] for j in ok]] / tA[ii[ok]] - 1)))
    except Exception as ex:
        tm[f] = dict(error=repr(ex))
res['template_files'] = tm
print('templates', tm, flush=True)

# ---------------------------------------------------------------- check 2: joint_T7x_rout6 and joint_T7_rout6 (+ the other T fits)
res['check2'] = {}
store = {}
for name in ('joint_T7x_rout6', 'joint_T7_rout6', 'joint_T7x_rout4', 'joint_T7_rout4', 'sep_T7x_rout6', 'sep_T7_rout6'):
    js = JC[name]; p = np.asarray(js['params'], float); a = np.asarray(js['amp'], float)
    pe = p[:9]; pt = p[9:16]
    tfun = T7x_tau if js['kind'] == 'T7x' else T7_tau
    amp_cur = amp_of(a, lzA)
    E = rows_eval(lambda i: (lambda x, i=i: E_law(x, pe, lmA[i], lzA[i])), allidx)
    one = rows_eval(lambda i: (lambda x, i=i: E_law(x, pe, lmA[i], lzA[i]) * tfun(x, pt, lmA[i], lzA[i]) / C_T), allidx)
    mine, m = scores(one, E)
    d = compare(name, mine, one, m, E)
    res['check2'][name] = d
    store[name] = (p, a, one, E)
    print(name, json.dumps({k: v for k, v in d.items() if k != 'outer_bands'}, indent=1), flush=True)

# ---------------------------------------------------------------- check 3: convergence of the split procedure (joint_T7x_rout6)
name = 'joint_T7x_rout6'
p, a, one, E = store[name]
# (a) amplitude given the law: least squares in ln on regions 1+2 with weights v
idx12 = np.where(r12m)[0]
def r_amp(aa):
    return np.sqrt(vA[idx12]) * np.log((one[idx12] + amp_of(aa, lzA[idx12]) * tA[idx12]) / dPA[idx12])
L6_0 = float(np.sum(r_amp(a) ** 2))
lsa = least_squares(r_amp, a, x_scale='jac', xtol=1e-14, ftol=1e-15, gtol=1e-14)
amp_new = lsa.x
c3 = dict(amp_reported=a.tolist(), amp_refit=amp_new.tolist(), amp_max_abs_change=float(np.max(np.abs(amp_new - a))),
          L_out6_at_reported=L6_0, L_out6_refit=float(2 * lsa.cost), L_out6_decrease=float(L6_0 - 2 * lsa.cost))
zs = sorted(set(np.round(zA, 4)))
c3['A2h_by_z_max_relchange'] = float(max(abs(amp_of(amp_new, np.log1p(z)) / amp_of(a, np.log1p(z)) - 1) for z in zs))
# (b) law given the amplitude: L_P + L_E on region 0, from the reported point
idx0 = np.where(r0m)[0]
A0 = amp_of(a, lzA[idx0]) * tA[idx0]
def r_law(pp):
    pe = pp[:9]; pt = pp[9:16]
    Ev = rows_eval(lambda i: (lambda x, i=i: E_law(x, pe, lmA[i], lzA[i])), idx0)
    Pv = rows_eval(lambda i: (lambda x, i=i: E_law(x, pe, lmA[i], lzA[i]) * T7x_tau(x, pt, lmA[i], lzA[i]) / C_T), idx0)
    return np.concatenate([np.sqrt(wPA[idx0]) * np.log((Pv + A0) / dPA[idx0]), np.sqrt(wEA[idx0]) * np.log(Ev / dEA[idx0])])
r_start = r_law(p)
Lsum0 = float(np.sum(r_start ** 2))
t1 = time.time()
lsl = least_squares(r_law, p, x_scale='jac', max_nfev=60, xtol=1e-12, ftol=1e-13, gtol=1e-12)
c3['law_LP_plus_LE_at_reported'] = Lsum0
c3['law_LP_plus_LE_after_LS'] = float(2 * lsl.cost)
c3['law_decrease'] = float(Lsum0 - 2 * lsl.cost)
c3['law_max_abs_param_change'] = float(np.max(np.abs(lsl.x - p)))
c3['law_param_change'] = (lsl.x - p).tolist()
c3['law_LS_nfev'] = int(lsl.nfev); c3['law_LS_status'] = int(lsl.status); c3['law_LS_seconds'] = time.time() - t1
# gradient at the reported point (finite differences of the summed loss) and random perturbations
gnorm = float(np.linalg.norm(2 * lsl.jac.T @ lsl.fun)) if lsl.jac is not None else None
rng = np.random.default_rng(7)
worst = 0.0
for k in range(30):
    dp = rng.normal(size=p.size) * 1e-3
    Lk = float(np.sum(r_law(p + dp) ** 2))
    worst = min(worst, Lk - Lsum0)
c3['random_perturbations_1e-3_min_change'] = worst
c3['study_L_P_plus_L_E'] = JC[name]['L_P'] + JC[name]['L_E']
res['check3'] = c3
print('check3', json.dumps(c3, indent=1), flush=True)

# ---------------------------------------------------------------- check 4: parameter counts
exp_counts = {'existing_B6': 22, 'existing_B4': 22, 'existing_A6': 21, 'existing_A4': 21, 'existing_MB4': 28,
              'joint_T7x_rout6': 19, 'joint_T7x_rout4': 19, 'sep_T7x_rout6': 19, 'sep_T7x_rout4': 19,
              'joint_T7_rout6': 19, 'joint_T7_rout4': 19, 'sep_T7_rout6': 19, 'sep_T7_rout4': 19,
              'joint_H6_xout6_rout6': 18, 'joint_H6_xout6_rout4': 18}
c4 = {}
for nm, ex in exp_counts.items():
    js = JC[nm]; cn = js['counts']
    npar = (9 + len(js['law']) if 'law' in js else len(js['params'])) + len(js['amp'])
    c4[nm] = dict(total_with_electrons=cn['total_with_electrons'], expected=ex,
                  sum_of_parts=int(sum(v for k, v in cn.items() if k != 'total_with_electrons')),
                  n_saved_parameters_plus_electron=npar, ok=bool(cn['total_with_electrons'] == ex == npar))
res['check4'] = c4
print('check4', json.dumps(c4, indent=1), flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(S3 + 'out/verify_step3_results.json', 'w'), indent=1, default=float)
print('done', res['seconds'])
