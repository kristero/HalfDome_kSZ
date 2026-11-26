"""Independent verification of the BE study (electron pressure from an electron-density law + temperature model).

Written from the be_fit.py / be_sums.py module docstrings only (no functions copied from the study).
Reads the study's outputs read-only; writes only to verify_independent/out.

Symbols (as in the specification):
  x = r/R200c, M = M200c [Msun], z redshift, lm = ln(M/1e14 Msun), lz = ln(1+z).
  Each law quantity Q: ln Q = lnA_Q + am_Q lm + az_Q lz.
  P = P_e/P200c, E = n_e mu_XGP/(rho_crit f_b), tau = T_e/T200, P = E tau / C_T.
Checks 1-6 of the verification brief; results to out/verify_results.json.
"""
import json, sys, time
import numpy as np
from numpy.polynomial.legendre import leggauss
import scipy.constants as sc
from scipy import integrate

t0 = time.time()
ROOT = '/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126/'
OUT = ROOT + 'out/'
VER = ROOT + 'verify_independent/'
BDJ = '/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118/results/flamingo_residuals.json'
BDN = '/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118/results/flamingo_residuals.npz'
res = {}

def rel(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.abs(a / b - 1.0)

# ---------------------------------------------------------------- check 1: constants
m_e, m_p = sc.m_e, sc.m_p
X = 0.76
mu_xgp = (m_e + 2.0 * m_p / 1.76) / 0.9 / m_p          # in units of m_p
C_T = 200.0 * mu_xgp / 0.59                           # 200 mu_XGP / (0.59 m_p)
K_hse = 0.01 * (2.0 / 1.76) / mu_xgp                  # 0.01 rho_gas/(E rho_crit f_b), rho_gas = n_e 2 m_p/1.76
K_hse_with_me = 0.01 * (m_e / m_p + 2.0 / 1.76) / mu_xgp   # if rho_gas includes the electron mass (= 0.009)
PE = (2 + 2 * X) / (3 + 5 * X)
J1 = json.load(open(OUT + 'be_fits.json'))
cst = J1['constants']
res['check1'] = dict(
    mu_xgp=mu_xgp, C_T=C_T, K_HSE=K_hse, K_HSE_if_rho_gas_includes_m_e=K_hse_with_me, PE_FRAC=PE,
    study=cst,
    rel_mu=float(rel(mu_xgp, cst['MU_XGP'])), rel_C_T=float(rel(C_T, cst['C_T'])),
    rel_K=float(rel(K_hse, cst['K_HSE'])), rel_PE=float(rel(PE, cst['PE_FRAC'])),
    rel_K_me_convention=float(rel(K_hse_with_me, K_hse)))

# numerical check of P = E tau / C_T in SI with arbitrary physical inputs (independent of the algebra)
G = sc.G; kB = sc.k
Msun = 1.98847e30; Mpc = 3.0856775814913673e22
M_t = 3e14 * Msun; rho_c = 9.2e-27; f_b = 0.157
R200 = (3 * M_t / (4 * np.pi * 200 * rho_c)) ** (1 / 3)
n_e = 1.3e3; T_e = 5.0e7                              # m^-3, K
P200c = G * M_t * 200 * rho_c * f_b / (2 * R200)
T200 = 0.59 * m_p * G * M_t / (2 * R200) / kB
Pn = n_e * kB * T_e / P200c
En = n_e * mu_xgp * m_p / (rho_c * f_b)
taun = T_e / T200
res['check1']['P_eq_Etau_over_CT_rel'] = float(rel(Pn, En * taun / C_T))
# numerical check of dP_tot/dx: dP/dr = -G M(<r) rho_gas / r^2, P/P200c, x = r/R200c, rho_gas = g E rho_crit f_b
x_t, m_t, g_t, E_t = 0.7, 0.8, 0.9, 50.0
rho_gas = g_t * E_t * rho_c * f_b
dPdr = -G * (m_t * M_t) * rho_gas / (x_t * R200) ** 2
dPdx_scaled = dPdr * R200 / P200c
res['check1']['dPdx_rel'] = float(rel(dPdx_scaled, -0.01 * g_t * E_t * m_t / x_t ** 2))
res['check1']['g_rho_gas_over_E'] = (2.0 / 1.76) / mu_xgp
print('check1', json.dumps(res['check1'], indent=1))

# ---------------------------------------------------------------- rows, blocks
Z = np.load(OUT + 'be_fits_rows.npz')
Z2 = np.load(OUT + 'be_fits2_rows.npz')
BJ = json.load(open(BDJ))
BN = np.load(BDN)
blocks = BJ['blocks']
r0 = np.where(Z['region'] == 0)[0]
blk = Z['block'][r0]; lo = Z['lo'][r0]; hi = Z['hi'][r0]; kind = Z['kind'][r0]
node = np.array([blocks[b]['node'] for b in blk]); zb = np.array([blocks[b]['z'] for b in blk])
dP = Z['d_P'][r0]; dE = Z['d_E'][r0]; wP = Z['wP'][r0]; wE = Z['wE'][r0]
rowchk = dict(
    n_region0=int(len(r0)),
    M_vs_10node_maxrel=float(np.max(rel(Z['M'][r0], 10 ** node))),
    z_vs_block_maxabs=float(np.max(np.abs(Z['z'][r0] - zb))),
    block_equal_BD=bool(np.all(blk == BN['B32__block'])),
    lo_equal_BD=float(np.max(np.abs(lo - BN['B32__lo']))), hi_equal_BD=float(np.max(np.abs(hi - BN['B32__hi']))),
    kind_equal_BD=bool(np.all(kind == BN['B32__kind'])),
    wP_vs_BD_weight_maxrel=float(np.max(rel(wP, BN['B32__weight']))),
    dP_vs_BD_target_maxrel=float(np.max(rel(dP, BN['B32__target_flamingo_Y']))),
    sum_wP=float(wP.sum()), sum_wE=float(wE.sum()))
res['rows'] = rowchk
print('rows', rowchk)

# ---------------------------------------------------------------- model building blocks
h = 0.681
AH = BJ['alpha_held']   # [a0, a1] of alpha(z) = a0 + a1 z of the record law

def q3(c, lm, lz):
    return np.exp(c[0] + c[1] * lm + c[2] * lz)

def E_law(x, pe, lm, lz):
    n0 = q3(pe[0:3], lm, lz); ae = q3(pe[3:6], lm, lz); be = q3(pe[6:9], lm, lz)
    y = x / 0.5
    return n0 * np.exp(-0.2 * np.log(y) - (be - 0.2) / ae * np.log1p(y ** ae))

def P_law(x, pp, lm, lz, z, alpha=None):
    P0 = q3(pp[0:3], lm, lz); xc = q3(pp[3:6], lm, lz); beta = q3(pp[6:9], lm, lz)
    a = (AH[0] + AH[1] * z) if alpha is None else alpha
    q = x / xc
    return PE * P0 * np.exp(-0.3 * np.log(q) - beta * np.log1p(q ** a))

def T7_tau(x, pt, lm, lz):
    lnT0, amT, azT, lnxt, c, c0, bm = pt
    xt = np.exp(lnxt); xcc = np.exp(c0 + bm * lm)
    s = (x / xcc) ** 2
    return np.exp(lnT0 + amT * lm + azT * lz - c * np.log1p(x / xt)) * s / (1 + s)

def nfw_m(x, c):
    y = c * x
    f = np.where(y < 1e-3, y * y / 2 - 2 * y ** 3 / 3 + 3 * y ** 4 / 4, np.log1p(y) - y / (1 + y))
    return f / (np.log1p(c) - c / (1 + c))

def duffy(M, z):
    return 5.71 * (M * h / 2e12) ** (-0.084) * (1 + z) ** (-0.47)

GLn, GLw = leggauss(12)

class Ptot:
    """P_tot(x)/P200c = K int_x^xout E(s) m(s)/s^2 ds by panel Gauss-Legendre in u = ln s (exact panel-edge cumulants)."""
    def __init__(self, pe, M, z, xout, K=K_hse, npan=4000, umin=np.log(1e-13)):
        self.pe, self.lm, self.lz, self.xout, self.K = pe, np.log(M / 1e14), np.log1p(z), xout, K
        self.c = duffy(M, z)
        self.ue = np.linspace(umin, np.log(xout), npan + 1)
        a, b = self.ue[:-1], self.ue[1:]
        u = 0.5 * (b - a)[:, None] * GLn[None, :] + 0.5 * (a + b)[:, None]
        I = (0.5 * (b - a))[:, None] * GLw[None, :] * self.g(np.exp(u))
        Ip = I.sum(1)
        self.top = np.concatenate([np.cumsum(Ip[::-1])[::-1], [0.0]])   # top[k] = int_{ue[k]}^{u_out}
    def g(self, s):       # integrand in u: E m / s^2 * s
        return E_law(s, self.pe, self.lm, self.lz) * nfw_m(s, self.c) / s
    def __call__(self, x):
        x = np.asarray(x, float)
        u = np.log(np.minimum(x, self.xout))
        k = np.clip(np.searchsorted(self.ue, u, side='right') - 1, 0, len(self.ue) - 2)
        b = self.ue[k + 1]
        uu = 0.5 * (b - u)[..., None] * GLn + 0.5 * (b + u)[..., None]
        part = (0.5 * (b - u))[..., None] * GLw * self.g(np.exp(uu))
        val = self.K * (part.sum(-1) + self.top[k + 1])
        return np.where(x >= self.xout, 0.0, val)

def H_P(x, pe, ph, M, z, xout, pt_obj=None, model='H6'):
    lm, lz = np.log(M / 1e14), np.log1p(z)
    a, b, n = ph[0:3]
    fnth = np.minimum(a * (1 + z) ** b * x ** n, 0.95)
    P = PE * (1 - fnth) * pt_obj(x)
    if model == 'H6':
        c0, bm, bz = ph[3:6]
        xcc = np.exp(c0 + bm * lm + bz * lz)
        s = (x / xcc) ** 2
        P = P * s / (1 + s)
    return P

# integration of a model function over a row
NQ, WQ = leggauss(48)

def row_integral(f, lo_, hi_, knd, kinks=()):
    """shell: volume mean over [lo, hi]; core/group: 4 pi int_0^hi x^2 f dx. GL in u = ln x, split at kinks."""
    if knd == 'shell':
        a, b = lo_, hi_
    else:
        a, b = hi_ * 1e-9, hi_
    pts = [a] + sorted(k for k in kinks if a < k < b) + [b]
    tot = 0.0
    for i in range(len(pts) - 1):
        ua, ub = np.log(pts[i]), np.log(pts[i + 1])
        if knd != 'shell':
            # composite panels over many decades for the integrals
            edges = np.linspace(ua, ub, max(2, int(np.ceil((ub - ua) / 0.5))) + 1)
        else:
            edges = np.array([ua, ub])
        A, B = edges[:-1], edges[1:]
        u = 0.5 * (B - A)[:, None] * NQ + 0.5 * (A + B)[:, None]
        xx = np.exp(u)
        tot += np.sum((0.5 * (B - A))[:, None] * WQ * xx ** 3 * f(xx))
    if knd == 'shell':
        return tot / ((hi_ ** 3 - lo_ ** 3) / 3)
    return 4 * np.pi * tot

def model_rows(name, params, kindm, xout=4.0, rows=None, Kval=K_hse):
    """Return (P, E) at every region-0 row for a model."""
    p = np.asarray(params, float)
    idx = np.arange(len(r0)) if rows is None else rows
    Pv = np.full(len(idx), np.nan); Ev = np.full(len(idx), np.nan)
    cache = {}
    for j, i in enumerate(idx):
        M = 10 ** node[i]; z = zb[i]; lm = np.log(M / 1e14); lz = np.log1p(z)
        if kindm == 'plaw':
            fP = lambda x: P_law(x, p, lm, lz, z)
            fE = None
            kinks = ()
        else:
            pe = p[0:9]
            fE = lambda x: E_law(x, pe, lm, lz)
            if kindm == 'T7':
                pt = p[9:16]
                fP = lambda x: E_law(x, pe, lm, lz) * T7_tau(x, pt, lm, lz) / C_T
                kinks = ()
            elif kindm in ('H6', 'H3'):
                key = blk[i]
                if key not in cache:
                    cache[key] = Ptot(pe, M, z, xout, K=Kval)
                pto = cache[key]
                ph = p[9:]
                fP = lambda x: H_P(x, pe, ph, M, z, xout, pto, kindm)
                a, b, n = ph[0:3]
                kinks = ((0.95 / (a * (1 + z) ** b)) ** (1 / n),) if (n > 0 and a > 0) else ()
        Pv[j] = row_integral(fP, lo[i], hi[i], kind[i], kinks)
        if fE is not None:
            Ev[j] = row_integral(fE, lo[i], hi[i], kind[i])
    return Pv, Ev

def losses(Pv, Ev, idx=None):
    s = slice(None) if idx is None else idx
    LP = float(np.sum(wP[s] * np.log(Pv / dP[s]) ** 2))
    LE = float(np.sum(wE[s] * np.log(Ev / dE[s]) ** 2)) if np.all(np.isfinite(Ev)) else None
    return LP, LE

# ---------------------------------------------------------------- check 2 + 3: model values and losses
J2 = json.load(open(OUT + 'be_fits2.json'))
models = [('plaw_held', J1, Z), ('electron_law', J1, Z), ('joint_T7', J1, Z), ('sep_H6', J1, Z), ('joint_H6', J1, Z),
          ('joint_H6_xout6', J2, Z2)]
res['check2'] = {}
modelvals = {}
for name, J, ZZ in models:
    r = J['results'][name]
    kindm = r.get('kind', 'electron')
    if name == 'electron_law':
        kindm = 'electron'
    xout = r.get('info', {}).get('xout', 4.0) if isinstance(r.get('info'), dict) else 4.0
    p = r['params']
    t1 = time.time()
    if kindm == 'electron':
        pe = np.asarray(p, float)
        Ev = np.array([row_integral(lambda x, i=i: E_law(x, pe, np.log(10 ** node[i] / 1e14), np.log1p(zb[i])),
                                    lo[i], hi[i], kind[i]) for i in range(len(r0))])
        Pv = None
        LP = None; LE = float(np.sum(wE * np.log(Ev / dE) ** 2))
    else:
        Pv, Ev = model_rows(name, p, kindm, xout)
        LP, LE = losses(Pv, Ev)
    d = dict(kind=kindm, xout=xout, seconds=time.time() - t1, L_P_mine=LP, L_E_mine=LE,
             L_P_study=r.get('L_P'), L_E_study=r.get('L_E'))
    if LP is not None:
        d['L_P_rel'] = float(rel(LP, r['L_P']))
    if LE is not None and r.get('L_E') is not None:
        d['L_E_rel'] = float(rel(LE, r['L_E']))
    # compare to study row values
    if name == 'electron_law':
        sP = None; sE = ZZ['E__sep'][r0]
    else:
        sP = ZZ['P__' + name][r0]
        sE = ZZ['E__' + name][r0] if ('E__' + name) in ZZ.files else None
    if sP is not None and Pv is not None:
        rr = rel(Pv, sP)
        d['P_rows_maxrel'] = float(rr.max()); d['P_rows_medrel'] = float(np.median(rr))
        d['P_rows_worst'] = dict(i=int(rr.argmax()), kind=str(kind[rr.argmax()]), lo=float(lo[rr.argmax()]),
                                 hi=float(hi[rr.argmax()]), mine=float(Pv[rr.argmax()]), study=float(sP[rr.argmax()]))
        for kk in ('shell', 'core', 'group'):
            s = kind == kk
            d['P_rows_maxrel_' + kk] = float(rr[s].max())
    if sE is not None and Ev is not None and np.all(np.isfinite(Ev)):
        rr = rel(Ev, sE)
        d['E_rows_maxrel'] = float(rr.max()); d['E_rows_medrel'] = float(np.median(rr))
        d['E_rows_worst_kind'] = str(kind[rr.argmax()])
    modelvals[name] = (Pv, Ev)
    res['check2'][name] = d
    print(name, json.dumps(d, indent=1), flush=True)

# check 3: record pressure law, also with the BD coefficients and the docstring-rounded alpha
pp_bd = np.asarray(BJ['lsq_fixed_alpha']['coef'])
Pbd = np.array([row_integral(lambda x, i=i: P_law(x, pp_bd, np.log(10 ** node[i] / 1e14), np.log1p(zb[i]), zb[i]),
                             lo[i], hi[i], kind[i]) for i in range(len(r0))])
pp_st = np.asarray(J1['results']['plaw_held']['params'])
Prd = np.array([row_integral(lambda x, i=i: P_law(x, pp_st, np.log(10 ** node[i] / 1e14), np.log1p(zb[i]), zb[i],
                                                   alpha=0.9394 + 0.2948 * zb[i]), lo[i], hi[i], kind[i]) for i in range(len(r0))])
res['check3'] = dict(L_P_plaw_held_mine=res['check2']['plaw_held']['L_P_mine'],
                     L_P_plaw_held_study=J1['results']['plaw_held']['L_P'],
                     L_P_BDcoef_mine=float(np.sum(wP * np.log(Pbd / dP) ** 2)),
                     L_P_BD_record=BJ['lsq_fixed_alpha']['loss'],
                     L_P_plaw_held_rounded_alpha=float(np.sum(wP * np.log(Prd / dP) ** 2)),
                     alpha_held=AH)
# local-minimum check of plaw_held with my own least squares (from the study's point and from the BD point)
from scipy.optimize import least_squares
sw = np.sqrt(wP)
def resid_plaw(pp):
    Pv = np.array([row_integral(lambda x, i=i: P_law(x, pp, np.log(10 ** node[i] / 1e14), np.log1p(zb[i]), zb[i]),
                                lo[i], hi[i], kind[i]) for i in range(len(r0))])
    return sw * np.log(Pv / dP)
t1 = time.time()
ls = least_squares(resid_plaw, pp_st, x_scale='jac', xtol=1e-12, ftol=1e-14, gtol=1e-12)
res['check3']['refit_from_study'] = dict(loss=float(2 * ls.cost), params=ls.x.tolist(),
                                          max_abs_param_shift=float(np.max(np.abs(ls.x - pp_st))),
                                          nfev=int(ls.nfev), seconds=time.time() - t1)
print('check3', json.dumps(res['check3'], indent=1), flush=True)

# ---------------------------------------------------------------- quad cross-check of the quadrature (independent method)
def quad_row(f, lo_, hi_, knd, kinks=()):
    if knd == 'shell':
        pts = [k for k in kinks if lo_ < k < hi_]
        v, _ = integrate.quad(lambda x: x * x * f(x), lo_, hi_, points=pts or None, epsabs=0, epsrel=1e-11, limit=200)
        return v / ((hi_ ** 3 - lo_ ** 3) / 3)
    pts = [k for k in kinks if 0 < k < hi_]
    v, _ = integrate.quad(lambda x: x * x * f(x), 0, hi_, points=pts or None, epsabs=0, epsrel=1e-11, limit=400)
    return 4 * np.pi * v

def quad_Ptot(pe, M, z, xout, x):
    lm, lz = np.log(M / 1e14), np.log1p(z); c = duffy(M, z)
    if x >= xout:
        return 0.0
    v, _ = integrate.quad(lambda s: E_law(s, pe, lm, lz) * nfw_m(np.array(s), c) / s ** 2, x, xout,
                          epsabs=0, epsrel=1e-12, limit=400)
    return K_hse * v

rng = np.random.default_rng(1)
pick = np.concatenate([rng.choice(np.where(kind == 'shell')[0], 6, replace=False),
                       rng.choice(np.where(kind == 'core')[0], 2, replace=False),
                       rng.choice(np.where(kind == 'group')[0], 2, replace=False)])
qres = []
pH = np.asarray(J1['results']['joint_H6']['params']); pT = np.asarray(J1['results']['joint_T7']['params'])
for i in pick:
    M = 10 ** node[i]; z = zb[i]; lm = np.log(M / 1e14); lz = np.log1p(z)
    pe = pH[:9]; ph = pH[9:]
    def fH(x):
        a, b, n, c0, bm, bz = ph
        fn = min(a * (1 + z) ** b * x ** n, 0.95)
        xcc = np.exp(c0 + bm * lm + bz * lz); s = (x / xcc) ** 2
        return PE * (1 - fn) * quad_Ptot(pe, M, z, 4.0, x) * s / (1 + s)
    kk = ((0.95 / (ph[0] * (1 + z) ** ph[1])) ** (1 / ph[2]),)
    vq = quad_row(fH, lo[i], hi[i], kind[i], kk)
    peT = pT[:9]; ptT = pT[9:]
    vT = quad_row(lambda x: E_law(x, peT, lm, lz) * T7_tau(x, ptT, lm, lz) / C_T, lo[i], hi[i], kind[i])
    qres.append(dict(i=int(i), kind=str(kind[i]), quad_H6=vq, gl_H6=float(modelvals['joint_H6'][0][i]),
                     study_H6=float(Z['P__joint_H6'][r0][i]), quad_T7=vT, gl_T7=float(modelvals['joint_T7'][0][i]),
                     study_T7=float(Z['P__joint_T7'][r0][i])))
res['quad_crosscheck'] = dict(rows=qres,
                              max_rel_quad_vs_gl_H6=float(max(abs(q['quad_H6'] / q['gl_H6'] - 1) for q in qres)),
                              max_rel_quad_vs_gl_T7=float(max(abs(q['quad_T7'] / q['gl_T7'] - 1) for q in qres)),
                              max_rel_quad_vs_study_H6=float(max(abs(q['quad_H6'] / q['study_H6'] - 1) for q in qres)),
                              max_rel_quad_vs_study_T7=float(max(abs(q['quad_T7'] / q['study_T7'] - 1) for q in qres)))
print('quad', json.dumps({k: v for k, v in res['quad_crosscheck'].items() if k != 'rows'}, indent=1), flush=True)

# ---------------------------------------------------------------- Y200 (joint_H6 and joint_H6_xout6)
S = np.load(OUT + 'be_sums.npz')
lab_of = {}
for bi, b in enumerate(blocks):
    lab_of[(b['snapshot'], b['label'])] = bi

def stack_mean(snap, label, F, q, weight='S'):
    key = '%d__%s__' % (snap, label)
    if weight == 'S':
        return S[key + 'S_%s_%s' % (F, q)].sum(0) / S[key + 'W'].sum()
    return S[key + 'U_%s_%s' % (F, q)].sum(0) / S[key + 'N'].sum()

res['Y200'] = {}
for name, J in (('joint_H6', J1), ('joint_H6_xout6', J2)):
    r = J['results'][name]; p = np.asarray(r['params']); xout = r['info']['xout']
    out = {}
    for key, v in r['Y200'].items():
        bi = int(np.argmin([abs(b['node'] - v['M']) + 10 * abs(b['z'] - v['z']) for b in blocks]))
        b = blocks[bi]; M = 10 ** b['node']; z = b['z']
        e = S['%d__edges' % b['snapshot']]
        Pc = stack_mean(b['snapshot'], b['label'], 'P', 'all')
        V1 = (np.minimum(e[1:], 1.0) ** 3 - np.minimum(e[:-1], 1.0) ** 3) / 3
        Ydata = 4 * np.pi * np.sum(Pc * V1)
        pto = Ptot(p[:9], M, z, xout)
        ph = p[9:]
        kk = ((0.95 / (ph[0] * (1 + z) ** ph[1])) ** (1 / ph[2]),)
        Ymod = row_integral(lambda x: H_P(x, p[:9], ph, M, z, xout, pto, 'H6'), 0, 1.0, 'core', kk)
        out[key] = dict(block=b['label'], snap=b['snapshot'], mine=Ymod / Ydata, study=v['Y200_model_over_data'],
                        rel=float(rel(Ymod / Ydata, v['Y200_model_over_data'])), Ydata=Ydata, Ymodel=Ymod)
    res['Y200'][name] = out
    print('Y200', name, json.dumps(out, indent=1), flush=True)

# ---------------------------------------------------------------- check 4: data identities
def cell_overlap_mean(e, F, a, b, knd):
    if knd == 'shell':
        ov = (np.clip(e[1:], a, b) ** 3 - np.clip(e[:-1], a, b) ** 3) / 3
        return np.sum(F * ov) / ((b ** 3 - a ** 3) / 3)
    ov = (np.minimum(e[1:], b) ** 3 - np.minimum(e[:-1], b) ** 3) / 3
    return 4 * np.pi * np.sum(F * ov)

dE_re = np.zeros(len(r0)); dP_re = np.zeros(len(r0)); dE_a2 = np.zeros(len(r0)); dP_a2 = np.zeros(len(r0))
for i in range(len(r0)):
    b = blocks[blk[i]]; e = S['%d__edges' % b['snapshot']]
    dE_re[i] = cell_overlap_mean(e, stack_mean(b['snapshot'], b['label'], 'E', 'all'), lo[i], hi[i], kind[i])
    dP_re[i] = cell_overlap_mean(e, stack_mean(b['snapshot'], b['label'], 'P', 'all'), lo[i], hi[i], kind[i])
    dE_a2[i] = cell_overlap_mean(e, stack_mean(b['snapshot'], b['label'], 'E', 'a2ii'), lo[i], hi[i], kind[i])
    dP_a2[i] = cell_overlap_mean(e, stack_mean(b['snapshot'], b['label'], 'P', 'a2ii'), lo[i], hi[i], kind[i])
# are row edges on storage edges?
on_edges = []
for i in range(len(r0)):
    e = S['%d__edges' % blocks[blk[i]]['snapshot']]
    ok_hi = np.min(np.abs(e - hi[i])) < 1e-12
    ok_lo = (kind[i] != 'shell') or (np.min(np.abs(e - lo[i])) < 1e-12)
    on_edges.append(ok_hi and ok_lo)
# electron weights from the documented formula
fE = S['resid_E_frac_scatter']; prio = BN['B32__prio1']
sE = np.maximum(fE, 0.05)
wE_re = 2701.0 * (prio / sE ** 2) / np.sum(prio / sE ** 2)
res['check4_identities'] = dict(
    dE_vs_volmean_Sall_W_maxrel=float(np.max(rel(dE_re, dE))),
    dP_vs_volmean_Sall_W_maxrel=float(np.max(rel(dP_re, dP))),
    dE_all_vs_a2ii_maxrel=float(np.max(rel(dE_a2, dE_re))), dP_all_vs_a2ii_maxrel=float(np.max(rel(dP_a2, dP_re))),
    rows_on_storage_edges=int(np.sum(on_edges)),
    dE_vs_resid_E_mean_maxrel=float(np.max(rel(S['resid_E_mean'], dE))),
    wE_recomputed_maxrel=float(np.max(rel(wE_re, wE))),
    n_sE_floored=int(np.sum(fE < 0.05)),
    region0_zero_or_negative_targets=int(np.sum((dE <= 0) | (dP <= 0))))
print('check4 identities', res['check4_identities'], flush=True)

# atlas comparison
A = np.load(VER + 'input/atlas_E.npz')
ea = A['edges70']; prof = A['prof_a2ii']; zat = A['redshift']; labs = A['entry_labels']; nat = A['n']
snap_of_z = {}
for b in blocks:
    snap_of_z[round(b['z'], 4)] = b['snapshot']
atl = dict(per_block=[], n_compared=0)
allrel = []
for io, zz in enumerate(zat):
    snap = snap_of_z.get(round(float(zz), 4))
    if snap is None:
        continue
    e = S['%d__edges' % snap]
    # atlas shells that are unions of storage cells
    for ie in range(16):
        lab = str(labs[ie])
        if (snap, lab) not in lab_of:
            continue
        U = stack_mean(snap, lab, 'E', 'a2ii', weight='U')
        Ncount = S['%d__%s__N' % (snap, lab)].sum(); nh = S['%d__%s__n' % (snap, lab)].sum()
        rr = []
        for j in range(len(ea) - 1):
            ia = np.where(np.abs(e - ea[j]) < 1e-9)[0]; ib = np.where(np.abs(e - ea[j + 1]) < 1e-9)[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            ia, ib = ia[0], ib[0]
            Vc = (e[ia + 1:ib + 1] ** 3 - e[ia:ib] ** 3) / 3
            v = np.sum(U[ia:ib] * Vc) / np.sum(Vc)
            if prof[io, ie, j] > 0:
                rr.append(v / prof[io, ie, j] - 1)
        rr = np.array(rr)
        allrel.extend(np.abs(rr).tolist())
        atl['per_block'].append(dict(z=float(zz), label=lab, n_shells=int(len(rr)), max_abs_rel=float(np.max(np.abs(rr))),
                                     median_rel=float(np.median(rr)), N_BE=float(Ncount), n_BE=int(nh),
                                     n_atlas=float(nat[io, ie])))
atl['n_compared'] = len(allrel)
atl['max_abs_rel'] = float(np.max(allrel)); atl['median_abs_rel'] = float(np.median(allrel))
atl['n_blocks'] = len(atl['per_block'])
atl['halo_count_mismatches'] = [d for d in atl['per_block'] if abs(d['n_BE'] - d['n_atlas']) > 0]
res['check4_atlas'] = atl
print('atlas', {k: v for k, v in atl.items() if k != 'per_block'}, flush=True)

# ---------------------------------------------------------------- check 5: held-out totals
H = json.load(open(OUT + 'be_heldout.json'))['results']
c5 = {}
for name, v in H.items():
    zs = [k for k in v if not k.startswith('total')]
    d = {}
    if any('L_P_test' in v[k] for k in zs):
        d['sum_L_P_test'] = float(sum(v[k]['L_P_test'] for k in zs))
        d['study_total_L_P_test'] = v.get('total_L_P_test')
    if any('L_E_test' in v[k] for k in zs):
        d['sum_L_E_test'] = float(sum(v[k]['L_E_test'] for k in zs))
        d['study_total_L_E_test'] = v.get('total_L_E_test')
    d['n_test_total'] = int(sum(v[k]['n_test'] for k in zs))
    d['n_outputs'] = len(zs)
    c5[name] = d
res['check5'] = c5
print('check5', json.dumps(c5, indent=1), flush=True)

# independent held-out refit of plaw_held (leave one output out), with my own least squares
zvals = np.unique(zb)
ho = {}
for zz in zvals:
    tr = np.where(zb != zz)[0]; te = np.where(zb == zz)[0]
    def rtr(pp):
        Pv = np.array([row_integral(lambda x, i=i: P_law(x, pp, np.log(10 ** node[i] / 1e14), np.log1p(zb[i]), zb[i]),
                                    lo[i], hi[i], kind[i]) for i in tr])
        return np.sqrt(wP[tr]) * np.log(Pv / dP[tr])
    lsz = least_squares(rtr, pp_st, x_scale='jac', xtol=1e-10, ftol=1e-12)
    Pte = np.array([row_integral(lambda x, i=i: P_law(x, lsz.x, np.log(10 ** node[i] / 1e14), np.log1p(zb[i]), zb[i]),
                                 lo[i], hi[i], kind[i]) for i in te])
    ho['%g' % zz] = dict(L_P_test=float(np.sum(wP[te] * np.log(Pte / dP[te]) ** 2)), n_test=int(len(te)),
                         w_test=float(wP[te].sum()), study=H['plaw_held']['%g' % zz]['L_P_test'])
ho_tot = float(sum(v['L_P_test'] for v in ho.values()))
res['check5_plaw_heldout_refit'] = dict(per_z=ho, total_mine=ho_tot, total_study=H['plaw_held']['total_L_P_test'],
                                        rel=float(rel(ho_tot, H['plaw_held']['total_L_P_test'])))
print('check5 refit', json.dumps(res['check5_plaw_heldout_refit'], indent=1), flush=True)

# ---------------------------------------------------------------- check 6: tau of the data
allr = np.arange(len(Z['region']))
zs_all = Z['z']; Ms = Z['M']; kd = Z['kind']
c6 = {}
tau_d = C_T * Z['d_P'] / Z['d_E']
c6['tau_column_vs_CT_dP_dE_maxrel'] = float(np.max(rel(tau_d[kd == 'shell'], Z['tau'][kd == 'shell'])))
for xr in (0.1, 0.5, 1.0):
    sel = (kd == 'shell') & (np.abs(zs_all) < 1e-9) & (Ms >= 1e14 * 0.999) & (Z['lo'] <= xr) & (Z['hi'] > xr)
    # one shell per block
    vals = []
    for b in np.unique(Z['block'][sel]):
        s = sel & (Z['block'] == b)
        ii = np.where(s)[0]
        ii = ii[np.argmin(Z['hi'][ii] - Z['lo'][ii])]
        vals.append((blocks[b]['label'], float(Z['lo'][ii]), float(Z['hi'][ii]), float(tau_d[ii]), int(Z['region'][ii])))
    t = np.array([v[3] for v in vals])
    c6['x%g' % xr] = dict(n_blocks=len(vals), min=float(t.min()), max=float(t.max()), per_block=vals)
res['check6'] = c6
print('check6', json.dumps(c6, indent=1), flush=True)

res['seconds'] = time.time() - t0
json.dump(res, open(VER + 'out/verify_results.json', 'w'), indent=1, default=float)
print('done', res['seconds'])
