"""BE step 2: electron pressure from the electron-density law and a temperature model (FLAMINGO L1_m9, BD residual set).

Question (user, 26 Nov 2026): can the pressure profiles be predicted from the density fit with a simple temperature profile
(physical, as Schneider et al. 2025 Sec. 6, or empirical), so that density + pressure need fewer than 9 + 9 law parameters
(+ the 2-halo amplitude)?

Data (read-only): the BD residual set via bd_mass.Data (BD_ROOT, BD_CODE): region 0 = the 1,494 residuals of the FLAMINGO B32
pressure fit of record (shells, core integrals, group integrals), region 1 = shells of the 4 R200c set beyond B32, region 2 = the
storage cells 4-4.5, 4.5-5, 5-6 R200c. Targets: Y200-weighted all-gas stacks. Electron targets on the same rows from be_sums.npz
(step 1), with the same Y200 weights, so that P/E of the stacks is a stack temperature.

Symbols. x = r/R200c; M = M200c; lm = ln(M / 1e14 Msun); lz = ln(1 + z).
  P  = P_e / P200c,  P200c = G M200c 200 rho_crit f_b / (2 R200c)  (total self-similar pressure scale, production convention)
  E  = n_e mu_XGP / (rho_crit f_b),  mu_XGP = (m_e + 2 m_p / 1.76) / 0.9  (production XGPaint convention)
  tau = T_e / T200,  T200 = 0.59 m_p G M200c / (2 R200c) / k_B;  then P = E tau / C_T,  C_T = 200 mu_XGP / (0.59 m_p) = 428.21
Laws (all coefficients of a quantity Q: ln Q = lnA_Q + am_Q lm + az_Q lz):
  pressure law of record  P = 0.5176 P0 q^-0.3 (1 + q^alpha)^-beta, q = x/x_c, alpha = 0.9394 + 0.2948 z ('held', 9 par) or one
                          constant (10 par)                                                    [bd_mass.Law]
  electron law (prod.)    E = n0 (x/0.5)^-0.2 [1 + (x/0.5)^a_e]^(-(beta_e - 0.2)/a_e); n0, a_e, beta_e  (9 par)
  temperature T5          ln tau = ln T0 + am lm + az lz - c ln(1 + x/x_t)                     (5 par: T0 x 3, x_t, c)
  temperature T7          T5 + ln[(x/x_cc)^2 / (1 + (x/x_cc)^2)],  ln x_cc = c0 + bm lm         (7 par; cool-core / multiphase dip)
  temperature T8          T7 with ln x_cc = c0 + bm lm + bz lz                                  (8 par)
  temperature T7x         T7 with the outer decline exp[-(x/x_e)^c_e] instead of (1 + x/x_t)^-c   (7 par; added 26 Nov)
  hydrostatic H3          P = 0.5176 (1 - f_nth) P_tot,  P_tot(x) = K int_x^x_out E(s) m(s) / s^2 ds,  K = 0.01 rho_gas/(E rho_crit f_b)
                          = 0.01 (2/1.76) m_p / mu_XGP; m(s) = M(<s R200c)/M200c of an NFW halo with Duffy+08 c200c(M, z)
                          (h = 0.681); P_tot(x_out) = 0, x_out = 4; f_nth = a (1+z)^b x^n, capped at 0.95  (3 par)
  hydrostatic H3c         H3 with c200c multiplied by a free constant                           (4 par)
  hydrostatic H6          H3 x [(x/x_cc)^2/(1 + (x/x_cc)^2)], ln x_cc = c0 + bm lm + bz lz       (6 par)
  0.5176 = (2 + 2 X)/(3 + 5 X), X = 0.76 (electron / total thermal pressure); the electron law is extrapolated beyond B32.
Losses (region 0 only): L_P = sum_i w_i [ln P_i(model) - ln d^P_i]^2, w = FLAMINGO best weights (mix_Y_lin_SO_90), sum w = W0 = 2701;
  L_E = sum_i w^E_i [ln E_i - ln d^E_i]^2, w^E_i = W0 (b_i / s_E,i^2) / sum (b / s_E^2), s_E,i = max(f^E_i, 0.05) (electron analogue of
  the record's scatter-only weights; b_i = priority weight 'prio1'). Model values are shell volume means (or 4 pi int x^2 dx for
  integrals) by 12-point Gauss-Legendre quadrature (bd_mass.Data xq, wq).
Fits: 'sep' = electron law from L_E alone, then the temperature parameters from L_P with the electron law fixed;
  'joint' = electron + temperature parameters from L_P + L_E. Reference: the pressure law alone from L_P (1-halo only).
  Least squares (scipy), several starts. Leave-one-output-out: fit without one output, loss on its region-0 rows.
  2-halo: bd_final split procedure, law from region 0 (+ L_E for joint), ln A_2h = a + b lz + c lz^2 from region 1 (to 4 R200c)
  or regions 1 + 2 (to 6 R200c), template Vikram+17 with exclusion 0.75 (bd_twohalo via bd_mass.Data.template('hm_ex0.75')).
Usage: be_fit.py --task fits|heldout|twohalo|all     (env BD_ROOT, BD_CODE, BE_SUMS, BE_OUT)
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, os.environ['BD_CODE'])
import numpy as np  # noqa: E402
from scipy.constants import electron_mass, proton_mass  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import bd_mass as BM  # noqa: E402

OUT = Path(os.environ['BE_OUT'])
SUMS = Path(os.environ['BE_SUMS'])
MU_XGP = (electron_mass+proton_mass*2/1.76)/.9/proton_mass
C_T = 200*MU_XGP/.59
K_HSE = .01*(2/1.76)/MU_XGP
PE_FRAC = (2+2*.76)/(3+5*.76)
H = .681
GRID = np.geomspace(1e-5, 6., 2400)
LNG = np.log(GRID)
GROUPS = BM.GROUPS
BANDS_IN = ((0., .3), (.3, .6), (.6, 1.), (1., 1.6))
BANDS_OUT = ((1.5, 2.), (2., 3.), (3., 4.01), (4.01, 6.01))
SIX = ((13., 0.), (14., 0.), (15., 0.), (13., 1.), (14., 1.), (13., 2.))


def operator(fine, lo, hi, shell):
    R = np.zeros((len(lo), len(fine)-1))
    for i in range(len(lo)):
        a = lo[i] if shell[i] else 0.
        v = np.maximum(np.minimum(fine[1:], hi[i])**3-np.maximum(fine[:-1], a)**3, 0.)
        R[i] = v/v.sum() if shell[i] else 4*np.pi/3*v
    return R


class BEData:
    """bd_mass.Data plus electron / host-bound targets on every row and the electron weights of region 0."""

    def __init__(self):
        D = BM.Data()
        self.D = D
        S = np.load(SUMS)
        blocks = D.cs.inner.meta['blocks']
        n = len(D.d)
        names = ['P_all', 'P_a2ii', 'P_hb', 'E_all', 'E_a2ii', 'E_hb']
        T = {k: np.zeros(n) for k in names+['E_err', 'P_err']}
        self.Y200 = {}
        for b in np.unique(D.block):
            m = blocks[b]
            key = f"{m['snapshot']}__{m['label']}"
            fine = S[f"{m['snapshot']}__edges"]
            W = S[f'{key}__W']
            rows = np.flatnonzero(D.block == b)
            R = operator(fine, D.lo[rows], D.hi[rows], D.shell[rows])
            for k in names:
                F, q = k.split('_')
                T[k][rows] = R @ (S[f'{key}__S_{F}_{q}'].sum(axis=0)/W.sum())
            for F in ('E', 'P'):
                folds = (S[f'{key}__S_{F}_all']/W[:, None]) @ R.T
                T[f'{F}_err'][rows] = folds.std(axis=0, ddof=1)/2.
            R1 = operator(fine, [0.], [1.], [False])
            self.Y200[int(b)] = float(R1 @ (S[f'{key}__S_P_all'].sum(axis=0)/W.sum()))
        self.T = T
        # identity checks against the BD targets (same weights, same operator)
        self.check = dict(P_all_vs_BD_max_rel=float(np.max(abs(T['P_all']/D.d-1))), P_a2ii_vs_BD_max_rel=float(np.max(abs(T['P_a2ii']/np.where(D.h > 0, D.h, 1.)-1)*(D.h > 0))))
        # electron weights (region 0 rows are the B32 residuals in file order)
        Z = np.load(Path(os.environ['BD_ROOT'])/'results/flamingo_residuals.npz')
        n0 = len(Z['B32__block'])
        assert np.array_equal(D.block[:n0], Z['B32__block']) and np.allclose(D.hi[:n0], Z['B32__hi']) and np.all(D.region[:n0] == 0)
        assert np.sum(D.region == 0) == n0
        self.n0 = n0
        b_i = Z['B32__prio1']
        fE, fP = S['resid_E_frac_scatter'], S['resid_P_frac_scatter']
        sE = np.maximum(fE, .05)
        wE = b_i/sE**2
        W0 = float(D.cs.inner.meta['constants']['W0'])
        self.wE = np.zeros(n)
        self.wE[:n0] = W0*wE/wE.sum()
        self.wP = D.inner_weight('best')
        sP_rec = Z['B32__s']
        src = Z['B32__s_source'].astype(str)
        direct = np.char.find(src, 'fallback') < 0
        if not direct.any():
            direct = np.ones(len(src), bool)
        sP_new = np.maximum(fP, .05)
        self.check.update(W0=W0, sum_wP=float(self.wP.sum()), sum_wE=float(self.wE.sum()),
                          s_P_recomputed_vs_record_median_rel=float(np.median(abs(sP_new[direct]/sP_rec[direct]-1))),
                          s_P_recomputed_vs_record_max_rel=float(np.max(abs(sP_new[direct]/sP_rec[direct]-1))),
                          s_P_sources=dict(zip(*[a.tolist() for a in np.unique(src, return_counts=True)])),
                          resid_mean_P_vs_target_max_rel=float(np.max(abs(S['resid_P_mean']/T['P_all'][:n0]-1))),
                          resid_mean_E_vs_target_max_rel=float(np.max(abs(S['resid_E_mean']/T['E_all'][:n0]-1))),
                          s_E_median=float(np.median(sE)), s_P_median=float(np.median(sP_rec)))
        self.lndE = np.log(T['E_all'])
        self.inner = np.flatnonzero(D.region == 0)


class ELaw:
    names = ['lnA_n0', 'am_n0', 'az_n0', 'lnA_ae', 'am_ae', 'az_ae', 'lnA_betae', 'am_betae', 'az_betae']
    lo = np.array([np.log(1e-2), -3., -4., np.log(.05), -2., -3., np.log(.3), -2., -3.])
    hi = np.array([np.log(1e6), 3., 4., np.log(20.), 2., 3., np.log(60.), 2., 3.])

    @staticmethod
    def point(p, x, lm, lz):
        n0, ae, be = (np.exp(p[3*k]+p[3*k+1]*lm+p[3*k+2]*lz) for k in range(3))
        q = x/.5
        return n0*q**-.2*(1+q**ae)**(-(be-.2)/ae)


TNAMES = {'T5': ['lnT0', 'am_T', 'az_T', 'ln_xt', 'c_out'],
          'T7': ['lnT0', 'am_T', 'az_T', 'ln_xt', 'c_out', 'lnxcc0', 'bm_cc'],
          'T8': ['lnT0', 'am_T', 'az_T', 'ln_xt', 'c_out', 'lnxcc0', 'bm_cc', 'bz_cc'],
          'T7x': ['lnT0', 'am_T', 'az_T', 'ln_xe', 'c_e', 'lnxcc0', 'bm_cc'],
          'H3': ['a_nth', 'b_nth', 'n_nth'], 'H3c': ['a_nth', 'b_nth', 'n_nth', 'ln_cnorm'],
          'H6': ['a_nth', 'b_nth', 'n_nth', 'lnxcc0', 'bm_cc', 'bz_cc'],
          'H3e': ['a_nth', 'b_nth', 'n_nth'], 'H6e': ['a_nth', 'b_nth', 'n_nth', 'lnxcc0', 'bm_cc', 'bz_cc']}
TRANGE = {'lnT0': (np.log(.05), np.log(50.), np.log(2.5)), 'am_T': (-2., 2., -.15), 'az_T': (-3., 3., -.3),
          'ln_xt': (np.log(.01), np.log(50.), np.log(.5)), 'c_out': (0., 10., 1.3),
          'lnxcc0': (np.log(1e-4), np.log(5.), np.log(.05)), 'bm_cc': (-4., 4., -.7), 'bz_cc': (-4., 4., .2),
          'ln_xe': (np.log(.05), np.log(20.), np.log(.8)), 'c_e': (.2, 6., 1.),
          'a_nth': (0., 2., .15), 'b_nth': (-3., 3., .3), 'n_nth': (0., 4., .9), 'ln_cnorm': (np.log(.2), np.log(5.), 0.)}


def cool_core(x, lxcc):
    u = 2.*(np.log(x)-lxcc)
    return np.exp(u-np.logaddexp(0., u))


def duffy(M, z):
    return 5.71*(M*H/2e12)**-.084*(1+z)**-.47


class Composite:
    """Pressure model for every row; parameters p = [electron 9] + [temperature / hydrostatic k]."""

    def __init__(self, B, kind, xout=4.):
        self.B, self.kind, self.xout = B, kind, xout
        D = B.D
        self.tn = TNAMES[kind]
        self.names = ELaw.names+self.tn
        self.lo = np.r_[ELaw.lo, [TRANGE[n][0] for n in self.tn]]
        self.hi = np.r_[ELaw.hi, [TRANGE[n][1] for n in self.tn]]
        self.tstart = np.array([TRANGE[n][2] for n in self.tn])
        ub = np.unique(D.block)
        self.ub = ub
        self.bpos = np.searchsorted(ub, D.block)
        Mb = np.array([D.M[D.block == b][0] for b in ub])
        zb = np.array([D.z[D.block == b][0] for b in ub])
        self.lmb, self.lzb, self.zb, self.Mb = np.log(Mb/1e14), np.log1p(zb), zb, Mb
        self.cb = duffy(Mb, zb)
        g = GRID[GRID <= xout]
        self.g = np.r_[g, xout] if g[-1] < xout else g

    def electrons(self, pE, rows=slice(None)):
        D = self.B.D
        return np.sum(D.wq[rows]*ELaw.point(pE, D.xq[rows], D.lm[rows][:, None], D.lz[rows][:, None]), axis=1)

    def p_tot_nodes(self, pE, x, bpos, cnorm=1.):
        """P_tot / P200c at nodes x (rows x nodes) of blocks bpos, hydrostatic integral of the electron law to x_out."""
        g = self.g
        lmb, lzb = self.lmb[:, None], self.lzb[:, None]
        c = (self.cb*cnorm)[:, None]
        mu = lambda y: np.log1p(y)-y/(1+y)  # noqa: E731
        m = mu(c*g[None, :])/mu(c)
        f = K_HSE*ELaw.point(pE, g[None, :], lmb, lzb)*m/g[None, :]**2*g[None, :]            # integrand x d ln x
        lg = np.log(g)
        seg = .5*(f[:, 1:]+f[:, :-1])*np.diff(lg)[None, :]
        I = np.c_[np.cumsum(seg[:, ::-1], axis=1)[:, ::-1], np.zeros(len(lmb))]             # int_g^x_out
        lx = np.log(np.minimum(x, g[-1]))
        j = np.clip(np.searchsorted(lg, lx)-1, 0, len(g)-2)
        t = (lx-lg[j])/(lg[j+1]-lg[j])
        bp = np.broadcast_to(bpos[:, None], x.shape)
        val = I[bp, j]*(1-t)+I[bp, j+1]*t
        return np.where(x >= g[-1], 0., val)

    def pressure_nodes(self, p, rows=slice(None), x=None, bpos=None):
        D = self.B.D
        pE, pT = p[:9], p[9:]
        if x is None:
            x = D.xq[rows]
            bpos = self.bpos[rows]
        lm, lz, z = self.lmb[bpos][:, None], self.lzb[bpos][:, None], self.zb[bpos][:, None]
        k = self.kind
        if k.startswith('T'):
            if k == 'T7x':
                lt = pT[0]+pT[1]*lm+pT[2]*lz-(x/np.exp(pT[3]))**pT[4]
            else:
                lt = pT[0]+pT[1]*lm+pT[2]*lz-pT[4]*np.log1p(x/np.exp(pT[3]))
            P = ELaw.point(pE, x, lm, lz)*np.exp(lt)/C_T
            if k in ('T7', 'T8', 'T7x'):
                lx = pT[5]+pT[6]*lm+(pT[7]*lz if k == 'T8' else 0.)
                P = P*cool_core(x, lx)
            return P
        cn = np.exp(pT[3]) if k == 'H3c' else 1.
        Pt = self.p_tot_nodes(pE, x, bpos, cn)
        if k.endswith('e'):
            fn = 1-np.exp(-pT[0]*(1+z)**pT[1]*x**pT[2])
        else:
            fn = np.clip(pT[0]*(1+z)**pT[1]*x**pT[2], 0., .95)
        P = PE_FRAC*(1-fn)*Pt
        if k in ('H6', 'H6e'):
            P = P*cool_core(x, pT[3]+pT[4]*lm+pT[5]*lz)
        return P

    def __call__(self, p, rows=slice(None)):
        D = self.B.D
        return np.sum(D.wq[rows]*self.pressure_nodes(p, rows), axis=1)


def lsq(fun, x0, lo, hi, nfev=3000):
    x0 = np.clip(np.asarray(x0, float), lo+1e-7, hi-1e-7)
    return least_squares(fun, x0, bounds=(lo, hi), x_scale='jac', max_nfev=nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)


def safe_log_res(m, lnd, sw):
    if np.any(~np.isfinite(m)) or np.any(m <= 0):
        return np.full(len(lnd), 1e3)
    return sw*(np.log(m)-lnd)


class Fitter:
    def __init__(self, B):
        self.B = B
        self.D = B.D
        self.estarts = None

    # ---------- electron law alone
    def fit_e(self, rows, nstart=6):
        B, D = self.B, self.D
        sw = np.sqrt(B.wE[rows])
        comp = Composite(B, 'T5')

        def res(p):
            return safe_log_res(comp.electrons(p, rows), B.lndE[rows], sw)
        rng = np.random.default_rng(5)
        base = [np.array([np.log(150.), .0, .0, np.log(1.), 0., 0., np.log(3.5), 0., 0.]),
                np.array([np.log(400.), .1, .3, np.log(2.), 0., 0., np.log(5.), 0., 0.]),
                np.array([np.log(60.), -.1, -.3, np.log(.6), 0., 0., np.log(2.5), 0., 0.])]
        if self.estarts is not None:
            base = [self.estarts]+base
        starts = base+[base[k % len(base)]+.25*rng.standard_normal(9) for k in range(max(nstart-len(base), 0))]
        fits = []
        for s in starts:
            o = lsq(res, s, ELaw.lo, ELaw.hi)
            fits.append((2*o.cost, o.x))
        fits.sort(key=lambda t: t[0])
        return fits[0][1], [float(f[0]) for f in fits]

    # ---------- pressure law of record (reference)
    def fit_plaw(self, alpha, rows):
        D, B = self.D, self.B
        law = BM.Law(D, 'pl', alpha, 'fixed')
        sw = np.sqrt(B.wP[rows])

        def res(p):
            return safe_log_res(law(p, rows), D.lnd[rows], sw)
        fits = []
        for s in BM.starts_for(law, 6):
            o = lsq(res, s, law.lo, law.hi)
            fits.append((2*o.cost, o.x))
        fits.sort(key=lambda t: t[0])
        return law, fits[0][1], [float(f[0]) for f in fits]

    # ---------- composites
    def fit_comp(self, kind, mode, rows_P, rows_E, pE_fixed=None, two=None, start=None, xout=4., nstart=4):
        B, D = self.B, self.D
        comp = Composite(B, kind, xout)
        swP, swE = np.sqrt(B.wP[rows_P]), np.sqrt(B.wE[rows_E])
        add = 0. if two is None else two[rows_P]
        rng = np.random.default_rng(7)
        if mode == 'sep':
            def res(pt):
                p = np.r_[pE_fixed, pt]
                return safe_log_res(comp(p, rows_P)+add, D.lnd[rows_P], swP)
            lo, hi = comp.lo[9:], comp.hi[9:]
            base = [comp.tstart if start is None else start[9:]]
        else:
            def res(p):
                return np.r_[safe_log_res(comp(p, rows_P)+add, D.lnd[rows_P], swP),
                             safe_log_res(comp.electrons(p[:9], rows_E), B.lndE[rows_E], swE)]
            lo, hi = comp.lo, comp.hi
            base = [np.r_[pE_fixed, comp.tstart] if start is None else start]
        starts = base+[np.clip(base[0]+np.r_[np.zeros(len(base[0])-len(comp.tn)) if mode == 'joint' else [],
                                             .3*rng.standard_normal(len(comp.tn))], lo+1e-6, hi-1e-6) for _ in range(nstart-1)]
        fits = []
        for s in starts:
            try:
                o = lsq(res, s, lo, hi)
                fits.append((2*o.cost, o.x))
            except Exception:                                    # noqa: BLE001
                fits.append((np.inf, s))
        fits.sort(key=lambda t: t[0])
        p = fits[0][1] if mode == 'joint' else np.r_[pE_fixed, fits[0][1]]
        return comp, p, [float(f[0]) for f in fits]


# ---------------------------------------------------------------- scores
def losses(B, Pm, Em):
    D = B.D
    rP = np.log(Pm/D.d)
    rE = np.log(Em/B.T['E_all'])
    i = B.inner
    return dict(L_P=float(np.sum(B.wP[i]*rP[i]**2)), L_E=float(np.sum(B.wE[i]*rE[i]**2)),
                L_P_scatter=float(np.sum(D.inner_weight('scatter')[i]*rP[i]**2)), L_P_uniform=float(np.sum(rP[i]**2)))


def bands(B, model, data, select, band_set):
    D = B.D
    r = np.log(model/data)
    lm10 = np.log10(D.M)
    out = {}
    for g, a, b in GROUPS:
        for lo, hi in band_set:
            s = select & D.shell & (lm10 >= a) & (lm10 < b) & (D.hi > lo) & (D.hi <= hi) & (D.nhalo >= 100)
            if s.sum() >= 3:
                out[f'{g}|{lo:g}-{hi:g}'] = dict(rms=float(np.sqrt(np.mean(r[s]**2))), mean=float(np.mean(r[s])), n=int(s.sum()))
        for kind in ('core', 'group'):
            s = select & np.char.startswith(D.kind, kind) & (lm10 >= a) & (lm10 < b) & (D.nhalo >= 100)
            if s.sum() >= 3:
                out[f'{g}|{kind} integral'] = dict(rms=float(np.sqrt(np.mean(r[s]**2))), mean=float(np.mean(r[s])), n=int(s.sum()))
        s = select & (lm10 >= a) & (lm10 < b) & (D.nhalo >= 100)
        if s.sum() >= 3:
            out[f'{g}|all'] = dict(rms=float(np.sqrt(np.mean(r[s]**2))), mean=float(np.mean(r[s])), n=int(s.sum()))
    return out


def extension(B, one):
    """Law (1-halo, extended beyond B32) against the halo-component, host-bound and all-gas stacks, outer shells."""
    D, T = B.D, B.T
    lm10 = np.log10(D.M)
    out = {}
    for g, a, b in GROUPS:
        for lo, hi in BANDS_OUT:
            s = D.shell & (D.region > 0) & (lm10 >= a) & (lm10 < b) & (D.hi > lo) & (D.hi <= hi) & (T['P_a2ii'] > 0) & (T['P_hb'] > 0) & (D.nhalo >= 100)
            if s.sum() >= 3:
                inside = (one[s] >= T['P_hb'][s]) & (one[s] <= T['P_a2ii'][s])
                out[f'{g}|{lo:g}-{hi:g}'] = dict(law_over_halo=float(np.median(one[s]/T['P_a2ii'][s])),
                                                 law_over_hostbound=float(np.median(one[s]/T['P_hb'][s])),
                                                 law_over_all=float(np.median(one[s]/D.d[s])),
                                                 frac_between_hostbound_and_halo=float(np.mean(inside)), n=int(s.sum()))
    return out


def physicality(model_fn, B):
    """Smooth / declining checks of the law from 0.3 to 4 R200c for every block: monotonic decline, log slope at 3-4 R200c."""
    D = B.D
    x = np.geomspace(.3, 4., 120)
    ub = np.unique(D.block)
    res = []
    for b in ub:
        P = model_fn(b, x)
        dl = np.diff(np.log(np.maximum(P, 1e-300)))/np.diff(np.log(x))
        res.append((bool(np.all(np.diff(P) < 0)), float(np.mean(dl[x[1:] >= 3.])), float(np.min(dl)), float(np.max(dl))))
    slopes = np.array([r[1] for r in res])
    return dict(n_blocks=len(ub), monotonic_decline=int(sum(r[0] for r in res)), slope_3_4R_median=float(np.median(slopes)),
                slope_3_4R_range=[float(slopes.min()), float(slopes.max())], max_local_slope=float(max(r[3] for r in res)))


def y200_points(B, Pfun_point):
    D = B.D
    out = {}
    xg, wg = np.polynomial.legendre.leggauss(48)
    xx = .5*(xg+1)
    for lm10, z in SIX:
        i = int(np.argmin(abs(np.log10(D.M)-lm10)+10*abs(D.z-z)))
        b = int(D.block[i])
        P = Pfun_point(b, xx)
        Ym = float(np.sum(.5*wg*4*np.pi*xx**2*P))
        out[f'{lm10:g}|{z:g}'] = dict(M=float(np.log10(D.M[i])), z=float(D.z[i]), Y200_model_over_data=Ym/B.Y200[b])
    return out


def point_fns(B, law_kind, obj, p):
    """Return f(block, x) -> P/P200c at radii x for one block, for the law of record or a composite."""
    D = B.D

    def f(b, x):
        i = int(np.flatnonzero(D.block == b)[0])
        if law_kind == 'plaw':
            p0, xc, beta, al, gam = obj.par(p, np.array([i]))
            q = x/xc[0]
            return .5176*p0[0]*q**gam[0]*(1+q**al[0])**-beta[0]
        bpos = np.array([obj.bpos[i]])
        return obj.pressure_nodes(p, x=x[None, :], bpos=bpos)[0]
    return f


def e_point_fn(B, pE):
    D = B.D

    def f(b, x):
        i = int(np.flatnonzero(D.block == b)[0])
        return ELaw.point(pE, x, D.lm[i], D.lz[i])
    return f


def summary(B, name, kind, obj, p, Pm, Em, k_total, k_P, info):
    D = B.D
    one = Pm
    L = losses(B, Pm, Em)
    n0 = B.n0
    pf = point_fns(B, 'plaw' if kind == 'plaw' else 'comp', obj, p)
    out = dict(name=name, kind=kind, names=list(obj.names), params=np.asarray(p).tolist(), k_total=k_total, k_P=k_P, info=info, **L,
               s2_P=L['L_P']/(n0-k_P), n_rows=n0,
               bands_P=bands(B, Pm, D.d, D.region == 0, BANDS_IN), bands_E=bands(B, Em, B.T['E_all'], D.region == 0, BANDS_IN),
               bands_P_outer_law_only=bands(B, Pm, D.d, D.region > 0, BANDS_OUT), extension=extension(B, one),
               physicality=physicality(pf, B), Y200=y200_points(B, pf))
    return out


def tau_rows(B):
    """Measured stack temperature tau = C_T P/E on every row (shells), and hydrostatic diagnostics use the electron law."""
    return C_T*B.D.d/B.T['E_all']


# ---------------------------------------------------------------- tasks
def task_fits(B, F):
    D = B.D
    i0 = B.inner
    res, arrays = {}, {}
    t0 = time.time()
    pE, eL = F.fit_e(i0, nstart=8)
    F.estarts = pE
    comp5 = Composite(B, 'T5')
    Em_sep = comp5.electrons(pE)
    print('E9', eL[:3], f'{time.time()-t0:.0f}s', flush=True)
    for alpha, nm, k in (('held', 'pressure law of record (alpha(z) held)', 9), ('const', 'pressure law, one constant alpha', 10)):
        law, pl, Ls = F.fit_plaw(alpha, i0)
        Pm = law(pl)
        res[f'plaw_{alpha}'] = summary(B, nm, 'plaw', law, pl, Pm, Em_sep, k+9, k, dict(start_losses=Ls))
        arrays[f'P__plaw_{alpha}'] = Pm
        print(f'plaw_{alpha}', Ls[:3], f'{time.time()-t0:.0f}s', flush=True)
    res['electron_law'] = dict(names=ELaw.names, params=pE.tolist(), start_losses=eL, L_E=eL[0],
                               bands_E=bands(B, Em_sep, B.T['E_all'], D.region == 0, BANDS_IN),
                               physicality_E=physicality(e_point_fn(B, pE), B))
    arrays['E__sep'] = Em_sep
    configs = [('T5', 'sep', 4.), ('T7', 'sep', 4.), ('T8', 'sep', 4.), ('H3', 'sep', 4.), ('H3c', 'sep', 4.), ('H6', 'sep', 4.),
               ('H6', 'sep', 3.), ('H6', 'sep', 6.), ('T7', 'joint', 4.), ('T8', 'joint', 4.), ('H3', 'joint', 4.), ('H6', 'joint', 4.)]
    sep_best = {}
    for kind, mode, xout in configs:
        name = f'{mode}_{kind}' + ('' if xout == 4. else f'_xout{xout:g}')
        start = None
        if mode == 'joint' and kind in sep_best:
            start = sep_best[kind]
        comp, p, Ls = F.fit_comp(kind, mode, i0, i0, pE_fixed=pE, start=start, xout=xout, nstart=5 if mode == 'sep' else 3)
        if mode == 'sep' and xout == 4.:
            sep_best[kind] = p
        Pm = comp(p)
        Em = comp.electrons(p[:9])
        res[name] = summary(B, name, kind, comp, p, Pm, Em, 9+len(comp.tn), 9+len(comp.tn), dict(start_losses=Ls, xout=xout, mode=mode))
        arrays[f'P__{name}'] = Pm
        arrays[f'E__{name}'] = Em
        print(name, Ls[:3], f'{time.time()-t0:.0f}s', flush=True)
    arrays.update(tau=tau_rows(B), d_P=D.d, d_E=B.T['E_all'], P_a2ii=B.T['P_a2ii'], P_hb=B.T['P_hb'], E_a2ii=B.T['E_a2ii'], E_hb=B.T['E_hb'],
                  block=D.block, lo=D.lo, hi=D.hi, region=D.region, shell=D.shell, M=D.M, z=D.z, nhalo=D.nhalo, b32=D.b32,
                  wP=B.wP, wE=B.wE, kind=D.kind.astype('U8'), P_err=B.T['P_err'], E_err=B.T['E_err'])
    out = dict(task='fits', checks=B.check, constants=dict(C_T=C_T, K_HSE=K_HSE, PE_FRAC=PE_FRAC, h=H, MU_XGP=MU_XGP), results=res,
               seconds=time.time()-t0)
    (OUT/'be_fits.json').write_text(json.dumps(out, indent=1)+'\n')
    np.savez_compressed(OUT/'be_fits_rows.npz', **arrays)
    return out


def nth_diagnostics(B, comp, p):
    """Non-thermal fraction at x = 0.5, 1, 1.5 for z = 0, 1, 2 and the share of region-0 quadrature nodes at the 0.95 cap."""
    D = B.D
    pT = p[9:]
    f = (lambda x, z: 1-np.exp(-pT[0]*(1+z)**pT[1]*x**pT[2])) if comp.kind.endswith('e') else \
        (lambda x, z: np.clip(pT[0]*(1+z)**pT[1]*x**pT[2], 0., .95))
    table = {f'z{z:g}': {f'x{x:g}': float(f(x, z)) for x in (.5, 1., 1.5)} for z in (0., 1., 2.)}
    i = B.inner
    fn = f(D.xq[i], D.z[i][:, None])
    return dict(f_nth=table, share_nodes_at_cap=float(np.mean(fn >= .95-1e-12)), share_nodes_above_0p5=float(np.mean(fn > .5)))


def task_fits2(B, F):
    """Follow-up: hydrostatic models integrated to 6 R200c and the smooth (uncapped) non-thermal fraction."""
    D = B.D
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    pE = np.array(fits['electron_law']['params'])
    i0 = B.inner
    res, arrays = {}, {}
    t0 = time.time()
    for name in ('sep_H3', 'sep_H6', 'sep_H6_xout3', 'sep_H6_xout6', 'joint_H3', 'joint_H6'):
        kind = name.split('_')[1]
        comp = Composite(B, kind, fits[name]['info'].get('xout', 4.))
        res[name] = dict(nth=nth_diagnostics(B, comp, np.array(fits[name]['params'])))
    for kind, mode, xout in (('H3e', 'sep', 6.), ('H6e', 'sep', 6.), ('H6', 'joint', 6.), ('H3e', 'joint', 6.), ('H6e', 'joint', 6.)):
        name = f'{mode}_{kind}_xout{xout:g}'
        base = fits.get(f'{mode}_{kind[:2]}' + ('' if mode == 'joint' else '_xout6'), fits.get(f'{mode}_{kind[:2]}'))
        start = np.array(base['params'])
        if kind.endswith('e'):
            start = start.copy()
        comp, p, Ls = F.fit_comp(kind, mode, i0, i0, pE_fixed=pE if mode == 'sep' else start[:9], start=start, xout=xout, nstart=4)
        Pm = comp(p)
        Em = comp.electrons(p[:9])
        res[name] = summary(B, name, kind, comp, p, Pm, Em, 9+len(comp.tn), 9+len(comp.tn), dict(start_losses=Ls, xout=xout, mode=mode))
        res[name]['nth'] = nth_diagnostics(B, comp, p)
        arrays[f'P__{name}'] = Pm
        arrays[f'E__{name}'] = Em
        print(name, Ls[:3], res[name]['nth']['share_nodes_at_cap'], f'{time.time()-t0:.0f}s', flush=True)
    (OUT/'be_fits2.json').write_text(json.dumps(dict(task='fits2', results=res, seconds=time.time()-t0), indent=1)+'\n')
    np.savez_compressed(OUT/'be_fits2_rows.npz', **arrays)


def task_heldout(B, F):
    """Leave one output out: refit on the other nine outputs (electron law too), loss on the held-out output's region-0 rows."""
    D = B.D
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    out = {}
    t0 = time.time()
    models = ['plaw_held', 'plaw_const', 'sep_T7', 'sep_H3', 'sep_H6', 'joint_T7', 'joint_H6']
    for name in models:
        out[name] = {}
    out['electron_law'] = {}
    for j, zj in enumerate(D.zs):
        train = B.inner[D.zi[B.inner] != j]
        test = B.inner[D.zi[B.inner] == j]
        F.estarts = np.array(fits['electron_law']['params'])
        pE, _ = F.fit_e(train, nstart=3)
        comp = Composite(B, 'T5')
        Em = comp.electrons(pE)
        rE = np.log(Em/B.T['E_all'])
        out['electron_law'][f'{zj:g}'] = dict(L_E_test=float(np.sum(B.wE[test]*rE[test]**2)), n_test=int(len(test)))
        for name in models:
            if name.startswith('plaw'):
                law, pl, _ = F.fit_plaw(name.split('_')[1], train)
                Pm = law(pl)
            else:
                mode, kind = name.split('_')
                start = np.array(fits[name]['params'])
                comp, p, _ = F.fit_comp(kind, mode, train, train, pE_fixed=pE if mode == 'sep' else start[:9], start=start, nstart=2)
                Pm = comp(p)
                if mode == 'joint':
                    Em = comp.electrons(p[:9])
            rP = np.log(Pm/D.d)
            out[name][f'{zj:g}'] = dict(L_P_test=float(np.sum(B.wP[test]*rP[test]**2)), w_test=float(B.wP[test].sum()), n_test=int(len(test)))
            if name.startswith('joint'):
                out[name][f'{zj:g}']['L_E_test'] = float(np.sum(B.wE[test]*np.log(Em/B.T['E_all'])[test]**2))
            print(zj, name, out[name][f'{zj:g}']['L_P_test'], f'{time.time()-t0:.0f}s', flush=True)
    for name in models:
        out[name]['total_L_P_test'] = float(sum(v['L_P_test'] for k, v in out[name].items() if isinstance(v, dict)))
    out['electron_law']['total_L_E_test'] = float(sum(v['L_E_test'] for k, v in out['electron_law'].items() if isinstance(v, dict)))
    (OUT/'be_heldout.json').write_text(json.dumps(dict(task='heldout', results=out, seconds=time.time()-t0), indent=1)+'\n')


def task_twohalo(B, F):
    """Split procedure with a 2-halo amplitude, to 4 R200c (region 1) and to 6 R200c (regions 1 + 2)."""
    D = B.D
    fits = json.loads((OUT/'be_fits.json').read_text())['results']
    T = D.template('hm_ex0.75')
    na, sa, la, ua, amp = BM.amp_model(D, 'A2zq')
    out, arrays = {}, {}
    t0 = time.time()
    pE = np.array(fits['electron_law']['params'])
    for name in ('plaw_const', 'plaw_held', 'sep_H6', 'sep_T7', 'joint_H6', 'joint_T7'):
        for rout, arows_mask in (('4', D.region == 1), ('6', D.region >= 1)):
            arows = np.flatnonzero(arows_mask & (D.d > 0))
            sv = np.sqrt(D.v[arows])
            p = np.array(fits[name]['params'])
            if name.startswith('plaw'):
                law = BM.Law(D, 'pl', name.split('_')[1], 'fixed')
                model = lambda q: law(q)  # noqa: E731
            else:
                mode, kind = name.split('_')
                comp = Composite(B, kind)
                model = lambda q: comp(q)  # noqa: E731
            pa = np.array(sa, float)
            for it in range(12):
                one = model(p)
                oa = lsq(lambda e: sv*(np.log(np.maximum(one[arows]+amp(e)[arows]*T[arows], 1e-300))-D.lnd[arows]), pa, np.array(la), np.array(ua))
                two = amp(oa.x)*T
                if name.startswith('plaw'):
                    sw = np.sqrt(B.wP[B.inner])
                    ol = lsq(lambda q: safe_log_res(law(q, B.inner)+two[B.inner], D.lnd[B.inner], sw), p, law.lo, law.hi)
                    pn = ol.x
                else:
                    _, pn, _ = F.fit_comp(kind, mode, B.inner, B.inner, pE_fixed=pE if mode == 'sep' else p[:9], two=two, start=p, nstart=1)
                step = max(np.max(abs(pn-p)), np.max(abs(oa.x-pa)))
                p, pa = pn, oa.x
                if step < 1e-5:
                    break
            one = model(p)
            A = amp(pa)
            full = one+A*T
            Em = Composite(B, 'T5').electrons(p[:9] if name.startswith('joint') else pE)
            key = f'{name}__rout{rout}'
            out[key] = dict(params=p.tolist(), amp=pa.tolist(), amp_by_z={f'{z:g}': float(np.median(A[D.zi == i])) for i, z in enumerate(D.zs)},
                            rounds=it+1, **losses(B, full, Em), bands_outer=bands(B, full, D.d, D.region > 0, BANDS_OUT),
                            bands_inner=bands(B, full, D.d, D.region == 0, BANDS_IN), extension_law_only=extension(B, one))
            arrays[f'full__{key}'] = full
            arrays[f'one__{key}'] = one
            print(key, out[key]['L_P'], out[key]['amp_by_z'].get('0'), f'{time.time()-t0:.0f}s', flush=True)
    arrays['template'] = T
    (OUT/'be_twohalo.json').write_text(json.dumps(dict(task='twohalo', results=out, seconds=time.time()-t0), indent=1)+'\n')
    np.savez_compressed(OUT/'be_twohalo_rows.npz', **arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--task', default='all')
    ap.add_argument('--mem-gib', type=float, default=float(os.environ.get('BE_MEM_GIB', 6)))
    a = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (int(a.mem_gib*2**30), int(a.mem_gib*2**30)))
    OUT.mkdir(parents=True, exist_ok=True)
    B = BEData()
    print('checks', json.dumps(B.check), flush=True)
    F = Fitter(B)
    if a.task in ('fits', 'all'):
        task_fits(B, F)
    if a.task == 'fits2':
        task_fits2(B, F)
    if a.task in ('heldout', 'all'):
        task_heldout(B, F)
    if a.task in ('twohalo', 'all'):
        task_twohalo(B, F)


if __name__ == '__main__':
    main()
