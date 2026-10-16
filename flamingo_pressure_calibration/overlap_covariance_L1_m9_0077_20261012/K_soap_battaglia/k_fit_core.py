"""k_fit_core: data-source-agnostic forward-model fitter for the Battaglia pressure (B12) and gas-density (B16) laws.

Test K uses it on SOAP-HBT aperture integrals (every halo at its own aperture radii); the later full-particle run can
use the SAME code on per-halo aperture integrals or on per-halo shell profiles (see `summarise_cell` and the usage
note at the end of this docstring).  Nothing in here reads files.

Conventions (XGPaint's, so fitted values paste directly into XGPaint's PowerLawParam constructors)
-------------------------------------------------------------------------------------------------
x = r / R200c (physical radii; R200c = radius of mean density 200 rho_c(z)).
Pressure (field 'P', form 'B12', Battaglia et al. 2012 eq. 10 + XGPaint profiles_y.jl):
    P_e(x) / P200 = 0.5176 * P0 * q^gamma * (1 + q^alpha)^(-beta),   q = x / x_c,
    P200 = G M200c (200 rho_c(z)) f_b / (2 R200c)  (thermal B12 normalisation), 0.5176 = P_e / P_th for X_H = 0.76.
    Published: alpha = 1, gamma = -0.3 fixed; P0, x_c, beta = A (M200c / 1e14 Msun)^alpha_m (1 + z)^alpha_z.
    XGPaint's Battaglia16ThermalSZProfile(P0_amp, x_c_amp, beta_amp, ...) takes this beta (it converts internally).
Gas density (field 'G', form 'B16', Battaglia 2016 eq. A1 as implemented by XGPaint profiles_tau.jl):
    rho_gas(x) / (f_b rho_c(z)) = rho0 * q^gamma * (1 + q^alpha)^(-(beta + gamma) / alpha),   q = x / x_c,
    published: x_c = 0.5, gamma = -0.2 fixed; rho0, alpha, beta = A M^alpha_m (1+z)^alpha_z.  (The paper's eq. A1 prints
    -(beta - gamma)/alpha and no f_b; the convention used here is XGPaint's: units of f_b rho_c and -(beta + gamma)/alpha.)
Data units (per halo, cumulative inside x):
    'P':  Ytilde(<x) = Y(<x) m_e c^2 / sigma_T / (P200 R200c^3) = 4 pi Int_0^x [P_e/P200](x') x'^2 dx'
    'G':  Mtilde(<x) = M_gas(<x) / (f_b M200c)                 = (3/200) Int_0^x [rho_gas/(f_b rho_c)](x') x'^2 dx'
    shell data (kind 1) are volume averages of the profile itself over [x1, x2] (P_e/P200 or rho_gas/(f_b rho_c)).

Laws (each profile parameter Q in {amp, xc, alpha, beta, gamma}; amp = P0 or rho0; ln|Q| is linear in the law
parameters; the sign of Q is fixed: gamma < 0, all others > 0):
    fixed:   Q = constant
    pl:      ln|Q| = ln|A| + alpha_m ln(M/1e14 Msun) + alpha_z ln(1+z)                     (Battaglia form)
    bpl:     ln|Q| = ln|Q_b| + a_lo|a_hi * ln(M/M_b) + alpha_z ln(1+z)  (a_lo for M < M_b, a_hi above; shared M_b)
    cell:    ln|Q| = free constant per cell (per-cell diagnostic fits)
    brk_amp: amp is 'pl' times (M/M_b)^a_brk for M < M_b (XGPaint BreakModel: alpha_break = a_brk, M_break = M_b)

Likelihood: data point p = ln(mean over the cell's haloes of a cumulative or shell quantity); model = ln(mean over
the haloes' quadrature nodes of the same quantity, each node at its own mass and radii); per cell the covariance is
C_boot (bootstrap over haloes) + (sigma_d^2 + sigma_0^2) I; sigma_d (model-discrepancy floor) is solved so that
chi^2/nu = 1.

Usage for a full-particle run (same code, same conventions)
----------------------------------------------------------
    import k_fit_core as K
    cells = []
    for each (mass bin, redshift) cell:                 # per-halo arrays of that cell
        # apertures: x[h, a] = r_a / R200c of halo h, F[h, a] = Ytilde or Mtilde (cumulative), valid[h, a]
        s = K.summarise_cell(lnM=ln(M200c_h), z=z, x2=x, F=F, valid=valid, kind=0, names=[...], n_boot=200, seed=...)
        # or shells: x1/x2 = shell edges in R200c ([h, s] or broadcast), F = shell-averaged P_e/P200 or rho/(f_b rho_c)
        cells.append(s)
    D = K.build_dataset(cells, field='P', use=lambda cell, j: True)     # choose points (fit range, masks)
    fit = K.fit_global(D, K.Law('B12', 'orig'))                        # theta, sigma_d, cov, chi2, ...
    boot = K.bootstrap_global(D, K.Law('B12', 'orig'), fit, n_boot=100)
    K.law_table(K.Law('B12', 'orig'), fit['theta'])                    # XGPaint-ready numbers
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import betainc, betaincc, betaln, hyp2f1
import scipy.sparse as sparse

E_OVER_TH = 0.5176                     # P_e / P_th, (2 + 2X)/(3 + 5X) for X = 0.76 (B12 and XGPaint)
PIVOT_MSUN = 1.0e14
LN_PIVOT = math.log(PIVOT_MSUN)
LN10 = math.log(10.0)
SHRINK_MIN = 0.05                      # minimum shrinkage of the bootstrap correlation towards the identity
SIGMA0 = 2.0e-3                        # numerical floor (ln units) always added to the covariance diagonal (strata compression <= 0.1%, 9-bit mantissa)
FIELD_FORM = {"P": "B12", "G": "B16"}
FIELD_NORM = {"P": 4.0 * math.pi, "G": 3.0 / 200.0}   # cumulative data = norm * Int_0^x F x'^2 dx'
QNAMES = ("amp", "xc", "alpha", "beta", "gamma")
QSIGN = {"amp": 1.0, "xc": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": -1.0}
XGP_NAME = {"amp": "P0", "xc": "x_c", "alpha": "alpha", "beta": "beta", "gamma": "gamma"}

# Published laws: (A, alpha_m, alpha_z) or a fixed constant.  B12: Battaglia et al. 2012 ApJ 758 75, Table 1 (AGN
# feedback, Delta = 200); B16: Battaglia 2016 JCAP 08 058, Table 2 (AGN feedback, Delta = 200).  Identical to the
# defaults of XGPaint's Battaglia16ThermalSZProfile and BattagliaTauProfile (validated branch 'cluster').
PUBLISHED = {
    "B12": {"amp": (18.1, 0.154, -0.758), "xc": (0.497, -0.00865, 0.731), "alpha": 1.0,
            "beta": (4.35, 0.0393, 0.415), "gamma": -0.3},
    "B16": {"amp": (4.0e3, 0.29, -0.66), "xc": 0.5, "alpha": (0.88, -0.03, 0.19),
            "beta": (3.83, 0.04, -0.025), "gamma": -0.2},
}
FREE3 = {"B12": ("amp", "xc", "beta"), "B16": ("amp", "alpha", "beta")}

# ---------------------------------------------------------------------------------------------------- profiles
def outer_exponent(form, alpha, beta, gamma):
    """Exponent e of (1 + q^alpha)^(-e)."""
    return beta if form == "B12" else (beta + gamma) / alpha


def profile(x, form, amp, xc, alpha, beta, gamma):
    """3D profile F(x): P_e/P200 (B12, includes 0.5176) or rho_gas/(f_b rho_c) (B16)."""
    pref = E_OVER_TH if form == "B12" else 1.0
    q = np.asarray(x, float) / xc
    return pref * amp * q ** gamma * (1.0 + q ** alpha) ** (-outer_exponent(form, alpha, beta, gamma))


LN_HALF = math.log(0.5)
B_SWITCH = 0.02                        # b >= B_SWITCH: regularised incomplete beta (scipy betainc / betaincc)
N_SERIES = 64                          # terms of the (1-w)^(a-1) series in _upper_part (geometric, ratio <= 1/2)
A_SPLIT = 4.0                          # a > A_SPLIT: series only on [y, 1/(a-1)], Gauss-Legendre on the rest
_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)
GL_PANELS = 4
LN_COMPLEMENT = math.log(0.01)         # t > 1/2, 0.02 <= b < 1: complement form when 1 - t < 0.01


def _expm1_over(v):
    """expm1(v) / v (= 1 at v = 0), elementwise, accurate for |v| -> 0."""
    v = np.asarray(v, float)
    out = 1.0 + 0.5 * v
    big = np.abs(v) > 1e-8
    with np.errstate(over="ignore", invalid="ignore"):
        out[big] = np.expm1(v[big]) / v[big]
    return out


def _series_part(a, b, l1t, lw):
    """Int_y^w v^(b-1) (1-v)^(a-1) dv with y = exp(l1t) <= w = exp(lw) <= 1/2, for a > 0 and ANY real b.
    (1-v)^(a-1) = sum_k c_k v^k, c_k = (1-a)_k / k!; Int_y^w v^(b+k-1) dv = w^p L E(-p L), p = b + k, L = ln(w/y),
    E(v) = expm1(v)/v, which stays exact when p -> 0 (b near 0 or a negative integer) and when y << w."""
    L = lw - l1t
    J = np.zeros(a.shape)
    c = np.ones(a.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(N_SERIES):
            p = b + k
            term = c * np.exp(p * lw) * L * _expm1_over(-p * L)
            J += term
            c = c * (k + 1 - a) / (k + 1)
            if k % 8 == 7 and np.all(np.abs(term) <= 1e-17 * np.abs(J)):
                break                                   # every element converged (geometric tail <= 2x last term)
    return J


def _gl_part(a, b, l_lo):
    """Int_{exp(l_lo)}^{1/2} v^(b-1) (1-v)^(a-1) dv by Gauss-Legendre in ln v (GL_PANELS x 32 nodes; the integrand
    exp(b tau) (1 - e^tau)^(a-1) is smooth on [l_lo, ln 1/2]; used only for a > A_SPLIT where l_lo >= -ln(a-1))."""
    out = np.zeros(a.shape)
    edges = l_lo[:, None] + (LN_HALF - l_lo)[:, None] * np.linspace(0.0, 1.0, GL_PANELS + 1)[None, :]
    for p in range(GL_PANELS):
        lo, hi = edges[:, p], edges[:, p + 1]
        tau = 0.5 * (hi - lo)[:, None] * _GL_X[None, :] + 0.5 * (hi + lo)[:, None]
        f = np.exp(b[:, None] * tau + (a[:, None] - 1.0) * np.log1p(-np.exp(tau)))
        out += 0.5 * (hi - lo) * (f @ _GL_W)
    return out


def incomplete_beta(a, b, lt, l1t):
    """B(t; a, b) = Int_0^t u^(a-1) (1-u)^(b-1) du for a > 0 and any real b (finite for t < 1), with ln t and
    ln(1 - t) both passed EXACTLY (1 - t is never formed from t, so t -> 1 keeps full precision).
      t <= 1/2, b >= 0.02:  B(a, b) I_t(a, b)                                (scipy betainc)
      t <= 1/2, b <  0.02:  t^a (1-t)^b / a 2F1(a + b, 1; a + 1; t)           (scipy hyp2f1, |t| <= 1/2)
      t >  1/2, b >= 0.02:  B(a, b) I_t(a, b) (betainc), or, for b < 1 and 1 - t < 0.01, B(a, b) [1 - I_{1-t}(b, a)]
                            (scipy betaincc at the exact 1 - t)
      t >  1/2, b <  0.02:  B(1/2; a, b) + Int_{1-t}^{1/2} w^(b-1) (1-w)^(a-1) dw, the second term by the series of
                            _series_part (a <= 4) or, for a > 4, the series up to w = 1/(a-1) plus Gauss-Legendre
                            from there to 1/2 (keeps the alternating series free of cancellation).
    Checked against 40-digit quadrature (mpmath) over alpha 0.1-30, x/x_c up to 400 (s up to e^180), b from -3.03
    to 15 including b ~ 0 and b ~ -1, -2, -3: max relative error 4.2e-15 in 1,245 cases (k_parity.check_integral,
    results/k5_parity_local.json)."""
    a, b, lt, l1t = (np.asarray(v, float) for v in (a, b, lt, l1t))
    out = np.empty(a.shape)
    low = lt <= LN_HALF
    pos = b >= B_SWITCH
    m = low & pos
    if np.any(m):
        out[m] = np.exp(betaln(a[m], b[m])) * betainc(a[m], b[m], np.exp(lt[m]))
    m = low & ~pos
    if np.any(m):
        out[m] = np.exp(a[m] * lt[m] + b[m] * l1t[m]) / a[m] * hyp2f1(a[m] + b[m], 1.0, a[m] + 1.0, np.exp(lt[m]))
    m = ~low & pos
    if np.any(m):
        # rounding t costs a relative error ~ eps (1-t)^(b-1) / B: negligible unless b < 1 and 1 - t is small, where the
        # complement is evaluated at the exact 1 - t (betaincc is ~10x slower than betainc, so only there)
        cm = m & (b < 1.0) & (l1t < LN_COMPLEMENT)
        rm = m & ~cm
        if np.any(rm):
            out[rm] = np.exp(betaln(a[rm], b[rm])) * betainc(a[rm], b[rm], np.exp(lt[rm]))
        if np.any(cm):
            out[cm] = np.exp(betaln(a[cm], b[cm])) * betaincc(b[cm], a[cm], np.exp(l1t[cm]))
    m = ~low & ~pos
    if np.any(m):
        am, bm, ly = a[m], b[m], l1t[m]
        res = np.exp((am + bm) * LN_HALF) / am * hyp2f1(am + bm, 1.0, am + 1.0, 0.5)      # B(1/2; a, b)
        small = am <= A_SPLIT
        if np.any(small):
            res[small] += _series_part(am[small], bm[small], ly[small], np.full(int(small.sum()), LN_HALF))
        big = ~small
        if np.any(big):
            lws = -np.log(am[big] - 1.0)                                                       # ln w_s, w_s < 1/2
            yb = ly[big]
            below = yb < lws
            part = np.zeros(int(big.sum()))
            if np.any(below):
                part[below] = _series_part(am[big][below], bm[big][below], yb[below], lws[below])
            part += _gl_part(am[big], bm[big], np.maximum(yb, lws))
            res[big] += part
        out[m] = res
    return out


def cumulative(x, form, amp, xc, alpha, beta, gamma):
    """C(x) = Int_0^x F(x') x'^2 dx' in closed form (all inputs broadcast; x > 0; +inf where gamma <= -3).

    With q = x'/x_c and s = q^alpha, u = s/(1+s):
        C = pref amp x_c^3 / alpha * Int_0^{s_x} s^(a-1) (1+s)^(-e) ds = pref amp x_c^3 / alpha * B(t; a, b),
    a = (3 + gamma)/alpha, e = outer exponent, b = e - a, t = s_x/(1+s_x), B(t; a, b) = Int_0^t u^(a-1)(1-u)^(b-1) du
    (the incomplete beta function, finite for t < 1 and any real b; b < 0 means an outer slope shallower than x^-3).
    ln t and ln(1 - t) = -ln(1 + s_x) are computed separately, so large s_x (high alpha, x >> x_c) keeps full
    precision; see incomplete_beta for the branches.  (The version used in run 2 of 2026-10-15 formed 1 - t from t in
    its b < 0.02 branch and was wrong for s_x >~ 1e12: at alpha = 11.6, x/x_c = 13.75 (B16, gamma = -0.55) it was
    +120% (beta = 2.8), +616% (2.95) or inf (3.01, 3.2); at alpha = 10 the error was 3-5e-7.)"""
    x, amp, xc, alpha, beta, gamma = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, amp, xc, alpha, beta, gamma)))
    pref = E_OVER_TH if form == "B12" else 1.0
    e = outer_exponent(form, alpha, beta, gamma)
    a = (3.0 + gamma) / alpha
    b = e - a
    ls = alpha * np.log(x / xc)                      # ln s_x
    lt = -np.logaddexp(0.0, -ls)                     # ln t = ln s/(1+s)
    l1t = -np.logaddexp(0.0, ls)                     # ln (1 - t) = ln 1/(1+s)
    # 3 + gamma <= 0 (a <= 0): the profile is not integrable at x -> 0 (infinite mass / Y); return +inf, which the
    # least-squares step control rejects (a free gamma law can reach gamma <= -3 at extreme nodes)
    ok = (a > 0) & (alpha > 0)
    out = np.full(x.shape, np.inf)
    if np.any(ok):
        out[ok] = incomplete_beta(a[ok], b[ok], lt[ok], l1t[ok])
    return pref * amp * xc ** 3 / alpha * out


def node_value(kind, x1, x2, form, field, phi):
    """Per-node model value: kind 0 = cumulative in data units (FIELD_NORM * C(x2)); kind 1 = volume average of the
    profile over [x1, x2] = 3 (C(x2) - C(x1)) / (x2^3 - x1^3)."""
    c2 = cumulative(x2, form, phi["amp"], phi["xc"], phi["alpha"], phi["beta"], phi["gamma"])
    out = FIELD_NORM[field] * c2
    shell = np.asarray(kind) == 1
    if np.any(shell):
        x1s, x2s = x1[shell], x2[shell]
        inner = x1s > 0
        c1 = np.zeros(len(x1s))
        if np.any(inner):
            c1[inner] = cumulative(x1s[inner], form, *(np.asarray(phi[q])[shell][inner] for q in QNAMES))
        out = out.copy()
        out[shell] = 3.0 * (c2[shell] - c1) / (x2s ** 3 - x1s ** 3)
    return out


# ---------------------------------------------------------------------------------------------------- laws
class Law:
    """Parametrisation of the five profile parameters as functions of (M200c, z[, cell]).

    form: 'B12' | 'B16';  kind: 'orig' | 'brk_amp' | 'brk_all' | 'free' | 'cell' | 'published'.
      orig     the published structure: 3 free parameters with (A, alpha_m, alpha_z) laws, 2 fixed (9 law parameters)
      brk_amp  orig, with the amplitude multiplied by (M/M_b)^a_brk below M_b (XGPaint BreakModel; 11)
      brk_all  each of the 3 free parameters a broken power law in mass with a shared M_b (Lee22 BPLall style)
               and its own (1+z) law (13)
      free     orig + gamma with its own law (B12, 12) / x_c and gamma with their own laws (B16, 15)
      cell     the 3 free parameters constant in each cell (per-cell diagnostic fits; 3 per cell)
    """

    def __init__(self, form, kind, n_cells=1, fixed=None):
        self.form, self.kind, self.n_cells = form, kind, n_cells
        pub = PUBLISHED[form]
        free = list(FREE3[form])
        if kind == "free":
            free += ["gamma"] if form == "B12" else ["xc", "gamma"]
        self.qkind, self.fixed = {}, {}
        for q in QNAMES:
            if q in free:
                self.qkind[q] = {"brk_all": "bpl", "cell": "cell"}.get(kind, "pl")
            else:
                self.qkind[q] = "fixed"
                self.fixed[q] = float(pub[q] if not isinstance(pub[q], tuple) else pub[q][0])
        if fixed:
            self.fixed.update(fixed)
        names = []
        for q in QNAMES:
            k = self.qkind[q]
            if k == "pl":
                names += [f"{q}:lnA", f"{q}:am", f"{q}:az"]
            elif k == "bpl":
                names += [f"{q}:lnQb", f"{q}:alo", f"{q}:ahi", f"{q}:az"]
            elif k == "cell":
                names += [f"{q}:ln[{c}]" for c in range(n_cells)]
        if kind == "brk_amp":
            names += ["amp:abrk"]
        if kind in ("brk_amp", "brk_all"):
            names += ["log10Mb"]
        self.names = names
        self.index = {n: i for i, n in enumerate(names)}
        self.free_shape = [q for q in QNAMES if self.qkind[q] != "fixed" and q != "amp"]

    @property
    def n(self):
        return len(self.names)

    # ---- start values and bounds
    def start(self, cell_lnM=None, cell_z=None):
        """Start vector: published laws (break mass 10^14; per-cell values at each cell's mean mass and z)."""
        pub = PUBLISHED[self.form]
        th = np.zeros(self.n)
        for q in QNAMES:
            k = self.qkind[q]
            law = pub[q] if isinstance(pub[q], tuple) else (pub[q], 0.0, 0.0)
            lnA = math.log(abs(law[0]))
            if k == "pl":
                th[self.index[f"{q}:lnA"]], th[self.index[f"{q}:am"]], th[self.index[f"{q}:az"]] = lnA, law[1], law[2]
            elif k == "bpl":
                th[self.index[f"{q}:lnQb"]] = lnA
                th[self.index[f"{q}:alo"]] = th[self.index[f"{q}:ahi"]] = law[1]
                th[self.index[f"{q}:az"]] = law[2]
            elif k == "cell":
                for c in range(self.n_cells):
                    lm = (cell_lnM[c] - LN_PIVOT) if cell_lnM is not None else 0.0
                    lz = math.log1p(cell_z[c]) if cell_z is not None else 0.0
                    th[self.index[f"{q}:ln[{c}]"]] = lnA + law[1] * lm + law[2] * lz
        if "log10Mb" in self.index:
            th[self.index["log10Mb"]] = 14.0
        return th

    def bounds(self):
        lo = np.full(self.n, -np.inf)
        hi = np.full(self.n, np.inf)
        box = {"B12": {"amp": (1e-3, 1e4), "xc": (1e-2, 10.0), "alpha": (0.1, 10.0), "beta": (0.5, 20.0), "gamma": (1e-3, 2.5)},
               "B16": {"amp": (1e-1, 1e8), "xc": (1e-2, 10.0), "alpha": (0.1, 10.0), "beta": (1.0, 20.0), "gamma": (1e-3, 2.5)}}[self.form]
        for i, n in enumerate(self.names):
            if n == "log10Mb":
                lo[i], hi[i] = 12.8, 15.0
                continue
            q, what = n.split(":")
            if what in ("lnA", "lnQb") or what.startswith("ln["):
                lo[i], hi[i] = math.log(box[q][0]), math.log(box[q][1])
            elif what in ("am", "alo", "ahi", "abrk"):
                lo[i], hi[i] = -3.0, 3.0
            elif what == "az":
                lo[i], hi[i] = -6.0, 6.0
        return lo, hi

    # ---- evaluation
    def ell(self, theta, lnM, lz, cell=None, need_grad=False):
        """ln|Q| per node for every non-fixed Q, and (optionally) d ln|Q| / d theta_j as a list of (j, q, array)."""
        th = np.asarray(theta, float)
        lm = lnM - LN_PIVOT
        out, grads = {}, []
        lnMb = th[self.index["log10Mb"]] * LN10 if "log10Mb" in self.index else None
        for q in QNAMES:
            k = self.qkind[q]
            if k == "pl":
                i0, i1, i2 = (self.index[f"{q}:{s}"] for s in ("lnA", "am", "az"))
                out[q] = th[i0] + th[i1] * lm + th[i2] * lz
                if need_grad:
                    grads += [(i0, q, 1.0), (i1, q, lm), (i2, q, lz)]
            elif k == "bpl":
                i0, il, ih, iz = (self.index[f"{q}:{s}"] for s in ("lnQb", "alo", "ahi", "az"))
                d = lnM - lnMb
                low = d < 0
                slope = np.where(low, th[il], th[ih])
                out[q] = th[i0] + slope * d + th[iz] * lz
                if need_grad:
                    grads += [(i0, q, 1.0), (il, q, np.where(low, d, 0.0)), (ih, q, np.where(low, 0.0, d)), (iz, q, lz),
                              (self.index["log10Mb"], q, -slope * LN10)]
            elif k == "cell":
                base = self.index[f"{q}:ln[0]"]
                out[q] = th[base + cell]
                if need_grad:
                    for c in range(self.n_cells):
                        grads.append((base + c, q, (cell == c).astype(float)))
        if self.kind == "brk_amp":
            ia, ib = self.index["amp:abrk"], self.index["log10Mb"]
            d = np.minimum(lnM - lnMb, 0.0)
            out["amp"] = out["amp"] + th[ia] * d
            if need_grad:
                grads += [(ia, "amp", d), (ib, "amp", np.where(lnM < lnMb, -th[ia] * LN10, 0.0))]
        return out, grads

    def phi(self, theta, lnM, lz, cell=None):
        """Profile parameters (signed) per node."""
        ell, _ = self.ell(theta, lnM, lz, cell)
        res = {}
        for q in QNAMES:
            res[q] = QSIGN[q] * np.exp(ell[q]) if q in ell else np.full(np.shape(lnM), self.fixed[q])
        return res


def published_phi(form, lnM, z):
    """Published profile parameters at (lnM, z) (same as Law(form, 'orig') at its start vector)."""
    law = Law(form, "orig")
    return law.phi(law.start(), np.asarray(lnM, float), np.log1p(np.asarray(z, float)))


# ---------------------------------------------------------------------------------------------------- data
def strata_groups(lnM, lnx, n_m=4, n_x=8):
    """Index groups of the quadrature strata of one aperture's haloes: n_m mass-quantile strata, each split into n_x
    ln-x-quantile strata.  With at most n_m*n_x haloes every halo is its own group."""
    n = len(lnM)
    if n <= n_m * n_x:
        return [np.array([i]) for i in range(n)]
    om = np.argsort(lnM, kind="stable")
    groups = []
    for part in np.array_split(om, n_m):
        ox = part[np.argsort(lnx[part], kind="stable")]
        groups += [g for g in np.array_split(ox, n_x) if len(g)]
    return groups


def summarise_cell(lnM, z, x2, F, valid, kind=0, x1=None, names=None, n_boot=200, seed=0, extra=None,
                   n_m=4, n_x=8, boot_counts=None):
    """Reduce one cell's per-halo data to what the fitter needs.

    lnM [n]: ln(M200c/Msun);  z: redshift;  x2 [n, A]: outer radius / R200c of every halo and aperture (shell);
    x1 [n, A] or None: inner shell edge (kind 1 only);  F [n, A]: per-halo cumulative (kind 0) or shell-averaged
    (kind 1) value in the field's data units;  valid [n, A] booleans.
    Returns per aperture j: mean over valid haloes, bootstrap replicate means [A, n_boot] (the same halo resamples for
    every aperture: indices drawn with replacement over the cell's haloes, numpy default_rng(seed); the replicate
    mean of aperture j is over the drawn haloes valid at j), N valid, mean x, and quadrature nodes: strata of
    (ln M, ln x2) (strata_groups), node = stratum mean of ln M and of ln x1, ln x2, weight = stratum share.
    boot_counts [n_boot, n] can be passed to share the resamples between several quantities."""
    lnM = np.asarray(lnM, float)
    n, A = np.shape(F)
    valid = np.asarray(valid, bool)
    Fv = np.where(valid, F, 0.0).astype(float)
    V = valid.astype(float)
    nval = V.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = Fv.sum(0) / nval
    if boot_counts is None:
        boot_counts = bootstrap_counts(n, n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        bmean = (boot_counts @ Fv) / (boot_counts @ V)          # [B, A]
    X2 = np.broadcast_to(np.asarray(x2, float), (n, A))
    X1 = None if x1 is None else np.broadcast_to(np.asarray(x1, float), (n, A))
    out = {"z": float(z), "kind": int(kind), "n_halo": int(n), "mean": mean, "n_valid": nval.astype(int),
           "boot_mean": bmean.T.astype(np.float32), "nodes": [], "mean_lnM": float(lnM.mean()) if n else np.nan}
    xbar = np.full(A, np.nan)
    x1bar = np.zeros(A)
    for j in range(A):
        ok = np.nonzero(valid[:, j])[0]
        if not len(ok):
            out["nodes"].append(None)
            continue
        lx2 = np.log(X2[ok, j])
        xbar[j] = X2[ok, j].mean()
        groups = strata_groups(lnM[ok], lx2, n_m, n_x)
        node = {"lnM": np.array([lnM[ok][g].mean() for g in groups]),
                "x2": np.array([np.exp(lx2[g].mean()) for g in groups]),
                "w": np.array([len(g) for g in groups], float) / len(ok)}
        if X1 is None:
            node["x1"] = np.zeros(len(groups))
        else:
            x1j = X1[ok, j]
            x1bar[j] = x1j.mean()
            node["x1"] = np.array([np.exp(np.log(x1j[g]).mean()) if np.all(x1j[g] > 0) else 0.0 for g in groups])
        out["nodes"].append(node)
    out["xbar"], out["x1bar"] = xbar, x1bar
    out["names"] = list(names) if names is not None else [f"a{j}" for j in range(A)]
    if extra:
        out.update(extra)
    return out


def bootstrap_counts(n, n_boot, seed):
    """[n_boot, n] multiplicities of a with-replacement resample of n haloes (numpy default_rng(seed))."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), np.float64)
    for b in range(n_boot):
        out[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return out


class Dataset:
    """Flat container of fit points.  Build with build_dataset()."""

    def __init__(self, field, cell, z, kind, lnd, boot, node_start, node_lnM, node_x1, node_x2, node_w, meta):
        self.field, self.form = field, FIELD_FORM[field]
        self.cell, self.z, self.kind, self.lnd, self.boot = cell, z, kind, lnd, boot
        self.node_start, self.node_lnM, self.node_x1, self.node_x2, self.node_w = node_start, node_lnM, node_x1, node_x2, node_w
        self.meta = meta
        counts = np.diff(np.r_[node_start, len(node_lnM)])
        self.node_point = np.repeat(np.arange(len(lnd)), counts)
        self.node_lz = np.log1p(z[self.node_point])
        self.node_kind = kind[self.node_point]
        self.node_cell = cell[self.node_point]
        self.n_cells = int(cell.max()) + 1 if len(cell) else 0
        self.cell_points = [np.nonzero(cell == c)[0] for c in range(self.n_cells)]
        self._eig = None

    @property
    def n(self):
        return len(self.lnd)

    def subset(self, points):
        """Dataset restricted to the given point indices (cells renumbered)."""
        points = np.asarray(points)
        starts = self.node_start
        ends = np.r_[starts[1:], len(self.node_lnM)]
        sel_nodes = np.concatenate([np.arange(starts[p], ends[p]) for p in points]) if len(points) else np.array([], int)
        counts = ends[points] - starts[points]
        new_start = np.r_[0, np.cumsum(counts)[:-1]] if len(points) else np.array([], int)
        uc, cell = np.unique(self.cell[points], return_inverse=True)
        meta = {k: (np.asarray(v)[points] if isinstance(v, np.ndarray) and len(v) == self.n and not k.startswith("cell_") else v)
                for k, v in self.meta.items()}
        for k in ("cell_label", "cell_lnM", "cell_n"):
            if k in self.meta and len(self.meta[k]) == self.n_cells:
                meta[k] = np.asarray(self.meta[k])[uc]
        meta["cell_original"] = uc
        return Dataset(self.field, cell, self.z[points], self.kind[points], self.lnd[points],
                       None if self.boot is None else self.boot[points], new_start, self.node_lnM[sel_nodes],
                       self.node_x1[sel_nodes], self.node_x2[sel_nodes], self.node_w[sel_nodes], meta)

    def eig(self):
        """Per-cell eigen-decomposition of the covariance of the ln means.

        C_c = D_c R_c D_c with D_c the bootstrap standard deviations and R_c the bootstrap correlation shrunk towards a
        pooled target: R_c <- (1 - lambda) R_c + lambda T_c, lambda = clip(p / N_halo, SHRINK_MIN, 1) (p = points in
        the cell), T_c[j, k] = N-weighted mean of the bootstrap correlation of the same two named apertures over all
        cells at the same redshift (0 where no other cell has the pair; 1 on the diagonal).  Small cells thus borrow
        the correlation structure of the well-sampled cells instead of inverting a noisy sample covariance.
        Without bootstrap replicates meta['sigma'] gives a diagonal covariance."""
        if self._eig is None:
            res = []
            if self.boot is None:
                for pts in self.cell_points:
                    lam, vec = np.linalg.eigh(np.diag(np.asarray(self.meta["sigma"])[pts] ** 2))
                    res.append((np.maximum(lam, 0.0), vec))
                self._eig = res
                return res
            names = np.asarray(self.meta["point_name"])
            ncell = np.asarray(self.meta.get("cell_n", np.full(self.n_cells, np.inf)), float)
            corr, sd = [], []
            for pts in self.cell_points:
                b = np.asarray(self.boot[pts], float)
                C = np.atleast_2d(np.cov(b)) if b.shape[1] > 1 else np.zeros((len(pts), len(pts)))
                d = np.sqrt(np.clip(np.diag(C), 0, None))
                with np.errstate(invalid="ignore", divide="ignore"):
                    R = C / np.outer(d, d)
                R[~np.isfinite(R)] = 0.0
                np.fill_diagonal(R, 1.0)
                corr.append(R); sd.append(d)
            # pooled target per redshift and aperture pair
            pool = {}
            for c, pts in enumerate(self.cell_points):
                zc = round(float(self.z[pts[0]]), 6)
                w = ncell[c] if np.isfinite(ncell[c]) else 1.0
                nm = names[pts]
                for j in range(len(pts)):
                    for k in range(j + 1, len(pts)):
                        key = (zc,) + tuple(sorted((nm[j], nm[k])))
                        acc = pool.setdefault(key, [0.0, 0.0])
                        acc[0] += w * corr[c][j, k]; acc[1] += w
            for c, pts in enumerate(self.cell_points):
                zc = round(float(self.z[pts[0]]), 6)
                nm = names[pts]
                T = np.eye(len(pts))
                for j in range(len(pts)):
                    for k in range(j + 1, len(pts)):
                        acc = pool.get((zc,) + tuple(sorted((nm[j], nm[k]))))
                        if acc and acc[1] > 0:
                            T[j, k] = T[k, j] = acc[0] / acc[1]
                lam_s = float(np.clip(len(pts) / ncell[c], SHRINK_MIN, 1.0)) if np.isfinite(ncell[c]) else SHRINK_MIN
                R = (1 - lam_s) * corr[c] + lam_s * T
                C = R * np.outer(sd[c], sd[c])
                lam, vec = np.linalg.eigh(C)
                res.append((np.maximum(lam, 0.0), vec))
            self._eig = res
        return self._eig

    def whitener(self, sigma_d):
        """Sparse block-diagonal W with W^T W = (C_boot + (sigma_d^2 + sigma_0^2) I)^-1 per cell."""
        rows, cols, vals = [], [], []
        s2 = sigma_d ** 2 + SIGMA0 ** 2
        for pts, (lam, vec) in zip(self.cell_points, self.eig()):
            Wc = (vec / np.sqrt(lam + s2)).T
            r, c = np.meshgrid(pts, pts, indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(Wc.ravel())
        if not rows:
            return sparse.csr_matrix((0, 0))
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self.n, self.n))

    def chi2_cells(self, resid, sigma_d):
        s2 = sigma_d ** 2 + SIGMA0 ** 2
        out = np.zeros(self.n_cells)
        for c, (pts, (lam, vec)) in enumerate(zip(self.cell_points, self.eig())):
            proj = vec.T @ resid[pts]
            out[c] = np.sum(proj ** 2 / (lam + s2))
        return out


def build_dataset(cells, field, use, quantity="mean", boot_key="boot_mean"):
    """Concatenate cell summaries into a Dataset.  use(cell_dict, j) -> bool selects the aperture/shell points; points
    with non-positive or non-finite means are skipped.  Each cell dict needs 'z', 'kind', quantity, boot_key,
    'nodes', 'xbar', 'names' and may carry 'cell_label'."""
    cell, z, kind, lnd, boot, starts, nl, n1, n2, nw = [], [], [], [], [], [], [], [], [], []
    meta = {"point_name": [], "point_xbar": [], "point_x1bar": [], "cell_label": [], "cell_lnM": [], "cell_n": []}
    total = 0
    ci = -1
    for c in cells:
        pts = [j for j in range(len(c["names"])) if c["nodes"][j] is not None and use(c, j)
               and np.isfinite(c[quantity][j]) and c[quantity][j] > 0]
        if not pts:
            continue
        ci += 1
        meta["cell_label"].append(c.get("cell_label", ci))
        meta["cell_lnM"].append(c.get("mean_lnM", np.nan))
        meta["cell_n"].append(c.get("n_halo", np.inf))
        for j in pts:
            nd = c["nodes"][j]
            cell.append(ci); z.append(c["z"]); kind.append(c["kind"]); lnd.append(math.log(c[quantity][j]))
            b = np.asarray(c[boot_key][j], float)
            boot.append(np.log(np.where(b > 0, b, np.nan)))
            starts.append(total); total += len(nd["w"])
            nl.append(nd["lnM"]); n1.append(nd["x1"]); n2.append(nd["x2"]); nw.append(nd["w"] / nd["w"].sum())
            meta["point_name"].append(c["names"][j]); meta["point_xbar"].append(c["xbar"][j])
            meta["point_x1bar"].append(c.get("x1bar", np.zeros(len(c["names"])))[j])
    boot = np.array(boot) if boot else None
    if boot is not None and np.isnan(boot).any():           # replicate without any valid halo: use the point's mean
        bad = np.isnan(boot)
        boot[bad] = np.broadcast_to(np.array(lnd)[:, None], boot.shape)[bad]
    for k in ("point_xbar", "point_x1bar", "cell_lnM", "cell_n"):
        meta[k] = np.array(meta[k], float)
    meta["point_name"] = np.array(meta["point_name"])
    cat = (lambda v: np.concatenate(v) if v else np.array([]))   # noqa: E731
    return Dataset(field, np.array(cell, int), np.array(z, float), np.array(kind, int), np.array(lnd, float), boot,
                   np.array(starts, int), cat(nl), cat(n1), cat(n2), cat(nw), meta)


# ---------------------------------------------------------------------------------------------------- model
def predict(theta, D, law, need_jac=False, h=1e-5):
    """ln model per point (and d ln model / d theta).  The Jacobian uses the chain rule through ln|Q| per node:
    exact for the amplitude, one forward difference per free shape parameter."""
    lnM = D.node_lnM
    ell, grads = law.ell(theta, lnM, D.node_lz, D.node_cell, need_grad=need_jac)

    def phi_of(ell_):
        return {q: (QSIGN[q] * np.exp(ell_[q]) if q in ell_ else np.full(lnM.shape, law.fixed[q])) for q in QNAMES}

    V = node_value(D.node_kind, D.node_x1, D.node_x2, D.form, D.field, phi_of(ell))
    S = np.add.reduceat(D.node_w * V, D.node_start) if len(V) else np.zeros(0)
    lnm = np.log(S)
    if not need_jac:
        return lnm
    G = {"amp": D.node_w * V}
    for q in law.free_shape:
        e2 = dict(ell); e2[q] = ell[q] + h
        G[q] = D.node_w * (node_value(D.node_kind, D.node_x1, D.node_x2, D.form, D.field, phi_of(e2)) - V) / h
    J = np.zeros((D.n, law.n))
    for j, q, dl in grads:
        J[:, j] += np.add.reduceat(G[q] * dl, D.node_start) / S
    return lnm, J


def _solve(D, law, theta0, sigma_d, lnd, max_nfev, W=None, tol=1e-10):
    W = D.whitener(sigma_d) if W is None else W
    lo, hi = law.bounds()
    x0 = np.clip(theta0, lo + 1e-9 * (hi - lo) * np.isfinite(hi - lo), hi - 1e-9 * (hi - lo) * np.isfinite(hi - lo))

    def fun(th):
        return W @ (lnd - predict(th, D, law))

    def jac(th):
        _, J = predict(th, D, law, need_jac=True)
        return -(W @ J)

    r = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=max_nfev,
                      xtol=tol, ftol=tol, gtol=tol)
    return r


def sigma_for_chi2(D, resid, target):
    """sigma_d with sum_c r_c^T (C_c + (sigma_d^2 + sigma_0^2) I)^-1 r_c = target (0 if already below)."""
    f = lambda s: D.chi2_cells(resid, s).sum() - target   # noqa: E731
    if f(0.0) <= 0:
        return 0.0
    hi = 0.05
    while f(hi) > 0 and hi < 50:
        hi *= 2
    return brentq(f, 0.0, hi, xtol=1e-7, rtol=1e-6)


def fit_global(D, law, theta0=None, sigma_d=None, max_iter=8, max_nfev=600, fixed_sigma=False):
    """Fit law to D.  Unless fixed_sigma, sigma_d is iterated so that chi^2/nu = 1 (nu = n_points - n_par).
    Returns dict: theta, names, sigma_d, chi2, nu, chi2_nu, cov (Hessian, (J^T W^T W J)^-1), lnm, resid, success,
    nfev, history."""
    theta = law.start() if theta0 is None else np.array(theta0, float)
    s = 0.05 if sigma_d is None else float(sigma_d)
    nu = D.n - law.n
    hist = []
    for it in range(max_iter):
        r = _solve(D, law, theta, s, D.lnd, max_nfev)
        theta = r.x
        resid = D.lnd - predict(theta, D, law)
        if fixed_sigma:
            break
        s_new = sigma_for_chi2(D, resid, max(nu, 1))
        hist.append({"iter": it, "sigma_d": s, "chi2": float(D.chi2_cells(resid, s).sum()), "sigma_new": s_new,
                     "nfev": int(r.nfev), "status": int(r.status)})
        if abs(s_new - s) <= 0.005 * s + 1e-6:
            s = s_new
            break
        s = s_new
    r = _solve(D, law, theta, s, D.lnd, max_nfev)
    theta = r.x
    lnm, J = predict(theta, D, law, need_jac=True)
    resid = D.lnd - lnm
    W = D.whitener(s)
    Jw = W @ J
    F = Jw.T @ Jw
    try:
        cov = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F)
    chi2 = float(D.chi2_cells(resid, s).sum())
    lo, hi = law.bounds()
    at_bound = [law.names[i] for i in range(law.n) if (np.isfinite(lo[i]) and theta[i] - lo[i] < 1e-6 * (hi[i] - lo[i]))
                or (np.isfinite(hi[i]) and hi[i] - theta[i] < 1e-6 * (hi[i] - lo[i]))]
    return {"theta": theta, "names": list(law.names), "sigma_d": s, "chi2": chi2, "nu": int(nu),
            "chi2_nu": chi2 / max(nu, 1), "cov": cov, "lnm": lnm, "resid": resid, "success": bool(r.success),
            "nfev": int(r.nfev), "history": hist, "at_bound": at_bound, "n_points": int(D.n), "n_par": int(law.n),
            "n_cells": int(D.n_cells)}


def evaluate_fixed(D, law, theta):
    """Score a fixed parameter vector (e.g. the published law): ln model, residuals and the sigma_d that makes
    chi^2/nu = 1 with nu = n_points."""
    lnm = predict(theta, D, law)
    resid = D.lnd - lnm
    s = sigma_for_chi2(D, resid, D.n)
    return {"lnm": lnm, "resid": resid, "sigma_d": s, "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
            "median_abs_resid": float(np.median(np.abs(resid)))}


def information_criteria(D, resid, sigma_d, k):
    """AIC = chi^2 + 2k and BIC = chi^2 + k ln n at a given (common) sigma_d."""
    chi2 = float(D.chi2_cells(resid, sigma_d).sum())
    return {"chi2_at_sigma": chi2, "AIC": chi2 + 2 * k, "BIC": chi2 + k * math.log(D.n), "sigma_ref": sigma_d}


def bootstrap_global(D, law, fit, n_boot=100, max_nfev=250, which=None, return_status=False):
    """Refit the bootstrap replicates (D.boot[:, b]) at fixed sigma_d, starting from the nominal fit.  Returns
    [n, n_par] parameter replicates (rows NaN where a refit raised or least_squares returned status < 0; a refit that
    stopped at max_nfev, status 0, is kept and flagged in the optional status array)."""
    W = D.whitener(fit["sigma_d"])
    idx = range(min(n_boot, D.boot.shape[1])) if which is None else which
    out, status = [], []
    for b in idx:
        try:
            r = _solve(D, law, fit["theta"], fit["sigma_d"], D.boot[:, b], max_nfev, W=W, tol=1e-8)
            out.append(r.x if r.status >= 0 else np.full(law.n, np.nan))
            status.append(int(r.status))
        except Exception:   # noqa: BLE001 - a failed replicate is recorded as NaN
            out.append(np.full(law.n, np.nan))
            status.append(-9)
    return (np.array(out), np.array(status)) if return_status else np.array(out)


def fit_cells(D, form, max_nfev=200):
    """Per-cell free fits (law 'cell', one cell at a time) with C_boot + sigma_0^2 only (no sigma_d).  Returns a list
    of dicts per cell: theta (ln amp, ln shape...), the three parameters, Hessian errors, chi2, n points."""
    res = []
    for c, pts in enumerate(D.cell_points):
        sub = D.subset(pts)
        law = Law(form, "cell", n_cells=1)
        lnMbar = float(np.average(sub.node_lnM, weights=sub.node_w))
        th0 = law.start(cell_lnM=[lnMbar], cell_z=[float(sub.z[0])])
        item = {"cell": c, "n_points": int(sub.n), "z": float(sub.z[0]), "lnM_mean_nodes": lnMbar}
        if sub.n < 3:
            item.update(ok=False, reason="fewer than 3 points")
            res.append(item)
            continue
        try:
            best = None
            for scale in (1.0, 0.7, 1.4):          # three starts in the shape parameters (degenerate cells)
                t0 = th0.copy()
                for q in law.free_shape:
                    t0[law.index[f"{q}:ln[0]"]] += math.log(scale)
                r = _solve(sub, law, t0, 0.0, sub.lnd, max_nfev)
                if best is None or r.cost < best.cost:
                    best = r
            lnm, J = predict(best.x, sub, law, need_jac=True)
            W = sub.whitener(0.0)
            Jw = W @ J
            try:
                cov = np.linalg.inv(Jw.T @ Jw)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(Jw.T @ Jw)
            resid = sub.lnd - lnm
            lo, hi = law.bounds()
            item.update(ok=bool(best.success), theta=best.x, cov=cov, chi2=float(sub.chi2_cells(resid, 0.0).sum()),
                        names=law.names, resid=resid,
                        at_bound=[law.names[i] for i in range(law.n) if best.x[i] - lo[i] < 1e-6 * (hi[i] - lo[i]) or hi[i] - best.x[i] < 1e-6 * (hi[i] - lo[i])])
        except Exception as exc:   # noqa: BLE001
            item.update(ok=False, reason=repr(exc))
        res.append(item)
    return res


# ---------------------------------------------------------------------------------------------------- reporting
def law_table(law, theta, cov=None, boot=None):
    """Human/XGPaint-ready table of a law: per profile parameter its kind and numbers.  Uncertainties: 'hess' from
    cov (delta method), 'boot' = standard deviation of the bootstrap replicates; amplitudes A = exp(lnA) with
    sigma_A = A sigma_lnA."""
    th = np.asarray(theta, float)
    sd_h = np.sqrt(np.clip(np.diag(cov), 0, None)) if cov is not None else np.full(len(th), np.nan)
    sd_b = np.nanstd(boot, axis=0, ddof=1) if boot is not None and len(boot) > 1 else np.full(len(th), np.nan)
    out = {"form": law.form, "kind": law.kind, "pivot_Msun": PIVOT_MSUN, "params": {}}
    for q in QNAMES:
        k = law.qkind[q]
        entry = {"kind": k}
        if k == "fixed":
            entry["value"] = law.fixed[q]
        elif k in ("pl", "bpl"):
            for i, n in enumerate(law.names):
                if not n.startswith(q + ":") or n == "amp:abrk":
                    continue
                what = n.split(":")[1]
                if what in ("lnA", "lnQb"):
                    key = "A" if what == "lnA" else "Q_b"
                    val = QSIGN[q] * math.exp(th[i])
                    entry[key] = val
                    entry[key + "_err_hess"] = abs(val) * sd_h[i]
                    entry[key + "_err_boot"] = abs(val) * sd_b[i]
                else:
                    key = {"am": "alpha_m", "az": "alpha_z", "alo": "alpha_m_low", "ahi": "alpha_m_high"}[what]
                    entry[key] = th[i]
                    entry[key + "_err_hess"] = sd_h[i]
                    entry[key + "_err_boot"] = sd_b[i]
        out["params"][q] = entry
    if "log10Mb" in law.index:
        i = law.index["log10Mb"]
        out["log10_M_break"] = th[i]
        out["log10_M_break_err_hess"] = sd_h[i]
        out["log10_M_break_err_boot"] = sd_b[i]
    if "amp:abrk" in law.index:
        i = law.index["amp:abrk"]
        out["alpha_break"] = th[i]
        out["alpha_break_err_hess"] = sd_h[i]
        out["alpha_break_err_boot"] = sd_b[i]
    return out


def correlation(cov):
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return cov / np.outer(sd, sd)


def law_profile_params(law, theta, M_msun, z):
    """Five profile parameters at arbitrary (M, z) (arrays broadcast)."""
    M, z = np.broadcast_arrays(np.asarray(M_msun, float), np.asarray(z, float))
    return law.phi(theta, np.log(M), np.log1p(z))


def law_profile(law, theta, x, M_msun, z):
    """F(x) at one (M, z)."""
    p = law_profile_params(law, theta, M_msun, z)
    return profile(np.asarray(x, float), law.form, *(np.asarray(p[q]).item() if np.ndim(p[q]) == 0 else p[q] for q in QNAMES))
