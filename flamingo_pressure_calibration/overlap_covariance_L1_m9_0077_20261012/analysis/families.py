"""Profile families for the general fitter (``fitting.fit_general``).

Every family maps natural parameters p to F(x) and to d ln F / d theta, where
theta_k = ln p_k for positive ("log") parameters and theta_k = p_k for signed
("linear") ones.  Shell averages use radii234 ``models.shell_grid`` exactly as
the legacy B12/B16 fits.  Families:

``b12``      0.5176 P0 q^-0.3 (1+q)^-beta, q = x/xc               (radii234 models.shape)
``b16``      rho0 q^-0.2 (1+q^alpha)^-((beta-0.2)/alpha), q = x/0.5 (radii234 models.shape)
``gnfw5``    P0 q^-gamma (1+q^alpha)^-((beta-gamma)/alpha), q = x/xc (Nagai 2007 form; all five free;
             beta is the outer log slope).  B12 is the special case
             gnfw5(0.5176 P0_b12, xc, alpha = 1, beta = beta_b12 + 0.3, gamma = 0.3).
``b12env``   B12(x) + A (x/x_p)^-s, x_p = 2 by default (two components: core + environment);
             A > 0 fitted in ln, s signed (linear, -2..10; s = 0 is a constant floor)
``b12spline``B12(x) [1 + s(ln x)], s a natural cubic spline through K knot values c_k (linear
             parameters), held constant beyond the end knots; ridge-regularised with penalty
             residuals c_k / prior_sigma (built by ``b12_spline_family(knots_x, prior_sigma)``).
Bounds are in natural units; acceptance flags are those of ``fitting.accepted``.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from . import _legacy

M = _legacy.models()
P = _legacy.pilot_physics()


class Family:
    """A profile family: names, bounds (natural units), transform flags, shape and derivatives."""

    def __init__(self, name, names, lower, upper, log, start, shape, dln, penalty=None, info=None):
        self.name = name
        self.names = tuple(names)
        self.lower = np.asarray(lower, float)
        self.upper = np.asarray(upper, float)
        self.log = np.asarray(log, bool)
        self.start = np.asarray(start, float)
        self.shape = shape          # (x, p) -> F
        self.dln = dln              # (x, p) -> [k, *x.shape] d ln F / d theta_k
        self.penalty = penalty      # (p) -> (residuals [m], d residuals / d theta [m, k]) or None
        self.info = info or {}

    @property
    def n(self) -> int:
        return len(self.names)

    def theta_bounds(self):
        lo = np.where(self.log, np.log(np.maximum(self.lower, 1e-300)), self.lower)
        hi = np.where(self.log, np.log(self.upper), self.upper)
        return lo, hi

    def to_theta(self, p):
        p = np.asarray(p, float)
        return np.where(self.log, np.log(np.maximum(p, 1e-300)), p)

    def to_params(self, theta):
        theta = np.asarray(theta, float)
        return np.where(self.log, np.exp(theta), theta)


# ----------------------------------------------------------------------------- B12 / B16
def _b12_dln(x, p):
    P0, xc, beta = p[:3]
    q = np.asarray(x, float) / xc
    out = np.empty((3,) + q.shape)
    out[0] = 1.0
    out[1] = .3 + beta * q / (1 + q)
    out[2] = -beta * np.log1p(q)
    return out


def _b16_dln(x, p):
    rho0, alpha, beta = p[:3]
    q = np.asarray(x, float) / .5
    qa = q**alpha
    out = np.empty((3,) + q.shape)
    out[0] = 1.0
    out[1] = (beta - .2) / alpha * np.log1p(qa) - (beta - .2) * qa / (1 + qa) * np.log(q)
    out[2] = -beta / alpha * np.log1p(qa)
    return out


B12 = Family("b12", ("P0", "xc", "beta"), M.BOUNDS["b12"][0], M.BOUNDS["b12"][1], [True] * 3,
             P.B12[:3], lambda x, p: M.shape("b12", x, p[:3]), _b12_dln)
B16 = Family("b16", ("rho0", "alpha", "beta"), M.BOUNDS["b16"][0], M.BOUNDS["b16"][1], [True] * 3,
             P.B16[:3], lambda x, p: M.shape("b16", x, p[:3]), _b16_dln)


# ----------------------------------------------------------------------------- gNFW (5 free)
def _gnfw_shape(x, p):
    P0, xc, alpha, beta, gamma = p
    q = np.asarray(x, float) / xc
    return P0 * q**(-gamma) * (1 + q**alpha)**(-(beta - gamma) / alpha)


def _gnfw_dln(x, p):
    P0, xc, alpha, beta, gamma = p
    q = np.asarray(x, float) / xc
    qa = q**alpha
    lq = np.log(q)
    l1 = np.log1p(qa)
    frac = qa / (1 + qa)
    out = np.empty((5,) + q.shape)
    out[0] = 1.0
    out[1] = gamma + (beta - gamma) * frac
    out[2] = (beta - gamma) / alpha * l1 - (beta - gamma) * frac * lq
    out[3] = -beta / alpha * l1
    out[4] = gamma * (-lq + l1 / alpha)
    return out


GNFW5 = Family("gnfw5", ("P0", "xc", "alpha", "beta", "gamma"),
               [1e-3, 0.01, 0.1, 0.5, 1e-3], [1e5, 10., 10., 20., 3.], [True] * 5,
               [0.5176 * P.B12[0], P.B12[1], 1.0, P.B12[2] + 0.3, 0.3], _gnfw_shape, _gnfw_dln,
               info={"note": "B12(P0, xc, beta) = gnfw5(0.5176 P0, xc, 1, beta + 0.3, 0.3)"})


# ----------------------------------------------------------------------------- B12 + environment
def b12_env_family(pivot: float = 2.0) -> Family:
    def shape(x, p):
        x = np.asarray(x, float)
        return M.shape("b12", x, p[:3]) + p[3] * (x / pivot)**(-p[4])

    def dln(x, p):
        x = np.asarray(x, float)
        core = M.shape("b12", x, p[:3])
        env = p[3] * (x / pivot)**(-p[4])
        total = core + env
        out = np.empty((5,) + x.shape)
        out[:3] = _b12_dln(x, p) * (core / total)
        out[3] = env / total
        out[4] = -np.log(x / pivot) * env / total          # s_env is a linear (signed) parameter
        return out

    lo, hi = M.BOUNDS["b12"]
    return Family("b12env", ("P0", "xc", "beta", "A_env", "s_env"), list(lo) + [1e-8, -2.0],
                  list(hi) + [1e2, 10.], [True] * 4 + [False], list(P.B12[:3]) + [1e-3, 1.0], shape, dln,
                  info={"pivot_x": pivot, "note": "s_env linear in [-2, 10]: s < 0 is a rising outer term"})


B12ENV = b12_env_family()


# ----------------------------------------------------------------------------- B12 x (1 + spline)
def b12_spline_family(knots_x, prior_sigma: float = 0.1, knot_bounds=(-0.6, 3.0)) -> Family:
    """B12 x [1 + s(ln x)] with a natural cubic spline through K knot values (K = 3-4).

    The spline is held constant beyond the end knots; penalty residuals c_k/prior_sigma
    (a Gaussian prior of width ``prior_sigma`` on each knot value) regularise it.
    """
    knots_u = np.log(np.asarray(knots_x, float))
    K = len(knots_u)
    bases = [CubicSpline(knots_u, np.eye(K)[k], bc_type="natural") for k in range(K)]

    def basis(x):
        u = np.clip(np.log(np.asarray(x, float)), knots_u[0], knots_u[-1])
        return np.stack([b(u) for b in bases])            # [K, *x.shape]

    def shape(x, p):
        return M.shape("b12", x, p[:3]) * np.maximum(1 + np.tensordot(p[3:], basis(x), axes=1), 1e-6)

    def dln(x, p):
        B = basis(x)
        corr = np.maximum(1 + np.tensordot(p[3:], B, axes=1), 1e-6)
        out = np.empty((3 + K,) + np.shape(x))
        out[:3] = _b12_dln(x, p)
        out[3:] = B / corr
        return out

    def penalty(p):
        c = np.asarray(p[3:], float)
        jac = np.zeros((K, 3 + K))
        jac[:, 3:] = np.eye(K) / prior_sigma
        return c / prior_sigma, jac

    lo, hi = M.BOUNDS["b12"]
    return Family("b12spline", ("P0", "xc", "beta") + tuple(f"c{k}" for k in range(K)),
                  list(lo) + [knot_bounds[0]] * K, list(hi) + [knot_bounds[1]] * K,
                  [True] * 3 + [False] * K, list(P.B12[:3]) + [0.0] * K, shape, dln, penalty,
                  info={"knots_x": np.asarray(knots_x, float).tolist(), "prior_sigma": prior_sigma})


REGISTRY = {"b12": B12, "b16": B16, "gnfw5": GNFW5, "b12env": B12ENV}


def get(family) -> Family:
    if isinstance(family, Family):
        return family
    return REGISTRY[family]


# ----------------------------------------------------------------------------- shell averages
def shell_grid(edges):
    return M.shell_grid(tuple(np.asarray(edges, float).tolist()))


def shell_model(family, params, edges):
    """Shell averages of F on every shell (same quadrature as models.shell)."""
    fam = get(family)
    x, w = shell_grid(edges)
    return np.sum(fam.shape(x, np.asarray(params, float)) * w, axis=1)


def shell_model_jacobian(family, params, edges):
    """S [shell] and d ln S / d theta [shell, k] (theta: ln for log parameters)."""
    fam = get(family)
    p = np.asarray(params, float)
    x, w = shell_grid(edges)
    Fw = fam.shape(x, p) * w
    S = np.sum(Fw, axis=1)
    D = fam.dln(x, p)
    return S, np.einsum("ksn,sn->sk", D, Fw) / S[:, None]


def shell_model_truncated(family, params, edges, cut: float, order: int = 32):
    """Shell averages of F(x) Theta(x < cut) over the *full* shell volumes.

    Shells beyond the cut are 0; a shell split by the cut is integrated exactly on
    [lo, cut] with its own Gauss-Legendre rule and normalised by the full shell volume.
    """
    fam = get(family)
    edges = np.asarray(edges, float)
    S = shell_model(fam, params, edges)
    lo, hi = edges[:-1], edges[1:]
    S = np.where(lo >= cut, 0.0, S)
    split = np.flatnonzero((lo < cut) & (hi > cut))
    if len(split):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        for s in split:
            a, b = lo[s], cut
            x = a + (b - a) * (nodes + 1) / 2
            w = 3 * x * x * weights * (b - a) / (2 * (hi[s]**3 - lo[s]**3))
            S[s] = np.sum(fam.shape(x, np.asarray(params, float)) * w)
    return S
