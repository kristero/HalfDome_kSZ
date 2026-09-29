#!/usr/bin/env python3
"""Independent Python tSZ halo model that uses CLASS-SZ's OWN halo-model ingredients (Tinker08 M200c dn/dlnM,
Tinker10 bias, linear P(k), D_A, H, rho_crit, all tabulated by run_class_sz.py from the b12_paint_all run) but its
own pressure profiles and Fourier transforms. Validated on Battaglia12 against CLASS-SZ, then used for the Lee22
profile exactly as painted (both mass pieces in one integral, so the 2-halo cross term between them is included),
for the XGPaint-like projected aperture, the catalogue-weighted terms and the mass/redshift decomposition.

Units: M in Msun/h (h = 0.68, M_phys = M/h), chi in Mpc/h, n in (h/Mpc)^3, P_lin in (Mpc/h)^3, H/c in h/Mpc;
profiles in physical cgs. y_ell = sigma_T/(m_e c^2) P200 R200 (R200/D_A)^2 P0_eff u(q), q = (ell+1/2) R200/D_A with
  sphere   : u(q) = 4 pi int_0^X x^2 g(x) sin(qx)/(qx) dx                       (CLASS-SZ x_outSZ = X)
  aperture : u(q) = 2 pi int_0^X x Sigma(x) J0(qx) dx, Sigma(x) = 2 int_0^inf g(sqrt(x^2+l^2)) dl  (XGPaint map)
g(x) = (x/xc)^-0.3 (1 + x/xc)^-beta, P200 = 200 G M rho_cr(z) f_b / (2 R200) (physical R200, thermal Battaglia P200),
P0_eff = 0.5176 P0_B12 (B12 fits P_th) or P0_Lee (Lee22 fits P_e).
C1h = int dz chi^2/(H/c) int dlnM n |y|^2 ;  C2h = int dz chi^2/(H/c) [int dlnM n b y]^2 P_lin((l+1/2)/chi, z)
"""
import sys
import numpy as np
from scipy import interpolate, special

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_tsz_verification/B_class_sz")
sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/lee22_normalization_forensics/C_pressure_consistency")
import common as C  # noqa: E402
import lee22_models as LM  # noqa: E402  (validated formulas; imported, not modified)

SIGMA_T = 6.6524587321e-25     # cm^2
ME_C2 = 8.1871057769e-7        # erg
G = LM.G
MSUN = LM.MSUN
MPC = LM.MPC
H0_PAINT = C.PAINT["h"]
FB_PAINT = C.PAINT["Omega_b"] / (C.PAINT["Omega_b"] + C.PAINT["Omega_c"])


# ------------------------------------------------------------------------------------------ quadrature
_GL = {}


def gl01(n, order=32):
    """n-node composite Gauss-Legendre rule on [0, 1] (panels of `order` nodes)."""
    key = (n, order)
    if key not in _GL:
        if n <= order:
            t, w = np.polynomial.legendre.leggauss(n)
            _GL[key] = (0.5 * (t + 1), 0.5 * w)
        else:
            p = int(np.ceil(n / order))
            t, w = np.polynomial.legendre.leggauss(order)
            t = 0.5 * (t + 1) / p; w = 0.5 * w / p
            e = np.arange(p) / p
            _GL[key] = ((e[:, None] + t[None, :]).ravel(), np.tile(w, p))
    return _GL[key]


def gl_segments(edges, n_per):
    """Gauss-Legendre nodes/weights on consecutive segments (edges ascending), n_per nodes each."""
    x, w = [], []
    t, tw = np.polynomial.legendre.leggauss(n_per)
    for a, b in zip(edges[:-1], edges[1:]):
        x.append(0.5 * (b - a) * (t + 1) + a); w.append(0.5 * (b - a) * tw)
    return np.concatenate(x), np.concatenate(w)


# ------------------------------------------------------------------------------------------ profiles
def params(model, M_hinv, z, h=H0_PAINT):
    """(P0_eff, xc, beta) of P_e = P0_eff P200 (x/xc)^-0.3 (1+x/xc)^-beta for M200c in Msun/h (paint h).
    model: 'b12' | 'lee22' (as painted: Table 7, BPL P0, shape frozen above 10^14.8/h) | 'lee22_noclip' |
           'lee22_m1' (lee22 x (1+z)^2) | 'hybrid' (lee22 below 10^14.8/h, b12 above)."""
    M_hinv = np.asarray(M_hinv, float); z = np.asarray(z, float)
    Mphys = M_hinv / h
    if model == "b12":
        m = Mphys / 1e14; z1 = 1 + z
        return (C.PE_OVER_PTH * 18.1 * m ** 0.154 * z1 ** -0.758, 0.497 * m ** -0.00865 * z1 ** 0.731,
                4.35 * m ** 0.0393 * z1 ** 0.415)
    if model in ("lee22", "lee22_m1"):
        P0, xc, beta = LM.lee_P_noc_params(M_hinv, z, shape_clip_hinv=C.LEE_CLIP_HINV)
        if model == "lee22_m1":
            P0 = P0 * (1 + z) ** 2
        return P0, xc, beta
    if model == "lee22_noclip":
        return LM.lee_P_noc_params(M_hinv, z, shape_clip_hinv=np.inf)
    if model == "hybrid":
        a = params("lee22", M_hinv, z, h); b = params("b12", M_hinv, z, h)
        hi = M_hinv >= C.LEE_CLIP_HINV
        return tuple(np.where(hi, bb, aa) for aa, bb in zip(a, b))
    raise ValueError(model)


class ProjectedTable:
    """Y(u; beta) = 2 int_0^inf (u^2+t^2)^-0.15 (1+sqrt(u^2+t^2))^-beta dt, so Sigma(x) = xc Y(x/xc; beta)."""

    def __init__(self, beta_min=2.5, beta_max=45.0):
        lnu = np.linspace(np.log(1e-7), np.log(40.0), 500)
        beta = np.linspace(beta_min, beta_max, 200)
        s, w = gl01(2048)
        lt_lo, lt_hi = np.log(1e-12), np.log(1e6)
        t = np.exp(s * (lt_hi - lt_lo) + lt_lo); wt = (lt_hi - lt_lo) * w * t
        out = np.empty((lnu.size, beta.size))
        for i, lu in enumerate(lnu):
            r = np.sqrt(np.exp(2 * lu) + t ** 2)
            base = r ** -0.3 * wt
            out[i] = 2 * (base[None, :] * np.exp(-beta[:, None] * np.log1p(r)[None, :])).sum(1)
        self.lnu, self.beta = lnu, beta
        self.spl = interpolate.RectBivariateSpline(lnu, beta, np.log(out), kx=3, ky=3)

    def __call__(self, u, beta):
        lu = np.log(np.clip(u, np.exp(self.lnu[0]), None))
        if lu.max() > self.lnu[-1] or np.min(beta) < self.beta[0] or np.max(beta) > self.beta[-1]:
            raise ValueError("projected table range exceeded")
        return np.exp(self.spl.ev(lu, beta))


_PTAB = None


def ptab():
    global _PTAB
    if _PTAB is None:
        _PTAB = ProjectedTable()
    return _PTAB


def u_transform(xc, beta, q, kind="sphere", X=4.0, chunk=16):
    """u(q) for rows of (xc, beta) (n,) and q (n, nl)."""
    n, nl = q.shape
    out = np.empty((n, nl))
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        nn = int(min(16384, max(256, 3.2 * q[a:b].max() * X + 96)))
        nn = int(2 ** np.ceil(np.log2(nn)))
        s, w = gl01(nn)
        x = X * s ** 2; jac = 2 * X * s * w
        xcb, bb = xc[a:b, None], beta[a:b, None]
        if kind == "sphere":
            g = (x[None, :] / xcb) ** -0.3 * (1 + x[None, :] / xcb) ** -bb
            integ = 4 * np.pi * x[None, :] ** 2 * jac[None, :] * g
            ker = np.sinc(q[a:b, :, None] * x[None, None, :] / np.pi)
        elif kind == "aperture":
            sig = xcb * ptab()(x[None, :] / xcb, np.broadcast_to(bb, (b - a, nn)))
            integ = 2 * np.pi * x[None, :] * jac[None, :] * sig
            ker = special.j0(q[a:b, :, None] * x[None, None, :])
        else:
            raise ValueError(kind)
        out[a:b] = np.einsum("mln,mn->ml", ker, integ)
    return out


# ------------------------------------------------------------------------------------------ CLASS-SZ ingredients
class Ingredients:
    def __init__(self, path=C.OUT + "/class_sz/b12_paint_all.npz"):
        d = np.load(path, allow_pickle=True)
        self.h = H0_PAINT
        self.ell = d["ell"]
        self.tz, self.tlnm = d["tab_z"], d["tab_lnm_msunh"]
        self._lnn = interpolate.RectBivariateSpline(self.tz, self.tlnm, np.log(d["tab_dndlnm_h3mpc3"]), kx=3, ky=3)
        self._b = interpolate.RectBivariateSpline(self.tz, self.tlnm, d["tab_bias"], kx=3, ky=3)
        # CLASS-SZ returns P_lin = 0 beyond its k_max (only at z < 0.01 and ell > ~3000, where the 2-halo term is
        # irrelevant); floor it so the log-spline stays finite (the floored entries contribute nothing)
        pk = np.maximum(d["tab_pklin_mpch3"], 1e-30)
        self._lnpk = interpolate.RectBivariateSpline(self.tz, np.log(self.ell + 0.5), np.log(pk), kx=1, ky=1)
        self.n_pk_zero = int((d["tab_pklin_mpch3"] <= 0).sum())
        self._da = interpolate.CubicSpline(self.tz, d["tab_da_mpc"])
        self._hub = interpolate.CubicSpline(self.tz, d["tab_H_over_c_invmpc"])
        self._rc = interpolate.CubicSpline(self.tz, np.log(d["tab_rho_crit_h2msun_mpc3"]))
        self.lnm_lo, self.lnm_hi = self.tlnm[0], self.tlnm[-1]

    def dndlnm(self, z, lnm):          # (h/Mpc)^3, grid evaluation (nz, nm)
        return np.exp(self._lnn(z, lnm))

    def bias(self, z, lnm):
        return self._b(z, lnm)

    def pk(self, z, ell):              # P_lin((ell+1/2)/chi, z) in (Mpc/h)^3, (nz, nl)
        return np.exp(self._lnpk(z, np.log(ell + 0.5)))

    def da(self, z):                   # Mpc
        return self._da(z)

    def chi_h(self, z):                # Mpc/h
        return self._da(z) * (1 + z) * self.h

    def hub_h(self, z):                # H/c in h/Mpc
        return self._hub(z) / self.h

    def rho_crit_cgs(self, z):
        return np.exp(self._rc(z)) * self.h ** 2 * MSUN / MPC ** 3


def y_ell_grid(ing, model, z, lnm, ell, kind="sphere", X=4.0):
    """y_ell(z, M) (nz, nm, nl) for M = exp(lnm) Msun/h."""
    h = ing.h
    M = np.exp(lnm)
    out = np.empty((z.size, M.size, ell.size))
    for j, zz in enumerate(z):
        rc = ing.rho_crit_cgs(zz)
        r200 = (3 * (M / h) * MSUN / (4 * np.pi * 200 * rc)) ** (1 / 3)            # cm, physical
        da = ing.da(zz) * MPC
        p200 = 200 * G * (M / h) * MSUN * rc * FB_PAINT / (2 * r200)              # erg/cm^3 (thermal P200)
        P0, xc, beta = params(model, M, np.full_like(M, zz), h)
        q = (ell[None, :] + 0.5) * (r200 / da)[:, None]
        u = u_transform(np.asarray(xc, float), np.asarray(beta, float), q, kind=kind, X=X)
        out[j] = (SIGMA_T / ME_C2 * p200 * r200 * (r200 / da) ** 2 * P0)[:, None] * u
    return out


def z_nodes(z_lo, z_hi, n_per=10):
    """Segmented GL nodes in ln(1+z)-ish panels, finer at low z; returns z, dz-weights."""
    cuts = [0.002141595, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25,
            1.5, 1.75, 2.0, 2.4, 2.8, 3.3, 3.8554277]
    e = sorted(set([z_lo, z_hi] + [c for c in cuts if z_lo < c < z_hi]))
    return gl_segments(np.array(e), n_per)


def lnm_nodes(m_lo, m_hi, extra=(), n_per=10, dlnm=0.35):
    br = [m_lo, m_hi] + [x for x in extra if m_lo < x < m_hi]
    br = np.log(np.array(sorted(set(br))))
    e = [br[0]]
    for a, b in zip(br[:-1], br[1:]):
        k = int(np.ceil((b - a) / dlnm))
        e += list(np.linspace(a, b, k + 1)[1:])
    return gl_segments(np.array(e), n_per)


BREAKS = (1e13, C.LEE_P_MCUT_HINV, 3e13, 1e14, 3e14, C.LEE_CLIP_HINV)


def halo_model_cl(ing, model, kind="sphere", X=4.0, m_lo=C.CAT_M_MIN, m_hi=C.CAT_M_MAX, z_lo=C.CAT_Z_MIN,
                  z_hi=C.CAT_Z_MAX, ell=None, keep=False, n_z=10, n_m=10):
    """C1h, C2h on the CLASS-SZ ell grid; keep=True also returns per-node contributions for decompositions."""
    ell = ing.ell if ell is None else ell
    z, wz = z_nodes(z_lo, z_hi, n_z)
    lnm, wm = lnm_nodes(m_lo, m_hi, BREAKS, n_m)
    y = y_ell_grid(ing, model, z, lnm, ell, kind, X)                     # (nz, nm, nl)
    n = ing.dndlnm(z, lnm); b = ing.bias(z, lnm)                          # (nz, nm)
    vol = ing.chi_h(z) ** 2 / ing.hub_h(z)                                 # (Mpc/h)^3 per unit z per sr
    pk = ing.pk(z, ell)                                                    # (nz, nl)
    d1 = (wz * vol)[:, None, None] * (wm * n)[:, :, None] * y ** 2        # 1h contribution per (z, M) node
    A = np.einsum("jm,jml->jl", wm * n * b, y)                             # (nz, nl)
    d2z = (wz * vol)[:, None] * A ** 2 * pk                               # 2h per z node
    res = dict(ell=ell, cl_1h=d1.sum((0, 1)), cl_2h=d2z.sum(0))
    if keep:
        # 2h mass attribution: node i gets (wm n b y)_i * A * P * vol (sums exactly to the 2h term)
        d2 = (wz * vol)[:, None, None] * (wm * n * b)[:, :, None] * y * (A * pk)[:, None, :]
        res.update(z=z, lnm=lnm, d1=d1, d2=d2, d2z=d2z, y=y, wz=wz, wm=wm)
    return res
