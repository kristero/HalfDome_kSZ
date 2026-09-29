#!/usr/bin/env python3
"""verify_5: independent catalogue-weighted 1-halo sums for kSZ and tSZ over the exact HalfDome lightcone.

Written from scratch for the adversarial check of claim C6 (does not import F_sz_budget/szhm.py or the
analytic engine). No mass function: the 85,224,251 painted halos are binned in (log10 M200c, z) with their
counts and sum of v_los^2, and each bin is evaluated at its count-weighted mean (log M, z).

Cosmology: paint contract h=0.68, Omega_m=0.31, Omega_b=0.049 (astropy FlatLambdaCDM, Tcmb 2.7255, Neff 3.046).
tSZ : y(theta) = sigma_T/(m_e c^2) A P200 R200 Sigma(theta/theta200), full LOS, aperture 4 theta200 (painted maps)
      Sigma(X) = 2 int_0^inf f(sqrt(X^2+t^2)) dt, f = (x/xc)^-0.3 (1 + x/xc)^-b   (alpha = 1, gamma = -0.3)
      B12 : A = 0.5176 x 18.1 m^0.154 (1+z)^-0.758, xc = 0.497 m^-0.00865 (1+z)^0.731, b = 4.35 m^0.0393 (1+z)^0.415
            (XGPaint get_params converts beta so that the painted exponent is -beta_raw, as printed by Battaglia12)
      Lee22p (Table 7): A = 2.8 (M/Mcut)^{1.08|0.80} (1+z)^-1.89, xc = 2.10 (M/Mcut)^-0.35 (1+z)^0.53,
            b = 9.4 (M/Mcut)^-0.06 (1+z)^0.60, Mcut = 10^13.60/h, shape frozen above 10^14.8/h
      P200 = 200 G M rho_cr(z) f_b / (2 R200), physical.
      y_l = 2 pi theta200^2 int_0^4 X y(X) J0(l theta200 X) dX ;  C_l^1h = (1/4pi) sum N |y_l|^2
kSZ : 3R200c sphere; tau_l = sigma_T R200^3/D_A^2 n_e0 int_0^3 4 pi x^2 g(x) j0(l theta200 x) dx
      Lee22 literal (Table 8, eq. 9 printed): n_e0 = n0 200 rho_cr f_b/(X_H m_p), g = (x/xc)^-0.3 (1+x/xc)^-beta'
        n0 = 6.8 (M/Mcut)^0.68 (1+z)^-2.11, xc = 7.9 (1+z)^-0.67 (M/Mcut)^{0.47|-0.45}, beta' = 19.5 (1+z)^-0.31
        (M/Mcut)^{0.70|-0.18}, Mcut = 10^13.61/h, shape frozen above 10^14.8/h
      B16 (XGPaint BattagliaTauProfile defaults): n_e = 0.9 f_b rho_cr P0 (x/xc)^g (1+(x/xc)^a)^(-(b+g)/a) / m_per_e
        P0 = 4000 m^0.29 (1+z)^-0.66, xc=0.5, a = 0.88 m^-0.03 (1+z)^0.19, b = 3.83 m^0.04 (1+z)^-0.025, g=-0.2
      C_l^1h = T_CMB^2/(4 pi) sum (v_los/c)^2 |tau_l|^2
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate, special

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/verify_5")
CAT = "/home/cbllover/HalfDome/lightcone_100.hdf5"
H, OM, OB = 0.68, 0.31, 0.049
FB = OB / OM
COS = FlatLambdaCDM(H0=100 * H, Om0=OM, Tcmb0=2.7255, Neff=3.046, Ob0=OB)
# cgs
SIGMA_T, MEC2, G, MSUN, MPC, MP, ME = 6.6524587321e-25, 8.1871057769e-7, 6.67430e-8, 1.98841e33, 3.0856775814913673e24, 1.67262192369e-24, 9.1093837015e-28
C_KMS, TCMB_UK, XH = 299792.458, 2.7255e6, 0.76
M_PER_E = ME + 2 * XH / (1 + XH) * MP + (1 - XH) / (2 * (1 + XH)) * 4 * MP
CUT = 10 ** 14.8 / H
ELLS = np.array([30, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000], float)
C0_TNG, K_TNG = 0.2841790493880425, 2.483190184608004


def rho_crit(z):
    return COS.critical_density(z).to("g/cm3").value


def gl_nodes(n, panels):
    t, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * (t + 1) / panels; w = 0.5 * w / panels
    e = np.arange(panels) / panels
    return (e[:, None] + t[None, :]).ravel(), np.tile(w, panels)


S_NODES, S_W = gl_nodes(32, 64)            # 2048 nodes on [0,1] in s, X = Xmax s^2


class ProjTable:
    """S(u; b) = 2 int_0^inf (u^2+s^2)^-0.15 (1+sqrt(u^2+s^2))^-b ds, by trapezoid in ln s (independent of F)."""

    def __init__(self):
        lnu = np.linspace(np.log(1e-7), np.log(60.0), 500)
        b = np.linspace(2.0, 48.0, 300)
        lns = np.linspace(np.log(1e-14), np.log(1e7), 6000)
        s = np.exp(lns); ds = np.gradient(lns)
        out = np.empty((lnu.size, b.size))
        for i, lu in enumerate(lnu):
            r = np.sqrt(np.exp(2 * lu) + s * s)
            base = r ** -0.3 * s * ds
            lg = np.log1p(r)
            out[i] = 2 * (base[None, :] * np.exp(-b[:, None] * lg[None, :])).sum(1)
        self.spl = interpolate.RectBivariateSpline(lnu, b, np.log(out), kx=3, ky=3)
        self.lnu, self.b = lnu, b

    def __call__(self, u, b):
        lu = np.log(np.clip(u, 1e-7, None))
        assert lu.max() <= self.lnu[-1] + 1e-9 and b.min() >= self.b[0] and b.max() <= self.b[-1], (lu.max(), b.min(), b.max())
        return np.exp(self.spl.ev(lu.ravel(), np.broadcast_to(b, u.shape).ravel())).reshape(u.shape)


def bpl(r, lo, hi):
    return np.where(r < 1, r ** lo, r ** hi)


def params_b12(M, z):
    m = M / 1e14; z1 = 1 + z
    return 0.5176 * 18.1 * m ** 0.154 * z1 ** -0.758, 0.497 * m ** -0.00865 * z1 ** 0.731, 4.35 * m ** 0.0393 * z1 ** 0.415


def params_lee22p(M, z):
    mc = 10 ** 13.60 / H; z1 = 1 + z
    r = M / mc; rs = np.minimum(M, CUT) / mc
    return 2.8 * bpl(r, 1.08, 0.80) * z1 ** -1.89, 2.10 * rs ** -0.35 * z1 ** 0.53, 9.4 * rs ** -0.06 * z1 ** 0.60


def params_lee22n(M, z):
    mc = 10 ** 13.61 / H; z1 = 1 + z
    r = M / mc; rs = np.minimum(M, CUT) / mc
    return 6.8 * r ** 0.68 * z1 ** -2.11, 7.9 * z1 ** -0.67 * bpl(rs, 0.47, -0.45), 19.5 * z1 ** -0.31 * bpl(rs, 0.70, -0.18)


def read_catalogue():
    t0 = time.time()
    with h5py.File(CAT, "r") as f:
        M = f["halo_mass_m200c"][:] / H
        z = f["redshift"][:].astype(np.float64)
        pos = f["Position"][:]
        vel = f["Velocity"][:]
    r = np.sqrt((pos.astype(np.float64) ** 2).sum(1))
    vlos = (pos * vel).sum(1, dtype=np.float64) / r
    del pos, vel
    print("catalogue read", round(time.time() - t0, 1), "s; N =", M.size, "rms v_los", np.sqrt(np.mean(vlos ** 2)), flush=True)
    return M, z, vlos


def bin_catalogue(M, z, vlos):
    lme = np.arange(12.80, 15.62, 0.02)
    ze = np.geomspace(0.002, 3.9, 361)
    lm = np.log10(M)
    im = np.digitize(lm, lme) - 1; iz = np.digitize(z, ze) - 1
    assert im.min() >= 0 and im.max() < lme.size - 1 and iz.min() >= 0 and iz.max() < ze.size - 1
    key = iz * (lme.size - 1) + im
    nb = (ze.size - 1) * (lme.size - 1)
    N = np.bincount(key, minlength=nb).astype(float)
    V2 = np.bincount(key, weights=vlos ** 2, minlength=nb)
    SL = np.bincount(key, weights=lm, minlength=nb)
    SZ = np.bincount(key, weights=z, minlength=nb)
    occ = N > 0
    return dict(N=N[occ], V2=V2[occ], lm=SL[occ] / N[occ], z=SZ[occ] / N[occ], M=10 ** (SL[occ] / N[occ]))


def transforms(b, table, chunk=400):
    """Per-bin y_l (B12, Lee22p) and tau_l (Lee22 literal, B16), arrays (nbin, nell)."""
    M, z = b["M"], b["z"]
    nb = M.size
    DA = COS.angular_diameter_distance(z).value * MPC            # cm
    rc = rho_crit(z)
    R200 = (3 * M * MSUN / (4 * np.pi * 200 * rc)) ** (1 / 3)  # cm, physical
    th = R200 / DA
    P200 = 200 * G * M * MSUN * rc * FB / (2 * R200)
    out = {k: np.empty((nb, ELLS.size)) for k in ("y_b12", "y_lee22p", "tau_lit", "tau_b16", "tau_lit_1r", "tau_b16_1r")}
    out.update(theta200=th, R200_cm=R200, DA_cm=DA)
    # tSZ aperture X in [0,4], nodes X = 4 s^2
    Xy = 4 * S_NODES ** 2; wy = 8 * S_NODES * S_W
    Xk = 3 * S_NODES ** 2; wk = 6 * S_NODES * S_W
    for a in range(0, nb, chunk):
        sl = slice(a, min(nb, a + chunk))
        q = ELLS[None, :] * th[sl, None]                                  # (c, nell)
        J = special.j0(q[:, :, None] * Xy[None, None, :])                 # (c, nell, n)
        for name, pf in (("y_b12", params_b12), ("y_lee22p", params_lee22p)):
            A, xc, bb = pf(M[sl], z[sl])
            sig = xc[:, None] * table(Xy[None, :] / xc[:, None], np.repeat(bb[:, None], Xy.size, 1))
            yX = SIGMA_T / MEC2 * (A * P200[sl] * R200[sl])[:, None] * sig      # y(X)
            integ = 2 * np.pi * th[sl, None] ** 2 * Xy[None, :] * wy[None, :] * yX
            out[name][sl] = np.einsum("cln,cn->cl", J, integ)
        for Xmax, suf in ((3.0, ""), (1.0, "_1r")):
            Xk = Xmax * S_NODES ** 2; wk = 2 * Xmax * S_NODES * S_W
            _tau_pair(out, sl, q, Xk, wk, M, z, rc, R200, DA, suf)
    return out


def _tau_pair(out, sl, q, Xk, wk, M, z, rc, R200, DA, suf):
    if True:
        jk = np.sinc(q[:, :, None] * Xk[None, None, :] / np.pi)
        n0, xc, bp = params_lee22n(M[sl], z[sl])
        ne0 = n0 * 200 * rc[sl] * FB / (XH * MP)
        g = (Xk[None, :] / xc[:, None]) ** -0.3 * (1 + Xk[None, :] / xc[:, None]) ** -bp[:, None]
        integ = 4 * np.pi * Xk[None, :] ** 2 * wk[None, :] * g * (SIGMA_T * R200[sl] ** 3 / DA[sl] ** 2 * ne0)[:, None]
        out["tau_lit" + suf][sl] = np.einsum("cln,cn->cl", jk, integ)
        m = M[sl] / 1e14; z1 = 1 + z[sl]
        P0 = 4000 * m ** 0.29 * z1 ** -0.66; al = 0.88 * m ** -0.03 * z1 ** 0.19; be = 3.83 * m ** 0.04 * z1 ** -0.025
        u = Xk[None, :] / 0.5
        g = u ** -0.2 * (1 + u ** al[:, None]) ** (-(be[:, None] - 0.2) / al[:, None])
        ne0 = 0.9 * FB * rc[sl] * P0 / M_PER_E
        integ = 4 * np.pi * Xk[None, :] ** 2 * wk[None, :] * g * (SIGMA_T * R200[sl] ** 3 / DA[sl] ** 2 * ne0)[:, None]
        out["tau_b16" + suf][sl] = np.einsum("cln,cn->cl", jk, integ)


def main():
    t0 = time.time()
    table = ProjTable()
    # spot validation of the projection table against scipy quad
    from scipy.integrate import quad
    val = []
    for u, bb in ((1e-4, 4.35), (0.3, 4.35), (2.0, 4.35), (1e-3, 9.4), (0.5, 20.0), (5.0, 12.0)):
        ref = 2 * quad(lambda s: (u * u + s * s) ** -0.15 * (1 + np.sqrt(u * u + s * s)) ** -bb, 0, np.inf, limit=400)[0]
        val.append(dict(u=u, b=bb, table=float(table(np.array([u]), np.array([bb]))[0]), quad=ref))
    print("projection table checks:", [round(v["table"] / v["quad"] - 1, 7) for v in val], flush=True)
    M, z, vlos = read_catalogue()
    b = bin_catalogue(M, z, vlos)
    del M, z, vlos
    print("occupied bins", b["N"].size, "halos", b["N"].sum(), round(time.time() - t0, 1), flush=True)
    tr = transforms(b, table)
    print("transforms done", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(OUT / "indep_halo_sums_tables.npz", ells=ELLS, **b, **tr)
    print("saved", round(time.time() - t0, 1), flush=True)
    (OUT / "indep_projection_table_checks.json").write_text(json.dumps(val, indent=1))


if __name__ == "__main__":
    main()
