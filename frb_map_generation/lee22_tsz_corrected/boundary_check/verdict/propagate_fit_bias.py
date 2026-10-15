"""Verdict re-derivation (2026-10-08): independent of the three investigation scripts.

1. Monte Carlo of the dropped-pair fraction for uniformly placed halos in a periodic box (drop rule of the
   c3fe9b8 hint path: a pair is lost iff its minimum image lies across x=0, y=0 or z=0), against 3r/(4L).
2. Expected stacked-profile bias in TNG300 for the bins Lee22 actually populated (upper mass edges quoted in the
   report: 10^14.0 at z<=0.2, ~10^13.6 at z=1, 10^13.2 at z=2; this matches ">29 halos per concentration
   quartile" with Tinker08 counts), at x = 0.04, 1.0, 1.34.
3. Propagation to the painted quantities: refit Lee22's no-c GNFW (gamma=-0.3, alpha=1) to model x (1 - loss) on
   27 log-spaced points in 0.04-1.34 R200c, regress the per-bin parameter shifts on ln(M/Mcut) and ln(1+z)
   (as a stand-in for Lee22's global power-law fit), extrapolate with the production shape clip (10^14.8 h^-1),
   and evaluate the ratio biased/unbiased of
     pressure: Y_sph(<1), Y_sph(<4), Y_cyl(<4) (= tSZ 4 R200c projected aperture, LOS to infinity)
     density : N_e,sph(<1) (= mean DM per hit, FRB rays inside the R200c sphere), N_e,sph(<4) (= kSZ 4 R200c sphere)
   The sign convention: ratio < 1 means the bug made the printed Lee22 quantity LOW; removing the bug would raise
   the HalfDome quantity by 1/ratio.
Usage: python propagate_fit_bias.py   (writes ../../../outputs/lee22_boundary_check_20261007/verdict/*)
"""
import json
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares

OUT = "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_boundary_check_20261007/verdict"
os.makedirs(OUT, exist_ok=True)
L = 205000.0                    # TNG300 BoxSize, ckpc/h
OM, OL = 0.3089, 0.6911
RHOC0 = 277.5                   # (Msun/h) / (kpc/h)^3  (2.775e11 h^2 Msun Mpc^-3)
XFIT = np.logspace(np.log10(0.04), np.log10(1.34), 27)
CLIP = 14.8                     # production shape clip, log10 h^-1 Msun
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def r200c_com(logm, z):
    e2 = OM * (1 + z) ** 3 + OL
    return (3 * 10 ** logm / (4 * np.pi * 200 * RHOC0 * e2)) ** (1 / 3) * (1 + z)


def loss(x, r200):
    return 0.75 * x * r200 / L


# ----------------------------------------------------------------------------------- 1. Monte Carlo of the law
rng = np.random.default_rng(20261008)
say("1. Monte Carlo: fraction of (halo, direction) pairs dropped by the c3fe9b8 hint path, uniform halo centres")
mc = []
for r in [300.0, 1000.0, 3000.0]:
    n = 4_000_000
    c = rng.uniform(0, L, size=(n, 3))
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    p = c + r * v
    lost = np.any(p < 0, axis=1)                  # across a low face -> hint +1 instead of -1 -> dropped
    f = lost.mean()
    err = np.sqrt(f * (1 - f) / n)
    mc.append(dict(r=r, mc=f, err=err, law=0.75 * r / L))
    say(f"   r = {r:6.0f} ckpc/h: dropped {f:.6f} +- {err:.6f}   3r/(4L) = {0.75 * r / L:.6f}")

# ----------------------------------------------------------------------------------- 2. TNG300 bins Lee22 fitted
ZSNAP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]          # TNG full snapshots with z <= 2


def upper_edge(z):
    return 14.0 if z <= 0.2 else 13.8 if z <= 0.5 else 13.6 if z <= 1.0 else 13.4 if z <= 1.5 else 13.2


bins = [(z, lo + 0.1) for z in ZSNAP for lo in np.arange(13.0, upper_edge(z) - 0.05, 0.2)]
say()
say("2. Expected stacked bias -(3/4) x R200c/L in the populated Lee22 bins (bin centres; comoving R200c)")
b004 = [-loss(0.04, r200c_com(m, z)) for z, m in bins]
b1 = [-loss(1.0, r200c_com(m, z)) for z, m in bins]
b134 = [-loss(1.34, r200c_com(m, z)) for z, m in bins]
ftouch = [3 * 1.34 * r200c_com(m, z) / L for z, m in bins]
say(f"   {len(bins)} bins; x=0.04: {100*max(b004):.4f}% .. {100*min(b004):.4f}%;  x=1: {100*max(b1):.3f}% .. "
    f"{100*min(b1):.3f}%;  x=1.34: {100*max(b134):.3f}% .. {100*min(b134):.3f}%")
say(f"   fraction of halos with a low face within 1.34 R200c: {100*min(ftouch):.2f}% .. {100*max(ftouch):.2f}%")


# ----------------------------------------------------------------------------------- 3. refit and propagate
def p_params(lm, z, shape_lm=None):
    shape_lm = lm if shape_lm is None else shape_lm
    rt = 10 ** (lm - 13.60)
    rs = 10 ** (shape_lm - 13.60)
    return (2.8 * (rt ** 1.08 if rt < 1 else rt ** 0.80) * (1 + z) ** -1.89,
            2.10 * rs ** -0.35 * (1 + z) ** 0.53, 9.4 * rs ** -0.06 * (1 + z) ** 0.60)


def n_params(lm, z, shape_lm=None):
    shape_lm = lm if shape_lm is None else shape_lm
    rt = 10 ** (lm - 13.61)
    rs = 10 ** (shape_lm - 13.61)
    return (6.8 * rt ** 0.68 * (1 + z) ** -2.11,
            7.9 * (1 + z) ** -0.67 * (rs ** 0.47 if rs < 1 else rs ** -0.45),
            19.5 * (1 + z) ** -0.31 * (rs ** 0.70 if rs < 1 else rs ** -0.18))


def lngnfw(x, lna, lnxc, beta):
    u = x / np.exp(lnxc)
    return lna - 0.3 * np.log(u) - beta * np.log1p(u)


def refit(par, r200):
    a, xc, be = par
    target = lngnfw(XFIT, np.log(a), np.log(xc), be) + np.log1p(-loss(XFIT, r200))
    sol = least_squares(lambda q: lngnfw(XFIT, *q) - target, x0=[np.log(a), np.log(xc), be],
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    resid = np.max(np.abs(sol.fun))
    return sol.x - np.array([np.log(a), np.log(xc), be]), resid


res = {}
for name, fpar, mcut in [("pressure", p_params, 13.60), ("density", n_params, 13.61)]:
    rows = []
    for z, m in bins:
        par = fpar(m, z)
        d, rr = refit(par, r200c_com(m, z))
        rows.append([z, m, d[0], d[1], d[2] / par[2], rr])
    rows = np.array(rows)
    # regression of the shifts on [1, ln(M/Mcut), ln(1+z)]: stand-in for the global power-law fit
    X = np.column_stack([np.ones(len(rows)), np.log(10 ** (rows[:, 1] - mcut)), np.log1p(rows[:, 0])])
    coef = {k: np.linalg.lstsq(X, rows[:, j], rcond=None)[0] for k, j in [("dlnA", 2), ("dlnxc", 3), ("dlnbeta", 4)]}
    rms = {k: float(np.sqrt(np.mean((X @ coef[k] - rows[:, j]) ** 2))) for k, j in [("dlnA", 2), ("dlnxc", 3), ("dlnbeta", 4)]}
    say()
    say(f"3. {name}: per-bin refit shifts over {len(rows)} bins: dlnA {rows[:,2].min():+.5f}..{rows[:,2].max():+.5f}, "
        f"dln xc {rows[:,3].min():+.5f}..{rows[:,3].max():+.5f}, dbeta/beta {rows[:,4].min():+.5f}..{rows[:,4].max():+.5f}, "
        f"max |ln residual| {rows[:,5].max():.1e}")
    for k in coef:
        say(f"   regression {k}: const {coef[k][0]:+.5f}, d/dln(M/Mcut) {coef[k][1]:+.6f} (= mass-slope shift), "
            f"d/dln(1+z) {coef[k][2]:+.6f} (= (1+z)-exponent shift); rms about regression {rms[k]:.1e}")

    def shifted(lm, z):
        sl = min(lm, CLIP)
        a, xc, be = fpar(lm, z, sl)
        va = np.array([1.0, np.log(10 ** (lm - mcut)), np.log1p(z)])
        vs = np.array([1.0, np.log(10 ** (sl - mcut)), np.log1p(z)])
        return ((np.log(a), np.log(xc), be),
                (np.log(a) + va @ coef["dlnA"], np.log(xc) + vs @ coef["dlnxc"], be * np.exp(vs @ coef["dlnbeta"])))

    def f3(x, q):
        return np.exp(lngnfw(x, *q))

    def sph(q, X):
        return quad(lambda x: 4 * np.pi * x * x * f3(x, q), 0, X, limit=400, points=[0.04, 1.0])[0]

    def cyl(q, X):
        inner = sph(q, X)
        outer = quad(lambda x: 4 * np.pi * x * x * f3(x, q) * (1 - np.sqrt(1 - (X / x) ** 2)), X, 1e4, limit=800,
                     points=[2 * X, 10 * X, 100 * X])[0]
        return inner + outer

    tab = []
    for z in [0.0, 0.5, 1.0]:
        for lm in [13.0, 13.5, 14.0, 14.5, 15.0, 15.4]:
            q0, q1 = shifted(lm, z)
            row = dict(z=z, log10M_hinv=lm)
            if name == "pressure":
                row.update(Ysph1=sph(q1, 1) / sph(q0, 1), Ysph4=sph(q1, 4) / sph(q0, 4), Ycyl4=cyl(q1, 4) / cyl(q0, 4),
                           P_x2=f3(2.0, q1) / f3(2.0, q0), P_x4=f3(4.0, q1) / f3(4.0, q0))
            else:
                row.update(Nsph1=sph(q1, 1) / sph(q0, 1), Nsph4=sph(q1, 4) / sph(q0, 4),
                           n_x2=f3(2.0, q1) / f3(2.0, q0), n_x4=f3(4.0, q1) / f3(4.0, q0))
            tab.append(row)
    say(f"   {name}: biased-fit / unbiased-fit for painted quantities (shape clip at 10^{CLIP} h^-1 Msun)")
    keys = [k for k in tab[0] if k not in ("z", "log10M_hinv")]
    say("     z    log10M  " + "  ".join(f"{k:>8s}" for k in keys))
    for row in tab:
        say(f"   {row['z']:4.1f}  {row['log10M_hinv']:6.1f}  " + "  ".join(f"{row[k]:8.5f}" for k in keys))
    res[name] = dict(per_bin=rows.tolist(), regression={k: v.tolist() for k, v in coef.items()},
                     regression_rms=rms, painted=tab)

json.dump(dict(monte_carlo=mc, bins=[list(b) for b in bins], bias_x004=b004, bias_x1=b1, bias_x134=b134,
               frac_touch_134=ftouch, propagation=res), open(os.path.join(OUT, "propagate_fit_bias.json"), "w"),
          indent=1)
open(os.path.join(OUT, "propagate_fit_bias.txt"), "w").write("\n".join(lines) + "\n")
