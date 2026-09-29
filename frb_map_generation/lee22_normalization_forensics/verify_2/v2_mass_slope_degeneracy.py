"""verify_2 / test T3: is the window dependence of the redshift exponent (k ~2.6 for 1e13-1e14 Msun,
~3.15 for 1e12-1e14 Msun) compatible with an EXACT (1+z)^3 plus a mass-dependent error of the Lee22 fit?

Model: TNG_all(W, slice) = A * sum_M (M/10^13.5 Msun)^s * (1+z)^k * literal(W, slice, M)
Grid over s; for each s fit (A,k) per window (all-ray means, bootstrap-free weights = relative 3%).
Then a joint fit of (A, s, k) to both windows, and the same with k fixed to 3.
Caveat: Konietzka windows are not additive (overlaps counted once), the model ignores overlaps.
"""
import json
import numpy as np
from scipy import integrate, optimize
from hmf import MassFunction
import v2_lee22_lib as L
from v2_konietzka_slices import astro, Hz, r200c_phys_mpc, In_grid, KD, OUT

zs = np.load(KD + "redshifts_DMhalo.npy")
nsl = 20
edges = np.concatenate([[0.0], zs[:nsl]])
mf = MassFunction(cosmo_model=astro, sigma_8=0.8159, n=0.9667, hmf_model="Tinker08",
                  mdef_model="SOCritical", mdef_params={"overdensity": 200},
                  Mmin=11.9, Mmax=14.1, dlog10m=0.01, z=0.0, transfer_model="CAMB")
lm_grid = None
per = []  # per slice: (lnm, contribution per lnM integrated over slice), zmid
for i in range(nsl):
    zz = np.linspace(edges[i], edges[i + 1], 9)
    rows = []
    for z in zz:
        mf.update(z=z)
        m = mf.m / L.H
        dn = mf.dndlnm * L.H ** 3
        In = np.array([In_grid(L.TABLE8_N, x * L.H, z) for x in m])
        Ne = m * L.MSUN_G * L.FB * In / (L.XH * L.MP_G)
        rows.append(L.C_KMS / Hz(z) * (1 + z) * dn * Ne / L.MPC_CM ** 2 / L.PC_CM)
    rows = np.array(rows)
    per.append((np.log10(m), integrate.simpson(rows, x=zz, axis=0), 0.5 * (edges[i] + edges[i + 1])))

tng = {}
for w in ("13to14", "12to14"):
    a = np.load(KD + f"DMhalo_r200_mass{w}.npy")[:nsl]
    tng[w] = np.diff(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0).mean(axis=1)
win = {"13to14": (13.0, 14.0), "12to14": (12.0, 14.0)}
zmid = np.array([p[2] for p in per])


def model(w, s):
    lo, hi = win[w]
    out = []
    for lm, c, _ in per:
        sel = (lm >= lo) & (lm <= hi)
        out.append(integrate.simpson(c[sel] * 10 ** (s * (lm[sel] - 13.5)), x=lm[sel] * np.log(10)))
    return np.array(out)


lines = ["# T3: exponent k of C(z)=A(1+z)^k per window, after multiplying literal by (M/10^13.5)^s",
         "# s | k(13to14) | k(12to14) | A(13to14) A(12to14)"]
res = {"scan": []}
for s in np.arange(-0.6, 0.61, 0.1):
    ks = []
    for w in ("13to14", "12to14"):
        C = tng[w] / model(w, s)
        p = np.polyfit(np.log1p(zmid), np.log(C), 1)
        ks.append((p[0], np.exp(p[1])))
    res["scan"].append(dict(s=float(s), k13=ks[0][0], k12=ks[1][0], A13=ks[0][1], A12=ks[1][1]))
    lines.append(f"{s:+.1f} | {ks[0][0]:.3f} | {ks[1][0]:.3f} | {ks[0][1]:.3f} {ks[1][1]:.3f}")


def resid(p, kfix=None):
    lnA, s, k = (p[0], p[1], p[2]) if kfix is None else (p[0], p[1], kfix)
    r = []
    for w in ("13to14", "12to14"):
        pred = np.exp(lnA) * model(w, s) * (1 + zmid) ** k
        r.append(np.log(tng[w] / pred))
    return np.concatenate(r)


fr = optimize.least_squares(resid, [np.log(0.25), 0.0, 3.0])
f3 = optimize.least_squares(lambda p: resid(p, 3.0), [np.log(0.25), 0.0])
f0 = optimize.least_squares(lambda p: resid(np.r_[p, 0.0]), [np.log(0.25), 0.0, 3.0][:2] + [])  if False else None
lines.append(f"# joint free fit: A={np.exp(fr.x[0]):.3f} s={fr.x[1]:+.3f} k={fr.x[2]:.3f}  rms(ln)={np.sqrt(np.mean(fr.fun**2)):.3f}")
lines.append(f"# joint k=3 fit:  A={np.exp(f3.x[0]):.3f} s={f3.x[1]:+.3f}           rms(ln)={np.sqrt(np.mean(f3.fun**2)):.3f}")
res["joint_free"] = dict(A=float(np.exp(fr.x[0])), s=float(fr.x[1]), k=float(fr.x[2]), rms=float(np.sqrt(np.mean(fr.fun ** 2))))
res["joint_k3"] = dict(A=float(np.exp(f3.x[0])), s=float(f3.x[1]), rms=float(np.sqrt(np.mean(f3.fun ** 2))))
# joint with (1+z)^3/E^2 shape and free s
fE = optimize.least_squares(lambda p: resid(np.r_[p[0] , p[1], 0.0]) - np.tile(np.log((1 + zmid) ** 3 / L.E2(zmid)), 2), [np.log(0.5), 0.0])
lines.append(f"# joint (1+z)^3/E^2 fit: A={np.exp(fE.x[0]):.3f} s={fE.x[1]:+.3f}  rms(ln)={np.sqrt(np.mean(fE.fun**2)):.3f}")
res["joint_comovhyp"] = dict(A=float(np.exp(fE.x[0])), s=float(fE.x[1]), rms=float(np.sqrt(np.mean(fE.fun ** 2))))
txt = "\n".join(lines)
print(txt)
open(OUT + "T3_mass_slope_degeneracy.txt", "w").write(txt + "\n")
json.dump(res, open(OUT + "T3_mass_slope_degeneracy.json", "w"), indent=1)
