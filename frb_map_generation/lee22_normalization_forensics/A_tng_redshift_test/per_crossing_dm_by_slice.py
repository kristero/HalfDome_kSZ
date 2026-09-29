#!/usr/bin/env python3
"""Per-crossing halo-DM distribution in each source-redshift slice: TNG (ray increments > 0) against Monte-Carlo
crossings of the smooth-profile models (same halo population as the analytic model: Tinker08 in Planck15, M200c
window, impact parameter uniform in area inside R200c, DM from the Julia chord tables of the HalfDome sphere products).

Purpose: explain why a model whose per-redshift amplitude is wrong by factors 0.6-2.5 (Lee22 x 0.602) reproduces the
shape of TNG's z_s = 1 per-hit PDF: compare the slice-by-slice median and 16-84 % width of DM per crossing.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import redshift_test as RT                      # noqa: E402
sys.path.insert(0, str(RT.ROOT / "frb_map_generation" / "analytic_pdf_torkamani2026"))
import halo_model_dm_pdf as hm                  # noqa: E402
import run_analytic_halo_pdfs as runner         # noqa: E402

OUT = RT.OUT
XGP = 0.6016316602183194
rng = np.random.default_rng(11)
fit = [r for r in csv.DictReader(open(OUT / "powerlaw_fits.csv")) if r["window"] == "mass13to14" and
       r["mode"] == "continuous" and r["model"] == "lee22_literal" and r["z_range"] == "0.1-2.0"][0]
C0, K = float(fit["C0"]), float(fit["k"])
lee = hm.HalfDomeSphereTable(runner.TABLE, "lee22_noconc_xgpnative")
b16 = hm.HalfDomeSphereTable(runner.TABLE, "b16")
cosmo = hm.PLANCK15_TNG
zs, arr = RT.load_tng()
a = arr["mass13to14"]
inc = np.diff(np.vstack([np.zeros((1, a.shape[1])), a]), axis=0)
lm = lee.log10m
frac = RT.window_fraction(np.round(np.arange(9.0, 16.0001, 0.01), 10), 1e13, 1e14)
sel = np.where(frac > 0)[0]
mass = 10 ** lm[sel]
rows = []
N = 200_000
for j in range(10):
    zlo, zhi = 0.1 * j, 0.1 * (j + 1)
    zi = np.where((lee.z > zlo) & (lee.z < zhi))[0]
    # expected crossings per (z, M) cell ~ chi^2/H dn/dlnM pi theta200^2
    w = np.zeros((len(zi), len(sel)))
    for ii, i in enumerate(zi):
        z = lee.z[i]
        w[ii] = (cosmo.chi(z)[0] ** 2 / cosmo.H_over_c(z) * hm.tinker08_dndlnm(cosmo, mass, z) * frac[sel] *
                 np.pi * lee.theta200[sel, i] ** 2)
    p = (w / w.sum()).ravel()
    pick = rng.choice(p.size, size=N, p=p)
    iz, im = np.unravel_index(pick, w.shape)
    x = np.sqrt(rng.random(N))
    out = dict(z_lo=zlo, z_hi=zhi)
    t = inc[j][inc[j] > 0]
    q = np.percentile(t, [16, 50, 84])
    out.update(tng_n=len(t), tng_p16=q[0], tng_median=q[1], tng_p84=q[2], tng_mean=t.mean())
    for name, tab, fac in (("lee22_xgpnative", lee, lambda z: np.ones_like(z)),
                           ("lee22_corrected", lee, lambda z: C0 * (1 + z) ** K / XGP),
                           ("b16", b16, lambda z: np.ones_like(z))):
        dm = np.empty(N)
        lx = np.log(np.clip(x, tab.x[0], tab.x[-1]))
        order = np.argsort(pick, kind="stable")
        cells, starts, counts = np.unique(pick[order], return_index=True, return_counts=True)
        for c, st, ct in zip(cells, starts, counts):
            s = order[st:st + ct]
            ii, mm = np.unravel_index(c, w.shape)
            g = tab.g[sel[mm], zi[ii], :]
            dm[s] = np.exp(np.interp(lx[s], np.log(tab.x), np.log(g))) * 2 * np.sqrt(1 - x[s] ** 2)
        dm *= fac(tab.z[zi][iz])
        q = np.percentile(dm, [16, 50, 84])
        out.update({name + "_p16": q[0], name + "_median": q[1], name + "_p84": q[2], name + "_mean": dm.mean()})
    rows.append(out)
    print("z {:.1f}-{:.1f}: TNG median {:6.1f} [{:5.1f},{:6.1f}] | xgp {:6.1f} [{:5.1f},{:6.1f}] | corr {:6.1f} [{:5.1f},{:6.1f}]"
          " | b16 {:6.1f} [{:5.1f},{:6.1f}]".format(zlo, zhi, out["tng_median"], out["tng_p16"], out["tng_p84"],
                                                   out["lee22_xgpnative_median"], out["lee22_xgpnative_p16"], out["lee22_xgpnative_p84"],
                                                   out["lee22_corrected_median"], out["lee22_corrected_p16"], out["lee22_corrected_p84"],
                                                   out["b16_median"], out["b16_p16"], out["b16_p84"]), flush=True)
with open(OUT / "per_crossing_dm_by_slice_mass13to14.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    for r in rows:
        wr.writerow(r)
