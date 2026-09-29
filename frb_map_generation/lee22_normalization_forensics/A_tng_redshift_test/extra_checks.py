#!/usr/bin/env python3
"""Extra checks for test A, reusing the model grids saved by redshift_test.py (snapshot redshift assignment):

 1. Mass units of the TNG windows: slice crossing rates (hit fractions) for the window read as physical Msun
    (default) or as Msun/h (edges divided by h = 0.6774).
 2. A second, lower mass bin: the per-ray difference mass12to14 - mass13to14 (halos of 1e12-1e13 Msun that are not
    inside a 1e13-1e14 halo's R200 sphere), against model halos of 1e12-1e13 Msun.
 3. Abundance-corrected C(z): C divided by the slice crossing-rate ratio TNG/model (removes Tinker08-vs-TNG abundance
    x area differences to first order).
 4. Redshift-scaling readings of Lee22, as C(z) of literal x f(z):
      f = (1+z)^3              n_e measured with comoving volumes, normalized by the physical n200(z)
      f = (1+z)^3 / E^2(z)     repo option :comoving_hypothesis (comoving n_e AND n200 at z = 0)
 5. The z -> 0 anchor: C of the lowest slice (z_DM-weighted ~ 0.05) extrapolated to z = 0 with k = 3 and k = fit.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import redshift_test as RT     # noqa: E402
import models as M             # noqa: E402

OUT = RT.OUT
rng = np.random.default_rng(7)
g = np.load(OUT / "model_grids.npz")
zf, lm_edges = g["zf"], g["lm_edges"]
MODE = "snapshot"
zeff = g[MODE + "__zeff"]
zs, arr = RT.load_tng()
arr = {w: arr[w] for w in ("mass13to14", "mass12to14")}
arr["pseudo12to13"] = arr["mass12to14"] - arr["mass13to14"]
print("pseudo12to13: min per-ray value", arr["pseudo12to13"].min(), " monotonic:",
      bool((np.diff(arr["pseudo12to13"], axis=0) >= -1e-9).all()))
tb = RT.tng_bootstrap(arr, rng)
slice_lo = np.concatenate([[0.0], zs[:-1]]); slice_hi = zs
nsl = len(zs)
member = [(zf >= slice_lo[j] - 1e-9) & (zf < slice_hi[j] - 1e-9) for j in range(nsl)]
h = M.PLANCK15_TNG.h
E2 = M.PLANCK15_TNG.E(zeff) ** 2
FACTORS = {"literal": np.ones_like(zf), "literal_x_(1+z)^3": (1 + zeff) ** 3,
           "literal_x_(1+z)^3/E^2": (1 + zeff) ** 3 / E2}


def slice_sum(grid, frac):
    return np.array([(grid[member[j]] * frac).sum() for j in range(nsl)])


out_rows, fit_rows = [], []
fitsel = np.where((slice_lo >= 0.1 - 1e-9) & (slice_hi <= 2.0 + 1e-9))[0]
lines = []
# ---------------- 1. mass units
for w, (lo, hi) in (("mass13to14", (1e13, 1e14)), ("mass12to14", (1e12, 1e14))):
    for units, div in (("Msun", 1.0), ("Msun/h", h)):
        frac = RT.window_fraction(lm_edges, lo / div, hi / div)
        nb = slice_sum(g[MODE + "__nbar"], frac)
        rat = -np.log1p(-tb[w]["hit_mean"]) / nb
        cum_model_hit = -np.expm1(-np.cumsum(nb))
        j1 = int(np.argmin(np.abs(zs - 1.0)))
        lines.append("{:11s} window read as {:7s}: slice crossing rate TNG/model median {:.3f} (range {:.3f}-{:.3f}) over "
                     "z<2; cumulative hit z_s=1: TNG {:.4f} model {:.4f}".format(
                         w, units, np.median(rat[fitsel]), rat[fitsel].min(), rat[fitsel].max(), tb[w]["cum_hit"][j1],
                         cum_model_hit[j1]))
        for j in range(nsl):
            out_rows.append(dict(check="mass_units", window=w, variant=units, z_lo=slice_lo[j], z_hi=slice_hi[j],
                                 value=rat[j], note="slice crossing-rate ratio TNG/model"))

# ---------------- 2-4. C(z) fits for variants
def do_fit(tag, w, frac, model_grid_key, factor, abundance_correct=False, abscissa="path"):
    dm = slice_sum(g[MODE + "__" + model_grid_key] * factor[:, None], frac)
    nb = slice_sum(g[MODE + "__nbar"], frac)
    zabs = zf if abscissa == "path" else zeff        # path redshift, or the redshift of the snapshot the halos come from
    zw = np.array([np.average(zabs[member[j]], weights=(g[MODE + "__" + model_grid_key][member[j]] * frac).sum(1) + 1e-300)
                   for j in range(nsl)])
    C = tb[w]["inc_mean"] / dm
    Cb = tb[w]["inc_boot"] / dm[None]
    if abundance_correct:
        rat = -np.log1p(-tb[w]["hit_mean"]) / nb
        ratb = -np.log1p(-np.clip(tb[w]["hit_boot"], 0, 0.999999)) / nb[None]
        C = C / rat
        Cb = Cb / ratb
    s = fitsel
    covl = np.cov(np.log(Cb[:, s]).T)
    beta, cov, chi2 = RT.fit_powerlaw(zw[s], C[s], covl)
    a3, a3e, chi3 = RT.fit_fixed_k(zw[s], C[s], covl, 3.0)
    a0, a0e, chi0 = RT.fit_fixed_k(zw[s], C[s], covl, 0.0)
    row = dict(tag=tag, window=w, model=model_grid_key, C0=np.exp(beta[0]), C0_err=np.exp(beta[0]) * np.sqrt(cov[0, 0]),
               k=beta[1], k_err=np.sqrt(cov[1, 1]), chi2=chi2, dof=len(s) - 2, C0_fixed_k3=np.exp(a3), chi2_fixed_k3=chi3,
               C0_fixed_k0=np.exp(a0), chi2_fixed_k0=chi0, C_lowest_slice=C[0], C_lowest_slice_err=Cb[:, 0].std(ddof=1),
               z_lowest_slice=zw[0])
    fit_rows.append(row)
    for j in range(nsl):
        out_rows.append(dict(check=tag, window=w, variant=model_grid_key, z_lo=slice_lo[j], z_hi=slice_hi[j], value=C[j],
                             note="C per slice; err {:.4g}; zw {:.4f}".format(Cb[:, j].std(ddof=1), zw[j])))
    return row


for w, (lo, hi) in (("mass13to14", (1e13, 1e14)), ("mass12to14", (1e12, 1e14)), ("pseudo12to13", (1e12, 1e13))):
    frac = RT.window_fraction(lm_edges, lo, hi)
    for fname, fac in FACTORS.items():
        do_fit("lee22_" + fname, w, frac, "lee22_literal", fac)
    for m in ("b16", "fb_full"):
        do_fit(m, w, frac, m, FACTORS["literal"])
    for m in ("lee22_literal", "b16", "fb_full"):
        do_fit(m + "_abundance_corrected", w, frac, m, FACTORS["literal"], abundance_correct=True)
    for m in ("lee22_literal", "b16", "fb_full"):
        do_fit(m + "_vs_snapshot_z", w, frac, m, FACTORS["literal"], abscissa="snapshot")
    do_fit("lee22_literal_x_(1+z)^3_vs_snapshot_z", w, frac, "lee22_literal", FACTORS["literal_x_(1+z)^3"], abscissa="snapshot")

# windows read as Msun/h (edges / h): does the redshift slope depend on the mass-unit reading?
for w, (lo, hi) in (("mass13to14", (1e13, 1e14)), ("mass12to14", (1e12, 1e14))):
    frac = RT.window_fraction(lm_edges, lo / h, hi / h)
    do_fit("MsunH_lee22_literal", w, frac, "lee22_literal", FACTORS["literal"])
    do_fit("MsunH_lee22_literal_x_(1+z)^3", w, frac, "lee22_literal", FACTORS["literal_x_(1+z)^3"])
    do_fit("MsunH_b16", w, frac, "b16", FACTORS["literal"])
    do_fit("MsunH_fb_full", w, frac, "fb_full", FACTORS["literal"])

for r in fit_rows:
    lines.append("{tag:40s} {window:12s} C0={C0:.3f}+-{C0_err:.3f} k={k:+.3f}+-{k_err:.3f} chi2/dof={chi2:.0f}/{dof} "
                 "C0(k=3)={C0_fixed_k3:.3f} C0(k=0)={C0_fixed_k0:.3f}  C(z={z_lowest_slice:.3f})={C_lowest_slice:.3f}"
                 "+-{C_lowest_slice_err:.3f}".format(**r))
# z -> 0 anchor for the literal model
for r in fit_rows:
    if r["tag"] in ("lee22_literal", "lee22_literal_abundance_corrected"):
        z0 = r["z_lowest_slice"]
        lines.append("z->0 anchor {:34s} {:12s}: C(z={:.3f}) = {:.3f}; extrapolated to z=0 with k=3: {:.3f}, with k=fit "
                     "({:.2f}): {:.3f}".format(r["tag"], r["window"], z0, r["C_lowest_slice"], r["C_lowest_slice"] / (1 + z0) ** 3,
                                               r["k"], r["C_lowest_slice"] / (1 + z0) ** r["k"]))
txt = "\n".join(lines)
print(txt)
(OUT / "extra_checks.txt").write_text(txt + "\n")
for name, rows in (("extra_checks_slices.csv", out_rows), ("extra_checks_fits.csv", fit_rows)):
    with open(OUT / name, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()})
