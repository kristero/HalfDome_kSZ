#!/usr/bin/env python3
"""Step 6: halo-DM PDF and hit statistics at z_s = 1 for Lee22 with the per-redshift correction C(z) measured
against TNG (redshift_test.py), compared with TNG and with the constant 0.602 (XGPaint-native) variant.

Engine: the validated Torkamani+26 compound-Poisson lattice (halo_model_dm_pdf.py) with Tinker08 in Planck15/TNG
cosmology and the Julia chord tables of the HalfDome sphere products (halfdome_sphere_profiles.h5; HalfDome-cosmology
R200c/theta200c, < 1 % from Planck15, exactly as in the 2026-09-25 analysis). Every Lee22 variant is the tabulated
XGPaint-native g(x) times a factor depending on z only:
    xgpnative     1
    literal       1 / 0.6016316602
    corr_free     C0 (1+z)^k / 0.6016...     (free power-law fit of C_lee22_literal, mass13to14, continuous, 0.1-2)
    corr_k3       C0' (1+z)^3 / 0.6016...    (same, k fixed at 3)
The shape of each halo's profile is untouched, so any PDF difference between variants is the amplitude alone.
"""
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "frb_map_generation" / "analytic_pdf_torkamani2026"))
import halo_model_dm_pdf as hm                 # noqa: E402
import run_analytic_halo_pdfs as runner        # noqa: E402

OUT = ROOT / "frb_map_generation" / "outputs" / "lee22_normalization_forensics_20260928" / "A_tng_redshift_test"
TABLE = runner.TABLE
TNG_DIR = ROOT / "frb_catalog_comparison_outputs" / "haloDM-20260819T104507Z-1-001" / "haloDM"
XGP = 0.6016316602183194
EDGES = runner.PDF_EDGES
WINDOWS = {"mass13to14": (1e13, 1e14), "mass12to14": (1e12, 1e14)}


def read_fit(window, mode="continuous", model="lee22_literal", zr="0.1-2.0"):
    with open(OUT / "powerlaw_fits.csv") as fh:
        for r in csv.DictReader(fh):
            if r["window"] == window and r["mode"] == mode and r["model"] == model and r["z_range"] == zr:
                return {k: float(v) for k, v in r.items() if k not in ("window", "mode", "model", "z_range")}
    raise KeyError(window)


def scaled_table(base, factor_of_z):
    t = types.SimpleNamespace(log10m=base.log10m, z=base.z, x=base.x, theta200=base.theta200)
    t.g = base.g * factor_of_z(base.z)[None, :, None]
    return t


def tng_stats(window):
    a = np.load(TNG_DIR / "DMhalo_r200_{}.npy".format(window))
    zs = np.round(np.load(TNG_DIR / "redshifts_DMhalo.npy"), 6)
    v = a[int(np.argmin(np.abs(zs - 1.0)))]
    h = v > 0
    inr = (v >= EDGES[0]) & (v < EDGES[-1])
    counts, _ = np.histogram(v[inr], EDGES)
    pdf = counts / (inr.sum() * np.diff(EDGES))
    st = dict(hit_fraction=float(h.mean()), mean_dm_all_rays=float(v.mean()), mean_dm_hit=float(v[h].mean()),
              median_dm_hit=float(np.median(v[h])))
    for t in (1, 3, 10, 30):
        st["frac_hit_below_{}".format(t)] = float((v[h] < t).mean())
    return st, pdf, counts


def pct_diff_summary(pdf_model, pdf_tng, counts):
    ctr = np.sqrt(EDGES[1:] * EDGES[:-1])
    ok = counts >= 10
    out = {}
    for lo, hi in ((1, 10), (10, 100), (100, 1000)):
        s = ok & (ctr >= lo) & (ctr < hi)
        d = 100 * (pdf_model[s] - pdf_tng[s]) / pdf_tng[s]
        out["pct_{}_{}".format(lo, hi)] = "{:+.1f} ({:.1f})".format(np.median(d), np.sqrt(np.mean(d ** 2))) if s.any() else "n/a"
    return out


def main():
    lattice = hm.Lattice(h=0.002, n=2 ** 24)
    x_edges = hm.impact_samples()
    lee = hm.HalfDomeSphereTable(TABLE, "lee22_noconc_xgpnative")
    b16 = hm.HalfDomeSphereTable(TABLE, "b16")
    cosmo = hm.PLANCK15_TNG
    rows = []
    for window, wr in WINDOWS.items():
        fit = read_fit(window)
        fit13 = read_fit("mass13to14")
        variants = {
            "b16": (b16, lambda z: np.ones_like(z)),
            "lee22_xgpnative": (lee, lambda z: np.ones_like(z)),
            "lee22_literal": (lee, lambda z: np.ones_like(z) / XGP),
            "lee22_corr_free_mass13to14fit": (lee, lambda z, f=fit13: f["C0"] * (1 + z) ** f["k"] / XGP),
            "lee22_corr_k3_mass13to14fit": (lee, lambda z, f=fit13: f["C0_fixed_k3"] * (1 + z) ** 3 / XGP),
        }
        if window != "mass13to14":
            variants["lee22_corr_free_ownfit"] = (lee, lambda z, f=fit: f["C0"] * (1 + z) ** f["k"] / XGP)
        st_t, pdf_t, cnt_t = tng_stats(window)
        rows.append(dict(window=window, variant="TNG (Konietzka within R200)", **st_t))
        for name, (tab, fac) in variants.items():
            lattice.dropped = 0.0
            t = scaled_table(tab, fac)
            res = hm.compound_poisson_pdf(lattice, runner.slices_for("table", "tinker08", wr, cosmo, x_edges, t), cosmo=cosmo)
            pdf_cl, _ = res.binned(EDGES, clustered=True)
            st = res.stats(True)
            st_u = res.stats(False)
            row = dict(window=window, variant=name, hit_fraction=st["hit_fraction"], mean_dm_all_rays=res.mean_dm_all_rays,
                       mean_dm_hit=st["mean_dm_hit"], median_dm_hit=st["median_dm_hit"],
                       hit_fraction_unclustered=st_u["hit_fraction"], mean_dm_hit_unclustered=st_u["mean_dm_hit"],
                       **{k: st[k] for k in st if k.startswith("frac_hit_below")})
            row.update(pct_diff_summary(pdf_cl, pdf_t, cnt_t))
            rows.append(row)
            np.savez(OUT / "pdf_zs1_{}_{}.npz".format(window, name), edges=EDGES, pdf_clustered=pdf_cl, pdf_tng=pdf_t,
                     counts_tng=cnt_t, stats=json.dumps(row))
            print("{:11s} {:32s} hit={:.4f} mean_all={:7.2f} mean_hit={:7.2f} median={:6.2f} {}".format(
                window, name, row["hit_fraction"], row["mean_dm_all_rays"], row["mean_dm_hit"], row["median_dm_hit"],
                {k: row[k] for k in row if k.startswith("pct_")}), flush=True)
        print("{:11s} {:32s} hit={:.4f} mean_all={:7.2f} mean_hit={:7.2f} median={:6.2f}".format(
            window, "TNG", st_t["hit_fraction"], st_t["mean_dm_all_rays"], st_t["mean_dm_hit"], st_t["median_dm_hit"]))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / "corrected_lee22_pdf_stats_zs1.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
