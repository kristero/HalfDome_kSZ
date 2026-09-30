#!/usr/bin/env python3
"""Analytic halo-model DM PDFs (Torkamani et al. 2026 compound-Poisson engine with clustering, imported unchanged from
../../analytic_pdf_torkamani2026/halo_model_dm_pdf.py) for the reproduction of Konietzka's TNG300 DM catalogue.

Groups (Planck15 = TNG cosmology, Tinker08 M200c HMF x abundance factor, Tinker10 bias, clustered and unclustered):
  halo    gas inside each halo's R200c sphere (the quantity of Konietzka's DMhalo_r200_* arrays), models
          b16, lee22_prev_xgpnative, lee22_corr_fn5p40, lee22_corr_fn5p57, bfc; targets
            all        [1e9, 1e16]   abundance tinker08 and tng300_full
            nominal    [1e13, 1e14]  tinker08 (the label read literally)
            reinterp   [1e13, 9e14]  tng300 (Pillepich+18b counts) and tinker08 (window upper edge x9)
  lss     Torkamani+26's own setup: BFC hot gas (Table 1) to 5 R200c, halos 1e8-1e16 Msun/h, Tinker08; the PDF of the
          halo term, to be shifted to the total mean in analyze_repro.py (their Sec. 2.4)
Source redshifts 0.1, 0.5, 1.0 (halo) and 0.3, 0.5, 0.7, 1.0 (lss).
Usage: python run_repro_pdfs.py [--workers 6] [--only substr ...] [--skip-existing]"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import repro_lib as R
import halo_model_dm_pdf as hm

RUNS_DIR = R.OUT / "analysis" / "runs"
LIN_GRID = np.arange(0.0, 6000.0001, 0.5)
G = {}


def configurations():
    runs = []
    for zs in (0.1, 0.5, 1.0):
        for m in R.MODELS:
            for target, window, abunds in (("all", (1e9, 1e16), ("tinker08", "tng300_full")),
                                           ("nominal", (1e13, 1e14), ("tinker08",)),
                                           ("reinterp", (1e13, R.C_EXT * 1e14), ("tng300", "tinker08"))):
                for ab in abunds:
                    runs.append(dict(name=f"halo_{m}_{target}_{ab}_zs{zs}", group="halo", table=m, target=target,
                                     window=window, abundance=ab, z_source=zs))
    for zs in (0.3, 0.5, 0.7, 1.0):
        for dm_scale in ((25.0, 10.0, 50.0) if zs == 0.7 else (25.0,)):          # taper DM scale 2 pi / lambda_c
            runs.append(dict(name=f"lss_bfc5r200_taper{dm_scale:g}_zs{zs}", group="lss", table="bfc_5r200", target="lss_total",
                             window=(1e8 / R.H_TNG, 1e16 / R.H_TNG), abundance="tinker08", z_source=zs,
                             taper_lambda=2 * np.pi / dm_scale))
    return runs


def table(name):
    if name not in G:
        G[name] = R.load_table(name)
    return G[name]


def guarded_zero(res, clustered):
    """DM = 0 delta weight. For Nbar >> 1 (every ray crosses many halos, e.g. 5 R200c discs down to 1e8 Msun/h) the
    engine's closed forms exp(-Nbar) and exp(-Nbar + <A^2>/2) under/overflow; the true weight is then 0 (the Gaussian
    peak-background average is not valid for the empty-ray probability in that limit)."""
    z = res.zero_clustered if clustered else res.zero_unclustered
    p0 = (res.p_clustered if clustered else res.p_unclustered)[0]
    if res.nbar > 50 or not np.isfinite(z) or z > p0:
        return 0.0
    return float(z)


def stats(res, clustered):
    p = (res.p_clustered if clustered else res.p_unclustered).copy()
    p[0] -= guarded_zero(res, clustered)
    centers = (np.arange(len(p)) + 0.5) * res.lattice.h
    hit = p.sum()
    cdf = np.cumsum(p) / hit
    out = dict(hit_fraction=float(hit), mean_dm_hit=float((p * centers).sum() / hit),
               median_dm_hit=float(np.interp(0.5, cdf, centers)))
    for t in (1, 3, 10, 30):
        out["frac_hit_below_{}".format(t)] = float(np.interp(t, centers, cdf))
    return out


def one(run):
    t0 = time.time()
    lattice = hm.Lattice(h=0.002, n=2 ** 24)
    sl = R.slices(table(run["table"]), run["window"], run["z_source"], run["abundance"])
    if run.get("taper_lambda") is None:
        res = hm.compound_poisson_pdf(lattice, sl, cosmo=R.COSMO)                  # the engine, unchanged
    else:
        res = R.compound_poisson_pdf_tapered(lattice, sl, cosmo=R.COSMO, taper_lambda=run["taper_lambda"])
    res.zero_clustered, res.zero_unclustered = guarded_zero(res, True), guarded_zero(res, False)
    pdf_cl, prob_cl = res.binned(R.PDF_EDGES, clustered=True)
    pdf_u, prob_u = res.binned(R.PDF_EDGES, clustered=False)
    st_cl, st_u = stats(res, True), stats(res, False)
    grid = np.arange(lattice.n + 1) * lattice.h
    extra = {}
    for tag, p, zero in (("clustered", res.p_clustered, res.zero_clustered), ("unclustered", res.p_unclustered, res.zero_unclustered)):
        c = np.concatenate([[0.0], np.cumsum(p)])
        extra["cdf_full_lin_" + tag] = np.interp(LIN_GRID, grid, c)                 # includes the DM = 0 delta
        c[1:] -= zero
        extra["cdf_hit_" + tag] = np.interp(R.CDF_GRID, grid, c)                    # rays with DM > 0 only
        cen = (np.arange(lattice.n) + 0.5) * lattice.h
        pp = p.copy(); pp[0] -= zero
        m1 = float((pp * cen).sum()); m2 = float((pp * cen ** 2).sum())
        extra["moments_all_rays_" + tag] = np.array([float(p.sum()), m1, m2])        # norm, <DM>, <DM^2> (zero rays add 0)
    np.savez(RUNS_DIR / (run["name"] + ".npz"), edges=R.PDF_EDGES, pdf_clustered=pdf_cl, pdf_unclustered=pdf_u,
             prob_clustered=prob_cl, prob_unclustered=prob_u, cdf_grid=R.CDF_GRID, lin_grid=LIN_GRID,
             run=json.dumps(run), stats_clustered=json.dumps(st_cl), stats_unclustered=json.dumps(st_u), nbar=res.nbar,
             zero_clustered=res.zero_clustered, zero_unclustered=res.zero_unclustered, mean_dm_all_rays=res.mean_dm_all_rays,
             dropped=lattice.dropped, min_p_clustered=float(res.p_clustered.min()), info=json.dumps(res.info), **extra)
    out = dict(run=run, nbar=res.nbar, clustered=st_cl, unclustered=st_u, mean_dm_all_rays_direct=res.mean_dm_all_rays,
               lattice_mean_clustered=float(extra["moments_all_rays_clustered"][1]), dropped=lattice.dropped,
               seconds=time.time() - t0)
    print("{:52s} Nbar {:9.3f} hit {:.4f} mean/hit {:8.2f} median {:7.2f} mean(all) {:8.2f} (lattice {:8.2f}) {:4.0f}s".format(
        run["name"], res.nbar, st_cl["hit_fraction"], st_cl["mean_dm_hit"], st_cl["median_dm_hit"], res.mean_dm_all_rays,
        out["lattice_mean_clustered"], out["seconds"]), flush=True)
    return run["name"], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    R.build_bfc_5r200_table()
    R.low_mass_abundance()
    runs = configurations()
    if a.only:
        runs = [r for r in runs if any(s in r["name"] for s in a.only)]
    if a.skip_existing:
        runs = [r for r in runs if not (RUNS_DIR / (r["name"] + ".npz")).exists()]
    # longest first (lss, then 'all')
    runs.sort(key=lambda r: (r["group"] != "lss", r["target"] != "all", -r["z_source"]))
    print(f"{len(runs)} runs on {a.workers} workers", flush=True)
    with Pool(a.workers, maxtasksperchild=6) as pool:
        results = dict(pool.imap_unordered(one, runs, chunksize=1))
    summary = R.OUT / "analysis" / "runs_summary.json"
    old = json.loads(summary.read_text()) if summary.exists() else {}
    old.update(results)
    summary.write_text(json.dumps(old, indent=1))
    print("wrote", summary)


if __name__ == "__main__":
    main()
