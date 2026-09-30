#!/usr/bin/env python3
"""Analytic halo-model DM PDFs (Torkamani et al. 2026, eqs. 29-31 and 46) for the corrected Lee22 densities, at source
redshifts 0.1, 0.5 and 1 (2026-09-30; runs on idark with Python 3.8). Reuses the engine and slice construction of
../../analytic_pdf_torkamani2026/ unchanged; the only addition is the source-redshift cut (slices with z < z_s).
  validation   HalfDome cosmology, catalogue abundance (lightcone counts / 4 pi, M >= 7.327e12 Msun): must reproduce
               the HalfDome ray histograms
  tng          Planck15 (TNG) cosmology, Tinker08 M200c abundance down to 1e9 Msun: the analytic prediction for TNG
Models: b16, lee22_noconc_xgpnative (production), lee22_noconc_tngcal (x C(z)), lee22_noconc_a3fn (x (1+z)^3/F_n).
Usage: python run_corrected_analytic_pdfs.py --table T.h5 --catalog lightcone_100.hdf5 --out DIR [--workers 8]"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[2] / "analytic_pdf_torkamani2026"
sys.path.insert(0, str(ENGINE))
import halo_model_dm_pdf as hm  # noqa: E402
import run_analytic_halo_pdfs as base  # noqa: E402

MODELS = ("b16", "lee22_noconc_xgpnative", "lee22_noconc_tngcal", "lee22_noconc_a3fn", "lee22_noconc_phys")
SOURCES = (0.1, 0.5, 1.0)
G = {}


def configurations(models=MODELS, sources=SOURCES, limit=None):
    runs = []
    for zs in sources:
        for model in models:
            for window in ("m1e13_to_1e14", "m1e12_to_1e14"):
                runs.append(dict(name="validation_{}_catalog_{}_zs{}".format(model, window, zs), group="validation", profile=model,
                                 abundance="catalog", window=window, cosmology="halfdome", mass_floor=base.HD_FLOOR_MSUN, z_source=zs))
            for window in ("m1e13_to_1e14", "m1e12_to_1e14", "all"):
                runs.append(dict(name="tng_{}_{}_zs{}".format(model, window, zs), group="tng", profile=model, abundance="tinker08",
                                 window=window, cosmology="planck15", mass_floor=None, z_source=zs))
    return runs[:limit] if limit else runs


def init(table_path, counts, lattice_h, lattice_log2n, out, models):
    G["tables"] = {m: hm.HalfDomeSphereTable(table_path, m) for m in models}
    G["counts"], G["out"] = counts, Path(out)
    G["lattice"] = hm.Lattice(h=lattice_h, n=2 ** lattice_log2n)
    G["x_edges"] = hm.impact_samples()
    G["cosmos"] = {"halfdome": hm.HALFDOME, "planck15": hm.PLANCK15_TNG}


def one(run):
    t0 = time.time()
    lattice = G["lattice"]; lattice.dropped = 0.0
    cosmo = G["cosmos"][run["cosmology"]]; table = G["tables"][run["profile"]]
    slices = (s for s in base.slices_for(run["profile"], run["abundance"], base.WINDOWS[run["window"]], cosmo, G["x_edges"], table,
                                         counts=G["counts"], mass_floor=run["mass_floor"]) if s["z"] < run["z_source"])
    res = hm.compound_poisson_pdf(lattice, slices, cosmo=cosmo)
    pdf_cl, prob_cl = res.binned(base.PDF_EDGES, clustered=True)
    pdf_u, prob_u = res.binned(base.PDF_EDGES, clustered=False)
    st_cl, st_u = res.stats(True), res.stats(False)
    np.savez(G["out"] / (run["name"] + ".npz"), edges=base.PDF_EDGES, pdf_clustered=pdf_cl, pdf_unclustered=pdf_u,
             prob_clustered=prob_cl, prob_unclustered=prob_u, run=json.dumps(run), stats_clustered=json.dumps(st_cl),
             stats_unclustered=json.dumps(st_u), nbar=res.nbar, zero_clustered=res.zero_clustered, zero_unclustered=res.zero_unclustered,
             mean_dm_all_rays=res.mean_dm_all_rays, lattice=json.dumps(dict(h=lattice.h, n=lattice.n)), info=json.dumps(res.info))
    out = dict(run=run, nbar=res.nbar, dropped_crossings_per_ray=lattice.dropped, clustered=st_cl, unclustered=st_u,
               mean_dm_all_rays_direct=res.mean_dm_all_rays, seconds=time.time() - t0)
    print("{:58s} hit {:.4f}  mean/hit {:7.2f}  median {:6.2f}  {:.0f}s".format(run["name"], st_cl["hit_fraction"], st_cl["mean_dm_hit"],
                                                                            st_cl["median_dm_hit"], out["seconds"]), flush=True)
    return run["name"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lattice-h", type=float, default=0.002)
    ap.add_argument("--lattice-log2n", type=int, default=24)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--sources", nargs="*", type=float, default=list(SOURCES))
    ap.add_argument("--limit", type=int, default=None, help="first N runs only (testing)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base.CATALOG = Path(a.catalog)
    base.CATALOG_COUNTS = out / "halfdome_catalog_counts_m200c_z.npz"
    counts = base.catalog_counts()[0]
    runs = configurations(a.models, a.sources, a.limit)
    with Pool(a.workers, initializer=init, initargs=(a.table, counts, a.lattice_h, a.lattice_log2n, str(out), a.models)) as pool:
        results = dict(pool.map(one, runs, chunksize=1))
    (out / "analytic_summary.json").write_text(json.dumps(results, indent=1))
    print("wrote", out / "analytic_summary.json")


if __name__ == "__main__":
    main()
