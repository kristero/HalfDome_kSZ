#!/usr/bin/env python3
"""Analytical halo-model DM PDFs (Torkamani et al. 2026 formalism) for the TNG / HalfDome
halo-only comparison at z_s = 1: gas of every M200c halo counted inside its R200c sphere,
rays to the source plane z_s = 1, M200c windows as in the published comparison.

Configurations
  validation   HalfDome cosmology, halos above the catalogue floor 7.327e12 Msun, Battaglia16 and Lee22
               (the exact HalfDome sphere densities, tabulated by tabulate_halfdome_sphere_profiles.jl),
               abundance either from the HalfDome lightcone itself (counts / 4 pi) or Tinker08 (M200c).
  tng          Planck15 (TNG) cosmology, Tinker08 abundance down to the TNG catalogue's 1e9 Msun,
               Battaglia16 and Lee22 (same tables, extrapolated below their fitted masses) and the BFC
               hot-gas profile with Torkamani et al.'s TNG-calibrated fiducial parameters (Table 1).
Each run gives the unclustered (eq. 31) and clustered (eq. 46) PDF.

Usage: python run_analytic_halo_pdfs.py [--only NAME ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import interpolate

sys.path.insert(0, str(Path(__file__).resolve().parent))
import halo_model_dm_pdf as hm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analytic_halo_pdf_20260925"
TABLE = OUT / "tables" / "halfdome_sphere_profiles.h5"
BFC_TABLE = OUT / "tables" / "bfc_tng_fiducial_sphere_profiles_planck15.npz"
CATALOG = ROOT.parent / "lightcone_100.hdf5"
CATALOG_COUNTS = OUT / "tables" / "halfdome_catalog_counts_m200c_z.npz"
HD_FLOOR_MSUN = 7.327e12
H_HALFDOME = 0.68
Z_SOURCE = 1.0
PDF_EDGES = np.logspace(-1, np.log10(3e4), 300)       # the simulation histograms' edges (299 log bins, 0.1 - 3e4)
CDF_GRID = np.logspace(-3, np.log10(3.3e4), 40001)    # saved CDF of the DM > 0 part, for any later rebinning

WINDOWS = {"all": (1e9, 1e16), "m1e10_to_1e12": (1e10, 1e12), "m1e10_to_1e13": (1e10, 1e13),
           "m1e12_to_1e14": (1e12, 1e14), "m1e13_to_1e14": (1e13, 1e14), "m1e14_to_1e16": (1e14, 1e16)}


def catalog_counts():
    """(z, log10 M200c) counts of the HalfDome lightcone on the analytic quadrature cells."""
    if CATALOG_COUNTS.exists():
        d = np.load(CATALOG_COUNTS)
        return d["counts"], d["z_edges"], d["log10m_edges"]
    import h5py
    z_edges = np.round(np.arange(0, 1.0001, 0.01), 10)
    lm_edges = np.round(np.arange(9.0, 16.0001, 0.01), 10)
    counts = np.zeros((len(z_edges) - 1, len(lm_edges) - 1))
    with h5py.File(CATALOG, "r") as f:
        n = f["redshift"].shape[0]
        step = 10_000_000
        for a in range(0, n, step):
            z = np.asarray(f["redshift"][a:a + step], float)
            m = np.asarray(f["halo_mass_m200c"][a:a + step], float) / H_HALFDOME
            keep = (z < Z_SOURCE) & (m >= HD_FLOOR_MSUN)
            c, _, _ = np.histogram2d(z[keep], np.log10(m[keep]), bins=(z_edges, lm_edges))
            counts += c
    np.savez(CATALOG_COUNTS, counts=counts, z_edges=z_edges, log10m_edges=lm_edges, n_total=n,
             note="lightcone_100.hdf5 halo_mass_m200c / 0.68 (Msun), z < 1, M >= 7.327e12")
    return counts, z_edges, lm_edges


def bfc_table(table_x, log10m, zs, cosmo):
    """BFC g(x; M, z) on the same nodes as the HalfDome tables (cached)."""
    if BFC_TABLE.exists():
        d = np.load(BFC_TABLE)
        if np.array_equal(d["x"], table_x) and np.array_equal(d["log10m"], log10m) and np.array_equal(d["z"], zs):
            return d["g"], d["theta200"]
    g = np.zeros((len(log10m), len(zs), len(table_x)))
    theta = np.zeros((len(log10m), len(zs)))
    for i, z in enumerate(zs):
        g[:, i, :], _ = hm.bfc_chord_mean_dm(cosmo, log10m, z, table_x)
        r200 = hm.r200c_mpc(cosmo, 10 ** log10m, z)
        theta[:, i] = np.arctan(r200 / (cosmo.chi(z)[0] / (1 + z)))
    np.savez(BFC_TABLE, g=g, theta200=theta, x=table_x, log10m=log10m, z=zs,
             params=json.dumps(hm.BFC_TNG_FIDUCIAL), electrons_per_mp=hm.ELECTRONS_PER_MP, cosmology=cosmo.name)
    return g, theta


def slices_for(profile, abundance, window, cosmo, x_edges, table, counts=None, bfc=None, mass_floor=None):
    """Generator of per-z-slice inputs for halo_model_dm_pdf.compound_poisson_pdf."""
    lo, hi = window
    if mass_floor is not None:
        lo = max(lo, mass_floor)
    lm = table.log10m
    dlm = 0.01
    cell_lo, cell_hi = lm - dlm / 2, lm + dlm / 2
    frac = np.clip((np.minimum(cell_hi, np.log10(hi)) - np.maximum(cell_lo, np.log10(lo))) / dlm, 0, 1)
    sel = frac > 0
    logx_nodes = np.log(table.x)
    xe = np.clip(x_edges, table.x[0], table.x[-1])
    chord = 2 * np.sqrt(np.clip(1 - x_edges ** 2, 0, None))
    area_frac = np.diff(x_edges ** 2)                              # annulus area / (pi theta200^2)
    for i, z in enumerate(table.z):
        dz = 0.01
        if profile == "bfc":
            g = bfc[0][sel, i, :]; theta = bfc[1][sel, i]
        else:
            g = table.g[sel, i, :]; theta = table.theta200[sel, i]
        spl = interpolate.CubicSpline(logx_nodes, np.log(g), axis=1)
        dm_edges = np.exp(spl(np.log(xe))) * chord[None, :]
        dm_edges = np.minimum.accumulate(dm_edges, axis=1)          # monotone in x (guards the spline)
        dm_edges[:, -1] = 0.0
        mass = 10 ** lm[sel]
        if abundance == "catalog":
            w = counts[i, sel] / (4 * np.pi)
        else:
            w = (cosmo.chi(z)[0] ** 2 / cosmo.H_over_c(z) * dz *
                 hm.tinker08_dndlnm(cosmo, mass, z) * dlm * np.log(10) * frac[sel])
        disc = np.pi * theta ** 2
        aw = (w * disc)[:, None] * area_frac[None, :]
        bias = hm.tinker10_bias(cosmo, mass, z)
        mid = 0.5 * (dm_edges[:, 1:] + dm_edges[:, :-1])
        yield dict(z=float(z), dz=dz, dm_edges=dm_edges, area_weights=aw, bias=bias,
                   nbar=float((w * disc).sum()), nbar_b=float((w * disc * bias).sum()),
                   mean_dm=float((aw * mid).sum()))


def configurations():
    runs = []
    for model in ("b16", "lee22_noconc_xgpnative"):
        for window in ("all", "m1e13_to_1e14", "m1e12_to_1e14", "m1e14_to_1e16"):
            for abundance in ("catalog", "tinker08"):
                runs.append(dict(name="validation_{}_{}_{}".format(model, abundance, window), group="validation",
                                 profile=model, abundance=abundance, window=window, cosmology="halfdome",
                                 mass_floor=HD_FLOOR_MSUN))
    for model in ("b16", "lee22_noconc_xgpnative", "bfc"):
        for window in ("all", "m1e13_to_1e14", "m1e10_to_1e13", "m1e12_to_1e14", "m1e10_to_1e12"):
            runs.append(dict(name="tng_{}_{}".format(model, window), group="tng", profile=model, abundance="tinker08",
                             window=window, cosmology="planck15", mass_floor=None))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--lattice-h", type=float, default=0.002)
    ap.add_argument("--lattice-log2n", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true", help="skip runs whose .npz already exists")
    a = ap.parse_args()
    lattice = hm.Lattice(h=a.lattice_h, n=2 ** a.lattice_log2n)
    x_edges = hm.impact_samples()
    tables = {m: hm.HalfDomeSphereTable(TABLE, m) for m in ("b16", "lee22_noconc_xgpnative")}
    cosmos = {"halfdome": hm.HALFDOME, "planck15": hm.PLANCK15_TNG}
    counts = catalog_counts()[0]
    bfc = None
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    results = {}
    for run in configurations():
        if a.only and run["name"] not in a.only and run["group"] not in a.only:
            continue
        if a.skip_existing and (OUT / "analysis" / (run["name"] + ".npz")).exists():
            continue
        t0 = time.time()
        lattice.dropped = 0.0
        cosmo = cosmos[run["cosmology"]]
        table = tables["b16" if run["profile"] == "bfc" else run["profile"]]
        if run["profile"] == "bfc" and bfc is None:
            bfc = bfc_table(table.x, table.log10m, table.z, cosmo)
        res = hm.compound_poisson_pdf(lattice, slices_for(run["profile"], run["abundance"], WINDOWS[run["window"]], cosmo,
                                                          x_edges, table, counts=counts, bfc=bfc, mass_floor=run["mass_floor"]),
                                      cosmo=cosmo)
        pdf_cl, prob_cl = res.binned(PDF_EDGES, clustered=True)
        pdf_u, prob_u = res.binned(PDF_EDGES, clustered=False)
        st_cl, st_u = res.stats(True), res.stats(False)
        cdfs = {}
        for tag, p, zero in (("clustered", res.p_clustered, res.zero_clustered), ("unclustered", res.p_unclustered, res.zero_unclustered)):
            c = np.concatenate([[0.0], np.cumsum(p)]); c[1:] -= zero            # remove the DM = 0 delta (cell 0)
            cdfs[tag] = np.interp(CDF_GRID, np.arange(len(c)) * lattice.h, c)
        lattice_mean_u = float((res.p_unclustered * (np.arange(lattice.n) + 0.5) * lattice.h).sum())
        np.savez(OUT / "analysis" / (run["name"] + ".npz"), edges=PDF_EDGES, pdf_clustered=pdf_cl, pdf_unclustered=pdf_u,
                 prob_clustered=prob_cl, prob_unclustered=prob_u, run=json.dumps(run),
                 cdf_grid=CDF_GRID, cdf_hit_clustered=cdfs["clustered"], cdf_hit_unclustered=cdfs["unclustered"],
                 stats_clustered=json.dumps(st_cl), stats_unclustered=json.dumps(st_u), nbar=res.nbar,
                 zero_clustered=res.zero_clustered, zero_unclustered=res.zero_unclustered,
                 mean_dm_all_rays=res.mean_dm_all_rays, lattice_mean_unclustered=lattice_mean_u,
                 lattice=json.dumps(dict(h=lattice.h, n=lattice.n)), info=json.dumps(res.info))
        results[run["name"]] = dict(run=run, nbar=res.nbar, dropped_crossings_per_ray=lattice.dropped, clustered=st_cl, unclustered=st_u,
                                    mean_dm_all_rays_direct=res.mean_dm_all_rays, mean_dm_all_rays_lattice=lattice_mean_u,
                                    min_p_clustered=float(res.p_clustered.min()), seconds=time.time() - t0)
        print("{:55s} Nbar={:8.4f} hit={:.4f}/{:.4f} mean_hit={:7.2f}/{:7.2f} median={:6.2f} (direct mean {:.3f} vs lattice {:.3f}) {:.0f}s".format(
            run["name"], res.nbar, st_u["hit_fraction"], st_cl["hit_fraction"], st_u["mean_dm_hit"], st_cl["mean_dm_hit"],
            st_cl["median_dm_hit"], res.mean_dm_all_rays, lattice_mean_u, time.time() - t0)
              + ("  [dropped {:.2e} crossings/ray beyond the lattice]".format(lattice.dropped) if lattice.dropped else ""), flush=True)
    summary = OUT / "analysis" / "analytic_summary.json"
    old = json.loads(summary.read_text()) if summary.exists() else {}
    old.update(results)
    summary.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
