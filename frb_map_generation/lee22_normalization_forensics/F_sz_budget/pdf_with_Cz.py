#!/usr/bin/env python3
"""Does the FRB halo-DM PDF at z_s = 1 (1e13-1e14 Msun, gas inside R200c) discriminate a constant Lee22 normalization
(0.6016, 'xgpnative') from the redshift-dependent C(z) = C0 (1+z)^k that the TNG z_s test requires?

Uses the repo's analytic compound-Poisson engine (Torkamani+26 formalism; halo_model_dm_pdf.py and the slice builder of
run_analytic_halo_pdfs.py, imported unchanged) with the exact HalfDome Lee22-xgpnative sphere table (z <= 1), Planck15
(TNG) cosmology and Tinker08 abundance, and rescales every z-slice's DM by C(z)/0.6016. Compares with the Konietzka
TNG300 within-R200 hit-ray PDF at z_s = 1. Writes pdf_with_Cz.json/.npz and a figure in the F_sz_budget output dir."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cbllover/HalfDome/frb_map_generation/analytic_pdf_torkamani2026")
import halo_model_dm_pdf as hm  # noqa: E402
import run_analytic_halo_pdfs as R  # noqa: E402  (module import only: paths, slice builder)

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget")
KON = Path("/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM")
XGP = 0.6016316602183194
res = json.loads((OUT / "budget_results.json").read_text())
fits = res["konietzka_zs_test"]["fits_zs_0p1_to_2"]
k0, k1 = fits["C0(1+z)^k"]["params"]
c3 = fits["C0(1+z)^3"]["params"][0]
MODELS = {"const_0.6016": lambda z: XGP,
          "tng_fit_{:.3f}(1+z)^{:.2f}".format(k0, k1): lambda z: k0 * (1 + z) ** k1,
          "tng_fit_{:.3f}(1+z)^3".format(c3): lambda z: c3 * (1 + z) ** 3}


def scaled(gen, C):
    for s in gen:
        f = C(s["z"]) / XGP
        s = dict(s)
        s["dm_edges"] = s["dm_edges"] * f
        s["mean_dm"] = s["mean_dm"] * f
        yield s


def truncated(gen, z_s):
    for s in gen:
        if s["z"] < z_s:
            yield s


def main():
    lattice = hm.Lattice(h=0.002, n=2 ** 24)
    x_edges = hm.impact_samples()
    table = hm.HalfDomeSphereTable(R.TABLE, "lee22_noconc_xgpnative")
    cosmo = hm.PLANCK15_TNG
    edges = R.PDF_EDGES
    zs = np.load(KON / "redshifts_DMhalo.npy")
    allrays = np.load(KON / "DMhalo_r200_mass13to14.npy")
    out = {}
    arrays = dict(edges=edges)
    for z_s in (0.1, 0.3, 0.5, 1.0):
        a = allrays[int(np.argmin(np.abs(zs - z_s)))]
        pos = a[(a > 0)]
        inr = pos[(pos >= edges[0]) & (pos < edges[-1])]
        h_tng, _ = np.histogram(inr, bins=edges)
        p_tng = h_tng / (h_tng.sum() * np.diff(edges))
        o = out["zs={}".format(z_s)] = dict(tng=dict(hit_fraction=float((a > 0).mean()), mean_dm_hit=float(pos.mean()),
                                                     median_dm_hit=float(np.median(pos)), p16=float(np.percentile(pos, 16)),
                                                     p84=float(np.percentile(pos, 84)), n_hit=int(pos.size)), models={})
        arrays["pdf_tng_zs{}".format(z_s)] = p_tng
        for name, C in MODELS.items():
            lattice.dropped = 0.0
            gen = truncated(R.slices_for("lee22_noconc_xgpnative", "tinker08", R.WINDOWS["m1e13_to_1e14"], cosmo, x_edges, table), z_s)
            r = hm.compound_poisson_pdf(lattice, scaled(gen, C), cosmo=cosmo)
            pdf, prob = r.binned(edges, clustered=True)
            st = r.stats(True)
            c = np.cumsum(prob) / prob.sum()
            good = h_tng >= 10
            wts = h_tng[good] / h_tng[good].sum()
            rel = pdf[good] / p_tng[good] - 1
            o["models"][name] = dict(stats_clustered=st, p16=float(np.interp(0.16, c, edges[1:])), p84=float(np.interp(0.84, c, edges[1:])),
                                     median_binned=float(np.interp(0.5, c, edges[1:])),
                                     mean_abs_pct=float(100 * np.mean(np.abs(rel))),
                                     tng_weighted_mean_abs_pct=float(100 * np.sum(wts * np.abs(rel))),
                                     mean_dm_all_rays=float(r.mean_dm_all_rays))
            arrays["pdf_{}_zs{}".format(name, z_s)] = pdf
            print(z_s, name, "hit", round(st["hit_fraction"], 4), "median", round(st["median_dm_hit"], 1), "mean", round(st["mean_dm_hit"], 1),
                  "wMAE", round(o["models"][name]["tng_weighted_mean_abs_pct"], 1), "| TNG median", round(o["tng"]["median_dm_hit"], 1),
                  "mean", round(o["tng"]["mean_dm_hit"], 1), flush=True)
    (OUT / "pdf_with_Cz.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / "pdf_with_Cz.npz", **arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cen = np.sqrt(edges[1:] * edges[:-1])
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
    for j, z_s in enumerate((0.1, 0.3, 0.5, 1.0)):
        p_tng = arrays["pdf_tng_zs{}".format(z_s)]
        ax = axes[:, j]
        ax[0].plot(cen, cen * p_tng, color="#0b0b0b", lw=2, label="TNG300 (Konietzka), within R200")
        for (name, col) in zip(MODELS, ("#2a78d6", "#1baf7a", "#eda100")):
            pm = arrays["pdf_{}_zs{}".format(name, z_s)]
            ax[0].plot(cen, cen * pm, color=col, lw=1.8, label="Lee22 x " + name.replace("_", " "))
            with np.errstate(divide="ignore", invalid="ignore"):
                ax[1].plot(cen, pm / p_tng - 1, color=col, lw=1.4)
        ax[1].axhline(0, color="#52514e", lw=0.8); ax[1].set_ylim(-0.6, 0.6)
        ax[0].set_xscale("log"); ax[0].set_xlim(1, 3000); ax[0].set_title("z_s = {}, 1e13-1e14 Msun, hit rays".format(z_s), fontsize=11)
        ax[1].set_xlabel("halo DM [pc cm^-3]")
        for x in ax:
            x.grid(alpha=.3)
    axes[0, 0].set_ylabel("DM p(DM)"); axes[1, 0].set_ylabel("model/TNG - 1")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "pdf_with_Cz.png", dpi=160)


if __name__ == "__main__":
    main()
