#!/usr/bin/env python3
"""Quantitative comparison of the analytic halo-model PDFs (run_repro_pdfs.py) with Konietzka's TNG300 catalogue.

A. Torkamani+26 target: TNG300 total LSS DM (DMall_IGM_halo) at z_s = 0.3, 0.5, 0.7, 1.0. Model = BFC hot gas
   (Table 1) to 5 R200c, 1e8-1e16 Msun/h, clustered, shifted by a constant so that its mean is TNG300's mean
   (Torkamani add the mean separately, their Sec. 2.4; only the shape is predicted). Also: their published Fig. 6 curve
   (digitized), their electron convention (x chi_e/mu_e / 0.88 = 0.852), the mean from their eq. 22 with the BFC
   f_IGM, and the Medlock+24 / Macquart+20 closed form (F = 0.182 and TNG300's own F).
B. Halo-only, gas inside R200c: 'all' (window-free) and 'm1e13_to_1e14' nominal / reinterpreted (upper edge x9),
   z_s = 0.1, 0.5, 1.0, for B16, previous Lee22 (x0.6016), corrected Lee22 (F_n = 5.4, 5.57), BFC.
Statistics: hit fraction, mean over all rays, mean / median per hit ray, KS distance (hit rays), per-bin percent
differences (bins with >= 10 TNG rays; median and rms in DM ranges), Poisson chi^2 per bin.
Outputs in analysis/: comparison_halo.csv, comparison_lss.csv, bins_halo.csv, bins_lss.csv, redshift_means.csv,
repro_summary.json."""
import csv
import json
import sys

import numpy as np
from scipy import integrate

import repro_lib as R
import halo_model_dm_pdf as hm

sys.path.insert(0, str(R.ENGINE))
from analytic_functions_total_lss import macquart_pdf  # noqa: E402

RUNS = R.OUT / "analysis" / "runs"
AN = R.OUT / "analysis"
MIN_COUNT = 10
RANGES = ((1, 10), (10, 100), (100, 1000))
F_MEDLOCK = 0.182
ELECTRON_TORKAMANI = (1 - 0.245 / 2) / 1.17 / hm.ELECTRONS_PER_MP      # chi_e / mu_e relative to the engine's 0.88


def load_run(name):
    return np.load(RUNS / (name + ".npz"))


# ------------------------------------------------------------------ B. halo-only
def halo_compare(name, tng_label, z_s):
    d = load_run(name)
    tv = R.tng(tng_label, z_s)
    th = tv[tv > 0]
    grid = d["cdf_grid"]
    c = d["cdf_hit_clustered"]
    hit = float(c[-1])
    cn = c / c[-1]
    mean_all = float(d["mean_dm_all_rays"])
    ks = R.ks_distance(th, grid, cn)
    edges = d["edges"]
    counts, _ = np.histogram(th, edges)
    inside = counts.sum()
    tpdf = counts / (inside * np.diff(edges))
    mpdf = d["pdf_clustered"]
    ok = counts >= MIN_COUNT
    pct = np.where(ok, 100 * (mpdf / np.where(tpdf > 0, tpdf, 1) - 1), np.nan)
    cen = np.sqrt(edges[1:] * edges[:-1])
    expct = mpdf * np.diff(edges) * inside
    chi2 = float(np.sum(((counts - expct) ** 2 / np.maximum(expct, 1e-30))[expct >= MIN_COUNT]) / max(1, np.sum(expct >= MIN_COUNT)))
    out = dict(run=name, tng_label=tng_label, z_s=z_s, hit_model=hit, hit_tng=float(np.mean(tv > 0)),
               mean_all_model=mean_all, mean_all_tng=float(tv.mean()), mean_hit_model=mean_all / hit, mean_hit_tng=float(th.mean()),
               median_hit_model=float(np.interp(0.5, cn, grid)), median_hit_tng=float(np.median(th)), ks_hit=ks,
               chi2_per_bin=chi2, n_bins=int(np.sum(expct >= MIN_COUNT)))
    for lo, hi in RANGES:
        s = ok & (cen >= lo) & (cen < hi)
        out[f"pct_median_{lo}_{hi}"] = float(np.nanmedian(pct[s])) if s.any() else np.nan
        out[f"pct_rms_{lo}_{hi}"] = float(np.sqrt(np.nanmean(pct[s] ** 2))) if s.any() else np.nan
    bins = [dict(run=name, dm_low=edges[b], dm_high=edges[b + 1], tng_count=int(counts[b]), tng_pdf=tpdf[b], model_pdf=mpdf[b],
                 percent=pct[b]) for b in range(len(cen)) if counts[b] > 0 or mpdf[b] > 1e-12]
    return out, bins


# ------------------------------------------------------------------ A. total LSS
def lin_cdf(d, clustered=True):
    g = d["lin_grid"]
    c = d["cdf_full_lin_" + ("clustered" if clustered else "unclustered")]
    c = np.maximum.accumulate(np.clip(c, 0, None))
    return g, c / c[-1]                  # drop the < 2e-4 of wrapped (negative-DM) probability


def cdf_moments(g, c):
    p = np.diff(c)
    x = 0.5 * (g[1:] + g[:-1])
    m = float((p * x).sum())
    return m, float(np.sqrt((p * (x - m) ** 2).sum()))


def macquart_cdf(dm, mean, sigma):
    x = np.linspace(1e-3, 12, 120001)
    p, _ = macquart_pdf(x, sigma)
    c = integrate.cumulative_trapezoid(p, x, initial=0)
    return np.interp(dm / mean, x, c / c[-1])


def bin_from_cdf(edges, cdf_fn):
    return np.diff(cdf_fn(edges)) / np.diff(edges)


def lss_compare(z_s, fig6=None):
    dm = R.tng("lss_total", z_s)
    mu, sd = float(dm.mean()), float(dm.std())
    d = load_run(f"lss_bfc5r200_taper25_zs{z_s}")
    g, c = lin_cdf(d)
    gu, cu = lin_cdf(d, clustered=False)
    mh, sh = cdf_moments(g, c)
    mhu, shu = cdf_moments(gu, cu)
    edges = np.linspace(0.3 * mu, 2.2 * mu, 77)
    cen = 0.5 * (edges[1:] + edges[:-1])
    counts, _ = np.histogram(dm, edges)
    tpdf = counts / (dm.size * np.diff(edges))
    ok = counts >= MIN_COUNT
    core = ok & (np.abs(cen - mu) < sd)
    models = {
        "bfc_halo_model_clustered": lambda x: np.interp(x - (mu - mh), g, c, left=0, right=1),
        "bfc_halo_model_unclustered": lambda x: np.interp(x - (mu - mhu), gu, cu, left=0, right=1),
        "bfc_halo_model_torkamani_electrons": lambda x: np.interp((x - (mu - ELECTRON_TORKAMANI * mh)) / ELECTRON_TORKAMANI, g, c, left=0, right=1),
        "macquart_F0.182_medlock": lambda x: macquart_cdf(x, mu, F_MEDLOCK / np.sqrt(z_s)),
        "macquart_F_tng300": lambda x: macquart_cdf(x, mu, sd / mu),
    }
    if fig6 is not None:
        fx, fp = fig6
        fc = integrate.cumulative_trapezoid(fp, fx, initial=0)
        fc = fc / fc[-1]
        models["torkamani_fig6_digitized"] = lambda x: np.interp(x, fx, fc, left=0, right=1)
    rows, bins = [], []
    for key, fn in models.items():
        mp = bin_from_cdf(edges, fn)
        pct = np.where(ok, 100 * (mp / np.where(tpdf > 0, tpdf, 1) - 1), np.nan)
        if key == "torkamani_fig6_digitized":
            inside = (edges[:-1] >= fx[0]) & (edges[1:] <= fx[-1])
            pct = np.where(inside, pct, np.nan)
            ks = np.nan
        else:
            ks = R.ks_distance(dm, np.linspace(0, 8000, 160001), fn(np.linspace(0, 8000, 160001)))
        sd_model = {"bfc_halo_model_clustered": sh, "bfc_halo_model_unclustered": shu,
                    "bfc_halo_model_torkamani_electrons": ELECTRON_TORKAMANI * sh}.get(key, np.nan)
        xs = np.linspace(0, 8000, 160001)
        med = float(np.interp(0.5, fn(xs), xs)) if key != "torkamani_fig6_digitized" else float(np.interp(0.5, fn(xs), xs))
        rows.append(dict(z_s=z_s, model=key, tng_mean=mu, tng_std=sd, tng_median=float(np.median(dm)), model_std=sd_model,
                         model_median=med, F_tng=sd / mu * np.sqrt(z_s), F_model=sd_model / mu * np.sqrt(z_s),
                         ks=ks, max_abs_pct_within_1sigma=float(np.nanmax(np.abs(pct[core]))) if np.any(np.isfinite(pct[core])) else np.nan,
                         rms_pct_within_1sigma=float(np.sqrt(np.nanmean(pct[core] ** 2))) if np.any(np.isfinite(pct[core])) else np.nan,
                         rms_pct_all_bins=float(np.sqrt(np.nanmean(pct[ok] ** 2))) if np.any(np.isfinite(pct[ok])) else np.nan,
                         halo_model_mean=mh, halo_model_mean_over_tng=mh / mu))
        bins += [dict(z_s=z_s, model=key, dm_low=edges[b], dm_high=edges[b + 1], tng_count=int(counts[b]), tng_pdf=tpdf[b],
                      model_pdf=mp[b], percent=pct[b]) for b in range(len(cen))]
    return rows, bins


def torkamani_mean(z_s, cosmo=R.COSMO):
    """<DM_LSS>(z) of Torkamani+26 eq. 22: (3 Omega_b c H0 / (8 pi G m_p)) chi_e int f_IGM (1+z)/E dz, chi_e = 1 - Y_He/2
    (Y_He = 0.245), f_IGM(z) = int (f_hga/f_b) M dn / int M dn over 1e8-1e16 Msun/h (Tinker08; the printed formula
    without the 1/f_b would give ~0.15)."""
    G = 6.67430e-8
    H0 = 100 * cosmo.h * 1e5 / hm.MPC_CM
    A = 3 * cosmo.omega_b * H0 * 2.99792458e10 / (8 * np.pi * G * hm.M_P_G) / hm.PC_CM
    chi_e = 1 - 0.245 / 2
    lm = np.arange(8.0, 16.0, 0.02) - np.log10(cosmo.h)
    M = 10 ** lm
    fb = cosmo.omega_b / cosmo.omega_m
    zz = np.linspace(0, z_s, 41)
    f = []
    for z in zz:
        dn = hm.tinker08_dndlnm(cosmo, M, z)
        fh = hm.bfc_fractions(cosmo, M)[0] / fb
        f.append(np.trapz(fh * M * dn, lm) / np.trapz(M * dn, lm))
    f = np.array(f)
    integrand = f * (1 + zz) / cosmo.E(zz)
    return float(A * chi_e * np.trapz(integrand, zz)), float(f.mean()), float(A * chi_e * np.trapz((1 + zz) / cosmo.E(zz), zz))


def main():
    halo_rows, halo_bins = [], []
    for z_s in (0.1, 0.5, 1.0):
        for m in R.MODELS:
            for target, abunds in (("all", ("tinker08", "tng300_full")), ("nominal", ("tinker08",)), ("reinterp", ("tng300", "tinker08"))):
                for ab in abunds:
                    name = f"halo_{m}_{target}_{ab}_zs{z_s}"
                    if not (RUNS / (name + ".npz")).exists():
                        print("missing", name); continue
                    r, b = halo_compare(name, "all" if target == "all" else "m1e13_to_1e14", z_s)
                    r.update(model=m, target=target, abundance=ab)
                    halo_rows.append(r); halo_bins += b
    fig6 = np.genfromtxt(R.BFC_FIG6, delimiter=",", names=True)
    lss_rows, lss_bins = [], []
    for z_s in (0.3, 0.5, 0.7, 1.0):
        r, b = lss_compare(z_s, (fig6["dm_pc_cm3"], fig6["pdf_per_pc_cm3"]) if np.isclose(z_s, 0.7) else None)
        lss_rows += r; lss_bins += b
    # taper sensitivity at z_s = 0.7
    taper = {}
    for s in (10, 25, 50):
        g, c = lin_cdf(load_run(f"lss_bfc5r200_taper{s}_zs0.7"))
        taper[s] = cdf_moments(g, c)
    # Fig. 6 curve itself: mean, std; and TNG / our model restricted to the digitized range [331, 1415]
    fx, fp = fig6["dm_pc_cm3"], fig6["pdf_per_pc_cm3"]
    fnorm = np.trapz(fp, fx)
    fmean = np.trapz(fx * fp, fx) / fnorm
    fstd = np.sqrt(np.trapz((fx - fmean) ** 2 * fp, fx) / fnorm)
    dm7 = R.tng("lss_total", 0.7)
    sel = dm7[(dm7 >= fx[0]) & (dm7 <= fx[-1])]
    g7, c7 = lin_cdf(load_run("lss_bfc5r200_taper25_zs0.7"))
    mh7, _ = cdf_moments(g7, c7)
    xx = np.linspace(fx[0], fx[-1], 4001)
    pm = np.gradient(np.interp(xx - (dm7.mean() - mh7), g7, c7), xx)
    pmn = pm / np.trapz(pm, xx)
    mm = np.trapz(xx * pmn, xx)
    fig6_range = dict(range=[float(fx[0]), float(fx[-1])], tng_mean=float(sel.mean()), tng_std=float(sel.std()),
                      tng_fraction_in_range=float(sel.size / dm7.size), fig6_mean=float(fmean), fig6_std=float(fstd),
                      model_mean=float(mm), model_std=float(np.sqrt(np.trapz((xx - mm) ** 2 * pmn, xx))),
                      model_fraction_in_range=float(np.trapz(pm, xx)))
    fpi = np.interp(xx, fx, fp) / fnorm
    core = np.abs(xx - dm7.mean()) < dm7.std()
    fig6_range["model_vs_fig6_rms_pct_within_1sigma"] = float(np.sqrt(np.mean((100 * (pmn[core] / fpi[core] - 1)) ** 2)))
    fig6_range["model_vs_fig6_max_pct_within_1sigma"] = float(np.max(np.abs(100 * (pmn[core] / fpi[core] - 1))))
    print("Fig. 6 range comparison:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fig6_range.items()})
    # redshift dependence of the means
    zgrid = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    tables = {m: R.load_table(m) for m in R.MODELS}
    zrows = []
    mom = {}
    for m, t in tables.items():
        for key, win, ab in (("all_tinker08", (1e9, 1e16), "tinker08"), ("all_tng300_full", (1e9, 1e16), "tng300_full"),
                             ("reinterp_tng300", (1e13, R.C_EXT * 1e14), "tng300"), ("nominal_tinker08", (1e13, 1e14), "tinker08")):
            mom[(m, key)] = R.moments(t, win, zgrid, ab)
    for z in zgrid:
        row = dict(z_s=z, tng_lss_mean=float(R.tng("lss_total", z).mean()), torkamani_eq22_mean=torkamani_mean(z)[0],
                   tng_all_mean=float(R.tng("all", z).mean()), tng_13to14_mean=float(R.tng("m1e13_to_1e14", z).mean()),
                   tng_13to14_hit=float(np.mean(R.tng("m1e13_to_1e14", z) > 0)))
        for (m, key), v in mom.items():
            row[f"{m}|{key}|mean"] = v[z][0]
            row[f"{m}|{key}|hit_unclustered"] = 1 - np.exp(-v[z][1])
        zrows.append(row)
        print(f"z_s {z}: TNG lss {row['tng_lss_mean']:.1f}  Torkamani eq22 {row['torkamani_eq22_mean']:.1f} | TNG all {row['tng_all_mean']:.2f} "
              f"corr5.4 {row['lee22_corr_fn5p40|all_tinker08|mean']:.2f} b16 {row['b16|all_tinker08|mean']:.2f} | TNG 13to14 {row['tng_13to14_mean']:.2f} "
              f"corr5.4 reinterp {row['lee22_corr_fn5p40|reinterp_tng300|mean']:.2f} corr5.57 {row['lee22_corr_fn5p57|reinterp_tng300|mean']:.2f} "
              f"b16 {row['b16|reinterp_tng300|mean']:.2f}", flush=True)
    for fname, rows in (("comparison_halo.csv", halo_rows), ("bins_halo.csv", halo_bins), ("comparison_lss.csv", lss_rows),
                        ("bins_lss.csv", lss_bins), ("redshift_means.csv", zrows)):
        with open(AN / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    summary = dict(halo=halo_rows, lss=lss_rows, taper_sensitivity_z0p7={str(k): dict(mean=v[0], std=v[1]) for k, v in taper.items()},
                   torkamani_fig6=dict(mean=fmean, std=fstd, norm_in_range=fnorm, range=[float(fx[0]), float(fx[-1])]),
                   fig6_range_comparison=fig6_range,
                   torkamani_eq22={str(z): dict(zip(("mean", "f_igm_mean", "mean_f_igm_1"), torkamani_mean(z))) for z in (0.3, 0.5, 0.7, 1.0)},
                   electron_factor_torkamani=ELECTRON_TORKAMANI)
    (AN / "repro_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nTotal LSS")
    for r in lss_rows:
        print(f"  z {r['z_s']} {r['model']:36s} std {r['model_std']:7.1f} (TNG {r['tng_std']:6.1f}) median {r['model_median']:7.1f} "
              f"(TNG {r['tng_median']:7.1f}) KS {r['ks']:.4f} max|%|<1sig {r['max_abs_pct_within_1sigma']:6.1f} rms<1sig "
              f"{r['rms_pct_within_1sigma']:6.1f} rms {r['rms_pct_all_bins']:6.1f}  halo mean/TNG {r['halo_model_mean_over_tng']:.3f}")
    print("Fig. 6 digitized: mean {:.1f} std {:.1f}; taper sensitivity {}".format(fmean, fstd, taper))
    print("\nHalo-only")
    for r in halo_rows:
        print(f"  {r['run']:52s} hit {r['hit_model']:.4f}/{r['hit_tng']:.4f} mean {r['mean_all_model']:7.2f}/{r['mean_all_tng']:7.2f} "
              f"mean/hit {r['mean_hit_model']:7.2f}/{r['mean_hit_tng']:7.2f} med {r['median_hit_model']:6.2f}/{r['median_hit_tng']:6.2f} "
              f"KS {r['ks_hit']:.3f} chi2/bin {r['chi2_per_bin']:8.1f} %med(10-100) {r['pct_median_10_100']:6.1f} %rms(10-100) {r['pct_rms_10_100']:6.1f} "
              f"%med(100-1000) {r['pct_median_100_1000']:6.1f}")


if __name__ == "__main__":
    main()
