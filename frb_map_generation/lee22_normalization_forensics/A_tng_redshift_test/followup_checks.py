#!/usr/bin/env python3
"""Follow-up checks for test A (Lee22 redshift test against TNG's own electron column).

Reuses the model grids of redshift_test.py (model_grids.npz; Tinker08 M200c in Planck15/TNG, electrons inside the
R200c sphere) and the Konietzka TNG300 ray arrays. All TNG quantities use the SNAPSHOT redshift assignment of the
model (Konietzka traces each ray segment through one full TNG snapshot; the continuous-mode crossing-rate ratio jumps
at the snapshot segment boundaries, see lee22_redshift_test_snapshot.png, lower-left panel).

 1. Macquart check of the TNG side: mean of DMall_IGM_halo (total LSS DM, same rays) against the all-baryons-ionized
    mean DM, n_e0 c/H0 int (1+z)/E dz with n_e0 = rho_crit0 Omega_b (1+X_H)/(2 m_p). If Konietzka's electron field had
    a comoving/physical (1+z)^3 error, f_d(z) = TNG/analytic would change by ~8x between z = 0 and 1.
 2. Pseudo mass bins from differences of the cumulative window arrays on the same rays:
        10-11 = 10to14 - 11to14, 11-12 = 11to14 - 12to14, 12-13 = 12to14 - 13to14, 13-14 (direct),
        14-15 = 12to15 - 12to14.
    A ray increment counts as a hit only above 1e-9 pc cm^-3 (the differences carry ~1e-14 float noise).
    C(z) = <dDM>_TNG / <dDM>_model per slice for Lee22 literal, B16 and fb_full, bootstrap over rays, GLS power-law
    fit C0 (1+z)^k over the slices inside [0.1, 2] that TNG populates; also the abundance-corrected C (divided by the
    slice crossing-rate ratio TNG/model), which is the mean DM per crossing ratio and is independent of the HMF.
 3. Cumulative ratios TNG/model versus source redshift (where does the constant 0.602 'work').
 4. Redshift budget of the z_s = 1 halo DM, 1e13-1e14 (TNG and models), and the redshift where C_literal = 0.602.
 5. Candidate constant factors against the z -> 0 anchors of C_literal.
 6. Implied enclosed gas fraction M_gas(<R200c)/(f_b M200c) of the corrected Lee22 model.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import redshift_test as RT      # noqa: E402
import models as M              # noqa: E402

OUT = RT.OUT
MODE = "snapshot"
HIT_THR = 1e-9
N_BOOT = 1000
rng = np.random.default_rng(20260928 + 1)
g = np.load(OUT / "model_grids.npz")
zf, lm_edges = g["zf"], g["lm_edges"]
zeff = g[MODE + "__zeff"]
zs = np.round(np.load(RT.TNG_DIR / "redshifts_DMhalo.npy"), 6)
nsl = len(zs)
slice_lo = np.concatenate([[0.0], zs[:-1]]); slice_hi = zs
member = [(zf >= slice_lo[j] - 1e-9) & (zf < slice_hi[j] - 1e-9) for j in range(nsl)]
COSMO = M.PLANCK15_TNG
h = COSMO.h
report = {}


def load(name):
    return np.load(RT.TNG_DIR / name)


def slice_sum(grid, frac):
    return np.array([(grid[member[j]] * frac).sum() for j in range(nsl)])


def zweighted(grid, frac, zabs):
    return np.array([np.average(zabs[member[j]], weights=(grid[member[j]] * frac).sum(1) + 1e-300) for j in range(nsl)])


# ------------------------------------------------------------------------------------------------ bootstrap
n_rays = 120000
idx_sets = [rng.integers(0, n_rays, n_rays) for _ in range(N_BOOT)]
counts = np.stack([np.bincount(ix, minlength=n_rays) for ix in idx_sets]).astype(np.float64)   # (B, N)


def tng_stats(cum):
    inc = np.diff(np.vstack([np.zeros((1, cum.shape[1])), cum]), axis=0)
    hit = (inc > HIT_THR).astype(np.float64)
    return dict(inc_mean=inc.mean(1), inc_boot=counts @ inc.T / n_rays, hit_mean=hit.mean(1),
                hit_boot=counts @ hit.T / n_rays, cum_mean=cum.mean(1), cum_boot=counts @ cum.T / n_rays,
                per_hit_mean=np.array([inc[j][hit[j] > 0].mean() if hit[j].any() else np.nan for j in range(nsl)]),
                n_hit=hit.sum(1))


# ------------------------------------------------------------------------------------------------ 1. Macquart
tot = load("DMall_IGM_halo.npy")
X_H = 0.76
rho_crit0_cgs = 3 * (100 * h * 1e5 / 3.0856775814913673e24) ** 2 / (8 * np.pi * 6.67430e-8)
ne0 = rho_crit0_cgs * COSMO.omega_b * (1 + X_H) / 2 / (M.M_P * 1e3)                    # comoving cm^-3
c_over_H0_pc = M.C_KMS / (100 * h) * 1e6
zz = np.linspace(0, 6, 60001)
integrand = (1 + zz) / COSMO.E(zz)
cumint = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zz))])
dm_all_ion = ne0 * c_over_H0_pc * np.interp(zs, zz, cumint)
tot_mean = tot.mean(1)
tot_boot = counts @ tot.T / n_rays
fd = tot_mean / dm_all_ion
fd_err = (tot_boot / dm_all_ion).std(0, ddof=1)
inc_tot = np.diff(np.concatenate([[0], tot_mean]))
inc_ana = np.diff(np.concatenate([[0], dm_all_ion]))
fd_slice = inc_tot / inc_ana
sel = zs <= 3.0 + 1e-9
Xf = np.vstack([np.ones(sel.sum()), np.log1p(0.5 * (slice_lo + slice_hi)[sel])]).T
kfd = np.linalg.lstsq(Xf, np.log(fd_slice[sel]), rcond=None)[0]
report["macquart"] = dict(
    ne0_comoving_cm3=ne0, c_over_H0_pc=c_over_H0_pc,
    zs=zs[:30].tolist(), tng_mean_total=tot_mean[:30].tolist(), analytic_all_ionized=dm_all_ion[:30].tolist(),
    fd_cumulative=fd[:30].tolist(), fd_cumulative_err=fd_err[:30].tolist(), fd_slice=fd_slice[:30].tolist(),
    fd_slice_powerlaw_k_z_le_3=float(kfd[1]), fd_slice_powerlaw_A=float(np.exp(kfd[0])))
print("Macquart check: f_d cumulative at z_s = 0.1, 0.5, 1, 2, 3: " +
      ", ".join("{:.3f}".format(fd[int(np.argmin(np.abs(zs - z)))]) for z in (0.1, 0.5, 1, 2, 3)) +
      "; per-slice f_d ~ (1+z)^{:.3f} (z<=3)".format(kfd[1]))

# ------------------------------------------------------------------------------------------------ 2. pseudo mass bins
BINS = {
    "10-11": (lambda: load("DMhalo_r200_mass10to14.npy") - load("DMhalo_r200_mass11to14.npy"), (1e10, 1e11)),
    "11-12": (lambda: load("DMhalo_r200_mass11to14.npy") - load("DMhalo_r200_mass12to14.npy"), (1e11, 1e12)),
    "12-13": (lambda: load("DMhalo_r200_mass12to14.npy") - load("DMhalo_r200_mass13to14.npy"), (1e12, 1e13)),
    "13-14": (lambda: load("DMhalo_r200_mass13to14.npy"), (1e13, 1e14)),
    "14-15": (lambda: load("DMhalo_r200_mass12to15.npy") - load("DMhalo_r200_mass12to14.npy"), (1e14, 1e15)),
    "12-14": (lambda: load("DMhalo_r200_mass12to14.npy"), (1e12, 1e14)),
}
MODELS = ("lee22_literal", "b16", "fb_full")
slice_rows, fit_rows = [], []
per_bin = {}
for name, (loader, (lo, hi)) in BINS.items():
    cum = loader()
    t = tng_stats(cum)
    frac = RT.window_fraction(lm_edges, lo, hi)
    nb = slice_sum(g[MODE + "__nbar"], frac)
    # crossing-rate ratio (Poisson inversion of hit fractions)
    rat = -np.log1p(-t["hit_mean"]) / nb
    ratb = -np.log1p(-np.clip(t["hit_boot"], 0, 0.999999)) / nb[None]
    per_bin[name] = dict(t=t, nb=nb, rat=rat)
    for m in MODELS:
        dm = slice_sum(g[MODE + "__" + m], frac)
        zw = zweighted(g[MODE + "__" + m], frac, zeff)          # redshift of the snapshot the halos come from
        C = t["inc_mean"] / dm
        Cb = t["inc_boot"] / dm[None]
        Ca = C / rat
        Cab = Cb / ratb
        for j in range(nsl):
            slice_rows.append(dict(bin=name, model=m, z_lo=slice_lo[j], z_hi=slice_hi[j], z_snap_weighted=zw[j],
                                   tng_hits=int(t["n_hit"][j]), tng_dDM=t["inc_mean"][j], model_dDM=dm[j],
                                   crossing_rate_ratio=rat[j], C=C[j], C_err=Cb[:, j].std(ddof=1),
                                   C_abundance_corrected=Ca[j], C_abundance_corrected_err=Cab[:, j].std(ddof=1)))
        ok = (slice_lo >= 0.1 - 1e-9) & (slice_hi <= 2.0 + 1e-9) & (t["n_hit"] >= 30)
        s = np.where(ok)[0]
        for tag, CC, CCb in (("C", C, Cb), ("C_abundance_corrected", Ca, Cab)):
            if len(s) < 3 or np.any(CCb[:, s] <= 0) or not np.all(np.isfinite(CCb[:, s])):
                continue
            covl = np.cov(np.log(CCb[:, s]).T)
            beta, cov, chi2 = RT.fit_powerlaw(zw[s], CC[s], covl)
            a3, a3e, chi3 = RT.fit_fixed_k(zw[s], CC[s], covl, 3.0)
            a0, a0e, chi0 = RT.fit_fixed_k(zw[s], CC[s], covl, 0.0)
            # z -> 0 anchor: lowest slice (0 < z < 0.1, snapshots z = 0 and 0.1), extrapolated to z = 0 with the fit k
            fit_rows.append(dict(bin=name, model=m, quantity=tag, n_slices=len(s),
                                 z_max_fit=float(slice_hi[s].max()), C0=float(np.exp(beta[0])),
                                 C0_err=float(np.exp(beta[0]) * np.sqrt(cov[0, 0])), k=float(beta[1]),
                                 k_err=float(np.sqrt(cov[1, 1])), chi2=chi2, dof=len(s) - 2,
                                 C0_fixed_k3=float(np.exp(a3)), C0_fixed_k3_err=float(np.exp(a3) * a3e), chi2_fixed_k3=chi3,
                                 C0_fixed_k0=float(np.exp(a0)), chi2_fixed_k0=chi0,
                                 C_lowest_slice=float(CC[0]), C_lowest_slice_err=float(CCb[:, 0].std(ddof=1)),
                                 z_lowest_slice=float(zw[0]), C_z0_extrap_fitk=float(CC[0] / (1 + zw[0]) ** beta[1]),
                                 C_z0_extrap_k3=float(CC[0] / (1 + zw[0]) ** 3)))
    print("bin {:6s}: TNG hits per slice (first 8) {}".format(name, t["n_hit"][:8].astype(int).tolist()))

for r in fit_rows:
    print("{bin:6s} {model:14s} {quantity:22s} n={n_slices:2d} zmax={z_max_fit:.1f} C0={C0:.3f}+-{C0_err:.3f} "
          "k={k:+.3f}+-{k_err:.3f} chi2/dof={chi2:.0f}/{dof}  C(z={z_lowest_slice:.3f})={C_lowest_slice:.3f}"
          "+-{C_lowest_slice_err:.3f} -> z=0 (k fit) {C_z0_extrap_fitk:.3f}".format(**r))

# ------------------------------------------------------------------------------------------------ 3. cumulative ratios vs z_s
cum_rows = []
XGP = M.XGP_NATIVE_FACTOR
for name in ("13-14", "12-14"):
    t = per_bin[name]["t"]
    lo, hi = BINS[name][1]
    frac = RT.window_fraction(lm_edges, lo, hi)
    dm = {m: np.cumsum(slice_sum(g[MODE + "__" + m], frac)) for m in ("lee22_literal", "b16", "fb_full")}
    nbc = np.cumsum(per_bin[name]["nb"])
    for j in range(30):
        row = dict(bin=name, z_s=zs[j], tng_mean_all=t["cum_mean"][j], tng_mean_all_err=t["cum_boot"][:, j].std(ddof=1),
                   ratio_tng_over_lee22_literal=t["cum_mean"][j] / dm["lee22_literal"][j],
                   ratio_tng_over_lee22_xgpnative=t["cum_mean"][j] / (XGP * dm["lee22_literal"][j]),
                   ratio_tng_over_b16=t["cum_mean"][j] / dm["b16"][j],
                   ratio_tng_over_fb_full=t["cum_mean"][j] / dm["fb_full"][j],
                   ratio_err_frac=t["cum_boot"][:, j].std(ddof=1) / t["cum_mean"][j],
                   tng_hit_cum=float((load("DMhalo_r200_mass13to14.npy" if name == "13-14" else
                                           "DMhalo_r200_mass12to14.npy")[j] > 0).mean()),
                   model_hit_cum_unclustered=float(-np.expm1(-nbc[j])))
        cum_rows.append(row)
print("\ncumulative TNG / model vs z_s (snapshot assignment):")
for r in cum_rows:
    if r["z_s"] in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        print("  {bin:6s} z_s={z_s:.1f}: /literal {ratio_tng_over_lee22_literal:.3f}  /xgpnative {ratio_tng_over_lee22_xgpnative:.3f}"
              "  /B16 {ratio_tng_over_b16:.3f}  /fb {ratio_tng_over_fb_full:.3f} (+-{ratio_err_frac:.3%})  hit TNG "
              "{tng_hit_cum:.4f} model {model_hit_cum_unclustered:.4f}".format(**r))

# ------------------------------------------------------------------------------------------------ 4. z budget at z_s = 1
fit13 = [r for r in fit_rows if r["bin"] == "13-14" and r["model"] == "lee22_literal" and r["quantity"] == "C"][0]
t = per_bin["13-14"]["t"]
frac = RT.window_fraction(lm_edges, 1e13, 1e14)
j1 = int(np.argmin(np.abs(zs - 1.0)))
lit_slice = slice_sum(g[MODE + "__lee22_literal"], frac)
corr_grid = g[MODE + "__lee22_literal"] * (fit13["C0"] * (1 + zeff) ** fit13["k"])[:, None]
corr_k3_grid = g[MODE + "__lee22_literal"] * (fit13["C0_fixed_k3"] * (1 + zeff) ** 3)[:, None]
budget = dict(TNG=t["inc_mean"][:j1 + 1], lee22_literal=lit_slice[:j1 + 1], lee22_xgpnative=XGP * lit_slice[:j1 + 1],
              b16=slice_sum(g[MODE + "__b16"], frac)[:j1 + 1],
              lee22_corrected_free=slice_sum(corr_grid, frac)[:j1 + 1],
              lee22_corrected_k3=slice_sum(corr_k3_grid, frac)[:j1 + 1])
tng_frac_boot = t["inc_boot"][:, :j1 + 1] / t["inc_boot"][:, :j1 + 1].sum(1, keepdims=True)
zmid = 0.5 * (slice_lo + slice_hi)[:j1 + 1]
budget_rows = []
for j in range(j1 + 1):
    row = dict(z_lo=slice_lo[j], z_hi=slice_hi[j], TNG_frac_err=tng_frac_boot[:, j].std(ddof=1))
    for k_, v in budget.items():
        row[k_ + "_dDM"] = v[j]
        row[k_ + "_frac"] = v[j] / v.sum()
    budget_rows.append(row)
bsum = {}
for k_, v in budget.items():
    f = v / v.sum()
    cf = np.cumsum(f)
    bsum[k_] = dict(total=float(v.sum()), mean_z=float((f * zmid).sum()),
                    median_z=float(np.interp(0.5, np.concatenate([[0], cf]), np.concatenate([[0], slice_hi[:j1 + 1]]))),
                    frac_z_lt_0p3=float(f[:3].sum()), frac_z_0p3_0p6=float(f[3:6].sum()), frac_z_0p6_1=float(f[6:].sum()))
z_cross = (XGP / fit13["C0"]) ** (1 / fit13["k"]) - 1
z_cross_k3 = (XGP / fit13["C0_fixed_k3"]) ** (1 / 3) - 1
report["zs1_budget_13_14"] = dict(summary=bsum, z_where_C_literal_equals_0p602_free_fit=z_cross,
                                   z_where_C_literal_equals_0p602_k3_fit=z_cross_k3, fit_used=fit13,
                                   C_eff_zs1_tng_over_literal=float(t["cum_mean"][j1] / lit_slice[:j1 + 1].sum()))
print("\nz_s = 1 halo-DM budget 1e13-1e14 (snapshot assignment):")
for k_, v in bsum.items():
    print("  {:22s} total {total:7.2f}  mean z {mean_z:.3f}  median z {median_z:.3f}  z<0.3 {frac_z_lt_0p3:.3f}  "
          "0.3-0.6 {frac_z_0p3_0p6:.3f}  0.6-1 {frac_z_0p6_1:.3f}".format(k_, **v))
print("  C_literal = 0.602 at z = {:.3f} (free fit), {:.3f} (k = 3 fit); C_eff(z_s=1) = TNG/literal = {:.4f}".format(
    z_cross, z_cross_k3, report["zs1_budget_13_14"]["C_eff_zs1_tng_over_literal"]))

# ------------------------------------------------------------------------------------------------ 5. candidates
CAND = dict(RT.CANDIDATES)
anchors = {}
for r in fit_rows:
    # 14-15 excluded: Konietzka's 12to15 - 12to14 difference has ~3 % of the expected cluster crossings and none at z > 0.6
    if r["model"] == "lee22_literal" and r["bin"] in ("12-13", "13-14", "12-14"):
        anchors["{}|{}|z0_extrap_fitk".format(r["bin"], r["quantity"])] = r["C_z0_extrap_fitk"]
        anchors["{}|{}|z0_extrap_k3".format(r["bin"], r["quantity"])] = r["C_z0_extrap_k3"]
        anchors["{}|{}|C0_fixed_k3".format(r["bin"], r["quantity"])] = r["C0_fixed_k3"]
vals = np.array(list(anchors.values()))
report["candidates"] = dict(values=CAND, anchors=anchors, anchor_min=float(vals.min()), anchor_max=float(vals.max()),
                            anchor_median=float(np.median(vals)),
                            ratio_anchor_median_over_candidate={k_: float(np.median(vals) / v) for k_, v in CAND.items()})
nb1415 = per_bin["14-15"]
report["bin_14_15_incompleteness"] = dict(tng_hits_per_slice=nb1415["t"]["n_hit"][:20].astype(int).tolist(),
                                          model_expected_hits_per_slice=(n_rays * -np.expm1(-nb1415["nb"][:20])).tolist(),
                                          crossing_rate_ratio=nb1415["rat"][:20].tolist())
print("14-15 bin: TNG hits / model-expected hits, first 6 slices: " + ", ".join(
    "{:d}/{:.0f}".format(int(a_), b_) for a_, b_ in zip(nb1415["t"]["n_hit"][:6], n_rays * -np.expm1(-nb1415["nb"][:6]))))
print("\nz->0 anchors of C_literal: min {:.3f} median {:.3f} max {:.3f}".format(vals.min(), np.median(vals), vals.max()))
for k_, v in CAND.items():
    print("  {:18s} {:.3f}  anchor-median/candidate = {:.2f}".format(k_, v, np.median(vals) / v))

# ------------------------------------------------------------------------------------------------ 6. implied gas fractions
gf_rows = []
mu_ratio = (2 / (1 + X_H)) / X_H          # M_gas / (f_b M) = <n_e/n200> * mu_e / X_H
for lmh in (13.0, 13.2, 13.61, 14.0, 14.5):
    mass = 10 ** lmh / h
    for z in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        nb_ = float(M.lee22_mean_nbar(np.array([mass]), np.array([z]), h)[0])
        lit = mu_ratio * nb_
        gf_rows.append(dict(log10_M200c_hinv=lmh, z=z, mean_ne_over_n200=nb_, fgas_over_fb_literal=lit,
                            fgas_over_fb_xgpnative=lit * XGP,
                            fgas_over_fb_corr_h3_1pz3=lit * h ** 3 * (1 + z) ** 3,
                            fgas_over_fb_corr_free_13to14=lit * fit13["C0"] * (1 + z) ** fit13["k"]))
print("\nimplied M_gas(<R200c)/(f_b M200c):")
for r in gf_rows:
    if r["z"] in (0.0, 0.5, 1.0, 2.0):
        print("  logMh={log10_M200c_hinv:5.2f} z={z:.1f}: literal {fgas_over_fb_literal:.3f}  xgpnative "
              "{fgas_over_fb_xgpnative:.3f}  h^3(1+z)^3 {fgas_over_fb_corr_h3_1pz3:.3f}  fitted C(z) "
              "{fgas_over_fb_corr_free_13to14:.3f}".format(**r))

# ------------------------------------------------------------------------------------------------ write
for fname, rows in (("followup_pseudo_bin_slices.csv", slice_rows), ("followup_pseudo_bin_fits.csv", fit_rows),
                    ("followup_cumulative_ratios_vs_zs.csv", cum_rows), ("followup_zs1_budget_13to14.csv", budget_rows),
                    ("followup_implied_gas_fractions.csv", gf_rows)):
    with open(OUT / fname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k_: (float(v) if isinstance(v, np.floating) else v) for k_, v in r.items()})
with open(OUT / "followup_macquart_check.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["z_s", "tng_mean_total_dm", "tng_err", "analytic_all_baryons_ionized", "fd_cumulative", "fd_slice"])
    for j in range(nsl):
        w.writerow([zs[j], tot_mean[j], tot_boot[:, j].std(ddof=1), dm_all_ion[j], fd[j], fd_slice[j]])
(OUT / "followup_summary.json").write_text(json.dumps(report, indent=1, default=float))
