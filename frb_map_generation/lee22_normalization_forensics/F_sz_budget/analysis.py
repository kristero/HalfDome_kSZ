#!/usr/bin/env python3
"""Redshift / mass budget of kSZ and tSZ halo-model power, FRB halo-DM weight, candidate normalization corrections.

Reads tables_{validation,physical}.npz from run_budget.py; writes budget_results.json, budget_*.csv and figures."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import szhm as S  # noqa: E402

OUT = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget")
KONIETZKA = Path("/home/cbllover/HalfDome/frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM")
trapz = S._trapz
H_TNG = 0.6774
XGP = S.XGPNATIVE_FACTOR
ELL_REPORT = (300, 1000, 3000)
ZCUTS = (0.25, 0.5, 1.0, 2.0)

# painted full-lightcone references (read from the repo; see RUN_REPORT.md / ksz_comparison.json / cl_yy files)
PAINTED_KSZ = {  # D_ell [uK^2], (2l+1)-weighted band means
    1000: dict(lee22_xgpnative=1.2425640817679164, lee22_literal=3.432870578764547, b16=0.3467822968027848,
               class_b16=0.40088799902905053),
    3000: dict(lee22_xgpnative=0.8885474065059171, lee22_literal=2.4548176584114616, b16=0.6388571683192528,
               class_b16=0.6850470818468528),
    5000: dict(lee22_xgpnative=0.4984359530746449, lee22_literal=1.3770445676120906, b16=0.6820529308313514,
               class_b16=0.6942433266193464)}


def painted_ksz_ell300():
    """(2l+1)-weighted band 300-327 of ksz_binned_comparison.csv (effective ell 313)."""
    p = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922/ksz_binned_comparison.csv")
    with p.open() as f:
        for row in csv.DictReader(f):
            if row["ell_min"] == "300":
                return dict(ell_eff=float(row["ell_effective"]), lee22_xgpnative=float(row["full_lee22_previous"]),
                            lee22_literal=float(row["full_lee22_paper"]), b16=float(row["full_battaglia16"]),
                            class_b16=float(row["full_class_sz_b16"]))


def painted_tsz_ratio():
    a = np.load("/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_dm_cross_updated_20260917/rays/annular_y_samples_lee22_pressure.npz")
    b = np.load("/home/cbllover/HalfDome/frb_map_generation/outputs/takahashi_cross_comparison_20260913/spectra/battaglia16.npz")
    cl1, cl2 = a["cl_yy"], b["cl_yy"]
    out = {}
    for l in (30, 100, 300, 1000, 2000, 3000, 5000, 8000):
        s = slice(int(l * 0.9), int(l * 1.1) + 1)
        out[l] = dict(ratio=float(cl1[s].mean() / cl2[s].mean()),
                      D_b12=float((np.arange(len(cl2))[s] * (np.arange(len(cl2))[s] + 1) / 2 / np.pi * cl2[s]).mean()),
                      D_lee22p=float((np.arange(len(cl1))[s] * (np.arange(len(cl1))[s] + 1) / 2 / np.pi * cl1[s]).mean()))
    return out, dict(mean_y_b12=1.0415762786251156e-06, mean_y_lee22p=4.5617586530636036e-07)


# ------------------------------------------------------------------------------------------------ spectra
TAG = ""


def load(run):
    d = dict(np.load(OUT / "tables_{}{}.npz".format(run, TAG if run == "validation" else "")))
    d["meta"] = json.loads(str(d["meta"]))
    return d


def spectra(d, key, kind):
    z, M, ell = d["z"], d["M"], d["ell"]
    lnM = np.log(M)
    dVdz = d["chi"] ** 2 / d["Hc"]
    W = S.T_CMB_UK ** 2 * d["v2"] / S.C_KMS ** 2 if kind == "ksz" else np.ones_like(z)
    X = d[key]
    dn = d["dn"]
    d1 = dVdz[:, None, None] * W[:, None, None] * dn[:, :, None] * X ** 2          # dC^1h / dz dlnM
    dC1 = trapz(d1, lnM, axis=1)
    I = trapz((dn * d["bias"])[:, :, None] * X, lnM, axis=1)
    k = (ell[None, :] + 0.5) / d["chi"][:, None]
    P = d["D"][:, None] ** 2 * np.exp(np.interp(np.log(k), np.log(d["k_lin"]), np.log(d["p_lin0"])))
    dC2 = dVdz[:, None] * W[:, None] * I ** 2 * P
    return dict(d1=d1, dC1=dC1, dC2=dC2, z=z, lnM=lnM, ell=ell)


def catalogue_spectra(d, key, kind, cw):
    """Catalogue-weighted halo sums on the table nodes: 1h = (1/4pi) sum_i W_i |X_l|^2 over all painted halos,
    2h = sum_zbins W P (sum_j N b X)^2 / (16 pi^2 dV/dz dz) (Tinker10 bias). Returns per-z-bin contributions."""
    z = d["z"]
    X = d[key]
    N, V = cw["N"], cw["V"]
    ze = np.concatenate([[d["meta"]["grid"]["z_lo"]], np.sqrt(z[1:] * z[:-1]), [d["meta"]["grid"]["z_hi"]]])
    dz = np.diff(ze)
    Wm = (S.T_CMB_UK ** 2 * V) if kind == "ksz" else N                      # per-(z,M) velocity-weighted counts
    c1 = (Wm[:, :, None] * X ** 2).sum(1) / (4 * np.pi)                     # (nz, nell): contribution of each z bin
    dVdz = d["chi"] ** 2 / d["Hc"]
    Wz = S.T_CMB_UK ** 2 * V.sum(1) / np.maximum(N.sum(1), 1) if kind == "ksz" else np.ones_like(z)
    I = (N[:, :, None] * d["bias"][:, :, None] * X).sum(1)
    ell = d["ell"]
    k = (ell[None, :] + 0.5) / d["chi"][:, None]
    P = d["D"][:, None] ** 2 * np.exp(np.interp(np.log(k), np.log(d["k_lin"]), np.log(d["p_lin0"])))
    c2 = Wz[:, None] * P * I ** 2 / (16 * np.pi ** 2 * dVdz[:, None] * dz[:, None])
    return dict(c1=c1, c2=c2, dz=dz, ell=ell, z=z)


def cl_total(sp, corr=None, which="both"):
    w = np.ones_like(sp["z"]) if corr is None else corr(sp["z"]) ** 2
    dd = (sp["dC1"] if which in ("both", "1h") else 0) + (sp["dC2"] if which in ("both", "2h") else 0)
    return trapz(w[:, None] * dd, sp["z"], axis=0)


def Dl(sp, cl):
    return sp["ell"] * (sp["ell"] + 1) / (2 * np.pi) * cl


def at(sp, arr, l):
    i = int(np.argmin(np.abs(sp["ell"] - l)))
    assert sp["ell"][i] == l, (l, sp["ell"][i])
    return float(arr[i])


def zbudget(sp, l, corr=None):
    i = int(np.argmin(np.abs(sp["ell"] - l)))
    w = np.ones_like(sp["z"]) if corr is None else corr(sp["z"]) ** 2
    dz = w * (sp["dC1"][:, i] + sp["dC2"][:, i])
    tot = trapz(dz, sp["z"])
    cum = np.concatenate([[0], np.cumsum(0.5 * (dz[1:] + dz[:-1]) * np.diff(sp["z"]))]) / tot
    # add the sliver below the first node (z_lo) as zero: grids start at z_lo
    out = {"frac_z_lt_{}".format(c): float(np.interp(c, sp["z"], cum)) for c in ZCUTS}
    out.update(z_median=float(np.interp(0.5, cum, sp["z"])), z_p16=float(np.interp(0.16, cum, sp["z"])),
               z_p84=float(np.interp(0.84, cum, sp["z"])), frac_2h=float(trapz(w * sp["dC2"][:, i], sp["z"]) / tot))
    # mass budget of the 1-halo term
    dm = trapz(w[:, None] * sp["d1"][:, :, i], sp["z"], axis=0)
    cm = np.concatenate([[0], np.cumsum(0.5 * (dm[1:] + dm[:-1]) * np.diff(sp["lnM"]))]) / trapz(dm, sp["lnM"])
    lm10 = sp["lnM"] / np.log(10)
    out.update(frac_1h_M_above_fit=float(1 - np.interp(14.8 - np.log10(0.68), lm10, cm)),
               frac_1h_M_below_fit=float(np.interp(13.0 - np.log10(0.68), lm10, cm)) if lm10[0] < 13.0 - np.log10(0.68) else 0.0)
    out.update(log10M_median_1h=float(np.interp(0.5, cm, sp["lnM"]) / np.log(10)),
               log10M_p16_1h=float(np.interp(0.16, cm, sp["lnM"]) / np.log(10)),
               log10M_p84_1h=float(np.interp(0.84, cm, sp["lnM"]) / np.log(10)))
    return out, dz / tot, dm / trapz(dm, sp["lnM"])


# ------------------------------------------------------------------------------------------ FRB weights
def frb_weight(cosmo, model, z_s, m_lo=1e13, m_hi=1e14, nz=240, nm=81, X=1.0):
    """d<DM>/dz over ALL rays [pc cm^-3 per unit z] and d Nbar/dz for halos in [m_lo, m_hi) inside an X R200c sphere."""
    z = np.linspace(1e-4, z_s, nz)
    M = np.geomspace(m_lo, m_hi, nm)
    lnM = np.log(M)
    chi = cosmo.chi(z)
    dVdz = chi ** 2 / cosmo.H_over_c(z)
    dDM = np.empty(nz); dN = np.empty(nz)
    for i, zz in enumerate(z):
        amp, shape = S.density_shape(model, cosmo, M, zz)
        r = S.r200c_mpc(cosmo, M, zz)
        Ne = amp * (r * S.MPC_CM) ** 3 * S.enclosed(shape, X, n=2048)
        DA = chi[i] / (1 + zz)
        dn = S.tinker08_dndlnm(cosmo, M, zz)
        dDM[i] = dVdz[i] * trapz(dn * Ne / ((DA * S.MPC_CM) ** 2 * (1 + zz)) / S.PC_CM, lnM)
        dN[i] = dVdz[i] * trapz(dn * np.pi * (X * r / DA) ** 2, lnM)
    return z, dDM, dN


def main():
    res = {}
    OUT.mkdir(parents=True, exist_ok=True)
    painted300 = painted_ksz_ell300()
    tsz_painted, meany_painted = painted_tsz_ratio()
    runs = {}
    for run in ("validation", "physical"):
        d = load(run)
        sp = {"lee22_literal": spectra(d, "tau_lee22_literal", "ksz"), "b16": spectra(d, "tau_b16", "ksz"),
              "lee22p": spectra(d, "y_lee22p", "tsz"), "b12": spectra(d, "y_b12", "tsz")}
        runs[run] = (d, sp)
        r = res[run] = dict(grid=d["meta"]["grid"], nz=d["meta"]["nz"], nm=d["meta"]["nm"])
        # kSZ D_ell (uK^2)
        k = r["ksz_D_ell_uK2"] = {}
        for l in (300, 1000, 2000, 3000, 5000):
            row = {}
            for name in ("lee22_literal", "b16"):
                for which in ("1h", "2h", "both"):
                    row["{}_{}".format(name, which)] = at(sp[name], Dl(sp[name], cl_total(sp[name], which=which)), l)
            row["lee22_xgpnative_both"] = row["lee22_literal_both"] * XGP ** 2
            row["ratio_xgpnative_over_b16"] = row["lee22_xgpnative_both"] / row["b16_both"]
            row["ratio_literal_over_b16"] = row["lee22_literal_both"] / row["b16_both"]
            k[l] = row
        # tSZ C_ell ratios and D_ell
        t = r["tsz"] = {}
        for l in (30, 100, 300, 1000, 2000, 3000, 5000, 8000):
            row = {}
            for name in ("lee22p", "b12"):
                for which in ("1h", "2h", "both"):
                    row["D_{}_{}".format(name, which)] = at(sp[name], Dl(sp[name], cl_total(sp[name], which=which)), l)
            row["ratio_lee22p_over_b12"] = row["D_lee22p_both"] / row["D_b12_both"]
            t[l] = row
        dVdz = d["chi"] ** 2 / d["Hc"]
        lnM = np.log(d["M"])
        r["mean_y"] = {name: float(trapz(dVdz * trapz(d["dn"] * d["y0_" + name], lnM, axis=1), d["z"]))
                       for name in ("lee22p", "b12")}
        r["mean_tau_3r200"] = {name: float(trapz(dVdz * trapz(
            d["dn"] * S.SIGMA_T * d["Ne3_" + name] / (d["chi"][:, None] / (1 + d["z"][:, None]) * S.MPC_CM) ** 2, lnM, axis=1),
            d["z"])) for name in ("lee22_literal", "b16")}
        # budgets
        b = r["budget"] = {}
        for name, ells in (("lee22_literal", ELL_REPORT), ("b16", ELL_REPORT), ("lee22p", ELL_REPORT + (8000,)),
                           ("b12", ELL_REPORT + (8000,))):
            b[name] = {l: zbudget(sp[name], l)[0] for l in ells}
    # validation summary against the painted maps
    v = res["validation"]
    val = res["validation_vs_painted"] = {}
    for l in (1000, 3000, 5000):
        row = v["ksz_D_ell_uK2"][l]
        p = PAINTED_KSZ[l]
        val["ksz_{}".format(l)] = dict(
            hm_b16=row["b16_both"], painted_b16=p["b16"], class_b16=p["class_b16"],
            hm_xgpnative=row["lee22_xgpnative_both"], painted_xgpnative=p["lee22_xgpnative"],
            hm_literal=row["lee22_literal_both"], painted_literal=p["lee22_literal"],
            hm_ratio_xgp_over_b16=row["ratio_xgpnative_over_b16"], painted_ratio_xgp_over_b16=p["lee22_xgpnative"] / p["b16"],
            ratio_of_ratios=row["ratio_xgpnative_over_b16"] / (p["lee22_xgpnative"] / p["b16"]))
    # catalogue-weighted sums (exact painted halo population and velocities)
    cw = dict(np.load(OUT / "catalogue_weights_validation.npz"))
    d_val = runs["validation"][0]
    cat = res["validation_catalogue_weighted"] = {}
    csp = {}
    for name, key, kind in (("lee22_literal", "tau_lee22_literal", "ksz"), ("b16", "tau_b16", "ksz"),
                            ("lee22p", "y_lee22p", "tsz"), ("b12", "y_b12", "tsz")):
        c = csp[name] = catalogue_spectra(d_val, key, kind, cw)
        fac = c["ell"] * (c["ell"] + 1) / (2 * np.pi)
        cat[name] = {int(l): dict(D_1h=float((fac * c["c1"].sum(0))[c["ell"] == l][0]),
                                  D_2h=float((fac * c["c2"].sum(0))[c["ell"] == l][0]))
                     for l in (30, 100, 300, 1000, 2000, 3000, 5000, 8000)}
        for l in cat[name]:
            cat[name][l]["D"] = cat[name][l]["D_1h"] + cat[name][l]["D_2h"]
    for l in (300, 1000, 3000, 5000):
        a_, b_ = cat["lee22_literal"][l]["D"] * XGP ** 2, cat["b16"][l]["D"]
        cat["ksz_ratio_xgp_over_b16_{}".format(l)] = a_ / b_
    for l in (30, 100, 300, 1000, 2000, 3000, 5000, 8000):
        cat["tsz_ratio_lee22p_over_b12_{}".format(l)] = cat["lee22p"][l]["D"] / cat["b12"][l]["D"]
    cat["catalogue_mean_y"] = {name: float((cw["N"] * d_val["y0_" + name]).sum() / (4 * np.pi)) for name in ("lee22p", "b12")}
    row = v["ksz_D_ell_uK2"][300]
    val["ksz_300"] = dict(hm_b16=row["b16_both"], painted_b16_band313=painted300["b16"], class_b16=painted300["class_b16"],
                          hm_xgpnative=row["lee22_xgpnative_both"], painted_xgpnative=painted300["lee22_xgpnative"],
                          hm_ratio_xgp_over_b16=row["ratio_xgpnative_over_b16"],
                          painted_ratio_xgp_over_b16=painted300["lee22_xgpnative"] / painted300["b16"])
    for l in (30, 100, 300, 1000, 2000, 3000, 5000, 8000):
        val["tsz_{}".format(l)] = dict(hm_ratio=v["tsz"][l]["ratio_lee22p_over_b12"], painted_ratio=tsz_painted[l]["ratio"],
                                       hm_D_b12=v["tsz"][l]["D_b12_both"], painted_D_b12=tsz_painted[l]["D_b12"])
    val["mean_y"] = dict(hm=v["mean_y"], painted=meany_painted)

    # ------------------------------------------------------------ FRB weights (paint cosmology, like the HalfDome rays)
    cosmo = S.PAINT_COSMO
    zf, wlit, dN = frb_weight(cosmo, "lee22_literal", 1.0)
    frb = res["frb_zs1_1e13_1e14"] = {}
    tot = trapz(wlit, zf)
    cum = np.concatenate([[0], np.cumsum(0.5 * (wlit[1:] + wlit[:-1]) * np.diff(zf))]) / tot
    frb["mean_dm_all_rays_literal"] = float(tot)
    frb["mean_dm_all_rays_xgpnative"] = float(tot * XGP)
    nbar = trapz(dN, zf)
    frb["nbar"] = float(nbar); frb["hit_fraction_unclustered"] = float(1 - np.exp(-nbar))
    frb["mean_dm_hit_xgpnative"] = float(tot * XGP / (1 - np.exp(-nbar)))
    frb["mean_dm_hit_literal"] = float(tot / (1 - np.exp(-nbar)))
    frb.update({"frac_z_lt_{}".format(c): float(np.interp(c, zf, cum)) for c in (0.25, 0.5, 0.75)})
    frb.update(z_median=float(np.interp(0.5, cum, zf)), z_p16=float(np.interp(0.16, cum, zf)), z_p84=float(np.interp(0.84, cum, zf)),
               z_mean=float(trapz(zf * wlit, zf) / tot))

    # --------------------------------------------------------------------- Konietzka TNG z_s test
    from scipy import optimize
    from halo_model_dm_pdf import PLANCK15_TNG
    zs_t = np.load(KONIETZKA / "redshifts_DMhalo.npy")
    a = np.load(KONIETZKA / "DMhalo_r200_mass13to14.npy")
    tng_mean = a.mean(1); tng_hit = (a > 0).mean(1)
    # delete-block jackknife error of the TNG mean (12 blocks of 10000 rays; rays are independent sightlines)
    blocks = a.reshape(a.shape[0], 12, -1).mean(2)
    tng_err = np.sqrt(11 / 12 * ((blocks.mean(1)[:, None] * 12 - blocks) / 11 - tng_mean[:, None]).__pow__(2).sum(1))
    zgrid, wl, dNt = frb_weight(PLANCK15_TNG, "lee22_literal", 2.5, nz=500)
    cumint = lambda w: np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(zgrid))])
    cumdm = cumint(wl); cumN = cumint(dNt)
    sel = (zs_t > 0.05) & (zs_t < 2.0 + 1e-9)
    zsf = zs_t[sel]; tm = tng_mean[sel]; te = tng_err[sel]
    E2t = lambda z: PLANCK15_TNG.E(z) ** 2

    def model_cum(C):
        return np.interp(zsf, zgrid, cumint(C(zgrid) * wl))

    def fit(shape, npar):
        def resid(p):
            return (model_cum(lambda z: shape(z, p)) - tm) / tm
        p0 = [0.3, 2.0][:npar]
        sol = optimize.least_squares(resid, p0)
        r = resid(sol.x)
        return sol.x, float(np.sqrt(np.mean(r ** 2))), float(np.max(np.abs(r)))
    fits = {}
    for key, shape, npar in (("constant", lambda z, p: p[0] + 0 * z, 1),
                             ("C0(1+z)^3", lambda z, p: p[0] * (1 + z) ** 3, 1),
                             ("C0(1+z)^k", lambda z, p: p[0] * (1 + z) ** p[1], 2),
                             ("C0(1+z)^3/E^2", lambda z, p: p[0] * (1 + z) ** 3 / E2t(z), 1),
                             ("C0(1+z)^2", lambda z, p: p[0] * (1 + z) ** 2, 1)):
        par, rms, mx = fit(shape, npar)
        fits[key] = dict(params=par.tolist(), rms_rel_resid=rms, max_rel_resid=mx)
    # fits restricted to z_s <= 1 (where the Lee22 mass-weighting is best covered)
    sel1 = zsf <= 1.0 + 1e-9
    def fit1(shape, npar):
        def resid(p):
            return ((model_cum(lambda z: shape(z, p)) - tm) / tm)[sel1]
        sol = optimize.least_squares(resid, [0.3, 2.0][:npar])
        return sol.x.tolist(), float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    fits["C0(1+z)^k_zs_le_1"] = dict(zip(("params", "rms_rel_resid"), fit1(lambda z, p: p[0] * (1 + z) ** p[1], 2)))
    test = []
    for i, zs in enumerate(zs_t):
        if zs > 2.5 + 1e-9:
            break
        lit = float(np.interp(zs, zgrid, cumdm))
        test.append(dict(z_s=float(zs), tng_mean_all_rays=float(tng_mean[i]), tng_err=float(tng_err[i]),
                         tng_hit_fraction=float(tng_hit[i]),
                         hm_hit_fraction_unclustered=float(1 - np.exp(-np.interp(zs, zgrid, cumN))),
                         hm_literal=lit, hm_xgpnative=lit * XGP, tng_over_literal=float(tng_mean[i] / lit)))
    eff = []
    for j in range(1, len(test)):
        dt = test[j]["tng_mean_all_rays"] - test[j - 1]["tng_mean_all_rays"]
        dl_ = test[j]["hm_literal"] - test[j - 1]["hm_literal"]
        eff.append(dict(z_mid=0.5 * (test[j]["z_s"] + test[j - 1]["z_s"]), C_eff=dt / dl_))
    res["konietzka_zs_test"] = dict(rows=test, C_eff_increments=eff, fits_zs_0p1_to_2=fits,
                                    note="TNG300 Konietzka within-R200 halo DM, 1e13-1e14 Msun, 120000 rays; halo model = "
                                         "Lee22 literal density (sphere R200c), Tinker08, Planck15 (TNG) cosmology; C_eff = "
                                         "d<DM>_TNG/dz_s / d<DM>_literal/dz_s over each 0.1 step; fits minimize relative "
                                         "residuals of the cumulative <DM>(z_s) at z_s = 0.1..2.0")
    kfit = fits["C0(1+z)^k"]["params"]
    # nonparametric TNG C_eff: smoothed increments up to z = 2, held constant beyond (extrapolation)
    ze_ = np.array([e["z_mid"] for e in eff if e["z_mid"] < 2.0]); ce_ = np.array([e["C_eff"] for e in eff if e["z_mid"] < 2.0])
    lfit = np.polyfit(np.log1p(ze_), np.log(ce_), 2)
    C_np = lambda z: np.exp(np.polyval(lfit, np.log1p(np.minimum(z, 2.0))))
    res["konietzka_zs_test"]["C_eff_quadratic_in_ln1pz"] = lfit.tolist()

    # ------------------------------------------------------------------------ candidate corrections
    E2 = lambda z: cosmo.E(z) ** 2

    def calibrate(shape_fn):
        """C0 such that int C0 shape w dz = XGP int w dz (FRB halo DM at z_s = 1 equals the xgpnative/TNG value)."""
        return XGP * tot / trapz(shape_fn(zf) * wlit, zf)

    cands = [("const 0.6016 (xgpnative)", lambda z: XGP + 0 * z)]
    for kexp in (1, 2, 3):
        c0 = calibrate(lambda z, kexp=kexp: (1 + z) ** kexp)
        cands.append(("C0 (1+z)^{} , C0={:.4f} (FRB-calibrated)".format(kexp, c0), lambda z, c0=c0, kexp=kexp: c0 * (1 + z) ** kexp))
    for c0, lab in ((0.18, "0.18"), (0.21, "0.21"), (H_TNG ** 3 * S.X_H ** 2, "h^3 X_H^2 = {:.4f}".format(H_TNG ** 3 * S.X_H ** 2)),
                    (H_TNG ** 3 * S.X_H, "h^3 X_H = {:.4f}".format(H_TNG ** 3 * S.X_H)),
                    (H_TNG ** 3, "h^3 = {:.4f}".format(H_TNG ** 3)), (H_TNG ** 2, "h^2 = {:.4f}".format(H_TNG ** 2))):
        cands.append(("{} (1+z)^3".format(lab), lambda z, c0=c0: c0 * (1 + z) ** 3))
    c0 = calibrate(lambda z: (1 + z) ** 3 / E2(z))
    cands.append(("C0 (1+z)^3/E^2(z), C0={:.4f} (FRB-calibrated)".format(c0), lambda z, c0=c0: c0 * (1 + z) ** 3 / E2(z)))
    cands.append(("TNG z_s fit: {:.4f} (1+z)^{:.3f}".format(*kfit), lambda z: kfit[0] * (1 + z) ** kfit[1]))
    cands.append(("TNG z_s fit: {:.4f} (1+z)^3".format(fits["C0(1+z)^3"]["params"][0]),
                  lambda z, c=fits["C0(1+z)^3"]["params"][0]: c * (1 + z) ** 3))
    cands.append(("TNG C_eff smoothed (z<=2), constant beyond", C_np))
    cands.append(("1 (literal)", lambda z: 1.0 + 0 * z))

    corr = res["density_corrections"] = []
    for lab, C in cands:
        row = dict(label=lab, C_at_z0=float(C(np.array([0.0]))[0]), C_at_z05=float(C(np.array([0.5]))[0]),
                   C_at_z1=float(C(np.array([1.0]))[0]), C_at_z2=float(C(np.array([2.0]))[0]),
                   frb_mean_dm_all_rays=float(trapz(C(zf) * wlit, zf)),
                   frb_ratio_to_xgpnative=float(trapz(C(zf) * wlit, zf) / (XGP * tot)),
                   tng_zs_rms_rel_resid=float(np.sqrt(np.mean(((model_cum(C) - tm) / tm) ** 2))),
                   tng_zs_ratio_at_0p1_0p5_1_2=[float(x) for x in (model_cum(C) / tm)[[0, 4, 9, 19]]])
        for run in ("validation", "physical"):
            d, sp = runs[run]
            s = sp["lee22_literal"]
            for l in ELL_REPORT + (5000,):
                Dc = at(s, Dl(s, cl_total(s, C)), l)
                Db = at(sp["b16"], Dl(sp["b16"], cl_total(sp["b16"])), l)
                Dlit = at(s, Dl(s, cl_total(s)), l)
                row["{}_ksz_D{}_uK2".format(run, l)] = Dc
                row["{}_ksz_D{}_over_b16".format(run, l)] = Dc / Db
                row["{}_ksz_D{}_over_literal".format(run, l)] = Dc / Dlit
                if run == "validation" and l in PAINTED_KSZ:
                    row["painted_equiv_ksz_D{}_uK2".format(l)] = PAINTED_KSZ[l]["lee22_literal"] * Dc / Dlit
            for l in (1000, 3000):
                bz = zbudget(s, l, C)[0]
                row["{}_ksz{}_zbudget".format(run, l)] = {k_: bz[k_] for k_ in ("z_median", "frac_z_lt_0.5", "frac_z_lt_1.0", "frac_z_lt_2.0")}
        corr.append(row)

    # pressure corrections C_P(z) = C0' (1+z)^k'
    pc = res["pressure_corrections"] = []
    pcands = [("1 (literal)", lambda z: 1.0 + 0 * z)]
    for kexp in (1, 2, 3):
        for c0, lab in ((1.0, "1"), (H_TNG ** 2, "h^2"), (H_TNG ** 3, "h^3"), (H_TNG ** 3 * S.X_H ** 2, "h^3 X_H^2")):
            pcands.append(("{} (1+z)^{}".format(lab, kexp), lambda z, c0=c0, kexp=kexp: c0 * (1 + z) ** kexp))
    c_b = [c for c in cands if c[0].startswith("TNG z_s fit: ") and "^3" in c[0] and "k" not in c[0]][0]
    pcands.append(("same as density: " + c_b[0], c_b[1]))
    pcands.append(("const 0.6016", lambda z: XGP + 0 * z))
    ells_p = (30, 100, 300, 1000, 2000, 3000, 5000, 8000)
    for lab, C in pcands:
        row = dict(label=lab, C_at_z0=float(C(np.array([0.0]))[0]), C_at_z05=float(C(np.array([0.5]))[0]),
                   C_at_z1=float(C(np.array([1.0]))[0]))
        for run in ("validation", "physical"):
            d, sp = runs[run]
            s, b = sp["lee22p"], sp["b12"]
            rat = []
            for l in ells_p:
                Dc = at(s, Dl(s, cl_total(s, C)), l)
                Db = at(b, Dl(b, cl_total(b)), l)
                rat.append(Dc / Db)
                row["{}_tsz_ratio_to_b12_l{}".format(run, l)] = Dc / Db
                row["{}_tsz_ratio_to_literal_l{}".format(run, l)] = Dc / at(s, Dl(s, cl_total(s)), l)
                if run == "validation":
                    row["painted_equiv_ratio_to_b12_l{}".format(l)] = tsz_painted[l]["ratio"] * Dc / at(s, Dl(s, cl_total(s)), l)
            rat = np.array(rat)
            row["{}_tsz_ratio_max_over_min_l100_8000".format(run)] = float(rat[1:].max() / rat[1:].min())
            dVdz = d["chi"] ** 2 / d["Hc"]
            lnM = np.log(d["M"])
            my = trapz(C(d["z"]) * dVdz * trapz(d["dn"] * d["y0_lee22p"], lnM, axis=1), d["z"])
            row["{}_mean_y_over_b12".format(run)] = float(my / res[run]["mean_y"]["b12"])
            for l in (300, 3000):
                bz = zbudget(s, l, C)[0]
                row["{}_tsz{}_zbudget".format(run, l)] = {k_: bz[k_] for k_ in ("z_median", "frac_z_lt_0.5", "frac_z_lt_1.0")}
        pc.append(row)

    # l-shape scans: exponent k of C(z) = (1+z)^k (amplitude-free) that makes the ratio to B16 / B12 flattest in l
    scan = res["shape_scan"] = {}
    d, sp = runs["validation"]
    for name, ref, ells_s in (("lee22_literal", "b16", (300, 1000, 2000, 3000, 5000)), ("lee22p", "b12", (100, 300, 1000, 2000, 3000, 5000, 8000))):
        rows = []
        for kexp in np.arange(0, 4.01, 0.25):
            C = lambda z, kexp=kexp: (1 + z) ** kexp
            r = np.array([at(sp[name], cl_total(sp[name], C), l) / at(sp[ref], cl_total(sp[ref]), l) for l in ells_s])
            rows.append(dict(k=float(kexp), max_over_min=float(r.max() / r.min()),
                             logslope=float(np.polyfit(np.log(ells_s), np.log(r), 1)[0])))
        scan[name] = dict(ells=list(ells_s), rows=rows)

    # per-halo diagnostics: enclosed gas fraction and electron temperature of the literal fits vs z
    diag = res["per_halo_diagnostics"] = {}
    for lmh in (13.2, 13.61, 14.0, 14.5):
        M = np.array([10 ** lmh / cosmo.h])
        rows = []
        for zz in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            amp, shape = S.density_shape("lee22_literal", cosmo, M, zz)
            r200 = S.r200c_mpc(cosmo, M, zz)[0] * S.MPC_CM
            Ne = amp[0] * r200 ** 3 * S.enclosed(shape, 1.0)[0]
            mgas = Ne * (2 / (1 + S.X_H)) * S.M_P_G            # fully ionized H+He, mu_e = 2/(1+X_H)
            fgas = mgas / (M[0] * S.MSUN_G * cosmo.omega_b / cosmo.omega_m)
            ampb, shb = S.density_shape("b16", cosmo, M, zz)
            Neb = ampb[0] * r200 ** 3 * S.enclosed(shb, 1.0)[0]
            P0, xc, beta = S.pressure_params("lee22p", cosmo, M, zz)
            E = (P0 * S.p200_cgs(cosmo, M, zz))[0] * r200 ** 3 * S.enclosed(S.RowShape("lee22", xc=xc, beta=beta), 1.0)[0]
            P0b, xcb, betab = S.pressure_params("b12", cosmo, M, zz)
            Eb = (P0b * S.p200_cgs(cosmo, M, zz))[0] * r200 ** 3 * S.enclosed(S.RowShape("lee22", xc=xcb, beta=betab), 1.0)[0]
            kT200 = 0.59 * S.M_P_G * S.G_CGS * M[0] * S.MSUN_G / (2 * r200)
            rows.append(dict(z=zz, fgas_literal_over_fb=fgas, Ne_literal_over_b16=Ne / Neb,
                             Eint_lee22p_over_b12=E / Eb,
                             kTe_lee22_over_kT200=(E / Ne) / kT200, kTe_b12b16_over_kT200=(Eb / Neb) / kT200))
        zz_ = np.array([r_["z"] for r_ in rows]); tt_ = np.array([r_["kTe_lee22_over_kT200"] for r_ in rows])
        diag["log10M_h={}".format(lmh)] = dict(rows=rows, kTe_ratio_powerlaw_index_in_1pz=float(np.polyfit(np.log1p(zz_), np.log(tt_), 1)[0]),
                                               fgas_powerlaw_index_in_1pz=float(np.polyfit(np.log1p(zz_), np.log([r_["fgas_literal_over_fb"] for r_ in rows]), 1)[0]),
                                               Eint_ratio_powerlaw_index_in_1pz=float(np.polyfit(np.log1p(zz_), np.log([r_["Eint_lee22p_over_b12"] for r_ in rows]), 1)[0]))

    (OUT / "budget_results.json").write_text(json.dumps(res, indent=1, default=float))
    # per-z curves for the figures
    curves = {}
    for run in ("validation", "physical"):
        d, sp = runs[run]
        for name in ("lee22_literal", "b16", "lee22p", "b12"):
            for l in ELL_REPORT + (8000,):
                if l == 8000 and name in ("lee22_literal", "b16"):
                    continue
                _, dz, dm = zbudget(sp[name], l)
                curves["{}_{}_{}_dz".format(run, name, l)] = dz
                curves["{}_{}_{}_dlnM".format(run, name, l)] = dm
        for lab, C in cands:
            if lab.startswith("TNG z_s fit: ") and "^3" in lab:
                for l in ELL_REPORT:
                    _, dz, _ = zbudget(sp["lee22_literal"], l, C)
                    curves["{}_lee22_tngcorr3_{}_dz".format(run, l)] = dz
        curves[run + "_z"] = d["z"]; curves[run + "_lnM"] = np.log(d["M"])
    curves["frb_z"] = zf; curves["frb_w_literal"] = wlit / tot
    curves["tng_zs"] = zs_t; curves["tng_mean"] = tng_mean; curves["tng_err"] = tng_err
    curves["hm_zgrid"] = zgrid; curves["hm_cum_literal"] = cumdm
    for lab, C in cands:
        if lab.startswith("TNG") or lab.startswith("const"):
            curves["hm_cum_" + lab.split(":")[0].replace(" ", "_") + ("_k" if "^{:.3f}".format(kfit[1]) in lab else "")
                   + ("_3" if lab.endswith("^3") else "")] = cumint(C(zgrid) * wl)
    curves["C_eff_z"] = np.array([e["z_mid"] for e in eff]); curves["C_eff"] = np.array([e["C_eff"] for e in eff])
    np.savez_compressed(OUT / "budget_curves.npz", **curves)
    print(json.dumps({k: res[k] for k in ("validation_vs_painted", "frb_zs1_1e13_1e14")}, indent=1, default=float))
    print(json.dumps(res["konietzka_zs_test"]["fits_zs_0p1_to_2"], indent=1))


if __name__ == "__main__":
    main()
