"""Numbers added to SECTION.md in the second correction round (2026-10-14).

Read-only.  Part A recomputes, from the cache through stackcore (same keys and
mask as compute_estimators.py), the extremes of the three identities that
compare the stored ComptonY with N_e k T.  results.json stores only their
0.1-99.9 percentiles.  Part B derives summary numbers from results.json:
jackknife/bootstrap and unstratified/stratified ratios per field and band, the
log-excess share left in smooth hot gas, the combined cold + substructure mass
share, and the kT_e split of the L_X-weighting factor by bin set.

Output: recheck_numbers.json in this folder.
Run: LD_LIBRARY_PATH=<env>/lib python -B recheck_numbers.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1]
sys.path.insert(0, str(OUT / "core"))
import stackcore as sc  # noqa: E402

R = json.loads((HERE / "results.json").read_text())
BANDS = R["bands_R200c"]
CAL = R["calibrated_bins"]
MAIN, TOP = CAL[:8], CAL[8:]
res = {}

# ---------------------------------------------------------------- A. identity extremes
GRID = "fit"
RES_OPTS = dict(min_count=20, softening_factor=4.0, hsml_factor=2.0, min_edge=0.04)
M_P, M_E = 1.67262192369e-27, 9.1093837015e-31
MU_XGP = (M_E + 2 * M_P / (1 + 0.76)) / 0.9
C_PET = 0.59 * M_P / (200.0 * MU_XGP)


def K(field, est=None):
    est = est or ("weighted" if field in ("T_M", "T_V", "T_X", "T_e", "K", "C_rho", "C_P") else "geometric")
    return sc.make_key(field, est, grid=GRID)


with np.errstate(divide="ignore", invalid="ignore"):
    Pg = np.asarray(sc.per_halo(K("P")))
    Eg = np.asarray(sc.per_halo(K("E")))
    PQV = np.asarray(sc.per_halo(K("P", "Q_V")))
    PQVy = np.asarray(sc.per_halo(K("P", "Q_V_y")))
    PnTX = np.asarray(sc.per_halo(K("P", "nT_X")))
    Te = np.asarray(sc.per_halo(K("T_e")))
    TX = np.asarray(sc.per_halo(K("T_X")))
res_ph = np.zeros_like(Pg, bool)
for b in sc.mass_bins("M200c_0.2"):
    res_ph[b["rows"]] = sc.resolution_mask(b["rows"], GRID, "per_halo", **RES_OPTS)
ident = {}
with np.errstate(divide="ignore", invalid="ignore"):
    for nm, rat in (("P_QV_over_P_QVy", PQV / PQVy), ("P_geom_over_cE_Te", Pg / (C_PET * Eg * Te)),
                    ("P_nTX_over_P_geom_times_Te_over_TX", PnTX / Pg * Te / TX)):
        a = rat[res_ph]
        a = a[np.isfinite(a)]
        d = np.abs(a - 1)
        ident[nm] = {"n": int(a.size), "min": float(a.min()), "max": float(a.max()), "max_absdev": float(d.max()),
                     "p0.1": float(np.percentile(a, 0.1)), "p99.9": float(np.percentile(a, 99.9)),
                     "frac_absdev_le_4e-4": float((d <= 4e-4).mean()),
                     "n_absdev_gt_4e-4": int((d > 4e-4).sum())}
res["identity_extremes_all_bins_resolved_halo_shells"] = ident

# geometric shell integral inside R200c vs each halo's own particle Y(<R200c): full range (results.json has percentiles)
SIGMA_T, MEC2, MPC = 6.6524587321e-29, 8.1871057769e-14, 3.0856775814913673e22
TAB = sc.halo_table()
yown = TAB["Y200_own_mpc2"] * MPC**2 * MEC2 / SIGMA_T / (TAB["p200_pa"] * (TAB["r200c"] * MPC) ** 3)
yrat = np.asarray(sc.integral(Pg, GRID, 1.0)) / np.asarray(yown)
res["Y200_own_vs_shell_integral_extremes"] = {"n": int(np.isfinite(yrat).sum()), "min": float(np.nanmin(yrat)),
                                              "max": float(np.nanmax(yrat)),
                                              "max_absdev": float(np.nanmax(np.abs(yrat - 1)))}

# ---------------------------------------------------------------- B. from results.json


def bootrow(field, b, band, key):
    for e in R["bootstrap"]["per_field"][field][b]:
        if e["band"] == band:
            return e.get(key)
    return None


boot = {}
for field in ("P", "E", "G", "T_e"):
    for key in ("jack_over_boot", "unstrat_over_strat"):
        for band in BANDS:
            vm = [(bootrow(field, b, band, key), b) for b in MAIN]
            vm = [x for x in vm if x[0] is not None]
            vt = {b: bootrow(field, b, band, key) for b in TOP}
            if not vm:
                continue
            vals = [x[0] for x in vm]
            boot.setdefault(field, {}).setdefault(key, {})[band] = {
                "main_min": min(vals), "main_max": max(vals), "main_argmax": max(vm)[1],
                "main_median": float(np.median(vals)), "main_n_gt_1": int(sum(v > 1 for v in vals)),
                "main_n": len(vals), "top": vt}
        # top-bin extremes over all bands
        for b in TOP:
            vals = [(bootrow(field, b, band, key), band) for band in BANDS]
            vals = [x for x in vals if x[0] is not None]
            boot[field][key][f"{b}_max_over_bands"] = max(vals)
            boot[field][key][f"{b}_min_over_bands"] = min(vals)
        # main bins, inside 2 R200c / beyond
        for lab, bands in (("inside_2R", BANDS[:6]), ("2-3", ["2-3"]), ("3-4", ["3-4"]), ("1.5-2", ["1.5-2"])):
            vals = [bootrow(field, b, band, key) for b in MAIN for band in bands]
            vals = [v for v in vals if v is not None]
            boot[field][key][f"main_{lab}_range"] = [min(vals), max(vals)]
res["bootstrap_summary"] = boot
# expected scatter of a 64-block delete-one jackknife SD (g - 1 = 63 dof) and of a 200-draw bootstrap SD
res["jackknife_sd_relative_noise_estimate"] = 1 / math.sqrt(2 * 63)
res["bootstrap_sd_relative_noise_estimate"] = 1 / math.sqrt(2 * 199)

mwd = R["diagnostics"]["mass_weighting_decomposition"]["per_bin_band"]
share = {}
for band in ("1.5-2", "2-3", "3-4"):
    rows = {}
    for b in MAIN:
        allp = mwd[b]["P_QM_over_geom_all"][band]
        sm = mwd[b]["P_QM_over_geom_smooth_hot_only"][band]
        allg = mwd[b]["G_QM_over_geom_all"][band]
        smg = mwd[b]["G_QM_over_geom_smooth_hot_only"][band]
        rows[b] = {"P_log_excess_share_smooth_hot": math.log(sm) / math.log(allp),
                   "G_log_excess_share_smooth_hot": math.log(smg) / math.log(allg),
                   "mass_share_not_smooth_hot": 1 - mwd[b]["mass_share_smooth_hot"][band],
                   "sum_cold_plus_sub_mass_share": mwd[b]["mass_share_cold"][band]
                   + mwd[b]["mass_share_substructure_bound"][band]}
    summ = {}
    for k in next(iter(rows.values())):
        v = [r[k] for r in rows.values()]
        summ[k] = [min(v), max(v)]
    share[band] = {"per_bin": rows, "main_range": summ}
res["mass_weighting_shares"] = share

txd = R["diagnostics"]["T_X_decomposition"]["per_bin_band"]
split = {}
for setname, bins in (("main_13.5-15.1", MAIN), ("calibrated_13.5-15.5", CAL)):
    groups = {"kTe_lt_0.7": [], "kTe_0.7_1.5": [], "kTe_ge_1.5": []}
    for b in bins:
        for band in BANDS:
            kt = txd[b]["kTe_keV"][band]
            lx = txd[b]["LX_weighting_TX_over_Te_xray"][band]
            if kt is None or lx is None:
                continue
            g = "kTe_lt_0.7" if kt < 0.7 else ("kTe_ge_1.5" if kt >= 1.5 else "kTe_0.7_1.5")
            groups[g].append(lx)
    split[setname] = {g: {"n": len(v), "min": min(v), "max": max(v)} for g, v in groups.items()}
res["LX_weighting_kTe_split"] = split

(HERE / "recheck_numbers.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["identity_extremes_all_bins_resolved_halo_shells"], indent=1))
print(json.dumps(res["LX_weighting_kTe_split"], indent=1))
for band, v in share.items():
    print(band, json.dumps(v["main_range"]))
