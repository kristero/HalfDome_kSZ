"""Quantify the stacking choices that can be evaluated from the stored particle-pass products (read-only).

Run:  LD_LIBRARY_PATH=<halfdome env>/lib <halfdome python> -B quantify_choices.py
Writes quantify_choices.json next to this file. Reads only the study products (no particle data, no remote access).

Everything is computed on the F grid (35 shells to 4 R200c, exact sub-grid of the stored 67-shell X200 grid) with the
analysis library, except where stated.  Pressure = P_e/P200 (geometric: sum y / V_shell); electrons = n_e mu_XGP/(rho_c f_b).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np

STUDY = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(STUDY))
from analysis import newdata, masks as MSK, stacking as S, variants as V  # noqa: E402

t0 = time.perf_counter()
d = newdata.load()
G = d.mod.G
P = d.mod.P

# F grid: the radii234 log edges inside 0.84 R200c plus 1, 1.15, 1.34, 1.5, 1.75, 2, 2.18, 2.5, 3, 3.30, 3.63, 4
_LOG = np.geomspace(.025, 3, 31)
_T = np.unique(np.r_[0, _LOG[_LOG < .9], 1.0, _LOG[(_LOG > 1.1) & (_LOG < 1.2)], 1.34, 1.5, 1.75, 2.0,
                     _LOG[(_LOG > 2.1) & (_LOG < 2.2)], 2.5, np.geomspace(3, 4, 4)])
X200 = G.X200
FE = X200[[int(np.argmin(np.abs(X200 - t))) for t in _T]]
assert len(FE) == 36 and np.allclose(FE, _T, rtol=1e-9, atol=1e-12)
GRID = ("x200", FE)
_, IDX = d.subgrid_index(GRID)
NF = len(FE) - 1
vol_dimless = 4 * np.pi / 3 * np.diff(FE**3)

halos = d.halos
n = d.n
mbin = d.mass_bin()
w = d.inclusion_weights()
logm = np.log10(d.mass("x200"))
edges_m = np.asarray(d.config["mass_edges_log10_Msun"])
nb = len(edges_m) - 1
labels = [f"{edges_m[b]:.1f}-{edges_m[b + 1]:.1f}" for b in range(nb)]
out: dict = {"inputs": {"study": str(STUDY), "grid": "F grid (35 shells to 4 R200c)", "f_edges": FE.tolist(),
                        "halo_rows": int(n)}}

# ------------------------------------------------------------------ 1. sample, weights, mass assigned to a stack
plan = d.plan
pop = np.asarray(plan["population_per_bin"], float)
samp = np.asarray(plan["sample_per_bin"], float)
rows_sample = []
for b in range(nb):
    r = np.flatnonzero(mbin == b)
    wb = w[r]
    wm = float(np.sum(wb * logm[r]) / np.sum(wb))
    rows_sample.append({"bin": labels[b], "N_sample": int(len(r)), "N_population": int(pop[b]),
                        "sampling_fraction": float(samp[b] / pop[b]),
                        "weight_min": float(wb.min()), "weight_max": float(wb.max()),
                        "weight_max_over_min": float(wb.max() / wb.min()),
                        "N_eff_inclusion_weights": float(wb.sum()**2 / np.sum(wb**2)),
                        "bin_centre_log10M": float((edges_m[b] + edges_m[b + 1]) / 2),
                        "population_weighted_mean_log10M": wm,
                        "equal_weight_mean_log10M": float(np.mean(logm[r])),
                        "median_log10M": float(np.median(logm[r]))})
out["sample_and_weights"] = rows_sample

# ------------------------------------------------------------------ 2. centre and extraction geometry
off = np.asarray(halos["centre_offset_R200c"], float)
r200, r500 = d.radius("x200"), d.radius("x500")
rext = np.asarray(halos["r_ext_mpc"], float)
out["centre_offset_R200c"] = {
    "definition": "|BoundSubhalo/CentreOfMass - InputHalos/HaloCentre| / R200c (selected_halos.npz)",
    "median": float(np.median(off)), "p90": float(np.percentile(off, 90)), "p99": float(np.percentile(off, 99)),
    "fraction_gt_0.07": float(np.mean(off > 0.07)), "fraction_gt_0.1": float(np.mean(off > 0.1))}
ratio500 = r500 / r200
out["extraction_sphere"] = {
    "R500c_over_R200c_median": float(np.median(ratio500)), "R500c_over_R200c_p5_p95": np.percentile(ratio500, [5, 95]).tolist(),
    "fraction_where_6R500c_exceeds_4R200c": float(np.mean(6 * r500 > 4 * r200)),
    "r_ext_over_R200c_median": float(np.median(rext / r200)), "r_ext_over_R200c_max": float(np.max(rext / r200)),
    "6R500c_over_R200c_median": float(np.median(6 * ratio500))}

# ------------------------------------------------------------------ 3. linear sums from the owner block (one load)
ob = d.array("owner200")                              # [n, 17, 3, 16, 67] float64 (~2 GB)
Q = {q: d.index("quantities", q) for q in d.names["quantities"]}
rb = lambda a: d.rebin(a, IDX)                         # noqa: E731
count = rb(ob[:, :, :, Q["count"], :].sum(axis=(1, 2)))
y_ph = rb(ob[:, :, :, Q["y"], :].sum(axis=1))         # [n, 3 phases, NF]
e_ph = rb(ob[:, :, :, Q["electrons"], :].sum(axis=1))
vol_sph = rb(ob[:, :, :, Q["volume"], :].sum(axis=(1, 2)))
m_ph = rb(ob[:, :, :, Q["mass"], :].sum(axis=1))
LC = newdata.LEGACY_COMPONENT_OWNERS
y_comp = np.stack([rb(ob[:, list(LC[c])][:, :, 1:, Q["y"], :].sum(axis=(1, 2))) for c in range(4)], axis=1)   # AGN removed
e_comp = np.stack([rb(ob[:, list(LC[c])][:, :, :, Q["electrons"], :].sum(axis=(1, 2))) for c in range(4)], axis=1)
own14 = rb(ob[:, 14, 1:, Q["y"], :].sum(axis=1))
d.release("owner200")
del ob
aux = d.array("aux200")                               # [n, 2 AGN, 3 aux, 67]
fully_ion = rb(aux[:, :, 2, :].sum(axis=1))
d.release("aux200")

geo_vol = 4 * np.pi / 3 * r200[:, None]**3 * np.diff(FE**3)[None, :]    # physical Mpc^3
p200 = d.p200()
rho_c = d.rho_c("x200")
fb = d.baryon_fraction
to_pa = P.electron_mass * P.c**2 / P.SIGMA_T / P.MPC_M


def press(ysum):
    return ysum * to_pa / geo_vol / p200[:, None]


def elec(esum):
    return esum * P.MU_XGP_MSUN / (geo_vol * rho_c[:, None] * fb)


P_rem = press(y_ph[:, 1:].sum(axis=1))
P_all = press(y_ph.sum(axis=1))
E_rem = elec(e_ph[:, 1:].sum(axis=1))
E_all = elec(e_ph.sum(axis=1))

# validation against Test F's stored per-halo profiles (same estimator, grid, selection)
pr = np.load(STUDY / "F_observables" / "results" / "profiles.npz")
with np.errstate(invalid="ignore", divide="ignore"):
    dv = {k: float(np.nanmax(np.abs(a / pr[k] - 1)[pr[k] > 0])) for k, a in
          (("P_main", P_rem), ("E_main", E_rem), ("P_agn_kept", P_all), ("E_agn_kept", E_all))}
out["validation_vs_F_profiles_npz_max_rel_diff"] = dv
base_mask = pr["base_mask"]

# ------------------------------------------------------------------ 4. resolution rules: first resolved shell per bin
variants_res = {
    "current (min_edge 0.04, 4 eps, >=20 particles, 2 h_upper-median, strict-or-diagnostic)": {},
    "min_count 32 (L2p8 / fullsnap)": {"min_count": 32},
    "min_count 100": {"min_count": 100},
    "hsml_factor 1 (kernel-smoothed core)": {"hsml_factor": 1.0},
    "hsml_factor 3": {"hsml_factor": 3.0},
    "softening_factor 2.8 (Plummer-equivalent 1 eps_spline)": {"softening_factor": 2.8},
    "min_edge 0.01": {"min_edge": 0.01},
}
res_table = {}
binding = {}
for name, opts in variants_res.items():
    rowsb = []
    for b in range(nb):
        r = np.flatnonzero(mbin == b)
        m = V.resolution(d, r, grid=GRID, **opts)
        base = m["base"]
        first = int(np.argmax(base)) if base.any() else -1
        rowsb.append({"bin": labels[b], "supported": bool(m["supported"]),
                      "first_base_inner_edge": float(FE[first]) if first >= 0 else None,
                      "base_shells": int(base.sum()),
                      "first_strict_inner_edge": float(FE[int(np.argmax(m["strict"]))]) if m["strict"].any() else None})
        if not opts:
            # which criterion binds at the innermost strict shell boundary: report, per criterion, the first shell
            # passed by every halo
            radius = d.radius("x200", r)
            geo = FE[None, :-1] >= np.maximum(0.04, 4 * d.softening_mpc / radius[:, None])
            cnt = count[r] >= 20
            hist = np.moveaxis(d.array("hs200")[r], -2, -1)
            hist = np.moveaxis(d.rebin(hist.astype(np.int64), IDX), -1, -2)
            from analysis import io as aio
            hs = aio.upper_median_from_hist(hist, d.array("hs_edges"))
            hsm = FE[None, :-1] >= 2 * hs

            def first_all(mask):
                ok = np.all(mask, axis=0)
                return float(FE[int(np.argmax(ok))]) if ok.any() else None
            binding[labels[b]] = {"geometric_first_edge": first_all(geo), "count20_first_edge": first_all(cnt),
                                  "hsml2_first_edge": first_all(hsm),
                                  "median_hsml_upper_x_at_0p1R": float(np.nanmedian(hs[:, int(np.argmin(np.abs(FE[:-1] - 0.1)))]))}
    res_table[name] = rowsb
d.release("hs200")
out["resolution_first_shell"] = res_table
out["resolution_binding_criterion_current"] = binding

# ------------------------------------------------------------------ 5. stack-level effects per bin (base shells of Test F)
regions = {"lt0.1R": (0, .1051), "0.1-1R": (.1051, 1.0), "1-2R": (1.0, 2.0), "2-4R": (2.0, 4.0)}
reg_mask = {k: (FE[:-1] >= a - 1e-9) & (FE[1:] <= b_ + 1e-9) for k, (a, b_) in regions.items()}
stack_rows = []
for b in range(nb):
    r = np.flatnonzero(mbin == b)
    wb = w[r]
    base = base_mask[b]
    row = {"bin": labels[b], "N": int(len(r))}
    mp, _, _ = S.weighted_stack(P_rem[r], wb)
    me, _, _ = S.weighted_stack(E_rem[r], wb)
    mp_all, _, _ = S.weighted_stack(P_all[r], wb)
    me_all, _, _ = S.weighted_stack(E_all[r], wb)
    mp_eq = P_rem[r].mean(axis=0)
    me_eq = E_rem[r].mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        med = np.median(P_rem[r], axis=0)
        gmean = S.log_mean_stack(P_rem[r], wb)
        mede = np.median(E_rem[r], axis=0)
    def rng_(a, msk):
        a = a[msk & np.isfinite(a)]
        return [float(a.min()), float(a.max())] if a.size else None
    row["equal_over_inclusion_weighted_P_base_range"] = rng_(mp_eq / mp - 1, base)
    row["equal_over_inclusion_weighted_E_base_range"] = rng_(me_eq / me - 1, base)
    row["agn_kept_over_removed_P_base_range"] = rng_(mp_all / mp - 1, base)
    row["agn_kept_over_removed_E_base_range"] = rng_(me_all / me - 1, base)
    row["agn_kept_over_removed_P_innermost_base_shell"] = float(mp_all[int(np.argmax(base))] / mp[int(np.argmax(base))] - 1) if base.any() else None
    for k, msk in reg_mask.items():
        row[f"equal_over_inclusion_weighted_P_{k}"] = rng_(mp_eq / mp - 1, base & msk)
        row[f"equal_over_inclusion_weighted_E_{k}"] = rng_(me_eq / me - 1, base & msk)
        row[f"median_over_mean_P_{k}"] = rng_(med / mp - 1, base & msk)
        row[f"median_over_mean_E_{k}"] = rng_(mede / me - 1, base & msk)
        row[f"geometric_mean_over_mean_P_{k}"] = rng_(gmean / mp - 1, base & msk)
    # SPH volume filling (Q_V denominators) on base shells by region
    vf = np.sum(wb[:, None] * vol_sph[r], axis=0) / np.sum(wb[:, None] * geo_vol[r], axis=0)
    for k, msk in reg_mask.items():
        row[f"sph_volume_filling_{k}"] = rng_(vf, msk & base)
    # electrons: stored ElectronNumberDensities vs fully ionised H+He (aux), all gas
    ef = np.sum(wb[:, None] * e_ph[r].sum(axis=1), axis=0) / np.sum(wb[:, None] * fully_ion[r], axis=0)
    for k, msk in reg_mask.items():
        row[f"ne_over_fully_ionised_{k}"] = rng_(ef, msk & base)
    # truncation: share of the stacked 4R volume integral inside 1, 2, 3 R (Y and N_e; stacks with NaN -> 0)
    for fld, mean in (("P", mp), ("E", me)):
        v = np.nan_to_num(mean) * vol_dimless
        tot = v.sum()
        for cut in (1.0, 2.0, 3.0):
            row[f"{fld}_int_lt{cut:g}R_over_lt4R"] = float(v[FE[1:] <= cut + 1e-9].sum() / tot) if tot > 0 else None
    # ownership shares of the stacked (AGN-removed) y and of electrons by region, inclusion-weighted
    comp_names = ("central_bound", "own_satellites", "other_haloes_bound", "unbound")
    for k, msk in reg_mask.items():
        ytot = np.sum(wb[:, None, None] * y_comp[r][:, :, msk])
        etot = np.sum(wb[:, None, None] * e_comp[r][:, :, msk])
        for c, cname in enumerate(comp_names):
            row[f"y_share_{cname}_{k}"] = float(np.sum(wb[:, None] * y_comp[r][:, c][:, msk]) / ytot) if ytot > 0 else None
            row[f"e_share_{cname}_{k}"] = float(np.sum(wb[:, None] * e_comp[r][:, c][:, msk]) / etot) if etot > 0 else None
        row[f"y_share_owner14_painted_other_{k}"] = float(np.sum(wb[:, None] * own14[r][:, msk]) / ytot) if ytot > 0 else None
    # per-halo resolution mask (L2p8 style) vs the common (strict) set: bias of the mean in partially resolved shells
    res_full = V.resolution(d, r, grid=GRID)
    resolved = res_full["resolved"]
    frac = resolved.mean(axis=0)
    part = (frac > 0.1) & (frac < 0.99)
    with np.errstate(invalid="ignore", divide="ignore"):
        wr = np.where(resolved, wb[:, None], 0.0)
        mean_res = (wr * P_rem[r]).sum(axis=0) / wr.sum(axis=0)
    row["per_halo_mask_bias_P_partial_shells"] = rng_(mean_res / mp - 1, part)
    row["partial_shells_x_range"] = [float(FE[:-1][part].min()), float(FE[1:][part].max())] if part.any() else None
    stack_rows.append(row)
out["stack_effects_per_bin"] = stack_rows
d.release("hs200")

# ------------------------------------------------------------------ 6. diagnostics of the pass
diag = d.array("diagnostics")
names = d.names["diagnostics"] if "diagnostics" in d.names else None
out["pass_diagnostics_totals"] = {str(k): int(v) for k, v in zip(names or range(diag.shape[1]), diag.sum(axis=0))}

# ------------------------------------------------------------------ 7. recent-AGN filter quantisation (stored as HalfFloat)
a_lim = float(plan["recent_agn_filter"]["a_limit"])
half = np.float16
grid16 = np.arange(np.float16(0.99).view(np.uint16), np.float16(1.0).view(np.uint16) + 1, dtype=np.uint16).view(np.float16).astype(float)
passing = grid16[grid16 >= a_lim]
below = grid16[grid16 < a_lim]
eff_a = float((passing.min() + below.max()) / 2)            # round-to-nearest boundary
h = float(np.atleast_1d(plan["cosmology"]["h"])[0])
hubble_time_myr = 9777.9222 / h                          # 1/H0 in Myr (1/(100 km/s/Mpc) = 9777.92 Myr)
out["recent_agn_filter"] = {"a_limit": a_lim, "tmin_k": plan["recent_agn_filter"]["tmin_k"],
                            "tmax_k": plan["recent_agn_filter"]["tmax_k"],
                            "delta_time_myr": plan["recent_agn_filter"].get("delta_time_myr"),
                            "float16_values_passing_near_1": passing.tolist(),
                            "float16_spacing_below_1": float(1 - grid16[-2]),
                            "effective_a_boundary_round_to_nearest": eff_a,
                            "nominal_window_myr_at_z0": (1 - a_lim) * hubble_time_myr,
                            "effective_window_myr_at_z0_estimate": (1 - eff_a) * hubble_time_myr,
                            "note": "LastAGNFeedbackScaleFactors carries the 'HalfFloat' lossy filter (probe_g4.json); "
                                    "z = 0 window converted with dt = da / H0 (a = 1). Estimate, not measured."}

# compact summary: min/max over the calibrated bins 13.5-15.5 (Test F support) of every per-bin metric
cal = [r for r in stack_rows if float(r["bin"].split("-")[0]) >= 13.5 - 1e-9]
summary = {}
for key in cal[0]:
    if key in ("bin", "N", "partial_shells_x_range"):
        continue
    vals = []
    for r in cal:
        v = r[key]
        if v is None:
            continue
        vals.extend(v if isinstance(v, list) else [v])
    summary[key] = [float(min(vals)), float(max(vals))] if vals else None
out["stack_effects_calibrated_bins_min_max"] = summary
out["seconds"] = time.perf_counter() - t0
(OUT / "quantify_choices.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps({"seconds": out["seconds"], "validation": dv}, indent=1))
