"""J stage 'prepare' (J1a): per-halo profiles, the stack-level resolution rule for the newly calibrated low-mass bins
(12.7-12.9 ... 13.3-13.5), per-bin moments, and the comparison of the measured low-mass stacks with the published
B12/B16 fallback.

Notation (x = r/R200c; s = fine X200 shell [x_s, x_s+1], 67 shells to 4 R200c; V_s = 4pi/3 (x_s+1^3 - x_s^3)):
  F_h,s      the profile of FLAMINGO halo h in shell s: P = P_e/P200 or E = n_e mu_XGP/(rho_c f_b) (dimensionless)
  w_h        inclusion weight of halo h (1 / inclusion probability of its stratum)
  F_b,s      = sum_h w_h F_h,s / sum_h w_h, the stacked (inclusion-weighted mean) profile of mass bin b
  N_b,s      = sum_h n_h,s, the number of gas particles of the stack in shell s (n_h,s: particles of halo h owned by the
               selection, recent-AGN gas excluded: count_a2ii for A2-ii P and E, count_hb for host-bound E,
               count_all for all gas)

Stack-level rule (replaces the per-halo strict mask in the low-mass bins).  Inside each of the radial segments
[0, x_h], [x_h, 1], [1, 2], [2, 4] (x_h = the hydrodynamic-resolution radius below; the breaks keep every merged shell
on one side of 1, 2 and x_h so that the sphere cuts D1/D2 and treatment T2 stay exact), consecutive fine shells are
merged greedily from the inside out into one stacked shell until sum N_b,s >= 400 (relative Poisson error
1/sqrt(400) = 5% of the stacked particle sum); a remainder below 400 at the outer end of a segment joins the previous
merged shell.  Merging = the volume-weighted mean of the fine shells, per halo:
      F_h,G = sum_{s in G} V_s F_h,s / sum_{s in G} V_s,   assigned to every fine shell of the group G (flat),
which conserves every halo's integral sum_s V_s F_h,s over each group exactly (T1).
Hydrodynamic-resolution radius x_h of a bin: the first fine edge x_s with x_s >= median_h (2 h_h,s / R200c), h_h,s the
upper-median SPH smoothing length of the halo's gas in shell s.  Shells inside x_h have an 'unresolved shape': their
integrated electron count is measured, their radial distribution is not.
Treatments of the low-mass bins (bins >= 13.5 are always the raw fine shells, as in Test H):
  T1 merged (baseline); T2 inside x_h the published shape (B16 for E, B12 for P, shell-averaged at each FLAMINGO halo's
  mass and inclusion-weighted over the bin) renormalised per halo to its measured sum_{s<k_h} V_s F_h,s; T1 outside;
  T3 raw fine shells.
Fallback comparison: published B16 / B12 exact shell averages (32-point Gauss-Legendre per shell),
  S_s(M) = 3 / (x_s+1^3 - x_s^3) int x^2 F_pub(M, z = 0, x) dx,
at every FLAMINGO halo's own M200c, inclusion-weighted over the bin like the stack.  Integrals:
  X(<R) = sum_{s: x_s+1 <= R} V_s F_s  (units rho_c f_b R200c^3 / mu_XGP for E, P200 R200c^3 for P);
  N_e(<R) mu_XGP / (f_b M200c) = X_E(<R) / (800 pi / 3)    (M200c = 200 rho_c 4pi/3 R200c^3).
Outputs: results/work/profiles_j.npz (per-halo rows and per-bin moments), results/stacks.npz, results/prepare.json,
tables/J1a_*.csv.
"""
from __future__ import annotations

import warnings

import j_common as JC
import numpy as np

FC, A = JC.FC, JC.A
X = JC.X200
VOL = JC.VOL
NB = JC.N_ENGINE
VARIANTS = {"P_a2ii": ("P_a2ii", "count_a2ii", 0), "E_a2ii": ("E_a2ii", "count_a2ii", 1),
            "E_hb": ("E_hb", "count_hb", 1), "E_all": ("E_all", "count_all", 1)}    # name: (array, counts, field)
T_NAMES = ("T1", "T2", "T3")
GL_T, GL_W = np.polynomial.legendre.leggauss(32)


# ============================================================================ helpers
def select_rows(h0):
    n = len(h0["weight"])
    if not JC.SMOKE:
        return np.arange(n)
    rng = np.random.default_rng(JC.SEED)
    rows = []
    for b in range(NB):
        r = np.flatnonzero(h0["mass_bin"] == b)
        rows.append(np.sort(rng.choice(r, size=min(25, len(r)), replace=False)))
    return np.concatenate(rows)


def shell_average_published(field, mass, z=0.0):
    """Exact shell averages [n, 67] of the published B12 (field 0) / B16 (field 1) profile at masses ``mass``."""
    mass = np.atleast_1d(np.asarray(mass, float))
    out = np.zeros((len(mass), JC.N_X))
    for s in range(JC.N_X):
        a, b = X[s], X[s + 1]
        xx = a + (b - a) * (GL_T + 1) / 2
        ww = (b - a) / 2 * GL_W
        F = A.published_profile(field, mass, z, xx)
        out[:, s] = (F * xx**2) @ ww * 3 / (b**3 - a**3)
    return out


def hydro_limit(hsml_rows):
    """(k_h, x_h, median 2h/R per shell): first fine edge x_s with x_s >= median(2 h/R200c) of shell s."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # shells empty in every halo: NaN (unresolved)
        med = np.nanmedian(2 * hsml_rows, axis=0)
    ok = np.flatnonzero(X[:-1] >= np.where(np.isfinite(med), med, np.inf))
    k = int(ok[0]) if len(ok) else JC.N_X
    return k, float(X[k]), med


def greedy_groups(counts, breaks, min_count=JC.MIN_STACK_COUNT):
    """Merged shells [(a, e)] (fine shell indices a..e-1), greedy inside each segment between consecutive breaks
    (fine-edge indices); a remainder below min_count joins the previous group of the same segment (if any)."""
    groups = []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        seg = []
        start, acc = lo, 0.0
        for s in range(lo, hi):
            acc += counts[s]
            if acc >= min_count:
                seg.append((start, s + 1))
                start, acc = s + 1, 0.0
        if start < hi:
            if seg:
                seg[-1] = (seg[-1][0], hi)
            else:
                seg.append((start, hi))
        groups += seg
    assert groups[0][0] == 0 and groups[-1][1] == JC.N_X
    assert all(g[1] == h[0] for g, h in zip(groups[:-1], groups[1:]))
    return groups


def merge(F, groups):
    """Per-halo volume-weighted mean inside each group, assigned to every fine shell of the group."""
    F = np.asarray(F, float)
    out = np.empty_like(F)
    for a, e in groups:
        v = VOL[a:e]
        out[..., a:e] = ((F[..., a:e] @ v) / v.sum())[..., None]
    return out


def integral(F, radius):
    ins = JC.inside(X, radius)
    return np.asarray(F)[..., ins] @ VOL[ins]


def second(a, b, w):
    w = np.asarray(w, float) / np.sum(w)
    return np.einsum("h,hs,ht->st", w, a, b)


# ============================================================================ main
def main():
    timer = JC.Timer()
    h0 = JC.load_npz(JC.J0)
    rows = select_rows(h0)
    n = len(rows)
    mb = h0["mass_bin"][rows]
    w_all = h0["weight"][rows]
    logm = h0["logm200c"][rows]
    mass = h0["m200c"][rows]
    hs = h0["hsml"][rows]
    raw = {v: np.asarray(h0[a][rows], float) for v, (a, _, _) in VARIANTS.items()}
    cnt = {c: np.asarray(h0[c][rows], float) for c in ("count_a2ii", "count_hb", "count_all")}
    pub = {0: shell_average_published(0, mass), 1: shell_average_published(1, mass)}
    out = {"x_edges": X, "rows": rows, "weight": w_all, "mass_bin": mb, "logm200c": logm, "m200c": mass,
           "r200c": h0["r200c"][rows], "strata": h0["strata"][rows], "pubP": pub[0], "pubE": pub[1]}
    info = {"mode": JC.MODE, "n_haloes": n, "j_bins": JC.J_BINS, "low_bins": JC.LOW_BINS,
            "definitions": {k: v["label"] for k, v in JC.DEFS.items()}, "treatments": JC.TREATMENTS,
            "min_stack_count": JC.MIN_STACK_COUNT, "k_cut": {f"{c:g}R": k for c, k in JC.K_CUT.items()}}
    # ---------------------------------------------------------------- lightcone haloes per engine bin
    sky = FC.make_sky()
    per_bin = np.bincount(sky.mass_bin, weights=sky.counts.sum(axis=0), minlength=16)
    per_bin_z1 = np.bincount(sky.mass_bin, weights=sky.counts[sky.z <= 1].sum(axis=0), minlength=16)
    lm_sky = np.log10(sky.mass)
    mean_logm_sky = np.array([np.sum(sky.counts.sum(axis=0)[sky.mass_bin == b] * lm_sky[sky.mass_bin == b])
                              / max(per_bin[b], 1e-300) for b in range(16)])
    info["lightcone"] = {"haloes_per_engine_bin": per_bin, "haloes_per_engine_bin_z_le_1": per_bin_z1,
                         "mean_logm200c_per_engine_bin": mean_logm_sky,
                         "min_logm200c_painted": float(lm_sky[sky.counts.sum(axis=0) > 0].min()),
                         "engine_bin_12.5-12.7_haloes": float(per_bin[0]),
                         "haloes_above_15.388": float(sky.counts.sum(axis=0)[lm_sky > FC.MAX_LOGMASS].sum())}
    out["lightcone_haloes_per_engine_bin"] = per_bin
    out["lightcone_mean_logm"] = mean_logm_sky
    # ---------------------------------------------------------------- resolution and merging per bin
    k_h = np.full(NB, JC.N_X)
    x_h = np.full(NB, np.nan)
    med2h = np.full((NB, JC.N_X), np.nan)
    soft = np.full(NB, np.nan)
    stacked_counts = {c: np.zeros((NB, JC.N_X)) for c in cnt}
    groups = {}
    for b in range(NB):
        rb = np.flatnonzero(mb == b)
        if not len(rb):
            continue
        k_h[b], x_h[b], med2h[b] = hydro_limit(hs[rb])
        soft[b] = float(np.median(4 * h0["eps_over_r200c"][rows][rb]))
        for c in cnt:
            stacked_counts[c][b] = cnt[c][rb].sum(axis=0)
        breaks = sorted({0, int(k_h[b]), JC.K_CUT[1.0], JC.K_CUT[2.0], JC.N_X})
        for v, (_, c, _) in VARIANTS.items():
            groups[(v, b)] = greedy_groups(stacked_counts[c][b], breaks)
    out.update(k_h=k_h, x_h=x_h, med2h=med2h, softening_limit=soft,
               **{f"stacked_{c}": v for c, v in stacked_counts.items()})
    # bin-1 subset painted mass range (>= 6.6e12) for the 12.7-12.9 sensitivity check
    sub_rows = np.flatnonzero((mb == 1) & (logm >= JC.PAINT_MIN_LOGM)) if 1 in JC.J_BINS else np.zeros(0, int)
    sub_groups = {}
    if len(sub_rows):
        breaks = sorted({0, int(k_h[1]), JC.K_CUT[1.0], JC.K_CUT[2.0], JC.N_X})
        for v, (_, c, _) in VARIANTS.items():
            sub_groups[v] = greedy_groups(cnt[c][sub_rows].sum(axis=0), breaks)
    # ---------------------------------------------------------------- per-halo treatments
    prof = {}
    for v, (_, c, field) in VARIANTS.items():
        T3 = raw[v].copy()
        T1 = raw[v].copy()
        T2 = raw[v].copy()
        for b in JC.LOW_BINS:
            rb = np.flatnonzero(mb == b)
            T1[rb] = merge(raw[v][rb], groups[(v, b)])
            k = int(k_h[b])
            w = w_all[rb] / w_all[rb].sum()
            shape = w @ pub[field][rb]                                  # bin-mean published shell averages
            X_in = raw[v][rb, :k] @ VOL[:k]                             # measured integral inside x_h, per halo
            T2[rb] = T1[rb]
            T2[rb, :k] = shape[None, :k] * (X_in / (shape[:k] @ VOL[:k]))[:, None]
        prof[v] = {"T1": T1, "T2": T2, "T3": T3}
        for t in T_NAMES:
            out[f"{v}_{t}"] = prof[v][t]
    # merged shells conserve every halo's integral over each group (check)
    worst = 0.0
    for (v, b), g in groups.items():
        if b not in JC.LOW_BINS:
            continue
        rb = np.flatnonzero(mb == b)
        for a, e in g:
            i1 = prof[v]["T1"][rb, a:e] @ VOL[a:e]
            i3 = raw[v][rb, a:e] @ VOL[a:e]
            nz = np.abs(i3) > 0
            if nz.any():
                worst = max(worst, float(np.max(np.abs(i1[nz] / i3[nz] - 1))))
    info["merge_integral_conservation_max_rel"] = worst
    # ---------------------------------------------------------------- per-bin moments (15 engine bins)
    def moments(rowsel, tag):
        res = {}
        for v in VARIANTS:
            for t in T_NAMES:
                m = np.zeros((NB, JC.N_X))
                sd = np.zeros((NB, JC.N_X))
                S = np.zeros((NB, JC.N_X, JC.N_X))
                for b in range(NB):
                    rb = rowsel(b)
                    if not len(rb):
                        continue
                    F = prof[v][t][rb]
                    m[b], sd[b], _ = FC.S.weighted_stack(F, w_all[rb])
                    S[b] = second(F, F, w_all[rb])
                res[f"mean_{v}_{t}{tag}"] = m
                res[f"sd_{v}_{t}{tag}"] = sd
                res[f"S_{v}_{t}{tag}"] = S
        for v in ("E_a2ii", "E_hb", "E_all"):
            for t in T_NAMES:
                S = np.zeros((NB, JC.N_X, JC.N_X))
                for b in range(NB):
                    rb = rowsel(b)
                    if len(rb):
                        S[b] = second(prof["P_a2ii"]["T1"][rb], prof[v][t][rb], w_all[rb])
                res[f"SPE_{v}_{t}{tag}"] = S
        return res

    out.update(moments(lambda b: np.flatnonzero(mb == b), ""))
    if len(sub_rows):
        # subset of bin 1 (painted mass range): its own T1 merge on the subset's stacked counts
        sub = {}
        for v in VARIANTS:
            sub[v] = merge(raw[v][sub_rows], sub_groups[v])
        ws = w_all[sub_rows]
        for v in VARIANTS:
            m, sd, _ = FC.S.weighted_stack(sub[v], ws)
            out[f"sub82_mean_{v}"] = m
            out[f"sub82_S_{v}"] = second(sub[v], sub[v], ws)
        for v in ("E_a2ii", "E_hb", "E_all"):
            out[f"sub82_SPE_{v}"] = second(sub["P_a2ii"], sub[v], ws)
        out["sub82_rows"] = sub_rows
        for v in VARIANTS:
            out[f"sub82_{v}"] = sub[v]
        info["subset_12.82"] = {"n": int(len(sub_rows)), "mean_logm": float(np.sum(ws * logm[sub_rows]) / ws.sum()),
                                "bin_mean_logm": float(np.sum(w_all[mb == 1] * logm[mb == 1]) / w_all[mb == 1].sum())}
    # ---------------------------------------------------------------- J1a comparisons (all bins; low bins reported)
    cmp = {}
    stacks = {"x_edges": X, "k_h": k_h, "x_h": x_h, "med2h": med2h, "softening_limit": soft,
              "E_cosmic": JC.E_COSMIC, "E_cosmic_fully_ionised": JC.E_COSMIC_THEORY}
    for c, v in stacked_counts.items():
        stacks[f"stacked_{c}"] = v
    for b in range(NB):
        rb = np.flatnonzero(mb == b)
        if not len(rb):
            continue
        w = w_all[rb] / w_all[rb].sum()
        lab = FC.bin_label(b)
        mpub = {f: w @ pub[f][rb] for f in (0, 1)}
        stacks.setdefault("pubP_mean", np.zeros((NB, JC.N_X)))[b] = mpub[0]
        stacks.setdefault("pubE_mean", np.zeros((NB, JC.N_X)))[b] = mpub[1]
        ml = mean_logm_sky[b] if per_bin[b] > 0 else np.nan
        if np.isfinite(ml):
            stacks.setdefault("pubE_lightcone_mass", np.zeros((NB, JC.N_X)))[b] = shell_average_published(1, [10**ml])[0]
            stacks.setdefault("pubP_lightcone_mass", np.zeros((NB, JC.N_X)))[b] = shell_average_published(0, [10**ml])[0]
        unit = 800 * np.pi / 3
        rec = {"n_haloes": int(len(rb)), "mean_logm200c_flamingo": float(w @ logm[rb]),
               "mean_logm200c_lightcone": float(ml) if np.isfinite(ml) else None,
               "x_h": float(x_h[b]), "softening_limit_4eps_over_R": float(soft[b])}
        for t in ("T1", "T3"):
            for v in VARIANTS:
                stacks.setdefault(f"{v}_{t}", np.zeros((NB, JC.N_X)))[b] = out[f"mean_{v}_{t}"][b]
        mE = out["mean_E_a2ii_T1"][b]
        mH = out["mean_E_hb_T1"][b]
        mA = out["mean_E_all_T1"][b]
        mP = out["mean_P_a2ii_T1"][b]
        for R in (1.0, 2.0, 4.0):
            rec[f"Ne_lt{R:g}R_over_fbM"] = {"A2ii": float(integral(mE, R) / unit), "B16": float(integral(mpub[1], R) / unit),
                                           "host_bound": float(integral(mH, R) / unit), "all_gas": float(integral(mA, R) / unit)}
            rec[f"Ne_lt{R:g}R_ratio_A2ii_over_B16"] = float(integral(mE, R) / integral(mpub[1], R))
        for R in (1.0, 2.0, 4.0):
            rec[f"Y_lt{R:g}R_ratio_A2ii_over_B12"] = float(integral(mP, R) / integral(mpub[0], R))
        rec["Y_lt1R_over_P200R3"] = {"A2ii": float(integral(mP, 1.0)), "B12": float(integral(mpub[0], 1.0))}
        rec["share_of_Ne_lt4R_inside_R200c"] = {"A2ii": float(integral(mE, 1.0) / integral(mE, 4.0)),
                                                "host_bound": float(integral(mH, 1.0) / integral(mH, 4.0)),
                                                "B16": float(integral(mpub[1], 1.0) / integral(mpub[1], 4.0))}
        bands = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (3.6342, 4)):
            sel = (X[:-1] >= lo - 1e-9) & (X[1:] <= hi + 1e-9)
            xa = mE[sel] @ VOL[sel]
            bands[f"{lo:g}-{hi:g}R"] = {"hb_over_A2ii": float((mH[sel] @ VOL[sel]) / xa),
                                       "A2ii_over_all": float(xa / (mA[sel] @ VOL[sel])),
                                       "A2ii_mean_over_cosmic": float(xa / VOL[sel].sum() / JC.E_COSMIC),
                                       "hb_mean_over_cosmic": float((mH[sel] @ VOL[sel]) / VOL[sel].sum() / JC.E_COSMIC)}
        rec["bands"] = bands
        rec["host_bound_fraction_of_hb_lt4R_in"] = {
            "2-4R": float(1 - integral(mH, 2.0) / integral(mH, 4.0)),
            "3-4R": float(1 - integral(mH, 3.0) / integral(mH, 4.0))}
        # beyond 4R (not measured): power law fitted to the raw (T3, unmerged) host-bound stack over the 3-4R fine
        # shells, integrated from 4R to infinity if its slope is steeper than -3, else to 12R (ESTIMATE)
        mH3 = out["mean_E_hb_T3"][b]
        sel = (X[:-1] >= 3.0 - 1e-9) & (mH3 > 0)
        lx = np.log(JC.mids()[sel])
        slope, icpt = np.polyfit(lx, np.log(mH3[sel]), 1)
        upper = np.inf if slope < -3 else 12.0
        if np.isfinite(upper):
            beyond = 4 * np.pi * np.exp(icpt) * (upper**(3 + slope) - 4**(3 + slope)) / (3 + slope)
        else:
            beyond = -4 * np.pi * np.exp(icpt) * 4**(3 + slope) / (3 + slope)
        rec["host_bound_beyond_4R_ESTIMATE"] = {"slope_3_4R": float(slope), "upper_x": float(upper),
                                                "fraction_of_hb_lt4R": float(beyond / integral(mH, 4.0))}
        inR = JC.inside(X, 1.0)
        rec["median_particles_per_fine_shell_inside_R200c"] = {
            "A2ii_median_over_haloes_and_shells": float(np.median(cnt["count_a2ii"][rb][:, inR])),
            "A2ii_per_shell_median_range": [float(np.min(np.median(cnt["count_a2ii"][rb][:, inR], axis=0))),
                                            float(np.max(np.median(cnt["count_a2ii"][rb][:, inR], axis=0)))]}
        rec["median_2h_over_R200c_at_2R"] = float(np.interp(2.0, JC.mids(), med2h[b]))
        if b in JC.LOW_BINS or b in JC.J_BINS:
            for v in VARIANTS:
                g = groups[(v, b)]
                c = stacked_counts[VARIANTS[v][1]][b]
                gc = np.array([c[a:e].sum() for a, e in g])
                multi = [(float(X[a]), float(X[e]), int(e - a)) for a, e in g if e - a > 1]
                rec[f"merge_{v}"] = {"n_stacked_shells": len(g), "merged_groups": multi,
                                     "min_stacked_count": float(gc.min()),
                                     "outer_edge_of_last_merged_group_inside_R200c":
                                         float(max([m[1] for m in multi if m[1] <= 1.0 + 1e-9], default=0.0)),
                                     "fine_shells_below_400_unmerged_T3": int(np.sum(c < JC.MIN_STACK_COUNT))}
                stacks[f"groups_{v}_{b}"] = np.array(g)
        # profile ratios measured / published (T1 and T3)
        for t in ("T1", "T3"):
            with np.errstate(all="ignore"):
                stacks.setdefault(f"ratio_E_a2ii_{t}_over_B16", np.full((NB, JC.N_X), np.nan))[b] = \
                    out[f"mean_E_a2ii_{t}"][b] / mpub[1]
                stacks.setdefault(f"ratio_P_a2ii_{t}_over_B12", np.full((NB, JC.N_X), np.nan))[b] = \
                    out[f"mean_P_a2ii_{t}"][b] / mpub[0]
        cmp[lab] = rec
    info["J1a_by_bin"] = cmp
    info["seconds"] = timer.mark("total")
    np.savez(JC.WORK / "profiles_j.npz", **out)
    np.savez_compressed(JC.RESULTS / "stacks.npz", **stacks)
    JC.write_json(JC.RESULTS / "prepare.json", info)
    rows_csv = []
    for lab, rec in cmp.items():
        row = {"bin": lab, "n": rec["n_haloes"], "mean_logm_flamingo": rec["mean_logm200c_flamingo"],
               "mean_logm_lightcone": rec["mean_logm200c_lightcone"], "x_h": rec["x_h"]}
        for R in (1, 2, 4):
            for k, val in rec[f"Ne_lt{R}R_over_fbM"].items():
                row[f"Ne<{R}R/fbM {k}"] = val
            row[f"Ne<{R}R A2ii/B16"] = rec[f"Ne_lt{R}R_ratio_A2ii_over_B16"]
            row[f"Y<{R}R A2ii/B12"] = rec[f"Y_lt{R}R_ratio_A2ii_over_B12"]
        rows_csv.append(row)
    JC.write_csv(JC.TABLES / "J1a_integrals.csv", rows_csv)
    JC.log("prepare done", timer.mark("total"))
    return info


if __name__ == "__main__":
    main()
