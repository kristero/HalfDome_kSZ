"""B0: per-halo profile arrays and resolution masks for every Test-B variant (one pass over the big blocks).

Reads the particle-pass products once (analysis.newdata / analysis.variants, read-only) and writes compact
arrays [halo, shell] for the fit grid (35 shells), the Lee22 grid (27 log shells 0.04-1.34 R200c) and the old
radii234 grid (36 shells), plus resolution masks per mass bin / fit unit / Lee22 bin.  Later stages read only
these arrays (the 2 GB owner block is loaded once, here).  Smoke mode: a stratified subset of SMOKE_PER_BIN
haloes in each of SMOKE_BINS (original row indices kept in 'orig_row').

Outputs (results/): halo_table.npz, profiles_{fit,lee22,old}.npz, masks_{fit,lee22,old}.npz, prepare.json.
"""
from __future__ import annotations

import bcommon as B  # noqa: I001  (sets single-threaded BLAS first)

import resource

import numpy as np

from analysis import newdata
from analysis import variants as V

AGN = "removed"


def fit_grid_edges(G) -> tuple[np.ndarray, np.ndarray]:
    """Fit-grid edges (exact values of grids.X200) and their indices."""
    targets = np.r_[G.OLD_EDGES[G.OLD_EDGES <= B.FIT_INNER_MAX], B.FIT_EXTRA_EDGES]
    idx = np.array([int(np.argmin(np.abs(G.X200 - e))) for e in targets])
    if not np.allclose(G.X200[idx], targets, rtol=1e-5, atol=0):
        raise ValueError("fit-grid edge not found in X200")
    if np.any(np.diff(idx) <= 0):
        raise ValueError("fit grid not increasing")
    return G.X200[idx], idx


def jackknife_block(centre_cmpc: np.ndarray) -> np.ndarray:
    k = B.JACKKNIFE_BLOCKS_PER_AXIS
    cell = np.floor(np.mod(centre_cmpc, B.BOX_MPC) / (B.BOX_MPC / k)).astype(int)
    cell = np.clip(cell, 0, k - 1)
    return cell[:, 0] * k * k + cell[:, 1] * k + cell[:, 2]


def select_rows(d) -> np.ndarray:
    """All rows (full) or the stratified smoke subset (CLEAN and CONT in every smoke bin)."""
    if not B.SMOKE:
        return np.arange(d.n)
    rng = np.random.default_rng(B.SEEDS["smoke"])
    mb = np.asarray(d.halos["mass_bin"])
    cls = np.asarray(d.halos["census_class_main"])
    out = []
    for b in B.SMOKE_BINS:
        cand = np.flatnonzero(mb == b)
        clean, cont = cand[cls[cand] == 0], cand[cls[cand] > 0]
        n_cont = min(len(cont), (2 * B.SMOKE_PER_BIN) // 5)
        out.append(np.sort(np.r_[rng.choice(clean, B.SMOKE_PER_BIN - n_cont, replace=False),
                                 rng.choice(cont, n_cont, replace=False)]))
    return np.concatenate(out)


def main():
    timer = B.Timer()
    B.ensure_dirs(B.RESULTS)
    d = newdata.load(B.ROOT)
    G = d.mod.G
    P = d.mod.P
    h = d.halos
    rows = select_rows(d)
    summary = {"mode": B.MODE, "haloes_total": d.n, "haloes_used": int(len(rows))}
    if abs(d.hubble() - B.H_FLAMINGO) > 1e-9:
        raise ValueError("unexpected h")
    box = float(d.plan["box_size_cmpc"])
    if abs(box - B.BOX_MPC) > 1e-3:
        raise ValueError(f"box {box}")
    # ------------------------------------------------------------------ halo table (rows of this run)
    take = lambda a: np.asarray(a)[rows]      # noqa: E731
    logm = np.log10(take(h["m200c_msun"]).astype(float))
    m200, r200 = take(h["m200c_msun"]).astype(float), take(h["r_200_crit_mpc"]).astype(float)
    kT200 = 0.59 * P.proton_mass * P.G * m200 * P.MSUN_KG / (2 * r200 * P.MPC_M)
    table = {"orig_row": rows, "mass_bin": take(h["mass_bin"]).astype(int),
             "class_main": take(h["census_class_main"]).astype(int), "weight": take(h["weight"]).astype(float),
             "inclusion_probability": take(h["inclusion_probability"]).astype(float), "logm": logm, "m200c": m200,
             "r200c": r200, "m500c": take(h["m500c_msun"]).astype(float), "r500c": take(h["r_500_crit_mpc"]).astype(float),
             "centre_cmpc": take(h["centre_cmpc"]).astype(float),
             "block": jackknife_block(take(h["centre_cmpc"]).astype(float)), "role": take(h["role"]).astype(int),
             "T200_K": kT200 / P.Boltzmann, "p200_pa": d.p200(rows),
             "concentration_200c": take(h["concentration_200c"]).astype(float),
             "mass_edges": np.asarray(h["mass_edges"], float)}
    for mu in (0.03, 0.1, 0.3):
        table[f"class_mu{mu:g}"] = d.census_class(mu, "all")[rows].astype(int)
    table["class_neighbours_mu0.1"] = d.census_class(0.1, "neighbours")[rows].astype(int)
    if not np.array_equal(table["class_mu0.1"], table["class_main"]):
        raise ValueError("census_class(0.1) != census_class_main")
    np.savez_compressed(B.RESULTS / "halo_table.npz", **table)
    summary["blocks_occupied"] = int(len(np.unique(table["block"])))
    summary["haloes_per_block"] = [int(x) for x in np.percentile(np.bincount(table["block"], minlength=64), [0, 50, 100])]
    B.log(f"halo table: {len(rows)} haloes ({timer():.0f} s)")

    # ------------------------------------------------------------------ grids
    fit_edges, fit_idx = fit_grid_edges(G)
    grids = {"fit": ("x200", fit_edges), "lee22": "lee22", "old": "old"}
    stored = d.grid("x200")
    summary["fit_grid"] = {"edges": fit_edges, "x200_index": fit_idx, "shells": len(fit_edges) - 1,
                           "max_abs_stored_minus_local": float(np.max(np.abs(stored[fit_idx] - fit_edges)))}

    def prof(field, estimator="geometric", agn=AGN, sel="all", grid="fit", **kw):
        return V.profile(d, field, estimator, agn, sel, grids[grid], rows=rows, **kw)

    out = {g: {} for g in grids}
    for g in grids:
        out[g]["P_geo"] = prof("pressure", grid=g).values
        out[g]["P_geo_agnkept"] = prof("pressure", agn="kept", grid=g).values
        out[g]["E_geo"] = prof("electrons", grid=g).values
        out[g]["E_geo_agnkept"] = prof("electrons", agn="kept", grid=g).values
        B.log(f"{g}: main fields ({timer():.0f} s)")
    fg = out["fit"]
    fg["G_geo"] = prof("gas").values
    for est in ("Q_V", "Q_V_y", "Q_M", "Q_X", "nT_X"):
        fg[f"P_{est.replace('_', '')}"] = prof("pressure", est).values
    fg["E_QV"] = prof("electrons", "Q_V").values
    fg["G_QV"] = prof("gas", "Q_V").values
    fg["G_QM"] = prof("gas", "Q_M").values
    for kind, key in (("T_M", "T_M"), ("T_V", "T_V"), ("T_X", "T_X"), ("clumping_density", "C_rho"),
                      ("clumping_pressure", "C_P")):
        fg[key] = V.diagnostic(d, kind, AGN, "all", grids["fit"], rows=rows)
    base, idx = d.subgrid_index(grids["fit"])
    vol_geo = V._volume(d, "x200", d.grid("x200")[idx], rows)
    sel_all = V.selection("all")
    for agn, key in (("removed", "Vsph_ratio"), ("kept", "Vsph_ratio_agnkept")):
        fg[key] = d.rebin(V.owner_sum(d, "x200", "volume", sel_all, agn, rows), idx) / vol_geo
    lx_pe = d.rebin(V.owner_sum(d, "x200", "lx_pe", sel_all, AGN, rows, only_xray=True), idx)
    lx_t = d.rebin(V.owner_sum(d, "x200", "lx_t", sel_all, AGN, rows, only_xray=True), idx)
    ne = d.rebin(V.owner_sum(d, "x200", "electrons", sel_all, AGN, rows), idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_x = np.where(lx_t > 0, lx_pe / (P.Boltzmann * lx_t), np.nan)            # m^-3
        n_geo = ne / (vol_geo * P.MPC_M ** 3)
        fg["nX_ratio"] = np.where(n_geo > 0, n_x / n_geo, np.nan)
    B.log(f"fit: weightings and diagnostics ({timer():.0f} s)")
    for flav, key in (("remove_all", "all"), ("keep_own_satellites", "keep")):
        s = V.selection("remove_overlappers", mu_min=0.1, reach="any", flavour=flav)
        fg[f"P_bound_{key}"] = prof("pressure", sel=s).values
        fg[f"E_bound_{key}"] = prof("electrons", sel=s).values
    for var, key in (("all", "all"), ("neighbours_only", "keep")):
        exc = {"variant": var, "kappa": 1.0, "members": "all", "min_retained": 0.5}
        p = prof("pressure", excision=exc)
        fg[f"P_exc_{key}"] = p.values
        fg[f"retained_{key}"] = p.retained_fraction
        fg[f"E_exc_{key}"] = prof("electrons", excision=exc).values
    d.release("excision")
    s = V.selection("remove_bound_to_other_painted")
    fg["P_A2i"] = prof("pressure", sel=s).values
    fg["E_A2i"] = prof("electrons", sel=s).values
    s = V.selection("partition_owned")
    fg["P_A2ii"] = prof("pressure", sel=s).values
    fg["E_A2ii"] = prof("electrons", sel=s).values
    summary["a2i_ambiguity"] = V.a2i_ambiguity(d)
    B.log(f"fit: overlap samples ({timer():.0f} s)")
    for g, arrays in out.items():
        edges = np.asarray(d.edges(grids[g]), float)
        local = fit_edges if g == "fit" else (G.LEE22_EDGES if g == "lee22" else G.OLD_EDGES)
        np.savez_compressed(B.RESULTS / f"profiles_{g}.npz", edges=local, stored_edges=edges,
                            **{k: np.asarray(v, float) for k, v in arrays.items()})
    summary["profile_keys"] = {g: sorted(a) for g, a in out.items()}
    B.log(f"profiles written ({timer():.0f} s), peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")

    # ------------------------------------------------------------------ resolution masks (positions = table rows)
    mass_bin = table["mass_bin"]
    units = {f"b{b}": np.flatnonzero(mass_bin == b) for b in range(15)}
    units["P"] = np.flatnonzero(np.isin(mass_bin, B.POOL_BINS))
    for k, (lo_h, hi_h) in enumerate(((13.2, 13.4), (13.8, 14.0))):
        lo, hi = lo_h - np.log10(B.H_FLAMINGO), hi_h - np.log10(B.H_FLAMINGO)
        units[f"L{k}"] = np.flatnonzero((logm >= lo) & (logm < hi))
        summary[f"lee22_bin_{k}"] = {"h_units": [lo_h, hi_h], "physical": [lo, hi], "N": int(len(units[f"L{k}"])),
                                     "strata": {B.bin_label(int(b)): int(np.sum(mass_bin[units[f'L{k}']] == b))
                                                for b in np.unique(mass_bin[units[f"L{k}"]])}}
    for b in range(15):     # radii234 rows only (quick-look reproduction)
        units[f"r234_b{b}"] = np.flatnonzero((mass_bin == b) & (table["role"] == 0))
    units = {k: v for k, v in units.items() if len(v) > 0}
    for g in grids:
        arrays = {}
        for name, pos in units.items():
            if name.startswith("r234") and g != "old":
                continue
            m = V.resolution(d, rows[pos], grids[g])
            for key in ("base", "strict", "diagnostic", "resolved_fraction"):
                arrays[f"{name}__{key}"] = np.asarray(m[key])
            arrays[f"{name}__supported"] = np.array(bool(m["supported"]))
            arrays[f"{name}__rows"] = pos
        np.savez_compressed(B.RESULTS / f"masks_{g}.npz", **arrays)
        B.log(f"masks {g} ({timer():.0f} s)")
    cont = table["class_main"] > 0
    summary["excision_nan_halo_shells_cont"] = {k: int(np.sum(~np.isfinite(fg[f"P_exc_{k}"][cont]))) for k in ("all", "keep")}
    summary["excision_nan_halo_shells_clean"] = {k: int(np.sum(~np.isfinite(fg[f"P_exc_{k}"][~cont]))) for k in ("all", "keep")}
    summary["QX_nan_halo_shells"] = int(np.sum(~np.isfinite(fg["P_QX"])))
    summary["units_with_masks"] = sorted(units)
    summary["wall_seconds"] = timer()
    summary["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    B.dump(B.RESULTS / "prepare.json", summary)
    B.log(f"done in {timer():.0f} s")


if __name__ == "__main__":
    main()
