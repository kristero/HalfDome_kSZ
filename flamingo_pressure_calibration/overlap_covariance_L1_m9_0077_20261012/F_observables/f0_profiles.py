"""F0: per-halo profiles on the F grid, resolution masks, truth moments and spatial CV folds.

Per halo (4,588 centrals) and F shell (35, to 4 R200c), from the particle-pass owner blocks through
analysis.variants.profile (linear sums rebinned before any ratio):
  pressure  = Pe/P200, geometric estimator (sum y / V_shell), electrons = n_e mu_XGP/(rho_c f_b);
  selections: main 'all' gas with recently AGN-heated gas removed (literature full signal, satellites
  kept); controls 'agn_kept' (all phases), 'a2i' (remove_bound_to_other_painted), 'a2ii' (partition_owned).
Per 0.2-dex bin: inclusion-weighted mean, reliability-weighted SD, second moments <F_s F_t> (PP, EE, PE),
resolution masks (analysis.masks via variants.resolution on the F grid).
Folds: 4^3 = 64 cubic blocks (250 Mpc) of the 1 Gpc box by halo centre; blocks assigned to 8 folds of 8
blocks by a seeded permutation (fold k holds out 1/8 of the volume).
Output: products/profiles.npz, products/profiles_summary.json.
"""
from __future__ import annotations

import f_common as FC
import numpy as np

SELECTIONS = {"main": ("all", "removed"), "agn_kept": ("all", "kept"),
              "a2i": ("remove_bound_to_other_painted", "removed"), "a2ii": ("partition_owned", "removed")}
FOLDS = FC.FOLDS
BLOCKS_PER_SIDE = 4


def weighted_moments(values, weights):
    """Inclusion-weighted mean, reliability-weighted SD (analysis.stacking.weighted_stack), n_eff."""
    mean, sd, neff = FC.S.weighted_stack(values, weights)
    return mean, sd, neff


def second_moment(a, b, weights):
    w = np.asarray(weights, float) / np.sum(weights)
    return np.einsum("h,hs,ht->st", w, a, b)


def main():
    timer = FC.Timer()
    d = FC.data()
    fe = FC.f_edges(d.mod.G.X200)                    # bitwise entries of the local grids.py X200
    grid = ("x200", fe)
    assert np.allclose(fe, FC.F_EDGES, rtol=1e-15, atol=1e-16)
    all_bins = d.mass_bin()
    if FC.SMOKE:                                      # smoke: <= 25 haloes in each of the two calibrated bins
        rng = np.random.default_rng(FC.SEED)
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(all_bins == b), FC.SMOKE_HALOES_PER_BIN, replace=False)
                                       for b in FC.CAL_BINS]))
    else:
        rows = np.arange(d.n)
    n = len(rows)
    bins = all_bins[rows]
    weights = d.inclusion_weights()[rows]
    logmass = np.log10(d.mass("x200")[rows])
    out = {"f_edges_data": fe, "f_edges": FC.F_EDGES, "mass_bin": bins, "weight": weights, "logmass": logmass,
           "m200c_msun": d.mass("x200")[rows], "r200c_mpc": d.radius("x200")[rows],
           "centre_mpc": np.asarray(d.halos["centre_mpc"], float)[rows], "halo_rows": rows}
    summary = {"mode": FC.MODE, "halo_rows": n, "selections": {}, "bins": {}}
    for name, (sel, agn) in SELECTIONS.items():
        for field, key in (("pressure", "P"), ("electrons", "E")):
            prof = FC.V.profile(d, field, "geometric", agn, sel, grid=grid, rows=None if not FC.SMOKE else rows)
            values = prof.values
            out[f"{key}_{name}"] = values
            summary["selections"][f"{key}_{name}"] = {
                "selection": prof.selection, "agn": agn, "label": prof.info.get("selection_label"),
                "finite_fraction": float(np.isfinite(values).mean()),
                "negative_entries": int(np.sum(values < 0))}
            FC.log(f"profile {key}_{name}: {timer.mark(key + name):.1f} s")
    # resolution masks per bin (all 15 bins)
    nb = len(FC.MASS_EDGES) - 1
    base = np.zeros((nb, FC.N_F), bool)
    strict = np.zeros((nb, FC.N_F), bool)
    diagnostic = np.zeros((nb, FC.N_F), bool)
    supported = np.zeros(nb, bool)
    for b in range(nb):
        rows_b = np.flatnonzero(bins == b)
        if not len(rows_b):
            continue
        m = FC.V.resolution(d, rows[rows_b], grid=grid)
        base[b], strict[b], diagnostic[b], supported[b] = m["base"], m["strict"], m["diagnostic"], m["supported"]
        first = int(np.argmax(m["base"])) if m["base"].any() else -1
        summary["bins"][FC.bin_label(b)] = {
            "N": int(len(rows_b)), "sum_weight": float(weights[rows_b].sum()), "supported": bool(m["supported"]),
            "first_resolved_inner_edge": float(fe[first]) if first >= 0 else None,
            "resolved_shells_to_4R": int(m["base"].sum())}
    out.update(base_mask=base, strict_mask=strict, diagnostic_mask=diagnostic, supported=supported)
    FC.log(f"masks: {timer.mark('masks'):.1f} s")
    # per-halo volume integrals (dimensionless: sum_s F_s 4pi/3 (x_hi^3 - x_lo^3))
    vol = FC.SHELL_VOLUME[:FC.N_F]
    for name in SELECTIONS:
        for key in ("P", "E"):
            v = np.nan_to_num(out[f"{key}_{name}"], nan=0.0)
            for radius in (1.0, 2.0, 4.0):
                inside = FC.shells_within(FC.F_EDGES, radius)
                out[f"{key}_{name}_int{radius:g}R"] = v[:, inside] @ vol[inside]
    # truth moments per bin
    for name in SELECTIONS:
        P = np.nan_to_num(out[f"P_{name}"], nan=0.0)
        Ee = np.nan_to_num(out[f"E_{name}"], nan=0.0)
        mean_p = np.zeros((nb, FC.N_F)); sd_p = np.zeros((nb, FC.N_F))
        mean_e = np.zeros((nb, FC.N_F)); sd_e = np.zeros((nb, FC.N_F))
        spp = np.zeros((nb, FC.N_F, FC.N_F)); see = np.zeros_like(spp); spe = np.zeros_like(spp)
        neff = np.zeros(nb)
        for b in range(nb):
            rb = np.flatnonzero(bins == b)
            if not len(rb):
                continue
            w = weights[rb]
            mean_p[b], sd_p[b], neff[b] = weighted_moments(P[rb], w)
            mean_e[b], sd_e[b], _ = weighted_moments(Ee[rb], w)
            spp[b] = second_moment(P[rb], P[rb], w)
            see[b] = second_moment(Ee[rb], Ee[rb], w)
            spe[b] = second_moment(P[rb], Ee[rb], w)
        out.update({f"meanP_{name}": mean_p, f"sdP_{name}": sd_p, f"meanE_{name}": mean_e, f"sdE_{name}": sd_e,
                    f"SPP_{name}": spp, f"SEE_{name}": see, f"SPE_{name}": spe, f"neff_{name}": neff})
    # spatial folds: 4^3 blocks -> 8 folds of 8 blocks (seeded permutation)
    box = float(d.plan["box_size_cmpc"])
    centre = np.mod(out["centre_mpc"], box)
    cell = np.clip(np.floor(centre / (box / BLOCKS_PER_SIDE)).astype(int), 0, BLOCKS_PER_SIDE - 1)
    block = (cell[:, 0] * BLOCKS_PER_SIDE + cell[:, 1]) * BLOCKS_PER_SIDE + cell[:, 2]
    perm = np.random.default_rng(FC.SEED).permutation(BLOCKS_PER_SIDE**3)
    block_fold = np.empty(BLOCKS_PER_SIDE**3, int)
    block_fold[perm] = np.arange(BLOCKS_PER_SIDE**3) // (BLOCKS_PER_SIDE**3 // FOLDS)
    fold = block_fold[block]
    out.update(block=block, fold=fold, block_fold=block_fold)
    counts = {FC.bin_label(b): np.bincount(fold[bins == b], minlength=FOLDS).tolist() for b in range(nb)
              if np.any(bins == b)}
    summary["folds"] = {"definition": f"{BLOCKS_PER_SIDE}^3 blocks of {box / BLOCKS_PER_SIDE:.0f} Mpc, 8 folds x 8 "
                                      f"blocks, seeded permutation (seed {FC.SEED})",
                        "haloes_per_fold_by_bin": counts}
    np.savez_compressed(FC.PRODUCTS / "profiles.npz", **out)
    d.release()                                       # drop the cached 2 GB owner block before later stages
    summary["seconds"] = timer.mark("total")
    FC.write_json(FC.PRODUCTS / "profiles_summary.json", summary)
    FC.log(f"done {summary['seconds']:.1f} s")


if __name__ == "__main__":
    main()
