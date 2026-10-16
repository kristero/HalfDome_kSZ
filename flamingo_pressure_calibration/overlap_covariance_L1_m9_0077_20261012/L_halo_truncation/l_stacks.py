"""L stage 'stacks' (local): per-bin stacked profiles with 200 paired bootstrap replicates, merged groups, bound fractions.

For every engine bin b in 12.7-12.9 ... 15.3-15.5 and every per-halo quantity q (field P, E or G; selection all, a2ii,
hb, cb, a2iia2i, p10, p11, and the nested brackets below), the stack row r of Wd (r = 0: the sample's inclusion weights;
r = 1..200: paired stratified halo resamples, stackcore's seeds) is
    S_q[r, s] = sum_h Wd[r, h] F_q[h, s] / sum_h Wd[r, h].
Nested brackets of the bound gas inside the painted A2-ii gas (per halo-shell; hb never belongs to p11):
    nest_lo = max(0, hb - p10) <= (hb AND a2ii) <= nest_hi = min(hb, a2iia2i)       (same for cb)
Temperatures and entropy (T_M, T_V, K; 'weighted' per-halo ratios) are stacked as geometric means over finite > 0
values.  The 2-halo field expectation per halo (stacked the same way, so the bootstrap includes it):
    E_2h,h(s) = Ebar [1 + b(M_h) <xi_lin(x R_h)>_s],    Ebar = kappa_E Omega_m = 0.3348
    P_2h,h(s) = [<P_e> + <b P_e> b(M_h) <xi_lin(x R_h)>_s] / P200_h
with <xi>_s the 32-point shell average (G's quad_nodes), b Tinker+10, xi_lin CLASS z = 0 (G's LinearHalo),
<P_e> = sum_h w_h Y_h / V_box and <b P_e> = sum_h w_h b(M_h) Y_h / V_box over all 4,588 haloes (Y_h = P200 R200c^3
sum_s V_s P_a2ii,h,s the halo's A2-ii pressure volume inside 4 R200c, Pa m^3; V_box = (1000 Mpc)^3): the mean and the
bias-weighted mean electron pressure carried by haloes >= 10^12.5 Msun (a lower bound on the cosmic mean, ESTIMATE).
Stack-level rule in 12.7-13.5 (Test J): fine shells are merged greedily (>= 400 stacked particles of the selection)
inside [0, x_h], [x_h, 1], [1, 2], [2, 4]; a ratio num/den is evaluated on the groups of the numerator's counts.
"""
from __future__ import annotations

from functools import lru_cache

import l_common as LC
import numpy as np

FIELDS = ("P", "E", "G")
CACHE_SELS = ("all", "a2ii", "hb", "cb", "a2iia2i")
COUNT_FOR = {"all": "count_all", "a2ii": "count_a2ii", "hb": "count_hb", "cb": "count_cb"}
# fraction definitions: name -> (numerator, denominator, count groups)
FRACTIONS = {
    "hb/all": ("hb", "all", "hb"),            # primary (nested: hb is a subset of all gas)
    "hb/a2ii": ("hb", "a2ii", "hb"),          # upper bound of the nested (hb AND a2ii)/a2ii
    "hbnest_lo/a2ii": ("nest_lo", "a2ii", "hb"),
    "hbnest_hi/a2ii": ("nest_hi", "a2ii", "hb"),
    "cb/all": ("cb", "all", "cb"),
    "cb/a2ii": ("cb", "a2ii", "cb"),
    "cbnest_lo/a2ii": ("cbnest_lo", "a2ii", "cb"),
    "cbnest_hi/a2ii": ("cbnest_hi", "a2ii", "cb"),
    "a2ii/all": ("a2ii", "all", "a2ii"),      # ownership bookkeeping (G's p_keep)
}


@lru_cache(maxsize=1)
def inputs():
    return dict(np.load(LC.INPUT_FILE))


@lru_cache(maxsize=1)
def bins():
    return LC.mass_bins()


@lru_cache(maxsize=1)
def jstacks():
    return dict(np.load(LC.J_DIR / "results" / "stacks.npz"))


@lru_cache(maxsize=1)
def halo_table():
    return LC.SC.halo_table()


@lru_cache(maxsize=1)
def linear_halo():
    return LC.GPH.LinearHalo()


@lru_cache(maxsize=1)
def xi_shell_all():
    """<b(M_h) xi_lin(x R_h)>_s per halo and shell [4588, 67] (32-point shell quadrature, G's quad_nodes)."""
    lh = linear_halo()
    t = halo_table()
    xn, wq = LC.GPH.quad_nodes(LC.X)
    out = np.zeros((len(t["m200c"]), LC.N_X))
    bh = lh.b(t["m200c"])
    for h in range(len(bh)):
        out[h] = bh[h] * np.sum(wq * lh.xi(xn * t["r200c"][h]), axis=1)
    return out


@lru_cache(maxsize=1)
def pressure_means():
    """<P_e> and <b P_e> [Pa] carried by the study's haloes (A2-ii pressure volume inside 4 R200c), and checks."""
    t = halo_table()
    P = np.asarray(LC.SC.per_halo(LC.key("P", "a2ii")))
    Y = t["p200_pa"] * (t["r200c"] * LC.MPC_M) ** 3 * (P @ LC.VOL)          # Pa m^3
    w = t["weight"]
    vbox = (LC.BOX_MPC * LC.MPC_M) ** 3
    b = linear_halo().b(t["m200c"])
    Pm = float(np.sum(w * Y) / vbox)
    bPm = float(np.sum(w * b * Y) / vbox)
    by_bin = {}
    for i, mb in bins().items():
        r = mb["rows"]
        by_bin[LC.LABELS[i]] = float(np.sum(w[r] * Y[r]) / vbox / Pm)
    return {"P_mean_pa": Pm, "bP_mean_pa": bPm, "b_P_effective": bPm / Pm, "share_of_P_mean_by_bin": by_bin,
            "sum_weights": float(np.sum(w)), "n_haloes": int(len(w))}


def per_halo_arrays(rows):
    """Every per-halo quantity of the bin rows [n, 67]."""
    inp = inputs()
    F = {}
    for f in LC.NORM:
        for s in CACHE_SELS:
            F[f"{f}_{s}"] = np.asarray(LC.SC.per_halo(LC.key(f, s))[rows], float)
        F[f"{f}_p10"] = inp[f"{f}_p10"][rows]
        F[f"{f}_p11"] = inp[f"{f}_p11"][rows]
        F[f"{f}_nest_lo"] = np.maximum(0.0, F[f"{f}_hb"] - F[f"{f}_p10"])
        F[f"{f}_nest_hi"] = np.minimum(F[f"{f}_hb"], F[f"{f}_a2iia2i"])
        F[f"{f}_cbnest_lo"] = np.maximum(0.0, F[f"{f}_cb"] - F[f"{f}_p10"])
        F[f"{f}_cbnest_hi"] = np.minimum(F[f"{f}_cb"], F[f"{f}_a2iia2i"])
    for name, k in LC.WEIGHTED_KEYS.items():
        F[name] = np.asarray(LC.SC.per_halo(k)[rows], float)
    t = halo_table()
    xi = xi_shell_all()[rows]
    pm = pressure_means()
    F["E_2h"] = LC.E_COSMIC * (1.0 + xi)
    F["P_2h"] = (pm["P_mean_pa"] + pm["bP_mean_pa"] * xi) / t["p200_pa"][rows][:, None]
    F["bxi"] = xi
    return F


def groups_for(b, rows):
    """Merged-shell groups per count selection: the stack-level rule in 12.7-13.5, raw fine shells above."""
    inp = inputs()
    raw = [(s, s + 1) for s in range(LC.N_X)]
    if b not in LC.LOW_BINS:
        return {k: raw for k in COUNT_FOR}, None
    js = jstacks()
    k_h = int(js["k_h"][b])
    breaks = sorted({0, k_h, LC.K_CUT[1.0], LC.K_CUT[2.0], LC.N_X})
    out = {k: LC.JP.greedy_groups(inp[c][rows].sum(axis=0), breaks) for k, c in COUNT_FOR.items()}
    # parity with Test J's stored groups (results/stacks.npz groups_<variant>_<bin>)
    parity = {}
    for k, jname in (("hb", "E_hb"), ("a2ii", "E_a2ii"), ("all", "E_all")):
        g = np.asarray(js[f"groups_{jname}_{b}"])
        parity[k] = bool(np.array_equal(np.asarray(out[k]), g))
    return out, parity


@lru_cache(maxsize=None)
def bin_stacks(b):
    """All stacks of engine bin b: dict(S={q: [R, 67]}, groups, parity, meta)."""
    mb = bins()[b]
    rows = np.asarray(mb["rows"])
    Wd = LC.boot_matrix(mb)
    F = per_halo_arrays(rows)
    S = {}
    for q, v in F.items():
        S[q] = LC.wstack_log(Wd, v) if q in LC.WEIGHTED_KEYS else LC.wstack(Wd, v)
    groups, parity = groups_for(b, rows)
    t = halo_table()
    inp = inputs()
    q200m = t["r200c"][rows] / inp["r200m_mpc"][rows]
    q500 = t["r200c"][rows] / t["r500c"][rows]
    wsum = Wd.sum(axis=1)
    meta = {"label": LC.LABELS[b], "N": int(len(rows)), "N_eff": float(mb["N_eff"]),
            "mean_logm200c": float(mb["mean_logm200c"]), "mean_logm500c": float(mb["mean_logm500c"]),
            "q200m": (Wd @ q200m) / wsum, "q500": (Wd @ q500) / wsum,          # <R200c/R200m>, <R200c/R500c> per row
            "q200m_pct": np.percentile(q200m, [16, 50, 84]), "q500_pct": np.percentile(q500, [16, 50, 84]),
            "x_h": float(jstacks()["x_h"][b]), "k_h": int(jstacks()["k_h"][b]),
            "stacked_counts": {k: inp[c][rows].sum(axis=0) for k, c in COUNT_FOR.items()}}
    return {"S": S, "groups": groups, "parity": parity, "meta": meta, "Wd_sum": wsum}


def fraction(b, field, name):
    """Bound-fraction points of a fraction definition: (values [R, G], radii [G], groups)."""
    st = bin_stacks(b)
    num, den, gk = FRACTIONS[name]
    g = st["groups"][gk]
    vn, xr = LC.group_points(st["S"][f"{field}_{num}"], g)
    vd, _ = LC.group_points(st["S"][f"{field}_{den}"], g)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(vd > 0, vn / vd, np.nan), xr, g


def main():
    t0 = LC.time.perf_counter()
    out, summ = {}, {"bins": {}, "pressure_means_ESTIMATE": pressure_means(), "fractions": list(FRACTIONS),
                     "n_boot": LC.N_BOOT, "seeds": {}}
    keep = ["P_all", "P_a2ii", "P_hb", "P_cb", "E_all", "E_a2ii", "E_hb", "E_cb", "G_all", "G_a2ii", "G_hb",
            "T_M", "T_V", "K", "E_2h", "P_2h", "P_nest_lo", "P_nest_hi", "E_nest_lo", "E_nest_hi"]
    nb = len(LC.BINS)
    for q in keep:
        out[f"stack_{q}"] = np.full((nb, LC.N_X), np.nan)
        out[f"q16_{q}"] = np.full((nb, LC.N_X), np.nan)
        out[f"q84_{q}"] = np.full((nb, LC.N_X), np.nan)
    for i, b in enumerate(LC.BINS):
        st = bin_stacks(b)
        for q in keep:
            v = st["S"][q]
            out[f"stack_{q}"][i] = v[0]
            out[f"q16_{q}"][i] = np.nanpercentile(v[1:], 16, axis=0)
            out[f"q84_{q}"][i] = np.nanpercentile(v[1:], 84, axis=0)
        # stackcore parity: row 0 equals stack_bin's value for every cache key used
        mb = bins()[b]
        par = {}
        for f in ("P", "E"):
            for s in ("all", "hb"):
                ref = LC.SC.stack_bin(LC.key(f, s), mb, n_boot=0, rule="none")["value"]
                par[f"{f}_{s}"] = float(np.nanmax(np.abs(st["S"][f"{f}_{s}"][0] - ref) / np.maximum(np.abs(ref), 1e-300)))
        rb = LC.SC.stack_bin(LC.key("E", "all"), mb, n_boot=LC.N_BOOT, rule="none", return_boot=True)
        par["E_all_boot_replicates_max_rel"] = float(np.nanmax(np.abs(st["S"]["E_all"][1:] - rb["boot"]) /
                                                                np.maximum(np.abs(rb["boot"]), 1e-300)))
        for f in ("P", "E"):
            for name in FRACTIONS:
                v, xr, g = fraction(b, f, name)
                out[f"frac_{f}_{name}_{b}"] = v[0].astype(np.float32)
                out[f"fracq16_{f}_{name}_{b}"] = np.nanpercentile(v[1:], 16, axis=0).astype(np.float32)
                out[f"fracq84_{f}_{name}_{b}"] = np.nanpercentile(v[1:], 84, axis=0).astype(np.float32)
                out[f"fracx_{f}_{name}_{b}"] = xr.astype(np.float32)
        m = st["meta"]
        summ["bins"][m["label"]] = {"engine_bin": b, "N": m["N"], "N_eff": m["N_eff"],
                                    "mean_logm200c": m["mean_logm200c"], "mean_logm500c": m["mean_logm500c"],
                                    "R200c_over_R200m_mean": float(m["q200m"][0]),
                                    "R200c_over_R200m_pct16_50_84": m["q200m_pct"],
                                    "R200c_over_R500c_mean": float(m["q500"][0]),
                                    "R200c_over_R500c_pct16_50_84": m["q500_pct"],
                                    "x_h": m["x_h"], "groups_parity_with_J": st["parity"],
                                    "n_groups": {k: len(g) for k, g in st["groups"].items()},
                                    "stackcore_parity_max_rel": par,
                                    "min_stacked_count_hb_group": float(min(
                                        m["stacked_counts"]["hb"][a:e].sum() for a, e in st["groups"]["hb"]))}
        summ["seeds"][m["label"]] = LC.SEED0 + 1000 * b
    out["bins"] = np.array(LC.BINS)
    out["x_edges"] = LC.X
    np.savez_compressed(LC.RESULTS / "stacks_L.npz", **out)
    summ["seconds"] = LC.time.perf_counter() - t0
    summ["peak_rss_gb"] = LC.rss_gb()
    LC.jdump(summ, LC.RESULTS / "stacks_summary.json")
    LC.log(f"stacks done: {nb} bins, {summ['seconds']:.0f} s, rss {summ['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
