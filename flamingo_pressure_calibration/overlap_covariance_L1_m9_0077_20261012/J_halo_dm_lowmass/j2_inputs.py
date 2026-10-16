"""J2 inputs (local, numpy only, < 1 GB RAM): the per-halo fit data of Test J2 on Test F's 35-shell grid, the fit masks,
the low-mass 'resolved shape + core' (RC) grids and the importance-derived weights of every definition, in one compact
file inputs/j2_fitdata.npz (+ inputs/j2_fitdata_meta.json with the checks).  Mirrored to maas with j2_upload.py.

Sources (read-only):
  inputs/j0_profiles_x200.npz        J0 export (Test J1): per-halo P_a2ii, E_a2ii, E_hb on the 67 X200 shells, owned
                                     particle counts, SPH smoothing lengths, halo table
  G_physical_outer/results/f_base/profiles.npz   Test F/G 35-shell per-halo profiles: exact F edges and the strict
                                     resolution masks of the bins >= 13.5 (base_mask = strict_mask there)
  results/stacks.npz                 Test J1: x_h, softening limit and the merged T1 groups of the low-mass bins (checked
                                     against this module's own recomputation)
  results/sens.npz                   Test J1: signed importance I_O[b, s] of every definition (C14, T1)

Quantities (x = r/R200c; s = shell; b = engine mass bin; h = halo; w_h = inclusion weight):
  F35_h,s   = sum_{f in s} V_f F_h,f / sum_{f in s} V_f   (volume-weighted mean of the fine shells f of the F shell s,
              V_f = 4pi/3 (x_f+1^3 - x_f^3)); F = P (P_e/P200) or E (n_e mu_XGP/(rho_c f_b)), dimensionless
  N_b,s     = sum_{h in b} n_h,s: stacked gas particles of the selection in shell s (n_h,s owned particles, AGN cut)
  x_h       = first fine edge x_f >= median_h(2 h_h,f / R200c), h_h,f the upper-median SPH smoothing length
  family weight w_fam,D[c, f] = mean over the family's observables O (and 100 <= ell <= 3000) of |I_O,D[c, f]|
              (I_O,D: J1's importance of definition D; DM family: mean, variance and the 16/50/84/97.5/99.9% quantiles;
              kSZ: 1-halo; tSZxDM_E: y x DM, electron side); fit weights of objective (d): w_DM,D summed into the fit
              shell and normalised to unit sum per bin; 'dI' (Test I's recipe, D0 only): 0.5 w_kSZ + 0.5 w_DM, each
              normalised per bin first.
Usage: OMP_NUM_THREADS=2 python -B j2_inputs.py   [J_MODE=smoke: J1's smoke bins 13.1-13.3, 14.5-14.7 and 25 haloes per bin]
"""
from __future__ import annotations

import hashlib
import warnings

import j2_common as C
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def hydro_limit(hsml_rows, X):
    """j_prepare.hydro_limit: (k_h, x_h) = first fine edge with x_s >= median(2 h/R200c) of shell s."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(2 * hsml_rows, axis=0)
    ok = np.flatnonzero(X[:-1] >= np.where(np.isfinite(med), med, np.inf))
    k = int(ok[0]) if len(ok) else len(X) - 1
    return k, float(X[k])


def greedy_groups(counts, breaks, n_x, min_count=C.MIN_STACK_COUNT):
    """j_prepare.greedy_groups (identical logic)."""
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
    assert groups[0][0] == 0 and groups[-1][1] == n_x
    return groups


def weighted_mean(values, w):
    w = np.asarray(w, float) / np.sum(w)
    return w @ values


def main():
    j0 = C.load_npz(C.J0)
    g35 = C.load_npz(C.G_PROFILES35, keys=["f_edges", "base_mask", "strict_mask", "P_a2ii", "E_a2ii", "mass_bin",
                                            "weight"])
    X = np.asarray(j0["x200_edges"], float)
    n_x = len(X) - 1
    F35 = np.asarray(g35["f_edges"], float)
    assert np.max(np.abs(F35 - C.F35)) < 1e-7
    idx35 = C.edge_index(X, F35)
    n_all = len(j0["weight"])
    rows = C.smoke_rows(j0["mass_bin"]) if C.SMOKE else np.arange(n_all)
    mb = j0["mass_bin"][rows]
    w = j0["weight"][rows]
    meta = {"mode": C.MODE, "n_haloes": int(len(rows)), "j_bins": C.J_BINS, "low_bins": C.LOW_BINS,
            "sources": {str(p.relative_to(C.ROOT)): sha(p) for p in (C.J0, C.G_PROFILES35, C.J1_FULL_RESULTS / "stacks.npz",
                                                                       C.J1_FULL_RESULTS / "sens.npz")},
            "definitions": C.__doc__, "checks": {}}
    out = {"f35_edges": F35, "x200_edges": X, "idx35": idx35, "rows": rows, "weight": w, "mass_bin": mb,
           "logm200c": j0["logm200c"][rows], "m200c": j0["m200c"][rows], "r200c": j0["r200c"][rows],
           "fold": j0["fold"][rows], "concentration_200c": j0["concentration_200c"][rows], "strata": j0["strata"][rows],
           "j_bins": np.array(C.J_BINS), "low_bins": np.array(C.LOW_BINS)}
    raw = {"P_a2ii": j0["P_a2ii"][rows], "E_a2ii": j0["E_a2ii"][rows], "E_hb": j0["E_hb"][rows]}
    cnt = {"a2ii": j0["count_a2ii"][rows], "hb": j0["count_hb"][rows]}
    for k, v in raw.items():
        out[f"{k}_35"] = C.rebin(v, X, idx35)
    # ---- parity of the rebinned A2-ii profiles with Test F/G's own 35-shell profiles
    gr = rows
    for k in ("P_a2ii", "E_a2ii"):
        ref = g35[k][gr]
        nz = ref != 0
        meta["checks"][f"{k}_35_vs_G_f35_max_rel"] = float(np.max(np.abs(out[f"{k}_35"][nz] / ref[nz] - 1)))
        meta["checks"][f"{k}_35_zero_pattern_equal_G"] = bool(np.array_equal(out[f"{k}_35"] == 0, ref == 0))
    # ---- stacked counts, x_h, softening
    cnt35 = {s: np.zeros((C.N_ENGINE, len(F35) - 1)) for s in cnt}
    cntf = {s: np.zeros((C.N_ENGINE, n_x)) for s in cnt}
    x_h = np.full(C.N_ENGINE, np.nan)
    k_h = np.full(C.N_ENGINE, n_x)
    soft = np.full(C.N_ENGINE, np.nan)
    for b in range(C.N_ENGINE):
        rb = np.flatnonzero(mb == b)
        if not len(rb):
            continue
        for s in cnt:
            cntf[s][b] = cnt[s][rb].sum(axis=0)
            cnt35[s][b] = np.add.reduceat(cntf[s][b], idx35[:-1])
        k_h[b], x_h[b] = hydro_limit(j0["hsml"][rows][rb], X)
        soft[b] = float(np.median(4 * j0["eps_over_r200c"][rows][rb]))
    out.update(x_h=x_h, k_h=k_h, softening_limit=soft, count35_a2ii=cnt35["a2ii"], count35_hb=cnt35["hb"])
    if not C.SMOKE:
        st = C.load_npz(C.J1_FULL_RESULTS / "stacks.npz")
        meta["checks"]["x_h_equal_J1"] = bool(np.allclose(st["x_h"][1:], x_h[1:], rtol=0, atol=0))
        meta["checks"]["softening_equal_J1"] = float(np.nanmax(np.abs(st["softening_limit"] - soft)))
        meta["checks"]["stacked_counts_equal_J1"] = bool(np.array_equal(st["stacked_count_a2ii"], cntf["a2ii"])
                                                         and np.array_equal(st["stacked_count_hb"], cntf["hb"]))
    # ---- fit masks on the 35-shell grid
    x_lo = F35[:-1]
    masks = {}
    for sel, ekey in (("a2ii", "E_a2ii"), ("hb", "E_hb"), ("P", "P_a2ii")):
        cs = "hb" if sel == "hb" else "a2ii"
        m = np.zeros((C.N_ENGINE, len(F35) - 1), bool)
        for b in C.J_BINS:
            rb = np.flatnonzero(mb == b)
            mean = weighted_mean(out[f"{ekey}_35"][rb], w[rb])
            if b >= 5:
                assert np.array_equal(g35["base_mask"][b], g35["strict_mask"][b])
                m[b] = np.asarray(g35["base_mask"][b], bool) & (mean > 0)
                if sel == "hb":
                    m[b] &= cnt35["hb"][b] >= C.MIN_STACK_COUNT
            else:
                m[b] = (x_lo >= max(C.SOFTENING_FLOOR, soft[b]) - 1e-12) & (cnt35[cs][b] >= C.MIN_STACK_COUNT) \
                    & (mean > 0)
        masks[sel] = m
        out[f"mask35_{sel}"] = m
        meta["checks"][f"mask35_{sel}_first_edge_per_bin"] = {
            C.bin_label(b): (float(F35[np.flatnonzero(m[b])[0]]) if m[b].any() else None) for b in C.J_BINS}
        meta["checks"][f"mask35_{sel}_shells_inside_1_2_4R"] = {
            C.bin_label(b): [int(np.sum(m[b] & C.inside(F35, r))) for r in (1.0, 2.0, 4.0)] for b in C.J_BINS}
        if sel == "hb":
            meta["checks"]["hb_count_rule_binds_above_13.5"] = int(sum(
                int(np.sum(np.asarray(g35["base_mask"][b], bool) & (cnt35["hb"][b] < C.MIN_STACK_COUNT)))
                for b in C.HIGH_BINS))
    # ---- RC grids (low bins): [0, x_h] + the J1 merged T1 groups outside x_h
    st_groups = C.load_npz(C.J1_FULL_RESULTS / "stacks.npz") if not C.SMOKE else {}
    for b in C.LOW_BINS:
        rb = np.flatnonzero(mb == b)
        out[f"rows_bin{b}"] = rb
        for sel, key, cs in (("a2ii", "E_a2ii", "a2ii"), ("hb", "E_hb", "hb")):
            breaks = sorted({0, int(k_h[b]), int(C.edge_index(X, 1.0)[0]), int(C.edge_index(X, 2.0)[0]), n_x})
            g = greedy_groups(cntf[cs][b], breaks, n_x)
            ref = st_groups.get(f"groups_{key}_{b}")
            if ref is not None:
                meta["checks"][f"groups_{key}_{b}_equal_J1"] = bool(np.array_equal(np.array(g), ref))
            eidx = [0, int(k_h[b])] + [a for a, e in g if a > k_h[b]] + [n_x]
            eidx = np.array(sorted(set(eidx)))
            out[f"rc_edges_{sel}_{b}"] = eidx
            out[f"rc_E_{sel}_{b}"] = C.rebin(raw[key][rb], X, eidx)
            out[f"rc_count_{sel}_{b}"] = np.add.reduceat(cntf[cs][b], eidx[:-1])
            if sel == "a2ii":
                out[f"rc_P_{sel}_{b}"] = C.rebin(raw["P_a2ii"][rb], X, eidx)
    # ---- importance-derived weights (J1 sens.npz; smoke uses the full maps restricted to the smoke bins)
    sens = C.load_npz(C.J1_FULL_RESULTS / "sens.npz")
    ell = np.asarray(sens["sel_ell"], float)
    ok_ell = (ell >= 100 - 1e-9) & (ell <= 3000 + 1e-9)
    jb = C.J_BINS
    fams = {"DM": C.IMP_DM, "kSZ": ("ksz1h",), "tSZxDM_E": ("ydmtot_E",)}
    imp_dm = {dd: tuple(n for n in C.IMP_DM if not (dd == "D1" and n == "dm_q16")) for dd in C.DEFS}
    for d in C.DEFS:
        for fam, names in fams.items():
            names = imp_dm[d] if fam == "DM" else names
            acc = []
            for nm in names:
                key = f"imp__{d}__{nm}"
                if key not in sens:
                    continue
                M = np.abs(np.asarray(sens[key], float))[jb][:, :n_x]
                if M.ndim == 3:
                    M = M[..., ok_ell].mean(axis=-1)
                acc.append(np.nan_to_num(M))
            if not acc:
                continue
            Wf = np.mean(acc, axis=0)                                  # [N_J, 67]
            out[f"wfine_{fam}_{d}"] = Wf
            W35 = np.add.reduceat(Wf, idx35[:-1], axis=1)
            out[f"w35_{fam}_{d}"] = W35 / W35.sum(axis=1, keepdims=True)
            for b in C.LOW_BINS:
                for sel in ("a2ii", "hb"):
                    if (sel == "hb") != (C.DEFS[d]["sel"] == "hb"):
                        continue
                    e = out[f"rc_edges_{sel}_{b}"]
                    Wrc = np.add.reduceat(Wf[jb.index(b)], e[:-1])
                    out[f"wrc_{fam}_{d}_{b}"] = Wrc / Wrc.sum()
        meta["checks"][f"DM_weight_maps_{d}_n_obs"] = sum(f"imp__{d}__{nm}" in sens for nm in imp_dm[d])
    out["w35_dI_D0"] = 0.5 * out["w35_kSZ_D0"] + 0.5 * out["w35_DM_D0"]
    for b in C.LOW_BINS:
        out[f"wrc_dI_D0_{b}"] = 0.5 * out[f"wrc_kSZ_D0_{b}"] + 0.5 * out[f"wrc_DM_D0_{b}"]
    # D1's 16% quantile map (inside the atom) - report its size relative to the other quantile maps
    for d in C.DEFS:
        q = {nm: float(np.abs(np.asarray(sens[f'imp__{d}__{nm}'], float)).sum()) for nm in C.IMP_DM[2:]
             if f"imp__{d}__{nm}" in sens}
        meta["checks"][f"abs_importance_sum_quantile_maps_{d}"] = q
    np.savez_compressed(C.J2_INPUT, **out)
    meta["output"] = {"path": str(C.J2_INPUT), "sha256": sha(C.J2_INPUT), "bytes": C.J2_INPUT.stat().st_size}
    C.write_json(C.J2_INPUT_META, meta)
    print(C.json.dumps(C.plain(meta["checks"]), indent=1)[:6000])


if __name__ == "__main__":
    main()
