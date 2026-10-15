"""H stage 'prepare': per-halo fine profiles and per-bin truth moments in Test F's profiles format (fine grid).

From inputs/h0_profiles_x200.npz (H0) builds results/work/g/f_base/profiles.npz with Test F's keys, on the 67 fine
shells, so that Test F's / G's functions (BinData, scatter tables, ...) run unchanged on the fine grid:
  P_<sel>, E_<sel> [n, 67]      per-halo profiles, sel = a2ii (the painting target) and main (all gas)
  weight, mass_bin, logmass, halo_rows, fold, logm500c, strata, class_main
  base_mask / strict_mask / diagnostic_mask [15, 67], supported [15]
  meanP_<sel>, sdP_<sel>, meanE_<sel>, sdE_<sel> [15, 67]  inclusion-weighted mean and reliability-weighted SD
  SPP_<sel>, SEE_<sel>, SPE_<sel> [15, 67, 67]             inclusion-weighted second moments <F_s G_t>
  P_<sel>_int{1,2,4}R, E_<sel>_int{1,2,4}R                 per-halo volume integrals (x^3 units)
Smoke mode keeps Test F's smoke rows (F_observables/results_smoke/profiles.npz halo_rows).
Also writes results/prepare.json (sample sizes, rebinning parity of the means / second moments with Test F's).
"""
from __future__ import annotations

import h_common as HC
import numpy as np

FC = HC.FC


def second_moment(a, b, w):
    w = np.asarray(w, float) / np.sum(w)
    return np.einsum("h,hs,ht->st", w, a, b)


def build_profiles():
    h0 = HC.load_npz(HC.H0_PROFILES)
    fpr = HC.load_npz(HC.F_PROFILES)
    n_all = len(h0["weight"])
    rows = np.asarray(fpr["halo_rows"]) if HC.SMOKE else np.arange(n_all)
    assert np.array_equal(np.asarray(fpr["halo_rows"]), rows)
    out = {"f_edges": HC.X200, "basis_edges": HC.FINE, "halo_rows": rows, "weight": h0["weight"][rows],
           "mass_bin": h0["mass_bin"][rows], "logmass": h0["logm200c"][rows], "logm500c": h0["logm500c"][rows],
           "m200c_msun": h0["m200c"][rows], "r200c_mpc": h0["r200c"][rows], "fold": h0["fold"][rows],
           "class_main": h0["class_main"][rows], "strata": h0["strata"][rows],
           "concentration_200c": h0["concentration_200c"][rows]}
    assert np.array_equal(out["mass_bin"], fpr["mass_bin"]) and np.allclose(out["weight"], fpr["weight"], rtol=0, atol=0)
    for k in ("base_mask", "strict_mask", "diagnostic_mask", "supported", "resolved_fraction"):
        out[k] = h0[k]
    for sel in HC.SELECTIONS:
        for f in ("P", "E"):
            out[f"{f}_{sel}"] = np.asarray(h0[f"{f}_{sel}"][rows], float)
    vol = HC.VOL_FINE[:HC.N_X]
    for sel in HC.SELECTIONS:
        for f in ("P", "E"):
            v = np.nan_to_num(out[f"{f}_{sel}"], nan=0.0)
            for radius in (1.0, 2.0, 4.0):
                ins = HC.inside(HC.X200, radius)
                out[f"{f}_{sel}_int{radius:g}R"] = v[:, ins] @ vol[ins]
    nb = len(FC.MASS_EDGES) - 1
    bins = out["mass_bin"]
    for sel in HC.SELECTIONS:
        P = np.nan_to_num(out[f"P_{sel}"], nan=0.0)
        Ee = np.nan_to_num(out[f"E_{sel}"], nan=0.0)
        res = {k: np.zeros((nb, HC.N_X)) for k in ("mp", "sp", "me", "se")}
        S = {k: np.zeros((nb, HC.N_X, HC.N_X)) for k in ("PP", "EE", "PE")}
        neff = np.zeros(nb)
        for b in range(nb):
            rb = np.flatnonzero(bins == b)
            if not len(rb):
                continue
            w = out["weight"][rb]
            res["mp"][b], res["sp"][b], neff[b] = FC.S.weighted_stack(P[rb], w)
            res["me"][b], res["se"][b], _ = FC.S.weighted_stack(Ee[rb], w)
            S["PP"][b] = second_moment(P[rb], P[rb], w)
            S["EE"][b] = second_moment(Ee[rb], Ee[rb], w)
            S["PE"][b] = second_moment(P[rb], Ee[rb], w)
        out.update({f"meanP_{sel}": res["mp"], f"sdP_{sel}": res["sp"], f"meanE_{sel}": res["me"],
                    f"sdE_{sel}": res["se"], f"SPP_{sel}": S["PP"], f"SEE_{sel}": S["EE"], f"SPE_{sel}": S["PE"],
                    f"neff_{sel}": neff})
    return out, fpr


def main():
    timer = HC.Timer()
    pr, fpr = build_profiles()
    np.savez(FC.PRODUCTS / "profiles.npz", **pr)
    info = {"mode": HC.MODE, "n_haloes": int(len(pr["weight"])), "calibrated_bins": HC.CAL_LABELS,
            "haloes_per_calibrated_bin": [int(np.sum(pr["mass_bin"] == b)) for b in HC.CAL_BINS],
            "fine_shells": HC.N_X, "basis_shells": HC.N_FINE, "fine_edges": HC.X200, "extension_edges": HC.EXT_H}
    # parity of the fine means / second moments, rebinned to Test F's shells, with Test F's own
    vol = HC.VOL_FINE[:HC.N_X]
    W = np.zeros((HC.N_X, HC.N_F35))
    W[np.arange(HC.N_X), HC.FINE_OF_F] = vol
    W /= W.sum(axis=0, keepdims=True)                     # fine -> F volume-weighted rebinning matrix
    worst = {}
    for sel in HC.SELECTIONS:
        for k in ("meanP", "meanE"):
            reb = pr[f"{k}_{sel}"] @ W
            ref = fpr[f"{k}_{sel}"]
            nz = ref != 0
            worst[f"{k}_{sel}"] = float(np.max(np.abs(reb[nz] / ref[nz] - 1)))
        for k in ("SPP", "SEE", "SPE"):
            reb = np.einsum("bst,sS,tT->bST", pr[f"{k}_{sel}"], W, W)
            ref = fpr[f"{k}_{sel}"]
            nz = ref != 0
            worst[f"{k}_{sel}"] = float(np.max(np.abs(reb[nz] / ref[nz] - 1)))
    info["rebinned_moments_vs_test_F_max_rel"] = worst
    info["seconds"] = timer.mark("total")
    HC.write_json(HC.RESULTS / "prepare.json", info)
    HC.log("prepare done", worst)
    return info


if __name__ == "__main__":
    main()
