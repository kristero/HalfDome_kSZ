"""I0b (local, light): inputs of the per-halo resolution-rule test (maas run 3, i_perhalo.py), from the validated
stacking cache (stacking_choices_20261014/core/stackcore.py, read-only) -> inputs/i0_perhalo_x200.npz (+ .json).

Why: in runs 1-2 the I5 variant 'perhalo' (A2-ii stacked with the per-halo resolution rule) was fitted on the STRICT
resolved shells only (i_g35.stage_fits35 ANDs every 0.2-dex variant's mask with Test G's base_mask; stage_lvariants
with strict_mask).  On those shells every halo is resolved, so its stack equals A2-ii exactly and the variant was the
reference by construction.  Run 3 refits it on its OWN resolved shells and also paints the per-halo-rule stack itself.

Resolution rules (stackcore.resolution_detail; x = r/R200c, shell s = [x_s, x_{s+1}]):
  halo-shell resolved  x_s >= max(0.04, 4 eps / R200c)  and  N_gas(shell) >= 20  and  x_s >= 2 h_med / R200c
                       (eps = 0.0057 Mpc softening; N_gas = gas particles of all owners and phases in the shell;
                       h_med = upper-median SPH smoothing length of the shell's gas, comoving -> physical)
  strict               a shell is used for the bin only if EVERY halo of the bin resolves it; all haloes enter
  per-halo             each halo enters only the shells it resolves itself; a shell is 'resolved' for the bin if at
                       least one halo resolves it (n_used > 0)
For every calibrated bin c (M200c 0.2-dex bins 13.5-15.5, log10 Msun) and field f in {P, E}:
  mean_strict_<f>   [10, 67]  inclusion-weighted A2-ii mean on the fine x200 grid, strict rule (all haloes in every
                              shell; must equal Test H's / Test I's A2-ii stack 'mstack' = the painting target)
  mean_perhalo_<f>  [10, 67]  the same with the per-halo rule (NaN where n_used = 0)
  nused_perhalo_<f> [10, 67]  number of haloes entering each shell under the per-halo rule
  wfrac_perhalo_<f> [10, 67]  their share of the bin's inclusion weight
  strict_<f>, own_<f> [10, 67] resolved shells of the two rules
  nused_fit_<f>, wfrac_fit_<f>, own_fit_<f>, strict_fit_<f> [10, 35]  the same on the 35-shell fit grid
  N [10], x_edges_x200 [68], x_edges_fit [36], bins [10, 2]
Units: P = P_e / P200, E = n_e mu_XGP / (rho_c f_b) (dimensionless; the cache's 'P200' / 'rhoc_fb' normalisations).
  python -B i_perhalo_inputs.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "stacking_choices_20261014" / "core"))
import stackcore as sc  # noqa: E402


def main():
    bins = [b for b in sc.mass_bins("M200c_0.2") if b["lo"] >= 13.5 - 1e-9 and b["hi"] <= 15.5 + 1e-9]
    assert len(bins) == 10
    out = {"x_edges_x200": sc.edges("x200"), "x_edges_fit": sc.edges("fit"),
           "bins": np.array([[b["lo"], b["hi"]] for b in bins]), "N": np.array([b["N"] for b in bins])}
    meta = {"doc": __doc__, "keys": {}}
    for f in ("P", "E"):
        for grid, tag in (("x200", ""), ("fit", "_fit")):
            key = sc.make_key(f, "geometric", "removed", "A2ii", "all", grid)
            meta["keys"][f"{f}{tag}"] = key
            ms, mp, nu, wf, own, st = [], [], [], [], [], []
            for b in bins:
                rs = sc.stack_bin(key, b, "mean", "strict", n_boot=2)
                rp = sc.stack_bin(key, b, "mean", "per_halo", n_boot=2)
                hm = sc.resolution_mask(b["rows"], grid, "per_halo")             # [n, S]
                w = np.asarray(b["weights"], float)
                ms.append(rs["value"])
                mp.append(np.where(rp["n_used"] > 0, rp["value"], np.nan))
                nu.append(rp["n_used"])
                wf.append((w[:, None] * hm).sum(axis=0) / w.sum())
                own.append(np.asarray(rp["resolved"], bool))
                st.append(np.asarray(rs["resolved"], bool))
            if grid == "x200":
                out[f"mean_strict_{f}"], out[f"mean_perhalo_{f}"] = np.array(ms), np.array(mp)
                out[f"nused_perhalo_{f}"], out[f"wfrac_perhalo_{f}"] = np.array(nu), np.array(wf)
                out[f"own_{f}"], out[f"strict_{f}"] = np.array(own), np.array(st)
            else:
                out[f"nused_fit_{f}"], out[f"wfrac_fit_{f}"] = np.array(nu), np.array(wf)
                out[f"own_fit_{f}"], out[f"strict_fit_{f}"] = np.array(own), np.array(st)
                out[f"mean_perhalo_fit_{f}"] = np.array(mp)
    # checks: strict x200 stack = Test I's painting target (evaluate.npz mstack); on the strict shells the per-halo
    # stack equals the strict one (every halo resolves them); the fit-grid per-halo stack = inputs/i0_variants.npz
    ev = np.load(HERE / "results" / "evaluate.npz")
    z0 = np.load(HERE / "inputs" / "i0_variants.npz")
    for f in ("P", "E"):
        m = ev[f"mstack_{f}"]
        ok = np.isfinite(m) & (m != 0)
        meta[f"strict_x200_vs_mstack_{f}_max_rel"] = float(np.max(np.abs(out[f"mean_strict_{f}"][ok] / m[ok] - 1)))
        s = out[f"strict_{f}"]
        a, b = out[f"mean_perhalo_{f}"][s], out[f"mean_strict_{f}"][s]
        meta[f"perhalo_eq_strict_on_strict_shells_{f}_max_rel"] = float(np.max(np.abs(a / b - 1)))
        own = out[f"own_fit_{f}"]
        v = z0[f"mean_perhalo_{f}"]
        meta[f"perhalo_fit_vs_i0_variants_{f}_max_rel"] = float(np.nanmax(np.abs(out[f"mean_perhalo_fit_{f}"][own] /
                                                                                v[own] - 1)))
        meta[f"own_fit_equals_i0_res_perhalo_{f}"] = bool(np.array_equal(own, z0[f"res_perhalo_{f}"].astype(bool)))
        meta[f"n_shells_own_x200_{f}"] = out[f"own_{f}"].sum(axis=1).tolist()
        meta[f"n_shells_strict_x200_{f}"] = s.sum(axis=1).tolist()
    np.savez_compressed(HERE / "inputs" / "i0_perhalo_x200.npz", **out)
    (HERE / "inputs" / "i0_perhalo_x200.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "doc"}, indent=1))


if __name__ == "__main__":
    main()
