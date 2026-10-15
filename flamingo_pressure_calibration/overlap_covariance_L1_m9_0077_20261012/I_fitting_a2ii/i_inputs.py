"""I0 (local): export the stacking-choice variants used in I5 from the validated stacking cache
(stacking_choices_20261014/core/stackcore.py, read-only) into I_fitting_a2ii/inputs/i0_variants.npz (+ json).

For every variant v, field f in {P, E} and mass bin b of the variant's scheme (calibrated range 13.5-15.5 in M200c):
  mean_<v>_<f>  [nb, 35]  the stack on the 'fit' grid (= Test F's 35-shell F grid, x = r/R200c to 4)
  sd_<v>_<f>    [nb, 35]  reliability-weighted halo-to-halo SD (objective (a) weights: sigma = max(SD/mean, 5%))
  res_<v>_<f>   [nb, 35]  resolved shells of the variant's rule (strict, or per-halo: n_used > 0)
  logm_<v>, lnc_<v> [nb]  inclusion-weighted mean log10 M200c [Msun] and mean ln c200c of the bin's haloes
  bins_<v>      [nb, 2]   bin edges (log10 M200c [Msun])
Units: P = P_e / P200 (dimensionless), E = n_e mu_XGP / (rho_c f_b) (dimensionless); as the cache's 'P200' and
'rhoc_fb' normalisations (identical to Test F / G).
Variants (A2-ii = the painting partition, recent-AGN gas removed, inclusion-weighted mean, strict resolution rule,
M200c 0.2-dex bins, unless stated):
  a2ii        reference (must reproduce Test G's 35-shell A2-ii stacks; checked in i0_variants.json)
  a2ii_agn    A2-ii with the recently AGN-heated gas kept
  allgas      all gas (no partition; Test F 'main')
  a2i         A2-i partition
  unweighted  A2-ii, unweighted mean
  median      A2-ii, inclusion-weighted median
  m05         A2-ii in M200c 0.5-dex bins (13.5-14, 14-14.5, 14.5-15, 15-15.5)
  perhalo     A2-ii, per-halo resolution rule (unresolved halo-shells dropped individually)
  xray        all gas (no partition), X-ray phase only (SFR = 0, T >= 1e5 K, not AGN-heated); the cache has no
              A2-ii x X-ray-phase variant, so compare with allgas to isolate the phase effect
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY / "stacking_choices_20261014" / "core"))
import stackcore as sc  # noqa: E402

VARIANTS = {
    "a2ii": dict(agn="removed", selection="A2ii", phase="all", statistic="mean", scheme="M200c_0.2", rule="strict"),
    "a2ii_agn": dict(agn="kept", selection="A2ii", phase="all", statistic="mean", scheme="M200c_0.2", rule="strict"),
    "allgas": dict(agn="removed", selection="all", phase="all", statistic="mean", scheme="M200c_0.2", rule="strict"),
    "a2i": dict(agn="removed", selection="A2i", phase="all", statistic="mean", scheme="M200c_0.2", rule="strict"),
    "unweighted": dict(agn="removed", selection="A2ii", phase="all", statistic="unweighted_mean", scheme="M200c_0.2",
                       rule="strict"),
    "median": dict(agn="removed", selection="A2ii", phase="all", statistic="median", scheme="M200c_0.2", rule="strict"),
    "m05": dict(agn="removed", selection="A2ii", phase="all", statistic="mean", scheme="M200c_0.5", rule="strict"),
    "perhalo": dict(agn="removed", selection="A2ii", phase="all", statistic="mean", scheme="M200c_0.2",
                    rule="per_halo"),
    "xray": dict(agn="removed", selection="all", phase="xray", statistic="mean", scheme="M200c_0.2", rule="strict"),
}


def main():
    ht = sc.halo_table()
    conc = np.asarray(ht["concentration_200c"], float)
    out, meta = {"x_edges_fit": sc.edges("fit")}, {"doc": __doc__, "variants": {}}
    for v, d in VARIANTS.items():
        bins = [b for b in sc.mass_bins(d["scheme"]) if b["lo"] >= 13.5 - 1e-9 and b["hi"] <= 15.5 + 1e-9]
        out[f"bins_{v}"] = np.array([[b["lo"], b["hi"]] for b in bins])
        out[f"logm_{v}"] = np.array([b["mean_logm200c"] for b in bins])
        out[f"lnc_{v}"] = np.array([np.sum(b["weights"] * np.log(conc[b["rows"]])) / np.sum(b["weights"]) for b in bins])
        out[f"N_{v}"] = np.array([b["N"] for b in bins])
        meta["variants"][v] = dict(d, keys={})
        for f in ("P", "E"):
            key = sc.make_key(f, "geometric", d["agn"], d["selection"], d["phase"], "fit")
            meta["variants"][v]["keys"][f] = key
            means, sds, res = [], [], []
            for b in bins:
                r = sc.stack_bin(key, b, d["statistic"], d["rule"], n_boot=2)
                means.append(r["value"])
                sds.append(r["sd"])
                res.append(np.asarray(r["resolved"], bool))
            out[f"mean_{v}_{f}"] = np.array(means)
            out[f"sd_{v}_{f}"] = np.array(sds)
            out[f"res_{v}_{f}"] = np.array(res)
        print(v, len(bins), flush=True)
    # parity: the reference against Test G's 35-shell A2-ii stacks (G_physical_outer/results/f_base/profiles.npz)
    g = np.load(STUDY / "G_physical_outer" / "results" / "f_base" / "profiles.npz")
    cal = list(range(5, 15))
    for f in ("P", "E"):
        a, b = out[f"mean_a2ii_{f}"], g[f"mean{f}_a2ii"][cal]
        ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
        meta[f"parity_vs_G_mean{f}_a2ii_max_rel"] = float(np.max(np.abs(a[ok] / b[ok] - 1)))
    np.savez_compressed(HERE / "inputs" / "i0_variants.npz", **out)
    (HERE / "inputs" / "i0_variants.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print({k: v for k, v in meta.items() if k.startswith("parity")})


if __name__ == "__main__":
    main()
