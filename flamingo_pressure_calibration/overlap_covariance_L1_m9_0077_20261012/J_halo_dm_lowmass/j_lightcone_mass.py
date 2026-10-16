"""Masses of the painted HalfDome lightcone haloes at z <= 1 (the DM range, sources at z_s = 1) in the lowest bins.

Local and light (one 0.1 MB histogram, no engine import). It reads the frozen lightcone histogram the engine paints
(kernels/engine.py Lightcone: N[z, M] haloes per redshift shell and ln-mass column; mass column centre
M = exp(ln-mass centre)/h in Msun (columns 0.0115-0.0117 dex wide) with h = 0.68 of lightcone_analytic.py:40; engine mass bins 12.5, 12.7, ..., 15.5
with the overflow bin >= 15.5) and writes post/lightcone_mass_z.json:
  - parity: per-engine-bin counts (all z and z <= 1) and the count-weighted mean log10 M per bin (all z) are recomputed
    and compared with results/prepare.json 'lightcone' (computed on maas by the engine's own Lightcone object), and the
    histogram's SHA256 is compared with the one recorded in results/run_manifest.json;
  - the smallest painted mass at all z and at z <= 1, and the lowest redshift at which each low mass column is occupied;
  - per engine bin at z <= 1: haloes and the count-weighted mean log10 M;
  - counts per mass column at z <= 1 below 10^13 and a completeness ESTIMATE: ratio of the counts to a power law in M
    (straight line in ln N vs log10 M) fitted to the z <= 1 columns with centres 12.9 < log10 M < 13.1 and extrapolated
    downwards.
z <= 1 selects redshift shells by their centre, as j_prepare.py and the engine's DM kernel (s_DM = 0 beyond z = 1) do.
Usage: python -B j_lightcone_mass.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REL = "../../frb_map_generation/outputs/lee22_tsz_verification_20260929/B_class_sz/catalogue/catalogue_hist.npz"
COUNTS = (ROOT / REL).resolve()
H_LITTLE = 0.68                                    # lightcone_analytic.py:40 (H = .68), as engine.Lightcone uses
EDGES = 12.5 + 0.2 * np.arange(16)                 # engine support mass_edges: 15 bins + overflow (>= 15.5)
LABELS = [f"{a:.1f}-{a + 0.2:.1f}" for a in EDGES[:-1]] + [">=15.5"]
FIT_RANGE = (12.9, 13.1)
Z_DM = 1.0


def main():
    raw = COUNTS.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    man = json.loads((HERE / "results" / "run_manifest.json").read_text())
    sha_maas = [it["sha256"] for inv in man["invocations"] for it in inv.get("inputs_read", [])
                if it.get("relative_to_study_root") == REL]
    h = np.load(COUNTS)
    N = np.asarray(h["N"], float)
    z = 0.5 * (h["z_edges"][1:] + h["z_edges"][:-1])
    lm = np.log10(np.exp(0.5 * (h["lnm_edges"][1:] + h["lnm_edges"][:-1])) / H_LITTLE)
    lme = np.log10(np.exp(h["lnm_edges"]) / H_LITTLE)                 # column edges in log10(M200c / Msun)
    width = np.diff(lme)
    mb = np.clip(np.searchsorted(EDGES, lm, side="right") - 1, 0, len(EDGES) - 1)
    z1 = z <= Z_DM
    c_all, c_z1 = N.sum(axis=0), N[z1].sum(axis=0)
    per_all = np.bincount(mb, weights=c_all, minlength=16)
    per_z1 = np.bincount(mb, weights=c_z1, minlength=16)
    mean_all = np.array([np.sum(c_all[mb == b] * lm[mb == b]) / max(per_all[b], 1e-300) for b in range(16)])
    mean_z1 = np.array([np.sum(c_z1[mb == b] * lm[mb == b]) / max(per_z1[b], 1e-300) for b in range(16)])
    # ---- parity with the maas run (prepare.json lightcone, engine Lightcone)
    lc = json.loads((HERE / "results" / "prepare.json").read_text())["lightcone"]
    occ_b = per_all > 0
    parity = {
        "counts_all_z_max_abs_diff": float(np.max(np.abs(per_all - np.asarray(lc["haloes_per_engine_bin"])))),
        "counts_z_le_1_max_abs_diff": float(np.max(np.abs(per_z1 - np.asarray(lc["haloes_per_engine_bin_z_le_1"])))),
        "mean_logm_all_z_max_abs_diff": float(np.max(np.abs(mean_all[occ_b]
                                                            - np.asarray(lc["mean_logm200c_per_engine_bin"])[occ_b]))),
        "min_logm_all_z_abs_diff": float(abs(lm[c_all > 0].min() - lc["min_logm200c_painted"])),
        "sha256_local": sha, "sha256_maas_manifest": sha_maas[0] if sha_maas else None,
    }
    parity["passed"] = bool(parity["counts_all_z_max_abs_diff"] == 0 and parity["counts_z_le_1_max_abs_diff"] == 0
                            and parity["mean_logm_all_z_max_abs_diff"] < 1e-12 and parity["min_logm_all_z_abs_diff"] < 1e-12
                            and parity["sha256_maas_manifest"] == sha)
    # ---- lowest columns: first redshift at which each is occupied, counts at z <= 1
    fit = (lm > FIT_RANGE[0]) & (lm < FIT_RANGE[1]) & (c_z1 > 0)
    coef = np.polyfit(lm[fit], np.log(c_z1[fit]), 1)
    resid = np.log(c_z1[fit]) - np.polyval(coef, lm[fit])
    cols = []
    for j in np.flatnonzero((lm < 13.0) & (c_all > 0)):
        occ = N[:, j] > 0
        i0 = int(np.flatnonzero(occ)[0])
        cols.append({"logm200c_centre": float(lm[j]), "logm200c_lo": float(lme[j]),
                     "logm200c_hi": float(lme[j + 1]), "width_dex": float(width[j]), "engine_bin": LABELS[mb[j]],
                     "count_all_z": float(c_all[j]), "count_z_le_1": float(c_z1[j]),
                     "z_min_occupied_shell_centre": float(z[i0]), "z_min_occupied_shell_lower_edge": float(h["z_edges"][i0]),
                     "completeness_z_le_1_ESTIMATE": float(c_z1[j] / np.exp(np.polyval(coef, lm[j])))})
    first_complete = next(c for c in cols if c["completeness_z_le_1_ESTIMATE"] > 0.95)
    peak = int(np.argmax(c_z1))
    s82 = json.loads((HERE / "results" / "prepare.json").read_text())["subset_12.82"]
    b1 = 1                                                  # engine bin 12.7-12.9
    out = {
        "generated_by": "j_lightcone_mass.py", "source_histogram": str(COUNTS), "source_relative_to_study_root": REL,
        "h": H_LITTLE, "mass_column_width_dex_range": [float(width.min()), float(width.max())], "z_selection": "redshift-shell centre <= 1 (DM sources at z_s = 1)",
        "parity_with_results_prepare_lightcone": parity,
        "min_logm200c_painted_all_z": float(lm[c_all > 0].min()),
        "min_logm200c_painted_z_le_1": float(lm[c_z1 > 0].min()),
        "z_min_of_lowest_painted_column_shell_centre": cols[0]["z_min_occupied_shell_centre"],
        "per_engine_bin": {"labels": LABELS, "haloes_all_z": per_all, "haloes_z_le_1": per_z1,
                           "mean_logm200c_all_z": mean_all, "mean_logm200c_z_le_1": mean_z1},
        "bin_12.7-12.9": {"haloes_all_z": float(per_all[b1]), "haloes_z_le_1": float(per_z1[b1]),
                          "mean_logm200c_all_z": float(mean_all[b1]), "mean_logm200c_z_le_1": float(mean_z1[b1]),
                          "haloes_z_le_1_over_bin_12.9-13.1": float(per_z1[b1] / per_z1[b1 + 1]),
                          "S82_mean_logm200c": float(s82["mean_logm"]),
                          "S82_minus_mean_logm200c_z_le_1": float(s82["mean_logm"] - mean_z1[b1])},
        "columns_below_13_z_le_1": cols,
        "peak_column_z_le_1": {"logm200c_centre": float(lm[peak]), "logm200c_lo": float(lme[peak]),
                               "count_z_le_1": float(c_z1[peak])},
        "completeness_ESTIMATE": {
            "method": "z <= 1 counts per mass column divided by exp(a + b log10 M), the least-squares line in ln N vs "
                      f"log10 M over the {int(fit.sum())} columns with {FIT_RANGE[0]} < log10 M < {FIT_RANGE[1]}, "
                      "extrapolated below 12.9",
            "fit_columns": int(fit.sum()), "slope_dlnN_dlog10M": float(coef[0]),
            "fit_max_abs_ln_residual": float(np.max(np.abs(resid))),
            "first_column_above_95pct": {"logm200c_centre": first_complete["logm200c_centre"],
                                         "logm200c_lo": first_complete["logm200c_lo"],
                                         "completeness": first_complete["completeness_z_le_1_ESTIMATE"]}},
    }
    p = HERE / "post" / "lightcone_mass_z.json"
    p.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")
    print("parity passed:", parity["passed"], "| z<=1 min", out["min_logm200c_painted_z_le_1"], "| bin 12.7-12.9 mean z<=1",
          mean_z1[b1], "all z", mean_all[b1])
    for c in cols:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})


if __name__ == "__main__":
    main()
