"""HalfDome Compton-y auto-spectra (B12, FLAMINGO-fit painted, Lee22 painted) against FLAMINGO L2p8_m9 lightcones 0 and 1.

Only existing products are used (no anafast, no class_sz run):
  * flamingo_pressure_calibration/halfdome_tsz_comparison/products/comparison_products.npz
      cl_halfdome_b12 (sphere-4R200c B12 map, truncation_comparison/radius_series), cl_halfdome_flamingo (FLAMINGO joint
      fit painted, sphere 4R200c), cl_flamingo_lc0 / lc1 (Yang+26 lensed maps), pixwin; healpy.anafast lmax 8192 iter 0.
  * frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz
      cl_y_b12 (production B12 map, XGPaint projected 4R200c aperture), cl_y_lee22p (Lee22 no-c Table 7, same painter,
      all haloes), cl_y_lee22p_calib (only 0 < z <= 2 and 1e13-10^14.8 h^-1 Msun); map2alm lmax 8192 iter 3; pixel window
      not deconvolved.
  * halfdome_tsz_comparison/class_sz/class_sz_{fl,hd}_{b12,flamingo}.npz: existing CLASS-SZ (Tinker08 M200c, 4R200c,
      HalfDome halo range) spectra, used only for the cosmology factor F_p = D(FLAMINGO cosmology)/D(HalfDome cosmology).
  * outputs/lee22_normalization_forensics_20260928/F_sz_budget/budget_results.json, pressure_corrections['1 (1+z)^2']:
      halo-model ratio D_ell[Lee22 x (1+z)^2] / D_ell[Lee22] at 8 multipoles (HalfDome halo range), used to TRANSFER the
      painted Lee22 spectrum to an estimate for Lee22 x (1+z)^2 (not painted).

Conventions (as in the halfdome_tsz_comparison README): HalfDome maps are point samples -> no pixel-window correction;
FLAMINGO spectra are divided by pixwin^2. Bands: EDGES = unique(round(geomspace(30, 8000, 31))), band mean of D_ell.
Lee22 is carried into the sphere-4 B12 convention of the products by Lee22_equiv = cl_y_lee22p / cl_y_b12 x cl_halfdome_b12
(both B12 spectra are shown; their ratio measures painter + iter differences). The cosmology-shifted Lee22 uses F_B12
(no CLASS-SZ Lee22 run exists; F differs by 1% between B12 and the FLAMINGO fit, so this is approximate).

Outputs: OUT/tsz/tsz_spectra.npz, tsz_ratio_table.csv, tsz_summary.json, fig_tsz_spectra.png.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common_c import OUT, BLACK, BLUE, ORANGE, GREEN, PINK  # noqa: E402

HD = Path("/home/cbllover/HalfDome/flamingo_pressure_calibration/halfdome_tsz_comparison")
FULLSKY = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/tsz_dm_fullsky_20260918/spectra/fullsky_spectra.npz")
BUDGET = Path("/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_normalization_forensics_20260928/F_sz_budget/budget_results.json")
EDGES = np.unique(np.round(np.geomspace(30, 8000, 31)).astype(int))
CENTRES = np.sqrt(EDGES[:-1] * EDGES[1:])
TABLE_ELLS = (100, 300, 1000, 2000, 3000, 5000)


def binned(v):
    return np.array([v[lo:hi].mean() for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def class_sz_on(name):
    d = np.load(HD / "class_sz" / f"class_sz_{name}.npz")
    return np.exp(np.interp(np.log(CENTRES), np.log(d["ell"]), np.log(d["dl_total"])))


def main():
    out = OUT / "tsz"
    out.mkdir(parents=True, exist_ok=True)
    P = np.load(HD / "products" / "comparison_products.npz")
    S = np.load(FULLSKY)
    ell = P["ell"].astype(float)
    assert np.array_equal(P["ell"], S["ell"])
    pw2 = P["pixwin"] ** 2
    dl = lambda cl: ell * (ell + 1) * cl / (2 * np.pi)  # noqa: E731
    B = {"FLAMINGO lc0": binned(dl(P["cl_flamingo_lc0"] / pw2)),
         "FLAMINGO lc1": binned(dl(P["cl_flamingo_lc1"] / pw2)),
         "B12 (sphere 4R200c)": binned(dl(P["cl_halfdome_b12"])),
         "FLAMINGO fit painted": binned(dl(P["cl_halfdome_flamingo"])),
         "B12 production (projected 4R200c)": binned(dl(S["cl_y_b12"])),
         "Lee22 painted (production)": binned(dl(S["cl_y_lee22p"])),
         "Lee22 painted, calibrated range only": binned(dl(S["cl_y_lee22p_calib"]))}
    conv = B["B12 (sphere 4R200c)"] / B["B12 production (projected 4R200c)"]
    B["Lee22 (sphere-4 convention)"] = B["Lee22 painted (production)"] * conv
    B["Lee22 calib (sphere-4 convention)"] = B["Lee22 painted, calibrated range only"] * conv
    # halo-model transfer to Lee22 x (1+z)^2
    res = json.loads(BUDGET.read_text())
    row = [r for r in res["pressure_corrections"] if r["label"] == "1 (1+z)^2"][0]
    ells_hm = np.array([30, 100, 300, 1000, 2000, 3000, 5000, 8000], float)
    r_hm = np.array([row[f"validation_tsz_ratio_to_literal_l{int(l)}"] for l in ells_hm])
    transfer = np.exp(np.interp(np.log(CENTRES), np.log(ells_hm), np.log(r_hm)))
    B["Lee22 x (1+z)^2, halo-model transfer"] = B["Lee22 (sphere-4 convention)"] * transfer
    # cosmology factors (existing CLASS-SZ outputs)
    F_b12 = class_sz_on("fl_b12") / class_sz_on("hd_b12")
    F_fit = class_sz_on("fl_flamingo") / class_sz_on("hd_flamingo")
    F = {"B12 (sphere 4R200c)": F_b12, "FLAMINGO fit painted": F_fit, "Lee22 (sphere-4 convention)": F_b12,
         "Lee22 calib (sphere-4 convention)": F_b12, "Lee22 x (1+z)^2, halo-model transfer": F_b12}
    ref = B["FLAMINGO lc0"]
    lc1 = B["FLAMINGO lc1"] / ref
    table = []
    for l in TABLE_ELLS:
        k = int(np.argmin(np.abs(CENTRES - l)))
        rec = {"ell": l, "band": f"{EDGES[k]}-{EDGES[k + 1]}", "1e12_Dl_FLAMINGO_lc0": 1e12 * ref[k], "lc1_over_lc0": lc1[k],
               "B12_sphere_over_B12_production": conv[k], "F_B12": F_b12[k], "F_FLfit": F_fit[k], "HM_transfer_a2": transfer[k]}
        for name in ("B12 (sphere 4R200c)", "FLAMINGO fit painted", "Lee22 (sphere-4 convention)", "Lee22 calib (sphere-4 convention)",
                     "Lee22 x (1+z)^2, halo-model transfer"):
            rec[f"{name} / lc0, as painted"] = B[name][k] / ref[k]
            rec[f"{name} / lc0, cosmology-shifted"] = B[name][k] * F[name][k] / ref[k]
        rec["Lee22 / B12 (same painter)"] = B["Lee22 painted (production)"][k] / B["B12 production (projected 4R200c)"][k]
        table.append(rec)
    with open(out / "tsz_ratio_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        for r in table:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez(out / "tsz_spectra.npz", edges=EDGES, centres=CENTRES, F_b12=F_b12, F_fit=F_fit, hm_transfer_a2=transfer,
             **{k.replace(" ", "_").replace("/", "_"): v for k, v in B.items()})
    (out / "tsz_summary.json").write_text(json.dumps({"table": table, "hm_transfer_points": dict(zip(ells_hm.astype(int).astype(str).tolist(), r_hm.tolist()))},
                                                     indent=1, default=float) + "\n")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]})
    curves = [("FLAMINGO fit painted", PINK, ":", 2.0, "HalfDome, FLAMINGO 9-coeff. fit painted"),
              ("B12 (sphere 4R200c)", BLUE, "-", 1.8, "HalfDome, Battaglia+12"),
              ("Lee22 (sphere-4 convention)", ORANGE, "-", 2.0, "HalfDome, Lee22 no-c painted (all haloes)"),
              ("Lee22 calib (sphere-4 convention)", ORANGE, "--", 1.4, "HalfDome, Lee22 painted, z<=2 & 1e13-10^14.8/h only"),
              ("Lee22 x (1+z)^2, halo-model transfer", GREEN, "--", 2.0, r"Lee22 $\times(1+z)^2$ (estimate: painted Lee22 $\times$ halo-model ratio)")]
    for col, shifted in enumerate((False, True)):
        ax, axr = axes[0, col], axes[1, col]
        ax.plot(CENTRES, 1e12 * ref, color=BLACK, lw=2.4, label="FLAMINGO L2p8_m9 lightcone 0 (Yang+26, /pixwin$^2$)")
        ax.plot(CENTRES, 1e12 * B["FLAMINGO lc1"], color=BLACK, lw=1.0, ls="-", alpha=0.55, label="FLAMINGO lightcone 1")
        for name, color, ls, lw, label in curves:
            y = B[name] * (F[name] if shifted else 1.0)
            ax.plot(CENTRES, 1e12 * y, color=color, ls=ls, lw=lw, label=label)
            axr.plot(CENTRES, y / ref, color=color, ls=ls, lw=lw)
        axr.fill_between(CENTRES, np.minimum(1, lc1), np.maximum(1, lc1), color="0.8", label="lc1/lc0 spread")
        axr.axhline(1, color=BLACK, lw=1)
        axr.text(0.02, 0.05, "grey band: FLAMINGO lightcone 1 / lightcone 0", transform=axr.transAxes, fontsize=8, color="0.35")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.02, 6)
        ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$")
        ax.set_title("as painted (HalfDome cosmology)" if not shifted else
                     "HalfDome shifted to FLAMINGO cosmology\n(x F$_p$ from existing CLASS-SZ runs; Lee22 curves use F$_{B12}$)",
                     fontsize=10.5)
        axr.set_xscale("log")
        axr.set_yscale("log")
        axr.set_ylim(0.2, 5)
        axr.set_yticks([0.25, 0.5, 1, 2, 4])
        axr.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        axr.set_ylabel("HalfDome / FLAMINGO lc0")
        axr.set_xlabel(r"$\ell$")
        for a in (ax, axr):
            a.grid(alpha=0.3, which="both", lw=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / "fig_tsz_spectra.png", dpi=130)
    plt.close(fig)
    for r in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
