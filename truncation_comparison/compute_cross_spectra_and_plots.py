#!/usr/bin/env python3
"""tSZ (Battaglia12 y) x halo-DM cross-spectra: Lee22 no-c vs Battaglia16 DM, projected vs spherical truncation."""
import argparse, json, csv
from pathlib import Path
import numpy as np, healpy as hp, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"lee22": "#eb6834", "b16": "#2a78d6", "y": "#52514e", "poisson": "#1baf7a"}
MAPS = {"y_projected": "y_projected", "y_sphere": "y_sphere", "dm_b16_projected": "dm_b16_projected", "dm_b16_sphere": "dm_b16_sphere",
        "dm_lee22noc_projected": "dm_lee22noc_projected", "dm_lee22noc_sphere": "dm_lee22noc_sphere"}
PAIRS = {"y_x_b16_projected": ("y_projected", "dm_b16_projected"), "y_x_b16_sphere": ("y_sphere", "dm_b16_sphere"),
         "y_x_lee22_projected": ("y_projected", "dm_lee22noc_projected"), "y_x_lee22_sphere": ("y_sphere", "dm_lee22noc_sphere")}

def log_bins(lmin, lmax, per_decade):
    return np.unique(np.round(np.logspace(np.log10(lmin), np.log10(lmax), int(per_decade * np.log10(lmax / lmin)) + 1)).astype(int))

def bin_dl(ell, dl, edges):
    out, centres = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi); w = 2 * ell[sel] + 1.0
        out.append(np.sum(w * dl[sel]) / np.sum(w)); centres.append(np.exp(np.sum(w * np.log(ell[sel])) / np.sum(w)))
    return np.array(centres), np.array(out)

def interp_loglog(x_th, y_th, x):
    good = y_th > 0
    return np.exp(np.interp(np.log(x), np.log(x_th[good]), np.log(y_th[good])))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="maps"); ap.add_argument("--out-dir", default="spectra"); ap.add_argument("--tag", default="")
    ap.add_argument("--lmax", type=int, default=8192); ap.add_argument("--poisson", default=None); ap.add_argument("--skip-alm", action="store_true")
    a = ap.parse_args()
    maps_dir, out_dir = Path(a.maps_dir), Path(a.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    raw_path = out_dir / f"raw_cross_cl{a.tag}.npz"
    if a.skip_alm and raw_path.exists():
        raw = dict(np.load(raw_path)); nside = int(raw["nside"]); lmax = int(raw["lmax"])
    else:
        alms = {}; raw = {}
        for key, stem in MAPS.items():
            path = next(maps_dir.glob(f"halfdome_{stem}_nside*_r200cx4{a.tag}.fits"))
            m = hp.read_map(path, dtype=np.float64); nside = hp.get_nside(m); lmax = min(a.lmax, 3 * nside - 1)
            print(f"map2alm {path.name}: mean={m.mean():.4e} rms={m.std():.4e}", flush=True)
            alms[key] = hp.map2alm(m, lmax=lmax, iter=0); del m
        pw2 = hp.pixwin(nside, lmax=lmax) ** 2
        for key in MAPS: raw[key] = hp.alm2cl(alms[key]) / pw2
        for key, (k1, k2) in PAIRS.items(): raw[key] = hp.alm2cl(alms[k1], alms[k2]) / pw2
        raw["nside"] = nside; raw["lmax"] = lmax; np.savez(raw_path, **raw)
    ell = np.arange(lmax + 1, dtype=float); dlf = ell * (ell + 1) / (2 * np.pi)
    edges = log_bins(30, lmax, 20); b = {}
    for key in list(MAPS) + list(PAIRS): lc, b[key] = bin_dl(ell[2:], (dlf * raw[key])[2:], edges)
    pois = None
    if a.poisson:
        pois = {}
        with h5py.File(a.poisson) as f:
            pe = f["ell"][:]; pf = pe * (pe + 1) / (2 * np.pi)
            for key, (k1, k2) in PAIRS.items():
                pois[key] = bin_dl(ell[2:], interp_loglog(pe, pf * f[f"cl_1h_{k1}_x_{k2}"][:], ell[2:]), edges)[1]
            for key in MAPS: pois[key] = bin_dl(ell[2:], interp_loglog(pe, pf * f[f"cl_1h_{key}_x_{key}"][:], ell[2:]), edges)[1]
    pm = lc >= 200
    # correlation coefficients
    r = {t: b[f"y_x_{p}_{t}"] / np.sqrt(b[f"y_{t}"] * b[f"dm_{'lee22noc' if p == 'lee22' else 'b16'}_{t}"]) for p in ("lee22", "b16") for t in ("projected", "sphere")}
    r = {f"{p}_{t}": b[f"y_x_{p}_{t}"] / np.sqrt(b[f"y_{t}"] * b[f"dm_{'lee22noc' if p == 'lee22' else 'b16'}_{t}"]) for p in ("lee22", "b16") for t in ("projected", "sphere")}
    with open(out_dir / f"binned_cross_spectra{a.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["ell_eff"] + [f"Dl_{k}" for k in list(PAIRS) + list(MAPS)] + [f"r_{k}" for k in r] + (["Dl_poisson_" + k for k in PAIRS] if pois else [])
        w.writerow(head)
        for i in range(len(lc)):
            row = [lc[i]] + [b[k][i] for k in list(PAIRS) + list(MAPS)] + [r[k][i] for k in r] + ([pois[k][i] for k in PAIRS] if pois else [])
            w.writerow([f"{v:.6g}" for v in row])
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
    # Figure 1: y x DM cross-spectra
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 10.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.4, 1.4], "hspace": 0.06})
    for p, lab in (("lee22", "Lee22 no-c DM"), ("b16", "Battaglia16 DM")):
        ax[0].plot(lc, b[f"y_x_{p}_sphere"], color=COL[p], lw=2.1, label=f"$y \\times$ {lab}, truncation fix (sphere)")
        ax[0].plot(lc, b[f"y_x_{p}_projected"], color=COL[p], lw=1.5, ls="--", alpha=0.85, label=f"$y \\times$ {lab}, previous code (projected)")
        if pois: ax[0].plot(lc[pm], pois[f"y_x_{p}_sphere"][pm], color=COL[p], lw=1.0, ls=(0, (1, 2)), label=f"catalogue 1-halo term, sphere ($\\ell\\geq200$)" if p == "lee22" else None)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell^{y\times{\rm DM}}/2\pi\;[{\rm pc\,cm^{-3}}]$")
    ax[0].set_title(r"tSZ $\times$ halo DM: Battaglia12 $y$ $\times$ Lee22 no-c vs Battaglia16 DM, HalfDome halos, $4R_{200c}$", fontsize=11, loc="left")
    ax[0].legend(fontsize=9, loc="lower right")
    ax[1].plot(lc, b["y_x_lee22_sphere"] / b["y_x_b16_sphere"], color=COL["lee22"], lw=2.1, label="sphere")
    ax[1].plot(lc, b["y_x_lee22_projected"] / b["y_x_b16_projected"], color=COL["lee22"], lw=1.5, ls="--", alpha=0.85, label="projected")
    ax[1].axhline(1, color="#9a9891", lw=0.8); ax[1].set_ylabel(r"$y\times$Lee22 / $y\times$B16"); ax[1].legend(fontsize=9.2, loc="best")
    for p, lab in (("lee22", "Lee22 no-c"), ("b16", "Battaglia16")):
        ax[2].plot(lc, 100 * (b[f"y_x_{p}_sphere"] / b[f"y_x_{p}_projected"] - 1), color=COL[p], lw=2, label=f"$y\\times${lab}: sphere / projected $-$ 1")
    ax[2].plot(lc, 100 * (b["y_sphere"] / b["y_projected"] - 1), color=COL["y"], lw=1.2, ls=":", label="$y$ auto, for reference")
    ax[2].axhline(0, color="#9a9891", lw=0.8); ax[2].set_ylabel("effect of the fix [%]"); ax[2].legend(fontsize=9.2, loc="best")
    ax[2].set_xscale("log"); ax[2].set_xlim(30, lmax); ax[2].set_xlabel(r"multipole $\ell$")
    for x in ax: x.tick_params(which="both", top=True, right=True)
    fig.savefig(out_dir / f"tsz_x_dm_lee22_vs_b16{a.tag}.png", dpi=200, bbox_inches="tight"); fig.savefig(out_dir / f"tsz_x_dm_lee22_vs_b16{a.tag}.pdf", bbox_inches="tight"); plt.close(fig)
    # Figure 2: DM auto-spectra and correlation coefficient
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 10.6), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.4, 1.4], "hspace": 0.06})
    for p, key, lab in (("lee22", "dm_lee22noc", "Lee22 no-c"), ("b16", "dm_b16", "Battaglia16")):
        ax[0].plot(lc, b[f"{key}_sphere"], color=COL[p], lw=2.1, label=f"{lab} DM, truncation fix (sphere)")
        ax[0].plot(lc, b[f"{key}_projected"], color=COL[p], lw=1.5, ls="--", alpha=0.85, label=f"{lab} DM, previous code (projected)")
        if pois: ax[0].plot(lc[pm], pois[f"{key}_sphere"][pm], color=COL[p], lw=1.0, ls=(0, (1, 2)), label="catalogue 1-halo term, sphere ($\\ell\\geq200$)" if p == "lee22" else None)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\ell(\ell+1)C_\ell^{\rm DM\,DM}/2\pi\;[({\rm pc\,cm^{-3}})^2]$")
    ax[0].set_title(r"Halo-DM auto-spectra and correlation with $y$: Lee22 no-c vs Battaglia16, $4R_{200c}$", fontsize=11, loc="left"); ax[0].legend(fontsize=9, loc="lower right")
    for p, lab in (("lee22", "Lee22 no-c"), ("b16", "Battaglia16")):
        ax[1].plot(lc, r[f"{p}_sphere"], color=COL[p], lw=2.1, label=f"{lab}, sphere"); ax[1].plot(lc, r[f"{p}_projected"], color=COL[p], lw=1.5, ls="--", alpha=0.85, label=f"{lab}, projected")
    ax[1].set_ylabel(r"$r_\ell = C^{y\times{\rm DM}}/\sqrt{C^{yy}C^{\rm DM\,DM}}$"); ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=9, loc="best")
    for p, key, lab in (("lee22", "dm_lee22noc", "Lee22 no-c"), ("b16", "dm_b16", "Battaglia16")):
        ax[2].plot(lc, 100 * (b[f"{key}_sphere"] / b[f"{key}_projected"] - 1), color=COL[p], lw=2, label=f"{lab} DM auto: sphere / projected $-$ 1")
    ax[2].axhline(0, color="#9a9891", lw=0.8); ax[2].set_ylabel("effect of the fix [%]"); ax[2].legend(fontsize=9.2, loc="best")
    ax[2].set_xscale("log"); ax[2].set_xlim(30, lmax); ax[2].set_xlabel(r"multipole $\ell$")
    for x in ax: x.tick_params(which="both", top=True, right=True)
    fig.savefig(out_dir / f"dm_auto_lee22_vs_b16{a.tag}.png", dpi=200, bbox_inches="tight"); fig.savefig(out_dir / f"dm_auto_lee22_vs_b16{a.tag}.pdf", bbox_inches="tight"); plt.close(fig)
    summary = {}
    for l0 in (300, 1000, 3000, 6000):
        i = int(np.argmin(np.abs(lc - l0))); s = {}
        for p in ("lee22", "b16"):
            s[f"y_x_{p}_sphere_over_projected"] = float(b[f"y_x_{p}_sphere"][i] / b[f"y_x_{p}_projected"][i])
            s[f"Dl_y_x_{p}_sphere_pc_cm3"] = float(b[f"y_x_{p}_sphere"][i]); s[f"r_{p}_sphere"] = float(r[f"{p}_sphere"][i])
            key = "dm_lee22noc" if p == "lee22" else "dm_b16"; s[f"{key}_auto_sphere_over_projected"] = float(b[f"{key}_sphere"][i] / b[f"{key}_projected"][i])
        s["y_x_lee22_over_y_x_b16_sphere"] = float(b["y_x_lee22_sphere"][i] / b["y_x_b16_sphere"][i]); s["y_x_lee22_over_y_x_b16_projected"] = float(b["y_x_lee22_projected"][i] / b["y_x_b16_projected"][i])
        if pois: s["map_over_poisson_y_x_lee22_sphere"] = float(b["y_x_lee22_sphere"][i] / pois["y_x_lee22_sphere"][i]); s["map_over_poisson_y_x_b16_sphere"] = float(b["y_x_b16_sphere"][i] / pois["y_x_b16_sphere"][i])
        summary[f"ell~{lc[i]:.0f}"] = s
    (out_dir / f"summary_cross{a.tag}.json").write_text(json.dumps(summary, indent=1)); print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
