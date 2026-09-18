#!/usr/bin/env python3
"""Angular power spectra of the four truncation-comparison maps and the two comparison figures.

Maps: halfdome_{y,ksz}_{projected,sphere}_nside*_r200cx4.fits (same halos, same 4 R200c disc;
projected = previous code, infinite LOS; sphere = gas inside the 4 R200c sphere only).
Theory: CLASS-SZ Battaglia12 tSZ (x_outSZ = 4) and Battaglia16 kSZ (electrons truncated at 4 R200c).
"""
import argparse, json
from pathlib import Path
import numpy as np, healpy as hp, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_CMB_UK = 2.7255e6
COL = {"projected": "#2a78d6", "sphere": "#eb6834", "theory": "#52514e", "theory1h": "#9a9891", "poisson": "#1baf7a"}

def log_bins(lmin, lmax, per_decade):
    edges = np.unique(np.round(np.logspace(np.log10(lmin), np.log10(lmax), int(per_decade * np.log10(lmax / lmin)) + 1)).astype(int))
    return edges

def bin_dl(ell, dl, edges):
    out, centres = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ell >= lo) & (ell < hi)
        w = 2 * ell[sel] + 1.0
        out.append(np.sum(w * dl[sel]) / np.sum(w)); centres.append(np.exp(np.sum(w * np.log(ell[sel])) / np.sum(w)))
    return np.array(centres), np.array(out)

def theory_on_integers(ell_th, dl_th, ell):
    good = dl_th > 0
    return np.exp(np.interp(np.log(ell), np.log(ell_th[good]), np.log(dl_th[good])))

def load_theory(path):
    d = np.load(path); md = json.loads(str(d["metadata_json"]))
    return d["ell"], d["dl_1h"], d["dl_2h"], md

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="maps"); ap.add_argument("--out-dir", default="spectra")
    ap.add_argument("--tag", default=""); ap.add_argument("--lmax", type=int, default=8192)
    ap.add_argument("--class-sz-tsz", default="spectra/class_sz_tsz_x4_reference.npz")
    ap.add_argument("--class-sz-ksz", default="spectra/class_sz_ksz_x4_reference.npz")
    ap.add_argument("--poisson", default=None, help="optional catalogue Poisson-term h5 from compute_catalogue_poisson_terms.jl")
    ap.add_argument("--skip-anafast", action="store_true", help="reuse raw_cl.npz")
    a = ap.parse_args()
    maps_dir, out_dir = Path(a.maps_dir), Path(a.out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    names = ["y_projected", "y_sphere", "ksz_projected", "ksz_sphere"]
    raw_path = out_dir / f"raw_cl{a.tag}.npz"
    if a.skip_anafast and raw_path.exists():
        raw = dict(np.load(raw_path)); nside = int(raw["nside"]); lmax = int(raw["lmax"])
    else:
        raw = {}
        for n in names:
            path = next(maps_dir.glob(f"halfdome_{n}_nside*_r200cx4{a.tag}.fits"))
            m = hp.read_map(path, dtype=np.float64); nside = hp.get_nside(m); lmax = min(a.lmax, 3 * nside - 1)
            print(f"anafast {path.name}: nside={nside} lmax={lmax} mean={m.mean():.3e} rms={m.std():.3e}", flush=True)
            cl = hp.anafast(m, lmax=lmax, iter=0)
            raw[n] = cl / hp.pixwin(nside, lmax=lmax) ** 2   # pixel-window corrected
            raw[n + "_nopixwin"] = cl
            del m
        raw["nside"] = nside; raw["lmax"] = lmax
        np.savez(raw_path, **raw)
    ell = np.arange(lmax + 1, dtype=float)
    dl = {n: ell * (ell + 1) * raw[n] / (2 * np.pi) for n in names}
    edges = log_bins(30, lmax, 20); rows = {}
    for n in names:
        lc, rows[n] = bin_dl(ell[2:], dl[n][2:], edges)
    # theory
    th = {}
    for kind, path in (("y", a.class_sz_tsz), ("ksz", a.class_sz_ksz)):
        e, h1, h2, md = load_theory(path)
        scale = 1e-12 if kind == "y" else 1.0    # class_sz tSZ is 1e12 D_ell; kSZ assumed dimensionless (Delta T/T)^2 D_ell
        _, th[kind + "_1h"] = bin_dl(ell[2:], theory_on_integers(e, h1 * scale, ell[2:]), edges)
        _, th[kind + "_tot"] = bin_dl(ell[2:], theory_on_integers(e, (h1 + h2) * scale, ell[2:]), edges)
    pois = None
    POIS_LMIN = 200   # flat-sky Hankel transform; below this the largest nearby halos (theta_max up to ~1 rad) are not treated correctly
    if a.poisson:
        with h5py.File(a.poisson) as f:
            pe = f["ell"][:]
            pois = {k: bin_dl(ell[2:], theory_on_integers(pe, pe * (pe + 1) * f[f"cl_1h_{k}"][:] / (2 * np.pi), ell[2:]), edges)[1] for k in names}
    # table
    import csv
    with open(out_dir / f"binned_spectra{a.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        head = ["ell_eff", "1e12_Dl_yy_projected", "1e12_Dl_yy_sphere", "1e12_Dl_yy_classsz_1h", "1e12_Dl_yy_classsz_tot",
                "Dl_ksz_projected_uK2", "Dl_ksz_sphere_uK2", "Dl_ksz_classsz_1h_uK2", "Dl_ksz_classsz_tot_uK2",
                "yy_sphere_over_projected", "ksz_sphere_over_projected", "yy_projected_over_classsz", "yy_sphere_over_classsz",
                "ksz_projected_over_classsz", "ksz_sphere_over_classsz"]
        if pois: head += ["1e12_Dl_yy_poisson_projected", "1e12_Dl_yy_poisson_sphere", "Dl_ksz_poisson_projected_uK2", "Dl_ksz_poisson_sphere_uK2"]
        w.writerow(head)
        for i in range(len(lc)):
            row = [lc[i], 1e12 * rows["y_projected"][i], 1e12 * rows["y_sphere"][i], 1e12 * th["y_1h"][i], 1e12 * th["y_tot"][i],
                   T_CMB_UK**2 * rows["ksz_projected"][i], T_CMB_UK**2 * rows["ksz_sphere"][i], T_CMB_UK**2 * th["ksz_1h"][i], T_CMB_UK**2 * th["ksz_tot"][i],
                   rows["y_sphere"][i] / rows["y_projected"][i], rows["ksz_sphere"][i] / rows["ksz_projected"][i],
                   rows["y_projected"][i] / th["y_tot"][i], rows["y_sphere"][i] / th["y_tot"][i],
                   rows["ksz_projected"][i] / th["ksz_tot"][i], rows["ksz_sphere"][i] / th["ksz_tot"][i]]
            if pois: row += [1e12 * pois["y_projected"][i], 1e12 * pois["y_sphere"][i], T_CMB_UK**2 * pois["ksz_projected"][i], T_CMB_UK**2 * pois["ksz_sphere"][i]]
            w.writerow([f"{v:.6g}" for v in row])
    # figures
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "grid.linestyle": "-",
                         "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
    for kind, unit, ylabel, title, ppath in (
        ("y", 1e12, r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/2\pi$", "tSZ Compton-$y$ auto-spectrum, HalfDome halos, Battaglia12 pressure, $4R_{200c}$", "tsz"),
        ("ksz", T_CMB_UK**2, r"$\ell(\ell+1)C_\ell^{\rm kSZ}/2\pi\;[\mu{\rm K}^2]$", "kSZ auto-spectrum, HalfDome halos, Battaglia16 AGN gas density, $4R_{200c}$", "ksz")):
        fig, ax = plt.subplots(3, 1, figsize=(8.2, 10.2), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.35, 1.35], "hspace": 0.06})
        p, s = rows[f"{kind}_projected"], rows[f"{kind}_sphere"]
        ax[0].plot(lc, unit * p, color=COL["projected"], lw=2, label="HalfDome, previous code: projected disc, infinite LOS")
        ax[0].plot(lc, unit * s, color=COL["sphere"], lw=2, label="HalfDome, truncation fix: gas inside the $4R_{200c}$ sphere")
        ax[0].plot(lc, unit * th[f"{kind}_tot"], color=COL["theory"], lw=1.8, ls="--", label="CLASS-SZ 1h+2h, 3-D truncation at $4R_{200c}$")
        ax[0].plot(lc, unit * th[f"{kind}_1h"], color=COL["theory1h"], lw=1.4, ls=":", label="CLASS-SZ 1h only")
        if pois:
            pm = lc >= POIS_LMIN
            ax[0].plot(lc[pm], unit * pois[f"{kind}_projected"][pm], color=COL["projected"], lw=1.1, ls=(0, (1, 2)), alpha=0.9, label=r"catalogue 1-halo (Poisson) term, projected ($\ell\geq200$)")
            ax[0].plot(lc[pm], unit * pois[f"{kind}_sphere"][pm], color=COL["sphere"], lw=1.1, ls=(0, (1, 2)), alpha=0.9, label="catalogue 1-halo (Poisson) term, sphere")
        ax[0].set_yscale("log"); ax[0].set_ylabel(ylabel); ax[0].set_title(title, fontsize=11.5, loc="left")
        ax[0].legend(fontsize=9.2, loc="lower right")
        ax[1].plot(lc, p / th[f"{kind}_tot"], color=COL["projected"], lw=2, label="previous code / CLASS-SZ")
        ax[1].plot(lc, s / th[f"{kind}_tot"], color=COL["sphere"], lw=2, label="truncation fix / CLASS-SZ")
        ax[1].axhline(1, color="#9a9891", lw=0.8); ax[1].set_ylabel("map / CLASS-SZ (1h+2h)"); ax[1].legend(fontsize=9.2, loc="best")
        ax[2].plot(lc, 100 * (s / p - 1), color=COL["sphere"], lw=2, label="map: sphere / projected $-$ 1")
        if pois:
            ax[2].plot(lc[pm], 100 * (pois[f"{kind}_sphere"][pm] / pois[f"{kind}_projected"][pm] - 1), color=COL["poisson"], lw=1.4, ls=(0, (3, 2)), label="catalogue 1-halo term: sphere / projected $-$ 1")
        ax[2].axhline(0, color="#9a9891", lw=0.8); ax[2].set_ylabel("effect of the fix [%]"); ax[2].legend(fontsize=9.2, loc="best")
        ax[2].set_xscale("log"); ax[2].set_xlim(30, lmax); ax[2].set_xlabel(r"multipole $\ell$")
        for x in ax: x.tick_params(which="both", top=True, right=True)
        fig.savefig(out_dir / f"{ppath}_truncation_comparison{a.tag}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / f"{ppath}_truncation_comparison{a.tag}.pdf", bbox_inches="tight"); plt.close(fig)
    # summary at reference multipoles
    summary = {}
    for l0 in (300, 1000, 3000, 6000):
        i = int(np.argmin(np.abs(lc - l0)))
        summary[f"ell~{lc[i]:.0f}"] = {
            "yy_sphere_over_projected": float(rows["y_sphere"][i] / rows["y_projected"][i]),
            "ksz_sphere_over_projected": float(rows["ksz_sphere"][i] / rows["ksz_projected"][i]),
            "yy_projected_over_classsz": float(rows["y_projected"][i] / th["y_tot"][i]), "yy_sphere_over_classsz": float(rows["y_sphere"][i] / th["y_tot"][i]),
            "ksz_projected_over_classsz": float(rows["ksz_projected"][i] / th["ksz_tot"][i]), "ksz_sphere_over_classsz": float(rows["ksz_sphere"][i] / th["ksz_tot"][i]),
            "1e12_Dl_yy_sphere": float(1e12 * rows["y_sphere"][i]), "Dl_ksz_sphere_uK2": float(T_CMB_UK**2 * rows["ksz_sphere"][i]),
            "Dl_ksz_classsz_tot_uK2": float(T_CMB_UK**2 * th["ksz_tot"][i]), "1e12_Dl_yy_classsz_tot": float(1e12 * th["y_tot"][i])}
        if pois:
            summary[f"ell~{lc[i]:.0f}"].update({"ksz_poisson_projected_over_classsz_1h": float(pois["ksz_projected"][i] / th["ksz_1h"][i]),
                                                 "yy_poisson_projected_over_classsz_1h": float(pois["y_projected"][i] / th["y_1h"][i]),
                                                 "ksz_map_projected_over_poisson_projected": float(rows["ksz_projected"][i] / pois["ksz_projected"][i])})
    (out_dir / f"summary{a.tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
