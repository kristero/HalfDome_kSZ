#!/usr/bin/env python
"""Step 1: validate the CAMELS precomputed (illstack) profiles against our direct computation from the raw
snapshots (all directly processed boxes CV_0-CV_3 x snapshots 090/074/060/052/044 = z 0/0.47/1.05/1.48/2.00), and
quantify what the precomputed files cannot capture (electrons from ElectronAbundance, star-forming cells, fixed
ckpc/h binning).

Inputs : direct/direct_CV_<b>_<new>.npz (camels_direct_profiles.py on idark), camels_profiles/CV_<b>/*_<old>.hdf5
Outputs: validation/validation_summary.txt, validation/validation_table.csv (one block / row set per box and
         snapshot), fig_D1_validation.png (z = 0, 1.05, 2.00; halos of all direct boxes pooled)

Run: LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib /home/cbllover/miniconda3/envs/halfdome/bin/python validate_precomputed.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import camels_common as cc  # noqa: E402

FIG_NEW = [90, 60, 44]   # raw snapshots shown in the figure (z = 0, 1.05, 2.00)
VDIR = os.path.join(cc.OUTDIR, "validation")
os.makedirs(VDIR, exist_ok=True)
FILES = sorted(glob.glob(os.path.join(cc.DIRECTDIR, "direct_CV_*_*.npz")))


def load_direct(fn):
    d = np.load(fn)
    return {k: d[k] for k in d.files}


def shell_vol_phys(x_edges, R200_kpc):
    """physical shell volumes [cm^3] for each halo (rows) and x bin (cols)"""
    re = x_edges[None, :] * R200_kpc[:, None] * cc.KPC
    return 4 * np.pi / 3 * (re[:, 1:] ** 3 - re[:, :-1] ** 3)


lines = []
rows = ["box,new_snap,old_snap,z,quantity,mass_selection,nhalo,nbins,median_ratio,p16,p84,max_abs_dev"]
pool = {n: dict(rat={q: [] for q in ("dens", "pth", "tmw")}, num={}, den={}, nh=0, boxes=[], vboxes=[], nmatch=0, maxdev=0.0)
        for n in FIG_NEW}
# CV_2: the raw snapshots/groups now hosted (dated 2023-01-07, Arepo commit of 2021-03-11, double precision, seed
# 3999666 in CosmoAstroSeed_IllustrisTNG_L25n256_CV.txt) are a re-run with a new random seed, whereas the precomputed
# CV_2 profiles (dated 2022-01-03) were made from the original CV_2 run.  The halo catalogues differ, so CV_2 cannot
# validate the precomputed files; its direct profiles are still a valid fiducial-TNG realisation.
REALISATION_MISMATCH = {2}
worst = {}
for fn in FILES:
    box = int(os.path.basename(fn).split("_")[2])
    new = int(os.path.basename(fn).split("_")[3].split(".")[0])
    old = cc.NEW_SNAPS.index(new)
    d = load_direct(fn)
    pc = cc.load_precomputed(box, old)
    z = float(d["z"])
    lines.append(f"=== CV_{box}: old snap {old:03d} (illstack z={cc.OLD_SNAP_Z[old]}) <-> raw snapshot_{new:03d} (header z={z:.5f}); "
                 f"search radius {float(d['search_radius'])} x rhmax={float(d['rhmax']):.1f} ckpc/h")
    # match halos by ID (group index)
    idpos = {int(i): k for k, i in enumerate(d["ID"])}
    both = [(k, idpos[int(i)]) for k, i in enumerate(pc["ID"]) if int(i) in idpos]
    kp = np.array([b[0] for b in both])
    kd = np.array([b[1] for b in both])
    lines.append(f"  precomputed halos {len(pc['ID'])}, direct {len(d['ID'])}, matched {len(both)}")
    # masses: file vs catalogue
    mfile = pc["M200"][kp] * cc.CAMELS.h / 1e10
    lines.append(f"  M200 file/catalogue: max |ratio-1| = {np.max(np.abs(mfile / d['M200_code'][kd] - 1)):.2e}")
    # bin-by-bin comparisons in the SAME illstack binning (code units)
    h, a = float(d["h"]), float(d["a"])
    dens_fac = 1e10 * cc.MSUN / cc.KPC ** 3 * h ** 2 / a ** 3
    comp = {"dens": (pc["rho"][kp] / dens_fac, d["ill_dens"][kd]),
            "pth": (pc["Pth"][kp] / dens_fac / 1e10, d["ill_pth"][kd]),
            "tmw": (pc["Tmw"][kp] / 1e10, d["ill_tmw"][kd]),
            "ncell": (pc["ncell"][kp].astype(float), d["ill_n"][kd])}
    # restrict to bins well inside the search radius and with >= 10 cells
    rs = float(d["search_radius"]) * float(d["rhmax"])
    inner = d["ill_edges"][1:] < rs
    for q, (P, D) in comp.items():
        for msel, mlab in ((np.ones(len(kp), bool), "all"), (pc["M200"][kp] > 10 ** 12.5, "M>1e12.5")):
            m = (D > 0) & (P > 0) & (d["ill_n"][kd] >= 10) & inner[None, :] & msel[:, None]
            if m.sum() == 0:
                continue
            rat = P[m] / D[m]
            p16, p50, p84 = np.percentile(rat, [16, 50, 84])
            rows.append(f"{box},{new},{old},{z:.4f},{q},{mlab},{msel.sum()},{m.sum()},{p50:.6f},{p16:.6f},{p84:.6f},"
                        f"{np.max(np.abs(rat - 1)):.3e}")
            if mlab != "all":
                wk = q + (" (CV_2, different realisation)" if box in REALISATION_MISMATCH else "")
                worst[wk] = max(worst.get(wk, 0.0), float(np.max(np.abs(rat - 1))))
            lines.append(f"  {q:6s} {mlab:9s}: precomputed/direct median {p50:.5f} (16-84%: {p16:.5f}-{p84:.5f}), "
                         f"max|dev| {np.max(np.abs(rat - 1)):.2e} over {m.sum()} bins")
    # outermost populated bin (to determine search radius)
    nz = [np.max(np.nonzero(pc["ncell"][k])[0]) if pc["ncell"][k].any() else -1 for k in kp]
    lines.append(f"  outermost populated precomputed bin: {sorted(set(nz))}; direct: "
                 f"{sorted(set([np.max(np.nonzero(d['ill_n'][k])[0]) for k in kd]))}")

    # --- what the precomputed files miss: electrons (ElectronAbundance) and SF cells, in x = r/R200c bins
    R200k = d["R200_code"] * a / h
    M200 = d["M200_code"] * 1e10 / h
    vol = shell_vol_phys(d["x_edges"], R200k)
    xc = np.sqrt(d["x_edges"][1:] * np.maximum(d["x_edges"][:-1], 1e-3))
    ne_true = d["true_Ne"] / vol
    ne_nosf = (d["true_Ne"] - d["true_Ne_sf"]) / vol
    ne_fullion = d["true_Mgas"] * cc.MSUN / (cc.MU_E * cc.M_P) / vol
    pe_true = d["true_PeV"] / vol
    pe_nosf = (d["true_PeV"] - d["true_PeV_sf"]) / vol
    pe_fullion = cc.PE_OVER_PTH * d["true_PthV"] / vol
    sel = (M200 > 10 ** 12.5)
    nsel = sel.sum()
    lines.append(f"  halos with M200c > 10^12.5 Msun: {nsel}  (max M200c = {M200.max():.3e} Msun)")
    for name, num, den in (("n_e(x_e)/n_e(full ion)", ne_true, ne_fullion), ("n_e(noSF)/n_e(all)", ne_nosf, ne_true),
                           ("P_e(x_e)/(0.5176 P_th)", pe_true, pe_fullion), ("P_e(noSF)/P_e(all)", pe_nosf, pe_true)):
        s = []
        for xq in cc.X_QUOTE:
            k = np.searchsorted(d["x_edges"], xq) - 1
            rr = np.nansum(num[sel, k]) / np.nansum(den[sel, k])
            s.append(f"x={xq}: {rr:.3f}")
        lines.append(f"  stacked (sum over halos) {name:24s} " + ", ".join(s))
    # integrated inside R200c
    for name, num, den in (("N_e(<R200)/(M_gas/mu_e m_p)", d["in200_Ne"], d["in200_Mgas"] * cc.MSUN / (cc.MU_E * cc.M_P)),
                           ("M_gas,SF(<R200)/M_gas", d["in200_Mgas_sf"], d["in200_Mgas"]),
                           ("Y_e(<R200)/(0.5176 Y_th)", d["in200_PeV"], cc.PE_OVER_PTH * d["in200_PthV"]),
                           ("Y_e,SF(<R200)/Y_e", d["in200_PeV_sf"], d["in200_PeV"])):
        lines.append(f"  M>1e12.5 sum {name:30s} = {num[sel].sum() / den[sel].sum():.4f}")

    # pooled figure inputs (all direct boxes) for the figure redshifts
    if new in pool:
        pl = pool[new]
        pl["nh"] += int(nsel)
        pl["boxes"].append(box)
        pl["z"] = z
        pl["rcen"] = d["ill_cen"]
        pl["xc"] = xc
        if box not in REALISATION_MISMATCH:
            pl["vboxes"].append(box)
            pl["nmatch"] += len(both)
            for q in ("dens", "pth", "tmw"):
                P, D = comp[q]
                m = (D > 0) & (P > 0) & (d["ill_n"][kd] >= 10) & inner[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    pl["rat"][q].append(np.where(m, P / np.where(D > 0, D, 1), np.nan))
                mm = m & (pc["M200"][kp] > 10 ** 12.5)[:, None]
                if mm.sum():
                    pl["maxdev"] = max(pl["maxdev"], float(np.max(np.abs(P[mm] / D[mm] - 1))))
        for key, num, den in (("xe", ne_true, ne_fullion), ("sf", ne_nosf, ne_true), ("pe", pe_true, pe_fullion),
                              ("pesf", pe_nosf, pe_true)):
            pl["num"][key] = pl["num"].get(key, 0) + np.nansum(num[sel], 0)
            pl["den"][key] = pl["den"].get(key, 0) + np.nansum(den[sel], 0)

lines.insert(0, "NOTE: CV_2 raw snapshot/groups files hosted now are dated 2023-01-07 (Arepo commit fc13193 of 2021-03-11, "
                "DOUBLEPRECISION, ~4.3 GB) and CosmoAstroSeed_IllustrisTNG_L25n256_CV.txt lists seed 3999666 for CV_2 (CV_0/1/3: "
                "seeds 1/2/4); the precomputed CV_2 profiles (2022-01-03) come from the original CV_2 run, so their halo catalogue "
                "differs from the hosted one and CV_2 does not test the precomputed files (both are fiducial-TNG realisations).")
lines.insert(0, f"SUMMARY: {len(FILES)} (box, snapshot) pairs; worst |precomputed/direct - 1| over all bins with >= 10 cells "
                f"of halos M200c > 10^12.5 Msun: " + ", ".join(f"{q} {v:.1e}" for q, v in worst.items()))
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
for col, new in enumerate(FIG_NEW):
    pl = pool[new]
    if not pl["boxes"]:
        continue
    ax = axs[0, col]
    for q, mk, lab in (("dens", "o", "gas density"), ("pth", "s", "thermal pressure"), ("tmw", "^", "mass-weighted T")):
        rat = np.concatenate(pl["rat"][q], 0)
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(rat, axis=0)
        ax.plot(pl["rcen"], med, mk + "-", color={"dens": "black", "pth": cc.COLORS["b12"], "tmw": cc.COLORS["other"]}[q],
                label=lab, ms=4)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0.9, 1.1)
    ax.set_xlabel("r [ckpc/h] (illstack bins)")
    ax.set_ylabel("precomputed / direct (median over halos)")
    ax.set_title(f"z = {pl['z']:.2f}: {pl['nmatch']} halos of " + ", ".join(f"CV_{b}" for b in pl["vboxes"]), fontsize=10)
    ax.text(0.97, 0.95, f"max |ratio - 1| = {pl['maxdev']:.1e}\n(M200c > 1e12.5 Msun, bins >= 10 cells)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8)
    ax.legend(fontsize=8, loc="lower left")
    ax = axs[1, col]
    for key, c, lab in (("xe", "black", r"$n_e(x_e)/n_e$(full ion.)"), ("sf", cc.COLORS["other"], r"$n_e$(no SF)/$n_e$(all)"),
                        ("pe", cc.COLORS["b12"], r"$P_e(x_e)/(0.5176\,P_{th})$"),
                        ("pesf", cc.COLORS["lee_corr"], r"$P_e$(no SF)/$P_e$(all)")):
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = pl["num"][key] / pl["den"][key]
        ax.plot(pl["xc"][1:], rr[1:], "-", color=c, label=lab)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.2)
    ax.set_xlim(0.01, 5)
    ax.set_xlabel(r"$x = r/R_{200c}$")
    ax.set_ylabel(f"stacked ratio, {pl['nh']} halos $M_{{200c}}>10^{{12.5}}M_\\odot$")
    ax.set_title(f"direct only, CV_{min(pl['boxes'])}-CV_{max(pl['boxes'])}", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle("CAMELS TNG CV. Top: precomputed illstack profiles / our re-computation from the raw snapshots (all halos "
             "M200c > 1e11 Msun/h; CV_2 excluded,\nits hosted raw data are a 2023 re-run with a new seed). "
             "Bottom: what the gas-only precomputed files miss (electrons from ElectronAbundance, star-forming cells)",
             fontsize=10.5)
fig.savefig(os.path.join(cc.OUTDIR, "fig_D1_validation.png"), dpi=130)
with open(os.path.join(VDIR, "validation_summary.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
with open(os.path.join(VDIR, "validation_table.csv"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(lines))
