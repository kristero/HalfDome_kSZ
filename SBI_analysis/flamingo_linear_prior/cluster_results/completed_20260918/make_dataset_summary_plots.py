#!/usr/bin/env python3
"""Summary figures for the completed 8192-row linear-prior HalfDome tSZ dataset (idark run halfdome_flamingo_linear_8192_20260915)."""
import glob, json, re, csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
d = {Path(p).stem: np.load(p) for p in glob.glob(str(root / "dataset/*.npy"))}
theta, clean, noisy, unmasked = d["theta"], d["masked_clean_dl40"], d["masked_noisy_cross_dl40"], d["unmasked_clean_dl40"]
edges = d["bin_edges"]; lc = 0.5 * (edges[:-1] + edges[1:])
manifest = json.load(open(root / "manifest.json")); names = manifest["parameter_order"]; lo = manifest["prior"]["lower"]; hi = manifest["prior"]["upper"]
obs = {n: np.load(root / f"observations/{n}.npz") for n in ("HalfDome", "L1_m9", "fgas-8sigma", "Mstar-1sigma")}
fits = {r["parameter"]: r for r in csv.DictReader(open(root / "plots/prior_and_fits.csv"))}
secs = []
for p in glob.glob(str(root / "logs/*.OU")):
    for line in open(p, errors="ignore"):
        m = re.match(r"Completed row_(\d+) seconds ([\d.]+)", line)
        if m: secs.append(float(m.group(2)))
secs = np.array(secs)
COL = {"HalfDome": "#52514e", "L1_m9": "#eb6834", "fgas-8sigma": "#1baf7a", "Mstar-1sigma": "#e87ba4", "band": "#2a78d6"}
LAB = {"HalfDome": "HalfDome Battaglia12 reference", "L1_m9": "FLAMINGO L1_m9", "fgas-8sigma": "FLAMINGO fgas-8$\\sigma$", "Mstar-1sigma": "FLAMINGO M*-1$\\sigma$"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
                     "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
S = 1e12
# ---------------- Figure 1: spectra ----------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
q = np.percentile(clean, [0, 2.5, 16, 50, 84, 97.5, 100], axis=0)
ax[0].fill_between(lc, S * q[0], S * q[6], color=COL["band"], alpha=0.12, lw=0, label="design: full range (8192 rows)")
ax[0].fill_between(lc, S * q[1], S * q[5], color=COL["band"], alpha=0.25, lw=0, label="design: 2.5–97.5%")
ax[0].fill_between(lc, S * q[2], S * q[4], color=COL["band"], alpha=0.45, lw=0, label="design: 16–84%")
ax[0].plot(lc, S * q[3], color=COL["band"], lw=1.8, label="design median")
for n, o in obs.items(): ax[0].plot(lc, S * o["masked_clean_dl40"], color=COL[n], lw=1.8, ls="--" if n == "HalfDome" else "-", label=LAB[n])
ax[0].set_yscale("log"); ax[0].set_ylabel(r"$10^{12}\,D_\ell^{yy}$, masked, beam-smoothed, clean"); ax[0].set_title("Clean masked spectra: training design vs targets", loc="left", fontsize=11)
ax[0].legend(fontsize=8.5, loc="lower right")
# quantile of each observation within the design per bin
for n, o in obs.items():
    frac = (clean < o["masked_clean_dl40"][None, :]).mean(axis=0)
    ax[1].plot(lc, frac, color=COL[n], lw=1.8, ls="--" if n == "HalfDome" else "-", marker="o", ms=3.5, label=LAB[n])
for y in (0.025, 0.975): ax[1].axhline(y, color="#9a9891", lw=0.8, ls=":")
ax[1].set_ylim(-0.02, 1.02); ax[1].set_ylabel("fraction of design rows below the target"); ax[1].set_title("Where each target sits inside the design (per bin)", loc="left", fontsize=11)
ax[1].legend(fontsize=8.5, loc="best")
qn = np.percentile(noisy, [2.5, 16, 50, 84, 97.5], axis=0)
ax[2].fill_between(lc, S * qn[0], S * qn[4], color=COL["band"], alpha=0.25, lw=0, label="noisy cross: 2.5–97.5%")
ax[2].fill_between(lc, S * qn[1], S * qn[3], color=COL["band"], alpha=0.45, lw=0, label="noisy cross: 16–84%")
ax[2].plot(lc, S * qn[2], color=COL["band"], lw=1.8, label="noisy cross median")
for n, o in obs.items(): ax[2].plot(lc, S * o["masked_noisy_cross_dl40"], color=COL[n], lw=1.6, ls="--" if n == "HalfDome" else "-", label=LAB[n] + " (noisy)")
ax[2].set_yscale("symlog", linthresh=1e-3); ax[2].set_ylabel(r"$10^{12}\,D_\ell$, masked noisy split-cross (SO baseline)")
ax[2].set_title(f"Noisy cross spectra ({100*(noisy<0).mean():.1f}% of bins negative)", loc="left", fontsize=11); ax[2].legend(fontsize=8.5, loc="upper left")
for a in ax: a.set_xlabel(r"multipole $\ell$ (40 linear bins, 80–7980)"); a.set_xlim(80, 7980); a.tick_params(which="both", top=True, right=True)
fig.suptitle("HalfDome tSZ training set, broad linear prior, 8192 rows, complete 2026-09-18", fontsize=12.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(root / "summary_spectra.png", dpi=190); fig.savefig(root / "summary_spectra.pdf"); plt.close(fig)
# ---------------- Figure 2: design marginals and runtime ----------------
fig, axes = plt.subplots(3, 4, figsize=(17, 11)); axes = axes.ravel()
labels = {"P0": r"$P_0$", "xc": r"$x_c$", "beta": r"$\beta$", "alpha_m_P0": r"$\alpha_m(P_0)$", "alpha_m_xc": r"$\alpha_m(x_c)$", "alpha_m_beta": r"$\alpha_m(\beta)$",
          "alpha_z_P0": r"$\alpha_z(P_0)$", "alpha_z_xc": r"$\alpha_z(x_c)$", "alpha_z_beta": r"$\alpha_z(\beta)$"}
for i, n in enumerate(names):
    a = axes[i]; a.hist(theta[:, i], bins=40, range=(lo[i], hi[i]), color=COL["band"], alpha=0.75, edgecolor="white", lw=0.4)
    f = fits[n]
    a.axvspan(float(f["old_sbi_lower"]), float(f["old_sbi_upper"]), color="#9a9891", alpha=0.15, lw=0, label="old SBI box" if i == 0 else None)
    for key, c in (("Battaglia12", COL["HalfDome"]), ("L1_m9", COL["L1_m9"]), ("fgas-8sigma", COL["fgas-8sigma"]), ("Mstar-1sigma", COL["Mstar-1sigma"])):
        a.axvline(float(f[key]), color=c, lw=1.6, ls="--" if key == "Battaglia12" else "-", label=(LAB["HalfDome"] if key == "Battaglia12" else LAB[key]) if i == 0 else None)
    a.set_xlim(lo[i], hi[i]); a.set_xlabel(labels[n]); a.set_ylabel("rows" if i % 4 == 0 else "")
axes[0].legend(fontsize=8, loc="upper right")
a = axes[9]; a.hist(secs / 60, bins=40, color="#eda100", alpha=0.8, edgecolor="white", lw=0.4); a.axvline(np.median(secs) / 60, color="#52514e", lw=1.2)
a.set_xlabel("wall time per map [min]"); a.set_ylabel("rows"); a.set_title(f"{secs.size} generated maps\nmedian {np.median(secs)/60:.1f} min, {secs.sum()/3600:.0f} worker-h", fontsize=9.5, loc="left")
a = axes[10]; ell_i = np.argmin(np.abs(lc - 3000)); amp = S * clean[:, ell_i]
sc = a.scatter(theta[:, 0], theta[:, 2], c=np.log10(amp), s=4, cmap="Blues", vmin=np.percentile(np.log10(amp), 1), vmax=np.percentile(np.log10(amp), 99))
a.set_xlabel(labels["P0"]); a.set_ylabel(labels["beta"]); a.set_title("colour: log10 of clean\n" + r"$10^{12}D^{yy}_{\ell\approx3000}$", fontsize=9.5, loc="left"); a.grid(False)
cb = fig.colorbar(sc, ax=a, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=8)
a = axes[11]; a.axis("off")
nval = int(d["validation_split"].sum()); nold = int((d["source_old_row"] >= 0).sum())
txt = (f"rows: {theta.shape[0]}  (validation split {nval}, training {theta.shape[0]-nval})\n"
       f"design points from the retired log-uniform run: {nold}\n  (23 with imported spectra, {nold-23} regenerated)\n"
       f"fresh Sobol proposals: {int((d['source_old_row']<0).sum())}\n"
       f"generated maps in logs: {secs.size}; median {np.median(secs):.0f} s, max {secs.max():.0f} s\n"
       f"total compute: {secs.sum()/3600:.0f} worker-hours ({26*secs.sum()/3600:.0f} core-hours)\n"
       f"noisy bins < 0: {100*(noisy<0).mean():.2f}%;  all clean bins > 0: {bool((clean>0).all())}\n"
       f"clean masked D_ell at ell~3000: median {np.median(amp):.3f}, range {amp.min():.2e}–{amp.max():.2e} (x1e-12)\n"
       f"targets at ell~3000 (x1e-12): " + ", ".join(f"{n} {S*o['masked_clean_dl40'][ell_i]:.3f}" for n, o in obs.items()))
a.text(0.0, 0.98, txt, va="top", ha="left", fontsize=9.6, family="monospace", transform=a.transAxes)
fig.suptitle("Accepted design marginals (linear-uniform base on the joint support), reference fits, and run statistics", fontsize=12.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(root / "summary_design_and_run.png", dpi=170); fig.savefig(root / "summary_design_and_run.pdf"); plt.close(fig)
summary = {"rows": int(theta.shape[0]), "validation_rows": nval, "reused_design_points": nold, "imported_spectra": 23, "fresh_points": int((d["source_old_row"] < 0).sum()),
           "generated_maps_in_logs": int(secs.size), "median_seconds": float(np.median(secs)), "max_seconds": float(secs.max()), "worker_hours": float(secs.sum() / 3600),
           "negative_noisy_bin_fraction": float((noisy < 0).mean()),
           "target_quantile_in_design_ell3000": {n: float((clean[:, ell_i] < o["masked_clean_dl40"][ell_i]).mean()) for n, o in obs.items()},
           "target_quantile_range_over_bins": {n: [float(v) for v in np.percentile((clean < o["masked_clean_dl40"][None, :]).mean(0), [0, 100])] for n, o in obs.items()}}
json.dump(summary, open(root / "summary.json", "w"), indent=1); print(json.dumps(summary, indent=1))
