"""FLAMINGO L2p8_m9 electron- and gas-density profiles and gas fractions against Battaglia+16 and Lee22 (literal,
x0.6016 production, x C(z) TNG-calibrated; M1 code-level hypothesis x (1+z)^3 in the trend table).

Data: idark products of density_products.py (job 600051.idark, 2026-09-29) for exactly the haloes and shells of the
preliminary pressure stacks (same completeness, same resolution cut, same bootstrap draws as pressure_comparison.py),
plus SOAP SO/200_crit/GasMass of every flagged central in the catalogue caches (notebook_cache/L2p8_m9_NNNN_reduced_haloes.npz).

Definitions (physical units throughout):
  n_e/n200 with n200 = 200 rho_cr f_b/(X_H m_p), X_H = 0.76 (Lee22), f_b = 0.15882 (FLAMINGO), 200 rho_cr = M200c/(4/3 pi R200c^3).
  FLAMINGO n_e: shell sum of N_e,i = n_e,i m_i/rho_i (SWIFT ElectronNumberDensities, cooling-table ionisation, all gas;
  variant 'nosf' drops star-forming particles) divided by the (R100c-truncated) shell volume.
  rho_gas/(200 f_b rho_cr): gas-mass density in units of the cosmic-baryon mean inside R200c.
  Electron fraction f_e = N_e(<R200c) / [f_b M200c/(mu_e m_p)], mu_e = 2/(1+X_H) = 1.136 (fully ionised, X_H = 0.76);
  for a Lee22-form model f_e = 3 (mu_e/X_H) int_0^1 nbar x^2 dx. Gas fraction f_gas = M_gas(<R200c)/(f_b M200c).
  kT_e/kT200 = (sum n_e kT V / sum N_e) / kT200, kT200 = G M200c mu m_p/(2 R200c), mu = 0.588.

Outputs in OUT/density/: density_metrics.csv, density_trends.csv, gas_fraction_soap.csv, density_summary.json and figures.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.constants import G, m_p, k as K_B  # noqa: E402

from common_c import (OUT, STACKS, SNAPSHOTS, FPC, DENSITY_MODELS, PRESSURE_MODELS, fs, L, shell_model_mean,  # noqa: E402
                      interp_log, slope_fit, BLACK, BLUE, ORANGE, GREEN, VERMILLION, PINK, H_SIM, FB_SIM, c_tng)

X_H = 0.76
MU_E = 2 / (1 + X_H)
MU = 4 / (3 + 5 * X_H)
MSUN = 1.98841e30
MPC = 3.085677581491367e22
XQ = np.array([0.1, 0.3, 0.5, 1.0])
DENS = OUT / "idark_density" / "products"


def load():
    products = {s: fs.load_products(STACKS / f"L2p8_m9_{s:04d}_stack_products.npz") for s in SNAPSHOTS}
    stacks = fs.build_stacks(list(products.values()), use_agn_filter=False, n_boot=200)
    dens = {s: dict(np.load(DENS / f"L2p8_m9_{s:04d}_density_products.npz")) for s in SNAPSHOTS}
    return products, stacks, dens


def per_halo_density(d, rows, st):
    """Per-halo shell n_e/n200 (all, nosf), rho_gas/(200 f_b rho_cr), kT_e/kT200 on the stack's truncated shells."""
    edges = st["edges"]
    r200 = d["r200_mpc"][rows] * MPC
    m200 = d["m200c_msun"][rows] * MSUN
    rho200 = m200 / (4 / 3 * np.pi * r200**3)                     # 200 rho_cr(z), physical kg m^-3
    n200 = rho200 * FB_SIM / (X_H * m_p)
    x_in = edges[None, :-1]
    x_out = st["halo_x_out"]                                       # NaN where incomplete
    vol = 4 / 3 * np.pi * (x_out**3 - x_in**3) * r200[:, None] ** 3
    with np.errstate(invalid="ignore", divide="ignore"):
        ne = d["ne3"][rows] / vol / n200[:, None]
        ne_nosf = d["ne3_nosf"][rows] / vol / n200[:, None]
        rho = d["m3"][rows] * MSUN / vol / (rho200[:, None] * FB_SIM)
        kt200 = G * m200 * MU * m_p / (2 * r200)
        te = d["pe3"][rows] / d["ne3"][rows] / kt200[:, None]
    f_gas = d["m_r200"][rows] * MSUN / (FB_SIM * m200)
    f_e = d["ne_r200"][rows] * MU_E * m_p / (FB_SIM * m200)
    return {"ne": ne, "ne_nosf": ne_nosf, "rho": rho, "te": te, "f_gas": f_gas, "f_e": f_e, "sf_frac": d["m_r200_sf"][rows] / d["m_r200"][rows]}


def stack_mean(values, complete, counts=None):
    w = complete.astype(float) if counts is None else complete * counts[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w.sum(0) > 0, (w * np.nan_to_num(values)).sum(0) / w.sum(0), np.nan)


def model_fe(func, masses, z):
    """f_e = 3 (mu_e/X_H) int_0^1 nbar x^2 dx per halo, halo mean."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    lnx = np.log(1e-4) * (1 - (nodes + 1) / 2)
    x = np.exp(lnx)
    jac = -np.log(1e-4) / 2 * weights * x**3
    vals = func(x[None, :], masses[:, None], z)
    return float(np.mean(3 * MU_E / X_H * np.sum(vals * jac[None, :], axis=1)))


def b16_fgas(masses, z):
    nodes, weights = np.polynomial.legendre.leggauss(64)
    lnx = np.log(1e-4) * (1 - (nodes + 1) / 2)
    x = np.exp(lnx)
    jac = -np.log(1e-4) / 2 * weights * x**3
    vals = L.b16_rho_over_fb_rhocrit(x[None, :], masses[:, None], z) / 200.0
    return float(np.mean(3 * np.sum(vals * jac[None, :], axis=1)))


def rho_b16(x, m, z):
    return L.b16_rho_over_fb_rhocrit(x, m, z) / 200.0


def analyse():
    products, stacks, dens = load()
    rows_out, per_stack = [], []
    for st in stacks:
        d, pr = dens[st["snapshot"]], products[st["snapshot"]]
        rows = np.flatnonzero(pr["bin"] == st["bin"])
        assert np.allclose(pr["cat_m200c_msun"][rows], st["masses"]) and np.allclose(d["m200c_msun"][rows], st["masses"])
        ph = per_halo_density(d, rows, st)
        comp = st["complete"]
        prof = {k: stack_mean(ph[k], comp) for k in ("ne", "ne_nosf", "rho", "te")}
        boot = {k: np.array([stack_mean(ph[k], comp, c) for c in st["boot_counts"]]) for k in ("ne", "rho")}
        err = {k: np.nanstd(boot[k], axis=0) for k in boot}
        ok = st["fit"] & np.isfinite(prof["ne"]) & (prof["ne"] > 0)
        outer = np.where(comp, st["halo_x_out"], np.nan)
        models = {name: shell_model_mean(func, st["masses"], st["z"], st["edges"][:-1], outer, comp)
                  for name, (func, *_r) in DENSITY_MODELS.items()}
        models["rho_B16"] = shell_model_mean(rho_b16, st["masses"], st["z"], st["edges"][:-1], outer, comp)
        # Lee22-implied temperature (printed P over printed n; volume conventions cancel), same shells
        p_lee = shell_model_mean(PRESSURE_MODELS["Lee22"][0], st["masses"], st["z"], st["edges"][:-1], outer, comp)
        p_b12 = shell_model_mean(PRESSURE_MODELS["B12"][0], st["masses"], st["z"], st["edges"][:-1], outer, comp)
        te_lee = X_H / MU * p_lee / models["Lee22_literal"]
        te_b = X_H / MU * p_b12 / models["B16"]
        e = {"snapshot": st["snapshot"], "z": round(st["z"], 3), "bin": st["bin"], "log10_m_lo": st["log10_m_lo"],
             "log10_m_hi": st["log10_m_hi"], "log10_m_median": round(st["log10_m_median"], 3), "n_haloes": st["n_haloes"],
             "x_resolved_min": float(st["x_mid"][ok].min()) if ok.any() else np.nan}
        q = interp_log(XQ, st["x_mid"], prof["ne"], ok)
        qn = interp_log(XQ, st["x_mid"], prof["ne_nosf"], ok)
        qr = interp_log(XQ, st["x_mid"], prof["rho"], ok)
        qt = interp_log(XQ, st["x_mid"], prof["te"], ok)
        rel = np.interp(np.log(XQ), np.log(st["x_mid"][ok]), (err["ne"] / prof["ne"])[ok]) if ok.sum() >= 2 else np.full(4, np.nan)
        for i, xq in enumerate(XQ):
            e[f"ne_FL_x{xq:g}"] = q[i]
            e[f"ne_FL_relerr_x{xq:g}"] = rel[i] if np.isfinite(q[i]) else np.nan
            e[f"ne_nosf_FL_x{xq:g}"] = qn[i]
            e[f"rho_FL_x{xq:g}"] = qr[i]
            e[f"Te_over_T200_FL_x{xq:g}"] = qt[i]
            e[f"Te_over_T200_Lee22printed_x{xq:g}"] = interp_log([xq], st["x_mid"], te_lee, ok)[0]
            e[f"Te_over_T200_B12overB16_x{xq:g}"] = interp_log([xq], st["x_mid"], te_b, ok)[0]
            for name in DENSITY_MODELS:
                mq = interp_log([xq], st["x_mid"], models[name], ok)[0]
                e[f"{name}_x{xq:g}"] = mq
                e[f"log10_{name}_over_FL_x{xq:g}"] = float(np.log10(mq / q[i])) if np.isfinite(q[i]) else np.nan
            mq = interp_log([xq], st["x_mid"], models["rho_B16"], ok)[0]
            e[f"log10_rhoB16_over_rhoFL_x{xq:g}"] = float(np.log10(mq / qr[i])) if np.isfinite(qr[i]) else np.nan
        e["f_gas_FL"] = float(np.mean(ph["f_gas"]))
        e["f_e_FL"] = float(np.mean(ph["f_e"]))
        e["f_e_FL_err"] = float(np.std([np.sum(c * ph["f_e"]) / np.sum(c) for c in st["boot_counts"]]))
        e["sf_mass_frac_r200"] = float(np.mean(ph["sf_frac"]))
        for name, (func, *_r) in DENSITY_MODELS.items():
            e[f"f_e_{name}"] = model_fe(func, st["masses"], st["z"])
        e["f_gas_B16"] = b16_fgas(st["masses"], st["z"])
        rows_out.append(e)
        per_stack.append({"st": st, "ok": ok, "prof": prof, "err": err, "models": models, "te_lee": te_lee})
    # ------------------------------------------------------------------ redshift trends at fixed mass bin and x
    trends = []
    for zmax in (2.01, 3.01):
        for b in sorted({e["bin"] for e in rows_out}):
            sel = [e for e in rows_out if e["bin"] == b and e["z"] <= zmax]
            if len(sel) < 3:
                continue
            lz = np.log(1 + np.array([e["z"] for e in sel]))
            for xq in list(XQ) + ["f_e"]:
                rec = {"zmax": round(zmax - 0.01, 1), "bin": b, "log10_m_lo": sel[0]["log10_m_lo"], "log10_m_hi": sel[0]["log10_m_hi"],
                       "x": xq, "z_values": " ".join(f"{e['z']:g}" for e in sel)}
                if xq == "f_e":
                    ly = np.log([e["f_e_FL"] for e in sel])
                    sig = np.hypot([e["f_e_FL_err"] / e["f_e_FL"] for e in sel], 0.03)
                    lyn = ly
                    lyr = np.log([e["f_gas_FL"] for e in sel])
                else:
                    ly = np.log([e[f"ne_FL_x{xq:g}"] for e in sel])
                    sig = np.hypot([e[f"ne_FL_relerr_x{xq:g}"] for e in sel], 0.03)
                    lyn = np.log([e[f"ne_nosf_FL_x{xq:g}"] for e in sel])
                    lyr = np.log([e[f"rho_FL_x{xq:g}"] for e in sel])
                good = np.isfinite(ly)
                if good.sum() < 3:
                    continue
                rec["n_z_used"] = int(good.sum())
                rec["s_FLAMINGO"], rec["s_FLAMINGO_err"] = slope_fit(lz, ly, sig)
                rec["s_FLAMINGO_nosf"] = slope_fit(lz, np.where(good, lyn, np.nan))[0]
                rec["s_FLAMINGO_gasmass"] = slope_fit(lz, np.where(good, lyr, np.nan))[0]
                for name in DENSITY_MODELS:
                    key = f"f_e_{name}" if xq == "f_e" else f"{name}_x{xq:g}"
                    rec[f"s_{name}"] = slope_fit(lz, np.where(good, np.log([e[key] for e in sel]), np.nan))[0]
                rec["s_Lee22_literal_plus3_M1"] = rec["s_Lee22_literal"] + 3.0
                trends.append(rec)
    return rows_out, trends, per_stack


def soap_gas_fractions():
    """f_gas(<R200c) = SOAP GasMass/(f_b M200c) of every flagged central, 0.2-dex bins, per snapshot."""
    out = []
    for s in SNAPSHOTS:
        cat = np.load(FPC / "notebook_cache" / f"L2p8_m9_{s:04d}_reduced_haloes.npz")
        meta = json.loads((FPC / "notebook_cache" / f"L2p8_m9_{s:04d}_reduced_haloes.json").read_text())
        z = meta["cosmology"]["redshift"]
        c = cat["is_central"].astype(bool)
        m = cat["m200c_msun"][c]
        fg = cat["gas_mass_200c_msun"][c] / (FB_SIM * m)
        lm = np.log10(m)
        for lo in np.arange(13.0, 15.6, 0.2):
            sel = (lm >= lo) & (lm < lo + 0.2)
            if sel.sum() < 5:
                continue
            ms = m[sel]
            rec = {"snapshot": s, "z": round(z, 3), "log10_m_lo": round(lo, 2), "n": int(sel.sum()),
                   "log10_m_median": float(np.log10(np.median(ms))), "f_gas_median": float(np.median(fg[sel])),
                   "f_gas_mean": float(np.mean(fg[sel])), "f_gas_p16": float(np.percentile(fg[sel], 16)),
                   "f_gas_p84": float(np.percentile(fg[sel], 84))}
            sub = ms[:: max(1, len(ms) // 60)]
            for name, (func, *_r) in DENSITY_MODELS.items():
                rec[f"f_e_{name}"] = model_fe(func, sub, z)
            rec["f_gas_B16"] = b16_fgas(sub, z)
            out.append(rec)
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------------------------------------- figures
PANEL_BINS = [(13.0, 13.2), (13.6, 13.8), (14.2, 14.4), (14.8, 15.0)]
PANEL_Z = [0.0, 0.5, 1.0, 2.0, 3.0]


def find(per_stack, lo, z):
    for p in per_stack:
        if abs(p["st"]["log10_m_lo"] - lo) < 1e-6 and abs(p["st"]["z"] - z) < 1e-3:
            return p
    return None


def fig_profiles(per_stack, path):
    fig, axes = plt.subplots(len(PANEL_BINS), len(PANEL_Z), figsize=(16, 12.5), sharex=True, sharey="row")
    for i, (lo, hi) in enumerate(PANEL_BINS):
        for j, z in enumerate(PANEL_Z):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.set_yscale("log")
            p = find(per_stack, lo, z)
            if p is None:
                ax.text(0.5, 0.5, "no FLAMINGO stack\n(<8 haloes)", ha="center", va="center", transform=ax.transAxes, color="0.4")
            else:
                st, ok, prof = p["st"], p["ok"], p["prof"]
                x = st["x_mid"]
                shown = st["shown"] & np.isfinite(prof["ne"]) & (prof["ne"] > 0)
                ax.errorbar(x[ok], prof["ne"][ok], p["err"]["ne"][ok], fmt="o", color=BLACK, ms=3.5, lw=1, zorder=5,
                            label="FLAMINGO n$_e$ (resolved)")
                und = shown & ~ok
                ax.plot(x[und], prof["ne"][und], "o", mfc="white", mec=BLACK, ms=3.5, zorder=5, label="FLAMINGO (< 2 h)")
                ax.plot(x[shown], prof["ne_nosf"][shown], "x", color="0.45", ms=3.5, zorder=4, label="FLAMINGO, no star-forming gas")
                for name, (func, color, ls, label) in DENSITY_MODELS.items():
                    ax.plot(x[shown], p["models"][name][shown], color=color, ls=ls, lw=1.8, label=label)
                ax.text(0.97, 0.96, f"N = {st['n_haloes']}, med log M = {st['log10_m_median']:.2f}", transform=ax.transAxes,
                        fontsize=8, color="0.3", ha="right", va="top")
            if i == 0:
                ax.set_title(f"z = {z:g}")
            if j == 0:
                ax.set_ylabel(r"$n_e/n_{200}$" + f"\nlog M200c = {lo:.1f}-{hi:.1f}")
            if i == len(PANEL_BINS) - 1:
                ax.set_xlabel(r"$r/R_{200c}$")
            ax.set_xlim(0.03, 1.6)
            ax.grid(alpha=0.25, which="both", lw=0.4)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda k: (not labels[k].startswith("FLAMINGO"), k))
    fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(r"FLAMINGO L2p8_m9 electron density, $n_{200} = 200\rho_{cr}f_b/(X_H m_p)$ with X$_H$ = 0.76, f$_b$ = 0.1588, "
                 "physical R200c (same haloes/shells as the pressure stacks)", y=0.955, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_slopes(trends, path):
    xs = [0.1, 0.3, 0.5, 1.0, "f_e"]
    fig, axes = plt.subplots(1, len(xs), figsize=(18, 4.8), sharey=True)
    for j, xq in enumerate(xs):
        ax = axes[j]
        tr = [t for t in trends if t["zmax"] == 2.0 and str(t["x"]) == str(xq)]
        lm = np.array([0.5 * (t["log10_m_lo"] + t["log10_m_hi"]) for t in tr])
        ax.errorbar(lm, [t["s_FLAMINGO"] for t in tr], [t["s_FLAMINGO_err"] for t in tr], fmt="o", color=BLACK,
                    label="FLAMINGO n$_e$ (all gas)")
        ax.plot(lm, [t["s_FLAMINGO_gasmass"] for t in tr], "s", mfc="white", mec=BLACK, ms=5, label="FLAMINGO gas mass density")
        for name, (func, color, ls, label) in DENSITY_MODELS.items():
            ax.plot(lm, [t["s_" + name] for t in tr], color=color, ls=ls, marker=".", lw=1.8, label=label)
        ax.plot(lm, [t["s_Lee22_literal_plus3_M1"] for t in tr], color=PINK, ls="-.", lw=1.5, label=r"Lee22 printed $\times(1+z)^3$ (M1)")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(r"$f_e(<R_{200c})$" if xq == "f_e" else f"x = {xq:g}")
        ax.set_xlabel(r"log$_{10}$ M200c [Msun] (bin centre)")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r"s = dln($n_e/n_{200}$)/dln(1+z), z $\leq$ 2")
    axes[0].text(0.03, 0.04, "Lee22 printed and x0.6016\nhave identical slopes (overlap)", transform=axes[0].transAxes, fontsize=8,
                 color=ORANGE)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_gas_fraction(soap, rows, path):
    zs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, z in zip(axes.flat, zs):
        s = [r for r in soap if abs(r["z"] - z) < 1e-3]
        lm = np.array([r["log10_m_median"] for r in s])
        ax.fill_between(lm, [r["f_gas_p16"] for r in s], [r["f_gas_p84"] for r in s], color="0.85", label="FLAMINGO SOAP 16-84%")
        ax.plot(lm, [r["f_gas_median"] for r in s], "o-", color=BLACK, ms=4, lw=1, label="FLAMINGO SOAP f$_{gas}$ median (all centrals)")
        st = [r for r in rows if abs(r["z"] - z) < 1e-3]
        ax.plot([r["log10_m_median"] for r in st], [r["f_e_FL"] for r in st], "D", mfc="white", mec=BLACK, ms=4.5,
                label="FLAMINGO f$_e$ (electrons, stack haloes)")
        for name, (func, color, ls, label) in DENSITY_MODELS.items():
            key = "f_gas_B16" if name == "B16" else f"f_e_{name}"
            lab = "Battaglia+16 gas (XGPaint)" if name == "B16" else label
            ax.plot(lm, [r[key] for r in s], color=color, ls=ls, lw=1.8, label=lab)
        ax.axhline(1, color="0.5", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 15)
        ax.set_title(f"z = {z:g}")
        ax.grid(alpha=0.3, which="both", lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel(r"log$_{10}$ M200c [Msun]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$M(<R_{200c})/(f_b M_{200c})$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    out = OUT / "density"
    out.mkdir(parents=True, exist_ok=True)
    rows, trends, per_stack = analyse()
    soap = soap_gas_fractions()
    write_csv(out / "density_metrics.csv", rows)
    write_csv(out / "density_trends.csv", trends)
    write_csv(out / "gas_fraction_soap.csv", soap)
    inside = [e for e in rows if e["z"] <= 2.01 and e["log10_m_hi"] + np.log10(H_SIM) <= 14.8 + 1e-6]
    summary = {"note": "medians over stacks with z <= 2 and M200c h <= 10^14.8 h^-1 Msun (Lee22 fitted range); log10 model/FLAMINGO",
               "X_H_n200": X_H, "mu_e": MU_E, "f_b": FB_SIM, "h_sim": H_SIM}
    for name in DENSITY_MODELS:
        summary[name] = {}
        for xq in XQ:
            v = np.array([e[f"log10_{name}_over_FL_x{xq:g}"] for e in inside])
            summary[name][f"median_log10_ratio_x{xq:g}"] = float(np.nanmedian(v))
            summary[name][f"range_log10_ratio_x{xq:g}"] = [float(np.nanmin(v)), float(np.nanmax(v))]
        v = np.array([e[f"f_e_{name}"] / e["f_e_FL"] for e in inside])
        summary[name]["median_fe_ratio"] = float(np.median(v))
        summary[name]["range_fe_ratio"] = [float(v.min()), float(v.max())]
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            vz = [e[f"log10_{name}_over_FL_x0.5"] for e in inside if abs(e["z"] - z) < 1e-3]
            summary[name][f"median_log10_ratio_x0.5_z{z:g}"] = float(np.nanmedian(vz))
    for xq in list(XQ) + ["f_e"]:
        tr = [t for t in trends if t["zmax"] == 2.0 and str(t["x"]) == str(xq) and t["log10_m_hi"] + np.log10(H_SIM) <= 14.8 + 1e-6]
        if not tr:
            continue
        s = np.array([t["s_FLAMINGO"] for t in tr])
        w = 1 / np.array([t["s_FLAMINGO_err"] for t in tr]) ** 2
        rec = {"n_bins": len(tr), "s_FLAMINGO_weighted_mean": float(np.sum(w * s) / np.sum(w)),
               "s_FLAMINGO_weighted_err": float(1 / np.sqrt(np.sum(w))), "s_FLAMINGO_range": [float(s.min()), float(s.max())],
               "s_FLAMINGO_nosf_mean": float(np.mean([t["s_FLAMINGO_nosf"] for t in tr])),
               "s_FLAMINGO_gasmass_mean": float(np.mean([t["s_FLAMINGO_gasmass"] for t in tr]))}
        for name in list(DENSITY_MODELS) + ["Lee22_literal_plus3_M1"]:
            m = np.array([t["s_" + name] for t in tr])
            rec[f"s_{name}_mean"] = float(np.mean(m))
            rec[f"s_{name}_minus_FL_weighted"] = float(np.sum(w * (m - s)) / np.sum(w))
        summary[f"trend_x{xq}"] = rec
    for xq in (0.3, 0.5, 1.0):
        summary[f"Te_over_T200_x{xq:g}"] = {
            "FLAMINGO_median_z0": float(np.nanmedian([e[f"Te_over_T200_FL_x{xq:g}"] for e in inside if e["z"] == 0])),
            "FLAMINGO_range_all": [float(np.nanmin([e[f"Te_over_T200_FL_x{xq:g}"] for e in inside])),
                                   float(np.nanmax([e[f"Te_over_T200_FL_x{xq:g}"] for e in inside]))],
            "Lee22_printed_median_z0": float(np.nanmedian([e[f"Te_over_T200_Lee22printed_x{xq:g}"] for e in inside if e["z"] == 0])),
            "B12_over_B16_median_z0": float(np.nanmedian([e[f"Te_over_T200_B12overB16_x{xq:g}"] for e in inside if e["z"] == 0]))}
    # F_n seen by FLAMINGO: Lee22 printed x (1+z)^3 over FLAMINGO (electron fraction inside R200c and n_e at x = 0.5)
    fn_rows = []
    for e in inside:
        a3 = (1 + e["z"]) ** 3
        fn_rows.append({"z": e["z"], "log10_m_lo": e["log10_m_lo"], "log10_m_median": e["log10_m_median"],
                        "Fn_fe": e["f_e_Lee22_literal"] * a3 / e["f_e_FL"],
                        "Fn_x0.5": 10 ** e["log10_Lee22_literal_over_FL_x0.5"] * a3 if np.isfinite(e["log10_Lee22_literal_over_FL_x0.5"]) else np.nan,
                        "f_e_FL": e["f_e_FL"], "f_e_Lee22_literal": e["f_e_Lee22_literal"]})
    write_csv(out / "Fn_flamingo.csv", fn_rows)
    summary["Fn_flamingo"] = {
        "definition": "Lee22 printed x (1+z)^3 / FLAMINGO; f_e inside R200c; per stack in the Lee22 range",
        "median_all": float(np.median([r["Fn_fe"] for r in fn_rows])),
        "by_mass_median_over_z": {f"{lo:.1f}": float(np.median([r["Fn_fe"] for r in fn_rows if abs(r["log10_m_lo"] - lo) < 1e-6]))
                                  for lo in sorted({r["log10_m_lo"] for r in fn_rows})},
        "by_z_median_over_mass": {f"{z:g}": float(np.median([r["Fn_fe"] for r in fn_rows if r["z"] == z]))
                                  for z in sorted({r["z"] for r in fn_rows})}}
    (out / "density_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    fig_profiles(per_stack, out / "fig_density_profiles.png")
    fig_slopes(trends, out / "fig_density_slopes.png")
    fig_gas_fraction(soap, rows, out / "fig_gas_fraction.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
