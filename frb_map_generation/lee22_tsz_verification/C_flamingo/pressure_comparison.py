"""FLAMINGO L2p8_m9 stacked electron-pressure profiles against Battaglia+12, Lee22 (as painted) and Lee22 x (1+z)^2.

Data: the preliminary stacks (flamingo_pressure_calibration/notebook_cache/stacks/, 2026-09-27; the full-statistics
idark pipeline l2p8_reduced_calibration/ was never run - /lustre/work/kristero10/flamingo_l2p8_reduced_20260928 does
not exist). Stacks: 0.2-dex bins of physical M200c, >= 8 haloes, equal-weight halo means of shell-mean Pe/P200,
bootstrap over haloes. Resolved shells = the notebook's fit cut (outside 2 pooled median smoothing lengths and
4 softenings), all inside each halo's R100c (reduced snapshots end there).

Models are evaluated on the same haloes and the same (truncated) shells: mean over haloes of the exact shell average.

Outputs (OUT/pressure/): pressure_metrics.csv (per stack: log10 model/FLAMINGO at x = 0.1, 0.3, 0.5, 1 and
Y(<R200c) ratios), pressure_trends.csv (dlog(P/P200)/dlog(1+z) at fixed x and fixed mass bin, z <= 2 and z <= 3),
pressure_summary.json, figures fig_pressure_profiles.png, fig_pressure_ratios.png, fig_pressure_ztrend.png,
fig_pressure_slopes.png.

Run: LD_LIBRARY_PATH=<halfdome env>/lib PYTHONDONTWRITEBYTECODE=1 <halfdome python> pressure_comparison.py [--agn-filter]
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common_c import (OUT, STACKS, SNAPSHOTS, PRESSURE_MODELS, fs, shell_model_mean, interp_log, slope_fit,  # noqa: E402
                      BLACK, H_SIM, FB_SIM)

XQ = np.array([0.1, 0.3, 0.5, 1.0])


def load_stacks(use_agn_filter):
    products = [fs.load_products(STACKS / f"L2p8_m9_{s:04d}_stack_products.npz") for s in SNAPSHOTS]
    return fs.build_stacks(products, use_agn_filter=use_agn_filter, n_boot=200)


def _y_within(func, masses, z, x_max):
    """4 pi int_0^x_max f x^2 dx per halo (log-Gauss from 1e-4); x_max scalar or per halo."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x_max = np.broadcast_to(np.asarray(x_max, float), masses.shape)
    t = (nodes + 1) / 2
    lnx = np.log(1e-4) + (np.log(x_max)[:, None] - np.log(1e-4)) * t[None, :]
    x = np.exp(lnx)
    jac = (np.log(x_max)[:, None] - np.log(1e-4)) / 2 * weights[None, :] * x**3
    return 4 * np.pi * np.sum(func(x, masses[:, None], z) * jac, axis=1)


def model_on_stack(st, func):
    """Model stack mean per shell, halo-mean Y(<R200c) and Y(<5 R500c) in units of sigma_T/(m_e c^2) P200 R200^3,
    and the model mean over SOAP's two outer shells (R100c-R50c, R50c-5R500c; full-snapshot apertures)."""
    edges = st["edges"]
    outer = np.where(st["complete"], st["halo_x_out"], np.nan)
    prof = shell_model_mean(func, st["masses"], st["z"], edges[:-1], outer, st["complete"])
    y200 = _y_within(func, st["masses"], st["z"], 1.0)
    y5r500 = _y_within(func, st["masses"], st["z"], st["halo_x_5r500"])
    soap = st["soap"]
    soap_model = []
    for j in range(2):
        use = soap["ok"][:, j]
        if not use.any():
            soap_model.append(np.nan)
            continue
        r, vw = fs.shell_nodes(soap["inner"][use, j], soap["outer"][use, j], 24)
        soap_model.append(float(np.mean(np.sum(vw * func(r, st["masses"][use][:, None], st["z"]), axis=-1))))
    return prof, float(np.mean(y200)), float(np.mean(y5r500)), soap_model


def analyse(use_agn_filter):
    tag = "noagn" if use_agn_filter else "raw"
    stacks = load_stacks(use_agn_filter)
    rows, per_stack = [], []
    for st in stacks:
        ok = st["fit"] & np.isfinite(st["ratio"]) & (st["ratio"] > 0)
        data_q = interp_log(XQ, st["x_mid"], st["ratio"], ok)
        rel_err = np.interp(np.log(XQ), np.log(st["x_mid"][ok]), (st["err"] / st["ratio"])[ok]) if ok.sum() >= 2 else np.full(4, np.nan)
        y_data = float(np.mean(st["halo_Yhat_200"]))
        y_boot = np.array([np.sum(c * st["halo_Yhat_200"]) / np.sum(c) for c in st["boot_counts"]])
        entry = {"snapshot": st["snapshot"], "z": round(st["z"], 3), "bin": st["bin"], "log10_m_lo": st["log10_m_lo"],
                 "log10_m_hi": st["log10_m_hi"], "log10_m_median": round(st["log10_m_median"], 3), "n_haloes": st["n_haloes"],
                 "x_resolved_min": float(st["x_mid"][ok].min()) if ok.any() else np.nan,
                 "Yhat200_flamingo": y_data, "Yhat200_flamingo_err": float(np.std(y_boot))}
        for i, xq in enumerate(XQ):
            entry[f"flamingo_x{xq:g}"] = data_q[i]
            entry[f"flamingo_relerr_x{xq:g}"] = rel_err[i] if np.isfinite(data_q[i]) else np.nan
        models = {}
        y5 = float(np.mean(st["halo_Yhat_5r500"]))
        entry["Yhat5r500_flamingo"] = y5
        entry["soap_shell1_x"] = f"{st['soap_inner_mean'][0]:.2f}-{st['soap_outer_mean'][0]:.2f}"
        entry["soap_shell2_x"] = f"{st['soap_inner_mean'][1]:.2f}-{st['soap_outer_mean'][1]:.2f}"
        for name, (func, *_rest) in PRESSURE_MODELS.items():
            prof, yhat, yhat5, soap_model = model_on_stack(st, func)
            entry[f"Y5r500_{name}_over_FL"] = yhat5 / y5
            for j in range(2):
                entry[f"log10_{name}_over_FL_soapshell{j + 1}"] = float(np.log10(soap_model[j] / st["soap_mean"][j])) \
                    if np.isfinite(soap_model[j]) and st["soap_mean"][j] > 0 else np.nan
            models[name] = prof
            mq = interp_log(XQ, st["x_mid"], prof, ok)
            for i, xq in enumerate(XQ):
                entry[f"{name}_x{xq:g}"] = mq[i]
                entry[f"log10_{name}_over_FL_x{xq:g}"] = float(np.log10(mq[i] / data_q[i])) if np.isfinite(data_q[i]) else np.nan
            entry[f"Yhat200_{name}"] = yhat
            entry[f"Y200_{name}_over_FL"] = yhat / y_data
        # shell-level rms of log10(model/data) over resolved shells
        for name, prof in models.items():
            r = np.log10(prof[ok] / st["ratio"][ok])
            entry[f"rms_log10_{name}"] = float(np.sqrt(np.mean(r**2))) if ok.any() else np.nan
            entry[f"mean_log10_{name}"] = float(np.mean(r)) if ok.any() else np.nan
        rows.append(entry)
        per_stack.append({"st": st, "ok": ok, "models": models})
    # ---------------------------------------------------------------- redshift trends at fixed mass bin and x
    trends = []
    bins = sorted({e["bin"] for e in rows})
    for zmax in (2.01, 3.01):
        for b in bins:
            sel = [e for e in rows if e["bin"] == b and e["z"] <= zmax]
            if len(sel) < 3:
                continue
            lz = np.log(1 + np.array([e["z"] for e in sel]))
            for xq in list(XQ) + ["Y200"]:
                rec = {"zmax": round(zmax - 0.01, 1), "bin": b, "log10_m_lo": sel[0]["log10_m_lo"], "log10_m_hi": sel[0]["log10_m_hi"],
                       "x": xq, "n_z": len(sel), "z_values": " ".join(f"{e['z']:g}" for e in sel),
                       "log10_m_median_by_z": " ".join(f"{e['log10_m_median']:.3f}" for e in sel)}
                if xq == "Y200":
                    ly = np.log([e["Yhat200_flamingo"] for e in sel])
                    sig = np.array([e["Yhat200_flamingo_err"] / e["Yhat200_flamingo"] for e in sel])
                else:
                    ly = np.log([e[f"flamingo_x{xq:g}"] for e in sel])
                    sig = np.array([e[f"flamingo_relerr_x{xq:g}"] for e in sel])
                sig = np.hypot(sig, 0.03)      # 3% floor as in the stack fits
                good = np.isfinite(ly)
                if good.sum() < 3:
                    continue
                rec["n_z_used"] = int(good.sum())
                rec["s_FLAMINGO"], rec["s_FLAMINGO_err"] = slope_fit(lz, ly, sig)
                for name in PRESSURE_MODELS:
                    if xq == "Y200":
                        lm = np.log([e[f"Yhat200_{name}"] for e in sel])
                    else:
                        lm = np.log([e[f"{name}_x{xq:g}"] for e in sel])
                    lm = np.where(good, lm, np.nan)
                    rec[f"s_{name}"] = slope_fit(lz, lm)[0]
                trends.append(rec)
    return tag, rows, trends, per_stack


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
        st = p["st"]
        if abs(st["log10_m_lo"] - lo) < 1e-6 and abs(st["z"] - z) < 1e-3:
            return p
    return None


def fig_profiles(per_stack, path, ratio=False):
    fig, axes = plt.subplots(len(PANEL_BINS), len(PANEL_Z), figsize=(16, 12.5), sharex=True, sharey="row" if not ratio else True)
    for i, (lo, hi) in enumerate(PANEL_BINS):
        for j, z in enumerate(PANEL_Z):
            ax = axes[i, j]
            p = find(per_stack, lo, z)
            ax.set_xscale("log")
            if not ratio:
                ax.set_yscale("log")
            if p is None:
                ax.text(0.5, 0.5, "no FLAMINGO stack\n(<8 haloes)", ha="center", va="center", transform=ax.transAxes, color="0.4")
            else:
                st, ok = p["st"], p["ok"]
                shown = st["shown"] & np.isfinite(st["ratio"]) & (st["ratio"] > 0)
                x = st["x_mid"]
                if not ratio:
                    ax.errorbar(x[ok], st["ratio"][ok], st["err"][ok], fmt="o", color=BLACK, ms=3.5, lw=1, zorder=5,
                                label="FLAMINGO L2p8_m9 (resolved)")
                    und = shown & ~ok
                    ax.plot(x[und], st["ratio"][und], "o", mfc="white", mec=BLACK, ms=3.5, zorder=5, label="FLAMINGO (< 2 h)")
                    for name, (func, color, ls, label) in PRESSURE_MODELS.items():
                        m = p["models"][name]
                        ax.plot(x[shown], m[shown], color=color, ls=ls, lw=1.8, label=label)
                else:
                    ax.axhline(1, color=BLACK, lw=0.8)
                    ax.fill_between(x[ok], 1 - st["err"][ok] / st["ratio"][ok], 1 + st["err"][ok] / st["ratio"][ok], color="0.8",
                                    label="FLAMINGO bootstrap 1$\\sigma$")
                    for name, (func, color, ls, label) in PRESSURE_MODELS.items():
                        m = p["models"][name]
                        ax.plot(x[ok], m[ok] / st["ratio"][ok], color=color, ls=ls, lw=1.8, label=label)
                        und = shown & ~ok
                        ax.plot(x[und], m[und] / st["ratio"][und], color=color, ls=ls, lw=0.8, alpha=0.5)
                    ax.set_yscale("log")
                    ax.set_ylim(0.03, 12)
                ax.text(0.97, 0.96, f"N = {st['n_haloes']}, med log M = {st['log10_m_median']:.2f}", transform=ax.transAxes,
                        fontsize=8, color="0.3", ha="right", va="top")
            if i == 0:
                ax.set_title(f"z = {z:g}")
            if j == 0:
                ax.set_ylabel(("model / FLAMINGO" if ratio else r"$P_e/P_{200}$") + f"\nlog M200c = {lo:.1f}-{hi:.1f}")
            if i == len(PANEL_BINS) - 1:
                ax.set_xlabel(r"$r/R_{200c}$")
            ax.set_xlim(0.03, 1.6)
            ax.grid(alpha=0.25, which="both", lw=0.4)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: (not labels[i].startswith("FLAMINGO L2p8") and not labels[i].startswith("FLAMINGO ("), i))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("FLAMINGO L2p8_m9 electron pressure vs models (same haloes, same shells; M200c in physical Msun; "
                 "P200 with f$_b$=0.1588 and physical R200c)", y=0.965, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_ztrend(rows, trends, path):
    xs = [0.3, 0.5, 1.0, "Y200"]
    bins_show = [(13.0, 13.2), (13.4, 13.6), (13.8, 14.0), (14.2, 14.4)]
    fig, axes = plt.subplots(len(bins_show), len(xs), figsize=(15, 12), sharex=True)
    for i, (lo, hi) in enumerate(bins_show):
        sel = sorted([e for e in rows if abs(e["log10_m_lo"] - lo) < 1e-6], key=lambda e: e["z"])
        z = np.array([e["z"] for e in sel])
        for j, xq in enumerate(xs):
            ax = axes[i, j]
            key = "Yhat200_flamingo" if xq == "Y200" else f"flamingo_x{xq:g}"
            d = np.array([e[key] for e in sel])
            if xq == "Y200":
                err = np.array([e["Yhat200_flamingo_err"] for e in sel])
            else:
                err = d * np.array([e[f"flamingo_relerr_x{xq:g}"] for e in sel])
            finite = np.flatnonzero(np.isfinite(d))
            if not len(finite):
                ax.text(0.5, 0.5, "not resolved", ha="center", va="center", transform=ax.transAxes, color="0.4")
                ax.set_xscale("log")
                ax.set_xlim(0.93, 5.6)
                continue
            ref = d[finite[0]]
            ax.errorbar(1 + z, d / ref, err / ref, fmt="o", color=BLACK, ms=4, zorder=5, label="FLAMINGO")
            tr = [t for t in trends if t["zmax"] == 2.0 and abs(t["log10_m_lo"] - lo) < 1e-6 and str(t["x"]) == str(xq)]
            short = {"B12": "B12", "Lee22": "L22", "Lee22_a2": "L22(1+z)$^2$", "FLfit": "FL fit"}
            txt = [f"s: FL {tr[0]['s_FLAMINGO']:+.2f}$\\pm${tr[0]['s_FLAMINGO_err']:.2f}"] if tr else []
            for name, (func, color, ls, label) in PRESSURE_MODELS.items():
                mkey = f"Yhat200_{name}" if xq == "Y200" else f"{name}_x{xq:g}"
                m = np.array([e[mkey] for e in sel])
                ax.plot(1 + z, m / ref, color=color, ls=ls, lw=1.8, marker=".", label=label)
                if tr:
                    txt.append(f"{short[name]} {tr[0]['s_' + name]:+.2f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axvline(3.0, color="0.6", lw=0.7, ls=":")
            ax.text(0.02, 0.02, "\n".join(txt), transform=ax.transAxes, va="bottom", fontsize=6.8,
                    bbox=dict(fc="white", ec="0.8", alpha=0.85))
            ax.set_xlim(0.93, 5.6)
            if i == 0:
                ax.set_title(r"$Y(<R_{200c})/(P_{200}R_{200}^3)$" if xq == "Y200" else f"$P_e/P_{{200}}$ at x = {xq:g}")
            if j == 0:
                ax.set_ylabel(f"relative to FLAMINGO at lowest z\nlog M = {lo:.1f}-{hi:.1f}")
            if i == len(bins_show) - 1:
                ax.set_xlabel("1 + z")
            ax.set_xticks([1, 1.5, 2, 3, 4])
            ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
            ax.grid(alpha=0.25, which="both", lw=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("Redshift evolution at fixed M200c bin and fixed x (s = dln/dln(1+z) fitted over z <= 2; dotted line z = 2)",
                 y=0.965, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_slopes(trends, path):
    xs = [0.1, 0.3, 0.5, 1.0, "Y200"]
    fig, axes = plt.subplots(1, len(xs), figsize=(18, 4.6), sharey=True)
    for j, xq in enumerate(xs):
        ax = axes[j]
        tr = [t for t in trends if t["zmax"] == 2.0 and str(t["x"]) == str(xq)]
        lm = np.array([0.5 * (t["log10_m_lo"] + t["log10_m_hi"]) for t in tr])
        ax.errorbar(lm, [t["s_FLAMINGO"] for t in tr], [t["s_FLAMINGO_err"] for t in tr], fmt="o", color=BLACK, label="FLAMINGO")
        for name, (func, color, ls, label) in PRESSURE_MODELS.items():
            ax.plot(lm, [t["s_" + name] for t in tr], color=color, ls=ls, marker=".", lw=1.8, label=label)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(r"$Y(<R_{200c})/(P_{200}R_{200}^3)$" if xq == "Y200" else f"x = {xq:g}")
        ax.set_xlabel(r"log$_{10}$ M200c [Msun] (bin centre)")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r"s = dln($P_e/P_{200}$)/dln(1+z), z $\leq$ 2")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agn-filter", action="store_true", help="drop SOAP's recently-AGN-heated gas (default: all gas)")
    args = parser.parse_args()
    out = OUT / "pressure"
    out.mkdir(parents=True, exist_ok=True)
    tag, rows, trends, per_stack = analyse(args.agn_filter)
    write_csv(out / f"pressure_metrics_{tag}.csv", rows)
    write_csv(out / f"pressure_trends_{tag}.csv", trends)
    # summary: medians over stacks inside Lee22's calibrated range (z <= 2, 1e13-10^14.8 h^-1 Msun)
    summary = {"gas": tag, "h_sim": H_SIM, "f_b": FB_SIM, "n_stacks": len(rows),
               "note": "log10 model/FLAMINGO medians over stacks with z <= 2 and M200c h <= 10^14.8 (Lee22 fitted range)"}
    inside = [e for e in rows if e["z"] <= 2.01 and e["log10_m_hi"] + np.log10(H_SIM) <= 14.8 + 1e-6]
    for name in PRESSURE_MODELS:
        summary[name] = {}
        for xq in XQ:
            v = np.array([e[f"log10_{name}_over_FL_x{xq:g}"] for e in inside])
            summary[name][f"median_log10_ratio_x{xq:g}"] = float(np.nanmedian(v))
            summary[name][f"range_log10_ratio_x{xq:g}"] = [float(np.nanmin(v)), float(np.nanmax(v))]
        v = np.array([e[f"Y200_{name}_over_FL"] for e in inside])
        summary[name]["median_Y200_ratio"] = float(np.median(v))
        summary[name]["range_Y200_ratio"] = [float(v.min()), float(v.max())]
        v5 = np.array([e[f"Y5r500_{name}_over_FL"] for e in inside])
        summary[name]["median_Y5r500_ratio"] = float(np.median(v5))
        summary[name]["range_Y5r500_ratio"] = [float(v5.min()), float(v5.max())]
        for j in (1, 2):
            vs = np.array([e[f"log10_{name}_over_FL_soapshell{j}"] for e in inside])
            summary[name][f"median_log10_ratio_soapshell{j}"] = float(np.nanmedian(vs))
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            vz = [e[f"Y200_{name}_over_FL"] for e in inside if abs(e["z"] - z) < 1e-3 and e["log10_m_lo"] >= 13.6 - 1e-6]
            summary[name][f"median_Y200_ratio_z{z:g}_M>=13.6"] = float(np.median(vz))
        summary[name]["median_rms_log10_shells"] = float(np.nanmedian([e[f"rms_log10_{name}"] for e in inside]))
    for xq in list(XQ) + ["Y200"]:
        tr = [t for t in trends if t["zmax"] == 2.0 and str(t["x"]) == str(xq) and t["log10_m_hi"] + np.log10(H_SIM) <= 14.8 + 1e-6]
        if not tr:
            continue
        s = np.array([t["s_FLAMINGO"] for t in tr])
        e = np.array([t["s_FLAMINGO_err"] for t in tr])
        w = 1 / e**2
        rec = {"n_bins": len(tr), "s_FLAMINGO_weighted_mean": float(np.sum(w * s) / np.sum(w)),
               "s_FLAMINGO_weighted_err": float(1 / np.sqrt(np.sum(w))), "s_FLAMINGO_range": [float(s.min()), float(s.max())]}
        for name in PRESSURE_MODELS:
            m = np.array([t["s_" + name] for t in tr])
            rec[f"s_{name}_mean"] = float(np.mean(m))
            rec[f"s_{name}_minus_FL_weighted"] = float(np.sum(w * (m - s)) / np.sum(w))
        summary[f"trend_x{xq}"] = rec
    (out / f"pressure_summary_{tag}.json").write_text(json.dumps(summary, indent=1) + "\n")
    if not args.agn_filter:
        fig_profiles(per_stack, out / "fig_pressure_profiles.png")
        fig_profiles(per_stack, out / "fig_pressure_ratios.png", ratio=True)
        fig_ztrend(rows, trends, out / "fig_pressure_ztrend.png")
        fig_slopes(trends, out / "fig_pressure_slopes.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
