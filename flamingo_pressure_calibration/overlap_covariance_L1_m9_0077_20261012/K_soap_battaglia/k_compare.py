"""Test K stage K4 (local, small): comparisons of the refitted laws with the published ones, the SOAP data, the z = 0
particle stacks and Test I's per-bin fits; parameter tables, XGPaint / Python snippets, electron factors.

  python -B k_compare.py [--results DIR]
Inputs: results/k3_fits.json, k3_predictions.npz, k2_cell_fits.json, cells/*.npz; stacking_choices_20261014 cache
(core/stackcore.py, z = 0 particle stacks); I_fitting_a2ii/results/fits35.json (Test I per-bin B12/B16);
stacking_choices_20261014/research/soap/soap_audit.json (gas-to-electron factors).
Outputs (results/): k4_param_tables.csv/.json, k4_profiles.npz, k4_profile_ratios.json (refitted / published at exact
x), k4_law_values.json (the five profile parameters of every law on a mass x z grid), k4_aperture_integrals.csv,
k4_electron_factors.json, k4_particle_z0.npz, k4_information.json, k4_summary.json; report/xgpaint_refit_snippet.jl
(the fits named in report/recommendation.json 'snippet'), report/k_params_usage.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k_common as KC  # noqa: E402
import k_fit_core as K  # noqa: E402

STUDY = HERE.parent
SC = STUDY / "stacking_choices_20261014"
MASSES = (12.8, 13.0, 13.5, 14.0, 14.5, 15.0)
ZS = (0.0, 0.5, 1.0, 2.0, 3.0)
XGRID = np.geomspace(0.01, 4.0, 160)


def jf(v, sig=10):
    """Julia Float64 literal with `sig` significant digits (always with a '.' or an exponent, so 1 -> 1.0)."""
    t = f"{float(v):.{sig}g}"
    return t if any(c in t for c in ".einf") else t + ".0"


def recommendation():
    """Recommended and alternative fits (report/recommendation.json, written after reading the results)."""
    p = HERE / "report" / "recommendation.json"
    return json.loads(p.read_text()) if p.exists() else {"snippet": {}}


def law_of(key):
    field, r, p = key.split("|")
    return K.Law(KC.FIELDS[field]["form"], p)


def param_rows(fits):
    rows = []
    pubs = {"B12": K.Law("B12", "orig"), "B16": K.Law("B16", "orig")}
    for form, law in pubs.items():
        tab = K.law_table(law, law.start())
        for q, e in tab["params"].items():
            rows.append({"field": "P" if form == "B12" else "G", "range": "-", "parametrization": "published", "q": q,
                         "xgp_name": K.XGP_NAME[q], **{k: v for k, v in e.items()}})
    for key, item in fits["configs"].items():
        if "table" not in item:
            continue
        field, r, p = key.split("|")
        for q, e in item["table"]["params"].items():
            rows.append({"field": field, "range": r, "parametrization": p, "q": q, "xgp_name": K.XGP_NAME[q],
                         **{k: v for k, v in e.items()}})
        if "log10_M_break" in item["table"]:
            rows.append({"field": field, "range": r, "parametrization": p, "q": "log10_M_break", "xgp_name": "M_break",
                         "value": item["table"]["log10_M_break"], "err_hess": item["table"]["log10_M_break_err_hess"],
                         "err_boot": item["table"]["log10_M_break_err_boot"]})
        if "alpha_break" in item["table"]:
            rows.append({"field": field, "range": r, "parametrization": p, "q": "alpha_break", "xgp_name": "alpha_break",
                         "value": item["table"]["alpha_break"], "err_hess": item["table"]["alpha_break_err_hess"],
                         "err_boot": item["table"]["alpha_break_err_boot"]})
    return rows


def profiles(fits):
    out = {"x": XGRID, "masses_log10": np.array(MASSES), "z": np.array(ZS)}
    keys = ["published|P", "published|G"] + [k for k in fits["configs"] if "table" in fits["configs"][k]]
    for key in keys:
        if key.startswith("published"):
            form = "B12" if key.endswith("P") else "B16"
            law = K.Law(form, "orig"); th = law.start()
        else:
            law = law_of(key); th = np.array(fits["configs"][key]["theta"])
        F = np.zeros((len(MASSES), len(ZS), len(XGRID)))
        Cm = np.zeros_like(F)
        for i, lm in enumerate(MASSES):
            for j, z in enumerate(ZS):
                p = K.law_profile_params(law, th, 10 ** lm, z)
                p = {q: float(np.ravel(p[q])[0]) for q in K.QNAMES}
                F[i, j] = K.profile(XGRID, law.form, *(p[q] for q in K.QNAMES))
                Cm[i, j] = K.cumulative(XGRID, law.form, *(p[q] for q in K.QNAMES))
        out[f"F__{key}"] = F
        out[f"C__{key}"] = Cm
    return out


def aperture_integrals(pred):
    """Y(<R500c), Y(<R200c) (Ytilde) and M_gas(<R200c) (Mtilde) per cell: SOAP mean vs each law (R3 dataset points;
    the laws are evaluated at the cell's own nodes)."""
    rows = []
    for field, name, label in (("P", "SO/500_crit", "Y500"), ("P", "SO/200_crit", "Y200"), ("G", "SO/200_crit", "Mgas200")):
        for r in KC.RANGES:
            base = f"{field}__{r}"
            if base + "__lnd" not in pred:
                continue
            names = pred[base + "__name"]
            sel = np.nonzero(names == name)[0]
            models = [k for k in pred.files if k.startswith(base + "__fit_")] + [base + "__published"]
            for i in sel:
                row = {"quantity": label, "eval_dataset": r, "cell": str(pred[base + "__cell"][i]), "z": float(pred[base + "__z"][i]),
                       "log10M_mean": float(pred[base + "__cell_lnM"][i] / math.log(10)), "N": float(pred[base + "__cell_n"][i]),
                       "soap": float(math.exp(pred[base + "__lnd"][i])), "sd_boot_ln": float(pred[base + "__sd_boot"][i])}
                for m in models:
                    tag = m.split("__")[-1]
                    row[f"ratio_{tag}"] = float(math.exp(pred[m][i] - pred[base + "__lnd"][i]))
                rows.append(row)
    return rows


def electron_factors():
    p = SC / "research" / "soap" / "soap_audit.json"
    d = json.loads(p.read_text())["validation_interpolated_all_apertures"]
    out = {"source": str(p.relative_to(STUDY)) + " -> validation_interpolated_all_apertures/<aperture>/<mass group>/electrons_from_Mgas",
           "definition": "r = [M_gas / mu_e(X = 0.76, fully ionised)] / N_e(particles), per halo at z = 0 (study haloes, "
                         "SOAP vs particle pass at the same radius); median and 16-84%",
           "use": "electrons = (gas-density law / mu_e) / r; in XGPaint's BattagliaTauProfile (n_e = 0.9 rho_gas / mu_e) "
                  "paint the refitted gas law with P0_amp multiplied by 1 / (0.9 r)",
           "factors": {}}
    for ap, groups in d.items():
        if not ap.startswith(("SO/", "InclusiveSphere/")):
            continue
        for g, v in groups.items():
            e = v.get("electrons_from_Mgas") if isinstance(v, dict) else None
            if e:
                out["factors"].setdefault(ap, {})[g] = {"median": e["median"], "p16": e["p16"], "p84": e["p84"],
                                                        "n": e["n"], "xgpaint_multiplier_1_over_0p9r": 1 / (0.9 * e["median"])}
        if isinstance(groups, dict) and "x_range_median" in groups:
            out["factors"].setdefault(ap, {})["x_range_median"] = groups["x_range_median"]
    return out


def particle_z0():
    """Inclusion-weighted z = 0 particle stacks on the 35-shell fit grid (P and G; all gas and A2-ii; recent AGN
    removed) per 0.2-dex bin, and Test I's per-bin B12 / B16 (objective a) parameters."""
    sys.path.insert(0, str(SC / "core"))
    out = {}
    try:
        import stackcore as sc
    except Exception as exc:   # noqa: BLE001
        return {"error": repr(exc)}
    bins = sc.mass_bins("M200c_0.2")
    out["edges"] = np.asarray(sc.edges("fit"))
    out["bin_labels"] = np.array([b["label"] for b in bins])
    out["bin_mean_log_mass"] = np.array([b["mean_log_mass"] for b in bins])
    for key, tag in (("P|geometric|removed|all|all|fit|P200", "P_all"), ("P|geometric|removed|A2ii|all|fit|P200", "P_A2ii"),
                     ("G|geometric|removed|all|all|fit|rhoc_fb", "G_all"), ("G|geometric|removed|A2ii|all|fit|rhoc_fb", "G_A2ii")):
        vals, errs, res = [], [], []
        for b in bins:
            r = sc.stack_bin(key, b, n_boot=100)
            vals.append(r["value"]); errs.append(r["err"]); res.append(r["resolved"])
        out[f"{tag}_mean"] = np.array(vals); out[f"{tag}_err"] = np.array(errs); out[f"{tag}_resolved"] = np.array(res)
    fi = STUDY / "I_fitting_a2ii" / "results" / "fits35.json"
    if fi.exists():
        f = json.loads(fi.read_text())["fits"]
        for fld in ("P", "E"):
            for rf in ("1", "2", "4"):
                arr = np.full((10, 3), np.nan)
                for b in range(10):
                    k = f"{fld}|a|{rf}|{b}"
                    if k in f and f[k].get("parameters"):
                        arr[b] = f[k]["parameters"]
                out[f"testI_{fld}_a_Rfit{rf}"] = arr
        out["testI_bins_log10M_lo"] = np.round(np.arange(13.5, 15.5, 0.2), 2)
        out["testI_note"] = np.array("Test I fits35.json 'fits' <field>|a|<R_fit>|<bin>: B12 (P0, x_c, beta) on P = P_e/P200 "
                                     "(model includes 0.5176); B16 (rho0, alpha, beta) on E = n_e mu_XGP/(rho_c f_b) "
                                     "(electrons incl. XGPaint's 1/0.9; E = G x kappa_E, kappa_E = 1.094 at 2-4 R200c)")
    return out


def information(fits):
    out = {}
    for field in ("P", "G"):
        for p in KC.PARAMETRIZATIONS:
            ld = {}
            for r in KC.RANGES:
                it = fits["configs"].get(f"{field}|{r}|{p}")
                if it and "cov" in it:
                    sign, val = np.linalg.slogdet(np.array(it["cov"]))
                    ld[r] = float(val) if sign > 0 else None
            out[f"{field}|{p}"] = {"logdet_cov": ld}
            for a, b in (("R1", "R2"), ("R2", "R3"), ("R1", "R3")):
                if ld.get(a) is not None and ld.get(b) is not None:
                    out[f"{field}|{p}"][f"info_gain_bits_{a}_to_{b}"] = 0.5 * (ld[a] - ld[b]) / math.log(2)
            # parameter shifts between ranges in units of the combined Hessian error
            for a, b in (("R1", "R2"), ("R2", "R3")):
                ia, ib = fits["configs"].get(f"{field}|{a}|{p}"), fits["configs"].get(f"{field}|{b}|{p}")
                if ia and ib and "theta" in ia and "theta" in ib:
                    ta, tb = np.array(ia["theta"]), np.array(ib["theta"])
                    sa, sb = np.sqrt(np.diag(ia["cov"])), np.sqrt(np.diag(ib["cov"]))
                    out[f"{field}|{p}"][f"shift_sigma_{a}_{b}"] = dict(zip(ia["names"], ((tb - ta) / np.hypot(sa, sb)).round(2).tolist()))
    return out


def snippet(fits, rec):
    """XGPaint (Julia) and Python snippets for the recommended fits.  Every number is a Float64 literal (XGPaint's
    keyword constructors require one type T for all arguments; an Int such as 1 raises a MethodError)."""
    ef = json.loads((HERE / "results" / "k4_electron_factors.json").read_text())["factors"]
    r200 = {g: ef["SO/200_crit"][g]["median"] for g in ("12.5-13.5", "13.5-14.5", ">=14.5")}
    lines = ["# Test K: Battaglia laws refitted to FLAMINGO L1_m9 SOAP-HBT (results/k_params.json); XGPaint branch 'cluster'",
             "# (checked: report/xgpaint_snippet_check.json, Julia 1.12.2 + /home/kn18001/.julia/dev/XGPaint).",
             "# Masses in Msun (no h), pivot 1e14 Msun, x = r/R200c.  The profiles scale with the painter's f_b = Omega_b/(Omega_b+Omega_c)",
             "# (fitted with FLAMINGO's f_b = 0.0486/0.306 = 0.15882; HalfDome's 0.0486/0.3089 = 0.15733 paints 0.9906 of it).",
             "using XGPaint",
             "Omega_b, Omega_c, h = 0.0486, 0.3089 - 0.0486, 0.6774      # HalfDome lightcone cosmology", ""]
    for label, key in rec.items():
        it = fits["configs"].get(key)
        if not it or "table" not in it:
            continue
        t = it["table"]["params"]
        field = key.split("|")[0]
        args = []
        for q in K.QNAMES:
            e = t[q]
            n = {"amp": "P0", "xc": "x_c", "alpha": "alpha", "beta": "beta", "gamma": "gamma"}[q]
            if e["kind"] == "fixed":
                args.append(f"{n}_amp={jf(e['value'])}, {n}_alpha_m=0.0, {n}_alpha_z=0.0")
            elif e["kind"] == "pl":
                args.append(f"{n}_amp={jf(e['A'])}, {n}_alpha_m={jf(e['alpha_m'])}, {n}_alpha_z={jf(e['alpha_z'])}")
        ctor = "Battaglia16ThermalSZProfile" if field == "P" else "BattagliaTauProfilePhysical"
        extra = ""
        if it["config"][2] == "brk_amp" and field == "P":
            ctor = "BreakModel"
            extra = f"alpha_break={jf(it['table']['alpha_break'])}, M_break={jf(10 ** it['table']['log10_M_break'])}, "
        if it["config"][2] in ("brk_all",) or (it["config"][2] == "brk_amp" and field != "P"):
            lines.append(f"# {label} ({key}): broken power laws are not an XGPaint type; evaluate with k_fit_core.Law")
            lines.append("")
            continue
        lines.append(f"# {label}: {key}  (sigma_d = {it['sigma_d']:.4f})")
        lines.append(f"model_{label} = XGPaint.{ctor}(; Omega_c=Omega_c, Omega_b=Omega_b, h=h, {extra}")
        lines.append("    " + ",\n    ".join(args) + ")")
        if field == "G":
            amp = t["amp"]["A"]
            lines.append(f"# gas density -> particle electrons with XGPaint (n_e = 0.9 rho_gas/mu_e): P0_amp = {jf(amp)} / (0.9 r), r = particle-measured")
            lines.append(f"# [M_gas/mu_e]/N_e at R200c (k4_electron_factors.json): {r200['12.5-13.5']:.4f} (10^12.5-13.5) -> P0_amp = {jf(amp / (0.9 * r200['12.5-13.5']))};"
                         f" {r200['13.5-14.5']:.4f} (10^13.5-14.5) -> {jf(amp / (0.9 * r200['13.5-14.5']))}; {r200['>=14.5']:.4f} (>= 10^14.5) -> {jf(amp / (0.9 * r200['>=14.5']))}")
        lines.append("")
    py = ['"""Evaluate any Test K law (k_params.json) with k_fit_core; works for every parametrisation incl. brk_all."""',
          "import json, numpy as np, sys",
          f"sys.path.insert(0, {str(HERE)!r})",
          "import k_fit_core as K",
          f"P = json.load(open({str(HERE / 'results' / 'k_params.json')!r}))",
          "def law(key):",
          "    f = P['fits'][key]; form = 'B12' if f['field'] in ('P', 'Pagn') else 'B16'",
          "    L = K.Law(form, f['parametrization']); return L, np.array(f['theta'])",
          "L, th = law('P|R3|orig')",
          "x = np.geomspace(0.05, 3, 50)",
          "print(K.law_profile(L, th, x, 1e14, 0.5))         # P_e/P200 at M200c = 1e14 Msun, z = 0.5",
          "p = {q: float(np.ravel(v)[0]) for q, v in K.law_profile_params(L, th, 1e14, 0.5).items()}",
          "print(4 * np.pi * K.cumulative(1.0, L.form, *(p[q] for q in K.QNAMES)))   # Ytilde(<R200c)"]
    return "\n".join(lines) + "\n", "\n".join(py) + "\n"


def profile_ratios_exact(fits, xs=(0.1, 0.3, 0.5, 1.0, 2.0, 3.0)):
    """Refitted / published profile at exactly x = xs (not grid points), for every fit, at MASSES x ZS."""
    out = {"x": list(xs), "log10_M200c": list(MASSES), "z": list(ZS), "ratios": {}, "values": {}}
    xa = np.array(xs, float)
    pub = {}
    for field, form in (("P", "B12"), ("G", "B16")):
        law = K.Law(form, "orig")
        pub[field] = {(lm, z): K.law_profile(law, law.start(), xa, 10 ** lm, z) for lm in MASSES for z in ZS}
        out["values"][f"published|{field}"] = {f"logM{lm:g}|z{z:g}": pub[field][(lm, z)].tolist() for lm in MASSES for z in ZS}
    for key, item in fits["configs"].items():
        if "theta" not in item:
            continue
        field = key.split("|")[0]
        if field not in ("P", "G", "Pagn"):
            continue
        law = law_of(key); th = np.array(item["theta"])
        ref = pub["P" if field == "Pagn" else field]
        tab, val = {}, {}
        for lm in MASSES:
            for z in ZS:
                F = K.law_profile(law, th, xa, 10 ** lm, z)
                val[f"logM{lm:g}|z{z:g}"] = F.tolist()
                tab[f"logM{lm:g}|z{z:g}"] = (F / ref[(lm, z)]).tolist()
        out["ratios"][key] = tab
        out["values"][key] = val
    return out


def law_parameter_values(fits, pred, logm=(12.8, 13.0, 13.5, 14.0, 14.5, 15.0, 15.3), zs=(0.0, 0.5, 1.0, 2.0, 3.0)):
    """The five profile parameters of every fit (and the published laws) on a (M, z) grid, and, per fit, their range
    over the cells of its own dataset (cell mean mass and output redshift of every fitted point: the data domain)."""
    out = {"log10_M200c": list(logm), "z": list(zs), "grid": {}, "data_domain": {},
           "note": "grid[key][q][i][j] = parameter q at log10 M200c = log10_M200c[i], z = z[j]; amplitude of brk_amp includes "
                   "the break factor; data_domain[key][q] = [min, max] over the fitted cells (k3_predictions.npz cell_lnM, z)"}
    keys = [("published|P", "B12", "orig", None, None), ("published|G", "B16", "orig", None, None)]
    keys += [(k, KC.FIELDS[k.split("|")[0]]["form"], k.split("|")[2], np.array(v["theta"]), f"{k.split('|')[0]}__{k.split('|')[1]}")
             for k, v in fits["configs"].items() if "theta" in v]
    for key, form, kind, th, base in keys:
        law = K.Law(form, kind)
        th = law.start() if th is None else th
        M, Z = np.meshgrid(10 ** np.array(logm), np.array(zs), indexing="ij")
        ph = law.phi(th, np.log(M), np.log1p(Z))
        out["grid"][key] = {q: np.asarray(ph[q], float).tolist() for q in K.QNAMES}
        if base is not None and base + "__cell_lnM" in pred.files:
            lnm, zz = pred[base + "__cell_lnM"], pred[base + "__z"]
            pd = law.phi(th, lnm, np.log1p(zz))
            out["data_domain"][key] = {q: [float(np.min(pd[q])), float(np.max(pd[q]))] for q in K.QNAMES}
            out["data_domain"][key]["log10M_range"] = [float(lnm.min() / math.log(10)), float(lnm.max() / math.log(10))]
            out["data_domain"][key]["z_range"] = [float(zz.min()), float(zz.max())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args()
    R = Path(a.results)
    fits = json.loads((R / "k3_fits.json").read_text())
    pred = np.load(R / "k3_predictions.npz", allow_pickle=False)
    rows = param_rows(fits)
    keys = sorted({k for r in rows for k in r})
    with open(R / "k4_param_tables.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    KC.write_json(R / "k4_param_tables.json", rows)
    np.savez_compressed(R / "k4_profiles.npz", **profiles(fits))
    ai = aperture_integrals(pred)
    if ai:
        keys = sorted({k for r in ai for k in r})
        with open(R / "k4_aperture_integrals.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in ai:
                w.writerow(r)
    KC.write_json(R / "k4_electron_factors.json", electron_factors())
    pz = particle_z0()
    if "error" not in pz:
        np.savez_compressed(R / "k4_particle_z0.npz", **pz)
    KC.write_json(R / "k4_information.json", information(fits))
    KC.write_json(R / "k4_profile_ratios.json", profile_ratios_exact(fits))
    KC.write_json(R / "k4_law_values.json", law_parameter_values(fits, pred))
    rec = recommendation().get("snippet", {})
    jl, py = snippet(fits, rec)
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "xgpaint_refit_snippet.jl").write_text(jl)
    (HERE / "report" / "k_params_usage.py").write_text(py)
    KC.write_json(R / "k4_summary.json", {"n_param_rows": len(rows), "n_aperture_rows": len(ai),
                                          "particle_z0": "ok" if "error" not in pz else pz["error"]})
    print("K4 done", len(rows), "parameter rows,", len(ai), "aperture rows")


if __name__ == "__main__":
    main()
