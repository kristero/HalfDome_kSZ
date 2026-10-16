"""Test K report: the website page 'Battaglia parameters refitted to FLAMINGO (SOAP)' (site content syntax of
stacking_choices_20261014/site/build_site.py) and report_numbers.json, built from the results files.

  python -B report_k.py [--results DIR] [--narrative report/narrative.json]
Outputs: report/fit_soap_battaglia.txt (copy into stacking_choices_20261014/site/content/ and rebuild the site),
report/report_numbers.json (every number of the page with its source file / field), report/param_tables.md.txt.
The narrative bullets (in short, recommendations) are read from report/narrative.json, written after reading the
results; every number in them is also in report_numbers.json.
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

REL = "K_soap_battaglia"
_AU = HERE / "results" / "k7_fit_audit.json"
NOT_CONVERGED = {k for k, c in json.loads(_AU.read_text())["convergence"].items() if not c["converged"]} if _AU.exists() else set()
QL = {"B12": {"amp": "$P_0$", "xc": "$x_c$", "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$"},
      "B16": {"amp": r"$\rho_0$", "xc": "$x_c$", "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$"}}


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    a = abs(v)
    if a >= 1000:
        return f"{v:.4g}"
    if a >= 100:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.{nd}f}"
    return f"{v:.{nd + 1}f}" if a >= 0.01 else f"{v:.2e}"


def pm(v, e, nd=3):
    if e is None or not (isinstance(e, float) and math.isfinite(e)):
        return fmt(v, nd)
    return f"{fmt(v, nd)} ± {fmt(e, nd)}"


def near_bound(key, it, tol=1e-3):
    """Law parameters within tol (in the fitted coordinate: ln A, exponents, log10 M_b) of a fit bound; stricter
    k_fit_core 'at_bound' (1e-6 of the range) misses e.g. x_c A = 9.9999 at the bound 10."""
    form = KC.FIELDS[key.split("|")[0]]["form"]
    law = K.Law(form, key.split("|")[2])
    lo, hi = law.bounds()
    th = np.asarray(it["theta"], float)
    return [n for i, n in enumerate(law.names) if (np.isfinite(lo[i]) and th[i] - lo[i] < tol) or (np.isfinite(hi[i]) and hi[i] - th[i] < tol)]


def law_rows(fits, field, configs):
    """Markdown table: one row per (config, profile parameter): A, alpha_m, alpha_z (Hessian / bootstrap errors)."""
    form = KC.FIELDS[field]["form"]
    pub = K.law_table(K.Law(form, "orig"), K.Law(form, "orig").start())
    lines = ["| fit | parameter | $A$ (at $10^{14}\\,M_\\odot$, z = 0) | $\\alpha_m$ | $\\alpha_z$ | $\\sigma_d$ |",
             "|---|---|---|---|---|---|"]
    for q in K.QNAMES:
        e = pub["params"][q]
        if e["kind"] == "fixed":
            continue
        lines.append(f"| published | {QL[form][q]} | {fmt(e['A'])} | {fmt(e['alpha_m'])} | {fmt(e['alpha_z'])} | – |")
    for key in configs:
        it = fits["configs"].get(key)
        if not it or "table" not in it:
            continue
        f_, r_, p_ = key.split("|")
        label = f"{p_} {r_}" + (" (Y incl. recent AGN)" if f_ == "Pagn" else "") + (" **(not converged, not usable)**" if key in NOT_CONVERGED else "")
        t = it["table"]["params"]
        bound = set(near_bound(key, it))
        for q in K.QNAMES:
            e = t[q]
            if e["kind"] == "fixed":
                continue
            flag = " **(at bound)**" if (f"{q}:lnA" in bound or f"{q}:lnQb" in bound) else ""
            if e["kind"] == "pl":
                A = pm(e["A"], max(e.get("A_err_hess", np.nan), e.get("A_err_boot", np.nan) if np.isfinite(e.get("A_err_boot", np.nan)) else 0)) + flag
                am = pm(e["alpha_m"], max(e.get("alpha_m_err_hess", 0), e.get("alpha_m_err_boot", 0) if np.isfinite(e.get("alpha_m_err_boot", np.nan)) else 0))
                az = pm(e["alpha_z"], max(e.get("alpha_z_err_hess", 0), e.get("alpha_z_err_boot", 0) if np.isfinite(e.get("alpha_z_err_boot", np.nan)) else 0))
            else:
                big = lambda k: max(e.get(k + "_err_hess", 0), e.get(k + "_err_boot", 0) if np.isfinite(e.get(k + "_err_boot", np.nan)) else 0)  # noqa: E731
                A = f"$Q_b$ = {pm(e['Q_b'], big('Q_b'))}" + flag
                am = f"{pm(e['alpha_m_low'], big('alpha_m_low'))} (low) / {pm(e['alpha_m_high'], big('alpha_m_high'))} (high)"
                az = pm(e["alpha_z"], big("alpha_z"))
            other = [n for n in bound if n.startswith(q + ":") and n not in (f"{q}:lnA", f"{q}:lnQb")]
            if other:
                am += " **(" + ", ".join(n.split(":")[1] for n in other) + " at bound)**"
            lines.append(f"| {label} | {QL[form][q]} | {A} | {am} | {az} | {fmt(it['sigma_d'])} |")
        if "log10_M_break" in it["table"]:
            extra = f"$\\log_{{10}}M_b$ = {pm(it['table']['log10_M_break'], it['table']['log10_M_break_err_hess'])}"
            if "alpha_break" in it["table"]:
                extra += f", $\\alpha_{{\\rm break}}$ = {pm(it['table']['alpha_break'], it['table']['alpha_break_err_hess'])}"
            lines.append(f"| {label} | break | {extra} | | | |")
    return "\n".join(lines)


def quality_table(fits):
    lines = ["| field | range | points / cells | published $\\sigma_d$ | orig | brk_amp | brk_all | free | ΔBIC vs orig (brk_amp / brk_all / free) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for field in ("P", "G"):
        for r in KC.RANGES:
            ds = fits["datasets"][f"{field}|{r}"]
            row = [field, r, f"{ds['n_points']} / {ds['n_cells']}", fmt(fits["published"][f"{field}|{r}"]["sigma_d"])]
            bic0 = fits["configs"].get(f"{field}|{r}|orig", {}).get("information", {}).get("BIC")
            dl = []
            for p in KC.PARAMETRIZATIONS:
                key = f"{field}|{r}|{p}"
                it = fits["configs"].get(key, {})
                nc = " (n.c.)" if key in NOT_CONVERGED else ""
                row.append(fmt(it.get("sigma_d")) + nc)
                if p != "orig" and bic0 is not None and it.get("information"):
                    dl.append(f"{it['information']['BIC'] - bic0:+.0f}" + nc)
            row.append(" / ".join(dl))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def electron_table(R):
    d = json.loads((R / "k4_electron_factors.json").read_text())
    lines = ["| aperture | $r$ ($10^{12.5}$–$10^{13.5}$) | $r$ ($10^{13.5}$–$10^{14.5}$) | $r$ ($\\ge 10^{14.5}$) | XGPaint multiplier $1/(0.9r)$ ($\\ge10^{13.5}$) |",
             "|---|---|---|---|---|"]
    for ap in ("SO/2500_crit", "SO/500_crit", "SO/200_crit", "SO/50_crit", "SO/5xR_500_crit", "InclusiveSphere/300kpc", "InclusiveSphere/1000kpc"):
        f = d["factors"].get(ap, {})
        cells = []
        for g in ("12.5-13.5", "13.5-14.5", ">=14.5"):
            e = f.get(g)
            cells.append(f"{e['median']:.4f} ({e['p16']:.3f}–{e['p84']:.3f})" if e else "–")
        e = f.get("13.5-14.5")
        lines.append(f"| {ap} | " + " | ".join(cells) + f" | {1 / (0.9 * e['median']):.3f} |" if e else f"| {ap} | " + " | ".join(cells) + " | – |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--narrative", default=str(HERE / "report" / "narrative.json"))
    a = ap.parse_args()
    R = Path(a.results)
    fits = json.loads((R / "k3_fits.json").read_text())
    nar = json.loads(Path(a.narrative).read_text()) if Path(a.narrative).exists() else {}
    out = HERE / "report"
    out.mkdir(exist_ok=True)
    tabP = law_rows(fits, "P", [f"P|{r}|orig" for r in KC.RANGES])
    tabG = law_rows(fits, "G", [f"G|{r}|orig" for r in KC.RANGES])
    tabP2 = law_rows(fits, "P", [f"P|{r}|{p}" for r in KC.RANGES for p in ("brk_amp", "brk_all", "free")] + [f"Pagn|{r}|orig" for r in KC.RANGES])
    tabG2 = law_rows(fits, "G", [f"G|{r}|{p}" for r in KC.RANGES for p in ("brk_amp", "brk_all", "free")])
    (out / "param_tables.md.txt").write_text("\n\n".join([tabP, tabG, tabP2, tabG2, quality_table(fits)]) + "\n")
    page = PAGE.format(
        summary=nar.get("summary", "Battaglia B12 pressure and B16 gas-density laws refitted to FLAMINGO L1_m9 SOAP-HBT aperture integrals at all 71 outputs."),
        in_short="\n".join(f"- {b}" for b in nar.get("in_short", [])),
        data=nar.get("data", ""), tabP=tabP, tabG=tabG, tabP2=tabP2, tabG2=tabG2, quality=quality_table(fits),
        electrons=electron_table(R), recommendations="\n".join(f"- {b}" for b in nar.get("recommendations", [])),
        limits="\n".join(f"- {b}" for b in nar.get("limits", [])), results_text=nar.get("results_text", ""),
        run=nar.get("run", ""), snippet=(HERE / "report" / "xgpaint_refit_snippet.jl").read_text() if (HERE / "report" / "xgpaint_refit_snippet.jl").exists() else "",
        quality_text=nar.get("quality_text", ""), profiles_text=nar.get("profiles_text", ""), trends_text=nar.get("trends_text", ""),
        apertures_text=nar.get("apertures_text", ""), electrons_text=nar.get("electrons_text", ""),
        changes="\n".join(f"- {b}" for b in nar.get("changes", [])),
        forward_note=nar.get("forward_note", ""), chi2_note=nar.get("chi2_note", ""), per_output_table=nar.get("per_output_table", ""))
    (out / "fit_soap_battaglia.txt").write_text(page)
    print("wrote", out / "fit_soap_battaglia.txt")


PAGE = r"""---
title: Battaglia parameters refitted to FLAMINGO (SOAP)
short: Battaglia refit (SOAP)
section: fitting
order: 7
summary: {summary}
---

## In short {{#in-short}}

{in_short}

## Data and selection {{#data}}

{data}

<details><summary>Haloes, cells and points per output (all 71 outputs)</summary>

Haloes = central haloes with $M_{{200c}}\ge10^{{12.7}}M_\odot$ (`results/k1_status.json` n_selected); cells = 0.2-dex mass bins with at least one halo, used = at least 10 haloes; points = aperture points of that output in each fit's dataset (`report/report_numbers.json` per_output_counts).

{per_output_table}

</details>

## Equations {{#equations}}

```equation id=eq-k-b12
title: B12 pressure profile (Battaglia et al. 2012, XGPaint convention)
latex: \begin{{gathered}}\frac{{P_e(x)}}{{P_{{200}}}}=0.5176\,P_0\left(\frac{{x}}{{x_c}}\right)^{{\gamma}}\left[1+\left(\frac{{x}}{{x_c}}\right)^{{\alpha}}\right]^{{-\beta}}\\ P_{{200}}=\frac{{G M_{{200c}}\,200\rho_c(z)\,f_b}}{{2R_{{200c}}}}\end{{gathered}}
meaning: The generalised NFW pressure profile. B12 fit it to the thermal pressure with $\alpha=1$ and $\gamma=-0.3$ fixed; the factor 0.5176 = $(2+2X_H)/(3+5X_H)$ converts thermal to electron pressure for a fully ionised gas with $X_H=0.76$, exactly as XGPaint's Battaglia16ThermalSZProfile does, so the fitted $P_0$, $x_c$, $\beta$ laws paste into XGPaint unchanged.
symbols:
- $x$ | $r/R_{{200c}}$, physical radius $r$ in units of $R_{{200c}}$ | dimensionless
- $r$ | physical distance from the halo centre | Mpc
- $P_e$ | electron pressure $n_e k_B T_e$ | Pa
- $P_{{200}}$ | self-similar thermal-pressure scale of the halo | Pa
- $P_0$ | amplitude (thermal pressure in units of $P_{{200}}$) | dimensionless
- $x_c$ | core radius | R200c
- $\alpha$ | transition sharpness (1 in B12) | dimensionless
- $\gamma$ | inner slope (-0.3 in B12) | dimensionless
- $\beta$ | outer exponent; $P_e\propto x^{{\gamma-\alpha\beta}}$ for $x\gg x_c$ | dimensionless
- $X_H$ | hydrogen mass fraction, 0.76 (enters only the constant 0.5176) | dimensionless
- $G$ | gravitational constant | m^3 kg^-1 s^-2
- $M_{{200c}}$ | mass inside $R_{{200c}}$ (SOAP SO/200_crit TotalMass) | Msun
- $R_{{200c}}$ | radius of the sphere of mean density $200\rho_c(z)$ (SOAP SO/200_crit SORadius, converted to physical) | Mpc
- $z$ | redshift of the SOAP output the halo belongs to | dimensionless
- $\rho_c(z)$ | critical density $3H(z)^2/(8\pi G)$ at redshift $z$ | kg m^-3
- $f_b$ | $\Omega_b/\Omega_m$ = 0.0486/0.306 = 0.15882 (FLAMINGO, $\Omega_m$ including massive neutrinos) | dimensionless
code: {rel}/k_fit_core.py
ref: Battaglia et al. (2012), ApJ 758, 75, eq. 10 and Table 1
```

```equation id=eq-k-b16
title: B16 gas-density profile (Battaglia 2016, XGPaint convention)
latex: \frac{{\rho_{{\rm gas}}(x)}}{{f_b\,\rho_c(z)}}=\rho_0\left(\frac{{x}}{{x_c}}\right)^{{\gamma}}\left[1+\left(\frac{{x}}{{x_c}}\right)^{{\alpha}}\right]^{{-(\beta+\gamma)/\alpha}}
meaning: The B16 density profile, with $x_c=0.5$ and $\gamma=-0.2$ fixed in the published fit. The paper's eq. A1 prints $-(\beta-\gamma)/\alpha$ and no $f_b$; the convention fitted here is XGPaint's BattagliaTauProfile (profiles_tau.jl): density in units of $f_b\rho_c(z)$ and exponent $-(\beta+\gamma)/\alpha$, so that the outer slope is $-\beta$. With it the published law gives $\tilde M(<R_{{200c}})$ = 0.70 at $10^{{14}}M_\odot$, $z=0$, i.e. a gas fraction of 0.70 $f_b$ inside $R_{{200c}}$.
symbols:
- $x$ | $r/R_{{200c}}$, physical radius $r$ in units of $R_{{200c}}$ | dimensionless
- $\rho_{{\rm gas}}$ | gas mass density (all gas, no electron conversion) | Msun Mpc^-3
- $\rho_0$ | amplitude in units of $f_b\rho_c(z)$ | dimensionless
- $x_c$ | scale radius (0.5 published) | R200c
- $\alpha$ | transition sharpness | dimensionless
- $\beta$ | outer slope: $\rho_{{\rm gas}}\propto x^{{-\beta}}$ for $x\gg x_c$ | dimensionless
- $\gamma$ | inner slope: $\rho_{{\rm gas}}\propto x^{{\gamma}}$ for $x\ll x_c$ (-0.2 published) | dimensionless
- $f_b$ | $\Omega_b/\Omega_m$ = 0.15882 (as in [](eq:eq-k-b12)) | dimensionless
- $\rho_c(z)$ | critical density $3H(z)^2/(8\pi G)$ at the redshift $z$ of the halo's SOAP output | Msun Mpc^-3
- $z$ | redshift of the SOAP output the halo belongs to | dimensionless
- $R_{{200c}}$ | radius of the sphere of mean density $200\rho_c(z)$ | Mpc
code: {rel}/k_fit_core.py
ref: Battaglia (2016), JCAP 08, 058, eq. A1, A2 and Table 2; XGPaint profiles_tau.jl
```

```equation id=eq-k-law
title: Mass and redshift laws
latex: \begin{{gathered}}Q=A\left(\frac{{M_{{200c}}}}{{10^{{14}}M_\odot}}\right)^{{\alpha_m}}(1+z)^{{\alpha_z}}\\ \ln Q=\ln Q_b+\begin{{cases}}\alpha_m^{{\rm low}} & (M_{{200c}}<M_b)\\ \alpha_m^{{\rm high}} & (M_{{200c}}\ge M_b)\end{{cases}}\ln\frac{{M_{{200c}}}}{{M_b}}+\alpha_z\ln(1+z)\\ Q_{{\rm amp}}\to Q_{{\rm amp}}\left(\frac{{M_{{200c}}}}{{M_b}}\right)^{{\alpha_{{\rm break}}}}\quad (M_{{200c}}<M_b)\end{{gathered}}
meaning: 'orig' gives each free profile parameter $Q$ the published power law (9 law parameters). 'brk_all' (Lee22 BPLall style) gives each of the three a broken power law in mass with a shared break mass $M_b$ ($\alpha_m^{{\rm low}}$ below, $\alpha_m^{{\rm high}}$ above) and its own $(1+z)$ law (13 parameters). 'brk_amp' keeps the orig laws and multiplies the amplitude $Q_{{\rm amp}}$ by $(M_{{200c}}/M_b)^{{\alpha_{{\rm break}}}}$ below $M_b$, which is XGPaint's BreakModel (11 parameters). 'free' adds a power law for $\gamma$ (B12, 12 parameters) or for $x_c$ and $\gamma$ (B16, 15 parameters). The sign of each $Q$ is fixed ($\gamma<0$, all others $>0$).
symbols:
- $Q$ | a profile parameter of [](eq:eq-k-b12) or [](eq:eq-k-b16) ($P_0$, $x_c$, $\beta$; $\rho_0$, $\alpha$, $\beta$; freed $\gamma$, $x_c$) | as Q
- $A$ | value of $Q$ at $M_{{200c}}=10^{{14}}M_\odot$ (no $h$) and $z=0$ | as Q
- $M_{{200c}}$ | halo mass inside $R_{{200c}}$ | Msun
- $z$ | redshift of the halo's SOAP output | dimensionless
- $\alpha_m$, $\alpha_z$ | mass and redshift exponents | dimensionless
- $M_b$ | break mass (shared by the three parameters in brk_all; XGPaint M_break in brk_amp) | Msun
- $Q_b$ | value of $Q$ at $M_{{200c}}=M_b$ and $z=0$ | as Q
- $\alpha_m^{{\rm low}}$, $\alpha_m^{{\rm high}}$ | mass exponents below and above $M_b$ | dimensionless
- $Q_{{\rm amp}}$ | the amplitude parameter: $P_0$ (B12) or $\rho_0$ (B16) | dimensionless
- $\alpha_{{\rm break}}$ | extra amplitude slope below $M_b$ (XGPaint BreakModel alpha_break) | dimensionless
code: {rel}/k_fit_core.py
```

```equation id=eq-k-data
title: Per-halo SOAP data in profile units
latex: \begin{{gathered}}\tilde Y(<r)=\frac{{Y(<r)\,m_ec^2/\sigma_T}}{{P_{{200}}R_{{200c}}^3}}=4\pi\int_0^{{x}}\frac{{P_e(x')}}{{P_{{200}}}}x'^2dx'\\ \tilde M(<r)=\frac{{M_{{\rm gas}}(<r)}}{{f_bM_{{200c}}}}=\frac{{3}}{{200}}\int_0^{{x}}\frac{{\rho_{{\rm gas}}(x')}}{{f_b\rho_c(z)}}x'^2dx'\end{{gathered}}
meaning: SOAP's spherical aperture integrals of every central halo, normalised so that each equals the radial integral of the profile in the units of [](eq:eq-k-b12) and [](eq:eq-k-b16). $Y$ is SO ComptonYWithoutRecentAGNHeating (ComptonY in the check fit); $M_{{\rm gas}}$ is SO or InclusiveSphere GasMass. The right-hand equalities use $M_{{200c}}=\tfrac{{4\pi}}{{3}}200\rho_c(z)R_{{200c}}^3$.
symbols:
- $r$ | aperture radius: SORadius$/(1+z)$ (SORadius is stored comoving) or 30-3000 physical kpc | Mpc
- $x$ | $r/R_{{200c}}$ | dimensionless
- $x'$ | integration variable (radius in units of $R_{{200c}}$) | dimensionless
- $Y(<r)$ | Compton-$y$ volume integral $\frac{{\sigma_T}}{{m_ec^2}}\int_{{<r}} P_e\,dV$ (SOAP, physical) | Mpc^2
- $m_e$ | electron mass | kg
- $c$ | speed of light | m s^-1
- $\sigma_T$ | Thomson cross-section | m^2
- $\tilde Y$, $\tilde M$ | normalised aperture integrals of pressure and gas mass | dimensionless
- $M_{{\rm gas}}(<r)$ | gas mass inside $r$ (all gas: SOAP SO and InclusiveSphere apertures are inclusive) | Msun
- $P_e$, $P_{{200}}$, $\rho_{{\rm gas}}$ | electron pressure, pressure scale and gas density of [](eq:eq-k-b12) and [](eq:eq-k-b16) | Pa, Pa, Msun Mpc^-3
- $M_{{200c}}$, $R_{{200c}}$ | mass and radius of the halo (SOAP SO/200_crit) | Msun, Mpc
- $f_b$ | cosmic baryon fraction 0.15882 | dimensionless
- $\rho_c(z)$ | critical density at the redshift $z$ of the SOAP output | Msun Mpc^-3
- $z$ | redshift of the halo's SOAP output | dimensionless
code: {rel}/k_read.py
```

```equation id=eq-k-forward
title: Forward model of a cell mean
latex: \begin{{gathered}}d_{{c,a}}=\ln\Big[\frac{{1}}{{N_{{c,a}}}}\sum_{{h\in c}}\tilde F_{{h,a}}\Big]\\ m_{{c,a}}(\theta)=\ln\Big[\sum_{{k=1}}^{{32}}w_k\,\mathcal{{I}}\big(x_{{k,a}};\,Q(M_k,z_c;\theta)\big)\Big]\end{{gathered}}
meaning: The data point of cell $c$ and aperture $a$ is the log of the inclusion-free mean over all central haloes of the cell with that aperture valid. The model integrates the 3D profile inside every halo's own aperture radius at that halo's own mass; the haloes are represented by 32 quadrature nodes (4 mass quantiles, each split into 8 radius quantiles; node = stratum mean of $\ln M_{{200c}}$ and $\ln x$, weight = stratum share). {forward_note} $\mathcal{{I}}$ is the closed-form incomplete-beta integral of the profile in the variable $t=s/(1+s)$, $s=(x/x_c)^\alpha$, evaluated with $\ln t$ and $\ln(1-t)=-\ln(1+s)$ passed separately so that it stays exact for large $s$ (checked against 40-digit quadrature for $\alpha$ = 0.1-30 and $x/x_c$ up to 400, and at the nodes of every fitted law; `verification.json`).
symbols:
- $c$ | cell index: one 0.2-dex bin of $\log_{{10}}M_{{200c}}$ at one SOAP output | –
- $a$ | aperture index (SO 2500c ... 5xR500c, physical-kpc spheres) | –
- $h$ | halo index; the sum runs over the $N_{{c,a}}$ central haloes of cell $c$ for which aperture $a$ is valid | –
- $k$ | quadrature-node index, 1 to 32 (fewer when the cell has fewer haloes: then one node per halo) | –
- $d_{{c,a}}$, $m_{{c,a}}$ | data and model (natural log) | dimensionless
- $\tilde F_{{h,a}}$ | $\tilde Y$ (pressure fits) or $\tilde M$ (gas fits) of halo $h$ in aperture $a$ ([](eq:eq-k-data)) | dimensionless
- $N_{{c,a}}$ | number of haloes of cell $c$ with aperture $a$ valid | count
- $w_k$ | node weight: share of those haloes in stratum $k$ | dimensionless
- $M_k$ | node mass: $\exp$ of the stratum mean of $\ln M_{{200c}}$ | Msun
- $x_{{k,a}}$ | node aperture radius: $\exp$ of the stratum mean of $\ln x$ (each halo's own aperture-$a$ radius over its own $R_{{200c}}$) | R200c
- $z_c$ | redshift of the SOAP output of cell $c$ | dimensionless
- $Q(M,z;\theta)$ | the profile parameters given by the laws of [](eq:eq-k-law) at mass $M$ and redshift $z$ | as Q
- $\mathcal{{I}}(x;Q)$ | $4\pi\int_0^x(P_e/P_{{200}})x'^2dx'$ or $\frac{{3}}{{200}}\int_0^x(\rho_{{\rm gas}}/f_b\rho_c)x'^2dx'$ for the profile with parameters $Q$ | dimensionless
- $\theta$ | the law parameters of [](eq:eq-k-law) (9 to 15) | various
code: {rel}/k_fit_core.py
```

```equation id=eq-k-chi2
title: Likelihood and model-discrepancy floor
latex: \begin{{gathered}}\chi^2(\theta)=\sum_c \mathbf{{r}}_c^{{\mathsf T}}\left[C_c+(\sigma_d^2+\sigma_0^2)\,\mathbf{{1}}\right]^{{-1}}\mathbf{{r}}_c\\ \mathbf{{r}}_c=\mathbf{{d}}_c-\mathbf{{m}}_c(\theta),\qquad \frac{{\chi^2_{{\min}}}}{{N_{{\rm pts}}-n_\theta}}=1\ \text{{defines}}\ \sigma_d\end{{gathered}}
meaning: Each cell's apertures are correlated (cumulative integrals of the same haloes). $C_c$ is the bootstrap covariance of the cell's log means (200 halo resamples), its correlation shrunk towards the redshift-pooled correlation by $\lambda=\min(1,\max(0.05,p/N))$. $\sigma_d$ is a common model-discrepancy floor, solved iteratively so that the reduced $\chi^2$ is 1; with thousands of haloes per cell it dominates, so the fit is close to an equal-weight least-squares in log aperture integrals. {chi2_note} Models are ranked with $\mathrm{{BIC}}=\chi^2(\theta;\sigma_{{\rm ref}})+n_\theta\ln N_{{\rm pts}}$, with $\sigma_{{\rm ref}}$ the orig fit's $\sigma_d$ of the same field and range.
symbols:
- $\chi^2(\theta)$ | goodness of fit of the law parameters $\theta$ | dimensionless
- $c$ | cell index (as in [](eq:eq-k-forward)) | –
- $\mathbf{{d}}_c$, $\mathbf{{m}}_c(\theta)$ | vectors of the cell's $p$ data points $d_{{c,a}}$ and model values $m_{{c,a}}$ | dimensionless
- $\mathbf{{r}}_c$ | residual vector of cell $c$ | dimensionless
- $\mathbf{{r}}_c^{{\mathsf T}}$ | transpose of $\mathbf{{r}}_c$ (a row vector) | dimensionless
- $C_c$ | $p\times p$ bootstrap covariance of the cell's log means | dimensionless
- $\mathbf{{1}}$ | $p\times p$ identity matrix | dimensionless
- $\sigma_d$ | model-discrepancy floor (rms model error per aperture point) | ln units
- $\sigma_0$ | numerical floor 0.002 (quadrature, 9-bit mantissa) | ln units
- $\chi^2_{{\min}}$ | $\chi^2$ at the best-fitting $\theta$ | dimensionless
- $N_{{\rm pts}}$ | number of data points of the fit (all cells) | count
- $n_\theta$ | number of law parameters (9 to 15) | count
- $\lambda$ | shrinkage weight of the pooled correlation | dimensionless
- $p$, $N$ | points in the cell, haloes in the cell | count
- $\sigma_{{\rm ref}}$ | the orig fit's $\sigma_d$ used for all BIC values of one field and range | ln units
code: {rel}/k_fit_core.py
```

## Refitted parameters {{#parameters}}

Values are in XGPaint's argument convention (mass in $M_\odot$, no $h$; pivot $10^{{14}}M_\odot$). Errors are the larger of the Hessian error (with $\sigma_d$) and the bootstrap error (halo resamples refitted at fixed $\sigma_d$: 100 for orig, 20 for brk_amp, brk_all and free, 10 for the checks); **(at bound)** marks a law parameter at its fit bound. Both errors are listed separately in `results/k3_fits.json` and `results/k_params.json`.

**Pressure (B12), published form ('orig'):**

{tabP}

**Gas density (B16), published form ('orig'):**

{tabG}

<details><summary>Mass-break and free-shape variants, and the pressure check with recently AGN-heated gas</summary>

{tabP2}

{tabG2}

</details>

{results_text}

## Fit quality and what each range adds {{#quality}}

{quality}

n.c. = not converged: the least-squares solve stopped at its iteration limit, so the value is not the $\chi^2/\nu=1$ floor (see the bullets below).

{quality_text}

![Model-discrepancy floor per field, fit range and parametrisation.](fig:{rel}/figures/fig_K_sigma_d.png){{script={rel}/make_figures.py}}

## Profiles at fixed masses {{#profiles}}

{profiles_text}

![Pressure: $x^3P_e/P_{{200}}$ at six masses and five redshifts.](fig:{rel}/figures/fig_K_profiles_P.png){{script={rel}/make_figures.py}}

![Gas density: $x^3\rho/(f_b\rho_c)$ at six masses and five redshifts.](fig:{rel}/figures/fig_K_profiles_G.png){{script={rel}/make_figures.py}}

![Refitted over published profile.](fig:{rel}/figures/fig_K_ratio_to_published.png){{script={rel}/make_figures.py}}

## Parameter trends with mass and redshift {{#trends}}

{trends_text}

![All five profile parameters of the published, orig R3, recommended and alternative laws vs mass at $z$ = 0, 1, 2; thick lines inside the SOAP data, thin lines are extrapolations.](fig:{rel}/figures/fig_K_law_params.png){{script={rel}/make_figures.py}}

![B12 parameters vs mass.](fig:{rel}/figures/fig_K_params_vs_mass_P.png){{script={rel}/make_figures.py}}
![B16 parameters vs mass.](fig:{rel}/figures/fig_K_params_vs_mass_G.png){{script={rel}/make_figures.py}}

![B12 parameters vs z.](fig:{rel}/figures/fig_K_params_vs_z_P.png){{script={rel}/make_figures.py}}
![B16 parameters vs z.](fig:{rel}/figures/fig_K_params_vs_z_G.png){{script={rel}/make_figures.py}}

## Residuals and aperture integrals {{#residuals}}

{apertures_text}

![Aperture integrals: law / SOAP.](fig:{rel}/figures/fig_K_aperture_integrals.png){{script={rel}/make_figures.py}}

![Pressure residuals per aperture.](fig:{rel}/figures/fig_K_residuals_P.png){{script={rel}/make_figures.py}}

![Gas-mass residuals per aperture.](fig:{rel}/figures/fig_K_residuals_G.png){{script={rel}/make_figures.py}}

## Electrons for DM and kSZ {{#electrons}}

{electrons_text}

{electrons}

## Using the parameters {{#usage}}

```julia
{snippet}
```

The machine-readable file `{rel}/results/k_params.json` holds every fit (law parameters, Hessian covariance, bootstrap standard deviations, $\sigma_d$, data range and `fit_status`: converged, usable). `{rel}/k_fit_core.py` is data-source agnostic: the later full-particle run can pass per-halo aperture integrals (kind 0) or per-halo shell profiles (kind 1) to `summarise_cell`, build a `Dataset` and call `fit_global` with the same `Law`, which gives parameters directly comparable with these, parameter by parameter.

## Known limits {{#limits}}

{limits}

## Recommendations {{#recommendations}}

{recommendations}

## Run record and verification {{#run}}

{run}

## Changes from earlier versions {{#changes}}

{changes}
""".replace("{rel}", REL)

if __name__ == "__main__":
    main()
