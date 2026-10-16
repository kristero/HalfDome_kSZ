"""Narrative of the Test K report (report/narrative.json), with every number read from the results files:
report/report_numbers.json (k_numbers.py), results/k3_fits.json, results/k6_run2_vs_run3.json, verification.json,
CLUSTER_RUN.json and report/recommendation.json (the recommended fits, chosen after reading the results).

  python -B report/write_narrative.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

KC_ORDER = ["SO/2500_crit", "SO/1000_crit", "SO/500_crit", "SO/200_crit", "SO/100_crit", "SO/BN98", "SO/200_mean", "SO/50_crit", "SO/5xR_500_crit"]

K = Path(__file__).resolve().parent.parent
R = Path(os.environ.get("K_NARR_RESULTS", K / "results"))           # overrides only for dry runs
N = json.loads(Path(os.environ.get("K_NARR_NUMBERS", K / "report" / "report_numbers.json")).read_text())
F = json.loads((R / "k3_fits.json").read_text())
V = json.loads(Path(os.environ.get("K_NARR_VERIFICATION", K / "verification.json")).read_text())
CR = json.loads((K / "CLUSTER_RUN.json").read_text())
REC = json.loads(Path(os.environ.get("K_NARR_REC", K / "report" / "recommendation.json")).read_text())
OUT = Path(os.environ.get("K_NARR_OUT", K / "report" / "narrative.json"))
RC = json.loads((R / "k6_run2_vs_run3.json").read_text()) if (R / "k6_run2_vs_run3.json").exists() else {"configs": {}}
EF = json.loads((R / "k4_electron_factors.json").read_text())["factors"]
PS = json.loads((R / "k5_parity_local.json").read_text())
SN = json.loads(Path(os.environ.get("K_NARR_SNIPPET", K / "report" / "xgpaint_snippet_check.json")).read_text())
K1 = json.loads((R / "k1_provenance_check.json").read_text())
RCPT = json.loads((R / "test_K_receipt.json").read_text()) if (R / "test_K_receipt.json").exists() else {}
AU = json.loads((R / "k7_fit_audit.json").read_text())
NC = [k for k, c in AU["convergence"].items() if not c["converged"]]          # fits whose least squares stopped at max_nfev


def v(key):
    return N[key]["value"]


def sd(key):
    return F["configs"][key]["sigma_d"]


def f(x, nd=3):
    return f"{x:.{nd}f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def sgn(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "−")


def ap(field, aperture, model, regime="all"):
    return v(f"aperture_ln_law_over_soap|{field}|{aperture}|{model}")[regime]


def model_tag(key):
    _, r, p = key.split("|")
    return f"fit_{r}_{p}"


def name(key):
    fld, r, p = key.split("|")
    return f"{'pressure' if fld == 'P' else 'gas'} {r} {p}"


def rng(vals, nd=3):
    return f"{min(vals):.{nd}f}–{max(vals):.{nd}f}"


def law_A(key, q):
    return F["configs"][key]["table"]["params"][q]


def ratio(key, lm, z, x):
    return v(f"profile_ratio_refit_over_published|{key}")[f"logM{lm:g}|z{z:g}"][f"x{x:g}"]


def lawval(key, q, lm, z):
    d = v(f"law_values|{key}")
    return d[q][d["log10_M200c"].index(lm)][d["z"].index(z)]


RANGES = ("R1", "R2", "R3")
PARS = ("orig", "brk_amp", "brk_all", "free")
recP, recG = REC["tSZ"]["key"], REC["kSZ_DM"]["key"]
altP, altG = REC["tSZ_alt"]["key"], REC["kSZ_DM_alt"]["key"]
nparP, nparG = F["configs"][recP]["n_par"], F["configs"][recG]["n_par"]
cov = v("coverage_top_cell_by_z")
r200 = {g: EF["SO/200_crit"][g]["median"] for g in ("12.5-13.5", "13.5-14.5", ">=14.5")}
mult = {g: 1 / (0.9 * r) for g, r in r200.items()}
inv = json.loads((R / "k0_inventory.json").read_text())
soap_TB = sum(fv["bytes"] for fv in inv["files"].values()) / 1e12
pub_sd = {fld: [v(f"published_sigma_d|{fld}|{r}") for r in RANGES] for fld in ("P", "G")}
orig_sd = {fld: [sd(f"{fld}|{r}|orig") for r in RANGES] for fld in ("P", "G")}
other_sd = {fld: [sd(f"{fld}|{r}|{p}") for r in RANGES for p in PARS[1:] if f"{fld}|{r}|{p}" not in NC] for fld in ("P", "G")}
gdom = v(f"law_domain|{recG}")
alpha_g = {(lm, z): lawval(recG, "alpha", lm, z) for lm in (13.0, 14.0, 14.5, 15.0, 15.3) for z in (0.0, 0.5, 1.0, 2.0)}
xc_g = {(lm, z): lawval(recG, "xc", lm, z) for lm in (13.0, 14.0, 14.5, 15.0, 15.3) for z in (0.0, 0.5, 1.0, 2.0)}
gam_g = {(lm, z): lawval(recG, "gamma", lm, z) for lm in (13.0, 14.0, 14.5, 15.0, 15.3) for z in (0.0, 0.5, 1.0, 2.0)}
bet_g = {(lm, z): lawval(recG, "beta", lm, z) for lm in (13.0, 14.0, 14.5, 15.0, 15.3) for z in (0.0, 0.5, 1.0, 2.0)}
gq = F["configs"][recG]["table"]["params"]


def law_str(e):
    return f"{abs(e['A']):.3g} (M/10^14)^{e['alpha_m']:.3f} (1+z)^{e['alpha_z']:.3f}"


KINK = (f"The fitted transition sharpness grows steeply with mass and redshift, $\\alpha$ = {law_str(gq['alpha'])}: "
        f"{alpha_g[(13.0, 0.0)]:.2f} at $10^{{13}}$, {alpha_g[(14.0, 0.0)]:.2f} at $10^{{14}}$, {alpha_g[(14.5, 0.0)]:.1f} at $10^{{14.5}}$, "
        f"{alpha_g[(15.0, 0.0)]:.1f} at $10^{{15}}$ and {alpha_g[(15.3, 0.0)]:.1f} at $10^{{15.3}}M_\\odot$ ($z=0$), and {alpha_g[(14.5, 1.0)]:.1f} at "
        f"$10^{{14.5}}$, $z=1$; over the fitted cells it spans {gdom['alpha'][0]:.2f}–{gdom['alpha'][1]:.1f} (`report/report_numbers.json` law_values, law_domain). "
        f"For $\\alpha\\gtrsim10$ the profile is in effect a broken power law, $\\rho\\propto x^{{\\gamma}}$ inside $x_c$ and $x^{{-\\beta}}$ outside "
        f"(at $10^{{15}}$, $z=0$: $x_c$ = {xc_g[(15.0, 0.0)]:.3f}, $\\gamma$ = {gam_g[(15.0, 0.0)]:.3f}, $\\beta$ = {bet_g[(15.0, 0.0)]:.3f}), "
        "with a sharp kink at $x_c$ in 3D (smoothed in projection). The aperture integrals cannot resolve how sharp the transition is, so treat the kink "
        "as a fitting artefact of the power-law form, not as a physical feature.")

CF = AU["compression_k1_at_fit_points"]; LR = AU["linear_response"]; CFF = AU["compression_fitted"]
lrmax = max(LR, key=lambda k_: LR[k_]["max_abs_shift_hess_sigma"])
big1 = sorted({(b["cell"], b["aperture"], b["N"]) for c in CF.values() for b in c["points_gt_1pct"]})
big_cells = sorted({(b["cell"], b["N"]) for c in CF.values() for b in c["points_gt_1pct"]}, key=lambda t: t[1])
sdlx = [b["sd_lnx"] for c in CF.values() for b in c["points_gt_1pct"]]
APL = {"SO/2500_crit": "$R_{2500c}$", "SO/1000_crit": "$R_{1000c}$", "SO/500_crit": "$R_{500c}$", "SO/200_crit": "$R_{200c}$"}
big_aps = " and ".join(APL.get(a_, a_) for a_ in sorted({a_ for _, a_, _ in big1}, key=lambda a_: KC_ORDER.index(a_) if a_ in KC_ORDER else 99))
cP, cG = CF["P|R3"]["max_of_both"], CF["G|R3"]["max_of_both"]
cmax = max(c["max_of_both"]["max"] for c in CF.values())
COMP_SHORT = (f"The 32-node forward model reproduces the exact mean over every halo of a cell with a 99th-percentile error of {pct(cP['p99'], 2)} (pressure) and "
              f"{pct(cG['p99'], 2)} (gas) at the R3 fit points and at most {pct(cmax)}; the {cP['n_gt_1pct']} points per field above 1% are the inner "
              f"apertures of {len(big_cells)} small cells ($N$ = {', '.join(str(n) for _, n in big_cells)}). Propagated through the fits, this error shifts no law parameter by more "
              f"than {LR[lrmax]['max_abs_shift_hess_sigma']:.2f}σ (`results/k7_fit_audit.json`).")
FORWARD_NOTE = (f"Against the exact mean over every halo, the median error of the nodes at the R3 fit points is {CF['P|R3']['published']['median']:.0e} (pressure) and "
                f"{CF['G|R3']['published']['median']:.0e} (gas) for the published law. Taking the larger of the published-law and perturbed-law "
                f"errors, the 99th percentile is {pct(cP['p99'], 2)} (pressure) and {pct(cG['p99'], 2)} (gas) and the maximum {pct(cmax, 2)}: {cP['n_gt_0.2pct']} of "
                f"{cP['n']:,} pressure and {cG['n_gt_0.2pct']} of {cG['n']:,} gas points exceed 0.2%, and {cP['n_gt_1pct']} of each exceed 1%. The points above 1% are the "
                f"{big_aps} apertures of {len(big_cells)} cells with {min(n for _, n in big_cells)}–{max(n for _, n in big_cells)} haloes "
                f"({', '.join(c_ for c_, _ in big_cells)}), where sd($\\ln x$) = {min(sdlx):.2f}–{max(sdlx):.2f}. For the fitted laws the error is similar "
                f"(99th percentile ≤ {pct(max(v_['all']['p99'] for v_ in CFF.values()), 2)}, maximum {pct(max(v_['all']['max'] for v_ in CFF.values()), 1)}). "
                f"Propagated linearly through each fit, it shifts no law parameter by more than {LR[lrmax]['max_abs_shift_hess_sigma']:.2f} Hessian σ "
                f"({lrmax}, {LR[lrmax]['param']}) (`results/k7_fit_audit.json`).")
nc0 = AU["convergence"][NC[0]] if NC else None
dev = max(abs(c["chi2_nu_at_reported_sigma_d"] - 1) for c in AU["convergence"].values() if c["converged"])
NC_TEXT = (f"Its least-squares solve stopped at the iteration limit of {nc0['max_nfev']} evaluations in all {len(nc0['iterations'])} $\\sigma_d$ iterations "
           f"(scipy status 0) and again in the final solve, so $\\chi^2/\\nu$ = {nc0['chi2_nu_at_reported_sigma_d']:.4f} "
           f"at its quoted $\\sigma_d$ = {nc0['sigma_d_reported']:.4f}; the floor that gives $\\chi^2/\\nu=1$ at its final $\\theta$ is "
           f"{nc0['sigma_d_chi2nu1_at_final_theta']:.4f}." if NC else "")
CHI2_NOTE = (f"Every converged fit has $\\chi^2/\\nu=1$ to within {dev:.1e} at its quoted $\\sigma_d$ (`results/k7_fit_audit.json` convergence)" +
             (f"; {', '.join(NC)} did not converge. " + NC_TEXT if NC else "."))

out = {}
out["forward_note"] = FORWARD_NOTE
out["chi2_note"] = CHI2_NOTE
out["summary"] = ("The Battaglia B12 pressure and B16 gas-density laws refitted to FLAMINGO L1_m9 (HYDRO_FIDUCIAL) SOAP-HBT "
                  f"aperture integrals of {v('haloes_total'):,} central-halo entries at all 71 outputs, forward-modelled at every "
                  "halo's own radii; three fit ranges and four parametrisations, compared with the published laws, the SOAP data "
                  "and the z = 0 particle stacks (Test K, run 3).")

# ------------------------------------------------------------------------------------------------ in short
ins = []
ins.append(f"**Data.** All 71 fiducial SOAP-HBT outputs (0007–0077, $z$ = 6.04 to 0; {soap_TB:.2f} TB) were read on STRW maas: "
           f"{v('haloes_total'):,} central-halo entries with $M_{{200c}}\\ge10^{{12.7}}\\,M_\\odot$ ({v('haloes_z0')[0]:,} at $z=0$). "
           f"In 0.2-dex mass bins per output they form {v('cells_total')} cells; the {v('cells_used')} cells with at least 10 haloes are used, "
           f"at {v('outputs_with_used_cells')} outputs from $z$ = 0 to {v('z_max_used'):.1f} (`results/k1_status.json`, `report/report_numbers.json` cells_used).")
P500 = ap("P", "SO/500_crit", "published", "z<1,M<10^13.5"); G200 = ap("G", "SO/200_crit", "published", "z<1,M<10^13.5")
ins.append(f"**The published laws fail for FLAMINGO groups.** Over all cells they need a model-discrepancy floor $\\sigma_d$ = "
           f"{rng(pub_sd['P'])} (B12) and {rng(pub_sd['G'])} (B16) in ln per aperture point (`results/k3_fits.json` published). "
           f"For $z<1$ and $M_{{200c}}<10^{{13.5}}$ the mean ln(law/SOAP) is {sgn(P500['mean'])} for $Y(<R_{{500c}})$ "
           f"(a factor {math.exp(P500['mean']):.1f}) and {sgn(G200['mean'])} for $M_{{\\rm gas}}(<R_{{200c}})$ (a factor {math.exp(G200['mean']):.1f}); "
           f"for clusters ($z<1$, $\\ge10^{{13.5}}$) the published B12 has ln rms {f(ap('P', 'SO/500_crit', 'published', 'z<1,M>=10^13.5')['rms'])} in $Y(<R_{{500c}})$; "
           f"at $z\\ge1$ the published B16 underpredicts $M_{{\\rm gas}}(<R_{{200c}})$ (mean {sgn(ap('G', 'SO/200_crit', 'published', 'z>=1')['mean'])}) "
           "(`report/report_numbers.json` aperture_ln_law_over_soap).")
ins.append(f"**Refitted published form ('orig').** $\\sigma_d$ = {', '.join(f(x, 4) for x in orig_sd['P'])} (pressure, R1/R2/R3) and "
           f"{', '.join(f(x, 4) for x in orig_sd['G'])} (gas). The converged mass-break and free-shape variants reach {rng(other_sd['P'], 4)} (pressure) and "
           f"{rng(other_sd['G'], 4)} (gas); {', '.join(NC)} did not converge and is not usable (see 'Fit quality'). No law of the Battaglia family reaches the few-per-cent level over the whole mass and redshift range: "
           "the per-cell core radius depends non-monotonically on mass and the group gas evolves quickly at $z<0.5$ (per-cell fits, residual maps).")
y5 = ap("P", "SO/500_crit", model_tag(recP)); y2 = ap("P", "SO/200_crit", model_tag(recP)); y55 = ap("P", "SO/5xR_500_crit", model_tag(recP))
ins.append(f"**Recommended for tSZ: {name(recP)}** ({REC['tSZ']['xgpaint']}, {nparP} parameters, $\\sigma_d$ = {f(sd(recP), 4)}). Over all "
           f"{y5['n']} cells the ln rms of law/SOAP is {f(y5['rms'])} for $Y(<R_{{500c}})$, {f(y2['rms'])} for $Y(<R_{{200c}})$ and "
           f"{f(y55['rms'])} for $Y(<5R_{{500c}})$; the median law/SOAP is {v('aperture_ratio|Y500|' + model_tag(recP))['median']:.3f} at $R_{{500c}}$ and "
           f"{v('aperture_ratio|Y200|' + model_tag(recP))['median']:.3f} at $R_{{200c}}$.")
g2 = ap("G", "SO/200_crit", model_tag(recG)); g55 = ap("G", "SO/5xR_500_crit", model_tag(recG)); g1000 = ap("G", "IS/1000kpc", model_tag(recG))
g25 = ap("G", "SO/2500_crit", model_tag(recG))
ins.append(f"**Recommended for kSZ and DM: {name(recG)}** ({REC['kSZ_DM']['xgpaint']}, {nparG} parameters, $\\sigma_d$ = {f(sd(recG), 4)}): "
           f"ln rms {f(g2['rms'])} for $M_{{\\rm gas}}(<R_{{200c}})$, {f(g55['rms'])} for $M_{{\\rm gas}}(<5R_{{500c}})$, {f(g1000['rms'])} for "
           f"$M_{{\\rm gas}}(<1000\\,{{\\rm kpc}})$ and {f(g25['rms'])} at $R_{{2500c}}$. Caveat: its $\\alpha$ reaches {alpha_g[(15.0, 0.0)]:.0f} at $10^{{15}}M_\\odot$, $z=0$ (a kink at $x_c$; see 'Reading the parameters'); {name(altG)} ($\\sigma_d$ = {f(sd(altG), 4)}, smooth, not an XGPaint type) and gas R3 orig ($\\sigma_d$ = {f(sd('G|R3|orig'), 4)}, published form) are the smooth alternatives. For electrons multiply the amplitude by "
           f"$1/(0.9r)$ = {min(mult.values()):.3f}–{max(mult.values()):.3f}.")
rowsP = [(lm, [ratio(recP, lm, 0.0, x) for x in (0.1, 0.5, 1.0, 2.0)]) for lm in (13.0, 14.0, 15.0)]
rowsG = [(lm, [ratio(recG, lm, 0.0, x) for x in (0.1, 0.5, 1.0, 2.0)]) for lm in (13.0, 14.0, 15.0)]
tab = ["| field | $M_{200c}$ | $x$ = 0.1 | 0.5 | 1 | 2 |", "|---|---|---|---|---|---|"]
for lab, rows in (("pressure", rowsP), ("gas", rowsG)):
    for lm, vals in rows:
        tab.append(f"| {lab} | $10^{{{lm:g}}}$ | " + " | ".join(f"{x:.3f}" for x in vals) + " |")
ins.append(f"**Profile change at $z=0$, recommended law / published**, at exactly $x=r/R_{{200c}}$ = 0.1, 0.5, 1, 2 "
           "(`report/report_numbers.json` profile_ratio_refit_over_published):\n\n" + "\n".join("  " + t for t in tab))
r1x = ap("P", "SO/5xR_500_crit", "fit_R1_orig")["mean"]
b2 = ap("P", "SO/5xR_500_crit", "fit_R2_brk_amp")["rms"]; b3 = ap("P", "SO/5xR_500_crit", "fit_R3_brk_amp")["rms"]
ins.append(f"**Fit ranges.** Laws fitted within $R_{{200c}}$ (R1) do not extrapolate: pressure orig R1 overpredicts $Y(<5R_{{500c}})$ by "
           f"$e^{{{r1x:.2f}}}$ = {math.exp(r1x):.2f} on average. R2 and R3 agree inside $R_{{200c}}$; R3 is needed for 2–3.3 $R_{{200c}}$ "
           f"(pressure brk_amp: $Y(<5R_{{500c}})$ ln rms {f(b2)} fitted to R2, {f(b3)} fitted to R3). R3 includes neighbour gas (inclusive apertures).")
ig = V["numbers"]["integral_at_fitted_nodes"]
ins.append(f"**Checks** (`verification.json`): the aperture integral agrees with 40-digit quadrature to {PS['integral']['max_rel_err']:.1e} over "
           f"$\\alpha$ = 0.1–30 and $x/x_c$ up to 400 ({PS['integral']['n']} cases), and with adaptive quadrature at the nodes of every fitted law to "
           f"{max(d['max_rel_err'] for d in ig.values()):.1e} (largest $\\ln[(x/x_c)^\\alpha]$ = {max(d['max_ln_s_in_dataset'] for d in ig.values()):.1f}); "
           f"Python and XGPaint agree to {max(PS['xgpaint']['max_rel_diff'].values()):.1e}; the XGPaint snippet runs with the validated XGPaint "
           f"and reproduces the fitted profiles to {SN['max_rel_diff']:.1e}; SOAP per-halo values equal the particle pass at $z=0$ to "
           f"{PS['particles']['max_abs_median_dev_above_13.5']:.2%} in the median ($\\ge10^{{13.5}}$); a re-read of 3 outputs with the final code "
           f"reproduces every array of the fitted cell files bit for bit (except the stored quadrature-error diagnostics, ≤ 1.1e-15). " + COMP_SHORT)
ins.append("**This version corrects the first report** (section 'Changes'): the closed-form integral was wrong for "
           "$(x/x_c)^\\alpha\\gtrsim10^{12}$ in its $b<0.02$ branch, which corrupted the free gas fits; the gas sphere cuts are now "
           "$\\bar x\\le1.8$ and 3.3 as decided (1.9 and 3.6 before); all fits were rerun on maas.")
out["in_short"] = ins

# ------------------------------------------------------------------------------------------------ data
po = v("per_output_counts"); por = po["rows"]; pot = po["totals"]
no_used = [r_ for r_ in por if r_["cells_used"] == 0]
PER_OUTPUT_BULLET = (f"- **Per output** (table 'Haloes, cells and points per output' below; `report/report_numbers.json` per_output_counts): the {pot['cells_used']} used cells hold "
                     f"{pot['haloes_in_used_cells']:,} of the {v('haloes_total'):,} selected halo entries; "
                     + ", ".join(f"output {r_['snap']} ($z$ = {r_['z']:g}, {r_['haloes_selected']} haloes)" for r_ in no_used)
                     + f" {'has' if len(no_used) == 1 else 'have'} no cell with ≥ 10 haloes. Used cells per output range from "
                     f"{min(r_['cells_used'] for r_ in por if r_['cells_used'])} to {max(r_['cells_used'] for r_ in por)}.")
tab_po = ["| output | $z$ | haloes ($\\ge10^{12.7}$) | cells: all / used | haloes in used cells | used $\\log_{10}M_{200c}$ | pressure points R1 / R2 / R3 | gas points R1 / R2 / R3 |",
          "|---|---|---|---|---|---|---|---|"]
for r_ in por:
    mr = f"{r_['used_log10M_range'][0]:.1f}–{r_['used_log10M_range'][1]:.1f}" if r_["used_log10M_range"] else "–"
    tab_po.append(f"| {r_['snap']} | {r_['z']:g} | {r_['haloes_selected']:,} | {r_['cells_all']} / {r_['cells_used']} | {r_['haloes_in_used_cells']:,} | {mr} | "
                  f"{r_['points|P|R1']} / {r_['points|P|R2']} / {r_['points|P|R3']} | {r_['points|G|R1']} / {r_['points|G|R2']} / {r_['points|G|R3']} |")
tab_po.append(f"| **all** | 0–6.04 | {v('haloes_total'):,} | {pot['cells_all']} / {pot['cells_used']} | {pot['haloes_in_used_cells']:,} | | "
              f"{pot['points|P|R1']:,} / {pot['points|P|R2']:,} / {pot['points|P|R3']:,} | {pot['points|G|R1']:,} / {pot['points|G|R2']:,} / {pot['points|G|R3']:,} |")
out["per_output_table"] = "\n".join(tab_po)
dsP = {r: F["datasets"][f"P|{r}"] for r in RANGES}; dsG = {r: F["datasets"][f"G|{r}"] for r in RANGES}
c25 = v("cells_with_R2500c|P|R3")
covline = "; ".join(f"$z={zt[1:]}$: {c['top_cell']} ({c['N']} haloes)" for zt, c in cov.items() if c.get("top_cell") and zt in ("z0", "z0.5", "z1", "z2", "z3", "z4", "z5"))
sph = {r: v(f"sphere_points|G|{r}") for r in RANGES}
out["data"] = "\n".join([
    f"- **Files.** `/net/hypernova/data2/FLAMINGO/L1000N1800/HYDRO_FIDUCIAL/SOAP-HBT/halo_properties_00{{07..77}}.hdf5`, 71 files, {soap_TB:.2f} TB, read-only on maas (`results/k0_inventory.json`); every file has every column used.",
    f"- **Haloes.** Central subhaloes (InputHalos/IsCentral = 1) with SO/200_crit TotalMass $\\ge10^{{12.7}}\\,M_\\odot$ at every output. The same halo appears at many outputs, so {v('haloes_total'):,} counts halo entries, not distinct haloes.",
    "- **Apertures.** SO 2500c, 1000c, 500c, 200c, 100c, BN98, 200m, 50c and 5xR500c (ComptonY, ComptonYWithoutRecentAGNHeating, GasMass, NumberOfGasParticles, SORadius) and InclusiveSphere 10–3000 physical kpc (GasMass, NumberOfGasParticles). SORadius is stored comoving and converted with its 'including cosmological corrections' factor (times the scale factor $1/(1+z)$).",
    f"- **Units check.** The SO radii are physical: $M_{{200c}}/(\\tfrac43\\pi R_{{200c}}^3)$ equals $200\\rho_c(z)$ from the header $H(z)$ to within {max(abs(x - 1) for x in V['numbers']['rho200_over_rhocrit_H_median_range']):.1e} (median per output, every output).",
    f"- **Normalisation.** $f_b=\\Omega_b/\\Omega_m$ = 0.0486/0.306 = {V['numbers']['f_b_header_range'][0]:.5f} (header, $\\Omega_m$ with massive neutrinos, as in Tests B–J); $P_{{200}}$ per halo from its own $M_{{200c}}$, $R_{{200c}}$ ([](eq:eq-k-data)).",
    f"- **Cells.** 0.2-dex bins of $\\log_{{10}}M_{{200c}}$ from 12.7, one set per output; a cell is used with at least 10 haloes ({v('cells_used')} of {v('cells_total')}). The data point is the inclusion-free mean over all valid centrals of the cell; the forward model represents the haloes by 32 strata (below).",
    "- **Aperture masks.** Per halo, an aperture is valid if it is filled (SORadius > 0; spheres need ≥ 100 bound particles). Per cell, it is used if ≥ 90% of the haloes have it, the median gas-particle count is ≥ 50 and its mean radius is ≥ 0.025 $R_{200c}$. The particle cut acts on the cell median (a cut on each halo would select gas-rich groups); every valid halo enters the mean. Spheres need a radius ≥ 22.8 kpc (4 softening lengths; drops 10 kpc). The gas fit uses spheres only from 100 kpc and mean $x\\ge0.1$ (inside, the central galaxy's cold gas dominates; the 'Gcore' check keeps them). Within a cell an aperture whose mean $x$ is within 5% of one already kept is dropped (priority 200c, 500c, 2500c, 1000c, 5xR500c, 50c, 100c, 200m, BN98, then spheres).",
    f"- **Fit ranges.** R1: SO 2500c, 1000c, 500c, 200c and spheres with cell-mean $x\\le1$; R2: adds 100c, BN98, 200m, 50c and spheres to $x\\le1.8$; R3: adds 5xR500c and spheres to $x\\le3.3$. The largest sphere mean $x$ in the gas datasets is {sph['R1']['max_xbar']:.3f} / {sph['R2']['max_xbar']:.3f} / {sph['R3']['max_xbar']:.3f} (R1/R2/R3; `report/report_numbers.json` sphere_points).",
    f"- **Points per fit** (`results/k3_fits.json` datasets): pressure {dsP['R1']['n_points']:,} / {dsP['R2']['n_points']:,} / {dsP['R3']['n_points']:,} (R1/R2/R3), gas {dsG['R1']['n_points']:,} / {dsG['R2']['n_points']:,} / {dsG['R3']['n_points']:,}, over {dsP['R3']['n_cells']} cells. $R_{{2500c}}$ enters {c25['with']} of the {c25['of']} cells.",
    PER_OUTPUT_BULLET,
    f"- **Redshift coverage** (highest mass bin with ≥ 10 haloes at the output nearest $z$; `report/report_numbers.json` coverage_top_cell_by_z): {covline}. The laws are extrapolations outside this domain.",
])

# ------------------------------------------------------------------------------------------------ results text
co = F["configs"]["P|R3|orig"]; cn = co["names"]; cc = co["corr"]
i_xc, i_b, i_a = cn.index("xc:lnA"), cn.index("beta:lnA"), cn.index("amp:lnA")
tP = co["table"]["params"]
ba = F["configs"]["P|R3|brk_amp"]["table"]
k2x = v("k2_z0_xc_vs_mass|P|R2")
k2ok = [r for r in k2x if not r[2]]
xc_lo, xc_hi = k2ok[0], k2ok[-1]
xc_max = max(k2ok, key=lambda r: r[1])
tG = F["configs"][recG]["table"]["params"]
out["results_text"] = "\n".join([
    f"**Reading the parameters.** $P_0$, $x_c$ and $\\beta$ are strongly degenerate on aperture integrals: in the pressure orig R3 fit corr($\\ln x_c$, $\\ln\\beta$) = {cc[i_xc][i_b]:.3f} and corr($\\ln P_0$, $\\ln x_c$) = {cc[i_a][i_xc]:.3f} (`results/k3_fits.json` corr). The refitted laws move along the degeneracy ($P_0$ = {tP['amp']['A']:.2f} at $10^{{14}}M_\\odot$ instead of 18.1, $x_c$ = {tP['xc']['A']:.3f} instead of 0.497, $\\beta$ = {tP['beta']['A']:.2f} instead of 4.35) while the profiles inside the data change much less. Compare profiles or aperture integrals, not single parameters.",
    "",
    f"**Mass dependence.** The refitted amplitude laws are steep: $P_0\\propto M^{{{tP['amp']['alpha_m']:.3f}}}$ (orig R3; published $M^{{0.154}}$); brk_amp R3 has $M^{{{ba['params']['amp']['alpha_m']:.3f}}}$ above and $M^{{{ba['params']['amp']['alpha_m'] + ba['alpha_break']:.3f}}}$ below $M_b=10^{{{ba['log10_M_break']:.2f}}}M_\\odot$. This is the gas depletion of FLAMINGO groups. The per-cell fits (published shape, three free parameters) give $x_c$ = {xc_lo[1]:.3f} at $10^{{{xc_lo[0]:.2f}}}$, a maximum {xc_max[1]:.3f} at $10^{{{xc_max[0]:.2f}}}$ and {xc_hi[1]:.3f} at $10^{{{xc_hi[0]:.2f}}}$ at $z=0$ (R2, cells not at a bound; `report/report_numbers.json` k2_z0_xc_vs_mass), a trend no single power law follows.",
    "",
    f"**Gas shape ({name(recG)}).** At $10^{{14}}M_\\odot$ and $z=0$: $\\rho_0$ = {tG['amp']['A']:.1f}, $x_c$ = {abs(tG['xc'].get('A', tG['xc'].get('value', float('nan')))):.3f}, $\\alpha$ = {abs(tG['alpha']['A']):.3f}, $\\beta$ = {tG['beta']['A']:.3f}, $\\gamma$ = {tG['gamma'].get('A', tG['gamma'].get('value', float('nan'))):.3f} (published 4000, 0.5, 0.88, 3.83, −0.2): FLAMINGO gas is cuspier inside and much more extended outside than B16. " + KINK + f" The orig gas fits reach a similar shape with a very broad transition instead ($\\alpha$ = {abs(F['configs']['G|R2|orig']['table']['params']['alpha']['A']):.3f} and {abs(F['configs']['G|R3|orig']['table']['params']['alpha']['A']):.3f} at $10^{{14}}$ in R2 / R3" + ("; at its lower bound in R1" if "alpha:lnA" in (F['configs']['G|R1|orig'].get('at_bound') or []) else "") + ").",
    "",
    f"**Recently AGN-heated gas** (check, ComptonY instead of ComptonYWithoutRecentAGNHeating): the orig fits change $P_0$ by {', '.join(sgn(100 * (F['configs'][f'Pagn|{r}|orig']['table']['params']['amp']['A'] / F['configs'][f'P|{r}|orig']['table']['params']['amp']['A'] - 1), 1) + '%' for r in RANGES)} (R1/R2/R3) and $\\sigma_d$ by at most {max(abs(sd(f'Pagn|{r}|orig') - sd(f'P|{r}|orig')) for r in RANGES):.4f}.",
    "",
    f"**Central-galaxy gas** (check 'Gcore', spheres from 30 kpc and $x\\ge0.025$): $\\sigma_d$ = {', '.join(f(sd(f'Gcore|{r}|orig'), 4) for r in RANGES)} (R1/R2/R3)" + (f"; in R1 the amplitude is at its bound" if "amp:lnA" in (F['configs']['Gcore|R1|orig'].get('at_bound') or []) else "") + ". The B16 form cannot describe the cold central gas and the hot halo together.",
])

# ------------------------------------------------------------------------------------------------ quality
spread = {}
for key, c in F["configs"].items():
    vals = [x["sigma_d"] for x in c.get("starts", []) if "sigma_d" in x]
    if len(vals) > 1:
        spread[key] = max(vals) - min(vals)
multi = sorted((k for k, d in spread.items() if d > 1e-4), key=lambda k: -spread[k])
nerr = sum(1 for c in F["configs"].values() for x in c.get("starts", []) if "error" in x)
STARTS_NOTE = (f"Different starts reached different optima (σ_d spread > 1e-4) in {len(multi)} of {len(spread)} fits"
               + (f" (largest: {', '.join(f'{k} {spread[k]:.4f}' for k in multi[:4])})" if multi else "")
               + f"; {nerr} start(s) were infeasible (a start vector whose $\\gamma$ law reaches $\\le-3$ at some node gives an infinite model and is rejected).")
sdb = {fld: v(f"sd_boot|{fld}|R3") for fld in ("P", "G")}
bh = [v(f"boot_over_hess|{k}")["median"] for k in F["configs"] if f"boot_over_hess|{k}" in N]
order = {}
for fld in ("P", "G"):
    for r in RANGES:
        b0 = F["configs"][f"{fld}|{r}|orig"]["information"]["BIC"]
        order[(fld, r)] = sorted(((F["configs"][f"{fld}|{r}|{p}"]["information"]["BIC"] - b0, p) for p in PARS if f"{fld}|{r}|{p}" not in NC), key=lambda t: t[0])
info = {fld: v(f"information|{fld}|orig") for fld in ("P", "G")}
shiftP = info["P"]
mx12 = max(abs(x) for x in shiftP["shift_sigma_R1_R2"].values()); mx23 = max(abs(x) for x in shiftP["shift_sigma_R2_R3"].values())
pfree = [F["configs"][f"P|{r}|free"] for r in RANGES]
out["quality_text"] = "\n".join([
    f"- $\\sigma_d$ is the rms model discrepancy per aperture point in ln (the floor that makes $\\chi^2/\\nu=1$). The statistical (bootstrap) errors of the cell means are much smaller: median {pct(sdb['P']['median'], 2)} (pressure) and {pct(sdb['G']['median'], 2)} (gas), 90th percentile {pct(sdb['P']['p90'])} and {pct(sdb['G']['p90'])}, up to {pct(sdb['P']['max'])} and {pct(sdb['G']['max'])} in the smallest high-mass cells (R3 datasets; `report/report_numbers.json` sd_boot). ΔBIC is evaluated at the orig fit's $\\sigma_d$ of the same field and range.",
    "- **Ranking by BIC** (best first; converged fits only): " + "; ".join(f"{fld} {r}: " + " < ".join(p for _, p in order[(fld, r)]) for fld in ("P", "G") for r in RANGES) + "."
    + (f" {', '.join(NC)} is left out: it did not converge (below)." if NC else ""),
    f"- **Pressure free fits.** $x_c$ at $10^{{14}}$ = {', '.join(f(c['table']['params']['xc']['A'], 2) for c in pfree)} (R1/R2/R3) and $\\gamma$ = {', '.join(f(c['table']['params']['gamma']['A'], 3) for c in pfree)}; at a bound: " + "; ".join(f"R{i + 1}: {', '.join(c.get('at_bound') or []) or 'none'}" for i, c in enumerate(pfree)) + ". With $x_c$ beyond most of the data they act as a near power law in $x$ inside $R_{200c}$; they interpolate well but should not be extrapolated.",
    "- **Parameters at a fit bound** (within 1e-3 in the fitted coordinate; flagged in the tables): " + "; ".join(f"{k}: {', '.join(v(f'near_bound|{k}'))}" for k in F["configs"] if v(f"near_bound|{k}")) + f". Gas R1 free is degenerate: $x_c$ at its upper bound 10 at $10^{{14}}$ and $\\gamma$ down to {V['numbers']['integral_at_fitted_nodes']['G|R1|free']['min_gamma']:.2f} at the nodes; it is not usable.",
    *([f"- **Not converged: {', '.join(NC)}.** " + NC_TEXT + f" Every other fit reaches $\\chi^2/\\nu=1$ to within {dev:.1e}. Its 20 bootstrap refits also stopped at their iteration limit "
       f"(status 0 in {F['configs'][NC[0]].get('boot_status_counts', {}).get('0', 0)} of {F['configs'][NC[0]].get('boot_n_ok')}). Its $\\sigma_d$ and ΔBIC in the table (marked n.c.) are therefore not converged values, and "
       "`results/k_params.json` marks it `usable: false`. Being degenerate as well, it was not refitted."] if NC else []),
    f"- **Information per range** (orig, $\\tfrac12\\Delta\\ln\\det$ of the Hessian covariance, `results/k4_information.json`): pressure R1→R2 {info['P']['info_gain_bits_R1_to_R2']:.2f} bits, R2→R3 {info['P']['info_gain_bits_R2_to_R3']:.2f} bits; gas {info['G']['info_gain_bits_R1_to_R2']:.2f} and {info['G']['info_gain_bits_R2_to_R3']:.2f} bits. The pressure orig parameters shift by up to {mx12:.1f}σ from R1 to R2 but by at most {mx23:.1f}σ from R2 to R3.",
    f"- **Uncertainties.** Hessian errors include $\\sigma_d$ as if the discrepancies were independent; bootstrap errors (halo resamples refitted at fixed $\\sigma_d$: 100 for orig, 20 for brk_amp / brk_all / free, 10 for the checks) measure halo sampling only and are {min(bh):.2f}–{max(bh):.2f} times the Hessian errors (median ratio per fit). The tables quote the larger; both are in `results/k3_fits.json` and `results/k_params.json`.",
    "- **Multiple starts.** Every fit was started from the published law or the orig fit (three break masses for the brk laws) and from the run-2 optimum; the free gas fits also from the verifier's corrected refit. The lowest $\\sigma_d$ wins; every start's result is in `results/k3_fits.json` (starts). " + STARTS_NOTE,
])

# ------------------------------------------------------------------------------------------------ profiles
rmP = v("particles_z0_A2ii_removed_range|P|12.9-14.7"); rmG = v("particles_z0_A2ii_removed_range|G|12.9-14.7")
out["profiles_text"] = "\n".join([
    "The figures show $x^3P_e/P_{200}$ and $x^3\\rho/(f_b\\rho_c)$ (the $x^3$ weighting shows where the volume-integrated signal sits) at $M_{200c}=10^{12.8}$–$10^{15}M_\\odot$ and $z$ = 0, 0.5, 1, 2, 3. Lines: the published law (grey dashed), the refitted orig laws of the three ranges, the recommended law and one alternative. Points: SOAP per-halo shells between consecutive apertures (each halo's own radii, mean over the cell, plotted at the mean radii; the innermost cumulative is not shown) from the cell containing the mass at the output nearest $z$; the label gives the cell's $N$ and $z$. At $z=0$: the study's particle stacks (all gas solid black, A2-ii dotted) and Test I's per-bin fits to A2-ii ($R_{\\rm fit}=4$; gas = Test I's electron fit divided by $\\kappa_E=1.094$). Where no SOAP cell exists the curves are extrapolations.",
    "",
    f"At $z=0$, in the 0.2-dex bins from $10^{{12.9}}$ to $10^{{14.7}}$, the A2-ii partition removes {pct(rmP['2-3'][0])}–{pct(rmP['2-3'][1])} of the all-gas pressure at 2–3 $R_{{200c}}$ and {pct(rmP['3-4'][0])}–{pct(rmP['3-4'][1])} at 3–4 $R_{{200c}}$; for the gas density {pct(rmG['2-3'][0])}–{pct(rmG['2-3'][1])} and {pct(rmG['3-4'][0])}–{pct(rmG['3-4'][1])} (`report/report_numbers.json` particles_z0_A2ii_removed_range). The inclusive SOAP apertures, and therefore the R3 laws, contain that gas.",
])
out["trends_text"] = ("The first figure shows all five profile parameters of the published, orig R3, recommended and alternative laws vs mass at $z$ = 0, 1, 2 "
                      "(thick inside the SOAP data, thin where extrapolated). The others show the per-cell free fits (stage K2: three parameters per cell, "
                      "published fixed shape, bootstrap covariance only; open symbols = a parameter at its bound) and the refitted orig law of each range. "
                      "With three parameters on 3–4 apertures, R1 cell fits are often degenerate (" +
                      f"{v('k2_cells|P|R1')['at_bound']} of {v('k2_cells|P|R1')['n']} pressure cells and {v('k2_cells|G|R1')['at_bound']} of {v('k2_cells|G|R1')['n']} gas cells at a bound); "
                      f"in R3, {v('k2_cells|P|R3')['at_bound']} pressure and {v('k2_cells|G|R3')['at_bound']} gas cells are at a bound (`report/report_numbers.json` k2_cells).")

# ------------------------------------------------------------------------------------------------ apertures table
def row(lab, fld, aper, model, mlab):
    d = v(f"aperture_ln_law_over_soap|{fld}|{aper}|{model}")
    cells = [f"{d[k]['rms']:.3f} / {sgn(d[k]['mean'])}" if k in d else "–" for k in ("all", "z<1,M>=10^13.5", "z<1,M<10^13.5", "z>=1")]
    return f"| {lab} | {mlab} | " + " | ".join(cells) + " |"


tP_ = model_tag(recP); tG_ = model_tag(recG)
apt = ["| quantity | law | all cells: rms / mean | $z<1$, $\\ge10^{13.5}$ | $z<1$, $<10^{13.5}$ | $z\\ge1$ |", "|---|---|---|---|---|---|"]
for lab, fld, aper in (("$Y(<R_{500c})$", "P", "SO/500_crit"), ("$Y(<R_{200c})$", "P", "SO/200_crit"), ("$Y(<5R_{500c})$", "P", "SO/5xR_500_crit")):
    for model, mlab in (("published", "published"), ("fit_R3_orig", "orig R3"), (tP_, name(recP).replace("pressure ", ""))):
        apt.append(row(lab, fld, aper, model, mlab))
for lab, fld, aper in (("$M_{\\rm gas}(<R_{200c})$", "G", "SO/200_crit"), ("$M_{\\rm gas}(<5R_{500c})$", "G", "SO/5xR_500_crit"), ("$M_{\\rm gas}(<R_{2500c})$", "G", "SO/2500_crit")):
    for model, mlab in (("published", "published"), ("fit_R3_orig", "orig R3"), (tG_, name(recG).replace("gas ", "")), (model_tag(altG), name(altG).replace("gas ", ""))):
        apt.append(row(lab, fld, aper, model, mlab))
p25 = ap("P", "SO/2500_crit", tP_)["rms"]
out["apertures_text"] = "\n".join([
    "The aperture-integral figure compares $Y(<R_{500c})$, $Y(<R_{200c})$ and $M_{\\rm gas}(<R_{200c})$ predicted by each law (at every halo's own radius and mass) with the SOAP cell mean; the grey band is the cells' bootstrap error. ln(law/SOAP), rms / mean over the cells, R3 datasets (`report/report_numbers.json` aperture_ln_law_over_soap):",
    "",
    *apt,
    "",
    f"The core is the weakest part: at $R_{{2500c}}$ the recommended pressure law has ln rms {p25:.3f} and the recommended gas law {g25['rms']:.3f}. The residual maps show coherent redshift structure in the inner apertures that a $(1+z)^{{\\alpha_z}}$ law cannot follow.",
])

# ------------------------------------------------------------------------------------------------ electrons
out["electrons_text"] = ("SOAP stores gas mass, not electrons. The factors below are the particle-measured ratio $r$ = [$M_{\\rm gas}/\\mu_e$, fully ionised, $X=0.76$] / "
                         "$N_e$(particles) at $z=0$ at the same radius (median and 16–84% over the study haloes; SOAP audit, `results/k4_electron_factors.json`). "
                         "Electrons = (gas law / $\\mu_e$) / $r$. XGPaint's BattagliaTauProfile paints $n_e=0.9\\rho_{\\rm gas}/\\mu_e$, so to paint the particle "
                         f"electrons with a refitted gas law multiply its $\\rho_0$ amplitude by $1/(0.9r)$: {mult['12.5-13.5']:.3f} ($10^{{12.5}}$–$10^{{13.5}}$), "
                         f"{mult['13.5-14.5']:.3f} ($10^{{13.5}}$–$10^{{14.5}}$), {mult['>=14.5']:.3f} ($\\ge10^{{14.5}}$) at $R_{{200c}}$. "
                         "The factors are measured at $z=0$ only; neutral and star-forming gas dominate the difference in cores below 0.1 $R_{200c}$.")

# ------------------------------------------------------------------------------------------------ limits
out["limits"] = [
    f"**Inclusive apertures.** SOAP SO apertures and InclusiveSpheres count all gas in the sphere, including satellites and neighbouring haloes. Beyond about 2 $R_{{200c}}$ the R3 laws contain neighbour gas that the study's A2-ii painting partition assigns to other haloes (at $z=0$, $10^{{12.9}}$–$10^{{14.7}}$: pressure {pct(rmP['2-3'][0])}–{pct(rmP['2-3'][1])} at 2–3 $R_{{200c}}$ and {pct(rmP['3-4'][0])}–{pct(rmP['3-4'][1])} at 3–4 $R_{{200c}}$; gas {pct(rmG['2-3'][0])}–{pct(rmG['2-3'][1])} and {pct(rmG['3-4'][0])}–{pct(rmG['3-4'][1])}). Painting every halo with an R3 law double-counts that gas.",
    "**The core.** Pressure inside about 0.27 $R_{200c}$ is constrained only by the $R_{2500c}$ integral (every halo forward-modelled at its own radius, so the +2–3% bias of stacking at the mean radius does not enter); the gas core only by spheres ≥ 100 kpc with $x\\ge0.1$ and $R_{2500c}$. Inner slopes of the free fits below about 0.1 $R_{200c}$ are extrapolations. " + f"The recommended gas law has $\\gamma$ = {gdom['gamma'][0]:.2f} to {gdom['gamma'][1]:.2f} over the fitted cells ({gam_g[(13.0, 0.0)]:.2f} at $10^{{13}}$, {gam_g[(14.0, 0.0)]:.2f} at $10^{{14}}$, {gam_g[(13.0, 2.0)]:.2f} at $10^{{13}}$, $z=2$): $\\gamma<-1$ gives a projected cusp.",
    f"**Few gas particles at low mass.** $R_{{2500c}}$ is unusable (mostly the median-50-gas-particle cut) in {c25['of'] - c25['with']} of {c25['of']} cells, all at low mass; those cells use only the outer apertures.",
    "**Physical-kpc spheres at high $z$.** $R_{200c}$ shrinks with $z$ at fixed mass, so a sphere of fixed physical radius moves outward in $x$; the 30–50 kpc spheres reach $x\\ge0.1$ at high $z$ but are still dominated by central-galaxy gas, hence the 100-kpc floor for the gas fit.",
    f"**Domain.** Cells reach {cov['z0']['top_cell']} at $z=0$, {cov['z1']['top_cell']} at $z=1$, {cov['z2']['top_cell']} at $z=2$ and {cov['z3']['top_cell']} at $z=3$. Clusters at high $z$ are extrapolated, and the laws differ from each other there; the recommended gas law's $\\alpha$ grows to {alpha_g[(15.0, 1.0)]:.0f} at $10^{{15}}$, $z=1$ and {alpha_g[(15.0, 2.0)]:.0f} at $z=2$ outside the data.",
    f"**Form.** Power laws in $M$ and $(1+z)$ cannot follow the non-monotonic $x_c(M)$ or the late-time evolution of group gas: the best $\\sigma_d$ is {min(other_sd['P'] + orig_sd['P']):.4f} (pressure) and {min(other_sd['G'] + orig_sd['G']):.4f} (gas). Equal weight per aperture point means the fits are dominated by the many group and $z>1$ cells, not by the clusters that dominate the tSZ power.",
    "**Uncertainties.** The quoted errors treat the model discrepancy as independent per point; the bootstrap measures halo sampling only (20 refits for the brk and free variants).",
    "**Single simulation.** Only L1_m9 HYDRO_FIDUCIAL; the feedback variants have SOAP-HBT at the same outputs and can be fitted with the same code.",
]
po = ap("P", "SO/500_crit", "fit_R3_orig"); poc = ap("P", "SO/500_crit", "fit_R3_orig", "z<1,M>=10^13.5")
cl_ratio = [ratio(recP, lm, 0.0, x) for lm in (14.0, 15.0) for x in (1.0, 2.0)]
gas_ratio = [ratio(recG, lm, 0.0, x) for lm in (14.0, 15.0) for x in (1.0, 2.0)]
out["recommendations"] = [
    f"**tSZ:** paint with **{name(recP)}** ({REC['tSZ']['xgpaint']}; `report/xgpaint_refit_snippet.jl`; $\\sigma_d$ = {f(sd(recP), 4)}). "
    f"If only Battaglia16ThermalSZProfile can be used, take **pressure R3 orig** ($\\sigma_d$ = {f(sd('P|R3|orig'), 4)}; $Y(<R_{{500c}})$ ln rms {f(po['rms'])} over all cells, "
    f"{f(poc['rms'])} for clusters at $z<1$). Pressure R3 free has a lower $\\sigma_d$ ({f(sd('P|R3|free'), 4)}) but puts $x_c$ beyond the data at low mass "
    f"({lawval('P|R3|free', 'xc', 13.0, 0.0):.1f} $R_{{200c}}$ at $10^{{13}}$), so it is a near power law that should not be extrapolated. Do not use R1 laws beyond $R_{{200c}}$; "
    "R2 laws are as good inside $R_{200c}$ but extrapolate worse to 3.3 $R_{200c}$.",
    f"**kSZ and FRB DM:** the best-fitting law is **{name(recG)}** (BattagliaTauProfilePhysical with power laws for all five parameters; $\\sigma_d$ = {f(sd(recG), 4)}), "
    f"with the amplitude multiplied by $1/(0.9r)$ for electrons ({mult['12.5-13.5']:.3f} below $10^{{13.5}}$, {mult['13.5-14.5']:.3f} for $10^{{13.5}}$–$10^{{14.5}}$, "
    f"{mult['>=14.5']:.3f} above). Its transition sharpness $\\alpha$ exceeds 10 above about $10^{{14.5}}M_\\odot$ at $z=0$ ({alpha_g[(14.5, 0.0)]:.1f} at $10^{{14.5}}$, "
    f"{alpha_g[(15.0, 0.0)]:.1f} at $10^{{15}}$) and at lower masses at higher $z$, i.e. a kink at $x_c$; if a smooth profile is required, use **{name(altG)}** "
    f"($\\sigma_d$ = {f(sd(altG), 4)}; not an XGPaint type, evaluate with `k_fit_core.Law`) or **gas R3 orig** ($\\sigma_d$ = {f(sd('G|R3|orig'), 4)}, published form, "
    "XGPaint-ready). A free-shape variant with a constant $\\alpha$ would remove the kink; it was not part of this run.",
    f"**Change from the published parameters:** for groups the published laws do not describe FLAMINGO (at $z<1$, $<10^{{13.5}}$: factor {math.exp(P500['mean']):.1f} too much "
    f"$Y(<R_{{500c}})$ and {math.exp(G200['mean']):.1f} too much $M_{{\\rm gas}}(<R_{{200c}})$); for clusters at $z<1$ the published B12 is within "
    f"{100 * ap('P', 'SO/500_crit', 'published', 'z<1,M>=10^13.5')['rms']:.0f}% rms in $Y(<R_{{500c}})$, but the recommended pressure law puts "
    f"{min(cl_ratio):.2f}–{max(cl_ratio):.2f} times the published pressure at 1–2 $R_{{200c}}$ for $10^{{14}}$–$10^{{15}}M_\\odot$ ($z=0$), and the recommended gas law "
    f"{min(gas_ratio):.2f}–{max(gas_ratio):.2f} times the published gas density there. Expect the tSZ and kSZ one-halo terms to drop where groups dominate and to change "
    "by tens of per cent at intermediate masses; quantify with the HalfDome painter before adopting.",
    "**Neighbour gas:** if the painting target is the A2-ii partition (Tests H–J), use the R3 laws inside 2 $R_{200c}$ and correct 2–4 $R_{200c}$ with the particle "
    "A2-ii / all-gas ratios, or refit A2-ii shells from the full-particle run with the same code.",
    "**Full-particle run:** pass per-halo shell profiles (kind 1) or aperture integrals (kind 0) to `k_fit_core.summarise_cell`, build the `Dataset` with the same "
    "cells and `Law` kinds and fit with `fit_global`; compare profiles and aperture integrals (`results/k4_profile_ratios.json`, `results/k4_profiles.npz`) and the "
    f"per-cell fits rather than single parameters ($P_0$, $x_c$ and $\\beta$ are degenerate: correlation up to {max(abs(cc[i_xc][i_b]), abs(cc[i_a][i_xc])):.3f} in pressure R3 orig).",
    "**Website:** the page source `K_soap_battaglia/report/fit_soap_battaglia.txt` (site content syntax) builds with `build_site.py --check` in a scratch copy "
    "of the site; it is installed into `stacking_choices_20261014/site/content/` by the workflow's site step (this test may write only in its own directory).",
]


# ------------------------------------------------------------------------------------------------ run record
runs = CR["maas_runs"]
lines = ["- **maas runs** (`CLUSTER_RUN.json` maas_runs; nice 19, ionice 7, the wrapper's load and memory guards; read-only on /net/hypernova):"]
for r_ in runs:
    lines.append(f"  - attempt {r_['attempt']} ({r_['kind']}; {r_.get('workers')} workers): {r_.get('outcome', '')}" + (f". {r_['note']}" if r_.get("note") else ""))
lines.append(f"- **K1 provenance:** {K1['check_run']['snaps'] if K1.get('check_run') else ''} re-read with the final code "
             f"(k_read.py {K1['check_run']['code_sha256']['k_read.py'][:12]}…, k_common.py {K1['check_run']['code_sha256']['k_common.py'][:12]}…): every array of the cell "
             "files used by the fits is bit-identical except the stored quadrature-error diagnostics comp_err_* (≤ 1.1e-15 absolute, from the corrected integral) "
             "(`results/k1_provenance_check.json`).")
lines.append("- **Local audit (this revision):** `k_fit_audit.py` → `results/k7_fit_audit.json`. It covers convergence and the compression error at the fit points, "
             f"exact in the cells with ≤ 256 haloes and a 256-halo proxy above, with its linear effect on every fit. It reproduces the stored published-law diagnostic to "
             f"{max(x['check_published_law_vs_stored_comp_err_pub_max_abs_diff'] for x in CFF.values()):.0e}. "
             "`k_params_status.py` adds `fit_status` and `electron_factors` to `results/k_params.json` (`results/k7_params_provenance.json`). "
             "No maas job was started for this revision.")
lines.append("- **Local:** K4 comparisons (`k_compare.py`), figures (`make_figures.py`), K5 parity (`k_parity.py`, XGPaint through the validated Julia 1.12.2 checkout), "
             "snippet check (`k_snippet_check.py`), run-2 comparison (`k_run_compare.py`), verification (`verify_k.py`), report (`k_numbers.py`, `report/write_narrative.py`, `report_k.py`).")
out["run"] = "\n".join(lines)

# ------------------------------------------------------------------------------------------------ changes from run 2
ex = {(e["alpha"], e["beta"]): e["old_over_new_minus_1"] for e in RC.get("old_integral_example_B16", [])}
nd = RC.get("run2_free_gas_thetas_at_run3_nodes", {})
def _e(a, b):
    x = ex.get((a, b))
    return "inf" if x is None else f"{100 * x:+.1f}%"
OLDINT = (f"at $x/x_c$ = 13.75 (B16, $\\gamma$ = −0.55) the old value was off by {_e(11.6, 2.8)}, {_e(11.6, 2.95)}, {_e(11.6, 3.01)} and {_e(11.6, 3.2)} "
          f"for $\\beta$ = 2.8, 2.95, 3.01, 3.2 at $\\alpha$ = 11.6, but only {min(abs(ex[(10.0, b)]) for b in (2.8, 2.95, 3.01, 3.2)):.0e}–{max(abs(ex[(10.0, b)]) for b in (2.8, 2.95, 3.01, 3.2)):.0e} at $\\alpha$ = 10. "
          + (f"At the run-2 free gas laws it was wrong by more than 1e-6 at {nd['G|R3|free']['nodes_rel_err_gt_1e-6']} of {nd['G|R3|free']['nodes']:,} quadrature nodes (R3; up to "
             f"{100 * nd['G|R3|free']['max_rel_err_finite_nodes']:.0f}%) and {nd['G|R2|free']['nodes_rel_err_gt_1e-6']} of {nd['G|R2|free']['nodes']:,} (R2; up to {100 * nd['G|R2|free']['max_rel_err_finite_nodes']:.0f}%), "
             f"and the run-2 gas R1 free law had $\\gamma\\le-3$ (an infinite core integral) at {nd['G|R1|free']['nodes_gamma_le_-3']} nodes (minimum {nd['G|R1|free']['min_gamma']:.2f}) "
             "(`results/k6_run2_vs_run3.json`)." if nd else ""))
ch = []
for key in ("G|R2|free", "G|R3|free", "G|R1|free", "P|R3|brk_amp", "P|R3|orig", "G|R3|orig"):
    c = RC["configs"].get(key)
    if c:
        ch.append(f"{key}: $\\sigma_d$ {c['sigma_d_run2']:.4f} → {c['sigma_d_run3']:.4f}, largest parameter shift {c['max_abs_shift_hess_sigma']:.1f}σ ({c['param_of_max_shift']}; run-3 Hessian σ)")
PV = json.loads((R / "k7_params_provenance.json").read_text())
lrP, lrG = LR[recP]["max_abs_shift_hess_sigma"], LR[recG]["max_abs_shift_hess_sigma"]
out["changes"] = [
    "**Revision after the second verification (local only; no new maas run, no fitted number changed).** "
    f"(1) The forward-model accuracy is now quoted correctly. The page used to say the 32 nodes reproduce the all-halo mean 'to within 0.2 per cent'; that "
    f"bound is wrong at the fit points, where the 99th percentile is {pct(cP['p99'], 2)} / {pct(cG['p99'], 2)} and the maximum {pct(cmax, 2)}. The new "
    f"`k_fit_audit.py` also propagates the error through every fit: at most {LR[lrmax]['max_abs_shift_hess_sigma']:.2f}σ, and {lrP:.2f}σ / {lrG:.2f}σ for the "
    "recommended fits. "
    f"(2) {', '.join(NC)} is marked as not converged (n.c.) in the tables and figure, left out of the BIC ranking and flagged `usable: false` in "
    "`results/k_params.json`. "
    "(3) A table of haloes, cells and points per output was added. "
    "(4) Every equation now defines all its symbols. "
    f"(5) `results/k_params.json` gained a `fit_status` entry per fit and the `electron_factors` block its B16 convention refers to. All maas fields are unchanged; "
    f"the maas file is kept as `{PV['maas_copy']}`, and the results/SHA256SUMS line was updated (`results/k7_params_provenance.json`).",
    "**Aperture integral.** `k_fit_core.cumulative` formed $1-t$ from $t=s/(1+s)$, $s=(x/x_c)^\\alpha$, in its $b<0.02$ branch (outer slope shallower than $x^{-3}$), which loses all precision for $s\\gtrsim10^{12}$: " + OLDINT + " It now passes $\\ln t$ and $\\ln(1-t)$ separately (incomplete beta / its complement at the exact $1-t$; for $b<0.02$ a series in $1-u$ with exact $\\int w^{p-1}dw$ terms, Gauss–Legendre above $w=1/(a-1)$ for $a>4$) and returns $+\\infty$ where $\\gamma\\le-3$ (non-integrable core, which a free $\\gamma$ law can reach at extreme nodes). The first report's statement '1.9e-9 against quadrature' was valid only for $\\alpha\\le5$.",
    "**Sphere cuts.** The gas datasets now use spheres with cell-mean $x\\le1.8$ (R2) and $\\le3.3$ (R3), as decided; run 2 used 1.9 and 3.6 (undocumented). Pressure datasets have no spheres and are unchanged.",
    f"**Fits rerun on maas** (k2, k3 from scratch; {RCPT.get('workers')} workers) with extra start vectors and 20 instead of 12 bootstrap refits for the brk and free variants. Changes (`results/k6_run2_vs_run3.json`): " + "; ".join(ch) + ".",
    "**Report fixes.** Profile ratios are now evaluated at exactly $x$ = 0.1, 0.5, 1, 2 (they were taken at the nearest grid points 0.0996, 0.5035, 0.992, 2.030); the redshift-coverage line was one mass bin too low for $z\\ge1$; the A2-ii neighbour-gas ranges are now quoted per field from the particle stacks; parameters at a fit bound are flagged in the tables; the XGPaint snippet uses Float64 literals (with `alpha_amp=1` it raised a MethodError) and is checked by running it.",
]
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print("ok", len(json.dumps(out)))
