"""Build the Test G report (LaTeX -> PDF): equations, fitted profiles against the stacks, and the analysis outputs.

Every number in the tables is read from Test G's results (results/*.npz, results/*.json, results/tables/*.csv) and
from report/figure_numbers.json (make_report_figures.py); the equations restate the code (g_models.py, g_physics.py,
g_fit.py, g_evaluate.py, g_pdf.py, g_selection.py, g_cv.py and Test F's f_basis.py / kernels/engine.py).

  python -B make_report_figures.py && python -B build_report.py      ->  report/Test_G_report.pdf
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
G = HERE.parent
RES = G / "results"
TAB = RES / "tables"
prep = dict(np.load(RES / "prepare.npz"))
prepj = json.loads((RES / "prepare.json").read_text())
fz = np.load(RES / "fits.npz")
pars = json.loads(str(fz["params_json"]))
pr = np.load(RES / "f_base" / "profiles.npz")
figj = json.loads((HERE / "figure_numbers.json").read_text())
scores = pd.read_csv(TAB / "G_scores.csv")
aic = pd.read_csv(TAB / "G_aic_bic.csv")
glob = pd.read_csv(TAB / "G_fit_globals.csv")
recs = pd.read_csv(TAB / "G_fit_records.csv")
ladder = pd.read_csv(TAB / "G_selection_ladder.csv")
selmodels = pd.read_csv(TAB / "G_selection_models.csv")
selobs = pd.read_csv(TAB / "G_selection_observables.csv")
cv = pd.read_csv(TAB / "G_cv_scores.csv")
cross = pd.read_csv(TAB / "G_cross_scores.csv")
selsum = json.loads((RES / "selection_summary.json").read_text())
LABELS = prepj["bins"]["labels"]
CAL = list(range(5, 15))
NB = len(LABELS)


def lab(s):
    return s.replace("-", "--")


def f(v, d=2, pct=False, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    v = float(v) * (100 if pct else 1)
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def table(cols, rows, caption, label, align=None, size="\\small", note=None):
    align = align or ("l" + "r" * (len(cols) - 1))
    out = [r"\begin{table}[H]", r"\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    out += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(f"\\par\\smallskip\\footnotesize {note}")
    out.append(r"\end{table}")
    return "\n".join(out)


def score(field, model, objective="a", r_fit=4.0, r_cut=4.0, ev="paint", pred="S1", col=None):
    s = scores[(scores.field == field) & (scores.model == model) & (scores.objective == objective) &
               (scores.R_fit == r_fit) & (scores.R_cut == r_cut) & (scores["eval"] == ev) &
               (scores.prediction == pred) & (scores.fit_target == "main")]
    return float(s[col].iloc[0]) if len(s) else np.nan


def cvs(field, model, objective="a", col=None, ev="paint", pred="S1"):
    s = cv[(cv.field == field) & (cv.model == model) & (cv.objective == objective) & (cv.R_fit == 4) &
           (cv.R_cut == 4) & (cv["eval"] == ev) & (cv.prediction == pred)]
    return float(s[col].iloc[0]) if len(s) else np.nan


# ============================================================================ tables
def t_bins():
    rows = []
    s1 = prep["S1_main"]
    for c in range(NB):
        b = CAL[c]
        rows.append([lab(LABELS[c]), prepj["bins"]["N"][c], f(float(pr["neff_main"][b]), 0),
                     f(prepj["bins"]["mean_logm"][c], 2), f(prepj["bins"]["mean_r200c_mpc"][c], 2),
                     f(prepj["bins"]["mean_b"][c], 2), f(prep["bin_nu"][c], 2), f(prep["r200m_over_r200c"][c], 2),
                     f(s1[c, 0], 3), f(s1[c, 1], 3)])
    return table(["log $M_{200c}$ bin", "$N$", "$N_{\\rm eff}$", r"$\langle\log M\rangle$", r"$\langle R_{200c}\rangle$ [Mpc]",
                  r"$\langle b\rangle$", r"$\nu$", "$R_{200m}/R_{200c}$", r"$\sigma^2_{\ln Y}$", r"$\sigma^2_{\ln N_e}$"], rows,
                 "The ten calibrated mass bins (FLAMINGO L1\\_m9, $z=0$, masses in M$_\\odot$). Averages are "
                 "inclusion-weighted; $b$ is the Tinker et al. (2010) bias at $\\Delta=200/\\Omega_m$, $\\nu$ the peak "
                 "height, $R_{200m}/R_{200c}$ from NFW with the SOAP concentration. The last two columns are the S1 scatter "
                 "variances, $\\sigma^2=\\ln(\\langle X^2\\rangle/\\langle X\\rangle^2)$ of the per-halo totals inside 4 $R_{200c}$.",
                 "tab:bins", size="\\footnotesize")


def t_twohalo():
    rows = []
    for c in range(NB):
        r = prep["ratio2h_main"][c]
        ra = prep["ratio2h_a2ii"][c]
        t = prep["temp_T_e_main"][c]
        rows.append([lab(LABELS[c]), f(r[3]), f(r[4]), f(r[5]), f(ra[5]), f(t[3]), f(t[4]), f(t[5])])
    return table(["bin", "$1.5$--$2$", "$2$--$3$", "$3$--$4$", "$3$--$4$ (A2-ii)", "$1.5$--$2$", "$2$--$3$", "$3$--$4$"],
                 rows, "Left: stacked electron density over the parameter-free linear 2-halo prediction "
                 "$\\bar E[1+\\langle b\\,\\xi_{\\rm lin}\\rangle]$, volume-integrated per band of $x=r/R_{200c}$ (all gas; last "
                 "column the A2-ii painting target). Right: measured electron-weighted temperature $T_e/T_{200}$ per band.",
                 "tab:twohalo", align="lrrrr|rrr", size="\\footnotesize")


def t_pkeep():
    rows = []
    for fld in ("P", "E"):
        p = prep[f"pkeep_params_{fld}"]
        rms = prepj[f"pkeep_{fld}"]["rms_rel_resid_per_bin"]
        rows.append(["pressure" if fld == "P" else "electrons", f(p[0], 3), f(p[1], 3), f(np.exp(p[2]), 2), f(p[3], 3),
                     f(p[4], 2), f"{100 * min(rms):.1f}--{100 * max(rms[:8]):.1f}\\% ({100 * rms[8]:.0f}, {100 * rms[9]:.0f}\\%)"])
    return table(["field", "$f_0$", "$f_1$", "$x_h(10^{14.5})$", "$\\ell_1$", "$n$", "RMS per bin (top two bins)"], rows,
                 "Ownership correction $p_{\\rm keep}(x,M)$, eq.~(\\ref{eq:pkeep}), fitted to the stacked A2-ii/all ratio "
                 "beyond 1.34 $R_{200c}$.", "tab:pkeep", size="\\footnotesize")


IC_ORDER = {"P": ["P2m", "P1", "P2g6", "P2gf", "P2i", "P2s", "P0"], "E": ["E3", "E2gf", "E2g6", "E1", "E0"]}


def t_ic(fld):
    rows = []
    a = aic[(aic.field == fld) & (aic.R_fit == 4) & (aic.objective == "a") & (aic.fit_target == "main")]
    for m in IC_ORDER[fld]:
        r = a[a.model == m].iloc[0]
        s = scores[(scores.field == fld) & (scores.model == m) & (scores.objective == "a") & (scores.R_fit == 4) &
                   (scores.R_cut == 4) & (scores["eval"] == "all") & (scores.prediction == "det") & (scores.fit_target == "main")].iloc[0]
        rows.append([m, int(r.k), f(s.profile_rms_inner_lt_1R, 1), f(s.profile_rms_1_to_2R, 1), f(s.profile_rms_outer_2_to_4R, 1),
                     f(r.sigma_d_median, 2, pct=True), f(r.dAIC, 1), f(r.dBIC, 1)])
    name = "Pressure" if fld == "P" else "Electron"
    return table(["model", "$k$", "RMS $<1R$ [\\%]", "1--2$R$", "2--4$R$", "$\\sigma_d$ [\\%]", "$\\Delta$AIC", "$\\Delta$BIC"], rows,
                 f"{name} models fitted to all gas, objective (a), $R_{{\\rm fit}}=4$: shell RMS of $\\ln(S/\\bar F)$ per region, "
                 "median model-discrepancy term $\\sigma_d$, and information criteria with a common $\\sigma_{\\rm ref}$ per bin "
                 f"($n={int(a.n_shells.iloc[0])}$ shells).", f"tab:ic{fld}", size="\\footnotesize")


OBS_ORDER = {"P": [("P2m", "a"), ("P1", "a"), ("P1", "d"), ("P2gf", "a"), ("P2g6", "a"), ("P0", "a")],
             "E": [("E2gf", "a"), ("E2g6", "a"), ("E1", "d"), ("E1", "a"), ("E3", "a"), ("E0", "a")]}


def t_obs(fld):
    rows = []
    for m, o in OBS_ORDER[fld]:
        if fld == "P":
            rows.append([f"{m} ({o})", f(score(fld, m, o, pred="det", col="rms_tSZ"), 2, True),
                         f(score(fld, m, o, pred="S1", col="rms_tSZ"), 2, True),
                         f(score(fld, m, o, pred="S2", col="rms_tSZ"), 2, True),
                         f(score(fld, m, o, pred="det", col="rms_tSZ_vs_mean"), 2, True),
                         f(cvs(fld, m, o, col="rms_tSZ"), 2, True),
                         f(score(fld, m, o, ev="all", pred="S1", col="rms_tSZ"), 2, True)])
        else:
            two = lambda pred, col1, col2, ev="paint": f"{f(score(fld, m, o, ev=ev, pred=pred, col=col1), 2, True)} / {f(score(fld, m, o, ev=ev, pred=pred, col=col2), 2, True)}"  # noqa: E731
            rows.append([f"{m} ({o})", two("det", "rms_kSZ", "rms_DM"), two("S1", "rms_kSZ", "rms_DM"),
                         two("S2", "rms_kSZ", "rms_DM"), two("det", "rms_kSZ_vs_mean", "rms_DM_vs_mean"),
                         f"{f(cvs(fld, m, o, col='rms_kSZ'), 2, True)} / {f(cvs(fld, m, o, col='rms_DM'), 2, True)}",
                         two("S1", "rms_kSZ", "rms_DM", ev="all")])
    if fld == "P":
        cols = ["model", "det", "S1", "S2", "det vs mean", "CV S1", "S1, all gas"]
        cap = ("tSZ family RMS [\\%] (relative errors of $C_\\ell^{yy}$ at $\\ell=100,300,1000,3000$), painting target "
               "(model $\\times\\,p_{\\rm keep}$ against the A2-ii truth) unless stated, $R_{\\rm fit}=R_{\\rm cut}=4$.")
    else:
        cols = ["model", "det", "S1", "S2", "det vs mean", "CV S1", "S1, all gas"]
        cap = ("kSZ / DM family RMS [\\%] (kSZ: $C_\\ell^{\\rm kSZ,1h}$ at $\\ell=100$--$3000$; DM: mean, variance and the "
               "16, 50, 84, 97.5, 99.9\\% quantiles of the halo DM PDF), painting target, $R_{\\rm fit}=R_{\\rm cut}=4$.")
    return table(cols, rows, cap, f"tab:obs{fld}", size="\\footnotesize")


def t_globals():
    g = glob[(glob.R_fit == 4) & (glob.objective == "a") & (glob.fit_target == "main") &
             glob.model.isin(["E2gf", "E3", "P2g6", "P2gf", "P2i", "P2s", "P2m"])]
    rows = []
    for _, r in g.sort_values("model").iterrows():
        rows.append([r.model, f(r.g_gamma_t, 2), f(r.g_tau_0, 3), f(r.g_q_tau, 2), f(r.g_p_M, 3), f(r.g_x_sh, 2),
                     f(r.g_A_inf, 3), f(r.g_a_inf, 2), f(r.g_s_inf, 2),
                     "--" if not isinstance(r.globals_at_bound, str) else r.globals_at_bound.replace("_", "\\_")])
    return table(["model", "$\\gamma_t$", "$\\tau_0$", "$q$", "$p_M$", "$x_{\\rm sh}$", "$A_{\\rm inf}$", "$a$", "$s_{\\rm inf}$",
                  "at bound"], rows, "Global parameters of the physical models (all gas, objective a, $R_{\\rm fit}=4$). "
                 "$\\gamma_t=6$ is fixed in P2g6 and E2g6.", "tab:globals", size="\\footnotesize")


def t_bins_params(key, names, tex_names, label, caption):
    pb = np.array(pars[key]["pb"], float)
    r = recs[recs.key == key].set_index("bin")
    rows = []
    for c in range(NB):
        rr = r.loc[LABELS[c]]
        flag = rr.at_bound if isinstance(rr.at_bound, str) and rr.at_bound else ""
        rows.append([lab(LABELS[c])] + [f(v, 3 if abs(v) < 10 else 1) for v in pb[c]] +
                    [f(rr.x_steepest, 2), f(rr.rms_inner_lt_1R, 1), f(rr.rms_1_to_2R, 1), f(rr.rms_outer_2_to_4R, 1),
                     flag.replace("_", "\\_")])
    return table(["bin"] + tex_names + ["$x_{\\rm steep}$", "RMS $<1R$", "1--2$R$", "2--4$R$", "at bound"], rows, caption, label,
                 size="\\footnotesize")


def t_s2():
    rows = []
    for c in range(NB):
        d = prepj["S2_main"]["Rc4"][c]
        rows.append([lab(LABELS[c]), f(d["P"]["sigma_in"], 3), f(d["P"]["sigma_out"], 3), f(d["P"]["rho"], 2),
                     f(d["E"]["sigma_in"], 3), f(d["E"]["sigma_out"], 3), f(d["E"]["rho"], 2)])
    return table(["bin", "$\\sigma_{\\rm in}$ (P)", "$\\sigma_{\\rm out}$ (P)", "$\\rho$ (P)", "$\\sigma_{\\rm in}$ (E)",
                  "$\\sigma_{\\rm out}$ (E)", "$\\rho$ (E)"], rows,
                 "Inputs of the two-component scatter S2 (all gas, $R_{\\rm cut}=4$): log-amplitude scatter of the per-halo "
                 "integrals inside 1 $R_{200c}$ (in) and over 1.5--4 $R_{200c}$ (out), and their correlation.", "tab:s2",
                 size="\\footnotesize")


def t_ladder(fld):
    l = ladder[(ladder.field == fld) & (ladder.dset == "mass") & (ladder.R_fit == 4)].sort_values(["k", "chi2nu_floor"])
    rows = []
    for _, r in l.iterrows():
        rows.append([r.model.replace("_", "\\_"), int(r.k), f(r.chi2nu_floor, 2), f(r.dBIC_floor, 1),
                     "yes" if r.pass_bic else "no", "yes" if r.pass_cv else "no", "\\textbf{selected}" if r.selected else ""])
    name = "pressure" if fld == "P" else "electrons"
    return table(["model", "$k$", "$\\chi^2/\\nu$ (floor)", "$\\Delta$BIC", "$\\Delta$BIC$\\le6$", "CV", ""], rows,
                 f"Lee22-style ladder, {name}, mass-binned stacks, $R_{{\\rm fit}}=4$ (floor likelihood; best outer option per "
                 "inner law).", f"tab:ladder{fld}", size="\\footnotesize")


def t_selobs():
    rows = []
    o = selobs[(selobs["eval"] == "paint")]
    for fld in ("P", "E"):
        for rf in (4.0, 1.34):
            sub = o[(o.field == fld) & (o.dset == "mass") & (o.R_fit == rf)]
            for m in sub.model.unique():
                d = sub[(sub.model == m) & (sub.prediction == "det")].iloc[0]
                s = sub[(sub.model == m) & (sub.prediction == "S1")].iloc[0]
                if fld == "P":
                    vals = [f(d.rms_tSZ, 2, True), f(s.rms_tSZ, 2, True), f(d.rms_tSZ_vs_mean, 2, True)]
                else:
                    vals = [f"{f(d.rms_kSZ, 2, True)} / {f(d.rms_DM, 2, True)}", f"{f(s.rms_kSZ, 2, True)} / {f(s.rms_DM, 2, True)}",
                            f"{f(d.rms_kSZ_vs_mean, 2, True)} / {f(d.rms_DM_vs_mean, 2, True)}"]
                k = selmodels[(selmodels.field == fld) & (selmodels.dset == "mass") & (selmodels.model == m) &
                              (selmodels.R_fit == rf)]
                kk = int(k.k.iloc[0]) if len(k) else "--"
                rows.append(["pressure" if fld == "P" else "electrons", f"{rf:g}", m, kk] + vals +
                            ["yes" if bool(d.selected) else ""])
    return table(["field", "$R_{\\rm fit}$", "model", "$k$", "det", "S1", "det vs mean", "selected"], rows,
                 "Observables of the ladder's selected minimal models and of reference models with the same outer option "
                 "(painting target, $R_{\\rm cut}=4$; pressure: tSZ RMS [\\%]; electrons: kSZ / DM RMS [\\%]). Fits at "
                 "1.34 $R_{200c}$ are extrapolated to 4 $R_{200c}$.", "tab:selobs", size="\\footnotesize")


def t_cross():
    c = cross[(cross["eval"] == "paint") & (cross.objective == "a") & (cross.R_fit == 4) & (cross.prediction == "S1") &
              cross.P_model.isin(["P0", "P1", "P2m"]) & cross.E_model.isin(["E0", "E1", "E2gf"])]
    rows = [[r.P_model, r.E_model, f(r.rms_tSZxDM, 2, True), f(r.rms_tSZxDM_vs_mean, 2, True)] for _, r in c.iterrows()]
    return table(["pressure model", "electron model", "tSZ$\\times$DM RMS (S1)", "vs truth mean"], rows,
                 "tSZ$\\times$DM cross-check, painting target, objective (a), $R_{\\rm fit}=R_{\\rm cut}=4$, S1 [\\%].",
                 "tab:cross", size="\\footnotesize")


def t_bpl_params():
    rows = []
    for fld in ("P", "E"):
        th = figj[f"{fld}_BPLall_O2_theta"]
        for k, v in th.items():
            rows.append(["pressure" if fld == "P" else "electrons", k.replace("_", "\\_"), f(v, 3)])
    half = (len(rows) + 1) // 2
    rows2 = [rows[i] + (rows[i + half] if i + half < len(rows) else ["", "", ""]) for i in range(half)]
    return table(["field", "parameter", "value", "field", "parameter", "value"], rows2,
                 "Fitted parameters of BPLall+O2 at $R_{\\rm fit}=4$ (floor likelihood). Amplitude, $x_c$ and $\\beta$ laws "
                 "are in $\\ln$ ($q\\_0$ = value at $M_{\\rm cut}$, am / amp = slopes below / above $M_{\\rm cut}$ in "
                 "$\\ln M$); $\\ln x_{t0}$, $a_{xt}$: transition; pressure: $\\ln\\tau_0$, $q$.", "tab:bpl",
                 size="\\footnotesize")


def t_fig_rms():
    rows = []
    for fld, names in (("P", ("P0", "P1", "P2m", "BPL")), ("E", ("E0", "E1", "E2gf", "BPL"))):
        st = figj[f"{fld}_shell_rms_by_bin"]
        for nm in names:
            v = st[nm]
            med = lambda key: np.median([x[key] for x in v[:8]])  # noqa: E731
            mx = lambda key: np.max([x[key] for x in v[:8]])  # noqa: E731
            rows.append(["pressure" if fld == "P" else "electrons", "BPLall+O2" if nm == "BPL" else nm,
                         f"{f(med('rms_lt1'), 1, True)} ({f(mx('rms_lt1'), 1, True)})",
                         f"{f(med('rms_1_2'), 1, True)} ({f(mx('rms_1_2'), 1, True)})",
                         f"{f(med('rms_2_4'), 1, True)} ({f(mx('rms_2_4'), 1, True)})"])
    return table(["field", "model", "$<1R$", "1--2$R$", "2--4$R$"], rows,
                 "Shell RMS of $\\ln(\\mathrm{model}/\\mathrm{stack})$ [\\%] in Figs.~\\ref{fig:profP}--\\ref{fig:profE}, "
                 "median (maximum) over the eight bins with $N\\ge127$.", "tab:figrms", size="\\footnotesize")


# ============================================================================ document
TEX = r"""
\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage{lmodern}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs,array}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor}
\usepackage{microtype}
\usepackage{enumitem}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\setlist{nosep,leftmargin=1.4em}
\setlength{\parskip}{3pt}
\setlength{\parindent}{0pt}
\newcommand{\Rc}{R_{200c}}
\newcommand{\Rfit}{R_{\rm fit}}
\newcommand{\Rcut}{R_{\rm cut}}
\newcommand{\ftr}{f_{\rm tr}}
\newcommand{\ntwo}{n_{\rm 2h}}
\newcommand{\pk}{p_{\rm keep}}
\begin{document}

\begin{center}
{\LARGE\bfseries Test G: physically motivated outer profiles\\[3pt] for the electron pressure and density}\\[8pt]
{\large Equations, fitted profiles against the stacks, and results}\\[6pt]
FLAMINGO L1\_m9, snapshot 0077 ($z=0$) \quad$\cdot$\quad overlap/covariance study \texttt{overlap\_covariance\_L1\_m9\_0077\_20261012}\quad$\cdot$\quad 13 October 2026
\end{center}

\section*{Summary}
\begin{itemize}
\item \textbf{Goal.} Calibrate the electron pressure (tSZ) and the electron density (kSZ, FRB DM) separately, with an outer
  component that is physically motivated and tested, not an extra free power law; compare Lee22-style
  parametrisations by reduced $\chi^2$ with as few parameters as possible.
\item \textbf{Pressure: P2m} (\S\ref{sec:pmodels}). The bound B12 profile with a splashback-like transition, plus the linear
  2-halo gas at an outer temperature that depends on mass; 44 parameters. tSZ error 0.52\% (S1; 0.38\% on held-out
  haloes). A self-similar outer temperature is 1.3--4.4 times too cold at 3--4~$\Rc$.
\item \textbf{Electrons: E2gf} (\S\ref{sec:emodels}). The bound B16 profile with the transition, plus the
  parameter-free linear 2-halo term; 41 parameters. kSZ 0.32\%, DM 0.50\% (S1). The transition sits at 2.1--2.4~$\Rc$
  (1.3--1.4~$R_{200m}$), the splashback scale.
\item \textbf{Fewest parameters.} Lee22's gNFW with every parameter a broken power law in mass (break at
  $10^{14.1}$~M$_\odot$) plus the physical outer term: 14 (pressure) and 12 (electron) parameters; tSZ 1.14\%,
  kSZ 1.26\%, DM 0.50\%.
\item \textbf{Scatter.} One lognormal amplitude per halo (S1) stays best; separate inner and outer amplitudes (S2) do worse.
\item \textbf{Fit and cut.} Both fields: fit to all gas at $\Rfit=\Rcut=4\,\Rc$ (objective a), hard cut at 4~$\Rc$, paint
  the model times the ownership correction $\pk$.
\end{itemize}

\tableofcontents

%==========================================================================================================
\section{Data, stacks and units}\label{sec:data}

\subsection{Haloes and mass bins}
The sample is the overlap study's 4,588 centrals of FLAMINGO L1\_m9 at $z=0$: 300 random haloes per 0.2-dex bin of
$M_{200c}$ below $10^{14.5}$~M$_\odot$, 213 added for the overlap tests, and every halo above. Test G calibrates the ten
bins from $10^{13.5}$ to $10^{15.5}$~M$_\odot$ (Table~\ref{tab:bins}). Halo $h$ carries the inclusion weight
$w_h=1/\pi_h$, the inverse of its inclusion probability.

<<TAB_BINS>>

\subsection{Per-halo shell profiles}
Radii are in units of each halo's own $\Rc$, $x=r/\Rc$. The 35 shells have edges
\[ x_s\in\{0,\,0.025,\dots,0.837\ (\text{0.069-dex log grid}),\,1,\,1.152,\,1.34,\,1.5,\,1.75,\,2,\,2.18,\,2.5,\,3,\,3.30,\,3.63,\,4\} \]
with geometric volumes $V_s=\tfrac{4\pi}{3}\Rc^3\,(x_{s+1}^3-x_s^3)$. For every halo $h$ and shell $s$, summed over the gas
particles $i$ in the shell:
\begin{align}
P_h(s) &= \frac{m_e c^2}{\sigma_T}\,\frac{\sum_{i\in s} y_i}{V_s\,P_{200}}
        = \frac{\sum_{i\in s} N_{e,i}\,k_B T_i}{V_s\,P_{200}},
\qquad P_{200}=\frac{G M_{200c}\,(200\rho_{\rm crit})\,f_b}{2\Rc}, \label{eq:P}\\
E_h(s) &= \frac{\mu_{\rm XGP}\sum_{i\in s} N_{e,i}}{V_s\,f_b\,\rho_{\rm crit}},
\qquad N_{e,i}=n_{e,i}\,\frac{m_i}{\rho_i},\qquad \mu_{\rm XGP}=\frac{m_e+2m_p/(1+X_H)}{0.9}=1.2632\,m_p, \label{eq:E}
\end{align}
where $y_i=\sigma_T k_B T_i N_{e,i}/(m_e c^2)$ is the particle's Compton-$y$ contribution (FLAMINGO's
\texttt{ComptonYParameters}), $X_H=0.76$ and $f_b=\Omega_b/\Omega_m$. $E$ is XGPaint's electron-density unit (it includes
XGPaint's factor $1/0.9$). Recently AGN-heated gas is removed: particles heated within the last 15~Myr ($a_{\rm last}\ge
0.998956$) with $8.8\times10^6\le T\le1.756\times10^8$~K (SOAP's filter). Two gas selections are used:
\begin{itemize}
\item \textbf{all gas} (``main''): everything in the shell, own satellites included;
\item \textbf{A2-ii} (the painting target): a particle belongs to the painted halo ($M_{200c}\ge6.6\times10^{12}$~M$_\odot$)
  with the smallest $r/\Rc$, so neighbouring painted haloes do not count each other's gas twice.
\end{itemize}

\subsection{Stacks}
In bin $c$ the stack is the inclusion-weighted mean, with the reliability-weighted standard deviation and the effective
number of haloes:
\begin{equation}
\bar F_c(s)=\frac{\sum_h w_h F_h(s)}{\sum_h w_h},\qquad
{\rm SD}_c(s)^2=\frac{\sum_h w_h\,[F_h(s)-\bar F_c(s)]^2}{\sum_h w_h\,(1-\sum_h \tilde w_h^2)},\qquad
N_{\rm eff}=\frac{(\sum_h w_h)^2}{\sum_h w_h^2},
\end{equation}
with $\tilde w_h=w_h/\sum w$. The error of the mean shown in the figures is ${\rm SD}/\sqrt{N_{\rm eff}}$. A shell enters a
bin's fit only if it is resolved for every halo of the bin: inner edge $\ge\max(0.04,\,4\epsilon/\Rc)$ ($\epsilon$ the
softening), at least 20 gas particles, and inner edge $\ge2\times$ the upper-median smoothing length in the shell.

\subsection{Normalisation constants}
With $k_BT_{200}=0.59\,m_p\,GM_{200c}/(2\Rc)$, the units of eqs.~(\ref{eq:P}) and (\ref{eq:E}) are related by
\begin{equation}
\frac{P}{P_{200}}=\kappa_P\,\frac{E}{200}\,\frac{T_e}{T_{200}},\qquad \kappa_P=\frac{0.59\,m_p}{\mu_{\rm XGP}}=0.4671,
\label{eq:kappaP}
\end{equation}
which the stacks satisfy to $0.3$--$1.2\%$ (measured $P/[(E/200)(T_M/T_{200})]=0.4685$--$0.4727$). In $E$ units the cosmic
mean electron density is not $\Omega_m$ but
\begin{equation}
\bar E=\kappa_E\,\Omega_m=<<EBAR>>,\qquad \kappa_E=\frac{E}{\rho_{\rm gas}/(f_b\rho_{\rm crit})}=<<KAPPAE>>\ \text{(measured at 2--4 }\Rc),
\label{eq:ebar}
\end{equation}
against $(1+X)/2\cdot\mu_{\rm XGP}/m_p=1.107$ for fully ionised primordial gas ($\Omega_m=<<OMEGAM>>$; stars neglected).

%==========================================================================================================
\section{Profile models}\label{sec:models}

\subsection{Bound (inner) profiles}
Pressure uses B12's generalised NFW and electrons B16's density profile, with their published shape constants:
\begin{align}
P_g(x)&=0.5176\,P_0\left(\frac{x}{x_c}\right)^{-0.3}\left(1+\frac{x}{x_c}\right)^{-\beta}, \label{eq:B12}\\
E_g(x)&=\rho_0\left(\frac{x}{0.5}\right)^{-0.2}\left[1+\left(\frac{x}{0.5}\right)^{\alpha}\right]^{-(\beta-0.2)/\alpha}. \label{eq:B16}
\end{align}
The factor $0.5176=(2+2X)/(3+5X)$ ($X=0.76$) is the electron share of B12's thermal pressure. Free per bin: $(P_0,x_c,\beta)$ and
$(\rho_0,\alpha,\beta)$.

\subsection{The splashback-like transition}\label{sec:transition}
The bound profile is multiplied by the transition term of Diemer \& Kravtsov (2014, DK14),
\begin{equation}
\ftr(x)=\left[1+\left(\frac{x}{x_t}\right)^4\right]^{-\gamma_t/4},
\qquad
\ftr\to1\ (x\ll x_t),\qquad \ftr\to(x/x_t)^{-\gamma_t}\ (x\gg x_t), \label{eq:ftr}
\end{equation}
with $x_t$ free per bin and $\gamma_t$ global (or fixed at 6). Beyond $x_t$ the bound component's logarithmic slope steepens
by $\gamma_t$, so the bound gas ends within a factor of a few in radius, and the outer term takes over.

\textbf{What ``splashback-like'' means.}
\begin{itemize}
\item \textbf{Splashback in dark matter.} The splashback radius is where matter reaches the apocentre of its first orbit
  after falling into a halo. Inside it, matter has orbited at least once; outside, it is still falling in. Stacked dark
  matter profiles steepen sharply there, typically at 0.8--1.5~$R_{200m}$, and further inside for faster-accreting
  (higher-$\nu$) haloes (Diemer \& Kravtsov 2014; Adhikari, Dalal \& Chamberlain 2014; More, Diemer \& Kravtsov 2015).
  DK14 model this edge with eq.~(\ref{eq:ftr}) multiplying the orbiting profile, plus an infalling (2-halo) term.
\item \textbf{The same form for gas.} Test G multiplies the bound gas profile by eq.~(\ref{eq:ftr}) and adds the 2-halo gas.
  It is ``splashback-like'' because the term is a phenomenological edge of the bound gas, not a dynamical measurement.
  Gas is collisional: infalling gas is shocked and does not orbit, so its edge need not coincide with the dark matter
  splashback radius, and the accretion shock is expected further out (roughly 1.5--2 times the splashback radius,
  2.5--5~$\Rc$ here).
\item \textbf{What the fit finds.} The electron edge sits at the dark matter splashback scale: $x_t=2.1$--$2.4\,\Rc$, i.e.
  1.3--1.4~$R_{200m}$ ($R_{200m}/\Rc=1.67$--$1.70$, Table~\ref{tab:bins}), with $\gamma_t=5.1$. The steepest slope of the
  total electron model moves inward with peak height, from 2.28 to 1.20~$\Rc$ (1.36 to 0.71~$R_{200m}$) as $\nu$ runs
  from 1.4 to 3.6 (Table~\ref{tab:E2gfbins}), as expected for splashback. Pressure shows the edge only in P2m ($x_t\approx2.0\,\Rc$); no accretion-shock cutoff is
  detected inside 4~$\Rc$. Figure~\ref{fig:components} shows the decomposition.
\end{itemize}

\subsection{The linear 2-halo term}
For a halo of mass $M$ and radius $\Rc$ the mean electron density around it at $r=x\Rc$ is
\begin{equation}
\ntwo(x)=\bar E\,\bigl[1+b(M)\,\xi_{\rm lin}(x\Rc)\bigr], \label{eq:n2h}
\end{equation}
with $\bar E$ from eq.~(\ref{eq:ebar}), $\xi_{\rm lin}$ the linear matter correlation function of the FLAMINGO cosmology at
$z=0$ (CLASS), and $b(M)$ the Tinker et al. (2010) bias at $\Delta=200/\Omega_m$ evaluated at $M_{200c}$. A bin uses the
inclusion-weighted mean of the per-halo terms, $T_c(x)=\sum_h w_h\,b(M_h)\,\xi_{\rm lin}(x R_h)/\sum_h w_h$, which is exact
for a stack in $x$. The term has no free parameter.

\subsection{Electron models}\label{sec:emodels}
\begin{align}
\text{E0:}\quad & E(x)=E_g(x)\\
\text{E1:}\quad & E(x)=E_g(x)+A\,(x/2)^{-s}\quad(\text{Test F; }A,\,s\text{ per bin},\ s\le3)\\
\text{E2g6, E2gf:}\quad & E(x)=E_g(x)\,\ftr(x)+\bar E\,[1+T_c(x)]\quad(\gamma_t=6,\text{ or global})\label{eq:E2}\\
\text{E3:}\quad & E(x)=\text{E2gf}+\bar E\,A_{\rm inf}\,(b/b_{\rm piv})^{a}\,(x/2)^{-s_{\rm inf}}\,D(x)\quad(\text{global }A_{\rm inf},a,s_{\rm inf})\\
\text{E3pb:}\quad & E(x)=\text{E2gf}+\bar E\,A_c\,(x/2)^{-s_c}\,D(x)\quad(\text{per-bin infall; universality test})
\end{align}
with $b_{\rm piv}=b(10^{14.5}\,{\rm M}_\odot)=<<BPIV>>$ and the core damping $D(x)=x^4/(x^4+0.1^4)$, which switches the
power-law terms off inside $0.1\,\Rc$.

\subsection{Pressure models}\label{sec:pmodels}
For pressure the outer gas is the 2-halo electron density at an outer temperature $\tau(x)=T_{\rm out}/T_{200}$, converted
with eq.~(\ref{eq:kappaP}):
\begin{align}
\text{P0:}\quad & P(x)=P_g(x)\\
\text{P1:}\quad & P(x)=P_g(x)+A\,(x/2)^{-s}\quad(\text{Test F})\\
\text{P2:}\quad & P(x)=P_g(x)\,\ftr(x)+\frac{\kappa_P}{200}\,n_{\rm out}(x)\,\tau(x)\,D(x),\qquad n_{\rm out}=\bar E\,[1+T_c(x)],\quad
\tau(x)=\tau_0\,(x/2)^{-q} \label{eq:P2}
\end{align}
P2g6 and P2gf have $\gamma_t=6$ or global, with global $\tau_0$ and $q$. Variants:
\begin{align}
\text{P2m:}\quad & \tau_0\to\tau_0\,(M/10^{14.5}\,{\rm M}_\odot)^{-p_M}\quad(\text{mass-dependent outer temperature, global }p_M)\label{eq:P2m}\\
\text{P2s:}\quad & \tau(x)\to\tau(x)\,\bigl[1+(x/x_{\rm sh})^8\bigr]^{-1}\quad(\text{accretion-shock cutoff, global }x_{\rm sh})\\
\text{P2i:}\quad & n_{\rm out}\to n_{\rm out}+\bar E\,A_{\rm inf}(b/b_{\rm piv})^{a}(x/2)^{-s_{\rm inf}}D(x)\quad(\text{E3's infall, fixed})
\end{align}
In P2m the mass enters through each bin's mean $\log M$. Since $T_{200}\propto M^{2/3}$, $T_{\rm out}\propto M^{2/3-p_M}$.

\subsection{Shell averages and the cut}
Models are compared with the data as volume-weighted shell averages, by 32-point Gauss--Legendre quadrature in every
shell,
\begin{equation}
S_c(s)=\frac{\int_{x_s}^{x_{s+1}}F_c(x)\,x^2\,dx}{\int_{x_s}^{x_{s+1}}x^2\,dx},\qquad S_c(s)=0\ \text{ for } x_{s+1}>\Rcut .
\end{equation}
Painting multiplies both components at every node by the ownership correction
\begin{equation}
\pk(x,M)=1-\frac{f(M)}{1+[x/x_h(M)]^{-n}},\qquad f=f_0+f_1 v,\quad \ln x_h=\ell_0+\ell_1 v,\quad v=\log_{10}(M/{\rm M}_\odot)-14.5,
\label{eq:pkeep}
\end{equation}
fitted per field to $\ln(\bar F_{\rm A2ii}/\bar F_{\rm all})$ on the shells beyond 1.34~$\Rc$ with weights $\sqrt{N_{\rm eff}}$
(Table~\ref{tab:pkeep}). Well beyond $x_h\approx3.3$--$3.4$ electrons keep half of the all-gas density ($f\simeq0.5$) and
pressure less than a fifth ($f\simeq0.82$): the A2-ii partition gives most of the hot outer gas to neighbouring painted
haloes. At 4~$\Rc$, $\pk=0.38$--$0.40$ for pressure, and from $0.56$ (high mass) to $0.71$ (low mass) for electrons.

<<TAB_PKEEP>>

\subsection{Parameter counts}
For the 10 bins: E0, P0 $3\times10=30$; E1, P1 $5\times10=50$; E2g6 $4\times10=40$; E2gf 41; E3 44; E3pb 61; P2g6
$4\times10+2=42$; P2gf, P2i 43; P2s, P2m 44.

%==========================================================================================================
\section{Fitting}\label{sec:fit}

\subsection{Objectives}
All fits work on $\ln$ of the stack, on the resolved shells inside $\Rfit$. Two objectives, minimised as
$\sum_c\sum_s r_{cs}^2$:
\begin{align}
\text{(a) current diagonal:}\quad & r_{cs}=\frac{\ln S_c(s)-\ln\bar F_c(s)}{\max\bigl[{\rm SD}_c(s)/\bar F_c(s),\,0.05\bigr]}, \label{eq:obja}\\
\text{(d) importance diagonal:}\quad & r_{cs}=\sqrt{W_{cs}}\,\bigl[\ln S_c(s)-\ln\bar F_c(s)\bigr],
\end{align}
with $W$ Test F's importance maps ($\sum_{cs}W_{cs}=1$): $W_{\rm tSZ}$ for pressure, $(W_{\rm kSZ}+W_{\rm DM})/2$ for
electrons. Objective (a) weights each shell by the halo-to-halo scatter, not by the error of the mean; it is the convention
of the earlier tests.

\subsection{Joint fit}
Models with global parameters are fitted jointly over the ten bins by trust-region least squares with the analytic
Jacobian, in $\theta=\ln p$ for positive parameters. Each fit starts from per-bin fits with the globals fixed (multi-start
$x_t\in\{1.5,2.5,4\}$) and from two global starts. Bounds: $x_t\in[0.5,20]$, $\gamma_t\in[0.5,30]$, $s_{\rm inf}\le3.9$,
$q\in[-4,3.9]$, $p_M\in[-3,3]$, $x_{\rm sh}\in[1.2,20]$. E0/P0 and E1/P1 use Test F's own fitting functions, so they
reproduce Test F's fits.

\subsection{Model discrepancy and information criteria}
For each fit, $\sigma_d$ solves
\begin{equation}
\mathbf r_c^{\rm T}\,(\mathbf C_c+\sigma_d^2\mathbf I)^{-1}\,\mathbf r_c=\nu_c,\qquad \nu_c=n_c-(k_{\rm bin}+k_{\rm glob}/N_{\rm bins}),
\end{equation}
with $\mathbf r_c=\ln\mathbf S_c-\ln\bar{\mathbf F}_c$ and $\mathbf C_c$ the bootstrap covariance of $\ln\bar F$ (1000
inclusion-weighted resamples), shrunk towards the neighbouring bins' pooled correlation. For AIC and BIC the errors of all
compared models are inflated by a common $\sigma_{{\rm ref},c}$ per bin, the smallest $\sigma_d$ among them:
\begin{equation}
\chi^2=\sum_c \mathbf r_c^{\rm T}(\mathbf C_c+\sigma_{{\rm ref},c}^2\mathbf I)^{-1}\mathbf r_c,\qquad
{\rm AIC}=\chi^2+2k,\qquad {\rm BIC}=\chi^2+k\ln n ,
\end{equation}
with $k$ all free parameters and $n$ all fitted shells.

%==========================================================================================================
\section{Observables and scores}\label{sec:obs}

\subsection{Halo model}
Profiles are propagated with the frozen lightcone halo model of the earlier studies. For a halo of mass $M$ at redshift $z$,
\begin{equation}
u_\ell(M,z)=a(M,z)\,4\pi\int_0^{\Rcut}x^2\,F(M,z,x)\,\frac{\sin(qx)}{qx}\,dx,\qquad q=\frac{(\ell+1/2)\,\Rc}{D_A(z)},
\end{equation}
with $a$ the Compton-$y$ (pressure) or optical-depth (electrons) normalisation and the declared redshift response
$F(M,z,x)=F_0(M,x)\,F_{\rm pub}(M,z,x)/F_{\rm pub}(M,0,x)$. Over the lightcone cells (halo counts $N$ in mass and redshift):
\begin{align}
C_\ell^{\rm 1h}&=\sum_{\rm cells}N\,\frac{|u_\ell|^2}{4\pi},\qquad
C_\ell^{\rm 2h}=\sum_z\frac{P_{\rm lin}(\ell,z)}{V_z}\Bigl[\sum_M N\,b\,\frac{u_\ell}{4\pi}\Bigr]^2,\\
C_\ell^{\rm kSZ,1h}&=\sum_{\rm cells}\sum_{\rm haloes}\frac{v_{\rm LOS}^2}{c^2}\,\frac{|\tau_\ell|^2}{4\pi}\quad(\text{catalogue velocities}).
\end{align}
DM is the halo-only observer DM for a source at $z_s=1$ (cells at $z\le1$, factor $1/(1+z)$): its mean, its compound-Poisson
variance, and the exact per-sightline PDF (FFT), summarised by the 16, 50, 84, 97.5 and 99.9\% quantiles. Outside the
calibrated mass range ($10^{13.5}$--$10^{15.388}$~M$_\odot$) every model and the truth use the published B12/B16 profiles.

\subsection{Evaluation in the shell basis}
The calibrated bins enter only through their shell averages. The 1-halo terms and the DM variance are bilinear in the
profile, so with the per-bin second moments $\langle F_sF_t\rangle_c$
\begin{equation}
C_\ell^{\rm 1h}=\sum_c\sum_{s,t}K_{c,st}(\ell)\,\langle F_sF_t\rangle_c+C_\ell^{\rm 1h,fb},\qquad
C_\ell^{\rm 2h}=\sum_z\frac{P_{\rm lin}}{V_z}\Bigl[W^{\rm fb}_z(\ell)+\sum_{c,s}\bar F_{c}(s)\,W_{z,cs}(\ell)\Bigr]^2,
\end{equation}
exact for any profile form (fb = the fallback haloes outside the calibrated range).

\subsection{Truth and scatter models}
The \textbf{truth} uses the measured per-halo profiles to 4~$\Rc$: the second moments $\langle F_sF_t\rangle$ over the bin's
haloes (inclusion-weighted), which include the halo-to-halo scatter, and every halo's own DM column. The \textbf{truth mean}
uses the stack alone, $\bar F_s\bar F_t$. A model is scored with three predictions of the second moments:
\begin{align}
\text{det:}\quad&\langle F_sF_t\rangle=S_sS_t,\\
\text{S1:}\quad&\langle F_sF_t\rangle=e^{\sigma_c^2}S_sS_t,\qquad \sigma_c^2=\ln\frac{\langle X^2\rangle}{\langle X\rangle^2},\quad
X=Y(<4\Rc)\ \text{or}\ N_e(<4\Rc)\ \text{per halo},\\
\text{S2:}\quad&\langle F_sF_t\rangle=M_{\rm bb}B_sB_t+M_{\rm bo}(B_sO_t+O_sB_t)+M_{\rm oo}O_sO_t,\qquad
M_{ij}=\frac{\langle X_iX_j\rangle}{\langle X_i\rangle\langle X_j\rangle},
\end{align}
with $B$, $O$ the bound and outer components, $X_{\rm b}$ the per-halo integral inside 1~$\Rc$ and $X_{\rm o}$ that over
1.5~$\Rc$--$\Rcut$. The tSZ$\times$DM cross term uses $\langle YN\rangle/(\langle Y\rangle\langle N\rangle)$. For the DM PDF
the amplitudes are lognormal with unit mean: 9 Gauss--Hermite nodes (S1), or $5\times5$ correlated nodes (S2).

\subsection{Errors and family scores}
For each observable $\epsilon_O=O_{\rm model}/O_{\rm truth}-1$, and a family scores the RMS of its members:
\begin{equation}
{\rm RMS}_{\rm fam}=\Bigl[\frac{1}{n_{\rm fam}}\sum_{O\in{\rm fam}}\epsilon_O^2\Bigr]^{1/2},
\end{equation}
with tSZ $=C_\ell^{yy}$ (1h+2h), kSZ $=C_\ell^{\rm kSZ,1h}$ and tSZ$\times$DM $=C_\ell^{y\,\rm DM}$ at $\ell=100,300,1000,3000$
($\ell=8000$ is reported but excluded); DM = mean, variance and the five quantiles. ``vs mean'' compares the det prediction
with the truth mean (the profile fit alone). Evaluations: \emph{all} (all-gas model against the all-gas truth),
\emph{paint} (all-gas model $\times\,\pk$ against the A2-ii truth: the HalfDome painting target), \emph{a2ii} (a model
fitted directly to the A2-ii stacks).

\subsection{Cross-validation}
The 1~Gpc box is split into $4^3$ blocks of 250~Mpc, assigned to 8 folds. For each fold everything is recalibrated on the
other 7/8 of the volume: stacks, fits (globals included), 2-halo templates, $\pk$ and the scatter tables. The
out-of-fold prediction weights the fold models by the held-out inclusion-weight shares $\omega_{ck}$: linear terms
$\sum_k\omega_{ck}F_k$, second moments $\sum_k\omega_{ck}\langle F_sF_t\rangle_k$, DM PDF a mixture of the fold models.

%==========================================================================================================
\section{Lee22-style parametrisations and model selection}\label{sec:sel}

\subsection{Profile and mass laws}
The inner profile is Lee22's gNFW for both fields,
\begin{equation}
F_{\rm in}(x)=A\left(\frac{x}{x_c}\right)^{\gamma}\left(1+\frac{x}{x_c}\right)^{-\beta},\qquad \gamma=-0.3, \label{eq:lee22}
\end{equation}
and each of $p\in\{A,x_c,\beta\}$ follows a law in mass (in $\ln p$):
\begin{align}
\text{PB:}\quad&\text{free per bin}\qquad\qquad
\text{const:}\quad \ln p=\ln p_0\qquad\qquad
\text{pl:}\quad \ln p=\ln p_0+a_m\ln(M/10^{14.5}\,{\rm M}_\odot),\\
\text{bpl:}\quad&\ln p=\ln p_0+\begin{cases}a_m\,\ln(M/M_{\rm cut}),&M<M_{\rm cut}\\ a'_m\,\ln(M/M_{\rm cut}),&M\ge M_{\rm cut}\end{cases}
\qquad(\text{one }M_{\rm cut}\text{, free or Lee22's}),\\
+c:\quad&\ln p\to\ln p+a_c\ln(c_{200c}/10),\qquad
\gamma:\ \text{fixed}\ (-0.3),\ \text{global, or}\ \gamma_0(M/10^{14.5})^{a_\gamma}.
\end{align}
Named models: PL1 (pl in $A$; $x_c$, $\beta$ constant), PL2 (pl in $A$, $x_c$; $\beta$ constant), PL3 (pl in all three),
BPLa (bpl in $A$, pl in $x_c$, $\beta$), BPLs (pl in $A$, bpl in $x_c$, $\beta$), BPLall (bpl in all three), PL3gM (PL3 with $\gamma$ a mass law), PL3c3 (PL3 + $c$ terms), and
\_L22 versions with $M_{\rm cut}$ fixed at Lee22's value ($10^{13.77}$--$10^{13.78}$~M$_\odot$ after the $h$ conversion).
Lee22's own choices are BPLa for pressure and BPLs for electrons.

\subsection{Outer options}
\begin{align}
\text{O0:}\quad&F=F_{\rm in}\\
\text{O1:}\quad&F=F_{\rm in}+\begin{cases}\bar E\,[1+T_c(x)]&\text{electrons (0 parameters)}\\
\frac{\kappa_P}{200}\,\bar E\,[1+T_c(x)]\,\tau_0\,(x/2)^{-q}\,D(x)&\text{pressure }(\tau_0,\,q)\end{cases}\\
\text{O2:}\quad&\text{O1 with }F_{\rm in}\to F_{\rm in}\,[1+(x/x_t)^4]^{-3/2},\quad x_t=x_{t0}\,(M/10^{14.5}\,{\rm M}_\odot)^{a_{xt}}\quad(+2)
\end{align}

\subsection{Likelihoods and selection rule}
The data vector is $\ln\bar F$ on the strictly resolved shells inside $\Rfit$ of each cell (mass bins; also concentration
quartiles or halves). Three likelihoods:
\begin{itemize}
\item \textbf{stat}: $\chi^2=\sum_c\mathbf r_c^{\rm T}\mathbf C_c^{-1}\mathbf r_c$ with the shrunk bootstrap covariance of
  $\ln\bar F$; Hartlap factors $h=(N-p-2)/(N-1)$ are reported.
\item \textbf{floor}: $\mathbf C_c\to\mathbf C_c+\sigma_d^2\mathbf I$ with one $\sigma_d$ per (field, data set, $\Rfit$),
  chosen so that the per-bin reference (PB with its best outer option) has $\chi^2/\nu=1$.
\item \textbf{L22} (Lee22's convention): $\mathbf C_c^{\rm L22}={\rm Cov}_h(\ln F)/N_{\rm eff}$, precision multiplied by $h$;
  cells with $N\le p+2$ are dropped.
\end{itemize}
Reduced $\chi^2=\chi^2/\nu$ with $\nu=n-k$; ${\rm BIC}=\chi^2+k\ln n$. The spatial cross-validation $\chi^2$ uses the
held-out stacks with ${\rm Cov}\times N_{\rm eff,all}/N_{\rm eff,fold}+\sigma_d^2\mathbf I$. \textbf{Selection}: the model
with the fewest parameters whose $\Delta{\rm BIC}\le6$ (floor) and whose CV $\chi^2$ lies within the best CV model's
fold-to-fold scatter ($\sqrt K\times$ the SD of its per-fold $\chi^2$).

%==========================================================================================================
\section{Fitted profiles against the stacked profiles}\label{sec:profiles}

Figures~\ref{fig:profP} and \ref{fig:profE} compare, for every calibrated bin, the stacked profile with the models fitted
to 4~$\Rc$: the minimum change (B12 or B16 refitted), Test F's empirical outer power law, Test G's physical model, and the
fewest-parameter Lee22-style model. Table~\ref{tab:figrms} summarises the shell residuals. Figure~\ref{fig:components}
splits the physical models into their components, and Fig.~\ref{fig:paint} shows the painted model against the A2-ii
stacks.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/profiles_pressure.pdf}
\caption{Electron pressure: stacked profiles (points; open = shells not resolved for every halo, not fitted) and models
fitted to 4~$\Rc$ with objective (a), all gas. Lower panels: model/stack; the grey band is the error of the stacked mean,
dotted lines mark $\pm10\%$. Light grey area in the upper panels: 16--84\% of the haloes.}
\label{fig:profP}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/profiles_electrons.pdf}
\caption{Electron density, as Fig.~\ref{fig:profP}.}
\label{fig:profE}
\end{figure}

<<TAB_FIGRMS>>

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/components.pdf}
\caption{Components of the recommended models for four bins, plotted as $x^2F$ (the stacks are the points). The core
alone (dotted) is the B12/B16 profile without the transition; multiplied by $\ftr$ it gives the bound component (dashed),
which steepens beyond $x_t$ (vertical line). The outer term (dash-dot) is the 2-halo gas; for pressure it is
$\kappa_P/200\,\bar E[1+T_c]\,\tau\,D$. Shaded: 1.0--1.5~$R_{200m}$, the dark matter splashback band. The thin grey line is
the 4~$\Rc$ cut.}
\label{fig:components}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/painting_ratio.pdf}
\caption{Painting target: the all-gas fit multiplied by $\pk(x,M)$, divided by the A2-ii stack, per bin.}
\label{fig:paint}
\end{figure}

%==========================================================================================================
\section{Results}\label{sec:results}

\subsection{Normalisation, 2-halo level and outer temperatures}
\begin{itemize}
\item The stacked electrons at 3--4~$\Rc$ match the parameter-free 2-halo term within $\pm10\%$ for $10^{13.9}$--$10^{14.9}$;
  groups lie 16--26\% above it and the most massive bins 15--19\% below (Table~\ref{tab:twohalo}). With A2-ii the outer
  electrons fall below the 2-halo level (to 0.50--0.59 in clusters), because the partition gives neighbour gas away.
\item The measured outer temperature falls with mass in units of $T_{200}$: $T_e/T_{200}=0.34\to0.10$ at 3--4~$\Rc$ from
  $10^{13.6}$ to $10^{15.35}$. A self-similar $\tau(x)$ cannot follow this; P2m's $T_{\rm out}\propto M^{0.24}$ can.
\item Earlier quick estimates used $\bar E=\Omega_m$; with eq.~(\ref{eq:ebar}) their 2-halo ratios are 0.915 times lower and
  their pressure-``implied'' temperatures 1.96 times higher.
\end{itemize}

<<TAB_TWOHALO>>

\begin{figure}[H]
\centering
\includegraphics[width=0.49\textwidth]{../figures/G2_two_halo_ratio.png}\hfill
\includegraphics[width=0.49\textwidth]{../figures/G3_outer_temperature.png}
\caption{Left: stacked electrons over the linear 2-halo prediction per band and bin. Right: measured outer temperature
against the fitted $\tau(x)$ of P2gf and P2m.}
\end{figure}

\subsection{Pressure}
<<TAB_ICP>>
<<TAB_OBSP>>
\begin{itemize}
\item P2m ties with Test F's P1 on BIC with 6 fewer parameters and is more accurate inside 2~$\Rc$; the self-similar P2
  variants leave 17\% residuals at 2--4~$\Rc$.
\item Best overall: objective (d) at $\Rfit=3$, $\Rcut=4$ gives 0.44--0.49\% for every P2 variant, within the CV scatter.
\item P2m's outer temperature law is checked only at 3--4~$\Rc$, where it matches the measured $T_e$ within $-35\%$ to
  $+12\%$. Inside 2~$\Rc$ its power law ($q=1.44$) runs hotter than the gas (up to about 3 times at $\Rc$ in groups), and
  the outer term carries 9--32\% of the pressure at 0.5~$\Rc$ and 15--35\% at $\Rc$, most in the lowest bins
  (Fig.~\ref{fig:components}). Inside $\Rc$ the bound/outer split is therefore a fit decomposition, not a physical one:
  the profile is accurate, but the outer term there should not be read as 2-halo gas.
\end{itemize}

\subsection{Electrons}
<<TAB_ICE>>
<<TAB_OBSE>>
\begin{itemize}
\item E3's best BIC comes from its infall term acting as an inner cusp ($s_{\rm inf}$ at its bound of 3.9); per bin, an
  outer excess exists only below $10^{14.1}$ (Table~\ref{tab:globals}). E3 is not adopted.
\item E2 is better than E1 inside 2~$\Rc$ and robust at any $\Rfit$: its det DM error is 1.6--2.2\% for $\Rfit=2$--4
  ($\Rcut=4$), while E1's is 77--387\% for $\Rfit=1$--3, because its free power law diverges in the unresolved core. E1
  follows 2--4~$\Rc$ better (3.5\% vs 10\%) because the 2-halo level is off by $-19\%$ to $+26\%$ across mass.
\end{itemize}

<<TAB_GLOBALS>>
<<TAB_CROSS>>

\begin{figure}[H]
\centering
\includegraphics[width=0.49\textwidth]{../figures/G4_residuals_P_main_a_Rf4.png}\hfill
\includegraphics[width=0.49\textwidth]{../figures/G4_residuals_E_main_a_Rf4.png}
\caption{Shell residuals of every model at $\Rfit=4$ (all gas, objective a), pressure (left) and electrons (right).}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.32\textwidth]{../figures/G6_grid_P_tSZ.png}\hfill
\includegraphics[width=0.32\textwidth]{../figures/G6_grid_E_kSZ.png}\hfill
\includegraphics[width=0.32\textwidth]{../figures/G6_grid_E_DM.png}
\caption{Fit radius $\times$ cut radius: tSZ, kSZ and DM family RMS of the physical models (painting target).}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.49\textwidth]{../figures/G5_transition_vs_mass.png}\hfill
\includegraphics[width=0.49\textwidth]{../figures/G10_pkeep.png}
\caption{Left: transition radius and steepest-slope radius against mass. Right: the ownership correction $\pk$ against
the stacked A2-ii/all ratio.}
\end{figure}

\subsection{Scatter}
<<TAB_S2>>
\begin{itemize}
\item S1 (one amplitude per halo) gives the best observables. S2 under-predicts the 1-halo boost: P2m tSZ 1.43\% against
  0.52\%; E2 kSZ/DM 1.02/1.22\% against 0.32/0.50\%.
\item The likely reason: the inner amplitude is taken from the total inside 1~$\Rc$, which averages over the anti-correlated
  core and 0.5--1.5~$\Rc$ shells. Concentration moves gas between them while nearly conserving that total, so
  $\sigma_{\rm in}$ underestimates the core scatter. A shape (pivot) scatter component would be the physical next step;
  it was not tested.
\end{itemize}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{../figures/G7_scatter_paint_a_Rf4.png}
\caption{Scatter predictions against the scatter-inclusive truth (painting target, $\Rfit=\Rcut=4$).}
\end{figure}

\subsection{Lee22-style ladder}
<<TAB_LADDERP>>
<<TAB_LADDERE>>
\begin{itemize}
\item With Lee22's convention (covariance of the mean) their best models give $\chi^2/\nu=3.7$ (P) and 5.4--5.5 (E) on our
  concentration-split cells, close to their 3.3 and 5.1; per-halo scatter would give 0.05--0.06. Lee22 therefore used the
  covariance of the mean.
\item The fitted break mass is $10^{14.10}$--$10^{14.23}$~M$_\odot$ for $\Rfit=2$--4 (<<MCUT>>), about 0.35 dex above Lee22's;
  Lee22's own break choices are rejected by $\Delta$BIC 190--340 at 1.34~$\Rc$.
\item Concentration terms help on concentration-split cells but not for painting: HalfDome has no concentrations, and the
  $c$-models painted at the bin-mean concentration describe the mass stacks worse than the mass-only fit.
\item In the pressure BPLall+O2 fit the outer temperature rises with radius ($q<0$): with a self-similar $\tau$ the O1 term
  absorbs shape mismatch rather than describing the gas.
\end{itemize}
<<TAB_SELOBS>>
<<TAB_BPL>>

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{../figures/G9_ladder_P_mass.png}\\[2pt]
\includegraphics[width=\textwidth]{../figures/G9_ladder_E_mass.png}
\caption{Parsimony ladders (mass-binned stacks): reduced $\chi^2$ with the floor and $\Delta$BIC against the number of free
parameters, per fit radius; circled = selected.}
\end{figure}

\subsection{Cross-validation}
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{../figures/G8_cv.png}
\caption{Out-of-fold (spatial 8-fold) family RMS of the candidate configurations.}
\end{figure}

%==========================================================================================================
\section{Recommendation}\label{sec:rec}
\begin{itemize}
\item \textbf{Pressure: P2m}, eqs.~(\ref{eq:B12}), (\ref{eq:ftr}), (\ref{eq:P2}), (\ref{eq:P2m}); 44 parameters
  (Table~\ref{tab:P2mbins}). Fit to all gas at $\Rfit=\Rcut=4$, objective (a), hard 4~$\Rc$ cut, S1 scatter, painted
  $\times\,\pk$. Errors: tSZ 0.52\% (CV 0.38\%), 0.30\% vs the truth mean.
\item \textbf{Electrons: E2gf}, eqs.~(\ref{eq:B16}), (\ref{eq:ftr}), (\ref{eq:E2}); 41 parameters (Table~\ref{tab:E2gfbins});
  same fit, cut, scatter and painting. Errors: kSZ 0.32\%, DM 0.50\%.
\item \textbf{Fewest parameters:} BPLall+O2 (14 and 12 parameters), at tSZ 1.14\%, kSZ 1.26\%, DM 0.50\%. A smooth-in-mass
  version of P2m (broken power laws plus the mass-dependent outer temperature) was not tested.
\end{itemize}

\section{Caveats}
\begin{itemize}
\item The 2-halo term uses the linear $\xi$, Tinker10 bias at $\Delta=200/\Omega_m$ evaluated at $M_{200c}$, bin-mean
  templates, no halo exclusion, and $\bar E$ with stars not removed (a few per cent high).
\item $\pk$ is a population average; everything is $z=0$ only; the top bins have $N=21$ and 6.
\item In P2m's lowest bin the B12 $\beta$ sits at its upper bound (20): the inner shells there start at 0.35~$\Rc$ and do
  not constrain the core shape (Table~\ref{tab:P2mbins}).
\item Exact DM quantiles were not computed for $\Rcut<2$; S1/S2 DM PDFs only for the listed subsets.
\item 2 of 161 joint fits hit the iteration limit (P2m at $\Rfit=1$, objective d); many $\Rfit\le1.5$ fits sit on bounds by
  construction.
\item The selection-stage fits use the GLS likelihood, where the noisy pressure outskirts carry little weight, so their
  observables are worse than those of the per-field models.
\end{itemize}

\section{Runs and verification}
\begin{itemize}
\item STRW maas, nice 19 with ionice and load/memory guards. Attempt 1 (4 workers; stopped to switch to 8 at the user's
  request): kernels 521~s, prepare 2~s, parity 33~s, fits 162~s. Attempt 2 (8 workers): evaluate 951~s, selection 382~s,
  cv 154~s; exit 0; peak tree RSS 7.0~GiB; load 24--25 on 48 threads, $\ge175$~GiB free.
\item Parity with Test F: the rebuilt basis and DM quantiles are bitwise equal; E0/P0 equal Test F's fits; E1/P1 reproduce
  Test F to $6\times10^{-6}$ / $4\times10^{-4}$. \texttt{verification.json}: all checks passed.
\item This report's curves are evaluated with Test G's own code at the stored parameters and reproduce the stored fit
  shells to $10^{-15}$.
\end{itemize}

%==========================================================================================================
\appendix
\section{Per-bin parameters}
<<TAB_P2M>>
<<TAB_E2GF>>

\section{Files}
\begin{itemize}
\item Code: \texttt{G\_physical\_outer/g\_*.py} and \texttt{run\_test\_g.py}.
\item This report: \texttt{report/make\_report\_figures.py} (figures) and \texttt{report/build\_report.py} (tables, text,
  PDF).
\item Results: \texttt{results/} (\texttt{fits.npz}, \texttt{evaluate.npz}, \texttt{prepare.npz}, \texttt{selection.npz},
  \texttt{tables/G\_*.csv}).
\item Figures: \texttt{figures/} and \texttt{report/figures/}; short summary: \texttt{REPORT.md}.
\end{itemize}

\end{document}
"""


def main():
    rep = {
        "<<TAB_BINS>>": t_bins(),
        "<<TAB_PKEEP>>": t_pkeep(),
        "<<TAB_TWOHALO>>": t_twohalo(),
        "<<TAB_ICP>>": t_ic("P"),
        "<<TAB_OBSP>>": t_obs("P"),
        "<<TAB_ICE>>": t_ic("E"),
        "<<TAB_OBSE>>": t_obs("E"),
        "<<TAB_GLOBALS>>": t_globals(),
        "<<TAB_CROSS>>": t_cross(),
        "<<TAB_S2>>": t_s2(),
        "<<TAB_LADDERP>>": t_ladder("P"),
        "<<TAB_LADDERE>>": t_ladder("E"),
        "<<TAB_SELOBS>>": t_selobs(),
        "<<TAB_BPL>>": t_bpl_params(),
        "<<TAB_FIGRMS>>": t_fig_rms(),
        "<<TAB_P2M>>": t_bins_params("P|P2m|main|a|4", ("P0", "xc", "beta", "x_t"), ["$P_0$", "$x_c$", "$\\beta$", "$x_t$"],
                                     "tab:P2mbins", "P2m per-bin parameters (all gas, objective a, $R_{\\rm fit}=4$); globals in "
                                     "Table~\\ref{tab:globals}. RMS in \\% (objective-a shells)."),
        "<<TAB_E2GF>>": t_bins_params("E|E2gf|main|a|4", ("rho0", "alpha", "beta", "x_t"),
                                      ["$\\rho_0$", "$\\alpha$", "$\\beta$", "$x_t$"], "tab:E2gfbins",
                                      "E2gf per-bin parameters (all gas, objective a, $R_{\\rm fit}=4$); $\\gamma_t$ in "
                                      "Table~\\ref{tab:globals}. RMS in \\%."),
        "<<EBAR>>": f"{float(prep['ebar']):.4f}",
        "<<KAPPAE>>": f"{float(prep['kappa_e']):.4f}",
        "<<OMEGAM>>": f"{float(prep['omega_m']):.3f}",
        "<<BPIV>>": f"{float(prep['b_pivot']):.2f}",
        "<<MCUT>>": "at 4 $R_{200c}$: " + ", ".join(
            f"{'P' if r['field'] == 'P' else 'E'} $10^{{{json.loads(r['params'])['log10Mcut']:.2f}}}$"
            for _, r in pd.read_csv(TAB / "G_selection_models.csv").iterrows()
            if r["dset"] == "mass" and r["model"] == "BPLall+O2" and float(r["R_fit"]) == 4.0),
    }
    tex = TEX
    for k, v in rep.items():
        assert k in tex, k
        tex = tex.replace(k, v)
    assert "<<" not in tex, tex[tex.index("<<"):tex.index("<<") + 40]
    out = HERE / "Test_G_report.tex"
    out.write_text(tex)
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", out.name], cwd=HERE,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-4000:])
            raise SystemExit("pdflatex failed")
    log = (HERE / "Test_G_report.log").read_text(errors="ignore")
    warnings = [l for l in log.splitlines() if "Warning" in l and "Font" not in l]
    for ext in ("aux", "log", "out", "toc"):                      # keep only the .tex and the .pdf
        (HERE / f"Test_G_report.{ext}").unlink(missing_ok=True)
    print(json.dumps({"pdf": str(HERE / "Test_G_report.pdf"), "warnings": warnings[:20]}, indent=1))


if __name__ == "__main__":
    main()
