"""G stage 'selection': Lee22-style parametrisation ladder for the stacked profiles, judged by reduced chi^2 with as
few parameters as possible (user request 2026-10-13; Lee, Coulton, Thiele & Ho 2022, MNRAS 517, 420 = Lee22).

Profile (Lee22 eq. 10, both fields):  F_in(x) = A (x/x_c)^gamma (1 + x/x_c)^-beta,  alpha = 1, gamma = -0.3
Parameter laws for each of A, x_c, beta (Lee22 eqs. 11-12 at z = 0), in ln:
  bin   free per 0.2-dex mass bin                      (PB, the reference)
  const one value
  pl    ln P = ln P0 + a_m ln(M / 10^14.5 Msun)
  bpl   ln P = ln P0 + (a_m below / a_m' above) ln(M / M_cut), one M_cut shared by the three (free, or fixed
        at Lee22's value converted with h = 0.6774: P 13.60 [no c, their Table 7] / 13.64 [with c, Table 1],
        E 13.61 [Table 8] / 13.75 [Table 3] h^-1 Msun)
  + c   a_c ln(c / 10), c = SOAP concentration_200c (Lee22: Rockstar/Klypin R200/R_scale)
  gamma fixed -0.3, global free, or gamma_0 (M / 10^14.5)^a_gamma
Outer options (R_fit > 1.34): O0 none; O1 the physical outer term of Test G (E: Ebar [1 + b xi_lin], 0 parameters;
P: KAPPA_P/200 Ebar [1 + b xi_lin] tau_0 (x/2)^-q, 2 global); O2 = O1 with the inner part times
f_tr = [1 + (x/x_t)^4]^-1.5 (gamma_t = 6), x_t = x_t0 (M / 10^14.5)^a_xt (2 global).
Data sets: 'mass' (the 10 calibrated bins) and 'massc' (each bin split in inclusion-weighted concentration
quartiles when N >= 120, halves when N >= 40, else one cell; as Lee22, who used c quartiles).
Data vector: ln(stacked mean) on the strictly resolved shells inside R_fit (all gas, recently AGN-heated gas removed).
Likelihoods (cells independent):
  stat   bootstrap Cov[ln mean] (1000 draws, inclusion-weighted) with Test B/F shrinkage to the neighbours' pooled
         correlation; Hartlap factors (N - p - 2)/(N - 1) reported (N = haloes in the cell)
  floor  stat + sigma_d^2 I, sigma_d common to every model of a (field, data set, R_fit): the value at which the most
         flexible per-bin reference (PB with the best outer option) reaches chi^2/nu = 1 (iterated with its refit)
  L22    Lee22's convention: lognormal, sample covariance of the haloes' ln F divided by N, precision x Hartlap;
         cells with N <= p + 2 excluded (Lee22 kept cells with > 29 haloes for 27 radial bins).  Lee22's reduced
         chi^2 of 3.3 (P) and 5.1 (E) for nu ~ 4200 with model residuals of up to 3.8 standard errors of the mean (their
         Fig. 2) imply the covariance of the mean; with the per-halo scatter (not divided by N) their fits would give
         chi^2/nu << 1.  The per-halo-scatter value (chi^2_L22 / N per cell) is reported as well.
Every model is fitted under each likelihood (least squares, analytic Jacobian).  Reported: chi^2, nu, chi^2/nu,
sqrt(2/nu), p-value, AIC, BIC, dBIC; spatial-block CV predictive chi^2 (Test F5's 8 folds; floor likelihood; the
held-out stacks with Cov x N_eff,all/N_eff,fold + sigma_d^2 I).
Selection (user's preference): the model with the fewest parameters whose dBIC <= 6 (floor likelihood) and whose CV
chi^2 is within the best CV model's fold-to-fold scatter (sqrt(K) x SD of its per-fold chi^2).
Selected minimal models (mass data set) are propagated to the observables (P: tSZ; E: kSZ, DM PDF) at R_cut = 4,
next to PB with the same outer option.
Outputs: tables/G_selection_models.csv, G_selection_ladder.csv, G_selection_cells.csv, G_selection_observables.csv;
results/selection_summary.json, results/selection.npz.
"""
from __future__ import annotations

import time

import g_common as GC
import g_models as GM
import g_physics as PH
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import chi2 as CHI2

FF = GC.FF
LN10 = np.log(10.0)
H_TNG = 0.6774
C_PIV = 10.0
LOGM_PIV = 14.5
R_FIT_SEL = (2.0, 4.0) if GC.SMOKE else (1.34, 2.0, 3.0, 4.0)
LEE22_MCUT = {("P", False): 13.60, ("P", True): 13.64, ("E", False): 13.61, ("E", True): 13.75}   # h^-1 Msun
QUARTILE_N = 20 if GC.SMOKE else 120
HALVES_N = 20 if GC.SMOKE else 40
QN = ("A", "xc", "beta")
MAX_NFEV = 400                     # per start; flagged in the tables when reached (degenerate break parameters)
_W: dict = {}


def lee22_mcut(field, with_c):
    return LEE22_MCUT[(field, with_c)] - np.log10(H_TNG)


# ============================================================================ cells
class Cells:
    """Cells of one field and data set (all rows, or a training / held-out subset of each cell's rows)."""

    def __init__(self, pr, field, dset, conc, lh, cell_rows=None, gls=True, seed=GC.SEED):
        self.field, self.dset = field, dset
        if cell_rows is None:
            cell_rows = make_cells(pr, dset, conc)
        self.cell_rows = cell_rows
        x35, _ = PH.quad_nodes(GC.F_EDGES)
        self.bin = np.array([c for c, _ in cell_rows])
        self.rows = [r for _, r in cell_rows]
        self.n = len(cell_rows)
        self.bds = [FF.BinData(pr, field, "main", c, r) for c, r in cell_rows]
        w = [pr["weight"][r] / pr["weight"][r].sum() for r in self.rows]
        self.N = np.array([len(r) for r in self.rows])
        self.neff = np.array([1 / np.sum(ww**2) for ww in w])
        self.logm = np.array([ww @ np.log10(pr["m200c_msun"][r]) for ww, r in zip(w, self.rows)])
        self.lnc = np.array([ww @ np.log(conc[r]) for ww, r in zip(w, self.rows)])
        self.mean = np.array([bd.mean for bd in self.bds])
        self.strict = np.array([pr["strict_mask"][GC.CAL_BINS[c]] & bd.base for c, bd in zip(self.bin, self.bds)])
        self.T2h = np.array([PH.two_halo_template(lh, pr["m200c_msun"][r], pr["r200c_mpc"][r], pr["weight"][r], x35)
                             for r in self.rows])
        self.lnF = [np.log(np.where(pr[f"{field}_main"][r] > 0, pr[f"{field}_main"][r], np.nan)) for r in self.rows]
        self.w = w
        self.targets = None
        if gls:
            for i, bd in enumerate(self.bds):
                bd.prepare_gls(seed + 31 * i + (7 if field == "E" else 0) + (500 if dset == "massc" else 0))
            self.targets = FF.pooled_targets(self.bds)

    def masks(self, r_fit):
        return self.strict & GC.x_hi_le(GC.F_EDGES, r_fit)[None, :]


def make_cells(pr, dset, conc):
    out = []
    for c, b in enumerate(GC.CAL_BINS):
        rows = np.flatnonzero(pr["mass_bin"] == b)
        if dset == "mass":
            out.append((c, rows))
            continue
        n = len(rows)
        k = 4 if n >= QUARTILE_N else (2 if n >= HALVES_N else 1)
        if k == 1:
            out.append((c, rows))
            continue
        order = rows[np.argsort(conc[rows], kind="stable")]
        cw = np.cumsum(pr["weight"][order]) / pr["weight"][order].sum()
        edges = np.searchsorted(cw, np.arange(1, k) / k)
        for part in np.split(order, edges):
            out.append((c, np.sort(part)))
    return out


def covariances(cells, r_fit):
    """Per cell: mask, shrunk bootstrap Cov[ln mean] (stat), Lee22-convention covariance and Hartlap factors."""
    masks = cells.masks(r_fit)
    out = []
    for i, bd in enumerate(cells.bds):
        m = masks[i]
        p = int(m.sum())
        C, lam, forced = bd.gls_cov(m, cells.targets[i])
        N = int(cells.N[i])
        h = (N - p - 2) / (N - 1) if N > p + 2 else np.nan
        L = cells.lnF[i][:, m]
        w = cells.w[i]
        ok = np.all(np.isfinite(L), axis=0)
        Cl = None
        if np.isfinite(h) and ok.all():
            mu = w @ L
            d = L - mu
            Cs = (d.T * w) @ d / max(1 - np.sum(w**2), 1e-12)          # weighted per-halo covariance of ln F
            Cl = Cs / cells.neff[i]
        out.append({"mask": m, "p": p, "C": C, "shrink_lambda": lam, "pooled_only": forced, "N": N,
                    "hartlap": h, "C_L22": Cl})
    return out


# ============================================================================ models
class SelModel:
    def __init__(self, name, field, laws, cterms=(), gamma="fixed", mcut=None, outer="O0", frozen=None, n_bins=None,
                 per_cell=False):
        self.name, self.field, self.laws, self.cterms = name, field, dict(laws), tuple(cterms)
        self.gamma, self.mcut, self.outer, self.per_cell = gamma, mcut, outer, per_cell
        self.frozen = dict(frozen or {})
        nb = GC.N_CAL if n_bins is None else n_bins
        self.n_bins = nb
        pars = []
        lims = {"A": (np.log(1e-5), np.log(1e6)), "xc": (np.log(0.01), np.log(10.0)), "beta": (np.log(0.2), np.log(30.0))}
        for q in QN:
            law = self.laws[q]
            lo, hi = lims[q]
            if law == "bin":
                pars += [(f"{q}_b{b}", lo, hi) for b in range(nb)]
            elif law == "const":
                pars.append((f"{q}_0", lo, hi))
            elif law == "pl":
                pars += [(f"{q}_0", lo, hi), (f"{q}_am", -6.0, 6.0)]
            elif law == "bpl":
                pars += [(f"{q}_0", lo, hi), (f"{q}_am", -6.0, 6.0), (f"{q}_amp", -6.0, 6.0)]
            if q in self.cterms:
                pars.append((f"{q}_ac", -6.0, 6.0))
        self.has_bpl = any(v == "bpl" for v in self.laws.values())
        if self.has_bpl and mcut == "free":
            pars.append(("log10Mcut", 13.5, 15.3))
        if gamma == "const":
            pars.append(("g0", -2.0, 1.2))
        elif gamma == "pl":
            pars += [("g0", -2.0, 1.2), ("g_am", -3.0, 3.0)]
        if outer in ("O1", "O2") and field == "P":
            pars += [("ln_tau0", np.log(1e-4), np.log(30.0)), ("q_tau", -4.0, 3.9)]
        if outer == "O2":
            pars += [("ln_xt0", np.log(0.5), np.log(20.0)), ("a_xt", -3.0, 3.0)]
        self.all_names = [p[0] for p in pars]
        self.free = [p for p in pars if p[0] not in self.frozen]
        self.names = [p[0] for p in self.free]
        self.lo = np.array([p[1] for p in self.free])
        self.hi = np.array([p[2] for p in self.free])
        self.k = len(self.free)
        self._ix = {n: i for i, n in enumerate(self.names)}

    def value(self, theta, name):
        if name in self.frozen:
            return self.frozen[name]
        return theta[self._ix[name]]

    def ix(self, name):
        return self._ix.get(name)

    # ---------------- per-cell parameters and their Jacobian
    def cell_params(self, theta, cells, logm=None, lnc=None):
        logm = cells.logm if logm is None else logm
        lnc = cells.lnc if lnc is None else lnc
        n = len(logm)
        phi = np.zeros((n, 4))
        J = np.zeros((n, 4, self.k))
        lnMp = LN10 * (logm - LOGM_PIV)
        lncp = lnc - np.log(C_PIV)
        if self.has_bpl:
            mc = self.value(theta, "log10Mcut") if self.mcut == "free" else float(self.mcut)
            lnMc = LN10 * (logm - mc)
            below = lnMc < 0
        for j, q in enumerate(QN):
            law = self.laws[q]
            if law == "bin":
                for i in range(n):
                    b = cells.bin[i] if not self.per_cell else i
                    nm = f"{q}_b{b}"
                    phi[i, j] = self.value(theta, nm)
                    if self.ix(nm) is not None:
                        J[i, j, self.ix(nm)] = 1.0
            elif law == "const":
                phi[:, j] = self.value(theta, f"{q}_0")
                if self.ix(f"{q}_0") is not None:
                    J[:, j, self.ix(f"{q}_0")] = 1.0
            elif law == "pl":
                v0, am = self.value(theta, f"{q}_0"), self.value(theta, f"{q}_am")
                phi[:, j] = v0 + am * lnMp
                for nm, d in ((f"{q}_0", 1.0), (f"{q}_am", lnMp)):
                    if self.ix(nm) is not None:
                        J[:, j, self.ix(nm)] = d
            elif law == "bpl":
                v0, am, amp = (self.value(theta, f"{q}_{s}") for s in ("0", "am", "amp"))
                slope = np.where(below, am, amp)
                phi[:, j] = v0 + slope * lnMc
                if self.ix(f"{q}_0") is not None:
                    J[:, j, self.ix(f"{q}_0")] = 1.0
                if self.ix(f"{q}_am") is not None:
                    J[:, j, self.ix(f"{q}_am")] = np.where(below, lnMc, 0.0)
                if self.ix(f"{q}_amp") is not None:
                    J[:, j, self.ix(f"{q}_amp")] = np.where(below, 0.0, lnMc)
                if self.mcut == "free":
                    J[:, j, self.ix("log10Mcut")] += -slope * LN10
            if q in self.cterms:
                ac = self.value(theta, f"{q}_ac")
                phi[:, j] += ac * lncp
                if self.ix(f"{q}_ac") is not None:
                    J[:, j, self.ix(f"{q}_ac")] = lncp
        if self.gamma == "fixed":
            phi[:, 3] = -0.3
        elif self.gamma == "const":
            phi[:, 3] = self.value(theta, "g0")
            J[:, 3, self.ix("g0")] = 1.0
        else:
            g0, ga = self.value(theta, "g0"), self.value(theta, "g_am")
            e = np.exp(ga * lnMp)
            phi[:, 3] = g0 * e
            J[:, 3, self.ix("g0")] = e
            J[:, 3, self.ix("g_am")] = g0 * e * lnMp
        return phi, J, lnMp

    # ---------------- shells and d ln S / d theta
    def evaluate(self, theta, cells, x, wq, S, T2h, ctx, jac=True, logm=None, lnc=None, factor=None, parts=False):
        phi, Jphi, lnMp = self.cell_params(theta, cells, logm, lnc)
        A, xc, beta, gam = np.exp(phi[:, 0]), np.exp(phi[:, 1]), np.exp(phi[:, 2]), phi[:, 3]
        X = x[None, :S, :]
        q = X / xc[:, None, None]
        lq, l1q = np.log(q), np.log1p(q)
        Fin = A[:, None, None] * np.exp(gam[:, None, None] * lq - beta[:, None, None] * l1q)
        if self.outer == "O2":
            lnxt = self.value(theta, "ln_xt0") + self.value(theta, "a_xt") * lnMp
            u = (X / np.exp(lnxt)[:, None, None])**4
            Fin = Fin * (1 + u)**(-1.5)
        if self.outer in ("O1", "O2"):
            n2h = ctx["ebar"] * (1 + T2h[:, :S])
            if self.field == "E":
                Fout = n2h
            else:
                Fout = ctx["kappa_P"] / 200.0 * n2h * np.exp(self.value(theta, "ln_tau0")) \
                    * (X / 2.0)**(-self.value(theta, "q_tau")) * GM.damping(X)
        else:
            Fout = np.zeros_like(Fin)
        if factor is not None:
            Fin, Fout = Fin * factor[:, :S], Fout * factor[:, :S]
        w = wq[None, :S, :]
        Sh = np.sum((Fin + Fout) * w, axis=-1)
        if parts:
            return np.sum(Fin * w, axis=-1), np.sum(Fout * w, axis=-1)
        if not jac:
            return np.log(Sh), None
        Fw = Fin * w
        dS = np.stack([Fw.sum(-1), (Fw * (-gam[:, None, None] + beta[:, None, None] * q / (1 + q))).sum(-1),
                       (Fw * (-beta[:, None, None] * l1q)).sum(-1), (Fw * lq).sum(-1)], axis=-1)        # [n, S, 4]
        D = np.einsum("nsj,njt->nst", dS / Sh[..., None], Jphi)
        if self.outer == "O2":
            dxt = (Fw * 6 * u / (1 + u)).sum(-1) / Sh
            if self.ix("ln_xt0") is not None:
                D[:, :, self.ix("ln_xt0")] += dxt
            if self.ix("a_xt") is not None:
                D[:, :, self.ix("a_xt")] += dxt * lnMp[:, None]
        if self.outer in ("O1", "O2") and self.field == "P":
            Fo = Fout * w
            if self.ix("ln_tau0") is not None:
                D[:, :, self.ix("ln_tau0")] += Fo.sum(-1) / Sh
            if self.ix("q_tau") is not None:
                D[:, :, self.ix("q_tau")] += (Fo * (-np.log(X / 2.0))).sum(-1) / Sh
        return np.log(Sh), D


def model_specs(field, dset):
    """Inner ladder (name, laws, cterms, gamma, mcut kind)."""
    pl3 = {"A": "pl", "xc": "pl", "beta": "pl"}
    l22 = ({"A": "bpl", "xc": "pl", "beta": "pl"} if field == "P" else {"A": "pl", "xc": "bpl", "beta": "bpl"})
    l22n = "BPLa" if field == "P" else "BPLs"
    out = [("PB", {"A": "bin", "xc": "bin", "beta": "bin"}, (), "fixed", None),
           ("PL3", pl3, (), "fixed", None),
           ("PL2", {"A": "pl", "xc": "pl", "beta": "const"}, (), "fixed", None),
           ("PL1", {"A": "pl", "xc": "const", "beta": "const"}, (), "fixed", None)]
    if dset == "mass":
        out += [("PL1s", {"A": "pl", "xc": "const", "beta": "const"}, (), "fixed", "2step"),
                ("BPLa", {"A": "bpl", "xc": "pl", "beta": "pl"}, (), "fixed", "free"),
                ("BPLa_L22", {"A": "bpl", "xc": "pl", "beta": "pl"}, (), "fixed", "L22"),
                ("BPLs", {"A": "pl", "xc": "bpl", "beta": "bpl"}, (), "fixed", "free"),
                ("BPLs_L22", {"A": "pl", "xc": "bpl", "beta": "bpl"}, (), "fixed", "L22"),
                ("BPLall", {"A": "bpl", "xc": "bpl", "beta": "bpl"}, (), "fixed", "free"),
                ("BPLall_L22", {"A": "bpl", "xc": "bpl", "beta": "bpl"}, (), "fixed", "L22"),
                ("PL3g", pl3, (), "const", None),
                ("PL3gM", pl3, (), "pl", None)]
    else:
        out += [("PBcell", {"A": "bin", "xc": "bin", "beta": "bin"}, (), "fixed", "percell"),
                (l22n, l22, (), "fixed", "free"), (l22n + "_L22", l22, (), "fixed", "L22"),
                ("PL3g", pl3, (), "const", None),
                ("PL1cA", {"A": "pl", "xc": "const", "beta": "const"}, ("A",), "fixed", None),
                ("PL3cA", pl3, ("A",), "fixed", None),
                ("PL3c3", pl3, ("A", "xc", "beta"), "fixed", None),
                ("L22best", l22, ("A", "xc", "beta"), "fixed", "free"),
                ("L22best_L22", l22, ("A", "xc", "beta"), "fixed", "L22")]
    return out


def build_model(field, spec, outer, n_cells, frozen=None):
    name, laws, cterms, gamma, mk = spec
    if mk == "L22":
        mcut = lee22_mcut(field, bool(cterms))
    elif mk == "free":
        mcut = "free"
    else:
        mcut = None
    per_cell = mk == "percell"
    return SelModel(f"{name}+{outer}", field, laws, cterms, gamma, mcut, outer, frozen=frozen,
                    n_bins=n_cells if per_cell else None, per_cell=per_cell)


# ============================================================================ likelihood and fitting
class Problem:
    def __init__(self, model, cells, covs, which, sigma2, ctx, r_fit, means=None, cov_scale=None, use=None):
        """which: 'stat' / 'floor' (C + sigma2 I) / 'L22' (h C_L22^-1); means: override stacks (CV training);
        cov_scale [cells]: multiply C (CV); use: boolean over cells."""
        self.model, self.cells, self.ctx = model, cells, ctx
        self.x, self.wq = PH.quad_nodes(GC.F_EDGES)
        self.S = int(np.sum(GC.x_hi_le(GC.F_EDGES, r_fit)))
        self.means = cells.mean if means is None else means
        self.Wh, self.idx, self.use = [], [], np.ones(cells.n, bool) if use is None else np.asarray(use, bool)
        for i, cv in enumerate(covs):
            m = cv["mask"][:self.S]
            if not self.use[i] or m.sum() == 0:
                self.use[i] = False
                self.Wh.append(None)
                self.idx.append(None)
                continue
            if which == "L22":
                if cv["C_L22"] is None:
                    self.use[i] = False
                    self.Wh.append(None)
                    self.idx.append(None)
                    continue
                C = cv["C_L22"] / cv["hartlap"]
            else:
                C = cv["C"] * (1.0 if cov_scale is None else cov_scale[i])
                if which == "floor":
                    C = C + sigma2 * np.eye(len(C))
            L = np.linalg.cholesky((C + C.T) / 2)
            self.Wh.append(np.linalg.inv(L))
            self.idx.append(np.flatnonzero(m))
        self.n_data = int(sum(len(i) for i, u in zip(self.idx, self.use) if u))
        self._cache = (None, None)

    def _eval(self, theta):
        key = theta.tobytes()
        if self._cache[0] != key:
            lnS, D = self.model.evaluate(theta, self.cells, self.x, self.wq, self.S, self.cells.T2h, self.ctx)
            self._cache = (key, (lnS, D))
        return self._cache[1]

    def residuals(self, theta, split=False):
        lnS, _ = self._eval(theta)
        out = []
        for i in range(self.cells.n):
            if not self.use[i]:
                out.append(np.zeros(0))
                continue
            s = self.idx[i]
            out.append(self.Wh[i] @ (lnS[i, s] - np.log(self.means[i, s])))
        return out if split else np.concatenate(out)

    def jac(self, theta):
        _, D = self._eval(theta)
        rows = []
        for i in range(self.cells.n):
            if self.use[i]:
                rows.append(self.Wh[i] @ D[i, self.idx[i]])
        return np.vstack(rows)

    def fit(self, starts):
        best = None
        for th0 in starts:
            th0 = np.clip(np.asarray(th0, float), self.model.lo + 1e-9, self.model.hi - 1e-9)
            try:
                sol = least_squares(self.residuals, th0, jac=self.jac, bounds=(self.model.lo, self.model.hi), method="trf",
                                    x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=MAX_NFEV)
            except Exception:                    # noqa: BLE001 - a failed start is skipped
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        return best


def chi2_by_cell(prob, theta):
    return np.array([float(np.sum(r**2)) for r in prob.residuals(theta, split=True)])


def solve_sigma(cells, covs, residual_lists, nu):
    """sigma_d^2 with sum_cells r^T (C + s I)^-1 r = nu (raw residuals r = ln S - ln m)."""
    eig = []
    for cv, r in zip(covs, residual_lists):
        if r is None or len(r) == 0:
            continue
        ev, U = np.linalg.eigh((cv["C"] + cv["C"].T) / 2)
        eig.append((np.maximum(ev, 0.0), (U.T @ r)**2))

    def g(s):
        return sum(float(np.sum(u2 / (ev + s))) for ev, u2 in eig) - nu

    if g(0.0) <= 0:
        return 0.0
    hi = 1e-6
    while g(hi) > 0:
        hi *= 4
    return brentq(g, 0.0, hi, xtol=1e-20, rtol=1e-12)


def raw_residuals(model, theta, cells, covs, r_fit, ctx):
    x, wq = PH.quad_nodes(GC.F_EDGES)
    S = int(np.sum(GC.x_hi_le(GC.F_EDGES, r_fit)))
    lnS, _ = model.evaluate(theta, cells, x, wq, S, cells.T2h, ctx, jac=False)
    out = []
    for i, cv in enumerate(covs):
        idx = np.flatnonzero(cv["mask"][:S])
        out.append(lnS[i, idx] - np.log(cells.mean[i, idx]) if len(idx) else None)
    return out


# ============================================================================ starting values
def pb_start(cells, covs, r_fit, ctx, outer, field):
    """Per-bin starts of the PB inner parameters: per-cell least squares (unweighted ln), multi-start in (x_c, beta)."""
    x, wq = PH.quad_nodes(GC.F_EDGES)
    S = int(np.sum(GC.x_hi_le(GC.F_EDGES, r_fit)))
    out = np.zeros((GC.N_CAL, 3))
    for b in range(GC.N_CAL):
        sel = np.flatnonzero(cells.bin == b)
        m = covs[sel[0]]["mask"][:S]
        lnm = np.log(np.mean(cells.mean[sel][:, :S][:, m], axis=0))
        best = None
        for xc0 in (0.2, 0.5, 1.0):
            for b0 in (3.0, 5.0):
                def res(p, m=m, lnm=lnm):
                    lnA, lnxc, lnb = p
                    qq = x[:S] / np.exp(lnxc)
                    Fs = np.sum(np.exp(lnA - 0.3 * np.log(qq) - np.exp(lnb) * np.log1p(qq)) * wq[:S], axis=-1)
                    return np.log(Fs[m]) - lnm
                p0 = np.array([0.0, np.log(xc0), np.log(b0)])
                p0[0] = -np.mean(res(p0))
                sol = least_squares(res, p0, bounds=([np.log(1e-5), np.log(0.01), np.log(0.2)],
                                                     [np.log(1e6), np.log(10.0), np.log(30.0)]), method="trf")
                if best is None or sol.cost < best.cost:
                    best = sol
        out[b] = best.x
    return out


def starts_for(model, pbpar, cells, field, extra=None):
    """Starting vectors: regression of the per-bin PB parameters on ln M (and 0 for c and break terms)."""
    logm_bin = np.array([np.average(cells.logm[cells.bin == b]) for b in range(GC.N_CAL)])
    lnMp = LN10 * (logm_bin - LOGM_PIV)
    th = {}
    for j, q in enumerate(QN):
        y = pbpar[:, j]
        slope, icpt = np.polyfit(lnMp, y, 1)
        law = model.laws[q]
        if law == "bin":
            if model.per_cell:
                for i in range(cells.n):
                    th[f"{q}_b{i}"] = y[cells.bin[i]]
            else:
                for b in range(GC.N_CAL):
                    th[f"{q}_b{b}"] = y[b]
        elif law == "const":
            th[f"{q}_0"] = float(np.mean(y))
        elif law == "pl":
            th[f"{q}_0"], th[f"{q}_am"] = icpt, slope
        elif law == "bpl":
            mc = float(model.mcut) if model.mcut not in (None, "free") else lee22_mcut(field, bool(model.cterms))
            th[f"{q}_0"] = icpt + slope * LN10 * (mc - LOGM_PIV)
            th[f"{q}_am"] = th[f"{q}_amp"] = slope
        if q in model.cterms:
            th[f"{q}_ac"] = 0.0
    th["log10Mcut"] = min(max(lee22_mcut(field, bool(model.cterms)), 13.6), 15.2)
    th["g0"], th["g_am"] = -0.3, 0.0
    th["ln_tau0"], th["q_tau"] = np.log(0.3), 1.0
    th["ln_xt0"], th["a_xt"] = np.log(2.5 if field == "E" else 3.0), 0.0
    th.update(extra or {})
    base = np.array([th[n] for n in model.names])
    starts = [base]
    if "ln_xt0" in model.names:
        alt = base.copy()
        alt[model.ix("ln_xt0")] = np.log(4.0)
        starts.append(alt)
    if "log10Mcut" in model.names:
        alt = base.copy()
        alt[model.ix("log10Mcut")] = 14.3
        starts.append(alt)
    if "ln_tau0" in model.names:
        alt = base.copy()
        alt[model.ix("ln_tau0")] = np.log(0.1)
        alt[model.ix("q_tau")] = 0.3
        starts.append(alt)
    return starts


# ============================================================================ worker tasks
def _sigma_job(task):
    field, dset, r_fit = task
    cells, covs, ctx = _W["cells"][(field, dset)], _W["covs"][(field, dset, r_fit)], _W["ctx"]
    pbpar = _W["pbstart"][(field, dset, r_fit)]
    outers = ("O0",) if r_fit <= 1.34 + 1e-9 else ("O0", "O1", "O2")
    # the per-bin reference: PB on the mass bins; PBcell (free per mass x concentration cell) on the c-split cells
    spec = [s for s in model_specs(field, dset) if s[0] == ("PB" if dset == "mass" else "PBcell")][0]
    out = {}
    for o in outers:
        model = build_model(field, spec, o, cells.n)
        starts = starts_for(model, pbpar, cells, field)
        s2 = 0.0
        theta = None
        for it in range(25):
            prob = Problem(model, cells, covs, "floor", s2, ctx, r_fit)
            sol = prob.fit(starts if theta is None else [theta] + starts[:1])
            theta = sol.x
            nu = prob.n_data - model.k
            new = solve_sigma(cells, covs, raw_residuals(model, theta, cells, covs, r_fit, ctx), nu)
            done = abs(np.sqrt(new) - np.sqrt(s2)) <= 1e-5 * max(np.sqrt(new), 1e-6)
            s2 = new
            if done:
                break
        out[o] = {"sigma_d": float(np.sqrt(s2)), "iterations": it + 1, "theta": theta}
    best = min(out, key=lambda o: out[o]["sigma_d"])
    return (field, dset, r_fit), out, best


def _model_job(task):
    field, dset, r_fit, outer, spec_name = task
    cells, covs, ctx = _W["cells"][(field, dset)], _W["covs"][(field, dset, r_fit)], _W["ctx"]
    sig = _W["sigma"][(field, dset, r_fit)]
    s2 = sig["sigma_ref"]**2
    pbpar = _W["pbstart"][(field, dset, r_fit)]
    spec = [s for s in model_specs(field, dset) if s[0] == spec_name][0]
    t0 = time.perf_counter()
    two_step = spec[4] == "2step"
    res = {"field": field, "dset": dset, "R_fit": r_fit, "outer": outer, "inner": spec_name}
    if two_step:
        shape_spec = ("PBAshape", {"A": "bin", "xc": "const", "beta": "const"}, (), "fixed", None)
        m1 = build_model(field, shape_spec, outer, cells.n)
        p1 = Problem(m1, cells, covs, "floor", s2, ctx, r_fit)
        s1 = p1.fit(starts_for(m1, pbpar, cells, field))
        frozen = {"xc_0": s1.x[m1.ix("xc_0")], "beta_0": s1.x[m1.ix("beta_0")]}
        model = build_model(field, spec, outer, cells.n, frozen=frozen)
        res["frozen_shape"] = frozen
        k_count = model.k + 2
    else:
        model = build_model(field, spec, outer, cells.n)
        k_count = model.k
    res["model"] = model.name
    res["k"] = k_count
    starts = starts_for(model, pbpar, cells, field)
    fits = {}
    for which in ("floor", "stat", "L22"):
        prob = Problem(model, cells, covs, which, s2, ctx, r_fit)
        if prob.n_data == 0:
            continue
        st = starts if which == "floor" or "floor" not in fits else [fits["floor"]["theta"]]
        sol = prob.fit(st)
        if sol is None:
            fits[which] = None
            continue
        c2 = chi2_by_cell(prob, sol.x)
        fits[which] = {"theta": sol.x, "chi2": float(c2.sum()), "n_data": prob.n_data, "chi2_cells": c2,
                       "success": bool(sol.success), "nfev": int(sol.nfev), "max_nfev_reached": bool(sol.nfev >= MAX_NFEV),
                       "at_bound": [n for n, v, lo, hi in zip(model.names, sol.x, model.lo, model.hi)
                                    if min(v - lo, hi - v) < 1e-6]}
        if which == "L22":
            N = np.array([cv["N"] for cv in covs], float)
            used = prob.use
            fits[which]["chi2_scatter"] = float(np.sum(c2[used] / N[used]))
            fits[which]["cells_used"] = int(used.sum())
    fits = {k: v for k, v in fits.items() if v is not None}
    res["fits"] = fits
    res["names"] = list(model.names)
    # ---------------- spatial CV (floor likelihood)
    if "floor" in fits:
        res["cv"] = cv_model(model, spec, cells, covs, r_fit, s2, ctx, fits["floor"]["theta"], two_step, pbpar)
    res["seconds"] = time.perf_counter() - t0
    return res


def cv_model(model, spec, cells, covs, r_fit, s2, ctx, theta_full, two_step, pbpar):
    pr = _W["pr"]
    fold = pr["fold"]
    K = GC.FC.FOLDS
    out = np.full(K, np.nan)
    npts = np.zeros(K, int)
    for k in range(K):
        tr_rows, ho_rows = [], []
        for r in cells.rows:
            tr_rows.append(r[fold[r] != k])
            ho_rows.append(r[fold[r] == k])
        use_tr = np.array([len(r) > 0 for r in tr_rows])
        use_ho = np.array([len(r) > 0 for r in ho_rows])

        def stack(rows_list, use):
            mean = np.ones_like(cells.mean)
            logm, lnc, neff = cells.logm.copy(), cells.lnc.copy(), np.ones(cells.n)
            for i, r in enumerate(rows_list):
                if not use[i]:
                    continue
                w = pr["weight"][r] / pr["weight"][r].sum()
                mean[i] = w @ np.nan_to_num(pr[f"{cells.field}_main"][r])
                logm[i] = w @ np.log10(pr["m200c_msun"][r])
                lnc[i] = w @ np.log(_W["conc"][r])
                neff[i] = 1 / np.sum(w**2)
            return mean, logm, lnc, neff

        m_tr, lm_tr, lc_tr, ne_tr = stack(tr_rows, use_tr)
        m_ho, lm_ho, lc_ho, ne_ho = stack(ho_rows, use_ho)
        ok_tr = use_tr & np.all((m_tr > 0) | ~cells.strict, axis=1)
        cv_tr = Cellview(cells, lm_tr, lc_tr)
        prob = Problem(model, cv_tr, covs, "floor", s2, ctx, r_fit, means=m_tr, cov_scale=cells.neff / ne_tr, use=ok_tr)
        if two_step:
            shape_spec = ("PBAshape", {"A": "bin", "xc": "const", "beta": "const"}, (), "fixed", None)
            m1 = build_model(cells.field, shape_spec, model.outer, cells.n)
            p1 = Problem(m1, cv_tr, covs, "floor", s2, ctx, r_fit, means=m_tr, cov_scale=cells.neff / ne_tr, use=ok_tr)
            s1 = p1.fit(starts_for(m1, pbpar, cells, cells.field))
            model_k = build_model(cells.field, spec, model.outer, cells.n,
                                  frozen={"xc_0": s1.x[m1.ix("xc_0")], "beta_0": s1.x[m1.ix("beta_0")]})
            prob = Problem(model_k, cv_tr, covs, "floor", s2, ctx, r_fit, means=m_tr, cov_scale=cells.neff / ne_tr,
                           use=ok_tr)
            sol = prob.fit([theta_full])
            mk = model_k
        else:
            sol = prob.fit([theta_full])
            mk = model
        if sol is None:
            continue
        ok_ho = use_ho & np.all((m_ho > 0) | ~cells.strict, axis=1)
        cv_ho = Cellview(cells, lm_ho, lc_ho)
        ph = Problem(mk, cv_ho, covs, "floor", s2, ctx, r_fit, means=m_ho, cov_scale=cells.neff / ne_ho, use=ok_ho)
        out[k] = float(np.sum(ph.residuals(sol.x)**2))
        npts[k] = ph.n_data
    return {"chi2_folds": out, "n_folds": npts}


class Cellview:
    """The same cells with substituted covariates (CV training / held-out masses and concentrations)."""

    def __init__(self, cells, logm, lnc):
        self.__dict__.update({k: v for k, v in cells.__dict__.items() if k not in ("logm", "lnc")})
        self.logm, self.lnc = logm, lnc


# ============================================================================ main
def main():
    timer = GC.Timer()
    pr = dict(np.load(GC.FBASE / "profiles.npz"))
    prep = GC.load_prepare()
    bhalo = np.load(GC.B_HALOS)
    conc = np.asarray(bhalo["concentration_200c"], float)[pr["halo_rows"]]
    lh = PH.LinearHalo()
    ctx = {"ebar": float(prep["ebar"]), "kappa_P": float(prep["kappa_p"])}
    cells, covs, pbstart = {}, {}, {}
    cell_rows = []
    for field in ("P", "E"):
        for dset in ("mass", "massc"):
            cs = Cells(pr, field, dset, conc, lh)
            cells[(field, dset)] = cs
            for rf in R_FIT_SEL:
                covs[(field, dset, rf)] = covariances(cs, rf)
                pbstart[(field, dset, rf)] = pb_start(cs, covs[(field, dset, rf)], rf, ctx, "O0", field)
                for i, cv in enumerate(covs[(field, dset, rf)]):
                    cell_rows.append({"field": field, "dset": dset, "R_fit": rf, "cell": i, "bin": GC.CAL_LABELS[cs.bin[i]],
                                      "N": cv["N"], "N_eff": cs.neff[i], "logM": cs.logm[i], "c_mean": float(np.exp(cs.lnc[i])),
                                      "shells": cv["p"], "hartlap": cv["hartlap"], "L22_used": cv["C_L22"] is not None,
                                      "shrink_lambda": cv["shrink_lambda"], "pooled_only": cv["pooled_only"]})
    GC.log("selection cells and covariances", round(timer.mark("cells"), 1))
    _W.clear()
    _W.update(cells=cells, covs=covs, ctx=ctx, pbstart=pbstart, pr=pr, conc=conc)
    sig_tasks = [(f, d, rf) for f in ("P", "E") for d in ("mass", "massc") for rf in R_FIT_SEL]
    sigma = {}
    for key, out, best in GC.pool_map(_sigma_job, sig_tasks, ordered=False):
        sigma[key] = {"by_outer": {o: {"sigma_d": v["sigma_d"], "iterations": v["iterations"]} for o, v in out.items()},
                      "reference_outer": best, "sigma_ref": out[best]["sigma_d"]}
    _W["sigma"] = sigma
    GC.log("sigma_d floors", round(timer.mark("sigma"), 1))
    tasks = []
    for (f, d, rf) in sig_tasks:
        outers = ("O0",) if rf <= 1.34 + 1e-9 else ("O0", "O1", "O2")
        for o in outers:
            for spec in model_specs(f, d):
                tasks.append((f, d, rf, o, spec[0]))
    results = []
    for res in GC.pool_map(_model_job, tasks, ordered=False):
        results.append(res)
    _W.clear()
    GC.log(f"selection fits: {len(results)} models", round(timer.mark("models"), 1))
    rows, ladder, chosen = summarise(results, sigma)
    # ---------------- observables of the selected minimal models (mass data set) and PB with the same outer option
    obs_rows = propagate(pr, prep, cells, chosen, results, ctx)
    GC.write_csv(GC.TABLES / "G_selection_models.csv", rows)
    GC.write_csv(GC.TABLES / "G_selection_ladder.csv", ladder)
    GC.write_csv(GC.TABLES / "G_selection_cells.csv", cell_rows)
    GC.write_csv(GC.TABLES / "G_selection_observables.csv", obs_rows)
    np.savez_compressed(GC.RESULTS / "selection.npz", results_json=np.array(GC.json.dumps(GC.plain(results))))
    GC.write_json(GC.RESULTS / "selection_summary.json", {
        "sigma_floor": sigma, "chosen": chosen, "lee22_mcut_msun": {f"{k[0]}{'_c' if k[1] else ''}": lee22_mcut(*k)
                                                                     for k in LEE22_MCUT},
        "conventions": {"stat": "bootstrap Cov[ln mean], 1000 draws, shrinkage to the neighbours' pooled correlation",
                        "floor": "stat + sigma_d^2 I; sigma_d from the per-bin reference PB (best outer option) at chi2/nu = 1",
                        "L22": "Lee22: sample covariance of per-halo ln F / N_eff, precision x Hartlap (N - p - 2)/(N - 1); "
                               "cells with N <= p + 2 dropped; chi2_scatter = per-halo-scatter covariance (x N)",
                        "selection_rule": "fewest parameters with dBIC <= 6 (floor) and CV chi2 <= CV_best + sqrt(K) SD_k(best)"},
        "seconds": timer.marks})
    GC.log("selection done", round(timer.mark("total"), 1))


def summarise(results, sigma):
    rows = []
    for r in results:
        fl, st, l2 = r["fits"].get("floor"), r["fits"].get("stat"), r["fits"].get("L22")
        k = r["k"]
        row = {"field": r["field"], "dset": r["dset"], "R_fit": r["R_fit"], "outer": r["outer"], "inner": r["inner"],
               "model": r["model"], "k": k, "sigma_floor": sigma[(r["field"], r["dset"], r["R_fit"])]["sigma_ref"]}
        for tag, fit in (("stat", st), ("floor", fl), ("L22", l2)):
            if fit is None:
                continue
            n = fit["n_data"]
            nu = n - k
            row[f"n_{tag}"] = n
            row[f"chi2_{tag}"] = fit["chi2"]
            row[f"nu_{tag}"] = nu
            row[f"chi2nu_{tag}"] = fit["chi2"] / nu if nu > 0 else np.nan
            row[f"spread_{tag}"] = np.sqrt(2.0 / nu) if nu > 0 else np.nan
            row[f"p_{tag}"] = float(CHI2.sf(fit["chi2"], nu)) if nu > 0 else np.nan
            row[f"AIC_{tag}"] = fit["chi2"] + 2 * k
            row[f"BIC_{tag}"] = fit["chi2"] + k * np.log(n)
            row[f"at_bound_{tag}"] = ";".join(fit["at_bound"])
            row[f"max_nfev_{tag}"] = fit.get("max_nfev_reached", False)
        if l2 is not None:
            row["chi2nu_L22_scatter"] = l2["chi2_scatter"] / (l2["n_data"] - k) if l2["n_data"] > k else np.nan
            row["L22_cells_used"] = l2.get("cells_used")
        if "cv" in r:
            c = r["cv"]["chi2_folds"]
            row["cv_chi2"] = float(np.nansum(c))
            row["cv_n"] = int(np.sum(r["cv"]["n_folds"]))
            row["cv_chi2_per_point"] = row["cv_chi2"] / max(row["cv_n"], 1)
            row["cv_fold_sd"] = float(np.nanstd(c, ddof=1))
        if fl is not None:
            row["params"] = GC.json.dumps({n: float(v) for n, v in zip(r["names"], fl["theta"])})
            if r.get("frozen_shape"):
                row["frozen_shape"] = GC.json.dumps({n: float(v) for n, v in r["frozen_shape"].items()})
        rows.append(row)
    chosen = {}
    ladder = []
    groups = {}
    for row in rows:
        groups.setdefault((row["field"], row["dset"], row["R_fit"]), []).append(row)
    res_by = {(r["field"], r["dset"], r["R_fit"], r["model"]): r for r in results}
    for key, grp in groups.items():
        grp = [g for g in grp if np.isfinite(g.get("BIC_floor", np.nan))]
        if not grp:
            continue
        bmin = min(g["BIC_floor"] for g in grp)
        best_cv = min(grp, key=lambda g: g.get("cv_chi2", np.inf))
        r_best = res_by[(key[0], key[1], key[2], best_cv["model"])]
        cfold = r_best["cv"]["chi2_folds"]
        scatter = float(np.sqrt(np.sum(np.isfinite(cfold))) * np.nanstd(cfold, ddof=1))
        for g in grp:
            g["dBIC_floor"] = g["BIC_floor"] - bmin
            g["dAIC_floor"] = g["AIC_floor"] - min(x["AIC_floor"] for x in grp)
            g["cv_best_model"] = best_cv["model"]
            g["cv_minus_best"] = g.get("cv_chi2", np.nan) - best_cv.get("cv_chi2", np.nan)
            g["cv_scatter_best"] = scatter
            rg = res_by[(key[0], key[1], key[2], g["model"])]
            d = rg["cv"]["chi2_folds"] - cfold
            g["cv_paired_se"] = float(np.sqrt(np.sum(np.isfinite(d))) * np.nanstd(d, ddof=1)) if np.isfinite(d).sum() > 1 else np.nan
            g["pass_bic"] = bool(g["dBIC_floor"] <= 6.0)
            g["pass_cv"] = bool(g["cv_minus_best"] <= scatter)
        cand = [g for g in grp if g["pass_bic"] and g["pass_cv"]]
        sel = min(cand, key=lambda g: (g["k"], g["BIC_floor"])) if cand else min(grp, key=lambda g: g["BIC_floor"])
        for g in grp:
            g["selected"] = g is sel
        chosen[f"{key[0]}|{key[1]}|{key[2]:g}"] = {"model": sel["model"], "k": sel["k"], "dBIC": sel["dBIC_floor"],
                                                   "chi2nu_floor": sel["chi2nu_floor"], "rule_satisfied": bool(cand),
                                                   "best_BIC": min(grp, key=lambda g: g["BIC_floor"])["model"],
                                                   "best_CV": best_cv["model"]}
        # parsimony ladder: best (lowest floor chi^2) model at each parameter count
        ks = sorted({g["k"] for g in grp})
        prev = None
        for kk in ks:
            rung = min([g for g in grp if g["k"] == kk], key=lambda g: g["chi2_floor"])
            step = {"field": key[0], "dset": key[1], "R_fit": key[2], "k": kk, "model": rung["model"],
                    "chi2_floor": rung["chi2_floor"], "chi2nu_floor": rung["chi2nu_floor"], "BIC_floor": rung["BIC_floor"],
                    "dBIC_floor": rung["dBIC_floor"], "cv_chi2": rung.get("cv_chi2"), "pass_bic": rung["pass_bic"],
                    "pass_cv": rung["pass_cv"], "selected": rung["selected"]}
            if prev is not None:
                step.update(d_k=kk - prev["k"], d_chi2=rung["chi2_floor"] - prev["chi2_floor"],
                            d_BIC=rung["BIC_floor"] - prev["BIC_floor"], from_model=prev["model"])
            ladder.append(step)
            prev = rung
    return rows, ladder, chosen


def chosen_inner(chosen, it):
    ch = chosen.get(f"{it['field']}|{it['dset']}|{it['R_fit']:g}")
    return ch["model"].split("+")[0] if ch else None


def model_by_name(field, dset, name, n_cells):
    inner, outer = name.split("+")
    spec = [s for s in model_specs(field, dset) if s[0] == inner][0]
    return build_model(field, spec, outer, n_cells), spec


def propagate(pr, prep, cells, chosen, results, ctx):
    """Observables (R_cut = 4) of the selected minimal model of every (field, R_fit) on the mass data set and of PB
    with the same outer option; c models (massc) are painted at each bin's mean ln c."""
    import f_basis as FB
    import g_evaluate as GE
    import g_pdf as GP
    tr = dict(np.load(GC.FBASE / "truth.npz"))
    ts = GC.read_json(GC.FBASE / "truth_summary.json")
    T = GE.truths(pr, tr, ts)
    B = FB.Basis("T")
    x42, wq42 = PH.quad_nodes(GC.BASIS_EDGES)
    keep4 = GC.x_hi_le(GC.BASIS_EDGES, 4.0)
    res_by = {(r["field"], r["dset"], r["R_fit"], r["model"]): r for r in results}
    lh = PH.LinearHalo()
    tpl = PH.bin_templates(lh, pr)
    pk = {f: PH.pkeep_nodes(prep[f"pkeep_params_{f}"], tpl["logm"]) for f in ("P", "E")}
    items = []
    for key, ch in chosen.items():
        f, dset, rf = key.split("|")
        rf = float(rf)
        outer = ch["model"].split("+")[1]
        names = [ch["model"]]
        for extra in ("PB", "PL3", "PL1"):
            nm = f"{extra}+{outer}"
            if nm not in names:
                names.append(nm)
        for name in names:
            r = res_by.get((f, dset, rf, name))
            if r is None or "floor" not in r["fits"]:
                continue
            frozen = r.get("frozen_shape")
            inner, outer = name.split("+")
            spec = [s for s in model_specs(f, dset) if s[0] == inner][0]
            if spec[4] == "percell":
                continue
            model = build_model(f, spec, outer, cells[(f, dset)].n, frozen=frozen)
            # one 'cell' per calibrated bin at the bin's covariates (mean ln c for c models)
            cb = cells[(f, "mass")]
            view = Cellview(cb, cb.logm, cb.lnc)           # bins at their mean log M and mean ln c
            view.bin = np.arange(GC.N_CAL)
            T42 = tpl["T2h"]
            for paint in (False, True):
                fac = pk[f] if paint else None
                Fin, Fout = model.evaluate(r["fits"]["floor"]["theta"], view, x42, wq42, GC.N_B, T42, ctx, jac=False,
                                           factor=fac, parts=True)
                items.append({"field": f, "dset": dset, "R_fit": rf, "model": name, "selected": name == ch["model"],
                              "paint": paint, "B": Fin * keep4[None], "O": Fout * keep4[None]})
    # DM PDF (det for all; S1 at R_fit = 4)
    specs = []
    for it in items:
        if it["field"] != "E":
            continue
        tsel = "a2ii" if it["paint"] else "main"
        lab = f"{it['field']}|{it['dset']}|{it['R_fit']:g}|{it['model']}|{int(it['paint'])}"
        specs.append((lab + "|det", GP.det_entries(it["B"] + it["O"])))
        if it["R_fit"] == 4.0 and it["model"].split("+")[0] in (chosen_inner(chosen, it), "PB"):
            specs.append((lab + "|S1", GP.s1_entries(it["B"] + it["O"], prep[f"S1_{tsel}"])))
    quant = {}
    if specs:
        fb = np.load(GC.WORK / "pdf_fallback_g.npz")
        quant, _, _ = GP.run_hists(specs, (fb["hist"], float(fb["overflow"])), label="selection PDF")
    rows = []
    for it in items:
        tsel = "a2ii" if it["paint"] else "main"
        truth = T[tsel]
        lab = f"{it['field']}|{it['dset']}|{it['R_fit']:g}|{it['model']}|{int(it['paint'])}"
        zero = np.zeros_like(it["B"])
        for v in ("det", "S1"):
            if it["field"] == "P":
                BP, OP, BE, OE = it["B"], it["O"], truth["Epad"], zero
            else:
                BP, OP, BE, OE = truth["Ppad"], zero, it["B"], it["O"]
            P, E, SPP, SEE, SPE = GE.moments(v, BP, OP, BE, OE, prep[f"S1_{tsel}"], None)
            sc = B.scalars(B.observables(P, E, SPP=SPP, SEE=SEE, SPE=SPE))
            q = quant.get(f"{lab}|{v}")
            if q is not None:
                for p_, qn in enumerate(GC.FC.Q_NAMES):
                    sc[f"dm_{qn}"] = float(q[p_])
            err, errm = GE.errors(sc, truth)
            row = {k: it[k] for k in ("field", "dset", "R_fit", "model", "selected")}
            row.update(eval="paint" if it["paint"] else "all", prediction=v, dm_quantiles_included=q is not None)
            for fam in GE.FAMILIES[it["field"]]:
                row[f"rms_{fam}"] = GE.family_rms(err, fam) if (fam != "DM" or q is not None) else np.nan
                row[f"rms_{fam}_vs_mean"] = GE.family_rms(errm, fam) if (fam != "DM" or q is not None) else np.nan
            rows.append(row)
    return rows


if __name__ == "__main__":
    main()
