"""In-basis evaluation of every engine observable from per-bin shell averages and second moments.

Inputs per calibrated bin c (10 bins) on the basis shells (35 to 4R, + 7 extension shells to 12R):
  P[c, s], E[c, s]        mean shell averages (pressure Pe/P200, electrons n_e mu/(rho_c f_b))
  SPP, SEE, SPE[c, s, t]  per-halo second moments <F_s G_t> over the bin's haloes (inclusion-weighted);
                          None -> deterministic (outer products of the means: one profile per bin)
Observables (engine conventions, kernels/REPORT.md):
  yy 1h / 2h / total, mean y; kSZ 1h (catalogue v_LOS) and approximate 2h; DM-field 1h / 2h; y x DM
  1h / 2h; mean DM and the compound-Poisson DM variance (z_s = 1); per-bin Y(<x) and N_e(<x) (z = 0).
The DM PDF quantiles come from f_pdf.py (they need the per-halo columns, not just moments).
Jacobians dO/dln F[c, s] (scaling the bin's per-halo profiles in one shell) are exact for every form.
"""
from __future__ import annotations

import f_common as FC
import numpy as np


class Basis:
    def __init__(self, which: str = "T"):
        z = np.load(FC.WORK / f"basis_{which}.npz")
        self.which = which
        self.ns = int(z["n_shell"])
        self.ell = z["ell"]
        for k in ("K_yy1h", "K_ksz1h", "K_dm1h", "K_ydm1h", "K_dmvar", "mean_y", "mean_tau", "dm_mean",
                  "W_y", "W_tau", "W_fb_y", "W_fb_tau", "pk", "volume", "use_dm", "dm_scale", "s2_lin",
                  "gbar_P", "gbar_E", "coef_P", "coef_E", "z", "shell_volume", "catalogue_haloes_per_engine_bin",
                  "catalogue_haloes_per_engine_bin_z_le_1"):
            setattr(self, k, z[k])
        self.fb = {k[3:]: z[k] for k in z.files if k.startswith("fb_")}
        self.Jfb2h = {k: z[k] for k in z.files if k.startswith("Jfb_")}
        self.pv = self.pk / self.volume[:, None]                  # P_lin / V_z  [z, ell]
        self.dmz = np.where(self.use_dm, self.dm_scale, 0.0)      # DM scale, 0 for z > 1
        self.shell_volume = self.shell_volume[:self.ns]
        self.timing = FC.json.loads(str(z["timing_json"]))

    # ------------------------------------------------------------------ helpers
    def pad(self, F):
        F = np.asarray(F, float)
        if F.shape[-1] == self.ns:
            return F
        width = [(0, 0)] * (F.ndim - 1) + [(0, self.ns - F.shape[-1])]
        return np.pad(F, width)

    def pad2(self, S):
        S = np.asarray(S, float)
        if S.shape[-1] == self.ns:
            return S
        k = self.ns - S.shape[-1]
        return np.pad(S, [(0, 0)] * (S.ndim - 2) + [(0, k), (0, k)])

    def windows(self, P, E):
        WP = self.W_fb_y + np.einsum("cs,zcsl->zl", P, self.W_y, optimize=True)
        WE = self.W_fb_tau + np.einsum("cs,zcsl->zl", E, self.W_tau, optimize=True)
        return WP, WE

    # ------------------------------------------------------------------ observables
    def observables(self, P, E, SPP=None, SEE=None, SPE=None, cal_only=False) -> dict:
        """All moment-type observables; cal_only drops the fallback (published) haloes everywhere."""
        P, E = self.pad(P), self.pad(E)
        SPP = np.einsum("cs,ct->cst", P, P) if SPP is None else self.pad2(SPP)
        SEE = np.einsum("cs,ct->cst", E, E) if SEE is None else self.pad2(SEE)
        SPE = np.einsum("cs,ct->cst", P, E) if SPE is None else self.pad2(SPE)
        f = 0.0 if cal_only else 1.0
        o = {}
        o["yy1h"] = np.einsum("cstl,cst->l", self.K_yy1h, SPP, optimize=True) + f * self.fb["yy1h"]
        o["ksz1h"] = np.einsum("cstl,cst->l", self.K_ksz1h, SEE, optimize=True) + f * self.fb["ksz1h"]
        o["dmcl1h"] = np.einsum("cstl,cst->l", self.K_dm1h, SEE, optimize=True) + f * self.fb["dm1h"]
        o["ydm1h"] = np.einsum("cstl,cst->l", self.K_ydm1h, SPE, optimize=True) + f * self.fb["ydm1h"]
        WP = f * self.W_fb_y + np.einsum("cs,zcsl->zl", P, self.W_y, optimize=True)
        WE = f * self.W_fb_tau + np.einsum("cs,zcsl->zl", E, self.W_tau, optimize=True)
        o["yy2h"] = np.sum(self.pv * WP * WP, axis=0)
        o["ksz2h"] = np.sum(self.s2_lin[:, None] * self.pv * WE * WE, axis=0)
        o["dmcl2h"] = np.sum(self.pv * (self.dmz[:, None] * WE)**2, axis=0)
        o["ydm2h"] = np.sum(self.pv * WP * self.dmz[:, None] * WE, axis=0)
        o["yytot"] = o["yy1h"] + o["yy2h"]
        o["ksztot"] = o["ksz1h"] + o["ksz2h"]
        o["dmcltot"] = o["dmcl1h"] + o["dmcl2h"]
        o["ydmtot"] = o["ydm1h"] + o["ydm2h"]
        o["ymean"] = float(np.sum(P * self.mean_y) + f * self.fb["mean_y"])
        o["dm_mean"] = float(np.sum(E * self.dm_mean) + f * self.fb["dm_mean"])
        o["dm_var"] = float(np.einsum("cst,cst->", self.K_dmvar, SEE) + f * self.fb["dm_var"])
        return o

    def scalars(self, o: dict) -> dict:
        """Flatten spectra at the selected multipoles: e.g. 'yytot_1000'."""
        out = {}
        for k, v in o.items():
            if np.ndim(v) == 0:
                out[k] = float(v)
            else:
                for e, i in zip(FC.SEL_ELL, FC.SEL_INDEX):
                    out[f"{k}_{e}"] = float(v[i])
        return out

    # ------------------------------------------------------------------ jacobians
    def jacobians(self, P, E, SPP=None, SEE=None, SPE=None) -> dict:
        """dO/dln F[c, s, (ell)] for the calibrated bins: every halo's profile of bin c scaled by (1 + eps)
        in shell s.  Second moments given -> scatter-inclusive (the truth); None -> deterministic."""
        P, E = self.pad(P), self.pad(E)
        SPP = np.einsum("cs,ct->cst", P, P) if SPP is None else self.pad2(SPP)
        SEE = np.einsum("cs,ct->cst", E, E) if SEE is None else self.pad2(SEE)
        SPE = np.einsum("cs,ct->cst", P, E) if SPE is None else self.pad2(SPE)
        WP, WE = self.windows(P, E)
        J = {}
        J["yy1h"] = 2 * np.einsum("cstl,cst->csl", self.K_yy1h, SPP, optimize=True)
        J["ksz1h"] = 2 * np.einsum("cstl,cst->csl", self.K_ksz1h, SEE, optimize=True)
        J["dmcl1h"] = 2 * np.einsum("cstl,cst->csl", self.K_dm1h, SEE, optimize=True)
        J["ydm1h_P"] = np.einsum("cstl,cst->csl", self.K_ydm1h, SPE, optimize=True)
        J["ydm1h_E"] = np.einsum("cstl,cst->ctl", self.K_ydm1h, SPE, optimize=True)
        gP = np.einsum("cs,zcsl->zcsl", P, self.W_y, optimize=True)
        gE = np.einsum("cs,zcsl->zcsl", E, self.W_tau, optimize=True)
        J["yy2h"] = 2 * np.einsum("zl,zcsl->csl", self.pv * WP, gP, optimize=True)
        J["ksz2h"] = 2 * np.einsum("zl,zcsl->csl", self.s2_lin[:, None] * self.pv * WE, gE, optimize=True)
        J["dmcl2h"] = 2 * np.einsum("zl,zcsl->csl", self.pv * self.dmz[:, None]**2 * WE, gE, optimize=True)
        J["ydm2h_P"] = np.einsum("zl,zcsl->csl", self.pv * self.dmz[:, None] * WE, gP, optimize=True)
        J["ydm2h_E"] = np.einsum("zl,zcsl->csl", self.pv * self.dmz[:, None] * WP, gE, optimize=True)
        J["yytot"] = J["yy1h"] + J["yy2h"]
        J["ksztot"] = J["ksz1h"] + J["ksz2h"]
        J["dmcltot"] = J["dmcl1h"] + J["dmcl2h"]
        J["ydmtot_P"] = J["ydm1h_P"] + J["ydm2h_P"]
        J["ydmtot_E"] = J["ydm1h_E"] + J["ydm2h_E"]
        J["ymean"] = P * self.mean_y
        J["dm_mean"] = E * self.dm_mean
        J["dm_var"] = 2 * np.einsum("cst,cst->cs", self.K_dmvar, SEE)
        return J

    # ------------------------------------------------------------------ z = 0 budgets
    def budgets(self, P, E, radius: float = 4.0) -> dict:
        """Per-bin Y(<radius) and N_e(<radius): sum_s F_s V_s over shells inside radius (dimensionless)."""
        inside = FC.BASIS_EDGES[1:self.ns + 1] <= radius + 1e-12
        v = self.shell_volume * inside
        return {"Y": self.pad(P) @ v, "Ne": self.pad(E) @ v}


def relative_errors(model: dict, truth: dict) -> dict:
    """(model - truth) / truth for every key present in both (scalars)."""
    out = {}
    for k, t in truth.items():
        if k in model and t is not None and np.isfinite(t) and t != 0:
            out[k] = float(model[k] / t - 1.0)
    return out
