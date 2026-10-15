"""Profile models of Test G, per field.  x = r / R200c; E in n_e mu_XGP/(rho_c f_b), P in P_e/P200.

Cores (Test F's forms, identical expressions to radii234 models.shape):
  B12 (pressure)  P_g(x) = 0.5176 P0 q^-0.3 (1 + q)^-beta,                 q = x / xc
  B16 (electrons) E_g(x) = rho0 q^-0.2 (1 + q^alpha)^-((beta - 0.2)/alpha), q = x / 0.5
Transition (Diemer & Kravtsov 2014):  f_tr(x) = [1 + (x/x_t)^4]^(-gamma_t/4)

Electrons
  E0    E_g                                                     3 per bin
  E1    E_g + A (x/2)^-s                                        5 per bin                   (Test F)
  E2    E_g f_tr + Ebar [1 + b xi_lin(x R)]                     4 per bin; gamma_t = 6 (E2g6) or global (E2gf)
  E3    E2 + Ebar A_inf (b/b_piv)^a (x/2)^-s_inf                4 per bin + global gamma_t, A_inf, a, s_inf
  E3pb  E2 + Ebar A_c (x/2)^-s_c  (per-bin infall; universality diagnostic)   6 per bin + gamma_t
Pressure
  P0    P_g                                                     3 per bin
  P1    P_g + A (x/2)^-s                                        5 per bin                   (Test F)
  P2    P_g f_tr + KAPPA_P/200 n_out(x) tau(x),  n_out = Ebar [1 + b xi_lin],  tau = tau_0 (x/2)^-q
        4 per bin + global tau_0, q; gamma_t = 6 (P2g6) or global (P2gf)
  P2i   P2gf with n_out including E3's infall term (E3's global A_inf, a, s_inf fixed)
  P2s   P2gf with an accretion-shock cutoff of tau: x [1 + (x/x_sh)^8]^-1, global x_sh
  P2m   P2gf with tau_0 (M/M_piv)^-p_M (mass-dependent outer temperature), global p_M
"bound" = core (x f_tr), "outer" = everything else; S2 scatters the two with separate amplitudes.
"""
from __future__ import annotations

import g_common as GC
import numpy as np

FAM = GC.FF.FAM
X_DAMP = 0.1          # R200c: the infall and outer-temperature terms of E3/E3pb/P2* are damped inside, D(x) = x^4/(x^4 + X_DAMP^4)


def damping(X):
    X4 = X**4
    return X4 / (X4 + X_DAMP**4)


LO_CORE = {"P": np.asarray(FAM.M.BOUNDS["b12"][0], float), "E": np.asarray(FAM.M.BOUNDS["b16"][0], float)}
HI_CORE = {"P": np.asarray(FAM.M.BOUNDS["b12"][1], float), "E": np.asarray(FAM.M.BOUNDS["b16"][1], float)}
CORE_NAMES = {"P": ("P0", "xc", "beta"), "E": ("rho0", "alpha", "beta")}
CORE_FAMILY = {"P": "b12", "E": "b16"}


class Par:
    def __init__(self, name, lo, hi, log, start):
        self.name, self.lo, self.hi, self.log, self.start = name, float(lo), float(hi), bool(log), float(start)

    def theta_bounds(self):
        return (np.log(self.lo), np.log(self.hi)) if self.log else (self.lo, self.hi)

    def to_theta(self, v):
        return np.log(v) if self.log else v

    def to_value(self, t):
        return np.exp(t) if self.log else t


class Model:
    """A field model: per-bin parameters (core [+ x_t] [+ extras]) and global parameters."""

    def __init__(self, name, field, outer, gamma=None, tau=None, shock=False, label=""):
        self.name, self.field, self.outer, self.gamma, self.tau, self.shock = name, field, outer, gamma, tau, shock
        self.label = label
        core = [Par(n, LO_CORE[field][k], HI_CORE[field][k], True, 1.0) for k, n in enumerate(CORE_NAMES[field])]
        self.per_bin = list(core)
        self.globals = []
        self.physical = outer in ("2h", "2h+inf", "2h+infpb", "2htau")
        if self.physical:
            self.per_bin.append(Par("x_t", 0.5, 20.0, True, 2.0 if field == "E" else 2.5))
            if gamma == "free":
                self.globals.append(Par("gamma_t", 0.5, 30.0, True, 6.0))
        if outer == "pl":
            lo_a, hi_a = (1e-6, 1e5) if field == "E" else (1e-8, 1e2)
            self.per_bin += [Par("A_env", lo_a, hi_a, True, 1e-3), Par("s_env", -2.0, 10.0, False, 1.0)]
        if outer == "2h+inf":
            self.globals += [Par("A_inf", 1e-4, 1e3, True, 1.0), Par("a_inf", -8.0, 8.0, False, 0.0),
                             Par("s_inf", -1.0, 3.9, False, 1.5)]
        if outer == "2h+infpb":
            self.per_bin += [Par("A_c", 1e-4, 1e3, True, 1.0), Par("s_c", -1.0, 3.9, False, 1.5)]
        if outer == "2htau":
            self.globals += [Par("tau_0", 1e-4, 30.0, True, 0.3), Par("q_tau", -4.0, 3.9, False, 1.0)]
            if shock:
                self.globals.append(Par("x_sh", 1.2, 20.0, True, 4.0))
            if tau == "mass":
                self.globals.append(Par("p_M", -3.0, 3.0, False, 0.3))
        self.kb = len(self.per_bin)
        self.kg = len(self.globals)
        self.uses_infall = False                 # P2i: n_out includes E3's infall term (fixed globals in ctx)

    # ------------------------------------------------------------------ parameter handling
    def names(self):
        return [p.name for p in self.per_bin], [p.name for p in self.globals]

    def n_free(self, n_bins):
        return self.kb * n_bins + self.kg

    def gamma_value(self, pg):
        if not self.physical:
            return None
        if self.gamma == "free":
            return float(pg[[p.name for p in self.globals].index("gamma_t")])
        return 6.0

    def gval(self, pg, name, default=None):
        names = [p.name for p in self.globals]
        return float(pg[names.index(name)]) if name in names else default

    # ------------------------------------------------------------------ evaluation at nodes
    def nodes(self, pb, pg, ctx, x, T2h):
        """(bound, outer) at the nodes x [S, n] for every bin; pb [nb, kb] natural units, pg [kg];
        T2h [nb, S, n] the 2-halo template <b xi_lin>; ctx: ebar, kappa_P, b [nb], logm [nb], b_pivot,
        infall (A_inf, a, s_inf) for P2i."""
        pb = np.asarray(pb, float)
        nb = pb.shape[0]
        X = x[None, :, :]
        c0, c1, c2 = (pb[:, k][:, None, None] for k in range(3))
        if self.field == "P":
            q = X / c1
            core = .5176 * c0 * q**(-.3) * (1 + q)**(-c2)
        else:
            q = X / .5
            core = c0 * q**(-.2) * (1 + q**c1)**(-(c2 - .2) / c1)
        outer = np.zeros((nb,) + x.shape)
        if self.outer == "none":
            return core, outer
        D = damping(X)
        if self.outer == "pl":                      # Test F's form, painted exactly as fitted (s_env <= 3)
            A, s = pb[:, 3][:, None, None], pb[:, 4][:, None, None]
            return core, A * (X / 2.0)**(-s)
        xt = pb[:, 3][:, None, None]
        g = self.gamma_value(pg)
        bound = core * (1 + (X / xt)**4)**(-g / 4)
        ebar = ctx["ebar"]
        n2h = ebar * (1 + T2h)
        if self.outer == "2h":
            return bound, n2h
        bb = (np.asarray(ctx["b"], float) / ctx["b_pivot"])[:, None, None]
        if self.outer == "2h+inf":
            A, a, s = self.gval(pg, "A_inf"), self.gval(pg, "a_inf"), self.gval(pg, "s_inf")
            return bound, n2h + ebar * A * bb**a * (X / 2.0)**(-s) * D
        if self.outer == "2h+infpb":
            A, s = pb[:, 4][:, None, None], pb[:, 5][:, None, None]
            return bound, n2h + ebar * A * (X / 2.0)**(-s) * D
        # pressure: KAPPA_P / 200 n_out tau D
        nout = n2h
        if self.uses_infall:
            A, a, s = ctx["infall"]
            nout = n2h + ebar * A * bb**a * (X / 2.0)**(-s) * D
        tau = self.gval(pg, "tau_0") * (X / 2.0)**(-self.gval(pg, "q_tau"))
        if self.tau == "mass":
            v = (np.asarray(ctx["logm"], float) - np.log10(GC.M_PIVOT))[:, None, None]
            tau = tau * 10**(-self.gval(pg, "p_M") * v)
        if self.shock:
            tau = tau / (1 + (X / self.gval(pg, "x_sh"))**8)
        return bound, ctx["kappa_P"] / 200.0 * nout * tau * D

    # ------------------------------------------------------------------ derivatives (theta space)
    def nodes_with_derivatives(self, pb, pg, ctx, x, T2h):
        """Total F = bound + outer at the nodes [nb, S, n] and its derivatives with respect to theta (ln p for
        positive parameters, p for signed ones): dB [nb, kb, S, n] (each bin's own parameters) and
        dG [nb, kg, S, n] (global parameters)."""
        pb = np.asarray(pb, float)
        nb = pb.shape[0]
        X = x[None, :, :]
        c0, c1, c2 = (pb[:, k][:, None, None] for k in range(3))
        dB = np.zeros((nb, self.kb) + x.shape)
        dG = np.zeros((nb, self.kg) + x.shape)
        if self.field == "P":
            q = X / c1
            core = .5176 * c0 * q**(-.3) * (1 + q)**(-c2)
            dcore = [np.ones_like(q), .3 + c2 * q / (1 + q), -c2 * np.log1p(q)]
        else:
            q = X / .5
            qa = q**c1
            l1 = np.log1p(qa)
            core = c0 * q**(-.2) * (1 + qa)**(-(c2 - .2) / c1)
            dcore = [np.ones_like(q) * np.ones_like(c1), (c2 - .2) / c1 * l1 - (c2 - .2) * qa / (1 + qa) * np.log(q),
                     -c2 / c1 * l1]
        gnames = [p.name for p in self.globals]
        ln_x2 = np.log(X / 2.0)
        D = damping(X)
        if self.outer in ("none", "pl"):
            for k in range(3):
                dB[:, k] = core * dcore[k]
            if self.outer == "none":
                return core, dB, dG
            A, s = pb[:, 3][:, None, None], pb[:, 4][:, None, None]
            O = A * (X / 2.0)**(-s)
            dB[:, 3] = O
            dB[:, 4] = -O * ln_x2
            return core + O, dB, dG
        xt = pb[:, 3][:, None, None]
        g = self.gamma_value(pg)
        u = (X / xt)**4
        ftr = (1 + u)**(-g / 4)
        bound = core * ftr
        for k in range(3):
            dB[:, k] = bound * dcore[k]
        dB[:, 3] = bound * g * u / (1 + u)
        if self.gamma == "free":
            dG[:, gnames.index("gamma_t")] = bound * (-(g / 4) * np.log1p(u))
        ebar = ctx["ebar"]
        n2h = ebar * (1 + T2h)
        if self.outer == "2h":
            return bound + n2h, dB, dG
        bb = (np.asarray(ctx["b"], float) / ctx["b_pivot"])[:, None, None]
        if self.outer == "2h+inf":
            A, a, s = self.gval(pg, "A_inf"), self.gval(pg, "a_inf"), self.gval(pg, "s_inf")
            I = ebar * A * bb**a * (X / 2.0)**(-s) * D
            dG[:, gnames.index("A_inf")] = I
            dG[:, gnames.index("a_inf")] = I * np.log(bb) * np.ones_like(I)
            dG[:, gnames.index("s_inf")] = -I * ln_x2
            return bound + n2h + I, dB, dG
        if self.outer == "2h+infpb":
            A, s = pb[:, 4][:, None, None], pb[:, 5][:, None, None]
            I = ebar * A * (X / 2.0)**(-s) * D
            dB[:, 4] = I
            dB[:, 5] = -I * ln_x2
            return bound + n2h + I, dB, dG
        nout = n2h
        if self.uses_infall:
            A, a, s = ctx["infall"]
            nout = n2h + ebar * A * bb**a * (X / 2.0)**(-s) * D
        tau = self.gval(pg, "tau_0") * (X / 2.0)**(-self.gval(pg, "q_tau"))
        if self.tau == "mass":
            v = (np.asarray(ctx["logm"], float) - np.log10(GC.M_PIVOT))[:, None, None]
            tau = tau * 10**(-self.gval(pg, "p_M") * v)
        if self.shock:
            w = (X / self.gval(pg, "x_sh"))**8
            tau = tau / (1 + w)
        O = ctx["kappa_P"] / 200.0 * nout * tau * D
        dG[:, gnames.index("tau_0")] = O
        dG[:, gnames.index("q_tau")] = -O * ln_x2
        if self.tau == "mass":
            dG[:, gnames.index("p_M")] = -O * v * np.log(10.0)
        if self.shock:
            dG[:, gnames.index("x_sh")] = O * 8 * w / (1 + w)
        return bound + O, dB, dG


MODELS = {
    "E0": Model("E0", "E", "none", label="B16 gNFW only (Lee22-like)"),
    "E1": Model("E1", "E", "pl", label="B16 + A (x/2)^-s (Test F)"),
    "E2g6": Model("E2g6", "E", "2h", gamma="fixed", label="B16 f_tr + linear 2-halo, gamma_t = 6"),
    "E2gf": Model("E2gf", "E", "2h", gamma="free", label="B16 f_tr + linear 2-halo, gamma_t global"),
    "E3": Model("E3", "E", "2h+inf", gamma="free", label="E2gf + infall Ebar A_inf (b/b_piv)^a (x/2)^-s_inf"),
    "P0": Model("P0", "P", "none", label="B12 gNFW only (Lee22-like)"),
    "P1": Model("P1", "P", "pl", label="B12 + A (x/2)^-s (Test F)"),
    "P2g6": Model("P2g6", "P", "2htau", gamma="fixed", tau="const",
                  label="B12 f_tr + KAPPA_P n_2h tau_0 (x/2)^-q / 200, gamma_t = 6"),
    "P2gf": Model("P2gf", "P", "2htau", gamma="free", tau="const",
                  label="B12 f_tr + KAPPA_P n_2h tau_0 (x/2)^-q / 200, gamma_t global"),
    "P2i": Model("P2i", "P", "2htau", gamma="free", tau="const",
                 label="P2gf with n_out = 2-halo + E3 infall"),
    "P2s": Model("P2s", "P", "2htau", gamma="free", tau="const", shock=True,
                 label="P2gf with an accretion-shock cutoff of tau at x_sh"),
    "P2m": Model("P2m", "P", "2htau", gamma="free", tau="mass",
                 label="P2gf with tau_0 (M/10^14.5)^-p_M"),
}
MODELS["P2i"].uses_infall = True
DIAGNOSTIC_MODELS = {"E3pb": Model("E3pb", "E", "2h+infpb", gamma="free",
                                   label="E2gf + per-bin infall Ebar A_c (x/2)^-s_c (universality test)")}
FIELD_MODELS = {"E": ("E0", "E1", "E2g6", "E2gf", "E3"), "P": ("P0", "P1", "P2g6", "P2gf", "P2i", "P2s", "P2m")}
PHYSICAL = {"E2g6", "E2gf", "E3", "P2g6", "P2gf", "P2i", "P2s", "P2m"}


def get(name):
    return MODELS[name] if name in MODELS else DIAGNOSTIC_MODELS[name]


def shells(model, pb, pg, ctx, x, wq, T2h, cut=None, edges=None, factor=None):
    """Shell averages (bound, outer) [nb, S]; ``cut`` (an edge of ``edges``) zeroes the shells beyond it;
    ``factor`` [nb, S, n] multiplies both components at the nodes (p_keep)."""
    b, o = model.nodes(pb, pg, ctx, x, T2h)
    if factor is not None:
        b, o = b * factor, o * factor
    B = np.sum(b * wq[None], axis=-1)
    O = np.sum(o * wq[None], axis=-1)
    if cut is not None:
        keep = GC.x_hi_le(edges, cut)
        B, O = B * keep[None], O * keep[None]
    return B, O


def transition_slope_radius(model, pb, pg, ctx, lh_fn=None, x=None):
    """Radius of the steepest logarithmic slope of the total model per bin (fine log grid 0.5-6 R200c), using
    the bin-mean 2-halo template interpolated in ln x; returns [nb] radii and slopes."""
    x = np.geomspace(0.5, 6.0, 400) if x is None else x
    T = ctx["T2h_fine"](x)                                  # [nb, len(x)]
    b, o = model.nodes(pb, pg, ctx, x[None, :], T[:, None, :])
    tot = (b + o)[:, 0, :]
    slope = np.gradient(np.log(tot), np.log(x), axis=1)
    k = np.argmin(slope, axis=1)
    return x[k], slope[np.arange(len(k)), k]
