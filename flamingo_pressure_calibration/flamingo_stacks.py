"""Stacks, Battaglia-form fits and the nine-coefficient joint fit for the stack_products.py outputs.

Conventions (identical to XGPaint / Battaglia et al. 2012):
    x = r / R200c,  P200 = G M200c (200 rho_crit) f_b / (2 R200c)
    Pe / P200 = 0.5176 * P0 * (x/xc)^gamma * (1 + (x/xc)^alpha)^(-beta),  alpha = 1, gamma = -0.3
    A = A0 (M200c / 1e14 Msun)^alpha_m (1+z)^alpha_z   for A in (P0, xc, beta)   [nine coefficients]
Masses are physical M200c in Msun. theta stores ln P0, ln xc, ln beta and the six exponents.

A stack is the arithmetic mean over its haloes of the per-halo shell-mean Pe/P200 (each halo
weighted equally; a shell enters for a halo only if it lies inside that halo's R100c). The
model of a stack is the same mean of the exactly shell-averaged model over the same haloes,
so the mass spread inside a 0.2-dex bin is treated exactly (through 0.02-dex mass nodes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import c as C_LIGHT, electron_mass
from scipy.optimize import least_squares
from scipy.special import expit

import flamingo_reduced_io as fio

ELECTRON_FRACTION = 0.5176
B12 = np.array([18.1, 0.497, 4.35, 0.154, -0.00865, 0.0393, -0.758, 0.731, 0.415])
NAMES = ["P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta", "alpha_z_P0", "alpha_z_xc", "alpha_z_beta"]
Y_PER_PR = fio.SIGMA_T / (electron_mass * C_LIGHT**2) * fio.MPC_M     # y200 = Y_PER_PR * P200[Pa] * R200c[Mpc]
COMOVING_SOFTENING_MPC, MAX_PHYSICAL_SOFTENING_MPC = 0.0223, 0.0057
LOWER = np.r_[np.log([1e-3, 1e-3, 0.3]), np.full(6, -5.0)]
UPPER = np.r_[np.log([1e4, 30.0, 30.0]), np.full(6, 5.0)]


# --------------------------------------------------------------------------- parameters
def encode(native):
    theta = np.array(native, dtype=float)
    theta[:3] = np.log(theta[:3])
    return theta


def decode(theta):
    native = np.array(theta, dtype=float)
    native[:3] = np.exp(native[:3])
    return native


def coefficients(native, mass, z):
    """(P0, xc, beta) of a nine-coefficient vector at M200c [Msun] and z."""
    native = np.asarray(native, float)
    lm, lz = np.log(np.asarray(mass, float) / 1e14), np.log1p(np.asarray(z, float))
    return tuple(native[i] * np.exp(native[3 + i] * lm + native[6 + i] * lz) for i in range(3))


def gnfw(x, P0, xc, beta, alpha=1.0, gamma=-0.3):
    q = np.asarray(x, float) / xc
    return ELECTRON_FRACTION * P0 * np.exp(gamma * np.log(q) - beta * np.logaddexp(0.0, alpha * np.log(q)))


def shell_nodes(inner, outer, order=16):
    inner, outer = np.broadcast_arrays(np.asarray(inner, float), np.asarray(outer, float))
    nodes, weights = leggauss(order)
    radius = inner[..., None] + (outer - inner)[..., None] * (nodes + 1) / 2
    volume_weights = 3 * radius**2 * weights * (outer - inner)[..., None] / 2 / (outer**3 - inner**3)[..., None]
    return radius, volume_weights


def shell_mean(func, inner, outer, order=24):
    radius, weights = shell_nodes(inner, outer, order)
    return np.sum(weights * func(radius), axis=-1)


def projected_annuli(func, edges, x_trunc, order=24):
    """Area-mean of the line-of-sight integral (in R200c units) of func through a sphere of radius x_trunc.

    Multiply by y200 = Y_PER_PR * P200 * R200c to get Compton y; for Pe/P200 profiles this is y / y200.
    """
    nodes, weights = leggauss(order)
    t = (nodes + 1) / 2
    r2 = edges[:-1, None] ** 2 + np.diff(edges**2)[:, None] * t                       # uniform in R^2 = area mean
    length = np.sqrt(np.maximum(x_trunc**2 - r2, 0.0))
    radius = np.sqrt(r2[..., None] + (length[..., None] * t) ** 2)
    los = 2 * length * np.sum(func(radius) * weights / 2, axis=-1)
    return np.sum(los * weights / 2, axis=-1)


def integrated(func, x_max, order=64):
    """4 pi int_0^x_max func(x) x^2 dx: Y_sph / (sigma_T/(m_e c^2) P200 R200c^3)."""
    nodes, weights = leggauss(order)
    lnx = np.log(1e-4) + (np.log(x_max) - np.log(1e-4)) * (nodes + 1) / 2
    x = np.exp(lnx)
    return 4 * np.pi * np.sum(weights * (np.log(x_max) - np.log(1e-4)) / 2 * func(x) * x**3)


# --------------------------------------------------------------------------- products and stacks
def load_products(path):
    data = dict(np.load(path))
    data["meta"] = json.loads(Path(path).with_suffix(".json").read_text())
    return data


def per_halo(d, use_agn_filter=True):
    """Per-halo shell ratios Pe/P200, projected y/y200 and maps y/y200 (all in R200c units).

    The shell (annulus) that straddles a halo's R100c is kept with its exact truncated volume
    (area) [x_in, x100]: every particle inside R100c is in the reduced snapshot, so the mean
    over the covered part is unbiased. Shells with less than 2% radial coverage are dropped.
    """
    edges = d["edges"]
    r200 = d["cat_r_200_crit_mpc"]
    p200 = d["cat_p200_pa"]
    x100 = d["cat_r_100_crit_mpc"] / r200
    # data exist inside this sphere: R100c for reduced snapshots, a larger sphere for full-snapshot products
    x_sphere = d["cat_r_sphere_mpc"] / r200 if "cat_r_sphere_mpc" in d else x100
    y3 = d["y3_noagn"] if use_agn_filter else d["y3_raw"]
    yp = d["yp_noagn"] if use_agn_filter else d["yp_raw"]
    maps = d["map_noagn"] if use_agn_filter else d["map_raw"]
    x_in = edges[None, :-1]
    x_out = np.minimum(edges[None, 1:], x_sphere[:, None])              # effective outer edge per halo
    usable = x_out > 1.02 * np.maximum(x_in, 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = fio.shell_pressure_from_y(y3, x_in * r200[:, None], x_out * r200[:, None])
        area = np.pi * (x_out**2 - x_in**2) * r200[:, None] ** 2
        y200 = Y_PER_PR * p200 * r200
        ratio = np.where(usable, pe / p200[:, None], np.nan)
        yhat = np.where(usable, yp / 3.0 / area / y200[:, None], np.nan)
    return {"ratio": ratio, "complete": usable, "x_out": np.where(usable, x_out, np.nan),
            "partial": usable & (edges[None, 1:] > x_sphere[:, None]),
            "yhat": yhat, "map_hat": maps / 3.0 / y200[:, None, None],
            "map_y": maps / 3.0, "x100": x100, "x_sphere": x_sphere, "y200": y200, "r200": r200, "p200": p200}


def _soap_shells(d, rows, use_agn_filter):
    """SOAP aperture shells starting at or beyond R100c (R100c-R50c, R50c-5R500c), per halo, x units."""
    aps = fio.APERTURES
    key = "cat_y_noagn_{}_mpc2" if use_agn_filter else "cat_y_{}_mpc2"
    radii = np.stack([d[f"cat_r_{ap}_mpc"][rows] for ap in aps], axis=1)
    ys = np.stack([d[key.format(ap)][rows] for ap in aps], axis=1)
    ngas = np.stack([d[f"cat_ngas_{ap}"][rows] for ap in aps], axis=1).astype(float)
    order = np.argsort(radii, axis=1)
    radii, ys, ngas = (np.take_along_axis(v, order, axis=1) for v in (radii, ys, ngas))
    inner = np.column_stack([np.zeros(len(rows)), radii[:, :-1]])
    dy = np.diff(np.column_stack([np.zeros(len(rows)), ys]), axis=1)
    dn = np.diff(np.column_stack([np.zeros(len(rows)), ngas]), axis=1)
    r200 = d["cat_r_200_crit_mpc"][rows]
    r100 = d["cat_r_100_crit_mpc"][rows]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = fio.shell_pressure_from_y(dy, inner, radii) / d["cat_p200_pa"][rows][:, None]
    outer_mask = inner >= 0.999 * r100[:, None]
    # every halo has exactly the same two outer apertures (50_crit and 5xR_500_crit) beyond R100c
    take = np.argsort(~outer_mask, axis=1, kind="stable")[:, :2]
    pick = lambda v: np.take_along_axis(v, take, axis=1)                  # noqa: E731
    ok = pick(outer_mask) & (pick(dy) > 0) & (pick(dn) >= 32)
    return {"inner": pick(inner) / r200[:, None], "outer": pick(radii) / r200[:, None], "ratio": pick(ratio), "ok": ok}


def build_stacks(products, use_agn_filter=True, hsml_factor=2.0, show_hsml_factor=1.0, min_particles=32,
                 min_haloes=8, n_boot=200, seed=1, x_min=0.0):
    """One stack per (snapshot, 0.2-dex mass bin) with at least min_haloes haloes.

    Fitted shells: outward of the outermost shell closer than hsml_factor pooled median smoothing lengths,
    beyond 4 softenings and x >= x_min. 'shown' uses show_hsml_factor and ignores x_min (drawn open).
    """
    stacks = []
    for d in products:
        z, a = float(d["redshift"]), float(d["scale_factor"])
        edges = d["edges"]
        nb = len(edges) - 1
        ph = per_halo(d, use_agn_filter)
        softening = min(COMOVING_SOFTENING_MPC * a, MAX_PHYSICAL_SOFTENING_MPC)
        h_centres = 0.5 * (d["hsml_log_edges"][:-1] + d["hsml_log_edges"][1:])
        for b in np.unique(d["bin"]):
            rows = np.flatnonzero(d["bin"] == b)
            if len(rows) < min_haloes:
                continue
            n = len(rows)
            ratio, complete = ph["ratio"][rows], ph["complete"][rows]
            weight = complete.astype(float)
            n_complete = weight.sum(0)

            def mean_of(values, w):
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(w.sum(0) > 0, (w * np.nan_to_num(values)).sum(0) / w.sum(0), np.nan)

            mean = mean_of(ratio, weight)
            pooled = (d["n3"][rows] * complete).sum(0)
            hist = d["hsml_hist"][rows].sum(0).astype(float)
            cumulative = np.cumsum(hist, axis=1)
            med_index = np.argmax(cumulative >= cumulative[:, -1:] / 2, axis=1)
            median_h = np.where(cumulative[:, -1] > 0, 10 ** h_centres[med_index], np.nan)
            r200_med = float(np.median(ph["r200"][rows]))
            with np.errstate(invalid="ignore"):
                base = ((n_complete >= max(min(3, n), 0.5 * n)) & (pooled >= min_particles) & (mean > 0)
                        & (edges[:-1] * r200_med >= 4 * softening))

                def outward_of_last_failure(passes):
                    # resolution is monotonic in radius: keep only shells beyond the outermost failing one
                    fails = np.flatnonzero(base & ~passes)
                    return base & (np.arange(nb) > (fails.max() if len(fails) else -1))

                fit = outward_of_last_failure(edges[:-1] >= hsml_factor * median_h) & (edges[:-1] >= x_min)
                shown = outward_of_last_failure(edges[:-1] >= show_hsml_factor * median_h) | fit
            # bootstrap over haloes, seeded per (snapshot, bin) so that adding other stacks never changes these draws
            stack_rng = np.random.default_rng([seed, int(d["snapshot"]), int(b), int(1000 * float(d["bin_edges"][0]))])
            counts = stack_rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
            boot = np.array([mean_of(ratio, weight * c[:, None]) for c in counts])
            err = np.nanstd(boot, axis=0)
            # projected y / y200 (sum over the three axes already inside each halo's value)
            yhat, pcomplete = ph["yhat"][rows], ph["complete"][rows]
            pmean = mean_of(yhat, pcomplete.astype(float))
            pboot = np.array([mean_of(yhat, pcomplete * c[:, None]) for c in counts])
            pvalid = ((pcomplete.sum(0) >= max(min(3, n), 0.5 * n)) & ((d["np"][rows] * pcomplete).sum(0) >= min_particles)
                      & (pmean > 0) & (edges[:-1] * r200_med >= 4 * softening))
            soap = _soap_shells(d, rows, use_agn_filter)
            soap_mean = np.array([np.nanmean(np.where(soap["ok"][:, j], soap["ratio"][:, j], np.nan)) for j in range(2)])
            soap_boot = np.array([[np.nansum(c * np.where(soap["ok"][:, j], soap["ratio"][:, j], 0)) /
                                   max(np.sum(c * soap["ok"][:, j]), 1) for j in range(2)] for c in counts])
            masses = d["cat_m200c_msun"][rows]
            # integrated spherical Compton Y from SOAP (full snapshot), normalised like the model integral
            ykey = "cat_y_noagn_{}_mpc2" if use_agn_filter else "cat_y_{}_mpc2"
            norm = (fio.SIGMA_T / (electron_mass * C_LIGHT**2) * ph["p200"][rows]
                    * (ph["r200"][rows] * fio.MPC_M) ** 3 / fio.MPC_M**2)               # Mpc^2
            yhat_200 = d[ykey.format("200_crit")][rows] / norm
            yhat_5r500 = d[ykey.format("5xR_500_crit")][rows] / norm
            x_5r500 = 5 * d["cat_r_500_crit_mpc"][rows] / ph["r200"][rows]
            x_out_eff = np.where(complete, ph["x_out"][rows], np.nan)
            partial = ph["partial"][rows].any(0)
            outer_mean = np.where(partial, np.nanmean(x_out_eff, axis=0), edges[1:])
            stacks.append({
                "snapshot": int(d["snapshot"]), "z": z, "bin": int(b),
                "log10_m_lo": float(d["bin_edges"][b]), "log10_m_hi": float(d["bin_edges"][b + 1]),
                "n_haloes": n, "masses": masses, "log10_m_median": float(np.log10(np.median(masses))),
                "edges": edges, "x_mid": np.sqrt(np.maximum(edges[:-1], 1e-3) * outer_mean),
                "outer_mean": outer_mean, "partial": partial, "halo_x_out": x_out_eff,
                "ratio": mean, "err": err, "boot": boot, "fit": fit, "shown": shown, "valid": base,
                "n_complete": n_complete, "pooled_particles": pooled, "median_h": median_h,
                "p16": np.nanpercentile(np.where(complete, ratio, np.nan), 16, axis=0),
                "p84": np.nanpercentile(np.where(complete, ratio, np.nan), 84, axis=0),
                "complete": complete, "halo_ratio": ratio,
                "yhat": pmean, "yhat_err": np.nanstd(pboot, axis=0), "yhat_valid": pvalid,
                "x_trunc": float(np.median(ph["x_sphere"][rows])), "halo_x_sphere": ph["x_sphere"][rows],
                "x100_median": float(np.median(ph["x100"][rows])),
                "map_hat": ph["map_hat"][rows].mean(0), "map_y": ph["map_y"][rows].mean(0),
                "map_half_extent": float(d["map_half_extent"]),
                "y200_median": float(np.median(ph["y200"][rows])), "r200_median_mpc": r200_med,
                "soap": soap, "soap_mean": soap_mean, "soap_err": np.nanstd(soap_boot, axis=0),
                "soap_inner_mean": np.nanmean(np.where(soap["ok"], soap["inner"], np.nan), axis=0),
                "soap_outer_mean": np.nanmean(np.where(soap["ok"], soap["outer"], np.nan), axis=0),
                "boot_counts": counts, "halo_Yhat_200": yhat_200, "halo_Yhat_5r500": yhat_5r500,
                "halo_x_5r500": x_5r500,
                "yhat_p16": np.nanpercentile(np.where(pcomplete, yhat, np.nan), 16, axis=0),
                "yhat_p84": np.nanpercentile(np.where(pcomplete, yhat, np.nan), 84, axis=0),
            })
    return stacks


# --------------------------------------------------------------------------- objective
def _mass_nodes(masses, complete, width=0.02):
    """Group a stack's haloes into 0.02-dex mass nodes: node log-mean mass and per-shell complete counts."""
    logm = np.log10(masses)
    node = np.floor((logm - logm.min()) / width).astype(int)
    ids = np.unique(node)
    node_mass = np.array([10 ** logm[node == i].mean() for i in ids])
    counts = np.array([complete[node == i].sum(0) for i in ids], dtype=float)       # (n_nodes, n_shells)
    return node_mass, counts, node, ids


def stack_rows(stacks, include_soap_outer=False, sigma_floor=0.03, boot_index=None, equal_stack_weight=True):
    """Rows (inner, outer, mass, z, weight, group) and groups (observed, sigma_log) for the objective.

    boot_index=(b) re-weights every stack's haloes by its b-th bootstrap draw (observed means and model weights).
    equal_stack_weight scales each stack's errors by sqrt(n_shells / mean n_shells), so that every
    (redshift, mass-bin) stack carries the same total weight however many shells it resolves.
    """
    rows = {k: [] for k in ["inner", "outer", "mass", "z", "weight", "group"]}
    groups = {"observed": [], "sigma_log": [], "stack": [], "shell": [], "is_soap": []}
    for s, st in enumerate(stacks):
        c = None if boot_index is None else st["boot_counts"][boot_index].astype(float)
        complete = st["complete"] * (1.0 if c is None else c[:, None])
        node_mass, counts, node, ids = _mass_nodes(st["masses"], complete)
        if c is None:
            observed = st["ratio"]
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                observed = (complete * np.nan_to_num(st["halo_ratio"])).sum(0) / complete.sum(0)
        for k in np.flatnonzero(st["fit"]):
            if not np.isfinite(observed[k]) or observed[k] <= 0 or counts[:, k].sum() <= 0:
                continue
            g = len(groups["observed"])
            groups["observed"].append(observed[k])
            groups["sigma_log"].append(np.hypot(st["err"][k] / st["ratio"][k], sigma_floor))
            groups["stack"].append(s); groups["shell"].append(int(k)); groups["is_soap"].append(False)
            if st["partial"][k]:
                # truncated at each halo's own R100c: one row per halo with its effective outer edge
                use = complete[:, k] > 0
                rows["inner"].extend([st["edges"][k]] * int(use.sum())); rows["outer"].extend(st["halo_x_out"][use, k])
                rows["mass"].extend(st["masses"][use]); rows["z"].extend([st["z"]] * int(use.sum()))
                rows["weight"].extend(complete[use, k]); rows["group"].extend([g] * int(use.sum()))
                continue
            keep = counts[:, k] > 0
            n_nodes = int(keep.sum())
            rows["inner"].extend([st["edges"][k]] * n_nodes); rows["outer"].extend([st["edges"][k + 1]] * n_nodes)
            rows["mass"].extend(node_mass[keep]); rows["z"].extend([st["z"]] * n_nodes)
            rows["weight"].extend(counts[keep, k]); rows["group"].extend([g] * n_nodes)
        if include_soap_outer:
            soap = st["soap"]
            w_all = np.ones(st["n_haloes"]) if c is None else c
            for j in range(2):
                use = soap["ok"][:, j] & (w_all > 0)
                if use.sum() < max(min(3, st["n_haloes"]), 0.5 * st["n_haloes"]) or not np.isfinite(st["soap_mean"][j]):
                    continue
                obs = np.sum(w_all[use] * soap["ratio"][use, j]) / np.sum(w_all[use])
                g = len(groups["observed"])
                groups["observed"].append(obs)
                groups["sigma_log"].append(np.hypot(st["soap_err"][j] / st["soap_mean"][j], sigma_floor))
                groups["stack"].append(s); groups["shell"].append(100 + j); groups["is_soap"].append(True)
                rows["inner"].extend(soap["inner"][use, j]); rows["outer"].extend(soap["outer"][use, j])
                rows["mass"].extend(st["masses"][use]); rows["z"].extend([st["z"]] * int(use.sum()))
                rows["weight"].extend(w_all[use]); rows["group"].extend([g] * int(use.sum()))
    groups = {k: np.asarray(v) for k, v in groups.items()}
    if equal_stack_weight and len(groups["stack"]):
        per_stack = np.bincount(groups["stack"].astype(int))
        present = per_stack[per_stack > 0]
        groups["sigma_log"] = groups["sigma_log"].astype(float) * np.sqrt(per_stack[groups["stack"].astype(int)] / present.mean())
    return {k: np.asarray(v, float) for k, v in rows.items()}, groups


class Objective:
    """log(model / data) per group, with the model averaged over the group's rows (exact shell averages)."""

    def __init__(self, rows, groups, order=16):
        self.group = rows["group"].astype(int)
        self.n_groups = len(groups["observed"])
        self.weight = rows["weight"]
        self.wsum = np.bincount(self.group, weights=self.weight, minlength=self.n_groups)
        radius, self.vw = shell_nodes(rows["inner"], rows["outer"], order)
        self.logr = np.log(radius)
        self.lm = np.log(rows["mass"] / 1e14)
        self.lz = np.log1p(rows["z"])
        self.log_observed = np.log(groups["observed"].astype(float))
        self.inv_sigma = 1.0 / groups["sigma_log"].astype(float)
        self._cache = (None, None)

    def evaluate(self, theta):
        if self._cache[0] is not None and np.array_equal(theta, self._cache[0]):
            return self._cache[1]
        lp = theta[0] + theta[3] * self.lm + theta[6] * self.lz
        lx = theta[1] + theta[4] * self.lm + theta[7] * self.lz
        lb = theta[2] + theta[5] * self.lm + theta[8] * self.lz
        lq = self.logr - lx[:, None]
        beta = np.exp(lb)[:, None]
        soft = np.logaddexp(0.0, lq)
        f = self.vw * np.exp(np.clip(np.log(ELECTRON_FRACTION) + lp[:, None] - 0.3 * lq - beta * soft, -700, 650))
        individual = f.sum(1)
        mean = lambda v: np.bincount(self.group, weights=self.weight * v, minlength=self.n_groups) / self.wsum  # noqa: E731
        prediction = np.maximum(mean(individual), 1e-300)
        bases = [individual, (f * (0.3 + beta * expit(lq))).sum(1), (f * (-beta * soft)).sum(1)]
        jac = np.empty((self.n_groups, 9))
        for i, base in enumerate(bases):
            for offset, factor in [(0, 1.0), (3, self.lm), (6, self.lz)]:
                jac[:, i + offset] = mean(base * factor) / prediction
        residual = np.log(prediction) - self.log_observed
        self._cache = (theta.copy(), (residual, jac, prediction))
        return residual, jac, prediction


def fit(objective, theta_start, free=range(9), fixed=None, starts=()):
    """Weighted least squares in log space over the free subset of theta; best of several starts."""
    free = np.asarray(list(free))
    base = np.array(theta_start, float) if fixed is None else np.array(fixed, float)

    def full(values):
        theta = base.copy()
        theta[free] = values
        return theta

    fun = lambda v: objective.evaluate(full(v))[0] * objective.inv_sigma                         # noqa: E731
    jac = lambda v: objective.evaluate(full(v))[1][:, free] * objective.inv_sigma[:, None]         # noqa: E731
    best = None
    for start in [theta_start, *starts]:
        x0 = np.clip(np.asarray(start, float)[free], LOWER[free] + 1e-9, UPPER[free] - 1e-9)
        result = least_squares(fun, x0, jac=jac, bounds=(LOWER[free], UPPER[free]), x_scale="jac",
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500)
        if best is None or result.cost < best.cost:
            best = result
    theta = full(best.x)
    residual = objective.evaluate(theta)[0]
    chi2 = float(np.sum((residual * objective.inv_sigma) ** 2))
    return {"theta": theta, "native": decode(theta), "chi2": chi2, "dof": int(max(len(residual) - len(free), 1)),
            "rms_log": float(np.sqrt(np.mean(residual**2))), "success": bool(best.success),
            "at_bound": bool(np.any(np.minimum(best.x - LOWER[free], UPPER[free] - best.x) < 1e-4)),
            "free": free.tolist(), "n_data": int(len(residual))}


def b12_theta_at(native, mass, z):
    """theta with the (P0, xc, beta) of a nine-coefficient model at (mass, z) and zero exponents."""
    p0, xc, beta = coefficients(native, mass, z)
    return np.r_[np.log([p0, xc, beta]), np.zeros(6)]


def fit_stack(stack, sigma_floor=0.03, include_soap_outer=False, n_boot=100):
    """Battaglia-form fit (free P0, xc, beta; alpha=1, gamma=-0.3) of one stack, with bootstrap errors."""
    m, z = 10 ** stack["log10_m_median"], stack["z"]
    starts = [b12_theta_at(B12, m, z), np.r_[np.log([5.0, 1.0, 5.0]), np.zeros(6)], np.r_[np.log([2.0, 2.0, 8.0]), np.zeros(6)]]
    rows, groups = stack_rows([stack], include_soap_outer, sigma_floor)
    result = fit(Objective(rows, groups), starts[0], free=[0, 1, 2], starts=starts[1:])
    draws = []
    for b in range(min(n_boot, len(stack["boot_counts"]))):
        rb, gb = stack_rows([stack], include_soap_outer, sigma_floor, boot_index=b)
        draws.append(fit(Objective(rb, gb), result["theta"], free=[0, 1, 2])["native"][:3])
    draws = np.array(draws)
    result["boot_native"] = draws
    result["p16"], result["p50"], result["p84"] = np.percentile(draws, [16, 50, 84], axis=0) if len(draws) else (None,) * 3
    return result


def fit_joint(stacks, sigma_floor=0.03, include_soap_outer=False, n_boot=100, start=B12, extra_starts=(),
              equal_stack_weight=True):
    """All nine coefficients from every stack at once; bootstrap over haloes inside each stack.

    Starts: `start` (Battaglia+12 by default), the same with zero exponents, and any `extra_starts`
    (native nine-vectors, e.g. the L1_m9 calibration); the lowest chi2 is kept.
    """
    rows, groups = stack_rows(stacks, include_soap_outer, sigma_floor, equal_stack_weight=equal_stack_weight)
    objective = Objective(rows, groups)
    flat = encode(start); flat[3:] = 0.0
    result = fit(objective, encode(start), starts=[flat, *[encode(e) for e in extra_starts if e is not None]])
    result["groups"], result["objective"] = groups, objective
    draws = []
    n_draws = min(n_boot, min(len(st["boot_counts"]) for st in stacks))
    for b in range(n_draws):
        rb, gb = stack_rows(stacks, include_soap_outer, sigma_floor, boot_index=b, equal_stack_weight=equal_stack_weight)
        draws.append(fit(Objective(rb, gb), result["theta"])["native"])
    result["boot_native"] = np.array(draws)
    if draws:
        result["p16"], result["p50"], result["p84"] = np.percentile(result["boot_native"], [16, 50, 84], axis=0)
    return result


def score(objective, native):
    """rms of log(model/data) and chi2 for a fixed nine-coefficient vector on the same groups."""
    residual = objective.evaluate(encode(native))[0]
    return {"rms_log": float(np.sqrt(np.mean(residual**2))), "chi2": float(np.sum((residual * objective.inv_sigma) ** 2)),
            "residual": residual}


def load_l1m9(path):
    try:
        return np.array(json.loads(Path(path).read_text())["parameters"], float)
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- projected model maps
def projected_model_profile(native, masses, z, x_sphere, radius, order=48, chunk=64):
    """Mean over haloes of the model's y / y200 at projected radii `radius` (R200c units).

    Each halo's line of sight is truncated at its own sphere x_sphere (R100c for reduced snapshots),
    exactly as its particles are. The substitution l = L t^2 makes the r^-0.3 cusp integrable.
    """
    nodes, weights = leggauss(order)
    t, w = (nodes + 1) / 2, weights / 2
    radius = np.asarray(radius, float)
    masses, x_sphere = np.atleast_1d(masses), np.atleast_1d(x_sphere)
    p0, xc, beta = (np.atleast_1d(v) for v in coefficients(native, masses, z))
    total = np.zeros_like(radius)
    for start in range(0, len(masses), chunk):
        sl = slice(start, start + chunk)
        R = radius[None, :, None]
        L = np.sqrt(np.maximum(x_sphere[sl, None, None] ** 2 - R**2, 0.0))
        r = np.sqrt(R**2 + (L * t**2) ** 2)
        f = gnfw(r, p0[sl, None, None], xc[sl, None, None], beta[sl, None, None])
        total += np.sum(2 * np.sum(f * 2 * L * t * w, axis=-1), axis=0)
    return total / len(masses)


def model_map(stack, native, oversample=4):
    """Pixel-averaged model map of y / y200 on the stack's map grid (mean over its haloes and axes)."""
    npix = stack["map_hat"].shape[0]
    half = stack["map_half_extent"]
    grid_r = np.r_[0.0, np.geomspace(1e-3, np.sqrt(2) * half * 1.01, 500)]
    profile = projected_model_profile(native, stack["masses"], stack["z"], stack["halo_x_sphere"], grid_r)
    sub = (np.arange(npix * oversample) + 0.5) / (npix * oversample) * 2 * half - half
    uu, vv = np.meshgrid(sub, sub)
    fine = np.interp(np.hypot(uu, vv), grid_r, profile)
    return fine.reshape(npix, oversample, npix, oversample).mean(axis=(1, 3))


def model_annuli(stack, native, n_sub=24):
    """Area mean of the model's y / y200 in the stack's annuli (complete annuli only are meaningful)."""
    edges = stack["edges"]
    frac = (np.arange(n_sub) + 0.5) / n_sub
    r2 = edges[:-1, None] ** 2 + np.diff(edges**2)[:, None] * frac          # uniform in R^2
    profile = projected_model_profile(native, stack["masses"], stack["z"], stack["halo_x_sphere"], np.sqrt(r2).ravel())
    return profile.reshape(r2.shape).mean(axis=1)


def evaluate_stack(stack, native):
    """Accuracy of a nine-coefficient model on one stack: 3D shells (resolved range) and complete annuli."""
    rows, groups = stack_rows([stack], include_soap_outer=False, equal_stack_weight=False)
    residual = Objective(rows, groups).evaluate(encode(native))[0] if len(groups["observed"]) else np.array([])
    complete_annuli = stack["yhat_valid"] & (stack["edges"][1:] <= np.min(stack["halo_x_sphere"]))
    ratio2d = stack["yhat"] / model_annuli(stack, native) - 1
    inside = stack["edges"][1:] <= np.minimum(1.0, np.min(stack["halo_x_sphere"]))
    beyond = stack["edges"][:-1] >= stack["x100_median"]
    shells = np.asarray(groups["shell"], int)
    return {"rms_log_3d": float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan,
            "mean_log_3d": float(np.mean(residual)) if residual.size else np.nan,
            "rms_log_3d_beyond_r100": float(np.sqrt(np.mean(residual[beyond[shells]] ** 2))) if np.any(beyond[shells]) else np.nan,
            "rms_frac_2d": float(np.sqrt(np.nanmean(ratio2d[complete_annuli] ** 2))) if complete_annuli.any() else np.nan,
            "mean_frac_2d_inside_r200": float(np.nanmean(ratio2d[complete_annuli & inside])) if np.any(complete_annuli & inside) else np.nan,
            "ratio2d": ratio2d, "complete_annuli": complete_annuli}
