"""Per-bin estimators (a)/(b)/(c) with paired bootstrap replicates.

A *sample* is a vector of positions into a bin's halo list (the full sample is
``arange(n)``; a replicate is one row of ``bootstrap_rows``).  For every sample
the same steps are applied, so the bootstrap describes the whole estimator:

1. stack: mean and SD (ddof=1) of the per-halo normalised profiles;
2. Cov[ln mean] on the bin's base (resolved) shells from ``n_inner`` halo
   bootstraps *of that sample* (positions ``sample[inner]``; one fixed
   ``inner`` index matrix per bin, i.e. common random numbers);
3. per cut-off (fit mask = base & outer edge <= cut & mean > 0, >= 6 shells):
   (a) diagonal sigma = max(SD/mean, 5%);
   (b) diagonal sigma = bootstrap SD of ln mean (no floor);
   (c) GLS with C = diag(sd) [lam R_pool + (1-lam) R_boot] diag(sd) + sigma_d^2 I,
       lam = sum Var(r_ij) / sum (r_ij - t_ij)^2 with Var(r_ij) from 50 resamples
       of the sample's linear profiles (the first 50 rows of ``inner``),
       lam = 1 (pooled correlation x own variances) when N < 3p, and sigma_d
       solved for chi2/nu = 1.
Replicates warm-start from the full-sample solution of the same objective and
cut-off (as radii234 did); only accepted fits are kept (others -> NaN).
"""
from __future__ import annotations

import numpy as np

from . import covariance as cov
from . import fitting as F
from . import masks as msk

OBJECTIVES = ("a", "b", "c")


def sample_statistics(values: np.ndarray, positions: np.ndarray, base: np.ndarray,
                      inner: np.ndarray | None, lambda_draws: int = 50) -> dict:
    sample = values[positions]
    out = {"mean": sample.mean(axis=0), "std": sample.std(axis=0, ddof=1), "sample": sample}
    if inner is not None:
        cb = cov.bootstrap_log_mean_covariance(values, base, positions[inner])
        out["cov_base"] = cb["cov"]
        out["sd_full"] = np.full(values.shape[1], np.nan)
        out["sd_full"][np.asarray(base, bool)] = cb["sd"]
        # Element-wise variance of the linear-profile correlation (shrinkage intensity),
        # from the first ``lambda_draws`` inner resamples of this sample.
        sub = values[:, np.asarray(base, bool)]
        with np.errstate(divide="ignore", invalid="ignore"):   # degenerate resamples of tiny bins -> NaN
            out["var_r_base"] = cov.correlation_bootstrap_variance(sub, positions[inner[:lambda_draws]])
    return out


def _sub(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base_idx = np.flatnonzero(base)
    return np.searchsorted(base_idx, np.flatnonzero(mask))


def fit_sample(family: str, stats: dict, base: np.ndarray, edges, cutoffs, pooled_corr=None,
               objectives=OBJECTIVES, starts=None, jac: str = "analytic", sigma_d="solve",
               min_halo_factor: float = 3.0, sigma_d_starts=None) -> dict:
    """Fits of one sample for every cut-off and objective: {(cut, obj): fit or None}."""
    mean, std = stats["mean"], stats["std"]
    n = len(stats["sample"])
    out = {}
    for cut in cutoffs:
        mask = msk.fit_mask(base, edges, cut, mean)
        if mask.sum() < F.MIN_SHELLS:
            for o in objectives:
                out[(cut, o)] = None
            continue
        start = (lambda o: None if starts is None else starts.get((cut, o)))
        if "a" in objectives:
            out[(cut, "a")] = F.objective_a(family, mean, std, mask, start=start("a"), edges=edges, jac=jac)
        if "b" in objectives:
            out[(cut, "b")] = F.objective_b(family, mean, stats["sd_full"], mask, start=start("b"),
                                            edges=edges, jac=jac)
        if "c" in objectives:
            sel = _sub(base, mask)
            C = stats["cov_base"][np.ix_(sel, sel)]
            p = int(mask.sum())
            force = n < min_halo_factor * p
            target = pooled_corr[np.ix_(mask, mask)]
            var_r = stats["var_r_base"][np.ix_(sel, sel)]
            force = force or not np.all(np.isfinite(var_r))
            Cs, lam = cov.shrunk_covariance(C, target, var_r=var_r, force_pooled=force)
            fit = F.objective_c(family, mean, Cs, mask, start=start("c"), edges=edges, jac=jac,
                                sigma_d=sigma_d,
                                sigma_d_start=None if sigma_d_starts is None else sigma_d_starts.get(cut))
            if fit is not None:
                fit["shrinkage_lambda"] = lam
                fit["pooled_only"] = bool(force)
                fit["hartlap"] = cov.hartlap(n, p)
            out[(cut, "c")] = fit
    return out


def run_variant(family: str, values: np.ndarray, base: np.ndarray, edges, cutoffs, rows: np.ndarray,
                inner: np.ndarray, pooled_corr=None, objectives=OBJECTIVES, jac: str = "analytic",
                keep_full_fits: bool = True) -> dict:
    """Full-sample fits + paired bootstrap replicates for one (bin, field, selection).

    Returns arrays indexed [draw, cut, objective, parameter] (ln theta), with draw 0
    the full sample and draws 1..B the replicates ``rows``.
    """
    n = len(values)
    cutoffs = list(cutoffs)
    nd = rows.shape[0] + 1
    shape = (nd, len(cutoffs), len(objectives))
    lnp = np.full(shape + (3,), np.nan)
    ok = np.zeros(shape, bool)
    chi2 = np.full(shape, np.nan)
    sigma_d = np.full((nd, len(cutoffs)), np.nan)
    lam = np.full((nd, len(cutoffs)), np.nan)
    chi2_0 = np.full((nd, len(cutoffs)), np.nan)
    means = np.full((nd, values.shape[1]), np.nan)
    sds = np.full((nd, values.shape[1]), np.nan)
    full_fits = {}
    starts = None
    sd_starts = None
    for d in range(nd):
        positions = np.arange(n) if d == 0 else rows[d - 1]
        stats = sample_statistics(values, positions, base, inner if ("b" in objectives or "c" in objectives) else None)
        means[d] = stats["mean"]
        if "sd_full" in stats:
            sds[d] = stats["sd_full"]
        fits = fit_sample(family, stats, base, edges, cutoffs, pooled_corr, objectives, starts, jac,
                          sigma_d_starts=sd_starts)
        if d == 0:
            full_fits = fits
            starts = {k: (f["parameters"] if f is not None else None) for k, f in fits.items()}
            sd_starts = {cut: fits[(cut, "c")]["sigma_d"] for cut in cutoffs
                         if "c" in objectives and fits[(cut, "c")] is not None}
        for i, cut in enumerate(cutoffs):
            for j, o in enumerate(objectives):
                f = fits[(cut, o)]
                if f is None:
                    continue
                chi2[d, i, j] = f["chi2_per_dof"]
                if F.accepted(f):
                    lnp[d, i, j] = f["log_parameters"]
                    ok[d, i, j] = True
                if o == "c":
                    sigma_d[d, i] = f["sigma_d"]
                    lam[d, i] = f.get("shrinkage_lambda", np.nan)
                    chi2_0[d, i] = f.get("chi2_per_dof_at_sigma_d_0", np.nan)
    result = {"ln_parameters": lnp, "accepted": ok, "chi2_per_dof": chi2, "sigma_d": sigma_d,
              "shrinkage_lambda": lam, "chi2_per_dof_sigma_d_0": chi2_0, "means": means,
              "ln_mean_sd": sds, "cutoffs": np.asarray(cutoffs, float), "objectives": np.asarray(objectives)}
    if keep_full_fits:
        result["full_fits"] = full_fits
    return result


def influence_table(full_fits: dict, cutoffs, objectives, n_shells: int) -> dict:
    """Influence rows [cut, objective, parameter, shell] (NaN off-mask) and N_eff."""
    H = np.full((len(cutoffs), len(objectives), 3, n_shells), np.nan)
    neff = np.full((len(cutoffs), len(objectives), 3), np.nan)
    sig = np.full((len(cutoffs), len(objectives), 3), np.nan)
    for i, cut in enumerate(cutoffs):
        for j, o in enumerate(objectives):
            f = full_fits.get((cut, o))
            if f is None:
                continue
            inf = F.fit_influence(f)
            H[i, j][:, f["mask"]] = inf["H"]
            neff[i, j] = inf["n_eff"]
            sig[i, j] = np.sqrt(np.clip(np.diag(inf["cov_log_parameters"]), 0, None))
    return {"H": H, "n_eff": neff, "analytic_sigma_ln": sig}


def summarise(lnp: np.ndarray, levels=(16, 50, 84)) -> np.ndarray:
    """Percentiles of exp(ln theta) over replicate draws (axis 0, draw 0 excluded)."""
    return np.nanpercentile(np.exp(lnp[1:]), levels, axis=0)


def paired_difference(ln_a: np.ndarray, ln_b: np.ndarray, levels=(16, 50, 84)) -> dict:
    """Paired replicate differences of ln theta (a - b), NaN-aware; draw 0 = full sample."""
    delta = ln_a - ln_b
    valid = np.isfinite(delta[1:]).all(axis=-1)
    return {"full_sample": delta[0], "percentiles": np.nanpercentile(delta[1:], levels, axis=0),
            "paired_valid_draws": valid.sum(axis=0) if valid.ndim else int(valid)}
