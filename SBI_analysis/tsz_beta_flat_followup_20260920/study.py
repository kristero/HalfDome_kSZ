"""Separate beta-support experiment; never writes to the 8k production run.

All distances below are in R200. Columns are dimensionless pressure integrals:
y = sigma_T P200 R200/(m_e c^2) * column, with the existing pressure convention.
Finite spherical cutoffs are alternative physical models, not numerical fixes.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import qmc

B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
BETA_LOW = np.array([2.8, -.2, -.5])
BETA_HIGH = np.array([16., .4, 2.])
SEED = 20260920


def evolved(theta, mass=1e13, redshift=2.):
    return theta[:3] * (mass / 1e14)**theta[3:6] * (1 + redshift)**theta[6:9]


def pressure(radius, p0, xc, beta):
    q = np.asarray(radius) / xc
    return p0 * np.exp(-.3 * np.log(q) - beta * np.log1p(q))


def column(radius, p0, xc, beta, los=1e5, outer=None, rtol=2e-10):
    """Positive, scaled quadrature, including beta <= 0.7 at finite endpoint.

    ell=radius*sinh(u) removes the disparate LOS/core length scales.
    For beta<=0.7 the transformed integrand peaks at its upper endpoint.
    The central singularity is integrable; plots/tests use strictly positive R.
    """
    if radius <= 0 or min(p0, xc, beta) <= 0:
        raise ValueError("Positive radius and evolved parameters required")
    if outer is not None:
        if radius >= outer:
            return 0.
        los = np.sqrt((outer - radius) * (outer + radius))
    loglo, loghi = np.log(radius), np.log(np.hypot(radius, los))
    peak = loghi if beta <= .7 else np.clip(np.log(.7 * xc / (beta - .7)), loglo, loghi)
    def logshape(logr):
        return .7 * logr + .3 * np.log(xc) - beta * np.logaddexp(0., logr - np.log(xc))
    scale = logshape(peak)
    upper = np.arcsinh(los / radius)
    value, error = quad(lambda u: np.exp(logshape(loglo + np.log(np.cosh(u))) - scale),
                        0., upper, epsabs=0., epsrel=rtol, limit=150)
    if not np.isfinite(value) or value <= 0 or error > 5 * rtol * value:
        raise RuntimeError("LOS quadrature failed its error check")
    return float(np.exp(np.log(2 * p0) + scale + np.log(value)))


def energy(outer, p0, xc, beta, rtol=2e-10):
    """Integral_0^outer P/P200 * (r/R200)^2 d(r/R200).

    Proportional to spherical integrated Y and thermal energy. The neglected
    lower log-radius tail is exponentially suppressed by exp(-2.7*55).
    """
    upper = np.log(outer / xc)
    peak = upper if beta <= 2.7 else min(upper, np.log(2.7 / (beta - 2.7)))
    def logshape(t):
        return 2.7 * t - beta * np.logaddexp(0., t)
    scale = logshape(peak)
    value = quad(lambda t: np.exp(logshape(t) - scale), min(peak, upper)-55,
                 upper, epsabs=0., epsrel=rtol, limit=150)[0]
    return float(np.exp(np.log(p0) + 3*np.log(xc) + scale + np.log(value)))


def savefig(root, name, figure):
    for extension in ("png", "pdf"):
        figure.savefig(root / "plots" / (name + "." + extension), bbox_inches="tight", dpi=180)
    plt.close(figure)


def profile_tests(root):
    cases = []
    for name, label, beta_values in (
        ("b12", "Battaglia12", B12[[2, 5, 8]]),
        ("shallow", "Shallow outskirts", [2.8, 0., -.5]),
        ("very_shallow", "Very shallow outskirts", [2.8, .4, -.5]),
    ):
        theta = B12.copy()
        theta[[2, 5, 8]] = beta_values
        cases.append(dict(name=name, label=label, theta=theta.tolist(),
                          evolved=evolved(theta).tolist()))
    radius = np.geomspace(1e-3, 1e5, 360)
    projected = np.geomspace(1e-3, 3.999, 180)
    colors = ["#202020", "#0072B2", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    pfig, paxes = plt.subplots(1, 3, figsize=(17, 5.2))
    records, probes, arrays = [], [], dict(radius=radius, projected=projected)
    max_refinement = 0.
    for case, color, ax in zip(cases, colors, paxes):
        p0, xc, beta = case["evolved"]
        p = pressure(radius, p0, xc, beta)
        e = np.array([energy(r, p0, xc, beta) for r in radius])
        arrays[case["name"] + "_pressure"] = p
        arrays[case["name"] + "_energy"] = e
        axes[0].loglog(radius, p, color=color, label=case["label"])
        axes[1].loglog(radius, e / energy(4., p0, xc, beta), color=color)
        vals = {}
        for outer, style, c in ((None, "-", "#202020"), (4., "--", "#0072B2"),
                                (8., "-.", "#D55E00"), (16., ":", "#009E73")):
            y = np.array([column(r, p0, xc, beta, outer=outer) for r in projected])
            key = "los1e5" if outer is None else "sphere%d" % outer
            arrays[case["name"] + "_" + key] = y
            vals[key] = column(1., p0, xc, beta, outer=outer)
            label = r"Current $L=10^5R_{200}$" if outer is None else r"$R_{\rm out}=%dR_{200}$" % outer
            ax.loglog(projected, y, style, color=c, label=label)
            for r in (.001, .1, 1., 3.999):
                endpoint = 1e5 if outer is None else np.sqrt(outer**2-r**2)
                value = column(r, p0, xc, beta, los=endpoint)
                refined = column(r, p0, xc, beta, los=endpoint, rtol=2e-12)
                max_refinement = max(max_refinement, abs(value/refined-1))
                probes.append([r, p0, xc, beta, endpoint, value])
        ax.set_title(r"%s: $\beta=%.3f$" % (case["label"], beta), fontsize=16)
        ax.set_xlabel(r"$R_\perp/R_{200}$")
        ax.grid(alpha=.18, which="major")
        records.append(dict(**case, y_at_R200=vals,
            energy_16_over_4=energy(16., p0, xc, beta)/energy(4., p0, xc, beta),
            los_doubling_fraction=column(1., p0, xc, beta, los=2e5)/vals["los1e5"]-1,
            infinite_energy_finite=bool(beta > 2.7), infinite_los_finite=bool(beta > .7)))
    axes[0].set_xlabel(r"$r/R_{200}$")
    axes[0].set_ylabel(r"$P_e/P_{200}$")
    axes[0].legend(fontsize=13)
    axes[1].set_xlabel(r"$R_{\rm out}/R_{200}$")
    axes[1].set_ylabel(r"$E_{\rm th}(<R_{\rm out})/E_{\rm th}(<4R_{200})$")
    for ax in axes:
        for boundary in (4., 8., 16.):
            ax.axvline(boundary, color=".65", linewidth=.7)
        ax.grid(alpha=.18)
    fig.tight_layout()
    paxes[0].set_ylabel(r"$y/[\sigma_T P_{200}R_{200}/(m_ec^2)]$")
    paxes[1].legend(fontsize=12, loc="lower left")
    pfig.tight_layout()
    savefig(root, "pressure_and_energy", fig)
    savefig(root, "projected_profiles", pfig)
    np.savez_compressed(root / "results/profiles.npz", **arrays)
    np.savetxt(root / "results/column_probes.csv", probes, delimiter=",",
               header="radius,p0,xc,beta,los,python_column", comments="")
    return dict(mass_Msun=1e13, redshift=2., cases=records,
                max_tolerance_refinement=max_refinement, python_probes=len(probes))


def minimum_beta(samples, points):
    result = np.empty(len(samples))
    for start in range(0, len(samples), 1024):
        batch = samples[start:start+1024]
        result[start:start+len(batch)] = np.exp(np.log(batch[:, :1]) +
            batch[:, 1:2]*points[None, :, 0] + batch[:, 2:3]*points[None, :, 1]).min(axis=1)
    return result


def prior_tests(root):
    unit = qmc.Sobol(3, scramble=True, seed=SEED).random_base2(18)
    samples = BETA_LOW + unit*(BETA_HIGH-BETA_LOW)
    hull = np.load(root / "inputs/catalogue_logmass_logredshift_hull.npy")
    cache = np.array([[np.log(m/1e14), np.log1p(z)] for m in (1e12, 10**15.7) for z in (.001, 5.)])
    cache_min, actual_min = minimum_beta(samples, cache), minimum_beta(samples, hull)
    old = np.load(root / "inputs/production_theta.npy")[:, [2, 5, 8]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for index, (ax, label) in enumerate(zip(axes, [r"$\beta_0$", r"$\alpha_{m,\beta}$", r"$\alpha_{z,\beta}$"])):
        bins = np.linspace(BETA_LOW[index], BETA_HIGH[index], 37)
        for data, name, color in ((old, "Existing 8k", "#777777"),
                (samples[cache_min>2.7], "Untruncated: cache", "#D55E00"),
                (samples[actual_min>2.7], "Untruncated: catalogue", "#0072B2")):
            ax.hist(data[:, index], bins=bins, density=True, histtype="step", linewidth=2,
                    label=name, color=color)
        ax.hlines(1/(BETA_HIGH[index]-BETA_LOW[index]), BETA_LOW[index], BETA_HIGH[index],
                  color="#009E73", linewidth=2.5, label="Flat: finite radius")
        ax.axvline(B12[[2, 5, 8]][index], color="black", linestyle=":", label="Battaglia12")
        ax.set_xlabel(label)
        ax.set_xlim(BETA_LOW[index], BETA_HIGH[index])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Probability density")
    axes[1].legend(fontsize=10)
    fig.tight_layout()
    savefig(root, "beta_priors", fig)
    np.savez_compressed(root / "results/beta_support.npz", beta=samples,
                        minimum_beta_cache=cache_min, minimum_beta_catalogue=actual_min)
    # A discretized feasibility experiment for uniform *correlated* marginals.
    # Success at cell centres is not certification of continuous cell volumes.
    nbin = 48
    centres = [lo+(np.arange(nbin)+.5)*(hi-lo)/nbin for lo, hi in zip(BETA_LOW, BETA_HIGH)]
    grid = np.stack(np.meshgrid(*centres, indexing="ij"), axis=-1).reshape(-1, 3)
    ipf = {}
    for name, points in (("cache", cache), ("catalogue", hull)):
        support = (minimum_beta(grid, points)>2.7).reshape((nbin,)*3)
        zero_slices = [np.flatnonzero(~support.any(axis=tuple(j for j in range(3) if j != i))).tolist() for i in range(3)]
        if any(zero_slices):
            ipf[name] = dict(feasible=False, reason="Empty marginal slices", empty_slices=zero_slices)
            continue
        weights = support.astype(float)
        weights /= weights.sum()
        for iteration in range(10000):
            for axis in range(3):
                marginal = weights.sum(axis=tuple(j for j in range(3) if j != axis))
                shape = [1, 1, 1]
                shape[axis] = nbin
                weights *= (1/(nbin*marginal)).reshape(shape)
            error = max(np.max(np.abs(weights.sum(axis=tuple(j for j in range(3) if j != i))*nbin-1)) for i in range(3))
            if error < 1e-8:
                break
        ipf[name] = dict(feasible=bool(error < 1e-8), iterations=iteration+1,
                         max_relative_marginal_error=float(error), scope="48 cubed cell centres only")
        np.savez_compressed(root / ("results/correlated_flat_%s.npz" % name), weights=weights,
                            centres=np.array(centres), support=support)
    return dict(samples=len(samples), beta_lower=BETA_LOW.tolist(), beta_upper=BETA_HIGH.tolist(),
        cache_energy_acceptance=float(np.mean(cache_min>2.7)),
        catalogue_energy_acceptance=float(np.mean(actual_min>2.7)),
        cache_los_failure_fraction=float(np.mean(cache_min<=.7)),
        catalogue_los_failure_fraction=float(np.mean(actual_min<=.7)),
        cache_alpha_m_beta_absolute_upper=float(np.log(16*1.001**2/2.7)/np.log(100)),
        correlated_uniform_grid_test=ipf)


def noise_tests(root):
    """Gaussian independent-mode check, conditional on a fixed signal.

    This isolates split-depth/correlation algebra; it is not a masked SO sky
    simulation. N=1 denotes instrumental full-depth power in this toy model.
    """
    rng = np.random.default_rng(SEED+1)
    modes, trials = 201, 30000
    rows = []
    for signal in (.01, 1., 100.):
        for name, noise, common in (("two_full_depth", 1., 0.), ("two_half_depth", 2., 0.),
                                     ("shared_residual", 2., .5)):
            # A common residual orthogonal to the fixed signal isolates F.
            f = np.linspace(-1., 1., modes)
            f *= np.sqrt(common / np.mean(f*f))
            sky = np.sqrt(signal) + f
            estimates = []
            for start in range(0, trials, 500):
                size = min(500, trials-start)
                a = sky + rng.normal(size=(size, modes))*np.sqrt(noise)
                b = sky + rng.normal(size=(size, modes))*np.sqrt(noise)
                estimates.append(np.mean(a*b, axis=1))
            estimates = np.concatenate(estimates)
            mean, variance = signal+common, (2*(signal+common)*noise+noise**2)/modes
            z = (estimates.mean()-mean)/np.sqrt(variance/trials)
            variance_ratio = estimates.var(ddof=1)/variance
            if abs(z)>6 or abs(variance_ratio-1)>.05:
                raise RuntimeError("Noise Monte Carlo disagrees with analytic moments")
            rows.append(dict(model=name, signal=signal, noise_each=noise, common_power=common,
                expected_mean=mean, measured_mean=float(estimates.mean()), mean_z=float(z),
                expected_variance=variance, measured_variance=float(estimates.var(ddof=1)),
                variance_ratio=float(variance_ratio)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ratio = np.geomspace(1e-3, 1e3, 400)
    axes[0].semilogx(ratio, np.sqrt((4*ratio+4)/(2*ratio+1)), color="#0072B2")
    measured = [np.sqrt(rows[k+1]["measured_variance"]/rows[k]["measured_variance"]) for k in (0, 3, 6)]
    axes[0].scatter([.01, 1, 100], measured, color="#D55E00", zorder=3, label="Monte Carlo")
    axes[0].set_xlabel(r"$C_\ell^{yy}/N_\ell^{\rm full}$")
    axes[0].set_ylabel(r"$\sigma_{\rm half-depth}/\sigma_{\rm full-depth}$")
    axes[0].legend(fontsize=13)
    subset = rows[3:6]
    axes[1].bar(np.arange(3), [r["measured_mean"]-1 for r in subset], color=["#777777", "#0072B2", "#D55E00"])
    axes[1].errorbar(np.arange(3), [r["measured_mean"]-1 for r in subset],
                    yerr=[np.sqrt(r["expected_variance"]/trials) for r in subset], fmt="none", color="black", capsize=5)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["Full-depth\nsplits", "Half-depth\nsplits", "Shared\nresidual"], fontsize=13)
    axes[1].set_ylabel(r"$\langle\widehat C_\ell^{AB}\rangle-C_\ell^{yy}$")
    fig.tight_layout()
    savefig(root, "noise_split_validation", fig)
    return dict(seed=SEED+1, trials_per_case=trials, independent_real_modes=modes,
                conditional_fixed_signal=True, results=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root
    for subdir in ("results", "plots", "logs"):
        (root/subdir).mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "lines.linewidth": 2., "pdf.fonttype": 42, "ps.fonttype": 42})
    start = time.perf_counter()
    results = dict(profiles=profile_tests(root), priors=prior_tests(root), noise=noise_tests(root))
    results["wall_seconds"] = time.perf_counter()-start
    results["scope"] = "Profile integration, support and ideal noise tests only; no new full-sky production certification"
    (root/"results/summary.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
