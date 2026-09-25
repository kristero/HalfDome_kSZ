"""Physical and sparse HEALPix resolution diagnostics after the full maps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# The cluster's shared Astropy 4.2 predates its user NumPy 1.24. Use a
# compatible, private Astropy without changing the user's scientific runtime.
for base in (Path(__file__).parent, Path(__file__).parent.parent):
    private_packages = base / "runtime/python"
    if private_packages.exists():
        sys.path.insert(0, str(private_packages))
        break

import healpy as hp
import h5py
import numpy as np
import toml
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import RectBivariateSpline
from scipy.special import betaln, betainc
from scipy.spatial import ConvexHull

from conditional_prior import admissible
from prior_model import (ELECTRON_FRACTION, FIDUCIAL, H, HIGH, KEYS, LOW, MEC2, MPC, OM, SIGMA_T, central_column,
    NAMES, domain_metrics, expansion_e2, finite_pressure_integral, parameters, physical_scales, save_json)
from run_pilot import extreme_design


def pressure_validation():
    errors = []
    for xc in (1e-4, .1, .497, 4., 100.):
        for beta in (2.71, 4.35, 16., 100., 1000.):
            reference = xc**3*np.exp(betaln(2.7, beta-2.7))*betainc(2.7, beta-2.7, 1/(1+xc))
            numerical = finite_pressure_integral(1., xc, beta)
            errors.append(abs(numerical/reference-1))
    # Finite Y200 also exists when the untruncated outer integral diverges.
    # Check that regime with independent adaptive log-radius quadrature.
    for xc in (1e-4, .1, .497, 4., 100.):
        for beta in (.1, .7, 1.3, 2.7):
            upper = np.log(1/xc)
            integrand = lambda t: np.exp(2.7*t-beta*np.logaddexp(0, t))
            reference = xc**3*quad(integrand, upper-80, upper,
                                   epsabs=0, epsrel=1e-10)[0]
            errors.append(abs(finite_pressure_integral(1., xc, beta)/reference-1))
    if max(errors) > 5e-5:
        raise ValueError("Finite Y200 quadrature exceeds 0.005 percent tolerance")
    return dict(cases=len(errors), maximum_relative_error=float(max(errors)),
                relative_tolerance=5e-5, status="passed")


def catalogue_hull_diagnostic(root):
    """Exact extrema of power-law parameters over actual catalogue M,z pairs."""
    destination = root / "audit/catalogue_logmass_logredshift_hull.npy"
    if destination.exists():
        hull = np.load(destination)
    else:
        hull = np.empty((0, 2))
        path = "/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5"
        with h5py.File(path, "r") as handle:
            for start in range(0, len(handle["redshift"]), 1000000):
                mass = handle["halo_mass_m200c"][start:start+1000000]/H
                z = handle["redshift"][start:start+1000000]
                keep = np.isfinite(mass) & np.isfinite(z) & (mass >= 1e12) & (z >= 0)
                points = np.vstack([hull, np.column_stack([np.log(mass[keep]/1e14), np.log1p(z[keep])])])
                hull = points[ConvexHull(points).vertices]
        np.save(destination, hull)
    design = np.load(root / "audit/prior_design.npz")
    records = {}
    for group, key in (("sobol", "theta"), ("corners", "corners"), ("examples", "examples")):
        theta = design[key]
        values = np.exp(np.log(theta[:, 2, None])+theta[:, 5, None]*hull[None, :, 0]
                        +theta[:, 8, None]*hull[None, :, 1])
        minimum = values.min(axis=1)
        records[group] = dict(count=len(theta), los_divergent=int(np.sum(minimum <= .7)),
            infinite_thermal_energy_divergent=int(np.sum(minimum <= 2.7)),
            conservative_beta_condition_failed=int(np.sum(minimum < 2.8)))
        np.save(root / "audit" / (group+"_minimum_actual_catalogue_beta.npy"), minimum)
    save_json(root / "audit/actual_catalogue_prior_audit.json", dict(groups=records,
        hull_vertices=len(hull), method="Convex hull of all selected log(M/1e14), log(1+z); exact extrema for power laws",
        distinction="Actual catalogue support is narrower than the interpolation grid that must also remain numerically valid"))


def reference_interpolation_diagnostic(root):
    """Audit the existing B12 cache to distinguish pre-existing grid errors."""
    campaign = Path("/lustre/work/kristero10/flamingo_tsz_comparison_20260914")
    output = root / "results/reference_Battaglia12"
    if (output / "profile_validation.toml").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    first = next((root / "results").glob("*/full_map_complete.json"))
    command = json.loads(first.read_text())["command"]
    replacements = dict(zip(KEYS, (format(x, ".17g") for x in FIDUCIAL)))
    replacements["cache_dir"] = str(campaign / "paint_cache")
    command = [arg.split("=")[0]+"="+replacements[arg.split("=")[0]]
               if "=" in arg and arg.split("=")[0] in replacements else arg for arg in command]
    runtime = campaign / "runtime"
    environment = os.environ.copy()
    environment.update(HALFDOME_SOURCE_DIR=str(campaign / "code/halfdome"),
        FLAMINGO_CAMPAIGN=str(campaign), PILOT_OUTPUT=str(output), PILOT_AUDIT_ONLY="1",
        JULIA_DEPOT_PATH=str(runtime / "depot")+":/home/kristero10/.julia",
        LD_LIBRARY_PATH=str(runtime / "julia-1.12.2/lib/julia")+":"+environment.get("LD_LIBRARY_PATH", ""),
        JULIA_PKG_PRECOMPILE_AUTO="0", HDF5_USE_FILE_LOCKING="FALSE")
    with (output / "audit.log").open("w") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)


def quadrature_cost_diagnostic(root):
    output = root / "audit/quadrature_cost"
    if (output / "quadrature_cost_dense.csv").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    campaign = Path("/lustre/work/kristero10/flamingo_tsz_comparison_20260914")
    command = json.loads(next((root / "results").glob("*/full_map_complete.json")).read_text())["command"]
    theta = json.loads((root / "proposals/fit_L1_m9_iter3.json").read_text())["theta"]
    replacements = dict(zip(KEYS, (format(x, ".17g") for x in theta)))
    command = [arg.split("=")[0]+"="+replacements[arg.split("=")[0]]
               if "=" in arg and arg.split("=")[0] in replacements else arg for arg in command]
    command[4] = str(root / "code/diagnose_quadrature.jl")
    runtime = campaign / "runtime"
    environment = os.environ.copy()
    environment.update(HALFDOME_SOURCE_DIR=str(campaign / "code/halfdome"),
        FLAMINGO_CAMPAIGN=str(campaign), PILOT_OUTPUT=str(output),
        JULIA_DEPOT_PATH=str(runtime / "depot")+":/home/kristero10/.julia",
        LD_LIBRARY_PATH=str(runtime / "julia-1.12.2/lib/julia")+":"+environment.get("LD_LIBRARY_PATH", ""),
        JULIA_PKG_PRECOMPILE_AUTO="0")
    with (output / "diagnostic.log").open("w") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=120)


def summarize_quadrature_cost(root):
    values = np.genfromtxt(root / "audit/quadrature_cost/quadrature_cost_dense.csv",
                          delimiter=",", names=True, dtype=None, encoding=None)
    regular = (values["native_y"] > 1e-280) & np.isfinite(values["log_y_scaled"])
    disagreement = np.abs(np.expm1(np.log(values["native_y"][regular])-values["log_y_scaled"][regular]))
    limited = values["native_evaluations"] >= 4096
    outside = np.array([str(x).lower() == "true" for x in values["outside_painted_disc"]])
    save_json(root / "audit/quadrature_cost_summary.json", dict(
        probes=len(values), native_budget_hits=int(limited.sum()),
        budget_hits_outside_painted_disc=int(np.sum(limited & outside)),
        largest_native_evaluation_count=int(values["native_evaluations"].max()),
        largest_scaled_evaluation_count=int(values["scaled_evaluations"].max()),
        scaled_counts_at_native_budget_hits=values["scaled_evaluations"][limited].tolist(),
        maximum_regular_value_relative_disagreement=float(max(disagreement)),
        regular_value_comparison_threshold=1e-280,
        conclusion="Native faint-tail integration can hit a relative-precision floor far outside 4R200; this is a numerical failure, not physical exclusion",
        scope="Bounded diagnostic only; production operators were not replaced"))


def summarize_interpolation(root):
    table = np.load(root / "audit/projection_table.npz")
    column = RectBivariateSpline(table["logbeta"], table["logq"], table["logcolumn"])
    validation_errors = []
    for q in (1e-8, 1e-4, .1, 1., 100., 1e4):
        for beta in (1.3, 4.35, 16., 256., 2048.):
            base = .7*np.log(q)-beta*np.log1p(q)
            def integrand(u):
                logr = np.log(q)+np.log(np.cosh(u))
                return np.exp(.7*logr-beta*np.logaddexp(0, logr)-base)
            integral = quad(integrand, 0, np.arcsinh(1e8/q), epsabs=0, epsrel=1e-9, limit=400)[0]
            log_direct = np.log(2)+base+np.log(integral)
            log_tabulated = float(column.ev(np.log(beta), np.log(q)))
            validation_errors.append(abs(np.expm1(log_tabulated-log_direct)))
    save_json(root / "audit/extended_column_validation.json",
        dict(cases=len(validation_errors), max_relative_error=float(max(validation_errors)),
             beta_range=[1.3, 2048], q_range=[1e-8, 1e4], method="Independent adaptive quadrature in log-scaled LOS coordinates"))
    assert max(validation_errors) < .005
    rows = {}
    for path in (root / "results").glob("*/profile_interpolation_check.csv"):
        values = np.genfromtxt(path, delimiter=",", names=True)
        actual = toml.load(path.parent / "candidate_parameters.toml")
        theta = np.array([actual[key.replace("battaglia_", "")] for key in KEYS])
        p0, xc, beta = parameters(theta, values["mass_Msun"], values["z"])
        r200, p200 = physical_scales(values["mass_Msun"], values["z"])
        amplitude = SIGMA_T/MEC2*ELECTRON_FRACTION*p200*r200*p0*xc
        reference = amplitude*np.exp(column.ev(np.log(beta), np.log(values["r_over_R200"]/xc)))
        reference_grouped = reference.reshape(-1, 8)
        significant = (reference_grouped > reference_grouped.max(axis=1)[:, None]*1e-8).ravel()
        interpolated_error = np.abs(values["interpolated_y"]/np.maximum(reference, np.finfo(float).tiny)-1)
        direct_error = np.abs(values["direct_y"]/np.maximum(reference, np.finfo(float).tiny)-1)
        record = {}
        for label, keep in (("all_grid", significant),
                            ("z_ge_0p1", significant & (values["z"] >= .1))):
            errors = interpolated_error[keep]
            errors = errors[np.isfinite(errors)]
            record[label] = dict(count=len(errors), p95=float(np.percentile(errors, 95)),
                maximum=float(max(errors)),
                direct_quadrature_p95=float(np.percentile(direct_error[keep], 95)),
                zero_direct_where_significant_reference=int(np.sum((values["direct_y"] <= 0) & keep)))
        np.savetxt(path.parent / "independent_profile_check.csv",
            np.column_stack([values["mass_Msun"], values["z"], values["r_over_R200"],
                             reference, values["direct_y"], values["interpolated_y"],
                             direct_error, interpolated_error]),
            delimiter=",", header="mass_Msun,z,r_over_R200,independent_y,production_direct_y,production_interpolated_y,direct_relative_error,interpolated_relative_error",
            comments="")
        rows[path.parent.name] = record
    save_json(root / "audit/interpolation_summary.json", rows)


def resolution_diagnostic(root, candidates):
    """Sample actual HEALPix pixel centres without allocating a full map.

    This isolates flux loss/scatter from point-sampling. It uses an independently
    validated pressure-column spline and does not test map interpolation again.
    A normalized Gaussian beam preserves total flux, so it cannot repair this.
    """
    table = np.load(root / "audit/projection_table.npz")
    column = RectBivariateSpline(table["logbeta"], table["logq"], table["logcolumn"])
    zgrid = np.linspace(0, 5, 20001)
    chi = cumulative_trapezoid(299792.458/(H*100*np.sqrt(expansion_e2(zgrid))), zgrid, initial=0)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    rng = np.random.default_rng(314159)
    zz = rng.uniform(-1, 1, 32)
    phi = rng.uniform(0, 2*np.pi, 32)
    directions = np.column_stack([np.sqrt(1-zz*zz)*np.cos(phi), np.sqrt(1-zz*zz)*np.sin(phi), zz])
    rows = []
    for name, theta in candidates.items():
        for mass in (1e13, 1e14, 1e15):
            for z in (.1, .5, 1., 3.):
                _, xc, beta = [float(value[0]) for value in parameters(theta, mass, z)]
                if beta < 1.3:
                    continue
                radius = physical_scales(mass, z)[0]
                distance = np.interp(z, zgrid, chi)*MPC/(1+z)
                theta200 = radius/distance
                def profile(angle):
                    return np.exp(column.ev(np.full_like(angle, np.log(beta)),
                                             np.log(angle/(theta200*xc))))
                # Log-angle quadrature resolves even extremely compact cores.
                lower, upper = np.log(theta200*xc*1e-12), np.log(4*theta200)
                logangle = lower+(nodes+1)*(upper-lower)/2
                angle = np.exp(logangle)
                flux = 2*np.pi*(upper-lower)/2*np.sum(weights*angle*np.sin(angle)*profile(angle))
                for nside in (4096, 8192):
                    ratios = []
                    for direction in directions:
                        pixels = hp.query_disc(nside, direction, 4*theta200, inclusive=False, nest=False)
                        vectors = np.array(hp.pix2vec(nside, pixels)).T
                        # Match the stable chord-based angle in the Julia painter.
                        dist2 = np.sum((vectors-direction)**2, axis=1)
                        separation = np.arccos(np.clip(1-dist2/2, -1, 1))
                        separation = np.maximum(separation, np.finfo(float).eps)
                        sampled = hp.nside2pixarea(nside)*np.sum(profile(separation))
                        ratios.append(sampled/flux)
                    rows.append(dict(name=name, mass_Msun=mass, z=z, nside=nside,
                        xc=xc, beta=beta, theta200_arcmin=float(theta200*180*60/np.pi),
                        flux_ratios=ratios, median=float(np.median(ratios)),
                        p16=float(np.percentile(ratios, 16)), p84=float(np.percentile(ratios, 84)),
                        minimum=float(min(ratios)), maximum=float(max(ratios))))
        print("Pixel resolution diagnostic: "+name, flush=True)
    save_json(root / "audit/pixel_resolution.json", dict(rows=rows,
        positions=32, seed=314159, reference="Continuous spherical integral of projected profile within 4R200",
        scope="Sparse actual HEALPix pixel-centre flux test; excludes production interpolation and full spectral convergence",
        caveat="32 orientations characterize examples, not a converged rare-tail population mean"))


def extreme_parameter_examples(root):
    """Save reproducible parameter combinations behind the widest Y200 range."""
    design = np.load(root / "audit/prior_design.npz")
    mass_grid, redshift_grid = np.meshgrid(
        10**np.array([12., 13., 14., 15., 15.7]), [0., .5, 1., 2., 4., 5.])
    mass, z = mass_grid.ravel(), redshift_grid.ravel()
    reference = finite_pressure_integral(*parameters(FIDUCIAL, mass, z))
    examples = {}
    for label, key in (("sobol", "theta"), ("corners", "corners")):
        theta = design[key]
        audit = np.load(root / "audit" / (label+"_audit.npz"))
        keep = admissible(theta)
        selected = [
            ("largest_Y200", int(np.argmax(audit["max_Y200_ratio"])), True),
            ("smallest_Y200", int(np.argmin(audit["min_Y200_ratio"])), False),
            ("largest_admitted_Y200", int(np.argmax(
                np.where(keep, audit["max_Y200_ratio"], -np.inf))), True),
        ]
        for name, index, maximum in selected:
            values = finite_pressure_integral(*parameters(theta[index], mass, z))/reference
            point = int(np.argmax(values) if maximum else np.argmin(values))
            examples[label+"_"+name] = dict(
                theta=dict(zip(NAMES, theta[index].tolist())),
                mass_Msun=float(mass[point]), z=float(z[point]),
                Y200_ratio=float(values[point]),
                minimum_grid_beta=float(domain_metrics(theta[index])["min_beta"]),
                passes_outer_slope_screen=bool(keep[index]))
    save_json(root / "audit/extreme_parameter_examples.json", examples)


def projected_cutoff_diagnostic(root, candidates):
    """Fraction of the untruncated projected halo flux retained within 4R200.

    A shell outside the cylinder contributes the solid-angle fraction
    1-sqrt(1-(R/r)^2). Its rationalized form avoids cancellation at large r.
    This is an intrinsic profile diagnostic, not a measured power bias.
    """
    rows = []
    for name, theta in candidates.items():
        for mass in (1e13, 1e14, 1e15):
            for z in (0., .5, 1., 3.):
                _, xc, beta = [float(v[0]) for v in parameters(theta, mass, z)]
                fraction = None
                if beta > 2.7:
                    total = xc**3*np.exp(betaln(2.7, beta-2.7))
                    inner = finite_pressure_integral(1., xc, beta, radius=4., order=256)
                    def shell(log_radius_ratio):
                        log_r = np.log(4.)+log_radius_ratio
                        log_q = log_r-np.log(xc)
                        geometry = np.exp(-2*log_radius_ratio)/(
                            1+np.sqrt(-np.expm1(-2*log_radius_ratio)))
                        return np.exp(3*log_r-.3*log_q-beta*np.logaddexp(0, log_q))*geometry
                    outer = quad(shell, 0., 100., epsabs=0, epsrel=1e-9, limit=200)[0]
                    fraction = float((inner+outer)/total)
                    if not 0 <= fraction <= 1+5e-5:
                        raise ValueError("Invalid projected-cylinder flux fraction")
                    fraction = min(fraction, 1.)
                rows.append(dict(name=name, mass_Msun=mass, z=z, xc=xc, beta=beta,
                    retained_projected_flux_fraction=fraction,
                    omitted_projected_flux_fraction=None if fraction is None else 1-fraction))
    save_json(root / "audit/projected_cutoff.json", dict(rows=rows,
        reference="Untruncated GNFW total flux; fraction undefined if beta<=2.7",
        scope="Intrinsic projected flux within 4R200, infinite LOS; not C_ell bias or observed exclusion"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    result = json.loads((root / "pilot_results.json").read_text())
    candidates = {"Battaglia12": FIDUCIAL}
    for name, data in result["fits"].items():
        candidates["fit_"+name] = np.array(data["best"]["theta"])
    candidates.update(extreme_design())
    design = np.load(root / "audit/prior_design.npz")
    summary = {}
    for label, key in (("sobol", "theta"), ("corners", "corners"), ("examples", "examples")):
        keep = admissible(design[key])
        audit = np.load(root / "audit" / (label+"_audit.npz"))
        summary[label] = dict(count=len(keep), accepted=int(keep.sum()), fraction=float(keep.mean()),
            retained_min_Y200_ratio=float(audit["min_Y200_ratio"][keep].min()),
            retained_max_Y200_ratio=float(audit["max_Y200_ratio"][keep].max()))
    save_json(root / "audit/conditional_prior_audit.json", dict(groups=summary,
        proposed_condition="min_grid_beta >= 2.8 and conservative central LOS tail <= 1 percent",
        purpose="Conservative outer-energy/LOS condition, not observational calibration or pixel-resolution guarantee",
        finite_pressure_validation=pressure_validation(),
        candidate_accepted={name: bool(admissible(theta)) for name, theta in candidates.items()}))
    extreme_parameter_examples(root)
    catalogue_hull_diagnostic(root)
    reference_interpolation_diagnostic(root)
    quadrature_cost_diagnostic(root)
    summarize_quadrature_cost(root)
    summarize_interpolation(root)
    projected_cutoff_diagnostic(root, candidates)
    resolution_diagnostic(root, candidates)
    print("Range diagnostics finished", flush=True)


if __name__ == "__main__":
    main()
