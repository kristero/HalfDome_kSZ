"""Separate pressure-fit radius from projected truncation in conditional tSZ.

Only compact, already saved halo counts and CLASS ingredients are read.
All sixteen scenarios use the same mass support and published fallback.
The two-halo term is a linear-bias forecast, not a gas-ownership measurement.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT.parent / "sensitivity_projects_L1_m9_0077_20261009"
spec = importlib.util.spec_from_file_location("saved_lightcone_analytic", PREVIOUS / "analytic_observables/lightcone_analytic.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
C = A.C

RADII = np.arange(1, 5, dtype=float)
CASES = np.array([(i, j) for i in range(4) for j in range(4)])
LABELS = np.array([f"fit{int(RADII[i])}_cut{int(RADII[j])}" for i, j in CASES])
REFERENCE = 15
ELL = np.unique(np.r_[np.geomspace(20., 6000., 80), 100., 1000., 3000.])
SELECT_ELL = np.array([100., 1000., 3000.])
MASS_GROUP_EDGES = np.array([0., 13.5, 14.5, 15.1, 15.388243, 20.])
MASS_GROUP_NAMES = np.array(["below common calibration", "13.5-14.5", "14.5-15.1", "15.1-max calibrated", "above max calibrated"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pressure_shape(parameters, x):
    """P_e/P200 in the exact XGPaint B12 convention; no extra h or X_H."""
    p0, xc, beta = np.moveaxis(np.asarray(parameters), -1, 0)
    q = x / xc[..., None]
    return .5176 * p0[..., None] * q**(-.3) * (1 + q)**(-beta[..., None])


def evaluate_shapes(mass, x, fit, support):
    """Use a bin's fitted dimensionless shape, or a common published fallback.

    A dimensionless stack is constant within each original 0.2-dex bin.
    The physical P200 and R200 are evaluated at each actual catalogue-cell mass.
    No interpolation across a supported/unsupported boundary is performed.
    """
    logmass = np.log10(mass)
    edges = fit["mass_edges"]
    bins = np.searchsorted(edges, logmass, side="right") - 1
    valid = (bins >= 0) & (bins < len(support))
    safe_bins = np.clip(bins, 0, len(support)-1)
    valid &= support[safe_bins] & (logmass <= float(fit["calibrated_logmass_max"]))
    parameters = fit["parameters"][safe_bins].transpose(1, 0, 2)
    profiles = pressure_shape(parameters, x)
    fallback = A.published_profile(0, mass, 0., x)
    profiles[:, ~valid] = fallback[None, ~valid]
    # Derivatives with respect to ln(P0), ln(xc), ln(beta).
    p0, xc, beta = np.moveaxis(parameters, -1, 0)
    q = x[None, None, :] / xc[..., None]
    derivative = profiles[:, None] * np.stack([
        np.ones_like(q), .3 + beta[..., None]*q/(1+q),
        -beta[..., None]*np.log1p(q),
    ], axis=1)
    derivative[:, :, ~valid] = 0.
    return profiles, derivative, safe_bins, valid


def edge_window(x, radius):
    """A cosine taper over the outermost ten percent of the finite sphere."""
    t = np.clip((x/radius-.9)/.1, 0., 1.)
    return .5*(1+np.cos(np.pi*t))


def transform_gate(fit, support):
    """Compare coarse transforms with independently refined analytic profiles.

    The normalization is the zero-mode integral, avoiding a misleading large
    percentage at a Fourier zero. Quad integrates oscillations analytically.
    """
    frequencies = np.r_[0., .001, .199, .201, 1., 3., 10., 30., 100., 300., 1000.]
    rows = []
    for b in np.flatnonzero(support)[::max(1, np.count_nonzero(support)//3)]:
        for ri in range(4):
            p = fit["parameters"][b, ri]
            for cut in RADII:
                x = np.unique(np.r_[np.geomspace(1e-5, cut, 256), .9*cut])
                actual = A.linear_radial_kernel(x, frequencies) @ pressure_shape(p, x)
                exact = []
                for q in frequencies:
                    if q == 0:
                        value = integrate.quad(lambda r: 4*np.pi*r*r*pressure_shape(p, np.array([max(r, 1e-30)]))[0], 0, cut, epsabs=1e-9, epsrel=1e-8)[0]
                    else:
                        value = 4*np.pi/q*integrate.quad(lambda r: r*pressure_shape(p, np.array([max(r, 1e-30)]))[0], 0, cut, weight="sin", wvar=q, epsabs=1e-9, epsrel=1e-8)[0]
                    exact.append(value)
                exact = np.asarray(exact)
                error = np.abs(actual-exact)/exact[0]
                rows.append(dict(bin_index=int(b), fit_radius=float(ri+1), cut_radius=float(cut), maximum_absolute_error_over_zero_mode=float(error.max())))
    maximum = max(row["maximum_absolute_error_over_zero_mode"] for row in rows)
    assert maximum < .003, maximum
    return dict(passed=True, q=frequencies, checks=rows, maximum_absolute_error_over_zero_mode=maximum,
                tolerance=.003, note="256-node exact piecewise-linear transform versus independent adaptive quadrature of the analytic gNFW; high-frequency absolute error normalized to Y0.")


def run():
    started = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE/"figures").mkdir(exist_ok=True)
    shape_path = ROOT/"fit_inputs/profile_shapes.npz"
    parameter_path = ROOT/"fit_inputs/fit_parameters.npz"
    shapes = dict(np.load(shape_path))
    fit = dict(np.load(parameter_path))
    for key in ("mass_edges", "calibrated_logmass_max"):
        fit[key] = shapes[key]
    support = shapes["common_support"].astype(bool)
    assert fit["parameters"].shape == (15, 4, 3)
    source_hashes = {str(p): sha(p) for p in (shape_path, parameter_path, A.COUNTS, A.INGREDIENTS)}
    hist = dict(np.load(A.COUNTS))
    counts = hist["N"]
    z = .5*(hist["z_edges"][1:]+hist["z_edges"][:-1])
    masses = np.exp(.5*(hist["lnm_edges"][1:]+hist["lnm_edges"][:-1]))/A.H
    ingredients = A.Ingredients()
    chi = ingredients.da_spline(hist["z_edges"])*(1+hist["z_edges"])
    volume = np.maximum(np.diff(chi**3)/3, 1e-30)  # comoving Mpc^3 per steradian
    ng = len(MASS_GROUP_NAMES)
    ne, nz, nm = len(ELL), len(z), len(masses)
    selected = np.searchsorted(ELL, SELECT_ELL)
    primary = np.array([3, 7, 11, 15, 12, 13, 14])
    zero = np.zeros(16)
    self_by_z = np.zeros((16, nz, ne))
    self_by_mass = np.zeros((16, nm, ne))
    two_by_z = np.zeros_like(self_by_z)
    two_cross_mass = np.zeros((16, ng, ng, ne))
    shape_self = np.zeros((16, ne))
    shape_two = np.zeros_like(shape_self)
    # Four taper cases hold the 4R fit fixed, with each cut radius.
    taper_self = np.zeros((4, ne)); taper_two = np.zeros_like(taper_self)
    taper_matched_self = np.zeros_like(taper_self); taper_matched_two = np.zeros_like(taper_self)
    taper_mean = np.zeros(4)
    grad_self = np.zeros((16, 15, len(SELECT_ELL), 3))
    grad_two = np.zeros_like(grad_self)
    for j, zz in enumerate(z):
        indices = np.flatnonzero(counts[j] > 0)
        if not len(indices):
            continue
        mass = masses[indices]
        N = counts[j, indices]
        da, radius, amp, _ = ingredients.geometry(zz, mass)
        q = radius[:, None]/da*(ELL[None]+.5)
        q_all = np.c_[np.zeros(len(mass)), q]
        bias = ingredients.bias(zz, mass)
        pk = ingredients.pk(zz, ELL)
        group = np.clip(np.searchsorted(MASS_GROUP_EDGES, np.log10(mass), side="right")-1, 0, ng-1)
        # First collect every finite-cut transform; shape normalization needs 4R.
        transforms = []; derivatives = []; tapered = []
        for cut in RADII:
            x = np.unique(np.r_[np.geomspace(1e-5, cut, 256), .9*cut])
            f0, df0, bins, valid = evaluate_shapes(mass, x, fit, support)
            evolution = A.published_profile(0, mass, zz, x)/A.published_profile(0, mass, 0., x)
            evolved = f0*evolution[None]
            weights = A.linear_radial_kernel(x, q_all)
            kernel = np.einsum("mlx,fmx->fml", weights, evolved, optimize=True)
            dkernel = np.einsum("mlx,fpmx->fpml", weights[:, 1:][:, selected], df0*evolution[None, None], optimize=True)
            window = edge_window(x, cut)
            tkernel = np.einsum("mlx,mx->ml", weights, evolved[3]*window, optimize=True)
            transforms.append(kernel*amp[None, :, None])
            derivatives.append(dkernel*amp[None, None, :, None])
            tapered.append(tkernel*amp[:, None])
        reference_y0 = transforms[3][3, :, 0]
        for out, (fi, ci) in enumerate(CASES):
            yy = transforms[ci][fi]
            y0, y = yy[:, 0], yy[:, 1:]
            d1 = N[:, None]*y*y/(4*np.pi)
            self_by_z[out, j] = d1.sum(axis=0)
            self_by_mass[out, indices] += d1
            zero[out] += N@y0/(4*np.pi)
            weighted = N[:, None]*bias[:, None]*y/(4*np.pi)
            W = weighted.sum(axis=0)
            two_by_z[out, j] = pk*W*W/volume[j]
            grouped = np.array([weighted[group == k].sum(axis=0) for k in range(ng)])
            two_cross_mass[out] += grouped[:, None]*grouped[None, :]*pk[None, None]/volume[j]
            renorm = reference_y0/y0
            sy = y*renorm[:, None]
            shape_self[out] += np.sum(N[:, None]*sy*sy, axis=0)/(4*np.pi)
            sW = np.sum(N[:, None]*bias[:, None]*sy, axis=0)/(4*np.pi)
            shape_two[out] += pk*sW*sW/volume[j]
            dy = derivatives[ci][fi].transpose(1, 2, 0)  # [mass, selected ell, parameter]
            d1_grad = 2*N[:, None, None]*y[:, selected, None]*dy/(4*np.pi)
            d2_grad = 2*pk[selected][None, :, None]*W[selected][None, :, None]/volume[j]*N[:, None, None]*bias[:, None, None]*dy/(4*np.pi)
            for b in np.flatnonzero(support):
                use = valid & (bins == b)
                grad_self[out, b] += d1_grad[use].sum(axis=0)
                grad_two[out, b] += d2_grad[use].sum(axis=0)
        for ci in range(4):
            yy = tapered[ci]
            y0, y = yy[:, 0], yy[:, 1:]
            taper_mean[ci] += N@y0/(4*np.pi)
            taper_self[ci] += np.sum(N[:, None]*y*y, axis=0)/(4*np.pi)
            W = np.sum(N[:, None]*bias[:, None]*y, axis=0)/(4*np.pi)
            taper_two[ci] += pk*W*W/volume[j]
            # Compare edge redistribution at the same integrated Y as sharp cut.
            ty = y*(transforms[ci][3, :, 0]/y0)[:, None]
            taper_matched_self[ci] += np.sum(N[:, None]*ty*ty, axis=0)/(4*np.pi)
            W = np.sum(N[:, None]*bias[:, None]*ty, axis=0)/(4*np.pi)
            taper_matched_two[ci] += pk*W*W/volume[j]
        if j % 75 == 0:
            print(f"tSZ halo model shell {j+1}/{nz}", flush=True)
    one = self_by_z.sum(axis=1)
    two = two_by_z.sum(axis=1)
    total = one+two
    closure = float(np.max(np.abs(two_cross_mass.sum(axis=(1, 2))-two)/np.maximum(two, 1e-100)))
    assert closure < 1e-12, closure
    gate = transform_gate(fit, support)
    factor = ELL*(ELL+1)/(2*np.pi)*A.T_CMB_UK**2
    np.savez_compressed(HERE/"tsz_halo_terms.npz", ell=ELL, fit_radii=RADII, cut_radii=RADII, cases=CASES,
        labels=LABELS, cl_1h=one, cl_linear_2h=two, cl_total=total,
        dl_1h_uK2=one*factor, dl_linear_2h_uK2=two*factor, dl_total_uK2=total*factor,
        mean_compton_y=zero, shape_matched_cl_1h=shape_self, shape_matched_cl_linear_2h=shape_two,
        cl_1h_by_z=self_by_z, cl_1h_by_mass=self_by_mass, cl_linear_2h_by_z=two_by_z,
        cl_linear_2h_cross_mass=two_cross_mass, mass_group_edges=MASS_GROUP_EDGES,
        mass_group_names=MASS_GROUP_NAMES, mass_msun=masses, z=z,
        taper_cl_1h=taper_self, taper_cl_linear_2h=taper_two, taper_mean_y=taper_mean,
        taper_Ymatched_cl_1h=taper_matched_self, taper_Ymatched_cl_linear_2h=taper_matched_two,
        derivative_ell=SELECT_ELL, derivative_cl_1h=grad_self, derivative_cl_linear_2h=grad_two,
        common_support=support, calibrated_logmass_max=fit["calibrated_logmass_max"])
    bootstrap = bootstrap_linear_response(fit, support, one[:, selected], two[:, selected], grad_self, grad_two)
    write_csv(one, two, zero, shape_self, shape_two)
    plotting(one, two, shape_self, shape_two, taper_self, taper_two, taper_matched_self, taper_matched_two)
    metrics = summary(one, two, zero, shape_self, shape_two)
    metadata = dict(passed=True, source_SHA256=source_hashes, source_hashes_unchanged=all(sha(p)==h for p,h in source_hashes.items()),
        catalogue_halos=int(counts.sum()), count_grid=counts.shape, common_support=support,
        calibration_mass_edges=fit["mass_edges"], calibrated_logmass_max=fit["calibrated_logmass_max"],
        scenarios=LABELS, cosmology=dict(h=.68, Omega_m=.31, Omega_b=.049),
        profile_definition="F=P_e/P200=.5176 P0 (x/xc)^(-.3) (1+x/xc)^(-beta)",
        redshift_evolution="F(M,z,x)=F(M,0,x)*B12published(M,z,x)/B12published(M,0,x); assumed response, not a calibrated redshift fit",
        mass_support="Identical support for all16 cases. Exact published B12 fallback for unsupported original 0.2dex bins or above the largest calibrated mass. No interpolation across boundaries.",
        physical_normalization="Actual catalogue cell mass in P200 and R200; cell mass/redshift centers; full original421x238 count histogram",
        projection="Finite sphere. y_l=(sigma_T/m_e c^2)P200 R200^3/DA^2 *4pi integral_0^Rcut x^2 F sinc[(ell+.5)R200 x/DA]dx.",
        one_halo="sum N |y_l|^2/(4pi), deterministic mean profiles. Individual halo scatter is not painted.",
        two_halo="Linear-bias Limber forecast: sum_z P_lin/V_shell_per_sr *[sum_mass N b y_l/(4pi)]^2; cross-mass terms explicitly retained.",
        two_halo_caveat="All-gas stacks include correlated environment. Adding a linear twohalo term is a conditional comparison and may double count that gas. This is not a unique gas 1h/2h ownership decomposition.",
        angular_geometry_caveat="Flat-sky profile kernel plus Limber linear2h. Low-ell accuracy is not certified by this matched sensitivity comparison.",
        truncation_reference="Fixed4R fit cutat1,2,3,4R. Fit-radius comparison uses cuts4R for allfourfits.",
        shape_control="Every finite-cut halo is renormalized to integratedY of the4Rfit cutat4R at the same mass,z; isolates shape redistribution from flux changes.",
        taper_control="Cosine taper over0.9Rcut–Rcut. Nativeflux and separately sharp-cut-Y-matched controls; not a physical preferred edge.",
        radial_transform=gate, cross_mass_closure_relative=closure,
        numerical_uncertainty="Transform interpolation gate only. No complete mass/redshift cell, halo-bias, cosmology, evolution, ownership or lowell model uncertainty.",
        bootstrap=bootstrap, zero_mode="Reported as meanComptonY only; not a Limber C0.",
        elapsed_seconds=time.perf_counter()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        application_threads=1, jobs_submitted=0, new_snapshot_particle_reads=0, CLASS_calls=0)
    assert metadata["source_hashes_unchanged"]
    C.write_json(HERE/"summary.json", metrics)
    C.write_json(HERE/"verification.json", metadata)
    print(json.dumps(C.plain(dict(metrics=metrics, elapsed_seconds=metadata["elapsed_seconds"], peak_RSS_GiB=metadata["peak_RSS_GiB"])), indent=2), flush=True)


def bootstrap_linear_response(fit, support, one, two, grad_one, grad_two):
    """Propagate paired bootstrap parameter vectors, without independent priors.

    This is a first-order response of the mean-profile forecast, not full
    nonlinear bootstrap spectra and not individual-halo scatter/cosmic variance.
    Unsupported/fallback bins contribute fixed flux and no fitted uncertainty.
    """
    draws = fit["bootstrap"]
    ndraw = draws.shape[2]
    valid = np.isfinite(draws).all(axis=-1) & (draws > 0).all(axis=-1)
    delta_one = np.zeros((16, ndraw, len(SELECT_ELL)))
    delta_two = np.zeros_like(delta_one)
    bound_bins = []
    for ci, (fi, _) in enumerate(CASES):
        for b in np.flatnonzero(support):
            use = valid[b, fi]
            if not use.all():
                bound_bins.append(dict(bin_index=int(b), fit_radius=int(fi+1), valid_draws=int(use.sum()), total_draws=ndraw))
            # Invalid draws have zero centered response; receipt flags these.
            log_draw = np.log(np.maximum(draws[b, fi], 1e-300))
            centered = log_draw - np.mean(log_draw[use], axis=0)
            centered[~use] = 0.
            delta_one[ci] += centered @ grad_one[ci, b].T
            delta_two[ci] += centered @ grad_two[ci, b].T
    delta_total = delta_one+delta_two
    baseline = one+two
    paired_percent = 100*(delta_total/baseline[:, None]-delta_total[REFERENCE]/baseline[REFERENCE])
    np.savez_compressed(HERE/"bootstrap_linear_response.npz", ell=SELECT_ELL, labels=LABELS,
        delta_cl_1h=delta_one, delta_cl_linear_2h=delta_two, delta_cl_total=delta_total,
        fractional_meanprofile_std_total=delta_total.std(axis=1, ddof=1)/baseline,
        paired_percent_response_vs_reference=paired_percent,
        paired_percent_p16_p84=np.percentile(paired_percent, [16, 84], axis=1),
        common_support=support, note="First-order correlated mean-profile bootstrap response; fixed fallback and fixed catalogue.")
    return dict(method="First-order response to each complete paired ln(P0),ln(xc),ln(beta) bootstrap vector; bins treated independent, matched draw labels acrossfitradii preserve pairedcasecovariance.",
        draws=ndraw, ell=SELECT_ELL, missing_draw_groups=bound_bins,
        excluded_uncertainties="Individual halo scatter, cosmic variance, fallback/evolution/ownership/bias/cosmology/core-model errors; nonlinear and boundary effects are not captured.",
        fractional_meanprofile_std_total=delta_total.std(axis=1, ddof=1)/baseline)


def write_csv(one, two, zero, shape_one, shape_two):
    with (HERE/"tsz_fit_cut_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fit_R200c", "cut_R200c", "ell", "Cl_1h", "Cl_linear_2h", "Cl_total", "linear_2h_fraction", "total_change_percent", "Ymatched_total_change_percent", "mean_y"])
        for c, (fi, ci) in enumerate(CASES):
            for l, ell in enumerate(ELL):
                writer.writerow([fi+1, ci+1, ell, one[c,l], two[c,l], one[c,l]+two[c,l], two[c,l]/(one[c,l]+two[c,l]), 100*((one[c,l]+two[c,l])/(one[REFERENCE,l]+two[REFERENCE,l])-1), 100*((shape_one[c,l]+shape_two[c,l])/(one[REFERENCE,l]+two[REFERENCE,l])-1), zero[c]])


def summary(one, two, zero, shape_one, shape_two):
    output = []
    for ell in SELECT_ELL:
        i = np.where(ELL == ell)[0][0]
        for c, (fi, ci) in enumerate(CASES):
            output.append(dict(fit_radius=int(fi+1), cut_radius=int(ci+1), ell=float(ell),
                total_change_percent=100*((one[c,i]+two[c,i])/(one[REFERENCE,i]+two[REFERENCE,i])-1),
                one_halo_change_percent=100*(one[c,i]/one[REFERENCE,i]-1),
                linear_twohalo_change_percent=100*(two[c,i]/two[REFERENCE,i]-1),
                linear_twohalo_fraction=two[c,i]/(one[c,i]+two[c,i]),
                mean_y=float(zero[c]),
                Ymatched_total_change_percent=100*((shape_one[c,i]+shape_two[c,i])/(one[REFERENCE,i]+two[REFERENCE,i])-1)))
    return dict(reference="fit4_cut4", selected_multipole_metrics=output)


def save(fig, name):
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(HERE/"figures"/f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plotting(one, two, shape_one, shape_two, taper_one, taper_two, taper_matched_one, taper_matched_two):
    plt.rcParams.update({"font.size":17, "axes.labelsize":20, "axes.titlesize":20,
        "legend.fontsize":14, "xtick.labelsize":17, "ytick.labelsize":17,
        "axes.spines.top":False, "axes.spines.right":False})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    factor = ELL*(ELL+1)/(2*np.pi)*A.T_CMB_UK**2
    total = one+two
    for title, cases, name in [("Fit radius; all cuts 4R", [3,7,11,15], "fit_radius_fixed4_terms"),
                                ("Cut radius; fixed 4R fit", [12,13,14,15], "truncation_fixed4fit_terms")]:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.3))
        for k,c in enumerate(cases):
            for ax, values, panel in zip(axes, (one,two,total), ("1-halo", "Linear 2-halo", "Sum")):
                ax.plot(ELL, values[c]*factor, color=colors[k], label=f"{k+1}R")
                ax.set(xscale="log", yscale="log", xlabel=r"$\ell$", ylabel=r"$T_{\rm CMB}^2D_\ell^{yy}\ [\mu K^2]$", title=panel)
        axes[0].legend(title=title, fontsize=14, title_fontsize=14)
        save(fig, name)
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.3))
        for k,c in enumerate(cases):
            for ax, values, panel in zip(axes, (one,two,total), ("1-halo", "Linear 2-halo", "Sum")):
                ax.plot(ELL, 100*(values[c]/values[REFERENCE]-1), color=colors[k], label=f"{k+1}R")
                ax.axhline(0,color=".5",lw=1)
                ax.set(xscale="log",xlabel=r"$\ell$",ylabel="Change from fit4/cut4 [%]",title=panel)
        axes[0].legend(title=title, fontsize=14,title_fontsize=14)
        save(fig, name+"_ratios")
    fig, axes = plt.subplots(1,3,figsize=(18,5.4))
    for ax,ell in zip(axes,SELECT_ELL):
        i=np.where(ELL==ell)[0][0]
        data=(100*(total[:,i]/total[REFERENCE,i]-1)).reshape(4,4)
        limit=max(np.max(np.abs(data)),1.)
        im=ax.imshow(data,origin="lower",cmap="RdBu_r",vmin=-limit,vmax=limit)
        for r in range(4):
            for c in range(4):ax.text(c,r,f"{data[r,c]:+.1f}%",ha="center",va="center",fontsize=15,color="white" if abs(data[r,c])>.65*limit else "black")
        ax.set(xticks=range(4),xticklabels=range(1,5),yticks=range(4),yticklabels=range(1,5),xlabel=r"Cut radius [$R_{200c}$]",ylabel=r"Fit radius [$R_{200c}$]",title=rf"$\ell={int(ell)}$")
    save(fig,"fit_cut_matrix_total")
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for k,c in enumerate([3,7,11,15]):
        axes[0].plot(ELL,100*(total[c]/total[REFERENCE]-1),color=colors[k],label=f"Fit {k+1}R")
        axes[0].plot(ELL,100*((shape_one[c]+shape_two[c])/total[REFERENCE]-1),color=colors[k],ls="--")
    for k,c in enumerate([12,13,14,15]):
        axes[1].plot(ELL,100*(total[c]/total[REFERENCE]-1),color=colors[k],label=f"Cut {k+1}R")
        axes[1].plot(ELL,100*((shape_one[c]+shape_two[c])/total[REFERENCE]-1),color=colors[k],ls="--")
    for ax,title in zip(axes,("All cuts 4R", "Fixed 4R fit")):
        ax.axhline(0,color=".5",lw=1);ax.set(xscale="log",xlabel=r"$\ell$",ylabel="Change in sum [%]",title=title);ax.legend()
    axes[0].plot([],[],ls="--",color=".3",label="Integrated Y matched");axes[0].legend()
    save(fig,"integrated_Y_matched_control")
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for k,c in enumerate([12,13,14,15]):
        sharp=total[c]
        axes[0].plot(ELL,100*((taper_one[k]+taper_two[k])/sharp-1),color=colors[k],label=f"Cut {k+1}R")
        axes[1].plot(ELL,100*((taper_matched_one[k]+taper_matched_two[k])/sharp-1),color=colors[k])
    for ax,title in zip(axes,("Taper: changed flux", "Taper: same integrated Y")):
        ax.axhline(0,color=".5",lw=1);ax.set(xscale="log",xlabel=r"$\ell$",ylabel="Change from sharp cut [%]",title=title)
    axes[0].legend();save(fig,"sharp_edge_taper_control")
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    for k,c in enumerate([3,7,11,15]):ax.plot(ELL,100*two[c]/total[c],color=colors[k],label=f"Fit{k+1}R, cut4R")
    ax.set(xscale="log",xlabel=r"$\ell$",ylabel="Linear 2-halo fraction [%]");ax.legend();save(fig,"linear_twohalo_fraction")


if __name__ == "__main__":
    run()
