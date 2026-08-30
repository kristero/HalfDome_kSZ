#!/usr/bin/env python3
"""Standalone Fisher sensitivity forecast using SO N_ell curves.

The calculation uses the same masked, 2ell+1 weighted, binned pseudo-D_ell
statistic as the prepared SBI dataset. It has no dependency on sbi, arviz,
numba, scipy, or getdist.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


DEFAULT_PROJECT = Path("/home/kristero10/HalfDome_kSZ")
DEFAULT_FISHER_ROOT = Path("/lustre/work/kristero10/adrian_fisher_baseline_deproj0")
DEFAULT_PREPARED = (
    DEFAULT_PROJECT / "SBI_analysis/data_for_cluster"
    / "adrian_so_sbi_cases_ell80_7979_dataset_row_sobolrow"
    / "so_masked_baseline_noise_cross_deproj0_ell80_7979_sbi_run.npz"
)
DEFAULT_NOISE_DIR = Path("/home/kristero10/tSZ_data/SO_noise")
DEFAULT_OUTPUT = Path("/lustre/work/kristero10/adrian_fisher_nell_sensitivity_deproj0")

BAT12 = {
    "P0": 18.1, "xc": 0.497, "beta": 4.35,
    "alpha_m_P0": 0.154, "alpha_m_xc": -0.00865, "alpha_m_beta": 0.0393,
    "alpha_z_P0": -0.758, "alpha_z_xc": 0.731, "alpha_z_beta": 0.415,
}
LATEX = {
    "P0": r"$P_0$", "xc": r"$x_{\rm c}$", "beta": r"$\beta$",
    "alpha_m_P0": r"$\alpha_{m,P_0}$",
    "alpha_m_xc": r"$\alpha_{m,x_{\rm c}}$",
    "alpha_m_beta": r"$\alpha_{m,\beta}$",
    "alpha_z_P0": r"$\alpha_{z,P_0}$",
    "alpha_z_xc": r"$\alpha_{z,x_{\rm c}}$",
    "alpha_z_beta": r"$\alpha_{z,\beta}$",
}
COLORS = {"baseline": "#d62728", "goal": "#1f77b4"}
SCOPE_LABELS = {
    "conditional_noise": "fixed signal + SO noise",
    "gaussian_total": "Gaussian signal + SO noise",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Fisher sensitivities from Battaglia12 derivatives and SO N_ell."
    )
    parser.add_argument("--prepared-dataset", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--fisher-root", type=Path, default=DEFAULT_FISHER_ROOT)
    parser.add_argument("--derivatives", type=Path)
    parser.add_argument("--fiducial-clean-cl", type=Path)
    parser.add_argument(
        "--baseline-noise", type=Path,
        default=DEFAULT_NOISE_DIR / "SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt",
    )
    parser.add_argument(
        "--goal-noise", type=Path,
        default=DEFAULT_NOISE_DIR / "SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt",
    )
    parser.add_argument(
        "--noise-cases", nargs="+", choices=("baseline", "goal"),
        default=("baseline", "goal"),
    )
    parser.add_argument(
        "--covariance-scopes", nargs="+",
        choices=("conditional_noise", "gaussian_total"),
        default=("conditional_noise", "gaussian_total"),
    )
    parser.add_argument("--deprojection", type=int, default=0)
    parser.add_argument("--noise-is-dl", action="store_true")
    parser.add_argument(
        "--split-noise-factor", type=float, default=1.0,
        help="Per-split N_ell/input N_ell. Use 2 only if the curve is coadd depth.",
    )
    parser.add_argument("--mask-fsky-effective", type=float, default=0.4)
    parser.add_argument("--covariance-rcond", type=float, default=1.0e-12)
    parser.add_argument("--fisher-rcond", type=float, default=1.0e-8)
    parser.add_argument(
        "--prior-sigma-fraction", type=float, default=1.0 / math.sqrt(12.0),
        help="Gaussian prior sigma / uniform-prior width.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def require_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(label, path))
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_bin_operator(ell, bin_min, bin_max):
    operator = np.zeros((len(bin_min), len(ell)), dtype=float)
    centers = np.zeros(len(bin_min), dtype=float)
    mode_weights = 2.0 * ell.astype(float) + 1.0
    dell_factor = ell.astype(float) * (ell.astype(float) + 1.0) / (2.0 * np.pi)
    assigned = np.zeros(len(ell), dtype=bool)
    for index, (low, high) in enumerate(zip(bin_min, bin_max)):
        selected = (ell >= low) & (ell <= high)
        if not np.any(selected):
            raise ValueError("Empty ell bin [{}, {}]".format(low, high))
        if np.any(assigned & selected):
            raise ValueError("Prepared ell bins overlap")
        assigned |= selected
        weights = mode_weights[selected]
        weights = weights / weights.sum()
        operator[index, selected] = weights * dell_factor[selected]
        centers[index] = np.sum(weights * ell[selected])
    return operator, centers


def load_contract(path):
    path = require_file(path, "prepared SBI dataset")
    with np.load(path, allow_pickle=True) as data:
        needed = ("prior_low", "prior_high", "param_names")
        missing = [key for key in needed if key not in data]
        if missing:
            raise KeyError("Prepared dataset is missing {}".format(missing))
        prior_low = np.asarray(data["prior_low"], dtype=float)
        prior_high = np.asarray(data["prior_high"], dtype=float)
        param_names = [str(value) for value in data["param_names"].tolist()]
        ell = (
            np.asarray(data["ell_unbinned"], dtype=int)
            if "ell_unbinned" in data else np.arange(80, 7980, dtype=int)
        )
        if "bin_ell_min" in data and "bin_ell_max" in data:
            bin_min = np.asarray(data["bin_ell_min"], dtype=int)
            bin_max = np.asarray(data["bin_ell_max"], dtype=int)
        else:
            edges = np.r_[np.arange(80, 7881, 200), 7979]
            bin_min = edges[:-1]
            bin_max = edges[1:].copy()
            bin_max[:-1] -= 1
        saved_centers = (
            np.asarray(data["ell_binned"], dtype=float)
            if "ell_binned" in data else None
        )
    if prior_low.ndim != 1 or prior_low.shape != prior_high.shape:
        raise ValueError("Prior bounds must be matching 1D vectors")
    if len(param_names) != len(prior_low) or np.any(prior_high <= prior_low):
        raise ValueError("Invalid prepared parameter/prior contract")
    if ell.ndim != 1 or np.any(np.diff(ell) <= 0):
        raise ValueError("ell_unbinned must be strictly increasing")
    operator, centers = build_bin_operator(ell, bin_min, bin_max)
    if saved_centers is not None:
        if saved_centers.shape != centers.shape:
            raise ValueError("ell_binned has the wrong shape")
        if np.max(np.abs(saved_centers - centers)) > 1.0e-4:
            raise ValueError("ell_binned does not match 2ell+1 weighting")
    return {
        "path": path, "prior_low": prior_low, "prior_high": prior_high,
        "prior_width": prior_high - prior_low, "param_names": param_names,
        "ell": ell, "ell_binned": centers, "bin_operator": operator,
    }


def derivative_paths(args):
    analysis = args.fisher_root.expanduser().resolve() / "analysis"
    richardson = args.derivatives or analysis / "derivatives_richardson.npy"
    paths = {"richardson": require_file(richardson, "Richardson derivatives")}
    for name, filename in (
        ("small", "derivatives_small_step.npy"),
        ("large", "derivatives_large_step.npy"),
    ):
        candidate = analysis / filename
        if candidate.is_file():
            paths[name] = candidate.resolve()
    return paths


def load_derivatives(paths, n_params, n_bins):
    output = {}
    for name, path in paths.items():
        values = np.asarray(np.load(path), dtype=float)
        if values.shape == (n_bins, n_params):
            values = values.T
        if values.shape != (n_params, n_bins):
            raise ValueError(
                "{} derivatives have shape {}, expected ({}, {})".format(
                    name, values.shape, n_params, n_bins
                )
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("{} derivatives contain non-finite values".format(name))
        output[name] = values
    return output


def discover_fiducial(args):
    if args.fiducial_clean_cl:
        return require_file(args.fiducial_clean_cl, "fiducial clean C_ell")
    root = args.fisher_root.expanduser().resolve()
    matches = set()
    for pattern in (
        "variations/row*_fiducial/**/*masked_no_noise_cl*.npy",
        "variations/row*_fiducial/*masked_no_noise_cl*.npy",
    ):
        matches.update(path.resolve() for path in root.glob(pattern) if path.is_file())
    matches = sorted(matches)
    if len(matches) != 1:
        raise FileNotFoundError(
            "Found {} fiducial masked no-noise C_ell files under {}. "
            "Pass --fiducial-clean-cl explicitly. Matches: {}".format(
                len(matches), root, matches
            )
        )
    return matches[0]


def align_spectrum(values, ell, label):
    values = np.asarray(values, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError("{} must be 1D, got {}".format(label, values.shape))
    if len(values) == len(ell):
        aligned = values.copy()
    elif len(values) > int(ell[-1]):
        aligned = values[ell]
    else:
        raise ValueError(
            "{} length {} does not cover ell={}..{}".format(
                label, len(values), int(ell[0]), int(ell[-1])
            )
        )
    if not np.all(np.isfinite(aligned)):
        raise ValueError("{} contains non-finite values".format(label))
    return aligned


def load_noise(path, ell, deprojection, noise_is_dl, split_factor):
    path = require_file(path, "SO N_ell")
    table = np.loadtxt(path, comments="#", dtype=float)
    if table.ndim == 1:
        table = table[None, :]
    value_column = 1 + deprojection
    if deprojection < 0 or value_column >= table.shape[1]:
        raise ValueError(
            "deprojection {} requires zero-based column {}, but {} has {} columns".format(
                deprojection, value_column, path, table.shape[1]
            )
        )
    file_ell_float = table[:, 0]
    file_ell = np.rint(file_ell_float).astype(int)
    if not np.allclose(file_ell_float, file_ell, atol=1.0e-8, rtol=0.0):
        raise ValueError("First SO noise column must be integer ell")
    order = np.argsort(file_ell)
    file_ell = file_ell[order]
    values = table[order, value_column]
    if np.any(np.diff(file_ell) == 0):
        raise ValueError("SO noise file has duplicate ell rows")
    positions = np.searchsorted(file_ell, ell)
    valid = positions < len(file_ell)
    matched = np.zeros(len(ell), dtype=bool)
    matched[valid] = file_ell[positions[valid]] == ell[valid]
    if not np.all(matched):
        raise ValueError(
            "SO noise file lacks requested multipoles; first missing: {}".format(
                ell[~matched][:10].tolist()
            )
        )
    noise = values[positions]
    if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
        raise ValueError("SO noise values must be finite and non-negative")
    if noise_is_dl:
        noise = noise * 2.0 * np.pi / (ell * (ell + 1.0))
    return noise * split_factor, path, value_column


def build_covariance(signal_pseudo, noise_full, ell, operator, fsky, scope):
    # Both signal and noise pseudo spectra are approximated as fsky times
    # deconvolved spectra. The output therefore matches the raw masked
    # statistic used for the saved derivatives.
    noise_pseudo = fsky * noise_full
    modes = (2.0 * ell.astype(float) + 1.0) * fsky
    if scope == "conditional_noise":
        variance = (2.0 * signal_pseudo * noise_pseudo + noise_pseudo ** 2) / modes
    elif scope == "gaussian_total":
        variance = (
            (signal_pseudo + noise_pseudo) ** 2 + signal_pseudo ** 2
        ) / modes
    else:
        raise ValueError("Unknown covariance scope {}".format(scope))
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("{} covariance has invalid variances".format(scope))
    covariance = (operator * variance[None, :]).dot(operator.T)
    return 0.5 * (covariance + covariance.T), variance, noise_pseudo


def symmetric_inverse(matrix, rcond):
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    largest = float(np.max(values))
    if largest <= 0.0:
        raise ValueError("Matrix has no positive eigenvalues")
    threshold = largest * rcond
    keep = values > threshold
    inverse = (vectors[:, keep] / values[keep]).dot(vectors[:, keep].T)
    return inverse, values, keep, threshold


def fisher_diagnostics(derivatives, covariance, prior_width, args):
    precision, cov_eig, cov_keep, cov_threshold = symmetric_inverse(
        covariance, args.covariance_rcond
    )
    derivatives_q = derivatives * prior_width[:, None]
    fisher = derivatives_q.dot(precision).dot(derivatives_q.T)
    fisher = 0.5 * (fisher + fisher.T)
    values, vectors = np.linalg.eigh(fisher)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    threshold = max(float(values[0]), 0.0) * args.fisher_rcond
    keep = values > threshold
    rank = int(np.count_nonzero(keep))
    diag = np.diag(fisher)
    conditional = np.full(len(diag), np.nan)
    conditional[diag > 0.0] = 1.0 / np.sqrt(diag[diag > 0.0])
    pseudocov = (
        (vectors[:, keep] / values[keep]).dot(vectors[:, keep].T)
        if rank else np.zeros_like(fisher)
    )
    projector = vectors[:, keep].dot(vectors[:, keep].T) if rank else np.zeros_like(fisher)
    estimable_fraction = np.clip(np.diag(projector), 0.0, 1.0)
    marginalized = np.full(len(diag), np.nan)
    estimable = estimable_fraction > 1.0 - 1.0e-8
    marginalized[estimable] = np.sqrt(np.maximum(np.diag(pseudocov)[estimable], 0.0))
    prior_covariance = None
    prior_sigma = np.full(len(diag), np.nan)
    fisher_plus_prior = None
    if args.prior_sigma_fraction > 0.0:
        fisher_plus_prior = fisher + np.eye(len(diag)) / args.prior_sigma_fraction ** 2
        total_values, total_vectors = np.linalg.eigh(fisher_plus_prior)
        prior_covariance = (total_vectors / total_values).dot(total_vectors.T)
        prior_covariance = 0.5 * (prior_covariance + prior_covariance.T)
        prior_sigma = np.sqrt(np.maximum(np.diag(prior_covariance), 0.0))
    return {
        "precision": precision, "cov_eigenvalues": cov_eig,
        "cov_keep": cov_keep, "cov_threshold": cov_threshold,
        "fisher": fisher, "eigenvalues": values, "eigenvectors": vectors,
        "keep": keep, "threshold": threshold, "rank": rank,
        "conditional_sigma_q": conditional, "pseudocovariance_q": pseudocov,
        "estimable_fraction": estimable_fraction,
        "marginalized_sigma_q": marginalized,
        "prior_covariance_q": prior_covariance, "prior_sigma_q": prior_sigma,
        "fisher_plus_prior": fisher_plus_prior,
    }


def covariance_correlation(covariance):
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(sigma, sigma)
    result = np.divide(
        covariance, denominator, out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    np.fill_diagonal(result, 1.0)
    return result


def save_result(directory, contract, fiducial_theta, covariance, variance,
                noise_full, noise_pseudo, diagnostics, case, scope, args):
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "covariance_binned_dell.npy": covariance,
        "correlation_binned_dell.npy": covariance_correlation(covariance),
        "variance_unbinned_pseudo_cl.npy": variance,
        "noise_fullsky_split_cl.npy": noise_full,
        "noise_masked_pseudo_cl.npy": noise_pseudo,
        "precision_binned_dell.npy": diagnostics["precision"],
        "fisher_normalized_by_prior_width.npy": diagnostics["fisher"],
        "fisher_eigenvalues.npy": diagnostics["eigenvalues"],
        "fisher_eigenvectors.npy": diagnostics["eigenvectors"],
        "fisher_identified_modes.npy": diagnostics["keep"],
        "fisher_pseudocovariance_normalized.npy": diagnostics["pseudocovariance_q"],
        "fisher_estimable_fraction.npy": diagnostics["estimable_fraction"],
    }
    if diagnostics["prior_covariance_q"] is not None:
        arrays["fisher_prior_regularized_covariance_normalized.npy"] = (
            diagnostics["prior_covariance_q"]
        )
        arrays["fisher_plus_prior_normalized.npy"] = diagnostics["fisher_plus_prior"]
    for filename, values in arrays.items():
        np.save(directory / filename, values)

    parameter_rows = []
    for index, name in enumerate(contract["param_names"]):
        width = contract["prior_width"][index]
        data_sigma = diagnostics["marginalized_sigma_q"][index]
        prior_sigma = diagnostics["prior_sigma_q"][index]
        parameter_rows.append({
            "parameter": name,
            "fiducial": "{:.16g}".format(fiducial_theta[index]),
            "prior_low": "{:.16g}".format(contract["prior_low"][index]),
            "prior_high": "{:.16g}".format(contract["prior_high"][index]),
            "prior_width": "{:.16g}".format(width),
            "conditional_sigma": "{:.16g}".format(
                diagnostics["conditional_sigma_q"][index] * width
            ),
            "conditional_sigma_over_prior_width": "{:.16g}".format(
                diagnostics["conditional_sigma_q"][index]
            ),
            "data_marginalized_sigma": (
                "{:.16g}".format(data_sigma * width)
                if np.isfinite(data_sigma) else ""
            ),
            "data_marginalized_sigma_over_prior_width": (
                "{:.16g}".format(data_sigma) if np.isfinite(data_sigma) else ""
            ),
            "prior_regularized_sigma": (
                "{:.16g}".format(prior_sigma * width)
                if np.isfinite(prior_sigma) else ""
            ),
            "prior_regularized_sigma_over_prior_width": (
                "{:.16g}".format(prior_sigma)
                if np.isfinite(prior_sigma) else ""
            ),
            "estimable_fraction": "{:.16g}".format(
                diagnostics["estimable_fraction"][index]
            ),
        })
    write_csv(directory / "parameter_sensitivities.csv", parameter_rows)

    mode_rows = []
    leading = diagnostics["eigenvalues"][0]
    for mode, eigenvalue in enumerate(diagnostics["eigenvalues"]):
        vector = diagnostics["eigenvectors"][:, mode]
        component_indices = np.argsort(np.abs(vector))[::-1][:3]
        components = "; ".join(
            "{}:{:+.4f}".format(contract["param_names"][idx], vector[idx])
            for idx in component_indices
        )
        mode_rows.append({
            "mode": mode + 1,
            "eigenvalue": "{:.16g}".format(eigenvalue),
            "relative_eigenvalue": (
                "{:.16g}".format(eigenvalue / leading) if leading > 0.0 else ""
            ),
            "identified": bool(diagnostics["keep"][mode]),
            "sigma_in_prior_width_coordinates": (
                "{:.16g}".format(1.0 / np.sqrt(eigenvalue))
                if eigenvalue > 0.0 else ""
            ),
            "largest_components": components,
        })
    write_csv(directory / "fisher_modes.csv", mode_rows)

    summary = {
        "case": case,
        "covariance_scope": scope,
        "covariance_scope_description": SCOPE_LABELS[scope],
        "statistic": "masked pseudo-D_ell, 2ell+1 weighted bins",
        "covariance_model": "narrow-kernel fsky Gaussian split-cross approximation",
        "fsky_effective": args.mask_fsky_effective,
        "deprojection": args.deprojection,
        "so_noise_zero_based_column": 1 + args.deprojection,
        "so_noise_one_based_column": 2 + args.deprojection,
        "input_noise_is_dl": bool(args.noise_is_dl),
        "split_noise_factor": args.split_noise_factor,
        "n_parameters": len(contract["param_names"]),
        "n_bins": int(covariance.shape[0]),
        "covariance_rank": int(np.count_nonzero(diagnostics["cov_keep"])),
        "fisher_rank": diagnostics["rank"],
        "fisher_rcond": args.fisher_rcond,
        "fisher_threshold": diagnostics["threshold"],
        "all_fisher_modes_identified": bool(np.all(diagnostics["keep"])),
        "prior_regularization": (
            "moment-matched Gaussian approximation to bounded uniform priors"
            if args.prior_sigma_fraction > 0.0 else "disabled"
        ),
        "prior_sigma_fraction": args.prior_sigma_fraction,
        "data_marginalized_blanks_mean_not_estimable": True,
        "limitations": [
            "No mask coupling matrix was supplied; this uses an fsky approximation.",
            "Gaussian-total omits connected tSZ trispectrum and super-sample covariance.",
            "Conditional-noise conditions on the fixed Battaglia12 signal.",
            "Finite prior-regularized errors do not establish data identifiability.",
        ],
    }
    with (directory / "fisher_sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return parameter_rows


def configure_plotting():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def plot_signal_noise(output_dir, ell, signal_pseudo, noise_by_case, fsky):
    factor = ell.astype(float) * (ell.astype(float) + 1.0) / (2.0 * np.pi)
    fig, ax = plt.subplots(figsize=(9.0 / 2.54, 6.5 / 2.54))
    ax.loglog(
        ell, factor * signal_pseudo / fsky, color="black", lw=1.1,
        label=r"Battaglia12 signal ($f_{\rm sky}$ approximation)",
    )
    for case, noise in noise_by_case.items():
        ax.loglog(
            ell, factor * noise, color=COLORS[case], lw=1.0,
            label="{} SO $N_\\ell$".format(case),
        )
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "battaglia12_signal_and_so_noise.jpg", dpi=300)
    plt.close(fig)


def plot_sensitivities(output_dir, contract, results):
    labels = [LATEX.get(name, name) for name in contract["param_names"]]
    x = np.arange(len(labels), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(results))
    fig, ax = plt.subplots(figsize=(18.0 / 2.54, 7.5 / 2.54))
    for offset, result in zip(offsets, results):
        diag = result["diagnostics"]
        color = COLORS[result["case"]]
        marker = "o" if result["scope"] == "conditional_noise" else "s"
        line_style = ":" if result["scope"] == "conditional_noise" else "--"
        ax.plot(
            x + offset, diag["conditional_sigma_q"], marker=marker, ms=3.3,
            lw=0.8, ls=line_style, color=color, alpha=0.65,
            label="{}: {} (others fixed)".format(
                result["case"], SCOPE_LABELS[result["scope"]]
            ),
        )
        if np.any(np.isfinite(diag["prior_sigma_q"])):
            ax.plot(
                x + offset, diag["prior_sigma_q"], marker=marker, ms=4.0,
                lw=1.2, color=color,
                label="{}: {} (marginal + prior)".format(
                    result["case"], SCOPE_LABELS[result["scope"]]
                ),
            )
    ax.axhline(
        1.0 / math.sqrt(12.0), color="0.2", lw=0.8, ls="--",
        label="uniform-prior standard deviation",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"Forecast $\sigma(\theta)/\Delta\theta_{\rm prior}$")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=6.2)
    fig.tight_layout()
    fig.savefig(output_dir / "fisher_parameter_sensitivity_summary.jpg", dpi=300)
    plt.close(fig)


def plot_eigenvalues(output_dir, results):
    fig, ax = plt.subplots(figsize=(9.0 / 2.54, 6.5 / 2.54))
    for result in results:
        values = result["diagnostics"]["eigenvalues"]
        ax.semilogy(
            np.arange(1, len(values) + 1),
            np.maximum(values, np.finfo(float).tiny),
            marker="o", ms=3.0, lw=1.0, color=COLORS[result["case"]],
            ls="-" if result["scope"] == "conditional_noise" else "--",
            label="{}: {}".format(result["case"], SCOPE_LABELS[result["scope"]]),
        )
    ax.set_xlabel("Fisher eigenmode")
    ax.set_ylabel("Eigenvalue in prior-width coordinates")
    ax.set_xticks(np.arange(1, len(results[0]["diagnostics"]["eigenvalues"]) + 1))
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fisher_eigenvalue_spectrum.jpg", dpi=300)
    plt.close(fig)


def plot_modes(output_dir, contract, results):
    labels = [LATEX.get(name, name) for name in contract["param_names"]]
    for result in results:
        vectors = result["diagnostics"]["eigenvectors"].T
        keep = result["diagnostics"]["keep"]
        fig, ax = plt.subplots(figsize=(9.0 / 2.54, 7.2 / 2.54))
        image = ax.imshow(
            vectors, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto"
        )
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(vectors)))
        ax.set_yticklabels([
            "{}{}".format(index + 1, "" if keep[index] else " (null)")
            for index in range(len(vectors))
        ])
        ax.set_ylabel("Fisher eigenmode")
        ax.set_title("{}: {}".format(result["case"], SCOPE_LABELS[result["scope"]]))
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Normalized mode coefficient")
        fig.tight_layout()
        fig.savefig(
            output_dir / "fisher_modes_{}_{}.jpg".format(
                result["case"], result["scope"]
            ),
            dpi=300,
        )
        plt.close(fig)


def add_ellipse(ax, center, covariance, color, linestyle, label):
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0.0):
        return
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    scale = math.sqrt(2.30)
    ax.add_patch(Ellipse(
        center,
        width=2.0 * scale * math.sqrt(values[0]),
        height=2.0 * scale * math.sqrt(values[1]),
        angle=angle, fill=False, edgecolor=color, linewidth=1.2,
        linestyle=linestyle, label=label,
    ))


def fixed_pair_covariance(fisher, indices):
    block = fisher[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh(0.5 * (block + block.T))
    if np.any(values <= 0.0):
        return None
    return (vectors / values).dot(vectors.T)


def plot_p0_beta(output_dir, contract, fiducial_theta, results):
    if "P0" not in contract["param_names"] or "beta" not in contract["param_names"]:
        return
    indices = [
        contract["param_names"].index("P0"),
        contract["param_names"].index("beta"),
    ]
    center = fiducial_theta[indices]
    widths = contract["prior_width"][indices]
    fig, ax = plt.subplots(figsize=(9.0 / 2.54, 7.5 / 2.54))
    for result in results:
        diag = result["diagnostics"]
        color = COLORS[result["case"]]
        prior_cov = diag["prior_covariance_q"]
        if prior_cov is not None:
            pair_cov = prior_cov[np.ix_(indices, indices)] * np.outer(widths, widths)
            add_ellipse(
                ax, center, pair_cov, color,
                "-" if result["scope"] == "conditional_noise" else "--",
                "{}: {} (marginal + prior)".format(
                    result["case"], SCOPE_LABELS[result["scope"]]
                ),
            )
        if result["scope"] == "conditional_noise":
            fixed_cov = fixed_pair_covariance(diag["fisher"], indices)
            if fixed_cov is not None:
                add_ellipse(
                    ax, center, fixed_cov * np.outer(widths, widths),
                    color, ":", "{}: other seven fixed".format(result["case"]),
                )
    ax.axvline(center[0], color="black", lw=0.7, ls="--")
    ax.axhline(center[1], color="black", lw=0.7, ls="--")
    ax.scatter(center[0], center[1], color="black", s=10, zorder=5)
    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel(r"$P_0$")
    ax.set_ylabel(r"$\beta$")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, fontsize=6.0)
    fig.tight_layout()
    fig.savefig(output_dir / "fisher_p0_beta_sensitivity.jpg", dpi=300)
    plt.close(fig)


def plot_covariance_correlations(output_dir, results):
    for result in results:
        fig, ax = plt.subplots(figsize=(7.5 / 2.54, 6.5 / 2.54))
        image = ax.imshow(
            covariance_correlation(result["covariance"]),
            cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        )
        ax.set_xlabel(r"$D_\ell$ bin")
        ax.set_ylabel(r"$D_\ell$ bin")
        ax.set_title("{}: {}".format(result["case"], SCOPE_LABELS[result["scope"]]))
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Correlation")
        fig.tight_layout()
        fig.savefig(
            output_dir / "covariance_correlation_{}_{}.jpg".format(
                result["case"], result["scope"]
            ),
            dpi=300,
        )
        plt.close(fig)


def main():
    args = parse_args()
    if not (0.0 < args.mask_fsky_effective <= 1.0):
        raise ValueError("--mask-fsky-effective must be in (0, 1]")
    if args.split_noise_factor <= 0.0:
        raise ValueError("--split-noise-factor must be positive")
    if args.covariance_rcond <= 0.0 or args.fisher_rcond <= 0.0:
        raise ValueError("rcond values must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = load_contract(args.prepared_dataset)
    paths = derivative_paths(args)
    derivatives = load_derivatives(
        paths, len(contract["param_names"]), len(contract["ell_binned"])
    )
    fiducial_path = discover_fiducial(args)
    fiducial_cl = align_spectrum(
        np.load(fiducial_path), contract["ell"], "fiducial masked no-noise C_ell"
    )
    if np.any(fiducial_cl < 0.0):
        raise ValueError("Fiducial auto C_ell contains negative values")
    fiducial_dell = contract["bin_operator"].dot(fiducial_cl)

    saved_fiducial = (
        args.fisher_root.expanduser().resolve()
        / "analysis/fiducial_battaglia12_dell.npy"
    )
    fiducial_difference = None
    if saved_fiducial.is_file():
        expected = np.asarray(np.load(saved_fiducial), dtype=float).squeeze()
        if expected.shape != fiducial_dell.shape:
            raise ValueError("Saved fiducial D_ell has an inconsistent shape")
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        fiducial_difference = float(np.max(np.abs(expected - fiducial_dell)) / scale)
        if fiducial_difference > 1.0e-5:
            raise ValueError(
                "Reconstructed fiducial D_ell disagrees with derivative analysis "
                "(relative max difference {:.6g})".format(fiducial_difference)
            )

    try:
        fiducial_theta = np.asarray(
            [BAT12[name] for name in contract["param_names"]], dtype=float
        )
    except KeyError as error:
        raise KeyError("No Battaglia12 value for {}".format(error))
    if np.any(fiducial_theta < contract["prior_low"]) or np.any(
        fiducial_theta > contract["prior_high"]
    ):
        raise ValueError("Battaglia12 fiducial lies outside the prepared prior")

    requested_paths = {"baseline": args.baseline_noise, "goal": args.goal_noise}
    noise_by_case = {}
    noise_metadata = {}
    for case in args.noise_cases:
        noise, path, column = load_noise(
            requested_paths[case], contract["ell"], args.deprojection,
            args.noise_is_dl, args.split_noise_factor,
        )
        noise_by_case[case] = noise
        noise_metadata[case] = {
            "path": str(path), "sha256": sha256_file(path),
            "zero_based_value_column": column,
            "one_based_value_column": column + 1,
        }

    results = []
    index_rows = []
    all_parameter_rows = []
    for case in args.noise_cases:
        for scope in args.covariance_scopes:
            covariance, variance, noise_pseudo = build_covariance(
                fiducial_cl, noise_by_case[case], contract["ell"],
                contract["bin_operator"], args.mask_fsky_effective, scope,
            )
            diagnostics = fisher_diagnostics(
                derivatives["richardson"], covariance,
                contract["prior_width"], args,
            )
            result_dir = output_dir / case / scope
            parameter_rows = save_result(
                result_dir, contract, fiducial_theta, covariance, variance,
                noise_by_case[case], noise_pseudo, diagnostics,
                case, scope, args,
            )
            for row in parameter_rows:
                combined = {"case": case, "covariance_scope": scope}
                combined.update(row)
                all_parameter_rows.append(combined)
            results.append({
                "case": case, "scope": scope, "covariance": covariance,
                "diagnostics": diagnostics, "output_dir": result_dir,
            })
            index_rows.append({
                "case": case, "covariance_scope": scope,
                "fisher_rank": diagnostics["rank"],
                "n_parameters": len(contract["param_names"]),
                "all_modes_identified": bool(np.all(diagnostics["keep"])),
                "output_dir": str(result_dir),
            })
    write_csv(output_dir / "fisher_sensitivity_index.csv", index_rows)
    write_csv(
        output_dir / "all_parameter_sensitivities.csv",
        all_parameter_rows,
    )

    stability_rows = []
    reference = derivatives["richardson"]
    reference_norm = np.linalg.norm(reference, axis=1)
    for name, values in derivatives.items():
        if name == "richardson":
            continue
        relative = np.divide(
            np.linalg.norm(values - reference, axis=1), reference_norm,
            out=np.full(len(reference_norm), np.nan), where=reference_norm > 0.0,
        )
        for parameter, value in zip(contract["param_names"], relative):
            stability_rows.append({
                "derivative_set": name, "parameter": parameter,
                "relative_l2_difference_from_richardson": "{:.16g}".format(value),
            })
    write_csv(output_dir / "derivative_stability.csv", stability_rows)

    provenance = {
        "prepared_dataset": str(contract["path"]),
        "prepared_dataset_sha256": sha256_file(contract["path"]),
        "fiducial_clean_cl": str(fiducial_path),
        "fiducial_clean_cl_sha256": sha256_file(fiducial_path),
        "derivatives": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "noise": noise_metadata,
        "fiducial_binning_relative_max_difference": fiducial_difference,
        "ell_min": int(contract["ell"][0]),
        "ell_max": int(contract["ell"][-1]),
        "n_ell": int(len(contract["ell"])),
        "n_bins": int(len(contract["ell_binned"])),
        "bin_weighting": "2ell_plus_1",
        "statistic": "linear masked pseudo-D_ell",
    }
    with (output_dir / "input_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    if not args.no_plots:
        configure_plotting()
        plot_signal_noise(
            output_dir, contract["ell"], fiducial_cl,
            noise_by_case, args.mask_fsky_effective,
        )
        plot_sensitivities(output_dir, contract, results)
        plot_eigenvalues(output_dir, results)
        plot_modes(output_dir, contract, results)
        plot_p0_beta(output_dir, contract, fiducial_theta, results)
        plot_covariance_correlations(output_dir, results)

    completion = {
        "complete": True, "output_dir": str(output_dir),
        "cases": list(args.noise_cases),
        "covariance_scopes": list(args.covariance_scopes),
        "fisher_ranks": {
            "{}_{}".format(item["case"], item["scope"]): item["diagnostics"]["rank"]
            for item in results
        },
        "all_fisher_modes_identified": {
            "{}_{}".format(item["case"], item["scope"]): bool(
                np.all(item["diagnostics"]["keep"])
            )
            for item in results
        },
    }
    with (output_dir / "fisher_sensitivity_complete.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)

    print("Completed SO N_ell Fisher sensitivity analysis")
    print("Output:", output_dir)
    for item in results:
        print(
            "  {} / {}: Fisher rank {}/{}".format(
                item["case"], item["scope"], item["diagnostics"]["rank"],
                len(contract["param_names"]),
            )
        )
        for parameter in ("P0", "beta"):
            if parameter not in contract["param_names"]:
                continue
            index = contract["param_names"].index(parameter)
            print(
                "    {}: sigma/prior={:.6g} (others fixed), "
                "{:.6g} (marginal + moment-matched prior)".format(
                    parameter,
                    item["diagnostics"]["conditional_sigma_q"][index],
                    item["diagnostics"]["prior_sigma_q"][index],
                )
            )
    print(
        "Finite prior-regularized errors are not evidence that null modes "
        "are data-constrained."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
