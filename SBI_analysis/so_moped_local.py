"""Matched local SO simulation and inference helpers for the MOPED notebook.

No retraining, refitting of MOPED, extra beam, clipping, or ad hoc noise model.
"""
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess

import numpy as np

from export_so_moped_bundle import sha256
from prepare_validate_battaglia12_sbi_observation import (
    align_to_ell, discover_profile, make_cl_to_dl_matrix, read_profile,
)
from so_sbi_compression import PARAM_NAMES, project, save_npz, write_json


JULIA_PARAMETERS = (
    "battaglia_P0_amp", "battaglia_x_c_amp", "battaglia_beta_amp",
    "battaglia_P0_alpha_m", "battaglia_x_c_alpha_m", "battaglia_beta_alpha_m",
    "battaglia_P0_alpha_z", "battaglia_x_c_alpha_z", "battaglia_beta_alpha_z",
)


def load_bundle(path):
    path = Path(path)
    manifest = json.loads((path / "bundle_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError(f"Missing or changed bundle member: {name}")
    with np.load(path / "observation_contract.npz", allow_pickle=False) as z:
        contract = dict(z)
    with np.load(path / "moped_transform.npz", allow_pickle=False) as z:
        transform = dict(z)
    if tuple(map(str, contract["param_names"])) != PARAM_NAMES:
        raise ValueError("Unexpected inference parameter order")
    if str(contract["experiment_id"].item()) != manifest["experiment_id"]:
        raise ValueError("Mismatched experiment IDs")
    np.testing.assert_array_equal(project(contract["reference_x"], transform),
                                  contract["reference_context"])
    return contract, transform, manifest


def truth_vector(parameters, contract):
    if set(parameters) != set(PARAM_NAMES):
        raise ValueError(f"Supply exactly these nine parameters: {PARAM_NAMES}")
    theta = np.array([parameters[k] for k in PARAM_NAMES], dtype=np.float64)
    if not np.isfinite(theta).all() or np.any((theta < contract["low"]) | (theta > contract["high"])):
        raise ValueError("Chosen parameters must lie inside the saved training prior")
    return theta


def simulator_command(repo, output, catalogue, noise, theta, julia="julia", threads=8,
                      mask_seed=12345, noise_seed=3000001, julia_project=None, julia_depot=None):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    if np.asarray(theta).shape != (9,) or not np.isfinite(theta).all():
        raise ValueError("Expected nine finite pressure parameters")
    if int(mask_seed) != 12345:
        raise ValueError("This model's simulator contract requires mask seed 12345")
    if int(threads) < 1 or int(noise_seed) < 0:
        raise ValueError("Invalid threads or noise seed")
    settings = dict(catalog_source="halfdome", simulation_name="halfdome_lightcone_100",
        batching_mode="full", halfdome_path=str(Path(catalogue).resolve()),
        apply_mass_cut="true", mass_min=1e12,
        cosmo_h=0.68, cosmo_omegab=0.049, cosmo_omegac=0.261,
        sobol_row=0, sobol_csv_path="", cleanup_nonpositive_profile_values="true",
        output_dir=str(output / "raw"), cache_dir=str(output / "cache"),
        baseline_noise_path=str(Path(noise).resolve()),
        goal_noise_path=str(Path(noise).with_name("SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt").resolve()),
        nside=4096, cl_lmax=7979, so_noise_lmax=7979, cl_niter=0,
        so_noise_deprojections=0, so_noise_is_dl="false",
        apply_gaussian_beam="true", gaussian_beam_fwhm_arcmin=2.0,
        mask_fsky=0.4, mask_apodization_arcmin=60.0,
        seed=int(mask_seed), mask_seed=int(mask_seed), noise_seed=int(noise_seed),
        model_exists="false", reuse_existing_cache="false", cache_wait_seconds=0,
        interpolator_pad=256, interpolator_logM_max=15.7,
        save_cl="true", save_no_noise_cl="true", save_baseline_noise_cross_cl="true",
        save_goal_noise_cross_cl="false", save_unmasked_no_noise_cl="false",
        save_healpix_map="false", save_bin_maps="false", save_mass_map="false",
        save_noise_maps="false", save_noisy_maps="false", save_mask_map="false",
        save_signal_map="false", save_masked_signal_map="false",
        skip_existing_outputs="false", skip_existing_any_run_instance="false",
        enforce_battaglia_guardrails="true", skip_invalid_battaglia_rows="false",
        run_instance_tag="", print_runtime_environment="true",
        battaglia_alpha_amp=1.0, battaglia_alpha_alpha_m=0.0, battaglia_alpha_alpha_z=0.0,
        battaglia_gamma_amp=-0.3, battaglia_gamma_alpha_m=0.0, battaglia_gamma_alpha_z=0.0)
    settings.update({k: float(v) for k, v in zip(JULIA_PARAMETERS, theta)})
    environment = Path(julia_project).resolve() if julia_project else repo / "julia_env"
    return [str(julia), f"--project={environment}", f"--threads={threads}", "--startup-file=no",
            str(repo / "tSZ_visuals/run_halfdome_fullsky_so_noise.jl")] + [
            f"{key}={value}" for key, value in settings.items()]


def simulate(repo, output, catalogue, noise, theta, **kwargs):
    """Stream Julia output and refuse to reuse results for a changed request."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    command = simulator_command(repo, output, catalogue, noise, theta, **kwargs)
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError("Julia not found. Set JULIA to its absolute Linux executable path.")
    environment_path = Path(kwargs.get("julia_project") or repo / "julia_env").resolve()
    for path in (catalogue, noise, environment_path / "Project.toml"):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith(("TSZ_", "BATTAGLIA_")):
            environment.pop(key)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       HDF5_USE_FILE_LOCKING="FALSE")
    if kwargs.get("julia_depot"):
        environment["JULIA_DEPOT_PATH"] = str(kwargs["julia_depot"])
    # Verify the loaded library supports all nine knobs, and record its actual source.
    preflight = subprocess.run([command[0], "--startup-file=no", command[1],
        str(repo / "SBI_analysis/check_so_moped_simulator.jl")], env=environment,
        text=True, capture_output=True, timeout=300)
    if preflight.returncode:
        raise RuntimeError("Julia/XGPaint preflight failed:\n" + preflight.stderr)
    import tomllib
    library = tomllib.loads(preflight.stdout)
    # Include helper source files, not just the top-level Julia wrapper.
    sources = list((repo / "tSZ_visuals").glob("*.jl"))
    sources += list(Path(library["xgpaint_source"]).parent.rglob("*.jl"))
    sources += [environment_path / "Project.toml", environment_path / "Manifest.toml"]
    request = dict(command=command, theta=np.asarray(theta).tolist(), library=library,
        catalogue=dict(path=str(Path(catalogue).resolve()), size=Path(catalogue).stat().st_size),
        noise_sha256=sha256(noise), sources={str(p): sha256(p) for p in sources})
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "simulation_complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["request"] != request or sha256(saved["raw_profile"]) != saved["raw_sha256"]:
            raise ValueError("Simulation inputs or output changed. Choose a new output directory.")
        return Path(saved["raw_profile"])
    if (output / "simulation_request.json").exists():
        raise RuntimeError("An earlier simulation did not finish. Inspect simulation.log and use a new run directory.")
    write_json(output / "simulation_request.json", request)
    with (output / "simulation.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repo, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True); log.write(line); log.flush()
            if process.wait() != 0:
                raise RuntimeError(f"Julia failed; inspect {output / 'simulation.log'}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise
    raw = discover_profile(output / "raw", kwargs.get("mask_seed", 12345), kwargs.get("noise_seed", 3000001))
    write_json(marker, dict(request=request, raw_profile=str(raw), raw_sha256=sha256(raw)))
    return raw


def prepare_context(raw_path, contract, transform):
    metadata = json.loads(str(contract["metadata_json"].item()))
    matrix = make_cl_to_dl_matrix(contract["ell_unbinned"], contract["bin_ell_min"],
                                  contract["bin_ell_max"], metadata["bin_weighting"])
    cl = align_to_ell(read_profile(Path(raw_path)), contract["ell_unbinned"])
    x = np.asarray(cl @ matrix, dtype=np.float32)
    if x.shape != (40,) or not np.isfinite(x).all():
        raise ValueError("Invalid binned signed D_ell observation")
    return x, project(x, transform)


def load_checked_model(bundle, contract):
    import torch
    # Only load trusted project-produced pickle files.
    try:
        with (Path(bundle) / "density_estimator.pkl").open("rb") as stream:
            estimator = pickle.load(stream).cpu().eval()
    except Exception as exc:
        raise RuntimeError("Model import failed. Use a compatible SBI/Torch/NumPy environment; "
                           "do not modify the saved estimator to bypass an import error.") from exc
    with torch.no_grad():
        actual = estimator.log_prob(torch.tensor(contract["reference_theta"][:32], dtype=torch.float32),
                                    context=torch.tensor(contract["reference_context"][:32])).cpu().numpy()
    np.testing.assert_allclose(actual, contract["reference_log_prob"], atol=2e-3, rtol=2e-4,
                               err_msg="Local model differs from cluster reference likelihood evaluations")
    return estimator


def sampling_diagnostics(estimator, context, contract, pilot_count=20000, seed=47):
    """Bounded raw proposal probe; these draws are NOT posterior samples.

    A zero count is not proof of zero probability, but should prevent an
    expensive rejection loop from hiding an extreme conditioning mismatch.
    """
    from diagnose_so_compression_acceptance import draw_raw, summarize_raw
    context = np.asarray(context, dtype=np.float32)
    reference = contract["reference_context"]
    if context.shape != reference.shape[1:] or not np.isfinite(context).all():
        raise ValueError("Invalid compressed observation shape or nonfinite values")
    if pilot_count < 1:
        raise ValueError("pilot_count must be positive")
    raw = draw_raw(estimator, context, pilot_count, seed + 1)
    result = summarize_raw(raw, contract["low"], contract["high"])
    result.update(context=context.tolist(), reference_min=reference.min(axis=0).tolist(),
        reference_max=reference.max(axis=0).tolist(),
        outside_reference_range=((context < reference.min(axis=0)) |
                                 (context > reference.max(axis=0))).tolist(),
        reference_count=len(reference),
        interpretation="Reference bounds are a heuristic, not a hard likelihood support test.")
    if result["accepted_count"] == 0:
        control = draw_raw(estimator, reference[0], min(pilot_count, 4096), seed + 2)
        result["training_control"] = summarize_raw(control, contract["low"], contract["high"])
        result["zero_count_95pct_upper_bound"] = float(-np.expm1(np.log(.05) / pilot_count))
    return result


def posterior_samples(estimator, context, contract, count=10000, seed=47,
                      max_proposals=1000000, seconds=300, pilot_count=20000,
                      diagnostics_dir=None):
    import torch
    from run_so_sbi_compression_comparison import bounded_samples
    if not 2 <= count <= max_proposals or seconds <= 0:
        raise ValueError("Invalid sampling count or finite sampling budgets")
    diagnostic = sampling_diagnostics(estimator, context, contract, pilot_count, seed)
    diagnostic_path = None
    if diagnostics_dir is not None:
        diagnostic_path = Path(diagnostics_dir) / "sampling_preflight.json"
        write_json(diagnostic_path, diagnostic)
    if diagnostic["accepted_count"] == 0:
        control = diagnostic["training_control"]["acceptance"]
        detail = f" Report: {diagnostic_path}." if diagnostic_path else ""
        raise RuntimeError(
            f"Sampling stopped at preflight: 0/{pilot_count} raw draws lie inside the prior, "
            f"but the training control accepts {control:.2%}. "
            f"Largest absolute standardized MOPED coordinate: {max(abs(np.asarray(context))):.2f}. "
            "Check the simulator/noise-realization contract before sampling again. "
            "More proposals, clipping, or an MCMC fallback do not repair a distribution mismatch."
            + detail)
    torch.manual_seed(seed)
    try:
        return bounded_samples(estimator, context, contract["low"], contract["high"], dict(
            posterior_samples=int(count), max_proposals=int(max_proposals), sampling_seconds=float(seconds)))
    except (RuntimeError, ValueError) as exc:
        if diagnostics_dir is not None:
            write_json(Path(diagnostics_dir) / "sampling_failure.json", dict(
                error=str(exc), pilot=diagnostic, seed=seed, count=count,
                max_proposals=max_proposals, sampling_seconds=seconds))
        raise


def plot_corner(samples, theta, contract, output, label="MOPED NPE: selected SO observation"):
    from getdist import MCSamples, plots
    labels = [r"P_0", r"x_{\rm c}", r"\beta", r"\alpha_{m,P_0}", r"\alpha_{m,x_{\rm c}}",
              r"\alpha_{m,\beta}", r"\alpha_{z,P_0}", r"\alpha_{z,x_{\rm c}}", r"\alpha_{z,\beta}"]
    names = list(PARAM_NAMES)
    ranges = {n: (float(lo), float(hi)) for n, lo, hi in zip(names, contract["low"], contract["high"])}
    gd = MCSamples(samples=samples, names=names, labels=labels, ranges=ranges,
                   settings={"smooth_scale_1D": .3, "smooth_scale_2D": .3})
    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.axes_fontsize = 9; g.settings.lab_fontsize = 11
    g.triangle_plot([gd], params=names, filled=True, contour_colors=["#228833"],
                    legend_labels=[label], markers=theta,
                    marker_args={"color": "black", "ls": ":", "lw": 1},
                    line_args=[{"color": "#228833", "ls": "--", "lw": 1.3}])
    g.export(str(Path(output) / "moped_sbi_corner.png"), dpi=200)
    g.export(str(Path(output) / "moped_sbi_corner.pdf"))
    return g
