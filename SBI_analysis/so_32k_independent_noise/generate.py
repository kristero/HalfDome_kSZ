#!/usr/bin/env python3
"""Resumable independent-noise generation for the two supplied Sobol designs."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

BUNDLE = Path(__file__).resolve().parent
NAMES = ("P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta",
         "alpha_z_P0", "alpha_z_xc", "alpha_z_beta")
MODES = ("two_param", "nine_param")
ROW_CAPACITY = 2**32


def products(config):
    return {f"{case}_deproj{d}": (case, d)
            for case in config["noise_cases"] for d in config["deprojections"]}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def seeds(config, mode, row, product=None):
    global_row = config["sequence_offset"] + row
    root = config["noise_seed_bases"][mode] + config["noise_seed_stride"] * (global_row - 1)
    case, deproj = products(config)[product or config["default_product"]]
    offset = 10000 * (deproj + 1) + {"baseline": 100, "goal": 200}[case]
    return root, root + offset + 1, root + offset + 2


def seed_record(config, mode, row):
    return {p: list(seeds(config, mode, row, p)) for p in products(config)}


def validate_config(config):
    n = config["n_rows"]
    if not isinstance(n, int) or n <= 0 or n & (n - 1):
        raise ValueError("n_rows must be a power of two")
    offset = config["sequence_offset"]
    if (not isinstance(offset, int) or offset < 0 or offset % n
            or offset + n > ROW_CAPACITY):
        raise ValueError("Offset must be a nonnegative multiple of n_rows, within 2**32 rows")
    if config["noise_seed_stride"] != 65536:
        raise ValueError("Keep stride=65536: smaller strides can reuse product/split RNG seeds")
    for key, allowed in (("noise_cases", {"baseline", "goal"}), ("deprojections", {0, 2})):
        values = config[key]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"Invalid or repeated {key}: {values}")
    if config["default_product"] not in products(config):
        raise ValueError("default_product must be one of the selected noise products")
    low, high = np.array(config["prior_low"]), np.array(config["prior_high"])
    if low.shape != (9,) or high.shape != (9,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError("Invalid nine-parameter bounds")
    fiducial = np.array(config["fiducial"])
    if fiducial.shape != (9,) or not np.isfinite(fiducial).all():
        raise ValueError("fiducial must have nine finite parameters")
    if not np.all((fiducial >= low) & (fiducial <= high)):
        raise ValueError("Battaglia12 is outside the prior")
    # Disjoint row blocks and disjoint dataset namespaces prove uniqueness without
    # allocating millions of integers during every worker's preflight.
    bases = [config["noise_seed_bases"][m] for m in MODES]
    span = ROW_CAPACITY * config["noise_seed_stride"]
    mask_seed = config["mask_seed"]
    if (any(not isinstance(b, int) or b < 0 or b + span >= 2**63 for b in bases)
            or abs(bases[1] - bases[0]) < span
            or not isinstance(mask_seed, int) or mask_seed < 0
            or any(b <= mask_seed < b + span for b in bases)):
        raise ValueError("Overlapping/invalid dataset seed namespaces or mask seed")
    offsets = [s - bases[0] for p in products(config)
               for s in seeds(dict(config, sequence_offset=0), MODES[0], 1, p)[1:]]
    if len(set(offsets)) != len(offsets) or not all(0 < s < 65536 for s in offsets):
        raise ValueError("Product seed offsets collide or exceed the reserved row block")
    if config["ell_min"] != 80 or config["ell_max"] != 7979 or config["nside"] != 4096:
        raise ValueError("This handoff fixes NSIDE=4096 and ell=80..7979")
    if not isinstance(config["delta_ell"], int) or config["delta_ell"] <= 0:
        raise ValueError("delta_ell must be a positive integer")
    return 2 * len(MODES) * n * len(products(config))


def validate_noise_tables(config):
    for case in config["noise_cases"]:
        path = BUNDLE / f"noise/SO_LAT_Nell_T_atmv1_{case}_fsky0p4_ILC_tSZ.txt"
        table = np.loadtxt(path)
        if table.ndim != 2 or table.shape[1] < max(config["deprojections"]) + 2:
            raise ValueError(f"Missing noise columns in {path}")
        # Reject silently dropped/duplicated multipoles in the permissive Julia reader.
        if not np.array_equal(table[:, 0], np.arange(80, 7980)):
            raise ValueError(f"Noise table must contain each native ell=80..7979 exactly once: {path}")
        values = table[:, np.array(config["deprojections"]) + 1]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid N_ell values: {path}")


def load_designs(config):
    designs = {}
    for mode in MODES:
        path = BUNDLE / config["design_dir"] / (mode + ".csv")
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = json.loads((path.parent / "provenance.json").read_text())[mode]
        if (metadata["n_rows"] != config["n_rows"] or metadata["sequence_offset"] != config["sequence_offset"]
                or metadata["csv_sha256"] != sha256(path)):
            raise ValueError(f"Design hash/row-offset mismatch: {path}")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or []) != NAMES:
                raise ValueError(f"Wrong parameter order: {path}")
            theta = np.array([[float(r[k]) for k in NAMES] for r in reader])
        if theta.shape != (config["n_rows"], 9) or not np.isfinite(theta).all():
            raise ValueError(f"Invalid design shape/values: {path}")
        if np.any(theta < np.array(config["prior_low"]) - 1e-12) or np.any(theta > np.array(config["prior_high"]) + 1e-12):
            raise ValueError(f"Out-of-prior row in {path}; do not change bounds silently")
        if mode == "two_param":
            fixed = [1, 3, 4, 5, 6, 7, 8]
            np.testing.assert_allclose(theta[:, fixed],
                np.broadcast_to(np.array(config["fiducial"])[fixed], (len(theta), 7)), rtol=0, atol=1e-12)
        designs[mode] = (path, theta)
    return designs


def provenance(config, designs, catalogue):
    sources = sorted((BUNDLE / "simulator/tSZ_visuals").glob("*.jl"))
    sources += sorted((BUNDLE / "vendor/XGPaint/src").glob("*.jl"))
    sources += [BUNDLE / "generate.py", BUNDLE / "simulate.jl", BUNDLE / "safe_paint.jl", BUNDLE / "vendor/XGPaint/Project.toml",
                BUNDLE / "vendor/XGPaint/Manifest.toml"]
    sources += sorted((BUNDLE / "noise").glob("*.txt"))
    record = dict(config=config, source_sha256={str(p.relative_to(BUNDLE)): sha256(p) for p in sources},
        designs={m: sha256(p) for m, (p, _) in designs.items()},
        catalogue=dict(path=str(catalogue.resolve()), size=catalogue.stat().st_size,
                       mtime_ns=catalogue.stat().st_mtime_ns))
    record["experiment_id"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def check(root, catalogue):
    config = json.loads((BUNDLE / "config.json").read_text())
    seed_count = validate_config(config)
    validate_noise_tables(config)
    designs = load_designs(config)
    if catalogue is None or not catalogue.is_file():
        raise FileNotFoundError("Set HALFDOME_PATH to the original lightcone_100.hdf5")
    record = provenance(config, designs, catalogue)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".experiment.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker = root / "experiment.json"
        if marker.exists() and json.loads(marker.read_text()) != record:
            raise ValueError("Changed inputs, code or catalogue: use a NEW output root")
        if not marker.exists():
            write_json(marker, record)
    print(f"Designs: {config['n_rows']} rows each; {seed_count} unique split seeds; fixed mask={config['mask_seed']}", flush=True)
    return config, designs, record


def command(config, mode, row, csv_path, catalogue, out, cache, julia, threads):
    noise_seed, _, _ = seeds(config, mode, row)
    settings = dict(catalog_source="halfdome", simulation_name="halfdome_lightcone_100",
        batching_mode="full", halfdome_path=str(catalogue), output_dir=str(out), cache_dir=str(cache),
        sobol_csv_path=str(csv_path), sobol_row=row, apply_mass_cut=True, mass_min=config["mass_min"],
        cosmo_h=config["cosmo_h"], cosmo_omegab=config["cosmo_omegab"], cosmo_omegac=config["cosmo_omegac"],
        baseline_noise_path=str(BUNDLE / "noise/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"),
        goal_noise_path=str(BUNDLE / "noise/SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt"),
        nside=config["nside"], cl_lmax=config["ell_max"], so_noise_lmax=config["ell_max"], cl_niter=0,
        so_noise_deprojections=",".join(map(str, config["deprojections"])), so_noise_is_dl=False, apply_gaussian_beam=True,
        gaussian_beam_fwhm_arcmin=config["beam_fwhm_arcmin"], mask_fsky=config["mask_fsky"],
        mask_apodization_arcmin=config["mask_apodization_arcmin"], seed=config["mask_seed"],
        mask_seed=config["mask_seed"], noise_seed=noise_seed, model_exists=False,
        reuse_existing_cache=False, cache_wait_seconds=0, interpolator_pad=config["interpolator_pad"],
        interpolator_logM_max=config["interpolator_logM_max"], cleanup_nonpositive_profile_values=True,
        save_cl=True, save_no_noise_cl=True, save_baseline_noise_cross_cl="baseline" in config["noise_cases"],
        save_goal_noise_cross_cl="goal" in config["noise_cases"], save_unmasked_no_noise_cl=False, save_healpix_map=False,
        save_bin_maps=False, save_mass_map=False, save_noise_maps=False, save_noisy_maps=False,
        save_mask_map=False, save_signal_map=False, save_masked_signal_map=False,
        skip_existing_outputs=False, skip_existing_any_run_instance=False,
        enforce_battaglia_guardrails=False, skip_invalid_battaglia_rows=False,
        run_instance_tag="", print_runtime_environment=True)
    args = [julia, "--startup-file=no", f"--project={BUNDLE / 'vendor/XGPaint'}", f"--threads={threads}",
            str(BUNDLE / "simulate.jl")]
    return args + [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in settings.items()]


def simulator_environment(threads):
    # These Julia helpers give environment settings precedence over CLI arguments.
    # Remove every literal override supported by the bundled simulator.
    env = os.environ.copy()
    for source in (BUNDLE / "simulator/tSZ_visuals").glob("*.jl"):
        for key in re.findall(r'env\s*=\s*"([A-Z0-9_]+)"', source.read_text()):
            env.pop(key, None)
    env.update(JULIA_NUM_THREADS=str(threads), JULIA_LOAD_PATH="@:@stdlib", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", HDF5_USE_FILE_LOCKING="FALSE")
    return env


def verify_row(folder, config, mode, row, theta, experiment_id):
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    d = json.loads(marker.read_text())
    if (d["experiment_id"] != experiment_id or d["mode"] != mode or d["row"] != row
            or d["seeds"] != seed_record(config, mode, row) or d["theta"] != theta.tolist()):
        raise ValueError(f"Stale identity: {marker}")
    for kind, item in d["spectra"].items():
        path = folder / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Changed/truncated {kind} spectrum: {path}")
        spectrum = np.load(path, allow_pickle=False)
        if spectrum.shape != (config["ell_max"] + 1,) or not np.isfinite(spectrum).all():
            raise ValueError(f"Invalid spectrum: {path}")
    if set(d["spectra"]) != set(products(config)) | {"clean"}:
        raise ValueError(f"Missing spectrum in {marker}")
    return d


def work(args, config, designs, record):
    if not 0 <= args.worker < args.workers:
        raise ValueError("Require 0 <= worker < workers")
    deadline = time.monotonic() + args.seconds
    cache_root = args.root / "cache"
    cache_root.mkdir(exist_ok=True)
    completed = failures = 0
    for row in range(1, config["n_rows"] + 1):
        for mode_index, mode in enumerate(MODES):
            task = 2 * (row - 1) + mode_index
            if task % args.workers != args.worker:
                continue
            path, theta = designs[mode]
            folder = args.root / mode / "raw" / f"row{row:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / ".lock").open("a") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if verify_row(folder, config, mode, row, theta[row - 1], record["experiment_id"]):
                    continue
                if time.monotonic() + args.row_timeout > deadline:
                    print("Walltime margin reached; resubmit the same worker to continue.", flush=True)
                    if failures:
                        raise RuntimeError("Rows failed before the walltime margin; inspect failure.json")
                    return
                with tempfile.TemporaryDirectory(prefix=f"{mode}_{row}_", dir=cache_root) as cache:
                    cmd = command(config, mode, row, path, args.catalogue, folder, Path(cache), args.julia, args.threads)
                    request = dict(experiment_id=record["experiment_id"], mode=mode, row=row,
                        theta=theta[row - 1].tolist(), seeds=seed_record(config, mode, row), command=cmd)
                    write_json(folder / "request.json", request)
                    print(f"START {mode} row={row}, seeds={request['seeds']}", flush=True)
                    start = time.monotonic()
                    try:
                        with (folder / "simulation.log").open("w") as log:
                            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True,
                                timeout=args.row_timeout, env=simulator_environment(args.threads))
                        text = (folder / "simulation.log").read_text()
                        beams = re.findall(r"^Actual beam FWHM:\s*(\S+)\s*$", text, flags=re.MULTILINE)
                        if len(beams) != 1 or float(beams[0]) != config["beam_fwhm_arcmin"] or "Painter: ring_locked" not in text:
                            raise ValueError("Runtime beam/painter did not match the requested physical contract")
                        for label, expected in (("Mask seed", config["mask_seed"]),
                                ("Noise seed", seeds(config, mode, row)[0])):
                            reported = re.findall(r"^" + label + r":\s*(\d+)\s*$", text, flags=re.MULTILINE)
                            if reported != [str(expected)]:
                                raise ValueError(f"Runtime {label} did not match the request")
                        actual = re.findall(r"^Actual split seeds (\w+):\s*(\d+),(\d+)\s*$", text, flags=re.MULTILINE)
                        expected = {(p, *map(str, ss[1:])) for p, ss in request["seeds"].items()}
                        if set(actual) != expected or len(actual) != len(expected):
                            raise ValueError("Runtime split RNG seeds did not match the unique-seed plan")
                        actual_theta = re.findall(r"^Actual theta:\s*(\[.*\])\s*$", text, flags=re.MULTILINE)
                        if len(actual_theta) != 1:
                            raise ValueError("Missing runtime pressure-parameter record")
                        if not np.allclose(json.loads(actual_theta[0]), request["theta"], rtol=5e-14, atol=1e-15):
                            raise ValueError("Simulator pressure parameters differ from the labelled CSV row")
                        spectra = {}
                        patterns = {p: f"*masked_{case}_noise_cross_cl*_deproj{d}_*.npy"
                                    for p, (case, d) in products(config).items()}
                        patterns["clean"] = "*masked_no_noise_cl*.npy"
                        for kind, pattern in patterns.items():
                            matches = list(folder.glob(pattern))
                            expected_count = len(config["deprojections"]) if kind == "clean" else 1
                            if len(matches) != expected_count:
                                raise ValueError(f"Expected one {kind} output, found {matches}")
                            if kind == "clean" and len({sha256(p) for p in matches}) != 1:
                                raise ValueError("Clean signal differs across deprojections")
                            value = np.load(matches[0], allow_pickle=False)
                            if value.shape != (config["ell_max"] + 1,) or not np.isfinite(value).all():
                                raise ValueError("Missing/nonfinite multipoles")
                            spectra[kind] = dict(file=matches[0].name, sha256=sha256(matches[0]))
                        request.update(spectra=spectra, elapsed_seconds=time.monotonic() - start)
                        write_json(folder / "complete.json", request)
                        completed += 1
                        print(f"DONE {mode} row={row}, seconds={request['elapsed_seconds']:.1f}", flush=True)
                    except (subprocess.SubprocessError, ValueError) as exc:
                        failures += 1
                        write_json(folder / "failure.json", dict(error=str(exc), request=request))
                        print(f"FAILED {mode} row={row}: {exc}", flush=True)
            if args.max_rows and completed + failures >= args.max_rows:
                if failures:
                    raise RuntimeError("Smoke check failed; inspect failure.json and simulation.log")
                return
    print(f"Worker complete; new completed={completed}, failures={failures}", flush=True)
    if failures:
        raise RuntimeError("Some rows failed; inspect row logs and rerun this worker")


def bin_dell(cl, config):
    ell = np.arange(config["ell_min"], config["ell_max"] + 1)
    group = ell // config["delta_ell"]
    weights = 2 * ell + 1
    dell = cl[ell] * ell * (ell + 1) / (2 * np.pi)
    return np.array([np.average(dell[group == g], weights=weights[group == g]) for g in np.unique(group)])


def status_or_combine(args, config, designs, record):
    with (args.root / ".combine.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _status_or_combine(args, config, designs, record)


def _status_or_combine(args, config, designs, record):
    missing = {}
    for mode, (_, theta) in designs.items():
        absent = []
        records = []
        for row in range(1, config["n_rows"] + 1):
            folder = args.root / mode / "raw" / f"row{row:05d}"
            done = verify_row(folder, config, mode, row, theta[row - 1], record["experiment_id"])
            if done is None:
                absent.append(row)
            else:
                records.append((folder, done))
        missing[mode] = absent
        print(f"{mode}: {len(records)}/{config['n_rows']} verified; missing={len(absent)}", flush=True)
        if args.stage != "combine" or absent:
            continue
        out = args.root / mode / "prepared"
        out.mkdir(exist_ok=True)
        (out / "complete.json").unlink(missing_ok=True)
        ell = np.arange(config["ell_min"], config["ell_max"] + 1)
        groups, weights = ell // config["delta_ell"], 2 * ell + 1
        unique = np.unique(groups)
        binned_products, raw_files = {}, {}
        for kind in ["clean", *products(config)]:
            binned = np.empty((len(theta), len(unique)), dtype=np.float32)
            name = ("cl_no_noise" if kind == "clean" else
                    "cl" if kind == config["default_product"] else "cl_" + kind)
            temporary = out / (name + ".tmp.npy")
            mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(theta), len(ell)))
            for i, (folder, done) in enumerate(records):
                cl = np.load(folder / done["spectra"][kind]["file"], allow_pickle=False)
                mmap[i] = cl[ell]
                binned[i] = bin_dell(cl, config)
            mmap.flush()
            del mmap
            temporary.replace(out / (name + ".npy"))
            binned_products[kind] = binned
            raw_files[kind] = name + ".npy"
        target = [0, 2] if mode == "two_param" else list(range(9))
        row_seeds = np.array([seeds(config, mode, r) for r in range(1, len(theta) + 1)])
        case, deproj = products(config)[config["default_product"]]
        meta = dict(complete=True, mode=mode, product=f"masked_{case}_noise_cross_deproj{deproj}",
            painter="ring_locked; no lost overlapping-halo pixel updates",
            products=list(products(config)), raw_cl_files=raw_files, noise_table_convention="N_ell per split",
            covariance_between_ilc_products="not modeled; independently drawn noise per product",
            statistic="weighted mean of linear D_ell; x equals binned D_ell", bin_weighting="2ell_plus_1",
            independent_noise_all_rows=True, same_mask_all_rows=True, beam_applied_to_signal=True,
            beam_fwhm_arcmin=config["beam_fwhm_arcmin"], noise_seed_base=config["noise_seed_bases"][mode],
            noise_seed_stride=config["noise_seed_stride"], noise_seed_formula="base + stride * (sobol_global_row - 1)",
            experiment_id=record["experiment_id"])
        temporary = out / "dataset.tmp.npz"
        extra = {f"x_{p}": binned_products[p] for p in products(config)}
        extra.update({f"noise_split_seeds_{p}": np.array([
            seeds(config, mode, r, p)[1:] for r in range(1, len(theta) + 1)]) for p in products(config)})
        # Keep theta at CSV precision; this avoids rounding an in-prior boundary row
        # outside the float64 saved prior during subsequent data-quality checks.
        np.savez_compressed(temporary, theta=theta[:, target], theta_full=theta,
            param_names=np.array(NAMES)[target], full_param_names=np.array(NAMES),
            prior_low=np.array(config["prior_low"])[target], prior_high=np.array(config["prior_high"])[target],
            x=binned_products[config["default_product"]], x_no_noise=binned_products["clean"], ell_unbinned=ell,
            ell_binned=np.array([np.average(ell[groups == g], weights=weights[groups == g]) for g in unique]),
            bin_ell_min=np.array([ell[groups == g].min() for g in unique]),
            bin_ell_max=np.array([ell[groups == g].max() for g in unique]),
            sobol_global_row=config["sequence_offset"] + np.arange(1, len(theta) + 1),
            mask_seed=np.full(len(theta), config["mask_seed"]), noise_seed=row_seeds[:, 0],
            noise_split_seeds=row_seeds[:, 1:], product=np.asarray(meta["product"]), metadata_json=json.dumps(meta), **extra)
        temporary.replace(out / "dataset.npz")
        write_json(out / "complete.json", dict(metadata=meta, dataset_sha256=sha256(out / "dataset.npz"),
            raw_cl_sha256={p: sha256(out / name) for p, name in raw_files.items()}))
    write_json(args.root / "missing_rows.json", missing)
    if args.stage == "combine" and any(missing.values()):
        raise RuntimeError("Not all rows are complete. Missing rows are never silently dropped.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("check", "work", "status", "combine"))
    p.add_argument("--root", type=Path, default=os.environ.get("OUTPUT_ROOT"), required="OUTPUT_ROOT" not in os.environ)
    p.add_argument("--catalogue", type=Path, default=os.environ.get("HALFDOME_PATH"))
    p.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    p.add_argument("--threads", type=int, default=int(os.environ.get("CPUS", 26)))
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--seconds", type=float, default=float(os.environ.get("WORK_SECONDS", 81000)))
    p.add_argument("--row-timeout", type=float, default=float(os.environ.get("ROW_TIMEOUT_SECONDS", 2400)))
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args()
    args.root = args.root.resolve()
    config, designs, record = check(args.root, args.catalogue)
    if args.stage == "work":
        if args.seconds <= args.row_timeout or args.row_timeout <= 0 or args.threads < 1:
            raise ValueError("Invalid finite runtime/CPU budget")
        work(args, config, designs, record)
    elif args.stage in ("status", "combine"):
        status_or_combine(args, config, designs, record)


if __name__ == "__main__":
    main()
