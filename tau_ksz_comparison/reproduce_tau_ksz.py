#!/usr/bin/env python3
"""Generate the HalfDome tau/kSZ versus CLASS-SZ comparison, with saved provenance.

Run through run_local.sh to select the local scientific Python environment.
No synthetic data, fitted amplitudes, noise, beam, or pixel-window corrections
are used. --smoke runs a deliberately incomplete catalogue for software checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    import numpy as np
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def write_toml(path, mapping):
    """Write this workflow's simple scalar/table configuration, readable in Julia."""
    def scalar(value):
        if isinstance(value, bool):
            return str(value).lower()
        return json.dumps(value)

    lines = [f"{json.dumps(k)} = {scalar(v)}" for k, v in mapping.items()
             if not isinstance(v, dict)]
    for section, values in mapping.items():
        if isinstance(values, dict):
            lines += ["", f"[{section}]"]
            lines += [f"{json.dumps(k)} = {scalar(v)}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n")


def scan_catalog(cfg, out, max_halos):
    import h5py
    import numpy as np

    path = out / "catalog_bounds.toml"
    if path.exists():
        return tomllib.loads(path.read_text())
    mass_min, z_min = float("inf"), float("inf")
    mass_max, z_max, valid = 0.0, 0.0, 0
    with h5py.File(cfg["catalog"], "r", locking=False) as catalog:
        count = len(catalog["halo_mass_m200c"])
        limit = min(count, max_halos) if max_halos else count
        if catalog["Position"].shape != (count, 3) or catalog["Velocity"].shape != (count, 3):
            raise ValueError("Expected HalfDome Position and Velocity with shape (N, 3)")
        for start in range(0, limit, cfg["chunk_size"]):
            rows = slice(start, min(start + cfg["chunk_size"], limit))
            mass = catalog["halo_mass_m200c"][rows].astype(float)
            if cfg["catalog_masses_msun_h"]:
                mass /= cfg["cosmology"]["h"]
            z = catalog["redshift"][rows].astype(float)
            position = catalog["Position"][rows]
            velocity = catalog["Velocity"][rows]
            good = (np.isfinite(mass) & (mass > 0) & np.isfinite(z) & (z > 0)
                    & np.isfinite(position).all(axis=1) & (np.sum(position**2, axis=1) > 0))
            if not np.isfinite(velocity[good]).all():
                raise ValueError("Valid halos have invalid velocities; cannot form matched tau/kSZ maps")
            if good.any():
                mass_min = min(mass_min, float(mass[good].min()))
                mass_max = max(mass_max, float(mass[good].max()))
                z_min = min(z_min, float(z[good].min()))
                z_max = max(z_max, float(z[good].max()))
                valid += int(good.sum())
            if start == 0 or rows.stop == limit or start % (20*cfg["chunk_size"]) == 0:
                print(f"Catalogue scan {rows.stop:,}/{limit:,}", flush=True)
    if not valid or mass_min == mass_max or z_min == z_max:
        raise ValueError("Catalogue has insufficient physical mass/redshift support")
    bounds = dict(total_catalog_rows=count, rows_scanned=limit, valid_halos=valid,
                  invalid_halos=limit-valid, complete_catalog=(limit == count),
                  mass_min_msun=mass_min, mass_max_msun=mass_max, z_min=z_min, z_max=z_max)
    write_toml(path, bounds)
    print(json.dumps(bounds, indent=2), flush=True)
    return bounds


def theory(cfg, bounds, out):
    import importlib.metadata
    import numpy as np
    from classy_sz import Class

    destination = out / "class_sz_spectra.npz"
    if destination.exists():
        print("Reusing CLASS-SZ spectra", flush=True)
        return
    cosmology = cfg["cosmology"]
    h = cosmology["h"]
    params = dict(cfg["class_sz"])
    # CLASS-SZ halo integrals use Msun/h; XGPaint receives physical Msun.
    params.update(h=h, omega_b=cosmology["Omega_b"]*h*h,
                  omega_cdm=cosmology["Omega_c"]*h*h,
                  n_s=cosmology["n_s"], sigma8=cosmology["sigma8"],
                  N_ncdm=0, N_ur=3.046,
                  output="tau_tau_1h,tau_tau_2h,kSZ_kSZ_1h,kSZ_kSZ_2h",
                  M_min=bounds["mass_min_msun"]*h, M_max=bounds["mass_max_msun"]*h,
                  z_min=bounds["z_min"], z_max=bounds["z_max"],
                  ell_max=float(cfg["lmax"]), f_b_gas=cosmology["Omega_b"]/(cosmology["Omega_b"]+cosmology["Omega_c"]),
                  hm_consistency=0)
    solver = Class()
    # This installed wrapper inserts massive-neutrino defaults at construction.
    # Remove parameters that become invalid when selecting massless HalfDome.
    for key in ("T_ncdm", "deg_ncdm", "m_ncdm", "omega_ncdm", "Omega_ncdm"):
        solver.pars.pop(key, None)
    solver.set(params)
    write_json(out / "class_sz_parameters.json", dict(solver.pars))
    print("Computing CLASS-SZ B16 1h+2h (full CLASS, massless neutrinos)", flush=True)
    started = time.monotonic()
    try:
        solver.compute()
        tau, ksz = solver.cl_ee(), solver.cl_ksz()
        ell = np.asarray(tau["ell"])
        if not np.array_equal(ell, ksz["ell"]):
            raise ValueError("CLASS-SZ tau and kSZ multipoles differ")
        arrays = {"ell": ell}
        # Despite their names these accessors return dimensionless D_l already.
        for name, values in (("tau", tau), ("ksz", ksz)):
            for component in ("1h", "2h"):
                data = np.asarray(values[component])
                if not np.isfinite(data).all() or (data < 0).any() or not (data > 0).any():
                    raise ValueError(f"Invalid CLASS-SZ {name} {component}")
                arrays[f"dl_{name}_{component}"] = data
        save_npz(destination, **arrays)
        write_json(out / "class_sz_runtime.json", dict(
            version=importlib.metadata.version("classy_sz"), seconds=time.monotonic()-started,
            sigma8=float(solver.sigma8()), units="dimensionless D_ell for tau and deltaT/T"))
    finally:
        solver.struct_cleanup()
        solver.empty()


def map_spectra(cfg, out):
    import h5py
    import healpy as hp
    import numpy as np

    destination = out / "map_spectra.npz"
    if destination.exists():
        print("Reusing map spectra", flush=True)
        return
    result = {"ell": np.arange(cfg["lmax"]+1)}
    with h5py.File(out / "maps.h5", "r") as maps:
        for name, dataset in (("tau", "tau"), ("ksz", "ksz_deltaT_over_T")):
            sky = maps[dataset][:]
            if len(sky) != hp.nside2npix(cfg["nside"]) or not np.isfinite(sky).all():
                raise ValueError("Invalid HEALPix map")
            # Monopoles do not contribute to l>=2 on a continuous full sky.
            # Removing the sample mean reduces numerical monopole leakage.
            mean = float(sky.mean())
            sky -= mean
            print(f"Computing {name} anafast, lmax={cfg['lmax']}", flush=True)
            cl = hp.anafast(sky, lmax=cfg["lmax"], iter=cfg["anafast_iterations"], pol=False)
            if not np.isfinite(cl).all():
                raise ValueError("Nonfinite power spectrum")
            result[f"cl_{name}"] = cl
            result[f"map_mean_{name}"] = mean
            del sky
    save_npz(destination, **result)


def plot(cfg, bounds, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import PchipInterpolator

    with np.load(out / "map_spectra.npz") as f:
        maps = dict(f)
    with np.load(out / "class_sz_spectra.npz") as f:
        prediction = dict(f)
    ell, theory_ell = maps["ell"], prediction["ell"]
    plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), sharex="col",
                             gridspec_kw={"hspace": 0.07, "wspace": 0.19})
    outputs = {"ell": ell}
    label = "XGPaint"
    for column, (name, factor, title, symbol) in enumerate([
        ("tau", 1e7, "tau", r"\tau\tau"),
        ("ksz", 1e13, "kSZ", r"(\Delta T/T)^2")]):
        top, bottom = axes[:, column]
        dl = ell*(ell+1)*maps[f"cl_{name}"]/(2*np.pi)
        theory_dl = prediction[f"dl_{name}_1h"] + prediction[f"dl_{name}_2h"]
        # A restricted smoke catalogue can lie outside the FFT profile's k grid
        # at low ell. Such zero theory values cannot define a log-space ratio.
        usable = np.isfinite(theory_dl) & (theory_dl > 0)
        indices = np.flatnonzero(usable)
        if len(indices) < 2 or np.any(np.diff(indices) != 1):
            raise ValueError("Theory needs one contiguous positive multipole interval")
        valid_ell, valid_dl = theory_ell[usable], theory_dl[usable]
        if not usable.all():
            print(f"{name}: ratio restricted to positive theory support "
                  f"{valid_ell[0]:.3g} <= ell <= {valid_ell[-1]:.3g}", flush=True)
        keep = (ell >= 3) & (dl > 0)
        top.loglog(ell[keep], factor*dl[keep], lw=1.5, label=label)
        top.loglog(valid_ell, factor*valid_dl, color="black", linestyle="--",
                   lw=2, label="CLASS-SZ")
        top.set_title(title)
        exponent = 7 if name == "tau" else 13
        top.set_ylabel(rf"$10^{{{exponent}}}\ell(\ell+1)C_\ell^{{{symbol}}}/(2\pi)$")
        top.legend(frameon=False, fontsize=13)
        # Interpolate only within theory support. Never extend endpoint values.
        overlap = (ell >= valid_ell.min()) & (ell <= valid_ell.max()) & (ell >= 2)
        matched = np.full(ell.shape, np.nan, dtype=float)
        interp = PchipInterpolator(np.log(valid_ell), np.log(valid_dl), extrapolate=False)
        matched[overlap] = np.exp(interp(np.log(ell[overlap])))
        ratio = np.divide(dl, matched)
        bottom.semilogx(ell[overlap], ratio[overlap], lw=1.0)
        bottom.axhline(1, color="black", lw=1.2)
        bottom.set_ylim(0.4, 1.6)
        bottom.set_ylabel(r"$C_\ell^{\rm map}/C_\ell^{\rm CLASS-SZ}$")
        bottom.set_xlabel(r"multipole $\ell$")
        for axis in (top, bottom):
            axis.grid(alpha=0.25)
            axis.set_xlim(2, max(15000 if bounds["complete_catalog"] else 0, cfg["lmax"]*1.1))
        outputs.update({f"dl_{name}": dl, f"dl_theory_{name}_on_map_ell": matched,
                        f"ratio_{name}": ratio})
    if not bounds["complete_catalog"]:
        fig.suptitle("Software smoke check: incomplete catalogue, low resolution", fontsize=15)
        fig.subplots_adjust(top=0.90)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"tau_ksz_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    save_npz(out / "comparison_arrays.npz", **outputs)
    print("Figure:", out / "tau_ksz_comparison.png", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "settings.toml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nside", type=int)
    parser.add_argument("--lmax", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--smoke", action="store_true", help="First 20,000 halos at NSIDE=128; NOT a science result")
    parser.add_argument("--stage", choices=("all", "scan", "theory", "maps", "spectra", "plot"), default="all")
    args = parser.parse_args()
    cfg = copy.deepcopy(tomllib.loads(args.config.read_text()))
    max_halos = 20000 if args.smoke else 0
    if args.smoke:
        cfg.update(nside=128, lmax=383, anafast_iterations=0)
        cfg["output"] = str(HERE / "outputs" / "smoke")
        cfg["interpolation"].update(n_radius=64, n_redshift=24, n_mass=24, validation_rtol=0.03)
        cfg["class_sz"].update(ndim_masses=60, ndim_redshifts=30,
                              n_m_density_profile=20, n_z_density_profile=20,
                              N_samp_fftw=1024, dlogell=0.15, redshift_epsrel=0.02, mass_epsrel=0.02)
    if args.nside is not None:
        cfg.update(nside=args.nside, lmax=3*args.nside-1)
        cfg["output"] = str(HERE / "outputs" / f"nside{args.nside}")
    if args.lmax is not None:
        cfg["lmax"] = args.lmax
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.output is not None:
        cfg["output"] = str(args.output.resolve())
    if cfg["nside"] <= 0 or cfg["nside"] & (cfg["nside"]-1):
        parser.error("nside must be a positive power of two")
    if not 10 < cfg["lmax"] <= 3*cfg["nside"]-1:
        parser.error("Require 10 < lmax <= 3*nside-1")
    if cfg["threads"] < 1 or cfg["chunk_size"] < 1:
        parser.error("threads and chunk_size must be positive")
    os.environ["OMP_NUM_THREADS"] = str(cfg["threads"])
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent runs from writing to the same restart directory.
    import fcntl
    with (out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "plot":
            # Restyle saved spectra using their original scientific settings.
            # Plot edits do not invalidate or rewrite the simulation provenance.
            saved_run = json.loads((out / "run_manifest.json").read_text())
            saved_bounds = tomllib.loads((out / "catalog_bounds.toml").read_text())
            print("Plotting saved spectra; using the recorded run settings", flush=True)
            plot(saved_run["settings"], saved_bounds, out)
            return
        catalog = Path(cfg["catalog"])
        sources = sorted((Path(cfg["xgpaint"]) / "src").glob("*.jl"))
        if not sources:
            raise FileNotFoundError("Configured XGPaint source is missing")
        source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sources + [HERE / "paint_maps.jl", Path(__file__)]}
        identity = dict(settings=cfg, max_halos=max_halos, source_hashes=source_hashes,
                        catalog_size=catalog.stat().st_size, catalog_mtime_ns=catalog.stat().st_mtime_ns)
        manifest = out / "run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Inputs/settings/source changed: choose a new --output directory")
        write_json(manifest, identity)
        write_toml(out / "effective_settings.toml", cfg)
        bounds = scan_catalog(cfg, out, max_halos)
        if args.stage == "scan":
            return
        if args.stage in ("all", "theory"):
            theory(cfg, bounds, out)
        if args.stage in ("all", "maps") and not (out / "maps.h5").exists():
            env = dict(os.environ, JULIA_NUM_THREADS=str(cfg["threads"]))
            command = [cfg["julia"], f"--project={cfg['julia_project']}", str(HERE / "paint_maps.jl"),
                       str(out / "effective_settings.toml"), str(out / "catalog_bounds.toml")]
            print("Starting XGPaint map generation; log:", out / "painting.log", flush=True)
            with (out / "painting.log").open("w") as log:
                with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    if process.wait():
                        raise RuntimeError(f"Julia failed; see {out / 'painting.log'}")
        if args.stage in ("all", "spectra"):
            map_spectra(cfg, out)
        if args.stage in ("all", "plot"):
            plot(cfg, bounds, out)


if __name__ == "__main__":
    main()
