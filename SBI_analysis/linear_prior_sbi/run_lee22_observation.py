#!/usr/bin/env python3
"""Process the Lee22 no-c pressure Compton-y map through the FLAMINGO-comparison operator (2 arcmin beam,
cached fsky=0.4 apodized mask, fixed SO baseline noise splits 22446/22447, lmax 7979) and bin it to the
40-bin observable of the linear-prior dataset. Mirrors run_campaign.py's environment for the Julia stage."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

CAMPAIGN = Path("/lustre/work/kristero10/flamingo_tsz_comparison_20260914")
RUN = Path("/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915")
ROOT = Path("/lustre/work/kristero10/lee22_tsz_observation_20260918")
SBI_ROOT = Path("/lustre/work/kristero10/sbi_linear_8k_20260918")
INPUT = ROOT / "inputs/lee22_noconc_pressure_compton_y_allz_nside4096_m200c_r200cx4.fits"
EXPECTED_SHA = sys.argv[1]
OUTPUT = ROOT / "results/Lee22_noconc"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""): h.update(block)
    return h.hexdigest()

def main():
    t0 = time.time()
    actual = sha256(INPUT); assert actual == EXPECTED_SHA, ("input sha256 mismatch", actual)
    print("input verified", INPUT, flush=True)
    if not (OUTPUT / "complete.toml").exists():
        metadata = json.loads((CAMPAIGN / "preflight/metadata_manifest.json").read_text())
        request = metadata["halfdome_reference"]["simulation_request"]
        noise = CAMPAIGN / "code/halfdome/other_sims/SO/SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_tSZ.txt"
        settings = dict(part.split("=", 1) for part in request["command"] if "=" in part and not part.startswith("--"))
        settings.update(halfdome_path="/lustre/work/Globus-lt/halfdome/full_res/halos/lightcone_100.hdf5",
                        baseline_noise_path=str(noise), goal_noise_path=str(noise.with_name("SO_LAT_Nell_T_atmv1_goal_fsky0p4_ILC_tSZ.txt")),
                        output_dir=str(ROOT / "raw"), cache_dir=str(CAMPAIGN / "paint_cache"))
        julia = [str(CAMPAIGN / "runtime/julia-1.12.2/bin/julia"), "--startup-file=no", "--threads=26",
                 "--project=" + str(CAMPAIGN / "runtime/julia_env"), str(SBI_ROOT / "code/process_fits_map.jl")]
        julia += [k + "=" + v for k, v in settings.items()]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("TSZ_", "BATTAGLIA_"))}
        env.update(HALFDOME_SOURCE_DIR=str(CAMPAIGN / "code/halfdome"), FLAMINGO_CAMPAIGN=str(CAMPAIGN), HDF5_USE_FILE_LOCKING="FALSE",
                   JULIA_DEPOT_PATH=str(CAMPAIGN / "runtime/depot") + ":/home/kristero10/.julia", JULIA_NUM_PRECOMPILE_TASKS="8",
                   JULIA_PKG_PRECOMPILE_AUTO="0", LD_LIBRARY_PATH=str(CAMPAIGN / "runtime/julia-1.12.2/lib/julia") + ":" + env.get("LD_LIBRARY_PATH", ""),
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="26",
                   FLAMINGO_MODE="map", FLAMINGO_OUTPUT=str(OUTPUT), FLAMINGO_INPUT=str(INPUT))
        (ROOT / "raw").mkdir(exist_ok=True); (ROOT / "logs").mkdir(exist_ok=True)
        print("running:", " ".join(julia[:5]), "...", flush=True)
        with (ROOT / "logs/process_lee22.log").open("w") as log:
            code = subprocess.call(julia, env=env, stdout=log, stderr=subprocess.STDOUT)
        assert code == 0, "Julia operator stage failed; see logs/process_lee22.log"
    # bin exactly as the dataset (worker.bin_cl, 40 bins ell 80..7979)
    sys.path.insert(0, str(RUN / "code"))
    from worker import BINNED_NAMES, EDGES, SPECTRA, bin_cl
    import toml
    values = {BINNED_NAMES[key]: bin_cl(np.load(OUTPUT / (key + ".npy"))) for key in SPECTRA}
    (SBI_ROOT / "observations").mkdir(exist_ok=True)
    np.savez(SBI_ROOT / "observations/Lee22_noconc.npz", bin_edges=EDGES, **values)
    (SBI_ROOT / "observations/Lee22_noconc.json").write_text(json.dumps(dict(
        source=str(OUTPUT), input=str(INPUT), input_sha256=actual, operator=toml.load(str(OUTPUT / "complete.toml")),
        profile="Lee22 no-concentration electron pressure (arXiv:2205.01710 Table 7), HalfDome halos, h=0.68, 4 R200c projected aperture, all z",
        note="same beam, mask and fixed SO noise splits (22446/22447) as the HalfDome and FLAMINGO observations"), indent=1, default=str))
    print("wrote", SBI_ROOT / "observations/Lee22_noconc.npz", {k: float(v[15]) for k, v in values.items()}, f"{time.time() - t0:.0f} s", flush=True)

if __name__ == "__main__":
    main()
