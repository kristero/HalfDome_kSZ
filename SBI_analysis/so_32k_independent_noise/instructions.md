# Adrian's agent: generate the 32k baseline-deproj0 mocks

Read this file before changing or executing anything in this package. It is the
runbook for the current request, not permission to run every example in README.md.
Work from the extracted `so_32k_independent_noise/` directory on Adrian's cluster.

## 1. Authorized scope and success criteria

Generate BOTH shipped parameter designs:

- `two_param`: 32768 rows varying P0 and beta; the other seven parameters fixed
  to Battaglia12.
- `nine_param`: 32768 rows varying all nine pressure parameters.

This is 65536 simulations total, NOT 32768 shared between the two designs.
Generate masked SO baseline deproj0 only, with independent instrumental-noise
splits for every row and the same mask and halo lightcone throughout. Save the
clean reference spectrum as well. Do not train an NPE or run PCA/MOPED/Fisher here.

Do NOT activate `config_524288_all_noise.example.json`, run `make_designs.py`,
change priors, sort the CSVs, enable goal/deproj2, or save full-sky maps for this
request. The shipped `config.json` and `designs/` are ready to use unchanged.

Finish only when both designs report 32768/32768 verified, combination succeeds,
both prepared completion markers and hashes pass, and the final checks below
pass. Submitted jobs, an empty queue, or a zero exit from `status` are NOT proof
of completion. Report partial progress honestly when time or resources run out.

## 2. Files to use

- `submit.sh`: PBS smoke/worker-wave submission entry point.
- `worker.pbs`: compute-node wrapper; do not submit without its required variables.
- `generate.py`: input checks, per-row generation, status and combination.
- `simulate.jl` and `safe_paint.jl`: Julia entry point and thread-safe halo painting.
- `cluster.env`: the only file normally edited for this run; local paths/resources.
- `config.json`, `designs/`, `noise/`, `simulator/`, `vendor/`: scientific inputs
  and pinned implementation; keep unchanged after verifying the archive.
- `README.md`, `VALIDATION.md`, `MOCK_AUDIT_EXPLAINED.md`: scientific conventions,
  prior table, previous local checks and known limitations.

Keep the whole extracted package on a shared filesystem accessible to workers.
Do not invoke a similarly named simulator from another checkout or installed
XGPaint environment. Do not rebuild the archive from upstream sources.

## 3. Discover the cluster before submitting

Confirm with Adrian any missing information rather than inventing it:

- Absolute path to the ORIGINAL HalfDome `lightcone_100.hdf5`, including its
  provenance and units. The catalogue is not in the archive.
- A new, writable, shared scratch `OUTPUT_ROOT`, adequate storage and file quota.
  The raw plus combined spectra alone require roughly 13 GB across both designs;
  leave headroom for row logs, metadata, temporary files and filesystem overhead.
- Scheduler, permitted queue, account if required, memory, CPU and walltime limits.
  `mini` is a supplied example from another cluster, not an assumed Adrian queue.
- Shared Julia 1.12.2 executable/depot and Python >=3.8 environment with NumPy.
  Install SciPy and Matplotlib as well for the complete tests and diagnostic plots.

PBS Professional/OpenPBS is supported by `submit.sh`. If Adrian uses Slurm or a
different PBS dialect, adapt ONLY the scheduling wrapper after confirming local
requirements. Preserve the work command, distinct worker IDs, finite budgets,
shared output root and five-active-job limit. Record wrapper changes. Do not try
PBS commands against a different scheduler or silently reduce simulation fidelity.

Default scheduling is 20 separate worker jobs, at most five running at once,
one node/job, 26 CPUs and Julia threads, 128 GB, 23:59:00. These are requests, not
guarantees of local availability or sufficient memory; measure the smoke run.
Do not submit 64k jobs or a PBS array. Do not run full-resolution simulations on
a login node. Do not change or cancel unrelated jobs.

## 4. Verify transfer, configure the environment, install once

In a fresh extraction, BEFORE editing `cluster.env`:

```bash
sha256sum -c SHA256SUMS
```

Stop on any mismatch. Do not regenerate SHA256SUMS to hide one. After intentional
edits to `cluster.env` its original checksum will no longer match; record that
change rather than treating it as a scientific-input update.

Edit `cluster.env` for Adrian's absolute `HALFDOME_PATH`, `OUTPUT_ROOT`, `JULIA`,
`PYTHON`, queue and allowed resources. `OUTPUT_ROOT` must not contain previous
fixed-noise datasets. Avoid spaces/commas in paths passed through PBS `-v`.
The `:=` defaults in this file KEEP pre-existing environment values; start a
fresh shell or unset conflicting variables first. Verify the effective values:

```bash
source cluster.env
printf 'Catalogue: %s\nOutput: %s\nJulia: %s\nPython: %s\nQueue: %s\n' \
  "$HALFDOME_PATH" "$OUTPUT_ROOT" "$JULIA" "$PYTHON" "$QUEUE"
test -r "$HALFDOME_PATH"
"$JULIA" --version
"$PYTHON" --version
"$JULIA" --startup-file=no setup_julia.jl
```

Require Julia 1.12.2 for the shipped Manifest and RNG reproducibility. Installation
needs network access and must occur only once where workers can read the packages.
If installation or a preflight command fails, stop and fix it before submitting.
Short validation commands may run on a login node only if local policy permits;
otherwise use a small compute allocation. Never instantiate in every worker.

Keep WORK_SECONDS below requested walltime with margin for cleanup. It must be
larger than ROW_TIMEOUT_SECONDS. Defaults are 81000 and 2400 seconds; a slow row
may require an increased timeout after measurement, not a new seed or new theta.

## 5. Preflight: verify the exact request and implementation

Run the following guard before any submission. It checks the current 32k-only
scope; `generate.py check` also checks the complete priors, CSV provenance,
noise curves and seed namespaces.

```bash
"$PYTHON" - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("config.json").read_text())
expected = {
    "n_rows": 32768, "sequence_offset": 0, "design_dir": "designs",
    "noise_cases": ["baseline"], "deprojections": [0],
    "default_product": "baseline_deproj0", "mask_seed": 12345,
    "nside": 4096, "ell_min": 80, "ell_max": 7979,
    "delta_ell": 200, "beam_fwhm_arcmin": 2.0,
}
for key, value in expected.items():
    if config[key] != value:
        raise ValueError(f"Out of scope: {key}={config[key]!r}; require {value!r}")
print("Scope passed: BOTH 32768-row designs, baseline deproj0 only")
PY
"$PYTHON" -m unittest -v test_generation.py test_sobol_plots.py
"$JULIA" --startup-file=no --project=vendor/XGPaint check_runtime.jl
"$PYTHON" check_simulator_cli.py --julia "$JULIA"
"$PYTHON" diagnose_noise.py --julia "$JULIA"
"$PYTHON" generate.py check
```

Expected Python preflight: 32768 rows per design, 131072 unique split seeds and
fixed mask seed 12345. Save stdout/stderr and dependency versions outside the
source directory, e.g. under `$OUTPUT_ROOT/agent_preflight/`.

The unit-test worker uses a mocked simulator. The real Julia noise diagnostics
use reduced-resolution maps. Neither proves a full-resolution production run
will fit memory or finish within walltime; the next step is mandatory.

## 6. Full-resolution smoke test, then production

```bash
bash submit.sh smoke
```

This is one compute job that generates the first pending row of each design in
a fresh output root, at production NSIDE=4096. Keep its valid outputs: the wave
will skip completed rows. Record its job ID from `$OUTPUT_ROOT/latest_jobs.txt`.
Follow `$OUTPUT_ROOT/logs/worker_<id>_<PBS_JOBID>.log` with `tail -F`; individual
simulation logs are under `<mode>/raw/row00001/simulation.log`.

After the smoke job has ended:

```bash
"$PYTHON" generate.py status
```

Inspect BOTH first-row `complete.json` records and simulation logs. Verify finite
spectra, requested theta/seeds, actual 2 arcmin beam and `Painter: ring_locked`.
Check scheduler exit status if available, elapsed time and peak memory. A clean
reference is expected; goal/deproj2 products are not. A `failure.json`, timeout,
OOM or missing completion marker needs investigation before production.

Only after this passes, inspect and submit the first production wave:

```bash
DRY_RUN=1 bash submit.sh wave
bash submit.sh wave
```

Record all 20 job IDs. The dependency lanes limit this wave to five active jobs;
they are not a global user-level limit. Never overlap waves or other copies of
this package to bypass that limit. Wait for every job in the wave to end before
running status and submitting the next wave if rows remain:

```bash
"$PYTHON" generate.py status
bash submit.sh wave
```

`status` writes `$OUTPUT_ROOT/missing_rows.json`; its exit code can be zero with
missing rows, so read the counts. Resubmission resumes unfinished work using the
same theta and seeds. Do not edit config, designs or simulator sources during a
run, remove experiment/completion/lock files to bypass checks, or silently skip
failed rows. Fix resource/environment failures and resume; scientific/source
changes require explicit agreement and a new output root.

20 jobs will NOT necessarily finish 65536 full-sky simulations in one day. Using
240 seconds/row only as an illustration, the work is about 4369 node-hours or
36.4 days with five continuously active workers, before queue and other overhead.
Use measured runtime, not this estimate, to plan waves. If the agent cannot stay
active, leave an exact progress report and resume commands; do not claim completion.

## 7. Combine and verify both final products

Run on a node allowed to do sustained shared-filesystem I/O, with no workers still
writing. Proceed only after BOTH counts are 32768/32768 verified, missing=0:

```bash
"$PYTHON" generate.py combine
```

Do not retrain anything. Verify the final files, original theta ordering, fixed
parameters, saved bounds, masks, seed uniqueness and checksums:

```bash
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
from generate import NAMES, load_designs, sha256

root = Path(os.environ["OUTPUT_ROOT"])
config = json.loads(Path("config.json").read_text())
designs = load_designs(config)
experiment = json.loads((root / "experiment.json").read_text())
missing = json.loads((root / "missing_rows.json").read_text())
assert set(missing) == {"two_param", "nine_param"}
assert not any(missing.values()), "Rows remain incomplete"
all_seeds = []
for mode, columns in (("two_param", [0, 2]), ("nine_param", list(range(9)))):
    folder = root / mode / "prepared"
    complete = json.loads((folder / "complete.json").read_text())
    meta = complete["metadata"]
    assert meta["complete"] and meta["experiment_id"] == experiment["experiment_id"]
    assert meta["products"] == ["baseline_deproj0"]
    assert meta["noise_table_convention"] == "N_ell per split"
    assert sha256(folder / "dataset.npz") == complete["dataset_sha256"]
    with np.load(folder / "dataset.npz", allow_pickle=False) as data:
        assert data["theta"].shape == (32768, len(columns))
        assert data["x"].shape == data["x_no_noise"].shape == (32768, 40)
        assert np.isfinite(data["x"]).all() and np.isfinite(data["x_no_noise"]).all()
        np.testing.assert_array_equal(data["theta_full"], designs[mode][1])
        np.testing.assert_array_equal(data["theta"], designs[mode][1][:, columns])
        np.testing.assert_array_equal(data["param_names"], np.array(NAMES)[columns])
        np.testing.assert_array_equal(data["prior_low"], np.array(config["prior_low"])[columns])
        np.testing.assert_array_equal(data["prior_high"], np.array(config["prior_high"])[columns])
        np.testing.assert_array_equal(data["sobol_global_row"], np.arange(1, 32769))
        np.testing.assert_array_equal(data["ell_unbinned"], np.arange(80, 7980))
        np.testing.assert_array_equal(data["x"], data["x_baseline_deproj0"])
        assert np.all(data["mask_seed"] == 12345)
        assert data["noise_split_seeds"].shape == (32768, 2)
        all_seeds.append(data["noise_split_seeds"].ravel())
    for product, filename in meta["raw_cl_files"].items():
        path = folder / filename
        assert sha256(path) == complete["raw_cl_sha256"][product]
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        assert raw.shape == (32768, 7900)
        for start in range(0, len(raw), 256):
            assert np.isfinite(raw[start:start + 256]).all()
        del raw
    print(f"PASSED: {mode}, 32768 labelled rows, spectra and hashes verified")
assert np.unique(np.concatenate(all_seeds)).size == 131072
print("PASSED: no split seed reused within or between the two datasets")
PY
```

Expected deliverables in each of `$OUTPUT_ROOT/two_param/prepared/` and
`$OUTPUT_ROOT/nine_param/prepared/`:

- `dataset.npz`: theta, full theta, prior bounds, 40-bin noisy/clean D_ell,
  ell/bin vectors, row identities and seed labels.
- `cl.npy` and `cl_no_noise.npy`: unbinned noisy/clean C_ell, use memory mapping.
- `complete.json`: final metadata and hashes.

Also retain `experiment.json`, original parameter CSVs/provenance, missing-row
report, logs and per-row completion records. Do not delete raw outputs to save
space without agreement. Transfer final products with their metadata together;
never transfer just spectra without theta and row identities.

## 8. Physical contract: do not silently change it

The observable is the cross-spectrum of
`mask * (beam_smoothed_y(theta) + noise_A)` and
`mask * (beam_smoothed_y(theta) + noise_B)`.
Both splits use the supplied baseline deproj0 N_ell AS PER-SPLIT POWER, with
independent RNG streams. Do not multiply by two or draw one fixed noise pair for
all rows. The apodized fsky=0.4 mask is fixed; the 2 arcmin beam is applied to the
signal once. This is a masked pseudo-spectrum, without beam/fsky deconvolution.

Independent noise means zero expected additive cross-noise bias, not zero
variance; negative individual spectra are allowed. Convert to
`D_ell = ell*(ell+1)*C_ell/(2*pi)` before 2ell+1-weighted Delta-ell=200 binning.
There is no data floor, log or asinh in this generation stage. The shared
lightcone means halo/cosmic variance is not independently resampled between rows.
The two-parameter and nine-parameter designs answer different nuisance-parameter
questions, even though their P0/beta generation bounds agree.

## 9. Required report to Adrian and the requesting user

Write a short `$OUTPUT_ROOT/AGENT_RUN_REPORT.md` with the extracted package hash,
catalogue provenance/path, exact effective environment/resources, dependency
versions, configuration/design hashes, smoke evidence, job IDs, finished/missing
counts for each design, validation results, failures and final absolute paths.
List every file changed. Distinguish earlier supplied local checks from checks
actually executed on Adrian's cluster. If unfinished, state the blocking issue
or next wave command and do not label the datasets complete.
