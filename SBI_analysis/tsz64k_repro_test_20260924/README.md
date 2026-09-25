# idark reproduction test of the tSZ_64k bundle (24-25 September 2026)

Purpose: check, from a **fresh clone of GitHub tag `tsz64k-flat-prior-v1`** (commit
`b46ab65`), that the bundle a collaborator receives reproduces the validated 256-row test
and that its production path works end to end. Everything ran on idark under
`/lustre/work/kristero10/tsz64k_repro_test_20260924` through the bundle's own launcher
(`pipeline/common.py`: frozen-source SHA-256 check, environment scrubbing, same command
line as the test). Nothing in the bundle was modified; the extra checks live in `extra/`.

## What was run

- `setup1.sh`: sparse clone of the tag, `setup/install_julia_env.sh` (Julia 1.12.2 downloaded
  and checksummed, pinned packages, XGPaint tag `validated-tsz256-20260922`, git tree
  `3b57cacb...`); `verify_env` OK (`setup1.log`).
- PBS jobs (`extra/jobs_wave1.txt`, `extra/jobs_wave2.txt`): precompile (599265), repro-check
  of rows 0-3 (599266), audit tasks 0, 1, 128, 255 = 1,024 rows (599267[], 599268, 599269),
  noise check with 384 realizations (599270, 599276), repro of test batch 040 = rows 160-163
  (599271), production batches 0-3 = rows 0-15 (599277), profile spot checks and independent
  line-of-sight integration (599278), production batches 127 and 255 = rows 508-511 and
  65532-65535 (599279); then `prepare-batches --defer-unresolved`, `collect --rows 16`,
  `compare-test256`, `extra/residual_check.py`, `extra/report.py`, `extra/plots.py`.

## Results (`extra/results/summary.json`, figures in `extra/results/plots/`)

1. **Same code, same seeds, same numbers.** repro-check rows 0-3: passed; clean spectra agree
   with the test to 1e-15..1e-13, the noisy cross to 1.2e-10 (tolerance 1e-6); random-number
   fingerprints identical; noise maps bitwise identical. Batch 040 (rows 160-163, two of them
   on the 512x256x128 grid): worst 3.2e-12. `compare-test256`: all 256 audits give the same
   cache grid and metrics as the test (difference 0); the 16 production rows with test
   parameters agree with the test's clean spectra at every multipole to 4.8e-13.
2. **Noise seeds are independent.** 131,072 distinct seeds, none equal to a test seed, no row
   with equal splits. 384 realizations drawn with the engine: auto-power z-scores mean 0.005,
   std 1.002 (one of 15,360 above 4 sigma); mean N_ell ratio per bin within 0.03 %; cross
   spectra of train-vs-test seeds, consecutive rows and same-row splits have z std 0.99-1.01
   and no |z| > 4; the same seed twice is bitwise identical; the check's maps equal the
   production maps (rows 0-3, 65532-65535) by SHA-256. Production residuals
   (noisy - clean)/sigma over 16 rows x 40 bins: mean -0.08, std 0.96 (test: 0.04, 0.97);
   same-row residuals of the new and test seeds correlate at 0.05 +/- 0.04.
3. **Profiles are accurate.** All 1,024 audited rows resolved: 875 on 256x128x64, 149 on
   512x256x128 (same 85/15 split as the test); worst visible relative error 3.997e-3 against
   the 0.4 % target, worst |error|/central 3.4e-4, worst area-weighted L1 1.6e-3.
   Independent SciPy integration at 150,528 points of 768 new rows: 1.1e-12. Four rows
   repainted with a 2x finer cache: clean D_ell differs by at most 9.4e-7 per multipole
   (target < 1 %).
4. **Production path.** Six batches of four rows: 21.4-23.6 min on 26 threads, peak RSS
   37.8-38.7 GiB; `collect` wrote `rows_16` with 32 independent noise realizations and no
   missing rows; `compare-test256` passed.
5. **Design structure** (`extra/design_structure/`): prefixes of the Sobol design are balanced,
   strided subsets are not (rows 0, 2, 4, ... all have xc in the lower half). Deliver or
   train on prefixes `rows 0..N-1`, never on every k-th row.

Not run here: the audits of the other 252 tasks and production beyond 24 rows (that is the
collaborator's run); `manage.py status` ends with 4/256 audit tasks and 6/256 planned batches
done, 64,512 rows deferred by the partial audit.

## Contents

`extra/` scripts (`run_extra.py`, `noise_check.jl`, `residual_check.py`, `report.py`,
`plots.py`, `report_figure.py`) and their outputs; `run_root/` manifests, plan, audit summary,
`repro/repro_result.json`, `dataset/rows_16/meta.json`, PBS logs; `setup1.sh`, `setup1.log`.
Per-row spectra and the large `.npy`/`.npz` arrays stayed on idark.
