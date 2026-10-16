# Instructions for coding agents (Codex, Claude, others)

## FLAMINGO-calibrated halo profiles for HalfDome

Work in progress, 2026-10-16. Start with
`flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012/HANDOFF_CODEX.md`. It records:

- the user's binding decisions;
- the status of every test (A-L);
- what is still open (Test L2/L3, the website, the GitHub push);
- how to extend the calibration to all snapshots with full particles.

### Rules the user has set

**Cluster use**
- Heavy computation runs on the clusters (STRW maas through `cluster_tests.py`, or idark), never on this laptop.
- Use at most 8 workers on maas, at nice 19 with the memory and load guards. Put large intermediates in node-local scratch.
- Never modify shared or other users' data. The FLAMINGO data on `/net/hypernova` are read-only.

**This laptop**
- C: is nearly full and the WSL disk lives on it.
- Do not run headless Chrome except through `.../stacking_choices_20261014/tools/safe_chrome.sh`.
- Do not search `/` or `/mnt/c`.
- Run local Python with 2 threads and less than 4 GB RAM.

**Earlier work**
- Do not edit earlier tests' folders or the particle-pass products. Import their modules read-only and configure them in your own process.
- Every result needs an independent verification. Every number must cite its source file and field. Every equation must define all its symbols.

**Halo profiles**
- Fit the halo profile to all gas inside the truncation radius R_t (rule B32, about 1.43 R200c at z = 0).
- HalfDome paints the fitted law extrapolated to 4 R200c.
- The goal is the halo component, not field gas, for kSZ and FRB DM.

**GitHub**
- The repository `kristero/HalfDome_kSZ` is public: push code and reports only, no data.
- Push over https with `gh`; the commands are in the handoff.
