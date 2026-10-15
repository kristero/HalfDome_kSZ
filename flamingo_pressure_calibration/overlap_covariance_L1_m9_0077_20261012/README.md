# FLAMINGO L1_m9 z=0: overlap, covariance, sigmoid, literature-recipe and AGN tests

Started 2026-10-12. Plan: `/root/.claude/plans/there-has-been-multiple-indexed-lantern.md` (approved).
Earlier studies in `..` are read-only inputs; nothing outside this directory is edited.

| Path | Step | Content |
|---|---|---|
| `kernels/` | 0.1 | profile-space kernels: per-shell importance for C_ell / Y / DM, propagation engine |
| `C_sigmoid/` | 0.2 | Test C: sigmoid (logistic in ln r) vs hard truncation |
| `analysis/` | 0.3 | shared analysis library (stacking, masks, covariance, stability, GLS fits) |
| `quicklook/` | 0.3 | first covariance / GLS / AGN / membership results on radii234 `measurements.npz` |
| `census_dryrun/` | 0.4 | local overlap-census dry run (centrals catalogue + own-satellite proxies) |
| `F_observables/` | F | Test F (added by the user): observable-driven calibration for tSZ/kSZ autos, DM PDF, tSZ x DM |
| top level | 1.x | maas census + particle-pass code (deployed only after Checkpoint 1) |

Python: `/home/cbllover/miniconda3/envs/flamingo/bin/python -B` (numpy 2.2.6, scipy 1.17.1, h5py 3.16, matplotlib 3.11.2).
Remote: `ssh.exe strw` gateway, compute on `ssh.exe strw-maas` (ProxyJump strw); remote shell is tcsh, so
commands are sent as Python programs on stdin (see `remote_control.py`).
