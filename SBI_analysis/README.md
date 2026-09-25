# HalfDome Battaglia SBI Analysis

This folder trains an SBI posterior for the 9 Battaglia tSZ parameters using:

- the existing y100/y102 generated spectra as the reference dataset,
- the trained Battaglia emulator as the fast simulator,
- `log10(D_l)` as the SBI data vector,
- the exact Sobol prior bounds used for the 9-parameter training set.

Default cluster config:

```bash
configs/default_cluster.json
```

The default expects:

```bash
/lustre/work/kristero10/tSZ_data/emulator_battaglia_y100_y102_full/combined_battaglia_profiles.npz
/lustre/work/kristero10/tSZ_data/emulator_battaglia_y100_y102_full/battaglia_tsz_emulator.joblib
```

The combined NPZ is required because the SBI noise model needs the separate `y100` and `y102` spectra.

## Noise Model

The default noise model is `paired_lightcone_diagonal`.

It converts `y100` and `y102` to `log10(D_l)` and estimates the per-ell scatter from their difference. With only two lightcones, a full 4095x4095 covariance is not well determined, so the pipeline uses a diagonal covariance by default.

`noise.target` controls the interpretation:

- `single_lightcone`: use this when the observed spectrum is one independent HalfDome lightcone.
- `two_lightcone_mean`: use this when the observed spectrum is the average of two independent lightcones.

## Run On Cluster

From the cluster copy of the repo:

```bash
cd /home/kristero10/HalfDome_kSZ/SBI_analysis
dos2unix run_sbi_pipeline.pbs
qsub run_sbi_pipeline.pbs
```

To condition on your Battaglia12 observation:

```bash
qsub -v OBSERVED_SPECTRUM_PATH=/path/to/your/battaglia12_spectrum.fits run_sbi_pipeline.pbs
```

If the observation is a precomputed `D_l` NumPy array:

```bash
qsub -v OBSERVED_SPECTRUM_PATH=/path/to/obs_dl.npy run_sbi_pipeline.pbs
```

Set `observed_input_kind` in the config to one of:

```text
auto, cl, dl, log10_dl
```

For pipeline-generated `tSZ_cl*.fits`, use `cl`.

## Main Outputs

The default output directory is:

```bash
/lustre/work/kristero10/tSZ_data/sbi_battaglia9_full_ell
```

Important files:

- `run_config.json`
- `sbi_run_summary.json`
- `sbi_noise_and_dataset_summary.npz`
- `sbi_training_simulations.npz`
- `sbi_prior.pkl`
- `sbi_inference.pkl`
- `sbi_density_estimator.pkl`
- `sbi_posterior.pkl`
- `sbi_training_summary.json`
- `emulator_dataset_validation.json`
- `emulator_dataset_validation.npz`
- `posterior_samples.npy`
- `posterior_samples.csv`
- `observed_log10_dl.npy`
- `plots/noise_sigma_log10_dl.png`
- `plots/training_parameter_coverage.png`
- `plots/training_data_vector_coverage.png`
- `plots/emulator_dataset_residual_log10_dl.png`
- `plots/posterior_corner.png`
- `plots/posterior_predictive_log10_dl.png`

If `observed_spectrum_path` is empty, the pipeline trains and saves the posterior object but skips posterior sampling.
