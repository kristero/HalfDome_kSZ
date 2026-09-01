# Noiseless 512k SO profile emulator

This pipeline trains a deterministic emulator

`9 Battaglia pressure parameters -> 40 binned SO D_ell values`

from the `masked_no_noise` product. It does not add SO instrumental noise during
training or prediction. Add noise only to the linear `D_ell` predictions saved
by the inference script.

## Dataset contract

The default cluster input is:

```text
/home/kristero10/HalfDome_kSZ/SBI_analysis/data_for_cluster/adrian_so_sbi_cases_ell80_7979_dataset_row_metadata_verified/so_masked_no_noise_ell80_7979_sbi_run.npz
```

The trainer requires:

- `theta`: shape `(524288, 9)`;
- `x`: shape `(524288, 40)`, strictly positive linear binned `D_ell`;
- the exact parameter order `P0, xc, beta, alpha_m_P0, alpha_m_xc,
  alpha_m_beta, alpha_z_P0, alpha_z_xc, alpha_z_beta`;
- saved prior bounds and 40 ell-bin coordinates;
- a no-noise product label;
- verified parameter/profile row ordering when the PBS default is used.

If the prepared `masked_no_noise` NPZ does not exist yet, create it from the
consolidated 512k product:

```bash
cd /home/kristero10/HalfDome_kSZ
qsub -v SBI_CASES=masked_no_noise,CASE_DATASET_TAG=dataset_row_metadata_verified \
  SBI_analysis/run_prepare_adrian_so_sbi_cases.pbs
```

Do not enable `ALLOW_SOBOL_IDENTITY_FALLBACK`: the consolidated profiles use
the verified source ordering/mapping rather than an assumed identity ordering.

## Split and model selection

For 524,288 rows, the seeded random outer split contains:

- 445,644 rows (85%) in the training/development partition;
- 78,644 rows (15%) in the untouched test partition.

Five percent of the 85% partition is used temporarily to select the stopping
epoch. A fresh model is then initialized and refitted for that number of epochs
on all 445,644 training rows. Only after that refit is the 15% test set read for
metrics. The exact indices are saved in `split_indices.npz`.

The residual MLP is trained on per-bin standardized `log10(D_ell)`. Prediction
is transformed back to positive, linear `D_ell` before all percentage tests.

## Submit on the cluster

First run a contract-only preflight:

```bash
cd /home/kristero10/HalfDome_kSZ
qsub -v VALIDATE_ONLY=1,OUTPUT_DIR=/lustre/work/kristero10/so_noiseless_emulator_512k_preflight \
  emulator_tSZ/run_train_so_noiseless_emulator.pbs
```

Then submit the complete training job:

```bash
cd /home/kristero10/HalfDome_kSZ
qsub emulator_tSZ/run_train_so_noiseless_emulator.pbs
```

The PBS defaults are the `mini` queue, 26 CPU threads, 128 GB memory, and
`23:59:00`. To use a specific environment, pass its Python executable:

```bash
qsub -v PYTHON=/path/to/environment/bin/python \
  emulator_tSZ/run_train_so_noiseless_emulator.pbs
```

The default output directory is:

```text
/lustre/work/kristero10/so_noiseless_emulator_512k
```

For an intentional rerun in the same directory, pass `OVERWRITE=1`. This
updates pipeline outputs but does not delete unrelated files.

## Test metrics and acceptance checks

The held-out test report includes:

- mean, median, 68th, 95th, 99th, and maximum absolute percentage difference;
- signed percentage bias and RMS percentage error;
- fractions of all test-bin predictions within 1%, 2%, 5%, and 10%;
- log10-space RMSE and MAE;
- per-bin RMSE, MAE, bias, percentage-error percentiles, R-squared, and Pearson
  correlation;
- the same error audit in each fixed prior quartile of every input parameter;
- per-profile metrics, including the worst test profiles.

The default quality gate requires:

- overall median absolute percentage difference <= 1%;
- overall 95th percentile absolute percentage difference <= 5%;
- absolute mean bias in every ell bin <= 1%;
- 95th percentile absolute percentage difference in every ell bin <= 7.5%;
- R-squared in every ell bin >= 0.99;
- 95th percentile absolute percentage difference in every parameter quartile
  <= 7.5%.

These thresholds can be changed with the `GATE_*` PBS variables. By default,
the PBS job exits with status 3 if a gate fails, but it still saves the model
and every diagnostic so the failure can be investigated.

Important outputs are:

```text
so_noiseless_emulator.pt
test_metrics_overall.json
test_metrics_by_bin.csv
test_metrics_by_parameter_quartile.csv
test_profile_metrics.csv
test_predictions.npz
quality_gate.json
training_history.csv
figures/
input_provenance.json
split_indices.npz
training_complete.json
```

## Predict new noiseless profiles

Supply a CSV with the nine named parameter columns, or an NPY/NPZ array in the
saved parameter order:

```bash
python emulator_tSZ/so_noiseless_emulator.py \
  --artifact /lustre/work/kristero10/so_noiseless_emulator_512k/so_noiseless_emulator.pt \
  --theta new_pressure_parameters.csv \
  --output new_noiseless_so_profiles.npz
```

The output key `dl` contains the linear 40-bin noiseless `D_ell` profiles. The
inference command rejects parameters outside the training prior unless
`--allow-extrapolation` is explicitly passed.
