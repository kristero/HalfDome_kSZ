| input | test log p | TARP 68 / 95 | mean r | mean test width / prior | mean test width / flat prior | mean edge mass ratio | B12 log p(truth) | B12 mean width / prior |
|---|---|---|---|---|---|---|---|---|
| Merlin: whitened PCA (75) -> learned head (N_feat = 2) -> NPE | -3.38 ± 0.06 | 67 / 95 | 0.720 | 0.383 | 0.561 | 1.049 | -4.12 ± 0.37 | 0.506 |
| whitened PCA (75) -> NPE | -4.67 ± 0.07 | 69 / 96 | 0.677 | 0.459 | 0.673 | 1.169 | -6.05 ± 0.64 | 0.545 |
| MOPED (fixed B12, 9) -> NPE | -4.38 ± 0.14 | 71 / 96 | 0.583 | 0.543 | 0.795 | 1.288 | -3.97 ± 0.28 | 0.531 |

Inputs: asinh. Errors: sd over the 3 seeds / sqrt(3) (test: of the per-seed mean log p over the 3,000 test rows; B12: of the per-seed log p of the true parameters for the fitted SO mock). Widths: median 68 % interval over the prior range (flat prior 0.683) and over 0.683 x the range (flat prior 1). Edge mass ratio: posterior mass in the outer 1 % of each prior range over the flat prior's 2 % (1 on average for an exact posterior). n/a: metrics from an older evaluate.
