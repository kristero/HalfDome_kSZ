# Prepared 8192 diagnostic - not submitted

256 saved independent-uniform Sobol rows, unchanged parameter values and split
noise seeds; 192 training-pool and 64 held-out rows. Raw NSIDE8192 is smoothed
with the 2 arcmin beam and synthesized to output NSIDE4096, with the original
fsky=0.4 mask. The pressure support is a 4 R200 sphere.

MOPED reads all 7900 individual multipoles at ell=80..7979. Separate 40-bin
and PCA runs are comparison baselines. No lower-ell cutoff has been adopted.

Numerical profile files are byte-identical to the completed 8192 validation.
The lean allocation block is copied from the validated control.jl. Four
workers own disjoint row IDs modulo four; atomic mkdir claims guard against
duplicate writers across Lustre clients. Only the final analysis job collects
the full dataset. Failed rows retain their identities and stop that worker;
the collector refuses incomplete data. No failed draw is replaced.

The scripts remain unsubmitted. Generation requires four separately assigned
DIAGNOSTIC_WORKER values and analysis after all four jobs succeed. The existing
256 design has not been relabelled as scientifically certified. Numerical
accuracy across the entire prior and full-catalogue pixel integration remain
open questions for a diagnostic/validation sample.
