#!/bin/bash
# Chain: sums (reads fitting_inputs) -> fits -> {held-out, 2-halo} in parallel (at most two processes) -> plots.
# Markers: logs/<job>.ok or logs/<job>.failed, logs/chain.done at the end.
C=/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126
R=/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118
cd $C/code
run() { local n=$1; shift; if bash run_job.sh $n "$@"; then touch $C/logs/$n.ok; else touch $C/logs/$n.failed; return 1; fi; }
if [ ! -f $C/out/be_sums.npz ]; then
  run sums be_sums.py $R/code/flamingo_request.json $R/results/flamingo_residuals.npz $R/results/flamingo_residuals.json $C/out --minutes 180 || { touch $C/logs/chain.done; exit 1; }
fi
run fits be_fit.py --task fits || { touch $C/logs/chain.done; exit 1; }
run heldout be_fit.py --task heldout &
run twohalo be_fit.py --task twohalo &
wait
if [ -f $C/code/be_plots.py ]; then run plots be_plots.py; fi
touch $C/logs/chain.done
