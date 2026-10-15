#!/bin/bash
# Current-engine Fisher derivatives at Battaglia+12: one clean case per julia pass (peak ~17 GB), sequential.
cd /home/cbllover/HalfDome/SBI_analysis/tsz64k_sbi_local
JULIA=/home/kn18001/.julia/juliaup/julia-1.12.2+0.x64.linux.gnu/bin/julia
ENVDIR=/home/cbllover/HalfDome/SBI_analysis/tsz_local_profile_tests/julia_env
for label in beta_p beta_m xc_p xc_m alpha_z_beta_p alpha_z_beta_m alpha_m_beta_p alpha_m_beta_m \
             alpha_z_xc_p alpha_z_xc_m alpha_m_xc_p alpha_m_xc_m alpha_z_P0_p alpha_z_P0_m alpha_m_P0_p alpha_m_P0_m; do
  if [ -s mocks/fisher_b12/$label/masked_clean_cl.npy ]; then echo "$label already done"; continue; fi
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 18 ]; do
    echo "$(date -u +%H:%M:%S) waiting for 18 GB free before $label"; sleep 60
  done
  echo "$(date -u +%H:%M:%S) start $label"
  JULIA_DEPOT_PATH=/home/kn18001/.julia JULIAUP_DEPOT_PATH=/home/kn18001/.julia /usr/bin/time -v -o mocks/fisher_$label.time \
    $JULIA --startup-file=no --project=$ENVDIR --threads=16 paint_mocks.jl mocks/fisher_$label.toml > mocks/fisher_$label.log 2>&1
  status=$?
  echo "$(date -u +%H:%M:%S) end $label status $status peak_kB $(awk -F': ' '/Maximum resident/ {print $2}' mocks/fisher_$label.time)"
  [ $status -eq 0 ] || { echo "stopping after failure of $label"; exit 1; }
done
echo "$(date -u +%H:%M:%S) all Fisher cases done"
