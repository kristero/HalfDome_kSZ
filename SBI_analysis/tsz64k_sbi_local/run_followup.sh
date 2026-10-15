#!/bin/bash
# After the capacity runs (run_capacity.sh) and the idark validation rows: the checks figure with the capacity panel,
# the validation rows (validation_idark.sh) and sbi_final.py validate. Polls idark every 10 min for at most 12 h.
set -uo pipefail
cd "$(dirname "$0")"
PY=/home/cbllover/miniconda3/envs/halfdome/bin/python
export LD_LIBRARY_PATH=/home/cbllover/miniconda3/envs/halfdome/lib
while pgrep -f "[r]un_capacity.sh" > /dev/null; do sleep 60; done
echo "$(date -u +%H:%M) capacity runs finished"
grep -hv -i warn logs_final/capacity_maf256x8.log logs_final/capacity_nsf128x5.log
$PY sbi_final.py checks --threads 16 2>&1 | grep -v Warning | grep -E "N61440_|rebuilt"
for i in $(seq 72); do
    n=$(ssh.exe -o BatchMode=yes -o ConnectTimeout=30 idark 'ls /home/kristero10/tsz64k_missing_20260928/run_root/rows | wc -l' 2> /dev/null | tr -d '\r')
    [ "$n" = "1096" ] && break
    sleep 600
done
[ "$n" = "1096" ] || { echo "$(date -u +%H:%M) validation rows still incomplete on idark ($n of 1096)"; exit 1; }
echo "$(date -u +%H:%M) validation rows complete on idark"
./validation_idark.sh || exit 1
$PY sbi_final.py validate --threads 16 2>&1 | grep -v Warning
echo "$(date -u +%H:%M) follow-up done"
