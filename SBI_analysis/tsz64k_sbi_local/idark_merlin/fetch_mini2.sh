#!/bin/bash
# poll the mini2 Merlin run (every 5 min, at most 12 h); when comparison.md exists, fetch the results here
set -uo pipefail
cd "$(dirname "$0")/.."
H=/home/kristero10/merlin_unbinned_20260929
SSH="ssh.exe -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 idark"
for i in $(seq 144); do
    s=$($SSH "test -e $H/results_merlin_unbinned/comparison.md && echo done; qstat -u kristero10 2>/dev/null | grep -c merlin_m2; qstat -f 600167 2>/dev/null | grep -E 'cpupercent' | tr -d ' '" 2>/dev/null | tr -d '\r' | tr '\n' ' ')
    echo "$(date -u +%H:%M) $s"
    case "$s" in done*) break ;; esac
    case "$s" in "0 "*) echo "mini2 jobs ended without comparison.md"; break ;; esac
    sleep 300
done
mkdir -p idark_merlin/mini2
$SSH "cd $H && tar -cf - results_merlin_unbinned/*.json results_merlin_unbinned/*.md results_merlin_unbinned/*.png results_merlin_unbinned/b12_samples_*.npy logs" | tar -xf - -C idark_merlin/mini2
cp -p idark_merlin/mini2/results_merlin_unbinned/* results_merlin_unbinned/ 2>/dev/null
ls -la results_merlin_unbinned/*.png results_merlin_unbinned/comparison.md 2>/dev/null
echo "$(date -u +%H:%M) fetched"
