#!/bin/bash
# One job of the BE pressure-from-density study on STRW maas: nice 19, idle I/O class, one thread, address-space guard inside the
# driver (RLIMIT_AS), log and JSON receipt per job (code hashes, BD code hashes read-only, output hashes).
# Usage: run_job.sh <job name> <script> [script arguments ...]
set -u
C=/net/schelde/data2/nagainis/HalfDome_runtime/BE_pressure_from_density_20261126
export BD_ROOT=/net/schelde/data2/nagainis/HalfDome_runtime/BD_halfdome_flamingo_exact_shells_20261118
export BD_CODE=$BD_ROOT/code BE_OUT=$C/out BE_SUMS=$C/out/be_sums.npz BE_MEM_GIB=${BE_MEM_GIB:-6}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=$C/mpl
PY=/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python
name=$1; shift; script=$1; shift
mkdir -p $C/out $C/logs $C/receipts $C/figures $C/mpl
cd $C/code
start=$(date -u +%FT%TZ)
nice -n 19 ionice -c3 /usr/bin/time -v -o $C/logs/$name.time $PY -B -u $script "$@" > $C/logs/$name.log 2>&1
status=$?
end=$(date -u +%FT%TZ)
$PY -B - "$C" "$name" "$script" "$status" "$start" "$end" "$*" <<'PYEOF'
import hashlib, json, os, socket, sys
from pathlib import Path
C, name, script, status, start, end, argv = sys.argv[1:8]
C = Path(C)
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
peak = None
for line in (C/'logs'/f'{name}.time').read_text().splitlines():
    if 'Maximum resident set size' in line:
        peak = int(line.split(':')[-1])/2**20
code = {p.name: h(p) for p in sorted((C/'code').glob('*')) if p.is_file()}
bd = Path(os.environ['BD_CODE'])
bd_code = {n: h(bd/n) for n in ('bd_explore.py', 'bd_explore2.py', 'bd_explore3.py', 'bd_explore_run.py', 'bd_mass.py', 'bd_twohalo.py') if (bd/n).exists()}
outs = {str(p.relative_to(C)): h(p) for p in sorted((C/'out').glob('*')) if p.is_file()}
rec = dict(job=name, script=script, arguments=argv, host=socket.gethostname(), start_utc=start, end_utc=end, exit_status=int(status), peak_rss_gib=peak,
           log=str(C/'logs'/f'{name}.log'), env={k: os.environ.get(k) for k in ('BD_ROOT', 'BD_CODE', 'BE_OUT', 'BE_SUMS', 'BE_MEM_GIB', 'OMP_NUM_THREADS')},
           nice=19, ionice='idle (-c3)', code_sha256=code, bd_code_sha256_read_only=bd_code, outputs_sha256=outs)
(C/'receipts'/f'{name}.json').write_text(json.dumps(rec, indent=1)+'\n')
print(json.dumps(dict(job=name, exit_status=int(status), peak_rss_gib=peak)))
PYEOF
exit $status
