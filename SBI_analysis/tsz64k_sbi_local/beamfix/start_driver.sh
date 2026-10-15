#!/bin/bash
# Start driver.py in its own session (survives the shell), log to driver.log, PID in driver.pid.
# Stop it with: kill -TERM "$(cat driver.pid)"   (never by pgrep/pkill on a pattern: the calling shell can match it)
cd "$(dirname "$0")"
if [ -s driver.pid ] && kill -0 "$(cat driver.pid)" 2>/dev/null; then echo "driver already running (pid $(cat driver.pid))"; exit 1; fi
rm -f relay.lock
setsid nohup /home/cbllover/miniconda3/envs/halfdome/bin/python driver.py "$@" >> driver.log 2>&1 < /dev/null &
echo $! > driver.pid
echo "driver started (pid $(cat driver.pid)) with: $*"
