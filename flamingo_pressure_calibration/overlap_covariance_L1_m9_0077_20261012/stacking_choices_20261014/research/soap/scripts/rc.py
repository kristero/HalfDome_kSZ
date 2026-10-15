"""Thin local wrapper around the study's remote_control.remote (read-only maas probes)."""
import json, sys, time
sys.path.insert(0, '/home/cbllover/HalfDome/flamingo_pressure_calibration/overlap_covariance_L1_m9_0077_20261012')
from remote_control import remote

PY = '/home/nagainis/HalfDome/flamingo_pressure_calibration/profile_test_L1_m9_0077_20261007/venv/bin/python'


def run(src, timeout=300):
    t0 = time.time()
    out = remote(src, python=PY, timeout=timeout)
    dt = time.time() - t0
    i = out.find('{')
    obj, _ = json.JSONDecoder().raw_decode(out[i:])
    return obj, dt, out
