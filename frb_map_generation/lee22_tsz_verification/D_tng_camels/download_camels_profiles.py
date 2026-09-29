#!/usr/bin/env python
"""Download the public CAMELS IllustrisTNG CV precomputed 3D gas profiles (illstack_CAMELS, Moser+22).

Files: https://users.flatironinstitute.org/~camels/Profiles/IllustrisTNG/CV/CV_<i>/IllustrisTNG_CV_<i>_NNN.hdf5
NNN is the OLD 34-snapshot numbering (033 = z = 0); the raw snapshots use the new 91-snapshot numbering,
old i <-> new [14, 18, 24, 28, 32, 34, 36, ..., 90][i] (= 24 + 2 i for i >= 4).
We take old snapshots 010..033 (z = 2.00 .. 0) for all 27 CV boxes (~0.75 MB each, ~480 MB total).

Usage: python download_camels_profiles.py [outdir]
"""
import os
import sys
import time
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_tsz_verification_20260929/D_tng_camels/camels_profiles"
BASE = "https://users.flatironinstitute.org/~camels/Profiles/IllustrisTNG/CV"
SNAPS = [f"{i:03d}" for i in range(10, 34)]
BOXES = range(27)

os.makedirs(OUT, exist_ok=True)
t0 = time.time()
n = 0
for b in BOXES:
    d = os.path.join(OUT, f"CV_{b}")
    os.makedirs(d, exist_ok=True)
    for s in SNAPS:
        fn = f"IllustrisTNG_CV_{b}_{s}.hdf5"
        dst = os.path.join(d, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 10000:
            continue
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(f"{BASE}/CV_{b}/{fn}", dst + ".part")
                os.replace(dst + ".part", dst)
                n += 1
                break
            except Exception as e:  # noqa: BLE001
                print("retry", fn, e, flush=True)
                time.sleep(3)
    print(f"CV_{b} done ({n} files, {time.time() - t0:.0f} s)", flush=True)
