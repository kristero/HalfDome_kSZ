"""Download one FLAMINGO L2p8_m9 integrated Compton-y map (Yang+26, NSIDE 4096) from the COSMA service.

The file holds a single dataset `data`: float64, RING ordering, 201,326,592 pixels (1.61 GB).
It is streamed in 16M-pixel slices with retries and written to a .npy memmap, so an interrupted
download resumes from the last completed slice (progress kept in a sidecar JSON).

Usage (flamingo venv):  python download_flamingo_tsz_map.py --lightcone 0
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import flamingo_reduced_io as fio  # noqa: E402

BASE = "FLAMINGO/L2p8_m9/L2p8_m9/integrated_maps/yang26"
NPIX = 12 * 4096**2
SLICE = 16 * 1024**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightcone", type=int, default=0)
    parser.add_argument("--map", default="lensed_tSZ_rot_same_rot")
    parser.add_argument("--out-dir", default=str(HERE / "flamingo_maps"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"L2p8_m9_lc{args.lightcone}_{args.map}_nside4096.npy"
    state_path = target.with_suffix(".json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done_slices": []}
    counter = fio.install_byte_counter()
    remote = fio.connect()
    dataset = remote[f"{BASE}/lightcone{args.lightcone}_shells/{args.map}.hdf5"]["data"]
    assert dataset.shape == (NPIX,), dataset.shape
    mode = "r+" if target.exists() else "w+"
    out = np.lib.format.open_memmap(target, mode=mode, dtype=np.float64, shape=(NPIX,))
    started = time.monotonic()
    for k, start in enumerate(range(0, NPIX, SLICE)):
        if k in state["done_slices"]:
            continue
        stop = min(start + SLICE, NPIX)
        out[start:stop] = fio.retry(lambda: dataset[start:stop], what=f"slice {k}")
        out.flush()
        state["done_slices"].append(k)
        state["source"] = f"{BASE}/lightcone{args.lightcone}_shells/{args.map}.hdf5:data"
        state_path.write_text(json.dumps(state))
        rate = counter["bytes"] / 1e6 / max(time.monotonic() - started, 1e-9)
        print(f"slice {k + 1}/{-(-NPIX // SLICE)} done, {counter['bytes'] / 1e9:.2f} GB this session, {rate:.2f} MB/s", flush=True)
    finite = np.isfinite(out)
    state.update(complete=True, mean=float(out[finite].mean()), finite_fraction=float(finite.mean()),
                 minimum=float(out[finite].min()), maximum=float(out[finite].max()))
    state_path.write_text(json.dumps(state, indent=1))
    print("DONE", target, {k: v for k, v in state.items() if k != "done_slices"}, flush=True)


if __name__ == "__main__":
    main()
