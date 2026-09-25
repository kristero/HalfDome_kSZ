"""Export a complete ordered dataset and matching FLAMINGO observations.

Requires every immutable row. Writes separate clean and independent-noise arrays;
does not apply an old emulator, compression matrix, or trained SBI posterior.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import toml

from prior import digest, write_json
from worker import BINNED_NAMES, EDGES, SPECTRA, bin_cl, verify_frozen


def observations(root, run):
    output = root/"observations"
    for name in ("HalfDome","L1_m9","fgas-8sigma","Mstar-1sigma"):
        source = Path(run["campaign"])/"results"/name
        values = {BINNED_NAMES[key]: bin_cl(np.load(source/(key+".npy"))) for key in SPECTRA}
        np.savez(output/(name+".npz"),bin_edges=EDGES,**values)
        write_json(output/(name+".json"),dict(source=str(source),
            source_sha256={key:digest(source/(key+".npy")) for key in SPECTRA},
            operator=toml.load(str(source/"complete.toml")),
            inference_status="observation only; new-prior training and coverage validation required"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--observations-only",action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    verify_frozen(root)
    run = json.loads((root/"run_config.json").read_text())
    observations(root,run)
    if args.observations_only:
        return
    n = run["count"]
    count = (n+run["chunk_size"]-1)//run["chunk_size"]
    paths = [root/"chunks"/("chunk_%05d.h5"%i) for i in range(count)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError("Dataset incomplete: %d missing chunks; first=%s"%(len(missing),missing[0]))
    output = root/"dataset"
    output.mkdir(exist_ok=True)
    expected = np.load(root/"design/theta.npy",mmap_mode="r")
    arrays = {BINNED_NAMES[key]:np.lib.format.open_memmap(
        output/(BINNED_NAMES[key]+".tmp.npy"),mode="w+",dtype=np.float64,shape=(n,40)) for key in SPECTRA}
    seen = np.zeros(n,dtype=bool)
    for path in paths:
        with h5py.File(path,"r") as handle:
            assert handle.attrs["manifest_sha256"] == digest(root/"manifest.json")
            for key,row in handle.items():
                index = int(key)
                assert 0 <= index < n and not seen[index]
                np.testing.assert_array_equal(row["theta"][:],expected[index])
                assert row.attrs["complete"]
                for name,array in arrays.items():
                    values = row[name][:]
                    assert values.shape == (40,) and np.isfinite(values).all()
                    array[index] = values
                seen[index] = True
    assert seen.all()
    for name,array in arrays.items():
        array.flush()
    arrays.clear()
    for path in output.glob("*.tmp.npy"):
        path.replace(output/path.name.replace(".tmp.npy",".npy"))
    np.save(output/"theta.npy",expected)
    np.save(output/"noise_split_seeds.npy", np.load(root/"design/noise_split_seeds.npy"))
    np.save(output/"bin_edges.npy",EDGES)
    write_json(output/"complete.json",dict(rows=n,manifest_sha256=digest(root/"manifest.json"),
        files={p.name:digest(p) for p in output.glob("*.npy")},
        theta_coordinates="physical parameters in frozen parameter_order",
        observable="dimensionless y^2, mode-count-weighted linear D_ell, 40 bins",
        prior="Independent linear BoxUniform in all nine physical coordinates; bounds in manifest.json",
        noise="independent SO noise in each row and each split; fixed sky/catalogue and mask"))


if __name__ == "__main__":
    main()
