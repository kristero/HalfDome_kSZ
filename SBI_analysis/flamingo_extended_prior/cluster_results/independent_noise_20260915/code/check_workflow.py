"""Fast integration checks of restart, ordering and corruption handling.

The simulator is deliberately synthetic here. Astronomical validation is the
separate six-full-map gate; this test only exercises dataset bookkeeping.
Run on Linux with --root pointing at the prepared campaign.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    campaign = args.root.resolve()
    sys.path.insert(0,str(campaign/"code"))
    import worker
    import collect
    from prior import JointPrior, digest, write_json
    manifest = json.loads((campaign/"manifest.json").read_text())
    prior = JointPrior(manifest["prior"])
    theta = np.load(campaign/"design/theta.npy")[:4]
    calls = []
    failed_once = [False]

    def synthetic_row(root,run,values,label):
        index = int(label.split("_")[-1])
        calls.append(index)
        if index == 2 and not failed_once[0]:
            failed_once[0] = True
            raise RuntimeError("deliberate interrupted row")
        spectra = {name: np.full(7980,(index+1)*1e-15) for name in worker.SPECTRA}
        return spectra,{"synthetic":True,"exit_code":0,"elapsed_seconds":0.},b"synthetic row\n"

    with tempfile.TemporaryDirectory(prefix="extended_restart_test_") as temp:
        root = Path(temp)
        for folder in ("chunks","logs","locks","scratch","preflight","design","observations"):
            (root/folder).mkdir()
        np.save(root/"design/theta.npy",theta)
        np.save(root/"design/noise_split_seeds.npy",np.load(campaign/"design/noise_split_seeds.npy")[:4])
        run = json.loads((campaign/"run_config.json").read_text())
        run.update(count=4,chunk_size=4)
        write_json(root/"run_config.json",run)
        write_json(root/"manifest.json",{"prior":prior.config})
        fixture_manifest = json.loads((root/"manifest.json").read_text())
        write_json(root/"preflight/quality_gate.json",dict(passed=True,manifest_sha256=digest(root/"manifest.json")))
        def invoke_worker():
            with patch.object(sys,"argv",["worker.py","--root",str(root),"--chunk","0"]):
                worker.main()
        with patch.object(worker,"verify_frozen",return_value=fixture_manifest), \
             patch.object(worker,"run_row",side_effect=synthetic_row),contextlib.redirect_stdout(io.StringIO()):
            try:
                invoke_worker()
            except RuntimeError as error:
                assert "deliberate interrupted row" in str(error)
            else:
                raise AssertionError("Expected the injected failure")
            assert len(list((root/"chunks/chunk_00000").glob("row_*.h5"))) == 2
            invoke_worker()
            assert calls == [0,1,2,2,3],calls
            invoke_worker()
            assert calls == [0,1,2,2,3],calls
        assert not list((root/"chunks").glob("chunk_*/*.h5"))
        with patch.object(collect,"verify_frozen",return_value=fixture_manifest), \
             patch.object(collect,"observations"), \
             patch.object(sys,"argv",["collect.py","--root",str(root)]):
            collect.main()
            np.testing.assert_array_equal(np.load(root/"dataset/theta.npy"),theta)
            x = np.load(root/"dataset/masked_clean_dl40.npy")
            np.testing.assert_allclose(x/x[0],np.arange(1,5)[:,None]*np.ones((1,40)))
            # Corruption must be detected, not silently collected.
            with h5py.File(root/"chunks/chunk_00000.h5","r+") as handle:
                handle["0/theta"][0] += 1
            try:
                collect.main()
            except AssertionError:
                pass
            else:
                raise AssertionError("Corrupted parameter row was accepted")
    write_json(campaign/"workflow_check.json",dict(passed=True,
        synthetic_forward_model=True,failed_row_retried_without_replacement=True,
        successful_rows_not_recomputed=True,packed_chunk_resume_passed=True,
        collector_ordering_passed=True,corrupted_parameters_rejected=True))
    print("Workflow restart, ordering and corruption checks passed")


if __name__ == "__main__":
    main()
