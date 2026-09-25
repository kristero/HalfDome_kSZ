"""Validate completed refits and export small artifacts without pixel maps."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

from prior import JointPrior
import report


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    assert json.loads((root / "audit/ell_integration_check.json").read_text())["passed"]
    prior = JointPrior(json.loads((root / "code/prior.json").read_text()))
    manifest = json.loads((root / "manifest.json").read_text())
    for folder, key in (("code", "code_sha256"), ("audit", "input_sha256")):
        for filename, expected in manifest[key].items():
            assert digest(root / folder / filename) == expected, filename
    checks = []
    for variant in report.VARIANTS:
        fit = json.loads((root / "results" / (variant + "_fit.json")).read_text())
        support, metrics = prior.contains(fit["best"]["theta"], return_metrics=True)
        assert support, variant
        report.write_json(root / "results" / (variant + "_support.json"),
                          dict(inside_current_joint_prior=support, metrics=metrics))
        for candidate in fit["candidates"]:
            assert prior.contains(candidate["theta"])
            assert candidate["response_resolution_error"] < .005
            if candidate["name"] == "amplitude_from_existing_map":
                continue
            name = candidate["name"]
            status = json.loads((root / "results" / (name + "_status.json")).read_text())
            assert status["exit_code"] == 0
            assert status["numerics"]["column_checks"] == 108
            assert status["numerics"]["maximum_column_relative_error"] < 1e-7
            np.testing.assert_array_equal(status["theta"], candidate["theta"])
            with np.load(root / "results" / (name + "_full_cl.npz")) as spectra:
                for key in spectra.files:
                    assert spectra[key].shape == (7980,)
                    assert np.isfinite(spectra[key]).all()
            proposal = json.loads((root / "proposals" / (name + ".json")).read_text())
            checks.append(dict(name=name, column_checks=108,
                maximum_column_relative_error=status["numerics"]["maximum_column_relative_error"],
                peak_rss_GiB=status["peak_rss_GiB"], elapsed_seconds=status["elapsed_seconds"],
                optimizer_success=proposal["success"], optimizer_message=proposal["message"],
                response_resolution_error=candidate["response_resolution_error"]))
    report.write_json(root / "audit/final_checks.json", dict(passed=True, maps=checks,
        scope="Full-map numerics and prior support; cosmology transfer remains approximate"))
    report.main(root, report.DEFAULT_CAMPAIGN, report.DEFAULT_PILOT)

    paths = []
    for folder in ("code", "audit", "results", "proposals", "logs", "plots"):
        paths.extend(path for path in (root / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    paths.extend(root.glob("*.json"))
    paths.append(root / "README.md")
    paths = sorted(set(paths) - {root / "artifact_manifest.json"})
    artifacts = {str(path.relative_to(root)): dict(sha256=digest(path), bytes=path.stat().st_size)
                 for path in paths}
    report.write_json(root / "artifact_manifest.json", artifacts)
    archive = root / "cosmology_refit_results.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for path in paths + [root / "artifact_manifest.json"]:
            stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size,
                          sha256=digest(archive), files=len(artifacts)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
