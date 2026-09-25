"""Check the multipole quadrature without changing the frozen fitting code.

The fitting response uses three Gauss nodes per band. This audit loads an
isolated in-memory copy of that source with only the node count and matching
reshape dimension changed. The original file and all fit results stay intact.
"""
import argparse
import json
from pathlib import Path
import types

import numpy as np

from prior_model import FIDUCIAL, save_json
from report import VARIANTS


def refined_response_class(root, nodes):
    source_path = root / "code/halo_response.py"
    source = source_path.read_text()
    replacements = {"leggauss(3)": "leggauss({})".format(nodes),
                    ".reshape(40,3)": ".reshape(40,{})".format(nodes)}
    for old, new in replacements.items():
        assert source.count(old) == 1, "Frozen source structure changed: " + old
        source = source.replace(old, new)
    module = types.ModuleType("isolated_multipole_quadrature_audit")
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path) + " [ell audit]", "exec"), module.__dict__)
    return module.CosmologyResponse


def main(root, nodes):
    assert nodes > 3
    response = refined_response_class(root, nodes)(root, nmass=48, nz=36, nradial=256)
    references = dict(np.load(root / "audit/reference_responses.npz"))
    points = [("Battaglia12", FIDUCIAL, references["Battaglia12"])]
    for variant in VARIANTS:
        fit = json.loads((root / "results" / (variant + "_fit.json")).read_text())
        points.append((variant + "/previous", fit["original"]["theta"], references[variant]))
        for candidate in fit["candidates"]:
            name = candidate["name"]
            filename = variant + "_amplitude" if name == "amplitude_from_existing_map" else name
            data = np.load(root / "results" / (filename + ".npz"))
            points.append((variant + "/" + name, data["theta"], data["response"]))
    errors, values = {}, {}
    for name, theta, old in points:
        values[name] = response(theta)
        errors[name] = float(np.max(abs(old/values[name]-1)))
        print(name, "maximum relative multipole integration change", errors[name], flush=True)
    maximum = max(errors.values())
    result = dict(passed=maximum < .005, original_nodes=3, refined_nodes=nodes,
                  maximum_relative_change=maximum, per_candidate=errors,
                  scope="Numerical response sensitivity, not a cosmological modeling error")
    save_json(root / "audit/ell_integration_check.json", result)
    np.savez(root / "audit/ell_refined_responses.npz", **values)
    assert result["passed"], "Refine the multipole integration before finalizing"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=9)
    args = parser.parse_args()
    main(args.root, args.nodes)
