"""Create an immutable comparison snapshot and explicit tail experiments locally."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
B12 = [18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415]


def prepare():
    baseline = HERE / "baseline"
    inputs = HERE / "inputs"
    baseline.mkdir(exist_ok=True)
    inputs.mkdir(exist_ok=True)
    for name in ("prior.py", "prior.json", "stable_los.jl"):
        shutil.copy2(BASE / "flamingo_linear_prior" / name, baseline / name)
    shutil.copy2(BASE / "flamingo_linear_prior/cluster_results/manifest.json", inputs / "active_manifest.json")
    refit = BASE / "flamingo_cosmology_refit/cluster_results/results/comparison_summary.json"
    shutil.copy2(refit, inputs / "flamingo_cosmology_refits.json")
    fits = json.loads(refit.read_text())["corrected_parameters"]
    cases = {"Battaglia12": B12}
    cases.update({"FL_"+name: values for name, values in fits.items()})
    for name, index, values in (
        ("xc_low", 1, [.1, .05, .025]), ("amP0_low", 3, [-.2, -.4, -.6]),
        ("amxc_low", 4, [-.6, -.8, -1.]), ("azP0_low", 6, [-4.5, -5.25, -6.]),
        ("azxc_high", 7, [2., 2.5, 3.]), ("azbeta_high", 8, [1.5, 1.75, 2.])):
        for i, value in enumerate(values):
            theta = B12.copy()
            theta[index] = value
            cases[name+"_"+str(i)] = theta
    combined = B12.copy()
    combined[1], combined[3], combined[4], combined[6], combined[7], combined[8] = .05, -.4, -.8, -5.25, 2.5, 1.75
    cases["combined_tails"] = combined
    # Factor ten pressure changes isolate whether arbitrary Y limits track failure.
    for name, p0 in (("P0_tiny", .01), ("P0_huge", 6000.)):
        theta = B12.copy()
        theta[0] = p0
        cases[name] = theta
    for name, beta in (("beta_near_energy", 2.7001), ("beta_steep", 128.)):
        theta = B12.copy()
        theta[2], theta[5], theta[8] = beta, 0., 0.
        cases[name] = theta
    (inputs / "cases.json").write_text(json.dumps(cases, indent=2)+"\n")
    hashes = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*baseline.glob("*"), *inputs.glob("*")] if p.is_file()}
    (inputs / "snapshot_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    with tarfile.open(HERE / "source.tar.gz", "w:gz") as archive:
        for path in HERE.rglob("*"):
            if path.is_file() and path.suffix not in (".gz", ".pyc") and "__pycache__" not in path.parts:
                archive.add(path, arcname=str(path.relative_to(HERE)))


if __name__ == "__main__":
    prepare()
