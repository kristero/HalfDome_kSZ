"""Recompute the completed catalogue/cache controls from saved spectra.

Run with the HalfDome scientific Python environment. No simulation is launched.
The two unbinned local MOPED matrices and independent held-out noise covariance
are fixed across comparisons. Outputs are separate from the September 21 logs.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import toml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "tsz_reuse_cache_20260921"
OLD = ROOT.parent / "tsz_8192_validation_recovery_20260921"
ELL = np.arange(80, 7980)
FACTOR = ELL * (ELL + 1) / (2 * np.pi)
CONTEXT = np.load(SOURCE / "followup/compression_context.npz")
WEIGHTS = [name for name in CONTEXT.files if "__" not in name]
INPUT_HASHES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dl(path):
    data = np.load(path)
    assert data.shape == (7980,) and np.isfinite(data).all(), path
    INPUT_HASHES[str(path.relative_to(ROOT.parent))] = sha(path)
    return data[ELL] * FACTOR


def distance(delta, samples):
    covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    keep = eigenvalues > max(eigenvalues[-1] * 1e-10, 1e-30)
    assert keep.all(), "Unexpected loss of held-out covariance rank"
    return float(np.linalg.norm((delta @ eigenvectors) / np.sqrt(eigenvalues)))


def compare(label, candidate, reference):
    delta = candidate - reference
    metrics = {name: distance(delta @ CONTEXT[name], CONTEXT[label + "__" + name])
               for name in WEIGHTS}
    return dict(label=label, relative_Dell_l2=float(np.linalg.norm(delta) / np.linalg.norm(reference)),
                max_fractional_Dell=float(np.max(np.abs(delta / reference))), moped=metrics,
                distance5=max(v for k, v in metrics.items() if "0.001" in k),
                distance9=max(v for k, v in metrics.items() if "1e-06" in k))


def timing(folder, task, group):
    status = json.loads((folder / "status.json").read_text())
    assert status["returncode"] == 0, folder
    data = toml.load(folder / "benchmark.toml")
    assert set(data["selected_halos_per_case"]) == {85224251}, folder
    assert data["ell_max"] == 7979 and data["output_nside"] == 4096
    text = (folder / "time.txt").read_text()
    rss = float(re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", text)[1]) / 2**20
    cpu = sum(float(re.search(key + r" time \(seconds\):\s+([\d.]+)", text)[1])
              for key in ("User", "System"))
    stage = data["timings"]
    return dict(group=group, id=task["id"], label=task["label"], raw_nside=data["raw_nside"],
                nodes=data["nodes"], profiles=data["cases"], process_seconds=status["seconds"],
                catalogue_seconds=data["catalogue_seconds"], painting_seconds=stage["painting"],
                read_seconds=stage["read_mass_redshift"] + stage["read_positions"],
                cache_seconds=stage["cache"], peak_RSS_GiB=rss, cpu_hours=cpu / 3600,
                mask_sha256=data["mask_sha256"], job=status["job"],
                extra_pixel_transforms=bool(task.get("pixel_targets")), stages=stage)


def audit_sources():
    checked = {}
    for relative in ("manifest.json", "edge_extension/manifest.json", "work_balance/manifest.json"):
        manifest = SOURCE / relative
        records = json.loads(manifest.read_text())["sha256"]
        for name, digest in records.items():
            path = manifest.parent / name
            assert sha(path) == digest, path
        checked[relative] = len(records)
    fetched = json.loads((SOURCE / "fetch_manifest.json").read_text())
    for name, digest in fetched.items():
        assert sha(SOURCE / name) == digest, name
    checked["fetched_products"] = len(fetched)
    return checked


def main():
    source_audit = audit_sources()
    plan = json.loads((SOURCE / "plan.json").read_text())["tasks"]
    cases = plan[2]["cases"]
    labels = [case["label"] for case in cases]
    default = {label: load_dl(SOURCE / "controls/002" / label / "masked_clean_cl.npy") for label in labels}
    smooth_reference = {label: load_dl(SOURCE / "edge_extension/controls/002" / label / "masked_clean_cl.npy")
                        for label in labels}
    times, errors, pixels, resolution = [], [], [], []
    spectra = {"ell": ELL}
    for task in plan:
        folder = SOURCE / "controls" / f"{task['id']:03d}"
        times.append(timing(folder, task, "original_exterior"))
        for case in task["cases"]:
            label = case["label"]
            value = load_dl(folder / label / "masked_clean_cl.npy")
            row = dict(group="original_exterior", task=task["id"], variant=task["label"],
                       nodes=task["nodes"], **compare(label, value, default[label]))
            errors.append(row)
            if task['nside'] == 8192:
                errors.append(dict(group="common_reference", task=task["id"], variant=task["label"],
                                   nodes=task["nodes"], **compare(label, value, smooth_reference[label])))
            spectra[f"original_{task['id']:03d}__{label}"] = value
            for parent in task.get("pixel_targets", []):
                for dewindow in (False, True):
                    suffix = f"_pixel{parent}" + ("_dewindow" if dewindow else "")
                    pixel_value = load_dl(folder / (label + suffix) / "masked_clean_cl.npy")
                    pixels.append(dict(fine_nside=task["nside"], parent_nside=parent, dewindow=dewindow,
                                       **compare(label, pixel_value, value)))
                    spectra[f"pixel_{task['nside']}_{parent}_{int(dewindow)}__{label}"] = pixel_value
    for task in json.loads((SOURCE / "edge_extension/plan.json").read_text()):
        folder = SOURCE / "edge_extension/controls" / f"{task['id']:03d}"
        times.append(timing(folder, task, "smooth_exterior"))
        for label in labels:
            value = load_dl(folder / label / "masked_clean_cl.npy")
            errors.append(dict(group="smooth_exterior", task=task["id"], variant=task["label"],
                               nodes=task["nodes"], **compare(label, value, smooth_reference[label])))
            spectra[f"smooth_{task['id']:03d}__{label}"] = value
    task = json.loads((SOURCE / "work_balance/plan.json").read_text())[0]
    folder = SOURCE / "work_balance/controls/000"
    times.append(timing(folder, task, "balanced"))
    for case in cases:
        label, identifier = case["label"], case["reference_id"]
        value = load_dl(folder / label / "masked_clean_cl.npy")
        errors.append(dict(group="balanced", task=0, variant="greedy256", nodes=task["nodes"],
                           **compare(label, value, default[label])))
        old_value = load_dl(OLD / "controls" / f"{identifier:03d}" / "masked_clean_cl.npy")
        assert compare(label, default[label], old_value)["relative_Dell_l2"] < 1e-10
        coarse_path = (OLD / "controls" / f"{identifier:03d}" / "paired4096_clean_cl.npy" if identifier < 2
                       else OLD / "controls" / f"{identifier + 1:03d}" / "masked_clean_cl.npy")
        coarse = load_dl(coarse_path)
        spectra[f"centre4096__{label}"] = coarse
        resolution.append(dict(pair="4096 vs 8192", **compare(label, coarse, default[label])))
        errors.append(dict(group="boundary_change", task=0, variant="old_default_vs_smooth_double",
                           nodes=[512, 256, 128], **compare(label, default[label], smooth_reference[label])))
    for task_id in (11, 12):
        label = plan[task_id]["cases"][0]["label"]
        fine = spectra[f"original_{task_id:03d}__{label}"]
        resolution.append(dict(pair="8192 vs 16384", **compare(label, default[label], fine)))
        # Compare the two quadrature refinements against the SAME fine reference.
        for fine_nside in (8192, 16384):
            for dewindow in (False, True):
                value = spectra[f"pixel_{fine_nside}_4096_{int(dewindow)}__{label}"]
                pixels.append(dict(fine_nside=fine_nside, parent_nside=4096, dewindow=dewindow,
                                   common_reference_nside=16384, **compare(label, value, fine)))
    # Existing FL reference uses the same numerical contract; identify it by plan.
    reference_root = ROOT.parent / "tsz_spherical_preflight_20260920"
    fine_path = reference_root / "fullsky/FL_L1_m9/nside16384_grid1/masked_clean_cl.npy"
    if fine_path.exists():
        fine = load_dl(fine_path)
        spectra["reference16384__FL_L1_m9"] = fine
        resolution.append(dict(pair="8192 vs 16384", **compare("FL_L1_m9", default["FL_L1_m9"], fine)))
    assert len(times) == 17 and len({t["mask_sha256"] for t in times}) == 1
    INPUT_HASHES[str((SOURCE / "followup/compression_context.npz").relative_to(ROOT.parent))] = sha(SOURCE / "followup/compression_context.npz")
    report = dict(utc=datetime.now(timezone.utc).isoformat(), audited=source_audit, completed_controls=len(times),
                  ell_min=80, ell_max=7979, cases=cases, timings=times, errors=errors,
                  resolution=resolution, pixel_quadrature=pixels, input_sha256=INPUT_HASHES,
                  metric="Maximum across two local unbinned MOPED anchors; held-out conditional SO noise",
                  diagnostic_256_submitted=False, bright_extreme_tolerance="User permits larger errors; no prior rejection")
    (ROOT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(ROOT / "spectra.npz", **spectra)
    for name, rows in (("timings", times), ("errors", errors), ("resolution", resolution), ("pixel_quadrature", pixels)):
        fields = list(dict.fromkeys(k for row in rows for k, v in row.items() if not isinstance(v, dict)))
        with (ROOT / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print("Verified:", source_audit, "full controls:", len(times))
    for t in times:
        print(t["group"], t["id"], t["label"], "paint/process/cache/RSS:",
              *(round(t[k], 3) for k in ("painting_seconds", "process_seconds", "cache_seconds", "peak_RSS_GiB")))
    for group in ("original_exterior", "smooth_exterior", "boundary_change"):
        for row in errors:
            if row["group"] == group and (row["task"] in range(3, 8) or group != "original_exterior"):
                print(group, row["task"], row["label"], "d5,d9,relative:",
                      *(f"{row[k]:.6g}" for k in ("distance5", "distance9", "relative_Dell_l2")))
    print("Resolution:", [(r["pair"], r["label"], r["distance5"], r["distance9"]) for r in resolution])


if __name__ == "__main__":
    main()
