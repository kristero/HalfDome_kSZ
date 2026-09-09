#!/usr/bin/env python3
"""Audit low NPE prior acceptance without retraining or completing evaluations.

Raw samples here are diagnostics, NOT prior-restricted posterior samples.
Use the environment that trained the pickles. Default sampling is small and
single-threaded; outputs go in a separate diagnostics directory.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from so_sbi_compression import METHODS, PARAM_NAMES, project, save_npz, write_json


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def summarize_raw(raw, low, high):
    if raw.ndim != 2 or raw.shape[1] != len(low) or not np.isfinite(raw).all():
        raise ValueError("Raw density samples have invalid shape or nonfinite entries")
    below, above = raw < low, raw > high
    inside = ~(below | above)
    normalized = (raw - low) / (high - low)
    q = np.quantile(normalized, [.05, .5, .95], axis=0)
    return dict(
        raw_count=len(raw), accepted_count=int(inside.all(axis=1).sum()),
        acceptance=float(inside.all(axis=1).mean()),
        per_parameter=[dict(param=name, below=float(below[:, j].mean()),
                            above=float(above[:, j].mean()),
                            prior_fraction_q05=float(q[0, j]),
                            prior_fraction_median=float(q[1, j]),
                            prior_fraction_q95=float(q[2, j]))
                       for j, name in enumerate(PARAM_NAMES)],
    )


def draw_raw(estimator, context, count, seed):
    import torch
    torch.manual_seed(seed)
    x = torch.tensor(np.asarray(context), dtype=torch.float32).reshape(1, -1)
    parameters = inspect.signature(estimator.sample).parameters
    blocks = []
    with torch.no_grad():
        for start in range(0, count, 2048):
            n = min(count - start, 2048)
            if "context" in parameters:
                block = estimator.sample(n, context=x)
            elif "condition" in parameters:
                block = estimator.sample(torch.Size([n]), condition=x)
            else:
                raise TypeError("Unknown estimator sampling interface")
            block = block.detach().cpu().numpy().reshape(-1, len(PARAM_NAMES))
            if len(block) != n:
                raise ValueError("Estimator returned an unexpected number of samples")
            blocks.append(block)
    return np.concatenate(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(
        "/lustre/work/kristero10/adrian_9param_compression_baseline_deproj0"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--raw-samples", type=int, default=20000)
    parser.add_argument("--reference-rows", type=int, default=50000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--indices", type=int, nargs="+")
    args = parser.parse_args()
    if min(args.raw_samples, args.reference_rows, args.threads) < 1:
        raise ValueError("Sampling, reference, and thread counts must be positive")
    root = args.run_root
    output = args.output_dir or root / "diagnostics/acceptance_audit"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "experiment.json").read_text())
    shared = arrays(root / "shared.npz")
    failures = {}
    for method in METHODS:
        path = root / method / "evaluation/sampling_failures.json"
        failures[method] = json.loads(path.read_text()) if path.exists() else []
    failed_indices = sorted({row["index"] for rows in failures.values() for row in rows})
    requested = args.indices or failed_indices
    controls = [int(config["example_index"])]
    controls += [int(idx) for idx in shared["test_indices"] if idx not in failed_indices][:1]
    query_indices = np.asarray(sorted(set(requested + controls)), dtype=np.int64)
    fit = shared["fit_indices"]
    validation = shared["validation_indices"]
    test = shared["test_indices"]
    if not np.isin(query_indices, test).all():
        raise ValueError("Requested queries must be in the actual held-out set")
    dataset_path = args.dataset or Path(config["dataset"])
    with np.load(dataset_path, allow_pickle=False) as data:
        theta_equal = np.array_equal(data["theta"], shared["theta"])
        mapping_equal = np.array_equal(data["sobol_global_row"], shared["sobol_global_row"])
        names_equal = tuple(str(x) for x in data["param_names"]) == PARAM_NAMES
        prior_equal = (np.array_equal(data["prior_low"], shared["low"])
                       and np.array_equal(data["prior_high"], shared["high"]))
        x_raw = data["x"]
    checks = dict(theta_equal=theta_equal, sobol_mapping_equal=mapping_equal,
                  param_names_equal=names_equal, prior_bounds_equal=prior_equal,
                  test_disjoint_from_pool=not bool(np.intersect1d(test, shared["pool_indices"]).size),
                  fit_validation_disjoint=not bool(np.intersect1d(fit, validation).size),
                  fit_is_pool_prefix=np.array_equal(fit, shared["pool_indices"][:len(fit)]),
                  validation_is_pool_suffix=np.array_equal(validation, shared["pool_indices"][len(fit):]))
    if not all(checks.values()):
        write_json(output / "failed_contract_checks.json", checks)
        raise ValueError(f"Dataset/partition mismatch: {checks}")
    import torch
    import sbi
    from scipy.spatial import cKDTree
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(config["seed"])
    reference = rng.choice(fit, size=min(args.reference_rows, len(fit)), replace=False)
    low, high = shared["low"], shared["high"]
    report = dict(experiment_id=config["experiment_id"], checks=checks,
                  python=sys.version, torch=str(torch.__version__), sbi=str(sbi.__version__),
                  failed_indices=failed_indices, queries=query_indices.tolist(),
                  sampling_failures=failures, results=[])
    print("Alignment/split checks:", checks, flush=True)
    print("Queries:", query_indices, "; failed:", failed_indices, flush=True)
    saved_heads = {}
    for method in METHODS:
        started = time.monotonic()
        transform = arrays(root / f"{method}_transform.npz")
        contexts = np.load(root / f"{method}_x.npy", mmap_mode="r")
        # Include both optimization/validation rows and all held-out rows in the
        # cache check; a query-only check cannot catch different training transforms.
        check_indices = np.unique(np.concatenate((reference[:128], validation[:128], test)))
        recomputed = project(x_raw[check_indices], transform)
        cache_matches = np.allclose(recomputed, contexts[check_indices], rtol=1e-6, atol=1e-6)
        max_error = float(np.max(np.abs(recomputed - contexts[check_indices])))
        if not cache_matches:
            write_json(output / f"{method}_context_mismatch.json", dict(max_abs_error=max_error))
            raise ValueError(f"{method}: cached context differs from the training transform")
        train_meta = json.loads((root / method / "training_complete.json").read_text())
        if train_meta["experiment_id"] != config["experiment_id"]:
            raise ValueError(f"{method}: model provenance mismatch")
        reference_values = np.asarray(contexts[reference], dtype=float)
        reference_low, reference_high = reference_values.min(axis=0), reference_values.max(axis=0)
        tree = cKDTree(reference_values)
        test_distance, _ = tree.query(contexts[test], k=1)
        neighbor_distance, neighbor = tree.query(contexts[query_indices], k=3)
        with (root / method / "density_estimator.pkl").open("rb") as stream:
            estimator = pickle.load(stream)
        estimator.eval()
        for q, idx in enumerate(query_indices):
            checkpoint = root / method / f"evaluation/profiles/row{idx}.npz"
            checkpoint_matches = None
            checkpoint_acceptance = None
            if checkpoint.exists():
                saved = arrays(checkpoint)
                checkpoint_matches = (str(saved["experiment_id"].item()) == config["experiment_id"]
                                      and str(saved["method"].item()) == method
                                      and np.array_equal(saved["context"], contexts[idx])
                                      and np.array_equal(saved["truth"], shared["theta"][idx]))
                if not checkpoint_matches:
                    raise ValueError(f"{method}, row {idx}: saved posterior conditioning mismatch")
                checkpoint_acceptance = float(saved["acceptance"])
            raw = draw_raw(estimator, contexts[idx], args.raw_samples, config["seed"] + int(idx))
            stats = summarize_raw(raw, low, high)
            neighbor_indices = reference[neighbor[q]]
            item = dict(method=method, index=int(idx),
                        was_failed=int(idx) in {row["index"] for row in failures[method]},
                        training=train_meta, context_cache_matches=bool(cache_matches),
                        context_cache_max_abs_error=max_error, checkpoint_matches=checkpoint_matches,
                        checkpoint_acceptance=checkpoint_acceptance,
                        truth=shared["theta"][idx].tolist(),
                        truth_prior_fraction=((shared["theta"][idx] - low) / (high - low)).tolist(),
                        nearest_reference_indices=neighbor_indices.tolist(),
                        nearest_reference_distances=neighbor_distance[q].tolist(),
                        nearest_reference_truths=shared["theta"][neighbor_indices].tolist(),
                        nearest_distance_test_percentile=float(100*np.mean(test_distance <= neighbor_distance[q, 0])),
                        context_outside_reference_marginal_range=(
                            (contexts[idx] < reference_low) | (contexts[idx] > reference_high)).tolist(),
                        **stats)
            report["results"].append(item)
            saved_heads[f"{method}_row{idx}"] = raw[:256]
            worst = sorted(stats["per_parameter"], key=lambda row: row["below"] + row["above"], reverse=True)[:3]
            summary = "; ".join(f"{p['param']}: below={p['below']:.1%}, above={p['above']:.1%}, "
                                f"median/prior={p['prior_fraction_median']:.3f}" for p in worst)
            print(f"{method} row={idx}: prior acceptance={stats['acceptance']:.5%}; "
                  f"input NN-distance percentile={item['nearest_distance_test_percentile']:.1f}; {summary}", flush=True)
            write_json(output / "acceptance_diagnostics.json", report)
        print(f"{method}: {time.monotonic() - started:.1f}s", flush=True)
    save_npz(output / "diagnostic_raw_sample_heads.npz", **saved_heads)
    print("Saved diagnostics:", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
