"""Engine timing for ~50 profile variants (Step 0.1, task 4).

50 variants on the shared fine 4R grid (1033 nodes) and the 83 original
multipoles: 25 paired bootstrap draws of the B12 pressure fit through 4R
(fit_parameters.npz) and 25 of the B16 electron fit through 4R (radii234
products/*_parameters.npz), each with the common support and published
fallback. Electron variants include the DM moments (mean, compound-Poisson
variance, DM-field C_ell), the most expensive part. Writes timing.json and
timing_bootstrap_outputs.npz (a by-product, not an uncertainty analysis).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_engine", HERE/"engine.py")
E = importlib.util.module_from_spec(spec)
sys.modules["kernel_engine"] = E
spec.loader.exec_module(E)

import numpy as np

DRAWS = 25
WORKERS = 8


def electron_bootstrap():
    """bootstrap[bin, draw, 3] of the 4R electron fit (sphere 2, field 1)."""
    bins = json.loads(E.SOURCES["study_summary"].read_text())["bins"]
    draws = []
    for case in bins:
        saved = np.load(E.RADII234/"products"/f"{case['label']}_parameters.npz")["bootstrap"]
        draws.append(saved[2, 1])
    return np.array(draws)


def main():
    started = time.perf_counter()
    support = E.support_definition()
    grid = E.make_grid(4., breaks=list(E.SHELL_EDGES[1:])+[3.6])
    pressure = np.load(E.SOURCES["fit_parameters"])["bootstrap"][:, 3]     # [bin, draw, 3]
    electrons = electron_bootstrap()
    variants = []
    # Unsupported diagnostic bins can hold NaN draws; they always use the fallback.
    clean = lambda draw: np.where(support["support"][:, None], draw, np.nan_to_num(draw, nan=1.))
    for d in range(DRAWS):
        variants.append(E.Variant(f"P4_boot{d:03d}", E.FitProfile(0, clean(pressure[:, d]), "P4", support["mass_edges"],
                                                                    support["support"], support["max_logmass"]), grid, 4.))
        parameters = clean(electrons[:, d])
        variants.append(E.Variant(f"E4_boot{d:03d}", E.FitProfile(1, parameters, "E4", support["mass_edges"],
                                                                    support["support"], support["max_logmass"]), grid, 4.))
    cpu_children_before = os.times().children_user+os.times().children_system
    result = E.Engine().run(variants, ell=E.ELL, workers=WORKERS)
    cpu_children = os.times().children_user+os.times().children_system-cpu_children_before
    labels = list(result["labels"])
    p = [labels.index(f"P4_boot{d:03d}") for d in range(DRAWS)]
    e = [labels.index(f"E4_boot{d:03d}") for d in range(DRAWS)]
    np.savez_compressed(HERE/"timing_bootstrap_outputs.npz", ell=result["ell"], labels=result["labels"],
                        cl_1h=result["cl_1h"], cl_2h=result["cl_2h"], mean=result["mean"],
                        dm_mean=result["dm_mean"], dm_var=result["dm_var"],
                        note="first 25 paired mean-fit bootstrap draws per field, 4R fit and cut; timing by-product only")
    i100 = int(np.flatnonzero(E.ELL == 100.)[0])
    timing = dict(variants=len(variants), pressure_variants=DRAWS, electron_variants_with_DM_moments=DRAWS,
                  radial_nodes=len(grid), multipoles=len(E.ELL), workers=WORKERS,
                  wall_seconds_engine=result["timing"]["wall_seconds"], setup_seconds=result["timing"]["setup_seconds"],
                  worker_cpu_seconds=cpu_children, wall_seconds_total=time.perf_counter()-started,
                  peak_RSS_GiB_parent=result["timing"]["peak_RSS_GiB_parent"],
                  peak_RSS_GiB_largest_worker=result["timing"]["peak_RSS_GiB_largest_child"],
                  by_product=dict(P4_bootstrap_total_Cl_ell100_fractional_sd=float(np.std(result["cl_total"][p, i100])/np.mean(result["cl_total"][p, i100])),
                                  E4_bootstrap_mean_DM_fractional_sd=float(np.std(result["dm_mean"][e])/np.mean(result["dm_mean"][e])),
                                  E4_bootstrap_DM_variance_fractional_sd=float(np.std(result["dm_var"][e])/np.mean(result["dm_var"][e])),
                                  note="25 draws only; conditional mean-fit sensitivity, not a calibrated uncertainty"))
    E.write_json(HERE/"timing.json", timing)
    print(json.dumps(E.C.plain(timing), indent=1), flush=True)


if __name__ == "__main__":
    main()
