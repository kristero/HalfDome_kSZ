#!/usr/bin/env python3
"""Check a local P0/beta Sobol design against the original nine-parameter priors.

The notebook is parsed as data, not executed. Its bounds must also reproduce
the original CSV prefix. This distinguishes generation bounds from later,
narrower inference priors without estimating either from sample extrema.
"""

import argparse
import ast
import csv
import json
import operator
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import qmc

from generate_so_two_param_sobol import BATTAGLIA12, PARAMETER_NAMES, sha256_file


PROJECT = Path(__file__).resolve().parents[1]


def read_prior_expression(node, constants):
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -read_prior_expression(node.operand, constants)
    if isinstance(node, ast.BinOp) and type(node.op) in operations:
        return operations[type(node.op)](read_prior_expression(node.left, constants),
                                         read_prior_expression(node.right, constants))
    if isinstance(node, (ast.Tuple, ast.List)):
        return [read_prior_expression(item, constants) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {read_prior_expression(k, constants): read_prior_expression(v, constants)
                for k, v in zip(node.keys, node.values)}
    raise ValueError("Unsupported prior expression: inspect notebook; do not execute it")


def notebook_priors(path):
    notebook = json.loads(path.read_text())
    found = []
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or "PRIORS = {" not in source:
            continue
        constants = {}
        for node in ast.parse(source).body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                name = node.targets[0].id
                if name.startswith("BATTAGLIA_") or name == "PRIORS":
                    constants[name] = read_prior_expression(node.value, constants)
        if "PRIORS" in constants:
            found.append((index, constants["PRIORS"]))
    if len(found) != 1:
        raise ValueError("Expected exactly one explicit nine-parameter PRIORS definition")
    return found[0]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-csv", type=Path, default=PROJECT / (
        "Sobol_tSZ/two_param_P0_beta_524288/battaglia_sobol_P0_beta_524288.csv"))
    parser.add_argument("--source-csv", type=Path,
                        default=PROJECT / "Sobol_tSZ/battaglia_sobol_1048576.csv")
    parser.add_argument("--source-notebook", type=Path,
                        default=PROJECT / "Sobol seq & GB applying.ipynb")
    args = parser.parse_args()
    cell_index, priors = notebook_priors(args.source_notebook)
    with args.source_csv.open(newline="") as stream:
        check(next(csv.reader(stream)) == PARAMETER_NAMES, "Source CSV column order mismatch")
    old_rows = np.loadtxt(args.source_csv, delimiter=",", skiprows=1, max_rows=1024)
    old_expected = qmc.scale(qmc.Sobol(d=9, scramble=True, seed=42).random_base2(10),
                            [priors[p][0] for p in PARAMETER_NAMES],
                            [priors[p][1] for p in PARAMETER_NAMES])
    check(np.array_equal(old_rows, old_expected), "Notebook priors/seed do not reproduce original CSV")

    with np.load(args.design_csv.with_suffix(".npz"), allow_pickle=False) as data:
        saved = {key: data[key] for key in data.files}
    meta = json.loads(args.design_csv.with_suffix(".json").read_text())
    n = meta["n_samples"]
    check(n >= 256 and n & (n - 1) == 0, "Expected a power-of-two design with N >= 256")
    check(int(saved["sequence_offset"]) == 0, "Prefix audit requires sequence offset zero")
    check(saved["theta_full"].shape == (n, 9), "Wrong full-theta shape")
    check(saved["theta"].shape == (n, 2), "Wrong target-theta shape")
    check(list(saved["param_names"]) == ["P0", "beta"], "Wrong target parameter names")
    check(list(saved["full_param_names"]) == PARAMETER_NAMES, "Wrong full parameter names")
    check(np.array_equal(saved["prior_low"], [priors[p][0] for p in ["P0", "beta"]]), "Lower bounds differ")
    check(np.array_equal(saved["prior_high"], [priors[p][1] for p in ["P0", "beta"]]), "Upper bounds differ")
    check(np.array_equal(saved["theta"], saved["theta_full"][:, [0, 2]]), "Target/full theta mismatch")
    check(np.array_equal(saved["theta"], qmc.scale(saved["sobol_unit"],
          saved["prior_low"], saved["prior_high"])), "Unit-to-physical scaling mismatch")
    for j, name in enumerate(PARAMETER_NAMES):
        if name not in ["P0", "beta"]:
            check(np.all(saved["theta_full"][:, j] == BATTAGLIA12[name]), f"{name} is not fixed to Battaglia12")
    check(np.all(saved["theta"] >= saved["prior_low"]), "Below-prior target values")
    check(np.all(saved["theta"] < saved["prior_high"]), "Above-prior target values")
    check(len(np.unique(saved["theta"], axis=0)) == n, "Duplicate target pairs")
    with args.design_csv.open(newline="") as stream:
        check(next(csv.reader(stream)) == PARAMETER_NAMES, "Generated CSV column order mismatch")
    csv_rows = np.loadtxt(args.design_csv, delimiter=",", skiprows=1)
    check(np.array_equal(csv_rows, saved["theta_full"]), "CSV/NPZ numerical mismatch")
    sizes = [2**m for m in range(8, n.bit_length())]
    for size in sizes:
        unit = qmc.Sobol(d=2, scramble=bool(saved["scramble"]), seed=int(saved["sobol_seed"]))
        check(np.array_equal(saved["sobol_unit"][:size], unit.random_base2(size.bit_length()-1)),
              f"Prefix N={size} does not reproduce saved sequence")
    grid_shape = [2**((n.bit_length()-1)//2), 2**((n.bit_length())//2)]
    cells = (saved["sobol_unit"] * grid_shape).astype(np.int64)
    counts = np.bincount(cells[:, 0]*grid_shape[1]+cells[:, 1], minlength=n)
    check(np.all(counts == 1), "Dyadic unit-square grid is not balanced")
    check(np.array_equal(saved["sobol_sequence_index"], np.arange(1, n+1)), "Sequence ordering mismatch")
    check(sha256_file(args.design_csv) == meta["csv_sha256"], "CSV checksum mismatch")
    report = dict(
        passed=True, n_rows=n, varying_parameters=["P0", "beta"],
        source_notebook=str(args.source_notebook.resolve()), source_cell_index_zero_based=cell_index,
        source_notebook_sha256=sha256_file(args.source_notebook),
        source_nine_parameter_csv=str(args.source_csv.resolve()), source_csv_sha256=sha256_file(args.source_csv),
        original_csv_prefix_reproduced_exactly=True, source_prefix_checked=1024,
        prior_source="original nine-parameter generation notebook, not narrower inference priors",
        original_generation_bounds=priors,
        fixed_parameters={p: BATTAGLIA12[p] for p in PARAMETER_NAMES if p not in ["P0", "beta"]},
        fixed_columns_exact=True, csv_npz_identical=True, unique_pairs=n,
        nested_prefix_sizes_checked=sizes, dyadic_grid_shape=grid_shape, dyadic_cells_one_point_each=True,
        source_dimension=9, new_dimension=2, seed=int(saved["sobol_seed"]),
        scramble=bool(saved["scramble"]), sequence_offset=0, scipy=scipy.__version__,
        generator_sha256=sha256_file(PROJECT / "SBI_analysis/generate_so_two_param_sobol.py"),
        design_csv_sha256=sha256_file(args.design_csv),
        scope="Parameter design only. No spectra or SBI constraints generated; train/test split not assigned.",
        comparison_note="Use common P0/beta evaluation bounds and RMSE normalization. Fixed nuisance parameters give conditional, not marginalized constraints.",
    )
    output = args.design_csv.parent / "design_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASSED {n:,} rows: original bounds, fixed parameters, uniqueness, CSV/NPZ identity and checksum")
    print("Nested Sobol prefixes:", sizes)
    print("One point in each unit-square dyadic cell; grid:", grid_shape)
    print("Saved audit:", output)


if __name__ == "__main__":
    main()
