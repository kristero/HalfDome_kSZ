"""Observable Jacobians dO/dln F per radial shell and 0.2-dex mass bin.

J_O[s, m] = dO/d eps for F -> F (1 + eps) inside radial shell s for haloes in
mass bin m (all redshifts; the profile there is the bin's own fit, or the
published fallback outside the calibrated support). Profiles: P2, P4 (B12
pressure fits through 2R and 4R) and E2, E4 (B16 electron fits through 2R and
4R), hard-truncated at 4R on a fine grid whose segment breaks are the x200
shell edges and the eight coarse importance edges.

Observables (ell: the 106 extended multipoles, 20..2e4, incl. the 83 original):
  P: yy 1h / linear 2h / total, mean y
  E: tau 1h / 2h, kSZ 1h (catalogue v_LOS), kSZ 1h (linear sigma_v), kSZ 2h (approx.),
     DM-field 1h / 2h (z_s=1), mean tau, mean DM, DM variance, DM quantiles
     (16, 50, 84, 97.5, 99.9%) of the compound-Poisson per-sightline PDF
  P x E: y x DM 1h / 2h, with respect to the pressure and the electron shells
  Y(<x), N_e(<x): z=0 segment integrals per bin centre (dY(<x_k)/dlnF_s = I_s 1[s<k])
Quadratic auto-spectra and the DM variance: sum over (s, m) of J = 2 O.
Cross spectra (each side), means and quantiles: sum of J = O.

Gates (jacobians_verification.json): exact sum rules, and central finite
differences of single (shell, mass-bin) perturbations on a grid with
sub-micro nodes at the shell boundaries.
Outputs: jacobians.npz (schema in its 'schema' entry), jacobians_engine_output.npz.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_engine", HERE/"engine.py")
E = importlib.util.module_from_spec(spec)
sys.modules["kernel_engine"] = E
spec.loader.exec_module(E)

import numpy as np

WORKERS = 8
X200_EDGES = np.unique(np.r_[0, np.geomspace(.025, 3, 31), np.geomspace(.04, 1.34, 28), 1, 1.5, 1.75, 2, 2.5,
                             np.geomspace(3, 4, 4)])
COARSE_EDGES = E.SHELL_EDGES
PROFILES = ("P2", "P4", "E2", "E4")
PAIRS = (("P2", "E2"), ("P4", "E4"))
FD_TESTS = ((.45, 10), (1.6, 7))       # (x inside the tested segment, mass bin index)
FD_EPS = .1
FD_DELTA = 1e-6


def jacobian_grid(extra=()):
    breaks = np.unique(np.r_[X200_EDGES[1:], COARSE_EDGES[1:]])
    return E.make_grid(4., breaks=breaks, linear=[(x, x, 1) for x in extra])


def segment_map(grid, edges):
    """Index of the shell (between consecutive edges) containing each segment."""
    mids = np.array([.5*(grid.x[a]+grid.x[b]) for a, b in grid.segments])
    return np.searchsorted(edges, mids)-1


def aggregate(array, mapping, count, axis):
    """Sum segments into shells along one axis."""
    moved = np.moveaxis(array, axis, 0)
    out = np.zeros((count,)+moved.shape[1:])
    np.add.at(out, mapping, moved)
    return np.moveaxis(out, 0, axis)


def build(support, grid):
    profiles = dict(P2=E.pressure_fit(2, support), P4=E.pressure_fit(4, support),
                    E2=E.electron_fit(2, support), E4=E.electron_fit(4, support))
    return profiles, [E.Variant(name, profiles[name], grid, 4., jacobian=True) for name in PROFILES]


def run_jacobians(engine, variants, ell, pairs):
    first = engine.run(variants, ell=ell, workers=WORKERS, pairs=pairs)
    electrons = [v for v in variants if v.field == 1 and v.jacobian]
    references = {v.label: first["dm_pdf_summaries"][v.label] for v in electrons}
    second = engine.run(electrons, ell=ell[:1], workers=WORKERS, pdf_reference=references, verbose=False)
    for label in references:
        first["jacobians"][label]["dm_quantile"] = second["jacobians"][label]["dm_quantile"]
        first["jacobians"][label]["dm_quantile_numerator"] = second["jacobians"][label]["dm_quantile_numerator"]
    first["timing_quantile_pass"] = second["timing"]
    return first


def sum_rules(result):
    rows = {}
    labels = list(result["labels"])
    for name, block in result["jacobians"].items():
        v = labels.index(name)
        checks = dict(cl_1h=(block["one"].sum(axis=(0, 1)), 2*result["cl_1h"][v]),
                      cl_2h=(block["two"].sum(axis=(0, 1)), 2*result["cl_2h"][v]),
                      mean=(block["mean"].sum(), result["mean"][v]))
        if "ksz_one" in block:
            checks.update(ksz_1h=(block["ksz_one"].sum(axis=(0, 1)), 2*result["ksz_cl_1h"][v]),
                          ksz_1h_linear=(block["ksz_one_lin"].sum(axis=(0, 1)), 2*result["ksz_cl_1h_linear_sigma_v"][v]),
                          ksz_2h=(block["ksz_two"].sum(axis=(0, 1)), 2*result["ksz_cl_2h_approx"][v]),
                          dm_cl_1h=(block["dm_one"].sum(axis=(0, 1)), 2*result["dm_cl_1h"][v]),
                          dm_cl_2h=(block["dm_two"].sum(axis=(0, 1)), 2*result["dm_cl_2h"][v]),
                          dm_mean=(block["dm_mean"].sum(), result["dm_mean"][v]),
                          dm_variance=(block["dm_var"].sum(), 2*result["dm_var"][v]),
                          dm_quantiles=(block["dm_quantile"].sum(axis=(1, 2)), result["dm_quantiles"][v]))
        rows[name] = {k: float(np.max(np.abs(np.asarray(a)/np.asarray(b)-1))) for k, (a, b) in checks.items()}
    pairs = list(result.get("pair_labels", []))
    for p, label in enumerate(pairs):
        block = result["pair_jacobians"][label]
        rows[label] = {k: float(np.max(np.abs(block[k].sum(axis=(0, 1))/(result["cross_cl_1h"][p] if "1h" in k else result["cross_cl_2h"][p])-1)))
                       for k in block}
    worst = max(max(r.values()) for r in rows.values())
    # Quantiles are degree-1 homogeneous in F: the sum rule holds to the FFT-grid accuracy.
    quantile_worst = max(r.get("dm_quantiles", 0.) for r in rows.values())
    other = max(max(v for k, v in r.items() if k != "dm_quantiles") for r in rows.values())
    return dict(passed=bool(other < 1e-12 and quantile_worst < 1e-4), max_relative_exact=other,
                max_relative_quantiles=quantile_worst, rows=rows,
                note="exact identities: sum J = 2 O (quadratic), O (linear/cross sides); quantiles: sum dQ/dlnF = Q "
                     "(degree-1 homogeneity) to the accuracy of the FFT grid")


def finite_difference_gate(engine, support, ell):
    """Central differences of single (segment, mass-bin) perturbations vs J."""
    base_grid = jacobian_grid()
    tests = []
    extra = []
    for x, m in FD_TESTS:
        s = int(np.flatnonzero([(base_grid.x[a] < x <= base_grid.x[b]) for a, b in base_grid.segments])[0])
        lo, hi = base_grid.x[base_grid.segments[s][0]], base_grid.x[base_grid.segments[s][1]]
        tests.append((s, m, lo, hi))
        extra += [lo*(1+FD_DELTA), hi*(1-FD_DELTA)]
    grid = jacobian_grid(extra)
    assert len(grid.segments) == len(base_grid.segments)
    profiles = dict(P4=E.pressure_fit(4, support), E4=E.electron_fit(4, support))
    bins = engine.sky.mass_bin
    variants = [E.Variant("P4", profiles["P4"], grid, 4., jacobian=True),
                E.Variant("E4", profiles["E4"], grid, 4., jacobian=True)]
    pairs = [("P4", "E4")]
    for t, (s, m, lo, hi) in enumerate(tests):
        inside = (grid.x >= lo*(1+FD_DELTA)*(1-1e-12)) & (grid.x <= hi*(1-FD_DELTA)*(1+1e-12))
        for sign, tag in ((1, "plus"), (-1, "minus")):
            window = np.ones((len(engine.sky.mass), len(grid)))
            window[np.ix_(bins == m, inside)] = 1+sign*FD_EPS
            for name in ("P4", "E4"):
                variants.append(E.Variant(f"{name}_t{t}_{tag}", profiles[name], grid, 4., window=window))
            pairs += [(f"P4_t{t}_{tag}", "E4"), ("P4", f"E4_t{t}_{tag}")]
    result = run_jacobians(engine, variants, ell, pairs)
    labels = list(result["labels"])
    pair_labels = list(result["pair_labels"])
    rows = []
    for t, (s, m, lo, hi) in enumerate(tests):
        entry = dict(segment=[float(lo), float(hi)], mass_bin=int(m))
        for name in ("P4", "E4"):
            plus, minus = labels.index(f"{name}_t{t}_plus"), labels.index(f"{name}_t{t}_minus")
            J = result["jacobians"][name]
            def fd(key):
                return (result[key][plus]-result[key][minus])/(2*FD_EPS)
            comparisons = dict(cl_1h=(fd("cl_1h"), J["one"][s, m]), cl_2h=(fd("cl_2h"), J["two"][s, m]),
                               mean=(fd("mean"), J["mean"][s, m]))
            if name == "E4":
                comparisons.update(ksz_1h=(fd("ksz_cl_1h"), J["ksz_one"][s, m]), ksz_2h=(fd("ksz_cl_2h_approx"), J["ksz_two"][s, m]),
                                   dm_cl_1h=(fd("dm_cl_1h"), J["dm_one"][s, m]), dm_mean=(fd("dm_mean"), J["dm_mean"][s, m]),
                                   dm_variance=(fd("dm_var"), J["dm_var"][s, m]),
                                   dm_quantiles=(fd("dm_quantiles"), J["dm_quantile"][:, s, m]))
            cross = result["pair_jacobians"]["P4xE4"]
            side = "P" if name == "P4" else "E"
            if name == "P4":
                pp, pm = pair_labels.index(f"P4_t{t}_plusxE4"), pair_labels.index(f"P4_t{t}_minusxE4")
            else:
                pp, pm = pair_labels.index(f"P4xE4_t{t}_plus"), pair_labels.index(f"P4xE4_t{t}_minus")
            comparisons["cross_1h"] = ((result["cross_cl_1h"][pp]-result["cross_cl_1h"][pm])/(2*FD_EPS), cross[f"cross_1h_{side}"][s, m])
            comparisons["cross_2h"] = ((result["cross_cl_2h"][pp]-result["cross_cl_2h"][pm])/(2*FD_EPS), cross[f"cross_2h_{side}"][s, m])
            stats = {}
            for key, (a, b) in comparisons.items():
                a, b = np.atleast_1d(a), np.atleast_1d(b)
                scale = np.max(np.abs(b))
                stats[key] = dict(max_abs_difference_over_max_J=float(np.max(np.abs(a-b))/scale) if scale > 0 else 0.,
                                  max_J=float(scale))
            entry[name] = stats
        rows.append(entry)
    worst_spectra = max(v["max_abs_difference_over_max_J"] for r in rows for n in ("P4", "E4") for k, v in r[n].items() if k != "dm_quantiles")
    worst_quantiles = max(r["E4"]["dm_quantiles"]["max_abs_difference_over_max_J"] for r in rows)
    return dict(passed=bool(worst_spectra < 1e-4 and worst_quantiles < 2e-2), eps=FD_EPS, boundary_offset=FD_DELTA,
                worst_spectra_means_variance=worst_spectra, worst_quantiles=worst_quantiles, tests=rows,
                note="window 1+-eps on nodes inside [lo(1+d), hi(1-d)] of one segment and one mass bin; quadratic observables are "
                     "exact under central differences up to O(d); quantile FD limited by the FFT grid (dx=0.125 pc cm^-3)")


def rerun_finite_differences():
    """Recompute only the finite-difference gate and update the verification file."""
    started = time.perf_counter()
    engine = E.Engine()
    gate = finite_difference_gate(engine, E.support_definition(), E.ELL_EXTENDED)
    path = HERE/"jacobians_verification.json"
    verification = json.loads(path.read_text())
    verification["gates"]["finite_differences"] = gate
    verification["finite_difference_rerun"] = dict(seconds=time.perf_counter()-started,
        reason="first run: make_grid skipped single-node zones, so the boundary nodes lo(1+d), hi(1-d) were absent "
               "and the perturbation missed the two boundary intervals (uniform 13%/6% shortfall); fixed in make_grid")
    verification["passed"] = bool(all(g["passed"] for g in verification["gates"].values()) and verification["frozen_inputs_unchanged"])
    E.write_json(path, verification)
    print(json.dumps(E.C.plain(dict(passed=verification["passed"], worst_spectra=gate["worst_spectra_means_variance"],
                                    worst_quantiles=gate["worst_quantiles"])), indent=1), flush=True)


def main():
    if "--finite-differences-only" in sys.argv:
        return rerun_finite_differences()
    started = time.perf_counter()
    hashes = E.source_hashes()
    support = E.support_definition()
    engine = E.Engine()
    grid = jacobian_grid()
    profiles, variants = build(support, grid)
    ell = E.ELL_EXTENDED
    result = run_jacobians(engine, variants, ell, PAIRS)
    E.save_result(HERE/"jacobians_engine_output.npz", result)
    gates = dict(sum_rules=sum_rules(result))
    print(json.dumps(E.C.plain(gates["sum_rules"]), indent=1), flush=True)
    gates["finite_differences"] = finite_difference_gate(engine, support, ell)
    print(json.dumps(E.C.plain(gates["finite_differences"]), indent=1), flush=True)
    x200 = segment_map(grid, X200_EDGES)
    coarse = segment_map(grid, COARSE_EDGES)
    n200, ncoarse = len(X200_EDGES)-1, len(COARSE_EDGES)-1
    arrays = dict(ell=ell, ell_original83_index=np.array([int(np.flatnonzero(ell == e)[0]) for e in E.ELL]),
                  x200_edges=X200_EDGES, coarse_edges=COARSE_EDGES,
                  segment_edges=np.array([[grid.x[a], grid.x[b]] for a, b in grid.segments]),
                  segment_to_x200=x200, segment_to_coarse=coarse,
                  mass_bin_edges_log10_msun=engine.sky.mass_edges,
                  mass_bin_labels=np.array([f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(engine.sky.mass_edges[:-1], engine.sky.mass_edges[1:])]+[">=15.5"]),
                  mass_bin_calibrated=np.r_[support["support"], False],
                  catalogue_halos_per_bin=np.bincount(engine.sky.mass_bin, weights=engine.sky.counts.sum(axis=0), minlength=engine.sky.n_mass_bins),
                  catalogue_halos_per_bin_z_le_1=np.bincount(engine.sky.mass_bin, weights=engine.sky.counts[engine.sky.z <= 1].sum(axis=0),
                                                             minlength=engine.sky.n_mass_bins),
                  dm_quantile_levels=E.QUANTILES, profiles=np.array(PROFILES), pairs=np.array([f"{a}x{b}" for a, b in PAIRS]))
    labels = list(result["labels"])
    names = dict(one="cl_1h", two="cl_2h", mean="mean", ksz_one="ksz_cl_1h", ksz_one_lin="ksz_cl_1h_linear_sigma_v",
                 ksz_two="ksz_cl_2h_approx", dm_one="dm_cl_1h", dm_two="dm_cl_2h", dm_mean="dm_mean", dm_var="dm_var")
    for name in PROFILES:
        v = labels.index(name)
        block = result["jacobians"][name]
        for key, observable in names.items():
            if key not in block:
                continue
            arrays[f"J_{observable}__{name}"] = aggregate(block[key], x200, n200, 0)
            arrays[f"O_{observable}__{name}"] = result[observable][v]
        if "dm_quantile" in block:
            arrays[f"J_dm_quantiles__{name}"] = aggregate(block["dm_quantile"], x200, n200, 1)
            arrays[f"O_dm_quantiles__{name}"] = result["dm_quantiles"][v]
        arrays[f"O_cl_total__{name}"] = result["cl_total"][v]
        arrays[f"J_cl_total__{name}"] = arrays[f"J_cl_1h__{name}"]+arrays[f"J_cl_2h__{name}"]
        # z=0 shell integrals per 0.2-dex bin centre (Y for P, N_e for E; dimensionless).
        centres = 10**(.5*(support["mass_edges"][1:]+support["mass_edges"][:-1]))
        f0 = profiles[name].base(centres, grid.x)
        pieces = np.stack([E.volume_integral(grid, f0, grid.x[b], grid.x[a]) for a, b in grid.segments], axis=-1)
        arrays[f"I_shell_z0__{name}"] = aggregate(pieces, x200, n200, 1)
    for p, label in enumerate(result["pair_labels"]):
        for key, block in result["pair_jacobians"][label].items():
            arrays[f"J_cross_{key[6:]}__{label}"] = aggregate(block, x200, n200, 0)
        arrays[f"O_cross_cl_1h__{label}"] = result["cross_cl_1h"][p]
        arrays[f"O_cross_cl_2h__{label}"] = result["cross_cl_2h"][p]
    arrays["schema"] = np.array(json.dumps({
        "J_<observable>__<profile>": "dO/dln F[x200 shell, mass bin(, ell)]: F -> F(1+eps) inside one x200 shell for haloes in one 0.2-dex "
                                     "mass bin (index 15 = >=15.5); spectra have a trailing ell axis (ell array, 106 values; "
                                     "ell_original83_index selects the 83 original multipoles)",
        "J_dm_quantiles__<E>": "[quantile, x200 shell, mass bin]; quantile levels dm_quantile_levels",
        "O_<observable>__<profile>": "observable value (same units as the engine output) for O^2 normalisation",
        "J_cross_1h_P__<pair>": "y x DM 1h wrt the pressure shells; _E wrt the electron shells (each side sums to O)",
        "I_shell_z0__<profile>": "[bin centre, x200 shell] z=0 dimensionless 4pi int x^2 F dx; Y(<x_k)=sum_{s<k} I_s and "
                                 "dY(<x_k)/dlnF_s = I_s 1[s<k] (pressure: Y; electrons: N_e) per halo",
        "observables": {"cl_1h/cl_2h/cl_total": "Compton-y C_ell (P) or tau C_ell (E), all z",
                        "ksz_cl_1h": "kSZ 1h with catalogue sum(v_LOS^2) per cell [(Delta T/T)^2]",
                        "ksz_cl_1h_linear_sigma_v": "kSZ 1h with N sigma_v,lin(z)^2",
                        "ksz_cl_2h_approx": "approximate linear kSZ 2h: sigma_v,lin^2/c^2 P_lin [sum N b tau/(4pi)]^2/V",
                        "dm_cl_1h/dm_cl_2h": "DM-field C_ell, z<=1, observer 1/(1+z), [pc cm^-3]^2",
                        "dm_mean/dm_var": "mean halo DM (z_s=1) and compound-Poisson variance",
                        "dm_quantiles": "compound-Poisson per-sightline PDF quantiles (FFT, exact up to the 0.125 pc cm^-3 grid)",
                        "mean": "mean y (P) or mean tau (E), all z"},
        "metric": "K = sum_O lambda_O J_O^T J_O / O^2 (flatten [shell, bin] as the parameter axis; ell or quantile as data axis)",
        "truncation": "hard cut at 4 R200c (Test C convention), common support 13.5<=logM<=15.388, published fallback elsewhere"}))
    np.savez_compressed(HERE/"jacobians.npz", **arrays)
    unchanged = E.source_hashes() == hashes
    verification = dict(passed=bool(all(g["passed"] for g in gates.values()) and unchanged), gates=gates,
                        frozen_inputs_unchanged=unchanged, timing=result["timing"],
                        timing_quantile_pass=result["timing_quantile_pass"], elapsed_seconds=time.perf_counter()-started,
                        grid_nodes=len(grid), segments=len(grid.segments), x200_shells=n200)
    E.write_json(HERE/"jacobians_verification.json", verification)
    print(json.dumps(dict(passed=verification["passed"], elapsed=verification["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
