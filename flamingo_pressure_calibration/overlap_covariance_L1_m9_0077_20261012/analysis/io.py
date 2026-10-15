"""Grid-agnostic loading, normalisation and rebinning of linear shell sums.

Data contract (a plain ``dict``; every later extraction must provide it):

``sums``          float64 [halo, *middle, shell] linear physical sums (shell axis last)
``axes``          names of the ``sums`` axes, first ``'halo'`` and last ``'shell'``; one
                  middle axis must be ``'quantity'`` (e.g. radii234: halo, component,
                  quantity, shell; the coming pass: halo, owner, phase, quantity, shell)
``coords``        {axis name: tuple of labels} for every middle axis
``edges``         [shell+1] edges in units of the normalising radius (x200 or x500)
``model_edges``   edges used for model shell averages (defaults to ``edges``)
``radius_mpc``    [halo] normalising radius (R200c or R500c), physical Mpc
``mass_msun``     [halo] normalising mass (M200c or M500c), physical Msun
``overdensity``   200 or 500 (critical); rho_c = M / (4 pi/3 R^3) / overdensity
``baryon_fraction``, ``softening_mpc``
``hsml_median_x`` [halo, shell] upper-median smoothing length / radius (optional), or
``hs_hist`` [halo, shell, nbin] + ``hs_edges`` (exact medians after rebinning)
``halos``         dict of per-halo catalogue arrays (``mass_bin``, ``inclusion_weight``...)

Normalisations reproduce radii234 ``extract_study.merge_products`` bit for bit
(same floating-point operation order), with the all-gas halo normalisation and
the full geometric shell volume for every component:

* pressure  : sum(y) m_e c^2 / sigma_T / (V Mpc) / P_Delta,  P_Delta = G M rho_Delta f_b / (2R)
* electrons : sum(n_e V) mu_XGP / (V rho_c f_b)
* gas       : sum(m) / (V rho_c f_b)

Selected components are normalised first and then summed in index order, as in
radii234 ``fit_study`` (``data[key][rows][:, components].sum(axis=1)``).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from . import _legacy

P = _legacy.pilot_physics()

SUM_NAMES = tuple(P.SUM_NAMES)
COMPONENTS = ("central_bound", "own_satellites_bound", "other_halos_bound", "unbound")
SELECTIONS = {"all": (0, 1, 2, 3), "no_other": (0, 1, 3),
              "no_satellites_or_other": (0, 3), "host_bound": (0, 1)}
FIELDS = {
    "pressure_noagn": {"sum": "y_noagn", "kind": "pressure", "family": "b12",
                       "label": "P_e/P200, recent-AGN gas removed", "stored": None},
    "pressure": {"sum": "y", "kind": "pressure", "family": "b12",
                 "label": "P_e/P200, all gas", "stored": "pressure_components"},
    "electrons": {"sum": "electrons", "kind": "electrons", "family": "b16",
                  "label": "mu n_e/(f_b rho_c)", "stored": "electron_components"},
    "gas": {"sum": "mass", "kind": "gas", "family": "b16",
            "label": "rho_gas/(f_b rho_c)", "stored": "gas_components"},
}
H_FLAMINGO = 0.681


# ----------------------------------------------------------------------------- loading
def load_radii234(path: Path | str = _legacy.RADII234, keep_stored: bool = True) -> dict:
    """Load radii234 ``measurements.npz`` + ``selected_halos.npz`` into the data contract."""
    path = Path(path)
    plan = json.loads((path / "metadata_plan.json").read_text())
    with np.load(path / "measurements.npz") as meas:
        stored_keys = ("pressure_components", "electron_components", "gas_components",
                       "pressure_ratio", "pressure_noagn_ratio", "p200_pa",
                       "critical_density_msun_mpc3", "gas_density_ratio", "tau_density_ratio")
        data = {
            "sums": meas["sums"],
            "edges": meas["edges"],
            "hsml_median_x": meas["hsml_median_upper_x"],
            "stored": {k: meas[k] for k in stored_keys} if keep_stored else {},
        }
    halos = dict(np.load(path / "selected_halos.npz"))
    population = np.asarray(plan["population_per_bin"], float)
    sample = np.bincount(halos["mass_bin"], minlength=len(population)).astype(float)
    halos["inclusion_weight"] = (population / sample)[halos["mass_bin"]]
    halos["logmass"] = np.log10(halos["m200c_msun"])
    data.update({
        "axes": ("halo", "component", "quantity", "shell"),
        "coords": {"component": COMPONENTS, "quantity": SUM_NAMES},
        # measurements.npz edges come from the maas numpy; the local recomputation of
        # pilot_physics.EDGES differs by one ULP at edge 15 (0.2335).  Volumes use the
        # stored edges (bit-exact normalisation); fits use pilot_physics.EDGES, as
        # radii234 fit_study did locally (bit-exact fits).
        "model_edges": np.asarray(P.EDGES, float),
        "radius_mpc": halos["r_200_crit_mpc"],
        "mass_msun": halos["m200c_msun"],
        "overdensity": 200.0,
        "baryon_fraction": float(plan["baryon_fraction"]),
        "softening_mpc": float(plan["softening_physical_mpc"]),
        "halos": halos,
        "mass_edges": np.asarray(halos["mass_edges"], float),
        "population_per_bin": population,
        "sample_per_bin": sample,
        "h": H_FLAMINGO,
        "grid": "x200 radii234 36 shells",
        "source": str(path),
    })
    return data


# ----------------------------------------------------------------------------- normalisation
def shell_volume(data: dict, rows=None) -> np.ndarray:
    """[halo, shell] geometric shell volume in physical Mpc^3 (radii234 op order)."""
    radius = data["radius_mpc"] if rows is None else data["radius_mpc"][rows]
    return 4 * np.pi / 3 * radius[:, None]**3 * np.diff(np.asarray(data["edges"])**3)


def critical_density(data: dict, rows=None) -> np.ndarray:
    mass = data["mass_msun"] if rows is None else data["mass_msun"][rows]
    radius = data["radius_mpc"] if rows is None else data["radius_mpc"][rows]
    rho_delta = mass / (4 * np.pi / 3 * radius**3)
    return rho_delta / data["overdensity"]


def pressure_scale(data: dict, rows=None) -> np.ndarray:
    """Self-similar P_Delta in Pa (P200 for the x200 grid)."""
    mass = data["mass_msun"] if rows is None else data["mass_msun"][rows]
    radius = data["radius_mpc"] if rows is None else data["radius_mpc"][rows]
    return P.p200(mass, radius, data["baryon_fraction"])


def _normalise(linear: np.ndarray, kind: str, data: dict, rows) -> np.ndarray:
    """Normalise [halo, *middle, shell] linear sums with radii234's op order."""
    extra = (slice(None),) + (None,) * (linear.ndim - 2) + (slice(None),)
    volume = shell_volume(data, rows)[extra]
    fb = data["baryon_fraction"]
    per_halo = (slice(None),) + (None,) * (linear.ndim - 1)
    if kind == "pressure":
        out = linear * P.electron_mass * P.c**2 / P.SIGMA_T / (volume * P.MPC_M)
        out /= pressure_scale(data, rows)[per_halo]
        return out
    rho_c = critical_density(data, rows)[per_halo]
    if kind == "electrons":
        return linear * P.MU_XGP_MSUN / (volume * rho_c * fb)
    if kind == "gas":
        return linear / (volume * rho_c * fb)
    raise ValueError(kind)


def _take_quantity(data: dict, quantity: str, rows=None) -> tuple[np.ndarray, list]:
    axes = list(data["axes"])
    q_axis = axes.index("quantity")
    index = list(data["coords"]["quantity"]).index(quantity)
    sums = data["sums"] if rows is None else data["sums"][rows]
    return np.take(sums, index, axis=q_axis), [a for a in axes if a != "quantity"]


def _resolve(axis: str, choice, data: dict):
    labels = list(data["coords"][axis])
    if choice is None:
        return list(range(len(labels)))
    if isinstance(choice, str):
        if axis == "component" and choice in SELECTIONS:
            return list(SELECTIONS[choice])
        return [labels.index(choice)]
    return [labels.index(c) if isinstance(c, str) else int(c) for c in choice]


def component_profiles(data: dict, field: str, rows=None) -> np.ndarray:
    """Normalised profiles for every middle-axis element: [halo, *middle, shell]."""
    spec = FIELDS[field]
    linear, _ = _take_quantity(data, spec["sum"], rows)
    return _normalise(linear, spec["kind"], data, rows)


def profile(data: dict, field: str, selection="all", rows=None, **select) -> np.ndarray:
    """Per-halo normalised profile [halo, shell] summed over the selected elements.

    ``selection`` applies to the ``component`` axis (radii234 names or index list);
    other middle axes are chosen with keyword arguments (``phase=['xray', 'other']``)
    and are summed in full when omitted.
    """
    values = component_profiles(data, field, rows)
    axes = [a for a in data["axes"] if a != "quantity"]
    choices = dict(select)
    if "component" in axes:
        choices.setdefault("component", selection)
    # Sum from the last middle axis backwards so axis numbers stay valid.
    for position in range(len(axes) - 2, 0, -1):
        name = axes[position]
        index = _resolve(name, choices.get(name), data)
        values = np.take(values, index, axis=position).sum(axis=position)
    return values


def linear_sum(data: dict, quantity: str, selection="all", rows=None, **select) -> np.ndarray:
    """Un-normalised linear sums [halo, shell] for one quantity and a selection."""
    values, axes = _take_quantity(data, quantity, rows)
    choices = dict(select)
    if "component" in axes:
        choices.setdefault("component", selection)
    for position in range(len(axes) - 2, 0, -1):
        name = axes[position]
        index = _resolve(name, choices.get(name), data)
        values = np.take(values, index, axis=position).sum(axis=position)
    return values


def integrated(values: np.ndarray, edges, cuts) -> np.ndarray:
    """Dimensionless volume integrals sum_s F_s (x_hi^3 - x_lo^3) inside each cut.

    In units of (4 pi/3) R^3 times the field normalisation: Y(<x)/(4pi/3 R^3 P200)
    for pressure, N_e or M_gas/(4pi/3 R^3 f_b rho_c) for densities.  Cuts must be
    existing edges.  Returns [halo, cut].
    """
    edges = np.asarray(edges, float)
    weight = np.diff(edges**3)
    out = []
    for cut in np.atleast_1d(cuts):
        inside = edges[1:] <= cut + 1e-12
        out.append((values[:, inside] * weight[inside]).sum(axis=1))
    return np.stack(out, axis=1)


# ----------------------------------------------------------------------------- rebinning
def edge_indices(edges, new_edges, rtol: float = 1e-9) -> np.ndarray:
    """Indices of ``new_edges`` inside ``edges``; raises unless an exact sub-grid."""
    edges = np.asarray(edges, float)
    new_edges = np.asarray(new_edges, float)
    index = np.array([int(np.argmin(np.abs(edges - e))) for e in new_edges])
    scale = np.maximum(np.abs(new_edges), 1e-300)
    if not np.all(np.abs(edges[index] - new_edges) <= rtol * scale + 1e-15):
        bad = new_edges[np.abs(edges[index] - new_edges) > rtol * scale + 1e-15]
        raise ValueError(f"new edges are not a sub-grid of the data edges: {bad}")
    if np.any(np.diff(index) <= 0):
        raise ValueError("new edges must be strictly increasing")
    return index


def rebin_sums(sums: np.ndarray, edges, new_edges) -> np.ndarray:
    """Exact linear rebinning along the last axis with ``np.add.reduceat``."""
    index = edge_indices(edges, new_edges)
    part = sums[..., index[0]:index[-1]]
    return np.add.reduceat(part, index[:-1] - index[0], axis=-1)


def upper_median_from_hist(hist: np.ndarray, hs_edges) -> np.ndarray:
    """radii234 measure.smoothing_upper_median for [halo, shell, nbin] histograms."""
    cumulative = np.cumsum(hist, axis=-1)
    count = cumulative[..., -1]
    target = count // 2 + 1
    index = np.argmax(cumulative >= target[..., None], axis=-1)
    result = np.asarray(hs_edges)[index + 1].astype(float)
    result[count == 0] = np.nan
    return result


def rebin(data: dict, new_edges) -> dict:
    """Return a shallow copy of ``data`` on a coarser sub-grid of its edges.

    Linear sums are rebinned exactly.  Smoothing lengths: exact upper medians when
    ``hs_hist`` is present; otherwise the conservative maximum of the constituent
    fine-shell medians (a coarse shell is resolved only if its inner edge is at
    least twice the worst fine-shell median).
    """
    index = edge_indices(data["edges"], new_edges)
    out = dict(data)
    out["sums"] = rebin_sums(data["sums"], data["edges"], new_edges)
    out["edges"] = np.asarray(data["edges"], float)[index]
    if "model_edges" in data:
        out["model_edges"] = np.asarray(data["model_edges"], float)[index]
    if "hs_hist" in data:
        hist = np.moveaxis(data["hs_hist"], -2, -1)          # [halo, nbin, shell]
        hist = rebin_sums(hist, data["edges"], new_edges)
        out["hs_hist"] = np.moveaxis(hist, -1, -2)
        out["hsml_median_x"] = upper_median_from_hist(out["hs_hist"], data["hs_edges"])
    elif "hsml_median_x" in data:
        h = data["hsml_median_x"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN (empty) halo-shells stay NaN
            segments = [np.nanmax(h[:, a:b], axis=1) for a, b in zip(index[:-1], index[1:])]
        out["hsml_median_x"] = np.stack(segments, axis=1)
    out["stored"] = {}
    out["grid"] = f"rebinned from {data.get('grid', '?')} to {len(index) - 1} shells"
    return out


def nearest_edge(edges, target: float) -> float:
    edges = np.asarray(edges, float)
    return float(edges[int(np.argmin(np.abs(edges - target)))])


def shell_centres(edges) -> np.ndarray:
    """Logarithmic shell centres sqrt(lo hi); the inner shell [0, e1] uses e1/2."""
    edges = np.asarray(edges, float)
    lo, hi = edges[:-1], edges[1:]
    return np.where(lo > 0, np.sqrt(np.maximum(lo, 1e-300) * hi), hi / 2)


# ----------------------------------------------------------------------------- bins
def bin_rows(data: dict, b: int) -> np.ndarray:
    return np.flatnonzero(data["halos"]["mass_bin"] == b)


def bin_label(lo: float, hi: float) -> str:
    return f"{lo:.1f}-{hi:.1f}"


def mass_range_rows(data: dict, log_lo: float, log_hi: float) -> np.ndarray:
    logm = data["halos"]["logmass"]
    return np.flatnonzero((logm >= log_lo) & (logm < log_hi))


def lee22_bins(data: dict) -> list[dict]:
    """Lee22 Fig. 1 bins (h^-1 Msun) converted to physical Msun with FLAMINGO h."""
    shift = -np.log10(data["h"])
    out = []
    for lo, hi in [(13.2, 13.4), (13.8, 14.0)]:
        plo, phi = lo + shift, hi + shift
        rows = mass_range_rows(data, plo, phi)
        strata = np.unique(data["halos"]["mass_bin"][rows])
        out.append({"label_h": f"10^{lo:.1f}-10^{hi:.1f} h^-1 Msun",
                    "log_lo": plo, "log_hi": phi, "rows": rows, "strata": strata.tolist(),
                    "weights": data["halos"]["inclusion_weight"][rows]})
    return out
