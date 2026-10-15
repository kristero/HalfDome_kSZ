"""Build the per-halo profile cache of the stacking-choices study (stacking_choices_20261014/cache/).

Run:  LD_LIBRARY_PATH=<env>/lib <env>/bin/python -B core/build_cache.py

Reads the particle-pass products of the study root (STUDY = the parent of stacking_choices_20261014) through the
study library ``analysis`` (newdata.load(), variants.profile(), variants.diagnostic(), variants.owner_sum(), the
variants normalisation helpers), read-only, and writes ONLY into stacking_choices_20261014/cache/:

  <variant file>.npy      per-halo profile [4588, S] float64, uncompressed (np.load(..., mmap_mode='r')); rows follow
                          STUDY/selected_halos.npz (= B_covariance halo_table orig_row = F profiles halo_rows)
  edges_<grid>.npy        shell edges in units of the grid radius (R200c for x200/fit/lee22/old, R500c for x500/b24)
  counts_<grid>.npy       [4588, S] gas-particle counts per shell (all owners, all phases incl. recent AGN)
  hsml_<grid>.npy         [4588, S] upper-median smoothing length h a / R per shell (NaN where the shell is empty),
                          from the rebinned hs200 / hs500 histograms (analysis.io.upper_median_from_hist)
  halo_table.npz          per-halo table (uncompressed): masses, radii, weights, bins, classes, blocks, folds, scales
  index.json              variant key -> metadata (file, field, estimator, agn, selection, phase, grid, norm, units...)
  build_summary.json      timings, peak RSS, counts

Variant key: 'field|estimator|agn|selection|phase|grid|norm' (e.g. 'P|geometric|removed|all|all|x200|P200').
See index.json 'conventions' for every token.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
STUDY = OUT.parent
CACHE = OUT / "cache"
sys.path.insert(0, str(STUDY))

from analysis import io as aio  # noqa: E402
from analysis import newdata  # noqa: E402
from analysis import variants as V  # noqa: E402

# ----------------------------------------------------------------------------------------------- configuration
# Test-B fit grid (B_covariance/bcommon.py FIT_INNER_MAX / FIT_EXTRA_EDGES, b0_prepare.fit_grid_edges): identical
# to Test F's F grid (35 shells to 4 R200c); checked bitwise against B_covariance/results/profiles_fit.npz below.
FIT_INNER_MAX = 0.8370
FIT_EXTRA_EDGES = (1.0, 1.15156, 1.34, 1.5, 1.75, 2.0, 2.18026, 2.5, 3.0, 3.30193, 3.63424, 4.0)
B24 = newdata.B24                    # f_B = 0.159, mu = 0.59, mu_e = 1.14
MU_T = 0.59                          # mean molecular weight of kT200 (as B_covariance halo_table T200_K)
MU_E_K = 1.14                        # mu_e of the self-similar n_e,200 in K200 (B24 / Pratt+10 convention)

X200_GRIDS = ("x200", "fit", "lee22", "old")
X500_GRIDS = ("x500", "b24")
FIELD_LIB = {"P": "pressure", "E": "electrons", "G": "gas"}

# selection token -> (library spec, kind, label)
SELECTIONS = {
    "all": ("all", "owner", "all gas in the sphere (literature 'full signal'; own satellites and neighbours kept)"),
    "central_bound": ("central_bound", "owner", "gas bound to the central subhalo only (SOAP/HBT membership)"),
    "no_other": ("no_other", "owner", "radii234: central + own satellites + unbound (gas bound to other haloes removed)"),
    "no_satellites_or_other": ("no_satellites_or_other", "owner",
                               "radii234: central-bound + unbound (all bound substructure removed)"),
    "host_bound": ("host_bound", "owner", "radii234: central + own satellites (FoF-like host-bound gas; unbound removed)"),
    "rm_ov_mu0.1": ({"name": "remove_overlappers", "mu_min": 0.1, "reach": "any", "flavour": "remove_all"}, "owner",
                    "remove gas bound to overlappers with mu >= 0.1 reaching inside 2R (own satellites and neighbours)"),
    "rm_ov_mu0.1_keepsat": ({"name": "remove_overlappers", "mu_min": 0.1, "reach": "any",
                             "flavour": "keep_own_satellites"}, "owner",
                            "remove gas bound to neighbour overlappers (mu >= 0.1, inside 2R); own satellites kept"),
    "rm_ov_mu0.03": ({"name": "remove_overlappers", "mu_min": 0.03, "reach": "any", "flavour": "remove_all"}, "owner",
                     "remove gas bound to overlappers with mu >= 0.03 reaching inside 2R"),
    "rm_ov_mu0.3": ({"name": "remove_overlappers", "mu_min": 0.3, "reach": "any", "flavour": "remove_all"}, "owner",
                    "remove gas bound to overlappers with mu >= 0.3 reaching inside 2R"),
    "rm_ov_mu0.1_reach1R": ({"name": "remove_overlappers", "mu_min": 0.1, "reach": "reach1R",
                             "flavour": "remove_all"}, "owner",
                            "remove gas bound to overlappers with mu >= 0.1 that reach inside 1R only"),
    "A2i": ("remove_bound_to_other_painted", "owner",
            "A2-i: remove gas bound to other painted distinct haloes (M200c >= 6.6e12; owner 14 + painted neighbour "
            "overlapper codes, bound-gas majority)"),
    "A2ii": ("partition_owned", "partition",
             "A2-ii: scaled-distance partition, keep gas whose r/R200c to this halo is smaller than to every painted "
             "neighbour (painting-consistent ownership)"),
    "A2ii_A2i": ("partition_owned_not_other_painted", "partition",
                 "A2-ii and A2-i: partition-owned and not bound to another painted halo"),
    "exc_all_k0.5": ({"variant": "all", "kappa": 0.5, "members": "all", "min_retained": 0.5}, "excision",
                     "excision of spheres around every overlapper (mu >= 0.1, 2R reach), zones s < 0.5 removed; "
                     "shells retaining < 50% of their volume are NaN"),
    "exc_all_k1": ({"variant": "all", "kappa": 1.0, "members": "all", "min_retained": 0.5}, "excision",
                   "excision around every overlapper, kappa = 1 (s < 1 R_j removed); retained < 50% -> NaN"),
    "exc_all_k2": ({"variant": "all", "kappa": 2.0, "members": "all", "min_retained": 0.5}, "excision",
                   "excision around every overlapper, kappa = 2; retained < 50% -> NaN"),
    "exc_nbr_k1": ({"variant": "neighbours_only", "kappa": 1.0, "members": "all", "min_retained": 0.5}, "excision",
                   "excision around neighbour overlappers only (own satellites kept), kappa = 1; retained < 50% -> NaN"),
}
PHASES = {
    "all": "all gas phases allowed by the agn token (agn removed: xray_selected + other; kept: all three)",
    "xray": "X-ray-selected phase only: SFR <= 0, T >= 1e5 K, not recently AGN-heated (no star-forming, cold or AGN gas)",
    "cold_sf": "'other' phase only: star-forming (SFR > 0) or T < 1e5 K, not recently AGN-heated",
}
AGN = {"removed": "recently AGN-heated gas removed (SOAP filter a_last >= 0.998956, 8.80e6 <= T <= 1.756e8 K; "
                  "radii234 y_noagn; default)",
       "kept": "all gas including the recently AGN-heated phase",
       "only": "only the recently AGN-heated phase"}
PHASE_W = {("removed", "all"): (0.0, 1.0, 1.0), ("kept", "all"): (1.0, 1.0, 1.0), ("only", "all"): (1.0, 0.0, 0.0),
           ("removed", "xray"): (0.0, 1.0, 0.0), ("removed", "cold_sf"): (0.0, 0.0, 1.0)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def fit_grid_edges(G):
    """B_covariance b0_prepare.fit_grid_edges, verbatim maths."""
    targets = np.r_[G.OLD_EDGES[G.OLD_EDGES <= FIT_INNER_MAX], FIT_EXTRA_EDGES]
    idx = np.array([int(np.argmin(np.abs(G.X200 - e))) for e in targets])
    if not np.allclose(G.X200[idx], targets, rtol=1e-5, atol=0):
        raise ValueError("fit-grid edge not found in X200")
    if np.any(np.diff(idx) <= 0):
        raise ValueError("fit grid not increasing")
    return G.X200[idx], idx


def file_name(key):
    return key.replace("|", "__") + ".npy"


class Builder:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.d = newdata.load(STUDY)
        d = self.d
        self.G, self.P = d.mod.G, d.mod.P
        fe, _ = fit_grid_edges(self.G)
        self.gridspec = {"x200": "x200", "fit": ("x200", fe), "lee22": "lee22", "old": "old", "x500": "x500",
                         "b24": "b24"}
        self.base = {g: ("x500" if g in X500_GRIDS else "x200") for g in self.gridspec}
        self.index = {}
        self.timings = {}
        self.n = d.n

    # ------------------------------------------------------------------ helpers
    def grid_index(self, g):
        return self.d.subgrid_index(self.gridspec[g])[1]

    def grid_edges(self, g):
        return np.asarray(self.d.edges(self.gridspec[g]), float)

    def save(self, key, values, **meta):
        values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        field, est, agn, sel, phase, grid, norm = key.split("|")
        S = len(self.grid_edges(grid)) - 1
        if values.shape != (self.n, S):
            raise ValueError(f"{key}: shape {values.shape} != {(self.n, S)}")
        if key in self.index:
            raise ValueError(f"duplicate key {key}")
        fname = file_name(key)
        np.save(CACHE / fname, values)
        fin = np.isfinite(values)
        entry = {"file": fname, "field": field, "estimator": est, "agn": agn, "selection": sel, "phase": phase,
                 "grid": grid, "base": self.base[grid], "norm": norm, "edges_file": f"edges_{grid}.npy",
                 "shape": list(values.shape), "finite_fraction": float(fin.mean()),
                 "nan_halo_shells": int((~fin).sum()), "negative_halo_shells": int(np.sum(values[fin] < 0)),
                 "zero_halo_shells": int(np.sum(values[fin] == 0))}
        entry.update(meta)
        self.index[key] = entry

    def scales(self, base, norm_lib):
        return V._scales(self.d, base, norm_lib, None)

    def phase_sum(self, base, quantity, sel, phase_w):
        """variants.owner_sum with an arbitrary phase weight vector (phases: recent_agn, xray_selected, other).
        Identical einsum to the library (bitwise equal for the library's own phase vectors; tested)."""
        d = self.d
        block = d.array("owner200" if base == "x200" else "owner500")
        sub = block[:, :, :, d.index("quantities", quantity), :]
        ph = np.asarray(phase_w, float)
        if sel.per_halo is not None:
            masks = V.owner_masks(d, sel, None).astype(float)
            return np.einsum("hops,ho,p->hs", sub, masks, ph)
        return np.einsum("hops,o,p->hs", sub, np.asarray(sel.owners, float), ph)

    def lib_norm(self, field, norm):
        """Cache norm token -> library normalisation name."""
        if norm in ("P200", "rhoc_fb"):
            return "P200"
        return norm

    # ------------------------------------------------------------------ units
    @staticmethod
    def units(field, norm):
        fb = "f_b = Omega_b/Omega_m = 0.158823 (metadata_plan.json)"
        table = {
            ("P", "P200"): f"P_e / P200, P200 = G M200c rho200 f_b / (2 R200c), rho200 = M200c/(4pi/3 R200c^3), {fb}",
            ("P", "B24"): "P_e / P_e,500 (B24: n_e,500 k T500; k T500 = 0.59 m_p G M500c/(2 R500c), "
                          "n_e,500 = 500 f_B rho_c/(1.14 m_p), f_B = 0.159)",
            ("P", "A10"): "P_e / P500 (Arnaud+10: 1.65e-3 E(z)^(8/3) [M500c/(3e14 h70^-1 Msun)]^(2/3) h70^2 keV cm^-3)",
            ("P", "self_similar_500"): f"P_e / (G M500c rho500 f_b / (2 R500c)), {fb}",
            ("E", "rhoc_fb"): f"n_e mu_XGP / (rho_c f_b) (XGPaint electron convention; mu_XGP = (m_e + 2 m_p/(1+0.76))/0.9; "
                              f"rho_c = M200c/(4pi/3 R200c^3)/200; {fb})",
            ("E", "B24"): "n_e / n_e,500 (n_e,500 = 500 f_B rho_c/(mu_e m_p), f_B = 0.159, mu_e = 1.14)",
            ("E", "A10"): "n_e / n_e,500 (B24 n_e,500; library: A10 changes the pressure scale only, so equal to B24)",
            ("E", "self_similar_500"): "n_e / n_e,500 (B24 n_e,500; identical to the B24 array)",
            ("E_fullion", "rhoc_fb"): f"n_e,full mu_XGP / (rho_c f_b), n_e,full = (m/m_p)(X_H + X_He/2) per particle "
                                      f"(fully ionised); {fb}",
            ("E_fullion", "B24"): "n_e,full / n_e,500 (B24)",
            ("G", "rhoc_fb"): f"rho_gas / (rho_c f_b), {fb}",
            ("G", "B24"): "rho_gas / (500 f_B rho_c) (B24 f_B = 0.159)",
            ("T_M", "T200"): "mass-weighted T / T200, k T200 = 0.59 m_p G M200c / (2 R200c)",
            ("T_V", "T200"): "SPH-volume-weighted T / T200",
            ("T_X", "T200"): "ROSAT-luminosity-weighted T (X-ray phase) / T200",
            ("T_e", "T200"): "electron-weighted kT_e = sum(n_e V k T)/sum(n_e V), / k T200",
            ("T_M", "T500"): "mass-weighted T / T500, k T500 = 0.59 m_p G M500c / (2 R500c) (B24)",
            ("T_V", "T500"): "SPH-volume-weighted T / T500",
            ("T_X", "T500"): "ROSAT-luminosity-weighted T (X-ray phase) / T500",
            ("T_e", "T500"): "electron-weighted kT_e / k T500",
            ("K", "K200"): "K / K200, K = kT_e n_e^(-2/3) with kT_e electron-weighted and n_e = sum(n_e V)/V_shell "
                           "(geometric); K200 = k T200 n_e,200^(-2/3), k T200 = 0.59 m_p G M200c/(2 R200c), "
                           "n_e,200 = 200 f_b rho_c/(1.14 m_p), f_b = 0.158823",
            ("K", "K500"): "K / K500, K500 = k T500 n_e,500^(-2/3) (B24 constants: mu = 0.59, f_B = 0.159, mu_e = 1.14)",
            ("C_rho", "none"): "density clumping <rho^2>_V / <rho>_V^2 = sum(m rho) sum(V) / sum(m)^2 (V = m/rho)",
            ("C_P", "none"): "pressure clumping <Pe^2>_V / <Pe>_V^2 = sum(V Pe^2) sum(V) / sum(V Pe)^2",
        }
        return table.get((field, norm), "")

    # ------------------------------------------------------------------ owner-block variants
    def owner_variants(self, base):
        d, P = self.d, self.P
        grids = X200_GRIDS if base == "x200" else X500_GRIDS
        t = time.perf_counter()
        n0 = len(self.index)
        if base == "x200":
            pe_norm, ed_norm = ["P200"], ["rhoc_fb"]
        else:
            pe_norm, ed_norm = ["B24", "A10", "self_similar_500", "P200"], ["B24", "A10", "self_similar_500", "rhoc_fb"]

        def prof(field, est, agn, sel, g, norm, **kw):
            return V.profile(d, FIELD_LIB[field], est, agn, SELECTIONS[sel][0], self.gridspec[g],
                             normalisation=self.lib_norm(field, norm), **kw).values

        def put(key, values, lib_call, desc=""):
            field, est, agn, sel, phase, g, norm = key.split("|")
            m = {"selection_spec": SELECTIONS.get(sel, ("",))[0], "selection_label": SELECTIONS.get(sel, ("", "", ""))[2],
                 "units": self.units(field, norm), "library_call": lib_call, "description": desc}
            self.save(key, values, **m)

        # (1) estimators, selection 'all', phase 'all', agn removed / kept
        for g in grids:
            norms_p = pe_norm if base == "x500" else ["P200"]
            for agn in ("removed", "kept"):
                for est in V.ESTIMATORS["pressure"]:
                    for norm in (norms_p if est == "geometric" else norms_p[:1]):
                        if base == "x500" and agn == "kept" and norm != "B24":
                            continue
                        key = f"P|{est}|{agn}|all|all|{g}|{norm}"
                        put(key, prof("P", est, agn, "all", g, norm),
                            f"variants.profile(d, 'pressure', '{est}', '{agn}', 'all', {g!r}, normalisation="
                            f"'{self.lib_norm('P', norm)}')",
                            "Q_X uses the X-ray phase for both sums, so its agn token has no effect" if est == "Q_X" else
                            ("n_e from the agn-selected phases, T_X from the X-ray phase" if est == "nT_X" else ""))
                for est in V.ESTIMATORS["electrons"]:
                    for norm in (ed_norm if est == "geometric" else ed_norm[:1]):
                        if base == "x500" and agn == "kept" and norm != "B24":
                            continue
                        key = f"E|{est}|{agn}|all|all|{g}|{norm}"
                        put(key, prof("E", est, agn, "all", g, norm),
                            f"variants.profile(d, 'electrons', '{est}', '{agn}', 'all', {g!r}, normalisation="
                            f"'{self.lib_norm('E', norm)}')",
                            "for E the library's A10 / self_similar_500 normalisations use the B24 n_e,500 "
                            "(identical arrays)" if norm in ("A10", "self_similar_500") else "")
                for est in V.ESTIMATORS["gas"]:
                    norm = "rhoc_fb" if base == "x200" else "B24"
                    key = f"G|{est}|{agn}|all|all|{g}|{norm}"
                    put(key, prof("G", est, agn, "all", g, norm),
                        f"variants.profile(d, 'gas', '{est}', '{agn}', 'all', {g!r}, normalisation='{self.lib_norm('G', norm)}')",
                        "Q_M = sum(m rho)/sum(m): clumping-biased" if est == "Q_M" else "")
        log(f"{base}: estimators done ({len(self.index) - n0} arrays, {time.perf_counter() - t:.0f} s)")

        # (2) owner-block selections (geometric), agn removed / kept
        owner_sels = [s for s, v in SELECTIONS.items() if v[1] == "owner" and s != "all"]
        for g in grids:
            for sel in owner_sels:
                if base == "x500" and sel not in ("central_bound", "rm_ov_mu0.1", "rm_ov_mu0.1_keepsat", "A2i"):
                    continue
                for agn in ("removed", "kept"):
                    if base == "x500" and agn == "kept":
                        continue
                    for field in ("P", "E", "G"):
                        norm = {"P": pe_norm[0], "E": ed_norm[0], "G": ("rhoc_fb" if base == "x200" else "B24")}[field]
                        key = f"{field}|geometric|{agn}|{sel}|all|{g}|{norm}"
                        put(key, prof(field, "geometric", agn, sel, g, norm),
                            f"variants.profile(d, '{FIELD_LIB[field]}', 'geometric', '{agn}', {SELECTIONS[sel][0]!r}, {g!r})")
        log(f"{base}: owner selections done ({len(self.index) - n0} arrays, {time.perf_counter() - t:.0f} s)")

        # (3) phase restrictions (custom phase vectors through the library's normalisation helpers)
        sel_objs = {s: V._as_selection(SELECTIONS[s][0]) for s in ("all", "central_bound")}
        for g in grids:
            idx = self.grid_index(g)
            edges = self.grid_edges(g)
            norm_p, norm_e = pe_norm[0], ed_norm[0]
            sc = self.scales(base, self.lib_norm("P", norm_p))
            vol = V._volume(d, base, edges, None)
            rb = lambda a: d.rebin(a, idx)       # noqa: E731
            combos = [("xray", "removed", s) for s in ("all", "central_bound")] + \
                     [("cold_sf", "removed", s) for s in ("all", "central_bound")] + [("all", "only", "all")]
            for phase, agn, sname in combos:
                if base == "x500" and (sname != "all"):
                    continue
                sel = sel_objs[sname]
                w = PHASE_W[(agn, phase)]
                y = rb(self.phase_sum(base, "y", sel, w))
                ne = rb(self.phase_sum(base, "electrons", sel, w))
                m = rb(self.phase_sum(base, "mass", sel, w))
                desc = f"phase weights (recent_agn, xray_selected, other) = {w}; library normalisation helpers"
                put(f"P|geometric|{agn}|{sname}|{phase}|{g}|{norm_p}", V._pressure_geometric(d, y, vol, sc),
                    "stackcore build: phase_sum('y') -> variants._pressure_geometric", desc)
                put(f"E|geometric|{agn}|{sname}|{phase}|{g}|{norm_e}", V._density_geometric(d, ne, vol, sc, "electrons"),
                    "phase_sum('electrons') -> variants._density_geometric(kind='electrons')", desc)
                gnorm = "rhoc_fb" if base == "x200" else "B24"
                put(f"G|geometric|{agn}|{sname}|{phase}|{g}|{gnorm}", V._density_geometric(d, m, vol, sc, "gas"),
                    "phase_sum('mass') -> variants._density_geometric(kind='gas')", desc)
                if sname == "all" and phase in ("xray", "cold_sf"):
                    ekt = rb(self.phase_sum(base, "electron_kT", sel, w))
                    vv = rb(self.phase_sum(base, "volume", sel, w))
                    put(f"P|Q_V|{agn}|{sname}|{phase}|{g}|{norm_p}", V._ratio(ekt, vv * P.MPC_M**3) / sc["pressure"][:, None],
                        "phase_sum(electron_kT)/phase_sum(volume) (variants Q_V maths)", desc)
                    tn = self.table["T200_K"] if base == "x200" else self.table["T500_K"]
                    tnorm = "T200" if base == "x200" else "T500"
                    put(f"T_e|weighted|{agn}|{sname}|{phase}|{g}|{tnorm}",
                        V._ratio(ekt, ne) / P.Boltzmann / tn[:, None], "electron_kT/electrons over the phase", desc)
                    mt = rb(self.phase_sum(base, "mass_t", sel, w))
                    put(f"T_M|weighted|{agn}|{sname}|{phase}|{g}|{tnorm}", V._ratio(mt, m) / tn[:, None],
                        "mass_t/mass over the phase", desc)
        log(f"{base}: phases done ({len(self.index) - n0} arrays, {time.perf_counter() - t:.0f} s)")

        # (4) temperatures, entropy, clumping (selection 'all', phase 'all')
        sel_all = V.selection("all")
        for g in grids:
            idx = self.grid_index(g)
            edges = self.grid_edges(g)
            vol = V._volume(d, base, edges, None)
            tn = self.table["T200_K"] if base == "x200" else self.table["T500_K"]
            tnorm = "T200" if base == "x200" else "T500"
            knorm = "K200" if base == "x200" else "K500"
            ne_ref = self.table["ne200_SS_m3"] if base == "x200" else self.table["ne500_B24_m3"]
            for agn in ("removed", "kept"):
                for kind in ("T_M", "T_V", "T_X"):
                    put(f"{kind}|weighted|{agn}|all|all|{g}|{tnorm}",
                        V.diagnostic(d, kind, agn, "all", self.gridspec[g]) / tn[:, None],
                        f"variants.diagnostic(d, '{kind}', '{agn}', 'all', {g!r}) / T{tnorm[1:]}",
                        "T_X: X-ray phase only (agn token has no effect)" if kind == "T_X" else "")
                ekt = d.rebin(V.owner_sum(d, base, "electron_kT", sel_all, agn), idx)
                ne = d.rebin(V.owner_sum(d, base, "electrons", sel_all, agn), idx)
                te = V._ratio(ekt, ne) / P.Boltzmann / tn[:, None]
                put(f"T_e|weighted|{agn}|all|all|{g}|{tnorm}", te,
                    "variants.owner_sum electron_kT / electrons / k_B / T_norm")
                n_e = ne / (vol * P.MPC_M**3)                       # m^-3, geometric
                with np.errstate(divide="ignore", invalid="ignore"):
                    kk = np.where(n_e > 0, te / (n_e / ne_ref[:, None])**(2 / 3), np.nan)
                put(f"K|weighted|{agn}|all|all|{g}|{knorm}", kk,
                    "(kT_e/kT_norm) / (n_e/n_e,norm)^(2/3), n_e geometric",
                    "entropy; NaN where the shell holds no electrons")
                for kind, tok in (("clumping_density", "C_rho"), ("clumping_pressure", "C_P")):
                    put(f"{tok}|weighted|{agn}|all|all|{g}|none", V.diagnostic(d, kind, agn, "all", self.gridspec[g]),
                        f"variants.diagnostic(d, '{kind}', '{agn}', 'all', {g!r})")
        log(f"{base}: temperatures/entropy/clumping done ({len(self.index) - n0} arrays, {time.perf_counter() - t:.0f} s)")

    # ------------------------------------------------------------------ partition / excision (x200 only)
    def partition_excision(self):
        d = self.d
        t = time.perf_counter()
        n0 = len(self.index)
        for g in X200_GRIDS:
            for sel in ("A2ii", "A2ii_A2i"):
                for agn in ("removed", "kept"):
                    for field in ("P", "E", "G"):
                        norm = "P200" if field == "P" else "rhoc_fb"
                        key = f"{field}|geometric|{agn}|{sel}|all|{g}|{norm}"
                        vals = V.profile(d, FIELD_LIB[field], "geometric", agn, SELECTIONS[sel][0], self.gridspec[g]).values
                        self.save(key, vals, selection_spec=SELECTIONS[sel][0], selection_label=SELECTIONS[sel][2],
                                  units=self.units(field, norm), description="partition block (reduced quantities)",
                                  library_call=f"variants.profile(d, '{FIELD_LIB[field]}', 'geometric', '{agn}', "
                                               f"'{SELECTIONS[sel][0]}', {g!r})")
            # Q_V densities exist for the partition block too
            for field in ("E", "G"):
                norm = "rhoc_fb"
                key = f"{field}|Q_V|removed|A2ii|all|{g}|{norm}"
                vals = V.profile(d, FIELD_LIB[field], "Q_V", "removed", "partition_owned", self.gridspec[g]).values
                self.save(key, vals, selection_spec="partition_owned", selection_label=SELECTIONS["A2ii"][2],
                          units=self.units(field, norm), library_call=f"variants.profile(d, '{FIELD_LIB[field]}', 'Q_V', "
                                                                      f"'removed', 'partition_owned', {g!r})", description="")
        d.release("partition")
        log(f"partition done ({len(self.index) - n0} arrays, {time.perf_counter() - t:.0f} s)")
        retained = {}
        for g in X200_GRIDS:
            for sel in ("exc_all_k0.5", "exc_all_k1", "exc_all_k2", "exc_nbr_k1"):
                exc = SELECTIONS[sel][0]
                for field in ("P", "E", "G"):
                    norm = "P200" if field == "P" else "rhoc_fb"
                    key = f"{field}|geometric|removed|{sel}|all|{g}|{norm}"
                    p = V.profile(d, FIELD_LIB[field], "geometric", "removed", "all", self.gridspec[g], excision=exc)
                    self.save(key, p.values, selection_spec={"selection": "all", "excision": exc},
                              selection_label=SELECTIONS[sel][2], units=self.units(field, norm),
                              library_call=f"variants.profile(d, '{FIELD_LIB[field]}', 'geometric', 'removed', 'all', "
                                           f"{g!r}, excision={exc})",
                              description="retained-volume corrected: retained sum / (V_shell (1 - f_exc))")
                    if field == "P":
                        retained[f"{sel}|{g}"] = p.retained_fraction
        for k, v in retained.items():
            sel, g = k.split("|")
            np.save(CACHE / f"retained__{sel}__{g}.npy", np.ascontiguousarray(v, dtype=np.float64))
        d.release("excision")
        log(f"excision done ({len(self.index) - n0} arrays, {time.perf_counter() - t:.0f} s)")

    # ------------------------------------------------------------------ fully ionised electrons (aux)
    def fullion(self, base):
        d, P = self.d, self.P
        aux = d.array("aux200" if base == "x200" else "aux500")
        ai = list(d.names["aux"]).index("fully_ionized_electrons")
        for g in (X200_GRIDS if base == "x200" else X500_GRIDS):
            idx = self.grid_index(g)
            edges = self.grid_edges(g)
            vol = V._volume(d, base, edges, None)
            norm = "rhoc_fb" if base == "x200" else "B24"
            sc = self.scales(base, "P200" if base == "x200" else "B24")
            for agn, agn_idx in (("removed", [0]), ("kept", [0, 1])):
                lin = d.rebin(aux[:, agn_idx, ai, :].sum(axis=1), idx)
                self.save(f"E_fullion|geometric|{agn}|all|all|{g}|{norm}",
                          V._density_geometric(d, lin, vol, sc, "electrons"),
                          selection_spec="all (aux block is not split by owner or phase; AGN axis only)",
                          selection_label=SELECTIONS["all"][2], units=self.units("E_fullion", norm),
                          library_call=f"aux{base[1:]}[:, agn, 'fully_ionized_electrons'] -> variants._density_geometric",
                          description="electrons assuming full ionisation of H and He (upper bound on n_e)")

    # ------------------------------------------------------------------ resolution ingredients
    def counts_hsml(self, base):
        d = self.d
        block = d.array("owner200" if base == "x200" else "owner500")
        qc = d.index("quantities", "count")
        count = block[:, :, :, qc, :].sum(axis=(1, 2))          # [n, base shells], all owners and phases
        hs = d.array("hs200" if base == "x200" else "hs500")    # [n, base shells, 384] uint32
        hs_edges = d.array("hs_edges")
        out = {}
        for g in (X200_GRIDS if base == "x200" else X500_GRIDS):
            idx = self.grid_index(g)
            c = d.rebin(count, idx)
            np.save(CACHE / f"counts_{g}.npy", np.ascontiguousarray(c, dtype=np.float64))
            hsml = np.empty(c.shape)
            for a in range(0, self.n, 512):                     # chunks keep the int64 copy small
                h = np.moveaxis(hs[a:a + 512], -2, -1).astype(np.int64)      # [h, nbin, shell]
                h = np.moveaxis(d.rebin(h, idx), -1, -2)                     # [h, shell, nbin]
                hsml[a:a + 512] = aio.upper_median_from_hist(h, hs_edges)
            np.save(CACHE / f"hsml_{g}.npy", np.ascontiguousarray(hsml, dtype=np.float64))
            out[g] = {"counts_file": f"counts_{g}.npy", "hsml_file": f"hsml_{g}.npy", "shells": int(c.shape[1]),
                      "empty_halo_shells": int(np.sum(c == 0))}
        return out

    # ------------------------------------------------------------------ halo table
    def build_table(self):
        d, P = self.d, self.P
        h = d.halos
        bt = dict(np.load(STUDY / "B_covariance" / "results" / "halo_table.npz"))
        fp = np.load(STUDY / "F_observables" / "results" / "profiles.npz")
        if not np.array_equal(bt["orig_row"], np.arange(self.n)) or not np.array_equal(fp["halo_rows"], np.arange(self.n)):
            raise ValueError("B / F halo order differs from the study order")
        m200, r200 = np.asarray(h["m200c_msun"], float), np.asarray(h["r_200_crit_mpc"], float)
        m500, r500 = np.asarray(h["m500c_msun"], float), np.asarray(h["r_500_crit_mpc"], float)
        kT200 = MU_T * P.proton_mass * P.G * m200 * P.MSUN_KG / (2 * r200 * P.MPC_M)
        rho_c200 = d.rho_c("x200") * P.MSUN_KG / P.MPC_M**3                      # kg m^-3
        tab = {
            "row": np.arange(self.n), "soap_row": np.asarray(h["soap_row"]), "halo_index": np.asarray(h["halo_index"]),
            "m200c": m200, "r200c": r200, "m500c": m500, "r500c": r500,
            "logm200c": np.log10(m200), "logm500c": np.log10(m500),
            "weight": np.asarray(h["weight"], float), "inclusion_probability": np.asarray(h["inclusion_probability"], float),
            "mass_bin": np.asarray(h["mass_bin"]).astype(np.int64), "mass_edges": np.asarray(h["mass_edges"], float),
            "class_main": np.asarray(h["census_class_main"]).astype(np.int64),
            "role": np.asarray(h["role"]).astype(np.int64),
            "concentration_200c": np.asarray(h["concentration_200c"], float),
            "dm_concentration_200c": np.asarray(h["dm_concentration_200c"], float),
            "centre_mpc": np.asarray(h["centre_mpc"], float), "centre_cmpc": np.asarray(h["centre_cmpc"], float),
            "block": np.asarray(bt["block"]).astype(np.int64), "fold": np.asarray(fp["fold"]).astype(np.int64),
            "block_fold": np.asarray(fp["block_fold"]).astype(np.int64),
            "T200_K": kT200 / P.Boltzmann, "T500_K": d.kT500() / P.Boltzmann,
            "p200_pa": d.p200(), "p500_B24_pa": d.p500("B24"), "p500_A10_pa": d.p500("A10"),
            "p500_ss_pa": d.p500("self_similar_500"),
            "ne200_SS_m3": 200 * d.baryon_fraction * rho_c200 / (MU_E_K * P.proton_mass),
            "ne500_B24_m3": d.ne500("B24"),
            "eps_over_r200c": d.softening_mpc / r200, "eps_over_r500c": d.softening_mpc / r500,
            "y200_soap_noagn_mpc2": np.asarray(h["y_noagn_200_mpc2"], float),
            "y500_soap_noagn_mpc2": np.asarray(h["y_noagn_500_mpc2"], float),
            "mass_fraction_satellites_200c": np.asarray(h["mass_fraction_satellites_200c"], float),
            "nearest_massive_distance_mpc": np.asarray(h["nearest_massive_distance_mpc"], float),
            "census_max_mu": np.asarray(h["census_max_mu"], float),
        }
        for k in ("class_mu0.03", "class_mu0.3", "class_neighbours_mu0.1"):
            tab[k.replace(".", "p")] = np.asarray(bt[k]).astype(np.int64)
        # checks against B's table
        for mine, theirs in (("weight", "weight"), ("mass_bin", "mass_bin"), ("class_main", "class_main"),
                             ("m200c", "m200c"), ("r200c", "r200c"), ("T200_K", "T200_K"), ("p200_pa", "p200_pa"),
                             ("concentration_200c", "concentration_200c")):
            if not np.allclose(tab[mine], bt[theirs], rtol=1e-12, atol=0, equal_nan=True):
                raise ValueError(f"halo table {mine} differs from B_covariance halo_table")
        # own integrated Y (sum of particle y) and electrons inside R200c / R500c, recent AGN removed, all gas
        sel = V.selection("all")
        y200 = V.owner_sum(d, "x200", "y", sel, "removed")
        x200 = d.grid("x200")
        tab["Y200_own_mpc2"] = y200[:, x200[1:] <= 1.0 + 1e-12].sum(axis=1)
        tab["Y4R_own_mpc2"] = y200.sum(axis=1)
        tab["Y200_own_agnkept_mpc2"] = V.owner_sum(d, "x200", "y", sel, "kept")[:, x200[1:] <= 1.0 + 1e-12].sum(axis=1)
        ne200 = V.owner_sum(d, "x200", "electrons", sel, "removed")
        tab["Ne200_own"] = ne200[:, x200[1:] <= 1.0 + 1e-12].sum(axis=1)
        tab["Ne4R_own"] = ne200.sum(axis=1)
        self.table = tab
        np.savez(CACHE / "halo_table.npz", **tab)
        return tab

    def y500(self):
        d = self.d
        x500 = d.grid("x500")
        y = V.owner_sum(d, "x500", "y", V.selection("all"), "removed")
        self.table["Y500_own_mpc2"] = y[:, x500[1:] <= 1.0 + 1e-12].sum(axis=1)
        np.savez(CACHE / "halo_table.npz", **self.table)

    # ------------------------------------------------------------------ main
    def run(self):
        d = self.d
        CACHE.mkdir(parents=True, exist_ok=True)
        summary = {"study": str(STUDY), "n_haloes": self.n, "grids": {}}
        mark = time.perf_counter()
        for g in self.gridspec:
            e = self.grid_edges(g)
            np.save(CACHE / f"edges_{g}.npy", e)
            summary["grids"][g] = {"base": self.base[g], "shells": len(e) - 1, "inner": float(e[0]), "outer": float(e[-1]),
                                   "radius": "R200c" if self.base[g] == "x200" else "R500c",
                                   "spec": self.gridspec[g] if isinstance(self.gridspec[g], str) else "('x200', B fit edges)"}
        bfit = np.load(STUDY / "B_covariance" / "results" / "profiles_fit.npz")
        if not np.array_equal(self.grid_edges("fit"), bfit["stored_edges"]):
            raise ValueError("fit edges differ from B_covariance profiles_fit stored_edges")
        fprof = np.load(STUDY / "F_observables" / "results" / "profiles.npz")
        if not np.array_equal(self.grid_edges("fit"), fprof["f_edges_data"]):
            raise ValueError("fit edges differ from the F grid")
        log("loading owner200 ...")
        d.array("owner200")
        self.timings["load_owner200"] = time.perf_counter() - mark
        mark = time.perf_counter()
        self.build_table()
        self.timings["halo_table"] = time.perf_counter() - mark
        log(f"halo table done ({self.timings['halo_table']:.1f} s), rss {rss_gb():.2f} GB")
        mark = time.perf_counter()
        self.owner_variants("x200")
        self.timings["x200_owner_variants"] = time.perf_counter() - mark
        mark = time.perf_counter()
        summary["resolution_files"] = self.counts_hsml("x200")
        self.timings["x200_counts_hsml"] = time.perf_counter() - mark
        log(f"x200 counts/hsml done ({self.timings['x200_counts_hsml']:.1f} s), rss {rss_gb():.2f} GB")
        d.release("owner200", "hs200")
        mark = time.perf_counter()
        self.partition_excision()
        self.timings["partition_excision"] = time.perf_counter() - mark
        mark = time.perf_counter()
        self.fullion("x200")
        d.release("aux200")
        log("loading owner500 ...")
        d.array("owner500")
        self.y500()
        self.owner_variants("x500")
        summary["resolution_files"].update(self.counts_hsml("x500"))
        self.fullion("x500")
        d.release()
        self.timings["x500_all"] = time.perf_counter() - mark
        log(f"x500 done ({self.timings['x500_all']:.1f} s), rss {rss_gb():.2f} GB")
        # index
        conventions = {
            "key": "field|estimator|agn|selection|phase|grid|norm",
            "rows": "4588 haloes in STUDY/selected_halos.npz order (= B_covariance halo_table orig_row = F profiles "
                    "halo_rows = cache/halo_table.npz row)",
            "fields": {"P": "electron pressure", "E": "free-electron density (SWIFT ElectronNumberDensities x V)",
                       "G": "gas mass density", "E_fullion": "fully ionised electron density (aux block)",
                       "T_M": "mass-weighted temperature", "T_V": "SPH-volume-weighted temperature",
                       "T_X": "ROSAT-luminosity-weighted temperature of the X-ray phase",
                       "T_e": "electron-weighted temperature (electron_kT / electrons)",
                       "K": "entropy kT_e n_e^(-2/3)", "C_rho": "density clumping", "C_P": "pressure clumping"},
            "estimators": {
                "geometric": "linear sum / geometric shell volume (tSZ/DM-correct; P: sum y / V_shell x m_e c^2/sigma_T)",
                "Q_V": "SPH-volume weighted: sum(V q)/sum(V) (P: electron_kT/volume = <n_e k T>_V)",
                "Q_V_y": "sum(y m_e c^2/sigma_T)/sum(V) (Q_V with the y-based particle pressure)",
                "Q_M": "mass weighted: sum(m q)/sum(m) (P: mass_pe/mass; G: sum(m rho)/sum(m), clumping-biased)",
                "Q_X": "ROSAT-luminosity weighted pressure over the X-ray phase: lx_pe/lx_rosat",
                "nT_X": "geometric n_e x k T_X (T_X = lx_t/lx_rosat of the X-ray phase)",
                "weighted": "ratio of linear sums (temperatures, entropy, clumping)"},
            "agn": AGN, "phases": PHASES,
            "selections": {k: {"spec": v[0], "block": v[1], "label": v[2]} for k, v in SELECTIONS.items()},
            "grids": summary["grids"],
            "norms": {"P200": "pressure / P200 (x200 radii234 convention); on x500 grids: P/P200 versus r/R500c",
                      "rhoc_fb": "density / (rho_c f_b) (E: n_e mu_XGP/(rho_c f_b), the radii234/XGPaint convention)",
                      "B24": "Braspenning+24 500c scales", "A10": "Arnaud+10 P500 (pressure only)",
                      "self_similar_500": "G M500 rho500 f_b/(2 R500) (pressure only)", "T200": "k T200 (mu = 0.59)",
                      "T500": "k T500 (B24)", "K200": "self-similar K200", "K500": "self-similar K500 (B24)",
                      "none": "dimensionless"},
            "nan": "NaN marks halo-shells where a ratio is undefined (no particles of the required phase, Q_X with no "
                   "X-ray gas) or excision retained < 50% of the shell volume; geometric P/E/G are never NaN",
            "resolution": "counts_<grid>.npy (all gas, all owners and phases) and hsml_<grid>.npy (upper-median h a/R), "
                          "softening 0.0057 Mpc physical (halo_table eps_over_r200c / eps_over_r500c)",
            "retained": "retained__<excision selection>__<grid>.npy: retained volume fraction of each halo-shell",
        }
        index = {"conventions": conventions, "softening_mpc": d.softening_mpc, "baryon_fraction": d.baryon_fraction,
                 "hubble": d.hubble(), "variants": self.index, "resolution_files": summary["resolution_files"],
                 "halo_table": "halo_table.npz", "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (CACHE / "index.json").write_text(json.dumps(index, indent=1, default=str) + "\n")
        self.timings["total"] = time.perf_counter() - self.t0
        summary.update(variants=len(self.index), timings_s=self.timings, peak_rss_gb=rss_gb(),
                       cache_bytes=int(sum(p.stat().st_size for p in CACHE.iterdir() if p.is_file())))
        (CACHE / "build_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        log(f"done: {len(self.index)} variants, {self.timings['total']:.0f} s, peak RSS {rss_gb():.2f} GB, "
            f"{summary['cache_bytes'] / 1e9:.2f} GB on disk")


if __name__ == "__main__":
    Builder().run()
