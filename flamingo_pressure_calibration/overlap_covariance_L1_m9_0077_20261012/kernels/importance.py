"""Per-shell and per-mass-bin importance kernels (Step 0.1, task 3 + extension).

Reads the Jacobians of jacobians.py (jacobians_engine_output.npz, per grid
segment, so the coarse shells are exact unions) and reports the normalised
sensitivity of every observable to the profile in each radial shell
[0,0.1],[0.1,0.25],[0.25,0.5],[0.5,1],[1,1.5],[1.5,2],[2,3],[3,4] R200c and in
each 0.2-dex mass bin, per ell (100, 300, 1000, 3000, 8000) for spectra and per
quantile for the DM PDF.

normalised = J/(k O): k = 2 for quantities quadratic in F (auto C_ell 1h and
2h, kSZ, DM variance), k = 1 for linear ones (means, Y, N_e), each side of the
bilinear y x DM cross, and the DM quantiles (degree-1 homogeneous). Every
normalised map sums to 1 over (shell, mass bin). Raw values are dO/dln F_s.
Outputs: importance.npz, importance.csv, importance_summary.json.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_engine", HERE/"engine.py")
E = importlib.util.module_from_spec(spec)
sys.modules["kernel_engine"] = E
spec.loader.exec_module(E)

import numpy as np

SELECT_ELL = np.array([100., 300., 1000., 3000., 8000.])
# (key in engine output, J key, profile kind, k, description)
OBSERVABLES = (
    ("cl_1h", "one", "P", 2, "tSZ yy 1-halo"),
    ("cl_2h", "two", "P", 2, "tSZ yy linear 2-halo"),
    ("cl_total", None, "P", 2, "tSZ yy total"),
    ("mean", "mean", "P", 1, "mean Compton y"),
    ("ksz_cl_1h", "ksz_one", "E", 2, "kSZ 1-halo (catalogue v_LOS)"),
    ("ksz_cl_1h_linear_sigma_v", "ksz_one_lin", "E", 2, "kSZ 1-halo (linear sigma_v)"),
    ("ksz_cl_2h_approx", "ksz_two", "E", 2, "kSZ 2-halo (approximate)"),
    ("dm_cl_1h", "dm_one", "E", 2, "DM field 1-halo (z_s=1)"),
    ("dm_cl_2h", "dm_two", "E", 2, "DM field linear 2-halo"),
    ("cl_1h", "one", "E", 2, "tau 1-halo (all z)"),
    ("dm_mean", "dm_mean", "E", 1, "mean halo DM (z_s=1)"),
    ("dm_var", "dm_var", "E", 2, "DM variance (compound Poisson)"),
    ("dm_quantiles", "dm_quantile", "E", 1, "DM PDF quantile"),
)


def coarse_map(segment_edges):
    mids = segment_edges.mean(axis=1)
    return np.searchsorted(E.SHELL_EDGES, mids)-1


def main():
    data = np.load(HERE/"jacobians_engine_output.npz")          # lazy: only arrays used are read
    jac = dict(np.load(HERE/"jacobians.npz"))
    labels = list(data["labels"])
    ell = data["ell"]
    sel = np.array([int(np.flatnonzero(ell == e)[0]) for e in SELECT_ELL])
    mapping = coarse_map(jac["segment_edges"])
    ns, nb = len(E.SHELL_EDGES)-1, len(jac["mass_bin_labels"])
    shells_text = [f"{lo:g}-{hi:g}" for lo, hi in zip(E.SHELL_EDGES[:-1], E.SHELL_EDGES[1:])]
    bins_text = [str(b) for b in jac["mass_bin_labels"]]

    def coarse(array, axis=0):
        moved = np.moveaxis(array, axis, 0)
        out = np.zeros((ns,)+moved.shape[1:])
        np.add.at(out, mapping, moved)
        return np.moveaxis(out, 0, axis)

    maps, rows, values = {}, [], {}
    for profile in ("P2", "P4", "E2", "E4"):
        v = labels.index(profile)
        for key, jkey, kind, k, text in OBSERVABLES:
            if kind != profile[0]:
                continue
            if key == "cl_total":
                raw = data[f"J__{profile}__one"]+data[f"J__{profile}__two"]
                value = data["cl_1h"][v]+data["cl_2h"][v]
            else:
                raw = data[f"J__{profile}__{jkey}"]
                value = data[key][v]
            if key == "dm_quantiles":
                raw = np.moveaxis(raw, 0, -1)                     # [segment, bin, quantile]
            raw = coarse(raw)                                     # [shell, bin(, ell or quantile)]
            if raw.ndim == 3 and key != "dm_quantiles":
                raw, value = raw[..., sel], value[sel]
            name = f"{profile}:{key}" if not (profile.startswith("E") and key == "cl_1h") else f"{profile}:tau_cl_1h"
            normalised = raw/(k*value)
            maps[name] = dict(raw=raw, normalised=normalised, value=value, text=text, k=k)
    for pair in ("P2xE2", "P4xE4"):
        p = list(data["pair_labels"]).index(pair)
        for term, key in (("1h", "cross_cl_1h"), ("2h", "cross_cl_2h")):
            for side in ("P", "E"):
                raw = coarse(data[f"Jpair__{pair}__cross_{term}_{side}"])[..., sel]
                value = data[key][p][sel]
                maps[f"{pair}:cross_{term}_{side}"] = dict(raw=raw, normalised=raw/value, value=value, k=1,
                                                          text=f"y x DM {term} wrt {'pressure' if side == 'P' else 'electron'} shells")
    # Sum checks: every normalised map sums to 1 over (shell, bin).
    sums = {name: float(np.max(np.abs(m["normalised"].sum(axis=(0, 1))-1))) for name, m in maps.items()}
    for name, m in maps.items():
        profile, key = name.split(":")
        columns = ([f"ell={int(e)}" for e in SELECT_ELL] if m["normalised"].ndim == 3 and "quantile" not in key
                   else [f"q={q:g}" for q in E.QUANTILES] if "quantile" in key else ["value"])
        normal = m["normalised"] if m["normalised"].ndim == 3 else m["normalised"][..., None]
        raw = m["raw"] if m["raw"].ndim == 3 else m["raw"][..., None]
        for c, column in enumerate(columns):
            for s in range(ns):
                rows.append(dict(profile=profile, observable=key, description=m["text"], at=column, dimension="shell",
                                 category=shells_text[s], raw_dO_dlnF=raw[s, :, c].sum(), normalised=normal[s, :, c].sum()))
            for b in range(nb):
                rows.append(dict(profile=profile, observable=key, description=m["text"], at=column, dimension="mass_bin",
                                 category=bins_text[b], raw_dO_dlnF=raw[:, b, c].sum(), normalised=normal[:, b, c].sum()))
    arrays = dict(shell_edges=E.SHELL_EDGES, shell_labels=np.array(shells_text), mass_bin_labels=np.array(bins_text),
                  mass_bin_calibrated=jac["mass_bin_calibrated"], ell=SELECT_ELL, quantile_levels=E.QUANTILES,
                  catalogue_halos_per_bin=jac["catalogue_halos_per_bin"],
                  convention=np.array("normalised = J/(k O), k=2 quadratic (auto C_ell, kSZ, DM variance), k=1 linear/cross/quantile; "
                                      "maps [shell, mass bin, ell or quantile] sum to 1 over shell and bin"))
    for name, m in maps.items():
        tag = name.replace(":", "__")
        arrays[f"normalised__{tag}"] = m["normalised"]
        arrays[f"raw__{tag}"] = m["raw"]
        arrays[f"value__{tag}"] = m["value"]
    # Y(<4R) and N_e(<4R) shell fractions per 0.2-dex bin centre (z=0, coarse shells).
    support = E.support_definition()
    grid = E.make_grid(4., breaks=list(E.SHELL_EDGES[1:]))
    centres = .5*(support["mass_edges"][1:]+support["mass_edges"][:-1])
    fits = dict(P2=E.pressure_fit(2, support), P4=E.pressure_fit(4, support),
                E2=E.electron_fit(2, support), E4=E.electron_fit(4, support))
    for profile, fit in fits.items():
        base = fit.base(10**centres, grid.x)
        pieces = np.stack([E.volume_integral(grid, base, hi, lo) for lo, hi in E.SHELLS], axis=-1)
        arrays[f"flux_fraction__{profile}"] = pieces/pieces.sum(axis=1, keepdims=True)
        arrays[f"flux_source__{profile}"] = np.where(fit.bins(10**centres)[1], "fit", "published fallback")
        for b, centre in enumerate(centres):
            for sh in range(ns):
                rows.append(dict(profile=profile, observable="Y(<4R)" if profile[0] == "P" else "N_e(<4R)",
                                 description="z=0 shell fraction of the 4R integral", at=f"logM={centre:.1f}",
                                 dimension="shell", category=shells_text[sh], raw_dO_dlnF=pieces[b, sh],
                                 normalised=pieces[b, sh]/pieces[b].sum()))
    arrays["flux_logmass_centres"] = centres
    with (HERE/"importance.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8g}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})
    np.savez_compressed(HERE/"importance.npz", **arrays)
    summary = dict(sum_check_max_abs_deviation=max(sums.values()), observables=list(maps),
                   inner_fractions={}, top_mass_bins={})
    cumulative_edges = (.5, 1., 2.)
    for name, m in maps.items():
        normal = m["normalised"] if m["normalised"].ndim == 3 else m["normalised"][..., None]
        by_shell = normal.sum(axis=1)                              # [shell, column]
        cum = np.cumsum(by_shell, axis=0)
        summary["inner_fractions"][name] = {f"x<{edge:g}": cum[list(E.SHELL_EDGES[1:]).index(edge)].tolist()
                                            for edge in cumulative_edges}
        by_bin = normal.sum(axis=0)                                # [bin, column]
        summary["top_mass_bins"][name] = [bins_text[int(i)] for i in np.argmax(by_bin, axis=0)]
    calibrated = np.asarray(jac["mass_bin_calibrated"], bool)
    bands = dict(core_0_0p1=[0], inner_0p1_0p5=[1, 2], mid_0p5_1=[3], outer_1_2=[4, 5], far_2_4=[6, 7])
    summary["radial_bands"], summary["calibrated_bin_share"] = {}, {}
    for name, m in maps.items():
        normal = m["normalised"] if m["normalised"].ndim == 3 else m["normalised"][..., None]
        by_shell = normal.sum(axis=1)
        summary["radial_bands"][name] = {band: by_shell[idx].sum(axis=0).tolist() for band, idx in bands.items()}
        summary["calibrated_bin_share"][name] = normal[:, calibrated].sum(axis=(0, 1)).tolist()
    summary["columns"] = dict(spectra=[f"ell={int(e)}" for e in SELECT_ELL], dm_quantiles=[f"q={q:g}" for q in E.QUANTILES],
                              scalars=["value"])
    summary["calibrated_bins_note"] = ("calibrated = 0.2-dex bins 13.5-15.5 with the fitted shape (logM<=15.388 within the last bin); "
                                       "other bins use the published B12/B16 fallback; the >=15.5 overflow bin is fallback")
    E.write_json(HERE/"importance_summary.json", summary)
    print(json.dumps(E.C.plain(dict(sum_check=summary["sum_check_max_abs_deviation"], observables=len(maps))), indent=1))


if __name__ == "__main__":
    main()
