"""Test K stages K0 (inventory) and K1 (reads + per-cell reduction), run on STRW maas (read-only /net/hypernova).

K0: for each of the 71 fiducial SOAP-HBT files: size, Header (Redshift, Scale-factor, NumSubhalos_Total), Cosmology,
    RecentlyHeatedGasFilter, presence of every column Test K reads (dtype / chunking / conversion attributes for 0007,
    0037 and 0077; K1 records the conversion attributes of every output) (headers
    and dataset metadata only); plus a smoke read of halo_properties_0077 (IsCentral, M200c, one SO column).
K1: one worker per file.  Selects central subhaloes (InputHalos/IsCentral == 1) with M200c >= 10^12.7 Msun, reads
    every column of k_common.COLUMNS over the contiguous row span of the selection (full decompression of those
    chunks), converts to physical units with the per-column 'Conversion factor to physical CGS (including
    cosmological corrections)', normalises per halo (Ytilde, Mtilde; k_common docstring), writes the per-halo arrays
    to the node-local scratch (float32) and the per-cell reduction to results/cells/cells_<snap>.npz:
      per (0.2-dex mass bin) cell and aperture: inclusion-free mean over all valid centrals of Ytilde_noAGN,
      Ytilde (with AGN), Mtilde; 200 bootstrap replicate means (halo resampling, shared by the three quantities);
      32 quadrature nodes of (ln M200c, x) (k_fit_core.strata_groups, 4 mass x 8 radius strata); x statistics;
      median gas-particle count; valid
      fraction; the mean over haloes with >= 50 gas particles (diagnostic of a halo-level mask); per-halo-shell
      display profiles (estimator B of the SOAP audit) between consecutive usable apertures; the compression error
      of the 32-node quadrature for the published laws; a random subsample of up to 256 haloes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import k_common as KC
import k_fit_core as K

ATTR_CGS = "Conversion factor to physical CGS (including cosmological corrections)"
ATTR_CGS_NC = "Conversion factor to CGS (not including cosmological corrections)"


def soap_path(snap):
    return f"{KC.SOAP_DIR}/halo_properties_{snap:04d}.hdf5"


def _attr(a, key, default=None):
    if key not in a:
        return default
    v = a[key]
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    v = np.ravel(v)
    if v.dtype.kind in "SO":
        return str(v[0].decode() if isinstance(v[0], bytes) else v[0])
    return float(v[0]) if len(v) == 1 else v.tolist()


def header(f):
    h = f["Header"].attrs
    out = {k: _attr(h, k) for k in ("Redshift", "Scale-factor", "NumSubhalos_Total", "RunName")}
    out["cosmology"] = {k: _attr(f["Cosmology"].attrs, k) for k in f["Cosmology"].attrs.keys()}
    out["heating_filter"] = {k: _attr(f["Parameters/RecentlyHeatedGasFilter"].attrs, k)
                             for k in f["Parameters/RecentlyHeatedGasFilter"].attrs.keys()} \
        if "Parameters/RecentlyHeatedGasFilter" in f else None
    out["units"] = {k: _attr(f["Units"].attrs, k) for k in f["Units"].attrs.keys()}
    return out


def k0_inventory(snaps):
    import h5py
    res = {"files": {}, "missing_files": [], "t0": time.time()}
    for snap in snaps:
        p = soap_path(snap)
        if not os.path.exists(p):
            res["missing_files"].append(p)
            continue
        t0 = time.time()
        with h5py.File(p, "r") as f:
            item = {"path": p, "bytes": os.path.getsize(p), **header(f), "columns": {}, "missing_columns": []}
            detail = snap in (7, 37, 77)                  # full column metadata for three files; presence for all
            for col in KC.COLUMNS:
                if col not in f:
                    item["missing_columns"].append(col)
                    continue
                if not detail:
                    continue
                d = f[col]
                a = d.attrs
                item["columns"][col] = {"dtype": str(d.dtype), "shape": list(d.shape), "chunks": list(d.chunks or []),
                                        "cgs": _attr(a, ATTR_CGS), "cgs_nocosmo": _attr(a, ATTR_CGS_NC),
                                        "a_exp": _attr(a, "a-scale exponent"),
                                        "physical": _attr(a, "Value stored as physical")}
        item["seconds"] = time.time() - t0
        res["files"][f"{snap:04d}"] = item
        KC.log(f"K0 {snap:04d} z={item['Redshift']:.3f} {item['bytes'] / 1e9:.1f} GB missing={len(item['missing_columns'])}")
    res["wall_s"] = time.time() - res.pop("t0")
    return res


def k0_smoke_read(snap=77, limit=None):
    """IsCentral + M200c + SO/500_crit/ComptonYWithoutRecentAGNHeating of one file; counts above 10^12.7."""
    import h5py
    t0 = time.time()
    with h5py.File(soap_path(snap), "r") as f:
        ic = f["InputHalos/IsCentral"][slice(None, limit)]
        dm = f["SO/200_crit/TotalMass"]
        m = dm[slice(None, limit)].astype(np.float64) * _attr(dm.attrs, ATTR_CGS) / KC.MSUN_G
        sel = (ic == 1) & (m >= 10 ** KC.LOG_MMIN)
        rows = np.nonzero(sel)[0]
        y = f["SO/500_crit/ComptonYWithoutRecentAGNHeating"][slice(None, limit)][rows]
    return {"snap": snap, "rows_read": int(len(ic)), "n_central": int((ic == 1).sum()), "n_sel": int(len(rows)),
            "row_span": [int(rows.min()), int(rows.max())] if len(rows) else None,
            "log10M_max": float(np.log10(m[rows].max())) if len(rows) else None,
            "y500_noagn_positive_frac": float(np.mean(y > 0)) if len(rows) else None, "seconds": time.time() - t0}


# ------------------------------------------------------------------------------------------------ K1 per file
def read_file(snap, limit=None):
    """Selected centrals of one SOAP file: dict of physical per-halo arrays + metadata."""
    import h5py
    t0 = time.time()
    meta = {"snap": snap, "path": soap_path(snap), "column_seconds": {}, "conversion": {}}
    with h5py.File(soap_path(snap), "r") as f:
        meta.update(header(f))
        nrow = int(f["InputHalos/IsCentral"].shape[0]) if limit is None else int(limit)
        ic = f["InputHalos/IsCentral"][:nrow]
        dm = f["SO/200_crit/TotalMass"]
        m200 = dm[:nrow].astype(np.float64) * _attr(dm.attrs, ATTR_CGS) / KC.MSUN_G
        sel = (ic == 1) & (m200 >= 10 ** KC.LOG_MMIN)
        rows = np.nonzero(sel)[0]
        meta["n_rows"] = nrow
        meta["n_central_all"] = int((ic == 1).sum())
        meta["n_selected"] = int(len(rows))
        del ic
        out = {"soap_row": rows.astype(np.int64), "M200c": m200[rows]}
        del m200
        if not len(rows):
            meta["seconds_read"] = time.time() - t0
            return out, meta
        lo, hi = int(rows.min()), int(rows.max()) + 1
        meta["row_span"] = [lo, hi]
        rel = rows - lo
        for col in KC.COLUMNS[2:]:
            tc = time.time()
            if col not in f:
                meta.setdefault("missing_columns", []).append(col)
                continue
            d = f[col]
            v = d[lo:hi][rel]
            a = d.attrs
            cgs = _attr(a, ATTR_CGS)
            meta["conversion"][col] = {"cgs": cgs, "cgs_nocosmo": _attr(a, ATTR_CGS_NC), "a_exp": _attr(a, "a-scale exponent"),
                                       "physical": _attr(a, "Value stored as physical"), "dtype": str(d.dtype)}
            if col.endswith("SORadius"):
                v = v.astype(np.float64) * cgs / KC.MPC_CM                  # physical Mpc
            elif "ComptonY" in col:
                v = v.astype(np.float64) * cgs / KC.MPC_CM ** 2             # physical Mpc^2
            elif col.endswith("Mass"):
                v = v.astype(np.float64) * cgs / KC.MSUN_G                  # Msun
            elif col.endswith("Concentration"):
                v = v.astype(np.float64)
            out[col] = v
            meta["column_seconds"][col] = time.time() - tc
    meta["seconds_read"] = time.time() - t0
    return out, meta


def derive(raw, meta):
    """Per-halo normalised arrays: lnM, x [n, 17], Yn/Ya/Mg [n, 17] (Ytilde_noAGN, Ytilde, Mtilde), Ngas [n, 17],
    valid [n, 17] (SO: SORadius > 0; spheres: >= 100 bound particles, the SOAP fill condition)."""
    n = len(raw["soap_row"])
    M = raw["M200c"]
    R = raw["SO/200_crit/SORadius"]
    out = {"soap_row": raw["soap_row"], "lnM": np.log(M), "M500c": raw.get("SO/500_crit/TotalMass", np.full(n, np.nan)),
           "conc": raw.get("SO/200_crit/Concentration", np.full(n, np.nan)),
           "Nbound": raw.get("InputHalos/NumberOfBoundParticles", np.full(n, -1)).astype(np.int64), "R200c": R}
    A = KC.N_AP
    x = np.full((n, A), np.nan)
    Yn = np.zeros((n, A)); Ya = np.zeros((n, A)); Mg = np.zeros((n, A)); Ng = np.zeros((n, A), np.int64)
    valid = np.zeros((n, A), bool)
    Mkg, Rm = M * KC.MSUN_KG, R * KC.MPC_M
    with np.errstate(divide="ignore", invalid="ignore"):
        P200 = KC.G_SI * Mkg * (Mkg / (4 * np.pi / 3 * Rm ** 3)) * KC.F_B / (2 * Rm)      # Pa
        ynorm = KC.MPC_M ** 2 * KC.ME_C2 / KC.SIGMA_T / (P200 * Rm ** 3)              # Ytilde per Mpc^2 of Y
    for j, ap in enumerate(KC.SO):
        r = raw.get(f"SO/{ap}/SORadius")
        if r is None:
            continue
        x[:, j] = r / R
        valid[:, j] = r > 0
        Yn[:, j] = raw[f"SO/{ap}/ComptonYWithoutRecentAGNHeating"] * ynorm
        Ya[:, j] = raw[f"SO/{ap}/ComptonY"] * ynorm
        Mg[:, j] = raw[f"SO/{ap}/GasMass"] / (KC.F_B * M)
        Ng[:, j] = raw[f"SO/{ap}/NumberOfGasParticles"]
    for i, k in enumerate(KC.KPC):
        j = KC.N_SO + i
        g = raw.get(f"InclusiveSphere/{k}kpc/GasMass")
        if g is None:
            continue
        x[:, j] = k / 1e3 / R
        valid[:, j] = out["Nbound"] >= 100
        Mg[:, j] = g / (KC.F_B * M)
        Ng[:, j] = raw[f"InclusiveSphere/{k}kpc/NumberOfGasParticles"]
        Yn[:, j] = np.nan
        Ya[:, j] = np.nan
    out.update(x=x, Yn=Yn, Ya=Ya, Mg=Mg, Ngas=Ng, valid=valid, P200_Pa=P200)
    # checks
    cos = meta.get("cosmology", {})
    zz = float(meta["Redshift"])
    chk = {}
    rho200 = M / (4 * np.pi / 3 * R ** 3) / 200.0                                       # Msun / Mpc^3
    if "Critical density [internal units]" in cos:
        rc = float(cos["Critical density [internal units]"]) * 1e10                    # 1e10 Msun / Mpc^3 internal
        chk["rho200_over_rhocrit_attr_median"] = float(np.median(rho200 / rc))
    if "H [internal units]" in cos:
        Hs = float(cos["H [internal units]"]) * 1e3 / KC.MPC_M                         # km/s/Mpc -> 1/s
        rc2 = 3 * Hs ** 2 / (8 * np.pi * KC.G_SI) * KC.MPC_M ** 3 / KC.MSUN_KG
        chk["rho200_over_rhocrit_H_median"] = float(np.median(rho200 / rc2))
        chk["rho200_over_rhocrit_H_p01_p99"] = [float(np.percentile(rho200 / rc2, 1)), float(np.percentile(rho200 / rc2, 99))]
    ob, om, onu = (float(cos.get(k, np.nan)) for k in ("Omega_b", "Omega_m", "Omega_nu_0"))
    chk["f_b_header_incl_nu"] = ob / (om + onu)
    chk["f_b_used"] = KC.F_B
    chk["z"] = zz
    chk["n_valid_frac_per_aperture"] = valid.mean(0).tolist()
    chk["Nbound_min"] = int(out["Nbound"].min()) if n else None
    chk["x_order_violations_frac"] = float(np.mean(np.diff(x[:, [0, 1, 2, 3]], axis=1) <= 0)) if n else None
    return out, chk


def usable_apertures(stats, field):
    """Cell-level usability per aperture for a field (k_common thresholds); returns bool [A]."""
    A = KC.N_AP
    ok = (stats["frac_valid"] >= KC.MIN_VALID_FRAC) & (stats["med_ngas"] >= KC.MIN_MEDIAN_GAS) & (stats["xbar"] >= KC.MIN_X)
    ok &= np.isfinite(stats["xbar"])
    is_so = np.arange(A) < KC.N_SO
    rk = np.array([np.inf] * KC.N_SO + [k for k in KC.KPC], float)
    if field in ("P", "Pagn"):
        ok &= is_so
    elif field == "Gcore":
        ok &= is_so | (rk >= KC.MIN_R_KPC)
    else:                                   # 'G': spheres only outside the central-galaxy region
        ok &= is_so | ((rk >= KC.MIN_R_KPC_GAS) & (stats["xbar"] >= KC.MIN_X_SPHERE))
    return ok


def dedup_order():
    """Aperture indices in the dedup priority order: SO_PRIORITY, then spheres from small to large."""
    pri = [KC.AP_NAMES.index(n) for n in KC.SO_PRIORITY] + list(range(KC.N_SO, KC.N_AP))
    return pri


def select_points(xbar, usable, names_allowed=None, xmax=None):
    """Usable apertures inside a range (SO by name, spheres by xbar <= xmax), deduplicated (DEDUP_LN)."""
    keep = []
    for j in dedup_order():
        if not usable[j]:
            continue
        name = KC.AP_NAMES[j]
        if names_allowed is not None:
            if j < KC.N_SO and name not in names_allowed:
                continue
            if j >= KC.N_SO and xmax is not None and xbar[j] > xmax:
                continue
        if any(abs(math.log(xbar[j] / xbar[k])) < KC.DEDUP_LN for k in keep):
            continue
        keep.append(j)
    return sorted(keep, key=lambda j: xbar[j])


def display_shells(h, idx, stats, field, qkey):
    """Per-halo shells (estimator B) between consecutive usable, deduplicated apertures (all ranges), mean over the
    cell's haloes with both apertures valid and increasing radius.  P: volume-averaged P_e/P200; G: rho/(f_b rho_c)
    (with the spheres down to 0.025 R200c, 'Gcore' usability, so that the central-galaxy gas is visible)."""
    keep = select_points(stats["xbar"], usable_apertures(stats, "Gcore" if field == "G" else field))
    rows = []
    prev = None
    for j in keep:
        xb = h["x"][idx, j]
        qb = h[qkey][idx, j]
        vb = h["valid"][idx, j]
        if prev is None:
            xa, qa, va = np.zeros(len(idx)), np.zeros(len(idx)), np.ones(len(idx), bool)
        else:
            xa, qa, va = h["x"][idx, prev], h[qkey][idx, prev], h["valid"][idx, prev]
        ok = vb & va & (xb > xa * 1.0001)
        if ok.sum() >= 2:
            dv = xb[ok] ** 3 - xa[ok] ** 3
            val = (qb[ok] - qa[ok]) / (4 * np.pi / 3 * dv) if field == "P" else 200.0 * (qb[ok] - qa[ok]) / dv
            rows.append([prev if prev is not None else -1, j, float(np.mean(xa[ok])), float(np.mean(xb[ok])),
                         float(np.mean(val)), float(np.std(val, ddof=1) / np.sqrt(ok.sum())), int(ok.sum())])
        prev = j
    return rows


def reduce_cells(h, snap, z, rng_seed):
    """Per-cell reduction (see module docstring).  Returns (npz dict, cell summary list)."""
    A, B, NN, NS = KC.N_AP, KC.N_BOOT, KC.N_STRATA_M * KC.N_STRATA_X, KC.N_SUB
    logm = h["lnM"] / math.log(10)
    nb = len(KC.MASS_EDGES) - 1
    bins = np.floor((logm - KC.MASS_EDGES[0]) / 0.2 + 1e-9).astype(int)
    cells = [b for b in range(nb) if np.any(bins == b)]
    nc = len(cells)
    arr = {"cell_bin": np.array(cells, int), "cell_N": np.zeros(nc, int), "cell_mean_lnM": np.zeros(nc),
           "cell_median_lnM": np.zeros(nc), "z": np.full(nc, z), "snap": np.full(nc, snap)}
    for q in ("Yn", "Ya", "Mg"):
        arr[f"mean_{q}"] = np.full((nc, A), np.nan)
        arr[f"boot_{q}"] = np.full((nc, A, B), np.nan, np.float32)
        arr[f"hmask_mean_{q}"] = np.full((nc, A), np.nan)
        arr[f"sd_{q}"] = np.full((nc, A), np.nan)          # halo-to-halo standard deviation
    for k in ("nval", "n_hmask"):
        arr[k] = np.zeros((nc, A), int)
    for k in ("xbar", "x_p16", "x_p50", "x_p84", "sd_lnx", "med_ngas", "frac_valid", "mean_ngas"):
        arr[k] = np.full((nc, A), np.nan)
    arr["node_lnM"] = np.zeros((nc, A, NN)); arr["node_x"] = np.zeros((nc, A, NN)); arr["node_w"] = np.zeros((nc, A, NN))
    arr["comp_err_pub"] = np.full((nc, A, 2), np.nan); arr["comp_err_pert"] = np.full((nc, A, 2), np.nan)
    arr["sub_n"] = np.zeros(nc, int)
    arr["sub_lnM"] = np.full((nc, NS), np.nan); arr["sub_conc"] = np.full((nc, NS), np.nan)
    arr["sub_row"] = np.full((nc, NS), -1, np.int64)
    for k in ("x", "Yn", "Ya", "Mg"):
        arr[f"sub_{k}"] = np.full((nc, NS, A), np.nan, np.float32)
    arr["sub_ngas"] = np.full((nc, NS, A), -1, np.int32); arr["sub_valid"] = np.zeros((nc, NS, A), bool)
    for f in ("P", "G"):
        arr[f"shell_{f}"] = np.full((nc, A, 7), np.nan)
    summary = []
    pub_b12 = K.Law("B12", "orig"); pub_b16 = K.Law("B16", "orig")
    th12, th16 = pub_b12.start(), pub_b16.start()
    pert12, pert16 = th12.copy(), th16.copy()
    pert12[pub_b12.index["xc:lnA"]] += math.log(1.6); pert12[pub_b12.index["beta:lnA"]] += math.log(0.8)
    pert16[pub_b16.index["alpha:lnA"]] += math.log(1.6); pert16[pub_b16.index["beta:lnA"]] += math.log(0.85)
    for ci, b in enumerate(cells):
        idx = np.nonzero(bins == b)[0]
        n = len(idx)
        seed = rng_seed + 1000 * snap + b
        arr["cell_N"][ci] = n
        arr["cell_mean_lnM"][ci] = h["lnM"][idx].mean()
        arr["cell_median_lnM"][ci] = np.median(h["lnM"][idx])
        V = h["valid"][idx]
        X = h["x"][idx]
        counts = K.bootstrap_counts(n, B, seed)
        Vf = V.astype(float)
        nv = Vf.sum(0)
        arr["nval"][ci] = nv.astype(int)
        arr["frac_valid"][ci] = nv / n
        with np.errstate(invalid="ignore", divide="ignore"):
            for q in ("Yn", "Ya", "Mg"):
                Q = np.where(V & np.isfinite(h[q][idx]), h[q][idx], 0.0)
                Vq = V & np.isfinite(h[q][idx])
                nq = Vq.sum(0)
                arr[f"mean_{q}"][ci] = np.where(nq > 0, Q.sum(0) / np.maximum(nq, 1), np.nan)
                arr[f"boot_{q}"][ci] = ((counts @ Q) / (counts @ Vq.astype(float))).T.astype(np.float32)
                sd = np.array([np.std(h[q][idx][Vq[:, j], j], ddof=1) if nq[j] > 1 else np.nan for j in range(A)])
                arr[f"sd_{q}"][ci] = sd
                hm = Vq & (h["Ngas"][idx] >= KC.MIN_MEDIAN_GAS)
                arr[f"hmask_mean_{q}"][ci] = np.where(hm.sum(0) > 0, np.where(hm, h[q][idx], 0).sum(0) / np.maximum(hm.sum(0), 1), np.nan)
            arr["n_hmask"][ci] = (V & (h["Ngas"][idx] >= KC.MIN_MEDIAN_GAS)).sum(0)
        for j in range(A):
            ok = V[:, j]
            if not ok.any():
                continue
            xj = X[ok, j]
            lnMj = h["lnM"][idx][ok]
            arr["xbar"][ci, j] = xj.mean()
            arr["x_p16"][ci, j], arr["x_p50"][ci, j], arr["x_p84"][ci, j] = np.percentile(xj, [16, 50, 84])
            arr["sd_lnx"][ci, j] = np.std(np.log(xj)) if len(xj) > 1 else 0.0
            ng = h["Ngas"][idx][ok, j]
            arr["med_ngas"][ci, j] = np.median(ng)
            arr["mean_ngas"][ci, j] = ng.mean()
            groups = K.strata_groups(lnMj, np.log(xj), KC.N_STRATA_M, KC.N_STRATA_X)
            for g_i, g in enumerate(groups):
                arr["node_lnM"][ci, j, g_i] = lnMj[g].mean()
                arr["node_x"][ci, j, g_i] = np.exp(np.log(xj[g]).mean())
                arr["node_w"][ci, j, g_i] = len(g) / len(xj)
            # compression error of the 32-node quadrature at the published (and a perturbed) law: exact mean over
            # every valid halo vs the nodes
            lz = np.full(len(xj), math.log1p(z))
            gw = arr["node_w"][ci, j, :len(groups)]
            for kf, (form, field, law, ths) in enumerate((("B12", "P", pub_b12, (th12, pert12)), ("B16", "G", pub_b16, (th16, pert16)))):
                if field == "P" and j >= KC.N_SO:
                    continue
                for key, th in zip(("comp_err_pub", "comp_err_pert"), ths):
                    ph = law.phi(th, lnMj, lz)
                    exact = np.mean(K.FIELD_NORM[field] * K.cumulative(xj, form, ph["amp"], ph["xc"], ph["alpha"], ph["beta"], ph["gamma"]))
                    pn = law.phi(th, arr["node_lnM"][ci, j, :len(groups)], np.full(len(groups), math.log1p(z)))
                    approx = np.sum(gw * K.FIELD_NORM[field] * K.cumulative(arr["node_x"][ci, j, :len(groups)], form, pn["amp"], pn["xc"], pn["alpha"], pn["beta"], pn["gamma"]))
                    arr[key][ci, j, kf] = math.log(approx / exact)
        stats = {k: arr[k][ci] for k in ("xbar", "frac_valid", "med_ngas")}
        for f, qk in (("P", "Yn"), ("G", "Mg")):
            rows = display_shells(h, idx, stats, f, qk)
            for r_i, row in enumerate(rows[:A]):
                arr[f"shell_{f}"][ci, r_i] = row
        rng = np.random.default_rng(seed + 7)
        sub = np.sort(rng.choice(n, size=min(n, NS), replace=False))
        ns = len(sub)
        arr["sub_n"][ci] = ns
        arr["sub_lnM"][ci, :ns] = h["lnM"][idx][sub]
        arr["sub_conc"][ci, :ns] = h["conc"][idx][sub]
        arr["sub_row"][ci, :ns] = h["soap_row"][idx][sub]
        for k in ("x", "Yn", "Ya", "Mg"):
            arr[f"sub_{k}"][ci, :ns] = h[k][idx][sub]
        arr["sub_ngas"][ci, :ns] = h["Ngas"][idx][sub]
        arr["sub_valid"][ci, :ns] = h["valid"][idx][sub]
        summary.append({"label": KC.cell_label(snap, b), "bin": b, "N": int(n),
                        "usable_P": [KC.AP_NAMES[j] for j in np.nonzero(usable_apertures(stats, "P"))[0]],
                        "usable_G": [KC.AP_NAMES[j] for j in np.nonzero(usable_apertures(stats, "G"))[0]]})
    return arr, summary


def k1_file(snap, results, scratch, limit=None, keep_halo=True):
    """Worker: read + derive + reduce one output; returns a small status dict."""
    t0 = time.time()
    status = {"snap": snap, "ok": False}
    try:
        raw, meta = read_file(snap, limit)
        z = float(meta["Redshift"])
        h, chk = derive(raw, meta)
        del raw
        status.update(z=z, n_selected=meta["n_selected"], seconds_read=meta["seconds_read"], checks=chk)
        if keep_halo and scratch is not None:
            Path(scratch).mkdir(parents=True, exist_ok=True)
            np.savez(Path(scratch) / f"halo_{snap:04d}.npz", **{k: (v.astype(np.float32) if v.dtype == np.float64 else v)
                                                              for k, v in h.items()})
        t1 = time.time()
        arr, summary = reduce_cells(h, snap, z, KC.SEED)
        status["seconds_reduce"] = time.time() - t1
        (Path(results) / "cells").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(results) / "cells" / f"cells_{snap:04d}.npz", **arr)
        meta_out = {k: v for k, v in meta.items() if k not in ("column_seconds",)}
        meta_out["column_seconds_total"] = float(sum(meta["column_seconds"].values()))
        meta_out["checks"] = chk
        meta_out["cells"] = summary
        KC.write_json(Path(results) / "cells" / f"cells_{snap:04d}.json", meta_out)
        # z = 0: the study haloes (selected_halos.npz soap_row) for the particle parity check (K5)
        if snap == 77:
            sel_path = KC.ROOT / "selected_halos.npz"
            if sel_path.exists():
                rows = np.load(sel_path)["soap_row"].astype(np.int64)
                pos = np.searchsorted(h["soap_row"], rows)
                pos = np.clip(pos, 0, len(h["soap_row"]) - 1)
                found = h["soap_row"][pos] == rows
                keep = {k: v[pos] for k, v in h.items() if isinstance(v, np.ndarray) and len(v) == len(h["soap_row"])}
                keep["found"] = found
                keep["study_soap_row"] = rows
                np.savez_compressed(Path(results) / "z0_study_rows.npz", **keep)
                status["z0_study_rows_found"] = int(found.sum())
        status["ok"] = True
    except Exception:   # noqa: BLE001 - a failed file is reported, the others continue
        status["error"] = traceback.format_exc()[-3000:]
    status["seconds"] = time.time() - t0
    return status
