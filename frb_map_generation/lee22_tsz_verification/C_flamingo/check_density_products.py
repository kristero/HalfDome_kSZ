"""Consistency of the idark density products with the preliminary pressure stacks and SOAP.

Checks per snapshot: same haloes; particle counts per shell identical to the stack products (n3);
summed ComptonY per shell (y3) against the stacks' y3_raw; electron-pressure-volume sum pe3 against y3;
gas mass inside R200c against SOAP SO/200_crit/GasMass (catalogue caches). Writes OUT/density/check_density_products.json.
"""
import json
import numpy as np
from scipy.constants import m_e, c, physical_constants
from common_c import OUT, STACKS, SNAPSHOTS, FPC

SIGMA_T = physical_constants["Thomson cross section"][0]
MPC_M = 3.085677581491367e22
res = {}
for s in SNAPSHOTS:
    d = np.load(OUT / "idark_density" / "products" / f"L2p8_m9_{s:04d}_density_products.npz")
    st = np.load(STACKS / f"L2p8_m9_{s:04d}_stack_products.npz")
    cat = np.load(FPC / "notebook_cache" / f"L2p8_m9_{s:04d}_reduced_haloes.npz")
    assert np.array_equal(d["halo_index"], st["cat_halo_index"])
    idx = {h: i for i, h in enumerate(cat["halo_index"])}
    rows = np.array([idx[h] for h in d["halo_index"]])
    gas_soap = cat["gas_mass_200c_msun"][rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        y_rel = np.where(st["y3_raw"] > 0, d["y3"] / st["y3_raw"] - 1, 0)
        pe_rel = np.where(d["y3"] > 0, d["pe3"] * SIGMA_T / (m_e * c**2) / MPC_M**2 / d["y3"] - 1, 0)
        gas_rel = d["m_r200"] / gas_soap - 1
    res[s] = {"z": float(d["redshift"]), "haloes": int(len(rows)),
              "n3_identical": bool(np.array_equal(d["n3"], st["n3"])),
              "y3_vs_stack_max_abs_rel": float(np.nanmax(np.abs(y_rel))),
              "pe3_vs_y3_max_abs_rel": float(np.nanmax(np.abs(pe_rel))),
              "gas_r200_vs_SOAP_median_rel": float(np.nanmedian(gas_rel)),
              "gas_r200_vs_SOAP_max_abs_rel": float(np.nanmax(np.abs(gas_rel))),
              "electron_to_gas_mp_over_M_median": float(np.median(d["ne_r200"] * 1.67262192369e-27 / (d["m_r200"] * 1.98841e30))),
              "sf_mass_fraction_r200_median": float(np.median(d["m_r200_sf"] / d["m_r200"]))}
    print(s, res[s])
(OUT / "density").mkdir(parents=True, exist_ok=True)
(OUT / "density" / "check_density_products.json").write_text(json.dumps(res, indent=1) + "\n")
