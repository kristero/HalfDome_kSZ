#!/usr/bin/env python3
"""Isabel Medlock's Baryon-Pasting (BP) halo-model tSZ x FRB-DM spectra, prepared for the
Takahashi+25 w_yDM(theta) figures.

Input (Isabel_Medlock_data/, from I. Medlock): Dell_yDM_param_variations.csv holds
D_ell^{y x DM} = ell(ell+1) C_ell / 2pi [pc cm^-3] on 1000 log-spaced ell from 10 to 10^4, for the
BP fiducial gas parameters and one-at-a-time variations "<param>__<mult>" (param = mult x fiducial)
of epsilon, fstar, Sstar, A_nt, B_nt and gamma_nt; the "__1" columns are the (identical) fiducial.
The meta file gives the fiducial parameters, cosmology, a halo mass grid (1e13 - 10^15.5 Msun) and
a redshift grid 0.01 - 2; it does not state the FRB source kernel or a beam. The redshift grid
ending at z = 2 is consistent with Medlock & Nagai's all-sources-at-z=2 kernel, and no beam is
assumed to be applied (theory spectrum), so the survey beam is applied here on the same footing
as the HalfDome curves.

C_ell on integer ell: log-log interpolation inside ell = 10 - 10^4; below ell = 10 a power-law
continuation of D_ell fitted over ell = 10 - 20 (these multipoles add a near-constant ~8e-8 pc cm^-3, under 5 %
of the fiducial at theta <= 1 deg; beyond ~3 deg they are most of the tiny (< 1e-7) fiducial w, far below the data
errors; setting them to zero moves any Takahashi annulus by < 0.05 sigma); zero above ell = 10^4 (the supplied
range; the file is unbeamed: D_ell peaks at ell ~ 5000 with no beam roll-off). The monopole is dropped by the
Legendre sums themselves (compare_halfdome_takahashi.angular_correlation / annular_correlation).
"""
import json
from pathlib import Path

import numpy as np

MEDLOCK_DIR = Path(__file__).resolve().parent.parent / "Isabel_Medlock_data"
CSV_NAME = "Dell_yDM_param_variations.csv"
META_NAME = "Dell_yDM_param_variations_meta.json"
FIDUCIAL = "epsilon__1"
LMAX = 10000
LOW_ELL_FIT = (10.0, 20.0)
COLOR = "#9E4A8A"
LABEL = "Medlock BP halo model, fiducial (FRBs at z = 2)"
BAND_LABEL = "Medlock BP, range of one-at-a-time variations"


def load(directory=MEDLOCK_DIR):
    """(ell, {curve name: D_ell}, meta). Validates the file contract."""
    path = Path(directory) / CSV_NAME
    with path.open() as fh:
        header = fh.readline().strip().split(",")
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    if header[0] != "ell" or table.shape[1] != len(header) or len(set(header)) != len(header):
        raise ValueError("Unexpected Medlock CSV layout")
    if not all(len(n.split("__")) == 2 for n in header[1:]):
        raise ValueError("Medlock curve names must be <param>__<multiplier>")
    ell = table[:, 0]
    if not (np.all(np.diff(ell) > 0) and np.isclose(ell[0], 10) and np.isclose(ell[-1], 1e4)):
        raise ValueError("Medlock ell grid is not the expected increasing 10 - 1e4 grid")
    curves = {name: table[:, i] for i, name in enumerate(header) if i > 0}
    fiducials = [n for n in curves if n.split("__")[1] == "1"]
    if FIDUCIAL not in fiducials or not all(np.array_equal(curves[n], curves[FIDUCIAL]) for n in fiducials):
        raise ValueError("The '__1' columns are not one common fiducial")
    if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in curves.values()):
        raise ValueError("Non-finite or non-positive D_ell in the Medlock file")
    meta = json.loads((Path(directory) / META_NAME).read_text())
    if meta.get("units") != "pc cm^-3":
        raise ValueError("Unexpected Medlock units: " + str(meta.get("units")))
    return ell, curves, meta


def variations(curves):
    """Distinct curves: the fiducial once, then every non-fiducial variation."""
    return [FIDUCIAL] + [n for n in curves if n.split("__")[1] != "1"]


def cl_on_integers(ell_in, dell, lmax=LMAX):
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.zeros(lmax + 1)
    inside = (ell >= ell_in[0]) & (ell <= ell_in[-1])
    dl = np.exp(np.interp(np.log(ell[inside]), np.log(ell_in), np.log(dell)))
    cl[inside] = 2 * np.pi * dl / (ell[inside] * (ell[inside] + 1))
    fit = (ell_in >= LOW_ELL_FIT[0]) & (ell_in <= LOW_ELL_FIT[1])
    slope, intercept = np.polyfit(np.log(ell_in[fit]), np.log(dell[fit]), 1)
    below = (ell >= 1) & (ell < ell_in[0])
    cl[below] = 2 * np.pi * np.exp(intercept + slope * np.log(ell[below])) / (ell[below] * (ell[below] + 1))
    return cl


def all_cl(directory=MEDLOCK_DIR, lmax=LMAX):
    """{curve name: C_ell} for the fiducial and every distinct variation, plus the meta."""
    ell_in, curves, meta = load(directory)
    return {name: cl_on_integers(ell_in, curves[name], lmax) for name in variations(curves)}, meta


def self_test():
    ell_in, curves, _ = load()
    cl = cl_on_integers(ell_in, curves[FIDUCIAL])
    ell = np.arange(cl.size, dtype=float)
    # the interpolated C_ell reproduces the supplied D_ell at the supplied (integer and non-integer) nodes
    nodes = np.unique(np.round(ell_in[(ell_in >= 11) & (ell_in <= 9990)]).astype(int))
    back = ell[nodes] * (ell[nodes] + 1) * cl[nodes] / (2 * np.pi)
    ref = np.exp(np.interp(np.log(nodes), np.log(ell_in), np.log(curves[FIDUCIAL])))
    np.testing.assert_allclose(back, ref, rtol=1e-12)
    # the low-ell continuation is continuous at ell = 10 to within the ell 10-20 fit residual
    d10 = 10 * 11 * cl[10] / (2 * np.pi)
    d9 = 9 * 10 * cl[9] / (2 * np.pi)
    assert 0.8 < d9 / d10 < 1.0, (d9, d10)
    print("medlock_bp_spectra self-test passed; curves:", len(variations(curves)))


if __name__ == "__main__":
    self_test()
