"""Exact mass windows of the Konietzka TNG300 halo-DM catalogue (2026-10-01).

Each per-ray array 'DMhalo_r200_massAtoB' holds the DM of halos with 10^A <= M < 10^(B+1) Msun, i.e. decades A..B inclusive:
DM(10to14) = DM(10to12) + DM(13to14) holds to < 1e-12 pc cm^-3 on every ray at every z_s (verify_v3/konietzka_windows), which is
impossible under the nominal reading [10^A, 10^B). The '13to14' array is therefore [1e13, 1e15) and the apparent 1e14-1e15 window
DM(10to15) - DM(10to14) is the 1e15-1e16 decade. DM adds over halos, so single decades follow ray by ray from differences."""
import numpy as np

# label -> (array to add, array to subtract or None, true mass range in Msun)
EXACT = {
    "m1e13_to_1e14": ("m1e10_to_1e13", "m1e10_to_1e12", (1e13, 1e14)),
    "m1e14_to_1e15": ("m1e10_to_1e14", "m1e10_to_1e13", (1e14, 1e15)),
    "m1e15_to_1e16": ("m1e10_to_1e15", "m1e10_to_1e14", (1e15, 1e16)),
    "m1e13_to_1e15": ("m1e13_to_1e14", None, (1e13, 1e15)),
    "m1e12_to_1e15": ("m1e12_to_1e14", None, (1e12, 1e15)),
}


def tng_exact(direct, label, zs):
    """Per-ray DM [pc cm^-3] of TNG halos in the true mass range named by `label` (HalfDome window labels, [lo, hi))."""
    if label == "m1e12_to_1e14":     # [1e12,1e15) minus the 1e14-1e15 decade
        v = np.asarray(direct.tng_values("m1e12_to_1e14", zs), float) - tng_exact(direct, "m1e14_to_1e15", zs)
        return np.where(v > 1e-9, v, 0.0)
    plus, minus, _ = EXACT[label]
    v = np.asarray(direct.tng_values(plus, zs), float)
    if minus is not None:
        v = v - np.asarray(direct.tng_values(minus, zs), float)
    return np.where(v > 1e-9, v, 0.0)  # differences of identical sums carry ~1e-13 round-off


def identity_residual(direct, zs):
    """max |DM(10to14) - DM(10to12) - DM(13to14)| over rays: ~1e-13 if 'AtoB' = [10^A, 10^(B+1))."""
    g = lambda k: np.asarray(direct.tng_values(k, zs), float)
    return float(np.max(np.abs(g("m1e10_to_1e14") - g("m1e10_to_1e12") - g("m1e13_to_1e14"))))
