"""Independent fresh SWIFT mask for metadata budgeting cross-checks."""
import json
from pathlib import Path
import time
import numpy as np
import swiftsimio as sw
import unyt

HERE=Path(__file__).resolve().parent


def fresh_mask(path, centre, radius_proper, scale_factor):
    mask=sw.mask(path,safe_padding=.1)
    radius=radius_proper/scale_factor
    region=sw.cosmo_array(np.column_stack((centre-radius,centre+radius)),
        unyt.Mpc,comoving=True,scale_factor=scale_factor,scale_exponent=1)
    mask.constrain_spatial(region)
    return mask

