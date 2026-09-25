"""Save the HEALPix temperature windows used only in rendering experiments."""
import hashlib
import json
from pathlib import Path
import healpy as hp
import numpy as np

root = Path(__file__).resolve().parent
records = []
for nside in (4096,8192):
    path = root/'inputs'/f'pixwin{nside}.txt'
    window = hp.pixwin(nside, lmax=3*4096-1)
    assert window.shape == (3*4096,) and np.isfinite(window).all() and np.all(window>0)
    np.savetxt(path,window,fmt='%.18e')
    records.append(dict(nside=nside, healpy=hp.__version__,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source='https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm'))
(root/'inputs/pixel_windows.json').write_text(json.dumps(records,indent=2)+'\n')
