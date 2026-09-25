"""Read-only environment discovery; run under each candidate interpreter."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

print(json.dumps(dict(
    executable=sys.executable,
    modules={name:bool(importlib.util.find_spec(name))
             for name in ['numpy','scipy','healpy','classy_sz','sbi','torch','h5py','mpmath']},
    julia=shutil.which('julia'),
    julia_candidates=[str(p) for base in [Path.home()/'.julia',Path.home()/'.juliaup']
                      if base.exists() for p in base.glob('**/bin/julia')],
),indent=2))
