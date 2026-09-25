"""Create the isolated validation source snapshot; never submit jobs."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'tsz_spherical_preflight_20260920'
for name in ['inputs','results','plots']:(ROOT/name).mkdir(exist_ok=True)
for name in ['fullsky_test.jl','spherical_truncation_profiles.jl','unbinned_moped.py',
             'design_tests.py','projection.py']:
    destination=ROOT/name
    if not destination.exists():shutil.copy2(PREVIOUS/name,destination)
for name in ['cases.json','columns.csv']:
    shutil.copy2(PREVIOUS/'inputs'/name,ROOT/'inputs'/name)
shutil.copy2(PREVIOUS/'spherical_truncation_profiles.jl',ROOT/'inputs/original_spherical_wrapper.jl')
print(ROOT)
