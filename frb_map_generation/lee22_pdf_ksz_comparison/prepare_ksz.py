#!/usr/bin/env python3
"""Write the audited 3R200c configuration and bounds without loading the catalogue."""
import json
import sys
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tau_ksz_comparison"))
from reproduce_tau_ksz import write_toml

OUT=ROOT/"frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922"
out=OUT/"ksz";out.mkdir(parents=True,exist_ok=True)
old=ROOT/"tau_ksz_comparison/outputs/nside4096"
bounds=tomllib.loads((old/"catalog_bounds.toml").read_text())
# Retain the FRB comparison's h, Omega_m and Omega_b. Catalogue masses are Msun/h.
old_h=.6774;h=.68
for key in ("mass_min_msun","mass_max_msun"):bounds[key]*=old_h/h
cfg=dict(catalog="/home/cbllover/HalfDome/lightcone_100.hdf5",
         output="/home/cbllover/HalfDome/frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922/ksz",
         nside=4096,lmax=6000,anafast_iterations=3,chunk_size=250000,sphere_r200c=3.,threads=20,
         cosmology=dict(h=h,Omega_b=.049,Omega_c=.31-.049,sigma8=.8159,n_s=.9667),
         interpolation=dict(n_radius=256,n_redshift=96,n_mass=192,radius_fraction_min=1e-7,validation_rtol=.01))
write_toml(out/"settings.toml",cfg);write_toml(out/"bounds.toml",bounds)
report=dict(peak_memory_estimate_gib=[10,16],map_storage_gib=4*12*4096**2*8/1024**3,
            local_run=True,cluster_run=False,prior_catalog_bounds_source=str(old/"catalog_bounds.toml"),
            h_conversion="Original bounds multiplied by 0.6774/0.68; physical masses use catalogue mass / 0.68.",
            full_halo_count=bounds["total_catalog_rows"],full_bounds=bounds,
            restricted=dict(mass_min_msun=1e13,mass_max_msun=1e14,z_max=1,sphere_r200c=3),
            extrapolation="Lee22: low masses, high masses with previous shape freeze, z>2, and r>1.34R200c.",
            scientific_scope="CLASS-SZ is the B16 comparison; Lee22 variants are different gas models.")
(OUT/"resource_preflight.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
