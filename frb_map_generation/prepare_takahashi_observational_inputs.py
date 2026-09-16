#!/usr/bin/env python3
"""Extract the published catalogue and record the observational input contract.

No model DMs are fabricated, no two FRBs are arbitrarily removed to force 131,
and no large survey maps are downloaded by this preparation step. The table
coordinates are rounded as published, not the original sub-arcsecond positions.
"""
import argparse
import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = "https://arxiv.org/src/2511.02155v2"
CLUSTER_FRBS = {"20220914A", "20231206A", "20231229A"}


def prepare(archive, output):
    with tarfile.open(str(archive), "r:*") as handle:
        # Read a single named member, never execute code or extract arbitrary
        # archive paths into the workspace.
        main_tex = handle.extractfile("main.tex").read().decode("utf-8")
    rows = []
    for line in main_tex.splitlines():
        if not re.match(r"^\s*20\d{6}[A-Z].*&", line):
            continue
        fields = [x.strip() for x in line.split("&")]
        if len(fields) != 7:
            continue
        name = re.match(r"(20\d{6}[A-Z])", fields[0]).group(1)
        ra, dec, dm, z = map(float, fields[1:5])
        flags = fields[5]
        if flags not in ("", "P", "A", "PA"):
            raise ValueError("Unknown survey flag: " + flags)
        rows.append(dict(frb=name, ra_deg=ra, dec_deg=dec, dm_observed_pc_cm3=dm,
                         redshift=z, table_planck_footprint=int("P" in flags),
                         table_act_footprint=int("A" in flags),
                         cluster_host_excluded_from_cross=int(name in CLUSTER_FRBS),
                         reference_key=fields[6].split("}")[0].split("{")[-1]))
    if len(rows) != 133 or len({row["frb"] for row in rows}) != 133:
        raise ValueError("Expected exactly 133 distinct published FRB entries")
    if {r["frb"] for r in rows if r["cluster_host_excluded_from_cross"]} != CLUSTER_FRBS:
        raise ValueError("Cluster exclusions do not match the paper")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output/"takahashi2025_v2_table8_133_frbs.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {"published_table": len(rows)}
    for flag in ("planck", "act"):
        counts[flag+"_table_footprint"] = sum(r["table_"+flag+"_footprint"] for r in rows)
        counts[flag+"_after_cluster_exclusion"] = sum(
            r["table_"+flag+"_footprint"] and not r["cluster_host_excluded_from_cross"] for r in rows)
    inventory = {
        "scope": "Halo-only partial prediction, explicitly requested by user; not total extragalactic DM",
        "tsz_pressure": "Keep Battaglia12 as the main prediction; published constant-temperature curves are references only",
        "observational_vectors": "User authorized approximate digitization from Figures 13-15 and 5; see digitized/ and digitization_provenance.json; full covariance remains unavailable",
        "catalogue_source": SOURCE_URL,
        "catalogue_source_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "catalogue_counts": counts,
        "coordinates": "Published Table 8 rounded RA/Dec; exact author positions may be required at mask edges",
        "fit_sample_131": "Not manufactured here: apply NE2001+YT20, exclude the two negative extragalactic DMs",
        "target_figures": {"2511.02155v2": [13, 14, 15], "2608.06455v1": [5]},
        "selection": {
            "source_redshifts": "Use each observed redshift, not a common z=1; catalogue extends to z=2.148",
            "source_positions": "User chose uniform random sightlines within survey selection, carrying the observed source redshifts; no simulated-host halo weighting or host-environment correlation",
            "frb_cluster_exclusions": sorted(CLUSTER_FRBS),
            "planck_expected_cross_count": 71, "act_expected_cross_count": 31,
            "figure14_galactic_mask_percent": [40, 50, 60, 70],
            "figure14_expected_counts": [71, 66, 54, 42],
            "dm_residual": "Subtract the halo-only ensemble mean at each source redshift; do not substitute the observed total-DM mean",
        },
        "estimator": {
            "quantity": "configuration-space w_yDM(theta), not C_ell or D_ell",
            "theta_arcmin_range": [1, 1000], "delta_log10_theta": 0.25,
            "y_monopole": "subtract unmasked survey mean",
            "weight": "[(DM_MW/2)^2 + (sigma_host0/(1+z)^beta_host)^2]^-1 * sigma_y(pixel)^-2",
            "default_Nbeta_MAP": {"fe": .972, "DM_host0": 120.2, "sigma_host0": 98.9, "beta_host": -.199},
            "random_subtraction": "3000 position randomizations inside survey; retain source DM marks",
            "covariance": "leave-one-FRB-out jackknife; compare 3000 bootstrap samples",
            "noise_warning": "Halo-only mocks omit host/IGM/MW scatter; their covariance is not the observed covariance",
        },
        "public_resources": {
            "Planck_PR2": {
                "url": "https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/ysz_index.html",
                "archive_url": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_Compton-SZMap_R2.02.tgz",
                "needed": ["milca_ymaps.fits", "nilc_ymaps.fits", "milca_stddev.fits", "nilc_stddev.fits", "milca_homnoise_spect.fits", "nilc_homnoise_spect.fits", "masks.fits"],
                "nside": 2048, "coordinates": "Galactic", "fwhm_arcmin": 10,
                "status": "public location verified; maps not yet downloaded",
                "mocks": "Match inhomogeneous noise using delivered standard-deviation maps and homogeneous noise spectra; validate against half-ring differences",
            },
            "ACT_DR6": {
                "url": "https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_compton_maps_get.html",
                "simulations_url": "https://portal.nersc.gov/project/act/dr6_nilc/",
                "needed": ["ilc_actplanck_ymap.fits", "wide_mask_GAL070_apod_1.50_deg_wExtended.fits", "ilc_beam.txt"],
                "coordinates": "equatorial CAR in archive; paper reprojects to HEALPix NSIDE8192",
                "paper_fwhm_arcmin": 1.6, "estimator_sigma_y": 1,
                "status": "public map, mask, beam and simulation locations verified; not downloaded",
            },
            "Planck_PR4_McCarthy_Hill": {
                "url": "https://users.flatironinstitute.org/~fmccarthy/ymaps_PR4_McCH23/",
                "needed": "NILC maps without and with CIB deprojection; beta_CIB=1.7, T_CIB=10.71 K",
                "estimator_sigma_y": 1,
                "archive_redirect": "https://zenodo.org/records/18405044",
                "status": "paper-provided URL redirects to Zenodo; record fetch timed out, exact file names unverified",
            },
            "Sharma_CHIME_Figure5_panel": {
                "paper": "https://arxiv.org/abs/2604.22105",
                "parent_sources": 3455, "localization_arcmin_approx": 15,
                "dm_cuts_pc_cm3": [500, 750, 1000],
                "needed": "Selection kernels for each observed-DM cut; CHIME Catalog 2 selection; full covariance if a future formal fit is requested",
                "warning": "This is not the 71/131 localized sample. Do not apply observed-DM cuts to simulated halo-only DMs.",
            },
        },
        "author_inputs_needed": [
            "Takahashi binned measurements and full covariance for Planck/ACT and Figures 13-15 (available on request)",
            "Exact source positions, residual DMs and MW predictions, masks/reprojection settings to reproduce their weights",
            "Medlock/Nagai Figure 5 maximum-likelihood BP curves and Sharma data/covariance",
        ],
        "important_reference_difference": "Medlock/Nagai assume all model sources at z=2; an observation-matched source kernel is a separate prediction",
        "not_yet_implemented": ["survey map ingestion", "observational mock generation", "cross-correlation measurement", "fit or goodness-of-fit calculation"],
    }
    (output/"observational_input_inventory.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps(counts, indent=2))
    print("Saved", csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.source_archive
    if archive is None:
        archive = args.output_dir/"takahashi2025_v2_source.tar"
        if not archive.exists():
            with urlopen(SOURCE_URL, timeout=45) as response:
                archive.write_bytes(response.read())
    prepare(archive, args.output_dir)
