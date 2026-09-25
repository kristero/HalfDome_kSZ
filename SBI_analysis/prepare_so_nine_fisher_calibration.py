#!/usr/bin/env python3
"""Generate matching Battaglia12 derivatives, noise covariance, and observation.

Use the EXACT independent-noise simulator bundle that produced the 32k data.
Run `init`, PBS `work` workers, then `combine`. No production job is launched
by this Python entry point. Completed rows are checksummed and resumable.
"""
import argparse
import csv
import importlib.util
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from export_so_moped_bundle import sha256
from generate_so_fisher_variations import BATTAGLIA12, PARAMETER_NAMES
from so_nine_fisher import load_independent_dataset, require, read_npz, simulation_signature
from so_sbi_compression import write_json, save_npz


def load_generator(bundle):
    spec=importlib.util.spec_from_file_location("independent_so_generator",bundle/"generate.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibration_design(low,high,n_noise):
    require(n_noise>=64,"Use at least 64 independent covariance realizations")
    theta=[BATTAGLIA12.copy()]
    rows=[dict(kind="fiducial",parameter=-1,fraction=0.,sign=0)]
    for j in range(9):
        for fraction in (.01,.02):
            for sign in (-1,1):
                t=BATTAGLIA12.copy()
                t[j]+=sign*fraction*(high-low)[j]
                require(np.all((t>low)&(t<high)),"Derivative step leaves prior")
                theta.append(t)
                rows.append(dict(kind="derivative",parameter=j,fraction=fraction,sign=sign))
    for _ in range(n_noise):
        theta.append(BATTAGLIA12.copy())
        rows.append(dict(kind="covariance",parameter=-1,fraction=0.,sign=0))
    theta.append(BATTAGLIA12.copy())
    rows.append(dict(kind="observation",parameter=-1,fraction=0.,sign=0))
    return np.asarray(theta),rows


def initialize(args):
    data,generation=load_independent_dataset(args.dataset,args.generation_manifest)
    gen=load_generator(args.bundle)
    config=json.loads((args.bundle/"config.json").read_text())
    gen.validate_config(config)
    gen.validate_noise_tables(config)
    actual=gen.provenance(config,gen.load_designs(config),args.catalogue)
    require(simulation_signature(actual)==simulation_signature(generation),
            "Calibration simulator/source/catalogue size differs from the 32k generation")
    # Both generation namespaces occupy fewer than 2**49 integer seeds.
    # Reserve a new distant namespace for ALL calibration rows/splits.
    config=dict(config,sequence_offset=0,
        noise_seed_bases=dict(config["noise_seed_bases"],nine_param=2**50+1000000))
    theta,rows=calibration_design(data["prior_low"],data["prior_high"],args.noise_realizations)
    splits=np.array([gen.seeds(config,"nine_param",i+1)[1:] for i in range(len(theta))])
    require(not np.intersect1d(splits,data["noise_split_seeds"]).size,"Calibration noise overlaps training")
    require(len(np.unique(splits))==splits.size,"Repeated calibration split seeds")
    require(not args.root.exists(),"Choose a new calibration root")
    args.root.mkdir(parents=True)
    with (args.root/"parameters.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(PARAMETER_NAMES)
        writer.writerows(theta)
    save_npz(args.root/"design.npz",theta=theta,split_seeds=splits,
        prior_low=data["prior_low"],prior_high=data["prior_high"],
        **{k:data[k] for k in ("ell_unbinned","ell_binned","bin_ell_min","bin_ell_max")})
    manifest=dict(config=config,rows=rows,bundle=str(args.bundle.resolve()),
        catalogue=str(args.catalogue.resolve()),source_signature=simulation_signature(generation),
        dataset=str(args.dataset.resolve()),dataset_sha256=sha256(args.dataset),
        generation_manifest_sha256=sha256(args.generation_manifest),source_sha256=generation["source_sha256"],
        design_sha256=sha256(args.root/"design.npz"),csv_sha256=sha256(args.root/"parameters.csv"),
        n_noise=args.noise_realizations,calibration_code_sha256=sha256(Path(__file__)))
    write_json(args.root/"manifest.json",manifest)
    print(f"Prepared {len(theta)} maps: 37 derivatives/fiducial, {args.noise_realizations} noise, one held-out observation")


def checked_setup(root):
    manifest=json.loads((root/"manifest.json").read_text())
    require(sha256(Path(__file__))==manifest["calibration_code_sha256"],"Calibration code changed; use a fresh root")
    require(sha256(root/"design.npz")==manifest["design_sha256"] and
            sha256(root/"parameters.csv")==manifest["csv_sha256"],"Changed calibration design")
    bundle=Path(manifest["bundle"])
    for relative,digest in manifest["source_sha256"].items():
        require(sha256(bundle/relative)==digest,f"Changed simulator source: {relative}")
    return manifest,read_npz(root/"design.npz"),load_generator(bundle)


def valid_row(folder,theta,seeds,identity):
    marker=folder/"complete.json"
    if not marker.exists():
        return False
    done=json.loads(marker.read_text())
    require(done["manifest_sha256"]==identity,"Stale calibration row")
    np.testing.assert_allclose(done["theta"],theta,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(done["split_seeds"],seeds)
    require(sha256(folder/"spectra.npz")==done["spectra_sha256"],"Corrupt calibration spectra")
    return True


def work(args):
    import fcntl
    manifest,design,gen=checked_setup(args.root)
    require(0<=args.worker<args.workers,"Invalid worker assignment")
    identity=sha256(args.root/"manifest.json")
    deadline=time.monotonic()+args.work_seconds
    for i in range(args.worker,len(design["theta"]),args.workers):
        folder=args.root/"rows"/f"row{i+1:04d}"
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/".lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if valid_row(folder,design["theta"][i],design["split_seeds"][i],identity):
                continue
            if time.monotonic()+args.row_timeout>deadline:
                print("Walltime margin reached; rerun this worker to resume",flush=True)
                return
            with tempfile.TemporaryDirectory(prefix="cache_",dir=folder) as cache:
                command=gen.command(manifest["config"],"nine_param",i+1,args.root/"parameters.csv",
                    Path(manifest["catalogue"]),folder,Path(cache),args.julia,args.threads)
                write_json(folder/"request.json",dict(command=command,kind=manifest["rows"][i]))
                print(f"Generating calibration row {i+1}/{len(design['theta'])}",flush=True)
                with (folder/"simulation.log").open("w") as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                        timeout=args.row_timeout,env=gen.simulator_environment(args.threads))
            log=(folder/"simulation.log").read_text()
            require("Painter: ring_locked" in log,"Wrong painter")
            reported=re.findall(r"^Actual theta:\s*(\[.*\])\s*$",log,re.MULTILINE)
            require(len(reported)==1,"Missing actual theta")
            np.testing.assert_allclose(json.loads(reported[0]),design["theta"][i],rtol=5e-14,atol=1e-15)
            reported=re.findall(r"^Actual split seeds baseline_deproj0:\s*(\d+),(\d+)\s*$",log,re.MULTILINE)
            np.testing.assert_array_equal(np.asarray(reported,dtype=np.int64),design["split_seeds"][i:i+1])
            require(re.findall(r"^Actual beam FWHM:\s*(\S+)\s*$",log,re.MULTILINE)==["2.0"],"Wrong beam")
            arrays={}
            for key,pattern in (("clean","*masked_no_noise_cl*_deproj0_*.npy"),
                                ("noisy","*masked_baseline_noise_cross_cl*_deproj0_*.npy")):
                paths=list(folder.glob(pattern))
                require(len(paths)==1,f"Missing/ambiguous {key} calibration spectrum")
                values=np.load(paths[0],allow_pickle=False)
                require(values.shape==(7980,) and np.isfinite(values).all(),"Invalid multipoles")
                arrays[key]=gen.bin_dell(values,manifest["config"])
            save_npz(folder/"spectra.npz",**arrays)
            write_json(folder/"complete.json",dict(manifest_sha256=identity,theta=design["theta"][i].tolist(),
                split_seeds=design["split_seeds"][i].tolist(),spectra_sha256=sha256(folder/"spectra.npz")))


def combine(args):
    manifest,design,_=checked_setup(args.root)
    identity=sha256(args.root/"manifest.json")
    values=[]
    for i,theta in enumerate(design["theta"]):
        folder=args.root/"rows"/f"row{i+1:04d}"
        require(valid_row(folder,theta,design["split_seeds"][i],identity),f"Missing calibration row {i+1}")
        values.append(read_npz(folder/"spectra.npz"))
    clean=np.array([v["clean"] for v in values])
    noisy=np.array([v["noisy"] for v in values])
    rows=manifest["rows"]
    width=design["prior_high"]-design["prior_low"]
    derivatives={}
    for fraction,name in ((.01,"small"),(.02,"large")):
        columns=[]
        for j in range(9):
            find=lambda sign:next(i for i,r in enumerate(rows) if r["kind"]=="derivative" and
                r["parameter"]==j and r["fraction"]==fraction and r["sign"]==sign)
            columns.append((clean[find(1)]-clean[find(-1)])/(2*fraction*width[j]))
        derivatives[f"derivative_{name}"]=np.asarray(columns).T
    derivatives["derivatives"]=(4*derivatives["derivative_small"]-derivatives["derivative_large"])/3
    selected=np.array([i for i,r in enumerate(rows) if r["kind"]=="covariance"])
    observation=next(i for i,r in enumerate(rows) if r["kind"]=="observation")
    # Repeated clean maps should be unchanged by noise seeds and thread ordering.
    relative=np.max(np.abs(clean[np.r_[selected,observation]]/clean[0]-1))
    require(relative<1e-5,f"Noise-only ensemble changes the clean signal: {relative}")
    destination=args.root/"calibration.npz"
    save_npz(destination,**derivatives,fiducial=BATTAGLIA12,param_names=np.array(PARAMETER_NAMES),
        fiducial_dell=clean[0],noise_ensemble=noisy[selected],observation=noisy[observation],
        covariance_split_seeds=design["split_seeds"][selected],observation_split_seeds=design["split_seeds"][observation],
        source_signature=np.asarray(manifest["source_signature"]),dataset_sha256=np.asarray(manifest["dataset_sha256"]),
        **{k:design[k] for k in ("prior_low","prior_high","ell_unbinned","ell_binned","bin_ell_min","bin_ell_max")})
    write_json(args.root/"calibration_complete.json",dict(complete=True,calibration_sha256=sha256(destination),
        manifest_sha256=identity,n_noise=len(selected),observation_excluded_from_covariance=True,
        clean_repeat_max_relative_difference=float(relative)))
    print(f"Calibration complete: {destination}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("init","work","combine"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--generation-manifest",type=Path)
    p.add_argument("--bundle",type=Path)
    p.add_argument("--catalogue",type=Path)
    p.add_argument("--noise-realizations",type=int,default=256)
    p.add_argument("--worker",type=int,default=0)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--threads",type=int,default=26)
    p.add_argument("--julia",default="julia")
    p.add_argument("--work-seconds",type=float,default=81000)
    p.add_argument("--row-timeout",type=float,default=2400)
    args=p.parse_args()
    args.root=args.root.resolve()
    if args.stage=="init":
        require(all(getattr(args,k) is not None for k in ("dataset","generation_manifest","bundle","catalogue")),
                "init needs --dataset, --generation-manifest, --bundle and --catalogue")
        initialize(args)
    elif args.stage=="work":
        work(args)
    else:
        combine(args)


if __name__=="__main__":
    main()
