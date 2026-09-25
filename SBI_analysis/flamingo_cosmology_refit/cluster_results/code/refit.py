"""Fit the approximate cosmology response, checking candidates with full HD maps.

Only the existing HalfDome part is validated by repainting. The cosmology ratio
remains a halo-model approximation, including smooth-mask and sample-variance
assumptions. All intermediate results and the original candidates are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from forward_proposals import CatalogueForward
from halo_response import CosmologyResponse
from prior import JointPrior, digest
from prior_model import FIDUCIAL, comparison_metrics, save_json
from worker import run_row, bin_cl

CAMPAIGN=Path("/lustre/work/kristero10/flamingo_tsz_comparison_20260914")
PILOT=Path("/lustre/work/kristero10/flamingo_prior_pilot_20260914")
VARIANTS=("L1_m9","fgas-8sigma","Mstar-1sigma")


def fit_candidate(model,response,prior,target,theta,correction,name,root):
    saved=root/"proposals"/(name+".json")
    if saved.exists():
        return np.array(json.loads(saved.read_text())["theta"])
    anchor=prior.to_unit(FIDUCIAL)
    start=prior.to_unit(theta)
    lower=np.maximum(1e-7,start-.08)
    upper=np.minimum(1-1e-7,start+.08)
    evaluations=[0]
    def residual(unit):
        values=prior.from_unit(unit)
        if not prior.contains(values): return np.full(49,1e3)
        predicted=model.raw(values)*correction*response(values)
        if not np.isfinite(predicted).all() or np.any(predicted<=0):return np.full(49,1e4)
        return np.r_[np.log(predicted/target),.005*(unit-anchor)]
    def jacobian(unit):
        central=residual(unit)
        steps=np.full(9,2e-5)
        steps=np.where(unit+steps>=upper,-steps,steps)
        trial=np.tile(unit,(9,1));trial[np.arange(9),np.arange(9)]+=steps
        with ThreadPoolExecutor(max_workers=3) as pool:
            shifted=np.array(list(pool.map(residual,trial)))
        evaluations[0]+=1
        print(name,"Jacobian",evaluations[0],"log RMS",np.sqrt(np.mean(central[:40]**2)),flush=True)
        return ((shifted-central)/steps[:,None]).T
    start_time=time.monotonic()
    result=least_squares(residual,start,jac=jacobian,bounds=(lower,upper),
                         max_nfev=30,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    values=prior.from_unit(result.x)
    assert prior.contains(values)
    record=dict(theta=values.tolist(),success=bool(result.success),message=result.message,
                nfev=result.nfev,seconds=time.monotonic()-start_time,
                regularizer_residual_coefficient=.005,anchor="Battaglia12",
                local_unit_cube_radius=.08,
                proposal_metrics=comparison_metrics(model.raw(values)*correction*response(values),target))
    save_json(root/"proposals"/(name+".json"),record)
    return values


def main(root,variant,iterations):
    gate=json.loads((root/"audit/response_check.json").read_text())
    assert gate["passed"]
    manifest=json.loads((root/"manifest.json").read_text())
    for folder,key in (("code","code_sha256"),("audit","input_sha256")):
        for filename,sha in manifest[key].items():assert digest(root/folder/filename)==sha
    prior=JointPrior(json.loads((root/"code/prior.json").read_text()))
    previous=json.loads((root/"audit/pilot_results.json").read_text())["fits"][variant]["best"]
    original=np.array(previous["theta"])
    old_clean=bin_cl(np.load(PILOT/"results"/previous["name"]/"masked_clean_cl.npy"))
    target=bin_cl(np.load(CAMPAIGN/"results"/variant/"masked_clean_cl.npy"))
    response=CosmologyResponse(root)
    model=CatalogueForward(root,CAMPAIGN)
    correction=old_clean/model.raw(original)
    ratio=response(original)
    scale=float(np.exp(.5*np.mean(np.log(target/(old_clean*ratio)))))
    theta=original.copy();theta[0]*=scale
    assert prior.contains(theta)
    exact_amplitude_clean=old_clean*scale**2
    corrected=exact_amplitude_clean*response(theta)
    best=dict(name="amplitude_from_existing_map",theta=theta.tolist(),
        **comparison_metrics(corrected,target),basis="Exact P0 scaling of old full HD map times halo-model response")
    records=[best]
    np.savez(root/"results"/(variant+"_amplitude.npz"),theta=theta,
             hd_clean=exact_amplitude_clean,response=response(theta),corrected_clean=corrected,target=target)
    print(variant,"amplitude seed",best,flush=True)
    run=json.loads((root/"code/run_config.json").read_text())
    for iteration in range(iterations):
        name=variant+"_iter"+str(iteration)
        theta=fit_candidate(model,response,prior,target,theta,correction,name,root)
        # Separate preflight-like row IDs prevent overlap with production noise.
        label="preflight_"+str(100+10*VARIANTS.index(variant)+iteration)
        saved_cl=root/"results"/(name+"_full_cl.npz")
        saved_status=root/"results"/(name+"_status.json")
        if saved_cl.exists() and saved_status.exists():
            values=dict(np.load(saved_cl))
            status=json.loads(saved_status.read_text())
            np.testing.assert_array_equal(status["theta"],theta)
        else:
            values,status,log=run_row(root,run,theta,label)
            np.savez(saved_cl,**values)
            (root/"logs"/(name+"_painting.log")).write_bytes(log)
            save_json(saved_status,status)
        clean=bin_cl(values["masked_clean_cl"])
        corrected=clean*response(theta)
        record=dict(name=name,theta=theta.tolist(),**comparison_metrics(corrected,target),
                    basis="Fresh full HD map times halo-model response",painting_seconds=status["elapsed_seconds"])
        np.savez(root/"results"/(name+".npz"),theta=theta,hd_clean=clean,
                 response=response(theta),corrected_clean=corrected,target=target)
        records.append(record)
        if record["rms_fractional"]<best["rms_fractional"]:best=record
        save_json(root/"results"/(variant+"_progress.json"),dict(best=best,candidates=records))
        print("FULL-MAP-BASED",variant,record,flush=True)
        correction=clean/model.raw(theta)
    del model,response
    gc.collect()
    fine=CosmologyResponse(root,nmass=48,nz=36,nradial=256)
    # Re-evaluate every retained candidate using the finer response; choose
    # by measured, corrected map residuals, never by optimizer convergence flag.
    for record in records:
        suffix="_amplitude" if record["name"]=="amplitude_from_existing_map" else "_iter"+record["name"].rsplit("iter",1)[1]
        path=root/"results"/(variant+suffix+".npz")
        data=dict(np.load(path));new_ratio=fine(data["theta"])
        error=float(np.max(abs(data["response"]/new_ratio-1)))
        assert error<.005,(variant,error)
        data["coarse_response"]=data["response"]
        data["response"]=new_ratio;data["corrected_clean"]=data["hd_clean"]*new_ratio
        np.savez(path,**data)
        record.update(comparison_metrics(data["corrected_clean"],target))
        record["response_resolution_error"]=error
    best=min(records,key=lambda x:x["rms_fractional"])
    save_json(root/"results"/(variant+"_fit.json"),dict(variant=variant,best=best,candidates=records,
        original=previous,scope="Approximate cosmology-corrected full-map spectral fit; not a posterior or direct halo pressure fit",
        production_run_modified=False,manifest_sha256=digest(root/"manifest.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--variant",choices=VARIANTS,required=True);p.add_argument("--iterations",type=int,default=2)
    a=p.parse_args();main(a.root,a.variant,a.iterations)
