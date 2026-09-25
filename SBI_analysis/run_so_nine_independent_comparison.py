#!/usr/bin/env python3
"""Matched Battaglia12 Fisher / 40-bin / PCA / MOPED analysis of collaborator data.

Stages: prepare, train, evaluate, summarize. The original hard bounds and bin
edges come from the delivered dataset. All nine parameters vary jointly.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from export_so_moped_bundle import sha256
from generate_so_fisher_variations import BATTAGLIA12, PARAMETER_NAMES
from so_nine_fisher import (conditional_covariance, fisher_matrix, fisher_modes,
    hard_prior_posterior, load_independent_dataset, moped_weights, read_npz,
    require, simulation_signature, write_csv)
from so_sbi_compression import array_digest, asinh_coordinates, fit_asinh, metrics_from_samples, save_npz, write_json
from so_fisher_compression import fit_linear_pca, project_likelihood

METHODS=("bins40","pca","moped")
LABELS=[r"P_0",r"x_{\rm c}",r"\beta",r"\alpha_{m,P_0}",r"\alpha_{m,x_{\rm c}}",
        r"\alpha_{m,\beta}",r"\alpha_{z,P_0}",r"\alpha_{z,x_{\rm c}}",r"\alpha_{z,\beta}"]


def features(x,transform):
    if str(transform["kind"])=="asinh":
        return np.asarray(asinh_coordinates(x,transform),dtype=np.float32)
    require(str(transform["kind"]) in ("moped_raw", "pca_raw"),"Unknown transform")
    return np.asarray(((x-transform["fiducial_dell"])@transform["matrix"]-
                       transform["mean"])/transform["std"],dtype=np.float32)


def validate_calibration(path,data,generation,dataset_path):
    complete=json.loads(path.with_name("calibration_complete.json").read_text())
    require(complete["complete"] and complete["calibration_sha256"]==sha256(path),"Incomplete/corrupt calibration")
    require(complete["observation_excluded_from_covariance"],"Covariance uses observation")
    cal=read_npz(path)
    require(str(cal["dataset_sha256"])==sha256(dataset_path),"Calibration is tied to different training data")
    require(str(cal["source_signature"])==simulation_signature(generation),"Calibration simulator differs")
    for key in ("param_names","prior_low","prior_high","ell_unbinned","ell_binned","bin_ell_min","bin_ell_max"):
        np.testing.assert_array_equal(cal[key],data[key])
    np.testing.assert_allclose(cal["fiducial"],BATTAGLIA12,rtol=0,atol=1e-14)
    require(cal["derivatives"].shape==(40,9) and cal["noise_ensemble"].shape[1:]==(40,) and
            len(cal["noise_ensemble"])>=64 and cal["observation"].shape==(40,),"Invalid calibration shapes")
    for key in ("derivatives","derivative_small","derivative_large","noise_ensemble","observation","fiducial_dell"):
        require(np.isfinite(cal[key]).all(),f"Nonfinite calibration {key}")
    np.testing.assert_allclose(cal["derivatives"],(4*cal["derivative_small"]-cal["derivative_large"])/3)
    covariance_seeds=cal["covariance_split_seeds"]
    observation_seeds=cal["observation_split_seeds"]
    require(covariance_seeds.shape==(len(cal["noise_ensemble"]),2) and observation_seeds.shape==(2,),"Invalid calibration seeds")
    all_seeds=np.concatenate([data["noise_split_seeds"].ravel(),covariance_seeds.ravel(),observation_seeds])
    require(len(np.unique(all_seeds))==len(all_seeds),"Training, covariance, or observation share a noise split")
    return cal


def prepare(args):
    data,generation=load_independent_dataset(args.dataset,args.generation_manifest,args.expected_rows)
    cal=validate_calibration(args.calibration,data,generation,args.dataset)
    require(not args.root.exists(),"Choose a fresh analysis root; existing results are preserved")
    n=len(data["theta"])
    require(4<=args.holdout<n-128,"Invalid held-out size")
    maximum=n-args.holdout
    sizes=sorted(set(maximum if s=="max" else int(s) for s in args.sizes.split(",")))
    require(128<=min(sizes)<=max(sizes)<=maximum,"Training sizes must fit the non-test pool")
    covariance,shrinkage=conditional_covariance(cal["noise_ensemble"])
    width=data["prior_high"]-data["prior_low"]
    weights,singular,error=moped_weights(cal["derivatives"]*width,covariance)
    fisher=fisher_matrix(cal["derivatives"]*width,covariance)
    values,vectors,keep=fisher_modes(fisher)
    args.root.mkdir(parents=True)
    save_npz(args.root/"data.npz",**data)
    save_npz(args.root/"calibration.npz",**cal,covariance=covariance,moped_weights=weights,
        fisher_normalized=fisher,fisher_theta=fisher/np.outer(width,width),
        fisher_eigenvalues=values,fisher_eigenvectors=vectors,resolved_modes=keep)
    rng=np.random.default_rng(args.seed)
    # Last rows held out before any training transform is fitted. Nested pool
    # subsets use one saved shuffle shared by both methods and all sizes.
    pool=rng.permutation(np.arange(maximum))
    test=np.arange(maximum,n)
    config=dict(dataset=str(args.dataset.resolve()),dataset_sha256=sha256(args.dataset),
        synthetic_test=bool(json.loads(str(data["metadata_json"])).get("synthetic_test",False)),
        calibration_sha256=sha256(args.calibration),generation_manifest_sha256=sha256(args.generation_manifest),
        source_signature=simulation_signature(generation),parameter_names=PARAMETER_NAMES,
        sizes=sizes,holdout=args.holdout,seed=args.seed,validation_fraction=.1,
        hidden_features=args.hidden_features,num_transforms=args.num_transforms,
        training_batch_size=args.batch_size,stop_after_epochs=args.patience,max_num_epochs=args.max_epochs,
        posterior_samples=args.posterior_samples,fiducial_samples=args.fiducial_samples,
        max_proposals=args.max_proposals,sampling_seconds=300,methods=list(METHODS),
        independent_noise=True,physical_contract_matched=True,n_noise=len(cal["noise_ensemble"]),
        oas_shrinkage=shrinkage,moped_fisher_relative_error=error,fisher_rank=int(keep.sum()),
        fisher_rcond=1e-8,prior_policy="use exact bounds saved in collaborator dataset",
        moped_definition="raw linear D_ell compression from matched calibration J and C",
        pca_components=9,
        pca_definition="linear PCA of train-standardized raw D_ell; training covariance only chooses axes",
        likelihood_covariance_policy="one conditional raw D_ell covariance; project as A.T @ C @ A; never refit after compression",
        fisher_approximation="linear mean and covariance frozen at Battaglia12; covariance derivatives omitted",
        source_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name("so_nine_fisher.py"),
                      Path(__file__).with_name("so_fisher_compression.py"),
                      Path(__file__).with_name("run_so_sbi_compression_comparison.py"))})
    config["experiment_id"]=array_digest(data["theta"],data["x"],covariance,weights,
        np.asarray(json.dumps(config,sort_keys=True)))
    for size in sizes:
        root=args.root/f"N{size}"
        root.mkdir()
        chosen=pool[:size]
        n_validation=max(2,int(round(size*.1)))
        fit,validation=chosen[:-n_validation],chosen[-n_validation:]
        ordered=np.concatenate([fit,validation])
        local=dict(config,n_train=size,n_fit=len(fit),n_test=len(test),
                   experiment_id=f"{config['experiment_id']}:N{size}")
        save_npz(root/"shared.npz",theta=data["theta"],low=data["prior_low"],high=data["prior_high"],
            param_names=data["param_names"],fit_indices=fit,validation_indices=validation,
            pool_indices=ordered,test_indices=test)
        transforms={"bins40":dict(kind=np.asarray("asinh"),**fit_asinh(data["x"][fit]))}
        transforms["pca"]=fit_linear_pca(data["x"][fit], components=9)
        compressed=(data["x"][fit]-cal["fiducial_dell"])@weights
        transforms["moped"]=dict(kind=np.asarray("moped_raw"),matrix=weights,fiducial_dell=cal["fiducial_dell"],
            mean=compressed.mean(axis=0),std=np.maximum(compressed.std(axis=0,ddof=1),1e-12))
        for method,transform in transforms.items():
            save_npz(root/f"{method}_transform.npz",**transform)
            np.save(root/f"{method}_x.npy",features(data["x"],transform))
        # All likelihoods inherit the same raw covariance. Output normalization
        # is part of A; neither PCA training scatter nor a fresh compressed OAS
        # estimate is a replacement for A.T @ C @ A.
        audit={}
        for method in METHODS:
            matrix=(np.eye(40) if method=="bins40" else
                    transforms[method]["matrix"]/transforms[method]["std"])
            projected=project_likelihood(cal["derivatives"]*width,covariance,
                cal["observation"]-cal["fiducial_dell"],matrix)
            save_npz(root/f"{method}_likelihood.npz",**projected,
                source_covariance_digest=np.asarray(array_digest(covariance)),
                parameter_coordinates=np.asarray("prior-width normalized"))
            audit[method]=dict(relative_fisher_change=float(projected["relative_fisher_change"]),
                covariance_dimension=len(projected["covariance"]))
            if method=="bins40":
                full_score=projected["score"]
            if method=="moped":
                score_error=np.linalg.norm(projected["score"]-full_score)/max(np.linalg.norm(full_score),1e-300)
                require(score_error<1e-8 and projected["relative_fisher_change"]<1e-8,
                        "MOPED fails the shared-covariance Fisher/observed-score identity")
                audit[method]["relative_score_error"]=float(score_error)
        write_json(root/"covariance_audit.json",dict(source_covariance_digest=array_digest(covariance),
            methods=audit,conditional_covariance_shared=True,
            bins40_note="Fisher computed in raw coordinates; NPE asinh is invertible",
            pca_note="Training scatter chooses PCA axes; it is not the conditional likelihood covariance"))
        write_json(root/"experiment.json",local)
    write_json(args.root/"experiment.json",config)
    write_json(args.root/"prepare_complete.json",dict(experiment_id=config["experiment_id"],
        artifacts={str(p.relative_to(args.root)):sha256(p) for p in args.root.rglob("*") if p.is_file()}))
    print(f"Prepared matched nine-parameter runs {sizes}; Fisher rank {keep.sum()}/9; MOPED error {error:.3g}")


def load_run(args):
    verify_prepared(args.root)
    root=args.root/f"N{args.n_train}"
    config=json.loads((root/"experiment.json").read_text())
    shared=read_npz(root/"shared.npz")
    top=json.loads((args.root/"experiment.json").read_text())
    require(config["experiment_id"]==f"{top['experiment_id']}:N{args.n_train}","Stale run")
    return root,config,shared


def verify_prepared(root):
    marker=json.loads((root/"prepare_complete.json").read_text())
    for relative,digest in marker["artifacts"].items():
        require(sha256(root/relative)==digest,f"Changed prepared input: {relative}")


def evaluate(args):
    import torch
    from run_so_sbi_compression_comparison import bounded_samples, configure_runtime_threads
    configure_runtime_threads()
    root,config,shared=load_run(args)
    run=root/args.method
    training=json.loads((run/"training_complete.json").read_text())
    require(training["experiment_id"]==config["experiment_id"] and
            training["weights_selection"]=="best_validation_snapshot","Wrong model or weight selection")
    with (run/"density_estimator.pkl").open("rb") as stream:
        model=pickle.load(stream).cpu().eval()
    transform=read_npz(root/f"{args.method}_transform.npz")
    cal=read_npz(args.root/"calibration.npz")
    data=read_npz(args.root/"data.npz")
    cached=np.load(root/f"{args.method}_x.npy",mmap_mode="r")
    np.testing.assert_array_equal(cached[shared["test_indices"]],features(data["x"][shared["test_indices"]],transform))
    output=run/"evaluation"
    output.mkdir(exist_ok=True)
    requests=[(f"row{int(i)}",int(i),cached[i],shared["theta"][i],config["posterior_samples"])
              for i in shared["test_indices"]]
    requests.append(("battaglia12",-1,features(cal["observation"],transform),BATTAGLIA12,config["fiducial_samples"]))
    metrics=[]
    for ordinal,(name,index,context,truth,count) in enumerate(requests):
        destination=output/f"{name}.npz"
        sampling_config=dict(config,posterior_samples=count)
        if destination.exists():
            saved=read_npz(destination)
            require(str(saved["experiment_id"])==config["experiment_id"],"Stale posterior samples")
            np.testing.assert_array_equal(saved["context"],context)
            np.testing.assert_array_equal(saved["truth"],truth)
            samples=saved["samples"]
            require(samples.shape==(count,9),"Incomplete posterior samples")
            acceptance=float(saved["acceptance"])
        else:
            torch.manual_seed(config["seed"]+ordinal)
            try:
                samples,proposals,acceptance=bounded_samples(model,context,shared["low"],shared["high"],sampling_config)
            except RuntimeError as exc:
                write_json(output/"failure.json",dict(observation=name,error=str(exc),
                    note="Evaluation is incomplete; no failed observation may be silently dropped"))
                raise
            save_npz(destination,samples=samples,truth=truth,context=context,acceptance=np.asarray(acceptance),
                proposals=np.asarray(proposals),experiment_id=np.asarray(config["experiment_id"]))
        require(np.isfinite(samples).all() and np.all((samples>=shared["low"])&(samples<=shared["high"])),
                "Posterior samples violate the original box prior")
        values=metrics_from_samples(samples,truth,shared["low"],shared["high"])
        for j,p in enumerate(PARAMETER_NAMES):
            metrics.append(dict(method=args.method,n_train=args.n_train,observation=name,test_index=index,
                parameter=p,acceptance=acceptance,**{k:float(v[j]) for k,v in values.items()}))
        if ordinal%25==0:
            print(f"{args.method} N={args.n_train}: {ordinal+1}/{len(requests)} evaluated",flush=True)
    write_csv(output/"metrics.csv",metrics)
    write_json(output/"complete.json",dict(experiment_id=config["experiment_id"],n_test=len(shared["test_indices"]),
        includes_independent_battaglia12=True,model_sha256=sha256(run/"density_estimator.pkl"),
        metrics_sha256=sha256(output/"metrics.csv")))


def summarize(args):
    import pandas as pd
    from scipy.linalg import solve_triangular
    verify_prepared(args.root)
    config=json.loads((args.root/"experiment.json").read_text())
    cal=read_npz(args.root/"calibration.npz")
    frames=[]
    for n in config["sizes"]:
        for method in config["methods"]:
            run=args.root/f"N{n}"/method
            folder=run/"evaluation"
            done=json.loads((folder/"complete.json").read_text())
            require(done["experiment_id"]==f"{config['experiment_id']}:N{n}" and done["n_test"]==config["holdout"],"Incomplete run")
            require(done["model_sha256"]==sha256(run/"density_estimator.pkl") and
                    done["metrics_sha256"]==sha256(folder/"metrics.csv"),"Changed evaluation/model")
            frame=pd.read_csv(folder/"metrics.csv")
            require(len(frame)==9*(config["holdout"]+1),"Missing metrics; selective omission is forbidden")
            require(not frame.duplicated(["observation","parameter"]).any(),"Duplicate metrics")
            frames.append(frame)
    output=args.root/"summary"
    require(not (output/"summary_complete.json").exists(),"Completed summary already exists; preserve it")
    output.mkdir(exist_ok=True)
    combined=pd.concat(frames,ignore_index=True)
    combined.to_csv(output/"per_profile_metrics.csv",index=False)
    aggregate=[]
    for (n,method,param),frame in combined[combined.test_index>=0].groupby(["n_train","method","parameter"]):
        aggregate.append(dict(n_train=n,method=method,parameter=param,
            rmse_prior=float(np.sqrt(np.mean(frame.normalized_error_prior**2))),
            rms_pull=float(np.sqrt(np.mean(frame.pull**2))),coverage68=float(frame.coverage68.mean()),
            coverage95=float(frame.coverage95.mean()),pearson_r=float(np.corrcoef(frame.truth,frame["mean"])[0,1])))
    write_csv(output/"convergence_metrics.csv",aggregate)
    low,high=cal["prior_low"],cal["prior_high"]
    samples={}
    sampler={}
    for i,(name,residual) in enumerate((("fisher_forecast",np.zeros(40)),
                                      ("fisher_observed",cal["observation"]-cal["fiducial_dell"]))):
        values,info=hard_prior_posterior(cal["derivatives"],cal["covariance"],residual,low,high,BATTAGLIA12,seed=config["seed"]+i)
        _,repeat=hard_prior_posterior(cal["derivatives"],cal["covariance"],residual,low,high,BATTAGLIA12,seed=config["seed"]+i+20)
        shift=np.abs(np.asarray(info["weighted_mean"])-repeat["weighted_mean"])/info["weighted_std"]
        change=np.abs(np.asarray(repeat["weighted_std"])/info["weighted_std"]-1)
        require(shift.max()<.05 and change.max()<.05,"Fisher integration is not stable")
        info.update(repeat_ess=repeat["importance_ess"],repeat_max_mean_shift_over_std=float(shift.max()),
                    repeat_max_relative_std_change=float(change.max()))
        samples[name],sampler[name]=values,info
        np.save(output/f"{name}_samples.npy",values)
    largest=max(config["sizes"])
    for method in config["methods"]:
        samples[method]=read_npz(args.root/f"N{largest}"/method/"evaluation/battaglia12.npz")["samples"]
    # PCA generally loses local information. Integrate its own projected
    # likelihood with the same observation and hard prior, rather than drawing
    # the full-data Fisher contour and relabelling it as PCA.
    if "pca" in config["methods"]:
        projected=read_npz(args.root/f"N{largest}"/"pca_likelihood.npz")
        values,info=hard_prior_posterior(projected["derivatives"]/(high-low),
            projected["covariance"],projected["residual"],low,high,BATTAGLIA12,seed=config["seed"]+30)
        _,repeat=hard_prior_posterior(projected["derivatives"]/(high-low),
            projected["covariance"],projected["residual"],low,high,BATTAGLIA12,seed=config["seed"]+31)
        shift=np.abs(np.asarray(info["weighted_mean"])-repeat["weighted_mean"])/info["weighted_std"]
        change=np.abs(np.asarray(repeat["weighted_std"])/info["weighted_std"]-1)
        require(shift.max()<.05 and change.max()<.05,"PCA Fisher integration is not stable")
        info.update(repeat_ess=repeat["importance_ess"],repeat_max_mean_shift_over_std=float(shift.max()),
                    repeat_max_relative_std_change=float(change.max()))
        samples["fisher_pca"],sampler["fisher_pca"]=values,info
        np.save(output/"fisher_pca_samples.npy",values)
    constraints=[]
    for name,array in samples.items():
        q=np.quantile(array,[.025,.16,.5,.84,.975],axis=0)
        for j,p in enumerate(PARAMETER_NAMES):
            constraints.append(dict(method=name,parameter=p,fiducial=BATTAGLIA12[j],mean=array[:,j].mean(),
                std=array[:,j].std(ddof=1),std_over_prior=array[:,j].std(ddof=1)/(high-low)[j],
                q025=q[0,j],q16=q[1,j],median=q[2,j],q84=q[3,j],q975=q[4,j]))
    write_csv(output/"battaglia12_constraints.csv",constraints)
    chol=np.linalg.cholesky(cal["covariance"])
    j=cal["derivatives"]
    stability=[]
    for key in ("derivative_small","derivative_large"):
        error=np.linalg.norm(solve_triangular(chol,cal[key]-j,lower=True),axis=0)/np.linalg.norm(solve_triangular(chol,j,lower=True),axis=0)
        for p,e in zip(PARAMETER_NAMES,error):
            stability.append(dict(derivative=key,parameter=p,relative_noise_weighted_change=e))
    write_csv(output/"derivative_stability.csv",stability)
    # Inspect finite-ensemble precision sensitivity without treating the Gaussian
    # prior-moment diagnostic as the reported hard-prior posterior.
    reference=np.sqrt(np.diag(np.linalg.inv(cal["fisher_normalized"]+12*np.eye(9))))
    sensitivity=[]
    for n in sorted(set([32,64,len(cal["noise_ensemble"])//2,len(cal["noise_ensemble"])])):
        c,s=conditional_covariance(cal["noise_ensemble"][:n])
        f=fisher_matrix(j*(high-low),c)
        sensitivity.append(dict(n_noise=n,shrinkage=s,rank=int(fisher_modes(f)[2].sum()),
            max_prior_moment_sigma_change=float(np.max(np.abs(np.sqrt(np.diag(np.linalg.inv(f+12*np.eye(9))))/reference-1)))))
    write_csv(output/"covariance_sensitivity.csv",sensitivity)
    write_csv(output/"fisher_modes.csv",[dict(mode=i+1,eigenvalue=cal["fisher_eigenvalues"][i],
        resolved=bool(cal["resolved_modes"][i]),
        **{p:cal["fisher_eigenvectors"][j,i] for j,p in enumerate(PARAMETER_NAMES)}) for i in range(9)])
    make_plots(output,samples,cal,pd.DataFrame(aggregate),largest,config.get("synthetic_test",False))
    write_json(output/"summary_complete.json",dict(complete=True,experiment_id=config["experiment_id"],
        physical_contract_matched=True,all_nine_parameters_varied=True,selective_omission=False,
        synthetic_test=config.get("synthetic_test",False),
        fisher_rank=config["fisher_rank"],moped_fisher_relative_error=config["moped_fisher_relative_error"],
        fisher_sampling=sampler,limitations=["Fixed halo lightcone and mask; no changing-sky variance.",
            "Fisher linearizes the mean and freezes covariance at Battaglia12.",
            "MOPED preserves the fitted local mean Fisher matrix, not necessarily global/covariance information.",
            "Finite covariance ensemble and local derivative errors require inspecting the saved sensitivity tables."],
        artifacts={p.name:sha256(p) for p in output.iterdir() if p.is_file()}))
    print(f"Matched comparison completed: {output}")


def make_plots(output,samples,cal,metrics,largest,synthetic_test=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples,plots
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":14})
    names=[f"p{i}" for i in range(9)]
    ranges={n:(cal["prior_low"][i],cal["prior_high"][i]) for i,n in enumerate(names)}
    labels={"fisher_observed":"Fisher + hard prior","fisher_forecast":"Fiducial Fisher + hard prior",
            "fisher_pca":"PCA Fisher + hard prior",
            "bins40":"40-bin SBI","pca":"PCA SBI","moped":"MOPED SBI"}
    colors={"fisher_observed":"#222222","fisher_forecast":"#222222","fisher_pca":"#CC79A7",
            "bins40":"#0072B2","pca":"#CC79A7","moped":"#D55E00"}
    available=[m for m in METHODS if m in samples]
    def save(fig,stem):
        for ext in ("png","pdf"):
            fig.savefig(output/f"{stem}.{ext}",dpi=180,bbox_inches="tight")
        plt.close(fig)
    comparisons=[("battaglia12_fisher_sbi_9param",["fisher_observed"]+available),
                 ("battaglia12_fiducial_fisher_9param",["fisher_forecast"])]
    if "fisher_pca" in samples:
        comparisons.append(("battaglia12_pca_fisher_sbi_9param",["fisher_pca","pca"]))
    for stem,methods in comparisons:
        roots=[]
        for method in methods:
            array=samples[method]
            select=np.random.default_rng(12).choice(len(array),min(20000,len(array)),replace=False)
            roots.append(MCSamples(samples=array[select],names=names,labels=LABELS,ranges=ranges,
                settings={"smooth_scale_1D":.3,"smooth_scale_2D":.4,"fine_bins_2D":128}))
        g=plots.get_subplot_plotter(width_inch=19)
        g.settings.scaling=False
        g.settings.axes_fontsize=13
        g.settings.lab_fontsize=17
        g.settings.legend_fontsize=16
        g.settings.figure_legend_frame=False
        g.triangle_plot(roots,filled=[m in METHODS for m in methods],markers=BATTAGLIA12,
            contour_colors=[colors[m] for m in methods],legend_labels=[labels[m] for m in methods],
            line_args=[dict(color=colors[m],ls="--" if m.startswith("fisher") else "-") for m in methods],
            marker_args={"color":"black","ls":":","lw":.7})
        title="Synthetic software test" if synthetic_test else "Battaglia12"
        g.fig.suptitle(title,y=1.01)
        save(g.fig,stem)
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for ax,p,label in zip(axes.flat,PARAMETER_NAMES,LABELS):
        for method in available:
            values=metrics[(metrics.parameter==p)&(metrics.method==method)].sort_values("n_train")
            ax.plot(values.n_train,values.rmse_prior,"o-",label=method,color=colors[method])
        ax.set(xscale="log",xlabel="Training + validation rows",ylabel="RMSE / prior width",title=f"${label}$")
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    fig.tight_layout()
    save(fig,"nine_parameter_convergence")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    eigenvalues=cal["fisher_eigenvalues"]
    axes[0].semilogy(np.arange(1,10),np.maximum(eigenvalues,1e-20),"o-")
    axes[0].axhline(eigenvalues.max()*1e-8,color="gray",ls=":",label="Numerical rank threshold")
    axes[0].axhline(12,color="black",ls="--",label="Prior-moment precision (diagnostic)")
    axes[0].set(xlabel="Ordered mode",ylabel="Fisher eigenvalue in prior-width units")
    axes[0].legend(fontsize=8)
    covariance=cal["covariance"]
    correlation=covariance/np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)))
    im=axes[1].imshow(correlation,vmin=-1,vmax=1,cmap="RdBu_r",origin="lower")
    axes[1].set(xlabel="Bandpower bin",ylabel="Bandpower bin",title="Independent-noise OAS correlation")
    fig.colorbar(im,ax=axes[1],shrink=.85)
    fig.tight_layout()
    save(fig,"fisher_eigenvalues_and_covariance")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","train","evaluate","summarize"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--generation-manifest",type=Path)
    p.add_argument("--calibration",type=Path)
    p.add_argument("--sizes",default="4096,8192,16384,max")
    p.add_argument("--holdout",type=int,default=1000)
    p.add_argument("--expected-rows",type=int,default=32768,help="Change only for a deliberately different dataset or synthetic tests")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--n-train",type=int)
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--hidden-features",type=int,default=50)
    p.add_argument("--num-transforms",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=50)
    p.add_argument("--patience",type=int,default=60)
    p.add_argument("--max-epochs",type=int,default=2000)
    p.add_argument("--posterior-samples",type=int,default=2000)
    p.add_argument("--fiducial-samples",type=int,default=20000)
    p.add_argument("--max-proposals",type=int,default=1000000)
    args=p.parse_args()
    if args.stage=="prepare":
        require(all(getattr(args,k) is not None for k in ("dataset","generation_manifest","calibration")),
                "prepare requires --dataset, --generation-manifest and --calibration")
        prepare(args)
    elif args.stage=="train":
        require(args.method is not None and args.n_train is not None,"Select --method and --n-train")
        from run_so_sbi_compression_comparison import train
        root,config,shared=load_run(args)
        train(root,args.method,config,shared)
    elif args.stage=="evaluate":
        require(args.method is not None and args.n_train is not None,"Select --method and --n-train")
        evaluate(args)
    else:
        summarize(args)


if __name__=="__main__":
    main()
