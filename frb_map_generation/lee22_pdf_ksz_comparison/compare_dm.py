#!/usr/bin/env python3
"""Rehistogram the exact retained rays for both Lee22 normalizations and Ralf."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "frb_map_generation/outputs/tng300_lee22_mass13to14_20260921"
OUT = ROOT / "frb_map_generation/outputs/lee22_pdf_ksz_comparison_20260922"
RALF = ROOT / "frb_catalog_comparison_outputs/haloDM-20260819T104507Z-1-001/haloDM"


def histogram(values, edges):
    positive = values[values > 0]
    counts = np.histogram(positive, edges)[0]
    pdf = counts / (len(positive)*np.diff(edges))
    assert counts.sum()+np.count_nonzero(positive < edges[0])+np.count_nonzero(positive > edges[-1]) == len(positive)
    return pdf, counts, dict(mean_all=float(values.mean()), mean_positive=float(positive.mean()),
                            median_positive=float(np.median(positive)), hit_fraction=len(positive)/len(values),
                            n_rays=len(values), histogram_probability=float(np.sum(pdf*np.diff(edges))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = ROOT / "frb_map_generation/outputs/lee22_fig5_audit_20260922/audit.json"
    audit = json.loads(audit_path.read_text())
    scale = audit["paper_over_native"]
    ralf_z = np.load(RALF/"redshifts_DMhalo.npy")
    iz, = np.flatnonzero(np.isclose(ralf_z,1,atol=1e-8,rtol=0))
    reference_path = RALF/"DMhalo_r200_mass13to14.npy"
    ralf = np.asarray(np.load(reference_path, mmap_mode="r")[iz],dtype=float)
    report = dict(dm_scale_paper_over_previous=scale, ksz_power_scale_paper_over_previous=scale**2,
                  method="Exact per-ray linear normalization change followed by fresh histograms; no painting rerun needed.",
                  mass_msun=[1e13,1e14], source_redshift=1, sphere_r200c=1, seed=42,n_rays=120000,
                  pdf="conditional on positive DM", signed_percentage_limits=[-20,20], no_arrows=True,
                  low_mass_extrapolation=True, statistics={}, errors={}, input_sha256={})
    colors = {"Ralf":"#222222","previous":"#D55E00","paper":"#0072B2"}
    labels = {"Ralf":"Ralf IllustrisTNG300", "previous":"Lee22: previous normalization", "paper":"Lee22: paper normalization"}
    plt.rcParams.update({"font.size":15,"axes.labelsize":18,"axes.titlesize":18,
                         "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":13,
                         "pdf.fonttype":42,"ps.fonttype":42})
    fig, axes = plt.subplots(2,2,figsize=(14.2,8.6),sharex=True,sharey="row",
                            gridspec_kw={"height_ratios":[3,1.2]},layout="constrained")
    bin_rows=[]
    arrays={"ralf_dm":ralf}
    for col,(name,subdir) in enumerate((("HalfDome","halfdome_direct"),("TNG300-1-Dark","tng"))):
        path=OLD/subdir/"halo_dm.h5"
        with h5py.File(path,"r") as f:
            for key,value in {"normalization":"xgpaint_ne2d","concentration":"none","n0_pivot":"mcut","halo_boundary":"spherical"}.items():
                actual=f.attrs[key];actual=actual.decode() if isinstance(actual,bytes) else str(actual)
                assert actual==value,(key,actual)
            assert f.attrs["complete"] and f.attrs["sphere_r200c"]==1
            assert f.attrs["source_redshift"]==1 and f.attrs["seed"]==42
            previous=f["dm_pc_cm3"][:][0]
            edges=f["pdf_bin_edges_pc_cm3"][:]
            np.testing.assert_array_equal(f["window_effective_min_msun"][:],[1e13])
            np.testing.assert_array_equal(f["window_effective_max_msun"][:],[1e14])
            old_count=f["pdf_count"][:][0]
        paper=previous*scale
        np.testing.assert_array_equal(paper==0,previous==0)
        values={"Ralf":ralf,"previous":previous,"paper":paper}
        results={key:histogram(v,edges) for key,v in values.items()}
        np.testing.assert_array_equal(results["previous"][1],old_count)
        assert abs(results["paper"][2]["mean_all"]/results["previous"][2]["mean_all"]-scale)<1e-13
        arrays[name+"_previous_dm"]=previous
        arrays[name+"_paper_dm"]=paper
        arrays["edges"]=edges
        report["statistics"][name]={key:r[2] for key,r in results.items()}
        report["errors"][name]={}
        centers=np.sqrt(edges[:-1]*edges[1:]);widths=np.diff(edges)
        ref,ref_count,_=results["Ralf"]
        residuals={}
        for key,(pdf,count,metrics) in results.items():
            axes[0,col].plot(centers,np.where(pdf>0,pdf,np.nan),lw=2,color=colors[key],label=labels[key])
            if key=="Ralf":continue
            valid=(count>=10)&(ref_count>=10)
            delta=np.full_like(pdf,np.nan);delta[valid]=100*(pdf[valid]/ref[valid]-1)
            residuals[key]=delta
            weights=ref[valid]*widths[valid]
            report["errors"][name][key]={"valid_bins":int(valid.sum()),
                "mean_signed_percent":float(delta[valid].mean()),
                "mean_absolute_percent":float(np.abs(delta[valid]).mean()),
                "reference_probability_weighted_absolute_percent":float(np.average(np.abs(delta[valid]),weights=weights)),
                "reference_probability_in_valid_bins":float(weights.sum()),
                "above20_bins":int(np.count_nonzero(delta>20)),"below_minus20_bins":int(np.count_nonzero(delta< -20))}
            axes[1,col].plot(centers,delta,lw=1.7,color=colors[key])
        for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            row=dict(simulation=name,dm_left=lo,dm_right=hi)
            for key in values:
                row["pdf_"+key]=results[key][0][i];row["count_"+key]=int(results[key][1][i])
            row.update({"percent_"+k:float(v[i]) for k,v in residuals.items()})
            bin_rows.append(row)
        axes[0,col].set(title=name,yscale="log",ylim=(1e-7,.08))
        axes[0,col].legend(loc="lower left",frameon=False)
        axes[1,col].set(ylim=(-20,20),yticks=[-20,-10,0,10,20],xlabel=r"$\mathrm{DM}_{\rm halo}\ [\mathrm{pc\,cm^{-3}}]$")
        axes[1,col].axhline(0,lw=1,color=".5")
        for ax in axes[:,col]:
            ax.set(xscale="log",xlim=(.1,30000));ax.grid(alpha=.15);ax.tick_params(which="both",direction="in",top=True,right=True)
        report["input_sha256"][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    axes[0,0].set_ylabel(r"$p(\mathrm{DM}_{\rm halo}\mid\mathrm{DM}_{\rm halo}>0)$")
    axes[1,0].set_ylabel(r"$\Delta p/p_{\rm Ralf}$ [%]")
    fig.suptitle(r"$10^{13}\leq M_{200c}/M_\odot<10^{14}$; $z_s=1$; $1R_{200c}$",fontsize=18)
    fig.supxlabel("Lee22 includes low-mass extrapolation; residual display limited to +/-20%",fontsize=11)
    for suffix in ("pdf","png","svg"):fig.savefig(OUT/f"dm_pdf_ralf_lee22_normalizations.{suffix}",dpi=220)
    plt.close(fig)
    np.savez_compressed(OUT/"dm_ray_arrays.npz",**arrays)
    with (OUT/"dm_pdf_and_percentages.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bin_rows[0]));writer.writeheader();writer.writerows(bin_rows)
    report["input_sha256"][str(reference_path.relative_to(ROOT))]=hashlib.sha256(reference_path.read_bytes()).hexdigest()
    (OUT/"dm_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"statistics":report["statistics"],"errors":report["errors"]},indent=2))


if __name__=="__main__":main()
