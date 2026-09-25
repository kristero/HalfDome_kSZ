"""Publication figures from completed full maps; no surrogate predictions."""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0,str(root/"code"))
    from worker import EDGES, bin_cl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":16,"axes.labelsize":18,"xtick.labelsize":14,
        "ytick.labelsize":14,"legend.fontsize":12,"lines.linewidth":2,
        "axes.titlesize":18,"pdf.fonttype":42,"ps.fonttype":42})
    cases = json.loads((root/"preflight/cases.json").read_text())
    run = json.loads((root/"run_config.json").read_text())
    ell = np.array([np.average(np.arange(a,b),weights=2*np.arange(a,b)+1)
                    for a,b in zip(EDGES[:-1],EDGES[1:])])
    clean,noisy = [],[]
    for i,case in enumerate(cases):
        with h5py.File(root/"preflight"/(str(i//2)+".h5"),"r") as handle:
            clean.append(bin_cl(handle[str(i)]["masked_clean_cl"][:]))
            noisy.append(bin_cl(handle[str(i)]["masked_noisy_cross_cl"][:]))
    colors = ["#0072B2","#D55E00","#009E73"]
    labels = ["FLAMINGO fid.","Low gas","Low stars"]
    fig,axes = plt.subplots(1,2,figsize=(14,5.2),constrained_layout=True)
    axes[0].plot(ell,1e12*clean[0],color="black",label="Battaglia12")
    for i,(case,color,label) in enumerate(zip(cases[1:4],colors,labels),1):
        observed = bin_cl(np.load(Path(run["campaign"])/"results"/case["name"]/"masked_clean_cl.npy"))
        axes[0].plot(ell,1e12*observed,color=color,label=label)
        axes[0].plot(ell,1e12*clean[i],color=color,ls="--")
    handles,legends = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color=".4",ls="--")); legends.append("HalfDome fits")
    axes[0].legend(handles,legends,frameon=False)
    axes[1].plot(ell,1e12*clean[0],color="black",label="Battaglia12")
    for i,color,label in ((4,"#CC79A7","Compact boundary"),(5,"#946000","Steep boundary")):
        axes[1].plot(ell,1e12*clean[i],color=color,label=label)
    axes[1].legend(frameon=False)
    for ax,title in zip(axes,("FLAMINGO matches","Accepted extremes")):
        ax.set(xlabel=r"$\ell$",ylabel=r"$10^{12}D_\ell^{yy}$",title=title,
               xlim=(80,8000),xscale="log",yscale="log")
    out=root/"plots"
    for ext in ("png","pdf"): fig.savefig(out/("tsz_preflight."+ext),dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.2),constrained_layout=True)
    for i,(case,color,label) in enumerate(zip(cases[1:4],colors,labels),1):
        observed=Path(run["campaign"])/"results"/case["name"]
        for ax,key in zip(axes,("masked_clean_cl","masked_noisy_cross_cl")):
            values=bin_cl(np.load(observed/(key+".npy")))
            ax.plot(ell,1e12*values,color=color,label=label)
    for ax,title in zip(axes,("Clean signal","Fixed SO noise")):
        ax.set(xlabel=r"$\ell$",ylabel=r"$10^{12}D_\ell^{yy}$",title=title,xlim=(80,8000))
        ax.axhline(0,color=".7",lw=1)
    axes[0].legend(frameon=False)
    for ext in ("png","pdf"): fig.savefig(out/("flamingo_noise."+ext),dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
