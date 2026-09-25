"""Read-only diagnosis of why beta cannot reach 16 on the fixed-B12 slice.

The shaded extension passes every existing guard except the minimum relative
size. It is a counterfactual comparison, not a replacement production prior.
"""
import csv
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

from flat_prior import FlatPrior, HERE
from guardrails import B12, JointPrior


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = FlatPrior()
    before = {name:digest(HERE/name) for name in
              ("prior.json","flat_prior.py","guardrails.py","guardrails.json","torch_prior.py")}
    out = HERE / "outputs_beta_extension"
    out.mkdir(exist_ok=True)
    beta = np.unique(np.r_[np.linspace(prior.beta_limits[0],16,1600),prior.beta_limits[1]])
    ymin,ymax = prior._unit_y_extrema(beta)
    pmin = np.maximum(1,.003/ymin)
    pmax = np.minimum(60,30/ymax)

    # Disable just the lower relative-size test in this temporary diagnostic
    # object. The active prior JSON and guard implementation are never edited.
    diagnostic_config = dict(prior.guards.config, size_ratio_to_battaglia12=[0.,8.])
    other_guards = JointPrior(diagnostic_config)
    representative = np.column_stack(((pmin+pmax)/2,beta))
    assert other_guards.contains(prior.expand(representative)).all()
    included = beta <= prior.beta_limits[1]
    assert prior.contains(representative[included]).all()
    assert not prior.contains(representative[~included]).any()

    reference = HERE.parent / "flamingo_linear_prior/cluster_results/inputs"
    metadata = json.loads((reference/"old_metadata.json").read_text())["verified_bundle"]
    fit = json.loads((reference/"flamingo_fits.json").read_text())
    points = [("Battaglia12",B12[[0,2]],"black","*")]
    for key,label,color,marker in zip(
        ["L1_m9","fgas-8sigma","Mstar-1sigma"],
        ["FLAMINGO fiducial",r"FLAMINGO $f_{\rm gas}-8\sigma$",r"FLAMINGO $M_\star-1\sigma$"],
        ["#0072B2","#D55E00","#009E73"],["o","^","D"]):
        points.append((label,np.asarray(fit["corrected_parameters"][key])[[0,2]],color,marker))

    plt.rcParams.update({"font.size":16,"axes.labelsize":21,"axes.titlesize":20,
                         "xtick.labelsize":15,"ytick.labelsize":15,"legend.fontsize":14,
                         "pdf.fonttype":42,"ps.fonttype":42})
    fig,axes = plt.subplots(1,2,figsize=(14,7.2),gridspec_kw={"width_ratios":[1.3,1]})
    for ax in axes:
        ax.fill_betweenx(beta[included],pmin[included],pmax[included],color="#D5D8DC")
        extension = beta >= prior.beta_limits[1]
        ax.fill_betweenx(beta[extension],pmin[extension],pmax[extension],
                         facecolor="#F2D2AF",edgecolor="#A56323",hatch="///",linewidth=.4)
        ax.plot(pmin,beta,color=".25",lw=2)
        ax.axhline(prior.beta_limits[1],color=".35",ls=":",lw=1.7)
        ax.set(xlabel=r"$P_{0,0}$",ylabel=r"$\beta_0$",ylim=(2.8,16.35))
        ax.spines[["right","top"]].set_visible(False)
    low=np.asarray(metadata["prior_low"])[[0,2]]
    high=np.asarray(metadata["prior_high"])[[0,2]]
    axes[0].add_patch(Rectangle(low,*(high-low),fill=False,edgecolor="#882E72",ls="--",lw=2))
    for _,values,color,marker in points:
        axes[0].scatter(*values,color=color,marker=marker,s=180 if marker=="*" else 95,
                        edgecolor="white",linewidth=.8,zorder=5)
    axes[0].set_xlim(0,61)
    axes[0].set_title("Full parameter range")
    axes[1].set_xlim(.8,5.2)
    axes[1].set_title("Amplitude boundary")
    axes[1].plot([float(pmin[-1])],[16],marker="o",color=".2",ms=6)
    handles=[Patch(facecolor="#D5D8DC",label="Passes all current guards"),
             Patch(facecolor="#F2D2AF",edgecolor="#A56323",hatch="///",label="Fails only the size guard"),
             Line2D([],[],color="#882E72",ls="--",lw=2,label="Old SBI boundaries")]
    handles += [Line2D([],[],marker=m,color=c,ls="none",markersize=10,label=label) for label,_,c,m in points]
    fig.legend(handles=handles,loc="upper center",ncol=3,frameon=False)
    fig.subplots_adjust(top=.73,bottom=.13,left=.075,right=.98,wspace=.25)
    for ext in ("png","pdf"):
        fig.savefig(out/("beta_extension_diagnosis."+ext),dpi=240,bbox_inches="tight",facecolor="white")
    plt.close(fig)

    rows=[]
    for b in [8.,10.5,10.875,12.,14.,16.]:
        lo,hi=prior._unit_y_extrema(b)
        rows.append(dict(beta0=b,P0_min_from_pressure=max(1.,float(.003/lo)),
                         P0_max_from_pressure=min(60.,float(30/hi)),
                         relative_size=4.35/b,passes_size_mathematically=4.35/b>=.4))
    with (out/"amplitude_vs_beta.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    np.savetxt(out/"pressure_boundary.csv",np.column_stack((beta,pmin,pmax)),delimiter=",",
               header="beta0,P0_lower_all_guards_except_minimum_size,P0_upper",comments="")
    result=dict(active_prior_changed=False,cluster_jobs_submitted=False,
                explanation="With seven fixed B12 parameters the size ratio is 4.35/beta0, independent of P0",
                upper_beta_from_size=4.35/.4,beta16_pressure_amplitude_range=[float(pmin[-1]),float(pmax[-1])],
                beta16_size_ratio=4.35/16,required_size_lower_bound_to_include_16=4.35/16,
                minimum_xc_amplitude_for_beta16_with_B12_evolution=.4*.497*16/4.35,
                diagnostic_curve_points=len(beta),all_other_guards_pass=True,
                counterfactual_is_validated_production_prior=False,source_sha256=before,
                script_sha256=digest(HERE/"explore_beta_extension.py"),
                reference_sha256={name:digest(reference/name) for name in ("old_metadata.json","flamingo_fits.json")})
    for name,expected in before.items():
        assert digest(HERE/name)==expected
    (out/"diagnosis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
