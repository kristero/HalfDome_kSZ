"""Evaluate the selected emulator once on untouched test rows and B12 mocks.

Writes publication-sized figures, per-bin/profile/parameter metrics, raw
predictions, a report, checksums and a completion marker. The fixed B12 noise
covariance is explicitly a reference precision scale away from B12.
"""
import argparse
import csv
import json
import shutil
import sys
import time
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import Emulator, PARAMETERS
from train import HERE, REPO, sha256, write_json

sys.path.insert(0, str(HERE.parent))
from train_so_noiseless_emulator import compute_test_metrics

B12 = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
EDGES = np.r_[np.arange(80, 7881, 200), 7980]
ELL = np.array([np.average(np.arange(a, b), weights=2*np.arange(a, b)+1)
                for a, b in zip(EDGES[:-1], EDGES[1:])])
BLUE, ORANGE, GREEN = "#2367b1", "#da6830", "#23856e"
LABELS = [r"$P_0$", r"$x_c$", r"$\beta$", r"$\alpha_{m,P_0}$", r"$\alpha_{m,x_c}$",
          r"$\alpha_{m,\beta}$", r"$\alpha_{z,P_0}$", r"$\alpha_{z,x_c}$", r"$\alpha_{z,\beta}$"]


def csv_rows(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def rebin_cl(path):
    cl = np.load(path)
    ell = np.arange(len(cl))
    dl = cl * ell*(ell+1)/(2*np.pi)
    return np.array([np.average(dl[a:b], weights=2*ell[a:b]+1)
                     for a, b in zip(EDGES[:-1], EDGES[1:])])


def summary_quantiles(values):
    return {name: float(np.quantile(values, q)) for name, q in
            (("min", 0), ("median", .5), ("p68", .68), ("p95", .95), ("p99", .99), ("max", 1))}


def extra_metrics(truth, prediction, covariance):
    difference = prediction-truth
    fractional = difference/truth
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), difference.T).T
    epsilon = np.linalg.norm(whitened, axis=1)
    rms_pct = 100*np.sqrt(np.mean(fractional**2, axis=1))
    return {"rmse_linear_dl": float(np.sqrt(np.mean(difference**2))),
            "mae_linear_dl": float(np.mean(np.abs(difference))),
            "profile_fractional_rmse_pct": summary_quantiles(rms_pct),
            "epsilon_B12": summary_quantiles(epsilon),
            "fraction_profiles_epsilon_below_0p1": float(np.mean(epsilon < .1)),
            "fraction_profiles_epsilon_below_0p3": float(np.mean(epsilon < .3)),
            "fraction_profiles_epsilon_below_1": float(np.mean(epsilon < 1)),
            "fraction_profiles_all_bins_below_1pct": float(np.mean(np.max(np.abs(fractional), axis=1)<.01)),
            "fraction_profiles_all_bins_below_5pct": float(np.mean(np.max(np.abs(fractional), axis=1)<.05))}, rms_pct, epsilon


def plot_profiles(figures, name, indices, truth, prediction, labels, scores, score_label):
    fig = plt.figure(figsize=(18, 10), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    for cell, i in enumerate(indices):
        inner = grid[cell//3, cell%3].subgridspec(2, 1, height_ratios=[2.1, 1], hspace=.04)
        ax = fig.add_subplot(inner[0]); residual = fig.add_subplot(inner[1], sharex=ax)
        ax.loglog(ELL, truth[i], color=BLUE, lw=2, label="Simulation")
        ax.loglog(ELL, prediction[i], color=ORANGE, lw=1.9, ls="--", label="Emulator")
        ax.set_title(f"{labels[i]}   {score_label} = {scores[i]:.2g}", fontsize=14)
        residual.semilogx(ELL, 100*(prediction[i]/truth[i]-1), color=ORANGE, lw=1.7)
        residual.axhline(0, color="0.45", lw=.8)
        residual.set_xlabel(r"$\ell$")
        residual.set_ylabel("Error [%]", fontsize=13)
        ax.set_ylabel(r"$D_\ell^{yy}$")
        ax.tick_params(labelbottom=False)
        if cell == 0:
            ax.legend(fontsize=12, loc="best")
    save_figure(fig, figures, name)


def make_plots(output, selection, truth, prediction, theta, row_ids, per_bin, rms_pct, epsilon):
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 16,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    candidates = selection["candidates"]
    fig, axes = plt.subplots(1, len(candidates), figsize=(7*len(candidates), 5), squeeze=False, layout="constrained")
    for ax, candidate in zip(axes.flat, candidates):
        history = np.genfromtxt(output/f"{candidate['label']}_history.csv", delimiter=",", names=True)
        ax.semilogy(history["epoch"], history["train_scaled_mse"], color=BLUE, label="Training")
        ax.semilogy(history["epoch"], history["validation_scaled_mse"], color=ORANGE, label="Validation")
        ax.axvline(candidate["best_epoch"], color=GREEN, ls=":", label="Selected epoch")
        ax.set(xlabel="Epoch", ylabel="Standardized log-spectrum MSE",
               title=f"Width {candidate['label'].split('_')[0][5:]}")
        ax.legend()
    save_figure(fig, folder, "loss_curves")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for key, label, color in [("median_ape_pct", "Median", BLUE), ("p95_ape_pct", "95th percentile", ORANGE),
                              ("p99_ape_pct", "99th percentile", GREEN)]:
        axes[0, 0].plot(ELL, [b[key] for b in per_bin], label=label, color=color, lw=2)
    axes[0, 0].set_ylabel("Absolute error [%]"); axes[0, 0].legend()
    axes[0, 1].plot(ELL, [b["mean_bias_pct"] for b in per_bin], color=BLUE, lw=2)
    axes[0, 1].axhline(0, color="0.5", lw=.8); axes[0, 1].set_ylabel("Mean signed error [%]")
    axes[1, 0].plot(ELL, np.sqrt(np.mean((100*(prediction/truth-1))**2, axis=0)), color=ORANGE, lw=2)
    axes[1, 0].set_ylabel("Fractional RMSE [%]")
    axes[1, 1].semilogy(ELL, [b["rmse_linear_dl"] for b in per_bin], color=GREEN, lw=2)
    axes[1, 1].set_ylabel(r"RMSE in $D_\ell^{yy}$")
    for ax in axes.flat:
        ax.set_xlabel(r"$\ell$"); ax.grid(alpha=.15)
    save_figure(fig, folder, "errors_by_multipole")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for values, label, color in [(rms_pct, "Profile fractional RMSE", BLUE),
                                 (np.max(np.abs(100*(prediction/truth-1)), axis=1), "Worst bin in profile", ORANGE)]:
        axes[0].semilogx(np.sort(values), np.arange(1, len(values)+1)/len(values), label=label, color=color, lw=2)
    axes[0].set(xlabel="Error [%]", ylabel="Cumulative fraction"); axes[0].legend(fontsize=12)
    axes[1].semilogx(np.sort(epsilon), np.arange(1,len(epsilon)+1)/len(epsilon), color=BLUE, lw=2)
    for threshold, color in [(.1, GREEN), (1., ORANGE)]:
        axes[1].axvline(threshold, ls="--", color=color, label=f"{threshold:g}")
    axes[1].set(xlabel=r"$\epsilon_{\rm B12}=\sqrt{\Delta D^T C_{\rm B12}^{-1}\Delta D}$", ylabel="Cumulative fraction")
    axes[1].legend(title="Reference thresholds")
    save_figure(fig, folder, "error_distributions")

    labels = [f"Row {r}" for r in row_ids]
    plot_profiles(folder, "worst_fractional_profiles", np.argsort(rms_pct)[-6:][::-1],
                  truth, prediction, labels, rms_pct, "RMSE [%]")
    plot_profiles(folder, "worst_covariance_profiles", np.argsort(epsilon)[-6:][::-1],
                  truth, prediction, labels, epsilon, r"$\epsilon_{\rm B12}$")
    ordered = np.argsort(rms_pct)
    representative = ordered[np.rint(np.array([0,.25,.5,.75,.95,.99])*(len(ordered)-1)).astype(int)]
    plot_profiles(folder, "representative_profiles", representative, truth, prediction, labels, rms_pct, "RMSE [%]")

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    for j, ax in enumerate(axes.flat):
        ax.scatter(theta[:, j], rms_pct, s=2, alpha=.12, color=BLUE, rasterized=True)
        edges = np.quantile(theta[:, j], np.linspace(0, 1, 9))
        centers, median, p95 = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            take = (theta[:, j]>=a)&(theta[:, j]<=b)
            centers.append((a+b)/2); median.append(np.median(rms_pct[take])); p95.append(np.quantile(rms_pct[take], .95))
        ax.plot(centers, median, color=ORANGE, lw=2, label="Median")
        ax.plot(centers, p95, color=GREEN, lw=2, label="95th percentile")
        ax.set_yscale("log"); ax.set_xlabel(LABELS[j])
        if j%3==0: ax.set_ylabel("Profile RMSE [%]")
        if j==0: ax.legend(fontsize=11)
    save_figure(fig, folder, "errors_by_parameter")

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    cloud = ax.hexbin(np.log10(truth).ravel(), np.log10(prediction).ravel(), gridsize=100,
                     bins="log", mincnt=1, cmap="viridis")
    bounds = [float(np.log10(truth).min()), float(np.log10(truth).max())]
    ax.plot(bounds, bounds, color=ORANGE, ls="--", lw=1.3)
    ax.set(xlabel=r"Simulation $\log_{10}D_\ell^{yy}$", ylabel=r"Emulator $\log_{10}D_\ell^{yy}$")
    fig.colorbar(cloud, ax=ax, label="Test-bin count")
    save_figure(fig, folder, "prediction_parity")


def external_diagnostics(emulator, output, covariance):
    root = REPO / "SBI_analysis/tsz64k_sbi_local/mocks"
    folders = [root/"pass1_B12/B12"] + sorted(p for p in (root/"fisher_b12").iterdir()
                 if p.is_dir() and (p/"case.toml").exists())
    labels, theta, truth, sources = [], [], [], []
    for folder in folders:
        config = tomllib.loads((folder/"case.toml").read_text())
        assert config["raw_nside"]==8192 and config["output_nside"]==4096
        assert config["mask_sha256"]==emulator.artifact["mask_sha256"]
        labels.append(config["label"]); theta.append(config["theta"])
        truth.append(rebin_cl(folder/"masked_clean_cl.npy"))
        sources.append({"path": str(folder/"masked_clean_cl.npy"), "sha256": sha256(folder/"masked_clean_cl.npy")})
    theta, truth = np.array(theta), np.array(truth)
    prediction = emulator.predict(theta)
    extra, rms, epsilon = extra_metrics(truth, prediction, covariance)
    rows = [{"label": label, "fractional_rmse_pct": float(rms[i]),
             "max_absolute_error_pct": float(np.max(np.abs(100*(prediction[i]/truth[i]-1)))),
             "epsilon_B12": float(epsilon[i]), **dict(zip(PARAMETERS, theta[i]))} for i,label in enumerate(labels)]
    csv_rows(output/"external_B12_metrics.csv", rows)
    np.savez_compressed(output/"external_B12_predictions.npz", theta=theta, truth=truth, prediction=prediction,
                        labels=labels, ell=ELL, epsilon_B12=epsilon)
    write_json(output/"external_B12_sources.json", sources)
    chosen = [0] + [int(i) for i in np.argsort(epsilon)[::-1] if i!=0][:5]
    plot_profiles(output/"figures", "battaglia12_and_local_checks", chosen, truth, prediction,
                  labels, epsilon, r"$\epsilon_{\rm B12}$")
    return rows, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"runs/main")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = args.output.resolve()
    torch.set_num_threads(4)
    selection = json.loads((out/"selection.json").read_text())
    emulator = Emulator(out/"emulator.pt", args.device)
    source = Path(emulator.artifact["dataset"])
    for name, digest in emulator.artifact["dataset_sha256"].items():
        assert sha256(source/name)==digest, f"Dataset changed: {name}"
    split = np.load(out/"split_indices.npz")
    all_indices = np.r_[split["train"], split["validation"], split["test"]]
    assert len(np.unique(all_indices))==65536 and np.array_equal(np.sort(all_indices), np.arange(65536))
    test = split["test"]
    theta = np.load(source/"theta.npy")[test]
    truth = np.load(source/"clean_dl.npy")[test]
    row_ids = np.load(source/"row_id.npy")[test]
    prediction = emulator.predict(theta)
    covariance_path = REPO/"SBI_analysis/tsz_guardrail_study/audit/noise_covariance.npz"
    covariance_bundle = np.load(covariance_path)
    covariance = covariance_bundle["covariance"]
    np.testing.assert_allclose(covariance_bundle["ell"], ELL, rtol=1e-6)
    assert covariance_bundle["ensemble"].shape==(64,40)
    overall, per_bin, profiles, _, _ = compute_test_metrics(truth,prediction,ELL,EDGES[:-1],EDGES[1:]-1)
    extra, rms_pct, epsilon = extra_metrics(truth,prediction,covariance)
    overall.update(extra)
    overall["covariance_scope"] = "64 SO split-noise draws at fixed older B12 sky; reference only away from B12 and across painter changes; excludes cosmic variance and model discrepancy"
    overall["covariance_sha256"] = sha256(covariance_path)
    write_json(out/"test_metrics.json", overall)
    csv_rows(out/"metrics_by_bin.csv", per_bin)
    profile_rows = [{"row_id": int(row_ids[i]), "source_index": int(test[i]),
                     "fractional_rmse_pct": float(rms_pct[i]), "epsilon_B12": float(epsilon[i]),
                     **{k: float(v[i]) for k,v in profiles.items()}, **dict(zip(PARAMETERS,theta[i]))}
                    for i in range(len(test))]
    csv_rows(out/"metrics_by_profile.csv", profile_rows)
    slice_rows = []
    for j,name in enumerate(PARAMETERS):
        unit = (theta[:,j]-emulator.artifact["lower"][j])/(emulator.artifact["upper"][j]-emulator.artifact["lower"][j])
        for quartile in range(4):
            take = (unit>=quartile/4)&(unit<=(quartile+1)/4)
            values = np.abs(100*(prediction[take]/truth[take]-1))
            slice_rows.append({"parameter":name,"prior_quartile":quartile+1,"profiles":int(take.sum()),
                "median_absolute_error_pct":float(np.median(values)),"p95_absolute_error_pct":float(np.quantile(values,.95)),
                "fractional_rmse_pct":float(np.sqrt(np.mean(values**2))),"median_epsilon_B12":float(np.median(epsilon[take]))})
    csv_rows(out/"metrics_by_parameter_quartile.csv",slice_rows)
    np.savez_compressed(out/"test_predictions.npz",theta=theta,row_id=row_ids,truth=truth,prediction=prediction,
                        ell=ELL,epsilon_B12=epsilon,fractional_rmse_pct=rms_pct)
    make_plots(out,selection,truth,prediction,theta,row_ids,per_bin,rms_pct,epsilon)
    external, external_extra = external_diagnostics(emulator,out,covariance)
    # Physics and serialization checks, independent of neural-network fit quality.
    pair = np.stack([B12, B12.copy()]); pair[1,0]*=1.5
    physical = emulator.predict(pair)
    np.testing.assert_allclose(physical[1]/physical[0],2.25,rtol=2e-12)
    np.testing.assert_allclose(emulator.predict(theta[:7],batch_size=1),prediction[:7],rtol=1e-4)
    cpu = Emulator(out/"emulator.pt","cpu")
    cpu_prediction = cpu.predict(theta[:32])
    np.testing.assert_allclose(cpu_prediction,prediction[:32],rtol=1e-4)
    rejected = False
    try:
        invalid=B12.copy();invalid[0]=61;cpu.predict(invalid)
    except ValueError:
        rejected=True
    assert rejected
    for _ in range(10):cpu.predict(B12)
    started=time.perf_counter()
    for _ in range(200):cpu.predict(B12)
    latency=(time.perf_counter()-started)/200
    checks={"split_disjoint_complete":True,"source_hashes_unchanged":True,"exact_P0_squared":True,
            "batch_consistency":True,"CPU_GPU_reload_agreement":True,"out_of_prior_rejected":True,
            "cpu_prediction_ms":1000*latency}
    write_json(out/"validation_checks.json",checks)
    engineering={"median_absolute_error_below_1pct":overall["median_ape_pct"]<=1,
                 "p95_absolute_error_below_5pct":overall["p95_ape_pct"]<=5,
                 "max_bin_bias_below_1pct":overall["max_abs_bin_bias_pct"]<=1,
                 "all_quartile_p95_below_7p5pct":max(r["p95_absolute_error_pct"] for r in slice_rows)<=7.5}
    scientific={"B12_epsilon_below_0p1":external[0]["epsilon_B12"]<.1,
                "all_external_local_epsilon_below_0p3":max(r["epsilon_B12"] for r in external)<.3,
                "test_p95_reference_epsilon_below_0p3":overall["epsilon_B12"]["p95"]<.3}
    gates={"engineering_checks":engineering,"engineering_pass":all(engineering.values()),
           "provisional_MCMC_checks":scientific,"provisional_MCMC_pass":all(scientific.values()),
           "interpretation":"Declared accuracy budgets, not calibrated posterior coverage; fixed B12 covariance is only a reference away from B12. No posterior validation performed."}
    write_json(out/"quality_gate.json",gates)
    report=f'''# 64k noiseless tSZ emulator

Trained on the complete 65,536-row design: 49,152 training, 6,554 validation,
9,830 untouched test rows. Selected **{selection['selected']}** using validation
log-spectrum RMSE only. All source rows were verified against the frozen design
and the two archived source pieces. No noisy spectra were used as targets.

## Held-out results

| Metric | Value |
|---|---:|
| Median absolute percentage error | {overall['median_ape_pct']:.6g}% |
| 95th percentile absolute percentage error | {overall['p95_ape_pct']:.6g}% |
| 99th percentile absolute percentage error | {overall['p99_ape_pct']:.6g}% |
| Fractional RMSE, all test bins | {overall['rms_percentage_error_pct']:.6g}% |
| RMSE in log10 D | {overall['rmse_log10_dl']:.6g} dex |
| RMSE in linear D | {overall['rmse_linear_dl']:.6g} |
| Mean signed percentage bias | {overall['mean_bias_pct']:.6g}% |
| Worst individual bin error | {overall['max_ape_pct']:.6g}% |
| Minimum per-bin linear R-squared | {overall['min_bin_r2']:.6g} |
| Test-bin predictions within 1% | {100*overall['fraction_abs_pct_below_1']:.6g}% |
| Test-bin predictions within 5% | {100*overall['fraction_abs_pct_below_5']:.6g}% |
| Median reference covariance distance | {overall['epsilon_B12']['median']:.6g} |
| 95th percentile reference covariance distance | {overall['epsilon_B12']['p95']:.6g} |
| Independent B12 reference covariance distance | {external[0]['epsilon_B12']:.6g} |
| Independent B12 fractional RMSE | {external[0]['fractional_rmse_pct']:.6g}% |

Engineering percentage gate: **{'PASS' if gates['engineering_pass'] else 'FAIL'}**.
Provisional MCMC accuracy gate: **{'PASS' if gates['provisional_MCMC_pass'] else 'FAIL'}**.

At the independent Battaglia12 point, the fractional RMSE is
{external[0]['fractional_rmse_pct']:.3f}%, but its reference covariance distance
is {external[0]['epsilon_B12']:.3f}. This is a distance in the 40-dimensional
spectrum space, not a measured parameter bias in sigma units. It exceeds the
declared local accuracy budget, so this version is not ready for the intended
MCMC comparison despite its good typical percentage accuracy.

The worst fractional-RMSE test profile is row {int(row_ids[np.argmax(rms_pct)])},
with RMSE {float(np.max(rms_pct)):.3f}% and reference covariance distance
{float(epsilon[np.argmax(rms_pct)]):.4g}. It is an extremely faint spectrum;
its large relative discrepancy has negligible absolute size on this reference
noise scale. Both rankings are retained so that this difference is visible.

The latter uses explicit budgets epsilon < 0.1 at B12, < 0.3 at every local
external point, and a 95th-percentile test reference epsilon < 0.3. These are
diagnostic accuracy budgets, not a posterior coverage certificate. The 64-draw
OAS covariance comes from the older fixed B12 signal and is only a reference
scale for the wide prior and updated painter. It excludes cosmic variance,
model discrepancy, and parameter-dependent signal-noise effects. It is not
divided by 64. Large distances at extremely bright prior corners should not be
read as candidate-specific posterior significances.

Linear RMSE and R-squared are dominated by the brightest spectra in this very
wide prior. Fractional RMSE, log RMSE, per-bin and parameter-slice diagnostics
are needed alongside them. Worst-profile panels rank whole-profile fractional
RMSE and reference covariance distance separately; all parameters and row IDs
are in metrics_by_profile.csv. The overall worst individual bin is preserved
in test_metrics.json and the per-profile max_ape_pct column.

## Physics and model

The pressure amplitude P0 multiplies the Compton-y map linearly, while its
power spectrum scales as P0 squared. The network therefore learns
log10(D/P0**2) from xc, beta and the six mass/redshift exponents, using log10
xc and log10 beta as input features. The final prediction restores P0**2 and
returns positive linear 40-bin D_ell = ell(ell+1) C_ell/(2 pi).

Targets are the same masked, 2-arcmin-beam clean spectra as the 64k SBI data,
with mode-count weights over ell=80..7979, and the saved wide nine-parameter
prior. Training uses per-bin standardized log targets and residual MLPs.
All normalizers are fitted only to training rows. Validation selects epochs,
network width, and the optional log-space ensemble; test rows do not select
any of these. Independent B12 and its finite-difference neighbours are used
only as external checks after training, never to tune this run.

## Figures and reusable artifacts

- figures/loss_curves.png: training and validation histories.
- figures/errors_by_multipole.png: percentiles, bias and both RMSE measures.
- figures/worst_fractional_profiles.png: six worst fractional-RMSE spectra.
- figures/worst_covariance_profiles.png: six worst reference-noise distances.
- figures/representative_profiles.png: ranked 0/25/50/75/95/99-percentile cases.
- figures/battaglia12_and_local_checks.png: independent B12 and local checks.
- figures/errors_by_parameter.png: error trends across all nine parameters.
- figures/error_distributions.png and prediction_parity.png: full test errors.

Every figure also has a vector PDF. emulator.pt is the reusable artifact;
model.py provides Emulator.predict(theta). split_indices.npz and provenance.json
record the split and sources. test_predictions.npz stores every test prediction.
The CPU single-profile timing here was {1000*latency:.3f} ms, with four Torch
threads and the selected artifact loaded; this is not an MCMC runtime.

No MCMC has been run. Good percentage performance alone does not establish
that the emulator is accurate enough for your conditional-noise likelihood.
'''
    (out/"RESULTS.md").write_text(report)
    snapshot=out/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ("model.py","train.py","evaluate.py","README.md"):
        shutil.copy2(HERE/name,snapshot/name)
    for name in ("train_so_noiseless_emulator.py","so_noiseless_emulator.py"):
        shutil.copy2(HERE.parent/name,snapshot/name)
    write_json(out/"COMPLETE.json",{"status":"complete","training_complete":True,"test_evaluation_complete":True,
        "test_rows":len(test),"external_profiles":len(external),"engineering_pass":gates["engineering_pass"],
        "provisional_MCMC_pass":gates["provisional_MCMC_pass"],"validation_checks_passed":True,
        "model_sha256":sha256(out/"emulator.pt")})
    files=sorted(p for p in out.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (out/"SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out).as_posix()}\n" for p in files))
    print(json.dumps({"metrics":overall,"quality_gate":gates,"checks":checks},indent=2))


if __name__=="__main__":
    main()
