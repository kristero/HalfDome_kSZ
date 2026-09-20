#!/usr/bin/env python3
"""NPE analysis of the 8192-row broad linear-prior HalfDome tSZ dataset (idark, sbi 0.22).

Stages
  prepare   nested training sizes from the 7349 non-validation rows; the 843 validation-split rows are
            the held-out test set; per size: signed-asinh coordinates, 40 bins / PCA(9) / local MOPED
            fitted on the optimization rows only; features saved for every row.
  train     one NPE (MAF or NSF) per (method, size, estimator) with the joint conditional prior.
  evaluate  batched posterior samples for every test row; RMSE / prior range and Pearson r per parameter.
  summarize convergence figures (dataset size vs RMSE, vs Pearson r) and a CSV table.
  corner    posteriors for the observations (HalfDome B12, Lee22 no-c, FLAMINGO L1_m9, fgas-8sigma,
            Mstar-1sigma) at the largest size with the best estimator per method; one triangle plot per
            observation, 40 bins / PCA / MOPED overlaid, prior marginals on the diagonal.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUN_DEFAULT = Path("/lustre/work/kristero10/halfdome_flamingo_linear_8192_20260915")
NAMES = ["P0", "xc", "beta", "alpha_m_P0", "alpha_m_xc", "alpha_m_beta", "alpha_z_P0", "alpha_z_xc", "alpha_z_beta"]
LATEX = {"P0": r"P_0", "xc": r"x_c", "beta": r"\beta", "alpha_m_P0": r"\alpha_m^{P_0}", "alpha_m_xc": r"\alpha_m^{x_c}",
         "alpha_m_beta": r"\alpha_m^{\beta}", "alpha_z_P0": r"\alpha_z^{P_0}", "alpha_z_xc": r"\alpha_z^{x_c}", "alpha_z_beta": r"\alpha_z^{\beta}"}
FIDUCIAL = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
METHODS = ("bins40", "pca", "moped")
METHOD_LABEL = {"bins40": "40 bins", "pca": "PCA (9)", "moped": "MOPED"}
METHOD_COLOR = {"bins40": "#2a78d6", "pca": "#eb6834", "moped": "#1baf7a"}
ESTIMATORS = ("maf", "nsf")
OBS_NAMES = ("HalfDome", "Lee22_noconc", "L1_m9", "fgas-8sigma", "Mstar-1sigma")
OBS_TITLE = {"HalfDome": "HalfDome, Battaglia12 pressure", "Lee22_noconc": "HalfDome, Lee22 no-c pressure",
             "L1_m9": "FLAMINGO L1_m9 (fiducial)", "fgas-8sigma": r"FLAMINGO fgas$-8\sigma$", "Mstar-1sigma": r"FLAMINGO M$_\star-1\sigma$"}

sys.path.insert(0, str(HERE))
from so_sbi_compression import (fit_asinh, asinh_coordinates, fit_pca, quadratic_design, shrunk_covariance,
                                moped_basis, project, finish_transform, pearson_columns)


# ----------------------------------------------------------------------------- data
def load_run(run):
    d = run / "dataset"
    data = dict(theta=np.load(d / "theta.npy"), noisy=np.load(d / "masked_noisy_cross_dl40.npy"),
                clean=np.load(d / "masked_clean_dl40.npy"), validation=np.load(d / "validation_split.npy").astype(bool),
                edges=np.load(d / "bin_edges.npy"))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["parameter_order"] == NAMES
    data["prior_config"] = manifest["prior"]
    data["low"] = np.asarray(manifest["prior"]["lower"], float); data["high"] = np.asarray(manifest["prior"]["upper"], float)
    data["Z"] = json.loads((run / "audit.json").read_text())["normalizing_mass_Z"]
    return data


def prior_objects(run, data):
    sys.path.insert(0, str(run / "code"))
    from prior import JointPrior
    from sbi_prior import ExtendedPrior
    return JointPrior(data["prior_config"]), ExtendedPrior(data["prior_config"], data["Z"])


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, default=float))


def hyper(n_train, smoke=False):
    batch = 32 if n_train <= 512 else 64 if n_train <= 2048 else 100
    h = dict(hidden_features=50, num_transforms=5, num_bins=8, training_batch_size=batch, learning_rate=5e-4,
             validation_fraction=0.1, stop_after_epochs=80, max_num_epochs=4000, clip_max_norm=5.0)
    if smoke:
        h.update(max_num_epochs=3, stop_after_epochs=2)
    return h


# ----------------------------------------------------------------------------- MOPED for small N
def linear_design(delta):
    return np.column_stack([np.ones(len(delta))] + [delta[:, j] for j in range(delta.shape[1])])


def fit_moped_adaptive(noisy, clean, theta, width, local_n, shrinkage, rcond):
    """Local MOPED around Battaglia12 (as so_sbi_compression.fit_moped) with a linear mean model when fewer
    than 600 optimization rows are available (the 55-term quadratic needs more rows)."""
    delta = (theta - FIDUCIAL) / width
    radius = np.linalg.norm(delta, axis=1)
    local_n = min(local_n, len(theta))
    quadratic = local_n >= 600
    nearest = np.argsort(radius, kind="stable")[:local_n]
    design = (quadratic_design if quadratic else linear_design)(delta[nearest])
    clean_coeff = np.linalg.lstsq(design, clean[nearest], rcond=None)[0]
    residual = noisy[nearest] - clean[nearest]
    bias_coeff, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Local design is rank deficient")
    mean_coeff = clean_coeff + bias_coeff
    derivatives = mean_coeff[1:10].T
    residual_centered = (residual - design @ bias_coeff) * np.sqrt((local_n - 1) / (local_n - rank))
    covariance = shrunk_covariance(residual_centered, shrinkage)
    result = moped_basis(derivatives, covariance, rcond)
    result.update(center=mean_coeff[0], derivatives=derivatives, covariance=covariance, local_n=local_n,
                  mean_model="quadratic" if quadratic else "linear", local_radius_max=float(radius[nearest].max()))
    return result


# ----------------------------------------------------------------------------- stages
def prepare(args, data):
    root = args.root; root.mkdir(parents=True, exist_ok=True)
    marker = root / "prepare.json"
    if marker.exists() and not args.force:
        print("prepare: reusing", marker); return json.loads(marker.read_text())
    rng = np.random.default_rng(args.seed)
    test = np.flatnonzero(data["validation"]); pool = rng.permutation(np.flatnonzero(~data["validation"]))
    sizes = sorted({min(s, len(pool)) for s in args.sizes}); sizes = [s for s in sizes if s >= 128]
    if args.all_size and len(pool) not in sizes: sizes.append(len(pool))
    info = dict(seed=args.seed, n_test=len(test), n_pool=len(pool), sizes=sizes, n_fit={}, pca_retained_variance={},
                moped=dict(), test_indices=test.tolist(), pool_order=pool.tolist())
    theta, noisy, clean, low, high = data["theta"], data["noisy"], data["clean"], data["low"], data["high"]
    (root / "features").mkdir(exist_ok=True)
    for N in sizes:
        idx = pool[:N]; n_fit = int(round((1 - 0.1) * N)); fit = idx[:n_fit]; info["n_fit"][str(N)] = n_fit
        base = fit_asinh(noisy[fit]); y = asinh_coordinates(noisy, base); yc = asinh_coordinates(clean, base)
        pca = fit_pca(y[fit], args.pca_components)
        info["pca_retained_variance"][str(N)] = float(pca["explained_variance_fraction"][:args.pca_components].sum())
        moped = fit_moped_adaptive(y[fit], yc[fit], theta[fit], high - low, args.moped_local_n, args.shrinkage, args.rcond)
        info["moped"][str(N)] = dict(components=int(moped["matrix"].shape[1]), mean_model=moped["mean_model"], local_n=int(moped["local_n"]),
                                    singular_values=[float(v) for v in moped["singular_values"]], local_radius_max=moped["local_radius_max"])
        for method, basis in (("bins40", dict(matrix=np.eye(40), center=np.zeros(40))), ("pca", pca), ("moped", moped)):
            tr = finish_transform(base, y[fit], basis["matrix"], basis["center"])
            with (root / "features" / f"{method}_N{N}_transform.pkl").open("wb") as f: pickle.dump(tr, f)
            np.save(root / "features" / f"{method}_N{N}.npy", project(noisy, tr))
        print(f"prepare: N={N} n_fit={n_fit} PCA var={info['pca_retained_variance'][str(N)]:.4f} MOPED k={info['moped'][str(N)]['components']} ({moped['mean_model']})", flush=True)
    write_json(marker, info); return info


def train(args, data, info):
    import torch
    from sbi.inference import SNPE
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        from sbi.utils.get_nn_models import posterior_nn
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "26")))
    _, prior = prior_objects(args.run, data)
    pool = np.asarray(info["pool_order"])
    for N in (args.sizes_run or info["sizes"]):
        for est in args.estimators:
            run_dir = args.root / args.method / f"N{N}_{est}"; run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "complete.json").exists() and not args.force:
                print("train: reusing", run_dir, flush=True); continue
            h = hyper(N, args.smoke); idx = pool[:N]
            theta = torch.as_tensor(data["theta"][idx], dtype=torch.float32)
            x = torch.as_tensor(np.load(args.root / "features" / f"{args.method}_N{N}.npy")[idx], dtype=torch.float32)
            torch.manual_seed(args.seed + N); np.random.seed(args.seed + N)
            builder = posterior_nn(model=est, hidden_features=h["hidden_features"], num_transforms=h["num_transforms"],
                                   num_bins=h["num_bins"], z_score_x="independent", z_score_theta="independent")
            inference = SNPE(prior=prior, density_estimator=builder, device="cpu", show_progress_bars=False)
            t0 = time.time()
            estimator = inference.append_simulations(theta, x).train(
                training_batch_size=h["training_batch_size"], learning_rate=h["learning_rate"], validation_fraction=h["validation_fraction"],
                stop_after_epochs=h["stop_after_epochs"], max_num_epochs=h["max_num_epochs"], clip_max_norm=h["clip_max_norm"], show_train_summary=False)
            seconds = time.time() - t0
            best = getattr(inference, "_best_model_state_dict", None)
            if best: estimator.load_state_dict(best, strict=True)
            summary = getattr(inference, "summary", None) or getattr(inference, "_summary", {})
            def last(key):
                v = summary.get(key, []); return float(v[-1]) if len(v) else None
            estimator.eval()
            with (run_dir / "estimator.pkl").open("wb") as f: pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)
            torch.save(estimator.state_dict(), run_dir / "state_dict.pt")
            write_json(run_dir / "complete.json", dict(method=args.method, N=int(N), estimator=est, seconds=seconds, hyper=h, x_dim=int(x.shape[1]),
                       best_validation_log_prob=last("best_validation_log_prob"), epochs_trained=last("epochs_trained"),
                       validation_log_probs=[float(v) for v in summary.get("validation_log_probs", [])][-5:], seed=args.seed + N))
            print(f"train: {args.method} N={N} {est}: {seconds:.0f} s, epochs {last('epochs_trained')}, best val logp {last('best_validation_log_prob')}", flush=True)


def sample_rows(estimator, X, n_samples, joint, max_rounds=6, min_accept=200, batch_rows=48):
    """Batched flow sampling with exact rejection on the joint prior support (like sbi's DirectPosterior)."""
    import torch
    X = np.asarray(X, dtype=np.float32); out = [None] * len(X); acceptance = np.zeros(len(X)); pending = list(range(len(X)))
    collected = {i: [] for i in range(len(X))}; proposals = np.zeros(len(X)); accepted = np.zeros(len(X))
    for rnd in range(max_rounds):
        if not pending: break
        k = int(min(2 * n_samples * 2 ** rnd, 40000))
        for start in range(0, len(pending), batch_rows):
            rows = pending[start:start + batch_rows]
            with torch.no_grad():
                raw = estimator.sample(k, context=torch.as_tensor(X[rows])).cpu().numpy().reshape(len(rows), k, 9)
            ok = joint.contains(raw.reshape(-1, 9)).reshape(len(rows), k)
            for r, i in enumerate(rows):
                collected[i].append(raw[r][ok[r]]); proposals[i] += k; accepted[i] += ok[r].sum()
        pending = [i for i in pending if accepted[i] < n_samples]
    for i in range(len(X)):
        s = np.concatenate(collected[i]) if collected[i] else np.empty((0, 9))
        out[i] = s[:n_samples]; acceptance[i] = accepted[i] / max(proposals[i], 1)
    min_accept = min(min_accept, max(n_samples // 2, 1))
    short = [i for i in range(len(X)) if len(out[i]) < min_accept]
    return out, acceptance, short


def bootstrap_sem(fn, *arrays, n=200, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        j = rng.integers(0, len(arrays[0]), len(arrays[0])); vals.append(fn(*[a[j] for a in arrays]))
    return np.std(np.asarray(vals), axis=0, ddof=1)


def evaluate(args, data, info):
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "26")))
    joint, _ = prior_objects(args.run, data)
    test = np.asarray(info["test_indices"])
    if args.max_test_rows: test = test[:args.max_test_rows]
    truth = data["theta"][test]; low, high = data["low"], data["high"]; width = high - low
    for N in (args.sizes_run or info["sizes"]):
        for est in args.estimators:
            run_dir = args.root / args.method / f"N{N}_{est}"
            if not (run_dir / "complete.json").exists(): print("evaluate: missing", run_dir); continue
            if (run_dir / "metrics.json").exists() and not args.force: print("evaluate: reusing", run_dir); continue
            with (run_dir / "estimator.pkl").open("rb") as f: estimator = pickle.load(f)
            estimator.eval(); X = np.load(args.root / "features" / f"{args.method}_N{N}.npy")[test]
            torch.manual_seed(1000 + N); t0 = time.time()
            samples, acc, short = sample_rows(estimator, X, args.posterior_samples, joint, max_rounds=6)
            means = np.array([s.mean(0) if len(s) else np.full(9, np.nan) for s in samples])
            stds = np.array([s.std(0, ddof=1) if len(s) > 1 else np.full(9, np.nan) for s in samples])
            good = np.isfinite(means).all(1) & (np.array([len(s) for s in samples]) >= min(200, max(args.posterior_samples // 2, 1)))
            err = (means[good] - truth[good]) / width
            rmse = np.sqrt((err ** 2).mean(0)); rmse_sem = bootstrap_sem(lambda e: np.sqrt((e ** 2).mean(0)), err)
            pear = pearson_columns(truth[good], means[good]); pear_sem = bootstrap_sem(lambda t, m: pearson_columns(t, m), truth[good], means[good])
            zt = ((truth[good] - low) / width).ravel(); zm = ((means[good] - low) / width).ravel()
            pooled = float(np.corrcoef(zt, zm)[0, 1])
            pooled_sem = float(bootstrap_sem(lambda t, m: np.corrcoef(((t - low) / width).ravel(), ((m - low) / width).ravel())[0, 1], truth[good], means[good]))
            metrics = dict(method=args.method, N=int(N), estimator=est, n_test=int(good.sum()), n_short=len(short), seconds=time.time() - t0,
                           acceptance_median=float(np.median(acc)), rmse_over_prior_range=rmse.tolist(), rmse_sem=rmse_sem.tolist(),
                           mean_rmse_over_prior_range=float(rmse.mean()), mean_rmse_sem=float(np.sqrt((rmse_sem ** 2).sum()) / 9),
                           pearson_r=pear.tolist(), pearson_sem=pear_sem.tolist(), pooled_pearson_r=pooled, pooled_pearson_sem=pooled_sem,
                           mean_posterior_std_over_prior_range=(np.nanmean(stds[good], 0) / width).tolist(),
                           coverage68=[float(np.mean(np.abs(means[good][:, k] - truth[good][:, k]) <= stds[good][:, k])) for k in range(9)])
            np.savez_compressed(run_dir / "posterior_summary.npz", means=means, stds=stds, truth=truth, acceptance=acc, test_indices=test)
            write_json(run_dir / "metrics.json", metrics)
            print(f"evaluate: {args.method} N={N} {est}: mean RMSE/range {rmse.mean():.4f}, pooled r {pooled:.3f}, acc {np.median(acc):.2f}, short {len(short)}, {time.time() - t0:.0f} s", flush=True)


def collect_metrics(root):
    rows = []
    for method in METHODS:
        for run_dir in sorted((root / method).glob("N*_*")) if (root / method).exists() else []:
            m, c = run_dir / "metrics.json", run_dir / "complete.json"
            if m.exists() and c.exists():
                row = json.loads(m.read_text()); row["best_validation_log_prob"] = json.loads(c.read_text())["best_validation_log_prob"]; rows.append(row)
    return rows


def best_estimator(rows, method, N):
    cands = [r for r in rows if r["method"] == method and r["N"] == N and r["best_validation_log_prob"] is not None]
    return max(cands, key=lambda r: r["best_validation_log_prob"])["estimator"] if cands else None


def style():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
                         "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False, "axes.titlesize": 11})
    return plt


def summarize(args, data, info):
    import csv
    plt = style(); rows = collect_metrics(args.root); out = args.root / "figures"; out.mkdir(exist_ok=True)
    with (out / "convergence_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "N", "estimator", "best_val_logp", "n_test", "mean_rmse_over_prior_range", "mean_rmse_sem", "pooled_pearson_r", "pooled_pearson_sem"]
                                       + [f"rmse_{n}" for n in NAMES] + [f"r_{n}" for n in NAMES])
        for r in sorted(rows, key=lambda r: (r["method"], r["N"], r["estimator"])):
            w.writerow([r["method"], r["N"], r["estimator"], r["best_validation_log_prob"], r["n_test"], r["mean_rmse_over_prior_range"], r["mean_rmse_sem"], r["pooled_pearson_r"], r["pooled_pearson_sem"]]
                       + r["rmse_over_prior_range"] + r["pearson_r"])
    sizes = sorted({r["N"] for r in rows}); choice = {(m, N): best_estimator(rows, m, N) for m in METHODS for N in sizes}
    write_json(out / "best_estimator_per_method_and_size.json", {f"{m}_N{N}": e for (m, N), e in choice.items()})
    SEM_KEY = {"mean_rmse_over_prior_range": "mean_rmse_sem", "pooled_pearson_r": "pooled_pearson_sem",
               "rmse_over_prior_range": "rmse_sem", "pearson_r": "pearson_sem"}
    def series(method, key, est=None, k=None):
        xs, ys, es = [], [], []
        for N in sizes:
            e = est or choice[(method, N)]; r = [r for r in rows if r["method"] == method and r["N"] == N and r["estimator"] == e]
            if not r: continue
            r = r[0]; xs.append(N)
            if k is None: ys.append(r[key]); es.append(r[SEM_KEY[key]])
            else: ys.append(r[key][k]); es.append(r[SEM_KEY[key]][k])
        return np.array(xs), np.array(ys), np.array(es)
    for kind, key_all, key_par, ylabel, fname, ylim in (
            ("rmse", "mean_rmse_over_prior_range", "rmse_over_prior_range", "RMSE of posterior mean / prior range", "convergence_rmse", None),
            ("pearson", "pooled_pearson_r", "pearson_r", r"Pearson $r$(truth, posterior mean)", "convergence_pearson", (0, 1))):
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.2), sharex=True); axes = axes.ravel()
        for p, ax in enumerate(axes):
            for method in METHODS:
                if p == 0: xs, ys, es = series(method, key_all)
                else: xs, ys, es = series(method, key_par, k=p - 1)
                if len(xs) == 0: continue
                ax.errorbar(xs, ys, yerr=es, color=METHOD_COLOR[method], marker="o", ms=4.5, lw=1.9, capsize=2.5, label=METHOD_LABEL[method])
                for est, ls in (("maf", ":"), ("nsf", "--")):
                    xs2, ys2, _ = series(method, key_all if p == 0 else key_par, est=est, k=None if p == 0 else p - 1)
                    if len(xs2): ax.plot(xs2, ys2, color=METHOD_COLOR[method], lw=0.9, ls=ls, alpha=0.55)
            ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes], rotation=45, fontsize=8.5)
            ax.set_title("all parameters (pooled)" if p == 0 else "$" + LATEX[NAMES[p - 1]] + "$", fontsize=11.5)
            if ylim: ax.set_ylim(*ylim)
            if p % 5 == 0: ax.set_ylabel(ylabel, fontsize=10)
            if p >= 5: ax.set_xlabel("training set size $N$")
        axes[0].legend(fontsize=9.5, loc="best")
        fig.suptitle(("RMSE" if kind == "rmse" else "Pearson correlation") + " vs training-set size, 843 held-out rows, independent SO noise. "
                     "Solid: best density estimator per point (by validation log-prob); dotted MAF, dashed NSF.", fontsize=10.5, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out / f"{fname}.png", dpi=170); fig.savefig(out / f"{fname}.pdf"); plt.close(fig)
    print("summarize: wrote", out)


def load_observations(args):
    obs = {}
    for name in OBS_NAMES:
        for d in (args.run / "observations", args.root / "observations", args.root.parent / "observations"):
            p = d / f"{name}.npz"
            if p.exists(): obs[name] = np.load(p)["masked_noisy_cross_dl40"]; break
    return obs


def truth_markers(args, name):
    if name == "HalfDome": return dict(zip(NAMES, FIDUCIAL)), "true values"
    import csv
    p = args.run / "plots" / "prior_and_fits.csv"
    if name in ("L1_m9", "fgas-8sigma", "Mstar-1sigma") and p.exists():
        rows = {r["parameter"]: float(r[name]) for r in csv.DictReader(p.open())}
        return {n: rows[n] for n in NAMES if n in rows}, "effective clean-spectrum fit (not a truth)"
    return None, "no reference values (outside the nine-parameter family)"


def corner(args, data, info):
    from getdist import MCSamples, plots
    import matplotlib; matplotlib.use("Agg")
    joint, _ = prior_objects(args.run, data); rows = collect_metrics(args.root)
    N = args.corner_size or max(info["sizes"]); out = args.root / "figures"; out.mkdir(exist_ok=True); obs = load_observations(args)
    low, high = data["low"], data["high"]; ranges = {n: [float(low[i]), float(high[i])] for i, n in enumerate(NAMES)}
    labels = [LATEX[n] for n in NAMES]
    prior_samples = MCSamples(samples=data["theta"], names=NAMES, labels=labels, ranges=ranges, label="prior", settings={"smooth_scale_1D": 0.35})
    results = {}
    for name, x40 in obs.items():
        sets, legend = [], []
        for method in METHODS:
            est = best_estimator(rows, method, N)
            if est is None: continue
            run_dir = args.root / method / f"N{N}_{est}"
            with (args.root / "features" / f"{method}_N{N}_transform.pkl").open("rb") as f: tr = pickle.load(f)
            with (run_dir / "estimator.pkl").open("rb") as f: estimator = pickle.load(f)
            estimator.eval(); feat = project(x40.reshape(1, -1), tr)
            s, acc, short = sample_rows(estimator, feat, args.corner_samples, joint, max_rounds=14, min_accept=1000)
            s = s[0]
            if len(s) < 300:
                print(f"corner: {name} {method}: only {len(s)} accepted samples (acceptance {acc[0]:.4f}); skipped", flush=True)
                results[f"{name}/{method}"] = dict(estimator=est, samples=len(s), acceptance=float(acc[0]), skipped=True); continue
            results[f"{name}/{method}"] = dict(estimator=est, samples=len(s), acceptance=float(acc[0]), mean=s.mean(0).tolist(), std=s.std(0).tolist())
            np.save(out / f"posterior_{name}_{method}.npy", s)
            sets.append(MCSamples(samples=s, names=NAMES, labels=labels, ranges=ranges, label=f"{METHOD_LABEL[method]} ({est.upper()})"))
            legend.append(f"{METHOD_LABEL[method]}, {est.upper()}")
        if not sets: continue
        markers, note = truth_markers(args, name)
        g = plots.get_subplot_plotter(width_inch=12)
        g.settings.axes_fontsize = 9; g.settings.axes_labelsize = 13; g.settings.legend_fontsize = 11.5; g.settings.alpha_filled_add = 0.55
        g.settings.figure_legend_frame = False; g.settings.title_limit_fontsize = 9
        g.triangle_plot(sets, filled=True, legend_labels=legend, legend_loc="upper right",
                        contour_colors=[METHOD_COLOR[m] for m in METHODS][:len(sets)], markers=markers, marker_args={"lw": 1.3, "color": "k", "ls": "--"})
        for i, n in enumerate(NAMES):
            ax = g.subplots[i, i]; d = prior_samples.get1DDensity(n)
            if d is not None: ax.fill_between(d.x, 0, d.P / d.P.max(), color="#9a9891", alpha=0.22, lw=0, zorder=0)
        g.fig.text(0.36, 0.90, f"{OBS_TITLE.get(name, name)}\nposteriors from the N={N} training set, independent SO noise\n"
                   f"grey on the diagonal: prior marginals\ndashed lines: {note}", fontsize=12.5, ha="left", va="top", linespacing=1.6)
        g.export(str(out / f"corner_{name}.png")); g.export(str(out / f"corner_{name}.pdf"))
        print("corner:", name, {k.split('/')[1]: (v['estimator'], v['samples'], round(v['acceptance'], 3)) for k, v in results.items() if k.startswith(name + '/')}, flush=True)
    write_json(out / "corner_posterior_summary.json", results)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("prepare", "train", "evaluate", "summarize", "corner", "all"))
    ap.add_argument("--root", type=Path, default=Path("/lustre/work/kristero10/sbi_linear_8k_20260918/analysis"))
    ap.add_argument("--run", type=Path, default=RUN_DEFAULT)
    ap.add_argument("--method", choices=METHODS, default="bins40")
    ap.add_argument("--estimators", default="maf,nsf"); ap.add_argument("--sizes", default="256,512,1024,2048,4096")
    ap.add_argument("--sizes-run", default="", help="subset of sizes for train/evaluate (default all prepared)")
    ap.add_argument("--all-size", action="store_true", default=True); ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--pca-components", type=int, default=9); ap.add_argument("--moped-local-n", type=int, default=2048)
    ap.add_argument("--shrinkage", type=float, default=0.05); ap.add_argument("--rcond", type=float, default=1e-6)
    ap.add_argument("--posterior-samples", type=int, default=1000); ap.add_argument("--corner-samples", type=int, default=10000)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-test-rows", type=int, default=0); ap.add_argument("--corner-size", type=int, default=0)
    args = ap.parse_args()
    args.estimators = tuple(e for e in args.estimators.split(",") if e); args.sizes = [int(s) for s in args.sizes.split(",") if s]
    args.sizes_run = [int(s) for s in args.sizes_run.split(",") if s] or None
    data = load_run(args.run)
    info = prepare(args, data) if args.stage in ("prepare", "all") else json.loads((args.root / "prepare.json").read_text())
    if args.stage in ("train", "all"): train(args, data, info)
    if args.stage in ("evaluate", "all"): evaluate(args, data, info)
    if args.stage == "summarize": summarize(args, data, info)
    if args.stage == "corner": corner(args, data, info)


if __name__ == "__main__":
    main()
