"""Numerics for the matched independent-noise nine-parameter forecast."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.covariance import oas

from export_so_moped_bundle import sha256
from generate_so_fisher_variations import BATTAGLIA12, PARAMETER_NAMES


def simulation_signature(generation):
    """Exclude machine paths and design size, retain physical/source settings."""
    config=dict(generation["config"])
    for key in ("n_rows","sequence_offset","design_dir","noise_seed_bases","noise_seed_stride"):
        config.pop(key,None)
    payload=dict(config=config,source_sha256=generation["source_sha256"],
                 catalogue_size=generation["catalogue"]["size"])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def load_independent_dataset(path, generation_path, expected_rows=32768):
    """Read the collaborator's completed nine_param/prepared/dataset.npz."""
    path=Path(path)
    complete=json.loads(path.with_name("complete.json").read_text())
    require(complete["dataset_sha256"] == sha256(path),"Dataset checksum differs from completion marker")
    generation=json.loads(Path(generation_path).read_text())
    data=read_npz(path)
    metadata=json.loads(str(data["metadata_json"]))
    require(metadata == complete["metadata"],"Dataset metadata differs from completion marker")
    require(metadata["complete"] and metadata["independent_noise_all_rows"] and metadata["same_mask_all_rows"],
            "Require completed independent noise with the same mask")
    require(metadata["mode"] == "nine_param" and metadata["product"] == "masked_baseline_noise_cross_deproj0",
            "Expected all-nine-parameter baseline deproj0 dataset")
    require(metadata["experiment_id"] == generation["experiment_id"],"Wrong generation manifest")
    require(metadata["noise_table_convention"] == "N_ell per split","Noise convention differs")
    require(metadata["bin_weighting"] == "2ell_plus_1" and metadata["beam_fwhm_arcmin"] == 2.0,
            "Unexpected statistic/beam")
    np.testing.assert_array_equal(data["param_names"],PARAMETER_NAMES)
    np.testing.assert_array_equal(data["full_param_names"],PARAMETER_NAMES)
    np.testing.assert_array_equal(data["theta"],data["theta_full"])
    n=len(data["theta"])
    require(n == expected_rows and data["theta"].shape == (n,9),f"Expected {expected_rows} rows and nine parameters")
    require(data["x"].shape == data["x_no_noise"].shape == (n,40),"Expected paired 40-bin spectra")
    low,high=data["prior_low"],data["prior_high"]
    require(low.shape == high.shape == (9,) and np.all(high>low),"Invalid saved inference bounds")
    for key in ("theta","x","x_no_noise","prior_low","prior_high"):
        require(np.isfinite(data[key]).all(),f"Nonfinite {key}")
    require(np.all((data["theta"]>=low)&(data["theta"]<=high)),"Out-of-prior rows; bounds must not be widened")
    require(np.all((BATTAGLIA12>low)&(BATTAGLIA12<high)),"Battaglia12 outside saved prior")
    cfg=generation["config"]
    np.testing.assert_array_equal(low,cfg["prior_low"])
    np.testing.assert_array_equal(high,cfg["prior_high"])
    np.testing.assert_array_equal(data["sobol_global_row"],cfg["sequence_offset"]+np.arange(1,n+1))
    base=cfg["noise_seed_bases"]["nine_param"]
    roots=base+cfg["noise_seed_stride"]*(data["sobol_global_row"]-1)
    expected=np.column_stack((roots+10101,roots+10102))
    np.testing.assert_array_equal(data["noise_seed"],roots)
    np.testing.assert_array_equal(data["noise_split_seeds"],expected)
    require(len(np.unique(expected))==2*n,"Noise split seeds overlap")
    np.testing.assert_array_equal(data["mask_seed"],np.full(n,cfg["mask_seed"]))
    ell=np.arange(cfg["ell_min"],cfg["ell_max"]+1)
    groups=ell//cfg["delta_ell"]
    unique=np.unique(groups)
    np.testing.assert_array_equal(data["ell_unbinned"],ell)
    np.testing.assert_array_equal(data["bin_ell_min"],[ell[groups==g].min() for g in unique])
    np.testing.assert_array_equal(data["bin_ell_max"],[ell[groups==g].max() for g in unique])
    np.testing.assert_allclose(data["ell_binned"],[np.average(ell[groups==g],weights=2*ell[groups==g]+1) for g in unique],rtol=1e-6)
    return data,generation


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def conditional_covariance(spectra):
    """OAS correlation shrinkage, retaining unbiased empirical bin variances.

    Do not apply the raw-sample-covariance Hartlap factor to this estimator.
    """
    require(spectra.ndim == 2 and len(spectra) >= 3 and np.isfinite(spectra).all(), "Invalid ensemble")
    std = spectra.std(axis=0, ddof=1)
    require(np.all(std > 0), "Zero ensemble variance")
    correlation, shrinkage = oas((spectra-spectra.mean(axis=0))/std, assume_centered=True)
    correlation /= np.sqrt(np.outer(np.diag(correlation), np.diag(correlation)))
    covariance = correlation*np.outer(std,std)
    np.linalg.cholesky(covariance)
    return covariance, float(shrinkage)


def fisher_matrix(jacobian, covariance):
    white = solve_triangular(np.linalg.cholesky(covariance), jacobian, lower=True)
    return white.T@white


def fisher_modes(fisher, rcond=1e-8):
    values, vectors = np.linalg.eigh((fisher+fisher.T)/2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:,order]
    return values, vectors, values > values[0]*rcond


def moped_weights(jacobian, covariance):
    """SVD rotation of MOPED retaining nine local mean-sensitivity coordinates."""
    chol = np.linalg.cholesky(covariance)
    white = solve_triangular(chol,jacobian,lower=True)
    left, singular, _ = np.linalg.svd(white,full_matrices=False)
    weights = solve_triangular(chol.T,left,lower=False)
    full = white.T@white
    compressed = fisher_matrix(weights.T@jacobian,weights.T@covariance@weights)
    error = np.linalg.norm(full-compressed)/np.linalg.norm(full)
    require(error < 1e-8,"MOPED does not preserve the fitted local Fisher matrix")
    return weights, singular, float(error)


def hard_prior_posterior(jacobian, covariance, residual, low, high, truth,
                         draws=1000000, count=100000, seed=42):
    """Sample the local Gaussian likelihood multiplied by the HARD uniform prior.

    Work in prior-width coordinates u. The importance proposal is likelihood
    times a Gaussian penalty about the box midpoint, with precision 12 I.
    Importance weights cancel the Gaussian penalty exactly. Weak Fisher modes
    are never assigned finite data-only errors by eigenvalue flooring.
    """
    width=high-low
    chol=np.linalg.cholesky(covariance)
    a=solve_triangular(chol,jacobian*width,lower=True)
    y=solve_triangular(chol,residual,lower=True)
    midpoint=((low+high)/2-truth)/width
    precision=a.T@a+12*np.eye(len(width))
    proposal_chol=np.linalg.cholesky(precision)
    mean=np.linalg.solve(precision,a.T@y+12*midpoint)
    rng=np.random.default_rng(seed)
    parts=[]
    for start in range(0,draws,100000):
        standard=rng.normal(size=(min(100000,draws-start),len(width)))
        u=mean+solve_triangular(proposal_chol.T,standard.T,lower=False).T
        inside=np.all((u >= (low-truth)/width)&(u <= (high-truth)/width),axis=1)
        parts.append(u[inside])
    candidates=np.concatenate(parts)
    require(len(candidates)>0,"No in-prior Fisher proposals")
    log_weights=6*np.sum((candidates-midpoint)**2,axis=1)
    weights=np.exp(log_weights-log_weights.max())
    weights/=weights.sum()
    ess=float(1/np.sum(weights**2))
    require(ess > min(5000,draws*.02),f"Insufficient importance ESS: {ess:.1f}")
    weighted_mean=weights@candidates
    weighted_std=np.sqrt(weights@(candidates-weighted_mean)**2)
    cumulative=np.cumsum(weights)
    cumulative[-1]=1
    positions=(np.arange(count)+rng.random())/count
    samples=truth+candidates[np.searchsorted(cumulative,positions)]*width
    return samples,dict(importance_ess=ess,proposals=draws,inside=len(candidates),
        weighted_mean=(truth+weighted_mean*width).tolist(),weighted_std=(weighted_std*width).tolist(),
        prior="hard uniform box; Gaussian proposal penalty exactly cancelled")
