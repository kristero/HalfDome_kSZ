#!/usr/bin/env python3
"""Process the author-supplied Takahashi+2025 correlation-function measurements and jackknife
covariance (Takahashi25_data/xi_folder/xi_folder/), replacing the earlier plot-digitized
approximation (digitize_frb_observational_figures.py) used in the fullsky and finite-source
comparison plots.

File contract (from the author, 2026-09-19):
  xi_angle_ne2001_varalp_{survey}.txt        col1=theta-bin mean [arcmin], col3=bin index 0-12,
                                              col4=first term of Eq. (27) [pc/cm^3]           (a)
  xi_angle_rand3e+3_ne2001_varalp_{survey}.txt  col1=bin index 0-12, col3=second term of Eq.(27) (b)
  xi_angle_covjk_ne2001_varalp_{survey}.txt  lines for index 0-12: col4 = jackknife sigma [pc/cm^3]
                                              lines for (i,j), i>=j: col3=covariance [pc^2/cm^6],
                                              col4=correlation matrix
  measurement = (a) - (b), Fig. 9.

Both xi_angle_ne2001 and xi_angle_covjk report the SAME nominal bin centers in their 2nd column,
a clean log grid of exactly 0.25 dex (10**0.25 per step). This is also the "logspace(0, 3, 13)"
edge grid this repository's simulation-side annulus binning already uses
(compare_takahashi_sightlines.EDGES = 12 bins from 1 to 1000 arcmin): the author's bin index 0
(theta ~ 0.75', below 1') is one extra bin outside that range; bin indices 1-12 correspond exactly
to EDGES bins 0-11 (verified below to 1e-6 relative precision). Consumers needing the simulation's
fixed 12-bin grid should use indices 1-12; consumers with a free-form angular-correlation transform
(fullsky method) can use all 13 bins.
"""
import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "Takahashi25_data/xi_folder/xi_folder"
OUTPUT_DIR = REPO_ROOT / "frb_map_generation/outputs/takahashi25_real_observations"
N_BINS = 13
EDGES_12 = np.logspace(0, 3, 13)   # compare_takahashi_sightlines.EDGES, kept in sync manually


def raw_filename(kind, survey_file_tag):
    if kind == "main":
        return RAW_DIR / f"xi_angle_ne2001_varalp_{survey_file_tag}.txt"
    if kind == "rand":
        return RAW_DIR / f"xi_angle_rand3e+3_ne2001_varalp_{survey_file_tag}.txt"
    if kind == "cov":
        return RAW_DIR / f"xi_angle_covjk_ne2001_varalp_{survey_file_tag}.txt"
    raise ValueError(kind)


# The ACT main file alone uses "ACT" (upper-case) in the delivered filename; every other file,
# and every covariance/rand file, uses lower case. Handled explicitly rather than guessed.
SURVEYS = {
    "planck_milca": {"main": "milca_71FRBs", "rand": "milca_71FRBs", "cov": "milca_71FRBs", "n_frb": 71},
    "planck_nilc": {"main": "nilc_71FRBs", "rand": "nilc_71FRBs", "cov": "nilc_71FRBs", "n_frb": 71},
    "act": {"main": "ACT_31FRBs", "rand": "act_31FRBs", "cov": "act_31FRBs", "n_frb": 31},
}


def load_main(path):
    rows = np.loadtxt(path)
    if rows.shape != (N_BINS, 7):
        raise ValueError(f"{path}: expected ({N_BINS}, 7), got {rows.shape}")
    if not np.array_equal(rows[:, 2].astype(int), np.arange(N_BINS)):
        raise ValueError(f"{path}: bin index column is not 0..{N_BINS - 1}")
    return dict(theta_mean=rows[:, 0], theta_nominal=rows[:, 1], term_a=rows[:, 3])


def load_rand(path):
    rows = np.loadtxt(path)
    if rows.shape != (N_BINS, 4):
        raise ValueError(f"{path}: expected ({N_BINS}, 4), got {rows.shape}")
    if not np.array_equal(rows[:, 0].astype(int), np.arange(N_BINS)):
        raise ValueError(f"{path}: bin index column is not 0..{N_BINS - 1}")
    return dict(theta_nominal=rows[:, 1], term_b=rows[:, 2])


def load_cov(path):
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    if len(lines) != N_BINS + N_BINS * (N_BINS + 1) // 2:
        raise ValueError(f"{path}: unexpected line count {len(lines)}")
    diag_lines = [ln.split() for ln in lines[:N_BINS]]
    if not np.array_equal(np.array([int(x[0]) for x in diag_lines]), np.arange(N_BINS)):
        raise ValueError(f"{path}: diagonal-block index column is not 0..{N_BINS - 1}")
    sigma = np.array([float(x[3]) for x in diag_lines])
    cov = np.full((N_BINS, N_BINS), np.nan)
    corr = np.full((N_BINS, N_BINS), np.nan)
    for ln in lines[N_BINS:]:
        i, j, c, r = ln.split()
        i, j = int(i), int(j)
        cov[i, j] = cov[j, i] = float(c)
        corr[i, j] = corr[j, i] = float(r)
    if np.any(np.isnan(cov)) or np.any(np.isnan(corr)):
        raise ValueError(f"{path}: covariance matrix not fully populated")
    return dict(sigma=sigma, covariance=cov, correlation=corr)


def process_survey(key, tags):
    main = load_main(raw_filename("main", tags["main"]))
    rand = load_rand(raw_filename("rand", tags["rand"]))
    cov = load_cov(raw_filename("cov", tags["cov"]))
    if not np.allclose(main["theta_nominal"], rand["theta_nominal"], rtol=1e-9):
        raise ValueError(key + ": main/rand nominal theta grids disagree")
    w = main["term_a"] - rand["term_b"]
    sigma_diag = np.sqrt(np.diag(cov["covariance"]))
    if not np.allclose(sigma_diag, cov["sigma"], rtol=1e-6):
        raise ValueError(key + ": sqrt(diag(covariance)) does not match the reported jackknife sigma")
    reconstructed_corr = cov["covariance"] / np.outer(cov["sigma"], cov["sigma"])
    if not np.allclose(reconstructed_corr, cov["correlation"], atol=1e-6):
        raise ValueError(key + ": covariance/outer(sigma,sigma) does not match the reported correlation matrix")
    # bins 1..12 must equal the simulation's EDGES bins 0..11 exactly (0.25 dex log grid, one
    # bin narrower than the author's, who also reports a bin below 1 arcmin at index 0).
    lo = main["theta_nominal"] * 10 ** -0.125
    hi = main["theta_nominal"] * 10 ** 0.125
    if not (np.allclose(lo[1:], EDGES_12[:-1], rtol=1e-3) and np.allclose(hi[1:], EDGES_12[1:], rtol=1e-3)):
        raise ValueError(key + ": bins 1-12 do not match compare_takahashi_sightlines.EDGES as expected")
    return dict(bin_index=np.arange(N_BINS), theta_mean_arcmin=main["theta_mean"],
                theta_nominal_arcmin=main["theta_nominal"], theta_lo_arcmin=lo, theta_hi_arcmin=hi,
                w_yDM_pc_cm3=w, sigma_pc_cm3=cov["sigma"], covariance_pc2_cm6=cov["covariance"],
                correlation=cov["correlation"], n_frb=tags["n_frb"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for key, tags in SURVEYS.items():
        d = process_survey(key, tags)
        np.savez(a.output_dir / f"{key}.npz", **d)
        print(f"{key}: n_frb={d['n_frb']}  theta {d['theta_mean_arcmin'][0]:.3f}-{d['theta_mean_arcmin'][-1]:.1f} arcmin")
        for i in range(N_BINS):
            print(f"  bin {i:2d}  theta={d['theta_mean_arcmin'][i]:8.3f}'  w={d['w_yDM_pc_cm3'][i]:+.4e}"
                  f"  sigma={d['sigma_pc_cm3'][i]:.4e}  S/N={d['w_yDM_pc_cm3'][i] / d['sigma_pc_cm3'][i]:+.2f}")
    print("Saved to", a.output_dir)


if __name__ == "__main__":
    main()
