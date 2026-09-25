"""Propagate ONE conditional bandpower covariance through linear compression.

PCA's training covariance chooses its axes; it never replaces the conditional
noise covariance in the likelihood. Matrices have shape (input, output), so
row observations transform as x @ matrix.
"""
import numpy as np
from scipy.linalg import solve_triangular

from so_nine_fisher import fisher_matrix, require


def fit_linear_pca(training, components=9):
    """PCA of standardized raw D_ell, fitted on optimization rows only.

    Using a linear map makes covariance propagation exact. This is a new
    transform, distinct from the legacy PCA after nonlinear asinh scaling.
    """
    training = np.asarray(training, dtype=np.float64)
    require(training.ndim == 2 and np.isfinite(training).all(), "Invalid PCA training rows")
    require(1 <= components <= training.shape[1], "Invalid PCA dimension")
    center = training.mean(axis=0)
    scale = training.std(axis=0, ddof=1)
    require(np.all(scale > 0), "Constant PCA input coordinate")
    standardized = (training - center) / scale
    training_covariance = np.cov(standardized, rowvar=False, ddof=1)
    eigenvalues, vectors = np.linalg.eigh(training_covariance)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    matrix = vectors[:, :components] / scale[:, None]
    projected = (training - center) @ matrix
    return dict(kind=np.asarray("pca_raw"), matrix=matrix, fiducial_dell=center,
                mean=projected.mean(axis=0), std=projected.std(axis=0, ddof=1),
                input_scale=scale, training_covariance=training_covariance,
                eigenvalues=eigenvalues,
                explained_variance_fraction=eigenvalues / eigenvalues.sum())


def project_likelihood(jacobian, covariance, residual, matrix):
    """Exact linear propagation of a frozen-covariance local Gaussian model.

    Return both the Fisher matrix and observed score. Equal Fisher matrices
    alone do not prove equal posteriors for a particular observation.
    """
    jacobian, covariance, residual, matrix = [np.asarray(x, dtype=np.float64)
                                             for x in (jacobian, covariance, residual, matrix)]
    require(covariance.shape == (jacobian.shape[0],) * 2, "Covariance/Jacobian mismatch")
    require(residual.shape == (len(covariance),) and matrix.shape[0] == len(covariance),
            "Observation/projection mismatch")
    require(all(np.isfinite(x).all() for x in (jacobian, covariance, residual, matrix)),
            "Nonfinite likelihood inputs")
    compressed_covariance = matrix.T @ covariance @ matrix
    compressed_jacobian = matrix.T @ jacobian
    compressed_residual = matrix.T @ residual
    chol = np.linalg.cholesky(compressed_covariance)
    white_j = solve_triangular(chol, compressed_jacobian, lower=True)
    white_r = solve_triangular(chol, compressed_residual, lower=True)
    full_fisher = fisher_matrix(jacobian, covariance)
    compressed_fisher = white_j.T @ white_j
    difference = (full_fisher - compressed_fisher)
    minimum_loss = np.linalg.eigvalsh((difference + difference.T) / 2).min()
    require(minimum_loss >= -1e-8 * max(np.linalg.norm(full_fisher), 1e-300),
            "Compression spuriously increases local Fisher information")
    return dict(matrix=matrix, covariance=compressed_covariance,
                derivatives=compressed_jacobian, residual=compressed_residual,
                fisher=compressed_fisher, score=white_j.T @ white_r,
                relative_fisher_change=np.asarray(np.linalg.norm(difference) /
                                                  max(np.linalg.norm(full_fisher), 1e-300)),
                minimum_information_loss_eigenvalue=np.asarray(minimum_loss))
