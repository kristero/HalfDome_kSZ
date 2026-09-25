"""Memory-bounded OAS covariance and regression-MOPED for individual multipoles.

The covariance is identical to OAS().fit(residuals).covariance_ plus the stated
ridge, but its inverse acts through an n_rows by n_rows system. No 7900 by 7900
matrix is constructed. This is an approximate local, fixed-covariance MOPED;
the regression singular values do not establish physical identifiability.
"""
import numpy as np


def oas_inverse_action(residuals, vectors):
    """Return C^-1 vectors and a callable for C times vectors."""
    centered = np.asarray(residuals, dtype=np.float64)
    centered = centered - centered.mean(axis=0)
    n, features = centered.shape
    gram = centered @ centered.T / n
    mu = np.trace(gram) / features
    alpha = np.sum(gram * gram) / features**2
    numerator = alpha + mu**2
    denominator = (n + 1) * (alpha - mu**2 / features)
    shrinkage = 1.0 if denominator <= 0 else min(numerator / denominator, 1.0)
    ridge = max(mu, 1e-12) * 1e-8
    diagonal = shrinkage * mu + ridge
    coefficient = (1.0 - shrinkage) / n
    small_system = np.eye(n) + coefficient / diagonal * (centered @ centered.T)
    solved = vectors / diagonal - coefficient / diagonal**2 * (
        centered.T @ np.linalg.solve(small_system, centered @ vectors))

    def covariance_action(value):
        return diagonal * value + coefficient * centered.T @ (centered @ value)

    return solved, covariance_action, dict(
        oas_shrinkage=float(shrinkage), covariance_ridge=float(ridge),
        covariance_rows=n, input_features=features,
        covariance_solver='Exact OAS plus ridge via low-rank Woodbury solve')


def regression_moped(clean, noisy, theta, low, high, rcond=1e-3):
    """Fit on supplied optimization rows only; inputs are per-ell coordinates."""
    fiducial = np.array([18.1, .497, 4.35, .154, -.00865, .0393, -.758, .731, .415])
    offset = (theta - fiducial) / (high - low)
    selected = np.argsort(np.linalg.norm(offset, axis=1))[:min(96, len(theta))]
    design = np.c_[np.ones(len(selected)), offset[selected]]
    coefficients, _, rank, _ = np.linalg.lstsq(design, clean[selected], rcond=None)
    if rank != design.shape[1]:
        raise RuntimeError('Local MOPED regression design is rank deficient')
    derivatives = coefficients[1:].T
    inverse_derivatives, covariance_action, metadata = oas_inverse_action(
        noisy[selected] - clean[selected], derivatives)
    fisher = derivatives.T @ inverse_derivatives
    eigenvalues, directions = np.linalg.eigh((fisher + fisher.T) / 2)
    eigenvalues, directions = eigenvalues[::-1], directions[:, ::-1]
    singular = np.sqrt(np.maximum(eigenvalues, 0))
    keep = singular > max(singular[0] * rcond, 1e-14)
    if not np.any(keep):
        raise RuntimeError('No numerically supported compression direction')
    matrix = (inverse_derivatives @ directions[:, keep]) / singular[keep]
    compressed_covariance = matrix.T @ covariance_action(matrix)
    identity_error = np.linalg.norm(compressed_covariance - np.eye(keep.sum()))
    if identity_error > 1e-6:
        raise RuntimeError(f'MOPED covariance identity failed: {identity_error:g}')
    projected_derivatives = matrix.T @ derivatives
    fisher_error = np.linalg.norm(fisher - projected_derivatives.T @ projected_derivatives)
    fisher_error /= max(np.linalg.norm(fisher), 1e-30)
    metadata.update(
        singular_values=singular.tolist(), retained=int(keep.sum()),
        local_rows=len(selected), local_regression_condition=float(np.linalg.cond(design)),
        compression_covariance_identity_error=float(identity_error),
        retained_fisher_relative_error=float(fisher_error),
        derivative_method='Local linear regression of transformed clean spectra',
        coordinate_transform='Training-only signed asinh per individual multipole',
        caveat='Approximate local mean compression; not a nine-parameter rank or SBI calibration certificate')
    return matrix, metadata
