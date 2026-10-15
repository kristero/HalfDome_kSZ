"""Shared analysis library of overlap_covariance_L1_m9_0077_20261012 (plan step 0.3).

Import with ``sys.path.insert(0, <study root>); import analysis``.  Submodules:
``io`` (data contract, normalisation, rebinning), ``masks`` (resolution masks),
``stacking`` (stacks, paired bootstraps), ``covariance`` (intrinsic and
mean-profile covariances, shrinkage, factor model, Hartlap), ``stability``
(permutation null and correlation-structure metrics), ``families`` (B12, B16,
gNFW5, B12 + outer power law, B12 x spline; truncated shell averages), ``fitting``
(shell-averaged diagonal/GLS fits, objectives (d)-(g) for observable-driven
weighting, influence rows, residual and fit/cut-radius diagnostics), ``newdata`` (loader of the
particle-pass products, normalisations, census classes), ``variants`` (profile constructor:
estimators, AGN phases, gas selections, excision, templates, grids), ``pipeline`` (per-bin
estimators (a)/(b)/(c) with paired bootstrap replicates) and ``plotting``
(figure style).  Earlier studies are loaded read-only by ``_legacy``.
See README.md.
"""
from . import _legacy, io, masks, stacking, covariance, stability, families, fitting, pipeline  # noqa: F401
from . import newdata, variants  # noqa: F401

__version__ = "0.3.0"
