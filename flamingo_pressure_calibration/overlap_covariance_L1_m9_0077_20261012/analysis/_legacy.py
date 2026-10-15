"""Exact-path loaders for frozen modules of earlier (read-only) studies.

Earlier study directories are never put on ``sys.path`` and never written to.
Each module is executed from its exact file path under a private module name.
Bare-name imports inside a legacy module (``import pilot_physics as P``) are
satisfied by temporarily registering the already-loaded dependency under that
bare name; the registration is removed again afterwards.  Byte-code writing is
disabled during these loads, so no ``__pycache__`` appears in old studies even
when the caller forgot ``python -B``.

Note: ``scatter_r4_bins02_*/pilot_physics.py`` has a *different* EDGES array;
always use :func:`pilot_physics` (radii234) for this study.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = STUDY_ROOT.parent
RADII234 = CALIBRATION / "radii234_contamination_L1_m9_0077_20261008"
SENSITIVITY = CALIBRATION / "sensitivity_projects_L1_m9_0077_20261009"

_CACHE: dict = {}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name: str, path, provides: dict | None = None):
    """Execute ``path`` as module ``name``; ``provides`` maps bare import names to modules."""
    path = Path(path)
    key = (name, str(path))
    if key in _CACHE:
        return _CACHE[key]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    saved = {}
    previous_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for alias, dependency in (provides or {}).items():
            saved[alias] = sys.modules.get(alias)
            sys.modules[alias] = dependency
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_flag
        for alias, old in saved.items():
            if old is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = old
    _CACHE[key] = module
    return module


def pilot_physics():
    """radii234 ``pilot_physics`` (36-shell EDGES, SUM_NAMES, p200, B12/B16)."""
    return load("radii234_pilot_physics", RADII234 / "pilot_physics.py")


def models():
    """radii234 ``models`` (shape, shell_grid, shell, fit_profile, BOUNDS)."""
    return load("radii234_models", RADII234 / "models.py",
                provides={"pilot_physics": pilot_physics()})


def measure():
    return load("radii234_measure", RADII234 / "measure.py",
                provides={"pilot_physics": pilot_physics()})


def fit_study():
    """radii234 ``fit_study`` -- only for parity tests (it sets OMP thread variables)."""
    meas = measure()
    mods = models()
    projection = load("radii234_projection_model", RADII234 / "projection_model.py",
                      provides={"measure": meas, "models": mods})
    return load("radii234_fit_study", RADII234 / "fit_study.py",
                provides={"pilot_physics": pilot_physics(), "measure": meas, "models": mods,
                          "projection_model": projection})


def analysis_common():
    """sensitivity_projects ``analysis_common`` -- only for parity tests.

    The module inserts the radii234 directory into ``sys.path`` at import; the
    insertion is undone here so later bare imports cannot resolve to old files.
    """
    before = list(sys.path)
    try:
        module = load("sensitivity_analysis_common", SENSITIVITY / "analysis_common.py",
                      provides={"pilot_physics": pilot_physics(), "fit_study": fit_study(),
                                "models": models()})
    finally:
        sys.path[:] = before
    return module


def frozen_source_check() -> dict:
    """Compare loaded legacy sources with radii234 ``executed_sources.json``."""
    recorded = json.loads((RADII234 / "executed_sources.json").read_text())
    result = {}
    for name in ("pilot_physics.py", "models.py", "measure.py", "fit_study.py", "projection_model.py"):
        actual = sha256(RADII234 / name)
        result[name] = {"sha256": actual, "recorded": recorded.get(name),
                        "matches_executed_sources": actual == recorded.get(name)}
    return result
