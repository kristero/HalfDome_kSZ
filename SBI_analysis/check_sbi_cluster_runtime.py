#!/usr/bin/env python3
"""Fail fast when the cluster Python scientific stack is inconsistent."""

from __future__ import annotations

import importlib
import json
import platform
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


MODULES = ("numpy", "torch", "numba", "arviz", "sbi")


def module_record(name: str) -> dict[str, str]:
    module = importlib.import_module(name)
    return {
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(Path(getattr(module, "__file__", "")).resolve()),
    }


def installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "installed_distributions": {
            name: installed_version(name) for name in MODULES
        },
        "modules": {},
    }
    try:
        for name in MODULES:
            report["modules"][name] = module_record(name)
    except Exception:
        report["failed_module"] = name
        report["traceback"] = traceback.format_exc()
        print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
        print(
            "\nSBI runtime preflight failed. Use one Python environment where "
            "importing numpy, torch, numba, arviz, and sbi succeeds. The common "
            "idark failure is user-site sbi/arviz loading together with the "
            "system-Anaconda numba binary. Point PYTHON at a consistent conda "
            "environment or virtual environment before submitting jobs.",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    print("SBI runtime preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
