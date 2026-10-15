#!/usr/bin/env python3
"""Fix corner typography from saved posterior draws; no new inference."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from plot_truth_mean import STYLE, corner_figures, load_verified, sha, write_json

ROOT = Path('/lustre/work/kristero10/tsz_truth_mean_validation_20261006')


def main():
    marker = ROOT/'PLOTS_COMPLETE.json'
    previous = json.loads(marker.read_text())
    for name, checksum in previous['outputs'].items():
        assert sha(ROOT/name) == checksum, name
    load_verified(ROOT)
    original_sha = sha(marker)
    archive = ROOT/'evidence/initial_plots_complete.json'
    if archive.exists():
        raise RuntimeError('Publication formatting has already run; do not overwrite its history')
    shutil.copy2(marker, archive)
    plt.rcParams.update(STYLE)
    corner_figures(ROOT)
    definition_path = ROOT/'CORNER_PLOT_DEFINITION.json'
    definition = json.loads(definition_path.read_text())
    definition['contour_interpretation'] = '68/95 percent TWO-PARAMETER MARGINAL regions; not nine-dimensional joint HPD'
    definition['typography'] = 'Larger labels/ticks; diagonal density tick labels removed; posterior draws unchanged'
    write_json(definition_path, definition)
    updated = dict(previous)
    updated['previous_plot_marker_sha256'] = original_sha
    updated['publication_job'] = os.environ['PBS_JOBID']
    updated['publication_source_sha256'] = {p.name: sha(p) for p in
        [ROOT/'publication_corners.py', ROOT/'plot_truth_mean.py', ROOT/'publication_corners.pbs']}
    updated['outputs'] = {name: sha(ROOT/name) for name in previous['outputs']}
    write_json(marker, updated)
    write_json(ROOT/'PUBLICATION_COMPLETE.json', dict(
        finished_utc=datetime.now(timezone.utc).isoformat(), job=os.environ['PBS_JOBID'],
        plots_marker_sha256=sha(marker), posterior_inference_unchanged=True,
        inference_marker_sha256=sha(ROOT/'INFERENCE_COMPLETE.json'),
        source_sha256=updated['publication_source_sha256']))


if __name__ == '__main__':
    main()
