"""Regenerate only the three figures whose captions / labels were corrected on 2026-10-14
(fig1 panel-b note, fig4 title, fig9a dashed-line label). Reads the same work/ files as make_figures.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as MF  # noqa: E402

MF.fig_innermost()
MF.fig_none_inner()
MF.fig_redshift()
