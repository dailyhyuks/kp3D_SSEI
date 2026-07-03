"""Reporting utilities for evaluation results.

- LaTeX table generation
- Comparison figure generation
- JSON result export/import
"""

from kp3d.evaluation.reporting.json_export import load_results, save_results
from kp3d.evaluation.reporting.latex_table import generate_latex_table
from kp3d.evaluation.reporting.figures import generate_comparison_figure

__all__ = [
    "save_results",
    "load_results",
    "generate_latex_table",
    "generate_comparison_figure",
]
