"""LaTeX table generation from evaluation results.

Generates publication-ready LaTeX tables with:
- Bold best values
- Proper formatting (dB, percentages, etc.)
- Standard table structure for the paper
"""

from typing import Dict, List, Optional, Tuple


def generate_latex_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    caption: str = "",
    label: str = "",
    metric_formats: Optional[Dict[str, str]] = None,
    metric_directions: Optional[Dict[str, str]] = None,
    method_display_names: Optional[Dict[str, str]] = None,
) -> str:
    """Generate LaTeX table from aggregated results.

    Args:
        results: Dict of method_name -> {metric: value}.
        metrics: Ordered list of metric columns to include.
        caption: LaTeX table caption.
        label: LaTeX table label.
        metric_formats: Dict of metric -> format string (e.g., "{:.2f}").
        metric_directions: Dict of metric -> "higher" or "lower" for bolding best.
        method_display_names: Dict of internal name -> display name.

    Returns:
        LaTeX table string.
    """
    if metric_formats is None:
        metric_formats = _default_formats()
    if metric_directions is None:
        metric_directions = _default_directions()
    if method_display_names is None:
        method_display_names = _default_method_names()

    # Find best values for each metric
    best_values = _find_best(results, metrics, metric_directions)

    # Build table
    n_cols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header_cells = ["Method"]
    for m in metrics:
        direction = metric_directions.get(m, "higher")
        arrow = "$\\uparrow$" if direction == "higher" else "$\\downarrow$"
        display = _metric_display_name(m)
        header_cells.append(f"{display}{arrow}")
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for method, values in results.items():
        display_name = method_display_names.get(method, method)
        cells = [display_name]

        for m in metrics:
            val = values.get(m, 0.0)
            fmt = metric_formats.get(m, "{:.3f}")

            # Handle inf/nan values
            if val == float("inf") or val == float("-inf"):
                formatted = "--"
            elif val != val:  # NaN check
                formatted = "--"
            else:
                formatted = fmt.format(val)
                # Bold if best
                if method in best_values.get(m, []):
                    formatted = f"\\textbf{{{formatted}}}"

            cells.append(formatted)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def generate_enhancement_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate Enhancement evaluation table (paper Table format).

    Real grid evaluation: no GT, so SSIM/PSNR measure fidelity to original.
    Key metrics: Grid Energy (lower=better), Band SNR (higher=better),
    Edge Preservation (higher=better), Naturalness (lower=better).
    """
    return generate_latex_table(
        results,
        metrics=["edge_preservation", "grid_energy", "band_snr", "naturalness", "ssim", "psnr"],
        caption="Enhancement (Weave Removal) quantitative comparison on real scanned images.",
        label="tab:enhancement_eval",
        metric_formats={
            "edge_preservation": "{:.3f}",
            "grid_energy": "{:.4f}",
            "band_snr": "{:.1f}",
            "naturalness": "{:.3f}",
            "ssim": "{:.3f}",
            "psnr": "{:.2f}",
        },
        metric_directions={
            "edge_preservation": "higher",
            "grid_energy": "lower",
            "band_snr": "higher",
            "naturalness": "lower",
            "ssim": "higher",
            "psnr": "higher",
        },
    )


def generate_inpainting_table(results: Dict[str, Dict[str, float]]) -> str:
    """Generate Inpainting evaluation table (paper Table format)."""
    return generate_latex_table(
        results,
        metrics=["psnr", "ssim", "lpips", "cor", "bs", "tc"],
        caption="Inpainting quantitative comparison.",
        label="tab:inpainting_eval",
        metric_formats={
            "psnr": "{:.2f}",
            "ssim": "{:.4f}",
            "lpips": "{:.4f}",
            "cor": "{:.4f}",
            "bs": "{:.3f}",
            "tc": "{:.3f}",
        },
        metric_directions={
            "psnr": "higher",
            "ssim": "higher",
            "lpips": "lower",
            "cor": "lower",
            "bs": "higher",
            "tc": "higher",
        },
    )


def _find_best(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    directions: Dict[str, str],
) -> Dict[str, List[str]]:
    """Find best method(s) for each metric (excluding inf/nan and no_processing)."""
    import math

    best = {}
    for m in metrics:
        direction = directions.get(m, "higher")
        # Exclude no_processing and inf/nan values from best selection
        values = {}
        for method, vals in results.items():
            if method == "no_processing":
                continue
            v = vals.get(m, 0)
            if math.isfinite(v):
                values[method] = v

        if not values:
            continue

        if direction == "higher":
            best_val = max(values.values())
        else:
            best_val = min(values.values())

        best[m] = [method for method, v in values.items() if v == best_val]

    return best


def _metric_display_name(metric: str) -> str:
    """Human-readable metric name for table header."""
    names = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "cor": "COR",
        "bs": "BS",
        "tc": "TC",
        "edge_preservation": "Edge",
        "grid_energy": "Grid E.",
        "band_snr": "BandSNR",
        "naturalness": "Natural.",
    }
    return names.get(metric, metric)


def _default_formats() -> Dict[str, str]:
    return {
        "psnr": "{:.2f}",
        "ssim": "{:.4f}",
        "lpips": "{:.4f}",
        "cor": "{:.4f}",
        "bs": "{:.3f}",
        "tc": "{:.3f}",
        "edge_preservation": "{:.3f}",
        "grid_energy": "{:.4f}",
        "band_snr": "{:.1f}",
        "naturalness": "{:.3f}",
    }


def _default_directions() -> Dict[str, str]:
    return {
        "psnr": "higher",
        "ssim": "higher",
        "lpips": "lower",
        "cor": "lower",
        "bs": "higher",
        "tc": "higher",
        "edge_preservation": "higher",
        "grid_energy": "lower",
        "band_snr": "higher",
        "naturalness": "lower",
    }


def _default_method_names() -> Dict[str, str]:
    return {
        "bilateral": "Bilateral Filter",
        "nlmeans": "NL-Means",
        "median": "Median Filter",
        "guided": "Guided Filter",
        "butterworth": "Butterworth Notch",
        "no_processing": "No Processing",
        "ours_spectral": "\\textbf{Ours (Spectral Interp.)}",
        "opencv_telea": "OpenCV Telea",
        "opencv_ns": "OpenCV NS",
        "lama": "LaMa",
        "mat": "MAT",
        "sd_inpaint": "SD Inpaint",
        "brushnet": "BrushNet",
        "powerpaint": "PowerPaint",
        "ours_v25": "\\textbf{Ours (SSEI)}",
    }
