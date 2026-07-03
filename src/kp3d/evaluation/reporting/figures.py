"""Comparison figure generation using matplotlib.

Generates visual comparison grids showing input, baseline results,
and our method results side-by-side for qualitative evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def generate_comparison_figure(
    images: Dict[str, np.ndarray],
    output_path: str,
    title: str = "",
    figsize: Tuple[float, float] = (16, 4),
    dpi: int = 150,
) -> None:
    """Generate a comparison figure with multiple methods side by side.

    Args:
        images: Dict of method_name -> image (H, W, 3) RGB uint8.
        output_path: Path to save figure.
        title: Figure title.
        figsize: Figure size (width, height) in inches.
        dpi: Figure DPI.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping figure generation")
        return

    n = len(images)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def generate_enhancement_comparison(
    original: np.ndarray,
    degraded: np.ndarray,
    results: Dict[str, np.ndarray],
    output_path: str,
    crop_region: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Generate enhancement comparison grid.

    Shows: Original | Degraded (grid) | Baseline1 | ... | Ours

    Args:
        original: Pseudo-clean GT image (BGR).
        degraded: Grid-degraded image (BGR).
        results: Dict of method_name -> result image (BGR).
        output_path: Save path.
        crop_region: Optional (y1, y2, x1, x2) for detail crop.
    """
    images = {}

    def to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def maybe_crop(img):
        if crop_region:
            y1, y2, x1, x2 = crop_region
            return img[y1:y2, x1:x2]
        return img

    images["GT (Pseudo-clean)"] = maybe_crop(to_rgb(original))
    images["Input (Grid)"] = maybe_crop(to_rgb(degraded))

    for name, result_img in results.items():
        display_name = _display_name(name)
        images[display_name] = maybe_crop(to_rgb(result_img))

    generate_comparison_figure(images, output_path, title="Enhancement Comparison")


def generate_inpainting_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    mask: np.ndarray,
    results: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Generate inpainting comparison grid.

    Shows: Original | Masked | Baseline1 | ... | Ours

    Args:
        original: Original unoccluded image (RGB).
        masked: Occluded input image (RGB).
        mask: Occlusion mask (H, W), 255=occluded.
        results: Dict of method_name -> inpainted image (RGB).
        output_path: Save path.
    """
    images = {}

    # Show mask overlay on input
    masked_vis = masked.copy()
    mask_overlay = np.zeros_like(masked_vis)
    mask_overlay[mask > 127] = [255, 0, 0]  # Red overlay for mask
    masked_vis = cv2.addWeighted(masked_vis, 0.7, mask_overlay, 0.3, 0)

    images["Ground Truth"] = original
    images["Input (Occluded)"] = masked_vis

    for name, result_img in results.items():
        display_name = _display_name(name)
        images[display_name] = result_img

    generate_comparison_figure(images, output_path, title="Inpainting Comparison")


def generate_ablation_figure(
    results: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Generate ablation study comparison grid.

    Args:
        results: Dict of config_name -> output image (BGR).
        output_path: Save path.
    """
    images = {}
    for name, img in results.items():
        images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    generate_comparison_figure(
        images, output_path, title="Pipeline Ablation", figsize=(20, 4)
    )


def _display_name(method_name: str) -> str:
    """Convert internal method name to display name."""
    names = {
        "bilateral": "Bilateral",
        "nlmeans": "NL-Means",
        "median": "Median",
        "guided": "Guided",
        "butterworth": "Butterworth",
        "ours_spectral": "Ours",
        "opencv_telea": "Telea",
        "opencv_ns": "Navier-Stokes",
        "lama": "LaMa",
        "sd_inpaint": "SD Inpaint",
        "brushnet": "BrushNet",
        "powerpaint": "PowerPaint",
        "ours_v25": "Ours (SSEI)",
        "no_processing": "No Processing",
    }
    return names.get(method_name, method_name)
