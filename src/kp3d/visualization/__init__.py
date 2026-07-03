"""Visualization utilities for Korean painting preprocessing.

Provides tools for visualizing preprocessing results, comparisons,
and intermediate processing stages.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def tensor_to_numpy(tensor: Tensor) -> "np.ndarray":
    """Convert a tensor to numpy array for visualization.

    Args:
        tensor: Image tensor (C, H, W) or (B, C, H, W).

    Returns:
        Numpy array (H, W, C) suitable for matplotlib.
    """
    import numpy as np

    if tensor.dim() == 4:
        tensor = tensor[0]

    # Handle grayscale
    if tensor.shape[0] == 1:
        return tensor.squeeze(0).cpu().numpy()

    # RGB
    return tensor.permute(1, 2, 0).cpu().numpy()


def create_comparison_grid(
    images: Dict[str, Tensor],
    titles: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[Union[str, Path]] = None,
) -> Optional["plt.Figure"]:
    """Create a grid visualization comparing multiple images.

    Args:
        images: Dictionary mapping names to image tensors.
        titles: Optional custom titles for each image.
        figsize: Figure size in inches.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure, or None if matplotlib unavailable.
    """
    if not HAS_MATPLOTLIB:
        return None

    n_images = len(images)
    if n_images == 0:
        return None

    # Calculate grid dimensions
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes to 1D list
    if n_images == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        # When only 1 row or 1 col, plt.subplots returns 1D array
        axes_flat = list(axes) if hasattr(axes, '__iter__') else [axes]
    else:
        # For 2D grid, flatten the array
        axes_flat = axes.flatten().tolist()

    for idx, (name, tensor) in enumerate(images.items()):
        ax = axes_flat[idx]
        img = tensor_to_numpy(tensor.clamp(0, 1))

        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        title = titles.get(name, name) if titles else name
        ax.set_title(title)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_images, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_side_by_side(
    original: Tensor,
    processed: Tensor,
    labels: Tuple[str, str] = ("Original", "Processed"),
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> Optional["plt.Figure"]:
    """Create side-by-side comparison of original and processed images.

    Args:
        original: Original image tensor.
        processed: Processed image tensor.
        labels: Labels for the two images.
        figsize: Figure size in inches.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure, or None if matplotlib unavailable.
    """
    return create_comparison_grid(
        {labels[0]: original, labels[1]: processed},
        figsize=figsize,
        save_path=save_path,
    )


def visualize_pipeline_results(
    results: Dict[str, "ModuleOutput"],
    original: Optional[Tensor] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> Optional["plt.Figure"]:
    """Visualize results from all pipeline stages.

    Args:
        results: Dictionary of ModuleOutput from each stage.
        original: Optional original image for comparison.
        save_dir: Directory to save visualizations.

    Returns:
        Main comparison figure.
    """
    images = {}

    if original is not None:
        images["Original"] = original

    for module_name, output in results.items():
        images[f"{module_name.title()} Output"] = output.result

        # Also show key intermediate results
        for key, tensor in output.intermediate.items():
            if tensor.dim() >= 3:
                images[f"{module_name}: {key}"] = tensor

    fig = create_comparison_grid(images)

    if save_dir and fig is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "pipeline_comparison.png", dpi=150, bbox_inches="tight")

    return fig


__all__ = [
    "tensor_to_numpy",
    "create_comparison_grid",
    "create_side_by_side",
    "visualize_pipeline_results",
]
