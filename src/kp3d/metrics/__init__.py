"""Metrics for evaluating Korean painting preprocessing quality.

Provides image quality metrics including PSNR, SSIM, LPIPS,
and custom metrics for traditional painting characteristics.
"""

from typing import Dict, Optional

import torch
from torch import Tensor

try:
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


class MetricsCalculator:
    """Calculator for image quality metrics.

    Computes various quality metrics between original and processed images.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        use_lpips: bool = True,
    ) -> None:
        """Initialize metrics calculator.

        Args:
            device: Compute device.
            use_lpips: Whether to compute LPIPS (requires lpips package).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._psnr = None
        self._ssim = None
        self._lpips = None

        self._init_metrics(use_lpips)

    def _init_metrics(self, use_lpips: bool) -> None:
        """Initialize metric functions."""
        if HAS_TORCHMETRICS:
            self._psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        if use_lpips and HAS_LPIPS:
            self._lpips = lpips.LPIPS(net="alex").to(self.device)
            self._lpips.eval()

    def compute_psnr(self, pred: Tensor, target: Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            PSNR value in dB.
        """
        if self._psnr is None:
            # Fallback implementation
            mse = torch.mean((pred - target) ** 2)
            if mse == 0:
                return float("inf")
            return (10 * torch.log10(1.0 / mse)).item()

        pred = pred.to(self.device)
        target = target.to(self.device)
        return self._psnr(pred, target).item()

    def compute_ssim(self, pred: Tensor, target: Tensor) -> float:
        """Compute Structural Similarity Index.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            SSIM value (0-1, higher is better).
        """
        if self._ssim is None:
            # Simplified fallback
            return 0.0

        pred = pred.to(self.device)
        target = target.to(self.device)

        # Ensure batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        return self._ssim(pred, target).item()

    def compute_lpips(self, pred: Tensor, target: Tensor) -> float:
        """Compute Learned Perceptual Image Patch Similarity.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            LPIPS value (0-1, lower is better).
        """
        if self._lpips is None:
            return 0.0

        pred = pred.to(self.device)
        target = target.to(self.device)

        # Ensure batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if target.dim() == 3:
            target = target.unsqueeze(0)

        # LPIPS expects [-1, 1] range
        pred = pred * 2 - 1
        target = target * 2 - 1

        with torch.no_grad():
            return self._lpips(pred, target).mean().item()

    def compute_all(
        self,
        pred: Tensor,
        target: Tensor,
    ) -> Dict[str, float]:
        """Compute all available metrics.

        Args:
            pred: Predicted image tensor.
            target: Target image tensor.

        Returns:
            Dictionary of metric name to value.
        """
        metrics = {}

        metrics["psnr"] = self.compute_psnr(pred, target)
        metrics["ssim"] = self.compute_ssim(pred, target)

        if self._lpips is not None:
            metrics["lpips"] = self.compute_lpips(pred, target)

        return metrics


def compute_edge_preservation(
    original_edges: Tensor,
    processed_edges: Tensor,
    threshold: float = 0.1,
) -> float:
    """Compute edge preservation ratio.

    Measures how well edges from the original are preserved in processed.

    Args:
        original_edges: Edge map from original image.
        processed_edges: Edge map from processed image.
        threshold: Threshold for edge detection.

    Returns:
        Preservation ratio (0-1).
    """
    orig_binary = (original_edges > threshold).float()
    proc_binary = (processed_edges > threshold).float()

    # Intersection over original edges
    intersection = (orig_binary * proc_binary).sum()
    original_total = orig_binary.sum()

    if original_total == 0:
        return 1.0

    return (intersection / original_total).item()


def compute_color_consistency(
    original: Tensor,
    processed: Tensor,
) -> float:
    """Compute color consistency between images.

    Measures how well the overall color distribution is preserved.

    Args:
        original: Original image tensor.
        processed: Processed image tensor.

    Returns:
        Color consistency score (0-1, higher is better).
    """
    # Compute mean color per channel
    orig_mean = original.mean(dim=(-2, -1))
    proc_mean = processed.mean(dim=(-2, -1))

    # Cosine similarity of color distributions
    similarity = torch.nn.functional.cosine_similarity(
        orig_mean.flatten().unsqueeze(0),
        proc_mean.flatten().unsqueeze(0),
    )

    return ((similarity + 1) / 2).item()  # Map from [-1, 1] to [0, 1]


from kp3d.metrics.inpainting_metrics import (
    color_outlier_rate,
    boundary_smoothness,
    texture_coherence,
    InpaintingMetrics,
)


__all__ = [
    "MetricsCalculator",
    "compute_edge_preservation",
    "compute_color_consistency",
    # V22 Inpainting 특화 지표
    "color_outlier_rate",
    "boundary_smoothness",
    "texture_coherence",
    "InpaintingMetrics",
]
