"""Depth estimation wrapper for occlusion pipeline.

Wraps the existing MiDaSDepthEstimator for integration with
the occlusion handling pipeline.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor

from kp3d.modules.shade.midas import MiDaSDepthEstimator
from kp3d.modules.shade.base import ShadeConfig


class DepthEstimatorWrapper:
    """Wrapper for depth estimation in occlusion pipeline.

    Uses MiDaS models for monocular depth estimation, providing
    normalized depth maps for layer ordering.
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        device: Optional[torch.device] = None
    ):
        """Initialize depth estimator.

        Args:
            model_type: MiDaS model variant ("DPT_Large", "DPT_Hybrid", "MiDaS_small").
            device: Computation device. Defaults to CUDA if available.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # Initialize MiDaS through existing module
        shade_config = ShadeConfig(depth_model=model_type)
        self.midas = MiDaSDepthEstimator(config=shade_config, device=self.device)
        self._initialized = True

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8 [0-255] or float [0-1].

        Returns:
            Depth map as numpy array (H, W), float32 [0-1].
            Higher values = farther from camera.
        """
        # Convert to tensor
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)

        # Get depth
        output = self.midas(tensor)
        depth = output.result.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        # MiDaS outputs inverse depth (higher = closer)
        # Convert to standard depth (lower = closer) for consistency
        depth = 1.0 - depth

        return depth.astype(np.float32)

    def estimate_tensor(self, image: Tensor) -> Tensor:
        """Estimate depth from tensor input.

        Args:
            image: RGB tensor (B, C, H, W) or (C, H, W).

        Returns:
            Depth tensor (B, 1, H, W) or (1, H, W), float32 [0-1].
        """
        output = self.midas(image)
        depth = output.result

        # Normalize
        b = depth.shape[0] if depth.dim() == 4 else 1
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)

        for i in range(b):
            d = depth[i]
            depth[i] = (d - d.min()) / (d.max() - d.min() + 1e-8)

        return depth

    def get_mean_depth(self, depth_map: np.ndarray, mask: np.ndarray) -> float:
        """Calculate mean depth within a masked region.

        Args:
            depth_map: Full depth map (H, W).
            mask: Binary mask (H, W).

        Returns:
            Mean depth value, or 0.0 if mask is empty.
        """
        if mask.sum() == 0:
            return 0.0

        masked_depth = depth_map[mask > 0]
        return float(np.mean(masked_depth))

    def compare_depths(
        self,
        depth_map: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray
    ) -> Tuple[str, float, float]:
        """Compare depths of two masked regions.

        Args:
            depth_map: Full depth map (H, W).
            mask_a: Binary mask for region A.
            mask_b: Binary mask for region B.

        Returns:
            Tuple of (closer_label, depth_a, depth_b).
            closer_label is "A" or "B".
        """
        depth_a = self.get_mean_depth(depth_map, mask_a)
        depth_b = self.get_mean_depth(depth_map, mask_b)

        # Lower depth = closer to camera (foreground)
        closer = "A" if depth_a < depth_b else "B"

        return closer, depth_a, depth_b
