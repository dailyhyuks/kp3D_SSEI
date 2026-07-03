"""MiDaS-based depth estimation for shade generation."""

import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from kp3d.core.base import ModuleOutput
from .base import BaseShadeGeneration, ShadeConfig


class MiDaSDepthEstimator(BaseShadeGeneration):
    """MiDaS-based depth estimation module.

    Uses Intel's MiDaS model for monocular depth estimation,
    optimized for generating realistic shading on 2D images.

    Supported models:
        - DPT_Large: Highest quality, slower
        - DPT_Hybrid: Balanced quality/speed
        - MiDaS_small: Fastest, lower quality
    """

    MODEL_TYPES = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]

    def __init__(self, config: Optional[ShadeConfig] = None, **kwargs):
        """Initialize MiDaS depth estimator.

        Args:
            config: Shade configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config=config, **kwargs)

        if self.config.depth_model not in self.MODEL_TYPES:
            raise ValueError(
                f"Invalid depth model: {self.config.depth_model}. "
                f"Choose from {self.MODEL_TYPES}"
            )

        self.model = None
        self.transform = None
        self._load_midas_model()

    def _load_midas_model(self) -> None:
        """Load MiDaS model from torch.hub."""
        try:
            # Load model from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS",
                self.config.depth_model,
                pretrained=True,
                trust_repo=True
            )

            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

            if self.config.depth_model == "DPT_Large" or self.config.depth_model == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            self._initialized = True

        except Exception as e:
            raise RuntimeError(
                f"Failed to load MiDaS model '{self.config.depth_model}': {e}"
            )

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights (uses torch.hub instead).

        Args:
            checkpoint_path: Not used, MiDaS loads from hub.
        """
        # MiDaS loads weights automatically from torch.hub
        if not self._initialized:
            self._load_midas_model()

    def estimate_depth(self, image: Tensor) -> Tensor:
        """Estimate depth map from image.

        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1].

        Returns:
            Depth map tensor (B, 1, H, W), normalized to [0, 1].
        """
        if not self._initialized:
            raise RuntimeError("MiDaS model not initialized. Call load_weights() first.")

        batch_size, _, orig_h, orig_w = image.shape

        # Convert to numpy for MiDaS transform (expects RGB uint8)
        # MiDaS expects (H, W, C) format in 0-255 range
        input_batch = []

        for i in range(batch_size):
            # Convert to numpy
            img_np = image[i].cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype('uint8')

            # Apply MiDaS transform (returns [1, 3, H, W])
            input_tensor = self.transform(img_np).to(self.device)
            input_batch.append(input_tensor)

        # Concatenate along batch dimension (each tensor is [1, 3, H, W])
        input_batch = torch.cat(input_batch, dim=0)

        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)

        # Resize to original resolution
        depth_map = F.interpolate(
            prediction.unsqueeze(1),
            size=(orig_h, orig_w),
            mode='bicubic',
            align_corners=False
        )

        # Normalize to [0, 1] range (MiDaS outputs inverse depth)
        # Invert so far=0, near=1
        depth_map = 1.0 / (depth_map + 1e-6)

        # Normalize to 0-1
        for i in range(batch_size):
            d = depth_map[i:i+1]
            d_min = d.min()
            d_max = d.max()
            if (d_max - d_min) > 1e-6:
                depth_map[i:i+1] = (d - d_min) / (d_max - d_min)

        return depth_map

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Estimate depth map from image.

        Args:
            image: Input image tensor (B, C, H, W).
            **kwargs: Additional arguments (edge_map for future use).

        Returns:
            ModuleOutput with depth map as result.
        """
        # Estimate depth
        depth_map = self.estimate_depth(image)

        # Optional: use edge_map to refine depth if provided
        edge_map = kwargs.get('edge_map', None)
        if edge_map is not None:
            # Edge-aware smoothing (simple version)
            # Strong edges = preserve depth discontinuities
            edge_weight = edge_map.mean(dim=1, keepdim=True) if edge_map.shape[1] > 1 else edge_map
            edge_weight = 1.0 - torch.clamp(edge_weight, 0, 1)

            # Smooth depth in non-edge regions
            smoothed = F.avg_pool2d(
                F.pad(depth_map, (1, 1, 1, 1), mode='replicate'),
                kernel_size=3,
                stride=1
            )
            depth_map = edge_weight * depth_map + (1 - edge_weight) * smoothed

        return ModuleOutput(
            result=depth_map,
            intermediate={
                "depth_map": depth_map,
            },
            metadata={
                "model": self.config.depth_model,
                "original_shape": image.shape,
            }
        )

    @property
    def name(self) -> str:
        """Module name."""
        return f"midas_depth_{self.config.depth_model}"
