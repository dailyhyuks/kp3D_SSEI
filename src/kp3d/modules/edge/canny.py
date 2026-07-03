"""Canny edge detector implementation."""

import time
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch import Tensor

from kp3d.core.base import ModuleOutput
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig


class CannyEdgeDetector(BaseEdgeDetection):
    """OpenCV Canny-based edge detector.

    Detects edges using the Canny algorithm with multi-scale support.
    Optimized for Korean traditional painting characteristics.
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        **kwargs
    ) -> None:
        """Initialize Canny edge detector.

        Args:
            config: Edge detection configuration.
            **kwargs: Additional arguments.
        """
        super().__init__(config=config, **kwargs)
        self._initialized = True  # No weights to load for Canny

    @property
    def name(self) -> str:
        """Module name."""
        return "canny_edge"

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights (not applicable for Canny).

        Args:
            checkpoint_path: Path to checkpoint (ignored).
        """
        # Canny doesn't use learned weights
        self._initialized = True

    def _tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to numpy array for OpenCV.

        Args:
            tensor: Input tensor (B, C, H, W) or (C, H, W).

        Returns:
            Numpy array in HWC format (uint8).
        """
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor[0]

        # Convert to CPU and numpy
        array = tensor.cpu().numpy()

        # CHW -> HWC
        if array.shape[0] in [1, 3]:
            array = np.transpose(array, (1, 2, 0))

        # Normalize to 0-255 if needed
        if array.max() <= 1.0:
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

        return array

    def _numpy_to_tensor(self, array: np.ndarray) -> Tensor:
        """Convert numpy array back to tensor.

        Args:
            array: Input array (H, W) or (H, W, C).

        Returns:
            Tensor in CHW format.
        """
        # Ensure float32 and 0-1 range
        array = array.astype(np.float32) / 255.0

        # Add channel dimension if grayscale
        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        # HWC -> CHW
        tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))

        return tensor.to(device=self.device, dtype=self.dtype)

    def _detect_single_scale(
        self,
        image: np.ndarray,
        scale: float = 1.0
    ) -> np.ndarray:
        """Detect edges at a single scale.

        Args:
            image: Input image (H, W, C) or (H, W).
            scale: Scale factor (1.0 = original size).

        Returns:
            Edge map (H, W) in range [0, 255].
        """
        # Convert to grayscale if needed
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.squeeze()

        # Resize if scale != 1.0
        if scale != 1.0:
            h, w = gray.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Canny edge detection
        edges = cv2.Canny(
            blurred,
            self.config.low_threshold,
            self.config.high_threshold,
            apertureSize=3,
            L2gradient=True  # More accurate gradient calculation
        )

        # Resize back to original size if needed
        if scale != 1.0:
            edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)

        return edges

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Detect edges using Canny algorithm.

        Args:
            image: Input image tensor (B, C, H, W).
            **kwargs: Additional parameters (can override config values).

        Returns:
            ModuleOutput with edge map and intermediate results.
        """
        start_time = time.time()

        # Override config with kwargs if provided
        low_threshold = kwargs.get('low_threshold', self.config.low_threshold)
        high_threshold = kwargs.get('high_threshold', self.config.high_threshold)
        multi_scale = kwargs.get('multi_scale', self.config.multi_scale)
        scales = kwargs.get('scales', self.config.scales)

        # Update thresholds temporarily
        original_low = self.config.low_threshold
        original_high = self.config.high_threshold
        self.config.low_threshold = low_threshold
        self.config.high_threshold = high_threshold

        # Convert to numpy
        np_image = self._tensor_to_numpy(image)

        # Multi-scale detection
        if multi_scale and len(scales) > 1:
            edge_maps = []
            for scale in scales:
                edge_map = self._detect_single_scale(np_image, scale)
                edge_maps.append(edge_map.astype(np.float32))

            # Weighted combination (prioritize larger scales)
            weights = np.array(scales) / sum(scales)
            combined = np.zeros_like(edge_maps[0])
            for edge_map, weight in zip(edge_maps, weights):
                combined += edge_map * weight

            edges = combined.astype(np.uint8)
            intermediate_maps = {
                f"scale_{scale}": self._numpy_to_tensor(em)
                for scale, em in zip(scales, edge_maps)
            }
        else:
            # Single scale
            edges = self._detect_single_scale(np_image, 1.0)
            intermediate_maps = {}

        # Convert back to tensor
        edge_tensor = self._numpy_to_tensor(edges)

        # Add batch dimension
        edge_tensor = edge_tensor.unsqueeze(0)

        # Restore original config
        self.config.low_threshold = original_low
        self.config.high_threshold = original_high

        elapsed = time.time() - start_time

        return ModuleOutput(
            result=edge_tensor,
            intermediate={
                "input": image,
                **intermediate_maps
            },
            metadata={
                "method": "canny",
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "multi_scale": multi_scale,
                "scales": scales if multi_scale else [1.0],
                "processing_time": elapsed
            }
        )
