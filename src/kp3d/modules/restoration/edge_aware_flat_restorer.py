"""Edge-Aware Flat Color restoration wrapper.

BaseRestoration wrapper around EdgeAwareFlatProcessor.
Handles Tensor <-> numpy conversion and ModuleOutput generation.
"""

import time

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Any, Optional

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig
from kp3d.modules.restoration.edge_aware_flat import EdgeAwareFlatProcessor


class EdgeAwareFlatRestorer(BaseRestoration):
    """Edge-aware flat color restorer for Korean traditional paintings."""

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = True

        # Create the core algorithm instance from config
        self._processor = EdgeAwareFlatProcessor(
            delta_e_high=self.config.eaf_delta_e_high,
            delta_e_low=self.config.eaf_delta_e_low,
            chrominance_sigma=self.config.eaf_chrominance_sigma,
            chrominance_threshold=self.config.eaf_chrominance_threshold,
            persistence_sigma=self.config.eaf_persistence_sigma,
            confidence_threshold=self.config.eaf_confidence_threshold,
            periodicity_threshold=self.config.eaf_periodicity_threshold,
            min_edge_length=self.config.eaf_min_edge_length,
            min_region_area=self.config.eaf_min_region_area,
            edge_dilate=self.config.eaf_edge_dilate,
            blend_width=self.config.eaf_blend_width,
            pre_blur_sigma=self.config.eaf_pre_blur_sigma,
            bilateral_iterations=self.config.eaf_bilateral_iterations,
            bilateral_d=self.config.eaf_bilateral_d,
            bilateral_sigma_color=self.config.eaf_bilateral_sigma_color,
            bilateral_sigma_space=self.config.eaf_bilateral_sigma_space,
        )

    @property
    def name(self) -> str:
        return "edge_aware_flat"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy_bgr(self, tensor: Tensor) -> np.ndarray:
        """Convert [C,H,W] or [1,C,H,W] float tensor (0-1, RGB) to BGR uint8 numpy."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
        # RGB to BGR
        arr_bgr = arr[:, :, ::-1].copy()
        return (np.clip(arr_bgr, 0, 1) * 255).astype(np.uint8)

    def _numpy_bgr_to_tensor(self, bgr: np.ndarray) -> Tensor:
        """Convert BGR uint8 numpy to [C,H,W] float tensor (0-1, RGB)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def _mask_to_tensor(self, mask: np.ndarray) -> Tensor:
        """Convert single-channel uint8 mask to 3-channel tensor for visualization."""
        vis = np.stack([mask, mask, mask], axis=-1)
        arr = vis.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Run edge-aware flat color restoration.

        Args:
            image: Input image tensor [1,C,H,W] or [C,H,W], float32, 0-1, RGB.

        Returns:
            ModuleOutput with result, intermediates, and metadata.
        """
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to BGR numpy for processing
        img_bgr = self._tensor_to_numpy_bgr(image[0])

        # Run the core algorithm
        result_bgr, intermediates_raw = self._processor.process(img_bgr)

        elapsed = time.time() - start

        # Build intermediate tensors for visualization
        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_bgr_to_tensor(img_bgr),
                'edge_map': self._mask_to_tensor(intermediates_raw['edge_map']),
                'cleaned_edge_map': self._mask_to_tensor(intermediates_raw['cleaned_edge_map']),
                'flattened': self._numpy_bgr_to_tensor(intermediates_raw['flattened']),
            }

        # Build result tensor
        result_tensor = self._numpy_bgr_to_tensor(result_bgr).unsqueeze(0)

        metadata = {
            'method': 'edge_aware_flat',
            'processing_time': elapsed,
            'n_regions': intermediates_raw['n_regions'],
            'delta_e_high': self.config.eaf_delta_e_high,
            'delta_e_low': self.config.eaf_delta_e_low,
            'min_region_area': self.config.eaf_min_region_area,
            'blend_width': self.config.eaf_blend_width,
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata=metadata,
        )
