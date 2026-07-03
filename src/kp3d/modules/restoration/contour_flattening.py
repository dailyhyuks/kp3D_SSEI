"""Contour-Based Region Flattening restoration wrapper (v10).

BaseRestoration wrapper around ContourRegionFlattener.
Handles Tensor ↔ numpy conversion and ModuleOutput generation.
"""

import time

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Any, Optional

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig
from kp3d.modules.restoration.contour_region_flattener import ContourRegionFlattener


class ContourFlatteningRestorer(BaseRestoration):
    """Contour-based region flattening restorer for grid removal."""

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = True

        # Create the core algorithm instance from config
        self._flattener = ContourRegionFlattener(
            period_x=self.config.contour_period_x,
            period_y=self.config.contour_period_y,
            edge_low=self.config.contour_edge_low,
            edge_high=self.config.contour_edge_high,
            confidence_threshold=self.config.contour_confidence_threshold,
            min_region_area=self.config.contour_min_region_area,
            flatten_method=self.config.contour_flatten_method,
            blend_width=self.config.contour_blend_width,
            min_edge_length=self.config.contour_min_edge_length,
            chrominance_threshold=self.config.contour_chrominance_threshold,
        )

    @property
    def name(self) -> str:
        return "contour_flattening"

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
        """Run contour-based region flattening.

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
        result_bgr, intermediates_raw = self._flattener.process(img_bgr)

        elapsed = time.time() - start

        # Build intermediate tensors for visualization
        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_bgr_to_tensor(img_bgr),
                'raw_edge_map': self._mask_to_tensor(intermediates_raw['raw_edge_map']),
                'final_edge_mask': self._mask_to_tensor(intermediates_raw['final_edge_mask']),
                'flattened': self._numpy_bgr_to_tensor(intermediates_raw['flattened']),
            }
            # Confidence map needs normalization for visualization
            conf = intermediates_raw['confidence_map']
            if conf.max() > 1e-8:
                conf_vis = (conf / conf.max() * 255).astype(np.uint8)
            else:
                conf_vis = np.zeros_like(conf, dtype=np.uint8)
            intermediates['confidence_map'] = self._mask_to_tensor(conf_vis)

        # Build result tensor
        result_tensor = self._numpy_bgr_to_tensor(result_bgr).unsqueeze(0)

        metadata = {
            'method': 'contour_flattening',
            'processing_time': elapsed,
            'detected_periods': intermediates_raw['detected_periods'],
            'n_regions': intermediates_raw['n_regions'],
            'flatten_method': self.config.contour_flatten_method,
            'confidence_threshold': self.config.contour_confidence_threshold,
            'blend_width': self.config.contour_blend_width,
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata=metadata,
        )
