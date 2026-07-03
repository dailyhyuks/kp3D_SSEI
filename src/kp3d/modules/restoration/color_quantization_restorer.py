"""Color Quantization restoration wrapper.

BaseRestoration wrapper around ColorQuantizationProcessor.
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
from kp3d.modules.restoration.color_quantization import ColorQuantizationProcessor


class ColorQuantizationRestorer(BaseRestoration):
    """Color quantization restorer for Korean traditional paintings (v12)."""

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = True

        self._processor = ColorQuantizationProcessor(
            k_min=self.config.cq_k_min,
            k_max=self.config.cq_k_max,
            k_selection=self.config.cq_k_selection,
            pre_filter=self.config.cq_pre_filter,
            rolling_iterations=self.config.cq_rolling_iterations,
            rolling_sigma_s=self.config.cq_rolling_sigma_s,
            rolling_sigma_r=self.config.cq_rolling_sigma_r,
            ink_l_threshold=self.config.cq_ink_l_threshold,
            ink_chroma_threshold=self.config.cq_ink_chroma_threshold,
            quantization_method=self.config.cq_quantization_method,
            min_region_area=self.config.cq_min_region_area,
            blend_width=self.config.cq_blend_width,
            flatten_strength=self.config.cq_flatten_strength,
            adaptive_flatten=self.config.cq_adaptive_flatten,
            variance_threshold=self.config.cq_variance_threshold,
        )

    @property
    def name(self) -> str:
        return "color_quantization"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy_bgr(self, tensor: Tensor) -> np.ndarray:
        """Convert [C,H,W] or [1,C,H,W] float tensor (0-1, RGB) to BGR uint8 numpy."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
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
        """Run color quantization restoration.

        Args:
            image: Input image tensor [1,C,H,W] or [C,H,W], float32, 0-1, RGB.

        Returns:
            ModuleOutput with result, intermediates, and metadata.
        """
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        img_bgr = self._tensor_to_numpy_bgr(image[0])

        result_bgr, intermediates_raw = self._processor.process(img_bgr)

        elapsed = time.time() - start

        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_bgr_to_tensor(img_bgr),
                'filtered': self._numpy_bgr_to_tensor(intermediates_raw['filtered']),
                'ink_mask': self._mask_to_tensor(intermediates_raw['ink_mask']),
                'quantized_preview': self._numpy_bgr_to_tensor(
                    intermediates_raw['quantized_preview']
                ),
            }

        result_tensor = self._numpy_bgr_to_tensor(result_bgr).unsqueeze(0)

        metadata = {
            'method': 'color_quantization',
            'processing_time': elapsed,
            'k': intermediates_raw['k'],
            'n_regions': intermediates_raw['n_regions'],
            'k_min': self.config.cq_k_min,
            'k_max': self.config.cq_k_max,
            'k_selection': self.config.cq_k_selection,
            'pre_filter': self.config.cq_pre_filter,
            'quantization_method': self.config.cq_quantization_method,
            'min_region_area': self.config.cq_min_region_area,
            'blend_width': self.config.cq_blend_width,
        }

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata=metadata,
        )
