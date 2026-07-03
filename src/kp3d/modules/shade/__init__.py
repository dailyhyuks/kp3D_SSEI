"""Shade generation module for Korean painting preprocessing.

Provides MiDaS-based depth estimation and physically-based shading
generation optimized for traditional Korean painting aesthetics.
"""

from typing import Any, Dict, List, Optional

import os
import torch
import torch.nn.functional as F
from torch import Tensor

from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.core.registry import register_module

# Import new shade generation components
from .base import BaseShadeGeneration, LightSource, ShadeConfig
from .midas import MiDaSDepthEstimator
from .lighting import LightingSimulator, ShadeGeneratorModule


@register_module("shade")
class ShadeModule(BasePreprocessModule):
    """Shade normalization module (legacy).

    Normalizes illumination variations and performs color correction
    while preserving the artistic shading of traditional paintings.

    Note: For depth-based shading, use ShadeGeneratorModule instead.
    """

    # MSRCR algorithm parameters
    DEFAULT_SCALES: List[float] = [15.0, 80.0, 250.0]
    DEFAULT_COLOR_GAIN: float = 125.0
    DEFAULT_COLOR_OFFSET: float = 128.0

    def __init__(
        self,
        target_illumination: float = 0.5,
        preserve_details: bool = True,
        color_correction: bool = True,
        gamma: float = 1.0,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize shade normalization module.

        Args:
            target_illumination: Target average illumination (0-1).
            preserve_details: Preserve fine details during normalization.
            color_correction: Apply color correction.
            gamma: Gamma correction value.
            device: Compute device.
        """
        super().__init__(device=device)
        self.target_illumination = target_illumination
        self.preserve_details = preserve_details
        self.color_correction = color_correction
        self.gamma = gamma
        self._scales = self.DEFAULT_SCALES
        self._color_gain = self.DEFAULT_COLOR_GAIN
        self._color_offset = self.DEFAULT_COLOR_OFFSET

    @property
    def name(self) -> str:
        """Module name."""
        return "shade"

    def load_weights(self, checkpoint_path: Optional[str] = None) -> None:
        """Load pretrained weights or initialize with defaults.

        This is a LEGACY fallback module that works WITHOUT external weights
        using a pure algorithmic MSRCR approach.

        Args:
            checkpoint_path: Optional path to checkpoint file.
        """
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            state_dict: Dict[str, Any] = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            if "scales" in state_dict:
                self._scales = state_dict["scales"]
            if "color_gain" in state_dict:
                self._color_gain = state_dict["color_gain"]
            if "target_illumination" in state_dict:
                self.target_illumination = state_dict["target_illumination"]
            if "gamma" in state_dict:
                self.gamma = state_dict["gamma"]
        self._initialized = True

    def _gaussian_blur(self, image: Tensor, sigma: float) -> Tensor:
        """Apply Gaussian blur."""
        kernel_size = int(6 * sigma + 1) | 1  # Ensure odd
        kernel_size = max(3, kernel_size)
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device)
        x = x - (kernel_size - 1) / 2.0
        gauss = torch.exp(-x.pow(2) / (2 * sigma * sigma))
        gauss = gauss / gauss.sum()
        gauss_h = gauss.view(1, 1, kernel_size, 1)
        gauss_v = gauss.view(1, 1, 1, kernel_size)
        pad = kernel_size // 2
        padded = F.pad(image, (pad, pad, pad, pad), mode='replicate')
        result = []
        for c in range(image.shape[1]):
            ch = padded[:, c:c+1, :, :]
            ch = F.conv2d(F.conv2d(ch, gauss_h), gauss_v)
            result.append(ch)
        return torch.cat(result, dim=1)

    def _multi_scale_retinex(self, image: Tensor) -> Tensor:
        """Compute Multi-Scale Retinex."""
        eps = 1e-6
        msr = torch.zeros_like(image)
        for sigma in self._scales:
            log_img = torch.log(image + eps)
            log_blur = torch.log(self._gaussian_blur(image, sigma) + eps)
            msr += log_img - log_blur
        return msr / len(self._scales)

    def _color_restoration(self, image: Tensor, msr: Tensor) -> Tensor:
        """Apply color restoration."""
        eps = 1e-6
        intensity_sum = image.sum(dim=1, keepdim=True) + eps
        color_factor = torch.log(self._color_gain * (image + eps) / intensity_sum)
        return msr * color_factor

    def _normalize_output(self, msrcr: Tensor) -> Tensor:
        """Normalize to [0, 1] range."""
        b, c, h, w = msrcr.shape
        flat = msrcr.view(b, c, -1)
        mean = flat.mean(dim=2, keepdim=True).view(b, c, 1, 1)
        std = flat.std(dim=2, keepdim=True).view(b, c, 1, 1)
        normalized = (msrcr - mean) / (std + 1e-6)
        return 0.5 * (1.0 + torch.tanh(normalized * 0.5))

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Normalize shading using Multi-Scale Retinex with Color Restoration.

        Args:
            image: Input image tensor (B, C, H, W).
            **kwargs: Additional parameters.

        Returns:
            ModuleOutput with normalized image.
        """
        channels = image.shape[1]
        luminance = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]) if channels == 3 else image
        original_illum = luminance.mean()

        # Multi-Scale Retinex
        msr = self._multi_scale_retinex(image)

        # Color Restoration
        msrcr = self._color_restoration(image, msr) if self.color_correction and channels == 3 else msr

        # Normalize
        normalized = self._normalize_output(msrcr)

        # Target illumination adjustment
        current_mean = normalized.mean()
        if current_mean > 1e-6:
            normalized = normalized * (self.target_illumination / current_mean)

        # Preserve details by blending
        if self.preserve_details:
            detail = image - self._gaussian_blur(image, 5.0)
            result = 0.7 * (normalized + detail * 0.5) + 0.3 * image
        else:
            result = normalized

        # Gamma correction
        if self.gamma != 1.0:
            result = torch.pow(torch.clamp(result, 0, 1) + 1e-8, self.gamma)

        result = torch.clamp(result, 0.0, 1.0)
        out_lum = (0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]) if channels == 3 else result

        return ModuleOutput(
            result=result,
            intermediate={"input": image, "luminance": luminance, "msr": msr, "normalized": normalized},
            metadata={
                "original_illumination": original_illum.item(),
                "target_illumination": self.target_illumination,
                "output_illumination": out_lum.mean().item(),
                "gamma": self.gamma,
                "scales": self._scales,
                "algorithm": "MSRCR",
            },
        )


__all__ = [
    # Legacy shade normalization
    "ShadeModule",
    # New shade generation components
    "BaseShadeGeneration",
    "LightSource",
    "ShadeConfig",
    "MiDaSDepthEstimator",
    "LightingSimulator",
    "ShadeGeneratorModule",
]
