"""Edge enhancement module for Korean painting preprocessing.

Provides edge detection and enhancement optimized for the line work
characteristics of traditional Korean paintings.

This module includes:
- Canny edge detection with multi-scale support
- HED (Holistically-nested Edge Detection) deep learning detector
- Korean ink specialized detector (먹선/담채)
- Advanced LAB fusion detector with morphological denoising
- Smart fusion detector (구조+색상 융합, 연결성분 분석) - RECOMMENDED
"""

from typing import Any, Optional

import torch
from torch import Tensor

from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.core.registry import register_module
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig
from kp3d.modules.edge.canny import CannyEdgeDetector
from kp3d.modules.edge.hed import HEDEdgeDetector
from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
from kp3d.modules.edge.advanced_edge import AdvancedEdgeDetector
from kp3d.modules.edge.smart_fusion import SmartFusionDetector
from kp3d.modules.edge.resynthesizer import EdgeResynthesizer, EdgeResynthesizerConfig
from kp3d.modules.edge.color_edge_inference import ColorEdgeInference, ColorEdgeConfig


@register_module("edge")
class EdgeModule(BasePreprocessModule):
    """Edge enhancement module using Canny, HED, or Korean ink detection.

    Detects and enhances edges/lines in paintings while preserving
    the artistic characteristics of brush strokes.

    This is a unified interface that selects the appropriate detector
    based on configuration.
    """

    def __init__(
        self,
        method: str = "smart_fusion",
        threshold: float = 0.5,
        config: Optional[EdgeConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize edge enhancement module.

        Args:
            method: Detection method - "smart_fusion" (recommended), "canny", "hed",
                   "korean_ink", or "advanced".
            threshold: Edge detection threshold (0-1).
            config: Edge detection configuration. If None, uses defaults.
            device: Compute device.
            **kwargs: Additional parameters passed to specific detector.
        """
        super().__init__(device=device)
        self.method = method
        self.threshold = threshold
        self.config = config or EdgeConfig()

        # Initialize the appropriate detector
        detector_kwargs = {
            "config": self.config,
            "device": self.device,
            "dtype": self.dtype,
            **kwargs
        }

        if method == "canny":
            self.detector = CannyEdgeDetector(**detector_kwargs)
        elif method == "hed":
            self.detector = HEDEdgeDetector(**detector_kwargs)
        elif method == "korean_ink":
            self.detector = KoreanInkEdgeDetector(**detector_kwargs)
        elif method == "advanced":
            self.detector = AdvancedEdgeDetector(**detector_kwargs)
        elif method == "smart_fusion":
            self.detector = SmartFusionDetector(**detector_kwargs)
        elif method == "resynth":
            self.detector = EdgeResynthesizer(**detector_kwargs)
        elif method == "color_inference":
            self.detector = ColorEdgeInference(**detector_kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'smart_fusion', 'canny', 'hed', 'korean_ink', 'advanced', 'resynth', 'color_inference'")

        self._initialized = self.detector.is_initialized

    @property
    def name(self) -> str:
        """Module name."""
        return f"edge_{self.method}"

    def load_weights(self, checkpoint_path: str) -> None:
        """Load pretrained weights.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        self.detector.load_weights(checkpoint_path)
        self._initialized = self.detector.is_initialized

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Detect and enhance edges in an image.

        Args:
            image: Input image tensor (B, C, H, W).
            **kwargs: Additional parameters passed to detector.

        Returns:
            ModuleOutput with edge map and enhanced image.
        """
        # Pass threshold if not in kwargs
        if "threshold" not in kwargs:
            kwargs["threshold"] = self.threshold

        return self.detector.forward(image, **kwargs)


__all__ = [
    "EdgeModule",
    "EdgeConfig",
    "BaseEdgeDetection",
    "CannyEdgeDetector",
    "HEDEdgeDetector",
    "KoreanInkEdgeDetector",
    "AdvancedEdgeDetector",
    "SmartFusionDetector",
    "EdgeResynthesizer",
    "EdgeResynthesizerConfig",
    "ColorEdgeInference",
    "ColorEdgeConfig",
]
