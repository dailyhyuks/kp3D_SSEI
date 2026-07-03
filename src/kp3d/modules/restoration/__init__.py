"""Restoration module for Korean painting preprocessing.

Provides image restoration optimized for traditional Korean paintings,
including fading noise removal and pigment degradation correction.
"""

from typing import Any, Optional

import torch
from torch import Tensor

from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.core.registry import register_module
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig
from kp3d.modules.restoration.fading_noise import FadingNoiseRestorer
from kp3d.modules.restoration.frequency_aware import FrequencyAwareRestorer
from kp3d.modules.restoration.grid_pattern import GridPatternRestorer
from kp3d.modules.restoration.enhanced_fading_noise import EnhancedFadingNoiseRestorer
from kp3d.modules.restoration.enhanced_frequency_aware import EnhancedFrequencyAwareRestorer
from kp3d.modules.restoration.enhanced_grid_pattern import EnhancedGridPatternRestorer
from kp3d.modules.restoration.hybrid_grid_pattern import HybridGridPatternRestorer
from kp3d.modules.restoration.contour_flattening import ContourFlatteningRestorer
from kp3d.modules.restoration.edge_aware_flat_restorer import EdgeAwareFlatRestorer
from kp3d.modules.restoration.color_quantization_restorer import ColorQuantizationRestorer
from kp3d.modules.restoration.deep_grid_restorer import DeepGridRestorer


@register_module("restoration")
class RestorationModule(BasePreprocessModule):
    """Restoration module for Korean paintings.

    Removes fading noise (pigment degradation) while preserving
    actual painting content and edges.
    """

    def __init__(
        self,
        method: str = "fading_noise",
        config: Optional[RestorationConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize restoration module.

        Args:
            method: Restoration method - "fading_noise", "frequency_aware", "grid_pattern",
                "enhanced_fading_noise", "enhanced_frequency_aware", or "enhanced_grid_pattern".
            config: Restoration configuration.
            device: Compute device.
            **kwargs: Additional parameters.
        """
        super().__init__(device=device)
        self.method = method
        self.config = config or RestorationConfig()

        restorer_kwargs = {
            "config": self.config,
            "device": self.device,
            "dtype": self.dtype,
            **kwargs
        }

        if method == "fading_noise":
            self.restorer = FadingNoiseRestorer(**restorer_kwargs)
        elif method == "frequency_aware":
            self.restorer = FrequencyAwareRestorer(**restorer_kwargs)
        elif method == "grid_pattern":
            self.restorer = GridPatternRestorer(**restorer_kwargs)
        elif method == "enhanced_fading_noise":
            self.restorer = EnhancedFadingNoiseRestorer(**restorer_kwargs)
        elif method == "enhanced_frequency_aware":
            self.restorer = EnhancedFrequencyAwareRestorer(**restorer_kwargs)
        elif method == "enhanced_grid_pattern":
            self.restorer = EnhancedGridPatternRestorer(**restorer_kwargs)
        elif method == "hybrid_grid_pattern":
            self.restorer = HybridGridPatternRestorer(**restorer_kwargs)
        elif method == "contour_flattening":
            self.restorer = ContourFlatteningRestorer(**restorer_kwargs)
        elif method == "edge_aware_flat":
            self.restorer = EdgeAwareFlatRestorer(**restorer_kwargs)
        elif method == "color_quantization":
            self.restorer = ColorQuantizationRestorer(**restorer_kwargs)
        elif method == "deep_grid":
            self.restorer = DeepGridRestorer(**restorer_kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'fading_noise', 'frequency_aware', "
                f"'grid_pattern', 'enhanced_fading_noise', 'enhanced_frequency_aware', "
                f"'enhanced_grid_pattern', 'hybrid_grid_pattern', 'contour_flattening', "
                f"'edge_aware_flat', 'color_quantization', or 'deep_grid'."
            )

        self._initialized = self.restorer.is_initialized

    @property
    def name(self) -> str:
        return f"restoration_{self.method}"

    def load_weights(self, checkpoint_path: str) -> None:
        self.restorer.load_weights(checkpoint_path)
        self._initialized = self.restorer.is_initialized

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        return self.restorer.forward(image, **kwargs)


__all__ = [
    "RestorationModule",
    "RestorationConfig",
    "BaseRestoration",
    "FadingNoiseRestorer",
    "FrequencyAwareRestorer",
    "GridPatternRestorer",
    "EnhancedFadingNoiseRestorer",
    "EnhancedFrequencyAwareRestorer",
    "EnhancedGridPatternRestorer",
    "HybridGridPatternRestorer",
    "ContourFlatteningRestorer",
    "EdgeAwareFlatRestorer",
    "ColorQuantizationRestorer",
    "DeepGridRestorer",
]
