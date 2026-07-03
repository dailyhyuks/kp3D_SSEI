"""Base classes for shade generation module."""

from typing import List, Optional, Tuple
from pydantic import BaseModel

from kp3d.core.base import BasePreprocessModule


class LightSource(BaseModel):
    """Light source configuration.

    Attributes:
        direction: Light direction as (x, y, z) unit vector.
        intensity: Light intensity multiplier (0-inf).
        color: Light color as (R, G, B) normalized values (0-1).
        ambient: Ambient light contribution (0-1).
    """
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    intensity: float = 1.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ambient: float = 0.2

    class Config:
        frozen = True


class ShadeConfig(BaseModel):
    """Configuration for shade generation.

    Attributes:
        depth_model: MiDaS model type to use.
        light_sources: List of light sources for shading.
        normal_smoothing: Smoothing factor for normal map (0-1).
        shadow_softness: Shadow edge softness (0-1).
        preserve_original_tones: Blend with original image to preserve artistic style.
        shade_intensity: Overall shading intensity (0-1).
    """
    depth_model: str = "DPT_Large"
    light_sources: List[LightSource] = [LightSource()]
    normal_smoothing: float = 0.5
    shadow_softness: float = 0.3
    preserve_original_tones: bool = True
    shade_intensity: float = 0.5  # 명암 강도 (0~1)

    class Config:
        frozen = True


class BaseShadeGeneration(BasePreprocessModule):
    """Base class for shade generation modules.

    Provides common functionality for depth-based shading generation,
    optimized for traditional Korean painting aesthetics.
    """

    def __init__(self, config: Optional[ShadeConfig] = None, **kwargs):
        """Initialize shade generation module.

        Args:
            config: Shade generation configuration.
            **kwargs: Additional arguments passed to BasePreprocessModule.
        """
        super().__init__(**kwargs)
        self.config = config or ShadeConfig()

    @property
    def name(self) -> str:
        """Module name."""
        return "shade_generation"
