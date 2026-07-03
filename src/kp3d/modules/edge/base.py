"""Base classes for edge detection modules."""

from typing import List, Optional
from pydantic import BaseModel, Field

from kp3d.core.base import BasePreprocessModule


class EdgeConfig(BaseModel):
    """Configuration for edge detection modules.

    Attributes:
        low_threshold: Lower threshold for Canny edge detection.
        high_threshold: Upper threshold for Canny edge detection.
        ink_line_weight: Weight multiplier for detecting ink lines (0-2).
        damchae_sensitivity: Sensitivity for detecting soft color transitions (0-1).
        multi_scale: Whether to use multi-scale edge detection.
        scales: Scale factors for multi-scale processing.
        use_hed: Whether to use HED neural network for edge detection.
        internal_detail_weight: Weight for internal detail detection (0-1).
        color_boundary_weight: Weight for color boundary detection (0-1).
        adaptive_threshold: Enable adaptive thresholding for internal details.
    """
    low_threshold: float = Field(default=25.0, ge=0.0, le=255.0)
    high_threshold: float = Field(default=100.0, ge=0.0, le=255.0)
    ink_line_weight: float = Field(default=1.5, ge=0.0, le=2.0)
    damchae_sensitivity: float = Field(default=0.7, ge=0.0, le=1.0)
    multi_scale: bool = True
    scales: List[float] = [1.0, 0.5, 0.25]
    use_hed: bool = True  # HED 신경망 사용 여부
    internal_detail_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    color_boundary_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    adaptive_threshold: bool = True

    class Config:
        """Pydantic config."""
        frozen = False


class BaseEdgeDetection(BasePreprocessModule):
    """Edge Detection 기본 클래스.

    All edge detection modules should inherit from this class.
    Provides common functionality for Korean painting edge detection.
    """

    def __init__(
        self,
        config: Optional[EdgeConfig] = None,
        **kwargs
    ) -> None:
        """Initialize edge detection module.

        Args:
            config: Edge detection configuration.
            **kwargs: Additional arguments passed to BasePreprocessModule.
        """
        super().__init__(**kwargs)
        self.config = config or EdgeConfig()
