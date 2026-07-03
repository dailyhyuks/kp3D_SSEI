"""Abstract base classes for evaluation baselines."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEnhancementBaseline(ABC):
    """Base class for enhancement (grid/weave removal) baselines.

    Subclasses implement alternative approaches to removing periodic
    grid artifacts from digitized paintings, for comparison with our
    Spectral Interpolation method (WeaveRemovalModule).
    """

    name: str = "base_enhancement"

    @abstractmethod
    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply enhancement/grid removal to an image.

        Args:
            image_bgr: Input BGR image (H, W, 3), uint8.

        Returns:
            Processed image (H, W, 3), uint8.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaseInpaintingBaseline(ABC):
    """Base class for inpainting baselines.

    Subclasses implement alternative inpainting approaches for
    comparison with our SSEI V25 PatchMatch method.
    """

    name: str = "base_inpainting"

    @property
    def available(self) -> bool:
        """Whether this baseline is available (dependencies met)."""
        return True

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint masked region of an image.

        Args:
            image: Input RGB image (H, W, 3), uint8.
            mask: Binary mask (H, W), uint8, 255=region to inpaint.

        Returns:
            Inpainted image (H, W, 3), uint8.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
