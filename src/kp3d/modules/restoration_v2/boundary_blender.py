"""Boundary blending for seamless per-object restoration compositing.

Applies Gaussian feathering to mask boundaries to avoid hard seams
when pasting restored crops back into the full image.
"""
from __future__ import annotations

import cv2
import numpy as np

from kp3d.modules.restoration_v2.config import SegmentAwareRestorationConfig


class BoundaryBlender:
    """Creates feathered blend masks for seamless object compositing.

    Applies Gaussian blur to binary masks to create soft transitions
    at object boundaries, preventing visible seams between restored
    and non-restored regions.
    """

    def __init__(self, config: SegmentAwareRestorationConfig) -> None:
        """Initialize boundary blender.

        Args:
            config: Restoration configuration with feather_radius_px.
        """
        self.config = config

    def create_feathered_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """Create a feathered (soft-edge) blend mask from binary mask.

        Applies Gaussian blur to the mask edges, creating a smooth
        transition zone at the boundary.

        Args:
            binary_mask: Binary mask (H, W) uint8, 0 or 255.

        Returns:
            Float mask (H, W) in [0.0, 1.0] with feathered edges.
        """
        radius = self.config.feather_radius_px

        if radius <= 0:
            # No feathering, return hard mask
            return binary_mask.astype(np.float32) / 255.0

        # Gaussian kernel size must be odd
        kernel_size = radius * 2 + 1

        # Convert to float for blur
        mask_float = binary_mask.astype(np.float32) / 255.0

        # Apply Gaussian blur for feathering
        feathered = cv2.GaussianBlur(
            mask_float,
            (kernel_size, kernel_size),
            sigmaX=radius / 2.0,
        )

        return feathered

    def create_ink_protection_mask(
        self,
        ink_mask: np.ndarray,
        protection_strength: float = 0.9,
    ) -> np.ndarray:
        """Create ink protection blend mask.

        Regions identified as ink will have high protection values,
        meaning the original (unprocessed) pixels are preserved.

        Args:
            ink_mask: Binary mask of ink pixels (H, W) uint8.
            protection_strength: How much to protect ink (0.0-1.0).
                1.0 means fully protect (original preserved).

        Returns:
            Float mask (H, W) where 1.0 = fully replace with restored,
            0.0 = keep original. Ink regions will be close to 0.
        """
        ink_float = ink_mask.astype(np.float32) / 255.0

        # Invert: ink regions → low blend (keep original)
        # Non-ink regions → high blend (use restored)
        blend_mask = 1.0 - (ink_float * protection_strength)

        return blend_mask

    def composite_with_ink_protection(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        object_mask: np.ndarray,
        ink_mask: np.ndarray,
    ) -> np.ndarray:
        """Composite restored result with ink line protection.

        Combines feathered object boundary blending with ink protection:
        - Object boundary: feathered transition
        - Inside object (non-ink): use restored
        - Ink regions: preserve original

        Args:
            original: Original cropped image (H, W, 3) uint8.
            restored: Restored cropped image (H, W, 3) uint8.
            object_mask: Binary object mask (H, W) uint8.
            ink_mask: Binary ink mask (H, W) uint8.

        Returns:
            Composited image (H, W, 3) uint8.
        """
        # Feathered object mask
        feathered_object = self.create_feathered_mask(object_mask)

        # Ink protection (within object)
        ink_protection = self.create_ink_protection_mask(
            ink_mask, self.config.ink_protection_strength
        )

        # Combined blend mask: feathered boundary AND ink protection
        blend_mask = feathered_object * ink_protection
        blend_3ch = blend_mask[:, :, np.newaxis]

        # Composite
        result = (
            original.astype(np.float32) * (1.0 - blend_3ch)
            + restored.astype(np.float32) * blend_3ch
        )
        return np.clip(result, 0, 255).astype(np.uint8)
