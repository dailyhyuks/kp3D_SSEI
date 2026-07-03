"""Per-object restoration pipeline.

Processes a single segmented object through the full restoration chain:
ink detection -> morphological cleanup -> fading/grid restoration ->
denoising -> color normalization -> ink protection merge.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from kp3d.modules.restoration_v2.config import SegmentAwareRestorationConfig
from kp3d.modules.restoration_v2.fading_restorer import FadingRestorer
from kp3d.modules.restoration_v2.boundary_blender import BoundaryBlender
from kp3d.modules.restoration_v2.utils import detect_ink_mask

logger = logging.getLogger(__name__)


class PerObjectRestorer:
    """Restores a single cropped object through the full pipeline.

    Pipeline steps:
    1. Ink line detection (LAB L* threshold, lower = less over-protection)
    2. Morphological cleanup of ink mask (remove noise pixels)
    3. Fading/grid restoration (iterative bilateral + guided filter + neural)
    4. Denoising (Non-local means)
    5. Color normalization (CLAHE on L channel, aggressive for faded pigments)
    6. Ink line protection merge (ink regions keep original)

    The canvas grid texture is physically embedded in painted objects,
    so aggressive texture smoothing is needed within each object mask.
    """

    def __init__(
        self,
        config: SegmentAwareRestorationConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.fading_restorer = FadingRestorer(config, device=self.device)
        self.boundary_blender = BoundaryBlender(config)

    def _clean_ink_mask(self, ink_mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening to remove noise from ink mask.

        Small isolated pixels incorrectly classified as ink are removed,
        keeping only connected ink strokes.
        """
        k = self.config.ink_morph_open_size
        if k <= 0:
            return ink_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)

    def _denoise(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising within mask."""
        # API: fastNlMeansDenoisingColored(src, dst, h, hColor, templateWindowSize, searchWindowSize)
        denoised = cv2.fastNlMeansDenoisingColored(
            image_bgr,
            None,
            self.config.denoise_h,
            self.config.denoise_h,
            self.config.denoise_template_window,
            self.config.denoise_search_window,
        )

        # Apply only within mask
        mask_float = mask.astype(np.float32) / 255.0
        mask_3ch = mask_float[:, :, np.newaxis]
        result = (
            image_bgr.astype(np.float32) * (1.0 - mask_3ch)
            + denoised.astype(np.float32) * mask_3ch
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_clahe(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply CLAHE color normalization on L channel.

        Uses aggressive settings (high clip_limit, small grid) to
        revive faded pigment colors.
        """
        if not self.config.use_clahe:
            return image_bgr

        # Convert to LAB
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
        )
        l_enhanced = clahe.apply(l_channel)

        # Merge back
        lab[:, :, 0] = l_enhanced
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Apply only within mask
        mask_float = mask.astype(np.float32) / 255.0
        mask_3ch = mask_float[:, :, np.newaxis]
        result = (
            image_bgr.astype(np.float32) * (1.0 - mask_3ch)
            + enhanced.astype(np.float32) * mask_3ch
        )
        return np.clip(result, 0, 255).astype(np.uint8)

    def restore_object(
        self,
        crop_image: np.ndarray,
        crop_mask: np.ndarray,
    ) -> np.ndarray:
        """Run the full per-object restoration pipeline.

        Args:
            crop_image: Cropped object image (H, W, 3) uint8 BGR.
            crop_mask: Binary mask of object within crop (H, W) uint8.

        Returns:
            Restored cropped image (H, W, 3) uint8 BGR.
        """
        original = crop_image.copy()

        # Step 1: Ink line detection (low threshold = only real ink strokes)
        ink_mask = detect_ink_mask(
            crop_image,
            l_threshold=self.config.ink_l_threshold,
            mask=crop_mask,
        )

        # Step 2: Morphological cleanup (remove scattered noise pixels from ink mask)
        ink_mask = self._clean_ink_mask(ink_mask)

        obj_area = np.count_nonzero(crop_mask)
        ink_area = np.count_nonzero(ink_mask)
        logger.debug(
            f"Ink detection: {ink_area} ink pixels / {obj_area} object pixels "
            f"({ink_area / max(obj_area, 1) * 100:.1f}%)"
        )

        # Step 3: Fading/grid restoration (iterative bilateral + guided + neural)
        restored = self.fading_restorer.restore(crop_image, crop_mask, ink_mask)

        # Step 4: Denoising (within object mask excluding ink)
        denoise_mask = cv2.bitwise_and(crop_mask, cv2.bitwise_not(ink_mask))
        restored = self._denoise(restored, denoise_mask)

        # Step 5: Color normalization (CLAHE, aggressive for faded pigments)
        restored = self._apply_clahe(restored, denoise_mask)

        # Step 6: Ink line protection merge + feathered boundary
        result = self.boundary_blender.composite_with_ink_protection(
            original, restored, crop_mask, ink_mask
        )

        return result
