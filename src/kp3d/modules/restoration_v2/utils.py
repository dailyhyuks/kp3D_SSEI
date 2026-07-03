"""Utility functions for segment-aware restoration.

Provides crop, pad, mask manipulation helpers used by per-object restorers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class CropRegion:
    """Represents a cropped region with coordinates for re-insertion.

    Attributes:
        x1: Left coordinate in original image.
        y1: Top coordinate in original image.
        x2: Right coordinate in original image.
        y2: Bottom coordinate in original image.
        crop_image: Cropped image patch (H, W, 3).
        crop_mask: Cropped binary mask (H, W), same size as crop_image.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    crop_image: np.ndarray
    crop_mask: np.ndarray


def crop_object_region(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 32,
) -> CropRegion:
    """Crop image region around a masked object with padding.

    Args:
        image: Full image (H, W, 3) uint8.
        mask: Binary mask (H, W) where object pixels are non-zero.
        padding: Extra pixels around the bounding box.

    Returns:
        CropRegion with coordinates and cropped patches.

    Raises:
        ValueError: If mask is empty (no non-zero pixels).
    """
    h, w = mask.shape[:2]

    # Find bounding box of mask
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        raise ValueError("Empty mask: no non-zero pixels found")

    y1 = max(0, int(ys.min()) - padding)
    y2 = min(h, int(ys.max()) + 1 + padding)
    x1 = max(0, int(xs.min()) - padding)
    x2 = min(w, int(xs.max()) + 1 + padding)

    crop_image = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

    return CropRegion(
        x1=x1, y1=y1, x2=x2, y2=y2,
        crop_image=crop_image,
        crop_mask=crop_mask,
    )


def paste_crop_back(
    full_image: np.ndarray,
    crop_region: CropRegion,
    restored_crop: np.ndarray,
    blend_mask: np.ndarray,
) -> np.ndarray:
    """Paste a restored crop back into the full image with blending.

    Args:
        full_image: Full output image (H, W, 3) uint8. Modified in-place.
        crop_region: CropRegion describing where to paste.
        restored_crop: Restored image patch (crop_h, crop_w, 3) uint8.
        blend_mask: Float mask (crop_h, crop_w) in [0, 1] for blending.

    Returns:
        The modified full_image (same reference).
    """
    y1, y2 = crop_region.y1, crop_region.y2
    x1, x2 = crop_region.x1, crop_region.x2

    # Ensure mask is 3-channel for broadcasting
    blend_3ch = blend_mask[:, :, np.newaxis].astype(np.float32)

    # Extract current region
    current = full_image[y1:y2, x1:x2].astype(np.float32)
    restored = restored_crop.astype(np.float32)

    # Alpha blend
    blended = current * (1.0 - blend_3ch) + restored * blend_3ch
    full_image[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return full_image


def detect_ink_mask(
    image: np.ndarray,
    l_threshold: float = 40.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Detect ink line regions using LAB L* channel threshold.

    Ink lines in traditional Korean paintings have low luminance (dark).
    Pixels with L* below threshold are considered ink.

    Args:
        image: BGR or RGB image (H, W, 3) uint8.
        l_threshold: L* value below which pixels are ink (0-100 scale).
        mask: Optional binary mask to restrict detection area.

    Returns:
        Binary mask (H, W) uint8 where ink pixels are 255.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float32)

    # OpenCV LAB L* is scaled 0-255, convert threshold
    # L* in OpenCV: 0-255 maps to 0-100 in standard
    l_threshold_cv = l_threshold * 255.0 / 100.0

    ink_mask = (l_channel < l_threshold_cv).astype(np.uint8) * 255

    # Restrict to object region if mask provided
    if mask is not None:
        ink_mask = cv2.bitwise_and(ink_mask, mask)

    return ink_mask


def compute_mask_area(mask: np.ndarray) -> int:
    """Compute the area (number of non-zero pixels) of a binary mask.

    Args:
        mask: Binary mask (H, W).

    Returns:
        Number of non-zero pixels.
    """
    return int(np.count_nonzero(mask))


def normalize_mask_to_float(mask: np.ndarray) -> np.ndarray:
    """Convert a uint8 binary mask to float [0, 1].

    Args:
        mask: Binary mask (H, W) uint8 (0 or 255).

    Returns:
        Float mask (H, W) in [0.0, 1.0].
    """
    return mask.astype(np.float32) / 255.0
