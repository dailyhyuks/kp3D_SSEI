"""Segment-Aware Restorer - main orchestrator.

Applies independent per-object restoration to each foreground segment,
avoiding background grid interference and cross-object color bleeding.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from kp3d.modules.restoration_v2.config import SegmentAwareRestorationConfig
from kp3d.modules.restoration_v2.per_object_restorer import PerObjectRestorer
from kp3d.modules.restoration_v2.boundary_blender import BoundaryBlender
from kp3d.modules.restoration_v2.utils import (
    CropRegion,
    compute_mask_area,
    crop_object_region,
    paste_crop_back,
)

logger = logging.getLogger(__name__)


@dataclass
class RestorationResult:
    """Result from segment-aware restoration.

    Attributes:
        restored_image: Full image with all objects restored (H, W, 3) uint8.
        objects_processed: Number of objects that were restored.
        objects_skipped: Number of objects skipped (too small, background, etc).
        per_object_times: Dict mapping label to processing time in seconds.
        metadata: Additional metadata.
    """
    restored_image: np.ndarray
    objects_processed: int = 0
    objects_skipped: int = 0
    per_object_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectMask:
    """A labeled object mask for restoration.

    Attributes:
        label: Object label/name from annotation.
        mask: Binary mask (H, W) uint8 where object pixels are 255.
        is_background: Whether this is a background segment.
    """
    label: str
    mask: np.ndarray
    is_background: bool = False


class SegmentAwareRestorer:
    """Orchestrates per-object restoration across all foreground segments.

    Workflow:
    1. Iterate over all object masks
    2. Skip background objects (if configured)
    3. Skip objects below minimum area threshold
    4. For each valid object: crop -> restore -> feather blend -> paste
    5. Return fully composited image

    The key advantage over global restoration (v13): each object is
    processed independently, eliminating cross-object interference.
    """

    def __init__(
        self,
        config: Optional[SegmentAwareRestorationConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize segment-aware restorer.

        Args:
            config: Restoration configuration. Uses defaults if None.
            device: Torch device for neural components.
        """
        self.config = config or SegmentAwareRestorationConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.per_object_restorer = PerObjectRestorer(self.config, device=self.device)
        self.boundary_blender = BoundaryBlender(self.config)

    def restore_all_objects(
        self,
        image: np.ndarray,
        object_masks: List[ObjectMask],
    ) -> RestorationResult:
        """Restore all foreground objects in the image.

        Processes each object independently: crop, restore, blend back.
        Background objects are skipped if skip_background is True.

        Args:
            image: Full image (H, W, 3) uint8 BGR.
            object_masks: List of ObjectMask with labels and binary masks.

        Returns:
            RestorationResult with the restored image and metadata.
        """
        start_time = time.time()
        result_image = image.copy()
        objects_processed = 0
        objects_skipped = 0
        per_object_times: Dict[str, float] = {}

        print(f"  [Segment-Aware Restoration] Processing {len(object_masks)} objects...")

        for obj in object_masks:
            # Skip background
            if self.config.skip_background and obj.is_background:
                logger.debug(f"Skipping background object: {obj.label}")
                objects_skipped += 1
                continue

            # Check minimum area
            area = compute_mask_area(obj.mask)
            if area < self.config.min_object_area_px:
                logger.debug(
                    f"Skipping small object: {obj.label} "
                    f"(area={area} < min={self.config.min_object_area_px})"
                )
                objects_skipped += 1
                continue

            # Process this object
            obj_start = time.time()
            try:
                result_image = self._restore_single_object(
                    result_image, obj.mask, obj.label
                )
                objects_processed += 1
            except Exception as e:
                logger.warning(f"Failed to restore object '{obj.label}': {e}")
                objects_skipped += 1

            per_object_times[obj.label] = time.time() - obj_start
            print(f"    {obj.label}: {per_object_times[obj.label]:.2f}s (area={area}px)")

        total_time = time.time() - start_time
        print(
            f"  [Segment-Aware Restoration] Done: "
            f"{objects_processed} restored, {objects_skipped} skipped, "
            f"total {total_time:.2f}s"
        )

        return RestorationResult(
            restored_image=result_image,
            objects_processed=objects_processed,
            objects_skipped=objects_skipped,
            per_object_times=per_object_times,
            metadata={
                "total_time": total_time,
                "config": {
                    "fading_method": self.config.fading_method,
                    "neural_strength": self.config.neural_strength,
                    "feather_radius_px": self.config.feather_radius_px,
                },
            },
        )

    def _restore_single_object(
        self,
        full_image: np.ndarray,
        mask: np.ndarray,
        label: str,
    ) -> np.ndarray:
        """Restore a single object and blend back into full image.

        Args:
            full_image: Full image (H, W, 3) uint8 BGR. Will be modified.
            mask: Binary mask of this object (H, W) uint8.
            label: Object label for logging.

        Returns:
            Modified full_image with this object restored.
        """
        # Crop object region with padding
        crop = crop_object_region(
            full_image, mask, padding=self.config.crop_padding_px
        )

        # Run per-object restoration pipeline
        restored_crop = self.per_object_restorer.restore_object(
            crop.crop_image, crop.crop_mask
        )

        # Create feathered blend mask
        blend_mask = self.boundary_blender.create_feathered_mask(crop.crop_mask)

        # Paste back with blending
        paste_crop_back(full_image, crop, restored_crop, blend_mask)

        return full_image
