"""Segment-Aware Restoration Module (v2).

Applies independent per-object restoration after segmentation,
eliminating background grid interference and cross-object color bleeding.

Pipeline position: Upscaling -> Segmentation -> SAM Refine -> **Restoration v2** -> Depth

Usage:
    from kp3d.modules.restoration_v2 import SegmentAwareRestorer, SegmentAwareRestorationConfig

    config = SegmentAwareRestorationConfig(neural_strength=0.3)
    restorer = SegmentAwareRestorer(config=config)
    result = restorer.restore_all_objects(image, object_masks)
    restored_image = result.restored_image
"""

from kp3d.modules.restoration_v2.config import SegmentAwareRestorationConfig
from kp3d.modules.restoration_v2.segment_aware_restorer import (
    ObjectMask,
    RestorationResult,
    SegmentAwareRestorer,
)

__all__ = [
    "SegmentAwareRestorer",
    "SegmentAwareRestorationConfig",
    "ObjectMask",
    "RestorationResult",
]
