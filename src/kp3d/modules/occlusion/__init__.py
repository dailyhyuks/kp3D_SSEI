"""Occlusion handling module for object separation and inpainting.

This module provides tools for separating overlapping objects in images
and inpainting occluded regions, enabling multi-layer 3D reconstruction
of traditional paintings.

Pipeline Stages:
1. Segmentation: SAM2 + Grounding DINO for text-prompted object detection
2. Depth: MiDaS for monocular depth estimation
3. Layer Ordering: Determine foreground/background relationships
4. Occlusion Detection: Find hidden regions
5. Inpainting: Fill occluded areas with cv2.inpaint
6. Output: Generate separated object images

Example:
    >>> from kp3d.modules.occlusion import OcclusionPipeline, OcclusionConfig
    >>>
    >>> config = OcclusionConfig(
    ...     text_prompts=["white ceramic vase", "red wooden table"],
    ...     inpaint_method="telea"
    ... )
    >>> pipeline = OcclusionPipeline(config=config)
    >>>
    >>> # Process image
    >>> output = pipeline(image_tensor)
    >>>
    >>> # Access results
    >>> background_inpainted = output.result
    >>> foreground = output.intermediate["foreground"]
"""

from kp3d.modules.occlusion.base import (
    OcclusionConfig,
    BaseOcclusion,
    LayerInfo,
    OcclusionResult
)
from kp3d.modules.occlusion.segmentation import (
    SegmentationModule,
    create_manual_mask
)
from kp3d.modules.occlusion.depth import DepthEstimatorWrapper
from kp3d.modules.occlusion.layer_ordering import (
    LayerOrderingModule,
    SimpleLayerOrdering
)
from kp3d.modules.occlusion.occlusion_detection import (
    OcclusionDetector,
    # New layered occlusion detection
    LayeredOcclusionResult,
    LayeredOcclusionDetector,
    OcclusionRelation,
    get_layer_priority,
    get_base_layer,
    LAYER_PRIORITY,
    # Layer order support (paper spec alignment)
    extract_layer_order_from_shape,
    layer_order_to_priority,
    LAYER_ORDER_TO_PRIORITY_BASE,
    # Legacy aliases
    SimpleOcclusionResult,
    SimpleLabelOcclusionDetector,
    # Utilities
    quick_occlusion_mask,
    predict_inpaint_regions,
    find_plate_region,
    detect_rim_occlusion,
    auto_detect_rim_region,
    masks_overlap
)
from kp3d.modules.occlusion.inpainting import (
    InpaintingModule,
    LamaInpainter,
    SDInpainter,
    quick_inpaint,
    quick_lama_inpaint,
    quick_sd_inpaint
)
from kp3d.modules.occlusion.mask_refinement import (
    MaskRefiner,
    refine_masks_for_pipeline
)
from kp3d.modules.occlusion.auto_mask_refinement import (
    AutoMaskRefiner,
    auto_refine_masks
)
from kp3d.modules.occlusion.auto_segmentation import (
    AutoSegmentation,
    auto_segment_and_refine
)
from kp3d.modules.occlusion.grounded_sam import (
    GroundedSAM,
    grounded_segment
)
from kp3d.modules.occlusion.tiled_segmentation import (
    TiledSegmentation,
    TileConfig,
    DetectedObject,
    tiled_segment,
    visualize_detections
)
from kp3d.modules.occlusion.symmetry_inpaint import (
    SymmetryDetector,
    PatchMatchInpainter,
    SymmetryGuidedInpainter,
    symmetry_guided_inpaint,
    detect_symmetry_axis
)
from kp3d.modules.occlusion.reference_guided_inpaint import (
    StyleFeatureExtractor,
    ReferenceGuidedInpainter,
    reference_guided_inpaint,
    match_histograms
)
from kp3d.modules.occlusion.hybrid_inpainter import (
    InpaintingStrategy,
    RegionAnalyzer,
    HybridInpainter,
    hybrid_inpaint
)
from kp3d.modules.occlusion.sam_mask_refiner import SAMMaskRefiner
from kp3d.modules.occlusion.pipeline import (
    OcclusionPipeline,
    run_pipeline
)


__all__ = [
    # Configuration
    "OcclusionConfig",

    # Base classes
    "BaseOcclusion",
    "LayerInfo",
    "OcclusionResult",

    # Stage 1: Segmentation
    "SegmentationModule",
    "create_manual_mask",

    # Stage 2: Depth
    "DepthEstimatorWrapper",

    # Stage 3: Layer Ordering
    "LayerOrderingModule",
    "SimpleLayerOrdering",

    # Stage 4: Occlusion Detection
    "OcclusionDetector",
    # New layered system
    "LayeredOcclusionResult",
    "LayeredOcclusionDetector",
    "OcclusionRelation",
    "get_layer_priority",
    "get_base_layer",
    "LAYER_PRIORITY",
    # Layer order support (paper spec)
    "extract_layer_order_from_shape",
    "layer_order_to_priority",
    "LAYER_ORDER_TO_PRIORITY_BASE",
    # Legacy aliases
    "SimpleOcclusionResult",
    "SimpleLabelOcclusionDetector",
    # Utilities
    "quick_occlusion_mask",
    "predict_inpaint_regions",
    "find_plate_region",
    "detect_rim_occlusion",
    "auto_detect_rim_region",
    "masks_overlap",

    # Stage 5: Inpainting
    "InpaintingModule",
    "LamaInpainter",
    "SDInpainter",
    "quick_inpaint",
    "quick_lama_inpaint",
    "quick_sd_inpaint",

    # Stage 6: Mask Refinement (Color-based)
    "MaskRefiner",
    "refine_masks_for_pipeline",

    # Stage 6b: Auto Mask Refinement (Universal)
    "AutoMaskRefiner",
    "auto_refine_masks",

    # Auto Segmentation (SAM AMG)
    "AutoSegmentation",
    "auto_segment_and_refine",

    # Grounded SAM (Text-guided)
    "GroundedSAM",
    "grounded_segment",

    # Tiled Segmentation (High-resolution)
    "TiledSegmentation",
    "TileConfig",
    "DetectedObject",
    "tiled_segment",
    "visualize_detections",

    # Symmetry-Guided Inpainting
    "SymmetryDetector",
    "PatchMatchInpainter",
    "SymmetryGuidedInpainter",
    "symmetry_guided_inpaint",
    "detect_symmetry_axis",

    # Reference-Guided Inpainting
    "StyleFeatureExtractor",
    "ReferenceGuidedInpainter",
    "reference_guided_inpaint",
    "match_histograms",

    # Hybrid Inpainting
    "InpaintingStrategy",
    "RegionAnalyzer",
    "HybridInpainter",
    "hybrid_inpaint",

    # SAM Mask Refinement
    "SAMMaskRefiner",

    # Complete Pipeline
    "OcclusionPipeline",
    "run_pipeline",
]
