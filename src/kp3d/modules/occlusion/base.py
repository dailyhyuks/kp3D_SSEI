"""Base classes and configuration for occlusion handling module."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict

import torch
from torch import Tensor
import numpy as np

from kp3d.core.base import BasePreprocessModule, ModuleOutput


class OcclusionConfig(BaseModel):
    """Configuration for occlusion handling pipeline.

    Attributes:
        # Segmentation settings
        text_prompts: Text prompts for Grounding DINO detection.
        sam_model_type: SAM2 model variant ("vit_b", "vit_l", "vit_h").
        box_threshold: Detection confidence threshold for bounding boxes.
        text_threshold: Text matching threshold for Grounding DINO.

        # Depth settings
        depth_model: MiDaS model type ("DPT_Large", "DPT_Hybrid", "MiDaS_small").

        # Inpainting settings
        inpaint_method: Inpainting algorithm ("telea", "ns", "lama", "sd", "controlnet").
        inpaint_radius: Radius for inpainting algorithm (for telea/ns only).

        # Occlusion detection settings
        dilation_kernel_size: Kernel size for mask dilation.
        dilation_iterations: Number of dilation iterations.
        use_convex_hull: Use convex hull for background boundary.
    """
    model_config = ConfigDict(frozen=True)

    # Segmentation mode
    segmentation_mode: str = "text"  # "text" (Grounding DINO) or "auto" (SAM AMG)

    # Text-based segmentation (Grounding DINO + SAM)
    text_prompts: List[str] = ["white ceramic vase", "red wooden table"]
    sam_model_type: str = "vit_h"
    box_threshold: float = 0.3
    text_threshold: float = 0.25

    # Auto segmentation (SAM AMG)
    auto_seg_points_per_side: int = 32
    auto_seg_pred_iou_thresh: float = 0.86
    auto_seg_stability_thresh: float = 0.90
    auto_seg_min_mask_area: int = 500
    auto_seg_min_area_ratio: float = 0.02
    auto_seg_max_area_ratio: float = 0.7

    # Depth
    depth_model: str = "DPT_Large"

    # Inpainting
    # Options: "telea", "ns", "lama", "lama_guided", "texture_clone", "seamless_clone", "boundary_guided", "patchmatch_guided"
    # lama_guided: LaMa + context color matching (best quality)
    # boundary_guided: V21 - samples colors from occlusion boundary neighborhood (best for adaptive color)
    # patchmatch_guided: V22 - PatchMatch texture propagation with boundary guidance (best texture coherence)
    inpaint_method: str = "boundary_guided"
    inpaint_radius: int = 5

    # Occlusion detection (minimal - closest to annotation)
    dilation_kernel_size: int = 1  # Minimal expansion
    dilation_iterations: int = 1   # Single iteration
    use_convex_hull: bool = False  # Exact annotation polygon

    # Mask refinement (color-based post-processing)
    use_color_refinement: bool = False
    color_refine_ceramic: bool = True
    color_refine_soban: bool = True

    # Auto mask refinement (universal color analysis)
    use_auto_refinement: bool = False
    auto_refine_n_colors: int = 3
    auto_refine_tolerance: int = 30
    auto_refine_min_cluster_ratio: float = 0.05

    # SAM mask refinement (refine LabelMe polygons with SAM)
    use_sam_refinement: bool = False          # Default OFF (existing behaviour unchanged)
    sam_refine_margin_px: int = 15            # Margin constraint (pixels) — reference at 512px
    sam_refine_min_area_ratio: float = 0.3    # Safety: keep original if refined < 30% of rough
    sam_refine_erode_px: int = 10             # Erosion radius for positive point sampling
    sam_refine_dilate_px: int = 15            # Dilation radius for negative point sampling
    sam_refine_adaptive: bool = True          # Scale px params by object size (ref=512px)

    # Inpaint mask source toggle (v6 ablation)
    use_refined_mask_for_inpaint: bool = False  # If True, use SAM-refined polygon for inpaint mask
    inpaint_mask_dilate_px: int = 0             # Pixels to dilate refined mask before inpainting (only when use_refined_mask_for_inpaint=True)

    # Hybrid inpainting settings
    use_hybrid_inpainting: bool = True
    enable_symmetry_inpainting: bool = True
    enable_reference_guided: bool = True

    # Segment-aware restoration (per-object, applied after SAM refinement)
    use_segment_aware_restoration: bool = False
    segment_aware_restoration_config: Optional[Dict[str, Any]] = None

    # Ablation study toggles
    skip_occlusion_detection: bool = False  # Skip depth-based occlusion detection
    skip_inpainting: bool = False           # Skip inpainting of occluded regions


@dataclass
class LayerInfo:
    """Information about a segmented layer/object.

    Attributes:
        label: Text label for the object.
        mask: Binary mask (H, W) as numpy array.
        bbox: Bounding box as (x1, y1, x2, y2).
        mean_depth: Average depth value for ordering.
        is_foreground: Whether this layer is in foreground.
    """
    label: str
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    mean_depth: float = 0.0
    is_foreground: bool = False


@dataclass
class OcclusionResult:
    """Result from occlusion pipeline.

    Attributes:
        foreground_image: Extracted foreground object (RGBA).
        background_inpainted: Background with occluded regions filled.
        foreground_mask: Binary mask of foreground.
        background_mask: Binary mask of background.
        occlusion_mask: Mask of occluded regions.
        depth_map: Full depth map.
        layers: List of layer information.
        metadata: Additional processing metadata.
    """
    foreground_image: np.ndarray
    background_inpainted: np.ndarray
    foreground_mask: np.ndarray
    background_mask: np.ndarray
    occlusion_mask: np.ndarray
    depth_map: np.ndarray
    layers: List[LayerInfo] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOcclusion(BasePreprocessModule):
    """Base class for occlusion handling modules.

    Provides common functionality for object separation and inpainting,
    designed for multi-layer 3D reconstruction of traditional paintings.
    """

    def __init__(
        self,
        config: Optional[OcclusionConfig] = None,
        **kwargs
    ):
        """Initialize occlusion module.

        Args:
            config: Occlusion handling configuration.
            **kwargs: Additional arguments passed to BasePreprocessModule.
        """
        super().__init__(**kwargs)
        self.config = config or OcclusionConfig()

    @property
    def name(self) -> str:
        """Module name."""
        return "occlusion"

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights (implemented by subclasses if needed)."""
        pass

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Process image through occlusion pipeline.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")
