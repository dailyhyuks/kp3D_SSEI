"""Weave (fabric grid) artifact removal for digitized Korean paintings.

Removes periodic silk weave patterns from high-resolution scans of
traditional Korean paintings using FFT-based spectral interpolation
with optional spatial-adaptive NLM and contour (ink line) enhancement.

Pipelines:
    QUALITY/CLEAN (legacy):
        1. Patchwise spectral interpolation (grid removal)
        2. Contour enhancement (ink line restoration)

    V3 (recommended, matches experiment variant_r_base + contour):
        1. Split Radius spectral interpolation (grid removal base)
        2. Spatial-adaptive NLM blending (narrow-region targeted)
        3. Contour enhancement (ink line restoration)

Usage Examples:
    # Legacy production (Split Radius + Contour)
    >>> from kp3d.modules.weave_removal import WeaveRemovalModule, WeaveRemovalPreset
    >>> module = WeaveRemovalModule(WeaveRemovalPreset.QUALITY.to_config())
    >>> result, _ = module.process_bgr(image_bgr)

    # V3 (Split Radius + NLM Adaptive + Contour) - 3-stage
    >>> module = WeaveRemovalModule(WeaveRemovalPreset.V3.to_config())
    >>> result, _ = module.process_bgr(image_bgr)

    # Direct NLM adaptive function (Stage 2 only, requires base_processed)
    >>> from kp3d.modules.weave_removal import spatial_adaptive_nlm
    >>> result = spatial_adaptive_nlm(image_bgr, split_out)
"""

from kp3d.modules.weave_removal.base import (
    WeaveRemovalConfig,
    WeaveRemovalModule,
    WeaveRemovalPreset,
)
from kp3d.modules.weave_removal.contour import enhance_contours
from kp3d.modules.weave_removal.spectral import (
    process_image_patchwise,
    spectral_interpolation_single,
)
from kp3d.modules.weave_removal.nlm_adaptive import (
    SpatialAdaptiveNLMConfig,
    spatial_adaptive_nlm,
    compute_narrow_region_mask,
)
from kp3d.core.registry import register_module


@register_module("weave_removal")
class WeaveRemovalPipelineModule(WeaveRemovalModule):
    """Pipeline-compatible weave removal module.

    Registered as 'weave_removal' for pipeline configuration compatibility.
    """
    pass


__all__ = [
    # Config and presets
    "WeaveRemovalConfig",
    "WeaveRemovalPreset",
    # Module classes
    "WeaveRemovalModule",
    "WeaveRemovalPipelineModule",
    # Public functions (spectral)
    "process_image_patchwise",
    "spectral_interpolation_single",
    "enhance_contours",
    # Public functions (V3 NLM adaptive)
    "SpatialAdaptiveNLMConfig",
    "spatial_adaptive_nlm",
    "compute_narrow_region_mask",
]
