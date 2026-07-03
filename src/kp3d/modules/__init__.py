"""Preprocessing modules for Korean painting 3D reconstruction.

Pipeline order (recommended):
0. Weave Removal - Remove silk fabric grid artifacts (spectral interpolation)
1. SuperResolution - Upscale low-res images
2. Restoration - Remove fading noise / pigment degradation
3. Edge Detection - Extract clean edges
4. Shade Generation - Generate shading for 3D
5. Occlusion - Separate overlapping objects and inpaint hidden regions
"""

# SuperResolution
from kp3d.modules.superres.base import BaseSuperResolution, ScaleFactor, SuperResConfig
from kp3d.modules.superres.real_esrgan import RealESRGANModule

# Restoration (NEW)
from kp3d.modules.restoration import RestorationModule
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig
from kp3d.modules.restoration.fading_noise import FadingNoiseRestorer

# Edge Detection
from kp3d.modules.edge import EdgeModule
from kp3d.modules.edge.base import BaseEdgeDetection, EdgeConfig
from kp3d.modules.edge.canny import CannyEdgeDetector
from kp3d.modules.edge.hed import HEDEdgeDetector
from kp3d.modules.edge.korean_ink import KoreanInkEdgeDetector
from kp3d.modules.edge.smart_fusion import SmartFusionDetector

# Shade Generation
from kp3d.modules.shade import ShadeModule
from kp3d.modules.shade.base import BaseShadeGeneration, LightSource, ShadeConfig
from kp3d.modules.shade.midas import MiDaSDepthEstimator
from kp3d.modules.shade.lighting import LightingSimulator, ShadeGeneratorModule

# Occlusion (object separation and inpainting)
from kp3d.modules.occlusion import (
    OcclusionPipeline,
    OcclusionConfig,
    BaseOcclusion,
    LayerInfo,
    OcclusionResult,
    SegmentationModule,
    DepthEstimatorWrapper,
    OcclusionDetector,
    InpaintingModule,
)

# Weave Removal (fabric grid artifact removal)
from kp3d.modules.weave_removal import (
    WeaveRemovalModule, WeaveRemovalConfig, WeaveRemovalPreset,
    WeaveRemovalPipelineModule,
)

# Enhancement Pipeline
from kp3d.modules.enhancement import EnhancementPipeline, EnhancementConfig

# Aliases for backwards compatibility
SuperResModule = RealESRGANModule

__all__ = [
    # Base classes
    "BaseSuperResolution", "BaseRestoration", "BaseEdgeDetection", "BaseShadeGeneration", "BaseOcclusion",
    # Configs
    "SuperResConfig", "RestorationConfig", "EdgeConfig", "ShadeConfig", "LightSource", "ScaleFactor",
    "OcclusionConfig",
    # Implementations
    "RealESRGANModule", "FadingNoiseRestorer",
    "CannyEdgeDetector", "HEDEdgeDetector", "KoreanInkEdgeDetector", "SmartFusionDetector",
    "MiDaSDepthEstimator", "LightingSimulator", "ShadeGeneratorModule",
    # Occlusion
    "OcclusionPipeline", "LayerInfo", "OcclusionResult",
    "SegmentationModule", "DepthEstimatorWrapper", "OcclusionDetector", "InpaintingModule",
    # Enhancement Pipeline
    "EnhancementPipeline", "EnhancementConfig",
    # Weave Removal
    "WeaveRemovalModule", "WeaveRemovalConfig", "WeaveRemovalPreset", "WeaveRemovalPipelineModule",
    # Main modules (registered in their respective __init__.py)
    "SuperResModule", "RestorationModule", "EdgeModule", "ShadeModule",
]
