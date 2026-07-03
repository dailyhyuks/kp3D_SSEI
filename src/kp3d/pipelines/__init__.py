"""Pipeline modules for Korean Painting 3D preprocessing.

This package contains the main pipeline classes that orchestrate
multiple processing modules.

Pipelines:
- IntegratedPipeline: Complete E2E workflow (7 stages) - RECOMMENDED
- PreprocessingPipeline: Image quality enhancement (Restoration → Edge → SR)
- SeparationPipeline: Object separation (Segmentation → Depth → Inpainting)
- UnifiedPipeline: Combined workflow (Separation → Preprocessing per layer)

Note:
    This is the new unified location for all pipelines.
    Legacy imports from kp3d.pipeline and kp3d.modules.occlusion.pipeline
    are still supported but deprecated.
"""

# NEW: Integrated Pipeline (v3 - complete E2E)
from kp3d.pipelines.integrated import (
    IntegratedPipeline,
    PipelineConfig,
    PipelineResult,
    SegmentationMode,
    run_integrated_pipeline
)

# Re-export from legacy locations for backward compatibility
from kp3d.pipeline import Pipeline as PreprocessingPipeline

# Alias for clarity
Pipeline = PreprocessingPipeline  # Deprecated: use PreprocessingPipeline

# Import SeparationPipeline (formerly OcclusionPipeline)
from kp3d.modules.occlusion.pipeline import OcclusionPipeline as SeparationPipeline

# Alias for backward compatibility
OcclusionPipeline = SeparationPipeline  # Deprecated: use SeparationPipeline

# Import UnifiedPipeline (v2)
from kp3d.pipelines.unified import UnifiedPipeline, UnifiedConfig, UnifiedOutput


__all__ = [
    # Integrated Pipeline (v3 - RECOMMENDED)
    "IntegratedPipeline",
    "PipelineConfig",
    "PipelineResult",
    "SegmentationMode",
    "run_integrated_pipeline",

    # Legacy pipelines
    "PreprocessingPipeline",
    "SeparationPipeline",
    "UnifiedPipeline",
    "UnifiedConfig",
    "UnifiedOutput",

    # Deprecated aliases
    "Pipeline",
    "OcclusionPipeline",
]
