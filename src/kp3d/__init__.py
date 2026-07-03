"""
KP3D - Korean Painting 3D Preprocessing Pipeline

A comprehensive preprocessing system for converting traditional Korean paintings
into 3D-reconstruction-ready formats.
"""

from kp3d.version import __version__
from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.core.registry import ModuleRegistry
from kp3d.core.device import DeviceManager
from kp3d.core.config import PipelineConfig

# Import pipelines
from kp3d.pipelines import (
    PreprocessingPipeline,
    SeparationPipeline,
    UnifiedPipeline,
    UnifiedConfig,
    UnifiedOutput,
)

__all__ = [
    "__version__",
    # Core
    "BasePreprocessModule",
    "ModuleOutput",
    "ModuleRegistry",
    "DeviceManager",
    "PipelineConfig",
    # Pipelines
    "PreprocessingPipeline",
    "SeparationPipeline",
    "UnifiedPipeline",
    "UnifiedConfig",
    "UnifiedOutput",
]
