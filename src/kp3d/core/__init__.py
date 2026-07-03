"""Core infrastructure for KP3D preprocessing pipeline."""

from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.core.registry import ModuleRegistry, register_module, get_module
from kp3d.core.device import DeviceManager, get_optimal_device
from kp3d.core.config import (
    PipelineConfig,
    SuperResConfig,
    EdgeConfig,
    ShadeConfig,
    OutputConfig,
)

__all__ = [
    # Base classes
    "BasePreprocessModule",
    "ModuleOutput",
    # Registry
    "ModuleRegistry",
    "register_module",
    "get_module",
    # Device management
    "DeviceManager",
    "get_optimal_device",
    # Configuration
    "PipelineConfig",
    "SuperResConfig",
    "EdgeConfig",
    "ShadeConfig",
    "OutputConfig",
]
