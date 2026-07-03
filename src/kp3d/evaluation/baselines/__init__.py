"""Baseline methods for comparative evaluation.

Registry pattern allows easy addition of new baselines.
"""

from typing import Dict, Type

from kp3d.evaluation.baselines.base import (
    BaseEnhancementBaseline,
    BaseInpaintingBaseline,
)

# Registries
_ENHANCEMENT_BASELINES: Dict[str, Type[BaseEnhancementBaseline]] = {}
_INPAINTING_BASELINES: Dict[str, Type[BaseInpaintingBaseline]] = {}


def register_enhancement_baseline(cls: Type[BaseEnhancementBaseline]):
    """Register an enhancement baseline class."""
    _ENHANCEMENT_BASELINES[cls.name] = cls
    return cls


def register_inpainting_baseline(cls: Type[BaseInpaintingBaseline]):
    """Register an inpainting baseline class."""
    _INPAINTING_BASELINES[cls.name] = cls
    return cls


def get_enhancement_baseline(name: str) -> BaseEnhancementBaseline:
    """Get an enhancement baseline instance by name."""
    if name not in _ENHANCEMENT_BASELINES:
        available = list(_ENHANCEMENT_BASELINES.keys())
        raise KeyError(
            f"Enhancement baseline '{name}' not found. Available: {available}"
        )
    return _ENHANCEMENT_BASELINES[name]()


def get_inpainting_baseline(name: str) -> BaseInpaintingBaseline:
    """Get an inpainting baseline instance by name."""
    if name not in _INPAINTING_BASELINES:
        available = list(_INPAINTING_BASELINES.keys())
        raise KeyError(
            f"Inpainting baseline '{name}' not found. Available: {available}"
        )
    return _INPAINTING_BASELINES[name]()


def list_enhancement_baselines() -> list:
    """List all registered enhancement baselines."""
    return list(_ENHANCEMENT_BASELINES.keys())


def list_inpainting_baselines() -> list:
    """List all registered inpainting baselines."""
    return list(_INPAINTING_BASELINES.keys())


# Import baseline modules to trigger registration
from kp3d.evaluation.baselines import opencv_inpaint  # noqa: E402, F401
from kp3d.evaluation.baselines import opencv_restoration  # noqa: E402, F401
from kp3d.evaluation.baselines import lama_inpaint  # noqa: E402, F401
from kp3d.evaluation.baselines import mat_inpaint  # noqa: E402, F401
from kp3d.evaluation.baselines import sd_inpaint  # noqa: E402, F401
from kp3d.evaluation.baselines import brushnet_inpaint  # noqa: E402, F401
from kp3d.evaluation.baselines import powerpaint_inpaint  # noqa: E402, F401

__all__ = [
    "BaseEnhancementBaseline",
    "BaseInpaintingBaseline",
    "register_enhancement_baseline",
    "register_inpainting_baseline",
    "get_enhancement_baseline",
    "get_inpainting_baseline",
    "list_enhancement_baselines",
    "list_inpainting_baselines",
]
