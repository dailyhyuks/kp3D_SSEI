"""Super-resolution module for upscaling Korean paintings.

Provides high-quality upscaling using Real-ESRGAN and other models,
optimized for traditional painting characteristics.
"""

from kp3d.modules.superres.base import (
    BaseSuperResolution,
    ScaleFactor,
    SuperResConfig,
)
from kp3d.modules.superres.real_esrgan import RealESRGANModule
from kp3d.core.registry import register_module

# Register as "superres" for pipeline compatibility
@register_module("superres")
class SuperResModule(RealESRGANModule):
    """Pipeline-compatible SuperResolution module.

    Wrapper around RealESRGANModule registered with the standard
    name 'superres' for pipeline configuration compatibility.
    """
    pass

__all__ = [
    "BaseSuperResolution",
    "ScaleFactor",
    "SuperResConfig",
    "RealESRGANModule",
    "SuperResModule",
]
