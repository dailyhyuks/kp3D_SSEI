"""Enhancement Pipeline module with configurable grid removal strategy.

Provides a complete preprocessing pipeline for digitized Korean paintings
that combines super-resolution with grid artifact removal.
Supports both SpectralGridRemover (default) and legacy MultiplicativeGridRemover.
"""

from kp3d.core.registry import register_module
from kp3d.modules.enhancement.config import EnhancementConfig
from kp3d.modules.enhancement.pipeline import EnhancementPipeline
from kp3d.modules.enhancement.skip_logic import GridPresenceChecker, ResolutionChecker
from kp3d.modules.enhancement.spectral_grid import SpectralGridRemover


@register_module("enhancement")
class EnhancementModule(EnhancementPipeline):
    """Registered enhancement module for the module registry."""

    pass


__all__ = [
    "EnhancementConfig",
    "EnhancementModule",
    "EnhancementPipeline",
    "GridPresenceChecker",
    "ResolutionChecker",
    "SpectralGridRemover",
]
