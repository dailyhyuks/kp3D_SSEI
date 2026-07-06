"""Stage 1 v2: 색 레이어 직물 제거 + 자가 경쟁 게이트 (스펙 §1.4, §2)."""
from .coherence import phase_coherence
from .lattice import LatticeResult, estimate_lattice, predict_peak_freqs
from .line_layer import normalize_line_contrast
from .notch import fit_peak_gaussian, interpolate_notch
from .removal import (
    WeaveRemovalV2Result,
    derive_patch_size,
    remove_weave,
    weave_band_energy,
)

__all__ = [
    "LatticeResult",
    "estimate_lattice",
    "predict_peak_freqs",
    "phase_coherence",
    "fit_peak_gaussian",
    "interpolate_notch",
    "normalize_line_contrast",
    "WeaveRemovalV2Result",
    "derive_patch_size",
    "weave_band_energy",
    "remove_weave",
]
