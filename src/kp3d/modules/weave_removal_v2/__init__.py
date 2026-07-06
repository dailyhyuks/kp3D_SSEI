"""Stage 1 v2: 색 레이어 직물 제거 + 자가 경쟁 게이트 (스펙 §1.4, §2)."""
from .coherence import phase_coherence
from .lattice import LatticeResult, estimate_lattice, predict_peak_freqs

__all__ = [
    "LatticeResult",
    "estimate_lattice",
    "predict_peak_freqs",
    "phase_coherence",
]
