"""Hybrid grid pattern removal (v8).

Extends EnhancedGridPatternRestorer with:
- Wavelet-based grid decomposition using Stationary Wavelet Transform
- Morphological grid detection using directional top-hat transforms
- Advanced edge preservation with DoG/LoG/Structure Tensor + anisotropic diffusion
- Confidence-weighted blending for optimal combination of methods

The hybrid approach intelligently combines multiple grid removal techniques,
weighting each method based on spatial confidence maps to achieve superior
results while preserving painting details and brushstrokes.

All v8 features are disabled by default (backward compatible with v7).
"""

import cv2
import numpy as np
import time
from torch import Tensor
from typing import Any, Dict, Optional, Tuple

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import RestorationConfig
from kp3d.modules.restoration.enhanced_grid_pattern import EnhancedGridPatternRestorer
from kp3d.modules.restoration.wavelet_grid import WaveletGridDecomposer
from kp3d.modules.restoration.morphological_grid import MorphologicalGridDetector
from kp3d.modules.restoration.edge_preserving import EdgePreservingProcessor
from kp3d.modules.restoration.stft_adaptive_grid import STFTAdaptiveGridRemover


# STFT adaptive grid removal presets
_STFT_PRESETS: Dict[str, Dict[str, Any]] = {
    "stft_adaptive": {
        "use_stft": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
    },
    "stft_aggressive": {
        "use_stft": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.2,
        "base_attenuation": 0.08,
    },
    "stft_conservative": {
        "use_stft": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.7,
        "base_attenuation": 0.20,
    },
    "stft_global": {
        "use_stft": False,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
    },
    "stft_no_edge": {
        "use_stft": True,
        "channel_adaptive": True,
        "edge_protection": False,
        "edge_preservation": 0.0,
        "base_attenuation": 0.15,
    },
    # V2 presets: point notch + 2-pass + adaptive edge
    "stft_v2": {
        "use_v2": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
    },
    "stft_v2_aggressive": {
        "use_v2": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.2,
        "base_attenuation": 0.08,
    },
    "stft_v2_quality": {
        "use_v2": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.7,
        "base_attenuation": 0.20,
    },
    # V3 presets: grid removal + edge restoration (dual-domain)
    "stft_v3": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.7,
        "aggressive_attenuation": 0.05,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
    },
    "stft_v3_aggressive": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.2,
        "base_attenuation": 0.08,
        "edge_strength": 0.4,
        "aggressive_attenuation": 0.03,
        "highpass_sigma": 1.0,
        "channel_weights": (0.15, 0.4, 0.45),
        "apply_diffusion": False,
    },
    "stft_v3_quality": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.7,
        "base_attenuation": 0.20,
        "edge_strength": 1.0,
        "aggressive_attenuation": 0.08,
        "highpass_sigma": 2.0,
        "channel_weights": (0.25, 0.4, 0.35),
        "apply_diffusion": True,
    },
    # V3.1 presets: multi-scale edge + adaptive weights + energy normalization
    "stft_v3.1": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.7,
        "aggressive_attenuation": 0.05,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
    },
    "stft_v3.1_aggressive": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.2,
        "base_attenuation": 0.08,
        "edge_strength": 0.4,
        "aggressive_attenuation": 0.03,
        "highpass_sigma": 1.0,
        "channel_weights": (0.15, 0.4, 0.45),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
    },
    "stft_v3.1_quality": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.7,
        "base_attenuation": 0.20,
        "edge_strength": 1.0,
        "aggressive_attenuation": 0.08,
        "highpass_sigma": 2.0,
        "channel_weights": (0.25, 0.4, 0.35),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
    },
    # V3.2 presets: tuned notch width + subpixel (point_sigma = notch_sigma * 1.5)
    # V3.1 baseline: notch_sigma=1.5 → point_sigma=2.25, GR=86%, PSNR=30dB
    # V3.2 target: slightly wider notch to catch spectral leakage, 1-pass only
    "stft_v3.2": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 1.8,       # point_sigma=2.7 (was 2.5→3.75)
        "n_harmonics_override": 6,          # was 7
        "n_passes": 1,                      # was 2
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012, # was 0.005
    },
    "stft_v3.2_aggressive": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.3,
        "base_attenuation": 0.10,
        "edge_strength": 0.4,
        "aggressive_attenuation": 0.01,
        "highpass_sigma": 1.5,
        "channel_weights": (0.15, 0.4, 0.45),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 2.2,        # point_sigma=3.3 (was 3.0→4.5)
        "n_harmonics_override": 7,
        "n_passes": 1,                       # was 2
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.008,  # was 0.003
    },
    "stft_v3.2_quality": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.7,
        "aggressive_attenuation": 0.03,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 1.7,        # point_sigma=2.55 (was 2.5→3.75)
        "n_harmonics_override": 5,           # was 7
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.015,  # was 0.005
    },
    # V3.3: Intensity-based ink protection presets
    "stft_v3.3": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012,
        "intensity_protection": True,
        "intensity_threshold": 60.0,
        "intensity_steepness": 0.08,
        "intensity_blur_sigma": 3.0,
        "intensity_protection_strength": 0.8,
    },
    "stft_v3.3_aggressive": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.3,
        "base_attenuation": 0.10,
        "edge_strength": 0.4,
        "aggressive_attenuation": 0.01,
        "highpass_sigma": 1.5,
        "channel_weights": (0.15, 0.4, 0.45),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 2.2,
        "n_harmonics_override": 7,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.008,
        "intensity_protection": True,
        "intensity_threshold": 50.0,
        "intensity_steepness": 0.10,
        "intensity_blur_sigma": 3.0,
        "intensity_protection_strength": 0.6,
    },
    "stft_v3.3_quality": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.7,
        "aggressive_attenuation": 0.03,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 1.7,
        "n_harmonics_override": 5,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.015,
        "intensity_protection": True,
        "intensity_threshold": 70.0,
        "intensity_steepness": 0.06,
        "intensity_blur_sigma": 3.0,
        "intensity_protection_strength": 0.9,
    },
    # V3.4: Energy-based grid protection (uses pre-computed energy_map directly)
    "stft_v3.4": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012,
        # V3.3 intensity OFF (replaced by energy)
        "intensity_protection": False,
        # V3.4 energy ON
        "energy_protection": True,
        "energy_protection_strength": 0.70,
        "energy_protection_blur": 1.5,
        "energy_protection_threshold": 0.3,
    },
    "stft_v3.4_aggressive": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.3,
        "base_attenuation": 0.10,
        "edge_strength": 0.4,
        "aggressive_attenuation": 0.01,
        "highpass_sigma": 1.5,
        "channel_weights": (0.15, 0.4, 0.45),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 2.2,
        "n_harmonics_override": 7,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.008,
        "intensity_protection": False,
        "energy_protection": True,
        "energy_protection_strength": 0.50,
        "energy_protection_blur": 1.0,
        "energy_protection_threshold": 0.2,
    },
    "stft_v3.4_quality": {
        "use_v3": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "edge_preservation": 0.5,
        "base_attenuation": 0.15,
        "edge_strength": 0.7,
        "aggressive_attenuation": 0.03,
        "highpass_sigma": 1.5,
        "channel_weights": (0.2, 0.4, 0.4),
        "apply_diffusion": False,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "notch_sigma_override": 1.7,
        "n_harmonics_override": 5,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.015,
        "intensity_protection": False,
        "energy_protection": True,
        "energy_protection_strength": 0.85,
        "energy_protection_blur": 2.0,
        "energy_protection_threshold": 0.4,
    },
    # V3.5: Multiplicative Grid Model - Log-Domain Notch Filtering
    "stft_v3.5_log": {
        "use_v3_log": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "epsilon": 3.0,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "content_gradient_mask": False,
        "channel_weights": (0.2, 0.4, 0.4),
    },
    "stft_v3.5_log_e1": {
        "use_v3_log": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "epsilon": 1.0,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "content_gradient_mask": False,
        "channel_weights": (0.2, 0.4, 0.4),
    },
    "stft_v3.5_log_e5": {
        "use_v3_log": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "epsilon": 5.0,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "content_gradient_mask": False,
        "channel_weights": (0.2, 0.4, 0.4),
    },
    # V3.5: Log-Domain + Content Gradient Mask combination
    "stft_v3.5_log_cg": {
        "use_v3_log": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "epsilon": 3.0,
        "edge_strength": 0.6,
        "aggressive_attenuation": 0.02,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "n_passes": 1,
        "subpixel_period": True,
        "edge_hf_notch_attenuation": 0.012,
        "multiscale_edge": True,
        "adaptive_weights": True,
        "energy_normalize": True,
        "content_gradient_mask": True,
        "channel_weights": (0.2, 0.4, 0.4),
    },
    # V3.5: Multiplicative Grid Model - Period-Folded Template Division
    "stft_v3.5_mult": {
        "use_mult": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "flat_threshold": 10.0,
        "residual_notch": True,
        "residual_attenuation": 0.10,
        "residual_sigma": 1.5,
        "edge_strength": 0.5,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "subpixel_period": False,
        "content_gradient_mask": False,
    },
    "stft_v3.5_mult_f8": {
        "use_mult": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "flat_threshold": 8.0,
        "residual_notch": True,
        "residual_attenuation": 0.10,
        "residual_sigma": 1.5,
        "edge_strength": 0.5,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "subpixel_period": False,
        "content_gradient_mask": False,
    },
    "stft_v3.5_mult_f15": {
        "use_mult": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "flat_threshold": 15.0,
        "residual_notch": True,
        "residual_attenuation": 0.10,
        "residual_sigma": 1.5,
        "edge_strength": 0.5,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "subpixel_period": False,
        "content_gradient_mask": False,
    },
    # V3.5: Multiplicative Division + Content Gradient Mask
    "stft_v3.5_mult_cg": {
        "use_mult": True,
        "channel_adaptive": True,
        "edge_protection": True,
        "base_attenuation": 0.15,
        "flat_threshold": 10.0,
        "residual_notch": True,
        "residual_attenuation": 0.10,
        "residual_sigma": 1.5,
        "edge_strength": 0.5,
        "notch_sigma_override": 1.8,
        "n_harmonics_override": 6,
        "subpixel_period": False,
        "content_gradient_mask": True,
    },
}


# Default weights for hybrid presets
_HYBRID_PRESETS: Dict[str, Dict[str, float]] = {
    "hybrid_balanced": {
        "wavelet": 0.4,
        "fft": 0.3,
        "morph": 0.3,
        "edge_preservation": 0.5,
    },
    "hybrid_aggressive": {
        "wavelet": 0.5,
        "fft": 0.3,
        "morph": 0.2,
        "edge_preservation": 0.2,
    },
    "hybrid_edge_safe": {
        "wavelet": 0.3,
        "fft": 0.2,
        "morph": 0.2,
        "edge_preservation": 0.8,
    },
}


class HybridGridPatternRestorer(EnhancedGridPatternRestorer):
    """v8 Hybrid grid pattern removal restorer.

    Combines multiple grid removal techniques for optimal results:
    - Wavelet: Translation-invariant grid pattern detection/suppression
    - FFT: Frequency-domain directional filtering (inherited from v7)
    - Morphological: Multi-scale directional top-hat for grid line detection
    - Edge Preservation: DoG/LoG/Structure Tensor with anisotropic diffusion

    When no v8 features are enabled (default), behavior is identical to
    EnhancedGridPatternRestorer (v7).

    Attributes:
        config: RestorationConfig with v8 parameters
        _wavelet_decomposer: Lazy-initialized WaveletGridDecomposer
        _morph_detector: Lazy-initialized MorphologicalGridDetector
        _edge_processor: Lazy-initialized EdgePreservingProcessor
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the hybrid grid pattern restorer.

        Args:
            config: RestorationConfig with optional v8 parameters.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(config=config, **kwargs)

        # Lazy initialization placeholders
        self._wavelet_decomposer: Optional[WaveletGridDecomposer] = None
        self._morph_detector: Optional[MorphologicalGridDetector] = None
        self._edge_processor: Optional[EdgePreservingProcessor] = None
        self._stft_remover: Optional[STFTAdaptiveGridRemover] = None
        self._components_initialized: bool = False

    @property
    def name(self) -> str:
        """Return module name."""
        return "hybrid_grid_pattern"

    def _ensure_components(self) -> None:
        """Lazily initialize v8 utility components from config.

        Components are only created on first use to avoid overhead
        when v8 features are disabled.
        """
        if self._components_initialized:
            return

        config = self.config

        # Initialize WaveletGridDecomposer if wavelet is enabled
        if self._is_wavelet_enabled():
            wavelet_type = getattr(config, 'grid_wavelet_type', 'db4')
            wavelet_levels = getattr(config, 'grid_wavelet_levels', 3)
            suppression = getattr(config, 'grid_wavelet_suppression', 0.3)
            detail_preservation = getattr(config, 'grid_wavelet_detail_preservation', 0.7)

            self._wavelet_decomposer = WaveletGridDecomposer(
                wavelet_type=wavelet_type,
                levels=wavelet_levels,
                suppression_strength=suppression,
                detail_preservation=detail_preservation,
            )

        # Initialize MorphologicalGridDetector if morphological is enabled
        if self._is_morphological_enabled():
            line_lengths = getattr(config, 'grid_morph_line_lengths', (5, 9, 15))
            line_width = getattr(config, 'grid_morph_line_width', 1)
            angles = getattr(config, 'grid_morph_angles', (0, 45, 90, 135))
            threshold = getattr(config, 'grid_morph_threshold', 0.3)

            self._morph_detector = MorphologicalGridDetector(
                line_lengths=line_lengths,
                line_width=line_width,
                angles=angles,
                threshold=threshold,
            )

        # Initialize EdgePreservingProcessor if advanced edge is enabled
        if self._is_advanced_edge_enabled():
            dog_sigma1 = getattr(config, 'grid_dog_sigma1', 1.0)
            dog_sigma2 = getattr(config, 'grid_dog_sigma2', 2.0)
            diffusion_iterations = getattr(config, 'grid_diffusion_iterations', 10)
            diffusion_kappa = getattr(config, 'grid_diffusion_kappa', 30.0)
            diffusion_gamma = getattr(config, 'grid_diffusion_gamma', 0.1)

            self._edge_processor = EdgePreservingProcessor(
                dog_sigma1=dog_sigma1,
                dog_sigma2=dog_sigma2,
                diffusion_iterations=diffusion_iterations,
                diffusion_kappa=diffusion_kappa,
                diffusion_gamma=diffusion_gamma,
            )

        self._components_initialized = True

    def _is_wavelet_enabled(self) -> bool:
        """Check if wavelet grid decomposition is enabled."""
        return getattr(self.config, 'grid_use_wavelet', False)

    def _is_morphological_enabled(self) -> bool:
        """Check if morphological grid detection is enabled."""
        return getattr(self.config, 'grid_use_morphological', False)

    def _is_advanced_edge_enabled(self) -> bool:
        """Check if advanced edge preservation is enabled."""
        return getattr(self.config, 'grid_use_advanced_edge', False)

    def _is_hybrid_enabled(self) -> bool:
        """Check if ANY v8 hybrid feature is enabled.

        Returns:
            True if wavelet, morphological, or advanced edge is enabled.
        """
        return (
            self._is_wavelet_enabled()
            or self._is_morphological_enabled()
            or self._is_advanced_edge_enabled()
        )

    def _get_blend_mode(self) -> str:
        """Get the hybrid blending mode from config."""
        return getattr(self.config, 'grid_hybrid_blend_mode', 'confidence')

    def _get_preset_weights(self, method: str) -> Dict[str, float]:
        """Get preset weights for a hybrid method.

        Args:
            method: Method name (e.g., "hybrid_balanced")

        Returns:
            Dictionary with wavelet, fft, morph, and edge_preservation weights.
        """
        if method in _HYBRID_PRESETS:
            return _HYBRID_PRESETS[method].copy()

        # Build custom weights from config
        return {
            "wavelet": getattr(self.config, 'grid_hybrid_wavelet_weight', 0.4),
            "fft": getattr(self.config, 'grid_hybrid_fft_weight', 0.3),
            "morph": getattr(self.config, 'grid_hybrid_morph_weight', 0.3),
            "edge_preservation": 0.5,
        }

    def restore_grid_pattern(
        self,
        image_bgr: np.ndarray,
        method: str = "guided_only",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Main grid pattern removal pipeline with hybrid support.

        If method starts with "hybrid_" or any v8 feature is enabled,
        runs the hybrid pipeline. Otherwise, delegates to parent class.

        Args:
            image_bgr: Input BGR image.
            method: Restoration method. Options:
                - Standard v6/v7 methods: "guided_only", "triple_medium",
                  "triple_strong", "aggressive", "ultra", "extreme"
                - v8 hybrid methods: "hybrid_balanced", "hybrid_aggressive",
                  "hybrid_edge_safe"

        Returns:
            Tuple of (restored_image, intermediates_dict)
        """
        # Check if we should use STFT pipeline
        if method.startswith("stft_"):
            return self._run_stft_pipeline(image_bgr, method)

        # Check if we should use hybrid pipeline
        use_hybrid = method.startswith("hybrid_") or self._is_hybrid_enabled()

        if use_hybrid:
            # Get preset weights if using a hybrid preset method
            preset_weights = self._get_preset_weights(method)

            # For hybrid presets, enable relevant features temporarily
            if method.startswith("hybrid_"):
                return self._run_hybrid_pipeline(
                    image_bgr,
                    method,
                    preset_weights=preset_weights,
                    enable_all=True,  # Enable all methods for presets
                )
            else:
                # Use current config settings
                return self._run_hybrid_pipeline(
                    image_bgr,
                    method,
                    preset_weights=preset_weights,
                    enable_all=False,
                )

        # Delegate to parent (v7) behavior
        return super().restore_grid_pattern(image_bgr, method)

    def _run_hybrid_pipeline(
        self,
        image_bgr: np.ndarray,
        method: str,
        preset_weights: Optional[Dict[str, float]] = None,
        enable_all: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run the v8 hybrid grid removal pipeline.

        Pipeline steps:
        1. Ensure v8 components are initialized
        2. Run wavelet decomposition (if enabled)
        3. Run FFT filtering (always - inherited from v7)
        4. Run morphological detection (if enabled)
        5. Compute edge protection mask (if enabled)
        6. Confidence-weighted blending of all results
        7. Apply anisotropic diffusion (if enabled)
        8. Apply edge-aware grid removal (if enabled)

        Args:
            image_bgr: Input BGR image.
            method: Method name for base FFT method selection.
            preset_weights: Optional dictionary with wavelet/fft/morph/edge weights.
            enable_all: If True, enable all hybrid methods for preset mode.

        Returns:
            Tuple of (restored_bgr_uint8, intermediates_dict)
        """
        intermediates: Dict[str, np.ndarray] = {}

        if preset_weights is None:
            preset_weights = self._get_preset_weights(method)

        # Step 1: Initialize components (lazy)
        self._ensure_components()

        # For presets, create temporary components if not already initialized
        if enable_all:
            self._ensure_all_components_for_preset()

        # Step 2: Run wavelet decomposition (if enabled)
        wavelet_result: Optional[np.ndarray] = None
        wavelet_conf: Optional[np.ndarray] = None

        if self._wavelet_decomposer is not None:
            wavelet_result, wavelet_conf = self._wavelet_decomposer.process(image_bgr)
            if self.config.store_intermediates:
                intermediates['wavelet_result'] = wavelet_result
                intermediates['wavelet_confidence'] = (
                    (wavelet_conf * 255).astype(np.uint8)
                    if wavelet_conf is not None
                    else np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                )

        # Step 3: Run FFT filtering (always - from parent)
        base_method = self._get_base_fft_method(method)
        fft_result, fft_intermediates = super().restore_grid_pattern(
            image_bgr, method=base_method
        )

        if self.config.store_intermediates:
            intermediates['fft_result'] = fft_result
            intermediates.update({
                f'fft_{k}': v
                for k, v in fft_intermediates.items()
            })

        # Step 4: Run morphological detection (if enabled)
        morph_conf: Optional[np.ndarray] = None
        morph_mask: Optional[np.ndarray] = None

        if self._morph_detector is not None:
            morph_conf, morph_mask = self._morph_detector.compute_grid_confidence_map(
                image_bgr
            )
            if self.config.store_intermediates:
                intermediates['morph_confidence'] = (morph_conf * 255).astype(np.uint8)
                intermediates['morph_mask'] = morph_mask

        # Step 5: Compute edge protection mask (if enabled)
        edge_mask: Optional[np.ndarray] = None

        if self._edge_processor is not None:
            edge_mask = self._edge_processor.compute_edge_protection_mask(image_bgr)
            if self.config.store_intermediates:
                intermediates['edge_mask'] = (edge_mask * 255).astype(np.uint8)

        # Step 6: Confidence-weighted blending
        result = self._confidence_weighted_blend(
            wavelet_result=wavelet_result,
            fft_result=fft_result,
            morph_conf=morph_conf,
            image_bgr=image_bgr,
            edge_mask=edge_mask,
            preset_weights=preset_weights,
        )

        if self.config.store_intermediates:
            intermediates['blended_result'] = result.copy()

        # Step 7: Apply anisotropic diffusion (if advanced edge enabled)
        if self._edge_processor is not None and self._is_advanced_edge_enabled():
            result = self._edge_processor.anisotropic_diffusion(result)
            if self.config.store_intermediates:
                intermediates['diffused_result'] = result.copy()

        # Step 8: Apply edge-aware grid removal (if edge mask available)
        if edge_mask is not None and self._edge_processor is not None:
            edge_preservation = preset_weights.get('edge_preservation', 0.5)
            # Scale edge mask by preservation factor
            scaled_edge_mask = edge_mask * edge_preservation
            result = self._edge_processor.edge_aware_grid_removal(
                image_bgr, result, scaled_edge_mask
            )

        # Ensure uint8 output
        result = np.clip(result, 0, 255).astype(np.uint8)

        if self.config.store_intermediates:
            intermediates['result'] = result

        return result, intermediates

    def _ensure_all_components_for_preset(self) -> None:
        """Ensure all components are initialized for preset mode.

        When using hybrid presets like "hybrid_balanced", we need all
        components even if they are not enabled in config.
        """
        config = self.config

        if self._wavelet_decomposer is None:
            wavelet_type = getattr(config, 'grid_wavelet_type', 'db4')
            wavelet_levels = getattr(config, 'grid_wavelet_levels', 3)
            suppression = getattr(config, 'grid_wavelet_suppression', 0.3)
            detail_preservation = getattr(config, 'grid_wavelet_detail_preservation', 0.7)

            self._wavelet_decomposer = WaveletGridDecomposer(
                wavelet_type=wavelet_type,
                levels=wavelet_levels,
                suppression_strength=suppression,
                detail_preservation=detail_preservation,
            )

        if self._morph_detector is None:
            line_lengths = getattr(config, 'grid_morph_line_lengths', (5, 9, 15))
            line_width = getattr(config, 'grid_morph_line_width', 1)
            angles = getattr(config, 'grid_morph_angles', (0, 45, 90, 135))
            threshold = getattr(config, 'grid_morph_threshold', 0.3)

            self._morph_detector = MorphologicalGridDetector(
                line_lengths=line_lengths,
                line_width=line_width,
                angles=angles,
                threshold=threshold,
            )

        if self._edge_processor is None:
            dog_sigma1 = getattr(config, 'grid_dog_sigma1', 1.0)
            dog_sigma2 = getattr(config, 'grid_dog_sigma2', 2.0)
            diffusion_iterations = getattr(config, 'grid_diffusion_iterations', 10)
            diffusion_kappa = getattr(config, 'grid_diffusion_kappa', 30.0)
            diffusion_gamma = getattr(config, 'grid_diffusion_gamma', 0.1)

            self._edge_processor = EdgePreservingProcessor(
                dog_sigma1=dog_sigma1,
                dog_sigma2=dog_sigma2,
                diffusion_iterations=diffusion_iterations,
                diffusion_kappa=diffusion_kappa,
                diffusion_gamma=diffusion_gamma,
            )

    def _run_stft_pipeline(
        self,
        image_bgr: np.ndarray,
        method: str,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run the v9 STFT adaptive grid removal pipeline.

        Pipeline:
        1. Initialize STFT remover with preset params
        2. Optionally compute edge protection mask
        3. Run STFT + channel-adaptive + edge protection
        4. Return result with intermediates

        Args:
            image_bgr: Input BGR image.
            method: STFT method name (stft_adaptive, stft_aggressive, etc.)

        Returns:
            Tuple of (restored_bgr_uint8, intermediates_dict)
        """
        # Get preset parameters
        preset = _STFT_PRESETS.get(method, _STFT_PRESETS["stft_adaptive"])

        # Initialize STFT remover
        config = self.config
        stft_remover = STFTAdaptiveGridRemover(
            period_x=getattr(config, 'grid_stft_period_x', 0),
            period_y=getattr(config, 'grid_stft_period_y', 0),
            window_size=getattr(config, 'grid_stft_window_size', 63),
            hop_size=getattr(config, 'grid_stft_hop_size', 16),
            notch_sigma=getattr(config, 'grid_stft_notch_sigma', 1.5),
            base_attenuation=preset['base_attenuation'],
            channel_adaptive=preset['channel_adaptive'],
        )

        # Compute edge protection mask if needed
        edge_mask = None
        if preset['edge_protection']:
            if self._edge_processor is None:
                self._edge_processor = EdgePreservingProcessor(
                    dog_sigma1=getattr(config, 'grid_dog_sigma1', 1.0),
                    dog_sigma2=getattr(config, 'grid_dog_sigma2', 2.0),
                    diffusion_iterations=getattr(
                        config, 'grid_diffusion_iterations', 10
                    ),
                    diffusion_kappa=getattr(config, 'grid_diffusion_kappa', 30.0),
                    diffusion_gamma=getattr(config, 'grid_diffusion_gamma', 0.1),
                )
            edge_mask = self._edge_processor.compute_edge_protection_mask(
                image_bgr
            )

        # Run the STFT pipeline (v3.5_log, v3.5_mult, v3, v2, or v1)
        if preset.get('use_v3_log', False):
            # V3.5: Log-domain notch filtering
            edge_mask_for_log = edge_mask
            result, intermediates = stft_remover.process_v3_log(
                image_bgr,
                edge_mask=edge_mask_for_log,
                epsilon=preset.get('epsilon', 3.0),
                edge_strength=preset.get('edge_strength', 0.6),
                aggressive_attenuation=preset.get('aggressive_attenuation', 0.02),
                notch_sigma_override=preset.get('notch_sigma_override', None),
                n_harmonics_override=preset.get('n_harmonics_override', None),
                n_passes=preset.get('n_passes', 1),
                subpixel_period=preset.get('subpixel_period', True),
                edge_hf_notch_attenuation=preset.get('edge_hf_notch_attenuation', 0.012),
                multiscale_edge=preset.get('multiscale_edge', True),
                adaptive_weights=preset.get('adaptive_weights', True),
                energy_normalize=preset.get('energy_normalize', True),
                content_gradient_mask=preset.get('content_gradient_mask', False),
                channel_weights=preset.get('channel_weights', (0.2, 0.4, 0.4)),
            )
        elif preset.get('use_mult', False):
            # V3.5: Period-folded template division
            result, intermediates = stft_remover.process_multiplicative(
                image_bgr,
                edge_mask=edge_mask,
                flat_threshold=preset.get('flat_threshold', 10.0),
                residual_notch=preset.get('residual_notch', True),
                residual_attenuation=preset.get('residual_attenuation', 0.10),
                residual_sigma=preset.get('residual_sigma', 1.5),
                edge_strength=preset.get('edge_strength', 0.5),
                notch_sigma_override=preset.get('notch_sigma_override', None),
                n_harmonics_override=preset.get('n_harmonics_override', None),
                subpixel_period=preset.get('subpixel_period', False),
                content_gradient_mask=preset.get('content_gradient_mask', False),
            )
        elif preset.get('use_v3', False):
            result, intermediates = stft_remover.process_v3(
                image_bgr,
                edge_mask=edge_mask,
                edge_strength=preset.get('edge_strength', 0.7),
                aggressive_attenuation=preset.get('aggressive_attenuation', 0.05),
                highpass_sigma=preset.get('highpass_sigma', 1.5),
                channel_weights=preset.get('channel_weights', (0.2, 0.4, 0.4)),
                apply_diffusion=preset.get('apply_diffusion', False),
                multiscale_edge=preset.get('multiscale_edge', False),
                adaptive_weights=preset.get('adaptive_weights', False),
                energy_normalize=preset.get('energy_normalize', False),
                notch_sigma_override=preset.get('notch_sigma_override', None),
                n_harmonics_override=preset.get('n_harmonics_override', None),
                n_passes=preset.get('n_passes', 1),
                subpixel_period=preset.get('subpixel_period', False),
                edge_hf_notch_attenuation=preset.get('edge_hf_notch_attenuation', 0.02),
                intensity_protection=preset.get('intensity_protection', False),
                intensity_threshold=preset.get('intensity_threshold', 60.0),
                intensity_steepness=preset.get('intensity_steepness', 0.08),
                intensity_blur_sigma=preset.get('intensity_blur_sigma', 3.0),
                intensity_protection_strength=preset.get('intensity_protection_strength', 0.8),
                energy_protection=preset.get('energy_protection', False),
                energy_protection_strength=preset.get('energy_protection_strength', 0.85),
                energy_protection_blur=preset.get('energy_protection_blur', 2.0),
                energy_protection_threshold=preset.get('energy_protection_threshold', 0.0),
            )
        elif preset.get('use_v2', False):
            result, intermediates = stft_remover.process_v2(
                image_bgr,
                edge_mask=edge_mask,
                edge_preservation=preset['edge_preservation'],
            )
        else:
            result, intermediates = stft_remover.process(
                image_bgr,
                edge_mask=edge_mask,
                edge_preservation=preset['edge_preservation'],
                use_stft=preset.get('use_stft', True),
            )

        # Store edge mask in intermediates
        if edge_mask is not None and self.config.store_intermediates:
            intermediates['edge_mask'] = (edge_mask * 255).astype(np.uint8)

        intermediates['method'] = np.array(
            [ord(c) for c in method], dtype=np.uint8
        )

        return result, intermediates

    def _get_base_fft_method(self, method: str) -> str:
        """Get the base FFT method for hybrid pipeline.

        Args:
            method: Original method name.

        Returns:
            Base method to use for FFT filtering.
        """
        if method.startswith("hybrid_"):
            # For hybrid methods, use guided_only as the base FFT method
            return "guided_only"

        # For config-based hybrid, use the specified method
        return method if not method.startswith("hybrid_") else "guided_only"

    def _compute_method_confidences(
        self,
        wavelet_conf: Optional[np.ndarray],
        morph_conf: Optional[np.ndarray],
        edge_mask: Optional[np.ndarray],
        preset_weights: Dict[str, float],
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spatial confidence maps for each method.

        The confidence maps weight each method spatially based on:
        - Wavelet confidence (grid detection strength)
        - Morphological confidence (grid line confidence)
        - Edge mask (edge protection - reduce grid removal on edges)

        Args:
            wavelet_conf: Wavelet grid confidence map or None.
            morph_conf: Morphological grid confidence map or None.
            edge_mask: Edge protection mask or None.
            preset_weights: Weight dictionary with wavelet/fft/morph keys.
            shape: Output shape (H, W).

        Returns:
            Tuple of (w_wavelet, w_fft, w_morph) normalized confidence maps.
        """
        h, w = shape

        # Initialize uniform confidence if not provided
        if wavelet_conf is None:
            wavelet_conf = np.ones((h, w), dtype=np.float32)
        if morph_conf is None:
            morph_conf = np.ones((h, w), dtype=np.float32)
        if edge_mask is None:
            edge_mask = np.zeros((h, w), dtype=np.float32)

        # Ensure float32
        wavelet_conf = wavelet_conf.astype(np.float32)
        morph_conf = morph_conf.astype(np.float32)
        edge_mask = edge_mask.astype(np.float32)

        # Compute raw weights with edge-based reduction
        # Less weight on edges to preserve brushstrokes
        w_wavelet = (
            wavelet_conf
            * preset_weights['wavelet']
            * (1.0 - edge_mask)
        )

        w_fft = (
            preset_weights['fft']
            * (1.0 - edge_mask * 0.5)  # Partial reduction on edges
        ) * np.ones((h, w), dtype=np.float32)

        w_morph = (
            morph_conf
            * preset_weights['morph']
            * (1.0 - edge_mask)
        )

        # Normalize so weights sum to 1 at each pixel
        total_weight = w_wavelet + w_fft + w_morph + 1e-8
        w_wavelet = w_wavelet / total_weight
        w_fft = w_fft / total_weight
        w_morph = w_morph / total_weight

        return w_wavelet, w_fft, w_morph

    def _confidence_weighted_blend(
        self,
        wavelet_result: Optional[np.ndarray],
        fft_result: np.ndarray,
        morph_conf: Optional[np.ndarray],
        image_bgr: np.ndarray,
        edge_mask: Optional[np.ndarray],
        preset_weights: Dict[str, float],
    ) -> np.ndarray:
        """Blend multiple grid removal results using confidence weighting.

        The blending formula:
        - Wavelet and FFT provide actual processed images
        - Morphological provides a confidence map for grid regions
        - Edge mask protects high-edge regions

        Morphological doesn't produce a separate result; instead, it weights
        the FFT result more heavily in detected grid regions.

        Args:
            wavelet_result: Wavelet-processed image or None.
            fft_result: FFT-processed image (always available).
            morph_conf: Morphological grid confidence map or None.
            image_bgr: Original input image.
            edge_mask: Edge protection mask or None.
            preset_weights: Weight dictionary.

        Returns:
            Blended result as uint8 BGR image.
        """
        h, w = image_bgr.shape[:2]

        # Compute confidence maps
        wavelet_conf = None
        if wavelet_result is not None and self._wavelet_decomposer is not None:
            # Get wavelet confidence (uniform for now based on SWT)
            wavelet_conf = np.ones((h, w), dtype=np.float32) * 0.5

        w_wavelet, w_fft, w_morph = self._compute_method_confidences(
            wavelet_conf=wavelet_conf,
            morph_conf=morph_conf,
            edge_mask=edge_mask,
            preset_weights=preset_weights,
            shape=(h, w),
        )

        # Expand weights to 3 channels
        w_wavelet_3d = w_wavelet[:, :, np.newaxis]
        w_fft_3d = w_fft[:, :, np.newaxis]
        w_morph_3d = w_morph[:, :, np.newaxis]

        # Convert to float for blending
        fft_float = fft_result.astype(np.float32)

        if wavelet_result is not None:
            wavelet_float = wavelet_result.astype(np.float32)
        else:
            wavelet_float = fft_float  # Fallback

        # Blend formula:
        # Morphological doesn't provide an image, just confidence for grid regions.
        # Use morph weight to boost FFT result in grid regions.
        # result = w_wavelet * wavelet + (w_fft + w_morph) * fft
        result = (
            w_wavelet_3d * wavelet_float
            + (w_fft_3d + w_morph_3d) * fft_float
        )

        # In edge regions: blend back toward original
        if edge_mask is not None:
            edge_preservation = preset_weights.get('edge_preservation', 0.5)
            # Scale edge mask by preservation factor for blending strength
            blend_mask = (edge_mask * edge_preservation)[:, :, np.newaxis]
            original_float = image_bgr.astype(np.float32)
            result = blend_mask * original_float + (1.0 - blend_mask) * result

        # Clip and convert to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """v8 hybrid grid pattern removal forward pass.

        Extends parent's forward with v8 metadata tracking.

        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W)
            **kwargs: Additional arguments
                - method: Grid removal method

        Returns:
            ModuleOutput with restored image and v8 metadata
        """
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy BGR
        img_np = self._tensor_to_numpy_rgb(image[0])
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Get method from kwargs or config
        method = kwargs.get('method', self.config.grid_method)

        # Run restoration (handles hybrid routing internally)
        result_bgr, intermediates = self.restore_grid_pattern(img_bgr, method=method)

        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        elapsed = time.time() - start

        # Compute texture reduction metric
        original_std = np.std(
            cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        )
        restored_std = np.std(
            cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        )
        texture_reduction = (1 - restored_std / (original_std + 1e-8)) * 100

        # Build intermediate tensors
        intermediate_tensors = {}
        if self.config.store_intermediates:
            for key, arr in intermediates.items():
                if arr is None or not isinstance(arr, np.ndarray):
                    continue
                if arr.ndim < 2:
                    continue  # Skip 1D metadata arrays
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                intermediate_tensors[key] = self._numpy_to_tensor(arr)

        result_tensor = self._numpy_to_tensor(result_rgb).unsqueeze(0)

        # Build metadata with v8 info
        metadata = {
            'method': f'hybrid_grid_pattern_{method}',
            'processing_time': elapsed,
            'texture_reduction_percent': texture_reduction,
            'grid_bilateral_iterations': self.config.grid_bilateral_iterations,
            'grid_fft_line_width': self.config.grid_fft_line_width,
            'grid_guided_radius': self.config.grid_guided_radius,
            # v7 auto-angle info
            'v7_auto_angle': self.config.fft_auto_angle,
            'v7_angle_tolerance': self.config.fft_angle_tolerance,
            'v7_radial_threshold': self.config.fft_radial_threshold,
            # v8 feature flags
            'v8_wavelet_enabled': self._is_wavelet_enabled(),
            'v8_morphological_enabled': self._is_morphological_enabled(),
            'v8_advanced_edge_enabled': self._is_advanced_edge_enabled(),
            'v8_hybrid_enabled': self._is_hybrid_enabled(),
            'v8_blend_mode': self._get_blend_mode(),
        }

        # Add v8 config parameters if enabled
        if self._is_hybrid_enabled() or method.startswith("hybrid_"):
            metadata.update({
                'v8_wavelet_type': getattr(self.config, 'grid_wavelet_type', 'db4'),
                'v8_wavelet_levels': getattr(self.config, 'grid_wavelet_levels', 3),
                'v8_wavelet_suppression': getattr(
                    self.config, 'grid_wavelet_suppression', 0.3
                ),
                'v8_morph_line_lengths': getattr(
                    self.config, 'grid_morph_line_lengths', (5, 9, 15)
                ),
                'v8_morph_threshold': getattr(
                    self.config, 'grid_morph_threshold', 0.3
                ),
                'v8_dog_sigma1': getattr(self.config, 'grid_dog_sigma1', 1.0),
                'v8_dog_sigma2': getattr(self.config, 'grid_dog_sigma2', 2.0),
                'v8_diffusion_iterations': getattr(
                    self.config, 'grid_diffusion_iterations', 10
                ),
            })

            # Add preset weights if using hybrid method
            if method.startswith("hybrid_"):
                weights = self._get_preset_weights(method)
                metadata['v8_preset_weights'] = weights

        # Add v9 STFT info if using STFT method
        if method.startswith("stft_"):
            preset = _STFT_PRESETS.get(method, _STFT_PRESETS["stft_adaptive"])
            metadata.update({
                'v9_stft_method': method,
                'v9_stft_preset': preset,
                'v9_stft_period_x': getattr(self.config, 'grid_stft_period_x', 0),
                'v9_stft_period_y': getattr(self.config, 'grid_stft_period_y', 0),
                'v9_stft_window_size': getattr(self.config, 'grid_stft_window_size', 63),
            })

        # Add detected angles if auto-angle was used
        if self.config.fft_auto_angle:
            detected = self.detect_grid_angles(
                img_bgr, self.config.fft_angle_tolerance
            )
            metadata['detected_grid_angles'] = detected

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediate_tensors,
            metadata=metadata,
        )


__all__ = ["HybridGridPatternRestorer"]
