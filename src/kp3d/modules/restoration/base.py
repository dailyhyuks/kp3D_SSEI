"""Base classes for restoration modules."""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from kp3d.core.base import BasePreprocessModule, ModuleOutput


@dataclass
class RestorationConfig:
    """Configuration for restoration modules.

    Presets:
    - conservative: window=21, threshold=2.0, max_area=100
    - moderate: window=15, threshold=1.5, max_area=150
    - aggressive (default): window=11, threshold=1.2, max_area=200
    """

    # Fading noise detection (aggressive defaults)
    window_size: int = 11
    outlier_threshold: float = 1.2
    min_blob_area: int = 2
    max_blob_area: int = 200

    # Edge exclusion
    edge_threshold: int = 50
    edge_dilation: int = 2

    # Inpainting
    inpaint_radius: int = 3
    inpaint_method: str = "telea"  # "telea" or "ns" (Navier-Stokes)

    # Multi-scale detection (v2)
    multi_scale: bool = True
    window_sizes: tuple = (7, 11, 15, 21)  # Multiple scales for detection
    scale_combination: str = "weighted"  # "or", "and", "weighted"

    # Adaptive thresholding (v2)
    adaptive_threshold: bool = True
    bright_threshold_factor: float = 0.7  # Lower = more sensitive in bright areas
    dark_threshold_factor: float = 1.4  # Higher = less sensitive in dark areas
    saturation_weight: float = 0.3  # Weight for saturation in threshold adjustment

    # Shape filtering (v2)
    circularity_threshold: float = 0.5  # 0-1, higher = more circular required
    density_filter: bool = True
    density_kernel_size: int = 31
    density_threshold: float = 0.3  # Max fraction of neighbors that can also be outliers

    # Color-aware inpainting (v3)
    inpaint_mode: str = "opencv"  # "opencv", "color_aware", "hybrid"
    color_sample_radius: int = 5  # Radius for color sampling around mask
    blend_radius: int = 2  # Boundary blending radius

    # Texture preservation (v3)
    preserve_texture: bool = False  # Whether to restore hanji texture
    texture_blur_sigma: float = 2.0  # Gaussian blur sigma for texture extraction
    texture_strength: float = 0.5  # Texture application strength (0-1)

    # Performance optimization (v4)
    fast_mode: bool = False  # Fast mode (slightly reduced accuracy)
    store_intermediates: bool = True  # Store intermediate results for debugging

    # Frequency-aware restoration (v5)
    freq_base_sigma: float = 2.5  # Base sigma for low-frequency extraction (2-4 recommended)
    freq_edge_sigma_factor: float = 0.4  # Sigma multiplier near edges (lower = sharper)
    freq_saturation_strength: bool = True  # Use saturation to adjust restoration strength
    freq_saturation_weight: float = 0.5  # How much saturation affects strength (0-1)
    freq_edge_proximity_sigma: float = 4.0  # Gaussian sigma for edge proximity map
    freq_texture_noise_reduction: float = 0.2  # Reduce high-freq in non-edge areas (0=none, 1=full)

    # Color-based edge inference (v6)
    use_color_edge_inference: bool = True  # Use color-based edge inference
    color_edge_weight: float = 0.5  # Weight of color-inferred edges vs Canny
    delta_e_threshold: float = 3.0  # Minimum ΔE for edge detection (lower = more sensitive)
    superpixel_segments: int = 300  # Number of superpixel segments
    weak_edge_boost: float = 2.5  # Amplification for weak edges
    edge_boost_strength: float = 0.15  # Final edge boosting strength (optimal: 0.15, range: 0.10-0.20)

    # v7: Adaptive Windows
    use_adaptive_windows: bool = False
    noise_scale_estimate: Optional[float] = None

    # v7: Continuous Sigma
    continuous_sigma: bool = False
    sigma_levels: int = 10
    sigma_min: float = 1.5
    sigma_max: float = 10.0

    # v7: PatchMatch Inpainting
    patchmatch_patch_size: int = 7
    patchmatch_iterations: int = 5
    patchmatch_search_samples: int = 100

    # v7: Noise Classification
    use_noise_classification: bool = False
    noise_class_inpaint_map: dict = field(default_factory=lambda: {
        "dust": "patchmatch",
        "crack": "ns",
        "stain": "color_aware",
        "mold": "patchmatch",
    })

    # v7: Rotated FFT
    fft_auto_angle: bool = False
    fft_angle_tolerance: float = 5.0
    fft_radial_threshold: float = 0.3

    # Grid pattern removal (V13d)
    # Methods: "guided_only" (~22%), "triple_medium" (~20%), "triple_strong" (~18%)
    #          "aggressive" (~33%), "ultra" (~37%) - stronger grid removal
    grid_method: str = "guided_only"
    grid_bilateral_iterations: int = 5  # Iterations for cartoon-texture decomposition
    grid_bilateral_sigma_color: float = 75.0  # Bilateral filter sigma for color
    grid_bilateral_sigma_space: float = 75.0  # Bilateral filter sigma for space
    grid_fft_line_width: int = 7  # FFT directional filter line width
    grid_guided_radius: int = 8  # Guided filter radius
    grid_guided_eps: float = 0.02  # Guided filter regularization
    grid_contour_threshold: float = 0.12  # Contour mask threshold
    grid_structure_threshold: float = 0.3  # Structure tensor threshold for edge enhancement

    # v8: Wavelet Grid Decomposition
    grid_use_wavelet: bool = False
    grid_wavelet_type: str = "db4"
    grid_wavelet_levels: int = 3
    grid_wavelet_suppression: float = 0.3
    grid_wavelet_detail_preservation: float = 0.7

    # v8: Morphological Grid Detection
    grid_use_morphological: bool = False
    grid_morph_line_lengths: tuple = (5, 9, 15)
    grid_morph_line_width: int = 1
    grid_morph_angles: tuple = (0, 45, 90, 135)
    grid_morph_threshold: float = 0.3

    # v8: Advanced Edge Preservation
    grid_use_advanced_edge: bool = False
    grid_dog_sigma1: float = 1.0
    grid_dog_sigma2: float = 2.0
    grid_diffusion_iterations: int = 10
    grid_diffusion_kappa: float = 30.0
    grid_diffusion_gamma: float = 0.1

    # v8: Hybrid Blending
    grid_hybrid_blend_mode: str = "confidence"
    grid_hybrid_wavelet_weight: float = 0.4
    grid_hybrid_fft_weight: float = 0.3
    grid_hybrid_morph_weight: float = 0.3

    # v9: STFT Adaptive Grid Removal
    grid_stft_period_x: int = 0  # 0 = auto-detect via autocorrelation
    grid_stft_period_y: int = 0  # 0 = auto-detect via autocorrelation
    grid_stft_window_size: int = 63  # LCM(7,9) for full period coverage
    grid_stft_hop_size: int = 16  # STFT hop size
    grid_stft_notch_sigma: float = 1.5  # Gaussian notch width
    grid_stft_base_attenuation: float = 0.15  # Baseline attenuation (0=full removal)
    grid_stft_edge_protection: bool = True  # Use edge protection mask
    grid_stft_edge_preservation: float = 0.5  # Edge blend strength (0-1)
    grid_stft_channel_adaptive: bool = True  # Per-channel modulation-based attenuation
    grid_stft_use_stft: bool = True  # True=STFT local, False=global notch

    # v10: Contour-Based Region Flattening
    contour_period_x: int = 0                  # Grid period X (0=auto-detect)
    contour_period_y: int = 0                  # Grid period Y (0=auto-detect)
    contour_edge_low: float = 80.0             # Canny low threshold
    contour_edge_high: float = 160.0            # Canny high threshold
    contour_confidence_threshold: float = 0.5  # Edge confidence minimum
    contour_min_region_area: int = 50          # Min region area (px)
    contour_flatten_method: str = "median"     # "median" | "trimmed_mean"
    contour_blend_width: int = 1               # Edge blending width (px)
    contour_min_edge_length: int = 15          # Min connected edge length
    contour_chrominance_threshold: float = 15.0 # LAB delta-E threshold (pre-smoothed)

    # v11: Edge-Aware Flat Color Restoration
    eaf_delta_e_high: float = 12.0              # LAB ΔE strong edge threshold
    eaf_delta_e_low: float = 5.0                # LAB ΔE weak edge threshold (hysteresis)
    eaf_chrominance_sigma: float = 3.0          # Chrominance pre-smoothing sigma
    eaf_chrominance_threshold: float = 12.0     # Chrominance ΔE edge threshold
    eaf_persistence_sigma: float = 3.0          # Persistence blur sigma
    eaf_confidence_threshold: float = 0.3       # Final confidence cutoff
    eaf_periodicity_threshold: float = 0.5     # Periodicity rejection threshold
    eaf_min_edge_length: int = 10               # Min connected edge length (px)
    eaf_min_region_area: int = 20               # Min region area (pixels)
    eaf_edge_dilate: int = 1                    # Edge dilation radius
    eaf_blend_width: int = 2                    # Edge zone blending width
    eaf_pre_blur_sigma: float = 1.0             # Post-bilateral Gaussian blur sigma
    eaf_bilateral_iterations: int = 3           # Iterative bilateral filter passes
    eaf_bilateral_d: int = 9                    # Bilateral filter diameter
    eaf_bilateral_sigma_color: float = 50.0     # Bilateral filter color sigma
    eaf_bilateral_sigma_space: float = 50.0     # Bilateral filter spatial sigma

    # v12: Color Quantization Restoration
    cq_k_min: int = 35                      # Minimum palette size
    cq_k_max: int = 80                      # Maximum palette size
    cq_k_selection: str = "elbow"           # k selection method ("elbow", "silhouette", "fixed")
    cq_pre_filter: str = "rolling_guidance"  # Pre-filter ("rolling_guidance", "bilateral", "none")
    cq_rolling_iterations: int = 4          # Rolling Guidance iterations
    cq_rolling_sigma_s: float = 3.0         # Spatial sigma
    cq_rolling_sigma_r: float = 0.05        # Range sigma (0-1 normalized)
    cq_ink_l_threshold: float = 40.0        # Ink line L* threshold
    cq_ink_chroma_threshold: float = 20.0   # Ink line chroma threshold
    cq_quantization_method: str = "guided"  # Assignment method ("hard", "guided")
    cq_min_region_area: int = 50            # Minimum region area (px)
    cq_blend_width: int = 1                 # (legacy, unused in new renderer)
    cq_flatten_strength: float = 0.3        # Gentle flattening toward median (0=none, 1=full)
    cq_adaptive_flatten: bool = True        # Per-region variance-based adaptive flattening
    cq_variance_threshold: float = 150.0    # Variance threshold for adaptive flattening

    # v13: Deep Grid Restoration (Multiplicative Deconvolution)
    dg_method: str = "deconv_neural"         # "deconv_only", "deconv_neural", "neural_only"
    dg_period_detection: str = "auto"        # "auto", "manual"
    dg_manual_period_x: int = 8              # Manual grid period X (if detection=manual)
    dg_manual_period_y: int = 8              # Manual grid period Y (if detection=manual)
    dg_template_method: str = "notch"          # "notch", "gaussian", "median", "mean"
    dg_notch_width: float = 1.5               # FFT notch filter width (sigma in freq bins)
    dg_notch_harmonics: int = 4               # Number of harmonics to suppress
    dg_notch_attenuation: float = 0.0         # Min attenuation at notch (0=full removal)
    dg_deconv_strength: float = 1.0          # Deconvolution strength (0-1, 1=full)
    dg_deconv_clamp_min: float = 0.85         # Template minimum clamp (prevents div-by-near-zero)
    dg_deconv_clamp_max: float = 1.15        # Template maximum clamp
    dg_neural_model: str = "scunet"          # "scunet", "nafnet", "none"
    dg_neural_strength: float = 0.3          # DL refinement blending strength (0-1)
    dg_edge_protection: bool = True          # Ink-line edge protection blending
    dg_ink_l_threshold: float = 40.0         # L* threshold for ink line detection
    dg_edge_enhance: bool = False            # Edge detail transfer after grid removal
    dg_edge_detail_strength: float = 0.5     # Edge detail injection strength (0-1)
    dg_edge_detail_sigma: float = 1.0        # Sigma for detail extraction
    dg_final_sharpen: bool = False           # Adaptive USM at pipeline end
    dg_final_sharpen_strength: float = 1.0   # USM sharpening strength
    dg_final_sharpen_sigma: float = 1.5      # USM Gaussian sigma
    dg_final_sharpen_edge_threshold: float = 0.2  # Edge threshold for USM

    # v14.1: Object-Edge-Only Enhancement (OEE)
    dg_oee_enabled: bool = False              # Enable OEE (replaces edge_enhance + final_sharpen)
    dg_oee_edge_sigma_scale: float = 2.0     # sigma = scale * max(period) for edge detection
    dg_oee_detail_source: str = "original"   # "restored" (safe) or "original" (stronger)
    dg_oee_detail_sigma: float = 1.5         # Detail extraction high-pass sigma
    dg_oee_enhance_strength: float = 0.3     # Enhancement intensity (0-1)
    dg_oee_edge_low: float = 0.05            # Soft threshold lower bound
    dg_oee_edge_high: float = 0.2            # Soft threshold upper bound
    dg_oee_periodicity_rejection: float = 0.0  # Periodicity suppression (0=disabled)


class BaseRestoration(BasePreprocessModule):
    """Base class for restoration modules."""

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config or RestorationConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name."""
        pass

    @abstractmethod
    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Restore the image."""
        pass
