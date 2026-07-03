"""Configuration for Segment-Aware Restoration (v2).

Per-object restoration applied after segmentation, avoiding background
grid interference and color bleeding between objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SegmentAwareRestorationConfig:
    """Configuration for segment-aware per-object restoration.

    Uses v12 ColorQuantizationProcessor as primary restoration method.
    Exploits limited color palette of Korean traditional paintings.
    """

    # Cropping
    crop_padding_px: int = 32
    min_object_area_px: int = 500

    # Fading restoration method
    # "cq": v12 ColorQuantization (default)
    # "cq_neural": v12 CQ + SCUNet
    # "deconv_only": v13 multiplicative deconvolution
    # "deconv_neural": v13 deconv + SCUNet
    # "bilateral_only": fallback bilateral filter
    # "bilateral_neural": bilateral + SCUNet
    fading_method: str = "cq"

    # --- v12 ColorQuantization ---
    cq_k_min: int = 35
    cq_k_max: int = 80
    cq_k_selection: str = "elbow"
    cq_pre_filter: str = "rolling_guidance"
    cq_rolling_iterations: int = 4
    cq_rolling_sigma_s: float = 3.0
    cq_rolling_sigma_r: float = 0.05
    cq_ink_l_threshold: float = 40.0
    cq_ink_chroma_threshold: float = 20.0
    cq_quantization_method: str = "guided"
    cq_min_region_area: int = 50
    cq_flatten_strength: float = 0.3
    cq_adaptive_flatten: bool = True
    cq_variance_threshold: float = 150.0
    cq_min_crop_size: int = 64  # Minimum crop size for CQ (needs enough pixels for k-means)

    # --- v13 Multiplicative Grid Removal (kept for comparison) ---
    grid_period_detection: str = "auto"
    grid_manual_period_x: int = 8
    grid_manual_period_y: int = 8
    grid_min_crop_size: int = 64
    grid_template_method: str = "median"
    grid_deconv_strength: float = 1.0
    grid_clamp_min: float = 0.5
    grid_clamp_max: float = 2.0
    grid_edge_protection: bool = True
    grid_ink_l_threshold: float = 40.0
    grid_notch_width: float = 1.5
    grid_notch_harmonics: int = 4
    grid_notch_attenuation: float = 0.0
    grid_edge_enhance: bool = False
    grid_edge_detail_strength: float = 0.5
    grid_edge_detail_sigma: float = 1.0
    grid_final_sharpen: bool = False
    grid_final_sharpen_strength: float = 1.0
    grid_final_sharpen_sigma: float = 1.5
    grid_final_sharpen_edge_threshold: float = 0.2
    grid_oee_enabled: bool = False
    grid_oee_edge_sigma_scale: float = 2.0
    grid_oee_detail_source: str = "original"
    grid_oee_detail_sigma: float = 1.5
    grid_oee_enhance_strength: float = 0.3
    grid_oee_edge_low: float = 0.05
    grid_oee_edge_high: float = 0.2

    # --- Bilateral filter (fallback for small crops) ---
    bilateral_d: int = 15
    bilateral_sigma_color: float = 120.0
    bilateral_sigma_space: float = 120.0
    bilateral_iterations: int = 2

    # Guided filter (used with bilateral fallback)
    use_guided_filter: bool = True
    guided_radius: int = 8
    guided_eps: float = 1000.0

    # --- Neural refinement (SCUNet) ---
    neural_model: str = "scunet"
    neural_strength: float = 0.4

    # --- Denoising (Non-local means) ---
    denoise_h: float = 8.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21

    # --- Ink line protection ---
    ink_l_threshold: float = 25.0
    ink_protection_strength: float = 0.85
    ink_morph_open_size: int = 3

    # --- Boundary blending ---
    feather_radius_px: int = 5

    # --- Background handling ---
    skip_background: bool = True

    # --- Color normalization (CLAHE) ---
    use_clahe: bool = False  # Disabled by default - v12 handles color internally
    clahe_clip_limit: float = 3.0
    clahe_grid_size: int = 4
