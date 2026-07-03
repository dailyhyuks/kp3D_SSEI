"""Configuration for the Enhancement Pipeline module."""

from pydantic import BaseModel, Field


class EnhancementConfig(BaseModel):
    """Configuration for the Enhancement Pipeline.

    Pipeline:
    - When use_spectral_grid=True (default): SpectralGridRemoval -> Upscale 2x -> (Optional Detail) -> Upscale 2x
    - When use_spectral_grid=False (legacy): Pre-smooth -> Upscale 2x -> Grid Removal -> Upscale 2x
    Result: effective 4x upscale with grid artifact removal.

    Pre-smooth stage uses bilateral filtering to suppress low-contrast
    grid patterns while preserving strong ink edges before upscaling.
    SpectralGridRemoval uses iterative Fourier-domain notch filtering for superior grid removal.
    """

    # === Pre-smooth Settings ===
    enable_pre_smooth: bool = False
    pre_smooth_method: str = "bilateral"
    pre_smooth_iterations: int = Field(default=2, ge=1, le=5)
    bilateral_d: int = Field(default=9, ge=3)
    bilateral_sigma_color: float = Field(default=75.0, gt=0.0)
    bilateral_sigma_space: float = Field(default=75.0, gt=0.0)

    # === Upscaling Settings ===
    upscale_model: str = "RealESRGAN_x2plus"
    upscale_tile_size: int = Field(default=512, gt=0)
    upscale_tile_overlap: int = Field(default=32, ge=0)
    upscale_half_precision: bool = False
    upscale_denoise: bool = False

    # === Grid Removal Settings (v14.1) ===
    grid_period_detection: str = "auto"
    grid_manual_period_x: int = Field(default=8, ge=2)
    grid_manual_period_y: int = Field(default=8, ge=2)
    grid_template_method: str = "notch"
    grid_deconv_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    grid_clamp_min: float = Field(default=0.85, gt=0.0)
    grid_clamp_max: float = Field(default=1.15, gt=0.0)
    grid_edge_protection: bool = True
    grid_ink_l_threshold: float = Field(default=40.0, ge=0.0, le=100.0)
    grid_notch_width: float = Field(default=1.5, gt=0.0)
    grid_notch_harmonics: int = Field(default=4, ge=1)

    # === Object Edge Enhancement (OEE) Settings (v14.1) ===
    oee_enabled: bool = True
    oee_edge_sigma_scale: float = Field(default=2.0, gt=0.0)
    oee_detail_source: str = "original"
    oee_enhance_strength: float = Field(default=0.3, ge=0.0, le=1.0)
    oee_edge_low: float = Field(default=0.05, ge=0.0, le=1.0)
    oee_edge_high: float = Field(default=0.2, ge=0.0, le=1.0)
    oee_periodicity_rejection: float = Field(default=0.0, ge=0.0, le=1.0)

    # === Spectral Grid Removal Settings ===
    use_spectral_grid: bool = True
    spectral_period_min: float = Field(default=4.0, gt=0.0)
    spectral_period_max: float = Field(default=20.0, gt=0.0)
    spectral_butterworth_order: int = Field(default=2, ge=1, le=6)
    spectral_max_iterations: int = Field(default=2, ge=1, le=5)
    spectral_convergence_threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    spectral_min_notch_width: float = Field(default=0.5, gt=0.0)
    spectral_max_notch_width: float = Field(default=3.0, gt=0.0)
    spectral_padding_factor: int = Field(default=2, ge=1, le=4)
    spectral_harmonic_threshold: float = Field(default=2.0, gt=0.0)
    spectral_min_peak_prominence: float = Field(default=3.0, gt=0.0)

    # === Skip Logic Settings ===
    skip_upscale_if_large: bool = True
    max_input_pixels: int = Field(default=4_000_000, gt=0)
    skip_grid_if_undetected: bool = True
    grid_confidence_threshold: float = Field(default=3.0, gt=0.0)

    # === Stage Toggles ===
    enable_first_upscale: bool = True
    enable_grid_removal: bool = True
    enable_second_upscale: bool = True
    enable_detail_enhance: bool = False
    store_intermediates: bool = True

    class Config:
        """Pydantic model configuration."""
        extra = "forbid"
