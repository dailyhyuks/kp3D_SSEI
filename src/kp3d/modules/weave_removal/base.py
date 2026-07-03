"""Base classes and configuration for the Weave Removal module.

Provides spectral interpolation-based grid artifact removal from
digitized silk-mounted Korean traditional paintings.
"""

from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from loguru import logger

from kp3d.core.base import BasePreprocessModule, ModuleOutput


class WeaveRemovalPreset(str, Enum):
    """Predefined parameter presets for common use cases."""
    QUALITY = "quality"   # Edge preservation priority (alpha=0.7, pr=2)
    CLEAN = "clean"       # Maximum grid removal (alpha=1.0, pr=3)
    V3 = "v3"             # Spatial-Adaptive NLM + Contour Enhancement

    def to_config(self) -> "WeaveRemovalConfig":
        """Create a WeaveRemovalConfig with this preset applied.

        Convenience method for quick preset usage.

        Returns:
            WeaveRemovalConfig with preset values applied.

        Example:
            >>> config = WeaveRemovalPreset.V3.to_config()
            >>> module = WeaveRemovalModule(config)
        """
        return WeaveRemovalConfig(preset=self).apply_preset()


class WeaveRemovalConfig(BaseModel):
    """Configuration for weave/grid artifact removal.

    Attributes:
        preset: Optional preset that overrides individual parameters.
        alpha: Correction strength (0.0=no change, 1.0=full replacement).
        method: Background estimation method.
        patch_size: Size of processing patches (pixels, power of 2 recommended).
        overlap_ratio: Overlap between adjacent patches (0.0-1.0).
        min_prominence: Minimum grid confidence to process a patch.
        channel_mode: Color channel processing mode.
        peak_radius: Interpolation radius for axis peaks.
        cross_peak_radius: Interpolation radius for cross-harmonic peaks.
        adaptive_radius: Enable prominence-based dynamic radius scaling.
        split_radius: Enable separate radius for axis vs cross peaks.
        include_cross_harmonics: Include cross-harmonic (diagonal) peaks.
        cross_harmonic_threshold: Background multiple threshold for cross peaks.
        edge_aware: Enable edge-aware alpha reduction.
        edge_alpha_min: Minimum alpha when edge_aware is enabled.
        contour_boost: Contour enhancement darkening amount (0=disabled).
        contour_block_size: Adaptive threshold block size for contour detection.
        contour_thresh_c: Adaptive threshold constant.
    """
    preset: Optional[WeaveRemovalPreset] = None

    # Core parameters
    alpha: float = Field(default=1.0, ge=0.0, le=1.0)
    method: str = Field(default="annular_mean")
    patch_size: int = Field(default=64, gt=0)
    overlap_ratio: float = Field(default=0.5, ge=0.0, lt=1.0)
    min_prominence: float = Field(default=0.1, ge=0.0)
    channel_mode: str = Field(default="lab_l")

    # Peak interpolation
    peak_radius: int = Field(default=3, ge=1)
    cross_peak_radius: int = Field(default=1, ge=1)
    adaptive_radius: bool = True
    split_radius: bool = True
    include_cross_harmonics: bool = True
    cross_harmonic_threshold: float = Field(default=1.5, gt=0.0)

    # Edge-aware
    edge_aware: bool = True
    edge_alpha_min: float = Field(default=0.3, ge=0.0, le=1.0)

    # Contour enhancement
    contour_boost: float = Field(default=10.0, ge=0.0)
    contour_block_size: int = Field(default=15, gt=0)
    contour_thresh_c: float = Field(default=6.0)

    # V3: Spatial-Adaptive NLM (Stage 2 in V3 pipeline)
    use_nlm_adaptive: bool = False
    nlm_h_base: float = Field(default=10.0, ge=0.0)
    nlm_h_max: float = Field(default=15.0, ge=0.0)
    nlm_h_color_base: float = Field(default=10.0, ge=0.0)
    nlm_h_color_max: float = Field(default=15.0, ge=0.0)
    nlm_narrow_threshold: float = Field(default=8.0, ge=0.0)
    nlm_edge_threshold: float = Field(default=5.0, ge=0.0)
    nlm_template_window: int = Field(default=7, gt=0)
    nlm_search_window: int = Field(default=21, gt=0)
    nlm_n_clusters: int = Field(default=5, ge=2)
    nlm_min_cluster_area: int = Field(default=100, ge=0)
    nlm_blur_sigma: float = Field(default=2.0, ge=0.0)

    def apply_preset(self) -> "WeaveRemovalConfig":
        """Apply preset values if a preset is set.

        Returns:
            Self with preset values applied.
        """
        if self.preset == WeaveRemovalPreset.QUALITY:
            self.alpha = 0.7
            self.peak_radius = 2
            self.cross_peak_radius = 1
        elif self.preset == WeaveRemovalPreset.CLEAN:
            self.alpha = 1.0
            self.peak_radius = 3
            self.cross_peak_radius = 1
        elif self.preset == WeaveRemovalPreset.V3:
            # V3: Split Radius + NLM Adaptive + Contour Enhancement (3-stage)
            # Stage 1: Split Radius (spectral interpolation with split_radius=True)
            self.split_radius = True
            # Stage 2: NLM Adaptive (narrow-region targeted blending)
            self.use_nlm_adaptive = True
            self.nlm_h_base = 10.0
            self.nlm_h_max = 15.0
            self.nlm_h_color_base = 10.0
            self.nlm_h_color_max = 15.0
            self.nlm_narrow_threshold = 8.0
            self.nlm_edge_threshold = 5.0
            self.nlm_template_window = 7
            self.nlm_search_window = 21
            # Stage 3: Contour Enhancement
            self.contour_boost = 10.0
            self.contour_block_size = 15
            self.contour_thresh_c = 6.0
        return self


class WeaveRemovalModule(BasePreprocessModule):
    """Spectral interpolation-based grid artifact removal module.

    Removes periodic weave patterns from digitized silk-mounted Korean
    traditional paintings using FFT-based spectral interpolation with
    harmonic peak detection, followed by optional contour enhancement.

    Pipeline:
        Input BGR -> Patchwise Spectral Interpolation -> Contour Enhancement -> Output BGR

    Attributes:
        config: Module configuration.
    """

    def __init__(
        self,
        config: Optional[WeaveRemovalConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """Initialize the weave removal module.

        Args:
            config: Configuration object. If None, uses defaults.
            device: Computation device (unused for this CPU-based module).
            dtype: Data type (unused for this CPU-based module).
            **kwargs: Additional configuration overrides.
        """
        super().__init__(device=device, dtype=dtype)
        self.config = config or WeaveRemovalConfig(**kwargs)
        if self.config.preset:
            self.config = self.config.apply_preset()
        self._initialized = True
        logger.info(
            f"WeaveRemovalModule initialized: alpha={self.config.alpha}, "
            f"peak_radius={self.config.peak_radius}, "
            f"preset={self.config.preset}"
        )

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Process an image through the weave removal pipeline.

        Converts tensor to BGR numpy, applies the appropriate pipeline
        (V3 3-stage or legacy), then converts back to tensor.

        Pipelines:
            V3 (use_nlm_adaptive=True):
                Stage 1: Split Radius spectral interpolation (contour OFF)
                Stage 2: NLM adaptive blending (narrow regions only)
                Stage 3: Contour enhancement

            Legacy (use_nlm_adaptive=False):
                Stage 1: Spectral interpolation
                Stage 2: Contour enhancement (optional)

        Args:
            image: Input image tensor (B, C, H, W) in [0, 1] range, RGB.
            **kwargs: Override config parameters for this call.

        Returns:
            ModuleOutput with processed image and metadata.
        """
        from kp3d.modules.weave_removal.spectral import process_image_patchwise
        from kp3d.modules.weave_removal.contour import enhance_contours

        # Tensor (B, C, H, W) RGB [0,1] -> BGR uint8
        img_rgb = image[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        img_bgr = (np.clip(img_rgb[:, :, ::-1], 0, 1) * 255).astype(np.uint8)

        cfg = self.config
        confidence = np.array([0.0])  # Default confidence for V3 path

        if cfg.use_nlm_adaptive:
            # V3 pipeline: Split Radius + NLM Adaptive + Contour (3-stage)
            from kp3d.modules.weave_removal.nlm_adaptive import (
                spatial_adaptive_nlm,
                SpatialAdaptiveNLMConfig,
            )

            # Stage 1: Split Radius spectral interpolation (contour OFF)
            split_out, confidence = process_image_patchwise(
                img_bgr,
                patch_size=kwargs.get("patch_size", cfg.patch_size),
                overlap_ratio=kwargs.get("overlap_ratio", cfg.overlap_ratio),
                alpha=kwargs.get("alpha", cfg.alpha),
                method=kwargs.get("method", cfg.method),
                min_prominence=kwargs.get("min_prominence", cfg.min_prominence),
                channel_mode=kwargs.get("channel_mode", cfg.channel_mode),
                edge_aware=kwargs.get("edge_aware", cfg.edge_aware),
                edge_alpha_min=kwargs.get("edge_alpha_min", cfg.edge_alpha_min),
                peak_radius=cfg.peak_radius,
                cross_peak_radius=cfg.cross_peak_radius,
                adaptive_radius=cfg.adaptive_radius,
                split_radius=cfg.split_radius,
                include_cross_harmonics=cfg.include_cross_harmonics,
                cross_harmonic_threshold=cfg.cross_harmonic_threshold,
            )

            # Stage 2: NLM adaptive blending (narrow regions only)
            nlm_cfg = SpatialAdaptiveNLMConfig(
                h_base=cfg.nlm_h_base,
                h_max=cfg.nlm_h_max,
                h_color_base=cfg.nlm_h_color_base,
                h_color_max=cfg.nlm_h_color_max,
                narrow_threshold=cfg.nlm_narrow_threshold,
                edge_threshold=cfg.nlm_edge_threshold,
                template_window=cfg.nlm_template_window,
                search_window=cfg.nlm_search_window,
                n_clusters=cfg.nlm_n_clusters,
                min_cluster_area=cfg.nlm_min_cluster_area,
                blur_sigma=cfg.nlm_blur_sigma,
            )
            denoised = spatial_adaptive_nlm(img_bgr, split_out, nlm_cfg)
        else:
            # Legacy pipeline: Spectral interpolation only
            denoised, confidence = process_image_patchwise(
                img_bgr,
                patch_size=kwargs.get("patch_size", cfg.patch_size),
                overlap_ratio=kwargs.get("overlap_ratio", cfg.overlap_ratio),
                alpha=kwargs.get("alpha", cfg.alpha),
                method=kwargs.get("method", cfg.method),
                min_prominence=kwargs.get("min_prominence", cfg.min_prominence),
                channel_mode=kwargs.get("channel_mode", cfg.channel_mode),
                edge_aware=kwargs.get("edge_aware", cfg.edge_aware),
                edge_alpha_min=kwargs.get("edge_alpha_min", cfg.edge_alpha_min),
                peak_radius=cfg.peak_radius,
                cross_peak_radius=cfg.cross_peak_radius,
                adaptive_radius=cfg.adaptive_radius,
                split_radius=cfg.split_radius,
                include_cross_harmonics=cfg.include_cross_harmonics,
                cross_harmonic_threshold=cfg.cross_harmonic_threshold,
            )

        # Final Stage: Contour enhancement (optional, both pipelines)
        result_bgr = denoised
        if cfg.contour_boost > 0:
            result_bgr = enhance_contours(
                img_bgr, denoised,
                boost=cfg.contour_boost,
                block_size=cfg.contour_block_size,
                thresh_c=cfg.contour_thresh_c,
            )

        # BGR uint8 -> RGB [0,1] tensor
        result_rgb = result_bgr[:, :, ::-1].astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(
            result_rgb.transpose(2, 0, 1).copy()
        ).unsqueeze(0).to(device=self.device, dtype=self.dtype)

        return ModuleOutput(
            result=result_tensor,
            metadata={
                "alpha": cfg.alpha,
                "peak_radius": cfg.peak_radius,
                "preset": cfg.preset.value if cfg.preset else None,
                "mean_confidence": float(np.mean(confidence)),
                "contour_boost": cfg.contour_boost,
                "use_nlm_adaptive": cfg.use_nlm_adaptive,
                "split_radius": cfg.split_radius,
            },
        )

    def process_bgr(
        self,
        img_bgr: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience method for direct BGR numpy processing.

        Skips tensor conversion overhead. Useful for experiment scripts.

        Pipelines:
            V3 (use_nlm_adaptive=True):
                Stage 1: Split Radius spectral interpolation (contour OFF)
                Stage 2: NLM adaptive blending (narrow regions only)
                Stage 3: Contour enhancement

            Legacy (use_nlm_adaptive=False):
                Stage 1: Spectral interpolation
                Stage 2: Contour enhancement (optional)

        Args:
            img_bgr: Input image BGR uint8 (H, W, 3).
            **kwargs: Override config parameters.

        Returns:
            Tuple of (result_bgr, confidence_map).
        """
        from kp3d.modules.weave_removal.spectral import process_image_patchwise
        from kp3d.modules.weave_removal.contour import enhance_contours

        cfg = self.config
        confidence = np.array([0.0])

        if cfg.use_nlm_adaptive:
            # V3 pipeline: Split Radius + NLM Adaptive + Contour (3-stage)
            from kp3d.modules.weave_removal.nlm_adaptive import (
                spatial_adaptive_nlm,
                SpatialAdaptiveNLMConfig,
            )

            # Stage 1: Split Radius spectral interpolation (contour OFF)
            split_out, confidence = process_image_patchwise(
                img_bgr,
                patch_size=kwargs.get("patch_size", cfg.patch_size),
                overlap_ratio=kwargs.get("overlap_ratio", cfg.overlap_ratio),
                alpha=kwargs.get("alpha", cfg.alpha),
                method=kwargs.get("method", cfg.method),
                min_prominence=kwargs.get("min_prominence", cfg.min_prominence),
                channel_mode=kwargs.get("channel_mode", cfg.channel_mode),
                edge_aware=kwargs.get("edge_aware", cfg.edge_aware),
                edge_alpha_min=kwargs.get("edge_alpha_min", cfg.edge_alpha_min),
                peak_radius=cfg.peak_radius,
                cross_peak_radius=cfg.cross_peak_radius,
                adaptive_radius=cfg.adaptive_radius,
                split_radius=cfg.split_radius,
                include_cross_harmonics=cfg.include_cross_harmonics,
                cross_harmonic_threshold=cfg.cross_harmonic_threshold,
            )

            # Stage 2: NLM adaptive blending (narrow regions only)
            nlm_cfg = SpatialAdaptiveNLMConfig(
                h_base=cfg.nlm_h_base,
                h_max=cfg.nlm_h_max,
                h_color_base=cfg.nlm_h_color_base,
                h_color_max=cfg.nlm_h_color_max,
                narrow_threshold=cfg.nlm_narrow_threshold,
                edge_threshold=cfg.nlm_edge_threshold,
                template_window=cfg.nlm_template_window,
                search_window=cfg.nlm_search_window,
                n_clusters=cfg.nlm_n_clusters,
                min_cluster_area=cfg.nlm_min_cluster_area,
                blur_sigma=cfg.nlm_blur_sigma,
            )
            denoised = spatial_adaptive_nlm(img_bgr, split_out, nlm_cfg)
        else:
            # Legacy pipeline: Spectral interpolation only
            denoised, confidence = process_image_patchwise(
                img_bgr,
                patch_size=kwargs.get("patch_size", cfg.patch_size),
                overlap_ratio=kwargs.get("overlap_ratio", cfg.overlap_ratio),
                alpha=kwargs.get("alpha", cfg.alpha),
                method=kwargs.get("method", cfg.method),
                min_prominence=kwargs.get("min_prominence", cfg.min_prominence),
                channel_mode=kwargs.get("channel_mode", cfg.channel_mode),
                edge_aware=kwargs.get("edge_aware", cfg.edge_aware),
                edge_alpha_min=kwargs.get("edge_alpha_min", cfg.edge_alpha_min),
                peak_radius=cfg.peak_radius,
                cross_peak_radius=cfg.cross_peak_radius,
                adaptive_radius=cfg.adaptive_radius,
                split_radius=cfg.split_radius,
                include_cross_harmonics=cfg.include_cross_harmonics,
                cross_harmonic_threshold=cfg.cross_harmonic_threshold,
            )

        # Final Stage: Contour enhancement (optional, both pipelines - process_bgr)
        result = denoised
        if cfg.contour_boost > 0:
            result = enhance_contours(
                img_bgr, denoised,
                boost=cfg.contour_boost,
                block_size=cfg.contour_block_size,
                thresh_c=cfg.contour_thresh_c,
            )

        return result, confidence

    def load_weights(self, checkpoint_path: str) -> None:
        """No-op: this module uses no learned weights.

        Args:
            checkpoint_path: Ignored.
        """
        logger.debug("WeaveRemovalModule uses no learned weights, skipping load")
        self._initialized = True

    @property
    def name(self) -> str:
        """Return the module's unique identifier."""
        return "weave_removal"
