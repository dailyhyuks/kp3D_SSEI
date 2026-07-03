"""Enhancement Pipeline with configurable grid removal strategy.

Default (use_spectral_grid=True):
  SpectralGridRemoval -> Upscale 2x -> (Optional Detail) -> Upscale 2x

Legacy (use_spectral_grid=False):
  Pre-smooth -> Upscale 2x -> Grid Removal (v14.1+OEE) -> Upscale 2x

Provides a single-module interface for the complete enhancement workflow,
yielding an effective 4x upscale with grid artifact removal.
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from kp3d.core.base import BasePreprocessModule, ModuleOutput
from kp3d.modules.enhancement.config import EnhancementConfig
from kp3d.modules.enhancement.skip_logic import GridPresenceChecker, ResolutionChecker


class EnhancementPipeline(BasePreprocessModule):
    """Enhancement pipeline with configurable grid removal and 4x upscaling.

    Designed for digitized Korean paintings that suffer from low resolution
    and periodic grid artifacts from scanning equipment.

    Default mode (use_spectral_grid=True):
        Stage 0: Spectral grid removal at original resolution (prevents grid amplification)
        Stage 1: First upscale 2x (RealESRGAN)
        Stage 2: Optional detail enhancement
        Stage 3: Second upscale 2x (RealESRGAN)

    Legacy mode (use_spectral_grid=False):
        Stage 0: Pre-smooth (bilateral filter)
        Stage 1: First upscale 2x (RealESRGAN)
        Stage 2: Grid removal (MultiplicativeGridRemover)
        Stage 3: Second upscale 2x (RealESRGAN)

    The pipeline uses a single RealESRGAN x2 model instance shared between
    both upscaling stages to save ~300MB GPU memory.

    Note:
        Only supports batch size B=1 due to grid remover's single-image
        processing constraint.
    """

    def __init__(
        self,
        config: Optional[EnhancementConfig] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        """Initialize the enhancement pipeline.

        Args:
            config: Enhancement configuration. Uses defaults if None.
            device: Computation device. Auto-detects CUDA if None.
            dtype: Data type for tensor operations.
            **kwargs: Additional parameters (ignored).
        """
        super().__init__(device=device, dtype=dtype)
        self.config = config or EnhancementConfig()

        # Lazy-loaded components
        self._upscaler = None
        self._grid_remover = None
        self._spectral_grid_remover = None

        # Skip logic utilities
        self._resolution_checker = ResolutionChecker()
        self._grid_checker = GridPresenceChecker(
            confidence_threshold=self.config.grid_confidence_threshold
        )

        logger.info(f"EnhancementPipeline initialized on {self.device}")

    # ==================== Lazy Loading ====================

    @property
    def upscaler(self):
        """Lazy load RealESRGAN x2 upscaler (single instance shared for both stages)."""
        if self._upscaler is None:
            from kp3d.modules.superres.real_esrgan import RealESRGANModule
            from kp3d.modules.superres.base import SuperResConfig, ScaleFactor

            config = SuperResConfig(
                scale=ScaleFactor.X2,
                model_name=self.config.upscale_model,
                tile_size=self.config.upscale_tile_size,
                tile_overlap=self.config.upscale_tile_overlap,
            )
            self._upscaler = RealESRGANModule(
                config=config,
                device=self.device,
                half_precision=self.config.upscale_half_precision,
            )
            logger.info(f"Upscaler loaded: {self.config.upscale_model}")
        return self._upscaler

    @property
    def grid_remover(self):
        """Lazy load MultiplicativeGridRemover."""
        if self._grid_remover is None:
            from kp3d.modules.restoration.multiplicative_grid import (
                MultiplicativeGridRemover,
            )

            self._grid_remover = MultiplicativeGridRemover()
            logger.info("MultiplicativeGridRemover loaded")
        return self._grid_remover

    @property
    def spectral_grid_remover(self):
        """Lazy load SpectralGridRemover."""
        if self._spectral_grid_remover is None:
            from kp3d.modules.enhancement.spectral_grid import SpectralGridRemover

            self._spectral_grid_remover = SpectralGridRemover(self.config)
            logger.info("SpectralGridRemover loaded")
        return self._spectral_grid_remover

    # ==================== Main Forward ====================

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Execute the enhancement pipeline.

        Pipeline order depends on use_spectral_grid config:
        - True: SpectralGridRemoval -> Upscale 2x -> (Detail) -> Upscale 2x
        - False: Pre-smooth -> Upscale 2x -> Grid Removal -> Upscale 2x

        Args:
            image: Input tensor of shape (B, C, H, W) with values in [0, 1].
            **kwargs: Override flags:
                - skip_pre_smooth (bool): Force skip pre-smoothing
                - skip_first_upscale (bool): Force skip first upscale
                - skip_grid_removal (bool): Force skip grid removal
                - skip_second_upscale (bool): Force skip second upscale
                - skip_spectral_grid (bool): Force skip spectral grid removal
                - skip_detail_enhance (bool): Force skip detail enhancement

        Returns:
            ModuleOutput with:
                - result: Enhanced image tensor
                - intermediate: Dict of stage outputs (if store_intermediates=True)
                - metadata: Timing, grid detection info, stage execution details
        """
        total_start = time.time()
        b, c, h_in, w_in = image.shape

        intermediates: Dict[str, Tensor] = {}
        metadata: Dict[str, Any] = {
            "input_size": (h_in, w_in),
            "stages": [],
            "pipeline_mode": "spectral" if self.config.use_spectral_grid else "legacy",
        }

        if self.config.store_intermediates:
            intermediates["input"] = image.clone()

        current = image

        if self.config.use_spectral_grid:
            # ==================== SPECTRAL MODE ====================

            # Stage 0: Spectral Grid Removal (at original resolution)
            skip_spectral = kwargs.get("skip_spectral_grid", False)
            if not skip_spectral and self.config.enable_grid_removal:
                current, stage_meta = self._spectral_grid_remove(current)
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_spectral_grid"] = current.clone()
            else:
                metadata["stages"].append({"name": "spectral_grid_removal", "skipped": True})

            # Stage 1: First Upscale 2x
            skip_first = kwargs.get("skip_first_upscale", False)
            if not skip_first and self.config.enable_first_upscale:
                if self.config.skip_upscale_if_large:
                    skip_first = self._resolution_checker.should_skip(
                        current, self.config.max_input_pixels
                    )

            if not skip_first and self.config.enable_first_upscale:
                current, stage_meta = self._upscale_2x(current, "upscale_1")
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_upscale_1"] = current.clone()
            else:
                metadata["stages"].append({"name": "upscale_1", "skipped": True})

            # Stage 2: Optional Detail Enhancement
            skip_detail = kwargs.get("skip_detail_enhance", False)
            if not skip_detail and self.config.enable_detail_enhance:
                current, stage_meta = self._detail_enhance(current)
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_detail_enhance"] = current.clone()
            else:
                metadata["stages"].append({"name": "detail_enhance", "skipped": True})

            # Stage 3: Second Upscale 2x
            skip_second = kwargs.get("skip_second_upscale", False)
            if not skip_second and self.config.enable_second_upscale:
                if self.config.skip_upscale_if_large:
                    skip_second = self._resolution_checker.should_skip(
                        current, self.config.max_input_pixels * 4
                    )

            if not skip_second and self.config.enable_second_upscale:
                current, stage_meta = self._upscale_2x(current, "upscale_2")
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_upscale_2"] = current.clone()
            else:
                metadata["stages"].append({"name": "upscale_2", "skipped": True})

        else:
            # ==================== LEGACY MODE ====================

            # Stage 0: Pre-smooth
            skip_smooth = kwargs.get("skip_pre_smooth", False)
            if not skip_smooth and self.config.enable_pre_smooth:
                current, stage_meta = self._pre_smooth(current)
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_pre_smooth"] = current.clone()
            else:
                metadata["stages"].append({"name": "pre_smooth", "skipped": True})

            # Stage 1: First Upscale 2x
            skip_first = kwargs.get("skip_first_upscale", False)
            if not skip_first and self.config.enable_first_upscale:
                if self.config.skip_upscale_if_large:
                    skip_first = self._resolution_checker.should_skip(
                        current, self.config.max_input_pixels
                    )

            if not skip_first and self.config.enable_first_upscale:
                current, stage_meta = self._upscale_2x(current, "upscale_1")
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_upscale_1"] = current.clone()
            else:
                metadata["stages"].append({"name": "upscale_1", "skipped": True})

            # Stage 2: Grid Removal
            skip_grid = kwargs.get("skip_grid_removal", False)
            if not skip_grid and self.config.enable_grid_removal:
                if self.config.skip_grid_if_undetected:
                    grid_detected, grid_info = self._check_grid_presence(current)
                    metadata["grid_detection"] = grid_info
                    skip_grid = not grid_detected
                else:
                    metadata["grid_detection"] = {"skipped_check": True}

            if not skip_grid and self.config.enable_grid_removal:
                current, stage_meta = self._remove_grid(current)
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_grid_removal"] = current.clone()
            else:
                metadata["stages"].append({"name": "grid_removal", "skipped": True})

            # Stage 3: Second Upscale 2x
            skip_second = kwargs.get("skip_second_upscale", False)
            if not skip_second and self.config.enable_second_upscale:
                if self.config.skip_upscale_if_large:
                    skip_second = self._resolution_checker.should_skip(
                        current, self.config.max_input_pixels * 4
                    )

            if not skip_second and self.config.enable_second_upscale:
                current, stage_meta = self._upscale_2x(current, "upscale_2")
                metadata["stages"].append(stage_meta)
                if self.config.store_intermediates:
                    intermediates["after_upscale_2"] = current.clone()
            else:
                metadata["stages"].append({"name": "upscale_2", "skipped": True})

        # ==================== Finalize ====================
        _, _, h_out, w_out = current.shape
        metadata["output_size"] = (h_out, w_out)
        metadata["effective_scale"] = (h_out / h_in, w_out / w_in)
        metadata["total_time"] = time.time() - total_start

        return ModuleOutput(
            result=current,
            intermediate=intermediates,
            metadata=metadata,
        )

    # ==================== Sub-steps ====================

    def _spectral_grid_remove(self, image: Tensor) -> Tuple[Tensor, Dict]:
        """Remove grid artifacts using SpectralGridRemover at original resolution.

        Converts tensor to BGR uint8 for processing, then back to tensor.
        Falls back to passthrough if removal fails.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Tuple of (processed_tensor, stage_metadata).
        """
        stage_start = time.time()
        meta = {"name": "spectral_grid_removal", "method": "butterworth_notch"}

        try:
            # Tensor -> BGR uint8
            img_np = image[0].detach().cpu().numpy()  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C) RGB
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Call SpectralGridRemover
            result_bgr, spectral_meta = self.spectral_grid_remover.process(img_bgr)

            # BGR uint8 -> tensor
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(
                result_rgb.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            result_tensor = result_tensor.to(device=self.device, dtype=self.dtype)

            meta["success"] = True
            meta.update(spectral_meta)

        except Exception as e:
            logger.warning(f"Spectral grid removal failed, passing through: {e}")
            result_tensor = image
            meta["method"] = "passthrough"
            meta["success"] = False
            meta["error"] = str(e)

        meta["time"] = time.time() - stage_start
        return result_tensor, meta

    def _detail_enhance(self, image: Tensor) -> Tuple[Tensor, Dict]:
        """Apply optional detail enhancement after first upscale.

        Uses adaptive unsharp masking to enhance edges and fine details
        without amplifying noise or grid residuals.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Tuple of (enhanced_tensor, stage_metadata).
        """
        stage_start = time.time()
        meta = {"name": "detail_enhance", "method": "adaptive_usm"}

        try:
            # Tensor -> BGR uint8
            img_np = image[0].detach().cpu().numpy()  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C) RGB
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Adaptive unsharp mask: enhance = original + strength * (original - blurred)
            blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2.0)
            strength = self.config.oee_enhance_strength
            enhanced = cv2.addWeighted(img_bgr, 1.0 + strength, blurred, -strength, 0)

            # BGR uint8 -> tensor
            result_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(
                result_rgb.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            result_tensor = result_tensor.to(device=self.device, dtype=self.dtype)

            meta["success"] = True
            meta["strength"] = strength

        except Exception as e:
            logger.warning(f"Detail enhancement failed, passing through: {e}")
            result_tensor = image
            meta["success"] = False
            meta["error"] = str(e)

        meta["time"] = time.time() - stage_start
        return result_tensor, meta

    def _pre_smooth(self, image: Tensor) -> Tuple[Tensor, Dict]:
        """Apply bilateral filtering to suppress grid patterns.

        Bilateral filter preserves strong edges (ink strokes) while
        smoothing low-contrast periodic patterns (grid artifacts).
        Applied multiple iterations for cumulative effect.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Tuple of (smoothed_tensor, stage_metadata).
        """
        stage_start = time.time()
        meta = {
            "name": "pre_smooth",
            "method": self.config.pre_smooth_method,
            "iterations": self.config.pre_smooth_iterations,
        }

        try:
            # Tensor -> uint8 BGR for OpenCV bilateral filter
            img_np = image[0].detach().cpu().numpy()  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C) RGB
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Apply bilateral filter N times
            result = img_bgr
            for i in range(self.config.pre_smooth_iterations):
                result = cv2.bilateralFilter(
                    result,
                    d=self.config.bilateral_d,
                    sigmaColor=self.config.bilateral_sigma_color,
                    sigmaSpace=self.config.bilateral_sigma_space,
                )

            # BGR uint8 -> tensor
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(
                result_rgb.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            result_tensor = result_tensor.to(device=self.device, dtype=self.dtype)

            meta["success"] = True
        except Exception as e:
            logger.warning(f"Pre-smooth failed, passing through: {e}")
            result_tensor = image
            meta["success"] = False
            meta["error"] = str(e)

        meta["time"] = time.time() - stage_start
        return result_tensor, meta

    def _upscale_2x(self, image: Tensor, stage_name: str) -> Tuple[Tensor, Dict]:
        """Upscale image by 2x using RealESRGAN.

        Falls back to bicubic interpolation if RealESRGAN fails.

        Args:
            image: Input tensor (B, C, H, W).
            stage_name: Name for metadata tracking.

        Returns:
            Tuple of (upscaled_tensor, stage_metadata).
        """
        from kp3d.modules.superres.base import ScaleFactor

        stage_start = time.time()
        meta = {"name": stage_name, "method": "real_esrgan"}

        try:
            output = self.upscaler.forward(
                image,
                scale=ScaleFactor.X2,
                denoise=self.config.upscale_denoise,
            )
            result = output.result
            meta["success"] = True
        except Exception as e:
            logger.warning(f"RealESRGAN failed at {stage_name}, using bicubic fallback: {e}")
            result = F.interpolate(
                image, scale_factor=2, mode="bicubic", align_corners=False
            ).clamp(0, 1)
            meta["method"] = "bicubic_fallback"
            meta["success"] = False
            meta["error"] = str(e)

        meta["time"] = time.time() - stage_start
        meta["output_shape"] = list(result.shape)
        return result, meta

    def _remove_grid(self, image: Tensor) -> Tuple[Tensor, Dict]:
        """Remove grid artifacts using MultiplicativeGridRemover.

        Converts tensor to BGR uint8 for processing, then back to tensor.
        Falls back to passthrough if grid removal fails.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Tuple of (processed_tensor, stage_metadata).
        """
        stage_start = time.time()
        meta = {"name": "grid_removal", "method": "multiplicative_v14.1"}

        try:
            # Tensor -> BGR uint8 (take first batch item)
            img_np = image[0].detach().cpu().numpy()  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C) RGB
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Call MultiplicativeGridRemover with v14.1 OEE parameters
            result_bgr, grid_intermediates = self.grid_remover.process(
                image_bgr=img_bgr,
                period_detection=self.config.grid_period_detection,
                manual_period_x=self.config.grid_manual_period_x,
                manual_period_y=self.config.grid_manual_period_y,
                template_method=self.config.grid_template_method,
                deconv_strength=self.config.grid_deconv_strength,
                clamp_min=self.config.grid_clamp_min,
                clamp_max=self.config.grid_clamp_max,
                edge_protection=self.config.grid_edge_protection,
                ink_l_threshold=self.config.grid_ink_l_threshold,
                notch_width=self.config.grid_notch_width,
                notch_harmonics=self.config.grid_notch_harmonics,
                # OEE parameters
                object_edge_enhance=self.config.oee_enabled,
                oee_edge_sigma_scale=self.config.oee_edge_sigma_scale,
                oee_detail_source=self.config.oee_detail_source,
                oee_enhance_strength=self.config.oee_enhance_strength,
                oee_edge_low=self.config.oee_edge_low,
                oee_edge_high=self.config.oee_edge_high,
                oee_periodicity_rejection=self.config.oee_periodicity_rejection,
            )

            # BGR uint8 -> tensor (B, C, H, W) in [0, 1]
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_tensor = torch.from_numpy(
                result_rgb.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
            result_tensor = result_tensor.to(device=self.device, dtype=self.dtype)

            meta["success"] = True
            meta["period_x"] = grid_intermediates.get("period_x")
            meta["period_y"] = grid_intermediates.get("period_y")

        except Exception as e:
            logger.warning(f"Grid removal failed, passing through: {e}")
            result_tensor = image
            meta["method"] = "passthrough"
            meta["success"] = False
            meta["error"] = str(e)

        meta["time"] = time.time() - stage_start
        return result_tensor, meta

    def _check_grid_presence(self, image: Tensor) -> Tuple[bool, Dict]:
        """Check if grid pattern is present in the image.

        Converts tensor to BGR for FFT analysis.

        Args:
            image: Input tensor (B, C, H, W) in [0, 1].

        Returns:
            Tuple of (grid_detected, detection_info).
        """
        try:
            # Tensor -> BGR uint8
            img_np = image[0].detach().cpu().numpy()  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C) RGB
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            return self._grid_checker.check(img_bgr)
        except Exception as e:
            logger.warning(f"Grid detection check failed: {e}")
            # Default to running grid removal if check fails
            return True, {"error": str(e)}

    # ==================== Interface ====================

    def load_weights(self, checkpoint_path: str) -> None:
        """Load weights (triggers lazy loading of upscaler).

        Args:
            checkpoint_path: Path to checkpoint (passed to upscaler).
        """
        # Force upscaler initialization
        _ = self.upscaler
        self._initialized = True

    @property
    def name(self) -> str:
        """Return the module's unique identifier name."""
        return "enhancement"
