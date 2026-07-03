"""Grid texture removal and fading restoration for per-object processing.

Uses v12 ColorQuantizationProcessor as primary restoration engine.
Exploits the limited color palette (5-20 colors) of Korean traditional
paintings to remove grid texture and restore faded pigments.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

from kp3d.modules.restoration_v2.config import SegmentAwareRestorationConfig

logger = logging.getLogger(__name__)


class FadingRestorer:
    """Removes grid texture and restores faded pigment regions.

    Uses v12 ColorQuantizationProcessor:
    1. Rolling Guidance Filter (remove grid texture)
    2. LAB K-means palette extraction
    3. Guided pixel assignment
    4. Ink line detection and preservation
    5. Region refinement
    6. Final rendering (region-boundary-aware smoothing)

    Falls back to bilateral+guided filter for very small crops.
    """

    def __init__(
        self,
        config: SegmentAwareRestorationConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._cq_processor = None
        self._neural_refiner = None

    def _get_cq_processor(self):
        """Lazily load ColorQuantizationProcessor."""
        if self._cq_processor is None:
            from kp3d.modules.restoration.color_quantization import ColorQuantizationProcessor
            self._cq_processor = ColorQuantizationProcessor(
                k_min=self.config.cq_k_min,
                k_max=self.config.cq_k_max,
                k_selection=self.config.cq_k_selection,
                pre_filter=self.config.cq_pre_filter,
                rolling_iterations=self.config.cq_rolling_iterations,
                rolling_sigma_s=self.config.cq_rolling_sigma_s,
                rolling_sigma_r=self.config.cq_rolling_sigma_r,
                ink_l_threshold=self.config.cq_ink_l_threshold,
                ink_chroma_threshold=self.config.cq_ink_chroma_threshold,
                quantization_method=self.config.cq_quantization_method,
                min_region_area=self.config.cq_min_region_area,
                flatten_strength=self.config.cq_flatten_strength,
                adaptive_flatten=self.config.cq_adaptive_flatten,
                variance_threshold=self.config.cq_variance_threshold,
            )
        return self._cq_processor

    def _get_neural_refiner(self):
        """Lazily load neural residual refiner."""
        if self._neural_refiner is None:
            from kp3d.modules.restoration.neural_residual import NeuralResidualRefiner
            self._neural_refiner = NeuralResidualRefiner(
                model_name=self.config.neural_model,
                device=self.device,
            )
        return self._neural_refiner

    def _apply_cq_restoration(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply v12 ColorQuantizationProcessor restoration."""
        processor = self._get_cq_processor()
        result, intermediates = processor.process(image_bgr)

        k = intermediates.get("k", -1)
        n_regions = intermediates.get("n_regions", -1)
        logger.debug(
            f"CQ restoration: k={k}, regions={n_regions}"
        )

        return result

    def _apply_iterative_bilateral(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply iterative bilateral filtering (fallback for small crops)."""
        result = image_bgr
        for _ in range(self.config.bilateral_iterations):
            result = cv2.bilateralFilter(
                result,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )
        return result

    def _apply_guided_filter(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply guided filter for edge-preserving smoothing (bilateral fallback)."""
        if not self.config.use_guided_filter:
            return image_bgr
        try:
            return cv2.ximgproc.guidedFilter(
                guide=image_bgr,
                src=image_bgr,
                radius=self.config.guided_radius,
                eps=self.config.guided_eps,
            )
        except AttributeError:
            logger.warning("cv2.ximgproc not available, skipping guided filter")
            return image_bgr

    def _apply_neural_refinement(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply SCUNet neural refinement."""
        refiner = self._get_neural_refiner()
        return refiner.refine(image_bgr, strength=self.config.neural_strength)

    def _should_use_cq(self, image_bgr: np.ndarray) -> bool:
        """Determine if CQ should be used based on crop size.

        ColorQuantizationProcessor needs enough pixels for K-means clustering.
        Very small crops fall back to bilateral filter.
        """
        h, w = image_bgr.shape[:2]
        min_size = self.config.cq_min_crop_size
        return h >= min_size and w >= min_size

    def restore(
        self,
        image_bgr: np.ndarray,
        object_mask: np.ndarray,
        ink_mask: np.ndarray,
    ) -> np.ndarray:
        """Restore faded pigment regions and remove grid texture.

        Args:
            image_bgr: Cropped object image (H, W, 3) uint8 BGR.
            object_mask: Binary mask of object region (H, W) uint8.
            ink_mask: Binary mask of ink line pixels (H, W) uint8.

        Returns:
            Restored image (H, W, 3) uint8 BGR. Ink regions unchanged.
        """
        # Create processing mask: object minus ink
        process_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(ink_mask))
        process_float = process_mask.astype(np.float32) / 255.0

        method = self.config.fading_method
        use_cq = method in ("cq", "cq_neural")

        if use_cq and self._should_use_cq(image_bgr):
            # Primary: v12 ColorQuantizationProcessor
            try:
                result = self._apply_cq_restoration(image_bgr)
            except Exception as e:
                logger.warning(
                    f"CQ restoration failed, falling back to bilateral: {e}"
                )
                result = self._apply_iterative_bilateral(image_bgr)
                result = self._apply_guided_filter(result)
        elif method in ("deconv_neural", "deconv_only"):
            # v13: MultiplicativeGridRemover
            try:
                from kp3d.modules.restoration.multiplicative_grid import MultiplicativeGridRemover
                grid_remover = MultiplicativeGridRemover()
                result, _ = grid_remover.process(
                    image_bgr,
                    period_detection=self.config.grid_period_detection,
                    template_method=self.config.grid_template_method,
                    deconv_strength=self.config.grid_deconv_strength,
                    clamp_min=self.config.grid_clamp_min,
                    clamp_max=self.config.grid_clamp_max,
                    edge_protection=self.config.grid_edge_protection,
                    ink_l_threshold=self.config.grid_ink_l_threshold,
                )
            except Exception as e:
                logger.warning(f"Grid deconv failed, falling back to bilateral: {e}")
                result = self._apply_iterative_bilateral(image_bgr)
                result = self._apply_guided_filter(result)
        else:
            # Fallback: bilateral + guided filter
            result = self._apply_iterative_bilateral(image_bgr)
            result = self._apply_guided_filter(result)

        # Neural refinement (if configured)
        if method in ("cq_neural", "deconv_neural", "bilateral_neural"):
            try:
                result = self._apply_neural_refinement(result)
            except Exception as e:
                logger.warning(f"Neural refinement failed, continuing without: {e}")

        # Apply only to non-ink object regions
        output = image_bgr.copy().astype(np.float32)
        process_3ch = process_float[:, :, np.newaxis]

        output = output * (1.0 - process_3ch) + result.astype(np.float32) * process_3ch
        return np.clip(output, 0, 255).astype(np.uint8)
