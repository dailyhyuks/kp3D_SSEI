"""Enhanced fading noise restoration (v7).

Extends FadingNoiseRestorer with:
- Adaptive window sizes based on image resolution and noise scale
- PatchMatch-based inpainting for texture-coherent restoration
- Shape-aware noise classification for type-specific inpainting
"""

import cv2
import numpy as np
import time
import torch
from torch import Tensor
from typing import Any, Dict, Optional, Tuple

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import RestorationConfig
from kp3d.modules.restoration.fading_noise import FadingNoiseRestorer
from kp3d.modules.restoration.adaptive_windows import (
    AdaptiveWindowDetector,
    compute_adaptive_windows,
)
from kp3d.modules.restoration.noise_classifier import NoiseShapeClassifier
from kp3d.modules.restoration.patchmatch_inpaint import RestorationPatchMatch


class EnhancedFadingNoiseRestorer(FadingNoiseRestorer):
    """v7 퇴색 노이즈 복원기 - 적응형 윈도우 + PatchMatch + 노이즈 분류

    FadingNoiseRestorer를 상속하여 v7 기능을 추가합니다.
    v7 config 플래그가 모두 비활성화되면 부모 클래스와 동일하게 동작합니다.
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)

        # v7: Adaptive window detector
        self._window_detector = AdaptiveWindowDetector()

        # v7: Noise classifier
        self._noise_classifier = None
        if self.config.use_noise_classification:
            self._noise_classifier = NoiseShapeClassifier(
                strategy_map=self.config.noise_class_inpaint_map.copy(),
            )

        # v7: PatchMatch inpainter
        self._patchmatch = None

    @property
    def name(self) -> str:
        return "enhanced_fading_noise"

    def _get_patchmatch(self) -> RestorationPatchMatch:
        """Lazy-initialize PatchMatch inpainter."""
        if self._patchmatch is None:
            self._patchmatch = RestorationPatchMatch(
                patch_size=self.config.patchmatch_patch_size,
                iterations=self.config.patchmatch_iterations,
                search_samples=self.config.patchmatch_search_samples,
                preserve_texture=self.config.preserve_texture,
                texture_sigma=self.config.texture_blur_sigma,
            )
        return self._patchmatch

    def detect_local_outliers_multiscale(
        self,
        lab: np.ndarray,
        hsv: np.ndarray,
        edge_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """다중 스케일 이상치 탐지 - 적응형 윈도우 지원 (v7)

        use_adaptive_windows=True인 경우 이미지 해상도와 노이즈 스케일에
        따라 윈도우 크기를 자동 조정합니다.

        Args:
            lab: LAB 색공간 이미지
            hsv: HSV 색공간 이미지
            edge_mask: 엣지 마스크

        Returns:
            (outlier_mask, diff_map, debug_info)
        """
        if not self.config.use_adaptive_windows:
            return super().detect_local_outliers_multiscale(lab, hsv, edge_mask)

        # v7: Compute adaptive windows
        original_windows = self.config.window_sizes
        adaptive_windows = self._window_detector.compute(
            lab.shape[:2],
            noise_scale=self.config.noise_scale_estimate,
        )
        self.config.window_sizes = tuple(adaptive_windows)

        try:
            outlier_mask, diff_map, debug_info = super().detect_local_outliers_multiscale(
                lab, hsv, edge_mask
            )
            debug_info['adaptive_windows'] = adaptive_windows
            debug_info['original_windows'] = list(original_windows)
        finally:
            # Restore original windows
            self.config.window_sizes = original_windows

        return outlier_mask, diff_map, debug_info

    def filter_blobs_advanced(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """고급 블롭 필터링 - 노이즈 분류 통합 (v7)

        use_noise_classification=True인 경우 각 블롭의 형태를 분석하여
        노이즈 유형을 분류하고, 분류 결과를 filter_stats에 포함합니다.

        Args:
            mask: 이상치 마스크

        Returns:
            (filtered_mask, filter_stats)
        """
        filtered_mask, filter_stats = super().filter_blobs_advanced(mask)

        if not self.config.use_noise_classification or self._noise_classifier is None:
            return filtered_mask, filter_stats

        # v7: Classify noise blobs
        if np.any(filtered_mask > 0):
            label_types, label_features = self._noise_classifier.classify_mask(filtered_mask)
            summary = self._noise_classifier.get_classification_summary(label_types)

            filter_stats['noise_classification'] = summary
            filter_stats['noise_label_types'] = label_types
            filter_stats['noise_strategy_masks'] = self._noise_classifier.create_strategy_masks(
                filtered_mask, label_types
            )

        return filtered_mask, filter_stats

    def inpaint_patchmatch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """PatchMatch 기반 inpainting (v7)

        Args:
            image: RGB 이미지 (uint8)
            mask: 복원할 영역 마스크 (uint8, 255=inpaint)

        Returns:
            복원된 이미지 (uint8)
        """
        pm = self._get_patchmatch()
        return pm.inpaint(image, mask)

    def inpaint_by_classification(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        strategy_masks: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """노이즈 유형별 최적 inpainting 적용 (v7)

        각 노이즈 유형에 맞는 inpainting 전략을 사용합니다.

        Args:
            image: RGB 이미지 (uint8)
            mask: 전체 노이즈 마스크
            strategy_masks: 전략별 마스크 dict

        Returns:
            복원된 이미지 (uint8)
        """
        result = image.copy()

        for strategy, smask in strategy_masks.items():
            if not np.any(smask > 0):
                continue

            if strategy == "patchmatch":
                result = self.inpaint_patchmatch(result, smask)
            elif strategy == "ns":
                bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                restored_bgr = cv2.inpaint(bgr, smask, self.config.inpaint_radius, cv2.INPAINT_NS)
                result = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            elif strategy == "color_aware":
                result = self.inpaint_color_aware(result, smask)
            else:
                # Default: telea
                result = self.inpaint_noise(result, smask)

        return result

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """v7 퇴색 노이즈 복원

        v7 기능이 비활성화된 경우 부모 클래스의 forward를 그대로 사용합니다.
        v7 기능이 하나라도 활성화된 경우 확장된 파이프라인을 실행합니다.
        """
        # Check if any v7 feature is enabled
        has_v7 = (
            self.config.use_adaptive_windows
            or self.config.use_noise_classification
            or self.config.inpaint_mode == "patchmatch"
        )

        if not has_v7:
            # Pure backward compatibility
            output = super().forward(image, **kwargs)
            output.metadata['method'] = 'enhanced_fading_noise'
            return output

        # v7 pipeline
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        img_np = self._tensor_to_numpy_rgb(image[0])

        # Fast mode: reduce windows
        if self.config.fast_mode and self.config.multi_scale:
            original_windows = self.config.window_sizes
            self.config.window_sizes = tuple(
                w for i, w in enumerate(original_windows) if i % 2 == 0
            )

        # Convert color spaces
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        hsv = None
        if self.config.multi_scale and self.config.adaptive_threshold:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Step 1: Edge detection
        edge_mask = self.detect_edges(gray)
        del gray

        # Step 2: Outlier detection (with adaptive windows if enabled)
        debug_info = {}
        filter_stats = {}

        if self.config.multi_scale:
            outlier_mask, diff_map, debug_info = self.detect_local_outliers_multiscale(
                lab, hsv, edge_mask
            )
            noise_mask, filter_stats = self.filter_blobs_advanced(outlier_mask)
        else:
            outlier_mask, diff_map = self.detect_local_outliers(lab, edge_mask)
            noise_mask = self.filter_small_blobs(outlier_mask)

        # Restore windows if fast mode
        if self.config.fast_mode and self.config.multi_scale:
            self.config.window_sizes = original_windows

        # Step 3: Inpainting with v7 strategies
        if np.any(noise_mask > 0):
            if (
                self.config.use_noise_classification
                and 'noise_strategy_masks' in filter_stats
            ):
                # v7: Type-specific inpainting
                restored = self.inpaint_by_classification(
                    img_np, noise_mask, filter_stats['noise_strategy_masks']
                )
            elif self.config.inpaint_mode == "patchmatch":
                # v7: PatchMatch inpainting
                restored = self.inpaint_patchmatch(img_np, noise_mask)
            elif self.config.inpaint_mode == "color_aware":
                restored = self.inpaint_color_aware(img_np, noise_mask)
            elif self.config.inpaint_mode == "hybrid":
                restored = self.inpaint_noise(img_np, noise_mask)
                restored = self.refine_colors(restored, img_np, noise_mask)
            else:
                restored = self.inpaint_noise(img_np, noise_mask)

            # Texture preservation
            if self.config.preserve_texture:
                texture = self.extract_texture(img_np)
                restored = self.apply_texture(restored, texture, noise_mask)
        else:
            restored = img_np.copy()

        elapsed = time.time() - start

        # Noise count
        if 'total_blobs' in filter_stats:
            noise_count = filter_stats['kept']
        else:
            num_labels = cv2.connectedComponents(noise_mask, connectivity=8)[0]
            noise_count = num_labels - 1

        # Intermediates
        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_to_tensor(img_np),
                'edge_mask': self._numpy_to_tensor(np.stack([edge_mask] * 3, axis=-1)),
                'outlier_mask': self._numpy_to_tensor(np.stack([outlier_mask] * 3, axis=-1)),
                'noise_mask': self._numpy_to_tensor(np.stack([noise_mask] * 3, axis=-1)),
                'diff_map': self._numpy_to_tensor(
                    np.stack(
                        [(diff_map / (diff_map.max() + 1e-8) * 255).astype(np.uint8)] * 3,
                        axis=-1,
                    )
                ),
            }
            if 'threshold_map' in debug_info and debug_info['threshold_map'] is not None:
                threshold_map = debug_info['threshold_map']
                threshold_vis = (threshold_map / (threshold_map.max() + 1e-8) * 255).astype(np.uint8)
                intermediates['threshold_map'] = self._numpy_to_tensor(
                    np.stack([threshold_vis] * 3, axis=-1)
                )

        result = self._numpy_to_tensor(restored).unsqueeze(0)

        metadata = {
            'method': 'enhanced_fading_noise',
            'processing_time': elapsed,
            'noise_spots_detected': noise_count,
            'multi_scale': self.config.multi_scale,
            'adaptive_threshold': self.config.adaptive_threshold,
            'inpaint_mode': self.config.inpaint_mode,
            'preserve_texture': self.config.preserve_texture,
            'v7_adaptive_windows': self.config.use_adaptive_windows,
            'v7_noise_classification': self.config.use_noise_classification,
        }

        if self.config.multi_scale:
            metadata.update({
                'window_sizes': list(self.config.window_sizes),
                'scale_combination': self.config.scale_combination,
                'circularity_threshold': self.config.circularity_threshold,
                'density_filter': self.config.density_filter,
            })
            if filter_stats:
                # Remove non-serializable items
                stats_clean = {
                    k: v for k, v in filter_stats.items()
                    if k != 'noise_strategy_masks' and k != 'noise_label_types'
                }
                metadata['filter_stats'] = stats_clean

        if 'adaptive_windows' in debug_info:
            metadata['adaptive_windows'] = debug_info['adaptive_windows']

        return ModuleOutput(
            result=result,
            intermediate=intermediates,
            metadata=metadata,
        )


__all__ = ["EnhancedFadingNoiseRestorer"]
