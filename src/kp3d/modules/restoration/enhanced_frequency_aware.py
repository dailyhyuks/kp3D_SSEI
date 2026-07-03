"""Enhanced frequency-aware restoration (v7).

Extends FrequencyAwareRestorer with:
- Continuous sigma selection (10 levels instead of 3) for smoother
  frequency separation
- Uses EnhancedFadingNoiseRestorer for low-frequency restoration
"""

import cv2
import numpy as np
import time
import torch
from torch import Tensor
from typing import Any, Dict, Optional, Tuple

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import RestorationConfig
from kp3d.modules.restoration.frequency_aware import FrequencyAwareRestorer
from kp3d.modules.restoration.enhanced_fading_noise import EnhancedFadingNoiseRestorer
from kp3d.modules.restoration.adaptive_windows import (
    ContinuousSigmaSelector,
    adaptive_gaussian_blur,
)


class EnhancedFrequencyAwareRestorer(FrequencyAwareRestorer):
    """v7 주파수 인식 복원기 - 연속 Sigma + Enhanced FadingNoise

    FrequencyAwareRestorer를 상속하여 v7 기능을 추가합니다.
    continuous_sigma=False이면 부모 클래스와 동일하게 동작합니다.
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)

        # v7: Replace fading restorer with enhanced version
        self.fading_restorer = EnhancedFadingNoiseRestorer(config=config, **kwargs)

        # v7: Continuous sigma selector
        self._sigma_selector = None
        if self.config.continuous_sigma:
            self._sigma_selector = ContinuousSigmaSelector(
                min_sigma=self.config.sigma_min,
                max_sigma=self.config.sigma_max,
                num_levels=self.config.sigma_levels,
            )

    @property
    def name(self) -> str:
        return "enhanced_frequency_aware"

    def extract_frequencies_adaptive(
        self,
        image: np.ndarray,
        sigma_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """적응형 주파수 분리 - 연속 sigma 지원 (v7)

        continuous_sigma=True인 경우 10단계 sigma 레벨로 부드러운
        주파수 분리를 수행합니다.

        Args:
            image: RGB 이미지 (uint8)
            sigma_map: 픽셀별 sigma 값

        Returns:
            (low_freq, high_freq) float32 배열
        """
        if not self.config.continuous_sigma or self._sigma_selector is None:
            return super().extract_frequencies_adaptive(image, sigma_map)

        img_float = image.astype(np.float32)

        # v7: Use continuous sigma with configurable levels
        low_freq = adaptive_gaussian_blur(
            img_float,
            sigma_map,
            num_levels=self.config.sigma_levels,
            min_sigma=self.config.sigma_min,
            max_sigma=self.config.sigma_max,
        )

        high_freq = img_float - low_freq

        return low_freq, high_freq

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """v7 주파수 인식 복원

        continuous_sigma가 비활성화되고 enhanced fading noise도
        v7 기능이 없으면 부모 클래스와 유사하게 동작합니다.
        """
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        img_np = self._tensor_to_numpy_rgb(image[0])
        h, w = img_np.shape[:2]

        # Step 1: Enhanced edge detection
        edges, edge_debug_maps = self.detect_edges_enhanced(img_np)
        edge_proximity = self.compute_edge_proximity(edges)

        # Step 2: Compute adaptive sigma map
        sigma_map = self.compute_sigma_map(edge_proximity, h, w)

        # Step 3: Separate frequencies (v7: continuous sigma)
        low_freq, high_freq = self.extract_frequencies_adaptive(img_np, sigma_map)

        # Step 4: Restore low frequency using EnhancedFadingNoiseRestorer
        low_freq_uint8 = np.clip(low_freq, 0, 255).astype(np.uint8)
        low_freq_tensor = self._numpy_to_tensor(low_freq_uint8).unsqueeze(0)

        restoration_output = self.fading_restorer.forward(low_freq_tensor)
        restored_low_tensor = restoration_output.result

        restored_low = self._tensor_to_numpy_rgb(restored_low_tensor[0]).astype(np.float32)

        # Apply saturation-based strength
        strength_map = self.compute_restoration_strength(img_np)
        strength_3d = strength_map[:, :, np.newaxis]

        final_low = low_freq * (1 - strength_3d) + restored_low * strength_3d

        # Step 5: Filter high frequency
        filtered_high = self.filter_texture_by_edge(high_freq, edge_proximity)

        # Step 6: Edge injection
        if self.config.edge_boost_strength > 0:
            filtered_high = self.inject_edge_to_highfreq(filtered_high, edges)

        # Step 7: Combine
        result = final_low + filtered_high
        result = np.clip(result, 0, 255).astype(np.uint8)

        elapsed = time.time() - start

        # Intermediates
        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_to_tensor(img_np),
                'edges': self._numpy_to_tensor(np.stack([edges] * 3, axis=-1)),
                'edge_proximity': self._numpy_to_tensor(
                    np.stack([(edge_proximity * 255).astype(np.uint8)] * 3, axis=-1)
                ),
                'sigma_map': self._numpy_to_tensor(
                    np.stack(
                        [((sigma_map / (sigma_map.max() + 1e-8)) * 255).astype(np.uint8)] * 3,
                        axis=-1,
                    )
                ),
                'low_freq': self._numpy_to_tensor(np.clip(low_freq, 0, 255).astype(np.uint8)),
                'high_freq': self._numpy_to_tensor(
                    np.clip(high_freq + 128, 0, 255).astype(np.uint8)
                ),
                'restored_low': self._numpy_to_tensor(
                    np.clip(restored_low, 0, 255).astype(np.uint8)
                ),
                'strength_map': self._numpy_to_tensor(
                    np.stack([(strength_map * 255).astype(np.uint8)] * 3, axis=-1)
                ),
            }
            for key, arr in edge_debug_maps.items():
                if arr is not None:
                    intermediates[f'edge_{key}'] = self._numpy_to_tensor(
                        np.stack([arr] * 3, axis=-1) if arr.ndim == 2 else arr
                    )

        result_tensor = self._numpy_to_tensor(result).unsqueeze(0)

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata={
                'method': 'enhanced_frequency_aware_v7',
                'processing_time': elapsed,
                'base_sigma': self.config.freq_base_sigma,
                'edge_sigma_factor': self.config.freq_edge_sigma_factor,
                'saturation_strength': self.config.freq_saturation_strength,
                'texture_noise_reduction': self.config.freq_texture_noise_reduction,
                'use_color_edge_inference': self.config.use_color_edge_inference,
                'color_edge_weight': self.config.color_edge_weight,
                'edge_boost_strength': self.config.edge_boost_strength,
                'v7_continuous_sigma': self.config.continuous_sigma,
                'v7_sigma_levels': self.config.sigma_levels,
                'fading_noise_metadata': restoration_output.metadata,
            },
        )


__all__ = ["EnhancedFrequencyAwareRestorer"]
