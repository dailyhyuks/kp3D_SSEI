"""Frequency-aware restoration for traditional Korean paintings.

Separates image into low and high frequency components:
- Low frequency (color/tone): Restored to fix fading
- High frequency (texture/brushwork): Preserved from original

This approach maintains uniform sharpness across the image,
avoiding the blur/sharp inconsistency of edge-only sharpening.

v6: Added color-based edge inference for paintings without clear outlines.
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig
from kp3d.modules.restoration.fading_noise import FadingNoiseRestorer
from kp3d.modules.edge.color_edge_inference import ColorEdgeInference, ColorEdgeConfig


class FrequencyAwareRestorer(BaseRestoration):
    """주파수 인식 복원기 (v6)

    저주파(색상/톤)만 복원하고 고주파(텍스처/붓질)는 원본을 보존합니다.
    Edge-guided sigma와 Saturation-based strength를 적용합니다.

    v6: 색상 기반 엣지 추론 추가 - 윤곽선이 없거나 희미한 그림 지원
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)

        # Create fading noise restorer for low-frequency restoration
        self.fading_restorer = FadingNoiseRestorer(config=config, **kwargs)

        # Create color edge inference module
        color_edge_config = ColorEdgeConfig(
            delta_e_threshold=self.config.delta_e_threshold,
            superpixel_segments=self.config.superpixel_segments,
            weak_edge_boost=self.config.weak_edge_boost,
        )
        self.color_edge_inference = ColorEdgeInference(
            color_config=color_edge_config,
            device=self.device,
            dtype=self.dtype,
        )

        self._initialized = True

    @property
    def name(self) -> str:
        return "frequency_aware"

    def load_weights(self, checkpoint_path: str) -> None:
        self._initialized = True

    def _tensor_to_numpy_rgb(self, tensor: Tensor) -> np.ndarray:
        """Tensor to RGB numpy array."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = tensor.cpu().numpy()
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        return (np.clip(arr, 0, 1) * 255).astype(np.uint8)

    def _numpy_to_tensor(self, arr: np.ndarray) -> Tensor:
        """RGB numpy array to tensor."""
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)

    def detect_edges_canny(self, gray: np.ndarray) -> np.ndarray:
        """Multi-scale Canny edge detection."""
        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-scale Canny
        edges1 = cv2.Canny(filtered, 30, 80)
        edges2 = cv2.Canny(filtered, 50, 150)
        edges = np.maximum(edges1, edges2)

        return edges

    def detect_edges_enhanced(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Enhanced edge detection combining Canny and color inference.

        For paintings without clear outlines, color-based edge inference
        fills in the missing edges by detecting color boundaries.

        Args:
            image: RGB image (uint8)

        Returns:
            Tuple of (combined_edges, debug_maps)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Step 1: Traditional Canny edges
        canny_edges = self.detect_edges_canny(gray)

        debug_maps = {'canny': canny_edges}

        # Step 2: Color-based edge inference (if enabled)
        if self.config.use_color_edge_inference:
            color_edges, color_intermediates = self.color_edge_inference.infer_edges(image)
            debug_maps['color_inferred'] = color_edges
            debug_maps.update({f'color_{k}': v for k, v in color_intermediates.items()})

            # Step 3: Combine Canny and color edges
            canny_weight = 1.0 - self.config.color_edge_weight
            color_weight = self.config.color_edge_weight

            # Normalize to float
            canny_float = canny_edges.astype(np.float32) / 255.0
            color_float = color_edges.astype(np.float32) / 255.0

            # Weighted combination with max fallback
            # Use max to ensure we don't lose strong edges from either source
            combined = np.maximum(
                canny_float * canny_weight + color_float * color_weight,
                np.maximum(canny_float * 0.5, color_float * 0.5)
            )

            combined_edges = (np.clip(combined, 0, 1) * 255).astype(np.uint8)
            debug_maps['combined'] = combined_edges
        else:
            combined_edges = canny_edges

        return combined_edges, debug_maps

    def inject_edge_to_highfreq(
        self,
        high_freq: np.ndarray,
        edges: np.ndarray
    ) -> np.ndarray:
        """Inject inferred edges into high frequency component.

        Instead of post-processing, we add edge information directly
        to the high frequency before combining with low frequency.
        This produces more natural edge enhancement.

        Args:
            high_freq: High frequency component (float32)
            edges: Edge map (uint8)

        Returns:
            Enhanced high frequency (float32)
        """
        strength = self.config.edge_boost_strength
        if strength <= 0:
            return high_freq

        # Normalize edge map to [-1, 1] range for edge injection
        edge_norm = edges.astype(np.float32) / 255.0

        # Smooth edges for natural blending
        edge_smooth = cv2.GaussianBlur(edge_norm, (5, 5), 1.0)

        # Create edge gradient (Sobel) for directional enhancement
        edge_dx = cv2.Sobel(edge_smooth, cv2.CV_32F, 1, 0, ksize=3)
        edge_dy = cv2.Sobel(edge_smooth, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(edge_dx**2 + edge_dy**2)

        # Normalize magnitude
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()

        # Convert to 3-channel for RGB
        edge_3d = edge_magnitude[:, :, np.newaxis]

        # Scale factor for injection (negative = darkening at edges)
        # This creates subtle contrast at edge boundaries
        edge_injection = edge_3d * strength * 30  # Scale to visible range

        # Add to high frequency (edges become sharper)
        enhanced_high = high_freq - edge_injection  # Subtract for darkening effect

        return enhanced_high

    def compute_edge_proximity(self, edges: np.ndarray) -> np.ndarray:
        """Compute smooth edge proximity map.

        Args:
            edges: Binary edge map (0 or 255)

        Returns:
            Proximity map (0-1, 1 = on edge, 0 = far from edge)
        """
        sigma = self.config.freq_edge_proximity_sigma

        # Distance transform (distance to nearest edge)
        # Invert edges so we measure distance FROM edges
        non_edges = (edges == 0).astype(np.uint8)
        dist = cv2.distanceTransform(non_edges, cv2.DIST_L2, 5)

        # Convert distance to proximity using Gaussian falloff
        proximity = np.exp(-(dist ** 2) / (2 * sigma ** 2))

        return proximity.astype(np.float32)

    def compute_sigma_map(
        self,
        edge_proximity: np.ndarray,
        h: int,
        w: int
    ) -> np.ndarray:
        """Compute per-pixel sigma map for frequency separation.

        Near edges: smaller sigma (preserve detail)
        Far from edges: larger sigma (smooth restoration)

        Args:
            edge_proximity: Edge proximity map (0-1)
            h, w: Image dimensions

        Returns:
            Sigma map for Gaussian blur
        """
        base_sigma = self.config.freq_base_sigma
        edge_factor = self.config.freq_edge_sigma_factor

        # Linear interpolation: high proximity = low sigma
        # sigma = base * (edge_factor + (1 - edge_factor) * (1 - proximity))
        sigma_map = base_sigma * (edge_factor + (1 - edge_factor) * (1 - edge_proximity))

        return sigma_map

    def extract_frequencies_adaptive(
        self,
        image: np.ndarray,
        sigma_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract low and high frequencies with adaptive sigma.

        Since per-pixel sigma is expensive, we use discrete sigma levels
        and blend them based on the sigma map.

        Args:
            image: RGB image (uint8)
            sigma_map: Per-pixel sigma values

        Returns:
            Tuple of (low_freq, high_freq) as float32 arrays
        """
        img_float = image.astype(np.float32)
        h, w = image.shape[:2]

        # Discrete sigma levels for efficiency
        base_sigma = self.config.freq_base_sigma
        edge_factor = self.config.freq_edge_sigma_factor

        sigma_min = base_sigma * edge_factor
        sigma_max = base_sigma

        # Use 3 levels for balance between quality and speed
        sigmas = [sigma_min, (sigma_min + sigma_max) / 2, sigma_max]

        # Compute blurred versions at each sigma
        blurred_versions = []
        for sigma in sigmas:
            ksize = int(np.ceil(sigma * 6)) | 1  # Make odd
            blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
            blurred_versions.append(blurred)

        # Compute weights for each sigma level based on sigma_map
        # Normalize sigma_map to [0, 1] range corresponding to [sigma_min, sigma_max]
        sigma_normalized = (sigma_map - sigma_min) / (sigma_max - sigma_min + 1e-8)
        sigma_normalized = np.clip(sigma_normalized, 0, 1)

        # Interpolate between the 3 levels
        # Level 0: sigma_normalized < 0.5 (blend between 0 and 1)
        # Level 1: sigma_normalized >= 0.5 (blend between 1 and 2)
        low_freq = np.zeros_like(img_float)

        for c in range(3):  # RGB channels
            # First half: blend level 0 and 1
            mask_low = sigma_normalized < 0.5
            weight_1 = sigma_normalized * 2  # 0 to 1 as sigma_normalized goes 0 to 0.5

            # Second half: blend level 1 and 2
            mask_high = sigma_normalized >= 0.5
            weight_2 = (sigma_normalized - 0.5) * 2  # 0 to 1 as sigma_normalized goes 0.5 to 1

            # Apply blending
            result = np.zeros((h, w), dtype=np.float32)

            # Low sigma region
            result[mask_low] = (
                blurred_versions[0][:, :, c][mask_low] * (1 - weight_1[mask_low]) +
                blurred_versions[1][:, :, c][mask_low] * weight_1[mask_low]
            )

            # High sigma region
            result[mask_high] = (
                blurred_versions[1][:, :, c][mask_high] * (1 - weight_2[mask_high]) +
                blurred_versions[2][:, :, c][mask_high] * weight_2[mask_high]
            )

            low_freq[:, :, c] = result

        # High frequency = original - low frequency
        high_freq = img_float - low_freq

        return low_freq, high_freq

    def compute_restoration_strength(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """Compute per-pixel restoration strength based on saturation.

        Low saturation = likely faded = stronger restoration
        High saturation = good condition = weaker restoration

        Args:
            image: RGB image (uint8)

        Returns:
            Strength map (0-1)
        """
        if not self.config.freq_saturation_strength:
            return np.ones(image.shape[:2], dtype=np.float32)

        # Convert to HSV and extract saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0

        # Invert: low saturation = high strength
        weight = self.config.freq_saturation_weight
        strength = 1.0 - saturation * weight

        return strength

    def filter_texture_by_edge(
        self,
        high_freq: np.ndarray,
        edge_proximity: np.ndarray
    ) -> np.ndarray:
        """Filter high-frequency based on edge proximity.

        Near edges: high-freq is meaningful (brushwork, lines)
        Far from edges: high-freq might be noise

        Args:
            high_freq: High frequency component (float32)
            edge_proximity: Edge proximity map (0-1)

        Returns:
            Filtered high frequency
        """
        reduction = self.config.freq_texture_noise_reduction

        if reduction <= 0:
            return high_freq

        # Keep full texture near edges, reduce away from edges
        # texture_weight = proximity + (1 - proximity) * (1 - reduction)
        # Simplify: texture_weight = 1 - reduction * (1 - proximity)
        texture_weight = 1.0 - reduction * (1.0 - edge_proximity)

        # Expand to 3 channels
        texture_weight_3d = texture_weight[:, :, np.newaxis]

        return high_freq * texture_weight_3d

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """Frequency-aware restoration with color edge inference.

        Pipeline:
        1. Detect edges (Canny + Color inference)
        2. Compute adaptive sigma map (small near edges)
        3. Separate low/high frequencies
        4. Restore low frequency (color/tone)
        5. Filter high frequency (reduce noise far from edges)
        6. Combine: restored_low + filtered_high
        7. Apply edge boosting (optional)
        """
        import time
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy
        img_np = self._tensor_to_numpy_rgb(image[0])
        h, w = img_np.shape[:2]

        # Step 1: Enhanced edge detection (Canny + Color inference)
        edges, edge_debug_maps = self.detect_edges_enhanced(img_np)
        edge_proximity = self.compute_edge_proximity(edges)

        # Step 2: Compute adaptive sigma map
        sigma_map = self.compute_sigma_map(edge_proximity, h, w)

        # Step 3: Separate frequencies with adaptive sigma
        low_freq, high_freq = self.extract_frequencies_adaptive(img_np, sigma_map)

        # Step 4: Restore low frequency using FadingNoiseRestorer
        # Convert low_freq to tensor for the restorer
        low_freq_uint8 = np.clip(low_freq, 0, 255).astype(np.uint8)
        low_freq_tensor = self._numpy_to_tensor(low_freq_uint8).unsqueeze(0)

        restoration_output = self.fading_restorer.forward(low_freq_tensor)
        restored_low_tensor = restoration_output.result

        # Convert back to numpy
        restored_low = self._tensor_to_numpy_rgb(restored_low_tensor[0]).astype(np.float32)

        # Apply saturation-based strength
        strength_map = self.compute_restoration_strength(img_np)
        strength_3d = strength_map[:, :, np.newaxis]

        # Blend original and restored low frequency based on strength
        final_low = low_freq * (1 - strength_3d) + restored_low * strength_3d

        # Step 5: Filter high frequency (reduce noise in non-edge areas)
        filtered_high = self.filter_texture_by_edge(high_freq, edge_proximity)

        # Step 6: Inject inferred edges into high frequency (edge enhancement)
        if self.config.edge_boost_strength > 0:
            filtered_high = self.inject_edge_to_highfreq(filtered_high, edges)

        # Step 7: Combine
        result = final_low + filtered_high
        result = np.clip(result, 0, 255).astype(np.uint8)

        elapsed = time.time() - start

        # Build intermediates
        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_to_tensor(img_np),
                'edges': self._numpy_to_tensor(np.stack([edges]*3, axis=-1)),
                'edge_proximity': self._numpy_to_tensor(
                    np.stack([(edge_proximity * 255).astype(np.uint8)]*3, axis=-1)
                ),
                'sigma_map': self._numpy_to_tensor(
                    np.stack([
                        ((sigma_map / sigma_map.max()) * 255).astype(np.uint8)
                    ]*3, axis=-1)
                ),
                'low_freq': self._numpy_to_tensor(np.clip(low_freq, 0, 255).astype(np.uint8)),
                'high_freq': self._numpy_to_tensor(
                    np.clip(high_freq + 128, 0, 255).astype(np.uint8)
                ),
                'restored_low': self._numpy_to_tensor(np.clip(restored_low, 0, 255).astype(np.uint8)),
                'strength_map': self._numpy_to_tensor(
                    np.stack([(strength_map * 255).astype(np.uint8)]*3, axis=-1)
                ),
            }

            # Add edge debug maps
            for key, arr in edge_debug_maps.items():
                if arr is not None:
                    intermediates[f'edge_{key}'] = self._numpy_to_tensor(
                        np.stack([arr]*3, axis=-1) if arr.ndim == 2 else arr
                    )

        result_tensor = self._numpy_to_tensor(result).unsqueeze(0)

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediates,
            metadata={
                'method': 'frequency_aware_v6',
                'processing_time': elapsed,
                'base_sigma': self.config.freq_base_sigma,
                'edge_sigma_factor': self.config.freq_edge_sigma_factor,
                'saturation_strength': self.config.freq_saturation_strength,
                'texture_noise_reduction': self.config.freq_texture_noise_reduction,
                'use_color_edge_inference': self.config.use_color_edge_inference,
                'color_edge_weight': self.config.color_edge_weight,
                'edge_boost_strength': self.config.edge_boost_strength,
                'fading_noise_metadata': restoration_output.metadata,
            }
        )


__all__ = ["FrequencyAwareRestorer"]
