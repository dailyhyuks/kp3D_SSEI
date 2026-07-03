"""Fading noise restoration for traditional Korean paintings.

Removes pigment degradation spots while preserving actual content:
- Darker spots (beige) in white areas
- Lighter/desaturated spots in colored areas
"""

import cv2
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Dict, Any, Tuple, List
from functools import lru_cache

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import BaseRestoration, RestorationConfig


class FadingNoiseRestorer(BaseRestoration):
    """퇴색 노이즈 복원기

    한지 위 염료 퇴색으로 인한 점군 노이즈를 제거합니다.
    엣지 영역을 제외하고 이상치를 탐지하여 inpainting합니다.
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = True
        self._median_cache = {}  # Cache for median blur results

    @property
    def name(self) -> str:
        return "fading_noise"

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

    def detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """엣지 영역 탐지 (제외할 영역)"""
        # Canny edge detection
        edges = cv2.Canny(gray, self.config.edge_threshold, self.config.edge_threshold * 2)

        # Dilate to create edge exclusion zone
        if self.config.edge_dilation > 0:
            kernel = np.ones((self.config.edge_dilation, self.config.edge_dilation), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def detect_local_outliers(
        self,
        lab: np.ndarray,
        edge_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """지역 이상치 탐지 (엣지 영역 제외) - 레거시 단일 스케일 버전

        Args:
            lab: LAB 색공간 이미지
            edge_mask: 엣지 영역 마스크 (제외)

        Returns:
            outlier_mask: 이상치 마스크
            diff_map: 차이 맵 (시각화용)
        """
        L, A, B = cv2.split(lab.astype(np.float32))
        ws = self.config.window_size

        # Local median (robust to outliers)
        L_med = cv2.medianBlur(L.astype(np.uint8), ws).astype(np.float32)
        A_med = cv2.medianBlur(A.astype(np.uint8), ws).astype(np.float32)
        B_med = cv2.medianBlur(B.astype(np.uint8), ws).astype(np.float32)

        # Color difference from local median
        L_diff = np.abs(L - L_med)
        A_diff = np.abs(A - A_med)
        B_diff = np.abs(B - B_med)

        # Combined difference (L weighted more)
        color_diff = np.sqrt(L_diff**2 * 2 + A_diff**2 + B_diff**2)

        # Threshold for outliers
        threshold = self.config.outlier_threshold * 10  # Scale for LAB
        outlier_mask = (color_diff > threshold).astype(np.uint8) * 255

        # Exclude edge regions
        outlier_mask[edge_mask > 0] = 0

        return outlier_mask, color_diff

    def _compute_single_scale_diff(
        self,
        L: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """단일 스케일에서 색상 차이 계산 (최적화됨)

        Args:
            L, A, B: LAB 채널 (float32)
            window_size: 윈도우 크기 (홀수)

        Returns:
            color_diff: 색상 차이 맵
        """
        ws = window_size if window_size % 2 == 1 else window_size + 1

        # Convert to uint8 once for all channels
        L_uint8 = L.astype(np.uint8)
        A_uint8 = A.astype(np.uint8)
        B_uint8 = B.astype(np.uint8)

        # Local median (robust to outliers)
        L_med = cv2.medianBlur(L_uint8, ws).astype(np.float32)
        A_med = cv2.medianBlur(A_uint8, ws).astype(np.float32)
        B_med = cv2.medianBlur(B_uint8, ws).astype(np.float32)

        # Color difference from local median (vectorized)
        L_diff = np.abs(L - L_med)
        A_diff = np.abs(A - A_med)
        B_diff = np.abs(B - B_med)

        # Combined difference (L weighted more) - avoid intermediate arrays
        color_diff = np.sqrt(L_diff * L_diff * 2 + A_diff * A_diff + B_diff * B_diff)

        return color_diff

    def _compute_adaptive_threshold(
        self,
        L: np.ndarray,
        saturation: np.ndarray,
        base_threshold: float
    ) -> np.ndarray:
        """적응형 임계값 계산 - 최적화됨

        밝은 영역: 더 민감하게 (threshold 낮춤)
        어두운 영역: 덜 민감하게 (threshold 높임)
        채도가 낮은 곳: 퇴색 가능성 높음 (threshold 낮춤)

        Args:
            L: 밝기 채널 (0-255)
            saturation: 채도 (0-255)
            base_threshold: 기본 임계값

        Returns:
            adaptive_threshold: 픽셀별 적응형 임계값
        """
        # Normalize L to 0-1 (in-place division)
        L_norm = L * (1.0 / 255.0)

        # Brightness adjustment: bright areas get lower threshold
        # Linear interpolation between bright_factor and dark_factor
        brightness_factor = (
            self.config.bright_threshold_factor * (1.0 - L_norm) +
            self.config.dark_threshold_factor * L_norm
        )

        # Saturation adjustment: low saturation = potentially faded = lower threshold
        # Normalize saturation
        sat_norm = saturation * (1.0 / 255.0)
        # Low saturation → factor < 1 (more sensitive)
        saturation_factor = 1.0 - self.config.saturation_weight * (1.0 - sat_norm)

        # Combine factors (in-place multiplication)
        adaptive_threshold = base_threshold * brightness_factor
        adaptive_threshold *= saturation_factor

        return adaptive_threshold

    def detect_local_outliers_multiscale(
        self,
        lab: np.ndarray,
        hsv: np.ndarray,
        edge_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """다중 스케일 + 적응형 임계값 이상치 탐지 (최적화됨)

        Args:
            lab: LAB 색공간 이미지
            hsv: HSV 색공간 이미지 (채도 정보용)
            edge_mask: 엣지 영역 마스크 (제외)

        Returns:
            outlier_mask: 최종 이상치 마스크
            diff_map: 결합된 차이 맵 (시각화용)
            debug_info: 디버그 정보 (각 스케일 결과 등)
        """
        L, A, B = cv2.split(lab.astype(np.float32))
        saturation = hsv[:, :, 1].astype(np.float32)
        h, w = L.shape

        window_sizes = self.config.window_sizes
        base_threshold = self.config.outlier_threshold * 10  # Scale for LAB

        # Compute adaptive threshold if enabled
        if self.config.adaptive_threshold:
            threshold_map = self._compute_adaptive_threshold(L, saturation, base_threshold)
        else:
            threshold_map = np.full((h, w), base_threshold, dtype=np.float32)

        # Pre-allocate arrays for better memory usage
        num_scales = len(window_sizes)
        scale_diffs = []
        scale_weights = np.zeros(num_scales, dtype=np.float32)

        # Compute weights first
        for i, ws in enumerate(window_sizes):
            scale_weights[i] = 1.0 / np.sqrt(ws)

        # Normalize weights
        scale_weights /= scale_weights.sum()

        # Multi-scale detection with optimized combination
        combination = self.config.scale_combination

        if combination == "or":
            # Union: any scale detects it (optimized)
            combined = np.zeros((h, w), dtype=np.float32)
            for ws in window_sizes:
                color_diff = self._compute_single_scale_diff(L, A, B, ws)
                scale_diffs.append(color_diff)
                scale_mask = (color_diff > threshold_map).astype(np.float32)
                np.maximum(combined, scale_mask, out=combined)
            outlier_mask = (combined > 0.5).astype(np.uint8) * 255

        elif combination == "and":
            # Intersection: all scales must detect it (optimized)
            combined = np.ones((h, w), dtype=np.float32)
            for ws in window_sizes:
                color_diff = self._compute_single_scale_diff(L, A, B, ws)
                scale_diffs.append(color_diff)
                scale_mask = (color_diff > threshold_map).astype(np.float32)
                np.minimum(combined, scale_mask, out=combined)
            outlier_mask = (combined > 0.5).astype(np.uint8) * 255

        else:  # "weighted" (default, most optimized)
            # Weighted combination with single-pass accumulation
            combined = np.zeros((h, w), dtype=np.float32)
            combined_diff = np.zeros((h, w), dtype=np.float32)

            for ws, weight in zip(window_sizes, scale_weights):
                color_diff = self._compute_single_scale_diff(L, A, B, ws)
                scale_diffs.append(color_diff)

                # Combine in-place to reduce memory
                scale_mask = (color_diff > threshold_map).astype(np.float32)
                combined += scale_mask * weight
                combined_diff += color_diff * weight

            outlier_mask = (combined > 0.5).astype(np.uint8) * 255

            # Exclude edge regions
            outlier_mask[edge_mask > 0] = 0

            debug_info = {
                'scale_diffs': scale_diffs if self.config.store_intermediates else [],
                'scale_weights': scale_weights.tolist(),
                'threshold_map': threshold_map if self.config.store_intermediates else None,
                'combined_score': combined if self.config.store_intermediates else None,
            }

            return outlier_mask, combined_diff, debug_info

        # For "or" and "and" modes, compute combined_diff separately
        combined_diff = np.zeros((h, w), dtype=np.float32)
        for diff, weight in zip(scale_diffs, scale_weights):
            combined_diff += diff * weight

        # Exclude edge regions
        outlier_mask[edge_mask > 0] = 0

        debug_info = {
            'scale_diffs': scale_diffs if self.config.store_intermediates else [],
            'scale_weights': scale_weights.tolist(),
            'threshold_map': threshold_map if self.config.store_intermediates else None,
            'combined_score': combined if self.config.store_intermediates else None,
        }

        return outlier_mask, combined_diff, debug_info

    def filter_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        """작은 점군만 필터링 (퇴색 노이즈 특성) - 레거시 버전"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        filtered = np.zeros_like(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Keep only small isolated blobs (fading spots)
            if self.config.min_blob_area <= area <= self.config.max_blob_area:
                filtered[labels == i] = 255

        return filtered

    def _compute_circularity(self, contour: np.ndarray) -> float:
        """원형도 계산 (4π * area / perimeter²)

        원: 1.0, 선: ~0, 사각형: ~0.785
        """
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6:
            return 0.0
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(circularity, 1.0)  # Clamp to 1.0

    def _compute_density_map(self, mask: np.ndarray) -> np.ndarray:
        """이상치 밀도 맵 계산

        각 픽셀 주변의 이상치 비율을 계산합니다.
        고립된 점군만 선택하기 위해 사용됩니다.
        """
        ks = self.config.density_kernel_size
        kernel = np.ones((ks, ks), dtype=np.float32) / (ks * ks)

        # Binary mask to float
        mask_float = (mask > 0).astype(np.float32)

        # Convolve to get local density
        density = cv2.filter2D(mask_float, -1, kernel)

        return density

    def filter_blobs_advanced(self, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """고급 블롭 필터링 (크기 + 원형도 + 밀도) - 최적화됨

        Args:
            mask: 이상치 마스크

        Returns:
            filtered: 필터링된 마스크
            filter_stats: 필터링 통계
        """
        # Single connected components call
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        filtered = np.zeros_like(mask)

        # Compute density map for density filtering
        density_map = None
        if self.config.density_filter:
            density_map = self._compute_density_map(mask)

        # Stats for debugging
        total_blobs = num_labels - 1
        size_filtered = 0
        circularity_filtered = 0
        density_filtered = 0
        kept = 0

        # Optimized contour handling - only if circularity check is needed
        contour_map = {}
        if self.config.circularity_threshold > 0:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Build contour map more efficiently
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]

                    # Use spatial indexing for faster lookup
                    # Find matching label - only check nearby centroids
                    for i in range(1, num_labels):
                        label_cx, label_cy = centroids[i]
                        if abs(cx - label_cx) < 2 and abs(cy - label_cy) < 2:
                            contour_map[i] = contour
                            break

        # Pre-compute density kernel area for reuse
        density_kernel_area = self.config.density_kernel_size ** 2 if self.config.density_filter else 1

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]

            # Size filter
            if not (self.config.min_blob_area <= area <= self.config.max_blob_area):
                size_filtered += 1
                continue

            # Circularity filter
            if self.config.circularity_threshold > 0 and i in contour_map:
                circularity = self._compute_circularity(contour_map[i])
                if circularity < self.config.circularity_threshold:
                    circularity_filtered += 1
                    continue

            # Density filter (isolated points only)
            if self.config.density_filter and density_map is not None:
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                # Check density at centroid
                if 0 <= cy < density_map.shape[0] and 0 <= cx < density_map.shape[1]:
                    local_density = density_map[cy, cx]
                    # Subtract self contribution (approximate)
                    self_contrib = area / density_kernel_area
                    neighbor_density = max(0, local_density - self_contrib)
                    if neighbor_density > self.config.density_threshold:
                        density_filtered += 1
                        continue

            # Passed all filters
            filtered[labels == i] = 255
            kept += 1

        filter_stats = {
            'total_blobs': total_blobs,
            'size_filtered': size_filtered,
            'circularity_filtered': circularity_filtered,
            'density_filtered': density_filtered,
            'kept': kept,
        }

        return filtered, filter_stats

    def inpaint_noise(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """노이즈 영역 복원 (inpainting)"""
        if self.config.inpaint_method == "telea":
            method = cv2.INPAINT_TELEA
        else:
            method = cv2.INPAINT_NS

        # OpenCV inpaint expects BGR
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(bgr, mask, self.config.inpaint_radius, method)
        restored = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)

        return restored

    def _get_distance_weights(self, h: int, w: int, center_y: int, center_x: int, radius: int) -> np.ndarray:
        """거리 기반 가중치 계산

        Args:
            h, w: 이미지 크기
            center_y, center_x: 중심 좌표
            radius: 가중치 반경

        Returns:
            weights: 거리 기반 가중치 (가까울수록 높음)
        """
        y_coords = np.arange(max(0, center_y - radius), min(h, center_y + radius + 1))
        x_coords = np.arange(max(0, center_x - radius), min(w, center_x + radius + 1))

        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

        # Inverse distance weighting (avoid division by zero)
        weights = 1.0 / (dist + 1e-6)
        weights[dist > radius] = 0

        return weights, y_coords, x_coords

    def inpaint_color_aware(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """색상 인식 inpainting - 최적화됨

        마스크 영역 주변의 색상 분포를 분석하고
        거리 가중 평균으로 자연스러운 색상 전이를 생성합니다.
        LAB 공간에서 혼합하여 지각적으로 자연스럽게 처리합니다.

        Args:
            image: RGB 이미지 (uint8)
            mask: 복원할 영역 마스크

        Returns:
            restored: 색상 인식 inpainting이 적용된 이미지
        """
        h, w = image.shape[:2]
        radius = self.config.color_sample_radius
        blend_radius = self.config.blend_radius

        # Convert to LAB for perceptually uniform color mixing
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        restored_lab = lab.copy()

        # Dilate mask to get boundary region for blending
        if blend_radius > 0:
            kernel = np.ones((blend_radius * 2 + 1, blend_radius * 2 + 1), np.uint8)
            dilated_mask = cv2.dilate(mask, kernel, iterations=1)
            blend_region = (dilated_mask > 0) & (mask == 0)
        else:
            blend_region = np.zeros_like(mask, dtype=bool)

        # Process each connected component for efficiency
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for label_idx in range(1, num_labels):
            # Get bounding box with padding
            y_min = max(0, stats[label_idx, cv2.CC_STAT_TOP] - radius)
            y_max = min(h, stats[label_idx, cv2.CC_STAT_TOP] + stats[label_idx, cv2.CC_STAT_HEIGHT] + radius)
            x_min = max(0, stats[label_idx, cv2.CC_STAT_LEFT] - radius)
            x_max = min(w, stats[label_idx, cv2.CC_STAT_LEFT] + stats[label_idx, cv2.CC_STAT_WIDTH] + radius)

            # Extract local patch
            local_lab = lab[y_min:y_max, x_min:x_max]
            local_mask = mask[y_min:y_max, x_min:x_max]

            # Sample only non-mask pixels in local region
            non_mask = local_mask == 0
            if not np.any(non_mask):
                continue

            # Compute average color of surrounding non-mask pixels (distance weighted)
            local_h, local_w = local_lab.shape[:2]
            cy, cx = int(centroids[label_idx][1]) - y_min, int(centroids[label_idx][0]) - x_min

            # Create distance weights from centroid (vectorized)
            yy, xx = np.mgrid[0:local_h, 0:local_w]
            dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
            dist = np.sqrt(dist_sq)
            weights = np.where(non_mask, 1.0 / (dist + 1e-6), 0)

            # Weighted average in LAB space (vectorized)
            total_weight = weights.sum()
            if total_weight > 0:
                # Compute all channels at once
                avg_color = np.array([
                    (local_lab[:, :, 0] * weights).sum() / total_weight,
                    (local_lab[:, :, 1] * weights).sum() / total_weight,
                    (local_lab[:, :, 2] * weights).sum() / total_weight
                ])

                # Fill mask pixels with weighted average (vectorized)
                component_mask = (labels == label_idx)
                restored_lab[component_mask] = avg_color

        # Apply boundary blending using distance transform
        if blend_radius > 0 and np.any(blend_region):
            # Distance transform from mask boundary
            dist_from_mask = cv2.distanceTransform((~(mask > 0)).astype(np.uint8), cv2.DIST_L2, 5)

            # Blend factor: 0 at mask boundary, 1 at blend_radius distance
            blend_factor = np.clip(dist_from_mask / blend_radius, 0, 1)

            # Expand blend_factor for 3 channels
            blend_factor_3d = blend_factor[:, :, np.newaxis]

            # Apply blending only in mask region (vectorized)
            mask_3d = (mask > 0)[:, :, np.newaxis]
            restored_lab = np.where(
                mask_3d,
                restored_lab * (1 - blend_factor_3d) + lab * blend_factor_3d,
                lab
            )

        # Convert back to RGB
        restored_lab = np.clip(restored_lab, 0, 255).astype(np.uint8)
        restored = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2RGB)

        return restored

    def refine_colors(
        self,
        restored: np.ndarray,
        original: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """OpenCV inpainting 결과의 색상 보정

        OpenCV inpainting 후 주변 색상과의 일관성을 개선합니다.

        Args:
            restored: OpenCV inpainting 결과
            original: 원본 이미지
            mask: 복원 영역 마스크

        Returns:
            refined: 색상 보정된 이미지
        """
        h, w = restored.shape[:2]
        radius = self.config.color_sample_radius

        # Convert to LAB
        restored_lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB).astype(np.float32)
        original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Dilate mask to get surrounding region
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        surrounding = (dilated_mask > 0) & (mask == 0)

        if not np.any(surrounding):
            return restored

        # Compute color statistics in surrounding region
        surround_L = original_lab[surrounding, 0]
        surround_A = original_lab[surrounding, 1]
        surround_B = original_lab[surrounding, 2]

        target_mean_L = surround_L.mean()
        target_mean_A = surround_A.mean()
        target_mean_B = surround_B.mean()

        # Get statistics of restored region
        mask_bool = mask > 0
        restored_L = restored_lab[mask_bool, 0]
        restored_A = restored_lab[mask_bool, 1]
        restored_B = restored_lab[mask_bool, 2]

        if len(restored_L) == 0:
            return restored

        current_mean_L = restored_L.mean()
        current_mean_A = restored_A.mean()
        current_mean_B = restored_B.mean()

        # Shift restored colors to match surrounding
        shift_L = target_mean_L - current_mean_L
        shift_A = target_mean_A - current_mean_A
        shift_B = target_mean_B - current_mean_B

        # Apply shift only to mask region
        refined_lab = restored_lab.copy()
        refined_lab[mask_bool, 0] = np.clip(restored_lab[mask_bool, 0] + shift_L, 0, 255)
        refined_lab[mask_bool, 1] = np.clip(restored_lab[mask_bool, 1] + shift_A, 0, 255)
        refined_lab[mask_bool, 2] = np.clip(restored_lab[mask_bool, 2] + shift_B, 0, 255)

        # Convert back to RGB
        refined_lab = refined_lab.astype(np.uint8)
        refined = cv2.cvtColor(refined_lab, cv2.COLOR_LAB2RGB)

        return refined

    def extract_texture(self, image: np.ndarray) -> np.ndarray:
        """고주파 텍스처 성분 추출 - 최적화됨

        한지의 자연스러운 질감을 보존하기 위해 고주파 성분을 추출합니다.
        Gaussian blur로 저주파 추출 후 원본에서 빼서 텍스처를 얻습니다.

        Args:
            image: RGB 이미지 (uint8)

        Returns:
            texture: 고주파 텍스처 성분 (float32, centered at 0)
        """
        sigma = self.config.texture_blur_sigma

        # Convert to float for processing (in-place conversion where possible)
        img_float = image.astype(np.float32)

        # Extract low-frequency component with Gaussian blur
        # Kernel size should be at least 6*sigma to capture the Gaussian well
        ksize = int(np.ceil(sigma * 6)) | 1  # Make odd
        low_freq = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

        # High-frequency = Original - Low-frequency (in-place)
        np.subtract(img_float, low_freq, out=img_float)

        return img_float

    def apply_texture(
        self,
        restored: np.ndarray,
        texture: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """복원 영역에 텍스처 재적용 - 최적화됨

        복원된 영역에 원본 이미지에서 추출한 텍스처를 합성하여
        한지의 자연스러운 질감을 복원합니다.

        Args:
            restored: 복원된 이미지 (uint8)
            texture: 고주파 텍스처 (float32)
            mask: 복원 영역 마스크

        Returns:
            result: 텍스처가 적용된 이미지
        """
        strength = self.config.texture_strength
        blend_radius = self.config.blend_radius

        # Convert restored to float
        restored_float = restored.astype(np.float32)

        # Create blending mask for smooth transition at boundaries
        if blend_radius > 0:
            # Distance transform for smooth blending
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            # Normalize to 0-1, with full strength at blend_radius distance
            blend_factor = np.clip(dist / max(blend_radius, 1), 0, 1)
        else:
            blend_factor = (mask > 0).astype(np.float32)

        # Apply texture with strength and blending
        # texture_amount ranges from 0 (at boundary) to strength (inside mask)
        texture_amount = blend_factor[:, :, np.newaxis] * strength

        # Add texture to restored image (in-place)
        np.add(restored_float, texture * texture_amount, out=restored_float)

        # Clip to valid range
        np.clip(restored_float, 0, 255, out=restored_float)

        return restored_float.astype(np.uint8)

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """퇴색 노이즈 복원 - 최적화됨

        multi_scale=True (기본값): 다중 스케일 + 적응형 임계값 + 고급 필터링
        multi_scale=False: 레거시 단일 스케일 탐지
        fast_mode=True: 성능 최적화 모드 (일부 정확도 희생)
        """
        import time
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Convert to numpy
        img_np = self._tensor_to_numpy_rgb(image[0])

        # Fast mode: reduce window sizes
        if self.config.fast_mode and self.config.multi_scale:
            original_windows = self.config.window_sizes
            self.config.window_sizes = tuple(w for i, w in enumerate(original_windows) if i % 2 == 0)

        # Convert to LAB, HSV, and grayscale
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        # Only compute HSV if needed
        if self.config.multi_scale and self.config.adaptive_threshold:
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        else:
            hsv = None

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Step 1: Detect edges (to exclude from noise detection)
        edge_mask = self.detect_edges(gray)

        # Free gray array if not needed anymore
        del gray

        # Step 2: Detect local outliers (excluding edges)
        debug_info = {}
        filter_stats = {}

        if self.config.multi_scale:
            # New multi-scale + adaptive threshold detection
            outlier_mask, diff_map, debug_info = self.detect_local_outliers_multiscale(
                lab, hsv, edge_mask
            )
            # Step 3: Advanced filtering with circularity and density
            noise_mask, filter_stats = self.filter_blobs_advanced(outlier_mask)
        else:
            # Legacy single-scale detection
            outlier_mask, diff_map = self.detect_local_outliers(lab, edge_mask)
            # Step 3: Simple size-based filtering
            noise_mask = self.filter_small_blobs(outlier_mask)

        # Restore original window sizes if fast mode was used
        if self.config.fast_mode and self.config.multi_scale:
            self.config.window_sizes = original_windows

        # Step 4: Inpaint detected noise regions with mode selection
        if np.any(noise_mask > 0):
            if self.config.inpaint_mode == "color_aware":
                restored = self.inpaint_color_aware(img_np, noise_mask)
            elif self.config.inpaint_mode == "hybrid":
                # OpenCV first, then color refinement
                restored = self.inpaint_noise(img_np, noise_mask)
                restored = self.refine_colors(restored, img_np, noise_mask)
            else:  # "opencv" (default/legacy)
                restored = self.inpaint_noise(img_np, noise_mask)

            # Step 5: Restore texture if enabled
            if self.config.preserve_texture:
                texture = self.extract_texture(img_np)
                restored = self.apply_texture(restored, texture, noise_mask)
        else:
            restored = img_np.copy()

        elapsed = time.time() - start

        # Count detected noise spots (reuse if available from filter_stats)
        if 'total_blobs' in filter_stats:
            noise_count = filter_stats['kept']
        else:
            num_labels = cv2.connectedComponents(noise_mask, connectivity=8)[0]
            noise_count = num_labels - 1  # Exclude background

        # Store intermediates conditionally
        intermediates = {}
        if self.config.store_intermediates:
            intermediates = {
                'original': self._numpy_to_tensor(img_np),
                'edge_mask': self._numpy_to_tensor(np.stack([edge_mask]*3, axis=-1)),
                'outlier_mask': self._numpy_to_tensor(np.stack([outlier_mask]*3, axis=-1)),
                'noise_mask': self._numpy_to_tensor(np.stack([noise_mask]*3, axis=-1)),
                'diff_map': self._numpy_to_tensor(
                    np.stack([
                        (diff_map / (diff_map.max() + 1e-8) * 255).astype(np.uint8)
                    ]*3, axis=-1)
                ),
            }

            # Add adaptive threshold visualization if available
            if 'threshold_map' in debug_info and debug_info['threshold_map'] is not None:
                threshold_map = debug_info['threshold_map']
                threshold_vis = (threshold_map / (threshold_map.max() + 1e-8) * 255).astype(np.uint8)
                intermediates['threshold_map'] = self._numpy_to_tensor(
                    np.stack([threshold_vis]*3, axis=-1)
                )

        result = self._numpy_to_tensor(restored).unsqueeze(0)

        # Build metadata
        metadata = {
            'method': 'fading_noise',
            'processing_time': elapsed,
            'noise_spots_detected': noise_count,
            'multi_scale': self.config.multi_scale,
            'adaptive_threshold': self.config.adaptive_threshold,
            'inpaint_mode': self.config.inpaint_mode,
            'preserve_texture': self.config.preserve_texture,
        }

        if self.config.multi_scale:
            metadata.update({
                'window_sizes': list(self.config.window_sizes),
                'scale_combination': self.config.scale_combination,
                'circularity_threshold': self.config.circularity_threshold,
                'density_filter': self.config.density_filter,
            })
            if filter_stats:
                metadata['filter_stats'] = filter_stats
        else:
            metadata.update({
                'window_size': self.config.window_size,
                'outlier_threshold': self.config.outlier_threshold,
            })

        return ModuleOutput(
            result=result,
            intermediate=intermediates,
            metadata=metadata
        )
