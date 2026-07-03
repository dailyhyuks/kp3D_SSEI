"""Enhanced grid pattern removal (v7).

Extends GridPatternRestorer with:
- Hough transform-based grid angle detection for rotated grids
- Radial profile analysis for precise frequency/angle detection
- Rotated notch filter for non-axis-aligned grid patterns
"""

import cv2
import numpy as np
import time
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple

from kp3d.core.base import ModuleOutput
from kp3d.modules.restoration.base import RestorationConfig
from kp3d.modules.restoration.grid_pattern import GridPatternRestorer


class EnhancedGridPatternRestorer(GridPatternRestorer):
    """v7 격자 패턴 제거 복원기 - 회전 보정 FFT

    GridPatternRestorer를 상속하여 회전된 격자 패턴 지원을 추가합니다.
    fft_auto_angle=False이면 부모 클래스와 동일하게 동작합니다.
    """

    def __init__(
        self,
        config: Optional[RestorationConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, **kwargs)

    @property
    def name(self) -> str:
        return "enhanced_grid_pattern"

    def detect_grid_angles(
        self,
        image: np.ndarray,
        angle_tolerance: float = 5.0,
    ) -> List[float]:
        """Hough 변환 기반 격자 각도 검출

        직물/캔버스의 격자 방향이 정확히 0°/90°가 아닌 경우
        실제 격자 각도를 자동 검출합니다.

        Args:
            image: 입력 이미지 (grayscale 또는 color)
            angle_tolerance: 검출된 각도의 허용 범위 (도)

        Returns:
            검출된 격자 각도 리스트 (도)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Edge detection for Hough
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return [0.0, 90.0]  # Default

        # Collect angles
        angles = []
        for line in lines:
            theta = line[0][1]
            angle_deg = np.degrees(theta)
            angles.append(angle_deg)

        angles = np.array(angles)

        # Cluster angles to find dominant directions
        # Grid should have ~2 perpendicular directions
        dominant = self._cluster_angles(angles, angle_tolerance)

        if len(dominant) == 0:
            return [0.0, 90.0]

        # Ensure we have perpendicular pairs
        result = []
        for angle in dominant[:2]:
            result.append(angle)
            # Add perpendicular
            perp = (angle + 90.0) % 180.0
            if not any(abs(a - perp) < angle_tolerance for a in result):
                result.append(perp)

        return sorted(set(result))[:4]

    def _cluster_angles(
        self,
        angles: np.ndarray,
        tolerance: float,
    ) -> List[float]:
        """Cluster angles by proximity and return dominant ones.

        Args:
            angles: Array of detected angles in degrees.
            tolerance: Clustering tolerance in degrees.

        Returns:
            List of dominant angles.
        """
        if len(angles) == 0:
            return []

        # Wrap angles to [0, 180) for line symmetry
        angles = angles % 180.0

        # Sort
        sorted_angles = np.sort(angles)

        # Simple clustering
        clusters = []
        current_cluster = [sorted_angles[0]]

        for angle in sorted_angles[1:]:
            if angle - current_cluster[-1] < tolerance:
                current_cluster.append(angle)
            else:
                clusters.append(current_cluster)
                current_cluster = [angle]
        clusters.append(current_cluster)

        # Check wrap-around (0° and 180° are the same for lines)
        if len(clusters) > 1:
            if (180.0 - clusters[-1][-1] + clusters[0][0]) < tolerance:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()

        # Sort by cluster size and return mean angles
        clusters.sort(key=len, reverse=True)

        dominant = []
        for cluster in clusters[:4]:  # Top 4 directions
            if len(cluster) >= max(3, len(angles) * 0.05):  # Min 5% of lines
                mean_angle = np.mean(cluster)
                dominant.append(float(mean_angle))

        return dominant

    def detect_grid_frequencies_radial(
        self,
        image: np.ndarray,
        angles: List[float] = None,
        num_peaks: int = 5,
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Radial profile 기반 정밀 주파수 검출 (v7)

        FFT 스펙트럼에서 각도별 radial profile을 분석하여
        격자 주파수와 각도를 정밀하게 검출합니다.

        Args:
            image: 입력 이미지
            angles: 검색할 각도 (None이면 전체 탐색)
            num_peaks: 검출할 피크 수

        Returns:
            Tuple of (frequencies, freq_angle_pairs)
            freq_angle_pairs: list of (frequency, angle) tuples
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # Compute FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        # Determine angles to analyze
        if angles is None:
            angles = np.arange(0, 180, 1).tolist()

        threshold = self.config.fft_radial_threshold

        # Radial profile for each angle
        freq_angle_pairs = []
        all_freqs = []

        max_radius = min(cy, cx) - 5

        for angle in angles:
            rad = np.deg2rad(angle)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            # Sample along the line at this angle
            profile = np.zeros(max_radius)
            for r in range(5, max_radius):  # Skip DC
                x_sample = int(cx + r * cos_a)
                y_sample = int(cy + r * sin_a)
                if 0 <= y_sample < h and 0 <= x_sample < w:
                    profile[r] = magnitude[y_sample, x_sample]

            # Also sample opposite direction
            for r in range(5, max_radius):
                x_sample = int(cx - r * cos_a)
                y_sample = int(cy - r * sin_a)
                if 0 <= y_sample < h and 0 <= x_sample < w:
                    profile[r] = max(profile[r], magnitude[y_sample, x_sample])

            # Find peaks in profile
            if profile.max() == 0:
                continue

            profile_norm = profile / (profile.max() + 1e-8)
            threshold_val = threshold

            for r in range(7, max_radius - 2):
                if (profile_norm[r] > threshold_val
                        and profile_norm[r] > profile_norm[r - 1]
                        and profile_norm[r] > profile_norm[r + 1]
                        and profile_norm[r] > profile_norm[r - 2]
                        and profile_norm[r] > profile_norm[r + 2]):
                    freq_angle_pairs.append((float(r), float(angle)))
                    all_freqs.append(float(r))

        # Deduplicate frequencies
        unique_freqs = sorted(set(all_freqs))[:num_peaks]

        return unique_freqs, freq_angle_pairs

    def rotated_notch_filter(
        self,
        image: np.ndarray,
        angles: List[float],
        line_width: int = 7,
        dc_exclusion: int = 15,
        target_freqs: List[float] = None,
        freq_band_width: int = 3,
    ) -> np.ndarray:
        """회전 보정 notch 필터 (v7)

        임의 각도의 격자 패턴을 제거하는 FFT 필터.
        기존 fft_directional_filter를 확장하여 Hough 검출 각도를 사용.

        Args:
            image: 입력 이미지
            angles: 필터링할 각도 리스트 (도)
            line_width: Notch 필터 폭
            dc_exclusion: DC 보호 반경
            target_freqs: 대상 주파수 리스트
            freq_band_width: 주파수 대역폭

        Returns:
            필터링된 이미지
        """
        # Delegate to parent's fft_directional_filter with detected angles
        return self.fft_directional_filter(
            image,
            line_width=line_width,
            dc_exclusion=dc_exclusion,
            angles=angles,
            target_freqs=target_freqs,
            freq_band_width=freq_band_width,
        )

    def fft_directional_filter(
        self,
        image: np.ndarray,
        line_width: int = 5,
        dc_exclusion: int = 15,
        angles: list = None,
        target_freqs: list = None,
        freq_band_width: int = 3,
    ) -> np.ndarray:
        """방향성 FFT 필터 - 자동 각도 검출 지원 (v7 오버라이드)

        fft_auto_angle=True인 경우 Hough transform으로 실제 격자 각도를
        검출하여 필터링합니다.
        """
        if not self.config.fft_auto_angle or angles is not None:
            return super().fft_directional_filter(
                image, line_width, dc_exclusion, angles,
                target_freqs, freq_band_width
            )

        # v7: Auto-detect grid angles
        detected_angles = self.detect_grid_angles(
            image, self.config.fft_angle_tolerance
        )

        return super().fft_directional_filter(
            image, line_width, dc_exclusion, detected_angles,
            target_freqs, freq_band_width
        )

    def detect_grid_frequencies(
        self,
        image: np.ndarray,
        num_peaks: int = 5,
    ) -> Tuple[list, list]:
        """격자 주파수 검출 - radial profile 지원 (v7 오버라이드)

        fft_auto_angle=True인 경우 radial profile 기반으로
        더 정밀한 주파수 검출을 수행합니다.
        """
        if not self.config.fft_auto_angle:
            return super().detect_grid_frequencies(image, num_peaks)

        # v7: Use radial profile analysis
        detected_angles = self.detect_grid_angles(
            image, self.config.fft_angle_tolerance
        )
        freqs, freq_angle_pairs = self.detect_grid_frequencies_radial(
            image, angles=detected_angles, num_peaks=num_peaks
        )

        # Convert to h_freqs, v_freqs format for compatibility
        h_freqs = []
        v_freqs = []
        for freq, angle in freq_angle_pairs:
            int_freq = int(freq)
            # Near-horizontal lines -> vertical frequency
            if angle < 45 or angle > 135:
                v_freqs.append(int_freq)
            else:
                h_freqs.append(int_freq)

        h_freqs = sorted(set(h_freqs))[:num_peaks]
        v_freqs = sorted(set(v_freqs))[:num_peaks]

        return h_freqs, v_freqs

    def forward(self, image: Tensor, **kwargs: Any) -> ModuleOutput:
        """v7 격자 패턴 제거

        fft_auto_angle가 비활성화되면 부모 클래스와 동일하게 동작합니다.
        """
        start = time.time()

        if image.dim() == 3:
            image = image.unsqueeze(0)

        img_np = self._tensor_to_numpy_rgb(image[0])
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        method = kwargs.get('method', self.config.grid_method)

        # Run restoration (overridden methods handle v7 features)
        result_bgr, intermediates = self.restore_grid_pattern(img_bgr, method=method)

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        elapsed = time.time() - start

        # Texture reduction metric
        original_std = np.std(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32))
        restored_std = np.std(cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
        texture_reduction = (1 - restored_std / (original_std + 1e-8)) * 100

        # Intermediates
        intermediate_tensors = {}
        if self.config.store_intermediates:
            for key, arr in intermediates.items():
                if arr is not None:
                    if arr.ndim == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    intermediate_tensors[key] = self._numpy_to_tensor(arr)

        result_tensor = self._numpy_to_tensor(result_rgb).unsqueeze(0)

        metadata = {
            'method': f'enhanced_grid_pattern_{method}',
            'processing_time': elapsed,
            'texture_reduction_percent': texture_reduction,
            'grid_bilateral_iterations': self.config.grid_bilateral_iterations,
            'grid_fft_line_width': self.config.grid_fft_line_width,
            'grid_guided_radius': self.config.grid_guided_radius,
            'v7_auto_angle': self.config.fft_auto_angle,
            'v7_angle_tolerance': self.config.fft_angle_tolerance,
            'v7_radial_threshold': self.config.fft_radial_threshold,
        }

        # Add detected angles if auto-angle was used
        if self.config.fft_auto_angle:
            detected = self.detect_grid_angles(
                img_bgr, self.config.fft_angle_tolerance
            )
            metadata['detected_grid_angles'] = detected

        return ModuleOutput(
            result=result_tensor,
            intermediate=intermediate_tensors,
            metadata=metadata,
        )


__all__ = ["EnhancedGridPatternRestorer"]
