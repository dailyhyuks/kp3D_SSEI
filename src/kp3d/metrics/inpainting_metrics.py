"""V22 PatchMatch-Guided Inpainting 특화 평가 지표.

GT 없이 inpainting 품질을 측정하는 3가지 핵심 지표:
- Color Outlier Rate (COR): 색상 이상치 비율
- Boundary Smoothness (BS): 경계 부드러움
- Texture Coherence (TC): 텍스처 일관성
"""

from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def color_outlier_rate(
    inpainted: np.ndarray,
    mask: np.ndarray,
    visible_region: Optional[np.ndarray] = None,
    threshold_sigma: float = 3.0,
    percentile_margin: float = 20.0,
    channel_imbalance_threshold: float = 50.0,
) -> float:
    """Color Outlier Rate (COR) 계산.

    Inpainted 영역에서 visible 영역 색상 분포 대비 이상치 비율 측정.
    V22의 multi-stage color filtering 효과를 정량화.

    Args:
        inpainted: Inpainting 결과 이미지 (H, W, 3), RGB, uint8
        mask: Inpainted 영역 마스크 (H, W), 255=inpainted 영역
        visible_region: 참조할 visible 영역 이미지. None이면 mask 외부 사용
        threshold_sigma: 이상치 판정 시그마 배수 (기본 3σ)
        percentile_margin: percentile 범위 마진
        channel_imbalance_threshold: 채널 불균형 임계값

    Returns:
        outlier_rate: 이상치 픽셀 비율 (0~1, 낮을수록 좋음)
    """
    # 입력 검증
    if inpainted.ndim != 3 or inpainted.shape[2] != 3:
        raise ValueError("inpainted must be RGB image (H, W, 3)")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D array")

    # 마스크 이진화
    mask_binary = (mask > 127).astype(np.uint8)

    # Visible 영역 추출
    if visible_region is None:
        visible_mask = 1 - mask_binary
        visible_pixels = inpainted[visible_mask > 0]
    else:
        visible_pixels = visible_region.reshape(-1, 3)

    if len(visible_pixels) == 0:
        return 0.0

    # Inpainted 영역 픽셀 추출
    inpainted_pixels = inpainted[mask_binary > 0]

    if len(inpainted_pixels) == 0:
        return 0.0

    # Visible 영역 통계
    mean_color = visible_pixels.mean(axis=0)
    std_color = visible_pixels.std(axis=0) + 1e-6  # 0 방지

    # Percentile 범위
    p10 = np.percentile(visible_pixels, 10, axis=0)
    p90 = np.percentile(visible_pixels, 90, axis=0)

    # 채널별 평균 차이 (채널 불균형 기준)
    channel_diffs = np.abs(visible_pixels - mean_color)
    avg_channel_diff = channel_diffs.mean(axis=0)

    # 이상치 판정
    outlier_count = 0

    for pixel in inpainted_pixels:
        is_outlier = False

        # Check 1: 시그마 기준 (평균에서 너무 멀리 떨어진 경우)
        z_scores = np.abs(pixel - mean_color) / std_color
        if np.any(z_scores > threshold_sigma):
            is_outlier = True

        # Check 2: Percentile 범위 벗어남
        if not is_outlier:
            if np.any(pixel < p10 - percentile_margin) or np.any(pixel > p90 + percentile_margin):
                is_outlier = True

        # Check 3: 채널 불균형 (특정 채널만 튀는 경우)
        if not is_outlier:
            pixel_channel_diff = np.abs(pixel - mean_color)
            if np.any(pixel_channel_diff > avg_channel_diff + channel_imbalance_threshold):
                is_outlier = True

        if is_outlier:
            outlier_count += 1

    return outlier_count / len(inpainted_pixels)


def boundary_smoothness(
    inpainted: np.ndarray,
    mask: np.ndarray,
    boundary_width: int = 3,
) -> float:
    """Boundary Smoothness (BS) 계산.

    마스크 경계에서의 gradient 연속성 측정.
    경계가 자연스러울수록 높은 값.

    Args:
        inpainted: Inpainting 결과 이미지 (H, W, 3), RGB, uint8
        mask: Inpainted 영역 마스크 (H, W), 255=inpainted 영역
        boundary_width: 경계 영역 너비 (픽셀)

    Returns:
        smoothness: 경계 부드러움 (0~1, 높을수록 좋음)
    """
    # 입력 검증
    if inpainted.ndim != 3:
        raise ValueError("inpainted must be 3D array")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D array")

    # 마스크 이진화
    mask_binary = (mask > 127).astype(np.uint8)

    # 경계 추출: dilate(mask) - mask
    kernel = np.ones((boundary_width * 2 + 1, boundary_width * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    boundary = dilated - mask_binary

    # 내부 경계도 추출: mask - erode(mask)
    eroded = cv2.erode(mask_binary, kernel, iterations=1)
    inner_boundary = mask_binary - eroded

    # 전체 경계 영역
    full_boundary = np.maximum(boundary, inner_boundary)

    if full_boundary.sum() == 0:
        return 1.0

    # Grayscale 변환
    gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Gradient 계산 (Sobel)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 경계에서의 gradient magnitude
    boundary_grads = grad_mag[full_boundary > 0]

    if len(boundary_grads) == 0:
        return 1.0

    # 내부 영역(마스크 안쪽)에서의 gradient magnitude
    interior_mask = cv2.erode(mask_binary, kernel, iterations=2)
    interior_grads = grad_mag[interior_mask > 0]

    if len(interior_grads) == 0:
        # 내부 영역이 없으면 전체 이미지 평균과 비교
        interior_mean = grad_mag.mean()
    else:
        interior_mean = interior_grads.mean()

    boundary_mean = boundary_grads.mean()

    # 경계 gradient가 내부와 비슷하면 자연스러움
    # 경계 gradient가 높으면 부자연스러움 (경계선이 보임)
    if interior_mean < 1e-6:
        interior_mean = 1e-6

    # 정규화: 경계가 내부보다 gradient가 낮거나 비슷하면 1에 가까움
    ratio = boundary_mean / interior_mean

    # ratio가 1 이하면 좋음 (경계가 눈에 안 띔)
    # ratio가 높으면 나쁨 (경계가 눈에 띔)
    smoothness = 1.0 / (1.0 + max(0, ratio - 1.0))

    return float(np.clip(smoothness, 0, 1))


def texture_coherence(
    inpainted: np.ndarray,
    mask: np.ndarray,
    visible_region: Optional[np.ndarray] = None,
) -> float:
    """Texture Coherence (TC) 계산.

    GLCM (Gray-Level Co-occurrence Matrix) 기반 텍스처 일관성 측정.
    Inpainted 영역과 visible 영역의 텍스처 특성 비교.

    Args:
        inpainted: Inpainting 결과 이미지 (H, W, 3), RGB, uint8
        mask: Inpainted 영역 마스크 (H, W), 255=inpainted 영역
        visible_region: 참조할 visible 영역 이미지. None이면 mask 외부 사용

    Returns:
        coherence: 텍스처 일관성 (0~1, 높을수록 좋음)
    """
    if not HAS_SKIMAGE:
        # skimage 없으면 대체 방법 사용
        return _texture_coherence_fallback(inpainted, mask, visible_region)

    # 입력 검증
    if inpainted.ndim != 3:
        raise ValueError("inpainted must be 3D array")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D array")

    # Grayscale 변환
    gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)

    # 마스크 이진화
    mask_binary = (mask > 127).astype(np.uint8)

    # Inpainted 영역 픽셀
    inpainted_pixels = gray[mask_binary > 0]

    # Visible 영역 픽셀
    if visible_region is None:
        visible_mask = 1 - mask_binary
        visible_pixels = gray[visible_mask > 0]
    else:
        if visible_region.ndim == 3:
            visible_gray = cv2.cvtColor(visible_region, cv2.COLOR_RGB2GRAY)
        else:
            visible_gray = visible_region
        visible_pixels = visible_gray.flatten()

    if len(inpainted_pixels) < 10 or len(visible_pixels) < 10:
        return 1.0  # 픽셀이 너무 적으면 비교 불가

    # GLCM 계산을 위해 2D 패치로 변환
    # 픽셀 배열을 정사각형에 가깝게 reshape
    inp_size = int(np.sqrt(len(inpainted_pixels)))
    vis_size = int(np.sqrt(len(visible_pixels)))

    if inp_size < 4 or vis_size < 4:
        return 1.0

    # 정사각형 패치 생성
    inp_patch = inpainted_pixels[:inp_size * inp_size].reshape(inp_size, inp_size)
    vis_patch = visible_pixels[:vis_size * vis_size].reshape(vis_size, vis_size)

    # GLCM 계산
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    try:
        glcm_inp = graycomatrix(inp_patch, distances, angles, 256, symmetric=True, normed=True)
        glcm_vis = graycomatrix(vis_patch, distances, angles, 256, symmetric=True, normed=True)
    except Exception:
        return _texture_coherence_fallback(inpainted, mask, visible_region)

    # 텍스처 속성 비교
    props = ['contrast', 'homogeneity', 'energy', 'correlation']
    coherence_scores = []

    for prop in props:
        try:
            val_inp = graycoprops(glcm_inp, prop).mean()
            val_vis = graycoprops(glcm_vis, prop).mean()

            # 유사도 계산
            max_val = max(abs(val_inp), abs(val_vis), 1e-6)
            similarity = 1 - abs(val_inp - val_vis) / max_val
            coherence_scores.append(max(0, similarity))
        except Exception:
            continue

    if len(coherence_scores) == 0:
        return 0.5

    return float(np.mean(coherence_scores))


def _texture_coherence_fallback(
    inpainted: np.ndarray,
    mask: np.ndarray,
    visible_region: Optional[np.ndarray] = None,
) -> float:
    """GLCM 없이 텍스처 일관성 계산 (대체 방법).

    High-frequency 에너지 비교 방식.
    """
    # Grayscale 변환
    if inpainted.ndim == 3:
        gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = inpainted.astype(np.float32)

    # 마스크 이진화
    mask_binary = (mask > 127).astype(np.uint8)

    # High-pass filter (Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Inpainted 영역의 high-freq 에너지
    inp_hf = np.abs(laplacian[mask_binary > 0])

    # Visible 영역의 high-freq 에너지
    visible_mask = 1 - mask_binary
    vis_hf = np.abs(laplacian[visible_mask > 0])

    if len(inp_hf) == 0 or len(vis_hf) == 0:
        return 1.0

    inp_energy = inp_hf.mean()
    vis_energy = vis_hf.mean()

    # 에너지 비율로 일관성 계산
    max_energy = max(inp_energy, vis_energy, 1e-6)
    similarity = 1 - abs(inp_energy - vis_energy) / max_energy

    return float(np.clip(similarity, 0, 1))


class InpaintingMetrics:
    """V22 Inpainting 평가 메트릭 통합 클래스."""

    def __init__(
        self,
        cor_threshold_sigma: float = 3.0,
        bs_boundary_width: int = 3,
    ):
        """초기화.

        Args:
            cor_threshold_sigma: COR 계산 시 시그마 배수
            bs_boundary_width: BS 계산 시 경계 너비
        """
        self.cor_threshold_sigma = cor_threshold_sigma
        self.bs_boundary_width = bs_boundary_width

    def compute_all(
        self,
        inpainted: np.ndarray,
        mask: np.ndarray,
        visible_region: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """모든 V22 특화 메트릭 계산.

        Args:
            inpainted: Inpainting 결과 이미지 (H, W, 3), RGB, uint8
            mask: Inpainted 영역 마스크 (H, W), 255=inpainted 영역
            visible_region: 참조할 visible 영역 이미지 (선택)

        Returns:
            메트릭 딕셔너리 {'cor': float, 'bs': float, 'tc': float}
        """
        return {
            'cor': color_outlier_rate(
                inpainted, mask, visible_region,
                threshold_sigma=self.cor_threshold_sigma
            ),
            'bs': boundary_smoothness(
                inpainted, mask,
                boundary_width=self.bs_boundary_width
            ),
            'tc': texture_coherence(
                inpainted, mask, visible_region
            ),
        }

    def compute_cor(
        self,
        inpainted: np.ndarray,
        mask: np.ndarray,
        visible_region: Optional[np.ndarray] = None,
    ) -> float:
        """Color Outlier Rate만 계산."""
        return color_outlier_rate(
            inpainted, mask, visible_region,
            threshold_sigma=self.cor_threshold_sigma
        )

    def compute_bs(
        self,
        inpainted: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        """Boundary Smoothness만 계산."""
        return boundary_smoothness(
            inpainted, mask,
            boundary_width=self.bs_boundary_width
        )

    def compute_tc(
        self,
        inpainted: np.ndarray,
        mask: np.ndarray,
        visible_region: Optional[np.ndarray] = None,
    ) -> float:
        """Texture Coherence만 계산."""
        return texture_coherence(inpainted, mask, visible_region)


__all__ = [
    'color_outlier_rate',
    'boundary_smoothness',
    'texture_coherence',
    'InpaintingMetrics',
]
