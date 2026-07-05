"""Scale-space DoG 선 검출 + 스켈레톤/선폭 측정.

단일 (sigma, k) DoG 대신 폭 범위를 커버하는 다중 스케일의 max 응답 사용
(v2 설계 1.3: 단일 k=1.6 의존 제거, 다중 굵기 대응).
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label
from skimage.morphology import skeletonize


def _derive_scales(min_width: float, max_width: float) -> np.ndarray:
    """P-adapt: 폭 범위 [min, max]를 커버하는 sigma 목록을 유도.

    sigma = width / 2 (선 단면 가우시안 근사에서 유도).
    샘플 수 = 옥타브당 2개 (인접 스케일 비 sqrt(2) — scale-space 표준 샘플링).
    """
    lo, hi = min_width / 2.0, max_width / 2.0
    num = max(3, int(np.ceil(np.log2(hi / lo) * 2)) + 1)
    return np.geomspace(lo, hi, num=num)


def detect_lines(
    structure_gray: np.ndarray, min_width: float, max_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """어두운 선(먹선)의 scale-normalized DoG 응답과 이진 마스크를 반환.

    Args:
        structure_gray: 2D float 구조 이미지 (RGF 출력의 grayscale).
        min_width, max_width: 검출 대상 선폭 범위 (px).
    Returns:
        (response, mask): response는 float64 (양수=선), mask는 bool.
    """
    g = np.asarray(structure_gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("structure_gray must be a 2D array")
    if min_width <= 0 or max_width <= 0 or min_width > max_width:
        raise ValueError("width bounds must satisfy 0 < min_width <= max_width")
    response = np.zeros_like(g)
    for s in _derive_scales(min_width, max_width):
        g1 = cv2.GaussianBlur(g, (0, 0), s)
        g2 = cv2.GaussianBlur(g, (0, 0), s * np.sqrt(2.0))
        dog = (g2 - g1) * s  # 어두운 선 -> 양수, scale 정규화
        response = np.maximum(response, dog)
    pos = np.clip(response, 0, None)
    if pos.max() <= 0:
        return response, np.zeros_like(g, dtype=bool)
    resp_u8 = (pos / pos.max() * 255.0).astype(np.uint8)
    _, mask_u8 = cv2.threshold(resp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return response, mask_u8.astype(bool)


_MIN_SKELETON_FRACTION = 0.005  # 정규화 상수: 대각선의 0.5% 미만 성분은 잡음으로 간주


def measure_line_widths(line_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """선 마스크에서 스켈레톤과 선폭 지도를 계산.

    선폭 = 스켈레톤 위치의 distance transform × 2 (중심축-경계 거리의 2배).
    최소 성분 길이는 이미지 대각선 비율로 정규화 (P-adapt).

    Args:
        line_mask: 2D bool.
    Returns:
        (skeleton: bool, width_map: float32 — 스켈레톤 외 0).
    """
    mask = np.asarray(line_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("line_mask must be a 2D array")

    skeleton = skeletonize(mask)

    # 빈 스켈레톤 조기 반환: distance transform 낭비 제거
    if not skeleton.any():
        width_map = np.zeros(mask.shape, dtype=np.float32)
        return skeleton, width_map

    # 소형 성분 제거 (벡터화): 스켈레톤 픽셀 수를 길이 프록시로 사용
    diag = float(np.hypot(*mask.shape))
    min_len = diag * _MIN_SKELETON_FRACTION
    labels = sk_label(skeleton, connectivity=2)
    counts = np.bincount(labels.ravel())
    keep = counts >= min_len
    skeleton = keep[labels] & (labels > 0)

    # 빈 스켈레톤 조기 반환: distance transform 낭비 제거 (필터링 후)
    if not skeleton.any():
        width_map = np.zeros(mask.shape, dtype=np.float32)
        return skeleton, width_map

    dist = distance_transform_edt(mask)
    width_map = np.zeros(mask.shape, dtype=np.float32)
    width_map[skeleton] = (dist[skeleton] * 2.0).astype(np.float32)
    return skeleton, width_map
