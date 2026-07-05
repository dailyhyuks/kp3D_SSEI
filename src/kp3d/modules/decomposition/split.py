"""L/C 레이어 분리와 재합성.

L (선 레이어) = 원본 RGB + soft alpha (선 응답 강도에서 유도).
C (색 레이어) = 선을 Telea inpainting으로 제거한 원본.
불변식: alpha==0 픽셀에서 recompose(I, alpha, C) == C, 그리고
inpaint 마스크 밖에서 C == I → 합성하면 L over C == I (선 영역 외 정확 일치).
"""
import cv2
import numpy as np


def split_layers(
    image_bgr: np.ndarray,
    response: np.ndarray,
    line_mask: np.ndarray,
    skeleton: np.ndarray,
    width_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """선·색 레이어 분리.

    Args:
        image_bgr: HxWx3 uint8 원본.
        response: detect_lines의 float 응답.
        line_mask: detect_lines의 bool 마스크.
        skeleton, width_map: measure_line_widths 출력.
    Returns:
        (line_alpha float32 [0,1], color_layer uint8 HxWx3, inpaint_mask bool)
    """
    img = np.asarray(image_bgr)
    mask = np.asarray(line_mask, dtype=bool)

    # soft alpha: 마스크 내 응답을 95th 백분위로 정규화 (P-adapt: 분포에서 유도)
    line_alpha = np.zeros(mask.shape, dtype=np.float32)
    inside = response[mask]
    if inside.size > 0:
        scale = float(np.percentile(inside, 95))
        if scale > 0:
            line_alpha[mask] = np.clip(response[mask] / scale, 0.0, 1.0).astype(
                np.float32
            )

    # inpaint 반경 = 스켈레톤 선폭 중앙값 (P-adapt: 폭 지도에서 유도)
    widths = width_map[skeleton]
    radius = int(np.ceil(float(np.median(widths)))) if widths.size > 0 else 1
    inpaint_mask = mask
    color_layer = cv2.inpaint(
        img, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA
    )
    # cv2.inpaint는 마스크 밖을 건드리지 않지만 불변식을 명시적으로 보장
    color_layer[~inpaint_mask] = img[~inpaint_mask]
    return line_alpha, color_layer, inpaint_mask


def recompose(
    image_bgr: np.ndarray, line_alpha: np.ndarray, color_layer: np.ndarray
) -> np.ndarray:
    """L over C 알파 합성. alpha==0 픽셀은 color_layer를 그대로 복사 (정확 일치 보장).

    Returns:
        HxWx3 uint8.
    """
    img = np.asarray(image_bgr, dtype=np.float64)
    c = np.asarray(color_layer, dtype=np.float64)
    a = np.asarray(line_alpha, dtype=np.float64)[..., None]
    blended = a * img + (1.0 - a) * c
    out = np.rint(blended).clip(0, 255).astype(np.uint8)
    zero = line_alpha == 0.0
    out[zero] = color_layer[zero]  # 부동소수 반올림 오차 차단
    return out
