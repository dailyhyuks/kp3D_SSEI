"""Stage 0 오케스트레이션: 2-pass 부트스트랩 분해 (v2 설계 1.1-1.2).

순서: 통계 측정 -> RGF -> 1차 선 추출 -> 선폭 분포 측정 ->
      파라미터 확정 -> 2차(최종) 선 추출 -> 레이어 분리.
"""
from dataclasses import dataclass

import numpy as np

# 8비트 양자화 노이즈 바닥 (Δ=1, σ=Δ/√12) — 수학 유도 상수
_QUANT_NOISE_FLOOR = 1.0 / np.sqrt(12.0)

from kp3d.modules.decomposition.lines import detect_lines, measure_line_widths
from kp3d.modules.decomposition.split import recompose, split_layers
from kp3d.modules.decomposition.statistics import (
    WeavePeriodResult,
    estimate_noise_sigma,
    estimate_weave_period,
)
from kp3d.modules.decomposition.structure import compute_structure_image


@dataclass
class DecompositionResult:
    """Stage 0 출력. 후속 스테이지 계약(설계 4.4)의 전달물."""
    line_alpha: np.ndarray
    color_layer: np.ndarray
    line_mask: np.ndarray
    skeleton: np.ndarray
    width_map: np.ndarray
    weave: WeavePeriodResult
    noise_sigma: float


def _initial_width_range(gray_shape: tuple[int, ...]) -> tuple[float, float]:
    """1차 패스의 광역 선폭 범위: [1px, 대각선의 1%].

    부트스트랩 시작점 — 2차 패스에서 실측 분포 백분위로 대체됨 (P-adapt).
    """
    diag = float(np.hypot(*gray_shape))
    return 1.0, max(2.0, diag * 0.01)


def decompose(image_bgr: np.ndarray) -> DecompositionResult:
    """한국화 BGR 이미지를 선/색 레이어로 분해.

    Args:
        image_bgr: HxWx3 uint8.
    """
    img = np.asarray(image_bgr)
    gray = img.astype(np.float64).mean(axis=2)

    noise_sigma = estimate_noise_sigma(gray)
    weave = estimate_weave_period(gray)
    periods = [p for p in (weave.period_x, weave.period_y) if np.isfinite(p)]
    diag = float(np.hypot(*gray.shape))
    sigma_s = max(periods) if periods else max(1.0, 0.005 * diag)  # 직조 미검출 시: 대각선 0.5% (정규화 상수), 최소 1px
    structure = compute_structure_image(
        gray.astype(np.float32), sigma_s=sigma_s, noise_sigma=max(noise_sigma, _QUANT_NOISE_FLOOR)
    )

    # 1차 패스: 광역 범위로 선 후보 추출 -> 선폭 분포 측정
    lo0, hi0 = _initial_width_range(gray.shape)
    _, mask1 = detect_lines(structure, min_width=lo0, max_width=hi0)
    skel1, wmap1 = measure_line_widths(mask1)
    widths1 = wmap1[skel1]

    # 2차 패스: 실측 폭 분포 [5th, 95th] 백분위로 범위 확정 (P-adapt)
    if widths1.size > 0:
        lo = max(1.0, float(np.percentile(widths1, 5)))
        hi = max(lo + 1.0, float(np.percentile(widths1, 95)))
    else:
        lo, hi = lo0, hi0
    response, line_mask = detect_lines(structure, min_width=lo, max_width=hi)
    skeleton, width_map = measure_line_widths(line_mask)

    line_alpha, color_layer, _ = split_layers(
        img, response, line_mask, skeleton, width_map
    )
    return DecompositionResult(
        line_alpha=line_alpha,
        color_layer=color_layer,
        line_mask=line_mask,
        skeleton=skeleton,
        width_map=width_map,
        weave=weave,
        noise_sigma=noise_sigma,
    )


def recompose_result(
    image_bgr: np.ndarray, result: DecompositionResult
) -> np.ndarray:
    """DecompositionResult에서 L over C 재합성."""
    return recompose(image_bgr, result.line_alpha, result.color_layer)
