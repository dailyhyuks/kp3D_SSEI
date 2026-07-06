"""이미지 통계 측정: 노이즈 바닥, 직조 주기.

P-adapt 원칙: 이 모듈의 출력이 파이프라인 전체의 동적 파라미터 기준이 된다.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import convolve2d


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkær(1996) 고속 노이즈 추정.

    라플라시안 유사 커널 응답의 절대값 평균에서 가우시안 노이즈 σ를 유도.
    √(π/2)와 분모 6은 커널에서 수학적으로 유도되는 상수 (튜닝 아님).

    Args:
        gray: 2D float 배열 (grayscale).
    Returns:
        추정 노이즈 표준편차 (밝기 단위).
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gray must be a 2D array")
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = g.shape
    conv = convolve2d(g, kernel, mode="valid")
    sigma = np.sqrt(np.pi / 2.0) * np.sum(np.abs(conv)) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)


@dataclass(frozen=True)
class WeavePeriodResult:
    """축별 직조 주기와 자기상관 피크 강도.

    strength는 정규화 자기상관(0~1)의 피크값. 임계 판단은 호출측 책임
    (P-adapt: 이 모듈은 측정만 하고 게이트는 자가 경쟁이 담당).
    """
    period_x: float
    period_y: float
    strength_x: float
    strength_y: float


def _first_peak(profile: np.ndarray) -> tuple[float, float]:
    """1D 정규화 자기상관 프로파일의 첫 국소 최대 (lag>=2)를 반환.

    Returns:
        (lag, value). 국소 최대가 없으면 (nan, 0.0).
    """
    for i in range(2, len(profile)):
        left_ok = profile[i] > profile[i - 1]
        right_ok = (i + 1 >= len(profile)) or (profile[i] >= profile[i + 1])
        if left_ok and right_ok:
            return float(i), float(profile[i])
    return float("nan"), 0.0


def estimate_weave_period(gray: np.ndarray) -> WeavePeriodResult:
    """Wiener-Khinchin 자기상관으로 축별 직조 주기를 추정.

    Args:
        gray: 2D float 배열.
    """
    g = np.asarray(gray, dtype=np.float64)
    g = g - g.mean()
    f = np.fft.rfft2(g)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=g.shape)

    # Guard against flat-image case (ac.flat[0] == 0 causes division-by-zero)
    if ac.flat[0] > 0:
        ac = ac / ac.flat[0]  # lag 0 = 1로 정규화
    else:
        return WeavePeriodResult(period_x=float("nan"), period_y=float("nan"),
                                 strength_x=0.0, strength_y=0.0)

    row = ac[0, : g.shape[1] // 2]
    col = ac[: g.shape[0] // 2, 0]
    px, sx = _first_peak(row)
    py, sy = _first_peak(col)
    return WeavePeriodResult(period_x=px, period_y=py, strength_x=sx, strength_y=sy)
