"""이미지 통계 측정: 노이즈 바닥, 직조 주기.

P-adapt 원칙: 이 모듈의 출력이 파이프라인 전체의 동적 파라미터 기준이 된다.
"""
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
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = g.shape
    conv = convolve2d(g, kernel, mode="valid")
    sigma = np.sqrt(np.pi / 2.0) * np.sum(np.abs(conv)) / (6.0 * (w - 2) * (h - 2))
    return float(sigma)
