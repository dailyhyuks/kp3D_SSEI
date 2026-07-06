"""패치 위상 결맞음 — 직물(전역 결맞음) vs 내용물(비결맞음) 판별 (스펙 §2 ②).

v1의 75th percentile 피크 선별을 물리적 판별 원리로 대체한다:
직물은 전역적으로 위상이 결맞는 주기 신호이고, 그림 내용물은 아니다.
"""
from __future__ import annotations

import numpy as np


def phase_coherence(
    patches_fft: np.ndarray,
    offsets: np.ndarray,
    freq: np.ndarray,
    patch_size: int,
) -> tuple[float, np.ndarray]:
    """오프셋 보상 위상의 크기 가중 결맞음.

    입력: patches_fft (P,S,S) complex128 — 각 패치의 2D FFT.
          offsets (P,2) float64 — 각 패치의 좌상단 (y0,x0).
          freq (2,) float64 — (fy,fx) cycles/pixel. patch_size S.
    반환: (R, w) — R float ∈[0,1] 전역 결맞음(크기 가중 평균 벡터 길이),
          w (P,) float64 ∈[0,1] 패치별 평균 위상 일치 가중 (1+cosΔ)/2.
    """
    s = patch_size
    ky = int(round(float(freq[0]) * s)) % s
    kx = int(round(float(freq[1]) * s)) % s
    coef = patches_fft[:, ky, kx]
    mag = np.abs(coef)
    total = float(mag.sum())
    if total == 0.0:
        return 0.0, np.zeros(patches_fft.shape[0], dtype=np.float64)
    comp = np.angle(coef) - 2.0 * np.pi * (
        float(freq[0]) * offsets[:, 0] + float(freq[1]) * offsets[:, 1]
    )
    z = np.exp(1j * comp)
    mean_z = (mag * z).sum() / total  # 크기 가중 — 무직물 패치의 위상 잡음 억제
    r = float(np.abs(mean_z))
    mu = float(np.angle(mean_z))
    w = 0.5 * (1.0 + np.cos(comp - mu))
    return r, w.astype(np.float64)
