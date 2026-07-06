"""피크별 Gaussian 피팅 적응 notch 보간 (스펙 §2 ③).

notch 크기 = 실측 피팅 폭 (고정 annulus [2,5] 폐기),
감쇠 = 완전 보간 × 결맞음 가중 (α_min=0.3 폐기).
"""
from __future__ import annotations

import numpy as np

# 단일 빈 집중 피크의 모멘트 하한: 빈 양자화 표준편차 1/√12 — 수학 유도 상수
_BIN_SIGMA_FLOOR = 1.0 / np.sqrt(12.0)


def _ring_values(arr: np.ndarray, peak: tuple[int, int], radius: int) -> np.ndarray:
    """피크 중심 정사각 창(반경 radius)의 경계 링 값들 (순환 인덱싱)."""
    s = arr.shape[0]
    ys = np.arange(peak[0] - radius, peak[0] + radius + 1) % s
    xs = np.arange(peak[1] - radius, peak[1] + radius + 1) % s
    win = arr[np.ix_(ys, xs)]
    return np.concatenate([win[0, :], win[-1, :], win[1:-1, 0], win[1:-1, -1]])


def fit_peak_gaussian(
    log_mag: np.ndarray, peak: tuple[int, int]
) -> tuple[float, float, float, int]:
    """로그 크기 스펙트럼의 피크에 모멘트 기반 2D Gaussian 피팅.

    창 반경은 경계 링 중앙값이 더 이상 감소하지 않을 때(국소 바닥 도달)까지
    확장한다 — 외부 상수 없음. 반경 상한 = 패치의 1/4 (이웃 고조파 침범 방지
    안전 상한).
    입력: log_mag (S,S) float64, peak (ky,kx).
    반환: (sigma_y, sigma_x, amplitude, support_radius).
    """
    s = log_mag.shape[0]
    max_r = max(1, s // 4)  # 안전 상한
    r = 1
    prev = float(np.median(_ring_values(log_mag, peak, 1)))
    while r + 1 <= max_r:
        nxt = float(np.median(_ring_values(log_mag, peak, r + 1)))
        if nxt >= prev:
            break
        prev = nxt
        r += 1
    floor = prev
    ys = np.arange(peak[0] - r, peak[0] + r + 1) % s
    xs = np.arange(peak[1] - r, peak[1] + r + 1) % s
    win = np.clip(log_mag[np.ix_(ys, xs)] - floor, 0.0, None)
    amplitude = float(win.max())
    total = float(win.sum())
    if amplitude <= 0.0 or total == 0.0:
        return _BIN_SIGMA_FLOOR, _BIN_SIGMA_FLOOR, 0.0, r
    dy = np.arange(-r, r + 1, dtype=np.float64)[:, None]
    dx = np.arange(-r, r + 1, dtype=np.float64)[None, :]
    my = float((win * dy).sum()) / total
    mx = float((win * dx).sum()) / total
    sy = float(np.sqrt(max(float((win * (dy - my) ** 2).sum()) / total, 0.0)))
    sx = float(np.sqrt(max(float((win * (dx - mx) ** 2).sum()) / total, 0.0)))
    return max(sy, _BIN_SIGMA_FLOOR), max(sx, _BIN_SIGMA_FLOOR), amplitude, r


def interpolate_notch(
    fft_patch: np.ndarray,
    peak: tuple[int, int],
    sigma_yx: tuple[float, float],
    amplitude: float,
    support_radius: int,
    weight: float,
) -> np.ndarray:
    """Gaussian 프로파일 notch를 링 배경 크기로 완전 보간. 새 배열 반환.

    입력: fft_patch (S,S) complex128. weight ∈[0,1] — 위상 결맞음 가중
          (0이면 무변화). 켤레 빈을 동시 처리해 ifft 실수성을 보존한다.
    """
    out = fft_patch.copy()
    if weight <= 0.0 or amplitude <= 0.0:
        return out
    s = fft_patch.shape[0]
    sy, sx = sigma_yx
    r = support_radius
    targets = {(peak[0] % s, peak[1] % s), ((-peak[0]) % s, (-peak[1]) % s)}
    for py, px in targets:
        ys = np.arange(py - r, py + r + 1) % s
        xs = np.arange(px - r, px + r + 1) % s
        sub = out[np.ix_(ys, xs)]
        dy = np.arange(-r, r + 1, dtype=np.float64)[:, None]
        dx = np.arange(-r, r + 1, dtype=np.float64)[None, :]
        g = np.exp(-0.5 * ((dy / sy) ** 2 + (dx / sx) ** 2))
        mag = np.abs(sub)
        ring = np.concatenate([mag[0, :], mag[-1, :], mag[1:-1, 0], mag[1:-1, -1]])
        bg = float(np.median(ring))
        blend = weight * g
        new_mag = mag * (1.0 - blend) + bg * blend
        phase = np.angle(sub)
        out[np.ix_(ys, xs)] = new_mag * np.exp(1j * phase)
    return out
