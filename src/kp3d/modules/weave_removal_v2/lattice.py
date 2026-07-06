"""직조 격자 추정 — 2-기저-벡터, 축 정렬 가정 없음 (스펙 §2 ①).

v1의 축 정렬 가정(r_axis/r_cross)을 폐기하고, 자기상관의 국소 최대에서
공간 기저 2개를 찾아 쌍대(주파수) 격자로 고조파 피크를 예측한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter

# 이산 픽셀 격자의 위치 불확실성 반 픽셀 — 수학 유도 상수 (표본화 한계)
_HALF_PIXEL = 0.5
# 실수 신호 나이퀴스트 한계 (cycles/pixel) — 수학 유도 상수
_NYQUIST = 0.5
# 최소 검출 가능 주기 2px — 표본화 정리 (수학 유도 상수)
_MIN_PERIOD = 2.0


@dataclass
class LatticeResult:
    """직조 격자. basis 행 = 공간 기저 (dy, dx) [px], freq_basis 행 = (fy, fx) [cyc/px]."""

    basis: np.ndarray       # (K, 2) float64, K ∈ {0, 1, 2}
    freq_basis: np.ndarray  # (K, 2) float64
    strength: float         # 채택 피크의 평균 정규화 자기상관 높이 (0~1)


def _autocorrelation(gray: np.ndarray) -> np.ndarray:
    """Wiener-Khinchin 정규화 순환 자기상관 (원점 = 1). 상수 이미지는 전부 0."""
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gray must be a 2D array")
    g = g - g.mean()
    spec = np.fft.fft2(g)
    ac = np.fft.ifft2(np.abs(spec) ** 2).real
    if ac.flat[0] <= 0.0:
        return np.zeros_like(ac)
    return ac / ac.flat[0]


def _halfplane_peaks(shifted: np.ndarray, cy: int, cx: int) -> list[tuple[float, np.ndarray]]:
    """상반평면(dy>0 또는 dy==0,dx>0)의 양수 국소 최대 [(높이, (dy,dx)), ...] 높이 내림차순, 동률은 L1놈 오름차순."""
    h, w = shifted.shape
    local_max = shifted == maximum_filter(shifted, size=3, mode="wrap")
    peaks: list[tuple[float, np.ndarray]] = []
    ys, xs = np.nonzero(local_max)
    for y, x in zip(ys, xs):
        dy, dx = int(y - cy), int(x - cx)
        if dy < 0 or (dy == 0 and dx <= 0):
            continue  # 실신호 대칭 대표만
        if float(np.hypot(dy, dx)) < _MIN_PERIOD:
            continue  # 원점 및 표본화 한계 미만 lag 제외
        if abs(dy) > h // 2 - 1 or abs(dx) > w // 2 - 1:
            continue  # 순환 경계 제외
        val = float(shifted[y, x])
        if val <= 0.0:
            continue
        peaks.append((val, np.array([dy, dx], dtype=np.float64)))
    # 높이 내림차순, 동률은 |dy|+|dx| 오름차순 (축 정렬 우선)
    peaks.sort(key=lambda p: (-p[0], abs(p[1][0]) + abs(p[1][1])))
    return peaks


def _is_local_max(shifted: np.ndarray, cy: int, cx: int, vec: np.ndarray) -> bool:
    y, x = cy + int(round(vec[0])), cx + int(round(vec[1]))
    if not (1 <= y < shifted.shape[0] - 1 and 1 <= x < shifted.shape[1] - 1):
        return False
    win = shifted[y - 1:y + 2, x - 1:x + 2]
    return bool(shifted[y, x] > 0.0 and shifted[y, x] == win.max())


def _reduce_to_fundamental(vec: np.ndarray, shifted: np.ndarray,
                           cy: int, cx: int) -> np.ndarray:
    """정수 약수 위치가 국소 최대이면 기본 주기로 축약 (고조파 제거, 상수 없음)."""
    best = vec
    norm = float(np.linalg.norm(vec))
    for k in range(2, int(np.floor(norm / _MIN_PERIOD)) + 1):
        cand = vec / k
        rounded = np.round(cand)
        if (float(np.linalg.norm(cand - rounded)) <= _HALF_PIXEL
                and _is_local_max(shifted, cy, cx, rounded)):
            best = rounded.astype(np.float64)
    return best


def _collinear(v: np.ndarray, b: np.ndarray) -> bool:
    """v가 b 방향 직선에서 수직 거리 반 픽셀 이내인지."""
    nb = float(np.linalg.norm(b))
    perp = abs(float(v[0] * b[1] - v[1] * b[0])) / nb
    return perp <= _HALF_PIXEL


def _is_alias_direction(vec: np.ndarray, shifted: np.ndarray,
                        cy: int, cx: int) -> bool:
    """표본화 한계 미만 lag에서 이미 완전 상관인 방향(상수/에일리어스 방향)인지.

    k = floor(|v|/최소주기)+1 로 나눈 부분 lag 상관이 피크 높이와
    FFT 반올림 오차 한도 내에서 같으면(릿지) 에일리어스로 기각.
    단측 부등호는 비주기 광역 상관(블롭/선)이 소 lag 상관을 키우는 경우
    진짜 기저를 오기각하므로 등식 판정을 사용.
    허용오차는 FFT 반올림 오차 상한 eps·size (수학 유도 상수).
    """
    norm = float(np.linalg.norm(vec))
    k = int(np.floor(norm / _MIN_PERIOD)) + 1
    sub = np.round(vec / k)
    y, x = cy + int(sub[0]), cx + int(sub[1])
    if not (0 <= y < shifted.shape[0] and 0 <= x < shifted.shape[1]):
        return False
    tol = float(np.finfo(np.float64).eps) * shifted.size
    vy, vx = cy + int(round(vec[0])), cx + int(round(vec[1]))
    return abs(float(shifted[y, x]) - float(shifted[vy, vx])) <= tol


def _gauss_reduce(b1: np.ndarray, b2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange-Gauss 2D 격자 축소 — 최단 기저쌍으로 정규화 (순수 수학, 상수 없음)."""
    a, b = b1.copy(), b2.copy()
    if np.linalg.norm(a) > np.linalg.norm(b):
        a, b = b, a
    while True:
        mu = round(float(np.dot(a, b)) / float(np.dot(a, a)))
        b = b - mu * a
        if np.linalg.norm(b) >= np.linalg.norm(a):
            return a, b
        a, b = b, a


def estimate_lattice(gray: np.ndarray) -> LatticeResult:
    """gray(2D float/uint)에서 직조 격자를 추정한다.

    반환: LatticeResult. 자기상관에 양수 국소 최대가 없으면 K=0 (strength 0.0).
    """
    ac = _autocorrelation(gray)
    empty = LatticeResult(basis=np.zeros((0, 2)), freq_basis=np.zeros((0, 2)),
                          strength=0.0)
    if not np.any(ac):
        return empty
    shifted = np.fft.fftshift(ac)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    peaks = _halfplane_peaks(shifted, cy, cx)
    if not peaks:
        return empty
    # b1: 첫 비에일리어스 피크 선택
    b1 = None
    strength1 = 0.0
    for val, vec in peaks:
        if not _is_alias_direction(vec, shifted, cy, cx):
            b1 = _reduce_to_fundamental(vec, shifted, cy, cx)
            strength1 = val
            break
    if b1 is None:
        return empty
    strengths = [strength1]
    basis = [b1]
    # b2: b1과 비공선, 비에일리어스 피크 선택
    for val, vec in peaks:
        if not _collinear(vec, b1) and not _is_alias_direction(vec, shifted, cy, cx):
            basis.append(_reduce_to_fundamental(vec, shifted, cy, cx))
            strengths.append(val)
            break
    if len(basis) == 2:
        r1, r2 = _gauss_reduce(basis[0], basis[1])
        # Gauss 축소 후 에일리어스 방향이 생성되었는지 검사 (거의 공선 기저의 차분이 릿지를 만드는 경우)
        if _is_alias_direction(r1, shifted, cy, cx) or _is_alias_direction(r2, shifted, cy, cx):
            # 축소 결과가 릿지 방향이면 더 축 정렬된(L1놈 작은) 원본 기저 하나만 사용
            l1_0 = abs(basis[0][0]) + abs(basis[0][1])
            l1_1 = abs(basis[1][0]) + abs(basis[1][1])
            bmat = np.array([basis[0] if l1_0 <= l1_1 else basis[1]], dtype=np.float64)
            fmat = (bmat / (float(np.linalg.norm(bmat[0])) ** 2)).reshape(1, 2)
        else:
            bmat = np.array([r1, r2], dtype=np.float64)
            # _collinear 필터가 수직 거리 ≥ 0.5px를 보장하고 Lagrange-Gauss 축소는 행렬식을 보존하므로 |det| ≥ 1 — 역행렬 안전
            fmat = np.linalg.inv(bmat).T  # 쌍대 격자: F @ B.T = I
    else:
        bmat = np.array(basis, dtype=np.float64)
        fmat = (bmat / (float(np.linalg.norm(bmat[0])) ** 2)).reshape(1, 2)
    return LatticeResult(basis=bmat, freq_basis=fmat,
                         strength=float(np.mean(strengths)))


def predict_peak_freqs(lattice: LatticeResult) -> np.ndarray:
    """격자 고조파 주파수 (N,2) float64 (fy,fx) — 나이퀴스트 이내 상반평면 대표만."""
    k = lattice.freq_basis.shape[0]
    if k == 0:
        return np.zeros((0, 2))
    # 기저별 최대 고조파 차수 = 주기의 절반 (나이퀴스트) — 수학 유도
    orders = [max(1, int(np.floor(float(np.linalg.norm(b)) / 2.0)))
              for b in lattice.basis]
    freqs: list[np.ndarray] = []
    if k == 1:
        for m in range(1, orders[0] + 1):
            freqs.append(m * lattice.freq_basis[0])
    else:
        f1, f2 = lattice.freq_basis
        for m in range(-orders[0], orders[0] + 1):
            for n in range(-orders[1], orders[1] + 1):
                if m == 0 and n == 0:
                    continue
                f = m * f1 + n * f2
                if f[0] < 0 or (f[0] == 0 and f[1] <= 0):
                    continue  # 상반평면 대표
                freqs.append(f)
    kept = [f for f in freqs
            if abs(f[0]) <= _NYQUIST and abs(f[1]) <= _NYQUIST]
    return np.array(kept, dtype=np.float64) if kept else np.zeros((0, 2))
