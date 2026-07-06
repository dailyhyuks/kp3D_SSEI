"""패치 크기 유도(자기상관 1/e)와 PatchMatch ANN (스펙 §3.3 ③, §3.5).

패치 크기: 마스크 자기상관 F⁻¹|F(g·m)|² / F⁻¹|F(m)|² 의 방사 평균이
1/e 아래로 처음 떨어지는 lag ℓ → p = 2·ceil(ℓ)+1 (텍스처 상관 길이 포착).
"""
from __future__ import annotations

import numpy as np

# EM/탐색 반복 안전 상한 (안전 상한)
_MAX_ITERS = 10
# 상관 길이 기준 1/e — 지수 감쇠 표준 척도 (수학 유도)
_INV_E = 1.0 / np.e
# 재현성 시드 — 결과 결정성 관례 (정규화 규칙)
_SEED = 0


def derive_patch_size(gray: np.ndarray, valid: np.ndarray) -> int:
    """텍스처 상관 길이에서 패치 크기(홀수)를 유도한다."""
    g = np.asarray(gray, dtype=np.float64)
    m = np.asarray(valid, dtype=np.float64)
    h, w = g.shape
    upper = max(3, min(h, w) // 4)  # 패치 상한 (안전 상한)
    if m.sum() <= 0.0:
        return 3
    mu = float((g * m).sum() / m.sum())
    gm = (g - mu) * m
    num = np.fft.ifft2(np.abs(np.fft.fft2(gm)) ** 2).real
    den = np.fft.ifft2(np.abs(np.fft.fft2(m)) ** 2).real
    den = np.maximum(den, 1.0)  # 겹침 표본 수 하한 1 (이산 하한)
    ac = num / den
    if ac.flat[0] <= 0.0:
        ell = 1.0  # 상수 신호 — 상관 구조 없음, 최소 lag (이산 하한)
    else:
        acs = np.fft.fftshift(ac / ac.flat[0])
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.hypot(yy - cy, xx - cx).astype(np.int64)
        rmax = min(cy, cx)
        prof = np.array([float(acs[rr == r].mean()) for r in range(rmax + 1)])
        below = np.nonzero(prof < _INV_E)[0]
        ell = float(below[0]) if below.size else float(rmax)
    p = 2 * int(np.ceil(ell)) + 1
    p = int(np.clip(p, 3, upper))
    if p % 2 == 0:
        p -= 1  # 패치는 중심 대칭(홀수) — 이산 격자 (수학 유도)
    return max(p, 3)


def patchmatch(image: np.ndarray, target_mask: np.ndarray,
               pool_mask: np.ndarray, patch_size: int,
               noise_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """PatchMatch ANN — 무작위 초기화 → 홀짝 스캔 전파 → 반경 반감 랜덤 탐색.

    수렴: 평균 패치 rms 개선 < noise_sigma (잡음 바닥 이하 개선은 무의미).
    target/pool 중심의 p×p 창 in-bounds는 호출자가 보장한다.
    """
    img = np.asarray(image, dtype=np.float64)
    tm = np.asarray(target_mask, dtype=bool)
    pm = np.asarray(pool_mask, dtype=bool)
    h, w = tm.shape
    r = patch_size // 2
    pool = np.argwhere(pm)
    targets = np.argwhere(tm)
    if len(pool) == 0:
        raise ValueError("pool_mask has no valid centers")
    nnf = np.zeros((h, w, 2), dtype=np.int64)
    dists = np.full((h, w), np.inf)
    if len(targets) == 0:
        return nnf, dists
    rng = np.random.default_rng(_SEED)
    nch = img.shape[2] if img.ndim == 3 else 1

    def ssd(ty: int, tx: int, sy: int, sx: int) -> float:
        a = img[ty - r:ty + r + 1, tx - r:tx + r + 1]
        b = img[sy - r:sy + r + 1, sx - r:sx + r + 1]
        return float(((a - b) ** 2).sum())

    # 무작위 초기화
    for (ty, tx), pi in zip(targets, rng.integers(0, len(pool), len(targets))):
        sy, sx = int(pool[pi][0]), int(pool[pi][1])
        nnf[ty, tx] = (sy, sx)
        dists[ty, tx] = ssd(ty, tx, sy, sx)
    denom = float(patch_size * patch_size * nch)
    prev_rms = np.inf
    for it in range(_MAX_ITERS):
        order = targets if it % 2 == 0 else targets[::-1]
        step = 1 if it % 2 == 0 else -1
        for ty, tx in order:
            ty, tx = int(ty), int(tx)
            # 전파: 스캔 방향 이웃의 대응을 평행 이동
            for dy, dx in ((step, 0), (0, step)):
                ny, nx = ty - dy, tx - dx
                if 0 <= ny < h and 0 <= nx < w and tm[ny, nx]:
                    cy = int(nnf[ny, nx, 0]) + dy
                    cx = int(nnf[ny, nx, 1]) + dx
                    if 0 <= cy < h and 0 <= cx < w and pm[cy, cx]:
                        d = ssd(ty, tx, cy, cx)
                        if d < dists[ty, tx]:
                            dists[ty, tx] = d
                            nnf[ty, tx] = (cy, cx)
            # 랜덤 탐색: 반경 반감 (배가의 역 — 지수 탐색, 정규화 규칙)
            rad = max(h, w)
            while rad >= 1:
                by, bx = int(nnf[ty, tx, 0]), int(nnf[ty, tx, 1])
                cy = by + int(rng.integers(-rad, rad + 1))
                cx = bx + int(rng.integers(-rad, rad + 1))
                if 0 <= cy < h and 0 <= cx < w and pm[cy, cx]:
                    d = ssd(ty, tx, cy, cx)
                    if d < dists[ty, tx]:
                        dists[ty, tx] = d
                        nnf[ty, tx] = (cy, cx)
                rad //= 2
        rms = float(np.sqrt(dists[tm].mean() / denom))
        if prev_rms - rms < noise_sigma:
            break  # 잡음 바닥 이하 개선 — 수렴 (P-adapt)
        prev_rms = rms
    return nnf, dists
