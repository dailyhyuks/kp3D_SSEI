# KP3D v2 — Plan 2/5: Stage 1 직물 제거 심화 + 자가 경쟁 게이트 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 색 레이어 C에서 직물(직조) 패턴을 격자 추정 → 위상 결맞음 판별 → 적응 notch 보간 → 잔차 검증 루프로 제거하고, 정규화된 선 레이어 L과 재합성해 복원 이미지 R을 만드는 `kp3d.modules.weave_removal_v2` 모듈 + 임계값 없는 자가 경쟁 게이트를 구축한다.

**Architecture:** 2-기저-벡터 격자 추정(축 정렬 가정 폐기) → 격자 고조파 예측 → 패치 위상 결맞음 검사(직물=전역 결맞음) → 피크별 Gaussian 피팅 notch 완전 보간(α_min 폐기) → 직조 대역 잔차 < σ_n 루프 → NLM h=f(σ_n). L은 [5,95] 백분위 대비 정규화. 게이트는 1/4 프록시에서 경로 A(분해)와 경로 B(v1)를 자가 경쟁시켜 argmax 선택.

**Tech Stack:** Python, numpy, scipy, opencv (cv2), pytest. Plan 1 산출물 `kp3d.modules.decomposition` 소비.

**Spec:** `docs/superpowers/specs/2026-07-05-kp3d-v2-line-color-decomposition-design.md` 섹션 1.4, 2 (+ 4.4 계약 1→2)

**소비 인터페이스 (Plan 1 산출물, main에 병합 완료):**

```python
from kp3d.modules.decomposition.decompose import DecompositionResult, decompose
# DecompositionResult: line_alpha(HxW f32 [0,1]), color_layer(HxWx3 u8), line_mask(HxW bool),
#                      skeleton(HxW bool), width_map(HxW f32), weave, noise_sigma(float)
from kp3d.modules.decomposition.statistics import estimate_noise_sigma  # (gray 2D) -> float
```

**v1 게이트 경로 B 인터페이스 (기존 코드, 수정 금지):**

```python
from kp3d.modules.weave_removal.base import WeaveRemovalConfig, WeaveRemovalModule
# WeaveRemovalModule(config=None).process_bgr(img_bgr) -> (result_bgr u8, confidence_map)
# WeaveRemovalConfig().patch_size == 64 (기본)
# 주의: base.py는 torch를 임포트하므로 게이트에서 지연 임포트한다.
```

## Global Constraints

- **Training-free**: 학습·fine-tuning 코드 금지. 순수 고전 알고리즘만 (게이트의 경로 B는 기존 v1 고전 모듈 호출).
- **P-adapt**: 튜닝 상수 금지. 허용 상수는 3종뿐이며 각각 코드 주석으로 근거 명시: ① 수학적 유도 상수(나이퀴스트 0.5, 반 픽셀 0.5, 1/√12, Hann 50% 겹침=COLA), ② 안전 상한(`_MAX_ITERS=10`, 피팅 창 반경 ≤ 패치/4), ③ 정규화 상수(스펙이 명시한 유도 규칙: 패치 = 최대 주기 ×8 이상, 프록시 1/4 해상도, [5,95] 백분위).
- **계약 1→2 (스펙 §4.4)**: 산출물은 R(RGB u8)과 σ_n. 기계 검증 불변식 = "직조 대역 잔차 에너지 < σ_n" (잔차 루프의 종료 조건이자 테스트 대상).
- **v1 코드 수정 금지**: `weave_removal/` 이하는 읽기 전용 (게이트가 호출만 한다).
- 모든 공개 함수는 입력/출력 dtype을 docstring에 명시.
- 테스트는 저장소 루트에서 `python -m pytest tests/<file> -v` 로 실행 (기존 flat 구조 준수).
- 커밋 메시지: 기존 저장소 스타일 (영어 명령형, 예: `Add ...`, `Fix ...`).
- 작업 디렉터리: `C:\Users\admin\korean-painting-3d`

## File Structure

```
src/kp3d/modules/weave_removal_v2/
  __init__.py      # 공개 API re-export
  lattice.py       # estimate_lattice, predict_peak_freqs, LatticeResult
  coherence.py     # phase_coherence
  notch.py         # fit_peak_gaussian, interpolate_notch
  removal.py       # derive_patch_size, weave_band_energy, remove_weave, WeaveRemovalV2Result
  line_layer.py    # normalize_line_contrast
  restore.py       # compose_over, restore, RestorationResult
  gate.py          # self_competition_gate, GateResult
tests/
  test_wr2_lattice.py
  test_wr2_coherence.py
  test_wr2_notch.py
  test_wr2_removal.py
  test_wr2_line_layer.py
  test_wr2_restore.py
  test_wr2_gate.py
  test_wr2_full.py
scripts/
  demo_weave_removal_v2.py   # 실이미지 육안 검증용 데모
```

---

### Task 1: 격자 추정 (2-기저-벡터, 축 정렬 가정 없음)

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Create: `src/kp3d/modules/weave_removal_v2/lattice.py`
- Test: `tests/test_wr2_lattice.py`

**Interfaces:**
- Consumes: 없음 (최초 태스크)
- Produces:
  - `LatticeResult` — 필드 `basis: np.ndarray (K,2) float64` (행 = 공간 기저 (dy,dx) 픽셀, K∈{0,1,2}), `freq_basis: np.ndarray (K,2) float64` (행 = 주파수 기저 (fy,fx) cycles/pixel), `strength: float` (채택 피크 평균 정규화 자기상관 높이, K=0이면 0.0)
  - `estimate_lattice(gray: np.ndarray) -> LatticeResult`
  - `predict_peak_freqs(lattice: LatticeResult) -> np.ndarray` — (N,2) (fy,fx), 나이퀴스트 이내 상반평면 대표만. Task 2·4가 소비.

- [ ] **Step 1: Write the failing test**

`tests/test_wr2_lattice.py` 생성:

```python
"""격자 추정 (2-기저-벡터) 테스트."""
import numpy as np

from kp3d.modules.weave_removal_v2.lattice import (
    LatticeResult,
    estimate_lattice,
    predict_peak_freqs,
)


def _grid(theta_deg: float, p1: float, p2: float, size: int = 192,
          amp: float = 10.0) -> np.ndarray:
    """회전각 theta의 2방향 격자 합성 이미지 (float64 2D)."""
    th = np.deg2rad(theta_deg)
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    return (128.0
            + amp * np.cos(2.0 * np.pi * u / p1)
            + amp * np.cos(2.0 * np.pi * v / p2))


def test_axis_aligned_grid_recovers_periods():
    lat = estimate_lattice(_grid(0.0, 8.0, 12.0))
    assert isinstance(lat, LatticeResult)
    assert lat.basis.shape == (2, 2)
    norms = sorted(np.linalg.norm(lat.basis, axis=1))
    assert abs(norms[0] - 8.0) <= 1.0
    assert abs(norms[1] - 12.0) <= 1.0
    assert lat.strength > 0.2


def test_rotated_grid_recovers_periods():
    lat = estimate_lattice(_grid(15.0, 7.0, 9.0))
    assert lat.basis.shape == (2, 2)
    norms = sorted(np.linalg.norm(lat.basis, axis=1))
    assert abs(norms[0] - 7.0) <= 1.0
    assert abs(norms[1] - 9.0) <= 1.0
    # 두 기저는 비공선
    b1, b2 = lat.basis
    assert abs(b1[0] * b2[1] - b1[1] * b2[0]) > 1.0


def test_flat_image_yields_empty_lattice():
    lat = estimate_lattice(np.full((96, 96), 130.0))
    assert lat.basis.shape[0] == 0
    assert lat.strength == 0.0
    assert predict_peak_freqs(lat).shape == (0, 2)


def test_predict_contains_fundamentals_within_nyquist():
    lat = estimate_lattice(_grid(0.0, 8.0, 12.0))
    freqs = predict_peak_freqs(lat)
    assert freqs.shape[0] >= 2
    assert np.all(np.abs(freqs) <= 0.5 + 1e-12)
    # 기본 주파수 (0, 1/8), (1/12, 0)이 예측에 포함 (부호 대칭 감안해 절대값 비교)
    def _has(target):
        d = np.abs(np.abs(freqs) - np.abs(np.asarray(target))).sum(axis=1)
        return d.min() < 1e-6
    assert _has((0.0, 1.0 / 8.0))
    assert _has((1.0 / 12.0, 0.0))


def test_estimate_lattice_rejects_non_2d():
    import pytest
    with pytest.raises(ValueError):
        estimate_lattice(np.zeros((4, 4, 3)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wr2_lattice.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kp3d.modules.weave_removal_v2'`

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/weave_removal_v2/__init__.py` 생성:

```python
"""Stage 1 v2: 색 레이어 직물 제거 + 자가 경쟁 게이트 (스펙 §1.4, §2)."""
from .lattice import LatticeResult, estimate_lattice, predict_peak_freqs

__all__ = [
    "LatticeResult",
    "estimate_lattice",
    "predict_peak_freqs",
]
```

`src/kp3d/modules/weave_removal_v2/lattice.py` 생성:

```python
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
    """상반평면(dy>0 또는 dy==0,dx>0)의 양수 국소 최대 [(높이, (dy,dx)), ...] 높이 내림차순."""
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
    peaks.sort(key=lambda p: -p[0])
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
    k = 2
    while True:
        cand = vec / k
        if float(np.linalg.norm(cand)) < _MIN_PERIOD:
            break
        rounded = np.round(cand)
        if (float(np.linalg.norm(cand - rounded)) <= _HALF_PIXEL
                and _is_local_max(shifted, cy, cx, rounded)):
            best = rounded.astype(np.float64)
        k += 1
    return best


def _collinear(v: np.ndarray, b: np.ndarray) -> bool:
    """v가 b 방향 직선에서 수직 거리 반 픽셀 이내인지."""
    nb = float(np.linalg.norm(b))
    perp = abs(float(v[0] * b[1] - v[1] * b[0])) / nb
    return perp <= _HALF_PIXEL


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
    b1 = _reduce_to_fundamental(peaks[0][1], shifted, cy, cx)
    strengths = [peaks[0][0]]
    basis = [b1]
    for val, vec in peaks[1:]:
        if not _collinear(vec, b1):
            basis.append(_reduce_to_fundamental(vec, shifted, cy, cx))
            strengths.append(val)
            break
    if len(basis) == 2:
        r1, r2 = _gauss_reduce(basis[0], basis[1])
        bmat = np.array([r1, r2], dtype=np.float64)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wr2_lattice.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/__init__.py src/kp3d/modules/weave_removal_v2/lattice.py tests/test_wr2_lattice.py
git commit -m "Add 2-basis-vector weave lattice estimation with harmonic prediction"
```

---

### Task 2: 위상 결맞음 검사

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/coherence.py`
- Modify: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Test: `tests/test_wr2_coherence.py`

**Interfaces:**
- Consumes: 없음 (freq는 튜플/배열로 직접 받음 — Task 1과 독립 테스트 가능)
- Produces: `phase_coherence(patches_fft: np.ndarray (P,S,S) complex, offsets: np.ndarray (P,2) float64, freq: np.ndarray (2,) float64, patch_size: int) -> tuple[float, np.ndarray]` — (전역 결맞음 R∈[0,1], 패치별 일치 가중 w (P,) float64 ∈[0,1]). Task 4가 notch 감쇠 가중으로 소비.

핵심 원리 (스펙 §2 ②): 전역 주기 신호는 패치 오프셋 (y0,x0)에 대해 위상이 정확히 2π(fy·y0+fx·x0)만큼 이동한다. 이를 보상한 위상들의 크기 가중 평균 벡터 길이 R이 결맞음 점수다 — 직물이면 R≈1, 내용물(비결맞음)이면 R≈0. 임계값 없이 R 자체를 감쇠 가중으로 쓴다 (P-adapt).

- [ ] **Step 1: Write the failing test**

`tests/test_wr2_coherence.py` 생성:

```python
"""패치 위상 결맞음 테스트."""
import numpy as np

from kp3d.modules.weave_removal_v2.coherence import phase_coherence


def test_global_sinusoid_is_coherent():
    size, s, n = 256, 64, 40
    rng = np.random.default_rng(7)
    xx = np.arange(size, dtype=np.float64)[None, :]
    img = 128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0) * np.ones((size, 1))
    offs = rng.integers(0, size - s, size=(n, 2)).astype(np.float64)
    patches = np.stack(
        [img[int(y):int(y) + s, int(x):int(x) + s] for y, x in offs]
    )
    pf = np.fft.fft2(patches)
    r, w = phase_coherence(pf, offs, np.array([0.0, 1.0 / 8.0]), s)
    assert r > 0.9
    assert w.shape == (n,)
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert w.mean() > 0.9


def test_random_phase_patches_are_incoherent():
    s, n = 64, 40
    rng = np.random.default_rng(11)
    xx = np.arange(s, dtype=np.float64)[None, :]
    patches = np.stack([
        128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0
                              + rng.uniform(0.0, 2.0 * np.pi)) * np.ones((s, 1))
        for _ in range(n)
    ])
    offs = np.zeros((n, 2), dtype=np.float64)
    pf = np.fft.fft2(patches)
    r, w = phase_coherence(pf, offs, np.array([0.0, 1.0 / 8.0]), s)
    assert r < 0.5
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_coherent_beats_incoherent():
    """서수 비교: 결맞음 점수는 전역 신호 > 무작위 위상 (P-adapt 검증)."""
    s, n = 64, 30
    rng = np.random.default_rng(3)
    xx = np.arange(s, dtype=np.float64)[None, :]
    coh = np.stack([128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0)
                    * np.ones((s, 1)) for _ in range(n)])
    inc = np.stack([128.0 + 10.0 * np.cos(2.0 * np.pi * xx / 8.0
                                          + rng.uniform(0.0, 2.0 * np.pi))
                    * np.ones((s, 1)) for _ in range(n)])
    offs = np.zeros((n, 2), dtype=np.float64)
    f = np.array([0.0, 1.0 / 8.0])
    r_coh, _ = phase_coherence(np.fft.fft2(coh), offs, f, s)
    r_inc, _ = phase_coherence(np.fft.fft2(inc), offs, f, s)
    assert r_coh > r_inc


def test_zero_magnitude_bin_returns_zero():
    s, n = 32, 5
    pf = np.zeros((n, s, s), dtype=np.complex128)
    offs = np.zeros((n, 2), dtype=np.float64)
    r, w = phase_coherence(pf, offs, np.array([0.0, 1.0 / 8.0]), s)
    assert r == 0.0
    assert np.all(w == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wr2_coherence.py -v`
Expected: FAIL — `ModuleNotFoundError` (coherence)

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/weave_removal_v2/coherence.py` 생성:

```python
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
```

`__init__.py`의 import/`__all__`에 `phase_coherence` 추가.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wr2_coherence.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/coherence.py src/kp3d/modules/weave_removal_v2/__init__.py tests/test_wr2_coherence.py
git commit -m "Add magnitude-weighted phase coherence test for weave discrimination"
```

---

### Task 3: 피크별 Gaussian 피팅 적응 notch 보간

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/notch.py`
- Modify: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Test: `tests/test_wr2_notch.py`

**Interfaces:**
- Consumes: 없음 (스펙트럼 배열 직접 입력 — 독립 테스트 가능)
- Produces:
  - `fit_peak_gaussian(log_mag: np.ndarray (S,S) float64, peak: tuple[int,int]) -> tuple[float, float, float, int]` — (sigma_y, sigma_x [빈], amplitude [바닥 초과 로그 크기], support_radius [빈])
  - `interpolate_notch(fft_patch: np.ndarray (S,S) complex128, peak, sigma_yx: tuple[float,float], amplitude: float, support_radius: int, weight: float) -> np.ndarray` — 새 complex128 배열. Task 4가 소비.

v1의 고정 annulus [2,5]와 α_min=0.3을 폐기한다: notch 형상 = 실측 Gaussian 피팅 폭, 배경 = 지지 창 경계 링 크기 중앙값, 감쇠 = 완전 보간(weight로 결맞음 가중). 켤레 빈을 동시 처리해 역변환의 실수성을 보존한다.

- [ ] **Step 1: Write the failing test**

`tests/test_wr2_notch.py` 생성:

```python
"""적응 notch 보간 테스트."""
import numpy as np
import pytest

from kp3d.modules.weave_removal_v2.notch import (
    fit_peak_gaussian,
    interpolate_notch,
)


def _spectrum_with_peak(s: int = 64, amp: float = 12.0, period: float = 8.0):
    rng = np.random.default_rng(3)
    xx = np.arange(s, dtype=np.float64)[None, :]
    img = (rng.normal(0.0, 1.0, (s, s))
           + amp * np.cos(2.0 * np.pi * xx / period) * np.ones((s, 1)))
    return img, np.fft.fft2(img)


def test_fit_peak_gaussian_finds_narrow_peak():
    _, spec = _spectrum_with_peak()
    sy, sx, amplitude, radius = fit_peak_gaussian(np.log1p(np.abs(spec)), (0, 8))
    assert amplitude > 0.0
    assert radius >= 1
    assert 1.0 / np.sqrt(12.0) <= sy <= 4.0
    assert 1.0 / np.sqrt(12.0) <= sx <= 4.0


def test_interpolate_notch_suppresses_peak_and_preserves_rest():
    img, spec = _spectrum_with_peak()
    logm = np.log1p(np.abs(spec))
    sy, sx, amplitude, radius = fit_peak_gaussian(logm, (0, 8))
    out = interpolate_notch(spec, (0, 8), (sy, sx), amplitude, radius, 1.0)
    # 피크 크기가 배경 수준으로 감쇠
    assert np.abs(out[0, 8]) < 0.1 * np.abs(spec[0, 8])
    # 지지 창 밖 원거리 빈은 완전 불변
    assert np.abs(out[32, 32]) == pytest.approx(np.abs(spec[32, 32]))
    # 실수성 보존 (켤레 동시 처리)
    rec = np.fft.ifft2(out)
    assert float(np.max(np.abs(rec.imag))) < 1e-8
    # 사인파 에너지 대부분 제거
    assert rec.real.std() < img.std() * 0.5


def test_zero_weight_is_identity():
    _, spec = _spectrum_with_peak()
    logm = np.log1p(np.abs(spec))
    sy, sx, amplitude, radius = fit_peak_gaussian(logm, (0, 8))
    out = interpolate_notch(spec, (0, 8), (sy, sx), amplitude, radius, 0.0)
    assert np.array_equal(out, spec)


def test_flat_spectrum_fit_returns_zero_amplitude():
    logm = np.zeros((32, 32))
    sy, sx, amplitude, _ = fit_peak_gaussian(logm, (4, 4))
    assert amplitude == 0.0
    assert sy == pytest.approx(1.0 / np.sqrt(12.0))
    assert sx == pytest.approx(1.0 / np.sqrt(12.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_wr2_notch.py -v`
Expected: FAIL — `ModuleNotFoundError` (notch)

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/weave_removal_v2/notch.py` 생성:

```python
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
```

`__init__.py`의 import/`__all__`에 `fit_peak_gaussian`, `interpolate_notch` 추가.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_wr2_notch.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/notch.py src/kp3d/modules/weave_removal_v2/__init__.py tests/test_wr2_notch.py
git commit -m "Add Gaussian-fitted adaptive notch interpolation"
```

---

### Task 4: 직물 제거 코어 (removal.py)

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/removal.py`
- Modify: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Test: `tests/test_wr2_removal.py`

**Interfaces:**
- Consumes:
  - Task 1: `estimate_lattice(gray) -> LatticeResult`, `predict_peak_freqs(lattice) -> (N,2) float64`
  - Task 2: `phase_coherence(patches_fft, offsets, freq, patch_size) -> (float, (P,) float64)`
  - Task 3: `fit_peak_gaussian(log_mag, peak) -> (sigma_y, sigma_x, amplitude, support_radius)`, `interpolate_notch(fft_patch, peak, sigma_yx, amplitude, support_radius, weight) -> complex128`
  - Plan 1: `kp3d.modules.decomposition.estimate_noise_sigma(gray) -> float`
- Produces:
  - `WeaveRemovalV2Result` dataclass: `cleaned (H,W,3) uint8`, `lattice: LatticeResult`, `iterations: int`, `residual_energy: float`, `noise_sigma: float`
  - `derive_patch_size(lattice: LatticeResult, image_shape: tuple[int,int]) -> int` — 최대 격자 주기×8을 2의 거듭제곱으로 올림, 상한 = min(H,W) 이하 최대 2의 거듭제곱. Task 7·8이 소비하지 않음(내부용이지만 테스트 가능하도록 공개).
  - `weave_band_energy(gray: np.ndarray float, lattice: LatticeResult) -> float` — 예측 피크 빈의 국소 바닥 초과 에너지를 공간 RMS 진폭(그레이 레벨 단위)으로 환산. Task 7 게이트의 감쇠율 지표가 소비.
  - `remove_weave(image_bgr: np.ndarray (H,W,3) uint8, noise_sigma: float | None = None) -> WeaveRemovalV2Result` — Task 6이 소비.

**P-adapt 근거 (이 태스크의 모든 수치):**
- 패치 = 최대 주기 × 8 (스펙 §2 명시 유도 규칙), 2의 거듭제곱 반올림은 FFT 효율(수학적 형식), 하한은 나이퀴스트에서 자동(최소 주기 2px × 8 = 16)
- Hann 50% 겹침 = COLA 조건(수학 유도)
- RMS 환산 `√(2·ΣE²)/N`: 진폭 A 사인파의 반평면 피크 크기 |F| = A·N/2, RMS = A/√2 → 피크 초과 E에 대해 RMS = √2·E/N (순수 유도)
- 국소 바닥 창 반경 = min(H,W)/(2·최대 주기) = 기본 주파수 빈 간격의 절반(나이퀴스트 논리)
- 루프 종료: 직조 대역 잔차 < σ_n (측정치 대 측정치), `_MAX_ITERS = 10` (안전 상한)
- NLM h = √(σ_n² + σ_res²) — 독립 잔차 에너지의 RMS 합성(수학 유도), OpenCV 창 크기는 라이브러리 기본값 사용

- [ ] **Step 1: Write the failing tests**

`tests/test_wr2_removal.py`:

```python
"""removal.py 테스트: 패치 유도, 대역 에너지 보정, 직물 제거 코어."""
import cv2
import numpy as np
import pytest

from kp3d.modules.weave_removal_v2 import (
    LatticeResult,
    derive_patch_size,
    estimate_lattice,
    remove_weave,
    weave_band_energy,
)


def _lattice_from_basis(basis: np.ndarray) -> LatticeResult:
    b = np.asarray(basis, dtype=np.float64)
    freq = np.linalg.inv(b).T if b.shape[0] == 2 else b / np.linalg.norm(b[0]) ** 2
    return LatticeResult(basis=b, freq_basis=freq, strength=1.0)


def _weave_painting(h: int = 256, w: int = 256, amp: float = 12.0) -> np.ndarray:
    """비주기 베이스(가우시안 블롭) + 축 정렬 직조(주기 8/12) 3채널 이미지."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 100.0 + 60.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = amp * np.cos(2 * np.pi * yy / 8.0) + amp * np.cos(2 * np.pi * xx / 12.0)
    gray = np.clip(base + weave, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def test_derive_patch_size_is_eight_periods_power_of_two():
    lat8 = _lattice_from_basis([[8.0, 0.0], [0.0, 8.0]])
    assert derive_patch_size(lat8, (512, 512)) == 64
    lat12 = _lattice_from_basis([[12.0, 0.0], [0.0, 12.0]])
    assert derive_patch_size(lat12, (512, 512)) == 128  # 96 -> 128
    # 상한: min(H,W)=100 -> 2^floor(log2(100)) = 64
    lat40 = _lattice_from_basis([[40.0, 0.0], [0.0, 40.0]])
    assert derive_patch_size(lat40, (100, 100)) == 64


def test_weave_band_energy_matches_sinusoid_rms():
    h = w = 256
    amp = 10.0
    yy = np.mgrid[0:h, 0:w][0].astype(np.float64)
    gray = (128.0 + amp * np.cos(2 * np.pi * yy / 8.0)).astype(np.float32)
    lattice = estimate_lattice(gray)
    assert lattice.basis.shape[0] >= 1
    energy = weave_band_energy(gray, lattice)
    expected = amp / np.sqrt(2.0)
    assert abs(energy - expected) < 0.15 * expected


def test_remove_weave_reduces_band_energy():
    img = _weave_painting()
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    e0 = weave_band_energy(gray0, lattice)
    result = remove_weave(img)
    assert result.iterations >= 1
    gray1 = cv2.cvtColor(result.cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray1, lattice)
    assert e1 < 0.5 * e0
    assert result.cleaned.shape == img.shape
    assert result.cleaned.dtype == np.uint8


def test_remove_weave_preserves_content():
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 100.0 + 60.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    img = _weave_painting()
    result = remove_weave(img)
    err_before = np.mean(np.abs(img[:, :, 0].astype(np.float64) - base))
    err_after = np.mean(np.abs(result.cleaned[:, :, 0].astype(np.float64) - base))
    assert err_after < err_before


def test_remove_weave_no_lattice_is_identity():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    result = remove_weave(img)
    assert result.iterations == 0
    assert np.array_equal(result.cleaned, img)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wr2_removal.py -v`
Expected: FAIL — `ImportError: cannot import name 'derive_patch_size'`

- [ ] **Step 3: Write the implementation**

`src/kp3d/modules/weave_removal_v2/removal.py`:

```python
"""직물 제거 코어: 격자 유도 적응 notch + WOLA 재합성 + 잔차 루프 (스펙 §2)."""
from dataclasses import dataclass

import cv2
import numpy as np

from kp3d.modules.decomposition import estimate_noise_sigma
from kp3d.modules.weave_removal_v2.coherence import phase_coherence
from kp3d.modules.weave_removal_v2.lattice import (
    LatticeResult,
    estimate_lattice,
    predict_peak_freqs,
)
from kp3d.modules.weave_removal_v2.notch import fit_peak_gaussian, interpolate_notch

_MAX_ITERS = 10  # 안전 상한 (P-adapt 허용 상수 ②)


@dataclass
class WeaveRemovalV2Result:
    """Stage 1 v2 직물 제거 결과."""
    cleaned: np.ndarray        # (H,W,3) uint8
    lattice: LatticeResult
    iterations: int
    residual_energy: float     # 최종 직조 대역 RMS (그레이 레벨)
    noise_sigma: float


def derive_patch_size(lattice: LatticeResult, image_shape: tuple[int, int]) -> int:
    """패치 = 최대 격자 주기 × 8, 2의 거듭제곱 올림, 상한 min(H,W) 이하."""
    h, w = image_shape
    max_period = max(float(np.linalg.norm(b)) for b in lattice.basis)
    target = max_period * 8.0
    size = 1 << int(np.ceil(np.log2(target)))
    cap = 1 << int(np.floor(np.log2(min(h, w))))
    return int(min(size, cap))


def weave_band_energy(gray: np.ndarray, lattice: LatticeResult) -> float:
    """예측 직조 피크의 국소 바닥 초과 에너지 -> 공간 RMS 진폭.

    유도: 진폭 A 사인파의 반평면 FFT 피크 |F| = A*N/2, RMS = A/sqrt(2)
    -> 초과 E_k에 대해 RMS 합성 = sqrt(2 * sum(E_k^2)) / N.
    """
    if lattice.basis.shape[0] == 0:
        return 0.0
    g = np.asarray(gray, dtype=np.float64)
    h, w = g.shape
    n = h * w
    mag = np.abs(np.fft.fft2(g - g.mean()))
    max_period = max(float(np.linalg.norm(b)) for b in lattice.basis)
    radius = max(1, int(round(min(h, w) / (2.0 * max_period))))
    total = 0.0
    for fy, fx in predict_peak_freqs(lattice):
        by = int(round(fy * h)) % h
        bx = int(round(fx * w)) % w
        ys = np.arange(by - radius, by + radius + 1) % h
        xs = np.arange(bx - radius, bx + radius + 1) % w
        window = mag[np.ix_(ys, xs)]
        ring = np.concatenate(
            [window[0, :], window[-1, :], window[1:-1, 0], window[1:-1, -1]]
        )
        floor = float(np.median(ring))
        excess = max(0.0, float(mag[by, bx]) - floor)
        total += excess * excess
    return float(np.sqrt(2.0 * total) / n)


def _hann2d(size: int) -> np.ndarray:
    """주기적 Hann 창 (50% 겹침 COLA)."""
    n = np.arange(size)
    w1 = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / size)
    return np.outer(w1, w1)


def _filter_once(gray: np.ndarray, lattice: LatticeResult, patch_size: int) -> np.ndarray:
    """Hann WOLA 패치 순회로 결맞음 가중 notch 1회 적용."""
    g = np.asarray(gray, dtype=np.float64)
    h, w = g.shape
    s = patch_size
    stride = s // 2
    padded = np.pad(g, stride, mode="reflect")
    ph, pw = padded.shape
    window = _hann2d(s)
    origins = [
        (y0, x0)
        for y0 in range(0, ph - s + 1, stride)
        for x0 in range(0, pw - s + 1, stride)
    ]

    patch_ffts = np.stack(
        [np.fft.fft2(padded[y0:y0 + s, x0:x0 + s] * window) for y0, x0 in origins]
    )
    offsets = np.asarray(origins, dtype=np.float64)

    peak_freqs = predict_peak_freqs(lattice)
    peak_weights = []
    for freq in peak_freqs:
        r_coh, w_patch = phase_coherence(patch_ffts, offsets, freq, s)
        peak_weights.append(r_coh * w_patch)  # 전역 결맞음 x 패치 일치, 둘 다 [0,1]

    acc = np.zeros((ph, pw), dtype=np.float64)
    norm = np.zeros((ph, pw), dtype=np.float64)
    for idx, (y0, x0) in enumerate(origins):
        spec = patch_ffts[idx]
        for k, freq in enumerate(peak_freqs):
            weight = float(peak_weights[k][idx])
            if weight <= 0.0:
                continue
            peak = (int(round(freq[0] * s)) % s, int(round(freq[1] * s)) % s)
            sy, sx, amplitude, radius = fit_peak_gaussian(
                np.log1p(np.abs(spec)), peak
            )
            if amplitude <= 0.0:
                continue
            spec = interpolate_notch(spec, peak, (sy, sx), amplitude, radius, weight)
        cleaned = np.real(np.fft.ifft2(spec))
        acc[y0:y0 + s, x0:x0 + s] += cleaned * window
        norm[y0:y0 + s, x0:x0 + s] += window * window
    out = acc / np.maximum(norm, np.finfo(np.float64).tiny)
    return out[stride:stride + h, stride:stride + w]


def remove_weave(
    image_bgr: np.ndarray, noise_sigma: float | None = None
) -> WeaveRemovalV2Result:
    """직조 대역 잔차 < sigma_n까지 notch 반복 후 NLM 잔차 정리."""
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"(H,W,3) 이미지가 필요합니다: {img.shape}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sigma_n = float(estimate_noise_sigma(gray)) if noise_sigma is None else float(noise_sigma)

    lattice = estimate_lattice(gray)
    if lattice.basis.shape[0] == 0:
        return WeaveRemovalV2Result(
            cleaned=img.copy(), lattice=lattice, iterations=0,
            residual_energy=0.0, noise_sigma=sigma_n,
        )

    patch_size = derive_patch_size(lattice, gray.shape)
    channels = [img[:, :, c].astype(np.float64) for c in range(3)]
    residual = weave_band_energy(gray, lattice)
    iterations = 0
    while residual > sigma_n and iterations < _MAX_ITERS:
        channels = [_filter_once(ch, lattice, patch_size) for ch in channels]
        iterations += 1
        merged = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        work_gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = weave_band_energy(work_gray, lattice)

    if iterations == 0:
        cleaned = img.copy()
    else:
        cleaned = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        h_nlm = float(np.hypot(sigma_n, residual))  # 독립 잔차의 RMS 합성
        if h_nlm > 0.0:
            cleaned = cv2.fastNlMeansDenoisingColored(cleaned, None, h_nlm, h_nlm)
    return WeaveRemovalV2Result(
        cleaned=cleaned, lattice=lattice, iterations=iterations,
        residual_energy=float(residual), noise_sigma=sigma_n,
    )
```

`__init__.py`의 import/`__all__`에 `WeaveRemovalV2Result`, `derive_patch_size`, `remove_weave`, `weave_band_energy` 추가.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wr2_removal.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run full v2 suite**

Run: `python -m pytest tests/test_wr2_lattice.py tests/test_wr2_coherence.py tests/test_wr2_notch.py tests/test_wr2_removal.py -v`
Expected: 전부 PASS

- [ ] **Step 6: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/removal.py src/kp3d/modules/weave_removal_v2/__init__.py tests/test_wr2_removal.py
git commit -m "Add coherence-weighted WOLA weave removal core"
```

---

### Task 5: 선 레이어 대비 정규화 (line_layer.py)

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/line_layer.py`
- Modify: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Test: `tests/test_wr2_line_layer.py`

**Interfaces:**
- Consumes: 없음 (순수 numpy)
- Produces: `normalize_line_contrast(line_alpha: np.ndarray (H,W) float32 [0,1]) -> np.ndarray (H,W) float32 [0,1]` — Task 6이 소비. v1의 contour boost(×10 상수)를 대체하는 P-adapt 대비 정규화 (스펙 §2.3).

**P-adapt 근거:** [5,95] 백분위 아핀 스트레치 (스펙 명시 정규화 상수 ③). 선 픽셀(alpha>0)의 분포에서 유도, 튜닝 상수 없음. 퇴화(p95<=p5) 시 무변경.

- [ ] **Step 1: Write the failing tests**

`tests/test_wr2_line_layer.py`:

```python
"""line_layer.py 테스트: [5,95] 백분위 대비 정규화."""
import numpy as np

from kp3d.modules.weave_removal_v2 import normalize_line_contrast


def test_stretches_weak_alpha_to_full_range():
    rng = np.random.default_rng(0)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[16:48, 16:48] = rng.uniform(0.2, 0.6, (32, 32)).astype(np.float32)
    out = normalize_line_contrast(alpha)
    inside = out[alpha > 0]
    assert inside.max() > 0.9          # p95 부근이 1.0으로 스트레치
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.dtype == np.float32


def test_zero_pixels_stay_zero():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[10, 10] = 0.5
    alpha[20, 20] = 0.9
    out = normalize_line_contrast(alpha)
    assert np.all(out[alpha == 0] == 0.0)


def test_order_preserved():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[5, 5], alpha[6, 6], alpha[7, 7] = 0.3, 0.5, 0.7
    alpha[1:4, 1:20] = np.linspace(0.1, 0.9, 57).reshape(3, 19).astype(np.float32)
    out = normalize_line_contrast(alpha)
    assert out[5, 5] <= out[6, 6] <= out[7, 7]


def test_degenerate_constant_alpha_unchanged():
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 0.4
    out = normalize_line_contrast(alpha)
    assert np.array_equal(out, alpha)


def test_all_zero_alpha_unchanged():
    alpha = np.zeros((16, 16), dtype=np.float32)
    out = normalize_line_contrast(alpha)
    assert np.array_equal(out, alpha)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wr2_line_layer.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_line_contrast'`

- [ ] **Step 3: Write the implementation**

`src/kp3d/modules/weave_removal_v2/line_layer.py`:

```python
"""선 레이어 대비 정규화: v1 contour boost 대체 (스펙 §2.3, P-adapt)."""
import numpy as np


def normalize_line_contrast(line_alpha: np.ndarray) -> np.ndarray:
    """선 픽셀(alpha>0) 분포의 [5,95] 백분위 아핀 스트레치.

    p5 -> 0, p95 -> 1로 사상 후 [0,1] 클립. 0 픽셀은 0 유지.
    퇴화(선 없음 또는 p95<=p5) 시 입력 복사본 반환.
    """
    alpha = np.asarray(line_alpha, dtype=np.float32)
    mask = alpha > 0.0
    if not mask.any():
        return alpha.copy()
    inside = alpha[mask]
    p5, p95 = np.percentile(inside, [5.0, 95.0])
    if p95 <= p5:
        return alpha.copy()
    out = np.zeros_like(alpha)
    out[mask] = np.clip((inside - p5) / (p95 - p5), 0.0, 1.0)
    return out
```

`__init__.py`의 import/`__all__`에 `normalize_line_contrast` 추가.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wr2_line_layer.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/line_layer.py src/kp3d/modules/weave_removal_v2/__init__.py tests/test_wr2_line_layer.py
git commit -m "Add percentile-based line contrast normalization"
```

---

### Task 6: 복원 파이프라인 조립 (restore.py)

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/restore.py`
- Modify: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Test: `tests/test_wr2_restore.py`

**Interfaces:**
- Consumes:
  - Plan 1: `kp3d.modules.decomposition.decompose(image_bgr) -> DecompositionResult`(`.line_alpha f32`, `.color_layer u8`, `.noise_sigma float`), `kp3d.modules.decomposition.recompose(image_bgr, line_alpha, color_layer) -> (H,W,3) u8` (alpha==0 → C 정확 복사 불변식 내장 — 재구현하지 않고 재사용, DRY)
  - Task 4: `remove_weave(image_bgr, noise_sigma) -> WeaveRemovalV2Result`
  - Task 5: `normalize_line_contrast(line_alpha) -> line_alpha`
- Produces:
  - `RestorationResult` dataclass: `restored (H,W,3) uint8`, `line_alpha (H,W) float32` (정규화 후), `color_cleaned (H,W,3) uint8`, `weave: WeaveRemovalV2Result`, `noise_sigma: float`
  - `restore(image_bgr: np.ndarray (H,W,3) uint8) -> RestorationResult` — Task 7 게이트의 경로 A이자 계약 1→2의 산출(R + σ_n, 스펙 §4.4).

**파이프라인 (스펙 §2):** `decompose` → C에 `remove_weave`(σ_n은 분해 결과 재사용, 중복 추정 금지) → L에 `normalize_line_contrast` → `recompose`로 R = L over C′.

- [ ] **Step 1: Write the failing tests**

`tests/test_wr2_restore.py`:

```python
"""restore.py 테스트: 분해 -> 직물 제거 -> 재합성 조립."""
import cv2
import numpy as np

from kp3d.modules.weave_removal_v2 import (
    estimate_lattice,
    restore,
    weave_band_energy,
)


def _weave_painting_with_lines(h: int = 256, w: int = 256) -> np.ndarray:
    """가우시안 블롭 베이스 + 어두운 선 몇 개 + 직조(주기 8/12)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 110.0 + 50.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = 12.0 * np.cos(2 * np.pi * yy / 8.0) + 12.0 * np.cos(2 * np.pi * xx / 12.0)
    gray = np.clip(base + weave, 0, 255).astype(np.uint8)
    img = np.stack([gray, gray, gray], axis=-1)
    cv2.line(img, (32, 32), (224, 32), (20, 20, 20), 3)
    cv2.line(img, (32, 32), (32, 224), (20, 20, 20), 3)
    cv2.circle(img, (128, 128), 60, (30, 30, 30), 3)
    return img


def test_restore_reduces_weave_band_energy():
    img = _weave_painting_with_lines()
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    e0 = weave_band_energy(gray0, lattice)
    result = restore(img)
    gray1 = cv2.cvtColor(result.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray1, lattice)
    assert e1 < e0
    assert result.restored.shape == img.shape
    assert result.restored.dtype == np.uint8


def test_zero_alpha_pixels_copy_cleaned_color():
    img = _weave_painting_with_lines()
    result = restore(img)
    zero = result.line_alpha == 0.0
    assert zero.any()
    assert np.array_equal(result.restored[zero], result.color_cleaned[zero])


def test_noise_sigma_from_decomposition_is_propagated():
    img = _weave_painting_with_lines()
    result = restore(img)
    assert result.noise_sigma == result.weave.noise_sigma
    assert result.noise_sigma >= 0.0


def test_line_alpha_is_normalized_range():
    img = _weave_painting_with_lines()
    result = restore(img)
    assert result.line_alpha.dtype == np.float32
    assert result.line_alpha.min() >= 0.0
    assert result.line_alpha.max() <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wr2_restore.py -v`
Expected: FAIL — `ImportError: cannot import name 'restore'`

- [ ] **Step 3: Write the implementation**

`src/kp3d/modules/weave_removal_v2/restore.py`:

```python
"""Stage 1 v2 복원 조립: 분해 -> C 직물 제거 -> L 정규화 -> 재합성 (스펙 §2, §4.4)."""
from dataclasses import dataclass

import numpy as np

from kp3d.modules.decomposition import decompose, recompose
from kp3d.modules.weave_removal_v2.line_layer import normalize_line_contrast
from kp3d.modules.weave_removal_v2.removal import WeaveRemovalV2Result, remove_weave


@dataclass
class RestorationResult:
    """계약 1->2 산출: R(restored) + noise_sigma."""
    restored: np.ndarray       # (H,W,3) uint8
    line_alpha: np.ndarray     # (H,W) float32, 정규화 후
    color_cleaned: np.ndarray  # (H,W,3) uint8
    weave: WeaveRemovalV2Result
    noise_sigma: float


def restore(image_bgr: np.ndarray) -> RestorationResult:
    """R = normalize(L) over remove_weave(C)."""
    img = np.asarray(image_bgr)
    dec = decompose(img)
    weave = remove_weave(dec.color_layer, noise_sigma=dec.noise_sigma)
    alpha = normalize_line_contrast(dec.line_alpha)
    restored = recompose(img, alpha, weave.cleaned)
    return RestorationResult(
        restored=restored,
        line_alpha=alpha,
        color_cleaned=weave.cleaned,
        weave=weave,
        noise_sigma=dec.noise_sigma,
    )
```

`__init__.py`의 import/`__all__`에 `RestorationResult`, `restore` 추가.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wr2_restore.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/restore.py src/kp3d/modules/weave_removal_v2/__init__.py tests/test_wr2_restore.py
git commit -m "Add restoration pipeline assembly for stage 1 v2"
```

---

### Task 7: 자가 경쟁 게이트 (gate.py)

**Files:**
- Create: `src/kp3d/modules/weave_removal_v2/gate.py`
- Modify: `src/kp3d/modules/weave_removal_v2/__init__.py`
- Test: `tests/test_wr2_gate.py`

**Interfaces:**
- Consumes:
  - Task 1: `estimate_lattice`, Task 4: `weave_band_energy`, Task 6: `restore`
  - Plan 1: `kp3d.modules.decomposition.decompose` (프록시의 line_mask), `estimate_noise_sigma`
  - v1 (읽기 전용, **지연 임포트 필수** — `kp3d.modules.weave_removal.base`가 torch를 임포트): `WeaveRemovalModule(config=None).process_bgr(img_bgr) -> (result_bgr, confidence)`, `WeaveRemovalConfig().patch_size == 64`
- Produces:
  - `GateResult` dataclass: `restored (H,W,3) uint8`, `winner: str` ("v2"|"v1"), `quality_v2: float`, `quality_v1: float`, `noise_sigma: float`
  - `self_competition_gate(image_bgr: np.ndarray (H,W,3) uint8) -> GateResult` — Stage 1 v2의 최상위 진입점. Task 8·차기 플랜(Stage 2)이 소비.
  - 내부 훅(테스트 대체용 모듈 레벨 함수): `_run_v2(image_bgr) -> (H,W,3) u8`, `_run_v1(image_bgr) -> (H,W,3) u8`, `_proxy_scale(image_shape, patch_size) -> float`

**게이트 알고리즘 (스펙 §1.4):**
1. 프록시 배율 `scale = min(1.0, max(0.25, patch_size / min(H,W)))` — 1/4 프록시(스펙 명시)이되 v1 patch_size(=64, v1 config에서 유도)보다 작아지지 않게 클램프. 새 상수 없음.
2. 프록시에서 경로 A(`_run_v2` = restore)·경로 B(`_run_v1` = v1 모듈) 실행.
3. 품질 `Q = reduction × preservation` (무차원 곱, 가중 상수 없음):
   - `reduction = clip(1 − E_after/E_before, 0, 1)` — `weave_band_energy`(프록시 원본 격자 기준). `E_before == 0`(격자 없음 포함)이면 두 경로 모두 reduction=0 → 동률.
   - `preservation` — Scharr 그래디언트 크기의 Pearson 상관(원본 vs 결과), `decompose(프록시).line_mask` 픽셀로 한정. 마스크가 비면(몰골법) 1.0(자명) → Q는 reduction만으로 결정. 상관은 [−1,1]을 [0,1]로 클립.
4. `winner = argmax(Q)`, **동률(Q_v2 <= Q_v1) 시 v1 안전망** (스펙 §1.4 명시).
5. 승자 경로를 전체 해상도에 적용, `noise_sigma = estimate_noise_sigma(전체 해상도 gray)`와 함께 반환 (계약 1→2).

- [ ] **Step 1: Write the failing tests**

`tests/test_wr2_gate.py` — v1(torch)을 실제 임포트하지 않도록 `_run_v1`/`_run_v2`를 monkeypatch하여 결정적 단위 테스트:

```python
"""gate.py 테스트: 자가 경쟁 게이트 (v1/v2 경로는 monkeypatch로 대체).

이미지는 128x128로 고정한다: 프록시 배율이 max(0.25, 64/128)=0.5가 되어
직조 주기 8/12px가 프록시에서 4/6px로 보존된다. (256px면 배율 0.25에서
주기 8px가 나이퀴스트 2px에 걸려 INTER_AREA 축소로 소멸 -> 항상 동률.)
"""
import cv2
import numpy as np

import kp3d.modules.weave_removal_v2.gate as gate_mod
from kp3d.modules.weave_removal_v2 import self_competition_gate
from kp3d.modules.weave_removal_v2.gate import _proxy_scale


def _painting_pair(h: int = 128, w: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """(직조 포함 이미지, 직조 없는 정답 이미지) 쌍. 선·베이스는 동일."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 110.0 + 50.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = 12.0 * np.cos(2 * np.pi * yy / 8.0) + 12.0 * np.cos(2 * np.pi * xx / 12.0)

    def _finish(field: np.ndarray) -> np.ndarray:
        gray = np.clip(field, 0, 255).astype(np.uint8)
        img = np.stack([gray, gray, gray], axis=-1)
        cv2.line(img, (16, 16), (112, 16), (20, 20, 20), 3)
        cv2.line(img, (16, 16), (16, 112), (20, 20, 20), 3)
        cv2.circle(img, (64, 64), 30, (30, 30, 30), 3)
        return img

    return _finish(base + weave), _finish(base)


def _fake_path(clean_full: np.ndarray):
    """어떤 크기 입력이 와도 정답(clean)을 그 크기로 반환하는 페이크 경로."""

    def run(image: np.ndarray) -> np.ndarray:
        if image.shape[:2] == clean_full.shape[:2]:
            return clean_full.copy()
        return cv2.resize(
            clean_full, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return run


def test_proxy_scale_derivation():
    assert _proxy_scale((1024, 1024), 64) == 0.25          # 1/4 프록시
    assert _proxy_scale((128, 200), 64) == 0.5             # 64/128 클램프
    assert _proxy_scale((48, 48), 64) == 1.0               # min(1.0, ...) 상한


def test_v2_wins_when_v1_is_identity(monkeypatch):
    img, clean = _painting_pair()
    monkeypatch.setattr(gate_mod, "_run_v2", _fake_path(clean))
    monkeypatch.setattr(gate_mod, "_run_v1", lambda image: image.copy())
    result = self_competition_gate(img)
    assert result.winner == "v2"
    assert result.quality_v2 > result.quality_v1
    assert result.restored.shape == img.shape
    assert np.array_equal(result.restored, clean)  # 승자 경로가 전체 해상도에 적용됨


def test_v1_wins_when_v2_is_identity(monkeypatch):
    img, clean = _painting_pair()
    monkeypatch.setattr(gate_mod, "_run_v2", lambda image: image.copy())
    monkeypatch.setattr(gate_mod, "_run_v1", _fake_path(clean))
    result = self_competition_gate(img)
    assert result.winner == "v1"
    assert result.quality_v1 > result.quality_v2
    assert np.array_equal(result.restored, clean)


def test_tie_falls_back_to_v1(monkeypatch):
    img, _clean = _painting_pair()
    monkeypatch.setattr(gate_mod, "_run_v2", lambda image: image.copy())
    monkeypatch.setattr(gate_mod, "_run_v1", lambda image: image.copy())
    result = self_competition_gate(img)
    assert result.winner == "v1"                            # 동률 -> v1 안전망


def test_no_weave_image_falls_back_to_v1(monkeypatch):
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(gate_mod, "_run_v2", lambda image: image.copy())
    monkeypatch.setattr(gate_mod, "_run_v1", lambda image: image.copy())
    result = self_competition_gate(img)
    assert result.winner == "v1"                            # E_before=0 -> 동률 -> v1
    assert result.noise_sigma >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wr2_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kp3d.modules.weave_removal_v2.gate'`

- [ ] **Step 3: Write the implementation**

`src/kp3d/modules/weave_removal_v2/gate.py`:

```python
"""자가 경쟁 게이트: 1/4 프록시에서 v2 vs v1 경합, 승자를 전체 해상도 적용 (스펙 §1.4)."""
from dataclasses import dataclass

import cv2
import numpy as np

from kp3d.modules.decomposition import decompose, estimate_noise_sigma
from kp3d.modules.weave_removal_v2.lattice import estimate_lattice
from kp3d.modules.weave_removal_v2.removal import weave_band_energy
from kp3d.modules.weave_removal_v2.restore import restore


@dataclass
class GateResult:
    """게이트 산출: 승자 경로의 R + noise_sigma (계약 1->2)."""
    restored: np.ndarray   # (H,W,3) uint8
    winner: str            # "v2" | "v1"
    quality_v2: float
    quality_v1: float
    noise_sigma: float


def _proxy_scale(image_shape: tuple[int, int], patch_size: int) -> float:
    """1/4 프록시(스펙 §1.4), 단 v1 patch_size 미만으로 줄지 않게 클램프."""
    h, w = image_shape[:2]
    return float(min(1.0, max(0.25, patch_size / min(h, w))))


def _run_v2(image_bgr: np.ndarray) -> np.ndarray:
    """경로 A: v2 복원 파이프라인."""
    return restore(image_bgr).restored


def _run_v1(image_bgr: np.ndarray) -> np.ndarray:
    """경로 B: v1 WeaveRemovalModule (torch 지연 임포트)."""
    from kp3d.modules.weave_removal import WeaveRemovalModule

    result_bgr, _confidence = WeaveRemovalModule(config=None).process_bgr(image_bgr)
    return np.asarray(result_bgr, dtype=np.uint8)


def _v1_patch_size() -> int:
    """v1 config에서 patch_size 유도 (torch 지연 임포트)."""
    from kp3d.modules.weave_removal import WeaveRemovalConfig

    return int(WeaveRemovalConfig().patch_size)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return np.hypot(gx, gy)


def _preservation(orig_gray: np.ndarray, result_gray: np.ndarray,
                  line_mask: np.ndarray) -> float:
    """line_mask 한정 Scharr 그래디언트 Pearson 상관, [0,1] 클립. 마스크 비면 1.0."""
    mask = np.asarray(line_mask, dtype=bool)
    if not mask.any():
        return 1.0
    a = _gradient_magnitude(orig_gray)[mask]
    b = _gradient_magnitude(result_gray)[mask]
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(corr, 0.0, 1.0))


def _quality(proxy_gray: np.ndarray, result_bgr: np.ndarray,
             lattice, e_before: float, line_mask: np.ndarray) -> float:
    """Q = 직조 잔차 감쇠율 x 선 보존 상관 (무차원 곱)."""
    result_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if e_before <= 0.0:
        reduction = 0.0
    else:
        e_after = weave_band_energy(result_gray, lattice)
        reduction = float(np.clip(1.0 - e_after / e_before, 0.0, 1.0))
    return reduction * _preservation(proxy_gray, result_gray, line_mask)


def self_competition_gate(image_bgr: np.ndarray) -> GateResult:
    """프록시 경합으로 v2/v1을 선택하고 승자를 전체 해상도에 적용."""
    img = np.asarray(image_bgr)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"(H,W,3) 이미지가 필요합니다: {img.shape}")

    scale = _proxy_scale(img.shape[:2], _v1_patch_size())
    if scale < 1.0:
        proxy = cv2.resize(img, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        proxy = img.copy()
    proxy_gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lattice = estimate_lattice(proxy_gray)
    e_before = weave_band_energy(proxy_gray, lattice)
    line_mask = decompose(proxy).line_mask

    q_v2 = _quality(proxy_gray, _run_v2(proxy), lattice, e_before, line_mask)
    q_v1 = _quality(proxy_gray, _run_v1(proxy), lattice, e_before, line_mask)

    winner = "v2" if q_v2 > q_v1 else "v1"  # 동률 -> v1 안전망 (스펙 §1.4)
    restored = _run_v2(img) if winner == "v2" else _run_v1(img)

    full_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return GateResult(
        restored=restored, winner=winner,
        quality_v2=q_v2, quality_v1=q_v1,
        noise_sigma=float(estimate_noise_sigma(full_gray)),
    )
```

**주의 (테스트 대체 가능성):** `self_competition_gate`는 `_run_v2`/`_run_v1`을 모듈 레벨 이름으로 호출해야 monkeypatch가 적용된다 — 함수 내부에서 지역 별칭으로 캡처하지 말 것. `_v1_patch_size`는 torch를 임포트하므로 게이트 단위 테스트에서 문제되면 `WeaveRemovalConfig`가 torch 없이 임포트되는지 확인하고, torch까지 끌려오면 테스트에서 `_v1_patch_size`도 monkeypatch한다(예: `lambda: 64`). 구현자는 `python -c "from kp3d.modules.weave_removal.config import WeaveRemovalConfig"`로 먼저 확인할 것.

`__init__.py`의 import/`__all__`에 `GateResult`, `self_competition_gate` 추가.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_wr2_gate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/weave_removal_v2/gate.py src/kp3d/modules/weave_removal_v2/__init__.py tests/test_wr2_gate.py
git commit -m "Add self-competition gate between v2 and v1 paths"
```

---

### Task 8: 통합 테스트 + 데모 스크립트

**Files:**
- Create: `tests/test_wr2_full.py`
- Create: `scripts/demo_weave_removal_v2.py`

**Interfaces:**
- Consumes: Task 7 `self_competition_gate`, Task 6 `restore`, Task 4 `weave_band_energy`, Task 1 `estimate_lattice`
- Produces: 실행 가능한 데모 스크립트 (차기 플랜의 평가 하네스가 참조). 새 공개 API 없음.

**주의:** 통합 테스트는 실제 v1 경로(torch)를 사용하므로 `pytest.importorskip("torch")`. 실이미지 스모크는 `data/ablation_study/images/1_0004.png` 부재 시 skip. 데모 출력은 `output/weave_removal_v2_demo/`에 저장하되 **커밋하지 않는다** (Plan 1 리뷰의 Minor 지적 재발 방지 — `output/`이 `.gitignore`에 없으면 이 태스크에서 추가).

- [ ] **Step 1: Write the integration tests**

`tests/test_wr2_full.py`:

```python
"""Stage 1 v2 전 구간 통합 테스트 (실제 v1 경로 포함)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")  # v1 경로가 torch를 요구

from kp3d.modules.weave_removal_v2 import (
    estimate_lattice,
    self_competition_gate,
    weave_band_energy,
)

_REAL_IMAGE = Path("data/ablation_study/images/1_0004.png")


def _weave_painting_with_lines(h: int = 128, w: int = 128) -> np.ndarray:
    """128x128 고정: 프록시 배율 0.5에서 직조 주기 8/12px가 4/6px로 보존됨."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 110.0 + 50.0 * np.exp(
        -((yy - h / 2) ** 2 + (xx - w / 2) ** 2) / (2 * (h / 4) ** 2)
    )
    weave = 12.0 * np.cos(2 * np.pi * yy / 8.0) + 12.0 * np.cos(2 * np.pi * xx / 12.0)
    gray = np.clip(base + weave, 0, 255).astype(np.uint8)
    img = np.stack([gray, gray, gray], axis=-1)
    cv2.line(img, (16, 16), (112, 16), (20, 20, 20), 3)
    cv2.circle(img, (64, 64), 30, (30, 30, 30), 3)
    return img


def test_gate_end_to_end_on_synthetic_weave():
    img = _weave_painting_with_lines()
    result = self_competition_gate(img)
    assert result.winner in ("v2", "v1")
    assert result.restored.shape == img.shape
    assert result.restored.dtype == np.uint8
    assert result.noise_sigma >= 0.0
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    e0 = weave_band_energy(gray0, lattice)
    gray1 = cv2.cvtColor(result.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray1, lattice)
    assert e1 < e0  # 어느 경로가 이기든 직조 에너지는 감소해야 함


@pytest.mark.skipif(not _REAL_IMAGE.exists(), reason="실이미지 데이터 없음")
def test_gate_smoke_on_real_image():
    img = cv2.imread(str(_REAL_IMAGE), cv2.IMREAD_COLOR)
    assert img is not None
    result = self_competition_gate(img)
    assert result.restored.shape == img.shape
    assert result.restored.dtype == np.uint8
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray0)
    if lattice.basis.shape[0] > 0:
        e0 = weave_band_energy(gray0, lattice)
        gray1 = cv2.cvtColor(result.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
        e1 = weave_band_energy(gray1, lattice)
        assert e1 <= e0  # 직조 에너지가 늘어나면 안 됨
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_wr2_full.py -v`
Expected: PASS (2 tests; torch 또는 데이터 부재 시 해당 항목 skip)

- [ ] **Step 3: Write the demo script**

`scripts/demo_weave_removal_v2.py` (기존 `scripts/demo_decomposition.py`의 구조를 따름):

```python
"""Stage 1 v2 데모: 게이트 실행 결과와 중간 산출물을 저장.

사용법: python scripts/demo_weave_removal_v2.py [이미지 경로]
기본 이미지: data/ablation_study/images/1_0004.png
출력: output/weave_removal_v2_demo/
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from kp3d.modules.weave_removal_v2 import (
    estimate_lattice,
    restore,
    self_competition_gate,
    weave_band_energy,
)

_DEFAULT = "data/ablation_study/images/1_0004.png"


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {path}")
        return 1
    out_dir = Path("output/weave_removal_v2_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lattice = estimate_lattice(gray)
    e0 = weave_band_energy(gray, lattice)

    r = restore(img)
    gate = self_competition_gate(img)
    gray_gate = cv2.cvtColor(gate.restored, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e1 = weave_band_energy(gray_gate, lattice)

    cv2.imwrite(str(out_dir / "input.png"), img)
    cv2.imwrite(str(out_dir / "v2_restored.png"), r.restored)
    cv2.imwrite(str(out_dir / "v2_color_cleaned.png"), r.color_cleaned)
    cv2.imwrite(str(out_dir / "v2_line_alpha.png"),
                (r.line_alpha * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "gate_restored.png"), gate.restored)

    print(f"격자 기저 수 K={lattice.basis.shape[0]}, strength={lattice.strength:.3f}")
    print(f"직조 대역 에너지: {e0:.3f} -> {e1:.3f}")
    print(f"v2 반복 횟수: {r.weave.iterations}, sigma_n={r.noise_sigma:.3f}")
    print(f"게이트 승자: {gate.winner} (Q_v2={gate.quality_v2:.4f}, "
          f"Q_v1={gate.quality_v1:.4f})")
    print(f"저장 위치: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the demo on the real image**

Run: `python scripts/demo_weave_removal_v2.py`
Expected: 종료 코드 0, `output/weave_removal_v2_demo/`에 PNG 5장, 게이트 승자·에너지 감소 로그 출력. 결과 이미지를 눈으로 확인할 필요는 없음(정량 로그로 충분).

- [ ] **Step 5: Ensure demo outputs are not committed**

`.gitignore`에 `output/` 항목이 없으면 추가:

```bash
grep -qx "output/" .gitignore || echo "output/" >> .gitignore
```

- [ ] **Step 6: Run the full v2 suite one last time**

Run: `python -m pytest tests/test_wr2_lattice.py tests/test_wr2_coherence.py tests/test_wr2_notch.py tests/test_wr2_removal.py tests/test_wr2_line_layer.py tests/test_wr2_restore.py tests/test_wr2_gate.py tests/test_wr2_full.py -v`
Expected: 전부 PASS (환경에 따른 skip 제외)

- [ ] **Step 7: Commit**

```bash
git add tests/test_wr2_full.py scripts/demo_weave_removal_v2.py .gitignore
git commit -m "Add stage 1 v2 integration tests and demo script"
```

---

## 태스크 의존 순서

Task 1 → Task 2 → Task 3 → Task 4(1·2·3 소비) → Task 5(독립, 4 이후 권장) → Task 6(4·5 소비) → Task 7(1·4·6 소비) → Task 8(7 소비)
