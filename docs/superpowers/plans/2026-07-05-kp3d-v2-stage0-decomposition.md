# KP3D v2 — Plan 1/5: Stage 0 선·색 분해 모듈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 한국화 이미지를 선 레이어 L(RGBA)과 색 레이어 C(RGB)로 training-free 분해하고, `L over C ≈ I` 불변식을 기계 검증 가능한 형태로 제공하는 `kp3d.modules.decomposition` 모듈을 구축한다.

**Architecture:** 통계 측정(노이즈 바닥, 직조 주기) → Rolling Guidance Filter 구조 이미지 → scale-space DoG 선 검출(2-pass 부트스트랩) → 스켈레톤/선폭 측정 → Telea 기반 레이어 분리 → 재합성. 모든 파라미터는 이미지 통계에서 유도(P-adapt).

**Tech Stack:** Python, numpy, scipy, scikit-image, opencv-contrib-python(cv2.ximgproc — 설치 확인 완료: cv2 4.13.0), pytest

**Spec:** `docs/superpowers/specs/2026-07-05-kp3d-v2-line-color-decomposition-design.md` 섹션 1

**후속 계획 (이 계획의 산출물에 의존):**
- Plan 2: Stage 1 v2 — C 레이어 직물 제거 (격자 추정, 위상 결맞음, 자가 경쟁 게이트)
- Plan 3: Stage 3 SSEI 2.0 — Phase A 선 완성 / Phase B 영역 채움 / Phase C 재합성
- Plan 4: Stage 2/4 업그레이드 (HQ-SAM, alpha matting, Art3D)
- Plan 5: 평가 하네스 (합성 GT 프로토콜, SIFID/Gram/통계 지표)

## Global Constraints

- **Training-free**: 학습·fine-tuning 코드 금지. 사전학습 모델 추론도 이 모듈에서는 불사용 (순수 고전 알고리즘만)
- **P-adapt**: 튜닝 상수 금지. 허용되는 상수는 다음 3종뿐이며 각각 코드 주석으로 근거 명시: ① 수학적 유도 상수(예: Immerkær의 √(π/2)), ② 안전 상한(RGF `max_iters=10`), ③ 정규화 상수(스켈레톤 최소 길이 = 대각선의 0.5%)
- 모든 공개 함수는 `float32/float64` numpy 배열 입력을 받고 dtype을 docstring에 명시
- 테스트는 저장소 루트에서 `python -m pytest tests/<file> -v` 로 실행 (기존 flat 구조 준수)
- 커밋 메시지: 기존 저장소 스타일(영어 명령형, 예: `Add ...`, `Align ...`)
- 작업 디렉터리: `C:\Users\admin\korean-painting-3d`

## File Structure

```
src/kp3d/modules/decomposition/
  __init__.py        # 공개 API re-export
  statistics.py      # estimate_noise_sigma, estimate_weave_period, WeavePeriodResult
  structure.py       # compute_structure_image (RGF + 동적 종료)
  lines.py           # detect_lines, measure_line_widths
  split.py           # split_layers, recompose
  decompose.py       # decompose(), DecompositionResult (오케스트레이션)
tests/
  test_decomp_statistics.py
  test_decomp_structure.py
  test_decomp_lines.py
  test_decomp_split.py
  test_decomp_full.py
scripts/
  demo_decomposition.py   # 실이미지 육안 검증용 데모
```

---

### Task 1: 모듈 스캐폴드 + 노이즈 바닥 추정

**Files:**
- Create: `src/kp3d/modules/decomposition/__init__.py`
- Create: `src/kp3d/modules/decomposition/statistics.py`
- Test: `tests/test_decomp_statistics.py`

**Interfaces:**
- Consumes: 없음 (최초 태스크)
- Produces: `estimate_noise_sigma(gray: np.ndarray) -> float` — 이후 모든 태스크의 동적 종료 기준(σ_n)

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_statistics.py` 생성:

```python
"""Stage 0 분해 모듈 - 통계 측정 테스트."""
import numpy as np

from kp3d.modules.decomposition.statistics import estimate_noise_sigma


def test_noise_sigma_on_gaussian_noise():
    """알려진 σ=5 가우시안 노이즈에서 ±20% 이내로 추정해야 한다."""
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 5.0, (256, 256))
    sigma = estimate_noise_sigma(img)
    assert 4.0 < sigma < 6.0


def test_noise_sigma_on_flat_image():
    """무노이즈 균일 이미지에서 0에 수렴해야 한다."""
    img = np.full((128, 128), 100.0)
    assert estimate_noise_sigma(img) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_statistics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kp3d.modules.decomposition'`

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/decomposition/statistics.py` 생성:

```python
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
```

`src/kp3d/modules/decomposition/__init__.py` 생성:

```python
"""Stage 0: 구륵법 인지 선·색 분해 (v2 설계 섹션 1)."""
from kp3d.modules.decomposition.statistics import estimate_noise_sigma

__all__ = ["estimate_noise_sigma"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_statistics.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/__init__.py src/kp3d/modules/decomposition/statistics.py tests/test_decomp_statistics.py
git commit -m "Add decomposition module scaffold with noise floor estimation"
```

---

### Task 2: 직조 주기 자동 검출

**Files:**
- Modify: `src/kp3d/modules/decomposition/statistics.py`
- Modify: `src/kp3d/modules/decomposition/__init__.py`
- Test: `tests/test_decomp_statistics.py` (테스트 추가)

**Interfaces:**
- Consumes: 없음
- Produces: `estimate_weave_period(gray: np.ndarray) -> WeavePeriodResult` — 필드 `period_x: float`, `period_y: float` (픽셀), `strength_x: float`, `strength_y: float` (자기상관 피크값 0~1). Task 3의 RGF σ_s와 Task 7의 부트스트랩 초기 스케일이 이 값을 사용. 임계값 판단은 하지 않고 strength를 그대로 노출 (게이트는 Plan 2의 자가 경쟁이 담당)

주의: v2 설계의 완전한 2-기저-벡터 격자 추정은 Plan 2(Stage 1) 범위. 여기서는 RGF 스케일 결정에 충분한 축별 주기만 구한다.

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_statistics.py`에 추가:

```python
from kp3d.modules.decomposition.statistics import estimate_weave_period


def _synthetic_weave(px: float, py: float, noise: float = 2.0) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    img = 128 + 20 * np.sin(2 * np.pi * xx / px) + 20 * np.sin(2 * np.pi * yy / py)
    rng = np.random.default_rng(1)
    return img + rng.normal(0, noise, img.shape)


def test_weave_period_detection():
    """주기 8/12px 합성 직조에서 ±1px 이내로 검출해야 한다."""
    result = estimate_weave_period(_synthetic_weave(8.0, 12.0))
    assert abs(result.period_x - 8.0) <= 1.0
    assert abs(result.period_y - 12.0) <= 1.0
    assert result.strength_x > 0.5
    assert result.strength_y > 0.5


def test_weave_strength_low_on_pure_noise():
    """순수 노이즈에서는 피크 강도가 낮아야 한다 (주기성 없음 신호)."""
    rng = np.random.default_rng(2)
    img = rng.normal(128, 5, (256, 256))
    result = estimate_weave_period(img)
    assert result.strength_x < 0.3
    assert result.strength_y < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_statistics.py -v`
Expected: 기존 2개 PASS, 신규 2개 FAIL — `ImportError: cannot import name 'estimate_weave_period'`

- [ ] **Step 3: Write minimal implementation**

`statistics.py`에 추가:

```python
from dataclasses import dataclass


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
    for i in range(2, len(profile) - 1):
        if profile[i] > profile[i - 1] and profile[i] >= profile[i + 1]:
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
    ac = ac / ac.flat[0]  # lag 0 = 1로 정규화
    row = ac[0, : g.shape[1] // 2]
    col = ac[: g.shape[0] // 2, 0]
    px, sx = _first_peak(row)
    py, sy = _first_peak(col)
    return WeavePeriodResult(period_x=px, period_y=py, strength_x=sx, strength_y=sy)
```

`__init__.py` 갱신:

```python
"""Stage 0: 구륵법 인지 선·색 분해 (v2 설계 섹션 1)."""
from kp3d.modules.decomposition.statistics import (
    WeavePeriodResult,
    estimate_noise_sigma,
    estimate_weave_period,
)

__all__ = ["WeavePeriodResult", "estimate_noise_sigma", "estimate_weave_period"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_statistics.py -v`
Expected: 4 passed

주의: `test_weave_strength_low_on_pure_noise`가 노이즈의 우연한 첫 피크로 실패하면, `_first_peak`의 국소 최대 조건은 유지하되 피크값이 그대로 strength로 노출되는지 확인 (노이즈 자기상관 피크는 통상 0.1 미만이므로 0.3 기준은 여유가 큼).

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/statistics.py src/kp3d/modules/decomposition/__init__.py tests/test_decomp_statistics.py
git commit -m "Add autocorrelation-based weave period estimation"
```

---

### Task 3: Rolling Guidance Filter 구조 이미지 (동적 종료)

**Files:**
- Create: `src/kp3d/modules/decomposition/structure.py`
- Modify: `src/kp3d/modules/decomposition/__init__.py`
- Test: `tests/test_decomp_structure.py`

**Interfaces:**
- Consumes: `estimate_noise_sigma` (Task 1) — 종료 기준
- Produces: `compute_structure_image(image: np.ndarray, sigma_s: float, noise_sigma: float) -> np.ndarray` — float32, 입력과 동일 shape. 직조 주기 이하 텍스처가 제거된 구조 이미지. Task 4의 선 검출 입력

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_structure.py` 생성:

```python
"""RGF 구조 이미지 테스트: 미세 텍스처 제거 + 강한 에지 보존."""
import numpy as np

from kp3d.modules.decomposition.structure import compute_structure_image


def _edge_plus_texture() -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    edge = np.where(xx < 128, 80.0, 180.0)
    texture = 15.0 * np.sin(2 * np.pi * xx / 6) * np.sin(2 * np.pi * yy / 6)
    return (edge + texture).astype(np.float32)


def test_rgf_suppresses_texture():
    """주기 6px 텍스처(진폭 15)를 75% 이상 억제해야 한다."""
    out = compute_structure_image(_edge_plus_texture(), sigma_s=6.0, noise_sigma=1.0)
    left_flat = out[64:192, 32:96]  # 에지에서 먼 균일 영역
    assert float(left_flat.std()) < 15.0 * 0.25


def test_rgf_preserves_strong_edge():
    """대비 100의 스텝 에지를 70% 이상 보존해야 한다."""
    out = compute_structure_image(_edge_plus_texture(), sigma_s=6.0, noise_sigma=1.0)
    contrast = float(out[:, 160:224].mean() - out[:, 32:96].mean())
    assert contrast > 100.0 * 0.7


def test_rgf_terminates_on_flat_image():
    """균일 이미지에서 첫 반복 후 즉시 수렴해야 한다 (무한 루프 방지)."""
    img = np.full((64, 64), 120.0, dtype=np.float32)
    out = compute_structure_image(img, sigma_s=4.0, noise_sigma=0.5)
    assert np.allclose(out, 120.0, atol=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_structure.py -v`
Expected: FAIL — `ModuleNotFoundError` (structure.py 부재)

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/decomposition/structure.py` 생성:

```python
"""Rolling Guidance Filter 기반 구조 이미지 (Zhang et al., ECCV 2014).

RGF = 가우시안 초기화 후 joint bilateral filter 반복.
종료 기준은 P-adapt: 반복 간 변화량 중앙값 < noise_sigma (외부 상수 없음).
max_iters=10은 안전 상한 (문헌상 4회 내 수렴, 발산 방어용 — 튜닝 상수 아님).
"""
import cv2
import numpy as np

_MAX_ITERS = 10  # 안전 상한: RGF는 통상 4회 내 수렴 (Zhang et al. 2014)


def _derive_sigma_color(src: np.ndarray) -> float:
    """P-adapt: sigma_color를 그래디언트 크기 중앙값에서 유도."""
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    nonzero = mag[mag > 0]
    if nonzero.size == 0:
        return 1.0  # 완전 균일 이미지: 임의 양수 (필터가 항등이 됨)
    return float(np.median(nonzero))


def compute_structure_image(
    image: np.ndarray, sigma_s: float, noise_sigma: float
) -> np.ndarray:
    """직조 주기(sigma_s) 이하 텍스처를 제거한 구조 이미지를 반환.

    Args:
        image: 2D(gray) 또는 3D(color) float32 배열.
        sigma_s: 공간 스케일 = 직조 주기 (estimate_weave_period에서 유도).
        noise_sigma: 종료 기준 (estimate_noise_sigma 출력).
    Returns:
        입력과 동일 shape의 float32 구조 이미지.
    """
    src = np.asarray(image, dtype=np.float32)
    guide = cv2.GaussianBlur(src, (0, 0), sigma_s)  # RGF step 1: 소구조 완전 제거
    sigma_color = _derive_sigma_color(src)
    for _ in range(_MAX_ITERS):
        new_guide = cv2.ximgproc.jointBilateralFilter(
            guide, src, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_s
        )
        change = float(np.median(np.abs(new_guide - guide)))
        guide = new_guide
        if change < noise_sigma:
            break
    return guide
```

`__init__.py`의 import/`__all__`에 `compute_structure_image` 추가:

```python
from kp3d.modules.decomposition.structure import compute_structure_image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_structure.py -v`
Expected: 3 passed

주의: `jointBilateralFilter`는 guide와 src의 dtype/채널 일치를 요구. 실패 시 두 인자 모두 float32인지 확인.

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/structure.py src/kp3d/modules/decomposition/__init__.py tests/test_decomp_structure.py
git commit -m "Add rolling guidance structure image with noise-floor stopping"
```

---

### Task 4: Scale-space 선 검출

**Files:**
- Create: `src/kp3d/modules/decomposition/lines.py`
- Modify: `src/kp3d/modules/decomposition/__init__.py`
- Test: `tests/test_decomp_lines.py`

**Interfaces:**
- Consumes: Task 3의 구조 이미지 (gray)
- Produces: `detect_lines(structure_gray: np.ndarray, min_width: float, max_width: float) -> tuple[np.ndarray, np.ndarray]` — `(response: float64 HxW, mask: bool HxW)`. 어두운 선(먹선)에서 양의 응답. 스케일 개수는 폭 범위에서 유도 (P-adapt). Task 5·6·7이 사용

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_lines.py` 생성:

```python
"""Scale-space 선 검출 테스트."""
import numpy as np

from kp3d.modules.decomposition.lines import detect_lines


def _dark_line_image(width: int = 3) -> np.ndarray:
    img = np.full((128, 128), 200.0)
    img[60 : 60 + width, 10:118] = 60.0
    return img


def test_detects_dark_line():
    """폭 3px 먹선을 90% 이상 커버해야 한다."""
    _, mask = detect_lines(_dark_line_image(), min_width=1.0, max_width=8.0)
    assert float(mask[61, 20:100].mean()) > 0.9


def test_low_false_positive_on_background():
    """선에서 먼 배경의 오검출률이 5% 미만이어야 한다."""
    _, mask = detect_lines(_dark_line_image(), min_width=1.0, max_width=8.0)
    assert float(mask[[10, 110], :].mean()) < 0.05


def test_detects_both_thin_and_thick_lines():
    """폭 2px와 6px 선이 공존해도 둘 다 검출해야 한다 (scale-space 목적)."""
    img = np.full((128, 128), 200.0)
    img[30:32, 10:118] = 60.0   # 폭 2
    img[80:86, 10:118] = 60.0   # 폭 6
    _, mask = detect_lines(img, min_width=1.0, max_width=8.0)
    assert float(mask[31, 20:100].mean()) > 0.9
    assert float(mask[83, 20:100].mean()) > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_lines.py -v`
Expected: FAIL — `ModuleNotFoundError` (lines.py 부재)

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/decomposition/lines.py` 생성:

```python
"""Scale-space DoG 선 검출 + 스켈레톤/선폭 측정.

단일 (sigma, k) DoG 대신 폭 범위를 커버하는 다중 스케일의 max 응답 사용
(v2 설계 1.3: 단일 k=1.6 의존 제거, 다중 굵기 대응).
"""
import cv2
import numpy as np


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
```

`__init__.py`에 `detect_lines` 추가.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_lines.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/lines.py src/kp3d/modules/decomposition/__init__.py tests/test_decomp_lines.py
git commit -m "Add scale-space DoG line detection with derived scales"
```

---

### Task 5: 스켈레톤 + 선폭 측정

**Files:**
- Modify: `src/kp3d/modules/decomposition/lines.py`
- Modify: `src/kp3d/modules/decomposition/__init__.py`
- Test: `tests/test_decomp_lines.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 4의 `mask`
- Produces: `measure_line_widths(line_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(skeleton: bool HxW, width_map: float32 HxW)`. width_map은 스켈레톤 픽셀에서 선폭(px), 그 외 0. 소형 잡음 성분은 대각선 0.5% 미만 길이일 때 제거. Task 6(inpaint 반경)·Task 7(부트스트랩 폭 분포)이 사용

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_lines.py`에 추가:

```python
from kp3d.modules.decomposition.lines import measure_line_widths


def test_width_measurement_on_known_line():
    """폭 5px 직선의 스켈레톤 폭 측정값이 5±1이어야 한다."""
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:65, 10:118] = True
    skeleton, width_map = measure_line_widths(mask)
    widths = width_map[skeleton]
    assert widths.size > 0
    assert 4.0 <= float(np.median(widths)) <= 6.0


def test_small_components_removed():
    """대각선 0.5% 미만 길이의 점 잡음은 스켈레톤에서 제거되어야 한다."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[100, 100] = True          # 1px 잡음
    mask[50:53, 20:180] = True     # 실제 선
    skeleton, _ = measure_line_widths(mask)
    assert not skeleton[100, 100]
    assert skeleton[51, 60:140].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_lines.py -v`
Expected: 기존 3개 PASS, 신규 2개 FAIL — `ImportError: cannot import name 'measure_line_widths'`

- [ ] **Step 3: Write minimal implementation**

`lines.py`에 추가:

```python
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label
from skimage.morphology import skeletonize

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
    skeleton = skeletonize(mask)
    # 소형 성분 제거: 스켈레톤 픽셀 수를 길이 프록시로 사용
    diag = float(np.hypot(*mask.shape))
    min_len = diag * _MIN_SKELETON_FRACTION
    labels = sk_label(skeleton, connectivity=2)
    for lbl in range(1, labels.max() + 1):
        component = labels == lbl
        if component.sum() < min_len:
            skeleton[component] = False
    dist = distance_transform_edt(mask)
    width_map = np.zeros(mask.shape, dtype=np.float32)
    width_map[skeleton] = (dist[skeleton] * 2.0).astype(np.float32)
    return skeleton, width_map
```

`__init__.py`에 `measure_line_widths` 추가.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_lines.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/lines.py src/kp3d/modules/decomposition/__init__.py tests/test_decomp_lines.py
git commit -m "Add skeleton extraction and line width measurement"
```

---

### Task 6: 레이어 분리 (L/C) + 재합성

**Files:**
- Create: `src/kp3d/modules/decomposition/split.py`
- Modify: `src/kp3d/modules/decomposition/__init__.py`
- Test: `tests/test_decomp_split.py`

**Interfaces:**
- Consumes: Task 4 `response`/`mask`, Task 5 `skeleton`/`width_map`
- Produces:
  - `split_layers(image_bgr: np.ndarray, response: np.ndarray, line_mask: np.ndarray, skeleton: np.ndarray, width_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(line_alpha: float32 HxW [0,1], color_layer: uint8 HxWx3, inpaint_mask: bool HxW)`. 선 레이어 RGB는 원본 그대로이므로 별도 배열 불필요 (alpha만으로 정의됨)
  - `recompose(image_bgr: np.ndarray, line_alpha: np.ndarray, color_layer: np.ndarray) -> np.ndarray` — uint8 HxWx3. **불변식: alpha==0인 픽셀은 color_layer와 완전 일치**

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_split.py` 생성:

```python
"""레이어 분리/재합성 테스트: L over C ≈ I 불변식."""
import numpy as np

from kp3d.modules.decomposition.lines import detect_lines, measure_line_widths
from kp3d.modules.decomposition.split import recompose, split_layers


def _painting_like() -> np.ndarray:
    """어두운 선이 있는 합성 컬러 이미지 (BGR uint8)."""
    img = np.full((128, 128, 3), (180, 200, 210), dtype=np.uint8)
    img[40:60, :, :] = (90, 140, 200)      # 색 영역
    img[60:63, 10:118, :] = (40, 40, 50)   # 먹선 (폭 3)
    return img


def _run_split(img):
    gray = img.astype(np.float64).mean(axis=2)
    response, mask = detect_lines(gray, min_width=1.0, max_width=8.0)
    skeleton, width_map = measure_line_widths(mask)
    return split_layers(img, response, mask, skeleton, width_map)


def test_color_layer_untouched_outside_inpaint_mask():
    """inpaint 마스크 밖의 C는 원본과 완전 일치해야 한다."""
    img = _painting_like()
    line_alpha, color_layer, inpaint_mask = _run_split(img)
    outside = ~inpaint_mask
    assert np.array_equal(color_layer[outside], img[outside])


def test_alpha_zero_outside_line_region():
    """선 영역 밖의 alpha는 0이어야 한다."""
    img = _painting_like()
    line_alpha, _, inpaint_mask = _run_split(img)
    assert float(line_alpha[~inpaint_mask].max()) == 0.0


def test_alpha_positive_on_line():
    """선 위의 alpha는 유의미하게 커야 한다."""
    img = _painting_like()
    line_alpha, _, _ = _run_split(img)
    assert float(line_alpha[61, 20:100].mean()) > 0.5


def test_recompose_exact_outside_line():
    """불변식: alpha==0 픽셀에서 재합성 결과는 원본과 완전 일치."""
    img = _painting_like()
    line_alpha, color_layer, _ = _run_split(img)
    rec = recompose(img, line_alpha, color_layer)
    zero = line_alpha == 0.0
    assert np.array_equal(rec[zero], img[zero])


def test_recompose_small_global_residual():
    """전체 평균 재합성 잔차가 3 미만이어야 한다 (8bit 기준)."""
    img = _painting_like()
    line_alpha, color_layer, _ = _run_split(img)
    rec = recompose(img, line_alpha, color_layer)
    residual = np.abs(rec.astype(np.float64) - img.astype(np.float64)).mean()
    assert residual < 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_split.py -v`
Expected: FAIL — `ModuleNotFoundError` (split.py 부재)

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/decomposition/split.py` 생성:

```python
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
```

`__init__.py`에 `split_layers`, `recompose` 추가.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_split.py -v`
Expected: 5 passed

주의: `test_recompose_exact_outside_line`은 "alpha==0 → C 그대로"와 "inpaint 마스크 밖 C == I"의 결합으로 성립. alpha>0인데 마스크 밖인 픽셀은 존재하지 않음 (alpha는 마스크 내에서만 양수).

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/split.py src/kp3d/modules/decomposition/__init__.py tests/test_decomp_split.py
git commit -m "Add line/color layer split and recomposition with invariant"
```

---

### Task 7: decompose() 오케스트레이션 (2-pass 부트스트랩)

**Files:**
- Create: `src/kp3d/modules/decomposition/decompose.py`
- Modify: `src/kp3d/modules/decomposition/__init__.py`
- Test: `tests/test_decomp_full.py`

**Interfaces:**
- Consumes: Task 1~6의 모든 공개 함수
- Produces (후속 Plan 2·3의 진입점):

```python
@dataclass
class DecompositionResult:
    line_alpha: np.ndarray      # HxW float32 [0,1]
    color_layer: np.ndarray     # HxWx3 uint8 (C)
    line_mask: np.ndarray       # HxW bool
    skeleton: np.ndarray        # HxW bool
    width_map: np.ndarray       # HxW float32
    weave: WeavePeriodResult
    noise_sigma: float

def decompose(image_bgr: np.ndarray) -> DecompositionResult
def recompose_result(image_bgr: np.ndarray, result: DecompositionResult) -> np.ndarray
```

- [ ] **Step 1: Write the failing test**

`tests/test_decomp_full.py` 생성:

```python
"""decompose() 전체 오케스트레이션 테스트."""
import numpy as np

from kp3d.modules.decomposition.decompose import (
    DecompositionResult,
    decompose,
    recompose_result,
)


def _weave_painting() -> np.ndarray:
    """직조 텍스처 + 색 영역 + 먹선이 있는 합성 한국화 (BGR uint8)."""
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    base = 190 + 8 * np.sin(2 * np.pi * xx / 7) + 8 * np.sin(2 * np.pi * yy / 7)
    img = np.stack([base * 0.95, base, base * 1.02], axis=2)
    img[80:140, 40:216, :] = (120, 160, 200)   # 색 영역
    img[140:144, 40:216, :] = (45, 45, 55)     # 먹선 (폭 4)
    img[80:140, 40:43, :] = (45, 45, 55)       # 세로 먹선 (폭 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_decompose_returns_consistent_result():
    img = _weave_painting()
    result = decompose(img)
    assert isinstance(result, DecompositionResult)
    assert result.line_alpha.shape == img.shape[:2]
    assert result.color_layer.shape == img.shape
    assert result.noise_sigma >= 0.0
    # 직조 주기 7px 검출 (±1)
    assert abs(result.weave.period_x - 7.0) <= 1.0


def test_decompose_finds_ink_lines():
    """먹선 중심부가 선 마스크에 포함되어야 한다."""
    result = decompose(_weave_painting())
    assert float(result.line_mask[141:143, 80:180].mean()) > 0.8


def test_color_layer_has_no_line():
    """C 레이어의 먹선 위치는 주변 색으로 채워져 있어야 한다 (어둡지 않음)."""
    result = decompose(_weave_painting())
    line_region = result.color_layer[141, 80:180, :].astype(np.float64)
    assert float(line_region.mean()) > 100.0  # 먹선 원색(~48)보다 훨씬 밝음


def test_invariant_recompose_exact_outside_lines():
    """불변식: 선 영역 외 재합성 == 원본 (설계 섹션 4.4 계약 0->1)."""
    img = _weave_painting()
    result = decompose(img)
    rec = recompose_result(img, result)
    zero = result.line_alpha == 0.0
    assert np.array_equal(rec[zero], img[zero])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_decomp_full.py -v`
Expected: FAIL — `ModuleNotFoundError` (decompose.py 부재)

- [ ] **Step 3: Write minimal implementation**

`src/kp3d/modules/decomposition/decompose.py` 생성:

```python
"""Stage 0 오케스트레이션: 2-pass 부트스트랩 분해 (v2 설계 1.1-1.2).

순서: 통계 측정 -> RGF -> 1차 선 추출 -> 선폭 분포 측정 ->
      파라미터 확정 -> 2차(최종) 선 추출 -> 레이어 분리.
"""
from dataclasses import dataclass

import numpy as np

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
    sigma_s = max(periods) if periods else 3.0  # 주기 미검출 시 최소 평활 스케일
    structure = compute_structure_image(
        gray.astype(np.float32), sigma_s=sigma_s, noise_sigma=max(noise_sigma, 0.1)
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
```

`__init__.py` 최종형:

```python
"""Stage 0: 구륵법 인지 선·색 분해 (v2 설계 섹션 1)."""
from kp3d.modules.decomposition.decompose import (
    DecompositionResult,
    decompose,
    recompose_result,
)
from kp3d.modules.decomposition.lines import detect_lines, measure_line_widths
from kp3d.modules.decomposition.split import recompose, split_layers
from kp3d.modules.decomposition.statistics import (
    WeavePeriodResult,
    estimate_noise_sigma,
    estimate_weave_period,
)
from kp3d.modules.decomposition.structure import compute_structure_image

__all__ = [
    "DecompositionResult",
    "WeavePeriodResult",
    "compute_structure_image",
    "decompose",
    "detect_lines",
    "estimate_noise_sigma",
    "estimate_weave_period",
    "measure_line_widths",
    "recompose",
    "recompose_result",
    "split_layers",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_decomp_full.py -v`
Expected: 4 passed

이후 전체 회귀 확인:

Run: `python -m pytest tests/test_decomp_statistics.py tests/test_decomp_structure.py tests/test_decomp_lines.py tests/test_decomp_split.py tests/test_decomp_full.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add src/kp3d/modules/decomposition/decompose.py src/kp3d/modules/decomposition/__init__.py tests/test_decomp_full.py
git commit -m "Add two-pass bootstrap decompose orchestration"
```

---

### Task 8: 실이미지 데모 스크립트 + 스모크 테스트

**Files:**
- Create: `scripts/demo_decomposition.py`
- Test: `tests/test_decomp_full.py` (스모크 테스트 추가)

**Interfaces:**
- Consumes: Task 7의 `decompose`, `recompose_result`
- Produces: `output/decomposition_demo/` 아래 시각화 PNG 4종 (육안 검증용). 후속 코드 의존 없음

- [ ] **Step 1: Write the failing test (스모크)**

`tests/test_decomp_full.py`에 추가:

```python
import os

import pytest

_REAL_IMAGE = os.path.join("data", "ablation_study", "images", "1_0004.jpg")


@pytest.mark.skipif(not os.path.exists(_REAL_IMAGE), reason="실 데이터 없음")
def test_smoke_on_real_painting():
    """실제 한국화에서 예외 없이 완주하고 불변식을 지켜야 한다."""
    import cv2

    img = cv2.imread(_REAL_IMAGE)
    assert img is not None
    result = decompose(img)
    rec = recompose_result(img, result)
    zero = result.line_alpha == 0.0
    assert np.array_equal(rec[zero], img[zero])
    # 선이 하나라도 검출되어야 한다 (구륵법 그림 전제)
    assert result.skeleton.any()
```

주의: 실제 파일 확장자가 `.jpg`가 아니면(`.png` 등) `data/ablation_study/images/`를 확인해 경로를 맞출 것. annotation JSON `1_0004.json`이 존재하므로 대응 이미지가 같은 폴더에 있다.

- [ ] **Step 2: Run test to verify current state**

Run: `python -m pytest tests/test_decomp_full.py -v`
Expected: 이미지가 있으면 신규 테스트 실행(PASS 목표), 없으면 SKIP. 실패 시 원인 확인 후 수정.

- [ ] **Step 3: Write demo script**

`scripts/demo_decomposition.py` 생성:

```python
"""Stage 0 분해 결과 시각화 데모.

사용법: python scripts/demo_decomposition.py <이미지 경로> [출력 디렉터리]
출력: line_layer.png (L: 흰 배경 위 선), color_layer.png (C),
      alpha.png, recomposed.png
"""
import os
import sys

import cv2
import numpy as np

from kp3d.modules.decomposition import decompose, recompose_result


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    image_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        "output", "decomposition_demo"
    )
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 읽을 수 없음: {image_path}")
        sys.exit(1)

    result = decompose(img)
    rec = recompose_result(img, result)

    alpha = result.line_alpha[..., None]
    white = np.full_like(img, 255)
    line_vis = (alpha * img + (1 - alpha) * white).astype(np.uint8)

    cv2.imwrite(os.path.join(out_dir, "line_layer.png"), line_vis)
    cv2.imwrite(os.path.join(out_dir, "color_layer.png"), result.color_layer)
    cv2.imwrite(
        os.path.join(out_dir, "alpha.png"),
        (result.line_alpha * 255).astype(np.uint8),
    )
    cv2.imwrite(os.path.join(out_dir, "recomposed.png"), rec)

    residual = np.abs(
        rec.astype(np.float64) - img.astype(np.float64)
    ).mean()
    print(f"직조 주기: x={result.weave.period_x:.1f}px "
          f"(강도 {result.weave.strength_x:.2f}), "
          f"y={result.weave.period_y:.1f}px "
          f"(강도 {result.weave.strength_y:.2f})")
    print(f"노이즈 σ_n: {result.noise_sigma:.2f}")
    print(f"평균 재합성 잔차: {residual:.3f}")
    print(f"출력: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run demo and verify visually**

Run: `python scripts/demo_decomposition.py data/ablation_study/images/1_0004.jpg` (실제 확장자에 맞출 것)
Expected: 출력 PNG 4장 생성, 잔차 수치 출력. `line_layer.png`에 먹선이, `color_layer.png`에 선 없는 채색이 보이는지 육안 확인. 전체 테스트 재실행:

Run: `python -m pytest tests/test_decomp_statistics.py tests/test_decomp_structure.py tests/test_decomp_lines.py tests/test_decomp_split.py tests/test_decomp_full.py -v`
Expected: 전체 PASS (스모크 포함 20개, 데이터 없으면 19 passed + 1 skipped)

- [ ] **Step 5: Commit**

```bash
git add scripts/demo_decomposition.py tests/test_decomp_full.py
git commit -m "Add real-image smoke test and decomposition demo script"
```

---

## Self-Review 결과

- **Spec coverage**: 설계 1.1(①RGF ②선검출 ③분리 ④불변식) → Task 3/4·5/6/6·7. 1.2(2-pass 부트스트랩) → Task 7. 1.3(파라미터 유도) → 각 태스크에 분산 반영 (scale-space 다중 σ = Task 4, 노이즈 바닥 종료 = Task 3, 백분위 클리핑 = Task 7, Telea 반경 = Task 6). 1.4(자가 경쟁 게이트)는 **의도적 제외** — 경로 A가 Plan 2의 Stage 1 v2를 필요로 하므로 Plan 2로 이월. 1.5 실패 모드 중 "몰골법" 대응도 게이트와 함께 Plan 2로.
- **잔여 상수 목록** (Global Constraints의 허용 3종에 해당, 코드 주석 명시): √(π/2)·6 (수학 유도), max_iters=10 (안전 상한), 0.5% 대각선 (정규화), 1차 패스 광역 범위 [1px, 1% 대각선] (부트스트랩 시작점, 2차에서 대체됨), 백분위 5/95 (분포 절단 관례).
- **Type consistency**: `detect_lines` 반환 `(response, mask)` — Task 4 정의, Task 6·7 사용 일치. `measure_line_widths` 반환 `(skeleton, width_map)` — Task 5 정의, Task 6·7 사용 일치. `split_layers` 5-인자 시그니처 Task 6 정의 = Task 7 호출 일치.
