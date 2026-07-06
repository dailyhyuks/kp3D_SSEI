# KP3D v2 Stage 3: Structure-first SSEI 2.0 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙 §3 — 선 완성(Phase A: Euler spiral + Hungarian + taper 가상 종결) → 영역 분할 색 채움(Phase B: 선 위상 제약 exemplar pool + PatchMatch ANN + Wexler voting + multi-scale) → 재합성(Phase C: by construction 기계 검증)을 구현한다.

**Architecture:** 새 모듈 `src/kp3d/modules/ssei_v2/`에 endpoints → clothoid → matching → render(Phase A) → pool → patchmatch → fill(Phase B) → inpaint(Phase C) 순 8파일. Stage 0(decomposition)의 산출(line_alpha, skeleton, width_map, color_layer, noise_sigma)을 입력 계약으로 삼고, 재합성은 `decomposition.recompose`를 재사용(DRY)한다. Stage 2 v2(Plan 4)의 객체별 오케스트레이션은 범위 외 — 본 모듈은 (입력, occlusion_mask) 단위 API만 제공한다.

**Tech Stack:** numpy, scipy(ndimage/optimize/special), OpenCV(cv2). torch 금지 (training-free).

**스펙:** `docs/superpowers/specs/2026-07-05-kp3d-v2-line-color-decomposition-design.md` §3, §3.5, §3.6, §5.1

## Global Constraints

- **P-adapt: 튜닝 상수 0.** 허용 3범주 — ① 수학 유도(반 픽셀 0.5, 나이퀴스트 2샘플/px, 상관 길이 1/e, δ_safe=패치 반경 p//2, 이산 하한), ② 안전 상한(`_MAX_ITERS = 10`, 교차 기각 루프 ≤ 후보 쌍 수, 패치 상한 min(H,W)//4), ③ 정규화 규칙(백분위 [5,95]·P75(Wexler 관례), 창=선폭×2 시작·2차 모델 잔차 RMS ≤ 반 픽셀인 동안 ×2 배가 확장(모델 타당성 판정 — 결합 양자화 RMS √(1/6)≈0.41 < 0.5이므로 래스터 노이즈만으로는 정지하지 않음), 순위(경험 CDF) 정규화, 배가 확장 ×2). 모든 수치 리터럴 옆에 범주 근거 한국어 주석 필수.
- **스펙 명시 승계 상수:** d_max 초기값은 v1 해상도 적응 규칙(15/25/40)을 유지한다 (스펙 §3.5 명시) — 주석에 "스펙 §3.5 승계" 표기.
- **주석·도크스트링 전부 한국어** (식별자는 영어).
- **기하 관례(모듈 공통):** 좌표 (y, x). 접선 t=(dy, dx) 단위 벡터, θ=atan2(dy, dx). 좌법선 N=(t_x, −t_y). r''(s)=κ·N. 진행 방향 반전 시 κ 부호 반전. Endpoint.tangent는 획 바깥(끊김 안쪽) 방향.
- **모듈 경로** `src/kp3d/modules/ssei_v2/`, **테스트** `tests/test_ssei_*.py` (기존 test_wr2_* 관례: 합성 헬퍼 함수 + 불변식 단정 + 실데이터 skipif).
- **DRY:** 재합성은 `kp3d.modules.decomposition.recompose` 재사용. v1 코드(`occlusion/inpainting.py`) 수정 금지.
- TDD (실패 테스트 먼저), 태스크마다 커밋. 테스트 실행: `python -m pytest tests/test_ssei_<name>.py -q`

## File Structure

| 파일 | 책임 |
|---|---|
| `src/kp3d/modules/ssei_v2/__init__.py` | 공개 API 재수출 (Task 8) |
| `src/kp3d/modules/ssei_v2/endpoints.py` | 끊김 endpoint 검출·기하 서술자·획 추적·가시 획 통계 (Task 1) |
| `src/kp3d/modules/ssei_v2/clothoid.py` | G2 biclothoid 연결 곡선 + quintic Hermite 안전망 (Task 2) |
| `src/kp3d/modules/ssei_v2/matching.py` | 순위 정규화 비용 + Hungarian + taper 가상 종결 + 교차 기각 (Task 3) |
| `src/kp3d/modules/ssei_v2/render.py` | 곡선 렌더링 + `complete_lines` Phase A 진입점 (Task 4) |
| `src/kp3d/modules/ssei_v2/pool.py` | 조각 분할 + 선 위상 제약 exemplar pool + 차용/전역 fallback (Task 5) |
| `src/kp3d/modules/ssei_v2/patchmatch.py` | 패치 크기 유도(자기상관 1/e) + PatchMatch ANN (Task 6) |
| `src/kp3d/modules/ssei_v2/fill.py` | Wexler voting + multi-scale + C2 onion-peel + by-construction (Task 7) |
| `src/kp3d/modules/ssei_v2/inpaint.py` | Phase C 통합 `inpaint` + G2/by-construction 기계 검증 (Task 8) |
| `scripts/demo_ssei_v2.py` | 데모 (Task 9) |
| `tests/test_ssei_endpoints.py` … `tests/test_ssei_full.py` | 태스크별 테스트 |

---

### Task 1: endpoints.py — 끊김 endpoint 검출과 기하 서술자

**Files:**
- Create: `src/kp3d/modules/ssei_v2/__init__.py` (빈 파일 placeholder — Task 8에서 완성)
- Create: `src/kp3d/modules/ssei_v2/endpoints.py`
- Test: `tests/test_ssei_endpoints.py`

**Interfaces:**
- Consumes: 없음 (numpy/scipy만)
- Produces: `Endpoint(pos (2,)f64, tangent (2,)f64 단위·바깥 방향, curvature float(바깥 방향 기준 부호), width float, ink float, stroke_id int)`, `trace_stroke(skeleton, start, max_arc=None) -> (K,2) int64`, `detect_break_endpoints(skeleton, width_map, line_alpha, occlusion_mask) -> list[Endpoint]`, `stroke_statistics(skeleton, width_map, line_alpha) -> tuple[np.ndarray, np.ndarray, np.ndarray]` (κ², |Δw|상대, |Δink|상대)

- [ ] **Step 1: 실패 테스트 작성**

```python
"""endpoints.py 테스트: 끊김 endpoint 검출, 접선/곡률 서술자, 획 통계."""
import numpy as np

from kp3d.modules.ssei_v2.endpoints import (
    detect_break_endpoints,
    stroke_statistics,
    trace_stroke,
)


def _gap_scene(h=64, w=64, y=32, gap=(24, 40), width=3.0, ink=0.8):
    """수평 획(스켈레톤 y행) 가운데 gap 열 구간을 가림으로 지운 장면."""
    skeleton = np.zeros((h, w), dtype=bool)
    skeleton[y, 4:w - 4] = True
    occlusion = np.zeros((h, w), dtype=bool)
    occlusion[y - 6:y + 7, gap[0]:gap[1]] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, width, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, ink, 0.0).astype(np.float32)
    return skeleton, width_map, line_alpha, occlusion


def test_trace_stroke_orders_from_endpoint():
    skeleton, *_ = _gap_scene()
    pts = trace_stroke(skeleton, (32, 23))
    assert pts.shape[1] == 2
    assert tuple(pts[0]) == (32, 23)
    # 왼쪽 조각을 따라 x가 단조 감소
    assert np.all(np.diff(pts[:, 1]) == -1)


def test_detects_two_facing_endpoints():
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    assert len(eps) == 2
    xs = sorted(float(e.pos[1]) for e in eps)
    assert xs[0] == 23.0 and xs[1] == 40.0
    left = min(eps, key=lambda e: e.pos[1])
    right = max(eps, key=lambda e: e.pos[1])
    # 바깥 접선이 서로 마주본다
    chord = (right.pos - left.pos) / np.linalg.norm(right.pos - left.pos)
    assert float(np.dot(left.tangent, chord)) > 0.9
    assert float(np.dot(right.tangent, -chord)) > 0.9
    # 직선이므로 곡률 ~ 0, 서술자 값 전달
    assert abs(left.curvature) < 0.05
    assert abs(left.width - 3.0) < 0.5
    assert abs(left.ink - 0.8) < 1e-6
    assert left.stroke_id != right.stroke_id


def test_far_endpoint_not_reported():
    """가림에서 먼 자연 끝점(획의 양 끝)은 검출 제외."""
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    xs = {int(e.pos[1]) for e in eps}
    assert 4 not in xs and 59 not in xs


def test_curvature_sign_on_arc():
    """반지름 R 원호의 |κ| ≈ 1/R."""
    h = w = 96
    skeleton = np.zeros((h, w), dtype=bool)
    R, cy, cx = 30.0, 48, 48
    ang = np.linspace(np.pi * 0.1, np.pi * 0.9, 200)
    ys = np.round(cy - R * np.sin(ang)).astype(int)
    xs = np.round(cx + R * np.cos(ang)).astype(int)
    skeleton[ys, xs] = True
    occlusion = np.zeros((h, w), dtype=bool)
    occlusion[:, 44:52] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, 2.0, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, 1.0, 0.0).astype(np.float32)
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    assert len(eps) >= 2
    for e in eps:
        assert abs(abs(e.curvature) - 1.0 / R) < 0.5 / R


def test_stroke_statistics_nonempty_and_positive():
    skeleton, width_map, line_alpha, _ = _gap_scene()
    k2, dw, di = stroke_statistics(skeleton, width_map, line_alpha)
    assert k2.size > 0 and np.all(k2 >= 0.0)
    assert np.all(dw >= 0.0) and np.all(di >= 0.0)
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_endpoints.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/__init__.py`: 빈 파일 생성.

`src/kp3d/modules/ssei_v2/endpoints.py`:

```python
"""끊김 endpoint 검출과 기하 서술자 (스펙 §3.2 ①).

기하 관례: 좌표 (y,x), 접선 t=(dy,dx), θ=atan2(dy,dx), 좌법선 N=(t_x,−t_y),
r''=κN. 진행 방향 반전 시 κ 부호 반전. Endpoint.tangent는 획 바깥 방향.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, distance_transform_edt, label

# 접선·곡률 추정 창 시작값 = 국소 선폭 × 2 — 모델 타당성 유지 동안 ×2 배가 확장 (정규화 규칙)
_GEOM_WINDOW_WIDTHS = 2.0
# 이산 격자 위치 불확실성 반 픽셀 — 2차 모델 잔차 RMS 상한. 좌표별 균일 ±0.5 양자화의
# 결합 RMS는 √(1/6)≈0.41 < 0.5 이므로 래스터 노이즈만으로는 정지하지 않는다 (수학 유도)
_HALF_PIXEL = 0.5
# 8-근방 구조 원소 — 이산 위상 정의 (수학 유도)
_N8 = np.ones((3, 3), dtype=np.int64)
# 2차 최소제곱의 유효 최소 표본 수 3 — 미지수 3개 (수학 유도)
_MIN_QUAD_PTS = 3


@dataclass
class Endpoint:
    """끊김 endpoint 서술자."""

    pos: np.ndarray       # (2,) float64 (y, x)
    tangent: np.ndarray   # (2,) float64 단위, 획 바깥(끊김) 방향
    curvature: float      # 바깥 방향 기준 부호 곡률 [1/px]
    width: float          # 국소 선폭 [px]
    ink: float            # 국소 평균 알파 (0~1)
    stroke_id: int        # 스켈레톤 8-연결 성분 라벨


def trace_stroke(skeleton: np.ndarray, start: tuple[int, int],
                 max_arc: float | None = None) -> np.ndarray:
    """끝점 start에서 스켈레톤을 따라 분기/끝까지 순서대로 걷는다.

    Returns:
        (K,2) int64 — start 포함 순서열. 분기점(이웃 2+)에서 중단.
    """
    sk = np.asarray(skeleton, dtype=bool)
    h, w = sk.shape
    pts = [np.array(start, dtype=np.int64)]
    visited = {tuple(start)}
    arc = 0.0
    cur = pts[0]
    while True:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                y, x = int(cur[0]) + dy, int(cur[1]) + dx
                if 0 <= y < h and 0 <= x < w and sk[y, x] and (y, x) not in visited:
                    nbrs.append((y, x))
        if len(nbrs) == 2 and (abs(nbrs[0][0] - nbrs[1][0]) <= 1
                               and abs(nbrs[0][1] - nbrs[1][1]) <= 1):
            # 계단 래스터: 두 이웃이 서로 8-인접 — L1 가까운 이웃 먼저 (수학 유도)
            nbrs.sort(key=lambda q: abs(q[0] - int(cur[0])) + abs(q[1] - int(cur[1])))
            nbrs = nbrs[:1]
        if len(nbrs) != 1:
            break  # 끝 또는 진짜 분기 — 이후 순서 정의 불가
        nxt = np.array(nbrs[0], dtype=np.int64)
        arc += float(np.hypot(*(nxt - cur).astype(np.float64)))
        if max_arc is not None and arc > max_arc:
            break
        visited.add(nbrs[0])
        pts.append(nxt)
        cur = nxt
    return np.asarray(pts, dtype=np.int64)


def _fit_geometry(pts: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """끝점부터의 순서열에 s-매개 2차 최소제곱 → (바깥 단위 접선, 바깥 기준 κ, 잔차 RMS)."""
    p = pts.astype(np.float64)
    if len(p) < 2:
        return None
    d = np.diff(p, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    deg = 2 if len(p) >= _MIN_QUAD_PTS else 1
    cy = np.polyfit(s, p[:, 0], deg)
    cx = np.polyfit(s, p[:, 1], deg)
    ry = p[:, 0] - np.polyval(cy, s)
    rx = p[:, 1] - np.polyval(cx, s)
    rms = float(np.sqrt(np.mean(ry ** 2 + rx ** 2)))  # 2D 결합 잔차 RMS (수학 유도)
    dy = float(np.polyval(np.polyder(cy), 0.0))
    dx = float(np.polyval(np.polyder(cx), 0.0))
    n = float(np.hypot(dy, dx))
    if n == 0.0:
        return None
    if deg == 2:
        ddy = float(np.polyval(np.polyder(cy, 2), 0.0))
        ddx = float(np.polyval(np.polyder(cx, 2), 0.0))
        kappa_in = (dx * ddy - dy * ddx) / n ** 3  # N=(t_x,−t_y) 관례의 κ
    else:
        kappa_in = 0.0
    # s는 획 안쪽으로 증가 → 바깥 접선 = −t_in, 방향 반전으로 κ 부호 반전
    return np.array([-dy / n, -dx / n]), float(-kappa_in), rms


def detect_break_endpoints(skeleton: np.ndarray, width_map: np.ndarray,
                           line_alpha: np.ndarray,
                           occlusion_mask: np.ndarray) -> list[Endpoint]:
    """가림 경계에 인접한 스켈레톤 끝점을 서술자와 함께 반환."""
    sk = np.asarray(skeleton, dtype=bool)
    occ = np.asarray(occlusion_mask, dtype=bool)
    nb = convolve(sk.astype(np.int64), _N8, mode="constant") - sk.astype(np.int64)
    ends = [tuple(int(v) for v in p) for p in np.argwhere(sk & (nb == 1))]
    # nb==2이지만 두 이웃이 서로 8-인접하면 위상적 '팁' (대각 래스터 절단부) — 수학 유도
    h_, w_ = sk.shape
    for y, x in np.argwhere(sk & (nb == 2)):
        ns = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if not (dy == 0 and dx == 0)
              and 0 <= y + dy < h_ and 0 <= x + dx < w_ and sk[y + dy, x + dx]]
        (ay, ax), (by, bx) = ns
        if abs(int(ay) - int(by)) <= 1 and abs(int(ax) - int(bx)) <= 1:
            ends.append((int(y), int(x)))
    if not ends:
        return []
    dist_occ = distance_transform_edt(~occ)
    labels, _ = label(sk, structure=_N8)
    out: list[Endpoint] = []
    for y, x in ends:
        w_here = max(float(width_map[y, x]), 1.0)  # 스켈레톤 픽셀 폭 하한 1px — 이산 하한 (수학 유도)
        if float(dist_occ[y, x]) > w_here:
            continue  # 가림 경계 인접 조건 — 폭 지도에서 유도 (P-adapt)
        # 창 = 선폭×2 시작; 2차 모델 잔차 RMS ≤ 반 픽셀인 동안 ×2 배가 확장 (모델 타당성 판정)
        arc = _GEOM_WINDOW_WIDTHS * w_here
        geom = None
        best_pts = None
        prev_pts = 0
        while True:
            pts = trace_stroke(sk, (int(y), int(x)), max_arc=arc)
            fit = _fit_geometry(pts)
            if fit is None:
                break  # 기하 실패 — 직전 창 결과 유지
            if fit[2] > _HALF_PIXEL and geom is not None:
                break  # 모델 붕괴(잔차 RMS > 표본화 한계) — 직전 창 채택
            geom, best_pts = fit, pts
            if len(pts) == prev_pts:
                break  # 획 소진
            prev_pts = len(pts)
            arc *= 2.0  # 배가 확장 (정규화 규칙)
        if geom is None:
            continue
        tangent, kappa = geom[0], geom[1]
        widths = width_map[best_pts[:, 0], best_pts[:, 1]]
        pos_w = widths[widths > 0]
        out.append(Endpoint(
            pos=np.array([float(y), float(x)]),
            tangent=tangent,
            curvature=kappa,
            width=float(np.median(pos_w)) if pos_w.size else 1.0,
            ink=float(np.mean(line_alpha[best_pts[:, 0], best_pts[:, 1]])),
            stroke_id=int(labels[y, x]),
        ))
    return out


def stroke_statistics(skeleton: np.ndarray, width_map: np.ndarray,
                      line_alpha: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가시 획의 자연 변동 통계 (κ², |Δw| 상대, |Δink| 상대) — 종결 비용 보정용.

    각 성분을 임의 끝점에서 전체 추적해 s-매개 미분으로 κ를 추정한다.
    끝점 없는 성분(고리)·표본 부족 성분은 건너뛴다 (통계는 있는 만큼만).
    """
    sk = np.asarray(skeleton, dtype=bool)
    nb = convolve(sk.astype(np.int64), _N8, mode="constant") - sk.astype(np.int64)
    labels, n_lab = label(sk, structure=_N8)
    k2_all: list[np.ndarray] = []
    dw_all: list[np.ndarray] = []
    di_all: list[np.ndarray] = []
    for lab_id in range(1, n_lab + 1):
        comp_ends = np.argwhere((labels == lab_id) & (nb == 1))
        if comp_ends.size == 0:
            continue
        pts = trace_stroke(sk, tuple(int(v) for v in comp_ends[0]))
        if len(pts) < 2 * _MIN_QUAD_PTS:  # 2차 미분에 필요한 최소 지지 (수학 유도)
            continue
        p = pts.astype(np.float64)
        d = np.diff(p, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
        dy = np.gradient(p[:, 0], s)
        dx = np.gradient(p[:, 1], s)
        ddy = np.gradient(dy, s)
        ddx = np.gradient(dx, s)
        norm = np.hypot(dy, dx)
        norm = np.where(norm > 0, norm, 1.0)
        kappa = (dx * ddy - dy * ddx) / norm ** 3
        k2_all.append(kappa ** 2)
        wv = np.maximum(width_map[pts[:, 0], pts[:, 1]].astype(np.float64), 1.0)
        av = np.clip(line_alpha[pts[:, 0], pts[:, 1]].astype(np.float64), 0.0, 1.0)
        dw_all.append(np.abs(np.diff(wv)) / (wv[:-1] + wv[1:]))
        di_all.append(np.abs(np.diff(av)) / np.maximum(av[:-1] + av[1:], 1.0))
    cat = (lambda lst: np.concatenate(lst) if lst else np.zeros(0))
    return cat(k2_all), cat(dw_all), cat(di_all)
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_endpoints.py -q` / Expected: 5 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/__init__.py src/kp3d/modules/ssei_v2/endpoints.py tests/test_ssei_endpoints.py
git commit -m "feat(ssei_v2): break endpoint detection with geometric descriptors"
```

### Task 2: clothoid.py — G2 연결 곡선 (biclothoid + quintic Hermite 안전망)

**Files:**
- Create: `src/kp3d/modules/ssei_v2/clothoid.py`
- Test: `tests/test_ssei_clothoid.py`

**Interfaces:**
- Consumes: 없음 (numpy/scipy만). Task 1과 동일한 기하 관례.
- Produces: `ConnectionCurve(points (N,2)f64, tangents (N,2)f64, curvatures (N,)f64, arc_length float, bending_energy float, is_clothoid bool)`, `connect_g2(p0, t0, k0, p1, t1, k1) -> ConnectionCurve` — 접선·곡률은 **진행 방향(p0→p1) 기준**. Task 3은 `connect_g2(e_i.pos, e_i.tangent, e_i.curvature, e_j.pos, -e_j.tangent, -e_j.curvature)` 로 호출한다 (Endpoint.tangent는 바깥 방향이므로 도착측은 부호 반전).

- [ ] **Step 1: 실패 테스트 작성**

```python
"""clothoid.py 테스트: G2 경계 조건, 굽힘 에너지, 안전망 강등."""
import numpy as np

import kp3d.modules.ssei_v2.clothoid as cl
from kp3d.modules.ssei_v2.clothoid import connect_g2


def test_straight_line_zero_energy_and_clothoid():
    p0, p1 = np.array([10.0, 10.0]), np.array([10.0, 40.0])
    t = np.array([0.0, 1.0])
    c = connect_g2(p0, t, 0.0, p1, t, 0.0)
    assert c.is_clothoid
    assert c.bending_energy < 1e-8
    assert np.linalg.norm(c.points[0] - p0) < 0.5
    assert np.linalg.norm(c.points[-1] - p1) < 0.5
    assert abs(c.arc_length - 30.0) < 0.5


def test_g2_boundary_conditions():
    p0, p1 = np.array([20.0, 10.0]), np.array([20.0, 40.0])
    t0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    t1 = np.array([0.0, 1.0])
    k0, k1 = 0.05, -0.03
    c = connect_g2(p0, t0, k0, p1, t1, k1)
    assert float(np.dot(c.tangents[0], t0)) > 0.999
    assert float(np.dot(c.tangents[-1], t1)) > 0.999
    assert abs(float(c.curvatures[0]) - k0) < 1e-6
    assert abs(float(c.curvatures[-1]) - k1) < 1e-6
    assert np.linalg.norm(c.points[0] - p0) < 0.5
    assert np.linalg.norm(c.points[-1] - p1) < 0.5


def test_solver_failure_falls_back_to_quintic(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("강제 실패")

    monkeypatch.setattr(cl, "least_squares", _boom)
    p0, p1 = np.array([20.0, 10.0]), np.array([20.0, 40.0])
    t0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    t1 = np.array([0.0, 1.0])
    c = connect_g2(p0, t0, 0.05, p1, t1, -0.03)
    assert not c.is_clothoid
    # 안전망(quintic 폐형)도 G2 경계 조건은 해석적으로 정확히 유지
    assert float(np.dot(c.tangents[0], t0)) > 0.999
    assert float(np.dot(c.tangents[-1], t1)) > 0.999
    assert abs(float(c.curvatures[0]) - 0.05) < 1e-6
    assert abs(float(c.curvatures[-1]) + 0.03) < 1e-6


def test_degenerate_short_gap():
    p0, p1 = np.array([5.0, 5.0]), np.array([5.0, 5.5])
    t = np.array([0.0, 1.0])
    c = connect_g2(p0, t, 0.0, p1, t, 0.0)
    assert c.points.shape[0] >= 2
    assert np.allclose(c.points[0], p0) and np.allclose(c.points[-1], p1)
    assert c.bending_energy == 0.0
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_clothoid.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/clothoid.py`:

```python
"""G2 연결 곡선 — biclothoid(주 해) + quintic Hermite(안전망) (스펙 §3.2 ③).

기하 관례: 좌표 (y,x), 접선 t=(dy,dx), θ=atan2(dy,dx), 좌법선 N=(t_x,−t_y),
r''=κN. connect_g2의 접선·곡률은 진행 방향(p0→p1) 기준이다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

# 위치 허용 오차 반 픽셀 — 이산 격자 표본화 한계 (수학 유도)
_HALF_PIXEL = 0.5
# 곡선 표본 밀도 2 샘플/px — 나이퀴스트 (수학 유도)
_SAMPLES_PER_PX = 2.0
# 이산 격자 최소 유의 현길이 1px — 이산 하한 (수학 유도)
_MIN_CHORD = 1.0


@dataclass
class ConnectionCurve:
    """G2 연결 곡선 표본열."""

    points: np.ndarray      # (N,2) float64 (y,x)
    tangents: np.ndarray    # (N,2) float64 단위 접선 (진행 방향)
    curvatures: np.ndarray  # (N,) float64 부호 곡률 [1/px]
    arc_length: float       # 총 호장 [px]
    bending_energy: float   # ∫ κ² ds
    is_clothoid: bool       # biclothoid 수렴 여부 (False = quintic 안전망)


def _left_normal(t: np.ndarray) -> np.ndarray:
    """좌법선 N=(t_x, −t_y) — 모듈 기하 관례."""
    return np.array([t[1], -t[0]], dtype=np.float64)


def _wrap_angle(a: float) -> float:
    """각도를 (−π, π]로 정규화 — 순수 수학."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _n_samples(length: float) -> int:
    """호장 length의 나이퀴스트 표본 수 — 최소 4 (3차 형상 식별 하한, 수학 유도)."""
    return max(int(np.ceil(length * _SAMPLES_PER_PX)) + 1, 4)


def _energy(kappa: np.ndarray, s: np.ndarray) -> float:
    """굽힘 에너지 ∫ κ² ds (사다리꼴 적분)."""
    return float(cumulative_trapezoid(kappa ** 2, s, initial=0.0)[-1])


def _quintic(p0, t0, k0, p1, t1, k1) -> ConnectionCurve:
    """quintic Hermite G2 폐형 — v=d(현길이) 매개, 가속 a=0, r''(u)=d²κN."""
    d = float(np.linalg.norm(p1 - p0))
    n = _n_samples(d)
    u = np.linspace(0.0, 1.0, n)
    u2, u3, u4, u5 = u ** 2, u ** 3, u ** 4, u ** 5
    # quintic Hermite 기저 (G2 보간의 폐형 — 순수 수학)
    H = [1 - 10 * u3 + 15 * u4 - 6 * u5,
         u - 6 * u3 + 8 * u4 - 3 * u5,
         0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5,
         0.5 * u3 - u4 + 0.5 * u5,
         -4 * u3 + 7 * u4 - 3 * u5,
         10 * u3 - 15 * u4 + 6 * u5]
    dH = [-30 * u2 + 60 * u3 - 30 * u4,
          1 - 18 * u2 + 32 * u3 - 15 * u4,
          u - 4.5 * u2 + 6 * u3 - 2.5 * u4,
          1.5 * u2 - 4 * u3 + 2.5 * u4,
          -12 * u2 + 28 * u3 - 15 * u4,
          30 * u2 - 60 * u3 + 30 * u4]
    ddH = [-60 * u + 180 * u2 - 120 * u3,
           -36 * u + 96 * u2 - 60 * u3,
           1 - 9 * u + 18 * u2 - 10 * u3,
           3 * u - 12 * u2 + 10 * u3,
           -24 * u + 84 * u2 - 60 * u3,
           60 * u - 180 * u2 + 120 * u3]
    ctrl = [p0, d * t0, d * d * k0 * _left_normal(t0),
            d * d * k1 * _left_normal(t1), d * t1, p1]
    pts = sum(np.outer(h, c) for h, c in zip(H, ctrl))
    d1 = sum(np.outer(h, c) for h, c in zip(dH, ctrl))
    d2 = sum(np.outer(h, c) for h, c in zip(ddH, ctrl))
    speed = np.hypot(d1[:, 0], d1[:, 1])
    speed = np.where(speed > 0.0, speed, 1.0)
    tangents = d1 / speed[:, None]
    # κ = (dx·ddy − dy·ddx)/|r'|³ — N=(t_x,−t_y) 관례 (Task 1과 동일)
    kappa = (d1[:, 1] * d2[:, 0] - d1[:, 0] * d2[:, 1]) / speed ** 3
    s = cumulative_trapezoid(speed, u, initial=0.0)
    return ConnectionCurve(points=pts, tangents=tangents, curvatures=kappa,
                           arc_length=float(s[-1]),
                           bending_energy=_energy(kappa, s), is_clothoid=False)


def _biclothoid_geometry(theta0: float, k0: float, km: float, k1: float,
                         L1: float, L2: float, p0: np.ndarray):
    """구간 선형 κ 곡선(biclothoid)을 수치 적분 → (pts, tangents, kappa, s, theta)."""
    L = L1 + L2
    n = _n_samples(L)
    s = np.linspace(0.0, L, n)
    kappa = np.where(s <= L1,
                     k0 + (km - k0) * s / L1,
                     km + (k1 - km) * (s - L1) / L2)
    theta = theta0 + cumulative_trapezoid(kappa, s, initial=0.0)
    dy, dx = np.sin(theta), np.cos(theta)
    y = p0[0] + cumulative_trapezoid(dy, s, initial=0.0)
    x = p0[1] + cumulative_trapezoid(dx, s, initial=0.0)
    pts = np.stack([y, x], axis=1)
    tangents = np.stack([dy, dx], axis=1)
    return pts, tangents, kappa, s, theta


def connect_g2(p0, t0, k0, p1, t1, k1) -> ConnectionCurve:
    """G2 경계 조건(위치·접선·곡률)을 만족하는 연결 곡선.

    biclothoid(미지수: log L1, log L2, κm)를 least_squares로 풀고,
    잔차가 반 픽셀 허용 오차를 넘으면 quintic Hermite 폐형으로 강등한다.

    Returns:
        ConnectionCurve — is_clothoid=True면 biclothoid, False면 안전망.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64) / float(np.linalg.norm(t0))
    t1 = np.asarray(t1, dtype=np.float64) / float(np.linalg.norm(t1))
    k0, k1 = float(k0), float(k1)
    d = float(np.linalg.norm(p1 - p0))
    if d < _MIN_CHORD:
        # 표본화 한계 미만 간격 — 2점 직선 퇴화 (이산 하한)
        return ConnectionCurve(points=np.stack([p0, p1]),
                               tangents=np.tile(t0, (2, 1)),
                               curvatures=np.zeros(2), arc_length=d,
                               bending_energy=0.0, is_clothoid=False)
    quintic = _quintic(p0, t0, k0, p1, t1, k1)
    theta0 = float(np.arctan2(t0[0], t0[1]))
    theta1 = float(np.arctan2(t1[0], t1[1]))
    L_q = max(quintic.arc_length, d)
    k_mid = float(quintic.curvatures[len(quintic.curvatures) // 2])

    def residual(u: np.ndarray) -> np.ndarray:
        L1, L2 = float(np.exp(u[0])), float(np.exp(u[1]))
        pts, _, _, _, theta = _biclothoid_geometry(
            theta0, k0, float(u[2]), k1, L1, L2, p0)
        # 각도 잔차에 호장을 곱해 위치 잔차와 단위(px)를 일치 (순수 수학)
        return np.array([pts[-1, 0] - p1[0], pts[-1, 1] - p1[1],
                         _wrap_angle(float(theta[-1]) - theta1) * (L1 + L2)])

    # 초기해: quintic의 호장 절반씩 + 중간 곡률 (해석해에서 유도)
    u0 = np.array([np.log(L_q / 2.0), np.log(L_q / 2.0), k_mid])
    try:
        sol = least_squares(residual, u0)
    except Exception:
        return quintic
    L1, L2 = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    km = float(sol.x[2])
    pts, tangents, kappa, s, theta = _biclothoid_geometry(
        theta0, k0, km, k1, L1, L2, p0)
    pos_err = float(np.linalg.norm(pts[-1] - p1))
    ang_err = abs(_wrap_angle(float(theta[-1]) - theta1))
    # 수락: 위치 ≤ 반 픽셀, 각도 ≤ 반 픽셀/호장 — 표본화 한계 (수학 유도)
    if pos_err > _HALF_PIXEL or ang_err > _HALF_PIXEL / (L1 + L2):
        return quintic
    return ConnectionCurve(points=pts, tangents=tangents, curvatures=kappa,
                           arc_length=float(s[-1]),
                           bending_energy=_energy(kappa, s), is_clothoid=True)
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_clothoid.py -q` / Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/clothoid.py tests/test_ssei_clothoid.py
git commit -m "feat(ssei_v2): G2 connection curves (biclothoid + quintic Hermite fallback)"
```

### Task 3: matching.py — 순위 정규화 비용 + Hungarian + 종결 + 교차 기각

**Files:**
- Create: `src/kp3d/modules/ssei_v2/matching.py`
- Test: `tests/test_ssei_matching.py`

**Interfaces:**
- Consumes: Task 1 `Endpoint`, `stroke_statistics`; Task 2 `connect_g2`, `ConnectionCurve`
- Produces: `Connection(i int, j int, curve ConnectionCurve, widths (float,float), inks (float,float))` — curve.points[0]=endpoints[i].pos, points[-1]≈endpoints[j].pos; `MatchResult(connections list[Connection], terminations list[int])`; `match_endpoints(endpoints, skeleton, width_map, line_alpha, occlusion_mask) -> MatchResult`

- [ ] **Step 1: 실패 테스트 작성**

```python
"""matching.py 테스트: 후보 생성, 순위 비용, Hungarian 종결, 상호 채택."""
import numpy as np

from kp3d.modules.ssei_v2.endpoints import detect_break_endpoints
from kp3d.modules.ssei_v2.matching import match_endpoints


def _scene(strokes, occ_box, h=80, w=64, width=3.0, ink=0.8):
    """strokes: (y, x0, x1) 수평 획 목록. occ_box: (y0, y1, x0, x1) 가림 상자."""
    skeleton = np.zeros((h, w), dtype=bool)
    for y, x0, x1 in strokes:
        skeleton[y, x0:x1] = True
    occlusion = np.zeros((h, w), dtype=bool)
    y0, y1, x0, x1 = occ_box
    occlusion[y0:y1, x0:x1] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, width, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, ink, 0.0).astype(np.float32)
    return skeleton, width_map, line_alpha, occlusion


def _match(skeleton, width_map, line_alpha, occlusion):
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    return eps, match_endpoints(eps, skeleton, width_map, line_alpha, occlusion)


def test_no_endpoints_empty():
    args = _scene([], (26, 41, 24, 40))
    eps, res = _match(*args)
    assert eps == [] and res.connections == [] and res.terminations == []


def test_single_endpoint_terminates():
    args = _scene([(32, 4, 40)], (26, 41, 24, 41))
    eps, res = _match(*args)
    assert len(eps) == 1
    assert res.connections == [] and res.terminations == [0]


def test_straight_gap_connects():
    args = _scene([(32, 4, 60)], (26, 41, 24, 41))
    eps, res = _match(*args)
    assert len(eps) == 2
    assert len(res.connections) == 1 and res.terminations == []
    c = res.connections[0]
    assert float(np.linalg.norm(c.curve.points[0] - eps[c.i].pos)) < 0.5
    assert float(np.linalg.norm(c.curve.points[-1] - eps[c.j].pos)) < 0.5
    assert c.widths == (eps[c.i].width, eps[c.j].width)


def test_prefers_geometric_continuation():
    # 왼쪽 획(y=32) ↔ 정렬된 오른쪽 획(y=32) 연결, 어긋난 획(y=52)은 종결
    args = _scene([(32, 4, 60), (52, 41, 60)], (26, 56, 24, 41))
    eps, res = _match(*args)
    assert len(eps) == 3
    assert len(res.connections) == 1
    c = res.connections[0]
    ys = {float(eps[c.i].pos[0]), float(eps[c.j].pos[0])}
    assert ys == {32.0}
    assert len(res.terminations) == 1
    assert float(eps[res.terminations[0]].pos[0]) == 52.0
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_matching.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/matching.py`:

```python
"""Endpoint 매칭 — 순위 정규화 비용 + Hungarian + 종결 가상 노드 + 교차 기각 (스펙 §3.2 ②④).

임계값 없음: 연결 비용은 후보 분포의 경험 CDF 순위, 종결 비용은 가시 획
자연 변동 P95의 순위. 동률이면 연결을 우선한다(순위 정의의 부등호 방향으로
구현 — 정규화 규칙).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .clothoid import ConnectionCurve, connect_g2
from .endpoints import Endpoint, stroke_statistics

# 표본 밀도 2/px — 나이퀴스트 (수학 유도)
_SAMPLES_PER_PX = 2.0
# 자연 변동 상위 백분위 — 백분위 관례 [5,95] (정규화 규칙)
_P_HIGH = 95.0


@dataclass
class Connection:
    """채택된 endpoint 연결."""

    i: int
    j: int
    curve: ConnectionCurve   # points[0]=endpoints[i].pos, points[-1]≈endpoints[j].pos
    widths: tuple[float, float]  # (w_i, w_j) — 렌더링 선형 보간용
    inks: tuple[float, float]    # (ink_i, ink_j)


@dataclass
class MatchResult:
    """매칭 결과: 연결 목록 + 종결(taper) endpoint 인덱스."""

    connections: list[Connection]
    terminations: list[int]


def _rank_less(values: np.ndarray, x: float) -> float:
    """경험 CDF 순위 (strict <) — 연결 후보 비용용 (정규화 규칙)."""
    return float(np.count_nonzero(values < x)) / float(len(values))


def _rank_leq(values: np.ndarray, x: float) -> float:
    """경험 CDF 순위 (≤) — 종결 기준용. 동률이면 연결 우선 관례 (정규화 규칙)."""
    return float(np.count_nonzero(values <= x)) / float(len(values))


def _chord_crosses_occlusion(p0: np.ndarray, p1: np.ndarray,
                             occ: np.ndarray) -> bool:
    """두 endpoint 현(직선)이 가림 영역을 통과하는지 — 구조 제약."""
    d = float(np.linalg.norm(p1 - p0))
    n = max(int(np.ceil(d * _SAMPLES_PER_PX)) + 1, 2)
    ys = np.round(np.linspace(p0[0], p1[0], n)).astype(int)
    xs = np.round(np.linspace(p0[1], p1[1], n)).astype(int)
    h, w = occ.shape
    ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    return bool(np.any(occ[ys[ok], xs[ok]]))


def _assign(E: int, keys: list[tuple[int, int]], cost: dict, T: float,
            banned: set) -> tuple[list[tuple[int, int]], list[int]]:
    """2E×2E Hungarian — 대각 가상 노드가 종결. 상호 배정 쌍만 연결로 채택."""
    M = np.full((2 * E, 2 * E), np.inf)
    for (i, j) in keys:
        c = np.inf if (i, j) in banned else cost[(i, j)]
        M[i, j] = M[j, i] = c
    for i in range(E):
        M[i, E + i] = T   # 실 → 자신의 가상 상대(종결)
        M[E + i, i] = T
    M[E:, E:] = 0.0       # 가상-가상: 미사용 가상 노드 흡수 (비용 0)
    rows, cols = linear_sum_assignment(M)
    col_of = {int(r): int(c) for r, c in zip(rows, cols)}
    pairs = [(i, col_of[i]) for i in range(E)
             if col_of[i] < E and col_of.get(col_of[i]) == i and i < col_of[i]]
    matched = {k for p in pairs for k in p}
    terms = [i for i in range(E) if i not in matched]
    return pairs, terms


def _trace_pixels(curve: ConnectionCurve) -> set[tuple[int, int]]:
    """곡선 표본을 정수 픽셀 자취로 (표본 밀도 2/px — 나이퀴스트)."""
    pts = np.round(curve.points).astype(int)
    return set(map(tuple, pts))


def _endpoint_disk(e: Endpoint) -> set[tuple[int, int]]:
    """endpoint 주변 반경 = 국소 선폭 디스크 — 자기 획 근방 제외 창 (폭에서 유도)."""
    r = int(np.ceil(max(e.width, 1.0)))
    y0, x0 = int(round(e.pos[0])), int(round(e.pos[1]))
    return {(y0 + dy, x0 + dx)
            for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r}


def _find_crossings(pairs, curves, cost, endpoints, sk, occ):
    """곡선 자취가 가시 스켈레톤 또는 타 곡선과 겹치는 쌍을 기각 대상으로."""
    h, w = sk.shape
    traces, bad = {}, set()
    for k in pairs:
        i, j = k
        excl = _endpoint_disk(endpoints[i]) | _endpoint_disk(endpoints[j])
        traces[k] = _trace_pixels(curves[k]) - excl
    for k in pairs:
        for (y, x) in traces[k]:
            if 0 <= y < h and 0 <= x < w and sk[y, x] and not occ[y, x]:
                bad.add(k)  # 가시 스켈레톤 위를 지나는 곡선 기각
                break
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            ka, kb = pairs[a], pairs[b]
            if traces[ka] & traces[kb]:
                # 곡선끼리 겹치면 비용 높은 쪽 기각 (순위 비교 — 임계값 없음)
                bad.add(ka if cost[ka] >= cost[kb] else kb)
    return bad


def match_endpoints(endpoints: list[Endpoint], skeleton: np.ndarray,
                    width_map: np.ndarray, line_alpha: np.ndarray,
                    occlusion_mask: np.ndarray) -> MatchResult:
    """끊김 endpoint를 Hungarian으로 매칭한다 (연결 또는 종결).

    비용 성분: (굽힘 에너지/호장, |Δw| 상대, |Δink| 상대)의 순위 평균.
    종결 비용 T: 가시 획 자연 변동 통계(stroke_statistics) P95의 순위
    (통계가 비면 후보 자기 분포의 P95로 대체).
    """
    E = len(endpoints)
    if E == 0:
        return MatchResult([], [])
    occ = np.asarray(occlusion_mask, dtype=bool)
    sk = np.asarray(skeleton, dtype=bool)
    # 후보 생성 — 구조 제약: 서로 다른 획 + 현이 가림을 통과 (임계값 없음)
    curves: dict[tuple[int, int], ConnectionCurve] = {}
    for i in range(E):
        for j in range(i + 1, E):
            a, b = endpoints[i], endpoints[j]
            if a.stroke_id == b.stroke_id:
                continue  # 같은 성분 자기 연결(고리) 금지 — 위상 제약
            if not _chord_crosses_occlusion(a.pos, b.pos, occ):
                continue
            # 도착측 접선·곡률은 진행 방향 기준으로 부호 반전 (Task 2 규약)
            curves[(i, j)] = connect_g2(a.pos, a.tangent, a.curvature,
                                        b.pos, -b.tangent, -b.curvature)
    if not curves:
        return MatchResult([], list(range(E)))
    keys = list(curves)
    bend = np.array([curves[k].bending_energy
                     / max(curves[k].arc_length, 1.0) for k in keys])
    dwv = np.array([abs(endpoints[i].width - endpoints[j].width)
                    / (endpoints[i].width + endpoints[j].width)
                    for i, j in keys])
    div = np.array([abs(endpoints[i].ink - endpoints[j].ink)
                    / max(endpoints[i].ink + endpoints[j].ink, 1.0)
                    for i, j in keys])
    cost = {k: (_rank_less(bend, float(bend[m])) + _rank_less(dwv, float(dwv[m]))
                + _rank_less(div, float(div[m]))) / 3.0
            for m, k in enumerate(keys)}
    k2s, dws, dis = stroke_statistics(sk, width_map, line_alpha)
    T = 0.0
    for vals, stats in ((bend, k2s), (dwv, dws), (div, dis)):
        ref = (float(np.percentile(stats, _P_HIGH)) if stats.size
               else float(np.percentile(vals, _P_HIGH)))
        T += _rank_leq(vals, ref)
    T /= 3.0
    banned: set[tuple[int, int]] = set()
    result = MatchResult([], list(range(E)))
    # 교차 기각 루프 — 안전 상한: 후보 쌍 수 + 1 (안전 상한)
    for _ in range(len(keys) + 1):
        pairs, terms = _assign(E, keys, cost, T, banned)
        conns = [Connection(i=i, j=j, curve=curves[(i, j)],
                            widths=(endpoints[i].width, endpoints[j].width),
                            inks=(endpoints[i].ink, endpoints[j].ink))
                 for i, j in pairs]
        result = MatchResult(conns, terms)
        bad = _find_crossings(pairs, curves, cost, endpoints, sk, occ)
        if not bad:
            break
        banned |= bad
    return result
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_matching.py -q` / Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/matching.py tests/test_ssei_matching.py
git commit -m "feat(ssei_v2): rank-normalized Hungarian endpoint matching with termination and crossing rejection"
```

### Task 4: render.py — 곡선 렌더링 + Phase A 진입점

**Files:**
- Create: `src/kp3d/modules/ssei_v2/render.py`
- Test: `tests/test_ssei_render.py`

**Interfaces:**
- Consumes: Task 1 `detect_break_endpoints`; Task 3 `match_endpoints`, `Connection`
- Produces: `LineCompletionResult(line_alpha (H,W)f32, skeleton (H,W)bool, width_map (H,W)f32, connections list[Connection], terminations list[int], endpoints list[Endpoint])`, `complete_lines(line_alpha, skeleton, width_map, occlusion_mask) -> LineCompletionResult` — Phase A 진입점. 종결 endpoint는 렌더링 없이 기록만 (획이 거기서 끝남).

- [ ] **Step 1: 실패 테스트 작성**

```python
"""render.py 테스트: 곡선 스탬프, 성분 병합, 무가림 항등."""
import numpy as np
from scipy.ndimage import label

from kp3d.modules.ssei_v2.render import complete_lines

_N8 = np.ones((3, 3), dtype=bool)


def _gap_scene(h=64, w=64, y=32, gap=(24, 41), width=3.0, ink=0.8):
    skeleton = np.zeros((h, w), dtype=bool)
    skeleton[y, 4:w - 4] = True
    occlusion = np.zeros((h, w), dtype=bool)
    occlusion[y - 6:y + 7, gap[0]:gap[1]] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, width, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, ink, 0.0).astype(np.float32)
    return skeleton, width_map, line_alpha, occlusion


def test_connection_merges_components():
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    assert label(skeleton, structure=_N8)[1] == 2
    res = complete_lines(line_alpha, skeleton, width_map, occlusion)
    assert len(res.connections) == 1
    assert label(res.skeleton, structure=_N8)[1] == 1
    # 간격 중앙이 잉크·폭으로 채워졌다
    assert float(res.line_alpha[32, 32]) > 0.5
    assert abs(float(res.width_map[32, 32]) - 3.0) < 1.0


def test_no_occlusion_identity():
    skeleton, width_map, line_alpha, _ = _gap_scene()
    occlusion = np.zeros_like(skeleton)
    res = complete_lines(line_alpha, skeleton, width_map, occlusion)
    assert res.connections == [] and res.terminations == []
    assert np.array_equal(res.skeleton, skeleton)
    assert np.array_equal(res.line_alpha, line_alpha)
    assert np.array_equal(res.width_map, width_map)


def test_inputs_not_mutated():
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    sk0, wm0, la0 = skeleton.copy(), width_map.copy(), line_alpha.copy()
    complete_lines(line_alpha, skeleton, width_map, occlusion)
    assert np.array_equal(skeleton, sk0)
    assert np.array_equal(width_map, wm0)
    assert np.array_equal(line_alpha, la0)
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_render.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/render.py`:

```python
"""연결 곡선 렌더링과 Phase A 진입점 (스펙 §3.2 ⑤).

스탬프: 곡선 표본마다 반경 w/2 디스크를 반 픽셀 안티에일리어싱 경계로
찍는다 — coverage = clip(w/2 + 0.5 − dist, 0, 1) · ink, alpha는 max 갱신.
폭·잉크는 두 endpoint 값의 호장 선형 보간.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clothoid import ConnectionCurve
from .endpoints import Endpoint, detect_break_endpoints
from .matching import Connection, match_endpoints

# 반 픽셀 AA 경계 — 이산 격자 커버리지 (수학 유도)
_HALF_PIXEL = 0.5


@dataclass
class LineCompletionResult:
    """Phase A 산출: 완성된 선 채널."""

    line_alpha: np.ndarray   # (H,W) float32
    skeleton: np.ndarray     # (H,W) bool — 연결 곡선 픽셀 추가됨
    width_map: np.ndarray    # (H,W) float32
    connections: list[Connection]
    terminations: list[int]
    endpoints: list[Endpoint]  # 검출 endpoint 전체 — connections의 i/j 인덱스 대상 (Task 8 G2 검증용)


def _stamp(alpha: np.ndarray, width_map: np.ndarray, skeleton: np.ndarray,
           curve: ConnectionCurve, widths: tuple[float, float],
           inks: tuple[float, float]) -> None:
    """곡선 하나를 in-place 스탬프."""
    h, w = alpha.shape
    n = len(curve.points)
    t = np.linspace(0.0, 1.0, n)
    ws = widths[0] + (widths[1] - widths[0]) * t   # 폭 선형 보간
    aks = inks[0] + (inks[1] - inks[0]) * t        # 잉크 선형 보간
    for (py, px), wk, ak in zip(curve.points, ws, aks):
        r = wk / 2.0 + _HALF_PIXEL
        y0 = max(int(np.floor(py - r)), 0)
        y1 = min(int(np.ceil(py + r)) + 1, h)
        x0 = max(int(np.floor(px - r)), 0)
        x1 = min(int(np.ceil(px + r)) + 1, w)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(yy - py, xx - px)
        cov = (np.clip(wk / 2.0 + _HALF_PIXEL - d, 0.0, 1.0)
               * float(ak)).astype(alpha.dtype)
        np.maximum(alpha[y0:y1, x0:x1], cov, out=alpha[y0:y1, x0:x1])
        iy, ix = int(round(float(py))), int(round(float(px)))
        if 0 <= iy < h and 0 <= ix < w:
            skeleton[iy, ix] = True
            width_map[iy, ix] = max(float(width_map[iy, ix]), float(wk))


def complete_lines(line_alpha: np.ndarray, skeleton: np.ndarray,
                   width_map: np.ndarray,
                   occlusion_mask: np.ndarray) -> LineCompletionResult:
    """Phase A: 끊김 검출 → 매칭 → G2 곡선 렌더링. 입력은 변형하지 않는다."""
    la = np.asarray(line_alpha, dtype=np.float32).copy()
    sk = np.asarray(skeleton, dtype=bool).copy()
    wm = np.asarray(width_map, dtype=np.float32).copy()
    occ = np.asarray(occlusion_mask, dtype=bool)
    eps = detect_break_endpoints(sk, wm, la, occ)
    match = match_endpoints(eps, sk, wm, la, occ)
    for conn in match.connections:
        _stamp(la, wm, sk, conn.curve, conn.widths, conn.inks)
    return LineCompletionResult(line_alpha=la, skeleton=sk, width_map=wm,
                                connections=match.connections,
                                terminations=match.terminations,
                                endpoints=eps)
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_render.py -q` / Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/render.py tests/test_ssei_render.py
git commit -m "feat(ssei_v2): connection curve rendering and Phase A entry point"
```

---

### Task 5: pool.py — 조각 분할 + 선 위상 제약 exemplar pool

**Files:**
- Create: `src/kp3d/modules/ssei_v2/pool.py`
- Test: `tests/test_ssei_pool.py`

**Interfaces:**
- Consumes: 없음 (numpy/scipy만)
- Produces: `PiecePool(piece_mask (H,W)bool, pool_mask (H,W)bool, borrowed bool)`, `build_piece_pools(color_layer (H,W,3)u8, occlusion_mask, line_mask, visible_mask, patch_size int) -> list[PiecePool]`, `_initial_dmax(shape) -> int`

- [ ] **Step 1: 실패 테스트 작성**

```python
"""pool.py 테스트: 선 위상 분할, δ_safe 중심, 고립 조각 차용."""
import numpy as np

from kp3d.modules.ssei_v2.pool import _initial_dmax, build_piece_pools


def test_initial_dmax_spec_rule():
    # v1 해상도 적응 규칙 (스펙 §3.5 승계)
    assert _initial_dmax((100, 150)) == 15
    assert _initial_dmax((300, 200)) == 25
    assert _initial_dmax((500, 400)) == 40


def _base(h=64, w=64):
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:32] = (0, 200, 0)
    color[32:] = (0, 0, 200)
    occ = np.zeros((h, w), dtype=bool)
    occ[20:44, 20:44] = True
    visible = np.ones((h, w), dtype=bool)
    return color, occ, visible


def test_line_splits_pieces_and_pools():
    color, occ, visible = _base()
    lines = np.zeros(occ.shape, dtype=bool)
    lines[32, :] = True
    pools = build_piece_pools(color, occ, lines, visible, patch_size=5)
    assert len(pools) == 2
    top = min(pools, key=lambda p: np.argwhere(p.piece_mask)[:, 0].mean())
    bot = max(pools, key=lambda p: np.argwhere(p.piece_mask)[:, 0].mean())
    assert not top.borrowed and not bot.borrowed
    # 선 장벽: pool이 반대편으로 새지 않는다
    assert np.argwhere(top.pool_mask)[:, 0].max() < 32
    assert np.argwhere(bot.pool_mask)[:, 0].min() > 32
    # 조각 합집합이 가림 전체(선이 덮은 픽셀 포함)를 덮는다
    assert np.array_equal(top.piece_mask | bot.piece_mask, occ)


def test_pool_centers_are_patch_safe():
    color, occ, visible = _base()
    lines = np.zeros(occ.shape, dtype=bool)
    lines[32, :] = True
    p = 5
    r = p // 2
    pools = build_piece_pools(color, occ, lines, visible, patch_size=p)
    for pp in pools:
        for y, x in np.argwhere(pp.pool_mask):
            win = occ[y - r:y + r + 1, x - r:x + r + 1]
            assert win.shape == (p, p) and not win.any()


def test_isolated_piece_borrows():
    color, occ, visible = _base()
    lines = np.zeros(occ.shape, dtype=bool)
    yy, xx = np.mgrid[0:64, 0:64]
    lines |= np.abs(np.hypot(yy - 32, xx - 32) - 8.0) < 1.0  # 폐곡선(원환) 선
    pools = build_piece_pools(color, occ, lines, visible, patch_size=5)
    inner = [p for p in pools if p.piece_mask[32, 32]]
    assert len(inner) == 1
    assert inner[0].borrowed
    assert inner[0].pool_mask.any()
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_pool.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/pool.py`:

```python
"""조각 분할과 선 위상 제약 exemplar pool (스펙 §3.3 ①②, §3.6).

가림 영역을 완성된 선으로 분할한 조각마다, 선 장벽을 넘지 않고 도달
가능한(체비쇼프 반복 팽창) 가시 픽셀만 exemplar 후보로 삼는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt, label)

# 8-근방(체비쇼프 1) 구조 원소 — 이산 위상 (수학 유도)
_N8 = np.ones((3, 3), dtype=bool)
# 배가 확장 계수 ×2 — 지수 탐색 관례 (정규화 규칙)
_GROWTH = 2


def _initial_dmax(shape: tuple[int, int]) -> int:
    """d_max 초기값 — v1 해상도 적응 규칙 (스펙 §3.5 승계)."""
    m = max(shape[0], shape[1])
    if m < 200:
        return 15
    if m < 400:
        return 25
    return 40


@dataclass
class PiecePool:
    """조각과 그 exemplar 패치 중심 후보."""

    piece_mask: np.ndarray  # (H,W) bool — 채울 조각
    pool_mask: np.ndarray   # (H,W) bool — 유효 패치 중심(δ_safe 적용)
    borrowed: bool          # 고립 조각의 타 조각/전역 pool 차용 여부


def _split_pieces(occ: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, int]:
    """가림을 선으로 분할. 선이 덮은 hole 픽셀은 최근접 조각에 귀속."""
    core = occ & ~lines
    labels, n = label(core, structure=_N8)
    covered = occ & lines
    if np.any(covered) and n > 0:
        # 최근접 조각 픽셀 인덱스 귀속 — 거리 변환 (순수 수학)
        _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
        labels = np.where(covered, labels[iy, ix], labels)
    return labels, n


def build_piece_pools(color_layer: np.ndarray, occlusion_mask: np.ndarray,
                      line_mask: np.ndarray, visible_mask: np.ndarray,
                      patch_size: int) -> list[PiecePool]:
    """조각별 exemplar pool을 만든다. pool이 빈 고립 조각은 차용한다."""
    occ = np.asarray(occlusion_mask, dtype=bool)
    lines = np.asarray(line_mask, dtype=bool)
    visible = np.asarray(visible_mask, dtype=bool) & ~occ
    color = np.asarray(color_layer, dtype=np.float64)
    h, w = occ.shape
    # δ_safe: 패치 창 p×p가 가림·이미지 경계와 겹치지 않는 중심만
    # (경계 처리: erosion 기본 border_value=0이 창의 in-bounds를 보장)
    safe = binary_erosion(
        ~occ, structure=np.ones((patch_size, patch_size), dtype=bool))
    labels, n = _split_pieces(occ, lines)
    diag = int(np.ceil(np.hypot(h, w)))  # 확장 상한: 이미지 대각선 (안전 상한)
    pools: list[PiecePool] = []
    for k in range(1, n + 1):
        piece = labels == k
        dmax = _initial_dmax((h, w))
        prev_grown = None
        while True:
            # 선 장벽을 넘지 않는 반복 팽창 (체비쇼프 거리 dmax)
            grown = binary_dilation(piece, structure=_N8, iterations=dmax,
                                    mask=~lines | piece)
            pool = grown & visible & safe
            if int(pool.sum()) >= int(piece.sum()) or dmax >= diag:
                break
            if prev_grown is not None and np.array_equal(grown, prev_grown):
                break  # 성장 포화 — 더 확장해도 도달 픽셀 없음
            prev_grown = grown
            dmax *= _GROWTH  # 배가 확장 (정규화 규칙)
        pools.append(PiecePool(piece_mask=piece, pool_mask=pool,
                               borrowed=False))
    # 고립 조각: 평균색 최근접 타 조각 pool 차용 → 전역 pool fallback (스펙 §3.6)
    global_pool = visible & safe
    for pp in pools:
        if np.any(pp.pool_mask):
            continue
        ring = binary_dilation(pp.piece_mask, structure=_N8,
                               iterations=_initial_dmax((h, w))) & visible
        best = None
        if np.any(ring):
            ref = color[ring].reshape(-1, 3).mean(axis=0)
            cands = [q for q in pools if q is not pp and np.any(q.pool_mask)]
            if cands:
                dists = [float(np.linalg.norm(
                    color[q.pool_mask].reshape(-1, 3).mean(axis=0) - ref))
                    for q in cands]
                best = cands[int(np.argmin(dists))]
        pp.pool_mask = (best.pool_mask.copy() if best is not None
                        else global_pool.copy())
        pp.borrowed = True
    return pools
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_pool.py -q` / Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/pool.py tests/test_ssei_pool.py
git commit -m "feat(ssei_v2): piece segmentation and line-topology-constrained exemplar pools"
```

### Task 6: patchmatch.py — 패치 크기 유도 + PatchMatch ANN

**Files:**
- Create: `src/kp3d/modules/ssei_v2/patchmatch.py`
- Test: `tests/test_ssei_patchmatch.py`

**Interfaces:**
- Consumes: 없음 (numpy만; 테스트만 cv2 사용)
- Produces: `derive_patch_size(gray (H,W)f64, valid (H,W)bool) -> int` (홀수, [3, min(H,W)//4]); `patchmatch(image (H,W,3)f64, target_mask, pool_mask, patch_size, noise_sigma) -> (nnf (H,W,2)i64, dists (H,W)f64)` — target/pool 중심의 p×p 창은 호출자가 in-bounds 보장; `_MAX_ITERS = 10`

- [ ] **Step 1: 실패 테스트 작성**

```python
"""patchmatch.py 테스트: 패치 크기 유도(상관 길이), ANN 정확 사본 발견."""
import cv2
import numpy as np

from kp3d.modules.ssei_v2.patchmatch import derive_patch_size, patchmatch


def test_derive_patch_size_bounds_and_order():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((64, 64))
    smooth = cv2.GaussianBlur(noise, (0, 0), 4.0)
    valid = np.ones((64, 64), dtype=bool)
    p_n = derive_patch_size(noise, valid)
    p_s = derive_patch_size(smooth, valid)
    for p in (p_n, p_s):
        assert p % 2 == 1 and 3 <= p <= 64 // 4
    # 상관 길이가 길수록 패치가 크다
    assert p_s > p_n


def test_derive_patch_size_constant_image_min():
    g = np.full((40, 40), 7.0)
    assert derive_patch_size(g, np.ones((40, 40), dtype=bool)) == 3


def test_patchmatch_finds_exact_copies():
    rng = np.random.default_rng(2)
    base = rng.random((8, 8, 3))
    img = np.tile(base, (5, 5, 1))  # 주기 8 텍스처 — 정확 사본 다수
    tm = np.zeros((40, 40), dtype=bool)
    tm[10:20, 5:12] = True
    pm = np.zeros((40, 40), dtype=bool)
    pm[3:37, 23:37] = True
    nnf, dists = patchmatch(img, tm, pm, patch_size=5, noise_sigma=1e-6)
    ys, xs = np.nonzero(tm)
    # 대응은 항상 pool 안
    assert pm[nnf[ys, xs, 0], nnf[ys, xs, 1]].all()
    # 정확 사본이 존재하므로 전파+랜덤 탐색이 SSD 0을 찾는다
    assert float(np.median(dists[tm])) < 1e-9
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_patchmatch.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/patchmatch.py`:

```python
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
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_patchmatch.py -q` / Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/patchmatch.py tests/test_ssei_patchmatch.py
git commit -m "feat(ssei_v2): autocorrelation-derived patch size and PatchMatch ANN"
```

---

### Task 7: fill.py — Wexler voting + multi-scale EM + by-construction

**Files:**
- Create: `src/kp3d/modules/ssei_v2/fill.py`
- Test: `tests/test_ssei_fill.py`

**Interfaces:**
- Consumes: Task 5 `build_piece_pools`, `PiecePool`; Task 6 `derive_patch_size`, `patchmatch`, `_MAX_ITERS`
- Produces: `ColorFillResult(filled (H,W,3)u8, pieces list[PiecePool], by_construction_violations int, patch_size int, levels int)`, `fill_color(color_layer (H,W,3)u8 BGR, occlusion_mask, line_mask, visible_mask, noise_sigma) -> ColorFillResult` — **noise_sigma는 Stage 0 `decompose()`가 반환한 0..255 스케일 값** (내부에서 /255 정규화; 구현자는 decomposition 모듈에서 단위를 확인할 것)

- [ ] **Step 1: 실패 테스트 작성**

```python
"""fill.py 테스트: 선 위상 존중 채움, 가시 보존, by-construction 불변식."""
import numpy as np

from kp3d.modules.ssei_v2.fill import fill_color


def _two_region_scene(h=64, w=64):
    color = np.zeros((h, w, 3), dtype=np.int64)
    color[:32] = (0, 180, 0)   # BGR — 위 녹색
    color[32:] = (0, 0, 180)   # 아래 적색
    rng = np.random.default_rng(1)
    color = np.clip(color + rng.integers(-15, 16, color.shape), 0, 255)
    color = color.astype(np.uint8)
    lines = np.zeros((h, w), dtype=bool)
    lines[32, :] = True
    occ = np.zeros((h, w), dtype=bool)
    occ[24:40, 24:40] = True
    visible = np.ones((h, w), dtype=bool)
    return color, occ, lines, visible


def test_fill_respects_line_topology():
    color, occ, lines, visible = _two_region_scene()
    res = fill_color(color, occ, lines, visible, noise_sigma=2.0)
    top = occ.copy()
    top[32:] = False
    bot = occ.copy()
    bot[:33] = False
    ft = res.filled[top].astype(np.float64).mean(axis=0)
    fb = res.filled[bot].astype(np.float64).mean(axis=0)
    assert ft[1] > ft[2] + 30  # 위쪽 채움은 녹색 우세
    assert fb[2] > fb[1] + 30  # 아래쪽 채움은 적색 우세


def test_fill_preserves_visible_and_invariants():
    color, occ, lines, visible = _two_region_scene()
    res = fill_color(color, occ, lines, visible, noise_sigma=2.0)
    assert np.array_equal(res.filled[~occ], color[~occ])
    assert res.filled.dtype == np.uint8
    assert res.patch_size % 2 == 1 and res.patch_size >= 3
    assert res.levels >= 1
    assert len(res.pieces) == 2
    # by construction: 채움 픽셀은 기여 exemplar [min,max]±½단위 안 (스펙 §5.1)
    assert res.by_construction_violations == 0
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_fill.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/fill.py`:

```python
"""Phase B 색 채움 — Wexler voting + multi-scale EM + by-construction (스펙 §3.3–3.4, §5.1)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from .patchmatch import _MAX_ITERS, derive_patch_size, patchmatch
from .pool import PiecePool, build_piece_pools

# 8-bit 반 단위(정규화) — 양자화 반 스텝 (수학 유도)
_HALF_LEVEL = 0.5 / 255.0
# Wexler 가중 스케일 백분위 P75 — Wexler 관례 (정규화 규칙)
_P_SIGMA = 75.0
# 8-근방 — 이산 위상 (수학 유도)
_N8 = np.ones((3, 3), dtype=bool)


@dataclass
class ColorFillResult:
    """Phase B 산출."""

    filled: np.ndarray      # (H,W,3) uint8 BGR
    pieces: list[PiecePool]
    by_construction_violations: int
    patch_size: int
    levels: int             # 사용된 최대 피라미드 레벨 수


def _levels_for(piece: np.ndarray, p: int) -> int:
    """레벨 수 = ceil(log2(결손 지름 / 패치)) — 스케일 커버리지 (수학 유도)."""
    rmax = float(distance_transform_edt(piece).max())
    return max(1, int(np.ceil(np.log2(max(2.0 * rmax / p, 1.0)))))


def _onion_peel(img: np.ndarray, hole: np.ndarray) -> None:
    """구멍을 경계층부터 이웃 평균으로 반복 초기화 (in-place, C2)."""
    rem = hole.copy()
    while np.any(rem):
        ring = rem & binary_dilation(~rem, structure=_N8)
        if not np.any(ring):
            break
        known = (~rem).astype(np.float64)
        cnt = cv2.blur(known, (3, 3)) * 9.0
        acc = cv2.blur(img * known[..., None], (3, 3)) * 9.0
        ys, xs = np.nonzero(ring)
        img[ys, xs] = acc[ys, xs] / np.maximum(cnt[ys, xs], 1.0)[..., None]
        rem[ys, xs] = False


def _vote(img, hole, tmask, nnf, dists, p, collect_minmax):
    """가중 Wexler voting으로 hole 픽셀 갱신 → (평균 변화량, vmin, vmax)."""
    h, w, _ = img.shape
    r = p // 2
    dv = dists[tmask]
    sigma = max(float(np.percentile(dv, _P_SIGMA)),
                float(np.finfo(np.float64).tiny))
    acc = np.zeros_like(img)
    wgt = np.zeros((h, w))
    vmin = np.full_like(img, np.inf) if collect_minmax else None
    vmax = np.full_like(img, -np.inf) if collect_minmax else None
    for ty, tx in np.argwhere(tmask):
        wc = float(np.exp(-float(dists[ty, tx]) ** 2 / (2.0 * sigma ** 2)))
        sy, sx = int(nnf[ty, tx, 0]), int(nnf[ty, tx, 1])
        src = img[sy - r:sy + r + 1, sx - r:sx + r + 1]
        sub = hole[ty - r:ty + r + 1, tx - r:tx + r + 1]
        acc[ty - r:ty + r + 1, tx - r:tx + r + 1] += wc * src * sub[..., None]
        wgt[ty - r:ty + r + 1, tx - r:tx + r + 1] += wc * sub
        if collect_minmax:
            win = (slice(ty - r, ty + r + 1), slice(tx - r, tx + r + 1))
            np.minimum(vmin[win], np.where(sub[..., None], src, np.inf),
                       out=vmin[win])
            np.maximum(vmax[win], np.where(sub[..., None], src, -np.inf),
                       out=vmax[win])
    upd = hole & (wgt > 0.0)
    if not np.any(upd):
        return 0.0, vmin, vmax
    old = img[upd].copy()
    img[upd] = acc[upd] / wgt[upd][..., None]
    return float(np.abs(img[upd] - old).mean()), vmin, vmax


def _fill_piece(img: np.ndarray, piece: np.ndarray, pool_mask: np.ndarray,
                p: int, sigma_n: float) -> tuple[int, int]:
    """한 조각을 multi-scale EM으로 채운다 (img in-place) → (위반 수, 레벨 수)."""
    levels = _levels_for(piece, p)
    imgs, holes, pools = [img], [piece], [pool_mask]
    for _ in range(levels - 1):
        prev = imgs[-1]
        nh, nw = (prev.shape[0] + 1) // 2, (prev.shape[1] + 1) // 2
        imgs.append(cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA))
        holes.append(cv2.resize(holes[-1].astype(np.uint8), (nw, nh),
                                interpolation=cv2.INTER_NEAREST).astype(bool))
        pools.append(cv2.resize(pools[-1].astype(np.uint8), (nw, nh),
                                interpolation=cv2.INTER_NEAREST).astype(bool))
    violations = 0
    square = np.ones((p, p), dtype=bool)
    r = p // 2
    for lv in range(levels - 1, -1, -1):  # 거친 → 미세 (C2: 스케일 순서)
        im, hl = imgs[lv], holes[lv]
        if not np.any(hl):
            continue
        hh, ww = hl.shape
        # 레벨별 δ_safe 재적용 (창 in-bounds + hole 비겹침)
        safe = binary_erosion(~hl, structure=square)
        pl = pools[lv] & safe
        if not np.any(pl):
            pl = safe  # 전역 fallback (스펙 §3.6)
        if not np.any(pl):
            continue  # 레벨이 너무 거칢 — 미세 레벨에서 처리
        inb = np.zeros_like(hl)
        inb[r:hh - r, r:ww - r] = True
        tmask = binary_dilation(hl, structure=square) & inb
        if not np.any(tmask):
            continue
        if lv == levels - 1:
            _onion_peel(im, hl)  # 최심 레벨 초기화 (C2)
        vmin = vmax = None
        for _ in range(_MAX_ITERS):  # EM 안전 상한
            nnf, dists = patchmatch(im, tmask, pl, p, sigma_n)
            change, vmin, vmax = _vote(im, hl, tmask, nnf, dists, p,
                                       collect_minmax=(lv == 0))
            if change < sigma_n:
                break  # 잡음 바닥 이하 변화 — 수렴 (P-adapt)
        if lv > 0:
            fine = imgs[lv - 1]
            up = cv2.resize(im, (fine.shape[1], fine.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
            fh = holes[lv - 1]
            fine[fh] = up[fh]  # 다음(미세) 레벨 초기화
        elif vmin is not None:
            # by construction: 기여 exemplar [min,max] ± 반 단위 (스펙 §5.1)
            sel = hl[..., None] & np.isfinite(vmin)
            lo = np.where(np.isfinite(vmin), vmin - _HALF_LEVEL, 0.0)
            hi = np.where(np.isfinite(vmax), vmax + _HALF_LEVEL, 1.0)
            bad = sel & ((im < lo) | (im > hi))
            violations = int(np.count_nonzero(np.any(bad, axis=-1)))
            im[:] = np.where(sel, np.clip(im, lo, hi), im)
    return violations, levels


def fill_color(color_layer: np.ndarray, occlusion_mask: np.ndarray,
               line_mask: np.ndarray, visible_mask: np.ndarray,
               noise_sigma: float) -> ColorFillResult:
    """Phase B 진입점: 조각별 선 위상 제약 exemplar 채움.

    noise_sigma는 Stage 0 decompose()의 0..255 스케일 — 내부 /255 정규화.
    """
    occ = np.asarray(occlusion_mask, dtype=bool)
    img8 = np.asarray(color_layer, dtype=np.uint8)
    visible = np.asarray(visible_mask, dtype=bool)
    gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY).astype(np.float64)
    p = derive_patch_size(gray, visible & ~occ)
    pieces = build_piece_pools(img8, occ, np.asarray(line_mask, dtype=bool),
                               visible, p)
    img = img8.astype(np.float64) / 255.0
    sigma_n = float(noise_sigma) / 255.0
    total_violations = 0
    max_levels = 1
    for pp in pieces:
        v, lv = _fill_piece(img, pp.piece_mask, pp.pool_mask, p, sigma_n)
        total_violations += v
        max_levels = max(max_levels, lv)
    filled = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    filled[~occ] = img8[~occ]  # 가시 픽셀 원본 보존
    return ColorFillResult(filled=filled, pieces=pieces,
                           by_construction_violations=total_violations,
                           patch_size=p, levels=max_levels)
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_fill.py -q` / Expected: 2 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/fill.py tests/test_ssei_fill.py
git commit -m "feat(ssei_v2): piecewise Wexler voting fill with multi-scale EM and by-construction check"
```

### Task 8: inpaint.py — Phase C 통합 + 기계 검증 + 공개 API

**Files:**
- Create: `src/kp3d/modules/ssei_v2/inpaint.py`
- Modify: `src/kp3d/modules/ssei_v2/__init__.py` (Task 1의 placeholder를 공개 API 재수출로 완성)
- Test: `tests/test_ssei_inpaint.py`

**Interfaces:**
- Consumes: `complete_lines` (Task 4, `LineCompletionResult.endpoints` 포함), `fill_color` (Task 7), `kp3d.modules.decomposition.recompose(image_bgr, line_alpha, color_layer) -> (H,W,3)u8` — alpha==0 픽셀은 color를 정확 복사 (Stage 0 불변식)
- Produces: `InpaintingResult(inpainted (H,W,3)u8, line LineCompletionResult, color ColorFillResult, g2_tangent_max float, g2_curvature_max float, by_construction_violations int)`, `inpaint(image_bgr, color_layer, line_alpha, skeleton, width_map, occlusion_mask, noise_sigma, visible_mask=None) -> InpaintingResult` — 모듈의 유일한 진입점 (Plan 4의 Stage 2 v2가 소비)

**설계 결정 — 선 RGB 소스:** Stage 0의 `recompose`는 선 RGB로 원본 이미지를 쓰지만, 가림 내부에는 원본(그림)이 없다 (가림 물체 픽셀뿐). 따라서 가시 선 픽셀의 잉크(alpha) 가중 평균색으로 가림 내부 선 RGB를 합성한다 — 데이터에서 유도, 상수 없음 (P-adapt ③). 이 때문에 `inpaint`는 `image_bgr`(원본, 가림 포함)를 첫 인자로 받는다.

- [ ] **Step 1: 실패 테스트 작성**

```python
"""inpaint.py 테스트: Phase C 통합, G2 조인트 기계 검증, 가림 밖 불변."""
import numpy as np
from scipy.ndimage import binary_dilation

from kp3d.modules.decomposition import recompose
from kp3d.modules.ssei_v2.inpaint import InpaintingResult, inpaint


def _scene(h=64, w=96):
    """수평 먹선(y=32, 폭 3) + 녹색 배경 + 중앙 가림 상자."""
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:] = (40, 160, 40)
    line_alpha = np.zeros((h, w), dtype=np.float32)
    skeleton = np.zeros((h, w), dtype=bool)
    width_map = np.zeros((h, w), dtype=np.float32)
    y = 32
    skeleton[y, 8:88] = True
    width_map[y, 8:88] = 3.0
    line_alpha[y - 1:y + 2, 8:88] = 1.0
    image = color.copy()
    image[line_alpha > 0] = (20, 20, 20)
    occ = np.zeros((h, w), dtype=bool)
    occ[24:40, 40:56] = True
    # 가림 내부 지식 소거 + 가림 픽셀은 마젠타(가림 물체)로 오염
    line_alpha[occ] = 0.0
    skeleton[occ] = False
    width_map[occ] = 0.0
    image[occ] = (255, 0, 255)
    color[occ] = (255, 0, 255)
    return image, color, line_alpha, skeleton, width_map, occ


def _run():
    image, color, la, sk, wm, occ = _scene()
    res = inpaint(image, color, la, sk, wm, occ, noise_sigma=1.0)
    return image, color, la, sk, wm, occ, res


def test_inpaint_connects_line_and_fills_color():
    image, color, la, sk, wm, occ, res = _run()
    assert isinstance(res, InpaintingResult)
    assert res.line.connections                  # 획이 실제로 연결됨
    assert res.line.line_alpha[32, 48] > 0.5     # 가림 중앙에 선 복원
    px = res.inpainted[26, 48]                   # 선에서 떨어진 가림 내부
    assert int(px[1]) > int(px[0]) and int(px[1]) > 100  # 녹색 배경 회복 (마젠타 아님)
    assert res.color.by_construction_violations == 0
    assert res.by_construction_violations == 0


def test_g2_joint_machine_verification():
    *_, res = _run()
    # 조인트 접선·곡률 — biclothoid/quintic 모두 경계에서 해석적으로 정확
    assert res.line.connections
    assert res.g2_tangent_max < 1e-6
    assert res.g2_curvature_max < 1e-6


def test_outside_occlusion_invariant():
    image, color, la, sk, wm, occ, res = _run()
    baseline = recompose(image, la, color)
    # 스탬프가 가림 밖 endpoint 주변(창=선폭×2, 정규화 규칙)까지 닿을 수 있어 링 제외
    ring = binary_dilation(occ, iterations=6)
    assert np.array_equal(res.inpainted[~ring], baseline[~ring])


def test_public_api_reexport():
    import kp3d.modules.ssei_v2 as m

    for name in ("Endpoint", "connect_g2", "match_endpoints", "complete_lines",
                 "build_piece_pools", "derive_patch_size", "patchmatch",
                 "fill_color", "inpaint", "InpaintingResult"):
        assert hasattr(m, name)
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_ssei_inpaint.py -q` / Expected: FAIL (`ImportError: cannot import name 'inpaint'`)

- [ ] **Step 3: 구현**

`src/kp3d/modules/ssei_v2/inpaint.py`:

```python
"""Phase C: 선 완성 + 색 채움 + 재합성 통합, 기계 검증 (스펙 §3.4, §3.6).

재합성은 Stage 0의 recompose를 재사용한다(DRY). recompose는 선 RGB
소스로 원본 이미지를 쓰지만 가림 내부에는 원본이 없으므로, 가시 선
픽셀의 잉크(alpha) 가중 평균색으로 가림 내부 선 RGB를 합성한다 —
데이터에서 유도, 상수 없음.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kp3d.modules.decomposition import recompose

from .fill import ColorFillResult, fill_color
from .render import LineCompletionResult, complete_lines


@dataclass
class InpaintingResult:
    """SSEI 2.0 산출 + 기계 검증 지표 (스펙 §3.6)."""

    inpainted: np.ndarray            # (H,W,3) uint8 BGR
    line: LineCompletionResult       # Phase A 산출
    color: ColorFillResult           # Phase B 산출
    g2_tangent_max: float            # 조인트 접선 각도 불연속 최대 [rad]
    g2_curvature_max: float          # 조인트 곡률 불연속 최대 [1/px]
    by_construction_violations: int  # Phase B 색 유래 위반 수 (== color.by_construction_violations)


def _g2_joint_errors(line: LineCompletionResult) -> tuple[float, float]:
    """연결 조인트의 접선 각도·곡률 불연속 최대값 (기계 검증).

    matching.py 호출 규약: 곡선 진행 방향은 시작에서 e_i.tangent,
    끝에서 −e_j.tangent (둘 다 endpoint의 바깥 방향 서술자 기준).
    """
    t_max, k_max = 0.0, 0.0
    for conn in line.connections:
        e_i = line.endpoints[conn.i]
        e_j = line.endpoints[conn.j]
        joints = (
            (conn.curve.tangents[0], e_i.tangent,
             conn.curve.curvatures[0], e_i.curvature),
            (conn.curve.tangents[-1], -e_j.tangent,
             conn.curve.curvatures[-1], -e_j.curvature),
        )
        for tc, te, kc, ke in joints:
            dot = float(np.clip(np.dot(tc, te), -1.0, 1.0))
            t_max = max(t_max, float(np.arccos(dot)))
            k_max = max(k_max, abs(float(kc) - float(ke)))
    return t_max, k_max


def _line_rgb_image(image_bgr: np.ndarray, line_alpha: np.ndarray,
                    occlusion_mask: np.ndarray,
                    visible_mask: np.ndarray) -> np.ndarray:
    """recompose용 선 RGB 소스 이미지.

    가시 영역은 원본 그대로 (Stage 0 불변식과 동일). 가림 내부는 가시 선
    픽셀의 잉크(alpha) 가중 평균색. 가시 선이 없으면 가림 내부 alpha도
    0이라 recompose가 color를 그대로 복사 — 값 미사용이므로 그대로 둔다.
    """
    img = np.asarray(image_bgr, dtype=np.uint8)
    out = img.copy()
    a = np.asarray(line_alpha, dtype=np.float64)
    occ = np.asarray(occlusion_mask, dtype=bool)
    src = (a > 0.0) & np.asarray(visible_mask, dtype=bool) & ~occ
    if np.any(src):
        wts = a[src]
        mean = (img[src].astype(np.float64) * wts[:, None]).sum(axis=0) / wts.sum()
        out[occ] = np.rint(mean).astype(np.uint8)
    return out


def inpaint(image_bgr: np.ndarray, color_layer: np.ndarray,
            line_alpha: np.ndarray, skeleton: np.ndarray,
            width_map: np.ndarray, occlusion_mask: np.ndarray,
            noise_sigma: float,
            visible_mask: np.ndarray | None = None) -> InpaintingResult:
    """SSEI 2.0 진입점: Phase A(선) → Phase B(색) → Phase C(재합성+검증).

    Args:
        image_bgr: (H,W,3) uint8 원본(가림 포함) — 가시 선 RGB 소스.
        color_layer, line_alpha, skeleton, width_map: Stage 0 decompose 산출.
        occlusion_mask: (H,W) bool 가림 마스크.
        noise_sigma: Stage 0 노이즈 표준편차 (0..255 스케일).
        visible_mask: exemplar 소스 허용 영역 (None이면 전체 —
            객체별 처리는 Plan 4에서 객체 마스크를 전달).
    """
    occ = np.asarray(occlusion_mask, dtype=bool)
    visible = (np.ones(occ.shape, dtype=bool) if visible_mask is None
               else np.asarray(visible_mask, dtype=bool))
    line = complete_lines(line_alpha, skeleton, width_map, occ)
    line_mask = line.line_alpha > 0.0  # 완성된 선 위상 — Phase B 분할 경계
    color = fill_color(color_layer, occ, line_mask, visible, noise_sigma)
    line_img = _line_rgb_image(image_bgr, line.line_alpha, occ, visible)
    inpainted = recompose(line_img, line.line_alpha, color.filled)
    g2_t, g2_k = _g2_joint_errors(line)
    return InpaintingResult(
        inpainted=inpainted, line=line, color=color,
        g2_tangent_max=g2_t, g2_curvature_max=g2_k,
        by_construction_violations=color.by_construction_violations)
```

`src/kp3d/modules/ssei_v2/__init__.py` (placeholder 교체):

```python
"""SSEI 2.0: Structure-first 선·색 인페인팅 (v2 설계 §3)."""
from kp3d.modules.ssei_v2.clothoid import ConnectionCurve, connect_g2
from kp3d.modules.ssei_v2.endpoints import (
    Endpoint,
    detect_break_endpoints,
    stroke_statistics,
    trace_stroke,
)
from kp3d.modules.ssei_v2.fill import ColorFillResult, fill_color
from kp3d.modules.ssei_v2.inpaint import InpaintingResult, inpaint
from kp3d.modules.ssei_v2.matching import Connection, MatchResult, match_endpoints
from kp3d.modules.ssei_v2.patchmatch import derive_patch_size, patchmatch
from kp3d.modules.ssei_v2.pool import PiecePool, build_piece_pools
from kp3d.modules.ssei_v2.render import LineCompletionResult, complete_lines

__all__ = [
    "ColorFillResult",
    "Connection",
    "ConnectionCurve",
    "Endpoint",
    "InpaintingResult",
    "LineCompletionResult",
    "MatchResult",
    "PiecePool",
    "build_piece_pools",
    "complete_lines",
    "connect_g2",
    "derive_patch_size",
    "detect_break_endpoints",
    "fill_color",
    "inpaint",
    "match_endpoints",
    "patchmatch",
    "stroke_statistics",
    "trace_stroke",
]
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_ssei_inpaint.py -q` / Expected: 4 passed. 그다음 전체: `python -m pytest tests/test_ssei_*.py -q` / Expected: 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/ssei_v2/inpaint.py src/kp3d/modules/ssei_v2/__init__.py tests/test_ssei_inpaint.py
git commit -m "feat(ssei_v2): Phase C inpaint entry point with G2/by-construction machine verification"
```

---

### Task 9: 통합 테스트 (프로토콜 B) + 데모 스크립트

**Files:**
- Create: `tests/test_ssei_full.py`
- Create: `scripts/demo_ssei_v2.py`

**Interfaces:**
- Consumes: `kp3d.modules.decomposition.decompose(image_bgr) -> DecompositionResult(line_alpha, color_layer, line_mask, skeleton, width_map, weave, noise_sigma)` (Stage 0), `kp3d.modules.ssei_v2.inpaint` (Task 8)
- Produces: 없음 (검증 산출물)

**프로토콜 B (스펙 §5.1):** 완전한 합성 작품 → Stage 0 분해 → 가림 영역 M의 지식 소거(line_alpha/skeleton/width_map 삭제 + 픽셀 오염) → inpaint → GT 대비 정량 지표(PSNR, 스켈레톤 재현율, 기계 검증 지표).

- [ ] **Step 1: 통합 테스트 작성**

```python
"""SSEI 2.0 통합 — 프로토콜 B (스펙 §5.1): 완전 작품 합성 → 지식 소거 → 복원 정량."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scipy.ndimage import binary_dilation

from kp3d.modules.decomposition import decompose
from kp3d.modules.ssei_v2 import inpaint


def _artwork(h=128, w=128):
    """합성 '완전' 작품: 2색 배경 + 완만한 사인 곡선 먹선 (반지름 2 디스크)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :w // 2] = (60, 170, 60)
    img[:, w // 2:] = (60, 60, 170)
    xs = np.arange(10, w - 10)
    ys = h / 2.0 + 12.0 * np.sin(2.0 * np.pi * (xs - 10) / (w - 20))
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(round(y))), 2, (20, 20, 20), -1)
    return img


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


def test_protocol_b_psnr_and_skeleton():
    gt = _artwork()
    dec = decompose(gt)
    occ = np.zeros(gt.shape[:2], dtype=bool)
    occ[52:76, 52:76] = True
    # 지식 소거 (가림 내부 정보 삭제) + 가림 물체(마젠타) 오염
    la = dec.line_alpha.copy(); la[occ] = 0.0
    sk = dec.skeleton.copy(); sk[occ] = False
    wm = dec.width_map.copy(); wm[occ] = 0.0
    col = dec.color_layer.copy(); col[occ] = (255, 0, 255)
    img_in = gt.copy(); img_in[occ] = (255, 0, 255)

    res = inpaint(img_in, col, la, sk, wm, occ, dec.noise_sigma)

    # ① PSNR(가림 내부): 가시 평균색 채움 베이스라인 초과
    base = gt.copy()
    base[occ] = np.rint(gt[~occ].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    assert _psnr(res.inpainted[occ], gt[occ]) > _psnr(base[occ], gt[occ])

    # ② 스켈레톤 재현율(가림 내부, 1px 팽창 허용): GT 획 자취 회복
    gt_sk = dec.skeleton & occ  # 소거 전 분해가 본 가림 내부 획 자취
    if np.any(gt_sk):
        cover = binary_dilation(res.line.skeleton, iterations=1)
        recall = (float(np.count_nonzero(gt_sk & cover))
                  / float(np.count_nonzero(gt_sk)))
        assert recall > 0.7

    # ③ 기계 검증 지표
    assert res.by_construction_violations == 0
    assert res.g2_tangent_max < 1e-6
    assert res.g2_curvature_max < 1e-6
```

- [ ] **Step 2: 실행 확인** — Run: `python -m pytest tests/test_ssei_full.py -q` / Expected: 1 passed (통합 검증 태스크 — 실패하면 superpowers:systematic-debugging으로 해당 모듈 결함을 추적, 임계값 완화 금지)

- [ ] **Step 3: 데모 스크립트 작성**

`scripts/demo_ssei_v2.py`:

```python
"""SSEI 2.0 데모: 작품 → Stage 0 분해 → 지식 소거 → inpaint → 산출 저장.

실행: python scripts/demo_ssei_v2.py [입력 이미지 경로]
입력 생략 시 통합 테스트와 같은 구성의 합성 작품(256×256)을 사용한다.
산출: outputs/ssei_v2_demo/ 아래 PNG.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kp3d.modules.decomposition import decompose  # noqa: E402
from kp3d.modules.ssei_v2 import inpaint  # noqa: E402


def synthetic_artwork(h: int = 256, w: int = 256) -> np.ndarray:
    """2색 배경 + 사인 곡선 먹선 합성 작품."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :w // 2] = (60, 170, 60)
    img[:, w // 2:] = (60, 60, 170)
    xs = np.arange(10, w - 10)
    ys = h / 2.0 + 0.1 * h * np.sin(2.0 * np.pi * (xs - 10) / (w - 20))
    for x, y in zip(xs, ys):
        cv2.circle(img, (int(x), int(round(y))), 2, (20, 20, 20), -1)
    return img


def main() -> None:
    out_dir = Path("outputs/ssei_v2_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise SystemExit(f"이미지를 읽을 수 없음: {sys.argv[1]}")
    else:
        img = synthetic_artwork()
    h, w = img.shape[:2]
    dec = decompose(img)
    occ = np.zeros((h, w), dtype=bool)
    occ[h * 2 // 5:h * 3 // 5, w * 2 // 5:w * 3 // 5] = True  # 중앙 상자 가림
    la = dec.line_alpha.copy(); la[occ] = 0.0
    sk = dec.skeleton.copy(); sk[occ] = False
    wm = dec.width_map.copy(); wm[occ] = 0.0
    col = dec.color_layer.copy(); col[occ] = (255, 0, 255)
    img_in = img.copy(); img_in[occ] = (255, 0, 255)

    res = inpaint(img_in, col, la, sk, wm, occ, dec.noise_sigma)

    cv2.imwrite(str(out_dir / "input_occluded.png"), img_in)
    cv2.imwrite(str(out_dir / "line_alpha.png"),
                np.rint(res.line.line_alpha * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "color_filled.png"), res.color.filled)
    cv2.imwrite(str(out_dir / "inpainted.png"), res.inpainted)
    print(f"연결 {len(res.line.connections)}건, 종결 {len(res.line.terminations)}건, "
          f"조각 {len(res.color.pieces)}개, 패치 {res.color.patch_size}px, "
          f"레벨 {res.color.levels}")
    print(f"G2 접선 불연속 최대 {res.g2_tangent_max:.2e} rad, "
          f"곡률 불연속 최대 {res.g2_curvature_max:.2e} 1/px")
    print(f"by-construction 위반 {res.by_construction_violations}건")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 데모 스모크 실행** — Run: `python scripts/demo_ssei_v2.py` / Expected: outputs/ssei_v2_demo/ 에 PNG 4개, 위반 0건 출력

- [ ] **Step 5: 전체 스위트 확인** — Run: `python -m pytest tests/test_ssei_*.py -q` / Expected: 전부 passed

- [ ] **Step 6: 커밋**

```bash
git add tests/test_ssei_full.py scripts/demo_ssei_v2.py
git commit -m "test(ssei_v2): protocol B integration test and demo script"
```

