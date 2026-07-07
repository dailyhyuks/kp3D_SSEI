# KP3D v2 Stage 2 객체별 오케스트레이션 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** labelme 어노테이션과 깊이 서열로 가림 그래프를 만들고, 객체별로 SSEI 2.0 아모달 완성을 수행해 객체별 RGBA를 산출하는 v2 오케스트레이터 (스펙 §4.1/§4.3).

**Architecture:** 신규 모듈 `src/kp3d/modules/orchestration/` — annotations(labelme 로딩·병합·깊이) → graph(가림 간선·후보 영역) → complete(객체별 지식 소거 + ssei_v2.inpaint) → orchestrate(자가 경쟁 게이트 → decompose → 그래프 → 객체 루프, P3 실패 비전파). SAM 2-pass 정련은 predictor 주입 훅(refine)으로 격리 — 모듈은 torch 무의존.

**Tech Stack:** numpy, scipy, OpenCV(cv2), pytest. 기존 v2 모듈(decomposition, weave_removal_v2, ssei_v2) 소비.

## Global Constraints

- **P-adapt: 튜닝 상수 0.** 모든 수치 리터럴은 ① 수학 유도 ② 안전 상한 ③ 정규화 규칙 중 하나이며, 각 리터럴 옆에 범주와 근거를 한국어 주석으로 명시한다.
- **torch 금지** — `src/kp3d/modules/orchestration/`과 그 테스트는 torch를 임포트하지 않는다. SAM 정련은 호출자가 만든 정련기 객체를 주입받는 훅으로만 존재하고, 실제 SAM 로딩은 데모 스크립트의 guarded import에서만 한다.
- 의존성: numpy, scipy, OpenCV(cv2)만 (테스트 포함).
- 좌표 규약: 마스크·배열은 (y,x). labelme `points`는 (x,y) — 경계에서만 명시 변환.
- 깊이 규약: `depth` 낮을수록 관찰자에 가까운 전경. labelme `layer_order`와 같은 방향. layer_order 부재 시 v1 라벨 규약 fallback: object_2\*(최전방) > object_3 > object_1 > background(최후방).
- 소비 인터페이스 (변경 금지):
  - `kp3d.modules.decomposition.decompose(image_bgr (H,W,3)u8) -> DecompositionResult(line_alpha (H,W)f32, color_layer (H,W,3)u8, line_mask (H,W)bool, skeleton (H,W)bool, width_map (H,W)f32, weave, noise_sigma float)`
  - `kp3d.modules.weave_removal_v2.gate.self_competition_gate(image_bgr) -> GateResult(restored (H,W,3)u8, winner "v2"|"v1", quality_v2, quality_v1, noise_sigma)`
  - `kp3d.modules.ssei_v2.inpaint(image_bgr, color_layer, line_alpha, skeleton, width_map, occlusion_mask (H,W)bool, noise_sigma, visible_mask=None) -> InpaintingResult(inpainted (H,W,3)u8, line, color, g2_tangent_max, g2_curvature_max, by_construction_violations)` — `visible_mask`는 exemplar 소스 허용 영역.
- 주석·docstring은 한국어. 테스트 파일명 `tests/test_orch_*.py`.
- 커밋 메시지: `feat(orchestration): ...` / `test(orchestration): ...`.
- 알파는 이진(0/255) — 선염 경계 closed-form matting은 스펙 §4.1의 후속 플랜 몫.

---

### Task 1: annotations.py — labelme 로딩·부품 병합·깊이·가시성

**Files:**
- Create: `src/kp3d/modules/orchestration/annotations.py`
- Create: `src/kp3d/modules/orchestration/__init__.py` (빈 placeholder — Task 4에서 확정)
- Test: `tests/test_orch_annotations.py`

**Interfaces:**
- Consumes: 없음 (numpy, cv2, re)
- Produces: `ObjectAnnotation(label: str, mask: (H,W)bool, depth: int)`; `load_annotations(shapes: list[dict], image_shape: (h,w)) -> list[ObjectAnnotation]` (depth 오름차순); `resolve_visibility(annotations) -> list[(H,W)bool]` (같은 순서)

- [ ] **Step 1: 실패 테스트 작성**

```python
"""annotations.py 테스트: 부품 병합, depth 규약, 가시성 해소."""
import numpy as np

from kp3d.modules.orchestration.annotations import (
    ObjectAnnotation, load_annotations, resolve_visibility)


def _sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_load_merges_parts_and_orders_by_depth():
    shapes = [
        {"label": "object_1", "shape_type": "polygon", "points": _sq(0, 0, 9, 9)},
        {"label": "object_2_1", "shape_type": "polygon", "points": _sq(20, 0, 29, 9)},
        {"label": "object_2_2", "shape_type": "polygon", "points": _sq(30, 0, 39, 9)},
        {"label": "background", "shape_type": "polygon", "points": _sq(0, 20, 39, 39)},
    ]
    annos = load_annotations(shapes, (40, 40))
    # v1 fallback 서열: object_2(0) < object_1(2) < background(3)
    assert [a.label for a in annos] == ["object_2", "object_1", "background"]
    o2 = annos[0]
    assert o2.mask[5, 25] and o2.mask[5, 35] and not o2.mask[5, 15]


def test_layer_order_field_overrides_fallback():
    shapes = [
        {"label": "object_1", "shape_type": "polygon",
         "points": _sq(0, 0, 9, 9), "layer_order": 1},
        {"label": "object_2_1", "shape_type": "polygon",
         "points": _sq(20, 0, 29, 9), "layer_order": 5},
    ]
    annos = load_annotations(shapes, (40, 40))
    assert [a.label for a in annos] == ["object_1", "object_2"]
    assert [a.depth for a in annos] == [1, 5]


def test_non_polygon_and_degenerate_shapes_skipped():
    shapes = [
        {"label": "object_1", "shape_type": "rectangle", "points": _sq(0, 0, 9, 9)},
        {"label": "object_1", "shape_type": "polygon", "points": [[0, 0], [5, 5]]},
    ]
    assert load_annotations(shapes, (20, 20)) == []


def test_resolve_visibility_removes_nearer_pixels():
    front = ObjectAnnotation("f", np.zeros((10, 10), dtype=bool), 0)
    rear = ObjectAnnotation("r", np.zeros((10, 10), dtype=bool), 1)
    front.mask[2:6, 2:6] = True
    rear.mask[4:9, 4:9] = True
    vis = resolve_visibility([front, rear])
    assert (vis[0] == front.mask).all()            # 전경은 그대로
    assert not (vis[1] & front.mask).any()         # 후방에서 전경 픽셀 제거
    assert vis[1].sum() == rear.mask.sum() - (rear.mask & front.mask).sum()
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_orch_annotations.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/orchestration/__init__.py` (placeholder):

```python
"""v2 객체별 오케스트레이션 (스펙 §4.1/§4.3) — 공개 API는 Task 4에서 확정."""
```

`src/kp3d/modules/orchestration/annotations.py`:

```python
"""labelme 어노테이션 로딩 — 폴리곤→마스크, 부품 병합, 깊이 규약 (스펙 §4.1).

깊이(depth): 낮을수록 관찰자에 가까운 전경 — labelme layer_order와 같은 방향.
layer_order 필드가 없으면 v1 라벨 규약(object_2* 최전방 > object_3 >
object_1 > background)을 따른다 (v1 get_layer_priority 서열 승계).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

# v1 라벨 규약 fallback: object 번호 -> depth (정규화 규칙 — v1 서열 승계)
_LABEL_DEPTH = {2: 0, 3: 1, 1: 2}
# background 는 항상 최후방 (정규화 규칙 — v1 서열 승계)
_BACKGROUND_DEPTH = 3


@dataclass
class ObjectAnnotation:
    """부품 병합이 끝난 객체 하나 (object_2_1 + object_2_2 -> object_2)."""

    label: str        # 기본 라벨
    mask: np.ndarray  # (H,W) bool — 어노테이션 폴리곤 합집합
    depth: int        # 낮을수록 전경


def _base_label(label: str) -> str:
    m = re.match(r"(object_\d+)", label)
    return m.group(1) if m else label


def _fallback_depth(base: str) -> int:
    m = re.match(r"object_(\d+)", base)
    if m:
        return _LABEL_DEPTH.get(int(m.group(1)), _BACKGROUND_DEPTH)
    return _BACKGROUND_DEPTH


def load_annotations(shapes: list[dict],
                     image_shape: tuple[int, int]) -> list[ObjectAnnotation]:
    """labelme shapes를 기본 라벨로 병합한 객체 목록으로 변환 (depth 오름차순)."""
    h, w = image_shape
    masks: dict[str, np.ndarray] = {}
    depths: dict[str, int] = {}
    for s in shapes:
        if s.get("shape_type") != "polygon":
            continue
        pts = np.asarray(s.get("points", ()), dtype=np.float64)
        if pts.ndim != 2 or len(pts) < 3:
            continue  # 폴리곤 최소 꼭짓점 3 (이산 하한)
        m8 = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m8, [np.rint(pts).astype(np.int32)], 1)  # points 는 (x,y)
        base = _base_label(str(s["label"]))
        lo = s.get("layer_order")
        depth = int(lo) if lo is not None else _fallback_depth(base)
        if base in masks:
            masks[base] |= m8.astype(bool)
            depths[base] = min(depths[base], depth)  # 부품 중 최전방 채택
        else:
            masks[base] = m8.astype(bool)
            depths[base] = depth
    annos = [ObjectAnnotation(label=k, mask=masks[k], depth=depths[k])
             for k in masks]
    return sorted(annos, key=lambda a: (a.depth, a.label))


def resolve_visibility(annotations: list[ObjectAnnotation]) -> list[np.ndarray]:
    """객체별 가시 마스크 — 더 앞(depth 작음) 객체가 차지한 픽셀 제거.

    어노테이션이 가시 영역만 그렸든 아모달로 겹치게 그렸든 결과가 같다.
    반환 순서는 annotations 와 동일.
    """
    out: list[np.ndarray] = []
    for a in annotations:
        nearer = np.zeros_like(a.mask)
        for b in annotations:
            if b.depth < a.depth:
                nearer |= b.mask
        out.append(a.mask & ~nearer)
    return out
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_orch_annotations.py -q` / Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/orchestration/__init__.py src/kp3d/modules/orchestration/annotations.py tests/test_orch_annotations.py
git commit -m "feat(orchestration): labelme annotation loading with depth convention"
```

---

### Task 2: graph.py — 가림 그래프와 복원 후보 영역

**Files:**
- Create: `src/kp3d/modules/orchestration/graph.py`
- Test: `tests/test_orch_graph.py`

**Interfaces:**
- Consumes: `ObjectAnnotation` (Task 1)
- Produces: `OcclusionEdge(occluder: str, occludee: str, region: (H,W)bool)`; `derive_dilation_radius(skeleton (H,W)bool, width_map (H,W)f, image_shape) -> int` (>=1); `build_occlusion_graph(annotations, visibles, radius) -> list[OcclusionEdge]` — region = dilate(전경 가시, radius) ∩ convex_hull(후방 가시) − 후방 가시

- [ ] **Step 1: 실패 테스트 작성**

```python
"""graph.py 테스트: 반경 유도, 간선 방향, 후보 영역 기하."""
import cv2
import numpy as np

from kp3d.modules.orchestration.annotations import ObjectAnnotation
from kp3d.modules.orchestration.graph import (
    build_occlusion_graph, derive_dilation_radius)


def test_radius_from_width_p95_and_fallback():
    sk = np.zeros((50, 50), dtype=bool)
    wm = np.zeros((50, 50))
    sk[10, 10:20] = True
    wm[10, 10:20] = 4.0
    assert derive_dilation_radius(sk, wm, (50, 50)) == 4
    empty = np.zeros((200, 200), dtype=bool)
    r = derive_dilation_radius(empty, np.zeros((200, 200)), (200, 200))
    assert r == max(1, round(0.005 * float(np.hypot(200, 200))))


def test_edges_front_to_rear_only_and_region_geometry():
    h = w = 60
    front = ObjectAnnotation("object_2", np.zeros((h, w), dtype=bool), 0)
    rear = ObjectAnnotation("object_1", np.zeros((h, w), dtype=bool), 2)
    front.mask[20:40, 25:35] = True
    rear.mask[25:45, 10:50] = True          # 아모달로 겹치게 그린 어노테이션
    vis = [front.mask, rear.mask & ~front.mask]
    edges = build_occlusion_graph([front, rear], vis, radius=2)
    assert [(e.occluder, e.occludee) for e in edges] == [("object_2", "object_1")]
    e = edges[0]
    assert e.region.any()
    assert not (e.region & vis[1]).any()     # 후방 가시와 배타
    assert (e.region & front.mask).any()     # 전경 아래에 위치
    dil = cv2.dilate(vis[0].astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    assert not (e.region & ~dil.astype(bool)).any()  # 팽창 전경 안에 한정


def test_disjoint_objects_no_edge():
    a = ObjectAnnotation("object_2", np.zeros((40, 40), dtype=bool), 0)
    b = ObjectAnnotation("object_1", np.zeros((40, 40), dtype=bool), 2)
    a.mask[2:8, 2:8] = True
    b.mask[30:38, 30:38] = True
    assert build_occlusion_graph([a, b], [a.mask, b.mask], radius=1) == []
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_orch_graph.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/orchestration/graph.py`:

```python
"""가림 그래프 — depth 서열 + 기하 접촉으로 occluder→occludee 후보 영역 (스펙 §4.3).

후보 영역 = dilate(전경 가시) ∩ convex_hull(후방 가시) − 후방 가시.
- 팽창 반경: 어노테이션 경계 불확실성 ≈ 윤곽선 폭 → 실측 선폭 p95
  (P-adapt: 이미지 통계 유도 — decompose 2-pass 상한과 같은 백분위)
- convex hull: 가시 조각의 아모달 연장 사전(prior) — v1 detect_occlusion 승계
  (정규화 규칙)
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .annotations import ObjectAnnotation


@dataclass
class OcclusionEdge:
    """방향 간선: occluder(전경)가 occludee(후방)를 가린다."""

    occluder: str
    occludee: str
    region: np.ndarray  # (H,W) bool — occludee 복원 후보 영역


def derive_dilation_radius(skeleton: np.ndarray, width_map: np.ndarray,
                           image_shape: tuple[int, int]) -> int:
    """어노테이션 경계 불확실성 반경을 실측 선폭에서 유도한다."""
    widths = np.asarray(width_map)[np.asarray(skeleton, dtype=bool)]
    if widths.size > 0:
        # 선폭 분포 95 백분위 — decompose 2-pass 상한과 동일 (이미지 통계 유도)
        return max(1, int(np.ceil(float(np.percentile(widths, 95)))))
    # 선 부재 시 이미지 대각선 0.005 — decompose sigma_s fallback 동일 (정규화 규칙)
    diag = float(np.hypot(*image_shape))
    return max(1, int(round(0.005 * diag)))


def _convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size < 3:  # hull 정의 최소 점 3 (이산 하한)
        return np.asarray(mask, dtype=bool).copy()
    pts = np.stack([xs, ys], axis=1).astype(np.int32)  # cv2 는 (x,y)
    hull = cv2.convexHull(pts)
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def build_occlusion_graph(annotations: list[ObjectAnnotation],
                          visibles: list[np.ndarray],
                          radius: int) -> list[OcclusionEdge]:
    """depth 가 낮은 객체 → 높은 객체로만 간선 생성 (동률은 간선 없음)."""
    k = 2 * radius + 1  # 반경 → 홀수 커널 변환 (수학 유도)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    edges: list[OcclusionEdge] = []
    for i, front in enumerate(annotations):
        dil = cv2.dilate(visibles[i].astype(np.uint8), kernel).astype(bool)
        for j, rear in enumerate(annotations):
            if rear.depth <= front.depth:
                continue
            region = dil & _convex_hull_mask(visibles[j]) & ~visibles[j]
            if region.any():
                edges.append(OcclusionEdge(occluder=front.label,
                                           occludee=rear.label,
                                           region=region))
    return edges
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_orch_graph.py -q` / Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/orchestration/graph.py tests/test_orch_graph.py
git commit -m "feat(orchestration): occlusion graph with width-derived boundary radius"
```

---

### Task 3: complete.py — 객체별 아모달 완성 (지식 소거 + SSEI 호출)

**Files:**
- Create: `src/kp3d/modules/orchestration/complete.py`
- Test: `tests/test_orch_complete.py`

**Interfaces:**
- Consumes: `kp3d.modules.decomposition.decompose.DecompositionResult`, `kp3d.modules.ssei_v2.inpaint` (Global Constraints의 시그니처)
- Produces: `ObjectCompletion(label: str, rgba: (H,W,4)u8, amodal_mask: (H,W)bool, result: InpaintingResult)`; `complete_object(label, image_bgr, dec, visible (H,W)bool, occ (H,W)bool) -> ObjectCompletion` — 입력 dec 배열은 변형하지 않는다

- [ ] **Step 1: 실패 테스트 작성**

```python
"""complete.py 테스트: RGBA 산출 규약, 입력 불변, 기계 검증."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from kp3d.modules.decomposition import decompose
from kp3d.modules.orchestration.complete import complete_object


def _scene(h=96, w=96):
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[30:70, :, :] = (80, 140, 60)                    # 객체 채색 띠
    cv2.line(img, (5, 50), (90, 50), (25, 25, 25), 3)   # 객체 내부 먹선
    return img


def test_complete_object_rgba_and_no_mutation():
    img = _scene()
    dec = decompose(img)
    occ = np.zeros(img.shape[:2], dtype=bool)
    occ[38:62, 40:56] = True
    band = np.zeros_like(occ)
    band[30:70, :] = True
    visible = band & ~occ
    la0 = dec.line_alpha.copy()
    sk0 = dec.skeleton.copy()
    wm0 = dec.width_map.copy()

    comp = complete_object("object_1", img, dec, visible, occ)

    assert comp.label == "object_1"
    assert (comp.amodal_mask == (visible | occ)).all()
    assert (comp.rgba[..., 3][comp.amodal_mask] == 255).all()
    assert (comp.rgba[..., 3][~comp.amodal_mask] == 0).all()
    assert (comp.rgba[..., :3][comp.amodal_mask]
            == comp.result.inpainted[comp.amodal_mask]).all()
    assert comp.result.by_construction_violations == 0
    # 입력 분해 산출은 불변 (여러 객체가 같은 dec 를 공유한다)
    assert (dec.line_alpha == la0).all()
    assert (dec.skeleton == sk0).all()
    assert (dec.width_map == wm0).all()
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_orch_complete.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 구현**

`src/kp3d/modules/orchestration/complete.py`:

```python
"""객체별 아모달 완성 — 가림 내부 지식 소거 후 SSEI 2.0 호출 (스펙 §4.3).

프로토콜 B와 동형: 가림 후보 영역 안의 선 지식(line_alpha/skeleton/width_map)은
전경 물체의 것이므로 소거한다. 또한 선 입력을 이 객체의 영역(visible ∪ occ)으로
한정해 다른 객체의 선이 연결 후보로 새지 않게 한다. 색 채움은
by-construction(가시 exemplar 볼록 결합)이라 색 레이어 소거는 불필요하다.
알파는 이진 — 선염 경계 matting 은 스펙 §4.1 후속 플랜.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kp3d.modules.decomposition.decompose import DecompositionResult
from kp3d.modules.ssei_v2 import InpaintingResult, inpaint


@dataclass
class ObjectCompletion:
    """아모달 완성 산출 — Stage 4 계약(객체별 RGBA)의 전달물."""

    label: str
    rgba: np.ndarray         # (H,W,4) uint8 (BGR+A) — 아모달 영역만 불투명
    amodal_mask: np.ndarray  # (H,W) bool = visible ∪ occ (hull 사전에 의한 상한)
    result: InpaintingResult


def complete_object(label: str, image_bgr: np.ndarray,
                    dec: DecompositionResult, visible: np.ndarray,
                    occ: np.ndarray) -> ObjectCompletion:
    """한 객체의 가림 후보 영역을 SSEI 2.0으로 복원해 RGBA로 반환한다."""
    visible = np.asarray(visible, dtype=bool)
    occ = np.asarray(occ, dtype=bool) & ~visible
    keep = visible | occ  # 이 객체의 선 완성에 참여하는 영역
    la = dec.line_alpha.copy()
    la[~keep] = 0.0
    la[occ] = 0.0  # 가림 내부 선 지식은 전경 물체의 것 — 소거
    sk = dec.skeleton.copy()
    sk[~keep] = False
    sk[occ] = False
    wm = dec.width_map.copy()
    wm[~keep] = 0.0
    wm[occ] = 0.0
    res = inpaint(image_bgr, dec.color_layer, la, sk, wm, occ,
                  dec.noise_sigma, visible_mask=visible)
    amodal = keep
    rgba = np.zeros((*occ.shape, 4), dtype=np.uint8)
    rgba[..., :3][amodal] = res.inpainted[amodal]
    rgba[..., 3][amodal] = 255
    return ObjectCompletion(label=label, rgba=rgba,
                            amodal_mask=amodal, result=res)
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_orch_complete.py -q` / Expected: 1 passed

- [ ] **Step 5: 커밋**

```bash
git add src/kp3d/modules/orchestration/complete.py tests/test_orch_complete.py
git commit -m "feat(orchestration): per-object amodal completion via SSEI 2.0"
```

---

### Task 4: orchestrate.py — 오케스트레이터 본체 + 공개 API + 통합 정량 테스트

**Files:**
- Create: `src/kp3d/modules/orchestration/orchestrate.py`
- Modify: `src/kp3d/modules/orchestration/__init__.py` (placeholder → 공개 API)
- Test: `tests/test_orch_full.py`

**Interfaces:**
- Consumes: Task 1~3의 전 산출물, `kp3d.modules.weave_removal_v2.gate.self_competition_gate`, `kp3d.modules.decomposition.decompose` (Global Constraints의 시그니처)
- Produces: `OrchestrationResult(restored (H,W,3)u8, winner str, annotations list[ObjectAnnotation], visibles list[(H,W)bool], edges list[OcclusionEdge], completions dict[str, ObjectCompletion], failures dict[str, str])`; `orchestrate(image_bgr, shapes, *, restore: bool = True) -> OrchestrationResult`

- [ ] **Step 1: 실패 테스트 작성**

```python
"""오케스트레이션 통합 — 2객체 합성 장면: 그래프 방향, 아모달 완성 정량(프로토콜 B 동형).

fixture 는 decompose 의 도메인(얇은 어두운 먹 획 + 등휘도 채색 wash + 미세 직조)을
따라야 한다. 수치 검증으로 확정된 제약 3가지:
① 직조(주기 4px) + 약한 노이즈 필수 — 없으면 weave 주기가 구조 크기(60px)로
   오검출되어 RGF sigma_s 가 커지고 3px 먹선이 blob 으로 뭉개진다.
② wash 들은 grayscale 등휘도(채널 평균 동일)여야 한다 — 아니면 wash 경계가
   선으로 분류된다.
③ 가려지는 먹 획의 가시 stub 은 국소 선폭(smear 후 ~20px)보다 길어야 한다 —
   짧으면 stub 양끝이 모두 endpoint 로 검출돼(E=4) 매칭이 퇴화 동률로 전원
   종결된다. 또한 획에 모서리가 닿으면 medial-axis 대각 가지가 끝점 접선을
   오염시키므로 모서리 없는 일자 획을 쓴다.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scipy.ndimage import binary_dilation

from kp3d.modules.decomposition import decompose
from kp3d.modules.orchestration import orchestrate


def _scene(h=160, w=160, with_disk=True, seed=0):
    """등휘도 wash 배경/사각 + 수평 먹 획(y=90) + 전경 원판(먹 윤곽) + 직조/노이즈."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (60, 170, 60)                                    # 배경 wash
    cv2.rectangle(img, (10, 60), (150, 120), (60, 60, 170), -1)  # 후방 사각 wash
    cv2.line(img, (15, 90), (145, 90), (20, 20, 20), 3)          # 사각 내부 수평 먹 획
    if with_disk:
        cv2.circle(img, (80, 90), 24, (170, 60, 60), -1)         # 전경 원판 wash
        cv2.circle(img, (80, 90), 24, (20, 20, 20), 3)           # 원판 먹 윤곽
    out = img.astype(np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    weave = 6.0 * (np.sin(2 * np.pi * xx / 4.0) + np.sin(2 * np.pi * yy / 4.0))
    noise = np.random.default_rng(seed).normal(0.0, 2.0, (h, w))
    out += (weave + noise)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _shapes():
    ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    circle = [[80.0 + 27.0 * np.cos(a), 90.0 + 27.0 * np.sin(a)] for a in ang]
    return [
        {"label": "object_2_1", "points": circle,
         "shape_type": "polygon", "layer_order": 1},
        {"label": "object_1",
         "points": [[10, 60], [150, 60], [150, 120], [10, 120]],
         "shape_type": "polygon", "layer_order": 2},
    ]


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)


def test_orchestrate_graph_direction_and_amodal_quality():
    img = _scene()
    gt = _scene(with_disk=False)  # 원판이 없는 장면 = object_1 의 GT

    res = orchestrate(img, _shapes(), restore=False)

    assert res.winner == "none"
    assert res.failures == {}
    # 그래프: 전경 원판(object_2)이 후방 사각(object_1)을 가린다 — 이 간선뿐
    assert [(e.occluder, e.occludee) for e in res.edges] == [("object_2", "object_1")]
    assert "object_2" not in res.completions  # 최전방 객체는 가려지지 않는다
    assert set(res.completions) == {"object_1"}

    comp = res.completions["object_1"]
    occ = res.edges[0].region
    # annotations 는 depth 오름차순 — [object_2, object_1]
    vis_rect = res.visibles[1]

    # ① PSNR(가림 후보 내부): 가시 평균색 채움 베이스라인 초과
    mean_col = np.rint(img[vis_rect].reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    base = gt.copy(); base[occ] = mean_col
    assert _psnr(comp.result.inpainted[occ], gt[occ]) > _psnr(base[occ], gt[occ])

    # ② 스켈레톤 재현율(1px 팽창 허용): 수평 먹 획 자취 회복
    gt_sk = decompose(gt).skeleton & occ
    assert np.any(gt_sk)  # 획 중앙이 원판에 가려져 있어야 실험이 성립
    cover = binary_dilation(comp.result.line.skeleton, iterations=1)
    recall = (float(np.count_nonzero(gt_sk & cover))
              / float(np.count_nonzero(gt_sk)))
    assert recall > 0.7

    # ③ 기계 검증 지표
    assert comp.result.by_construction_violations == 0
    assert comp.result.g2_tangent_max < 1e-6
    assert comp.result.g2_curvature_max < 1e-6
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_orch_full.py -q` / Expected: FAIL (ImportError: cannot import name 'orchestrate')

- [ ] **Step 3: 구현**

`src/kp3d/modules/orchestration/orchestrate.py`:

```python
"""Stage 2 오케스트레이터 — 게이트 → 분해 → 가림 그래프 → 객체별 아모달 완성 (스펙 §4.3).

한 객체의 완성 실패는 다른 객체로 전파하지 않는다(P3) — failures 에 기록하고 계속.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kp3d.modules.decomposition import decompose
from kp3d.modules.weave_removal_v2.gate import self_competition_gate

from .annotations import ObjectAnnotation, load_annotations, resolve_visibility
from .complete import ObjectCompletion, complete_object
from .graph import OcclusionEdge, build_occlusion_graph, derive_dilation_radius


@dataclass
class OrchestrationResult:
    """Stage 2 산출 — Stage 4 계약(객체별 RGBA + 가림 그래프)의 전달물."""

    restored: np.ndarray                      # (H,W,3) u8 — 게이트 통과 복원 R
    winner: str                               # "v2" | "v1" | "none"(restore=False)
    annotations: list[ObjectAnnotation]       # depth 오름차순
    visibles: list[np.ndarray]                # annotations 와 같은 순서
    edges: list[OcclusionEdge]
    completions: dict[str, ObjectCompletion]  # occludee label → 아모달 완성
    failures: dict[str, str]                  # occludee label → 예외 요약


def orchestrate(image_bgr: np.ndarray, shapes: list[dict], *,
                restore: bool = True) -> OrchestrationResult:
    """작품 이미지 + labelme shapes → 객체별 아모달 완성."""
    img = np.asarray(image_bgr)
    if restore:
        gate = self_competition_gate(img)
        restored, winner = gate.restored, gate.winner
    else:
        restored, winner = img.copy(), "none"

    dec = decompose(restored)  # 전 객체가 공유하는 단일 분해 (dec 는 불변으로 취급)
    annotations = load_annotations(shapes, restored.shape[:2])
    visibles = resolve_visibility(annotations)
    radius = derive_dilation_radius(dec.skeleton, dec.width_map,
                                    restored.shape[:2])
    edges = build_occlusion_graph(annotations, visibles, radius)

    # occludee 별 가림 후보 합집합 — 여러 전경에 동시에 가린 객체 대응
    occ_by_label: dict[str, np.ndarray] = {}
    for e in edges:
        acc = occ_by_label.get(e.occludee)
        occ_by_label[e.occludee] = e.region if acc is None else (acc | e.region)

    vis_by_label = {a.label: v for a, v in zip(annotations, visibles)}
    completions: dict[str, ObjectCompletion] = {}
    failures: dict[str, str] = {}
    for label, occ in occ_by_label.items():
        try:
            completions[label] = complete_object(label, restored, dec,
                                                 vis_by_label[label], occ)
        except Exception as exc:  # P3: 실패 객체 격리 — 나머지는 계속
            failures[label] = f"{type(exc).__name__}: {exc}"

    return OrchestrationResult(restored=restored, winner=winner,
                               annotations=annotations, visibles=visibles,
                               edges=edges, completions=completions,
                               failures=failures)
```

`src/kp3d/modules/orchestration/__init__.py` (placeholder 전체 교체):

```python
"""Stage 2 오케스트레이션 — labelme 어노테이션 기반 객체별 아모달 완성 공개 API."""
from .annotations import ObjectAnnotation, load_annotations, resolve_visibility
from .complete import ObjectCompletion, complete_object
from .graph import OcclusionEdge, build_occlusion_graph, derive_dilation_radius
from .orchestrate import OrchestrationResult, orchestrate

__all__ = [
    "ObjectAnnotation", "load_annotations", "resolve_visibility",
    "OcclusionEdge", "build_occlusion_graph", "derive_dilation_radius",
    "ObjectCompletion", "complete_object",
    "OrchestrationResult", "orchestrate",
]
```

- [ ] **Step 4: 통과 확인** — Run: `python -m pytest tests/test_orch_full.py -q` / Expected: 1 passed (통합 검증 태스크 — 실패하면 superpowers:systematic-debugging 으로 해당 모듈 결함을 추적, 임계값 완화 금지)

- [ ] **Step 5: 회귀 확인** — Run: `python -m pytest tests/test_orch_*.py -q` / Expected: 전부 passed

- [ ] **Step 6: 커밋**

```bash
git add src/kp3d/modules/orchestration/orchestrate.py src/kp3d/modules/orchestration/__init__.py tests/test_orch_full.py
git commit -m "feat(orchestration): stage 2 orchestrator with per-object amodal completion"
```

---

### Task 5: refine.py — SAM 정련 훅(주입식) + orchestrate 연결 + 데모 스크립트

**Files:**
- Create: `src/kp3d/modules/orchestration/refine.py`
- Modify: `src/kp3d/modules/orchestration/orchestrate.py` (`refiner` 파라미터 추가)
- Modify: `src/kp3d/modules/orchestration/__init__.py` (refine export 추가)
- Create: `scripts/demo_orchestration_v2.py`
- Test: `tests/test_orch_refine.py`

**Interfaces:**
- Consumes: v1 `SAMMaskRefiner`(`kp3d.modules.occlusion.sam_mask_refiner`) 계약 — `refine_mask(image: (H,W,3)u8 RGB, rough_mask: (H,W)u8 0/255, label: str, *, set_image: bool) -> (H,W) mask` (duck typing — torch 는 이 모듈에서 import 하지 않는다)
- Produces: `refine_annotations(image_rgb, annotations, refiner) -> list[ObjectAnnotation]` (라벨·depth 보존, mask 만 교체; 첫 호출만 `set_image=True`); `orchestrate(..., refiner=None)` — refiner 가 주어지면 어노테이션 정련 후 그래프 계산

- [ ] **Step 1: 실패 테스트 작성**

```python
"""refine.py 테스트: 주입 정련기 호출 규약(set_image 1회), 라벨·depth 보존."""
import numpy as np

from kp3d.modules.orchestration.annotations import ObjectAnnotation
from kp3d.modules.orchestration.refine import refine_annotations


class _StubRefiner:
    """호출 기록 스텁 — SAM 없이 계약만 검증한다."""

    def __init__(self):
        self.calls = []

    def refine_mask(self, image, rough_mask, label, *, set_image):
        self.calls.append((label, bool(set_image)))
        out = np.asarray(rough_mask, dtype=bool).copy()
        out[0, 0] = True  # 정련이 마스크를 실제로 바꾸는지 관측용
        return out


def test_refine_annotations_contract():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    m1 = np.zeros((20, 20), dtype=bool); m1[2:8, 2:8] = True
    m2 = np.zeros((20, 20), dtype=bool); m2[10:18, 10:18] = True
    annos = [ObjectAnnotation(label="object_2", mask=m1, depth=0),
             ObjectAnnotation(label="object_1", mask=m2, depth=2)]
    ref = _StubRefiner()

    out = refine_annotations(img, annos, ref)

    # 첫 호출만 set_image=True — SAM 이미지 임베딩 1회 재사용 계약
    assert ref.calls == [("object_2", True), ("object_1", False)]
    assert [a.label for a in out] == ["object_2", "object_1"]
    assert [a.depth for a in out] == [0, 2]
    assert out[0].mask[0, 0] and out[1].mask[0, 0]  # 정련 결과가 반영됐다
    assert not annos[0].mask[0, 0]  # 입력 어노테이션은 불변
```

- [ ] **Step 2: 실패 확인** — Run: `python -m pytest tests/test_orch_refine.py -q` / Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: refine.py 구현**

`src/kp3d/modules/orchestration/refine.py`:

```python
"""어노테이션 경계 정련 훅 — SAM 등 외부 정련기를 주입받는다 (스펙 §4.1).

torch 를 import 하지 않는다: 정련기는 duck-typed 계약
`refine_mask(image_rgb, rough_mask u8 0/255, label, *, set_image) -> mask`
만 만족하면 된다 (v1 SAMMaskRefiner 가 이 계약을 이미 구현).
"""
from __future__ import annotations

import numpy as np

from .annotations import ObjectAnnotation


def refine_annotations(image_rgb: np.ndarray,
                       annotations: list[ObjectAnnotation],
                       refiner) -> list[ObjectAnnotation]:
    """각 어노테이션 마스크를 정련기로 교체한다 — 라벨·depth 보존.

    같은 이미지의 임베딩을 재사용하도록 첫 호출만 set_image=True 로 부른다.
    """
    out: list[ObjectAnnotation] = []
    for i, a in enumerate(annotations):
        rough = a.mask.astype(np.uint8) * 255  # v1 계약: u8 0/255 (정규화 규칙)
        refined = refiner.refine_mask(image_rgb, rough, a.label,
                                      set_image=(i == 0))
        out.append(ObjectAnnotation(label=a.label,
                                    mask=np.asarray(refined, dtype=bool),
                                    depth=a.depth))
    return out
```

- [ ] **Step 4: orchestrate.py 에 refiner 연결** — `orchestrate` 함수를 아래 전체로 교체 (import 에 `import cv2` 와 `from .refine import refine_annotations` 추가):

```python
def orchestrate(image_bgr: np.ndarray, shapes: list[dict], *,
                restore: bool = True, refiner=None) -> OrchestrationResult:
    """작품 이미지 + labelme shapes → 객체별 아모달 완성.

    refiner 가 주어지면(예: v1 SAMMaskRefiner) 어노테이션 마스크를
    정련한 뒤 그래프를 계산한다. 계약은 refine.refine_annotations 참조.
    """
    img = np.asarray(image_bgr)
    if restore:
        gate = self_competition_gate(img)
        restored, winner = gate.restored, gate.winner
    else:
        restored, winner = img.copy(), "none"

    dec = decompose(restored)  # 전 객체가 공유하는 단일 분해 (dec 는 불변으로 취급)
    annotations = load_annotations(shapes, restored.shape[:2])
    if refiner is not None:
        rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)  # SAM 계약은 RGB
        annotations = refine_annotations(rgb, annotations, refiner)
    visibles = resolve_visibility(annotations)
    radius = derive_dilation_radius(dec.skeleton, dec.width_map,
                                    restored.shape[:2])
    edges = build_occlusion_graph(annotations, visibles, radius)

    # occludee 별 가림 후보 합집합 — 여러 전경에 동시에 가린 객체 대응
    occ_by_label: dict[str, np.ndarray] = {}
    for e in edges:
        acc = occ_by_label.get(e.occludee)
        occ_by_label[e.occludee] = e.region if acc is None else (acc | e.region)

    vis_by_label = {a.label: v for a, v in zip(annotations, visibles)}
    completions: dict[str, ObjectCompletion] = {}
    failures: dict[str, str] = {}
    for label, occ in occ_by_label.items():
        try:
            completions[label] = complete_object(label, restored, dec,
                                                 vis_by_label[label], occ)
        except Exception as exc:  # P3: 실패 객체 격리 — 나머지는 계속
            failures[label] = f"{type(exc).__name__}: {exc}"

    return OrchestrationResult(restored=restored, winner=winner,
                               annotations=annotations, visibles=visibles,
                               edges=edges, completions=completions,
                               failures=failures)
```

`__init__.py` 에 export 추가:

```python
from .refine import refine_annotations
```

(그리고 `__all__` 에 `"refine_annotations"` 추가)

- [ ] **Step 5: 통과 확인** — Run: `python -m pytest tests/test_orch_refine.py tests/test_orch_full.py -q` / Expected: 2 passed

- [ ] **Step 6: 데모 스크립트 작성**

`scripts/demo_orchestration_v2.py`:

```python
"""Stage 2 오케스트레이션 데모: 실제 작품 + labelme 어노테이션 → 객체별 RGBA.

실행: python scripts/demo_orchestration_v2.py [stem] [--sam]
stem 기본 1_0004 (data_original_painting/target_data/<stem>.png + .json).
--sam: ~/.cache/sam/sam_vit_h.pth 가 있으면 SAM 정련 훅을 주입 (없으면 경고 후 생략).
산출: outputs/orchestration_v2/<stem>/ 아래 restored.png + <label>_rgba.png.
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kp3d.modules.orchestration import orchestrate  # noqa: E402

DATA_DIR = Path("data_original_painting/target_data")


def _try_build_refiner():
    """SAM 체크포인트가 있을 때만 v1 정련기를 구성한다 (guarded import)."""
    ckpt = Path.home() / ".cache" / "sam" / "sam_vit_h.pth"
    if not ckpt.exists():
        print(f"경고: SAM 체크포인트 없음({ckpt}) — 정련 생략")
        return None
    try:
        from segment_anything import SamPredictor, sam_model_registry
        from kp3d.modules.occlusion.sam_mask_refiner import SAMMaskRefiner
    except ImportError as exc:
        print(f"경고: SAM 의존성 없음({exc}) — 정련 생략")
        return None
    sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
    return SAMMaskRefiner(sam_predictor=SamPredictor(sam))


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--sam"]
    stem = args[0] if args else "1_0004"
    use_sam = "--sam" in sys.argv[1:]

    img = cv2.imread(str(DATA_DIR / f"{stem}.png"))
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없음: {DATA_DIR / f'{stem}.png'}")
    with open(DATA_DIR / f"{stem}.json", encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]

    refiner = _try_build_refiner() if use_sam else None
    res = orchestrate(img, shapes, refiner=refiner)

    out_dir = Path("outputs/orchestration_v2") / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "restored.png"), res.restored)
    for label, comp in res.completions.items():
        cv2.imwrite(str(out_dir / f"{label}_rgba.png"), comp.rgba)

    print(f"{stem}: 게이트 승자 {res.winner}, 객체 {len(res.annotations)}개, "
          f"간선 {len(res.edges)}건")
    for e in res.edges:
        print(f"  {e.occluder} → {e.occludee} (면적 {int(e.region.sum())}px)")
    for label, comp in res.completions.items():
        r = comp.result
        print(f"  {label}: 연결 {len(r.line.connections)}건, "
              f"G2 {r.g2_tangent_max:.2e}/{r.g2_curvature_max:.2e}, "
              f"위반 {r.by_construction_violations}건")
    for label, msg in res.failures.items():
        print(f"  {label}: 실패 — {msg}")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 데모 스모크 실행** — Run: `python scripts/demo_orchestration_v2.py` (--sam 없이) / Expected: outputs/orchestration_v2/1_0004/ 에 restored.png + 완성 객체별 `<label>_rgba.png`, 실패 0건 출력

- [ ] **Step 8: 전체 스위트 확인** — Run: `python -m pytest tests/test_orch_*.py tests/test_ssei_*.py tests/test_decomp_*.py tests/test_wr2_*.py -q` / Expected: 전부 passed

- [ ] **Step 9: 커밋**

```bash
git add src/kp3d/modules/orchestration/refine.py src/kp3d/modules/orchestration/orchestrate.py src/kp3d/modules/orchestration/__init__.py tests/test_orch_refine.py scripts/demo_orchestration_v2.py
git commit -m "feat(orchestration): injectable SAM refinement hook and real-data demo"
```

