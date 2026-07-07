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
