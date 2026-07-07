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
