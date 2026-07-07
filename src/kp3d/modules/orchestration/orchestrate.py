"""Stage 2 오케스트레이터 — 게이트 → 분해 → 가림 그래프 → 객체별 아모달 완성 (스펙 §4.3).

한 객체의 완성 실패는 다른 객체로 전파하지 않는다(P3) — failures 에 기록하고 계속.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kp3d.modules.decomposition import decompose
from kp3d.modules.weave_removal_v2.gate import self_competition_gate

from .annotations import ObjectAnnotation, load_annotations, resolve_visibility
from .complete import ObjectCompletion, complete_object
from .graph import OcclusionEdge, build_occlusion_graph, derive_dilation_radius
from .refine import refine_annotations


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
