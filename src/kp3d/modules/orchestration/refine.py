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
