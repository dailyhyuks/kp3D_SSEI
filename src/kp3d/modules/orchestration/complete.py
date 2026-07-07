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
