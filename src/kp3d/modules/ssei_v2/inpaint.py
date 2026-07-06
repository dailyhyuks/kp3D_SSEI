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
