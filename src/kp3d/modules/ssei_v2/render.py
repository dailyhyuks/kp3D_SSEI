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
