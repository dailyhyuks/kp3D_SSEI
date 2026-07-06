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
