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
