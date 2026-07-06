"""조각 분할과 선 위상 제약 exemplar pool (스펙 §3.3 ①②, §3.6).

가림 영역을 완성된 선으로 분할한 조각마다, 선 장벽을 넘지 않고 도달
가능한(체비쇼프 반복 팽창) 가시 픽셀만 exemplar 후보로 삼는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt, label)

# 8-근방(체비쇼프 1) 구조 원소 — 이산 위상 (수학 유도)
_N8 = np.ones((3, 3), dtype=bool)
# 배가 확장 계수 ×2 — 지수 탐색 관례 (정규화 규칙)
_GROWTH = 2


def _initial_dmax(shape: tuple[int, int]) -> int:
    """d_max 초기값 — v1 해상도 적응 규칙 (스펙 §3.5 승계)."""
    m = max(shape[0], shape[1])
    if m < 200:
        return 15
    if m < 400:
        return 25
    return 40


@dataclass
class PiecePool:
    """조각과 그 exemplar 패치 중심 후보."""

    piece_mask: np.ndarray  # (H,W) bool — 채울 조각
    pool_mask: np.ndarray   # (H,W) bool — 유효 패치 중심(δ_safe 적용)
    borrowed: bool          # 고립 조각의 타 조각/전역 pool 차용 여부


def _split_pieces(occ: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, int]:
    """가림을 선으로 분할. 선이 덮은 hole 픽셀은 최근접 조각에 귀속."""
    core = occ & ~lines
    labels, n = label(core, structure=_N8)
    covered = occ & lines
    if np.any(covered) and n > 0:
        # 최근접 조각 픽셀 인덱스 귀속 — 거리 변환 (순수 수학)
        _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
        labels = np.where(covered, labels[iy, ix], labels)
    return labels, n


def build_piece_pools(color_layer: np.ndarray, occlusion_mask: np.ndarray,
                      line_mask: np.ndarray, visible_mask: np.ndarray,
                      patch_size: int) -> list[PiecePool]:
    """조각별 exemplar pool을 만든다. pool이 빈 고립 조각은 차용한다."""
    occ = np.asarray(occlusion_mask, dtype=bool)
    lines = np.asarray(line_mask, dtype=bool)
    visible = np.asarray(visible_mask, dtype=bool) & ~occ
    color = np.asarray(color_layer, dtype=np.float64)
    h, w = occ.shape
    # δ_safe: 패치 창 p×p가 가림·이미지 경계와 겹치지 않는 중심만
    # (경계 처리: erosion 기본 border_value=0이 창의 in-bounds를 보장)
    safe = binary_erosion(
        ~occ, structure=np.ones((patch_size, patch_size), dtype=bool))
    labels, n = _split_pieces(occ, lines)
    diag = int(np.ceil(np.hypot(h, w)))  # 확장 상한: 이미지 대각선 (안전 상한)
    pools: list[PiecePool] = []
    for k in range(1, n + 1):
        piece = labels == k
        dmax = _initial_dmax((h, w))
        prev_grown = None
        while True:
            # 선 장벽을 넘지 않는 반복 팽창 (체비쇼프 거리 dmax)
            grown = binary_dilation(piece & ~lines, structure=_N8, iterations=dmax,
                                    mask=~lines)
            pool = grown & visible & safe
            if int(pool.sum()) >= int(piece.sum()) or dmax >= diag:
                break
            if prev_grown is not None and np.array_equal(grown, prev_grown):
                break  # 성장 포화 — 더 확장해도 도달 픽셀 없음
            prev_grown = grown
            dmax *= _GROWTH  # 배가 확장 (정규화 규칙)
        pools.append(PiecePool(piece_mask=piece, pool_mask=pool,
                               borrowed=False))
    # 고립 조각: 평균색 최근접 타 조각 pool 차용 → 전역 pool fallback (스펙 §3.6)
    global_pool = visible & safe
    for pp in pools:
        if np.any(pp.pool_mask):
            continue
        ring = binary_dilation(pp.piece_mask, structure=_N8,
                               iterations=_initial_dmax((h, w))) & visible
        best = None
        if np.any(ring):
            ref = color[ring].reshape(-1, 3).mean(axis=0)
            cands = [q for q in pools if q is not pp and np.any(q.pool_mask)]
            if cands:
                dists = [float(np.linalg.norm(
                    color[q.pool_mask].reshape(-1, 3).mean(axis=0) - ref))
                    for q in cands]
                best = cands[int(np.argmin(dists))]
        pp.pool_mask = (best.pool_mask.copy() if best is not None
                        else global_pool.copy())
        pp.borrowed = True
    return pools
