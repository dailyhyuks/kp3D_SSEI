"""Phase B 색 채움 — Wexler voting + multi-scale EM + by-construction (스펙 §3.3–3.4, §5.1)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from .patchmatch import _MAX_ITERS, derive_patch_size, patchmatch
from .pool import PiecePool, build_piece_pools

# 8-bit 반 단위(정규화) — 양자화 반 스텝 (수학 유도)
_HALF_LEVEL = 0.5 / 255.0
# Wexler 가중 스케일 백분위 P75 — Wexler 관례 (정규화 규칙)
_P_SIGMA = 75.0
# 8-근방 — 이산 위상 (수학 유도)
_N8 = np.ones((3, 3), dtype=bool)


@dataclass
class ColorFillResult:
    """Phase B 산출."""

    filled: np.ndarray      # (H,W,3) uint8 BGR
    pieces: list[PiecePool]
    by_construction_violations: int
    patch_size: int
    levels: int             # 사용된 최대 피라미드 레벨 수


def _levels_for(piece: np.ndarray, p: int) -> int:
    """레벨 수 = ceil(log2(결손 지름 / 패치)) — 스케일 커버리지 (수학 유도)."""
    rmax = float(distance_transform_edt(piece).max())
    return max(1, int(np.ceil(np.log2(max(2.0 * rmax / p, 1.0)))))


def _onion_peel(img: np.ndarray, hole: np.ndarray) -> None:
    """구멍을 경계층부터 이웃 평균으로 반복 초기화 (in-place, C2)."""
    rem = hole.copy()
    while np.any(rem):
        ring = rem & binary_dilation(~rem, structure=_N8)
        if not np.any(ring):
            break
        known = (~rem).astype(np.float64)
        cnt = cv2.blur(known, (3, 3)) * 9.0
        acc = cv2.blur(img * known[..., None], (3, 3)) * 9.0
        ys, xs = np.nonzero(ring)
        img[ys, xs] = acc[ys, xs] / np.maximum(cnt[ys, xs], 1.0)[..., None]
        rem[ys, xs] = False


def _vote(img, hole, tmask, nnf, dists, p, collect_minmax):
    """가중 Wexler voting으로 hole 픽셀 갱신 → (평균 변화량, vmin, vmax)."""
    h, w, _ = img.shape
    r = p // 2
    dv = dists[tmask]
    sigma = max(float(np.percentile(dv, _P_SIGMA)),
                float(np.finfo(np.float64).tiny))
    acc = np.zeros_like(img)
    wgt = np.zeros((h, w))
    vmin = np.full_like(img, np.inf) if collect_minmax else None
    vmax = np.full_like(img, -np.inf) if collect_minmax else None
    for ty, tx in np.argwhere(tmask):
        wc = float(np.exp(-float(dists[ty, tx]) ** 2 / (2.0 * sigma ** 2)))
        sy, sx = int(nnf[ty, tx, 0]), int(nnf[ty, tx, 1])
        src = img[sy - r:sy + r + 1, sx - r:sx + r + 1]
        sub = hole[ty - r:ty + r + 1, tx - r:tx + r + 1]
        acc[ty - r:ty + r + 1, tx - r:tx + r + 1] += wc * src * sub[..., None]
        wgt[ty - r:ty + r + 1, tx - r:tx + r + 1] += wc * sub
        if collect_minmax:
            win = (slice(ty - r, ty + r + 1), slice(tx - r, tx + r + 1))
            np.minimum(vmin[win], np.where(sub[..., None], src, np.inf),
                       out=vmin[win])
            np.maximum(vmax[win], np.where(sub[..., None], src, -np.inf),
                       out=vmax[win])
    upd = hole & (wgt > 0.0)
    if not np.any(upd):
        return 0.0, vmin, vmax
    old = img[upd].copy()
    img[upd] = acc[upd] / wgt[upd][..., None]
    return float(np.abs(img[upd] - old).mean()), vmin, vmax


def _fill_piece(img: np.ndarray, piece: np.ndarray, pool_mask: np.ndarray,
                p: int, sigma_n: float) -> tuple[int, int]:
    """한 조각을 multi-scale EM으로 채운다 (img in-place) → (위반 수, 레벨 수)."""
    levels = _levels_for(piece, p)
    imgs, holes, pools = [img], [piece], [pool_mask]
    for _ in range(levels - 1):
        prev = imgs[-1]
        nh, nw = (prev.shape[0] + 1) // 2, (prev.shape[1] + 1) // 2
        imgs.append(cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA))
        holes.append(cv2.resize(holes[-1].astype(np.uint8), (nw, nh),
                                interpolation=cv2.INTER_NEAREST).astype(bool))
        pools.append(cv2.resize(pools[-1].astype(np.uint8), (nw, nh),
                                interpolation=cv2.INTER_NEAREST).astype(bool))
    violations = 0
    square = np.ones((p, p), dtype=bool)
    r = p // 2
    for lv in range(levels - 1, -1, -1):  # 거친 → 미세 (C2: 스케일 순서)
        im, hl = imgs[lv], holes[lv]
        if not np.any(hl):
            continue
        hh, ww = hl.shape
        # 레벨별 δ_safe 재적용 (창 in-bounds + hole 비겹침)
        safe = binary_erosion(~hl, structure=square)
        pl = pools[lv] & safe
        if not np.any(pl):
            pl = safe  # 전역 fallback (스펙 §3.6)
        if not np.any(pl):
            continue  # 레벨이 너무 거칢 — 미세 레벨에서 처리
        inb = np.zeros_like(hl)
        inb[r:hh - r, r:ww - r] = True
        tmask = binary_dilation(hl, structure=square) & inb
        if not np.any(tmask):
            continue
        if lv == levels - 1:
            _onion_peel(im, hl)  # 최심 레벨 초기화 (C2)
        vmin = vmax = None
        for _ in range(_MAX_ITERS):  # EM 안전 상한
            nnf, dists = patchmatch(im, tmask, pl, p, sigma_n)
            change, vmin, vmax = _vote(im, hl, tmask, nnf, dists, p,
                                       collect_minmax=(lv == 0))
            if change < sigma_n:
                break  # 잡음 바닥 이하 변화 — 수렴 (P-adapt)
        if lv > 0:
            fine = imgs[lv - 1]
            up = cv2.resize(im, (fine.shape[1], fine.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
            fh = holes[lv - 1]
            fine[fh] = up[fh]  # 다음(미세) 레벨 초기화
        elif vmin is not None:
            # by construction: 기여 exemplar [min,max] ± 반 단위 (스펙 §5.1)
            sel = hl[..., None] & np.isfinite(vmin)
            lo = np.where(np.isfinite(vmin), vmin - _HALF_LEVEL, 0.0)
            hi = np.where(np.isfinite(vmax), vmax + _HALF_LEVEL, 1.0)
            bad = sel & ((im < lo) | (im > hi))
            violations = int(np.count_nonzero(np.any(bad, axis=-1)))
            im[:] = np.where(sel, np.clip(im, lo, hi), im)
    return violations, levels


def fill_color(color_layer: np.ndarray, occlusion_mask: np.ndarray,
               line_mask: np.ndarray, visible_mask: np.ndarray,
               noise_sigma: float) -> ColorFillResult:
    """Phase B 진입점: 조각별 선 위상 제약 exemplar 채움.

    noise_sigma는 Stage 0 decompose()의 0..255 스케일 — 내부 /255 정규화.
    """
    occ = np.asarray(occlusion_mask, dtype=bool)
    img8 = np.asarray(color_layer, dtype=np.uint8)
    visible = np.asarray(visible_mask, dtype=bool)
    gray = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY).astype(np.float64)
    p = derive_patch_size(gray, visible & ~occ)
    pieces = build_piece_pools(img8, occ, np.asarray(line_mask, dtype=bool),
                               visible, p)
    img = img8.astype(np.float64) / 255.0
    sigma_n = float(noise_sigma) / 255.0
    total_violations = 0
    max_levels = 1
    for pp in pieces:
        v, lv = _fill_piece(img, pp.piece_mask, pp.pool_mask, p, sigma_n)
        total_violations += v
        max_levels = max(max_levels, lv)
    filled = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    filled[~occ] = img8[~occ]  # 가시 픽셀 원본 보존
    return ColorFillResult(filled=filled, pieces=pieces,
                           by_construction_violations=total_violations,
                           patch_size=p, levels=max_levels)
