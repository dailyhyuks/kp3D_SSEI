"""pool.py 테스트: 선 위상 분할, δ_safe 중심, 고립 조각 차용."""
import numpy as np

from kp3d.modules.ssei_v2.pool import _initial_dmax, build_piece_pools


def test_initial_dmax_spec_rule():
    # v1 해상도 적응 규칙 (스펙 §3.5 승계)
    assert _initial_dmax((100, 150)) == 15
    assert _initial_dmax((300, 200)) == 25
    assert _initial_dmax((500, 400)) == 40


def _base(h=64, w=64):
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:32] = (0, 200, 0)
    color[32:] = (0, 0, 200)
    occ = np.zeros((h, w), dtype=bool)
    occ[20:44, 20:44] = True
    visible = np.ones((h, w), dtype=bool)
    return color, occ, visible


def test_line_splits_pieces_and_pools():
    color, occ, visible = _base()
    lines = np.zeros(occ.shape, dtype=bool)
    lines[32, :] = True
    pools = build_piece_pools(color, occ, lines, visible, patch_size=5)
    assert len(pools) == 2
    top = min(pools, key=lambda p: np.argwhere(p.piece_mask)[:, 0].mean())
    bot = max(pools, key=lambda p: np.argwhere(p.piece_mask)[:, 0].mean())
    assert not top.borrowed and not bot.borrowed
    # 선 장벽: pool이 반대편으로 새지 않는다
    assert np.argwhere(top.pool_mask)[:, 0].max() < 32
    assert np.argwhere(bot.pool_mask)[:, 0].min() > 32
    # 조각 합집합이 가림 전체(선이 덮은 픽셀 포함)를 덮는다
    assert np.array_equal(top.piece_mask | bot.piece_mask, occ)


def test_pool_centers_are_patch_safe():
    color, occ, visible = _base()
    lines = np.zeros(occ.shape, dtype=bool)
    lines[32, :] = True
    p = 5
    r = p // 2
    pools = build_piece_pools(color, occ, lines, visible, patch_size=p)
    for pp in pools:
        for y, x in np.argwhere(pp.pool_mask):
            win = occ[y - r:y + r + 1, x - r:x + r + 1]
            assert win.shape == (p, p) and not win.any()


def test_isolated_piece_borrows():
    color, occ, visible = _base()
    lines = np.zeros(occ.shape, dtype=bool)
    yy, xx = np.mgrid[0:64, 0:64]
    lines |= np.abs(np.hypot(yy - 32, xx - 32) - 8.0) < 1.0  # 폐곡선(원환) 선
    pools = build_piece_pools(color, occ, lines, visible, patch_size=5)
    inner = [p for p in pools if p.piece_mask[32, 32]]
    assert len(inner) == 1
    assert inner[0].borrowed
    assert inner[0].pool_mask.any()
