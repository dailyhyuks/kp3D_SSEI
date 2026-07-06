"""fill.py 테스트: 선 위상 존중 채움, 가시 보존, by-construction 불변식."""
import numpy as np

from kp3d.modules.ssei_v2.fill import fill_color


def _two_region_scene(h=64, w=64):
    color = np.zeros((h, w, 3), dtype=np.int64)
    color[:32] = (0, 180, 0)   # BGR — 위 녹색
    color[32:] = (0, 0, 180)   # 아래 적색
    rng = np.random.default_rng(1)
    color = np.clip(color + rng.integers(-15, 16, color.shape), 0, 255)
    color = color.astype(np.uint8)
    lines = np.zeros((h, w), dtype=bool)
    lines[32, :] = True
    occ = np.zeros((h, w), dtype=bool)
    occ[24:40, 24:40] = True
    visible = np.ones((h, w), dtype=bool)
    return color, occ, lines, visible


def test_fill_respects_line_topology():
    color, occ, lines, visible = _two_region_scene()
    res = fill_color(color, occ, lines, visible, noise_sigma=2.0)
    top = occ.copy()
    top[32:] = False
    bot = occ.copy()
    bot[:33] = False
    ft = res.filled[top].astype(np.float64).mean(axis=0)
    fb = res.filled[bot].astype(np.float64).mean(axis=0)
    assert ft[1] > ft[2] + 30  # 위쪽 채움은 녹색 우세
    assert fb[2] > fb[1] + 30  # 아래쪽 채움은 적색 우세


def test_fill_preserves_visible_and_invariants():
    color, occ, lines, visible = _two_region_scene()
    res = fill_color(color, occ, lines, visible, noise_sigma=2.0)
    assert np.array_equal(res.filled[~occ], color[~occ])
    assert res.filled.dtype == np.uint8
    assert res.patch_size % 2 == 1 and res.patch_size >= 3
    assert res.levels >= 1
    assert len(res.pieces) == 2
    # by construction: 채움 픽셀은 기여 exemplar [min,max]±½단위 안 (스펙 §5.1)
    assert res.by_construction_violations == 0
