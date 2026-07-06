"""render.py 테스트: 곡선 스탬프, 성분 병합, 무가림 항등."""
import numpy as np
from scipy.ndimage import label

from kp3d.modules.ssei_v2.render import complete_lines

_N8 = np.ones((3, 3), dtype=bool)


def _gap_scene(h=64, w=64, y=32, gap=(24, 41), width=3.0, ink=0.8):
    skeleton = np.zeros((h, w), dtype=bool)
    skeleton[y, 4:w - 4] = True
    occlusion = np.zeros((h, w), dtype=bool)
    occlusion[y - 6:y + 7, gap[0]:gap[1]] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, width, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, ink, 0.0).astype(np.float32)
    return skeleton, width_map, line_alpha, occlusion


def test_connection_merges_components():
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    assert label(skeleton, structure=_N8)[1] == 2
    res = complete_lines(line_alpha, skeleton, width_map, occlusion)
    assert len(res.connections) == 1
    assert label(res.skeleton, structure=_N8)[1] == 1
    # 간격 중앙이 잉크·폭으로 채워졌다
    assert float(res.line_alpha[32, 32]) > 0.5
    assert abs(float(res.width_map[32, 32]) - 3.0) < 1.0


def test_no_occlusion_identity():
    skeleton, width_map, line_alpha, _ = _gap_scene()
    occlusion = np.zeros_like(skeleton)
    res = complete_lines(line_alpha, skeleton, width_map, occlusion)
    assert res.connections == [] and res.terminations == []
    assert np.array_equal(res.skeleton, skeleton)
    assert np.array_equal(res.line_alpha, line_alpha)
    assert np.array_equal(res.width_map, width_map)


def test_inputs_not_mutated():
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    sk0, wm0, la0 = skeleton.copy(), width_map.copy(), line_alpha.copy()
    complete_lines(line_alpha, skeleton, width_map, occlusion)
    assert np.array_equal(skeleton, sk0)
    assert np.array_equal(width_map, wm0)
    assert np.array_equal(line_alpha, la0)
