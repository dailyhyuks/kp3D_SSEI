"""matching.py 테스트: 후보 생성, 순위 비용, Hungarian 종결, 상호 채택."""
import numpy as np

from kp3d.modules.ssei_v2.endpoints import detect_break_endpoints
from kp3d.modules.ssei_v2.matching import match_endpoints


def _scene(strokes, occ_box, h=80, w=64, width=3.0, ink=0.8):
    """strokes: (y, x0, x1) 수평 획 목록. occ_box: (y0, y1, x0, x1) 가림 상자."""
    skeleton = np.zeros((h, w), dtype=bool)
    for y, x0, x1 in strokes:
        skeleton[y, x0:x1] = True
    occlusion = np.zeros((h, w), dtype=bool)
    y0, y1, x0, x1 = occ_box
    occlusion[y0:y1, x0:x1] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, width, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, ink, 0.0).astype(np.float32)
    return skeleton, width_map, line_alpha, occlusion


def _match(skeleton, width_map, line_alpha, occlusion):
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    return eps, match_endpoints(eps, skeleton, width_map, line_alpha, occlusion)


def test_no_endpoints_empty():
    args = _scene([], (26, 41, 24, 40))
    eps, res = _match(*args)
    assert eps == [] and res.connections == [] and res.terminations == []


def test_single_endpoint_terminates():
    args = _scene([(32, 4, 40)], (26, 41, 24, 41))
    eps, res = _match(*args)
    assert len(eps) == 1
    assert res.connections == [] and res.terminations == [0]


def test_straight_gap_connects():
    args = _scene([(32, 4, 60)], (26, 41, 24, 41))
    eps, res = _match(*args)
    assert len(eps) == 2
    assert len(res.connections) == 1 and res.terminations == []
    c = res.connections[0]
    assert float(np.linalg.norm(c.curve.points[0] - eps[c.i].pos)) < 0.5
    assert float(np.linalg.norm(c.curve.points[-1] - eps[c.j].pos)) < 0.5
    assert c.widths == (eps[c.i].width, eps[c.j].width)


def test_prefers_geometric_continuation():
    # 왼쪽 획(y=32) ↔ 정렬된 오른쪽 획(y=32) 연결, 어긋난 획(y=52)은 종결
    args = _scene([(32, 4, 60), (52, 41, 60)], (26, 56, 24, 41))
    eps, res = _match(*args)
    assert len(eps) == 3
    assert len(res.connections) == 1
    c = res.connections[0]
    ys = {float(eps[c.i].pos[0]), float(eps[c.j].pos[0])}
    assert ys == {32.0}
    assert len(res.terminations) == 1
    assert float(eps[res.terminations[0]].pos[0]) == 52.0
