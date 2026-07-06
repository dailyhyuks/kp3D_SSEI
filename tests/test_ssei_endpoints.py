"""endpoints.py 테스트: 끊김 endpoint 검출, 접선/곡률 서술자, 획 통계."""
import numpy as np

from kp3d.modules.ssei_v2.endpoints import (
    detect_break_endpoints,
    stroke_statistics,
    trace_stroke,
)


def _gap_scene(h=64, w=64, y=32, gap=(24, 40), width=3.0, ink=0.8):
    """수평 획(스켈레톤 y행) 가운데 gap 열 구간을 가림으로 지운 장면."""
    skeleton = np.zeros((h, w), dtype=bool)
    skeleton[y, 4:w - 4] = True
    occlusion = np.zeros((h, w), dtype=bool)
    occlusion[y - 6:y + 7, gap[0]:gap[1]] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, width, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, ink, 0.0).astype(np.float32)
    return skeleton, width_map, line_alpha, occlusion


def test_trace_stroke_orders_from_endpoint():
    skeleton, *_ = _gap_scene()
    pts = trace_stroke(skeleton, (32, 23))
    assert pts.shape[1] == 2
    assert tuple(pts[0]) == (32, 23)
    # 왼쪽 조각을 따라 x가 단조 감소
    assert np.all(np.diff(pts[:, 1]) == -1)


def test_detects_two_facing_endpoints():
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    assert len(eps) == 2
    xs = sorted(float(e.pos[1]) for e in eps)
    assert xs[0] == 23.0 and xs[1] == 40.0
    left = min(eps, key=lambda e: e.pos[1])
    right = max(eps, key=lambda e: e.pos[1])
    # 바깥 접선이 서로 마주본다
    chord = (right.pos - left.pos) / np.linalg.norm(right.pos - left.pos)
    assert float(np.dot(left.tangent, chord)) > 0.9
    assert float(np.dot(right.tangent, -chord)) > 0.9
    # 직선이므로 곡률 ~ 0, 서술자 값 전달
    assert abs(left.curvature) < 0.05
    assert abs(left.width - 3.0) < 0.5
    assert abs(left.ink - 0.8) < 1e-6
    assert left.stroke_id != right.stroke_id


def test_far_endpoint_not_reported():
    """가림에서 먼 자연 끝점(획의 양 끝)은 검출 제외."""
    skeleton, width_map, line_alpha, occlusion = _gap_scene()
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    xs = {int(e.pos[1]) for e in eps}
    assert 4 not in xs and 59 not in xs


def test_curvature_sign_on_arc():
    """반지름 R 원호의 |κ| ≈ 1/R."""
    h = w = 96
    skeleton = np.zeros((h, w), dtype=bool)
    R, cy, cx = 30.0, 48, 48
    ang = np.linspace(np.pi * 0.1, np.pi * 0.9, 200)
    ys = np.round(cy - R * np.sin(ang)).astype(int)
    xs = np.round(cx + R * np.cos(ang)).astype(int)
    skeleton[ys, xs] = True
    occlusion = np.zeros((h, w), dtype=bool)
    occlusion[:, 44:52] = True
    skeleton[occlusion] = False
    width_map = np.where(skeleton, 2.0, 0.0).astype(np.float32)
    line_alpha = np.where(skeleton, 1.0, 0.0).astype(np.float32)
    eps = detect_break_endpoints(skeleton, width_map, line_alpha, occlusion)
    assert len(eps) >= 2
    for e in eps:
        assert abs(abs(e.curvature) - 1.0 / R) < 0.5 / R


def test_stroke_statistics_nonempty_and_positive():
    skeleton, width_map, line_alpha, _ = _gap_scene()
    k2, dw, di = stroke_statistics(skeleton, width_map, line_alpha)
    assert k2.size > 0 and np.all(k2 >= 0.0)
    assert np.all(dw >= 0.0) and np.all(di >= 0.0)
