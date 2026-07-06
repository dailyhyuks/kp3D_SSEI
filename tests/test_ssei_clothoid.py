"""clothoid.py 테스트: G2 경계 조건, 굽힘 에너지, 안전망 강등."""
import numpy as np

import kp3d.modules.ssei_v2.clothoid as cl
from kp3d.modules.ssei_v2.clothoid import connect_g2


def test_straight_line_zero_energy_and_clothoid():
    p0, p1 = np.array([10.0, 10.0]), np.array([10.0, 40.0])
    t = np.array([0.0, 1.0])
    c = connect_g2(p0, t, 0.0, p1, t, 0.0)
    assert c.is_clothoid
    assert c.bending_energy < 1e-8
    assert np.linalg.norm(c.points[0] - p0) < 0.5
    assert np.linalg.norm(c.points[-1] - p1) < 0.5
    assert abs(c.arc_length - 30.0) < 0.5


def test_g2_boundary_conditions():
    p0, p1 = np.array([20.0, 10.0]), np.array([20.0, 40.0])
    t0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    t1 = np.array([0.0, 1.0])
    k0, k1 = 0.05, -0.03
    c = connect_g2(p0, t0, k0, p1, t1, k1)
    assert float(np.dot(c.tangents[0], t0)) > 0.999
    assert float(np.dot(c.tangents[-1], t1)) > 0.999
    assert abs(float(c.curvatures[0]) - k0) < 1e-6
    assert abs(float(c.curvatures[-1]) - k1) < 1e-6
    assert np.linalg.norm(c.points[0] - p0) < 0.5
    assert np.linalg.norm(c.points[-1] - p1) < 0.5


def test_solver_failure_falls_back_to_quintic(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("강제 실패")

    monkeypatch.setattr(cl, "least_squares", _boom)
    p0, p1 = np.array([20.0, 10.0]), np.array([20.0, 40.0])
    t0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    t1 = np.array([0.0, 1.0])
    c = connect_g2(p0, t0, 0.05, p1, t1, -0.03)
    assert not c.is_clothoid
    # 안전망(quintic 폐형)도 G2 경계 조건은 해석적으로 정확히 유지
    assert float(np.dot(c.tangents[0], t0)) > 0.999
    assert float(np.dot(c.tangents[-1], t1)) > 0.999
    assert abs(float(c.curvatures[0]) - 0.05) < 1e-6
    assert abs(float(c.curvatures[-1]) + 0.03) < 1e-6


def test_degenerate_short_gap():
    p0, p1 = np.array([5.0, 5.0]), np.array([5.0, 5.5])
    t = np.array([0.0, 1.0])
    c = connect_g2(p0, t, 0.0, p1, t, 0.0)
    assert c.points.shape[0] >= 2
    assert np.allclose(c.points[0], p0) and np.allclose(c.points[-1], p1)
    assert c.bending_energy == 0.0
