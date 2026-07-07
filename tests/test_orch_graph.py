"""graph.py 테스트: 반경 유도, 간선 방향, 후보 영역 기하."""
import cv2
import numpy as np

from kp3d.modules.orchestration.annotations import ObjectAnnotation
from kp3d.modules.orchestration.graph import (
    build_occlusion_graph, derive_dilation_radius)


def test_radius_from_width_p95_and_fallback():
    sk = np.zeros((50, 50), dtype=bool)
    wm = np.zeros((50, 50))
    sk[10, 10:20] = True
    wm[10, 10:20] = 4.0
    assert derive_dilation_radius(sk, wm, (50, 50)) == 4
    empty = np.zeros((200, 200), dtype=bool)
    r = derive_dilation_radius(empty, np.zeros((200, 200)), (200, 200))
    assert r == max(1, round(0.005 * float(np.hypot(200, 200))))


def test_edges_front_to_rear_only_and_region_geometry():
    h = w = 60
    front = ObjectAnnotation("object_2", np.zeros((h, w), dtype=bool), 0)
    rear = ObjectAnnotation("object_1", np.zeros((h, w), dtype=bool), 2)
    front.mask[20:40, 25:35] = True
    rear.mask[25:45, 10:50] = True          # 아모달로 겹치게 그린 어노테이션
    vis = [front.mask, rear.mask & ~front.mask]
    edges = build_occlusion_graph([front, rear], vis, radius=2)
    assert [(e.occluder, e.occludee) for e in edges] == [("object_2", "object_1")]
    e = edges[0]
    assert e.region.any()
    assert not (e.region & vis[1]).any()     # 후방 가시와 배타
    assert (e.region & front.mask).any()     # 전경 아래에 위치
    dil = cv2.dilate(vis[0].astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    assert not (e.region & ~dil.astype(bool)).any()  # 팽창 전경 안에 한정


def test_disjoint_objects_no_edge():
    a = ObjectAnnotation("object_2", np.zeros((40, 40), dtype=bool), 0)
    b = ObjectAnnotation("object_1", np.zeros((40, 40), dtype=bool), 2)
    a.mask[2:8, 2:8] = True
    b.mask[30:38, 30:38] = True
    assert build_occlusion_graph([a, b], [a.mask, b.mask], radius=1) == []
