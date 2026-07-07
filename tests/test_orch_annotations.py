"""annotations.py 테스트: 부품 병합, depth 규약, 가시성 해소."""
import numpy as np

from kp3d.modules.orchestration.annotations import (
    ObjectAnnotation, load_annotations, resolve_visibility)


def _sq(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_load_merges_parts_and_orders_by_depth():
    shapes = [
        {"label": "object_1", "shape_type": "polygon", "points": _sq(0, 0, 9, 9)},
        {"label": "object_2_1", "shape_type": "polygon", "points": _sq(20, 0, 29, 9)},
        {"label": "object_2_2", "shape_type": "polygon", "points": _sq(30, 0, 39, 9)},
        {"label": "background", "shape_type": "polygon", "points": _sq(0, 20, 39, 39)},
    ]
    annos = load_annotations(shapes, (40, 40))
    # v1 fallback 서열: object_2(0) < object_1(2) < background(3)
    assert [a.label for a in annos] == ["object_2", "object_1", "background"]
    o2 = annos[0]
    assert o2.mask[5, 25] and o2.mask[5, 35] and not o2.mask[5, 15]


def test_layer_order_field_overrides_fallback():
    shapes = [
        {"label": "object_1", "shape_type": "polygon",
         "points": _sq(0, 0, 9, 9), "layer_order": 1},
        {"label": "object_2_1", "shape_type": "polygon",
         "points": _sq(20, 0, 29, 9), "layer_order": 5},
    ]
    annos = load_annotations(shapes, (40, 40))
    assert [a.label for a in annos] == ["object_1", "object_2"]
    assert [a.depth for a in annos] == [1, 5]


def test_non_polygon_and_degenerate_shapes_skipped():
    shapes = [
        {"label": "object_1", "shape_type": "rectangle", "points": _sq(0, 0, 9, 9)},
        {"label": "object_1", "shape_type": "polygon", "points": [[0, 0], [5, 5]]},
    ]
    assert load_annotations(shapes, (20, 20)) == []


def test_resolve_visibility_removes_nearer_pixels():
    front = ObjectAnnotation("f", np.zeros((10, 10), dtype=bool), 0)
    rear = ObjectAnnotation("r", np.zeros((10, 10), dtype=bool), 1)
    front.mask[2:6, 2:6] = True
    rear.mask[4:9, 4:9] = True
    vis = resolve_visibility([front, rear])
    assert (vis[0] == front.mask).all()            # 전경은 그대로
    assert not (vis[1] & front.mask).any()         # 후방에서 전경 픽셀 제거
    assert vis[1].sum() == rear.mask.sum() - (rear.mask & front.mask).sum()
